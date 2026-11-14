#!/usr/bin/env python3
"""Prediction sets (BUILD_CONTRACTS.md section 2): a user's tiles, ready to be predicted by one registry model.

    python -m jobs.predict_set adopt --from <outside run> --model <model id> --images <dir> [--images <dir> ...]
        --name <text> [--owner <person>] [--nm-confirmed-by <person>] [--sets A,B] [--root <dir>] [--registry <dir>]
        [--do]                                  (round 14, contract C6: see adopt(); a dry run unless --do)
    python -m jobs.predict_set create --from-dir <dir> [--manifest <MANIFEST.csv>] [--only <list file>]
        --model <model id> --owner <person>
        (--sample-label <label> | --sample-label-from manifest:<column>)
        (--nm-per-px <x> | --nm-from-manifest) --nm-confirmed-by <person>
        [--conf 0.25] [--max-det 1000] [--imgsz N] [--session YYYYMMDD] [--name <text>] [--link] [--registry <dir>] [--root <dir>]
        [--note <text>] [--provenance <provenance.json>] [--dry-run]

Writes <root>/<ps id>/ (default root <CryoAI-laptop>/predict; the id ps_<YYYYMMDD>_<6 hex> is made here):
  images/<nnnn>.png|.jpg   the tiles, numbered in name order; the extension comes from the content. Copied (md5
                           checked); hard-linked only with --link, on the same drive and not from a protected folder.
                           The original file name is never in a path, only in images.csv orig_name
  images.csv               image,file,orig_name,sample_label,nm_per_px,nm_confirmed_by,source,md5 (raw values: it is
                           machine-read; labels and names a spreadsheet would run are refused at create)
  request.json             ps, owner, model, conf, max_det, imgsz, source, created, and `name` (only when one was
                           given: clean_name(), at most 80 characters; not in the registry; the portal renames it);
                           `note` and `provenance` (only when given; also in the registry record), and
                           `source_manifest` ("source_manifest.csv", only with --manifest)
  source_manifest.csv      only with --manifest: a byte-identical copy of it (md5 checked), so per-tile columns
                           (parent image, x/y offset, ...) stay with the set
  status.json              state queued, n_done 0, n_total, where laptop, job null, error null, updated, pid null,
                           waiting null
and appends a prediction_set/1 record (status queued) to the registry. The folder is built in <root>/.incoming/ and
renamed into place, so a reader never sees half a set.

How the images were made (round 11b), for a set made in an unusual way (tiling, resampling, a subset):
  --note TEXT          free text, at most 500 characters and 10 lines; no control or format characters but newline;
                       not starting like a spreadsheet formula (= + - @). Stored as `note`.
  --provenance FILE    a JSON object of at most 16 KB with only these keys (anything else is refused):
                         derived_from  [{"kind": dataset|label_version|session|prediction_set|model|other,
                                         "id": str <= 80}]  (at most 10)
                         source_images str <= 300
                         steps         [str <= 200]  (at most 15)
                         tiling        {"tile_px": int, "overlap_px": int, "overlap_frac": number in [0, 1),
                                        "nm_per_px": number, "downsample": number}  (any, at least one)
                         script        str <= 200
                         git           hex str <= 40
                       Stored as `provenance`. Both are data fields of the prediction_set record, so update_record
                       carries them to every later status line. The Data page shows them (Note, How the images
                       were made, Derived from).

--dry-run (create(dry_run=True)): every check below runs (the tiles are read, md5'd and decoded) and the registry
record is validated with a placeholder id, then the plan is printed; nothing is written, no lock file either. The id
itself is chosen only by a real create. `python -m cryoev predict new` uses it as its dry run.

Refused, with every problem listed and nothing written: no images, or more than 1000; a file that is not an 8-bit
PNG or JPEG by its content (the extension is not trusted; MANIFEST.csv, the --manifest file and OS files such as
desktop.ini are skipped), or one with a side over 8192 px (checked before it is decoded); nm/px missing for any
image; no --nm-confirmed-by; a sample label or file name starting with = + - @ (a spreadsheet would run it); mixed tile sizes (or nm/px spread over 1 % of the median) within one sample label; an unknown model; --only
names that are not plain file names. The manifest's name column is output_name (or image, file_name, filename,
name), its nm/px column nm_per_px.

Python API (the portal): create(...) -> the set's folder. It reads, md5s and decodes every tile (seconds per 100 tiles; minutes for 1000):
call it from a worker thread, never on the event loop. find(ps) -> folder; read_status / write_status (atomic,
read-modify-write under a lock); update_record(ps_id, **fields) appends the registry line after a status change.
Imported as jobs.predict_set it changes no sys.path and uses the registry package (`from registry import store`,
the one registry module of the process), so no portal module is shadowed.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import math
import os
import re
import secrets
import shutil
import statistics
import sys
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if __package__:                                  # jobs.predict_set (the portal, python -m): sys.path untouched
    from . import common as C
else:                                            # a script, or the jobs/ modules that import it by name
    for _p in (HERE, REPO):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import common as C  # noqa: E402
from data_utils import config  # noqa: E402
from data_utils.oslock import acquire_lock  # noqa: E402
from data_utils.path_guard import protected_problem  # noqa: E402

MAX_IMAGES = 1000
NAME_MAX = 80                                    # an analysis's optional display name (request.json `name`)
MAX_SIDE = 8192                                  # px, per side, read from the header before any decode
PREDICT_DIR = "predict"                          # <CryoAI-laptop>/predict/<ps id>/
INCOMING, LOCKS = ".incoming", ".locks"
IMAGES, IMAGES_CSV, REQUEST, STATUS = "images", "images.csv", "request.json", "status.json"
CSV_COLUMNS = ["image", "file", "orig_name", "sample_label", "nm_per_px", "nm_confirmed_by", "source", "md5"]
STATUS_KEYS = ("state", "n_done", "n_total", "where", "job", "error", "updated", "pid", "waiting")
NAME_COLUMNS = ("output_name", "image", "file_name", "filename", "name")   # as inference/inference_check.py
NM_COLUMN = "nm_per_px"
NM_TOLERANCE = 0.01                              # one sample's nm/px spread / median above this = refused; pinned
                                                 # equal to results.compute's and the portal's (portal tests)
NM_EPS = 1e-12                                   # float slack on that comparison: exactly 1 % is accepted (pinned)
SKIP_NAMES = {"manifest.csv", "desktop.ini", "thumbs.db", ".ds_store"}
PS_RE = re.compile(r"^ps_\d{8}_[0-9a-f]{6}$")
PERSON_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")       # registry/schemas/prediction_set.json owner
SESSION_RE = re.compile(r"^\d{8}(-\d+)?$")
PNG_SIG, JPEG_SIG = b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff"
PNG_MODES, JPEG_MODES = {"L", "LA", "RGB", "RGBA", "P"}, {"L", "RGB"}
FORMULA_START = ("=", "+", "-", "@")
MODELS_YAML = REPO / "models" / "models.yaml"
LAPTOP_PREFIX = "<CryoAI-laptop>"
REGISTRY_LOCK = ".prediction_set.lock"           # in the registry folder: id choice + read-modify-append
LOCK_WAIT = 30.0
SHOW = 12                                        # problems listed per refusal
SOURCE_MANIFEST = "source_manifest.csv"          # a copy of --manifest in the set folder
NOTE_MAX, NOTE_LINES = 500, 10                   # --note: characters, lines
PROV_MAX_BYTES = 16 * 1024                       # --provenance file size
DERIVED_KINDS = ("dataset", "label_version", "session", "prediction_set", "model", "other")
DERIVED_MAX, ID_MAX, STEPS_MAX, STEP_MAX, SOURCE_IMAGES_MAX, SCRIPT_MAX, GIT_MAX = 10, 80, 15, 200, 300, 200, 40
TILING_KEYS = {"tile_px": "int", "overlap_px": "int", "overlap_frac": "frac", "nm_per_px": "pos",
               "downsample": "pos"}
GIT_RE = re.compile(r"^[0-9a-fA-F]{1,40}\Z")


class PredictSetError(ValueError):
    """A refusal or a broken set; the message lists every problem found."""


# ---------------------------------------------------------------- registry, locks, atomic writes
_REG_LOCK = threading.Lock()
_THREAD_LOCKS: dict = {}
_THREAD_LOCKS_LOCK = threading.Lock()


def registry_module():
    """registry.store (the one registry module object of the process)."""
    from registry import store
    return store


@contextlib.contextmanager
def locked(path: Path, wait=LOCK_WAIT):
    """Exclusive across threads (a lock per path) and processes (an OS lock on `path`)."""
    path = Path(path)
    key = os.path.normcase(str(path.resolve()))
    with _THREAD_LOCKS_LOCK:
        tl = _THREAD_LOCKS.setdefault(key, threading.Lock())
    with tl:
        path.parent.mkdir(parents=True, exist_ok=True)
        f = acquire_lock(path, wait)
        if f is None:
            raise PredictSetError(f"busy: another process held {path} for {wait:g} s")
        try:
            yield
        finally:
            f.close()


def write_json_atomic(path, obj):
    """A tmp file unique to this process and thread, then os.replace (retried: a reader may hold the target)."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    for i in range(6):
        try:
            return os.replace(tmp, path)
        except PermissionError:
            if i == 5:
                raise
            time.sleep(0.1 * 2 ** i)


def md5_file(path) -> str:
    return C.md5_file(path)


# ---------------------------------------------------------------- where sets live
def default_root() -> Path:
    return config.laptop_root() / PREDICT_DIR


def record_path(ps_dir: Path) -> str:
    """The registry `path`: <CryoAI-laptop>/predict/<id> when under the laptop root, else the absolute path."""
    lap = config.laptop_root(required=False)
    try:
        if lap is not None:
            return f"{LAPTOP_PREFIX}/" + Path(ps_dir).resolve().relative_to(Path(lap).resolve()).as_posix()
    except ValueError:
        pass
    return Path(ps_dir).resolve().as_posix()


def resolve_record_path(path: str) -> Path | None:
    """Inverse of record_path."""
    if not path:
        return None
    if path.startswith(LAPTOP_PREFIX):
        lap = config.laptop_root(required=False)
        return Path(lap) / path[len(LAPTOP_PREFIX):].lstrip("/\\") if lap else None
    return Path(path)


def find(ps, root=None, registry_root=None) -> Path:
    """A set's folder from its id or its folder: <root>/<id>, else the registry record's path, else the default
    root. Raises PredictSetError."""
    p = Path(str(ps))
    if (p / REQUEST).is_file():
        return p
    ps = str(ps)
    if not PS_RE.match(ps):
        raise PredictSetError(f"{ps!r} is neither a prediction set folder nor an id ps_<YYYYMMDD>_<6 hex>")
    cands = []
    if root:
        cands.append(Path(root) / ps)
    else:
        try:
            rec = registry_module().current("prediction_set", registry_root).get(ps)
        except (config.ConfigError, OSError):
            rec = None
        got = resolve_record_path((rec or {}).get("path"))
        if got is not None:
            cands.append(got)
        cands.append(default_root() / ps)
    for c in cands:
        if (c / REQUEST).is_file():
            return c
    raise PredictSetError(f"prediction set {ps} not found (looked in {', '.join(str(c) for c in cands)})")


# ---------------------------------------------------------------- reading a set
def read_request(ps_dir) -> dict:
    return C.read_json(Path(ps_dir) / REQUEST)


def read_status(ps_dir) -> dict:
    return C.read_json(Path(ps_dir) / STATUS)


def status_lock_path(ps_dir) -> Path:
    """<root>/.locks/<ps>.status.lock: outside the set folder, which keeps the contract layout."""
    ps_dir = Path(ps_dir)
    return ps_dir.parent / LOCKS / f"{ps_dir.name}.status.lock"


def update_status(ps_dir, change) -> dict | None:
    """Conditional read-modify-write under the set's status lock: change(current status) -> fields to merge, or
    None to leave the file alone. -> the new status, or None when change declined (e.g. another run owns the set)."""
    p = Path(ps_dir) / STATUS
    with locked(status_lock_path(ps_dir)):
        st = C.read_json(p) if p.is_file() else {}
        fields = change(dict(st))
        if fields is None:
            return None
        st.update(fields, updated=C.now())
        out = {k: st.get(k) for k in STATUS_KEYS}
        write_json_atomic(p, out)
    return out


def write_status(ps_dir, **fields) -> dict:
    """Merge fields into status.json, `updated` = now: read-modify-write under the set's status lock, written
    atomically. Keys: state n_done n_total where job error updated pid waiting. -> the new status.
    pid: while running, the predictor's; while queued with `waiting` set, the predict_local process waiting for it."""
    return update_status(ps_dir, lambda st: fields)


def read_images(ps_dir) -> list:
    with open(Path(ps_dir) / IMAGES_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def update_record(ps_id, registry_root=None, recorded_by="jobs/predict_set.py", **fields):
    """Append the set's registry line with `fields` changed (status, where, job, ...): read and append under the
    registry's prediction_set lock. -> the stored line. Raises PredictSetError when the set has no record.
    Every data field of the current line carries forward (owner_via: the agent the set was made by stays on every
    later status line); the envelope `via` does not (registry.append sets it for the line's own writer)."""
    reg = registry_module()
    root = Path(registry_root) if registry_root else config.registry_root()
    with locked(root / REGISTRY_LOCK):
        cur = reg.current("prediction_set", root).get(ps_id)
        if cur is None:
            raise PredictSetError(f"{ps_id} has no prediction_set record in the registry")
        rec = {**cur, **fields, "logged": C.now(), "recorded_by": recorded_by}
        return reg.append(rec, root)


# ---------------------------------------------------------------- checks
def inspect_image(path: Path) -> tuple:
    """-> (ext, (w, h)) for an 8-bit PNG or JPEG, judged by content; each side <= MAX_SIDE before any decode.
    Raises ValueError with the reason."""
    with open(path, "rb") as f:
        head = f.read(32)
    if head.startswith(PNG_SIG):
        if head[12:16] != b"IHDR":
            raise ValueError("PNG without an IHDR header")
        depth = head[24]
        if depth != 8:
            raise ValueError(f"{depth}-bit PNG (need 8-bit)")
        ext, modes = ".png", PNG_MODES
    elif head.startswith(JPEG_SIG):
        ext, modes = ".jpg", JPEG_MODES
    else:
        raise ValueError("not a PNG or JPEG (by content)")
    from PIL import Image
    try:
        with Image.open(path) as im:                   # reads the header only
            mode, size = im.mode, im.size
            if max(size) > MAX_SIDE:
                raise ValueError(f"{size[0]}x{size[1]} px: a side over {MAX_SIDE} px")
            im.load()                                  # a truncated or corrupt file fails here, not on the GPU
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"unreadable {ext[1:].upper()}: {type(e).__name__}: {e}")
    if mode not in modes:
        raise ValueError(f"{ext[1:].upper()} mode {mode} (need 8-bit {', '.join(sorted(modes))})")
    return ext, size


def nm_one_scale(values) -> bool:
    """Contract section 8: one sample's nm/px values (one per image) are one scale when (max - min) / median <=
    NM_TOLERANCE (+ NM_EPS float slack). = results.compute.one_scale and the portal's predict.one_scale."""
    vals = [float(v) for v in values]
    return (max(vals) - min(vals)) / statistics.median(vals) <= NM_TOLERANCE + NM_EPS


def read_manifest(path: Path) -> dict:
    """{file name: row} from a tile MANIFEST.csv."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    key = next((c for c in NAME_COLUMNS if rows and c in rows[0]), None)
    if key is None:
        raise PredictSetError(f"{path.name} has none of the name columns {NAME_COLUMNS}")
    return {r[key]: r for r in rows}


def model_spec(model: str) -> dict:
    import yaml
    reg = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))["models"]
    if model not in reg:
        raise PredictSetError(f"unknown model {model!r}; models/models.yaml has {', '.join(reg)}")
    if not (REPO / reg[model]["weights"]).is_file():
        raise PredictSetError(f"model {model}: weights {reg[model]['weights']} not found in the repo")
    return reg[model]


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def _formula_like(s) -> bool:
    """Starts like a spreadsheet formula (= + - @): refused at create, so images.csv can stay raw."""
    return str(s).startswith(FORMULA_START)


def _list(problems):
    more = f"\n  ... and {len(problems) - SHOW} more" if len(problems) > SHOW else ""
    return "\n  " + "\n  ".join(problems[:SHOW]) + more


def _scan(from_dir: Path, manifest: Path | None, names) -> list:
    """The candidate files, sorted by name (names: only these plain file names)."""
    if names is not None:
        bad = [n for n in names if not n or Path(n).name != n or n in (".", "..")]
        if bad:
            raise PredictSetError("refused: --only takes plain file names in --from-dir, not paths:" + _list(bad))
        missing = [n for n in names if not (from_dir / n).is_file()]
        if missing:
            raise PredictSetError(f"{len(missing)} listed file(s) not in {from_dir}:" + _list(missing))
        return sorted({from_dir / n for n in names}, key=lambda p: p.name)
    skip = {manifest.resolve()} if manifest else set()
    return sorted((p for p in from_dir.iterdir() if p.is_file() and p.name.lower() not in SKIP_NAMES
                   and not p.name.startswith(".") and p.resolve() not in skip), key=lambda p: p.name)


def clean_name(s) -> str | None:
    """An analysis name for display: NFC; tab, newline, line and paragraph separators become a space; control (Cc)
    and format (Cf: bidi controls, zero-width, soft hyphen, word joiners, tag characters) characters are dropped;
    runs of whitespace collapse to one space; trimmed, at most NAME_MAX characters, and a combining mark left
    dangling by the cut is removed. Nothing left -> None. Not an escape: pages still escape it."""
    if not isinstance(s, str):
        return None
    s = unicodedata.normalize("NFC", s)
    s = "".join(" " if ch in "\t\n\r\u2028\u2029" else ch for ch in s)
    s = "".join(ch for ch in s if unicodedata.category(ch) not in ("Cc", "Cf"))
    s = " ".join(s.split())[:NAME_MAX]
    while s and unicodedata.category(s[-1]).startswith("M"):
        s = s[:-1]
    return s.strip() or None


def _bad_chars(s, allow="") -> list:
    """The control (Cc), format (Cf) and line/paragraph separator characters in s, except those in `allow`."""
    return sorted({f"U+{ord(ch):04X}" for ch in s if ch not in allow
                   and (unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp"))})


def check_note(note) -> tuple:
    """--note -> (the note to store or None, [problems]). NFC, CRLF/CR made LF, trimmed; at most NOTE_MAX characters
    and NOTE_LINES lines; no control or format character but newline; not starting like a spreadsheet formula."""
    if note is None:
        return None, []
    if not isinstance(note, str):
        return None, [f"note must be text, not {type(note).__name__}"]
    s = unicodedata.normalize("NFC", note).replace("\r\n", "\n").strip()
    if not s:
        return None, ["note is empty (leave --note out instead)"]
    problems = []
    bad = _bad_chars(s, allow="\n")
    if bad:
        problems.append(f"note has control or format characters ({', '.join(bad[:6])}); only newline is allowed")
    if len(s) > NOTE_MAX:
        problems.append(f"note is {len(s)} characters; at most {NOTE_MAX}")
    if s.count("\n") + 1 > NOTE_LINES:
        problems.append(f"note has {s.count(chr(10)) + 1} lines; at most {NOTE_LINES}")
    if _formula_like(s):
        problems.append(f"note starts with {s[0]!r}: a spreadsheet would run it as a formula; start it with a "
                        f"letter or digit")
    return (None if problems else s), problems


def _text_problem(where, v, limit, empty_ok=False):
    if not isinstance(v, str):
        return f"{where} must be a string, not {type(v).__name__}"
    if not v.strip() and not empty_ok:
        return f"{where} is empty"
    if len(v) > limit:
        return f"{where} is {len(v)} characters; at most {limit}"
    bad = _bad_chars(v)
    if bad:
        return f"{where} has control or format characters ({', '.join(bad[:6])})"
    return None


def _number_problem(where, v, kind):
    """kind int: a whole number >= 0 (tile_px >= 1); frac: 0 <= x < 1; pos: a finite number > 0. No booleans."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return f"{where} must be a number, not {json.dumps(v)}"
    if kind == "int":
        if not isinstance(v, int):
            return f"{where} must be a whole number, not {v!r}"
        low = 1 if where.endswith("tile_px") else 0
        return None if v >= low else f"{where} must be >= {low}"
    if kind == "frac":
        return None if 0 <= v < 1 else f"{where} must be from 0 up to (not including) 1"
    return None if v > 0 else f"{where} must be > 0"


def check_provenance(obj) -> list:
    """[problems] for a provenance object (see the module doc); [] when valid. Unknown keys and wrong types are
    problems, never ignored."""
    if not isinstance(obj, dict):
        return [f"provenance must be a JSON object, not {type(obj).__name__}"]
    if not obj:
        return ["provenance is empty (leave --provenance out instead)"]
    allowed = ("derived_from", "source_images", "steps", "tiling", "script", "git")
    problems = [f"provenance: unknown key {k!r} (allowed: {', '.join(allowed)})" for k in obj if k not in allowed]
    if "derived_from" in obj:
        d = obj["derived_from"]
        if not isinstance(d, list):
            problems.append("provenance.derived_from must be a list of {\"kind\", \"id\"}")
        else:
            if len(d) > DERIVED_MAX:
                problems.append(f"provenance.derived_from has {len(d)} entries; at most {DERIVED_MAX}")
            for n, e in enumerate(d):
                w = f"provenance.derived_from[{n}]"
                if not isinstance(e, dict):
                    problems.append(f"{w} must be an object {{\"kind\", \"id\"}}")
                    continue
                problems += [f"{w}: unknown key {k!r} (allowed: kind, id)" for k in e if k not in ("kind", "id")]
                if e.get("kind") not in DERIVED_KINDS:
                    problems.append(f"{w}.kind {json.dumps(e.get('kind'))} must be one of {', '.join(DERIVED_KINDS)}")
                problems.append(_text_problem(f"{w}.id", e.get("id"), ID_MAX))
    if "source_images" in obj:
        problems.append(_text_problem("provenance.source_images", obj["source_images"], SOURCE_IMAGES_MAX))
    if "steps" in obj:
        st = obj["steps"]
        if not isinstance(st, list):
            problems.append("provenance.steps must be a list of strings")
        else:
            if len(st) > STEPS_MAX:
                problems.append(f"provenance.steps has {len(st)} entries; at most {STEPS_MAX}")
            problems += [_text_problem(f"provenance.steps[{n}]", v, STEP_MAX) for n, v in enumerate(st)]
    if "tiling" in obj:
        t = obj["tiling"]
        if not isinstance(t, dict) or not t:
            problems.append(f"provenance.tiling must be an object with at least one of {', '.join(TILING_KEYS)}")
        else:
            for k, v in t.items():
                if k not in TILING_KEYS:
                    problems.append(f"provenance.tiling: unknown key {k!r} (allowed: {', '.join(TILING_KEYS)})")
                else:
                    problems.append(_number_problem(f"provenance.tiling.{k}", v, TILING_KEYS[k]))
    if "script" in obj:
        problems.append(_text_problem("provenance.script", obj["script"], SCRIPT_MAX))
    if "git" in obj:
        g = obj["git"]
        if not isinstance(g, str) or not GIT_RE.match(g):
            problems.append(f"provenance.git {json.dumps(g)} must be a hex commit id of 1-{GIT_MAX} characters")
    return [p for p in problems if p]


def load_provenance(path) -> tuple:
    """--provenance FILE -> (the object, [problems]): at most PROV_MAX_BYTES, UTF-8 JSON (a BOM is allowed), no
    duplicate keys, no NaN/Infinity, then check_provenance. Nothing is read past the size cap."""
    path = Path(path)
    if not path.is_file():
        return None, [f"--provenance {path} not found"]
    size = path.stat().st_size
    if size > PROV_MAX_BYTES:
        return None, [f"--provenance {path.name} is {size} bytes; at most {PROV_MAX_BYTES}"]

    def pairs(kv):
        keys = [k for k, _ in kv]
        dup = sorted({k for k in keys if keys.count(k) > 1})
        if dup:
            raise ValueError(f"duplicate key(s) {', '.join(map(repr, dup))}")
        return dict(kv)

    def no_constant(c):
        raise ValueError(f"{c} is not a number JSON allows")
    try:
        obj = json.loads(path.read_bytes().decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=no_constant)
    except (UnicodeDecodeError, ValueError) as e:
        return None, [f"--provenance {path.name} is not valid UTF-8 JSON: {e}"]
    problems = check_provenance(obj)
    return (None if problems else obj), problems


def _place(src: Path, dst: Path, md5: str, link: bool) -> str:
    """Copy (md5 checked), or with link=True hard-link when on one drive and not from a protected folder."""
    if link and protected_problem(src) is None and os.stat(src).st_dev == os.stat(dst.parent).st_dev:
        try:
            os.link(src, dst)
            return "link"
        except OSError:
            pass
    shutil.copy2(src, dst)
    if md5_file(dst) != md5:
        raise PredictSetError(f"copy of {src.name} differs from the source (md5)")
    return "copy"


# ---------------------------------------------------------------- create
def create(from_dir, model, owner, nm_confirmed_by, sample_label=None, sample_label_column=None, nm_per_px=None,
           nm_from_manifest=False, manifest=None, names=None, conf=0.25, max_det=1000, imgsz=None, session=None,
           registry_root=None, root=None, link=False, where="laptop", name=None, progress=None, dry_run=False,
           via=None, note=None, provenance=None):
    """Make a prediction set (see the module doc). -> its folder. Raises PredictSetError (nothing written).
    Slow (md5 + decode of every tile): the portal calls it from a worker thread.
    via: the agent that asked for the set for `owner` (the portal passes a write token's agent; round 11 #65), or
    None. Stored as "owner_via" in request.json and in the registry record (a data field: update_record carries it
    to every later line), and as the create line's own `via`; also "nm_confirmed_via" (the nm/px confirmation of
    nm_confirmed_by came through that agent) in both. Not given: nothing is stored and the line's via is
    registry.append's usual ($CRYOEV_VIA, which the portal strips).
    name: an optional display name (clean_name) written to request.json only when set. progress: optional
    callback(done, total, phase) with phase "checking" (md5 + decode) then "placing" (copy or link); additive, the
    CLI passes none. dry_run: all checks, then -> the plan dict (nothing written; the id is a placeholder).
    note: free text (check_note). provenance: a provenance.json path, or the object itself (check_provenance). Both
    are refused before anything is written, and stored (when given) in request.json and the registry record. With
    a manifest, request.json also gets source_manifest and the set a byte-identical source_manifest.csv."""
    tick = progress or (lambda *a: None)
    set_name = clean_name(name)                 # `name` is reused below as a file name
    from_dir = Path(from_dir)
    root = Path(root) if root else default_root()
    reg = Path(registry_root) if registry_root else config.registry_root()
    manifest = Path(manifest) if manifest else None
    problems = []
    # ---- arguments
    if not from_dir.is_dir():
        raise PredictSetError(f"--from-dir {from_dir} is not a folder")
    if not owner or not PERSON_RE.match(str(owner)):
        problems.append(f"owner {owner!r} is not a person id (lowercase, e.g. pujan)")
    if not nm_confirmed_by:
        problems.append("--nm-confirmed-by is required: who confirmed the nm/px (no silent defaults)")
    elif not PERSON_RE.match(str(nm_confirmed_by)):
        problems.append(f"--nm-confirmed-by {nm_confirmed_by!r} is not a person id")
    if (nm_per_px is not None) == bool(nm_from_manifest):
        problems.append("give exactly one of --nm-per-px or --nm-from-manifest (nm/px is required per image)")
    elif nm_per_px is not None and _num(nm_per_px) is None:
        problems.append(f"--nm-per-px {nm_per_px!r} is not a positive number")
    if (sample_label is None) == (sample_label_column is None):
        problems.append("give exactly one of --sample-label or --sample-label-from manifest:<column>")
    elif sample_label is not None and (not str(sample_label).strip() or len(str(sample_label)) > 80
                                       or any(ord(ch) < 32 for ch in str(sample_label))):
        problems.append(f"sample label {sample_label!r}: 1-80 printable characters")
    elif sample_label is not None and _formula_like(str(sample_label).strip()):
        problems.append(f"sample label {sample_label!r} starts with {str(sample_label).strip()[0]!r}: a spreadsheet "
                        f"would run it as a formula; start it with a letter or digit (e.g. 'EDTA+' or 'with EDTA')")
    if (nm_from_manifest or sample_label_column) and manifest is None:
        problems.append("--nm-from-manifest / --sample-label-from manifest:... need --manifest")
    if manifest is not None and not manifest.is_file():
        problems.append(f"--manifest {manifest} not found")
    try:
        conf = float(conf)
        if not 0 < conf < 1:
            raise ValueError
    except (TypeError, ValueError):
        problems.append(f"conf {conf!r} must be a number between 0 and 1")
    try:
        if isinstance(max_det, bool) or int(max_det) != float(max_det) or int(max_det) < 1:
            raise ValueError
        max_det = int(max_det)
    except (TypeError, ValueError):
        problems.append(f"max_det {max_det!r} must be a whole number >= 1")
    if imgsz is not None:
        try:
            if isinstance(imgsz, bool) or int(imgsz) != float(imgsz) or not 32 <= int(imgsz) <= MAX_SIDE:
                raise ValueError
            imgsz = int(imgsz)
        except (TypeError, ValueError):
            problems.append(f"imgsz {imgsz!r} must be a whole number from 32 to {MAX_SIDE}")
    if session is not None and not SESSION_RE.match(str(session)):
        problems.append(f"session {session!r} is not YYYYMMDD[-N]")
    if where not in ("laptop", "desktop"):
        problems.append(f"where {where!r} must be laptop or desktop")
    if via is not None and not re.match(r"^[a-z][a-z0-9_-]{0,31}\Z", str(via)):
        problems.append(f"via {via!r} is not an agent name (lower-case, e.g. claude)")
    set_note, why = check_note(note)
    problems += why
    prov = None
    if provenance is not None:
        if isinstance(provenance, dict):
            why = check_provenance(provenance)
            prov = None if why else json.loads(json.dumps(provenance))     # a plain copy
        else:
            prov, why = load_provenance(provenance)
        problems += why
    try:
        model_spec(model)
    except PredictSetError as e:
        problems.append(str(e))
    guard = protected_problem(root)
    if guard:
        problems.append(f"--root refused: {guard}")
    if problems:
        raise PredictSetError(f"refused ({len(problems)} problem(s)):" + _list(problems))
    # ---- images
    files = _scan(from_dir, manifest, names)
    if not files:
        raise PredictSetError(f"refused: no images in {from_dir}")
    if len(files) > MAX_IMAGES:
        raise PredictSetError(f"refused: {len(files)} files in {from_dir}; a prediction set takes at most "
                              f"{MAX_IMAGES} images (pick a subset with --only)")
    man = read_manifest(manifest) if manifest is not None else {}
    label_col = None
    if sample_label_column is not None:
        label_col = str(sample_label_column)
        if label_col.startswith("manifest:"):
            label_col = label_col[len("manifest:"):]
        if man and label_col not in next(iter(man.values())):
            raise PredictSetError(f"refused: {manifest.name} has no column {label_col!r}")
    if nm_from_manifest and man and NM_COLUMN not in next(iter(man.values())):
        raise PredictSetError(f"refused: {manifest.name} has no {NM_COLUMN} column")
    items = []
    for k, p in enumerate(files, 1):
        tick(k - 1, len(files), "checking")
        try:
            ext, size = inspect_image(p)
        except (ValueError, OSError) as e:
            problems.append(f"{p.name}: {e}")
            continue
        row = man.get(p.name)
        nm = _num(nm_per_px) if nm_per_px is not None else _num((row or {}).get(NM_COLUMN))
        if nm is None:
            problems.append(f"{p.name}: nm/px missing" + ("" if row else " (no manifest row)") if nm_from_manifest
                            else f"{p.name}: nm/px missing")
        label = str(sample_label).strip() if sample_label is not None else str((row or {}).get(label_col) or "").strip()
        if not label:
            problems.append(f"{p.name}: no sample label" + ("" if row else " (no manifest row)"))
        elif sample_label is None and _formula_like(label):
            problems.append(f"{p.name}: sample label {label!r} starts with {label[0]!r}: a spreadsheet would run it "
                            f"as a formula; fix the manifest's {label_col} column")
        if _formula_like(p.name):
            problems.append(f"{p.name}: the file name starts with {p.name[0]!r}: a spreadsheet would run it as a "
                            f"formula; rename the file")
        items.append({"src": p, "ext": ext, "size": size, "nm": nm, "label": label, "md5": md5_file(p)})
    if problems:
        raise PredictSetError(f"refused ({len(problems)} problem(s)):" + _list(problems))
    by_label = {}
    for it in items:
        by_label.setdefault(it["label"], []).append(it)
    for label, its in by_label.items():
        sizes, nms = {}, {}
        for it in its:
            sizes.setdefault("x".join(map(str, it["size"])), []).append(it["src"].name)
            nms.setdefault(it["nm"], []).append(it["src"].name)
        if len(sizes) > 1:
            problems.append(f"sample {label!r}: mixed tile sizes " + "; ".join(
                f"{s} x{len(n)} (e.g. {n[0]})" for s, n in sorted(sizes.items(), key=lambda kv: -len(kv[1]))))
        # contract section 8: one scale when (max - min) / median of every image's nm/px <= NM_TOLERANCE; each
        # image keeps its own value in images.csv
        if len(nms) > 1 and not nm_one_scale([it["nm"] for it in its]):
            problems.append(f"sample {label!r}: mixed nm/px (spread over {NM_TOLERANCE * 100:g} % of the median) "
                            + "; ".join(f"{v:g} x{len(n)} (e.g. {n[0]})"
                                        for v, n in sorted(nms.items(), key=lambda kv: -len(kv[1]))))
    if problems:
        raise PredictSetError(f"refused ({len(problems)} problem(s)); results would refuse mixed sets:"
                              + _list(problems))
    regm = registry_module()
    day = datetime.now().strftime("%Y%m%d")
    if dry_run:
        ps_id = f"ps_{day}_000000"                   # placeholder: a real create picks a free random id
        rec = _set_record(ps_id, C.now(), owner, model, where, record_path(root / ps_id), session, len(items), conf,
                          max_det, via, set_note, prov)
        errs = regm.check_record(rec, regm.schemas())
        if errs:
            raise PredictSetError(f"refused: registry record invalid: {'; '.join(errs[:4])}")
        return {"would_create": str(root / f"ps_{day}_<6 hex>"), "registry": str(reg), "n_images": len(items),
                "samples": {k: len(v) for k, v in sorted(by_label.items())}, "model": model, "owner": owner,
                "conf": conf, "max_det": max_det, "imgsz": imgsz, "session": session, "name": set_name,
                "place": "hard-link where possible" if link else "copy (md5 checked)",
                "note": set_note, "provenance": prov,
                "source_manifest": f"{SOURCE_MANIFEST} (a copy of {manifest})" if manifest is not None else None,
                "record": {k: v for k, v in rec.items() if k not in ("id", "path", "logged", "created")}}
    # ---- id + record, validated before anything is written; the id is reserved by its .incoming folder
    with locked(reg / REGISTRY_LOCK):
        taken = set(regm.current("prediction_set", reg))
        for _ in range(50):
            ps_id = f"ps_{day}_{secrets.token_hex(3)}"
            if ps_id not in taken and not (root / ps_id).exists() and not (root / INCOMING / ps_id).exists():
                break
        else:
            raise PredictSetError("could not make a free prediction set id")
        inc = root / INCOMING / ps_id
        (inc / IMAGES).mkdir(parents=True)
    created = C.now()
    final = root / ps_id
    rec = _set_record(ps_id, created, owner, model, where, record_path(final), session, len(items), conf, max_det,
                      via, set_note, prov)
    try:
        errs = regm.check_record(rec, regm.schemas())
        if errs:
            raise PredictSetError(f"refused: registry record invalid: {'; '.join(errs[:4])}")
        rows, how = [], {"link": 0, "copy": 0}
        tag = f"session:{session}" if session else "upload"
        for i, it in enumerate(items, 1):
            name = f"{i:04d}{it['ext']}"
            tick(i - 1, len(items), "placing")
            how[_place(it["src"], inc / IMAGES / name, it["md5"], link)] += 1
            rows.append({"image": f"{i:04d}", "file": f"{IMAGES}/{name}", "orig_name": it["src"].name,
                         "sample_label": it["label"], "nm_per_px": repr(float(it["nm"])),
                         "nm_confirmed_by": nm_confirmed_by, "source": tag, "md5": it["md5"]})
        if manifest is not None:                 # per-tile columns (parent image, x/y offset, ...) stay with the set
            shutil.copyfile(manifest, inc / SOURCE_MANIFEST)
            if md5_file(inc / SOURCE_MANIFEST) != md5_file(manifest):
                raise PredictSetError(f"copy of {manifest.name} differs from the source (md5)")
        with open(inc / IMAGES_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        source = {"kind": "session", "session": str(session), "dir": str(from_dir.resolve())} if session \
            else {"kind": "upload"}
        request = {"ps": ps_id, "owner": owner, "model": model, "conf": conf, "max_det": max_det, "imgsz": imgsz,
                   "source": source, "created": created}
        if set_name:
            request["name"] = set_name
        if via:                                  # an agent asked, and the pixel size was confirmed through it
            request["owner_via"] = via
            request["nm_confirmed_via"] = via
        if set_note:
            request["note"] = set_note
        if prov:
            request["provenance"] = prov
        if manifest is not None:
            request["source_manifest"] = SOURCE_MANIFEST
        write_json_atomic(inc / REQUEST, request)
        write_json_atomic(inc / STATUS, {"state": "queued", "n_done": 0, "n_total": len(rows), "where": where,
                                         "job": None, "error": None, "updated": created, "pid": None,
                                         "waiting": None})
        tick(len(items), len(items), "placing")
        with locked(reg / REGISTRY_LOCK):
            inc.rename(final)
            try:
                regm.append(rec, reg, **({"via": via} if via else {}))
            except Exception as e:
                write_status(final, state="failed", error=f"registry record not written: {e}")
                raise PredictSetError(f"{final} was written but its registry record was not ({e}); "
                                      f"status set to failed")
    except BaseException:
        shutil.rmtree(inc, ignore_errors=True)          # only this call's own half-built folder
        raise
    print(f"[predict_set] {ps_id}: {len(rows)} image(s) ({how['copy']} copied, {how['link']} linked), "
          f"{len(by_label)} sample(s), model {model} -> {final}")
    return final


def _set_record(ps_id, created, owner, model, where, path, session, n_images, conf, max_det, via=None, note=None,
                provenance=None) -> dict:
    """The prediction_set/1 line create appends (status queued); owner_via only when an agent asked for it; note
    and provenance only when given (data fields: update_record carries them to every later line)."""
    src = {"kind": "session", "session": str(session)} if session else {"kind": "upload"}
    rec = {"schema": "prediction_set/1", "id": ps_id, "logged": created, "created": created, "owner": owner,
           "model": model, "job": None, "where": where, "path": path, "source": src,
           "n_images": n_images, "conf": conf, "max_det": max_det, "status": "queued",
           "review_package": None, "results": None, "recorded_by": "jobs/predict_set.py create"}
    if via:
        rec["owner_via"] = rec["nm_confirmed_via"] = via
    if note:
        rec["note"] = note
    if provenance:
        rec["provenance"] = provenance
    return rec


# ---------------------------------------------------------------- adopt (round 14, contract C6)
ADOPTED = "adopted"                              # <ps>/adopted/: what the set was adopted from
PARTICLE_LINES = "particle_lines.csv"            # <ps>/adopted/particle_lines.csv (annotation/object_set.py reads it)
SOURCE_FILES = "source_files.csv"                # <ps>/adopted/source_files.csv: rel path, size, sha256
RUN_FILES = ("run_manifest.json", "data/D1_particles.csv", "data/D2_images.csv")
ADOPT_IMAGE_EXT = (".png", ".jpg", ".jpeg")
ADOPT_MAX_DET = 300                              # the poster run's infer.py (max_det=300)
RUN_IMAGE_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.+ -]{0,199}[A-Za-z0-9_+-])?\Z")  # a run image id names
#   a file: no separators, ':' or leading dot; no trailing dot or space; the poster's ids hold '+', '.', '-' and spaces
WIN_RESERVED_RE = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z", re.I)


def plain_run_id(x) -> bool:
    """A run image id that is safe as a file name stem: RUN_IMAGE_ID_RE, no '..', no Windows reserved name."""
    x = str(x or "")
    return bool(RUN_IMAGE_ID_RE.match(x)) and ".." not in x and not WIN_RESERVED_RE.match(x.split(".")[0].strip()) \
        and not WIN_RESERVED_RE.match(x)


PARTICLE_COLUMNS = ["image", "run_image_id", "particle_id", "line", "cls", "conf", "area_px", "run_area_px",
                    "area_match"]


def run_short(label) -> str:
    """The run's prediction folder name for a set label (the poster's infer.short: letters and digits, <= 12)."""
    return re.sub(r"[^A-Za-z0-9]+", "", str(label))[:12]


def _sha256_file(p) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_particle_lines(txt: Path, w: int, h: int, conf: float, classes: dict) -> list:
    """The outside run's particle ids -> prediction lines (the poster's infer.particles(), read only): every
    non-blank line of the .txt in file order, rasterized like its loader (integer-truncated points, cv2.fillPoly on
    the polygon's box, clipped to w x h; an empty mask drops the line); kept when its class is ev / non_ev and conf
    >= conf. particle_id = index among the EV ones, then the non-EV ones after them.
    -> [(particle_id, line, class id, conf, mask area px)]."""
    import cv2
    import numpy as np
    ev, non = [], []
    for i, ln in enumerate(Path(txt).read_text().splitlines()):
        p = ln.split()
        if not p:
            continue
        c = [float(v) for v in p[2:]]
        pts = np.asarray([(int(c[k] * w), int(c[k + 1] * h)) for k in range(0, len(c) - 1, 2)], np.int32).reshape(-1, 2)
        if not len(pts):
            continue
        x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x1, y1 = int(pts[:, 0].max()) + 1, int(pts[:, 1].max()) + 1
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [pts - [x0, y0]], 1)
        if not m.any():
            continue
        cy0, cy1, cx0, cx1 = max(y0, 0), min(y1, h), max(x0, 0), min(x1, w)
        if cy1 <= cy0 or cx1 <= cx0:
            continue
        mc = m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        if not mc.any():
            continue
        k, cf = int(p[0]), float(p[1])
        raw = classes.get(k, "other")
        if cf >= conf and raw == "ev":
            ev.append((i, k, cf, int(mc.sum())))
        elif cf >= conf and raw == "non_ev":
            non.append((i, k, cf, int(mc.sum())))
    return [(pid, ln, k, cf, a) for pid, (ln, k, cf, a) in enumerate(ev + non)]


def _read_csv_rows(p: Path) -> list:
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def adopt_plan(run_dir, model, images, sets=None) -> dict:
    """Everything adopt needs, read only (nothing is written anywhere). Raises PredictSetError with every problem.
    run_dir: an outside run (run_manifest.json, data/D1_particles.csv, data/D2_images.csv, data/pa/<set>/<image>.txt).
    images: folders searched (recursively) for each image by its D2 file name or image id, .png / .jpg / .jpeg."""
    run_dir = Path(run_dir)
    problems = []
    for rel in RUN_FILES:
        if not (run_dir / rel).is_file():
            problems.append(f"{run_dir / rel} not found")
    if problems:
        raise PredictSetError("refused:" + _list(problems))
    man = json.loads((run_dir / RUN_FILES[0]).read_text(encoding="utf-8"))
    cfg = man.get("config") or {}
    run_model = (cfg.get("model") or {}).get("id") or man.get("model_id")
    if run_model != model:
        problems.append(f"--model {model} is not the run's model ({run_model})")
    try:
        model_spec(model)
    except PredictSetError as e:
        problems.append(str(e))
    params = cfg.get("params") or {}
    conf = params.get("conf")
    nm5760 = (cfg.get("pixel_size") or {}).get("nm_per_px_5760")
    classes = {int(k): v for k, v in ((cfg.get("model") or {}).get("classes") or {}).items()}
    if not isinstance(conf, (int, float)) or not isinstance(nm5760, (int, float)) or not classes:
        problems.append("run_manifest.json lacks config.params.conf, config.pixel_size.nm_per_px_5760 or "
                        "config.model.classes")
    apps = {k: v for k, v in (cfg.get("applications") or {}).items() if isinstance(v, dict) and "path" in v}
    sets = list(sets) if sets else list(apps)
    for s_ in sets:
        if s_ not in apps:
            problems.append(f"set {s_!r} is not one of the run's applications ({', '.join(apps)})")
    excl = {k: set(v) for k, v in ((cfg.get("applications") or {}).get("exclude_images") or {}).items()}
    if problems:
        raise PredictSetError("refused:" + _list(problems))
    d2 = [r for r in _read_csv_rows(run_dir / "data" / "D2_images.csv")
          if r.get("set") in sets and r.get("width_px") and r.get("file") not in excl.get(r.get("set"), set())]
    bad_ids = [str(r.get("image_id")) for r in d2 if not plain_run_id(r.get("image_id"))]
    if bad_ids:                                  # checked before any path is built from them
        raise PredictSetError(f"refused: D2_images.csv image ids that are not plain names: {bad_ids[:5]}")
    d1 = {}
    for r in _read_csv_rows(run_dir / "data" / "D1_particles.csv"):
        if r.get("set") in sets:
            d1.setdefault((r["set"], r["image_id"]), {})[str(r["particle_id"])] = r
    if not d2:
        raise PredictSetError(f"refused: no images of sets {sets} in {run_dir / 'data' / 'D2_images.csv'}")
    if len(d2) > MAX_IMAGES:
        raise PredictSetError(f"refused: {len(d2)} images; a prediction set takes at most {MAX_IMAGES}")
    wanted = {}
    for r in d2:
        wanted.setdefault(r["file"].lower(), []).append(r)
        wanted.setdefault(Path(r["file"]).stem.lower(), []).append(r)
        wanted.setdefault(r["image_id"].lower(), []).append(r)
    found = {}
    for root in images or ():
        root = Path(root)
        if not root.is_dir():
            problems.append(f"--images {root} is not a folder")
            continue
        if protected_problem(root):
            problems.append(f"--images {root} refused: {protected_problem(root)}")
            continue
        for dirpath, _, names in os.walk(root):
            for n in names:
                p = Path(dirpath) / n
                if p.suffix.lower() not in ADOPT_IMAGE_EXT:
                    continue
                for key in {n.lower(), p.stem.lower()}:
                    for r in wanted.get(key, ()):
                        found.setdefault((r["set"], r["image_id"]), set()).add(p)
    rows, sizes = [], {}
    for r in sorted(d2, key=lambda r: (sets.index(r["set"]), r["image_id"])):
        k = (r["set"], r["image_id"])
        txt = run_dir / "data" / "pa" / run_short(r["set"]) / f"{r['image_id']}.txt"
        if not _inside(txt, run_dir / "data" / "pa"):          # belt and braces: never outside the run's pa/
            problems.append(f"{r['set']} {r['image_id']}: its predictions path leaves the run folder")
            continue
        if not txt.is_file():
            problems.append(f"{r['set']} {r['image_id']}: no predictions file {txt.name}")
            continue
        hits = sorted(found.get(k, ()))
        if not hits:
            problems.append(f"{r['set']} {r['image_id']}: image not found under --images")
            continue
        if len(hits) > 1:
            problems.append(f"{r['set']} {r['image_id']}: {len(hits)} candidate images ({hits[0].name}, "
                            f"{hits[1].name}...): pass narrower --images folders")
            continue
        try:
            ext, (w, h) = inspect_image(hits[0])
        except (ValueError, OSError) as e:
            problems.append(f"{hits[0].name}: {e}")
            continue
        W, H = int(r["width_px"]), int(r["height_px"])
        if round(h * W / w) != H:
            problems.append(f"{r['image_id']}: image {w}x{h} is not the run's {W}x{H} shape")
            continue
        nm = nm5760 * 5760 / w
        if w == W and abs(nm - float(r["nm_per_px"])) > 1e-3 * nm:
            problems.append(f"{r['image_id']}: nm/px {nm:.4f} != the run's {r['nm_per_px']}")
            continue
        parts = run_particle_lines(txt, W, H, float(conf), classes)
        ref = d1.get(k, {})
        plines = []
        for pid, ln, kk, cf, area in parts:
            rr = ref.get(str(pid))
            ra = int(rr["mask_area_px"]) if rr and str(rr.get("mask_area_px", "")).isdigit() else None
            plines.append({"particle_id": pid, "line": ln, "cls": kk, "conf": cf, "area_px": area,
                           "run_area_px": ra, "area_match": int(ra == area)})
        extra = sorted(set(ref) - {str(p["particle_id"]) for p in plines})
        if extra:
            problems.append(f"{r['image_id']}: run particles {extra[:5]} have no line (the mapping rule is off)")
        rows.append({"set": r["set"], "image_id": r["image_id"], "file": r["file"], "src": hits[0], "ext": ext,
                     "w": w, "h": h, "nm": nm, "txt": txt, "plines": plines})
        sizes[hits[0]] = (w, h)
    if problems:
        raise PredictSetError(f"refused ({len(problems)} problem(s)):" + _list(problems))
    return {"run_dir": run_dir, "run_id": man.get("run_id") or run_dir.name, "model": model, "sets": sets,
            "conf": conf, "rows": rows, "classes": classes, "imgsz": (cfg.get("model") or {}).get("imgsz")}


def adopt_file_list(P) -> tuple:
    """[(path relative to the run, size, sha256)] of every run file adopt reads, and the sha256 of that list."""
    import hashlib
    files = [P["run_dir"] / f for f in RUN_FILES] + [r["txt"] for r in P["rows"]]
    out = []
    for f in files:
        out.append((Path(f).relative_to(P["run_dir"]).as_posix(), Path(f).stat().st_size, _sha256_file(f)))
    text = "".join(f"{a}\t{b}\t{c}\n" for a, b, c in out)
    return out, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _inside(p: Path, parent: Path) -> bool:
    try:
        Path(p).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def adopted_before(reg: Path, list_sha: str) -> list:
    """Ids of registered prediction sets adopted from the same files (source.sha256 == list_sha)."""
    regm = registry_module()
    out = []
    for pid, rec in regm.current("prediction_set", reg).items():
        src = rec.get("source") if isinstance(rec, dict) else None
        if isinstance(src, dict) and src.get("kind") == "adopted" and src.get("sha256") == list_sha:
            out.append(pid)
    return sorted(out)


def adopt(run_dir, model, images, name, owner, nm_confirmed_by, sets=None, root=None, registry_root=None,
          dry_run=True, via=None, again=False):
    """Register an outside prediction run (e.g. the poster run) as an ordinary prediction set (status done), read
    only on the source: its images are copied (md5 checked) from --images, its prediction files byte-copied to
    p/<model>/predictions/<nnnn>.txt, and <ps>/adopted/ gets particle_lines.csv (run particle id -> line, each
    checked against the run's own mask area) and source_files.csv. source = {kind: adopted, path, run, sha256 of
    the file list}. dry_run (default): every check and the plan; nothing written. -> plan dict or the new folder.
    Refused: --root or --registry inside the run folder (read only means nothing is written there); a run whose file
    list (source.sha256) is already registered as an adopted set, unless again=True (--again)."""
    set_name = clean_name(name)
    problems = []
    if not owner or not PERSON_RE.match(str(owner)):
        problems.append(f"owner {owner!r} is not a person id")
    if not nm_confirmed_by or not PERSON_RE.match(str(nm_confirmed_by)):
        problems.append("--nm-confirmed-by (a person id) is required: who confirmed the run's nm/px")
    if not set_name:
        problems.append("--name is required (the set's display name)")
    root = Path(root) if root else default_root()
    guard = protected_problem(root)
    if guard:
        problems.append(f"--root refused: {guard}")
    reg = Path(registry_root) if registry_root else config.registry_root()
    for flag, d in (("--root", root), ("--registry", reg)):
        if _inside(d, run_dir):
            problems.append(f"{flag} is inside --from: the run folder is read only")
    if problems:
        raise PredictSetError("refused:" + _list(problems))
    P = adopt_plan(run_dir, model, images, sets)
    files, list_sha = adopt_file_list(P)
    n_part = sum(len(r["plines"]) for r in P["rows"])
    n_match = sum(p["area_match"] for r in P["rows"] for p in r["plines"])
    by_set = {}
    for r in P["rows"]:
        by_set[r["set"]] = by_set.get(r["set"], 0) + 1
    prior = adopted_before(reg, list_sha) if reg.is_dir() else []
    if prior and not again:
        raise PredictSetError(f"refused: this run (the same files, sha256 {list_sha[:12]}) is already adopted as "
                              f"{', '.join(prior)}; pass --again to adopt it once more")
    source = {"kind": "adopted", "path": str(P["run_dir"].resolve()), "run": str(P["run_id"]), "sha256": list_sha}
    summary = {"run": str(P["run_id"]), "model": model, "n_images": len(P["rows"]), "by_set": by_set,
               "n_particles": n_part, "n_area_match": n_match, "files_sha256": list_sha, "n_files": len(files),
               "root": str(root), "registry": str(reg), "name": set_name, "owner": owner,
               "examples": [{"image_id": r["image_id"], **p} for r in P["rows"][:5] for p in r["plines"][:1]]}
    if n_match != n_part:
        bad = [(r["image_id"], p["particle_id"]) for r in P["rows"] for p in r["plines"] if not p["area_match"]]
        summary["area_mismatch"] = bad[:20]
    regm = registry_module()
    if dry_run:
        rec = _adopt_record("ps_00000000_000000", C.now(), owner, model, record_path(root / "ps_x"), source,
                            len(P["rows"]), P["conf"], via, set_name, P)
        errs = regm.check_record(rec, regm.schemas())
        if errs:
            raise PredictSetError(f"refused: registry record invalid: {'; '.join(errs[:4])}")
        return summary
    if n_match != n_part:
        raise PredictSetError(f"refused: {n_part - n_match} run particles don't match their line by mask area")
    day = datetime.now().strftime("%Y%m%d")
    with locked(reg / REGISTRY_LOCK):
        if not again and adopted_before(reg, list_sha):          # re-checked under the lock (a racing adopt)
            raise PredictSetError("refused: this run was just adopted; pass --again to adopt it once more")
        taken = set(regm.current("prediction_set", reg))
        for _ in range(50):
            ps_id = f"ps_{day}_{secrets.token_hex(3)}"
            if ps_id not in taken and not (root / ps_id).exists() and not (root / INCOMING / ps_id).exists():
                break
        else:
            raise PredictSetError("could not make a free prediction set id")
        inc = root / INCOMING / ps_id
        (inc / IMAGES).mkdir(parents=True)
    created = C.now()
    final = root / ps_id
    rec = _adopt_record(ps_id, created, owner, model, record_path(final), source, len(P["rows"]), P["conf"], via,
                        set_name, P)
    try:
        errs = regm.check_record(rec, regm.schemas())
        if errs:
            raise PredictSetError(f"refused: registry record invalid: {'; '.join(errs[:4])}")
        pred = inc / "p" / model / "predictions"
        pred.mkdir(parents=True)
        (inc / ADOPTED).mkdir()
        rows, plines = [], []
        for i, r in enumerate(P["rows"], 1):
            img = f"{i:04d}"
            name_ = f"{img}{r['ext']}"
            _place(r["src"], inc / IMAGES / name_, md5_file(r["src"]), False)
            shutil.copyfile(r["txt"], pred / f"{img}.txt")
            rows.append({"image": img, "file": f"{IMAGES}/{name_}", "orig_name": r["src"].name,
                         "sample_label": r["set"], "nm_per_px": repr(float(r["nm"])),
                         "nm_confirmed_by": nm_confirmed_by, "source": f"adopted:{P['run_id']}",
                         "md5": md5_file(inc / IMAGES / name_)})
            plines += [{"image": img, "run_image_id": r["image_id"], **p} for p in r["plines"]]
        with open(inc / IMAGES_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        with open(inc / ADOPTED / PARTICLE_LINES, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=PARTICLE_COLUMNS, lineterminator="\n")
            w.writeheader()
            w.writerows(plines)
        with open(inc / ADOPTED / SOURCE_FILES, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["path", "size", "sha256"])
            w.writerows(files)
        request = {"ps": ps_id, "owner": owner, "model": model, "conf": P["conf"], "max_det": ADOPT_MAX_DET,
                   "imgsz": P["imgsz"],
                   "source": source, "created": created, "name": set_name}
        if via:
            request["owner_via"] = request["nm_confirmed_via"] = via
        write_json_atomic(inc / REQUEST, request)
        write_json_atomic(inc / STATUS, {"state": "done", "n_done": len(rows), "n_total": len(rows),
                                         "where": "laptop", "job": None, "error": None, "updated": created,
                                         "pid": None, "waiting": None})
        with locked(reg / REGISTRY_LOCK):
            inc.rename(final)
            try:
                regm.append(rec, reg, **({"via": via} if via else {}))
            except Exception as e:
                write_status(final, state="failed", error=f"registry record not written: {e}")
                raise PredictSetError(f"{final} was written but its registry record was not ({e}); status set to "
                                      f"failed")
    except BaseException:
        shutil.rmtree(inc, ignore_errors=True)
        raise
    print(f"[predict_set] {ps_id}: adopted {len(rows)} image(s) from run {P['run_id']} -> {final}")
    return final


def _adopt_record(ps_id, created, owner, model, path, source, n_images, conf, via, name, P) -> dict:
    rec = {"schema": "prediction_set/1", "id": ps_id, "logged": created, "created": created, "owner": owner,
           "model": model, "job": None, "where": "laptop", "path": path, "source": source, "n_images": n_images,
           "conf": float(conf), "status": "done", "review_package": None, "results": None,
           "recorded_by": "jobs/predict_set.py adopt",
           "note": f"adopted from outside run {P['run_id']} (sets {', '.join(P['sets'])}); predictions copied "
                   f"byte-identical; particle ids mapped to lines in adopted/{PARTICLE_LINES}"[:NOTE_MAX]}
    if via:
        rec["owner_via"] = rec["nm_confirmed_via"] = via
    return rec


# ---------------------------------------------------------------- CLI
def build_parser(allow_abbrev=True) -> argparse.ArgumentParser:
    """The CLI parser; `.create` is the create subparser (python -m cryoev predict new reads its exact options)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=allow_abbrev)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="make a prediction set (status queued)", allow_abbrev=allow_abbrev)
    c.add_argument("--from-dir", required=True, type=Path, help="folder of 8-bit PNG/JPG tiles")
    c.add_argument("--manifest", type=Path, help="tile MANIFEST.csv (output_name, nm_per_px, sample columns)")
    c.add_argument("--only", type=Path, help="list file: only these file names from --from-dir (one per line)")
    c.add_argument("--model", required=True, help="model id from models/models.yaml")
    c.add_argument("--owner", required=True, help="person id of the set's owner")
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample-label")
    g.add_argument("--sample-label-from", help="manifest:<column>")
    n = c.add_mutually_exclusive_group(required=True)
    n.add_argument("--nm-per-px", type=float, help="nm per pixel, the same for every image")
    n.add_argument("--nm-from-manifest", action="store_true", help=f"per image, from the manifest's {NM_COLUMN}")
    c.add_argument("--nm-confirmed-by", help="person id who confirmed the nm/px (required)")
    c.add_argument("--conf", default=0.25)
    c.add_argument("--max-det", default=1000)
    c.add_argument("--imgsz", help="inference size (default: the model's registered imgsz)")
    c.add_argument("--session", help="YYYYMMDD when the tiles are a session's (source kind session)")
    c.add_argument("--name", help="display name of the analysis (at most 80 characters)")
    c.add_argument("--link", action="store_true", help="hard-link tiles on the same drive instead of copying")
    c.add_argument("--registry", type=Path, help="registry folder (default <CryoAI-laptop>/registry)")
    c.add_argument("--root", type=Path, help=f"where sets live (default <CryoAI-laptop>/{PREDICT_DIR})")
    c.add_argument("--note", help=f"free text: how/why this set was made (at most {NOTE_MAX} characters)")
    c.add_argument("--provenance", type=Path, help="provenance.json: derived_from, source_images, steps, tiling, "
                                                    "script, git (module doc; at most 16 KB)")
    c.add_argument("--dry-run", action="store_true", help="every check, then the plan; nothing is written")
    ap.create = c
    d = sub.add_parser("adopt", help="register an outside prediction run as a prediction set (dry run unless --do)",
                       allow_abbrev=allow_abbrev)
    d.add_argument("--from", dest="from_run", required=True, type=Path, help="the outside run folder (read only)")
    d.add_argument("--model", required=True, help="model id from models/models.yaml (must be the run's)")
    d.add_argument("--images", required=True, type=Path, action="append",
                   help="folder searched for the run's images (repeatable; never under Box)")
    d.add_argument("--name", required=True, help="display name of the set (at most 80 characters)")
    d.add_argument("--owner", help="person id of the set's owner (default $CRYOEV_FOR)")
    d.add_argument("--nm-confirmed-by", help="person id who confirmed the run's nm/px (default: the owner)")
    d.add_argument("--sets", help="comma-separated run set labels (default: every application set)")
    d.add_argument("--registry", type=Path, help="registry folder (default <CryoAI-laptop>/registry)")
    d.add_argument("--root", type=Path, help=f"where sets live (default <CryoAI-laptop>/{PREDICT_DIR})")
    d.add_argument("--again", action="store_true", help="adopt a run whose files are already adopted once more")
    d.add_argument("--do", action="store_true", help="adopt it (default: a dry run that writes nothing)")
    ap.adopt = d
    return ap


def create_options() -> dict:
    """{exact option string of `create`: number of values it takes (0 for a flag)}."""
    return {o: (0 if act.nargs == 0 else 1) for o, act in build_parser(False).create._option_string_actions.items()}


def main(argv=None):
    a = build_parser().parse_args(argv)
    from data_utils import actor
    try:                                         # an agent session's CRYOEV_VIA (round 11 TODO #70): the CLI path
        via = actor.via()                        # stamps owner_via like the portal's write tokens do; unset = None
    except actor.ActorError as e:
        print(f"[predict_set] {e}", file=sys.stderr)
        return 2
    if a.cmd == "adopt":
        return _adopt_main(a, via, actor)
    try:
        names = None
        if a.only:
            names = [ln.strip() for ln in a.only.read_text(encoding="utf-8-sig").splitlines()
                     if ln.strip() and not ln.startswith("#")]
        ps = create(a.from_dir, a.model, a.owner, a.nm_confirmed_by, sample_label=a.sample_label,
                    sample_label_column=a.sample_label_from, nm_per_px=a.nm_per_px,
                    nm_from_manifest=a.nm_from_manifest, manifest=a.manifest, names=names, conf=a.conf,
                    max_det=a.max_det, imgsz=a.imgsz, session=a.session, registry_root=a.registry, root=a.root,
                    link=a.link, name=a.name, dry_run=a.dry_run, note=a.note, provenance=a.provenance,
                    via=via)
    except (PredictSetError, config.ConfigError, OSError) as e:
        print(f"[predict_set] {e}", file=sys.stderr)
        return 2
    if a.dry_run:
        print(f"[predict_set] DRY RUN, nothing written: would create {ps['would_create']} ({ps['n_images']} "
              f"image(s), {ps['place']}) and append a prediction_set/1 line (status queued) to "
              f"{ps['registry']}/prediction_set.jsonl")
        print(json.dumps({k: v for k, v in ps.items() if k not in ("would_create", "registry")}, indent=1,
                         default=str))
        return 0
    print(ps.name)
    return 0


def _adopt_main(a, via, actor) -> int:
    try:
        owner = a.owner or actor.who(None, None)
        res = adopt(a.from_run, a.model, a.images, a.name, owner, a.nm_confirmed_by or owner,
                    sets=[x.strip() for x in a.sets.split(",") if x.strip()] if a.sets else None, root=a.root,
                    registry_root=a.registry, dry_run=not a.do, via=via, again=a.again)
    except (PredictSetError, config.ConfigError, OSError, actor.ActorError) as e:
        print(f"[predict_set] {e}", file=sys.stderr)
        return 2
    if not a.do:
        print(f"[predict_set] DRY RUN, nothing written: would adopt {res['n_images']} image(s) of run {res['run']} "
              f"({res['by_set']}) as a prediction set (status done) under {res['root']} and append a "
              f"prediction_set/1 line to {res['registry']}/prediction_set.jsonl")
        print(json.dumps(res, indent=1, default=str))
        return 0 if res["n_area_match"] == res["n_particles"] else 2
    print(res.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
