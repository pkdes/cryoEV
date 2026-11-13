#!/usr/bin/env python3
"""Prediction sets (BUILD_CONTRACTS.md section 2): a user's tiles, ready to be predicted by one registry model.

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


if __name__ == "__main__":
    sys.exit(main())
