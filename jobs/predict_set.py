#!/usr/bin/env python3
"""Prediction sets (BUILD_CONTRACTS.md section 2): a user's tiles, ready to be predicted by one registry model.

    python -m jobs.predict_set create --from-dir <dir> [--manifest <MANIFEST.csv>] [--only <list file>]
        --model <model id> --owner <person>
        (--sample-label <label> | --sample-label-from manifest:<column>)
        (--nm-per-px <x> | --nm-from-manifest) --nm-confirmed-by <person>
        [--conf 0.25] [--max-det 1000] [--imgsz N] [--session YYYYMMDD] [--link] [--registry <dir>] [--root <dir>]

Writes <root>/<ps id>/ (default root <CryoAI-laptop>/predict; the id ps_<YYYYMMDD>_<6 hex> is made here):
  images/<nnnn>.png|.jpg   the tiles, numbered in name order; the extension comes from the content. Copied (md5
                           checked); hard-linked only with --link, on the same drive and not from a protected folder.
                           The original file name is never in a path, only in images.csv orig_name
  images.csv               image,file,orig_name,sample_label,nm_per_px,nm_confirmed_by,source,md5 (raw values: it is
                           machine-read; labels and names a spreadsheet would run are refused at create)
  request.json             ps, owner, model, conf, max_det, imgsz, source, created
  status.json              state queued, n_done 0, n_total, where laptop, job null, error null, updated, pid null,
                           waiting null
and appends a prediction_set/1 record (status queued) to the registry. The folder is built in <root>/.incoming/ and
renamed into place, so a reader never sees half a set.

Refused, with every problem listed and nothing written: no images, or more than 200; a file that is not an 8-bit
PNG or JPEG by its content (the extension is not trusted; MANIFEST.csv, the --manifest file and OS files such as
desktop.ini are skipped), or one with a side over 8192 px (checked before it is decoded); nm/px missing for any
image; no --nm-confirmed-by; a sample label or file name starting with = + - @ (a spreadsheet would run it); mixed tile sizes (or nm/px spread over 1 % of the median) within one sample label; an unknown model; --only
names that are not plain file names. The manifest's name column is output_name (or image, file_name, filename,
name), its nm/px column nm_per_px.

Python API (the portal): create(...) -> the set's folder. It reads, md5s and decodes every tile (seconds for 200):
call it from a worker thread, never on the event loop. find(ps) -> folder; read_status / write_status (atomic,
read-modify-write under a lock); update_record(ps_id, **fields) appends the registry line after a status change.
Imported as jobs.predict_set it changes no sys.path and loads registry/registry.py by path (as cj_registry, like
portal/app.py), so no portal module is shadowed.
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

MAX_IMAGES = 200
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


class PredictSetError(ValueError):
    """A refusal or a broken set; the message lists every problem found."""


# ---------------------------------------------------------------- registry, locks, atomic writes
_REG_LOCK = threading.Lock()
_THREAD_LOCKS: dict = {}
_THREAD_LOCKS_LOCK = threading.Lock()


def registry_module():
    """registry/registry.py loaded by path as `cj_registry` (once per process, as portal/app.py names it)."""
    with _REG_LOCK:
        mod = sys.modules.get("cj_registry")
        if mod is None:
            spec = importlib.util.spec_from_file_location("cj_registry", REPO / "registry" / "registry.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules["cj_registry"] = mod
        return mod


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
    registry's prediction_set lock. -> the stored line. Raises PredictSetError when the set has no record."""
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
           registry_root=None, root=None, link=False, where="laptop") -> Path:
    """Make a prediction set (see the module doc). -> its folder. Raises PredictSetError (nothing written).
    Slow (md5 + decode of every tile): the portal calls it from a worker thread."""
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
    for p in files:
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
    # ---- id + record, validated before anything is written; the id is reserved by its .incoming folder
    regm = registry_module()
    day = datetime.now().strftime("%Y%m%d")
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
    src = {"kind": "session", "session": str(session)} if session else {"kind": "upload"}
    rec = {"schema": "prediction_set/1", "id": ps_id, "logged": created, "created": created, "owner": owner,
           "model": model, "job": None, "where": where, "path": record_path(final), "source": src,
           "n_images": len(items), "conf": conf, "max_det": max_det, "status": "queued",
           "review_package": None, "results": None, "recorded_by": "jobs/predict_set.py create"}
    try:
        errs = regm.check_record(rec, regm.schemas())
        if errs:
            raise PredictSetError(f"refused: registry record invalid: {'; '.join(errs[:4])}")
        rows, how = [], {"link": 0, "copy": 0}
        tag = f"session:{session}" if session else "upload"
        for i, it in enumerate(items, 1):
            name = f"{i:04d}{it['ext']}"
            how[_place(it["src"], inc / IMAGES / name, it["md5"], link)] += 1
            rows.append({"image": f"{i:04d}", "file": f"{IMAGES}/{name}", "orig_name": it["src"].name,
                         "sample_label": it["label"], "nm_per_px": repr(float(it["nm"])),
                         "nm_confirmed_by": nm_confirmed_by, "source": tag, "md5": it["md5"]})
        with open(inc / IMAGES_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        source = {"kind": "session", "session": str(session), "dir": str(from_dir.resolve())} if session \
            else {"kind": "upload"}
        write_json_atomic(inc / REQUEST, {"ps": ps_id, "owner": owner, "model": model, "conf": conf,
                                          "max_det": max_det, "imgsz": imgsz, "source": source, "created": created})
        write_json_atomic(inc / STATUS, {"state": "queued", "n_done": 0, "n_total": len(rows), "where": where,
                                         "job": None, "error": None, "updated": created, "pid": None,
                                         "waiting": None})
        with locked(reg / REGISTRY_LOCK):
            inc.rename(final)
            try:
                regm.append(rec, reg)
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


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="make a prediction set (status queued)")
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
    c.add_argument("--link", action="store_true", help="hard-link tiles on the same drive instead of copying")
    c.add_argument("--registry", type=Path, help="registry folder (default <CryoAI-laptop>/registry)")
    c.add_argument("--root", type=Path, help=f"where sets live (default <CryoAI-laptop>/{PREDICT_DIR})")
    a = ap.parse_args(argv)
    try:
        names = None
        if a.only:
            names = [ln.strip() for ln in a.only.read_text(encoding="utf-8-sig").splitlines()
                     if ln.strip() and not ln.startswith("#")]
        ps = create(a.from_dir, a.model, a.owner, a.nm_confirmed_by, sample_label=a.sample_label,
                    sample_label_column=a.sample_label_from, nm_per_px=a.nm_per_px,
                    nm_from_manifest=a.nm_from_manifest, manifest=a.manifest, names=names, conf=a.conf,
                    max_det=a.max_det, imgsz=a.imgsz, session=a.session, registry_root=a.registry, root=a.root,
                    link=a.link)
    except (PredictSetError, config.ConfigError, OSError) as e:
        print(f"[predict_set] {e}", file=sys.stderr)
        return 2
    print(ps.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
