#!/usr/bin/env python3
"""Prediction sets (BUILD_CONTRACTS.md section 2): a user's tiles, ready to be predicted by one registry model.

    python -m jobs.predict_set create --from-dir <dir> [--manifest <MANIFEST.csv>] [--only <list file>]
        --model <model id> --owner <person>
        (--sample-label <label> | --sample-label-from manifest:<column>)
        (--nm-per-px <x> | --nm-from-manifest) --nm-confirmed-by <person>
        [--conf 0.25] [--max-det 1000] [--imgsz N] [--session YYYYMMDD] [--registry <dir>] [--root <dir>]

Writes <root>/<ps id>/ (default root <CryoAI-laptop>/predict; the id ps_<YYYYMMDD>_<6 hex> is made here):
  images/<nnnn>.png|.jpg   the tiles, numbered in name order; the extension comes from the content. Hard-linked
                           when on the same drive (and not from a protected folder), else copied. The original file
                           name is never in a path, only in images.csv orig_name
  images.csv               image,file,orig_name,sample_label,nm_per_px,nm_confirmed_by,source
  request.json             ps, owner, model, conf, max_det, imgsz, source, created
  status.json              state queued, n_done 0, n_total, where laptop, job null, error null, updated
and appends a prediction_set/1 record (status queued) to the registry. The folder is built in <root>/.incoming/ and
renamed into place, so a reader never sees half a set.

Refused, with every problem listed and nothing written: no images, or more than 200; a file that is not an 8-bit
PNG or JPEG by its content (the extension is not trusted; MANIFEST.csv, the --manifest file and OS files such as
desktop.ini are skipped); nm/px missing for any image; no --nm-confirmed-by; mixed tile sizes (or mixed nm/px)
within one sample label; an unknown model. The manifest's name column is output_name (or image, file_name, filename,
name), its nm/px column nm_per_px.

Python API (the portal): create(...) -> the set's folder; find(ps) -> folder; read_status / write_status;
update_record(ps_id, **fields) appends the registry line that follows a status change.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import secrets
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "registry"))
import common as C  # noqa: E402
from data_utils import config  # noqa: E402
from data_utils.path_guard import protected_problem  # noqa: E402

MAX_IMAGES = 200
PREDICT_DIR = "predict"                          # <CryoAI-laptop>/predict/<ps id>/
INCOMING = ".incoming"
IMAGES, IMAGES_CSV, REQUEST, STATUS = "images", "images.csv", "request.json", "status.json"
CSV_COLUMNS = ["image", "file", "orig_name", "sample_label", "nm_per_px", "nm_confirmed_by", "source"]
NAME_COLUMNS = ("output_name", "image", "file_name", "filename", "name")   # as inference/inference_check.py
NM_COLUMN = "nm_per_px"
SKIP_NAMES = {"manifest.csv", "desktop.ini", "thumbs.db", ".ds_store"}
PS_RE = re.compile(r"^ps_\d{8}_[0-9a-f]{6}$")
PERSON_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")       # registry/schemas/prediction_set.json owner
SESSION_RE = re.compile(r"^\d{8}(-\d+)?$")
PNG_SIG, JPEG_SIG = b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff"
PNG_MODES, JPEG_MODES = {"L", "LA", "RGB", "RGBA", "P"}, {"L", "RGB"}
MODELS_YAML = REPO / "models" / "models.yaml"
LAPTOP_PREFIX = "<CryoAI-laptop>"
SHOW = 12                                        # problems listed per refusal


class PredictSetError(ValueError):
    """A refusal or a broken set; the message lists every problem found."""


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
            import registry
            rec = registry.current("prediction_set", registry_root).get(ps)
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


def write_status(ps_dir, **fields) -> dict:
    """Merge fields into status.json (atomic write), with `updated` = now. -> the new status."""
    p = Path(ps_dir) / STATUS
    st = C.read_json(p) if p.is_file() else {}
    st.update(fields, updated=C.now())
    out = {k: st.get(k) for k in ("state", "n_done", "n_total", "where", "job", "error", "updated")}
    C.write_json(p, out)
    return out


def read_images(ps_dir) -> list:
    with open(Path(ps_dir) / IMAGES_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def update_record(ps_id, registry_root=None, recorded_by="jobs/predict_set.py", **fields):
    """Append the set's registry line with `fields` changed (status, where, job, ...). -> the stored line.
    Raises PredictSetError when the set has no record."""
    import registry
    cur = registry.current("prediction_set", registry_root).get(ps_id)
    if cur is None:
        raise PredictSetError(f"{ps_id} has no prediction_set record in the registry")
    rec = {**cur, **fields, "logged": C.now(), "recorded_by": recorded_by}
    return registry.append(rec, registry_root)


# ---------------------------------------------------------------- checks
def inspect_image(path: Path) -> tuple:
    """-> (ext, (w, h)) for an 8-bit PNG or JPEG, judged by content. Raises ValueError with the reason."""
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
        with Image.open(path) as im:
            mode, size = im.mode, im.size
            im.load()                                  # a truncated or corrupt file fails here, not on the GPU
    except Exception as e:
        raise ValueError(f"unreadable {ext[1:].upper()}: {type(e).__name__}: {e}")
    if mode not in modes:
        raise ValueError(f"{ext[1:].upper()} mode {mode} (need 8-bit {', '.join(sorted(modes))})")
    return ext, size


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


def _list(problems):
    more = f"\n  ... and {len(problems) - SHOW} more" if len(problems) > SHOW else ""
    return "\n  " + "\n  ".join(problems[:SHOW]) + more


def _scan(from_dir: Path, manifest: Path | None, names) -> list:
    """The candidate files, sorted by name (names: only these)."""
    if names is not None:
        missing = [n for n in names if not (from_dir / n).is_file()]
        if missing:
            raise PredictSetError(f"{len(missing)} listed file(s) not in {from_dir}:" + _list(missing))
        return sorted({from_dir / n for n in names}, key=lambda p: p.name)
    skip = {manifest.resolve()} if manifest else set()
    return sorted((p for p in from_dir.iterdir() if p.is_file() and p.name.lower() not in SKIP_NAMES
                   and not p.name.startswith(".") and p.resolve() not in skip), key=lambda p: p.name)


def _link_or_copy(src: Path, dst: Path, link: bool) -> str:
    if link and protected_problem(src) is None and os.stat(src).st_dev == os.stat(dst.parent).st_dev:
        try:
            os.link(src, dst)
            return "link"
        except OSError:
            pass
    shutil.copy2(src, dst)
    if dst.stat().st_size != src.stat().st_size:
        raise PredictSetError(f"copy of {src.name} is incomplete")
    return "copy"


# ---------------------------------------------------------------- create
def create(from_dir, model, owner, nm_confirmed_by, sample_label=None, sample_label_column=None, nm_per_px=None,
           nm_from_manifest=False, manifest=None, names=None, conf=0.25, max_det=1000, imgsz=None, session=None,
           registry_root=None, root=None, link=True, where="laptop") -> Path:
    """Make a prediction set (see the module doc). -> its folder. Raises PredictSetError (nothing written)."""
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
    if (nm_from_manifest or sample_label_column) and manifest is None:
        problems.append("--nm-from-manifest / --sample-label-from manifest:... need --manifest")
    if manifest is not None and not manifest.is_file():
        problems.append(f"--manifest {manifest} not found")
    if not 0 < float(conf) < 1:
        problems.append(f"conf {conf} must be between 0 and 1")
    if int(max_det) < 1:
        problems.append(f"max_det {max_det} must be >= 1")
    if imgsz is not None and int(imgsz) < 32:
        problems.append(f"imgsz {imgsz} must be >= 32")
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
        items.append({"src": p, "ext": ext, "size": size, "nm": nm, "label": label})
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
        if len(nms) > 1:
            problems.append(f"sample {label!r}: mixed nm/px " + "; ".join(
                f"{v:g} x{len(n)} (e.g. {n[0]})" for v, n in sorted(nms.items(), key=lambda kv: -len(kv[1]))))
    if problems:
        raise PredictSetError(f"refused ({len(problems)} problem(s)); results would refuse mixed sets:"
                              + _list(problems))
    # ---- id, record (validated before anything is written)
    import registry
    from schema_check import check_record
    taken = set(registry.current("prediction_set", reg)) if (reg / "prediction_set.jsonl").exists() else set()
    day = datetime.now().strftime("%Y%m%d")
    for _ in range(50):
        ps_id = f"ps_{day}_{secrets.token_hex(3)}"
        if ps_id not in taken and not (root / ps_id).exists() and not (root / INCOMING / ps_id).exists():
            break
    else:
        raise PredictSetError("could not make a free prediction set id")
    created = C.now()
    final = root / ps_id
    src = {"kind": "session", "session": str(session)} if session else {"kind": "upload"}
    rec = {"schema": "prediction_set/1", "id": ps_id, "logged": created, "created": created, "owner": owner,
           "model": model, "job": None, "where": where, "path": record_path(final), "source": src,
           "n_images": len(items), "conf": float(conf), "max_det": int(max_det), "status": "queued",
           "review_package": None, "results": None, "recorded_by": "jobs/predict_set.py create"}
    errs = check_record(rec, registry.schemas())
    if errs:
        raise PredictSetError(f"refused: registry record invalid: {'; '.join(errs[:4])}")
    # ---- write: .incoming/<id>/, then rename into place, then the registry line
    inc = root / INCOMING / ps_id
    (inc / IMAGES).mkdir(parents=True)
    try:
        rows, how = [], {"link": 0, "copy": 0}
        tag = f"session:{session}" if session else "upload"
        for i, it in enumerate(items, 1):
            name = f"{i:04d}{it['ext']}"
            how[_link_or_copy(it["src"], inc / IMAGES / name, link)] += 1
            rows.append({"image": f"{i:04d}", "file": f"{IMAGES}/{name}", "orig_name": it["src"].name,
                         "sample_label": it["label"], "nm_per_px": repr(float(it["nm"])),
                         "nm_confirmed_by": nm_confirmed_by, "source": tag})
        with open(inc / IMAGES_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        source = {"kind": "session", "session": str(session), "dir": str(from_dir.resolve())} if session \
            else {"kind": "upload"}
        C.write_json(inc / REQUEST, {"ps": ps_id, "owner": owner, "model": model, "conf": float(conf),
                                     "max_det": int(max_det), "imgsz": int(imgsz) if imgsz is not None else None,
                                     "source": source, "created": created})
        C.write_json(inc / STATUS, {"state": "queued", "n_done": 0, "n_total": len(rows), "where": where,
                                    "job": None, "error": None, "updated": created})
        inc.rename(final)
    except BaseException:
        shutil.rmtree(inc, ignore_errors=True)          # only this call's own half-built folder
        raise
    try:
        registry.append(rec, reg)
    except Exception as e:
        write_status(final, state="failed", error=f"registry record not written: {e}")
        raise PredictSetError(f"{final} was written but its registry record was not ({e}); status set to failed")
    print(f"[predict_set] {ps_id}: {len(rows)} image(s) ({how['link']} linked, {how['copy']} copied), "
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
    c.add_argument("--conf", type=float, default=0.25)
    c.add_argument("--max-det", type=int, default=1000)
    c.add_argument("--imgsz", type=int, help="inference size (default: the model's registered imgsz)")
    c.add_argument("--session", help="YYYYMMDD when the tiles are a session's (source kind session)")
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
                    max_det=a.max_det, imgsz=a.imgsz, session=a.session, registry_root=a.registry, root=a.root)
    except (PredictSetError, config.ConfigError, OSError) as e:
        print(f"[predict_set] {e}", file=sys.stderr)
        return 2
    print(ps.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
