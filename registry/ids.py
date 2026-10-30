"""Id helpers for the tracking contract (docs/TRACKING_CONTRACT.md). Pure functions, no I/O."""
import re
from pathlib import Path

RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+\.\w+$")
PATTERNS = {
    "stem": r"^[^/\\\s]+_[A-Za-z0-9]+$",
    "session": r"^\d{8}(-\d+)?$",
    "grid": r"^\d{4,}$",
    "rf_id": r"^[A-Za-z0-9]{20}$",
    "pull": r"^roboflow_pull_\d{8}_\d{4}$",
    "export": r"^roboflow.+$",
    "lv": r"^lv_\d{8}_\d{4}$",
    "pkg": r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    "ic": r"^\d{8}(-\d+)?_.+_\d{8}-\d{6}$",
    "dataset": r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    "bundle": r"^.+__job_\d{8}-\d{6}$",
    "run": r"^\d{4}-\d{2}-\d{2}_\d{6}_.+$",
    "model": r"^[a-z0-9][a-z0-9_]*$",
    "model_new": r"^v\d+_[a-z0-9]+(_[a-z0-9]+)*_\d{8}$",
    "person": r"^[a-z0-9_]+$",
}
_RX = {k: re.compile(v) for k, v in PATTERNS.items()}
PART = re.compile(r"^(?P<parent>.+)_part(?P<n>\d+)$")
RUN = re.compile(r"^(?P<start>\d{4}-\d{2}-\d{2}_\d{6})_y(?P<ver>\d+)(?P<size>[nsmlx])_(?P<imgsz>\d+)_(?P<batch>\d+)b_"
                 r"(?P<epochs>\d+)ep_Run_(?P<num>\d+)$")


def is_id(kind, value):
    return isinstance(value, str) and bool(_RX[kind].match(value))


def stem_of_name(name):
    """Image key: the file name with its extension joined by '_' (`0004_Jul30.jpg` -> `0004_Jul30_jpg`).
    Same rule as annotation/backfill_tracking.stem_of_rf; used for Roboflow names and MANIFEST output_name."""
    p = Path(name)
    return f"{p.stem}_{p.suffix[1:]}" if p.suffix else p.name


def image_key(name):
    """THE image id: rf_key(stem_of_name(name)). This is the form already in pulls (`extra.name`),
    changes.jsonl `image`, tracking CSVs and export file names (`20260126 - MDA-MB KH__x.png` ->
    `20260126-MDA-MB-KH__x_png`). Idempotent on keys and on export stems."""
    n = Path(name).name
    return rf_key(stem_of_export(n) if RF_SUFFIX.search(n) else stem_of_name(n))


def stem_of_export(file_name):
    """Roboflow export file name -> stem (drops `.rf.<hash>.<ext>`)."""
    return RF_SUFFIX.sub("", Path(file_name).name)


def rf_key(stem):
    """Match key between Roboflow-name stems and export stems (exports turn ' - ' and ' ' into '-')."""
    return re.sub(r"[\s-]+", "-", stem)


def parse_run(run_id):
    """run_experiments.py folder name -> dict (start, ver, size, imgsz, batch, epochs, num), or None."""
    m = RUN.match(run_id or "")
    if not m:
        return None
    d = m.groupdict()
    return {**d, **{k: int(d[k]) for k in ("imgsz", "batch", "epochs", "num")}}


def bundle_dataset(bundle_id):
    """`<dataset>__job_<YYYYmmdd-HHMMSS>` -> dataset id, or None."""
    return bundle_id.rsplit("__job_", 1)[0] if is_id("bundle", bundle_id) else None


def package_parent(pkg_id):
    m = PART.match(pkg_id or "")
    return m.group("parent") if m else None


def session_of_ic(batch_id):
    return batch_id.split("_", 1)[0] if is_id("ic", batch_id) else None
