"""
Data module (CJ portal, read only): a plain catalog of WHAT DATA EXISTS, from the registry (sessions, grids, image
sets, label versions, datasets, models) and the processed images on disk.

    GET /data/                          overview: counts and short lists, search box
    GET /data/search?q=                 search by id or sample label across everything
    GET /data/list/<kind>               the full list (sessions, grids, images, labels, datasets, models)
    GET /data/session/<id>              acquisition summary, its grids, a strip of up to 12 images;
                                        ?page=N shows all its images, 24 per page
    GET /data/grid/<id>                 one grid: sample, type, sessions, thumbnails (paged, 24 per page)
    GET /data/model/<id>                status (production badge), dataset, scores from the registry
    GET /data/dataset/<id>              splits, classes, source, models trained on it, thumbnails
    GET /data/labels/<id>               a label version: what it changed, how much was verified, thumbnails
    GET /data/images/<id>               an image set (review package, prediction set, Roboflow export), thumbnails
    GET /data/thumb/<kind>/<id>/<n>.jpg?v=<version>
                                        one thumbnail (about 240 px), disk-cached under <state dir>/data_thumbs
    GET /data/api/overview | search?q= | list/<kind> | <kind>/<id>[?page=N]    the same information as JSON
                                        (schema cj_data/1; other modules reuse it)

Rules:
  - Nothing here writes to the registry or to a data folder. The only write is the thumbnail cache in the portal's
    own state folder.
  - A thumbnail file is resolved on the server from registry / package data only (the kind + id + index in the URL
    pick it); the client never sends a path. Each kind reads only its own folder: a session's
    `<processing>/<id>/3. Processed images`, `<laptop>/review` for packages, `<laptop>/predict` for prediction sets.
    A path is checked as TEXT against that folder first (so nothing outside it is ever touched, which matters for
    network shares), then by real path with links followed.
  - Privacy fails closed. A prediction set, and a review package that is a prediction package (kind "prediction",
    or has a prediction_set, or an id starting pred_), is listed only for its owner and admins (ctx.can_see). The
    owner is read from the registry record, else the package's review_package.json, else the prediction set's
    request.json; when none is known only admins see it. Prediction packages that are on disk but not in the
    registry are found by their folder name and follow the same rule. A label version or dataset shows (and loads
    images from) its package only to people who may see that package.
  - Plain words; every value is escaped; no script on the page.
"""
from __future__ import annotations

import hashlib
import html
import importlib.util
import io
import json
import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import quote

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

PREFIX = "/data"
ROLES = ("admin", "advanced")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TEMPLATE = HERE.parent / "templates" / "data.html"
SCHEMA = "cj_data/1"
REGISTRY_TTL_S = 10.0
FILES_TTL_S = 10.0
OVERVIEW_N = 5               # rows per card on the overview
PAGE_N = 24                  # thumbnails per page
STRIP_N = 12                 # thumbnails in a session's strip
THUMB_PX = 240
THUMB_MAX_PIXELS = 100_000_000      # a source larger than this is refused, not decoded
THUMB_DECODERS = 4                  # at most this many images decoded at once
SEARCH_CAP = 50
MAX_JSON_BYTES = 2_000_000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ENTITIES = ("session", "grid", "image", "dataset", "label_version", "model", "review_package", "roboflow_export",
            "prediction_set", "inference_check")
SESSION_IMAGE_FOLDERS = ("3a. Individual tiles", "3c. Tiles no scalebar - native PNG",
                         "3d. Tiles no scalebar - 14.7 A-px PNG", "3b. Stitched montages")
PROCESSED = "3. Processed images"
LIST_KINDS = ("sessions", "grids", "images", "labels", "datasets", "models")
DETAIL_KINDS = ("session", "grid", "images", "labels", "dataset", "model")
KIND_TITLES = {"sessions": "Sessions", "grids": "Grids", "images": "Image sets", "labels": "Label versions",
               "datasets": "Datasets", "models": "Models"}
DETAIL_LIST = {"session": "sessions", "grid": "grids", "images": "images", "labels": "labels",
               "dataset": "datasets", "model": "models"}
SET_ENTITIES = ("prediction_set", "review_package", "roboflow_export")
SET_KIND_NAMES = {"prediction_set": "Prediction set", "review_package": "Review package",
                  "roboflow_export": "Roboflow export"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$")
SESSION_ID = re.compile(r"^\d{8}(-\d+)?$")
PREDICTION_SET_ID = re.compile(r"^ps_\d{8}_[0-9a-f]{6}$")
PREDICTION_PACKAGE_ID = re.compile(r"^pred_ps_\d{8}_[0-9a-f]{6}$")
SPLITS = ("train", "valid", "test")

STATUS_WORDS = {
    "in_review": "In review", "imaged": "Imaged", "downloaded": "Downloaded", "processed": "Processed",
    "uploaded": "Uploaded", "excluded": "Excluded", "candidate": "Candidate", "production": "Production",
    "registered": "Registered", "retired": "Retired", "superseded": "Replaced by a newer version",
    "abandoned": "Abandoned", "staged": "Staged", "scored": "Scored", "trained": "Trained", "built": "Built",
    "reviewed": "Reviewed", "queued": "Waiting to run", "running": "Running", "done": "Done", "failed": "Failed",
    "cancelled": "Cancelled", "active": "Current", "ready": "Ready", "blocked": "Blocked",
    "in_progress": "In progress"}
METRIC_NAMES = {"mask_map50": "Mask mAP50", "mask_map50_95": "Mask mAP50-95", "box_map50": "Box mAP50",
                "box_map50_95": "Box mAP50-95", "map50": "mAP50", "map50_95": "mAP50-95",
                "precision": "Precision", "recall": "Recall", "per_class": "Score by class"}

esc = html.escape


# ---------------------------------------------------------------- small helpers
def _dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _list(x) -> list:
    return x if isinstance(x, list) else []


def _text(x):
    return x if isinstance(x, str) and x else None


def _number(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def format_number(x, decimals=2) -> str:
    x = _number(x)
    if x is None:
        return "?"
    return f"{x:,.{decimals}f}".rstrip("0").rstrip(".") if decimals else f"{x:,.0f}"


def plural(n, word) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def status_word(status) -> str:
    s = str(status)
    return STATUS_WORDS.get(s) or s.replace("_", " ").capitalize()


def metric_name(key, sub=None) -> str:
    base = METRIC_NAMES.get(str(key)) or str(key).replace("_", " ").capitalize()
    return f"{base} ({sub})" if sub is not None else base


def entity_href(kind: str, eid: str) -> str:
    return f"{PREFIX}/{kind}/{quote(str(eid), safe='')}"


def short_date(s) -> str:
    return s[:10] if isinstance(s, str) else ""


def grid_sort_key(gid):
    return (int(gid), gid) if str(gid).isdigit() else (-1, str(gid))


def slot_sort_key(slot):
    return (0, int(slot), "") if str(slot).isdigit() else (1, 0, str(slot))


class Viewer:
    """Who is asking: may_see(owner) is ctx.can_see, so an unknown owner (None) is for admins only."""

    def __init__(self, can_see, user):
        self.can_see, self.user = can_see, user

    def may_see(self, owner) -> bool:
        return bool(self.can_see(self.user, owner))


# ---------------------------------------------------------------- caches and the registry (read every 10 s)
_lock = threading.Lock()
_registry_cache: dict = {}
_files_cache: dict = {}
_package_cache: dict = {}
_owner_cache: dict = {}
_registry_module_cache = None
_decoders = threading.BoundedSemaphore(THUMB_DECODERS)


def _registry_module():
    """registry/registry.py, loaded once by path (registry/ is not a package)."""
    global _registry_module_cache
    if _registry_module_cache is None:
        spec = importlib.util.spec_from_file_location("cj_data_registry", REPO / "registry" / "registry.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _registry_module_cache = mod
    return _registry_module_cache


def cached(store: dict, key, compute, ttl=FILES_TTL_S):
    """compute() runs outside the lock; two threads may both compute once, the last result is kept."""
    now = time.monotonic()
    with _lock:
        hit = store.get(key)
        if hit is not None and 0 <= now - hit[0] < ttl:
            return hit[1]
    value = compute()
    with _lock:
        store[key] = (now, value)
        if len(store) > 512:
            store.pop(next(iter(store)))
    return value


def read_registry(root, now=None) -> tuple:
    """-> ({entity: {id: record}}, error or None). One read per root per REGISTRY_TTL_S; callers never change the
    records (they are shared between requests)."""
    empty = {e: {} for e in ENTITIES}
    if root is None:
        return empty, "no registry folder configured"
    now = time.monotonic() if now is None else now
    key = str(root)
    with _lock:
        hit = _registry_cache.get(key)
        if hit is not None and 0 <= now - hit[0] < REGISTRY_TTL_S:
            return hit[1], hit[2]
    records, error = {e: {} for e in ENTITIES}, None
    if not Path(root).is_dir():
        error = "registry folder not found"
    else:
        try:
            registry = _registry_module()
            for entity in ENTITIES:
                records[entity] = registry.current(entity, Path(root))
        except Exception as exc:                       # an unreadable registry must not take the page down
            records, error = {e: {} for e in ENTITIES}, f"registry not readable ({type(exc).__name__})"
    with _lock:
        _registry_cache[key] = (now, records, error)
    return records, error


# ---------------------------------------------------------------- where the data folders are, and path checks
def configured_root(which: str):
    """The CryoAI-laptop ("laptop") or Cryo-image-processing ("processing") folder, or None."""
    config = _registry_module().config
    return config.laptop_root(False) if which == "laptop" else config.processing_root()


def known_roots() -> list:
    """Both data roots as real paths in the OS's comparison case (a defence-in-depth check on served files)."""
    return [os.path.normcase(os.path.realpath(r)) for r in (configured_root("laptop"), configured_root("processing"))
            if r]


def _normalised(path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def is_lexically_under(path, root) -> bool:
    """Text-only test (no disk access): is `path` the folder `root` or below it, after collapsing '..'?"""
    p, r = _normalised(path), _normalised(root)
    return p == r or p.startswith(r.rstrip("\\/") + os.sep)


def is_contained(path, root) -> bool:
    """`path` is under `root` as text FIRST (a network share outside it is never touched), then by real path with
    links followed."""
    if not is_lexically_under(path, root):
        return False
    try:
        return is_lexically_under(os.path.realpath(path), os.path.realpath(root))
    except (OSError, ValueError):
        return False


def inside(path, roots) -> bool:
    return any(is_contained(path, r) for r in roots)


def review_folder():
    lap = configured_root("laptop")
    return Path(lap) / "review" if lap else None


def predict_folder():
    lap = configured_root("laptop")
    return Path(lap) / "predict" if lap else None


def session_folder(session_id: str):
    proc = configured_root("processing")
    if not proc or not SESSION_ID.match(session_id or ""):
        return None
    return Path(proc) / session_id / PROCESSED


# ---------------------------------------------------------------- images on disk (read only)
def scan_images(folder: Path, root: Path, prefix=None) -> list:
    """Image files directly in `folder` (sorted by name). `folder` must be under `root`; each file must stay there
    after links are followed."""
    def compute():
        if not is_contained(folder, root):
            return []
        try:
            if not folder.is_dir():
                return []
            found = []
            with os.scandir(folder) as entries:
                for entry in entries:
                    name = entry.name
                    if name.startswith(".") or os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
                        continue
                    if prefix and not name.lower().startswith(prefix):
                        continue
                    if not entry.is_file():
                        continue
                    if entry.is_symlink() and not is_contained(entry.path, root):
                        continue
                    found.append(Path(entry.path))
            found.sort(key=lambda p: p.name.lower())
            return found
        except OSError:
            return []
    return cached(_files_cache, ("scan", str(folder), str(root), prefix), compute)


def session_images(session_id: str, prefix=None) -> list:
    """The session's processed images: the first of 3a / 3c / 3d / 3b that has any."""
    base = session_folder(session_id)
    if base is None:
        return []
    for sub in SESSION_IMAGE_FOLDERS:
        found = scan_images(base / sub, base, prefix)
        if found:
            return found
    return []


def read_small_json(path: Path) -> dict:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            return {}
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


def package_folder(package_id: str, record: dict):
    """The package's folder under <laptop>/review, from its id (else the registry path); None when not there."""
    root = review_folder()
    lap = configured_root("laptop")
    if root is None:
        return None
    candidates = []
    if SAFE_ID.match(package_id or ""):
        candidates.append(root / package_id)
    path_text = _text(record.get("path"))
    if path_text:
        path = Path(path_text)
        candidates += [path] if path.is_absolute() else [Path(lap).parent / path, Path(lap) / path]
    for candidate in candidates:
        if not is_contained(candidate, root):
            continue
        try:
            if candidate.is_dir():
                return candidate
        except OSError:
            continue
    return None


def package_images(folder: Path) -> list:
    """Images of a review package: data.js `images[].src` (relative, under the package), else its images/ folder."""
    root = review_folder()

    def compute():
        found = []
        try:
            text = (folder / "data.js").read_text(encoding="utf-8")
            data, _ = json.JSONDecoder().raw_decode(text, text.index("{"))
        except (OSError, ValueError):
            data = None
        if not isinstance(data, dict):
            return scan_images(folder / "images", root)
        for image in _list(data.get("images")):
            src = _text(_dict(image).get("src"))
            if not src or src.startswith(("/", "\\")) or ":" in src:
                continue
            path = folder / src
            if os.path.splitext(src)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            if not (is_lexically_under(path, folder) and is_contained(path, folder) and is_contained(path, root)):
                continue
            try:
                if path.is_file():
                    found.append(path)
            except OSError:
                continue
        return found
    try:
        key = ("pkg", str(folder), (folder / "data.js").stat().st_mtime_ns)
    except OSError:
        key = ("pkg", str(folder), None)
    return cached(_package_cache, key, compute)


def prediction_images(set_id: str) -> list:
    root = predict_folder()
    if root is None or not PREDICTION_SET_ID.match(set_id or ""):
        return []
    return scan_images(root / set_id / "images", root)


def thumb_version(path) -> str:
    """Changes when the file does, so a cached thumbnail URL can't show a replaced image."""
    try:
        st = Path(path).stat()
    except OSError:
        return ""
    return hashlib.sha256(f"{path}|{st.st_mtime_ns}|{st.st_size}".encode("utf-8", "replace")).hexdigest()[:12]


# ---------------------------------------------------------------- image sets: lookup, owner, visibility
def with_disk_packages(records: dict) -> dict:
    """Prediction packages that are on disk but not in the registry (found by folder name) join the review packages,
    so they follow the same owner rule instead of slipping past it."""
    root = review_folder()
    if root is None:
        return records

    def compute():
        found = {}
        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    if PREDICTION_PACKAGE_ID.match(entry.name) and entry.is_dir():
                        meta = read_small_json(Path(entry.path) / "review_package.json")
                        found[entry.name] = {**meta, "id": entry.name, "kind": "prediction", "unregistered": True}
        except OSError:
            pass
        return found
    extra = cached(_files_cache, ("disk-packages", str(root)), compute)
    if not extra:
        return records
    return {**records, "review_package": {**extra, **records["review_package"]}}


def find_set(records, set_id):
    """An image set by id -> (entity, record) or (None, None): prediction set, review package, Roboflow export."""
    for entity in SET_ENTITIES:
        record = records[entity].get(set_id)
        if record:
            return entity, record
    return None, None


def is_private_set(entity: str, record: dict) -> bool:
    if entity == "prediction_set":
        return True
    return (entity == "review_package" and (record.get("kind") == "prediction" or bool(record.get("prediction_set"))
                                            or str(record.get("id", "")).startswith("pred_")))


def set_owner(entity: str, record: dict):
    """Owner id of a private set: the registry record, else the package's review_package.json, else the prediction
    set's request.json; None when none says."""
    def compute():
        owner = record.get("owner")
        if _text(owner):
            return owner
        set_id = str(record.get("id", ""))
        meta = {}
        if entity == "review_package":
            folder = package_folder(set_id, record)
            meta = read_small_json(folder / "review_package.json") if folder else {}
            if _text(meta.get("owner")):
                return meta["owner"]
        ps = set_id if entity == "prediction_set" else (
            _text(record.get("prediction_set")) or _text(meta.get("prediction_set"))
            or (set_id[len("pred_"):] if set_id.startswith("pred_") else None))
        root = predict_folder()
        if root is not None and ps and PREDICTION_SET_ID.match(ps):
            owner = read_small_json(root / ps / "request.json").get("owner")
            return owner if _text(owner) else None
        return None
    return cached(_owner_cache, (entity, str(record.get("id")), str(record.get("owner"))), compute)


def set_is_visible(viewer: Viewer, entity: str, record: dict) -> bool:
    if is_private_set(entity, record):
        return viewer.may_see(set_owner(entity, record))
    owner = record.get("owner")
    return owner is None or viewer.may_see(owner)


def visible_set(records, set_id, viewer):
    """(entity, record) when the set exists AND this viewer may see it, else (None, None)."""
    entity, record = find_set(records, set_id)
    if record and set_is_visible(viewer, entity, record):
        return entity, record
    return None, None


def package_images_for(records, package_id, viewer) -> list:
    """A review package's images, only when the viewer may see the package (also when it isn't in the registry)."""
    record = records["review_package"].get(package_id) or {"id": package_id}
    if not set_is_visible(viewer, "review_package", record):
        return []
    folder = package_folder(package_id, record)
    return package_images(folder) if folder else []


def set_images(records, set_id, viewer) -> list:
    entity, record = visible_set(records, set_id, viewer)
    if record is None:
        return []
    if entity == "prediction_set":
        return prediction_images(set_id)
    if entity == "review_package":
        folder = package_folder(set_id, record)
        return package_images(folder) if folder else []
    return []


# ---------------------------------------------------------------- list rows
def _when(record):
    return str(record.get("created") or record.get("built") or record.get("logged") or "")


def split_counts(record):
    counts = record.get("counts_trained") if isinstance(record.get("counts_trained"), dict) else None
    if counts is None and isinstance(record.get("counts"), dict):
        counts = {s: _dict(v).get("images") for s, v in record["counts"].items()}
    if not counts:
        return None
    names = [s for s in SPLITS if s in counts] + sorted(set(counts) - set(SPLITS))
    return "/".join("?" if counts[s] is None else str(counts[s]) for s in names) + " " + "/".join(names)


def review_note(record) -> tuple:
    if record.get("review_scope") == "reviewed_only":
        n, m = record.get("n_reviewed"), record.get("n_package_images")
        return (f"verified {'?' if n is None else n} of {'?' if m is None else m} images", "acc")
    return ("includes edits not yet checked by a second person", "warn")


def magnification_text(record) -> str:
    return " and ".join(f"{m:,}×" for m in _list(record.get("magnification")) if _number(m) is not None)


def sample_labels(record) -> list:
    items = sorted(_dict(record.get("sample_labels")).items(), key=lambda kv: slot_sort_key(kv[0]))
    return [v for _, v in items if _text(v)]


def image_count_of(record):
    n = record.get("n_images")
    if n is None and isinstance(record.get("counts"), dict):
        n = sum(_number(_dict(v).get("images")) or 0 for v in record["counts"].values())
    return n


def make_row(kind, entity_id, title, sub, tags=(), highlight=False) -> dict:
    quoted = quote(str(entity_id), safe="")
    return {"kind": kind, "id": entity_id, "title": title, "sub": sub, "url": entity_href(kind, entity_id),
            "api": f"{PREFIX}/api/{kind}/{quoted}", "tags": [list(t) for t in tags], "hl": highlight}


def status_tags(record) -> list:
    return [(status_word(record["status"]), "")] if _text(record.get("status")) else []


def join_bits(*bits) -> str:
    return " · ".join(str(b) for b in bits if b)


def session_rows(records) -> list:
    rows = []
    for sid, record in sorted(records["session"].items(), reverse=True):
        n = record.get("n_imaging")
        labels = sample_labels(record)
        label_text = ", ".join(labels[:3]) + (f" +{len(labels) - 3} more" if len(labels) > 3 else "")
        sub = join_bits(short_date(record.get("date")), record.get("instrument"), magnification_text(record),
                        plural(n, "imaging file") if _number(n) is not None else None, label_text)
        rows.append(make_row("session", sid, f"Session {sid}", sub, status_tags(record)))
    return rows


def grid_rows(records) -> list:
    rows = []
    for gid, record in sorted(records["grid"].items(), key=lambda kv: grid_sort_key(kv[0]), reverse=True):
        sub = join_bits(*[b for b in (record.get("sample_label"), record.get("grid_type"),
                                      short_date(record.get("prep_date"))) if _text(b)])
        rows.append(make_row("grid", gid, f"Grid {gid}", sub))
    return rows


def image_set_rows(records, viewer) -> list:
    sets = [(e, r) for e in SET_ENTITIES for r in records[e].values() if set_is_visible(viewer, e, r)]
    rows = []
    for entity, record in sorted(sets, key=lambda er: (_when(er[1]), er[1]["id"]), reverse=True):
        n = image_count_of(record)
        sub = join_bits(SET_KIND_NAMES[entity], plural(n, "image") if n is not None else None,
                        short_date(_when(record)),
                        status_word(record["status"]) if _text(record.get("status")) else None)
        rows.append(make_row("images", record["id"], record["id"], sub))
    return rows


def label_rows(records) -> list:
    rows = []
    for record in sorted(records["label_version"].values(), key=lambda r: (_when(r), r["id"]), reverse=True):
        n = record.get("n_images_applied")
        sub = join_bits(review_note(record)[0], plural(n, "image") + " changed" if n is not None else None,
                        short_date(_when(record)))
        rows.append(make_row("labels", record["id"], record["id"], sub, status_tags(record)))
    return rows


def dataset_rows(records) -> list:
    rows = []
    for record in sorted(records["dataset"].values(), key=lambda r: (_when(r), r["id"]), reverse=True):
        sub = join_bits(split_counts(record), ", ".join(map(str, _list(record.get("classes")))),
                        short_date(_when(record)))
        rows.append(make_row("dataset", record["id"], record["id"], sub, status_tags(record)))
    return rows


def model_rows(records) -> list:
    ordered = sorted(records["model"].values(),
                     key=lambda r: (r.get("status") == "production", _when(r), r["id"]), reverse=True)
    rows = []
    for record in ordered:
        production = record.get("status") == "production"
        sub = join_bits(f"{record['imgsz']} px" if record.get("imgsz") else None,
                        f"started from {record['init_from']}" if record.get("init_from") else None,
                        short_date(_when(record)))
        tags = [("Production", "prod")] if production else status_tags(record)
        rows.append(make_row("model", record["id"], record["id"], sub, tags, production))
    return rows


def rows_for(records, kind, viewer) -> list:
    if kind == "images":
        return image_set_rows(records, viewer)
    builder = {"sessions": session_rows, "grids": grid_rows, "labels": label_rows, "datasets": dataset_rows,
               "models": model_rows}[kind]
    return builder(records)


def overview(records, viewer) -> dict:
    cards = []
    for kind in ("sessions", "images", "labels", "datasets", "models"):
        rows = rows_for(records, kind, viewer)
        cards.append({"kind": kind, "title": KIND_TITLES[kind], "count": len(rows), "rows": rows[:OVERVIEW_N],
                      "all": f"{PREFIX}/list/{kind}"})
    counts = {c["kind"]: c["count"] for c in cards}
    counts["grids"] = len(records["grid"])
    return {"schema": SCHEMA, "cards": cards, "counts": counts}


def search_words(kind, row, record_of) -> list:
    """The text a row is matched against: its id and title, plus sample labels / notes where there are any."""
    record = record_of(kind, row["id"]) or {}
    words = [row["id"], row["title"]]
    if kind == "sessions":
        words += sample_labels(record)
    elif kind == "grids":
        words += [_text(record.get("sample_label")) or "", _text(record.get("grid_type")) or ""]
    elif kind in ("datasets", "models"):
        words.append(_text(record.get("note")) or "")
    return words


def search(records, query, viewer) -> dict:
    query = (query or "").strip()[:100]
    needle = query.lower()
    result = {"schema": SCHEMA, "q": query, "groups": []}
    if not needle:
        return result
    plain = {"sessions": records["session"], "grids": records["grid"], "labels": records["label_version"],
             "datasets": records["dataset"], "models": records["model"]}

    def record_of(kind, entity_id):
        return plain[kind].get(entity_id) if kind in plain else find_set(records, entity_id)[1]

    # grids whose sample label matches also pull in the sessions that imaged them
    matching_grids = {gid for gid, g in records["grid"].items() if needle in str(g.get("sample_label") or "").lower()}
    for kind in LIST_KINDS:
        rows = rows_for(records, kind, viewer)
        hits = [r for r in rows if any(needle in str(w).lower() for w in search_words(kind, r, record_of))]
        if kind == "sessions" and matching_grids:
            have = {r["id"] for r in hits}
            by_id = {r["id"]: r for r in rows}
            for sid, session in records["session"].items():
                grids_of_session = {v for v in _dict(session.get("session_grids")).values() if v}
                if sid not in have and matching_grids & grids_of_session:
                    hits.append(by_id[sid])
        if hits:
            result["groups"].append({"kind": kind, "title": KIND_TITLES[kind], "count": len(hits),
                                     "rows": hits[:SEARCH_CAP]})
    return result


# ---------------------------------------------------------------- detail dicts (one source for page and API)
def make_detail(kind, entity_id, title, chips=(), fields=(), **extra) -> dict:
    detail = {"schema": SCHEMA, "kind": kind, "id": entity_id, "title": title, "chips": [list(c) for c in chips],
              "fields": [list(f) for f in fields], "n_images": 0, "thumbs": [], "page": 1, "pages": 1,
              "view": "paged"}
    detail.update(extra)
    return detail


def field(label, value, href=None):
    return (label, value, href) if value not in (None, "", []) else None


def present(*fields):
    return [f for f in fields if f]


def thumbnails(kind, entity_id, files, page=None, indexes=None) -> dict:
    """The thumbnail part of a detail: a page of 24, or the given file indexes (a strip)."""
    n = len(files)
    pages = max(1, -(-n // PAGE_N))
    if indexes is None:
        page = min(max(page or 1, 1), pages)
        indexes = range((page - 1) * PAGE_N, min(n, page * PAGE_N))
    else:
        page = 1
    base = f"{PREFIX}/thumb/{kind}/{quote(str(entity_id), safe='')}"
    thumbs = []
    for i in indexes:
        version = thumb_version(files[i])
        thumbs.append({"n": i, "url": f"{base}/{i}.jpg" + (f"?v={version}" if version else "")})
    return {"n_images": n, "page": page, "pages": pages, "thumbs": thumbs}


def evenly_spaced(n, k):
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def acquisition_fields(record, files) -> list:
    defocus, dose = _dict(record.get("defocus_um")), _dict(record.get("dose"))
    pixel_sizes = [x for x in _list(record.get("pixel_size_a")) if _number(x) is not None]
    other_mags = [x for x in _list(record.get("other_magnification")) if _number(x) is not None]
    dates = _dict(record.get("acq_dates"))
    n_other = record.get("n_other")
    other_text = None
    if _number(n_other):
        other_text = f"{plural(n_other, 'file')} below 5,000×" + (
            " (" + ", ".join(f"{x:,}×" for x in other_mags) + ")" if other_mags else "")
    elif other_mags:
        other_text = ", ".join(f"{x:,}×" for x in other_mags)
    voltages = [f"{format_number(x, 0)} kV" for x in _list(record.get("voltage_kv")) if _number(x) is not None]
    imaged = None
    if dates.get("first") and dates.get("last"):
        imaged = f"{dates['first'].replace('T', ' ')[:16]} to {dates['last'].replace('T', ' ')[:16]}"
    n_imaging = record.get("n_imaging")
    return present(
        field("Date", short_date(record.get("date"))),
        field("Instrument", _text(record.get("instrument"))),
        field("Magnification", magnification_text(record)),
        field("Pixel size", " and ".join(f"{format_number(x, 3)} Å/px" for x in pixel_sizes)),
        field("Defocus", f"median {format_number(defocus.get('median'))} µm "
                         f"(range {format_number(defocus.get('min'))} to {format_number(defocus.get('max'))})"
              if defocus else None),
        field("Dose", f"median {format_number(dose.get('median'))} e/Å² "
                      f"(range {format_number(dose.get('min'))} to {format_number(dose.get('max'))})"
              if dose else None),
        field("Voltage", " and ".join(voltages)),
        field("Imaged", imaged),
        field("Imaging files", str(n_imaging) if _number(n_imaging) is not None else None),
        field("Other files", other_text),
        field("Processed images", str(len(files)) if files else None))


def session_grid_cards(records, record, files) -> list:
    sid = record["id"]
    slots = sorted(set(_dict(record.get("sample_labels"))) | set(_dict(record.get("session_grids"))),
                   key=slot_sort_key)
    registry_counts: dict = {}
    for image in records["image"].values():
        if image.get("session") == sid and image.get("session_grid") is not None:
            slot = str(image["session_grid"])
            registry_counts[slot] = registry_counts.get(slot, 0) + 1
    cards = []
    for slot in slots:
        grid_id = _dict(record.get("session_grids")).get(slot)
        grid = records["grid"].get(grid_id) if grid_id else None
        processed = len(session_images(sid, f"grid{slot}_".lower())) if files else 0
        cards.append({"slot": slot, "label": _text(_dict(record.get("sample_labels")).get(slot))
                      or (grid or {}).get("sample_label"),
                      "grid": grid_id if grid else None, "grid_id_recorded": grid_id,
                      "grid_type": (grid or {}).get("grid_type"),
                      "n_images": processed or registry_counts.get(slot, 0),
                      "count_source": "processed files" if processed else "registry records"})
    return cards


def detail_session(records, record, page=None) -> dict:
    sid = record["id"]
    files = session_images(sid)
    detail = make_detail(
        "session", sid, f"Session {sid}", status_tags(record), acquisition_fields(record, files),
        grids=session_grid_cards(records, record, files),
        acquisition={k: record.get(k) for k in (
            "instrument", "magnification", "pixel_size_a", "defocus_um", "dose", "voltage_kv", "binning",
            "acq_dates", "n_mdoc", "n_imaging", "n_other", "other_magnification")},
        gap=_text(record.get("acquisition_gap")))
    if page is None:
        detail.update(thumbnails("session", sid, files, indexes=evenly_spaced(len(files), STRIP_N)))
        detail["view"] = "strip"
    else:
        detail.update(thumbnails("session", sid, files, page))
    return detail


def grid_images(records, grid_id) -> list:
    pairs = set()
    for sid, session in records["session"].items():
        for slot, gid in _dict(session.get("session_grids")).items():
            if gid == grid_id:
                pairs.add((sid, str(slot)))
    for image in records["image"].values():
        if image.get("grid") == grid_id and image.get("session") and image.get("session_grid") is not None:
            pairs.add((str(image["session"]), str(image["session_grid"])))
    files = []
    for sid, slot in sorted(pairs):
        files += session_images(sid, f"grid{slot}_".lower())
    return files


def detail_grid(records, record, page) -> dict:
    gid = record["id"]
    sessions = sorted({sid for sid, s in records["session"].items() if gid in _dict(s.get("session_grids")).values()}
                      | {im["session"] for im in records["image"].values()
                         if im.get("grid") == gid and im.get("session")})
    replaced_by = record.get("replaced_by")
    fields = present(
        field("Sample", _text(record.get("sample_label"))), field("Grid type", _text(record.get("grid_type"))),
        field("Project", _text(record.get("project"))), field("Prepared", short_date(record.get("prep_date"))),
        field("Replicate", str(record["replicate"]) if record.get("replicate") is not None else None),
        field("Glow discharge", _text(record.get("glow_discharge"))), field("Storage", _text(record.get("storage"))),
        field("Montages", str(record["n_montages"]) if record.get("n_montages") is not None else None),
        field("Notes", _text(record.get("notes"))),
        field("Flags", _text(record.get("flags")) if isinstance(record.get("flags"), str) else None),
        field("Replaced by", f"Grid {replaced_by}" if _text(replaced_by) else None,
              entity_href("grid", replaced_by) if _text(replaced_by) and replaced_by in records["grid"] else None))
    detail = make_detail("grid", gid, f"Grid {gid}", [("Replaced", "warn")] if _text(replaced_by) else [], fields,
                         sessions=[{"id": s, "url": entity_href("session", s)} for s in sessions])
    detail.update(thumbnails("grid", gid, grid_images(records, gid), page))
    return detail


def metrics_of(record) -> dict:
    metrics = record.get("metrics") or record.get("scores")
    return metrics if isinstance(metrics, dict) else {}


def metric_rows(metrics: dict) -> list:
    """-> [[plain name, value text]]; one level of nesting becomes 'Name (sub)'."""
    rows = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            rows += [[metric_name(key, sub), format_number(v, 3) if _number(v) is not None else str(v)]
                     for sub, v in value.items()]
        else:
            rows.append([metric_name(key), format_number(value, 3) if _number(value) is not None else str(value)])
    return rows


def detail_model(records, record) -> dict:
    model_id = record["id"]
    names = {model_id, *[a for a in _list(record.get("aliases")) if isinstance(a, str)]}
    checks = [{"id": c["id"], "session": c.get("session"), "n": c.get("n"),
               "status": status_word(c["status"]) if c.get("status") else None, "metrics": metric_rows(metrics_of(c))}
              for c in sorted(records["inference_check"].values(), key=lambda c: c["id"]) if c.get("model") in names]
    production = record.get("status") == "production"
    dataset, init = _text(record.get("dataset")), _text(record.get("init_from"))
    fields = present(
        field("Status", status_word(record["status"]) if _text(record.get("status")) else None),
        field("Trained on", dataset, entity_href("dataset", dataset) if dataset in records["dataset"] else None),
        field("Started from", init, entity_href("model", init) if init in records["model"] else None),
        field("Classes", ", ".join(map(str, _list(record.get("classes"))))),
        field("Image size", f"{record['imgsz']} px" if record.get("imgsz") else None),
        field("Detection threshold", format_number(record.get("conf")) if _number(record.get("conf")) is not None
              else None),
        field("Registered", short_date(record.get("logged"))),
        field("Weights fingerprint", (record.get("weights_sha256") or "")[:12]),
        field("Note", _text(record.get("note"))))
    chips = [("Production", "prod")] if production else status_tags(record)
    return make_detail("model", model_id, model_id, chips, fields, metrics=metric_rows(metrics_of(record)),
                       checks=checks, production=production)


def detail_images(records, entity, record, viewer, page) -> dict:
    set_id = record["id"]
    model, source = _text(record.get("model")), _dict(record.get("source"))
    n = image_count_of(record)
    package_of_set = _text(record.get("review_package"))
    package_shown = package_of_set if package_of_set and (
        package_of_set not in records["review_package"]
        or set_is_visible(viewer, "review_package", records["review_package"][package_of_set])) else None
    fields = present(
        field("Kind", SET_KIND_NAMES[entity]), field("Images", str(n) if n is not None else None),
        field("Status", status_word(record["status"]) if _text(record.get("status")) else None),
        field("Created", short_date(_when(record))),
        field("Model", model, entity_href("model", model) if model in records["model"] else None),
        field("Detection threshold", format_number(record.get("conf")) if _number(record.get("conf")) is not None
              else None),
        field("Source", (f"session {source['session']}" if source.get("session") else _text(source.get("kind")))
              if source else None,
              entity_href("session", source["session"]) if source.get("session") in records["session"] else None),
        field("Reviewers", ", ".join(map(str, _list(record.get("reviewers"))))),
        field("Edits recorded", str(record["n_edit_records"]) if record.get("n_edit_records") is not None else None),
        field("Starts from", _text(record.get("base")) or _text(record.get("derived_from"))),
        field("Splits", split_counts(record)), field("Classes", ", ".join(map(str, _list(record.get("classes"))))),
        field("Review package", package_shown,
              entity_href("images", package_shown) if package_shown in records["review_package"] else None))
    open_review = None
    if entity == "review_package" and PREDICTION_PACKAGE_ID.match(set_id):
        open_review = f"/review/{quote(set_id, safe='')}/"
    elif package_shown and PREDICTION_PACKAGE_ID.match(package_shown):
        open_review = f"/review/{quote(package_shown, safe='')}/"
    detail = make_detail("images", set_id, set_id, [(SET_KIND_NAMES[entity], "acc")], fields,
                         set_kind=entity, review_url=open_review)
    if entity == "roboflow_export":
        detail["note"] = "Thumbnails are not shown for Roboflow exports."
    detail.update(thumbnails("images", set_id, set_images(records, set_id, viewer), page))
    return detail


def detail_labels(records, record, viewer, page) -> dict:
    label_id = record["id"]
    package = _text(record.get("pkg"))
    package_record = records["review_package"].get(package) or {"id": package}
    package_allowed = bool(package) and set_is_visible(viewer, "review_package", package_record)
    note, tone = review_note(record)
    built = sorted(d["id"] for d in records["dataset"].values() if d.get("label_version") == label_id)
    ops = {k: v for k, v in _dict(record.get("counts_by_op")).items() if _number(v)}
    replaces = _text(record.get("supersedes"))
    shown_package = package if package_allowed else None
    fields = present(
        field("Status", status_word(record["status"]) if _text(record.get("status")) else None),
        field("Checking", note),
        field("Images changed", str(record["n_images_applied"]) if record.get("n_images_applied") is not None
              else None),
        field("Changes", ", ".join(f"{k} {v}" for k, v in ops.items())),
        field("Starts from", _text(record.get("base"))),
        field("Review package", shown_package,
              entity_href("images", shown_package) if shown_package in records["review_package"] else None),
        field("Replaces", replaces, entity_href("labels", replaces) if replaces in records["label_version"] else None),
        field("Created", short_date(_when(record))), field("Note", _text(record.get("note"))))
    detail = make_detail("labels", label_id, label_id, [(note, tone)] + status_tags(record), fields,
                         datasets=[{"id": i, "url": entity_href("dataset", i)} for i in built])
    files = package_images_for(records, package, viewer) if package else []
    detail.update(thumbnails("labels", label_id, files, page))
    return detail


def dataset_images(records, record, viewer) -> list:
    label = records["label_version"].get(record.get("label_version") or "")
    package = _text((label or {}).get("pkg"))
    return package_images_for(records, package, viewer) if package else []


def detail_dataset(records, record, viewer, page) -> dict:
    dataset_id = record["id"]
    files = dataset_images(records, record, viewer)
    source, label = _dict(record.get("source")), _text(record.get("label_version"))
    models = sorted(m["id"] for m in records["model"].values() if m.get("dataset") == dataset_id)
    class_map = _dict(record.get("class_map"))
    split = _dict(record.get("split"))
    source_id = _text(source.get("id"))
    source_is_set = bool(source_id) and visible_set(records, source_id, viewer)[1] is not None
    fields = present(
        field("Status", status_word(record["status"]) if _text(record.get("status")) else None),
        field("Created", short_date(_when(record))), field("Splits (images)", split_counts(record)),
        field("Classes", ", ".join(map(str, _list(record.get("classes"))))),
        field("Merged classes", ", ".join(f"{k} as {v}" for k, v in class_map.items())),
        field("Label version", label, entity_href("labels", label) if label in records["label_version"] else None),
        field("Source", (source_id or _text(source.get("kind"))) if source else None,
              entity_href("images", source_id) if source_is_set else None),
        field("Split rule", f"{split['method']}, seed {split.get('seed')}" if split.get("method") else None),
        field("Note", _text(record.get("note"))))
    detail = make_detail("dataset", dataset_id, dataset_id, status_tags(record), fields,
                         models=[{"id": m, "url": entity_href("model", m)} for m in models])
    if not files:
        detail["note"] = "No example images are linked to this dataset."
    detail.update(thumbnails("dataset", dataset_id, files, page))
    return detail


def detail(records, kind, entity_id, viewer, page=None):
    """-> detail dict, or None when unknown or hidden from this viewer."""
    if kind == "session":
        record = records["session"].get(entity_id)
        return detail_session(records, record, page) if record else None
    if kind == "grid":
        record = records["grid"].get(entity_id)
        return detail_grid(records, record, page) if record else None
    if kind == "model":
        record = records["model"].get(entity_id)
        return detail_model(records, record) if record else None
    if kind == "dataset":
        record = records["dataset"].get(entity_id)
        return detail_dataset(records, record, viewer, page) if record else None
    if kind == "labels":
        record = records["label_version"].get(entity_id)
        return detail_labels(records, record, viewer, page) if record else None
    if kind == "images":
        entity, record = visible_set(records, entity_id, viewer)
        return detail_images(records, entity, record, viewer, page) if record else None
    return None


def thumbnail_sources(records, kind, entity_id, viewer) -> list:
    """The ordered image files behind /data/thumb/<kind>/<id>/<n>.jpg, same order as the detail pages use."""
    if kind == "session":
        return session_images(entity_id) if entity_id in records["session"] else []
    if kind == "grid":
        return grid_images(records, entity_id) if entity_id in records["grid"] else []
    if kind == "images":
        return set_images(records, entity_id, viewer)
    if kind == "labels":
        package = _text(_dict(records["label_version"].get(entity_id)).get("pkg"))
        return package_images_for(records, package, viewer) if package else []
    if kind == "dataset":
        record = records["dataset"].get(entity_id)
        return dataset_images(records, record, viewer) if record else []
    return []


# ---------------------------------------------------------------- thumbnails
def to_8bit(image):
    """A 16-bit / float image -> 8-bit grey, stretched from its own min to max so a dim image stays visible."""
    image = image.convert("I") if image.mode != "F" else image
    low, high = image.getextrema()
    scale = 255.0 / (high - low) if high > low else 1.0
    return image.point(lambda v: v * scale - low * scale).convert("L")


def make_thumb(path: Path, cache_dir: Path):
    """-> JPEG bytes (cached on disk by path + mtime + size), or None when the file can't be read or is too large."""
    from PIL import Image
    try:
        st = path.stat()
        stamp = f"{path}|{st.st_mtime_ns}|{st.st_size}|{THUMB_PX}"
        key = hashlib.sha256(stamp.encode("utf-8", "replace")).hexdigest()[:40]
        cache_file = cache_dir / key[:2] / f"{key}.jpg"
        try:
            return cache_file.read_bytes()
        except OSError:
            pass
        Image.MAX_IMAGE_PIXELS = THUMB_MAX_PIXELS
        with _decoders:                                  # at most THUMB_DECODERS decodes at once
            with Image.open(path) as image:
                if image.size[0] * image.size[1] > THUMB_MAX_PIXELS:
                    return None
                image.draft("RGB", (THUMB_PX * 2, THUMB_PX * 2))
                if image.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                    image = to_8bit(image)
                elif image.mode not in ("L", "RGB"):
                    image = image.convert("RGB")
                image.thumbnail((THUMB_PX, THUMB_PX))
                buffer = io.BytesIO()
                image.save(buffer, "JPEG", quality=80, optimize=True)
        data = buffer.getvalue()
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_name(cache_file.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, cache_file)
        except OSError:
            pass                                         # a cache that can't be written still serves the image
        return data
    except Exception:                                    # unreadable / too large / not an image
        return None


# ---------------------------------------------------------------- HTML
def chip(text, tone="") -> str:
    cls = {"prod": " cj-data-t-prod", "warn": " cj-data-t-warn", "acc": " cj-data-t-acc"}.get(tone, "")
    return f'<span class="cj-data-chip{cls}">{esc(str(text))}</span>'


def render_rows(rows) -> str:
    if not rows:
        return '<p class="cj-mut cj-data-empty">Nothing here yet.</p>'
    items = []
    for r in rows:
        tags = "".join(chip(t, tone) for t, tone in r["tags"])
        sub = f'<span class="cj-mut">{esc(r["sub"])}</span>' if r["sub"] else ""
        items.append(f'<li class="cj-data-row"><span class="cj-data-top"><a href="{esc(r["url"])}">'
                     f'{esc(r["title"])}</a>{tags}</span>{sub}</li>')
    return '<ul class="cj-data-rows">' + "".join(items) + "</ul>"


def search_form(query="") -> str:
    return ('<form class="cj-data-search" role="search" method="get" action="/data/search">'
            '<input type="search" name="q" maxlength="100" placeholder="Search by id or sample label" '
            f'aria-label="Search the catalog" value="{esc(query, quote=True)}"><button type="submit">Search</button>'
            '</form>')


def banner(error) -> str:
    if not error:
        return ""
    return ('<p class="cj-data-banner" role="status">The catalog could not be read just now, so the lists below may '
            f'be empty. ({esc(error)})</p>')


def breadcrumb(*parts) -> str:
    return '<p class="cj-data-crumb cj-mut">' + " / ".join(
        f'<a href="{esc(href)}">{esc(text)}</a>' for text, href in parts) + "</p>"


def card_section(title, count, rows_html, extra_id="") -> str:
    return (f'<section class="cj-card cj-data-card"{extra_id}><h2><span>{esc(title)}</span>'
            f'<span class="cj-data-n">{count}</span></h2>{rows_html}</section>')


def render_overview(ov, error) -> str:
    cards = []
    for c in ov["cards"]:
        more = (f'<p class="cj-data-more"><a href="{esc(c["all"])}">See all {c["count"]}</a></p>'
                if c["count"] > len(c["rows"]) else "")
        cards.append(card_section(c["title"], c["count"], render_rows(c["rows"]) + more,
                                  f' id="cj-data-c-{esc(c["kind"])}"'))
    grids = ov["counts"].get("grids", 0)
    return ('<h1>Data</h1><p class="cj-mut cj-data-sub">What data exists: imaging sessions, image sets, labels, '
            f'datasets and models. <a href="{PREFIX}/list/grids">{plural(grids, "grid")}</a> are listed under '
            'their sessions.</p>' + banner(error) + search_form()
            + '<div class="cj-data-cards">' + "".join(cards) + "</div>"
            + '<p class="cj-mut cj-data-foot">Read only: nothing here changes any data.</p>')


def render_search(result, error) -> str:
    if not result["q"]:
        body = '<p class="cj-mut">Type an id or a sample label.</p>'
    elif not result["groups"]:
        body = f'<p class="cj-mut">Nothing found for &ldquo;{esc(result["q"])}&rdquo;.</p>'
    else:
        body = "".join(card_section(g["title"], g["count"], render_rows(g["rows"])) for g in result["groups"])
    return (breadcrumb(("Data", PREFIX + "/")) + "<h1>Search</h1>" + banner(error) + search_form(result["q"])
            + '<div class="cj-data-cards">' + body + "</div>")


def render_list(title, rows, error) -> str:
    return (breadcrumb(("Data", PREFIX + "/")) + f"<h1>{esc(title)}</h1>"
            f'<p class="cj-mut cj-data-sub">{len(rows)} in all.</p>' + banner(error)
            + '<div class="cj-card">' + render_rows(rows) + "</div>")


def render_fields(fields) -> str:
    if not fields:
        return ""
    items = []
    for label, value, href in fields:
        shown = f'<a href="{esc(href)}">{esc(value)}</a>' if href else esc(value)
        items.append(f"<div><dt>{esc(label)}</dt><dd>{shown}</dd></div>")
    return '<dl class="cj-data-kv">' + "".join(items) + "</dl>"


def render_thumbs(d, strip=False) -> str:
    if not d["thumbs"]:
        return '<p class="cj-mut cj-data-empty">' + esc(d.get("note") or "No images found for this yet.") + "</p>"
    n = d["n_images"]
    items = "".join(f'<li><img class="cj-data-img" src="{esc(t["url"])}" alt="Image {t["n"] + 1} of {n}" '
                    f'loading="lazy" decoding="async" width="{THUMB_PX}" height="{THUMB_PX}"></li>'
                    for t in d["thumbs"])
    out = f'<ul class="{"cj-data-strip" if strip else "cj-data-thumbs"}">{items}</ul>'
    if strip:
        if n > len(d["thumbs"]):
            out += (f'<p class="cj-data-more"><a href="{esc(entity_href(d["kind"], d["id"]))}?page=1">'
                    f'See all {plural(n, "image")}</a></p>')
    elif d["pages"] > 1:
        base = entity_href(d["kind"], d["id"])
        first, last = (d["page"] - 1) * PAGE_N + 1, min(n, d["page"] * PAGE_N)
        previous = (f'<a href="{esc(base)}?page={d["page"] - 1}" rel="prev">Previous</a>' if d["page"] > 1
                    else '<span class="cj-data-off">Previous</span>')
        following = (f'<a href="{esc(base)}?page={d["page"] + 1}" rel="next">Next</a>' if d["page"] < d["pages"]
                     else '<span class="cj-data-off">Next</span>')
        out += (f'<nav class="cj-data-pager" aria-label="Pages">{previous}'
                f'<span class="cj-mut">Showing {first}-{last} of {n}</span>{following}</nav>')
    return out


def render_metric_table(rows) -> str:
    if not rows:
        return ""
    return '<table class="cj-data-metrics"><tbody>' + "".join(
        f'<tr><th scope="row">{esc(name)}</th><td>{esc(value)}</td></tr>' for name, value in rows) + "</tbody></table>"


def render_session_extras(d) -> str:
    out = ['<h2 class="cj-sec">Grids</h2>']
    if d["grids"]:
        items = []
        for g in d["grids"]:
            slot_name = f'Slot {g["slot"]}'
            head = (f'<a href="{esc(entity_href("grid", g["grid"]))}">Grid {esc(g["grid"])}</a>' if g["grid"]
                    else esc(slot_name))
            if g["grid"]:
                where = slot_name
            elif not g["grid_id_recorded"]:
                where = "grid number not recorded"
            else:
                where = f'grid {g["grid_id_recorded"]} not in the list'
            count = f'{plural(g["n_images"], "image")} ({g["count_source"]})' if g["n_images"] else None
            items.append(f'<li><span class="cj-data-top"><b>{head}</b></span>'
                         f'<span class="cj-mut">{esc(join_bits(g["label"], g["grid_type"], count, where))}</span></li>')
        out.append('<ul class="cj-data-grids">' + "".join(items) + "</ul>")
    else:
        out.append('<p class="cj-mut">No grids recorded for this session.</p>')
    out.append(f'<h2 class="cj-sec">Images ({d["n_images"]} processed)</h2>'
               + render_thumbs(d, strip=d["view"] == "strip"))
    return "".join(out)


def render_model_extras(d) -> str:
    out = ['<h2 class="cj-sec">Scores</h2>']
    table = render_metric_table(d["metrics"])
    if table:
        out.append(f'<div class="cj-card">{table}</div>')
    if d["checks"]:
        items = []
        for c in d["checks"]:
            sub = join_bits(f"session {c['session']}" if c.get("session") else None,
                            plural(c["n"], "image") if c.get("n") is not None else None, c.get("status"))
            items.append(f'<li class="cj-data-row"><span>{esc(c["id"])}</span><span class="cj-mut">{esc(sub)}</span>'
                         + render_metric_table(c["metrics"]) + "</li>")
        out.append('<div class="cj-card"><h2>Checks on new images</h2><ul class="cj-data-rows">'
                   + "".join(items) + "</ul></div>")
    if not table and not d["checks"]:
        out.append('<p class="cj-mut">No scores are recorded in the registry for this model yet.</p>')
    return "".join(out)


def link_list(heading, items) -> str:
    return (f'<h2 class="cj-sec">{esc(heading)}</h2><p>'
            + ", ".join(f'<a href="{esc(m["url"])}">{esc(m["id"])}</a>' for m in items) + "</p>")


def render_detail(d, error) -> str:
    kind = d["kind"]
    list_kind = DETAIL_LIST[kind]
    card_cls = "cj-card" + (" cj-data-hl" if d.get("production") else "")
    out = [breadcrumb(("Data", PREFIX + "/"), (KIND_TITLES[list_kind], f"{PREFIX}/list/{list_kind}")),
           f'<div class="cj-data-top"><h1 class="cj-wrap">{esc(d["title"])}</h1>'
           + "".join(chip(t, tone) for t, tone in d["chips"]) + "</div>",
           banner(error), f'<div class="{card_cls}">{render_fields(d["fields"])}</div>']
    if d.get("review_url"):
        out.append(f'<p class="cj-data-more"><a href="{esc(d["review_url"])}">Open in review</a></p>')
    if d.get("gap"):
        out.append(f'<p class="cj-mut">Acquisition details are not available: {esc(d["gap"])}</p>')
    if kind == "session":
        out.append(render_session_extras(d))
    elif kind == "model":
        out.append(render_model_extras(d))
    else:
        if kind == "grid" and d["sessions"]:
            out.append(link_list("Imaged in", [{"id": f"Session {s['id']}", "url": s["url"]} for s in d["sessions"]]))
        if kind == "dataset" and d["models"]:
            out.append(link_list("Models trained on it", d["models"]))
        if kind == "labels" and d["datasets"]:
            out.append(link_list("Datasets built from it", d["datasets"]))
        heading = "Example images" if kind == "dataset" else "Images"
        out.append(f'<h2 class="cj-sec">{heading} ({d["n_images"]})</h2>' + render_thumbs(d))
    return "".join(out)


def page_body(content: str) -> str:
    """data.html with the content put in ONE replace (text inside the content is never re-scanned)."""
    return TEMPLATE.read_text(encoding="utf-8").replace("{{content}}", content, 1)


# ---------------------------------------------------------------- routes
def routes(ctx) -> list:
    def state(request: Request):
        records, error = read_registry(getattr(ctx.cfg, "registry", None))
        return with_disk_packages(records), error, Viewer(ctx.can_see, ctx.user(request))

    def requested_page(request: Request, default=None):
        value = request.query_params.get("page")
        if value is None:
            return default
        return int(value) if value.isdigit() and len(value) < 6 else 1

    def html_page(request, title, content, status=200):
        response = ctx.page(request, title, page_body(content), active="data")
        response.status_code = status
        return response

    def missing(request, what="That item"):
        return html_page(request, "Not found", breadcrumb(("Data", PREFIX + "/")) + "<h1>Not found</h1>"
                         f'<p class="cj-mut">{esc(what)} is not in the catalog, or you do not have access to it.</p>',
                         404)

    def index(request: Request):
        records, error, viewer = state(request)
        return html_page(request, "Data", render_overview(overview(records, viewer), error))

    def find(request: Request):
        records, error, viewer = state(request)
        return html_page(request, "Search",
                         render_search(search(records, request.query_params.get("q", ""), viewer), error))

    def listing(request: Request):
        kind = request.path_params["kind"]
        if kind not in LIST_KINDS:
            return missing(request, "That list")
        records, error, viewer = state(request)
        return html_page(request, KIND_TITLES[kind], render_list(KIND_TITLES[kind], rows_for(records, kind, viewer),
                                                                 error))

    def item(kind):
        def handler(request: Request):
            records, error, viewer = state(request)
            page = requested_page(request, None if kind == "session" else 1)
            d = detail(records, kind, request.path_params["eid"], viewer, page)
            if d is None:
                return missing(request)
            return html_page(request, d["title"], render_detail(d, error))
        return handler

    def not_found_text():
        return Response("not found", status_code=404, media_type="text/plain")

    def thumb(request: Request):
        kind, eid, n = request.path_params["kind"], request.path_params["eid"], request.path_params["n"]
        if kind not in DETAIL_KINDS or not n.isdigit() or len(n) > 7 or not eid or len(eid) > 200 or "\x00" in eid:
            return not_found_text()
        records, _, viewer = state(request)
        files = thumbnail_sources(records, kind, eid, viewer)
        index_ = int(n)
        if index_ >= len(files) or not inside(files[index_], known_roots()):
            return not_found_text()
        data = make_thumb(files[index_], Path(ctx.cfg.state_dir) / "data_thumbs")
        if data is None:
            return not_found_text()
        current = request.query_params.get("v") == thumb_version(files[index_])
        response = Response(data, media_type="image/jpeg")
        response.headers["Cache-Control"] = "private, max-age=86400" if current else "private, max-age=60"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def api_overview(request: Request):
        records, error, viewer = state(request)
        return ctx.json(overview(records, viewer) | {"error": error})

    def api_search(request: Request):
        records, _, viewer = state(request)
        return ctx.json(search(records, request.query_params.get("q", ""), viewer))

    def api_list(request: Request):
        kind = request.path_params["kind"]
        if kind not in LIST_KINDS:
            return ctx.json({"error": "not found"}, 404)
        records, error, viewer = state(request)
        return ctx.json({"schema": SCHEMA, "kind": kind, "rows": rows_for(records, kind, viewer), "error": error})

    def api_item(request: Request):
        records, _, viewer = state(request)
        kind = request.path_params["kind"]
        d = detail(records, kind, request.path_params["eid"], viewer,
                   requested_page(request, None if kind == "session" else 1))
        return ctx.json(d) if d is not None else ctx.json({"error": "not found"}, 404)

    found = [Route(PREFIX + "/", index, methods=["GET"]),
             Route(PREFIX + "/search", find, methods=["GET"]),
             Route(PREFIX + "/list/{kind}", listing, methods=["GET"])]
    found += [Route(PREFIX + f"/{k}/{{eid}}", item(k), methods=["GET"]) for k in DETAIL_KINDS]
    found += [Route(PREFIX + "/thumb/{kind}/{eid}/{n}.jpg", thumb, methods=["GET"]),
              Route(PREFIX + "/api/overview", api_overview, methods=["GET"]),
              Route(PREFIX + "/api/search", api_search, methods=["GET"]),
              Route(PREFIX + "/api/list/{kind}", api_list, methods=["GET"]),
              Route(PREFIX + "/api/{kind}/{eid}", api_item, methods=["GET"])]
    return found
