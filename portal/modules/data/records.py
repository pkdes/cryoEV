"""
The catalog's records: constants and small helpers, the Viewer, the caches and the registry read, the
data folders and path checks, the images on disk, image-set visibility, list rows and search.

Part of the Data module (portal/modules/data/__init__.py has the rules); moved from data.py unchanged.
"""
from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import quote

from portal import names as NM
from portal import titles as TT
from portal import upload_sessions as US


PREFIX = "/data"

HERE = Path(__file__).resolve().parent.parent       # portal/modules (this file is one level down, in data/)
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
            "prediction_set", "inference_check", "run", "pull", "roboflow_image")
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


def lineage_href(kind: str, entity_id: str) -> str:
    return f"{PREFIX}/lineage/{kind}/{quote(str(entity_id), safe='')}/"


def short_date(s) -> str:
    return s[:10] if isinstance(s, str) else ""


def grid_sort_key(gid):
    return (int(gid), gid) if str(gid).isdigit() else (-1, str(gid))


def slot_sort_key(slot):
    return (0, int(slot), "") if str(slot).isdigit() else (1, 0, str(slot))


class Viewer:
    """Who is asking: may_see(owner) is ctx.can_see, so an unknown owner (None) is for admins only."""

    def __init__(self, can_see, user, cfg=None):
        self.can_see, self.user, self.cfg = can_see, user, cfg     # cfg: the portal's (conversions); None = none

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


def _actor_module():
    """data_utils/actor.py: label() is the one "claude for pujan" display form."""
    from data_utils import actor
    return actor


def who_label(who, via=None) -> str:
    """"claude for pujan" when an agent acted (via), else the person; "" when neither is known. Callers mask a
    hidden owner BEFORE calling and escape the result."""
    return _actor_module().label(_text(who), _text(via))


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


def upload_images(record: dict, viewer) -> list:
    """An upload session's images from the conversions this viewer may see (montages, then tiles); none without the
    portal's cfg. Cached FILES_TTL_S per session and viewer."""
    if viewer is None or getattr(viewer, "cfg", None) is None or not US.is_upload_session(record):
        return []
    user = viewer.user or {}
    key = ("upload-images", str(record.get("id")), str(user.get("id")), str(user.get("role")),
           json.dumps([c.get("id") for c in US.conversions_of(record)]))
    return cached(_files_cache, key, lambda: US.conversion_images(viewer.cfg, record, user, viewer.can_see))


def session_files(record: dict, viewer=None) -> list:
    """The session's images: the lab's processed folders first, else (an upload session) its visible conversions'."""
    return session_images(record["id"]) or upload_images(record, viewer)


def upload_roots(cfg) -> list:
    """The conversion roots as real paths in the OS's comparison case (thumbnails of upload sessions)."""
    return [os.path.normcase(os.path.realpath(r)) for r in US.conversion_roots(cfg)] if cfg is not None else []


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
                                            or str(record.get("id", "")).startswith("pred_")
                                            or str(record.get("parent") or "").startswith("pred_")))  # a part of one


def set_owner(entity: str, record: dict):
    """Owner id of a private set: the registry record, else the package's review_package.json, else the prediction
    set's request.json (for a part of a prediction package: its parent's set); None when none says."""
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
            or (set_id[len("pred_"):] if set_id.startswith("pred_") else None)
            or (str(record.get("parent"))[len("pred_"):] if str(record.get("parent") or "").startswith("pred_")
                else None))
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


def private_stub(ref_id, hint=None):
    """(entity, stub record) when an id is of a private kind (prediction set / prediction package), else None. Used
    to test visibility of an id that has no registry record, the way detail_labels does."""
    ref_id = str(ref_id)
    if PREDICTION_SET_ID.match(ref_id) or str(hint) == "prediction_set":
        return "prediction_set", {"id": ref_id}
    if PREDICTION_PACKAGE_ID.match(ref_id) or ref_id.startswith("pred_"):
        return "review_package", {"id": ref_id, "kind": "prediction"}
    return None


def ref_hidden(records, viewer, ref_id, hint=None) -> bool:
    """True when `ref_id` names a private set this viewer may not see (recorded or not): such an id must never be
    quoted. Fails closed: an owner nobody knows means admins only."""
    if not _text(ref_id):
        return False
    entity, record = find_set(records, ref_id)
    if record is not None:
        return not set_is_visible(viewer, entity, record)
    stub = private_stub(ref_id, hint)
    return bool(stub) and not set_is_visible(viewer, *stub)


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
    """Sample labels in slot order: the details' display names, else the lab's, else an upload's."""
    items = sorted(US.effective_labels(record).items(), key=lambda kv: slot_sort_key(kv[0]))
    return [v for _, v in items if _text(v)]


def image_count_of(record):
    n = record.get("n_images")
    if n is None and isinstance(record.get("counts"), dict):
        n = sum(_number(_dict(v).get("images")) or 0 for v in record["counts"].values())
    return n


def package_meta(records, package_id) -> dict:
    """The package's review_package.json (read-only, cached like the other file reads); {} when not on disk."""
    def compute():
        folder = package_folder(package_id, records["review_package"].get(package_id) or {})
        return read_small_json(folder / "review_package.json") if folder else {}
    return cached(_files_cache, ("pkgmeta", str(package_id)), compute)


def pred_display_name(package_id) -> str:
    """The prediction set's display name for pred_<ps> (callers apply the privacy rule first)."""
    root, ps = predict_folder(), str(package_id)[len("pred_"):]
    req = read_small_json(root / ps / "request.json") if root is not None and PREDICTION_SET_ID.match(ps) else {}
    return NM.display_name(req) if req else ""


def package_visible(records, viewer, package_id) -> bool:
    record = {**(records["review_package"].get(package_id) or {}), "id": package_id}
    return set_is_visible(viewer, "review_package", record)


def label_title(records, label_id, viewer) -> str:
    """Readable title; a review package this viewer may not see is left out of it (never named)."""
    return TT.label_title(records, label_id, lambda pid: package_meta(records, pid),
                          pkg_visible=lambda pid: package_visible(records, viewer, pid))


def set_title(records, entity, record) -> str:
    """Readable title of an image set (review packages by kind; other sets keep their id). Call only for sets the
    viewer may see."""
    if entity != "review_package":
        return record["id"]
    return TT.package_title(records, record["id"], package_meta(records, record["id"]), pred_display_name)


def make_row(kind, entity_id, title, sub, tags=(), highlight=False) -> dict:
    quoted = quote(str(entity_id), safe="")
    return {"kind": kind, "id": entity_id, "title": title, "sub": sub, "url": entity_href(kind, entity_id),
            "api": f"{PREFIX}/api/{kind}/{quoted}", "lineage": lineage_href(kind, entity_id), "tags": [list(t) for t in tags], "hl": highlight}


def status_tags(record) -> list:
    return [(status_word(record["status"]), "")] if _text(record.get("status")) else []


def session_tags(record) -> list:
    """Status, plus "Uploaded" for a session with a portal raw upload."""
    return status_tags(record) + ([("Uploaded", "acc")] if US.is_upload_session(record) else [])


def join_bits(*bits) -> str:
    return " · ".join(str(b) for b in bits if b)


def session_rows(records) -> list:
    rows = []
    for sid, record in sorted(records["session"].items(), reverse=True):
        n = record.get("n_imaging")
        labels = sample_labels(record)
        label_text = ", ".join(labels[:3]) + (f" +{len(labels) - 3} more" if len(labels) > 3 else "")
        acq = record if US.has_acquisition(record) else {**record, **US.upload_acquisition(record)}
        sub = join_bits(short_date(record.get("date")), _text(acq.get("instrument")), magnification_text(acq),
                        plural(n, "imaging file") if _number(n) is not None else None, label_text)
        rows.append(make_row("session", sid, f"Session {sid}", sub, session_tags(record)))
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
        rows.append(make_row("images", record["id"], set_title(records, entity, record), sub))
    return rows


def label_rows(records, viewer) -> list:
    rows = []
    for record in sorted(records["label_version"].values(), key=lambda r: (_when(r), r["id"]), reverse=True):
        n = record.get("n_images_applied")
        sub = join_bits(review_note(record)[0], plural(n, "image") + " changed" if n is not None else None,
                        short_date(_when(record)))
        rows.append(make_row("labels", record["id"], label_title(records, record["id"], viewer), sub, status_tags(record)))
    return rows


def class_names(record) -> list:
    return [str(c) for c in _list(record.get("classes")) if c not in (None, "")]


def classes_text(record):
    """'3 classes: EV, non-EV, Undetermined' (a one-line row subtitle), or None."""
    names = class_names(record)
    if not names:
        return None
    return f"{len(names)} {'class' if len(names) == 1 else 'classes'}: {', '.join(names)}"


def class_mapping(record) -> dict | None:
    """A dataset's class_map (source label class -> training class, None = dropped) as one line per training class,
    in the dataset's class order: {"lines": [[training class, [source classes]]], "dropped": [source classes]}."""
    cmap = _dict(record.get("class_map"))
    if not cmap:
        return None
    order = class_names(record)
    sources: dict = {}
    dropped = []
    for src, dst in cmap.items():
        if dst in (None, ""):
            dropped.append(str(src))
        else:
            sources.setdefault(str(dst), []).append(str(src))
    targets = [c for c in order if c in sources] + [c for c in sources if c not in order]
    return {"lines": [[t, sources[t]] for t in targets], "dropped": dropped}


def dataset_rows(records) -> list:
    rows = []
    for record in sorted(records["dataset"].values(), key=lambda r: (_when(r), r["id"]), reverse=True):
        sub = join_bits(split_counts(record), classes_text(record), short_date(_when(record)))
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
    if kind == "labels":
        return label_rows(records, viewer)
    builder = {"sessions": session_rows, "grids": grid_rows, "datasets": dataset_rows, "models": model_rows}[kind]
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
