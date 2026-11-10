"""
Data module (CryoEV portal, read only): a plain catalog of WHAT DATA EXISTS, from the registry (sessions, grids, image
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
    GET /data/lineage/<kind>/<id>/      the chain back to the raw imaging sessions (model, training run, dataset,
                                        label version / Roboflow export, image sets, grids, session), with each
                                        step's key facts and a few thumbnails; <kind> is one of the detail kinds
    GET /data/api/overview | search?q= | list/<kind> | <kind>/<id>[?page=N]    the same information as JSON
                                        (schema cj_data/1; other modules reuse it)
    GET /data/api/lineage/<kind>/<id>   the lineage as a JSON tree (schema cj_data/1)
    GET /data/lineage/graph/[?focus=<entity>|<id>]
                                        the Lab graph: History's lineage graph (label sets -> datasets -> models ->
                                        review / revise) for the whole lab, drawn by /data/data.js
    GET /data/api/lineage/graph[?focus=<entity>|<id>]   its data as JSON (schema cj_data/1)
    GET /data/data.js                   the Lab graph's script (portal/static/data.js; no inline script)
    POST /data/session/<id>/details     the Session details form of an uploaded session (its uploader or the admin;
                                        portal/upload_sessions.save_details, only when CJ_UPLOAD_SESSIONS=on)

Rules:
  - Nothing here writes to a data folder. The only registry write is the Session details form above (a merge-append
    through portal/upload_sessions.py); the other write is the thumbnail cache in the portal's own state folder.
  - Upload sessions (registry `conversions`, round 7): the list shows an "Uploaded" chip; the session page shows the
    acquisition summary (the record's own, else the newest upload's), sample per grid (display names from the
    details first), grid ids and prep notes (every admin / advanced user), and links to the conversion page and its
    Gallery (/gallery/<cid>/) ONLY for viewers who may see that conversion (ctx.can_see on its convert.json owner),
    plus the analyses (prediction sets whose source session is this one) the viewer may see. Without lab images the
    thumbnails come from those visible conversions' montages / tiles, through the same cache and containment checks
    (the conversion folders' roots are added to the known roots for those files only).
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
  - Lineage: every object in the chain goes through the same visibility test as its own page (ctx.can_see for
    prediction sets and prediction packages). A hidden object is shown as "not visible to you" with no id, and
    nothing below it is followed. A parent the registry does not record is "unknown", never guessed. The walk is
    capped (depth and number of steps) and stops at a step it is already inside, so bad registry data cannot loop.
    Folder paths are shown to admins only. A Roboflow export with an image list (registry `lists/`, written by the
    importer) leads to its images' grids and sessions; a re-split shows them under the export it was made from only
    when the viewer may see that one and its list leads to exactly the same grids and sessions. The summary counts
    listed images (datasets and exports shown) that trace to no session, as a number only.
  - Lab graph: History's own reader and builders (registry/history_page.py read_registry, build_model ->
    graph_model, imported, never changed), run on a FILTERED registry: every prediction set, review package and
    Roboflow export the viewer may not see (set_is_visible on Data's own record, the rule above; one Data does not
    have is for admins only) is removed BEFORE the graph is built, and so is every job that quotes one (any value or
    key, or its own id; ids under 6 characters as whole words only) and any record whose id is a hidden id or its
    pred_ / _part<N> form. A part whose parent is a prediction package is private like its parent. A subtitle,
    status or edge role quoting a hidden id is blanked. So the fade-out and the node order never reflect a hidden record; a label version or
    dataset built from a hidden set stays and only that edge goes. Each node sends only its id (History's box
    text), kind, column, faded flag, part-of, subtitle, status and its Data Lineage address; edges are
    [from, to, role]. ?focus names a node to highlight; a key the viewer may not see highlights nothing.
  - Session / grid Graph (?focus=session|<id> or grid|<id> with scope=item; source_graph): the session, its grids,
    its upload conversions (only those the viewer may open), and what used their images, read from the same filtered
    registry (image records' session / grid; History's image lookup for datasets, label versions, review packages,
    jobs and pulls; exports' image lists; prediction sets' source session, their packages, and the conversion a set
    was run on), then the lab graph's own edges downstream through item_graph. A hidden item is not in the filtered
    graph, so it is never linked or counted.
  - Plain words; every value is escaped; no script on the pages except the Lab graph's file /data/data.js.
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

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route

from portal import upload_sessions as US

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
    slots = sorted(set(_dict(record.get("sample_labels"))) | set(_dict(record.get("session_grids")))
                   | set(US.session_slots(record)), key=slot_sort_key)
    labels = US.effective_labels(record)
    typed = _dict(_dict(record.get("details")).get("grid_ids"))
    registry_counts: dict = {}
    for image in records["image"].values():
        if image.get("session") == sid and image.get("session_grid") is not None:
            slot = str(image["session_grid"])
            registry_counts[slot] = registry_counts.get(slot, 0) + 1
    uploaded: dict = {}
    for c in US.conversions_of(record):
        for slot, n in _dict(c.get("tiles_per_grid")).items():
            if _number(n) is not None:
                uploaded[str(slot)] = uploaded.get(str(slot), 0) + int(n)
    cards = []
    for slot in slots:
        grid_id = _dict(record.get("session_grids")).get(slot)
        grid = records["grid"].get(grid_id) if grid_id else None
        processed = len(session_images(sid, f"grid{slot}_".lower())) if files else 0
        n, source = ((processed, "processed files") if processed else
                     (registry_counts[slot], "registry records") if registry_counts.get(slot) else
                     (uploaded[slot], "uploaded tiles") if uploaded.get(slot) else (0, "registry records"))
        cards.append({"slot": slot, "label": _text(labels.get(slot)) or (grid or {}).get("sample_label"),
                      "grid": grid_id if grid else None, "grid_id_recorded": grid_id,
                      "grid_id_typed": _text(typed.get(slot)),
                      "grid_type": (grid or {}).get("grid_type"),
                      "n_images": n, "count_source": source})
    return cards


def session_analyses(records, sid, viewer) -> list:
    """Analyses (prediction sets) whose source session is `sid` that the viewer may see: the registry's records,
    plus the Predict folder's request.json files (cached FILES_TTL_S per viewer). [{id, title, url, status}]"""
    found = {}
    for record in records["prediction_set"].values():
        if _dict(record.get("source")).get("session") == sid and set_is_visible(viewer, "prediction_set", record):
            found[record["id"]] = {"id": record["id"], "title": _text(record.get("name")) or record["id"],
                                   "url": f"/predict/{quote(record['id'], safe='')}/",
                                   "status": status_word(record["status"]) if _text(record.get("status")) else None}
    cfg = getattr(viewer, "cfg", None)
    if cfg is not None:
        user = viewer.user or {}

        def scan():
            try:
                from portal.modules import predict as P
                return [(d.name, req, st) for d, req, st in P.list_sets(cfg, user, viewer.can_see)]
            except Exception:                            # noqa: BLE001 - no predict root: the registry's only
                return []
        for ps, req, st in cached(_files_cache, ("analyses", str(user.get("id")), str(user.get("role"))), scan):
            src = _dict(req.get("source"))
            if src.get("session") == sid and ps not in found:
                found[ps] = {"id": ps, "title": _text(req.get("name")) or ps, "url": f"/predict/{quote(ps, safe='')}/",
                             "status": status_word(st["state"]) if _text(_dict(st).get("state")) else None}
    return sorted(found.values(), key=lambda a: a["id"], reverse=True)


def detail_session(records, record, page=None, viewer=None) -> dict:
    sid = record["id"]
    lab_files = session_images(sid)
    files = lab_files or upload_images(record, viewer)
    acq, from_upload = record, False
    if not US.has_acquisition(record) and US.upload_acquisition(record):
        acq, from_upload = {**record, **US.upload_acquisition(record)}, True
    elif US.is_upload_session(record) and not record.get("local_folder") and US.has_acquisition(record):
        from_upload = True                               # a session the portal made: its summary is the upload's
    fields = acquisition_fields(acq, files)
    if from_upload:
        fields.append(("Acquisition read from", "the uploaded raw files", None))
    upload = None
    if US.is_upload_session(record):
        det = _dict(record.get("details"))
        convs = (US.visible_conversions(viewer.cfg, record, viewer.user or {}, viewer.can_see)
                 if viewer is not None and viewer.cfg is not None else [])
        upload = {"conversions": [{"id": c["id"], "url": c["href"], "gallery": c["gallery"], "n_tiles": c["n_tiles"]}
                                  for c in convs],
                  "details": {"grid_ids": _dict(det.get("grid_ids")), "sample_display": _dict(det.get("sample_display")),
                              "prep_notes": _text(det.get("prep_notes")), "by": _text(det.get("by")),
                              "at": _text(det.get("at"))},
                  "can_edit": bool(viewer is not None and US.enabled(viewer.cfg) and US.may_edit(viewer.user, record))}
    detail = make_detail(
        "session", sid, f"Session {sid}", session_tags(record), fields,
        grids=session_grid_cards(records, record, files),
        acquisition={k: acq.get(k) for k in (
            "instrument", "magnification", "pixel_size_a", "defocus_um", "dose", "voltage_kv", "binning",
            "acq_dates", "n_mdoc", "n_imaging", "n_other", "other_magnification")},
        gap=None if from_upload else _text(record.get("acquisition_gap")), upload=upload,
        images_word="processed" if lab_files or not files else "from the upload",
        analyses=session_analyses(records, sid, viewer) if viewer is not None else [])
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


def hide_private(records, viewer, ref_id, hint=None):
    """An id to quote in a field, or "Not visible to you" when it names a private set this viewer may not see."""
    return "Not visible to you" if ref_hidden(records, viewer, ref_id, hint) else ref_id


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
        field("Starts from", hide_private(records, viewer, _text(record.get("base"))
                                          or _text(record.get("derived_from")), record.get("base_kind"))),
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
        field("Starts from", hide_private(records, viewer, _text(record.get("base")))),
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
    source_hidden = bool(source_id) and ref_hidden(records, viewer, source_id, source.get("kind"))
    fields = present(
        field("Status", status_word(record["status"]) if _text(record.get("status")) else None),
        field("Created", short_date(_when(record))), field("Splits (images)", split_counts(record)),
        field("Classes", ", ".join(map(str, _list(record.get("classes"))))),
        field("Merged classes", ", ".join(f"{k} as {v}" for k, v in class_map.items())),
        field("Label version", label, entity_href("labels", label) if label in records["label_version"] else None),
        field("Source", ("Not visible to you" if source_hidden else source_id or _text(source.get("kind")))
              if source else None,
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
        return detail_session(records, record, page, viewer) if record else None
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
        record = records["session"].get(entity_id)
        return session_files(record, viewer) if record else []
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


# ---------------------------------------------------------------- lineage (the chain back to the raw sessions)
LINEAGE_MAX_DEPTH = 10      # steps from the item you asked about
LINEAGE_MAX_NODES = 80      # steps in one lineage, however the registry links them
LINEAGE_FANOUT = 8          # grids / sessions listed under one step before "and N more"
LINEAGE_THUMBS = 4
KIND_TO_NODE = {"session": "session", "grid": "grid", "images": "images", "labels": "label", "dataset": "dataset",
                "model": "model"}
STEP_NAMES = {"model": "Model", "run": "Training run", "dataset": "Dataset", "label": "Label version",
              "export": "Roboflow export", "pull": "Roboflow download", "images": "Image set", "grid": "Grid",
              "session": "Imaging session", "source": "Source", "hidden": "Step"}
NODE_THUMB_KIND = {"session": "session", "grid": "grid", "images": "images", "label": "labels", "dataset": "dataset"}
HINT_TO_NODE = {"roboflow_export": "export", "export": "export", "label_version": "label", "lv": "label",
                "pull": "pull", "review_package": "images", "package": "images", "prediction_set": "images",
                "dataset": "dataset", "model": "model", "session": "session", "run": "run", "grid": "grid"}
NODE_RECORDS = {"model": "model", "run": "run", "dataset": "dataset", "label": "label_version",
                "export": "roboflow_export", "pull": "pull", "grid": "grid", "session": "session"}


def lineage_href(kind: str, entity_id: str) -> str:
    return f"{PREFIX}/lineage/{kind}/{quote(str(entity_id), safe='')}/"


def locate(records, ref_id, hint=None):
    """Which node type a registry id is: the hinted one if it is there, else the first record that has the id; None
    when no record has it (never guessed)."""
    def has(ntype):
        if ntype == "images":
            return find_set(records, ref_id)[1] is not None
        return ntype in NODE_RECORDS and ref_id in records[NODE_RECORDS[ntype]]
    first = HINT_TO_NODE.get(str(hint))
    for ntype in ([first] if first else []) + ["export", "label", "pull", "images", "dataset", "model", "run",
                                                        "grid", "session"]:
        if has(ntype):
            return ntype
    return None


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


def ref_spec(records, viewer, ref_id, hint=None):
    """A parent link for a recorded id: ('node', type, id) when something has it, ('hidden',) when it is a private
    set the viewer may not see, else ('unknown', type, why, id)."""
    if not _text(ref_id):
        return None
    ntype = locate(records, ref_id, hint)
    if ntype:
        return ("node", ntype, ref_id)
    if ref_hidden(records, viewer, ref_id, hint):
        return ("hidden",)
    return ("unknown", HINT_TO_NODE.get(str(hint), "source"),
            "It is recorded here but is not in the registry.", ref_id)


def unknown_spec(ntype, why):
    return ("unknown", ntype, why, None)


def input_specs(records, viewer, record, seen) -> list:
    """Parent links from the record's generic `inputs` list (child -> parent), skipping ids already linked."""
    specs = []
    for item in _list(record.get("inputs")):
        item = _dict(item)
        ref_id = _text(item.get("id"))
        if ref_id and ref_id not in seen and item.get("role") not in ("init_weights", "replaces"):
            seen.add(ref_id)
            specs.append(ref_spec(records, viewer, ref_id, item.get("entity")))
    return capped(specs, lambda spec: spec, "sources")


def image_sources(records, match) -> dict:
    """Grids and sessions of the registry's image records that `match` picks (those naming a session):
    {"grids": {grid: n}, "sessions": {session: n} for images with no grid, "all_sessions": set, "n": traced}."""
    grids: dict = {}
    sessions: dict = {}
    seen_sessions = set()
    n = 0
    for image in records["image"].values():
        if not match(image) or not _text(image.get("session")):
            continue
        n += 1
        sid, slot = image["session"], image.get("session_grid")
        seen_sessions.add(sid)
        grid = _text(image.get("grid")) or _dict(_dict(records["session"].get(sid)).get("session_grids")).get(
            str(slot) if slot is not None else "")
        if grid:
            grids[grid] = grids.get(grid, 0) + 1
        else:
            sessions[sid] = sessions.get(sid, 0) + 1
    return {"grids": grids, "sessions": sessions, "all_sessions": seen_sessions, "n": n}


def pull_sources(records, pull_id) -> tuple:
    """Grids and sessions of a download's images, from the registry only: the Roboflow image records first seen in
    this download, matched by image name to the image records that name a session. -> (grid counts, session counts
    for images with no grid)."""
    names = {r.get("name") for r in records["roboflow_image"].values() if r.get("first_seen_pull") == pull_id}
    names.discard(None)
    found = image_sources(records, lambda image: image.get("rf_name") in names)
    return found["grids"], found["sessions"]


def read_image_list(root, record):
    """The stems in a dataset's image list (first column of registry/<image_list.file>), or None when there is no
    readable list. The file must sit under the registry folder."""
    name = _text(_dict(record.get("image_list")).get("file"))
    if not name or root is None:
        return None
    path = Path(root) / name
    if path.suffix.lower() not in (".tsv", ".txt", ".csv") or not is_contained(path, root):
        return None
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return None
    return [ln.split("\t")[0].strip() for ln in lines if ln.strip()]


def list_trace(records, root, record):
    """A dataset's / export's image list -> (listed stems, image_sources result, stems with no session in the
    registry), or None when there is no readable list."""
    stems = read_image_list(root, record)
    if stems is None:
        return None
    wanted = set(stems)
    found = image_sources(records, lambda image: image.get("id") in wanted)
    untraced = {s for s in wanted if not _text(_dict(records["image"].get(s)).get("session"))}
    return wanted, found, untraced


def traced_note(wanted, found) -> str:
    sessions = sorted(found["all_sessions"])
    shown = ", ".join(sessions[:6]) + (f" and {len(sessions) - 6} more" if len(sessions) > 6 else "")
    return (f"{found['n']} of {len(wanted)} images traced to sessions {shown}." if sessions else
            f"0 of {len(wanted)} images traced to a session.")


def capped(specs: list, make, label: str) -> list:
    """The first LINEAGE_FANOUT specs, then a ('more', n, label) marker."""
    out = [make(s) for s in specs[:LINEAGE_FANOUT]]
    if len(specs) > LINEAGE_FANOUT:
        out.append(("more", len(specs) - LINEAGE_FANOUT, label))
    return out


class Lineage:
    """One walk over the registry for one viewer. Every object passes the viewer's visibility test before any of its
    facts are read."""

    def __init__(self, records, viewer, root=None):
        self.records, self.viewer, self.root, self.count = records, viewer, root, 0
        self.admin = _dict(viewer.user).get("role") == "admin"
        self.untraced = set()       # listed images (datasets / exports shown) with no session in the registry

    # ---- lookup and visibility
    def lookup(self, ntype, nid):
        """-> (ntype, record, entity) | ('hidden', None, None) | None when there is no such record."""
        if ntype == "images":
            entity, record = find_set(self.records, nid)
            if record is None:
                return None
            if not set_is_visible(self.viewer, entity, record):
                return ("hidden", None, None)
            return ("export", record, entity) if entity == "roboflow_export" else ("images", record, entity)
        record = self.records[NODE_RECORDS[ntype]].get(nid) if ntype in NODE_RECORDS else None
        if record and ntype == "export" and not set_is_visible(self.viewer, "roboflow_export", record):
            return ("hidden", None, None)
        return (ntype, record, None) if record else None

    # ---- the walk
    def node(self, ntype, nid, status="ok", title=None, **extra) -> dict:
        node = {"type": ntype, "step": STEP_NAMES[ntype], "status": status, "id": nid, "title": title or STEP_NAMES[ntype],
                "chips": [], "fields": [], "note": None, "url": None, "lineage": None, "thumbs": None,
                "parents": []}
        node.update(extra)
        self.count += 1                                     # every step shown counts toward the cap
        return node

    def walk(self, ntype, nid, path, depth) -> dict:
        found = self.lookup(ntype, nid)
        if found is None:
            if ref_hidden(self.records, self.viewer, nid):
                return self.hidden_node()
            return self.node(ntype, None, "unknown", note="Recorded as a source but not found in the registry.",
                             chips=[("Unknown", "warn")], recorded=nid)
        ntype, record, entity = found
        if ntype == "hidden":
            return self.hidden_node()
        key = (ntype, nid)
        if key in path:                                     # bad data: this step leads back to itself
            return self.node(ntype, nid, "loop", title=self.title(ntype, nid), chips=[("Loop in the registry", "warn")],
                             note="The registry links this step back to one above it, so the chain stops here.")
        if depth >= LINEAGE_MAX_DEPTH or self.count >= LINEAGE_MAX_NODES:
            return self.node(ntype, nid, "cut", title=self.title(ntype, nid),
                             note="Not shown: limit reached." + (" Use \"Lineage from here\" to continue."
                                                                  if self.lineage_url(ntype, nid, entity) else ""),
                             url=self.url(ntype, nid, entity), lineage=self.lineage_url(ntype, nid, entity))
        title, chips, fields, specs, note = self.describe(ntype, nid, record, entity)
        specs = [sp for sp in specs if sp]
        room = max(0, LINEAGE_MAX_NODES - self.count - 1)
        if len(specs) > room + 1:                           # keep the whole tree under the cap
            specs = specs[:room] + [("more", len(specs) - room, "steps")]
        node = self.node(ntype, nid, title=title, chips=chips, fields=fields, note=note,
                         url=self.url(ntype, nid, entity), lineage=self.lineage_url(ntype, nid, entity))
        if entity:
            node["set_kind"] = SET_KIND_NAMES.get(entity)
        node["thumbs"] = self.thumbs(ntype, nid, record, depth)
        child_path = path | {key}
        node["parents"] = [self.child(spec, child_path, depth + 1) for spec in specs]
        return node

    def hidden_node(self) -> dict:
        return self.node("hidden", None, "hidden", "Not visible to you", chips=[("Not visible to you", "warn")],
                         note="This step exists but belongs to someone else, so its details are not shown.")

    def child(self, spec, path, depth) -> dict:
        if spec[0] == "hidden":
            return self.hidden_node()
        if spec[0] == "node":
            return self.walk(spec[1], spec[2], path, depth)
        if spec[0] == "more":
            return self.node("source", None, "more", f"and {spec[1]} more {spec[2]}", note=None)
        _, ntype, why, recorded = spec
        return self.node(ntype, None, "unknown", note=why, chips=[("Unknown", "warn")], recorded=recorded)

    # ---- addresses
    @staticmethod
    def catalog_kind(ntype, entity):
        return {"label": "labels", "images": "images", "export": "images", "pull": None, "run": None}.get(ntype, ntype)

    def url(self, ntype, nid, entity=None):
        kind = self.catalog_kind(ntype, entity)
        return entity_href(kind, nid) if kind else None

    def lineage_url(self, ntype, nid, entity=None):
        kind = self.catalog_kind(ntype, entity)
        return lineage_href(kind, nid) if kind else None

    @staticmethod
    def title(ntype, nid):
        return f"Session {nid}" if ntype == "session" else f"Grid {nid}" if ntype == "grid" else str(nid)

    # ---- thumbnails (only where the catalog already has them)
    def thumbs(self, ntype, nid, record, depth):
        kind = NODE_THUMB_KIND.get(ntype)
        if kind is None or (ntype == "grid" and depth > 0):
            return None
        if ntype == "session":
            files = session_images(nid)
        elif ntype == "grid":
            files = grid_images(self.records, nid)
        elif ntype == "images":
            files = set_images(self.records, nid, self.viewer)
        elif ntype == "label":
            package = _text(record.get("pkg"))
            files = package_images_for(self.records, package, self.viewer) if package else []
        else:
            files = dataset_images(self.records, record, self.viewer)
        if not files:
            return None
        part = thumbnails(kind, nid, files, indexes=evenly_spaced(len(files), LINEAGE_THUMBS))
        return {"kind": kind, "id": nid, **part}

    # ---- each kind: title, chips, key facts, parent links, note
    def link(self, ntype, nid, entity=None):
        """Catalog address for a field value, only when that object exists and may be seen."""
        found = self.lookup(ntype, nid) if nid else None
        return self.url(ntype, nid, entity) if found and found[0] != "hidden" else None

    def describe(self, ntype, nid, record, entity):
        fn = getattr(self, "describe_" + ntype)
        return fn(nid, record, entity) if ntype == "images" else fn(nid, record)

    def describe_model(self, nid, rec):
        run, dataset = _text(rec.get("run")), _text(rec.get("dataset"))
        run_ok = run in self.records["run"]
        specs = []
        if run_ok and _text(self.records["run"][run].get("dataset")):
            specs.append(("node", "run", run))
        else:
            specs.append(("node", "run", run) if run_ok else (
                ("unknown", "run", "It is recorded here but is not in the registry.", run) if run else
                unknown_spec("run", "The registry does not say which training run made this model.")))
            specs.append(ref_spec(self.records, self.viewer, dataset, "dataset") or
                         unknown_spec("dataset", "The registry does not say which dataset this model was trained on."))
        specs += input_specs(self.records, self.viewer, rec, {nid, run, dataset})
        production = rec.get("status") == "production"
        fields = present(
            field("Status", status_word(rec["status"]) if _text(rec.get("status")) else None),
            field("Classes", ", ".join(map(str, _list(rec.get("classes"))))),
            field("Image size", f"{rec['imgsz']} px" if rec.get("imgsz") else None),
            field("Started from", _text(rec.get("init_from")), self.link("model", _text(rec.get("init_from")))),
            field("Registered", short_date(rec.get("logged"))),
            field("Weights file", _text(rec.get("weights")) if self.admin else None))
        return nid, [("Production", "prod")] if production else status_tags(rec), fields, specs, None

    def describe_run(self, nid, rec):
        config, metrics = _dict(rec.get("config")), _dict(rec.get("metrics"))
        dataset = _text(rec.get("dataset"))
        specs = [ref_spec(self.records, self.viewer, dataset, "dataset") or
                 unknown_spec("dataset", "The registry does not say which dataset this run trained on.")]
        fields = present(
            field("Status", status_word(rec["status"]) if _text(rec.get("status")) else None),
            field("Model type", _text(config.get("arch"))), field("Epochs", str(config["epochs"]) if
                                                                   _number(config.get("epochs")) is not None else None),
            field("Image size", f"{config['imgsz']} px" if _number(config.get("imgsz")) is not None else None),
            field("Machine", _text(rec.get("machine"))),
            field("Started", (rec.get("started") or "").replace("T", " ")[:16]),
            field("Mask mAP50", format_number(metrics.get("mask_map50"), 3)
                  if _number(metrics.get("mask_map50")) is not None else None),
            field("Job bundle", _text(rec.get("job_bundle"))))
        gap = _text(rec.get("job_bundle_gap"))
        return nid, status_tags(rec), fields, specs, f"No job bundle: {gap}." if gap and not rec.get("job_bundle") else None

    def describe_dataset(self, nid, rec):
        label, source = _text(rec.get("label_version")), _dict(rec.get("source"))
        seen = {nid}
        specs = []
        for ref_id, hint in ((label, "label_version"), (_text(source.get("id")), source.get("kind"))):
            if ref_id and ref_id not in seen:
                seen.add(ref_id)
                specs.append(ref_spec(self.records, self.viewer, ref_id, hint))
        specs += input_specs(self.records, self.viewer, rec, seen)
        note = None
        trace = list_trace(self.records, self.root, rec)
        if trace and trace[0]:
            wanted, found, untraced = trace
            self.untraced |= untraced
            note = traced_note(wanted, found)
            specs += self.source_specs(found["grids"], found["sessions"])
        if not specs:
            specs = [unknown_spec("source", "The registry does not say what this dataset was built from.")]
        fields = present(
            field("Status", status_word(rec["status"]) if _text(rec.get("status")) else None),
            field("Images per split", split_counts(rec)),
            field("Classes", ", ".join(map(str, _list(rec.get("classes"))))),
            field("Created", short_date(_when(rec))))
        return nid, status_tags(rec), fields, specs, note

    def source_specs(self, grids, sessions) -> list:
        """Grid and session parents for traced images: grids by image count, then sessions with no recorded grid."""
        ordered = sorted(grids.items(), key=lambda kv: (-kv[1], grid_sort_key(kv[0])))
        specs = capped(ordered, lambda kv: ("node", "grid", kv[0]) if kv[0] in self.records["grid"] else
                       ("unknown", "grid", "The grid is recorded on its images but is not in the registry.", kv[0]),
                       "grids")
        by_count = sorted(sessions.items(), key=lambda kv: (-kv[1], kv[0]))
        return specs + capped(by_count, lambda kv: ("node", "session", kv[0]) if kv[0] in self.records["session"] else
                              ("unknown", "session", "The session is recorded on its images but is not in the "
                               "registry.", kv[0]), "sessions without a recorded grid")

    def describe_label(self, nid, rec):
        seen = {nid}
        specs = []
        base = _text(rec.get("base"))
        base_hint = next((_dict(i).get("entity") for i in _list(rec.get("inputs")) if _dict(i).get("id") == base), None)
        for ref_id, hint in ((base, base_hint), (_text(rec.get("pkg")), "review_package")):
            if ref_id and ref_id not in seen:
                seen.add(ref_id)
                specs.append(ref_spec(self.records, self.viewer, ref_id, hint))
        specs += input_specs(self.records, self.viewer, rec, seen)
        if not specs:
            specs = [unknown_spec("source", "The registry does not say what these labels were made from.")]
        note, tone = review_note(rec)
        replaces = _text(rec.get("supersedes"))
        fields = present(
            field("Status", status_word(rec["status"]) if _text(rec.get("status")) else None),
            field("Checking", note),
            field("Images changed", str(rec["n_images_applied"]) if rec.get("n_images_applied") is not None else None),
            field("Replaces", replaces, self.link("label", replaces)),
            field("Created", short_date(_when(rec))))
        return nid, [(note, tone)] + status_tags(rec), fields, specs, None

    def describe_export(self, nid, rec, entity=None):
        parent, label = _text(rec.get("derived_from")), _text(rec.get("label_version"))
        seen = {nid}
        specs = []
        for ref_id, hint in ((parent, "roboflow_export"), (label, "label_version")):
            if ref_id and ref_id not in seen:
                seen.add(ref_id)
                specs.append(ref_spec(self.records, self.viewer, ref_id, hint))
        specs += input_specs(self.records, self.viewer, rec, seen)
        note = None
        trace = list_trace(self.records, self.root, rec)
        if trace is not None:
            wanted, found, untraced = trace
            self.untraced |= untraced
            note = traced_note(wanted, found)
            if str(_dict(rec.get("image_list")).get("method", "")).startswith("tracking"):
                note += " The image list comes from the image tracker, which lists only the images it tracks."
            # a re-split shows its grids / sessions under the export it was made from only when that one is visible,
            # lists its images and they lead to exactly the same grids and sessions; otherwise it shows its own
            if not (parent and self.same_sources_as_parent(parent, found)):
                specs += self.source_specs(found["grids"], found["sessions"])
            if not specs:
                specs = [unknown_spec("session", "No image in this export is matched to an imaging session in the "
                                                 "registry.")]
        if not specs:
            specs = [unknown_spec("source", "The registry does not record which images went into this export.")]
        fields = present(field("Classes", ", ".join(map(str, _list(rec.get("classes"))))),
                         field("Images per split", split_counts(rec)),
                         field("Made from export", hide_private(self.records, self.viewer, parent, "roboflow_export")
                               if parent else None, self.link("images", parent) if parent else None))
        return nid, [("Roboflow export", "acc")], fields, specs, note

    def same_sources_as_parent(self, parent, found) -> bool:
        """True when the parent export is visible to this viewer, has a readable image list, and its images lead to
        the same grids and the same sessions (those with no recorded grid) as this export's."""
        got = self.lookup("export", parent)
        if not got or got[0] != "export":
            return False
        trace = list_trace(self.records, self.root, got[1])
        return trace is not None and (set(trace[1]["grids"]), set(trace[1]["sessions"])) == \
            (set(found["grids"]), set(found["sessions"]))

    def describe_pull(self, nid, rec):
        grids, sessions = pull_sources(self.records, nid)
        specs = self.source_specs(grids, sessions)
        note = ("Grids and sessions come from the registry's image records, matched by image name to the images "
                "first seen in this download.")
        if not specs:
            specs = [unknown_spec("session", "No image in this download is matched to an imaging session in the "
                                             "registry.")]
            note = None
        fields = present(field("Images", str(rec["n_images"]) if rec.get("n_images") is not None else None),
                         field("Annotations", str(rec["n_annotations"]) if rec.get("n_annotations") is not None
                               else None),
                         field("Downloaded", short_date(_when(rec) or rec.get("created"))),
                         field("Image group", _text(rec.get("image_set"))))
        return nid, [], fields, specs, note

    def describe_images(self, nid, rec, entity):
        specs, note = [], None
        if entity == "prediction_set":
            source = _dict(rec.get("source"))
            session = _text(source.get("session"))
            if session:
                specs.append(ref_spec(self.records, self.viewer, session, "session"))
            elif source.get("kind") == "upload":
                note = "Uploaded by its owner. It does not come from a recorded imaging session."
            else:
                specs.append(unknown_spec("session", "The registry does not say which images this was run on."))
        else:
            seen = {nid}
            base = _text(rec.get("base"))
            for ref_id, hint in ((_text(rec.get("prediction_set")), "prediction_set"), (_text(rec.get("parent")),
                                 "review_package"), (base, rec.get("base_kind"))):
                if ref_id and ref_id not in seen:
                    seen.add(ref_id)
                    specs.append(ref_spec(self.records, self.viewer, ref_id, hint))
            specs += input_specs(self.records, self.viewer, rec, seen)
            if not specs:
                specs = [unknown_spec("source", "The registry does not say what this package was made from.")]
        model, n = _text(rec.get("model")), image_count_of(rec)
        fields = present(
            field("Images", str(n) if n is not None else None),
            field("Status", status_word(rec["status"]) if _text(rec.get("status")) else None),
            field("Created", short_date(_when(rec))),
            field("Model", model, self.link("model", model) if model else None),
            field("Folder", _text(rec.get("path")) if self.admin and entity == "review_package" else None))
        return nid, [(SET_KIND_NAMES[entity], "acc")], fields, specs, note

    def describe_grid(self, nid, rec):
        sessions = sorted({sid for sid, s in self.records["session"].items()
                           if nid in _dict(s.get("session_grids")).values()}
                          | {im["session"] for im in self.records["image"].values()
                             if im.get("grid") == nid and im.get("session")})
        specs = capped(sessions, lambda sid: ("node", "session", sid) if sid in self.records["session"] else
                       ("unknown", "session", "The session is recorded here but is not in the registry.", sid),
                       "sessions")
        if not specs:
            specs = [unknown_spec("session", "No imaging session is recorded for this grid.")]
        fields = present(field("Sample", _text(rec.get("sample_label"))), field("Grid type", _text(rec.get("grid_type"))),
                         field("Prepared", short_date(rec.get("prep_date"))),
                         field("Replicate", str(rec["replicate"]) if rec.get("replicate") is not None else None))
        return f"Grid {nid}", [("Replaced", "warn")] if _text(rec.get("replaced_by")) else [], fields, specs, None

    def describe_session(self, nid, rec):
        files = session_images(nid)
        keep = ("Date", "Instrument", "Magnification", "Pixel size", "Defocus", "Dose", "Voltage")
        fields = [f for f in acquisition_fields(rec, files) if f[0] in keep]
        if self.admin:
            fields += present(field("Folder", _text(rec.get("local_folder"))))
        gap = _text(rec.get("acquisition_gap"))
        note = f"Acquisition details are not available: {gap}" if gap else None
        return f"Session {nid}", status_tags(rec), fields, [], note


def lineage_tree(records, viewer, kind, entity_id, root=None):
    """-> the lineage dict for a catalog item, or None when it is unknown or hidden from this viewer."""
    ntype = KIND_TO_NODE.get(kind)
    if ntype is None or not _text(entity_id):
        return None
    walker = Lineage(records, viewer, root)
    found = walker.lookup(ntype, entity_id)
    if found is None or found[0] == "hidden":
        return None
    root = walker.walk(ntype, entity_id, frozenset(), 0)
    counts = {"steps": 0, "known": 0, "unknown": 0, "hidden": 0, "cut": 0, "loop": 0}

    def tally(node):
        if node["status"] == "more":
            return
        counts["steps"] += 1
        counts["known" if node["status"] == "ok" else node["status"] if node["status"] in counts else "unknown"] += 1
        for parent in node["parents"]:
            tally(parent)
    tally(root)
    counts["untraced_images"] = len(walker.untraced)        # a number only: never the ids
    return {"schema": SCHEMA, "kind": kind, "id": entity_id, "title": root["title"], "summary": counts,
            "limits": {"depth": LINEAGE_MAX_DEPTH, "steps": LINEAGE_MAX_NODES}, "root": root}


# ---------------------------------------------------------------- Lab graph (History's lineage graph, filtered)
GRAPH_PAGE_KIND = {"roboflow_export": "images", "prediction_set": "images", "review_package": "images",
                   "label_version": "labels", "dataset": "dataset", "model": "model"}   # pulls and jobs: no page
GRAPH_SHORT = {"roboflow_export": "export", "review_package": "package", "label_version": "label version",
               "prediction_set": "predictions"}                 # the box's tag, as History's drawGraph SHORT
GRAPH_NODE_KEYS = ("title", "kind", "col", "hist", "part", "sub", "status", "url")   # everything a node sends
SCRIPT = HERE.parent / "static" / "data.js"                    # served as /data/data.js (app.py serves only /cj/*)
GRAPH_CACHE_N = 32                  # registry reads and built graphs kept (one graph per registry state per group)
GRAPH_SHORT_ID = 6                  # a hidden id shorter than this is matched as a whole word, never as a substring
_graph_cache: dict = {}             # key -> value, oldest first
_history_module_cache = None


def _history_module():
    """registry/history_page.py, loaded once by path (read only here: its reader and builders are reused)."""
    global _history_module_cache
    if _history_module_cache is None:
        spec = importlib.util.spec_from_file_location("cj_data_history_page", REPO / "registry" / "history_page.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _history_module_cache = mod
    return _history_module_cache


def _recall(key):
    with _lock:
        return _graph_cache.get(key)


def _remember(key, value):
    with _lock:
        _graph_cache.pop(key, None)
        _graph_cache[key] = value
        while len(_graph_cache) > GRAPH_CACHE_N:
            _graph_cache.pop(next(iter(_graph_cache)))


def files_signature(paths) -> tuple:
    """(path, mtime_ns, size) per file; a file that cannot be read counts as (path, None, None)."""
    out = []
    for p in paths:
        try:
            st = os.stat(p)
            out.append((str(p), st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((str(p), None, None))
    return tuple(out)


def history_registry(root) -> tuple:
    """History's own read of <root>/*.jsonl (plus the label versions' owner files it reads) -> ((signature, current,
    history, stats) or None, error or None). Read again only when one of those files changes (mtime_ns, size); a
    failed read is not kept. Records are shared between requests and never changed."""
    if root is None:
        return None, "no registry folder configured"
    if not Path(root).is_dir():
        return None, "registry folder not found"
    hp = _history_module()
    try:
        files = hp.input_files([Path(root)])
    except OSError as exc:
        return None, f"registry not readable ({type(exc).__name__})"
    sig = files_signature(files)                       # taken BEFORE the read: a write during it shows next time
    hit = _recall(("read", str(root)))
    if hit is not None and hit[0] == sig and files_signature(hit[1]) == hit[2]:
        return hit[3], None
    try:
        current, history, stats = hp.read_registry(files)
    except Exception as exc:                           # an unreadable registry must not take the page down
        return None, f"registry not readable ({type(exc).__name__})"
    owners = sorted({str(r["owner"]).partition("#")[0] for k, r in current.items()
                     if k.split("|", 1)[0] in ("label_version", "inference_check") and _text(r.get("owner"))})
    owner_sig = files_signature(owners)
    base = (sig + owner_sig, current, history, stats)
    _remember(("read", str(root)), (sig, owners, owner_sig, base))
    return base, None


def _strings_in(value):
    """Every string inside a record value (nested dicts and lists included, dict keys too)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings_in(k)
            yield from _strings_in(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings_in(v)


def hidden_matcher(hidden_ids):
    """-> text -> True when the text quotes a hidden id: a substring test for ids of GRAPH_SHORT_ID characters or
    more, a whole-word test for shorter ones (so a hidden 'rf_x' does not catch 'rf_x_v2'); empty ids never match."""
    long_ids = [h for h in hidden_ids if len(h) >= GRAPH_SHORT_ID]
    short = [h for h in hidden_ids if 0 < len(h) < GRAPH_SHORT_ID]
    word = re.compile("|".join(rf"(?<![A-Za-z0-9_]){re.escape(h)}(?![A-Za-z0-9_])" for h in short)) if short else None

    def quotes(text) -> bool:
        text = str(text)
        return any(h in text for h in long_ids) or bool(word and word.search(text))
    return quotes


def derived_ids(hidden_ids):
    """Matches a record id that IS a hidden id or a known derived form: pred_<id>, <id>_part<N>, pred_<id>_part<N>."""
    alts = "|".join(re.escape(h) for h in sorted(hidden_ids) if h)
    return re.compile(rf"(?:pred_)?(?:{alts})(?:_part\d+)?") if alts else None


def graph_hidden(records, viewer: Viewer, current: dict) -> tuple:
    """-> (History keys to leave out of the graph's INPUT, hidden set ids). Visibility is decided on Data's own record
    (records[entity][id], the last line, as the Lineage tree reads it): a prediction set, review package or Roboflow
    export the viewer may not see, or one Data does not have (unknown owner: admins only). Then every job that quotes
    a hidden id anywhere (any string or key, its own id too: hidden_matcher), and any other record whose id is a
    hidden id or a derived form of one (derived_ids). A dataset or label version built from a hidden set stays."""
    drop, hidden_ids = set(), set()
    for key in current:
        entity, rid = key.split("|", 1)
        if entity not in SET_ENTITIES:
            continue
        mine = records.get(entity, {}).get(rid)
        if not (set_is_visible(viewer, entity, mine) if mine else viewer.may_see(None)):
            drop.add(key)
            hidden_ids.add(rid)
    quotes, derived = hidden_matcher(hidden_ids), derived_ids(hidden_ids)
    if derived is None:
        return drop, hidden_ids
    for key, rec in current.items():
        if key in drop:
            continue
        entity, rid = key.split("|", 1)
        if entity == "job":
            if quotes(rid) or any(quotes(s) for s in _strings_in(rec)):
                drop.add(key)
        elif derived.fullmatch(rid):
            drop.add(key)
    return drop, hidden_ids


def graph_url(walker: "Lineage", entity: str, rid: str):
    """The item's Data Lineage page when it has one and this viewer may open it, else None."""
    kind = GRAPH_PAGE_KIND.get(entity)
    if kind is None:
        return None
    found = walker.lookup(KIND_TO_NODE[kind], rid)
    return lineage_href(kind, rid) if found and found[0] != "hidden" else None


def lab_graph(records, viewer: Viewer, root, focus=None, extra=None) -> dict:
    """History's lineage graph for this viewer: {schema, cols, nodes: [{GRAPH_NODE_KEYS}], edges: [[from, to, role]],
    focus: node index or None, error}. The registry is filtered BEFORE graph_model runs (graph_hidden), with History's
    own read_registry / build_model, so an admin gets exactly History's graph.
    Cost: build_model also builds History's image lookup and model pages (about 0.4 s on the lab registry, 11-2026);
    the graph is kept per registry state (the files' mtime_ns + size) and per set of hidden records (a visibility
    group), so it is built again only when the registry changes or a new group asks. A subtitle, status or edge role
    that quotes a hidden id is blanked.
    extra: a dict that, when given, also receives {keys: History key per node, base, drop} for source_graph (never
    sent to a client)."""
    base, error = history_registry(root)
    if base is None:
        return {"schema": SCHEMA, "cols": [], "nodes": [], "edges": [], "focus": None, "error": error}
    sig, current, history, stats = base
    drop, hidden_ids = graph_hidden(records, viewer, current)
    key = ("graph", str(root), sig, frozenset(drop))
    graph = _recall(key)
    if graph is None:
        kept = {k: r for k, r in current.items() if k not in drop}
        # sources and changes only feed History's image lookup and model pages, never the graph
        graph = _history_module().build_model(kept, {k: history[k] for k in kept}, stats, [], [])["graph"]
        _remember(key, graph)
    quotes = hidden_matcher(hidden_ids)
    walker = Lineage(records, viewer, root)
    nodes, focus_index = [], None
    for i, (k, col, hist, part, sub) in enumerate(graph["nodes"]):
        entity, rid = k.split("|", 1)
        status = current[k].get("status")
        status = None if status is None else str(status)
        nodes.append({"title": rid, "kind": GRAPH_SHORT.get(entity, entity), "col": col, "hist": hist, "part": part,
                      "sub": "" if quotes(sub) else sub,        # an export's note may quote a hidden id
                      "status": "" if status is not None and quotes(status) else status,
                      "url": graph_url(walker, entity, rid)})
        if k == focus:
            focus_index = i
    edges = [[a, b, "input" if quotes(role) else role] for a, b, role in graph["edges"]]
    if extra is not None:
        extra.update(keys=[n[0] for n in graph["nodes"]], base=base, drop=drop)
    return {"schema": SCHEMA, "cols": list(graph["cols"]), "nodes": nodes, "edges": edges, "focus": focus_index,
            "error": error}


ITEM_DOWNSTREAM_MAX = 150           # descendants in one item graph; past it a link to the lab graph (no count)


def item_graph(g: dict) -> dict:
    """The slice of a lab_graph() result (already visibility-filtered and whitelisted) that one item needs: its
    ancestors over REVERSE edges, its descendants over FORWARD edges (never an undirected walk: that would reach a
    sibling dataset through a shared label set), following `part` links too (a part's parent package is an ancestor,
    a package's parts are descendants). Descendants stop at ITEM_DOWNSTREAM_MAX, then "more" is true (no count).
    Node payloads are copied as they are; edges, `part` and `focus` are remapped to the slice's indices (a part whose
    parent is outside the slice -> -1). An unknown or hidden focus (focus None) gives an empty graph."""
    nodes, edges, f = g["nodes"], g["edges"], g.get("focus")
    out = {"schema": g["schema"], "scope": "item", "cols": [], "nodes": [], "edges": [], "focus": None,
           "more": False, "error": g.get("error")}
    if f is None:
        return out
    up, down, kids = {}, {}, {}
    for a, b, _ in edges:
        up.setdefault(b, []).append(a)
        down.setdefault(a, []).append(b)
    def pof(i):                                             # the package a part belongs to, else -1 (never itself)
        p = nodes[i]["part"]
        return p if p is not None and p >= 0 and p != i else -1

    for i in range(len(nodes)):
        if pof(i) >= 0:
            kids.setdefault(pof(i), []).append(i)

    def parents(i):
        return up.get(i, []) + ([pof(i)] if pof(i) >= 0 else [])

    def children(i):
        return down.get(i, []) + kids.get(i, [])

    keep, queue = {f}, [f]
    for i in queue:                                         # ancestors: uncapped
        for j in parents(i):
            if j not in keep:
                keep.add(j)
                queue.append(j)
    seen, queue, n_down = {f}, [f], 0
    for i in queue:                                         # descendants: breadth first, capped; an ancestor that
        for j in children(i):                               # is also downstream (a cycle) is still expanded
            if j in seen:
                continue
            if j not in keep:
                if n_down >= ITEM_DOWNSTREAM_MAX:
                    out["more"] = True
                    continue
                n_down += 1
                keep.add(j)
            seen.add(j)
            queue.append(j)
    order = sorted(keep)
    new = {old: k for k, old in enumerate(order)}
    out["cols"] = list(g["cols"])
    out["nodes"] = [dict(nodes[i], part=new.get(pof(i), -1)) for i in order]
    out["edges"] = [[new[a], new[b], role] for a, b, role in edges if a in new and b in new]
    out["focus"] = new[f]
    return out


def graph_focus_key(records, kind, entity_id):
    """The graph key (History's '<entity>|<id>') of a catalog item, for its Lineage page's Graph and "Lab graph" link.
    A session or grid gets 'session|<id>' / 'grid|<id>': not a lab graph node, its Graph is source_graph()."""
    if kind == "images":
        entity, _ = find_set(records, entity_id)
    else:
        entity = {"labels": "label_version", "dataset": "dataset", "model": "model", "session": "session",
                  "grid": "grid"}.get(kind)
    return f"{entity}|{entity_id}" if entity else None


def lab_graph_href(focus=None) -> str:
    focus = None if source_focus(focus) else focus          # a session / grid is not in the lab graph
    return f"{PREFIX}/lineage/graph/" + (f"?focus={quote(focus, safe='')}" if focus else "")


# ---------------------------------------------------------------- Session / grid graph (lab graph + its sources)
SOURCE_COLS = ("Imaging sessions", "Grids and uploads")    # drawn left of the lab graph's columns
SOURCE_ENTITIES = ("session", "grid")
SOURCE_ROLES = {"grid": "grid", "images": "used its images", "upload": "upload", "analysed": "analysed"}


def source_focus(focus):
    """('session' | 'grid', id) for a session / grid graph key, else None."""
    if not isinstance(focus, str):
        return None
    entity, sep, rid = focus.partition("|")
    return (entity, rid) if sep and entity in SOURCE_ENTITIES and rid else None


def source_links(root, base, drop) -> dict:
    """What used the images of each session and grid, from the registry the viewer's lab graph was built from (the
    hidden records already left out), so only visible lab-graph keys can appear:
      links[(session, grid or None)] = {History key}: an image record names the session (its grid: the image's own,
        else the session's session_grids slot; a grid the registry does not have counts as no grid), and an item
        lists that image -- History's own image lookup (registry/history_page.py lookup_index: a dataset's image
        list, a label version's change log and reviewed images, a review package's data.js, a revise / annotate
        job's list, a Roboflow image -> the pull it was first seen in), plus a Roboflow export's image list;
      grids[session] = its grids; sessions[grid] = its sessions; ps[session] = prediction sets whose source is that
      session; pkgs[prediction set id] = its prediction packages.
    Kept per registry state and visibility group, like the lab graph (built once, about the cost of History's image
    lookup)."""
    sig, current, _, _ = base
    key = ("sources", str(root), sig, frozenset(drop))
    hit = _recall(key)
    if hit is not None:
        return hit
    hp = _history_module()
    kept = {k: r for k, r in current.items() if k not in drop}
    where = {}                                              # image id -> (session, grid or None)
    for k, r in kept.items():
        if not k.startswith("image|") or not _text(r.get("session")):
            continue
        sid, slot = r["session"], r.get("session_grid")
        gid = _text(r.get("grid")) or _dict(_dict(kept.get("session|" + sid)).get("session_grids")).get(
            str(slot) if slot is not None else "")
        where[r["id"]] = (sid, str(gid) if gid and f"grid|{gid}" in kept else None)
    links: dict = {}

    def add(image, k):
        w = where.get(image)
        if w and k:
            links.setdefault(w, set()).add(k)
    lk = hp.lookup_index(kept, [Path(root)], [])
    for name, refs in lk["i"]:
        if name not in where:
            continue
        for ref in refs:
            k = lk["e"][ref if isinstance(ref, int) else ref[0]]
            if k.startswith("roboflow_image|"):
                pull = _text(kept[k].get("first_seen_pull"))
                k = f"pull|{pull}" if pull else None
            add(name, k)
    for k, r in kept.items():                               # History's lookup reads datasets' lists, not exports'
        if k.startswith("roboflow_export|") and isinstance(r.get("image_list"), dict):
            for stem, _ in hp.read_list_file(r["image_list"], [Path(root)])["rows"] or []:
                add(stem, k)
    grids, sessions = {}, {}
    for k, r in kept.items():
        if k.startswith("session|"):
            for gid in _dict(r.get("session_grids")).values():
                if f"grid|{gid}" in kept:
                    grids.setdefault(r["id"], set()).add(str(gid))
    for sid, gid in where.values():
        if gid:
            grids.setdefault(sid, set()).add(gid)
    for sid, gs in grids.items():
        for gid in gs:
            sessions.setdefault(gid, set()).add(sid)
    ps, pkgs = {}, {}
    for k, r in kept.items():
        if k.startswith("prediction_set|") and _text(_dict(r.get("source")).get("session")):
            ps.setdefault(r["source"]["session"], []).append(k)
        elif k.startswith("review_package|"):
            set_id = _text(r.get("prediction_set")) or (_text(r.get("base")) if r.get("base_kind") == "prediction_set"
                                                        else None)
            if set_id:
                pkgs.setdefault(set_id, []).append(k)
    out = {"links": {w: frozenset(ks) for w, ks in links.items()},
           "grids": {s: sorted(g, key=grid_sort_key) for s, g in grids.items()},
           "sessions": {g: sorted(s) for g, s in sessions.items()},
           "ps": {s: sorted(ks) for s, ks in ps.items()}, "pkgs": {p: sorted(ks) for p, ks in pkgs.items()}}
    _remember(key, out)
    return out


def source_node(entity, rid, col, **kw) -> dict:
    """A session / grid / upload box: exactly GRAPH_NODE_KEYS, never a path or a person."""
    node = {"title": rid, "kind": entity, "col": SOURCE_COLS.index(col), "hist": 0, "part": -1, "sub": "",
            "status": None, "url": None}
    node.update(kw)
    return node


def session_box(records, sid) -> dict:
    rec = records["session"].get(sid)
    if not rec:                                             # named by images only: no page to open
        return source_node("session", sid, "Imaging sessions")
    return source_node("session", sid, "Imaging sessions", sub=short_date(rec.get("date")),
                       status=str(rec["status"]) if rec.get("status") is not None else None,
                       url=lineage_href("session", sid))


def grid_box(records, gid) -> dict:
    rec = records["grid"].get(gid)
    return source_node("grid", gid, "Grids and uploads", sub=_text(_dict(rec).get("sample_label")) or "",
                       url=lineage_href("grid", gid) if rec else None)


def analysis_conversion(ps_id):
    """The conversion a prediction set was run on (its request.json source folder is <raw_tiles>/<cid>/tiles), or
    None. Only the folder NAME is read, compared with the session's conversions; the path is never sent."""
    def compute():
        root = predict_folder()
        if root is None or not PREDICTION_SET_ID.match(str(ps_id)):
            return None
        folder = _text(_dict(read_small_json(root / ps_id / "request.json").get("source")).get("dir"))
        if not folder:
            return None
        p = Path(folder)
        return p.parent.name if p.name == "tiles" else None
    return cached(_files_cache, ("ps-conversion", str(ps_id)), compute)


def source_graph(records, viewer: Viewer, root, focus) -> dict:
    """The Graph of a session's or grid's Lineage page, in item_graph's shape (scope item, cap and `more` included):
    the session -> its grids -> what used their images (source_links), its upload conversions (only those the viewer
    may open: the owner and the admin) -> the analyses run on them, and analyses of the session; then everything
    downstream of those over the lab graph's own edges (item_graph's directed walk, never a second walker). A grid's
    graph has its sessions as ancestors and only what used the grid's own images.
    Built from the viewer's FILTERED lab graph: an item the viewer may not see is not a node there, so it is never
    linked, counted or reached. Session / grid / upload boxes are added left of the lab columns (SOURCE_COLS).
    An unknown session / grid (or any other key) gives the empty graph with focus None and no echo."""
    extra: dict = {}
    g = lab_graph(records, viewer, root, None, extra)
    want = source_focus(focus)
    if want is None or "keys" not in extra or want[1] not in records[want[0]]:
        return item_graph(g)                                # focus None: the empty graph
    entity, rid = want
    sl = source_links(root, extra["base"], extra["drop"])
    idx = {k: i for i, k in enumerate(extra["keys"])}
    nodes = g["nodes"]                                      # fresh dicts per call: shifting the columns is safe
    for n in nodes:
        n["col"] += len(SOURCE_COLS)
    added: dict = {}

    def box(node) -> int:
        nodes.append(node)
        return len(nodes) - 1

    def link(a, b, role):
        if a is not None and b is not None and a != b:
            added.setdefault((a, b), SOURCE_ROLES[role])

    if entity == "session":
        focus_i = box(session_box(records, rid))
        for gid in sl["grids"].get(rid, []):
            gi = box(grid_box(records, gid))
            link(focus_i, gi, "grid")
            for k in sorted(sl["links"].get((rid, gid), ())):
                link(gi, idx.get(k), "images")
        for k in sorted(sl["links"].get((rid, None), ())):
            link(focus_i, idx.get(k), "images")
        convs = {}
        if viewer.cfg is not None and US.is_upload_session(records["session"][rid]):
            for c in US.visible_conversions(viewer.cfg, records["session"][rid], viewer.user or {}, viewer.can_see):
                n_tiles = _number(c.get("n_tiles"))
                convs[c["id"]] = box(source_node("upload", c["id"], "Grids and uploads",
                                                 sub=plural(int(n_tiles), "tile") if n_tiles is not None else "",
                                                 url=c["href"]))
                link(focus_i, convs[c["id"]], "upload")
        for k in sl["ps"].get(rid, []):
            if k not in idx:
                continue
            set_id = k.split("|", 1)[1]
            src = convs.get(analysis_conversion(set_id), focus_i)
            link(src, idx[k], "analysed")
            for pk in sl["pkgs"].get(set_id, []):
                link(src, idx.get(pk), "analysed")
    else:
        focus_i = box(grid_box(records, rid))
        for sid in sl["sessions"].get(rid, []):
            link(box(session_box(records, sid)), focus_i, "grid")
        for (sid, gid), ks in sl["links"].items():
            if gid == rid:
                for k in sorted(ks):
                    link(focus_i, idx.get(k), "images")
    full = {"schema": g["schema"], "cols": list(SOURCE_COLS) + g["cols"], "nodes": nodes,
            "edges": g["edges"] + [[a, b, role] for (a, b), role in added.items()], "focus": focus_i,
            "error": g["error"]}
    return item_graph(full)


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
                     f'{esc(r["title"])}</a>{tags}<a class="cj-data-lnk" href="{esc(r["lineage"])}">Lineage</a>'
                     f'</span>{sub}</li>')
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
            f'their sessions. <a href="{lab_graph_href()}">Lab graph</a>: how label sets, datasets, models and '
            'reviews connect.</p>' + banner(error) + search_form()
            + '<div class="cj-data-cards">' + "".join(cards) + "</div>"
            + '<p class="cj-mut cj-data-foot">Read only: nothing here changes any data (except the session details '
              'of an upload, which its uploader and the admin can edit).</p>')


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


def render_session_extras(d, extra="") -> str:
    """extra: the Session details form (the page handler renders it from the record; never in the JSON)."""
    out = ['<h2 class="cj-sec">Grids</h2>']
    if d["grids"]:
        items = []
        for g in d["grids"]:
            slot_name = f'Slot {g["slot"]}'
            head = (f'<a href="{esc(entity_href("grid", g["grid"]))}">Grid {esc(g["grid"])}</a>' if g["grid"]
                    else esc(slot_name))
            if g["grid"]:
                where = slot_name
            elif g["grid_id_recorded"]:
                where = f'grid {g["grid_id_recorded"]} not in the list'
            elif g.get("grid_id_typed"):
                where = f'grid id {g["grid_id_typed"]} (session details)'
            else:
                where = "grid number not recorded"
            count = f'{plural(g["n_images"], "image")} ({g["count_source"]})' if g["n_images"] else None
            items.append(f'<li><span class="cj-data-top"><b>{head}</b></span>'
                         f'<span class="cj-mut">{esc(join_bits(g["label"], g["grid_type"], count, where))}</span></li>')
        out.append('<ul class="cj-data-grids">' + "".join(items) + "</ul>")
    else:
        out.append('<p class="cj-mut">No grids recorded for this session.</p>')
    up = d.get("upload")
    if up is not None:
        out.append('<h2 class="cj-sec">Uploaded raw files</h2>')
        if up["conversions"]:
            items = "".join(
                f'<li class="cj-data-row"><span class="cj-data-top"><a href="{esc(c["url"])}">Conversion '
                f'{esc(c["id"])}</a><a class="cj-data-lnk" href="{esc(c["gallery"])}">Gallery</a></span>'
                + (f'<span class="cj-mut">{esc(plural(c["n_tiles"], "tile"))}</span>'
                   if _number(c.get("n_tiles")) is not None else "") + "</li>"
                for c in up["conversions"])
            out.append(f'<div class="cj-card"><ul class="cj-data-rows">{items}</ul></div>')
        else:
            out.append('<p class="cj-mut">Converted from a portal upload. Its conversion and gallery are open to the '
                       'person who uploaded it and the admin.</p>')
        out.append(extra or US.details_view({"details": up["details"]}))
    if d.get("analyses"):
        items = "".join(f'<li class="cj-data-row"><span class="cj-data-top"><a href="{esc(a["url"])}">'
                        f'{esc(a["title"])}</a>' + (chip(a["status"]) if a.get("status") else "") + "</span></li>"
                        for a in d["analyses"])
        out.append(f'<h2 class="cj-sec">Analyses of this session</h2><div class="cj-card"><ul class="cj-data-rows">'
                   f'{items}</ul></div>')
    out.append(f'<h2 class="cj-sec">Images ({d["n_images"]} {esc(d.get("images_word") or "processed")})</h2>'
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


def render_detail(d, error, extra="") -> str:
    kind = d["kind"]
    list_kind = DETAIL_LIST[kind]
    card_cls = "cj-card" + (" cj-data-hl" if d.get("production") else "")
    out = [breadcrumb(("Data", PREFIX + "/"), (KIND_TITLES[list_kind], f"{PREFIX}/list/{list_kind}")),
           f'<div class="cj-data-top"><h1 class="cj-wrap">{esc(d["title"])}</h1>'
           + "".join(chip(t, tone) for t, tone in d["chips"]) + "</div>",
           banner(error), f'<div class="{card_cls}">{render_fields(d["fields"])}</div>']
    if d.get("review_url"):
        out.append(f'<p class="cj-data-more"><a href="{esc(d["review_url"])}">Open in review</a></p>')
    out.append(f'<p class="cj-data-more"><a class="cj-data-lnk" href="{esc(lineage_href(kind, d["id"]))}">'
               + ("Lineage: what was made from it" if kind in SOURCE_ENTITIES else "Lineage: where this came from")
               + "</a></p>")
    if d.get("gap"):
        out.append(f'<p class="cj-mut">Acquisition details are not available: {esc(d["gap"])}</p>')
    if kind == "session":
        out.append(render_session_extras(d, extra))
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


def render_lineage_node(node) -> str:
    status = node["status"]
    if status == "more":
        return f'<li class="cj-data-ln"><p class="cj-mut cj-data-lmore">{esc(node["title"])}</p></li>'
    title = esc(node["title"])
    if node.get("url") and status in ("ok", "cut", "loop"):
        title = f'<a href="{esc(node["url"])}">{title}</a>'
    links = ""
    if node.get("lineage") and status in ("ok", "cut"):
        links = f'<a class="cj-data-lnk" href="{esc(node["lineage"])}">Lineage from here</a>'
    chips = "".join(chip(t, tone) for t, tone in node["chips"])
    step = esc(node["step"] if status != "hidden" else "Step")
    if node.get("set_kind"):
        step = esc(node["set_kind"])
    parts = [f'<span class="cj-data-lstep">{step}</span>',
             f'<div class="cj-data-top"><b class="cj-wrap">{title}</b>{chips}{links}</div>']
    if status == "unknown" and node.get("recorded"):
        parts.append(f'<p class="cj-mut cj-data-lnote">Recorded as {esc(node["recorded"])}.</p>')
    if node.get("note"):
        parts.append(f'<p class="cj-mut cj-data-lnote">{esc(node["note"])}</p>')
    parts.append(render_fields(node["fields"]))
    if node.get("thumbs"):
        parts.append(render_thumbs(node["thumbs"], strip=True))
    card = f'<div class="cj-card cj-data-lc cj-data-ls-{esc(status)}">{"".join(parts)}</div>'
    kids = ""
    if node["parents"]:
        kids = ('<p class="cj-data-lfrom cj-mut">Made from</p><ol class="cj-data-lkids">'
                + "".join(render_lineage_node(p) for p in node["parents"]) + "</ol>")
    return f'<li class="cj-data-ln">{card}{kids}</li>'


def view_toggle(focus) -> str:
    """Tree | Graph, for an item with a graph (a lab graph node, or a session / grid: source_graph). Hidden until
    /data/data.js runs: without script the Tree is the page. A session's or grid's tree is only itself (and a grid's
    sessions), so its page opens on the Graph unless the person last chose Tree (data-default)."""
    if not focus:
        return ""
    default = ' data-default="graph"' if source_focus(focus) else ""
    return ('<div class="cj-data-graph-toggle" id="cj-data-view-toggle" role="group" aria-label="View"'
            f'{default} hidden>'
            '<button type="button" data-view="tree" aria-pressed="true">Tree</button>'
            '<button type="button" data-view="graph" aria-pressed="false">Graph</button></div>')


SOURCE_NOTES = {"session": "This session, its grids and uploads, and everything made from their images. "
                           "Arrows: parent → child (inputs).",
                "grid": "This grid, the sessions that imaged it, and everything made from its images. "
                        "Arrows: parent → child (inputs)."}


def graph_panel(focus) -> str:
    if not focus:
        return ""
    ep = f"{PREFIX}/api/lineage/graph?focus={quote(focus, safe='')}&scope=item"
    src = source_focus(focus)
    return ('<div id="cj-data-graph-panel" hidden>'
            + graph_frame(ep, True, lab_graph_href(focus), SOURCE_NOTES[src[0]] if src else None) + "</div>"
            f'<script src="{PREFIX}/data.js" defer></script>')


def render_lineage(tree, error, focus=None) -> str:
    kind, root = tree["kind"], tree["root"]
    s = tree["summary"]
    cut = f", {s['cut']} not followed" if s["cut"] else ""
    loop = f", {plural(s['loop'], 'loop')} in the registry" if s["loop"] else ""
    untraced = (f" {plural(s['untraced_images'], 'listed image')} not traced to an imaging session."
                if s.get("untraced_images") else "")
    summary = (f'{plural(s["steps"], "step")}: {s["known"]} found, {s["unknown"]} unknown, '
               f'{s["hidden"]} not visible to you{cut}{loop}.{untraced}')
    lab = (f'Graph: what was made from its images. <a href="{esc(lab_graph_href())}">Lab graph</a> shows how '
           'everything connects.' if source_focus(focus) else
           f'<a href="{esc(lab_graph_href(focus))}">Lab graph</a> shows it among everything else.')
    return (breadcrumb(("Data", PREFIX + "/"), (KIND_TITLES[DETAIL_LIST[kind]], f"{PREFIX}/list/{DETAIL_LIST[kind]}"),
                       (tree["title"], entity_href(kind, tree["id"])))
            + f'<h1 class="cj-wrap">Lineage of {esc(tree["title"])}</h1>'
            + '<p class="cj-mut cj-data-sub">The chain back to the raw imaging sessions, read from the registry. '
              f'A step the registry does not record is shown as unknown, never guessed. {lab}</p>'
            + banner(error) + view_toggle(focus)
            + '<div id="cj-data-tree-panel">' + f'<p class="cj-data-lsum">{esc(summary)}</p>'
            + '<ol class="cj-data-lroot">' + render_lineage_node(root) + "</ol></div>"
            + graph_panel(focus))


def graph_frame(endpoint: str, item: bool = False, lab_href: str = "", note=None) -> str:
    """The graph's controls and empty canvas. /data/data.js finds the frame by id cj-data-graph-root and takes its
    endpoint (and, on the lab page, the page's own ?focus) from data-* attributes; no record text is in this HTML.
    note: the line beside the controls (a session's / grid's own; default: the lab's or an item's)."""
    attrs = (f'data-endpoint="{esc(endpoint)}"' + (' data-scope="item"' if item else ' data-focus-from-url="1"'))
    note = note or ("Arrows: parent \u2192 child (inputs). Dashed: model predictions \u2192 review / revise \u2192 "
                    "label version. \u2605 = production." if not item else
                    "This item, what it was built from, and what was built from it. Arrows: parent \u2192 child "
                    "(inputs).")
    return ('<div class="cj-data-graph-bar">'
            '<label><input type="checkbox" id="cj-data-graph-hist"> <span id="cj-data-graph-histl">Show history'
            '</span></label>'
            '<label><input type="checkbox" id="cj-data-graph-an"> <span id="cj-data-graph-anl">Show analyses'
            '</span></label>'
            f'<span class="cj-mut cj-data-graph-note">{note}</span></div>'
            f'<div id="cj-data-graph-root" {attrs}>'
            '<p class="cj-mut cj-data-graph-status" id="cj-data-graph-status" role="status">Loading the graph\u2026</p>'
            '<div class="cj-data-graph-wrap" id="cj-data-graph-wrap"><div class="cj-data-graph-canvas" '
            'id="cj-data-graph-canvas"><svg id="cj-data-graph-svg" aria-hidden="true"></svg></div></div>'
            + ('<p class="cj-data-graph-more" id="cj-data-graph-more" hidden>'
               f'<a href="{esc(lab_href)}">More downstream: open lab graph</a></p>' if item else "")
            + '</div>')


def render_lab_graph(error) -> str:
    """The Lab graph page: a frame only. /data/data.js reads /data/api/lineage/graph (with this page's ?focus) and
    draws it, so no record text is in this HTML."""
    return (breadcrumb(("Data", PREFIX + "/")) + '<h1>Lab graph</h1>'
            '<p class="cj-mut cj-data-sub">How label sets, datasets, models and reviews connect, read from the '
            'registry. Open a box for its lineage.</p>' + banner(error)
            + graph_frame(PREFIX + "/api/lineage/graph")
            + '<noscript><p class="cj-mut">The Lab graph needs JavaScript.</p></noscript>'
            + f'<script src="{PREFIX}/data.js" defer></script>')


def page_body(content: str) -> str:
    """data.html with the content put in ONE replace (text inside the content is never re-scanned)."""
    return TEMPLATE.read_text(encoding="utf-8").replace("{{content}}", content, 1)


# ---------------------------------------------------------------- routes
def routes(ctx) -> list:
    def state(request: Request):
        records, error = read_registry(getattr(ctx.cfg, "registry", None))
        return with_disk_packages(records), error, Viewer(ctx.can_see, ctx.user(request), ctx.cfg)

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

    def details_html(records, d, viewer, problem=None, saved=False) -> str:
        """The Session details form for its uploader / the admin (when switched on); else the read-only view."""
        up = d.get("upload")
        if not up or not up["can_edit"]:
            return ""
        record = records["session"].get(d["id"]) or {}
        return US.details_form(record, f'{PREFIX}/session/{quote(d["id"], safe="")}/details', problem, saved,
                               viewer.user)

    def item(kind):
        def handler(request: Request):
            records, error, viewer = state(request)
            page = requested_page(request, None if kind == "session" else 1)
            d = detail(records, kind, request.path_params["eid"], viewer, page)
            if d is None:
                return missing(request)
            extra = details_html(records, d, viewer, saved=request.query_params.get("saved") == "1") \
                if kind == "session" else ""
            return html_page(request, d["title"], render_detail(d, error, extra))
        return handler

    async def session_details(request: Request):
        """POST the Session details form: the session's uploader or the admin, an upload session, switched on."""
        user, sid = ctx.user(request), request.path_params["eid"]
        record = await run_in_threadpool(US.current_record, ctx.cfg, sid)
        if not record or not US.may_edit(user, record):
            return await run_in_threadpool(missing, request)
        fd = await request.form(max_files=0, max_fields=4 * US.MAX_GRIDS + 10)
        form = {k: v for k, v in fd.items() if isinstance(v, str)}
        res = await run_in_threadpool(US.save_details, sid, form, user.get("id"), ctx.cfg,
                                      user.get("role") == "admin")
        root = getattr(ctx.cfg, "registry", None)
        with _lock:
            _registry_cache.pop(str(root), None)              # the page after the redirect shows the new line
        if res.get("ok"):
            return RedirectResponse(f"{PREFIX}/session/{quote(sid, safe='')}?saved=1#cj-us-details", status_code=303)
        problem = res["error"] + (f"\n(admin) {res['tech']}" if res.get("tech") and user.get("role") == "admin"
                                  else "")

        def again():
            records, error, viewer = state(request)
            d = detail(records, "session", sid, viewer)
            if d is None:
                return missing(request)
            return html_page(request, d["title"], render_detail(d, error, details_html(records, d, viewer, problem)),
                             res.get("status", 400))
        return await run_in_threadpool(again)

    def not_found_text():
        return Response("not found", status_code=404, media_type="text/plain")

    def thumb(request: Request):
        kind, eid, n = request.path_params["kind"], request.path_params["eid"], request.path_params["n"]
        if kind not in DETAIL_KINDS or not n.isdigit() or len(n) > 7 or not eid or len(eid) > 200 or "\x00" in eid:
            return not_found_text()
        records, _, viewer = state(request)
        files = thumbnail_sources(records, kind, eid, viewer)
        index_ = int(n)
        roots = known_roots() + (upload_roots(ctx.cfg) if kind == "session" else [])
        if index_ >= len(files) or not inside(files[index_], roots):
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

    def lineage_page(request: Request):
        records, error, viewer = state(request)
        tree = lineage_tree(records, viewer, request.path_params["kind"], request.path_params["eid"],
                            getattr(ctx.cfg, "registry", None))
        if tree is None:
            return missing(request)
        focus = graph_focus_key(records, tree["kind"], tree["id"])
        return html_page(request, "Lineage of " + tree["title"], render_lineage(tree, error, focus))

    def api_lineage(request: Request):
        records, _, viewer = state(request)
        tree = lineage_tree(records, viewer, request.path_params["kind"], request.path_params["eid"],
                            getattr(ctx.cfg, "registry", None))
        return ctx.json(tree) if tree is not None else ctx.json({"error": "not found"}, 404)

    def graph_page(request: Request):
        _, error = history_registry(getattr(ctx.cfg, "registry", None))
        return html_page(request, "Lab graph", render_lab_graph(error))

    def api_graph(request: Request):
        records, _, viewer = state(request)
        focus = request.query_params.get("focus")
        focus = focus if focus and len(focus) <= 300 else None
        root = getattr(ctx.cfg, "registry", None)
        if request.query_params.get("scope") == "item" and source_focus(focus):
            return ctx.json(source_graph(records, viewer, root, focus))     # a session / grid
        g = lab_graph(records, viewer, root, focus)
        return ctx.json(item_graph(g) if request.query_params.get("scope") == "item" else g)

    def script(request: Request):
        r = Response(SCRIPT.read_bytes(), media_type="text/javascript; charset=utf-8")
        r.headers["Cache-Control"] = "no-cache"
        r.headers["X-Content-Type-Options"] = "nosniff"
        return r

    found = [Route(PREFIX + "/lineage/graph/", graph_page, methods=["GET"]),     # before /lineage/{kind}/{eid}
             Route(PREFIX + "/lineage/graph", graph_page, methods=["GET"]),
             Route(PREFIX + "/api/lineage/graph", api_graph, methods=["GET"]),   # before /api/{kind}/{eid}
             Route(PREFIX + "/data.js", script, methods=["GET"])]
    found += [Route(PREFIX + "/", index, methods=["GET"]),
             Route(PREFIX + "/search", find, methods=["GET"]),
             Route(PREFIX + "/list/{kind}", listing, methods=["GET"])]
    found += [Route(PREFIX + f"/{k}/{{eid}}", item(k), methods=["GET"]) for k in DETAIL_KINDS]
    found += [Route(PREFIX + "/session/{eid}/details", session_details, methods=["POST"])]
    found += [Route(PREFIX + "/lineage/{kind}/{eid}/", lineage_page, methods=["GET"]),
              Route(PREFIX + "/lineage/{kind}/{eid}", lineage_page, methods=["GET"]),
              Route(PREFIX + "/api/lineage/{kind}/{eid}", api_lineage, methods=["GET"])]
    found += [Route(PREFIX + "/thumb/{kind}/{eid}/{n}.jpg", thumb, methods=["GET"]),
              Route(PREFIX + "/api/overview", api_overview, methods=["GET"]),
              Route(PREFIX + "/api/search", api_search, methods=["GET"]),
              Route(PREFIX + "/api/list/{kind}", api_list, methods=["GET"]),
              Route(PREFIX + "/api/{kind}/{eid}", api_item, methods=["GET"])]
    return found
