"""
Data module (CJ portal, read only): a plain catalog of WHAT DATA EXISTS, from the registry (sessions, grids, image
sets, label versions, datasets, models) and the processed images on disk.

    GET /data/                          overview: counts and short lists, search box
    GET /data/search?q=                 search by id or sample label across everything
    GET /data/list/<kind>               the full list (sessions, grids, images, labels, datasets, models)
    GET /data/session/<id>              acquisition summary, its grids, a strip of up to 12 images
    GET /data/grid/<id>                 one grid: sample, type, sessions, thumbnails (paged, 24 per page)
    GET /data/model/<id>                status (production badge), dataset, scores from the registry
    GET /data/dataset/<id>              splits, classes, source, models trained on it, thumbnails
    GET /data/labels/<id>               a label version: what it changed, how much was verified, thumbnails
    GET /data/images/<id>               an image set (review package, prediction set, Roboflow export), thumbnails
    GET /data/thumb/<kind>/<id>/<n>.jpg one thumbnail (about 240 px), disk-cached under <state dir>/data_thumbs
    GET /data/api/overview | search?q= | list/<kind> | <kind>/<id>[?page=N]    the same information as JSON
                                        (schema cj_data/1; other modules reuse it)

Rules:
  - Nothing here writes to the registry or to a data folder. The only write is the thumbnail cache in the portal's
    own state folder.
  - A thumbnail file is resolved on the server from registry / package data only (the kind + id + index in the URL
    pick it); the client never sends a path. A file counts only when its real path (links followed) is under a known
    root: the CryoAI-laptop folder or the Cryo-image-processing folder (both read only here).
  - Privacy: a prediction set (or a package with an `owner`) is listed only for its owner and admins
    (ctx.can_see); everyone else gets neither the row, the page nor the thumbnails.
  - Plain words; every value is escaped; no script on the page.
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

from starlette.requests import Request
from starlette.responses import HTMLResponse, Response
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
STRIP_N = 12                 # thumbnails on a session page
THUMB_PX = 240
SEARCH_CAP = 50
IMG_EXT = {".jpg", ".jpeg", ".png"}
ENTITIES = ("session", "grid", "image", "dataset", "label_version", "model", "review_package", "roboflow_export",
            "prediction_set", "inference_check")
SESSION_DIRS = ("3a. Individual tiles", "3c. Tiles no scalebar - native PNG",
                "3d. Tiles no scalebar - 14.7 A-px PNG", "3b. Stitched montages")
PROCESSED = "3. Processed images"
KINDS = ("sessions", "grids", "images", "labels", "datasets", "models")
DETAIL_KINDS = ("session", "grid", "images", "labels", "dataset", "model")
ID_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$")
SESSION_ID = re.compile(r"^\d{8}(-\d+)?$")
PS_ID = re.compile(r"^ps_\d{8}_[0-9a-f]{6}$")
SPLITS = ("train", "valid", "test")

esc = html.escape


# ---------------------------------------------------------------- small helpers
def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _l(x) -> list:
    return x if isinstance(x, list) else []


def _s(x):
    return x if isinstance(x, str) and x else None


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def fnum(x, d=2) -> str:
    x = _num(x)
    if x is None:
        return "?"
    return f"{x:,.{d}f}".rstrip("0").rstrip(".") if d else f"{x:,.0f}"


def link(path: str) -> str:
    return PREFIX + path


def ref(kind: str, eid: str) -> str:
    return f"{PREFIX}/{kind}/{quote(str(eid), safe='')}"


def _short_date(s) -> str:
    return s[:10] if isinstance(s, str) else ""


# ---------------------------------------------------------------- registry (cached 10 s) and roots
_lock = threading.Lock()
_reg_cache: dict = {}
_files_cache: dict = {}
_pkg_cache: dict = {}
_reg_mod = None


def _registry_module():
    """registry/registry.py, loaded once by path (registry/ is not a package)."""
    global _reg_mod
    if _reg_mod is None:
        spec = importlib.util.spec_from_file_location("cj_data_registry", REPO / "registry" / "registry.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _reg_mod = mod
    return _reg_mod


def read_registry(root, now=None) -> tuple:
    """-> ({entity: {id: record}}, error or None). One read per root per REGISTRY_TTL_S; callers never change the
    records (they are shared between requests)."""
    if root is None:
        return {e: {} for e in ENTITIES}, "no registry folder configured"
    now = time.monotonic() if now is None else now
    key = str(root)
    with _lock:
        hit = _reg_cache.get(key)
        if hit is not None and 0 <= now - hit[0] < REGISTRY_TTL_S:
            return hit[1], hit[2]
        S, err = {e: {} for e in ENTITIES}, None
        if not Path(root).is_dir():
            err = "registry folder not found"
        else:
            try:
                reg = _registry_module()
                for e in ENTITIES:
                    S[e] = reg.current(e, Path(root))
            except Exception as ex:                      # an unreadable registry must not take the page down
                S, err = {e: {} for e in ENTITIES}, f"registry not readable ({type(ex).__name__})"
        _reg_cache[key] = (now, S, err)
        return S, err


def known_roots() -> list:
    """The CryoAI-laptop and Cryo-image-processing folders (env CRYOEV_LAPTOP_ROOT / CRYOEV_PROCESSING_ROOT, else the
    sibling folders above the repo). Real paths, in the OS's comparison case."""
    cfgm = _registry_module().config
    out = []
    for r in (cfgm.laptop_root(False), cfgm.processing_root()):
        if r:
            out.append(os.path.normcase(os.path.realpath(r)))
    return out


def _root_of(which: str):
    cfgm = _registry_module().config
    return cfgm.laptop_root(False) if which == "laptop" else cfgm.processing_root()


def inside(p, roots) -> bool:
    try:
        rp = os.path.normcase(os.path.realpath(p))
    except (OSError, ValueError):
        return False
    return any(rp == r or rp.startswith(r.rstrip("\\/") + os.sep) for r in roots)


def _cached(store: dict, key, fn, ttl=FILES_TTL_S):
    now = time.monotonic()
    with _lock:
        hit = store.get(key)
        if hit is not None and 0 <= now - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        store[key] = (now, val)
        if len(store) > 512:
            store.pop(next(iter(store)))
    return val


# ---------------------------------------------------------------- images on disk (read only)
def _scan(d: Path, roots, prefix=None) -> list:
    """Image files directly in `d` (sorted by name), only when `d` and each file are under a known root."""
    def go():
        try:
            if not d.is_dir() or not inside(d, roots):
                return []
            out = []
            with os.scandir(d) as it:
                for e in it:
                    n = e.name
                    if n.startswith(".") or os.path.splitext(n)[1].lower() not in IMG_EXT:
                        continue
                    if prefix and not n.lower().startswith(prefix):
                        continue
                    if not e.is_file():
                        continue
                    p = Path(e.path)
                    if e.is_symlink() and not inside(p, roots):
                        continue
                    out.append(p)
            out.sort(key=lambda p: p.name.lower())
            return out
        except OSError:
            return []
    return _cached(_files_cache, ("scan", str(d), prefix, tuple(roots)), go)


def session_files(sid: str, roots, prefix=None) -> list:
    """The session's processed images: the first of 3a / 3c / 3d / 3b that has any."""
    proc = _root_of("processing")
    if not proc or not SESSION_ID.match(sid or ""):
        return []
    base = Path(proc) / sid / PROCESSED
    for sub in SESSION_DIRS:
        got = _scan(base / sub, roots, prefix)
        if got:
            return got
    return []


def package_dir(pid: str, rec: dict, roots):
    lap = _root_of("laptop")
    cands = []
    if lap and ID_SAFE.match(pid or ""):
        cands.append(Path(lap) / "review" / pid)
    p = _s(rec.get("path"))
    if p:
        pp = Path(p)
        if pp.is_absolute():
            cands.append(pp)
        elif lap:
            cands += [Path(lap).parent / pp, Path(lap) / pp]
    for c in cands:
        try:
            if c.is_dir() and inside(c, roots):
                return c
        except OSError:
            continue
    return None


def package_files(pdir: Path, roots) -> list:
    """Images of a review package: data.js `images[].src` (relative, under the package), else its images/ folder."""
    def go():
        out = []
        dj = pdir / "data.js"
        base = os.path.normcase(os.path.realpath(pdir))
        try:
            text = dj.read_text(encoding="utf-8")
            obj, _ = json.JSONDecoder().raw_decode(text, text.index("{"))
        except (OSError, ValueError):
            obj = None
        if isinstance(obj, dict):
            for im in _l(obj.get("images")):
                src = _s(_d(im).get("src"))
                if not src or src.startswith(("/", "\\")) or ":" in src:
                    continue
                p = pdir / src
                try:
                    ok = (os.path.splitext(src)[1].lower() in IMG_EXT and p.is_file() and inside(p, roots)
                          and inside(p, [base]))
                except OSError:
                    ok = False
                if ok:
                    out.append(p)
            return out
        return _scan(pdir / "images", roots)
    try:
        key = ("pkg", str(pdir), (pdir / "data.js").stat().st_mtime_ns, tuple(roots))
    except OSError:
        key = ("pkg", str(pdir), None, tuple(roots))
    return _cached(_pkg_cache, key, go)


def prediction_files(psid: str, roots) -> list:
    lap = _root_of("laptop")
    if not lap or not PS_ID.match(psid or ""):
        return []
    return _scan(Path(lap) / "predict" / psid / "images", roots)


# ---------------------------------------------------------------- entities: ordering, visibility, rows
def find_set(S, sid):
    """An image set by id -> (entity, record) or (None, None): prediction set, review package, Roboflow export."""
    for e in ("prediction_set", "review_package", "roboflow_export"):
        r = S[e].get(sid)
        if r:
            return e, r
    return None, None


def visible(can_see, user, rec) -> bool:
    owner = rec.get("owner")
    return owner is None or bool(can_see(user, owner))


def _when(rec):
    return str(rec.get("created") or rec.get("built") or rec.get("logged") or "")


def split_counts(rec):
    ct = rec.get("counts_trained") if isinstance(rec.get("counts_trained"), dict) else None
    if ct is None and isinstance(rec.get("counts"), dict):
        ct = {s: _d(v).get("images") for s, v in rec["counts"].items()}
    if not ct:
        return None
    names = [s for s in SPLITS if s in ct] + sorted(set(ct) - set(SPLITS))
    return "/".join("?" if ct[s] is None else str(ct[s]) for s in names) + " " + "/".join(names)


def review_chip(rec) -> tuple:
    if rec.get("review_scope") == "reviewed_only":
        n, m = rec.get("n_reviewed"), rec.get("n_package_images")
        return (f"verified {'?' if n is None else n} of {'?' if m is None else m}", "acc")
    return ("includes edits not yet checked by a second person", "warn")


def mags(rec) -> str:
    m = [x for x in _l(rec.get("magnification")) if _num(x) is not None]
    return " and ".join(f"{x:,}×" for x in m)


def set_kind_label(e):
    return {"prediction_set": "Prediction set", "review_package": "Review package",
            "roboflow_export": "Roboflow export"}.get(e, e)


def row(kind, eid, title, sub, tags=(), hl=False, **extra) -> dict:
    r = {"kind": kind, "id": eid, "title": title, "sub": sub, "url": ref(kind, eid),
         "api": f"{PREFIX}/api/{kind}/{quote(str(eid), safe='')}", "tags": [list(t) for t in tags], "hl": hl}
    r.update(extra)
    return r


def session_labels(rec) -> list:
    return [v for _, v in sorted(_d(rec.get("sample_labels")).items(), key=lambda kv: _slot(kv[0])) if _s(v)]


def _slot(k):
    return (0, int(k), "") if str(k).isdigit() else (1, 0, str(k))


def rows_for(S, kind, can_see, user) -> list:
    out = []
    if kind == "sessions":
        for sid, r in sorted(S["session"].items(), reverse=True):
            bits = [_short_date(r.get("date")), r.get("instrument"), mags(r),
                    f"{r['n_imaging']} imaging files" if _num(r.get("n_imaging")) is not None else None]
            labels = session_labels(r)
            if labels:
                bits.append(", ".join(labels[:3]) + (f" +{len(labels) - 3} more" if len(labels) > 3 else ""))
            out.append(row("session", sid, f"Session {sid}", " · ".join(b for b in bits if b),
                           [(r["status"], "")] if _s(r.get("status")) else []))
    elif kind == "grids":
        for gid, r in sorted(S["grid"].items(), reverse=True):
            bits = [r.get("sample_label"), r.get("grid_type"), _short_date(r.get("prep_date"))]
            out.append(row("grid", gid, f"Grid {gid}", " · ".join(b for b in bits if _s(b))))
    elif kind == "images":
        recs = []
        for e in ("prediction_set", "review_package", "roboflow_export"):
            recs += [(e, r) for r in S[e].values() if visible(can_see, user, r)]
        for e, r in sorted(recs, key=lambda er: (_when(er[1]), er[1]["id"]), reverse=True):
            n = r.get("n_images")
            if n is None and isinstance(r.get("counts"), dict):
                n = sum(_num(_d(v).get("images")) or 0 for v in r["counts"].values())
            bits = [set_kind_label(e), f"{n} images" if n is not None else None, _short_date(_when(r)), r.get("status")]
            out.append(row("images", r["id"], r["id"], " · ".join(str(b) for b in bits if b)))
    elif kind == "labels":
        for r in sorted(S["label_version"].values(), key=lambda r: (_when(r), r["id"]), reverse=True):
            txt, tone = review_chip(r)
            n = r.get("n_images_applied")
            bits = [txt, f"{n} images changed" if n is not None else None, _short_date(_when(r))]
            tags = [(r["status"], "")] if _s(r.get("status")) else []
            out.append(row("labels", r["id"], r["id"], " · ".join(b for b in bits if b), tags))
    elif kind == "datasets":
        for r in sorted(S["dataset"].values(), key=lambda r: (_when(r), r["id"]), reverse=True):
            bits = [split_counts(r), ", ".join(map(str, _l(r.get("classes")))) or None, _short_date(_when(r))]
            tags = [(r["status"], "")] if _s(r.get("status")) else []
            out.append(row("dataset", r["id"], r["id"], " · ".join(b for b in bits if b), tags))
    elif kind == "models":
        for r in sorted(S["model"].values(), key=lambda r: (r.get("status") == "production", _when(r), r["id"]),
                        reverse=True):
            prod = r.get("status") == "production"
            bits = [f"{r['imgsz']} px" if r.get("imgsz") else None, f"started from {r['init_from']}" if r.get("init_from") else None,
                    _short_date(_when(r))]
            tags = [("production", "prod")] if prod else ([(r["status"], "")] if _s(r.get("status")) else [])
            out.append(row("model", r["id"], r["id"], " · ".join(b for b in bits if b), tags, hl=prod))
    return out


def overview(S, can_see, user) -> dict:
    cards = []
    titles = {"sessions": "Sessions", "images": "Image sets", "labels": "Label versions",
              "datasets": "Datasets", "models": "Models"}
    for kind in ("sessions", "images", "labels", "datasets", "models"):
        rows = rows_for(S, kind, can_see, user)
        cards.append({"kind": kind, "title": titles[kind], "count": len(rows), "rows": rows[:OVERVIEW_N],
                      "all": f"{PREFIX}/list/{kind}"})
    return {"schema": SCHEMA, "cards": cards, "counts": {c["kind"]: c["count"] for c in cards} | {"grids": len(S["grid"])}}


def search(S, q, can_see, user) -> dict:
    q = (q or "").strip()[:100]
    ql = q.lower()
    out = {"schema": SCHEMA, "q": q, "groups": []}
    if not ql:
        return out
    labels = {"sessions": "Sessions", "grids": "Grids", "images": "Image sets", "labels": "Label versions",
              "datasets": "Datasets", "models": "Models"}
    for kind in KINDS:
        hits = []
        for r in rows_for(S, kind, can_see, user):
            rec = _record_for(S, kind, r["id"])
            hay = [r["id"], r["title"]]
            if kind == "sessions":
                hay += session_labels(rec or {})
            elif kind == "grids":
                hay += [_s(rec.get("sample_label")) or "", _s(rec.get("grid_type")) or ""] if rec else []
            elif kind == "datasets" or kind == "models":
                hay += [_s(rec.get("note")) or ""] if rec else []
            if any(ql in str(h).lower() for h in hay):
                hits.append(r)
        # a grid's sample label also finds the sessions that imaged it
        if kind == "sessions":
            gl = {gid for gid, g in S["grid"].items() if ql in str(g.get("sample_label") or "").lower()}
            have = {h["id"] for h in hits}
            for sid, s in S["session"].items():
                if sid not in have and gl & {v for v in _d(s.get("session_grids")).values() if v}:
                    hits.append(next(r for r in rows_for(S, "sessions", can_see, user) if r["id"] == sid))
        if hits:
            out["groups"].append({"kind": kind, "title": labels[kind], "count": len(hits), "rows": hits[:SEARCH_CAP]})
    return out


def _record_for(S, kind, eid):
    return {"sessions": S["session"], "grids": S["grid"], "labels": S["label_version"], "datasets": S["dataset"],
            "models": S["model"]}.get(kind, {}).get(eid) or (find_set(S, eid)[1] if kind == "images" else None)


# ---------------------------------------------------------------- detail dicts (one source for page and API)
def _mk(kind, eid, title, chips=(), fields=(), **extra) -> dict:
    d = {"schema": SCHEMA, "kind": kind, "id": eid, "title": title, "chips": [list(c) for c in chips],
         "fields": [list(f) for f in fields], "n_images": 0, "thumbs": [], "page": 1, "pages": 1}
    d.update(extra)
    return d


def _f(label, value, href=None):
    return (label, value, href) if value not in (None, "", []) else None


def _fields(*items):
    return [i for i in items if i]


def _thumbs(kind, eid, files, page=None, idxs=None) -> dict:
    n = len(files)
    pages = max(1, -(-n // PAGE_N))
    if idxs is None:
        page = min(max(page or 1, 1), pages)
        idxs = range((page - 1) * PAGE_N, min(n, page * PAGE_N))
    else:
        page = 1
    base = f"{PREFIX}/thumb/{kind}/{quote(str(eid), safe='')}"
    return {"n_images": n, "page": page, "pages": pages, "thumbs": [{"n": i, "url": f"{base}/{i}.jpg"} for i in idxs]}


def evenly(n, k):
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def detail_session(S, rec, roots) -> dict:
    sid = rec["id"]
    files = session_files(sid, roots)
    dz = _d(rec.get("defocus_um"))
    ds = _d(rec.get("dose"))
    px = [x for x in _l(rec.get("pixel_size_a")) if _num(x) is not None]
    om = [x for x in _l(rec.get("other_magnification")) if _num(x) is not None]
    acq = _d(rec.get("acq_dates"))
    f = _fields(
        _f("Date", _short_date(rec.get("date"))),
        _f("Instrument", _s(rec.get("instrument"))),
        _f("Magnification", mags(rec)),
        _f("Pixel size", " and ".join(f"{fnum(x, 3)} Å/px" for x in px)),
        _f("Defocus", f"median {fnum(dz.get('median'))} µm (range {fnum(dz.get('min'))} to {fnum(dz.get('max'))})" if dz else None),
        _f("Dose", f"median {fnum(ds.get('median'))} e/Å² (range {fnum(ds.get('min'))} to {fnum(ds.get('max'))})" if ds else None),
        _f("Voltage", " and ".join(f"{fnum(x, 0)} kV" for x in _l(rec.get("voltage_kv")) if _num(x) is not None)),
        _f("Imaged", f"{acq['first'].replace('T', ' ')[:16]} to {acq['last'].replace('T', ' ')[:16]}" if acq.get("first") and acq.get("last") else None),
        _f("Imaging files", str(rec["n_imaging"]) if _num(rec.get("n_imaging")) is not None else None),
        _f("Other files", (f"{rec['n_other']} below 5,000×" + (" (" + ", ".join(f"{x:,}×" for x in om) + ")" if om else "")) if _num(rec.get("n_other")) else (
            ", ".join(f"{x:,}×" for x in om) if om else None)),
        _f("Processed images", str(len(files)) if files else None))
    slots = sorted(set(_d(rec.get("sample_labels"))) | set(_d(rec.get("session_grids"))), key=_slot)
    imgs_by_slot: dict = {}
    for im in S["image"].values():
        if im.get("session") == sid and im.get("session_grid") is not None:
            imgs_by_slot[str(im["session_grid"])] = imgs_by_slot.get(str(im["session_grid"]), 0) + 1
    grids = []
    for sl in slots:
        gid = _d(rec.get("session_grids")).get(sl)
        g = S["grid"].get(gid) if gid else None
        n = len(session_files(sid, roots, f"grid{sl}_".lower())) if files else 0
        grids.append({"slot": sl, "label": _s(_d(rec.get("sample_labels")).get(sl)) or (g or {}).get("sample_label"),
                      "grid": gid if g else None, "grid_id_recorded": gid, "grid_type": (g or {}).get("grid_type"),
                      "n_images": n or imgs_by_slot.get(sl, 0)})
    d = _mk("session", sid, f"Session {sid}", [(rec["status"], "")] if _s(rec.get("status")) else [], f,
            grids=grids, acquisition={k: rec.get(k) for k in (
                "instrument", "magnification", "pixel_size_a", "defocus_um", "dose", "voltage_kv", "binning",
                "acq_dates", "n_mdoc", "n_imaging", "n_other", "other_magnification")},
            gap=_s(rec.get("acquisition_gap")))
    d.update(_thumbs("session", sid, files, idxs=evenly(len(files), STRIP_N)))
    return d


def grid_files(S, gid, roots) -> list:
    pairs = set()
    for sid, s in S["session"].items():
        for sl, g in _d(s.get("session_grids")).items():
            if g == gid:
                pairs.add((sid, str(sl)))
    for im in S["image"].values():
        if im.get("grid") == gid and im.get("session") and im.get("session_grid") is not None:
            pairs.add((str(im["session"]), str(im["session_grid"])))
    out = []
    for sid, sl in sorted(pairs):
        out += session_files(sid, roots, f"grid{sl}_".lower())
    return out


def detail_grid(S, rec, roots, page) -> dict:
    gid = rec["id"]
    sess = sorted({sid for sid, s in S["session"].items() if gid in _d(s.get("session_grids")).values()} |
                  {im["session"] for im in S["image"].values() if im.get("grid") == gid and im.get("session")})
    files = grid_files(S, gid, roots)
    rb = rec.get("replaced_by")
    f = _fields(_f("Sample", _s(rec.get("sample_label"))), _f("Grid type", _s(rec.get("grid_type"))),
                _f("Project", _s(rec.get("project"))), _f("Prepared", _short_date(rec.get("prep_date"))),
                _f("Replicate", str(rec["replicate"]) if rec.get("replicate") is not None else None),
                _f("Glow discharge", _s(rec.get("glow_discharge"))), _f("Storage", _s(rec.get("storage"))),
                _f("Montages", str(rec["n_montages"]) if rec.get("n_montages") is not None else None),
                _f("Notes", _s(rec.get("notes"))), _f("Flags", _s(rec.get("flags")) if isinstance(rec.get("flags"), str) else None),
                _f("Replaced by", f"Grid {rb}" if _s(rb) else None, ref("grid", rb) if _s(rb) and rb in S["grid"] else None))
    d = _mk("grid", gid, f"Grid {gid}", [("replaced", "warn")] if _s(rb) else [], f,
            sessions=[{"id": s, "url": ref("session", s)} for s in sess])
    d.update(_thumbs("grid", gid, files, page))
    return d


def metrics_of(rec) -> dict:
    m = rec.get("metrics") or rec.get("scores")
    return m if isinstance(m, dict) else {}


def detail_model(S, rec) -> dict:
    mid = rec["id"]
    names = {mid, *[a for a in _l(rec.get("aliases")) if isinstance(a, str)]}
    checks = [{"id": c["id"], "session": c.get("session"), "n": c.get("n"), "status": c.get("status"),
               "metrics": metrics_of(c)} for c in sorted(S["inference_check"].values(), key=lambda c: c["id"])
              if c.get("model") in names]
    prod = rec.get("status") == "production"
    ds = _s(rec.get("dataset"))
    init = _s(rec.get("init_from"))
    f = _fields(_f("Status", _s(rec.get("status"))),
                _f("Trained on", ds, ref("dataset", ds) if ds in S["dataset"] else None),
                _f("Started from", init, ref("model", init) if init in S["model"] else None),
                _f("Classes", ", ".join(map(str, _l(rec.get("classes")))) or None),
                _f("Image size", f"{rec['imgsz']} px" if rec.get("imgsz") else None),
                _f("Detection threshold", fnum(rec.get("conf")) if _num(rec.get("conf")) is not None else None),
                _f("Registered", _short_date(rec.get("logged"))),
                _f("Weights fingerprint", (rec.get("weights_sha256") or "")[:12] or None),
                _f("Note", _s(rec.get("note"))))
    return _mk("model", mid, mid, [("production", "prod")] if prod else ([(rec["status"], "")] if _s(rec.get("status")) else []),
               f, metrics=metrics_of(rec), checks=checks, production=prod)


def package_for(S, pid, can_see, user, roots):
    """-> (files, record or None) for a review package id: nothing when hidden or not found."""
    r = S["review_package"].get(pid) or {}
    if r and not visible(can_see, user, r):
        return [], r
    pdir = package_dir(pid, r, roots)
    return (package_files(pdir, roots) if pdir else []), r


def set_files(S, sid, can_see, user, roots) -> list:
    e, r = find_set(S, sid)
    if not r or not visible(can_see, user, r):
        return []
    if e == "prediction_set":
        return prediction_files(sid, roots)
    if e == "review_package":
        return package_files(pd, roots) if (pd := package_dir(sid, r, roots)) else []
    return []


def detail_images(S, e, rec, can_see, user, roots, page) -> dict:
    sid = rec["id"]
    files = set_files(S, sid, can_see, user, roots)
    model = _s(rec.get("model"))
    src = _d(rec.get("source"))
    n = rec.get("n_images")
    if n is None and isinstance(rec.get("counts"), dict):
        n = sum(_num(_d(v).get("images")) or 0 for v in rec["counts"].values())
    pkg_of = _s(rec.get("review_package"))
    f = _fields(_f("Kind", set_kind_label(e)), _f("Images", str(n) if n is not None else None),
                _f("Status", _s(rec.get("status"))), _f("Created", _short_date(_when(rec))),
                _f("Model", model, ref("model", model) if model in S["model"] else None),
                _f("Detection threshold", fnum(rec.get("conf")) if _num(rec.get("conf")) is not None else None),
                _f("Source", (f"session {src['session']}" if src.get("session") else _s(src.get("kind"))) if src else None,
                   ref("session", src["session"]) if src.get("session") in S["session"] else None),
                _f("Reviewers", ", ".join(map(str, _l(rec.get("reviewers")))) or None),
                _f("Edits recorded", str(rec["n_edit_records"]) if rec.get("n_edit_records") is not None else None),
                _f("Starts from", _s(rec.get("base")) or _s(rec.get("derived_from"))),
                _f("Splits", split_counts(rec)), _f("Classes", ", ".join(map(str, _l(rec.get("classes")))) or None),
                _f("Review package", pkg_of, ref("images", pkg_of) if pkg_of in S["review_package"] else None))
    d = _mk("images", sid, sid, [(set_kind_label(e), "acc")], f, set_kind=e)
    if e == "roboflow_export":
        d["note"] = "Thumbnails are not shown for Roboflow exports."
    d.update(_thumbs("images", sid, files, page))
    return d


def detail_labels(S, rec, can_see, user, roots, page) -> dict:
    lid = rec["id"]
    pkg = _s(rec.get("pkg"))
    files, _ = package_for(S, pkg, can_see, user, roots) if pkg else ([], None)
    txt, tone = review_chip(rec)
    built = sorted(d["id"] for d in S["dataset"].values() if d.get("label_version") == lid)
    ops = {k: v for k, v in _d(rec.get("counts_by_op")).items() if _num(v)}
    sup = _s(rec.get("supersedes"))
    f = _fields(_f("Status", _s(rec.get("status"))), _f("Checking", txt),
                _f("Images changed", str(rec["n_images_applied"]) if rec.get("n_images_applied") is not None else None),
                _f("Changes", ", ".join(f"{k} {v}" for k, v in ops.items()) or None),
                _f("Starts from", _s(rec.get("base"))),
                _f("Review package", pkg, ref("images", pkg) if pkg in S["review_package"] else None),
                _f("Replaces", sup, ref("labels", sup) if sup in S["label_version"] else None),
                _f("Created", _short_date(_when(rec))), _f("Note", _s(rec.get("note"))))
    d = _mk("labels", lid, lid, [(txt, tone)] + ([(rec["status"], "")] if _s(rec.get("status")) else []), f,
            datasets=[{"id": i, "url": ref("dataset", i)} for i in built])
    d.update(_thumbs("labels", lid, files, page))
    return d


def dataset_files(S, rec, can_see, user, roots) -> list:
    lv = S["label_version"].get(rec.get("label_version") or "")
    pkg = _s((lv or {}).get("pkg"))
    return package_for(S, pkg, can_see, user, roots)[0] if pkg else []


def detail_dataset(S, rec, can_see, user, roots, page) -> dict:
    did = rec["id"]
    files = dataset_files(S, rec, can_see, user, roots)
    src = _d(rec.get("source"))
    lv = _s(rec.get("label_version"))
    models = sorted(m["id"] for m in S["model"].values() if m.get("dataset") == did)
    cm = _d(rec.get("class_map"))
    f = _fields(_f("Status", _s(rec.get("status"))), _f("Created", _short_date(_when(rec))),
                _f("Splits (images)", split_counts(rec)),
                _f("Classes", ", ".join(map(str, _l(rec.get("classes")))) or None),
                _f("Merged classes", ", ".join(f"{k} as {v}" for k, v in cm.items()) or None),
                _f("Label version", lv, ref("labels", lv) if lv in S["label_version"] else None),
                _f("Source", (_s(src.get("id")) or _s(src.get("kind"))) if src else None,
                   ref("images", src["id"]) if _s(src.get("id")) and find_set(S, src["id"])[1] else None),
                _f("Split rule", f"{_d(rec.get('split')).get('method')}, seed {_d(rec.get('split')).get('seed')}" if _d(rec.get("split")).get("method") else None),
                _f("Note", _s(rec.get("note"))))
    d = _mk("dataset", did, did, [(rec["status"], "")] if _s(rec.get("status")) else [], f,
            models=[{"id": m, "url": ref("model", m)} for m in models])
    if not files:
        d["note"] = "No example images are linked to this dataset."
    d.update(_thumbs("dataset", did, files, page))
    return d


def detail(S, kind, eid, can_see, user, roots, page=1):
    """-> detail dict, or None when unknown / hidden."""
    if kind == "session":
        r = S["session"].get(eid)
        return detail_session(S, r, roots) if r else None
    if kind == "grid":
        r = S["grid"].get(eid)
        return detail_grid(S, r, roots, page) if r else None
    if kind == "model":
        r = S["model"].get(eid)
        return detail_model(S, r) if r else None
    if kind == "dataset":
        r = S["dataset"].get(eid)
        return detail_dataset(S, r, can_see, user, roots, page) if r else None
    if kind == "labels":
        r = S["label_version"].get(eid)
        return detail_labels(S, r, can_see, user, roots, page) if r else None
    if kind == "images":
        e, r = find_set(S, eid)
        return detail_images(S, e, r, can_see, user, roots, page) if r and visible(can_see, user, r) else None
    return None


def thumb_files(S, kind, eid, can_see, user, roots) -> list:
    """The ordered image files behind /data/thumb/<kind>/<id>/<n>.jpg, same order as the detail pages use."""
    if kind == "session":
        return session_files(eid, roots) if eid in S["session"] else []
    if kind == "grid":
        return grid_files(S, eid, roots) if eid in S["grid"] else []
    if kind == "images":
        return set_files(S, eid, can_see, user, roots)
    if kind == "labels":
        r = S["label_version"].get(eid)
        pkg = _s((r or {}).get("pkg"))
        return package_for(S, pkg, can_see, user, roots)[0] if pkg else []
    if kind == "dataset":
        r = S["dataset"].get(eid)
        return dataset_files(S, r, can_see, user, roots) if r else []
    return []


# ---------------------------------------------------------------- thumbnails
def make_thumb(path: Path, cache_dir: Path):
    """-> JPEG bytes (cached on disk by path + mtime + size + size), or None when the file can't be read."""
    from PIL import Image
    try:
        st = path.stat()
        key = hashlib.sha256(f"{path}|{st.st_mtime_ns}|{st.st_size}|{THUMB_PX}".encode("utf-8", "replace")).hexdigest()[:40]
        cf = cache_dir / key[:2] / f"{key}.jpg"
        try:
            return cf.read_bytes()
        except OSError:
            pass
        with Image.open(path) as im:
            im.draft("RGB", (THUMB_PX * 2, THUMB_PX * 2))
            if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                im = im.convert("I").point(lambda v: v / 256).convert("L")
            elif im.mode not in ("L", "RGB"):
                im = im.convert("RGB")
            im.thumbnail((THUMB_PX, THUMB_PX))
            import io
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80, optimize=True)
        data = buf.getvalue()
        try:
            cf.parent.mkdir(parents=True, exist_ok=True)
            tmp = cf.with_name(cf.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, cf)
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
    lis = []
    for r in rows:
        tags = "".join(chip(t, tone) for t, tone in r["tags"])
        lis.append(f'<li class="cj-data-row"><span class="cj-data-top"><a href="{esc(r["url"])}">{esc(r["title"])}</a>{tags}</span>'
                   + (f'<span class="cj-mut">{esc(r["sub"])}</span>' if r["sub"] else "") + "</li>")
    return '<ul class="cj-data-rows">' + "".join(lis) + "</ul>"


def search_form(q="") -> str:
    return ('<form class="cj-data-search" role="search" method="get" action="/data/search">'
            '<input type="search" name="q" maxlength="100" placeholder="Search by id or sample label" '
            f'aria-label="Search the catalog" value="{esc(q, quote=True)}"><button type="submit">Search</button></form>')


def banner(err) -> str:
    return (f'<p class="cj-data-banner" role="status">The catalog could not be read just now, so the lists below may be empty. '
            f'({esc(err)})</p>') if err else ""


def crumb(*parts) -> str:
    return '<p class="cj-data-crumb cj-mut">' + " / ".join(f'<a href="{esc(h)}">{esc(t)}</a>' for t, h in parts) + "</p>"


def render_overview(ov, err) -> str:
    cards = []
    for c in ov["cards"]:
        cls = "cj-card cj-data-card"
        cards.append(f'<section class="{cls}" id="cj-data-c-{esc(c["kind"])}"><h2><span>{esc(c["title"])}</span>'
                     f'<span class="cj-data-n">{c["count"]}</span></h2>' + render_rows(c["rows"])
                     + (f'<p class="cj-data-more"><a href="{esc(c["all"])}">See all {c["count"]}</a></p>' if c["count"] > len(c["rows"]) else "")
                     + "</section>")
    g = ov["counts"].get("grids", 0)
    return ('<h1>Data</h1><p class="cj-mut cj-data-sub">What data exists: imaging sessions, image sets, labels, datasets and models. '
            f'<a href="{PREFIX}/list/grids">{g} grids</a> are listed under their sessions.</p>' + banner(err) + search_form()
            + '<div class="cj-data-cards">' + "".join(cards) + "</div>"
            + '<p class="cj-mut cj-data-foot">Read only. Open items are never changed from here.</p>')


def render_search(res, err) -> str:
    body = ""
    if not res["q"]:
        body = '<p class="cj-mut">Type an id or a sample label.</p>'
    elif not res["groups"]:
        body = f'<p class="cj-mut">Nothing found for &ldquo;{esc(res["q"])}&rdquo;.</p>'
    for g in res["groups"]:
        body += (f'<section class="cj-card cj-data-card"><h2><span>{esc(g["title"])}</span><span class="cj-data-n">{g["count"]}</span></h2>'
                 + render_rows(g["rows"]) + "</section>")
    return (crumb(("Data", PREFIX + "/")) + "<h1>Search</h1>" + banner(err) + search_form(res["q"])
            + '<div class="cj-data-cards">' + body + "</div>")


def render_list(kind, title, rows, err) -> str:
    return (crumb(("Data", PREFIX + "/")) + f"<h1>{esc(title)}</h1>"
            f'<p class="cj-mut cj-data-sub">{len(rows)} in all.</p>' + banner(err) + '<div class="cj-card">' + render_rows(rows) + "</div>")


def kv(fields) -> str:
    if not fields:
        return ""
    items = []
    for label, value, href in fields:
        v = f'<a href="{esc(href)}">{esc(value)}</a>' if href else esc(value)
        items.append(f"<div><dt>{esc(label)}</dt><dd>{v}</dd></div>")
    return '<dl class="cj-data-kv">' + "".join(items) + "</dl>"


def thumb_list(d, strip=False) -> str:
    if not d["thumbs"]:
        return '<p class="cj-mut cj-data-empty">' + esc(d.get("note") or "No images found for this yet.") + "</p>"
    n = d["n_images"]
    lis = "".join(f'<li><img class="cj-data-img" src="{esc(t["url"])}" alt="Image {t["n"] + 1} of {n}" loading="lazy" '
                  f'decoding="async" width="{THUMB_PX}" height="{THUMB_PX}"></li>' for t in d["thumbs"])
    out = f'<ul class="{"cj-data-strip" if strip else "cj-data-thumbs"}">{lis}</ul>'
    if not strip and d["pages"] > 1:
        base = ref(d["kind"], d["id"])
        a, b = (d["page"] - 1) * PAGE_N + 1, min(n, d["page"] * PAGE_N)
        prev = f'<a href="{esc(base)}?page={d["page"] - 1}" rel="prev">Previous</a>' if d["page"] > 1 else '<span class="cj-data-off">Previous</span>'
        nxt = f'<a href="{esc(base)}?page={d["page"] + 1}" rel="next">Next</a>' if d["page"] < d["pages"] else '<span class="cj-data-off">Next</span>'
        out += f'<nav class="cj-data-pager" aria-label="Pages">{prev}<span class="cj-mut">Showing {a}-{b} of {n}</span>{nxt}</nav>'
    return out


def render_metrics(m) -> str:
    rows = []
    for k, v in m.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                rows.append((f"{k} {k2}", v2))
        else:
            rows.append((k, v))
    if not rows:
        return ""
    return '<table class="cj-data-metrics"><tbody>' + "".join(
        f"<tr><th scope=\"row\">{esc(str(k))}</th><td>{esc(fnum(v, 3) if _num(v) is not None else str(v))}</td></tr>" for k, v in rows) + "</tbody></table>"


def render_detail(d, err) -> str:
    kind = d["kind"]
    parent = {"session": ("Sessions", "sessions"), "grid": ("Grids", "grids"), "images": ("Image sets", "images"),
              "labels": ("Label versions", "labels"), "dataset": ("Datasets", "datasets"), "model": ("Models", "models")}[kind]
    out = [crumb(("Data", PREFIX + "/"), (parent[0], f"{PREFIX}/list/{parent[1]}")),
           f'<div class="cj-data-top"><h1 class="cj-wrap">{esc(d["title"])}</h1>' + "".join(chip(t, tone) for t, tone in d["chips"]) + "</div>",
           banner(err), '<div class="cj-card' + (" cj-data-hl" if d.get("production") else "") + '">' + kv(d["fields"]) + "</div>"]
    if d.get("gap"):
        out.append(f'<p class="cj-mut">Acquisition details are not available: {esc(d["gap"])}</p>')
    if kind == "session":
        out.append('<h2 class="cj-sec">Grids</h2>')
        if d["grids"]:
            lis = []
            for g in d["grids"]:
                name = f'Slot {g["slot"]}'
                head = (f'<a href="{esc(ref("grid", g["grid"]))}">Grid {esc(g["grid"])}</a>' if g["grid"] else esc(name))
                bits = [g["label"], g["grid_type"], f'{g["n_images"]} images' if g["n_images"] else None, name if g["grid"] else
                        ("grid number not recorded" if not g["grid_id_recorded"] else f'grid {g["grid_id_recorded"]} not in the list')]
                lis.append(f'<li><span class="cj-data-top"><b>{head}</b></span><span class="cj-mut">{esc(" · ".join(b for b in bits if b))}</span></li>')
            out.append('<ul class="cj-data-grids">' + "".join(lis) + "</ul>")
        else:
            out.append('<p class="cj-mut">No grids recorded for this session.</p>')
        out.append(f'<h2 class="cj-sec">Images ({d["n_images"]} processed)</h2>' + thumb_list(d, strip=True))
    elif kind == "grid":
        if d["sessions"]:
            out.append('<h2 class="cj-sec">Imaged in</h2><p>' + ", ".join(
                f'<a href="{esc(s["url"])}">Session {esc(s["id"])}</a>' for s in d["sessions"]) + "</p>")
        out.append(f'<h2 class="cj-sec">Images ({d["n_images"]})</h2>' + thumb_list(d))
    elif kind == "model":
        out.append('<h2 class="cj-sec">Scores</h2>')
        mt = render_metrics(d["metrics"])
        out.append('<div class="cj-card">' + mt + "</div>" if mt else "")
        if d["checks"]:
            lis = []
            for c in d["checks"]:
                sub = " · ".join(str(x) for x in (f"session {c['session']}" if c.get("session") else None,
                                                 f"{c['n']} images" if c.get("n") is not None else None, c.get("status")) if x)
                lis.append(f'<li class="cj-data-row"><span>{esc(c["id"])}</span><span class="cj-mut">{esc(sub)}</span>' + render_metrics(c["metrics"]) + "</li>")
            out.append('<div class="cj-card"><h2>Checks on new images</h2><ul class="cj-data-rows">' + "".join(lis) + "</ul></div>")
        if not mt and not d["checks"]:
            out.append('<p class="cj-mut">No scores are recorded in the registry for this model yet.</p>')
    elif kind == "dataset":
        if d["models"]:
            out.append('<h2 class="cj-sec">Models trained on it</h2><p>' + ", ".join(
                f'<a href="{esc(m["url"])}">{esc(m["id"])}</a>' for m in d["models"]) + "</p>")
        out.append(f'<h2 class="cj-sec">Example images ({d["n_images"]})</h2>' + thumb_list(d))
    elif kind == "labels":
        if d["datasets"]:
            out.append('<h2 class="cj-sec">Datasets built from it</h2><p>' + ", ".join(
                f'<a href="{esc(m["url"])}">{esc(m["id"])}</a>' for m in d["datasets"]) + "</p>")
        out.append(f'<h2 class="cj-sec">Images ({d["n_images"]})</h2>' + thumb_list(d))
    else:
        out.append(f'<h2 class="cj-sec">Images ({d["n_images"]})</h2>' + thumb_list(d))
    return "".join(out)


def page_body(content: str) -> str:
    """data.html with the content put in ONE replace (text inside the content is never re-scanned)."""
    return TEMPLATE.read_text(encoding="utf-8").replace("{{content}}", content, 1)


# ---------------------------------------------------------------- routes
def routes(ctx) -> list:
    def state(request: Request):
        S, err = read_registry(getattr(ctx.cfg, "registry", None))
        return S, err, ctx.user(request)

    def pagenum(request: Request) -> int:
        v = request.query_params.get("page", "1")
        return int(v) if v.isdigit() and len(v) < 6 else 1

    def html_page(request, title, content, status=200):
        r = ctx.page(request, title, page_body(content), active="data")
        r.status_code = status
        return r

    def missing(request, what="That item"):
        return html_page(request, "Not found", crumb(("Data", PREFIX + "/")) + "<h1>Not found</h1>"
                         f'<p class="cj-mut">{esc(what)} is not in the catalog, or you do not have access to it.</p>', 404)

    def api_missing():
        return ctx.json({"error": "not found"}, 404)

    def index(request: Request):
        S, err, u = state(request)
        return html_page(request, "Data", render_overview(overview(S, ctx.can_see, u), err))

    def find(request: Request):
        S, err, u = state(request)
        return html_page(request, "Search", render_search(search(S, request.query_params.get("q", ""), ctx.can_see, u), err))

    def listing(request: Request):
        kind = request.path_params["kind"]
        if kind not in KINDS:
            return missing(request, "That list")
        S, err, u = state(request)
        title = {"sessions": "Sessions", "grids": "Grids", "images": "Image sets", "labels": "Label versions",
                 "datasets": "Datasets", "models": "Models"}[kind]
        return html_page(request, title, render_list(kind, title, rows_for(S, kind, ctx.can_see, u), err))

    def item(kind):
        def go(request: Request):
            S, err, u = state(request)
            d = detail(S, kind, request.path_params["eid"], ctx.can_see, u, known_roots(), pagenum(request))
            if d is None:
                return missing(request)
            return html_page(request, d["title"], render_detail(d, err))
        return go

    def thumb(request: Request):
        kind, eid, n = request.path_params["kind"], request.path_params["eid"], request.path_params["n"]
        if kind not in DETAIL_KINDS or not n.isdigit() or len(n) > 7 or not eid or len(eid) > 200 or "\x00" in eid:
            return Response("not found", status_code=404, media_type="text/plain")
        S, _, u = state(request)
        roots = known_roots()
        files = thumb_files(S, kind, eid, ctx.can_see, u, roots)
        i = int(n)
        if i >= len(files) or not inside(files[i], roots):
            return Response("not found", status_code=404, media_type="text/plain")
        data = make_thumb(files[i], Path(ctx.cfg.state_dir) / "data_thumbs")
        if data is None:
            return Response("not found", status_code=404, media_type="text/plain")
        r = Response(data, media_type="image/jpeg")
        r.headers["Cache-Control"] = "private, max-age=3600"
        r.headers["X-Content-Type-Options"] = "nosniff"
        return r

    def api_overview(request: Request):
        S, err, u = state(request)
        return ctx.json(overview(S, ctx.can_see, u) | {"error": err})

    def api_search(request: Request):
        S, err, u = state(request)
        return ctx.json(search(S, request.query_params.get("q", ""), ctx.can_see, u))

    def api_list(request: Request):
        kind = request.path_params["kind"]
        if kind not in KINDS:
            return api_missing()
        S, err, u = state(request)
        return ctx.json({"schema": SCHEMA, "kind": kind, "rows": rows_for(S, kind, ctx.can_see, u), "error": err})

    def api_item(request: Request):
        kind, eid = request.path_params["kind"], request.path_params["eid"]
        S, err, u = state(request)
        d = detail(S, kind, eid, ctx.can_see, u, known_roots(), pagenum(request))
        return ctx.json(d) if d is not None else api_missing()

    rs = [Route(PREFIX + "/", index, methods=["GET"]),
          Route(PREFIX + "/search", find, methods=["GET"]),
          Route(PREFIX + "/list/{kind}", listing, methods=["GET"])]
    for k in DETAIL_KINDS:
        rs.append(Route(PREFIX + f"/{k}/{{eid}}", item(k), methods=["GET"]))
    rs += [Route(PREFIX + "/thumb/{kind}/{eid}/{n}.jpg", thumb, methods=["GET"]),
           Route(PREFIX + "/api/overview", api_overview, methods=["GET"]),
           Route(PREFIX + "/api/search", api_search, methods=["GET"]),
           Route(PREFIX + "/api/list/{kind}", api_list, methods=["GET"]),
           Route(PREFIX + "/api/{kind}/{eid}", api_item, methods=["GET"])]
    return rs
