"""
Image metadata for the review tool (round 12, feedback #20): GET <pkg>/api/meta on review_server.py and on
multi_review_server.py (the multi server hands every request to review_server's handler, so one route serves both).

Everything is derived at request time from folders that already exist, read only, cached by each file's mtime and
size; nothing is written into a package, a prediction set or a conversion. A value that can't be found is left out
(the page shows "not recorded"); a missing or broken file never fails the request.

    package_meta(pkg) -> {"schema": "review_meta/1",
                          "package": {id, kind, prediction_set?, model?, source_kind?, session?, conversion?,
                                      source_folder?, recipe? {id?, params?, variant?, overrides?, gpu?},
                                      session_acquisition?, name?, note?, provenance?},
                          "images": {<data.js file>: {split, sample?, orig_name?, grid?, montage?, tile? {row, col,
                                     piece?} | {x0, y0, size?}, nm_per_px?, mag?, pixel_A?, defocus_um?, dose?,
                                     exposure_s?, acquired?, n_pieces?, parent_image?}}}

Where each value comes from (a prediction package, annotation/prediction_package.py):
  <pkg>/prediction_package.json   ps_dir, and per image id its data.js file and real sample label
  <ps>/images.csv                 orig_name, sample_label, nm_per_px, source ("session:<id>" | "upload")
  <ps>/request.json               source {kind, session, dir}, name, note, provenance (round 11b), source_manifest
  <source dir>/MANIFEST.csv       per output_name: montage, grid, sample_label (jobs/convert_raw.py and the processing
                                  repo's export both write it)
  <conversion>/convert.json       a portal conversion (the source dir is <conversion>/tiles): recipe id,
                                  params_effective (clip_lo_pct, clip_hi_pct, bandpass_nm, denoise), params_overrides,
                                  variant, gpu
  <conversion>/acquisition.json   per montage: mag, pixel_A, defocus_um, dose, acquired, exposure_s, n_pieces; the
                                  session summary
  <conversion>/montages/layout.json  per output_name: the tile's row / col in its montage
  <ps>/source_manifest.csv        sets made by `predict new --from-dir --manifest`: parent_image, x0, y0, tile_px
  the registry's session record   the session's acquisition summary, when no acquisition.json has one (a set made
                                  from the processing repo's own tile folder)
Montage and row / col fall back to the tile's name (parse_tile_name, the processing repo's naming). A package with
no prediction set behind it (a lab / compare package) gets split only, plus what its file names give.

Paths are never sent: only names (session, conversion id, recipe id, the source folder's own name, a provenance
script's file name).
"""
from __future__ import annotations

import csv
import json
import math
import re
import threading
from pathlib import Path

SCHEMA = "review_meta/1"
META = "prediction_package.json"
RECORD = "review_package.json"
# export_detector_tiles.training_name (the processing repo): g<grid>_<label>__<source stem>__denoised_nobar_<A>Apx.png,
# where the source stem is cryoem_core.pipeline's tile name <montage basename>_r<row>c<col>_t<idx:02d>
TILE_RE = re.compile(r"^(?P<montage>.+?)_r(?P<row>\d+)c(?P<col>\d+)(?:_t(?P<piece>\d+))?$")
GRID_RE = re.compile(r"^g(?P<grid>\d+)_")
IMG_EXT = re.compile(r"(?:\.(?:png|jpe?g|tiff?))+$", re.I)
ACQ_FIELDS = ("mag", "pixel_A", "defocus_um", "dose", "exposure_s", "acquired", "n_pieces")
PARAM_KEYS = ("clip_lo_pct", "clip_hi_pct", "bandpass_nm", "denoise")      # jobs/convert_raw.py PARAM_KEYS
SESSION_ACQ = ("instrument", "magnification", "pixel_size_a", "defocus_um", "target_defocus_um", "dose", "voltage_kv",
               "binning", "acq_dates")
PROV_KEYS = ("derived_from", "source_images", "steps", "tiling", "script", "git")
MANIFEST_NAMES = ("output_name", "image", "file_name", "filename", "name")   # jobs/predict_set.py's manifest columns
TEXT_MAX = 300

_cache: dict = {}
_lock = threading.Lock()
CACHE_MAX = 64


# ---------------------------------------------------------------- small readers (cached by mtime + size)
def _sig(p: Path):
    try:
        st = Path(p).stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _cached(kind: str, p: Path, load):
    """load(p), kept while the file's mtime and size stay the same; None for a missing or unreadable file."""
    p = Path(p)
    sig = _sig(p)
    if sig is None:
        return None
    key = (kind, str(p))
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == sig:
            return hit[1]
    try:
        val = load(p)
    except (OSError, ValueError, UnicodeError, csv.Error, KeyError, TypeError, AttributeError, IndexError):
        val = None
    with _lock:
        if len(_cache) >= CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[key] = (sig, val)
    return val


def _json_file(p: Path):
    val = json.loads(Path(p).read_text(encoding="utf-8-sig"))
    return val if isinstance(val, dict) else None


def _csv_rows(p: Path):
    with open(p, encoding="utf-8-sig", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _datajs(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))


def read_json(p):
    return _cached("json", p, _json_file)


def read_csv(p):
    return _cached("csv", p, _csv_rows) or []


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and not isinstance(v, bool) else None


def _text(v, n=TEXT_MAX):
    return v.strip()[:n] if isinstance(v, str) and v.strip() else None


def _name_only(v):
    """The last part of a path (either slash), never the path itself."""
    t = _text(v, 1000)
    return re.split(r"[\\/]+", t.rstrip("\\/"))[-1][:TEXT_MAX] if t else None


# ---------------------------------------------------------------- the tile's name
def parse_tile_name(name) -> dict:
    """{grid?, montage, row, col, piece?} from a tile name, {} when it doesn't follow the processing repo's naming.

    Detector tiles (export_detector_tiles.training_name): g<grid>_<label>__<montage>_r<row>c<col>_t<nn>__<variant>.png;
    the label never holds "__" (runs of other characters become one "_"), so the source stem is everything between
    the first and the last "__". A delivered tile (MANIFEST source_file, cryoem_core.pipeline) is the stem alone:
    <montage>_r<row>c<col>_t<nn>.jpg. A Roboflow copy (..._png.rf.<hash>.jpg) keeps the same middle part."""
    if not isinstance(name, str) or not name.strip():
        return {}
    stem = IMG_EXT.sub("", re.split(r"[\\/]", name.strip())[-1])
    parts = stem.split("__")
    cands = ["__".join(parts[1:-1])] if len(parts) >= 3 else []
    cands += parts if len(parts) < 3 else []
    out = {}
    for c in cands:
        m = TILE_RE.match(c)
        if m:
            out = {"montage": m["montage"], "row": int(m["row"]), "col": int(m["col"])}
            if m["piece"] is not None:
                out["piece"] = int(m["piece"])
            break
    g = GRID_RE.match(parts[0]) if len(parts) >= 2 else None
    if out and g:
        out["grid"] = int(g["grid"])
    return out


# ---------------------------------------------------------------- the prediction set behind a package
def _ps_dir(pkg: Path, meta: dict) -> Path | None:
    """The prediction set folder: prediction_package.json ps_dir, else <review root>/../predict/<ps> (the laptop
    layout), else None."""
    ps = meta.get("prediction_set")
    cands = []
    if isinstance(meta.get("ps_dir"), str) and meta["ps_dir"].strip():
        cands.append(Path(meta["ps_dir"]))
    if isinstance(ps, str) and re.fullmatch(r"ps_\d{8}_[0-9a-f]{6}", ps):
        cands.append(Path(pkg).resolve().parent.parent / "predict" / ps)
    for c in cands:
        try:
            if (c / "images.csv").is_file():
                return c
        except OSError:
            continue
    return None


def _session_of(req: dict, rows: list) -> str | None:
    src = req.get("source") if isinstance(req.get("source"), dict) else {}
    s = _text(src.get("session"), 60)
    if s:
        return s
    tags = {r.get("source") for r in rows if isinstance(r.get("source"), str) and r["source"].startswith("session:")}
    return tags.pop()[8:68] if len(tags) == 1 else None


def _recipe(conv: dict) -> dict | None:
    out = {}
    if _text(conv.get("recipe"), 60):
        out["id"] = conv["recipe"].strip()[:60]
    pe = conv.get("params_effective")
    if isinstance(pe, dict):
        params = {}
        for k in PARAM_KEYS:
            v = pe.get(k)
            if isinstance(v, bool) or v is None:
                params[k] = v
            elif _num(v) is not None:
                params[k] = _num(v)
        if params:
            out["params"] = params
    ov = conv.get("params_overrides")
    if isinstance(ov, dict) and ov:
        out["overrides"] = {k: ov[k] for k in PARAM_KEYS if k in ov and (isinstance(ov[k], bool) or _num(ov[k]) is not None)}
    if _text(conv.get("variant"), 20):
        out["variant"] = conv["variant"].strip()[:20]
    if isinstance(conv.get("gpu"), bool):
        out["gpu"] = conv["gpu"]
    return out or None


def _session_acq(s) -> dict | None:
    if not isinstance(s, dict):
        return None
    out = {k: s[k] for k in SESSION_ACQ if k in s and s[k] not in (None, [], {})}
    try:
        return json.loads(json.dumps(out, allow_nan=False)) if out else None
    except (TypeError, ValueError):
        return None


def _provenance(p) -> dict | None:
    if not isinstance(p, dict):
        return None
    out = {k: p[k] for k in PROV_KEYS if k in p}
    if "script" in out:
        out["script"] = _name_only(out["script"])  # a file name, not a local path
    try:
        return json.loads(json.dumps(out, allow_nan=False)) or None
    except (TypeError, ValueError):
        return None


def registry_session(session: str) -> dict | None:
    """The registry's session record (read only), for its acquisition summary; None when there is no registry or no
    such session. Cached by session.jsonl's mtime."""
    if not session:
        return None
    try:
        from registry import store
        p = Path(store.default_root()) / "session.jsonl"
    except Exception:  # noqa: BLE001  no registry configured here: nothing to add
        return None

    def load(path):
        last = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and isinstance(rec.get("id"), str):
                    last[rec["id"]] = rec
        return last
    recs = _cached("registry_sessions", p, load) or {}
    rec = recs.get(session)
    return rec if isinstance(rec, dict) else None


def _put_num(d: dict, k: str, v):
    x = _num(v)
    if x is not None:
        d[k] = int(x) if k in ("mag", "n_pieces", "grid") and x == int(x) else x


def package_meta(pkg) -> dict:
    """review_meta/1 for one package folder (module doc). Never raises for missing or odd files."""
    pkg = Path(pkg)
    data = _cached("datajs", pkg / "data.js", _datajs) or {}
    imgs = data.get("images") if isinstance(data.get("images"), list) else []
    out_imgs = {}
    for im in imgs:
        if isinstance(im, dict) and isinstance(im.get("file"), str):
            out_imgs[im["file"]] = {"split": im.get("split") if isinstance(im.get("split"), str) else ""}
    info = {"id": data.get("id") if isinstance(data.get("id"), str) else pkg.name,
            "kind": data.get("kind") if isinstance(data.get("kind"), str) else "label"}
    try:
        _prediction_meta(pkg, info, out_imgs)
    except Exception as e:  # noqa: BLE001  one odd file must never cost the reviewer the page (module doc)
        info["error"] = f"metadata partly unavailable ({type(e).__name__})"
    for file, m in out_imgs.items():  # names alone: montage and row / col when nothing better said
        if "montage" not in m:
            t = parse_tile_name(m.get("orig_name") or file)
            if t:
                m.setdefault("montage", t["montage"])
                m.setdefault("tile", {k: t[k] for k in ("row", "col", "piece") if k in t})
                if "grid" in t:
                    m.setdefault("grid", t["grid"])
    return {"schema": SCHEMA, "package": info, "images": out_imgs}


def _prediction_meta(pkg: Path, info: dict, out_imgs: dict):
    meta = read_json(pkg / META)
    if not meta:
        return
    for k in ("prediction_set", "model"):
        if _text(meta.get(k), 80):
            info[k] = meta[k].strip()
    file_of = {}  # image id -> data.js file
    for image, m in (meta.get("images") or {}).items():
        if isinstance(m, dict) and m.get("file") in out_imgs:
            file_of[image] = m["file"]
            if _text(m.get("sample_label"), 200):
                out_imgs[m["file"]]["sample"] = m["sample_label"].strip()[:200]
    ps = _ps_dir(pkg, meta)
    if ps is None:
        return
    rows = read_csv(ps / "images.csv")
    req = read_json(ps / "request.json") or {}
    src = req.get("source") if isinstance(req.get("source"), dict) else {}
    if _text(src.get("kind"), 20):
        info["source_kind"] = src["kind"].strip()
    session = _session_of(req, rows)
    if session:
        info["session"] = session
    for k in ("name", "note"):
        if _text(req.get(k), 600):
            info[k] = req[k].strip()[:600]
    prov = _provenance(req.get("provenance"))
    if prov:
        info["provenance"] = prov

    # the tiles' source folder: a portal conversion (<conversion>/tiles) or the processing repo's tile folder
    man, conv_dir = {}, None
    sdir = Path(src["dir"]) if _text(src.get("dir"), 2000) else None
    if sdir is not None:
        for r in read_csv(sdir / "MANIFEST.csv"):
            if _text(r.get("output_name"), 400):
                man[r["output_name"].strip()] = r
        if sdir.name == "tiles" and (sdir.parent / "convert.json").is_file():
            conv_dir = sdir.parent
        else:
            info["source_folder"] = _name_only(str(sdir))
    acq_by, layout_by = {}, {}
    if conv_dir is not None:
        conv = read_json(conv_dir / "convert.json") or {}
        info["conversion"] = _text(conv.get("id"), 80) or conv_dir.name
        rec = _recipe(conv)
        if rec:
            info["recipe"] = rec
        acq = read_json(conv_dir / "acquisition.json") or {}
        for m in acq.get("montages") or []:
            if isinstance(m, dict) and _text(m.get("montage"), 200):
                acq_by[m["montage"].strip()] = m
        sa = _session_acq(acq.get("session"))
        if sa:
            info["session_acquisition"] = sa
        lay = read_json(conv_dir / "montages" / "layout.json") or {}
        for m in lay.get("montages") or []:
            for t in (m.get("tiles") if isinstance(m, dict) and isinstance(m.get("tiles"), list) else []):
                if isinstance(t, dict) and isinstance(t.get("output_name"), str):
                    layout_by[t["output_name"]] = (m.get("name"), t)
    if "session_acquisition" not in info and session:
        rec = registry_session(session)
        sa = _session_acq(rec) if rec else None
        if sa:
            info["session_acquisition"] = sa
    src_man = {}
    if _text(req.get("source_manifest"), 200):
        for r in read_csv(ps / Path(req["source_manifest"]).name):
            key = next((r[c].strip() for c in MANIFEST_NAMES if _text(r.get(c), 400)), None)
            if key:
                src_man[key] = r

    for r in rows:
        file = file_of.get(r.get("image").strip() if isinstance(r.get("image"), str) else "")
        if file is None:
            continue
        try:
            _image_meta(out_imgs[file], r, man, layout_by, acq_by, src_man)
        except Exception:  # noqa: BLE001  one odd row: that image keeps what it has
            pass


def _image_meta(m: dict, r: dict, man: dict, layout_by: dict, acq_by: dict, src_man: dict):
    """One images.csv row -> its metadata entry m (module doc)."""
    name = _text(r.get("orig_name"), 400)
    if name:
        m["orig_name"] = name
    if "sample" not in m and _text(r.get("sample_label"), 200):
        m["sample"] = r["sample_label"].strip()
    _put_num(m, "nm_per_px", r.get("nm_per_px"))
    if not name:
        return
    row = man.get(name)
    parsed = parse_tile_name(name)
    if row is not None:
        if _text(row.get("montage"), 200):
            m["montage"] = row["montage"].strip()
        _put_num(m, "grid", row.get("grid"))
        if "nm_per_px" not in m:
            _put_num(m, "nm_per_px", row.get("nm_per_px"))
    if "montage" not in m and parsed:
        m["montage"] = parsed["montage"]
    if "grid" not in m and "grid" in parsed:
        m["grid"] = parsed["grid"]
    lay = layout_by.get(name)
    if lay and _num(lay[1].get("row")) is not None and _num(lay[1].get("col")) is not None:
        m["tile"] = {"row": int(_num(lay[1]["row"])), "col": int(_num(lay[1]["col"]))}
        if "piece" in parsed:
            m["tile"]["piece"] = parsed["piece"]
    elif parsed:
        m["tile"] = {k: parsed[k] for k in ("row", "col", "piece") if k in parsed}
    acq = acq_by.get(m.get("montage"))
    if acq:
        for k in ACQ_FIELDS:
            if k == "acquired":
                if _text(acq.get(k), 40):
                    m[k] = acq[k].strip()
            else:
                _put_num(m, k, acq.get(k))
    sm = src_man.get(name)
    if sm:  # a tile cut from a bigger image: the parent image is its "montage", x0 / y0 its place
        parent = _text(sm.get("parent_image"), 200) or _name_only(sm.get("parent_file"))
        if parent:
            m["parent_image"] = parent
            m.setdefault("montage", parent)
        x0, y0 = _num(sm.get("x0")), _num(sm.get("y0"))
        if x0 is not None and y0 is not None and "tile" not in m:
            m["tile"] = {"x0": int(x0), "y0": int(y0)}
            if _num(sm.get("tile_px")) is not None:
                m["tile"]["size"] = int(_num(sm["tile_px"]))
