"""
Predict module (CryoEV portal, journey J1): a person runs the production model on their tiles and gets a prediction set
(BUILD_CONTRACTS.md section 2) to review and count. Raw cryo-EM montages can be converted to tiles first.

    GET  /predict/                        My analyses (admin: everyone's), raw conversions, "New analysis"
    GET  /predict/predict.js              the module's script (portal/static/predict.js; no inline script anywhere)
    GET  /predict/new/                    step 1: what images work; pick a processed session, upload, or convert raw
    GET  /predict/new/session/<key>/      step 2: tiles (all, a count, by sample), nm/px per sample + the
    POST /predict/new/session/<key>/        confirmation tick, model -> create() + start_detached() -> 303 set page
    GET  /predict/new/upload/             upload processed 8-bit PNG/JPEG tiles (caps MAX_FILES x MAX_FILE_BYTES),
    POST /predict/new/upload/               sample label + nm/px + tick. The page's script sends one file per request
                                            (the upload API below); this POST is the no-JavaScript form, for small
                                            sets only (at most NOJS_MAX_BYTES in all; else refused, saying why)
    GET  /predict/new/raw/                admin, advanced: upload raw montages (.mrc + .mrc.mdoc, or .idoc + .tif
                                            tiles; an optional sample key .csv), a session id and a recipe -> convert
    GET  /predict/new/raw/sessions/       admin: raw session folders under the scan roots with no 3d tiles yet
    GET  /predict/new/raw/sessions/<key>/ admin: pick montages of one; POST: convert (the folder is only read)
    GET  /predict/convert/<id>/           one conversion: status (auto-refresh), warnings, then "Analyse these tiles"
    POST /predict/convert/<id>/stop       admin: kill a running converter (or drop a queued one) -> failed "stopped"
    POST /predict/convert/<id>/retry      admin: a failed conversion whose inputs are still there is queued again
    POST /predict/convert/<id>/inputs     admin: delete its uploaded raw files now
    GET  /predict/<ps>/                   the set: status (auto-refresh), Review / Results, spot-check list
    POST /predict/<ps>/review             build the prediction review package in a thread (if missing)
    POST /predict/<ps>/results            compute results/results.json in a thread (if missing or older than edits)
    POST /predict/<ps>/start              start the laptop run again (failed, or no heartbeat for STALE_S)
    GET  /predict/api/mine                {"schema": "cj_predict_mine/1", "items": [...]}: Home's "My results"
    GET  /predict/api/set/<ps>            one set's status; ?html=1 adds the page fragments (the refresh swaps them)
    GET  /predict/api/convert/<id>        one conversion's status; ?html=1 adds the page fragments
  Upload API: one file, or one piece of at most CHUNK_BYTES of a bigger file, per request (a Cloudflare free tunnel
  caps a request at about 100 MB; a raw .mrc is often more):
    POST /predict/api/upload              {"kind": "tiles"|"raw", "fields": {...}, "files": [{"name", "size"}],
                                           "resume": id|null} -> the session {"upload", "chunk", "received", "done"}.
                                           Fields, names, caps and free disk (the declared total is held) are checked
                                           here, before a byte is sent. "resume": the same person's session with the
                                           same files carries on where it stopped.
    GET  /predict/api/upload/<id>         the session (what has arrived)
    POST /predict/api/upload/<id>/file?name=<file>&offset=<n>   body = the bytes. offset must be what has arrived
                                           (else 409 + the session, and the page resumes from there). A file is
                                           content-checked once complete (tiles: 8-bit PNG/JPEG; raw: MRC, SerialEM
                                           sidecar, TIFF, sample key)
    POST /predict/api/upload/<id>/create  every file there -> the prediction set (tiles) or the conversion (raw)
    POST /predict/api/upload/<id>/cancel
  A session lives in <root>/.portal/us_<12 hex>/ (upload.json + files/); only its owner reaches it. Names: tiles
  .png/.jpg/.jpeg, raw the raw set; never a name ending in .part. A size or content violation, or a full disk, removes
  it ("nothing was kept"); a bad request (unknown name, bad offset: 400; wrong offset: 409), a dropped connection or
  a missing piece does not. A session never finished (no "create") is removed SCRATCH_MAX_AGE_S after its last
  request, and its disk hold released.
  Sidecars: every [Image = <name>] of an uploaded .idoc / .mrc.mdoc (read as cryoem_core.metadata.parse_metadata
  reads it) must be a bare file name (no \\ / :, not . or ..) and one of the uploaded files; a picked session's
  sidecars must name bare files too. Else refused: cryoem_core would open whatever path it names (another folder, a
  UNC share). jobs/convert_raw.py checks the same again before any tile is opened.

Sessions people may pick (portal/predict_sources.py): the session folders found under the scan roots
(CJ_PREDICT_SCAN_ROOTS, default the processing root) that have `3. Processed images/3d. Tiles no scalebar*`, plus the
admin's hand list (CJ_PREDICT_SOURCES), with the Admin panel's overrides (Admin > Predict sessions: hide, rename). A
session is named by its full registry id, e.g. 20261027 or 20260827-2. Plus the finished raw conversions, each seen
only by its owner and the admin (ctx.can_see).

Raw conversion (BUILD_CONTRACTS.md section 10): `python -m jobs.convert_raw` as a detached one-shot subprocess with
the processing repo from CJ_PROCESSING_REPO, one at a time (MAX_CONVERT_RUNNING; the others wait queued; a watcher
thread starts the next and records the end), at most MAX_CONV_PER_USER queued or running per person, and only with
DISK_FACTOR x the raw input free on the disk. Output in <root>/raw_tiles/<session id>_<6 hex>/ (the name starts with
the session id: Predict reads the id from it):
    convert.json   {"schema": "cj_convert/1", id, owner, session, recipe, gpu, source, inputs, sample_key, created}
    status.json    {state: queued|running|done|failed, updated, pid, started, finished, error, kind, warnings,
                    result (convert_raw's JSON), n_tiles}
    convert.out    the converter's stdout (one JSON object); convert.log its stderr (the pipeline's progress)
    input/         the uploaded raw files: kept CJ_RAW_KEEP_DAYS (default 7; 0 = removed at once) after the
                   conversion ends, done or failed, then swept; the admin may delete them sooner
    tiles/         the converter's --out: PNG tiles + MANIFEST.csv, the Predict source once done
  The recipe is "default" (convert_raw's DEFAULT_RECIPE; the page warns) or one of the processing repo's session
  recipes that convert_raw's static check accepts (the portal never runs one). The GPU is used only when ticked and the
  GPU check passes (else refused, saying why); the CPU otherwise. A session folder the admin picks is never written:
  the converter reads it and writes only under raw_tiles/.
  Stuck: a converter running over CONVERT_MAX_S, or whose log is silent for CONVERT_STALE_S, is killed and marked
  failed ("timeout"). Its process is known by pid AND start time (pid_created), so a reused pid is never taken for
  it (nor killed).

Configuration (environment, read per request; Config attributes of the same lower-case name win when present):
  CJ_PREDICT_SCAN_ROOTS, CJ_PREDICT_SOURCES   the session list (portal/predict_sources.py)
  CJ_PREDICT_ROOT      prediction sets (default <CryoAI-laptop>/predict, jobs.predict_set.default_root())
  CJ_REVIEW_ROOT       prediction review packages, when cfg.review_root (the shell's /review/<pkg_id>/ root) is unset
                       (default <CryoAI-laptop>/review)
  CJ_PROCESSING_REPO   the lab's processing repo (read only) for raw conversion; unset = no raw conversion
  CJ_RAW_ROOT          where the whole raw pipeline lives (raw-upload .portal staging + raw_tiles/), e.g. a NAS or an
                       external drive (cfg.raw_root wins); default = the predict root. Old <predict root>/raw_tiles
                       conversions stay listed and swept. Prediction sets made from it copy tiles (never link).
  CJ_RAW_KEEP_DAYS     days uploaded raw files are kept after their conversion ends (default 7; 0 = not kept)
  The registry is the portal's (cfg.registry).

Rules:
  - Every set view, API and action checks ctx.can_see(user, request.json owner); not found and not allowed give the
    same 404. The same for conversions (convert.json owner).
  - nm/px is never defaulted: each sample's value is shown (from the manifest when it has one) and the person must
    tick "I confirm the pixel size"; their id goes into images.csv nm_confirmed_by. No tick: refused, nothing written.
  - Pixel-size tolerance (BUILD_CONTRACTS.md section 8): a sample whose manifest nm/px values spread (max - min) by
    at most NM_TOLERANCE (1 %) of their median is one scale. The form shows the range; left empty, each tile keeps
    its own value in images.csv; a number typed there applies to all of that sample's tiles and is recorded in the
    portal audit log (predict.nm_override). Over the tolerance: refused, as before.
  - Model: the registry's production model (also in models/models.yaml) by default; basic users get only that one.
  - Raw upload: admin and advanced. Raw session pick: admin. Basic users get 403 on both.
  - Long work (create(), uploads to disk, the review package, results) runs in worker threads, never on the event
    loop. The predictor and the converter are detached processes; status.json says the rest.
  - Plain words for scientists: paths, ids of folders and raw errors only on admin-only lines.
  - Everything shown is escaped here.
  - Uploads: caps checked on the declared list and again while streaming; free disk checked at the start (the whole
    declared size held), at each piece and every 64 MB, counting every upload in flight (FREE_MARGIN_BYTES kept free).
  - A basic user has at most MAX_ACTIVE_BASIC sets queued or running. A review package built here is appended to
    the registry (review_package) once.
"""
from __future__ import annotations

import asyncio
import csv
import errno
import hashlib
import html
import json
import math
import os
import re
import secrets
import shutil
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect, Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route

from jobs import convert_raw as CR
from jobs import predict_local as PL
from jobs import predict_set as PS
from portal import predict_sources as PSRC

PREFIX = "/predict"
ROLES = ("admin", "advanced", "basic")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TEMPLATE = HERE.parent / "templates" / "predict.html"
MODELS_YAML = REPO / "models" / "models.yaml"
PORTAL_TMP = ".portal"                    # <root>/.portal/: the portal's own scratch (consumers skip dot entries)
IMAGE_EXT = (".png", ".jpg", ".jpeg")
SAMPLE_COLUMNS = ("sample_label", "sample", "grid")
MAX_FILES = PS.MAX_IMAGES                 # 200 tiles per analysis or upload (= a prediction set's maximum)
MAX_FILE_BYTES = 100 * 1024 * 1024        # 100 MB per file
MAX_FIELD_BYTES = 4096                    # a text field of the upload form
MAX_FIELDS = 20
FREE_MARGIN_BYTES = 2 * 1024 ** 3         # an upload must leave this much disk free (all uploads in flight counted)
DISK_CHECK_BYTES = 64 * 1024 ** 2         # free space re-checked at each file start and every 64 MB written
MAX_ACTIVE_BASIC = 3                      # queued or running sets a basic user may have at once
SCRATCH_MAX_AGE_S = 3600                  # <root>/.portal/* older than this is swept at start
CACHE_MAX = 256                           # entries kept in the spot-check cache and the finished-task table
NM_MIN, NM_MAX = 0.01, 100.0              # a plausible nm/px (cryo-EM tiles); outside = a typo
NM_TOLERANCE = 0.01                       # one sample's nm/px spread (max - min) <= this x median = one scale;
                                          # = results.compute / jobs.predict_set / portal results (a test pins all)
NM_EPS = 1e-12                            # float slack on that comparison (exactly 1 % is accepted); pinned too
SIZE_COLUMN = "output_size"               # the session manifest's tile size after processing ("1440x1023")
STALE_S = 300                             # predict_local refreshes `updated` every few s while running or waiting
REFRESH_S = 5
SPOT_N = 10
SCHEMA_MINE = "cj_predict_mine/1"
SCHEMA_SET = "cj_predict_set/1"
NAME_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
WIN_RESERVED = re.compile(r"^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$", re.I)
# uploads: one file (or one piece of a big file) per request
CHUNK_BYTES = 32 * 1024 ** 2              # a piece; a Cloudflare free tunnel caps a request at about 100 MB
NOJS_MAX_BYTES = 90 * 1024 ** 2           # the no-JavaScript multi-file form: the whole request
START_MAX_BYTES = 512 * 1024              # the JSON that opens an upload session (the file list)
UPLOAD_RE = re.compile(r"^us_[0-9a-f]{12}$")
UPLOAD_META, UPLOAD_FILES, PART = "upload.json", "files", ".part"
SCHEMA_UPLOAD = "cj_upload/1"
WRITE_BATCH = 4 * 1024 ** 2               # bytes gathered before one write in a worker thread
# raw conversion (jobs/convert_raw.py)
RAW_ROLES = ("admin", "advanced")
RAW_TILES = "raw_tiles"                   # <root>/raw_tiles/<session id>_<6 hex>/
RAW_ID_RE = re.compile(r"^(\d{8}(?:-\d+)?)_([0-9a-f]{6})$")
CONVERT_JSON, CONVERT_OUT, CONVERT_LOG = "convert.json", "convert.out", "convert.log"
CONVERT_INPUT, CONVERT_TILES, CONVERT_LOCK = "input", "tiles", ".convert.lock"
SCHEMA_CONVERT, SCHEMA_CONVERT_VIEW = "cj_convert/1", "cj_predict_convert/1"
MAX_CONVERT_RUNNING = 1                   # one montage render holds GBs of RAM: the others wait, queued
MAX_CONV_PER_USER = 2                     # one person's conversions queued or running at once
CONVERT_MAX_S = 2 * 3600                  # a running conversion is stopped (failed, "timeout") after this ...
CONVERT_STALE_S = 20 * 60                 # ... or when its log has been silent this long
PID_START_SLACK_S = 120                   # a pid whose process started further from `started` is someone else
KEEP_SETTING, KEEP_DEFAULT_DAYS = "CJ_RAW_KEEP_DAYS", 7   # uploaded raw input/ kept after the end (0 = not)
DISK_FACTOR = 2                           # free disk a conversion needs: this x its raw input (+ the margin)
CONVERT_POLL_S = 2.0                      # the watcher thread's beat while a conversion is queued or running
RECIPE_TTL_S = 60.0
RAW_EXT = (".mrc.mdoc", ".mrc", ".idoc", ".tif", ".tiff", ".csv")
RAW_MAX_FILES = 1000                      # one upload: an .idoc montage brings one .tif per tile
RAW_MAX_FILE_BYTES = 8 * 1024 ** 3        # an MRC stack
RAW_SIDE_MAX_BYTES = 32 * 1024 ** 2       # an .mdoc / .idoc / sample key
RAW_MAX_TOTAL_BYTES = 64 * 1024 ** 3
RAW_SCAN_DEPTH = 2                        # a session folder is searched this deep for montages
RAW_SCAN_MAX = 20000                      # entries looked at, at most
RAW_SKIP_DIR = re.compile(r"^(\.|[345]\.\s|processed|montages|archive|figures|__)", re.I)
NOTES_SUB, SAMPLE_KEY = "1. Notes + reports", "sample_key.csv"
GPU_CHECK_S = 60
TEST_HOOKS = ("CRYOEV_JOBS_TESTING", "CRYOEV_JOBS_TEST_HOOKS", "CRYOEV_JOBS_TEST_HOOKS_FILE")   # = jobs/run_job.py

NO_MODEL = "No production model is set up yet, so analyses can't start. Ask the admin."
NM_FROM_MANIFEST = "from the session’s MANIFEST.csv (tile pixel size after processing)"
NM_HINT = ("Where to find it: the microscope’s .mdoc file gives PixelSpacing in Å per pixel; multiply by the "
           "downsampling factor used in processing and divide by 10 for nm per pixel.")

esc = html.escape


class Refusal(Exception):
    """A request we refuse with a message for the person (plain words) and an HTTP status."""

    def __init__(self, msg, status=400, tech=None, drop=False):
        super().__init__(msg)
        self.msg, self.status, self.tech = msg, status, tech
        self.drop = drop                   # an upload piece: this refusal ends the upload session (size, content)


# ---------------------------------------------------------------- configuration
def _setting(cfg, name):
    v = getattr(cfg, name.lower(), None)
    return v if v not in (None, "") else os.environ.get(name.upper())


def predict_root(cfg) -> Path:
    v = _setting(cfg, "CJ_PREDICT_ROOT")
    return Path(v) if v else PS.default_root()


def review_root(cfg) -> Path:
    """cfg.review_root (the folder the shell's /review/<pkg_id>/ serves), else CJ_REVIEW_ROOT, else the default."""
    v = getattr(cfg, "review_root", None) or _setting(cfg, "CJ_REVIEW_ROOT")
    if v:
        return Path(v)
    from data_utils import config
    return config.laptop_root() / "review"


def registry_root(cfg):
    return getattr(cfg, "registry", None)


def sources(cfg) -> list:
    """The sessions people may pick: [{key, label, dir, manifest, session, sample_column, problem}]. The admin's
    CJ_PREDICT_SOURCES list (as before) merged with the sessions found by the scan, with the Admin page's overrides
    (portal/predict_sources.py). A hidden or incompatible source has a problem (shown to the admin only). A bad
    setting raises (sources_safe: an empty list, and the admin sees why on the page)."""
    return PSRC.for_predict(cfg)


def sources_safe(cfg, user=None, can_see=None):
    """(sources, problem). With a user: also the finished raw conversions that person may see (owner or admin)."""
    extra = converted_sources(cfg, user, can_see) if user is not None and can_see is not None else []
    try:
        return sources(cfg) + extra, None
    except (OSError, ValueError) as e:
        return extra, f"Session list unreadable: {type(e).__name__}: {e}"


# ---------------------------------------------------------------- models
def _models_yaml() -> dict:
    import yaml
    return (yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8")) or {}).get("models") or {}


def quality_line(spec: dict) -> str | None:
    """One plain line on how good a model is, from models.yaml (the EV score on held-out test images)."""
    s = str(spec.get("test_seed_spread") or "")
    m = re.search(r"EV\s+([0-9.]+)\s*\+/-\s*([0-9.]+)", s)
    if m:
        return f"EV score {float(m.group(1)):.2f} ± {float(m.group(2)):.2f} on held-out test images (1 = perfect)"
    m = re.search(r"EV\s+([0-9.]+)", str(spec.get("test_mask_map50_95") or ""))
    if m:
        return f"EV score {float(m.group(1)):.2f} on held-out test images (1 = perfect)"
    return None


def models(cfg) -> dict:
    """{"production": id | None, "choices": [{id, quality, production}]}: models.yaml entries whose weights exist;
    production = the newest registry model record with status production that models.yaml also has."""
    try:
        spec = _models_yaml()
    except (OSError, ValueError, ImportError):
        spec = {}
    usable = [m for m, s in spec.items() if isinstance(s, dict) and (REPO / str(s.get("weights", ""))).is_file()]
    prod = None
    try:
        recs = PS.registry_module().current("model", registry_root(cfg))
        cands = sorted((r for r in recs.values() if r.get("status") == "production" and r.get("id") in usable),
                       key=lambda r: str(r.get("logged") or ""))
        prod = cands[-1]["id"] if cands else None
    except Exception:                                    # registry unreadable: no default, never a guess
        prod = None
    return {"production": prod, "choices": [{"id": m, "quality": quality_line(spec[m]), "production": m == prod}
                                            for m in usable]}


def allowed_models(cfg, user) -> tuple:
    """-> (default id or None, [choices this person may pick])."""
    ms = models(cfg)
    if user.get("role") == "basic":
        return ms["production"], [c for c in ms["choices"] if c["production"]]
    return ms["production"], ms["choices"]


# ---------------------------------------------------------------- a session's tiles
def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def nm_spread(values) -> float:
    """(max - min) / median of nm/px values, one per tile (0.0 for a single value). = results.compute.nm_spread."""
    vals = sorted(float(v) for v in values)
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return (vals[-1] - vals[0]) / med


def one_scale(values) -> bool:
    """These nm/px values (one per tile of one sample) count as one scale: spread <= NM_TOLERANCE (+ NM_EPS, so
    exactly 1 % is not lost to float rounding). = results.compute.one_scale and jobs.predict_set.nm_one_scale."""
    return nm_spread(values) <= NM_TOLERANCE + NM_EPS


def nm_range_text(lo, hi) -> str:
    return f"{lo:g}" if lo == hi else f"{lo:g}–{hi:g}"


SIZE_RE = re.compile(r"^\s*(\d+)\s*[x×X]\s*(\d+)\s*$")


def tile_size(path: Path, row: dict):
    """(w, h) from the manifest's output_size ("1440x1023"), else from the image header (no decode); None when
    neither can be read (create() then refuses that tile with the reason)."""
    m = SIZE_RE.match(str(row.get(SIZE_COLUMN) or ""))
    if m:
        return int(m.group(1)), int(m.group(2))
    try:
        from PIL import Image
        with Image.open(path) as im:                 # lazy: reads the header only
            return tuple(im.size)
    except Exception:
        return None


def size_suffix(size) -> str:
    return f" · {size[0]}×{size[1]} tiles" if size else " · tiles of unknown size"


def tiles_of(src: dict) -> dict:
    """{"groups": [{label, base, suffix, size, names, nm, nm_values, nm_of, nm_min, nm_max, nm_missing, nm_range,
    mixed}], "sig", "n", "column"}.
    Groups come from the manifest's sample column (without a manifest: one sample, the source label), split by tile
    size (manifest output_size, else the image header): a sample with more than one tile size becomes one group per
    size, labelled "<sample> · <W>×<H> tiles" (suffix; that label goes into images.csv, so results counts them as
    separate samples: one sample per scale). nm: the one value when every tile has the same; None when missing (any
    tile), several or mixed. nm_of: tile name -> its manifest value (None = missing); nm_missing: tiles without one.
    nm_range: several values, all tiles have one, spread within NM_TOLERANCE (one scale; each tile keeps its own).
    mixed: the values spread more than that (refused)."""
    names = sorted((p.name for p in src["dir"].iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT
                    and not p.name.startswith(".")), key=str)
    rows, col = {}, None
    if src["manifest"] is not None:
        with open(src["manifest"], newline="", encoding="utf-8-sig") as f:
            data = list(csv.DictReader(f))
        head = data[0].keys() if data else []
        key = next((c for c in PS.NAME_COLUMNS if c in head), None)
        if key:
            rows = {r[key]: r for r in data}
        want = src.get("sample_column")
        col = want if want in head else next((c for c in SAMPLE_COLUMNS if c in head), None)
    groups = {}
    for n in names:
        r = rows.get(n) or {}
        if col:
            v = str(r.get(col) or "").strip()
            label = (f"grid {v}" if col == "grid" and v and not v.lower().startswith("grid") else v) or "no sample"
        else:
            label = src["label"]
        size = tile_size(src["dir"] / n, r)
        g = groups.setdefault((label, size), {"base": label, "size": size, "names": [], "nm_of": {}})
        g["names"].append(n)
        g["nm_of"][n] = _num(r.get(PS.NM_COLUMN))
    n_sizes = {}
    for base, _ in groups:
        n_sizes[base] = n_sizes.get(base, 0) + 1
    for g in groups.values():
        g["suffix"] = size_suffix(g["size"]) if n_sizes[g["base"]] > 1 else ""
        g["label"] = g["base"] + g["suffix"]
    out = []
    for g in sorted(groups.values(), key=lambda g: (str(g["base"]), g["size"] or (0, 0))):
        per_tile = [v for v in g["nm_of"].values() if v is not None]
        missing = len(g["names"]) - len(per_tile)
        vals = sorted(set(per_tile))
        mixed = len(vals) > 1 and not one_scale(per_tile)
        out.append({"label": g["label"], "base": g["base"], "suffix": g["suffix"], "size": g["size"],
                    "names": g["names"], "nm_of": g["nm_of"],
                    "nm": vals[0] if len(vals) == 1 and not missing else None, "nm_values": vals,
                    "nm_min": vals[0] if vals else None, "nm_max": vals[-1] if vals else None,
                    "nm_missing": missing, "nm_range": len(vals) > 1 and not mixed and not missing, "mixed": mixed})
    # the values are in the signature too: a range group's tile values are not sent back by the form, so a manifest
    # changed since the person confirmed the range must be seen again
    sig = hashlib.sha1(json.dumps([[g["label"], len(g["names"]), g["nm_values"]] for g in out]).encode()
                       ).hexdigest()[:12]
    return {"groups": out, "sig": sig, "n": len(names), "column": col}


def pick(groups: list, count) -> list:
    """The tiles to use: every tile of `groups`, or `count` of them spread over the groups in proportion to their
    size (>= 1 each when count >= the number of groups), evenly spaced within each group. Deterministic."""
    total = sum(len(g["names"]) for g in groups)
    if count is None or count >= total:
        return [n for g in groups for n in g["names"]]
    if count < len(groups):
        groups = sorted(groups, key=lambda g: (-len(g["names"]), g["label"]))[:count]
        quota = {g["label"]: 1 for g in groups}
    else:
        exact = {g["label"]: count * len(g["names"]) / total for g in groups}
        quota = {g["label"]: min(len(g["names"]), max(1, math.floor(exact[g["label"]]))) for g in groups}
        size = {g["label"]: len(g["names"]) for g in groups}
        while sum(quota.values()) > count:
            k = max((k for k in quota if quota[k] > 1), key=lambda k: (quota[k] - exact[k], k))
            quota[k] -= 1
        while sum(quota.values()) < count:
            k = max((k for k in quota if quota[k] < size[k]), key=lambda k: (exact[k] - quota[k], k))
            quota[k] += 1
    out = []
    for g in groups:
        n, q = len(g["names"]), quota[g["label"]]
        out += [g["names"][min(n - 1, int((i + 0.5) * n / q))] for i in range(q)]
    return out


# ---------------------------------------------------------------- reading sets
def _read_json(p: Path):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def set_dir(cfg, ps) -> Path | None:
    if not isinstance(ps, str) or not PS.PS_RE.match(ps):
        return None
    d = predict_root(cfg) / ps
    return d if (d / PS.REQUEST).is_file() else None


def load_set(cfg, user, can_see, ps) -> tuple:
    """-> (folder, request) when it exists AND this person may see it; else (None, None) (one answer for both)."""
    d = set_dir(cfg, ps)
    req = _read_json(d / PS.REQUEST) if d else None
    if not isinstance(req, dict) or not can_see(user, req.get("owner")):
        return None, None
    return d, req


def list_sets(cfg, user, can_see) -> list:
    root = predict_root(cfg)
    try:
        dirs = [p for p in root.iterdir() if p.is_dir() and PS.PS_RE.match(p.name)]
    except OSError:
        return []
    out = []
    for d in dirs:
        req = _read_json(d / PS.REQUEST)
        if not isinstance(req, dict) or not can_see(user, req.get("owner")):
            continue
        st = _read_json(d / PS.STATUS) or {}
        out.append((d, req, st))
    out.sort(key=lambda t: (str(t[1].get("created") or ""), t[0].name), reverse=True)
    return out


def images_rows(d: Path) -> list:
    try:
        return PS.read_images(d)
    except (OSError, ValueError, csv.Error):
        return []


def samples_of(rows) -> list:
    by = {}
    for r in rows:
        s = by.setdefault(r.get("sample_label") or "", {"label": r.get("sample_label") or "", "n": 0, "nm": set(),
                                                         "by": set()})
        s["n"] += 1
        s["nm"].add(r.get("nm_per_px") or "")
        s["by"].add(r.get("nm_confirmed_by") or "")
    def nm_text(vals):
        nums = sorted(_num(v) for v in vals if _num(v))
        if len(nums) > 1 and len(nums) == len(vals):     # several values (within NM_TOLERANCE): each image its own
            return f"{nm_range_text(nums[0], nums[-1])} (each image its own)"
        return ", ".join(f"{float(v):g}" if _num(v) else "?" for v in sorted(vals))
    return [{"label": s["label"], "n": s["n"], "nm_per_px": nm_text(s["nm"]),
             "confirmed_by": ", ".join(sorted(b for b in s["by"] if b)) or None} for s in by.values()]


def package_dir(cfg, ps) -> Path | None:
    try:
        p = review_root(cfg) / f"pred_{ps}"
    except Exception:
        return None
    return p if (p / "review_package.json").is_file() else None


def results_file(d: Path) -> Path | None:
    p = d / "results" / "results.json"
    return p if p.is_file() else None


def edits_mtime(pkg: Path | None) -> float:
    if pkg is None or not (pkg / "edits").is_dir():
        return 0.0
    return max((p.stat().st_mtime for p in (pkg / "edits").glob("*.json")), default=0.0)


def _parse_local(s) -> float | None:
    try:
        dt = datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None
    return dt.timestamp()


def when(s) -> str:
    try:
        dt = datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return "?"
    if dt.tzinfo is not None:
        dt = dt.astimezone()
    return f"{dt.day} {dt.strftime('%b %Y, %H:%M')}"


def ago(seconds) -> str:
    s = max(0, int(seconds))
    if s < 90:
        return f"{s} s"
    if s < 5400:
        return f"{round(s / 60)} min"
    return f"{s / 3600:.1f} h"


# ---------------------------------------------------------------- background work (review package, results)
_TASKS: dict = {}
_TASK_LOCK = threading.Lock()


def task(kind, ps) -> dict | None:
    with _TASK_LOCK:
        t = _TASKS.get((kind, ps))
        return dict(t) if t else None


def start_task(kind, ps, fn) -> bool:
    """Run fn() in a daemon thread unless the same (kind, ps) is running. A failure is kept for the page."""
    with _TASK_LOCK:
        cur = _TASKS.get((kind, ps))
        if cur and cur["state"] == "running":
            return False
        _TASKS.pop((kind, ps), None)
        _TASKS[(kind, ps)] = {"state": "running", "error": None, "started": time.time()}
        for k in [k for k, v in _TASKS.items() if v["state"] != "running"][:max(0, len(_TASKS) - CACHE_MAX)]:
            _TASKS.pop(k, None)                          # oldest finished entries first: the table stays bounded

    def run():
        try:
            fn()
            with _TASK_LOCK:
                _TASKS.pop((kind, ps), None)
        except BaseException as e:          # noqa: BLE001 - kept and shown, never lost in a thread
            with _TASK_LOCK:
                _TASKS[(kind, ps)] = {"state": "failed", "error": f"{type(e).__name__}: {e}", "started": time.time()}

    threading.Thread(target=run, name=f"cj-predict-{kind}-{ps}", daemon=True).start()
    return True


def build_review(cfg, d: Path):
    from annotation import prediction_package as PP
    root = review_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    pkg = PP.build(d, root)
    PP.spot_check(pkg, n=SPOT_N, seed=0, write=True, mode="representative")
    register_package(cfg, pkg)
    return pkg


REVIEW_PACKAGE_LOCK = ".review_package.lock"     # in the registry folder: read + append, as predict_set's lock


def register_package(cfg, pkg: Path):
    """Append the package's registry line (its own review_package.json, prediction_package.registry_record) unless
    the registry already has that id, under a registry-folder lock (check + append as one step)."""
    reg = registry_root(cfg)
    if reg is None:
        return None
    rec = _read_json(pkg / "review_package.json")
    if not isinstance(rec, dict) or not rec.get("id"):
        raise ValueError(f"{pkg.name}: review_package.json missing or unreadable")
    regm = PS.registry_module()
    with PS.locked(Path(reg) / REVIEW_PACKAGE_LOCK):
        if rec["id"] in regm.current("review_package", reg):
            return None
        return regm.append(rec, reg)


def build_results(cfg, d: Path, ps: str):
    from results import compute_and_write
    pkg = package_dir(cfg, ps)
    reviewed = None
    if pkg is not None and edits_mtime(pkg):
        from annotation import prediction_package as PP
        reviewed = PP.reviewed_polygons(pkg, write=True)
    compute_and_write(d, reviewed)


_SPOT_CACHE: dict = {}
_SPOT_LOCK = threading.Lock()


def spot_list(cfg, d: Path, ps: str, rows) -> tuple:
    """-> (rows to check first [{image, file, sample_label, orig_name}], mode text). From the package's
    spotcheck.json when there is one, else spot_check() on the set (cached per set)."""
    by_image = {r.get("image"): r for r in rows}
    pkg = package_dir(cfg, ps)
    ids = None
    if pkg is not None:
        sc = _read_json(pkg / "spotcheck.json")
        if isinstance(sc, dict) and isinstance(sc.get("images"), list):
            ids = [str(i.get("image")) for i in sc["images"] if isinstance(i, dict)]
    if ids is None:
        key = (str(d), (d / PS.STATUS).stat().st_mtime if (d / PS.STATUS).exists() else 0)
        with _SPOT_LOCK:
            ids = _SPOT_CACHE.get(key)
        if ids is None:
            from annotation import prediction_package as PP
            try:
                ids = PP.spot_check(d, n=SPOT_N, seed=0, write=False, mode="representative")
            except Exception:
                ids = []
            with _SPOT_LOCK:
                _SPOT_CACHE[key] = ids
                while len(_SPOT_CACHE) > CACHE_MAX:
                    _SPOT_CACHE.pop(next(iter(_SPOT_CACHE)))
    out = []
    for i in ids:
        r = by_image.get(i) or {}
        out.append({"image": i, "file": Path(r.get("file") or "").name or None,
                    "sample_label": r.get("sample_label"), "orig_name": r.get("orig_name")})
    return out


# ---------------------------------------------------------------- status in plain words
RESTART_REQUESTED = "restart requested"


def waiting_text(waiting) -> str:
    """predict_local's `waiting` reason in plain words (admins also get the raw reason on the tech line)."""
    w = str(waiting or "").lower()
    if w.startswith(RESTART_REQUESTED):
        return "Starting again…"
    if "gpu busy" in w:
        return "Waiting for the GPU: another job is using it."
    if any(k in w for k in ("lock", "another", "queue", "after")):
        return "Waiting for the GPU: another analysis is running."
    return "Waiting for the computer to be free."


def age_s(st: dict, now) -> float:
    return now - (_parse_local(st.get("updated")) or now)


def status_view(st: dict, admin: bool, now=None) -> dict:
    """status.json -> {state, tone, label, detail, tech (admin only), final, n_done, n_total, progress}. Queued or
    running with `updated` older than STALE_S (predict_local keeps a heartbeat while it runs or waits) = stale."""
    now = time.time() if now is None else now
    state = st.get("state")
    n_done, n_total = st.get("n_done"), st.get("n_total")
    out = {"state": state, "tone": "wait", "label": "Unknown", "detail": None, "tech": None, "final": False,
           "n_done": n_done, "n_total": n_total, "progress": False}
    stale = state in ("queued", "running") and not st.get("job") and age_s(st, now) > STALE_S
    if state == "queued":
        out["label"] = "Waiting to start"
        if st.get("waiting"):
            out["detail"] = waiting_text(st["waiting"])
        elif st.get("job"):
            out["detail"] = "Queued on the GPU computer."
        else:
            out["detail"] = "It starts in a moment."
    elif state == "running":
        out.update(tone="run", label="Running", progress=bool(n_total))
        out["detail"] = f"{int(n_done or 0)} of {int(n_total or 0)} images done"
    elif state == "done":
        out.update(tone="ok", label="Done", final=True, detail=f"{int(n_total or 0)} images analysed")
    elif state == "failed":
        out.update(tone="warn", label="Stopped with an error", final=True,
                   detail="Try “Start again”; if it fails again, ask the admin (they can see the details).")
    if stale:                                          # no heartbeat: the run (or the wait) may have stopped
        out.update(tone="warn", label="No progress")
        out["detail"] = (out["detail"] + ". " if out["detail"] else "") + \
            f"No news for {ago(age_s(st, now))}: it may have stopped. Try “Start again” or ask the admin."
    if admin:
        bits = [f"where {st.get('where')}"] + [f"{k} {st.get(k)}" for k in ("job", "pid", "waiting", "error")
                                                if st.get(k)] + [f"updated {st.get('updated')}"]
        out["tech"] = "; ".join(bits)
    return out


def can_restart(st: dict, now=None) -> bool:
    """Failed, or queued / running with no heartbeat for STALE_S. A set waiting with a fresh heartbeat is left to
    predict_local (it waits for the lock / GPU and chains the next set); a desktop-queue set is never started here."""
    now = time.time() if now is None else now
    state = st.get("state")
    if state == "failed":
        return True
    if st.get("job") and state in ("queued", "running"):
        return False
    return state in ("queued", "running") and age_s(st, now) > STALE_S


def restart(cfg, d: Path) -> bool:
    """"Start again": mark the set queued with waiting "restart requested" and a fresh `updated` (so a double click
    is not a second restart), then spawn predict_local (its own lock keeps one run per root)."""
    st = _read_json(d / PS.STATUS) or {}
    if not can_restart(st):
        return False
    PS.write_status(d, state="queued", error=None, pid=None, waiting=RESTART_REQUESTED)
    try:
        PL.start_detached(d, root=predict_root(cfg), registry_root=registry_root(cfg))
    except Exception as e:                       # shown to the admin; goes stale and offers Start again later
        PS.write_status(d, waiting=f"could not start the run: {type(e).__name__}: {e}")
    return True


# ---------------------------------------------------------------- html (every text through esc)
def chip(label, tone) -> str:
    return f'<span class="cj-predict-chip cj-predict-t-{esc(tone)}">{esc(label)}</span>'


def tech(text) -> str:
    return f'<p class="cj-mut cj-wrap cj-predict-tech">{esc(text)}</p>' if text else ""


def msg_box(text, tone="warn", extra="") -> str:
    return f'<div class="cj-predict-msg cj-predict-t-{esc(tone)}" role="alert"><p>{esc(text)}</p>{extra}</div>'


def problem_list(text) -> str:
    """A refusal's 'problem(s):' list as a bulleted list (plain lines)."""
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if len(lines) <= 1:
        return esc(lines[0] if lines else "")
    return esc(lines[0]) + "<ul>" + "".join(f"<li>{esc(ln)}</li>" for ln in lines[1:]) + "</ul>"


_PH = re.compile(r"\{\{(body)\}\}")


def wrap(body: str) -> str:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    return _PH.sub(lambda m: body, tpl)


def source_text(req: dict, srcs, admin=False) -> str:
    """A source with a problem (hidden, incompatible) lends its label only when admin=True (callers pass nothing
    yet, so for everyone it is "Session <id>" alone)."""
    src = req.get("source") or {}
    if src.get("kind") == "upload":
        return "Uploaded images"
    label = next((s["label"] for s in srcs if src.get("dir") and (admin or not s.get("problem")) and
                  os.path.normcase(str(s["dir"].resolve())) == os.path.normcase(str(src.get("dir")))), None)
    return f"Session {src.get('session') or '?'}" + (f" ({label})" if label else "")


def status_html(view: dict) -> str:
    bar = ""
    if view["progress"]:
        v, t = max(0, min(int(view["n_done"] or 0), int(view["n_total"]))), int(view["n_total"])
        bar = (f'<progress class="cj-predict-bar" max="{t}" value="{v}" '
               f'aria-label="{v} of {t} images done"></progress>')
    detail = f'<p class="cj-predict-detail">{esc(view["detail"])}</p>' if view["detail"] else ""
    return (f'<div class="cj-card cj-predict-status"><div class="cj-predict-top"><b>Status</b>'
            f'{chip(view["label"], view["tone"])}</div>{bar}{detail}{tech(view["tech"])}</div>')


def actions_html(cfg, d: Path, req: dict, st: dict, admin: bool, rows) -> str:
    ps = d.name
    parts = []
    if can_restart(st):
        parts.append(f'<form method="post" action="/predict/{esc(ps)}/start" class="cj-predict-inline">'
                     f'<button class="cj-predict-btn" type="submit">Start again</button></form>')
    if st.get("state") != "done":
        return "".join(parts)
    pkg = package_dir(cfg, ps)
    res = results_file(d)
    t_rev, t_res = task("review", ps), task("results", ps)
    # Review
    if pkg is not None:
        rev = (f'<a class="cj-predict-btn cj-predict-primary" href="/review/{esc(pkg.name)}/">Review</a>'
               f'<p class="cj-mut">Check the model’s outlines and fix mistakes. Start with the suggested '
               f'images below.</p>')
        if admin and t_rev and t_rev["state"] == "failed":     # e.g. built, but the registry line failed
            rev += tech(f"review task: {t_rev['error']}")
    elif t_rev and t_rev["state"] == "running":
        rev = '<p class="cj-predict-busy" role="status">Preparing the review… this page updates by itself.</p>'
    else:
        err = ""
        if t_rev and t_rev["state"] == "failed":
            err = msg_box("Preparing the review failed.", extra=tech(t_rev["error"]) if admin else "")
        rev = (f'{err}<form method="post" action="/predict/{esc(ps)}/review" class="cj-predict-inline">'
               f'<button class="cj-predict-btn cj-predict-primary" type="submit">Review</button></form>'
               f'<p class="cj-mut">Prepares the images for checking (about a minute), then opens the review.</p>')
    # Results
    stale = res is not None and edits_mtime(pkg) > res.stat().st_mtime
    if t_res and t_res["state"] == "running":
        out = '<p class="cj-predict-busy" role="status">Counting… this page updates by itself.</p>'
    else:
        err = ""
        if t_res and t_res["state"] == "failed":
            err = msg_box("Counting failed.", extra=tech(t_res["error"]) if admin else "")
        if res is not None:
            out = (f'{err}<a class="cj-predict-btn" href="/results/{esc(ps)}/">Results</a>'
                   f'<p class="cj-mut">EV counts, sizes and multilayer % per sample.</p>')
            if stale:
                out += (f'<form method="post" action="/predict/{esc(ps)}/results" class="cj-predict-inline">'
                        f'<p class="cj-mut">The review has changed since these results were counted.</p>'
                        f'<button class="cj-predict-btn" type="submit">Update results</button></form>')
        else:
            out = (f'{err}<form method="post" action="/predict/{esc(ps)}/results" class="cj-predict-inline">'
                   f'<button class="cj-predict-btn" type="submit">Results</button></form>'
                   f'<p class="cj-mut">Counts EVs per sample (model only until images are reviewed).</p>')
    parts.append(f'<div class="cj-predict-acts"><div class="cj-card">{rev}</div><div class="cj-card">{out}</div>'
                 f'</div>')
    # spot check
    spot = spot_list(cfg, d, ps, rows)
    if spot:
        items = []
        for s in spot:
            name = f'Image {esc(s["image"])}' + (f' <span class="cj-mut cj-wrap">{esc(s["orig_name"])}</span>'
                                                  if s.get("orig_name") else "")
            link = (f'<a href="/review/{esc(pkg.name)}/?img={esc(s["file"])}">{name}</a>'
                    if pkg is not None and s.get("file") else name)
            items.append(f'<li><span class="cj-predict-li-main">{link}</span>'
                         f'<span class="cj-predict-tag">{esc(s.get("sample_label") or "")}</span></li>')
        parts.append(f'<h2 class="cj-sec">Check these first</h2><div class="cj-card"><p class="cj-mut">'
                     f'A few images spread across your samples. Checking them (strongly encouraged) shows how much '
                     f'to trust the counts, and your fixes count in the results.</p><ol class="cj-predict-list">'
                     f'{"".join(items)}</ol></div>')
    return "".join(parts)


def set_view(cfg, user, d: Path, req: dict, with_html: bool) -> dict:
    admin = user.get("role") == "admin"
    st = _read_json(d / PS.STATUS) or {}
    view = status_view(st, admin)
    ps = d.name
    busy = any((task(k, ps) or {}).get("state") == "running" for k in ("review", "results"))
    pkg, res = package_dir(cfg, ps), results_file(d)
    out = {"schema": SCHEMA_SET, "ps": ps, "state": view["state"], "label": view["label"], "detail": view["detail"],
           "n_done": view["n_done"], "n_total": view["n_total"], "live": (not view["final"]) or busy,
           "review_href": f"/review/{pkg.name}/" if pkg else None,
           "results_href": f"/results/{ps}/" if res else None}
    if with_html:
        rows = images_rows(d)
        out["html"] = {"status": status_html(view), "actions": actions_html(cfg, d, req, st, admin, rows)}
    return out


def item_of(cfg, user, d: Path, req: dict, st: dict, srcs) -> dict:
    ps = d.name
    view = status_view(st, user.get("role") == "admin")
    pkg = package_dir(cfg, ps)
    return {"ps": ps, "href": f"/predict/{ps}/", "title": f"Analysis of {when(req.get('created'))}",
            "created": req.get("created"), "owner": req.get("owner"), "model": req.get("model"),
            "source": source_text(req, srcs, admin=user.get("role") == "admin"), "state": view["state"], "label": view["label"],
            "n_done": view["n_done"], "n_total": view["n_total"],
            "review_href": f"/review/{pkg.name}/" if pkg else None,
            "results_href": f"/results/{ps}/" if results_file(d) else None}


def mine_html(items, admin) -> str:
    if not items:
        return '<p class="cj-mut">No analyses yet. Start one with “New analysis”.</p>'
    li = []
    for it in items:
        tone = {"done": "ok", "running": "run", "failed": "warn"}.get(it["state"], "wait")
        prog = f'{int(it["n_done"] or 0)} of {int(it["n_total"] or 0)} images' if it["state"] == "running" \
            else f'{int(it["n_total"] or 0)} images'
        who = f' · {esc(it["owner"])}' if admin else ""
        links = "".join(f'<a href="{esc(h)}">{t}</a>' for h, t in ((it["review_href"], "Review"),
                                                                    (it["results_href"], "Results")) if h)
        li.append(f'<li class="cj-card cj-predict-item"><div class="cj-predict-top">'
                  f'<a class="cj-predict-ttl" href="{esc(it["href"])}">{esc(it["source"])}</a>'
                  f'{chip(it["label"], tone)}</div>'
                  f'<p class="cj-mut">{esc(when(it["created"]))} · {esc(prog)}{who}</p>'
                  + (f'<p class="cj-predict-links">{links}</p>' if links else "") + '</li>')
    return f'<ul class="cj-predict-items">{"".join(li)}</ul>'


def page_mine(cfg, user, can_see) -> str:
    admin = user.get("role") == "admin"
    srcs, _ = sources_safe(cfg, user, can_see)
    items = [item_of(cfg, user, d, req, st, srcs) for d, req, st in list_sets(cfg, user, can_see)]
    convs = list_conversions(cfg, user, can_see)
    conv = ""
    if convs or user.get("role") in RAW_ROLES:
        conv = (f'<h2 class="cj-sec">{"Raw conversions" if admin else "My raw conversions"}</h2>'
                + conversions_html(convs, admin))
    return wrap(f'<div class="cj-predict"><div class="cj-predict-head"><div><h1>Predict</h1>'
                f'<p class="cj-mut">Count and size the EVs in your images with the lab’s model.</p></div>'
                f'<a class="cj-predict-btn cj-predict-primary" href="/predict/new/">New analysis</a></div>'
                f'<h2 class="cj-sec">{"All analyses" if admin else "My analyses"}</h2>{mine_html(items, admin)}'
                f'{conv}</div>')


def model_block(cfg, user, chosen=None) -> str:
    default, choices = allowed_models(cfg, user)
    if not choices:
        return msg_box(NO_MODEL)
    pick_id = chosen if any(c["id"] == chosen for c in choices) else default
    if len(choices) == 1:
        c = choices[0]
        q = f'<p class="cj-mut">{esc(c["quality"])}</p>' if c["quality"] else ""
        return (f'<fieldset class="cj-card cj-predict-fs"><legend>Model</legend>'
                f'<input type="hidden" name="model" value="{esc(c["id"])}">'
                f'<p><b>{esc(c["id"])}</b>{" (the lab’s production model)" if c["production"] else ""}</p>'
                f'{q}</fieldset>')
    opts = f'<option value=""{"" if pick_id else " selected"} disabled>Choose…</option>' + "".join(
        f'<option value="{esc(c["id"])}"{" selected" if c["id"] == pick_id else ""}>'
        f'{esc(c["id"])}{" (production)" if c["production"] else ""}</option>' for c in choices)
    qual = "".join(f'<li><b>{esc(c["id"])}</b>: {esc(c["quality"])}</li>' for c in choices if c["quality"])
    return (f'<fieldset class="cj-card cj-predict-fs"><legend>Model</legend>'
            f'<label class="cj-predict-field"><span>Model (keep the production model unless you know why)</span>'
            f'<select name="model" required>{opts}</select></label>'
            + (f'<ul class="cj-mut cj-predict-qual">{qual}</ul>' if qual else "") + '</fieldset>')


def page_new(cfg, user, can_see) -> str:
    admin = user.get("role") == "admin"
    srcs, err = sources_safe(cfg, user, can_see)
    cards = []
    for s in srcs:
        if s["problem"]:
            if admin:
                cards.append(f'<div class="cj-card cj-predict-src cj-predict-off"><b>{esc(s["label"])}</b>'
                             f'<p class="cj-mut">Not available: {esc(s["problem"])}</p>{tech(str(s["dir"]))}</div>')
            continue
        who = (f' · converted by {esc(s["owner"])}' if admin and s.get("owner") else
               ' · converted from raw' if s.get("owner") else "")
        cards.append(f'<a class="cj-card cj-predict-src" href="/predict/new/session/{esc(s["key"])}/">'
                     f'<b>{esc(s["label"])}</b><span class="cj-mut">Session {esc(s["session"])}{who}</span>'
                     + (tech(str(s["dir"])) if admin else "") + '</a>')
    if err and admin:
        cards.append(msg_box("The session list setting can't be read.", extra=tech(err)))
    no_model = "" if allowed_models(cfg, user)[1] else msg_box(NO_MODEL)
    sess = ("".join(cards) or '<p class="cj-mut">No processed sessions are set up for analysis yet'
            + (' (set CJ_PREDICT_SCAN_ROOTS or CJ_PREDICT_SOURCES).' if admin else
               '; ask the admin, or upload your own images.') + '</p>')
    raw = ""
    if user.get("role") in RAW_ROLES:
        repo, why = processing_repo(cfg)
        if repo is None:
            raw = ('<h2 class="cj-sec">Or convert raw images</h2>'
                   + msg_box("Raw conversion is not set up on this server.",
                             extra=tech(why) if admin else '<p>Ask the admin.</p>'))
        else:
            raw = ('<h2 class="cj-sec">Or convert raw images</h2><div class="cj-predict-srcs">'
                   '<a class="cj-card cj-predict-src" href="/predict/new/raw/"><b>Upload raw images</b>'
                   '<span class="cj-mut">.mrc with its .mrc.mdoc, or .idoc with its .tif tiles: made into tiles '
                   'the way the lab’s processing does it, then analysed here</span></a>'
                   + ('<a class="cj-card cj-predict-src" href="/predict/new/raw/sessions/"><b>Pick a raw session</b>'
                      '<span class="cj-mut">Imaging sessions on the server with no processed tiles yet</span></a>'
                      if admin else "") + '</div>')
    return wrap(
        '<div class="cj-predict"><p class="cj-mut"><a href="/predict/">← My analyses</a></p>'
        f'<h1>New analysis</h1>{no_model}'
        '<div class="cj-card cj-predict-what"><h2>What images work</h2><ul>'
        '<li>Processed cryo-EM tiles: 8-bit PNG or JPEG, contrast-adjusted the way the lab’s processing does '
        'it' + (' (raw MRC or TIFF montages: convert them first, below).</li>' if raw else
                ' (not raw MRC or TIFF).</li>') +
        '<li>The same kind of images and magnification as the model was trained on. If unsure, ask the admin.</li>'
        f'<li>Up to {MAX_FILES} images per analysis. You must confirm the pixel size (nm per pixel) of each '
        'sample: sizes in nm depend on it.</li></ul></div>'
        '<h2 class="cj-sec">1. Pick a processed session</h2>'
        f'<div class="cj-predict-srcs">{sess}</div>'
        '<h2 class="cj-sec">Or upload your own tiles</h2>'
        '<a class="cj-card cj-predict-src" href="/predict/new/upload/"><b>Upload images</b>'
        f'<span class="cj-mut">Up to {MAX_FILES} PNG or JPEG files, {MAX_FILE_BYTES // 2 ** 20} MB each</span></a>'
        f'{raw}</div>')


def page_session_form(cfg, user, src: dict, t: dict, form=None, error=None) -> str:
    admin = user.get("role") == "admin"
    form = form or {}
    picked = set(form["samples"] if "samples" in form else
                 [f"g{i}" for i, g in enumerate(t["groups"]) if not g["mixed"]])
    rows = []
    for i, g in enumerate(t["groups"]):
        gid = f"g{i}"
        nm_val = form.get(f"nm_{gid}", f"{g['nm']:g}" if g["nm"] else "")
        if g["mixed"]:
            note = ('<span class="cj-predict-warn">Pixel sizes in this sample differ by more than '
                    f'{NM_TOLERANCE * 100:g} % (' + esc(", ".join(f"{v:g}" for v in g["nm_values"])) +
                    ' nm/px): it can’t be analysed as one sample. Ask the admin.</span>')
            rows.append(f'<tr><td data-l="Sample"><label class="cj-predict-off"><input type="checkbox" disabled> '
                        f'{esc(g["label"])}</label></td><td data-l="Tiles" class="cj-num">{len(g["names"])}</td>'
                        f'<td data-l="nm/px">{note}</td></tr>')
            continue
        if g["nm_range"]:
            # several values within the tolerance: each tile keeps its own; a number typed here replaces them all
            nm_val = form.get(f"nm_{gid}", "")
            nm_cell = (f'<span class="cj-predict-range">{esc(nm_range_text(g["nm_min"], g["nm_max"]))} nm/px across '
                       f'tiles, within {NM_TOLERANCE * 100:g} %; each tile keeps its own value</span> '
                       f'<span class="cj-mut">({NM_FROM_MANIFEST})</span><br>'
                       f'<input class="cj-predict-nm" name="nm_{gid}" inputmode="decimal" '
                       f'aria-label="one nm per pixel for all tiles of {esc(g["label"])} (optional)" '
                       f'placeholder="each tile’s own" value="{esc(nm_val)}">'
                       f' <span class="cj-mut">leave empty, or type one value to use for all '
                       f'{len(g["names"])} tiles</span>')
        else:
            src_note = NM_FROM_MANIFEST if g["nm"] else "not recorded: enter it"
            if g["nm_missing"] and g["nm_values"]:
                src_note = (f"not recorded for {g['nm_missing']} of {len(g['names'])} tiles (the others: "
                            f"{nm_range_text(g['nm_min'], g['nm_max'])}): enter one value for all of them")
            nm_cell = (f'<input class="cj-predict-nm" name="nm_{gid}" inputmode="decimal" '
                       f'aria-label="nm per pixel for {esc(g["label"])}" value="{esc(nm_val)}" required>'
                       f' <span class="cj-mut">{esc(src_note)}</span>')
        rows.append(
            f'<tr><td data-l="Sample"><label><input type="checkbox" name="samples" value="{gid}" '
            f'data-n="{len(g["names"])}"{" checked" if gid in picked else ""}> {esc(g["label"])}</label></td>'
            f'<td data-l="Tiles" class="cj-num">{len(g["names"])}</td>'
            f'<td data-l="nm/px">{nm_cell}</td></tr>')
    label_field = ""
    if t["column"] is None:
        label_field = (f'<label class="cj-predict-field"><span>Sample name (shown in the results)</span>'
                       f'<input name="label" maxlength="80" value="{esc(form.get("label", src["label"]))}" '
                       f'required></label>')
    how = form.get("how", "all")
    count = form.get("count", "")
    over = t["n"] > MAX_FILES
    err = msg_box("", extra=f'<div>{problem_list(error)}</div>') if error else ""
    return wrap(
        f'<div class="cj-predict"><p class="cj-mut"><a href="/predict/new/">← Back</a></p>'
        f'<h1>{esc(src["label"])}</h1><p class="cj-mut">Session {esc(src["session"])} · {t["n"]} tiles'
        f'</p>{tech(str(src["dir"])) if admin else ""}{err}'
        f'<form method="post" class="cj-predict-form" id="cj-predict-session" data-max="{MAX_FILES}">'
        f'<input type="hidden" name="sig" value="{esc(t["sig"])}">'
        f'<fieldset class="cj-card cj-predict-fs"><legend>Samples and pixel size</legend>{label_field}'
        f'<div class="cj-predict-tablewrap"><table class="cj-predict-table"><thead><tr><th>Sample</th><th>Tiles</th>'
        f'<th>nm per pixel</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        f'<label class="cj-predict-confirm"><input type="checkbox" name="confirm" value="yes" required> '
        f'I confirm the pixel size of each sample above</label>'
        f'<p class="cj-mut">Sizes in nm are computed from it. Your name is recorded as the person who confirmed it.'
        f'</p></fieldset>'
        f'<fieldset class="cj-card cj-predict-fs"><legend>Tiles</legend>'
        f'<label class="cj-predict-radio"><input type="radio" name="how" value="all"'
        f'{" checked" if how != "count" else ""}> All tiles of the ticked samples</label>'
        f'<label class="cj-predict-radio"><input type="radio" name="how" value="count"'
        f'{" checked" if how == "count" else ""}> Only <input class="cj-predict-count" name="count" '
        f'inputmode="numeric" aria-label="number of tiles" value="{esc(count)}"> tiles, spread across the '
        f'ticked samples</label>'
        f'<p class="cj-mut" id="cj-predict-sel" role="status"></p>'
        + (f'<p class="cj-mut">At most {MAX_FILES} tiles per analysis: choose a number.</p>' if over else "")
        + f'</fieldset>{model_block(cfg, user, form.get("model"))}'
        f'<p><button class="cj-predict-btn cj-predict-primary" type="submit" data-busy="Preparing your images… '
        f'this can take a minute">Start analysis</button></p></form></div>')


def page_upload_form(cfg, user, form=None, error=None) -> str:
    form = form or {}
    err = msg_box("", extra=f'<div>{problem_list(error)}</div>') if error else ""
    again = '<p class="cj-mut">Choose the files again (a browser forgets them after a refusal).</p>' if error else ""
    return wrap(
        f'<div class="cj-predict"><p class="cj-mut"><a href="/predict/new/">← Back</a></p>'
        f'<h1>Upload images</h1>{err}{again}'
        f'<form method="post" action="/predict/new/upload/" enctype="multipart/form-data" class="cj-predict-form" '
        f'id="cj-predict-upload" data-cj-upload="tiles" data-max-files="{MAX_FILES}" '
        f'data-max-bytes="{MAX_FILE_BYTES}">'
        f'<div class="cj-predict-msg cj-predict-t-warn" data-cj-upload-msg role="alert" hidden></div>'
        f'<noscript><p class="cj-predict-msg">JavaScript is off: this form then sends everything in one request, '
        f'at most {NOJS_MAX_BYTES // 2 ** 20} MB in total. Turn JavaScript on for bigger sets (files are then '
        f'sent one at a time).</p></noscript>'
        # the text fields come BEFORE the files: the browser sends parts in this order, so the server checks the
        # tick, nm/px and model as soon as the first file part arrives (fail fast, before any big file lands)
        f'<fieldset class="cj-card cj-predict-fs"><legend>Sample and pixel size</legend>'
        f'<label class="cj-predict-field"><span>Sample name (shown in the results)</span>'
        f'<input name="sample_label" maxlength="80" required value="{esc(form.get("sample_label", ""))}"></label>'
        f'<label class="cj-predict-field"><span>Pixel size (nm per pixel)</span>'
        f'<input class="cj-predict-nm" name="nm_per_px" inputmode="decimal" required '
        f'value="{esc(form.get("nm_per_px", ""))}"></label>'
        f'<p class="cj-mut">{esc(NM_HINT)}</p>'
        f'<label class="cj-predict-confirm"><input type="checkbox" name="confirm" value="yes" required> '
        f'I confirm the pixel size</label>'
        f'<p class="cj-mut">Sizes in nm are computed from it. Your name is recorded as the person who confirmed it.'
        f'</p></fieldset>{model_block(cfg, user, form.get("model"))}'
        f'<fieldset class="cj-card cj-predict-fs"><legend>Images</legend>'
        f'<label class="cj-predict-field"><span>Processed 8-bit PNG or JPEG tiles of one sample (up to {MAX_FILES}, '
        f'{MAX_FILE_BYTES // 2 ** 20} MB each)</span><input type="file" name="files" multiple required '
        f'accept=".png,.jpg,.jpeg,image/png,image/jpeg"></label>'
        f'<p class="cj-mut" id="cj-predict-files" data-cj-upload-info role="status"></p></fieldset>'
        f'<p><button class="cj-predict-btn cj-predict-primary" type="submit" data-busy="Uploading… keep this '
        f'page open">Upload and start</button></p></form></div>')


def page_set(cfg, user, d: Path, req: dict, can_see=None) -> str:
    admin = user.get("role") == "admin"
    v = set_view(cfg, user, d, req, True)
    rows = images_rows(d)
    srcs, _ = sources_safe(cfg, user, can_see)
    samples = "".join(f'<tr><td data-l="Sample">{esc(s["label"])}</td><td data-l="Images" class="cj-num">{s["n"]}'
                      f'</td><td data-l="nm/px">{esc(s["nm_per_px"])}</td>'
                      f'<td data-l="Confirmed by">{esc(s["confirmed_by"] or "?")}</td></tr>'
                      for s in samples_of(rows))
    owner = f'<div><dt>Owner</dt><dd>{esc(req.get("owner"))}</dd></div>' if admin else ""
    src_tech = (tech(f"{d} · source {json.dumps(req.get('source'))}") + tech(audit_problem(d.name))) if admin else ""
    return wrap(
        f'<div class="cj-predict" id="cj-predict-set" data-ps="{esc(d.name)}" data-live="{int(v["live"])}" '
        f'data-every="{REFRESH_S}"><p class="cj-mut"><a href="/predict/">← My analyses</a></p>'
        f'<h1>Analysis of {esc(when(req.get("created")))}</h1>'
        + (f'<p class="cj-mut cj-wrap">{esc(d.name)}</p>' if user.get("role") != "basic" else "") +
        f'<p id="cj-predict-err" class="cj-mut" role="status" hidden></p>'
        f'<section id="cj-predict-status">{v["html"]["status"]}</section>'
        f'<section id="cj-predict-actions">{v["html"]["actions"]}</section>'
        f'<h2 class="cj-sec">Details</h2><div class="cj-card"><dl class="cj-predict-kv">'
        f'<div><dt>Images from</dt><dd>{esc(source_text(req, srcs, admin=admin))}</dd></div>'
        f'<div><dt>Model</dt><dd>{esc(req.get("model"))}</dd></div>'
        f'<div><dt>Started</dt><dd>{esc(when(req.get("created")))}</dd></div>{owner}</dl>'
        f'<div class="cj-predict-tablewrap"><table class="cj-predict-table"><thead><tr><th>Sample</th><th>Images</th>'
        f'<th>nm/px</th><th>Confirmed by</th></tr></thead><tbody>{samples}</tbody></table></div>{src_tech}</div>'
        f'</div>')


# ---------------------------------------------------------------- creating sets
def _plain(msg: str, paths) -> str:
    """A refusal without our folder paths in it (the person never typed them)."""
    out = str(msg)
    for p in paths:
        for s in {str(p), str(Path(p).resolve())}:
            out = out.replace(s + os.sep, "").replace(s, "the images")
    return out


def check_nm(text, what) -> tuple:
    v = _num(str(text or "").strip().replace(",", "."))
    if v is None:
        return None, f"{what}: enter the pixel size as a number (nm per pixel), e.g. 1.47"
    if not NM_MIN <= v <= NM_MAX:
        return None, f"{what}: {v:g} nm per pixel is outside {NM_MIN:g}-{NM_MAX:g}; check the value"
    return v, None


def check_model(cfg, user, model) -> str | None:
    _, choices = allowed_models(cfg, user)
    if not choices:
        return NO_MODEL
    if model not in {c["id"] for c in choices}:
        return "Pick a model from the list."
    return None


def scratch(cfg, kind, raw=False) -> Path:
    """A fresh scratch folder under <root>/.portal; raw=True: under the raw root (a raw upload stays on the drive its
    conversion will use, so the move into raw_tiles/<id>/input is one os.replace on one drive)."""
    base = raw_base(cfg) if raw else predict_root(cfg)
    d = base / PORTAL_TMP / f"{kind}_{secrets.token_hex(6)}"
    d.mkdir(parents=True)
    return d


def create_and_start(cfg, d_from, user, model, *, names=None, manifest=None, sample_label=None, nm=None,
                     session=None) -> Path:
    """create() then start_detached(). Runs in a worker thread. Raises Refusal (plain) on a refusal."""
    root, reg = predict_root(cfg), registry_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    kw = dict(model=model, owner=user["id"], nm_confirmed_by=user["id"], registry_root=reg, root=root,
              session=session or None)
    if manifest is not None:
        kw.update(manifest=manifest, sample_label_column="manifest:sample_label", nm_from_manifest=True, names=names)
    else:
        kw.update(sample_label=sample_label, nm_per_px=nm)
    try:
        ps_dir = PS.create(d_from, **kw)
    except PS.PredictSetError as e:
        text = re.sub(r"^refused \((\d+) problem\(s\)\)", r"Not started: \1 problem(s)", _plain(str(e), [d_from, root]))
        raise Refusal(re.sub(r"^refused: ", "Not started: ", text), 400, tech=str(e))
    try:
        PL.start_detached(ps_dir, root=root, registry_root=reg)
    except Exception as e:                       # the set exists (queued): its page offers "Start again"
        PS.write_status(ps_dir, waiting=f"could not start the run: {type(e).__name__}: {e}")
    return ps_dir


def read_tiles(src: dict) -> dict:
    """tiles_of(), with an unreadable folder or manifest as a plain refusal (details for the admin)."""
    try:
        return tiles_of(src)
    except (OSError, csv.Error, UnicodeDecodeError) as e:
        raise Refusal("This session’s tile list can’t be read right now. Try again later, or ask the admin.", 503,
                      tech=f"{type(e).__name__}: {e}")


def active_sets(cfg, user) -> int:
    """This person's own sets that are queued or running."""
    return sum(1 for _, req, st in list_sets(cfg, user, lambda u, owner: owner == u.get("id"))
               if st.get("state") in ("queued", "running"))


def check_active(cfg, user):
    """Basic users may have at most MAX_ACTIVE_BASIC sets queued or running (failed and done sets don't count)."""
    if user.get("role") == "basic" and active_sets(cfg, user) >= MAX_ACTIVE_BASIC:
        raise Refusal(f"You already have {MAX_ACTIVE_BASIC} analyses waiting or running. Start a new one when one "
                      f"of them is done.", 429)


def session_submit(cfg, user, src: dict, form: dict) -> Path:
    if form.get("confirm") != "yes":
        raise Refusal("Tick “I confirm the pixel size”: nothing was started.")
    check_active(cfg, user)
    t = read_tiles(src)
    if form.get("sig") != t["sig"]:
        raise Refusal("The session’s tiles changed since this page was opened. Check the samples again.")
    problems = []
    chosen = []
    for i, g in enumerate(t["groups"]):
        gid = f"g{i}"
        if gid not in (form.get("samples") or []):
            continue
        if g["mixed"]:
            problems.append(f"{g['label']}: pixel sizes differ by more than {NM_TOLERANCE * 100:g} %; it can't be "
                            f"analysed as one sample")
            continue
        typed = str(form.get(f"nm_{gid}") or "").strip()
        if g["nm_range"] and not typed:          # each tile keeps its own manifest value (confirmed by the tick)
            bad = [v for v in (g["nm_min"], g["nm_max"]) if not NM_MIN <= v <= NM_MAX]
            if bad:
                problems.append(f"{g['label']}: {bad[0]:g} nm per pixel (from the manifest) is outside "
                                f"{NM_MIN:g}-{NM_MAX:g}; check the value, or type the right one")
                continue
            chosen.append({**g, "nm_in": None})
            continue
        nm, err = check_nm(typed, g["label"])
        if err:
            problems.append(err)
        chosen.append({**g, "nm_in": nm})
    if not chosen and not problems:
        problems.append("Tick at least one sample.")
    label = None
    if t["column"] is None:
        label = str(form.get("label") or "").strip()
        if not label or len(label) > 80:
            problems.append("Sample name: 1-80 characters.")
    count = None
    if form.get("how") == "count":
        try:
            count = int(str(form.get("count") or "").strip())
            if count < 1:
                raise ValueError
        except ValueError:
            problems.append("Number of tiles: a whole number of 1 or more.")
    m_err = check_model(cfg, user, form.get("model"))
    if m_err:
        problems.append(m_err)
    if problems:
        raise Refusal(f"Please fix ({len(problems)}):\n" + "\n".join(problems))
    names = pick(chosen, count)
    if len(names) > MAX_FILES:
        raise Refusal(f"{len(names)} tiles chosen; an analysis takes at most {MAX_FILES}. Choose “Only ... "
                      f"tiles” with a number up to {MAX_FILES}, or fewer samples.")
    # a typed sample name keeps the tile-size suffix of a split sample (one sample per scale in the results)
    label_of = {n: (label + g["suffix"] if label is not None else g["label"]) for g in chosen for n in g["names"]}
    # a typed value applies to every tile of its sample; a range sample left empty keeps each tile's own value
    nm_of = {n: (g["nm_in"] if g["nm_in"] is not None else g["nm_of"][n]) for g in chosen for n in g["names"]}
    # recorded: a typed value that replaced several manifest values (a range), or manifest values of a sample where
    # some tiles had none
    overrides = [{"sample": label_of[g["names"][0]], "typed_nm_per_px": g["nm_in"],
                  "manifest_nm_per_px": nm_range_text(g["nm_min"], g["nm_max"])
                  + (f" ({g['nm_missing']} tile(s) without a value)" if g["nm_missing"] else ""),
                  "tiles": len(set(names) & set(g["names"]))}
                 for g in chosen if g["nm_in"] is not None and (g["nm_range"] or (g["nm_missing"] and g["nm_values"]))]
    tmp = scratch(cfg, "sel")
    try:
        man = tmp / "MANIFEST.csv"
        with open(man, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["output_name", "sample_label", PS.NM_COLUMN])
            for n in names:
                w.writerow([n, label_of[n], repr(float(nm_of[n]))])
        ps_dir = create_and_start(cfg, src["dir"], user, form["model"], names=names, manifest=man,
                                  session=src["session"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for o in overrides:
        record_nm_override(cfg, user, ps_dir.name, o)
    return ps_dir


_AUDIT_FAILED: dict = {}                 # ps id -> why its nm/px override record was not written (admin sees it)
_AUDIT_LOCK = threading.Lock()


def record_nm_override(cfg, user, ps_id: str, detail: dict) -> bool:
    """A person typed one nm/px for a sample whose tiles had other manifest values (a range, or some tiles with none):
    every tile of it got the typed value in images.csv. Recorded in the portal audit log
    (<state dir>/audit/portal.jsonl) as predict.nm_override. -> written. A failure does not undo the analysis (it
    already exists): it is logged to the console and shown to the admin on the set's page (audit_problem)."""
    state = getattr(cfg, "state_dir", None)
    try:
        if not state:
            raise OSError("the portal has no state folder configured")
        from portal import audit as PA
        PA.audit(state, user.get("id"), "predict.nm_override", ps_id, detail)
        return True
    except OSError as e:
        why = (f"nm/px override not written to the audit log: {type(e).__name__}: {e}; "
               f"{json.dumps(detail, ensure_ascii=False)}")
        print(f"[predict] {ps_id}: {why}", file=sys.stderr, flush=True)
        with _AUDIT_LOCK:
            _AUDIT_FAILED[ps_id] = why
            for k in list(_AUDIT_FAILED)[:max(0, len(_AUDIT_FAILED) - CACHE_MAX)]:
                _AUDIT_FAILED.pop(k, None)
        return False


def audit_problem(ps_id: str) -> str | None:
    with _AUDIT_LOCK:
        return _AUDIT_FAILED.get(ps_id)


# ---------------------------------------------------------------- upload (multipart, streamed to disk)
def safe_name(raw: str) -> str | None:
    name = re.split(r"[\\/]", raw or "")[-1].strip()
    if (not name or name in (".", "..") or name.startswith(".") or len(name) > 150 or NAME_BAD.search(name)
            or WIN_RESERVED.match(name) or name.endswith((" ", "."))):
        return None
    return name


def disk_free(path) -> int:
    return shutil.disk_usage(path).free


_RESERVED: dict = {}                     # upload id -> bytes it may still write (its Content-Length minus written)
_RES_DEV: dict = {}                      # upload id -> the drive (st_dev) its hold is on
_RES_LOCK = threading.Lock()


def _dev(path):
    """The volume a path is (or would be) on; None when it can't be told."""
    try:
        return os.stat(_existing(path)).st_dev
    except OSError:
        return None


def _held(dev) -> int:
    """Bytes the uploads in flight may still write on this drive. Callers hold _RES_LOCK."""
    return sum(n for k, n in _RESERVED.items() if _RES_DEV.get(k) == dev)


class Reservation:
    """Disk space held for one upload while it streams. check() refuses (507) when the free space minus what every
    upload in flight may still write would drop below FREE_MARGIN_BYTES. A chunked upload (no Content-Length)
    holds nothing ahead; the checks during the upload still see the space it has used."""

    def __init__(self, dest: Path, length: int):
        self.dest, self.key = dest, secrets.token_hex(8)
        self.dev = _dev(dest)
        with _RES_LOCK:
            _RESERVED[self.key] = max(0, int(length or 0))
            _RES_DEV[self.key] = self.dev

    def written(self, n: int):
        with _RES_LOCK:
            if self.key in _RESERVED:
                _RESERVED[self.key] = max(0, _RESERVED[self.key] - n)

    def check(self):
        free = disk_free(self.dest)
        with _RES_LOCK:
            held = _held(self.dev)
        if free - held < FREE_MARGIN_BYTES:
            raise Refusal("Not enough disk space on the server for this upload. Nothing was kept. Ask the admin.",
                          507, tech=f"{free} bytes free, {held} held by uploads in flight, "
                                    f"margin {FREE_MARGIN_BYTES}")

    def release(self):
        with _RES_LOCK:
            _RESERVED.pop(self.key, None)
            _RES_DEV.pop(self.key, None)


class UploadSink:
    """Takes the parser's events (in order) and writes file parts to `dest` under their checked names; enforces
    the caps; re-checks the disk at each file start and every DISK_CHECK_BYTES; calls first_file(fields) when the
    first file part starts (the form puts the text fields first, so a refusal comes before any big file lands).
    Runs in a worker thread (file writes)."""

    def __init__(self, dest: Path, reservation: Reservation | None = None, first_file=None):
        self.dest, self.fields, self.files, self.seen = dest, {}, [], set()
        self.cur, self.fh, self.size, self.n_fields = None, None, 0, 0
        self.res, self.first_file, self.since_check = reservation, first_file, 0

    def apply(self, events):
        for ev in events:
            kind = ev[0]
            if kind == "part":
                _, name, filename = ev
                if filename is None:
                    self.n_fields += 1
                    if self.n_fields > MAX_FIELDS:
                        raise Refusal("Too many form fields.", 413)
                    self.cur = {"kind": "field", "name": name, "buf": bytearray()}
                elif filename == "":
                    self.cur = {"kind": "skip"}              # an empty file input
                else:
                    if name != "files":
                        raise Refusal("Unexpected file field.", 400)
                    if not self.files and self.first_file is not None:
                        self.first_file(self.fields)
                    if self.res is not None:
                        self.res.check()
                        self.since_check = 0
                    if len(self.files) >= MAX_FILES:
                        raise Refusal(f"More than {MAX_FILES} files: an analysis takes at most {MAX_FILES}. "
                                      f"Nothing was kept.", 413)
                    safe = safe_name(filename)
                    if safe is None:
                        raise Refusal(f"File name not accepted: {filename[:80]!r}. Rename it (letters, digits, "
                                      f"- _ . and spaces) and upload again.", 400)
                    if safe.lower() in self.seen:
                        raise Refusal(f"Two files are named {safe!r}; each name must be unique.", 400)
                    self.seen.add(safe.lower())
                    self.files.append(safe)
                    self.fh = open(self.dest / safe, "xb")
                    self.size = 0
                    self.cur = {"kind": "file", "name": safe}
            elif kind == "data":
                c = self.cur or {}
                if c.get("kind") == "file":
                    self.size += len(ev[1])
                    if self.size > MAX_FILE_BYTES:
                        raise Refusal(f"{c['name']} is over {MAX_FILE_BYTES // 2 ** 20} MB. Nothing was kept.", 413)
                    self.fh.write(ev[1])
                    if self.res is not None:
                        self.res.written(len(ev[1]))
                        self.since_check += len(ev[1])
                        if self.since_check >= DISK_CHECK_BYTES:
                            self.res.check()
                            self.since_check = 0
                elif c.get("kind") == "field":
                    c["buf"] += ev[1]
                    if len(c["buf"]) > MAX_FIELD_BYTES:
                        raise Refusal("A form field is too long.", 413)
            elif kind == "end":
                c = self.cur or {}
                if c.get("kind") == "file":
                    self.fh.close()
                    self.fh = None
                elif c.get("kind") == "field":
                    self.fields.setdefault(c["name"], c["buf"].decode("utf-8", "replace"))
                self.cur = None

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None


async def receive_upload(request: Request, dest: Path, first_file=None) -> UploadSink:
    """Stream a multipart/form-data body into `dest` (python-multipart's push parser; file writes and checks in a
    worker thread). Raises Refusal on a bad body, a cap, low disk space, or first_file's refusal."""
    from python_multipart.multipart import parse_options_header
    ctype, params = parse_options_header(request.headers.get("content-type"))
    boundary = params.get(b"boundary")
    if ctype != b"multipart/form-data" or not boundary:
        raise Refusal("Expected a file upload (multipart/form-data).", 400)
    try:
        length = int(request.headers.get("content-length") or 0)
    except ValueError:
        length = 0
    limit = min(MAX_FILES * (MAX_FILE_BYTES + 4096) + MAX_FIELDS * (MAX_FIELD_BYTES + 1024), NOJS_MAX_BYTES)
    if length > limit:
        raise Refusal(nojs_too_big(), 413)
    res = Reservation(dest, length)
    try:
        await run_in_threadpool(res.check)
        return await _stream(request, boundary, dest, res, limit, first_file)
    finally:
        res.release()


async def _stream(request: Request, boundary: bytes, dest: Path, res: Reservation, limit: int,
                  first_file) -> UploadSink:
    from python_multipart.multipart import MultipartParser, parse_options_header
    events, hdr = [], {"field": b"", "value": b"", "headers": {}}

    def on_part_begin():
        hdr["headers"] = {}

    def on_header_field(data, start, end):
        hdr["field"] += data[start:end]

    def on_header_value(data, start, end):
        hdr["value"] += data[start:end]

    def on_header_end():
        hdr["headers"][hdr["field"].lower()] = hdr["value"]
        hdr["field"], hdr["value"] = b"", b""

    def on_headers_finished():
        _, opts = parse_options_header(hdr["headers"].get(b"content-disposition"))
        name = opts.get(b"name", b"").decode("utf-8", "replace")
        fn = opts.get(b"filename")
        events.append(("part", name, None if fn is None else fn.decode("utf-8", "replace")))

    def on_part_data(data, start, end):
        events.append(("data", bytes(data[start:end])))

    def on_part_end():
        events.append(("end",))

    callbacks = {"on_part_begin": on_part_begin, "on_header_field": on_header_field,
                 "on_header_value": on_header_value, "on_header_end": on_header_end,
                 "on_headers_finished": on_headers_finished, "on_part_data": on_part_data, "on_part_end": on_part_end}
    parser = MultipartParser(boundary, callbacks)
    sink = UploadSink(dest, res, first_file)
    total = 0
    try:
        async for chunk in request.stream():
            total += len(chunk)
            if total > limit:
                raise Refusal(nojs_too_big(), 413)
            try:
                parser.write(chunk)
            except Exception as e:
                raise Refusal("The upload could not be read; try again.", 400, tech=str(e))
            if events:
                batch, events[:] = list(events), []
                await run_in_threadpool(sink.apply, batch)
        parser.finalize()
        if events:
            await run_in_threadpool(sink.apply, list(events))
    finally:
        sink.close()
    return sink


def upload_fields(cfg, user, f: dict, files=True) -> tuple:
    """The upload form's text fields -> (label, nm). Raises Refusal listing every problem. Called when the first
    file part arrives (files=True there: the files are coming) and again at the end."""
    problems = []
    if f.get("confirm") != "yes":
        raise Refusal("Tick “I confirm the pixel size”: nothing was started.")
    if not files:
        problems.append("Choose at least one image.")
    label = str(f.get("sample_label") or "").strip()
    if not label or len(label) > 80:
        problems.append("Sample name: 1-80 characters.")
    nm, err = check_nm(f.get("nm_per_px"), "Pixel size")
    if err:
        problems.append(err)
    m_err = check_model(cfg, user, f.get("model"))
    if m_err:
        problems.append(m_err)
    if problems:
        raise Refusal(f"Please fix ({len(problems)}):\n" + "\n".join(problems))
    return label, nm


def upload_submit(cfg, user, dest: Path, sink: UploadSink) -> Path:
    label, nm = upload_fields(cfg, user, sink.fields, bool(sink.files))
    check_active(cfg, user)
    return create_and_start(cfg, dest, user, sink.fields["model"], sample_label=label, nm=nm)


def nojs_too_big() -> str:
    return (f"Without JavaScript, one upload is at most {NOJS_MAX_BYTES // 2 ** 20} MB in total (the link to the "
            f"server caps one request at about 100 MB). Nothing was kept. Turn JavaScript on (the page then sends "
            f"the files one at a time), or upload fewer files.")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- raw conversion: settings, recipes, checks
def test_hooks() -> dict:
    """jobs/run_job.py's test hooks (only when CRYOEV_JOBS_TESTING=1), read here without importing run_job (it puts
    jobs/ and training/ on sys.path, which the portal process must not get)."""
    if os.environ.get(TEST_HOOKS[0]) != "1":
        return {}
    try:
        h = json.loads(os.environ.get(TEST_HOOKS[1]) or "{}")
        f = os.environ.get(TEST_HOOKS[2])
        if f and Path(f).is_file():
            h.update(json.loads(Path(f).read_text(encoding="utf-8")))
        return h if isinstance(h, dict) else {}
    except (ValueError, OSError):
        return {}


def processing_repo(cfg) -> tuple:
    """-> (repo Path, None) or (None, why): CJ_PROCESSING_REPO, checked the way jobs/convert_raw.py checks it."""
    v = _setting(cfg, CR.REPO_ENV)
    if not v:
        return None, f"{CR.REPO_ENV} is not set: raw conversion is off"
    try:
        return CR.processing_repo(str(v)), None
    except CR.ConvertError as e:
        return None, str(e)


_RECIPES: dict = {}
_RECIPE_LOCK = threading.Lock()


def recipes(repo) -> list:
    """The processing repo's session recipes (datasets/process_<name>.py) that convert_raw's static check accepts,
    newest name first. Read only and never executed; cached RECIPE_TTL_S."""
    d = Path(repo) / "datasets"
    key = os.path.normcase(str(d))
    with _RECIPE_LOCK:
        hit = _RECIPES.get(key)
        if hit and 0 <= time.monotonic() - hit[0] < RECIPE_TTL_S:
            return list(hit[1])
    names = []
    try:
        files = sorted(d.glob("process_*.py"))
    except OSError:
        files = []
    for p in files:
        name = p.stem[len("process_"):]
        if not CR.RECIPE_RE.match(name):
            continue
        try:
            CR.check_recipe_source(p, name)
        except (CR.ConvertError, OSError):
            continue
        names.append(name)
    names.sort(reverse=True)
    with _RECIPE_LOCK:
        _RECIPES.clear()
        _RECIPES[key] = (time.monotonic(), names)
    return list(names)


def session_recipe(session, names) -> str | None:
    """The recipe named after this session (20260827-2 -> 20260827_2), when the repo has one."""
    want = str(session or "").replace("-", "_")
    return want if want in names else None


def valid_session_id(s) -> bool:
    return isinstance(s, str) and PSRC.session_of(s) == s and bool(PS.SESSION_RE.match(s))


def raw_fields(cfg, user, f: dict, session=None) -> dict:
    """The raw form's fields -> {session, recipe, gpu, repo}. Raises Refusal (403 for a basic user, 503 when the
    processing repo is not set up, else 400 listing every problem). session: fixed by a picked session folder."""
    if user.get("role") not in RAW_ROLES:
        raise Refusal("Converting raw images is for advanced users and the admin.", 403)
    repo, why = processing_repo(cfg)
    if repo is None:
        raise Refusal("Raw conversion is not set up on this server. Ask the admin.", 503, tech=why)
    problems = []
    sid = session if session is not None else str(f.get("session") or "").strip()
    if not valid_session_id(sid):
        problems.append("Session: the imaging date as YYYYMMDD, or YYYYMMDD-n for a later session that day "
                        "(e.g. 20261027 or 20260827-2).")
    recipe = str(f.get("recipe") or "default").strip()
    if recipe != "default" and recipe not in recipes(repo):
        problems.append("Recipe: pick one from the list.")
    if problems:
        raise Refusal(f"Please fix ({len(problems)}):\n" + "\n".join(problems))
    return {"session": sid, "recipe": recipe, "gpu": f.get("gpu") == "yes", "repo": repo}


def gpu_check() -> tuple:
    """jobs/convert_raw.gpu_check() (= jobs/predict_local.py's) in a short child process, so the portal never imports
    jobs/run_job.py. -> (ok, note)."""
    code = "import json; from jobs import convert_raw as CR; print(json.dumps(list(CR.gpu_check())))"
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                           timeout=GPU_CHECK_S, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        ok, note = json.loads(r.stdout.strip().splitlines()[-1])
        return bool(ok), str(note)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as e:
        return False, f"GPU check failed: {type(e).__name__}: {e}"


def check_gpu(fields: dict):
    if fields.get("gpu"):
        ok, note = gpu_check()
        if not ok:
            raise Refusal("The GPU can’t be used now. Untick “Use the GPU” to convert on the CPU, or try later.",
                          409, tech=note)


def recipe_select(repo, chosen=None, session=None) -> str:
    """The recipe <select> (default first) and the warning about the default."""
    names = recipes(repo) if repo else []
    chosen = chosen if chosen in names else "default"
    opts = (f'<option value="default"{" selected" if chosen == "default" else ""}>default ({esc(CR.DEFAULT_RECIPE)}, '
            f'the lab’s template)</option>' +
            "".join(f'<option value="{esc(n)}"{" selected" if n == chosen else ""}>{esc(n)}</option>' for n in names))
    own = session_recipe(session, names) if session else None
    hint = (f' This session has its own recipe: <b>{esc(own)}</b>.' if own else
            ' A session’s own recipe is named after it (20260827-2 → 20260827_2).')
    return (f'<label class="cj-predict-field"><span>Recipe (the lab’s processing settings for a session: denoise, '
            f'contrast)</span><select name="recipe" data-cj-recipe>{opts}</select></label>'
            f'<p class="cj-predict-warn" data-cj-recipe-warn>“default” is the lab’s template ({esc(CR.DEFAULT_RECIPE)}): '
            f'its contrast and denoise were tuned on that session, and a sampling far from it is refused. Pick the '
            f'recipe of the session these images come from when there is one.{hint}</p>')


def gpu_box(checked=False) -> str:
    return (f'<label class="cj-predict-radio"><input type="checkbox" name="gpu" value="yes"'
            f'{" checked" if checked else ""}> Use the GPU (faster; only if it is free. Results differ from the CPU '
            f'by ±1 grey level on under 0.01 % of pixels)</label>')


# ---------------------------------------------------------------- raw conversion: records and the runner
def raw_base(cfg) -> Path:
    """The folder holding the whole raw pipeline (.portal staging of raw uploads + raw_tiles/): cfg.raw_root, else
    CJ_RAW_ROOT, else the predict root (today's behaviour)."""
    v = getattr(cfg, "raw_root", None) or os.environ.get("CJ_RAW_ROOT")
    return Path(v) if v else predict_root(cfg)


def raw_root(cfg) -> Path:
    return raw_base(cfg) / RAW_TILES


def _same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def raw_roots(cfg) -> list:
    """Where conversions are listed from: the raw root, then (when CJ_RAW_ROOT moved it) the old
    <predict root>/raw_tiles, whose conversions stay listed, swept and usable."""
    new, old = raw_root(cfg), predict_root(cfg) / RAW_TILES
    return [new] if _same_path(new, old) else [new, old]


def tmp_dirs(cfg) -> list:
    """The .portal scratch folders: the predict root's, and the raw root's when that is somewhere else."""
    a, b = predict_root(cfg) / PORTAL_TMP, raw_base(cfg) / PORTAL_TMP
    return [a] if _same_path(a, b) else [a, b]


def conv_dir(cfg, cid) -> Path | None:
    if not isinstance(cid, str) or not RAW_ID_RE.match(cid):
        return None
    for root in raw_roots(cfg):
        d = root / cid
        if (d / CONVERT_JSON).is_file():
            return d
    return None


def load_conversion(cfg, user, can_see, cid) -> tuple:
    """-> (folder, convert.json, status) when it exists AND this person may see it; else (None, None, None)."""
    d = conv_dir(cfg, cid)
    meta = _read_json(d / CONVERT_JSON) if d else None
    if not isinstance(meta, dict) or not can_see(user, meta.get("owner")):
        return None, None, None
    return d, meta, _read_json(d / PS.STATUS) or {}


def all_conversions(cfg) -> list:
    """[(folder, convert.json, status)] of every conversion, oldest first."""
    dirs = []
    for root in raw_roots(cfg):
        try:
            dirs += [p for p in root.iterdir() if p.is_dir() and RAW_ID_RE.match(p.name)]
        except OSError:
            continue
    out = []
    for d in dirs:
        meta = _read_json(d / CONVERT_JSON)
        if isinstance(meta, dict):
            out.append((d, meta, _read_json(d / PS.STATUS) or {}))
    out.sort(key=lambda t: (str(t[1].get("created") or ""), t[0].name))
    return out


def list_conversions(cfg, user, can_see) -> list:
    """This person's conversions (admin: all), newest first."""
    return [t for t in reversed(all_conversions(cfg)) if can_see(user, t[1].get("owner"))]


def write_conv_status(d: Path, **fields) -> dict:
    """Merge fields into the conversion's status.json (atomic). Callers hold the runner lock, or own the folder
    (just made)."""
    st = _read_json(d / PS.STATUS) or {}
    st.update(fields, updated=now_iso())
    PS.write_json_atomic(d / PS.STATUS, st)
    return st


def keep_days(cfg) -> float:
    """CJ_RAW_KEEP_DAYS: days an uploaded conversion keeps its raw input/ after it ends (done or failed); 0 = removed
    as soon as it ends. Default KEEP_DEFAULT_DAYS; an unreadable value = the default."""
    v = _setting(cfg, KEEP_SETTING)
    try:
        x = float(v) if v not in (None, "") else KEEP_DEFAULT_DAYS
    except (TypeError, ValueError):
        return float(KEEP_DEFAULT_DAYS)
    return max(0.0, x) if math.isfinite(x) else float(KEEP_DEFAULT_DAYS)


def _existing(p: Path) -> Path:
    p = Path(p)
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def input_bytes(meta: dict) -> int:
    """The raw bytes a conversion reads: recorded at creation, else estimated from its sidecars (the MRC beside an
    .mrc.mdoc; for an .idoc, the tiles it names that are plain files beside it)."""
    if isinstance(meta.get("input_bytes"), int):
        return meta["input_bytes"]
    total = 0
    for s in meta.get("inputs") or []:
        side = Path(s)
        try:
            total += side.stat().st_size
            if side.name.lower().endswith(".mrc.mdoc"):
                mrc = side.with_name(side.name[:-len(".mdoc")])
                total += mrc.stat().st_size if mrc.is_file() else 0
            else:
                for n in sidecar_images(side):
                    if image_ref_problem(n) is None and (side.parent / n).is_file():
                        total += (side.parent / n).stat().st_size
        except OSError:
            continue
    return total


def check_disk_for(where: Path, nbytes: int):
    """A conversion writes its work copies and tiles next to the predict root: refuse (507) unless about
    DISK_FACTOR x its raw input is free on top of FREE_MARGIN_BYTES and the uploads in flight."""
    free = disk_free(_existing(where))
    dev = _dev(where)
    with _RES_LOCK:
        held = _held(dev)
    need = DISK_FACTOR * int(nbytes or 0) + FREE_MARGIN_BYTES
    if free - held < need:
        raise Refusal(f"Not enough disk space on the server to convert these (about "
                      f"{DISK_FACTOR * int(nbytes or 0) / 2 ** 30:.1f} GB needed). Ask the admin.", 507,
                      tech=f"{free} bytes free, {held} held by uploads, need {need}")


def check_conv_cap(cfg, user):
    """At most MAX_CONV_PER_USER conversions of one person queued or running."""
    mine = [1 for _, meta, st in all_conversions(cfg)
            if meta.get("owner") == user.get("id") and st.get("state") in ("queued", "running")]
    if len(mine) >= MAX_CONV_PER_USER:
        raise Refusal(f"You already have {MAX_CONV_PER_USER} conversions waiting or running. Start another when "
                      f"one of them is done.", 429)


def move_tree(src: Path, dst: Path):
    """Move a folder: os.replace on one drive; across drives (EXDEV, e.g. an upload staged before CJ_RAW_ROOT was set)
    copy, verify every file's size, then remove the source. A failure leaves the source whole and no dst."""
    try:
        os.replace(src, dst)
        return
    except OSError as e:
        if e.errno != errno.EXDEV and not (os.name == "nt" and getattr(e, "winerror", None) in (17, 5)):
            raise
        if Path(dst).exists() and any(Path(dst).iterdir()):
            raise
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        for a in Path(src).rglob("*"):
            if a.is_file() and (Path(dst) / a.relative_to(src)).stat().st_size != a.stat().st_size:
                raise OSError(f"copy of {a.name} has a different size")
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    shutil.rmtree(src, ignore_errors=True)


def new_conversion(cfg, user, fields: dict, *, source: dict, inputs=None, upload_files: Path | None = None,
                   sample_key=None, nbytes=None) -> Path:
    """Make <root>/raw_tiles/<session>_<6 hex>/ with convert.json + status queued (uploaded files moved into
    input/), then start it when nothing else runs. -> the folder."""
    root = raw_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(50):
        name = f"{fields['session']}_{secrets.token_hex(3)}"
        if any((r / name).exists() for r in raw_roots(cfg)):      # free in BOTH roots
            continue
        d = root / name
        try:
            d.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise Refusal("Could not make a folder for the conversion. Try again.", 500)
    try:
        if upload_files is not None:
            inp = d / CONVERT_INPUT
            # same drive: staging (.portal) and raw_tiles/ are both under the raw root; an upload staged under the
            # predict root before CJ_RAW_ROOT was set is copied across (move_tree), never lost
            move_tree(upload_files, inp)
            names = sorted(p.name for p in inp.iterdir() if p.is_file())
            inputs = [str(inp / n) for n in names if n.lower().endswith((".mrc.mdoc", ".idoc"))]
            keys = [n for n in names if n.lower().endswith(".csv")]
            sample_key = str(inp / keys[0]) if keys else None
        meta = {"schema": SCHEMA_CONVERT, "id": d.name, "owner": user["id"], "session": fields["session"],
                "recipe": fields["recipe"], "gpu": bool(fields["gpu"]), "source": source,
                "inputs": [str(x) for x in inputs or []], "sample_key": str(sample_key) if sample_key else None,
                "created": now_iso()}
        meta["input_bytes"] = int(nbytes) if nbytes is not None else input_bytes(meta)
        PS.write_json_atomic(d / CONVERT_JSON, meta)
        write_conv_status(d, state="queued", pid=None, pid_created=None, started=None, finished=None, error=None,
                          kind=None, warnings=[], result=None, n_tiles=None)
    except BaseException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    kick(cfg)
    ensure_watcher(cfg)
    return d


def convert_argv(d: Path, meta: dict, repo: Path) -> list:
    """`python -m jobs.convert_raw --in ... --out <d>/tiles --recipe <r> [--sample-key k] [--gpu]
    --processing-repo <repo>` (tests: the hooks' convert_cmd in place of `python -m jobs.convert_raw`)."""
    hook = test_hooks().get("convert_cmd")
    cmd = [str(x) for x in hook] if isinstance(hook, list) and hook else [sys.executable, "-m", "jobs.convert_raw"]
    argv = cmd + ["--in", *meta["inputs"], "--out", str(d / CONVERT_TILES), "--recipe", str(meta["recipe"])]
    if meta.get("sample_key"):
        argv += ["--sample-key", str(meta["sample_key"])]
    if meta.get("gpu"):
        argv.append("--gpu")
    return argv + ["--processing-repo", str(repo)]


# -- the converter's process: identity by pid AND start time (a pid can be reused by another program)
def proc_created(pid) -> float | None:
    """The start time (epoch s) of process pid, or None when it can't be read (no psutil, gone, no access)."""
    try:
        import psutil
        return float(psutil.Process(int(pid)).create_time())
    except Exception:
        return None


def proc_alive(st: dict) -> bool:
    """The converter recorded in status.json is still running: its pid is alive AND that process started when the
    converter did (pid_created; else within PID_START_SLACK_S of `started`). A reused pid is not it."""
    pid = st.get("pid")
    if not pid or not PL.pid_alive(pid):
        return False
    t = proc_created(pid)
    if t is None:
        return True                                     # can't tell: trust the pid (the max runtime still applies)
    if st.get("pid_created") is not None:
        try:
            return abs(t - float(st["pid_created"])) <= 2.0
        except (TypeError, ValueError):
            pass
    started = _parse_local(st.get("started"))
    return started is None or abs(t - started) <= PID_START_SLACK_S


def kill_proc(st: dict) -> bool:
    """Kill the converter (and anything it started) when proc_alive(st). -> killed."""
    if not proc_alive(st):
        return False
    pid = int(st["pid"])
    try:
        import psutil
        p = psutil.Process(pid)
        for c in p.children(recursive=True):
            try:
                c.kill()
            except psutil.Error:
                pass
        p.kill()
        p.wait(10)
        return True
    except ImportError:
        pass
    except Exception:
        return not PL.pid_alive(pid)
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=30,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            os.kill(pid, 9)
    except (OSError, subprocess.SubprocessError):
        pass
    return not PL.pid_alive(pid)


def overdue(d: Path, st: dict, now=None) -> str | None:
    """Why a running conversion should be stopped: over CONVERT_MAX_S since it started, or its log silent for
    CONVERT_STALE_S. None while it is fine."""
    now = time.time() if now is None else now
    started = _parse_local(st.get("started"))
    if started is None:
        return None
    if now - started > CONVERT_MAX_S:
        return f"It ran for over {CONVERT_MAX_S / 3600:g} h, so it was stopped."
    try:
        last = max(started, (d / CONVERT_LOG).stat().st_mtime)
    except OSError:
        last = started
    if now - last > CONVERT_STALE_S:
        return f"No progress for {CONVERT_STALE_S // 60} min (its log stopped), so it was stopped."
    return None


def start_conversion(cfg, d: Path, meta: dict) -> dict:
    """Spawn the converter detached (no console window; it outlives the portal): stdout -> convert.out, stderr ->
    convert.log. -> the new status. Refused (failed) without a processing repo, a montage or the disk it needs."""
    repo, why = processing_repo(cfg)
    if repo is None:
        return write_conv_status(d, state="failed", kind="refused", finished=now_iso(),
                                 error=f"Raw conversion is not set up: {why}")
    if not meta.get("inputs"):
        return write_conv_status(d, state="failed", kind="refused", finished=now_iso(), error="No montage to convert.")
    try:
        check_disk_for(d, input_bytes(meta))
    except Refusal as e:
        return write_conv_status(d, state="failed", kind="refused", finished=now_iso(), error=e.msg)
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    env = {**os.environ, CR.REPO_ENV: str(repo), "PYTHONIOENCODING": "utf-8"}
    try:
        with open(d / CONVERT_OUT, "wb") as out, open(d / CONVERT_LOG, "ab") as log:
            p = subprocess.Popen(convert_argv(d, meta, repo), cwd=REPO, stdin=subprocess.DEVNULL, stdout=out,
                                 stderr=log, env=env, close_fds=True, **kw)
    except OSError as e:
        return write_conv_status(d, state="failed", kind="failed", finished=now_iso(),
                                 error=f"could not start the converter: {type(e).__name__}: {e}")
    return write_conv_status(d, state="running", pid=p.pid, pid_created=proc_created(p.pid), started=now_iso())


def converter_result(d: Path):
    """The converter's one JSON object from convert.out, or None (not finished, or nothing printed)."""
    try:
        text = (d / CONVERT_OUT).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                doc = json.loads(line)
            except ValueError:
                return None
            return doc if isinstance(doc, dict) and "ok" in doc else None
    return None


def log_tail(d: Path, n=4000) -> str:
    try:
        with open(d / CONVERT_LOG, "rb") as f:
            f.seek(max(0, os.fstat(f.fileno()).st_size - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def finalize(d: Path, meta: dict, st: dict) -> dict:
    """A running conversion: its result once printed (done / failed); failed when its process is gone without one;
    stopped (killed, failed "timeout") when overdue(); else unchanged."""
    res = converter_result(d)
    if res is None:
        if proc_alive(st):
            why = overdue(d, st)
            if why is None:
                return st
            kill_proc(st)
            return write_conv_status(d, state="failed", kind="timeout", pid=None, finished=now_iso(), error=why)
        res = converter_result(d)                       # it may have printed just before it ended
        if res is None:
            return write_conv_status(d, state="failed", kind="failed", pid=None, finished=now_iso(),
                                     error="The converter stopped without a result.")
    warnings = [str(w) for w in (res.get("warnings") or [])]
    if res.get("ok"):
        return write_conv_status(d, state="done", pid=None, finished=now_iso(), error=None, kind=None,
                                 warnings=warnings, result=res, n_tiles=res.get("n_tiles"))
    return write_conv_status(d, state="failed", pid=None, finished=now_iso(), kind=res.get("kind") or "failed",
                             error=str(res.get("error") or "unknown error"), warnings=warnings, result=res)


def inputs_kept_until(cfg, d: Path, st: dict) -> float | None:
    """When an ended conversion's uploaded input/ goes (epoch s), or None (none kept, or not ended)."""
    if st.get("state") not in ("done", "failed") or not (d / CONVERT_INPUT).exists():
        return None
    fin = _parse_local(st.get("finished")) or time.time()
    return fin + keep_days(cfg) * 86400


def remove_inputs(d: Path) -> bool:
    """Delete a conversion's uploaded raw files (input/). Callers hold the runner lock."""
    if not (d / CONVERT_INPUT).exists():
        return False
    shutil.rmtree(d / CONVERT_INPUT, ignore_errors=True)
    write_conv_status(d, inputs_removed=now_iso())
    return True


def sweep_inputs(cfg, items) -> int:
    """Remove input/ of ended conversions past CJ_RAW_KEEP_DAYS (0: at once). Callers hold the runner lock."""
    n = 0
    for d, _meta, st in items:
        until = inputs_kept_until(cfg, d, st)
        if until is not None and time.time() >= until:
            n += remove_inputs(d)
    return n


def runner_lock(cfg):
    root = raw_root(cfg)
    if not root.is_dir():
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:                                   # raw drive away: old conversions still need a lock
            root = raw_roots(cfg)[-1]
    return PS.locked(root / CONVERT_LOCK)


def kick(cfg) -> int:
    """Record finished (or overdue) conversions, start queued ones (at most MAX_CONVERT_RUNNING at once) and sweep
    old inputs, under the runner lock. -> how many are still queued or running."""
    if not any(r.is_dir() for r in raw_roots(cfg)):
        return 0
    with runner_lock(cfg):
        items = all_conversions(cfg)
        running = 0
        for i, (d, meta, st) in enumerate(items):
            if st.get("state") == "running":
                st = finalize(d, meta, st)
                items[i] = (d, meta, st)
                running += st.get("state") == "running"
        pending = running
        for i, (d, meta, st) in enumerate(items):
            if st.get("state") != "queued":
                continue
            if running < MAX_CONVERT_RUNNING:
                st = start_conversion(cfg, d, meta)
                items[i] = (d, meta, st)
                running += st.get("state") == "running"
                pending += st.get("state") == "running"
            else:
                pending += 1
        sweep_inputs(cfg, items)
    return pending


# -- the admin's actions on one conversion
def stop_conversion(cfg, d: Path) -> str:
    """Admin "Stop": kill a running converter (or drop a queued one) and mark it failed ("stopped")."""
    with runner_lock(cfg):
        st = _read_json(d / PS.STATUS) or {}
        if st.get("state") == "running":
            kill_proc(st)
            write_conv_status(d, state="failed", kind="stopped", pid=None, finished=now_iso(),
                              error="Stopped by the admin.")
            return "stopped"
        if st.get("state") == "queued":
            write_conv_status(d, state="failed", kind="stopped", pid=None, finished=now_iso(),
                              error="Stopped by the admin before it started.")
            return "stopped"
    return "not running"


def can_retry(meta: dict, d: Path, st: dict) -> bool:
    if st.get("state") != "failed":
        return False
    if (meta.get("source") or {}).get("kind") == "upload":
        return (d / CONVERT_INPUT).is_dir()
    return all(Path(p).is_file() for p in meta.get("inputs") or [])


def retry_conversion(cfg, d: Path, meta: dict) -> str:
    """Admin "Retry": a failed conversion whose inputs are still there goes back to the queue (its tiles/ and
    result cleared; the log is kept and appended to)."""
    with runner_lock(cfg):
        st = _read_json(d / PS.STATUS) or {}
        if not can_retry(meta, d, st):
            raise Refusal("This conversion can’t be tried again (it is not stopped, or its raw files were removed: "
                          "upload them again).", 409)
        shutil.rmtree(d / CONVERT_TILES, ignore_errors=True)
        try:
            (d / CONVERT_OUT).unlink()
        except OSError:
            pass
        write_conv_status(d, state="queued", pid=None, pid_created=None, started=None, finished=None, error=None,
                          kind=None, warnings=[], result=None, n_tiles=None, retries=int(st.get("retries") or 0) + 1)
    kick(cfg)
    ensure_watcher(cfg)
    return "queued"


def delete_inputs_now(cfg, d: Path) -> bool:
    """Admin "Delete raw files now" on an ended conversion."""
    with runner_lock(cfg):
        st = _read_json(d / PS.STATUS) or {}
        if st.get("state") in ("queued", "running"):
            raise Refusal("Stop the conversion first.", 409)
        return remove_inputs(d)


_WATCH: dict = {}                     # raw root -> {"dirty": bool}: one watcher thread per root
_WATCH_LOCK = threading.Lock()


def ensure_watcher(cfg):
    """A daemon thread that calls kick() every CONVERT_POLL_S while anything is queued or running, then ends."""
    key = os.path.normcase(str(raw_root(cfg)))
    with _WATCH_LOCK:
        if key in _WATCH:
            _WATCH[key]["dirty"] = True
            return
        _WATCH[key] = {"dirty": True}

    def loop():
        while True:
            with _WATCH_LOCK:
                _WATCH[key]["dirty"] = False
            try:
                pending = kick(cfg)
            except Exception as e:                     # noqa: BLE001 - logged; the watcher keeps going
                print(f"[predict] conversion watcher: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                pending = 1
            if not pending:
                with _WATCH_LOCK:
                    if not _WATCH[key]["dirty"]:
                        _WATCH.pop(key, None)
                        return
                continue
            time.sleep(CONVERT_POLL_S)

    threading.Thread(target=loop, name="cj-predict-convert-watch", daemon=True).start()


_SRC_CACHE: dict = {}                 # (tiles folder, status.json mtime_ns) -> check_folder() of a done conversion
_SRC_LOCK = threading.Lock()


def converted_sources(cfg, user, can_see) -> list:
    """The finished conversions this person may see (owner, admin) as Predict sources: [{key, label, dir, manifest,
    session (from the folder name), sample_column, problem, owner, conversion}]. The folder check is cached per
    status.json mtime (a done conversion's tiles no longer change)."""
    out = []
    for d, meta, st in list_conversions(cfg, user, can_see):
        if st.get("state") != "done":
            continue
        m = RAW_ID_RE.match(d.name)
        tiles = d / CONVERT_TILES
        man = tiles / PSRC.MANIFEST
        try:
            key = (str(tiles), (d / PS.STATUS).stat().st_mtime_ns)
        except OSError:
            continue
        with _SRC_LOCK:
            chk = _SRC_CACHE.get(key)
        if chk is None:
            chk = PSRC.check_folder(tiles)
            with _SRC_LOCK:
                _SRC_CACHE[key] = chk
                while len(_SRC_CACHE) > CACHE_MAX:
                    _SRC_CACHE.pop(next(iter(_SRC_CACHE)))
        out.append({"key": PSRC.source_key(tiles),
                    "label": f"{PSRC.default_label(m.group(1))} · converted ({meta.get('recipe')} recipe, "
                             f"{when(meta.get('created'))})",
                    "dir": tiles, "manifest": man if man.is_file() else None, "session": m.group(1),
                    "sample_column": None, "problem": None if chk["compatible"] else chk["reason"],
                    "owner": meta.get("owner"), "conversion": d.name})
    return out


# ---------------------------------------------------------------- raw session folders (admin; read only)
def raw_sessions(cfg) -> list:
    """Session folders under the scan roots with no model-ready tiles yet (predict_sources' "no model-ready tiles"):
    [{key, folder, root, dir, session, label}]."""
    out = []
    for s in PSRC.discover(cfg)["skipped"]:
        if not str(s.get("reason") or "").startswith("no model-ready tiles"):
            continue
        sid = PSRC.session_of(s["folder"])
        if not sid:
            continue
        d = Path(s["root"]) / s["folder"]
        out.append({"key": hashlib.sha1(os.path.normcase(str(d)).encode("utf-8")).hexdigest()[:10],
                    "folder": s["folder"], "root": Path(s["root"]), "dir": d, "session": sid,
                    "label": PSRC.default_label(sid)})
    out.sort(key=lambda s: (s["session"][:8], s["session"][8:]), reverse=True)
    return out


def raw_montages(s: dict) -> dict:
    """{"montages": [{rel, name, kind, bytes, files, atlas}], "sample_key": Path | None, "truncated": bool} for one
    session folder: .mrc.mdoc (with its .mrc beside it) and .idoc files, RAW_SCAN_DEPTH folders deep (processed,
    gallery and model folders skipped). Reads only; follows no link out of the scan root."""
    rr = PSRC.real(s["root"])
    found, seen = [], {"n": 0, "cut": False}

    def walk(p: Path, depth: int):
        try:
            entries = sorted(os.scandir(p), key=lambda e: e.name.lower())
        except OSError:
            return
        for e in entries:
            seen["n"] += 1
            if seen["n"] > RAW_SCAN_MAX:
                seen["cut"] = True
                return
            try:
                if e.is_dir(follow_symlinks=False):
                    if depth < RAW_SCAN_DEPTH and not RAW_SKIP_DIR.match(e.name) and PSRC.inside(e.path, rr):
                        walk(Path(e.path), depth + 1)
                    continue
                if not e.is_file(follow_symlinks=False):
                    continue
            except OSError:
                continue
            low, path = e.name.lower(), Path(e.path)
            if low.endswith(".mrc.mdoc"):
                mrc = path.with_name(e.name[:-len(".mdoc")])
                if not (mrc.is_file() and PSRC.inside(mrc, rr)):
                    continue
                kind, size = "MRC", mrc.stat().st_size
            elif low.endswith(".idoc"):
                kind, size = "TIFF", None
            else:
                continue
            if not PSRC.inside(path, rr):
                continue
            found.append({"rel": path.relative_to(s["dir"]).as_posix(), "name": e.name, "kind": kind, "bytes": size,
                          "atlas": "atlas" in low})
    walk(s["dir"], 0)
    key = None
    for cand in (s["dir"] / NOTES_SUB / SAMPLE_KEY, s["dir"] / SAMPLE_KEY):
        if cand.is_file() and PSRC.inside(cand, rr):
            key = cand
            break
    return {"montages": found, "sample_key": key, "truncated": seen["cut"]}


def session_convert(cfg, user, s: dict, form: dict) -> Path:
    """The admin's pick of montages in a raw session folder -> a conversion (inputs = the sidecars, read in place)."""
    if user.get("role") != "admin":
        raise Refusal("Picking a raw session is for the admin.", 403)
    fields = raw_fields(cfg, user, form, session=s["session"])
    listing = raw_montages(s)
    by_rel = {m["rel"]: m for m in listing["montages"]}
    picked = [r for r in form.get("montages") or [] if r in by_rel]
    if not picked:
        raise Refusal("Tick at least one montage.")
    if len(picked) != len(set(form.get("montages") or [])):
        raise Refusal("The session’s files changed since this page was opened. Check the montages again.")
    check_conv_cap(cfg, user)
    problems = []                         # a sidecar naming a path (absolute, UNC, ..): never handed to the converter
    for r in picked:
        try:
            problems += sidecar_problems(s["dir"] / r)
        except OSError as e:
            problems.append(f"{r}: can't be read ({type(e).__name__})")
    if problems:
        raise Refusal(f"Not converted ({len(problems)} problem(s)):\n" + "\n".join(problems[:12]))
    inputs = [str(s["dir"] / r) for r in picked]
    nbytes = input_bytes({"inputs": inputs})
    check_disk_for(raw_root(cfg), nbytes)
    check_gpu(fields)
    return new_conversion(cfg, user, fields, source={"kind": "session", "folder": s["folder"], "montages": picked},
                          inputs=inputs, sample_key=listing["sample_key"], nbytes=nbytes)


# ---------------------------------------------------------------- upload sessions (one file or piece per request)
_UP_RES: dict = {}                      # upload id -> Reservation (the bytes it may still write)
_UP_RES_LOCK = threading.Lock()
_UP_ALOCKS: dict = {}                   # upload id -> asyncio.Lock (one request at a time per session)


def upload_dir(cfg, uid) -> Path | None:
    if not isinstance(uid, str) or not UPLOAD_RE.match(uid):
        return None
    for tmp in tmp_dirs(cfg):
        d = tmp / uid
        if (d / UPLOAD_META).is_file():
            return d
    return None


def load_upload(cfg, user, uid) -> tuple:
    """-> (folder, upload.json) when it exists and is this person's own (an admin too: uploads are personal)."""
    d = upload_dir(cfg, uid)
    meta = _read_json(d / UPLOAD_META) if d else None
    if not isinstance(meta, dict) or meta.get("owner") != user.get("id"):
        return None, None
    return d, meta


def upload_reservation(uid, d: Path, meta: dict) -> Reservation:
    """The session's disk hold (made again after a portal restart from what is still to come)."""
    with _UP_RES_LOCK:
        res = _UP_RES.get(uid)
        if res is None:
            res = Reservation(d, max(0, sum(meta["declared"].values()) - sum(upload_received(d, meta).values())))
            _UP_RES[uid] = res
        return res


def drop_upload(d: Path, uid: str):
    with _UP_RES_LOCK:
        res = _UP_RES.pop(uid, None)
    if res is not None:
        res.release()
    _UP_ALOCKS.pop(uid, None)
    shutil.rmtree(d, ignore_errors=True)


def upload_received(d: Path, meta: dict) -> dict:
    out = {}
    done = set(meta.get("done") or [])
    for n, size in meta["declared"].items():
        if n in done:
            out[n] = size
        else:
            p = d / UPLOAD_FILES / (n + PART)
            try:
                out[n] = p.stat().st_size
            except OSError:
                out[n] = 0
    return out


def upload_state(d: Path, meta: dict) -> dict:
    return {"schema": SCHEMA_UPLOAD, "upload": meta["id"], "kind": meta["kind"], "chunk": CHUNK_BYTES,
            "received": upload_received(d, meta), "done": list(meta.get("done") or []),
            "total": sum(meta["declared"].values())}


def is_sidecar(name: str) -> bool:
    return name.lower().endswith((".mdoc", ".idoc", ".csv"))


def raw_pairing(names) -> list:
    """Problems with a raw file list: each .mrc with its .mrc.mdoc, .tif tiles with an .idoc, one sample key."""
    low = {n.lower(): n for n in names}
    problems = []
    for lo, n in low.items():
        if lo.endswith(".mrc") and lo + ".mdoc" not in low:
            problems.append(f"{n}: its {n}.mdoc is missing (the pixel size and tile positions come from it)")
        if lo.endswith(".mrc.mdoc") and lo[:-len(".mdoc")] not in low:
            problems.append(f"{n}: its {n[:-len('.mdoc')]} is missing")
    idocs = [n for lo, n in low.items() if lo.endswith(".idoc")]
    tifs = [n for lo, n in low.items() if lo.endswith((".tif", ".tiff"))]
    if tifs and not idocs:
        problems.append("TIFF tiles need the montage’s .idoc file")
    if idocs and not tifs:
        problems.append("an .idoc needs its .tif tiles")
    if sum(1 for lo in low if lo.endswith(".csv")) > 1:
        problems.append("at most one sample key (.csv)")
    if not any(lo.endswith((".mrc.mdoc", ".idoc")) for lo in low) and not problems:
        problems.append("no montage: add an .mrc with its .mrc.mdoc, or an .idoc with its .tif tiles")
    return problems


def declared_files(kind: str, files) -> dict:
    """The file list a page declares at the start -> {name: size}. Raises Refusal on a cap (413) or a bad list."""
    if not isinstance(files, list) or not files:
        raise Refusal("Choose at least one image." if kind == "tiles" else "Choose the raw files.")
    cap_n = MAX_FILES if kind == "tiles" else RAW_MAX_FILES
    if len(files) > cap_n:
        raise Refusal(f"More than {cap_n} files: " + (f"an analysis takes at most {cap_n}." if kind == "tiles" else
                                                       f"convert at most {cap_n} files at once.") +
                      " Nothing was kept.", 413)
    out, seen = {}, set()
    for it in files:
        name = str((it or {}).get("name") or "") if isinstance(it, dict) else ""
        size = (it or {}).get("size") if isinstance(it, dict) else None
        safe = safe_name(name)
        if safe is None or safe != name:
            raise Refusal(f"File name not accepted: {name[:80]!r}. Rename it (letters, digits, - _ . and spaces) and "
                          f"upload again.")
        if safe.lower() in seen:
            raise Refusal(f"Two files are named {safe!r}; each name must be unique.")
        seen.add(safe.lower())
        if safe.lower().endswith(PART):                  # never the name of another file's partial copy
            raise Refusal(f"File name not accepted: {name[:80]!r} (it ends in {PART}). Rename it and upload again.")
        if kind == "tiles" and not safe.lower().endswith(IMAGE_EXT):
            raise Refusal(f"{safe}: not a PNG or JPEG file name. Upload processed .png / .jpg tiles.")
        if kind == "raw" and not safe.lower().endswith(RAW_EXT):
            raise Refusal(f"{safe}: not a raw montage file. Upload .mrc files with their .mrc.mdoc, or an .idoc with "
                          f"its .tif tiles (and, if you have it, the session’s sample key .csv).")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise Refusal(f"{safe} is empty.")
        cap = MAX_FILE_BYTES if kind == "tiles" else RAW_SIDE_MAX_BYTES if is_sidecar(safe) else RAW_MAX_FILE_BYTES
        if size > cap:
            raise Refusal(f"{safe} is over {cap // 2 ** 20} MB. Nothing was kept.", 413)
        out[safe] = size
    if kind == "raw":
        if sum(out.values()) > RAW_MAX_TOTAL_BYTES:
            raise Refusal(f"Over {RAW_MAX_TOTAL_BYTES // 2 ** 30} GB in one conversion. Convert fewer montages at "
                          f"once.", 413)
        problems = raw_pairing(out)
        if problems:
            raise Refusal(f"Please fix ({len(problems)}):\n" + "\n".join(problems))
    return out


def clean_fields(raw) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict) or len(raw) > MAX_FIELDS:
        raise Refusal("Too many form fields.", 413)
    out = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, (str, int, float)) or isinstance(v, bool):
            continue
        v = str(v)
        if len(k) > 64 or len(v.encode("utf-8")) > MAX_FIELD_BYTES:
            raise Refusal("A form field is too long.", 413)
        out[k] = v
    return out


def check_upload_fields(cfg, user, kind, fields, has_files=True) -> dict:
    """Tiles: the label / nm / tick / model checks of the plain form (+ the basic user's limit). Raw: raw_fields."""
    if kind == "tiles":
        upload_fields(cfg, user, fields, has_files)
        check_active(cfg, user)
        return {}
    return raw_fields(cfg, user, fields)


def upload_start(cfg, user, body) -> dict:
    """Open (or resume) an upload session. Every check that can be made before a byte is sent is made here."""
    if not isinstance(body, dict):
        raise Refusal("Bad request.", 400)
    kind = body.get("kind")
    if kind not in ("tiles", "raw"):
        raise Refusal("Unknown upload.", 400)
    if kind == "raw" and user.get("role") not in RAW_ROLES:
        raise Refusal("Converting raw images is for advanced users and the admin.", 403)
    sweep_scratch(cfg)
    fields = clean_fields(body.get("fields"))
    if kind == "tiles" and fields.get("confirm") != "yes":       # the plain form's first refusal, before the list
        raise Refusal("Tick “I confirm the pixel size”: nothing was started.")
    decl = declared_files(kind, body.get("files"))
    raw = check_upload_fields(cfg, user, kind, fields)
    if raw:
        check_conv_cap(cfg, user)
        check_gpu(raw)
    resume = body.get("resume")
    if isinstance(resume, str):
        d, meta = load_upload(cfg, user, resume)
        if meta and meta.get("kind") == kind and meta.get("declared") == decl:
            meta["fields"] = fields
            PS.write_json_atomic(d / UPLOAD_META, meta)
            return upload_state(d, meta)
    # one open session per person and kind: an earlier one (other files) is given up
    mine = []
    for tmp in tmp_dirs(cfg):
        try:
            mine += [p for p in tmp.iterdir() if UPLOAD_RE.match(p.name)]
        except OSError:
            continue
    for p in mine:
        m = _read_json(p / UPLOAD_META)
        if isinstance(m, dict) and m.get("owner") == user.get("id") and m.get("kind") == kind:
            drop_upload(p, p.name)
    d = scratch(cfg, "us", raw=(kind == "raw"))
    uid = d.name
    res = Reservation(d, sum(decl.values()))
    try:
        res.check()
        (d / UPLOAD_FILES).mkdir()
        meta = {"schema": SCHEMA_UPLOAD, "id": uid, "owner": user["id"], "kind": kind, "fields": fields,
                "declared": decl, "done": [], "created": now_iso()}
        PS.write_json_atomic(d / UPLOAD_META, meta)
    except BaseException:
        res.release()
        shutil.rmtree(d, ignore_errors=True)
        raise
    with _UP_RES_LOCK:
        _UP_RES[uid] = res
    return upload_state(d, meta)


MRC_MAGIC = b"MAP"
TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


SIDECAR_EXT = (".idoc", ".mdoc")


def sidecar_images(path: Path) -> list:
    """Every [Image = <name>] of a SerialEM .idoc / .mrc.mdoc, read exactly as cryoem_core.metadata.parse_metadata
    reads it (a line starting with "[" and holding "=": the section type before the first "=", the name between the
    first and second "=", "]" and spaces trimmed). Latin-1, so every byte is kept and no path character is lost."""
    out = []
    with open(path, "rb") as f:
        text = f.read().decode("latin-1")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and "=" in line:
            if line.split("=")[0].strip()[1:].strip() == "Image":
                out.append(line.split("=")[1].strip().rstrip("]").strip())
    return out


def image_ref_problem(name: str) -> str | None:
    """A tile name in a sidecar must be a bare file name: no \\ / : (no drive, folder or UNC share), not . or ..,
    not empty, no control characters."""
    if not name or name in (".", "..") or any(c in name for c in "\\/:") or any(ord(c) < 32 for c in name):
        return f"names a tile {name[:80]!r} that is not a plain file name (no paths allowed)"
    return None


def sidecar_problems(path: Path, declared=None) -> list:
    """Problems with the tiles a sidecar names: each must be a bare file name and, for an upload (declared = the
    uploaded names), one of the uploaded files."""
    out = []
    for name in sidecar_images(path):
        why = image_ref_problem(name)
        if why is None and declared is not None and name not in declared:
            why = f"names a tile {name[:80]!r} that was not uploaded with it"
        if why:
            out.append(f"{path.name}: {why}")
    return list(dict.fromkeys(out))


def raw_problem(path: Path) -> str | None:
    """A raw upload file checked by its content: an MRC stack, a SerialEM sidecar, a TIFF tile, a sample key."""
    n = path.name.lower()
    with open(path, "rb") as f:
        head = f.read(65536)
    if n.endswith(".mrc"):
        if len(head) < 1024 or head[208:211] != MRC_MAGIC:
            return f"{path.name}: not an MRC file (by content)"
        nx, ny, nz = struct.unpack("<3i", head[:12])
        if min(nx, ny, nz) < 1:
            return f"{path.name}: an MRC file with no image in it"
    elif n.endswith((".mdoc", ".idoc")):
        if b"\x00" in head or b"PixelSpacing" not in head or not (b"[ZValue" in head or b"[Image" in head):
            return f"{path.name}: not a SerialEM {n.rsplit('.', 1)[-1]} file (by content: no PixelSpacing and tiles)"
        bad = sidecar_problems(path)                     # a path in [Image = ...]: refused at once
        if bad:
            return bad[0]
    elif n.endswith((".tif", ".tiff")):
        if head[:4] not in TIFF_MAGIC:
            return f"{path.name}: not a TIFF (by content)"
    elif n.endswith(".csv"):
        try:
            first = head.decode("utf-8-sig").splitlines()[0]
        except (UnicodeDecodeError, IndexError):
            return f"{path.name}: not a sample key (a text .csv with a #,Label header)"
        if b"\x00" in head or "label" not in first.lower():
            return f"{path.name}: not a sample key (a text .csv with a #,Label header)"
    return None


def file_problem(kind: str, path: Path) -> str | None:
    if kind == "tiles":
        try:
            PS.inspect_image(path)
        except (ValueError, OSError) as e:
            return f"{path.name}: {e}"
        return None
    return raw_problem(path)


class Resync(Exception):
    """The piece does not start where the file stands: answer 409 with the session (the page resumes there)."""


async def receive_piece(request: Request, d: Path, uid: str) -> dict:
    """One piece of one file: ?name=&offset=, body = the bytes. Appends to <name>.part; the complete file is renamed
    and content-checked. Raises Refusal (the caller drops the session only when drop is set or the disk is full:
    a size or content violation; a bad request leaves it as it is), Resync, ClientDisconnect."""
    meta = await run_in_threadpool(_read_json, d / UPLOAD_META)
    if not isinstance(meta, dict):
        raise Refusal("This upload is gone. Start again.", 404)
    name = request.query_params.get("name") or ""
    try:
        offset = int(request.query_params.get("offset") or "")
    except ValueError:
        raise Refusal("Bad request (offset).", 400)
    decl = meta["declared"]
    if name not in decl:
        raise Refusal(f"{name[:80]!r} is not one of the files of this upload.", 400)
    if name in (meta.get("done") or []):
        return upload_state(d, meta)                     # already complete: answering again is harmless
    size = decl[name]
    part = d / UPLOAD_FILES / (name + PART)
    have = part.stat().st_size if part.is_file() else 0
    if offset != have:
        raise Resync()
    try:
        length = int(request.headers.get("content-length") or "")
    except ValueError:
        raise Refusal("Each piece must say its length (Content-Length).", 411)
    if length < 1 or length > CHUNK_BYTES:
        raise Refusal(f"One request carries at most {CHUNK_BYTES // 2 ** 20} MB. Nothing was kept.", 413, drop=True)
    if offset + length > size:
        raise Refusal(f"{name}: more bytes than its size. Nothing was kept.", 400, drop=True)
    res = upload_reservation(uid, d, meta)
    await run_in_threadpool(res.check)
    fh = await run_in_threadpool(open, part, "ab")
    got, buf, since = 0, bytearray(), 0
    try:
        async for piece in request.stream():
            if not piece:
                continue
            got += len(piece)
            if got > length:
                raise Refusal("A piece was longer than it said. Nothing was kept.", 400, drop=True)
            buf += piece
            if len(buf) >= WRITE_BATCH:
                await run_in_threadpool(fh.write, bytes(buf))
                res.written(len(buf))
                since += len(buf)
                buf.clear()
                if since >= DISK_CHECK_BYTES:
                    await run_in_threadpool(res.check)
                    since = 0
        if buf:
            await run_in_threadpool(fh.write, bytes(buf))
            res.written(len(buf))
    finally:
        await run_in_threadpool(fh.close)
    if got != length:
        raise Resync()                                   # cut short: the page asks again from what arrived
    try:
        os.utime(d / UPLOAD_META)                        # last activity (the sweep's clock)
    except OSError:
        pass
    if offset + length == size:
        final = d / UPLOAD_FILES / name
        await run_in_threadpool(os.replace, part, final)
        problem = await run_in_threadpool(file_problem, meta["kind"], final)
        if problem:
            raise Refusal(f"{problem}. Nothing was kept.", 400, drop=True)
        meta["done"] = list(meta.get("done") or []) + [name]
        await run_in_threadpool(PS.write_json_atomic, d / UPLOAD_META, meta)
    return upload_state(d, meta)


def upload_create(cfg, user, d: Path, meta: dict) -> str:
    """Every declared file is there -> the prediction set (tiles) or the conversion (raw). -> where to go next.
    Refusal 409 (keep the session) when files are missing."""
    missing = [n for n in meta["declared"] if n not in set(meta.get("done") or [])]
    if missing:
        raise Refusal(f"{len(missing)} file(s) not sent yet ({', '.join(missing[:3])}"
                      f"{', …' if len(missing) > 3 else ''}). Press the button again to send them.", 409)
    fields = meta.get("fields") or {}
    if meta["kind"] == "tiles":
        label, nm = upload_fields(cfg, user, fields, True)
        check_active(cfg, user)
        ps_dir = create_and_start(cfg, d / UPLOAD_FILES, user, fields["model"], sample_label=label, nm=nm)
        return f"/predict/{ps_dir.name}/"
    raw = raw_fields(cfg, user, fields)
    check_conv_cap(cfg, user)
    # every sidecar's [Image = ...]: a bare name, and one of the uploaded files (else cryoem_core would open whatever
    # path it names on the server: an absolute path, a UNC share, ../)
    problems = []
    for n in meta["declared"]:
        if n.lower().endswith(SIDECAR_EXT):
            problems += sidecar_problems(d / UPLOAD_FILES / n, set(meta["declared"]))
    if problems:
        raise Refusal(f"Not converted ({len(problems)} problem(s)). Nothing was kept:\n" + "\n".join(problems[:12]))
    check_disk_for(raw_root(cfg), sum(meta["declared"].values()))
    check_gpu(raw)
    conv = new_conversion(cfg, user, raw, upload_files=d / UPLOAD_FILES,
                          source={"kind": "upload", "files": list(meta["declared"])},
                          nbytes=sum(meta["declared"].values()))
    return f"/predict/convert/{conv.name}/"


# ---------------------------------------------------------------- raw pages (every text through esc)
CONV_TONE = {"queued": "wait", "running": "run", "done": "ok", "failed": "warn"}


def conv_status_view(meta: dict, st: dict, admin: bool, paths=()) -> dict:
    state = st.get("state")
    out = {"state": state, "tone": CONV_TONE.get(state, "wait"), "label": "Unknown", "detail": None, "tech": None,
           "final": state in ("done", "failed")}
    if state == "queued":
        out.update(label="Waiting to start", detail="Another conversion is running; this one starts after it.")
    elif state == "running":
        out.update(label="Converting", detail="About half a minute per montage on the CPU. This page updates by "
                                              "itself; you can close it.")
    elif state == "done":
        out.update(label="Done", detail=f"{int(st.get('n_tiles') or 0)} tiles made.")
    elif state == "failed":
        msg = _plain(str(st.get("error") or ""), paths)
        out.update(label="Stopped", detail=(f"Not converted: {msg}" if st.get("kind") == "refused" else
                                            f"Stopped: {msg}" if st.get("kind") in ("stopped", "timeout") else
                                            "The conversion failed. Ask the admin (they can see the details)."))
    if admin:
        bits = [f"{k} {st.get(k)}" for k in ("pid", "kind", "error", "started", "finished") if st.get(k)]
        out["tech"] = "; ".join(bits + [f"updated {st.get('updated')}"])
    return out


def conv_paths(d: Path, meta: dict) -> list:
    """Folders whose spelling is left out of what people read (they never typed them)."""
    out = [d / CONVERT_INPUT, d / CONVERT_TILES, d]
    for p in meta.get("inputs") or []:
        out.append(Path(p).parent)
    if meta.get("sample_key"):
        out.append(Path(meta["sample_key"]).parent)
    return list(dict.fromkeys(out))


def conv_fragments(cfg, user, can_see, d: Path, meta: dict, st: dict) -> dict:
    admin = user.get("role") == "admin"
    paths = conv_paths(d, meta)
    v = conv_status_view(meta, st, admin, paths)
    status = (f'<div class="cj-card cj-predict-status"><div class="cj-predict-top"><b>Status</b>'
              f'{chip(v["label"], v["tone"])}</div>'
              + (f'<p class="cj-predict-detail">{esc(v["detail"])}</p>' if v["detail"] else "") + tech(v["tech"])
              + '</div>')
    parts = []
    warns = [_plain(w, paths) for w in st.get("warnings") or []]
    if st.get("state") == "done":
        src = next((s for s in converted_sources(cfg, user, can_see) if s.get("conversion") == d.name), None)
        if src is not None and not src["problem"]:
            parts.append(f'<div class="cj-card"><a class="cj-predict-btn cj-predict-primary" '
                         f'href="/predict/new/session/{esc(src["key"])}/">Analyse these tiles</a>'
                         f'<p class="cj-mut">Pick samples and confirm the pixel size, as for a processed session. '
                         f'Also listed under “New analysis”.</p></div>')
        elif src is not None:
            parts.append(msg_box(f"These tiles can’t be analysed: {src['problem']}"))
        res = st.get("result") or {}
        nm = ", ".join(f"{float(x):g}" for x in res.get("nm_per_px") or [])
        parts.append(f'<div class="cj-card"><dl class="cj-predict-kv">'
                     f'<div><dt>Tiles</dt><dd>{int(st.get("n_tiles") or 0)}</dd></div>'
                     f'<div><dt>nm per pixel</dt><dd>{esc(nm or "?")}</dd></div>'
                     f'<div><dt>Montages</dt><dd>{esc(", ".join(res.get("montages") or []))}</dd></div>'
                     f'<div><dt>Computed on</dt><dd>{esc(res.get("compute") or "?")}</dd></div></dl></div>')
    if warns:
        parts.append('<h2 class="cj-sec">Warnings</h2><div class="cj-card"><ul class="cj-predict-warns">'
                     + "".join(f"<li>{esc(w)}</li>" for w in warns) + '</ul></div>')
    until = inputs_kept_until(cfg, d, st)
    if until is not None:
        parts.append(f'<p class="cj-mut">The uploaded raw files are kept until '
                     f'{esc(when(datetime.fromtimestamp(until).isoformat(timespec="seconds")))}, then removed.</p>')
    if admin:
        acts = []
        base = f'/predict/convert/{esc(d.name)}'
        if st.get("state") in ("queued", "running"):
            acts.append(f'<form method="post" action="{base}/stop" class="cj-predict-inline">'
                        f'<button class="cj-predict-btn" type="submit">Stop</button></form>')
        if can_retry(meta, d, st):
            acts.append(f'<form method="post" action="{base}/retry" class="cj-predict-inline">'
                        f'<button class="cj-predict-btn" type="submit">Retry</button></form>')
        if until is not None:
            acts.append(f'<form method="post" action="{base}/inputs" class="cj-predict-inline">'
                        f'<button class="cj-predict-btn" type="submit">Delete raw files now</button></form>')
        if acts:
            parts.append(f'<div class="cj-card cj-predict-adminacts"><b>Admin</b>{"".join(acts)}</div>')
    if admin and st.get("state") in ("failed", "running", "done"):
        tail = log_tail(d)
        if tail:
            parts.append(f'<details class="cj-card"><summary>Converter log (admin)</summary>'
                         f'<pre class="cj-predict-log">{esc(tail)}</pre></details>')
    return {"status": status, "result": "".join(parts), "live": not v["final"], "state": st.get("state")}


def page_convert(cfg, user, can_see, d: Path, meta: dict, st: dict) -> str:
    admin = user.get("role") == "admin"
    fr = conv_fragments(cfg, user, can_see, d, meta, st)
    src = meta.get("source") or {}
    what = ("Uploaded files" if src.get("kind") == "upload" else
            f"Raw session folder {src.get('folder') or '?'}")
    default = (msg_box(f"Converted with the default recipe ({CR.DEFAULT_RECIPE}, the lab’s template), not this "
                       f"session’s own: contrast and denoise may differ from the lab’s delivered tiles.")
               if meta.get("recipe") == "default" else "")
    owner = f'<div><dt>Owner</dt><dd>{esc(meta.get("owner"))}</dd></div>' if admin else ""
    return wrap(
        f'<div class="cj-predict" id="cj-predict-conv" data-cid="{esc(d.name)}" data-live="{int(fr["live"])}" '
        f'data-every="{REFRESH_S}"><p class="cj-mut"><a href="/predict/">← My analyses</a></p>'
        f'<h1>Raw conversion: {esc(PSRC.default_label(meta.get("session") or "19700101"))}</h1>'
        + (f'<p class="cj-mut cj-wrap">{esc(d.name)}</p>' if user.get("role") != "basic" else "") +
        f'<p id="cj-predict-err" class="cj-mut" role="status" hidden></p>{default}'
        f'<section id="cj-predict-cstatus">{fr["status"]}</section>'
        f'<section id="cj-predict-cresult">{fr["result"]}</section>'
        f'<h2 class="cj-sec">Details</h2><div class="cj-card"><dl class="cj-predict-kv">'
        f'<div><dt>Session</dt><dd>{esc(meta.get("session"))}</dd></div>'
        f'<div><dt>Recipe</dt><dd>{esc(meta.get("recipe"))}</dd></div>'
        f'<div><dt>Compute</dt><dd>{"GPU (asked)" if meta.get("gpu") else "CPU"}</dd></div>'
        f'<div><dt>Images from</dt><dd>{esc(what)}</dd></div>'
        f'<div><dt>Started</dt><dd>{esc(when(meta.get("created")))}</dd></div>{owner}</dl>'
        + (tech(f"{d} · inputs {json.dumps(meta.get('inputs'))} · sample key {meta.get('sample_key')}")
           if admin else "") + '</div></div>')


def conversions_html(convs, admin) -> str:
    if not convs:
        return '<p class="cj-mut">No raw conversions yet. Start one from “New analysis”.</p>'
    li = []
    for d, meta, st in convs:
        v = conv_status_view(meta, st, False)
        n = f' · {int(st.get("n_tiles") or 0)} tiles' if st.get("state") == "done" else ""
        who = f' · {esc(meta.get("owner"))}' if admin else ""
        li.append(f'<li class="cj-card cj-predict-item"><div class="cj-predict-top">'
                  f'<a class="cj-predict-ttl" href="/predict/convert/{esc(d.name)}/">'
                  f'{esc(PSRC.default_label(RAW_ID_RE.match(d.name).group(1)))} · {esc(meta.get("recipe"))} recipe'
                  f'</a>{chip(v["label"], v["tone"])}</div>'
                  f'<p class="cj-mut">{esc(when(meta.get("created")))}{n}{who}</p></li>')
    return f'<ul class="cj-predict-items">{"".join(li)}</ul>'


def not_set_up(cfg, user) -> str:
    _, why = processing_repo(cfg)
    return msg_box("Raw conversion is not set up on this server.",
                   extra=tech(why) if user.get("role") == "admin" else "<p>Ask the admin.</p>")


def page_raw_upload(cfg, user, error=None) -> str:
    repo, _ = processing_repo(cfg)
    head = '<div class="cj-predict"><p class="cj-mut"><a href="/predict/new/">← Back</a></p><h1>Convert raw images</h1>'
    if repo is None:
        return wrap(head + not_set_up(cfg, user) + '</div>')
    err = msg_box("", extra=f'<div>{problem_list(error)}</div>') if error else ""
    return wrap(
        head + err +
        '<div class="cj-card cj-predict-what"><h2>What this does</h2><ul>'
        '<li>Your raw montages are made into model-ready tiles by the lab’s own processing (the same code that made '
        'the delivered “3d” tile folders), then you analyse them as a session.</li>'
        '<li>Upload whole montages: each <b>.mrc</b> with its <b>.mrc.mdoc</b>, or an <b>.idoc</b> with all its '
        '<b>.tif</b> tiles. Add the session’s <b>sample_key.csv</b> if you have it (else samples are “Grid n”).</li>'
        '<li>The pixel size comes from the .mdoc / .idoc; it is never guessed.</li></ul></div>'
        f'<form method="post" action="/predict/new/raw/" enctype="multipart/form-data" class="cj-predict-form" '
        f'id="cj-predict-raw" data-cj-upload="raw" data-max-files="{RAW_MAX_FILES}" '
        f'data-max-bytes="{RAW_MAX_FILE_BYTES}">'
        f'<div class="cj-predict-msg cj-predict-t-warn" data-cj-upload-msg role="alert" hidden></div>'
        f'<noscript><p class="cj-predict-msg">Raw files are sent one at a time, which needs JavaScript. Turn it on to '
        f'upload here.</p></noscript>'
        f'<fieldset class="cj-card cj-predict-fs"><legend>Session and recipe</legend>'
        f'<label class="cj-predict-field"><span>Session id (the imaging date, YYYYMMDD; YYYYMMDD-n for a later '
        f'session that day)</span><input name="session" required maxlength="12" '
        f'pattern="[0-9]{{8}}(-[0-9]+)?" inputmode="numeric" placeholder="20261027"></label>'
        f'{recipe_select(repo)}{gpu_box()}</fieldset>'
        f'<fieldset class="cj-card cj-predict-fs"><legend>Raw files</legend>'
        f'<label class="cj-predict-field"><span>.mrc + .mrc.mdoc, or .idoc + .tif tiles (+ sample_key.csv)</span>'
        f'<input type="file" name="files" multiple required accept=".mrc,.mdoc,.idoc,.tif,.tiff,.csv"></label>'
        f'<p class="cj-mut" data-cj-upload-info role="status"></p></fieldset>'
        f'<p><button class="cj-predict-btn cj-predict-primary" type="submit" data-busy="Uploading… keep this page '
        f'open">Upload and convert</button></p></form></div>')


def page_raw_sessions(cfg, user) -> str:
    repo, _ = processing_repo(cfg)
    head = ('<div class="cj-predict"><p class="cj-mut"><a href="/predict/new/">← Back</a></p>'
            '<h1>Pick a raw session</h1>')
    if repo is None:
        return wrap(head + not_set_up(cfg, user) + '</div>')
    cards = [f'<a class="cj-card cj-predict-src" href="/predict/new/raw/sessions/{esc(s["key"])}/">'
             f'<b>{esc(s["label"])}</b><span class="cj-mut">Session {esc(s["session"])} · folder '
             f'{esc(s["folder"])}</span>{tech(str(s["dir"]))}</a>' for s in raw_sessions(cfg)]
    body = "".join(cards) or ('<p class="cj-mut">Every session folder under the scan roots already has processed '
                              'tiles (or none was found).</p>')
    return wrap(head + '<p class="cj-mut">Session folders under the scan roots that have no processed tiles yet. The '
                       'folder is only read; the tiles are written under the Predict folder.</p>'
                f'<div class="cj-predict-srcs">{body}</div></div>')


def human_bytes(n) -> str:
    if not n:
        return ""
    return f"{n / 2 ** 30:.1f} GB" if n >= 2 ** 30 else f"{n / 2 ** 20:.0f} MB"


def page_raw_session(cfg, user, s: dict, listing: dict, form=None, error=None) -> str:
    repo, _ = processing_repo(cfg)
    head = (f'<div class="cj-predict"><p class="cj-mut"><a href="/predict/new/raw/sessions/">← Back</a></p>'
            f'<h1>{esc(s["label"])}</h1><p class="cj-mut">Session {esc(s["session"])} · folder {esc(s["folder"])}'
            f'</p>{tech(str(s["dir"]))}')
    if repo is None:
        return wrap(head + not_set_up(cfg, user) + '</div>')
    form = form or {}
    picked = set(form.get("montages") or [m["rel"] for m in listing["montages"] if not m["atlas"]])
    def row(m):
        size = f' · {esc(human_bytes(m["bytes"]))}' if m["bytes"] else ""
        atlas = ' <span class="cj-predict-warn">atlas: usually not wanted</span>' if m["atlas"] else ""
        return (f'<li><label><input type="checkbox" name="montages" value="{esc(m["rel"])}"'
                f'{" checked" if m["rel"] in picked else ""}> <span class="cj-wrap">{esc(m["rel"])}</span></label> '
                f'<span class="cj-predict-tag">{esc(m["kind"])}{size}</span>{atlas}</li>')
    rows = "".join(row(m) for m in listing["montages"])
    err = msg_box("", extra=f'<div>{problem_list(error)}</div>') if error else ""
    if not rows:
        return wrap(head + err + '<p class="cj-mut">No montages (.mrc with its .mrc.mdoc, or .idoc) found in this '
                                 'folder.</p></div>')
    key = (f'Sample key: {esc(listing["sample_key"].relative_to(s["dir"]).as_posix())} (sample names come from it).'
           if listing["sample_key"] else 'No sample_key.csv found: samples will be named “Grid n”.')
    cut = '<p class="cj-predict-warn">The folder is very big: only part of it was looked at.</p>' \
        if listing["truncated"] else ""
    return wrap(
        head + err +
        f'<form method="post" class="cj-predict-form" id="cj-predict-rawsession">'
        f'<fieldset class="cj-card cj-predict-fs"><legend>Montages ({len(listing["montages"])})</legend>{cut}'
        f'<ul class="cj-predict-mlist">{rows}</ul><p class="cj-mut">{key}</p></fieldset>'
        f'<fieldset class="cj-card cj-predict-fs"><legend>Recipe</legend>'
        f'{recipe_select(repo, form.get("recipe"), s["session"])}{gpu_box(form.get("gpu") == "yes")}</fieldset>'
        f'<p><button class="cj-predict-btn cj-predict-primary" type="submit" data-busy="Starting…">Convert</button>'
        f'</p></form></div>')


# ---------------------------------------------------------------- the page's script (served as /predict/predict.js)
SCRIPT = HERE.parent / "static" / "predict.js"      # served as /predict/predict.js (app.py serves only /cj/*)
JS = SCRIPT.read_text(encoding="utf-8")


# ---------------------------------------------------------------- routes
def sweep_scratch(cfg, max_age_s=SCRATCH_MAX_AGE_S) -> int:
    """Remove <root>/.portal/* left by a portal that stopped mid-upload, and upload sessions never finished (no
    "create"), older than max_age_s (a session: since its last request; its disk hold goes too). -> how many."""
    def age(p: Path) -> float:
        t = p.stat().st_mtime
        try:
            t = max(t, (p / UPLOAD_META).stat().st_mtime)
        except OSError:
            pass
        return time.time() - t
    old = []
    for tmp in tmp_dirs(cfg):                        # the predict root's .portal and the raw root's
        try:
            old += [p for p in tmp.iterdir() if p.is_dir() and p.name.startswith(("up_", "sel_", "us_"))
                    and age(p) > max_age_s]
        except Exception:                            # no root configured or not there yet: nothing to sweep
            continue
    for p in old:
        if UPLOAD_RE.match(p.name):
            drop_upload(p, p.name)
        else:
            shutil.rmtree(p, ignore_errors=True)
    return len(old)


def os_error_text(e: OSError) -> str:
    too_long = getattr(e, "winerror", None) == 206 or getattr(e, "errno", None) == errno.ENAMETOOLONG or \
        "too long" in str(e).lower()
    if too_long:
        return "A file name is too long for the server. Shorten the file names and upload again."
    return "The server could not store the upload. Try again, or ask the admin."


def unreadable_page(src: dict, e: Refusal, user) -> str:
    extra = tech(e.tech) if user.get("role") == "admin" else ""
    return wrap(f'<div class="cj-predict"><p class="cj-mut"><a href="/predict/new/">← Back</a></p>'
                f'<h1>{esc(src["label"])}</h1>{msg_box(e.msg, extra=extra)}</div>')


def routes(ctx) -> list:
    cfg = ctx.cfg
    sweep_scratch(cfg)                               # at start (routes() runs once, before the portal serves)
    try:                                             # a conversion left queued / running by a portal that stopped;
        if kick(cfg):                                # old raw inputs swept
            ensure_watcher(cfg)
    except Exception:                                # no root configured or not there yet
        pass

    def not_found(request):
        r = ctx.page(request, "Not found", wrap('<div class="cj-predict"><h1>Not found</h1><p>There is no analysis '
                                                'here that you can open.</p><p><a href="/predict/">My analyses</a>'
                                                '</p></div>'), active="predict")
        r.status_code = 404
        return r

    def refused_page(request, title, body, status):
        r = ctx.page(request, title, body, active="predict")
        r.status_code = status
        return r

    def forbidden(request, text):
        return refused_page(request, "Not allowed", wrap(f'<div class="cj-predict"><h1>Not allowed</h1>'
                                                         f'<p>{esc(text)}</p><p><a href="/predict/new/">Back</a></p>'
                                                         f'</div>'), 403)

    def find_source(key, user):
        srcs, _ = sources_safe(cfg, user, ctx.can_see)
        return next((s for s in srcs if s["key"] == key and not s["problem"]), None)

    def admin_text(e: Refusal, user) -> str:
        return e.msg + (f"\n(admin) {e.tech}" if e.tech and user.get("role") == "admin" else "")

    def refusal_json(e: Refusal, user, dropped: bool, extra=None):
        doc = {"error": admin_text(e, user), "dropped": dropped}
        doc.update(extra or {})
        return ctx.json(doc, e.status)

    # ---- pages (sync: Starlette runs them in a worker thread)
    def index(request: Request):
        try:
            if kick(cfg):                            # records ends, sweeps old raw inputs (cheap: a few small files)
                ensure_watcher(cfg)
        except Exception as e:                       # noqa: BLE001 - the list still shows
            print(f"[predict] kick: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return ctx.page(request, "Predict", page_mine(cfg, ctx.user(request), ctx.can_see), active="predict")

    def new(request: Request):
        return ctx.page(request, "New analysis", page_new(cfg, ctx.user(request), ctx.can_see), active="predict")

    def session_form(request: Request):
        user = ctx.user(request)
        src = find_source(request.path_params["key"], user)
        if src is None:
            return not_found(request)
        try:
            t = read_tiles(src)
        except Refusal as e:
            return refused_page(request, src["label"], unreadable_page(src, e, user), e.status)
        return ctx.page(request, src["label"], page_session_form(cfg, user, src, t), active="predict")

    async def session_post(request: Request):
        user = ctx.user(request)
        src = await run_in_threadpool(find_source, request.path_params["key"], user)
        if src is None:
            return await run_in_threadpool(not_found, request)
        fd = await request.form(max_files=0, max_fields=1000)
        form = {k: v for k, v in fd.items() if isinstance(v, str)}
        form["samples"] = [v for v in fd.getlist("samples") if isinstance(v, str)]
        try:
            ps_dir = await run_in_threadpool(session_submit, cfg, user, src, form)
        except Refusal as e:
            msg, status = admin_text(e, user), e.status

            def again():
                try:
                    t = read_tiles(src)
                except Refusal as e2:
                    return refused_page(request, src["label"], unreadable_page(src, e2, user), e2.status)
                return refused_page(request, src["label"], page_session_form(cfg, user, src, t, form, msg), status)
            return await run_in_threadpool(again)
        return RedirectResponse(f"/predict/{ps_dir.name}/", status_code=303)

    def upload_form(request: Request):
        return ctx.page(request, "Upload images", page_upload_form(cfg, ctx.user(request)), active="predict")

    async def upload_post(request: Request):
        """The no-JavaScript form: one multipart request, at most NOJS_MAX_BYTES."""
        user = ctx.user(request)
        dest = await run_in_threadpool(scratch, cfg, "up")
        sink = None
        try:
            try:
                await run_in_threadpool(check_active, cfg, user)          # before a byte is stored
                sink = await receive_upload(request, dest, lambda fields: upload_fields(cfg, user, fields, True))
                ps_dir = await run_in_threadpool(upload_submit, cfg, user, dest, sink)
            except OSError as e:                     # disk full, a locked file, a path too long: nothing is kept
                raise Refusal(os_error_text(e), 500, tech=f"{type(e).__name__}: {e}")
        except Refusal as e:
            msg, status = admin_text(e, user), e.status
            form = dict(sink.fields) if sink else {}

            def again():
                return refused_page(request, "Upload images", page_upload_form(cfg, user, form, msg), status)
            return await run_in_threadpool(again)
        finally:
            await run_in_threadpool(shutil.rmtree, dest, True)
        return RedirectResponse(f"/predict/{ps_dir.name}/", status_code=303)

    def raw_form(request: Request):
        user = ctx.user(request)
        if user.get("role") not in RAW_ROLES:
            return forbidden(request, "Converting raw images is for advanced users and the admin.")
        return ctx.page(request, "Convert raw images", page_raw_upload(cfg, user), active="predict")

    def raw_post(request: Request):
        """The raw form without JavaScript: refused (raw files go one per request, which needs the page's script)."""
        user = ctx.user(request)
        if user.get("role") not in RAW_ROLES:
            return forbidden(request, "Converting raw images is for advanced users and the admin.")
        return refused_page(request, "Convert raw images", page_raw_upload(
            cfg, user, "Raw files are sent one at a time, which needs JavaScript. Turn JavaScript on and try again "
                       "(nothing was kept)."), 400)

    def raw_sessions_page(request: Request):
        user = ctx.user(request)
        if user.get("role") != "admin":
            return forbidden(request, "Picking a raw session is for the admin.")
        return ctx.page(request, "Pick a raw session", page_raw_sessions(cfg, user), active="predict")

    def find_raw(key):
        return next((s for s in raw_sessions(cfg) if s["key"] == key), None)

    def raw_session_page(request: Request):
        user = ctx.user(request)
        if user.get("role") != "admin":
            return forbidden(request, "Picking a raw session is for the admin.")
        s = find_raw(request.path_params["key"])
        if s is None:
            return not_found(request)
        return ctx.page(request, s["label"], page_raw_session(cfg, user, s, raw_montages(s)), active="predict")

    async def raw_session_post(request: Request):
        user = ctx.user(request)
        if user.get("role") != "admin":
            return await run_in_threadpool(forbidden, request, "Picking a raw session is for the admin.")
        s = await run_in_threadpool(find_raw, request.path_params["key"])
        if s is None:
            return await run_in_threadpool(not_found, request)
        fd = await request.form(max_files=0, max_fields=RAW_MAX_FILES + 20)
        form = {k: v for k, v in fd.items() if isinstance(v, str)}
        form["montages"] = [v for v in fd.getlist("montages") if isinstance(v, str)]
        try:
            d = await run_in_threadpool(session_convert, cfg, user, s, form)
        except Refusal as e:
            msg, status = admin_text(e, user), e.status
            return await run_in_threadpool(lambda: refused_page(
                request, s["label"], page_raw_session(cfg, user, s, raw_montages(s), form, msg), status))
        return RedirectResponse(f"/predict/convert/{d.name}/", status_code=303)

    def convert_page(request: Request):
        user = ctx.user(request)
        d, meta, st = load_conversion(cfg, user, ctx.can_see, request.path_params["cid"])
        if d is None:
            return not_found(request)
        if st.get("state") in ("queued", "running"):
            kick(cfg)
            ensure_watcher(cfg)
            st = _read_json(d / PS.STATUS) or st
        return ctx.page(request, "Raw conversion", page_convert(cfg, user, ctx.can_see, d, meta, st),
                        active="predict")

    def convert_action(kind):
        def act(request: Request):            # sync: a kill or a few file changes under the runner lock
            user = ctx.user(request)
            if user.get("role") != "admin":
                return forbidden(request, "Only the admin can stop, retry or clean up a conversion.")
            d, meta, _st = load_conversion(cfg, user, ctx.can_see, request.path_params["cid"])
            if d is None:
                return not_found(request)
            try:
                if kind == "stop":
                    stop_conversion(cfg, d)
                elif kind == "retry":
                    retry_conversion(cfg, d, meta)
                else:
                    delete_inputs_now(cfg, d)
            except Refusal as e:
                return refused_page(request, "Raw conversion", wrap(
                    f'<div class="cj-predict"><p class="cj-mut"><a href="/predict/convert/{esc(d.name)}/">← Back</a>'
                    f'</p>{msg_box(e.msg)}</div>'), e.status)
            return RedirectResponse(f"/predict/convert/{d.name}/", status_code=303)
        return act

    def set_page(request: Request):
        user = ctx.user(request)
        d, req = load_set(cfg, user, ctx.can_see, request.path_params["ps"])
        if d is None:
            return not_found(request)
        return ctx.page(request, "Analysis", page_set(cfg, user, d, req, ctx.can_see), active="predict")

    def action(kind):
        def act(request: Request):            # sync: file checks and a thread start, no long work here
            user = ctx.user(request)
            d, req = load_set(cfg, user, ctx.can_see, request.path_params["ps"])
            if d is None:
                return not_found(request)
            ps = d.name
            st = _read_json(d / PS.STATUS) or {}
            back = RedirectResponse(f"/predict/{ps}/", status_code=303)
            if kind == "start":
                restart(cfg, d)                    # only when can_restart(); marks the set before spawning
                return back
            if st.get("state") != "done":
                return back
            if kind == "review":
                pkg = package_dir(cfg, ps)
                if pkg is not None:
                    return RedirectResponse(f"/review/{pkg.name}/", status_code=303)
                start_task("review", ps, lambda: build_review(cfg, d))
            else:
                res, pkg = results_file(d), package_dir(cfg, ps)
                if res is None or edits_mtime(pkg) > res.stat().st_mtime:
                    start_task("results", ps, lambda: build_results(cfg, d, ps))
            return back
        return act

    # ---- JSON
    def api_mine(request: Request):
        user = ctx.user(request)
        srcs, _ = sources_safe(cfg, user, ctx.can_see)
        items = [item_of(cfg, user, d, req, st, srcs) for d, req, st in list_sets(cfg, user, ctx.can_see)]
        return ctx.json({"schema": SCHEMA_MINE, "items": items})

    def api_set(request: Request):
        user = ctx.user(request)
        d, req = load_set(cfg, user, ctx.can_see, request.path_params["ps"])
        if d is None:
            return ctx.json({"error": "not found"}, 404)
        return ctx.json(set_view(cfg, user, d, req, request.query_params.get("html") == "1"))

    def api_convert(request: Request):
        user = ctx.user(request)
        d, meta, st = load_conversion(cfg, user, ctx.can_see, request.path_params["cid"])
        if d is None:
            return ctx.json({"error": "not found"}, 404)
        if st.get("state") in ("queued", "running"):
            kick(cfg)
            ensure_watcher(cfg)
            st = _read_json(d / PS.STATUS) or st
        fr = conv_fragments(cfg, user, ctx.can_see, d, meta, st)
        out = {"schema": SCHEMA_CONVERT_VIEW, "id": d.name, "session": meta.get("session"), "state": st.get("state"),
               "n_tiles": st.get("n_tiles"), "warnings": st.get("warnings") or [], "live": fr["live"]}
        if request.query_params.get("html") == "1":
            out["html"] = {"status": fr["status"], "result": fr["result"]}
        return ctx.json(out)

    # ---- the upload API (one file, or one piece of a file, per request)
    async def read_body(request: Request, cap: int):
        try:
            if int(request.headers.get("content-length") or 0) > cap:
                raise Refusal("Request too big.", 413)
        except ValueError:
            raise Refusal("Bad request.", 400)
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > cap:
                raise Refusal("Request too big.", 413)
        try:
            return json.loads(bytes(buf) or b"{}")
        except ValueError:
            raise Refusal("Bad request (not JSON).", 400)

    async def api_upload_start(request: Request):
        user = ctx.user(request)
        try:
            body = await read_body(request, START_MAX_BYTES)
            return ctx.json(await run_in_threadpool(upload_start, cfg, user, body))
        except Refusal as e:
            return refusal_json(e, user, True)
        except OSError as e:
            return refusal_json(Refusal(os_error_text(e), 500, tech=f"{type(e).__name__}: {e}"), user, True)

    def upload_lock(uid):
        lock = _UP_ALOCKS.get(uid)
        if lock is None:
            lock = _UP_ALOCKS[uid] = asyncio.Lock()
        return lock

    GONE = {"error": "This upload is gone (finished, refused, or unused for an hour). Start again.", "dropped": True}

    def api_upload_state(request: Request):
        d, meta = load_upload(cfg, ctx.user(request), request.path_params["uid"])
        if d is None:
            return ctx.json(GONE, 404)
        return ctx.json(upload_state(d, meta))

    async def api_upload_file(request: Request):
        user, uid = ctx.user(request), request.path_params["uid"]
        d, _ = await run_in_threadpool(load_upload, cfg, user, uid)
        if d is None:
            return ctx.json(GONE, 404)
        async with upload_lock(uid):
            try:
                return ctx.json(await receive_piece(request, d, uid))
            except Resync:
                meta = await run_in_threadpool(_read_json, d / UPLOAD_META)
                if not isinstance(meta, dict):
                    return ctx.json(GONE, 404)
                return ctx.json(upload_state(d, meta), 409)
            except ClientDisconnect:                     # the page resumes from what arrived
                return Response(status_code=400)
            except Refusal as e:
                # a size or content violation (or a full disk) ends the session; a bad request leaves it as it is
                drop = e.drop or e.status == 507
                if drop:
                    await run_in_threadpool(drop_upload, d, uid)
                return refusal_json(e, user, drop)
            except OSError as e:
                await run_in_threadpool(drop_upload, d, uid)
                return refusal_json(Refusal(os_error_text(e), 500, tech=f"{type(e).__name__}: {e}"), user, True)

    async def api_upload_create(request: Request):
        user, uid = ctx.user(request), request.path_params["uid"]
        d, meta = await run_in_threadpool(load_upload, cfg, user, uid)
        if d is None:
            return ctx.json(GONE, 404)
        async with upload_lock(uid):
            try:
                href = await run_in_threadpool(upload_create, cfg, user, d, meta)
            except Refusal as e:
                keep = e.status in (409, 429, 503)       # files still to send, a limit, a busy GPU: try again later
                if not keep:
                    await run_in_threadpool(drop_upload, d, uid)
                return refusal_json(e, user, not keep)
            except OSError as e:                         # the files stay staged: pressing the button again retries
                return refusal_json(Refusal(os_error_text(e), 500, tech=f"{type(e).__name__}: {e}"), user, False)
            await run_in_threadpool(drop_upload, d, uid)
        return ctx.json({"href": href})

    async def api_upload_cancel(request: Request):
        user, uid = ctx.user(request), request.path_params["uid"]
        d, _ = await run_in_threadpool(load_upload, cfg, user, uid)
        if d is None:
            return ctx.json(GONE, 404)
        async with upload_lock(uid):
            await run_in_threadpool(drop_upload, d, uid)
        return ctx.json({"ok": True})

    def script(request: Request):
        r = Response(JS, media_type="text/javascript; charset=utf-8")
        r.headers["Cache-Control"] = "no-cache"
        return r

    return [Route(PREFIX + "/", index, methods=["GET"]),
            Route(PREFIX + "/predict.js", script, methods=["GET"]),
            Route(PREFIX + "/new/", new, methods=["GET"]),
            Route(PREFIX + "/new/session/{key}/", session_form, methods=["GET"]),
            Route(PREFIX + "/new/session/{key}/", session_post, methods=["POST"]),
            Route(PREFIX + "/new/upload/", upload_form, methods=["GET"]),
            Route(PREFIX + "/new/upload/", upload_post, methods=["POST"]),
            Route(PREFIX + "/new/raw/", raw_form, methods=["GET"]),
            Route(PREFIX + "/new/raw/", raw_post, methods=["POST"]),
            Route(PREFIX + "/new/raw/sessions/", raw_sessions_page, methods=["GET"]),
            Route(PREFIX + "/new/raw/sessions/{key}/", raw_session_page, methods=["GET"]),
            Route(PREFIX + "/new/raw/sessions/{key}/", raw_session_post, methods=["POST"]),
            Route(PREFIX + "/convert/{cid}/", convert_page, methods=["GET"]),
            Route(PREFIX + "/convert/{cid}/stop", convert_action("stop"), methods=["POST"]),
            Route(PREFIX + "/convert/{cid}/retry", convert_action("retry"), methods=["POST"]),
            Route(PREFIX + "/convert/{cid}/inputs", convert_action("inputs"), methods=["POST"]),
            Route(PREFIX + "/api/mine", api_mine, methods=["GET"]),
            Route(PREFIX + "/api/set/{ps}", api_set, methods=["GET"]),
            Route(PREFIX + "/api/convert/{cid}", api_convert, methods=["GET"]),
            Route(PREFIX + "/api/upload", api_upload_start, methods=["POST"]),
            Route(PREFIX + "/api/upload/{uid}", api_upload_state, methods=["GET"]),
            Route(PREFIX + "/api/upload/{uid}/file", api_upload_file, methods=["POST"]),
            Route(PREFIX + "/api/upload/{uid}/create", api_upload_create, methods=["POST"]),
            Route(PREFIX + "/api/upload/{uid}/cancel", api_upload_cancel, methods=["POST"]),
            Route(PREFIX + "/{ps}/", set_page, methods=["GET"]),
            Route(PREFIX + "/{ps}/review", action("review"), methods=["POST"]),
            Route(PREFIX + "/{ps}/results", action("results"), methods=["POST"]),
            Route(PREFIX + "/{ps}/start", action("start"), methods=["POST"])]
