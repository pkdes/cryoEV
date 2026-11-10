"""
Predict, shared pieces: the settings and limits, Refusal, the predict / review / registry roots, small readers
(JSON, local times) and the HTML helpers every page uses (all text through esc).

Split out of portal/modules/predict.py (round 7, P7; no behaviour change); predict.py re-exports every name.
Names the tests patch are used from other parts only as <PART>.<name> (portal/predict_parts/__init__.py).
"""
from __future__ import annotations

import errno
import html
import json
import os
import re
from datetime import datetime
from pathlib import Path

from jobs import predict_set as PS
from portal import names as NM


HERE = Path(__file__).resolve().parent.parent / "modules"   # portal/modules, as before the split
REPO = HERE.parent.parent
TEMPLATE = HERE.parent / "templates" / "predict.html"
MODELS_YAML = REPO / "models" / "models.yaml"
PORTAL_TMP = ".portal"                    # <root>/.portal/: the portal's own scratch (consumers skip dot entries)
IMAGE_EXT = (".png", ".jpg", ".jpeg")
SAMPLE_COLUMNS = ("sample_label", "sample", "grid")
MAX_FILES = PS.MAX_IMAGES                 # 1000 tiles per analysis or upload (= a prediction set's maximum)
TILES_MAX_TOTAL_BYTES = 20 * 1024 ** 3    # one tile upload, all files together (raw uploads: RAW_MAX_TOTAL_BYTES)
DISK_NEED_FACTOR = 1.1                    # a set's tiles are copied: their total + 10 % must be free (links need none)
EST_FALLBACK = (12.0, 0.17)               # seconds: a + b * images, until 3 finished laptop sets fit a line
RESUBMIT_S = 600                          # a finished pick pressed again within this opens its set
EST_TTL_S = 600                           # the fitted line is cached this long
EST_MAX_S_PER_IMAGE = 5                   # a finished run slower than this per image is a stall, not data
EST_MAX_SETS = 300                        # finished sets read for the fit, newest first
CREATE_ID_RE = re.compile(r"^(?:us|sc)_[0-9a-f]{12}$")   # a background create: us_ = an upload id, sc_ = a session pick
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


# ---------------------------------------------------------------- small readers and times
def _read_json(p: Path):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _parse_local(s) -> float | None:
    try:
        dt = datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None
    return dt.timestamp()


when = NM.when                            # shared with names.py (Results and My reviews use it too)


def ago(seconds) -> str:
    s = max(0, int(seconds))
    if s < 90:
        return f"{s} s"
    if s < 5400:
        return f"{round(s / 60)} min"
    return f"{s / 3600:.1f} h"


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


# ---------------------------------------------------------------- text for people (no folder paths; OS errors)
def _plain(msg: str, paths) -> str:
    """A refusal without our folder paths in it (the person never typed them)."""
    out = str(msg)
    for p in paths:
        for s in {str(p), str(Path(p).resolve())}:
            out = out.replace(s + os.sep, "").replace(s, "the images")
    return out


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def os_error_text(e: OSError) -> str:
    too_long = getattr(e, "winerror", None) == 206 or getattr(e, "errno", None) == errno.ENAMETOOLONG or \
        "too long" in str(e).lower()
    if too_long:
        return "A file name is too long for the server. Shorten the file names and upload again."
    return "The server could not store the upload. Try again, or ask the admin."
