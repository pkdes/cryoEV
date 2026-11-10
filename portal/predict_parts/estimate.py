"""
Predict, how long an analysis takes: a + b * images fitted to finished laptop runs (cached), the queue ahead.

Split out of portal/modules/predict.py (round 7, P7; no behaviour change); predict.py re-exports every name.
Names the tests patch are used from other parts only as <PART>.<name> (portal/predict_parts/__init__.py).
"""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

from jobs import predict_set as PS

from portal.predict_parts.common import (EST_FALLBACK, EST_MAX_SETS, EST_MAX_S_PER_IMAGE, EST_TTL_S, _parse_local,
                                         _read_json, esc)
from portal.predict_parts import common as COMMON


# ---------------------------------------------------------------- how long an analysis takes (estimate)
_EST_LOCK = threading.Lock()
_EST_CACHE: dict = {}                    # predict root -> (time, (a, b, fitted))
_RUN_START = re.compile(r"^=== (\S+) predict_local ")


def run_seconds(d: Path):
    """(images, seconds) of one finished laptop run: the LAST `=== <iso> predict_local` line of predict.log (the log
    is appended across restarts; the last start is the run that finished) to status.json `updated`. None when this
    set is not a finished laptop run, its log can't be read, or the figure is implausible (over EST_MAX_S_PER_IMAGE
    seconds per image, or not positive)."""
    st = _read_json(d / PS.STATUS)
    if not isinstance(st, dict) or st.get("state") != "done" or st.get("where") not in (None, "laptop"):
        return None
    n = st.get("n_total")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        return None
    try:
        last = None
        with open(d / PS.C.PREDICT_LOG, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _RUN_START.match(line)
                if m:
                    last = m.group(1)
        if last is None:
            return None
        t0, t1 = _parse_local(last), _parse_local(st.get("updated"))
        if t0 is None or t1 is None or not 0 < t1 - t0 <= EST_MAX_S_PER_IMAGE * n:
            return None
        return n, t1 - t0
    except OSError:
        pass
    return None


def fit_line(points):
    """Least squares through (images, seconds) -> (a, b), or None: fewer than 3 distinct image counts, or a line
    that does not rise (a bigger set is not faster)."""
    pts = [(float(n), float(t)) for n, t in points]
    if len({n for n, _ in pts}) < 3:
        return None
    mx, my = sum(n for n, _ in pts) / len(pts), sum(t for _, t in pts) / len(pts)
    sxx = sum((n - mx) ** 2 for n, _ in pts)
    b = sum((n - mx) * (t - my) for n, t in pts) / sxx
    a = my - b * mx
    return (a, b) if b > 0 else None


def estimate_line(cfg, now=None):
    """-> (a, b, fitted): seconds = a + b * images. Fitted across the finished laptop sets (3+ with distinct image
    counts), else EST_FALLBACK. Cached EST_TTL_S."""
    now = time.time() if now is None else now
    root = COMMON.predict_root(cfg)
    key = str(root)
    with _EST_LOCK:
        hit = _EST_CACHE.get(key)
        if hit and now - hit[0] < EST_TTL_S:
            return hit[1]
    pts = []
    try:
        dirs = sorted((p for p in root.iterdir() if p.is_dir() and PS.PS_RE.match(p.name)), reverse=True)
    except OSError:
        dirs = []
    for d in dirs[:EST_MAX_SETS]:
        r = run_seconds(d)
        if r:
            pts.append(r)
    fit = fit_line(pts)
    line = (fit[0], fit[1], True) if fit else (*EST_FALLBACK, False)
    with _EST_LOCK:
        _EST_CACHE[key] = (now, line)
    return line


def queue_ahead(cfg) -> int:
    """Analyses queued or running on this server (anyone's)."""
    try:
        dirs = [p for p in COMMON.predict_root(cfg).iterdir() if p.is_dir() and PS.PS_RE.match(p.name)]
    except OSError:
        return 0
    return sum(1 for d in dirs if (_read_json(d / PS.STATUS) or {}).get("state") in ("queued", "running"))


def estimate_text(a: float, b: float, n: int, ahead: int = 0) -> str:
    """'about 3 min once it starts' (+ ' · 2 analyses ahead of you'); '' when there are no images."""
    if n < 1:
        return ""
    mins = max(1, int((a + b * n) / 60 + 0.5))     # half up, as predict.js's Math.round
    out = f"about {mins} min once it starts"
    if ahead > 0:
        out += f" · {ahead} {'analysis' if ahead == 1 else 'analyses'} ahead of you"
    return out


def estimate_html(cfg, n: int) -> str:
    """The time line next to a tile count. predict.js rewrites it when the count changes (data-a, data-b, data-ahead)."""
    a, b, _ = estimate_line(cfg)
    ahead = queue_ahead(cfg)
    return (f'<p class="cj-mut" id="cj-predict-est" data-a="{a:.4f}" data-b="{b:.5f}" data-ahead="{ahead}" '
            f'role="status">{esc(estimate_text(a, b, n, ahead))}</p>')
