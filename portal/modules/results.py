"""
Results module (CryoEV portal): EV counts, sizes and multilayer % for one prediction set, in plain words, phone first.
Reads the cached <CryoAI-laptop>/predict/<ps_id>/results/results.json (BUILD_CONTRACTS.md section 4, results/1);
when it is missing, a button computes it in a worker thread (results.compute_and_write) and the page polls.

    GET  /results/                         my analyses: every prediction set the viewer may see
    GET  /results/<ps_id>/                 the results page (or the compute button / progress)
    POST /results/<ps_id>/compute          start (or redo) the computation in a thread; 303 back to the page
    GET  /results/<ps_id>/api/status       {"state": none|queued|running|ready|error|waiting, ...} for the poll
    GET  /results/<ps_id>/download/<file>  per_sample.csv, per_image.csv, results.json, results_<ps_id>.zip
                                           (the three + README.txt with the provenance); always attachments
    GET  /results/results.js               the page's script (a file: no inline script)

Rules:
  - Owner = <ps>/request.json `owner`; ctx.can_see(user, owner) for every route under /results/<ps_id>/. A missing
    set and someone else's set get the same 404 page (no hint which).
  - Honest about review: the banner says how many images a person checked ("Model only" when none), the reviewer
    columns appear only when an image was reviewed, and each image is marked reviewed / edited / model only.
  - Geometry multilayer % is the headline; the reviewer % is shown beside it with its base (n EVs on n images),
    flagged "small sample" under SMALL_BASE reviewer EVs.
  - Out of date: when the review package (<review root>/pred_<ps_id>/edits/*.json, or its build) changed after the
    worker READ it (results/.review_read.json: read time + edit file names, written after results.json for those
    exact results), or an edit file it counted is gone; without that record (a CLI run), after results.json was
    written (it is written last). A banner says so and the index chip reads "Out of date". After a failed
    recompute the earlier results stay, labelled.
  - Round 7 (R2): rejected images (reviewed/1 `rejected`) are dropped before every per-sample number (see
    results/compute.py); the page says "Counted EVs 120 (8 low conf) · 3 images rejected", marks rejected images and
    shows an "All images rejected" notice for a sample with none left. Streamed zips (portal/zipstream.stream_zip,
    the Gallery's 2-per-person slots): GET /results/<ps>/download/{images,model_outlines,reviewed_labels,everything}_
    <ps>.zip. Images and Model outlines need no results.json; Reviewed labels are built at download time from
    reviewed_source() (COCO under the one COCO rule + YOLO + rejected_images.csv + provenance.csv); Everything adds
    the results and a README that says when they are out of date.
  - Downloads: a CSV newer than results.json (a write in progress or interrupted) is not served: 404 "compute
    again".
  - Images of a refused sample are listed muted, "Not counted (sample refused)".
  - Links: the spot-check subset at /predict/<ps_id>/; "Review these images" at /review/pred_<ps_id>/ when the
    package exists (the shell checks the owner there).
  - Standalone CSV downloads start with # provenance lines; the zip (tables + README.txt) is the recommended one.
  - The set folder is resolved and must still be <predict root>/<ps_id> (a junction or link out of it = 404).
  - A refused sample (nm/px spread over 1 %, mixed tile size, no confirmed nm/px) is a notice with the reason; the
    other samples are still shown. A sample whose nm/px varies within 1 % (BUILD_CONTRACTS.md section 8) is counted;
    its range is in "How these numbers were made", the CSV # lines, the README and a note under the sample table. A whole-set refusal (predictions not finished) shows the notice only.
  - Compute: one at a time (a second one waits, "queued"). It uses the review package's edits when
    <CryoAI-laptop>/review/pred_<ps_id>/ exists (annotation.prediction_package.reviewed_polygons, write=False, so
    the package is never written), else the model predictions. Writes only <ps>/results/ (compute_and_write).
  - Roots: cfg.predict_root / cfg.review_root when the portal Config has them, else
    data_utils.config.laptop_root() / "predict" and / "review".
  - What is counted (round 6): a header line says EVs = outlines at conf >= the rule's conf plus everything kept or
    drawn in review; per sample the table shows the unsure outlines still open (results.json n_unsure_open: candidates
    nobody accepted or rejected; untouched image = all, edited = open ones, Reviewed = 0) and "checked N of M
    images" (n_checked). A nudge links to the review when any are open. A results.json written before those fields
    shows "Update to see unsure counts" and an Update button.
  - Update: the out-of-date banner has an "Update results now" form (POST /results/<ps>/compute, plain form, Origin
    checked); an index card is a title link + meta + an "Update" form when out of date (no form inside a link). Nothing
    recomputes by itself.
  - EV size: next to the median/Q1-Q3 chart, one server-drawn histogram per sample with EVs (results.json diam_hist:
    10 nm bins, 0-500 nm + a >= 500 nm bin; hover titles; a table under <details>).
  - The analysis is shown by its display name (portal/names.py), the ps id as small text.
  - Every text goes through html.escape; the page loads nothing from another origin.
"""
from __future__ import annotations

import html
import io
import json
import math
import re
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route

from portal import names as NM

PREFIX = "/results"
ROLES = ("admin", "advanced", "basic")

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "templates" / "results.html"
PS_RE = re.compile(r"^ps_\d{8}_[0-9a-f]{6}$")
SCHEMA = "results/1"
FILES = {"per_sample.csv": "text/csv; charset=utf-8", "per_image.csv": "text/csv; charset=utf-8",
         "results.json": "application/json"}
NOT_FOUND = "No analysis with this name, or it is not yours."
SMALL_BASE = 20              # reviewer % on fewer reviewer EVs than this is flagged "small sample"
NM_TOLERANCE = 0.01          # = results.compute.NM_TOLERANCE (the text here; a test pins both equal)

esc = html.escape

# compute_and_write(ps_dir, reviewed_json) -> (result, out_dir); None = results.compute_and_write (loaded lazily, so
# the portal starts without numpy / the counting chain). Tests replace it.
COMPUTE = None
REVIEWED = None              # reviewed_polygons(pkg_dir, write=False) -> dict; None = annotation.prediction_package

_jobs: dict = {}             # str(ps_dir) -> {"state": queued|running|error, "error", "detail", "started"}
_jobs_lock = threading.Lock()
_slot = threading.Semaphore(1)      # one computation at a time (CPU + memory on the laptop)


# ---------------------------------------------------------------- where things are
def _laptop():
    from data_utils import config
    return config.laptop_root(required=False)


def predict_root(cfg) -> Path | None:
    p = getattr(cfg, "predict_root", None)
    if p:
        return Path(p)
    lap = _laptop()
    return lap / "predict" if lap else None


def review_root(cfg) -> Path | None:
    """cfg.review_root (portal Config; None there = no review root), else <CryoAI-laptop>/review."""
    if hasattr(cfg, "review_root"):
        return Path(cfg.review_root) if cfg.review_root else None
    lap = _laptop()
    return lap / "review" if lap else None


def package_dir(cfg, ps_id: str) -> Path | None:
    """<review root>/pred_<ps_id> when that package is complete (review_package.json is written last), else None."""
    root = review_root(cfg)
    pkg = root / f"pred_{ps_id}" if root else None
    return pkg if pkg is not None and (pkg / "review_package.json").is_file() else None


REVIEW_READ = ".review_read.json"     # <ps>/results/: when the worker read the review, and which edit files existed


def edit_names(cfg, ps_id: str) -> list:
    pkg = package_dir(cfg, ps_id)
    try:
        return sorted(p.name for p in (pkg / "edits").glob("*.json")) if pkg else []
    except OSError:
        return []


def review_change(cfg, ps_id: str) -> tuple:
    """(newest mtime of the package's edits/*.json or its build, 'edit' | 'build') or (None, None)."""
    pkg = package_dir(cfg, ps_id)
    if pkg is None:
        return None, None
    best, kind = None, None
    try:
        for p in (pkg / "edits").glob("*.json"):
            t = p.stat().st_mtime
            if best is None or t > best:
                best, kind = t, "edit"
        t = (pkg / "review_package.json").stat().st_mtime
        if best is None or t > best:
            best, kind = t, "build"
    except OSError:
        pass
    return best, kind


def review_read(ps: dict, res_computed) -> dict | None:
    """The worker's record of the review it counted, when it belongs to these results (same `computed`)."""
    doc = _read_json(ps["dir"] / "results" / REVIEW_READ)
    if (isinstance(doc, dict) and doc.get("computed") == res_computed and isinstance(doc.get("read_at"), (int, float))
            and isinstance(doc.get("edits"), list)):
        return doc
    return None


def stale_info(cfg, ps: dict) -> dict | None:
    """{"at": time, "kind": "edit" | "build" | "removed"} when the review changed after the results counted it; else
    None. The reference time is when the worker READ the review (results/.review_read.json, written by the portal's
    worker for these exact results), so an edit saved while the computation ran is caught; without that record (a CLI
    run), results.json's mtime (it is written last). With the record, an edit file that is gone = "removed"."""
    rdir = ps["dir"] / "results"
    try:
        ref = (rdir / "results.json").stat().st_mtime
    except OSError:
        return None
    res = _read_json(rdir / "results.json")
    rr = review_read(ps, res.get("computed") if isinstance(res, dict) else None)
    if rr:
        ref = float(rr["read_at"])
    at, kind = review_change(cfg, ps["id"])
    if at is not None and at > ref:
        return {"at": at, "kind": kind}
    if rr and set(rr["edits"]) - set(edit_names(cfg, ps["id"])):
        return {"at": None, "kind": "removed"}
    return None


def _write_review_read(ps: dict, computed, read_at: float, edits: list) -> None:
    p = ps["dir"] / "results" / REVIEW_READ
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps({"computed": computed, "read_at": read_at, "edits": edits}), encoding="utf-8")
    tmp.replace(p)


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_set(cfg, ps_id: str, user: dict, can_see) -> dict | None:
    """The prediction set the user may see, or None (missing, malformed or not theirs: one answer for all)."""
    root = predict_root(cfg)
    if not PS_RE.match(ps_id or "") or root is None:
        return None
    try:                             # resolved, and still <root>/<ps_id>: no junction / link out of the root
        root = root.resolve()
        d = (root / ps_id).resolve()
    except OSError:
        return None
    if d.parent != root or d.name != ps_id:
        return None
    req = _read_json(d / "request.json") if d.is_dir() else None
    if not isinstance(req, dict) or req.get("ps") != ps_id or not can_see(user, req.get("owner")):
        return None
    status = _read_json(d / "status.json")
    return {"id": ps_id, "dir": d, "request": req, "status": status if isinstance(status, dict) else None}


def read_results(ps: dict):
    """-> (results dict | None, problem text | None)."""
    p = ps["dir"] / "results" / "results.json"
    if not p.is_file():
        return None, None
    res = _read_json(p)
    if not isinstance(res, dict) or res.get("schema") != SCHEMA or res.get("prediction_set") != ps["id"]:
        return None, "The saved results file can not be read. Compute the results again."
    for k in ("samples", "images"):
        if not isinstance(res.get(k), dict):
            res[k] = {}
    return res, None


# ---------------------------------------------------------------- the computation (a worker thread)
def _compute_fn():
    if COMPUTE is not None:
        return COMPUTE
    from results.compute import compute_and_write
    return compute_and_write


def reviewed_source(cfg, ps_id: str):
    """The review package's current edits as reviewed/1 (not written anywhere), or None without a package."""
    pkg = package_dir(cfg, ps_id)
    if pkg is None or not (pkg / "prediction_package.json").is_file():
        return None
    fn = REVIEWED
    if fn is None:
        from annotation.prediction_package import reviewed_polygons as fn
    return fn(pkg, write=False)


def job_state(ps: dict) -> dict | None:
    with _jobs_lock:
        j = _jobs.get(str(ps["dir"]))
        return dict(j) if j else None


def _worker(cfg, ps: dict, key: str):
    try:
        with _slot:
            with _jobs_lock:
                _jobs[key]["state"] = "running"
            read_at, edits = time.time(), edit_names(cfg, ps["id"])     # taken BEFORE the read: conservative
            reviewed = reviewed_source(cfg, ps["id"])
            res, _ = _compute_fn()(ps["dir"], reviewed)
            # results.json is committed; record which review it counted (stale_info compares edits with read_at)
            _write_review_read(ps, res.get("computed"), read_at, edits)
        with _jobs_lock:
            _jobs.pop(key, None)
    except Exception as e:  # shown on the page: plain words for everyone, the detail for admins
        with _jobs_lock:
            _jobs[key] = {"state": "error", "error": "Computing the results failed.",
                          "detail": f"{type(e).__name__}: {e}", "started": _jobs.get(key, {}).get("started")}


def start_compute(cfg, ps: dict) -> bool:
    """Start the worker unless one is already queued or running for this set. -> True when started."""
    key = str(ps["dir"])
    with _jobs_lock:
        j = _jobs.get(key)
        if j and j["state"] in ("queued", "running"):
            return False
        _jobs[key] = {"state": "queued", "error": None, "detail": None, "started": time.time()}
    threading.Thread(target=_worker, args=(cfg, ps, key), daemon=True, name=f"cj-results-{ps['id']}").start()
    return True


def predictions_ready(ps: dict) -> bool:
    return bool(ps["status"]) and ps["status"].get("state") == "done"


def status_doc(cfg, ps: dict) -> dict:
    j = job_state(ps)
    has = (ps["dir"] / "results" / "results.json").is_file()
    stale = stale_info(cfg, ps) if has else None
    if j:
        state = j["state"]
    elif has:
        state = "ready"
    elif not predictions_ready(ps):
        state = "waiting"
    else:
        state = "none"
    return {"ps": ps["id"], "state": state, "has_results": has, "out_of_date": bool(stale),
            "error": j.get("error") if j else None}


# ---------------------------------------------------------------- formatting
def fnum(v, nd=1) -> str:
    """1 decimal, a trailing .0 dropped; '–' for null."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    s = f"{float(v):.{nd}f}"
    return s[:-2] if s.endswith(".0") else s


def fint(v) -> str:
    return "–" if v is None else str(int(v))


def fpct(v) -> str:
    return "–" if v is None else fnum(v) + "%"


def fdiam(s: dict) -> tuple:
    med = s.get("diam_nm_median")
    if med is None:
        return "–", "no EVs"
    return f"{float(med):.0f} nm", f"{float(s.get('diam_nm_q1')):.0f}–{float(s.get('diam_nm_q3')):.0f}"


def plural(n, one, many=None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def when(iso) -> str:
    """An ISO time as the portal's local time (an offset is converted; a naive time is taken as local)."""
    try:
        return datetime.fromisoformat(str(iso)).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(iso or "unknown")


def hhmm(ts: float) -> str:
    """A file time: HH:MM today, else YYYY-MM-DD HH:MM (local)."""
    t = datetime.fromtimestamp(ts).astimezone()
    return t.strftime("%H:%M") if t.date() == datetime.now().astimezone().date() else t.strftime("%Y-%m-%d %H:%M")


def ev_rule(classes) -> str:
    cs = list(classes or [])
    return "EV = " + (", ".join(f"class {c}" + (" (EV)" if c == 0 else "") for c in cs) if cs else "?")


def chip(label, tone) -> str:
    return f'<span class="cj-results-chip cj-results-t-{esc(tone)}">{esc(label)}</span>'


def image_chip(img: dict, counted=True) -> str:
    if img.get("rejected"):
        why = str(img.get("reject_reason") or "").strip()
        return chip("Rejected" + (f": {why}" if why else ""), "warn")
    if not counted:
        return chip("Not counted (sample refused)", "off")
    if img.get("reviewed"):
        return chip("Reviewed", "ok")
    if img.get("edited"):
        return chip("Edited", "edit")
    return chip("Model only", "model")


def td(label, value, cls="", k=None, sub=None, flag=None) -> str:
    kk = f' data-k="{esc(k)}"' if k else ""
    s = f'<small class="cj-results-sub">{esc(sub)}</small>' if sub else ""
    f = (" " + chip(flag, "warn")) if flag else ""
    c = f' class="{cls}"' if cls else ""
    return f'<td data-label="{esc(label)}"{c}{kk}><span class="cj-results-v">{esc(value)}</span>{f}{s}</td>'


# ---------------------------------------------------------------- charts (inline SVG; x in %, so it fits any width)
def nice_ticks(lo: float, hi: float, n=5) -> list:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    t, out = math.floor(lo / step) * step, []
    while t <= hi + step * 1e-9:
        out.append(round(t, 6))
        t += step
    if out[-1] < hi:
        out.append(round(out[-1] + step, 6))
    return out


def diam_chart(samples: dict) -> str:
    """EV diameter per sample: a dot at the median and a bar over the middle half (Q1–Q3), one shared nm axis.
    results/1 carries only these three numbers per sample (no per-EV list), so this is not a histogram."""
    rows = [(k, s) for k, s in samples.items() if s.get("diam_nm_median") is not None]
    if not rows:
        return '<p class="cj-mut">No EV sizes to draw.</p>'
    lo = min(float(s["diam_nm_q1"]) for _, s in rows)
    hi = max(float(s["diam_nm_q3"]) for _, s in rows)
    pad = max((hi - lo) * 0.1, 5.0)
    ticks = nice_ticks(max(0.0, lo - pad), hi + pad)
    a, b = ticks[0], ticks[-1]
    L, R = 2.0, 98.0                                    # plot area, % of the width (room for the end labels)

    def x(v):
        return L + (float(v) - a) / (b - a) * (R - L)

    row_h, top = 46, 4
    h = top + row_h * len(rows) + 26
    parts = [f'<svg class="cj-results-svg" width="100%" height="{h}" role="img" aria-labelledby="cj-results-ch-t">'
             f'<title id="cj-results-ch-t">{esc(chart_summary(rows))}</title>']
    for t in ticks:
        parts.append(f'<line class="cj-results-grid" x1="{x(t):.2f}%" x2="{x(t):.2f}%" y1="{top}" '
                     f'y2="{h - 22}"/>')
    for i, t in enumerate(ticks):
        anchor = "start" if i == 0 else ("end" if i == len(ticks) - 1 else "middle")
        parts.append(f'<text class="cj-results-axis" x="{x(t):.2f}%" y="{h - 6}" text-anchor="{anchor}">'
                     f'{esc((fnum(t, 0) if float(t).is_integer() else fnum(t)) + " nm")}</text>')
    for n, (label, s) in enumerate(rows):
        y0 = top + n * row_h
        yl, ym = y0 + 14, y0 + 30
        med, q1, q3 = float(s["diam_nm_median"]), float(s["diam_nm_q1"]), float(s["diam_nm_q3"])
        tip = (f"{label}: median {med:.0f} nm, middle half {q1:.0f}–{q3:.0f} nm, "
               f"{plural(int(s.get('ev_total') or 0), 'EV')}")
        parts.append(f'<g class="cj-results-row"><title>{esc(tip)}</title>'
                     f'<rect class="cj-results-hit" x="0" y="{y0}" width="100%" height="{row_h}"/>'
                     f'<text class="cj-results-lab" x="{L}%" y="{yl}"><tspan class="cj-results-lab-s">'
                     f'{esc(short(label))}</tspan><tspan class="cj-results-lab-v" dx="8">{esc(f"{med:.0f} nm")}</tspan>'
                     f'</text>'
                     f'<line class="cj-results-iqr" x1="{x(q1):.2f}%" x2="{x(q3):.2f}%" y1="{ym}" y2="{ym}"/>'
                     f'<circle class="cj-results-med" cx="{x(med):.2f}%" cy="{ym}" r="5"/></g>')
    parts.append("</svg>")
    return "".join(parts)


def hist_ok(s: dict) -> bool:
    h = s.get("diam_hist")
    return (isinstance(h, dict) and isinstance(h.get("counts"), list) and len(h["counts"]) >= 2
            and isinstance(h.get("bin_nm"), (int, float)) and h["bin_nm"] > 0
            and all(isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in h["counts"])
            and sum(h["counts"]) >= 1)


def bin_label(h: dict, k: int) -> str:
    """'40–50 nm' for bin k; the last bin is '500 nm or more' (max_nm and above)."""
    step, n = h["bin_nm"], len(h["counts"])
    return f"{k * step:g} nm or more" if k == n - 1 else f"{k * step:g}–{(k + 1) * step:g} nm"


def shared_xmax(samples: dict):
    """The one x-axis end (nm) for every histogram on the page: the upper edge of the bin holding the 99th
    percentile of all samples' EVs together, rounded up to 50 nm, at least 100 nm. None = draw the whole range."""
    hs = [s["diam_hist"] for s in samples.values() if hist_ok(s)]
    if not hs:
        return None
    step, n = hs[0]["bin_nm"], len(hs[0]["counts"])
    if any(h["bin_nm"] != step or len(h["counts"]) != n for h in hs):
        return None
    pooled = [sum(h["counts"][k] for h in hs) for k in range(n)]
    total, cum = sum(pooled), 0
    for k, c in enumerate(pooled):
        cum += c
        if cum >= 0.99 * total:
            break
    edge = (k + 1) * step
    return max(100, int(math.ceil(edge / 50.0)) * 50)


def hist_chart(idx: int, label: str, s: dict, xmax=None) -> str:
    """Server-drawn histogram of one sample's EV diameters (results.json diam_hist). One hue, bars separated by a
    2 px card-coloured stroke, recessive grid, a hover title per bin, no script. x in %, y in px, like diam_chart.
    xmax (nm; the page's shared axis end): bins from there up are folded into one '>= xmax' bar; the table still
    lists the original bins."""
    h = s["diam_hist"]
    full, step0 = h["counts"], h["bin_nm"]
    if xmax is not None and xmax < (len(full) - 1) * step0:
        k0 = int(xmax // step0)
        counts = full[:k0] + [sum(full[k0:])]
    else:
        counts = full
    n = len(counts)
    h2 = {"bin_nm": step0, "counts": counts}
    peak = max(counts)
    yt = nice_ticks(0, peak, 4)
    ymax = yt[-1]
    H, top, bot = 150, 8, 22
    ph = H - top - bot
    L, R = 7.0, 99.0
    bw = (R - L) / n
    step = h["bin_nm"]

    def y(v):
        return top + ph - v / ymax * ph

    tid = f"cj-results-hist-t{idx}"
    best = counts.index(peak)
    summary = (f"{label}: {plural(sum(counts), 'EV')} by diameter; most in {bin_label(h2, best)} "
               f"({plural(peak, 'EV')})")
    parts = [f'<svg class="cj-results-svg cj-results-hist" width="100%" height="{H}" role="img" '
             f'aria-labelledby="{tid}"><title id="{tid}">{esc(summary)}</title>']
    for t in yt:
        parts.append(f'<line class="cj-results-grid" x1="{L}%" x2="{R}%" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                     f'<text class="cj-results-axis" x="{L - 1}%" y="{y(t) + 4:.1f}" text-anchor="end">'
                     f'{esc(fnum(t, 0))}</text>')
    last = (n - 1) * step
    for t in nice_ticks(0, last, 5):
        if t >= last:
            continue                      # the overflow bin has its own label
        parts.append(f'<text class="cj-results-axis" x="{L + t / step * bw:.2f}%" y="{H - 6}" '
                     f'text-anchor="middle">{esc(fnum(t, 0))}</text>')
    parts.append(f'<text class="cj-results-axis" x="{R}%" y="{H - 6}" text-anchor="end">'
                 f'{esc("≥" + fnum(last, 0))}</text>')
    for k, c in enumerate(counts):
        x0 = L + k * bw
        tip = f"{bin_label(h2, k)}: {plural(c, 'EV')}"
        bar = (f'<rect class="cj-results-bar" x="{x0:.3f}%" y="{y(c):.1f}" width="{bw:.3f}%" '
               f'height="{top + ph - y(c):.1f}"/>') if c else ""
        parts.append(f'<g class="cj-results-binx"><title>{esc(tip)}</title>'
                     f'<rect class="cj-results-hit" x="{x0:.3f}%" y="{top}" width="{bw:.3f}%" height="{ph}"/>{bar}</g>')
    parts.append("</svg>")
    rows = "".join(f'<tr><td class="cj-results-l">{esc(bin_label(h, k))}</td><td>{c}</td></tr>'
                   for k, c in enumerate(full) if c)
    table = (f'<details class="cj-results-det cj-results-histtab"><summary>{esc("Table of the bins")}</summary>'
             f'<div class="cj-results-wrap"><table class="cj-results-tab"><thead><tr><th scope="col">Diameter</th>'
             f'<th scope="col">EVs</th></tr></thead><tbody>{rows}</tbody></table></div>'
             f'<p class="cj-mut cj-results-note">{esc("Only bins with EVs are listed. A bin includes its lower edge and excludes its upper (10 ≤ d < 20 nm).")}</p></details>')
    return (f'<figure class="cj-card cj-results-fig cj-results-histfig" data-sample="{esc(str(label))}">'
            f'<figcaption>{esc(short(label))}: {esc(plural(sum(counts), "EV"))} by diameter'
            f'<span class="cj-mut">{esc("Each bar is " + fnum(step, 0) + " nm wide. Hover a bar for its count.")}</span>'
            f'</figcaption>{"".join(parts)}{table}</figure>')


def hist_charts(samples: dict) -> str:
    x = shared_xmax(samples)      # one x-axis for all samples; the y-axis stays per sample
    return "".join(hist_chart(i, k, s, x) for i, (k, s) in enumerate(samples.items()) if hist_ok(s))


def short(label, n=28) -> str:
    """A chart label that fits a phone row (SVG text does not wrap); the row's tooltip has the full label."""
    s = str(label)
    return s if len(s) <= n else s[:n - 1] + "…"


def chart_summary(rows) -> str:
    bits = [f"{k} median {float(s['diam_nm_median']):.0f} nm (middle half {float(s['diam_nm_q1']):.0f} to "
            f"{float(s['diam_nm_q3']):.0f})" for k, s in rows]
    return "EV diameter per sample: " + "; ".join(bits)


# ---------------------------------------------------------------- page parts
def honesty(res: dict, ps_id: str, spot_n, has_pkg: bool) -> str:
    imgs = [i for i in res["images"].values() if not i.get("rejected")]      # rejected images are not counted
    n = len(imgs)
    n_rev = sum(1 for i in imgs if i.get("reviewed"))
    n_edit = sum(1 for i in imgs if i.get("edited") and not i.get("reviewed"))
    links = [f'<a class="cj-results-btn cj-results-btn-s" href="/predict/{esc(ps_id)}/">'
             + esc(f"Spot-check the {spot_n} suggested images" if spot_n else "Spot-check a suggested subset")
             + "</a>"]
    if has_pkg:
        links.append(f'<a class="cj-results-btn cj-results-btn-s" href="/review/pred_{esc(ps_id)}/">'
                     f'{esc("Review these images")}</a>')
    if n == 0:
        return ""
    edits = (f"{plural(n_edit, 'image')} {'has' if n_edit == 1 else 'have'} edits but "
             f"{'was' if n_edit == 1 else 'were'} not marked reviewed: the edits are counted, "
             "the image counts as not reviewed.") if n_edit else ""
    if n_rev == 0 and n_edit:
        head, tone = f"Not reviewed; {plural(n_edit, 'image')} edited (edits are counted).", "warn"
        text = "No image was marked reviewed, so there is no reviewer column; the rest are model only."
    elif n_rev == 0:
        head, tone = "Model only: none of these images were checked by a person.", "warn"
        text = "The numbers come straight from the model. Checking a few images makes them more trustworthy."
    elif n_rev < n:
        head, tone = f"{n_rev} of {n} images reviewed.", "part"
        text = (f"The other {n - n_rev} are not reviewed. " + edits) if n_edit else \
            f"The other {n - n_rev} are model only."
    else:
        head, tone, text, links = f"All {n} images reviewed by a person.", "ok", "", links[1:]
    return (f'<div class="cj-results-banner cj-results-b-{tone}" role="status" id="cj-results-honesty">'
            f'<p><b>{esc(head)}</b> {esc(text.strip())}</p>'
            f'<span class="cj-results-links">{"".join(links)}</span></div>')


def refused_notice(res: dict) -> str:
    why = res.get("refused")
    if not why:
        return ""
    out_of = {}
    for img in res["images"].values():
        lab = img.get("sample_label")
        if lab not in res["samples"] and not img.get("rejected"):      # rejected images are not counted anywhere
            out_of[lab] = out_of.get(lab, 0) + 1
    ru = res.get("refused_unsure") or {}
    items = "".join(f"<li><b>{esc(str(k))}</b> ({esc(plural(v, 'image'))}"
                    + (f", {esc(plural(ru[k], 'unsure outline'))} still open"
                       if isinstance(ru.get(k), int) and ru[k] else "") + ")</li>" for k, v in out_of.items())
    head = ("No sample could be counted." if not res["samples"] else "Some samples were not counted.")
    if not res["samples"] and not out_of:
        head = "No results for this set."
    return (f'<div class="cj-results-banner cj-results-b-warn" role="alert" id="cj-results-refused">'
            f'<p><b>{esc(head)}</b></p>' + (f'<ul class="cj-results-ul">{items}</ul>' if items else "")
            + f'<p class="cj-results-why">{esc("Why: " + str(why))}</p></div>')


def has_unsure_fields(res: dict) -> bool:
    """False for a results.json written before n_unsure_open / n_checked existed (any sample without them)."""
    return all("n_unsure_open" in s and "n_checked" in s for s in res["samples"].values())


def unsure_total(res: dict) -> int:
    """Open unsure outlines over the counted samples and the refused ones (refused_unsure: still work to do)."""
    rows = list(res["samples"].values()) + [{"n_unsure_open": v} for v in (res.get("refused_unsure") or {}).values()]
    return sum(int(s["n_unsure_open"]) for s in rows if isinstance(s.get("n_unsure_open"), int))


def checked_cell(s: dict, n_img: int) -> str:
    if "n_checked" not in s:
        return td("Checked", "–", k="n_checked")
    return td("Checked", f"{int(s['n_checked'])} of {n_img}", k="n_checked", sub="images edited or reviewed")


def unsure_cell(s: dict) -> str:
    if "n_unsure_open" not in s:
        return td("Unsure, not counted", "–", k="n_unsure_open", sub="Update to see unsure counts")
    if s["n_unsure_open"] is None:
        return td("Unsure, not counted", "–", k="n_unsure_open", sub="unknown: update after the review is rebuilt")
    return td("Unsure, not counted", fint(s["n_unsure_open"]), k="n_unsure_open", sub="open outlines, not in EVs")


def counted_note(res: dict) -> str:
    """The header line under the title: what the EV numbers include and exclude."""
    conf = (res.get("rule") or {}).get("conf", 0.5)
    try:
        pct = f"{float(conf) * 100:g}"
    except (TypeError, ValueError):
        pct = "50"
    text = (f"EV outlines the model is sure about (≥{pct}%) plus everything kept or drawn in review. "
            "Unsure outlines are not counted until you accept them.")
    return f'<p class="cj-results-counted" id="cj-results-counted"><b>Counted:</b> {esc(text)}</p>'


def n_rejected_total(res: dict) -> int:
    return sum(1 for i in res["images"].values() if i.get("rejected"))


def counts_line(res: dict) -> str:
    """'Counted EVs 120 (8 low conf) · 3 images rejected' (counted samples only; parts that are 0 are left out)."""
    rows = [s for s in res["samples"].values() if not s.get("all_rejected")]
    if not rows:
        return ""
    ev = sum(int(s.get("ev_total") or 0) for s in rows)
    low = sum(int(s.get("n_low_conf") or 0) for s in rows)
    rej = n_rejected_total(res)
    text = (f"Counted EVs {ev}" + (f" ({low} low conf)" if low else "")
            + (f" · {plural(rej, 'image')} rejected" if rej else ""))
    return f'<p class="cj-results-counts" id="cj-results-counts"><b>{esc(text)}</b></p>'


def all_rejected_notice(res: dict) -> str:
    names = [k for k, s in res["samples"].items() if s.get("all_rejected")]
    if not names:
        return ""
    items = "".join(f"<li><b>{esc(str(k))}</b> ({esc(plural(int(res['samples'][k].get('n_rejected') or 0), 'image'))}"
                    " rejected)</li>" for k in names)
    return ('<div class="cj-results-banner cj-results-b-warn" role="status" id="cj-results-allrej">'
            f'<p><b>{esc("All images rejected.")}</b> {esc("Nothing is counted for:")}</p>'
            f'<ul class="cj-results-ul">{items}</ul></div>')


def unsure_banner(cfg, res: dict, ps: dict, st: dict) -> str:
    """The nudge (open unsure outlines) or, for an old results.json, the Update prompt. '' when neither applies."""
    ps_id = ps["id"]
    if res["samples"] and not has_unsure_fields(res):
        busy = st["state"] in ("queued", "running")
        return ('<div class="cj-results-banner cj-results-b-part" role="status" id="cj-results-oldfields">'
                f'<p><b>{esc("Update to see unsure counts.")}</b> '
                f'{esc("These results were computed before unsure outlines were counted.")}</p>'
                + ("" if busy else compute_form(ps_id, "Update results now")) + "</div>")
    n = unsure_total(res)
    if n <= 0:
        return ""
    imgs = list(res["images"].values())
    has_pkg = package_dir(cfg, ps_id) is not None
    # the honesty banner already carries "Review these images" while some image is not Reviewed: no second button
    already = bool(imgs) and not all(i.get("reviewed") for i in imgs) and has_pkg
    link = ("" if already or not has_pkg else
            f'<a class="cj-results-btn cj-results-btn-s" href="/review/pred_{esc(ps_id)}/">'
            f'{esc("Review these images")}</a>')
    return ('<div class="cj-results-banner cj-results-b-part" role="status" id="cj-results-nudge">'
            f'<p><b>{esc(plural(n, "unsure outline") + " not checked yet.")}</b> '
            f'{esc("They are not in the EV counts until you accept them.")}</p>{link}</div>')


def sample_table(res: dict) -> str:
    samples = res["samples"]
    if not samples:
        return ""
    reviewer = any(int(s.get("n_reviewed") or 0) > 0 for s in samples.values())
    head = ["Sample", "Images", "Reviewed", "Checked", "EVs", "Unsure, not counted", "EVs per image", "Median diameter (Q1–Q3)",
            "Multilayer (geometry)"]
    if reviewer:
        head += ["Multilayer (reviewer)", "Reviewer vs geometry"]
    head += ["Non-EV"]
    rows = []
    for label, s in samples.items():
        dv, dsub = fdiam(s)
        n_img, n_rev = int(s.get("n_images") or 0), int(s.get("n_reviewed") or 0)
        n_rej, n_low = int(s.get("n_rejected") or 0), int(s.get("n_low_conf") or 0)
        gone = bool(s.get("all_rejected"))
        cells = [f'<th scope="row" data-label="Sample" class="cj-results-sname">{esc(str(label))}</th>',
                 td("Images", fint(n_img), k="n_images",
                    sub="All images rejected" if gone else (f"{n_rej} rejected" if n_rej else None)),
                 td("Reviewed", f"{n_rev} / {n_img}", k="n_reviewed"),
                 checked_cell(s, n_img),
                 td("EVs", fint(s.get("ev_total")), k="ev_total",
                    sub=None if gone else "counted" + (f" · {n_low} low conf" if n_low else "")),
                 unsure_cell(s),
                 td("EVs per image", fnum(s.get("ev_per_image_mean")), k="ev_per_image_mean"),
                 td("Median diameter (Q1–Q3)", dv, k="diam_nm_median", sub=dsub),
                 td("Multilayer (geometry)", fpct(s.get("pct_multilayer_geometry")), "cj-results-hl",
                    k="pct_multilayer_geometry")]
        if reviewer:
            if n_rev:
                rev_ev = int(s.get("reviewer_ev") or 0)
                base = (f"n = {plural(rev_ev, 'EV')} on "
                        f"{plural(int(s.get('reviewed_images') or n_rev), 'reviewed image')}")
                dis = int(s.get("reviewer_geometry_disagree") or 0)
                cells.append(td("Multilayer (reviewer)", fpct(s.get("pct_multilayer_reviewer")),
                                k="pct_multilayer_reviewer", sub=base,
                                flag="small sample" if rev_ev < SMALL_BASE else None))
                cells.append(td("Reviewer vs geometry", fint(dis), k="reviewer_geometry_disagree",
                                sub="disagreement" if dis == 1 else "disagreements"))
            else:
                cells.append(td("Multilayer (reviewer)", "–", k="pct_multilayer_reviewer", sub="none reviewed"))
                cells.append(td("Reviewer vs geometry", "–", k="reviewer_geometry_disagree"))
        cells.append(td("Non-EV", fint(s.get("nonev_total")), k="nonev_total"))
        rows.append(f'<tr data-sample="{esc(str(label))}">{"".join(cells)}</tr>')
    th = "".join(f'<th scope="col">{esc(h)}</th>' for h in head)
    notes = ["Multilayer (geometry): the share of EVs whose outline contains another outline. This is the main "
             "number; it does not depend on anyone's class labels.",
             "Inner layers of a multilayer EV are not counted as separate EVs. Non-EV: other objects the model "
             "found, counted separately."]
    varied = [f"{k} ({lo:g}–{hi:g} nm/px)" for k, (lo, hi) in nm_ranges(res).items() if lo != hi]
    if varied:
        notes.append(f"Pixel size differs slightly between the images of {', '.join(varied)}, within "
                     f"{NM_TOLERANCE * 100:g} %: counted as one sample; each image's diameters use its own value.")
    if reviewer:
        notes.append("Multilayer (reviewer): the share of EVs a reviewer called multilayer, on reviewed images "
                     f"only (n shows the base; under {SMALL_BASE} EVs it is marked small sample). Reviewer vs "
                     "geometry: reviewed outlines where the reviewer's call differs from the geometry.")
    return (f'<div class="cj-results-wrap"><table class="cj-results-tab cj-results-stack" id="cj-results-samples">'
            f'<thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            + "".join(f'<p class="cj-mut cj-results-note">{esc(n)}</p>' for n in notes))


def image_table(res: dict) -> str:
    imgs = res["images"]
    if not imgs:
        return ""
    head = ["Image", "Sample", "Checked", "EVs", "Multi­layer", "Inner", "Non-EV"]   # soft hyphen: phone
    rows = []
    for k, i in imgs.items():
        counted = i.get("sample_label") in res["samples"]
        cls = "" if counted and not i.get("rejected") else ' class="cj-results-off"'
        rows.append(f'<tr data-image="{esc(str(k))}"{cls}><td class="cj-num">{esc(str(k))}</td>'
                    f'<td class="cj-results-l">{esc(str(i.get("sample_label")))}</td><td>{image_chip(i, counted)}</td>'
                    f'<td>{esc(fint(i.get("ev")))}</td><td>{esc(fint(i.get("multilayer")))}</td>'
                    f'<td>{esc(fint(i.get("inner")))}</td><td>{esc(fint(i.get("nonev")))}</td></tr>')
    th = "".join(f'<th scope="col">{esc(h)}</th>' for h in head)
    return (f'<details class="cj-results-det" id="cj-results-images"><summary>{esc(f"Per image ({len(imgs)})")}'
            f'</summary><div class="cj-results-wrap"><table class="cj-results-tab cj-results-img">'
            f'<thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></details>')


def nm_ranges(res: dict) -> dict:
    """{counted sample: (min, max) nm/px used}: the sample's nm_per_px_min / _max, else (results written before
    those fields) its images' own values."""
    out = {}
    for label, s in res["samples"].items():
        lo, hi = s.get("nm_per_px_min"), s.get("nm_per_px_max")
        if s.get("all_rejected"):
            continue
        if lo is None or hi is None:
            vals = [float(i["nm_per_px"]) for i in res["images"].values()
                    if i.get("sample_label") == label and i.get("nm_per_px") is not None
                    and not i.get("rejected")]
            if not vals:
                continue
            lo, hi = min(vals), max(vals)
        out[label] = (float(lo), float(hi))
    return out


def nm_text(res: dict) -> str:
    """'S1 1.4724; S2 1.47125–1.4724 (...)' for the provenance; '' when no sample was counted."""
    rng = nm_ranges(res)
    parts = [f"{k} {lo:g}" if lo == hi else f"{k} {lo:g}–{hi:g}" for k, (lo, hi) in rng.items()]
    if not parts:
        return ""
    varied = any(lo != hi for lo, hi in rng.values())
    return "; ".join(parts) + (f" (varies within {NM_TOLERANCE * 100:g} % between images of a sample; each "
                               "image's diameters use its own value)" if varied else "")


def provenance(res: dict, ps: dict) -> dict:
    imgs = [i for i in res["images"].values() if not i.get("rejected")]      # counted images only
    rule = res.get("rule") or {}
    nm = nm_text(res)
    return {"Prediction set": ps["id"],
            "Model": str(res.get("model")),
            "Computed": when(res.get("computed")),
            "Counting rule": (f"outlines at confidence {rule.get('conf')} or more (on images nobody edited); "
                              f"{ev_rule(rule.get('ev_classes'))}; inner layers not counted; "
                              f"multilayer = {rule.get('multilayer')}"),
            "Checked by a person": (f"{sum(1 for i in imgs if i.get('reviewed'))} of {len(imgs)} "
                                    f"{'counted ' if len(imgs) != len(res['images']) else ''}images reviewed, {sum(1 for i in imgs if i.get('edited') and not i.get('reviewed'))}"
                                    " more edited"),
            **({"Pixel size (nm per pixel)": nm} if nm else {}),
            **({"Rejected images (not counted)": f"{n_rejected_total(res)} of {len(res['images'])} images"}
               if n_rejected_total(res) else {}),
            "Not counted": str(res.get("refused") or "nothing")}


def downloads(ps_id: str) -> str:
    base = f"/results/{esc(ps_id)}/download/"
    items = [("per_sample.csv", "Per sample (CSV)"), ("per_image.csv", "Per image (CSV)"),
             ("results.json", "All numbers (JSON)")]
    return (f'<p class="cj-results-dlmain"><a class="cj-results-btn" href="{base}{esc(f"results_{ps_id}.zip")}" '
            f'download>{esc("Results tables (zip)")}</a> <span class="cj-mut">'
            f'{esc("The tables, all numbers and a README of how they were made.")}</span></p>'
            '<ul class="cj-results-dl">' + "".join(
                f'<li><a href="{base}{esc(f)}" download>{esc(t)}</a></li>' for f, t in items) + "</ul>"
            f'<p class="cj-mut cj-results-note">'
            f'{esc(CSV_NOTE)}</p>')


CSV_NOTE = "CSV files start with # lines describing how they were made (pandas: comment='#')."


INCOMPLETE = "Results are being written or were interrupted: compute again."


def incomplete(ps: dict) -> bool:
    """A CSV newer than results.json (which is written last) = a write in progress or one that stopped halfway."""
    rdir = ps["dir"] / "results"
    try:
        done = (rdir / "results.json").stat().st_mtime
        return any((rdir / f).is_file() and (rdir / f).stat().st_mtime > done
                   for f in FILES if f.endswith(".csv"))
    except OSError:
        return True


def csv_header(res: dict, ps: dict) -> bytes:
    """The provenance comment lines put on top of a standalone CSV download (the file on disk is unchanged)."""
    imgs = [i for i in res["images"].values() if not i.get("rejected")]      # counted images only
    rule = res.get("rule") or {}
    lines = [f"# CryoEV results for prediction set {ps['id']}",
             f"# model={res.get('model')}, computed={res.get('computed')}, "
             f"reviewed={sum(1 for i in imgs if i.get('reviewed'))}/{len(imgs)} {'counted ' if len(imgs) != len(res['images']) else ''}images, "
             f"edited (not reviewed)={sum(1 for i in imgs if i.get('edited') and not i.get('reviewed'))}",
             f"# rule=conf>={rule.get('conf')}; {ev_rule(rule.get('ev_classes'))}; inner layers not counted; "
             f"multilayer={rule.get('multilayer')}"]
    if nm_text(res):
        lines.append(f"# nm_per_px={nm_text(res)}")
    if n_rejected_total(res):
        lines.append(f"# rejected images (not counted)={n_rejected_total(res)}")
    lines.append(f"# not counted={res.get('refused') or 'nothing'}")
    return ("\n".join(x.replace("\r", " ").replace("\n", " ") for x in lines) + "\n").encode("utf-8")


def compute_form(ps_id: str, label: str, primary=True) -> str:
    cls = "cj-results-btn" + ("" if primary else " cj-results-btn-s")
    return (f'<form class="cj-results-form" method="post" action="/results/{esc(ps_id)}/compute">'
            f'<button type="submit" class="{cls}">{esc(label)}</button></form>')


def spot_count(cfg, ps_id: str):
    root = review_root(cfg)
    doc = _read_json(root / f"pred_{ps_id}" / "spotcheck.json") if root else None
    n = doc.get("n") if isinstance(doc, dict) else None
    return n if isinstance(n, int) and n > 0 else None


def render_set(cfg, ps: dict, user: dict) -> str:
    ps_id, admin = ps["id"], user.get("role") == "admin"
    res, problem = read_results(ps)
    st, job = status_doc(cfg, ps), job_state(ps)
    req = ps["request"]
    out = [f'<p class="cj-mut cj-results-crumb"><a href="/results/">Results</a> / '
           f'<span class="cj-wrap">{esc(ps_id)}</span></p>',
           f'<h1 class="cj-wrap">{esc(NM.display_name(req))}</h1>',
           f'<p class="cj-mut cj-results-psid">{esc(ps_id)}</p>',
           f'<p class="cj-mut">{esc("Model " + str(req.get("model")))}'
           f'{esc(" · started " + when(req.get("created")) if req.get("created") else "")}</p>']

    if st["state"] in ("queued", "running"):
        msg = ("Waiting for another computation to finish, then this one starts." if st["state"] == "queued"
               else "Computing the results. Big sets take about a minute.")
        out.append(f'<div class="cj-results-banner cj-results-b-part" role="status" id="cj-results-job" '
                   f'data-cj-results-poll="/results/{esc(ps_id)}/api/status">'
                   f'<p><b>{esc(msg)}</b> {esc("This page updates by itself.")}</p>'
                   f'<p class="cj-mut" id="cj-results-pollerr" hidden></p>'
                   f'<noscript><p>{esc("Reload this page in a minute.")}</p></noscript></div>')
    elif job and job["state"] == "error":
        earlier = (f'<p>{esc("The results below are from the earlier computation (computed " + when(res.get("computed")) + ").")}</p>'
                   if res is not None else "")
        out.append(f'<div class="cj-results-banner cj-results-b-warn" role="alert" id="cj-results-job">'
                   f'<p><b>{esc(job["error"])}</b> {esc("Try again; if it fails again, tell the admin.")}</p>'
                   + earlier
                   + (f'<p class="cj-mut cj-wrap">{esc(job["detail"])}</p>' if admin and job.get("detail") else "")
                   + "</div>")
    stale = stale_info(cfg, ps) if res is not None else None
    if stale:
        if stale["kind"] == "removed":
            head = "These results count review edits that have since been removed."
        elif stale["kind"] == "edit":
            head = f"These results are older than the latest review edits (last edit {hhmm(stale['at'])})."
        else:
            head = f"These results are older than the review package (built {hhmm(stale['at'])})."
        out.append(f'<div class="cj-results-banner cj-results-b-warn" role="alert" id="cj-results-stale">'
                   f'<p><b>{esc(head)}</b> {esc("Update the results to include the latest review.")}</p>'
                   + ("" if st["state"] in ("queued", "running") else compute_form(ps_id, "Update results now"))
                   + "</div>")
    if problem:
        out.append(f'<div class="cj-results-banner cj-results-b-warn" role="alert"><p>{esc(problem)}</p></div>')

    if res is None:
        busy = st["state"] in ("queued", "running")
        if not predictions_ready(ps):
            s = ps["status"] or {}
            state = s.get("state")
            if state == "failed":
                text = "The prediction run failed, so there is nothing to count."
            elif state in ("queued", "running"):
                text = (f"The model is still running ({s.get('n_done', 0)} of {s.get('n_total', '?')} images). "
                        "Results can be computed when it finishes.")
            else:
                text = "The predictions are not finished yet."
            out.append(f'<div class="cj-card cj-results-empty" id="cj-results-none"><p>{esc(text)}</p>'
                       f'<p><a href="/predict/{esc(ps_id)}/">{esc("See the prediction set")}</a></p></div>')
        elif not busy:
            out.append('<div class="cj-card cj-results-empty" id="cj-results-none">'
                       f'<p><b>{esc("Results are not computed yet.")}</b> '
                       f'{esc("Counting takes up to a minute on a big set; you can leave the page.")}</p>'
                       + compute_form(ps_id, "Compute results") + "</div>")
        if predictions_ready(ps):          # images and outlines do not need results.json
            out.append('<h2 class="cj-sec">Download</h2>')
            out.append(downloads_zips(ps_id, package_dir(cfg, ps_id) is not None, (ps["dir"] / "p").is_dir()))
        return "".join(out)

    out.append(counted_note(res))
    out.append(counts_line(res))
    out.append(all_rejected_notice(res))
    out.append(honesty(res, ps_id, spot_count(cfg, ps_id), package_dir(cfg, ps_id) is not None))
    out.append(unsure_banner(cfg, res, ps, st))
    out.append(refused_notice(res))
    if res["samples"]:
        out.append('<h2 class="cj-sec">Per sample</h2>')
        out.append(sample_table(res))
        out.append('<h2 class="cj-sec">EV size</h2>')
        out.append('<figure class="cj-card cj-results-fig"><figcaption>EV diameter by sample (nm)'
                   '<span class="cj-mut">Dot: median. Bar: the middle half of the EVs (Q1 to Q3).</span>'
                   f'</figcaption>{diam_chart(res["samples"])}</figure>')
        out.append(hist_charts(res["samples"]))
    if res["images"]:
        out.append('<h2 class="cj-sec">Images</h2>')
        out.append(image_table(res))
    out.append('<h2 class="cj-sec">Download</h2>')
    out.append(downloads_zips(ps_id, package_dir(cfg, ps_id) is not None, (ps["dir"] / "p").is_dir()))
    out.append(downloads(ps_id))
    out.append('<h2 class="cj-sec">How these numbers were made</h2>')
    out.append('<dl class="cj-card cj-results-prov" id="cj-results-prov">' + "".join(
        f"<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>" for k, v in provenance(res, ps).items()) + "</dl>")
    if st["state"] not in ("queued", "running"):
        out.append('<div class="cj-results-again"><p class="cj-mut">'
                   + esc("After more review, compute again to include it.") + "</p>"
                   + compute_form(ps_id, "Update results now", primary=False) + "</div>")
    return "".join(out)


def list_sets(cfg, user: dict, can_see) -> list:
    root = predict_root(cfg)
    out = []
    if root is None or not root.is_dir():
        return out
    for d in root.iterdir():
        if d.name.startswith(".") or not PS_RE.match(d.name) or not d.is_dir():
            continue
        ps = load_set(cfg, d.name, user, can_see)
        if ps:
            out.append(ps)
    out.sort(key=lambda p: (str(p["request"].get("created") or ""), p["id"]), reverse=True)
    return out


def render_index(cfg, user: dict, can_see) -> str:
    sets = list_sets(cfg, user, can_see)
    out = ['<h1>Results</h1>', f'<p class="cj-mut">{esc("Counts, sizes and multilayer % for each analysis.")}</p>']
    if not sets:
        out.append(f'<p class="cj-card cj-results-empty">{esc("No analyses yet. Run a model in Predict first.")}</p>')
        return "".join(out)
    items = []
    for ps in sets:
        st = status_doc(cfg, ps)
        s = ps["status"] or {}
        n = s.get("n_total")
        word = {"ready": ("Results ready", "ok"), "none": ("Not computed yet", "model"),
                "queued": ("Computing", "edit"), "running": ("Computing", "edit"),
                "error": ("Computing failed", "warn"),
                "waiting": ("Model failed" if s.get("state") == "failed" else "Model still running", "model")}
        label, tone = word.get(st["state"], ("?", "model"))
        if st["state"] == "ready" and st["out_of_date"]:
            label, tone = "Out of date", "warn"
        meta = " · ".join(v for v in (when(ps["request"].get("created")) if ps["request"].get("created") else None,
                                      plural(int(n), "image") if isinstance(n, int) else None,
                                      f"model {ps['request'].get('model')}") if v)
        # a title link + meta (+ an Update form when out of date): a form is not allowed inside a link
        update = (compute_form(ps["id"], "Update", primary=False)
                  if st["state"] == "ready" and st["out_of_date"] else "")
        items.append(f'<li class="cj-card cj-results-set">'
                     f'<a class="cj-results-set-t" href="/results/{esc(ps["id"])}/">'
                     f'<b class="cj-wrap">{esc(NM.display_name(ps["request"]))}</b>{chip(label, tone)}</a>'
                     f'<span class="cj-mut cj-results-psid cj-wrap">{esc(ps["id"])}</span>'
                     f'<span class="cj-mut">{esc(meta)}</span>{update}</li>')
    out.append(f'<ul class="cj-results-sets">{"".join(items)}</ul>')
    return "".join(out)


_PH = re.compile(r"\{\{content\}\}")


def page_body(content: str) -> str:
    return _PH.sub(lambda m: content, TEMPLATE.read_text(encoding="utf-8"), count=1)


def results_doc_lines(prefix: str = "") -> list:
    """The Files + Columns text of the results tables (README of the results zip and of the Everything zip)."""
    p = prefix
    return [
        "Files",
        f"  {p}per_sample.csv  one row per sample that could be counted (a sample the engine refused is left out; see",
        "                  'Not counted'). A sample whose images were ALL rejected is still a row: n_images 0,",
        "                  n_rejected = how many, every other column blank.",
        "                  The CSVs in this zip have no # header lines (this README is their provenance).",
        f"  {p}per_image.csv   one row per image; reviewed / edited say whether a person checked it (reviewed = true),",
        "                  changed it without marking it reviewed (edited = true), or neither (model only).",
        "                  rejected / reject_reason / note: a rejected image stays in this file but is not counted.",
        f"  {p}results.json    everything above, schema results/1", "",
        "Columns",
        "  n_images                  images COUNTED in the sample (rejected images are not in this number)",
        "  n_rejected                images of the sample a reviewer rejected (thick ice, empty, ...); they are left out",
        "                            of every other number, and of the refusal check on nm/px and tile size",
        "  n_low_conf                counted EVs a reviewer tagged 'low confidence' (they ARE counted as EVs)",
        "  ev_total                  EVs (outlines of the EV classes that are not inner layers)",
        "  pct_multilayer_geometry   % of EVs whose outline contains another outline (the main number)",
        "  pct_multilayer_reviewer   % of EVs a reviewer called multilayer, reviewed images only (base: reviewer_ev)",
        "  diam_nm_median / q1 / q3  area-equivalent diameter in nm (each image converted with its own nm/px)",
        "  nm_per_px_min / max       the pixel sizes used in the sample (they may differ by up to "
        f"{NM_TOLERANCE * 100:g} %)",
        "  inner_layers_excluded     inner outlines of multilayer EVs, not counted as EVs",
        "  nonev_total               other objects, counted separately",
        "  reviewer_geometry_disagree  reviewed outlines where the reviewer's call differs from the geometry",
        "  rejected (per_image)      true when the image was rejected; reject_reason says why",
        "  note (per_image)          the reviewer's free-text note on the image (text starting with = + - @ gets a ' in",
        "                            front so a spreadsheet does not run it); also says when a rejected image could not",
        "                            be read",
        ""]


def readme(res: dict, ps: dict, user: dict) -> str:
    lines = [f"CryoEV results for prediction set {ps['id']}", ""]
    lines += [f"{k}: {v}" for k, v in provenance(res, ps).items()]
    lines += [f"Downloaded: {datetime.now().astimezone().isoformat(timespec='seconds')} by {user.get('id')}", ""]
    lines += results_doc_lines()
    return "\r\n".join(lines)


def attachment(body: bytes, name: str, media: str) -> Response:
    r = Response(body, media_type=media)
    r.headers["Content-Disposition"] = f'attachment; filename="{name}"'
    r.headers["X-Content-Type-Options"] = "nosniff"
    r.headers["Cache-Control"] = "no-store"
    return r


# ---------------------------------------------------------------- round 7: downloads (streamed zips)
# Images / Model outlines / Reviewed labels / Everything. Built and streamed with portal.zipstream.stream_zip (no
# whole zip in memory, no temp file). Images and Model outlines do not need results.json; Reviewed labels are built
# at download time from reviewed_source() (the review package's current edits, never a stale reviewed.json).
CAND_CONF = 0.25             # = results.compute.CAND_CONF: the lowest model confidence the outlines download keeps
ZIP_KINDS = {"images": "images", "outlines": "model_outlines", "reviewed": "reviewed_labels",
             "everything": "everything"}
LOW_CONF_NAME = "Low conf EV"
COCO_FILE = {"outlines": "annotations_model.coco.json", "reviewed": "annotations_reviewed.coco.json"}
LABEL_DIR = {"outlines": "labels_model", "reviewed": "labels_reviewed"}


def guard_cell(text: str) -> str:
    """CSV formula guard (the same rule as results.compute.guard_cell; a test pins both)."""
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def csv_text(header: list, rows) -> str:
    import csv as _csv
    buf = io.StringIO()
    w = _csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for row in rows:
        w.writerow(["" if v is None else ("true" if v is True else "false" if v is False else
                                           guard_cell(v) if isinstance(v, str) else v) for v in row])
    return buf.getvalue()


def zip_kind(name: str, ps_id: str):
    for kind, stem in ZIP_KINDS.items():
        if name == f"{stem}_{ps_id}.zip":
            return kind
    return None


def lp(path):
    """A long-path-safe Path (data_utils.longpath.long_path: \\\\?\\ prefix over the Windows limit), as the counting
    chain does; the plain path where that module is not available."""
    try:
        from data_utils.longpath import long_path
    except ImportError:
        return Path(path)
    return Path(long_path(Path(path)))


def images_csv_bytes(ps: dict) -> bytes | None:
    """images.csv for the zips, every text cell through the CSV formula guard (orig_name is an uploaded file name).
    None when it can not be read."""
    import csv as _csv
    try:
        with open(lp(ps["dir"] / "images.csv"), newline="", encoding="utf-8") as fh:
            rd = _csv.reader(fh)
            rows = list(rd)
    except (OSError, ValueError, _csv.Error):
        return None
    if not rows:
        return None
    return csv_text(rows[0], rows[1:]).encode("utf-8")


def set_rows(ps: dict) -> list:
    """The set's images from images.csv: [{image, sample_label, path (Path | None), arc, problem}]. A file that is
    missing, or resolves outside the set folder, has path None and a problem text (listed, never fatal)."""
    import csv as _csv
    try:
        with open(lp(ps["dir"] / "images.csv"), newline="", encoding="utf-8") as fh:
            raw = list(_csv.DictReader(fh))
    except (OSError, ValueError, _csv.Error):
        return []
    out, root = [], ps["dir"].resolve()
    for r in sorted(raw, key=lambda r: r.get("image") or ""):
        image, rel = (r.get("image") or "").strip(), (r.get("file") or "").strip()
        if not image:
            continue
        row = {"image": image, "sample_label": (r.get("sample_label") or "").strip(), "path": None,
               "arc": None, "problem": None}
        try:
            p = (ps["dir"] / rel).resolve()
            if root not in p.parents:
                row["problem"] = "file is outside the analysis folder"
            elif not lp(p).is_file():
                row["problem"] = "file is missing"
            else:
                row["path"] = p
                row["arc"] = zipstream_mod().safe_arcname("images/" + p.name)
        except (OSError, ValueError):
            row["problem"] = "file can not be read"
        out.append(row)
    return out


def zipstream_mod():
    from portal import zipstream
    return zipstream


def image_size(path: Path):
    from PIL import Image
    try:
        from data_utils.longpath import long_path
        path = long_path(path)
    except ImportError:
        pass
    with Image.open(path) as im:
        return im.size


def class_names(model, objs_cls=()):
    """({class id: name}, EV class ids | None) for COCO categories. Names come from the model registry (lazy: the
    portal starts without it); a class the registry does not name gets EV / non-EV / 'class k'."""
    names, ev_ids = {}, None
    try:
        from inference.model_classes import ev_class_ids, model_classes
        names, ev_ids = model_classes(model), ev_class_ids(model)
    except Exception:  # noqa: BLE001 - no registry here: fall back to generic names
        pass
    for k in list(objs_cls) + [0]:
        names.setdefault(k, {0: "EV", 1: "non-EV"}.get(k, f"class {k}"))
    return names, ev_ids


def coco_categories(names: dict) -> tuple:
    """([categories], {class id: category id}, low conf EV category id): category id = class id + 1; 'Low conf EV'
    takes the next free id."""
    cats = [{"id": k + 1, "name": names[k]} for k in sorted(names)]
    low = max(c["id"] for c in cats) + 1
    cats.append({"id": low, "name": LOW_CONF_NAME})
    return cats, {k: k + 1 for k in names}, low


def poly_area(pts) -> float:
    a = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def coco_annotation(aid, image_id, cat, pts, score=None, attrs=None) -> dict:
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    a = {"id": aid, "image_id": image_id, "category_id": cat,
         "segmentation": [[round(c, 2) for p in pts for c in p]],
         "bbox": [round(min(xs), 2), round(min(ys), 2), round(max(xs) - min(xs), 2), round(max(ys) - min(ys), 2)],
         "area": round(poly_area(pts), 2), "iscrowd": 0}
    if score is not None:
        a["score"] = round(float(score), 4)
    if attrs:
        a["attributes"] = attrs
    return a


def reviewed_annotation(aid, image_id, o: dict, names, ev_ids, cat_of, low_id) -> dict | None:
    """ONE COCO RULE (BUILD_CONTRACTS.md section 14): an EV with low conf -> category 'Low conf EV'; any other class
    keeps its class and gets attributes.low_conf = true; a note goes in attributes.note. None for a polygon with
    fewer than 3 points."""
    pts = [(float(p[0]), float(p[1])) for p in (o.get("poly") or [])]
    if len(pts) < 3:
        return None
    cls = int(o.get("cls") or 0)
    label = o.get("reviewer_label")
    # the FINAL class decides: the reviewer's label when there is one, else the model's class. Only a final class
    # of EV (or "Low conf EV") with low conf becomes "Low conf EV"; MV, inner, Incomplete, non-EV, ... keep their
    # class and carry attributes.low_conf
    ev_names = {"EV", LOW_CONF_NAME} | {names[k] for k in names if ev_ids is None or k in ev_ids}
    if label:
        is_ev = str(label) in ev_names
    else:
        is_ev = (ev_ids is None or cls in ev_ids) and o.get("reviewer_class") != "nonEV"
    low = bool(o.get("low_conf"))
    attrs = {}
    cat = cat_of[cls]
    if label == LOW_CONF_NAME or (low and is_ev):
        cat = low_id
    elif low:
        attrs["low_conf"] = True
    if o.get("note"):
        attrs["note"] = str(o["note"])
    if o.get("reviewer_label"):
        attrs["reviewer_label"] = str(o["reviewer_label"])
    if o.get("source"):
        attrs["source"] = str(o["source"])
    return coco_annotation(aid, image_id, cat, pts, o.get("conf"), attrs)


def coco_stream(info: dict, images: list, cats: list, annots):
    """The COCO json as byte pieces (annotations one by one: the whole file is never built in memory)."""
    head = json.dumps({"info": info, "categories": cats, "images": images}, separators=(",", ":"))
    yield (head[:-1] + ',"annotations":[').encode("utf-8")
    first = True
    for a in annots:
        yield (("" if first else ",") + json.dumps(a, separators=(",", ":"))).encode("utf-8")
        first = False
    yield b"]}"


def yolo_line(cls, pts, w, h) -> str:
    return f"{cls} " + " ".join(f"{p[0] / w:.6f} {p[1] / h:.6f}" for p in pts)


def prediction_lines(pred_dir: Path, image: str):
    """Iterator of (cls, conf, [coords...]) from the model's prediction .txt (normalised coordinates), read lazily
    one line at a time; None when the file is absent."""
    p = lp(pred_dir / f"{image}.txt")
    if not p.is_file():
        return None

    def gen():
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                t = line.split()
                if len(t) >= 8:
                    yield int(t[0]), float(t[1]), [float(v) for v in t[2:]]
    return gen()


def outlines_entries(ps: dict, rows: list, problems: list) -> list:
    """Model outlines: labels_model/<image>.txt (YOLO-seg, no score) + annotations_model.coco.json (with scores)
    for predictions with conf >= CAND_CONF. The images folder is not in this zip; file_name = images/<name>.
    Prediction files are read again while streaming (nothing big is held in memory)."""
    model = ps["request"].get("model")
    pred_dir = ps["dir"] / "p" / str(model) / "predictions"
    sizes, ok, seen = {}, [], set()
    for r in rows:
        if r["path"] is None:
            continue
        try:
            sizes[r["image"]] = image_size(r["path"])
        except Exception as e:  # noqa: BLE001 - one bad image is listed, not fatal
            problems.append(f"{r['image']}: image can not be read ({type(e).__name__})")
            continue
        lines = prediction_lines(pred_dir, r["image"])
        if lines is None:
            problems.append(f"{r['image']}: no prediction file")
            continue
        seen.update(ln[0] for ln in lines)
        ok.append(r)
    names, _ = class_names(model, seen)
    cats, cat_of, _low = coco_categories(names)
    images = [{"id": n + 1, "file_name": r["arc"], "width": sizes[r["image"]][0], "height": sizes[r["image"]][1]}
              for n, r in enumerate(ok)]

    def kept(image):
        return (ln for ln in (prediction_lines(pred_dir, image) or ()) if ln[1] >= CAND_CONF)

    def annots():
        aid = 0
        for n, r in enumerate(ok):
            w, h = sizes[r["image"]]
            for cls, conf, c in kept(r["image"]):
                aid += 1
                pts = [(c[k] * w, c[k + 1] * h) for k in range(0, len(c) - 1, 2)]
                yield coco_annotation(aid, n + 1, cat_of[cls], pts, conf)

    def yolo(image):
        for cls, _conf, c in kept(image):
            yield (f"{cls} " + " ".join(f"{v:.6f}" for v in c) + "\n").encode("utf-8")

    info = {"description": f"Model outlines for {ps['id']} (model {model}), conf >= {CAND_CONF}",
            "pixel_coordinates": True}
    entries = [(COCO_FILE["outlines"], coco_stream(info, images, cats, annots()))]
    entries += [(f"{LABEL_DIR['outlines']}/{r['image']}.txt", yolo(r["image"])) for r in ok]
    return entries


def status_of(entry: dict | None) -> str:
    if not entry:
        return "model-only"
    if entry.get("rejected"):
        return "rejected"
    if entry.get("reviewed"):
        return "reviewed"
    return "edited" if entry.get("edited") else "model-only"


def reviewed_entries(ps: dict, rows: list, rev: dict, problems: list) -> list:
    """Reviewed labels from reviewed/1 (reviewed_source(): current edits): annotations_reviewed.coco.json (the one
    COCO rule) + labels_reviewed/<image>.txt (YOLO-seg, original class ids) + rejected_images.csv + provenance.csv.
    Rejected images are KEPT in the labels (a reject changes the counts, never the label data): their COCO image
    entry is marked rejected: true + reject_reason, and they are listed in rejected_images.csv and provenance.csv."""
    model = ps["request"].get("model")
    imgs = rev.get("images") or {}
    seen = {int(o.get("cls") or 0) for e in imgs.values() for o in (e.get("objects") or [])}
    names, ev_ids = class_names(model, seen)
    cats, cat_of, low_id = coco_categories(names)
    sizes, kept = {}, []
    for r in rows:
        e = imgs.get(r["image"])
        if e is None:
            continue
        if r["path"] is None:
            problems.append(f"{r['image']}: {r['problem']}; its labels are not in the reviewed labels")
            continue
        try:
            sizes[r["image"]] = image_size(r["path"])
        except Exception as e2:  # noqa: BLE001
            problems.append(f"{r['image']}: image can not be read ({type(e2).__name__})")
            continue
        kept.append(r)
    def coco_image(n, r):
        d = {"id": n + 1, "file_name": r["arc"], "width": sizes[r["image"]][0], "height": sizes[r["image"]][1]}
        if imgs[r["image"]].get("rejected"):          # a reject changes the counts only, never the label data
            d["rejected"] = True
            d["reject_reason"] = imgs[r["image"]].get("reject_reason")
        return d

    images = [coco_image(n, r) for n, r in enumerate(kept)]

    def annots():
        aid = 0
        for n, r in enumerate(kept):
            for o in imgs[r["image"]].get("objects") or []:
                a = reviewed_annotation(aid + 1, n + 1, o, names, ev_ids, cat_of, low_id)
                if a is None:
                    problems.append(f"{r['image']}: an outline with fewer than 3 points was left out")
                    continue
                aid += 1
                yield a

    info = {"description": f"Reviewed labels for {ps['id']} (model {model}), built {datetime.now().astimezone().isoformat(timespec='seconds')}",
            "pixel_coordinates": True, "rule": f"EV + low conf -> '{LOW_CONF_NAME}'; other classes keep their "
            "class with attributes.low_conf; notes in attributes.note"}
    entries = [(COCO_FILE["reviewed"], coco_stream(info, images, cats, annots()))]
    for r in kept:
        w, h = sizes[r["image"]]
        lines = []
        for o in imgs[r["image"]].get("objects") or []:
            pts = [(float(p[0]), float(p[1])) for p in (o.get("poly") or [])]
            if len(pts) >= 3:
                lines.append(yolo_line(int(o.get("cls") or 0), pts, w, h) + "\n")
        entries.append((f"{LABEL_DIR['reviewed']}/{r['image']}.txt", "".join(lines).encode("utf-8")))
    rej = [[r["image"], r["sample_label"], (imgs[r["image"]] or {}).get("reject_reason"),
            (imgs[r["image"]] or {}).get("note")]
           for r in rows if (imgs.get(r["image"]) or {}).get("rejected")]
    entries.append(("rejected_images.csv", csv_text(["image", "sample_label", "reject_reason", "note"], rej).encode("utf-8")))
    prov = []
    for r in rows:
        e = imgs.get(r["image"])
        objs = (e or {}).get("objects") or []
        prov.append([r["image"], r["sample_label"], status_of(e), len(objs),
                     sum(1 for o in objs if o.get("low_conf")), ";".join(str(x) for x in (e or {}).get("reviewed_by") or []),
                     (e or {}).get("reject_reason") if (e or {}).get("rejected") else None, (e or {}).get("note")])
    entries.append(("provenance.csv", csv_text(["image", "sample_label", "status", "n_objects", "n_low_conf",
                                                "reviewed_by", "reject_reason", "note"], prov).encode("utf-8")))
    return entries


ZIP_NOTE = ("These zips are streamed: entries have data descriptors (fine for desktop unzip, 7-Zip and Python; "
            "Java's ZipInputStream rejects them).")


def zip_readme(kind: str, ps: dict, user: dict, stale, has_results: bool, has_pkg: bool, problems: list) -> str:
    L = [f"CryoEV download ({ZIP_KINDS[kind]}) for prediction set {ps['id']}", "",
         f"Model: {ps['request'].get('model')}",
         f"Downloaded: {datetime.now().astimezone().isoformat(timespec='seconds')} by {user.get('id')}", ""]
    if kind == "everything":
        if not has_results:
            L += ["RESULTS: not computed yet, so there is no results/ folder.", ""]
        elif stale:
            L += ["RESULTS ARE OUT OF DATE: the review changed after they were computed ("
                  + {"edit": "a later edit", "build": "the review package was rebuilt",
                     "removed": "an edit that was counted has been removed"}.get(stale.get("kind"), "review changed")
                  + "). The reviewed labels in this zip are built from the CURRENT review and do not match results/.",
                  "Open the results page and press 'Update results now', then download again.", ""]
        else:
            L += ["Results are up to date with the review at download time.", ""]
    L += ["Files"]
    if kind in ("images", "everything"):
        L += ["  images/<name>.png        the analysed images", "  images.csv               image, file, sample_label, nm_per_px, ..."]
    if kind in ("outlines", "everything"):
        L += [f"  {COCO_FILE['outlines']}  COCO (pixel coordinates) with a score per outline, conf >= {CAND_CONF}",
              f"  {LABEL_DIR['outlines']}/<image>.txt    YOLO-seg: class x y x y ... (normalised, no score)"]
    if kind in ("reviewed", "everything"):
        if has_pkg:
            L += [f"  {COCO_FILE['reviewed']}   COCO (pixel coordinates) built at download time from the review",
                  f"  {LABEL_DIR['reviewed']}/<image>.txt   YOLO-seg: class x y ... (normalised; original class ids, so "
                  "'Low conf EV' trains as EV; the flag is in the COCO and provenance.csv)",
                  "  rejected_images.csv      rejected images with reason and note. They are NOT left out of the labels: "
                  "a reject changes the counts, not the label data (the COCO image entry says rejected: true).",
                  "  provenance.csv           per image: rejected / reviewed / edited / model-only. 'model-only' images "
                  "carry the model's outlines at conf >= 0.5, NOT checked by a person.",
                  "",
                  f"COCO rule: an EV with low confidence is category '{LOW_CONF_NAME}'; any other class keeps its class "
                  "and gets attributes.low_conf = true; notes are in attributes.note."]
        else:
            L += ["  (no review package for this analysis, so there are no reviewed labels)"]
    L += ["  PROBLEMS.txt             the last entry: anything that had to be left out while the zip was written "
          "(for example an outline with fewer than 3 points)"]
    if kind == "everything" and has_results:
        L += [""] + results_doc_lines("results/")
    L += ["", "COCO file_name is the image's name inside the images zip / this zip (images/<name>): unpack the "
          "images next to the COCO file.", ZIP_NOTE]
    if problems:
        L += ["", f"Not included ({len(problems)}):"] + [f"  {p}" for p in problems[:200]]
    return "\r\n".join(L) + "\r\n"


def build_zip(cfg, ps: dict, user: dict, kind: str):
    """-> (entries for stream_zip, None) or (None, problem text). Never reads results.json except for 'everything'."""
    rows = set_rows(ps)
    if not rows:
        return None, "This analysis has no images."
    problems = [f"{r['image']}: {r['problem']}" for r in rows if r["problem"]]
    want_out = kind in ("outlines", "everything")
    want_rev = kind in ("reviewed", "everything")
    rev = None
    if want_rev:
        try:
            rev = reviewed_source(cfg, ps["id"])
        except Exception as e:  # noqa: BLE001 - the review package can be mid-write or damaged
            if kind == "reviewed":
                return None, f"The review could not be read just now ({type(e).__name__}). Try again in a moment."
            problems.append(f"reviewed labels: the review could not be read ({type(e).__name__}: {e}); "
                            "this zip has no reviewed labels")
        if rev is None and kind == "reviewed":
            return None, "There is no review package for this analysis yet, so there are no reviewed labels."
    if want_out and not (ps["dir"] / "p").is_dir() and kind == "outlines":
        return None, "The model has not produced outlines for this analysis."
    entries = []
    if kind in ("images", "everything"):
        entries += [(r["arc"], lp(r["path"])) for r in rows if r["path"] is not None]
        csv_bytes = images_csv_bytes(ps)               # through the CSV formula guard (orig_name is user text)
        if csv_bytes is not None:
            entries.append(("images.csv", csv_bytes))
    if want_out and (ps["dir"] / "p").is_dir():
        entries += outlines_entries(ps, rows, problems)
    if rev is not None:
        try:
            entries += reviewed_entries(ps, rows, rev, problems)
        except Exception as e:  # noqa: BLE001 - a malformed reviewed/1 must not turn into a 500
            if kind == "reviewed":
                return None, f"The review could not be read just now ({type(e).__name__}). Try again in a moment."
            problems.append(f"reviewed labels: could not be built ({type(e).__name__}: {e})")
            rev = None
    has_results = False
    if kind == "everything":
        rdir = ps["dir"] / "results"
        res, _ = read_results(ps)
        has_results = res is not None and not incomplete(ps)
        if has_results:
            entries += [(f"results/{f}", lp(rdir / f)) for f in FILES if (rdir / f).is_file()]
    stale = stale_info(cfg, ps) if has_results else None
    readme_txt = zip_readme(kind, ps, user, stale, has_results, rev is not None, problems)

    def problems_txt():                  # evaluated when the zip reaches it, i.e. after every other entry streamed
        yield ("\r\n".join(f"{p}" for p in problems) + "\r\n" if problems else "No problems.\r\n").encode("utf-8")
    return [("README.txt", readme_txt.encode("utf-8"))] + entries + [("PROBLEMS.txt", problems_txt())], None


def downloads_zips(ps_id: str, has_pkg: bool, outlines: bool = True) -> str:
    base = f"/results/{esc(ps_id)}/download/"
    items = [("images", "Images (PNG + images.csv)")]
    if outlines:
        items.append(("outlines", "Model outlines (YOLO-seg + COCO with scores)"))
    if has_pkg:
        items.append(("reviewed", "Reviewed labels (COCO + YOLO, rejected list, provenance)"))
    items.append(("everything", "Everything (zip with README)"))
    lis = "".join(f'<li><a href="{base}{esc(ZIP_KINDS[k])}_{esc(ps_id)}.zip" download>{esc(t)}</a></li>' for k, t in items)
    return (f'<h3 class="cj-results-h3">{esc("Images and labels")}</h3><ul class="cj-results-dl" id="cj-results-zips">{lis}</ul>'
            f'<p class="cj-mut cj-results-note">{esc("Big sets are streamed, so the download starts at once and takes a while. Reviewed labels are built from the review as it is now.")}</p>')


# ---------------------------------------------------------------- the page's script (served as /results/results.js)
JS = r"""// Results page (portal/modules/results.py): while a computation runs, GET api/status (same origin) every few
// seconds and reload the page when it is done. The button form is disabled after one press. No HTML built here.
(function () {
  'use strict';
  if (location.protocol !== 'http:' && location.protocol !== 'https:') return;
  Array.prototype.forEach.call(document.querySelectorAll('form.cj-results-form'), function (f) {
    f.addEventListener('submit', function () {
      var b = f.querySelector('button');
      if (b) { setTimeout(function () { b.disabled = true; b.textContent = 'Starting…'; }, 0); }
    });
  });
  var el = document.querySelector('[data-cj-results-poll]');
  if (!el) return;
  var url = el.getAttribute('data-cj-results-poll'), n = 0;
  function note(text) {
    var m = document.getElementById('cj-results-pollerr');
    if (!m) return;
    m.textContent = text || '';
    m.hidden = !text;
  }
  function tick() {
    n += 1;
    fetch(url, {cache: 'no-store', credentials: 'same-origin'})
      .then(function (r) {
        if (!r.ok) throw new Error('status ' + r.status);
        return r.json();
      })
      .then(function (j) {
        note('');
        if (j.state !== 'running' && j.state !== 'queued') { location.reload(); return; }
        setTimeout(tick, n < 30 ? 2000 : 5000);
      })
      .catch(function (e) {
        note('Could not check progress (' + e.message + '); trying again.');
        setTimeout(tick, 10000);
      });
  }
  setTimeout(tick, 1500);
})();
"""


# ---------------------------------------------------------------- routes
def routes(ctx) -> list:
    def not_found(request: Request):
        r = ctx.page(request, "Not found", f'<h1>Not found</h1><p>{esc(NOT_FOUND)}</p>'
                     '<p><a href="/results/">All results</a></p>', active="results")
        r.status_code = 404
        return r

    def get_set(request: Request):
        return load_set(ctx.cfg, request.path_params["ps"], ctx.user(request), ctx.can_see)

    def index(request: Request):                # sync handlers: Starlette runs them in a worker thread
        body = render_index(ctx.cfg, ctx.user(request), ctx.can_see)
        return ctx.page(request, "Results", page_body(body), active="results")

    def show(request: Request):
        ps = get_set(request)
        if ps is None:
            return not_found(request)
        return ctx.page(request, f"Results · {NM.display_name(ps['request'])}", page_body(render_set(ctx.cfg, ps, ctx.user(request))),
                        active="results")

    def compute(request: Request):
        ps = get_set(request)
        if ps is None:
            return not_found(request)
        if predictions_ready(ps):
            start_compute(ctx.cfg, ps)
        elif request.scope.get("cj_auth") == "bearer":   # an agent can't see the page's "not finished" note
            state = (ps["status"] or {}).get("state") or "unknown"
            return ctx.json({"error": f"The predictions of {ps['id']} are not finished (state {state}): results can "
                                      f"be computed once it is done.", "state": state}, 409)
        return ctx.no_store(RedirectResponse(f"/results/{ps['id']}/", status_code=303))

    def api_status(request: Request):
        ps = get_set(request)
        if ps is None:
            return ctx.json({"error": "not found"}, status=404)
        return ctx.json(status_doc(ctx.cfg, ps))

    def stream_download(request: Request, ps: dict, kind: str):
        user = ctx.user(request)
        entries, problem = build_zip(ctx.cfg, ps, user, kind)
        if entries is None:
            r = ctx.page(request, "Nothing to download", f'<h1>Nothing to download</h1><p>{esc(problem)}</p>'
                         f'<p><a href="/results/{esc(ps["id"])}/">Back to the results</a></p>', active="results")
            r.status_code = 404
            return r
        from portal.modules import gallery as G           # the shared "2 downloads per person" slots and stream classes
        slot = G.take_slot(str(user.get("id") or ""))
        if slot is None:
            r = ctx.no_store(PlainTextResponse("Two downloads are already running for you. Wait for one to finish.",
                                               status_code=429))
            r.headers["Retry-After"] = "30"
            return r
        try:
            stream = G.Progress(zipstream_mod().stream_zip(entries), slot)
        except BaseException:
            slot.release()
            raise
        r = G.SlotResponse(stream, slot, media_type="application/zip")
        r.headers["Content-Disposition"] = f'attachment; filename="{ZIP_KINDS[kind]}_{ps["id"]}.zip"'
        r.headers["Cache-Control"] = "no-store"
        r.headers["X-Content-Type-Options"] = "nosniff"
        return r

    def download(request: Request):
        ps = get_set(request)
        name = request.path_params["name"]
        if ps is None:
            return not_found(request)
        kind = zip_kind(name, ps["id"])
        if kind:                                # round 7: streamed zips; images / outlines need no results.json
            return stream_download(request, ps, kind)
        rdir = ps["dir"] / "results"
        res, _ = read_results(ps)               # results.json is written last: without it nothing is complete
        if res is None:
            return not_found(request)
        if (name in FILES or name == f"results_{ps['id']}.zip") and incomplete(ps):
            r = ctx.page(request, "Results incomplete", f'<h1>Results incomplete</h1><p>{esc(INCOMPLETE)}</p>'
                         f'<p><a href="/results/{esc(ps["id"])}/">Back to the results</a></p>', active="results")
            r.status_code = 404
            return r
        if name in FILES:
            p = rdir / name
            if not p.is_file():
                return not_found(request)
            head = csv_header(res, ps) if name.endswith(".csv") else b""
            return attachment(head + p.read_bytes(), f"{ps['id']}_{name}", FILES[name])
        if name == f"results_{ps['id']}.zip":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("README.txt", readme(res, ps, ctx.user(request)))
                for f in FILES:
                    if (rdir / f).is_file():
                        z.write(rdir / f, f)
            return attachment(buf.getvalue(), name, "application/zip")
        return not_found(request)

    def slash(request: Request):
        ps = request.path_params["ps"]
        if not PS_RE.match(ps):
            return not_found(request)
        return ctx.no_store(RedirectResponse(f"/results/{ps}/", status_code=307))

    def script(request: Request):
        r = Response(JS, media_type="text/javascript; charset=utf-8")
        r.headers["Cache-Control"] = "no-cache"
        return r

    return [Route(PREFIX + "/", index, methods=["GET"]),
            Route(PREFIX + "/results.js", script, methods=["GET"]),
            Route(PREFIX + "/{ps}", slash, methods=["GET"]),
            Route(PREFIX + "/{ps}/", show, methods=["GET"]),
            Route(PREFIX + "/{ps}/compute", compute, methods=["POST"]),
            Route(PREFIX + "/{ps}/api/status", api_status, methods=["GET"]),
            Route(PREFIX + "/{ps}/download/{name}", download, methods=["GET"])]
