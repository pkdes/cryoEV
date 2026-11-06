"""
Results module (CJ portal): EV counts, sizes and multilayer % for one prediction set, in plain words, phone first.
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
  - Downloads: a CSV newer than results.json (a write in progress or interrupted) is not served: 404 "compute
    again".
  - Images of a refused sample are listed muted, "Not counted (sample refused)".
  - Links: the spot-check subset at /predict/<ps_id>/; "Review these images" at /review/pred_<ps_id>/ when the
    package exists (the shell checks the owner there).
  - Standalone CSV downloads start with # provenance lines; the zip (tables + README.txt) is the recommended one.
  - The set folder is resolved and must still be <predict root>/<ps_id> (a junction or link out of it = 404).
  - A refused sample (mixed nm/px or tile size, no confirmed nm/px) is a notice with the reason; the other samples
    are still shown. A whole-set refusal (predictions not finished) shows the notice only.
  - Compute: one at a time (a second one waits, "queued"). It uses the review package's edits when
    <CryoAI-laptop>/review/pred_<ps_id>/ exists (annotation.prediction_package.reviewed_polygons, write=False, so
    the package is never written), else the model predictions. Writes only <ps>/results/ (compute_and_write).
  - Roots: cfg.predict_root / cfg.review_root when the portal Config has them, else
    data_utils.config.laptop_root() / "predict" and / "review".
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
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route

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
    imgs = res["images"].values()
    n = len(res["images"])
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
        if lab not in res["samples"]:
            out_of[lab] = out_of.get(lab, 0) + 1
    items = "".join(f"<li><b>{esc(str(k))}</b> ({esc(plural(v, 'image'))})</li>" for k, v in out_of.items())
    head = ("No sample could be counted." if not res["samples"] else "Some samples were not counted.")
    if not res["samples"] and not out_of:
        head = "No results for this set."
    return (f'<div class="cj-results-banner cj-results-b-warn" role="alert" id="cj-results-refused">'
            f'<p><b>{esc(head)}</b></p>' + (f'<ul class="cj-results-ul">{items}</ul>' if items else "")
            + f'<p class="cj-results-why">{esc("Why: " + str(why))}</p></div>')


def sample_table(res: dict) -> str:
    samples = res["samples"]
    if not samples:
        return ""
    reviewer = any(int(s.get("n_reviewed") or 0) > 0 for s in samples.values())
    head = ["Sample", "Images", "Reviewed", "EVs", "EVs per image", "Median diameter (Q1–Q3)",
            "Multilayer (geometry)"]
    if reviewer:
        head += ["Multilayer (reviewer)", "Reviewer vs geometry"]
    head += ["Non-EV"]
    rows = []
    for label, s in samples.items():
        dv, dsub = fdiam(s)
        n_img, n_rev = int(s.get("n_images") or 0), int(s.get("n_reviewed") or 0)
        cells = [f'<th scope="row" data-label="Sample" class="cj-results-sname">{esc(str(label))}</th>',
                 td("Images", fint(n_img), k="n_images"),
                 td("Reviewed", f"{n_rev} / {n_img}", k="n_reviewed"),
                 td("EVs", fint(s.get("ev_total")), k="ev_total"),
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
        cls = "" if counted else ' class="cj-results-off"'
        rows.append(f'<tr data-image="{esc(str(k))}"{cls}><td class="cj-num">{esc(str(k))}</td>'
                    f'<td class="cj-results-l">{esc(str(i.get("sample_label")))}</td><td>{image_chip(i, counted)}</td>'
                    f'<td>{esc(fint(i.get("ev")))}</td><td>{esc(fint(i.get("multilayer")))}</td>'
                    f'<td>{esc(fint(i.get("inner")))}</td><td>{esc(fint(i.get("nonev")))}</td></tr>')
    th = "".join(f'<th scope="col">{esc(h)}</th>' for h in head)
    return (f'<details class="cj-results-det" id="cj-results-images"><summary>{esc(f"Per image ({len(imgs)})")}'
            f'</summary><div class="cj-results-wrap"><table class="cj-results-tab cj-results-img">'
            f'<thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></details>')


def provenance(res: dict, ps: dict) -> dict:
    imgs = res["images"].values()
    rule = res.get("rule") or {}
    return {"Prediction set": ps["id"],
            "Model": str(res.get("model")),
            "Computed": when(res.get("computed")),
            "Counting rule": (f"outlines at confidence {rule.get('conf')} or more (on images nobody edited); "
                              f"{ev_rule(rule.get('ev_classes'))}; inner layers not counted; "
                              f"multilayer = {rule.get('multilayer')}"),
            "Checked by a person": (f"{sum(1 for i in imgs if i.get('reviewed'))} of {len(res['images'])} images "
                                    f"reviewed, {sum(1 for i in imgs if i.get('edited') and not i.get('reviewed'))}"
                                    " more edited"),
            "Not counted": str(res.get("refused") or "nothing")}


def downloads(ps_id: str) -> str:
    base = f"/results/{esc(ps_id)}/download/"
    items = [("per_sample.csv", "Per sample (CSV)"), ("per_image.csv", "Per image (CSV)"),
             ("results.json", "All numbers (JSON)")]
    return (f'<p class="cj-results-dlmain"><a class="cj-results-btn" href="{base}{esc(f"results_{ps_id}.zip")}" '
            f'download>{esc("Download everything (zip)")}</a> <span class="cj-mut">'
            f'{esc("Recommended: the tables, all numbers and a README of how they were made.")}</span></p>'
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
    imgs = res["images"].values()
    rule = res.get("rule") or {}
    lines = [f"# CJ results for prediction set {ps['id']}",
             f"# model={res.get('model')}, computed={res.get('computed')}, "
             f"reviewed={sum(1 for i in imgs if i.get('reviewed'))}/{len(res['images'])} images, "
             f"edited (not reviewed)={sum(1 for i in imgs if i.get('edited') and not i.get('reviewed'))}",
             f"# rule=conf>={rule.get('conf')}; {ev_rule(rule.get('ev_classes'))}; inner layers not counted; "
             f"multilayer={rule.get('multilayer')}",
             f"# not counted={res.get('refused') or 'nothing'}"]
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
           '<h1>Results</h1>',
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
                   f'<p><b>{esc(head)}</b> {esc("Compute again.")}</p></div>')
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
        return "".join(out)

    out.append(honesty(res, ps_id, spot_count(cfg, ps_id), package_dir(cfg, ps_id) is not None))
    out.append(refused_notice(res))
    if res["samples"]:
        out.append('<h2 class="cj-sec">Per sample</h2>')
        out.append(sample_table(res))
        out.append('<h2 class="cj-sec">EV size</h2>')
        out.append('<figure class="cj-card cj-results-fig"><figcaption>EV diameter by sample (nm)'
                   '<span class="cj-mut">Dot: median. Bar: the middle half of the EVs (Q1 to Q3).</span>'
                   f'</figcaption>{diam_chart(res["samples"])}</figure>')
    if res["images"]:
        out.append('<h2 class="cj-sec">Images</h2>')
        out.append(image_table(res))
    out.append('<h2 class="cj-sec">Download</h2>')
    out.append(downloads(ps_id))
    out.append('<h2 class="cj-sec">How these numbers were made</h2>')
    out.append('<dl class="cj-card cj-results-prov" id="cj-results-prov">' + "".join(
        f"<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>" for k, v in provenance(res, ps).items()) + "</dl>")
    if st["state"] not in ("queued", "running"):
        out.append('<div class="cj-results-again"><p class="cj-mut">'
                   + esc("After more review, compute again to include it.") + "</p>"
                   + compute_form(ps_id, "Compute again", primary=False) + "</div>")
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
        items.append(f'<li><a class="cj-card cj-results-set" href="/results/{esc(ps["id"])}/">'
                     f'<span class="cj-results-set-t"><b class="cj-wrap">{esc(ps["id"])}</b>{chip(label, tone)}</span>'
                     f'<span class="cj-mut">{esc(meta)}</span></a></li>')
    out.append(f'<ul class="cj-results-sets">{"".join(items)}</ul>')
    return "".join(out)


_PH = re.compile(r"\{\{content\}\}")


def page_body(content: str) -> str:
    return _PH.sub(lambda m: content, TEMPLATE.read_text(encoding="utf-8"), count=1)


def readme(res: dict, ps: dict, user: dict) -> str:
    lines = [f"CJ results for prediction set {ps['id']}", ""]
    lines += [f"{k}: {v}" for k, v in provenance(res, ps).items()]
    lines += [f"Downloaded: {datetime.now().astimezone().isoformat(timespec='seconds')} by {user.get('id')}", "",
              "Files",
              "  per_sample.csv  one row per counted sample (refused samples are left out; see 'Not counted');",
              "                  the CSVs in this zip have no # header lines (this README is their provenance)",
              "  per_image.csv   one row per image; columns reviewed / edited say whether a person checked it",
              "                  (reviewed = true), changed it without marking it reviewed (edited = true), or",
              "                  neither (model only)",
              "  results.json    everything above, schema results/1", "",
              "Columns",
              "  ev_total                  EVs (outlines of the EV classes that are not inner layers)",
              "  pct_multilayer_geometry   % of EVs whose outline contains another outline (the main number)",
              "  pct_multilayer_reviewer   % of EVs a reviewer called multilayer, reviewed images only (base: "
              "reviewer_ev)",
              "  diam_nm_median / q1 / q3  area-equivalent diameter in nm",
              "  inner_layers_excluded     inner outlines of multilayer EVs, not counted as EVs",
              "  nonev_total               other objects, counted separately",
              "  reviewer_geometry_disagree  reviewed outlines where the reviewer's call differs from the geometry",
              ""]
    return "\r\n".join(lines)


def attachment(body: bytes, name: str, media: str) -> Response:
    r = Response(body, media_type=media)
    r.headers["Content-Disposition"] = f'attachment; filename="{name}"'
    r.headers["X-Content-Type-Options"] = "nosniff"
    r.headers["Cache-Control"] = "no-store"
    return r


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
        return ctx.page(request, f"Results {ps['id']}", page_body(render_set(ctx.cfg, ps, ctx.user(request))),
                        active="results")

    def compute(request: Request):
        ps = get_set(request)
        if ps is None:
            return not_found(request)
        if predictions_ready(ps):
            start_compute(ctx.cfg, ps)
        return ctx.no_store(RedirectResponse(f"/results/{ps['id']}/", status_code=303))

    def api_status(request: Request):
        ps = get_set(request)
        if ps is None:
            return ctx.json({"error": "not found"}, status=404)
        return ctx.json(status_doc(ctx.cfg, ps))

    def download(request: Request):
        ps = get_set(request)
        name = request.path_params["name"]
        if ps is None:
            return not_found(request)
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
