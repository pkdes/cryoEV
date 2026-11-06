"""
Predict module (CJ portal, journey J1): a person runs the production model on their tiles and gets a prediction set
(BUILD_CONTRACTS.md section 2) to review and count.

    GET  /predict/                        My analyses (admin: everyone's) + "New analysis"
    GET  /predict/predict.js              the module's script (a file: no inline script anywhere)
    GET  /predict/new/                    step 1: what images work; pick a processed session
    GET  /predict/new/session/<key>/      step 2: tiles (all, a count, by sample), nm/px per sample + the
    POST /predict/new/session/<key>/        confirmation tick, model -> create() + start_detached() -> 303 set page
    GET  /predict/<ps>/                   the set: status (auto-refresh), Review / Results, spot-check list
    POST /predict/<ps>/review             build the prediction review package in a thread (if missing)
    POST /predict/<ps>/results            compute results/results.json in a thread (if missing or older than edits)
    POST /predict/<ps>/start              start the laptop run again (a failed set, or one waiting for the GPU)
    GET  /predict/api/mine                {"schema": "cj_predict_mine/1", "items": [...]}: Home's "My results"
    GET  /predict/api/set/<ps>            one set's status; ?html=1 adds the page fragments (the refresh swaps them)

Configuration (environment, read per request; Config attributes of the same lower-case name win when present):
  CJ_PREDICT_SOURCES   JSON list, or the path of a JSON file holding it: [{"label": "Session 27 Oct (EV prep A)",
                       "dir": "<processed tile folder>", "manifest": "<MANIFEST.csv>" (optional; default
                       <dir>/MANIFEST.csv when present), "session": "YYYYMMDD" (optional; else the first 8-digit
                       date in the folder name), "sample_column": "grid" (optional; default the first of
                       sample_label, sample, grid in the manifest)}]. Only the admin sets this: people pick from
                       these folders, never type a path.
  CJ_PREDICT_ROOT      prediction sets (default <CryoAI-laptop>/predict, jobs.predict_set.default_root())
  CJ_REVIEW_ROOT       prediction review packages (default <CryoAI-laptop>/review)
  The registry is the portal's (cfg.registry).

Rules:
  - Every set view, API and action checks ctx.can_see(user, request.json owner); not found and not allowed give the
    same 404.
  - nm/px is never defaulted: each sample's value is shown (from the manifest when it has one) and the person must
    tick "I confirm the pixel size"; their id goes into images.csv nm_confirmed_by. No tick: refused, nothing written.
  - Model: the registry's production model (also in models/models.yaml) by default; basic users get only that one.
  - Long work (create(), the review package, results) runs in worker threads, never on the event
    loop. The predictor itself is a detached process (jobs.predict_local.start_detached); status.json says the rest.
  - Plain words for scientists: paths, ids of folders and raw errors only on admin-only lines.
  - Everything shown is escaped here.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import re
import secrets
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route

from jobs import predict_local as PL
from jobs import predict_set as PS

PREFIX = "/predict"
ROLES = ("admin", "advanced", "basic")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TEMPLATE = HERE.parent / "templates" / "predict.html"
MODELS_YAML = REPO / "models" / "models.yaml"
PORTAL_TMP = ".portal"                    # <root>/.portal/: the portal's own scratch (consumers skip dot entries)
IMAGE_EXT = (".png", ".jpg", ".jpeg")
SAMPLE_COLUMNS = ("sample_label", "sample", "grid")
MAX_FILES = PS.MAX_IMAGES                 # 200 tiles per analysis (= a prediction set's maximum)
NM_MIN, NM_MAX = 0.01, 100.0              # a plausible nm/px (cryo-EM tiles); outside = a typo
STALE_RUNNING_S = 120                     # predict_local refreshes `updated` every 2 s while running
REFRESH_S = 5
SPOT_N = 10
SCHEMA_MINE = "cj_predict_mine/1"
SCHEMA_SET = "cj_predict_set/1"

esc = html.escape


class Refusal(Exception):
    """A request we refuse with a message for the person (plain words) and an HTTP status."""

    def __init__(self, msg, status=400, tech=None):
        super().__init__(msg)
        self.msg, self.status, self.tech = msg, status, tech


# ---------------------------------------------------------------- configuration
def _setting(cfg, name):
    v = getattr(cfg, name.lower(), None)
    return v if v not in (None, "") else os.environ.get(name.upper())


def predict_root(cfg) -> Path:
    v = _setting(cfg, "CJ_PREDICT_ROOT")
    return Path(v) if v else PS.default_root()


def review_root(cfg) -> Path:
    v = _setting(cfg, "CJ_REVIEW_ROOT")
    if v:
        return Path(v)
    from data_utils import config
    return config.laptop_root() / "review"


def registry_root(cfg):
    return getattr(cfg, "registry", None)


def sources(cfg) -> list:
    """The admin's source folders: [{key, label, dir, manifest, session, sample_column, problem}]. A bad setting
    gives an empty list (and the admin sees why on the page)."""
    raw = _setting(cfg, "CJ_PREDICT_SOURCES")
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        text = str(raw).strip()
        if not text.startswith("["):
            text = Path(text).read_text(encoding="utf-8-sig")
        items = json.loads(text)
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not it.get("dir"):
            continue
        d = Path(str(it["dir"]))
        man = Path(str(it["manifest"])) if it.get("manifest") else (d / "MANIFEST.csv")
        session = str(it.get("session") or "")
        if not session:
            m = re.search(r"(?<!\d)(\d{8})(?!\d)", d.name) or re.search(r"(?<!\d)(\d{8})(?!\d)", str(d))
            session = m.group(1) if m else ""
        problem = None
        if not d.is_dir():
            problem = "folder not found"
        elif not PS.SESSION_RE.match(session):
            problem = 'no session date: add "session": "YYYYMMDD" to this source'
        out.append({"key": hashlib.sha1(os.path.normcase(str(d)).encode("utf-8")).hexdigest()[:10],
                    "label": str(it.get("label") or d.name), "dir": d,
                    "manifest": man if man.is_file() else None, "session": session,
                    "sample_column": it.get("sample_column"), "problem": problem})
    return out


def sources_safe(cfg):
    try:
        return sources(cfg), None
    except (OSError, ValueError) as e:
        return [], f"CJ_PREDICT_SOURCES unreadable: {type(e).__name__}: {e}"


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


def tiles_of(src: dict) -> dict:
    """{"groups": [{label, names, nm (None when missing or mixed), nm_values, mixed}], "sig", "n", "column"}.
    Groups come from the manifest's sample column; without a manifest, one group (label = the source label)."""
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
        g = groups.setdefault(label, {"label": label, "names": [], "nm_values": set()})
        g["names"].append(n)
        g["nm_values"].add(_num(r.get(PS.NM_COLUMN)))
    out = []
    for label in sorted(groups, key=str):
        g = groups[label]
        vals = {v for v in g["nm_values"] if v is not None}
        nm = next(iter(vals)) if len(vals) == 1 and None not in g["nm_values"] else None
        out.append({"label": label, "names": g["names"], "nm": nm, "nm_values": sorted(vals),
                    "mixed": len(vals) > 1})
    sig = hashlib.sha1(json.dumps([[g["label"], len(g["names"])] for g in out]).encode()).hexdigest()[:12]
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
    return [{"label": s["label"], "n": s["n"], "nm_per_px": ", ".join(f"{float(v):g}" if _num(v) else "?"
                                                                       for v in sorted(s["nm"])),
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
        _TASKS[(kind, ps)] = {"state": "running", "error": None, "started": time.time()}

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
    return pkg


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
    out = []
    for i in ids:
        r = by_image.get(i) or {}
        out.append({"image": i, "file": Path(r.get("file") or "").name or None,
                    "sample_label": r.get("sample_label"), "orig_name": r.get("orig_name")})
    return out


# ---------------------------------------------------------------- status in plain words
def status_view(st: dict, admin: bool, now=None) -> dict:
    now = time.time() if now is None else now
    state = st.get("state")
    n_done, n_total = st.get("n_done"), st.get("n_total")
    out = {"state": state, "tone": "wait", "label": "Unknown", "detail": None, "tech": None, "final": False,
           "n_done": n_done, "n_total": n_total, "progress": False}
    waiting = st.get("waiting")
    if state == "queued":
        out["label"] = "Waiting to start"
        if waiting and str(waiting).startswith("gpu busy"):
            out["detail"] = "Waiting for the graphics card: another job is using it. Use “Start again” " \
                            "later, or ask the admin."
        elif waiting:
            out["detail"] = "Waiting: " + str(waiting) if admin else "Waiting for the computer to be free."
        elif st.get("job"):
            out["detail"] = "Queued on the GPU computer."
        elif now - (_parse_local(st.get("updated")) or now) > STALE_RUNNING_S:
            out["detail"] = "Not started yet (another analysis may have been running). Use “Start again”."
        else:
            out["detail"] = "It starts in a moment."
    elif state == "running":
        out.update(tone="run", label="Running", progress=bool(n_total))
        out["detail"] = f"{int(n_done or 0)} of {int(n_total or 0)} images done"
        age = now - (_parse_local(st.get("updated")) or now)
        if age > STALE_RUNNING_S:
            out.update(tone="warn", label="No progress")
            out["detail"] += f"; no progress for {ago(age)}: the run may have stopped. Try “Start again” " \
                             "or ask the admin."
    elif state == "done":
        out.update(tone="ok", label="Done", final=True, detail=f"{int(n_total or 0)} images analysed")
    elif state == "failed":
        out.update(tone="warn", label="Stopped with an error", final=True,
                   detail="Try “Start again”; if it fails again, ask the admin (they can see the details).")
    if admin:
        bits = [f"where {st.get('where')}"] + [f"{k} {st.get(k)}" for k in ("job", "pid", "waiting", "error")
                                                if st.get(k)] + [f"updated {st.get('updated')}"]
        out["tech"] = "; ".join(bits)
    return out


def can_restart(st: dict, now=None) -> bool:
    now = time.time() if now is None else now
    state = st.get("state")
    if st.get("job") and state in ("queued", "running"):
        return False                                   # on the desktop queue: not ours to start here
    if state == "failed" or (state == "queued" and st.get("waiting")):
        return True
    # running without a heartbeat, or queued and never picked up (predict_local exits at once when another run
    # holds <root>/predict.lock): offer a manual start; predict_local's own lock keeps it to one run at a time
    return state in ("running", "queued") and now - (_parse_local(st.get("updated")) or now) > STALE_RUNNING_S


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


def source_text(req: dict, srcs) -> str:
    src = req.get("source") or {}
    if src.get("kind") == "upload":
        return "Uploaded images"
    label = next((s["label"] for s in srcs if src.get("dir") and
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
            "source": source_text(req, srcs), "state": view["state"], "label": view["label"],
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
    srcs, _ = sources_safe(cfg)
    items = [item_of(cfg, user, d, req, st, srcs) for d, req, st in list_sets(cfg, user, can_see)]
    return wrap(f'<div class="cj-predict"><div class="cj-predict-head"><div><h1>Predict</h1>'
                f'<p class="cj-mut">Count and size the EVs in your images with the lab’s model.</p></div>'
                f'<a class="cj-predict-btn cj-predict-primary" href="/predict/new/">New analysis</a></div>'
                f'<h2 class="cj-sec">{"All analyses" if admin else "My analyses"}</h2>{mine_html(items, admin)}'
                f'</div>')


def model_block(cfg, user, chosen=None) -> str:
    default, choices = allowed_models(cfg, user)
    if not choices:
        return msg_box("No production model is set up yet, so analyses can't start. Ask the admin.")
    pick_id = chosen if any(c["id"] == chosen for c in choices) else default
    if len(choices) == 1:
        c = choices[0]
        q = f'<p class="cj-mut">{esc(c["quality"])}</p>' if c["quality"] else ""
        return (f'<fieldset class="cj-card cj-predict-fs"><legend>Model</legend>'
                f'<input type="hidden" name="model" value="{esc(c["id"])}">'
                f'<p><b>{esc(c["id"])}</b>{" (the lab’s production model)" if c["production"] else ""}</p>'
                f'{q}</fieldset>')
    opts = "".join(f'<option value="{esc(c["id"])}"{" selected" if c["id"] == pick_id else ""}>'
                   f'{esc(c["id"])}{" (production)" if c["production"] else ""}</option>' for c in choices)
    qual = "".join(f'<li><b>{esc(c["id"])}</b>: {esc(c["quality"])}</li>' for c in choices if c["quality"])
    return (f'<fieldset class="cj-card cj-predict-fs"><legend>Model</legend>'
            f'<label class="cj-predict-field"><span>Model (keep the production model unless you know why)</span>'
            f'<select name="model">{opts}</select></label>'
            + (f'<ul class="cj-mut cj-predict-qual">{qual}</ul>' if qual else "") + '</fieldset>')


def page_new(cfg, user) -> str:
    admin = user.get("role") == "admin"
    srcs, err = sources_safe(cfg)
    cards = []
    for s in srcs:
        if s["problem"]:
            if admin:
                cards.append(f'<div class="cj-card cj-predict-src cj-predict-off"><b>{esc(s["label"])}</b>'
                             f'<p class="cj-mut">Not available: {esc(s["problem"])}</p>{tech(str(s["dir"]))}</div>')
            continue
        cards.append(f'<a class="cj-card cj-predict-src" href="/predict/new/session/{esc(s["key"])}/">'
                     f'<b>{esc(s["label"])}</b><span class="cj-mut">Session {esc(s["session"])}</span>'
                     + (tech(str(s["dir"])) if admin else "") + '</a>')
    if err and admin:
        cards.append(msg_box("The session list setting can't be read.", extra=tech(err)))
    sess = ("".join(cards) or '<p class="cj-mut">No processed sessions are set up for analysis yet'
            + (' (set CJ_PREDICT_SOURCES).' if admin else '; ask the admin.') + '</p>')
    return wrap(
        '<div class="cj-predict"><p class="cj-mut"><a href="/predict/">← My analyses</a></p>'
        '<h1>New analysis</h1>'
        '<div class="cj-card cj-predict-what"><h2>What images work</h2><ul>'
        '<li>Processed cryo-EM tiles: 8-bit PNG or JPEG, contrast-adjusted the way the lab’s processing does '
        'it (not raw MRC or TIFF).</li>'
        '<li>The same kind of images and magnification as the model was trained on. If unsure, ask the admin.</li>'
        f'<li>Up to {MAX_FILES} images per analysis. You must confirm the pixel size (nm per pixel) of each '
        'sample: sizes in nm depend on it.</li></ul></div>'
        '<h2 class="cj-sec">1. Pick a processed session</h2>'
        f'<div class="cj-predict-srcs">{sess}</div>'
        '</div>')


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
            note = ('<span class="cj-predict-warn">Mixed pixel sizes in this sample (' +
                    esc(", ".join(f"{v:g}" for v in g["nm_values"])) + ' nm/px): it can’t be analysed as one '
                    'sample. Ask the admin.</span>')
            rows.append(f'<tr><td data-l="Sample"><label class="cj-predict-off"><input type="checkbox" disabled> '
                        f'{esc(g["label"])}</label></td><td data-l="Tiles" class="cj-num">{len(g["names"])}</td>'
                        f'<td data-l="nm/px">{note}</td></tr>')
            continue
        src_note = "from the session’s records" if g["nm"] else "not recorded: enter it"
        rows.append(
            f'<tr><td data-l="Sample"><label><input type="checkbox" name="samples" value="{gid}" '
            f'data-n="{len(g["names"])}"{" checked" if gid in picked else ""}> {esc(g["label"])}</label></td>'
            f'<td data-l="Tiles" class="cj-num">{len(g["names"])}</td>'
            f'<td data-l="nm/px"><input class="cj-predict-nm" name="nm_{gid}" inputmode="decimal" '
            f'aria-label="nm per pixel for {esc(g["label"])}" value="{esc(nm_val)}" required>'
            f' <span class="cj-mut">{src_note}</span></td></tr>')
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


def page_set(cfg, user, d: Path, req: dict) -> str:
    admin = user.get("role") == "admin"
    v = set_view(cfg, user, d, req, True)
    rows = images_rows(d)
    srcs, _ = sources_safe(cfg)
    samples = "".join(f'<tr><td data-l="Sample">{esc(s["label"])}</td><td data-l="Images" class="cj-num">{s["n"]}'
                      f'</td><td data-l="nm/px" class="cj-num">{esc(s["nm_per_px"])}</td>'
                      f'<td data-l="Confirmed by">{esc(s["confirmed_by"] or "?")}</td></tr>'
                      for s in samples_of(rows))
    owner = f'<div><dt>Owner</dt><dd>{esc(req.get("owner"))}</dd></div>' if admin else ""
    src_tech = tech(f"{d} · source {json.dumps(req.get('source'))}") if admin else ""
    return wrap(
        f'<div class="cj-predict" id="cj-predict-set" data-ps="{esc(d.name)}" data-live="{int(v["live"])}" '
        f'data-every="{REFRESH_S}"><p class="cj-mut"><a href="/predict/">← My analyses</a></p>'
        f'<h1>Analysis of {esc(when(req.get("created")))}</h1><p class="cj-mut cj-wrap">{esc(d.name)}</p>'
        f'<p id="cj-predict-err" class="cj-mut" role="status" hidden></p>'
        f'<section id="cj-predict-status">{v["html"]["status"]}</section>'
        f'<section id="cj-predict-actions">{v["html"]["actions"]}</section>'
        f'<h2 class="cj-sec">Details</h2><div class="cj-card"><dl class="cj-predict-kv">'
        f'<div><dt>Images from</dt><dd>{esc(source_text(req, srcs))}</dd></div>'
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
        return "No production model is set up yet, so analyses can't start. Ask the admin."
    if model not in {c["id"] for c in choices}:
        return "Pick a model from the list."
    return None


def scratch(cfg, kind) -> Path:
    d = predict_root(cfg) / PORTAL_TMP / f"{kind}_{secrets.token_hex(6)}"
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


def session_submit(cfg, user, src: dict, form: dict) -> Path:
    if form.get("confirm") != "yes":
        raise Refusal("Tick “I confirm the pixel size”: nothing was started.")
    t = tiles_of(src)
    if form.get("sig") != t["sig"]:
        raise Refusal("The session’s tiles changed since this page was opened. Check the samples again.")
    problems = []
    chosen = []
    for i, g in enumerate(t["groups"]):
        gid = f"g{i}"
        if gid not in (form.get("samples") or []):
            continue
        if g["mixed"]:
            problems.append(f"{g['label']}: mixed pixel sizes; it can't be analysed as one sample")
            continue
        nm, err = check_nm(form.get(f"nm_{gid}"), g["label"])
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
    label_of = {n: (label if label is not None else g["label"]) for g in chosen for n in g["names"]}
    nm_of = {n: g["nm_in"] for g in chosen for n in g["names"]}
    tmp = scratch(cfg, "sel")
    try:
        man = tmp / "MANIFEST.csv"
        with open(man, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["output_name", "sample_label", PS.NM_COLUMN])
            for n in names:
                w.writerow([n, label_of[n], repr(float(nm_of[n]))])
        return create_and_start(cfg, src["dir"], user, form["model"], names=names, manifest=man,
                                session=src["session"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- the page's script (served as /predict/predict.js)
JS = r"""// Predict module (portal/modules/predict.py). No inline script: this file is the module's only script.
// - Set page (#cj-predict-set[data-live="1"]): every few seconds GET /predict/api/set/<ps>?html=1 (same origin) and
//   swap in the status and actions fragments, rendered and escaped by the server; stops when nothing is left to do.
// - Forms: show what is selected, disable the button while it works.
(function () {
  'use strict';
  if (location.protocol !== 'http:' && location.protocol !== 'https:') return;

  function swap(id, frag) {
    var box = document.getElementById(id);
    if (!box || typeof frag !== 'string') return;
    var t = document.createElement('template');
    t.innerHTML = frag;
    box.replaceChildren(t.content);
  }

  var set = document.getElementById('cj-predict-set');
  if (set && set.dataset.ps) {
    var every = Math.max(2, parseInt(set.dataset.every, 10) || 5) * 1000, busy = false, timer = null;
    var note = document.getElementById('cj-predict-err');
    var tick = function () {
      if (busy || document.hidden) return;
      busy = true;
      fetch('/predict/api/set/' + encodeURIComponent(set.dataset.ps) + '?html=1',
            {cache: 'no-store', credentials: 'same-origin'})
        .then(function (r) { if (!r.ok) throw new Error('status ' + r.status); return r.json(); })
        .then(function (j) {
          var h = (j && j.html) || {};
          swap('cj-predict-status', h.status);
          swap('cj-predict-actions', h.actions);
          if (note) { note.textContent = ''; note.hidden = true; }
          if (!j.live && timer) { clearInterval(timer); timer = null; }
        })
        .catch(function (e) {
          if (note) { note.textContent = 'Could not refresh (' + e.message + '); showing the last state.'; note.hidden = false; }
        })
        .then(function () { busy = false; });
    };
    if (set.dataset.live === '1') timer = setInterval(tick, every);
    document.addEventListener('visibilitychange', function () { if (!document.hidden && timer) tick(); });
  }

  document.addEventListener('submit', function (e) {
    var f = e.target;
    if (!f || f.tagName !== 'FORM') return;
    if (f.dataset.sent === '1') { e.preventDefault(); return; }
    f.dataset.sent = '1';
    var b = f.querySelector('button[type="submit"]');
    if (b) { b.disabled = true; if (b.dataset.busy) b.textContent = b.dataset.busy; }
  });

  var sess = document.getElementById('cj-predict-session');
  if (sess) {
    var out = document.getElementById('cj-predict-sel'), max = parseInt(sess.dataset.max, 10) || 200;
    var count = sess.querySelector('input[name="count"]');
    var update = function () {
      var n = 0;
      sess.querySelectorAll('input[name="samples"]:checked').forEach(function (c) { n += parseInt(c.dataset.n, 10) || 0; });
      var mode = sess.querySelector('input[name="how"]:checked');
      if (mode && mode.value === 'count') {
        var k = parseInt(count.value, 10);
        if (k > 0) n = Math.min(n, k);
      }
      out.textContent = n + ' tile' + (n === 1 ? '' : 's') + ' will be analysed' + (n > max ? ' (too many: at most ' + max + ')' : '') + '.';
    };
    sess.addEventListener('input', update);
    sess.addEventListener('change', update);
    if (count) count.addEventListener('focus', function () {
      var r = sess.querySelector('input[name="how"][value="count"]');
      if (r) r.checked = true;
      update();
    });
    update();
  }
})();
"""


# ---------------------------------------------------------------- routes
def routes(ctx) -> list:
    cfg = ctx.cfg

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

    def find_source(key):
        srcs, _ = sources_safe(cfg)
        return next((s for s in srcs if s["key"] == key and not s["problem"]), None)

    # ---- pages (sync: Starlette runs them in a worker thread)
    def index(request: Request):
        return ctx.page(request, "Predict", page_mine(cfg, ctx.user(request), ctx.can_see), active="predict")

    def new(request: Request):
        return ctx.page(request, "New analysis", page_new(cfg, ctx.user(request)), active="predict")

    def session_form(request: Request):
        src = find_source(request.path_params["key"])
        if src is None:
            return not_found(request)
        user = ctx.user(request)
        return ctx.page(request, src["label"], page_session_form(cfg, user, src, tiles_of(src)), active="predict")

    async def session_post(request: Request):
        src = await run_in_threadpool(find_source, request.path_params["key"])
        if src is None:
            return await run_in_threadpool(not_found, request)
        user = ctx.user(request)
        fd = await request.form(max_files=0, max_fields=1000)
        form = {k: v for k, v in fd.items() if isinstance(v, str)}
        form["samples"] = [v for v in fd.getlist("samples") if isinstance(v, str)]
        try:
            ps_dir = await run_in_threadpool(session_submit, cfg, user, src, form)
        except Refusal as e:
            msg, status = e.msg + (f"\n(admin) {e.tech}" if e.tech and user.get("role") == "admin" else ""), e.status

            def again():
                return refused_page(request, src["label"],
                                    page_session_form(cfg, user, src, tiles_of(src), form, msg), status)
            return await run_in_threadpool(again)
        return RedirectResponse(f"/predict/{ps_dir.name}/", status_code=303)

    def set_page(request: Request):
        user = ctx.user(request)
        d, req = load_set(cfg, user, ctx.can_see, request.path_params["ps"])
        if d is None:
            return not_found(request)
        return ctx.page(request, "Analysis", page_set(cfg, user, d, req), active="predict")

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
                if can_restart(st):
                    PL.start_detached(d, root=predict_root(cfg), registry_root=registry_root(cfg))
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
        srcs, _ = sources_safe(cfg)
        items = [item_of(cfg, user, d, req, st, srcs) for d, req, st in list_sets(cfg, user, ctx.can_see)]
        return ctx.json({"schema": SCHEMA_MINE, "items": items})

    def api_set(request: Request):
        user = ctx.user(request)
        d, req = load_set(cfg, user, ctx.can_see, request.path_params["ps"])
        if d is None:
            return ctx.json({"error": "not found"}, 404)
        return ctx.json(set_view(cfg, user, d, req, request.query_params.get("html") == "1"))

    def script(request: Request):
        r = Response(JS, media_type="text/javascript; charset=utf-8")
        r.headers["Cache-Control"] = "no-cache"
        return r

    return [Route(PREFIX + "/", index, methods=["GET"]),
            Route(PREFIX + "/predict.js", script, methods=["GET"]),
            Route(PREFIX + "/new/", new, methods=["GET"]),
            Route(PREFIX + "/new/session/{key}/", session_form, methods=["GET"]),
            Route(PREFIX + "/new/session/{key}/", session_post, methods=["POST"]),
            Route(PREFIX + "/api/mine", api_mine, methods=["GET"]),
            Route(PREFIX + "/api/set/{ps}", api_set, methods=["GET"]),
            Route(PREFIX + "/{ps}/", set_page, methods=["GET"]),
            Route(PREFIX + "/{ps}/review", action("review"), methods=["POST"]),
            Route(PREFIX + "/{ps}/results", action("results"), methods=["POST"]),
            Route(PREFIX + "/{ps}/start", action("start"), methods=["POST"])]
