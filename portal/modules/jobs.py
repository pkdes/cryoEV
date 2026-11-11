"""
Jobs module (CryoEV portal): what is running, waiting, stuck and done, from two sources:
  - the GPU computers' queues: <registry>/live/live.json, written by jobs/monitor.py (train jobs; predict jobs carry
    kind "predict" + n_done / n_total; queued jobs may carry ready_at, the time they became READY);
  - the registry's jobs: registry.current("job") (review / annotation / check jobs, and any train / predict job
    that is not on a queue), cached REGISTRY_TTL_S per registry folder (current() re-reads the files).

    GET /jobs/               the page: Running now, Waiting to run, Needs attention, Other open jobs, Recently finished
    GET /jobs/jobs.js        the page's refresh script (a file, so a later CSP script-src 'self' keeps working)
    GET /jobs/api/summary    the same as JSON (schema cj_jobs/1, see summary()); ?html=1 adds "html": the page's
                             section fragments, rendered and escaped here (the page swaps them in every 30 s)
    POST /jobs/api/cancel    {job, reason}: sync.cancel on the job's queue (admin: any job; owner: own jobs)
    POST /jobs/api/release   {job}: sync.release of a held job (admin only)
  The two POSTs (the Gate already requires a same-origin Origin) re-check the permission on every call, take the
  share ONLY from the server's live.json queues[].share (a client-sent share/path is ignored), refuse when the
  snapshot is stale or the queue unreachable, run the SMB write in a worker thread with a timeout, and append
  audit lines through portal/audit.py (<state dir>/audit/portal.jsonl; never a key): {t, actor, action,
  target: job, detail: {reason, result, phase: requested | result}}.

Rules:
  - A job on a queue in live.json is shown from live.json only; the registry fills in its name and owner.
  - Queue order: by ready_at (the runner takes the oldest READY first, then the name) when every waiting job has
    one; otherwise live.json order with plain bullets, no numbers.
  - A snapshot older than STALE_FACTOR x its interval is stale (as the home page): a banner, and every desktop
    line says "as of <age> ago".
  - Plain words first. Commands, paths and terms (watcher, heartbeat, registry, live.json) only in each item's
    `tech` line, for admins only; non-admins get no share paths, monitor machine or raw error text.
  - Needs attention lists blocked jobs only while they are not pulled or are from the last RECENT_S; older ones
    are under Recently finished (desktop) or Other open jobs (registry) only.
  - Privacy fails closed. A predict job belongs to its registry `who`; ctx.can_see(user, who) compares the PORTAL
    ROSTER id with that `who`, so both must be the same person ids (the portal ids are the people.json ids, e.g.
    "pujan"). For a non-admin, a job is hidden (no id, name, owner, reason or hint; state and place are kept)
    when it is someone else's predict job, when a queue job is not in the registry (owner unknown), and for every
    queue job while the registry can't be read.
The only writes are the two actions above (CANCEL / READY files on the share, and the audit line).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import html
import importlib.util
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from portal import audit as PA

PREFIX = "/jobs"
ROLES = ("admin", "advanced")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TEMPLATE = HERE.parent / "templates" / "jobs.html"
SCHEMA = "cj_jobs/1"
REGISTRY_TTL_S = 10.0         # registry.current() re-reads every *.jsonl line; one read serves every request for 10 s
STALE_FACTOR = 3              # live.json older than 3 x interval_s is stale (portal/app.py is_stale, home page)
RECENT_S = 48 * 3600          # a blocked job older than this (and pulled) leaves Needs attention
N_FINISHED = 10
OPEN = ("in_progress", "ready", "blocked")
SECTIONS = ("head", "running", "queue", "attention", "review", "finished")
PRIVATE_NAME = "Job details hidden"
ACTION_TIMEOUT_S = 25.0       # the SMB write runs in a worker thread; past this the page is told the result is unknown
REASON_MAX = 300
CANCELLABLE = ("running", "queued", "held")

esc = html.escape


# ---------------------------------------------------------------- small helpers


def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _l(x) -> list:
    return x if isinstance(x, list) else []


def _s(x) -> str | None:
    return x if isinstance(x, str) and x else None


def _n(x) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x)


def _i(x) -> int | None:
    v = _n(x)
    return None if v is None else int(v)


def dur(s) -> str:
    """Seconds -> '40 s' / '12 min' / '3 h 5 min' / '2 d 4 h' (as the home page and the history Live tab)."""
    v = _n(s)
    if v is None:
        return "?"
    v = max(0, round(v))
    if v < 60:
        return f"{v} s"
    m = round(v / 60)
    if m < 60:
        return f"{m} min"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h} h" + (f" {m} min" if m else "")
    return f"{h // 24} d {h % 24} h"


def parse_utc(s) -> float | None:
    """generated_utc -> epoch seconds (a time without a zone is UTC)."""
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.timestamp()


def parse_local(s) -> float | None:
    """DONE.json `ended`, registry `logged` -> epoch seconds (a time without a zone is this machine's local time)."""
    if not isinstance(s, str) or not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def ago(ts, now) -> str | None:
    if ts is None:
        return None
    if now - ts < 48 * 3600:
        return dur(now - ts) + " ago"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def pct(done, total) -> int | None:
    d, t = _n(done), _n(total)
    if d is None or not t or t <= 0:
        return None
    return max(0, min(100, int(100 * d / t)))


# ---------------------------------------------------------------- sources
_reg_lock = threading.Lock()
_reg_cache: dict = {}            # str(root) -> (monotonic time, {id: record}, error or None)
_reg_mod = None


def _registry_module():
    """registry/registry.py, loaded once by path (as portal/app.py does; registry/ is not a package)."""
    global _reg_mod
    if _reg_mod is None:
        spec = importlib.util.spec_from_file_location("cj_jobs_registry", REPO / "registry" / "registry.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _reg_mod = mod
    return _reg_mod


def registry_jobs(root, now=None) -> tuple:
    """-> ({id: job record}, error or None, read age in s). One registry read per root per REGISTRY_TTL_S; the
    records are shared between requests, so callers never change them."""
    if root is None:
        return {}, "no registry folder configured", None
    now = time.monotonic() if now is None else now
    key = str(root)
    with _reg_lock:
        hit = _reg_cache.get(key)
        if hit is not None and 0 <= now - hit[0] < REGISTRY_TTL_S:
            return hit[1], hit[2], now - hit[0]
        if not Path(root).is_dir():
            jobs, err = {}, "registry folder not found"
        else:
            try:
                jobs = _registry_module().current("job", Path(root))
                err = None
            except Exception as e:                      # an unreadable registry must not take the page down
                jobs, err = {}, f"registry not readable ({type(e).__name__}: {str(e)[:160]})"
        _reg_cache[key] = (now, jobs, err)
        return jobs, err, 0.0


def read_live(registry) -> tuple:
    """-> (live.json dict or None, problem or None). monitor.py replaces the file atomically; on Windows a read
    that meets the replace can fail for a few ms, so a failed read is tried twice more."""
    if registry is None:
        return None, "no registry folder configured"
    p = Path(registry) / "live" / "live.json"
    err = None
    for i in range(3):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            return (doc, None) if isinstance(doc, dict) else (None, "live/live.json is not a JSON object")
        except FileNotFoundError:
            return None, "no live snapshot (live/live.json not found: is jobs/monitor.py running?)"
        except (OSError, ValueError) as e:
            err = f"live/live.json not readable ({type(e).__name__})"
            time.sleep(0.05 * (i + 1))
    return None, err


def snapshot_info(live, err, now) -> dict:
    if live is None:
        return {"present": False, "stale": True, "age_s": None, "interval_s": None, "generated_utc": None,
                "monitor": None, "problem": err}
    every = _n(live.get("interval_s")) or 30.0
    t = parse_utc(live.get("generated_utc"))
    age = None if t is None else max(0.0, now - t)
    return {"present": True, "stale": age is None or age > STALE_FACTOR * every,
            "age_s": None if age is None else round(age), "interval_s": every,
            "generated_utc": _s(live.get("generated_utc")), "monitor": _s(_d(live.get("monitor")).get("machine")),
            "problem": None if t is not None else "live.json has no readable generated_utc"}



# ---------------------------------------------------------------- words
DESK_STATE = {"running": ("running", "run"), "queued": ("waiting", "wait"), "held": ("on hold", "wait"),
              "incomplete": ("upload incomplete", "warn"), "blocked": ("error", "warn"),
              "cancelled": ("cancelled", "off"), "done": ("done", "ok"), "unknown": ("unknown", "warn")}
REG_STATE = {"in_progress": ("in progress", "run"), "ready": ("not started", "wait"), "blocked": ("blocked", "warn"),
             "done": ("done", "ok"), "cancelled": ("cancelled", "off")}
TYPE_WORD = {"revise": "review", "annotate": "labeling", "check": "check", "train": "training",
             "predict": "prediction"}
WATCH_WORD = {"alive": "working normally", "paused": "paused", "waiting for GPU": "waiting for the GPU to be free",
              "waiting": "waiting to start a job", "stale": "status updates have stopped",
              "no watcher": "not set up to run jobs", "stopped": "stopped", "unknown": "status unknown"}
WATCH_BAD = ("stale", "no watcher", "stopped", "unknown")


def hint(rec: dict) -> tuple:
    """-> (what happens next in plain words, technical note). registry/history_page.py next_action, reworded."""
    st, typ, where = rec.get("status"), rec.get("type"), _s(rec.get("where")) or "?"
    if st == "blocked":
        return _s(rec.get("blocked_reason")) or "Blocked; no reason was written down.", "registry status: blocked"
    if st == "ready":
        if typ in ("train", "predict") and where == "desktop":
            return "Not sent to the GPU computer yet.", "jobs/sync.py push <job> --bundle <zip> --share <queue>"
        return "Ready to start.", "registry status: ready"
    if st == "in_progress":
        if where == "app":
            return (f"Being {'reviewed' if typ == 'revise' else 'labeled'} in the review app.",
                    f"review app port {rec.get('port') or 8766}")
        plain = {"annotate": f"Being labeled in {where}.", "revise": f"Being reviewed in {where}.",
                 "train": f"Training on {where}.", "predict": f"Predicting on {where}.",
                 "check": f"Waiting to be scored on {where}."}.get(typ, f"In progress in {where}.")
        return plain, ("the run is imported by jobs/sync.py pull" if typ == "train" else "registry status: in_progress")
    if st == "done":
        return "Done.", None
    if st == "cancelled":
        return "Cancelled.", None
    return f"Status: {st}.", None


def eta_text(r: dict) -> str:
    if r.get("phase") == "finishing":
        return "finishing (final check)"
    bound = _n(r.get("eta_patience_bound_s"))
    if bound is not None:
        return f"up to {dur(bound)} (may stop early; all epochs {dur(r.get('eta_s'))})"
    e = _n(r.get("eta_s"))
    if e is None:
        return "not known yet"
    return dur(e) + (" (time limit)" if r.get("eta_capped") else "")


def may_cancel(user, can_see, reg, reg_err, job_id) -> bool:
    """Who may cancel a job: an admin, or its registry owner. One rule for the buttons (summary) and the POST (act);
    an unreadable registry or a job with no owner is admin-only."""
    if (user or {}).get("role") == "admin":
        return True
    uid = (user or {}).get("id")
    return bool(reg_err is None and uid and job_id and can_see(user, _d(_d(reg).get(job_id)).get("who")))


# ---------------------------------------------------------------- summary
def summary(cfg, user: dict, can_see, now=None) -> dict:
    """Everything the page shows, as data (schema cj_jobs/1). `can_see(user, owner)` is ctx.can_see."""
    now = time.time() if now is None else now
    admin = (user or {}).get("role") == "admin"
    live, live_err = read_live(getattr(cfg, "registry", None))
    snap = snapshot_info(live, live_err, now)
    reg, reg_err, reg_age = registry_jobs(getattr(cfg, "registry", None))
    stale = bool(snap["present"] and snap["stale"])
    asof = None
    if stale:
        asof = f"as of {dur(snap['age_s'])} ago" if snap["age_s"] is not None else "as of an unknown time"

    def private(job_id, kind=None, queue_job=False) -> bool:
        """Fail closed (module docstring): hidden unless we can tell it is not someone else's."""
        if admin:
            return False
        rec = reg.get(job_id) if job_id else None
        if queue_job and (reg_err is not None or rec is None):
            return True                                 # owner unknown
        rec = _d(rec)
        if kind != "predict" and rec.get("type") != "predict":
            return False
        return not can_see(user, rec.get("who"))

    def acts(job_id, state, requested=False) -> dict:
        """Which buttons this user gets for a queue job (the POST re-checks all of it): cancel for an admin or the
        job's registry owner, release for an admin; none on a stale snapshot, a hidden job or a cancel in flight."""
        if not job_id or stale or private(job_id, None, True):
            return {"can_cancel": False, "can_release": False}
        mine = may_cancel(user, can_see, reg, reg_err, job_id)
        return {"can_cancel": bool(mine and state in CANCELLABLE and not requested),
                "can_release": bool(admin and state == "held")}

    def ident(job_id, kind=None, queue_job=True) -> dict:
        """id + display name (the registry's name when it has one), or hidden."""
        if job_id and private(job_id, kind, queue_job):
            return {"id": None, "name": PRIVATE_NAME, "private": True}
        name = _s(_d(reg.get(job_id)).get("name")) if job_id else None
        return {"id": job_id, "name": name or job_id or "?", "private": False}

    queues, running, queue, attention, finished = [], [], [], [], []
    seen = set()
    for qi, q in enumerate(_l((live or {}).get("queues"))):
        q = _d(q)
        w = _d(q.get("watcher"))
        share, detail = _s(q.get("share")), _s(q.get("detail"))
        where = _s(w.get("machine")) or (share if admin else None) or f"GPU computer {qi + 1}"
        reach = _s(q.get("reach")) or "unknown"
        jobs = [j for j in (_d(j) for j in _l(q.get("jobs"))) if _s(j.get("job"))]
        fins = [f for f in (_d(f) for f in _l(q.get("finished"))) if _s(f.get("job"))]
        r = _d(q.get("running"))
        seen.update(j["job"] for j in jobs)
        seen.update(f["job"] for f in fins)
        if _s(r.get("job")):
            seen.add(r["job"])
        waiting = [j for j in jobs if j.get("state") == "queued"]
        qinfo = {"name": where, "share": share, "reach": reach, "detail": detail, "watcher": None}
        queues.append(qinfo)
        if reach != "ok":
            attention.append({"kind": "offline", "chip": "not reachable", **ident(None), "name": where,
                              "queue": where, "source": "desktop", "title": f"Can't reach {where}",
                              "reason": "The laptop can't see this computer's job list right now, so its jobs are "
                                        "unknown.", "tech": f"queue {reach}" + (f": {detail}" if detail else ""),
                              "stale": stale})
            continue
        disp = _s(w.get("display")) or "unknown"
        wdet, hb = _s(w.get("detail")), _n(w.get("age_s"))
        qinfo["watcher"] = {"display": disp, "plain": WATCH_WORD.get(disp, disp), "state": _s(w.get("state")),
                            "detail": wdet, "age_s": hb, "stale": bool(w.get("stale"))}
        if disp in WATCH_BAD:
            attention.append({"kind": "watcher", "chip": "no updates", **ident(None), "name": where, "queue": where,
                              "source": "desktop",
                              "title": f"{where}: {WATCH_WORD.get(disp, disp)}",
                              "reason": (f"Last update {dur(hb)} ago." if hb is not None else "No update received.")
                              + (" Waiting jobs will not start until it is back." if waiting else ""),
                              "tech": f"watcher {disp}" + (f"; {wdet}" if wdet else "")
                              + "; check jobs/watch.py on the desktop", "stale": stale})
        elif disp == "paused" and waiting:
            attention.append({"kind": "paused", "chip": "paused", **ident(None), "name": where, "queue": where,
                              "source": "desktop", "title": f"{where} is paused",
                              "reason": f"{len(waiting)} waiting job(s) will not start until it is resumed.",
                              "tech": (wdet or "STOP is set") + "; jobs/sync.py start resumes it", "stale": stale})
        if _s(r.get("job")):
            kind = "predict" if r.get("kind") == "predict" else "train"
            if kind == "predict":
                done, total, unit = _i(r.get("n_done")), _i(r.get("n_total")), "images"
            else:
                done, total, unit = _i(r.get("epoch")), _i(r.get("epochs")), "epochs"
            phase = _s(r.get("phase")) or ("predicting" if kind == "predict" else "training")
            hung = bool(r.get("hung"))
            hung_text = hung_tech = None
            if hung and kind == "train":
                hung_text = (f"No progress for {dur(r.get('last_epoch_age_s'))} (normally a new epoch every "
                             f"{dur(r.get('epoch_median_s'))}).")
                hung_tech = "check train.log on the desktop"
            elif hung:
                hung_text, hung_tech = "No progress recently.", "check the job's log on the desktop"
            idn = ident(r["job"], kind)
            item = {**idn, "queue": where, "kind": kind, "phase": phase, "done": done, "total": total, "unit": unit,
                    "pct": pct(done, total), "eta_s": _n(r.get("eta_s")),
                    "eta_text": eta_text({**r, "kind": kind, "phase": phase}), "elapsed_s": _n(r.get("elapsed_s")),
                    "hung": hung, "hung_text": hung_text, "cancel_requested": bool(r.get("cancel_requested")),
                    **acts(r["job"], "running", bool(r.get("cancel_requested"))),
                    "map50_95": _n(r.get("map50_95")) if kind == "train" else None,
                    "watcher": {"display": disp, "plain": WATCH_WORD.get(disp, disp), "age_s": hb},
                    "tech": " · ".join(x for x in (f"run {r['run']}" if _s(r.get("run")) and not idn["private"]
                                                   else None, f"watcher {disp}", hung_tech) if x),
                    "stale": stale}
            running.append(item)
            if hung:
                attention.append({"kind": "hung", "chip": "may be stuck", **idn, "queue": where, "source": "desktop",
                                  "title": "May be stuck", "reason": hung_text, "tech": hung_tech, "stale": stale})
        times = [parse_local(j.get("ready_at")) for j in waiting]
        numbered = bool(waiting) and all(t is not None for t in times)
        if numbered:
            waiting = [j for _, j in sorted(zip(times, waiting), key=lambda tj: (tj[0], tj[1]["job"]))]
        pos = 0
        for j in waiting + [j for st in ("held", "incomplete") for j in jobs if j.get("state") == st]:
            state = j["state"]
            if state == "queued":
                pos += 1
            note = {"queued": f"Waiting its turn on {where}.",
                    "held": f"On hold on {where}: starts when someone releases it.",
                    "incomplete": f"The upload to {where} did not finish."}[state]
            tech = {"queued": f"ready since {j['ready_at']}" if _s(j.get("ready_at")) else
                    "no ready_at in live.json: order unknown",
                    "held": f"jobs/sync.py release {j['job']}", "incomplete": "no READY file in q/<job>"}[state]
            if state == "queued" and w.get("job") == j["job"] and disp != "alive":
                note = f"Next up, but {where} is {WATCH_WORD.get(disp, disp)}."
                tech += f"; watcher {disp}" + (f": {wdet}" if wdet else "")
            if j.get("cancel_requested"):
                note += " Cancel requested."
            label, tone = DESK_STATE[state]
            queue.append({**ident(j["job"]), "queue": where, "source": "desktop", "state": state, "label": label,
                          "tone": tone, "position": pos if numbered and state == "queued" else None,
                          "ready_at": _s(j.get("ready_at")), "note": note, "tech": tech, "stale": stale,
                          **acts(j["job"], state, bool(j.get("cancel_requested")))})
        ended = {f["job"]: parse_local(f.get("ended")) for f in fins}
        for j in jobs:
            st, jid = j.get("state"), j["job"]
            if st == "unknown":
                attention.append({"kind": "unknown", "chip": "unknown", **ident(jid), "queue": where,
                                  "source": "desktop", "title": "Its state can't be read",
                                  "reason": "An admin needs to look at it.", "tech": _s(j.get("reason")),
                                  "stale": stale})
            elif st == "blocked":
                ts = ended.get(jid)
                if j.get("pulled") and not (ts is not None and now - ts <= RECENT_S):
                    continue                            # old and already pulled: Recently finished only
                attention.append({"kind": "blocked", "chip": "error", **ident(jid), "queue": where,
                                  "source": "desktop", "title": "Stopped with an error",
                                  "reason": "Its results have been copied to the laptop." if j.get("pulled") else
                                  "Its results have not been copied back to the laptop yet.",
                                  "tech": _s(j.get("reason")) or "no reason recorded", "stale": stale})
        for f in fins:
            st = _s(f.get("status")) or "unknown"
            label, tone = DESK_STATE.get(st, (st, "wait"))
            m, ts = _d(f.get("metrics")), parse_local(f.get("ended"))
            finished.append({**ident(f["job"]), "queue": where, "source": "desktop", "status": st, "label": label,
                             "tone": tone, "ended": _s(f.get("ended")), "ts": ts, "when": ago(ts, now),
                             "pulled": bool(f.get("pulled")), "map50_95": _n(m.get("mask_map50_95")),
                             "map50": _n(m.get("mask_map50")), "best_epoch": _i(f.get("best_epoch")),
                             "tech": _s(f.get("reason")), "stale": stale})

    review = []
    order = {"in_progress": 0, "blocked": 1, "ready": 2}
    for jid, rec in reg.items():
        rec = _d(rec)
        if jid in seen:                                 # on a queue: live.json is the truth for it
            continue
        st, typ = _s(rec.get("status")) or "?", _s(rec.get("type")) or "?"
        label, tone = REG_STATE.get(st, (st, "wait"))
        idn = ident(jid, queue_job=False)
        if st in OPEN:
            prog = _d(rec.get("progress"))
            p = ({"done": _i(prog.get("done")), "total": _i(prog.get("total")), "unit": _s(prog.get("unit")) or ""}
                 if prog else None)
            plain, tech = hint(rec) if not idn["private"] else ("Details hidden.", None)
            review.append({**idn, "source": "registry", "type": typ, "type_label": TYPE_WORD.get(typ, typ),
                           "where": _s(rec.get("where")), "status": st, "label": label, "tone": tone,
                           "who": None if idn["private"] else _s(rec.get("who")), "progress": p,
                           "pct": pct(p["done"], p["total"]) if p else None, "hint": plain, "tech": tech,
                           "logged": _s(rec.get("logged"))})
            ts = parse_local(rec.get("logged"))
            if st == "blocked" and ts is not None and now - ts <= RECENT_S:
                attention.append({"kind": "blocked", "chip": "blocked", **idn, "queue": None, "source": "registry",
                                  "title": f"Blocked {TYPE_WORD.get(typ, typ)} job",
                                  "reason": None if idn["private"] else
                                  (_s(rec.get("blocked_reason")) or "No reason was written down."),
                                  "tech": f"registry: {typ} job in {_s(rec.get('where')) or '?'}", "stale": False})
        elif st in ("done", "cancelled"):
            ts = parse_local(rec.get("logged"))
            finished.append({**idn, "queue": None, "source": "registry", "type": typ, "status": st, "label": label,
                             "tone": tone, "ended": _s(rec.get("logged")), "ts": ts, "when": ago(ts, now),
                             "pulled": None, "map50_95": None, "map50": None, "best_epoch": None,
                             "tech": "registry", "stale": False})
    review.sort(key=lambda x: (order.get(x["status"], 9), x["logged"] or "", x["id"] or ""))
    finished.sort(key=lambda x: x["ts"] or 0, reverse=True)
    finished = finished[:N_FINISHED]
    kinds = {"offline": 0, "watcher": 1, "paused": 2, "hung": 3, "blocked": 4, "unknown": 5}
    attention.sort(key=lambda x: kinds.get(x["kind"], 9))
    reg_info = {"present": reg_err is None, "problem": reg_err, "n_jobs": len(reg),
                "read_age_s": None if reg_age is None else round(reg_age, 1)}
    if not admin:                                       # no commands, paths, machine names or raw errors
        for x in (*running, *queue, *attention, *review, *finished):
            x["tech"] = None
        for q in queues:
            q["share"] = q["detail"] = None
            if q["watcher"]:
                q["watcher"]["detail"] = None
        snap = {**snap, "monitor": None, "problem": None}
        if reg_err is not None:
            reg_info["problem"] = "not readable"
    return {"schema": SCHEMA, "now_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
            "pkgs": review_packages(cfg) if admin else None, "admin": admin, "asof": asof, "snapshot": snap, "registry": reg_info, "queues": queues,
            "running": running, "queue": queue, "attention": attention, "review": review, "finished": finished,
            "counts": {"running": len(running), "queued": sum(1 for x in queue if x["state"] == "queued"),
                       "held": sum(1 for x in queue if x["state"] != "queued"), "attention": len(attention),
                       "review_open": len(review), "finished": len(finished)}}


def review_packages(cfg) -> list:
    """Admin only: lab labeling and compare packages under the review root with live progress (cards decided / total,
    images reviewed / total, last edit and by whom, ready / in progress / applied). Computed on view by portal.titles
    (cached by the edits folder's mtime); never prediction packages, never decoys.json."""
    from portal import names as NM
    from portal import titles as TT
    from portal.modules import data as DATA
    from portal.modules import results as RES
    root = RES.review_root(cfg)
    if root is None:
        return []
    records, _err = DATA.read_registry(getattr(cfg, "registry", None))
    rows = []
    for p in TT.scan_packages(root, records):
        last = p["last"]
        rows.append({"id": p["id"], "title": p["title"], "kind": p["kind"], "cards": p["cards"],
                     "decided": p["decided"], "images": p["images"], "reviewed": p["reviewed"],
                     "last": NM.when(datetime.fromtimestamp(last).isoformat()) if last else None, "_ts": last or 0,
                     "who": p["who"], "status": p["status"], "applied_by": p["applied_by"]})
    rows.sort(key=lambda r: (r["status"] == "applied", -r["_ts"], r["id"]))
    for r in rows:
        del r["_ts"]
    return rows


# ---------------------------------------------------------------- html (every text through esc)
def chip(label, tone) -> str:
    return f'<span class="cj-jobs-chip cj-jobs-t-{esc(tone)}">{esc(label)}</span>'


def sec_head(key, title, n=None) -> str:
    count = "" if n is None else f' <span class="cj-jobs-n">{int(n)}</span>'
    return f'<h2 class="cj-sec" id="cj-jobs-h-{key}">{esc(title)}{count}</h2>'


def empty(text) -> str:
    return f'<p class="cj-mut cj-jobs-empty">{esc(text)}</p>'


def tech(x) -> str:
    """The admin-only second line (summary() leaves `tech` None for everyone else)."""
    t = x.get("tech") if isinstance(x, dict) else x
    return f'<span class="cj-mut cj-wrap cj-jobs-tech">{esc(t)}</span>' if t else ""


def name_line(x) -> str:
    sub = ""
    if x.get("id") and x["id"] != x.get("name"):
        sub = f'<span class="cj-mut cj-wrap cj-jobs-id">{esc(x["id"])}</span>'
    return f'<span class="cj-jobs-ttl"><b class="cj-wrap cj-jobs-name">{esc(x.get("name") or "?")}</b>{sub}</span>'


def kv(pairs) -> str:
    rows = "".join(f"<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>" for k, v in pairs if v is not None)
    return f'<dl class="cj-jobs-kv">{rows}</dl>' if rows else ""


def bar(done, total, unit_label) -> str:
    if done is None or not total:
        return ""
    v = max(0, min(int(done), int(total)))
    return (f'<progress class="cj-jobs-bar" max="{int(total)}" value="{v}" '
            f'aria-label="{esc(unit_label)} {int(done)} of {int(total)}"></progress>')


def actions_html(x: dict) -> str:
    """The buttons for one job (summary() decided who gets them). Cancel is two steps built into the page: the
    button reveals a form (reason + 'Yes, cancel'); nothing is sent before that. Script: /jobs/jobs.js."""
    if not x.get("id") or not (x.get("can_cancel") or x.get("can_release")):
        return ""
    job, out = esc(x["id"], quote=True), []
    if x.get("can_release"):
        out.append(f'<button type="button" class="cj-jobs-btn" data-act="release" data-job="{job}">'
                   "Release (let it start)</button>")
    if x.get("can_cancel"):
        out.append(f'<button type="button" class="cj-jobs-btn cj-jobs-btn-warn" data-act="ask" data-job="{job}">'
                   "Cancel job&hellip;</button>"
                   f'<form class="cj-jobs-cf" data-job="{job}" hidden><p class="cj-jobs-cq"><b>Cancel this job?</b> '
                   "A running job is stopped (its partial run is kept); a waiting one is never started.</p>"
                   f'<label class="cj-jobs-lbl">Reason (required)<input class="cj-jobs-in" name="reason" required '
                   f'maxlength="{REASON_MAX}" autocomplete="off"></label>'
                   '<span class="cj-jobs-btns"><button type="submit" class="cj-jobs-btn cj-jobs-btn-warn">'
                   'Yes, cancel</button><button type="button" class="cj-jobs-btn" data-act="back">'
                   "No, keep it</button></span></form>")
    return f'<div class="cj-jobs-act">{"".join(out)}<p class="cj-jobs-res" role="status" hidden></p></div>'


def render(s: dict) -> dict:
    """summary() -> {section: html fragment}; the page's placeholders and the refresh both use these."""
    snap, out, asof, admin = s["snapshot"], {}, s["asof"], s["admin"]
    sfx = f" ({asof})" if asof else ""

    # head: when, banners, counts
    head = ""
    if snap["present"] and not snap["stale"]:
        head += f'<p class="cj-mut cj-jobs-snap">{esc("Updated " + dur(snap["age_s"]) + " ago.")}</p>'
    elif snap["present"]:
        head += ('<p class="cj-jobs-banner" role="status"><b>Status updates have stopped.</b> '
                 + esc(f"The last one came {dur(snap['age_s'])} ago, so the GPU computer jobs below are as of then, "
                       "not now.") + "</p>")
    else:
        head += ('<p class="cj-jobs-banner" role="status"><b>No status from the GPU computer.</b> '
                 + esc("What runs there is unknown; the other jobs are still shown.") + "</p>")
    if not s["registry"]["present"]:
        head += f'<p class="cj-jobs-banner" role="status">{esc("The lab job list can not be read right now.")}</p>'
    if admin:
        bits = [f"live.json from jobs/monitor.py on {snap['monitor'] or '?'}, every {dur(snap['interval_s'])}"
                if snap["present"] else (snap["problem"] or "no live.json"),
                f"registry: {s['registry']['n_jobs']} job(s)" if s["registry"]["present"] else
                f"registry: {s['registry']['problem']}"]
        if snap["present"] and snap["stale"]:
            bits.insert(1, "jobs/monitor.py has stopped or cannot write")
        head += f'<p class="cj-jobs-snap">{tech("; ".join(bits))}</p>'
    c = s["counts"]
    links = [("running", c["running"], "running", ""), ("queue", c["queued"], "waiting", ""),
             ("attention", c["attention"], "need attention", " cj-jobs-t-warn" if c["attention"] else ""),
             ("review", c["review_open"], "other open", "")]
    head += ('<nav class="cj-jobs-counts" aria-label="Jobs summary">'
             + "".join(f'<a class="cj-jobs-count{cls}" href="#cj-jobs-s-{k}"><b>{int(n)}</b> {esc(w)}</a>'
                       for k, n, w, cls in links) + "</nav>")
    out["head"] = head

    # running now
    parts = [sec_head("running", "Running now" if not asof else f"Running ({asof})", len(s["running"]))]
    wl = []
    for q in s["queues"]:
        if q["reach"] != "ok":
            wl.append(f'<li><span class="cj-dot down" aria-hidden="true"></span><b>{esc(q["name"])}</b> '
                      f'<span class="cj-mut">{esc("not reachable" + sfx)}</span>'
                      + (tech(f"queue {q['reach']}" + (f": {q['detail']}" if q["detail"] else "")) if admin else "")
                      + "</li>")
            continue
        w = q["watcher"] or {}
        dot = "unk" if asof else ("up" if w.get("display") == "alive" else "down")
        last = f" · last update {dur(w.get('age_s'))} ago" if w.get("age_s") is not None else ""
        t = (f"watcher {w.get('display')}" + (f", heartbeat {dur(w.get('age_s'))} ago" if w.get("age_s") is not None
                                                else "") + (f", {w['detail']}" if w.get("detail") else ""))
        wl.append(f'<li><span class="cj-dot {dot}" aria-hidden="true"></span><b>{esc(q["name"])}</b> '
                  f'<span class="cj-mut">{esc(str(w.get("plain") or "?") + last + sfx)}</span>'
                  + (tech(t) if admin else "") + "</li>")
    if wl:
        parts.append(f'<ul class="cj-list cj-jobs-machines">{"".join(wl)}</ul>')
    cards = []
    for r in s["running"]:
        unit = "Images" if r["kind"] == "predict" else "Epoch"
        prog = (f"{unit} {r['done']} / {r['total']}" if r["done"] is not None and r["total"]
                else ("Setting up: nothing finished yet" if r["phase"] == "setting up" else f"{unit} ?"))
        tags = chip(asof if asof else r["phase"], "wait" if asof else "run")
        if r["hung"]:
            tags += chip("may be stuck", "warn")
        w = r["watcher"]
        body = [f'<div class="cj-jobs-top">{name_line(r)}<span class="cj-jobs-tags">{tags}</span></div>',
                f'<div class="cj-jobs-prog"><span>{esc(prog)}</span>'
                + (f'<span class="cj-mut">{r["pct"]} %</span>' if r["pct"] is not None else "") + "</div>",
                bar(r["done"], r["total"], unit),
                kv([("Time left", r["eta_text"]),
                    ("Running for", dur(r["elapsed_s"]) if r["elapsed_s"] is not None else None),
                    ("On", r["queue"]),
                    ("Last update", (f"{dur(w['age_s'])} ago" if w["age_s"] is not None else "unknown")
                     + ("" if w["display"] == "alive" else f" ({w['plain']})")),
                    ("Val mAP50-95", f"{r['map50_95']:.3f}" if r["map50_95"] is not None else None)])]
        if r["hung_text"]:
            body.append(f'<p class="cj-jobs-warn">{esc(r["hung_text"])}</p>')
        if r["cancel_requested"]:
            body.append(f'<p class="cj-jobs-warn">{esc("Cancel requested: it stops within a few seconds.")}</p>')
        body.append(tech(r))
        body.append(actions_html(r))
        cards.append(f'<article class="cj-card cj-jobs-item">{"".join(body)}</article>')
    if cards:
        parts.append(f'<div class="cj-jobs-grid">{"".join(cards)}</div>')
    elif not s["snapshot"]["present"]:
        parts.append(empty("Unknown: no status from the GPU computer."))
    elif asof:
        parts.append(empty(f"Nothing was running {asof}."))
    else:
        parts.append(empty("Nothing is running on the GPU computer."))
    out["running"] = "".join(parts)

    # waiting to run
    parts = [sec_head("queue", "Waiting to run", len(s["queue"]))]
    rows = []
    for x in s["queue"]:
        num = (f'<span class="cj-jobs-pos">{x["position"]}</span>' if x["position"] else
               '<span class="cj-jobs-pos cj-jobs-t-off" aria-hidden="true">•</span>')
        rows.append(f'<li class="cj-jobs-row">{num}<span class="cj-jobs-main">{name_line(x)}'
                    f'<span class="cj-mut">{esc(x["note"] + sfx)}</span>{tech(x)}{actions_html(x)}</span>'
                    f'{chip(x["label"], x["tone"])}</li>')
    parts.append(f'<ol class="cj-card cj-jobs-rows">{"".join(rows)}</ol>' if rows else
                 empty(f"Nothing was waiting {asof}." if asof else "Nothing is waiting." if s["snapshot"]["present"]
                       else "Unknown: no status from the GPU computer."))
    out["queue"] = "".join(parts)

    # needs attention
    parts = [sec_head("attention", "Needs attention", len(s["attention"]))]
    cards = []
    for a in s["attention"]:
        where = ""
        if a["kind"] not in ("offline", "watcher", "paused") and a.get("queue"):
            where = f" · on {a['queue']}"
        if a["source"] == "desktop" and asof:
            where += f" · {asof}"
        reason = f'<p class="cj-jobs-reason cj-wrap">{esc(a["reason"])}</p>' if a.get("reason") else ""
        cards.append(f'<article class="cj-card cj-jobs-item cj-jobs-att"><div class="cj-jobs-top">{name_line(a)}'
                     f'{chip(a["chip"], "warn")}</div><p class="cj-jobs-why"><b>{esc(a["title"])}</b>'
                     f'<span class="cj-mut">{esc(where)}</span></p>{reason}{tech(a)}</article>')
    parts.append(f'<div class="cj-jobs-grid">{"".join(cards)}</div>' if cards else empty("Nothing needs attention."))
    out["attention"] = "".join(parts)

    # other open jobs (open registry jobs not on a queue)
    parts = [sec_head("review", "Other open jobs", len(s["review"]))]
    cards = []
    for x in s["review"]:
        p, prog = x["progress"], ""
        if p and (p["total"] or p["done"] is not None):
            text = (f"{p['done'] if p['done'] is not None else '?'} / {p['total']} {p['unit']}" if p["total"]
                    else f"{p['done']} {p['unit']}").strip()
            prog = (f'<div class="cj-jobs-prog"><span>{esc(text)}</span>'
                    + (f'<span class="cj-mut">{x["pct"]} %</span>' if x["pct"] is not None else "")
                    + "</div>" + bar(p["done"], p["total"], p["unit"] or "progress"))
        meta = " · ".join(v for v in (x["type_label"], x["where"] and f"in {x['where']}",
                                      x["who"] and f"by {x['who']}") if v)
        cards.append(f'<article class="cj-card cj-jobs-item"><div class="cj-jobs-top">{name_line(x)}'
                     f'{chip(x["label"], x["tone"])}</div><div class="cj-mut">{esc(meta)}</div>{prog}'
                     f'<p class="cj-jobs-hint">{esc(x["hint"])}</p>{tech(x)}</article>')
    if cards:
        parts.append(f'<div class="cj-jobs-grid">{"".join(cards)}</div>')
    else:
        parts.append(empty("No other open job." if s["registry"]["present"] else
                           "Unknown: the lab job list can not be read."))
    out["review"] = "".join(parts)

    # recently finished
    parts = [sec_head("finished", "Recently finished", len(s["finished"]))]
    rows = []
    for x in s["finished"]:
        bits = [x["when"] or "time unknown"]
        if x["source"] == "desktop":
            if x["map50_95"] is not None:
                bits.append(f"mask mAP50-95 {x['map50_95']:.3f}")
            if x["best_epoch"] is not None:
                bits.append(f"best epoch {x['best_epoch']}")
            bits.append(f"on {x['queue']}")
            bits.append("results on the laptop" if x["pulled"] else "results not copied back yet")
        else:
            bits.append(TYPE_WORD.get(x.get("type"), x.get("type") or "?"))
        rows.append(f'<li class="cj-jobs-row"><span class="cj-jobs-main">{name_line(x)}'
                    f'<span class="cj-mut">{esc(" · ".join(bits))}</span>{tech(x)}</span>'
                    f'{chip(x["label"], x["tone"])}</li>')
    parts.append(f'<ul class="cj-card cj-jobs-rows">{"".join(rows)}</ul>' if rows else empty("Nothing finished yet."))
    out["finished"] = "".join(parts)
    if admin and s.get("pkgs") is not None:
        out["finished"] += render_packages(s["pkgs"])
    return out


PKG_STATUS = {"ready": ("Ready", "wait"), "in_progress": ("In progress", "run"), "applied": ("Applied", "ok")}


def render_packages(pkgs: list) -> str:
    parts = [sec_head("pkgs", "Review packages", len(pkgs))]
    items = []
    for r in pkgs:
        label, tone = PKG_STATUS.get(r["status"], (r["status"], "wait"))
        status = chip(label, tone)
        if r["status"] == "applied" and r.get("applied_by"):
            status += (f' <a href="/data/labels/{esc(quote(str(r["applied_by"]), safe=""))}">'
                       f'{esc(str(r["applied_by"]))}</a>')
        last = f"last edit {r['last']}" + (f" by {r['who']}" if r.get("who") else "") if r["last"] else "no edits yet"
        meta = (f"{r['decided']} of {r['cards']} cards decided", f"{r['reviewed']} of {r['images']} images reviewed",
                last)
        items.append(f'<li class="cj-vis-pkg"><div class="cj-vis-pkg-top"><span><b>{esc(r["title"])}</b>'
                     f'<span class="cj-mut cj-vis-id">{esc(r["id"])}</span></span><span>{status}</span></div>'
                     f'<div class="cj-vis-pkg-meta cj-mut">' + "".join(f"<span>{esc(m)}</span>" for m in meta)
                     + "</div>"
                     + bar(r["decided"], r["cards"], "cards decided") + "</li>")
    parts.append(f'<ul class="cj-card cj-vis-pkgs">{"".join(items)}</ul>' if items
                 else empty("No lab labeling or compare packages."))
    return "".join(parts)


_PH = re.compile(r"\{\{(head|running|queue|attention|review|finished)\}\}")


def page_body(fragments: dict) -> str:
    """jobs.html with each placeholder replaced in ONE pass, so text inside a fragment is never re-scanned."""
    tpl = TEMPLATE.read_text(encoding="utf-8")
    return _PH.sub(lambda m: fragments.get(m.group(1), ""), tpl)


# ---------------------------------------------------------------- the page's script (served as /jobs/jobs.js)
JS = r"""// Jobs page (portal/modules/jobs.py): cancel / release buttons (two-step cancel form, no dialogs); refresh every 30 s, and when the tab comes back, GET api/summary?html=1
// (same origin) and swap in the section fragments. The fragments are rendered and escaped by the server, exactly
// as on first load; this script never builds HTML from data. Opened as a file, it does nothing.
(function () {
  'use strict';
  if (location.protocol !== 'http:' && location.protocol !== 'https:') return;
  var IDS = {head: 'cj-jobs-head', running: 'cj-jobs-s-running', queue: 'cj-jobs-s-queue',
             attention: 'cj-jobs-s-attention', review: 'cj-jobs-s-review', finished: 'cj-jobs-s-finished'};
  var busy = false;
  function note(text) {
    var err = document.getElementById('cj-jobs-err');
    if (!err) return;
    err.textContent = text || '';
    err.hidden = !text;
  }
  var inflight = 0, clearTimer = null;
  function editing() {   // a refresh must not wipe a cancel form being filled in, or a request in flight
    return inflight > 0 || !!document.querySelector('.cj-jobs-cf:not([hidden])');
  }
  function swap(frag) {
    Object.keys(IDS).forEach(function (k) {
      var box = document.getElementById(IDS[k]);
      if (!box || typeof frag[k] !== 'string') return;
      var t = document.createElement('template');
      t.innerHTML = frag[k];
      box.replaceChildren(t.content);
    });
  }
  function load() {
    if (busy || document.hidden || editing()) return;
    busy = true;
    fetch('api/summary?html=1', {cache: 'no-store', credentials: 'same-origin'})
      .then(function (r) {
        if (!r.ok) throw new Error('status ' + r.status);
        return r.json();
      })
      .then(function (j) { swap((j && j.html) || {}); note(''); })
      .catch(function (e) { note('Could not refresh (' + e.message + '); showing the last loaded state.'); })
      .then(function () { busy = false; });
  }
  function say(act, text, bad) {
    var res = act.querySelector('.cj-jobs-res');
    if (!res) return;
    res.textContent = text;
    res.hidden = !text;
    res.classList.toggle('cj-jobs-res-bad', !!bad);
    if (clearTimer) clearTimeout(clearTimer);
    if (text && inflight === 0) clearTimer = setTimeout(function () { res.textContent = ''; res.hidden = true; }, 10000);
  }
  function post(act, path, body, btn) {
    btn.disabled = true;
    inflight += 1;
    say(act, 'Sending...', false);
    fetch(path, {method: 'POST', credentials: 'same-origin', cache: 'no-store',
                 headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
      .then(function (r) { return r.json().catch(function () { return {}; }).then(function (j) { return [r.status, j]; }); })
      .then(function (sj) {
        inflight = Math.max(0, inflight - 1);
        var ok = sj[0] === 200 && sj[1].ok;
        say(act, sj[1].message || (ok ? 'Done.' : 'It did not work (status ' + sj[0] + ').'), !ok);
        if (ok) {
          act.querySelectorAll('button').forEach(function (b) { b.hidden = true; });
          var f = act.querySelector('form'); if (f) f.hidden = true;
        } else { btn.disabled = false; }
      })
      .catch(function () { inflight = Math.max(0, inflight - 1); say(act, 'Could not reach the portal. Nothing was confirmed; try again.', true); btn.disabled = false; });
  }
  document.addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('button[data-act]');
    var act = b && b.closest('.cj-jobs-act');
    if (!act) return;
    var form = act.querySelector('form.cj-jobs-cf');
    if (b.dataset.act === 'ask' && form) {
      form.hidden = false; b.hidden = true; say(act, '', false);
      var inp = form.querySelector('input'); if (inp) inp.focus();
    } else if (b.dataset.act === 'back' && form) {
      form.hidden = true; say(act, '', false);
      var ask = act.querySelector('button[data-act="ask"]'); if (ask) ask.hidden = false;
    } else if (b.dataset.act === 'release') {
      post(act, 'api/release', {job: b.dataset.job}, b);
    }
  });
  document.addEventListener('submit', function (e) {
    var form = e.target.closest && e.target.closest('form.cj-jobs-cf');
    if (!form) return;
    e.preventDefault();
    var reason = form.elements.reason.value.trim();
    var act = form.closest('.cj-jobs-act');
    if (!reason) { say(act, 'Please type a reason first.', true); return; }
    post(act, 'api/cancel', {job: form.dataset.job, reason: reason}, form.querySelector('button[type=submit]'));
  });
  setInterval(load, 30000);
  document.addEventListener('visibilitychange', function () { if (!document.hidden) load(); });
})();
"""


# ---------------------------------------------------------------- actions (cancel / release)
log = logging.getLogger("cj.jobs")
_sync_mod = None
_ids_mod = None
_job_locks: dict = {}
_job_locks_guard = threading.Lock()
POOL_SIZE = 16
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=POOL_SIZE, thread_name_prefix="cj-jobs-act")
_slots = threading.BoundedSemaphore(POOL_SIZE)        # a request that finds none free is told "busy", not queued
CTRL = re.compile(r"[\x00-\x1f\x7f\u0085\u2028\u2029]")
MAX_JOB_ID = 120
MAX_BODY = 8192
UNC = re.compile(r"^\\\\[^\\/]+\\[^\\/]+")
LOCAL_SHARES = tuple(x for x in os.environ.get("CJ_JOBS_LOCAL_SHARES", "").split(os.pathsep) if x)


def sync_module():
    """jobs/sync.py, loaded once by path (jobs/ is not a package; sync.py puts its own folders on sys.path)."""
    global _sync_mod
    if _sync_mod is None:
        spec = importlib.util.spec_from_file_location("cj_jobs_sync", REPO / "jobs" / "sync.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _sync_mod = mod
    return _sync_mod


def valid_job_id(job) -> bool:
    """registry/ids.py is_id("job", ...): job_<yyyymmdd>_<name>, so no slash, dot or '..'."""
    global _ids_mod
    if _ids_mod is None:
        spec = importlib.util.spec_from_file_location("cj_jobs_ids", REPO / "registry" / "ids.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ids_mod = mod
    return isinstance(job, str) and len(job) <= MAX_JOB_ID and _ids_mod.is_id("job", job)


def share_ok(share) -> bool:
    """live.json's share must look like a UNC path (\\\\host\\share), or be one of $CJ_JOBS_LOCAL_SHARES (a local
    queue folder, e.g. the laptop's own queue)."""
    return bool(UNC.match(share)) or share in LOCAL_SHARES


def audit_path(cfg) -> Path:
    return PA.audit_path(cfg.state_dir)


def audit(cfg, actor, action, job, reason, result):
    """One line through portal/audit.py. Raises OSError if it can't be written. A request writes "requested" BEFORE
    the share is touched (write-ahead: no line, nothing done), then a second line with the outcome."""
    PA.audit(cfg.state_dir, actor, action, job,
             {"reason": reason, "result": result, "phase": "requested" if result == "requested" else "result"})


def job_lock(job) -> threading.Lock:
    with _job_locks_guard:
        return _job_locks.setdefault(job, threading.Lock())


def act(cfg, user: dict, can_see, action: str, job, reason, now=None) -> tuple:
    """Runs in a worker thread. -> (http status, {"ok": bool, "message": str}). Every check is made here, on the
    server, from live.json and the registry; nothing the client sends but the job id and the reason is used."""
    now = time.time() if now is None else now
    admin = (user or {}).get("role") == "admin"
    uid = (user or {}).get("id")

    def no(code, msg):
        return code, {"ok": False, "message": msg}
    if not valid_job_id(job):
        return no(400, "That job id is not valid.")
    if action == "release" and not admin:
        return no(403, "Only an admin can release a held job.")
    if action == "cancel":
        reg, reg_err, _ = registry_jobs(getattr(cfg, "registry", None))
        if not may_cancel(user, can_see, reg, reg_err, job):
            return no(403, "You can only cancel your own jobs.")
        reason = CTRL.sub(" ", reason).strip() if isinstance(reason, str) else ""
        if not reason:
            return no(400, "Please type a reason for cancelling.")
        if len(reason) > REASON_MAX:
            return no(400, f"The reason is too long (at most {REASON_MAX} characters).")
    else:
        reason = None
    live, err = read_live(getattr(cfg, "registry", None))
    snap = snapshot_info(live, err, now)
    if not snap["present"]:
        return no(409, "There is no status from the GPU computer, so this was not sent.")
    if snap["stale"]:
        age = f"{dur(snap['age_s'])} ago" if snap["age_s"] is not None else "at an unknown time"
        return no(409, f"The GPU computer's status updates have stopped (the last one came {age}), so this was "
                       "not sent. Ask an admin to check the monitor.")
    matches = []
    for q in _l(live.get("queues")):
        q = _d(q)
        for j in _l(q.get("jobs")):
            if _d(j).get("job") == job:
                matches.append((q, _d(j)))
                break
        else:
            r = _d(q.get("running"))
            if r.get("job") == job:
                matches.append((q, {"job": job, "state": "running", "cancel_requested": r.get("cancel_requested")}))
    if not matches:
        return no(404, "That job is not on a GPU computer's list (it may have finished). Refresh the page.")
    q, j = matches[0]
    note = " (This job id is on more than one GPU computer's list; the first was used.)" if len(matches) > 1 else ""
    share, state = _s(q.get("share")), j.get("state")
    if _s(q.get("reach")) != "ok" or not share:
        return no(409, "The laptop can't reach that GPU computer right now, so this was not sent.")
    if not share_ok(share):
        return no(409, "That GPU computer's address is not one this page may write to, so this was not sent.")
    if action == "cancel":
        if state not in CANCELLABLE:
            return no(409, "That job is not waiting or running any more.")
        if j.get("cancel_requested"):
            return no(409, "A cancel was already requested for this job.")
    elif state != "held":
        return no(409, "That job is not on hold.")
    with job_lock(job):                                  # two cancels of one job: the second sees the first's file
        if action == "cancel" and (Path(share) / "q" / job / "CANCEL").exists():
            return no(409, "A cancel was already requested for this job.")
        try:
            audit(cfg, uid, action, job, reason, "requested")     # write-ahead: no line, no write to the share
        except (OSError, TypeError, AttributeError) as e:
            log.error("jobs audit not writable (%s): %s not done", type(e).__name__, action)
            return no(500, "The audit log can't be written, so nothing was done.")
        try:
            S = sync_module()
            if action == "cancel":
                S.cancel(job, share, who=uid, reason=reason)
                msg = ("Cancel requested. The GPU computer stops it within about 10 s." if state == "running" else
                       "Cancel requested. It will not be started.")
            else:
                S.release(job, share)
                msg = "Released. It starts when its turn comes."
            result, out = "ok", (200, {"ok": True, "message": msg + note})
        except Exception as e:                           # SyncError, or the share failing mid-write
            result = f"error: {type(e).__name__}"
            why = f" ({e})" if admin else ""
            out = no(502, f"It was not {'cancelled' if action == 'cancel' else 'released'}: the job may have just "
                          f"finished or the GPU computer's share is not answering. Refresh and try again.{why}")
        try:
            audit(cfg, uid, action, job, reason, result)
        except (OSError, TypeError, AttributeError) as e:
            log.warning("jobs audit result line not written (%s): %s %s %s", type(e).__name__, action, job, result)
        return out


async def read_body(request: Request):
    """The JSON body, read in chunks and refused past MAX_BODY bytes. -> dict, or None when unusable."""
    buf = b""
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > MAX_BODY:
            return None
    try:
        body = json.loads(buf or b"{}")
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


# ---------------------------------------------------------------- routes
def routes(ctx) -> list:
    def make(request: Request) -> dict:
        return summary(ctx.cfg, ctx.user(request), ctx.can_see)

    def page(request: Request):                 # sync: Starlette runs it in a worker thread (file reads)
        return ctx.page(request, "Jobs", page_body(render(make(request))), active="jobs")

    def api_summary(request: Request):
        s = make(request)
        if request.query_params.get("html") == "1":
            s["html"] = render(s)
        return ctx.json(s)

    def script(request: Request):
        r = Response(JS, media_type="text/javascript; charset=utf-8")
        r.headers["Cache-Control"] = "no-cache"
        return r

    async def api_action(request: Request, action: str):
        body = await read_body(request)
        if body is None:
            return ctx.json({"ok": False, "message": "That request was not understood."}, 400)
        user = ctx.user(request)       # a "share" / "path" in the body is never read
        if not _slots.acquire(blocking=False):
            return ctx.json({"ok": False, "message": "The page is busy with other requests. Try again in a moment."},
                            503)

        def work():
            try:
                return act(ctx.cfg, user, ctx.can_see, action, body.get("job"), body.get("reason"))
            finally:
                _slots.release()
        try:
            fut = asyncio.get_running_loop().run_in_executor(_pool, work)
        except Exception:
            _slots.release()
            raise
        try:
            code, out = await asyncio.wait_for(asyncio.shield(fut), ACTION_TIMEOUT_S)
        except asyncio.TimeoutError:
            return ctx.json({"ok": False, "message": "The GPU computer's share is slow to answer. The request may "
                             "still go through; refresh the page in a minute to check."}, 504)
        return ctx.json(out, code)

    async def api_cancel(request: Request):
        return await api_action(request, "cancel")

    async def api_release(request: Request):
        return await api_action(request, "release")

    return [Route(PREFIX + "/", page, methods=["GET"]),
            Route(PREFIX + "/jobs.js", script, methods=["GET"]),
            Route(PREFIX + "/api/summary", api_summary, methods=["GET"]),
            Route(PREFIX + "/api/cancel", api_cancel, methods=["POST"]),
            Route(PREFIX + "/api/release", api_release, methods=["POST"])]
