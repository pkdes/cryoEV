"""
Jobs module (CJ portal, read only): what is running, queued, blocked and done, from two sources:
  - the desktop queues: <registry>/live/live.json, written by jobs/monitor.py (train jobs; predict jobs carry
    kind "predict" + n_done / n_total);
  - the registry's jobs: registry.current("job") (review / annotation / check jobs, and any train / predict job
    that is not on a desktop queue), cached REGISTRY_TTL_S per registry folder (current() re-reads the files).

    GET /jobs/               the page: Running now, Queue, Needs attention, Review & annotation, Recently finished
    GET /jobs/api/summary    the same as JSON (schema cj_jobs/1, see summary()); ?html=1 adds "html": the page's
                             section fragments, rendered and escaped here (the page swaps them in every 30 s)

Rules: a desktop job (on a queue in live.json) is shown from live.json only; the registry fills in its name.
A snapshot older than STALE_FACTOR x its interval is stale (as the home page): a banner says so and every
desktop item is marked "last known". A predict job belongs to a person (registry `who`): anyone ctx.can_see
does not allow sees its state but not its id or name. Nothing here writes anything.
"""
from __future__ import annotations

import html
import importlib.util
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from starlette.requests import Request
from starlette.routing import Route

PREFIX = "/jobs"
ROLES = ("admin", "advanced")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TEMPLATE = HERE.parent / "templates" / "jobs.html"
SCHEMA = "cj_jobs/1"
REGISTRY_TTL_S = 10.0         # registry.current() re-reads every *.jsonl line; one read serves every request for 10 s
STALE_FACTOR = 3              # live.json older than 3 x interval_s is stale (portal/app.py is_stale, home page)
N_FINISHED = 10
OPEN = ("in_progress", "ready", "blocked")
SECTIONS = ("head", "running", "queue", "attention", "review", "finished")
PRIVATE_NAME = "Someone else's prediction job"

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
DESK_STATE = {"running": ("running", "run"), "queued": ("waiting", "wait"), "held": ("held", "wait"),
              "incomplete": ("push incomplete", "warn"), "blocked": ("blocked", "warn"),
              "cancelled": ("cancelled", "off"), "done": ("done", "ok"), "unknown": ("unknown", "warn")}
REG_STATE = {"in_progress": ("in progress", "run"), "ready": ("not started", "wait"), "blocked": ("blocked", "warn"),
             "done": ("done", "ok"), "cancelled": ("cancelled", "off")}
TYPE_WORD = {"revise": "review", "annotate": "labeling", "check": "check", "train": "training",
             "predict": "prediction"}


def hint(rec: dict) -> str:
    """What happens next, in plain words (registry/history_page.py next_action, reworded)."""
    st, typ, where = rec.get("status"), rec.get("type"), _s(rec.get("where")) or "?"
    if st == "blocked":
        return _s(rec.get("blocked_reason")) or "Blocked (no reason recorded)"
    if st == "ready":
        if typ in ("train", "predict") and where == "desktop":
            return "Not on a desktop queue yet: push it (jobs/sync.py push)"
        return "Ready to start"
    if st == "in_progress":
        if where == "app":
            port = rec.get("port") or 8766
            return f"{'Review' if typ == 'revise' else 'Label'} in the review app (port {port})"
        return {"annotate": f"Labeling in {where}", "revise": f"Reviewing in {where}",
                "train": f"Training on {where}: import the run when it returns",
                "predict": f"Predicting on {where}", "check": f"Score the check on {where}"}.get(typ, f"Going on in {where}")
    if st == "done":
        return "Done"
    if st == "cancelled":
        return "Cancelled"
    return f"Status {st!r}"


def eta_text(r: dict) -> str:
    if r.get("phase") == "finishing":
        return "finishing (final validation)"
    bound = _n(r.get("eta_patience_bound_s"))
    if bound is not None:
        return f"≤ {dur(bound)} (early stop unless val improves; all epochs {dur(r.get('eta_s'))})"
    e = _n(r.get("eta_s"))
    if e is None:
        return "not known yet (after epoch 2)" if r.get("kind", "train") == "train" else "not known yet"
    return dur(e) + (" (time cap)" if r.get("eta_capped") else "")


# ---------------------------------------------------------------- summary
def summary(cfg, user: dict, can_see, now=None) -> dict:
    """Everything the page shows, as data (schema cj_jobs/1). `can_see(user, owner)` is ctx.can_see."""
    now = time.time() if now is None else now
    live, live_err = read_live(getattr(cfg, "registry", None))
    snap = snapshot_info(live, live_err, now)
    reg, reg_err, reg_age = registry_jobs(getattr(cfg, "registry", None))
    stale = snap["stale"]

    def private(job_id, kind=None) -> bool:
        """A predict job (live.json kind, else the registry type) the viewer may not see; no owner = admin only."""
        rec = _d(reg.get(job_id)) if job_id else {}
        if kind != "predict" and rec.get("type") != "predict":
            return False
        return not can_see(user, rec.get("who"))

    def ident(job_id, kind=None) -> dict:
        """id + display name (the registry's name when it has one); hidden for another person's predict job."""
        if private(job_id, kind):
            return {"id": None, "name": PRIVATE_NAME, "private": True}
        name = _s(_d(reg.get(job_id)).get("name")) if job_id else None
        return {"id": job_id, "name": name or job_id or "?", "private": False}

    queues, running, queue, attention, finished = [], [], [], [], []
    seen = set()
    for qi, q in enumerate(_l((live or {}).get("queues"))):
        q = _d(q)
        w = _d(q.get("watcher"))
        where = _s(w.get("machine")) or _s(q.get("share")) or f"queue {qi + 1}"
        reach = _s(q.get("reach")) or "unknown"
        jobs = [_d(j) for j in _l(q.get("jobs"))]
        n_waiting = sum(1 for j in jobs if j.get("state") == "queued")
        qinfo = {"name": where, "share": _s(q.get("share")), "reach": reach, "detail": _s(q.get("detail")),
                 "watcher": None}
        queues.append(qinfo)
        for j in jobs:
            if _s(j.get("job")):
                seen.add(j["job"])
        for f in _l(q.get("finished")):
            if _s(_d(f).get("job")):
                seen.add(f["job"])
        r = _d(q.get("running"))
        if _s(r.get("job")):
            seen.add(r["job"])
        if reach != "ok":
            attention.append({"kind": "offline", **ident(None), "name": f"Queue on {where}", "queue": where,
                              "source": "desktop", "title": "Can't reach this queue",
                              "reason": f"{reach}" + (f": {qinfo['detail']}" if qinfo["detail"] else "")})
            continue
        disp = _s(w.get("display")) or "unknown"
        qinfo["watcher"] = {"display": disp, "state": _s(w.get("state")), "detail": _s(w.get("detail")),
                            "age_s": _n(w.get("age_s")), "stale": bool(w.get("stale"))}
        if disp in ("stale", "no watcher", "stopped", "unknown"):
            attention.append({"kind": "watcher", **ident(None), "name": f"Watcher on {where}", "queue": where,
                              "source": "desktop", "title": f"The desktop watcher on {where} is {disp}",
                              "reason": (f"last heartbeat {dur(w.get('age_s'))} ago" if _n(w.get("age_s")) is not None
                                         else (_s(w.get("detail")) or "no heartbeat"))
                              + ("; queued jobs will not start" if n_waiting else "")})
        elif disp == "paused" and n_waiting:
            attention.append({"kind": "watcher", **ident(None), "name": f"Watcher on {where}", "queue": where,
                              "source": "desktop", "title": f"The desktop watcher on {where} is paused",
                              "reason": (_s(w.get("detail")) or "paused") + f"; {n_waiting} queued job(s) wait"})
        if r and _s(r.get("job")):
            kind = "predict" if r.get("kind") == "predict" else "train"
            if kind == "predict":
                done, total, unit = _i(r.get("n_done")), _i(r.get("n_total")), "images"
            else:
                done, total, unit = _i(r.get("epoch")), _i(r.get("epochs")), "epochs"
            phase = _s(r.get("phase")) or ("predicting" if kind == "predict" else "training")
            hung = bool(r.get("hung"))
            hung_text = None
            if hung:
                hung_text = (f"No new epoch for {dur(r.get('last_epoch_age_s'))} (expected every "
                             f"{dur(r.get('epoch_median_s'))}). Check train.log on the desktop."
                             if kind == "train" else
                             f"No progress for {dur(r.get('last_epoch_age_s') or r.get('last_update_age_s'))}. "
                             "Check the job's log on the desktop.")
            item = {**ident(r["job"], kind), "queue": where, "kind": kind, "phase": phase, "done": done,
                    "total": total, "unit": unit, "pct": pct(done, total), "eta_s": _n(r.get("eta_s")),
                    "eta_text": eta_text({**r, "kind": kind, "phase": phase}), "elapsed_s": _n(r.get("elapsed_s")),
                    "hung": hung, "hung_text": hung_text, "cancel_requested": bool(r.get("cancel_requested")),
                    "map50_95": _n(r.get("map50_95")) if kind == "train" else None,
                    "watcher": {"display": disp, "age_s": _n(w.get("age_s"))}, "stale": stale}
            running.append(item)
            if hung:
                attention.append({"kind": "hung", **ident(r["job"], kind), "queue": where, "source": "desktop",
                                  "title": "Running but hung?" + (" (last known)" if stale else ""),
                                  "reason": hung_text})
        pos = 0
        for state in ("queued", "held", "incomplete"):
            for j in jobs:
                if j.get("state") != state or not _s(j.get("job")):
                    continue
                if state == "queued":
                    pos += 1
                note = {"queued": "waiting for the desktop",
                        "held": "held: starts after release (jobs/sync.py release)",
                        "incomplete": "the push did not finish (no READY)"}[state]
                if state == "queued" and w.get("job") == j["job"] and disp != "alive":
                    note = f"watcher {disp}" + (f": {w['detail']}" if _s(w.get("detail")) else "")
                if j.get("cancel_requested"):
                    note += "; cancel requested"
                label, tone = DESK_STATE[state]
                queue.append({**ident(j["job"]), "queue": where, "source": "desktop", "state": state,
                              "label": label, "tone": tone, "position": pos if state == "queued" else None,
                              "note": note, "stale": stale})
        for j in jobs:
            if j.get("state") in ("blocked", "unknown") and _s(j.get("job")):
                pulled = "results pulled" if j.get("pulled") else "not pulled yet"
                attention.append({"kind": j["state"], **ident(j["job"]), "queue": where, "source": "desktop",
                                  "title": "Blocked on the desktop" if j["state"] == "blocked" else "State unknown",
                                  "reason": (_s(j.get("reason")) or "no reason recorded") + f" ({pulled})"})
        for f in _l(q.get("finished")):
            f = _d(f)
            if not _s(f.get("job")):
                continue
            st = _s(f.get("status")) or "unknown"
            label, tone = DESK_STATE.get(st, (st, "wait"))
            m = _d(f.get("metrics"))
            ts = parse_local(f.get("ended"))
            finished.append({**ident(f["job"]), "queue": where, "source": "desktop", "status": st, "label": label,
                             "tone": tone, "reason": _s(f.get("reason")), "ended": _s(f.get("ended")), "ts": ts,
                             "when": ago(ts, now), "pulled": bool(f.get("pulled")),
                             "map50_95": _n(m.get("mask_map50_95")), "map50": _n(m.get("mask_map50")),
                             "best_epoch": _i(f.get("best_epoch"))})

    review = []
    order = {"in_progress": 0, "blocked": 1, "ready": 2}
    for jid, rec in reg.items():
        rec = _d(rec)
        if jid in seen:                                 # on a desktop queue: live.json is the truth for it
            continue
        st, typ = _s(rec.get("status")) or "?", _s(rec.get("type")) or "?"
        label, tone = REG_STATE.get(st, (st, "wait"))
        idn = ident(jid)
        if st in OPEN:
            prog = _d(rec.get("progress"))
            p = ({"done": _i(prog.get("done")), "total": _i(prog.get("total")), "unit": _s(prog.get("unit")) or ""}
                 if prog else None)
            review.append({**idn, "source": "registry", "type": typ, "type_label": TYPE_WORD.get(typ, typ),
                           "where": _s(rec.get("where")), "status": st, "label": label, "tone": tone,
                           "who": None if idn["private"] else _s(rec.get("who")), "progress": p,
                           "pct": pct(p["done"], p["total"]) if p else None, "hint": hint(rec),
                           "logged": _s(rec.get("logged"))})
            if st == "blocked":
                attention.append({"kind": "blocked", **idn, "queue": None, "source": "registry",
                                  "title": f"Blocked {TYPE_WORD.get(typ, typ)} job",
                                  "reason": _s(rec.get("blocked_reason")) or "no reason recorded"})
        elif st in ("done", "cancelled"):
            ts = parse_local(rec.get("logged"))
            finished.append({**idn, "queue": None, "source": "registry", "type": typ, "status": st,
                             "label": label, "tone": tone, "reason": None, "ended": _s(rec.get("logged")),
                             "ts": ts, "when": ago(ts, now), "pulled": None, "map50_95": None, "map50": None,
                             "best_epoch": None})
    review.sort(key=lambda x: (order.get(x["status"], 9), x["logged"] or "", x["id"] or ""))
    finished.sort(key=lambda x: x["ts"] or 0, reverse=True)
    finished = finished[:N_FINISHED]
    kinds = {"offline": 0, "watcher": 1, "hung": 2, "blocked": 3, "unknown": 4}
    attention.sort(key=lambda x: kinds.get(x["kind"], 9))
    return {"schema": SCHEMA, "now_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
            "snapshot": snap, "registry": {"present": reg_err is None, "problem": reg_err, "n_jobs": len(reg),
                                           "read_age_s": None if reg_age is None else round(reg_age, 1)},
            "queues": queues, "running": running, "queue": queue, "attention": attention, "review": review,
            "finished": finished,
            "counts": {"running": len(running), "queued": sum(1 for x in queue if x["state"] == "queued"),
                       "held": sum(1 for x in queue if x["state"] != "queued"), "attention": len(attention),
                       "review_open": len(review), "finished": len(finished)}}


# ---------------------------------------------------------------- html (every text through esc)
def chip(label, tone) -> str:
    return f'<span class="cj-jobs-chip cj-jobs-t-{esc(tone)}">{esc(label)}</span>'


def sec_head(key, title, n=None) -> str:
    count = "" if n is None else f' <span class="cj-jobs-n">{int(n)}</span>'
    return f'<h2 class="cj-sec" id="cj-jobs-h-{key}">{esc(title)}{count}</h2>'


def empty(text) -> str:
    return f'<p class="cj-mut cj-jobs-empty">{esc(text)}</p>'


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


def render(s: dict) -> dict:
    """summary() -> {section: html fragment}; the page's placeholders and the refresh both use these."""
    snap, out = s["snapshot"], {}

    # head: snapshot line, stale banner, counts
    if not snap["present"]:
        line = snap["problem"] or "No live snapshot."
    else:
        line = (f"Desktop snapshot {dur(snap['age_s'])} old (updated every {dur(snap['interval_s'])}"
                + (f", monitor on {snap['monitor']}" if snap["monitor"] else "") + ").")
    reg = s["registry"]
    line += " " + (f"Registry: {reg['n_jobs']} job(s)." if reg["present"] else f"Registry: {reg['problem']}.")
    head = f'<p class="cj-mut cj-jobs-snap">{esc(line)}</p>'
    if snap["present"] and snap["stale"]:
        head += ('<p class="cj-jobs-banner" role="status"><b>Stale snapshot.</b> '
                 + esc(f"live.json is {dur(snap['age_s'])} old (the monitor writes every {dur(snap['interval_s'])}): "
                       "jobs/monitor.py has stopped or cannot write. Desktop jobs below are the last known state, "
                       "not now.") + "</p>")
    elif not snap["present"]:
        head += ('<p class="cj-jobs-banner" role="status"><b>No desktop snapshot.</b> '
                 + esc("What runs on the desktop is unknown; registry jobs are still shown.") + "</p>")
    c = s["counts"]
    links = [("running", c["running"], "running", ""), ("queue", c["queued"], "waiting", ""),
             ("attention", c["attention"], "need attention", " cj-jobs-t-warn" if c["attention"] else ""),
             ("review", c["review_open"], "review & annotation", "")]
    head += ('<nav class="cj-jobs-counts" aria-label="Jobs summary">'
             + "".join(f'<a class="cj-jobs-count{cls}" href="#cj-jobs-s-{k}"><b>{int(n)}</b> {esc(w)}</a>'
                       for k, n, w, cls in links) + "</nav>")
    out["head"] = head

    # running now
    parts = [sec_head("running", "Running now", len(s["running"]))]
    wl = []
    for q in s["queues"]:
        if q["reach"] != "ok":
            wl.append(f'<li><span class="cj-dot down" aria-hidden="true"></span><b>{esc(q["name"])}</b> '
                      f'<span class="cj-mut">{esc(q["reach"])}</span></li>')
            continue
        w = q["watcher"] or {}
        ok = w.get("display") == "alive" and not s["snapshot"]["stale"]
        dot = "up" if ok else ("unk" if s["snapshot"]["stale"] else "down")
        hb = f" · heartbeat {dur(w.get('age_s'))} ago" if w.get("age_s") is not None else ""
        det = f" · {w['detail']}" if w.get("detail") and w.get("display") != "alive" else ""
        wl.append(f'<li><span class="cj-dot {dot}" aria-hidden="true"></span><b>{esc(q["name"])}</b> '
                  f'<span class="cj-mut">watcher {esc(str(w.get("display") or "?") + hb + det)}</span></li>')
    if wl:
        parts.append(f'<ul class="cj-list cj-jobs-watchers">{"".join(wl)}</ul>')
    cards = []
    for r in s["running"]:
        unit = "Images" if r["kind"] == "predict" else "Epoch"
        prog = (f"{unit} {r['done']} / {r['total']}" if r["done"] is not None and r["total"]
                else ("Setting up: no epoch finished yet" if r["phase"] == "setting up" else f"{unit} ?"))
        tags = chip("last known" if r["stale"] else r["phase"], "wait" if r["stale"] else "run")
        if r["hung"]:
            tags += chip("hung?", "warn")
        body = [f'<div class="cj-jobs-top">{name_line(r)}<span class="cj-jobs-tags">{tags}</span></div>',
                f'<div class="cj-jobs-prog"><span>{esc(prog)}</span>'
                + (f'<span class="cj-mut">{r["pct"]} %</span>' if r["pct"] is not None else "") + "</div>",
                bar(r["done"], r["total"], unit),
                kv([("ETA", r["eta_text"]), ("Elapsed", dur(r["elapsed_s"]) if r["elapsed_s"] is not None else None),
                    ("On", r["queue"]),
                    ("Watcher", f"{r['watcher']['display']}"
                     + (f", heartbeat {dur(r['watcher']['age_s'])} ago" if r["watcher"]["age_s"] is not None else "")),
                    ("Val mAP50-95", f"{r['map50_95']:.3f}" if r["map50_95"] is not None else None)])]
        if r["hung_text"]:
            body.append(f'<p class="cj-jobs-warn">{esc(r["hung_text"])}</p>')
        if r["cancel_requested"]:
            body.append(f'<p class="cj-jobs-warn">{esc("Cancel requested: the runner stops it within ~10 s.")}</p>')
        cards.append(f'<article class="cj-card cj-jobs-item">{"".join(body)}</article>')
    if cards:
        parts.append(f'<div class="cj-jobs-grid">{"".join(cards)}</div>')
    elif not s["snapshot"]["present"]:
        parts.append(empty("Unknown: no desktop snapshot."))
    elif s["snapshot"]["stale"]:
        parts.append(empty("Nothing was running when the last snapshot was taken."))
    else:
        parts.append(empty("Nothing is running on the desktop."))
    out["running"] = "".join(parts)

    # queue
    parts = [sec_head("queue", "Queue", len(s["queue"]))]
    rows = []
    for x in s["queue"]:
        num = f'<span class="cj-jobs-pos">{x["position"]}</span>' if x["position"] else \
            '<span class="cj-jobs-pos cj-jobs-t-off" aria-hidden="true">·</span>'
        rows.append(f'<li class="cj-jobs-row">{num}<span class="cj-jobs-main">{name_line(x)}'
                    f'<span class="cj-mut">{esc(x["note"] + " · on " + x["queue"] + (" · last known" if x["stale"] else ""))}'
                    f'</span></span>{chip(x["label"], x["tone"])}</li>')
    parts.append(f'<ol class="cj-card cj-jobs-rows">{"".join(rows)}</ol>' if rows else
                 empty("No job waiting on the desktop." if s["snapshot"]["present"] else
                       "Unknown: no desktop snapshot."))
    out["queue"] = "".join(parts)

    # needs attention
    parts = [sec_head("attention", "Needs attention", len(s["attention"]))]
    cards = []
    for a in s["attention"]:
        where = ("" if a["kind"] in ("offline", "watcher") else          # the queue is already in the title
                 f" · on {a['queue']}" if a.get("queue") else (" · registry" if a["source"] == "registry" else ""))
        cards.append(f'<article class="cj-card cj-jobs-item cj-jobs-att"><div class="cj-jobs-top">{name_line(a)}'
                     f'{chip(a["kind"], "warn")}</div><p class="cj-jobs-why"><b>{esc(a["title"])}</b>'
                     f'<span class="cj-mut">{esc(where)}</span></p>'
                     f'<p class="cj-jobs-reason cj-wrap">{esc(a["reason"] or "")}</p></article>')
    parts.append(f'<div class="cj-jobs-grid">{"".join(cards)}</div>' if cards else empty("Nothing needs attention."))
    out["attention"] = "".join(parts)

    # review & annotation (open registry jobs not on a desktop queue)
    parts = [sec_head("review", "Review & annotation jobs", len(s["review"]))]
    cards = []
    for x in s["review"]:
        p = x["progress"]
        prog = ""
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
                     f'<p class="cj-jobs-hint">{esc(x["hint"])}</p></article>')
    if cards:
        parts.append(f'<div class="cj-jobs-grid">{"".join(cards)}</div>')
    else:
        parts.append(empty("No open review or annotation job." if s["registry"]["present"] else
                           f"Unknown: {s['registry']['problem']}."))
    out["review"] = "".join(parts)

    # recently finished
    parts = [sec_head("finished", "Recently finished", len(s["finished"]))]
    rows = []
    for x in s["finished"]:
        bits = [x["when"] or x["ended"] or "time unknown"]
        if x["source"] == "desktop":
            if x["map50_95"] is not None:
                bits.append(f"mask mAP50-95 {x['map50_95']:.3f}")
            if x["best_epoch"] is not None:
                bits.append(f"best epoch {x['best_epoch']}")
            bits.append(f"on {x['queue']}")
            if x["status"] in ("done", "blocked", "cancelled"):
                bits.append("pulled" if x["pulled"] else "not pulled yet")
        else:
            bits.append(f"{TYPE_WORD.get(x.get('type'), x.get('type') or '?')} · registry")
        reason = f'<span class="cj-mut cj-wrap">{esc(x["reason"])}</span>' if x["reason"] else ""
        rows.append(f'<li class="cj-jobs-row"><span class="cj-jobs-main">{name_line(x)}'
                    f'<span class="cj-mut">{esc(" · ".join(bits))}</span>{reason}</span>'
                    f'{chip(x["label"], x["tone"])}</li>')
    parts.append(f'<ul class="cj-card cj-jobs-rows">{"".join(rows)}</ul>' if rows else empty("Nothing finished yet."))
    out["finished"] = "".join(parts)
    return out


_PH = re.compile(r"\{\{(head|running|queue|attention|review|finished)\}\}")


def page_body(fragments: dict) -> str:
    """jobs.html with each placeholder replaced in ONE pass, so text inside a fragment is never re-scanned."""
    tpl = TEMPLATE.read_text(encoding="utf-8")
    return _PH.sub(lambda m: fragments.get(m.group(1), ""), tpl)


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

    return [Route(PREFIX + "/", page, methods=["GET"]),
            Route(PREFIX + "/api/summary", api_summary, methods=["GET"])]
