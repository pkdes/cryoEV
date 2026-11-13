"""status: one read-only look at the platform.

- services: tools/services.py service_status() for each entry of the services config (state, port, health);
- desktop jobs: the job monitor's snapshot (<registry>/live/live.json, written by jobs/monitor.py from the same queue
  reader as `jobs/sync.py status`), or with --share the queue itself through jobs/sync.py queue_rows / watch_state;
- review packages: portal/titles.py scan_packages(): cards decided / total and images reviewed / total, COUNTS only
  (never who decided what, never `_work_*` folders or decoys);
- feedback: open reports (not Done / Won't do) from portal/feedback_store.py.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from cryoev import common as CM
from cryoev.commands import feedback as FB

PKG_KEYS = ("id", "kind", "status", "cards", "decided", "images", "reviewed")
JOB_KEYS = ("job", "state", "reason", "pulled", "cancel_requested")


def add(sub):
    p = sub.add_parser("status", help="services, desktop jobs, review package counts, open feedback (read)")
    p.add_argument("--share", type=Path, help="read the desktop queue itself (jobs/sync.py) instead of the "
                                              "monitor's snapshot")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=run)


def services_part() -> dict:
    S = CM.services()
    try:
        cfg = S.load_config(CM.services_config_path())
    except S.ConfigError as e:
        return {"problem": str(e), "rows": []}
    rows = []
    for svc in cfg.services:
        r = S.service_status(cfg, svc)
        rows.append({k: r.get(k) for k in ("name", "enabled", "state", "port", "health")})
    return {"problem": None, "rows": rows}


def jobs_part(share=None) -> dict:
    if share is not None:
        sync = CM.load("cj_sync", "jobs/sync.py")
        if not Path(share).is_dir():
            return {"source": "queue", "problem": f"share {share} not reachable", "queues": []}
        w = sync.watch_state(share) or {}
        rows = [{k: r.get(k) for k in JOB_KEYS} for r in sync.queue_rows(share)]
        return {"source": "queue", "problem": None,
                "queues": [{"reach": "ok", "watcher": {k: w.get(k) for k in ("state", "detail", "job", "age_s")},
                            "jobs": rows}]}
    live = CM.registry_root() / "live" / "live.json"
    try:
        d = json.loads(live.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"source": "monitor snapshot", "problem": f"no readable {live} (is jobs/monitor.py running?)",
                "queues": []}
    age = None
    try:
        age = round((datetime.now(timezone.utc) - datetime.fromisoformat(str(d.get("generated_utc")))
                     .astimezone(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        pass
    queues = []
    for q in d.get("queues") or []:
        w = q.get("watcher") or {}
        queues.append({"reach": q.get("reach"), "watcher": {k: w.get(k) for k in ("state", "detail", "job", "age_s")},
                       "jobs": [{k: j.get(k) for k in JOB_KEYS} for j in q.get("jobs") or []]})
    return {"source": "monitor snapshot", "age_s": age, "problem": None, "queues": queues}


def packages_part() -> dict:
    from portal import titles as TT
    root = CM.laptop_root() / "review"
    reg, rr = CM.registry(), CM.registry_root()
    records = {e: reg.current(e, rr) for e in ("label_version", "review_package", "pull")}
    rows = [{k: p.get(k) for k in PKG_KEYS} for p in TT.scan_packages(root, records)]
    return {"rows": rows}


def feedback_part() -> dict:
    path = FB.feedback_file()
    if path is None:
        return {"problem": "no feedback file configured (CJ_FEEDBACK_FILE or the services config)", "open": None}
    from portal import feedback_store as FS
    reports = FS.read_all(path).values()
    by = {}
    for r in reports:
        by[r["status"]] = by.get(r["status"], 0) + 1
    return {"problem": None, "total": len(reports),
            "open": sum(n for s, n in by.items() if s not in ("Done", "Won't do")), "by_status": by}


def run(a) -> int:
    out = {"services": services_part(), "desktop_jobs": jobs_part(a.share), "review_packages": packages_part(),
           "feedback": feedback_part()}
    if a.json:
        CM.emit_json(out)
        return 0
    s = out["services"]
    print("Services" + (f": {s['problem']}" if s["problem"] else ""))
    for r in s["rows"]:
        print(f"  {r['name']:24} {r['state']:24} port {r['port'] or '-':<6} health {r['health']}")
    j = out["desktop_jobs"]
    print(f"Desktop jobs ({j['source']}" + (f", {j['age_s']} s old" if j.get("age_s") is not None else "") + ")"
          + (f": {j['problem']}" if j["problem"] else ""))
    for q in j["queues"]:
        w = q["watcher"]
        print(f"  queue {q['reach']}; watcher {w.get('state')}" + (f" ({w['detail']})" if w.get("detail") else "")
              + (f", job {w['job']}" if w.get("job") else ""))
        for r in q["jobs"]:
            print(f"    {r['job']:50} {r['state']}" + (f" ({r['reason']})" if r.get("reason") else "")
                  + ("" if r.get("pulled") or r["state"] not in ("done", "blocked", "cancelled") else " not pulled"))
    print("Review packages (counts only)")
    for r in out["review_packages"]["rows"]:
        print(f"  {r['id']:44} {r['status']:12} cards {r['decided']}/{r['cards']}  images reviewed "
              f"{r['reviewed']}/{r['images']}")
    f = out["feedback"]
    print("Feedback: " + (f["problem"] if f["problem"] else f"{f['open']} open of {f['total']} "
                          + ", ".join(f"{k} {v}" for k, v in sorted(f["by_status"].items()))))
    return 0
