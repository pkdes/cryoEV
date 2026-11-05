#!/usr/bin/env python3
"""Laptop side of the training-job queue (docs/JOBS.md): push bundles to the desktop's queue share, pull results.

    python jobs/sync.py push <job id> --bundle <zip> --share \\\\desktop\\cj
    python jobs/sync.py pull --share \\\\desktop\\cj [--who pujan]
    python jobs/sync.py status --share \\\\desktop\\cj
    python jobs/sync.py retry <job id> --share \\\\desktop\\cj
    python jobs/sync.py release <job id> --share ...      (a job pushed with --hold)
    python jobs/sync.py stop|start --share ...            (pause / resume the desktop watcher, jobs/watch.py)
    python jobs/sync.py local <job id> --bundle <zip> [--max-hours H] [--allow-big]   (train on this machine)

push: the job must be a `train` job in the registry (`registry.py job new --type train --where desktop ...`).
  Copies the zip (+ .md5) to q/.incoming/<job>/, checks the copy's md5, writes job.json, renames the folder to
  q/<job>/ and writes READY last.
pull: for each d/<job>/ with DONE.json not pulled yet, copies runs/<run>/ to <CryoAI>/training outputs/<dataset>/
  runs/<run>/ (new folders only, never overwrites; staged under <dataset>/.pull_tmp/), runs the importer
  (`registry.py import`) and `registry.py set-status <job> done|blocked --reason`, then logs the pull in
  <CryoAI-laptop>/jobs/pulled.jsonl. Pulling again adds nothing.
retry: moves d/<job>/ to d/.retired/<job>_<time>/ so the runner picks the job again (nothing is deleted).
local: push to a folder on this machine (default C:\\cj_local), run jobs/run_job.py on it, then pull.
Copies use robocopy /E /Z /R:2 /W:5 when the share is a UNC path (never /MIR, /PURGE or /MOV*; exit >= 8 =
failure), shutil for a local folder. The registry is written only through registry.py, and only by pull.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "registry"))
import common as C  # noqa: E402
from data_utils import config  # noqa: E402
from data_utils.path_guard import protected_problem  # noqa: E402

REGISTRY_PY = REPO / "registry" / "registry.py"
LEDGER = Path("jobs") / "pulled.jsonl"          # under <CryoAI-laptop>
PULL_TMP = ".pull_tmp"                           # <CryoAI>/training outputs/<dataset>/.pull_tmp/<run> while copying
LOCAL_QUEUE = Path("C:/cj_local")                # `local`: the laptop's own queue (short: Windows path limit)


class SyncError(RuntimeError):
    pass


def registry_job(job_id):
    """The job's current registry record (read only), or None."""
    import registry
    return registry.current("job", config.registry_root()).get(job_id)


# ---------------------------------------------------------------- push
def git(*args, timeout=60):
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=timeout, env=env)


def commit_on_origin(commit):
    """None when `commit` is on origin/main after a fetch (the desktop watcher trains with main), else why not."""
    try:
        r = git("fetch", "origin", "main", timeout=120)
        if r.returncode:
            return f"git fetch failed ({r.stderr.strip()[:200]}); can't check that the code is on GitHub"
        if git("merge-base", "--is-ancestor", str(commit), "origin/main").returncode:
            return f"bundle commit {commit} is not on origin/main: push the repo first (the desktop trains with main)"
    except (OSError, subprocess.SubprocessError) as e:
        return f"git failed: {e}"
    return None


def push(job_id, bundle, share, copier="auto", check_registry=True, check_origin=True, max_hours=None, hold=False):
    import ids
    bundle, share = Path(bundle), Path(share)
    if not ids.is_id("job", job_id) or len(job_id) > C.MAX_JOB_ID:
        raise SyncError(f"{job_id!r} is not a job id of <= {C.MAX_JOB_ID} characters (job_<YYYYMMDD>_<slug>)")
    if check_registry:
        rec = registry_job(job_id)
        if rec is None or rec.get("type") != "train":
            raise SyncError(f"{job_id} is not a train job in the registry; create it with "
                            f"`registry.py job new --type train --where desktop ...`")
    if not bundle.is_file():
        raise SyncError(f"no bundle {bundle}")
    md5 = C.md5_file(bundle)
    side = C.sidecar_md5(bundle)
    if side and side != md5:
        raise SyncError(f"md5 mismatch: {bundle.name} is {md5}, its .md5 sidecar says {side}")
    with zipfile.ZipFile(bundle) as z:
        names = [n for n in z.namelist() if n.count("/") == 1 and n.endswith("/JOB.json")]
        if len(names) != 1:
            raise SyncError(f"{bundle.name} has no <dataset>/JOB.json (make it with training/make_job_bundle.py)")
        job = json.loads(z.read(names[0]))
    if check_origin:
        why = commit_on_origin(job.get("cryoEV_commit"))
        if why:
            raise SyncError(why)
    q = share / C.Q
    if (q / job_id).exists() or (share / C.D / job_id).exists():
        raise SyncError(f"{job_id} is already in the queue (q/ or d/); use a new job, or `retry` for d/")
    inc = q / C.INCOMING / job_id
    files = [bundle.name] + ([bundle.name + ".md5"] if side else [])
    C.copy_files(bundle.parent, inc, files, copier)
    if C.md5_file(inc / bundle.name) != md5:
        raise SyncError(f"md5 of the copy in {inc} differs; left there, push again")
    C.write_json(inc / C.JOB_FILE, {"schema": "queue_job/1", "job": job_id, "type": "train",
                                    "bundle_id": job.get("bundle_id") or bundle.stem, "zip": bundle.name,
                                    "zip_md5": md5, "dataset": job.get("dataset_name"),
                                    "pushed": C.now(), "pushed_from": platform.node(), "max_hours": max_hours})
    inc.rename(q / job_id)
    if not hold:
        (q / job_id / C.READY).write_text(C.now() + "\n", encoding="utf-8")  # last
    return q / job_id


def release(job_id, share):
    """Write READY for a job pushed with --hold."""
    qd = Path(share) / C.Q / job_id
    if not (qd / C.JOB_FILE).is_file():
        raise SyncError(f"{job_id} is not in {qd.parent}")
    if (qd / C.READY).exists():
        raise SyncError(f"{job_id} is already released")
    (qd / C.READY).write_text(C.now() + "\n", encoding="utf-8")
    return qd


def stop(share, who="pujan"):
    """The watcher finishes its current job, then pauses (it stays running)."""
    p = Path(share) / C.STOP
    p.write_text(f"{C.now()} {who} {platform.node()}\n", encoding="utf-8")
    return p


def start(share):
    """Lift a stop: STOP is renamed into .retired/ (nothing on the share is deleted)."""
    share = Path(share)
    if not (share / C.STOP).exists():
        print("[start] no STOP file: the watcher isn't paused by a stop")
        return None
    dst = share / C.RETIRED / f"STOP_{time.strftime('%Y%m%d-%H%M%S')}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    (share / C.STOP).rename(dst)
    return dst


def watch_state(share):
    """watch.json plus the heartbeat age in seconds, or None."""
    try:
        w = C.read_json(Path(share) / C.WATCH_JSON)
    except (OSError, ValueError):
        return None
    from datetime import datetime
    try:
        w["age_s"] = round((datetime.now() - datetime.fromisoformat(w["heartbeat"])).total_seconds())
    except (KeyError, ValueError, TypeError):
        w["age_s"] = None
    return w


# ---------------------------------------------------------------- pull
def ledger_path():
    return config.laptop_root() / LEDGER


def ledger(path):
    out = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(line)
                out.add((o["job"], o["done_sha256"]))
            except (ValueError, KeyError):
                continue
    return out


def guard(path):
    problem = protected_problem(path)
    if problem:
        raise SyncError(f"refused: {problem}")


def copy_runs(jd, dataset, copier):
    """d/<job>/runs/<run>/ -> <CryoAI>/training outputs/<dataset>/runs/<run>/ (new only). -> (copied, skipped)."""
    src = jd / "runs"
    if not src.is_dir():
        return [], []
    ds = config.cryoai_root() / "training outputs" / dataset
    copied, skipped = [], []
    for rd in sorted(p for p in src.iterdir() if p.is_dir()):
        dest = ds / "runs" / rd.name
        if dest.exists():
            skipped.append(rd.name)                     # never overwrite an existing run folder
            continue
        part = ds / PULL_TMP / rd.name
        guard(part)
        guard(dest)
        part.parent.mkdir(parents=True, exist_ok=True)
        C.copy_tree(rd, part, copier)                   # raises on failure; the partial copy stays in .pull_tmp
        dest.parent.mkdir(parents=True, exist_ok=True)
        part.rename(dest)
        copied.append(rd.name)
    return copied, skipped


def registry_cmd(*args):
    r = subprocess.run([sys.executable, str(REGISTRY_PY), *args], cwd=REPO, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip())
    if r.returncode:
        raise SyncError(f"registry.py {' '.join(args[:2])} failed (exit {r.returncode})")
    return r.stdout


def pull(share, who="pujan", copier="auto"):
    """-> list of {job, status, copied, skipped}. Raises SyncError after logging what succeeded."""
    share = Path(share)
    dd = share / C.D
    led_path = ledger_path()
    seen = ledger(led_path)
    todo = []
    for jd in sorted(dd.iterdir()) if dd.is_dir() else []:
        if jd.name.startswith(".") or not jd.is_dir() or not (jd / C.DONE).is_file():
            continue
        raw = (jd / C.DONE).read_bytes()
        key = (jd.name, hashlib.sha256(raw).hexdigest())
        if key in seen:
            continue
        todo.append((jd, json.loads(raw), key[1]))
    if not todo:
        print("[pull] nothing new")
        return []
    pulled, errors = [], []
    for jd, done, sha in todo:
        try:
            copied, skipped = copy_runs(jd, done["dataset"], copier) if done.get("dataset") else ([], [])
            pulled.append((jd.name, done, sha, copied, skipped))
            print(f"[pull] {jd.name}: {done['status']}; copied {copied or 'no runs'}"
                  + (f"; kept existing {skipped}" if skipped else ""))
        except (OSError, SyncError) as e:
            errors.append(f"{jd.name}: copy failed: {e}")
    if pulled:
        registry_cmd("import")
    out = []
    for job_id, done, sha, copied, skipped in pulled:
        target = "done" if done["status"] == "done" else "blocked"
        reason = done.get("reason") or ("blocked on the runner" if target == "blocked" else None)
        cur = registry_job(job_id)
        if cur is None:
            errors.append(f"{job_id}: not in the registry; status not set")
        elif cur.get("status") != target or (target == "blocked" and cur.get("blocked_reason") != reason):
            args = ["set-status", job_id, target, "--who", who]
            if target == "blocked":
                args += ["--reason", reason[:300]]
            else:
                args += ["--note", f"run {done.get('run')} ({done.get('machine')}"
                                   + (", --allow-big" if done.get("allow_big") else "") + ")"]
            try:
                registry_cmd(*args)
            except SyncError as e:
                errors.append(f"{job_id}: {e}")
                continue
        guard(led_path)
        led_path.parent.mkdir(parents=True, exist_ok=True)
        with open(led_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"job": job_id, "done_sha256": sha, "status": target, "reason": reason,
                                "copied": copied, "skipped": skipped, "pulled": C.now()}) + "\n")
        out.append({"job": job_id, "status": target, "copied": copied, "skipped": skipped})
    if errors:
        raise SyncError("; ".join(errors))
    return out


# ---------------------------------------------------------------- status / retry
def status(share):
    share = Path(share)
    seen = {j for j, _ in ledger(ledger_path())} if config.laptop_root(required=False) else set()
    rows = []
    for d in sorted((share / C.Q).iterdir()) if (share / C.Q).is_dir() else []:
        if d.name.startswith(".") or not d.is_dir():
            continue
        if (share / C.D / d.name / C.DONE).is_file():
            st = C.read_json(share / C.D / d.name / C.DONE)
            what = f"{st['status']}" + (f" ({st.get('reason')})" if st.get("reason") else "")
            what += " pulled" if d.name in seen else " not pulled"
        elif (share / C.D / f".tmp_{d.name}").is_dir():
            what = "running (or stopped mid-job)"
        elif (d / C.READY).is_file():
            what = "queued"
        else:
            what = "no READY (push incomplete)"
        rows.append((d.name, what))
    for name, what in rows:
        print(f"{name:50} {what}")
    w = watch_state(share)
    if w:
        print(f"watcher: {w.get('state')}" + (f" ({w.get('detail')})" if w.get("detail") else "")
              + (f", job {w['job']}" if w.get("job") else "") + f", heartbeat {w.get('age_s')} s ago"
              + ("  ** no heartbeat for 5 min: is the watcher running? **"
                 if w.get("age_s") is not None and w["age_s"] > 300 else ""))
    elif (share / C.STOP).exists():
        print("watcher: no watch.json; STOP is set")
    return rows


def retry(job_id, share, force=False):
    """Move d/<job> aside so the runner picks the job again. Refuses a bad id (e.g. a live .tmp_ folder)
    and a done job that was never pulled (its run would never be imported) unless force."""
    import ids
    share = Path(share)
    if not ids.is_id("job", job_id):
        raise SyncError(f"{job_id!r} is not a job id")
    src = share / C.D / job_id
    if not src.is_dir():
        raise SyncError(f"no d/{job_id} to retry")
    done = src / C.DONE
    if not force and done.is_file() and C.read_json(done).get("status") == "done":
        pulled = {j for j, _ in ledger(ledger_path())} if config.laptop_root(required=False) else set()
        if job_id not in pulled:
            raise SyncError(f"{job_id} is done but not pulled yet; pull it first (or --force)")
    dst = share / C.D / C.RETIRED / f"{job_id}_{time.strftime('%Y%m%d-%H%M%S')}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    ready = share / C.Q / job_id / C.READY
    if ready.exists():                                 # fresh mtime: a retried job goes to the back of the queue
        ready.write_text(C.now() + "\n", encoding="utf-8")
    print(f"[retry] d/{job_id} -> d/{C.RETIRED}/{dst.name}; the runner will pick {job_id} again")
    return dst


# ---------------------------------------------------------------- local (the laptop trains its own job)
def local(job_id, bundle, queue=LOCAL_QUEUE, allow_big=False, who="pujan", runner_argv=None, max_hours=None):
    """push to a local queue folder, run jobs/run_job.py on it for this job, pull. -> pull() output ([] when
    the runner refused to start). The same code paths as the desktop; only the folder is local."""
    from run_job import is_network_path
    queue = Path(queue)
    if not max_hours:
        raise SyncError("local needs --max-hours (the runner stops the trainer then; the partial run is kept)")
    if is_network_path(queue):
        raise SyncError(f"{queue} is a network folder; `local` uses a folder on this machine (default {LOCAL_QUEUE})")
    (queue / C.Q).mkdir(parents=True, exist_ok=True)
    push(job_id, bundle, queue, copier="shutil", check_origin=False)   # same machine: no GitHub needed
    argv = (runner_argv or [sys.executable, str(HERE / "run_job.py")]) + ["--queue-dir", str(queue), "--job", job_id]
    if allow_big:
        argv.append("--allow-big")
    if max_hours:
        argv += ["--max-hours", str(max_hours)]
    rc = subprocess.run(argv, cwd=REPO).returncode
    if rc in (2, 3):
        raise SyncError(f"run_job.py exit {rc} ({'refused at start' if rc == 2 else 'another runner holds the lock'});"
                        f" {job_id} stays queued in {queue}: fix it, then `run_job.py --queue-dir {queue}` and `pull`")
    return pull(queue, who, copier="shutil")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("local")
    lp.add_argument("job_id")
    lp.add_argument("--bundle", required=True, type=Path, help="zip from training/make_job_bundle.py")
    lp.add_argument("--queue", type=Path, default=LOCAL_QUEUE, help=f"local queue folder (default {LOCAL_QUEUE})")
    lp.add_argument("--allow-big", action="store_true", help="y8m/l/x or imgsz > 640 on a small GPU (needs --max-hours)")
    lp.add_argument("--max-hours", type=float, help="the runner stops the trainer after this many hours")
    lp.add_argument("--who", default="pujan", help="people.json id for the set-status lines")
    for name in ("push", "pull", "status", "retry", "release", "stop", "start"):
        p = sub.add_parser(name)
        if name in ("push", "retry", "release"):
            p.add_argument("job_id")
        p.add_argument("--share", required=True, type=Path, help="queue root on the desktop, e.g. \\\\desktop\\cj")
        if name in ("push", "pull"):
            p.add_argument("--copier", choices=["auto", "robocopy", "shutil"], default="auto")
        if name == "push":
            p.add_argument("--bundle", required=True, type=Path, help="zip from training/make_job_bundle.py")
            p.add_argument("--max-hours", type=float, help="the runner stops the trainer after this many hours "
                           "(default on the watcher: 48)")
            p.add_argument("--hold", action="store_true", help="queue without READY; `release` starts it")
        if name in ("pull", "stop"):
            p.add_argument("--who", default="pujan", help="people.json id for the set-status lines")
        if name == "retry":
            p.add_argument("--force", action="store_true", help="also retry a done job that was never pulled")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "local":
            local(a.job_id, a.bundle, a.queue, a.allow_big, a.who, max_hours=a.max_hours)
        elif a.cmd == "push":
            q = push(a.job_id, a.bundle, a.share, a.copier, max_hours=a.max_hours, hold=a.hold)
            print(f"[push] {'held (run `release` to start it)' if a.hold else 'queued'}: {q}")
        elif a.cmd == "release":
            print(f"[release] {release(a.job_id, a.share)} is ready")
        elif a.cmd == "stop":
            print(f"[stop] wrote {stop(a.share, a.who)}: the watcher finishes its current job, then pauses")
        elif a.cmd == "start":
            dst = start(a.share)
            if dst:
                print(f"[start] STOP moved to {dst}: the watcher resumes within a minute")
        elif a.cmd == "pull":
            pull(a.share, a.who, a.copier)
        elif a.cmd == "status":
            status(a.share)
        else:
            retry(a.job_id, a.share, a.force)
    except (SyncError, config.ConfigError, OSError) as e:
        print(f"[{a.cmd}] {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
