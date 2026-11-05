#!/usr/bin/env python3
"""Desktop watcher: run every job pushed to the queue, one at a time, with the latest main (docs/JOBS.md).

    python jobs/watch.py --queue-dir C:\\cj [--repo C:\\cj_repo] [--interval 60]
    python jobs/watch.py --stamp-venv            (once after `pip install -r requirements.txt`, and after each update)

Each loop (every --interval seconds) it writes <queue>/watch.json (state, job, heartbeat), then:
  - <queue>/STOP exists (sync.py stop)        -> paused; the current job is never interrupted
  - no READY job                               -> idle
  - holding runner.lock, before touching a job:
      GPU busy (a labmate)                      -> wait; the job stays queued; one ntfy after --gpu-notify-hours
      update the clone: git fetch + merge --ff-only origin/main (only main, only fast-forward)
      machine checks: on branch main, no modified tracked files, no untracked .py, requirements.txt matches the
      venv stamp, free disk                     -> on a problem: paused + one ntfy; no job is ever blocked for it
  - then runs `run_job.py --queue-dir Q --job X --wait 60 --max-hours H` (H from job.json, default 48) as a child.
Child exit: 0/1 done/blocked -> next job; 3 lock held / 4 GPU busy -> wait; 2 refused -> writes STOP (paused) +
ntfy. One watcher per queue (watch.lock; a second one exits 3). Never writes the registry; deletes nothing.
Self-update: when idle it updates the clone every --update-every seconds (and at start); when its own code
(CODE_FILES) changed, it exits 75 and the supervisor loop in this file starts it again from the new file, so code
pushed to main reaches the watcher without a person. A running job is never interrupted.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import run_job  # noqa: E402

DEFAULT_HOURS = 48.0
EXIT_RESTART = 75                              # the watcher's own code changed: the supervisor starts it again
UPDATE_EVERY_S = 600                           # when idle, update the clone this often (and at start)
CODE_FILES = ("jobs/watch.py", "jobs/common.py", "jobs/run_job.py", "jobs/dataset_cache.py",
              "data_utils/oslock.py")                                       # what this process loaded
GIT_TIMEOUT = 120
STAMP_NAME = "cryoev_requirements.sha256"     # in the venv (sys.prefix): the requirements.txt it was built from


def code_sha(repo: Path) -> str:
    """One hash over the files this process has loaded: when it changes after an update, restart."""
    return hashlib.sha256("".join(f"{f}:{file_sha(repo / f)}" for f in CODE_FILES).encode()).hexdigest()


def file_sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


class Watcher:
    def __init__(self, root: Path, repo: Path, interval=60.0, stamp: Path | None = None, default_hours=DEFAULT_HOURS,
                 gpu_notify_hours=2.0):
        self.root, self.repo, self.interval = root, repo, interval
        self.stamp = stamp or Path(sys.prefix) / STAMP_NAME
        self.default_hours, self.gpu_notify_s = default_hours, gpu_notify_hours * 3600
        self.notified: set = set()
        self.gpu_since = None
        self.started = C.now()
        self.last = None
        self.code = code_sha(repo)
        self.last_update = 0.0

    # ------------------------------------------------------------ reporting
    def log(self, msg):
        line = f"{C.now()} {msg}"
        print(line, flush=True)
        try:
            with open(self.root / C.WATCH_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def state(self, state, detail=None, job=None):
        self.last = f"{state}" + (f": {detail}" if detail else "")
        try:                                           # a status file must never stop the watcher (or its job)
            C.write_json(self.root / C.WATCH_JSON, {
                "schema": "watch/1", "state": state, "detail": detail, "job": job, "heartbeat": C.now(),
                "heartbeat_utc": C.now_utc(), "since": self.started, "pid": os.getpid(), "machine": platform.node(),
                "repo": str(self.repo), "repo_commit": self.git("rev-parse", "HEAD").stdout.strip() or None})
        except OSError as e:
            print(f"[watch] watch.json not written: {e}", flush=True)

    def notify_once(self, key, title, msg):
        """One push per distinct problem until it clears (see clear())."""
        if key in self.notified:
            return
        self.notified.add(key)
        self.log(f"NOTIFY {title}: {msg}")
        try:
            from notify import _send_ntfy
            _send_ntfy(title, msg, priority=4)
        except Exception as e:
            self.log(f"no push: {e}")

    def clear(self, prefix):
        self.notified = {k for k in self.notified if not k.startswith(prefix)}

    # ------------------------------------------------------------ the clone
    def git(self, *args, timeout=30):
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
        try:
            return subprocess.run(["git", "-C", str(self.repo), *args], capture_output=True, text=True,
                                  timeout=timeout, env=env)
        except (OSError, subprocess.SubprocessError) as e:
            return subprocess.CompletedProcess(args, 99, "", str(e))

    def machine_problem(self) -> str | None:
        """Update the clone to origin/main, then check it, the venv and the disk. -> None or the problem."""
        branch = self.git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        if branch != "main":
            return f"the clone is on {branch or '?'}, not main (git -C {self.repo} switch main)"
        st = self.git("status", "--porcelain", "--untracked-files=all").stdout.splitlines()
        modified = [s[3:] for s in st if s.strip() and not s.startswith("??")]
        if modified:
            return f"modified tracked files in the clone: {', '.join(modified[:3])} (the runner clone is never edited)"
        stray = [s[3:] for s in st if s.startswith("??") and s.rstrip().endswith(".py")]
        if stray:
            return f"untracked .py in the clone: {', '.join(stray[:3])} (could change what runs)"
        r = self.git("fetch", "origin", "main", timeout=GIT_TIMEOUT)
        if r.returncode:
            return f"git fetch failed: {(r.stderr or r.stdout).strip()[:200]}"
        r = self.git("merge", "--ff-only", "origin/main")
        if r.returncode:
            return f"git merge --ff-only origin/main failed: {(r.stderr or r.stdout).strip()[:200]}"
        want, have = file_sha(self.repo / "requirements.txt"), (self.stamp.read_text(encoding="utf-8").strip()
                                                                if self.stamp.is_file() else None)
        if want and want != have:
            return (f"venv out of date: requirements.txt changed ({'no stamp' if have is None else 'stamp differs'}); "
                    f"run pip install -r {self.repo / 'requirements.txt'}, then watch.py --stamp-venv")
        free = run_job.free_gb(self.root)
        if free < run_job.MIN_FREE_GB:
            return f"disk: {free:.1f} GB free on the queue drive (need {run_job.MIN_FREE_GB})"
        return None

    def idle_update(self, every_s) -> bool:
        """When idle: at most every `every_s`, update the clone (holding runner.lock, like before a job) and run
        the machine checks. -> True when this process's own code changed (time to restart)."""
        if time.time() - self.last_update < every_s:
            return False
        lock = run_job.acquire_lock(self.root / C.LOCK, 0)
        if lock is None:
            return False
        try:
            self.last_update = time.time()
            problem = self.machine_problem()
            if problem:
                self.state("paused", problem)
                self.notify_once("machine:" + problem[:60], "Desktop watcher paused", problem)
                return False
            self.clear("machine:")
        finally:
            lock.close()
        return self.code_changed()

    def code_changed(self) -> bool:
        return code_sha(self.repo) != self.code

    # ------------------------------------------------------------ one loop
    def child_argv(self, qdir: Path):
        try:
            hours = C.read_json(qdir / C.JOB_FILE).get("max_hours") or self.default_hours
        except (OSError, ValueError):
            hours = self.default_hours
        return [sys.executable, str(self.repo / "jobs" / "run_job.py"), "--queue-dir", str(self.root),
                "--job", qdir.name, "--wait", "60", "--max-hours", str(hours)]

    def step(self) -> str:
        """One pass. -> 'ran' (go again now), 'idle', 'wait' or 'paused'."""
        if (self.root / C.STOP).exists():
            why = (self.root / C.STOP).read_text(encoding="utf-8", errors="replace").strip()[:200]
            self.state("paused", f"STOP: {why} (sync.py start resumes)")
            return "paused"
        qdir = run_job.pick(self.root)
        if qdir is None:
            self.state("idle")
            return "idle"
        lock = run_job.acquire_lock(self.root / C.LOCK, 0)
        if lock is None:
            self.state("waiting", "runner.lock is held (a runner started by hand?)", qdir.name)
            return "wait"
        try:
            busy = run_job.gpu_state()["busy"]
            if busy:
                self.gpu_since = self.gpu_since or time.time()
                waited = time.time() - self.gpu_since
                who = ", ".join(f"pid {p} {Path(n).name} {m} MB" for p, n, m in busy[:3])
                self.state("waiting", f"GPU busy for {waited / 60:.0f} min: {who}", qdir.name)
                if waited >= self.gpu_notify_s:
                    self.notify_once("gpu", "Desktop GPU busy", f"{qdir.name} waiting {waited / 3600:.1f} h: {who}")
                return "wait"
            self.gpu_since = None
            self.clear("gpu")
            problem = self.machine_problem()
            if problem:
                self.state("paused", problem, qdir.name)
                self.notify_once("machine:" + problem[:60], "Desktop watcher paused", problem)
                return "paused"
            self.clear("machine:")
        finally:
            lock.close()
        return self.run_child(qdir)

    def run_child(self, qdir: Path) -> str:
        argv = self.child_argv(qdir)
        self.log(f"start {qdir.name}: {' '.join(argv[2:])}")
        p = subprocess.Popen(argv, cwd=self.repo)
        while True:                                    # heartbeat while the job runs
            self.state("running", None, qdir.name)
            try:
                rc = p.wait(timeout=max(self.interval, 1))
                break
            except subprocess.TimeoutExpired:
                continue
        action = after_child(rc)
        self.log(f"end {qdir.name}: exit {rc} -> {action}")
        if action == "pause":
            why = f"run_job.py refused {qdir.name} (exit {rc}); see watch.log"
            (self.root / C.STOP).write_text(f"{C.now()} watcher {platform.node()}: {why}\n", encoding="utf-8")
            self.notify_once("refused:" + qdir.name, "Desktop watcher paused", why)
            return "paused"
        return "ran" if action == "next" else "wait"


def after_child(rc: int) -> str:
    """run_job.py exit code -> what the watcher does next."""
    if rc in (0, 1):
        return "next"                                  # done / blocked (DONE.json written)
    if rc in (3, run_job.EXIT_GPU_BUSY):
        return "wait"                                  # lock held / GPU busy: the job is untouched
    if rc == 2:
        return "pause"                                 # refused at start: a config problem, needs a person
    return "wait"                                      # crashed: the next try sets .tmp_ aside (stale cap)


def stamp_venv(repo: Path, stamp: Path):
    sha = file_sha(repo / "requirements.txt")
    if not sha:
        raise SystemExit(f"no {repo / 'requirements.txt'}")
    stamp.write_text(sha + "\n", encoding="utf-8")
    print(f"[watch] venv stamp {stamp} = {sha[:12]} (requirements.txt)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue-dir", type=Path, help="queue root (holds q/, d/, work/)")
    ap.add_argument("--repo", type=Path, default=HERE.parent, help="the runner clone (default: this repo)")
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between checks (default 60)")
    ap.add_argument("--stamp", type=Path, help=f"venv requirements stamp (default <sys.prefix>/{STAMP_NAME})")
    ap.add_argument("--stamp-venv", action="store_true", help="record requirements.txt as installed, then exit")
    ap.add_argument("--default-hours", type=float, default=DEFAULT_HOURS, help="--max-hours when job.json has none")
    ap.add_argument("--gpu-notify-hours", type=float, default=2.0, help="push once after waiting this long for the GPU")
    ap.add_argument("--max-loops", type=int, help="stop after this many passes (tests)")
    ap.add_argument("--exit-when-idle", action="store_true", help="exit when no job is ready (tests)")
    ap.add_argument("--update-every", type=float, default=UPDATE_EVERY_S,
                    help=f"when idle, update the clone every this many seconds (default {UPDATE_EVERY_S:g})")
    ap.add_argument("--no-supervise", action="store_true", help="run the watcher in this process (the supervisor "
                    "uses it for its child; a restart after an update then needs a person)")
    a = ap.parse_args(argv)
    repo = a.repo.resolve()
    stamp = a.stamp or Path(sys.prefix) / STAMP_NAME
    if a.stamp_venv:
        stamp_venv(repo, stamp)
        return 0
    if not a.queue_dir:
        ap.error("--queue-dir is required")
    root = a.queue_dir.resolve()
    if not (root / C.Q).is_dir():
        print(f"[watch] {root} has no {C.Q}/ folder", file=sys.stderr)
        return 2
    lock = run_job.acquire_lock(root / C.WATCH_LOCK, 0)
    if lock is None:
        print("[watch] another watcher holds watch.lock; not starting", file=sys.stderr)
        return 3
    w = Watcher(root, repo, a.interval, stamp, a.default_hours, a.gpu_notify_hours)
    w.log(f"watcher started: queue {root}, repo {repo}, pid {os.getpid()}")
    loops, rc = 0, 0
    try:
        while a.max_loops is None or loops < a.max_loops:
            loops += 1
            result = w.step()
            if result in ("idle", "paused") and w.idle_update(a.update_every) or result == "ran" and w.code_changed():
                w.log(f"code updated to {w.git('rev-parse', '--short', 'HEAD').stdout.strip()}: restarting")
                rc = EXIT_RESTART
                break
            if result == "idle" and a.exit_when_idle:
                break
            if result != "ran":
                time.sleep(a.interval)
    except KeyboardInterrupt:
        w.log("stopped (Ctrl+C)")
    finally:
        try:
            w.state("restarting" if rc == EXIT_RESTART else "stopped", f"last: {w.last}")
        except OSError:
            pass
        lock.close()
    return rc


def supervise(argv) -> int:
    """Run the watcher as a child and start it again, from the (updated) file on disk, when it exits 75 after an
    update. This loop is all that stays in memory from the old code. Gives up after 10 restarts in an hour."""
    if "--no-supervise" in argv or "--stamp-venv" in argv or "-h" in argv or "--help" in argv:
        return main([x for x in argv if x != "--no-supervise"])
    starts = []
    while True:
        try:
            rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), *argv, "--no-supervise"])
        except KeyboardInterrupt:
            return 0
        if rc != EXIT_RESTART:
            return rc
        starts = [t for t in starts if time.time() - t < 3600] + [time.time()]
        if len(starts) > 10:
            print("[watch] 10 restarts within an hour: stopping (see watch.log)", file=sys.stderr)
            return 1
        time.sleep(1)


if __name__ == "__main__":
    sys.exit(supervise(sys.argv[1:]))
