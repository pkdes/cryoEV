"""jobs/watch.py: the desktop watcher, simulated with a temp git origin ("GitHub"), a runner clone, the fake trainer.

The runner clone holds a copy of jobs/, data_utils/oslock.py and training/run_experiments.py committed to a temp bare
origin, so the child run_job.py reports the clone's commit. Known answers: an older bundle runs on the newer main and
records it; a stop/problem/busy GPU never touches a queued job; a refused child pauses the watcher (STOP).
"""
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobtest as T  # noqa: E402
from jobtest import C, run_job  # noqa: E402

import sync  # noqa: E402
import watch  # noqa: E402

COPY = ["jobs/common.py", "jobs/run_job.py", "jobs/watch.py", "jobs/sync.py", "data_utils/__init__.py",
        "data_utils/oslock.py", "training/run_experiments.py", "requirements.txt"]


def git(cwd, *args):
    r = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, capture_output=True,
                       text=True)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr}")
    return r.stdout.strip()


class WatchTests(T.TempRoots, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.origin, self.dev, self.repo = self.tmp / "origin.git", self.tmp / "dev", self.tmp / "cj_repo"
        git(self.tmp, "init", "--bare", "-b", "main", str(self.origin))
        git(self.tmp, "clone", str(self.origin), str(self.dev))
        for rel in COPY:
            (self.dev / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(T.REPO / rel, self.dev / rel)
        git(self.dev, "checkout", "-b", "main")
        git(self.dev, "add", "-A")
        git(self.dev, "commit", "-m", "c1")
        git(self.dev, "push", "origin", "main")
        self.c1 = git(self.dev, "rev-parse", "HEAD")
        git(self.tmp, "clone", str(self.origin), str(self.repo))
        self.stamp = self.tmp / "stamp.sha256"
        self.stamp.write_text(watch.file_sha(self.repo / "requirements.txt") + "\n", encoding="utf-8")
        self.hooks_file = self.tmp / "hooks.json"

    def new_commit(self, name="notes.txt"):
        (self.dev / name).write_text(name, encoding="utf-8")
        git(self.dev, "add", name)
        git(self.dev, "commit", "-m", name)
        git(self.dev, "push", "origin", "main")
        return git(self.dev, "rev-parse", "HEAD")

    def push(self, job, commit=None, stamp="20261104-120000", **kw):
        zp = T.make_bundle(self.bundles / job, commit=commit or self.c1, stamp=stamp, **kw)
        return T.queue(self.share, job, zp)

    def watch(self, *extra, loops=30, idle_exit=True, **hook_kw):
        env = {**os.environ, **T.hooks_env(**hook_kw), run_job.HOOKS_FILE_ENV: str(self.hooks_file)}
        argv = [sys.executable, str(self.repo / "jobs" / "watch.py"), "--queue-dir", str(self.share), "--repo",
                str(self.repo), "--interval", "0.2", "--stamp", str(self.stamp), "--max-loops", str(loops), *extra]
        if idle_exit:
            argv.append("--exit-when-idle")
        r = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=300)
        return r.returncode, r.stdout + r.stderr

    def done(self, job):
        return C.read_json(self.share / C.D / job / C.DONE)

    def last_state(self):
        return C.read_json(self.share / C.WATCH_JSON)["detail"]

    def untouched(self, job):
        self.assertFalse((self.share / C.D / job).exists(), f"{job} was touched")
        self.assertTrue((self.share / C.Q / job / C.READY).is_file(), f"{job} not queued")

    def test_older_bundle_runs_on_latest_main(self):
        self.push("job_20261104_a")
        rc, out = self.watch()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.done("job_20261104_a")["status"], "done", out)
        self.assertEqual(self.done("job_20261104_a")["runner_commit"], self.c1)
        c2 = self.new_commit()                                  # pushed to "GitHub" between the two jobs
        self.push("job_20261104_b", stamp="20261104-120001")    # bundle still names c1
        rc, out = self.watch()
        d = self.done("job_20261104_b")
        self.assertEqual((d["status"], d["cryoEV_commit"], d["runner_commit"]), ("done", self.c1, c2), out)
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), c2, "the clone fast-forwarded")
        self.assertIn("end job_20261104_b: exit 0 -> next", (self.share / C.WATCH_LOG).read_text(encoding="utf-8"))
        self.assertIn("max-hours 48", out)                      # default cap

    def test_stop_pauses_start_resumes(self):
        q = self.push("job_20261104_a")
        sync.stop(self.share)
        rc, out = self.watch(loops=3, idle_exit=False)
        self.untouched(q.name)
        self.assertIn("paused: STOP", self.last_state())
        sync.start(self.share)
        self.assertFalse((self.share / C.STOP).exists())
        self.assertEqual(len(list((self.share / C.RETIRED).glob("STOP_*"))), 1, "renamed, not deleted")
        self.watch()
        self.assertEqual(self.done(q.name)["status"], "done")

    def test_gpu_busy_waits_then_runs(self):
        q = self.push("job_20261104_a")
        self.hooks_file.write_text(json.dumps({"gpu": {"total_mb": 49140, "apps": [], "used_mb": 6000}}))
        self.watch(loops=3, idle_exit=False)
        self.untouched(q.name)
        self.assertIn("waiting: GPU busy", self.last_state())
        self.hooks_file.write_text(json.dumps({"gpu": {"total_mb": 49140, "apps": [], "used_mb": 500}}))
        self.watch()
        self.assertEqual(self.done(q.name)["status"], "done")

    def test_machine_problems_pause_and_block_nothing(self):
        q = self.push("job_20261104_a")
        cases = [("modified tracked files", lambda: (self.repo / "jobs" / "common.py").open("a").write("# x\n"),
                  lambda: git(self.repo, "checkout", "--", "jobs/common.py")),
                 ("untracked .py", lambda: (self.repo / "jobs" / "evil.py").write_text("x = 1\n"),
                  lambda: (self.repo / "jobs" / "evil.py").unlink()),
                 ("venv out of date", lambda: self.stamp.write_text("0" * 64 + "\n"),
                  lambda: self.stamp.write_text(watch.file_sha(self.repo / "requirements.txt") + "\n"))]
        for want, break_it, fix_it in cases:
            break_it()
            self.watch(loops=2, idle_exit=False)
            self.untouched(q.name)
            self.assertIn(want, self.last_state())
            fix_it()
        self.watch()                                            # control: fixed -> it runs
        self.assertEqual(self.done(q.name)["status"], "done")

    def test_refused_child_pauses_the_watcher(self):
        q = self.push("job_20261104_a")
        rc, out = self.watch(loops=3, gpu=T.LAPTOP_GPU, on_battery=True)   # run_job.py exits 2 (on battery)
        self.untouched(q.name)
        self.assertTrue((self.share / C.STOP).is_file())
        self.assertIn("refused", (self.share / C.STOP).read_text(encoding="utf-8"))
        self.assertEqual(out.count(f"start {q.name}"), 1, "not retried every loop")

    def test_held_runner_lock_waits(self):
        q = self.push("job_20261104_a")
        (self.share / C.LOCK).touch()
        held = run_job.acquire_lock(self.share / C.LOCK)
        try:
            self.watch(loops=2, idle_exit=False)
        finally:
            held.close()
        self.untouched(q.name)
        self.assertIn("runner.lock is held", self.last_state())

    def test_second_watcher_exits_3(self):
        held = run_job.acquire_lock(self.share / C.WATCH_LOCK)
        try:
            rc, out = self.watch(loops=1)
        finally:
            held.close()
        self.assertEqual(rc, 3, out)

    def test_after_child(self):
        self.assertEqual([watch.after_child(rc) for rc in (0, 1, 2, 3, 4, -1)],
                         ["next", "next", "pause", "wait", "wait", "wait"])


class OriginTests(unittest.TestCase):
    """sync.push refuses a bundle whose commit isn't on origin/main (the desktop trains with main)."""

    def test_unpushed_commit_refused(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="cj_o_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        origin, dev = tmp / "origin.git", tmp / "dev"
        git(tmp, "init", "--bare", "-b", "main", str(origin))
        git(tmp, "clone", str(origin), str(dev))
        git(dev, "checkout", "-b", "main")
        (dev / "a.txt").write_text("a")
        git(dev, "add", "a.txt")
        git(dev, "commit", "-m", "a")
        git(dev, "push", "origin", "main")
        pushed = git(dev, "rev-parse", "HEAD")
        (dev / "b.txt").write_text("b")
        git(dev, "add", "b.txt")
        git(dev, "commit", "-m", "b")                            # local only
        local = git(dev, "rev-parse", "HEAD")
        (tmp / "s").mkdir()
        with mock.patch.object(sync, "REPO", dev):
            self.assertIsNone(sync.commit_on_origin(pushed))     # control
            self.assertIn("push the repo first", sync.commit_on_origin(local))
            zp = T.make_bundle(tmp / "b", commit=local)
            with self.assertRaisesRegex(sync.SyncError, "push the repo first"):
                sync.push("job_20261104_x", zp, tmp / "s", copier="shutil", check_registry=False)
            self.assertFalse((tmp / "s" / C.Q / "job_20261104_x").exists())

    def test_hold_then_release(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="cj_h_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        zp = T.make_bundle(tmp / "b")
        (tmp / "s").mkdir()
        q = sync.push("job_20261104_h", zp, tmp / "s", copier="shutil", check_registry=False, check_origin=False,
                      hold=True, max_hours=6)
        self.assertFalse((q / C.READY).exists())
        self.assertIsNone(run_job.pick(tmp / "s"), "a held job is not picked")
        self.assertEqual(C.read_json(q / C.JOB_FILE)["max_hours"], 6)
        sync.release("job_20261104_h", tmp / "s")
        self.assertEqual(run_job.pick(tmp / "s"), q)
        with self.assertRaisesRegex(sync.SyncError, "already released"):
            sync.release("job_20261104_h", tmp / "s")


if __name__ == "__main__":
    unittest.main()
