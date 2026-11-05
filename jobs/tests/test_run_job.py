"""jobs/run_job.py with temp queue roots and a fake trainer (no GPU).

Known-answer controls: the same bundle that is blocked "desktop only" on a 4 GB GPU runs to done on a 48 GB
one, and a y8n@320 job runs on the 4 GB GPU; the lock tests first show a free lock lets the runner through.
"""
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobtest as T  # noqa: E402
from jobtest import C, run_job


class RunJobTests(T.TempRoots, unittest.TestCase):
    def push(self, job="job_20261102_t1", **kw):
        return T.queue(self.share, job, T.make_bundle(self.bundles, **kw))

    def test_done_end_to_end(self):
        q = self.push(args=["--model-size", "n", "--imgsz", "320"])
        self.assertTrue((q / C.READY).is_file())
        self.assertEqual(self.run_job(), 0)
        d = self.share / C.D / q.name
        done = self.done(q.name)
        self.assertEqual(done["status"], "done", done)
        self.assertEqual(done["bundle_id"], "ds_tiny__job_20261102-120000")
        self.assertEqual(C.read_json(d / "run.json")["status"], "finished")
        self.assertTrue((d / "runs" / done["run"] / "weights" / "best.pt").is_file())
        self.assertTrue((d / "runs" / done["run"] / "train.log").is_file())
        self.assertEqual(C.read_json(d / C.STATUS)["status"], "done")
        self.assertFalse((self.share / C.D / f".tmp_{q.name}").exists())
        self.assertFalse((self.share / C.WORK / q.name).exists(), "work folder kept after done")
        self.assertTrue((q / C.READY).is_file(), "runner must not touch q/")
        self.assertEqual(self.run_job(), 0)                     # nothing left: done jobs are not rerun
        self.assertEqual(len([p for p in (self.share / C.D).iterdir()]), 1)

    def test_no_ready_is_ignored(self):
        q = self.push()
        (q / C.READY).unlink()
        self.assertEqual(self.run_job(), 0)
        self.assertFalse((self.share / C.D / q.name).exists())

    def test_incoming_is_ignored(self):
        inc = self.share / C.Q / C.INCOMING / "job_20261102_t1"
        inc.mkdir(parents=True)
        (inc / C.READY).write_text("x")
        self.assertEqual(self.run_job(), 0)
        self.assertFalse((self.share / C.D).exists() and any((self.share / C.D).iterdir()))

    def test_oldest_ready_first(self):
        a = self.push("job_20261102_b_newer", stamp="20261102-120001")
        b = self.push("job_20261102_a_older", stamp="20261102-120002")
        os.utime(a / C.READY, (time.time(), time.time()))
        os.utime(b / C.READY, (time.time() - 100, time.time() - 100))
        self.assertEqual(run_job.pick(self.share).name, b.name)

    def test_failed_run_is_blocked(self):
        q = self.push()
        with mock.patch.dict(os.environ, {"FAKE_TRAINER": "error"}):
            self.assertEqual(self.run_job(), 1)
        done = self.done(q.name)
        self.assertEqual(done["status"], "blocked")
        self.assertIn("run error", done["reason"])
        self.assertTrue((self.share / C.WORK / q.name).exists(), "work kept for a blocked job")

    def test_status_from_run_json_not_exit_code(self):
        q = self.push()
        with mock.patch.dict(os.environ, {"FAKE_TRAINER": "error_exit0"}):
            self.assertEqual(self.run_job(), 1)
        self.assertEqual(self.done(q.name)["status"], "blocked")
        self.assertEqual(C.read_json(self.share / C.D / q.name / C.STATUS)["exit_code"], 0)

    def test_finished_run_json_but_nonzero_exit_is_blocked(self):
        q = self.push()
        with mock.patch.dict(os.environ, {"FAKE_TRAINER": "finished_exit1"}):
            self.assertEqual(self.run_job(), 1)
        done = self.done(q.name)
        self.assertEqual((done["status"], done["run_status"]), ("blocked", "finished"))
        self.assertIn("exit 1 after a finished run.json", done["reason"])

    def test_missing_yaml_is_blocked_fake(self):
        q = self.push(yaml=False)
        self.assertEqual(self.run_job(), 1)
        self.assertIn("no run.json", self.done(q.name)["reason"])

    def test_missing_yaml_is_blocked_real_run_experiments(self):
        """The real training/run_experiments.py (stops at the missing dataset.yaml, before any GPU use)."""
        q = self.push(yaml=False)
        self.assertEqual(self.run_job(train_cmd=None), 1)
        done = self.done(q.name)
        self.assertEqual(done["status"], "blocked")
        self.assertIn("no run.json", done["reason"])
        self.assertIn("dataset.yaml", (self.share / C.D / q.name / "train.log").read_text(encoding="utf-8"))

    def test_failed_real_run_writes_run_json_and_exits_nonzero(self):
        """The real run_experiments.py with missing --weights: fails before the GPU, writes run.json status
        error and exits 1 (it used to exit 0)."""
        missing = str(self.tmp / "no_such_dir" / "nope.pt")
        q = self.push(args=["--model-size", "n", "--imgsz", "320", "--epochs", "1", "--workers", "0",
                            "--weights", missing])
        self.assertEqual(self.run_job(train_cmd=None), 1)
        done = self.done(q.name)
        self.assertEqual(done["status"], "blocked", done)
        self.assertEqual(C.read_json(self.share / C.D / q.name / C.STATUS)["exit_code"], 1)
        rj = C.read_json(self.share / C.D / q.name / "run.json")
        self.assertEqual(rj["status"], "error")
        self.assertEqual(rj["bundle_id"], "ds_tiny__job_20261102-120000")
        self.assertEqual((rj["config"]["imgsz"], rj["config"]["workers"], rj["config"]["weights_init"]),
                         (320, 0, missing))
        self.assertTrue(done["reason"].startswith("run error"), done["reason"])

    def test_desktop_only_on_small_gpu_with_controls(self):
        q = self.push(args=["--model-size", "m", "--imgsz", "1440"])
        self.assertEqual(self.run_job("--max-hours", "1", gpu=T.LAPTOP_GPU), 1)
        self.assertIn("desktop only", self.done(q.name)["reason"])
        # control 1: the same bundle on the desktop GPU runs
        q2 = self.push("job_20261102_t2", args=["--model-size", "m", "--imgsz", "1440"], stamp="20261102-120003")
        self.assertEqual(self.run_job(gpu=T.DESKTOP_GPU), 0)
        self.assertEqual(self.done(q2.name)["status"], "done")
        # control 2: y8n at 320 runs on the laptop GPU; the default imgsz (1440) does not
        q3 = self.push("job_20261102_t3", args=["--model-size", "n", "--imgsz", "320", "--workers", "2"],
                       stamp="20261102-120004")
        self.assertEqual(self.run_job("--max-hours", "1", gpu=T.LAPTOP_GPU), 0)
        self.assertEqual(self.done(q3.name)["status"], "done")
        q4 = self.push("job_20261102_t4", args=["--model-size", "n", "--workers", "2"], stamp="20261102-120005")
        self.assertEqual(self.run_job("--max-hours", "1", gpu=T.LAPTOP_GPU), 1)
        self.assertIn("imgsz 1440", self.done(q4.name)["reason"])

    def test_allow_big_on_small_gpu(self):
        big = ["--model-size", "m", "--imgsz", "640", "--workers", "2"]
        q = self.push(args=big)
        self.assertEqual(self.run_job("--allow-big", gpu=T.LAPTOP_GPU), 2)          # no --max-hours: refused
        self.assertEqual(self.run_job(gpu=T.LAPTOP_GPU), 2)                        # the laptop always needs it
        self.assertFalse((self.share / C.D / q.name).exists(), "refused at start: still queued")
        self.assertEqual(self.run_job("--max-hours", "1", gpu=T.LAPTOP_GPU), 1)                        # control: no flag, desktop only
        self.assertIn("desktop only", self.done(q.name)["reason"])
        q3 = self.push("job_20261102_t3", args=big, stamp="20261102-120004")
        self.assertEqual(self.run_job("--allow-big", "--max-hours", "1", gpu=T.LAPTOP_GPU), 0)
        d = self.done(q3.name)
        self.assertEqual((d["status"], d["allow_big"], d["max_hours"], d["time_capped"]), ("done", True, 1.0, None))
        q4 = self.push("job_20261102_t4", args=["--model-size", "n", "--imgsz", "320"], stamp="20261102-120005")
        self.assertEqual(self.run_job("--max-hours", "1", gpu=T.LAPTOP_GPU), 1)                        # default 8 workers on the laptop
        self.assertIn("--workers 8 (default)", self.done(q4.name)["reason"])
        self.assertIsNone(self.done(q4.name)["allow_big"])

    def test_max_hours_stops_the_trainer(self):
        q = self.push()
        t0 = time.time()
        with mock.patch.dict(os.environ, {"FAKE_TRAINER": "hang"}):
            self.assertEqual(self.run_job("--max-hours", "0.001"), 1)               # 3.6 s
        self.assertLess(time.time() - t0, 60, "the trainer was stopped, not waited for (it sleeps 600 s)")
        d = self.done(q.name)
        self.assertTrue(d["time_capped"])
        self.assertIn("time_cap", d["reason"])
        run = self.share / C.D / q.name / "runs" / d["run"]
        self.assertTrue((run / "weights" / "last.pt").is_file(), "partial run kept")
        self.assertEqual(C.read_json(run / "run.json")["status"], "time_cap")

    def test_cancel_stops_a_running_job(self):
        import threading
        import sync
        q = self.push()
        other = self.push("job_20261102_t2", stamp="20261102-120003")
        sync.cancel("job_20261102_t2", self.share, reason="control: another job")   # must not touch t1

        def cancel_soon():
            started = self.share / C.WORK / q.name / "fake_started.txt"
            for _ in range(200):
                if started.exists():
                    break
                time.sleep(0.05)
            time.sleep(0.5)
            sync.cancel(q.name, self.share, reason="not going well")
        threading.Thread(target=cancel_soon, daemon=True).start()
        t0 = time.time()
        with mock.patch.object(run_job, "CANCEL_POLL_S", 0.3), mock.patch.dict(os.environ, {"FAKE_TRAINER": "hang"}):
            self.assertEqual(self.run_job("--job", q.name), 1)
        self.assertLess(time.time() - t0, 30, "stopped, not waited for (the fake trainer sleeps 600 s)")
        d = self.done(q.name)
        self.assertEqual(d["status"], "cancelled")
        self.assertIn("not going well", d["reason"])
        run = self.share / C.D / q.name / "runs" / d["run"]
        self.assertTrue((run / "weights" / "last.pt").is_file(), "partial run kept")
        self.assertEqual(C.read_json(run / "run.json")["status"], "cancelled")
        # the job cancelled while queued is never started
        self.assertEqual(self.run_job("--job", other.name), 1)
        d2 = self.done(other.name)
        self.assertEqual(d2["status"], "cancelled")
        self.assertIn("before it started", d2["reason"])
        self.assertFalse((self.share / C.WORK / other.name).exists(), "never unzipped")
        with self.assertRaisesRegex(sync.SyncError, "already finished"):
            sync.cancel(q.name, self.share)

    def test_laptop_refuses_battery_and_network_queue_untouched(self):
        q = self.push(args=["--model-size", "n", "--imgsz", "320", "--workers", "2"])
        self.assertEqual(self.run_job(gpu=T.LAPTOP_GPU, on_battery=True), 2)
        self.assertFalse((self.share / C.D / q.name).exists(), "a refused start leaves the job queued")
        with mock.patch.object(run_job, "is_network_path", lambda p: True):
            self.assertEqual(self.run_job("--max-hours", "1", gpu=T.LAPTOP_GPU), 2)
            self.assertFalse((self.share / C.D / q.name).exists())
            self.assertEqual(self.run_job(gpu=T.DESKTOP_GPU), 0)      # control: the desktop runs its own share
        self.assertEqual(self.done(q.name)["status"], "done")
        self.assertTrue(run_job.is_network_path("\\\\desk\\cj"))
        self.assertFalse(run_job.is_network_path(self.share))

    def test_low_commit_memory_is_blocked(self):
        q = self.push(args=["--model-size", "n", "--imgsz", "320", "--workers", "2"])
        self.assertEqual(self.run_job("--max-hours", "1", gpu=T.LAPTOP_GPU, free_commit_gb=6), 1)   # needs 4 + 2x2
        self.assertIn("memory: 6.0 GB commit free", self.done(q.name)["reason"])
        q2 = self.push("job_20261102_t2", args=["--model-size", "n", "--imgsz", "320", "--workers", "0"],
                       stamp="20261102-120003")
        self.assertEqual(self.run_job("--max-hours", "1", gpu=T.LAPTOP_GPU, free_commit_gb=6), 0)   # control: 0 workers
        self.assertEqual(self.done(q2.name)["status"], "done")

    def test_low_disk_is_blocked(self):
        q = self.push()
        self.assertEqual(self.run_job(free_gb=5), 1)
        self.assertIn("disk: 5.0 GB free", self.done(q.name)["reason"])

    def test_hooks_need_the_testing_guard(self):
        env = T.hooks_env(commit="deadbeefcafe")
        with mock.patch.dict(os.environ, {run_job.HOOKS_ENV: env[run_job.HOOKS_ENV]}):
            self.assertNotEqual(run_job.repo_commit(), "deadbeefcafe")   # production ignores the hooks
        with mock.patch.dict(os.environ, env):
            self.assertEqual(run_job.repo_commit(), "deadbeefcafe")

    def test_md5_mismatch_is_blocked(self):
        q = self.push()
        zp = next(q.glob("*.zip"))
        with open(zp, "ab") as f:
            f.write(b"x")
        self.assertEqual(self.run_job(), 1)
        self.assertIn("md5 mismatch", self.done(q.name)["reason"])

    def test_path_too_long_is_blocked(self):
        long = "images/train/" + "x" * 230 + ".png"
        q = self.push(files={long: b"png"})
        self.assertEqual(self.run_job(), 1)
        self.assertIn("path too long", self.done(q.name)["reason"])
        self.assertFalse((self.share / C.WORK / q.name / "training outputs").exists(), "nothing unzipped")

    def test_commit_mismatch_is_blocked(self):
        q = self.push(commit="deadbee")
        self.assertEqual(self.run_job(), 1)
        self.assertIn("commit mismatch", self.done(q.name)["reason"])

    def test_ultralytics_version_is_blocked(self):
        q = self.push()
        self.assertEqual(self.run_job(ultralytics="8.3.40"), 1)
        self.assertIn("Ultralytics 8.3.40", self.done(q.name)["reason"])

    def test_gpu_busy_leaves_the_job_queued(self):
        q = self.push()
        self.assertEqual(self.run_job(gpu={"total_mb": 49140, "apps": [[4242, "C:/py/python.exe", 9000]]}),
                         run_job.EXIT_GPU_BUSY)
        self.assertFalse((self.share / C.D).exists() and any((self.share / C.D).iterdir()), "job untouched")
        # WDDM: no per-app memory, but 3 GB in use in total -> busy too
        self.assertEqual(self.run_job(gpu={"total_mb": 49140, "apps": [], "used_mb": 3000}), run_job.EXIT_GPU_BUSY)
        self.assertEqual(self.run_job(gpu={"total_mb": 49140, "apps": [], "used_mb": 900}), 0)   # control: idle
        self.assertEqual(self.done(q.name)["status"], "done")

    def test_older_bundle_commit_runs_on_newer_code(self):
        prev = subprocess.run(["git", "rev-parse", "--short", "HEAD~1"], cwd=T.REPO, capture_output=True,
                              text=True).stdout.strip()
        q = self.push(commit=prev)
        self.assertEqual(self.run_job(), 0)
        d = self.done(q.name)
        self.assertEqual(d["status"], "done")
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=T.REPO, capture_output=True, text=True).stdout.strip()
        self.assertEqual((d["cryoEV_commit"], d["runner_commit"]), (prev, head))   # bundle vs code that trained

    def test_download_key_is_blocked(self):
        q = self.push(yaml_text=T.YAML + "download: https://example.invalid/x.zip" + chr(10))
        self.assertEqual(self.run_job(), 1)
        self.assertIn("download: key", self.done(q.name)["reason"])

    def test_killed_twice_is_blocked_not_rerun(self):
        q = self.push()
        dd = self.share / C.D
        dd.mkdir()
        (dd / f".tmp_{q.name}.stale_20261104-010000").mkdir()      # one earlier interrupted attempt
        (dd / f".tmp_{q.name}").mkdir()                              # and the one just killed
        self.assertEqual(self.run_job(), 1)
        self.assertIn("killed twice", self.done(q.name)["reason"])
        self.assertFalse((self.share / C.WORK / q.name).exists(), "not run a third time")

    def test_weights_repointed_into_work(self):
        job = {"run_experiments_args": ["--dataset", "d", "--weights", "../CryoAI/training outputs/d/weights/w.pt"]}
        with mock.patch.dict(os.environ, T.hooks_env(train_cmd=None)):
            argv = run_job.train_argv(job, Path("C:/cj/work/j"))
        self.assertEqual(argv[argv.index("--weights") + 1], str(Path("C:/cj/work/j/training outputs/d/weights/w.pt")))
        self.assertIn("--single-run", argv)
        self.assertIn("--no-notify", argv)

    def test_best_config_defaults_read_without_import(self):
        d = run_job.best_config_defaults()
        self.assertEqual((d["model_size"], d["imgsz"]), ("n", 1440))


class LockTests(T.TempRoots, unittest.TestCase):
    """Real processes: the msvcrt lock is per process handle."""

    def runner(self, *extra, env_extra=None):
        env = {**os.environ, **T.hooks_env(), **(env_extra or {})}
        return subprocess.Popen([sys.executable, str(T.JOBS / "run_job.py"), "--queue-dir", str(self.share), *extra],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def test_second_runner_skips_then_waits(self):
        q = T.queue(self.share, "job_20261102_t1", T.make_bundle(self.bundles))
        held = run_job.acquire_lock(self.share / C.LOCK)
        self.assertIsNotNone(held)
        p = self.runner()
        out, _ = p.communicate(timeout=120)
        self.assertEqual(p.returncode, 3, out)
        self.assertIn("lock", out)
        self.assertFalse((self.share / C.D / q.name).exists())
        p = self.runner("--wait", "60")                      # waits for the lock, then runs
        time.sleep(2)
        self.assertIsNone(p.poll(), "should still be waiting")
        held.close()
        out, _ = p.communicate(timeout=120)
        self.assertEqual(p.returncode, 0, out)
        self.assertEqual(self.done(q.name)["status"], "done")

    def test_killed_runner_releases_lock(self):
        q = T.queue(self.share, "job_20261102_t1", T.make_bundle(self.bundles))
        p = self.runner(env_extra={"FAKE_SLEEP": "120"})
        marker = self.share / C.WORK / q.name / "fake_started.txt"
        t0 = time.time()
        while not marker.exists() and time.time() - t0 < 60:
            time.sleep(0.3)
        self.assertTrue(marker.exists(), "fake trainer never started")
        time.sleep(0.5)
        p.kill()
        p.communicate(timeout=30)
        subprocess.run(["taskkill", "/F", "/PID", marker.read_text().strip()] if os.name == "nt"
                       else ["kill", "-9", marker.read_text().strip()], capture_output=True)
        self.assertTrue((self.share / C.D / f".tmp_{q.name}").is_dir())
        self.assertFalse((self.share / C.D / q.name).exists())
        p2 = self.runner()
        out, _ = p2.communicate(timeout=120)
        self.assertEqual(p2.returncode, 0, out)
        self.assertEqual(self.done(q.name)["status"], "done")
        stale = [x.name for x in (self.share / C.D).iterdir() if ".stale_" in x.name]
        self.assertEqual(len(stale), 1, "the killed attempt's .tmp_ is set aside, not deleted")


class RunExperimentsTests(unittest.TestCase):
    """training/run_experiments.py: OOM classification (imports torch; a few seconds)."""

    def test_classify_error(self):
        code = (
            "import numpy as np, run_experiments as r\n"
            "from numpy._core._exceptions import _ArrayMemoryError\n"
            "am = _ArrayMemoryError((10**11,), np.dtype('f8'))\n"
            "cases = [RuntimeError('CUDA out of memory. Tried'), am, OSError(22, 'The paging file is too small', None, 1455),\n"
            "         RuntimeError('DataLoader worker exited: [WinError 1455] The paging file is too small'),\n"
            "         ValueError('bad'), FileNotFoundError('x.pt')]\n"
            "print([r.classify_error(c) for c in cases], type(am).__module__)\n")
        env = {**os.environ, "CRYOEV_CRYOAI_ROOT": str(T.TESTS)}
        r = subprocess.run([sys.executable, "-c", code], cwd=T.REPO / "training", env=env, capture_output=True,
                           text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("['oom', 'oom', 'oom', 'oom', 'error', 'error'] numpy._core._exceptions", r.stdout)


if __name__ == "__main__":
    unittest.main()
