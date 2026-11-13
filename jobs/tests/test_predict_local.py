"""jobs/predict_local.py with a fake predictor (no GPU): create -> run here -> done, in temp roots.

Known answers: the fake predictor copies the golden predictions, so the finished set's predictions are byte-identical
to tests/fixtures/cj_golden and summary.csv's counts equal the golden's (compared as numbers); status.json's n_done
is seen between 0 and n_total while it runs, with the predictor's pid and a fresh `updated` every poll. Controls:
every failure (a predictor error, an injected status-write error, a rename that never succeeds) is followed by the
same set running to done; a busy GPU comes with this process and its parent not counting as busy, and --allow-cpu.
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
from jobtest import C  # noqa: E402
from test_predict_set import FAKE_PRED, GOLDEN_PRED, MODEL, PredictRoots, assert_contract_layout  # noqa: E402

import predict_local as PL  # noqa: E402
from registry import store as registry  # noqa: E402
from data_utils.oslock import acquire_lock  # noqa: E402

BUSY_GPU = {"total_mb": 4096, "apps": [[4242, "C:\\Python\\python.exe", 900]]}
NO_GPU = {"total_mb": 0, "apps": []}


class PredictLocalTests(PredictRoots, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.argv_file = self.tmp / "argv.jsonl"
        os.environ.update(T.hooks_env(train_cmd=None, gpu=T.LAPTOP_GPU, predict_cmd=FAKE_PRED),
                          FAKE_PRED_FROM=str(GOLDEN_PRED), FAKE_PRED_ARGV=str(self.argv_file))
        self.ps = self.make()

    def run_local(self, *extra, **env):
        with mock.patch.dict(os.environ, env):
            return PL.main(["--ps", self.ps.name, "--root", str(self.root), "--registry", str(self.reg),
                            "--poll-s", "0.1", *extra])

    def statuses(self):
        return [r["status"] for r in registry.history(self.ps.name, self.reg)]

    def argvs(self):
        return [json.loads(ln) for ln in self.argv_file.read_text(encoding="utf-8").splitlines()] \
            if self.argv_file.exists() else []

    def test_done_with_progress(self):
        p = subprocess.Popen([sys.executable, "-m", "jobs.predict_local", "--ps", self.ps.name, "--root",
                              str(self.root), "--registry", str(self.reg), "--poll-s", "0.1"], cwd=T.REPO,
                             env={**os.environ, "FAKE_PRED_DELAY": "1.5"}, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        seen, beats = set(), {}
        while p.poll() is None:
            try:
                st = C.read_json(self.ps / "status.json")
                seen.add((st["state"], st["n_done"], st["pid"] is not None))
                if st["state"] == "running":
                    beats.setdefault(st["n_done"], set()).add(st["updated"])
            except (OSError, ValueError):
                pass
            time.sleep(0.05)
        out = p.stdout.read()
        self.assertEqual(p.returncode, 0, out)
        self.assertTrue({("running", 1, True), ("running", 2, True)} & seen, f"no progress seen: {sorted(seen)}")
        self.assertTrue(any(len(v) >= 2 for v in beats.values()), f"`updated` not refreshed per poll: {beats}")
        assert_contract_layout(self, self.ps)
        self.assertIsNone(C.read_json(self.ps / "status.json")["pid"])
        for f in GOLDEN_PRED.iterdir():
            self.assertEqual((self.ps / "p" / MODEL / "predictions" / f.name).read_bytes(), f.read_bytes())

        def counts(path):
            lines = path.read_text(encoding="utf-8").splitlines()
            return {Path(r.split(",")[0]).stem: int(r.split(",")[2]) for r in lines[1:]}
        summary = self.ps / "p" / MODEL / "summary.csv"
        self.assertEqual(summary.read_text(encoding="utf-8").splitlines()[0], "image,model_id,n_detections,mean_conf")
        self.assertEqual(counts(summary), counts(GOLDEN_PRED.parent / "summary.csv"))   # joined on the stem
        self.assertEqual(self.statuses(), ["queued", "running", "done"])
        argv = self.argvs()[0]
        for flag, val in (("--device", "cuda"), ("--conf", "0.25"), ("--max-det", "1000"), ("--models", MODEL)):
            self.assertEqual(argv[argv.index(flag) + 1], val)
        self.assertIn("--no-overlays", argv)
        self.assertNotIn("--imgsz", argv)                  # request imgsz null: the model's registered size
        self.assertIn("done", (self.ps / "predict.log").read_text(encoding="utf-8"))
        self.assertEqual(self.run_local(), PL.EXIT_REFUSED, "a done set is not run again")

    def test_failure_then_rerun(self):
        self.assertEqual(self.run_local(FAKE_PRED="fail_after:1"), PL.EXIT_FAILED)
        st = C.read_json(self.ps / "status.json")
        self.assertEqual((st["state"], st["n_done"], st["n_total"], st["pid"]), ("failed", 1, 3, None))
        self.assertIn("out of memory", st["error"])
        self.assertFalse((self.ps / "p").exists(), "p/ only for a complete set")
        self.assertTrue((self.ps / PL.PRED_TMP).is_dir())
        self.assertEqual(self.statuses(), ["queued", "running", "failed"])
        self.assertEqual(self.run_local(), PL.EXIT_DONE)   # control: the failed set runs again to done
        assert_contract_layout(self, self.ps)
        self.assertEqual(self.statuses()[-1], "done")

    def test_status_write_failure_kills_the_predictor(self):
        real = PL.PS.write_status

        def flaky(ps_dir, **fields):
            if set(fields) == {"n_done"}:              # the per-poll heartbeat
                raise OSError("injected: disk full")
            return real(ps_dir, **fields)
        pred = self.ps / PL.PRED_TMP / MODEL / "predictions"
        t0 = time.time()
        with mock.patch.object(PL.PS, "write_status", flaky), mock.patch.dict(os.environ, FAKE_PRED_DELAY="1.0"):
            out = PL.run(self.ps.name, self.root, self.reg, poll_s=0.1)
        self.assertEqual(out["exit"], PL.EXIT_FAILED)
        self.assertLess(time.time() - t0, 2.5, "returned before the predictor could have finished")
        st = C.read_json(self.ps / "status.json")
        self.assertEqual((st["state"], st["pid"]), ("failed", None))
        self.assertIn("injected: disk full", st["error"])
        n = C.count_predictions(pred)
        time.sleep(2.0)
        self.assertEqual(C.count_predictions(pred), n, "the predictor kept writing: not killed")
        self.assertLess(n, 3)
        self.assertEqual(self.statuses()[-1], "failed")
        self.assertEqual(self.run_local(), PL.EXIT_DONE)   # control: the same set runs to done

    def test_rename_retry(self):
        calls = []

        class Flaky:
            def __init__(self, fails):
                self.fails = fails

            def rename(self, dst):
                calls.append(dst)
                if len(calls) <= self.fails:
                    raise PermissionError("held by the indexer")
        PL.rename_retry(Flaky(2), "x", tries=4, backoff=0.01)
        self.assertEqual(len(calls), 3)
        calls.clear()
        with self.assertRaises(PermissionError):
            PL.rename_retry(Flaky(9), "x", tries=3, backoff=0.01)
        self.assertEqual(len(calls), 3)

        def never(src, dst, **kw):
            raise PermissionError("held by an open window")
        with mock.patch.object(PL, "rename_retry", never):
            self.assertEqual(self.run_local(), PL.EXIT_FAILED)
        st = C.read_json(self.ps / "status.json")
        self.assertEqual(st["state"], "failed")
        self.assertIn("could not be renamed to p/", st["error"])
        self.assertFalse((self.ps / "p").exists())
        self.assertEqual(self.run_local(), PL.EXIT_DONE)   # control
        assert_contract_layout(self, self.ps)

    def test_incomplete_output_fails(self):
        for mode, msg in (("skip_one", "1 of 3 images have no prediction file"), ("no_summary", "no summary.csv")):
            with self.subTest(mode):
                self.assertEqual(self.run_local(FAKE_PRED=mode), PL.EXIT_FAILED)
                self.assertIn(msg, C.read_json(self.ps / "status.json")["error"])
                self.assertFalse((self.ps / "p").exists())

    def test_gpu_busy_stays_queued_missing_fails(self):
        rc = self.run_local(**T.hooks_env(train_cmd=None, gpu=BUSY_GPU, predict_cmd=FAKE_PRED))
        self.assertEqual(rc, PL.EXIT_GPU)
        st = C.read_json(self.ps / "status.json")
        self.assertEqual((st["state"], st["error"]), ("queued", None))
        self.assertIn("gpu busy: pid 4242 python.exe 900 MB", st["waiting"])
        self.assertEqual(self.statuses(), ["queued"])
        mine = {"total_mb": 4096, "apps": [[os.getpid(), "python.exe", 900], [os.getppid(), "python.exe", 500]]}
        with mock.patch.dict(os.environ, T.hooks_env(train_cmd=None, gpu=mine, predict_cmd=FAKE_PRED)):
            self.assertEqual(PL.gpu_check(False)[0], "cuda", "this process and its parent are not 'busy'")
        rc = self.run_local(**T.hooks_env(train_cmd=None, gpu=NO_GPU, predict_cmd=FAKE_PRED))
        self.assertEqual(rc, PL.EXIT_GPU)
        st = C.read_json(self.ps / "status.json")
        self.assertEqual(st["state"], "failed")
        self.assertIn("CUDA unavailable", st["error"])
        self.assertEqual(self.argvs(), [], "the predictor never started")
        rc = self.run_local("--allow-cpu", **T.hooks_env(train_cmd=None, gpu=BUSY_GPU, predict_cmd=FAKE_PRED))
        self.assertEqual(rc, PL.EXIT_DONE)                 # control: --allow-cpu runs it, on the CPU
        argv = self.argvs()[0]
        self.assertEqual(argv[argv.index("--device") + 1], "cpu")
        self.assertIsNone(C.read_json(self.ps / "status.json")["waiting"])
        assert_contract_layout(self, self.ps)

    def test_labels_round_trip_into_results(self):
        """images.csv keeps raw values: odd but legal sample labels reach results.json exactly (B4a's engine)."""
        from results.compute import compute
        labels = {"S1": 'EV prep A, 50% EDTA "wash"', "S2": "control (n=3) µm"}
        man = self.tmp / "labels.csv"
        man.write_text(self.manifest.read_text(encoding="utf-8").replace(",S1,", ',"EV prep A, 50% EDTA ""wash""",')
                       .replace(",S2,", ",control (n=3) µm,"), encoding="utf-8")
        ps = self.make(manifest=man)
        self.assertEqual(PL.main(["--ps", ps.name, "--root", str(self.root), "--registry", str(self.reg),
                                  "--poll-s", "0.1"]), PL.EXIT_DONE)
        res = compute(ps)
        self.assertIsNone(res["refused"])
        self.assertEqual(set(res["samples"]), set(labels.values()))
        self.assertEqual(res["samples"][labels["S1"]]["n_images"], 2)

    def visible_windows(self, pids):
        """Visible top-level windows owned by these pids (a console window counts as its client process's)."""
        import ctypes
        from ctypes import wintypes
        u32, found = ctypes.windll.user32, []
        proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def cb(hwnd, _):
            pid = wintypes.DWORD()
            u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value in pids and u32.IsWindowVisible(hwnd):
                found.append((pid.value, hwnd))
            return True
        u32.EnumWindows(proc(cb), 0)
        return found

    def test_start_detached(self):
        os.environ["FAKE_PRED_DELAY"] = "1.0"
        pid = PL.start_detached(self.ps, root=self.root, registry_root=self.reg)
        self.assertIsInstance(pid, int)
        deadline, windows, checked = time.time() + 60, [], 0
        while time.time() < deadline and C.read_json(self.ps / "status.json")["state"] not in ("done", "failed"):
            if os.name == "nt":                            # no window for predict_local or its predictor
                pids = {pid, C.read_json(self.ps / "status.json").get("pid")} - {None}
                windows += self.visible_windows(pids)
                checked += 1
            time.sleep(0.2)
        self.assertEqual(windows, [], "a detached run showed a window")
        if os.name == "nt":
            self.assertGreater(checked, 5)
        log = self.root / PL.LOG_DIR / f"{self.ps.name}.out"
        self.assertEqual(C.read_json(self.ps / "status.json")["state"], "done",
                         log.read_text(encoding="utf-8", errors="replace") if log.exists() else "no log")
        assert_contract_layout(self, self.ps)

    def test_lock_and_desktop_refusals(self):
        lock = acquire_lock(self.root / PL.LOCK_FILE)
        try:
            self.assertEqual(self.run_local(), PL.EXIT_LOCKED)
        finally:
            lock.close()
        self.assertEqual(C.read_json(self.ps / "status.json")["state"], "queued", "a locked-out run changes nothing")
        self.assertEqual(self.argvs(), [])
        import predict_set as PS
        PS.write_status(self.ps, where="desktop", job="job_20261105_x")
        self.assertEqual(self.run_local(), PL.EXIT_REFUSED)
        PS.write_status(self.ps, where="laptop", job=None)
        self.assertEqual(self.run_local(), PL.EXIT_DONE)   # control

    def test_imports_leave_sys_path_alone(self):
        code = ("import sys; before = list(sys.path); import jobs.predict_set, jobs.predict_local; "
                "assert sys.path == before, [p for p in sys.path if p not in before]; "
                "assert 'common' not in sys.modules and 'registry' not in sys.modules and 'run_job' not in sys.modules"
                "; print('clean')")
        r = subprocess.run([sys.executable, "-c", code], cwd=T.REPO, capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "clean"), r.stderr)


if __name__ == "__main__":
    unittest.main()
