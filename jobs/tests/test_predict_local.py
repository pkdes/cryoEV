"""jobs/predict_local.py with a fake predictor (no GPU): create -> run here -> done, in temp roots.

Known answers: the fake predictor copies the golden predictions, so the finished set's predictions are byte-identical
to tests/fixtures/cj_golden; status.json's n_done is seen between 0 and n_total while it runs. Controls: a failed run
is re-run to done; a GPU refusal comes with the same set running on the CPU under --allow-cpu.
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
import registry  # noqa: E402
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
                             env={**os.environ, "FAKE_PRED_DELAY": "0.6"}, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        seen = set()
        while p.poll() is None:
            try:
                st = C.read_json(self.ps / "status.json")
                seen.add((st["state"], st["n_done"]))
            except (OSError, ValueError):
                pass
            time.sleep(0.05)
        out = p.stdout.read()
        self.assertEqual(p.returncode, 0, out)
        self.assertTrue({("running", 1), ("running", 2)} & seen, f"no progress seen: {sorted(seen)}")
        assert_contract_layout(self, self.ps)
        for f in GOLDEN_PRED.iterdir():
            self.assertEqual((self.ps / "p" / MODEL / "predictions" / f.name).read_bytes(), f.read_bytes())
        summary = (self.ps / "p" / MODEL / "summary.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(summary[0], "image,model_id,n_detections,mean_conf")
        self.assertEqual([ln.split(",")[2] for ln in summary[1:]], ["5", "5", "3"])     # golden: 5, 5, 3 lines
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
        self.assertEqual((st["state"], st["n_done"], st["n_total"]), ("failed", 1, 3))
        self.assertIn("out of memory", st["error"])
        self.assertFalse((self.ps / "p").exists(), "p/ only for a complete set")
        self.assertTrue((self.ps / PL.PRED_TMP).is_dir())
        self.assertEqual(self.statuses(), ["queued", "running", "failed"])
        self.assertEqual(self.run_local(), PL.EXIT_DONE)   # control: the failed set runs again to done
        assert_contract_layout(self, self.ps)
        self.assertEqual(self.statuses()[-1], "done")

    def test_incomplete_output_fails(self):
        for mode, msg in (("skip_one", "1 of 3 images have no prediction file"), ("no_summary", "no summary.csv")):
            with self.subTest(mode):
                self.assertEqual(self.run_local(FAKE_PRED=mode), PL.EXIT_FAILED)
                self.assertIn(msg, C.read_json(self.ps / "status.json")["error"])
                self.assertFalse((self.ps / "p").exists())

    def test_gpu_busy_or_missing(self):
        for gpu, msg in ((BUSY_GPU, "GPU busy: pid 4242 python.exe 900 MB"), (NO_GPU, "CUDA unavailable")):
            with self.subTest(msg):
                rc = self.run_local(**T.hooks_env(train_cmd=None, gpu=gpu, predict_cmd=FAKE_PRED))
                self.assertEqual(rc, PL.EXIT_GPU)
                st = C.read_json(self.ps / "status.json")
                self.assertEqual(st["state"], "failed")
                self.assertIn(msg, st["error"])
                self.assertEqual(self.argvs(), [], "the predictor never started")
        rc = self.run_local("--allow-cpu", **T.hooks_env(train_cmd=None, gpu=BUSY_GPU, predict_cmd=FAKE_PRED))
        self.assertEqual(rc, PL.EXIT_DONE)                 # control: --allow-cpu runs it, on the CPU
        argv = self.argvs()[0]
        self.assertEqual(argv[argv.index("--device") + 1], "cpu")
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


if __name__ == "__main__":
    unittest.main()
