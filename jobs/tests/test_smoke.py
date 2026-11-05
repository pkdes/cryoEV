"""Optional real smoke test (GPU, ~2 min): make_job_bundle -> registry job new -> sync push -> run_job (real
run_experiments, y8n-seg imgsz 320, 2 epochs) -> sync pull, with temp laptop/CryoAI roots under a SHORT folder.

    set CRYOEV_JOBS_SMOKE_DATA=<tiny YOLO dataset: dataset.yaml, images/, labels/>
    set CRYOEV_JOBS_SMOKE_ROOT=C:\\cj_test         (short; default C:\\cj_test; temp folders made inside it)
    set CRYOEV_JOBS_SMOKE_WEIGHTS=<yolov8n-seg.pt>  (optional; avoids a download, and tests the packed --weights)
    python -m unittest discover -s jobs/tests -p test_smoke.py -v

The temp laptop root gets a copy of the real registry/ only (read, never written). Skipped when DATA is unset.
"""
import os
import shutil
import subprocess
import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobtest as T  # noqa: E402
from jobtest import C  # noqa: E402

DATA = os.environ.get("CRYOEV_JOBS_SMOKE_DATA")


@unittest.skipUnless(DATA, "set CRYOEV_JOBS_SMOKE_DATA to run the real smoke test")
class SmokeTest(unittest.TestCase):
    def test_push_run_pull(self):
        from data_utils import config
        real_reg = config.registry_root()                       # read only: copied into the temp root
        base = Path(os.environ.get("CRYOEV_JOBS_SMOKE_ROOT", "C:/cj_test"))
        root = base / datetime.now().strftime("%H%M%S")
        lap, cai, share, out = root / "lap", root / "cai", root / "s", root / "b"
        shutil.copytree(real_reg, lap / "registry", ignore=shutil.ignore_patterns("*.html"))
        for d in (cai / "training outputs", cai / "training images", share / C.Q):
            d.mkdir(parents=True)
        env = {k: v for k, v in os.environ.items() if k not in ("CRYOEV_REGISTRY", "CRYOEV_NTFY_TOPIC")}
        env.update(CRYOEV_LAPTOP_ROOT=str(lap), CRYOEV_CRYOAI_ROOT=str(cai),
                   CRYOEV_PROCESSING_ROOT=str(root / "no_processing"), PYTHONIOENCODING="utf-8")

        def py(*args, ok=(0,)):
            r = subprocess.run([sys.executable, *map(str, args)], cwd=T.REPO, env=env, capture_output=True,
                               text=True, timeout=1800)
            print(f"$ {' '.join(map(str, args[:3]))} -> {r.returncode}\n{(r.stdout + r.stderr)[-1500:]}")
            self.assertIn(r.returncode, ok)
            return r.stdout

        exp = ["--single-run", "--model-size", "n", "--imgsz", "320", "--epochs", "2", "--batch", "4",
               "--workers", "0", "--patience", "2", "--eval-splits", "val"]
        if os.environ.get("CRYOEV_JOBS_SMOKE_WEIGHTS"):
            exp += ["--weights", os.environ["CRYOEV_JOBS_SMOKE_WEIGHTS"]]
        py("training/make_job_bundle.py", "--dataset-dir", DATA, "--dataset-name", "smoke_tiny", "--out", out,
           "--", *exp)
        zp = next(out.glob("*__job_*.zip"))                  # thin job zip; the dataset pack ds_*.zip is pushed with it
        self.assertTrue(Path(str(zp) + ".md5").is_file())
        lst = root / "imgs.txt"
        lst.write_text("".join(p.name + "\n" for p in Path(DATA, "images").rglob("*") if p.is_file()), encoding="utf-8")
        py("registry/registry.py", "job", "new", "--type", "train", "--where", "laptop", "--name", "smoke",
           "--images", lst, "--who", "pujan")
        job = f"job_{datetime.now():%Y%m%d}_smoke"
        py("jobs/sync.py", "push", job, "--bundle", zp, "--share", share)
        py("jobs/sync.py", "status", "--share", share)
        py("jobs/run_job.py", "--queue-dir", share, "--max-hours", "0.5")
        done = C.read_json(share / C.D / job / C.DONE)
        self.assertEqual(done["status"], "done", done)
        rj = C.read_json(share / C.D / job / "run.json")
        self.assertEqual(rj["status"], "finished")
        self.assertEqual(rj["bundle_id"], zp.stem)
        py("jobs/sync.py", "pull", "--share", share)
        py("jobs/sync.py", "pull", "--share", share)
        hist = py("registry/registry.py", "show", job)
        self.assertIn("statuses: ready -> done", hist)
        run = py("registry/registry.py", "show", done["run"])
        self.assertIn('"status": "finished"', run)
        self.assertIn(zp.stem, run)
        print(f"smoke root (delete when done): {root}")


if __name__ == "__main__":
    unittest.main()
