"""importers.py reads a run folder's own run.json (run_experiments.py) and derives older runs as before.

Temp CryoAI only. Control: a run folder without run.json in the same scan still gets the derived record
(pre-bundle, status from the model link), so the run.json branch can't be hiding a broken scan.
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REG.parent))                   # the repo root: `from registry import ...`
from registry import importers  # noqa: E402
from registry import store as registry  # noqa: E402

OLD = "2026-10-01_120000_y8n_1440_24b_300ep_Run_3"
NEW = "2026-11-02_120000_y8m_640_8b_50ep_Run_4"
BAD = "2026-11-02_130000_y8n_320_4b_2ep_Run_5"
BUNDLE = "ds_tiny__job_20261102-120000"


class RunJsonImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rj_"))
        runs = self.tmp / "cai" / "training outputs" / "ds_tiny" / "runs"
        (self.tmp / "cai" / "training images").mkdir(parents=True)
        for r in (OLD, NEW, BAD):
            (runs / r).mkdir(parents=True)
        (runs / NEW / "run.json").write_text(json.dumps({
            "schema": "run/1", "id": NEW, "dataset": "ds_tiny", "bundle_id": BUNDLE, "job_bundle": BUNDLE,
            "cryoEV_commit": "abc1234", "machine": "DESK", "ultralytics": "8.4.94",
            "config": {"arch": "yolov8m-seg", "imgsz": 640, "batch": 6, "epochs": 50, "workers": 4},
            "status": "oom", "error": "ArrayMemoryError: Unable to allocate", "started": "2026-11-02T12:00:00.123456",
            "ended": "2026-11-02T13:00:00.5", "metrics": {"mask_map50_95": 0.0}}), encoding="utf-8")
        (runs / BAD / "run.json").write_text("{not json", encoding="utf-8")
        (self.tmp / "lap").mkdir()
        (self.tmp / "reg").mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_json_used_and_old_runs_derived(self):
        imp = importers.Importer(self.tmp / "reg", laptop=self.tmp / "lap", cache=self.tmp / "cai",
                                 processing=None).scan()
        runs = {r["id"]: r for e, r in imp.out if e == "run"}
        new = runs[NEW]
        self.assertEqual(new["status"], "failed")
        self.assertIn("run.json status oom", new["note"])
        self.assertEqual(new["job_bundle"], BUNDLE)
        self.assertNotIn("job_bundle_gap", new)
        self.assertEqual(new["config"], {"arch": "yolov8m-seg", "imgsz": 640, "batch": 6, "epochs": 50})
        self.assertEqual((new["started"], new["finished"]), ("2026-11-02T12:00:00", "2026-11-02T13:00:00"))
        self.assertEqual(new["cryoEV_commit"], "abc1234")
        # control: no run.json -> derived exactly as before (name config, pre-bundle, not finished w/o a model)
        old = runs[OLD]
        self.assertEqual((old["status"], old.get("job_bundle"), old["job_bundle_gap"]), ("running", None, "pre-bundle run"))
        self.assertEqual(old["config"], {"arch": "yolov8n-seg", "imgsz": 1440, "batch": 24, "epochs": 300})
        # unreadable run.json: reported, then derived from the folder name
        self.assertEqual(imp.refused["run: run.json unreadable"], 1)
        self.assertEqual(runs[BAD]["job_bundle_gap"], "pre-bundle run")
        # every record passes the schema (what registry.append would check)
        for r in runs.values():
            registry._check({**r, "logged": "2026-11-02T00:00:00"}, self.tmp / "reg")


if __name__ == "__main__":
    unittest.main()
