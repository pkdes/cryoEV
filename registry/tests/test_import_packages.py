"""Prediction review packages (contract 3): data.js source `prediction_set:<ps_id>`, base_kind prediction_set.

validate checks `<CryoAI-laptop>/predict/<ps_id>/` instead of an export; the importer leaves their registry line
to their builder (it would otherwise re-derive the record without `owner` / `prediction_set`) and makes no
revise job for them.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REG))
import importers  # noqa: E402
import registry  # noqa: E402
import validate  # noqa: E402

PS_OK, PS_MISSING = "ps_20991231_abc123", "ps_20991231_def456"
PULL = "roboflow_pull_20990101_0000"


def w(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def datajs(pid, source):
    return "window.REVIEW_DATA = " + json.dumps({"id": pid, "source": source, "run": "v5_evnonev_y8m_20261029",
                                                 "created": "2099-12-31T10:00:00", "images": [{"file": "0001.png"}]}) + ";"


class PredictionPackages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.lap, cls.cache, cls.root = t / "CryoAI-laptop", t / "CryoAI", t / "registry"
        (cls.cache / "training images").mkdir(parents=True)
        (cls.cache / "training outputs").mkdir(parents=True)
        w(cls.lap / "predict" / PS_OK / "request.json", json.dumps({"ps": PS_OK}))
        for ps in (PS_OK, PS_MISSING):
            w(cls.lap / "review" / f"pred_{ps}" / "data.js", datajs(f"pred_{ps}", f"prediction_set:{ps}"))
        w(cls.lap / "review" / "pkgX" / "data.js", datajs("pkgX", f"C:\\x\\{PULL}\\current.coco.json"))   # control
        cls.scan = validate.Scan(cls.lap, cls.cache, None).run()
        cls.t = cls.scan.t["review_package"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def problems(self, pid):
        return {k for k, ex in self.t.examples.items() if pid in ex}

    def test_prediction_set_base_checked_not_export(self):
        self.assertEqual(self.problems(f"pred_{PS_OK}"), set())
        self.assertEqual(self.problems(f"pred_{PS_MISSING}"), {"base prediction set not found"})
        self.assertNotIn("base export not found locally", self.t.missing)
        self.assertEqual(self.problems("pkgX"), {"base pull not found"})          # control: pull path unchanged
        e = self.scan.packages[f"pred_{PS_OK}"][0]
        self.assertEqual((e["base"], e["base_kind"]), (PS_OK, "prediction_set"))
        self.assertEqual(self.scan.packages["pkgX"][0]["base_kind"], "pull")

    def test_importer_leaves_prediction_packages_to_their_builder(self):
        res = importers.run(self.root, laptop=self.lap, cache=self.cache, processing=None)
        cur = registry.current(root=self.root)
        self.assertEqual(res["excluded"]["review_package (prediction, own line)"], 2)
        self.assertNotIn(f"pred_{PS_OK}", cur)
        self.assertIn("pkgX", cur)
        jobs = [r.get("package") for r in cur.values() if r["schema"] == "job/1"]
        self.assertEqual(jobs, ["pkgX"])


if __name__ == "__main__":
    unittest.main()
