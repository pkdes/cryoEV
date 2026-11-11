"""Importer: a label version applied on top of another label version (base = lv_*, e.g. a review of a review).

Known answer: lv B (base lv A, supersedes lv A) is registered (schema accepts an lv base), its base input is linked
as a label_version (not a roboflow_export), and lv A becomes superseded. Control: lv A (base = a pull) is still
linked to its pull.
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
REPO = REG.parent
sys.path.insert(0, str(REG))
sys.path.insert(0, str(REPO / "data_utils" / "tests"))
import importers  # noqa: E402
import registry  # noqa: E402
from test_dataset_from_lv import make_pull, write_lv  # noqa: E402

A, B = "lv_20990101_0000", "lv_20990102_0000"


class LvChainImport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lvchain_")
        t = Path(self.tmp.name)
        self.lap, self.cache, self.root = t / "CryoAI-laptop", t / "CryoAI", t / "registry"
        labels = self.lap / "labels"
        labels.mkdir(parents=True)
        (self.cache / "training images").mkdir(parents=True)
        (self.cache / "training outputs").mkdir()
        a = Path(write_lv(t / "lvs", A, make_pull(), review_scope="all_edits"))
        a_dir = a if a.is_dir() else a.parent
        shutil.copytree(a_dir, labels / A)
        shutil.copytree(a_dir, labels / B)
        full = {"pkg": "pkg_fixture", "edits_dir": "edits", "n_images_applied": 1, "counts_by_op": {"add": 1},
                "class_groups_sha256": "0" * 64, "people_sha256": "0" * 64, "tool": "apply_review_edits.py test",
                "created": "2099-01-01T00:00:00"}
        self.pull = json.loads((labels / A / "version.json").read_text(encoding="utf-8"))["base"]
        for lv, extra in ((A, {}), (B, {"base": A, "supersedes": A})):
            v = json.loads((labels / lv / "version.json").read_text(encoding="utf-8"))
            v.update(full, id=lv, **extra)
            (labels / lv / "version.json").write_text(json.dumps(v), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_lv_on_lv(self):
        res = importers.run(self.root, laptop=self.lap, cache=self.cache, processing=None)
        refused = [k for k in res["refused"] if k.startswith("label_version")]
        self.assertEqual(refused, [])
        cur = registry.current("label_version", self.root)
        self.assertIn(B, cur)
        base_b = [i for i in cur[B]["inputs"] if i["role"] == "base"]
        self.assertEqual(base_b, [{"id": A, "role": "base", "entity": "label_version"}])
        self.assertEqual(cur[A]["status"], "superseded")
        base_a = [i for i in cur[A]["inputs"] if i["role"] == "base"]
        self.assertEqual(base_a[0]["entity"], "pull" if self.pull.startswith("roboflow_pull_") else "roboflow_export")


if __name__ == "__main__":
    unittest.main()
