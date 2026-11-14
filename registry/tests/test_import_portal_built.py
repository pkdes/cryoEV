"""Round 13: the importer skips packages the portal built (built_by "portal") entirely: no review_package line and no
revise job, whether the marker is in review_package.json (a finished build) or only in data.js (mid-build). A CLI-built
lab_ package on the same registry still gets its line and its revise job (control: behaviour as before). The schema
takes the portal's allowlisted line (kind compare, base_kind label_version, no path), and history_page links it to
its label version.

  python -m unittest registry.tests.test_import_portal_built -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REG.parent))                   # the repo root: `from registry import ...`
from registry import history_page  # noqa: E402
from registry import importers  # noqa: E402
from registry import store as registry  # noqa: E402
from registry.schema_check import check_record  # noqa: E402

LV = "lv_20991231_0101"
CLI, DONE, MID = "lab_compare_cli_20991231", "lab_revise_lv_20991231_0101_20991231_0102_ab12", \
    "lab_audit_lv_20991231_0101_20991231_0103_cd34"


def w(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def datajs(pid, source, **extra):
    return "window.REVIEW_DATA = " + json.dumps({"id": pid, "kind": "compare", "source": source,
                                                 "run": "v5_evnonev_y8m_20261029", "created": "2099-12-31T10:00:00",
                                                 "images": [{"file": "0001.png"}], **extra}) + ";"


class PortalBuiltSkipped(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.lap, self.cache, self.root = t / "CryoAI-laptop", t / "CryoAI", t / "registry"
        (self.cache / "training images").mkdir(parents=True)
        (self.cache / "training outputs").mkdir(parents=True)
        labels = str(self.lap / "labels" / LV / "labels.coco.json")
        rev = self.lap / "review"
        w(rev / CLI / "data.js", datajs(CLI, labels))
        w(rev / CLI / "review_package.json", json.dumps({"id": CLI, "kind": "compare", "owner": "pujan"}))
        w(rev / DONE / "data.js", datajs(DONE, labels, built_by="portal"))
        w(rev / DONE / "review_package.json", json.dumps({"id": DONE, "kind": "compare", "purpose": "revise",
                                                          "built_by": "portal", "owner": "pujan"}))
        w(rev / MID / "data.js", datajs(MID, labels, built_by="portal"))      # mid-build: no review_package.json

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_run_appends_nothing_for_portal_built(self):
        res = importers.run(self.root, dry_run=True, laptop=self.lap, cache=self.cache, processing=None)
        self.assertEqual(res["excluded"]["review_package (portal-built, own line)"], 2)
        self.assertEqual((res["appended"]["review_package"], res["appended"]["job"]), (1, 1))   # the CLI one only
        self.assertEqual(registry.current(root=self.root), {})                 # a dry run writes nothing

    def test_real_run_on_temp_registry(self):
        importers.run(self.root, laptop=self.lap, cache=self.cache, processing=None)
        cur = registry.current(root=self.root)
        self.assertIn(CLI, cur)                                                # control: as before
        self.assertNotIn(DONE, cur)
        self.assertNotIn(MID, cur)
        jobs = [r.get("package") for r in cur.values() if r["schema"] == "job/1"]
        self.assertEqual(jobs, [CLI])
        res = importers.run(self.root, laptop=self.lap, cache=self.cache, processing=None)   # rerun: nothing new
        self.assertEqual(sum(res["appended"].values()), 0)

    def test_portal_built_helper(self):
        self.assertTrue(importers.portal_built(self.lap / "review" / DONE))
        self.assertTrue(importers.portal_built(self.lap / "review" / MID, {"built_by": "portal"}))
        self.assertFalse(importers.portal_built(self.lap / "review" / MID))    # no data.js given, no json yet
        self.assertFalse(importers.portal_built(self.lap / "review" / CLI, {}))
        w(self.lap / "review" / "bad" / "review_package.json", "{not json")
        self.assertFalse(importers.portal_built(self.lap / "review" / "bad"))

    def test_schema_takes_the_portal_line(self):
        line = {"schema": "review_package/1", "id": DONE, "logged": "2099-12-31T10:00:00", "kind": "compare",
                "purpose": "revise", "base": LV, "base_kind": "label_version", "n_images": 3,
                "built": "2099-12-31T10:00:00", "status": "built", "owner": "pujan", "built_by": "portal",
                "model": "v5_evnonev_y8m_20261029", "flag_conf": 0.55, "tool": "annotation/compare_package.py",
                "recorded_by": "portal/review_build.py"}
        self.assertEqual(check_record(line, registry.schemas()), [])
        self.assertEqual(check_record(dict(line, purpose="audit", b_label_version=LV, model=None), registry.schemas()), [])
        self.assertTrue(check_record(dict(line, purpose="other"), registry.schemas()))    # controls: refused
        self.assertTrue(check_record(dict(line, built_by="cli"), registry.schemas()))
        self.assertTrue(check_record(dict(line, base_kind="lv"), registry.schemas()))
        self.assertEqual(history_page.BASE_KIND["label_version"], "label_version")


if __name__ == "__main__":
    unittest.main()
