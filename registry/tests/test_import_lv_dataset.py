"""Importer + validate on a dataset built from a label version (data_utils/dataset_from_lv.py).

The importer keeps such a dataset's own record (registered, or its folder's provenance.json) and only derives
frozen -> trained; it never re-derives it as an export dataset, never calls membership() for it and leaves its
lists/<did>.tsv alone. Control: an ordinary export dataset folder next to it is still derived as before.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
REPO = REG.parent
sys.path.insert(0, str(REG.parent))                   # the repo root: `from registry import ...`
sys.path.insert(0, str(REPO / "data_utils" / "tests"))
from registry import importers  # noqa: E402
from registry import store as registry  # noqa: E402
from registry import validate  # noqa: E402
from test_dataset_from_lv import D, make_export, make_pull, write_lv  # noqa: E402

DS, DS2, PLAIN = "lvds_fixture", "lvds_unregistered", "plain_export_ds"
RUN = "2099-01-02_120000_y8n_320_2b_1ep_Run_1"


class LvDatasetImport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="dfl_imp_")
        t = Path(self.tmp.name)
        self.lap, self.cache, self.root = t / "CryoAI-laptop", t / "CryoAI", t / "registry"
        self.lap.mkdir()
        export = make_export(self.cache)
        lv = write_lv(t / "lvs", "lv_20990101_0000", make_pull(), review_scope="reviewed_only")
        self.outs = self.cache / "training outputs"
        for did in (DS, DS2):
            D.build(str(lv), str(export), did, self.outs, "ev_nonev", export_only="drop", log=lambda *a: None)
        D.register_existing(self.outs / DS, self.root)
        (self.outs / PLAIN).mkdir()
        (self.outs / PLAIN / "dataset.yaml").write_text("nc: 1\n", encoding="utf-8")
        self.lst = self.root / "lists" / f"{DS}.tsv"
        self.lst_bytes = self.lst.read_bytes()

    def tearDown(self):
        self.tmp.cleanup()

    def run_import(self):
        return importers.run(self.root, laptop=self.lap, cache=self.cache, processing=None)

    def dataset_lines(self, did):
        return [json.loads(x) for x in (self.root / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
                if json.loads(x)["id"] == did]

    def test_registered_lv_dataset_is_kept(self):
        before = self.dataset_lines(DS)
        self.run_import()
        self.assertEqual(self.dataset_lines(DS), before)                      # 0 appends for it
        self.assertEqual(self.lst.read_bytes(), self.lst_bytes)
        cur = registry.current("dataset", self.root)
        self.assertEqual(cur[DS]["source"]["kind"], "label_version")
        # control: the export dataset folder is still derived the old way
        self.assertEqual(cur[PLAIN]["source"], {"kind": "roboflow_export", "id": "unknown"})
        # a built but unregistered lv dataset gets its own provenance record and list, not an export one
        self.assertEqual(cur[DS2]["source"]["kind"], "label_version")
        self.assertEqual((self.root / "lists" / f"{DS2}.tsv").read_bytes(),
                         (self.outs / DS2 / D.IMAGE_LIST).read_bytes())
        res = self.run_import()
        self.assertEqual(res["appended"].get("dataset", 0), 0)                 # rerun: nothing

    def test_a_run_only_changes_the_status(self):
        self.run_import()
        before = self.dataset_lines(DS)
        self.assertEqual(len(before), 1)                                      # only the registered line
        self.assertEqual(before[-1]["source"]["kind"], "label_version")
        (self.outs / DS / "runs" / RUN).mkdir(parents=True)
        res = self.run_import()
        after = self.dataset_lines(DS)
        self.assertEqual(len(after), len(before) + 1)
        self.assertEqual(res["appended"].get("dataset", 0), 1)
        old, new = before[-1], after[-1]
        diff = {k for k in set(old) | set(new) if old.get(k) != new.get(k)} - {"logged", "recorded_by"}
        self.assertEqual(diff, {"status"})
        self.assertEqual((old["status"], new["status"]), ("frozen", "trained"))
        self.assertEqual(self.lst.read_bytes(), self.lst_bytes)
        self.assertEqual(self.run_import()["appended"].get("dataset", 0), 0)

    def test_validate_has_no_label_version_problem_for_it(self):
        scan = validate.Scan(self.lap, self.cache, None).run()
        flagged = scan.t["dataset"].examples.get("no label_version (trained from a Roboflow export)", [])
        self.assertNotIn(DS, flagged)
        self.assertNotIn(DS2, flagged)
        self.assertIn(PLAIN, flagged)                                         # control


if __name__ == "__main__":
    unittest.main()
