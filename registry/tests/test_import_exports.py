"""Export membership: which images went into each roboflow_export (image_list, like datasets).

- Unit: COCO members -> list (file-name key, else the image whose rf_name is `extra.name`); excluded-session images
  counted, not listed; images with no record counted. tracking/v2 `in: <export>` only when no row is `unread`.
- Real data (temp copy; skipped when not found), known answers:
  * v5 S1: the export list equals the dataset's list (roboflow_20261028_cryoai_v5_S1_evnonev.tsv) minus exactly the
    session-20261012 rows (excluded session: counted in not_listed, never listed), split by split.
  * control for the tracking method: the `in:` columns rebuild the v5 and v5 S1 COCO lists exactly, so the v4 list
    (no local COCO) comes from a method shown right where the answer is known; v4 gives 100 / 20 / 50 (models.yaml).
  * a second import adds 0 lines; the real registry is untouched.
"""
import csv
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REG))
import ids  # noqa: E402
import importers  # noqa: E402
import registry  # noqa: E402

V5, V5S1, V4 = "roboflow - 20261028 cryoai v5", "roboflow - 20261028 cryoai v5 S1", "roboflow - 20260924 cryoai v4"
V5DS = "roboflow_20261028_cryoai_v5_S1_evnonev"
CACHE = importers.find_sibling("CryoAI")
LAPTOP = importers.find_sibling("CryoAI-laptop")
REAL_REG = LAPTOP / "registry" if LAPTOP else None


class _Scan:
    def __init__(self, excluded):
        self.excluded = excluded

    def is_excluded_stem(self, stem):
        return ids.image_key(stem) in self.excluded


def rows(path):
    return {tuple(x.split("\t")) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()}


class ExportMembershipUnit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.imp = importers.Importer(Path(self.tmp.name) / "registry", laptop=self.tmp.name, cache=self.tmp.name,
                                      processing=None)
        self.imp.s = _Scan({"lipo_x_png"})
        self.imp._img_ids = {"a_1_png", "renamed_png"}
        self.imp._by_rf_name = {"b 1.png": "renamed_png"}
        self.imp.tracking = {}

    def tearDown(self):
        self.tmp.cleanup()

    def test_coco_members(self):
        members = [({"file_name": "a_1_png.rf.0a.jpg"}, "train", True),
                   ({"file_name": "b-1_png.rf.0b.jpg", "extra": {"name": "b 1.png"}}, "valid", True),   # rf_name join
                   ({"file_name": "nope_png.rf.0c.jpg"}, "test", True),
                   ({"file_name": "lipo_x_png.rf.0d.jpg"}, "train", False)]
        out = self.imp.export_membership("e", "lists/export_e.tsv", members)
        self.assertEqual(rows_of(self.imp.lists["lists/export_e.tsv"]),
                         {("a_1_png", "train"), ("renamed_png", "valid"), ("nope_png", "test")})
        self.assertEqual(out["image_list"]["n"], 3)
        self.assertEqual(out["image_list"]["method"], "export COCO split folders")
        self.assertEqual(out["counts_listed"], {"train": 1, "valid": 1})
        self.assertEqual(out["not_listed"], {"excluded_session": 1, "not_in_registry": 1})
        self.assertIsNone(out["image_list_gap"])

    def test_tracking_column(self):
        col = "in: old exp"
        self.imp.tracking = {"a_1_png": {col: "train (3)"}, "renamed_png": {col: ""}, "lipo_x_png": {col: "test (1)"}}
        out = self.imp.export_membership("old exp", "lists/export_old_exp.tsv", None)
        self.assertEqual(rows_of(self.imp.lists["lists/export_old_exp.tsv"]), {("a_1_png", "train")})
        self.assertTrue(out["image_list"]["method"].startswith("tracking/v2"))
        self.assertEqual(out["not_listed"], {"excluded_session": 1})
        self.imp.tracking["renamed_png"][col] = "unread"                  # one unread row: unknown, not a list
        out = self.imp.export_membership("old exp", "lists/x.tsv", None)
        self.assertIsNone(out["image_list"])
        self.assertIn("tracking/v2", out["image_list_gap"])
        self.assertNotIn("lists/x.tsv", self.imp.lists)
        self.assertIsNone(self.imp.export_membership("never tracked", "lists/y.tsv", None)["image_list"])

    def test_list_names(self):
        names = importers.export_list_names([V5, V5 + " (rf download)", "roboflow20260604", "roboflow-20260604"])
        self.assertEqual(names[V5], "lists/export_roboflow_20261028_cryoai_v5.tsv")
        self.assertEqual(names[V5 + " (rf download)"], "lists/export_roboflow_20261028_cryoai_v5_rf_download.tsv")
        self.assertEqual(len(set(names.values())), 4)                      # same slug -> sha suffix, no collision


def rows_of(text):
    return {tuple(x.split("\t")) for x in text.splitlines() if x.strip()}


@unittest.skipUnless(REAL_REG and REAL_REG.is_dir() and CACHE and (CACHE / "training images" / V5S1).is_dir(),
                     "real registry / CryoAI cache not found")
class RealExportLists(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "registry"
        cls.root.mkdir()
        for f in REAL_REG.glob("*.jsonl"):                               # never append to the real registry
            shutil.copy2(f, cls.root / f.name)
        if (REAL_REG / "lists").is_dir():
            shutil.copytree(REAL_REG / "lists", cls.root / "lists")
        cls.real = {f.name: f.read_bytes() for f in REAL_REG.glob("*.jsonl")}
        cls.first = importers.run(cls.root)
        cls.second = importers.run(cls.root)
        cls.cur = registry.current(root=cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def export_rows(self, eid):
        return rows(self.root / self.cur[eid]["image_list"]["file"])

    def test_s1_known_answer(self):
        ex = self.export_rows(V5S1)
        ds = rows(self.root / self.cur[V5DS]["image_list"]["file"])
        sa = CACHE / "training images" / V5S1 / "split_assignment.csv"
        with open(sa, newline="", encoding="utf-8") as fh:
            batch = {ids.image_key(r["image"]): r["batch"] for r in csv.DictReader(fh)}
        self.assertEqual(ex - ds, set())
        left_out = ds - ex
        self.assertEqual({batch[s][:8] for s, _ in left_out}, {"20261012"})          # only the excluded session
        self.assertEqual(Counter(s for _, s in ex), Counter({"train": 103, "valid": 23, "test": 34}))
        self.assertEqual(self.cur[V5S1]["not_listed"], {"excluded_session": len(left_out)})
        self.assertEqual({k: v["images"] for k, v in self.cur[V5S1]["counts"].items()},
                         dict(Counter(s for _, s in ex)))                           # = the record's own counts

    def test_tracking_method_control_and_v4(self):
        with open(LAPTOP / "tracking" / "v2" / "prelipo_tracking.csv", newline="", encoding="utf-8-sig") as fh:
            tr = list(csv.DictReader(fh))
        for eid in (V5, V5S1):                                              # known: the COCO lists
            col = f"in: {eid}"
            got = {(ids.image_key(r["stem"]), r[col].split(" ")[0]) for r in tr if r[col]}
            self.assertEqual(got, self.export_rows(eid), eid)
        v4 = self.cur[V4]
        self.assertTrue(v4["image_list"]["method"].startswith("tracking/v2"))
        self.assertEqual(Counter(s for _, s in self.export_rows(V4)), Counter({"train": 100, "valid": 20, "test": 50}))
        self.assertIsNone(self.cur["roboflow - 20260824 cryoai v3"]["image_list"])     # unread: unknown, not empty
        self.assertTrue(self.cur["roboflow - 20260824 cryoai v3"]["image_list_gap"])

    def test_sha_reimport_and_real_untouched(self):
        for eid in (V5, V5S1, V4):
            self.assertEqual(importers.sha256_file(self.root / self.cur[eid]["image_list"]["file"]),
                             self.cur[eid]["image_list"]["sha256"])
        self.assertEqual(sum(self.second["appended"].values()), 0, self.second["appended"])
        self.assertEqual(self.real, {f.name: f.read_bytes() for f in REAL_REG.glob("*.jsonl")})


if __name__ == "__main__":
    unittest.main()
