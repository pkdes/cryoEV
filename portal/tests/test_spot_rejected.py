"""spot_list skips images rejected after spotcheck.json was written (round 7 R1 QC deviation 8).

Known answer: spotcheck.json lists 0001, 0002, 0003; 0002 is rejected in its edit record -> [0001, 0003].
Control: with no edit records the list is unchanged; an unreadable record counts as not rejected.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402,F401  (sets up the import path like the other portal tests)
from portal.modules import predict as P  # noqa: E402
from portal.predict_parts import sets as SETS  # noqa: E402


class SpotSkipsRejected(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pkg = Path(self.tmp.name) / "pred_ps_x"
        (self.pkg / "edits").mkdir(parents=True)
        (self.pkg / "spotcheck.json").write_text(json.dumps(
            {"n": 3, "images": [{"image": i} for i in ("0001", "0002", "0003")]}), encoding="utf-8")
        self.rows = [{"image": i, "file": f"images/{i}.png", "sample_label": "S", "orig_name": f"{i}.tif"}
                     for i in ("0001", "0002", "0003")]

    def tearDown(self):
        self.tmp.cleanup()

    def ids(self):
        with mock.patch.object(SETS, "package_dir", return_value=self.pkg):
            return [r["image"] for r in P.spot_list(None, Path(self.tmp.name), "ps_x", self.rows)]

    def edit(self, name, edit):
        (self.pkg / "edits" / f"{name}.json").write_text(json.dumps({"edit": edit}), encoding="utf-8")

    def test_control_no_edits(self):
        self.assertEqual(self.ids(), ["0001", "0002", "0003"])

    def test_rejected_image_is_skipped(self):
        self.edit("0002.png", {"dec": {"i:reject": {"v": "reject:thick ice", "t": 1, "who": "a"}}})
        self.assertEqual(self.ids(), ["0001", "0003"])

    def test_cleared_reject_and_note_only_are_kept(self):
        self.edit("0001.png", {"dec": {"i:reject": {"v": None, "t": 2, "who": "a"}}})
        self.edit("0003.png", {"dec": {"i:note": {"v": "faint", "t": 1, "who": "a"}}})
        self.assertEqual(self.ids(), ["0001", "0002", "0003"])

    def test_unreadable_record_counts_as_not_rejected(self):
        (self.pkg / "edits" / "0002.png.json").write_text("{torn", encoding="utf-8")
        self.assertEqual(self.ids(), ["0001", "0002", "0003"])


if __name__ == "__main__":
    unittest.main()
