"""Prediction files on Windows paths over 260 characters (TODO #54): data_utils/longpath.py and its use in
size_profile_from_predictions / add_layer_roles.

    python -m unittest tests.test_longpath -v        (from the repo root)

Known answers, on the multi-class fixture of test_counts_multiclass (hand-counted there):
- the same predictions in a short folder and in a folder whose files are > 260 characters give byte-identical
  morphology_all.csv, per_image_summary.csv and morphology_all_roles.csv, with the hand counts;
- control: the plain (pre-fix) read of the deep file sees zero detections (when Windows long paths are off);
- a missing predictions file is an error naming the image; an empty one is zero detections.
Everything is written to temp dirs.
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data_utils import config  # noqa: E402
from data_utils.longpath import LONG_PATH, long_path, prediction_file  # noqa: E402
from tests.test_counts_multiclass import V5, WANT, make_v5_fixture  # noqa: E402

OUTPUTS = ("morphology_all.csv", "per_image_summary.csv", "morphology_all_roles.csv")


def windows_long_paths_enabled() -> bool:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            return bool(winreg.QueryValueEx(k, "LongPathsEnabled")[0])
    except OSError:
        return False


def rmtree(path):
    """shutil.rmtree that also removes files past 260 characters (the \\\\?\\ form of the root)."""
    shutil.rmtree("\\\\?\\" + os.path.abspath(path) if os.name == "nt" else path, ignore_errors=True)


class LongPathTest(unittest.TestCase):
    def test_short_paths_unchanged(self):
        p = Path("predictions") / "a.txt"
        self.assertEqual(str(long_path(p)), str(p))                            # not even made absolute

    @unittest.skipUnless(os.name == "nt", "the \\\\?\\ prefix is Windows only")
    def test_long_paths_prefixed(self):
        deep = Path("C:/") / ("a" * 120) / ".." / ("b" * 130) / ("c" * 120) / "d.txt"   # 260 once normalized
        want = "C:\\" + "b" * 130 + "\\" + "c" * 120 + "\\d.txt"
        self.assertGreater(len(want), LONG_PATH)
        self.assertEqual(str(long_path(deep)), "\\\\?\\" + want)
        unc = "\\\\host\\share\\" + "d" * 240 + "\\e.txt"
        self.assertEqual(str(long_path(unc)), "\\\\?\\UNC\\host\\share\\" + "d" * 240 + "\\e.txt")
        self.assertEqual(str(long_path(long_path(unc))), str(long_path(unc)))        # idempotent

    @unittest.skipUnless(os.name == "nt", "the 260-character limit is Windows only")
    def test_deep_file_read_and_missing_is_an_error(self):
        from training.train_yolo import load_predictions_yolo_format
        tmp = Path(tempfile.mkdtemp(prefix="lp_"))
        try:
            deep = long_path(tmp / ("p" * 120) / ("q" * 120) / "r.txt")
            os.makedirs(deep.parent)
            deep.write_text("0 0.9 0.1 0.1 0.5 0.1 0.5 0.5\n", encoding="utf-8")
            plain = tmp / ("p" * 120) / ("q" * 120) / "r.txt"
            self.assertGreater(len(str(plain)), 260)
            if not windows_long_paths_enabled():                             # control: the old read sees nothing
                self.assertEqual(load_predictions_yolo_format(plain, 64, 64)[1], [])
            self.assertEqual(load_predictions_yolo_format(prediction_file(plain.parent, "r"), 64, 64)[1], [0.9])
            with self.assertRaisesRegex(FileNotFoundError, r"no predictions file for image s\.png: .*characters"):
                prediction_file(plain.parent, "s", "s.png")
        finally:
            rmtree(tmp)


@unittest.skipIf(config.cryoai_root(required=False) is None, "add_layer_roles needs the CryoAI folder")
class PipelineTest(unittest.TestCase):
    """size_profile_from_predictions -> add_layer_roles, as in test_counts_multiclass."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lp_"))
        self.addCleanup(rmtree, self.tmp)
        self.images, self.preds = make_v5_fixture(self.tmp / "short")

    def run_pipeline(self, preds, out, check=True):
        rs = []
        for cmd in ([sys.executable, "-m", "inference.size_profile_from_predictions", "--predictions-dir",
                     str(preds), "--images-dir", str(self.images), "--output-dir", str(out),
                     "--pixel-size", "1.0", "--no-plot"],
                    [sys.executable, "-m", "analysis.add_layer_roles", "--morphology-csv",
                     str(out / "morphology_all.csv"), "--predictions-dir", str(preds),
                     "--images-dir", str(self.images)]):
            r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            rs.append(r)
            if check:
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            if r.returncode:
                break
        return rs

    def read(self, out):
        return {n: (out / n).read_bytes() for n in OUTPUTS}

    @unittest.skipUnless(os.name == "nt", "the 260-character limit is Windows only")
    def test_deep_folder_reads_the_same_detections(self):
        deep = self.tmp / ("p" * 110) / ("q" * 110) / V5 / "predictions"   # model id still from the folder
        shutil.copytree(self.preds, long_path(deep))
        longest = max(len(str(deep / f.name)) for f in self.preds.iterdir())
        self.assertGreater(longest, 260)
        self.run_pipeline(self.preds, self.tmp / "out_short")
        self.run_pipeline(deep, self.tmp / "out_deep")
        short = self.read(self.tmp / "out_short")
        self.assertEqual(self.read(self.tmp / "out_deep"), short)            # byte-identical
        with open(self.tmp / "out_short" / "morphology_all_roles.csv", newline="", encoding="utf-8") as f:
            roles = list(csv.DictReader(f))
        for name, (n_out, n_ev, *_rest) in WANT.items():                       # the hand counts
            rs = [r for r in roles if r["image"] == name]
            self.assertEqual((len(rs), sum(r["role"] != "inner-layer" for r in rs)), (n_out, n_ev), name)

    def test_missing_file_is_an_error_empty_file_is_zero(self):
        (self.preds / "grid1_2_b.txt").write_text("", encoding="utf-8")       # empty = no detections
        self.run_pipeline(self.preds, self.tmp / "out")
        with open(self.tmp / "out" / "per_image_summary.csv", newline="", encoding="utf-8") as f:
            rows = {r["image"]: r for r in csv.DictReader(f)}
        self.assertEqual(rows["grid1_2_b.png"]["n_detected_objects"], "0")
        self.assertEqual(rows["grid1_1_a.png"]["n_detected_objects"], "5")
        (self.preds / "grid1_2_b.txt").unlink()                                # missing = error
        r = self.run_pipeline(self.preds, self.tmp / "out2", check=False)
        self.assertEqual(len(r), 1, "add_layer_roles not reached")
        self.assertNotEqual(r[0].returncode, 0)
        self.assertIn("no predictions file for image grid1_2_b.png", r[0].stderr)
        self.assertIn("characters", r[0].stderr)
        self.assertFalse((self.tmp / "out2" / "per_image_summary.csv").exists(), "no partial outputs")
        (self.preds / "grid1_1_a.txt").unlink()                                # add_layer_roles too
        r = subprocess.run([sys.executable, "-m", "analysis.add_layer_roles", "--morphology-csv",
                            str(self.tmp / "out" / "morphology_all.csv"), "--predictions-dir", str(self.preds),
                            "--images-dir", str(self.images)], cwd=REPO, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no predictions file for image grid1_1_a.png", r.stderr)


if __name__ == "__main__":
    unittest.main()
