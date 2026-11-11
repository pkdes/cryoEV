"""compare_package: known-answer cards and decoys on a synthetic image.

One 400x400 image. Set A labels: EV at (100,100) matched by B; EV at (300,100) that B misses (FN card);
Low conf EV at (100,300) that B misses (never a card: fair view ignores it).
Set B predictions: EV 0.90 on the first label; EV 0.70 at (300,300) with no label (FP card); non-EV 0.80 at
(200,200) with no label (shown, never a card); EV 0.40 at (200,50) below flag_conf (shown, never a card).
Known answer: 1 FP card, 1 FN card. Control: --decoys 0 adds none; --decoys 1 adds accept + reject decoys whose
key stays out of data.js, and an accept decoy's B match is hidden.
"""
import argparse
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import compare_package as CP  # noqa: E402

W = H = 400


def ring(cx, cy, r, n=16):
    return [round(v, 2) for k in range(n) for v in (cx + r * math.cos(2 * math.pi * k / n),
                                                   cy + r * math.sin(2 * math.pi * k / n))]


def yolo(k, conf, cx, cy, r):
    return f"{k} {conf} " + " ".join(f"{v / W:.5f}" for v in ring(cx, cy, r))


class CompareKnownAnswer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        (t / "imgs").mkdir()
        Image.new("L", (W, H), 128).save(t / "imgs" / "tile_a.png")
        cats = [{"id": 1, "name": "EV"}, {"id": 2, "name": "Low conf EV"}, {"id": 3, "name": "non-EV"}]
        anns = [{"id": 11, "image_id": 1, "category_id": 1, "segmentation": [ring(100, 100, 20)]},
                {"id": 12, "image_id": 1, "category_id": 1, "segmentation": [ring(300, 100, 20)]},
                {"id": 13, "image_id": 1, "category_id": 2, "segmentation": [ring(100, 300, 20)]}]
        self.labels = t / "labels.coco.json"
        self.labels.write_text(json.dumps({"images": [{"id": 1, "file_name": "tile_a.png", "width": W, "height": H,
                                                       "extra": {"name": "tile_a_png", "split": "train"}}],
                                           "categories": cats, "annotations": anns}), encoding="utf-8")
        (t / "preds").mkdir()
        (t / "preds" / "tile_a_png.txt").write_text("\n".join([
            yolo(0, 0.90, 100, 100, 20), yolo(0, 0.70, 300, 300, 20), yolo(1, 0.80, 200, 200, 20),
            yolo(0, 0.40, 200, 50, 15)]), encoding="utf-8")
        self.t = t

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, decoys, name):
        a = argparse.Namespace(labels=self.labels, images_root=[self.t / "imgs"], select_csv=None, where=None,
                               split_csv=None, b_model=None, b_preds=self.t / "preds", b_coco=None, b_label="B",
                               imgsz=1440, nms=0.7, max_det=1000, device="cpu", pred_classes="EV,non-EV",
                               class_groups=CP.B.CLASS_GROUPS, flag_conf=0.55, decoys=decoys, seed=0, owner="tester",
                               reviewer=None, reuse_key=None, index_html=None, width=1440, quality=72,
                               out=self.t / "review" / name)
        out, work = CP.build(a)
        data = json.loads((out / "data.js").read_text(encoding="utf-8").split("=", 1)[1].rstrip().rstrip(";"))
        return out, work, data

    def test_cards_known_answer_without_decoys(self):
        out, work, data = self.build(0, "lab_cmp_none")
        im = data["images"][0]
        fp = [p for p in im["pr"] if p.get("u")]
        fn = [g for g in im["gt"] if g.get("u")]
        self.assertEqual(len(fp), 1)
        self.assertAlmostEqual(fp[0]["s"], 0.70, places=2)
        self.assertEqual([g["id"] for g in fn], [12])                 # Low conf EV (13) is not a card
        self.assertEqual(len(im["pr"]), 4)                             # every B object is still shown
        meta = json.loads((out / "review_package.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["kind"], meta["cards"]["fp"], meta["cards"]["fn"]), ("compare", 1, 1))
        self.assertEqual(json.loads((work / "decoys.json").read_text(encoding="utf-8"))["decoys"], [])
        self.assertTrue((out / ".review_key").read_text(encoding="utf-8").strip())

    def test_decoys_known_answer_and_key_not_served(self):
        out, work, data = self.build(1.0, "lab_cmp_mixed")
        key = json.loads((work / "decoys.json").read_text(encoding="utf-8"))["decoys"]
        kinds = sorted(k["kind"] for k in key)
        self.assertIn("accept", kinds)
        self.assertIn("reject", kinds)
        im = data["images"][0]
        for k in key:
            if k["kind"] == "accept":
                self.assertEqual(k["obj"], "g11")                      # the only label B also found
                self.assertTrue(next(g for g in im["gt"] if g["id"] == 11).get("u"))
                self.assertFalse(any(abs(p["s"] - 0.90) < 1e-6 for p in im["pr"]))  # its B match is hidden
            else:
                p = im["pr"][int(k["obj"][1:])]
                self.assertTrue(p.get("u"))
                self.assertGreaterEqual(p["s"], 0.55)
        served = (out / "data.js").read_text(encoding="utf-8")
        self.assertNotIn("decoy", served)
        self.assertNotIn("answer", served)
        self.assertFalse(any(out.rglob("decoys.json")))


if __name__ == "__main__":
    unittest.main()
