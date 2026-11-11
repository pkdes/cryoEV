"""Known-answer controls for analysis/annotation_agreement.py."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis"))
import annotation_agreement as AA  # noqa: E402

CLASSES = ["EV", "MV", "inner", "VLDL"]
W, H = 2000, 1400


def circle(cx, cy, r, n=48):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]


def grid_objs(n=100):
    """n separated circles (pixel coords) with classes cycling through CLASSES."""
    objs, k = [], 0
    for row in range(10):
        for col in range(10):
            if k == n:
                return objs
            objs.append((CLASSES[k % 4], circle(100 + col * 190, 80 + row * 130, 40)))
            k += 1
    return objs


def norm(objs, w=W, h=H):
    return [(c, p / (w, h)) for c, p in objs]


def write_coco(path, tiles):
    """tiles: {stem: (w, h, [(cls, pixel polygon)])}; category ids deliberately differ per file."""
    names = sorted({c for _, _, objs in tiles.values() for c, _ in objs}, reverse=path.name.startswith("b"))
    cats = {n: i + 7 for i, n in enumerate(names)}
    images, anns = [], []
    for i, (stem, (w, h, objs)) in enumerate(tiles.items()):
        images.append({"id": i, "file_name": f"{stem}_jpg.rf.Zz9{i}.jpg", "width": w, "height": h,
                       "extra": {"name": f"{stem}.jpg"}})
        for c, p in objs:
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cats[c],
                         "segmentation": [p.ravel().tolist()]})
    path.write_text(json.dumps({"images": images, "annotations": anns,
                                "categories": [{"id": v, "name": k} for k, v in cats.items()]}))


class Controls(unittest.TestCase):
    def test_self_compare_is_perfect(self):
        o = norm(grid_objs())
        s = AA.summarize(AA.compare_tile(o, o, W, H))
        self.assertEqual((s["f1"], s["kappa"], s["kept"], s["matched"]), (1.0, 1.0, 100, 100))

    def test_drop_10pct_reclass_5pct(self):
        pre = grid_objs()
        post = []
        for k, (c, p) in enumerate(pre):
            if k % 10 == 3:                 # 10 dropped
                continue
            if k % 20 == 0:                 # 5 reclassed (k = 0, 20, ..., 80; none dropped)
                c = "VLDL" if c != "VLDL" else "EV"
            post.append((c, p))
        rows = AA.compare_tile(norm(pre), norm(post), W, H)
        s = AA.summarize(rows)
        self.assertEqual(s["recall"], 0.9)
        self.assertEqual(s["precision"], 1.0)
        self.assertEqual((s["deleted"], s["reclassed"], s["kept"], s["added"]), (10, 5, 85, 0))
        # kappa independently from the matched-pair confusion matrix
        m = [r for r in rows if r["op"] != "deleted"]
        lab = sorted({r["pre_cls"] for r in m} | {r["post_cls"] for r in m})
        cm = np.array([[sum(r["pre_cls"] == a and r["post_cls"] == b for r in m) for b in lab] for a in lab], float)
        n = cm.sum()
        po, pe = np.trace(cm) / n, (cm.sum(1) @ cm.sum(0)) / n ** 2
        self.assertAlmostEqual(s["kappa"], round((po - pe) / (1 - pe), 4), places=4)
        self.assertLess(s["kappa"], 1.0)

    def test_added_outline(self):
        pre = grid_objs(20)
        post = pre + [("EV", circle(1900, 1300, 30))]
        s = AA.summarize(AA.compare_tile(norm(pre), norm(post), W, H))
        self.assertEqual((s["added"], s["recall"], s["matched"]), (1, 1.0, 20))

    def test_reshape_vs_kept(self):
        pre = [("EV", circle(500, 500, 40))]
        moved = [("EV", circle(510, 500, 40))]        # IoU ~0.84: reshaped, still matched
        far = [("EV", circle(560, 500, 40))]          # IoU < 0.5: delete + add
        self.assertEqual(AA.summarize(AA.compare_tile(norm(pre), norm(moved), W, H))["reshaped"], 1)
        s = AA.summarize(AA.compare_tile(norm(pre), norm(far), W, H))
        self.assertEqual((s["deleted"], s["added"], s["matched"]), (1, 1, 0))

    def test_resolution_independent(self):
        pre = grid_objs()
        half = [(c, p / 2) for c, p in pre]           # same field at half the pixels
        s = AA.summarize(AA.compare_tile(norm(pre), norm(half, W // 2, H // 2), W, H))
        self.assertEqual(s["matched"], 100)
        self.assertGreater(s["mean_iou"], 0.95)

    def test_kappa_textbook(self):
        pairs = [("y", "y")] * 20 + [("y", "n")] * 5 + [("n", "y")] * 10 + [("n", "n")] * 15
        self.assertAlmostEqual(AA.cohen_kappa(pairs), 0.4)
        self.assertIsNone(AA.cohen_kappa([]))

    def test_per_class_and_confusion_sum(self):
        pre = grid_objs(40)
        post = [(("MV" if c == "EV" else c), p) for c, p in pre[:30]]
        rows = AA.compare_tile(norm(pre), norm(post), W, H)
        pc = {r["class"]: r for r in AA.per_class(rows)}
        self.assertEqual(pc["EV"]["same_class_matched"], 0)
        self.assertEqual(pc["EV"]["to_other_class"], 8)
        classes, cm = AA.confusion(rows)
        self.assertNotIn(AA.NONE, classes[:-1])
        self.assertEqual(sum(map(sum, cm)), len(rows))

    def test_breakdown_known_answer(self):
        # 100 objects: drop 10 (only pre), reclass 5, add 3 new (only post)
        pre = grid_objs()
        post = []
        for k, (c, p) in enumerate(pre):
            if k % 10 == 3:
                continue
            if k % 20 == 0:
                c = "VLDL" if c != "VLDL" else "EV"
            post.append((c, p))
        post += [("EV", circle(1900, 1330 - 0, 20)), ("EV", circle(1960, 1330, 20)), ("MV", circle(1900, 30, 20))]
        rows = AA.compare_tile(norm(pre), norm(post), W, H)
        s = AA.summarize(rows)
        self.assertEqual((s["n_union"], s["agreed"], s["class_differs"], s["only_pre"], s["only_post"]),
                         (103, 85, 5, 10, 3))
        self.assertAlmostEqual(s["pct_agreed"] + s["pct_class_differs"] + s["pct_only_pre"] + s["pct_only_post"],
                               1.0, places=3)
        # per class: every row of a class's union counted once; a reclass counts under both classes
        for r in AA.per_class(rows):
            self.assertEqual(r["union"], r["agreed"] + r["class_differs"] + r["only_pre"] + r["only_post"])
        self.assertEqual(sum(r["class_differs"] for r in AA.per_class(rows)), 2 * 5)


def last_row(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))[-1]


class Cli(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(REPO / "analysis" / "annotation_agreement.py"), *args],
                              capture_output=True, text=True)

    def test_mirror_map_and_name_matching(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            objs = grid_objs(30)
            flipped = [(c, np.c_[p[:, 0], H - p[:, 1]]) for c, p in objs]     # 'v' = up-down
            write_coco(d / "a.coco.json", {"blind_01": (W, H, flipped)})
            write_coco(d / "b.coco.json", {"grid2_25_r1c0_t01": (W, H, objs)})
            with open(d / "map.csv", "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["tile", "blind_name", "mirror"])
                wr.writerow(["grid2_25_r1c0_t01", "blind_01.png", "v"])
            r = self.run_cli("--pre", str(d / "a.coco.json"), "--post", str(d / "b.coco.json"),
                             "--mirror-map", str(d / "map.csv"), "--out", str(d / "out"))
            self.assertEqual(r.returncode, 0, r.stderr)
            tot = last_row(d / "out" / "per_tile.csv")
            self.assertEqual((tot["f1"], tot["kappa"], tot["kept"]), ("1.0", "1.0", "30"))
            # control: the wrong flip must NOT agree
            with open(d / "map2.csv", "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["tile", "blind_name", "mirror"])
                wr.writerow(["grid2_25_r1c0_t01", "blind_01.png", "h"])
            self.run_cli("--pre", str(d / "a.coco.json"), "--post", str(d / "b.coco.json"),
                         "--mirror-map", str(d / "map2.csv"), "--out", str(d / "out2"))
            tot2 = last_row(d / "out2" / "per_tile.csv")
            self.assertLess(float(tot2["f1"]), 0.5)
            # never overwrites
            r = self.run_cli("--pre", str(d / "a.coco.json"), "--post", str(d / "b.coco.json"), "--out", str(d / "out"))
            self.assertNotEqual(r.returncode, 0)

    def test_crops_one_cell_per_disagreement(self):
        import cv2
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pre = grid_objs(30)
            post = [(("MV" if k == 4 else c), p) for k, (c, p) in enumerate(pre) if k not in (7, 9)]
            post.append(("EV", circle(1900, 1300, 30)))
            write_coco(d / "a.coco.json", {"tileA": (W, H, pre)})
            write_coco(d / "b.coco.json", {"tileA": (W, H, post)})
            (d / "img").mkdir()
            cv2.imwrite(str(d / "img" / "x__tileA__y.png"), np.full((H, W, 3), 128, np.uint8))
            r = self.run_cli("--pre", str(d / "a.coco.json"), "--post", str(d / "b.coco.json"),
                             "--image-dir", str(d / "img"), "--crops", "--nm-per-px", "0.5", "--out", str(d / "out"))
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(d / "out" / "crops" / "index.csv", encoding="utf-8") as fh:
                idx = list(csv.DictReader(fh))
            self.assertEqual(sorted(x["category"] for x in idx),
                             ["class_differs", "only_post", "only_pre", "only_pre"])
            cd = [x for x in idx if x["category"] == "class_differs"][0]
            self.assertEqual((cd["pre_cls"], cd["post_cls"]), (pre[4][0], "MV"))
            add = [x for x in idx if x["category"] == "only_post"][0]
            self.assertAlmostEqual(float(add["cx_px"]), 1900, delta=1)
            self.assertAlmostEqual(float(add["diam_px"]), 60, delta=2)
            self.assertAlmostEqual(float(add["diam_nm"]), 30, delta=1)
            sheets = sorted(p.name for p in (d / "out" / "crops").glob("*.jpg"))
            self.assertEqual(sheets, ["class_differs_p1.jpg", "only_post_p1.jpg", "only_pre_p1.jpg"])
            self.assertIn("Both, same class | 27 |", (d / "out" / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
