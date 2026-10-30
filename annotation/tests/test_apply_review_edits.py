"""
Tests for annotation/apply_review_edits.py: a synthetic package (planted control: one edit of each kind, with an
exact expected COCO diff), plus the real inputs when they sit beside the repo (copied to temp, never written):
the v2 September seeds against resolution.csv, the v1 sandbox smoke edits (the MDA-MB-231 duplicate), a round
trip through build_review_package.py, and the page's effGt() against effective_labels() in headless Edge.

  python -m unittest discover -s annotation/tests -p "test_apply_review_edits.py" -v
"""
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "annotation"))
sys.path.insert(0, str(REPO / "training"))
import apply_review_edits as ar  # noqa: E402
import review_server as rs  # noqa: E402

LAPTOP = next((p / "CryoAI-laptop" for p in REPO.parents if (p / "CryoAI-laptop").is_dir()), None)
REAL_BASE = LAPTOP / "roboflow_pull_20261029_1811" / "current.coco.json" if LAPTOP else None
REAL_V2 = LAPTOP / "review" / "prelipo_v5F5_review_v2" if LAPTOP else None
REAL_V1 = LAPTOP / "review" / "SANDBOX_prelipo_v5F5_review" if LAPTOP else None
SMOKE_EDITS = LAPTOP / "fixtures" / "sandbox_v1_smoketest_20261030" / "edits" if LAPTOP else None
RESOLUTION = LAPTOP / "september_edits" / "v2" / "resolution.csv" if LAPTOP else None
MDA = "MDA-MB-231_RM_square237_hole31"
W, H = 1000, 800


def have(*paths):
    return all(p is not None and p.exists() for p in paths)


def sq(x, y, s):
    return [float(x), float(y), float(x + s), float(y), float(x + s), float(y + s), float(x), float(y + s)]


def norm(ring):
    return [round(v / (W if k % 2 == 0 else H), 5) for k, v in enumerate(ring)]


def md5_tree(folder: Path) -> dict:
    return {str(p.relative_to(folder)): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def run(base, pkg, edits, out_root, stamp="20990101_0000", extra=()):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        v = ar.main(["--base", str(base), "--package", str(pkg), "--edits", str(edits), "--out-root", str(out_root),
                     "--stamp", stamp, *extra])
    out = Path(out_root) / f"lv_{stamp}"
    return v, out, buf.getvalue()


def read_out(out: Path):
    coco = json.loads((out / "labels.coco.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (out / "changes.jsonl").read_text(encoding="utf-8").splitlines()]
    with open(out / "conflicts.csv", newline="", encoding="utf-8") as f:
        conflicts = list(csv.DictReader(f))
    return coco, rows, conflicts, json.loads((out / "version.json").read_text(encoding="utf-8"))


def key_of(a):
    return (a["id"], a["image_id"], a["category_id"], json.dumps(a["segmentation"]), a.get("iscrowd"))


# ---------------------------------------------------------------- synthetic package

BASE_ANNS = [  # id, image, category, rings (pixels)
    (1, 1, 1, [sq(10, 16, 40)]), (2, 1, 1, [sq(100, 16, 40)]), (3, 1, 1, [sq(200, 16, 40)]),
    (4, 1, 1, [sq(300, 16, 40)]), (5, 1, 2, [sq(400, 16, 40)]), (6, 1, 1, [sq(500, 16, 40)]),
    (7, 1, 3, [sq(10, 400, 40), sq(80, 400, 24)]),
    (8, 2, 1, [sq(10, 16, 80)]), (9, 2, 1, [sq(200, 16, 80)]),
]
PR_A = [sq(600, 504, 48), sq(700, 504, 48), sq(800, 504, 48)]
RESHAPE3 = [sq(208, 24, 32), sq(208, 120, 16)]  # multi-ring reshape of ann 3
RESHAPE4 = [sq(296, 8, 56)]
ADD_P1 = [sq(696, 496, 64)]
NEW_VLDL = [[600.0, 200.0, 1000.0005, 200.0, 1000.0005, 280.0, 600.0, 280.0]]  # past the right edge: clamped
LIST_RESHAPE9 = [sq(208, 24, 64)]
LIST_NEW = [sq(500, 400, 40)]


def build_synthetic(root: Path, extra=()):
    """base.coco.json, pkg/data.js, pkg/edits/*.json -> (base, pkg, edits)."""
    root.mkdir(parents=True, exist_ok=True)
    base = {"info": {"source": "synthetic"},
            "images": [{"id": 1, "file_name": "a.png", "width": W, "height": H,
                        "extra": {"name": "a_png", "split": "train", "roboflow_id": "rfA"}},
                       {"id": 2, "file_name": "b.png", "width": W, "height": H,
                        "extra": {"name": "b_png", "split": "valid", "roboflow_id": "rfB"}},
                       {"id": 3, "file_name": "c.png", "width": W, "height": H,
                        "extra": {"name": "c_png", "split": "test", "roboflow_id": "rfC"}}],
            "annotations": [{"id": i, "image_id": im, "category_id": c, "segmentation": rings, "iscrowd": 0}
                            for i, im, c, rings in [*BASE_ANNS, *extra]],
            "categories": [{"id": 1, "name": "EV"}, {"id": 2, "name": "Low conf EV"}, {"id": 3, "name": "MV"}]}
    (root / "base").mkdir(exist_ok=True)
    (root / "base" / "current.coco.json").write_text(json.dumps(base), encoding="utf-8")
    names = {c["id"]: c["name"] for c in base["categories"]}
    images = []
    for img in base["images"]:
        gt = [{"id": i, "c": names[c], "p": [norm(r) for r in rings]} for i, im, c, rings in [*BASE_ANNS, *extra] if im == img["id"]]
        pr = [{"s": 0.9, "c": "EV", "p": [norm(r)]} for r in (PR_A if img["id"] == 1 else [sq(10, 16, 80)] if img["id"] == 2 else [])]
        images.append({"file": img["file_name"], "split": "train", "w": W, "h": H, "src": f"images/{img['extra']['name']}.jpg",
                       "gt": gt, "pr": pr, "fg": 0, "fp": 0})
    pkg = root / "pkg"
    (pkg / "edits").mkdir(parents=True, exist_ok=True)
    (pkg / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(
        {"id": "synthetic_pkg", "categories": ["EV", "Low conf EV", "MV"], "images": images}) + ";\n", encoding="utf-8")
    np_ = lambda rings: [norm(r) for r in rings]
    t = 1000
    rec_a = {"file": "a.png", "pkg": "synthetic_pkg", "fmt": 1, "rx": {"g2": 5555, "p0": 6666},
             "edit": {"dec": {
                 "g1": {"v": "keep:EV", "t": t, "who": "Puj"},
                 "g2": {"v": "keep:MV", "t": t, "who": "pujan", "seed": "september:x.jpg:g12"},
                 "g3": {"v": "keep:EV", "p": np_(RESHAPE3), "t": t, "who": "Pujan"},
                 "g4": {"v": "keep:MV", "p": np_(RESHAPE4), "t": t, "who": "September review"},
                 "g5": {"v": "del", "t": t, "who": "Zed"},
                 "g6": {"v": None, "t": t, "who": "Pujan"},
                 "p0": {"v": "add:Low conf EV", "t": t, "who": "Pujan"},
                 "p1": {"v": "add:EV", "p": np_(ADD_P1), "t": t, "who": "Pujan"},
                 "p2": {"v": "no", "t": t, "who": "Pujan"},
                 "nabc": {"v": "new:VLDL", "p": [[round(v / (W if k % 2 == 0 else H), 7) for k, v in enumerate(NEW_VLDL[0])]],
                          "t": t, "who": "Pujan"},
                 "nxyz": {"v": "del", "t": t, "who": "Pujan"},
                 "g99": {"v": "keep:MV", "t": t, "who": "Pujan"},
                 "p9": {"v": "add:EV", "t": t, "who": "Pujan"},
                 "nbad": {"v": "new:EV", "p": [[0.1, 0.1, 0.2, 0.2, 0.3, 0.3]], "t": t, "who": "Pujan"},
             }, "t": t, "modified": True, "reviewed": True, "reviewedT": t}}
    gt_b = [{"id": 8, "c": "EV", "p": np_([sq(10, 16, 80)])}, {"id": 9, "c": "EV", "p": np_(LIST_RESHAPE9)},
            {"id": None, "c": "EV", "p": np_(LIST_NEW), "seed": "september:y.jpg:p5"}]
    rec_b = {"file": "b.png",  # an old v1 record: no wrapper fields, a gtT=1 list with a keyless item
             "edit": {"dec": {"p0": {"v": "add:MV", "t": t, "who": "Pujan"}}, "gt": gt_b, "gtT": 1,
                      "t": t, "modified": True, "reviewed": False, "reviewedT": 0}}
    rec_c = {"file": "c.png", "edit": {"dec": {}, "t": t, "modified": False, "reviewed": True, "reviewedT": t}}
    for rec in (rec_a, rec_b, rec_c):
        (pkg / "edits" / f"{rec['file']}.json").write_text(json.dumps(rec), encoding="utf-8")
    return root / "base" / "current.coco.json", pkg, pkg / "edits"


def px(rings):
    return [[round(min(max(v, 0.0), W if k % 2 == 0 else H), 2) for k, v in enumerate(r)] for r in rings]


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="apply_edits_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.base, self.pkg, self.edits = build_synthetic(self.tmp / "in")
        self.out_root = self.tmp / "labels"

    def test_1_zero_edits_equal_base(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        v, out, _ = run(self.base, self.pkg, empty, self.out_root)
        coco, rows, conflicts, ver = read_out(out)
        base = json.loads(self.base.read_text(encoding="utf-8"))
        self.assertEqual([key_of(a) for a in coco["annotations"]], [key_of(a) for a in base["annotations"]])
        self.assertEqual(coco["images"], base["images"])
        self.assertEqual(coco["categories"], base["categories"])
        self.assertEqual((rows, conflicts, ver["n_images_applied"]), ([], [], 0))
        self.assertEqual(coco["annotations"][0]["bbox"], [10.0, 16.0, 40.0, 40.0])
        self.assertEqual(coco["annotations"][0]["area"], 1600.0)

    def test_3_planted_control_exact_diff(self):
        before = md5_tree(self.tmp / "in")
        v, out, log = run(self.base, self.pkg, self.edits, self.out_root)
        self.assertEqual(md5_tree(self.tmp / "in"), before, "inputs were modified")
        coco, rows, conflicts, ver = read_out(out)
        base = {a["id"]: a for a in json.loads(self.base.read_text(encoding="utf-8"))["annotations"]}
        got = {a["id"]: a for a in coco["annotations"]}
        expect = {i: dict(a) for i, a in base.items()}
        expect[2]["category_id"] = 3                                   # reclass: base outline kept
        expect[3]["segmentation"] = px(RESHAPE3)                       # reshape, two rings, one annotation
        expect[4].update(category_id=3, segmentation=px(RESHAPE4))     # reclass+reshape
        del expect[5]                                                  # delete
        expect[9]["segmentation"] = px(LIST_RESHAPE9)                  # reshape from an old list (dec_key gt)
        expect[10] = {"id": 10, "image_id": 1, "category_id": 2, "segmentation": px([PR_A[0]]), "iscrowd": 0}  # add p0
        expect[11] = {"id": 11, "image_id": 1, "category_id": 1, "segmentation": px(ADD_P1), "iscrowd": 0}    # add p1 + p
        expect[12] = {"id": 12, "image_id": 1, "category_id": 4, "segmentation": [[600.0, 200.0, 1000.0, 200.0, 1000.0, 280.0, 600.0, 280.0]],
                      "iscrowd": 0}                                    # new VLDL, clamped at x = 1000
        expect[13] = {"id": 13, "image_id": 2, "category_id": 1, "segmentation": px(LIST_NEW), "iscrowd": 0}  # keyless list item
        self.assertEqual(sorted(got), sorted(expect))
        for i in expect:
            self.assertEqual(key_of(got[i]), key_of(expect[i]), i)
            bb, area = ar.bbox_area(got[i]["segmentation"])
            self.assertEqual((got[i]["bbox"], got[i]["area"]), (bb, area))
            self.assertFalse(set(got[i]) & set(ar.INTERNAL), i)
        self.assertEqual(coco["categories"][-1], {"id": 4, "name": "VLDL"})
        self.assertTrue(ver["new_categories_flag"])
        self.assertEqual([c["name"] for c in ver["new_categories"]], ["VLDL"])

        by = {(r["image"], r["dec_key"], r["op"]): r for r in rows}
        ops = Counter((r["op"], r["status"]) for r in rows)
        self.assertEqual(ops, Counter({("keep", "applied"): 1, ("reclass", "applied"): 1, ("reshape", "applied"): 2,
                                       ("reclass+reshape", "applied"): 1, ("delete", "applied"): 1, ("add", "applied"): 2,
                                       ("new", "applied"): 2, ("reject", "applied"): 1, ("reclass", "conflict"): 1,
                                       ("add", "conflict"): 2, ("new", "conflict"): 1}))
        self.assertEqual(ver["counts_by_op"], {"keep": 1, "reclass": 1, "reshape": 2, "reclass+reshape": 1, "delete": 1,
                                               "add": 2, "new": 2, "reject": 1})
        self.assertEqual(ver["skipped"], {"removed_again": 1, "undone": 1})
        r = by[("a_png", "g2", "reclass")]
        self.assertEqual((r["ann_old"], r["ann_new"], r["cls_from"], r["cls_to"], r["who_raw"], r["who"], r["rx"], r["seed"],
                          r["reviewed"], r["rf_id"], r["who_auth"], r["t"], r["schema"], r["label_version"], r["pkg"], r["base"]),
                         (2, 2, "EV", "MV", "pujan", "pujan", 5555, "september:x.jpg:g12", True, "rfA", None, 1000,
                          "label_change/1", "lv_20990101_0000", "synthetic_pkg", "base"))
        self.assertEqual(by[("a_png", "g1", "keep")]["who"], "pujan")              # alias Puj
        self.assertEqual(by[("a_png", "g4", "reclass+reshape")]["who"], "pujan")   # alias September review
        self.assertEqual(by[("a_png", "g5", "delete")]["who"], "unknown")          # not in the roster
        self.assertEqual((by[("a_png", "g5", "delete")]["ann_new"], by[("a_png", "g5", "delete")]["cls_to"]), (None, None))
        self.assertEqual(by[("a_png", "p0", "add")]["ann_new"], 10)
        self.assertEqual(by[("a_png", "p0", "add")]["rx"], 6666)
        self.assertEqual(by[("a_png", "p2", "reject")]["ann_new"], None)
        self.assertEqual(by[("a_png", "nabc", "new")]["cls_to"], "VLDL")
        r = by[("b_png", "gt", "new")]
        self.assertEqual((r["seed"], r["who"], r["t"], r["ann_new"], r["reviewed"], r["rx"]),
                         ("september:y.jpg:p5", "unknown", 1, 13, False, None))
        self.assertEqual(by[("b_png", "gt", "reshape")]["ann_old"], 9)
        reasons = {(c["image"], c["dec_key"]): c["reason"] for c in conflicts}
        self.assertEqual(reasons[("a_png", "g99")], "g_missing_from_base")
        self.assertEqual(reasons[("a_png", "p9")], "pred_index_out_of_range")
        self.assertTrue(reasons[("a_png", "nbad")].startswith("invalid_polygon"))
        self.assertEqual(len(conflicts), 4)
        for r in rows:
            self.assertEqual(r["status"] == "conflict", (r["image"], r["dec_key"]) in reasons)
            if r["status"] == "conflict":
                self.assertIsNone(r["ann_new"])
        self.assertEqual(ver["n_images_applied"], 2)  # c.png: reviewed only, nothing to apply
        for f in ("id", "base", "pkg", "edits_dir", "n_images_applied", "counts_by_op", "class_groups_sha256",
                  "people_sha256", "tool", "created"):
            self.assertIn(f, ver)
        self.assertTrue(ver["tool"].startswith("apply_review_edits.py "))
        self.assertIn("[steps]", log)

    def test_4_duplicate_outline_flagged(self):
        v, out, _ = run(self.base, self.pkg, self.edits, self.out_root)
        coco, rows, conflicts, _ = read_out(out)
        dup = [c for c in conflicts if c["image"] == "b_png"]
        self.assertEqual([(c["dec_key"], c["op"]) for c in dup], [("p0", "add")])
        self.assertIn("different class", dup[0]["reason"])
        self.assertFalse(any(a["image_id"] == 2 and a["category_id"] == 3 for a in coco["annotations"]))
        # a same-class duplicate is caught too, and a lower threshold catches a near copy
        rec = json.loads((self.edits / "b.png.json").read_text(encoding="utf-8"))
        rec["edit"]["dec"]["p0"]["v"] = "add:EV"
        (self.edits / "b.png.json").write_text(json.dumps(rec), encoding="utf-8")
        _, out2, _ = run(self.base, self.pkg, self.edits, self.out_root, stamp="20990101_0001")
        coco2, rows2, c2, ver2 = read_out(out2)
        c2 = [c for c in c2 if c["image"] == "b_png"]
        self.assertEqual([(c["severity"], c["dec_key"]) for c in c2], [("dropped", "p0")])  # the base label stays
        self.assertIn("same class", c2[0]["reason"])
        r = next(r for r in rows2 if r["image"] == "b_png" and r["dec_key"] == "p0")
        self.assertEqual((r["status"], r["ann_new"]), ("duplicate-dropped", None))
        self.assertEqual(ver2["n_dropped"], 1)
        self.assertEqual(sum(a["image_id"] == 2 for a in coco2["annotations"]), 3)  # 8, 9 and the list's new one
        self.assertEqual(ar.mask_iou(px([sq(10, 16, 80)]), px([sq(10, 16, 80)])), 1.0)
        self.assertLess(ar.mask_iou(px([sq(10, 16, 80)]), px([sq(18, 16, 80)])), 0.9)

    def test_5_round_trip_build_review_package(self):
        v, out, _ = run(self.base, self.pkg, self.edits, self.out_root)
        coco, rows, conflicts, _ = read_out(out)
        imgs, preds = self.tmp / "imgs", self.tmp / "preds"
        imgs.mkdir(); preds.mkdir()
        for name in ("a_png", "b_png", "c_png"):
            cv2.imwrite(str(imgs / f"{name}.png"), np.zeros((H, W), np.uint8))
        pkg2 = self.tmp / "pkg2"
        subprocess.run([sys.executable, str(REPO / "annotation" / "build_review_package.py"), "--coco-json",
                        str(out / "labels.coco.json"), "--images-dir", str(imgs), "--pred-dir", str(preds),
                        "--out-dir", str(pkg2), "--no-zip", "--part-size", "0"], check=True, capture_output=True)
        assert_round_trip(self, self.pkg, self.edits, pkg2, conflicts, rows)

    def test_7_rerun_same_hash_no_duplicate_rows(self):
        _, o1, _ = run(self.base, self.pkg, self.edits, self.out_root, stamp="20990101_0000")
        _, o2, _ = run(self.base, self.pkg, self.edits, self.out_root, stamp="20990101_0001")
        h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        self.assertEqual(h(o1 / "labels.coco.json"), h(o2 / "labels.coco.json"))
        r1, r2 = read_out(o1)[1], read_out(o2)[1]
        strip = lambda rows: [{k: v for k, v in r.items() if k != "label_version"} for r in rows]
        self.assertEqual(strip(r1), strip(r2))
        self.assertEqual(h(o1 / "conflicts.csv"), h(o2 / "conflicts.csv"))
        keys = [(r["image"], r["dec_key"], r["op"], r["ann_old"], r["ann_new"]) for r in r1 if r["dec_key"] != "gt"]
        self.assertEqual(len(keys), len(set(keys)))
        with self.assertRaises(SystemExit):  # the same version id again: never overwritten
            run(self.base, self.pkg, self.edits, self.out_root, stamp="20990101_0000")
        self.assertEqual(h(o1 / "labels.coco.json"), h(o2 / "labels.coco.json"))

    def test_8_guards(self):
        for bad in (Path.home() / "Box" / "x", Path.home() / "Desktop" / "Cryo-image-processing" / "labels",
                    Path(str(Path.home() / "BOX").upper()) / "y"):
            with self.assertRaises(SystemExit) as cm:
                run(self.base, self.pkg, self.edits, bad)
            self.assertIn("read-only", str(cm.exception))
        self.assertIsNone(ar.out_root_problem(Path.home() / "Boxes" / "x"))  # a sibling name is fine
        _, out, log = run(self.base, self.pkg, self.edits, self.out_root, extra=["--dry-run"])
        self.assertFalse(self.out_root.exists())
        self.assertIn("[dry-run]", log)
        path = self.edits / "a.png.json"
        good = path.read_text(encoding="utf-8")
        rec = json.loads(good)
        rec["pkg"] = "other_package"
        path.write_text(json.dumps(rec), encoding="utf-8")
        with self.assertRaises(SystemExit) as cm:  # a record keyed against another package
            run(self.base, self.pkg, self.edits, self.out_root)
        self.assertIn("other_package", str(cm.exception))
        path.write_text(good, encoding="utf-8")
        base = json.loads(self.base.read_text(encoding="utf-8"))
        base["annotations"][0]["category_id"] = 3  # a different base than the package was built from
        self.base.write_text(json.dumps(base), encoding="utf-8")
        with self.assertRaises(SystemExit) as cm:
            run(self.base, self.pkg, self.edits, self.out_root)
        self.assertIn("not built from this base", str(cm.exception))
        self.assertFalse(self.out_root.exists())


class QcFixTests(unittest.TestCase):
    """QC round 1: numeric outline equality, base drift, path-guard bypasses, duplicate rules, minor cases."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ar_qc_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out_root = self.tmp / "labels"

    def build(self, extra=()):
        self.base, self.pkg, self.edits = build_synthetic(self.tmp / "in", extra)

    def record(self, file, dec, **edit):
        rec = {"file": file, "edit": {"dec": dec, "t": 1000, "modified": True, "reviewed": False, "reviewedT": 0, **edit}}
        (self.edits / f"{file}.json").write_text(json.dumps(rec), encoding="utf-8")

    def only(self, *files):
        for f in self.edits.glob("*.json"):
            if f.name[:-5] not in files:
                f.unlink()

    def test_zero_vs_zero_point_zero_is_the_same_outline(self):
        self.assertTrue(ar._same([[0, 0.5, 1, 0.5, 1, 1]], [[0.0, 0.5, 1.0, 0.5, 1.0, 1.0]]))
        self.assertFalse(ar._same([[0, 0.5, 1, 0.5, 1, 1]], [[0.0, 0.5, 1.0, 0.5, 1.0, 0.9]]))
        self.assertFalse(ar._same([[0, 0.5, 1, 0.5, 1, 1]], [[0, 0.5, 1, 0.5, 1, 1, 0, 1]]))
        spec = __import__("importlib.util").util.spec_from_file_location("rre", REPO / "annotation" / "resolve_review_edits.py")
        rre = __import__("importlib.util").util.module_from_spec(spec)
        spec.loader.exec_module(rre)
        self.assertTrue(rre.same_poly([[0, 0.5, 1, 0.5, 1, 1]], [[0.0, 0.5, 1.0, 0.5, 1.0, 1.0]]))
        self.assertFalse(rre.same_poly([[0, 0.5, 1, 0.5, 1, 1]], [[0.0, 0.5, 1.0, 0.5, 1.0, 0.9]]))
        # end to end: a keep whose outline is the package's written as ints is a keep, with no segmentation change
        self.build([(10, 3, 1, [[0.0, 0.0, 80.0, 0.0, 80.0, 80.0, 0.0, 80.0]])])
        self.only("c.png")
        self.record("c.png", {"g10": {"v": "keep:EV", "p": [[0, 0, 0.08, 0, 0.08, 0.1, 0, 0.1]], "t": 1000, "who": "Puj"}})
        _, out, _ = run(self.base, self.pkg, self.edits, self.out_root)
        coco, rows, _, _ = read_out(out)
        self.assertEqual([(r["dec_key"], r["op"], r["ann_new"]) for r in rows], [("g10", "keep", 10)])
        self.assertEqual(next(a for a in coco["annotations"] if a["id"] == 10)["segmentation"],
                         [[0.0, 0.0, 80.0, 0.0, 80.0, 80.0, 0.0, 80.0]])

    def test_swapped_outlines_and_foreign_source_refused(self):
        self.build()
        base = json.loads(self.base.read_text(encoding="utf-8"))
        a1, a6 = (next(a for a in base["annotations"] if a["id"] == i) for i in (1, 6))  # both EV, same image
        a1["segmentation"], a6["segmentation"] = a6["segmentation"], a1["segmentation"]
        swapped = self.tmp / "swapped" / "current.coco.json"
        swapped.parent.mkdir()
        swapped.write_text(json.dumps(base), encoding="utf-8")
        with self.assertRaises(SystemExit) as cm:
            run(swapped, self.pkg, self.edits, self.out_root)
        self.assertIn("not built from this base", str(cm.exception))
        self.assertIn("g1", str(cm.exception))
        base["annotations"] = [a for a in base["annotations"] if a["id"] != 7]  # a missing id
        swapped.write_text(json.dumps(base), encoding="utf-8")
        with self.assertRaises(SystemExit) as cm:
            run(swapped, self.pkg, self.edits, self.out_root)
        self.assertIn("label ids differ", str(cm.exception))
        data = ar.read_data_js(self.pkg / "data.js")
        other = self.tmp / "other.coco.json"
        other.write_text("{}", encoding="utf-8")
        for src, ok in ((str(other), False), (str(self.base), True), (str(self.base).upper(), True)):
            data["source"] = src
            (self.pkg / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
            if ok:
                v = run(self.base, self.pkg, self.edits, self.out_root, extra=["--dry-run"])[0]
                self.assertEqual(v["source_check"], "same path")
            else:
                with self.assertRaises(SystemExit) as cm:
                    run(self.base, self.pkg, self.edits, self.out_root)
                self.assertIn("not --base", str(cm.exception))
        self.assertFalse(self.out_root.exists())

    def test_duplicate_file_names_refused(self):
        self.build()
        base = json.loads(self.base.read_text(encoding="utf-8"))
        base["images"].append({**base["images"][2], "id": 4})
        self.base.write_text(json.dumps(base), encoding="utf-8")
        with self.assertRaises(SystemExit) as cm:
            run(self.base, self.pkg, self.edits, self.out_root)
        self.assertIn("duplicate base file_name", str(cm.exception))
        self.build()
        data = ar.read_data_js(self.pkg / "data.js")
        data["images"].append(data["images"][0])
        (self.pkg / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as cm:
            run(self.base, self.pkg, self.edits, self.out_root)
        self.assertIn("duplicate package file", str(cm.exception))

    def test_duplicate_rules(self):
        """Same class, two changes: keep the earliest (t, then key order), drop the other. Different classes: both
        conflicts. A near copy over another class's label: applied with a warning. Base pairs: not flagged."""
        self.build([(10, 3, 1, [sq(500, 504, 80)]), (11, 3, 1, [sq(501, 504, 80)])])  # a base duplicate pair
        self.only("c.png")
        n = lambda r: [norm(x) for x in r]
        self.record("c.png", {
            "nlate": {"v": "new:EV", "p": n([sq(96, 96, 80)]), "t": 900, "who": "Puj"},
            "nearly": {"v": "new:EV", "p": n([sq(96, 96, 80)]), "t": 800, "who": "Puj"},
            "nev": {"v": "new:EV", "p": n([sq(296, 96, 80)]), "t": 800, "who": "Puj"},
            "nmv": {"v": "new:MV", "p": n([sq(296, 96, 80)]), "t": 900, "who": "Puj"},
            "nnear": {"v": "new:MV", "p": n([sq(520, 504, 80)]), "t": 800, "who": "Puj"},
            "g10": {"v": "keep:MV", "t": 800, "who": "Puj"}})
        v, out, _ = run(self.base, self.pkg, self.edits, self.out_root)
        coco, rows, conflicts, ver = read_out(out)
        st = {r["dec_key"]: (r["op"], r["status"]) for r in rows}
        self.assertEqual(st, {"nlate": ("new", "duplicate-dropped"), "nearly": ("new", "applied"),
                              "nev": ("new", "conflict"), "nmv": ("new", "conflict"),
                              "nnear": ("new", "applied"), "g10": ("reclass", "applied")})
        sev = {c["dec_key"]: c["severity"] for c in conflicts}
        self.assertEqual(sev, {"nlate": "dropped", "nev": "conflict", "nmv": "conflict", "nnear": "warn"})
        self.assertIn("near_duplicate", next(c["reason"] for c in conflicts if c["dec_key"] == "nnear"))
        self.assertIn("kept nearly", next(c["reason"] for c in conflicts if c["dec_key"] == "nlate"))
        self.assertEqual((ver["n_conflicts"], ver["n_dropped"], ver["n_warnings"]), (2, 1, 1))
        self.assertEqual([(d["image"], d["ids"]) for d in ver["base_duplicates"]], [("c_png", [10, 11])])
        self.assertEqual(sum(a["image_id"] == 3 for a in coco["annotations"]), 2 + 2)  # 10, 11, nearly, nnear

    def test_minor_cases(self):
        self.build()
        self.only("a.png", "b.png")
        # p-1: Python's pr[-1] would be the last prediction; the page reads it as out of range, and so do we
        self.record("a.png", {"p-1": {"v": "add:EV", "t": 1000, "who": "Puj"}})
        # an old list item from an accepted prediction, whose decision was then undone: one add, not add + undone
        self.record("b.png", {"p0": {"v": None, "t": 1000, "who": "Puj"}}, gtT=1,
                    gt=[{"id": 8, "c": "EV", "p": [norm(sq(10, 16, 80))]}, {"id": 9, "c": "EV", "p": [norm(sq(200, 16, 80))]},
                        {"id": None, "c": "EV", "p": [norm(sq(400, 400, 40))], "src": "p0"}])
        self.assertEqual(len(rs.effective_labels(ar.read_data_js(self.pkg / "data.js")["images"][0],
                                                 json.loads((self.edits / "a.png.json").read_text())["edit"])), 7 + 1)  # 7 labels + pr[-1]
        v, out, _ = run(self.base, self.pkg, self.edits, self.out_root)
        coco, rows, conflicts, ver = read_out(out)
        self.assertEqual([(r["image"], r["dec_key"], r["op"], r["status"], r["reason"]) for r in rows],
                         [("a_png", "p-1", "add", "conflict", "pred_index_out_of_range"),
                          ("b_png", "p0", "add", "applied", None)])
        self.assertEqual(ver["skipped"], {})

    @unittest.skipUnless(os.name == "nt", "Windows path spellings")
    def test_guard_bypasses(self):
        sys.path.insert(0, str(REPO))
        from data_utils import path_guard
        from inference import inference_check as ic
        h = str(Path.home())
        tiles = Path(tempfile.gettempdir()) / "tiles"
        bad = [h + "\\Box\\x", h + "\\Box \\x", h + "\\Box.\\x", h + "\\Box. .\\x", h + "\\Box \\", h + "/Box/x",
               h.lower() + "\\BOX\\x", "\\\\?\\" + h + "\\Box\\x", "\\\\?\\" + h + "\\Box \\x",
               "\\\\localhost\\C$" + h[2:] + "\\Box\\x", "\\\\127.0.0.1\\c$" + h[2:] + "\\Box\\x",
               "\\\\?\\UNC\\localhost\\C$" + h[2:] + "\\Box\\x", "\\\\.\\C:" + h[2:] + "\\Box\\x",
               h + "\\Box\\..\\Box\\x", h + "\\Desktop\\Cryo-image-processing \\x",
               h + "\\Desktop\\Cryo-image-processing\\20261027\\a"]
        for p in bad:
            self.assertIsNotNone(ar.out_root_problem(Path(p)), p)
            self.assertIsNotNone(ar.out_root_problem(p), p)
            self.assertIsNotNone(ic.out_root_problem(Path(p), tiles), p)
        for ok in (h + "\\Boxes\\x", h + "\\Box2", str(self.tmp / "labels")):
            self.assertIsNone(ar.out_root_problem(Path(ok)), ok)
            self.assertIsNone(ic.out_root_problem(Path(ok), tiles), ok)
        # a junction into a protected folder is caught by realpath / samefile
        prot, link = self.tmp / "prot", self.tmp / "link"
        prot.mkdir()
        try:
            import _winapi
            _winapi.CreateJunction(str(prot), str(link))
        except (ImportError, OSError) as e:
            self.skipTest(f"no junctions here: {e}")
        self.addCleanup(os.rmdir, link)
        self.assertIsNotNone(path_guard.protected_problem(link / "x" / "y", [prot]))
        self.assertIsNotNone(path_guard.protected_problem(link, [prot]))
        self.assertIsNone(path_guard.protected_problem(self.tmp / "elsewhere", [prot]))
        self.assertIsNotNone(ic.out_root_problem(link / "x", tiles, protected=[prot]))
        # end to end: nothing is created
        self.build()
        for p in ("\\\\localhost\\C$" + h[2:] + "\\Box\\_ar_must_not_exist", h + "\\Box \\_ar_must_not_exist"):
            with self.assertRaises(SystemExit):
                run(self.base, self.pkg, self.edits, p)
        self.assertFalse((Path.home() / "Box" / "_ar_must_not_exist").exists())


def assert_round_trip(tc, pkg, edits, pkg2, conflicts, rows):
    """data.js rebuilt from the output COCO (no edits) == effective_labels() of the applied edits, minus the
    conflicts, within 1 px. Base labels by id; added ones paired by class and position."""
    old = {im["file"]: im for im in ar.read_data_js(pkg / "data.js")["images"]}
    new = {im["file"]: im for im in ar.read_data_js(pkg2 / "data.js")["images"]}
    recs = {r["file"]: r for _, r in ar.load_records(edits)}
    bad = {(c["file"], c["dec_key"]) for c in conflicts}
    n_checked = 0
    for file, im in old.items():
        pkg_gt = {g["id"]: g for g in im["gt"]}
        eff = rs.effective_labels(im, recs[file]["edit"]) if file in recs else im["gt"]
        want = []
        for g in eff:  # a conflict writes nothing: an added object is dropped, a base label stays as packaged
            key = g.get("src") or (f"g{g['id']}" if g.get("id") is not None else "gt")
            if (file, key) not in bad and (g.get("id") is None or (file, "gt") not in bad):
                want.append(g)
            elif g.get("id") in pkg_gt:
                want.append(pkg_gt[g["id"]])
        got = new[file]["gt"]
        tc.assertEqual(len(got), len(want), file)
        w, h = im["w"], im["h"]
        dist = lambda a, b: max(abs(x - y) * (w if k % 2 == 0 else h) for ra, rb in zip(a["p"], b["p"]) for k, (x, y) in enumerate(zip(ra, rb))) \
            if len(a["p"]) == len(b["p"]) and all(len(ra) == len(rb) for ra, rb in zip(a["p"], b["p"])) else 1e9
        got_by_id = {g["id"]: g for g in got}
        free = [g for g in got if g["id"] not in {x.get("id") for x in want if x.get("id") is not None}]
        for g in want:
            if g.get("id") is not None:
                o = got_by_id[g["id"]]
            else:
                o = min((x for x in free if x["c"] == g["c"]), key=lambda x: dist(x, g))
                free.remove(o)
            tc.assertEqual(o["c"], g["c"], (file, g.get("id")))
            tc.assertLessEqual(dist(o, g), 1.0, (file, g.get("id"), g.get("src")))
            n_checked += 1
    tc.assertGreater(n_checked, 0)


# ---------------------------------------------------------------- real inputs (copied, read-only)

def copy_package(src: Path, dst: Path, edits: Path = None):
    """data.js + edits to a temp folder. Long real file names can pass MAX_PATH under a deep temp root: skip then."""
    dst.mkdir(parents=True)
    try:
        shutil.copy(src / "data.js", dst / "data.js")
        shutil.copytree(edits or src / "edits", dst / "edits")
    except OSError as e:
        raise unittest.SkipTest(f"copying the real package to {dst} failed (path too long?): {e}")
    return dst


@unittest.skipUnless(have(REAL_BASE, REAL_V2, RESOLUTION), "real v2 package / base / resolution.csv not found")
class RealV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="apply_v2_"))
        cls.before = md5_tree(REAL_V2 / "edits")
        cls.pkg = copy_package(REAL_V2, cls.tmp / "pkg")
        shutil.copytree(REAL_BASE.parent, cls.tmp / REAL_BASE.parent.name, ignore=shutil.ignore_patterns("*.zip"))
        cls.base = cls.tmp / REAL_BASE.parent.name / REAL_BASE.name
        _, cls.out, cls.log = run(cls.base, cls.pkg, cls.pkg / "edits", cls.tmp / "labels")
        cls.coco, cls.rows, cls.conflicts, cls.ver = read_out(cls.out)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_2_reproduces_resolution_applied_ready(self):
        """resolution.csv lists 4 'reshape' rows (labels 606, 635, 636, 2966) whose old and new outlines are equal
        numerically: the seeds wrote 0 where data.js has 0.0 (edge-touching vertices), and the old json.dumps
        comparison called that a reshape. With a numeric comparison they are keep:<same class> + an unchanged
        outline, so they are logged as keep and their segmentation is not rewritten."""
        with open(RESOLUTION, newline="", encoding="utf-8") as f:
            ready = [r for r in csv.DictReader(f) if r["status"] == "applied-ready"]
        want = Counter(r["op"] for r in ready)
        self.assertEqual(want, Counter({"add": 95, "reclass": 18, "delete": 5, "reshape": 4}))
        fake = [r for r in ready if r["op"] == "reshape"]
        self.assertTrue(all(ar._same(json.loads(r["old"]), json.loads(r["new"])) and r["from"] == r["to"] for r in fake))
        self.assertTrue(all(r["old"] != r["new"] for r in fake))  # differ as text only
        self.assertEqual(sorted(int(r["cur_id"]) for r in fake), [606, 635, 636, 2966])
        c = self.ver["counts_by_op"]
        self.assertEqual((c["add"] + c["new"], c["reclass"], c["delete"], c["reshape"], c["reclass+reshape"], c["keep"]),
                         (95, 18, 5, 0, 0, 4))
        self.assertEqual((c["reject"], self.ver["n_conflicts"], self.ver["n_dropped"], self.ver["n_warnings"]), (0, 0, 0, 0))
        base = {a["id"]: a for a in json.loads(self.base.read_text(encoding="utf-8"))["annotations"]}
        out = {a["id"]: a for a in self.coco["annotations"]}
        for r in fake:  # not rewritten: the base segmentation, byte for byte
            self.assertEqual(json.dumps(out[int(r["cur_id"])]["segmentation"]), json.dumps(base[int(r["cur_id"])]["segmentation"]))
        mine = {(r["image"], r["dec_key"]): r for r in self.rows}
        self.assertEqual(len(mine), len(self.rows))
        self.assertEqual(len(self.rows), len(ready))
        for r in ready:
            m = mine[(r["target_stem"], r["seed_key"])]
            op = {"add": "new" if r["seed_key"].startswith("n") else "add", "reshape": "keep"}.get(r["op"], r["op"])
            self.assertEqual((m["op"], m["cls_from"] or "", m["cls_to"] or ""), (op, r["from"], r["to"]), r["seed_key"])
            self.assertEqual(m["ann_old"], int(r["cur_id"]) if r["cur_id"] else None)
            self.assertTrue(m["seed"].startswith(f"september:{r['file']}:{r['key']}"), m["seed"])
            self.assertEqual(m["status"], "applied")
            self.assertEqual(m["who"], "pujan")
        self.assertEqual(len(self.coco["annotations"]), 3843 + 95 - 5)
        new_ids = sorted(r["ann_new"] for r in self.rows if r["op"] in ("add", "new"))
        self.assertEqual(new_ids, list(range(3844, 3844 + 95)))
        self.assertEqual(md5_tree(REAL_V2 / "edits"), self.before)

    def test_5_round_trip_real(self):
        pkg2 = self.tmp / "pkg2"
        subprocess.run([sys.executable, str(REPO / "annotation" / "build_review_package.py"), "--coco-json",
                        str(self.out / "labels.coco.json"), "--images-dir", str(REAL_V2 / "images"),
                        "--pred-dir", str(self.tmp / "nopreds"), "--out-dir", str(pkg2), "--no-zip", "--part-size", "0"],
                       check=True, capture_output=True)
        assert_round_trip(self, self.pkg, self.pkg / "edits", pkg2, self.conflicts, self.rows)

    def test_1_zero_edits_real(self):
        empty = self.tmp / "empty"
        empty.mkdir(exist_ok=True)
        _, out, _ = run(self.base, self.pkg, empty, self.tmp / "labels0")
        coco = read_out(out)[0]
        base = json.loads(self.base.read_text(encoding="utf-8"))
        self.assertEqual([key_of(a) for a in coco["annotations"]], [key_of(a) for a in base["annotations"]])


@unittest.skipUnless(have(REAL_BASE, REAL_V1, SMOKE_EDITS), "v1 sandbox package / smoke edits not found")
class SmokeFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="apply_smoke_"))
        cls.before = md5_tree(SMOKE_EDITS)
        cls.pkg = copy_package(REAL_V1, cls.tmp / "pkg", edits=SMOKE_EDITS)
        _, cls.out, cls.log = run(REAL_BASE, cls.pkg, cls.pkg / "edits", cls.tmp / "labels")
        cls.coco, cls.rows, cls.conflicts, cls.ver = read_out(cls.out)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_4_mda_duplicate_in_conflicts_not_labels(self):
        dup = [c for c in self.conflicts if MDA in c["image"] and c["reason"].startswith("duplicate_outline")]
        self.assertGreaterEqual(len(dup), 2)
        self.assertTrue(any("different class" in c["reason"] for c in dup))
        self.assertEqual({c["cls_to"] for c in dup}, {"EV", "Low conf EV"})
        # nothing written for them: that image's output = base + applied adds/new - applied deletes
        b = next(i for i in self.coco["images"] if MDA in i["extra"]["name"])
        base = json.loads(REAL_BASE.read_text(encoding="utf-8"))
        n_base = sum(1 for a in base["annotations"] if a["image_id"] == b["id"])
        mine = [r for r in self.rows if MDA in r["image"] and r["status"] == "applied"]
        n_out = sum(1 for a in self.coco["annotations"] if a["image_id"] == b["id"])
        self.assertEqual(n_out, n_base + sum(r["op"] in ("add", "new") for r in mine) - sum(r["op"] == "delete" for r in mine))
        self.assertEqual(md5_tree(SMOKE_EDITS), self.before)

    def test_smoke_rows_well_formed(self):
        self.assertEqual(len(self.rows), sum(self.ver["counts_by_op"].values()) + self.ver["n_conflicts"])
        self.assertTrue(all(r["who_auth"] is None and r["schema"] == "label_change/1" for r in self.rows))
        self.assertTrue(any(r["dec_key"] == "gt" and r["op"] == "new" and r["seed"] for r in self.rows))


# ---------------------------------------------------------------- 6: the page's effGt() vs effective_labels()

sys.path.insert(0, str(HERE))
try:
    from test_review_tool_browser import launch, norm as js_norm, sync_playwright  # noqa: E402
    from test_review_server import LiveServer  # noqa: E402
    HARNESS = True
except Exception:  # noqa: BLE001  (the harness could not be imported)
    HARNESS = False


@unittest.skipUnless(HARNESS and have(REAL_V2), "Playwright harness or real packages not found")
class JsParityTests(unittest.TestCase):
    """effGt() in review_tool/index.html == review_server.effective_labels() on every real record (v2 seeds and the
    v1 smoke edits), served from temp copies by review_server.py in headless Edge/Chrome."""

    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        cls.pw = sync_playwright().start()
        cls.browser = launch(cls.pw)
        if cls.browser is None:
            cls.pw.stop()
            raise unittest.SkipTest("no Edge / Chrome for Playwright")

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def check(self, src, edits):
        def build(pkg):
            copy_package(src, pkg, edits=edits)
            shutil.copy(REPO / "annotation" / "review_tool" / "index.html", pkg / "index.html")
        srv = LiveServer(build)
        self.addCleanup(srv.close)
        page = self.browser.new_page()
        self.addCleanup(page.close)
        page.goto(srv.url + "&who=tester&mode=editor")
        page.wait_for_function("window.__review && __review.state().server", timeout=20000)
        page.wait_for_function("__review.state().dirty === 0", timeout=15000)
        saved = srv.request("GET", "/api/edits")[1]
        self.assertEqual(len(saved), len(list(edits.glob("*.json"))))
        data = {im["file"]: im for im in ar.read_data_js(src / "data.js")["images"]}
        js = page.evaluate("files => Object.fromEntries(files.map(f => [f, __review.effGt(__review.D.images.find(im => im.file === f))]))",
                           list(saved))
        for f, rec in saved.items():
            self.assertEqual(js_norm(js[f]), js_norm(rs.effective_labels(data[f], rec)), f)
        return len(saved)

    def test_6_v2_seeds(self):
        self.assertEqual(self.check(REAL_V2, REAL_V2 / "edits"), 28)

    @unittest.skipUnless(have(REAL_V1, SMOKE_EDITS), "v1 sandbox / smoke edits not found")
    def test_6_v1_smoke_edits(self):
        self.assertEqual(self.check(REAL_V1, SMOKE_EDITS), 33)


if __name__ == "__main__":
    unittest.main()
