"""results.compute against the CryoEV golden fixtures (BUILD_CONTRACTS.md section 6) plus controls.

    python -m unittest discover -s results/tests        (from the repo root)

The fixtures and their hand-derived expectations are tests/fixtures/cj_golden (make_golden.py explains the
scene); they are read, never written: every test that changes an input works on a temp copy. The reviewer
columns added after QC (contract section 4 amendment) are not in the fixture files; their expected values
are derived by hand below (REVIEWER_*).
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import importlib  # noqa: E402

import numpy as np  # noqa: E402

from results import compute, write_results  # noqa: E402

C = importlib.import_module("results.compute")   # the module (results.compute is also the function's name)

GOLDEN = REPO / "tests" / "fixtures" / "cj_golden"
PS = GOLDEN / "predict" / "ps_20261105_abc123"
REVIEWED = GOLDEN / "reviewed.json"
MODEL = "v5_evnonev_y8m_20261029"

# Hand-derived reviewer columns. No reviewed image -> all zero, pct null. Reviewed scenario: only 0002 is
# reviewed; its outlines F (cls 0, reviewer MV), H (cls 1, nonEV), I (cls 1, nonEV), J (cls 0, EV, inside I).
# Geometry EVs there: F and J (J is not inside another EV-class outline) -> ev_reviewed_geometry 2.
# Reviewer: EV+MV calls = F, J -> 2; MV = F -> 1; inner 0; 1/2 = 50 %. Disagree: F is MV by the reviewer but
# geometry-standalone -> 1 (J: EV vs standalone agrees; H and I are nonEV calls on class-1 outlines: agree,
# and reviewer_nonev counts only nonEV calls on EV-class outlines -> 0).
REVIEWER_NONE = {"reviewed_images": 0, "ev_reviewed_geometry": 0, "reviewer_ev": 0, "reviewer_mv": 0,
                 "reviewer_inner": 0, "reviewer_nonev": 0, "pct_multilayer_reviewer": None,
                 "reviewer_geometry_disagree": 0}
REVIEWER_S1 = {"reviewed_images": 1, "ev_reviewed_geometry": 2, "reviewer_ev": 2, "reviewer_mv": 1,
               "reviewer_inner": 0, "reviewer_nonev": 0, "pct_multilayer_reviewer": 50.0,
               "reviewer_geometry_disagree": 1}
NM_RANGE_GOLDEN = {"nm_per_px_min": 1.4724, "nm_per_px_max": 1.4724}
NEW_FIELDS = {"n_checked", "n_unsure_open", "diam_hist"}   # round 6 (P2), results.json only / not per_sample.csv
GEOMETRY = ["n_images", "ev_total", "ev_per_image_mean", "diam_nm_median", "diam_nm_q1", "diam_nm_q3",
            "pct_multilayer_geometry", "inner_layers_excluded", "nonev_total"]


def load(name):
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


def expected(name):
    """A fixture expectation plus the hand-derived reviewer columns."""
    exp = load(name)
    for i, img in exp["images"].items():   # `edited` (added after QC): reviewed.json's flag, missing = reviewed
        img["edited"] = img["reviewed"]
        img.update(rejected=False, reject_reason=None, note=None)   # round 7 (R2): additive, nothing rejected
    for s, e in exp["samples"].items():
        e.update(REVIEWER_S1 if (name == "expected_results_reviewed.json" and s == "S1") else REVIEWER_NONE)
        e.update(n_rejected=0, n_low_conf=0)
        e.update(NM_RANGE_GOLDEN)   # contract section 8: every golden image is 1.4724 nm/px (images.csv)
    return exp


def assert_matches(tc, got, exp):
    """Exact, except diam_nm_* within the fixture's relative tolerance; no keys missing or extra."""
    exp = {k: v for k, v in exp.items() if not k.startswith("_")}
    tol = 0.02
    tc.assertEqual(set(got) - {"computed"}, set(exp))
    datetime.fromisoformat(got["computed"])
    for k in ("schema", "prediction_set", "model", "rule", "refused"):
        tc.assertEqual(got[k], exp[k], k)
    tc.assertEqual(got["images"], exp["images"])
    tc.assertEqual(set(got["samples"]), set(exp["samples"]))
    for s, e in exp["samples"].items():
        g = got["samples"][s]
        tc.assertEqual(NEW_FIELDS - set(g), set(), s)           # additive round-6 fields: present, tested in test_p2
        tc.assertEqual(sum(g["diam_hist"]["counts"]), g["ev_total"], s)
        tc.assertEqual(set(g) - NEW_FIELDS, set(e), s)
        for f, v in e.items():
            if f.startswith("diam_nm_"):
                tc.assertLess(abs(g[f] / v - 1), tol, f"{s} {f}: {g[f]} vs {v}")
            else:
                tc.assertEqual(g[f], v, f"{s} {f}")


def without_time(res):
    return {k: v for k, v in res.items() if k != "computed"}


def diameter_pairs(ps_dir):
    """Per image: (analysis.morphology.analyze_instances equivalent diameters, results' _diameter) for the same
    EV masks at conf >= 0.5 from the prediction files. Used by the golden and known-answer tests."""
    from analysis.morphology import analyze_instances
    from inference.model_classes import ev_class_ids
    load_txt, _, _, cm = C._chain()
    ps_dir = Path(ps_dir)
    model = json.loads((ps_dir / "request.json").read_text(encoding="utf-8"))["model"]
    ev_ids = ev_class_ids(model)
    for r in C._read_images_csv(ps_dir):
        w, h = C._image_size(ps_dir, r["file"])
        nm = C._nm_per_px(r)
        objs = C._objects_from_predictions(ps_dir / "p" / model / "predictions", r["image"], w, h, load_txt)
        masks = [o.mask for o in objs if ev_ids is None or o.cls in ev_ids]
        theirs = [rec["equivalent_diameter"] for rec in analyze_instances(masks, pixel_size=nm)]
        mine = [d for d in (C._diameter(m, nm, cm) for m in masks) if d is not None]
        yield r["image"], theirs, mine


class TempPS:
    """A writable copy of the golden prediction set."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ps = Path(self._tmp.name) / PS.name
        shutil.copytree(PS, self.ps)
        return self

    def __exit__(self, *exc):
        self._tmp.cleanup()

    def edit_images_csv(self, image, **values):
        path = self.ps / "images.csv"
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            if r["image"] == image:
                r.update(values)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def txt_objects(image, min_conf=0.0):
    """reviewed.json-style objects (pixel polygons, no reviewer label) from a golden prediction file."""
    with Image.open(PS / "images" / f"{image}.png") as im:
        w, h = im.size
    out = []
    for line in (PS / "p" / MODEL / "predictions" / f"{image}.txt").read_text(encoding="utf-8").splitlines():
        p = line.split()
        xy = [float(v) for v in p[2:]]
        if float(p[1]) >= min_conf:
            out.append({"cls": int(p[0]), "conf": float(p[1]), "source": "model", "reviewer_class": None,
                        "poly": [[xy[i] * w, xy[i + 1] * h] for i in range(0, len(xy), 2)]})
    return out


class Golden(unittest.TestCase):
    def test_noedit_equals_expected(self):
        assert_matches(self, compute(PS), expected("expected_results_noedit.json"))

    def test_reviewed_equals_expected(self):
        assert_matches(self, compute(PS, REVIEWED), expected("expected_results_reviewed.json"))

    def test_reviewed_dict_same_as_path(self):
        self.assertEqual(without_time(compute(PS, load("reviewed.json"))), without_time(compute(PS, REVIEWED)))

    def test_unreviewed_images_come_from_predictions(self):
        """Only 0002 reviewed: 0001 and 0003 equal the no-edit run exactly, even when reviewed.json's copies of
        their outlines are wrong (emptied here), because unreviewed images are read from the .txt files."""
        rv = load("reviewed.json")
        rv["images"]["0001"]["objects"] = []
        rv["images"]["0003"]["objects"] = []
        got, base = compute(PS, rv), compute(PS)
        for i in ("0001", "0003"):
            self.assertEqual(got["images"][i], base["images"][i])
        self.assertEqual(got["samples"]["S2"], base["samples"]["S2"])

    def test_edited_not_reviewed_moves_geometry_only(self):
        """0002 edited (G deleted, F/J labelled) but Reviewed never pressed: its outlines come from reviewed.json,
        so geometry equals the reviewed scenario, but it is not a reviewed image (n_reviewed 0, reviewer columns
        empty), and the reviewer labels on it are not counted."""
        rv = load("reviewed.json")
        rv["images"]["0002"].update(reviewed=False, edited=True)
        got, as_reviewed = compute(PS, rv), compute(PS, REVIEWED)
        self.assertEqual(got["images"]["0002"], dict(as_reviewed["images"]["0002"], reviewed=False))
        self.assertEqual((got["images"]["0002"]["edited"], got["images"]["0001"]["edited"]), (True, False))
        s, r = got["samples"]["S1"], as_reviewed["samples"]["S1"]
        self.assertEqual({f: s[f] for f in GEOMETRY}, {f: r[f] for f in GEOMETRY})
        self.assertEqual(s["n_reviewed"], 0)
        self.assertEqual({f: s[f] for f in REVIEWER_NONE}, REVIEWER_NONE)

    def test_not_edited_not_reviewed_reads_predictions(self):
        """`edited: false, reviewed: false` -> the .txt, whatever reviewed.json lists for that image."""
        rv = load("reviewed.json")
        rv["images"]["0002"].update(reviewed=False, edited=False)   # its listed objects lack G
        got, base = compute(PS, rv), compute(PS)
        self.assertEqual(got["images"], base["images"])
        self.assertEqual(without_time(got)["samples"], base["samples"])

    def test_reviewed_polygons_reproduce_prediction_geometry(self):
        """Every image 'reviewed' with exactly its conf >= 0.5 predictions as pixel polygons and no reviewer
        labels: the geometry numbers equal the no-edit run exactly, diameters too (same rasterization)."""
        rv = {"schema": "reviewed/1", "package": "pred_x", "prediction_set": PS.name, "model": MODEL,
              "images": {i: {"reviewed": True, "reviewed_by": ["t"], "objects": txt_objects(i, 0.5)}
                         for i in ("0001", "0002", "0003")}}
        got, base = compute(PS, rv), compute(PS)
        for i, img in base["images"].items():
            self.assertEqual(got["images"][i], dict(img, reviewed=True, edited=True))
        for s, smp in base["samples"].items():
            self.assertEqual({f: got["samples"][s][f] for f in GEOMETRY}, {f: smp[f] for f in GEOMETRY})
            self.assertEqual(got["samples"][s]["ev_reviewed_geometry"], smp["ev_total"])
            self.assertIsNone(got["samples"][s]["pct_multilayer_reviewer"])

    def test_diameter_equals_analyze_instances(self):
        n = 0
        for image, theirs, mine in diameter_pairs(PS):
            self.assertEqual(mine, theirs, image)
            n += len(mine)
        self.assertEqual(n, 9)   # EV-class outlines at conf >= 0.5: A B C | F G J | K L N (D is below 0.5)
        cm = C._chain()[3]
        empty, dot = np.zeros((5, 5), bool), np.zeros((5, 5), bool)
        dot[2, 2] = True   # one pixel: zero-length contour, analyze_instances drops it
        self.assertEqual((C._diameter(empty, 1.0, cm), C._diameter(dot, 1.0, cm)), (None, None))
        from analysis.morphology import analyze_instances
        self.assertEqual(analyze_instances([empty, dot], pixel_size=1.0), [])


class ClassBeforeRoles(unittest.TestCase):
    def test_ev_inside_nonev_outline_is_standalone(self):
        """J (class 0) sits inside I (class 1): the class-0 filter runs before roles, so J is an EV."""
        img = compute(PS)["images"]["0002"]
        self.assertEqual((img["ev"], img["multilayer"], img["inner"], img["nonev"]), (3, 0, 0, 2))

    def test_control_containment_sees_j_inside_i(self):
        """Control with a known answer: give I the EV class (a reviewed redraw) and the same geometry must call I
        multilayer and J its inner layer. If this failed, the test above would prove nothing."""
        rv = load("reviewed.json")
        i_obj = self._largest(rv["images"]["0002"]["objects"])   # I is r45
        self.assertEqual(i_obj["cls"], 1)
        i_obj["cls"] = 0
        img = compute(PS, rv)["images"]["0002"]
        self.assertEqual((img["ev"], img["multilayer"], img["inner"], img["nonev"]), (2, 1, 1, 1))

    @staticmethod
    def _largest(objs):
        return max(objs, key=lambda o: max(x for x, _ in o["poly"]) - min(x for x, _ in o["poly"]))


class ReviewerLabels(unittest.TestCase):
    """Reviewer class labels move only the reviewer columns, never the geometry headline."""

    def relabel(self, labels):
        rv = load("reviewed.json")
        objs = rv["images"]["0002"]["objects"]
        by_old = {o["reviewer_class"]: o for o in objs if o["reviewer_class"] in ("MV", "EV")}   # F, J
        for old, new in labels.items():
            if old == "I":
                ClassBeforeRoles._largest(objs)["reviewer_class"] = new
            else:
                by_old[old]["reviewer_class"] = new
        return compute(PS, rv)

    def test_labels_do_not_move_geometry(self):
        base = compute(PS, REVIEWED)
        got = self.relabel({"MV": "inner", "EV": "nonEV"})   # F -> inner, J -> nonEV
        self.assertEqual(got["images"], base["images"])
        for f in GEOMETRY + ["ev_reviewed_geometry"]:
            self.assertEqual(got["samples"]["S1"][f], base["samples"]["S1"][f], f)
        s = got["samples"]["S1"]
        self.assertEqual((s["reviewer_ev"], s["reviewer_mv"], s["reviewer_inner"], s["reviewer_nonev"],
                          s["pct_multilayer_reviewer"], s["reviewer_geometry_disagree"]),
                         (0, 0, 1, 1, None, 2))   # F inner vs standalone; J nonEV but geometry EV

    def test_nonev_call_on_ev_outline_disagrees(self):
        """Hand-checked: J (class 0, geometry standalone EV) called nonEV -> reviewer_nonev 1; disagree = F (MV vs
        standalone) + J (nonEV vs EV) = 2; reviewer_ev = F only = 1, reviewer_mv 1 -> 100 %. Geometry unchanged."""
        got = self.relabel({"EV": "nonEV"})
        s = got["samples"]["S1"]
        self.assertEqual((s["reviewer_nonev"], s["reviewer_geometry_disagree"], s["reviewer_ev"],
                          s["pct_multilayer_reviewer"], s["ev_total"]), (1, 2, 1, 100.0, 4))

    def test_nonev_call_on_inner_layer_disagrees(self):
        """Reviewed 0001 (A contains B): B, a geometry inner layer, called nonEV -> nonev 1, disagree 1; A called MV
        agrees with multilayer."""
        rv = load("reviewed.json")
        e = rv["images"]["0001"]
        e["reviewed"] = True
        size = lambda o: max(x for x, _ in o["poly"]) - min(x for x, _ in o["poly"])   # noqa: E731
        ev_objs = sorted((o for o in e["objects"] if o["cls"] == 0), key=size)   # B r25, C r30, A r40
        ev_objs[0]["reviewer_class"], ev_objs[2]["reviewer_class"] = "nonEV", "MV"
        s = compute(PS, rv)["samples"]["S1"]
        self.assertEqual((s["reviewed_images"], s["reviewer_nonev"], s["reviewer_mv"], s["reviewer_geometry_disagree"],
                          s["inner_layers_excluded"]), (2, 1, 2, 2, 1))   # disagree: B here + F on 0002

    def test_call_on_nonev_outline_disagrees(self):
        s = self.relabel({"I": "EV"})["samples"]["S1"]   # I is class 1: no geometry role
        self.assertEqual((s["reviewer_ev"], s["reviewer_mv"], s["pct_multilayer_reviewer"],
                          s["reviewer_geometry_disagree"]), (3, 1, 33.3333, 2))
        self.assertEqual(s["ev_total"], 4)

    def test_unknown_reviewer_class_is_an_error(self):
        rv = load("reviewed.json")
        rv["images"]["0002"]["objects"][0]["reviewer_class"] = "vesicle"
        with self.assertRaises(ValueError):
            compute(PS, rv)


class Refusals(unittest.TestCase):
    def test_mixed_nm_per_px(self):
        with TempPS() as t:
            t.edit_images_csv("0002", nm_per_px="0.7362")
            res = compute(t.ps)
        self.assertIn("'S1'", res["refused"])
        self.assertIn("0.7362", res["refused"])
        self.assertIn("1.4724", res["refused"])
        self.assertNotIn("S1", res["samples"])
        self.assertEqual(res["samples"]["S2"]["ev_total"], 2)   # an unaffected sample is still reported
        self.assertEqual(res["images"]["0002"]["nm_per_px"], 0.7362)

    def test_mixed_tile_size(self):
        with TempPS() as t:
            Image.new("L", (800, 800), 150).save(t.ps / "images" / "0002.png")
            res = compute(t.ps)
        self.assertIn("400x400", res["refused"])
        self.assertIn("800x800", res["refused"])
        self.assertNotIn("S1", res["samples"])

    def test_unconfirmed_nm_per_px(self):
        with TempPS() as t:
            t.edit_images_csv("0003", nm_confirmed_by="")
            res = compute(t.ps)
        self.assertIn("'S2'", res["refused"])
        self.assertEqual(set(res["samples"]), {"S1"})

    def test_missing_nm_per_px(self):
        with TempPS() as t:
            t.edit_images_csv("0001", nm_per_px="")
            res = compute(t.ps)
        self.assertIn("0001", res["refused"])
        self.assertEqual(set(res["samples"]), {"S2"})

    def test_set_not_done(self):
        with TempPS() as t:
            st = json.loads((t.ps / "status.json").read_text(encoding="utf-8"))
            st.update(state="running", n_done=1)
            (t.ps / "status.json").write_text(json.dumps(st), encoding="utf-8")
            res = compute(t.ps)
        self.assertIn("running", res["refused"])
        self.assertEqual((res["samples"], res["images"]), ({}, {}))

    def test_missing_status_is_not_finished(self):
        with TempPS() as t:
            (t.ps / "status.json").unlink()
            res = compute(t.ps)
        self.assertIn("set not finished", res["refused"])
        self.assertEqual((res["samples"], res["images"]), ({}, {}))

    def test_sample_label_is_stripped(self):
        with TempPS() as t:
            t.edit_images_csv("0002", sample_label=" S1 ")
            res = compute(t.ps)
        self.assertIsNone(res["refused"])
        self.assertEqual(set(res["samples"]), {"S1", "S2"})
        self.assertEqual(res["samples"]["S1"]["n_images"], 2)


class PixelTolerance(unittest.TestCase):
    """Contract section 8: within one sample, nm/px spread (max - min) <= 1 % of the median is one scale; each
    image keeps its own nm/px. Spread > 1 % is still refused."""

    def test_spread_function(self):
        self.assertEqual(C.NM_TOLERANCE, 0.01)
        self.assertEqual(C.nm_spread([1.4724]), 0.0)                       # control: one value, no spread
        self.assertTrue(C.one_scale([1.4724, 1.4724, 1.4724]))
        self.assertFalse(C.one_scale([0.7362, 1.4724]))                   # control: 2x apart
        self.assertTrue(C.one_scale([1.000, 1.009]))                      # 0.009 / 1.0045 = 0.896 %
        self.assertFalse(C.one_scale([1.000, 1.011]))                     # 0.011 / 1.0055 = 1.094 %
        self.assertTrue(C.one_scale([1.47125, 1.4724]))                   # the real 20261027 liposome sample
        self.assertEqual(C.NM_EPS, 1e-12)
        self.assertGreater(C.nm_spread([1.0, 1.0, 1.01]), 0.01)           # exactly 1 %, 0.010000000000000009 in floats
        self.assertTrue(C.one_scale([1.0, 1.0, 1.01]))                    # ... accepted (NM_EPS)
        self.assertFalse(C.one_scale([1.0, 1.0, 1.0101]))                 # control: just over

    def test_0_9_percent_accepted_with_range(self):
        with TempPS() as t:
            t.edit_images_csv("0001", nm_per_px="1.000")
            t.edit_images_csv("0002", nm_per_px="1.009")
            res = compute(t.ps)
        self.assertIsNone(res["refused"])
        s = res["samples"]["S1"]
        self.assertEqual((s["nm_per_px_min"], s["nm_per_px_max"]), (1.0, 1.009))
        self.assertEqual((res["images"]["0001"]["nm_per_px"], res["images"]["0002"]["nm_per_px"]), (1.0, 1.009))
        self.assertEqual(s["ev_total"], 5)                                 # geometry counts don't depend on nm/px

    def test_1_1_percent_refused(self):
        with TempPS() as t:
            t.edit_images_csv("0001", nm_per_px="1.000")
            t.edit_images_csv("0002", nm_per_px="1.011")
            res = compute(t.ps)
        self.assertIn("'S1'", res["refused"])
        self.assertIn("1.011", res["refused"])
        self.assertIn("1 % tolerance", res["refused"])
        self.assertNotIn("S1", res["samples"])
        self.assertEqual(res["samples"]["S2"]["nm_per_px_min"], 1.4724)     # the other sample is still reported

    def test_each_image_uses_its_own_nm_per_px(self):
        """Known answer: two images of one sample with the SAME single EV outline (same pixel diameter d_px),
        nm/px 1.000 and 1.009 -> diameters d and 1.009 d (0.9 % apart). With two values, numpy's percentiles
        are d (1 + 0.009 p): Q1 = d (1 + 0.00225), median = d (1 + 0.0045), Q3 = d (1 + 0.00675)."""
        one = (PS / "p" / MODEL / "predictions" / "0003.txt").read_text(encoding="utf-8").splitlines()[0]
        self.assertTrue(one.startswith("0 "))                              # an EV-class outline, conf >= 0.5
        _, _, roles_for, cm = C._chain()
        load_txt = C._chain()[0]

        def scene(nm1, nm2):
            with TempPS() as t:
                shutil.copyfile(t.ps / "images" / "0001.png", t.ps / "images" / "0002.png")
                for image in ("0001", "0002"):
                    (t.ps / "p" / MODEL / "predictions" / f"{image}.txt").write_text(one + "\n", encoding="utf-8")
                t.edit_images_csv("0001", nm_per_px=nm1)
                t.edit_images_csv("0002", nm_per_px=nm2)
                per_image = {}
                for r in C._read_images_csv(t.ps):
                    if r["sample_label"] != "S1":
                        continue
                    w, h = C._image_size(t.ps, r["file"])
                    objs = C._objects_from_predictions(t.ps / "p" / MODEL / "predictions", r["image"], w, h,
                                                       load_txt)
                    per_image[r["image"]] = C._measure(objs, C._nm_per_px(r), {0}, roles_for, cm)["_diam"]
                return compute(t.ps), per_image

        base, px = scene("1.000", "1.000")
        self.assertEqual(px["0001"], px["0002"])                           # control: same outline, same diameter
        d = px["0001"][0]
        self.assertEqual(base["samples"]["S1"]["diam_nm_median"], round(d, 4))
        res, per = scene("1.000", "1.009")
        self.assertIsNone(res["refused"])
        self.assertEqual(per["0001"], [d])
        self.assertAlmostEqual(per["0002"][0] / per["0001"][0], 1.009, places=12)   # 0.9 % apart
        s = res["samples"]["S1"]
        self.assertEqual((s["ev_total"], s["n_images"]), (2, 2))
        for f, k in (("diam_nm_q1", 0.25), ("diam_nm_median", 0.5), ("diam_nm_q3", 0.75)):
            self.assertAlmostEqual(s[f], d * (1 + 0.009 * k), places=3, msg=f)


class InputErrors(unittest.TestCase):
    def test_missing_prediction_file(self):
        with TempPS() as t:
            (t.ps / "p" / MODEL / "predictions" / "0003.txt").unlink()
            with self.assertRaises(FileNotFoundError):
                compute(t.ps)

    def test_reviewed_for_another_model(self):
        rv = load("reviewed.json")
        rv["model"] = "v4_y8m_20260924"
        with self.assertRaises(ValueError):
            compute(PS, rv)

    def test_reviewed_missing_an_image(self):
        rv = load("reviewed.json")
        del rv["images"]["0003"]
        with self.assertRaises(ValueError):
            compute(PS, rv)


class Outputs(unittest.TestCase):
    def test_write_results(self):
        res = compute(PS, REVIEWED)
        with tempfile.TemporaryDirectory() as d:
            out = write_results(res, Path(d) / "results")
            self.assertEqual(json.loads((out / "results.json").read_text(encoding="utf-8")), res)
            with (out / "per_sample.csv").open(newline="", encoding="utf-8") as fh:
                rows = {r["sample_label"]: r for r in csv.DictReader(fh)}
            self.assertEqual(rows["S1"]["ev_total"], "4")
            self.assertEqual(rows["S1"]["pct_multilayer_reviewer"], "50.0")
            self.assertEqual(rows["S1"]["reviewer_geometry_disagree"], "1")
            self.assertEqual(rows["S2"]["pct_multilayer_reviewer"], "")
            with (out / "per_image.csv").open(newline="", encoding="utf-8") as fh:
                imgs = {r["image"]: r for r in csv.DictReader(fh)}
            self.assertEqual([imgs[i]["reviewed"] for i in ("0001", "0002", "0003")], ["false", "true", "false"])

    def test_write_order_and_stray_tmp(self):
        """results.json is written last (the commit marker); a stray *.tmp from an interrupted write is removed."""
        C = sys.modules["results.compute"]                 # the module (results.compute the name is the function)
        res = compute(PS)
        order, real = [], C._atomic_write

        def rec(path, text):
            order.append(Path(path).name)
            real(path, text)

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "results"
            out.mkdir()
            (out / "results.json.tmp").write_text("half", encoding="utf-8")
            (out / "per_image.csv.tmp").write_text("half", encoding="utf-8")
            C._atomic_write = rec
            try:
                write_results(res, out)
            finally:
                C._atomic_write = real
            # contract C5 (round 14): objects.csv, then steps.json, then results.json last
            self.assertEqual(order, ["per_image.csv", "per_sample.csv", "objects.csv", "steps.json", "results.json"])
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["objects.csv", "per_image.csv", "per_sample.csv", "results.json", "steps.json"])

    def test_cli(self):
        with TempPS() as t:
            r = subprocess.run([sys.executable, "-m", "results", "--ps", str(t.ps), "--reviewed", str(REVIEWED)],
                               cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            got = json.loads((t.ps / "results" / "results.json").read_text(encoding="utf-8"))
            assert_matches(self, got, expected("expected_results_reviewed.json"))
            self.assertTrue((t.ps / "results" / "per_image.csv").is_file())
            t.edit_images_csv("0002", nm_per_px="0.7362")
            r = subprocess.run([sys.executable, "-m", "results", "--ps", str(t.ps)],
                               cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("REFUSED", r.stderr)

    def test_threads_without_cryoai_folder(self):
        """add_layer_roles -> layer_count_stats resolves the CryoAI folder at import; counting must not need it.
        Eight threads start compute() at once: the chain is imported once, every thread gets the same answer,
        and $CRYOEV_CRYOAI_ROOT is left exactly as it was."""
        missing = str(REPO / "no_such_cryoai_folder")
        env = dict(os.environ, CRYOEV_CRYOAI_ROOT=missing)
        code = (
            "import json, os, sys, threading\n"
            "from results import compute\n"
            "out = []\n"
            "def go():\n"
            "    r = compute(sys.argv[1]); out.append((r['samples']['S1']['ev_total'], r['samples']['S2']['ev_total']))\n"
            "ts = [threading.Thread(target=go) for _ in range(8)]\n"
            "[t.start() for t in ts]; [t.join() for t in ts]\n"
            "print(json.dumps({'out': sorted(set(out)), 'n': len(out), 'env': os.environ['CRYOEV_CRYOAI_ROOT']}))\n")
        r = subprocess.run([sys.executable, "-c", code, str(PS)], cwd=REPO, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        got = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(got, {"out": [[5, 2]], "n": 8, "env": missing})


if __name__ == "__main__":
    unittest.main()
