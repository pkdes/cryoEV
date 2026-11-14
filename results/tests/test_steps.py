"""Analysis steps (round 14, contract C5): analysis/steps (Step API, recipe check, roles v1, nesting v1) and what
results.compute writes with them (objects.csv, steps.json, results.json unchanged under the default recipe).

    python -m unittest discover -s results/tests -p test_steps.py        (from the repo root)

Known answers: a synthetic set with one 3-layer stack and two single EVs (the add_layer_roles self-test scene:
outer layer_count 2), concentric discs for the nesting gap (r 60 vs 40 px -> about 20 px), and the golden fixtures
(results.json byte-identical with and without an explicit default recipe).
"""

import csv
import hashlib
import importlib
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis import steps as ST  # noqa: E402
from analysis.steps import nesting as NEST  # noqa: E402
from results import compute, write_results  # noqa: E402

C = importlib.import_module("results.compute")

GOLDEN = REPO / "tests" / "fixtures" / "cj_golden"
PS = GOLDEN / "predict" / "ps_20261105_abc123"
MODEL = "v5_evnonev_y8m_20261029"
PS_ID = "ps_20261105_abc123"


def circle(cx, cy, r, w, h, n=180):
    return " ".join(f"{(cx + r * math.cos(2 * math.pi * k / n)) / w:.6f} {(cy + r * math.sin(2 * math.pi * k / n)) / h:.6f}"
                    for k in range(n))


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class SynthPS:
    """A one-image prediction set (400 x 400): a 3-layer stack (r 60 / 40 / 20 at 100,100), two singles, one non-EV
    and one low-confidence line (0.3)."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        shutil.copytree(PS, Path(self._tmp.name) / PS_ID)
        self.ps = Path(self._tmp.name) / PS_ID
        for f in (self.ps / "images").iterdir():
            f.unlink()
        for f in (self.ps / "p" / MODEL / "predictions").iterdir():
            f.unlink()
        Image.new("L", (400, 400), 128).save(self.ps / "images" / "0001.png")
        lines = [(0, 0.9, circle(100, 100, 60, 400, 400)), (0, 0.8, circle(100, 100, 40, 400, 400)),
                 (0, 0.7, circle(100, 100, 20, 400, 400)), (0, 0.9, circle(300, 300, 30, 400, 400)),
                 (0, 0.6, circle(300, 100, 25, 400, 400)), (1, 0.9, circle(100, 300, 30, 400, 400)),
                 (0, 0.3, circle(200, 200, 15, 400, 400))]
        (self.ps / "p" / MODEL / "predictions" / "0001.txt").write_text(
            "".join(f"{c} {s} {p}\n" for c, s, p in lines), encoding="utf-8")
        with open(self.ps / "images.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["image", "file", "orig_name", "sample_label", "nm_per_px", "nm_confirmed_by", "source"])
            w.writerow(["0001", "images/0001.png", "synth.png", "S1", "1.0", "tester", "session:synth"])
        return self

    def __exit__(self, *exc):
        self._tmp.cleanup()


class Recipe(unittest.TestCase):
    def test_default_and_known_steps(self):
        self.assertEqual(ST.DEFAULT_RECIPE, [("roles", 1)])
        self.assertEqual([s.key for s in ST.validate_recipe()], [("roles", 1)])
        self.assertTrue(ST.is_default(ST.validate_recipe([{"step": "roles", "version": 1}])))
        self.assertEqual([s.key for s in ST.validate_recipe([("roles", 1), ("nesting", 1)])],
                         [("roles", 1), ("nesting", 1)])
        self.assertFalse(ST.is_default(ST.validate_recipe([("roles", 1), ("nesting", 1)])))

    def test_refusals(self):
        for bad, why in (([("roles", 2)], "unknown step"), ([("bogus", 1)], "unknown step"),
                         ([("roles", 1), ("roles", 1)], "twice"), ([("nesting", 1)], "must include roles"),
                         ([("roles", "x")], "whole number"), (["roles"], "expected")):
            with self.assertRaises(ValueError) as cm:
                ST.validate_recipe(bad)
            self.assertIn(why, str(cm.exception), bad)

    def test_unmet_needs_refused(self):
        """A step whose needs are not EARLIER in the recipe is refused (fixture step needing roles)."""
        late = ST.Step("late", 1, needs=("roles",), columns=("x",), params={}, fn=lambda c, o, p: {"x": [1] * len(o)})
        steps = {**ST.STEPS, late.key: late}
        self.assertEqual([s.name for s in ST.validate_recipe([("roles", 1), ("late", 1)], steps)], ["roles", "late"])
        with self.assertRaises(ValueError) as cm:
            ST.validate_recipe([("late", 1), ("roles", 1)], steps)
        self.assertIn("needs roles earlier", str(cm.exception))

    def test_step_output_shape_checked(self):
        bad = ST.Step("bad", 1, needs=(), columns=("x",), params={}, fn=lambda c, o, p: {"x": []})
        with self.assertRaises(ValueError):
            bad.run_image({}, [{"cls": 0}], {})

    def test_roles_params_are_the_counting_pair(self):
        from analysis.steps.roles import load_roles_for
        load_roles_for()
        from analysis.layer_count_stats import CONTAINMENT_THRESHOLD, MAX_SIZE_RATIO
        self.assertEqual(ST.STEPS[("roles", 1)].params, {"containment": CONTAINMENT_THRESHOLD,
                                                         "size_ratio": MAX_SIZE_RATIO})

    def test_code_sha_stamp(self):
        st = ST.STEPS[("roles", 1)].stamp()
        self.assertEqual(set(st), {"step", "version", "params", "code_sha"})
        self.assertRegex(st["code_sha"], r"^[0-9a-f]{64}$")
        self.assertEqual(st["code_sha"], ST.STEPS[("roles", 1)].code_sha())
        self.assertNotEqual(st["code_sha"], ST.STEPS[("nesting", 1)].code_sha())


def disc(cx, cy, r, shape=(400, 400)):
    import cv2
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    return m.astype(bool)


class Steps(unittest.TestCase):
    CTX = {"image": "x", "nm_per_px": 2.0, "ev_ids": {0}, "H": 400, "W": 400}

    def objs(self):
        ms = [disc(100, 100, 60), disc(100, 100, 40), disc(100, 100, 20), disc(300, 300, 30), disc(300, 100, 25),
              disc(100, 300, 30)]
        return [{"line": k, "cls": 1 if k == 5 else 0, "conf": 0.9, "poly_px": None, "mask": m} for k, m in enumerate(ms)]

    def test_roles_self_test_scene(self):
        out = ST.STEPS[("roles", 1)].run_image(self.CTX, self.objs(), {})
        self.assertEqual(out["role"], ["multilayer", "inner-layer", "inner-layer", "standalone", "standalone", None])
        self.assertEqual(out["layer_count"], [2, 0, 0, 0, 0, None])

    def test_nesting_known_gaps(self):
        out = ST.STEPS[("nesting", 1)].run_image(self.CTX, self.objs(), {})
        self.assertEqual(out["n_contained"], [2, 1, 0, 0, 0, None])
        self.assertEqual(out["n_direct_children"], [1, 1, 0, 0, 0, None])
        self.assertEqual(out["d_outer_nm"][1:3], [None, None])           # contained: not an outer outline
        self.assertAlmostEqual(out["d_outer_nm"][0], 2 * math.sqrt(disc(100, 100, 60).sum() / math.pi) * 2.0, places=1)
        kids = json.loads(out["children"][0])
        self.assertEqual([k["child"] for k in kids], [1])                  # direct child only (not the r 20 disc)
        self.assertLess(abs(kids[0]["gap_median_nm"] - 20 * 2.0), 1.5 * 2.0)   # r 60 - r 40 = 20 px, x 2 nm/px
        self.assertLessEqual(kids[0]["gap_min_nm"], kids[0]["gap_median_nm"])
        self.assertLessEqual(kids[0]["gap_median_nm"], kids[0]["gap_p90_nm"])
        self.assertAlmostEqual(kids[0]["area_ratio"], round(disc(100, 100, 40).sum() / disc(100, 100, 60).sum(), 3))
        inner = json.loads(out["children"][1])
        self.assertLess(abs(inner[0]["gap_median_nm"] - 20 * 2.0), 1.5 * 2.0)

    def test_nesting_no_size_ratio_filter(self):
        """A tight layer (r 58 in r 60, ratio 0.93 > the roles 0.75) is not a roles child but IS a nesting child."""
        objs = [{"line": 0, "cls": 0, "conf": 0.9, "mask": disc(100, 100, 60)},
                {"line": 1, "cls": 0, "conf": 0.9, "mask": disc(100, 100, 58)}]
        self.assertEqual(ST.STEPS[("roles", 1)].run_image(self.CTX, objs, {})["role"], ["standalone", "standalone"])
        out = ST.STEPS[("nesting", 1)].run_image(self.CTX, objs, {})
        self.assertEqual(out["n_direct_children"], [1, 0])

    def test_nesting_without_nm(self):
        out = ST.STEPS[("nesting", 1)].run_image(dict(self.CTX, nm_per_px=None), self.objs(), {})
        self.assertIsNone(out["d_outer_nm"][0])
        self.assertIsNone(json.loads(out["children"][0])[0]["gap_median_nm"])

    def test_poly_only_objects_are_rasterised(self):
        objs = [{"line": 0, "cls": 0, "conf": 0.9,
                 "poly_px": [[100 + 60 * math.cos(a), 100 + 60 * math.sin(a)] for a in np.linspace(0, 6.28, 90)]},
                {"line": 1, "cls": 0, "conf": 0.9,
                 "poly_px": [[100 + 20 * math.cos(a), 100 + 20 * math.sin(a)] for a in np.linspace(0, 6.28, 90)]}]
        self.assertEqual(ST.STEPS[("roles", 1)].run_image(self.CTX, objs, {})["role"], ["multilayer", "inner-layer"])


class ComputeWithSteps(unittest.TestCase):
    def test_three_layer_self_test_through_compute(self):
        """objects.csv: the 3-layer stack's outer outline is multilayer with layer_count 2; included multilayer
        rows == the multilayer count; the conf 0.3 line and the non-EV are listed, not counted."""
        with SynthPS() as t:
            res, out = C.compute_and_write(t.ps)
            rows = read_csv(out / "objects.csv")
            self.assertEqual(list(rows[0]), C.OBJECT_COLUMNS + ["roles.role", "roles.layer_count"])
            self.assertEqual([r["line"] for r in rows], [str(k) for k in range(7)])
            self.assertEqual(rows[0]["roles.role"], "multilayer")
            self.assertEqual(rows[0]["roles.layer_count"], "2")
            self.assertEqual([r["exclude_reason"] for r in rows], ["", "", "", "", "", "non_ev", "conf"])
            self.assertEqual([r["included"] for r in rows], ["1"] * 5 + ["0", "0"])
            self.assertEqual([r["pkg_key"] for r in rows], ["g1", "g2", "g3", "g4", "g5", "g6", "p0"])
            ml = sum(1 for r in rows if r["included"] == "1" and r["roles.role"] == "multilayer")
            self.assertEqual(ml, res["images"]["0001"]["multilayer"])
            self.assertEqual((res["images"]["0001"]["ev"], res["images"]["0001"]["inner"]), (3, 2))
            doc = json.loads((out / "steps.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["schema"], "steps/1")
            self.assertEqual([(s["step"], s["version"]) for s in doc["recipe"]], [("roles", 1)])
            self.assertEqual(doc["objects_csv_sha256"], hashlib.sha256((out / "objects.csv").read_bytes()).hexdigest())
            self.assertEqual((doc["n_objects"], doc["n_included"], doc["computed"]), (7, 5, res["computed"]))
            self.assertIn("1 multilayer", doc["summary"]["roles"])

    def test_nesting_recipe_adds_columns_and_steps_key(self):
        with SynthPS() as t:
            base, _ = C.compute_and_write(t.ps)
            res, out = C.compute_and_write(t.ps, recipe=[("roles", 1), ("nesting", 1)])
            self.assertEqual(res["steps"], [{"step": "roles", "version": 1}, {"step": "nesting", "version": 1}])
            self.assertEqual({k: v for k, v in res.items() if k not in ("steps", "computed")},
                             {k: v for k, v in base.items() if k != "computed"})          # counts unchanged
            rows = read_csv(out / "objects.csv")
            self.assertEqual(list(rows[0])[-4:], ["nesting.n_contained", "nesting.n_direct_children",
                                                   "nesting.d_outer_nm", "nesting.children"])
            self.assertEqual(rows[0]["nesting.n_contained"], "2")
            self.assertEqual([c["child"] for c in json.loads(rows[0]["nesting.children"])], [1])
            self.assertEqual(rows[6]["nesting.children"], "")                  # conf 0.3: no step ran on it
            self.assertIn('"steps"', (out / "results.json").read_text(encoding="utf-8"))

    def test_bad_recipe_refused_before_reading(self):
        with self.assertRaises(ValueError):
            compute(PS, recipe=[("nesting", 1)])

    def test_golden_default_recipe_byte_identical(self):
        """results.json with recipe=None and recipe=[roles v1] are the same bytes (bar `computed`) and carry no
        "steps" key; per_image / per_sample identical."""
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            ra = compute(PS, GOLDEN / "reviewed.json")
            rb = compute(PS, GOLDEN / "reviewed.json", recipe=[("roles", 1)])
            rb["computed"] = ra["computed"]
            write_results(ra, a)
            write_results(rb, b)
            for f in ("results.json", "per_image.csv", "per_sample.csv", "objects.csv"):
                self.assertEqual((a / f).read_bytes(), (b / f).read_bytes(), f)
            self.assertNotIn("steps", ra)
            self.assertEqual(json.loads((a / "results.json").read_text(encoding="utf-8")), dict(ra))

    def test_reasons_rejected_and_refused(self):
        """A rejected image's rows say rejected_image; a refused sample's rows say refused_sample."""
        rev = json.loads((GOLDEN / "reviewed.json").read_text(encoding="utf-8"))
        rev["images"]["0003"]["rejected"] = True
        with tempfile.TemporaryDirectory() as d:
            ps = Path(d) / PS_ID
            shutil.copytree(PS, ps)
            res = compute(ps, rev)
            rows = res.objects
            self.assertTrue(rows)
            self.assertEqual({o["exclude_reason"] for o in rows if o["image"] == "0003"}, {"rejected_image"})
            text = (ps / "images.csv").read_text(encoding="utf-8").replace("0002,images/0002.png,golden_0002.png,S1,1.4724",
                                                                          "0002,images/0002.png,golden_0002.png,S1,2.0")
            (ps / "images.csv").write_text(text, encoding="utf-8")
            res = compute(ps)
            self.assertIn("S1", res["refused"])
            self.assertEqual({o["exclude_reason"] for o in res.objects if o["image"] in ("0001", "0002")},
                             {"refused_sample"})
            self.assertTrue(all(o["exclude_reason"] in C.EXCLUDE_REASONS or o["included"] == 1 for o in res.objects))

    def test_reviewed_objects_find_their_line(self):
        """An edited image's objects kept as the model drew them get their prediction line (polygon match); a
        reviewed.json object that carries its package key gets the key's line."""
        rev = json.loads((GOLDEN / "reviewed.json").read_text(encoding="utf-8"))
        res = compute(PS, rev)
        with_line = [o for o in res.objects if o["image"] == "0002"]
        self.assertTrue(with_line)
        rev2 = json.loads(json.dumps(rev))
        objs = rev2["images"]["0002"]["objects"]
        objs[0]["key"] = "p0"
        res2 = compute(PS, rev2)
        o0 = [o for o in res2.objects if o["image"] == "0002"][0]
        self.assertEqual(o0["pkg_key"], "p0")

    def test_plain_dict_write_drops_stale_objects(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            res = compute(PS)
            write_results(res, out)
            self.assertTrue((out / "objects.csv").is_file())
            write_results(dict(res), out)
            self.assertFalse((out / "objects.csv").exists())
            self.assertFalse((out / "steps.json").exists())


class NestingPort(unittest.TestCase):
    def test_gap_stats_matches_the_poster_definition(self):
        """gap_stats on crops = distance transform of the parent sampled on the child's contour, minus 0.5."""
        import cv2
        par, ch = NEST.crop(disc(100, 100, 60)), NEST.crop(disc(100, 100, 40))
        g = NEST.gap_stats(par, ch, 400, 400)
        dt = cv2.distanceTransform(disc(100, 100, 60).astype(np.uint8), cv2.DIST_L2, 5)
        cs, _ = cv2.findContours(disc(100, 100, 40).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = np.vstack([c[:, 0, :] for c in cs])
        self.assertTrue(np.allclose(np.sort(g), np.sort(dt[pts[:, 1], pts[:, 0]] - 0.5), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
