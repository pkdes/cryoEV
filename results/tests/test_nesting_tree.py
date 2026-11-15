"""nesting_tree v1 (round 15, contract K1): the poster's tree rule and polar probe as unit tests, with the params
passed in, plus boundary cases, edge cases and the step run through results.compute.

    python -m unittest discover -s results/tests -p test_nesting_tree.py        (from the repo root)

Known answers:
  * the poster's nesting_tree.py --selftest cases (12, polar mode, g_ml 16) under the FINAL params (offset 0.65,
    shape 0.35) and under the trial params (0.5 / 0.3) passed in, so nothing reads a module global;
  * the poster's nesting_polar_probe.py --selftest cases (6), and the analytic offsets 0.511 / 1.102 that the probe
    gave as 0.515 / 1.104;
  * boundary cases at offset 0.50-0.65, shape 0.30-0.35, gap 16 nm and gap_min 1.5 nm (values as the step sees
    them: polar rounded to 3 dp, gaps to 2 dp);
  * edge parity: the tree's parent links are nesting v1's parent_line links, and nesting v1's direct pairs minus the
    tree's are exactly the outlines inside two non-nested outlines.
"""

import importlib
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis import steps as ST  # noqa: E402
from analysis.steps import nesting_tree as T  # noqa: E402
from analysis.steps import polar as POL  # noqa: E402
from results.tests.test_steps import PS, SynthPS, disc, read_csv  # noqa: E402
C = importlib.import_module("results.compute")

FINAL = {"rule": "tree_v2", "g_ml_nm": 16, "offset_max": 0.65, "shape_max": 0.35, "gap_min_nm": 1.5,
         "wide_as_layer": False, "measure_px": 1440}
TRIAL = dict(FINAL, offset_max=0.5, shape_max=0.3)


def edge(p, c, g, cv=0.2, mn=6.0, ar=0.6, off=None, sh=0.1):
    """The poster selftest's E(): polar fields default to offset = cv, shape = 0.1."""
    return {"parent": p, "child": c, "gap_median_nm": g, "gap_cv": cv, "gap_min_nm": mn, "area_ratio": ar,
            "offset_idx": cv if off is None else off, "shape_idx": sh}


def call(edges, p=FINAL):
    return T.notation(T.build(edges, p)[0])


POSTER_CASES = [   # nesting_tree.py selftest, polar mode (the POLAR branches taken)
    ("tight pair", [edge(1, 2, 9)], "ML2"),
    ("three tight layers", [edge(1, 2, 9), edge(2, 3, 10)], "ML3"),
    ("off-centre inner", [edge(1, 2, 30, cv=0.8, ar=0.2)], "MV[s]"),
    ("touching inner (min 0)", [edge(1, 2, 8, cv=0.3, mn=0.5)], "MV[s]"),
    ("ML inside MV", [edge(1, 2, 30, cv=0.8, ar=0.3), edge(2, 3, 9)], "MV[ML2]"),
    ("ML holding a vesicle", [edge(1, 2, 9), edge(2, 3, 25, cv=0.9, ar=0.1)], "ML2[s]"),
    ("MV: ML + single + nested MV", [edge(1, 2, 40, cv=0.7, ar=0.2), edge(2, 3, 8), edge(1, 4, 30, cv=0.9, ar=0.1),
                                     edge(1, 5, 35, cv=0.8, ar=0.3), edge(5, 6, 20, cv=0.9, ar=0.1)], "MV[ML2,MV[s],s]"),
    ("two lamella candidates: largest wins", [edge(1, 2, 9, ar=0.7), edge(1, 3, 11, ar=0.5)], "ML2[s]"),
    ("concentric but wide inner", [edge(1, 2, 22, cv=0.2, ar=0.4)], "MV[c:s]"),
    ("MV holding a wide-concentric ML", [edge(1, 2, 30, cv=0.8, ar=0.3), edge(2, 3, 20, cv=0.2), edge(3, 4, 9)],
     "MV[MV[c:ML2]]"),
    ("even gap but oval mismatch (shape 0.4)", [edge(1, 2, 10, cv=0.2, sh=0.4)], "MV[s]"),
    ("offset 0.45 but cv 0.6", [edge(1, 2, 10, cv=0.6, off=0.45)], "ML2"),
]


class PosterTreeSelftest(unittest.TestCase):
    def test_twelve_cases_final_params(self):
        for name, edges, want in POSTER_CASES:
            self.assertEqual(call(edges, FINAL), want, name)
        self.assertEqual(len(POSTER_CASES), 12)

    def test_twelve_cases_trial_params_passed_in(self):
        """The poster ran these at 0.5 / 0.3 too; no case sits between the two cuts, so the same calls."""
        for name, edges, want in POSTER_CASES:
            self.assertEqual(call(edges, TRIAL), want, name)

    def test_classes(self):
        want_cls = {"MV[ML2]": "MV", "ML2[s]": "ML+MV", "ML3": "ML"}
        for name, edges, want in POSTER_CASES:
            if want in want_cls:
                self.assertEqual(T.cls(T.build(edges, FINAL)[0]), want_cls[want], name)
                self.assertEqual(T.class_of_call(want), want_cls[want])
        self.assertEqual(T.class_of_call("s"), "single")
        self.assertEqual(T.class_of_call("MV[ML2,s]"), "MV")
        self.assertIsNone(T.class_of_call(None))

    def test_wide_as_layer_passed_in(self):
        p = dict(FINAL, wide_as_layer=True)
        self.assertEqual(call([edge(1, 2, 22, cv=0.2, ar=0.4)], p), "ML2")
        self.assertEqual(call([edge(1, 2, 30, cv=0.8, ar=0.3), edge(2, 3, 20, cv=0.2), edge(3, 4, 9)], p), "MV[ML3]")

    def test_params_are_the_final_rule(self):
        self.assertEqual(T.PARAMS, FINAL)
        self.assertEqual(ST.STEPS[("nesting_tree", 1)].params, FINAL)


class Boundaries(unittest.TestCase):
    """The FINAL cut sits inside offset 0.50-0.65 / shape 0.30-0.35: values at, inside and just past each edge, as the
    step sees them (polar 3 dp, gaps 2 dp)."""

    def test_offset(self):
        for off, want in ((0.5, "ML2"), (0.55, "ML2"), (0.6, "ML2"), (0.649, "ML2"), (0.65, "ML2"), (0.651, "MV[s]")):
            self.assertEqual(call([edge(1, 2, 10, off=off)]), want, off)
        for off, want in ((0.5, "ML2"), (0.501, "MV[s]"), (0.6, "MV[s]")):
            self.assertEqual(call([edge(1, 2, 10, off=off)], TRIAL), want, off)

    def test_shape(self):
        for sh, want in ((0.3, "ML2"), (0.33, "ML2"), (0.35, "ML2"), (0.351, "MV[s]")):
            self.assertEqual(call([edge(1, 2, 10, sh=sh)]), want, sh)
        self.assertEqual(call([edge(1, 2, 10, sh=0.33)], TRIAL), "MV[s]")

    def test_gap_and_min(self):
        self.assertEqual(call([edge(1, 2, 16.0)]), "ML2")
        self.assertEqual(call([edge(1, 2, 16.01)]), "MV[c:s]")              # concentric but wide
        self.assertEqual(call([edge(1, 2, 16.01, off=0.7)]), "MV[s]")
        self.assertEqual(call([edge(1, 2, 10, mn=1.5)]), "ML2")
        self.assertEqual(call([edge(1, 2, 10, mn=1.49)]), "MV[s]")
        self.assertEqual(call([edge(1, 2, 22, mn=1.49)]), "MV[s]")          # wide needs gap_min too

    def test_rounding_as_the_csvs(self):
        """A raw 0.6504 is 0.650 after the probe's 3 dp, so a lamella; the step only ever sees rounded values."""
        self.assertEqual(call([edge(1, 2, 10, off=round(0.6504, 3))]), "ML2")
        self.assertEqual(call([edge(1, 2, 10, off=round(0.6506, 3))]), "MV[s]")

    def test_no_polar_profile_is_an_inner_vesicle(self):
        self.assertEqual(call([edge(1, 2, 10, off=None, cv=None)]), "MV[s]")
        e = edge(1, 2, 22)
        e["offset_idx"] = e["shape_idx"] = None
        self.assertEqual(call([e]), "MV[s]")                                 # not wide either


# ---- the polar probe's selftest (nesting_polar_probe.py), its helpers copied ----------------------------------

def raster(fn, n=240):
    yy, xx = np.mgrid[0:n, 0:n].astype(float) + 0.5
    return fn(xx, yy).astype(np.uint8)


def polar_shape(c, base, amp=0.0, k=0, ph=0.0):
    cx, cy = c
    return lambda x, y: np.hypot(x - cx, y - cy) <= base + amp * np.cos(k * np.arctan2(y - cy, x - cx) + ph)


def exact_offset(Rr, rr, d):
    t = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    g = d * np.cos(t) + np.sqrt(Rr ** 2 - (d * np.sin(t)) ** 2) - rr
    _, amp = POL.harmonics(g)
    return amp[1] / amp[0]


class PolarSelftest(unittest.TestCase):
    def test_six_cases(self):
        C0 = (120, 120)
        circ = lambda c, rr: (lambda x, y: np.hypot(x - c[0], y - c[1]) <= rr)            # noqa: E731
        ell = lambda c, a, b: (lambda x, y: ((x - c[0]) / a) ** 2 + ((y - c[1]) / b) ** 2 <= 1)   # noqa: E731
        e2, e3 = exact_offset(60, 40, 10), exact_offset(60, 30, 29)
        self.assertEqual((round(e2, 3), round(e3, 3)), (0.511, 1.102))
        cases = [
            ("concentric circles, gap 8 px", circ(C0, 60), circ(C0, 52),
             dict(a0_nm=(7, 9), offset_idx=(0, 0.1), shape_idx=(0, 0.1), even_frac=(0.95, 1))),
            ("child shifted 10 px, mean gap 20", circ(C0, 60), circ((130, 120), 40),
             dict(a0_nm=(19, 21.5), offset_idx=(e2 - 0.05, e2 + 0.05), shape_idx=(0, 0.12))),
            ("touching: shift 29 of 30", circ(C0, 60), circ((149, 120), 30), dict(offset_idx=(e3 - 0.05, e3 + 0.05))),
            ("elliptical child 50x40 in circle 60", circ(C0, 60), ell(C0, 50, 40),
             dict(offset_idx=(0, 0.1), shape_idx=(0.25, 0.42))),
            ("parallel deformed (k=3, both)", polar_shape(C0, 60, 6, 3), polar_shape(C0, 50, 6, 3),
             dict(offset_idx=(0, 0.1), coupling=(0.85, 1), even_frac=(0.9, 1))),
            ("independent deformed (k=3, rotated 60 deg)", polar_shape(C0, 60, 6, 3), polar_shape(C0, 45, 6, 3, np.pi),
             dict(coupling=(-1, 0.0), rough_idx=(0.5, 1.5))),
        ]
        got = {}
        for name, fp, fc, want in cases:
            m = POL.metrics(raster(fp), raster(fc))
            got[name] = m
            for k, (lo, hi) in want.items():
                self.assertTrue(lo <= m[k] <= hi, f"{name}: {k}={m[k]} not in ({lo}, {hi})")
        # the probe's recorded values (Plan_20261114_lamellarity_ML_MV.md): 0.515 and 1.104
        self.assertEqual(got["child shifted 10 px, mean gap 20"]["offset_idx"], 0.515)
        self.assertEqual(got["touching: shift 29 of 30"]["offset_idx"], 1.104)

    def test_too_few_bins_is_none(self):
        """A straight 2 px bar seen from its own centroid covers only a few of the 72 bins: no profile."""
        bar = np.zeros((240, 240), np.uint8)
        bar[119:121, 80:160] = 1
        self.assertIsNone(POL.metrics(raster(polar_shape((120, 120), 60)), bar))

    def test_polar_is_the_poster_copy(self):
        """The copied block is the probe's lines 27-85; the module says so and keeps its sha."""
        text = (REPO / "analysis" / "steps" / "polar.py").read_text(encoding="utf-8")
        self.assertIn("7141fcef72ef7220421121379d17f53bc8e4392e09538353cccf0ac93bcd8671", text)
        for name in ("polar_profile", "harmonics", "strip_low", "metrics"):
            self.assertTrue(callable(getattr(POL, name)))
        self.assertEqual(POL.NB, 72)


# ---- the step on masks ---------------------------------------------------------------------------------------

def ring_scene():
    """400 x 400 at 1 nm/px. 0: r 100 at (200,200) holding 1: r 90 (tight layer, gap 10) which holds 2: r 30
    off-centre (inner vesicle) holding 3: r 22 (tight, gap 8); 4: r 40 off-centre in 0's lamella 1 too; 5: a 2 px bar
    in 1 (no polar profile); 6: a lone r 20 disc. Expected: unit 0 = ML2[ML2,s,s] ... see the test."""
    ms = [disc(200, 200, 100), disc(200, 200, 90), disc(170, 160, 30), disc(170, 160, 22), disc(240, 240, 30),
          None, disc(360, 360, 20)]
    bar = np.zeros((400, 400), bool)
    bar[148:150, 225:265] = True
    ms[5] = bar
    return [{"line": k, "cls": 0, "conf": 0.9, "poly_px": None, "mask": m, "key": f"g{k + 1}"} for k, m in enumerate(ms)]


class StepOnMasks(unittest.TestCase):
    CTX = {"image": "x", "nm_per_px": 1.0, "ev_ids": {0}, "H": 400, "W": 400}

    def run_tree(self, objs, ctx=None, params=None):
        ctx = ctx or self.CTX
        prior = {"nesting": ST.STEPS[("nesting", 1)].run_image(ctx, objs, {})}
        if params is None:
            return ST.STEPS[("nesting_tree", 1)].run_image(ctx, objs, prior), prior
        return T.run_image(ctx, objs, prior, params), prior

    def test_scene(self):
        out, prior = self.run_tree(ring_scene())
        self.assertEqual(prior["nesting"]["parent_line"], [None, 0, 1, 2, 1, 1, None])
        self.assertEqual(out["role"], ["outer", "lamella", "inner_vesicle", "lamella", "inner_vesicle", "inner_vesicle",
                                       "outer"])
        self.assertEqual(out["unit"], ["g1", "g1", "g3", "g3", "g5", "g6", "g7"])
        self.assertEqual(out["parent_unit"], [None, None, "g1", "g1", "g1", "g1", None])
        self.assertEqual(out["layer_index"], [1, 2, 1, 2, 1, 1, 1])
        self.assertEqual(out["vesicle_depth"], [0, 0, 1, 1, 1, 1, 0])
        self.assertEqual(out["call"], ["ML2[ML2,s,s]", None, "ML2", None, "s", "s", "s"])
        self.assertEqual(out["wide"], [0, 0, 0, 0, 0, 0, 0])
        self.assertIsNone(out["offset_idx"][0])
        self.assertIsNone(out["offset_idx"][5])                              # the bar: no polar profile
        self.assertLess(out["offset_idx"][1], 0.1)
        self.assertGreater(out["offset_idx"][2], 0.65)
        for v in out["offset_idx"] + out["shape_idx"]:
            self.assertTrue(v is None or v == round(v, 3))
        rows = [{"image": "x", "nesting.children": prior["nesting"]["children"][k],
                 **{c: out[c][k] for c in T.COLUMNS}} for k in range(7)]
        self.assertEqual(T.counters(rows), {"two_parents": 0, "no_polar": 1})
        self.assertIn("Outermost EVs: 0 ML, 0 MV, 1 ML+MV, 1 single; inner vesicles: 3 (1 multilamellar)",
                      T.summary(rows))

    def test_params_passed_in_change_the_calls(self):
        out, _ = self.run_tree(ring_scene(), params=dict(FINAL, g_ml_nm=5))
        self.assertEqual(out["call"][0], "MV[c:MV[MV[c:s],s,s]]")     # the gap 10 / 8 layers are now "wide"
        self.assertEqual(out["role"][1], "inner_vesicle")

    def test_no_nm_blanks_the_calls(self):
        out, _ = self.run_tree(ring_scene(), ctx=dict(self.CTX, nm_per_px=None))
        for c in ("unit", "parent_unit", "role", "layer_index", "vesicle_depth", "wide", "call"):
            self.assertEqual(out[c], [None] * 7, c)
        self.assertIsNotNone(out["offset_idx"][1])                           # ratios are still given

    def test_non_ev_and_empty_are_blank(self):
        objs = ring_scene()
        objs[4]["cls"] = 1
        objs.append({"line": 7, "cls": 0, "conf": 0.9, "poly_px": None, "mask": np.zeros((400, 400), bool)})
        out, _ = self.run_tree(objs)
        self.assertEqual([out[c][4] for c in T.COLUMNS], [None] * len(T.COLUMNS))
        self.assertEqual([out[c][7] for c in T.COLUMNS], [None] * len(T.COLUMNS))
        self.assertEqual(out["call"][0], "ML2[ML2,s]")

    def test_line_less_objects_take_the_recompute_path(self):
        """A drawn outline has no line, so lines cannot name parents: nest() is re-run and the result is the same."""
        objs = ring_scene()
        a, _ = self.run_tree(objs)
        for o in objs:
            o["line"] = None
        b, prior = self.run_tree(objs)
        self.assertEqual(prior["nesting"]["parent_line"], [None] * 7)      # nesting v1 alone cannot link them
        self.assertEqual(a, b)
        objs[2]["key"] = None                                                # no key, no line: #index
        c, _ = self.run_tree(objs)
        self.assertEqual(c["unit"][2:4], ["#2", "#2"])

    def test_two_direct_parents(self):
        """C (r 15 at 200,200) is inside A and B (two overlapping r 60 discs, neither inside the other, B larger):
        nesting v1 lists C as a direct child of both; the tree puts it under the smaller container (A) only."""
        A, B = disc(170, 200, 60), disc(232, 200, 62)
        objs = [{"line": k, "cls": 0, "conf": 0.9, "poly_px": None, "mask": m}
                for k, m in enumerate((A, B, disc(200, 200, 15)))]
        out, prior = self.run_tree(objs)
        kids = [[c["child"] for c in json.loads(prior["nesting"]["children"][k])] for k in range(3)]
        self.assertEqual(kids, [[2], [2], []])
        self.assertEqual(prior["nesting"]["parent_line"], [None, None, 0])
        self.assertEqual(out["parent_unit"], [None, None, "L0"])
        self.assertEqual(out["call"], ["MV[s]", "s", "s"])
        rows = [{"image": "x", "nesting.children": prior["nesting"]["children"][k],
                 **{c: out[c][k] for c in T.COLUMNS}} for k in range(3)]
        self.assertEqual(T.counters(rows)["two_parents"], 1)
        self.assertIn("1 outlines sit directly inside two outlines", T.summary(rows))

    def test_edge_parity_with_nesting(self):
        """Every tree link (outline -> its geometric parent, from unit / layer / role) is a nesting v1 parent_line
        link and vice versa; nesting's direct pairs minus those links are the two-parent outlines."""
        for objs in (ring_scene(),):
            out, prior = self.run_tree(objs)
            self.assertEqual(tree_links(out, objs, prior), {(p, k) for k, p in enumerate(prior["nesting"]["parent_line"])
                                                            if p is not None})


def tree_links(out, objs, prior):
    """(parent line, child line) links the tree implies: a lamella's parent is the previous outline of its unit's
    chain; an inner unit's first outline sits in some layer of its parent unit (that layer = nesting's parent_line,
    which must belong to the parent unit)."""
    line_of = {k: o["line"] for k, o in enumerate(objs)}
    by_unit = {}
    for k in range(len(objs)):
        if out["unit"][k] is not None:
            by_unit.setdefault(out["unit"][k], {})[out["layer_index"][k]] = k
    links = set()
    for k in range(len(objs)):
        if out["role"][k] == "lamella":
            links.add((line_of[by_unit[out["unit"][k]][out["layer_index"][k] - 1]], line_of[k]))
        elif out["role"][k] == "inner_vesicle":
            host = prior["nesting"]["parent_line"][k]
            assert host in {line_of[j] for j in by_unit[out["parent_unit"][k]].values()}
            links.add((host, line_of[k]))
    return links


def circle_poly(cx, cy, r, n=180):
    import math
    return [[cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)] for k in range(n)]


def poly_objs(spec, scale):
    """[(cx, cy, r)] at 1440-px coordinates -> objects at `scale` x that size, each with poly_px and the mask
    obj_mask rasterises from it (int truncation + fillPoly, as results.compute)."""
    W = H = 1440 * scale
    out = []
    for k, (cx, cy, r) in enumerate(spec):
        o = {"line": k, "cls": 0, "conf": 0.9, "poly_px": circle_poly(cx * scale, cy * scale, r * scale),
             "key": f"g{k + 1}"}
        o["mask"] = ST.obj_mask({"H": H, "W": W}, o)
        out.append(o)
    return out


class MeasurePx(unittest.TestCase):
    """measure_px (user decision 11-14): the rule measures gaps / polar at 1440 px wide, as calibrated on the poster.
    Known answer: a 2880 px image (0.5 nm/px) measured at 1440 gives exactly what the same polygons halved on a
    1440 px image (1.0 nm/px) give at native resolution; measure_px None measures the 2880 image natively."""
    # 0: r 300 holding 1: r 286 (a tight layer, ~14 nm at 1440) holding 2: off-centre r 80; 3: lone r 60
    SPEC = [(600, 600, 300), (601, 600, 286), (400, 600, 60), (1200, 1200, 60)]

    def tree(self, objs, W, nm, params):
        ctx = {"image": "x", "nm_per_px": nm, "ev_ids": {0}, "H": W, "W": W}
        prior = {"nesting": ST.STEPS[("nesting", 1)].run_image(ctx, objs, {})}
        seen = []
        real = T.build

        def spy(edges, p):
            seen.extend(edges)
            return real(edges, p)
        with unittest.mock.patch.object(T, "build", spy):
            out = T.run_image(ctx, objs, prior, params)
        return out, sorted((e["parent"], e["child"], e["gap_median_nm"], e["gap_min_nm"], e["area_ratio"])
                           for e in seen)

    def test_default_is_1440(self):
        self.assertEqual(T.PARAMS["measure_px"], 1440)
        self.assertEqual(ST.STEPS[("nesting_tree", 1)].params["measure_px"], 1440)

    def test_2880_measured_at_1440_equals_native_1440(self):
        big, small = poly_objs(self.SPEC, 2), poly_objs(self.SPEC, 1)
        a, ea = self.tree(big, 2880, 0.5, FINAL)
        b, eb = self.tree(small, 1440, 1.0, dict(FINAL, measure_px=None))
        self.assertEqual(a, b)
        self.assertEqual(ea, eb)                       # the gap stats the rule sees are the 1440 raster's
        self.assertEqual(a["call"][0], "ML2[s]")
        _, ec = self.tree(big, 2880, 0.5, dict(FINAL, measure_px=None))   # native: nesting v1's own gap stats
        kids = json.loads(ST.STEPS[("nesting", 1)].run_image({"image": "x", "nm_per_px": 0.5, "ev_ids": {0},
                                                              "H": 2880, "W": 2880}, big, {})["children"][0])
        self.assertEqual([e[2] for e in ec if e[0] == 0], [c["gap_median_nm"] for c in kids])
        self.assertNotEqual(ec, ea)

    def test_narrow_images_are_not_upsampled(self):
        small = poly_objs(self.SPEC, 1)
        self.assertEqual(self.tree(small, 1440, 1.0, FINAL), self.tree(small, 1440, 1.0, dict(FINAL, measure_px=None)))
        self.assertIsNone(T.measure_scale({"W": 1000}, 1440))
        self.assertEqual(T.measure_scale({"W": 5760}, 1440), 0.25)

    def test_child_gone_at_1440_is_an_inner_vesicle(self):
        """A mask-only child of one odd pixel vanishes in the nearest resize: no gap, no polar, an inner vesicle."""
        big = poly_objs(self.SPEC[:1], 2)
        dot = np.zeros((2880, 2880), bool)
        dot[1201, 1201] = True
        big.append({"line": 1, "cls": 0, "conf": 0.9, "poly_px": None, "mask": dot, "key": "g9"})
        out, edges = self.tree(big, 2880, 0.5, FINAL)
        self.assertEqual(out["role"], ["outer", "inner_vesicle"])
        self.assertIsNone(out["offset_idx"][1])
        self.assertEqual(out["call"][0], "MV[s]")


class Recipe(unittest.TestCase):
    def test_needs_and_pin(self):
        self.assertEqual([s.key for s in ST.validate_recipe([("roles", 1), ("nesting", 1), ("nesting_tree", 1)])],
                         [("roles", 1), ("nesting", 1), ("nesting_tree", 1)])
        with self.assertRaises(ValueError) as cm:
            ST.validate_recipe([("roles", 1), ("nesting_tree", 1)])
        self.assertIn("needs nesting earlier", str(cm.exception))
        fake = ST.Step("nesting", 2, needs=(), columns=("x",), params={}, fn=lambda c, o, p: {"x": [None] * len(o)})
        steps = {**ST.STEPS, fake.key: fake}
        with self.assertRaises(ValueError) as cm:
            ST.validate_recipe([("roles", 1), ("nesting", 2), ("nesting_tree", 1)], steps)
        self.assertIn("built on nesting v1", str(cm.exception))
        self.assertEqual(ST.with_needs([("nesting_tree", 1)], steps), [("nesting", 1), ("nesting_tree", 1)])

    def test_code_sha_covers_nesting(self):
        st = ST.STEPS[("nesting_tree", 1)]
        self.assertIn("analysis/steps/nesting.py", st.sources)
        self.assertIn("analysis/steps/polar.py", st.sources)
        self.assertEqual(st.needs, ("nesting",))
        self.assertEqual(st.pins, {"nesting": 1})
        self.assertEqual(list(ST.STEPS)[-2:], [("nesting", 1), ("nesting_tree", 1)])   # listed after nesting


class ThroughCompute(unittest.TestCase):
    RECIPE = [("roles", 1), ("nesting", 1), ("nesting_tree", 1)]

    def test_columns_keys_and_counts_unchanged(self):
        with SynthPS() as t:
            base, _ = C.compute_and_write(t.ps)
            res, out = C.compute_and_write(t.ps, recipe=self.RECIPE)
            self.assertEqual({k: v for k, v in res.items() if k not in ("steps", "computed")},
                             {k: v for k, v in base.items() if k != "computed"})
            rows = read_csv(out / "objects.csv")
            self.assertEqual(list(rows[0])[-9:], [f"nesting_tree.{c}" for c in T.COLUMNS])
            # the 3-layer stack r 60 / 40 / 20 at 1 nm/px: gaps 20 nm > 16, so wide inner vesicles
            self.assertEqual(rows[0]["nesting_tree.call"], "MV[c:MV[c:s]]")   # even 20 px gaps: concentric, wide
            self.assertEqual(rows[1]["nesting_tree.unit"], rows[1]["pkg_key"])      # unit id = the package key
            self.assertEqual(rows[2]["nesting_tree.parent_unit"], rows[1]["pkg_key"])
            self.assertEqual(rows[6]["nesting_tree.unit"], "")                    # conf 0.3: no step ran
            doc = json.loads((out / "steps.json").read_text(encoding="utf-8"))
            self.assertEqual([e["step"] for e in doc["recipe"]], ["roles", "nesting", "nesting_tree"])
            self.assertIn("Outermost EVs", doc["summary"]["nesting_tree"])

    def test_reviewed_objects_carry_their_key(self):
        """On reviewed images the step objects carry the package key (the objects.csv pkg_key), so tree rows map."""
        rev = json.loads((PS.parents[1] / "reviewed.json").read_text(encoding="utf-8"))
        rev["images"]["0002"]["objects"][0]["key"] = "p0"          # an object known by its package key only
        seen = []
        orig = T.NESTING_TREE_V1._fn

        def spy(ctx, objs, prior):
            seen.append((ctx["image"], [o.get("key") for o in objs]))
            return orig(ctx, objs, prior)
        T.NESTING_TREE_V1._fn = spy
        try:
            res = C.compute(PS, rev, recipe=self.RECIPE)
        finally:
            T.NESTING_TREE_V1._fn = orig
        by_image = {}
        for o in res.objects:
            if o["exclude_reason"] != "conf":
                by_image.setdefault(o["image"], []).append(o["pkg_key"])
        self.assertTrue(seen)
        for image, keys in seen:
            self.assertEqual(keys, by_image[image], image)
        self.assertEqual(dict(seen)["0002"][0], "p0")
        self.assertTrue(all(dict(seen)["0001"]))                      # untouched image: every outline has its g<id>

    def test_default_recipe_unchanged_by_the_prepass(self):
        """The key pre-pass runs only for a non-default recipe; objects.csv pkg_key is the same either way."""
        a = C.compute(PS)
        b = C.compute(PS, recipe=self.RECIPE)
        self.assertEqual([(o["image"], o["line"], o["pkg_key"]) for o in a.objects],
                         [(o["image"], o["line"], o["pkg_key"]) for o in b.objects])


if __name__ == "__main__":
    unittest.main()
