"""Round 7 / R2: rejected images and low-conf EVs in the results engine (reviewed/1 additive fields, read with .get).
Synthetic prediction sets in temp folders only.

    python -m unittest discover -s results/tests -p test_reject.py -v

Known-answer control (one sample S1, 400x400 px, circles; nm/px 1.0 except A):
  A REJECTED, Reviewed: 5 EVs (one is a multilayer container), 1 inner layer inside the container, 1 non-EV, one
    reviewer MV call, and a bad (unconfirmed) nm/px
  B Reviewed: 3 EVs (one low conf, one a reviewer MV call) + 1 low-conf non-EV
  C Reviewed: 2 EVs
Expected (A dropped before the refusal check and every number): not refused, ev_total 5, mean 2.5 per image,
n_images 2, n_rejected 1, n_low_conf 1 (the low-conf EV only, not the non-EV), reviewer EV 5 / MV 1 = 20 %,
inner_layers_excluded 0, nonev_total 1, sum(diam_hist) == ev_total, nm_per_px range 1.0-1.0.
Wrong rules give: A kept -> refused (A has no confirmed nm/px); with that fixed, ev_total 10 / mean 3.33 / inner 1 /
nonev 2 / reviewer 3 of 10 = 30 % (A carries two MV calls, so its leak shows); counting the low-conf non-EV ->
n_low_conf 2.
"""
import csv
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from results import compute, write_results  # noqa: E402
import importlib  # noqa: E402
from results.tests import test_p2 as TP  # noqa: E402
from results.tests.test_p2 import MODEL, PS_ID  # noqa: E402

C = importlib.import_module("results.compute")


def poly(cx, cy, r, n=48):
    return [[cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)] for k in range(n)]


def obj(cx, cy, r, cls=0, rc="EV", low=False, note=None):
    o = {"cls": cls, "poly": poly(cx, cy, r), "conf": 0.9, "source": "model",
         "reviewer_label": rc, "reviewer_class": rc}
    if low:
        o["low_conf"] = True
    if note:
        o["note"] = note
    return o


def img(objs, reviewed=True, **extra):
    d = {"reviewed": reviewed, "edited": True, "reviewed_by": ["rev1"], "n_candidates_open": 0, "objects": objs}
    d.update(extra)
    return d


def A_IMG():
    objs = [obj(60, 60, 40, rc="MV"),            # container (geometry multilayer; reviewer MV)
            obj(60, 60, 15, rc="inner"),         # the inner layer
            obj(200, 60, 25, rc="MV"),           # a second reviewer MV call, geometry standalone
            obj(320, 60, 25), obj(60, 200, 25), obj(200, 200, 25),
            obj(320, 200, 25, cls=1, rc="nonEV")]
    return img(objs, rejected=True, reject_reason="thick ice", note="=HYPERLINK(\"http://x\")")


def B_IMG():
    return img([obj(60, 60, 25), obj(200, 60, 25, rc="EV", low=True, note="blurry edge"), obj(320, 60, 25, rc="MV"),
                obj(60, 300, 25, cls=1, rc="nonEV", low=True)])


def C_IMG():
    return img([obj(100, 100, 25), obj(300, 300, 30)])


def reviewed_doc(a=None, b=None, c=None):
    return {"schema": "reviewed/1", "package": f"pred_{PS_ID}", "prediction_set": PS_ID, "model": MODEL,
            "unknown_labels": 0, "images": {"0001": a or A_IMG(), "0002": b or B_IMG(), "0003": c or C_IMG()}}


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rej_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def make(self, labels=None, a_nm="2.0", a_confirmed=""):
        ps = TP.make_ps(self.tmp, {"0001": [], "0002": [], "0003": []}, labels=labels)
        rows = list(csv.DictReader(open(ps / "images.csv", newline="", encoding="utf-8")))
        for r in rows:
            if r["image"] == "0001":
                r["nm_per_px"], r["nm_confirmed_by"] = a_nm, a_confirmed       # bad: nobody confirmed it
        with open(ps / "images.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return ps


class KnownAnswer(Tmp):
    def test_rejected_image_leaves_every_number(self):
        ps = self.make()
        res = compute(ps, reviewed_doc())
        self.assertIsNone(res["refused"])                  # A's bad nm/px would refuse the sample if it counted
        s = res["samples"]["S1"]
        self.assertEqual((s["n_images"], s["n_rejected"]), (2, 1))
        self.assertEqual((s["ev_total"], s["ev_per_image_mean"]), (5, 2.5))     # NOT 10 / 3.33
        self.assertEqual(s["n_low_conf"], 1)                                    # NOT 2: the non-EV is not counted
        self.assertEqual((s["inner_layers_excluded"], s["nonev_total"]), (0, 1))
        self.assertEqual((s["reviewer_ev"], s["reviewer_mv"], s["pct_multilayer_reviewer"]), (5, 1, 20.0))
        self.assertEqual((s["n_reviewed"], s["n_checked"], s["n_unsure_open"]), (2, 2, 0))
        self.assertEqual((s["nm_per_px_min"], s["nm_per_px_max"]), (1.0, 1.0))
        self.assertEqual(sum(s["diam_hist"]["counts"]), s["ev_total"])
        self.assertEqual(s["pct_multilayer_geometry"], 0.0)                     # A's container is gone too
        self.assertNotIn("all_rejected", s)

    def test_wrong_rule_control_the_dropped_image_would_have_changed_things(self):
        """The control for the control: with A NOT rejected (and a confirmed nm/px) the same data gives the numbers
        the wrong rules would give, so the fixture does discriminate."""
        ps = self.make(a_nm="1.0", a_confirmed="tester")
        doc = reviewed_doc()
        doc["images"]["0001"].pop("rejected")
        s = compute(ps, doc)["samples"]["S1"]
        self.assertEqual((s["ev_total"], s["n_images"], s["inner_layers_excluded"], s["nonev_total"]), (10, 3, 1, 2))
        self.assertEqual(s["ev_per_image_mean"], 3.3333)
        self.assertEqual(s["pct_multilayer_reviewer"], 30.0)       # (2 + 1) / 10: A's MV calls leak in

    def test_rejected_image_stays_in_images_and_csv(self):
        ps = self.make()
        res = compute(ps, reviewed_doc())
        a = res["images"]["0001"]
        self.assertEqual((a["rejected"], a["reject_reason"]), (True, "thick ice"))
        self.assertFalse(res["images"]["0002"]["rejected"])
        out = write_results(res, self.tmp / "out")
        rows = list(csv.DictReader(open(out / "per_image.csv", newline="", encoding="utf-8")))
        self.assertEqual([r["rejected"] for r in rows], ["true", "false", "false"])
        self.assertEqual(rows[0]["reject_reason"], "thick ice")
        self.assertEqual(rows[0]["note"], "'=HYPERLINK(\"http://x\")")        # formula guard
        self.assertEqual(rows[1]["note"], "")
        head = open(out / "per_sample.csv", encoding="utf-8").readline().strip().split(",")
        self.assertEqual(head[-2:], ["n_rejected", "n_low_conf"])
        srow = list(csv.DictReader(open(out / "per_sample.csv", newline="", encoding="utf-8")))[0]
        self.assertEqual((srow["n_rejected"], srow["n_low_conf"], srow["n_images"]), ("1", "1", "2"))

    def test_all_rejected_sample_is_a_notice_not_a_crash(self):
        ps = self.make(a_nm="1.0", a_confirmed="tester", labels={"0001": "S1", "0002": "S1", "0003": "S2"})
        doc = reviewed_doc()
        doc["images"]["0001"].pop("rejected")
        doc["images"]["0003"]["rejected"] = True
        res = compute(ps, doc)
        s2 = res["samples"]["S2"]
        self.assertEqual((res["samples"]["S1"]["n_images"], res["samples"]["S1"]["ev_total"]), (2, 8))
        self.assertTrue(s2["all_rejected"])
        self.assertEqual((s2["n_images"], s2["n_rejected"], s2["n_low_conf"]), (0, 1, 0))
        for f in ("ev_total", "ev_per_image_mean", "diam_nm_median", "pct_multilayer_geometry", "nm_per_px_min",
                  "nm_per_px_max", "pct_multilayer_reviewer"):
            self.assertIsNone(s2[f], f)
        self.assertEqual(sum(s2["diam_hist"]["counts"]), 0)
        self.assertIsNone(res["refused"])
        out = write_results(res, self.tmp / "out")                                # CSV writes with blanks
        rows = {r["sample_label"]: r for r in csv.DictReader(open(out / "per_sample.csv", newline="", encoding="utf-8"))}
        self.assertEqual((rows["S2"]["n_images"], rows["S2"]["n_rejected"], rows["S2"]["ev_total"]), ("0", "1", ""))

    def test_everything_rejected(self):
        ps = self.make(a_nm="1.0", a_confirmed="tester")
        doc = reviewed_doc()
        for v in doc["images"].values():
            v["rejected"] = True
        res = compute(ps, doc)
        self.assertEqual(list(res["samples"]), ["S1"])
        self.assertTrue(res["samples"]["S1"]["all_rejected"])
        write_results(res, self.tmp / "out")

    def test_refused_sample_still_ignores_rejected_images(self):
        """A refused sample's open-unsure tally is over its counted images only."""
        ps = self.make(a_nm="1.0", a_confirmed="tester")
        doc = reviewed_doc()
        # A: rejected but NOT reviewed and NOT edited (its outlines come from the .txt) and 9 open candidates: if
        # it leaked into the refused tally, S1 would show 9
        doc["images"]["0001"].update(reviewed=False, edited=False, n_candidates_open=9)
        rows = list(csv.DictReader(open(ps / "images.csv", newline="", encoding="utf-8")))
        for r in rows:
            if r["image"] == "0003":
                r["nm_per_px"] = "1.5"
        with open(ps / "images.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        res = compute(ps, doc)
        self.assertIn("S1", res["refused_unsure"])
        self.assertEqual(res["refused_unsure"]["S1"], 0)
        self.assertEqual(res["images"]["0001"]["rejected"], True)
        self.assertEqual(res["images"]["0001"]["reviewed"], False)
        self.assertIn("S1", res["refused"])                    # B and C differ in nm/px (A is not what refused it)

    def test_mixed_paths_rejected_unedited_image_from_the_txt(self):
        """A rejected, not edited, not reviewed: its outlines come from the .txt (a real EV there); it is still
        dropped: n_images 2, n_rejected 1, and its EV is not in ev_total."""
        ps = self.make(a_nm="1.0", a_confirmed="tester")
        pred = ps / "p" / MODEL / "predictions" / "0001.txt"
        pred.write_text(TP.pred_line(0, 0.9, 200, 200, 40, 400) + "\n", encoding="utf-8")
        doc = reviewed_doc()
        doc["images"]["0001"] = {"reviewed": False, "edited": False, "reviewed_by": [], "n_candidates_open": 0,
                                 "objects": [], "rejected": True, "reject_reason": "empty"}
        res = compute(ps, doc)
        s = res["samples"]["S1"]
        self.assertEqual((s["n_images"], s["n_rejected"], s["ev_total"], s["ev_per_image_mean"]), (2, 1, 5, 2.5))
        self.assertEqual(res["images"]["0001"]["ev"], 1)          # measured (it is on its own row), not counted
        self.assertEqual(res["images"]["0001"]["reject_reason"], "empty")

    def test_rejected_image_that_cannot_be_read_does_not_fail_the_set(self):
        ps = self.make(a_nm="1.0", a_confirmed="tester")
        (ps / "p" / MODEL / "predictions" / "0001.txt").unlink()            # its .txt is gone ...
        doc = reviewed_doc()
        doc["images"]["0001"] = {"reviewed": False, "edited": False, "reviewed_by": [], "n_candidates_open": 0,
                                 "objects": [], "rejected": True, "reject_reason": "other", "note": "bad file"}
        res = compute(ps, doc)                                              # ... and the set still computes
        a = res["images"]["0001"]
        self.assertIn("bad file | not read (FileNotFoundError", a["note"])
        self.assertEqual(res["samples"]["S1"]["ev_total"], 5)
        (ps / "images" / "0001.png").write_bytes(b"not a png")             # a broken picture as well
        res = compute(ps, doc)
        self.assertIn("not read", res["images"]["0001"]["note"])
        write_results(res, self.tmp / "out")
        # a COUNTED image that can't be read is still a real error
        (ps / "images" / "0002.png").write_bytes(b"not a png")
        with self.assertRaises(Exception):
            compute(ps, reviewed_doc())


class Guard(unittest.TestCase):
    def test_guard_cell(self):
        for t in ("=1+1", "+1", "-1", "@SUM(A1)", "\tx", "\rx"):
            self.assertEqual(C.guard_cell(t), "'" + t)
        for t in ("", "plain", "a=b", "'=x", "1.5"):
            self.assertEqual(C.guard_cell(t), t)

    def test_numbers_and_sample_labels(self):
        self.assertEqual(C._cell(-3), -3)
        self.assertEqual(C._cell("-ctrl"), "'-ctrl")


if __name__ == "__main__":
    unittest.main()
