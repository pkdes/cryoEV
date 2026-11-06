"""Round 6 / P2: unsure counts, checked images and the diameter histogram (results.json additive fields), with the
package side (reviewed/1 n_candidates_open, data.js label_conf). Synthetic prediction sets in temp folders only.

    python -m unittest discover -s results/tests -p test_p2.py -v

Known-answer control (one sample, 3 images, 400x400 px, 1 nm/px, circles so the diameters are known):
  A untouched          3 confident (0.9) non-nested EV outlines + 1 candidate (0.30)
  B edited, not Rev.   2 candidates (0.40, 0.35); the edits accept the first, leave the second undecided
  C Reviewed           1 candidate (0.30), undecided
  unsure open = A 1 (untouched: all) + B 1 (edited: the open one) + C 0 (Reviewed: closed) = 2.
  Wrong rules: "n_candidates_open everywhere" gives 1+1+1 = 3; "edited images only" gives 0+1+0 = 1;
  "all candidates everywhere" gives 1+2+1 = 4.
  counted EVs = A's 3 + B's accepted 1 = 4 (C's undecided candidate is not an EV). checked = B + C = 2.
"""
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import importlib  # noqa: E402

from annotation import prediction_package as pp  # noqa: E402
from results import compute, write_results  # noqa: E402

C = importlib.import_module("results.compute")

MODEL = "v5_evnonev_y8m_20261029"
PS_ID = "ps_20261106_a1b2c3"


def circle(cx, cy, r, size, n=48):
    return [(cx + r * math.cos(2 * math.pi * k / n)) / size for k in range(n)], \
           [(cy + r * math.sin(2 * math.pi * k / n)) / size for k in range(n)]


def pred_line(cls, conf, cx, cy, r, size):
    xs, ys = circle(cx, cy, r, size)
    return f"{cls} {conf:.2f} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))


def make_ps(root: Path, images: dict, sizes=None, ps_id=PS_ID, labels=None) -> Path:
    """images: {"0001": [(cls, conf, cx, cy, r), ...]} -> a done prediction_set (contract section 2), 1 nm/px."""
    ps = root / "predict" / ps_id
    (ps / "images").mkdir(parents=True)
    pred = ps / "p" / MODEL / "predictions"
    pred.mkdir(parents=True)
    rows = ["image,file,orig_name,sample_label,nm_per_px,nm_confirmed_by,source"]
    for image, objs in images.items():
        size = (sizes or {}).get(image, 400)
        Image.new("L", (size, size), 128).save(ps / "images" / f"{image}.png")
        (pred / f"{image}.txt").write_text("\n".join(pred_line(c, f, x, y, r, size) for c, f, x, y, r in objs)
                                           + ("\n" if objs else ""), encoding="utf-8")
        rows.append(f"{image},images/{image}.png,t_{image}.png,{(labels or {}).get(image, 'S1')},1.0,tester,upload")
    (ps / "images.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (ps / "request.json").write_text(json.dumps({
        "ps": ps_id, "owner": "tester", "model": MODEL, "conf": 0.25, "max_det": 1000, "imgsz": None,
        "source": {"kind": "upload"}, "created": "2026-11-06T10:00:00"}), encoding="utf-8")
    (ps / "status.json").write_text(json.dumps({"state": "done", "n_done": len(images), "n_total": len(images)}),
                                    encoding="utf-8")
    return ps


def save_edit(pkg: Path, file: str, edit: dict):
    (pkg / "edits").mkdir(exist_ok=True)
    (pkg / "edits" / f"{file}.json").write_text(json.dumps({"edit": edit}), encoding="utf-8")


def dec(v):
    return {"v": v, "t": 1, "who": "rev1"}


# A: three confident EVs (r 22.5 -> about 45 nm) and one candidate; B: two candidates r 62 (about 124 nm); C: one.
A_OBJS = [(0, 0.9, 80, 80, 22.5), (0, 0.9, 200, 80, 22.5), (0, 0.9, 320, 80, 22.5), (0, 0.30, 80, 300, 30)]
B_OBJS = [(0, 0.40, 100, 300, 62), (0, 0.35, 300, 300, 62)]
C_OBJS = [(0, 0.30, 200, 200, 30)]


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="p2_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)


class KnownAnswer(Tmp):
    def build(self):
        ps = make_ps(self.tmp, {"0001": A_OBJS, "0002": B_OBJS, "0003": C_OBJS})
        pkg = pp.build(ps, self.tmp / "review")
        save_edit(pkg, "0002.png", {"dec": {"p0": dec("add:EV")}})        # B: accept p0, p1 undecided
        save_edit(pkg, "0003.png", {"reviewed": True, "dec": {}})         # C: Reviewed, candidate undecided
        return ps, pkg

    def test_unsure_open_is_two(self):
        ps, pkg = self.build()
        rev = pp.reviewed_polygons(pkg, write=False)
        self.assertEqual([rev["images"][i]["n_candidates_open"] for i in ("0001", "0002", "0003")], [1, 1, 1])
        res = compute(ps, rev)
        s = res["samples"]["S1"]
        self.assertEqual(s["n_unsure_open"], 2)          # NOT 3 (open count everywhere), 1 (edited only), 4 (all)
        self.assertEqual(s["n_checked"], 2)
        self.assertEqual(s["n_images"], 3)
        self.assertEqual(s["ev_total"], 4)               # 3 confident + B's accepted candidate
        self.assertEqual(sum(s["diam_hist"]["counts"]), s["ev_total"])
        counts = s["diam_hist"]["counts"]
        self.assertEqual((counts[4], counts[12]), (3, 1), counts)   # ~45 nm x3, ~124 nm x1

    def test_without_a_package_candidates_come_from_the_predictions(self):
        ps, _ = self.build()
        s = compute(ps)["samples"]["S1"]                 # no reviewed.json: every image untouched
        self.assertEqual((s["n_unsure_open"], s["n_checked"], s["ev_total"]), (1 + 2 + 1, 0, 3))

    def test_package_count_must_agree_with_predictions_for_untouched(self):
        ps, pkg = self.build()
        rev = pp.reviewed_polygons(pkg, write=False)
        for i in rev["images"]:
            rev["images"][i]["edited"] = rev["images"][i]["reviewed"] = False
            rev["images"][i].pop("n_candidates_open")
        self.assertEqual(compute(ps, rev)["samples"]["S1"]["n_unsure_open"], 4)   # falls back to the .txt count

    def test_old_reviewed_json_edited_image_is_unknown_not_zero(self):
        ps, pkg = self.build()
        rev = pp.reviewed_polygons(pkg, write=False)
        for v in rev["images"].values():
            v.pop("n_candidates_open")
        self.assertIsNone(compute(ps, rev)["samples"]["S1"]["n_unsure_open"])

    def test_csv_columns_unchanged_and_hist_in_json_only(self):
        ps, pkg = self.build()
        res = compute(ps, pp.reviewed_polygons(pkg, write=False))
        out = write_results(res, self.tmp / "out")
        header = (out / "per_sample.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
        self.assertEqual(header, ["sample_label"] + C.SAMPLE_FIELDS)
        for f in ("diam_hist", "n_unsure_open", "n_checked"):
            self.assertNotIn(f, header)
            self.assertNotIn(f, C.SAMPLE_FIELDS)
        self.assertEqual(res["schema"], "results/1")
        stored = json.loads((out / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["samples"]["S1"]["diam_hist"]["bin_nm"], 10)
        self.assertEqual(len(stored["samples"]["S1"]["diam_hist"]["counts"]), 51)


class Histogram(Tmp):
    def test_known_bins_and_overflow(self):
        h = C.diam_hist([0.0, 9.99, 10.0, 49.9, 50.0, 499.99, 500.0, 1234.0])
        self.assertEqual((h["bin_nm"], h["max_nm"], len(h["counts"])), (10, 500, 51))
        c = h["counts"]
        self.assertEqual((c[0], c[1], c[4], c[5], c[49], c[50]), (2, 1, 1, 1, 1, 2))
        self.assertEqual(sum(c), 8)

    def test_circles_of_known_area_land_in_known_bins(self):
        # r 22.5 -> ~45 nm (bin 4), r 62 -> ~123.6 (bin 12), r 270 -> ~539 (overflow); the 1200 px tile holds the big one
        ps = make_ps(self.tmp, {"0001": [(0, 0.9, 100, 100, 22.5), (0, 0.9, 300, 100, 62), (0, 0.9, 600, 600, 270)]},
                     sizes={"0001": 1200})
        s = compute(ps)["samples"]["S1"]
        c = s["diam_hist"]["counts"]
        self.assertEqual((c[4], c[12], c[50], sum(c), s["ev_total"]), (1, 1, 1, 3, 3))
        self.assertEqual(sum(1 for x in c if x), 3)

    def test_sample_without_evs_has_an_all_zero_histogram(self):
        ps = make_ps(self.tmp, {"0001": [(0, 0.3, 100, 100, 25)]})
        s = compute(ps)["samples"]["S1"]
        self.assertEqual((s["ev_total"], sum(s["diam_hist"]["counts"]), s["n_unsure_open"]), (0, 0, 1))


class PackageSide(Tmp):
    def test_data_js_has_label_conf_and_candidate_counts_by_decision(self):
        objs = [(0, 0.9, 80, 80, 22.5), (0, 0.4, 80, 300, 30), (0, 0.35, 200, 300, 30), (0, 0.3, 320, 300, 30),
                (0, 0.28, 320, 200, 30)]
        ps = make_ps(self.tmp, {"0001": objs, "0002": objs, "0003": []})
        pkg = pp.build(ps, self.tmp / "review")
        data = pp.read_datajs(pkg)
        self.assertEqual(data["label_conf"], pp.LABEL_CONF)
        self.assertEqual(data["cand_conf"], pp.CAND_CONF)
        self.assertEqual(len(data["images"][0]["pr"]), 4)
        # p0 accepted, p1 rejected, p2 undone (empty v = open again), p3 no decision
        save_edit(pkg, "0001.png", {"dec": {"p0": dec("add:EV"), "p1": dec("no"), "p2": dec("")}})
        rev = pp.reviewed_polygons(pkg)
        self.assertEqual(rev["images"]["0001"]["n_candidates_open"], 2)       # p2 and p3
        self.assertEqual(rev["images"]["0002"]["n_candidates_open"], 4)       # untouched: all
        self.assertEqual(rev["images"]["0003"]["n_candidates_open"], 0)       # no candidates
        self.assertEqual(rev["schema"], "reviewed/1")
        on_disk = json.loads((pkg / "reviewed.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["images"]["0001"]["n_candidates_open"], 2)

    def test_real_tool_save_format(self):
        """Decisions exactly as annotation/review_tool/index.html setDec() saves them: {v, t, who, pending[, p]},
        plus server_t once the server has merged them; undo = v null; Del of an accepted candidate = "no";
        a vertex-moved candidate = "add:<cls>" carrying its new outline in p."""
        objs = [(0, 0.9, 80, 80, 22.5)] + [(0, 0.45 - 0.02 * k, 60 + 70 * k, 300, 25) for k in range(5)]
        ps = make_ps(self.tmp, {"0001": objs})
        pkg = pp.build(ps, self.tmp / "review")
        im = pp.read_datajs(pkg)["images"][0]
        self.assertEqual(len(im["pr"]), 5)
        moved = [[v + (0.01 if i % 2 == 0 else 0.0) for i, v in enumerate(ring)] for ring in im["pr"][0]["p"]]
        real = {"t": 1_762_400_000_999, "reviewed": False, "dec": {
            "p0": {"v": "add:EV", "t": 1_762_400_000_100, "who": "rev1", "p": moved, "pending": True,
                   "server_t": 1_762_400_000_150},                                    # accepted, vertex moved
            "p1": {"v": "no", "t": 1_762_400_000_300, "who": "rev1", "server_t": 1_762_400_000_310},  # accepted, then Del
            "p2": {"v": None, "t": 1_762_400_000_500, "who": "rev1", "pending": True},                 # accepted, then undone
            "p4": {"v": "no", "t": 1_762_400_000_600, "who": "rev1", "server_t": 1_762_400_000_610}}}  # rejected; p3 untouched
        save_edit(pkg, "0001.png", real)
        rev = pp.reviewed_polygons(pkg, write=False)
        v = rev["images"]["0001"]
        self.assertTrue(v["edited"])
        self.assertEqual(v["n_candidates_open"], 2)                       # p2 (undone) and p3 (untouched)
        cand = [o for o in v["objects"] if o["conf"] is not None and o["conf"] < 0.5]
        self.assertEqual(len(cand), 1)                                    # only the moved one is an outline now
        self.assertEqual(cand[0]["source"], "human")                      # its vertices differ from the model's
        s = compute(ps, rev)["samples"]["S1"]
        self.assertEqual((s["n_unsure_open"], s["n_checked"], s["ev_total"]), (2, 1, 2))

    def test_cand_conf_pinned_equal(self):
        self.assertEqual((C.CAND_CONF, C.CONF), (pp.CAND_CONF, pp.LABEL_CONF))


if __name__ == "__main__":
    unittest.main()
