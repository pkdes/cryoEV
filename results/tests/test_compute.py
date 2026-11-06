"""results.compute against the CJ golden fixtures (BUILD_CONTRACTS.md section 6) plus controls.

    python -m unittest discover -s results/tests        (from the repo root)

The fixtures and their hand-derived expectations are tests/fixtures/cj_golden (make_golden.py explains the
scene); they are read, never written: every test that changes an input works on a temp copy.
"""

import copy
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

from results import compute, write_results  # noqa: E402

GOLDEN = REPO / "tests" / "fixtures" / "cj_golden"
PS = GOLDEN / "predict" / "ps_20261105_abc123"
REVIEWED = GOLDEN / "reviewed.json"
MODEL = "v5_evnonev_y8m_20261029"


def load(name):
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


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
        tc.assertEqual(set(g), set(e), s)
        for f, v in e.items():
            if f.startswith("diam_nm_"):
                tc.assertLess(abs(g[f] / v - 1), tol, f"{s} {f}: {g[f]} vs {v}")
            else:
                tc.assertEqual(g[f], v, f"{s} {f}")


def without_time(res):
    return {k: v for k, v in res.items() if k != "computed"}


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


class Golden(unittest.TestCase):
    def test_noedit_equals_expected(self):
        assert_matches(self, compute(PS), load("expected_results_noedit.json"))

    def test_reviewed_equals_expected(self):
        assert_matches(self, compute(PS, REVIEWED), load("expected_results_reviewed.json"))

    def test_reviewed_dict_same_as_path(self):
        self.assertEqual(without_time(compute(PS, load("reviewed.json"))), without_time(compute(PS, REVIEWED)))

    def test_reviewed_json_of_unedited_predictions_equals_noedit(self):
        """The two input paths agree: a reviewed.json holding every prediction line (pixel polygons, nothing
        reviewed, the conf-0.30 outline D included) gives the no-edit result exactly, diameters too."""
        rv = {"schema": "reviewed/1", "package": "pred_x", "prediction_set": PS.name, "model": MODEL, "images": {}}
        for txt in sorted((PS / "p" / MODEL / "predictions").glob("*.txt")):
            with Image.open(PS / "images" / f"{txt.stem}.png") as im:
                w, h = im.size
            objs = []
            for line in txt.read_text(encoding="utf-8").splitlines():
                p = line.split()
                xy = [float(v) for v in p[2:]]
                objs.append({"cls": int(p[0]), "conf": float(p[1]), "source": "model", "reviewer_class": None,
                             "poly": [[xy[i] * w, xy[i + 1] * h] for i in range(0, len(xy), 2)]})
            rv["images"][txt.stem] = {"reviewed": False, "reviewed_by": [], "objects": objs}
        self.assertEqual(without_time(compute(PS, rv)), without_time(compute(PS)))


class ClassBeforeRoles(unittest.TestCase):
    def test_ev_inside_nonev_outline_is_standalone(self):
        """J (class 0) sits inside I (class 1): the class-0 filter runs before roles, so J is an EV."""
        img = compute(PS)["images"]["0002"]
        self.assertEqual((img["ev"], img["multilayer"], img["inner"], img["nonev"]), (3, 0, 0, 2))

    def test_control_containment_sees_j_inside_i(self):
        """Control with a known answer: make I an EV (reviewer reclass) and the same geometry must call I
        multilayer and J its inner layer. If this failed, the test above would prove nothing."""
        rv = load("reviewed.json")
        objs = rv["images"]["0002"]["objects"]
        i_obj = max(objs, key=lambda o: max(x for x, _ in o["poly"]) - min(x for x, _ in o["poly"]))   # I is r45
        self.assertEqual(i_obj["cls"], 1)
        i_obj["reviewer_class"] = "EV"
        img = compute(PS, rv)["images"]["0002"]
        self.assertEqual((img["ev"], img["multilayer"], img["inner"], img["nonev"]), (2, 1, 1, 1))

    def test_reviewer_nonev_wins_over_class(self):
        rv = load("reviewed.json")
        for o in rv["images"]["0002"]["objects"]:
            if o["reviewer_class"] == "EV":   # J
                o["reviewer_class"] = "nonEV"
        img = compute(PS, rv)["images"]["0002"]
        self.assertEqual((img["ev"], img["nonev"]), (1, 3))

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
            self.assertEqual(rows["S2"]["pct_multilayer_reviewer"], "")
            with (out / "per_image.csv").open(newline="", encoding="utf-8") as fh:
                imgs = {r["image"]: r for r in csv.DictReader(fh)}
            self.assertEqual([imgs[i]["reviewed"] for i in ("0001", "0002", "0003")], ["False", "True", "False"])

    def test_cli(self):
        with TempPS() as t:
            r = subprocess.run([sys.executable, "-m", "results", "--ps", str(t.ps), "--reviewed", str(REVIEWED)],
                               cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            got = json.loads((t.ps / "results" / "results.json").read_text(encoding="utf-8"))
            assert_matches(self, got, load("expected_results_reviewed.json"))
            self.assertTrue((t.ps / "results" / "per_image.csv").is_file())
            t.edit_images_csv("0002", nm_per_px="0.7362")
            r = subprocess.run([sys.executable, "-m", "results", "--ps", str(t.ps)],
                               cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("REFUSED", r.stderr)

    def test_works_without_cryoai_folder(self):
        """add_layer_roles -> layer_count_stats resolves the CryoAI folder at import; counting must not need it."""
        env = dict(os.environ, CRYOEV_CRYOAI_ROOT=str(REPO / "no_such_cryoai_folder"))
        code = ("import sys, json; from results import compute; r = compute(sys.argv[1]); "
                "print(json.dumps([r['samples']['S1']['ev_total'], r['samples']['S2']['ev_total']]))")
        r = subprocess.run([sys.executable, "-c", code, str(PS)], cwd=REPO, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout.strip().splitlines()[-1]), [5, 2])


if __name__ == "__main__":
    unittest.main()
