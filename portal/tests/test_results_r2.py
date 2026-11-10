"""Round 7 / R2 on the Results pages: rejected images and low conf on the page, and the streamed downloads
(Images, Model outlines, Reviewed labels, Everything). Synthetic sets in a temp predict root; the reviewed/1 source is
a hand-built dict with the additive R1 fields (R.REVIEWED hook), so no R1 code is needed.

  python -m unittest discover -s portal/tests -p test_results_r2.py -v

Known answer: the discriminating sample of results/tests/test_reject.py (A rejected, B 3 EVs one low conf plus a
low-conf non-EV, C 2 EVs): the page says "Counted EVs 5 (1 low conf) · 1 image rejected".
"""
import csv
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_results_module as M  # noqa: E402
import cjtest as T  # noqa: E402
from portal.modules import gallery as G  # noqa: E402
from portal.modules import results as R  # noqa: E402
import importlib  # noqa: E402
from results.compute import compute_and_write  # noqa: E402
from results.tests import test_p2 as TP  # noqa: E402
from results.tests import test_reject as TR  # noqa: E402

CC = importlib.import_module("results.compute")

PS = "ps_20261109_c10001"          # the discriminating set, owner = M.OWNER
PS2 = "ps_20261109_c10002"         # all images of S2 rejected
PS3 = "ps_20261109_c10003"         # no review package, no results.json
OTHER = "ps_20261109_c10004"       # someone else's
MISSING = "ps_20261109_c1ffff"


def preds():
    """Model predictions per image (cls, conf, cx, cy, r): confident, candidate (0.30) and below the floor (0.10)."""
    return {"0001": [(0, 0.9, 60, 60, 40), (0, 0.9, 200, 200, 25), (0, 0.30, 320, 320, 20)],
            "0002": [(0, 0.8, 60, 60, 25), (1, 0.7, 200, 200, 25), (0, 0.10, 320, 320, 20)],
            "0003": [(0, 0.95, 100, 100, 25)]}


def make(cls, ps_id, owner=M.OWNER, labels=None, with_pkg=True, a_ok=False):
    ps = TP.make_ps(cls.tmp, preds(), ps_id=ps_id, labels=labels)
    req = json.loads((ps / "request.json").read_text(encoding="utf-8"))
    req["owner"] = owner
    (ps / "request.json").write_text(json.dumps(req), encoding="utf-8")
    rows = list(csv.DictReader(open(ps / "images.csv", newline="", encoding="utf-8")))
    for r in rows:
        if r["image"] == "0001":
            r["nm_per_px"], r["nm_confirmed_by"] = ("1.0", "tester") if a_ok else ("2.0", "")   # bad unless a_ok
    with open(ps / "images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if with_pkg:
        pkg = cls.review / f"pred_{ps_id}"
        pkg.mkdir(parents=True)
        for f in ("review_package.json", "prediction_package.json"):
            (pkg / f).write_text("{}", encoding="utf-8")
    return ps


def doc_for(ps_id):
    d = TR.reviewed_doc()
    d["prediction_set"], d["package"] = ps_id, f"pred_{ps_id}"
    return d


class R2(M.Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.docs = {}
        R.REVIEWED = lambda pkg, write=False: cls.docs[Path(pkg).name[len("pred_"):]]
        for ps_id in (PS, PS2, OTHER):
            make(cls, ps_id, owner="other" if ps_id == OTHER else M.OWNER,
                 labels={"0001": "S1", "0002": "S1", "0003": "S2"} if ps_id == PS2 else None, a_ok=ps_id == PS2)
            cls.docs[ps_id] = doc_for(ps_id)
        cls.docs[PS2]["images"]["0001"].pop("rejected")
        cls.docs[PS2]["images"]["0001"]["note"] = None
        cls.docs[PS2]["images"]["0003"]["rejected"] = True
        make(cls, PS3, with_pkg=False)
        for ps_id in (PS, PS2):
            compute_and_write(cls.root / ps_id, cls.docs[ps_id])
        compute_and_write(cls.root / OTHER, cls.docs[OTHER])

    @classmethod
    def tearDownClass(cls):
        R.REVIEWED = None
        super().tearDownClass()

    def zip(self, ps_id, kind, who="owner"):
        st, h, body = self.get(f"/results/{ps_id}/download/{R.ZIP_KINDS[kind]}_{ps_id}.zip", who)
        self.assertEqual(st, 200, body[:300])
        self.assertEqual(T.hget(h, "content-disposition"), f'attachment; filename="{R.ZIP_KINDS[kind]}_{ps_id}.zip"')
        self.assertEqual(T.hget(h, "content-type"), "application/zip")
        return zipfile.ZipFile(io.BytesIO(body))


class Page(R2):
    def test_known_answer_line_and_table(self):
        page = self.page(PS, "owner")
        self.assertIn("Counted EVs 5 (1 low conf) · 1 image rejected", page)
        c = M.cells(page, "S1")
        self.assertEqual((c["n_images"][0], c["ev_total"][0], c["ev_per_image_mean"][0]), ("2", "5", "2.5"))
        self.assertEqual(c["n_images"][1], "1 rejected")
        self.assertIn("1 low conf", c["ev_total"][1])
        self.assertNotIn("cj-results-refused", page)
        rows = M.image_rows(page)
        self.assertEqual(rows["0001"]["chip"], "Rejected: thick ice")
        self.assertTrue(rows["0001"]["muted"])
        self.assertEqual(rows["0002"]["chip"], "Reviewed")
        self.assertIn("All 2 images reviewed by a person.", page)          # the rejected image is not in the base

    def test_note_with_markup_is_text(self):
        d = self.docs[PS]
        d["images"]["0001"]["reject_reason"] = "<img src=x onerror=alert(1)>"
        try:
            res = compute_and_write(self.root / PS, d)[0]
            self.assertIn("<img src=x", res["images"]["0001"]["reject_reason"])
            page = self.page(PS, "owner")
            self.assertNotIn("<img src=x", page)
            self.assertIn("&lt;img src=x", page)
        finally:
            d["images"]["0001"]["reject_reason"] = "thick ice"
            compute_and_write(self.root / PS, d)

    def test_all_rejected_sample_is_a_notice(self):
        page = self.page(PS2, "owner")
        self.assertIn('id="cj-results-allrej"', page)
        self.assertIn("All images rejected", page)
        c = M.cells(page, "S2")
        self.assertEqual((c["n_images"][0], c["n_images"][1], c["ev_total"][0]), ("0", "All images rejected", "–"))
        self.assertEqual(M.cells(page, "S1")["n_images"][0], "2")
        self.assertIn("Counted EVs", page)

    def test_old_results_without_the_fields_still_render(self):
        d = self.root / PS / "results" / "results.json"
        keep = d.read_text(encoding="utf-8")
        doc = json.loads(keep)
        for s in doc["samples"].values():
            s.pop("n_rejected"), s.pop("n_low_conf")
        for i in doc["images"].values():
            for f in ("rejected", "reject_reason", "note"):
                i.pop(f)
        d.write_text(json.dumps(doc), encoding="utf-8")
        try:
            page = self.page(PS, "owner")
            self.assertIn("Counted EVs 5", page)
            self.assertNotIn("rejected", page.split('id="cj-results-counts"')[1][:80])
        finally:
            d.write_text(keep, encoding="utf-8")


class Downloads(R2):
    def test_images_zip_needs_no_results_json(self):
        z = self.zip(PS3, "images", "owner")                     # PS3: never computed, no package
        self.assertEqual(sorted(z.namelist()), ["PROBLEMS.txt", "README.txt", "images.csv", "images/0001.png", "images/0002.png",
                                                 "images/0003.png"])
        self.assertEqual(z.read("images/0002.png"), (self.root / PS3 / "images" / "0002.png").read_bytes())
        self.assertEqual(z.testzip(), None)
        self.assertIn("0001.png", z.read("images.csv").decode())

    def test_outlines_zip(self):
        z = self.zip(PS3, "outlines", "owner")
        names = z.namelist()
        self.assertIn("annotations_model.coco.json", names)
        coco = json.loads(z.read("annotations_model.coco.json"))
        self.assertEqual([i["file_name"] for i in coco["images"]], ["images/0001.png", "images/0002.png", "images/0003.png"])
        # 0001: 0.9, 0.9, 0.30 kept; 0002: 0.8, 0.7 kept (0.10 is below the floor); 0003: 0.95
        self.assertEqual(len(coco["annotations"]), 3 + 2 + 1)
        self.assertEqual(sorted(a["score"] for a in coco["annotations"]), [0.3, 0.7, 0.8, 0.9, 0.9, 0.95])
        self.assertEqual({c["name"] for c in coco["categories"]} >= {"EV", "non-EV"}, True)
        self.assertEqual(len(z.read("labels_model/0001.txt").decode().splitlines()), 3)
        self.assertEqual(len(z.read("labels_model/0002.txt").decode().splitlines()), 2)
        a = coco["annotations"][0]
        self.assertEqual(a["image_id"], 1)
        self.assertEqual(len(a["segmentation"][0]) % 2, 0)
        # YOLO-seg: no score column: class + 2 numbers per point (48 points)
        self.assertEqual(len(z.read("labels_model/0003.txt").decode().split()), 1 + 96)
        # areas: r = 25 circle (48-gon) about pi r^2
        circle = [x for x in coco["annotations"] if x["image_id"] == 3][0]
        self.assertAlmostEqual(circle["area"], 3.14159 * 25 * 25, delta=40)

    def test_reviewed_zip_polygons_equal_the_source(self):
        z = self.zip(PS, "reviewed", "owner")
        self.assertEqual(sorted(n for n in z.namelist() if "/" not in n),
                         ["PROBLEMS.txt", "README.txt", "annotations_reviewed.coco.json", "provenance.csv",
                          "rejected_images.csv"])
        self.assertEqual(z.namelist()[-1], "PROBLEMS.txt")                                    # always the last entry
        coco = json.loads(z.read("annotations_reviewed.coco.json"))
        src = R.reviewed_source(self.cfg, PS)
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        names = {i["id"]: i["file_name"] for i in coco["images"]}
        # a reject changes the counts, never the label data: A (rejected) is in the COCO, marked, and has its YOLO txt
        self.assertEqual(sorted(names.values()), ["images/0001.png", "images/0002.png", "images/0003.png"])
        flags = {i["file_name"]: (i.get("rejected"), i.get("reject_reason")) for i in coco["images"]}
        self.assertEqual(flags, {"images/0001.png": (True, "thick ice"), "images/0002.png": (None, None),
                                 "images/0003.png": (None, None)})
        self.assertEqual(len(z.read("labels_reviewed/0001.txt").decode().splitlines()), 7)
        got = {}
        for a in coco["annotations"]:
            got.setdefault(names[a["image_id"]], []).append(a)
        for image in ("0001", "0002", "0003"):
            want = src["images"][image]["objects"]
            have = got[f"images/{image}.png"]
            self.assertEqual(len(have), len(want))
            for a, o in zip(have, want):
                seg = a["segmentation"][0]
                self.assertEqual(len(seg), 2 * len(o["poly"]))
                for k, (x, y) in enumerate(o["poly"]):
                    self.assertAlmostEqual(seg[2 * k], x, delta=0.5)
                    self.assertAlmostEqual(seg[2 * k + 1], y, delta=0.5)
        # the one COCO rule: EV + low conf -> 'Low conf EV'; the low-conf non-EV keeps its class + attributes.low_conf
        b = got["images/0002.png"]
        self.assertEqual([cats[a["category_id"]] for a in b], ["EV", "Low conf EV", "EV", "non-EV"])
        self.assertNotIn("low_conf", (b[1].get("attributes") or {}))
        self.assertEqual(b[1]["attributes"]["note"], "blurry edge")
        self.assertIs(b[3]["attributes"]["low_conf"], True)
        self.assertNotIn("low_conf", b[0].get("attributes") or {})
        # YOLO rows are normalised; original class ids (a low-conf EV trains as an EV)
        rows = z.read("labels_reviewed/0002.txt").decode().splitlines()
        self.assertEqual([r.split()[0] for r in rows], ["0", "0", "0", "1"])
        for v in rows[0].split()[1:]:
            self.assertTrue(0.0 <= float(v) <= 1.0)
        rej =list(csv.DictReader(io.StringIO(z.read("rejected_images.csv").decode())))
        self.assertEqual([(r["image"], r["reject_reason"], r["note"]) for r in rej],
                         [("0001", "thick ice", "'=HYPERLINK(\"http://x\")")])               # formula guard
        prov = {r["image"]: r for r in csv.DictReader(io.StringIO(z.read("provenance.csv").decode()))}
        self.assertEqual({k: v["status"] for k, v in prov.items()},
                         {"0001": "rejected", "0002": "reviewed", "0003": "reviewed"})
        self.assertEqual((prov["0002"]["n_objects"], prov["0002"]["n_low_conf"]), ("4", "2"))

    def test_provenance_marks_model_only_and_edited(self):
        d = self.docs[PS]
        keep = json.dumps(d)
        try:
            d["images"]["0002"].update(reviewed=False, edited=True)
            d["images"]["0003"].update(reviewed=False, edited=False)
            z = self.zip(PS, "reviewed", "owner")
            prov = {r["image"]: r["status"] for r in csv.DictReader(io.StringIO(z.read("provenance.csv").decode()))}
            self.assertEqual(prov, {"0001": "rejected", "0002": "edited", "0003": "model-only"})
        finally:
            self.docs[PS] = json.loads(keep)

    def test_reviewed_is_built_at_download_time_not_from_a_stale_file(self):
        pkg = self.review / f"pred_{PS}"
        (pkg / "reviewed.json").write_text(json.dumps({"stale": True}), encoding="utf-8")
        coco = json.loads(self.zip(PS, "reviewed", "owner").read("annotations_reviewed.coco.json"))
        self.assertEqual(len(coco["annotations"]), 7 + 4 + 2)
        d = self.docs[PS]
        d["images"]["0003"]["objects"] = d["images"]["0003"]["objects"][:1]               # a new edit
        try:
            coco = json.loads(self.zip(PS, "reviewed", "owner").read("annotations_reviewed.coco.json"))
            self.assertEqual(len(coco["annotations"]), 7 + 4 + 1)
        finally:
            self.docs[PS] = doc_for(PS)
            (pkg / "reviewed.json").unlink()

    def test_coco_rule_decides_on_the_final_class(self):
        """MV / inner / Incomplete with low conf keep their class + attributes.low_conf; only a final EV (or
        'Low conf EV') with low conf becomes 'Low conf EV'. An unlabelled model EV with low conf is an EV."""
        d = self.docs[PS]
        keep = json.dumps(d)
        try:
            objs = d["images"]["0003"]["objects"]
            for label, low in (("MV", True), ("inner", True), ("Incomplete", True), ("EV", True), ("EV", False),
                               ("Low conf EV", True), (None, True)):
                o = TR.obj(100, 100, 20, rc=label, low=low)
                if label is None:
                    o["reviewer_label"] = o["reviewer_class"] = None
                if label == "Incomplete":
                    o["reviewer_class"] = "EV"
                objs.append(o)
            z = self.zip(PS, "reviewed", "owner")
            coco = json.loads(z.read("annotations_reviewed.coco.json"))
            cats = {c["id"]: c["name"] for c in coco["categories"]}
            three = [a for a in coco["annotations"] if a["image_id"] == 3][2:]          # the 7 added after C's two
            got = [(cats[a["category_id"]], (a.get("attributes") or {}).get("low_conf")) for a in three]
            self.assertEqual(got, [("EV", True), ("EV", True), ("EV", True), ("Low conf EV", None), ("EV", None),
                                   ("Low conf EV", None), ("Low conf EV", None)])
            self.assertEqual([(a.get("attributes") or {}).get("reviewer_label") for a in three][:3],
                             ["MV", "inner", "Incomplete"])
        finally:
            self.docs[PS] = json.loads(keep)

    def test_images_csv_in_the_zips_is_guarded(self):
        ps = make(type(self), "ps_20261109_c10008", with_pkg=False)
        rows = list(csv.DictReader(open(ps / "images.csv", newline="", encoding="utf-8")))
        rows[0]["orig_name"] = "=HYPERLINK(\"http://x\")"
        with open(ps / "images.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        for kind in ("images", "everything"):
            z = self.zip("ps_20261109_c10008", kind, "owner")
            got = list(csv.DictReader(io.StringIO(z.read("images.csv").decode())))
            self.assertEqual(got[0]["orig_name"], "'=HYPERLINK(\"http://x\")")
            self.assertEqual([r["image"] for r in got], ["0001", "0002", "0003"])

    def test_problems_found_while_streaming_land_in_problems_txt(self):
        d = self.docs[PS]
        keep = json.dumps(d)
        try:
            d["images"]["0003"]["objects"].append({"cls": 0, "poly": [[1, 1], [2, 2]], "conf": 0.9})   # 2 points
            z = self.zip(PS, "reviewed", "owner")
            self.assertEqual(z.namelist()[-1], "PROBLEMS.txt")
            self.assertIn("0003: an outline with fewer than 3 points was left out", z.read("PROBLEMS.txt").decode())
        finally:
            self.docs[PS] = json.loads(keep)
        self.assertEqual(self.zip(PS3, "images", "owner").read("PROBLEMS.txt").decode().strip(), "No problems.")

    def test_everything_survives_a_review_that_cannot_be_read(self):
        old = R.REVIEWED

        def boom(pkg, write=False):
            raise ValueError("half-written edit")
        R.REVIEWED = boom
        try:
            z = self.zip(PS, "everything", "owner")                           # 200, not a 500
            names = set(z.namelist())
            self.assertIn("images/0001.png", names)
            self.assertIn("annotations_model.coco.json", names)
            self.assertIn("results/per_sample.csv", names)
            self.assertNotIn("annotations_reviewed.coco.json", names)
            self.assertIn("review could not be read", z.read("PROBLEMS.txt").decode())
            st, _, body = self.get(f"/results/{PS}/download/reviewed_labels_{PS}.zip", "owner")
            self.assertEqual(st, 404)
            self.assertIn(b"could not be read", body)
        finally:
            R.REVIEWED = old

    def test_readme_documents_the_new_columns(self):
        readme = self.zip(PS, "everything", "owner").read("README.txt").decode()
        for want in ("n_rejected", "n_low_conf", "reject_reason", "note (per_image)", "results/per_sample.csv",
                     "n_images 0", "PROBLEMS.txt", "NOT left out of the labels"):
            self.assertIn(want, readme)
        self.assertNotIn("one row per counted sample", readme)
        tables = zipfile.ZipFile(io.BytesIO(self.get(f"/results/{PS}/download/results_{PS}.zip", "owner")[2]))
        t = tables.read("README.txt").decode()
        for want in ("n_rejected", "n_low_conf", "reject_reason", "n_images 0"):
            self.assertIn(want, t)

    def test_counted_images_only_in_provenance_and_refused_notice(self):
        res = json.loads((self.root / PS / "results" / "results.json").read_text(encoding="utf-8"))
        prov = R.provenance(res, {"id": PS})
        self.assertEqual(prov["Checked by a person"], "2 of 2 counted images reviewed, 0 more edited")
        self.assertIn("reviewed=2/2 counted images", R.csv_header(res, {"id": PS}).decode())
        self.assertIn("# rejected images (not counted)=1", R.csv_header(res, {"id": PS}).decode())
        res["refused"] = "x"
        res["samples"] = {}
        self.assertNotIn("0001", R.refused_notice(res))
        self.assertIn("(2 images", R.refused_notice(res))                    # B and C; the rejected A is not counted

    def test_outlines_link_hidden_without_predictions(self):
        import shutil as _sh
        ps = make(type(self), "ps_20261109_c10007", with_pkg=False)
        _sh.rmtree(ps / "p")
        page = self.page("ps_20261109_c10007", "owner")
        self.assertNotIn("model_outlines_", page)
        self.assertIn("images_ps_20261109_c10007.zip", page)
        self.assertIn("model_outlines_" + PS, self.page(PS, "owner"))

    def test_reviewed_without_a_package_is_a_plain_404(self):
        st, _, body = self.get(f"/results/{PS3}/download/reviewed_labels_{PS3}.zip", "owner")
        self.assertEqual(st, 404)
        self.assertIn(b"no review package", body.lower())

    def test_everything_zip_and_stale_flag(self):
        z = self.zip(PS, "everything", "owner")
        names = set(z.namelist())
        for want in ("README.txt", "images/0001.png", "images.csv", "annotations_model.coco.json",
                     "annotations_reviewed.coco.json", "rejected_images.csv", "provenance.csv",
                     "results/per_sample.csv", "results/per_image.csv", "results/results.json"):
            self.assertIn(want, names)
        readme = z.read("README.txt").decode()
        self.assertIn("up to date", readme)
        self.assertNotIn("OUT OF DATE", readme)
        # an edit saved after the results were computed
        done = (self.root / PS / "results" / "results.json").stat().st_mtime
        edit = self.review / f"pred_{PS}" / "edits" / "0002.png.json"
        edit.parent.mkdir(exist_ok=True)
        edit.write_text("{}", encoding="utf-8")
        os.utime(edit, (done + 300, done + 300))
        try:
            self.assertIn("RESULTS ARE OUT OF DATE", self.zip(PS, "everything", "owner").read("README.txt").decode())
        finally:
            edit.unlink()
        # no results.json: still works, README says so
        z3 = self.zip(PS3, "everything", "owner")
        self.assertIn("not computed yet", z3.read("README.txt").decode())
        self.assertFalse([n for n in z3.namelist() if n.startswith("results/")])

    def test_missing_image_is_listed_not_fatal(self):
        ps = make(type(self), "ps_20261109_c10009", with_pkg=False)
        (ps / "images" / "0002.png").unlink()
        z = self.zip("ps_20261109_c10009", "images", "owner")
        self.assertEqual(sorted(n for n in z.namelist() if n.startswith("images/")), ["images/0001.png", "images/0003.png"])
        self.assertIn("0002: file is missing", z.read("README.txt").decode())

    def test_same_404_for_missing_and_not_yours(self):
        outs = []
        for ps_id in (OTHER, MISSING):
            for kind in R.ZIP_KINDS:
                st, h, body = self.get(f"/results/{ps_id}/download/{R.ZIP_KINDS[kind]}_{ps_id}.zip", "owner")
                self.assertEqual(st, 404, (ps_id, kind))
                self.assertIsNone(T.hget(h, "content-disposition"))
                outs.append(body.replace(ps_id.encode(), b"X"))
        self.assertEqual(len(set(outs)), 1)
        # the admin may; a name that is not this set's is a 404
        self.assertEqual(self.get(f"/results/{OTHER}/download/images_{OTHER}.zip", "admin")[0], 200)
        self.assertEqual(self.get(f"/results/{PS}/download/images_{OTHER}.zip", "owner")[0], 404)

    def test_two_streams_at_most(self):
        held = [G.take_slot("pujan"), G.take_slot("pujan")]
        try:
            self.assertTrue(all(held))
            st, h, _ = self.get(f"/results/{PS3}/download/images_{PS3}.zip", "owner")
            self.assertEqual(st, 429)
            self.assertEqual(T.hget(h, "retry-after"), "30")
        finally:
            for s in held:
                s.release()
        self.assertEqual(self.get(f"/results/{PS3}/download/images_{PS3}.zip", "owner")[0], 200)

    def test_download_links_on_the_page(self):
        page = self.page(PS, "owner")
        for kind in R.ZIP_KINDS:
            self.assertIn(f"/results/{PS}/download/{R.ZIP_KINDS[kind]}_{PS}.zip", page)
        self.assertNotIn("reviewed_labels", self.page(PS3, "owner"))                     # no package: no link
        self.assertIn(f"images_{PS3}.zip", self.page(PS3, "owner"))                       # no results yet: still there


class Pinned(unittest.TestCase):
    def test_constants_and_guard_equal_the_engine(self):
        self.assertEqual(R.CAND_CONF, CC.CAND_CONF)
        for t in ("=1", "+1", "-1", "@a", "\tx", "\rx", "ok", "", "a=b"):
            self.assertEqual(R.guard_cell(t), CC.guard_cell(t))

    def test_csv_text_guards(self):
        self.assertEqual(R.csv_text(["a", "b"], [["=x", None], [True, 3]]), "a,b\n'=x,\ntrue,3\n")


if __name__ == "__main__":
    unittest.main()
