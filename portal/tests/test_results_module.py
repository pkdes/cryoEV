"""
Results module (portal/modules/results.py): the golden prediction set (tests/fixtures/cj_golden) copied into a temp
predict root, results computed with the real results engine, and the page's numbers checked against the
hand-derived expected_results_noedit.json / expected_results_reviewed.json; the honesty banner and the reviewer
columns; a refused sample; roles and ownership (one 404 for missing and not-yours); the compute button running in a
thread (a fake slow compute: the page is not blocked, the poll endpoint reports progress, one at a time); the review
package as the compute's source (a control: an unedited package gives the no-edit numbers and is not written);
downloads as attachments; and the page in a browser (Playwright: 390 and 1280 px, light and dark, no horizontal
scroll, same-origin requests only, the compute -> poll -> reload flow). Screenshots go to $CJ_SHOT_DIR when set
(else a temp folder, removed afterwards).

Temp roster / registry / predict root only; never the real ones.

  python -m unittest discover -s portal/tests -v
"""
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import results as R  # noqa: E402
from results.compute import compute_and_write  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

GOLDEN = T.REPO / "tests" / "fixtures" / "cj_golden"
GPS = "ps_20261105_abc123"
OWNER = "pujan"                                  # request.json owner in the golden set
XSS = '<img src=x onerror=alert(1)>'

NOEDIT = GPS                                     # results from model predictions
REVIEWED = "ps_20261105_abc124"                  # results with the golden reviewed.json (0002 reviewed)
REFUSED = "ps_20261105_bbb222"                   # S1 mixes nm/px -> refused; S2 (label = XSS) still shown
EDITED = "ps_20261105_fff666"                    # 0001 edited but not marked reviewed
RANGE = "ps_20261105_f1a001"                     # S1 at 1.4724 / 1.4734 nm/px (0.07 %): one scale, counted
OVER = "ps_20261105_f1a002"                      # S1 at 1.4724 / 1.4890 nm/px (1.12 %): refused
TODO1, TODO2, TODO3 = "ps_20261105_ccc331", "ps_20261105_ccc332", "ps_20261105_ccc334"   # not computed
PKG = "ps_20261105_ccc333"                       # not computed; has an (unedited) review package
RUNNING = "ps_20261105_ddd444"                   # predictions still running
OTHERS = "ps_20261105_eee555"                    # someone else's set, with results
MISSING = "ps_20261105_000000"


sys.path.append(str(Path(__file__).resolve().parents[2] / "tests"))
from cryoev_testkit import browser  # noqa: E402  (the gate skips these when no UI file changed)


def golden_reviewed(ps_id: str, edited_0001=False) -> dict:
    doc = json.loads((GOLDEN / "reviewed.json").read_text(encoding="utf-8"))
    doc["prediction_set"] = ps_id
    if edited_0001:
        doc["images"]["0001"]["edited"] = True
    return doc


def make_set(root: Path, ps_id: str, owner=OWNER, nm_0002=None, label_s2=None, state="done") -> Path:
    d = root / ps_id
    shutil.copytree(GOLDEN / "predict" / GPS, d)
    req = json.loads((d / "request.json").read_text(encoding="utf-8"))
    req.update(ps=ps_id, owner=owner)
    (d / "request.json").write_text(json.dumps(req, indent=1), encoding="utf-8")
    if nm_0002 or label_s2:
        with open(d / "images.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["image"] == "0002" and nm_0002:
                r["nm_per_px"] = nm_0002
            if r["sample_label"] == "S2" and label_s2:
                r["sample_label"] = label_s2
        with open(d / "images.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if state != "done":
        st = json.loads((d / "status.json").read_text(encoding="utf-8"))
        st.update(state=state, n_done=1)
        (d / "status.json").write_text(json.dumps(st), encoding="utf-8")
    return d


def tree_md5(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def cells(page: str, sample: str) -> dict:
    m = re.search(r'<tr data-sample="%s">(.*?)</tr>' % re.escape(sample), page, re.S)
    assert m, sample
    out = {}
    for k, v, flag, sub in re.findall(r'data-k="(\w+)"><span class="cj-results-v">(.*?)</span>'
                                      r'(?: <span class="cj-results-chip[^"]*">([^<]*)</span>)?'
                                      r'(?:<small class="cj-results-sub">(.*?)</small>)?', m.group(1)):
        out[k] = (v, sub)
        if flag:
            out[k + ".flag"] = flag
    return out


def image_rows(page: str) -> dict:
    out = {}
    for img, attrs, body in re.findall(r'<tr data-image="(\d+)"([^>]*)>(.*?)</tr>', page, re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", body, re.S)
        chip = re.sub(r"<[^>]+>", "", tds[2])
        out[img] = {"sample": tds[1], "chip": chip, "ev": int(tds[3]), "multilayer": int(tds[4]),
                    "inner": int(tds[5]), "nonev": int(tds[6]), "muted": "cj-results-off" in attrs}
    return out


def html_unescape(page: str) -> str:
    import html
    return html.unescape(page)


def num(text: str) -> float:
    return float(re.match(r"[-\d.]+", text).group(0))


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.tmp = Path(tempfile.mkdtemp(prefix="cj_results_"))
        cls.root, cls.review = cls.tmp / "predict", cls.tmp / "review"
        cls.root.mkdir()
        cls.review.mkdir()
        for name in (".incoming", ".locks"):
            (cls.root / name).mkdir()
        (cls.root / "predict.lock").write_text("", encoding="utf-8")
        compute_and_write(make_set(cls.root, NOEDIT))
        compute_and_write(make_set(cls.root, REVIEWED), golden_reviewed(REVIEWED))
        compute_and_write(make_set(cls.root, REFUSED, nm_0002="2.0", label_s2=XSS))
        compute_and_write(make_set(cls.root, EDITED), golden_reviewed(EDITED, edited_0001=True))
        compute_and_write(make_set(cls.root, OTHERS, owner="other"))
        compute_and_write(make_set(cls.root, RANGE, nm_0002="1.4734"))
        compute_and_write(make_set(cls.root, OVER, nm_0002="1.4890"))
        for ps in (TODO1, TODO2, TODO3, PKG):
            make_set(cls.root, ps)
        make_set(cls.root, RUNNING, state="running")
        (cls.review / f"pred_{REVIEWED}").mkdir()
        (cls.review / f"pred_{REVIEWED}" / "spotcheck.json").write_text(json.dumps({"n": 2}), encoding="utf-8")
        cls.keys = {"admin": cls.env.key, "owner": A.add_key(cls.env.roster, OWNER, "basic"),
                    "advanced": A.add_key(cls.env.roster, "adv", "advanced"),
                    "basic": A.add_key(cls.env.roster, "bas", "basic"),
                    "other": A.add_key(cls.env.roster, "other", "basic")}
        cfg = cls.env.config()
        cfg.predict_root, cfg.review_root = cls.root, cls.review
        cls.cfg = cfg
        cls.p = T.Portal(cfg)
        cls.tok = {k: T.login(cls.p.port, v) for k, v in cls.keys.items()}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        R._jobs.clear()
        self.addCleanup(R._jobs.clear)

    def get(self, path, who="admin"):
        return T.request(self.p.port, "GET", path, T.cookie(self.tok[who]) if who else {})

    def post(self, path, who="admin", origin=True):
        h = T.cookie(self.tok[who])
        if origin is True:
            h["Origin"] = self.p.origin
        elif origin:
            h["Origin"] = origin
        return T.request(self.p.port, "POST", path, h, body=b"")

    def page(self, ps, who="admin") -> str:
        st, h, body = self.get(f"/results/{ps}/", who)
        self.assertEqual(st, 200, body[:300])
        self.assertEqual(T.hget(h, "cache-control"), "no-store")
        return body.decode("utf-8")

    def status(self, ps, who="admin") -> dict:
        st, _, body = self.get(f"/results/{ps}/api/status", who)
        self.assertEqual(st, 200)
        return json.loads(body)

    def wait_state(self, ps, states, timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = self.status(ps)
            if s["state"] in states:
                return s
            time.sleep(0.1)
        self.fail(f"{ps} never reached {states}: {s}")

    def patch_compute(self, fn):
        self.addCleanup(setattr, R, "COMPUTE", R.COMPUTE)
        R.COMPUTE = fn

    def clear_results(self, ps):
        self.addCleanup(lambda: shutil.rmtree(self.root / ps / "results", ignore_errors=True))


class Golden(Base):
    def check_samples(self, page, expected):
        for label, e in expected["samples"].items():
            with self.subTest(sample=label):
                c = cells(page, label)
                self.assertEqual(int(c["n_images"][0]), e["n_images"])
                self.assertEqual(c["n_reviewed"][0], f"{e['n_reviewed']} / {e['n_images']}")
                self.assertEqual(int(c["ev_total"][0]), e["ev_total"])
                self.assertAlmostEqual(num(c["ev_per_image_mean"][0]), e["ev_per_image_mean"], places=1)
                self.assertTrue(c["diam_nm_median"][0].endswith(" nm"))
                q1, q3 = (float(x) for x in c["diam_nm_median"][1].split("–"))
                for shown, want in ((num(c["diam_nm_median"][0]), e["diam_nm_median"]), (q1, e["diam_nm_q1"]),
                                    (q3, e["diam_nm_q3"])):
                    self.assertLessEqual(abs(shown - want), 0.02 * want + 0.5, (shown, want))   # ±2 % + rounding
                self.assertEqual(c["pct_multilayer_geometry"][0], f"{e['pct_multilayer_geometry']:g}%")
                self.assertEqual(int(c["nonev_total"][0]), e["nonev_total"])
        self.assertEqual(sorted(re.findall(r'<tr data-sample="([^"]+)">', page)), sorted(expected["samples"]))
        rows = image_rows(page)
        self.assertEqual(sorted(rows), sorted(expected["images"]))
        for img, e in expected["images"].items():
            r = rows[img]
            self.assertEqual((r["sample"], r["ev"], r["multilayer"], r["inner"], r["nonev"]),
                             (e["sample_label"], e["ev"], e["multilayer"], e["inner"], e["nonev"]), img)
            self.assertEqual(r["chip"], "Reviewed" if e["reviewed"] else "Model only")

    def test_noedit_numbers_equal_expected(self):
        exp = json.loads((GOLDEN / "expected_results_noedit.json").read_text(encoding="utf-8"))
        page = self.page(NOEDIT)
        self.check_samples(page, exp)
        self.assertNotIn("Multilayer (reviewer)", page)                       # no reviewed image: no columns
        self.assertIn("Model only: none of these images were checked by a person.", page)
        self.assertIn(f'href="/predict/{NOEDIT}/"', page)
        self.assertIn("Spot-check a suggested subset", page)
        self.assertNotIn("/review/pred_", page)                     # no review package: no review link
        self.assertIn(' nm</text>', page)                           # units on the chart ticks
        self.assertNotIn('id="cj-results-refused"', page)

    def test_reviewed_numbers_columns_and_banner(self):
        exp = json.loads((GOLDEN / "expected_results_reviewed.json").read_text(encoding="utf-8"))
        page = self.page(REVIEWED)
        self.check_samples(page, exp)
        self.assertIn('<th scope="col">Multilayer (reviewer)</th>', page)
        s1, s2 = cells(page, "S1"), cells(page, "S2")
        self.assertEqual(s1["pct_multilayer_reviewer"][0], f"{exp['samples']['S1']['pct_multilayer_reviewer']:g}%")
        res = json.loads((self.root / REVIEWED / "results" / "results.json").read_text(encoding="utf-8"))
        r1 = res["samples"]["S1"]
        self.assertEqual(s1["pct_multilayer_reviewer"][1],
                         f"n = {r1['reviewer_ev']} EVs on {r1['reviewed_images']} reviewed image")
        self.assertEqual(int(s1["reviewer_geometry_disagree"][0]), r1["reviewer_geometry_disagree"])
        self.assertEqual(s2["pct_multilayer_reviewer"], ("–", "none reviewed"))
        self.assertIn("<b>1 of 3 images reviewed.</b> The other 2 are model only.", page)
        self.assertIn("Spot-check the 2 suggested images", page)               # spotcheck.json n
        self.assertEqual(s1.get("pct_multilayer_reviewer.flag"), "small sample")     # reviewer_ev 2 < 20
        self.assertEqual(s1["reviewer_geometry_disagree"][1], "disagreement")        # 1: singular
        self.assertIn("EV = class 0 (EV)", page)
        self.assertIn("1 of 3 images reviewed, 0 more edited", page)          # provenance

    def test_edited_not_reviewed(self):
        page = self.page(EDITED)
        rows = image_rows(page)
        self.assertEqual((rows["0001"]["chip"], rows["0002"]["chip"], rows["0003"]["chip"]),
                         ("Edited", "Reviewed", "Model only"))
        self.assertIn("1 image has edits but was not marked reviewed", page)

    def test_refused_sample_notice_and_partial(self):
        page = self.page(REFUSED)
        notice = re.search(r'<div class="cj-results-banner cj-results-b-warn" role="alert" '
                           r'id="cj-results-refused">(.*?)</div>', page, re.S).group(1)
        self.assertIn("Some samples were not counted.", notice)
        self.assertIn("<b>S1</b> (2 images, 1 unsure outline still open)", notice)
        self.assertIn("mixes nm/px 1.4724, 2", notice)
        self.assertEqual(re.findall(r'<tr data-sample="([^"]+)">', page), [R.esc(XSS)])   # S2 still shown
        self.assertNotIn(XSS, page)                                                  # escaped everywhere
        self.assertIn(R.esc(XSS), page)
        rows = image_rows(page)
        self.assertEqual(len(rows), 3)
        for img in ("0001", "0002"):                                   # S1 refused: not counted, muted
            self.assertEqual((rows[img]["chip"], rows[img]["muted"]), ("Not counted (sample refused)", True))
        self.assertEqual((rows["0003"]["chip"], rows["0003"]["muted"]), ("Model only", False))   # control

    def test_pixel_size_range_within_tolerance(self):
        """Contract section 8: S1's two images at 1.4724 and 1.4734 nm/px (0.07 % apart) are counted as one sample;
        the range shows on the page, in the CSV # lines and the README. Control: the golden set (one value) shows
        a single value and no range note; 1.12 % apart is refused."""
        import importlib
        self.assertEqual(R.NM_TOLERANCE, importlib.import_module("results.compute").NM_TOLERANCE)
        page = self.page(RANGE)
        self.assertNotIn('id="cj-results-refused"', page)
        self.assertEqual(sorted(re.findall(r'<tr data-sample="([^"]+)">', page)), ["S1", "S2"])
        self.assertEqual(int(cells(page, "S1")["ev_total"][0]), 5)                   # same geometry as golden
        self.assertIn("S1 1.4724–1.4734; S2 1.4724 (varies within 1 % between images of a sample; each image's "
                      "diameters use its own value)", html_unescape(page))
        self.assertIn("Pixel size differs slightly between the images of S1 (1.4724–1.4734 nm/px), within 1 %",
                      html_unescape(page))
        body = self.get(f"/results/{RANGE}/download/per_sample.csv", "owner")[2].decode()
        self.assertIn("# nm_per_px=S1 1.4724–1.4734; S2 1.4724 (varies within 1 %", body)
        rows = {r["sample_label"]: r for r in csv.DictReader(x for x in body.splitlines() if not x.startswith("#"))}
        self.assertEqual((rows["S1"]["nm_per_px_min"], rows["S1"]["nm_per_px_max"]), ("1.4724", "1.4734"))
        z = zipfile.ZipFile(io.BytesIO(self.get(f"/results/{RANGE}/download/results_{RANGE}.zip", "owner")[2]))
        self.assertIn("Pixel size (nm per pixel): S1 1.4724–1.4734", z.read("README.txt").decode())
        golden = html_unescape(self.page(NOEDIT))                                     # control: one value
        self.assertIn("S1 1.4724; S2 1.4724</dd>", golden)
        self.assertNotIn("Pixel size differs slightly", golden)
        over = html_unescape(self.page(OVER))
        self.assertIn("mixes nm/px 1.4724, 1.489 (spread 1.1 % of the median, over the 1 % tolerance)", over)
        self.assertEqual(re.findall(r'<tr data-sample="([^"]+)">', over), ["S2"])

    def test_edited_only_headline(self):
        d = make_set(self.root, "ps_20261105_abc777")
        self.addCleanup(shutil.rmtree, d, True)
        rv = golden_reviewed("ps_20261105_abc777")
        rv["images"]["0002"].update(reviewed=False, edited=True)
        compute_and_write(d, rv)
        page = self.page("ps_20261105_abc777")
        self.assertIn("<b>Not reviewed; 1 image edited (edits are counted).</b>", page)
        self.assertNotIn("Model only: none", page)
        self.assertNotIn("Multilayer (reviewer)", page)
        self.assertEqual(image_rows(page)["0002"]["chip"], "Edited")

    def test_whole_set_refused(self):
        d = make_set(self.root, "ps_20261105_abc999", state="running")
        self.addCleanup(shutil.rmtree, d, True)
        compute_and_write(d)                       # refused whole: set not finished
        page = self.page("ps_20261105_abc999")
        self.assertIn("No results for this set.", page)
        self.assertIn("set not finished", page)
        self.assertNotIn('id="cj-results-samples"', page)


    def test_out_of_date_after_review_edits(self):
        ps_id = "ps_20261105_abc888"
        d = make_set(self.root, ps_id)
        self.addCleanup(shutil.rmtree, d, True)
        compute_and_write(d)
        pkg = self.review / f"pred_{ps_id}"
        (pkg / "edits").mkdir(parents=True)
        self.addCleanup(shutil.rmtree, pkg, True)
        (pkg / "review_package.json").write_text("{}", encoding="utf-8")
        edit = pkg / "edits" / "0001.png.json"
        edit.write_text("{}", encoding="utf-8")
        done = (d / "results" / "results.json").stat().st_mtime
        os.utime(pkg / "review_package.json", (done - 600, done - 600))

        os.utime(edit, (done - 300, done - 300))                  # control: edits older than the results
        page = self.page(ps_id)
        self.assertNotIn('id="cj-results-stale"', page)
        self.assertIn(f'href="/review/pred_{ps_id}/"', page)      # package exists: "Review these images"
        self.assertIn("Review these images", page)
        self.assertFalse(self.status(ps_id)["out_of_date"])
        self.assertNotIn("Out of date", self.get("/results/", "owner")[2].decode())

        os.utime(edit, (done + 120, done + 120))                  # an edit after the results were written
        page = self.page(ps_id)
        self.assertIn(f"These results are older than the latest review edits (last edit {R.hhmm(done + 120)}).",
                      page)
        self.assertIn("Update the results to include the latest review.", page)
        self.assertIn("Update results now", page)
        self.assertTrue(self.status(ps_id)["out_of_date"])
        idx = self.get("/results/", "owner")[2].decode()
        card = re.search(r'href="/results/%s/">(.*?)</a>' % ps_id, idx, re.S).group(1)
        self.assertIn("Out of date", card)

        compute_and_write(d)                                      # computed again: current
        os.utime(d / "results" / "results.json", (done + 300, done + 300))
        self.assertNotIn('id="cj-results-stale"', self.page(ps_id))


class Access(Base):
    def test_set_reached_through_a_junction_is_404(self):
        if os.name != "nt":
            raise unittest.SkipTest("junctions are Windows only")
        import _winapi
        outside = self.tmp / "outside"
        make_set(outside, "ps_20261105_aaa777")
        self.addCleanup(shutil.rmtree, outside, True)
        link = self.root / "ps_20261105_aaa777"
        _winapi.CreateJunction(str(outside / "ps_20261105_aaa777"), str(link))
        self.addCleanup(os.rmdir, link)
        self.assertTrue((link / "request.json").is_file())          # the junction works on disk
        self.assertEqual(self.get("/results/ps_20261105_aaa777/", "owner")[0], 404)
        self.assertNotIn("aaa777", self.get("/results/", "owner")[2].decode())
        self.assertEqual(self.get(f"/results/{TODO1}/", "owner")[0], 200)      # control: a real folder

    def test_module_declaration(self):
        self.assertEqual((R.PREFIX, R.ROLES), ("/results", ("admin", "advanced", "basic")))
        st, h, _ = self.get("/results")
        self.assertEqual((st, T.hget(h, "location")), (307, "/results/"))
        st, h, _ = self.get(f"/results/{NOEDIT}")
        self.assertEqual((st, T.hget(h, "location")), (307, f"/results/{NOEDIT}/"))

    def test_every_role_enters(self):
        for who in ("admin", "advanced", "basic", "owner"):
            with self.subTest(who=who):
                self.assertEqual(self.get("/results/", who)[0], 200)
        self.assertEqual(self.get("/results/", None)[0], 401)

    def test_index_lists_only_own_sets(self):
        body = self.get("/results/", "owner")[2].decode()
        self.assertIn(f'href="/results/{NOEDIT}/"', body)
        self.assertNotIn(OTHERS, body)
        for junk in (".incoming", ".locks", "predict.lock"):
            self.assertNotIn(junk, body)
        self.assertIn(OTHERS, self.get("/results/", "admin")[2].decode())   # admin sees everyone's
        body = self.get("/results/", "basic")[2].decode()                    # owns nothing
        self.assertNotIn("ps_2026", body)
        self.assertIn("No analyses yet", body)

    def test_one_404_for_missing_and_not_yours(self):
        paths = ("/", "/api/status", "/download/per_sample.csv", "/download/results.json")
        for suffix in paths:
            with self.subTest(suffix=suffix):
                miss = self.get(f"/results/{MISSING}{suffix}", "owner")
                theirs = self.get(f"/results/{OTHERS}{suffix}", "owner")
                self.assertEqual(miss[0], 404)
                self.assertEqual(theirs[0], 404)
                self.assertEqual(miss[2], theirs[2])
                self.assertNotIn(b"eee555", theirs[2])
        self.assertEqual(self.get(f"/results/{OTHERS}/", "advanced")[0], 404)      # advanced is not admin
        self.assertEqual(self.get(f"/results/{OTHERS}/", "admin")[0], 200)         # control: it exists
        self.assertEqual(self.get(f"/results/{NOEDIT}/", "owner")[0], 200)          # control: owner sees own
        st, _, _ = self.post(f"/results/{OTHERS}/compute", "owner")
        self.assertEqual(st, 404)
        self.assertEqual(R._jobs, {})
        for bad in ("ps_..", "PS_20261105_ABC123", "ps_20261105_abc12", "..%2f..%2fx"):
            with self.subTest(bad=bad):
                self.assertEqual(self.get(f"/results/{bad}/", "admin")[0], 404)

    def test_script_is_a_file(self):
        st, h, body = self.get("/results/results.js")
        self.assertEqual(st, 200)
        self.assertTrue(T.hget(h, "content-type").startswith("text/javascript"))
        self.assertIn(b"data-cj-results-poll", body)
        for ps in (NOEDIT, TODO1):
            page = self.page(ps)
            self.assertIn('<script src="/results/results.js" defer></script>', page)
            self.assertEqual(re.findall(r"<script(?![^>]*\ssrc=)[^>]*>", page), [])
            self.assertNotIn("http://", page.split('<main class="cj-main">', 1)[1])
            self.assertNotIn("https://", page.split('<main class="cj-main">', 1)[1])

    def test_interrupted_write_not_served(self):
        ps_id = "ps_20261105_abc444"
        d = make_set(self.root, ps_id)
        self.addCleanup(shutil.rmtree, d, True)
        compute_and_write(d)
        names = ("per_sample.csv", "per_image.csv", "results.json", f"results_{ps_id}.zip")
        for name in names:                                                     # control: complete
            self.assertEqual(self.get(f"/results/{ps_id}/download/{name}", "owner")[0], 200, name)
        self.assertIn(R.CSV_NOTE.replace("'", "&#x27;"), self.page(ps_id, "owner"))
        done = (d / "results" / "results.json").stat().st_mtime
        os.utime(d / "results" / "per_image.csv", (done + 60, done + 60))      # a CSV newer than results.json
        for name in names:
            with self.subTest(name=name):
                st, h, body = self.get(f"/results/{ps_id}/download/{name}", "owner")
                self.assertEqual(st, 404)
                self.assertIsNone(T.hget(h, "content-disposition"))
                self.assertIn(b"Results are being written or were interrupted: compute again.", body)
        self.assertEqual(self.get(f"/results/{ps_id}/download/per_image.csv", "other")[0], 404)   # not theirs
        self.assertNotIn(b"interrupted", self.get(f"/results/{ps_id}/download/per_image.csv", "other")[2])

    def test_downloads_are_attachments(self):
        rdir = self.root / NOEDIT / "results"
        for name, ctype in (("per_sample.csv", "text/csv"), ("per_image.csv", "text/csv"),
                            ("results.json", "application/json")):
            with self.subTest(name=name):
                st, h, body = self.get(f"/results/{NOEDIT}/download/{name}", "owner")
                self.assertEqual(st, 200)
                self.assertEqual(T.hget(h, "content-disposition"), f'attachment; filename="{NOEDIT}_{name}"')
                self.assertEqual(T.hget(h, "x-content-type-options"), "nosniff")
                self.assertTrue(T.hget(h, "content-type").startswith(ctype))
                self.assertEqual(T.hget(h, "cache-control"), "no-store")
                disk = (rdir / name).read_bytes()
                if name.endswith(".csv"):                                     # provenance # lines, then the file
                    head, rest = body[:len(body) - len(disk)], body[len(body) - len(disk):]
                    self.assertEqual(rest, disk)
                    lines = head.decode().splitlines()
                    self.assertTrue(lines and all(x.startswith("# ") for x in lines), lines)
                    self.assertIn("model=v5_evnonev_y8m_20261029", head.decode())
                    self.assertIn("reviewed=0/3 images", head.decode())
                    self.assertIn("rule=conf>=0.5; EV = class 0 (EV)", head.decode())
                    rows = list(csv.DictReader(x for x in body.decode().splitlines() if not x.startswith("#")))
                    self.assertTrue(rows)
                else:
                    self.assertEqual(body, disk)                              # results.json as written
        st, h, body = self.get(f"/results/{REVIEWED}/download/results_{REVIEWED}.zip", "owner")
        self.assertEqual(st, 200)
        self.assertEqual(T.hget(h, "content-disposition"), f'attachment; filename="results_{REVIEWED}.zip"')
        self.assertEqual(T.hget(h, "x-content-type-options"), "nosniff")
        z = zipfile.ZipFile(io.BytesIO(body))
        self.assertEqual(sorted(z.namelist()), ["README.txt", "per_image.csv", "per_sample.csv", "results.json"])
        readme = z.read("README.txt").decode()
        for want in ("Model: v5_evnonev_y8m_20261029", "1 of 3 images reviewed", "Counting rule:",
                     f"Prediction set: {REVIEWED}", "by pujan", "Computed:"):
            self.assertIn(want, readme)
        self.assertEqual(z.read("per_image.csv"), (self.root / REVIEWED / "results" / "per_image.csv").read_bytes())
        for bad in ("../request.json", "request.json", "results_ps_20261105_abc123.zip", "x.csv"):
            with self.subTest(bad=bad):
                self.assertEqual(self.get(f"/results/{REVIEWED}/download/{bad}", "owner")[0], 404)
        self.assertEqual(self.get(f"/results/{TODO1}/download/per_sample.csv", "owner")[0], 404)   # none yet


class Compute(Base):
    def test_cross_origin_post_refused(self):
        for origin in ("https://evil.example", None):
            with self.subTest(origin=origin):
                st, _, _ = self.post(f"/results/{TODO1}/compute", "owner", origin=origin or False)
                self.assertEqual(st, 403)
        self.assertEqual(R._jobs, {})
        self.assertEqual(self.status(TODO1)["state"], "none")
        self.assertEqual(self.get(f"/results/{TODO1}/compute", "owner")[0], 405)

    def test_button_when_missing(self):
        page = self.page(TODO1, "owner")
        self.assertIn("Results are not computed yet.", page)
        self.assertIn(f'<form class="cj-results-form" method="post" action="/results/{TODO1}/compute">', page)
        self.assertNotIn("data-cj-results-poll", page)

    def test_predictions_running_no_button(self):
        page = self.page(RUNNING, "owner")
        self.assertIn("The model is still running (1 of 3 images)", page)
        self.assertNotIn('<form class="cj-results-form"', page)
        self.assertEqual(self.status(RUNNING)["state"], "waiting")
        st, h, _ = self.post(f"/results/{RUNNING}/compute", "owner")
        self.assertEqual((st, T.hget(h, "location")), (303, f"/results/{RUNNING}/"))
        self.assertEqual(R._jobs, {})                                  # nothing started

    def test_slow_compute_runs_in_a_thread(self):
        release, calls = threading.Event(), []

        def slow(ps_dir, reviewed):
            calls.append((Path(ps_dir).name, reviewed))
            release.wait(20)
            return compute_and_write(ps_dir, reviewed)

        self.patch_compute(slow)
        self.addCleanup(release.set)
        self.clear_results(TODO1)
        self.clear_results(TODO2)
        t0 = time.time()
        st, h, _ = self.post(f"/results/{TODO1}/compute", "owner")
        self.assertEqual((st, T.hget(h, "location")), (303, f"/results/{TODO1}/"))
        self.assertLess(time.time() - t0, 3)
        self.assertEqual(self.wait_state(TODO1, ("running",))["state"], "running")
        t0 = time.time()
        page = self.page(TODO1, "owner")                                # not blocked by the running compute
        self.assertLess(time.time() - t0, 3)
        self.assertIn(f'data-cj-results-poll="/results/{TODO1}/api/status"', page)
        self.assertIn("Computing the results.", page)
        self.assertNotIn('<form class="cj-results-form"', page)
        self.assertEqual(self.post(f"/results/{TODO1}/compute", "owner")[0], 303)      # a second press: no 2nd run
        self.assertEqual(self.post(f"/results/{TODO2}/compute", "owner")[0], 303)      # another set: waits its turn
        self.assertEqual(self.status(TODO2)["state"], "queued")
        self.assertIn("Waiting for another computation", self.page(TODO2, "owner"))
        self.assertEqual([c[0] for c in calls], [TODO1])
        release.set()
        self.wait_state(TODO1, ("ready",))
        self.wait_state(TODO2, ("ready",))
        self.assertEqual([c[0] for c in calls], [TODO1, TODO2])
        self.assertEqual(calls[0][1], None)                           # no review package: model predictions
        exp = json.loads((GOLDEN / "expected_results_noedit.json").read_text(encoding="utf-8"))
        page = self.page(TODO1, "owner")
        Golden.check_samples(self, page, exp)
        self.assertNotIn("data-cj-results-poll", page)
        self.assertIn("Update results now", page)

    def test_failed_compute(self):
        def boom(ps_dir, reviewed):
            raise ValueError(f"reviewed json images differ <x> {ps_dir}")

        self.patch_compute(boom)
        self.assertEqual(self.post(f"/results/{TODO3}/compute", "owner")[0], 303)
        self.assertEqual(self.wait_state(TODO3, ("error",))["error"], "Computing the results failed.")
        owner, admin = self.page(TODO3, "owner"), self.page(TODO3, "admin")
        for page in (owner, admin):
            self.assertIn("Computing the results failed.", page)
            self.assertIn("Compute results", page)                       # can try again
            self.assertNotIn("<x>", page)
        self.assertIn("ValueError: reviewed json images differ &lt;x&gt;", admin)
        self.assertNotIn("ValueError", owner)                             # detail for admins only
        self.assertNotIn(str(self.root), owner)

    def test_edit_saved_during_compute_is_stale(self):
        """An edit saved after the worker read the review but before results.json was written: the results are
        newer than the edit file, yet they did not count it, so they are out of date. Then a clean recompute
        (control: current) and a removed edit (out of date again)."""
        ps_id = "ps_20261105_abc555"
        d = make_set(self.root, ps_id)
        self.addCleanup(shutil.rmtree, d, True)
        pkg = self.review / f"pred_{ps_id}"
        (pkg / "edits").mkdir(parents=True)
        self.addCleanup(shutil.rmtree, pkg, True)
        for f in ("review_package.json", "prediction_package.json", "edits/0001.png.json"):
            (pkg / f).write_text("{}", encoding="utf-8")
            os.utime(pkg / f, (time.time() - 600, time.time() - 600))
        self.addCleanup(setattr, R, "REVIEWED", R.REVIEWED)
        R.REVIEWED = lambda pkg_dir, write=True: None          # the read itself (model predictions)
        mid = pkg / "edits" / "0002.png.json"

        def edit_meanwhile(ps_dir, reviewed):
            time.sleep(0.05)
            mid.write_text("{}", encoding="utf-8")             # saved while the computation runs
            time.sleep(0.05)
            return compute_and_write(ps_dir, reviewed)

        self.patch_compute(edit_meanwhile)
        self.assertEqual(self.post(f"/results/{ps_id}/compute", "owner")[0], 303)
        self.wait_state(ps_id, ("ready", "error"))
        self.assertTrue(self.status(ps_id)["has_results"])
        self.assertGreater((d / "results" / "results.json").stat().st_mtime, mid.stat().st_mtime)  # mtime rule misses
        rr = json.loads((d / "results" / R.REVIEW_READ).read_text(encoding="utf-8"))
        self.assertEqual(rr["edits"], ["0001.png.json"])
        self.assertLess(rr["read_at"], mid.stat().st_mtime)
        page = self.page(ps_id, "owner")
        self.assertIn(f"older than the latest review edits (last edit {R.hhmm(mid.stat().st_mtime)})", page)
        self.assertTrue(self.status(ps_id)["out_of_date"])

        self.patch_compute(lambda ps_dir, reviewed: compute_and_write(ps_dir, reviewed))   # control: recompute
        self.assertEqual(self.post(f"/results/{ps_id}/compute", "owner")[0], 303)
        self.wait_state(ps_id, ("ready", "error"))
        self.assertNotIn('id="cj-results-stale"', self.page(ps_id, "owner"))
        self.assertFalse(self.status(ps_id)["out_of_date"])

        (pkg / "edits" / "0001.png.json").unlink()                # an edit removed after the computation
        self.assertIn("These results count review edits that have since been removed.", self.page(ps_id, "owner"))
        card = re.search(r'href="/results/%s/">(.*?)</a>' % ps_id, self.get("/results/", "owner")[2].decode(),
                         re.S).group(1)
        self.assertIn("Out of date", card)

    def test_failed_recompute_keeps_earlier_results(self):
        def boom(ps_dir, reviewed):
            raise RuntimeError("disk full")

        self.patch_compute(boom)
        before = (self.root / EDITED / "results" / "results.json").read_bytes()
        computed = json.loads(before)["computed"]
        self.assertEqual(self.post(f"/results/{EDITED}/compute", "owner")[0], 303)
        self.wait_state(EDITED, ("error",))
        page = self.page(EDITED, "owner")
        self.assertIn("Computing the results failed.", page)
        self.assertIn(f"The results below are from the earlier computation (computed {R.when(computed)}).", page)
        self.assertIn('id="cj-results-samples"', page)                    # the earlier results still shown
        self.assertEqual((self.root / EDITED / "results" / "results.json").read_bytes(), before)

    def test_review_package_is_the_source(self):
        """Control: an unedited package gives exactly the no-edit numbers, and the package is not written."""
        from annotation import prediction_package as PP
        pkg = PP.build(self.root / PKG, self.review)
        self.addCleanup(shutil.rmtree, pkg, True)
        self.clear_results(PKG)
        before = tree_md5(pkg)
        seen = []

        def spy(ps_dir, reviewed):
            seen.append(reviewed)
            return compute_and_write(ps_dir, reviewed)

        self.patch_compute(spy)
        self.assertEqual(self.post(f"/results/{PKG}/compute", "owner")[0], 303)
        self.wait_state(PKG, ("ready", "error"))
        self.assertEqual(self.status(PKG)["state"], "ready", R._jobs)
        self.assertEqual(seen[0]["schema"], "reviewed/1")
        self.assertEqual(seen[0]["prediction_set"], PKG)
        self.assertEqual(tree_md5(pkg), before)                         # reviewed_polygons(write=False)
        self.assertFalse((pkg / "reviewed.json").exists())
        exp = json.loads((GOLDEN / "expected_results_noedit.json").read_text(encoding="utf-8"))
        Golden.check_samples(self, self.page(PKG, "owner"), exp)


def launch(p):
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:
            continue
    return None


@browser
class Browser(Base):
    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        super().setUpClass()
        cls.own_shots = not os.environ.get("CJ_SHOT_DIR")
        cls.shots = Path(os.environ.get("CJ_SHOT_DIR") or tempfile.mkdtemp(prefix="cj_shots_"))
        cls.shots.mkdir(parents=True, exist_ok=True)
        cls.pw = sync_playwright().start()
        cls.browser = launch(cls.pw)
        if cls.browser is None:
            cls.tearDownClass()
            raise unittest.SkipTest("no Edge / Chrome for Playwright")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "browser", None):
            cls.browser.close()
        if getattr(cls, "pw", None):
            cls.pw.stop()
        super().tearDownClass()
        if cls.own_shots:
            shutil.rmtree(cls.shots, ignore_errors=True)

    def session(self, width, scheme, who="owner"):
        ctx = self.browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
        origin = self.p.origin
        rec = {"requests": [], "foreign": [], "errors": [], "console": []}
        ctx.route("**/*", lambda r: (r.continue_() if r.request.url.startswith(origin + "/")
                                     else (rec["foreign"].append(r.request.url), r.abort())))
        page = ctx.new_page()
        page.on("request", lambda r: rec["requests"].append(r.url))
        page.on("pageerror", lambda e: rec["errors"].append(str(e)))
        page.on("console", lambda m: m.type == "error" and rec["console"].append(m.text))
        page.goto(f"{origin}/login?k={self.keys[who]}")
        self.assertEqual(page.url, origin + "/")
        return ctx, page, rec

    def check_clean(self, rec):
        self.assertEqual(rec["foreign"], [])
        self.assertTrue(rec["requests"])
        self.assertTrue(all(u.startswith(self.p.origin + "/") for u in rec["requests"]),
                        [u for u in rec["requests"] if not u.startswith(self.p.origin + "/")])
        self.assertEqual(rec["errors"], [])
        self.assertEqual(rec["console"], [])

    def no_hscroll(self, page):
        sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
        self.assertLessEqual(sw, iw, "horizontal scroll")

    def test_pages_sizes_and_schemes(self):
        for scheme in ("light", "dark"):
            for width in (390, 1280):
                for ps, tag in ((REVIEWED, "reviewed"), (REFUSED, "refused"), (NOEDIT, "noedit")):
                    with self.subTest(width=width, scheme=scheme, ps=tag):
                        ctx, page, rec = self.session(width, scheme)
                        try:
                            page.goto(f"{self.p.origin}/results/{ps}/")
                            page.wait_for_load_state("networkidle")
                            page.wait_for_selector("#cj-bar a")
                            page.click("#cj-results-images summary")
                            self.no_hscroll(page)
                            sw, cw = page.evaluate("(e => [e.scrollWidth, e.clientWidth])(document.querySelector("
                                                   "'#cj-results-images .cj-results-wrap'))")
                            self.assertLessEqual(sw, cw, "per-image table wider than its box")
                            self.assertEqual(page.locator("body img").count(), 0)     # XSS label stayed text
                            self.assertEqual(page.locator("svg.cj-results-svg circle").count(),
                                             1 if ps == REFUSED else 2)
                            fill = page.evaluate("getComputedStyle(document.querySelector("
                                                 "'.cj-results-med')).fill")
                            self.assertEqual(fill, "rgb(57, 135, 229)" if scheme == "dark"
                                             else "rgb(42, 120, 214)")                # --s1, light/dark
                            bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
                            self.assertEqual(bg, "rgb(15, 21, 27)" if scheme == "dark" else "rgb(243, 245, 247)")
                            if ps == REVIEWED:
                                self.assertIn("1 of 3 images reviewed",
                                              page.text_content("#cj-results-honesty"))
                            page.screenshot(path=str(self.shots / f"results_{tag}_{width}_{scheme}.png"),
                                            full_page=True)
                            self.check_clean(rec)
                        finally:
                            ctx.close()

    def test_index_page(self):
        for width, scheme in ((390, "light"), (1280, "dark")):
            ctx, page, rec = self.session(width, scheme)
            try:
                page.goto(f"{self.p.origin}/results/")
                page.wait_for_load_state("networkidle")
                self.no_hscroll(page)
                self.assertGreaterEqual(page.locator(".cj-results-set").count(), 5)
                page.screenshot(path=str(self.shots / f"results_index_{width}_{scheme}.png"), full_page=True)
                self.check_clean(rec)
            finally:
                ctx.close()

    def test_compute_poll_reload(self):
        release = threading.Event()

        def slow(ps_dir, reviewed):
            release.wait(20)
            return compute_and_write(ps_dir, reviewed)

        self.patch_compute(slow)
        self.addCleanup(release.set)
        self.clear_results(TODO1)
        ctx, page, rec = self.session(390, "light")
        try:
            page.goto(f"{self.p.origin}/results/{TODO1}/")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=str(self.shots / "results_compute_390_light_1_button.png"), full_page=True)
            page.click("form.cj-results-form button")
            page.wait_for_selector("#cj-results-job[data-cj-results-poll]")
            with page.expect_response(lambda r: r.url.endswith(f"/results/{TODO1}/api/status")) as resp:
                pass
            self.assertEqual(resp.value.status, 200)
            self.no_hscroll(page)
            page.screenshot(path=str(self.shots / "results_compute_390_light_2_running.png"), full_page=True)
            release.set()
            page.wait_for_selector("#cj-results-samples", timeout=30000)        # the poll reloaded the page
            self.assertEqual(page.locator("#cj-results-job").count(), 0)
            self.assertIn("Model only", page.text_content("#cj-results-honesty"))
            page.screenshot(path=str(self.shots / "results_compute_390_light_3_done.png"), full_page=True)
            self.check_clean(rec)
        finally:
            ctx.close()


if __name__ == "__main__":
    unittest.main()
