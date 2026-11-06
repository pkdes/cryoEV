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
TODO1, TODO2, TODO3 = "ps_20261105_ccc331", "ps_20261105_ccc332", "ps_20261105_ccc334"   # not computed
PKG = "ps_20261105_ccc333"                       # not computed; has an (unedited) review package
RUNNING = "ps_20261105_ddd444"                   # predictions still running
OTHERS = "ps_20261105_eee555"                    # someone else's set, with results
MISSING = "ps_20261105_000000"


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
    for k, v, sub in re.findall(r'data-k="(\w+)"><span class="cj-results-v">(.*?)</span>'
                                r'(?:<small class="cj-results-sub">(.*?)</small>)?', m.group(1)):
        out[k] = (v, sub)
    return out


def image_rows(page: str) -> dict:
    out = {}
    for img, body in re.findall(r'<tr data-image="(\d+)">(.*?)</tr>', page, re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", body, re.S)
        chip = re.sub(r"<[^>]+>", "", tds[2])
        out[img] = {"sample": tds[1], "chip": chip, "ev": int(tds[3]), "multilayer": int(tds[4]),
                    "inner": int(tds[5]), "nonev": int(tds[6])}
    return out


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
        self.assertIn("Review a suggested spot-check", page)
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
        self.assertIn("Review the 2 suggested images", page)                   # spotcheck.json n
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
        self.assertIn("<b>S1</b> (2 images)", notice)
        self.assertIn("mixes nm/px 1.4724, 2", notice)
        self.assertEqual(re.findall(r'<tr data-sample="([^"]+)">', page), [R.esc(XSS)])   # S2 still shown
        self.assertNotIn(XSS, page)                                                  # escaped everywhere
        self.assertIn(R.esc(XSS), page)
        self.assertEqual(len(image_rows(page)), 3)

    def test_whole_set_refused(self):
        d = make_set(self.root, "ps_20261105_abc999", state="running")
        self.addCleanup(shutil.rmtree, d, True)
        compute_and_write(d)                       # refused whole: set not finished
        page = self.page("ps_20261105_abc999")
        self.assertIn("No results for this set.", page)
        self.assertIn("set not finished", page)
        self.assertNotIn('id="cj-results-samples"', page)


class Access(Base):
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
                self.assertEqual(body, (rdir / name).read_bytes())          # the files as written, unchanged
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
        self.assertIn("Compute again", page)

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
