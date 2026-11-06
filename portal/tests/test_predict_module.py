"""
Predict module (portal/modules/predict.py): roles through the Gate, the session-pick flow end to end with the fake
predictor (jobs/tests/fake_predictor.py through the jobs test hooks; never the real GPU), the set page reaching
done, Review / Results building the package and results.json in temp roots (known answer: the golden fixture's
expected counts), the nm/px confirmation (no tick = refused, nothing written), ownership (someone else's set = the
same 404 as no set), the Origin check, escaping, and the pages in a browser (Playwright, Edge channel: 390 and
1280 px, light and dark, no horizontal scroll, same-origin requests only, no console errors, the status
refresh). Screenshots go to $CJ_SHOT_DIR when set (else a temp folder, removed afterwards).

Temp roster / registry / predict and review roots only; never the real ones.

  python -m unittest discover -s portal/tests -v
"""
import csv
import html
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import predict as P  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

REPO = T.REPO
GOLDEN = REPO / "tests" / "fixtures" / "cj_golden" / "predict" / "ps_20261105_abc123"
MODEL = "v5_evnonev_y8m_20261029"
GOLDEN_PRED = GOLDEN / "p" / MODEL / "predictions"
FAKE_PRED = [sys.executable, str(REPO / "jobs" / "tests" / "fake_predictor.py")]
XSS = '<img src=x onerror=alert(1)>'
ENV_KEYS = ("CJ_PREDICT_SOURCES", "CJ_PREDICT_ROOT", "CJ_REVIEW_ROOT", "CRYOEV_LAPTOP_ROOT", "CRYOEV_CRYOAI_ROOT",
            "CRYOEV_PROCESSING_ROOT", "CRYOEV_REGISTRY", "CRYOEV_JOBS_TESTING", "CRYOEV_JOBS_TEST_HOOKS",
            "FAKE_PRED_FROM", "FAKE_PRED_DELAY")


def golden_rows():
    with open(GOLDEN / "images.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_rows(path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls._saved = {k: os.environ.get(k) for k in ENV_KEYS}
        tmp = cls.env.tmp
        cls.keys = {"admin": cls.env.key, "advanced": A.add_key(cls.env.roster, "adv", "advanced"),
                    "basic": A.add_key(cls.env.roster, "bas", "basic"),
                    "other": A.add_key(cls.env.roster, "bas2", "basic")}
        cls.lap = tmp / "lap"
        cls.root, cls.review = cls.lap / "predict", cls.lap / "review"
        (tmp / "cai" / "training outputs").mkdir(parents=True)
        (tmp / "cai" / "training images").mkdir()
        # the registry: the production model
        (cls.env.registry / "model.jsonl").write_text(json.dumps(
            {"schema": "model/1", "id": MODEL, "status": "production", "logged": "2026-11-05T00:00:00"}) + "\n",
            encoding="utf-8")
        # source A: the golden tiles under their original names + a manifest (sample_label S1 S1 S2, 1.4724 nm/px)
        cls.src_a = tmp / "sessions" / "20261027_tiles"
        cls.src_a.mkdir(parents=True)
        rows = golden_rows()
        for r in rows:
            shutil.copyfile(GOLDEN / r["file"], cls.src_a / r["orig_name"])
        (cls.src_a / "MANIFEST.csv").write_text("output_name,sample_label,nm_per_px\n" + "".join(
            f"{r['orig_name']},{r['sample_label']},{r['nm_per_px']}\n" for r in rows), encoding="utf-8")
        # source B: no manifest (one sample; nm/px must be typed), session from the config
        cls.src_b = tmp / "sessions" / "plain_tiles"
        cls.src_b.mkdir(parents=True)
        shutil.copyfile(GOLDEN / "images" / "0001.png", cls.src_b / "t1.png")
        srcs = [{"label": "EV prep A " + XSS, "dir": str(cls.src_a)},
                {"label": "Plain tiles", "dir": str(cls.src_b), "session": "20261028"},
                {"label": "Gone", "dir": str(tmp / "nowhere")}]
        os.environ.update(CJ_PREDICT_SOURCES=json.dumps(srcs), CJ_PREDICT_ROOT=str(cls.root),
                          CJ_REVIEW_ROOT=str(cls.review), CRYOEV_LAPTOP_ROOT=str(cls.lap),
                          CRYOEV_CRYOAI_ROOT=str(tmp / "cai"), CRYOEV_PROCESSING_ROOT=str(tmp / "no_processing"),
                          CRYOEV_JOBS_TESTING="1", FAKE_PRED_FROM=str(GOLDEN_PRED),
                          CRYOEV_JOBS_TEST_HOOKS=json.dumps({"gpu": {"total_mb": 4096, "apps": []},
                                                             "predict_cmd": FAKE_PRED}))
        os.environ.pop("CRYOEV_REGISTRY", None)
        os.environ.pop("FAKE_PRED_DELAY", None)
        # modules.json with Predict on (the entry this module needs; the real file is the manager's)
        man = json.loads((REPO / "portal" / "modules.json").read_text(encoding="utf-8"))
        for m in man["modules"]:
            if m["id"] == "predict":
                m.update(route="/predict/", status="on")
        cls.manifest = tmp / "modules.json"
        cls.manifest.write_text(json.dumps(man), encoding="utf-8")
        cls.p = T.Portal(cls.env.config(manifest=cls.manifest))
        cls.tok = {r: T.login(cls.p.port, k) for r, k in cls.keys.items()}
        cls.key_a = next(s["key"] for s in P.sources(None) if s["label"].startswith("EV prep A"))
        cls.key_b = next(s["key"] for s in P.sources(None) if s["label"] == "Plain tiles")

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        for k, v in cls._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        time.sleep(0.5)                                    # a detached predictor may still be closing its log
        cls.env.cleanup()

    # ---- helpers
    def get(self, path, who="admin"):
        return T.request(self.p.port, "GET", path, T.cookie(self.tok[who]))

    def post(self, path, data, who="admin", origin=True, ctype="application/x-www-form-urlencoded"):
        from urllib.parse import urlencode
        body = data if isinstance(data, bytes) else urlencode(data, doseq=True).encode()
        h = {**T.cookie(self.tok[who]), "Content-Type": ctype}
        if origin is True:
            h["Origin"] = self.p.origin
        elif origin:
            h["Origin"] = origin
        return T.request(self.p.port, "POST", path, h, body, timeout=120)

    def page(self, path, who="admin", status=200) -> str:
        st, h, body = self.get(path, who)
        self.assertEqual(st, status, body[:400])
        if status == 200:
            self.assertEqual(T.hget(h, "cache-control"), "no-store")
        return body.decode("utf-8")

    def sets(self) -> list:
        return sorted(p.name for p in self.root.iterdir() if p.name.startswith("ps_")) if self.root.exists() else []

    def form_a(self, who="basic", **kw):
        html_ = self.page(f"/predict/new/session/{self.key_a}/", who)
        sig = re.search(r'name="sig" value="([0-9a-f]+)"', html_).group(1)
        data = {"sig": sig, "samples": ["g0", "g1"], "nm_g0": "1.4724", "nm_g1": "1.4724", "how": "all",
                "confirm": "yes", "model": MODEL}
        data.update(kw)
        return {k: v for k, v in data.items() if v is not None}

    def submit_a(self, who="basic", **kw):
        st, h, body = self.post(f"/predict/new/session/{self.key_a}/", self.form_a(who, **kw), who)
        return st, T.hget(h, "location"), body.decode("utf-8")

    def wait_state(self, ps, states=("done", "failed"), timeout=90):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                st = json.loads((self.root / ps / "status.json").read_text(encoding="utf-8"))
                if st["state"] in states:
                    return st
            except (OSError, ValueError):
                pass
            time.sleep(0.2)
        self.fail(f"{ps} not {states} after {timeout} s: {(self.root / ps / 'status.json').read_text()}")

    def wait_file(self, path, timeout=120):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if Path(path).is_file():
                return
            time.sleep(0.2)
        self.fail(f"{path} not written after {timeout} s")

    def make_done_set(self, who="basic"):
        before = set(self.sets())
        st, loc, body = self.submit_a(who)
        self.assertEqual(st, 303, body[:600])
        ps = (set(self.sets()) - before).pop()
        self.assertEqual(loc, f"/predict/{ps}/")
        self.wait_state(ps, ("done",))
        return ps


class Roles(Base):
    def test_module_declaration(self):
        self.assertEqual((P.PREFIX, P.ROLES), ("/predict", ("admin", "advanced", "basic")))
        st, h, _ = self.get("/predict")
        self.assertEqual((st, T.hget(h, "location")), (307, "/predict/"))

    def test_every_role_in_no_session_out(self):
        for who in ("admin", "advanced", "basic"):
            for path in ("/predict/", "/predict/new/", "/predict/predict.js",
                         "/predict/api/mine", f"/predict/new/session/{self.key_a}/"):
                with self.subTest(who=who, path=path):
                    self.assertEqual(self.get(path, who)[0], 200)
        self.assertEqual(T.request(self.p.port, "GET", "/predict/api/mine")[0], 401)

    def test_script_is_a_file_no_inline_script(self):
        st, h, body = self.get("/predict/predict.js")
        self.assertEqual(st, 200)
        self.assertTrue(T.hget(h, "content-type").startswith("text/javascript"))
        for path in ("/predict/", "/predict/new/", f"/predict/new/session/{self.key_a}/"):
            page = self.page(path)
            self.assertIn('<script src="/predict/predict.js" defer></script>', page)
            self.assertEqual(re.findall(r"<script(?![^>]*\ssrc=)[^>]*>", page), [], path)
            self.assertEqual(re.findall(r"<[a-z][^>]*\son[a-z]+\s*=", page), [], path)   # no inline handlers

    def test_sources_plain_for_basic_paths_for_admin(self):
        basic, admin = self.page("/predict/new/", "basic"), self.page("/predict/new/", "admin")
        for page in (basic, admin):
            self.assertIn("EV prep A &lt;img src=x", page)
            self.assertNotIn(XSS, page)
        self.assertNotIn(str(self.src_a), basic)
        self.assertNotIn("Gone", basic)                                   # a broken source is hidden from users
        self.assertIn(html.escape(str(self.src_a)), admin)
        self.assertIn("Not available: folder not found", admin)
        self.assertEqual(self.get("/predict/new/session/0123456789/", "basic")[0], 404)

    def test_basic_gets_only_the_production_model(self):
        page = self.page(f"/predict/new/session/{self.key_a}/", "basic")
        self.assertIn(f'<input type="hidden" name="model" value="{MODEL}">', page)
        self.assertIn("EV score 0.70 ± 0.01", page)
        st, _, body = self.submit_a("basic", model="v4_y8m_20260924")
        self.assertEqual(st, 400)
        self.assertIn("Pick a model from the list", body)
        adv = self.page(f"/predict/new/session/{self.key_a}/", "advanced")
        self.assertIn("<select name=\"model\">", adv)
        self.assertIn(f'<option value="{MODEL}" selected>', adv)


class SessionFlow(Base):
    def test_end_to_end_review_results(self):
        ps = self.make_done_set("basic")
        d = self.root / ps
        req = json.loads((d / "request.json").read_text(encoding="utf-8"))
        self.assertEqual((req["owner"], req["model"], req["source"]["kind"], req["source"]["session"]),
                         ("bas", MODEL, "session", "20261027"))
        rows = read_rows(d / "images.csv")
        self.assertEqual([(r["orig_name"], r["sample_label"], float(r["nm_per_px"]), r["nm_confirmed_by"])
                          for r in rows],
                         [(g["orig_name"], g["sample_label"], 1.4724, "bas") for g in golden_rows()])
        page = self.page(f"/predict/{ps}/", "basic")
        self.assertIn("Done", page)
        self.assertIn(f'action="/predict/{ps}/review"', page)
        self.assertIn(f'action="/predict/{ps}/results"', page)
        self.assertIn("Check these first", page)                         # spot check from the set (no package)
        self.assertEqual(page.count("<li><span class=\"cj-predict-li-main\">"), 3)
        self.assertNotIn(str(d), page)                                    # no paths for a scientist
        # someone else: the same 404 as no set at all, for the page, the API and every action
        _, _, nobody = self.get("/predict/ps_20990101_abcdef/", "other")
        st, _, theirs = self.get(f"/predict/{ps}/", "other")
        self.assertEqual((st, theirs), (404, nobody))
        self.assertEqual(self.get(f"/predict/api/set/{ps}", "other")[0], 404)
        for act in ("review", "results", "start"):
            self.assertEqual(self.post(f"/predict/{ps}/{act}", {}, "other")[0], 404)
        self.assertFalse(self.review.exists() and any(self.review.iterdir()))
        mine = json.loads(self.get("/predict/api/mine", "basic")[2])
        self.assertEqual(mine["schema"], "cj_predict_mine/1")
        self.assertIn(ps, [i["ps"] for i in mine["items"]])
        self.assertNotIn(ps, [i["ps"] for i in json.loads(self.get("/predict/api/mine", "other")[2])["items"]])
        self.assertIn(ps, [i["ps"] for i in json.loads(self.get("/predict/api/mine", "admin")[2])["items"]])
        self.assertEqual(self.get(f"/predict/{ps}/", "admin")[0], 200)  # admin sees all
        # Review: builds the package in a thread, then links to it
        st, h, _ = self.post(f"/predict/{ps}/review", {}, "basic")
        self.assertEqual((st, T.hget(h, "location")), (303, f"/predict/{ps}/"))
        pkg = self.review / f"pred_{ps}"
        self.wait_file(pkg / "review_package.json")
        self.wait_file(pkg / "spotcheck.json")
        for _ in range(50):
            if P.task("review", ps) is None:
                break
            time.sleep(0.1)
        self.assertTrue((pkg / "data.js").is_file())
        self.assertEqual(len(list((pkg / "images").glob("*.jpg"))), 3)
        page = self.page(f"/predict/{ps}/", "basic")
        self.assertIn(f'href="/review/pred_{ps}/"', page)
        self.assertRegex(page, rf'href="/review/pred_{ps}/\?img=000[123]\.png"')
        st, h, _ = self.post(f"/predict/{ps}/review", {}, "basic")
        self.assertEqual((st, T.hget(h, "location")), (303, f"/review/pred_{ps}/"))
        # Results: computed in a thread; known answer = the golden fixture's expected counts (no edits)
        st, h, _ = self.post(f"/predict/{ps}/results", {}, "basic")
        self.assertEqual(st, 303)
        self.wait_file(d / "results" / "results.json")
        for _ in range(50):
            if P.task("results", ps) is None:
                break
            time.sleep(0.1)
        res = json.loads((d / "results" / "results.json").read_text(encoding="utf-8"))
        exp = json.loads((GOLDEN.parent.parent / "expected_results_noedit.json").read_text(encoding="utf-8"))
        for s in ("S1", "S2"):
            for f in ("n_images", "ev_total", "nonev_total", "inner_layers_excluded"):
                self.assertEqual(res["samples"][s][f], exp["samples"][s][f], (s, f))
        page = self.page(f"/predict/{ps}/", "basic")
        self.assertIn(f'href="/results/{ps}/"', page)
        api = json.loads(self.get(f"/predict/api/set/{ps}?html=1", "basic")[2])
        self.assertEqual((api["state"], api["live"], api["review_href"], api["results_href"]),
                         ("done", False, f"/review/pred_{ps}/", f"/results/{ps}/"))
        self.assertEqual(set(api["html"]), {"status", "actions"})

    def test_nm_confirmation_required(self):
        before = self.sets()
        st, _, body = self.submit_a("basic", confirm=None)
        self.assertEqual(st, 400)
        self.assertIn("I confirm the pixel size", body)
        self.assertEqual(self.sets(), before)
        st, _, body = self.submit_a("basic", nm_g1="")                   # a sample without nm/px: refused too
        self.assertEqual(st, 400)
        self.assertIn("enter the pixel size", body)
        st, _, body = self.submit_a("basic", nm_g0="1472")               # a typo outside 0.01-100
        self.assertEqual(st, 400)
        self.assertEqual(self.sets(), before)
        self.assertFalse((self.root / P.PORTAL_TMP).exists() and any((self.root / P.PORTAL_TMP).iterdir()))

    def test_origin_check(self):
        before = self.sets()
        self.assertEqual(self.post(f"/predict/new/session/{self.key_a}/", self.form_a(), "basic", origin=False)[0],
                         403)
        self.assertEqual(self.post(f"/predict/new/session/{self.key_a}/", self.form_a(), "basic",
                                   origin="https://evil.example")[0], 403)
        self.assertEqual(self.sets(), before)
        st, _, _ = self.submit_a("basic")                                  # control: same form, own Origin
        self.assertEqual(st, 303)
        ps = (set(self.sets()) - set(before)).pop()
        self.wait_state(ps)

    def test_count_spreads_over_samples(self):
        before = set(self.sets())
        st, loc, body = self.submit_a("basic", how="count", count="2")
        self.assertEqual(st, 303, body[:500])
        ps = (set(self.sets()) - before).pop()
        rows = read_rows(self.root / ps / "images.csv")
        self.assertEqual(sorted(r["sample_label"] for r in rows), ["S1", "S2"])
        self.wait_state(ps)

    def test_pick_is_proportional_and_deterministic(self):
        groups = [{"label": "a", "names": [f"a{i:02d}" for i in range(10)]},
                  {"label": "b", "names": [f"b{i:02d}" for i in range(30)]}]
        got = P.pick(groups, 8)
        self.assertEqual(got, P.pick(groups, 8))
        self.assertEqual((sum(n.startswith("a") for n in got), sum(n.startswith("b") for n in got)), (2, 6))
        self.assertEqual(len(P.pick(groups, None)), 40)
        self.assertEqual(len(P.pick(groups, 1)), 1)

    def test_source_without_manifest_needs_typed_nm(self):
        page = self.page(f"/predict/new/session/{self.key_b}/", "basic")
        self.assertIn('name="label"', page)
        self.assertIn("not recorded: enter it", page)
        sig = re.search(r'name="sig" value="([0-9a-f]+)"', page).group(1)
        before = set(self.sets())
        data = {"sig": sig, "samples": ["g0"], "nm_g0": "", "how": "all", "confirm": "yes", "model": MODEL,
                "label": "Prep B"}
        self.assertEqual(self.post(f"/predict/new/session/{self.key_b}/", data, "basic")[0], 400)
        data["nm_g0"] = "1.5"
        st, _, body = self.post(f"/predict/new/session/{self.key_b}/", data, "basic")
        self.assertEqual(st, 303, body[:500])
        ps = (set(self.sets()) - before).pop()
        row = read_rows(self.root / ps / "images.csv")[0]
        self.assertEqual((row["sample_label"], float(row["nm_per_px"]), row["source"]),
                         ("Prep B", 1.5, "session:20261028"))
        self.wait_state(ps)


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

    def session(self, width, scheme, who="basic"):
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
        self.assertTrue(all(u.startswith(self.p.origin + "/") for u in rec["requests"]))
        self.assertEqual(rec["errors"], [])
        self.assertEqual(rec["console"], [])

    def fake_set(self, owner, state, **st):
        """A set copied from the golden fixture under a new id (status as given): for the page views only."""
        ps = f"ps_20261105_{uuid.uuid4().hex[:6]}"
        d = self.root / ps
        shutil.copytree(GOLDEN, d)
        req = json.loads((d / "request.json").read_text(encoding="utf-8"))
        req.update(ps=ps, owner=owner, created=time.strftime("%Y-%m-%dT%H:%M:%S"))
        (d / "request.json").write_text(json.dumps(req), encoding="utf-8")
        status = {"state": state, "n_done": 0, "n_total": 3, "where": "laptop", "job": None, "error": None,
                  "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": None, "waiting": None}
        status.update(st)
        (d / "status.json").write_text(json.dumps(status), encoding="utf-8")
        return ps

    def test_pages_sizes_and_schemes(self):
        ps = self.make_done_set("basic")
        self.fake_set("bas", "running", n_done=1)
        self.fake_set("bas", "queued", waiting="gpu busy: pid 1 python.exe 900 MB")
        pages = {"list": "/predict/", "new": "/predict/new/", "session": f"/predict/new/session/{self.key_a}/",
                 "set": f"/predict/{ps}/"}
        for scheme in ("light", "dark"):
            for width in (390, 1280):
                ctx, page, rec = self.session(width, scheme)
                try:
                    for name, path in pages.items():
                        with self.subTest(width=width, scheme=scheme, page=name):
                            page.goto(self.p.origin + path)
                            page.wait_for_load_state("networkidle")
                            page.wait_for_selector("#cj-bar")
                            sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
                            self.assertLessEqual(sw, iw, "horizontal scroll")
                            self.assertEqual(page.locator("main img").count(), 0)   # the XSS label stayed text
                            bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
                            self.assertEqual(bg, "rgb(15, 21, 27)" if scheme == "dark" else "rgb(243, 245, 247)")
                            page.screenshot(path=str(self.shots / f"predict_{name}_{width}_{scheme}.png"),
                                            full_page=True)
                    self.check_clean(rec)
                finally:
                    ctx.close()

    def test_session_form_counts_and_status_refresh(self):
        ctx, page, rec = self.session(390, "light")
        try:
            page.goto(f"{self.p.origin}/predict/new/session/{self.key_a}/")
            page.wait_for_load_state("networkidle")
            self.assertIn("3 tiles will be analysed", page.text_content("#cj-predict-sel"))
            page.uncheck('input[name="samples"][value="g1"]')
            self.assertIn("2 tiles will be analysed", page.text_content("#cj-predict-sel"))
            ps = self.fake_set("bas", "running", n_done=1)
            page.goto(f"{self.p.origin}/predict/{ps}/")
            page.wait_for_load_state("networkidle")
            self.assertIn("1 of 3 images done", page.text_content("#cj-predict-status"))
            st = json.loads((self.root / ps / "status.json").read_text(encoding="utf-8"))
            st.update(state="done", n_done=3)
            (self.root / ps / "status.json").write_text(json.dumps(st), encoding="utf-8")
            with page.expect_response(lambda r: f"/predict/api/set/{ps}?html=1" in r.url, timeout=15000):
                pass
            page.wait_for_selector(f'form[action="/predict/{ps}/review"]', timeout=15000)
            self.assertIn("Done", page.text_content("#cj-predict-status"))
            page.screenshot(path=str(self.shots / "predict_set_refreshed_390_light.png"), full_page=True)
            self.check_clean(rec)
        finally:
            ctx.close()


if __name__ == "__main__":
    unittest.main()
