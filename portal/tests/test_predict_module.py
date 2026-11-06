"""
Predict module (portal/modules/predict.py): roles through the Gate, the session-pick flow end to end with the fake
predictor (jobs/tests/fake_predictor.py through the jobs test hooks; never the real GPU), the set page reaching
done, Review / Results building the package and results.json in temp roots (known answer: the golden fixture's
expected counts), the nm/px confirmation (no tick = refused, nothing written), ownership (someone else's set = the
same 404 as no set), the Origin check, uploads (caps, content checks: a non-image and a 16-bit PNG refused, the
scratch folder emptied), escaping, and the pages in a browser (Playwright, Edge channel: 390 and 1280 px, light and
dark, no horizontal scroll, same-origin requests only, no console errors, the status refresh). Screenshots go to
$CJ_SHOT_DIR when set (else a temp folder, removed afterwards).

Temp roster / registry / predict and review roots only; never the real ones.

  python -m unittest discover -s portal/tests -v
"""
import csv
import errno
import html
import http.client
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

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


def multipart(fields: dict, files: list) -> tuple:
    """-> (content-type, body) for fields {name: value} and files [(field, filename, bytes)]."""
    b = "cjb" + uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode("utf-8"))
    for field, name, data in files:
        out.write(f'--{b}\r\nContent-Disposition: form-data; name="{field}"; filename="{name}"\r\n'
                  f'Content-Type: application/octet-stream\r\n\r\n'.encode("utf-8"))
        out.write(data + b"\r\n")
    out.write(f"--{b}--\r\n".encode())
    return f"multipart/form-data; boundary={b}", out.getvalue()


def png16() -> bytes:
    from PIL import Image
    im = Image.new("I;16", (64, 64), 1000)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


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
        cls.p = T.Portal(cls.env.config(manifest=cls.manifest, review_root=cls.review))
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
    def cfg(self):
        """The module's view of the portal config (what routes(ctx) gets as ctx.cfg)."""
        return types.SimpleNamespace(registry=self.env.registry, review_root=self.review)

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

    def follow(self, loc, who="basic", timeout=120):
        """A 303 to the progress page (the set is made in the background) -> where it ends: /predict/<ps>/."""
        if not (loc or "").startswith("/predict/preparing/"):
            return loc
        tid = loc.rstrip("/").split("/")[-1]
        t0 = time.time()
        while time.time() - t0 < timeout:
            st, _, body = self.get(f"/predict/api/preparing/{tid}", who)
            j = json.loads(body)
            if j.get("href"):
                return j["href"]
            if j.get("state") == "failed":
                self.fail(f"create failed: {j}")
            time.sleep(0.1)
        self.fail(f"{loc} not done after {timeout} s")

    def submit_a(self, who="basic", **kw):
        st, h, body = self.post(f"/predict/new/session/{self.key_a}/", self.form_a(who, **kw), who)
        loc = T.hget(h, "location")
        return st, (self.follow(loc, who) if st == 303 else loc), body.decode("utf-8")

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
            for path in ("/predict/", "/predict/new/", "/predict/new/upload/", "/predict/predict.js",
                         "/predict/api/mine", f"/predict/new/session/{self.key_a}/"):
                with self.subTest(who=who, path=path):
                    self.assertEqual(self.get(path, who)[0], 200)
        self.assertEqual(T.request(self.p.port, "GET", "/predict/api/mine")[0], 401)

    def test_script_is_a_file_no_inline_script(self):
        st, h, body = self.get("/predict/predict.js")
        self.assertEqual(st, 200)
        self.assertTrue(T.hget(h, "content-type").startswith("text/javascript"))
        for path in ("/predict/", "/predict/new/", f"/predict/new/session/{self.key_a}/", "/predict/new/upload/"):
            page = self.page(path)
            self.assertIn('<script src="/predict/predict.js" defer></script>', page)
            self.assertEqual(re.findall(r"<script(?![^>]*\ssrc=)[^>]*>", page), [], path)
            bare = re.sub(r'"[^"]*"', '""', page)             # attribute values are escaped text, not handlers
            self.assertEqual(re.findall(r"<[a-z][^>]*\son[a-z]+\s*=", bare), [], path)   # no inline handlers

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
        self.assertIn("<select name=\"model\" required><option value=\"\" disabled>Choose…</option>", adv)
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
        self.assertNotIn(f'<p class="cj-mut cj-wrap">{ps}</p>', page)    # nor the set id
        self.assertIn(f'<p class="cj-mut cj-wrap">{ps}</p>', self.page(f"/predict/{ps}/", "admin"))
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
        for _ in range(100):
            if P.task("review", ps) is None:
                break
            time.sleep(0.1)
        self.assertIsNone(P.task("review", ps))                           # finished without an error
        self.assertTrue((pkg / "data.js").is_file())
        self.assertEqual(len(list((pkg / "images").glob("*.jpg"))), 3)
        # the package is in the registry, once (a second registration adds nothing)
        recs = [json.loads(x) for x in (self.env.registry / "review_package.jsonl").read_text(
            encoding="utf-8").splitlines() if x.strip()]
        self.assertEqual([(r["id"], r["kind"], r["owner"], r["prediction_set"]) for r in recs],
                         [(f"pred_{ps}", "prediction", "bas", ps)])
        self.assertIsNone(P.register_package(self.cfg(), pkg))
        self.assertEqual(len((self.env.registry / "review_package.jsonl").read_text(encoding="utf-8").splitlines()),
                         1)
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
        st, h, body = self.post(f"/predict/new/session/{self.key_b}/", data, "basic")
        self.assertEqual(st, 303, body[:500])
        self.follow(T.hget(h, "location"))
        ps = (set(self.sets()) - before).pop()
        row = read_rows(self.root / ps / "images.csv")[0]
        self.assertEqual((row["sample_label"], float(row["nm_per_px"]), row["source"]),
                         ("Prep B", 1.5, "session:20261028"))
        self.wait_state(ps)


def stamp(seconds_ago: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - seconds_ago))


class StatusRules(unittest.TestCase):
    """status_view / can_restart on status.json shapes (predict_local keeps `updated` fresh while it runs or waits)."""

    def st(self, state, age=0, **kw):
        return {"state": state, "n_done": 1, "n_total": 3, "where": "laptop", "job": None, "error": None,
                "updated": stamp(age), "pid": None, "waiting": None, **kw}

    def test_queued_waiting_with_a_heartbeat(self):
        for waiting, text in (("another prediction run holds predict.lock; waiting", "another analysis is running"),
                              ("gpu busy: pid 4242 python.exe 900 MB", "another job is using it"),
                              ("restart requested", "Starting again")):
            with self.subTest(waiting=waiting):
                st = self.st("queued", 30, waiting=waiting)
                v = P.status_view(st, admin=False)
                self.assertEqual(v["label"], "Waiting to start")
                self.assertIn(text, v["detail"])
                self.assertIsNone(v["tech"])                               # the raw reason only for admins
                self.assertIn(waiting, P.status_view(st, admin=True)["tech"])
                self.assertFalse(P.can_restart(st))                        # predict_local is on it: no button

    def test_stale_and_failed_offer_start_again(self):
        for state in ("queued", "running"):
            with self.subTest(state=state):
                st = self.st(state, P.STALE_S + 60, waiting="gpu busy: pid 1" if state == "queued" else None)
                self.assertTrue(P.can_restart(st))
                v = P.status_view(st, admin=False)
                self.assertEqual((v["label"], v["tone"]), ("No progress", "warn"))
                self.assertIn("Start again", v["detail"])
                self.assertFalse(P.can_restart(self.st(state, P.STALE_S - 60)))   # control: fresh
        self.assertTrue(P.can_restart(self.st("failed", 0, error="boom")))
        self.assertFalse(P.can_restart(self.st("done", 10 ** 6)))
        self.assertFalse(P.can_restart(self.st("queued", 10 ** 6, job="job_20261105_predict_x")))  # desktop queue
        v = P.status_view(self.st("running", 5), admin=False)
        self.assertEqual((v["label"], v["detail"]), ("Running", "1 of 3 images done"))


class PixelTolerance(unittest.TestCase):
    """BUILD_CONTRACTS.md section 8: a sample's nm/px values within 1 % (spread / median) are one scale; each tile
    keeps its own value; a typed value applies to all tiles of the sample (recorded); over 1 % refused."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src_dir = self.tmp / "tiles"
        self.src_dir.mkdir()
        # A 1.000 / 1.009 (0.896 %), B 1.000 / 1.011 (1.094 %), C 1.4724 x2 (control: one value)
        self.nm = {"a1.png": ("A", "1.000"), "a2.png": ("A", "1.009"), "b1.png": ("B", "1.000"),
                   "b2.png": ("B", "1.011"), "c1.png": ("C", "1.4724"), "c2.png": ("C", "1.4724")}
        for n in self.nm:
            shutil.copyfile(GOLDEN / "images" / "0001.png", self.src_dir / n)
        (self.src_dir / "MANIFEST.csv").write_text("output_name,sample_label,nm_per_px\n" + "".join(
            f"{n},{s},{v}\n" for n, (s, v) in self.nm.items()), encoding="utf-8")
        self.src = {"key": "k", "label": "Tol", "dir": self.src_dir, "manifest": self.src_dir / "MANIFEST.csv",
                    "session": "20261027", "sample_column": None}
        self.cfg = types.SimpleNamespace(state_dir=self.tmp / "state", registry=self.tmp / "reg",
                                         cj_predict_root=str(self.tmp / "predict"))
        self.user = {"id": "bas", "role": "basic"}

    def tearDown(self):
        self._tmp.cleanup()

    def groups(self):
        return {g["label"]: g for g in P.tiles_of(self.src)["groups"]}

    def test_constant_pinned_to_results(self):
        import importlib
        RC = importlib.import_module("results.compute")
        from portal.modules import results as PR
        self.assertEqual(P.NM_TOLERANCE, RC.NM_TOLERANCE)                # all four copies pinned equal
        self.assertEqual(P.NM_TOLERANCE, PR.NM_TOLERANCE)
        self.assertEqual(P.NM_TOLERANCE, P.PS.NM_TOLERANCE)              # jobs/predict_set.py create()
        self.assertEqual(P.NM_TOLERANCE, 0.01)
        self.assertEqual((P.NM_EPS, RC.NM_EPS, P.PS.NM_EPS), (1e-12, 1e-12, 1e-12))
        for vals in ([1.4724], [1.47125, 1.4724], [1.0, 1.009], [1.0, 1.011], [1.0, 1.0, 1.02], [1.0, 1.005, 1.011],
                     [0.7362, 1.4724], [2.0, 1.0, 1.0, 1.01], [1.0, 1.0, 1.01]):
            with self.subTest(vals=vals):
                self.assertAlmostEqual(P.nm_spread(vals), RC.nm_spread(vals), places=15)
                self.assertEqual(P.one_scale(vals), RC.one_scale(vals))
                self.assertEqual(P.one_scale(vals), P.PS.nm_one_scale(vals))
        # exactly 1 %: (1.01 - 1.0) / 1.0 is 0.010000000000000009 in floats; the epsilon keeps it accepted everywhere
        self.assertGreater((1.01 - 1.0) / 1.0, 0.01)
        self.assertTrue(P.one_scale([1.0, 1.0, 1.01]))
        self.assertTrue(RC.one_scale([1.0, 1.0, 1.01]))
        self.assertTrue(P.PS.nm_one_scale([1.0, 1.0, 1.01]))
        self.assertFalse(P.one_scale([1.0, 1.0, 1.0101]))                # control: just over

    def test_grouping_boundary(self):
        g = self.groups()
        self.assertEqual((g["A"]["nm_range"], g["A"]["mixed"], g["A"]["nm"]), (True, False, None))
        self.assertEqual((g["A"]["nm_min"], g["A"]["nm_max"]), (1.0, 1.009))
        self.assertEqual(g["A"]["nm_of"], {"a1.png": 1.0, "a2.png": 1.009})
        self.assertEqual((g["B"]["nm_range"], g["B"]["mixed"]), (False, True))
        self.assertEqual((g["C"]["nm_range"], g["C"]["mixed"], g["C"]["nm"]), (False, False, 1.4724))   # control

    def test_missing_value_still_needs_typing(self):
        (self.src_dir / "MANIFEST.csv").write_text("output_name,sample_label,nm_per_px\na1.png,A,1.000\n"
                                                   "a2.png,A,\n", encoding="utf-8")
        for n in list(self.nm)[2:]:
            (self.src_dir / n).unlink()
        g = self.groups()["A"]
        self.assertEqual((g["nm"], g["nm_range"], g["mixed"]), (None, False, False))

    def test_form_shows_range_and_refuses_mixed(self):
        t = P.tiles_of(self.src)
        with mock.patch.object(P, "model_block", return_value=""):
            page = P.page_session_form(self.cfg, self.user, self.src, t)
        self.assertIn("1–1.009 nm/px across tiles, within 1 %; each tile keeps its own value", page)
        a_input = re.search(r'<input[^>]*name="nm_g0"[^>]*>', page).group(0)
        self.assertNotIn("required", a_input)                       # empty = each tile's own
        self.assertIn('name="confirm" value="yes" required', page)  # the tick is still required
        self.assertIn("differ by more than 1 % (1, 1.011 nm/px)", page)
        self.assertNotIn('name="nm_g1"', page)                       # B: no input, its box disabled
        self.assertIn('value="1.4724" required', page)              # C: as before

    def submit(self, **form):
        """session_submit with PS.create captured (the manifest it gets and its keywords); nothing starts."""
        seen = {}

        def fake_create(d_from, **kw):
            with open(kw["manifest"], newline="", encoding="utf-8") as f:
                seen["rows"] = {r["output_name"]: r for r in csv.DictReader(f)}
            seen["kw"] = kw
            d = self.tmp / "predict" / "ps_20261105_aaaaaa"
            d.mkdir(parents=True, exist_ok=True)
            return d

        t = P.tiles_of(self.src)
        data = {"confirm": "yes", "sig": t["sig"], "samples": ["g0", "g2"], "nm_g0": "", "nm_g2": "1.4724",
                "how": "all", "model": MODEL}
        data.update(form)
        with mock.patch.object(P.PS, "create", side_effect=fake_create), \
                mock.patch.object(P.PL, "start_detached"), mock.patch.object(P, "check_model", return_value=None), \
                mock.patch.object(P, "check_active"):
            P.session_submit(self.cfg, self.user, self.src, data)
        return seen

    def audit_lines(self):
        p = self.cfg.state_dir / "audit" / "portal.jsonl"
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.is_file() else []

    def test_each_tile_keeps_its_own_value(self):
        seen = self.submit()
        self.assertEqual({n: float(r["nm_per_px"]) for n, r in seen["rows"].items()},
                         {"a1.png": 1.0, "a2.png": 1.009, "c1.png": 1.4724, "c2.png": 1.4724})
        self.assertEqual(seen["kw"]["nm_confirmed_by"], "bas")     # no silent defaults: who confirmed
        self.assertTrue(seen["kw"]["nm_from_manifest"])
        self.assertEqual(self.audit_lines(), [])                    # nothing typed: nothing to record

    def test_typed_value_applies_to_all_tiles_and_is_recorded(self):
        seen = self.submit(nm_g0="1.005")
        self.assertEqual([float(seen["rows"][n]["nm_per_px"]) for n in ("a1.png", "a2.png")], [1.005, 1.005])
        self.assertEqual(seen["kw"]["nm_confirmed_by"], "bas")
        (rec,) = self.audit_lines()
        self.assertEqual((rec["actor"], rec["action"], rec["target"]), ("bas", "predict.nm_override",
                                                                         "ps_20261105_aaaaaa"))
        self.assertEqual(rec["detail"], {"sample": "A", "typed_nm_per_px": 1.005, "manifest_nm_per_px": "1–1.009",
                                         "tiles": 2})

    def test_mixed_sample_refused_and_bad_typed_value(self):
        with self.assertRaises(P.Refusal) as e:
            self.submit(samples=["g1"])
        self.assertIn("B: pixel sizes differ by more than 1 %", e.exception.msg)
        with self.assertRaises(P.Refusal) as e:
            self.submit(nm_g0="abc")
        self.assertIn("enter the pixel size as a number", e.exception.msg)

    def test_no_tick_refused(self):
        with self.assertRaises(P.Refusal):
            self.submit(confirm="")

    def test_changed_values_change_the_signature(self):
        sig = P.tiles_of(self.src)["sig"]
        (self.src_dir / "MANIFEST.csv").write_text((self.src_dir / "MANIFEST.csv").read_text(encoding="utf-8")
                                                   .replace("a2.png,A,1.009", "a2.png,A,1.008"), encoding="utf-8")
        self.assertNotEqual(P.tiles_of(self.src)["sig"], sig)

    def test_set_page_shows_the_range(self):
        rows = [{"sample_label": "A", "nm_per_px": "1.0", "nm_confirmed_by": "bas"},
                {"sample_label": "A", "nm_per_px": "1.009", "nm_confirmed_by": "bas"},
                {"sample_label": "C", "nm_per_px": "1.4724", "nm_confirmed_by": "bas"}]
        got = {s["label"]: s["nm_per_px"] for s in P.samples_of(rows)}
        self.assertEqual(got, {"A": "1–1.009 (each image its own)", "C": "1.4724"})

    def test_split_by_tile_size_real_shape(self):
        """The 20261027 shape: the liposome sample has 297 tiles at 1440x1023 (1.4724) and 96 at 352x250 (1.47125,
        the 45k re-run); sizes from the manifest's output_size (the files are not read). Control: the carbon
        sample, one size, is not split."""
        d = self.tmp / "real_shape"
        d.mkdir()
        lines = ["output_name,sample_label,output_size,nm_per_px"]
        for i in range(297):
            lines.append(f"L{i:03d}.png,Liposomes,1440x1023,1.47240")
        for i in range(96):
            lines.append(f"S{i:03d}.png,Liposomes,352x250,1.47125")
        for i in range(159):
            lines.append(f"C{i:03d}.png,Carbon,1440x1023,1.47240")
        for ln in lines[1:]:
            (d / ln.split(",")[0]).write_bytes(b"")      # never opened: output_size gives the size
        (d / "MANIFEST.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        t = P.tiles_of({**self.src, "dir": d, "manifest": d / "MANIFEST.csv"})
        got = [(g["label"], len(g["names"]), g["nm"], g["nm_range"], g["mixed"]) for g in t["groups"]]
        self.assertEqual(got, [("Carbon", 159, 1.4724, False, False),
                               ("Liposomes · 352×250 tiles", 96, 1.47125, False, False),
                               ("Liposomes · 1440×1023 tiles", 297, 1.4724, False, False)])
        self.assertEqual(t["n"], 552)
        with mock.patch.object(P, "model_block", return_value=""):
            page = P.page_session_form(self.cfg, self.user, {**self.src, "dir": d}, t)
        self.assertEqual(page.count('name="samples"'), 3)                # one tick per group
        self.assertIn("Liposomes · 352×250 tiles", page)

    def test_split_sizes_from_headers(self):
        from PIL import Image
        d = self.tmp / "hdr"
        d.mkdir()
        for n, size in (("a.png", (64, 48)), ("b.png", (64, 48)), ("c.png", (48, 64))):
            Image.new("L", size, 128).save(d / n)
        (d / "MANIFEST.csv").write_text("output_name,sample_label,nm_per_px\na.png,A,1.0\nb.png,A,1.0\nc.png,A,1.0\n",
                                        encoding="utf-8")
        t = P.tiles_of({**self.src, "dir": d, "manifest": d / "MANIFEST.csv"})
        self.assertEqual([(g["label"], g["names"]) for g in t["groups"]],
                         [("A · 48×64 tiles", ["c.png"]), ("A · 64×48 tiles", ["a.png", "b.png"])])
        self.assertEqual(len(self.groups()), 3)                         # control: the setUp tiles, one size each

    def test_range_outside_bounds_refused(self):
        """A range left empty is still checked against NM_MIN-NM_MAX: 1000 / 1005 (0.5 %) is one scale but no
        plausible pixel size."""
        (self.src_dir / "MANIFEST.csv").write_text("output_name,sample_label,nm_per_px\na1.png,A,1000\n"
                                                   "a2.png,A,1005\n", encoding="utf-8")
        for n in list(self.nm)[2:]:
            (self.src_dir / n).unlink()
        self.assertTrue(self.groups()["A"]["nm_range"])
        with self.assertRaises(P.Refusal) as e:
            self.submit(samples=["g0"])
        self.assertIn("A: 1000 nm per pixel (from the manifest) is outside 0.01-100", e.exception.msg)
        seen = self.submit(samples=["g0"], nm_g0="1.4724")              # control: a typed plausible value
        self.assertEqual({float(r["nm_per_px"]) for r in seen["rows"].values()}, {1.4724})

    def test_typed_value_over_partly_missing_values_is_recorded(self):
        (self.src_dir / "MANIFEST.csv").write_text("output_name,sample_label,nm_per_px\na1.png,A,1.000\n"
                                                   "a2.png,A,\n", encoding="utf-8")
        for n in list(self.nm)[2:]:
            (self.src_dir / n).unlink()
        seen = self.submit(samples=["g0"], nm_g0="1.2")
        self.assertEqual([float(seen["rows"][n]["nm_per_px"]) for n in ("a1.png", "a2.png")], [1.2, 1.2])
        (rec,) = self.audit_lines()
        self.assertEqual(rec["detail"], {"sample": "A", "typed_nm_per_px": 1.2,
                                         "manifest_nm_per_px": "1 (1 tile(s) without a value)", "tiles": 2})

    def test_audit_failure_is_logged_and_kept_for_the_admin(self):
        self.cfg.state_dir.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.state_dir.write_text("a file, not a folder", encoding="utf-8")   # the audit folder can't be made
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            seen = self.submit(nm_g0="1.005")
        self.assertEqual([float(seen["rows"][n]["nm_per_px"]) for n in ("a1.png", "a2.png")], [1.005, 1.005])
        self.assertIn("ps_20261105_aaaaaa: nm/px override not written to the audit log", err.getvalue())
        self.assertIn("nm/px override not written", P.audit_problem("ps_20261105_aaaaaa"))
        self.assertIsNone(P.audit_problem("ps_20261105_bbbbbb"))


class PixelToleranceCreate(Base):
    """Through the real jobs.predict_set.create: a sample within the 1 % tolerance must make a set whose images.csv
    has each tile's own nm/px (the 20261027 liposome pair, 1.47125 / 1.4724)."""

    def test_range_sample_creates_a_set(self):
        d = self.env.tmp / "sessions" / "tol_tiles"
        d.mkdir(parents=True, exist_ok=True)
        for n in ("t1.png", "t2.png"):
            shutil.copyfile(GOLDEN / "images" / "0001.png", d / n)
        (d / "MANIFEST.csv").write_text("output_name,sample_label,nm_per_px\nt1.png,Lipo,1.47125\nt2.png,Lipo,1.4724\n",
                                        encoding="utf-8")
        src = {"key": "tol", "label": "Tol", "dir": d, "manifest": d / "MANIFEST.csv", "session": "20261027",
               "sample_column": None}
        t = P.tiles_of(src)
        cfg = types.SimpleNamespace(registry=self.env.registry, review_root=self.review, state_dir=self.env.state)
        with mock.patch.object(P.PL, "start_detached"):
            ps_dir = P.session_submit(cfg, {"id": "bas", "role": "basic"}, src,
                                      {"confirm": "yes", "sig": t["sig"], "samples": ["g0"], "nm_g0": "",
                                       "how": "all", "model": MODEL})
        rows = read_rows(ps_dir / "images.csv")
        self.assertEqual([(r["orig_name"], float(r["nm_per_px"]), r["nm_confirmed_by"]) for r in rows],
                         [("t1.png", 1.47125, "bas"), ("t2.png", 1.4724, "bas")])

    def test_split_groups_pass_create(self):
        """Known answer: one sample with two tile sizes (the 20261027 shape, small) is split by tile size; both
        groups ticked make ONE set that create() accepts, the size in each sample_label, each tile's own nm/px.
        Control: the same tiles as one sample label go straight to create() and are refused (mixed tile sizes)."""
        from PIL import Image
        d = self.env.tmp / "sessions" / "split_tiles"
        d.mkdir(parents=True, exist_ok=True)
        Image.new("L", (144, 102), 120).save(d / "big1.png")
        Image.new("L", (144, 102), 130).save(d / "big2.png")
        Image.new("L", (35, 25), 140).save(d / "small1.png")
        (d / "MANIFEST.csv").write_text("output_name,sample_label,output_size,nm_per_px\n"
                                        "big1.png,Lipo,144x102,1.4724\nbig2.png,Lipo,144x102,1.4724\n"
                                        "small1.png,Lipo,35x25,1.47125\n", encoding="utf-8")
        src = {"key": "split", "label": "Split", "dir": d, "manifest": d / "MANIFEST.csv", "session": "20261027",
               "sample_column": None}
        t = P.tiles_of(src)
        self.assertEqual([g["label"] for g in t["groups"]], ["Lipo · 35×25 tiles", "Lipo · 144×102 tiles"])
        cfg = types.SimpleNamespace(registry=self.env.registry, review_root=self.review, state_dir=self.env.state)
        with mock.patch.object(P.PL, "start_detached"):
            ps_dir = P.session_submit(cfg, {"id": "bas", "role": "basic"}, src,
                                      {"confirm": "yes", "sig": t["sig"], "samples": ["g0", "g1"], "nm_g0": "1.47125",
                                       "nm_g1": "1.4724", "how": "all", "model": MODEL})
        rows = read_rows(ps_dir / "images.csv")
        self.assertEqual([(r["orig_name"], r["sample_label"], float(r["nm_per_px"])) for r in rows],
                         [("big1.png", "Lipo · 144×102 tiles", 1.4724), ("big2.png", "Lipo · 144×102 tiles", 1.4724),
                          ("small1.png", "Lipo · 35×25 tiles", 1.47125)])
        with self.assertRaisesRegex(P.PS.PredictSetError, "mixed tile sizes"):
            P.PS.create(d, model=MODEL, owner="bas", nm_confirmed_by="bas", sample_label_column="manifest:sample_label",
                        nm_from_manifest=True, manifest=d / "MANIFEST.csv", registry_root=self.env.registry,
                        root=self.root)

    def test_audit_failure_shown_to_the_admin_only(self):
        ps = self.fake_set("bas", "done", n_done=3)
        with mock.patch("sys.stderr", io.StringIO()):
            P.record_nm_override(types.SimpleNamespace(state_dir=None), {"id": "bas"}, ps, {"sample": "S1"})
        self.assertIn("nm/px override not written to the audit log", html.unescape(self.page(f"/predict/{ps}/", "admin")))
        self.assertNotIn("override not written", self.page(f"/predict/{ps}/", "basic"))


class Limits(Base):
    def test_start_again_marks_the_set_before_spawning(self):
        ps = self.fake_set("bas", "running", updated=stamp(P.STALE_S + 60), pid=99999)
        seen = []

        def spawn(d, root=None, registry_root=None, allow_cpu=False):
            seen.append(json.loads((Path(d) / "status.json").read_text(encoding="utf-8")))
            return 1
        with mock.patch.object(P.PL, "start_detached", side_effect=spawn):
            st, h, _ = self.post(f"/predict/{ps}/start", {}, "basic")
            self.assertEqual((st, T.hget(h, "location")), (303, f"/predict/{ps}/"))
            self.assertEqual(len(seen), 1)
            self.assertEqual((seen[0]["state"], seen[0]["waiting"], seen[0]["pid"]), ("queued", "restart requested",
                                                                                      None))
            self.assertLess(P.age_s(seen[0], time.time()), 30)            # `updated` is fresh
            self.post(f"/predict/{ps}/start", {}, "basic")                # a double click: no second spawn
            self.assertEqual(len(seen), 1)
        page = self.page(f"/predict/{ps}/", "basic")
        self.assertIn("Starting again", page)
        self.assertNotIn(f'action="/predict/{ps}/start"', page)

    def test_basic_user_at_most_three_waiting_or_running(self):
        made = [self.fake_set("bas2", s) for s in ("queued", "running", "queued")]
        for ps in made:
            self.addCleanup(shutil.rmtree, self.root / ps, True)
        before = self.sets()
        st, _, body = self.submit_a("other")
        self.assertEqual(st, 429)
        self.assertIn("already have 3 analyses", body)
        ctype, mp = multipart({"sample_label": "S", "nm_per_px": "1.4", "confirm": "yes", "model": MODEL},
                              [("files", "a.png", (GOLDEN / "images" / "0001.png").read_bytes())])
        self.assertEqual(self.post("/predict/new/upload/", mp, "other", ctype=ctype)[0], 429)
        self.assertEqual(self.sets(), before)
        self.assertIsNone(P.check_active(self.cfg(), {"id": "bas2", "role": "advanced"}))   # advanced: no limit
        st_path = self.root / made[0] / "status.json"                       # control: one fails -> room for one
        st_ = json.loads(st_path.read_text(encoding="utf-8"))
        st_.update(state="failed")
        st_path.write_text(json.dumps(st_), encoding="utf-8")
        st, _, body = self.submit_a("other")
        self.assertEqual(st, 303, body[:400])
        self.wait_state((set(self.sets()) - set(before)).pop())

    def test_unreadable_session_is_a_plain_refusal(self):
        with mock.patch.object(P, "tiles_of", side_effect=OSError("[WinError 59] An unexpected network error")):
            st, _, body = self.get(f"/predict/new/session/{self.key_a}/", "basic")
            self.assertEqual(st, 503)
            self.assertIn("can’t be read right now", body.decode("utf-8"))
            self.assertNotIn("WinError", body.decode("utf-8"))
            self.assertIn("WinError", self.get(f"/predict/new/session/{self.key_a}/", "admin")[2].decode("utf-8"))
            st, _, body = self.post(f"/predict/new/session/{self.key_a}/",
                                    {"sig": "x", "samples": ["g0"], "nm_g0": "1.4", "how": "all", "confirm": "yes",
                                     "model": MODEL}, "basic")
            self.assertEqual(st, 503)
        self.assertEqual(self.get(f"/predict/new/session/{self.key_a}/", "basic")[0], 200)   # control

    def test_no_production_model_message(self):
        path = self.env.registry / "model.jsonl"
        keep = path.read_bytes()
        self.addCleanup(path.write_bytes, keep)
        path.write_text(json.dumps({"schema": "model/1", "id": MODEL, "status": "candidate",
                                    "logged": "2026-11-05T01:00:00"}) + "\n", encoding="utf-8")
        for page in (self.page("/predict/new/", "basic"), self.page(f"/predict/new/session/{self.key_a}/", "basic"),
                     self.page("/predict/new/upload/", "basic")):
            self.assertIn("No production model is set up yet", page)
        st, _, body = self.submit_a("basic")
        self.assertEqual(st, 400)
        self.assertIn("No production model", body)
        adv = self.page(f"/predict/new/session/{self.key_a}/", "advanced")     # advanced: must choose
        self.assertIn('<option value="" selected disabled>Choose…</option>', adv)
        self.assertNotIn("No production model is set up yet", self.page("/predict/new/", "advanced"))

    def test_sweep_old_scratch_only(self):
        tmp = self.root / P.PORTAL_TMP
        old, new = tmp / "up_old000000", tmp / "up_new000000"
        old.mkdir(parents=True)
        new.mkdir()
        (old / "x.png").write_bytes(b"x")
        t = time.time() - P.SCRATCH_MAX_AGE_S - 60
        os.utime(old, (t, t))
        self.assertEqual(P.sweep_scratch(self.cfg()), 1)
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())
        shutil.rmtree(new)

    def test_bounded_tables(self):
        with mock.patch.object(P, "CACHE_MAX", 3):
            for i in range(6):
                P._TASKS[("results", f"ps_x{i}")] = {"state": "failed", "error": "e", "started": 0}
            P.start_task("results", "ps_bounded", lambda: None)
            self.assertLessEqual(len(P._TASKS), 4)
        for k in [k for k in P._TASKS if str(k[1]).startswith(("ps_x", "ps_bounded"))]:
            P._TASKS.pop(k, None)


class Upload(Base):
    def up(self, fields, files, who="basic", origin=True):
        ctype, body = multipart(fields, files)
        st, h, out = self.post("/predict/new/upload/", body, who, origin=origin, ctype=ctype)
        return st, T.hget(h, "location"), out.decode("utf-8")

    def ok_fields(self, **kw):
        f = {"sample_label": "Upload S", "nm_per_px": "1.4724", "confirm": "yes", "model": MODEL}
        f.update(kw)
        return {k: v for k, v in f.items() if v is not None}

    def tiles(self, n=2):
        return [("files", f"my tile {i}.png", (GOLDEN / "images" / f"000{i}.png").read_bytes()) for i in range(1, n + 1)]

    def assert_scratch_empty(self):
        tmp = self.root / P.PORTAL_TMP
        self.assertFalse(tmp.exists() and any(tmp.iterdir()), list(tmp.iterdir()) if tmp.exists() else None)

    def test_upload_creates_a_set(self):
        before = set(self.sets())
        st, loc, body = self.up(self.ok_fields(sample_label="S " + XSS), self.tiles(2))
        self.assertEqual(st, 303, body[:800])
        ps = (set(self.sets()) - before).pop()
        self.assertEqual(loc, f"/predict/{ps}/")
        req = json.loads((self.root / ps / "request.json").read_text(encoding="utf-8"))
        self.assertEqual((req["owner"], req["source"]), ("bas", {"kind": "upload"}))
        rows = read_rows(self.root / ps / "images.csv")
        self.assertEqual([(r["image"], r["orig_name"], r["nm_confirmed_by"]) for r in rows],
                         [("0001", "my tile 1.png", "bas"), ("0002", "my tile 2.png", "bas")])
        self.assert_scratch_empty()
        page = self.page(f"/predict/{ps}/", "basic")
        self.assertIn("Uploaded images", page)
        self.assertIn("S &lt;img src=x onerror=alert(1)&gt;", page)
        self.assertNotIn(XSS, page)
        self.wait_state(ps)

    def test_content_checks(self):
        before = self.sets()
        cases = {"not a PNG or JPEG": [("files", "fake.png", b"MZ\x90\x00 not an image" * 10)],
                 "16-bit PNG": [("files", "deep.png", png16())]}
        for why, files in cases.items():
            with self.subTest(why=why):
                st, _, body = self.up(self.ok_fields(), files)
                self.assertEqual(st, 400)
                self.assertIn(why, body)
                self.assertNotIn(str(self.root), body)                    # no server paths in the message
                self.assertEqual(self.sets(), before)
                self.assert_scratch_empty()

    def test_caps(self):
        before = self.sets()
        with mock.patch.object(P, "MAX_FILES", 2):
            st, _, body = self.up(self.ok_fields(), self.tiles(3))
            self.assertEqual(st, 413)
            self.assertIn("More than 2 files", body)
        with mock.patch.object(P, "MAX_FILE_BYTES", 1000):
            st, _, body = self.up(self.ok_fields(), self.tiles(1))
            self.assertEqual(st, 413)
            self.assertIn("is over", body)
        self.assertEqual(self.sets(), before)
        self.assert_scratch_empty()

    def test_refusals_write_nothing(self):
        before = self.sets()
        for fields, files, why in (
                (self.ok_fields(confirm=None), self.tiles(1), "I confirm the pixel size"),
                (self.ok_fields(nm_per_px="abc"), self.tiles(1), "enter the pixel size"),
                (self.ok_fields(sample_label=""), self.tiles(1), "Sample name"),
                (self.ok_fields(), [("files", "a:b.png", b"x")], "File name not accepted"),
                (self.ok_fields(), self.tiles(1) + [("files", "MY TILE 1.png", b"x")], "Two files are named"),
                (self.ok_fields(), [], "Choose at least one image")):
            with self.subTest(why=why):
                st, _, body = self.up(fields, files)
                self.assertIn(st, (400, 413))
                self.assertIn(why, body)
        self.assertEqual(self.sets(), before)
        self.assert_scratch_empty()

    def test_upload_needs_origin(self):
        before = self.sets()
        st, _, _ = self.up(self.ok_fields(), self.tiles(1), origin=False)
        self.assertEqual(st, 403)
        self.assertEqual(self.sets(), before)

    def up_chunked(self, fields, files, who="basic"):
        """The same multipart body sent with Transfer-Encoding: chunked (no Content-Length), 4 KB per chunk."""
        ctype, body = multipart(fields, files)
        c = http.client.HTTPConnection("127.0.0.1", self.p.port, timeout=120)
        try:
            c.request("POST", "/predict/new/upload/", body=(body[i:i + 4096] for i in range(0, len(body), 4096)),
                      headers={**T.cookie(self.tok[who]), "Content-Type": ctype, "Origin": self.p.origin},
                      encode_chunked=True)
            r = c.getresponse()
            return r.status, r.getheader("location"), r.read().decode("utf-8")
        finally:
            c.close()

    def test_disk_checked_up_front_and_during_the_upload(self):
        before = self.sets()
        margin = P.FREE_MARGIN_BYTES
        # Content-Length known: refused before anything is stored when free - length < margin
        with mock.patch.object(P, "disk_free", return_value=margin + 10):
            st, _, body = self.up(self.ok_fields(), self.tiles(1))
        self.assertEqual(st, 507)
        self.assertIn("Not enough disk space", body)
        # chunked (no length to hold ahead): fine at the start, refused at the first file start when space is low
        calls = iter([margin + 10 ** 9] + [margin - 1] * 50)
        with mock.patch.object(P, "disk_free", side_effect=lambda p: next(calls)):
            st, _, body = self.up_chunked(self.ok_fields(), self.tiles(2))
        self.assertEqual(st, 507)
        self.assertIn("Not enough disk space", body)
        self.assertEqual(self.sets(), before)
        self.assert_scratch_empty()
        self.assertEqual(P._RESERVED, {})                                   # every hold released
        # control: the same chunked upload with room to spare goes through
        st, loc, body = self.up_chunked(self.ok_fields(), self.tiles(2))
        self.assertEqual(st, 303, body[:500])
        ps = (set(self.sets()) - set(before)).pop()
        self.assertEqual(len(read_rows(self.root / ps / "images.csv")), 2)
        self.wait_state(ps)

    def test_disk_check_every_64_mb(self):
        dest = Path(tempfile.mkdtemp(prefix="cj_sink_"))
        self.addCleanup(shutil.rmtree, dest, True)
        res = P.Reservation(dest, 0)
        self.addCleanup(res.release)
        n = {"calls": 0}

        def free(_):
            n["calls"] += 1
            return P.FREE_MARGIN_BYTES + (10 ** 12 if n["calls"] < 3 else -1)   # 3rd check: the disk filled up
        sink = P.UploadSink(dest, res)
        with mock.patch.object(P, "disk_free", side_effect=free), \
                mock.patch.object(P, "DISK_CHECK_BYTES", 1000), mock.patch.object(P, "MAX_FILE_BYTES", 10 ** 6):
            sink.apply([("part", "files", "a.png"), ("data", b"x" * 600)])       # check 1 (file start)
            sink.apply([("data", b"x" * 600)])                                     # check 2 (1200 bytes)
            sink.apply([("data", b"x" * 600)])
            with self.assertRaises(P.Refusal) as cm:
                sink.apply([("data", b"x" * 600)])                                 # check 3 (2400 bytes): refused
        sink.close()
        self.assertEqual(cm.exception.status, 507)

    def test_reservation_counts_other_uploads(self):
        dest = Path(tempfile.mkdtemp(prefix="cj_res_"))
        self.addCleanup(shutil.rmtree, dest, True)
        big = P.Reservation(dest, 10 ** 9)                                  # another upload still streaming 1 GB
        try:
            with mock.patch.object(P, "disk_free", return_value=P.FREE_MARGIN_BYTES + 5 * 10 ** 8):
                with self.assertRaises(P.Refusal) as cm:
                    P.Reservation(dest, 0).check()
                self.assertEqual(cm.exception.status, 507)
                big.written(6 * 10 ** 8)                                    # it has written most of it
                P.Reservation(dest, 0).check()                              # 0.4 GB still held: room again
        finally:
            big.release()
            P._RESERVED.clear()

    def test_fields_checked_before_files(self):
        """The form's text fields come first: a missing tick is refused at the first file part, before a byte of
        a too-big file is counted (400 for the tick, not 413 for the size)."""
        with mock.patch.object(P, "MAX_FILE_BYTES", 1000):
            st, _, body = self.up(self.ok_fields(confirm=None), self.tiles(1))
            self.assertEqual(st, 400)
            self.assertIn("I confirm the pixel size", body)
            st, _, body = self.up(self.ok_fields(nm_per_px="0"), self.tiles(1))
            self.assertEqual(st, 400)
            self.assertIn("enter the pixel size", body)
            st, _, _ = self.up(self.ok_fields(), self.tiles(1))             # control: valid fields reach the cap
            self.assertEqual(st, 413)
        page = self.page("/predict/new/upload/", "basic")
        self.assertLess(page.index('name="confirm"'), page.index('type="file"'))
        self.assertLess(page.index('name="model"'), page.index('type="file"'))
        self.assertIn("PixelSpacing", page)
        self.assert_scratch_empty()

    def test_os_error_text(self):
        self.assertIn("too long", P.os_error_text(OSError(errno.ENAMETOOLONG, "File name too long")))
        self.assertIn("could not store", P.os_error_text(OSError(errno.ENOSPC, "No space left on device")))


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

    def test_pages_sizes_and_schemes(self):
        ps = self.make_done_set("basic")
        self.fake_set("bas", "running", n_done=1)
        self.fake_set("bas", "queued", waiting="gpu busy: pid 1 python.exe 900 MB")
        pages = {"list": "/predict/", "new": "/predict/new/", "session": f"/predict/new/session/{self.key_a}/",
                 "upload": "/predict/new/upload/", "set": f"/predict/{ps}/"}
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
