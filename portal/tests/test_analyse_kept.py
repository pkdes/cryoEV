"""
"Analyse kept tiles" (BUILD_CONTRACTS.md section 14, piece AK): the Gallery button, the preselected session form, and
session_post re-deriving the tile list on the server from gallery/events.jsonl.

Known answer: a conversion of 10 tiles = 3 keep, 2 maybe, 2 reject, 1 keep-then-reject (latest wins -> reject), 2
unflagged (one has stars only). Expected: exactly the 3 keep tiles, 5 with kept_maybe=1.
Control: the "all tiles" path gives the same tile list as before this change (golden list, and the code of `main`
loaded next to it when git can show it).

Temp roster / state / raw root only.   python -m unittest portal.tests.test_analyse_kept -v
"""
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.parse
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402,F401
import test_gallery as TG  # noqa: E402
from portal.modules import gallery as G  # noqa: E402
from portal.modules import predict as P  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except Exception:                                            # noqa: BLE001
    sync_playwright = None

CID, CID_Z = TG.CID_A, TG.CID_B
NAMES = [f"g1_Grid_1__m1_r{i // 4}c{i % 4}_t{i:02d}__denoised.png" for i in range(10)]
KEEP, MAYBE, REJECT, FLIPPED, PLAIN = NAMES[0:3], NAMES[3:5], NAMES[5:7], NAMES[7], NAMES[8:10]
MODEL = "v5_evnonev_y8m_20261029"
FORM = {"Content-Type": "application/x-www-form-urlencoded"}


def flag_tiles(d: Path, who="alyssia"):
    c = G.Conv(d, json.loads((d / "convert.json").read_text()), {})
    for n in KEEP:
        G.append_event(c, who, n, "flag", "keep")
    for n in MAYBE:
        G.append_event(c, who, n, "flag", "maybe")
    for n in REJECT:
        G.append_event(c, who, n, "flag", "reject")
    G.append_event(c, who, FLIPPED, "flag", "keep")
    G.append_event(c, who, FLIPPED, "flag", "reject")            # latest wins
    G.append_event(c, who, PLAIN[0], "stars", 4)                  # stars alone are not a flag
    return c


class World(TG.World):
    def __init__(self):
        saved = TG.make_conv
        TG.make_conv = lambda raw, cid, owner, **kw: saved(raw, cid, owner, **{**kw, "names": NAMES, "n": 10, "montage": False})
        try:
            super().__init__()
        finally:
            TG.make_conv = saved
        self.cfg = self.env.config()
        flag_tiles(self.a)                                        # CID_B (self.b) has no flags: 0 kept

    def source(self, cid=CID):
        can_see = lambda u, o: o == u["id"] or u["role"] == "admin"  # noqa: E731
        return next(s for s in P.converted_sources(self.cfg, {"id": "alyssia", "role": "advanced"}, can_see)
                    if s["conversion"] == cid)


def tile_sig(page):
    return re.search(r'name="sig" value="([^"]+)"', page).group(1)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = World()
        cls.user = {"id": "alyssia", "role": "advanced"}

    @classmethod
    def tearDownClass(cls):
        cls.w.close()

    def session_path(self, cid=CID, query=""):
        return f"/predict/new/session/{self.w.source(cid)['key']}/{query}"

    def submit(self, cid=CID, **form):
        """session_submit (the server path of session_post) with PS.create captured -> the names it was given."""
        src = self.w.source(cid)
        t = P.kept_scope(self.w.cfg, src, form, P.read_tiles(src))
        data = {"confirm": "yes", "sig": t["sig"], "samples": ["g0"], "nm_g0": "1.4724", "how": "all", "model": MODEL}
        data.update(form)
        seen = {}

        def fake_create(d_from, **kw):
            with open(kw["manifest"], newline="", encoding="utf-8") as f:
                seen["names"] = sorted(r["output_name"] for r in csv.DictReader(f))
            d = Path(self.w.predict) / "ps_20261105_aaaaaa"
            d.mkdir(parents=True, exist_ok=True)
            return d

        with mock.patch.object(P.PS, "create", side_effect=fake_create), mock.patch.object(P.PL, "start_detached"), \
                mock.patch.object(P, "check_model", return_value=None), mock.patch.object(P, "check_active"):
            P.session_submit(self.w.cfg, self.user, src, data)
        return seen["names"]


class KnownAnswer(Base):
    def test_exactly_the_keep_tiles(self):
        self.assertEqual(self.submit(which="kept", **{"from": CID}), sorted(KEEP))

    def test_with_maybe(self):
        self.assertEqual(self.submit(which="kept", kept_maybe="1", **{"from": CID}), sorted(KEEP + MAYBE))

    def test_reject_flipped_and_unflagged_never_sent(self):
        for extra in ({}, {"kept_maybe": "1"}):
            got = set(self.submit(which="kept", **{"from": CID}, **extra))
            self.assertFalse(got & (set(REJECT) | {FLIPPED} | set(PLAIN)), extra)

    def test_posted_tile_list_is_ignored(self):
        got = self.submit(which="kept", names=NAMES, tiles=NAMES, files=json.dumps(NAMES), **{"from": CID})
        self.assertEqual(got, sorted(KEEP))

    def test_count_cannot_sample_the_kept_tiles(self):
        self.assertEqual(self.submit(which="kept", how="count", count="1", **{"from": CID}), sorted(KEEP))

    def test_kept_changes_after_the_form_was_opened(self):
        """The list is re-derived at post time, and the page's sig covers it: a stale form is refused."""
        src = self.w.source()
        old = P.kept_scope(self.w.cfg, src, {"which": "kept", "from": CID}, P.read_tiles(src))["sig"]
        c = G.Conv(self.w.a, json.loads((self.w.a / "convert.json").read_text()), {})
        G.append_event(c, "alyssia", NAMES[9], "flag", "keep")
        try:
            with self.assertRaises(P.Refusal) as cm:
                self.submit(which="kept", sig=old, **{"from": CID})
            self.assertIn("changed", cm.exception.msg)
            self.assertEqual(self.submit(which="kept", **{"from": CID}), sorted(KEEP + [NAMES[9]]))
        finally:
            G.append_event(c, "alyssia", NAMES[9], "flag", "")


class Control(Base):
    """The all-tiles path is what it was: golden list, and the code of `main` run on the same input."""

    def test_all_tiles_unchanged_golden(self):
        src = self.w.source()
        t = P.read_tiles(src)
        self.assertIs(P.kept_scope(self.w.cfg, src, {}, t), t)
        self.assertNotIn("kept", t)
        self.assertEqual(self.submit(), sorted(NAMES))
        self.assertEqual(len(self.submit(how="count", count="4")), 4)

    def test_same_as_main(self):
        root = Path(__file__).resolve().parents[2]
        try:
            old_src = subprocess.run(["git", "show", "main:portal/modules/predict.py"], cwd=root, capture_output=True,
                                     check=True, timeout=30).stdout.decode("utf-8")
        except (OSError, subprocess.SubprocessError):
            self.skipTest("git cannot show main")
        old = importlib.util.module_from_spec(importlib.util.spec_from_loader("predict_main_copy", loader=None))
        old.__dict__["__file__"] = str(root / "portal" / "modules" / "predict.py")
        exec(compile(old_src, "predict_main_copy", "exec"), old.__dict__)
        src = self.w.source()
        for form in ({}, {"how": "count", "count": "4"}):
            t_old = old.read_tiles(src)
            data = {"confirm": "yes", "sig": t_old["sig"], "samples": ["g0"], "nm_g0": "1.4724", "how": "all", "model": MODEL}
            data.update(form)
            with mock.patch.object(old, "check_model", return_value=None), mock.patch.object(old, "check_active"), \
                    mock.patch.object(P, "check_active"), mock.patch.object(P, "check_model", return_value=None):
                a = old.session_plan(self.w.cfg, self.user, src, dict(data))["names"]
                b = P.session_plan(self.w.cfg, self.user, src, dict(data))["names"]
            self.assertEqual(a, b)
            self.assertTrue(a)


class Page(Base):
    def test_button_counts_and_link(self):
        st, _, body = self.w.get("owner", f"/gallery/{CID}/")
        html = body.decode()
        self.assertEqual(st, 200)
        self.assertIn("Analyse kept tiles (3)", html)
        self.assertIn("also include maybe (2)", html)
        self.assertRegex(html, r'action="/predict/new/session/[0-9a-f]+/"')
        self.assertIn('name="which" value="kept"', html)
        self.assertIn(f'name="from" value="{CID}"', html)

    def test_admin_sees_it_non_owner_gets_404(self):
        self.assertIn("Analyse kept tiles (3)", self.w.get("admin", f"/gallery/{CID}/")[2].decode())
        self.assertEqual(self.w.get("other", f"/gallery/{CID}/")[0], 404)

    def test_hidden_with_zero_kept(self):
        st, _, body = self.w.get("owner", f"/gallery/{CID_Z}/")
        self.assertEqual(st, 200)
        self.assertNotIn("Analyse kept tiles", body.decode())

    def test_form_preselects_and_says_so(self):
        for q, n in (("?which=kept&from=" + CID, 3), ("?which=kept&from=" + CID + "&kept_maybe=1", 5)):
            st, _, body = self.w.get("owner", self.session_path(query=q))
            html = body.decode()
            self.assertEqual(st, 200)
            self.assertIn(f"{n} kept tiles from the gallery", html)
            self.assertIn(f'data-n="{n}"', html)
            self.assertIn('name="which" value="kept"', html)
            self.assertNotIn('name="how" value="count"', html)

    def test_plain_form_unchanged(self):
        html = self.w.get("owner", self.session_path())[2].decode()
        self.assertNotIn("kept tiles from the gallery", html)
        self.assertIn('name="how" value="count"', html)
        self.assertIn('data-n="10"', html)

    def test_non_owner_same_404(self):
        key = self.w.source()["key"]
        miss = self.w.get("other", "/predict/new/session/0123456789/?which=kept&from=" + CID)
        got = self.w.get("other", f"/predict/new/session/{key}/?which=kept&from={CID}")
        self.assertEqual((got[0], got[2]), (miss[0], miss[2]))
        self.assertEqual(got[0], 404)

    def test_zero_kept_refused_clearly(self):
        q = "?which=kept&from=" + CID_Z
        st, _, body = self.w.get("owner", self.session_path(CID_Z, q))
        self.assertEqual(st, 409)
        self.assertIn("No tiles are marked keep", body.decode())
        self.assertEqual(self.w.get("owner", self.session_path(CID_Z, q + "&kept_maybe=1"))[0], 409)

    def test_from_other_conversion_refused(self):
        st, _, body = self.w.get("owner", self.session_path(CID, "?which=kept&from=" + CID_Z))
        self.assertEqual(st, 400)
        self.assertIn("different conversion", body.decode())

    def test_post_over_http_uses_server_side_list(self):
        seen = {}

        def fake_create(d_from, **kw):
            with open(kw["manifest"], newline="", encoding="utf-8") as f:
                seen["names"] = sorted(r["output_name"] for r in csv.DictReader(f))
            d = Path(self.w.predict) / "ps_20261105_bbbbbb"
            d.mkdir(parents=True, exist_ok=True)
            return d

        path = self.session_path(query="?which=kept&from=" + CID)
        html = self.w.get("owner", path)[2].decode()
        body = urllib.parse.urlencode([("sig", tile_sig(html)), ("which", "kept"), ("from", CID), ("confirm", "yes"),
                                       ("samples", "g0"), ("nm_g0", "1.4724"), ("model", MODEL), ("how", "all"),
                                       ("names", NAMES[9])]).encode()       # a forged extra tile
        with mock.patch.object(P.PS, "create", side_effect=fake_create), mock.patch.object(P.PL, "start_detached"), \
                mock.patch.object(P, "check_model", return_value=None), mock.patch.object(P, "check_active"):
            st, _, _ = self.w.req("owner", "POST", path, body, extra=FORM)
            self.assertEqual(st, 303)
            for _ in range(100):
                if seen:
                    break
                time.sleep(0.1)
        self.assertEqual(seen.get("names"), sorted(KEEP))

    def test_post_zero_kept_refused(self):
        path = self.session_path(CID_Z)
        sig = tile_sig(self.w.get("owner", path)[2].decode())
        body = urllib.parse.urlencode([("sig", sig), ("which", "kept"), ("from", CID_Z), ("confirm", "yes"),
                                       ("samples", "g0"), ("nm_g0", "1.4724"), ("model", MODEL)]).encode()
        with mock.patch.object(P, "check_model", return_value=None), mock.patch.object(P, "check_active"):
            st, _, html = self.w.req("owner", "POST", path, body, extra=FORM)
        self.assertEqual(st, 409)
        self.assertIn("No tiles are marked keep", html.decode())


def launch(p):
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:                                                                 # noqa: BLE001
            continue
    return None


class Browser(Base):
    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        super().setUpClass()
        cls.shots = Path(os.environ.get("CJ_SHOT_DIR") or tempfile.mkdtemp(prefix="cj_ak_shots_"))
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

    def test_button_to_preselected_form(self):
        origin = self.w.portal.origin
        for scheme in ("light", "dark"):
            for width in (360, 390, 1280):
                with self.subTest(width=width, scheme=scheme):
                    ctx = self.browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
                    page = ctx.new_page()
                    errors = []
                    page.on("pageerror", lambda e: errors.append(str(e)))
                    page.goto(f"{origin}/login?k={self.w.keys['owner']}")
                    page.goto(f"{origin}/gallery/{CID}/")
                    tag = f"{scheme}_{width}"
                    btn = page.get_by_role("button", name="Analyse kept tiles (3)")
                    self.assertTrue(btn.is_visible())
                    self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), width)
                    page.screenshot(path=str(self.shots / f"ak_gallery_{tag}.png"))
                    page.get_by_label("also include maybe").check()
                    btn.click()
                    page.wait_for_url(re.compile(r"/predict/new/session/"))
                    self.assertIn("5 kept tiles from the gallery", page.content())
                    self.assertEqual(page.locator('input[name="samples"]:checked').count(), 1)
                    self.assertIn("5 tiles will be analysed", page.locator("#cj-predict-sel").inner_text())
                    self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), width)
                    page.screenshot(path=str(self.shots / f"ak_form_{tag}.png"), full_page=True)
                    self.assertEqual(errors, [])
                    ctx.close()


if __name__ == "__main__":
    unittest.main()
