"""
Pages through the shell in a real browser (Playwright, Edge channel as in registry/tests): home, History and Review
at 390 and 1280 px, light and dark. Sign-in through the key link in the browser itself. Per page: no horizontal
scroll, no page or console errors, only same-origin requests (anything else is aborted and fails the test).
Screenshots go to $CJ_SHOT_DIR when set (else a temp folder, removed afterwards).

Temp registry (history.html from registry/samples, a monitor-shaped live.json) and the real review_server.py
handler on a temp fixture package; never the real registry or review servers.

  python -m unittest discover -s portal/tests -v
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402

sys.path.insert(0, str(T.REPO / "registry"))
import history_page as hp  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None


def launch(p):
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:
            continue
    return None


PAGES = (("home", "/", "#cj-bar .cj-nav a"), ("history", "/history/history.html", "#cj-bar .cj-nav a"),
         ("review", "/review/", "#cj-bar .cj-nav a"))
_MODS = __import__("json").loads((__import__("pathlib").Path(__file__).resolve().parents[1] / "modules.json")
                                 .read_text(encoding="utf-8"))["modules"]
NAV_LINKS = [m["name"] for m in _MODS if m.get("nav") and m["status"] == "on" and "admin" in m["roles"]]  # admin, on
NAV_ALL = ["Home", "Data", "Jobs", "Review", "Predict", "Results", "My reviews", "Train", "Admin", "Sandbox"]   # + greyed "coming"


class Pages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        cls.env = T.Env()
        hp.build([T.REPO / "registry" / "samples"], cls.env.registry / "history.html")
        cls.env.write_live(T.live_doc())
        cls.rv = T.ReviewUpstream()
        cls.p = T.Portal(cls.env.config(review_upstream=cls.rv.url, review_key=cls.rv.key))
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
        cls.p.stop()
        cls.rv.stop()
        cls.env.cleanup()
        if cls.own_shots:
            shutil.rmtree(cls.shots, ignore_errors=True)

    def session(self, width, scheme):
        ctx = self.browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
        origin = self.p.origin
        rec = {"requests": [], "foreign": [], "errors": [], "console": []}
        ctx.route("**/*", lambda r: (r.continue_() if r.request.url.startswith(origin + "/")
                                     else (rec["foreign"].append(r.request.url), r.abort())))
        page = ctx.new_page()
        page.on("request", lambda r: rec["requests"].append(r.url))
        page.on("pageerror", lambda e: rec["errors"].append(str(e)))
        page.on("console", lambda m: m.type == "error" and rec["console"].append(m.text))
        page.goto(f"{origin}/login?k={self.env.key}")
        self.assertEqual(page.url, origin + "/")                       # the key link signs in and lands home
        names = [c["name"] for c in ctx.cookies()]
        self.assertIn("__Host-cj_session", names)
        return ctx, page, rec

    def test_pages(self):
        for scheme in ("light", "dark"):
            for width in (390, 1280):
                ctx, page, rec = self.session(width, scheme)
                try:
                    for name, path, sel in PAGES:
                        with self.subTest(page=name, width=width, scheme=scheme):
                            page.goto(self.p.origin + path)
                            page.wait_for_load_state("networkidle")
                            if sel:
                                page.wait_for_selector(sel)
                            sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
                            self.assertLessEqual(sw, iw, "horizontal scroll")
                            page.screenshot(path=str(self.shots / f"{name}_{width}_{scheme}.png"), full_page=False)
                            if name == "home":
                                nav = page.eval_on_selector_all("#cj-bar .cj-nav a", "a => a.map(x => x.textContent)")
                                self.assertEqual(nav, NAV_LINKS)
                                nav = page.eval_on_selector_all("#cj-bar .cj-nav > *", "a => a.map(x => x.textContent)")
                                self.assertEqual(nav, NAV_ALL)
                                self.assertEqual(page.get_attribute('#cj-bar a[data-id="review"] .cj-dot', "class"),
                                                 "cj-dot up")
                                self.assertEqual(page.get_attribute('#cj-bar a[data-id="sandbox"] .cj-dot', "class"),
                                                 "cj-dot down")
                                self.assertEqual(page.text_content("#cj-bar .cj-user .cj-nm > span"), T.USER)
                                self.assertEqual(page.locator(".cj-mod.off").count(),
                                                 sum(1 for m in _MODS if m.get("nav") and m["status"] != "on"
                                                     and "admin" in m["roles"]))
                                page.wait_for_selector("#cj-train .cj-lv-snap")     # CJLive drew the snapshot
                            if name == "history":
                                self.assertTrue(page.locator("#srv").is_hidden())   # the shell nav replaces it
                                self.assertTrue(page.locator("#cj-bar").is_visible())
                                page.wait_for_selector("#tree .rec")                # the page itself still works
                            if name == "review":                                # the tool, framed under the bar
                                self.assertTrue(page.locator("#cj-bar").is_visible())
                                self.assertEqual(page.get_attribute("#cj-bar .cj-nav a[aria-current=page]", "data-id"),
                                                 "review")
                                self.assertTrue(any(u.endswith("/review/app/api/edits") for u in rec["requests"]))
                                frame = page.frame_locator("#cj-review-frame")
                                self.assertEqual(frame.locator("#cj-bar").count(), 0)   # the tool keeps its own look
                    self.assertEqual(rec["foreign"], [])
                    self.assertTrue(rec["requests"])
                    self.assertTrue(all(u.startswith(self.p.origin + "/") for u in rec["requests"]),
                                    [u for u in rec["requests"] if not u.startswith(self.p.origin + "/")])
                    self.assertEqual(rec["errors"], [])
                    self.assertEqual(rec["console"], [])
                finally:
                    ctx.close()

    def test_signed_out_browser_sees_401_page(self):
        ctx = self.browser.new_context(viewport={"width": 390, "height": 900})
        try:
            page = ctx.new_page()
            resp = page.goto(self.p.origin + "/")
            self.assertEqual(resp.status, 401)
            self.assertIn("key link", page.text_content("body"))
            sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
            self.assertLessEqual(sw, iw)
        finally:
            ctx.close()


if __name__ == "__main__":
    unittest.main()
