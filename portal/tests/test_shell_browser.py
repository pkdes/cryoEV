"""
The shell in a real browser (Playwright, Edge channel as test_pages.py): the bar's geometry at 390 / 1280 / 1600 px on
Home, History and the Review shell (left edge 0, full width, fixed height, no content jump), the review frame (fills
the rest, carries the query, mirrors its URL back, leaves the shell when it leaves the tool), nav and home cards per
role (admin / advanced / basic from a temp roster), design A's light and dark tokens (History keeps its own), no
horizontal scroll, no page or console errors, same-origin requests only (anything else is aborted and fails).
Screenshots go to $CJ_SHOT_DIR when set (else a temp folder, removed afterwards).

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
from portal import auth as A  # noqa: E402

sys.path.insert(0, str(T.REPO / "registry"))
import history_page as hp  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

BAR_H = {390: 88, 1280: 52, 1600: 52}            # cj.css --cj-bar-h; the review shell's phone bar is compact (44)
PAGES = (("home", "/"), ("history", "/history/history.html"), ("review", "/review/"))
CONTENT = {"home": "main.cj-main", "history": "body > header"}      # the first thing under the bar
_MODS = __import__("json").loads((__import__("pathlib").Path(__file__).resolve().parents[1] / "modules.json")
                                 .read_text(encoding="utf-8"))["modules"]


def _on(role):
    return [m["name"] for m in _MODS if m.get("nav") and m["status"] == "on" and role in m["roles"]]


NAV = {"admin": (["Home", "Data", "Jobs", "Review", "Predict", "Results", "My reviews", "Train", "Admin", "Sandbox"], _on("admin")),
       "advanced": (["Home", "Data", "Jobs", "Predict", "Results", "My reviews", "Train"], _on("advanced")),
       "basic": (["Home", "Predict", "Results", "My reviews"], _on("basic"))}
CARDS = {"admin": ["data", "jobs", "review", "predict", "results", "reviews", "train", "admin", "sandbox"],
         "advanced": ["data", "jobs", "predict", "results", "reviews", "train"],
         "basic": ["predict", "results", "reviews"]}
TOKENS = {"light": {"bg": "rgb(243, 245, 247)", "card": "rgb(255, 255, 255)", "hist_bg": "rgb(250, 250, 249)"},
          "dark": {"bg": "rgb(15, 21, 27)", "card": "rgb(23, 32, 41)", "hist_bg": "rgb(20, 19, 18)"}}


def launch(p):
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:
            continue
    return None


class Shell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        cls.env = T.Env()
        hp.build([T.REPO / "registry" / "samples"], cls.env.registry / "history.html")
        cls.env.write_live(T.live_doc())
        cls.keys = {"admin": cls.env.key}
        for role in ("advanced", "basic"):
            cls.keys[role] = A.add_key(cls.env.roster, f"u-{role}", role)
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

    def session(self, width, scheme="light", role="admin", js=True, height=900):
        ctx = self.browser.new_context(viewport={"width": width, "height": height}, color_scheme=scheme,
                                       java_script_enabled=js)
        origin = self.p.origin
        rec = {"requests": [], "foreign": [], "errors": [], "console": []}
        ctx.route("**/*", lambda r: (r.continue_() if r.request.url.startswith(origin + "/")
                                     else (rec["foreign"].append(r.request.url), r.abort())))
        page = ctx.new_page()
        page.on("request", lambda r: rec["requests"].append(r.url))
        page.on("pageerror", lambda e: rec["errors"].append(str(e)))
        page.on("console", lambda m: m.type == "error" and rec["console"].append(m.text))
        page.goto(f"{origin}/login?k={self.keys[role]}")
        self.assertEqual(page.url, origin + "/")
        return ctx, page, rec

    def clean(self, rec):
        self.assertEqual(rec["foreign"], [])
        self.assertTrue(all(u.startswith(self.p.origin + "/") for u in rec["requests"]))
        self.assertEqual(rec["errors"], [])
        self.assertEqual(rec["console"], [])

    def bar_box(self, page):
        return page.evaluate("""() => { const b = document.getElementById('cj-bar').getBoundingClientRect();
            return {left: b.left, top: b.top, width: b.width, height: b.height, bottom: b.bottom,
                    vw: document.documentElement.clientWidth, sw: document.documentElement.scrollWidth,
                    iw: window.innerWidth, ih: window.innerHeight,
                    tag: document.getElementById('cj-bar').tagName, role: document.getElementById('cj-bar').getAttribute('role')}; }""")

    def test_bar_geometry(self):
        for scheme in ("light", "dark"):
            for width in (390, 1280, 1600):
                ctx, page, rec = self.session(width, scheme)
                try:
                    for name, path in PAGES:
                        with self.subTest(page=name, width=width, scheme=scheme):
                            page.goto(self.p.origin + path)
                            page.wait_for_load_state("networkidle")
                            page.wait_for_selector("#cj-bar .cj-nav a")
                            b = self.bar_box(page)
                            self.assertEqual((b["tag"], b["role"]), ("DIV", "banner"))
                            self.assertEqual((b["left"], b["top"]), (0, 0))
                            self.assertEqual(b["width"], b["vw"])                       # full width, also > 1280
                            want_h = 44 if (name == "review" and width == 390) else BAR_H[width]
                            self.assertEqual(b["height"], want_h)
                            self.assertLessEqual(b["sw"], b["iw"], "horizontal scroll")
                            page.screenshot(path=str(self.shots / f"shell_{name}_{width}_{scheme}.png"))
                    self.clean(rec)
                finally:
                    ctx.close()

    def test_no_content_jump(self):
        """The first content's top is the same before header.js runs (JS off: cj.css holds the space) and after the
        bar is built (JS on), on Home and on History."""
        for width in (390, 1600):
            tops = {}
            for js in (False, True):
                ctx, page, rec = self.session(width, js=js)
                try:
                    for name, path in PAGES[:2]:
                        page.goto(self.p.origin + path)
                        page.wait_for_load_state("networkidle")
                        if js:
                            page.wait_for_selector("#cj-bar .cj-nav a")
                        else:
                            self.assertEqual(page.locator("#cj-bar .cj-nav").count(), 0)   # control: no bar built
                        tops[(name, js)] = page.evaluate(
                            f"document.querySelector({CONTENT[name]!r}).getBoundingClientRect().top")
                finally:
                    ctx.close()
            for name, _ in PAGES[:2]:
                with self.subTest(page=name, width=width):
                    self.assertEqual(tops[(name, False)], tops[(name, True)])
                    self.assertGreaterEqual(tops[(name, True)], BAR_H[width])

    def test_review_frame(self):
        for width in (390, 1280):
            ctx, page, rec = self.session(width)
            try:
                with self.subTest(width=width):
                    page.goto(self.p.origin + "/review/?mode=cards&k=stale-key-55")
                    page.wait_for_load_state("networkidle")
                    page.wait_for_selector("#cj-bar .cj-nav a")
                    self.assertEqual(page.get_attribute("#cj-review-frame", "src"), "/review/app/?mode=cards")
                    frame = page.frame(url=lambda u: "/review/app/" in u)
                    self.assertIsNotNone(frame)
                    frame.wait_for_selector("body")
                    self.assertTrue(any(u.endswith("/review/app/api/edits") for u in rec["requests"]))
                    tool = [u for u in rec["requests"] if "/review/app/" in u]
                    self.assertTrue(tool)
                    self.assertFalse(any("stale-key-55" in u for u in tool))                 # k never reaches it
                    page.wait_for_function("!location.search.includes('k=')")               # mirrored: k gone
                    self.assertEqual(page.url, self.p.origin + "/review/?mode=cards")
                    g = page.evaluate("""() => { const f = document.getElementById('cj-review-frame').getBoundingClientRect(),
                        b = document.getElementById('cj-bar').getBoundingClientRect();
                        return {ft: f.top, fb: f.bottom, fw: f.width, bb: b.bottom, ih: innerHeight, iw: innerWidth,
                                sh: document.documentElement.scrollHeight, sw: document.documentElement.scrollWidth}; }""")
                    self.assertEqual(g["ft"], g["bb"])                                       # right under the bar
                    self.assertEqual((g["fb"], g["fw"]), (g["ih"], g["iw"]))                 # fills the rest
                    self.assertLessEqual((g["sh"], g["sw"]), (g["ih"], g["iw"]))             # no page scroll
                    page.screenshot(path=str(self.shots / f"shell_review_frame_{width}.png"))
                    # the tool changes its URL -> the window's address follows (reload / bookmarks / Back)
                    frame.evaluate("history.pushState(null, '', '?img=nope.png&mode=editor#x')")
                    page.wait_for_function("location.search.includes('img=nope.png')")
                    self.assertEqual(page.url, self.p.origin + "/review/?img=nope.png&mode=editor#x")
                    frame.evaluate("history.replaceState(null, '', '?mode=classes')")
                    page.wait_for_function("location.search === '?mode=classes'")
                    page.reload()
                    page.wait_for_load_state("networkidle")
                    self.assertEqual(page.get_attribute("#cj-review-frame", "src"), "/review/app/?mode=classes")
                    # a link out of the tool opens in the window, not the frame (shell pages are DENY-framed)
                    frame = page.frame(url=lambda u: "/review/app/" in u)
                    frame.evaluate("""() => { const a = document.createElement('a'); a.href = '/'; a.id = 'cj-test-out';
                                             a.textContent = 'Home'; document.body.appendChild(a); }""")
                    frame.locator("#cj-test-out").dispatch_event("click")
                    page.wait_for_url(self.p.origin + "/")
                    self.assertEqual(page.locator("#cj-bar").count(), 1)
                    self.clean(rec)
            finally:
                ctx.close()

    def test_review_frame_never_leaves_the_origin(self):
        """"/.//evil.example/x" is a same-origin URL whose path starts with "//": navigating to that PATH would go to
        evil.example. review.js only ever navigates to the absolute same-origin URL."""
        ctx, page, rec = self.session(1280)
        try:
            for how in ("click", "frame"):
                with self.subTest(how=how):
                    page.goto(self.p.origin + "/review/?mode=cards")
                    page.wait_for_load_state("networkidle")
                    frame = page.frame(url=lambda u: "/review/app/" in u)
                    if how == "click":                       # a link out of the tool -> this window
                        frame.evaluate("""() => { const a = document.createElement('a'); a.href = '/.//evil.example/x';
                                                 a.id = 'cj-test-evil'; a.textContent = 'x'; document.body.appendChild(a); }""")
                        frame.locator("#cj-test-evil").dispatch_event("click")
                        want = self.p.origin + "//evil.example/x"
                    else:                                    # the frame lands outside the tool -> this window follows
                        frame.evaluate("location.href = '/review/.//evil.example/x'")
                        want = self.p.origin + "/review//evil.example/x"
                    page.wait_for_url(want)
                    page.wait_for_load_state("networkidle")
                    self.assertTrue(page.url.startswith(self.p.origin + "/"), page.url)
                    self.assertEqual(rec["foreign"], [])                        # nothing left for evil.example
            self.assertFalse(any("evil.example" in u and not u.startswith(self.p.origin + "/") for u in rec["requests"]))
        finally:
            ctx.close()

    def test_frame_on_the_shell_base_does_not_loop(self):
        """If the frame lands on /review/ itself, the window stays where it is (no reload of the shell, no loop)."""
        ctx, page, rec = self.session(1280)
        try:
            page.goto(self.p.origin + "/review/?mode=cards")
            page.wait_for_load_state("networkidle")
            page.evaluate("window.cjMarker = 1")                       # gone if the window reloads
            frame = page.frame(url=lambda u: "/review/app/" in u)
            frame.evaluate("location.href = '/review/?mode=classes'")
            page.wait_for_function("""() => { try { return document.getElementById('cj-review-frame')
                                     .contentWindow.location.pathname === '/review/'; } catch (e) { return false; } }""")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(300)
            self.assertEqual(page.evaluate("window.cjMarker"), 1)
            self.assertEqual(page.url, self.p.origin + "/review/?mode=cards")
            self.clean(rec)
        finally:
            ctx.close()

    def test_copied_tool_link_opens_in_the_shell(self):
        """A /review/app/ link opened as the whole page (e.g. copied from the tool) lands in the shell, k dropped."""
        ctx, page, rec = self.session(1280)
        try:
            resp = page.goto(self.p.origin + "/review/app/?mode=classes&k=stale-77")
            page.wait_for_load_state("networkidle")
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.request.redirected_from.url, self.p.origin + "/review/app/?mode=classes&k=stale-77")
            self.assertEqual(page.url, self.p.origin + "/review/?mode=classes")
            self.assertEqual(page.get_attribute("#cj-review-frame", "src"), "/review/app/?mode=classes")
            frame = page.frame(url=lambda u: "/review/app/" in u)
            frame.wait_for_selector("body")
            self.assertEqual(frame.locator("#cj-bar").count(), 0)          # the frame's own load was not redirected
            self.assertTrue(any(u.endswith("/review/app/api/edits") for u in rec["requests"]))
            self.clean(rec)
        finally:
            ctx.close()

    def test_nav_and_cards_by_role(self):
        for role in ("admin", "advanced", "basic"):
            ctx, page, rec = self.session(1280, role=role)
            try:
                with self.subTest(role=role):
                    page.goto(self.p.origin + "/")
                    page.wait_for_load_state("networkidle")
                    page.wait_for_selector("#cj-bar .cj-nav a")
                    items = page.eval_on_selector_all("#cj-bar .cj-nav > *", "a => a.map(x => x.textContent)")
                    links = page.eval_on_selector_all("#cj-bar .cj-nav > a", "a => a.map(x => x.textContent)")
                    self.assertEqual((items, links), NAV[role])
                    self.assertEqual(page.locator("#cj-bar .cj-nav .cj-off[aria-disabled=true]").count(),
                                     len(NAV[role][0]) - len(NAV[role][1]))                  # coming: greyed, no link
                    self.assertEqual(page.eval_on_selector_all("#cj-modules [data-id]", "a => a.map(x => x.dataset.id)"),
                                     CARDS[role])
                    self.assertEqual(page.text_content("#cj-bar .cj-nm small"), role)
                    reads_live = any(u.endswith("/history/live/live.json") for u in rec["requests"])
                    self.assertEqual(reads_live, role == "admin")                            # only History's roles
                    self.assertEqual(page.locator("#cj-classic").count(), int(role == "admin"))
                    page.screenshot(path=str(self.shots / f"shell_home_{role}_1280.png"))
                    self.clean(rec)
                    if role != "admin":                                                     # the Gate, in the browser
                        for path in ("/review/", "/review/app/", "/history/history.html"):
                            resp = page.goto(self.p.origin + path)
                            self.assertEqual(resp.status, 403, path)
            finally:
                ctx.close()

    def test_theme_tokens(self):
        for scheme in ("light", "dark"):
            ctx, page, rec = self.session(1280, scheme)
            try:
                with self.subTest(scheme=scheme):
                    want = TOKENS[scheme]
                    page.goto(self.p.origin + "/")
                    page.wait_for_selector("#cj-bar .cj-nav a")
                    css = page.evaluate("""() => [getComputedStyle(document.body).backgroundColor,
                        getComputedStyle(document.getElementById('cj-bar')).backgroundColor,
                        getComputedStyle(document.body).getPropertyValue('--hot').trim()]""")
                    self.assertEqual(css[:2], [want["bg"], want["card"]])
                    self.assertEqual(css[2], "#c0135a" if scheme == "light" else "#ff5c9b")
                    # the gradient label: Review card only, text painted by the gradient
                    self.assertEqual(page.locator(".cj-evh").count(), 1)
                    self.assertEqual(page.locator(".cj-mod-review .cj-evh").text_content(), "eVHarmony")
                    clip = page.evaluate("getComputedStyle(document.querySelector('.cj-evh')).backgroundClip")
                    self.assertEqual(clip, "text")
                    # History keeps its own tokens; only the bar wears the shell's
                    page.goto(self.p.origin + "/history/history.html")
                    page.wait_for_selector("#cj-bar .cj-nav a")
                    css = page.evaluate("""() => [getComputedStyle(document.body).backgroundColor,
                        getComputedStyle(document.getElementById('cj-bar')).backgroundColor]""")
                    self.assertEqual(css, [want["hist_bg"], want["card"]])
                    self.clean(rec)
            finally:
                ctx.close()


if __name__ == "__main__":
    unittest.main()
