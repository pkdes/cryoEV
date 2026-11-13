"""
Reviewer identity through the portal (BUILD_CONTRACTS.md section 8): a signed-in user who opens a prediction review
package at /review/<pkg_id>/ is the reviewer. The pred proxy sends the package key (X-Review-Key) and the user's id
(X-CJ-User; any client-sent X-CJ-* is dropped first, portal/proxy.py); annotation/multi_review_server.py trusts the id
only with that key, puts it in the page (no "Who is reviewing?" prompt) and stamps every save with it. The page is
the repo's current tool, also for packages whose own index.html predates this (OldPackageCopy: card mode must not
skip the user's own cards), and its drafts sit under a per-user localStorage key (a shared browser).

Controls with known answers: the multi server reached directly with ?k= (no portal) still shows the prompt and saves
the page's own name; a request to it without the key is refused, header or not.

Temp roster, temp review root (packages built from the CryoEV golden prediction set); never the live review folder.

  python -m unittest discover -s portal/tests -v
"""
import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from test_shell_pred import PKG_A, PKG_B, MultiDummy, PredEnv, build_root, hdrs  # noqa: E402
from annotation import multi_review_server as mrs  # noqa: E402
from annotation import prediction_package as pp  # noqa: E402
sys.path.insert(0, str(T.REPO / "annotation" / "tests"))
from test_review_identity import old_tool_page  # noqa: E402

SPOOF = {"X-CJ-User": "mallory", "X-CJ-Reviewer": "mallory", "X-CJ-Identity": "mallory"}
# duplicates, case variants and a client X-Review-Key: none of them may reach the multi server
SPOOF_RAW = [("X-CJ-User", "mallory"), ("x-cj-user", "mallory-2"), ("X-CJ-USER", "mallory-3"),
             ("x-Cj-Reviewer", "mallory"), ("X-CJ-REVIEWER", "mallory"), ("X-Review-Key", "mallory-key"),
             ("x-review-key", "mallory-key-2")]
IDENT = b"<script>window.REVIEW_IDENTITY = "
T0 = 1_762_400_000_000
PHONE = {"viewport": {"width": 390, "height": 800}, "is_mobile": True, "has_touch": True}


def send(port, method, path, headers, body=None) -> int:
    """A request with headers as a list of pairs (one name may repeat, in any case) -> status."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        c.putrequest(method, path, skip_accept_encoding=True)
        for k, v in headers:
            c.putheader(k, v)
        if body is not None:
            c.putheader("Content-Length", str(len(body)))
        c.endheaders(body)
        r = c.getresponse()
        r.read()
        return r.status
    finally:
        c.close()


def launch(pw):
    """Headless Edge, else Chrome, else Playwright's Chromium; None when none starts."""
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return pw.chromium.launch(headless=True, **kw)
        except Exception:
            continue
    return None


def sync_playwright_or_skip(tc):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:  # pragma: no cover
        tc.skipTest("playwright not installed")
    return sync_playwright


def tool_frame(page, origin, pkg_id):
    """The review tool inside the portal shell's frame, once it is connected to the server."""
    page.wait_for_selector("#cj-review-frame")
    page.frame_locator("#cj-review-frame").locator("#triWho").wait_for(state="attached")
    fr = page.frame(url=lambda u: u.startswith(f"{origin}/review/{pkg_id}/app/"))
    fr.wait_for_function("window.__review && __review.state().server && __review.state().tri.ready")
    return fr


class HeaderToUpstream(unittest.TestCase):
    """What reaches the multi server: the signed-in user's id once, never the client's X-CJ-* headers."""

    @classmethod
    def setUpClass(cls):
        cls.up = MultiDummy()
        cls.e = PredEnv(cls.up.url)

    @classmethod
    def tearDownClass(cls):
        cls.e.stop()
        cls.up.stop()

    def test_client_identity_headers_stripped(self):
        for method, path, body in (("GET", f"/review/{PKG_A}/app/", None),
                                   ("PUT", f"/review/{PKG_A}/app/api/edits?f=0001.png", b'{"t": 1}'),
                                   ("POST", f"/review/{PKG_A}/app/api/claim?c=x&who=mallory", b"")):
            for spoof in (list(SPOOF.items()), SPOOF_RAW):
                h = [*T.cookie(self.e.tokens["basic"]).items(), ("Content-Type", "application/json"), *spoof]
                if method != "GET":
                    h.append(("Origin", self.e.p.origin))
                st = send(self.e.p.port, method, path, h, body)
                self.assertEqual(st, 200, (method, path))
                rec = self.up.last()
                names = [k.lower() for k, _ in rec["headers"]]
                self.assertEqual(names.count("x-cj-user"), 1)
                self.assertEqual(hdrs(rec)["x-cj-user"], "u-basic")
                self.assertEqual([n for n in names if n.startswith("x-cj-") and n != "x-cj-user"], [])
                self.assertEqual(names.count("x-review-key"), 1)
                self.assertEqual(hdrs(rec)["x-review-key"], self.e.key[PKG_A])     # the package's, not the client's
                self.assertNotIn("mallory", json.dumps(rec["headers"]))


class ThroughThePortal(unittest.TestCase):
    """annotation/multi_review_server.py behind the portal, on a temp review root."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cj_pred_identity_"))
        cls.root = build_root(cls.tmp)
        cls.srv = mrs.make_server(cls.root, "127.0.0.1", 0, people=[])
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]
        cls.e = PredEnv(f"http://127.0.0.1:{cls.port}", cls.root)
        cls.pkg = cls.root / PKG_A
        cls.data = pp.read_datajs(cls.pkg)

    @classmethod
    def tearDownClass(cls):
        cls.e.stop()
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        for pid in (PKG_A, PKG_B):
            for f in (self.root / pid / "edits").glob("*.json"):
                f.unlink()

    def saved(self, file, pid=PKG_A):
        return json.loads((self.root / pid / "edits" / f"{file}.json").read_text(encoding="utf-8"))["edit"]

    def test_page_names_the_signed_in_user(self):
        index = mrs.TOOL_PAGE.read_bytes()                       # the repo's current tool (multi_review_server)
        for who, uid in (("basic", "u-basic"), ("admin", T.USER)):
            st, _, body = self.e.req(who, "GET", f"/review/{PKG_A}/app/", SPOOF)
            self.assertEqual(st, 200, who)
            self.assertEqual(body.replace(mrs.identity_script(uid), b"", 1), index)    # that id, nothing else
            self.assertNotIn(b"mallory", body)
        st, _, body = self.e.req("adv", "GET", f"/review/{PKG_B}/app/")
        self.assertIn(mrs.identity_script("u-adv"), body)

    def test_saves_record_the_signed_in_user(self):
        im = self.data["images"][0]
        file, g1, g2 = im["file"], f"g{im['gt'][0]['id']}", f"g{im['gt'][1]['id']}"
        edit = {"t": T0 + 2, "modified": True, "reviewed": True, "reviewedT": T0 + 2, "dec": {
            g1: {"v": "keep:MV", "t": T0 + 1, "who": "mallory", "pending": True},
            g2: {"v": "del", "t": T0 + 2, "who": "pujan", "server_t": T0 + 10 ** 9}}}   # forged, not pending
        st, _, body = self.e.req("basic", "PUT", f"/review/{PKG_A}/app/api/edits?f={file}",
                                 {**SPOOF, "Content-Type": "application/json"}, json.dumps(edit).encode())
        self.assertEqual(st, 200, body)
        dec = self.saved(file)["dec"]
        self.assertEqual((dec[g1]["who"], dec[g2]["who"]), ("u-basic", "u-basic"))
        rp = pp.reviewed_polygons(self.pkg, write=False)
        self.assertEqual(rp["images"][Path(file).stem]["reviewed_by"], ["u-basic"])
        self.assertTrue(rp["images"][Path(file).stem]["reviewed"])
        st, _, body = self.e.req("basic", "POST", f"/review/{PKG_A}/app/api/claim?c={file}%7C{g1}&who=mallory", SPOOF,
                                 b"")
        self.assertEqual((st, json.loads(body)["claims"].get(f"{file}|{g1}")), (200, "u-basic"))

    def test_direct_without_key_refused(self):
        for h in ({"X-CJ-User": "u-basic"}, {"X-CJ-User": "u-basic", "X-Review-Key": "wrong"}):
            for method, path, body in (("GET", f"/{PKG_A}/", None), ("PUT", f"/{PKG_A}/api/edits?f=0001.png",
                                                                      b'{"t": 1}')):
                st, _, _ = T.request(self.port, method, path, h, body)
                self.assertEqual(st, 403, (h, method))
        self.assertEqual(list((self.pkg / "edits").glob("*.json")), [])

    def test_control_direct_link_keeps_the_pages_name(self):
        """Known answer: the same server reached with ?k= (a phone link, no portal): the file as on disk, and the
        page's who is saved."""
        key = self.e.key[PKG_A]
        st, h, body = T.request(self.port, "GET", f"/{PKG_A}/?k={key}", {"X-CJ-User": "mallory"})
        self.assertEqual((st, body), (200, (self.pkg / "index.html").read_bytes()))
        cookie = T.hget(h, "set-cookie").split(";")[0]
        im = self.data["images"][1]
        file, g1 = im["file"], f"g{im['gt'][0]['id']}"
        edit = {"t": T0 + 1, "dec": {g1: {"v": "del", "t": T0 + 1, "who": "pujan", "pending": True}}}
        st, _, body = T.request(self.port, "PUT", f"/{PKG_A}/api/edits?f={file}",
                                {"Cookie": cookie, "X-CJ-User": "mallory"}, json.dumps(edit).encode())
        self.assertEqual(st, 200, body)
        self.assertEqual(self.saved(file)["dec"][g1]["who"], "pujan")

    def test_in_a_browser_at_390(self):
        """The owner opens the package through the portal at 390 px: no name prompt, the badge shows their id, a card
        pick is saved under it, and drafts sit under a per-user key. Then the admin on the SAME browser: their own id
        and their own drafts key. Control: the same package opened directly with ?k= does show the prompt."""
        sync_playwright = sync_playwright_or_skip(self)
        origin = self.e.p.origin
        with sync_playwright() as pw:
            browser = launch(pw)
            if browser is None:
                self.skipTest("no Edge / Chrome for Playwright")
            try:
                # control first: the multi server directly, no portal -> the prompt (so the check below can see one)
                ctx = browser.new_context(**PHONE)
                page = ctx.new_page()
                page.goto(f"http://127.0.0.1:{self.port}/{PKG_A}/?k={self.e.key[PKG_A]}&mode=cards")
                page.wait_for_function("window.__review && __review.state().server")
                page.wait_for_selector("#whoDlg:not([hidden])", timeout=10000)
                self.assertEqual(page.evaluate("__review.state().who"), "")
                ctx.close()

                ctx = browser.new_context(**PHONE)
                errs = []
                page = ctx.new_page()
                page.on("pageerror", lambda e: errs.append(str(e)))
                page.goto(f"{origin}/login?k={self.e.keys['basic']}")
                page.goto(f"{origin}/review/{PKG_A}/?mode=cards")
                fr, frame = tool_frame(page, origin, PKG_A), page.frame_locator("#cj-review-frame")
                self.assertEqual(fr.evaluate("__review.state().who"), "u-basic")
                self.assertTrue(frame.locator("#whoDlg").is_hidden())
                self.assertIn("u-basic", frame.locator("#triWho").inner_text())
                c = fr.evaluate("(() => { const s = __review.state(); return s.tri.queue[s.tri.pos]; })()")
                file = self.data["images"][c["i"]]["file"]
                frame.locator("#tri button:text-is('MV')").tap()          # the candidate card: as MV
                fr.wait_for_function("__review.state().dirty === 0 && document.getElementById('sync').textContent"
                                     ".includes('saved')", timeout=15000)
                self.assertTrue(frame.locator("#whoDlg").is_hidden())        # the pick never asked for a name
                frame.locator("#triWho").tap()                               # and the badge can't change it
                self.assertTrue(frame.locator("#whoDlg").is_hidden())
                d = self.saved(file)["dec"][c["key"]]
                self.assertEqual((d["v"], d["who"]), ("add:MV", "u-basic"))
                drafts = f"cryoev-review:{PKG_A}"
                keys_js = "(() => Object.keys(localStorage).filter(k => k.startsWith(%s)).sort())()" % json.dumps(drafts)
                fr.wait_for_function("localStorage.getItem(%s) !== null" % json.dumps(drafts + ":u:u-basic"))
                self.assertEqual(fr.evaluate(keys_js), [drafts + ":u:u-basic", drafts + ":u:u-basic:dirty"])
                self.assertIsNone(fr.evaluate("localStorage.getItem('cryoev-review:who')"))   # no name stored
                mine = fr.evaluate("localStorage.getItem(%s)" % json.dumps(drafts + ":u:u-basic"))
                self.assertIn('"add:MV"', mine)
                shots = os.environ.get("CJ_SHOT_DIR")
                if shots:
                    page.screenshot(path=str(Path(shots) / "pred_identity_390.png"))

                # the next user on the same browser: their own id, their own drafts key; the first user's untouched
                page.goto(f"{origin}/login?k={self.e.keys['admin']}")
                page.goto(f"{origin}/review/{PKG_A}/?mode=cards")
                fr = tool_frame(page, origin, PKG_A)
                self.assertEqual(fr.evaluate("__review.state().who"), T.USER)
                fr.wait_for_function("localStorage.getItem(%s) !== null" % json.dumps(f"{drafts}:u:{T.USER}"))
                self.assertEqual(fr.evaluate(keys_js), sorted([drafts + ":u:u-basic", drafts + ":u:u-basic:dirty",
                                                               f"{drafts}:u:{T.USER}", f"{drafts}:u:{T.USER}:dirty"]))
                self.assertEqual(fr.evaluate("localStorage.getItem(%s)" % json.dumps(drafts + ":u:u-basic")), mine)
                self.assertEqual(errs, [])
            finally:
                browser.close()


class OldPackageCopy(unittest.TestCase):
    """Packages built before F2 hold an index.html without LOCKED_WHO (old_tool_page() == main's page before F2).
    Through the portal they get the repo's tool: no prompt, claims under the portal id, and card mode does not skip
    the user's own card. Control: the same old page, reached directly, does skip a card claimed by someone else."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cj_pred_identity_old_"))
        cls.root = build_root(cls.tmp)
        cls.old = old_tool_page()
        for pid in (PKG_A, PKG_B):
            (cls.root / pid / "index.html").write_bytes(cls.old)
        cls.srv = mrs.make_server(cls.root, "127.0.0.1", 0, people=[])
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]
        cls.e = PredEnv(f"http://127.0.0.1:{cls.port}", cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.e.stop()
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def claims(self, pid):
        st, _, body = T.request(self.port, "GET", f"/{pid}/api/claims", {"X-Review-Key": self.e.key[pid]})
        self.assertEqual(st, 200)
        return json.loads(body)["claims"]

    def test_served_page(self):
        self.assertNotIn(b"LOCKED_WHO", self.old)
        st, _, body = self.e.req("basic", "GET", f"/review/{PKG_A}/app/")
        self.assertEqual((st, body), (200, mrs.inject_identity(mrs.TOOL_PAGE.read_bytes(), "u-basic")))
        st, _, body = T.request(self.port, "GET", f"/{PKG_A}/?k={self.e.key[PKG_A]}")
        self.assertEqual((st, body), (200, self.old))                     # direct link: the package's own copy

    def test_cards_do_not_skip_own_card(self):
        sync_playwright = sync_playwright_or_skip(self)
        origin, card = self.e.p.origin, "0001.png|p0"                    # the golden set's one candidate card
        state = "(() => { const s = __review.state().tri; return [s.pos, s.queue.length]; })()"
        with sync_playwright() as pw:
            browser = launch(pw)
            if browser is None:
                self.skipTest("no Edge / Chrome for Playwright")
            try:
                # control (known answer): the old page directly, its one card claimed by someone else -> skipped
                st, _, body = T.request(self.port, "POST", f"/{PKG_B}/api/claim?c={card.replace('|', '%7C')}&who=other",
                                        {"X-Review-Key": self.e.key[PKG_B]}, b"")
                self.assertEqual((st, json.loads(body)["claims"].get(card)), (200, "other"))
                ctx = browser.new_context(**PHONE)
                page = ctx.new_page()
                page.goto(f"http://127.0.0.1:{self.port}/{PKG_B}/?k={self.e.key[PKG_B]}&who=x&mode=cards")
                page.wait_for_function("window.__review && __review.state().server && __review.state().tri.queue.length")
                page.wait_for_function("__review.state().tri.pos >= __review.state().tri.queue.length", timeout=15000)
                self.assertEqual(page.evaluate(state), [1, 1])            # the one card, skipped
                ctx.close()

                # through the portal: the current tool, no prompt, the card claimed as u-basic and kept on screen
                ctx = browser.new_context(**PHONE)
                errs, claims_seen = [], []
                page = ctx.new_page()
                page.on("pageerror", lambda e: errs.append(str(e)))
                page.on("request", lambda r: "api/claims" in r.url and claims_seen.append(r.url))
                page.goto(f"{origin}/login?k={self.e.keys['basic']}")
                page.goto(f"{origin}/review/{PKG_A}/?mode=cards")
                fr, frame = tool_frame(page, origin, PKG_A), page.frame_locator("#cj-review-frame")
                self.assertEqual(fr.evaluate("__review.state().who"), "u-basic")
                self.assertTrue(frame.locator("#whoDlg").is_hidden())
                for _ in range(100):
                    if self.claims(PKG_A).get(card) == "u-basic":
                        break
                    page.wait_for_timeout(100)
                self.assertEqual(self.claims(PKG_A).get(card), "u-basic")
                n = len(claims_seen)
                for _ in range(150):                                      # a claims poll after the claim (5 s)
                    if len(claims_seen) > n:
                        break
                    page.wait_for_timeout(100)
                self.assertGreater(len(claims_seen), n)
                for _ in range(150):                                      # one more poll: it had its chance to move us
                    if len(claims_seen) > n + 1:
                        break
                    page.wait_for_timeout(100)
                self.assertGreater(len(claims_seen), n + 1)
                self.assertEqual(fr.evaluate(state), [0, 1])              # still on its own card
                self.assertTrue(frame.locator("#whoDlg").is_hidden())
                self.assertEqual(errs, [])
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
