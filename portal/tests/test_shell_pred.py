"""
Prediction review packages in the shell: /review/<pkg_id>/ = the review shell, /review/<pkg_id>/app/... proxied to the
multi-package review server at <pred upstream>/<pkg_id>/..., with that package's own .review_key. Owner check
(ctx.can_see: admins all, others their own; unknown and not-yours give one identical 404), every proxy rule, ids
outside ^pred_ps_<8 digits>_<6 hex>$ -> 404 with nothing upstream; the real /review stays admin only.

Packages are built by annotation/prediction_package.py from the CryoEV golden prediction set into a temp review root.
A recording dummy upstream (cjtest.Upstream), and annotation/multi_review_server.py itself as the control.

  python -m unittest discover -s portal/tests -v
"""
import http.server
import json
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from annotation import multi_review_server as mrs  # noqa: E402
from annotation import prediction_package as pp  # noqa: E402
from portal import app as APP  # noqa: E402
from portal import auth as A  # noqa: E402
from portal import proxy as P  # noqa: E402

GOLDEN = T.REPO / "tests" / "fixtures" / "cj_golden" / "predict" / "ps_20261105_abc123"
PKG_A = "pred_ps_20261105_abc123"            # owner u-basic
PKG_B = "pred_ps_20261105_def456"            # owner u-adv
PKG_NOKEY = "pred_ps_20261105_0000aa"        # owner u-basic, no .review_key
UNKNOWN = "pred_ps_20261105_ffffff"          # well-formed, no such package


def build_root(tmp: Path) -> Path:
    """<tmp>/review with three packages built from copies of the golden prediction set."""
    root = tmp / "review"
    root.mkdir()
    for ps_id, owner in (("ps_20261105_abc123", "u-basic"), ("ps_20261105_def456", "u-adv"),
                         ("ps_20261105_0000aa", "u-basic")):
        src = tmp / "predict" / ps_id
        shutil.copytree(GOLDEN, src)
        req = json.loads((src / "request.json").read_text(encoding="utf-8"))
        req["ps"] = ps_id
        (src / "request.json").write_text(json.dumps(req), encoding="utf-8")
        pp.build(src, root, owner=owner)
    (root / PKG_NOKEY / ".review_key").unlink()
    return root


class PredEnv:
    """Temp roster (admin tester, basic u-basic, advanced u-adv), review root with the packages, the portal."""

    def __init__(self, upstream_url, root=None):
        self.env = T.Env()
        self.root = root or build_root(self.env.tmp)
        keys = {"admin": self.env.key, "basic": A.add_key(self.env.roster, "u-basic", "basic"),
                "adv": A.add_key(self.env.roster, "u-adv", "advanced"),
                "basic2": A.add_key(self.env.roster, "u-other", "basic")}
        self.p = T.Portal(self.env.config(pred_upstream=upstream_url, review_root=self.root))
        self.keys = keys
        self.tokens = {who: T.login(self.p.port, k) for who, k in keys.items()}
        self.key = {pid: mrs.package_key(self.root / pid) for pid in (PKG_A, PKG_B)}

    def req(self, who, method, path, headers=None, body=None):
        h = {**T.cookie(self.tokens[who]), **(headers or {})}
        if method in APP.UNSAFE:
            h.setdefault("Origin", self.p.origin)
        return T.request(self.p.port, method, path, h, body)

    def stop(self):
        self.p.stop()
        self.env.cleanup()


def hdrs(rec):
    return {k.lower(): v for k, v in rec["headers"]}


class MultiDummy:
    """A dummy multi-package server: records every request; for /<pkg>/<what>: setcookie (Set-Cookie + a hop-by-hop
    header), in (302 to /<pkg>/y), out (302 to /x), abs (302 to its own absolute /<pkg>/z), else a JSON echo."""

    def __init__(self):
        self.seen = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def handle_any(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                rec = {"method": self.command, "path": self.path, "headers": list(self.headers.items()),
                       "body_len": len(body)}
                outer.seen.append(rec)
                parts = self.path.split("?", 1)[0].split("/")
                pkg, what = (parts[1], parts[2]) if len(parts) > 2 else ("", "")
                hd, out = [], json.dumps(rec).encode()
                code = 200
                if what == "setcookie":
                    hd, out = [("Set-Cookie", "review_key_8767_x=secret; Path=/"), ("Connection", "x-hop"),
                               ("X-Hop", "1"), ("X-Keep", "1")], b"ok"
                elif what in ("in", "out", "abs"):
                    code = 302
                    hd = [("Location", {"in": f"/{pkg}/y?q=1", "out": "/x",
                                        "abs": f"{outer.url}/{pkg}/z"}[what])]
                    out = b""
                self.send_response(code)
                for k, v in hd:
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(out)

            do_GET = do_HEAD = do_PUT = do_POST = do_DELETE = handle_any

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def last(self):
        return self.seen[-1] if self.seen else None

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()


class PredDummy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.up = MultiDummy()
        cls.e = PredEnv(cls.up.url)

    @classmethod
    def tearDownClass(cls):
        cls.e.stop()
        cls.up.stop()

    def test_owner_and_admin_can(self):
        for who in ("basic", "admin"):
            st, h, body = self.e.req(who, "GET", f"/review/{PKG_A}/?img=0002.png&k=leak-1")
            self.assertEqual(st, 200, who)
            self.assertIn(f'src="/review/{PKG_A}/app/?img=0002.png"'.encode(), body)
            self.assertIn(f'data-base="/review/{PKG_A}/"'.encode(), body)
            self.assertNotIn(b"leak-1", body)
            self.assertEqual(T.hget(h, "x-frame-options"), "SAMEORIGIN")
            st, _, _ = self.e.req(who, "GET", f"/review/{PKG_A}/app/echo?k=stale&f=1")
            self.assertEqual(st, 200, who)
            rec = self.up.last()
            self.assertEqual(rec["path"], f"/{PKG_A}/echo?f=1")                   # /<pkg id>/..., k gone
            h = hdrs(rec)
            self.assertEqual(h.get("x-review-key"), self.e.key[PKG_A])            # that package's own key
            self.assertEqual(h.get("x-cj-user"), {"basic": "u-basic", "admin": T.USER}[who])
            self.assertNotIn("cookie", h)
        st, _, _ = self.e.req("adv", "GET", f"/review/{PKG_B}/app/echo")         # advanced owner: their own
        self.assertEqual((st, hdrs(self.up.last()).get("x-review-key")), (200, self.e.key[PKG_B]))
        self.assertNotEqual(self.e.key[PKG_A], self.e.key[PKG_B])

    def test_non_owner_and_unknown_get_one_answer(self):
        n = len(self.up.seen)
        answers = set()
        for who, pid in (("basic2", PKG_A), ("adv", PKG_A), ("basic", PKG_B), ("basic", UNKNOWN), ("admin", UNKNOWN),
                         ("basic", PKG_NOKEY)):
            for path in (f"/review/{pid}/", f"/review/{pid}/app", f"/review/{pid}/app/", f"/review/{pid}/app/api/edits"):
                st, _, body = self.e.req(who, "GET", path)
                answers.add((st, body))
            st, _, body = self.e.req(who, "PUT", f"/review/{pid}/app/api/edits?f=0001.png", body=b"{}")
            answers.add((st, body))
        self.assertEqual(len(answers), 1, answers)                                # identical: can't tell them apart
        self.assertEqual(next(iter(answers))[0], 404)
        self.assertEqual(len(self.up.seen), n)                                    # nothing reached the server

    def test_ids_outside_the_pattern(self):
        n = len(self.up.seen)
        for pid in ("pred_ps_2026110_abc123", "pred_ps_20261105_ABC123", "PRED_ps_20261105_abc123", "pred_x",
                    "pred_ps_20261105_abc1234", "app2", "pred_ps_20261105_abc123.", "ps_20261105_abc123"):
            for who in ("admin", "basic"):
                st, _, _ = self.e.req(who, "GET", f"/review/{pid}/app/echo")
                self.assertEqual(st, 404, (who, pid))
        self.assertEqual(len(self.up.seen), n)

    def test_real_review_stays_admin_only(self):
        for who in ("basic", "adv"):
            for path in ("/review/", "/review/app/x", "/review"):
                self.assertEqual(self.e.req(who, "GET", path)[0], 403, (who, path))

    def test_proxy_rules(self):
        st, h, body = self.e.req("basic", "GET", f"/review/{PKG_A}/app/setcookie")
        self.assertEqual((st, body), (200, b"ok"))
        self.assertIsNone(T.hget(h, "set-cookie"))
        self.assertEqual(T.hget(h, "x-keep"), "1")
        self.assertIsNone(T.hget(h, "x-hop"))
        for src, want in (("in", f"/review/{PKG_A}/app/y?q=1"), ("abs", f"/review/{PKG_A}/app/z"),
                          ("out", f"/review/{PKG_A}/app/")):       # outside this package -> its own root
            st, h, _ = self.e.req("basic", "GET", f"/review/{PKG_A}/app/{src}")
            self.assertEqual((st, T.hget(h, "location")), (302, want), src)
        st, _, _ = self.e.req("basic", "PUT", f"/review/{PKG_A}/app/api/edits?f=0001.png&k=x",
                              {"Content-Type": "application/json"}, b'{"t": 1}')
        self.assertEqual((st, self.up.last()["path"]), (200, f"/{PKG_A}/api/edits?f=0001.png"))
        n = len(self.up.seen)
        st, _, _ = self.e.req("basic", "PUT", f"/review/{PKG_A}/app/api/edits?f=a", {"Origin": "http://evil.example"},
                              b"{}")
        self.assertEqual(st, 403)
        for t in (f"/review/{PKG_A}/app/../x", f"/review/{PKG_A}/app/%2e%2e/x", f"/review/{PKG_A}/../app/x",
                  f"/review/{PKG_A}/app//x", f"/review/{PKG_A}/app/%252e%252e/{PKG_B}/x", f"/review/{PKG_A}/./app/x"):
            resp = T.raw(self.e.p.port, b"GET " + t.encode() + b" HTTP/1.1\r\nHost: 127.0.0.1\r\nCookie: "
                         b"__Host-cj_session=" + self.e.tokens["basic"].encode() + b"\r\nConnection: close\r\n\r\n")
            self.assertIn(T.raw_status(resp), (400, 403, 404), t)
        self.assertEqual(len(self.up.seen), n)
        st, h, _ = self.e.req("basic", "GET", f"/review/{PKG_A}/app/?mode=cards&k=z", {"Sec-Fetch-Dest": "document"})
        self.assertEqual((st, T.hget(h, "location")), (302, f"/review/{PKG_A}/?mode=cards"))
        st, h, _ = self.e.req("basic", "GET", f"/review/{PKG_A}/app")
        self.assertEqual((st, T.hget(h, "location")), (307, f"/review/{PKG_A}/app/"))

    def test_location_inside_the_package(self):
        up, pre, path = "http://127.0.0.1:8767", f"/review/{PKG_A}/app", f"/{PKG_A}"
        for value, want in ((f"/{PKG_A}/x?a=1", f"{pre}/x?a=1"), (f"/{PKG_A}", f"{pre}/"), (f"/{PKG_A}/", f"{pre}/"),
                            (f"{up}/{PKG_A}/y", f"{pre}/y"), (f"/{PKG_B}/x", f"{pre}/"), (f"/{PKG_A}x/y", f"{pre}/"),
                            ("/y", f"{pre}/"), (f"/{PKG_A}/../{PKG_B}/x", f"{pre}/"), ("http:evil.com", f"{pre}/")):
            self.assertEqual(P.rewrite_location(value, up, pre, path), want, value)

    def test_no_review_root(self):
        env = T.Env()
        try:
            p = T.Portal(env.config(pred_upstream=self.up.url, review_root=None))
            try:
                tok = T.login(p.port, env.key)
                n = len(self.up.seen)
                st, _, _ = T.request(p.port, "GET", f"/review/{PKG_A}/app/echo", T.cookie(tok))
                self.assertEqual(st, 404)
                self.assertEqual(len(self.up.seen), n)
            finally:
                p.stop()
        finally:
            env.cleanup()


class PredRealServer(unittest.TestCase):
    """Control with known answers: annotation/multi_review_server.py on the temp root. Direct without the key: 403;
    through the shell with the owner's session: 200, and an edit lands in that package."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cj_pred_"))
        cls.root = build_root(cls.tmp)
        cls.srv = mrs.make_server(cls.root, "127.0.0.1", 0, people=[])
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.e = PredEnv(f"http://127.0.0.1:{cls.srv.server_address[1]}", cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.e.stop()
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_through_the_shell(self):
        port = self.srv.server_address[1]
        self.assertEqual(T.request(port, "GET", f"/{PKG_A}/api/ping")[0], 403)          # control: no key
        st, _, body = self.e.req("basic", "GET", f"/review/{PKG_A}/app/api/ping")
        ping = json.loads(body)  # the review server's own answer, passed through: ok, and its round-7 caps
        self.assertEqual((st, ping["ok"]), (200, True))
        self.assertTrue({"i", "t"} <= set(ping.get("caps") or []), ping)
        st, h, body = self.e.req("basic", "GET", f"/review/{PKG_A}/app/?k=wrong")
        self.assertEqual(st, 200)
        self.assertIn(b"<html", body.lower())
        self.assertIsNone(T.hget(h, "set-cookie"))
        st, _, body = self.e.req("basic", "GET", f"/review/{PKG_A}/app/data.js")
        self.assertEqual(st, 200)
        self.assertIn(PKG_A.encode(), body)
        self.assertEqual(self.e.req("adv", "GET", f"/review/{PKG_A}/app/api/ping")[0], 404)   # not theirs
        data = pp.read_datajs(self.root / PKG_A)
        f = data["images"][0]["file"]
        edit = {"t": 5, "dec": {}, "reviewed": True, "reviewedT": 5}
        st, _, body = self.e.req("basic", "PUT", f"/review/{PKG_A}/app/api/edits?f={f}",
                                 {"Content-Type": "application/json"}, json.dumps(edit).encode())
        self.assertEqual(st, 200, body)
        saved = json.loads((self.root / PKG_A / "edits" / f"{f}.json").read_text(encoding="utf-8"))
        self.assertTrue(saved["edit"]["reviewed"])
        self.assertFalse((self.root / PKG_B / "edits" / f"{f}.json").exists())

    def test_in_a_browser(self):
        """The owner (basic) opens the package's shell: the tool loads in the frame from /review/<pkg>/app/, only
        same-origin requests, no page or console errors."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:  # pragma: no cover
            self.skipTest("playwright not installed")
        key = self.e.keys["basic"]                                  # the owner's key link, in the browser
        origin = self.e.p.origin
        with sync_playwright() as pw:
            browser = None
            for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
                try:
                    browser = pw.chromium.launch(headless=True, **kw)
                    break
                except Exception:
                    continue
            if browser is None:
                self.skipTest("no Edge / Chrome for Playwright")
            try:
                ctx = browser.new_context(viewport={"width": 390, "height": 800})
                rec = {"req": [], "foreign": [], "err": []}
                ctx.route("**/*", lambda r: (r.continue_() if r.request.url.startswith(origin + "/")
                                             else (rec["foreign"].append(r.request.url), r.abort())))
                page = ctx.new_page()
                page.on("request", lambda r: rec["req"].append(r.url))
                page.on("pageerror", lambda e: rec["err"].append(str(e)))
                page.on("console", lambda m: m.type == "error" and rec["err"].append(m.text))
                page.goto(f"{origin}/login?k={key}")
                page.goto(f"{origin}/review/{PKG_A}/?mode=cards")
                page.wait_for_load_state("networkidle")
                page.wait_for_selector("#cj-bar .cj-nav a")
                self.assertEqual(page.get_attribute("#cj-review-frame", "src"), f"/review/{PKG_A}/app/?mode=cards")
                self.assertTrue(any(u.endswith(f"/review/{PKG_A}/app/api/edits") for u in rec["req"]), rec["req"])
                self.assertTrue(any(f"/review/{PKG_A}/app/data.js" in u for u in rec["req"]))
                self.assertEqual(page.url, f"{origin}/review/{PKG_A}/?mode=cards")
                self.assertEqual((rec["foreign"], rec["err"]), ([], []))
                shots = __import__("os").environ.get("CJ_SHOT_DIR")
                if shots:
                    page.screenshot(path=str(Path(shots) / "shell_pred_390.png"))
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
