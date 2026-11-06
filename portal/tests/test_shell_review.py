"""
The review inside the shell, over HTTP (no browser): /review/ = shell page (bar + iframe of /review/app/), the tool
proxied at /review/app/... with every proxy rule (k stripped, header allowlist, no Cookie, Set-Cookie dropped,
Location kept under /review/app/), /review/<other> kept free (404, nothing upstream), and the role Gate for core pages
(review and history are admin only, from modules.json).

A dummy upstream that records what reaches it (cjtest.Upstream), and the real review_server.py handler on a temp
fixture package as the control with known answers (cjtest.ReviewUpstream).

  python -m unittest discover -s portal/tests -v
"""
import html
import json
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import app as APP  # noqa: E402
from portal import auth as A  # noqa: E402
from portal import proxy as P  # noqa: E402

SPOOF = {"X-CJ-User": "mallory", "X-Forwarded-For": "6.6.6.6", "Forwarded": "for=6.6.6.6",
         "Tailscale-User-Login": "mallory@example.com", "Cf-Access-Jwt-Assertion": "forged.jwt",
         "X-Review-Key": "browser-supplied-key", "Authorization": "Basic Zm9vOmJhcg=="}


def hdrs(rec):
    return {k.lower(): v for k, v in rec["headers"]}


def iframe_src(body: bytes) -> str:
    m = re.search(rb'<iframe id="cj-review-frame"[^>]* src="([^"]*)"', body)
    return html.unescape(m.group(1).decode("utf-8")) if m else None


class ShellDummy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.up = T.Upstream()
        cls.p = T.Portal(cls.env.config(review_upstream=cls.up.url))
        cls.token = T.login(cls.p.port, cls.env.key)

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.up.stop()
        cls.env.cleanup()

    def req(self, method, path, headers=None, body=None):
        h = {**T.cookie(self.token), **(headers or {})}
        if method in APP.UNSAFE:
            h.setdefault("Origin", self.p.origin)
        return T.request(self.p.port, method, path, h, body)

    def test_shell_page_carries_query_into_iframe(self):
        n = len(self.up.seen)
        st, h, body = self.req("GET", "/review/?img=a%20b.png&mode=editor&who=pujan&k=secret-key-77&obj=g1")
        self.assertEqual(st, 200)
        self.assertEqual(iframe_src(body), "/review/app/?img=a%20b.png&mode=editor&who=pujan&obj=g1")
        self.assertNotIn(b"secret-key-77", body)                           # k never goes into the frame
        self.assertIn(b'<body class="cj-page" data-active="review">', body)  # base.html: the bar + the frame
        self.assertIn(b'<div id="cj-bar" class="cj-bar" role="banner">', body)
        self.assertIn(b'<script src="/cj/review.js" defer></script>', body)
        self.assertEqual(T.hget(h, "x-frame-options"), "SAMEORIGIN")
        self.assertEqual(T.hget(h, "cache-control"), "no-store")
        self.assertEqual(len(self.up.seen), n)                             # the shell itself asks the upstream nothing
        st, _, body = self.req("GET", "/review/")                         # control: no query, plain src
        self.assertEqual((st, iframe_src(body)), (200, "/review/app/"))

    def test_shell_query_escaped(self):
        st, _, body = self.req("GET", '/review/?img=%22%3E%3Cscript%3E&x="><script>alert(1)</script>')
        self.assertEqual(st, 200)
        self.assertNotIn(b"<script>alert", body)
        self.assertIn(b"&quot;&gt;&lt;script&gt;", body)

    def test_app_proxied_with_prefix_stripped(self):
        st, _, body = self.req("GET", "/review/app/echo?k=stale&f=a%7Cb&K=upper&%6B=enc&c=x%20y",
                               {"Cookie": f"__Host-cj_session={self.token}; review_key_8766=old", "Accept": "x/y"})
        self.assertEqual(st, 200)
        rec = self.up.last()
        self.assertEqual(rec["path"], "/echo?f=a%7Cb&K=upper&c=x%20y")      # /review/app removed, every k gone
        h = hdrs(rec)
        self.assertEqual((h.get("x-review-key"), h.get("x-cj-user"), h.get("accept")), (T.UPKEY, T.USER, "x/y"))
        self.assertNotIn("cookie", h)
        st, _, _ = self.req("GET", "/review/app/")
        self.assertEqual((st, self.up.last()["path"]), (200, "/"))

    def test_spoofed_headers_stripped(self):
        st, _, _ = self.req("GET", "/review/app/echo", SPOOF)
        self.assertEqual(st, 200)
        rec = self.up.last()
        h = hdrs(rec)
        self.assertEqual((h.get("x-cj-user"), h.get("x-review-key")), (T.USER, T.UPKEY))
        self.assertEqual([k for k, _ in rec["headers"] if k.lower() == "x-cj-user"], ["x-cj-user"])
        for k in h:
            self.assertFalse(k.startswith(("x-forwarded-", "tailscale-", "cf-")) or k in ("forwarded", "authorization"), k)

    def test_set_cookie_dropped_and_location_kept_under_app(self):
        st, h, body = self.req("GET", "/review/app/setcookie")
        self.assertEqual((st, body), (200, b"ok"))
        self.assertIsNone(T.hget(h, "set-cookie"))
        self.assertIsNone(T.hget(h, "x-hop"))
        self.assertEqual(T.hget(h, "x-keep"), "1")                          # control: an ordinary header passes
        self.assertEqual(T.hget(h, "x-frame-options"), "SAMEORIGIN")
        self.req("GET", "/review/app/echo")
        self.assertNotIn("cookie", hdrs(self.up.last()))                    # the proxy's client kept no cookie
        for src, want in (("/review/app/redir-abs", "/review/app/x"), ("/review/app/redir-rel", "/review/app/y")):
            st, h, _ = self.req("GET", src)
            self.assertEqual((st, T.hget(h, "location")), (302, want))

    def test_put_and_origin(self):
        body = json.dumps({"t": 1}).encode()
        st, _, _ = self.req("PUT", "/review/app/api/edits?f=a.png&k=x", {"Content-Type": "application/json"}, body)
        self.assertEqual(st, 200)
        self.assertEqual((self.up.last()["method"], self.up.last()["path"]), ("PUT", "/api/edits?f=a.png"))
        n = len(self.up.seen)
        st, _, _ = self.req("PUT", "/review/app/api/edits?f=a", {"Origin": "http://evil.example"}, b"{}")
        self.assertEqual(st, 403)
        self.assertEqual(len(self.up.seen), n)

    def test_redirects(self):
        for path, want in (("/review", "/review/"), ("/review/app", "/review/app/")):
            st, h, _ = self.req("GET", path)
            self.assertEqual((st, T.hget(h, "location")), (307, want), path)

    def test_other_review_paths_kept_free(self):
        n = len(self.up.seen)
        for path in ("/review/x", "/review/pkg_20261105_abc/", "/review/api/edits", "/review/data.js", "/review/apps/x"):
            for method in ("GET", "PUT"):
                st, h, _ = self.req(method, path, body=b"{}" if method == "PUT" else None)
                self.assertEqual(st, 404, (method, path))
        self.assertEqual(len(self.up.seen), n)                             # nothing reached the review server

    def test_traversal_under_app_rejected(self):
        n = len(self.up.seen)
        for t in (b"/review/app/../x", b"/review/app/%2e%2e/x", b"/review/app//x", b"/review/app/..%2f..%2fhistory",
                  b"/review/app/%252e%252e/x", b"/review/app/a%00b", b"/review/app/..\\x"):
            resp = T.raw(self.p.port, b"GET " + t + b" HTTP/1.1\r\nHost: 127.0.0.1\r\nCookie: __Host-cj_session="
                         + self.token.encode() + b"\r\nConnection: close\r\n\r\n")
            self.assertIn(T.raw_status(resp), (400, 404), t)
        self.assertEqual(len(self.up.seen), n, [r["path"] for r in self.up.seen[n:]])
        st, _, _ = self.req("GET", "/review/app/images/ok.png")             # control
        self.assertEqual((st, self.up.last()["path"]), (200, "/images/ok.png"))

    def test_review_mount_checks_proxy_prefix(self):
        with self.assertRaises(ValueError):
            APP.review_mount("/review", P.ReviewProxy("http://127.0.0.1:1", None))      # default prefix /review
        routes = APP.review_mount("/review/pkg1", P.ReviewProxy("http://127.0.0.1:1", None, prefix="/review/pkg1/app"))
        self.assertEqual([r.path for r in routes], ["/review/pkg1/", "/review/pkg1/app", "/review/pkg1/app/{path:path}"])

    def test_x_frame_options(self):
        for path, want in (("/", "DENY"), ("/cj/me.json", "DENY"), ("/review/", "SAMEORIGIN"),
                           ("/review/app/echo", "SAMEORIGIN"), ("/history/", "DENY")):
            _, h, _ = self.req("GET", path)
            self.assertEqual(T.hall(h, "x-frame-options"), [want], path)


class ShellRoles(unittest.TestCase):
    """The role Gate on core pages: review and history are admin only (modules.json); advanced and basic get 403 and
    nothing reaches the review server. Home and the shell's files are open to every signed-in role."""

    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.up = T.Upstream()
        (cls.env.registry / "history.html").write_text("<html><head></head><body>h</body></html>", encoding="utf-8")
        cls.env.write_live(T.live_doc())
        cls.p = T.Portal(cls.env.config(review_upstream=cls.up.url))
        cls.tokens = {"admin": T.login(cls.p.port, cls.env.key)}
        for role in ("advanced", "basic"):
            cls.tokens[role] = T.login(cls.p.port, A.add_key(cls.env.roster, f"u-{role}", role))

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.up.stop()
        cls.env.cleanup()

    def get(self, role, path):
        return T.request(self.p.port, "GET", path, T.cookie(self.tokens[role]))

    def test_admin_only_pages(self):
        closed = ("/review/", "/review/app/", "/review/app/api/ping", "/review", "/history/history.html",
                  "/history/live/live.json", "/history/")
        n = len(self.up.seen)
        for role in ("advanced", "basic"):
            for path in closed:
                st, _, body = self.get(role, path)
                self.assertEqual(st, 403, (role, path))
                self.assertIn(b"Your role", body)
        self.assertEqual(len(self.up.seen), n)
        for path in ("/review/", "/history/history.html", "/history/live/live.json"):       # control: admin
            self.assertEqual(self.get("admin", path)[0], 200, path)
        self.assertEqual(self.get("admin", "/review/app/api/ping")[0], 200)

    def test_open_to_every_role(self):
        for role in ("admin", "advanced", "basic"):
            for path in ("/", "/cj/me.json", "/cj/modules.json", "/cj/header.js", "/cj/live.js", "/cj/review.js",
                         "/cj/brand.json", "/cj/cj.css"):
                self.assertEqual(self.get(role, path)[0], 200, (role, path))
            st, _, body = self.get(role, "/cj/me.json")
            self.assertEqual(json.loads(body)["role"], role)

    def test_manifest_rules(self):
        mods = [{"id": "home", "route": "/", "roles": ["admin", "basic"]},
                {"id": "review", "route": "/review/", "roles": ["admin", "advanced"]},
                {"id": "jobs", "route": "/jobs/", "roles": ["admin", "advanced"]},
                {"id": "sandbox", "route": None, "external": {"public_port": 8443}, "roles": ["admin"]},
                {"id": "x", "route": None, "roles": ["admin"]}]
        self.assertEqual(APP.manifest_rules(mods), [("/review", ("admin", "advanced")), ("/history", ("admin",)),
                                                    ("/jobs", ("admin", "advanced"))])
        self.assertEqual(APP.manifest_rules(mods, {"/jobs"}),                         # a plug-in's ROLES win
                         [("/review", ("admin", "advanced")), ("/history", ("admin",))])
        # fail closed: nothing, no route, bad roles or junk -> the built-in defaults stay
        defaults = [("/review", ("admin",)), ("/history", ("admin",))]
        for bad in ([], [{"id": "review", "route": None, "roles": ["basic"]}],
                    [{"id": "review", "route": "/review/", "roles": []}],
                    [{"id": "review", "route": "/review/", "roles": ["basic", "root"]}],
                    [{"id": "review", "route": "/review/", "roles": "basic"}],
                    [{"id": "review", "route": "//review/", "roles": ["basic"]}], ["junk", 3]):
            self.assertEqual(APP.manifest_rules(bad), defaults, bad)
        rules = APP.manifest_rules(APP.load_manifest())                               # the shipped file
        self.assertIn(("/review", ("admin",)), rules)
        self.assertIn(("/history", ("admin",)), rules)

    def test_modules_json_filtered_by_role(self):
        want = {"admin": None, "advanced": {"home", "data", "jobs", "predict", "results", "train"},
                "basic": {"home", "predict", "results"}}
        every = {m["id"] for m in APP.load_manifest()}
        for role, ids in want.items():
            st, h, body = self.get(role, "/cj/modules.json")
            self.assertEqual(st, 200)
            got = {m["id"] for m in json.loads(body)["modules"]}
            self.assertEqual(got, ids or every, role)
            self.assertEqual(T.hget(h, "cache-control"), "no-store")


class FailClosed(unittest.TestCase):
    """The Gate's core rules never come from guesses: built-in defaults (review, history: admin), replaced only by a
    valid modules.json entry; a modules.json that can't be read or parsed stops the portal."""

    def portal_with(self, text):
        env = T.Env()
        self.addCleanup(env.cleanup)
        up = T.Upstream()
        self.addCleanup(up.stop)
        (env.registry / "history.html").write_text("<html><head></head><body>h</body></html>", encoding="utf-8")
        man = env.tmp / "modules.json"
        man.write_text(text, encoding="utf-8")
        p = T.Portal(env.config(review_upstream=up.url, manifest=man))
        self.addCleanup(p.stop)
        basic = T.login(p.port, A.add_key(env.roster, "u-basic", "basic"))
        return p, up, basic

    def statuses(self, p, token, paths=("/review/", "/review/app/x", "/history/history.html")):
        return [T.request(p.port, "GET", path, T.cookie(token))[0] for path in paths]

    def test_weak_manifests_keep_defaults(self):
        shipped = APP.load_manifest()
        no_review = [m for m in shipped if m["id"] != "review"]
        null_route = [dict(m, route=None) if m["id"] in ("review", "history") else m for m in shipped]
        bad_roles = [dict(m, roles=["basic", "root"]) if m["id"] == "review" else m for m in shipped]
        empty_roles = [dict(m, roles=[]) if m["id"] == "review" else m for m in shipped]
        for name, mods in (("empty", []), ("no review", no_review), ("route null", null_route),
                           ("bad roles", bad_roles), ("empty roles", empty_roles)):
            with self.subTest(manifest=name):
                p, up, basic = self.portal_with(json.dumps({"modules": mods}))
                self.assertEqual(self.statuses(p, basic), [403, 403, 403])
                self.assertEqual(up.seen, [])                                 # nothing reached the review server

    def test_valid_entry_replaces_default(self):
        """Control: the same harness does open /review to basic when modules.json validly says so."""
        mods = [dict(m, roles=["admin", "basic"]) if m["id"] == "review" else m for m in APP.load_manifest()]
        p, up, basic = self.portal_with(json.dumps({"modules": mods}))
        self.assertEqual(self.statuses(p, basic), [200, 200, 403])
        self.assertEqual(up.last()["path"], "/x")

    def test_unreadable_manifest_refuses_to_start(self):
        env = T.Env()
        self.addCleanup(env.cleanup)
        man = env.tmp / "modules.json"
        for text in (None, "{not json", "[]", '{"modules": {"a": 1}}', '{"modules": ["x"]}'):
            with self.subTest(text=text):
                if text is None:
                    man.unlink(missing_ok=True)
                else:
                    man.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    APP.create_app(env.config(manifest=man))
        with mock.patch("uvicorn.run") as run, mock.patch("logging.basicConfig"), \
                mock.patch("builtins.print") as out:
            rc = APP.main(["--registry", str(env.registry), "--roster", str(env.roster), "--state-dir",
                           str(env.state), "--review-package", str(env.tmp / "none"), "--modules", str(man)])
        self.assertEqual(rc, 2)
        run.assert_not_called()
        self.assertTrue(any("refusing to start" in str(c) for c in out.call_args_list))

    def test_startup_note_on_roles(self):
        env = T.Env()
        self.addCleanup(env.cleanup)
        lines = "\n".join(APP.startup_lines(env.config(), "127.0.0.1", 8770))
        self.assertIn("/review admin", lines)
        self.assertIn("Restart the portal after changing roles in modules.json", lines)


class ToShell(unittest.TestCase):
    """A copied /review/app/ link opened as a whole page goes to the shell (302 /review/?<query minus k>); the frame's
    own loads and every other request still reach the tool."""

    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.up = T.Upstream()
        cls.p = T.Portal(cls.env.config(review_upstream=cls.up.url))
        cls.token = T.login(cls.p.port, cls.env.key)

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.up.stop()
        cls.env.cleanup()

    def get(self, path, headers=None):
        return T.request(self.p.port, "GET", path, {**T.cookie(self.token), **(headers or {})})

    def test_top_level_document_redirects(self):
        n = len(self.up.seen)
        shell_ref = f"{self.p.origin}/review/?mode=cards"
        cases = [{"Sec-Fetch-Dest": "document"}, {"Sec-Fetch-Dest": "document", "Referer": shell_ref},
                 {"Accept": "text/html,application/xhtml+xml"},
                 {"Accept": "text/html", "Referer": f"{self.p.origin}/"},
                 {"Accept": "text/html", "Referer": "http://elsewhere.example/review/"}]
        for hdrs in cases:
            for path in ("/review/app/?img=a.png&k=secret-9&mode=editor",
                         "/review/app/index.html?img=a.png&k=s&mode=editor"):
                st, h, _ = self.get(path, hdrs)
                self.assertEqual((st, T.hget(h, "location")), (302, "/review/?img=a.png&mode=editor"), (hdrs, path))
        st, h, _ = self.get("/review/app/", {"Sec-Fetch-Dest": "document"})
        self.assertEqual((st, T.hget(h, "location")), (302, "/review/"))
        self.assertEqual(len(self.up.seen), n)                               # none reached the tool

    def test_frame_and_other_requests_pass(self):
        shell_ref = f"{self.p.origin}/review/?mode=cards"
        for hdrs, path in (({"Sec-Fetch-Dest": "iframe"}, "/review/app/?mode=cards"),
                           ({"Accept": "text/html", "Referer": shell_ref}, "/review/app/?mode=cards"),
                           ({}, "/review/app/?mode=cards"),                      # no browser headers at all
                           ({"Sec-Fetch-Dest": "document"}, "/review/app/data.js"),   # only the tool's page moves
                           ({"Sec-Fetch-Dest": "empty"}, "/review/app/api/edits")):
            st, _, _ = self.get(path, hdrs)
            self.assertEqual(st, 200, (hdrs, path))
            self.assertEqual(self.up.last()["path"], path[len("/review/app"):], (hdrs, path))


class LocationRewrite(unittest.TestCase):
    def test_rewrite_location(self):
        up, pre = "http://127.0.0.1:8766", "/review/app"
        for value, want in (("/y", "/review/app/y"), ("http://127.0.0.1:8766/x?a=1", "/review/app/x?a=1"),
                            ("http://127.0.0.1:8766", "/review/app/"), ("y?a=1", "y?a=1"),
                            ("https://elsewhere.example/x", "https://elsewhere.example/x"),
                            ("//evil.example/x", "/review/app/"), ("/\\evil.example", "/review/app/"),
                            ("http://127.0.0.1:8766//evil.example/x", "/review/app/"),
                            ("/a/../../history/history.html", "/review/app/"), ("/%2e%2e/x", "/review/app/"),
                            ("/./x", "/review/app/"), ("../x", "/review/app/"), ("/a//b", "/review/app/"),
                            ("/ok?next=../x", "/review/app/ok?next=../x")):
            self.assertEqual(P.rewrite_location(value, up, pre), want, value)


class ShellRealReviewServer(unittest.TestCase):
    """Control with known answers: review_server.py directly without a key answers 403; through /review/app/ (key from
    the package's .review_key) 200, also with a stale ?k= that would outrank the header if it were forwarded."""

    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.rv = T.ReviewUpstream()
        cls.p = T.Portal(cls.env.config(review_upstream=cls.rv.url, review_key=APP.read_review_key(cls.rv.pkg)))
        cls.token = T.login(cls.p.port, cls.env.key)

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.rv.stop()
        cls.env.cleanup()

    def test_through_shell(self):
        c = T.cookie(self.token)
        self.assertEqual(T.request(self.rv.port, "GET", "/api/ping")[0], 403)          # control: no key, refused
        st, _, body = T.request(self.p.port, "GET", "/review/app/api/ping", c)
        self.assertEqual((st, json.loads(body)), (200, {"ok": True}))
        st, h, body = T.request(self.p.port, "GET", "/review/app/?k=stale-wrong-key", c)
        self.assertEqual(st, 200)
        self.assertIn(b"<title>EV Label Review</title>", body)
        self.assertIsNone(T.hget(h, "set-cookie"))
        img = self.rv.files[0]
        st, _, body = T.request(self.p.port, "GET", f"/review/app/images/{img}", c)
        self.assertEqual((st, body), (200, (self.rv.pkg / "images" / img).read_bytes()))
        edit = {"t": 5, "dec": {"g1": {"v": "del", "t": 5, "who": "tester"}}, "reviewed": True, "reviewedT": 5}
        st, _, body = T.request(self.p.port, "PUT", f"/review/app/api/edits?f={img}",
                                {**c, "Origin": self.p.origin, "Content-Type": "application/json"},
                                json.dumps(edit).encode())
        self.assertEqual(st, 200, body)
        saved = json.loads((self.rv.pkg / "edits" / f"{img}.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["edit"]["dec"]["g1"]["v"], "del")


if __name__ == "__main__":
    unittest.main()
