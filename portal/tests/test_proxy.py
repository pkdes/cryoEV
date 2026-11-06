"""
The /review/app/ proxy (portal/proxy.py) through the whole app under uvicorn: a dummy upstream that records what reaches
it, plus the real review_server.py handler on a temp fixture package as the known-good control.

  python -m unittest discover -s portal/tests -v
"""
import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import app as APP  # noqa: E402
from portal import proxy as P  # noqa: E402

SPOOF = {"X-CJ-User": "mallory", "X-CJ-Gateway": "forged", "X-Forwarded-For": "6.6.6.6", "X-Forwarded-Host": "evil",
         "X-Forwarded-Proto": "https", "Forwarded": "for=6.6.6.6", "Tailscale-User-Login": "mallory@example.com",
         "Tailscale-User-Name": "Mallory", "Cf-Access-Jwt-Assertion": "forged.jwt", "Cf-Connecting-Ip": "6.6.6.6",
         "X-Review-Key": "browser-supplied-key", "Authorization": "Basic Zm9vOmJhcg=="}


def hdrs(rec):
    return {k.lower(): v for k, v in rec["headers"]}


class ProxyDummy(unittest.TestCase):
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

    def req(self, method, path, headers=None, body=None, origin=True):
        h = {**T.cookie(self.token), **(headers or {})}
        if origin and method in APP.UNSAFE:
            h.setdefault("Origin", self.p.origin)
        return T.request(self.p.port, method, path, h, body)

    def test_upstream_sees_key_and_user_but_no_cookie_or_k(self):
        st, _, body = self.req("GET", "/review/app/echo?k=stale-browser-key&f=a%7Cb&K=upper&k=again&c=x%20y",
                               {"Cookie": f"__Host-cj_session={self.token}; review_key_8766=old; other=1",
                                "Accept": "application/json", "User-Agent": "cj-test", "Accept-Language": "en"})
        self.assertEqual(st, 200)
        rec = self.up.last()
        h = hdrs(rec)
        self.assertEqual(h.get("x-review-key"), T.UPKEY)
        self.assertEqual(h.get("x-cj-user"), T.USER)
        self.assertNotIn("cookie", h)
        self.assertEqual(rec["path"], "/echo?f=a%7Cb&K=upper&c=x%20y")      # every k gone, the rest byte for byte
        self.assertEqual((h.get("accept"), h.get("user-agent"), h.get("accept-language")),
                         ("application/json", "cj-test", "en"))
        self.assertEqual(json.loads(body)["path"], rec["path"])

    def test_encoded_k_name_also_stripped(self):
        self.req("GET", "/review/app/echo?%6B=enc&k+=space&f=1")
        self.assertEqual(self.up.last()["path"], "/echo?k+=space&f=1")    # "k " is not k upstream either

    def test_spoofed_inbound_headers_stripped(self):
        st, _, _ = self.req("GET", "/review/app/echo", SPOOF)
        self.assertEqual(st, 200)
        h = hdrs(self.up.last())
        self.assertEqual(h.get("x-cj-user"), T.USER)
        self.assertEqual(h.get("x-review-key"), T.UPKEY)
        self.assertEqual([k for k, _ in self.up.last()["headers"] if k.lower() == "x-cj-user"], ["x-cj-user"])
        for k in h:
            self.assertFalse(k.startswith(("x-forwarded-", "tailscale-", "cf-")) or k in ("forwarded", "authorization")
                             or (k.startswith("x-cj-") and k != "x-cj-user"), k)

    def test_set_cookie_and_hop_by_hop_dropped(self):
        st, h, body = self.req("GET", "/review/app/setcookie")
        self.assertEqual((st, body), (200, b"ok"))
        names = {k.lower() for k, _ in h}
        self.assertNotIn("set-cookie", names)
        self.assertNotIn("x-hop", names)                                   # named in Connection: hop-by-hop
        self.assertNotIn("keep-alive", names)
        self.assertEqual(T.hget(h, "x-keep"), "1")                         # control: an ordinary header passes
        # and the proxy's own client never keeps it: the next request upstream carries no cookie
        self.req("GET", "/review/app/echo")
        self.assertNotIn("cookie", hdrs(self.up.last()))

    def test_range_and_head(self):
        st, h, body = self.req("GET", "/review/app/range", {"Range": "bytes=10-19"})
        self.assertEqual(st, 206)
        self.assertEqual(T.hget(h, "content-range"), "bytes 10-19/1024")
        self.assertEqual(body, T.Upstream.DATA[10:20])
        self.assertEqual(hdrs(self.up.last()).get("range"), "bytes=10-19")
        st, h, body = self.req("HEAD", "/review/app/range")
        self.assertEqual((st, body), (200, b""))
        self.assertEqual(T.hget(h, "content-length"), "1024")
        self.assertEqual(self.up.last()["method"], "HEAD")
        st, h, body = self.req("GET", "/review/app/range")                    # control: the full body
        self.assertEqual((st, len(body)), (200, 1024))

    def test_large_response_streams_intact(self):
        st, h, body = self.req("GET", "/review/app/big")
        self.assertEqual(st, 200)
        self.assertEqual(hashlib.md5(body).hexdigest(), hashlib.md5(T.Upstream.BIG).hexdigest())

    def test_put_body_forwarded(self):
        body = json.dumps({"t": 1, "dec": {}, "pad": "x" * 2_000_000}).encode()
        st, _, out = self.req("PUT", "/review/app/api/edits?f=a.png&k=stale", {"Content-Type": "application/json"}, body)
        self.assertEqual(st, 200)
        rec = self.up.last()
        self.assertEqual((rec["method"], rec["path"], rec["body_len"], rec["body_md5"]),
                         ("PUT", "/api/edits?f=a.png", len(body), hashlib.md5(body).hexdigest()))
        self.assertEqual(hdrs(rec).get("content-type"), "application/json")

    def test_cross_origin_refused(self):
        n = len(self.up.seen)
        for method in ("PUT", "POST", "DELETE", "PATCH"):
            for origin in ("http://evil.example", "https://127.0.0.1.evil.example", "null",
                           f"https://127.0.0.1:{self.p.port + 1}", f"{self.p.origin}/path"):
                st, _, _ = self.req(method, "/review/app/api/edits?f=a", {"Origin": origin}, b"{}")
                self.assertEqual(st, 403, (method, origin))
            st, _, _ = self.req(method, "/review/app/api/edits?f=a", None, b"{}", origin=False)   # no Origin at all
            self.assertEqual(st, 403, method)
        self.assertEqual(len(self.up.seen), n)                             # none reached the review server
        st, _, _ = self.req("POST", "/review/app/api/claim?c=x&who=t", {"Origin": self.p.origin}, b"")  # control
        self.assertEqual(st, 200)

    def test_traversal_rejected(self):
        n = len(self.up.seen)
        targets = [b"/review/app/../history/history.html", b"/review/app/%2e%2e/x", b"/review/app/%2E%2E/x", b"/review/app/..%2fx",
                   b"/review/app/..%2Fhistory", b"/review/app/%2e%2e%2fx", b"/review/app/%252e%252e/x", b"/review/app/a/../../x",
                   b"/review/app//x", b"/review/app/a//b", b"/review/app/%2f/x", b"/review/app/%2F%2Fx", b"/review/app/./x",
                   b"/review/app/..\\x", b"/review/app/%5c..%5cx", b"/review/app/a%00b", b"/review/app/a\x00b", b"/review/app/a%0d%0ab",
                   b"/review/app/%25%32%65%25%32%65/x"]
        for t in targets:
            resp = T.raw(self.p.port, b"GET " + t + b" HTTP/1.1\r\nHost: 127.0.0.1\r\nCookie: __Host-cj_session="
                         + self.token.encode() + b"\r\nConnection: close\r\n\r\n")
            self.assertIn(T.raw_status(resp), (400, 404), t)
        self.assertEqual(len(self.up.seen), n, [r["path"] for r in self.up.seen[n:]])
        st, _, _ = self.req("GET", "/review/app/images/ok_name.png")           # control: a normal path passes
        self.assertEqual((st, self.up.last()["path"]), (200, "/images/ok_name.png"))

    def test_clean_subpath_unit(self):
        for ok in ("/", "/api/edits", "/images/a.png", "/images/a%20b.png", "/x/"):
            self.assertEqual(P.clean_subpath(ok), ok)
        for bad in ("", "x", "/..", "/a/..", "/.", "//", "/a//b", "/%2e%2e", "/%252e%252e", "/a\\b", "/a%5cb",
                    "/a\x00", "/a%00", "/%ff%fe"):
            with self.assertRaises((P.BadPath, UnicodeDecodeError), msg=repr(bad)):
                P.clean_subpath(bad)

    def test_redirects_relative(self):
        st, h, _ = self.req("GET", "/review")
        self.assertEqual((st, T.hget(h, "location")), (307, "/review/"))
        st, h, _ = self.req("GET", "/review/app/redir-abs")
        self.assertEqual((st, T.hget(h, "location")), (302, "/review/app/x"))
        st, h, _ = self.req("GET", "/review/app/redir-rel")
        self.assertEqual((st, T.hget(h, "location")), (302, "/review/app/y"))

    def test_body_cap_413(self):
        self.assertEqual(P.MAX_BODY, 25 * 2**20)
        n = len(self.up.seen)
        head = (b"PUT /review/app/api/edits?f=a HTTP/1.1\r\nHost: 127.0.0.1:" + str(self.p.port).encode()
                + b"\r\nOrigin: " + self.p.origin.encode() + b"\r\nCookie: __Host-cj_session=" + self.token.encode()
                + b"\r\nContent-Type: application/json\r\nConnection: close\r\n")
        # declared too large: refused before any body byte is read (none sent, so no reset races the reply)
        resp = T.raw(self.p.port, head + b"Content-Length: " + str(P.MAX_BODY + 1).encode() + b"\r\n\r\n")
        self.assertEqual(T.raw_status(resp), 413)
        # chunked: read up to the cap; exactly cap + 1 bytes are sent, so the portal has read them all when it refuses
        chunk = b"y" * (1 << 20)
        chunked = b"".join(b"%x\r\n" % len(chunk) + chunk + b"\r\n" for _ in range(25)) + b"1\r\ny\r\n0\r\n\r\n"
        resp = T.raw(self.p.port, head + b"Transfer-Encoding: chunked\r\n\r\n" + chunked, timeout=60)
        self.assertEqual(T.raw_status(resp), 413)
        self.assertEqual(len(self.up.seen), n)
        small = b"z" * 4096                                                 # control: a small chunked body passes
        resp = T.raw(self.p.port, head + b"Transfer-Encoding: chunked\r\n\r\n" + b"%x\r\n" % len(small) + small
                     + b"\r\n0\r\n\r\n")
        self.assertEqual(T.raw_status(resp), 200)
        self.assertEqual(self.up.last()["body_len"], 4096)


class ProxyDown(unittest.TestCase):
    def test_upstream_down_502(self):
        env = T.Env()
        p = T.Portal(env.config(review_upstream=f"http://127.0.0.1:{T.free_port()}"))
        try:
            token = T.login(p.port, env.key)
            st, h, body = T.request(p.port, "GET", "/review/app/", T.cookie(token))
            self.assertEqual(st, 502)
            self.assertIn(b"not running", body)
            self.assertEqual(T.hget(h, "x-frame-options"), "SAMEORIGIN")   # /review/app/* may be framed by the shell
        finally:
            p.stop()
            env.cleanup()

    def test_upstream_timeout_502(self):
        env = T.Env()
        up = T.Upstream()
        p = T.Portal(env.config(review_upstream=up.url, review_timeout=0.5))
        try:
            token = T.login(p.port, env.key)
            st, _, body = T.request(p.port, "GET", "/review/app/slow", T.cookie(token))
            self.assertEqual(st, 502)
            self.assertIn(b"did not answer", body)
            self.assertEqual(APP.Config(None, env.roster, env.state).review_timeout, 30.0)
        finally:
            p.stop()
            up.stop()
            env.cleanup()


class ProxyRealReviewServer(unittest.TestCase):
    """The control with known answers: annotation/review_server.py's own handler on a temp fixture package. Direct
    without a key it answers 403; through the portal (key from the package's .review_key) it answers 200, also when
    the browser still carries a stale ?k=, which would outrank the header if it were forwarded."""

    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.rv = T.ReviewUpstream()
        cls.key = APP.read_review_key(cls.rv.pkg)
        cls.p = T.Portal(cls.env.config(review_upstream=cls.rv.url, review_key=cls.key))
        cls.token = T.login(cls.p.port, cls.env.key)

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.rv.stop()
        cls.env.cleanup()

    def test_control_direct_without_key_403(self):
        self.assertEqual(self.key, self.rv.key)
        self.assertEqual(T.request(self.rv.port, "GET", "/api/ping")[0], 403)
        self.assertEqual(T.request(self.rv.port, "GET", "/api/ping?k=wrong", {"X-Review-Key": self.key})[0], 403)

    def test_through_portal(self):
        c = T.cookie(self.token)
        st, h, body = T.request(self.p.port, "GET", "/review/app/api/ping", c)
        self.assertEqual((st, json.loads(body)), (200, {"ok": True}))
        st, h, body = T.request(self.p.port, "GET", "/review/app/?k=stale-wrong-key", c)
        self.assertEqual(st, 200)
        self.assertIn(b"<html", body.lower())
        self.assertIsNone(T.hget(h, "set-cookie"))
        st, _, body = T.request(self.p.port, "GET", "/review/app/data.js", c)
        self.assertEqual(st, 200)
        img = self.rv.files[0]
        st, h, body = T.request(self.p.port, "GET", f"/review/app/images/{img}", c)
        self.assertEqual((st, body), (200, (self.rv.pkg / "images" / img).read_bytes()))

    def test_put_edit_through_portal(self):
        f = self.rv.files[0]
        edit = {"t": 5, "dec": {"g1": {"v": "del", "t": 5, "who": "tester"}}, "reviewed": True, "reviewedT": 5}
        st, _, body = T.request(self.p.port, "PUT", f"/review/app/api/edits?f={f}",
                                {**T.cookie(self.token), "Origin": self.p.origin, "Content-Type": "application/json"},
                                json.dumps(edit).encode())
        self.assertEqual(st, 200, body)
        saved = json.loads((self.rv.pkg / "edits" / f"{f}.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["edit"]["dec"]["g1"]["v"], "del")


if __name__ == "__main__":
    unittest.main()
