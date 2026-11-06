"""
Auth tests for the CryoEV portal (portal/auth.py + the Gate in portal/app.py), against the app under a real uvicorn.
Temp roster and state dir only.

  python -m unittest discover -s portal/tests -v
"""
import contextlib
import io
import json
import re
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import app as APP  # noqa: E402
from portal import auth as A  # noqa: E402

SPEC_OPEN = {"/login", "/cj/cj.css", "/cj/brand.json", "/healthz"}
SEC = {"x-content-type-options": "nosniff", "x-frame-options": "DENY", "referrer-policy": "same-origin"}


class AuthLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.up = T.Upstream()
        cls.env.write_live(T.live_doc())
        (cls.env.registry / "history.html").write_text("<html><head></head><body></body></html>", encoding="utf-8")
        cls.p = T.Portal(cls.env.config(review_upstream=cls.up.url))

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.up.stop()
        cls.env.cleanup()

    def get(self, path, token=None, method="GET", headers=None):
        h = dict(headers or {})
        if token:
            h.update(T.cookie(token))
        return T.request(self.p.port, method, path, h)

    # -- positive control first: if this fails, every "denied" below means nothing
    def test_positive_control(self):
        st, h, _ = self.get(f"/login?k={self.env.key}")
        self.assertEqual(st, 303)
        self.assertEqual(T.hget(h, "location"), "/")
        token = T.hget(h, "set-cookie").split(";", 1)[0].split("=", 1)[1]
        st, _, body = self.get("/", token)
        self.assertEqual(st, 200)
        self.assertIn(b"signed in as tester", body)
        st, _, body = self.get("/cj/me.json", token)
        self.assertEqual((st, json.loads(body)), (200, {"id": "tester", "role": "admin"}))

    def test_no_session_401(self):
        n = len(self.up.seen)
        for path in ("/", "/cj/me.json", "/cj/header.js", "/cj/modules.json", "/history/history.html",
                     "/history/live/live.json", "/review/", "/review/api/ping", "/nope"):
            st, _, body = self.get(path)
            self.assertEqual(st, 401, path)
            self.assertIn(b"key link", body)
        self.assertEqual(len(self.up.seen), n)                       # nothing reached the review server

    def test_wrong_key_403(self):
        st, h, _ = self.get("/login?k=not-a-real-key")
        self.assertEqual(st, 403)
        self.assertIsNone(T.hget(h, "set-cookie"))
        st, _, _ = self.get("/login?k=" + "x" * 600)                   # absurd length: refused, not hashed
        self.assertEqual(st, 403)

    def test_login_without_key(self):
        self.assertEqual(self.get("/login")[0], 401)
        token = T.login(self.p.port, self.env.key)
        st, h, _ = self.get("/login", token)                          # already signed in: back home
        self.assertEqual((st, T.hget(h, "location")), (303, "/"))

    def test_unknown_session_403(self):
        self.assertEqual(self.get("/", "made-up-session-id")[0], 403)

    def test_revoked_and_rekeyed_403(self):
        key = A.add_key(self.env.roster, "temp", "admin")
        token = T.login(self.p.port, key)
        self.assertEqual(self.get("/", token)[0], 200)                # control
        time.sleep(0.02)                                              # a new mtime for the roster
        A.revoke(self.env.roster, "temp")
        self.assertEqual(self.get("/", token)[0], 403)
        self.assertEqual(self.get(f"/login?k={key}")[0], 403)
        key2 = A.add_key(self.env.roster, "temp", "admin")            # re-keyed: the new key works ...
        token2 = T.login(self.p.port, key2)
        self.assertEqual(self.get("/", token2)[0], 200)
        key3 = A.add_key(self.env.roster, "temp", "admin")            # ... and re-keying again ends its session
        self.assertEqual(self.get("/", token2)[0], 403)
        self.assertEqual(self.get(f"/login?k={key2}")[0], 403)
        self.assertEqual(T.request(self.p.port, "GET", f"/login?k={key3}")[0], 303)

    def test_removed_from_roster_403(self):
        key = A.add_key(self.env.roster, "gone", "admin")
        token = T.login(self.p.port, key)
        doc = json.loads(self.env.roster.read_text(encoding="utf-8"))
        doc["people"] = [p for p in doc["people"] if p["id"] != "gone"]
        time.sleep(0.02)
        A.write_json_atomic(self.env.roster, doc)
        self.assertEqual(self.get("/", token)[0], 403)

    def test_cookie_flags(self):
        st, h, _ = self.get(f"/login?k={self.env.key}")
        sc = T.hall(h, "set-cookie")
        self.assertEqual(len(sc), 1)
        parts = [x.strip() for x in sc[0].split(";")]
        name, token = parts[0].split("=", 1)
        attrs = {x.split("=", 1)[0].lower(): (x.split("=", 1) + [""])[1] for x in parts[1:]}
        self.assertEqual(name, "__Host-cj_session")
        self.assertGreaterEqual(len(token), 43)                       # 32 random bytes, base64url
        self.assertEqual(attrs.get("path"), "/")
        self.assertEqual(attrs.get("max-age"), str(30 * 86400))
        self.assertEqual(attrs.get("samesite"), "Lax")
        self.assertIn("httponly", attrs)
        self.assertIn("secure", attrs)
        self.assertNotIn("domain", attrs)                             # __Host- forbids Domain
        self.assertEqual(T.hget(h, "cache-control"), "no-store")

    def test_only_hashes_stored(self):
        token = T.login(self.p.port, self.env.key)
        roster = self.env.roster.read_text(encoding="utf-8")
        self.assertNotIn(self.env.key, roster)
        self.assertIn(A.sha256_hex(self.env.key), roster)
        sessions = (self.env.state / "sessions.json").read_text(encoding="utf-8")
        self.assertNotIn(token, sessions)
        self.assertIn(A.sha256_hex(token), sessions)
        # sessions survive a restart: a fresh Auth on the same files knows the token
        self.assertEqual(A.Auth(self.env.roster, self.env.state).check(token)[0], "ok")

    def test_expired_session_403(self):
        auth = A.Auth(self.env.roster, self.env.state)
        person = auth.roster.people()[T.USER]
        token = auth.sessions.create(person, now=time.time() - A.SESSION_S - 10)
        self.assertEqual(auth.check(token), ("denied", None))
        self.assertEqual(self.get("/", token)[0], 403)

    def test_open_list_from_router(self):
        """Every route the router has, each method: 401 without a session unless in the spec's open list."""
        self.assertEqual(set(APP.OPEN), SPEC_OPEN)
        paths = set()
        for r in self.p.app.routes:
            path = re.sub(r"\{[^}]+\}", "x", r.path)
            if r.path == "/login":                                    # its own 401 means "no key": give one
                path += f"?k={self.env.key}"
            paths.add(r.path)
            for m in sorted(r.methods or {"GET"}):
                st, _, _ = self.get(path, method=m, headers={"Origin": self.p.origin} if m in APP.UNSAFE else None)
                if r.path in SPEC_OPEN:
                    self.assertNotIn(st, (401, 403), (m, path))
                else:
                    self.assertEqual(st, 401, (m, path))
        self.assertTrue(SPEC_OPEN <= paths, "every open path is a real route")
        for path in ("/x", "/cj/", "/cj/other.js", "/static/cj.css", "/history", "/review", "/login/", "/healthz/"):
            self.assertEqual(self.get(path)[0], 401 if path not in SPEC_OPEN else 200, path)

    def test_security_headers_everywhere(self):
        token = T.login(self.p.port, self.env.key)
        cases = [("/healthz", None, 200), ("/", None, 401), ("/", "bad", 403), ("/nope", token, 404),
                 (f"/login?k={self.env.key}", None, 303), ("/cj/cj.css", None, 200), ("/review", token, 307),
                 ("/review/app/x", token, 200)]
        for path, tok, want in cases:
            st, h, _ = self.get(path, tok)
            self.assertEqual(st, want, path)
            for k, v in SEC.items():
                if k == "x-frame-options" and path.startswith("/review/"):
                    v = "SAMEORIGIN"        # the shell may frame the review tool (same origin only); DENY elsewhere
                self.assertEqual(T.hall(h, k), [v], (path, k))

    def test_relative_redirects(self):
        token = T.login(self.p.port, self.env.key)
        for path, want in (("/history", "/history/"), ("/history/", "history.html"), ("/review", "/review/"),
                           (f"/login?k={self.env.key}", "/")):
            st, h, _ = self.get(path, token)
            self.assertIn(st, (303, 307), path)
            loc = T.hget(h, "location")
            self.assertEqual(loc, want, path)
            self.assertNotIn("://", loc)
            self.assertFalse(loc.startswith("//"))
        # no slash redirects: Starlette's redirect_slashes is off
        self.assertEqual(self.get("/healthz/", token)[0], 404)

    def test_no_websockets(self):
        token = T.login(self.p.port, self.env.key)
        st, _, _ = self.get("/review/", token, headers={"Connection": "Upgrade", "Upgrade": "websocket",
                                                        "Sec-WebSocket-Version": "13",
                                                        "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="})
        self.assertNotEqual(st, 101)


class AuthTool(unittest.TestCase):
    def setUp(self):
        self.env = T.Env()

    def tearDown(self):
        self.env.cleanup()

    def test_add_key_prints_link_once(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = A.main(["add-key", "--id", "alice", "--role", "admin", "--roster", str(self.env.roster),
                         "--base", "https://portal.example:8446"])
        self.assertEqual(rc, 0)
        links = re.findall(r"https://portal\.example:8446/login\?k=(\S+)", out.getvalue())
        self.assertEqual(len(links), 1)
        key = links[0]
        roster = json.loads(self.env.roster.read_text(encoding="utf-8"))
        alice = [p for p in roster["people"] if p["id"] == "alice"]
        self.assertEqual(len(alice), 1)
        self.assertEqual(alice[0]["key_sha256"], A.sha256_hex(key))
        self.assertNotIn(key, self.env.roster.read_text(encoding="utf-8"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            A.main(["list", "--roster", str(self.env.roster)])
        self.assertIn("alice\tadmin\tactive", out.getvalue())
        self.assertNotIn(key, out.getvalue())
        self.assertNotIn(alice[0]["key_sha256"], out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(A.main(["revoke", "--id", "alice", "--roster", str(self.env.roster)]), 0)
        self.assertIsNone(A.Roster(self.env.roster).match_key(key))

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            A.add_key(self.env.roster, "Bad Id!", "admin")
        with self.assertRaises(ValueError):
            A.add_key(self.env.roster, "bob", "superuser")

    def test_missing_roster_denies(self):
        auth = A.Auth(self.env.tmp / "no_roster.json", self.env.state)
        self.assertIsNone(auth.login(self.env.key))
        broken = self.env.tmp / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        self.assertIsNone(A.Auth(broken, self.env.state).login(self.env.key))

    def test_defaults_outside_repo(self):
        self.assertEqual(A.default_roster(), Path.home() / ".cryoev" / "portal_roster.json")
        self.assertFalse(str(A.default_roster()).startswith(str(T.REPO)))
        self.assertFalse(str(A.default_state_dir()).startswith(str(T.REPO)))


class WriteJsonAtomic(unittest.TestCase):
    def setUp(self):
        import os
        import tempfile
        from unittest import mock
        self.os, self.mock = os, mock
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.target = self.dir / "x.json"
        self.real = os.replace

    def tearDown(self):
        self.tmp.cleanup()

    def test_normal_path(self):
        A.write_json_atomic(self.target, {"a": 1})
        self.assertEqual(json.loads(self.target.read_text(encoding="utf-8")), {"a": 1})
        self.assertEqual([p.name for p in self.dir.iterdir()], ["x.json"])

    def test_retries_then_succeeds(self):
        calls = []

        def flaky(src, dst):
            calls.append(1)
            if len(calls) <= 2:
                raise PermissionError("in use")
            return self.real(src, dst)
        with self.mock.patch.object(A.os, "replace", flaky):
            A.write_json_atomic(self.target, {"a": 2})
        self.assertEqual(len(calls), 3)
        self.assertEqual(json.loads(self.target.read_text(encoding="utf-8")), {"a": 2})
        self.assertEqual([p.name for p in self.dir.iterdir()], ["x.json"])

    def test_always_fails_raises_cleanly_and_bounded(self):
        calls = []

        def never(src, dst):
            calls.append(1)
            raise PermissionError("in use")
        t0 = time.monotonic()
        with self.mock.patch.object(A.os, "replace", never):
            with self.assertRaises(PermissionError):
                A.write_json_atomic(self.target, {"a": 3})
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(len(calls), A.REPLACE_TRIES)
        self.assertEqual(list(self.dir.iterdir()), [], "no temp file left, target not created")


if __name__ == "__main__":
    unittest.main()
