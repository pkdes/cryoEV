"""
Cloudflare access mode (portal/access.py + the Gate's access branch in portal/app.py): only a verified
Cf-Access-Jwt-Assertion JWT signs anyone in, mapped by its email to one active roster person.

A local RSA key signs the JWTs; its JWKS is served by a fake fetcher put into the app's KeySet (no network). The
positive control (a good JWT -> 200 as that user, with that role) must pass, or every refusal below means nothing.
Placeholders only (example-team, portal.example.org): the repo is public.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import io
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cjtest as T  # noqa: E402

import jwt  # noqa: E402
import uvicorn  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402
from jwt.algorithms import RSAAlgorithm  # noqa: E402

from portal import access as AX  # noqa: E402
from portal import app as APP  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import admin as ADM  # noqa: E402

TEAM = "example-team"
AUD = "test-aud-0123456789abcdef"
HOST = "portal.example.org"
ISS = f"https://{TEAM}.cloudflareaccess.com"
ORIGIN = f"https://{HOST}"
ACFG = AX.AccessConfig(team=TEAM, aud=AUD, host=HOST)
HDR = "Cf-Access-Jwt-Assertion"


def new_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


KEY1 = new_key()
KEY2 = new_key()          # a second team key (rotation) / an attacker's key


def jwk(priv, kid):
    d = RSAAlgorithm.to_jwk(priv.public_key(), as_dict=True)
    d.update({"kid": kid, "alg": "RS256", "use": "sig"})
    return d


def jwks(*pairs):
    return {"keys": [jwk(k, kid) for k, kid in pairs], "public_cert": {"kid": "x", "cert": "not used"}}


def claims(email="alice@example.org", **kw):
    now = int(time.time())
    c = {"aud": [AUD], "iss": ISS, "email": email, "exp": now + 300, "iat": now, "nbf": now, "sub": "s", "type": "app"}
    c.update(kw)
    return {k: v for k, v in c.items() if v is not None}


def sign(email="alice@example.org", key=KEY1, kid="k1", **kw):
    return jwt.encode(claims(email, **kw), key, algorithm="RS256", headers={"kid": kid})


def b64(obj) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj, separators=(",", ":")).encode()).rstrip(b"=").decode()


def forged(alg, secret=b""):
    """A hand-made token with header alg `alg`: 'none' -> empty signature; 'HS256' -> HMAC with `secret`."""
    head = b64({"alg": alg, "typ": "JWT", "kid": "k1"})
    body = b64(claims())
    sig = b""
    if alg == "HS256":
        sig = hmac.new(secret, f"{head}.{body}".encode(), hashlib.sha256).digest()
    return f"{head}.{body}." + base64.urlsafe_b64encode(sig).rstrip(b"=").decode()


class Fetcher:
    """Stands in for portal.access.fetch_certs: counts calls; serves `doc`, or raises `fail`."""

    def __init__(self, doc=None, fail=None):
        self.doc = doc if doc is not None else jwks((KEY1, "k1"))
        self.fail = fail
        self.calls = 0

    def __call__(self, url, timeout):
        self.calls += 1
        assert url == ACFG.certs_url, url
        assert timeout and timeout > 0
        if self.fail:
            raise self.fail
        return self.doc


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def roster_people(env):
    """tester (admin, from cjtest), + alice basic, adv advanced, boss admin, gone basic (revoked); emails set."""
    ADM.add_person(env.roster, "alice", "basic", "Alice@Example.org")
    ADM.add_person(env.roster, "adv", "advanced", "adv@example.org")
    ADM.add_person(env.roster, "boss", "admin", "boss@example.org")
    ADM.add_person(env.roster, "kate", "basic", "kate@example.org")
    ADM.add_person(env.roster, "gone", "basic", "gone@example.org")
    ADM.revoke_person(env.roster, "gone")
    A.set_email(env.roster, "tester", "tester@example.org")


def access_portal(env, fetcher, admin=False, **kw):
    p = T.Portal(env.config(auth="access", access=ACFG, access_admin=admin, **kw))
    p.app.verifier.keys = AX.KeySet(ACFG.certs_url, fetch=fetcher)
    return p


def get(port, path, token=None, headers=None, method="GET", body=None):
    h = dict(headers or {})
    if token is not None:
        h[HDR] = token
    return T.request(port, method, path, h, body)


class AccessGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        roster_people(cls.env)
        cls.fetch = Fetcher()
        cls.p = access_portal(cls.env, cls.fetch)

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()

    def me(self, token, **kw):
        return get(self.p.port, "/cj/me.json", token, **kw)

    # ---- positive control: must pass, or the refusals below prove nothing
    def test_good_jwt_is_the_matching_user_with_their_role(self):
        st, _, body = self.me(sign())
        self.assertEqual(st, 200, body)
        self.assertEqual(json.loads(body), {"id": "alice", "role": "basic", "signout": "/cdn-cgi/access/logout"})
        st, _, body = self.me(sign("ADV@example.ORG"))                  # email case-insensitive
        self.assertEqual((st, json.loads(body)["id"], json.loads(body)["role"]), (200, "adv", "advanced"))
        st, _, body = get(self.p.port, "/", sign())
        self.assertEqual(st, 200)
        st, _, _ = self.me(sign(exp=int(time.time()) - 30))            # inside the 60 s leeway
        self.assertEqual(st, 200)

    # ---- refused (403)
    def assert_refused(self, token, headers=None, text=b"ask the admin to add your email"):
        st, _, body = self.me(token, headers=headers)
        self.assertEqual(st, 403, body)
        self.assertIn(text.lower(), body.replace(b"&#x27;", b"'").replace(b"\n", b" ").lower())
        self.assertNotIn(b"key link", body)
        return body

    def test_missing_header(self):
        self.assert_refused(None)
        self.assert_refused("")

    def test_bad_signature_other_key(self):
        self.assert_refused(sign(key=KEY2, kid="k1"))

    def test_alg_none(self):
        self.assert_refused(forged("none"))

    def test_hs256_with_the_public_key_as_secret(self):
        pem = KEY1.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        self.assert_refused(forged("HS256", pem))
        jwk_secret = json.dumps(jwk(KEY1, "k1")).encode()
        self.assert_refused(forged("HS256", jwk_secret))

    def test_wrong_aud(self):
        self.assert_refused(sign(aud=["some-other-application-aud"]))
        self.assert_refused(sign(aud=None))

    def test_wrong_iss(self):
        self.assert_refused(sign(iss="https://other-team.cloudflareaccess.com"))
        self.assert_refused(sign(iss=None))

    def test_expired_and_not_yet_valid(self):
        now = int(time.time())
        self.assert_refused(sign(exp=now - 120, iat=now - 900, nbf=now - 900))
        self.assert_refused(sign(nbf=now + 600))
        self.assert_refused(sign(exp=None))

    def test_unknown_email(self):
        body = self.assert_refused(sign("stranger@example.org"))
        self.assertIn(b"stranger@example.org", body)
        self.assert_refused(sign(email=None), text=b"ask the admin")

    def test_kelvin_sign_email_does_not_pose_as_kate(self):
        st, _, body = self.me(sign("kate@example.org"))                 # control: kate herself gets in
        self.assertEqual((st, json.loads(body)["id"]), (200, "kate"))
        self.assertEqual("Kate@example.org".lower(), "kate@example.org")   # why: str.lower() folds it
        for e in ("Kate@example.org", "KATE@example.org", "kate@examplе.org", "katé@example.org"):
            st, _, body = self.me(sign(e))
            self.assertEqual(st, 403, e)
            self.assertNotIn(b'"id"', body)
        self.assertIsNone(A.Roster(self.env.roster).match_email("Kate@example.org"))

    def test_revoked_user_with_valid_jwt(self):
        self.assert_refused(sign("gone@example.org"))

    def test_admin_refused_when_access_admin_off(self):
        st, _, body = self.me(sign("boss@example.org"))
        self.assertEqual(st, 403)
        self.assertIn(b"Admin is available on Tailscale only", body)
        st, _, _ = get(self.p.port, "/admin/", sign("boss@example.org"))
        self.assertEqual(st, 403)

    def test_jwt_only_as_cookie_is_refused(self):
        tok = sign()
        self.assert_refused(None, headers={"Cookie": f"CF_Authorization={tok}"})

    def test_key_session_cookie_is_ignored(self):
        sess, _ = A.Auth(self.env.roster, self.env.state).login(self.env.key)    # a valid key-mode session
        self.assert_refused(None, headers={"Cookie": f"{A.COOKIE}={sess}"})
        st, _, body = self.me(sign(), headers={"Cookie": f"{A.COOKIE}={sess}"})  # JWT decides, not the cookie
        self.assertEqual((st, json.loads(body)["id"]), (200, "alice"))

    # ---- the rest of the mode
    def test_login_is_404(self):
        st, h, body = get(self.p.port, f"/login?k={self.env.key}", sign())
        self.assertEqual(st, 404)
        self.assertIsNone(T.hget(h, "set-cookie"))
        self.assertEqual(get(self.p.port, f"/login?k={self.env.key}")[0], 404)

    def test_cross_origin_post_refused_same_public_origin_accepted(self):
        tok = sign()
        for bad in (None, "https://evil.example.com", self.p.origin, f"http://{HOST}", "null"):
            hdr = {"Origin": bad} if bad else {}
            st, _, body = get(self.p.port, "/cj/me.json", tok, hdr, method="POST", body=b"")
            self.assertEqual(st, 403, bad)
            self.assertIn(b"Cross-origin request refused", body)
        st, _, _ = get(self.p.port, "/cj/me.json", tok, {"Origin": ORIGIN}, method="POST", body=b"")
        self.assertEqual(st, 405)                                         # past the Gate: only the method is wrong

    def test_sandbox_hidden_and_sign_out(self):
        st, _, body = get(self.p.port, "/cj/modules.json", sign("adv@example.org"))
        mods = json.loads(body)["modules"]
        self.assertEqual(st, 200)
        self.assertFalse([m for m in mods if m.get("external")], mods)
        self.assertTrue(any(m["id"] == "data" for m in mods))            # control: the list itself is there
        js = get(self.p.port, "/cj/header.js", sign())[2]
        self.assertIn(b"/cdn-cgi/access/logout", js)
        self.assertIn(b"Sign out", js)

    def test_refusals_logged_with_reason_and_ip_never_the_token(self):
        records = []

        class Grab(logging.Handler):
            def emit(self, rec):
                records.append(rec.getMessage())
        h = Grab()
        lg = logging.getLogger("cj")
        old = lg.level
        lg.setLevel(logging.INFO)
        lg.addHandler(h)
        try:
            bad = sign(aud=["nope-nope-nope"])
            good = sign()
            self.me(bad, headers={"Cf-Connecting-IP": "203.0.113.7"})
            self.me(sign("stranger@example.org"), headers={"Cf-Connecting-IP": "203.0.113.8 <b>x</b>"})
            self.me(good, headers={"Cf-Connecting-IP": "2001:db8::1"})
            time.sleep(0.1)
        finally:
            lg.removeHandler(h)
            lg.setLevel(old)
        text = "\n".join(records)
        self.assertIn("wrong aud", text)
        self.assertIn("203.0.113.7", text)
        self.assertIn("stranger@example.org", text)
        self.assertIn("cf=2001:db8::1", text)
        self.assertNotIn("<b>x</b>", text)                                # a header that isn't an IP: '-'                            # the access log line
        self.assertNotIn(bad, text)
        self.assertNotIn(good, text)
        self.assertNotIn(bad.split(".")[2], text)


class AccessAdminOn(unittest.TestCase):
    """CJ_ACCESS_ADMIN on: admins get in; the Admin form token and messages are bound to the verified user."""

    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        roster_people(cls.env)
        cls.up = T.Upstream()
        cls.p = access_portal(cls.env, Fetcher(), admin=True, review_upstream=cls.up.url)

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.up.stop()
        cls.env.cleanup()

    def token_of(self, email):
        st, _, body = get(self.p.port, "/admin/", sign(email))
        self.assertEqual(st, 200, body)
        return re.search(rb'name="t" value="([^"]+)"', body).group(1).decode(), body

    def post(self, email, path, fields, origin=ORIGIN):
        body = "&".join(f"{k}={v}" for k, v in fields.items()).encode()
        return get(self.p.port, path, sign(email), {"Origin": origin,
                                                     "Content-Type": "application/x-www-form-urlencoded"},
                   method="POST", body=body)

    def test_admin_allowed_when_on(self):
        st, _, body = get(self.p.port, "/cj/me.json", sign("boss@example.org"))
        self.assertEqual((st, json.loads(body)["role"]), (200, "admin"))

    def test_admin_page_email_list_and_form(self):
        _, body = self.token_of("boss@example.org")
        self.assertIn(b"alice@example.org", body)                       # shown in the list (lower-cased)
        self.assertIn(b'name="email"', body)
        self.assertIn(b"/admin/people/email", body)

    def test_csrf_token_differs_per_verified_user(self):
        t_boss, _ = self.token_of("boss@example.org")
        t_tester, _ = self.token_of("tester@example.org")
        self.assertNotEqual(t_boss, t_tester)
        self.assertEqual(t_boss, self.token_of("boss@example.org")[0])  # stable for one user (no cookie needed)
        st, _, body = self.post("tester@example.org", "/admin/people/email",
                                {"t": t_boss, "id": "adv", "email": "x@example.org"})
        self.assertEqual(st, 403)                                        # boss's token is not tester's
        self.assertIn(b"expired", body)
        self.assertEqual(A.Roster(self.env.roster).people()["adv"]["email"], "adv@example.org")

    def test_set_email_and_flash_bound_to_user(self):
        t, _ = self.token_of("boss@example.org")
        st, h, _ = self.post("boss@example.org", "/admin/people/email",
                             {"t": t, "id": "adv", "email": "Adv.New%40Example.org"})
        self.assertEqual(st, 303)
        self.assertEqual(A.Roster(self.env.roster).people()["adv"]["email"], "adv.new@example.org")
        loc = T.hget(h, "location")
        st, _, other = get(self.p.port, loc, sign("tester@example.org"))   # someone else: no message
        self.assertNotIn(b"signs in through Cloudflare", other)
        st, _, mine = get(self.p.port, loc, sign("boss@example.org"))
        self.assertIn(b"adv now signs in through Cloudflare as adv.new@example.org", mine)
        st, _, _ = get(self.p.port, "/cj/me.json", sign("adv.new@example.org"))
        self.assertEqual(st, 200)
        self.assertEqual(get(self.p.port, "/cj/me.json", sign("adv@example.org"))[0], 403)   # old email: out
        t, _ = self.token_of("boss@example.org")
        st, _, body = self.post("boss@example.org", "/admin/people/email",
                                {"t": t, "id": "adv", "email": "alice%40example.org"})
        self.assertEqual(st, 409)
        self.assertIn(b"already the email of alice", body)

    def test_add_person_with_email_same_origin_only(self):
        t, _ = self.token_of("boss@example.org")
        st, _, _ = self.post("boss@example.org", "/admin/people/add",
                             {"t": t, "id": "carol", "role": "basic", "email": "carol%40example.org"},
                             origin=self.p.origin)                       # Host-matching origin: not the public one
        self.assertEqual(st, 403)
        st, _, body = self.post("boss@example.org", "/admin/people/add",
                                {"t": t, "id": "carol", "role": "basic", "email": "carol%40example.org"})
        self.assertEqual(st, 200, body)
        self.assertNotIn(f"{ORIGIN}/login".encode(), body)               # no key link to the 404 address
        self.assertEqual(A.Roster(self.env.roster).people()["carol"]["email"], "carol@example.org")
        self.assertEqual(json.loads(get(self.p.port, "/cj/me.json", sign("carol@example.org"))[2])["id"], "carol")

    def test_client_x_cj_user_never_reaches_upstream(self):
        tok = sign("boss@example.org")
        st, _, body = get(self.p.port, "/review/app/echo", tok, {"X-CJ-User": "mallory", "X-CJ-Role": "admin",
                                                                  "Cf-Connecting-IP": "203.0.113.9"})
        self.assertEqual(st, 200, body)
        seen = {k.lower(): v for k, v in self.up.last()["headers"]}
        self.assertEqual(seen.get("x-cj-user"), "boss")
        self.assertNotIn("mallory", json.dumps(self.up.last()))
        self.assertNotIn("x-cj-role", seen)
        self.assertNotIn("cf-access-jwt-assertion", seen)               # the token never goes upstream
        self.assertNotIn("cf-connecting-ip", seen)


class KeyModeIgnoresJwt(unittest.TestCase):
    def test_jwt_on_key_mode_app_is_ignored(self):
        env = T.Env()
        roster_people(env)
        p = T.Portal(env.config())
        try:
            st, _, body = get(p.port, "/cj/me.json", sign("tester@example.org"))
            self.assertEqual(st, 401)
            self.assertIn(b"key link", body)
            st, _, body = get(p.port, "/cj/me.json", None, T.cookie(T.login(p.port, env.key)))   # control
            self.assertEqual((st, json.loads(body)), (200, {"id": "tester", "role": "admin"}))  # no signout
            mods = json.loads(get(p.port, "/cj/modules.json", None, T.cookie(T.login(p.port, env.key)))[2])
            self.assertTrue(any(m.get("external") for m in mods["modules"]))   # sandbox still linked in key mode
        finally:
            p.stop()
            env.cleanup()

    def test_key_mode_does_not_import_pyjwt(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + '/portal/tests');"
                "import cjtest as T; from portal import app as APP; e = T.Env(); APP.create_app(e.config());"
                "e.cleanup(); print('jwt' in sys.modules, 'portal.access' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code, str(T.REPO)], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.stdout.strip(), "False False", out.stderr)


class Keys(unittest.TestCase):
    """KeySet / Verifier without the app: fake clock + fetcher."""

    def verify(self, v, token):
        return asyncio.run(v.verify(token))

    def test_unknown_kid_refetches_once_then_refuses_within_60s(self):
        f, clock = Fetcher(jwks((KEY1, "k1"))), Clock()
        v = AX.Verifier(ACFG, AX.KeySet(ACFG.certs_url, fetch=f, clock=clock))
        self.assertIsNotNone(self.verify(v, sign())[0])                  # control: k1 works
        self.assertEqual(f.calls, 1)
        clock.t += 100
        c, why = self.verify(v, sign(key=KEY2, kid="k2"))
        self.assertEqual((c, why, f.calls), (None, "unknown kid", 2))      # refetched once, still unknown
        t_fetch = clock.t
        for dt in (1, 30, 59):
            clock.t = t_fetch + dt
            self.assertIsNone(self.verify(v, sign(key=KEY2, kid="k2"))[0])
            self.assertIsNone(self.verify(v, sign(key=KEY2, kid="k3"))[0])
        self.assertEqual(f.calls, 2)                                       # no fetch storm inside the minute
        self.assertIsNotNone(self.verify(v, sign())[0])                  # known kid: no fetch, still fine
        f.doc = jwks((KEY1, "k1"), (KEY2, "k2"))                         # Cloudflare rotates
        clock.t = t_fetch + 61
        self.assertIsNotNone(self.verify(v, sign(key=KEY2, kid="k2"))[0])
        self.assertEqual(f.calls, 3)

    def test_no_kid_is_logged_as_no_kid(self):
        f = Fetcher()
        v = AX.Verifier(ACFG, AX.KeySet(ACFG.certs_url, fetch=f, clock=Clock()))
        no_kid = jwt.encode(claims(), KEY1, algorithm="RS256")
        self.assertNotIn("kid", jwt.get_unverified_header(no_kid))
        self.assertEqual(self.verify(v, no_kid), (None, "no kid"))
        self.assertEqual(f.calls, 0)                                       # nothing fetched for it
        self.assertIsNotNone(self.verify(v, sign())[0])                  # control

    def test_keys_that_cannot_be_fetched_refuse_everyone(self):
        for fetcher in (Fetcher(fail=OSError("down")), Fetcher(fail=TimeoutError()), Fetcher(doc={"keys": []}),
                        Fetcher(doc={"keys": [{"kty": "EC", "kid": "k1"}]}), Fetcher(doc=["not", "a", "dict"]),
                        Fetcher(doc={"keys": [{"kty": "RSA", "kid": "k1", "n": "bad", "e": "AQAB"}]})):
            v = AX.Verifier(ACFG, AX.KeySet(ACFG.certs_url, fetch=fetcher, clock=Clock()))
            c, why = self.verify(v, sign())
            self.assertIsNone(c)
            self.assertIn("no signing keys", why)

    def test_failed_refresh_keeps_keys_until_max_age(self):
        f, clock = Fetcher(), Clock()
        v = AX.Verifier(ACFG, AX.KeySet(ACFG.certs_url, fetch=f, clock=clock))
        self.assertIsNotNone(self.verify(v, sign())[0])
        f.fail = OSError("down")
        clock.t += AX.KEYS_TTL_S + 1                                      # refresh due, fails: old keys still good
        self.assertIsNotNone(self.verify(v, sign())[0])
        clock.t += AX.KEYS_MAX_AGE_S                                      # nothing good for too long: refuse
        self.assertIsNone(self.verify(v, sign())[0])

    def test_slow_fetch_times_out(self):
        def slow(url, timeout):
            time.sleep(timeout + 1.5)                    # a fetcher that ignores its own timeout
            return jwks((KEY1, "k1"))
        ks = AX.KeySet(ACFG.certs_url, fetch=slow, timeout=0.2, clock=Clock())

        async def timed():
            t0 = time.monotonic()
            res = await AX.Verifier(ACFG, ks).verify(sign())
            return res, time.monotonic() - t0
        (c, why), took = asyncio.run(timed())
        self.assertLess(took, 1.5)                       # the request gave up at timeout + 1 s
        self.assertIsNone(c)
        self.assertIn("TimeoutError", why)

    def test_app_refuses_when_keys_cannot_be_fetched(self):
        env = T.Env()
        roster_people(env)
        p = access_portal(env, Fetcher(fail=OSError("down")))
        try:
            st, _, body = get(p.port, "/cj/me.json", sign())
            self.assertEqual(st, 403)
            self.assertIn(b"ask the admin", body)
        finally:
            p.stop()
            env.cleanup()


class RosterEmails(unittest.TestCase):
    def setUp(self):
        self.env = T.Env()
        roster_people(self.env)

    def tearDown(self):
        self.env.cleanup()

    def entry(self, pid):
        doc = json.loads(self.env.roster.read_text(encoding="utf-8"))
        return [p for p in doc["people"] if p["id"] == pid][-1]

    def test_duplicate_email_on_add_is_refused(self):
        for e in ("alice@example.org", "ALICE@example.org", " alice@example.org "):
            with self.assertRaises(ADM.Refused):
                ADM.add_person(self.env.roster, "bob", "basic", e)
            with self.assertRaises(ValueError):
                A.add_key(self.env.roster, "bob", "basic", e)
            with self.assertRaises(ADM.Refused):
                ADM.set_email(self.env.roster, "adv", e)
            with self.assertRaises(ValueError):
                A.set_email(self.env.roster, "adv", e)
        self.assertNotIn("bob", A.Roster(self.env.roster).people())
        ADM.add_person(self.env.roster, "bob", "basic", "gone@example.org")    # a revoked person's email is free
        with self.assertRaises(ADM.Refused):
            ADM.add_person(self.env.roster, "eve", "basic", "not an email")

    def test_non_ascii_email_refused_in_the_roster(self):
        for e in ("Kate@example.org", "katé@example.org", "kate@examplе.org"):
            with self.assertRaises(ADM.Refused):
                ADM.add_person(self.env.roster, "kate2", "basic", e)
            with self.assertRaises(ValueError):
                A.set_email(self.env.roster, "adv", e)
            with self.assertRaises(ValueError):
                A.add_key(self.env.roster, "kate3", "basic", e)
        self.assertEqual(A.norm_email(" Kate@Example.ORG "), "kate@example.org")    # control: ASCII still fine

    def test_last_admin_guard_sees_email_only_admins(self):
        doc = {"schema": "cj_roster/1", "people": [
            {"id": "solo", "role": "admin", "email": "solo@example.org", "added": "2026-11-06T10:00:00"},
            {"id": "pat", "role": "basic", "email": "pat@example.org", "added": "2026-11-06T10:00:00"}]}
        self.env.roster.write_text(json.dumps(doc), encoding="utf-8")
        self.assertFalse(A.Roster.usable(doc["people"][0]))              # no key: invisible to key mode's check
        for fn in (lambda: ADM.revoke_person(self.env.roster, "solo"),
                   lambda: ADM.set_role(self.env.roster, "solo", "basic")):
            with self.assertRaises(ADM.Refused) as cm:
                fn()
            self.assertIn("last admin", str(cm.exception))
        self.assertIn("last admin", ADM.revoke_check(self.env.roster, "solo"))
        self.assertEqual(A.Roster(self.env.roster).people()["solo"]["role"], "admin")
        self.assertEqual([r["state"] for r in ADM.people(self.env.roster) if r["id"] == "solo"], ["active"])
        ADM.set_role(self.env.roster, "pat", "admin")                     # control: with a second admin, allowed
        ADM.set_role(self.env.roster, "solo", "basic")
        self.assertEqual(A.Roster(self.env.roster).people()["solo"]["role"], "basic")

    def test_add_key_keeps_the_email(self):
        A.add_key(self.env.roster, "alice", "advanced")                    # re-key replaces the entry
        e = self.entry("alice")
        self.assertEqual((e["email"], e["role"]), ("alice@example.org", "advanced"))
        A.add_key(self.env.roster, "alice", "basic", "alice2@example.org")
        self.assertEqual(self.entry("alice")["email"], "alice2@example.org")
        ADM.set_role(self.env.roster, "alice", "advanced")
        self.assertEqual(self.entry("alice")["email"], "alice2@example.org")
        A.add_key(self.env.roster, "newbie", "basic")
        self.assertNotIn("email", self.entry("newbie"))

    def test_usable_access_and_match_email(self):
        R = A.Roster(self.env.roster)
        self.assertEqual(R.match_email("ALICE@example.org")["id"], "alice")
        self.assertIsNone(R.match_email("gone@example.org"))               # revoked
        self.assertIsNone(R.match_email(""))
        self.assertTrue(A.Roster.usable_access({"id": "x", "role": "basic", "email": "x@example.org"}))   # no key
        self.assertFalse(A.Roster.usable_access({"id": "x", "role": "basic", "key_sha256": "ab"}))      # no email
        doc = json.loads(self.env.roster.read_text(encoding="utf-8"))   # a hand-edited duplicate: nobody gets in
        doc["people"].append({"id": "alice2", "role": "basic", "email": "alice@example.org"})
        self.env.roster.write_text(json.dumps(doc), encoding="utf-8")
        self.assertIsNone(A.Roster(self.env.roster).match_email("alice@example.org"))

    def test_cli_set_email_and_list(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(A.main(["set-email", "--id", "adv", "--email", "a2@example.org",
                                     "--roster", str(self.env.roster)]), 0)
            self.assertEqual(A.main(["set-email", "--id", "adv", "--email", "alice@example.org",
                                     "--roster", str(self.env.roster)]), 1)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            A.main(["list", "--roster", str(self.env.roster)])
        self.assertIn("a2@example.org", out.getvalue())


class ConfigAndStart(unittest.TestCase):
    def setUp(self):
        self.env = T.Env()
        self.cfgfile = self.env.tmp / "cloudflare_access.json"
        self.cfgfile.write_text(json.dumps({"team": TEAM, "aud": AUD, "host": HOST}), encoding="utf-8")
        self.clean = {k: v for k, v in os.environ.items() if not k.startswith("CJ_ACCESS_")}

    def tearDown(self):
        self.env.cleanup()

    def test_config_file_and_env_override(self):
        c = AX.load_config(self.cfgfile, env={})
        self.assertEqual((c.issuer, c.origin, c.certs_url),
                         (ISS, ORIGIN, f"{ISS}/cdn-cgi/access/certs"))
        c = AX.load_config(self.cfgfile, env={"CJ_ACCESS_HOST": "Other.Example.org", "CJ_ACCESS_TEAM": "team-b"})
        self.assertEqual((c.host, c.team, c.aud), ("other.example.org", "team-b", AUD))
        c = AX.load_config(self.env.tmp / "none.json",
                           env={"CJ_ACCESS_TEAM": TEAM, "CJ_ACCESS_AUD": AUD, "CJ_ACCESS_HOST": HOST})
        self.assertEqual(c, ACFG)
        for bad in ({"team": TEAM, "aud": AUD}, {"team": "https://x.cloudflareaccess.com", "aud": AUD, "host": HOST},
                    {"team": TEAM, "aud": AUD, "host": "https://portal.example.org"}):
            self.cfgfile.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(AX.ConfigError):
                AX.load_config(self.cfgfile, env={})
        with self.assertRaises(AX.ConfigError):
            AX.load_config(self.env.tmp / "none.json", env={})

    def test_access_admin_setting(self):
        self.assertFalse(AX.admin_allowed({}))
        for v in ("off", "yes", "1", "true", "onn", ""):
            self.assertFalse(AX.admin_allowed({"CJ_ACCESS_ADMIN": v}), v)
        self.assertTrue(AX.admin_allowed({"CJ_ACCESS_ADMIN": "on"}))
        with mock.patch.dict(os.environ, {"CJ_ACCESS_ADMIN": "on"}):    # Config None -> the env decides
            app = APP.create_app(self.env.config(auth="access", access=ACFG))
            self.assertTrue(app.access_admin)
        with mock.patch.dict(os.environ, {"CJ_ACCESS_ADMIN": "yes"}):
            self.assertFalse(APP.create_app(self.env.config(auth="access", access=ACFG)).access_admin)

    def args(self, *extra):
        return ["--auth", "access", "--registry", str(self.env.registry), "--roster", str(self.env.roster),
                "--state-dir", str(self.env.state), "--review-package", str(self.env.tmp / "none"), *extra]

    def test_refuses_to_start_without_config(self):
        out = io.StringIO()
        with mock.patch.dict(os.environ, self.clean, clear=True), mock.patch("logging.basicConfig"), \
                mock.patch.object(APP, "serve_sockets") as serve, contextlib.redirect_stdout(out):
            rc = APP.main(self.args("--access-config", str(self.env.tmp / "missing.json"), "--port",
                                    str(T.free_port())))
        self.assertEqual(rc, 2)
        serve.assert_not_called()
        self.assertIn("refusing to start", out.getvalue())
        self.assertIn("missing team, aud, host", out.getvalue())

    def test_main_binds_both_loopbacks_and_never_a_wildcard(self):
        port = T.free_port()
        got = {}

        def fake_serve(server, socks):
            got["addrs"] = [s.getsockname()[0] for s in socks]
            got["port"] = {s.getsockname()[1] for s in socks}
            for s in socks:
                s.close()
        out = io.StringIO()
        with mock.patch.dict(os.environ, self.clean, clear=True), mock.patch("logging.basicConfig"), \
                mock.patch.object(APP, "serve_sockets", fake_serve), mock.patch("uvicorn.run") as krun, \
                contextlib.redirect_stdout(out):
            rc = APP.main(self.args("--access-config", str(self.cfgfile), "--port", str(port), "--host", "0.0.0.0"))
        self.assertEqual(rc, 0)
        krun.assert_not_called()
        self.assertEqual(got["port"], {port})
        self.assertIn("127.0.0.1", got["addrs"])
        self.assertNotIn("0.0.0.0", got["addrs"])
        self.assertNotIn("::", got["addrs"])
        if self.v6():
            self.assertIn("::1", got["addrs"])
        text = out.getvalue()
        self.assertIn("ACCESS mode", text)
        self.assertNotIn(AUD, text)                                        # no AUD in the console

    @staticmethod
    def v6() -> bool:
        if not socket.has_ipv6:
            return False
        try:
            s = socket.socket(socket.AF_INET6)
            s.bind(("::1", 0))
            s.close()
            return True
        except OSError:
            return False

    def test_both_loopbacks_answer(self):
        roster_people(self.env)
        app = APP.create_app(self.env.config(auth="access", access=ACFG, access_admin=False))
        app.verifier.keys = AX.KeySet(ACFG.certs_url, fetch=Fetcher())
        port = T.free_port()
        socks = APP.loopback_sockets(port)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, access_log=False,
                                               proxy_headers=False, log_config=None, log_level="warning"))
        th = threading.Thread(target=APP.serve_sockets, args=(server, socks), daemon=True)
        th.start()
        try:
            for _ in range(200):
                if server.started:
                    break
                time.sleep(0.025)
            hosts = ["127.0.0.1"] + (["::1"] if self.v6() else [])
            if not self.v6():
                print("[skip] no IPv6 loopback here: [::1] not checked", file=sys.stderr)
            for h in hosts:
                import http.client
                c = http.client.HTTPConnection(h, port, timeout=10)
                c.request("GET", "/cj/me.json", headers={HDR: sign()})
                r = c.getresponse()
                self.assertEqual((h, r.status, json.loads(r.read())["id"]), (h, 200, "alice"))
                c.close()
            with self.assertRaises(OSError):                               # wildcard on that port: not ours, but the
                APP.loopback_sockets(port)                                 # port is held exclusively
        finally:
            server.should_exit = True
            th.join(10)


if __name__ == "__main__":
    unittest.main()
