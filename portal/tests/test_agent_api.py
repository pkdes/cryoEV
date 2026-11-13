"""
The agent API, read-only (round 10 #60; docs/API.md): the token store (portal/auth.py), the Gate's bearer branch
(portal/app.py BEARER_ROUTES), its audit lines (portal/audit.py), My account (portal/modules/account.py), the tokens on
Admin > People, the CLI, the one-file client (client/cryoev_client.py) and the pre-push pattern.

Positive controls first in each class (a valid token answers 200 as that person), or the refusals prove nothing.
Temp roster / state dir / token store / predict root only (cjtest.Env); never ~/.cryoev.

  python -m unittest discover -s portal/tests -p test_agent_api.py -v
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cjtest as T  # noqa: E402

from portal import app as APP  # noqa: E402
from portal import audit as PA  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import admin as ADM  # noqa: E402

TOKEN_SHAPE = re.compile(r"cev_[A-Za-z0-9_-]{43}")
FORM_T = re.compile(r'name="t" value="([0-9a-f]+)"')
SETS = {"tester": "ps_20261112_a00001", "adv": "ps_20261112_a00002", "bas": "ps_20261112_a00003",
        "other": "ps_20261112_a00004"}


def clean_env():
    """No agent / API env from the outside (CRYOEV_*, CJ_API, CJ_AGENT_TOKENS) leaks into a test."""
    for k in [k for k in os.environ if k.upper().startswith("CRYOEV_") or k.upper() in ("CJ_API", "CJ_AGENT_TOKENS")]:
        os.environ.pop(k, None)


def load_client():
    spec = importlib.util.spec_from_file_location("cryoev_client_t", T.REPO / "client" / "cryoev_client.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_sets(root: Path):
    """The golden prediction set (tests/fixtures/cj_golden) copied once per owner (as test_results_module)."""
    import test_results_module as TR
    for owner, ps in SETS.items():
        TR.make_set(root, ps, owner=owner)


def audit_lines(env) -> list:
    p = PA.audit_path(env.state)
    if not p.exists():
        return []
    out = []
    for x in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(x))
        except ValueError:
            pass
    return out


# ================================================================ the store (no portal)
class Store(unittest.TestCase):
    def setUp(self):
        clean_env()
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_tok_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.roster, self.path = self.tmp / "roster.json", self.tmp / "agent_tokens.json"
        A.add_key(self.roster, "ann", "advanced")
        ADM.add_person(self.roster, "mail", "basic", "mail@example.org", with_key=False)   # email-only (QC 3)

    def test_token_shown_once_stored_as_sha256(self):
        tok, rec = A.create_token(self.path, "ann", "claude", 30, roster=self.roster)
        self.assertRegex(tok, r"\Acev_[A-Za-z0-9_-]{43}\Z")
        raw = self.path.read_text(encoding="utf-8")
        self.assertNotIn(tok, raw)
        self.assertIn(A.sha256_hex(tok), raw)
        self.assertNotIn("sha256", rec)                          # the public record never carries it
        self.assertEqual((rec["person"], rec["agent"], rec["scope"], rec["state"]), ("ann", "claude", "read", "active"))
        self.assertTrue(A.TOKEN_ID_RE.fullmatch(rec["id"]))
        p, got = A.TokenStore(self.path).check(tok, A.Roster(self.roster))
        self.assertEqual((p["id"], got["id"]), ("ann", rec["id"]))   # positive control

    def test_create_refusals(self):
        for kw in ({"days": 0}, {"days": 91}, {"days": "x"}, {"agent": "Claude"}, {"agent": ""}, {"agent": "9x"},
                   {"agent": "a" * 33}, {"scope": "write"}, {"scope": "admin"}):
            args = {"days": 30, "agent": "claude", "scope": "read", **kw}
            with self.subTest(kw=kw), self.assertRaises(A.TokenError):
                A.create_token(self.path, "ann", args["agent"], args["days"], args["scope"], roster=self.roster)
        with self.assertRaisesRegex(A.TokenError, "not available yet"):
            A.create_token(self.path, "ann", "claude", scope="write")
        with self.assertRaises(A.TokenError):
            A.create_token(self.path, "nobody", "claude", roster=self.roster)
        self.assertFalse(self.path.exists())                       # nothing was written

    def test_at_most_ten_live_tokens_per_person(self):
        old = time.time() - 40 * 86400
        A.create_token(self.path, "ann", "claude", 30, now=old)    # expired: doesn't count
        made = [A.create_token(self.path, "ann", "claude", 30)[1] for _ in range(A.TOKENS_PER_PERSON)]
        with self.assertRaisesRegex(A.TokenError, "live tokens already"):
            A.create_token(self.path, "ann", "claude")
        A.create_token(self.path, "mail", "claude")                # someone else still can
        A.revoke_token(self.path, made[0]["id"], by="ann")
        A.create_token(self.path, "ann", "claude")                 # a revoked one frees a place

    def test_validity(self):
        st, ro = A.TokenStore(self.path), A.Roster(self.roster)
        good, _ = A.create_token(self.path, "mail", "claude")      # email-only person: valid (QC 3)
        self.assertEqual(st.check(good, ro)[0]["id"], "mail")
        expired, _ = A.create_token(self.path, "ann", "claude", 1, now=time.time() - 2 * 86400)
        self.assertEqual(st.check(expired, ro), (None, "expired"))
        rev, rec = A.create_token(self.path, "ann", "claude")
        A.revoke_token(self.path, rec["id"], by="ann")
        self.assertEqual(st.check(rev, ro), (None, "revoked"))
        self.assertEqual(st.check("cev_" + "A" * 43, ro), (None, "unknown"))
        for bad in ("", "cev_short", "Bearer x", None, "cev_" + "A" * 44, "xev_" + "A" * 43):
            self.assertEqual(st.check(bad, ro), (None, "malformed"), bad)
        # a hand-planted write-scope or over-long token is refused
        doc = json.loads(self.path.read_text(encoding="utf-8"))
        w = "cev_" + "w" * 43
        long_ = "cev_" + "L" * 43
        doc["tokens"] += [dict(doc["tokens"][0], id="tk_00000000000a", sha256=A.sha256_hex(w), scope="write",
                               person="ann"),
                          dict(doc["tokens"][0], id="tk_00000000000b", sha256=A.sha256_hex(long_), person="ann",
                               expires=A._utc(time.time() + 400 * 86400))]
        A.write_json_atomic(self.path, doc)
        self.assertEqual(st.check(w, ro), (None, "scope"))
        self.assertEqual(st.check(long_, ro), (None, "expired"))
        # the person revoked / removed / a bad role
        ok, _ = A.create_token(self.path, "ann", "claude")
        self.assertIsNotNone(st.check(ok, ro)[0])
        A.revoke(self.roster, "ann")
        self.assertEqual(st.check(ok, ro), (None, "person"))
        A.write_json_atomic(self.roster, {"people": []})
        self.assertEqual(st.check(good, ro), (None, "person"))

    def test_missing_or_unreadable_store_refuses_everyone(self):
        st, ro = A.TokenStore(self.path), A.Roster(self.roster)
        self.assertEqual(st.check("cev_" + "A" * 43, ro), (None, "unknown"))
        tok, _ = A.create_token(self.path, "ann", "claude")
        self.assertIsNotNone(st.check(tok, ro)[0])
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(st.check(tok, ro), (None, "unknown"))
        with self.assertRaises(A.TokenError):
            A.create_token(self.path, "ann", "claude")
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{not json")     # never overwritten

    def test_last_used_flush_is_throttled_and_never_clears_a_revoke(self):
        clock = [1_000_000.0]
        st = A.TokenStore(self.path, flush_s=300, clock=lambda: clock[0])
        tok, rec = A.create_token(self.path, "ann", "claude")
        st.used(rec["id"])
        self.assertEqual(st.flush(), 0)                            # < 5 min since the last flush: memory only
        self.assertIsNone(A.list_tokens(self.path)[0]["last_used"])
        A.revoke_token(self.path, rec["id"], by="admin")          # meanwhile, elsewhere
        clock[0] += 301
        self.assertEqual(st.flush(), 1)
        row = A.list_tokens(self.path)[0]
        self.assertEqual(row["state"], "revoked")                 # the merge kept the revoke
        self.assertIsNotNone(row["last_used"])
        self.assertEqual(st.check(tok, A.Roster(self.roster)), (None, "revoked"))

    def test_two_processes_on_the_store_dont_lose_a_revoke(self):
        """A second process flushes last_used and makes tokens in a loop while this one revokes and makes tokens:
        afterwards the revoke stands and every token made by either is there."""
        victim, rec = A.create_token(self.path, "ann", "claude")
        ready = self.tmp / "ready"
        script = textwrap.dedent(f"""
            import sys, time
            sys.path.insert(0, {str(T.REPO)!r})
            from pathlib import Path
            from portal import auth as A
            p = Path({str(self.path)!r})
            st = A.TokenStore(p, flush_s=0)
            Path({str(ready)!r}).write_text("go")
            for i in range(40):
                st.used({rec["id"]!r}, now=2_000_000_000 + i)
                st.flush(force=True)
                if i % 8 == 0:
                    A.create_token(p, "mail", "worker")
            """)
        proc = subprocess.Popen([sys.executable, "-c", script], cwd=str(T.REPO))
        t0 = time.time()
        while not ready.exists() and time.time() - t0 < 30:
            time.sleep(0.02)
        A.revoke_token(self.path, rec["id"], by="admin")
        mine = [A.create_token(self.path, "ann", "main")[1]["id"] for _ in range(3)]
        self.assertEqual(proc.wait(60), 0)
        rows = {r["id"]: r for r in A.list_tokens(self.path)}
        self.assertEqual(rows[rec["id"]]["state"], "revoked")
        self.assertIsNotNone(rows[rec["id"]]["last_used"])         # the other process's flushes landed too
        self.assertTrue(set(mine) <= set(rows))
        self.assertEqual(sum(1 for r in rows.values() if r["agent"] == "worker"), 5)
        self.assertEqual(A.TokenStore(self.path).check(victim, A.Roster(self.roster)), (None, "revoked"))

    def test_cli_mint_list_revoke(self):
        state = self.tmp / "state"
        common = ["--tokens", str(self.path)]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = A.main(["mint-token", "--id", "ann", "--agent", "claude", "--days", "7", "--roster", str(self.roster),
                         "--state-dir", str(state), *common])
        self.assertEqual(rc, 0)
        toks = TOKEN_SHAPE.findall(out.getvalue())
        self.assertEqual(len(toks), 1)
        tok = toks[0]
        p, rec = A.TokenStore(self.path).check(tok, A.Roster(self.roster))
        self.assertEqual(p["id"], "ann")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(A.main(["list-tokens", *common]), 0)
        self.assertIn(rec["id"], out.getvalue())
        self.assertNotIn(tok, out.getvalue())
        self.assertNotIn(A.sha256_hex(tok), out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(A.main(["revoke-token", "--token-id", rec["id"], "--state-dir", str(state), *common]), 0)
            self.assertEqual(A.main(["revoke-token", "--token-id", rec["id"], *common]), 1)    # already
            self.assertEqual(A.main(["mint-token", "--id", "ann", "--agent", "claude", "--scope", "write",
                                     "--roster", str(self.roster), *common]), 1)
            self.assertEqual(A.main(["mint-token", "--id", "nobody", "--agent", "claude", "--roster",
                                     str(self.roster), *common]), 1)
        lines = [json.loads(x) for x in PA.audit_path(state).read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(x["action"], x["target"]) for x in lines],
                         [("agent token made", rec["id"]), ("agent token revoked", rec["id"])])
        self.assertEqual(lines[0]["detail"]["for"], "ann")
        blob = PA.audit_path(state).read_text(encoding="utf-8")
        self.assertNotIn(tok, blob)
        self.assertNotIn(A.sha256_hex(tok), blob)

    def test_cli_person_revoke_revokes_their_tokens_but_never_the_default_store(self):
        _, rec = A.create_token(self.path, "ann", "claude")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(A.main(["revoke", "--id", "ann", "--roster", str(self.roster), "--tokens",
                                     str(self.path)]), 0)
        self.assertEqual(A.list_tokens(self.path)[0]["state"], "revoked")
        home = self.tmp / "home"
        with mock.patch.object(A, "default_dir", lambda: home), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(A.main(["revoke", "--id", "mail", "--roster", str(self.roster)]), 0)
        self.assertFalse(home.exists())                            # a --roster elsewhere: the default store untouched

    def test_prepush_scan_blocks_a_token(self):
        sys.path.insert(0, str(T.REPO / "tools"))
        import prepush_scan as S
        tok, _ = A.create_token(self.path, "ann", "claude")
        self.assertIn("cryoev-agent-token", [r for r, _ in S.scan_line(f"CRYOEV_TOKEN={tok}", S.KEY_RULES)])
        self.assertEqual(S.scan_line("CRYOEV_TOKEN=cev_...", S.KEY_RULES), [])          # the docs' placeholder


# ================================================================ the Gate (a real portal)
class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clean_env()
        cls.env = T.Env()
        cls.tmp = Path(tempfile.mkdtemp(prefix="cj_api_"))
        root, review = cls.tmp / "predict", cls.tmp / "review"
        for d in (root / ".incoming", root / ".locks", review):
            d.mkdir(parents=True)
        (root / "predict.lock").write_text("", encoding="utf-8")
        make_sets(root)
        r = cls.env.roster
        cls.keys = {"tester": cls.env.key, "adv": A.add_key(r, "adv", "advanced"), "bas": A.add_key(r, "bas", "basic"),
                    "other": A.add_key(r, "other", "basic"), "adv2": A.add_key(r, "adv2", "advanced"),
                    "aud": A.add_key(r, "aud", "advanced"), "gone": A.add_key(r, "gone", "basic")}
        ADM.add_person(r, "mail", "advanced", "mail@example.org", with_key=False)
        cfg = cls.env.config()
        cfg.predict_root, cfg.review_root = root, review          # Results reads predict_root
        cfg.cj_predict_root = str(root)                            # Predict reads cj_predict_root (its _setting)
        cls.p = T.Portal(cfg)
        cls.cookies = {k: T.login(cls.p.port, v) for k, v in cls.keys.items()}
        cls.tok = {k: cls.mint(k) for k in ("tester", "adv", "bas", "other", "adv2", "mail")}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        clean_env()

    @classmethod
    def mint(cls, pid, agent="claude", **kw):
        return A.create_token(cls.env.tokens, pid, agent, roster=cls.env.roster, **kw)[0]

    def bget(self, path, tok, headers=None, method="GET", body=None):
        return T.request(self.p.port, method, path, {**T.bearer(tok), **(headers or {})}, body)

    def cget(self, path, who, headers=None):
        return T.request(self.p.port, "GET", path, {**T.cookie(self.cookies[who]), **(headers or {})})

    @staticmethod
    def paths(ps) -> dict:
        """One concrete path per allowlisted template."""
        return {"/cj/me.json": "/cj/me.json", "/data/api/overview": "/data/api/overview",
                "/data/api/search": "/data/api/search?q=grid", "/data/api/list/{kind}": "/data/api/list/sessions",
                "/data/api/{kind}/{eid}": "/data/api/session/nope",
                "/data/api/lineage/{kind}/{eid}": "/data/api/lineage/session/nope",
                "/data/api/lineage/graph": "/data/api/lineage/graph", "/predict/api/mine": "/predict/api/mine",
                "/predict/api/set/{ps}": f"/predict/api/set/{ps}", "/results/{ps}/api/status": f"/results/{ps}/api/status",
                "/jobs/api/summary": "/jobs/api/summary"}

    def wait_lines(self, pred, timeout=10.0) -> list:
        end = time.time() + timeout
        while True:
            self.p.app.writer.drain()
            got = [x for x in audit_lines(self.env) if pred(x)]
            if got or time.time() > end:
                return got
            time.sleep(0.05)


class Gate(Base):
    def test_positive_control_me_json(self):
        st, h, body = self.bget("/cj/me.json", self.tok["adv"])
        self.assertEqual(st, 200, body)
        self.assertEqual(json.loads(body), {"id": "adv", "role": "advanced", "via": "claude", "auth": "bearer"})
        self.assertEqual(T.hget(h, "x-frame-options"), "DENY")              # security headers as always
        st, _, body = self.bget("/cj/me.json", self.tok["tester"])
        self.assertEqual(json.loads(body)["role"], "advanced")              # an admin's agent is capped

    def test_allowlist_is_exact_and_every_template_exists(self):
        self.assertEqual(set(self.paths("x")), {t for _, t in APP.BEARER_ROUTES})
        self.assertEqual({m for m, _ in APP.BEARER_ROUTES}, {"GET"})
        have = {getattr(r, "path", None) for r in self.p.app.routes}
        self.assertEqual({t for _, t in APP.BEARER_ROUTES} - have, set())
        for tpl, path in self.paths(SETS["adv"]).items():
            st, _, body = self.bget(path, self.tok["adv"])
            self.assertIn(st, (200, 404), f"{tpl}: {st} {body[:200]}")   # reached the handler (404 = no such id)

    def test_everything_else_is_401_without_a_cookie_fallback(self):
        ps = SETS["adv"]
        refused = [("GET", p) for p in (
            "/", "/healthz", "/login?k=x", "/cj/cj.css", "/cj/brand.json", "/cj/modules.json", "/cj/version.json",
            "/cj/cryoev_client.py", "/about", "/changes", "/account/", "/admin/", "/admin/api/people",
            "/admin/api/audit", "/data/", "/data/search?q=x", "/data/api/overview/", "/DATA/api/overview",
            "/data/lineage/graph", "/data/thumb/session/x/1.jpg", "/jobs/", "/jobs/api/summary/", "/predict/",
            "/predict/api/convert/x", "/predict/api/upload/x", "/predict/api/recipe/x", f"/predict/{ps}/",
            f"/results/{ps}/", f"/results/{ps}/download/x", "/gallery/", "/train/api/runs", "/feedback/",
            "/reviews/", "/history/live/live.json", "/review/", "/review/app/", "/nope")]
        refused += [("HEAD", "/cj/me.json"), ("POST", "/cj/me.json"), ("PUT", "/data/api/overview"),
                    ("POST", "/jobs/api/cancel"), ("POST", "/predict/api/upload"), ("POST", "/account/tokens/new"),
                    ("POST", "/admin/people/add"), ("DELETE", "/data/api/session/x")]
        for method, path in refused:
            with self.subTest(method=method, path=path):
                st, h, body = self.bget(path, self.tok["adv"], T.cookie(self.cookies["tester"]), method=method,
                                        body=b"" if method != "GET" else None)
                self.assertEqual(st, 401, body[:200])
                self.assertEqual(T.hget(h, "www-authenticate"), 'Bearer realm="cryoev"')
                if method != "HEAD":
                    self.assertIn("read routes", json.loads(body)["error"])

    def test_route_role_matrix_matches_the_browser_rules(self):
        """For every allowlisted route x role: the agent's answer is the browser's answer for that person with the
        role capped at advanced (an admin's agent = an advanced person's browser), on that person's own set."""
        twin = {"tester": "adv2", "adv": "adv", "bas": "bas"}         # the browser that has the capped role
        own = {"tester": SETS["adv"], "adv": SETS["adv"], "bas": SETS["bas"]}
        own_twin = {"adv2": SETS["adv"], "adv": SETS["adv"], "bas": SETS["bas"]}
        for who in ("tester", "adv", "bas"):
            for tpl, path in self.paths(own[who]).items():
                with self.subTest(who=who, route=tpl):
                    b_st, _, b_body = self.bget(path, self.tok[who])
                    tw = twin[who]
                    c_st, _, c_body = self.cget(self.paths(own_twin[tw])[tpl], tw)
                    if tpl in ("/predict/api/set/{ps}", "/results/{ps}/api/status") and who == "tester":
                        # the twin (adv2) doesn't own adv's set; the agent of tester doesn't either: both 404
                        self.assertEqual((b_st, c_st), (404, 404))
                        continue
                    self.assertEqual(b_st, c_st, f"bearer {b_body[:150]} / browser {c_body[:150]}")
                    rules = self.p.app.rules
                    capped = APP.BEARER_ROLE_CAP.get(A.Roster(self.env.roster).people()[who]["role"])
                    capped = capped or A.Roster(self.env.roster).people()[who]["role"]
                    if not APP.M.allowed(rules, path.split("?")[0], capped):
                        self.assertEqual(b_st, 403)
        # basic: no Data, no Jobs; advanced: both
        self.assertEqual(self.bget("/data/api/overview", self.tok["bas"])[0], 403)
        self.assertEqual(self.bget("/jobs/api/summary", self.tok["bas"])[0], 403)
        self.assertEqual(self.bget("/jobs/api/summary", self.tok["adv"])[0], 200)

    def test_owner_checks_apply_to_an_admins_agent(self):
        other = SETS["other"]
        self.assertEqual(self.cget(f"/predict/api/set/{other}", "tester")[0], 200)       # the admin's browser: all
        self.assertEqual(self.cget(f"/results/{other}/api/status", "tester")[0], 200)
        self.assertEqual(self.bget(f"/predict/api/set/{other}", self.tok["tester"])[0], 404)   # the agent: own only
        self.assertEqual(self.bget(f"/results/{other}/api/status", self.tok["tester"])[0], 404)
        self.assertEqual(self.bget(f"/predict/api/set/{SETS['tester']}", self.tok["tester"])[0], 200)
        st, _, body = self.bget("/predict/api/mine", self.tok["tester"])
        ids = {i.get("id") or i.get("ps") for i in json.loads(body)["items"]}
        self.assertEqual(ids, {SETS["tester"]})
        st, _, body = self.bget(f"/predict/api/set/{SETS['bas']}", self.tok["other"])
        self.assertEqual(st, 404)                                                         # basic: not someone else's

    def test_expired_revoked_and_person_revoked_tokens_fail(self):
        expired = self.mint("adv", days=1, now=time.time() - 3 * 86400)
        revoked, rec = A.create_token(self.env.tokens, "adv", "claude")
        A.revoke_token(self.env.tokens, rec["id"], by="t")
        gone = self.mint("gone")
        self.assertEqual(self.bget("/cj/me.json", gone)[0], 200)                         # control
        ADM.revoke_person(self.env.roster, "gone")
        planted = "cev_" + "P" * 43
        doc = json.loads(self.env.tokens.read_text(encoding="utf-8"))
        doc["tokens"].append(dict(doc["tokens"][0], id="tk_0000000000ff", sha256=A.sha256_hex(planted),
                                  scope="write", person="adv", revoked=None))
        A.write_json_atomic(self.env.tokens, doc)
        for name, tok in (("expired", expired), ("revoked", revoked), ("person revoked", gone),
                          ("write scope", planted), ("unknown", "cev_" + "Z" * 43), ("malformed", "abc"),
                          ("empty", "")):
            with self.subTest(name):
                st, _, body = self.bget("/cj/me.json", tok)
                self.assertEqual(st, 401, body)
                self.assertIn("unknown, expired or revoked", json.loads(body)["error"])
        self.assertEqual(self.bget("/cj/me.json", self.tok["adv"])[0], 200)              # others unaffected

    def test_email_only_person_token_works(self):
        st, _, body = self.bget("/cj/me.json", self.tok["mail"])
        self.assertEqual((st, json.loads(body)["id"]), (200, "mail"))

    def test_two_authorization_headers_are_refused(self):
        tok = self.tok["adv"]
        raw = (f"GET /cj/me.json HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {tok}\r\n"
               f"Authorization: Basic eDp5\r\nConnection: close\r\n\r\n").encode()
        self.assertEqual(T.raw_status(T.raw(self.p.port, raw)), 401)
        st, _, _ = T.request(self.p.port, "GET", "/cj/me.json", {"Authorization": f"bearer {tok}"})
        self.assertEqual(st, 200)                                                        # scheme case-insensitive

    def test_cookies_are_ignored_with_a_bearer(self):
        admin = T.cookie(self.cookies["tester"])
        self.assertEqual(self.cget("/admin/api/people", "tester")[0], 200)               # control: the cookie works
        self.assertEqual(self.bget("/cj/me.json", "cev_" + "Q" * 43, admin)[0], 401)     # bad token: no fallback
        st, _, body = self.bget("/cj/me.json", self.tok["adv"], admin)
        self.assertEqual(json.loads(body)["id"], "adv")                                   # the token's person
        self.assertEqual(self.bget("/data/api/overview", self.tok["bas"], admin)[0], 403)  # not the admin's rights
        self.assertEqual(self.bget("/admin/api/people", self.tok["adv"], admin)[0], 401)

    def test_handlers_never_see_the_cookie_or_the_token(self):
        seen = {}

        async def me(request):
            seen["headers"] = dict(request.headers)
            seen["auth"], seen["via"] = request.scope.get("cj_auth"), request.scope.get("cj_via")
            from starlette.responses import JSONResponse
            return JSONResponse({})

        from starlette.applications import Starlette
        from starlette.routing import Route
        star = Starlette(routes=[Route("/cj/me.json", me, methods=["GET"])])
        gate = APP.Gate(star, A.Auth(self.env.roster, self.env.state), tokens=A.TokenStore(self.env.tokens))
        tok = self.tok["adv"]
        scope = {"type": "http", "method": "GET", "path": "/cj/me.json", "query_string": b"", "client": ("x", 1),
                 "headers": [(b"host", b"x"), (b"authorization", f"Bearer {tok}".encode()),
                             (b"cookie", f"{A.COOKIE}={self.cookies['tester']}".encode())]}
        sent = []

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(msg):
            sent.append(msg)
        asyncio.run(gate(scope, receive, send))
        self.assertEqual(sent[0]["status"], 200)
        self.assertNotIn("cookie", seen["headers"])
        self.assertNotIn("authorization", seen["headers"])
        self.assertEqual((seen["auth"], seen["via"]), ("bearer", "claude"))

    def test_kill_switch(self):
        with mock.patch.dict(os.environ, {"CJ_API": "off"}):
            st, _, body = self.bget("/cj/me.json", self.tok["adv"])
            self.assertEqual(st, 401)
            self.assertIn("switched off", json.loads(body)["error"])
            self.assertEqual(self.cget("/cj/me.json", "adv")[0], 200)                    # browsers unaffected
        self.assertEqual(self.bget("/cj/me.json", self.tok["adv"])[0], 200)

    def test_audit_lines_carry_via_on_the_verified_door(self):
        claude, codex = self.mint("aud", "claude"), self.mint("aud", "codex")
        for tok in (claude, claude, codex):
            self.assertEqual(self.bget("/cj/me.json", tok)[0], 200)
        signs = self.wait_lines(lambda x: x["actor"] == "aud" and x["action"] == "signed in"
                                and x["detail"].get("door") == PA.API_DOOR)
        time.sleep(0.2)
        signs = [x for x in audit_lines(self.env) if x["actor"] == "aud" and x["action"] == "signed in"
                 and x["detail"].get("door") == PA.API_DOOR]
        self.assertEqual(sorted((x["detail"]["door"], x["detail"]["via"], x["detail"]["how"]) for x in signs),
                         [(PA.API_DOOR, "claude", "bearer"), (PA.API_DOOR, "codex", "bearer")])   # one per agent
        # a refused write (a valid token on a route it doesn't open): the person + the agent, status 401
        self.assertEqual(self.bget("/jobs/api/cancel", claude, method="POST", body=b"")[0], 401)
        got = self.wait_lines(lambda x: x["actor"] == "aud" and x["action"] == "job cancel")
        self.assertEqual(got[0]["detail"], {"status": 401, "via": "claude"})
        # frequent routes: the first line now, the rest counted, the counter line carries via
        for _ in range(3):
            self.bget("/gallery/c1/api/event", claude, method="POST", body=b"")
        self.p.app.writer.flush()
        lines = self.wait_lines(lambda x: x["actor"] == "aud" and x["action"] == "gallery mark"
                                and (x["detail"] or {}).get("n") == 3)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["detail"]["via"], "claude")
        firsts = [x for x in audit_lines(self.env) if x["actor"] == "aud" and x["action"] == "gallery mark"
                  and "n" not in x["detail"]]
        self.assertEqual([x["detail"] for x in firsts], [{"status": 401, "via": "claude"}])
        for x in signs + got + lines:
            self.assertTrue(T.gate_line(x), x)                                           # still a Gate line
            row = ADM.audit_row(x, own_door="tailnet key")
            self.assertEqual((row["name"], row["ai"], row["unverified"], row["door"]),
                             (f"{x['detail']['via']} for aud", True, False, PA.API_DOOR))
        st, _, body = self.cget("/admin/audit", "tester")
        page = body.decode("utf-8")
        self.assertIn("claude for aud", page)
        self.assertIn(PA.API_DOOR, page)
        aud_rows = [r for r in re.findall(r'<li class="cj-admin-aud" data-who="([^"]*)"', page) if "aud" in r]
        self.assertTrue(aud_rows and all("unverified" not in r for r in aud_rows), aud_rows)

    def test_a_token_is_never_logged_or_audited(self):
        rec = logging.getLogger()
        buf = io.StringIO()
        h = logging.StreamHandler(buf)
        h.setLevel(logging.DEBUG)
        old = rec.level
        rec.addHandler(h)
        rec.setLevel(logging.DEBUG)
        try:
            tok = self.mint("adv2", "logcheck")
            bad = "cev_" + "N" * 43
            for t, path, method in ((tok, "/cj/me.json", "GET"), (tok, "/admin/", "GET"),
                                    (tok, "/jobs/api/cancel", "POST"), (bad, "/cj/me.json", "GET"),
                                    (tok, f"/cj/me.json?k={tok}", "GET")):
                self.bget(path, t, method=method, body=b"" if method == "POST" else None)
            self.p.app.writer.flush()
            self.p.app.writer.drain()
        finally:
            rec.removeHandler(h)
            rec.setLevel(old)
        logs = buf.getvalue()
        self.assertIn("adv2 via logcheck (bearer)", logs)                                # control: lines were logged
        files = "".join(p.read_text(encoding="utf-8", errors="replace") for p in self.env.state.rglob("*")
                        if p.is_file())
        for secret in (tok, bad, A.sha256_hex(tok), A.sha256_hex(bad)):
            self.assertNotIn(secret, logs)
            self.assertNotIn(secret, files)

    def test_client_end_to_end(self):
        C = load_client()
        c = C.Client(self.p.origin, self.tok["adv"])
        self.assertNotIn(self.tok["adv"], repr(c))
        self.assertEqual(c.me()["id"], "adv")
        self.assertIn("error", c.overview())
        self.assertIsInstance(c.list("sessions")["rows"], list)
        self.assertIsInstance(c.search("grid"), dict)
        self.assertIsInstance(c.graph(), dict)
        self.assertIsInstance(c.jobs_summary(), dict)
        self.assertEqual({i.get("id") or i.get("ps") for i in c.mine()["items"]}, {SETS["adv"]})
        self.assertIsInstance(c.prediction_set(SETS["adv"]), dict)
        self.assertIsInstance(c.results_status(SETS["adv"]), dict)
        with self.assertRaises(C.ApiError) as e:
            c.prediction_set(SETS["other"])
        self.assertEqual(e.exception.status, 404)
        with self.assertRaises(C.ApiError) as e:
            C.Client(self.p.origin, "cev_" + "R" * 43).me()
        self.assertEqual(e.exception.status, 401)
        with self.assertRaises(C.ApiError) as e:
            C.Client(self.p.origin, self.tok["bas"]).overview()
        self.assertEqual(e.exception.status, 403)
        with self.assertRaises(C.ApiError) as e:
            c.get("/admin/api/people")
        self.assertEqual(e.exception.status, 401)
        # the command line, from the env
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"CRYOEV_URL": self.p.origin, "CRYOEV_TOKEN": self.tok["adv"]}), \
                contextlib.redirect_stdout(out):
            self.assertEqual(C.main(["me"]), 0)
        self.assertEqual(json.loads(out.getvalue())["via"], "claude")
        # served to signed-in people, byte for byte
        st, h, body = self.cget("/cj/cryoev_client.py", "bas")
        self.assertEqual((st, body), (200, (T.REPO / "client" / "cryoev_client.py").read_bytes()))
        self.assertEqual(T.request(self.p.port, "GET", "/cj/cryoev_client.py")[0], 401)   # not without a sign-in

    def test_client_is_stdlib_only(self):
        import ast
        tree = ast.parse((T.REPO / "client" / "cryoev_client.py").read_text(encoding="utf-8"))
        mods = {(n.module if isinstance(n, ast.ImportFrom) else a.name).split(".")[0]
                for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in (n.names if isinstance(n, ast.Import) else [n])}
        self.assertTrue(mods <= {"__future__", "json", "os", "sys", "urllib"}, mods)


class Account(Base):
    def form_t(self, who, path="/account/") -> str:
        st, _, body = self.cget(path, who)
        self.assertEqual(st, 200)
        return FORM_T.search(body.decode("utf-8")).group(1)

    def post(self, who, path, fields, origin=True):
        from urllib.parse import urlencode
        h = {**T.cookie(self.cookies[who]), "Content-Type": "application/x-www-form-urlencoded"}
        if origin:
            h["Origin"] = self.p.origin
        return T.request(self.p.port, "POST", path, h, urlencode(fields).encode())

    def test_make_use_list_and_revoke_your_own_token(self):
        t = self.form_t("bas")
        st, h, body = self.post("bas", "/account/tokens/new", {"t": t, "agent": "claude", "days": "30"})
        self.assertEqual(st, 200, body[:300])
        self.assertEqual(T.hget(h, "cache-control"), "no-store")
        page = body.decode("utf-8")
        toks = TOKEN_SHAPE.findall(page)
        self.assertEqual(len(set(toks)), 1)
        tok = toks[0]
        tid = re.search(r"Token (tk_[0-9a-f]{12}) made", page).group(1)
        st, _, me = self.bget("/cj/me.json", tok)
        self.assertEqual((st, json.loads(me)["id"]), (200, "bas"))
        st, _, body = self.cget("/account/", "bas")
        page = body.decode("utf-8")
        self.assertNotIn(tok, page)                                                       # shown once only
        rec = next(r for r in A.list_tokens(self.env.tokens, "bas") if r["id"] == tid)
        self.assertIn(rec["id"], page)
        # someone else can't revoke it
        st, _, _ = self.post("other", "/account/tokens/revoke", {"t": self.form_t("other"), "id": rec["id"]})
        self.assertEqual(st, 404)
        self.assertEqual(self.bget("/cj/me.json", tok)[0], 200)
        st, h, _ = self.post("bas", "/account/tokens/revoke", {"t": t, "id": rec["id"]})
        self.assertEqual(st, 303)
        self.assertEqual(self.bget("/cj/me.json", tok)[0], 401)
        lines = self.wait_lines(lambda x: x["target"] == rec["id"])
        self.assertEqual(sorted((x["actor"], x["action"]) for x in lines),
                         [("bas", "agent token made"), ("bas", "agent token revoked")])
        blob = PA.audit_path(self.env.state).read_text(encoding="utf-8")
        self.assertNotIn(tok, blob)
        self.assertNotIn(A.sha256_hex(tok), blob)

    def test_refusals(self):
        t = self.form_t("adv2")
        for fields, code in (({"agent": "Claude", "days": "30"}, 400), ({"agent": "claude", "days": "91"}, 400),
                             ({"agent": "claude", "days": "0"}, 400)):
            with self.subTest(fields=fields):
                self.assertEqual(self.post("adv2", "/account/tokens/new", {"t": t, **fields})[0], code)
        self.assertEqual(self.post("adv2", "/account/tokens/new", {"t": "bad", "agent": "claude", "days": "7"})[0],
                         403)                                                             # form token
        self.assertEqual(self.post("adv2", "/account/tokens/new", {"t": t, "agent": "claude", "days": "7"},
                                   origin=False)[0], 403)                                 # Origin (the Gate)

    def test_admin_people_lists_tokens_and_revokes(self):
        tok = self.mint("other", "codex")
        rec = next(r for r in A.list_tokens(self.env.tokens, "other") if r["agent"] == "codex")
        st, _, body = self.cget("/admin/", "tester")
        page = body.decode("utf-8")
        self.assertIn("codex for other", page)
        self.assertIn(f'data-token-id="{rec["id"]}"', page)
        self.assertNotIn(tok, page)
        t = FORM_T.search(page).group(1)
        st, _, _ = self.post("tester", "/admin/people/token-revoke", {"t": t, "id": rec["id"]})
        self.assertEqual(st, 303)
        self.assertEqual(self.bget("/cj/me.json", tok)[0], 401)
        line = self.wait_lines(lambda x: x["target"] == rec["id"] and x["action"] == "agent token revoked")
        self.assertEqual((line[0]["actor"], line[0]["detail"]["for"]), ("tester", "other"))

    def test_admin_revoking_a_person_revokes_their_tokens(self):
        ADM.add_person(self.env.roster, "leaver", "basic", "leaver@example.org", with_key=False)
        tok = self.mint("leaver")
        self.assertEqual(self.bget("/cj/me.json", tok)[0], 200)
        t = self.form_t("tester", "/admin/")
        st, _, _ = self.post("tester", "/admin/people/revoke", {"t": t, "id": "leaver", "confirm": "yes"})
        self.assertEqual(st, 303)
        self.assertEqual([r["state"] for r in A.list_tokens(self.env.tokens, "leaver")], ["revoked"])
        self.assertEqual(self.bget("/cj/me.json", tok)[0], 401)
        line = self.wait_lines(lambda x: x["target"] == "leaver" and x["action"] == "person_revoked")
        self.assertEqual(line[0]["detail"].get("tokens_revoked"), 1)


class AccessDoor(unittest.TestCase):
    """The Cloudflare door refuses bearer (tailnet-only for now); its JWT sign-in and My account still work."""

    @classmethod
    def setUpClass(cls):
        clean_env()
        import test_access as TA
        cls.TA = TA
        cls.env = T.Env()
        TA.roster_people(cls.env)
        cls.p = TA.access_portal(cls.env, TA.Fetcher())
        cls.tok = A.create_token(cls.env.tokens, "adv", "claude", roster=cls.env.roster)[0]

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()

    def test_jwt_works_bearer_is_refused_with_a_clear_message(self):
        jwt_ = self.TA.sign("adv@example.org")
        st, _, body = self.TA.get(self.p.port, "/cj/me.json", jwt_)
        self.assertEqual((st, json.loads(body)["id"]), (200, "adv"))                     # control
        for hdrs in ({**T.bearer(self.tok), self.TA.HDR: jwt_}, T.bearer(self.tok)):
            st, _, body = T.request(self.p.port, "GET", "/cj/me.json", hdrs)
            self.assertEqual(st, 403)
            self.assertIn("API is tailnet-only for now", json.loads(body)["error"])
        st, _, body = self.TA.get(self.p.port, "/account/", jwt_)
        self.assertEqual(st, 200)
        self.assertIn("Tailscale address only", body.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
