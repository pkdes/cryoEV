"""
Round 17 guest access (portal/guests.py): an unlisted email at EXACTLY a configured domain signs in through the
Cloudflare door as role "guest" and can open only review sets with open_to "guests".

Positive controls first (an exact-domain guest gets in; a guest saves an answer on an open object set through the real
multi review server), then every refusal. Placeholders only (example.edu, example.org): the repo is public. Temp
roster, temp review root, temp access config; never ~/.cryoev or the live review folder.

  python -m unittest portal.tests.test_guest_access -v
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cjtest as T  # noqa: E402
from test_access import ACFG, Fetcher, ORIGIN, access_portal, get, roster_people, sign  # noqa: E402

sys.path.insert(0, str(T.REPO / "annotation" / "tests"))
from test_object_sets import golden_ps, items_csv, ALL10, SENT_NUM, SENT_TXT  # noqa: E402

from annotation import multi_review_server as MRS  # noqa: E402
from annotation import object_set as OS  # noqa: E402
from portal import access as AX  # noqa: E402
from portal import auth as A  # noqa: E402
from portal import guests as G  # noqa: E402
from portal.modules import reviews as REV  # noqa: E402

DOM = "example.edu"
GCFG = replace(ACFG, guest_domains=(DOM,))
FORBIDDEN = b"Your role can"


def people(roster) -> list:
    return json.loads(Path(roster).read_text(encoding="utf-8"))["people"]


def md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def guest_portal(cfg):
    p = T.Portal(cfg)
    p.app.verifier.keys = AX.KeySet(ACFG.certs_url, fetch=Fetcher())
    return p


def load_data(pkg: Path) -> dict:
    return json.loads((pkg / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))


# ---------------------------------------------------------------- units: config, domain rule, roster entry
class DomainRule(unittest.TestCase):
    def test_parse_domains(self):
        self.assertEqual(G.parse_domains(None), ())
        self.assertEqual(G.parse_domains([" Example.EDU ", "example.edu"]), ("example.edu",))
        for bad in ("example.edu", [""], ["*.example.edu"], ["@example.edu"], ["https://example.edu"],
                    ["example.edu/x"], ["example.edu:443"], ["exa mple.edu"], ["examplе.edu"], [1], ["edu"],
                    ["gmail.com"], ["Outlook.com"], ["yahoo.com"], ["hotmail.com"], ["icloud.com"]):
            with self.assertRaises(ValueError, msg=bad):
                G.parse_domains(bad)

    def test_exact_domain_only(self):
        d = (DOM,)
        for ok in ("ann@example.edu", "Ann@Example.EDU", "ann+evh@example.edu", "a.b@example.edu",
                   " ann@example.edu\n"):                                       # stripped, as match_email does
            self.assertTrue(G.domain_ok(ok, d), ok)
        for bad in ("ann@sub.example.edu", "ann@evil-example.edu", "ann@example.edu.evil.com", "ann@xexample.edu",
                    "ann@example.edux", "ann@example.org", "ann@example.edu.", "ann@example.edu@evil.com",
                    "example.edu", "@example.edu", "ann@examplе.edu",       # Cyrillic e
                    "annK@example.edu",                                 # Kelvin sign (lower() -> k)
                    "ann @example.edu", "ann@exa\nmple.edu", None, 3, ""):
            self.assertFalse(G.domain_ok(bad, d), repr(bad))
        self.assertFalse(G.domain_ok("ann@example.edu", ()))                 # feature off

    def test_guest_id_stable(self):
        a = G.guest_id("Ann@Example.edu")
        self.assertEqual(a, G.guest_id("ann@example.edu"))
        self.assertRegex(a, G.GUEST_ID_RE)
        self.assertEqual(a, G.guest_id("ann+x@example.edu"))                  # +tag dropped: one per mailbox
        self.assertEqual(a, G.guest_id("ANN+Other+Tag@example.edu"))
        self.assertRegex(a, r"\Aguest-[0-9a-f]{16}\Z")
        self.assertTrue(A.id_ok(a))
        self.assertIsNone(G.canonical("+x@example.edu"))

    def test_load_config(self):
        tmp = Path(tempfile.mkdtemp(prefix="cj_guest_cfg_"))
        try:
            f = tmp / "cf.json"
            base = {"team": "example-team", "aud": "test-aud-0123456789abcdef", "host": "portal.example.org"}
            env = {}
            f.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(AX.load_config(f, env).guest_domains, ())          # missing key: off
            f.write_text(json.dumps({**base, G.CONFIG_KEY: []}), encoding="utf-8")
            self.assertEqual(AX.load_config(f, env).guest_domains, ())
            f.write_text(json.dumps({**base, G.CONFIG_KEY: ["Example.edu"]}), encoding="utf-8")
            self.assertEqual(AX.load_config(f, env).guest_domains, ("example.edu",))
            for bad in ("example.edu", ["*.example.edu"], [None]):
                f.write_text(json.dumps({**base, G.CONFIG_KEY: bad}), encoding="utf-8")
                with self.assertRaises(AX.ConfigError):
                    AX.load_config(f, env)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class RosterEntry(unittest.TestCase):
    def setUp(self):
        self.env = T.Env()
        roster_people(self.env)

    def tearDown(self):
        self.env.cleanup()

    def test_made_once_stable_fields(self):
        p = G.ensure_guest(self.env.roster, "Ann@Example.edu", (DOM,))
        self.assertEqual(set(p), {"id", "role", "email", "added", "auto"})
        self.assertEqual((p["id"], p["role"], p["email"], p["auto"]),
                         (G.guest_id("ann@example.edu"), "guest", "ann@example.edu", True))
        n = len(people(self.env.roster))
        q = G.ensure_guest(self.env.roster, "ann@example.edu", (DOM,))
        self.assertEqual(q["id"], p["id"])
        self.assertEqual(len(people(self.env.roster)), n)
        self.assertEqual(A.Roster(self.env.roster).match_email("ann@example.edu")["id"], p["id"])

    def test_revoked_guest_not_recreated(self):
        p = G.ensure_guest(self.env.roster, "ann@example.edu", (DOM,))
        self.assertTrue(A.revoke(self.env.roster, p["id"]))
        before = md5(self.env.roster)
        self.assertIsNone(G.ensure_guest(self.env.roster, "ann@example.edu", (DOM,)))
        self.assertEqual(md5(self.env.roster), before)

    def test_existing_email_or_id_blocks(self):
        A.set_email(self.env.roster, "kate", "kate@example.edu")       # a real person at the guest domain
        A.revoke(self.env.roster, "kate")                              # ... revoked: never comes back as a guest
        before = md5(self.env.roster)
        self.assertIsNone(G.ensure_guest(self.env.roster, "kate@example.edu", (DOM,)))
        self.assertEqual(md5(self.env.roster), before)
        doc = json.loads(Path(self.env.roster).read_text(encoding="utf-8"))   # an id collision with another email
        doc["people"].append({"id": G.guest_id("bob@example.edu"), "role": "guest", "email": "other@example.edu"})
        A.write_json_atomic(self.env.roster, doc)
        self.assertIsNone(G.ensure_guest(self.env.roster, "bob@example.edu", (DOM,)))

    def test_wrong_domain_or_missing_roster_writes_nothing(self):
        before = md5(self.env.roster)
        self.assertIsNone(G.ensure_guest(self.env.roster, "ann@sub.example.edu", (DOM,)))
        self.assertIsNone(G.ensure_guest(self.env.roster, "ann@example.edu", ()))
        self.assertEqual(md5(self.env.roster), before)
        gone = self.env.tmp / "no_roster.json"
        self.assertIsNone(G.ensure_guest(gone, "ann@example.edu", (DOM,)))
        self.assertFalse(gone.exists())

    def test_concurrent_writes_atomic_under_lock(self):
        emails = [f"g{i}@example.edu" for i in range(8)] * 3            # 8 people, each 3 times at once
        out, errs = [], []

        def go(e):
            try:
                out.append(G.ensure_guest(self.env.roster, e, (DOM,))["id"])
            except Exception as ex:                                     # noqa: BLE001
                errs.append(ex)
        ts = [threading.Thread(target=go, args=(e,)) for e in emails]
        [t.start() for t in ts]
        [t.join(30) for t in ts]
        self.assertEqual(errs, [])
        ids = [p["id"] for p in people(self.env.roster) if p.get("role") == "guest"]
        self.assertEqual(sorted(ids), sorted({G.guest_id(e) for e in emails}))       # one each, none lost
        self.assertEqual(sorted(set(out)), sorted(ids))
        self.assertEqual([f for f in os.listdir(self.env.tmp) if f.endswith(".tmp")], [])

    def test_plus_tag_is_the_same_guest(self):
        p = G.ensure_guest(self.env.roster, "ann+one@example.edu", (DOM,))
        self.assertEqual(p["email"], "ann@example.edu")
        self.assertEqual(G.ensure_guest(self.env.roster, "ann+two@example.edu", (DOM,))["id"], p["id"])
        self.assertEqual(len([q for q in people(self.env.roster) if q.get("role") == "guest"]), 1)
        A.revoke(self.env.roster, p["id"])                                   # revoked: no +tag way back in
        self.assertIsNone(G.ensure_guest(self.env.roster, "ann+three@example.edu", (DOM,)))

    def test_caps(self):
        from datetime import datetime
        day = datetime(2026, 11, 17, 12, 0)
        with mock.patch.object(G, "MAX_GUESTS_PER_DAY", 3):
            for i in range(3):
                G.ensure_guest(self.env.roster, f"d{i}@example.edu", (DOM,), now=day)
            before = md5(self.env.roster)
            with self.assertRaises(G.GuestRefused):
                G.ensure_guest(self.env.roster, "d9@example.edu", (DOM,), now=day)
            self.assertEqual(md5(self.env.roster), before)
            G.ensure_guest(self.env.roster, "d9@example.edu", (DOM,), now=datetime(2026, 11, 18, 9, 0))  # next day
            self.assertIsNotNone(G.ensure_guest(self.env.roster, "d0@example.edu", (DOM,), now=day))     # known: ok
        with mock.patch.object(G, "MAX_GUESTS", 4):
            A.revoke(self.env.roster, G.guest_id("d1@example.edu"))           # revoked ones still count
            with self.assertRaises(G.GuestRefused):
                G.ensure_guest(self.env.roster, "new@example.edu", (DOM,))

    def test_known_or_revoked_needs_no_lock(self):
        p = G.ensure_guest(self.env.roster, "ann@example.edu", (DOM,))
        G.ensure_guest(self.env.roster, "rev@example.edu", (DOM,))
        A.revoke(self.env.roster, G.guest_id("rev@example.edu"))
        known = list(A.Roster(self.env.roster).people().values())
        with mock.patch.object(A, "acquire_lock", side_effect=AssertionError("locked")):
            self.assertEqual(G.ensure_guest(self.env.roster, "ann@example.edu", (DOM,), known)["id"], p["id"])
            self.assertIsNone(G.ensure_guest(self.env.roster, "rev@example.edu", (DOM,), known))
            self.assertIsNone(G.ensure_guest(self.env.roster, "rev+x@example.edu", (DOM,), known))
            with self.assertRaises(AssertionError):                           # a new email does take the lock
                G.ensure_guest(self.env.roster, "brand-new@example.edu", (DOM,), known)

    def test_transient_read_failure_is_busy_not_blocked(self):
        real = Path.read_text
        fails = {"n": 2}

        def flaky(self_, *a, **k):
            if self_.name == "roster.json" and fails["n"] > 0:
                fails["n"] -= 1
                raise PermissionError("being replaced")
            return real(self_, *a, **k)
        with mock.patch.object(Path, "read_text", flaky):                     # brief: retried, then made
            self.assertEqual(G.ensure_guest(self.env.roster, "ann@example.edu", (DOM,))["role"], "guest")
        fails["n"] = 10 ** 6
        before = md5(self.env.roster)
        with mock.patch.object(Path, "read_text", flaky), mock.patch.object(G, "READ_PAUSE_S", 0.001):
            with self.assertRaises(OSError):                                   # lasting: busy (503), never None
                G.ensure_guest(self.env.roster, "bob@example.edu", (DOM,))
        self.assertEqual(md5(self.env.roster), before)
        bad = self.env.tmp / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with mock.patch.object(G, "READ_PAUSE_S", 0.001):
            self.assertIsNone(G.ensure_guest(bad, "ann@example.edu", (DOM,)))   # malformed: nobody

    def test_lock_busy_changes_nothing(self):
        before = md5(self.env.roster)
        with mock.patch.object(A, "acquire_lock", return_value=None):
            with self.assertRaises(OSError):
                G.ensure_guest(self.env.roster, "ann@example.edu", (DOM,))
        self.assertEqual(md5(self.env.roster), before)

    def test_guest_never_gets_tokens_or_key_sessions(self):
        p = G.ensure_guest(self.env.roster, "ann@example.edu", (DOM,))
        with self.assertRaises(A.TokenError):
            A.create_token(self.env.tokens, p["id"], "claude", roster=self.env.roster)
        self.assertFalse(A.person_ok(A.Roster(self.env.roster).people()[p["id"]]))
        self.assertFalse(A.Roster.usable({**p, "key_sha256": "x"}))           # key mode: never a guest


# ---------------------------------------------------------------- the Gate (access mode)
class GateOff(unittest.TestCase):
    """Feature off (no guest_email_domains): a guest-domain email is just not on the list."""

    def test_off(self):
        env = T.Env()
        roster_people(env)
        p = access_portal(env, Fetcher())
        try:
            before = md5(env.roster)
            st, _, body = get(p.port, "/cj/me.json", sign("ann@example.edu"))
            self.assertEqual(st, 403)
            self.assertIn(b"not on the list", body.lower())
            self.assertEqual(md5(env.roster), before)
            self.assertEqual(p.app.guest_domains, ())
        finally:
            p.stop()
            env.cleanup()

    def test_key_mode_never_makes_guests(self):
        env = T.Env()
        p = T.Portal(env.config())
        try:
            self.assertEqual(p.app.guest_domains, ())
            before = md5(env.roster)
            st, _, _ = T.request(p.port, "GET", "/cj/me.json", {"Cf-Access-Jwt-Assertion": sign("ann@example.edu")})
            self.assertEqual(st, 401)
            self.assertEqual(md5(env.roster), before)
        finally:
            p.stop()
            env.cleanup()


def build_sets(tmp: Path, review: Path):
    """Two object sets (golden prediction set): evharmony (open to guests) and private_set (not), lab reviewers
    alice and kate, owner adv (who may open them but is not a reviewer: the server-check control)."""
    ps = golden_ps(tmp)
    src = items_csv(tmp / "items.csv", [(i, ln, SENT_NUM, SENT_TXT) for i, ln in ALL10])
    out = []
    for slug in ("evharmony", "private_set"):
        with contextlib.redirect_stdout(io.StringIO()):
            out.append(OS.build(ps, src, "ls_lamellarity_v1", review, line_col="line", blind_cols=("gap_nm", "secret"),
                                slug=slug, seed=11, reviewers=("alice", "kate"), owner="adv", title=f"{slug} set"))
    return out


class GuestPortal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        roster_people(cls.env)
        cls.review = cls.env.tmp / "review"
        cls.review.mkdir()
        cls.open_pkg, cls.closed_pkg = build_sets(cls.env.tmp, cls.review)
        with contextlib.redirect_stdout(io.StringIO()):
            G.main(["open", "--package", str(cls.open_pkg)])
        env = {"CJ_REVIEW_EXCLUDE": "", "CJ_REVIEW_PREFIXES": ""}
        with mock.patch.dict(os.environ, env):
            cls.srv = MRS.make_server(cls.review, "127.0.0.1", 0, people=[])
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.cfg = cls.env.config(auth="access", access=GCFG, access_admin=False, review_root=cls.review,
                                 pred_upstream=f"http://127.0.0.1:{cls.srv.server_address[1]}")
        cls.p = guest_portal(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.env.cleanup()

    def g(self, path, email="ann@example.edu", **kw):
        return get(self.p.port, path, sign(email), **kw)

    def me(self, email):
        st, _, body = self.g("/cj/me.json", email)
        return st, (json.loads(body) if st == 200 else body)

    # ---- positive controls
    def test_exact_domain_admitted_once_stable(self):
        st, me = self.me("first@example.edu")
        self.assertEqual(st, 200, me)
        self.assertEqual(me, {"id": G.guest_id("first@example.edu"), "role": "guest",
                              "signout": "/cdn-cgi/access/logout"})
        self.assertEqual(self.me("FIRST@Example.EDU"), (200, me))                     # same person, same id
        hits = [p for p in people(self.env.roster) if p.get("email") == "first@example.edu"]
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0]["auto"])
        self.assertEqual(self.me("first+evh@example.edu"), (200, me))                  # +tag: the same guest

    def test_existing_people_unaffected(self):
        st, me = self.me("alice@example.org")
        self.assertEqual((st, me["id"], me["role"]), (200, "alice", "basic"))

    # ---- refusals at sign-in
    def test_lookalikes_refused_and_nothing_written(self):
        before = md5(self.env.roster)
        for e in ("ann@sub.example.edu", "ann@evil-example.edu", "ann@example.edu.evil.com", "ann@xexample.edu",
                  "ann@example.edux", "ann@examplе.edu", "annK@example.edu", "stranger@example.org"):
            st, body = self.me(e)
            self.assertEqual(st, 403, e)
            self.assertIn(b"not on the list", body.lower())
        self.assertEqual(md5(self.env.roster), before)

    def test_unverified_or_missing_jwt_refused(self):
        from test_access import KEY2, forged
        before = md5(self.env.roster)
        for tok in (None, "", sign("ann2@example.edu", key=KEY2), forged("none"),
                    sign("ann2@example.edu", aud=["other-aud-0123456789"]), sign("ann2@example.edu", exp=1)):
            st, _, body = get(self.p.port, "/cj/me.json", tok)
            self.assertEqual(st, 403)
            self.assertNotIn(b'"guest"', body)
        st, _, _ = get(self.p.port, "/cj/me.json", None, headers={"Cookie": f"CF_Authorization={sign('ann2@example.edu')}"})
        self.assertEqual(st, 403)
        self.assertEqual(md5(self.env.roster), before)

    def test_revoked_guest_refused_not_recreated(self):
        st, me = self.me("rev@example.edu")
        self.assertEqual(st, 200)
        self.assertTrue(A.revoke(self.env.roster, me["id"]))
        before = md5(self.env.roster)
        st, body = self.me("rev@example.edu")
        self.assertEqual(st, 403)
        self.assertEqual(md5(self.env.roster), before)
        self.assertEqual(len([p for p in people(self.env.roster) if p.get("email") == "rev@example.edu"]), 1)

    def test_busy_roster_is_503_and_nothing_written(self):
        before = md5(self.env.roster)
        with mock.patch.object(A, "acquire_lock", return_value=None):
            st, _, _ = self.g("/cj/me.json", "busy@example.edu")
        self.assertEqual(st, 503)
        self.assertEqual(md5(self.env.roster), before)

    # ---- what a guest may reach
    def test_other_routes_403(self):
        self.assertEqual(self.me("ann@example.edu")[0], 200)
        for path in ("/admin/", "/data/", "/jobs/", "/gallery/", "/predict/", "/results/", "/reviews/", "/train/",
                     "/account/", "/feedback/", "/apply/", "/about", "/changes", "/history/", "/history/history.html",
                     "/review/", "/review/app/", "/cj/cryoev_client.py", "/nope", f"/reviews/go/{self.open_pkg.name}",
                     f"/reviews/{self.open_pkg.name}/export.csv"):
            st, _, body = self.g(path)
            self.assertEqual(st, 403, path)
            self.assertIn(FORBIDDEN, body, path)
        for path in ("/account/tokens/new", "/admin/people/add", "/feedback/"):
            st, _, body = self.g(path, method="POST", headers={"Origin": ORIGIN}, body=b"agent=claude&days=1")
            self.assertEqual(st, 403, path)
        st, _, _ = self.g("/cj/me.json", headers={"Authorization": "Bearer cev_" + "a" * 43})
        self.assertEqual(st, 403)
        st, _, body = self.g("/cj/modules.json")
        self.assertEqual((st, json.loads(body)["modules"]), (200, []))                 # no nav

    def test_non_open_package_same_as_unknown(self):
        unknown = "pred_obj_nothing_here_0000"
        for sub in ("/", "/app/", "/app/data.js", "/app/api/edits"):
            a = self.g(f"/review/{self.closed_pkg.name}{sub}")
            b = self.g(f"/review/{unknown}{sub}")
            self.assertEqual((a[0], a[2]), (b[0], b[2]), sub)
            self.assertEqual(a[0], 404)
        # control: alice (a listed reviewer, basic) opens the same package
        self.assertEqual(self.g(f"/review/{self.closed_pkg.name}/", "alice@example.org")[0], 200)
        revs = json.loads((self.closed_pkg / "review_package.json").read_text(encoding="utf-8"))["reviewers"]
        self.assertFalse([r for r in revs if r.startswith("guest-")])

    def test_home_lists_only_open_sets(self):
        st, h, _ = self.g("/", "home@example.edu")
        self.assertEqual(st, 303)
        self.assertEqual(T.hget(h, "location"), f"/review/{self.open_pkg.name}/")
        from portal import app as APP
        sets = APP.guest_sets(self.cfg, {"id": "guest-0000000000000000", "role": "guest"})
        self.assertEqual([s[0] for s in sets], [self.open_pkg.name])
        body = APP.render_guest_home(sets).encode()
        self.assertNotIn(self.closed_pkg.name.encode(), body)
        self.assertIn(b"Nothing is open", APP.render_guest_home([]).encode())

    def test_guest_saves_answer_on_open_set_and_sees_no_emails(self):
        email = "answer@example.edu"
        gid = G.guest_id(email)
        st, _, body = self.g(f"/review/{self.open_pkg.name}/", email)                   # opening joins the set
        self.assertEqual(st, 200, body)
        self.assertNotIn(b"@example.org", body)
        meta = json.loads((self.open_pkg / "review_package.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["reviewers"][:2], ["alice", "kate"])
        self.assertIn(gid, meta["reviewers"])
        self.assertEqual(meta["open_to"], "guests")
        st, _, page = self.g(f"/review/{self.open_pkg.name}/app/", email, headers={"Sec-Fetch-Dest": "iframe"})
        self.assertEqual(st, 200)
        self.assertIn(json.dumps(gid).encode(), page)
        self.assertNotIn(b"@example.org", page)
        self.assertNotIn(b"@example.edu", page)
        obj = next(it["obj"] for it in load_data(self.open_pkg)["items"] if it["image"] == "0001")
        dec = {f"a:{obj}:type@x": {"v": "ML", "t": 1000, "who": "x", "pending": True}}
        st, _, out = self.g(f"/review/{self.open_pkg.name}/app/api/edits?f=0001.png", email, method="PUT",
                            headers={"Origin": ORIGIN, "Content-Type": "application/json"},
                            body=json.dumps({"t": 1000, "dec": dec}).encode())
        self.assertEqual(st, 200, out)
        self.assertEqual(json.loads(out)["edit"]["dec"][f"a:{obj}:type@{gid}"]["v"], "ML")
        # control: a lab person who is NOT one of the set's reviewers is refused by the same server check
        st, _, out = self.g(f"/review/{self.open_pkg.name}/app/api/edits?f=0001.png", "adv@example.org", method="PUT",
                            headers={"Origin": ORIGIN, "Content-Type": "application/json"},
                            body=json.dumps({"t": 1000, "dec": dec}).encode())
        self.assertEqual(st, 403, out)
        self.assertIn(b"not a reviewer", out)
        # opening again does not add the guest twice
        self.assertEqual(self.g(f"/review/{self.open_pkg.name}/", email)[0], 200)
        meta = json.loads((self.open_pkg / "review_package.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["reviewers"].count(gid), 1)

    def test_closed_to_guests_is_404_again(self):
        tmp_pkg = self.review / "pred_obj_closing_set_beef"
        shutil.copytree(self.open_pkg, tmp_pkg)
        try:
            self.assertEqual(self.g(f"/review/{tmp_pkg.name}/", "closer@example.edu")[0], 200)
            self.assertTrue(G.set_open(tmp_pkg, False))
            self.assertFalse(G.set_open(tmp_pkg, False))
            a = self.g(f"/review/{tmp_pkg.name}/", "closer@example.edu")
            b = self.g("/review/pred_obj_nothing_here_0000/", "closer@example.edu")
            self.assertEqual((a[0], a[2]), (b[0], b[2]))
            with self.assertRaises(REV.ReviewersError):                                # join refused once closed
                REV.join_guest(self.review, tmp_pkg, G.guest_id("late@example.edu"))
            with self.assertRaises(REV.ReviewersError):
                REV.join_guest(self.review, self.open_pkg, "alice")                     # only guest ids join
        finally:
            shutil.rmtree(tmp_pkg, ignore_errors=True)


class GuestTool(unittest.TestCase):
    """QC round 17 fix 1/2/4/8/9: what a guest may do inside an open set's tool, and when they may join."""

    setUpClass = classmethod(GuestPortal.setUpClass.__func__)
    tearDownClass = classmethod(GuestPortal.tearDownClass.__func__)
    g = GuestPortal.g

    def meta(self, pkg):
        return json.loads((pkg / "review_package.json").read_text(encoding="utf-8"))

    def tool(self, email, sub, method="GET", body=None, pkg=None):
        h = {"Sec-Fetch-Dest": "iframe"}
        if method not in ("GET", "HEAD"):
            h.update({"Origin": ORIGIN, "Content-Type": "application/json"})
        data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
        return self.g(f"/review/{(pkg or self.open_pkg).name}/app/{sub}", email, method=method, headers=h, body=data)

    def put(self, email, body, f="0001.png", pkg=None):
        return self.tool(email, f"api/edits?f={f}", "PUT", body, pkg)

    def obj(self, pkg=None):
        return next(it["obj"] for it in load_data(pkg or self.open_pkg)["items"] if it["image"] == "0001")

    def join(self, email, pkg=None):
        self.assertEqual(self.g(f"/review/{(pkg or self.open_pkg).name}/", email)[0], 200)

    def test_reads_allowed_others_refused(self):
        e = "reader@example.edu"
        self.join(e)
        img = sorted(f.name for f in (self.open_pkg / "images").iterdir())[0]
        for sub in ("", "index.html", "data.js", "api/ping", "api/meta", "api/edits", f"images/{img}"):
            st, _, body = self.tool(e, sub)
            self.assertEqual(st, 200, (sub, body[:200]))
        self.assertEqual(self.tool(e, "data.js", "HEAD")[0], 200)
        for sub in ("api/people", "api/claims", "api/groups", "review_package.json", "schema.json", ".review_key",
                    "edits/_attr_log.jsonl", "edits/0001.png.json", "images/../data.js", "images/", "API/People",
                    "api/people/", "api%2fpeople", "api/edits/x", "nope"):
            st, _, body = self.tool(e, sub)
            self.assertEqual(st, 403, (sub, st, body[:200]))
            self.assertTrue(b"Guests can only answer" in body or FORBIDDEN in body, sub)   # tool rule or the Gate
        # control: a lab reviewer still reaches what guests can't
        self.assertEqual(self.tool("alice@example.org", "api/claims")[0], 200)

    def test_writes_refused_but_attribute_answers(self):
        e = "writer@example.edu"
        gid = G.guest_id(e)
        self.join(e)
        ok = {f"a:{self.obj()}:type@x": {"v": "ML", "t": 3000, "pending": True}}
        for body in ({"t": 3000, "dec": {"i:reject": {"v": "reject:empty", "t": 3000, "pending": True}}},
                     {"t": 3000, "dec": {"i:note": {"v": "guest note", "t": 3000, "pending": True}}},
                     {"t": 3000, "dec": {"g1": {"v": "del", "t": 3000, "pending": True}}},
                     {"t": 3000, "dec": {"g1": "del"}},                                       # a bare-string decision
                     {"t": 3000, "dec": {**ok, "i:note": {"v": "x", "t": 3000, "pending": True}}},   # mixed: all refused
                     {"t": 3000, "dec": {}}, {"t": 3000}, [1], "x", b"not json", b"",
                     {"t": 3000, "dec": {"i:note": {"v": "echo", "t": 1, "who": "alice"}}}):     # nothing to answer
            st, _, out = self.put(e, body)
            self.assertEqual(st, 403, (body, out[:200]))
        for method, sub in (("POST", "api/group?who=x"), ("POST", "api/claim?c=0001.png|g1"), ("DELETE", "api/edits"),
                            ("PUT", "api/group"), ("POST", "api/edits?f=0001.png")):
            st, _, out = self.tool(e, sub, method, {"action": "comment", "kind": "sample", "sample": "s", "reason": "x"})
            self.assertEqual(st, 403, (method, sub, out[:200]))
        # gt / gtT / reviewedT and other people's keys are dropped; the attribute answer is saved under the guest
        alice_note = {"i:note": {"v": "alice says fine", "t": 1500, "pending": True}}
        self.assertEqual(self.put("alice@example.org", {"t": 1500, "dec": alice_note})[0], 200)
        body = {"t": 3000, "gt": [], "gtT": 3000, "reviewedT": 3000,
                "dec": {**ok, "i:note": {"v": "guest overwrote", "t": 9000, "who": "alice"},
                        f"a:{self.obj()}:type@alice": {"v": "ML", "t": 9000, "who": "alice"}}}
        st, _, out = self.put(e, body)
        self.assertEqual(st, 200, out)
        st, _, out = self.tool("alice@example.org", "api/edits")
        rec = json.loads(out)["0001.png"]
        self.assertEqual(rec["dec"]["i:note"]["v"], "alice says fine")
        self.assertFalse(rec.get("reviewedT"))                                # the guest's reviewedT was dropped
        self.assertFalse(rec.get("reviewed"))
        self.assertNotIn(f"a:{self.obj()}:type@{gid}", rec["dec"])            # alice never sees the guest's answer
        st, _, out = self.tool(e, "api/edits")
        self.assertEqual(json.loads(out)["0001.png"]["dec"][f"a:{self.obj()}:type@{gid}"]["v"], "ML")

    def _copy(self, name, **meta):
        pkg = self.review / name
        shutil.copytree(self.open_pkg, pkg)
        m = self.meta(pkg)
        m["reviewers"] = ["alice", "kate"]
        m.update(meta)
        (pkg / "review_package.json").write_text(json.dumps(m), encoding="utf-8")
        self.addCleanup(shutil.rmtree, pkg, True)
        return pkg

    def test_path_tricks_never_reach_a_closed_set(self):
        """QC probe 01: case, dot, encoded and traversal variants of a closed set's path; other doors and methods."""
        e = "paths@example.edu"
        self.join(e)
        cp, op = self.closed_pkg.name, self.open_pkg.name
        for path in (f"/review/{cp.upper()}/", f"/review/{cp}./", f"/review/{cp}%2e/", f"/review/{cp}%20/",
                     f"/review/{cp}/app/", f"/review/{cp}//app/data.js", f"/review/{op}/../{cp}/",
                     f"/review/{op}/app/../../{cp}/app/data.js", f"/review/{op}/app/..%2f..%2f{cp}/data.js",
                     f"/review/{op}/app/%2e%2e/{cp}/data.js", f"/review/{op}/app/%252e%252e%252f{cp}/data.js",
                     "/review/app/", "/review/APP/", "/review/app%2f", "/review//", "//review/app/",
                     "/history/live/live.json", "/cj/cryoev_client.py", "/cj/version.json", "/cj/../admin/"):
            st, _, body = self.g(path, e)
            self.assertIn(st, (403, 404), path)
            self.assertNotIn(b"REVIEW_DATA", body, path)
        for m in ("HEAD", "OPTIONS", "PATCH", "TRACE", "DELETE"):
            self.assertIn(self.g("/admin/", e, method=m)[0], (403, 405), m)

    def test_guest_t_and_size(self):
        e = "tclamp@example.edu"
        self.join(e)
        from portal import app as APP
        out = json.loads(APP.guest_edit_body(json.dumps({"t": 9e15, "dec": {
            f"a:{self.obj()}:type@x": {"v": "ML", "t": 1, "pending": True}}}).encode()))
        self.assertEqual(out["t"], 0)
        big = {"t": 1, "dec": {f"a:{self.obj()}:type@x": {"v": "ML", "t": 1, "pending": True}}, "pad": "x" * (1 << 20)}
        self.assertEqual(self.put(e, big)[0], 413)

    def test_last_lab_reviewer_kept_while_open(self):
        pkg = self._copy("pred_obj_lastlab_set_abcd", reviewers=["alice", G.guest_id("g@example.edu")])
        with self.assertRaises(REV.ReviewersError):
            REV.write_reviewers(self.review, pkg, remove=["alice"])
        self.assertEqual(REV.write_reviewers(self.review, pkg, add=["kate"], remove=["alice"]), (["kate"], ["alice"]))
        G.set_open(pkg, False)                                                 # closed to guests: may empty it
        self.assertEqual(REV.write_reviewers(self.review, pkg, remove=["kate"]), ([], ["kate"]))

    def test_empty_reviewers_never_locked_out(self):
        pkg = self._copy("pred_obj_noreviewers_set_abcd", reviewers=[])
        with self.assertRaises(ValueError):                                    # can't be opened with no reviewers
            G.set_open(pkg, True)
        self.assertEqual(G.main(["open", "--package", str(pkg)]), 1)
        self.join("lock@example.edu", pkg)                                     # opened by hand anyway: no join
        self.assertEqual(self.meta(pkg)["reviewers"], [])
        with self.assertRaises(REV.ReviewersError):
            REV.join_guest(self.review, pkg, G.guest_id("lock@example.edu"))
        m = self.meta(pkg)                                                     # only guests listed: still no join
        m["reviewers"] = [G.guest_id("someone@example.edu")]
        (pkg / "review_package.json").write_text(json.dumps(m), encoding="utf-8")
        with self.assertRaises(REV.ReviewersError):
            REV.join_guest(self.review, pkg, G.guest_id("lock@example.edu"))

    def test_closed_set_no_join(self):
        pkg = self._copy("pred_obj_closedset_set_abcd", closed=True)
        with self.assertRaises(REV.ReviewersError):
            REV.join_guest(self.review, pkg, G.guest_id("late@example.edu"))
        self.assertTrue(G.joinable({"open_to": "guests", "reviewers": ["alice"]}))
        self.assertFalse(G.joinable({"open_to": "guests", "reviewers": ["alice"], "closed": True}))

    def test_guests_dont_count_toward_cap(self):
        pkg = self._copy("pred_obj_capped_set_beef")
        for i in range(REV.MAX_REVIEWERS + 5):
            self.assertTrue(REV.join_guest(self.review, pkg, f"guest-{i:016x}"))
        self.assertEqual(REV.write_reviewers(self.review, pkg, add=["adv"]), (["adv"], []))   # the owner still adds
        with mock.patch.object(REV, "GUEST_MAX_REVIEWERS", REV.MAX_REVIEWERS + 5):
            with self.assertRaises(REV.ReviewersError):
                REV.join_guest(self.review, pkg, "guest-" + "f" * 16)
        with mock.patch.object(REV, "MAX_REVIEWERS", 3):                      # the lab cap: lab people only
            with self.assertRaises(REV.ReviewersError):
                REV.write_reviewers(self.review, pkg, add=["boss"])

    def test_join_only_on_shell_and_failures_cached(self):
        pkg = self._copy("pred_obj_joinonce_set_abcd")
        e = "shell@example.edu"
        gid = G.guest_id(e)
        self.assertEqual(self.tool(e, "data.js", pkg=pkg)[0], 200)            # a tool read never joins
        self.assertNotIn(gid, self.meta(pkg)["reviewers"])
        calls = []
        real = REV.join_guest

        def failing(*a, **k):
            calls.append(a)
            raise REV.ReviewersError("busy", busy=True)
        with mock.patch.object(REV, "join_guest", failing):
            self.join(e, pkg)
            self.join(e, pkg)                                                  # cached failure: not retried
        self.assertEqual(len(calls), 1)
        self.assertNotIn(gid, self.meta(pkg)["reviewers"])
        self.assertIs(REV.join_guest, real)

    def test_revoked_guest_warning_rate_limited(self):
        st, me = GuestPortal.me(self, "noisy@example.edu")
        self.assertEqual(st, 200)
        A.revoke(self.env.roster, me["id"])
        with self.assertLogs("cj", level="WARNING") as cm:
            for _ in range(5):
                self.assertEqual(self.g("/cj/me.json", "noisy@example.edu")[0], 403)
            import logging
            logging.getLogger("cj").warning("end marker")
        self.assertEqual(sum("revoked or blocked" in r for r in cm.output), 1)


class MayOpen(unittest.TestCase):
    def test_rules(self):
        can_see = lambda u, o: u.get("role") == "admin" or u.get("id") == o            # noqa: E731
        g = {"id": "guest-0123abcd0123abcd", "role": "guest"}
        self.assertTrue(REV.may_open(g, {"open_to": "guests"}, can_see))
        for meta in ({}, {"open_to": "Guests"}, {"open_to": ["guests"]}, {"open_to": True},
                     {"reviewers": ["guest-0123abcd0123abcd"]}, {"owner": "guest-0123abcd0123abcd"}):
            self.assertFalse(REV.may_open(g, meta, can_see), meta)
        b = {"id": "bob", "role": "basic"}                                               # others: unchanged
        self.assertFalse(REV.may_open(b, {"open_to": "guests"}, can_see))
        self.assertTrue(REV.may_open(b, {"reviewers": ["bob"]}, can_see))


if __name__ == "__main__":
    unittest.main()
