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
                    ["example.edu/x"], ["example.edu:443"], ["exa mple.edu"], ["examplе.edu"], [1], ["edu"]):
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
        self.assertNotEqual(a, G.guest_id("ann+x@example.edu"))
        self.assertTrue(A.id_ok(a))

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
        st, me2 = self.me("first+evh@example.edu")                                     # plus-addressing: its own guest
        self.assertEqual((st, me2["role"]), (200, "guest"))
        self.assertNotEqual(me2["id"], me["id"])

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
        sets = APP.guest_sets(self.cfg, {"id": "guest-00000000", "role": "guest"})
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
        self.assertEqual(meta["reviewers"], ["alice", "kate", gid])
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


class MayOpen(unittest.TestCase):
    def test_rules(self):
        can_see = lambda u, o: u.get("role") == "admin" or u.get("id") == o            # noqa: E731
        g = {"id": "guest-0123abcd", "role": "guest"}
        self.assertTrue(REV.may_open(g, {"open_to": "guests"}, can_see))
        for meta in ({}, {"open_to": "Guests"}, {"open_to": ["guests"]}, {"open_to": True},
                     {"reviewers": ["guest-0123abcd"]}, {"owner": "guest-0123abcd"}):
            self.assertFalse(REV.may_open(g, meta, can_see), meta)
        b = {"id": "bob", "role": "basic"}                                               # others: unchanged
        self.assertFalse(REV.may_open(b, {"open_to": "guests"}, can_see))
        self.assertTrue(REV.may_open(b, {"reviewers": ["bob"]}, can_see))


if __name__ == "__main__":
    unittest.main()
