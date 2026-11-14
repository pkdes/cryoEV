"""
P5: Admin add-by-email (email-only people, "Make key link", health display) and the My reviews tab.
Temp roster / state / review folders only (cjtest.Env); host names are portal.example.org. Never ~/.cryoev.
"""
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
import test_access as TA  # noqa: E402
from portal import access_config as AC  # noqa: E402
from portal import audit as PA  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import admin as ADM  # noqa: E402
from portal.modules import reviews as REV  # noqa: E402

FORM = "application/x-www-form-urlencoded"
TOKEN_RE = re.compile(r'name="t" value="([0-9a-f]{40})"')


def entry(env, pid):
    return [p for p in json.loads(env.roster.read_text(encoding="utf-8"))["people"] if p["id"] == pid][-1]


class AdminBase(unittest.TestCase):
    host = None          # patched into access_config.public_host

    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.cfg_dir = Path(tempfile.mkdtemp(prefix="cj_p5cfg_"))
        cls.svc = cls.cfg_dir / "services.json"
        cls.svc.write_text(json.dumps({"run_dir": str(cls.cfg_dir / "run"), "services": []}), encoding="utf-8")
        cls.saved = {k: os.environ.get(k) for k in ("CRYOEV_SERVICES_CONFIG", "CJ_ACCESS_HOST")}
        os.environ["CRYOEV_SERVICES_CONFIG"] = str(cls.svc)
        os.environ.pop("CJ_ACCESS_HOST", None)
        cls.saved_pub = AC.public_host
        AC.public_host = lambda *a, **k: cls.host
        cls.p = T.Portal(cls.env.config())
        cls.tok = T.login(cls.p.port, cls.env.key)

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        AC.public_host = cls.saved_pub
        for k, v in cls.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cls.env.cleanup()
        shutil.rmtree(cls.cfg_dir, ignore_errors=True)

    def get(self, path):
        return T.request(self.p.port, "GET", path, T.cookie(self.tok))

    def form_token(self):
        st, _, body = self.get("/admin/")
        self.assertEqual(st, 200)
        return TOKEN_RE.search(body.decode()).group(1)

    def post(self, path, fields, origin=True):
        h = T.cookie(self.tok)
        h["Content-Type"] = FORM
        if origin:
            h["Origin"] = self.p.origin
        return T.request(self.p.port, "POST", path, h, urlencode({"t": self.form_token(), **fields}).encode())


class AddByEmail(AdminBase):
    def test_email_only_creates_no_key_and_says_where_to_sign_in(self):
        type(self).host = "portal.example.org"
        st, _, body = self.post("/admin/people/add", {"id": "eve", "role": "basic", "email": "Eve@Example.org"})
        self.assertEqual(st, 200)
        text = body.decode()
        self.assertIn("Send them https://portal.example.org/", text)
        self.assertIn("eve@example.org (one-time code by email)", text)
        self.assertIn("Email-only people can&#x27;t use the Tailscale link.", text.replace("can't", "can&#x27;t"))
        self.assertNotIn("/login?k=", text)
        p = entry(self.env, "eve")
        self.assertNotIn("key_sha256", p)
        self.assertFalse(A.Roster.usable(p))              # not in key mode
        self.assertTrue(A.Roster.usable_access(p))        # yes in access mode
        self.assertEqual(A.Roster(self.env.roster).match_key("anything"), None)
        audit = PA.audit_path(self.env.state).read_text(encoding="utf-8")
        self.assertIn('"person_added"', audit)
        self.assertIn('"key": false', audit)

    def test_no_host_means_no_url_line(self):
        type(self).host = None
        st, _, body = self.post("/admin/people/add", {"id": "nohost", "role": "basic", "email": "nh@example.org"})
        self.assertEqual(st, 200)
        text = body.decode()
        self.assertNotIn("https://", text.split("<main")[-1] if "<main" in text else text.split("</header>")[-1])
        self.assertNotIn("None", text)
        self.assertIn("nh@example.org (one-time code by email)", text)

    def test_also_key_checkbox_shows_key_page(self):
        st, _, body = self.post("/admin/people/add", {"id": "both", "role": "basic", "email": "both@example.org",
                                                      "also_key": "1"})
        self.assertEqual(st, 200)
        m = re.search(r"/login\?k=([A-Za-z0-9_-]+)", body.decode())
        self.assertTrue(m)
        self.assertTrue(A.Roster.usable(entry(self.env, "both")))
        T.login(self.p.port, m.group(1))

    def test_no_email_is_the_key_flow(self):
        st, _, body = self.post("/admin/people/add", {"id": "plain", "role": "basic"})
        self.assertEqual(st, 200)
        self.assertIn("/login?k=", body.decode())
        self.assertNotIn("email", entry(self.env, "plain"))

    def test_make_key_for_email_only_and_audited(self):
        ADM.add_person(self.env.roster, "mk", "basic", "mk@example.org", with_key=False)
        st, _, body = self.post("/admin/people/key", {"id": "mk"})
        self.assertEqual(st, 200)
        m = re.search(r"/login\?k=([A-Za-z0-9_-]+)", body.decode())
        self.assertTrue(m)
        self.assertEqual(entry(self.env, "mk")["email"], "mk@example.org")
        T.login(self.p.port, m.group(1))                  # the new key signs in
        k1 = m.group(1)
        st, _, body = self.post("/admin/people/key", {"id": "mk"})            # rotate
        k2 = re.search(r"/login\?k=([A-Za-z0-9_-]+)", body.decode()).group(1)
        self.assertNotEqual(k1, k2)
        self.assertIn(T.request(self.p.port, "GET", f"/login?k={k1}")[0], (401, 403))
        log = PA.audit_path(self.env.state).read_text(encoding="utf-8")
        self.assertIn("key_made", log)
        self.assertNotIn(k2, log)

    def test_make_key_refused_for_revoked_and_needs_origin(self):
        ADM.add_person(self.env.roster, "rv", "basic", "rv@example.org", with_key=False)
        ADM.revoke_person(self.env.roster, "rv")
        st, _, body = self.post("/admin/people/key", {"id": "rv"})
        self.assertEqual(st, 404)
        self.assertNotIn("key_sha256", entry(self.env, "rv"))
        ADM.add_person(self.env.roster, "og", "basic", "og@example.org", with_key=False)
        st, _, _ = self.post("/admin/people/key", {"id": "og"}, origin=False)
        self.assertEqual(st, 403)
        self.assertNotIn("key_sha256", entry(self.env, "og"))

    def test_card_has_make_key_button(self):
        ADM.add_person(self.env.roster, "btn", "basic", "btn@example.org", with_key=False)
        _, _, body = self.get("/admin/")
        self.assertIn("Make key link", body.decode())

    def test_without_csrf_token_refused(self):
        ADM.add_person(self.env.roster, "nocsrf", "basic", "nc@example.org", with_key=False)
        h = T.cookie(self.tok)
        h["Content-Type"] = FORM
        h["Origin"] = self.p.origin
        st, _, body = T.request(self.p.port, "POST", "/admin/people/key", h, b"id=nocsrf")
        self.assertIn(st, (403, 400))
        self.assertNotIn("key_sha256", entry(self.env, "nocsrf"))

    def test_also_key_only_accepts_1_or_on(self):
        self.post("/admin/people/add", {"id": "ak", "role": "basic", "email": "ak@example.org", "also_key": "yes"})
        self.assertNotIn("key_sha256", entry(self.env, "ak"))

    def test_malformed_services_config_tolerated(self):
        old = os.environ["CRYOEV_SERVICES_CONFIG"]
        try:
            for text in ('{"services": 5}', '{"services": [1, null, "x"]}', "[1,2]", "not json"):
                self.svc.write_text(text, encoding="utf-8")
                self.assertEqual(ADM._health_paths(), set())
        finally:
            self.svc.write_text(json.dumps({"run_dir": str(self.cfg_dir / "run"), "services": []}), encoding="utf-8")
            os.environ["CRYOEV_SERVICES_CONFIG"] = old


class KeyedAdminGuard(unittest.TestCase):
    """The tailnet is the break-glass door: at least one active admin with a key must always remain."""

    def setUp(self):
        self.env = T.Env()                                   # tester: the keyed admin
        ADM.add_person(self.env.roster, "boss2", "admin", "boss2@example.org", with_key=False)

    def tearDown(self):
        self.env.cleanup()

    def test_sole_keyed_admin_cannot_demote_or_revoke(self):
        with self.assertRaises(ADM.Refused) as c:
            ADM.set_role(self.env.roster, "tester", "basic")
        self.assertIn("key link", str(c.exception))
        with self.assertRaises(ADM.Refused):
            ADM.revoke_person(self.env.roster, "tester")
        self.assertIn("key link", ADM.revoke_check(self.env.roster, "tester"))
        self.assertEqual(entry(self.env, "tester")["role"], "admin")
        self.assertNotIn("revoked", entry(self.env, "tester"))

    def test_email_only_admin_cannot_lose_email(self):
        with self.assertRaises(ADM.Refused):
            ADM.set_email(self.env.roster, "boss2", "")

    def test_allowed_once_another_admin_has_a_key(self):
        ADM.make_key(self.env.roster, "boss2")
        self.assertEqual(ADM.set_role(self.env.roster, "tester", "basic"), "admin")
        with self.assertRaises(ADM.Refused):                  # boss2 is now the sole keyed admin
            ADM.revoke_person(self.env.roster, "boss2")

    def test_email_only_admin_can_be_revoked_freely(self):
        self.assertEqual(ADM.revoke_person(self.env.roster, "boss2"), "admin")


class KeysTailnetOnly(unittest.TestCase):
    def test_access_mode_hides_and_refuses_key_minting(self):
        env = T.Env()
        try:
            TA.roster_people(env)
            p = TA.access_portal(env, TA.Fetcher(), admin=True)
            try:
                st, _, page = TA.get(p.port, "/admin/", TA.sign("boss@example.org"))
                self.assertEqual(st, 200)
                self.assertNotIn(b"Make key link", page)
                self.assertNotIn(b"Make a new key link", page)
                self.assertNotIn(b"also_key", page)
                tok = TOKEN_RE.search(page.decode()).group(1)
                body = urlencode({"t": tok, "id": "adv"}).encode()
                st, _, out = TA.get(p.port, "/admin/people/key", TA.sign("boss@example.org"),
                                    {"Origin": TA.ORIGIN, "Content-Type": FORM}, method="POST", body=body)
                self.assertEqual(st, 403)
                self.assertIn(b"Use the Tailscale admin page to make key links.", out)
            finally:
                p.stop()
        finally:
            env.cleanup()

    def test_key_mode_shows_them(self):
        env = T.Env()
        try:
            p = T.Portal(env.config())
            try:
                tok = T.login(p.port, env.key)
                _, _, page = T.request(p.port, "GET", "/admin/", T.cookie(tok))
                self.assertIn(b"also_key", page)
            finally:
                p.stop()
        finally:
            env.cleanup()


class AccessSignIn(unittest.TestCase):
    def test_email_only_signs_in_through_access_not_key(self):
        env = T.Env()
        try:
            ADM.add_person(env.roster, "newbie", "basic", "newbie@example.org", with_key=False)
            p = TA.access_portal(env, TA.Fetcher())
            try:
                st, _, body = TA.get(p.port, "/cj/me.json", TA.sign("newbie@example.org"))
                self.assertEqual((st, json.loads(body)["id"]), (200, "newbie"))
            finally:
                p.stop()
            self.assertIsNone(A.Roster(env.roster).match_key(""))
            self.assertFalse(A.Roster.usable(entry(env, "newbie")))
        finally:
            env.cleanup()


class AccessConfigTests(unittest.TestCase):
    def test_public_host_file_and_env_override(self):
        d = Path(tempfile.mkdtemp(prefix="cj_p5ac_"))
        try:
            f = d / "cloudflare_access.json"
            f.write_text(json.dumps({"team": "example-team", "aud": "x" * 12, "host": "Portal.Example.org"}))
            self.assertEqual(AC.public_host(f, env={}), "portal.example.org")
            self.assertEqual(AC.public_host(f, env={"CJ_ACCESS_HOST": "other.example.org"}), "other.example.org")
            self.assertIsNone(AC.public_host(d / "missing.json", env={}))
            self.assertEqual(AC.public_host(d / "missing.json", env={"CJ_ACCESS_HOST": "env.example.org"}),
                             "env.example.org")
            f.write_text(json.dumps({"host": "https://bad/path"}))
            self.assertIsNone(AC.public_host(f, env={}))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_module_has_no_jwt_import(self):
        src = Path(AC.__file__).read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"^\s*(import|from)\s+jwt", src, re.M))


class HealthDisplay(unittest.TestCase):
    def render(self, rows):
        h = {"generated_utc": "2026-11-06T10:00:00+00:00", "services": {"ok": True, "rows": rows},
             "watchers": {"ok": False, "problem": "x"},
             "backup": {"commit": {"problem": "x"}, "task": {"found": False, "problem": "x"}}}
        return ADM.render_health(h)

    def row(self, name, health, path):
        return {"name": name, "state": "running", "port": 1, "health": health, "health_path": path, "uptime_s": 5}

    def test_http_error_with_health_path_is_a_problem(self):
        out = self.render([self.row("portal", "http 503", True)])
        self.assertIn("problem: http 503", out)
        self.assertNotIn("answers", out)
        self.assertIn("Running, unhealthy", out)

    def test_review_server_without_health_path_still_up(self):          # control
        out = self.render([self.row("review", "http 404", False)])
        self.assertIn("answers", out)
        self.assertNotIn("problem", out)
        self.assertNotIn("unhealthy", out)

    def test_any_non_ok_with_health_path_is_unhealthy(self):
        for hs in ("bad body", "tcp", "http 500"):
            out = self.render([self.row("portal", hs, True)])
            self.assertIn(f"problem: {hs}", out, hs)
            self.assertIn("unhealthy", out)

    def test_ok_stays_ok(self):
        out = self.render([self.row("portal", "ok", True)])
        self.assertIn("ok", out)
        self.assertNotIn("problem", out)

    def test_health_paths_from_services_config(self):
        d = Path(tempfile.mkdtemp(prefix="cj_p5h_"))
        old = os.environ.get("CRYOEV_SERVICES_CONFIG")
        try:
            f = d / "services.json"
            f.write_text(json.dumps({"services": [{"name": "a", "health": "/healthz"}, {"name": "b"}]}))
            os.environ["CRYOEV_SERVICES_CONFIG"] = str(f)
            self.assertEqual(ADM._health_paths(), {"a"})
        finally:
            if old is None:
                os.environ.pop("CRYOEV_SERVICES_CONFIG", None)
            else:
                os.environ["CRYOEV_SERVICES_CONFIG"] = old
            shutil.rmtree(d, ignore_errors=True)


def make_pkg(root: Path, ps: str, owner: str, n_images: int, checked: int, edited: int = 0, unchecked_files: int = 0,
             keyed: bool = True):
    """review/pred_<ps>/ with review_package.json and edits: `checked` Reviewed images, `edited` images with a
    decision, `unchecked_files` edit files holding nothing."""
    d = root / f"pred_{ps}"
    (d / "edits").mkdir(parents=True)
    (d / "review_package.json").write_text(json.dumps({"owner": owner, "n_images": n_images, "prediction_set": ps}))
    if keyed:
        (d / "data.js").write_text("window.DATA={}")
        (d / ".review_key").write_text("k" * 20)
    i = 0
    for _ in range(checked):
        (d / "edits" / f"img{i}.json").write_text(json.dumps({"edit": {"reviewed": True, "dec": {}}}))
        i += 1
    for _ in range(edited):
        (d / "edits" / f"img{i}.json").write_text(json.dumps({"edit": {"dec": {"c1": {"v": "accept", "t": 1}}}}))
        i += 1
    for _ in range(unchecked_files):
        (d / "edits" / f"img{i}.json").write_text(json.dumps({"edit": {"dec": {}}}))
        i += 1
    return d


class MyReviews(unittest.TestCase):
    """The Reviews hub (round 13; was My reviews) on the P5 fixture packages."""

    @classmethod
    def setUpClass(cls):
        from contextlib import ExitStack
        import test_reviews_hub as RH
        cls.stack = ExitStack()
        RH.fake_listing(cls.stack)                     # only while portal/listing.py is the contract stub
        cls.env = T.Env()
        cls.tmp = Path(tempfile.mkdtemp(prefix="cj_p5rev_"))
        cls.review, cls.predict = cls.tmp / "review", cls.tmp / "predict"
        cls.review.mkdir()
        cls.predict.mkdir()
        cls.keys = {"admin": cls.env.key, "bas": A.add_key(cls.env.roster, "bas", "basic"),
                    "other": A.add_key(cls.env.roster, "other", "basic"),
                    "adv": A.add_key(cls.env.roster, "adv", "advanced")}
        make_pkg(cls.review, "ps_20261101_aaaaa1", "bas", n_images=10, checked=2, edited=1, unchecked_files=3)
        make_pkg(cls.review, "ps_20261102_bbbbb2", "other", n_images=5, checked=5)
        make_pkg(cls.review, "ps_20261103_ccccc3", "tester", n_images=4, checked=0)
        make_pkg(cls.review, "ps_20261104_ddddd4", "bas", n_images=3, checked=1, keyed=False)    # no key: broken
        (cls.predict / "ps_20261101_aaaaa1").mkdir()
        (cls.predict / "ps_20261101_aaaaa1" / "request.json").write_text(json.dumps({"name": "My <b>EV</b> run"}))
        (cls.predict / "ps_20261102_bbbbb2").mkdir()
        (cls.predict / "ps_20261102_bbbbb2" / "request.json").write_text(json.dumps({"name": "SecretOtherName"}))
        cfg = cls.env.config()
        cfg.review_root, cfg.predict_root = cls.review, cls.predict
        cls.p = T.Portal(cfg)
        cls.tok = {k: T.login(cls.p.port, v) for k, v in cls.keys.items()}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        cls.stack.close()

    def get(self, path, who):
        st, _, body = T.request(self.p.port, "GET", path, T.cookie(self.tok[who]))
        return st, body.decode()

    def test_basic_sees_only_own(self):
        st, text = self.get("/reviews/", "bas")
        self.assertEqual(st, 200)
        self.assertIn("My &lt;b&gt;EV&lt;/b&gt; run", text)
        self.assertIn("/reviews/go/pred_ps_20261101_aaaaa1", text)
        self.assertIn("10 images", text)
        for hidden in ("ps_20261102_bbbbb2", "SecretOtherName", "other", "ps_20261103_ccccc3", "tester"):
            self.assertNotIn(hidden, text.split("<main")[-1] if "<main" in text else text)
        # the other user sees theirs, not bas's
        st, text = self.get("/reviews/", "other")
        self.assertIn("ps_20261102_bbbbb2", text)
        self.assertNotIn("ps_20261101_aaaaa1", text)

    def test_broken_package_not_listed(self):
        for who, path in (("bas", "/reviews/"), ("admin", "/reviews/?show=all")):
            self.assertNotIn("ddddd4", self.get(path, who)[1], who)

    def test_basic_cannot_widen_with_show_all(self):
        st, text = self.get("/reviews/?show=all", "bas")
        self.assertEqual(st, 200)
        self.assertNotIn("ps_20261102_bbbbb2", text)
        self.assertNotIn("SecretOtherName", text)
        self.assertNotIn("Mine</a>", text)                            # Mine | All is admin only

    def test_admin_mine_and_all(self):
        st, mine = self.get("/reviews/", "admin")
        self.assertEqual(st, 200)
        self.assertIn("pred_ps_20261103_ccccc3", mine.replace("/review/", " "))
        self.assertNotIn("ps_20261101_aaaaa1", mine)
        self.assertNotIn("ps_20261102_bbbbb2", mine)
        self.assertIn("Mine</a>", mine)
        st, every = self.get("/reviews/?show=all", "admin")
        for want in ("ps_20261101_aaaaa1", "ps_20261102_bbbbb2", "ps_20261103_ccccc3", "SecretOtherName"):
            self.assertIn(want, every)
        self.assertIn(">other<", every)
        self.assertIn(">bas<", every)

    def test_all_roles_200_and_gate(self):
        for who in ("admin", "adv", "bas", "other"):
            self.assertEqual(self.get("/reviews/", who)[0], 200, who)
        self.assertEqual(T.request(self.p.port, "GET", "/reviews/")[0] in (401, 303, 302), True)
        # /review/ (the real package) is still admin-only; /reviews/ does not match its owner-checked rule
        from portal import app as APP
        self.assertIsNone(APP.OWNER_PATH_RE.match("/reviews/"))
        self.assertIsNone(APP.OWNER_PATH_RE.match("/reviews"))
        self.assertEqual(self.get("/review/", "bas")[0], 403)

    def test_nav_shows_reviews_for_all_roles(self):
        for who in ("admin", "adv", "bas"):
            st, _, body = T.request(self.p.port, "GET", "/cj/header.js", T.cookie(self.tok[who]))
            st2, text = self.get("/", who)
            st3, _, mods = T.request(self.p.port, "GET", "/cj/modules.json", T.cookie(self.tok[who]))
            self.assertIn('"/reviews/"', mods.decode(), who)
            self.assertIn("reviews", (mods.decode() + text + body.decode()).lower(), who)
        _, page = self.get("/reviews/", "bas")
        self.assertIn("<title>Reviews</title>", page)

    def test_cache_follows_edits_folder_mtime(self):
        d = self.review / "pred_ps_20261101_aaaaa1"
        before = REV.edit_stats(d)[0]
        (d / "edits" / "new.json").write_text(json.dumps({"edit": {"reviewed": True}}))
        os.utime(d / "edits", (time.time() + 5, time.time() + 5))
        self.assertEqual(REV.edit_stats(d)[0], before + 1)
        (d / "edits" / "new.json").unlink()
        os.utime(d / "edits", (time.time() + 10, time.time() + 10))
        self.assertEqual(REV.edit_stats(d)[0], before)


if __name__ == "__main__":
    unittest.main()
