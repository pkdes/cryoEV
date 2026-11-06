"""
Admin module (portal/modules/admin.py): roles through the Gate, add a person (key link shown once, never again,
works for sign-in), role change on the next request, revoke ending sessions, last-admin protection, the form token
and Origin check, the audit log (lines, viewer, escaping, the helper), no key or hash in any page / JSON / log /
audit line, the health panels with every tool missing and with known answers (a temp services config, a temp
backup clone with a fixed commit time, a schtasks CSV), and the pages in a browser (Playwright, Edge channel: 390
and 1280 px, light and dark, no horizontal scroll, same-origin requests only, no console errors). Screenshots go to
$CJ_SHOT_DIR when set (else a temp folder, removed afterwards).

Temp roster / state dir / registry / services config / backup clone only (cjtest.Env); never the real ones.

  python -m unittest discover -s portal/tests -v
"""
import html
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import admin as ADM  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

FORM = "application/x-www-form-urlencoded"
TOKEN_RE = re.compile(r'name="t" value="([0-9a-f]{40})"')
LINK_RE = re.compile(r"/login\?k=([A-Za-z0-9_-]+)")


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):                     # Handler.handle already holds self.lock
        self.lines.append(record.getMessage())


def roster_doc(env) -> dict:
    return json.loads(env.roster.read_text(encoding="utf-8"))


def person(env, pid) -> dict:
    return [p for p in roster_doc(env)["people"] if p["id"] == pid][-1]


class Base(unittest.TestCase):
    """A portal on a temp Env with an admin (tester), an advanced and a basic person. Health runs against a temp
    services config, no backup clone and a task name that does not exist (never the real ones)."""

    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.adv_key = A.add_key(cls.env.roster, "adv", "advanced")
        cls.bas_key = A.add_key(cls.env.roster, "bas", "basic")
        cls.svc_cfg = cls.env.tmp / "services.json"
        cls.svc_cfg.write_text(json.dumps({"run_dir": str(cls.env.tmp / "run"), "services": []}), encoding="utf-8")
        cls.saved_env = {k: os.environ.get(k) for k in ("CRYOEV_SERVICES_CONFIG", "CJ_BACKUP_REPO")}
        os.environ["CRYOEV_SERVICES_CONFIG"] = str(cls.svc_cfg)
        os.environ.pop("CJ_BACKUP_REPO", None)
        cls.saved_mod = {k: getattr(ADM, k) for k in ("SERVICES_SCRIPT", "GIT", "SCHTASKS", "BACKUP_TASK",
                                                      "HEALTH_TTL_S")}
        ADM.BACKUP_TASK = "cj-test-no-such-task-c1"
        ADM.HEALTH_TTL_S = 0
        cls.p = T.Portal(cls.env.config())
        cls.tok = {"admin": T.login(cls.p.port, cls.env.key), "advanced": T.login(cls.p.port, cls.adv_key),
                   "basic": T.login(cls.p.port, cls.bas_key)}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        for k, v in cls.saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for k, v in cls.saved_mod.items():
            setattr(ADM, k, v)
        ADM._health_cache.clear()
        cls.env.cleanup()

    def get(self, path, tok=None, role="admin"):
        return T.request(self.p.port, "GET", path, T.cookie(tok or self.tok[role]))

    def form_token(self, tok=None) -> str:
        st, _, body = self.get("/admin/", tok)
        self.assertEqual(st, 200)
        return TOKEN_RE.search(body.decode()).group(1)

    def post(self, path, fields, tok=None, origin=True, token=True, ctype=FORM, body=None):
        tok = tok or self.tok["admin"]
        f = dict(fields)
        if token is True:
            f["t"] = self.form_token(tok)
        elif token:
            f["t"] = token
        h = {**T.cookie(tok), "Content-Type": ctype}
        if origin is True:
            h["Origin"] = self.p.origin
        elif origin:
            h["Origin"] = origin
        data = body if body is not None else urlencode(f).encode()
        return T.request(self.p.port, "POST", path, h, data)

    def add(self, pid, role="basic", tok=None) -> str:
        """-> the key from the key link on the answer page."""
        st, h, body = self.post("/admin/people/add", {"id": pid, "role": role}, tok)
        self.assertEqual(st, 200, body[:400])
        self.assertEqual(T.hget(h, "cache-control"), "no-store")
        return LINK_RE.search(body.decode()).group(1)


class Roles(Base):
    PATHS = ("/admin/", "/admin/health", "/admin/audit", "/admin/admin.js", "/admin/api/people",
             "/admin/api/health", "/admin/api/audit", "/admin/people/revoke?id=bas")

    def test_basic_and_advanced_get_403(self):
        for role in ("basic", "advanced"):
            for path in self.PATHS:
                with self.subTest(role=role, path=path):
                    st, _, body = self.get(path, role=role)
                    self.assertEqual(st, 403)
                    self.assertIn(b"Your role can&#x27;t open this part", body)
        before = self.env.roster.read_bytes()
        for role in ("basic", "advanced"):
            for path in ("/admin/people/add", "/admin/people/role", "/admin/people/revoke"):
                st, _, _ = T.request(self.p.port, "POST", path,
                                     {**T.cookie(self.tok[role]), "Origin": self.p.origin, "Content-Type": FORM},
                                     urlencode({"id": "bas", "role": "admin", "confirm": "yes"}).encode())
                self.assertEqual(st, 403, (role, path))
        self.assertEqual(self.env.roster.read_bytes(), before)

    def test_no_session_401_and_admin_200(self):
        st, _, _ = T.request(self.p.port, "GET", "/admin/")
        self.assertEqual(st, 401)
        st, h, _ = self.get("/admin")
        self.assertEqual((st, T.hget(h, "location")), (307, "/admin/"))
        for path in self.PATHS:
            st, h, _ = self.get(path)
            self.assertEqual(st, 200, path)
            if path != "/admin/admin.js":
                self.assertEqual(T.hget(h, "cache-control"), "no-store", path)


class People(Base):
    def test_list_shows_people_never_keys(self):
        st, _, body = self.get("/admin/api/people")
        rows = {r["id"]: r for r in json.loads(body)["people"]}
        self.assertEqual({k: (v["role"], v["state"]) for k, v in rows.items() if k in ("tester", "adv", "bas")},
                         {"tester": ("admin", "active"), "adv": ("advanced", "active"), "bas": ("basic", "active")})
        self.assertEqual(set(rows["adv"]), {"id", "role", "state", "added", "revoked", "role_changed"})
        page = self.get("/admin/")[2].decode()
        for pid in ("tester", "adv", "bas"):
            self.assertIn(f">{pid}", page)
        self.assertIn("(you)", page)
        self.assertNotIn("onclick", page.lower())
        self.assertNotIn("confirm(", page)
        self.assertEqual(re.findall(r"<script[^>]*>", page), ['<script src="/cj/header.js" defer>',
                                                              '<script src="/admin/admin.js" defer>'])

    def test_add_shows_key_once_and_key_signs_in(self):
        st, h, body = self.post("/admin/people/add", {"id": "newbie", "role": "basic"})
        self.assertEqual(st, 200)
        self.assertEqual(T.hget(h, "cache-control"), "no-store")
        text = body.decode()
        key = LINK_RE.search(text).group(1)
        self.assertIn(f"{self.p.origin}/login?k={key}", text)        # the link uses the portal's own origin
        self.assertIn("Shown only now", text)
        self.assertEqual(person(self.env, "newbie")["key_sha256"], A.sha256_hex(key))
        self.assertNotIn(key, self.env.roster.read_text(encoding="utf-8"))
        # a refresh re-sends the form: refused, and no key on the page
        st, h, body = self.post("/admin/people/add", {"id": "newbie", "role": "basic"})
        self.assertEqual(st, 409)
        self.assertNotIn(b"login?k=", body)
        self.assertIn(b"already on the list", body)
        self.assertEqual(person(self.env, "newbie")["key_sha256"], A.sha256_hex(key))   # not re-keyed
        for path in ("/admin/", "/admin/api/people", "/admin/audit", "/admin/api/audit"):
            b = self.get(path)[2]
            self.assertNotIn(key.encode(), b, path)
            self.assertNotIn(b"login?k=", b, path)
        # the key works (cjtest login), as a basic person
        tok = T.login(self.p.port, key)
        self.assertEqual(self.get("/", tok)[0], 200)
        self.assertEqual(json.loads(self.get("/cj/me.json", tok)[2]), {"id": "newbie", "role": "basic"})
        self.assertEqual(self.get("/admin/", tok)[0], 403)

    def test_add_validation(self):
        before = self.env.roster.read_bytes()
        for fields, status in (({"id": "Bad Id!", "role": "basic"}, 400), ({"id": "", "role": "basic"}, 400),
                               ({"id": "<img src=x onerror=alert(1)>", "role": "basic"}, 400),
                               ({"id": "okid", "role": "root"}, 400), ({"id": "tester", "role": "basic"}, 409)):
            with self.subTest(fields=fields):
                st, _, body = self.post("/admin/people/add", fields)
                self.assertEqual(st, status)
                self.assertNotIn(b"login?k=", body)
                self.assertNotIn(b"<img src=x", body)
        st, _, _ = self.post("/admin/people/add", {"id": "x1", "role": "basic"}, ctype="multipart/form-data")
        self.assertEqual(st, 415)
        st, _, _ = self.post("/admin/people/add", {}, body=b"id=" + b"a" * 9000)
        self.assertEqual(st, 413)
        self.assertEqual(self.env.roster.read_bytes(), before)

    def test_add_upper_case_id_is_lowered(self):
        key = self.add("CaseMe")
        self.assertEqual(person(self.env, "caseme")["key_sha256"], A.sha256_hex(key))

    def test_role_change_takes_effect_next_request(self):
        key = self.add("rolly", "basic")
        tok = T.login(self.p.port, key)
        self.assertEqual(self.get("/admin/", tok)[0], 403)
        st, h, _ = self.post("/admin/people/role", {"id": "rolly", "role": "admin"})
        self.assertEqual((st, T.hget(h, "location")), (303, "/admin/?done=role&id=rolly"))
        self.assertEqual(person(self.env, "rolly")["key_sha256"], A.sha256_hex(key))    # same key
        self.assertEqual(self.get("/admin/", tok)[0], 200)                               # same session
        self.assertIn(b"Role changed for rolly", self.get("/admin/?done=role&id=rolly")[2])
        st, _, _ = self.post("/admin/people/role", {"id": "rolly", "role": "basic"})
        self.assertEqual(st, 303)
        self.assertEqual(self.get("/admin/", tok)[0], 403)
        self.assertEqual(self.get("/", tok)[0], 200)
        st, _, body = self.post("/admin/people/role", {"id": "rolly", "role": "basic"})
        self.assertEqual(st, 409)
        st, _, _ = self.post("/admin/people/role", {"id": "nobody", "role": "basic"})
        self.assertEqual(st, 404)
        st, _, _ = self.post("/admin/people/role", {"id": "rolly", "role": "root"})
        self.assertEqual(st, 400)

    def test_role_change_same_size_back_and_forth(self):
        """admin <-> basic keeps the file size; the Roster cache must still see every change at once."""
        key = self.add("flip", "basic")
        tok = T.login(self.p.port, key)
        for i in range(4):
            new = "admin" if i % 2 == 0 else "basic"
            self.assertEqual(self.post("/admin/people/role", {"id": "flip", "role": new})[0], 303)
            self.assertEqual(self.get("/admin/", tok)[0], 200 if new == "admin" else 403, i)

    def test_revoke_confirm_then_sessions_end(self):
        key = self.add("gone1", "advanced")
        tok = T.login(self.p.port, key)
        self.assertEqual(self.get("/", tok)[0], 200)
        st, _, body = self.get("/admin/people/revoke?id=gone1")
        self.assertEqual(st, 200)
        text = body.decode()
        self.assertIn("Yes, revoke gone1", text)
        self.assertIn('name="confirm" value="yes"', text)
        self.assertNotIn("confirm(", text)
        # without confirm=yes: the confirmation page again, nothing changed
        st, _, body = self.post("/admin/people/revoke", {"id": "gone1"})
        self.assertEqual(st, 200)
        self.assertIn(b"Yes, revoke gone1", body)
        self.assertNotIn("revoked", person(self.env, "gone1"))
        st, h, _ = self.post("/admin/people/revoke", {"id": "gone1", "confirm": "yes"})
        self.assertEqual((st, T.hget(h, "location")), (303, "/admin/?done=revoked&id=gone1"))
        self.assertEqual(self.get("/", tok)[0], 403)                       # the session ended
        st, _, _ = T.request(self.p.port, "GET", f"/login?k={key}")
        self.assertEqual(st, 403)                                         # the key no longer works
        rows = {r["id"]: r for r in json.loads(self.get("/admin/api/people")[2])["people"]}
        self.assertEqual(rows["gone1"]["state"], "revoked")
        st, _, body = self.post("/admin/people/revoke", {"id": "gone1", "confirm": "yes"})
        self.assertEqual(st, 404)
        st, _, _ = self.get("/admin/people/revoke?id=%3Cscript%3E")
        self.assertEqual(st, 404)
        # re-adding a revoked id gives a new key
        key2 = self.add("gone1", "basic")
        self.assertNotEqual(key, key2)
        self.assertEqual(self.get("/", T.login(self.p.port, key2))[0], 200)

    def test_csrf_and_origin(self):
        before = self.env.roster.read_bytes()
        fields = {"id": "csrf1", "role": "admin"}
        self.assertEqual(self.post("/admin/people/add", fields, origin=None)[0], 403)              # Gate: no Origin
        self.assertEqual(self.post("/admin/people/add", fields, origin="https://evil.example")[0], 403)
        st, _, body = self.post("/admin/people/add", fields, token=None)                           # no form token
        self.assertEqual(st, 403)
        self.assertIn(b"expired", body)
        self.assertEqual(self.post("/admin/people/add", fields, token="0" * 40)[0], 403)
        other = T.login(self.p.port, self.env.key)                       # another admin session's token
        self.assertEqual(self.post("/admin/people/add", fields, token=self.form_token(other))[0], 403)
        self.assertEqual(self.env.roster.read_bytes(), before)
        self.assertEqual(T.request(self.p.port, "GET", "/admin/people/add", T.cookie(self.tok["admin"]))[0], 405)


class LastAdmin(Base):
    def test_last_admin_protection(self):
        before = self.env.roster.read_bytes()
        st, _, body = self.post("/admin/people/role", {"id": "tester", "role": "basic"})
        self.assertEqual(st, 409)
        self.assertIn(b"last admin", body)
        st, _, body = self.post("/admin/people/revoke", {"id": "tester", "confirm": "yes"})
        self.assertEqual(st, 409)
        st, _, body = self.get("/admin/people/revoke?id=tester")
        self.assertEqual(st, 409)
        self.assertIn(b"last admin", body)
        self.assertNotIn(b"Yes, revoke", body)
        page = self.get("/admin/")[2]
        self.assertIn(b"can't be demoted or revoked", page)
        self.assertNotIn(b"/admin/people/revoke?id=tester", page)          # no actions offered on the last admin
        self.assertEqual(self.env.roster.read_bytes(), before)
        # a revoked admin does not count
        self.add("ex", "admin")
        self.assertEqual(self.post("/admin/people/revoke", {"id": "ex", "confirm": "yes"})[0], 303)
        self.assertEqual(self.post("/admin/people/role", {"id": "tester", "role": "basic"})[0], 409)
        # with a second admin, demoting yourself is allowed and sends you home
        key2 = self.add("admin2", "admin")
        tok2 = T.login(self.p.port, key2)
        st, h, _ = self.post("/admin/people/role", {"id": "tester", "role": "advanced"})
        self.assertEqual((st, T.hget(h, "location")), (303, "/"))
        self.assertEqual(self.get("/admin/")[0], 403)
        # now admin2 is the last admin: can't revoke themselves
        st, _, _ = self.post("/admin/people/revoke", {"id": "admin2", "confirm": "yes"}, tok2)
        self.assertEqual(st, 409)
        self.assertEqual(self.get("/admin/", tok2)[0], 200)
        # admin2 restores tester; then admin2 revokes themselves: signed out
        self.assertEqual(self.post("/admin/people/role", {"id": "tester", "role": "admin"}, tok2)[0], 303)
        st, h, _ = self.post("/admin/people/revoke", {"id": "admin2", "confirm": "yes"}, tok2)
        self.assertEqual((st, T.hget(h, "location")), (303, "/"))
        self.assertEqual(self.get("/", tok2)[0], 403)
        self.assertEqual(self.get("/admin/")[0], 200)


class Audit(Base):
    def test_audit_lines_viewer_and_helper(self):
        self.add("aud1", "basic")
        self.assertEqual(self.post("/admin/people/role", {"id": "aud1", "role": "advanced"})[0], 303)
        self.assertEqual(self.post("/admin/people/revoke", {"id": "aud1", "confirm": "yes"})[0], 303)
        path = self.env.state / "audit" / "portal.jsonl"
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(r["actor"], r["action"], r["target"], r["detail"]) for r in lines[-3:]],
                         [("tester", "person_added", "aud1", {"role": "basic"}),
                          ("tester", "role_changed", "aud1", {"from": "basic", "to": "advanced"}),
                          ("tester", "person_revoked", "aud1", {"role": "advanced"})])
        for r in lines:
            self.assertTrue(r["t"].endswith("+00:00"))
            datetime.fromisoformat(r["t"])
        self.assertFalse(str(self.env.state.resolve()).startswith(str(self.env.registry.resolve())))
        self.assertFalse(any(self.env.registry.rglob("*.jsonl")))         # never the registry
        # the helper, as another module would call it (ctx-like), with markup to escape
        ctx = types.SimpleNamespace(cfg=self.env.config())
        rec = ADM.audit(ctx, "tester", "model_promoted", "<script>alert(1)</script>", {"why": "<b>x</b>"})
        self.assertEqual(rec["target"], "<script>alert(1)</script>")
        st, _, body = self.get("/admin/audit")
        text = body.decode()
        self.assertEqual(st, 200)
        self.assertNotIn("<script>alert", text)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", text)
        self.assertLess(text.index("model promoted"), text.index("revoked"))     # newest first
        api = json.loads(self.get("/admin/api/audit")[2])
        self.assertEqual(api["entries"][0]["action"], "model_promoted")
        self.assertEqual(api["entries"][1]["action"], "person_revoked")
        # append only: earlier lines are unchanged byte for byte
        self.assertTrue(path.read_text(encoding="utf-8").startswith(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines)))

    def test_viewer_tail_and_bad_lines(self):
        cfg = types.SimpleNamespace(state_dir=self.env.tmp / "auditonly")
        p = ADM.audit_path(cfg)
        p.parent.mkdir(parents=True)
        p.write_text("not json\n" + "".join(json.dumps({"t": "2026-11-05T00:00:00+00:00", "actor": "a",
                                                         "action": "x", "target": str(i), "detail": None}) + "\n"
                                             for i in range(250)) + "{broken\n", encoding="utf-8")
        entries, prob = ADM.read_audit(cfg)
        self.assertIsNone(prob)
        self.assertEqual(len(entries), 200)
        self.assertEqual((entries[0]["target"], entries[-1]["target"]), ("249", "50"))
        self.assertEqual(ADM.read_audit(types.SimpleNamespace(state_dir=self.env.tmp / "none")), ([], None))


class NoSecrets(Base):
    def test_no_key_or_hash_anywhere(self):
        cap = Capture()
        cj = logging.getLogger("cj")
        old_level = cj.level
        cj.setLevel(logging.DEBUG)
        root = logging.getLogger()
        root.addHandler(cap)
        self.addCleanup(root.removeHandler, cap)
        self.addCleanup(cj.setLevel, old_level)
        seen, key_pages = [], []
        orig = T.request

        def rec(*a, **k):
            st, h, body = orig(*a, **k)
            seen.append((a[2], "\n".join(f"{x}: {y}" for x, y in h).encode() + b"\n" + body))
            return st, h, body
        T.request = rec
        self.addCleanup(setattr, T, "request", orig)
        keys = []
        for pid, role in (("s1", "basic"), ("s2", "advanced")):
            k = self.add(pid, role)
            keys.append(k)
            key_pages.append(len(seen) - 1)
        tok = T.login(self.p.port, keys[0])
        self.post("/admin/people/role", {"id": "s1", "role": "admin"})
        self.get("/admin/", tok)
        self.post("/admin/people/revoke", {"id": "s2", "confirm": "yes"})
        self.post("/admin/people/add", {"id": "s1", "role": "basic"})            # refused re-add
        for path in ("/admin/", "/admin/health", "/admin/audit", "/admin/api/people", "/admin/api/health",
                     "/admin/api/audit", "/admin/people/revoke?id=s1", "/cj/me.json", "/"):
            self.get(path)
        T.request = orig
        hashes = [p["key_sha256"] for p in roster_doc(self.env)["people"]]
        secrets_all = keys + [self.env.key, self.adv_key, self.bas_key] + list(self.tok.values()) + [tok]
        for i, (path, blob) in enumerate(seen):
            for hsh in hashes:
                self.assertNotIn(hsh.encode(), blob, path)
            if path.startswith("/login?"):
                continue                                        # sign-in sets the session cookie, by design
            for s in secrets_all:
                if i in key_pages and s in keys and path == "/admin/people/add":
                    continue                                    # the one answer that shows it
                self.assertNotIn(s.encode(), blob, (path, i))
        for i in key_pages:                                     # each key page shows only its own key
            self.assertEqual(sum(k.encode() in seen[i][1] for k in keys), 1)
        audit_text = (self.env.state / "audit" / "portal.jsonl").read_text(encoding="utf-8")
        logs = "\n".join(cap.lines)
        self.assertIn("/admin/people/add", logs)                # the access log did record the requests
        for s in secrets_all + hashes:
            self.assertNotIn(s, audit_text)
            self.assertNotIn(s, logs)


def git_repo_with_commit(root: Path, when_iso: str) -> Path:
    repo = root / "backup_clone"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_DATE": when_iso, "GIT_COMMITTER_DATE": when_iso,
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=env)
    (repo / "MANIFEST.md5").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "backup 20261104T100000Z: 3 changed"],
                   check=True, env=env)
    return repo


SCHTASKS_CSV = ('"HostName","TaskName","Next Run Time","Status","Logon Mode","Last Run Time","Last Result",'
                '"Author","Task To Run","Scheduled Task State"\r\n'
                '"HOME-LAPTOP","\\cryoai-state-backup","11/6/2026 2:00:00 AM","Ready","Interactive only",'
                '"11/5/2026 2:00:01 AM","{res}","u","python backup_state.py","Enabled"\r\n')


class Health(Base):
    def setUp(self):
        ADM._health_cache.clear()
        for k in ("SERVICES_SCRIPT", "GIT", "SCHTASKS"):
            self.addCleanup(setattr, ADM, k, getattr(ADM, k))
        self.addCleanup(os.environ.pop, "CJ_BACKUP_REPO", None)
        live = self.env.registry / "live" / "live.json"
        if live.exists():
            keep = live.read_bytes()
            self.addCleanup(live.write_bytes, keep)
        else:
            self.addCleanup(lambda: live.unlink(missing_ok=True))

    def health(self) -> dict:
        st, h, body = self.get("/admin/api/health")
        self.assertEqual(st, 200)
        self.assertEqual(T.hget(h, "cache-control"), "no-store")
        page = self.get("/admin/health")
        self.assertEqual(page[0], 200)
        return json.loads(body)

    def test_every_tool_missing(self):
        ADM.SERVICES_SCRIPT = self.env.tmp / "no_such_services.py"
        ADM.GIT = "cj-no-such-git-c1"
        ADM.SCHTASKS = "cj-no-such-schtasks-c1"
        os.environ["CJ_BACKUP_REPO"] = str(self.env.tmp)          # a folder, but git is missing
        (self.env.registry / "live" / "live.json").unlink(missing_ok=True)
        h = self.health()
        self.assertFalse(h["services"]["ok"])
        self.assertIn("not on this computer", h["services"]["problem"])
        self.assertFalse(h["watchers"]["ok"])
        self.assertIn("job monitor is not running", h["watchers"]["problem"])
        self.assertIsNone(h["backup"]["commit"]["time"])
        self.assertIn("Unknown", h["backup"]["commit"]["problem"])
        self.assertFalse(h["backup"]["task"]["found"])
        page = self.get("/admin/health")[2].decode()
        self.assertIn("Servers on this laptop", page)
        self.assertIn("Unknown", page)

    def test_not_configured(self):
        ADM.SCHTASKS = "cj-no-such-schtasks-c1"
        h = self.health()
        self.assertEqual(h["services"], {"ok": True, "rows": [], "problem": None})      # empty temp config
        self.assertFalse(h["backup"]["commit"]["configured"])
        self.assertIn("CJ_BACKUP_REPO", h["backup"]["commit"]["problem"])
        os.environ["CJ_BACKUP_REPO"] = str(self.env.tmp / "nope")
        ADM._health_cache.clear()
        self.assertIn("not on this computer", self.health()["backup"]["commit"]["problem"])
        # an unreadable services config / a services.py that fails
        bad = self.env.tmp / "bad_services.json"
        bad.write_text("{not json", encoding="utf-8")
        os.environ["CRYOEV_SERVICES_CONFIG"] = str(bad)
        self.addCleanup(os.environ.__setitem__, "CRYOEV_SERVICES_CONFIG", str(self.svc_cfg))
        ADM._health_cache.clear()
        self.assertIn("failed", self.health()["services"]["problem"])
        os.environ["CRYOEV_SERVICES_CONFIG"] = str(self.env.tmp / "missing_services.json")
        ADM._health_cache.clear()
        self.assertIn("not set up", self.health()["services"]["problem"])

    def test_known_answers(self):
        """Controls whose answers are fixed by the fixtures: a listening dummy server, a fixed commit time, a
        known live.json, a schtasks CSV."""
        up = T.Upstream()
        self.addCleanup(up.stop)
        cfg = self.env.tmp / "services_known.json"
        cfg.write_text(json.dumps({"run_dir": str(self.env.tmp / "run2"), "services": [
            {"name": "demo", "enabled": True, "port": up.port, "argv": ["x"], "cwd": str(self.env.tmp)},
            {"name": "off", "enabled": False, "port": T.free_port(), "argv": ["x"], "cwd": str(self.env.tmp)}]}),
            encoding="utf-8")
        os.environ["CRYOEV_SERVICES_CONFIG"] = str(cfg)
        self.addCleanup(os.environ.__setitem__, "CRYOEV_SERVICES_CONFIG", str(self.svc_cfg))
        os.environ["CJ_BACKUP_REPO"] = str(git_repo_with_commit(self.env.tmp, "2026-11-04T10:00:00+00:00"))
        self.env.write_live(T.live_doc())
        ADM.SCHTASKS = "cj-no-such-schtasks-c1"
        h = self.health()
        rows = {r["name"]: r for r in h["services"]["rows"]}
        self.assertEqual((rows["demo"]["state"], rows["demo"]["listening"], rows["demo"]["health"]),
                         ("port busy (not ours)", True, "http 200"))
        self.assertEqual(rows["off"]["state"], "disabled")
        self.assertNotIn("log", rows["demo"])
        c = h["backup"]["commit"]
        self.assertEqual(datetime.fromisoformat(c["time"]),
                         datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc))
        self.assertEqual(c["subject"], "backup 20261104T100000Z: 3 changed")
        self.assertTrue(c["late"])                                    # older than 36 h at test time
        self.assertEqual([(q["machine"], q["display"], q["reach"]) for q in h["watchers"]["queues"]],
                         [("gpu-desktop", "alive", "ok")])
        self.assertFalse(h["watchers"]["stale"])
        page = html.unescape(self.get("/admin/health")[2].decode())
        self.assertIn("Running (started by hand)", page)
        self.assertIn("Turned off", page)
        self.assertIn("Working normally", page)
        self.assertIn("2026-11-04", page)
        # stale snapshot
        self.env.write_live(T.live_doc(age_s=3600))
        ADM._health_cache.clear()
        self.assertTrue(self.health()["watchers"]["stale"])
        self.assertIn("Updates have stopped", self.get("/admin/health")[2].decode())

    def test_schtasks_csv(self):
        ok = ADM.parse_schtasks_csv(SCHTASKS_CSV.format(res="0"))
        self.assertEqual(ok, {"last_run": "11/5/2026 2:00:01 AM", "last_result": "0",
                              "next_run": "11/6/2026 2:00:00 AM", "status": "Enabled"})
        self.assertIsNone(ADM.parse_schtasks_csv(""))
        self.assertIsNone(ADM.parse_schtasks_csv("ERROR: The system cannot find the file specified."))
        self.assertIsNone(ADM.parse_schtasks_csv('"a","b"\r\n"1","2"\r\n'))
        if os.name == "nt":
            orig = ADM.run
            self.addCleanup(setattr, ADM, "run", orig)
            for res, want in (("0", (True, False)), ("1", (False, False)), ("267011", (None, True))):
                ADM.run = lambda argv, timeout=10, env=None, r=res: (0, SCHTASKS_CSV.format(res=r), None)
                t = ADM.backup_task()
                self.assertEqual((t["found"], t["ok"], t["never"]), (True, *want), res)
            ADM.run = orig
            ADM.BACKUP_TASK = "cj-test-no-such-task-c1"
            t = ADM.backup_task()                                       # the real schtasks, a missing task
            self.assertFalse(t["found"])
            self.assertTrue(t["problem"])

    def test_health_off_the_event_loop(self):
        """A slow services check must not block other requests."""
        slow = self.env.tmp / "slow_services.py"
        slow.write_text("import time, json; time.sleep(2); print(json.dumps([]))\n", encoding="utf-8")
        ADM.SERVICES_SCRIPT = slow
        ADM.SCHTASKS = "cj-no-such-schtasks-c1"
        box = {}
        th = threading.Thread(target=lambda: box.setdefault("r", self.get("/admin/api/health")))
        import time
        t0 = time.monotonic()
        th.start()
        time.sleep(0.3)
        st, _, _ = self.get("/healthz")
        self.assertEqual(st, 200)
        self.assertLess(time.monotonic() - t0, 1.5)
        th.join(30)
        self.assertEqual(box["r"][0], 200)


def launch(p):
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:
            continue
    return None


class Browser(Base):
    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        super().setUpClass()
        cls.own_shots = not os.environ.get("CJ_SHOT_DIR")
        cls.shots = Path(os.environ.get("CJ_SHOT_DIR") or tempfile.mkdtemp(prefix="cj_shots_"))
        cls.shots.mkdir(parents=True, exist_ok=True)
        cls.env.write_live(T.live_doc())
        ADM.SCHTASKS = "cj-no-such-schtasks-c1"
        for pid, role in (("au", "advanced"), ("sample-sci", "basic"), ("old.user", "basic")):
            A.add_key(cls.env.roster, pid, role)
        A.revoke(cls.env.roster, "old.user")
        cls.pw = sync_playwright().start()
        cls.browser = launch(cls.pw)
        if cls.browser is None:
            cls.tearDownClass()
            raise unittest.SkipTest("no Edge / Chrome for Playwright")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "browser", None):
            cls.browser.close()
        if getattr(cls, "pw", None):
            cls.pw.stop()
        super().tearDownClass()
        if cls.own_shots:
            shutil.rmtree(cls.shots, ignore_errors=True)

    def session(self, width, scheme):
        ctx = self.browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
        origin = self.p.origin
        rec = {"requests": [], "foreign": [], "errors": [], "console": []}
        ctx.route("**/*", lambda r: (r.continue_() if r.request.url.startswith(origin + "/")
                                     else (rec["foreign"].append(r.request.url), r.abort())))
        page = ctx.new_page()
        page.on("request", lambda r: rec["requests"].append(r.url))
        page.on("pageerror", lambda e: rec["errors"].append(str(e)))
        page.on("console", lambda m: m.type == "error" and rec["console"].append(m.text))
        page.goto(f"{origin}/login?k={self.env.key}")
        self.assertEqual(page.url, origin + "/")
        return ctx, page, rec

    def check(self, page, rec, scheme):
        sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
        self.assertLessEqual(sw, iw, "horizontal scroll")
        bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
        self.assertEqual(bg, "rgb(15, 21, 27)" if scheme == "dark" else "rgb(243, 245, 247)")
        self.assertEqual(rec["foreign"], [])
        self.assertTrue(all(u.startswith(self.p.origin + "/") for u in rec["requests"]))
        self.assertEqual(rec["errors"], [])
        self.assertEqual(rec["console"], [])

    def test_pages_sizes_and_schemes(self):
        ADM.audit(types.SimpleNamespace(cfg=self.env.config()), "tester", "role_changed", "au",
                  {"from": "basic", "to": "advanced"})
        for scheme in ("light", "dark"):
            for width in (390, 1280):
                with self.subTest(width=width, scheme=scheme):
                    ctx, page, rec = self.session(width, scheme)
                    try:
                        for name, path in (("people", "/admin/"), ("health", "/admin/health"),
                                           ("audit", "/admin/audit")):
                            page.goto(self.p.origin + path)
                            page.wait_for_load_state("networkidle")
                            page.wait_for_selector("#cj-bar")
                            self.assertEqual(page.locator(".cj-admin-sub a[aria-current=page]").count(), 1)
                            if name == "people":
                                self.assertEqual(page.locator(".cj-admin-people").first.locator(
                                    ".cj-admin-person").count(), 5)
                                page.click(".cj-admin-gone summary")
                            self.check(page, rec, scheme)
                            page.screenshot(path=str(self.shots / f"admin_{name}_{width}_{scheme}.png"),
                                            full_page=True)
                    finally:
                        ctx.close()

    def test_add_and_revoke_flow(self):
        for scheme in ("light", "dark"):
            pid = f"flow-{scheme}"
            ctx, page, rec = self.session(390, scheme)
            try:
                page.goto(self.p.origin + "/admin/")
                page.fill("#cj-admin-id", pid)
                page.select_option("#cj-admin-role", "advanced")
                page.click(".cj-admin-add button[type=submit]")
                page.wait_for_selector("#cj-admin-link")
                link = page.input_value("#cj-admin-link")
                self.assertTrue(link.startswith(self.p.origin + "/login?k="), link)
                self.assertTrue(page.is_visible("[data-cj-admin-copy]"))       # 127.0.0.1 is a secure context
                self.check(page, rec, scheme)
                page.screenshot(path=str(self.shots / f"admin_keylink_390_{scheme}.png"), full_page=True)
                page.goto(self.p.origin + "/admin/")
                self.assertNotIn("login?k=", page.content())
                page.click(f'a[href="/admin/people/revoke?id={pid}"]')
                page.wait_for_selector(".cj-admin-confirm")
                self.check(page, rec, scheme)
                page.screenshot(path=str(self.shots / f"admin_revoke_confirm_390_{scheme}.png"), full_page=True)
                page.click(".cj-admin-dangerbtn")
                page.wait_for_url(f"**/admin/?done=revoked&id={pid}")
                self.assertIn(f"{pid} is revoked", page.text_content(".cj-admin-note"))
                self.check(page, rec, scheme)
                page.screenshot(path=str(self.shots / f"admin_revoked_390_{scheme}.png"), full_page=True)
            finally:
                ctx.close()
        st, _, _ = T.request(self.p.port, "GET", "/login?k=" + link.rsplit("=", 1)[1])
        self.assertEqual(st, 403)


if __name__ == "__main__":
    unittest.main()
