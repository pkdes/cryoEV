"""
Round 8c: admin Recent actions merging other doors' audit logs (CJ_AUDIT_EXTRA, read only), the version stamp, and
the All reports table.   Temp roster / state only.   python -m unittest portal.tests.test_r8c_po -v
"""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import app as APP  # noqa: E402
from portal import audit as PA  # noqa: E402
from portal import auth as A  # noqa: E402
from portal import feedback_store as FS  # noqa: E402

XSS = '<img src=x onerror=alert(1)>'
DOT = "\u00b7"


def line(t, actor, action, target):
    return json.dumps({"t": t, "actor": actor, "action": action, "target": target, "detail": None})


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_r8c_"))
        self.own = self.tmp / "portal"
        self.other = self.tmp / "portal-access" / "audit" / "portal.jsonl"
        self.other.parent.mkdir(parents=True)
        (self.own / "audit").mkdir(parents=True)
        (self.own / "audit" / "portal.jsonl").write_text(
            line("2026-11-10T10:00:00+00:00", "ann", "cancel", "j1") + "\nnot json\n[1,2]\n"
            + line("2026-11-10T12:00:00+00:00", "ann", "cancel", "j3") + "\n", encoding="utf-8")
        self.other.write_text(line("2026-11-10T11:00:00+00:00", "bob@example.org", "release", "j2") + "\n{broken\n",
                              encoding="utf-8")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_merged_sorted_with_door_and_bad_lines_skipped(self):
        rows, probs = PA.read_merged(self.own, [self.other])
        self.assertEqual([(r["target"], r["door"]) for r in rows],
                         [("j3", "this door"), ("j2", "portal-access"), ("j1", "this door")])
        self.assertEqual(probs, [])

    def test_unreadable_or_missing_extra_is_a_note(self):
        rows, probs = PA.read_merged(self.own, [self.tmp / "nope" / "audit" / "portal.jsonl", self.tmp])
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(probs), 2)
        self.assertNotIn(str(self.tmp), " ".join(probs))

    def test_env_parse_absolute_only_and_own_not_doubled(self):
        own_file = self.own / "audit" / "portal.jsonl"
        env = {PA.EXTRA_ENV: f"{self.other};relative/x.jsonl; ;{own_file}"}
        self.assertEqual(PA.extra_paths(env), [self.other, own_file])
        with mock.patch.dict(os.environ, env):
            rows, _ = PA.read_merged(self.own)
        self.assertEqual(len(rows), 3)                       # the own file named as extra is not read twice


class AdminAuditPortal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.bas = A.add_key(cls.env.roster, "bas", "basic")
        cls.other = cls.env.tmp / "door2" / "audit" / "portal.jsonl"
        cls.other.parent.mkdir(parents=True)
        cls.other.write_text(line("2026-11-10T11:00:00+00:00", "zed-other", "release", "xj9") + "\n", encoding="utf-8")
        cls.saved = os.environ.get(PA.EXTRA_ENV)
        os.environ[PA.EXTRA_ENV] = str(cls.other)
        cls.p = T.Portal(cls.env.config())
        cls.tok = {"admin": T.login(cls.p.port, cls.env.key), "basic": T.login(cls.p.port, cls.bas)}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()
        if cls.saved is None:
            os.environ.pop(PA.EXTRA_ENV, None)
        else:
            os.environ[PA.EXTRA_ENV] = cls.saved

    def get(self, who, path):
        return T.request(self.p.port, "GET", path, T.cookie(self.tok[who]))

    def test_admin_sees_other_door_with_label(self):
        st, _, b = self.get("admin", "/admin/audit")
        text = b.decode()
        self.assertEqual(st, 200)
        self.assertIn("zed-other", text)
        self.assertIn("[door2]", text)
        api = json.loads(self.get("admin", "/admin/api/audit")[2])
        self.assertIn(("zed-other", "door2"), [(e["actor"], e["door"]) for e in api["entries"]])
        self.assertNotIn(str(self.env.tmp), json.dumps(api))

    def test_basic_cannot_reach(self):
        for path in ("/admin/audit", "/admin/api/audit"):
            st, _, b = self.get("basic", path)
            self.assertEqual(st, 403)
            self.assertNotIn(b"zed-other", b)


class VersionTests(unittest.TestCase):
    def test_stamp_format_and_fallback(self):
        out = APP.version_stamp()
        self.assertTrue(out == APP.VERSION_UNKNOWN
                        or re.fullmatch(r"Updated \d{4}-\d\d-\d\d \d\d:\d\d " + DOT + r" [0-9a-f]+", out), out)
        self.assertEqual(APP.version_stamp(Path(tempfile.gettempdir()) / "no_such_repo_zz"), APP.VERSION_UNKNOWN)
        with mock.patch("subprocess.run", side_effect=OSError("no git")):
            self.assertEqual(APP.version_stamp(), APP.VERSION_UNKNOWN)

    def test_version_json_carries_patched_value_and_header_shows_it(self):
        env = T.Env()
        stamp = "Updated 2026-01-02 03:04 " + DOT + " abc1234"
        try:
            with mock.patch.object(APP, "version_stamp", return_value=stamp):
                p = T.Portal(env.config())
            try:
                tok = T.login(p.port, env.key)
                me = json.loads(T.request(p.port, "GET", "/cj/version.json", T.cookie(tok))[2])
                self.assertEqual(me["version"], stamp)
                self.assertEqual(T.request(p.port, "GET", "/cj/version.json")[0], 401)   # session only
                js = T.request(p.port, "GET", "/cj/header.js", T.cookie(tok))[2].decode()
                self.assertIn("/cj/version.json", js)
            finally:
                p.stop()
        finally:
            env.cleanup()


class TableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.f = cls.env.tmp / "fb.jsonl"
        cls.saved = os.environ.get(FS.ENV)
        os.environ[FS.ENV] = str(cls.f)
        cls.p = T.Portal(cls.env.config())
        cls.tok = T.login(cls.p.port, cls.env.key)

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()
        if cls.saved is None:
            os.environ.pop(FS.ENV, None)
        else:
            os.environ[FS.ENV] = cls.saved

    def setUp(self):                                         # fresh reports per test: no order dependence
        self.f.unlink(missing_ok=True)
        FS.create(self.f, "bas", "bug", XSS, XSS, "/gallery/x")
        FS.create(self.f, "bas", "question", "L" * 300)

    def page(self):
        return T.request(self.p.port, "GET", "/feedback/all", T.cookie(self.tok))[2].decode()

    def test_columns_escaping_snippet_and_note(self):
        page = self.page()
        for h in ("#", "From", "When", "Type", "Page", "What they wrote", "Status", "Reply"):
            self.assertIn(f'<th scope="col">{h}</th>', page)
        self.assertNotIn(XSS, page)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", page)
        self.assertIn("<code>/gallery/x</code>", page)
        self.assertIn("<details>", page)
        self.assertIn("L" * 140 + "\u2026", page)
        self.assertIn("L" * 300, page)                       # full text inside the details
        self.assertRegex(page, r"\d\d \w{3} 20\d\d \d\d:\d\d")
        self.assertIn("2 new reports from everyone.", page)
        self.assertEqual(re.findall(r"<script[^>]*>", page), ['<script src="/cj/header.js" defer>'])

    def test_actions_still_post(self):
        page = self.page()
        self.assertEqual(len(re.findall(r'action="/feedback/\d+/status"', page)), 2)
        self.assertEqual(len(re.findall(r'action="/feedback/\d+/reply"', page)), 2)
        t = re.search(r'name="t" value="([0-9a-f]+)"', page).group(1)
        h = T.cookie(self.tok) | {"Origin": self.p.origin, "Content-Type": "application/x-www-form-urlencoded"}
        st, _, _ = T.request(self.p.port, "POST", "/feedback/1/status", h,
                             urlencode({"status": "Planned", "t": t}).encode())
        self.assertEqual(st, 303)
        st, _, _ = T.request(self.p.port, "POST", "/feedback/1/reply", h,
                             urlencode({"reply": "<b>r</b>", "t": t}).encode())
        self.assertEqual(st, 303)
        self.assertEqual(FS.read_all(self.f)[1]["status"], "Planned")
        page = self.page()
        self.assertIn("&lt;b&gt;r&lt;/b&gt;", page)
        self.assertIn("1 new report from everyone.", page)


if __name__ == "__main__":
    unittest.main()
