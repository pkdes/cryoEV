"""
Round 9 G2: the in-app changelog (/changes, signed in only, CHANGELOG.md as escaped HTML, newest first) and the version
stamps (/cj/version.json carries the review page's own PAGE_VERSION + file time; the bar links to /changes).
Temp roster / state / review package only.   python -m unittest portal.tests.test_changes -v
"""
import json
import os
import re
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import app as APP  # noqa: E402
from portal import auth as A  # noqa: E402

FIXTURE_MD = """# Log

## 2026-11-12 19:30 · NEWEST-zz <script>alert(1)</script>

- a **bold** bit and `<b>code</b>`
- <img src=x onerror=alert(2)>
  wrapped tail

**Components:** portal

## 2026-11-10 10:00 · OLDER-zz

- plain
"""


def clean_env(test):
    p = mock.patch.dict(os.environ)
    p.start()
    test.addCleanup(p.stop)
    for k in [k for k in os.environ if k.startswith("CRYOEV_")]:
        os.environ.pop(k)


class Base(unittest.TestCase):
    PAGE = '<!doctype html><title>t</title><script>\nconst PAGE_VERSION = "2026-11-12.1";\n</script>'

    @classmethod
    def setUpClass(cls):
        cls._env_patch = mock.patch.dict(os.environ)
        cls._env_patch.start()
        for k in [k for k in os.environ if k.startswith("CRYOEV_")]:
            os.environ.pop(k)
        cls.env = T.Env()
        cls.pkg = cls.env.tmp / "pkg"
        cls.pkg.mkdir()
        if cls.PAGE is not None:
            (cls.pkg / "index.html").write_text(cls.PAGE, encoding="utf-8")
        cls.bas = A.add_key(cls.env.roster, "bas", "basic")
        cls.p = T.Portal(cls.env.config(review_package=cls.pkg))
        cls.tok = {"admin": T.login(cls.p.port, cls.env.key), "basic": T.login(cls.p.port, cls.bas)}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()
        cls._env_patch.stop()

    def setUp(self):
        clean_env(self)

    def get(self, path, who="basic", extra=None):
        h = dict(T.cookie(self.tok[who])) if who else {}
        h.update(extra or {})
        st, _, b = T.request(self.p.port, "GET", path, h)
        return st, b


class Changes(Base):
    def setUp(self):
        super().setUp()
        self.md = self.env.tmp / "CHANGELOG.md"
        self.md.write_text(FIXTURE_MD, encoding="utf-8")
        p = mock.patch.dict(os.environ, {APP.CHANGELOG_ENV: str(self.md)})
        p.start()
        self.addCleanup(p.stop)

    def test_needs_sign_in(self):
        self.assertNotIn("/changes", APP.OPEN)
        st, b = self.get("/changes", who=None)
        self.assertEqual(st, 401)
        self.assertNotIn(b"NEWEST-zz", b)
        st, b = self.get("/changes", who=None, extra={"Cookie": f"{A.COOKIE}=forged"})
        self.assertEqual(st, 403)
        self.assertNotIn(b"NEWEST-zz", b)

    def test_every_role_reads_it_escaped_newest_first(self):
        for who in ("basic", "admin"):
            st, b = self.get("/changes", who=who)
            self.assertEqual(st, 200, who)
        text = b.decode()
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", text)
        self.assertNotIn("<script>alert", text)
        self.assertIn("&lt;img src=x onerror=alert(2)&gt; wrapped tail", text)
        self.assertNotIn("<img", text)
        self.assertIn("<strong>bold</strong>", text)
        self.assertIn("<code>&lt;b&gt;code&lt;/b&gt;</code>", text)
        self.assertIn("<strong>Components:</strong> portal", text)
        self.assertLess(text.index("NEWEST-zz"), text.index("OLDER-zz"))
        self.assertNotIn(str(self.env.tmp), text)

    def test_missing_file(self):
        with mock.patch.dict(os.environ, {APP.CHANGELOG_ENV: str(self.env.tmp / "nope.md")}):
            st, b = self.get("/changes")
        self.assertEqual(st, 200)
        self.assertIn(b"No changelog yet", b)

    def test_about_links_changes(self):
        st, b = self.get("/about")
        self.assertEqual(st, 200)
        self.assertIn(b'href="/changes"', b)

    def test_bar_links_changes(self):
        js = self.get("/cj/header.js")[1].decode()
        self.assertIn("'/changes'", js)
        self.assertIn("review_page", js)
        self.assertIn("review_tool", js)


class Render(unittest.TestCase):
    def test_no_raw_html_passthrough(self):
        out = APP.render_changelog('<div onclick="x">hi</div>\n\n- `unpaired <i>\n## **<b>**')
        self.assertNotRegex(out, r"<(div|i|b)[ >]")
        self.assertIn("&lt;div onclick=&quot;x&quot;&gt;", out)
        self.assertIn("<strong>&lt;b&gt;</strong>", out)

    def test_real_changelog_newest_first_and_public_safe(self):
        md = (APP.REPO / "CHANGELOG.md").read_text(encoding="utf-8")
        heads = re.findall(r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) · (.+)$", md, re.M)
        self.assertGreaterEqual(len(heads), 3)
        when = [datetime.strptime(h[0], "%Y-%m-%d %H:%M") for h in heads]
        self.assertEqual(when, sorted(when, reverse=True))
        self.assertEqual(len(re.findall(r"^## ", md, re.M)), len(heads))      # every entry has the dated form
        for block in re.split(r"^## ", md, flags=re.M)[1:]:
            self.assertIn("**Components:**", block)
            self.assertTrue(2 <= len(re.findall(r"^- ", block, re.M)) <= 8, block[:60])
        self.assertNotRegex(md, r"@[\w-]+\.\w+")                                  # no emails
        out = APP.render_changelog(md)
        self.assertLess(out.index(heads[0][1].strip()), out.index(heads[-1][1].strip()))


class VersionWithConstant(Base):
    def test_version_json_has_review_page_version(self):
        st, b = self.get("/cj/version.json")
        self.assertEqual(st, 200)
        doc = json.loads(b)
        self.assertIn("version", doc)
        self.assertEqual(doc["review_page"]["version"], "2026-11-12.1")
        self.assertRegex(doc["review_page"]["updated"], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")
        from annotation import multi_review_server as MRS
        self.assertEqual(doc["review_tool"], APP.page_version(MRS.TOOL_PAGE))      # the repo's tool page
        self.assertNotIn(str(self.env.tmp), b.decode())

    def test_page_swap_shows_without_restart(self):
        page = self.pkg / "index.html"
        self.addCleanup(page.write_text, self.PAGE, encoding="utf-8")            # the other tests see the original
        page.write_text(self.PAGE.replace("2026-11-12.1", "2026-11-13.2") + "<!-- longer -->", encoding="utf-8")
        os.utime(page, (1_800_000_000, 1_800_000_000))
        doc = json.loads(self.get("/cj/version.json")[1])
        self.assertEqual(doc["review_page"]["version"], "2026-11-13.2")
        self.assertEqual(doc["review_page"]["updated"],
                         datetime.fromtimestamp(1_800_000_000).strftime("%Y-%m-%d %H:%M"))

    def test_anonymous_refused(self):
        self.assertEqual(self.get("/cj/version.json", who=None)[0], 401)


class VersionWithoutConstant(Base):
    PAGE = "<!doctype html><title>old page</title><script>const X = 1;</script>"

    def test_unknown(self):
        doc = json.loads(self.get("/cj/version.json")[1])
        self.assertEqual(doc["review_page"]["version"], "unknown")
        self.assertIsNotNone(doc["review_page"]["updated"])


class VersionNoPage(Base):
    PAGE = None

    def test_unknown_no_file(self):
        doc = json.loads(self.get("/cj/version.json")[1])
        self.assertEqual(doc["review_page"], {"version": "unknown", "updated": None})


class RepoToolPage(unittest.TestCase):
    def test_review_tool_has_one_page_version(self):
        from annotation import multi_review_server as MRS
        text = MRS.TOOL_PAGE.read_text(encoding="utf-8")
        self.assertEqual(len(APP.PAGE_VERSION_RE.findall(text)), 1)
        self.assertRegex(APP.page_version(MRS.TOOL_PAGE)["version"], r"^\d{4}-\d{2}-\d{2}\.\d+$")
        self.assertIn('"/changes"', text)


if __name__ == "__main__":
    unittest.main()
