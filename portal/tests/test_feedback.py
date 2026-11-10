"""
Feedback module (round 7, F): fold, who sees what (admin as the control), the Gate's Origin check, the admin form
token, the 20-a-day limit, the ntfy body (no report text, no user id), page = path only, XSS inert, one shared file
for two app instances, the header link.   Temp roster / state only.   python -m unittest portal.tests.test_feedback -v
"""
import json
import os
import re
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import auth as A  # noqa: E402
from portal import feedback_store as FS  # noqa: E402
from portal.modules import feedback as F  # noqa: E402

XSS = '<img src=x onerror=alert(1)>'
SECRET_TEXT = "my-private-sentence-zq81"


def form(d):
    return urlencode(d).encode()


def _mp_worker(args):
    f, n, who = args
    got, limited = [], 0
    for i in range(n):
        try:
            got.append(FS.create(f, who, "bug", ("long text \u00e9 " * 300) + str(i))["id"])
        except FS.RateLimited:
            limited += 1
    return got, limited


class StoreTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_fb_"))
        self.f = self.tmp / "fb.jsonl"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fold_after_status_and_reply(self):
        a = FS.create(self.f, "ann", "bug", "broke", "click", "/gallery/x?secret=1#h")
        b = FS.create(self.f, "bob", "feature", "idea")
        self.assertEqual((a["id"], b["id"]), (1, 2))
        FS.set_status(self.f, 1, "admin1", "Planned")
        FS.set_reply(self.f, 1, "admin1", "soon")
        FS.set_status(self.f, 1, "admin1", "Done")
        r = FS.read_all(self.f)
        self.assertEqual((r[1]["status"], r[1]["reply"], r[1]["page"]), ("Done", "soon", "/gallery/x"))
        self.assertEqual((r[2]["status"], r[2]["reply"]), ("New", ""))
        with self.assertRaises(FS.Invalid):
            FS.set_status(self.f, 99, "admin1", "Done")
        with self.assertRaises(FS.Invalid):
            FS.set_status(self.f, 1, "admin1", "Bogus")
        with open(self.f, "a", encoding="utf-8") as fh:             # garbage lines don't break the fold
            fh.write("not json\n" + json.dumps({"type": "status", "id": 7, "status": "Done"}) + "\n")
        self.assertEqual(sorted(FS.read_all(self.f)), [1, 2])

    def test_limits_and_page(self):
        for p, want in (("/a/b?x=1", "/a/b"), ("/a#f", "/a"), ("//evil.com", ""), ("http://x/y", ""), ("a", ""),
                        ("/" + "x" * 200, ""), ("/" + "x" * 199, "/" + "x" * 199)):
            self.assertEqual(FS.clean_page(p), want, p)
        with self.assertRaises(FS.Invalid):
            FS.create(self.f, "ann", "bug", "x" * 4001)
        with self.assertRaises(FS.Invalid):
            FS.create(self.f, "ann", "bug", "")
        with self.assertRaises(FS.Invalid):
            FS.set_reply(self.f, 1, "a", "x")                        # no such report
        FS.create(self.f, "ann", "bug", "ok")
        with self.assertRaises(FS.Invalid):
            FS.set_reply(self.f, 1, "a", "x" * 2001)

    def test_rate_limit_21st_refused_per_user(self):
        for i in range(20):
            FS.create(self.f, "ann", "question", f"q{i}")
        with self.assertRaises(FS.RateLimited):
            FS.create(self.f, "ann", "question", "one too many")
        FS.create(self.f, "bob", "question", "bob is not limited by ann")
        self.assertEqual(len(FS.read_all(self.f)), 21)

    def test_concurrent_creates_get_distinct_ids(self):
        def go(n):
            FS.create(self.f, f"u{n}", "bug", "x")
        ts = [threading.Thread(target=go, args=(i,)) for i in range(12)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(sorted(FS.read_all(self.f)), list(range(1, 13)))

    def test_page_rejects_backslash_and_format_chars(self):
        for p in ("/a\\b", "/a‮b", "/a​b", "/a\x7fb"):
            self.assertEqual(FS.clean_page(p), "", repr(p))

    def test_torn_last_line_does_not_swallow_the_next_create(self):
        a = FS.create(self.f, "ann", "bug", "first")
        with open(self.f, "ab") as fh:
            fh.write(b'{"type": "create", "id": 2, "who": "bob", "kind": "bug", "te')      # a crash mid-write
        b = FS.create(self.f, "cat", "bug", "second")
        self.assertEqual((a["id"], b["id"]), (1, 2))
        r = FS.read_all(self.f)
        self.assertEqual({k: v["who"] for k, v in r.items()}, {1: "ann", 2: "cat"})

    def test_cross_process_creates(self):
        import multiprocessing as mp
        with mp.Pool(4) as pool:
            res = pool.map(_mp_worker, [(str(self.f), 12, f"u{i % 2}") for i in range(4)])   # 2 users x 2 procs x 12
        ids = [i for got, _ in res for i in got]
        self.assertEqual(len(ids), len(set(ids)))                                      # no id handed out twice
        r = FS.read_all(self.f)
        self.assertEqual(sorted(r), sorted(ids))
        for u in ("u0", "u1"):
            self.assertEqual(sum(1 for x in r.values() if x["who"] == u), 20)         # the limit held across processes
        lines = self.f.read_text(encoding="utf-8").splitlines()
        self.assertTrue(all(ln.startswith("{") and ln.endswith("}") for ln in lines))   # no torn lines
        self.assertEqual(len(lines), 40)

    def test_cli_list_json(self):
        import contextlib
        import io
        FS.create(self.f, "ann", "bug", "hello")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            FS.main(["list", "--file", str(self.f), "--json"])
        self.assertEqual(json.loads(out.getvalue())[0]["text"], "hello")


class World:
    def __init__(self, fb_file=None):
        self.saved = os.environ.get(FS.ENV)
        if fb_file:
            os.environ[FS.ENV] = str(fb_file)
        else:
            os.environ.pop(FS.ENV, None)
        self.env = T.Env()
        self.keys = {"admin": self.env.key, "adv": A.add_key(self.env.roster, "alyssia", "advanced"),
                     "basic": A.add_key(self.env.roster, "bas", "basic"), "basic2": A.add_key(self.env.roster, "bas2", "basic"),
                     "adv2": A.add_key(self.env.roster, "bea", "advanced")}
        self.env.write_live(T.live_doc())
        self.portal = T.Portal(self.env.config())
        self.tok = {w: T.login(self.portal.port, k) for w, k in self.keys.items()}

    def req(self, who, method, path, body=None, origin=True, ctype=True):
        h = T.cookie(self.tok[who]) if who else {}
        if method != "GET":
            if origin:
                h["Origin"] = self.portal.origin
            if ctype:
                h["Content-Type"] = "application/x-www-form-urlencoded"
        return T.request(self.portal.port, method, path, h, body)

    def get(self, who, path):
        st, h, b = self.req(who, "GET", path)
        return st, b.decode("utf-8", "replace")

    def stop(self):
        self.portal.stop()
        self.env.cleanup()
        if self.saved is None:
            os.environ.pop(FS.ENV, None)
        else:
            os.environ[FS.ENV] = self.saved

    def token(self, who, path="/feedback/all"):
        st, page = self.get(who, path)
        return re.search(r'name="t" value="([0-9a-f]+)"', page).group(1)


class PortalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = World()

    @classmethod
    def tearDownClass(cls):
        cls.w.stop()

    def test_create_prg_and_my_reports(self):
        w = self.w
        st, h, _ = w.req("basic", "POST", "/feedback/", form({"kind": "bug", "text": "mine-one", "steps": "do x",
                                                               "page": "/data/?q=1"}))
        self.assertEqual(st, 303)
        loc = T.hget(h, "location")
        st, page = w.get("basic", loc)
        self.assertIn("Thank you", page)
        self.assertIn("mine-one", page)
        self.assertIn("<code>/data/</code>", page)
        self.assertNotIn("q=1", page)

    def test_basic_sees_only_own_admin_sees_all(self):
        w = self.w
        w.req("basic", "POST", "/feedback/", form({"kind": "bug", "text": "text-of-bas-AAA"}))
        w.req("basic2", "POST", "/feedback/", form({"kind": "question", "text": "text-of-bas2-BBB"}))
        st, mine = w.get("basic", "/feedback/")
        self.assertIn("text-of-bas-AAA", mine)
        self.assertNotIn("text-of-bas2-BBB", mine)
        st, page = w.get("basic", "/feedback/all")                 # a basic person can't open the admin list
        self.assertEqual(st, 403)
        self.assertNotIn("text-of-bas2-BBB", page)
        st, allp = w.get("admin", "/feedback/all")                 # control: the admin sees both
        self.assertEqual(st, 200)
        self.assertIn("text-of-bas-AAA", allp)
        self.assertIn("text-of-bas2-BBB", allp)
        st, f = w.get("admin", "/feedback/all?kind=question")
        self.assertIn("text-of-bas2-BBB", f)
        self.assertNotIn("text-of-bas-AAA", f)

    def test_cross_origin_post_refused(self):
        w = self.w
        n0 = len(FS.read_all(FS.feedback_path(w.env.state)))
        for origin in (False, "https://evil.example"):
            h = T.cookie(w.tok["basic"]) | {"Content-Type": "application/x-www-form-urlencoded"}
            if origin:
                h["Origin"] = origin
            st, _, _ = T.request(w.portal.port, "POST", "/feedback/", h, form({"kind": "bug", "text": "csrf-text"}))
            self.assertIn(st, (400, 403), origin)
        self.assertEqual(len(FS.read_all(FS.feedback_path(w.env.state))), n0)

    def test_admin_actions_need_the_token_and_role(self):
        w = self.w
        w.req("basic", "POST", "/feedback/", form({"kind": "bug", "text": "needs-triage"}))
        rid = max(FS.read_all(FS.feedback_path(w.env.state)))
        st, _, _ = w.req("admin", "POST", f"/feedback/{rid}/status", form({"status": "Done"}))
        self.assertEqual(st, 403)                                     # no token
        st, _, _ = w.req("admin", "POST", f"/feedback/{rid}/status", form({"status": "Done", "t": "0" * 40}))
        self.assertEqual(st, 403)                                     # wrong token
        st, _, _ = w.req("basic", "POST", f"/feedback/{rid}/status", form({"status": "Done", "t": w.token("admin")}))
        self.assertEqual(st, 403)                                     # not an admin, even with an admin's token
        self.assertEqual(FS.read_all(FS.feedback_path(w.env.state))[rid]["status"], "New")
        tok = w.token("admin")                                        # control: the right token works
        st, _, _ = w.req("admin", "POST", f"/feedback/{rid}/status", form({"status": "Planned", "t": tok}))
        self.assertEqual(st, 303)
        st, _, _ = w.req("admin", "POST", f"/feedback/{rid}/reply", form({"reply": "on our list", "t": tok}))
        self.assertEqual(st, 303)
        st, page = w.get("basic", "/feedback/")
        self.assertIn("Planned", page)
        self.assertIn("on our list", page)

    def test_xss_inert(self):
        w = self.w
        w.req("basic", "POST", "/feedback/", form({"kind": "bug", "text": XSS, "steps": XSS, "page": "/x"}))
        for who, path in (("basic", "/feedback/"), ("admin", "/feedback/all")):
            st, page = w.get(who, path)
            self.assertNotIn(XSS, page)
            self.assertIn("&lt;img src=x onerror=alert(1)&gt;", page)
            self.assertEqual(re.findall(r"<script[^>]*>", page), ['<script src="/cj/header.js" defer>'])
        st, page = w.get("basic", "/feedback/?page=/x%22%3E%3Cscript%3Ealert(1)%3C/script%3E")
        self.assertNotIn("<script>alert", page)

    def test_page_prefill_is_path_only(self):
        st, page = self.w.get("basic", "/feedback/?page=/gallery/abc%3Fk%3Dzzquery")
        self.assertIn('value="/gallery/abc"', page)
        self.assertNotIn("zzquery", page)

    def test_ntfy_has_no_text_or_user(self):
        sent = []
        done = threading.Event()

        def fake(title, message):
            sent.append((title, message))
            done.set()
        with mock.patch.object(F, "_ntfy", fake):
            st, _, _ = self.w.req("basic", "POST", "/feedback/", form({"kind": "bug", "text": SECRET_TEXT}))
        self.assertEqual(st, 303)
        self.assertTrue(done.wait(5))
        title, message = sent[0]
        self.assertRegex(title, r"^CryoEV: new bug report #\d+$")
        for s in (title, message):
            self.assertNotIn(SECRET_TEXT, s)
            self.assertNotIn("bas", s.replace("base", ""))
        # a failing push never fails the request
        with mock.patch.object(F, "_ntfy", side_effect=RuntimeError("down")):
            st, _, _ = self.w.req("basic2", "POST", "/feedback/", form({"kind": "feature", "text": "still ok"}))
        self.assertEqual(st, 303)

    def test_rate_limit_over_http(self):
        w = self.w
        with mock.patch.object(F, "_ntfy"):
            codes = [w.req("adv2", "POST", "/feedback/", form({"kind": "question", "text": f"q{i}"}))[0]
                     for i in range(20)]
            st, _, body = w.req("adv2", "POST", "/feedback/", form({"kind": "question", "text": "one-too-many-ZZ"}))
        self.assertEqual(codes, [303] * 20)
        self.assertEqual(st, 429)
        self.assertIn("one-too-many-ZZ", body.decode())                     # the typed text is kept in the form
        stored = [r["text"] for r in FS.read_all(FS.feedback_path(w.env.state)).values() if r["who"] == "bea"]
        self.assertEqual(len(stored), 20)
        self.assertNotIn("one-too-many-ZZ", stored)                          # and it was not saved

    def test_steps_only_kept_for_bugs(self):
        w = self.w
        with mock.patch.object(F, "_ntfy"):
            w.req("basic", "POST", "/feedback/", form({"kind": "feature", "text": "idea-STEPS", "steps": "should-vanish"}))
            w.req("basic", "POST", "/feedback/", form({"kind": "bug", "text": "bug-STEPS", "steps": "should-stay"}))
        by = {r["text"]: r for r in FS.read_all(FS.feedback_path(w.env.state)).values()}
        self.assertEqual(by["idea-STEPS"]["steps"], "")
        self.assertEqual(by["bug-STEPS"]["steps"], "should-stay")

    def test_advanced_sees_only_own(self):
        w = self.w
        with mock.patch.object(F, "_ntfy"):
            w.req("adv", "POST", "/feedback/", form({"kind": "bug", "text": "adv-secret-QQ1"}))
            w.req("basic", "POST", "/feedback/", form({"kind": "bug", "text": "basic-secret-QQ2"}))
        st, a = w.get("adv", "/feedback/")
        self.assertIn("adv-secret-QQ1", a)
        self.assertNotIn("basic-secret-QQ2", a)
        st, b = w.get("basic", "/feedback/")
        self.assertIn("basic-secret-QQ2", b)
        self.assertNotIn("adv-secret-QQ1", b)
        self.assertEqual(w.get("adv", "/feedback/all")[0], 403)
        self.assertIn("adv-secret-QQ1", w.get("admin", "/feedback/all")[1])        # control

    def test_crlf_counts_as_one_character(self):
        text = "\r\n".join(["x" * 39] * 100)            # 3,900 + 99 newlines = 3,999 stored; with \r\n it would be 4,098
        with mock.patch.object(F, "_ntfy"):
            st, _, _ = self.w.req("basic2", "POST", "/feedback/", form({"kind": "question", "text": text}))
        self.assertEqual(st, 303)
        r = [r for r in FS.read_all(FS.feedback_path(self.w.env.state)).values()
             if r["text"].startswith("xxxx") and r["who"] == "bas2"]
        self.assertTrue(r and "\r" not in r[-1]["text"])

    def test_oversize_form_413_keeps_typed_text(self):
        w = self.w
        big = form({"kind": "bug", "text": "kept-text-WW", "steps": "y" * (F.MAX_FORM + 10)})
        st, _, body = w.req("basic", "POST", "/feedback/", big)
        self.assertEqual(st, 413)
        self.assertIn("kept-text-WW", body.decode())
        # a legitimate worst-case form (every field at its limit, all characters percent-encoded) is accepted
        worst = form({"kind": "bug", "text": "\U0001F600" * FS.TEXT_MAX, "steps": "\U0001F600" * FS.STEPS_MAX,
                      "page": "/" + "a" * 199})
        self.assertLessEqual(len(worst), F.MAX_FORM)
        with mock.patch.object(F, "_ntfy"):
            st, _, _ = w.req("adv", "POST", "/feedback/", worst)
        self.assertEqual(st, 303)

    def test_chunked_body_is_capped(self):
        import socket
        w = self.w
        n0 = len(FS.read_all(FS.feedback_path(w.env.state)))
        s = socket.create_connection(("127.0.0.1", w.portal.port), timeout=10)
        ck = (A.COOKIE, w.tok["basic"])
        head = (f"POST /feedback/ HTTP/1.1\r\nHost: 127.0.0.1:{w.portal.port}\r\nOrigin: {w.portal.origin}\r\n"
                f"Cookie: {ck[0]}={ck[1]}\r\n"
                "Content-Type: application/x-www-form-urlencoded\r\nTransfer-Encoding: chunked\r\n\r\n")
        s.sendall(head.encode())
        chunk = b"text=" + b"a" * 60000
        data = b""
        try:
            for _ in range(10):                                        # 600 KB, far over the cap
                s.sendall(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            s.sendall(b"0\r\n\r\n")
            data = s.recv(65536)
        except OSError:
            pass                                                       # the server may stop reading and close
        s.close()
        self.assertTrue(data == b"" or b" 413 " in data.split(b"\r\n", 1)[0], data[:80])
        self.assertEqual(len(FS.read_all(FS.feedback_path(w.env.state))), n0)

    def test_header_has_the_link(self):
        js = (Path(__file__).resolve().parent.parent / "static" / "header.js").read_text(encoding="utf-8")
        self.assertIn("'/feedback/?page=' + encodeURIComponent(location.pathname)", js)
        self.assertNotIn("location.search", js)
        self.assertIn("setAttribute('aria-current', 'page')", js[js.index("cj-feedback-link"):])
        css = (Path(__file__).resolve().parent.parent / "static" / "cj.css").read_text(encoding="utf-8")
        self.assertTrue(css.rstrip().endswith("/* cj-feedback:end */"))
        mods = json.loads((Path(__file__).resolve().parent.parent / "modules.json").read_text(encoding="utf-8"))
        self.assertNotIn("feedback", [m["id"] for m in mods["modules"]])      # no second link in the nav


class SharedFileTests(unittest.TestCase):
    def test_two_apps_one_file(self):
        import tempfile
        d = Path(tempfile.mkdtemp(prefix="cj_fbshared_"))
        a = World(d / "shared.jsonl")
        try:
            b = World(d / "shared.jsonl")
            try:
                a.req("basic", "POST", "/feedback/", form({"kind": "bug", "text": "from-app-a"}))
                st, page = b.get("basic", "/feedback/")                   # same person id, other portal
                self.assertIn("from-app-a", page)
                self.assertFalse((a.env.state / "feedback.jsonl").exists())
            finally:
                b.stop()
        finally:
            a.stop()
            import shutil
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
