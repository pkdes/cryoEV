"""
Round 9 #17 + #1: the Gate's audit catch-all (every write route leaves a line, refused ones too; frequent routes
counted per window; no secrets; never breaks a request), the first sign-in per day, Admin > Recent actions (filters,
counter lines, agent lines, the laptop CLI door), /about (signed in only), and strip_env in main() only.
Temp roster / state only.   python -m unittest portal.tests.test_audit_r9 -v
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import app as APP  # noqa: E402
from portal import audit as PA  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import admin as AD  # noqa: E402

UNSAFE = ("POST", "PUT", "DELETE", "PATCH")


def clean_env(test):
    """CRYOEV_* cleared for the test (restored after)."""
    p = mock.patch.dict(os.environ)
    p.start()
    test.addCleanup(p.stop)
    for k in [k for k in os.environ if k.startswith("CRYOEV_")]:
        os.environ.pop(k)


def lines(state) -> list:
    f = PA.audit_path(state)
    if not f.exists():
        return []
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]


def concrete(path_tpl: str) -> str:
    """A request path for a route template: {x:int} -> 1, {path:path} -> api/edits, any other {x} -> x1."""
    def sub(m):
        name, _, conv = m.group(1).partition(":")
        return {"int": "1", "path": "api/edits"}.get(conv, f"{name}1")
    return re.sub(r"\{([^}]+)\}", sub, path_tpl)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._env_patch = mock.patch.dict(os.environ)
        cls._env_patch.start()
        for k in [k for k in os.environ if k.startswith("CRYOEV_")]:
            os.environ.pop(k)
        cls.env = T.Env()
        cls.bas = A.add_key(cls.env.roster, "bas", "basic")
        cls.p = T.Portal(cls.env.config())
        cls.w = cls.p.app.writer
        cls.tok = {"admin": T.login(cls.p.port, cls.env.key), "basic": T.login(cls.p.port, cls.bas)}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()
        cls._env_patch.stop()

    def setUp(self):
        clean_env(self)

    def send(self, method, path, who="admin", origin=True, body=b"", extra=None):
        h = dict(T.cookie(self.tok[who])) if who else {}
        if origin:
            h["Origin"] = self.p.origin
        h.update(extra or {})
        st, _, b = T.request(self.p.port, method, path, h, body)
        return st, b

    def new_lines(self, before):
        self.w.drain()
        return lines(self.env.state)[before:]


class RouteTableControl(Base):
    """Known-answer control: every registered write route (core + modules) leaves exactly one catch-all line with
    the label for its template; a route added later without a label still gets a line ("METHOD template")."""

    def test_every_write_route_leaves_a_line(self):
        routes = [r for r in self.p.app.routes if getattr(r, "methods", None) and set(r.methods) & set(UNSAFE)]
        self.assertGreaterEqual(len(routes), 40)                        # the table really was walked
        self.w.window_s = 0                                             # every event its own window: one line each
        self.addCleanup(setattr, self.w, "window_s", PA.WINDOW_S)
        seen = []
        for r in routes:
            for method in sorted(set(r.methods) & set(UNSAFE)):
                with self.subTest(route=r.path, method=method):
                    before = len(lines(self.env.state))
                    st, _ = self.send(method, concrete(r.path))
                    got = self.new_lines(before)
                    mine = [x for x in got if x["action"] != "signed in"]
                    if r.path in APP.SELF_LOGGED and st < 400:
                        self.assertTrue(mine, f"{r.path}: success without the module's own line")
                        continue
                    self.assertEqual(len(mine), 1, (r.path, st, got))
                    label, _ = APP.audit_label(r.path, {"path": "api/edits"}, method)
                    self.assertEqual(mine[0]["action"], label)
                    self.assertEqual(mine[0]["actor"], T.USER)
                    self.assertEqual(mine[0]["detail"], {"status": st})
                    seen.append(r.path)
        # every current write route has a readable label (no "POST /x/{y}" fallback in today's table)
        unlabelled = [r.path for r in routes if r.path not in APP.AUDIT_LABELS
                      and r.path not in APP.REVIEW_TOOL_TEMPLATES]
        self.assertEqual(unlabelled, [])

    def test_successful_write_is_logged_without_its_body(self):
        before = len(lines(self.env.state))
        st, _ = self.send("POST", "/feedback/", who="basic", body=b"kind=bug&text=BODY-TEXT-zz",
                          extra={"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(st, 303)
        got = [x for x in self.new_lines(before) if x["action"] != "signed in"]
        self.assertEqual([(x["actor"], x["action"], x["target"], x["detail"]) for x in got],
                         [("bas", "feedback filed", None, {"status": 303})])
        self.assertNotIn("BODY-TEXT-zz", PA.audit_path(self.env.state).read_text(encoding="utf-8"))

    def test_unknown_route_and_fallback_label(self):
        before = len(lines(self.env.state))
        st, _ = self.send("POST", "/no/such/place/secret-ish")
        got = [x for x in self.new_lines(before) if x["action"] != "signed in"]
        self.assertEqual([x["action"] for x in got], ["write to an unknown address"])
        self.assertNotIn("secret-ish", json.dumps(got))
        self.assertEqual(APP.audit_label("/x/{y}", {}, "POST"), ("POST /x/{y}", False))


class RefusedAndSafe(Base):
    def test_refused_writes_are_logged(self):
        before = len(lines(self.env.state))
        st1, _ = self.send("POST", "/feedback/", who=None)                    # no session
        st2, _ = self.send("POST", "/feedback/", origin=False)                # cross-origin (no Origin)
        st3, _ = self.send("POST", "/admin/people/add", who="basic")          # role
        self.assertEqual((st1, st2, st3), (401, 403, 403))
        got = [x for x in self.new_lines(before) if x["action"] != "signed in"]
        self.assertEqual([(x["actor"], x["action"], x["detail"]["status"]) for x in got],
                         [("-", "feedback filed", 401), (T.USER, "feedback filed", 403),
                          ("bas", "person added", 403)])

    def test_anonymous_refusals_are_counted_not_flooded(self):
        before = len(lines(self.env.state))
        for i in range(5):
            self.send("POST", f"/gallery/c{i}/api/event", who=None)             # distinct targets: still one
        got = [x for x in self.new_lines(before) if x["actor"] == "-"]
        self.assertEqual([(x["action"], x["target"]) for x in got], [("gallery mark", None)])

    def test_no_secrets_or_paths_in_lines(self):
        before = len(lines(self.env.state))
        self.send("POST", "/predict/new/session/C%3A%5Csecret%5Cdir/?k=" + self.env.key,
                  body=b"token=" + self.tok["admin"].encode())
        self.send("PUT", "/review/app/api/edits?f=../../etc/x&k=" + self.env.key)
        self.send("POST", "/review/..%2F..%2Fx/app/api/claim")
        self.send("POST", "/data/session/" + "a" * 200 + "/details")
        new = self.new_lines(before)
        text = "\n".join(json.dumps(x) for x in new)
        for bad in (self.env.key, self.tok["admin"], self.tok["basic"], str(self.env.tmp), "secret", "etc",
                    "Cookie", A.COOKIE, "a" * 81, ".."):
            self.assertNotIn(bad, text)
        for x in new:
            self.assertTrue(x["target"] is None or APP.SAFE_ID_RE.match(x["target"].lstrip("#")), x)
            self.assertTrue(x["detail"] is None or set(x["detail"]) <= {"status", "door", "how", "from", "to", "n"},
                            x)


class NeverBreaks(Base):
    def test_failing_or_slow_write_does_not_touch_the_request(self):
        st0, b0 = self.send("POST", "/feedback/9/reply")
        self.w.drain()           # the first line is written off the request: settle it before the baseline, or a
        failed0 = self.w.failed  # late write lands inside the failing-writer window below and counts twice (2 != 1)

        def boom(*a, **k):
            raise OSError("the audit log is locked by another process")
        with mock.patch.object(self.w, "_write_fn", boom):
            st1, b1 = self.send("POST", "/feedback/9/reply")
            self.w.drain()
        self.assertEqual(st1, st0)
        self.assertEqual(self.w.failed, failed0 + 1)

        # Failure mode: the request waits for the audit write. Structural, not a stopwatch (a 1.0 s bound on a 1.5 s
        # write flaked under load): the write blocks on an Event, so the request can only return if it does not wait
        # for it; the 30 s expiry would make it fail instead of hang.
        entered, release, expired = threading.Event(), threading.Event(), []

        def slow(*a, **k):
            entered.set()
            if not release.wait(30):
                expired.append(True)
        with mock.patch.object(self.w, "_write_fn", slow):
            try:
                st2, _ = self.send("POST", "/feedback/9/reply")
                self.assertFalse(release.is_set())
                self.assertEqual(expired, [])                              # the write ran off the event loop
            finally:
                release.set()
            self.w.drain()
        self.assertTrue(entered.wait(5))                                   # control: the slow write really ran
        self.assertEqual(st2, st0)

    def test_catch_all_exception_is_swallowed(self):
        with mock.patch.object(self.w, "event", side_effect=RuntimeError("bug")):
            st, _ = self.send("POST", "/feedback/9/reply")
        self.assertNotEqual(st, 500)


class Aggregation(unittest.TestCase):
    def setUp(self):
        clean_env(self)
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_r9agg_"))
        self.now = [1_800_000_000.0]
        self.w = PA.Writer(self.tmp, clock=lambda: self.now[0])
        self.addCleanup(self.w.close)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_first_immediately_then_counter_on_sweep(self):
        for dt in (0, 30, 200):
            self.now[0] = 1_800_000_000.0 + dt
            self.w.event("ann", "review save", "pkgA", {"status": 200}, frequent=True)
        self.w.event("ann", "review save", "pkgB", {"status": 200}, frequent=True)    # another target: own window
        self.w.drain()
        got = lines(self.tmp)
        self.assertEqual([(x["action"], x["target"], x["detail"]) for x in got],
                         [("review save", "pkgA", {"status": 200}), ("review save", "pkgB", {"status": 200})])
        self.assertEqual(self.w.sweep(1_800_000_000.0 + PA.WINDOW_S - 1), 0)
        self.assertEqual(self.w.sweep(1_800_000_000.0 + PA.WINDOW_S), 1)           # pkgA's window (opened at 0)
        self.assertEqual(self.w.sweep(1_800_000_200.0 + PA.WINDOW_S), 1)           # pkgB's (opened at 200)
        self.w.drain()
        got = lines(self.tmp)[2:]
        self.assertEqual(len(got), 1)                                   # pkgB had n = 1: no counter line
        d = got[0]["detail"]
        self.assertEqual((got[0]["target"], d["n"], d["status"]), ("pkgA", 3, 200))
        self.assertEqual(d["from"], datetime.fromtimestamp(1_800_000_000, timezone.utc).isoformat())
        self.assertEqual(d["to"], datetime.fromtimestamp(1_800_000_200, timezone.utc).isoformat())

    def test_event_after_window_closes_old_and_opens_new(self):
        self.w.event("ann", "upload chunk", "us_1", None, frequent=True)
        self.w.event("ann", "upload chunk", "us_1", None, frequent=True)
        self.now[0] += PA.WINDOW_S + 1
        self.w.event("ann", "upload chunk", "us_1", None, frequent=True)
        self.w.drain()
        got = lines(self.tmp)
        self.assertEqual([x["detail"] and x["detail"].get("n") for x in got], [None, 2, None])
        self.w.event("ann", "upload chunk", "us_1", {"status": 403}, frequent=True)  # refused: its own window
        self.w.drain()
        self.assertEqual(lines(self.tmp)[-1]["detail"], {"status": 403})

    def test_flush_on_close(self):
        for _ in range(4):
            self.w.event("ann", "gallery mark", "c1", None, frequent=True)
        self.w.close()
        got = lines(self.tmp)
        self.assertEqual(len(got), 2)
        self.assertEqual(got[1]["detail"]["n"], 4)
        self.assertFalse(self.w.submit("ann", "late", None))               # closed: dropped, never raises

    def test_signin_once_per_day_per_door(self):
        self.assertTrue(self.w.signin("ann", "tailnet key", "key link"))
        self.assertFalse(self.w.signin("ann", "tailnet key", "session"))
        self.assertTrue(self.w.signin("ann", "access", "cloudflare"))
        self.assertTrue(self.w.signin("bob", "tailnet key"))
        self.now[0] += 86400
        self.assertTrue(self.w.signin("ann", "tailnet key", "session"))
        self.w.drain()
        self.assertEqual(len([x for x in lines(self.tmp) if x["action"] == "signed in"]), 4)

    def test_flood_is_bounded(self):
        """50,000 frequent events: anonymous ones over distinct targets are one window; signed-in ones over distinct
        targets stop at max_windows, the rest fold into one "other" window that still counts."""
        got = []
        w = PA.Writer(self.tmp, clock=lambda: self.now[0], write=lambda *a: got.append(a))
        self.addCleanup(w.close)
        for i in range(50_000):
            w.event("-", "gallery mark", f"c{i}", {"status": 401}, frequent=True)
        w.drain()
        self.assertEqual((len(w._windows), len(got)), (1, 1))
        self.assertIsNone(got[0][3])                                         # no target for anonymous
        for i in range(50_000):
            w.event("ann", "upload chunk", f"us_{i:012x}", {"status": 200}, frequent=True)
        w.drain()
        self.assertLessEqual(len(w._windows), PA.MAX_WINDOWS + 1)
        self.assertLessEqual(len(got), PA.MAX_WINDOWS + 2)                   # one per real window + "other" + anon
        w.flush()
        w.drain()
        other = [a for a in got if a[2] == PA.OVERFLOW_KEY[1]]
        self.assertEqual(len(other), 2)                                      # its first line + its counter line
        self.assertEqual(other[1][4]["n"], 50_000 - (PA.MAX_WINDOWS - 1))    # the anon window holds one slot
        self.assertEqual([a for a in got if a[1] == "-" and a[2] == "gallery mark"][-1][4]["n"], 50_000)

    def test_locked_log_cannot_hang_shutdown(self):
        import threading
        gate = threading.Event()
        self.addCleanup(gate.set)
        w = PA.Writer(self.tmp, write=lambda *a: gate.wait(30))              # the OS lock never comes free
        for i in range(20):
            w.event("ann", "feedback filed", None, {"status": 303})
        t0 = time.monotonic()
        with self.assertLogs("cj.audit", "ERROR"):
            w.close(timeout=0.5)
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertTrue(w.closed)

    def test_full_queue_drops_and_logs(self):
        w = PA.Writer(self.tmp, max_pending=0)
        self.addCleanup(w.close)
        with self.assertLogs("cj.audit", "ERROR"):
            self.assertFalse(w.submit("ann", "x", None))
        self.assertEqual(w.dropped, 1)


class PortalAggregation(unittest.TestCase):
    """Through the real Gate: the first gallery mark at once, the rest counted, the counter line at shutdown."""

    def test_counter_line_flushed_on_shutdown(self):
        clean_env(self)
        env = T.Env()
        self.addCleanup(env.cleanup)
        p = T.Portal(env.config())
        try:
            tok = T.login(p.port, env.key)
            h = {**T.cookie(tok), "Origin": p.origin}
            for _ in range(3):
                T.request(p.port, "POST", "/gallery/c1/api/event", h, b"{}")
            p.app.writer.drain()
            marks = [x for x in lines(env.state) if x["action"] == "gallery mark"]
            self.assertEqual(len(marks), 1)
            self.assertEqual(marks[0]["target"], "c1")
            signins = [x for x in lines(env.state) if x["action"] == "signed in"]
            T.login(p.port, env.key)                                        # a second key login the same day
            p.app.writer.drain()
            self.assertEqual(len(signins), 1)
            self.assertEqual(len([x for x in lines(env.state) if x["action"] == "signed in"]), 1)
        finally:
            p.stop()
        marks = [x for x in lines(env.state) if x["action"] == "gallery mark"]
        self.assertEqual(len(marks), 2)
        self.assertEqual(marks[1]["detail"]["n"], 3)


class AdminPage(Base):
    def test_render_counter_agent_cli_door_and_filters(self):
        tmp = Path(tempfile.mkdtemp(prefix="cj_r9adm_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        cli = tmp / PA.CLI_FILE
        cli.write_text(json.dumps({"t": "2026-11-12T16:05:00+00:00", "actor": "pujan", "action": "feedback status",
                                   "target": "#3", "detail": {"via": "claude", "to": "Done"}}) + "\n"
                       + json.dumps({"time": "2026-11-12T09:00:00+00:00", "actor": "ann", "action": "cancel",
                                     "job": "j1", "reason": "old format"}) + "\n", encoding="utf-8")
        own = PA.audit_path(self.env.state)
        before = len(lines(self.env.state))
        PA.audit(self.env.state, "ann", "review save", "pkgA",
                 {"from": "2026-11-12T15:20:00+00:00", "to": "2026-11-12T15:30:00+00:00", "n": 14})
        PA.audit(self.env.state, "-", "feedback filed", None, {"status": 401})
        self.assertTrue(own.exists() and len(lines(self.env.state)) == before + 2)
        with mock.patch.dict(os.environ, {PA.EXTRA_ENV: str(cli)}):
            st, b = self.send("GET", "/admin/audit", origin=False)
            js = self.send("GET", "/admin/admin.js", origin=False)[1].decode()
        text = b.decode()
        self.assertEqual(st, 200)
        self.assertIn("Filters search the last 200 lines.", text)
        self.assertIn("Logged: every change made in the portal", text)
        self.assertNotIn("people, job cancels and releases", text)            # the old, wrong statement
        self.assertIn("14 × review save", text)
        t0 = datetime.fromisoformat("2026-11-12T15:20:00+00:00").astimezone()
        t1 = datetime.fromisoformat("2026-11-12T15:30:00+00:00").astimezone()
        self.assertIn(f"{t0:%Y-%m-%d %H:%M}–{t1:%H:%M}", text)
        self.assertIn('data-who="claude for pujan (unverified)"', text)       # never mixed with portal sign-ins
        self.assertNotIn('data-who="claude for pujan"', text)
        self.assertRegex(text, r'<b class="cj-wrap">claude for pujan</b> <span class="cj-admin-ai"[^>]*>AI</span>'
                               r' <span class="cj-admin-ai cj-admin-unvtag"[^>]*>unverified</span>')
        self.assertIn('cj-admin-unv">laptop CLI (unverified)</span>', text)
        self.assertIn("refused 401", text)
        self.assertIn('data-who="ann (unverified)" data-kind="jobs"', text)    # old Jobs format still read
        self.assertIn('<option value="claude for pujan (unverified)">', text)
        self.assertIn('<option value="feedback">feedback</option>', text)
        self.assertIn('id="cj-admin-audf" hidden', text)                       # shown by admin.js only
        self.assertIn("data-cj-aud-filter", js)
        self.assertIn(">tailnet key</span>", text)                             # this door, named
        self.assertNotIn(str(tmp), text)

    def test_audit_row_kinds(self):
        self.assertEqual(AD.aud_kind("prediction set created"), "predict")
        self.assertEqual(AD.aud_kind("review save"), "review")
        self.assertEqual(AD.aud_kind("person_added"), "admin")
        self.assertEqual(AD.aud_kind("feedback filed", 403), "refused")
        self.assertEqual(AD.aud_kind("signed in"), "sign-in")
        r = AD.audit_row({"t": "2026-11-12T10:00:00+00:00", "actor": "<x>", "action": "a_b", "target": None,
                          "detail": {"via": "claude"}})
        self.assertEqual((r["who"], r["ai"], r["what"]), ("claude for <x>", True, "a b"))
        self.assertIn("&lt;x&gt;", AD.render_audit([{"t": "2026-11-12T10:00:00+00:00", "actor": "<x>",
                                                     "action": "a", "target": None, "detail": None}], None))


class About(Base):
    def test_anonymous_never_gets_the_text(self):
        self.assertNotIn("/about", APP.OPEN)
        doc = self.env.tmp / "about.json"
        doc.write_text(json.dumps({"platform": "PLATFORM-TEXT-zz", "credit": "<b>Lab</b>", "support": "FUND-zz"}),
                       encoding="utf-8")
        with mock.patch.dict(os.environ, {APP.ABOUT_ENV: str(doc)}):
            st, b = self.send("GET", "/about", who=None, origin=False)
            self.assertEqual(st, 401)
            self.assertNotIn(b"PLATFORM-TEXT-zz", b)
            st, b = self.send("GET", "/about", origin=False, extra={"Cookie": f"{A.COOKIE}=forged"})
            self.assertEqual(st, 403)
            self.assertNotIn(b"PLATFORM-TEXT-zz", b)
            st, b = self.send("GET", "/about", who="basic", origin=False)
        text = b.decode()
        self.assertEqual(st, 200)
        self.assertIn("PLATFORM-TEXT-zz", text)
        self.assertIn("&lt;b&gt;Lab&lt;/b&gt;", text)
        self.assertIn("How to cite", text)
        self.assertIn(APP.ABOUT_UNSET, text)                                   # cite: not set
        self.assertIn("Portal version:", text)
        self.assertNotIn(str(self.env.tmp), text)
        self.assertFalse(any(p.name.startswith("about") for p in (APP.HERE / "static").iterdir()))

    def test_missing_file_placeholders(self):
        with mock.patch.dict(os.environ, {APP.ABOUT_ENV: str(self.env.tmp / "nope.json")}):
            st, b = self.send("GET", "/about", who="basic", origin=False)
        self.assertEqual(st, 200)
        self.assertEqual(b.decode().count(APP.ABOUT_UNSET), len(APP.ABOUT_KEYS))

    def test_header_links_about(self):
        js = self.send("GET", "/cj/header.js", origin=False)[1].decode()
        self.assertIn("'/about'", js)


class StripEnv(unittest.TestCase):
    def setUp(self):
        clean_env(self)

    def test_main_removes_agent_env_and_logs_once(self):
        env = T.Env()
        self.addCleanup(env.cleanup)
        os.environ["CRYOEV_VIA"], os.environ["CRYOEV_FOR"] = "claude", "pujan"
        with mock.patch("uvicorn.run") as run, mock.patch("logging.basicConfig"), \
                mock.patch("builtins.print") as out:
            rc = APP.main(["--registry", str(env.registry), "--roster", str(env.roster), "--state-dir",
                           str(env.state), "--review-package", str(env.tmp / "none")])
        self.assertEqual(rc, 0)
        run.assert_called_once()
        self.assertNotIn("CRYOEV_VIA", os.environ)
        self.assertNotIn("CRYOEV_FOR", os.environ)
        said = [str(c) for c in out.call_args_list if "CRYOEV_VIA" in str(c)]
        self.assertEqual(len(said), 1)
        with mock.patch("uvicorn.run"), mock.patch("logging.basicConfig"), mock.patch("builtins.print") as out:
            APP.main(["--registry", str(env.registry), "--roster", str(env.roster), "--state-dir", str(env.state),
                      "--review-package", str(env.tmp / "none")])
        self.assertFalse([c for c in out.call_args_list if "CRYOEV_VIA" in str(c)])   # nothing to remove: silent

    def test_import_keeps_the_env(self):
        code = ("import os; os.environ['CRYOEV_VIA']='claude'; os.environ['CRYOEV_FOR']='pujan'; "
                "import portal.app, portal.modules.admin, portal.audit; "
                "print(os.environ.get('CRYOEV_VIA'), os.environ.get('CRYOEV_FOR'))")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(T.REPO), capture_output=True, text=True,
                           timeout=120)
        self.assertEqual(r.stdout.strip(), "claude pujan", r.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
