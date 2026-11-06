"""
Home page and manifests: status cards from a live.json written by jobs/monitor.py --once (temp queues built by
jobs/tests/jobtest.py, as registry/tests does), module cards from modules.json, brand.json, escaping.

  python -m unittest discover -s portal/tests -v
"""
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import app as APP  # noqa: E402
from portal import auth as A  # noqa: E402

PORTAL = T.REPO / "portal"


class Manifests(unittest.TestCase):
    def test_modules_json(self):
        doc = json.loads((PORTAL / "modules.json").read_text(encoding="utf-8"))
        mods = doc["modules"]
        ids = [m["id"] for m in mods]
        self.assertEqual(len(ids), len(set(ids)))
        for m in mods:
            for f in ("id", "name", "route", "icon", "roles", "health"):
                self.assertIn(f, m, (m["id"], f))
        by = {m["id"]: m for m in mods}
        on = [i for i in ids if by[i]["status"] == "on"]
        self.assertTrue({"home", "review", "sandbox", "history"} <= set(on))      # the core is always on
        for i in on:                                                         # anything on has a real route
            self.assertTrue(by[i]["route"] or by[i].get("external"), i)
        self.assertEqual([by[i]["name"] for i in ids if by[i].get("nav")],     # the nav, in order
                         ["Home", "Data", "Jobs", "Review", "Predict", "Results", "My reviews", "Train", "Admin", "Sandbox"])
        self.assertFalse(by["history"]["nav"])                               # History: hidden "classic" page
        self.assertEqual((by["home"]["route"], by["history"]["route"], by["review"]["route"]), ("/", "/history/", "/review/"))
        for m in mods:
            self.assertTrue(set(m["roles"]) <= {"admin", "advanced", "basic"}, m["id"])
            self.assertIn("admin", m["roles"], m["id"])
            self.assertIn(m["status"], ("on", "coming"), m["id"])
        self.assertEqual(by["review"]["roles"], ["admin"])                   # the real review stays admin only
        self.assertEqual(by["history"]["roles"], ["admin"])
        self.assertEqual(by["admin"]["roles"], ["admin"])
        self.assertIsNone(by["sandbox"]["route"])                            # sandbox is NOT under the shell
        self.assertEqual(by["sandbox"]["external"]["public_port"], 8443)
        self.assertFalse(any(str(m.get("route") or "").startswith("/sandbox") for m in mods))
        self.assertTrue({i for i in ids if by[i]["status"] == "coming"}           # only planned modules wait
                        <= {"data", "predict", "results", "train", "admin"})

    def test_brand_json(self):
        brand = json.loads((PORTAL / "static" / "brand.json").read_text(encoding="utf-8"))
        self.assertEqual((brand["name"], brand["wordmark"]), ("CryoEV", "CryoEV"))
        self.assertIn("tagline", brand)

    def test_css_tokens(self):
        """Direction A's values under the frozen names, scoped to the shell's own pages and the bar, never :root
        (History includes cj.css as the portal serves it and keeps its own look). Was test_css_reuses_history_tokens:
        the values changed on purpose (design A), so the test now checks A's values and the scoping instead."""
        css = (PORTAL / "static" / "cj.css").read_text(encoding="utf-8")
        light = ("--bg:#f3f5f7", "--fg:#16202a", "--mut:#58636f", "--card:#ffffff", "--line:#dde3e9", "--acc:#1d5f99",
                 "--accbg:#e3eef8", "--prod:#12703f", "--prodbg:#ddf2e5", "--warn:#8a5300", "--warnbg:#fbefd9")
        dark = ("--bg:#0f151b", "--fg:#e6ecf1", "--mut:#9aa8b5", "--card:#172029", "--line:#2a3642", "--acc:#7db8ec",
                "--accbg:#16304a", "--prod:#6fd49a", "--prodbg:#133326", "--warn:#f0c06a", "--warnbg:#3a2b10")
        for decl in light + dark + ("--dim:.5", "--hot:#c0135a", "--hot2:#b8431a", "--hot:#ff5c9b", "--hot2:#ff8f63"):
            self.assertIn(decl, css)
        self.assertIn("prefers-color-scheme:dark", css)
        roots = re.findall(r":root\{([^}]*)\}", css)
        self.assertTrue(roots)                                                # control: the pattern finds :root blocks
        for name in ("bg", "fg", "mut", "card", "line", "acc", "accbg", "prod", "prodbg", "warn", "warnbg", "dim"):
            for block in roots:                                               # frozen tokens are never set on :root
                self.assertNotIn(f"--{name}:", block)
        self.assertNotIn("@import", css)
        self.assertNotIn("url(", css)                                         # no web fonts: nothing is fetched
        self.assertNotRegex(css, r"(?m)^\s*header\s*[,{]")                    # the bar is not a <header>

    def test_external_href(self):
        m = {"external": {"public_port": 8443, "local_port": 8765, "path": "/"}}
        self.assertEqual(APP.external_href(m, "box.example.ts.net:8446"), "https://box.example.ts.net:8443/")
        self.assertEqual(APP.external_href(m, "127.0.0.1:8770"), "http://127.0.0.1:8765/")
        self.assertEqual(APP.external_href(m, "[::1]:8770"), "http://[::1]:8765/")
        self.assertIsNone(APP.external_href(m, 'x"><script>:8446'))
        self.assertIsNone(APP.external_href(m, None))


class Home(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(T.REPO / "jobs" / "tests"))
        import jobtest as J
        cls.env = T.Env()
        t = cls.env.tmp
        a = t / "qa"
        (a / "q").mkdir(parents=True)
        J.watch_fixture(a, job="job_20261104_live")
        J.running_fixture(a, progress={"batch": 8, "batch_asked": 16}, patience=300)
        cls.up = T.Upstream()
        lap = t / "lap"
        lap.mkdir()
        env = {k: v for k, v in os.environ.items() if k != "CRYOEV_REGISTRY"}
        env["CRYOEV_LAPTOP_ROOT"] = str(lap)
        servers = f"review={cls.up.port},sandbox={T.free_port()}"
        r = subprocess.run([sys.executable, str(T.REPO / "jobs" / "monitor.py"), "--share", str(a), "--once",
                            "--out", str(cls.env.registry / "live" / "live.json"), "--servers", servers],
                           capture_output=True, text=True, env=env, timeout=120)
        assert r.returncode == 0, r.stderr
        cls.live = json.loads((cls.env.registry / "live" / "live.json").read_text(encoding="utf-8"))
        cls.p = T.Portal(cls.env.config(review_upstream=cls.up.url))
        cls.token = T.login(cls.p.port, cls.env.key)

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.up.stop()
        cls.env.cleanup()

    def home(self, host=None):
        h = T.cookie(self.token)
        if host:
            h["Host"] = host
        st, _, body = T.request(self.p.port, "GET", "/", h)
        self.assertEqual(st, 200)
        return body.decode("utf-8")

    def test_status_from_monitor_snapshot(self):
        page = self.home()
        run = self.live["queues"][0]["running"]
        self.assertEqual((run["epoch"], run["epochs"]), (4, 300))           # control: the snapshot says so
        self.assertIn("From live.json", page)
        self.assertIn('<span class="cj-dot up" aria-hidden="true"></span><b>Review</b>', page)
        self.assertIn('<span class="cj-dot down" aria-hidden="true"></span><b>Sandbox</b>', page)
        self.assertIn("1 <small>of 2 up</small>", page)
        self.assertIn("alive · heartbeat", page)
        self.assertIn("job_20261104_live", page)
        self.assertIn("epoch 4/300", page)
        self.assertIn(f"ETA {APP.dur(run['eta_s'])}", page)
        self.assertIn("Last backup", page)
        self.assertIn(">Unknown<", page)
        # the training snapshot: the embedded slim snapshot for CJLive (live.js), refreshed from History's live.json
        m = re.search(r'<script type="application/json" id="cj-live-data">(.*?)</script>', page)
        slim = json.loads(m.group(1))
        self.assertEqual(slim["queues"][0]["running"]["epoch"], 4)
        self.assertEqual(set(slim), {"generated_utc", "interval_s", "queues"})
        self.assertIn('data-cj-live="compact"', page)
        self.assertIn('data-cj-live-src="/history/live/live.json"', page)
        self.assertIn('<script src="/cj/live.js" defer></script>', page)
        self.assertIn('<body class="cj-page" data-active="home">', page)     # rendered through base.html
        self.assertIn('<div id="cj-bar" class="cj-bar" role="banner"></div>', page)

    def test_module_cards(self):
        page = self.home()
        self.assertIn('id="cj-classic"', page)                               # History: only the footer link
        self.assertIn('href="/history/"', page)
        self.assertNotIn('data-id="history"', page)
        self.assertIn('href="/review/"', page)
        self.assertIn('<span class="cj-evh">eVHarmony</span>', page)
        self.assertEqual(page.count("cj-evh"), 1)                            # the gradient label: Review card only
        self.assertIn('href="http://127.0.0.1:8765/"', page)                 # sandbox: its own (local) address
        mods = json.loads((PORTAL / "modules.json").read_text(encoding="utf-8"))["modules"]
        coming = sum(1 for m in mods if m.get("nav") and m["status"] != "on" and "admin" in m["roles"])
        self.assertEqual(page.count('class="cj-mod off"'), coming)
        self.assertEqual(page.count(">coming<"), coming)
        order = re.findall(r'class="cj-mod[^"]*" data-id="([a-z]+)"', page)
        self.assertEqual(order, ["data", "jobs", "review", "predict", "results", "reviews", "train", "admin", "sandbox"])
        page = self.home("box.example.ts.net:8446")
        self.assertIn('href="https://box.example.ts.net:8443/"', page)

    def test_roles_see_their_modules_only(self):
        keys = {r: A.add_key(self.env.roster, f"user-{r}", r) for r in ("advanced", "basic")}
        try:
            want = {"advanced": ["data", "jobs", "predict", "results", "reviews", "train"],
                    "basic": ["predict", "results", "reviews"]}
            for role, key in keys.items():
                tok = T.login(self.p.port, key)
                st, _, body = T.request(self.p.port, "GET", "/", T.cookie(tok))
                page = body.decode("utf-8")
                self.assertEqual(st, 200)
                self.assertEqual(re.findall(r'class="cj-mod[^"]*" data-id="([a-z]+)"', page), want[role], role)
                self.assertNotIn('id="cj-status"', page, role)               # status = History's data: admin only
                self.assertNotIn("cj-live-data", page, role)
                self.assertNotIn("cj-classic", page, role)
                self.assertNotIn("job_20261104_live", page, role)
                self.assertIn(f"signed in as user-{role}", page)
        finally:
            for r in keys:
                A.revoke(self.env.roster, f"user-{r}")

    def test_stale_and_missing_snapshot(self):
        path = self.env.registry / "live" / "live.json"
        keep = path.read_bytes()
        try:
            self.env.write_live(T.live_doc(age_s=3600))
            page = self.home()
            self.assertIn("stale", page)
            self.assertNotIn('class="cj-dot up"', page)
            path.unlink()
            page = self.home()
            self.assertIn("No live snapshot", page)
            self.assertIn("unknown", page)
        finally:
            path.write_bytes(keep)

    def test_values_escaped(self):
        path = self.env.registry / "live" / "live.json"
        keep = path.read_bytes()
        try:
            q = T.live_doc()["queues"]
            q[0]["running"]["job"] = "<script>alert(1)</script>"
            q[0]["watcher"]["machine"] = "<img src=x onerror=alert(1)>"
            self.env.write_live(T.live_doc(queues=q))
            page = self.home()
            self.assertNotIn("<script>alert", page)
            self.assertNotIn("<img src=x", page)
            self.assertIn("&lt;script&gt;", page)
        finally:
            path.write_bytes(keep)


class ReviewPackageDefault(unittest.TestCase):
    def test_from_registry(self):
        env = T.Env()
        try:
            reg = env.registry
            pk_old, pk_new = env.tmp / "pkg_old", env.tmp / "pkg_new"
            for p, k in ((pk_old, "old-key"), (pk_new, "new-key")):
                p.mkdir()
                (p / ".review_key").write_text(k + "\n", encoding="utf-8")
            jobs = [{"id": "job_20260924_old", "type": "revise", "port": 8766, "status": "in_progress", "package": "old"},
                    {"id": "job_20261029_new", "type": "revise", "port": 8766, "status": "in_progress", "package": "new"},
                    {"id": "job_20261101_done", "type": "revise", "port": 8766, "status": "done", "package": "old"},
                    {"id": "job_20261102_sbx", "type": "revise", "port": 8765, "status": "ready", "package": "old"}]
            pkgs = [{"id": "old", "path": str(pk_old)}, {"id": "new", "path": str(pk_new)}]
            for ent, rows in (("job", jobs), ("review_package", pkgs)):
                (reg / f"{ent}.jsonl").write_text("".join(json.dumps({"schema": f"{ent}/1", **r}) + "\n"
                                                          for r in rows), encoding="utf-8")
            pkg = APP.registry_review_package(reg)
            self.assertEqual(pkg, pk_new)
            self.assertEqual(APP.read_review_key(pkg), "new-key")
            self.assertIsNone(APP.registry_review_package(env.tmp / "nowhere"))
            self.assertIsNone(APP.read_review_key(env.tmp / "nowhere"))
        finally:
            env.cleanup()


if __name__ == "__main__":
    unittest.main()
