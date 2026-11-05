"""
Home page and manifests: status cards from a live.json written by jobs/monitor.py --once (temp queues built by
jobs/tests/jobtest.py, as registry/tests does), module cards from modules.json, brand.json, escaping.

  python -m unittest discover -s portal/tests -v
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import app as APP  # noqa: E402

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
        self.assertEqual([i for i in ids if by[i]["status"] == "on"], ["home", "history", "review", "sandbox"])
        self.assertEqual((by["home"]["route"], by["history"]["route"], by["review"]["route"]), ("/", "/history/", "/review/"))
        self.assertIsNone(by["sandbox"]["route"])                            # sandbox is NOT under the shell
        self.assertEqual(by["sandbox"]["external"]["public_port"], 8443)
        self.assertFalse(any(str(m.get("route") or "").startswith("/sandbox") for m in mods))
        self.assertTrue({"predict", "results", "train", "admin"} <= {i for i in ids if by[i]["status"] == "coming"})

    def test_brand_json(self):
        brand = json.loads((PORTAL / "static" / "brand.json").read_text(encoding="utf-8"))
        self.assertEqual((brand["name"], brand["wordmark"]), ("CJ", "CJ"))
        self.assertIn("tagline", brand)

    def test_css_reuses_history_tokens(self):
        sys.path.insert(0, str(T.REPO / "registry"))
        import history_page as hp
        css = (PORTAL / "static" / "cj.css").read_text(encoding="utf-8")
        for decl in ("--bg:#fafaf9", "--fg:#1c1917", "--mut:#6b6560", "--card:#fff", "--line:#e3e0dc", "--acc:#2456c7",
                     "--bg:#141312", "--acc:#8fb0ff"):
            self.assertIn(decl, hp.CSS)                                       # the history page's own values ...
            self.assertIn(decl, css)                                          # ... repeated, so injection is a no-op
        self.assertIn("prefers-color-scheme:dark", css)

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
        self.assertIn("Live snapshot", page)
        self.assertIn('<span class="cj-dot up" aria-hidden="true"></span><b>Review</b>', page)
        self.assertIn('<span class="cj-dot down" aria-hidden="true"></span><b>Sandbox</b>', page)
        self.assertIn("alive · heartbeat", page)
        self.assertIn("job_20261104_live", page)
        self.assertIn("epoch 4/300", page)
        self.assertIn(f"ETA {APP.dur(run['eta_s'])}", page)
        self.assertIn("Last backup", page)
        self.assertIn("<b>unknown</b>", page)

    def test_module_cards(self):
        page = self.home()
        self.assertIn('href="/history/"', page)
        self.assertIn('href="/review/"', page)
        self.assertIn('href="http://127.0.0.1:8765/"', page)                 # sandbox: its own (local) address
        self.assertEqual(page.count('class="cj-mod off"'), 4)
        self.assertEqual(page.count(">coming<"), 4)
        page = self.home("box.example.ts.net:8446")
        self.assertIn('href="https://box.example.ts.net:8443/"', page)

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
