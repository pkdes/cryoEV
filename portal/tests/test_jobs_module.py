"""
Jobs module (portal/modules/jobs.py): roles through the Gate, every section from a fixture live.json + registry,
the API matching the page, stale / missing snapshots, another person's prediction job hidden, escaping, the
registry cache, a real jobs/monitor.py --once snapshot as a known-answer control, and the page in a browser
(Playwright, Edge channel: 390 and 1280 px, light and dark, no horizontal scroll, same-origin requests only, no
console errors, the 30 s refresh). Screenshots go to $CJ_SHOT_DIR when set (else a temp folder, removed afterwards).

Temp roster / registry / live.json only (cjtest.Env); never the real ones.

  python -m unittest discover -s portal/tests -v
"""
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import jobs as J  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

XSS = '<img src=x onerror=alert(1)> tiles'
REASON_XSS = 'run failed: <script>alert("r")</script>'


def local(hours_ago: float) -> str:
    return (datetime.now() - timedelta(hours=hours_ago)).isoformat(timespec="seconds")


def queues_fixture() -> list:
    """Three queues as jobs/monitor.py writes them: a busy one, one whose running job is hung, one offline."""
    a = {"share": "\\\\gpu-desktop\\cj", "reach": "ok", "detail": None,
         "watcher": {"display": "alive", "state": "running", "detail": None, "job": "job_20261104_train_live",
                     "age_s": 40, "stale": False, "machine": "gpu-desktop"},
         "jobs": [{"job": "job_20261101_train_old", "state": "done", "reason": None, "pulled": True,
                   "cancel_requested": False},
                  {"job": "job_20261102_train_blocked", "state": "blocked", "reason": REASON_XSS, "pulled": False,
                   "cancel_requested": False},
                  {"job": "job_20261104_train_live", "state": "running", "reason": None, "pulled": False,
                   "cancel_requested": False},
                  {"job": "job_20261105_predict_mine", "state": "queued", "reason": None, "pulled": False,
                   "cancel_requested": False},
                  {"job": "job_20261105_predict_theirs", "state": "queued", "reason": None, "pulled": False,
                   "cancel_requested": False},
                  {"job": "job_20261105_train_after", "state": "queued", "reason": None, "pulled": False,
                   "cancel_requested": True},
                  {"job": "job_20261105_train_hold", "state": "held", "reason": None, "pulled": False,
                   "cancel_requested": False},
                  {"job": "job_20261105_train_next", "state": "queued", "reason": None, "pulled": False,
                   "cancel_requested": False}],
         "running": {"job": "job_20261104_train_live", "run": "2026-11-04_120000_y8m_1440_16b_300ep_Run_1",
                     "phase": "training", "epoch": 12, "epochs": 300, "elapsed_s": 900, "batch": 16,
                     "batch_asked": 16, "map50": 0.5, "map50_95": 0.25, "best_epoch": 12, "patience": 100,
                     "max_hours": 48, "epoch_median_s": 13, "last_epoch_age_s": 20, "hung": False,
                     "hung_after_s": 600, "eta_s": 7380, "eta_patience_bound_s": None, "eta_capped": False,
                     "source": "progress.json", "cancel_requested": False, "watcher_on_it": True,
                     "series": {"epoch": list(range(1, 13)), "box": [2.0 / i for i in range(1, 13)],
                                "seg": [3.0 / i for i in range(1, 13)], "cls": [1.0 / i for i in range(1, 13)],
                                "dfl": [1.5 / i for i in range(1, 13)], "map50": [0.04 * i for i in range(1, 13)],
                                "map50_95": [0.02 * i for i in range(1, 13)]}},
         "finished": [{"job": "job_20261101_train_old", "status": "done", "reason": None, "ended": local(2),
                       "run": "r1", "run_status": "finished", "pulled": True,
                       "metrics": {"mask_map50": 0.61, "mask_map50_95": 0.321}, "best_epoch": 140},
                      {"job": "job_20261102_train_blocked", "status": "blocked", "reason": REASON_XSS,
                       "ended": local(30), "run": None, "run_status": None, "pulled": False, "metrics": None,
                       "best_epoch": None}]}
    b = {"share": "\\\\queue-share\\cj", "reach": "ok", "detail": None,
         "watcher": {"display": "alive", "state": "running", "detail": None, "job": "job_20261103_train_stuck",
                     "age_s": 25, "stale": False, "machine": "lab-login"},
         "jobs": [{"job": "job_20261103_train_stuck", "state": "running", "reason": None, "pulled": False,
                   "cancel_requested": False}],
         "running": {"job": "job_20261103_train_stuck", "phase": "training", "epoch": 40, "epochs": 100,
                     "elapsed_s": 20000, "epoch_median_s": 60, "last_epoch_age_s": 7200, "hung": True,
                     "hung_after_s": 600, "eta_s": 3600, "eta_patience_bound_s": None, "eta_capped": False,
                     "map50_95": 0.3, "cancel_requested": False, "series": {"epoch": [1], "map50_95": [0.3]}},
         "finished": []}
    c = {"share": "\\\\home-laptop\\cj", "reach": "offline", "detail": "not found: \\\\home-laptop\\cj"}
    return [a, b, c]


def registry_fixture() -> list:
    def rec(id, name, type, where, status, logged, **kw):
        return {"schema": "job/1", "id": id, "logged": logged, "name": name, "type": type, "where": where,
                "status": status, **kw}
    return [
        rec("job_20261102_prelipo_review_v2", "Pre-lipo review (v2)", "revise", "app", "in_progress", local(50),
            progress={"done": 3, "total": 170, "unit": "images reviewed"}, n_images=170, who="tester",
            package="prelipo_review_v2"),
        rec("job_20261102_annotate_blind", "Annotate blind tiles", "annotate", "roboflow", "blocked", local(49),
            blocked_reason="tiles not uploaded yet", progress={"done": 0, "total": 12, "unit": "images labeled"}),
        rec("job_20261103_annotate_ready", XSS, "annotate", "app", "ready", local(40)),
        rec("job_20261101_check_done", "Check v5 on 20261027", "check", "laptop", "done", local(4)),
        rec("job_20261101_annotate_cancelled", "Old labeling", "annotate", "roboflow", "cancelled", local(3)),
        rec("job_20261104_train_live", "Train v6 on lv_20261103", "train", "desktop", "ready", local(30)),
        rec("job_20261105_train_notpushed", "Train v7 (not pushed)", "train", "desktop", "ready", local(1)),
        rec("job_20261105_predict_mine", "Predict my tiles", "predict", "desktop", "ready", local(1), who="adv"),
        rec("job_20261105_predict_theirs", "Predict client-x tiles", "predict", "desktop", "ready", local(1),
            who="someone"),
    ]


def write_registry(root: Path, recs):
    (root / "job.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def section(page: str, key: str) -> str:
    """The HTML inside <section id="cj-jobs-s-<key>">."""
    m = re.search(r'<section class="cj-jobs-sec" id="cj-jobs-s-%s"[^>]*>(.*?)</section>' % key, page, re.S)
    assert m, key
    return m.group(1)


def badge(frag: str) -> int:
    return int(re.search(r'<span class="cj-jobs-n">(\d+)</span>', frag).group(1))


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.adv_key = A.add_key(cls.env.roster, "adv", "advanced")
        cls.bas_key = A.add_key(cls.env.roster, "bas", "basic")
        write_registry(cls.env.registry, registry_fixture())
        cls.env.write_live(T.live_doc(queues=queues_fixture()))
        cls.p = T.Portal(cls.env.config())
        cls.tok = {"admin": T.login(cls.p.port, cls.env.key), "advanced": T.login(cls.p.port, cls.adv_key),
                   "basic": T.login(cls.p.port, cls.bas_key)}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()

    def get(self, path, role="admin"):
        return T.request(self.p.port, "GET", path, T.cookie(self.tok[role]))

    def page(self, role="admin") -> str:
        st, h, body = self.get("/jobs/", role)
        self.assertEqual(st, 200)
        self.assertEqual(T.hget(h, "cache-control"), "no-store")
        return body.decode("utf-8")

    def api(self, role="admin", q="") -> dict:
        st, h, body = self.get("/jobs/api/summary" + q, role)
        self.assertEqual(st, 200)
        self.assertEqual(T.hget(h, "cache-control"), "no-store")
        self.assertTrue(T.hget(h, "content-type").startswith("application/json"))
        return json.loads(body)


class Roles(Base):
    def test_module_declaration(self):
        self.assertEqual((J.PREFIX, J.ROLES), ("/jobs", ("admin", "advanced")))
        st, h, _ = self.get("/jobs")
        self.assertEqual((st, T.hget(h, "location")), (307, "/jobs/"))

    def test_admin_and_advanced_in_basic_403(self):
        for role in ("admin", "advanced"):
            with self.subTest(role=role):
                self.assertEqual(self.get("/jobs/", role)[0], 200)
                self.assertEqual(self.get("/jobs/api/summary", role)[0], 200)
        for path in ("/jobs/", "/jobs", "/jobs/api/summary"):
            with self.subTest(path=path):
                st, _, body = self.get(path, "basic")
                self.assertEqual(st, 403)
                self.assertNotIn(b"job_2026", body)
        self.assertEqual(self.get("/", "basic")[0], 200)              # control: basic is signed in, just not here
        st, _, _ = T.request(self.p.port, "GET", "/jobs/api/summary")  # no session at all
        self.assertEqual(st, 401)


class Sections(Base):
    def test_running_now(self):
        frag = section(self.page(), "running")
        self.assertEqual(badge(frag), 2)
        self.assertIn("Train v6 on lv_20261103", frag)                  # registry name for the desktop job
        self.assertIn("job_20261104_train_live", frag)
        self.assertIn("Epoch 12 / 300", frag)
        self.assertIn('<progress class="cj-jobs-bar" max="300" value="12"', frag)
        self.assertIn("2 h 3 min", frag)                                # ETA 7380 s
        self.assertIn("alive, heartbeat 40 s", frag)                    # watcher heartbeat age
        self.assertIn("0.250", frag)                                    # latest val mAP50-95
        self.assertIn("job_20261103_train_stuck", frag)
        self.assertIn("hung?", frag)
        self.assertIn("No new epoch for 2 h (expected every 1 min)", frag)
        self.assertIn("gpu-desktop", frag)
        self.assertIn("lab-login", frag)
        self.assertNotIn("Stale snapshot", self.page())

    def test_queue_in_order(self):
        frag = section(self.page(), "queue")
        order = ["job_20261105_predict_mine", "job_20261105_predict_theirs", "job_20261105_train_after",
                 "job_20261105_train_next", "job_20261105_train_hold"]
        pos = [frag.find(j) for j in order]
        self.assertTrue(all(p >= 0 for p in pos), pos)
        self.assertEqual(pos, sorted(pos))                              # queued in queue order, then held
        self.assertIn("cancel requested", frag)
        self.assertIn("held: starts after release", frag)
        self.assertNotIn("job_20261104_train_live", frag)               # running is not "waiting"
        self.assertNotIn("job_20261101_train_old", frag)                # done is not "waiting"
        self.assertEqual(badge(frag), 5)

    def test_needs_attention(self):
        frag = section(self.page(), "attention")
        self.assertEqual(badge(frag), 4)
        for text in ("Queue on \\\\home-laptop\\cj", "Can&#x27;t reach this queue", "job_20261103_train_stuck",
                     "job_20261102_train_blocked", "not pulled yet", "Annotate blind tiles", "tiles not uploaded yet"):
            self.assertIn(text, frag)
        self.assertIn(html.escape(REASON_XSS), frag)
        kinds = re.findall(r'cj-jobs-chip cj-jobs-t-warn">(\w+)</span>', frag)
        self.assertEqual(kinds, ["offline", "hung", "blocked", "blocked"])

    def test_review_and_annotation(self):
        frag = section(self.page(), "review")
        names = ["Pre-lipo review (v2)", "Annotate blind tiles", html.escape(XSS), "Train v7 (not pushed)"]
        pos = [frag.find(n) for n in names]
        self.assertTrue(all(p >= 0 for p in pos), pos)
        self.assertEqual(pos, sorted(pos))                              # in progress, blocked, ready
        self.assertIn("3 / 170 images reviewed", frag)
        self.assertIn("Review in the review app (port 8766)", frag)
        self.assertIn("Not on a desktop queue yet", frag)
        for on_queue in ("job_20261104_train_live", "Predict my tiles", "job_20261105_predict_theirs"):
            self.assertNotIn(on_queue, frag)                            # on a desktop queue: shown from live.json
        self.assertNotIn("Check v5", frag)                              # done: under Recently finished
        self.assertEqual(badge(frag), 4)

    def test_recently_finished(self):
        frag = section(self.page(), "finished")
        order = ["job_20261101_train_old", "Old labeling", "Check v5 on 20261027", "job_20261102_train_blocked"]
        pos = [frag.find(n) for n in order]
        self.assertTrue(all(p >= 0 for p in pos), pos)
        self.assertEqual(pos, sorted(pos))                              # newest first, both sources
        self.assertIn("mask mAP50-95 0.321", frag)
        self.assertRegex(frag, r"2 h( \d+ min)? ago")

    def test_api_matches_page(self):
        s = self.api()
        page = self.page()
        self.assertEqual(s["schema"], "cj_jobs/1")
        self.assertNotIn("html", s)
        self.assertEqual(s["counts"], {"running": 2, "queued": 4, "held": 1, "attention": 4, "review_open": 4,
                                       "finished": 4})
        r = {x["id"]: x for x in s["running"]}
        live = r["job_20261104_train_live"]
        self.assertEqual((live["kind"], live["done"], live["total"], live["unit"], live["pct"], live["hung"]),
                         ("train", 12, 300, "epochs", 4, False))
        self.assertEqual((live["eta_s"], live["watcher"]["age_s"], live["queue"]), (7380, 40, "gpu-desktop"))
        self.assertTrue(r["job_20261103_train_stuck"]["hung"])
        self.assertEqual([(x["id"], x["state"], x["position"]) for x in s["queue"]],
                         [("job_20261105_predict_mine", "queued", 1), ("job_20261105_predict_theirs", "queued", 2),
                          ("job_20261105_train_after", "queued", 3), ("job_20261105_train_next", "queued", 4),
                          ("job_20261105_train_hold", "held", None)])
        self.assertEqual([(x["kind"], x["source"]) for x in s["attention"]],
                         [("offline", "desktop"), ("hung", "desktop"), ("blocked", "desktop"), ("blocked", "registry")])
        self.assertEqual([x["status"] for x in s["review"]], ["in_progress", "blocked", "ready", "ready"])
        self.assertEqual([x["id"] for x in s["finished"]],
                         ["job_20261101_train_old", "job_20261101_annotate_cancelled", "job_20261101_check_done",
                          "job_20261102_train_blocked"])
        self.assertEqual([q["reach"] for q in s["queues"]], ["ok", "ok", "offline"])
        self.assertFalse(s["snapshot"]["stale"])
        for key in ("running", "queue", "attention", "review", "finished"):
            frag = section(page, key)
            self.assertEqual(badge(frag), len(s[key]), key)
            for x in s[key]:
                self.assertIn(html.escape(x["name"]), frag, (key, x["name"]))
        with_html = self.api(q="?html=1")
        self.assertEqual(set(with_html["html"]), set(J.SECTIONS))

        def norm(t):                                                    # "2 h ago" may tick between two requests
            return re.sub(r"(\d+ (d|h|min|s) )+ago", "ago", t)
        for key in ("running", "queue", "attention", "review", "finished"):
            self.assertEqual(norm(with_html["html"][key]), norm(section(page, key)))   # the refresh: same HTML

    def test_values_escaped(self):
        page = self.page()
        self.assertNotIn("<img src=x", page)
        self.assertNotIn('<script>alert("r")', page)
        self.assertIn(html.escape(XSS), page)
        self.assertNotIn(" on" + "error=", re.sub(r"&lt;img[^<]*", "", page))
        self.assertNotRegex(page, r"<[a-z]+[^>]*\son[a-z]+=")          # no inline event handlers anywhere


class Privacy(Base):
    def test_other_persons_prediction_job_hidden(self):
        adv_page, adv_api = self.page("advanced"), json.dumps(self.api("advanced"))
        for text in ("job_20261105_predict_theirs", "Predict client-x tiles", "someone"):
            self.assertNotIn(text, adv_page)
            self.assertNotIn(text, adv_api)
        self.assertIn("Someone else&#x27;s prediction job", section(adv_page, "queue"))
        self.assertIn("job_20261105_predict_mine", adv_page)            # their own: shown
        self.assertIn("job_20261105_train_next", adv_page)              # lab train jobs: shown
        q = self.api("advanced")["queue"]
        self.assertEqual([x["position"] for x in q if x["state"] == "queued"], [1, 2, 3, 4])   # place still counted
        admin_page = self.page("admin")                                 # control: admin sees it
        self.assertIn("job_20261105_predict_theirs", admin_page)
        self.assertIn("Predict client-x tiles", admin_page)


class Snapshots(Base):
    def swap_live(self, doc):
        path = self.env.registry / "live" / "live.json"
        keep = path.read_bytes()
        if doc is None:
            path.unlink()
        else:
            self.env.write_live(doc)
        self.addCleanup(path.write_bytes, keep)

    def test_stale_banner(self):
        self.swap_live(T.live_doc(age_s=3600, queues=queues_fixture()))
        page = self.page()
        self.assertIn("Stale snapshot.", page)
        self.assertIn("1 h old", page)
        self.assertIn("last known", section(page, "running"))
        s = self.api()
        self.assertTrue(s["snapshot"]["stale"])
        self.assertTrue(all(x["stale"] for x in s["running"]))
        self.assertEqual(s["snapshot"]["age_s"] // 60, 60)

    def test_fresh_has_no_banner(self):                                 # control for the banner test
        self.swap_live(T.live_doc(age_s=60, queues=queues_fixture()))   # 60 s < 3 x 30 s
        page = self.page()
        self.assertNotIn("Stale snapshot.", page)
        self.assertNotIn("last known", section(page, "running"))

    def test_missing_snapshot(self):
        self.swap_live(None)
        page = self.page()
        self.assertIn("No desktop snapshot.", page)
        self.assertIn("Unknown: no desktop snapshot.", section(page, "running"))
        self.assertIn("Pre-lipo review (v2)", section(page, "review"))   # the registry still shows
        s = self.api()
        self.assertFalse(s["snapshot"]["present"])
        self.assertEqual(s["running"], [])

    def test_predict_job_running(self):
        q = queues_fixture()[:1]
        q[0]["running"] = {"job": "job_20261105_predict_mine", "kind": "predict", "n_done": 40, "n_total": 120,
                           "phase": "predicting", "eta_s": 300, "hung": False}
        self.swap_live(T.live_doc(queues=q))
        frag = section(self.page("advanced"), "running")
        self.assertIn("Images 40 / 120", frag)
        self.assertIn("Predict my tiles", frag)
        self.assertIn("5 min", frag)
        r = self.api("advanced")["running"][0]
        self.assertEqual((r["kind"], r["done"], r["total"], r["unit"], r["pct"]), ("predict", 40, 120, "images", 33))
        q[0]["running"]["job"] = "job_20261105_predict_theirs"           # someone else's: progress yes, name no
        self.env.write_live(T.live_doc(queues=q))
        frag = section(self.page("advanced"), "running")
        self.assertIn("Someone else&#x27;s prediction job", frag)
        self.assertIn("Images 40 / 120", frag)
        self.assertNotIn("predict_theirs", frag)

    def test_watcher_stale_and_paused(self):
        q = queues_fixture()[:1]
        q[0]["watcher"].update(display="stale", age_s=900, stale=True)
        self.swap_live(T.live_doc(queues=q))
        att = self.api()["attention"]
        self.assertEqual(att[0]["kind"], "watcher")
        self.assertIn("last heartbeat 15 min ago; queued jobs will not start", att[0]["reason"])
        q[0]["watcher"].update(display="paused", age_s=30, stale=False, detail="STOP: GPU maintenance")
        self.env.write_live(T.live_doc(queues=q))
        att = self.api()["attention"]
        self.assertIn("STOP: GPU maintenance; 4 queued job(s) wait", att[0]["reason"])


class RegistryCache(unittest.TestCase):
    def test_one_read_per_ttl(self):
        root = Path(tempfile.mkdtemp(prefix="cj_jobs_reg_"))
        try:
            write_registry(root, registry_fixture()[:1])
            jobs, err, _ = J.registry_jobs(root, now=1000.0)
            self.assertIsNone(err)
            self.assertEqual(list(jobs), ["job_20261102_prelipo_review_v2"])
            write_registry(root, registry_fixture()[:2])
            self.assertEqual(len(J.registry_jobs(root, now=1000.0 + J.REGISTRY_TTL_S - 1)[0]), 1)   # cached
            self.assertEqual(len(J.registry_jobs(root, now=1000.0 + J.REGISTRY_TTL_S + 1)[0]), 2)   # re-read
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_missing_registry(self):
        jobs, err, _ = J.registry_jobs(Path(tempfile.gettempdir()) / "cj_jobs_no_such_dir_x")
        self.assertEqual((jobs, err), ({}, "registry folder not found"))
        self.assertEqual(J.registry_jobs(None)[1], "no registry folder configured")


class MonitorControl(unittest.TestCase):
    """Known answer: a real jobs/monitor.py --once snapshot of a fixture queue (jobs/tests/jobtest.py) says the
    running job is at epoch 4/300 with one finished job; the module must show exactly that."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(T.REPO / "jobs" / "tests"))
        import jobtest as JT
        cls.env = T.Env()
        share = cls.env.tmp / "qa"
        (share / "q").mkdir(parents=True)
        JT.watch_fixture(share, job="job_20261104_live")
        JT.running_fixture(share, progress={"batch": 8, "batch_asked": 16}, patience=300)
        JT.done_fixture(share, "job_20261103_done", metrics={"mask_map50": .6, "mask_map50_95": .31})
        lap = cls.env.tmp / "lap"
        lap.mkdir()
        env = {k: v for k, v in os.environ.items() if k != "CRYOEV_REGISTRY"}
        env["CRYOEV_LAPTOP_ROOT"] = str(lap)
        r = subprocess.run([sys.executable, str(T.REPO / "jobs" / "monitor.py"), "--share", str(share), "--once",
                            "--out", str(cls.env.registry / "live" / "live.json"), "--servers", ""],
                           capture_output=True, text=True, env=env, timeout=120)
        assert r.returncode == 0, r.stderr
        cls.live = json.loads((cls.env.registry / "live" / "live.json").read_text(encoding="utf-8"))
        cls.p = T.Portal(cls.env.config())
        cls.token = T.login(cls.p.port, cls.env.key)

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()

    def test_real_snapshot(self):
        run = self.live["queues"][0]["running"]
        self.assertEqual((run["epoch"], run["epochs"]), (4, 300))                   # the control itself
        st, _, body = T.request(self.p.port, "GET", "/jobs/api/summary", T.cookie(self.token))
        self.assertEqual(st, 200)
        s = json.loads(body)
        self.assertEqual([(x["id"], x["done"], x["total"]) for x in s["running"]], [("job_20261104_live", 4, 300)])
        self.assertEqual(s["running"][0]["watcher"]["display"], "alive")
        self.assertEqual(s["running"][0]["eta_text"], J.eta_text(run))
        self.assertEqual([(x["id"], x["status"]) for x in s["finished"]], [("job_20261103_done", "done")])
        self.assertEqual(s["finished"][0]["map50_95"], .31)
        self.assertEqual(s["queue"], [])
        self.assertEqual(s["attention"], [])


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

    def session(self, width, scheme, key=None):
        ctx = self.browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
        origin = self.p.origin
        rec = {"requests": [], "foreign": [], "errors": [], "console": []}
        ctx.route("**/*", lambda r: (r.continue_() if r.request.url.startswith(origin + "/")
                                     else (rec["foreign"].append(r.request.url), r.abort())))
        page = ctx.new_page()
        page.on("request", lambda r: rec["requests"].append(r.url))
        page.on("pageerror", lambda e: rec["errors"].append(str(e)))
        page.on("console", lambda m: m.type == "error" and rec["console"].append(m.text))
        page.goto(f"{origin}/login?k={key or self.env.key}")
        self.assertEqual(page.url, origin + "/")
        return ctx, page, rec

    def check_clean(self, rec):
        self.assertEqual(rec["foreign"], [])
        self.assertTrue(rec["requests"])
        self.assertTrue(all(u.startswith(self.p.origin + "/") for u in rec["requests"]),
                        [u for u in rec["requests"] if not u.startswith(self.p.origin + "/")])
        self.assertEqual(rec["errors"], [])
        self.assertEqual(rec["console"], [])

    def test_page_sizes_and_schemes(self):
        for scheme in ("light", "dark"):
            for width in (390, 1280):
                with self.subTest(width=width, scheme=scheme):
                    ctx, page, rec = self.session(width, scheme)
                    try:
                        page.goto(self.p.origin + "/jobs/")
                        page.wait_for_load_state("networkidle")
                        page.wait_for_selector("#cj-bar")
                        sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
                        self.assertLessEqual(sw, iw, "horizontal scroll")
                        self.assertEqual(page.locator("#cj-jobs-s-running article").count(), 2)
                        self.assertEqual(page.locator("#cj-jobs-s-queue li").count(), 5)
                        self.assertEqual(page.locator("#cj-jobs-s-attention article").count(), 4)
                        self.assertEqual(page.locator("#cj-jobs-s-review article").count(), 4)
                        self.assertEqual(page.locator("#cj-jobs-s-finished li").count(), 4)
                        self.assertEqual(page.locator("body img").count(), 0)          # the XSS name stayed text
                        bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
                        self.assertEqual(bg, "rgb(20, 19, 18)" if scheme == "dark" else "rgb(250, 250, 249)")
                        page.screenshot(path=str(self.shots / f"jobs_{width}_{scheme}.png"), full_page=True)
                        self.check_clean(rec)
                    finally:
                        ctx.close()

    def test_refresh_swaps_sections(self):
        path = self.env.registry / "live" / "live.json"
        keep = path.read_bytes()
        self.addCleanup(path.write_bytes, keep)
        ctx, page, rec = self.session(390, "light")
        try:
            page.goto(self.p.origin + "/jobs/")
            page.wait_for_load_state("networkidle")
            self.assertEqual(page.locator(".cj-jobs-banner").count(), 0)
            self.env.write_live(T.live_doc(age_s=3600, queues=queues_fixture()))
            with page.expect_response(lambda r: "/jobs/api/summary?html=1" in r.url) as resp:
                page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")   # what a 30 s tick does
            self.assertEqual(resp.value.status, 200)
            page.wait_for_selector(".cj-jobs-banner")
            self.assertIn("Stale snapshot", page.text_content(".cj-jobs-banner"))
            self.assertEqual(page.locator("#cj-jobs-s-running article").count(), 2)
            self.assertTrue(page.locator("#cj-jobs-err").is_hidden())
            page.screenshot(path=str(self.shots / "jobs_390_light_stale.png"), full_page=True)
            self.check_clean(rec)
        finally:
            ctx.close()

    def test_advanced_user_page(self):
        ctx, page, rec = self.session(390, "dark", key=self.adv_key)
        try:
            page.goto(self.p.origin + "/jobs/")
            page.wait_for_load_state("networkidle")
            self.assertIn("Someone else's prediction job", page.text_content("#cj-jobs-s-queue"))
            sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
            self.assertLessEqual(sw, iw)
            self.check_clean(rec)
        finally:
            ctx.close()


if __name__ == "__main__":
    unittest.main()
