"""
Jobs module (portal/modules/jobs.py): roles through the Gate, every section from a fixture live.json + registry,
the API matching the page, queue order (ready_at, else bullets), stale / missing snapshots, privacy failing closed
(someone else's predict job, a queue job the registry doesn't know, an unreadable registry), plain words for
non-admins (no commands, paths or raw errors), escaping, the registry cache, a real jobs/monitor.py --once
snapshot as a known-answer control, and the page in a browser (Playwright, Edge channel: 390 and 1280 px, light and
dark, no horizontal scroll, same-origin requests only, no console errors, the 30 s refresh). Screenshots go to
$CJ_SHOT_DIR when set (else a temp folder, removed afterwards).

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
HIDDEN = J.PRIVATE_NAME
TECH_WORDS = ("watcher", "heartbeat", "registry", "jobs/sync", "jobs/monitor", "jobs/watch", "live.json",
              "\\\\", "home-laptop", "train.log", "ready_at", "&lt;script", "run failed")


def local(hours_ago: float) -> str:
    return (datetime.now() - timedelta(hours=hours_ago)).isoformat(timespec="seconds")


def qjob(job, state, pulled=False, cancel=False, reason=None, ready_at=None):
    d = {"job": job, "state": state, "reason": reason, "pulled": pulled, "cancel_requested": cancel}
    if ready_at:
        d["ready_at"] = ready_at
    return d


def queues_fixture() -> list:
    """Three queues as jobs/monitor.py writes them: a busy one, one whose running job is hung, one offline.
    ready_at puts train_next first although its name sorts last."""
    a = {"share": "\\\\gpu-desktop\\cj", "reach": "ok", "detail": None,
         "watcher": {"display": "alive", "state": "running", "detail": None, "job": "job_20261104_train_live",
                     "age_s": 40, "stale": False, "machine": "gpu-desktop"},
         "jobs": [qjob("job_20261030_train_oldblock", "blocked", pulled=True, reason="old failure"),
                  qjob("job_20261101_train_old", "done", pulled=True),
                  qjob("job_20261102_train_blocked", "blocked", reason=REASON_XSS),
                  qjob("job_20261104_train_live", "running"),
                  qjob("job_20261105_predict_mine", "queued", ready_at=local(4)),
                  qjob("job_20261105_predict_theirs", "queued", ready_at=local(3)),
                  qjob("job_20261105_train_after", "queued", cancel=True, ready_at=local(2)),
                  qjob("job_20261105_train_hold", "held"),
                  qjob("job_20261105_train_next", "queued", ready_at=local(5))],
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
                       "best_epoch": None},
                      {"job": "job_20261030_train_oldblock", "status": "blocked", "reason": "old failure",
                       "ended": local(100), "run": None, "run_status": None, "pulled": True, "metrics": None,
                       "best_epoch": None}]}
    b = {"share": "\\\\queue-share\\cj", "reach": "ok", "detail": None,
         "watcher": {"display": "alive", "state": "running", "detail": None, "job": "job_20261103_train_stuck",
                     "age_s": 25, "stale": False, "machine": "lab-login"},
         "jobs": [qjob("job_20261103_train_stuck", "running")],
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

    def train(id, name, status="ready"):
        return rec(id, name, "train", "desktop", status, local(6), who="tester")
    return [
        rec("job_20261102_prelipo_review_v2", "Pre-lipo review (v2)", "revise", "app", "in_progress", local(50),
            progress={"done": 3, "total": 170, "unit": "images reviewed"}, n_images=170, who="tester",
            package="prelipo_review_v2"),
        rec("job_20261025_annotate_oldblock", "Old blocked labeling", "annotate", "roboflow", "blocked", local(100),
            blocked_reason="waiting on a decision"),
        rec("job_20261102_annotate_blind", "Annotate blind tiles", "annotate", "roboflow", "blocked", local(20),
            blocked_reason="tiles not uploaded yet", progress={"done": 0, "total": 12, "unit": "images labeled"}),
        rec("job_20261103_annotate_ready", XSS, "annotate", "app", "ready", local(40)),
        rec("job_20261101_check_done", "Check v5 on 20261027", "check", "laptop", "done", local(4)),
        rec("job_20261101_annotate_cancelled", "Old labeling", "annotate", "roboflow", "cancelled", local(3)),
        train("job_20261104_train_live", "Train v6 on lv_20261103"),
        train("job_20261103_train_stuck", "Train v6b"),
        train("job_20261101_train_old", "Train v5", "done"),
        train("job_20261102_train_blocked", "Train v5b", "blocked"),
        train("job_20261030_train_oldblock", "Train v4", "blocked"),
        train("job_20261105_train_after", "Train after"),
        train("job_20261105_train_hold", "Train hold"),
        train("job_20261105_train_next", "Train next"),
        rec("job_20261105_train_notpushed", "Train v7 (not pushed)", "train", "desktop", "ready", local(1)),
        rec("job_20261105_predict_mine", "Predict my tiles", "predict", "desktop", "ready", local(1), who="adv"),
        rec("job_20261105_predict_theirs", "Predict client-x tiles", "predict", "desktop", "ready", local(1),
            who="otherperson"),
    ]


def write_registry(root: Path, recs):
    (root / "job.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    J._reg_cache.clear()                                # the module caches registry reads for 10 s


def section(page: str, key: str) -> str:
    """The HTML inside <section id="cj-jobs-s-<key>">."""
    m = re.search(r'<section class="cj-jobs-sec" id="cj-jobs-s-%s"[^>]*>(.*?)</section>' % key, page, re.S)
    assert m, key
    return m.group(1)


def badge(frag: str) -> int:
    return int(re.search(r'<span class="cj-jobs-n">(\d+)</span>', frag).group(1))


def main_html(page: str) -> str:
    """The module's own HTML (between <main> tags), for word checks."""
    body = page.split('<main class="cj-main">', 1)[1].split("</main>", 1)[0]
    return re.sub(r"<style>.*?</style>", "", body, flags=re.S)


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

    def swap_live(self, doc):
        path = self.env.registry / "live" / "live.json"
        keep = path.read_bytes()
        if doc is None:
            path.unlink()
        else:
            self.env.write_live(doc)
        self.addCleanup(path.write_bytes, keep)

    def swap_registry(self, recs=None, text=None):
        path = self.env.registry / "job.jsonl"
        keep = path.read_bytes()
        if text is not None:
            path.write_text(text, encoding="utf-8")
            J._reg_cache.clear()
        else:
            write_registry(self.env.registry, recs)
        self.addCleanup(J._reg_cache.clear)
        self.addCleanup(path.write_bytes, keep)


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
                self.assertEqual(self.get("/jobs/jobs.js", role)[0], 200)
        for path in ("/jobs/", "/jobs", "/jobs/api/summary", "/jobs/jobs.js"):
            with self.subTest(path=path):
                st, _, body = self.get(path, "basic")
                self.assertEqual(st, 403)
                self.assertNotIn(b"job_2026", body)
        self.assertEqual(self.get("/", "basic")[0], 200)              # control: basic is signed in, just not here
        st, _, _ = T.request(self.p.port, "GET", "/jobs/api/summary")  # no session at all
        self.assertEqual(st, 401)

    def test_script_is_a_file(self):
        st, h, body = self.get("/jobs/jobs.js")
        self.assertEqual(st, 200)
        self.assertTrue(T.hget(h, "content-type").startswith("text/javascript"))
        self.assertIn(b"api/summary?html=1", body)
        page = self.page()
        self.assertIn('<script src="/jobs/jobs.js" defer></script>', page)
        self.assertEqual(re.findall(r"<script(?![^>]*\ssrc=)[^>]*>", page), [])   # no inline script left


class Sections(Base):
    def test_running_now(self):
        frag = section(self.page(), "running")
        self.assertEqual(badge(frag), 2)
        self.assertIn("Train v6 on lv_20261103", frag)                  # registry name for the desktop job
        self.assertIn("job_20261104_train_live", frag)
        self.assertIn("Epoch 12 / 300", frag)
        self.assertIn('<progress class="cj-jobs-bar" max="300" value="12"', frag)
        self.assertIn("2 h 3 min", frag)                                # time left: 7380 s
        self.assertIn("working normally · last update 40 s ago", frag)
        self.assertIn("<dd>40 s ago</dd>", frag)
        self.assertIn("0.250", frag)                                    # latest val mAP50-95
        self.assertIn("job_20261103_train_stuck", frag)
        self.assertIn("may be stuck", frag)
        self.assertIn("No progress for 2 h (normally a new epoch every 1 min).", frag)
        self.assertIn("watcher alive", frag)                            # admin: the technical line
        self.assertIn("gpu-desktop", frag)
        self.assertIn("lab-login", frag)
        self.assertNotIn("as of", self.page())

    def test_queue_in_ready_order(self):
        frag = section(self.page(), "queue")
        order = ["job_20261105_train_next", "job_20261105_predict_mine", "job_20261105_predict_theirs",
                 "job_20261105_train_after", "job_20261105_train_hold"]
        pos = [frag.find(j) for j in order]
        self.assertTrue(all(p >= 0 for p in pos), pos)
        self.assertEqual(pos, sorted(pos))                              # oldest READY first, then held
        self.assertEqual(re.findall(r'<span class="cj-jobs-pos">(\d)</span>', frag), ["1", "2", "3", "4"])
        self.assertIn("Cancel requested.", frag)
        self.assertIn("On hold on gpu-desktop: starts when someone releases it.", frag)
        self.assertNotIn("job_20261104_train_live", frag)               # running is not "waiting"
        self.assertNotIn("job_20261101_train_old", frag)                # done is not "waiting"
        self.assertEqual(badge(frag), 5)

    def test_queue_without_ready_at_has_bullets(self):
        q = queues_fixture()
        for j in q[0]["jobs"]:
            j.pop("ready_at", None)
        self.swap_live(T.live_doc(queues=q))
        frag = section(self.page(), "queue")
        self.assertEqual(re.findall(r'<span class="cj-jobs-pos">(\d)</span>', frag), [])
        self.assertEqual(frag.count("cj-jobs-t-off\" aria-hidden=\"true\">•</span>"), 5)
        order = ["job_20261105_predict_mine", "job_20261105_predict_theirs", "job_20261105_train_after",
                 "job_20261105_train_next", "job_20261105_train_hold"]
        pos = [frag.find(j) for j in order]
        self.assertEqual(pos, sorted(pos))                              # live.json order, not claimed as run order
        self.assertEqual([x["position"] for x in self.api()["queue"]], [None] * 5)

    def test_needs_attention(self):
        frag = section(self.page(), "attention")
        self.assertEqual(badge(frag), 4)
        for text in ("Can&#x27;t reach \\\\home-laptop\\cj", "May be stuck", "job_20261103_train_stuck",
                     "job_20261102_train_blocked", "Stopped with an error", "not been copied back",
                     "Annotate blind tiles", "tiles not uploaded yet"):
            self.assertIn(text, frag)
        self.assertIn(html.escape(REASON_XSS), frag)                   # admin: the raw error, escaped
        for old in ("job_20261030_train_oldblock", "Old blocked labeling"):
            self.assertNotIn(old, frag)                                 # older than 48 h and pulled / registry
        chips = re.findall(r'cj-jobs-chip cj-jobs-t-warn">([^<]+)</span>', frag)
        self.assertEqual(chips, ["not reachable", "may be stuck", "error", "blocked"])

    def test_other_open_jobs(self):
        frag = section(self.page(), "review")
        self.assertIn("Other open jobs", frag)
        names = ["Pre-lipo review (v2)", "Old blocked labeling", "Annotate blind tiles", html.escape(XSS),
                 "Train v7 (not pushed)"]
        pos = [frag.find(n) for n in names]
        self.assertTrue(all(p >= 0 for p in pos), pos)
        self.assertEqual(pos, sorted(pos))                              # in progress, blocked, ready
        self.assertIn("3 / 170 images reviewed", frag)
        self.assertIn("Being reviewed in the review app.", frag)
        self.assertIn("Not sent to the GPU computer yet.", frag)
        for on_queue in ("job_20261104_train_live", "Predict my tiles", "job_20261105_predict_theirs"):
            self.assertNotIn(on_queue, frag)                            # on a queue: shown from live.json
        self.assertNotIn("Check v5", frag)                              # done: under Recently finished
        self.assertEqual(badge(frag), 5)

    def test_recently_finished(self):
        frag = section(self.page(), "finished")
        order = ["job_20261101_train_old", "Old labeling", "Check v5 on 20261027", "job_20261102_train_blocked",
                 "job_20261030_train_oldblock"]
        pos = [frag.find(n) for n in order]
        self.assertTrue(all(p >= 0 for p in pos), pos)
        self.assertEqual(pos, sorted(pos))                              # newest first, both sources
        self.assertIn("mask mAP50-95 0.321", frag)
        self.assertIn("results not copied back yet", frag)
        self.assertRegex(frag, r"2 h( \d+ min)? ago")

    def test_api_matches_page(self):
        s = self.api()
        page = self.page()
        self.assertEqual(s["schema"], "cj_jobs/1")
        self.assertNotIn("html", s)
        self.assertEqual(s["counts"], {"running": 2, "queued": 4, "held": 1, "attention": 4, "review_open": 5,
                                       "finished": 5})
        r = {x["id"]: x for x in s["running"]}
        live = r["job_20261104_train_live"]
        self.assertEqual((live["kind"], live["done"], live["total"], live["unit"], live["pct"], live["hung"]),
                         ("train", 12, 300, "epochs", 4, False))
        self.assertEqual((live["eta_s"], live["watcher"]["age_s"], live["queue"]), (7380, 40, "gpu-desktop"))
        self.assertTrue(r["job_20261103_train_stuck"]["hung"])
        self.assertEqual([(x["id"], x["state"], x["position"]) for x in s["queue"]],
                         [("job_20261105_train_next", "queued", 1), ("job_20261105_predict_mine", "queued", 2),
                          ("job_20261105_predict_theirs", "queued", 3), ("job_20261105_train_after", "queued", 4),
                          ("job_20261105_train_hold", "held", None)])
        self.assertEqual([(x["kind"], x["source"]) for x in s["attention"]],
                         [("offline", "desktop"), ("hung", "desktop"), ("blocked", "desktop"), ("blocked", "registry")])
        self.assertEqual([x["status"] for x in s["review"]], ["in_progress", "blocked", "blocked", "ready", "ready"])
        self.assertEqual([x["id"] for x in s["finished"]],
                         ["job_20261101_train_old", "job_20261101_annotate_cancelled", "job_20261101_check_done",
                          "job_20261102_train_blocked", "job_20261030_train_oldblock"])
        self.assertEqual([q["reach"] for q in s["queues"]], ["ok", "ok", "offline"])
        self.assertEqual(s["queues"][0]["share"], "\\\\gpu-desktop\\cj")   # admin gets the share
        self.assertFalse(s["snapshot"]["stale"])
        self.assertIsNone(s["asof"])
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
        self.assertNotRegex(page, r"<[a-z]+[^>]*\son[a-z]+=")          # no inline event handlers anywhere


class Privacy(Base):
    def test_other_persons_prediction_job_hidden(self):
        adv_page, adv_api = self.page("advanced"), json.dumps(self.api("advanced"))
        for text in ("job_20261105_predict_theirs", "Predict client-x tiles", "otherperson"):
            self.assertNotIn(text, adv_page)
            self.assertNotIn(text, adv_api)
        self.assertIn(HIDDEN, section(adv_page, "queue"))
        q = self.api("advanced")["queue"]
        self.assertEqual([x["position"] for x in q if x["state"] == "queued"], [1, 2, 3, 4])   # place still counted
        admin_page = self.page("admin")                                 # control: admin sees it
        self.assertIn("job_20261105_predict_theirs", admin_page)
        self.assertIn("Predict client-x tiles", admin_page)

    def test_owner_is_viewer_sees_it(self):
        # registry who "adv" == the portal roster id "adv" (both are person ids)
        q = {x["id"]: x for x in self.api("advanced")["queue"] if x["id"]}
        self.assertEqual(q["job_20261105_predict_mine"]["name"], "Predict my tiles")
        self.assertIn("job_20261105_train_next", q)                     # registered lab train jobs: shown

    def test_unregistered_queue_job_hidden(self):
        q = queues_fixture()
        q[0]["jobs"].append(qjob("job_20261105_predict_unreg", "queued", ready_at=local(1)))
        q[0]["jobs"].append(qjob("job_20261105_train_unreg", "blocked", reason="secret path C:\\x"))
        self.swap_live(T.live_doc(queues=q))
        adv = json.dumps(self.api("advanced")) + self.page("advanced")
        for text in ("predict_unreg", "train_unreg", "secret path"):
            self.assertNotIn(text, adv)
        hidden = [x for x in self.api("advanced")["queue"] if x["private"]]
        self.assertEqual(len(hidden), 2)                                # someone else's + the unregistered one
        att = [x for x in self.api("advanced")["attention"] if x["private"]]
        self.assertEqual([(x["kind"], x["tech"]) for x in att], [("blocked", None)])
        admin = json.dumps(self.api("admin"))                           # control: admin sees both, with the reason
        for text in ("job_20261105_predict_unreg", "job_20261105_train_unreg", "secret path"):
            self.assertIn(text, admin)

    def test_registry_unreadable_fails_closed(self):
        self.swap_registry(text='{"schema": "job/1"}\n')                # a line without an id: current() raises
        s = self.api("advanced")
        self.assertFalse(s["registry"]["present"])
        self.assertEqual(s["registry"]["problem"], "not readable")      # no exception text for non-admins
        for key in ("running", "queue", "finished"):
            self.assertTrue(s[key])
            self.assertTrue(all(x["private"] and x["id"] is None for x in s[key]), key)
        self.assertTrue(all(x["private"] for x in s["attention"] if x["kind"] in ("hung", "blocked")))
        page = self.page("advanced")
        self.assertNotRegex(main_html(page), r"job_2026\d{4}_")
        self.assertIn("The lab job list can not be read right now.", page)
        admin = self.api("admin")                                       # control: admin still sees the ids
        self.assertIn("job_20261105_train_next", [x["id"] for x in admin["queue"]])
        self.assertIn("KeyError", admin["registry"]["problem"])

    def test_no_technical_words_for_non_admins(self):
        page, s = main_html(self.page("advanced")), self.api("advanced")
        for w in TECH_WORDS:
            self.assertNotIn(w, page, w)
        self.assertNotIn("cj-jobs-tech", page)
        self.assertTrue(all(q["share"] is None and q["detail"] is None for q in s["queues"]))
        self.assertIsNone(s["snapshot"]["monitor"])
        self.assertEqual(s["queues"][2]["name"], "GPU computer 3")      # the offline queue: no share path
        self.assertIn("Can&#x27;t reach GPU computer 3", page)
        admin = main_html(self.page("admin"))                           # control: admin gets the tech lines
        for w in ("watcher", "registry", "jobs/sync", "live.json", "\\\\home-laptop\\cj", "train.log"):
            self.assertIn(w, admin, w)


class Snapshots(Base):
    def test_stale_banner(self):
        self.swap_live(T.live_doc(age_s=3600, queues=queues_fixture()))
        page = self.page()
        self.assertIn("Status updates have stopped.", page)
        self.assertIn("The last one came 1 h ago", page)
        self.assertIn("Running (as of 1 h ago)", section(page, "running"))
        self.assertIn("working normally · last update 40 s ago (as of 1 h ago)", section(page, "running"))
        self.assertEqual(section(page, "queue").count("(as of 1 h ago)"), 5)
        self.assertIn("as of 1 h ago", section(page, "attention"))
        self.assertNotIn("Nothing is running", page)
        s = self.api()
        self.assertTrue(s["snapshot"]["stale"])
        self.assertEqual(s["asof"], "as of 1 h ago")
        self.assertTrue(all(x["stale"] for x in s["running"] + s["queue"]))

    def test_fresh_has_no_banner(self):                                 # control for the banner test
        self.swap_live(T.live_doc(age_s=60, queues=queues_fixture()))   # 60 s < 3 x 30 s
        page = self.page()
        self.assertNotIn("Status updates have stopped.", page)
        self.assertNotIn("as of", page)
        self.assertIn("Updated 1 min ago.", page)

    def test_missing_snapshot(self):
        self.swap_live(None)
        page = self.page()
        self.assertIn("No status from the GPU computer.", page)
        self.assertIn("Unknown: no status from the GPU computer.", section(page, "running"))
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
        self.assertIn(HIDDEN, frag)
        self.assertIn("Images 40 / 120", frag)
        self.assertNotIn("predict_theirs", frag)

    def test_watcher_stale_and_paused(self):
        q = queues_fixture()[:1]
        q[0]["watcher"].update(display="stale", age_s=900, stale=True)
        self.swap_live(T.live_doc(queues=q))
        att = self.api()["attention"]
        self.assertEqual(att[0]["kind"], "watcher")
        self.assertEqual(att[0]["title"], "gpu-desktop: status updates have stopped")
        self.assertEqual(att[0]["reason"], "Last update 15 min ago. Waiting jobs will not start until it is back.")
        q[0]["watcher"].update(display="paused", age_s=30, stale=False, detail="STOP: GPU maintenance")
        self.env.write_live(T.live_doc(queues=q))
        att = self.api()["attention"]
        self.assertEqual((att[0]["kind"], att[0]["reason"]),
                         ("paused", "4 waiting job(s) will not start until it is resumed."))
        self.assertIn("STOP: GPU maintenance", att[0]["tech"])
        self.assertIsNone(self.api("advanced")["attention"][0]["tech"])


class RegistryCache(unittest.TestCase):
    def test_one_read_per_ttl(self):
        root = Path(tempfile.mkdtemp(prefix="cj_jobs_reg_"))
        try:
            write_registry(root, registry_fixture()[:1])
            jobs, err, _ = J.registry_jobs(root, now=1000.0)
            self.assertIsNone(err)
            self.assertEqual(list(jobs), ["job_20261102_prelipo_review_v2"])
            lines = "".join(json.dumps(r) + "\n" for r in registry_fixture()[:2])
            (root / "job.jsonl").write_text(lines, encoding="utf-8")   # no cache clear: that is what is tested
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
                        self.assertEqual(page.locator("#cj-jobs-s-review article").count(), 5)
                        self.assertEqual(page.locator("#cj-jobs-s-finished li").count(), 5)
                        self.assertEqual(page.locator("body img").count(), 0)          # the XSS name stayed text
                        bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
                        self.assertEqual(bg, "rgb(15, 21, 27)" if scheme == "dark" else "rgb(243, 245, 247)")   # cj.css design A --bg
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
            self.assertIn("Status updates have stopped", page.text_content(".cj-jobs-banner"))
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
            self.assertIn(HIDDEN, page.text_content("#cj-jobs-s-queue"))
            sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
            self.assertLessEqual(sw, iw)
            self.check_clean(rec)
        finally:
            ctx.close()


class ActionBase(unittest.TestCase):
    """Cancel / release against a TEMP share folder laid out like a desktop queue (q/<job>/job.json [+ READY]).
    live.json's queues[].share points at it; the real share is never touched."""

    JOBS = {"job_20261105_admin_q": ("queued", "tester"), "job_20261105_adv_q": ("queued", "adv"),
            "job_20261105_other_q": ("queued", "adv2"), "job_20261105_hold": ("held", "adv"),
            "job_20261105_run": ("running", "adv"), "job_20261105_orphan": ("queued", None)}

    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.adv_key = A.add_key(cls.env.roster, "adv", "advanced")
        cls.adv2_key = A.add_key(cls.env.roster, "adv2", "advanced")
        cls.bas_key = A.add_key(cls.env.roster, "bas", "basic")
        cls.share = cls.env.tmp / "share"
        cls.decoy = cls.env.tmp / "decoy"
        for root in (cls.share, cls.decoy):
            for job in cls.JOBS:
                (root / "q" / job).mkdir(parents=True)
                (root / "q" / job / "job.json").write_text("{}", encoding="utf-8")
        for job, (state, _) in cls.JOBS.items():
            if state != "held":
                (cls.share / "q" / job / "READY").write_text("x", encoding="utf-8")
        recs = [{"schema": "job/1", "id": j, "logged": local(1), "name": j, "type": "train", "where": "desktop",
                 "status": "ready", **({"who": who} if who else {})} for j, (_, who) in cls.JOBS.items()]
        write_registry(cls.env.registry, recs)
        cls.write_live()
        cls.p = T.Portal(cls.env.config())
        cls.tok = {"admin": T.login(cls.p.port, cls.env.key), "advanced": T.login(cls.p.port, cls.adv_key),
                   "adv2": T.login(cls.p.port, cls.adv2_key), "basic": T.login(cls.p.port, cls.bas_key)}

    @classmethod
    def live_queues(cls, reach="ok"):
        jobs = [qjob(j, st) for j, (st, _) in cls.JOBS.items()]
        return [{"share": str(cls.share), "reach": reach, "detail": None,
                 "watcher": {"display": "alive", "state": "running", "detail": None, "job": "job_20261105_run",
                             "age_s": 5, "stale": False, "machine": "gpu-desktop"},
                 "jobs": jobs, "running": {"job": "job_20261105_run", "phase": "training", "epoch": 3, "epochs": 100,
                                           "eta_s": 600, "elapsed_s": 60, "cancel_requested": False},
                 "finished": []}]

    @classmethod
    def write_live(cls, age_s=5, reach="ok"):
        cls.env.write_live(T.live_doc(age_s=age_s, queues=cls.live_queues(reach)))

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()

    def setUp(self):
        for root in (self.share, self.decoy):
            for job in self.JOBS:
                (root / "q" / job / "CANCEL").unlink(missing_ok=True)
        for job, (st, _) in self.JOBS.items():
            r = self.share / "q" / job / "READY"
            if st == "held":
                r.unlink(missing_ok=True)
            else:
                r.write_text("x", encoding="utf-8")
        self.audit_file = self.env.state / "audit" / "portal.jsonl"
        self.audit_file.unlink(missing_ok=True)
        self.write_live()

    def post(self, path, body, role="admin", origin="same", raw=None):
        h = {**T.cookie(self.tok[role]), "Content-Type": "application/json"}
        if origin == "same":
            h["Origin"] = self.p.origin
        elif origin:
            h["Origin"] = origin
        data = raw if raw is not None else json.dumps(body).encode()
        st, hd, out = T.request(self.p.port, "POST", path, h, data)
        return st, (json.loads(out) if out[:1] == b"{" else out), hd

    def cancel_file(self, job, root=None):
        return (root or self.share) / "q" / job / "CANCEL"

    def audit_lines(self):
        if not self.audit_file.exists():
            return []
        return [json.loads(x) for x in self.audit_file.read_text(encoding="utf-8").splitlines()]


class Actions(ActionBase):
    def test_admin_cancels_queued_job(self):
        st, out, hd = self.post("/jobs/api/cancel", {"job": "job_20261105_admin_q", "reason": "wrong dataset"})
        self.assertEqual((st, out["ok"]), (200, True), out)
        self.assertEqual(T.hget(hd, "cache-control"), "no-store")
        self.assertIn("will not be started", out["message"])
        doc = json.loads(self.cancel_file("job_20261105_admin_q").read_text(encoding="utf-8"))
        self.assertEqual((doc["job"], doc["who"], doc["reason"]), ("job_20261105_admin_q", "tester", "wrong dataset"))
        self.assertFalse(self.cancel_file("job_20261105_admin_q", self.decoy).exists())

    def test_cancel_running_says_ten_seconds(self):
        st, out, _ = self.post("/jobs/api/cancel", {"job": "job_20261105_run", "reason": "stuck"})
        self.assertEqual(st, 200)
        self.assertIn("within about 10 s", out["message"])
        self.assertTrue(self.cancel_file("job_20261105_run").is_file())

    def test_owner_cancels_own_job(self):
        st, out, _ = self.post("/jobs/api/cancel", {"job": "job_20261105_adv_q", "reason": "my mistake"}, "advanced")
        self.assertEqual((st, out["ok"]), (200, True), out)
        doc = json.loads(self.cancel_file("job_20261105_adv_q").read_text(encoding="utf-8"))
        self.assertEqual((doc["who"], doc["reason"]), ("adv", "my mistake"))

    def test_non_owner_advanced_gets_403_and_no_file(self):
        for job in ("job_20261105_adv_q", "job_20261105_admin_q", "job_20261105_orphan"):   # last: owner unknown
            with self.subTest(job=job):
                st, out, _ = self.post("/jobs/api/cancel", {"job": job, "reason": "x"}, "adv2")
                self.assertEqual(st, 403)
                self.assertFalse(out["ok"])
                self.assertFalse(self.cancel_file(job).exists())
        st, _, _ = self.post("/jobs/api/cancel", {"job": "job_20261105_orphan", "reason": "x"}, "advanced")
        self.assertEqual(st, 403)                                   # an advanced user is not the owner of an unowned job
        self.assertEqual(self.audit_lines(), [])                    # a refused request is not an action

    def test_basic_role_403_at_the_gate(self):
        for path, body in (("/jobs/api/cancel", {"job": "job_20261105_adv_q", "reason": "x"}),
                           ("/jobs/api/release", {"job": "job_20261105_hold"})):
            st, _, _ = self.post(path, body, "basic")
            self.assertEqual(st, 403)
        self.assertFalse(self.cancel_file("job_20261105_adv_q").exists())
        self.assertFalse((self.share / "q" / "job_20261105_hold" / "READY").exists())

    def test_missing_reason_refused(self):
        for body in ({"job": "job_20261105_admin_q"}, {"job": "job_20261105_admin_q", "reason": ""},
                     {"job": "job_20261105_admin_q", "reason": "  \n "}, {"job": "job_20261105_admin_q", "reason": 5},
                     {"job": "job_20261105_admin_q", "reason": "x" * 400}):
            with self.subTest(body=body):
                st, out, _ = self.post("/jobs/api/cancel", body)
                self.assertEqual(st, 400)
                self.assertFalse(out["ok"])
        self.assertFalse(self.cancel_file("job_20261105_admin_q").exists())
        st, _, _ = self.post("/jobs/api/cancel", None, raw=b"not json")
        self.assertEqual(st, 400)

    def test_control_characters_in_reason_are_flattened(self):
        st, _, _ = self.post("/jobs/api/cancel", {"job": "job_20261105_admin_q", "reason": "a\r\nb\x00c"})
        self.assertEqual(st, 200)
        doc = json.loads(self.cancel_file("job_20261105_admin_q").read_text(encoding="utf-8"))
        self.assertEqual(doc["reason"], "a  b c")

    def test_stale_live_json_refused(self):
        self.write_live(age_s=3600)
        for path, body in (("/jobs/api/cancel", {"job": "job_20261105_admin_q", "reason": "x"}),
                           ("/jobs/api/release", {"job": "job_20261105_hold"})):
            st, out, _ = self.post(path, body)
            self.assertEqual(st, 409)
            self.assertIn("out of date", out["message"])
        self.assertFalse(self.cancel_file("job_20261105_admin_q").exists())
        self.assertFalse((self.share / "q" / "job_20261105_hold" / "READY").exists())
        (self.env.registry / "live" / "live.json").unlink()          # no snapshot at all
        st, _, _ = self.post("/jobs/api/cancel", {"job": "job_20261105_admin_q", "reason": "x"})
        self.assertEqual(st, 409)

    def test_unreachable_queue_refused(self):
        self.write_live(reach="offline")
        st, out, _ = self.post("/jobs/api/cancel", {"job": "job_20261105_admin_q", "reason": "x"})
        self.assertEqual(st, 409)
        self.assertIn("can't reach", out["message"])
        self.assertFalse(self.cancel_file("job_20261105_admin_q").exists())

    def test_unknown_job_and_bad_ids(self):
        st, _, _ = self.post("/jobs/api/cancel", {"job": "job_20261105_nope", "reason": "x"})
        self.assertEqual(st, 404)
        for bad in ("../../etc", "a/b", "..", "", None, 7):
            with self.subTest(job=bad):
                st, _, _ = self.post("/jobs/api/cancel", {"job": bad, "reason": "x"})
                self.assertEqual(st, 400)

    def test_client_supplied_share_is_ignored(self):
        body = {"job": "job_20261105_admin_q", "reason": "x", "share": str(self.decoy), "path": str(self.decoy),
                "queue": str(self.decoy)}
        st, _, _ = self.post("/jobs/api/cancel", body)
        self.assertEqual(st, 200)
        self.assertTrue(self.cancel_file("job_20261105_admin_q").is_file())
        self.assertFalse(self.cancel_file("job_20261105_admin_q", self.decoy).exists())
        st, _, _ = self.post("/jobs/api/release", {"job": "job_20261105_hold", "share": str(self.decoy)})
        self.assertEqual(st, 200)
        self.assertTrue((self.share / "q" / "job_20261105_hold" / "READY").is_file())
        self.assertFalse((self.decoy / "q" / "job_20261105_hold" / "READY").exists())

    def test_release_admin_only(self):
        for role in ("advanced", "adv2"):                            # even the held job's own owner
            st, _, _ = self.post("/jobs/api/release", {"job": "job_20261105_hold"}, role)
            self.assertEqual(st, 403)
        self.assertFalse((self.share / "q" / "job_20261105_hold" / "READY").exists())
        st, out, _ = self.post("/jobs/api/release", {"job": "job_20261105_hold"})
        self.assertEqual((st, out["ok"]), (200, True), out)
        self.assertTrue((self.share / "q" / "job_20261105_hold" / "READY").is_file())
        st, _, _ = self.post("/jobs/api/release", {"job": "job_20261105_admin_q"})     # not held
        self.assertEqual(st, 409)

    def test_already_requested_or_finished_is_refused_plainly(self):
        q = self.live_queues()
        q[0]["jobs"][0]["cancel_requested"] = True
        self.env.write_live(T.live_doc(queues=q))
        st, out, _ = self.post("/jobs/api/cancel", {"job": "job_20261105_admin_q", "reason": "x"})
        self.assertEqual(st, 409)
        self.write_live()
        done = self.share / "d" / "job_20261105_admin_q"             # finished between snapshots: sync.cancel refuses
        done.mkdir(parents=True)
        (done / "DONE.json").write_text("{}", encoding="utf-8")
        self.addCleanup(shutil.rmtree, done, True)
        st, out, _ = self.post("/jobs/api/cancel", {"job": "job_20261105_admin_q", "reason": "x"})
        self.assertEqual(st, 502)
        self.assertFalse(self.cancel_file("job_20261105_admin_q").exists())
        self.assertEqual(self.audit_lines()[-1]["result"], "error: SyncError")

    def test_audit_lines(self):
        self.post("/jobs/api/cancel", {"job": "job_20261105_adv_q", "reason": "because"}, "advanced")
        self.post("/jobs/api/release", {"job": "job_20261105_hold"})
        self.post("/jobs/api/cancel", {"job": "job_20261105_admin_q", "reason": "x"}, "adv2")   # refused: no line
        lines = self.audit_lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual([(x["actor"], x["action"], x["job"], x["reason"]) for x in lines],
                         [("adv", "cancel", "job_20261105_adv_q", "because"),
                          ("tester", "release", "job_20261105_hold", None)])
        for x in lines:
            self.assertTrue({"time", "actor", "action", "job", "reason"} <= set(x))
            self.assertEqual(x["result"], "ok")
            self.assertRegex(x["time"], r"^\d{4}-\d\d-\d\dT")
        text = self.audit_file.read_text(encoding="utf-8")
        for secret in (self.env.key, self.adv_key, self.tok["admin"], self.tok["advanced"]):
            self.assertNotIn(secret, text)

    def test_cross_origin_post_403(self):
        for origin in ("http://evil.example", "null", None):
            with self.subTest(origin=origin):
                st, _, _ = self.post("/jobs/api/cancel", {"job": "job_20261105_admin_q", "reason": "x"},
                                     origin=origin)
                self.assertEqual(st, 403)
                self.assertFalse(self.cancel_file("job_20261105_admin_q").exists())
        st, _, _ = self.post("/jobs/api/release", {"job": "job_20261105_hold"}, origin="http://evil.example")
        self.assertEqual(st, 403)
        self.assertEqual(self.audit_lines(), [])

    def test_get_not_allowed(self):
        st, _, _ = T.request(self.p.port, "GET", "/jobs/api/cancel", T.cookie(self.tok["admin"]))
        self.assertEqual(st, 405)

    def test_audit_unwritable_means_nothing_done(self):
        bad = self.env.state / "audit"
        shutil.rmtree(bad, ignore_errors=True)
        bad.write_text("a file where the folder should be", encoding="utf-8")
        self.addCleanup(lambda: (bad.unlink(missing_ok=True)))
        st, out, _ = self.post("/jobs/api/cancel", {"job": "job_20261105_admin_q", "reason": "x"})
        self.assertEqual(st, 500)
        self.assertFalse(self.cancel_file("job_20261105_admin_q").exists())


class Buttons(ActionBase):
    def jobs_with(self, page, act):
        return set(re.findall(r'data-act="%s" data-job="([^"]+)"' % act, page))

    def page(self, role):
        st, _, body = T.request(self.p.port, "GET", "/jobs/", T.cookie(self.tok[role]))
        self.assertEqual(st, 200)
        return body.decode("utf-8")

    def test_admin_sees_cancel_on_every_open_job_and_release_on_held(self):
        pg = self.page("admin")
        self.assertEqual(self.jobs_with(pg, "ask"), set(self.JOBS))
        self.assertEqual(self.jobs_with(pg, "release"), {"job_20261105_hold"})
        self.assertIn("Yes, cancel", pg)
        self.assertIn("Cancel this job?", pg)
        self.assertIn(" hidden>", pg)                                # the confirm form starts hidden

    def test_owner_sees_only_own_cancel_buttons(self):
        pg = self.page("advanced")
        self.assertEqual(self.jobs_with(pg, "ask"), {"job_20261105_adv_q", "job_20261105_hold", "job_20261105_run"})
        self.assertEqual(self.jobs_with(pg, "release"), set())
        pg2 = self.page("adv2")
        self.assertEqual(self.jobs_with(pg2, "ask"), {"job_20261105_other_q"})

    def test_no_buttons_on_stale_snapshot_or_cancel_in_flight(self):
        self.write_live(age_s=3600)
        self.assertNotIn("data-act=", self.page("admin"))
        q = self.live_queues()
        q[0]["jobs"][0]["cancel_requested"] = True
        self.env.write_live(T.live_doc(queues=q))
        self.assertNotIn('data-act="ask" data-job="job_20261105_admin_q"', self.page("admin"))
        self.assertIn('data-act="ask" data-job="job_20261105_adv_q"', self.page("admin"))

    def test_api_flags_match(self):
        st, _, body = T.request(self.p.port, "GET", "/jobs/api/summary", T.cookie(self.tok["advanced"]))
        s = json.loads(body)
        self.assertEqual({x["id"] for x in s["queue"] if x["can_cancel"]}, {"job_20261105_adv_q", "job_20261105_hold"})
        self.assertFalse(any(x["can_release"] for x in s["queue"]))


@unittest.skipIf(sync_playwright is None, "playwright not installed")
class CancelFlowBrowser(ActionBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.pw = sync_playwright().start()
        cls.browser = launch(cls.pw)
        if cls.browser is None:
            cls.pw.stop()
            super().tearDownClass()
            raise unittest.SkipTest("no Edge / Chrome for Playwright")

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        super().tearDownClass()

    def test_two_step_cancel_at_390px_without_dialogs(self):
        ctx = self.browser.new_context(viewport={"width": 390, "height": 900})
        dialogs, errors = [], []
        page = ctx.new_page()
        page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            page.goto(f"{self.p.origin}/login?k={self.adv_key}")
            page.goto(self.p.origin + "/jobs/")
            page.wait_for_load_state("networkidle")
            job = "job_20261105_adv_q"
            act = page.locator(f'.cj-jobs-act:has(button[data-act="ask"][data-job="{job}"])')
            form = act.locator("form.cj-jobs-cf")
            self.assertTrue(form.is_hidden())
            act.locator('button[data-act="ask"]').click()
            self.assertTrue(form.is_visible())
            self.assertIn("Cancel this job?", form.inner_text())
            sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
            self.assertLessEqual(sw, iw, "horizontal scroll")
            form.locator("button[data-act=back]").click()              # step back: nothing sent
            self.assertTrue(form.is_hidden())
            self.assertFalse(self.cancel_file(job).exists())
            act.locator('button[data-act="ask"]').click()
            form.locator("button[type=submit]").click()                # no reason: the browser blocks the submit
            self.assertTrue(form.is_visible())
            self.assertFalse(self.cancel_file(job).exists())
            form.locator("input[name=reason]").fill("typed in the browser")
            with page.expect_response(lambda r: r.url.endswith("/jobs/api/cancel")) as resp:
                form.locator("button[type=submit]").click()
            self.assertEqual(resp.value.status, 200)
            res = act.locator(".cj-jobs-res")
            res.wait_for(state="visible")
            self.assertIn("Cancel requested", res.inner_text())
            self.assertTrue(form.is_hidden())
            doc = json.loads(self.cancel_file(job).read_text(encoding="utf-8"))
            self.assertEqual((doc["who"], doc["reason"]), ("adv", "typed in the browser"))
            self.assertEqual(dialogs, [])
            self.assertEqual(errors, [])
        finally:
            ctx.close()


if __name__ == "__main__":
    unittest.main()
