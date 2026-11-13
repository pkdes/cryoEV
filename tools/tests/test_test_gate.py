"""tools/test_gate.py: which suites a change runs (known answers), browser detection, env strip, the lock, timings."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_gate as G  # noqa: E402


class Affected(unittest.TestCase):
    def test_known_answers(self):
        self.assertEqual(G.affected_suites(["portal/modules/gallery.py"])[0],
                         {"tests", "portal", "cryoev", "annotation"})
        self.assertEqual(G.affected_suites(["docs/TOOLS.md", "README.md"])[0], {"tests"})
        self.assertEqual(G.affected_suites(["results/compute.py"])[0], {"tests", "results", "portal", "jobs"})
        self.assertEqual(G.affected_suites(["jobs/convert_raw.py"])[0], {"tests", "jobs", "portal", "tools", "cryoev"})
        # registry/ also runs data_utils (dataset_from_lv appends through the registry; QC r9 #13), the CLI, and the
        # review tool and tools that read it (round 10)
        self.assertEqual(G.affected_suites(["registry/registry.py"])[0],
                         {"tests", "registry", "portal", "jobs", "data_utils", "cryoev", "annotation", "tools"})
        self.assertEqual(G.affected_suites(["cryoev/common.py"])[0], {"tests", "cryoev"})
        self.assertEqual(G.affected_suites(["models/models.yaml"])[0],
                         {"tests", "jobs", "portal", "registry", "cryoev"})
        self.assertEqual(G.affected_suites(["training/train_yolo.py"])[0],
                         {"tests", "jobs", "portal", "annotation", "results"})
        self.assertEqual(G.affected_suites(["inference/predict_models.py"])[0],
                         {"tests", "jobs", "portal", "annotation", "results"})
        self.assertEqual(G.affected_suites(["tools/services.py"])[0], {"tests", "tools", "cryoev", "portal"})
        self.assertEqual(G.affected_suites(["client/cryoev_client.py"])[0], {"tests", "portal"})

    def test_shared_code_and_unknown_files_run_everything(self):
        self.assertEqual(G.affected_suites(["data_utils/oslock.py"])[0], set(G.SUITES))
        suites, unknown = G.affected_suites(["setup.cfg"])
        self.assertEqual(suites, set(G.SUITES))
        self.assertEqual(unknown, ["setup.cfg"])

    def test_changelog_and_requirements_are_not_doc_only(self):
        for f in ("CHANGELOG.md", "requirements.txt", "requirements-dev.txt"):
            suites, unknown = G.affected_suites([f])
            self.assertEqual(suites, set(G.SUITES), f)
            self.assertEqual(unknown, [f])
        self.assertEqual(G.affected_suites(["docs/CHANGELOG_NOTES.md"])[0], {"tests"})   # still plain docs

    def test_every_suite_dir_exists(self):
        for name, d in G.SUITES.items():
            self.assertTrue((G.REPO / d).is_dir(), name)


class BrowserDetection(unittest.TestCase):
    def check(self, text):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "test_x.py"
            p.write_text(text, encoding="utf-8")
            return G.is_browser_file(p)

    def test_imports_decide(self):
        self.assertTrue(self.check("from playwright.sync_api import sync_playwright\n"))
        self.assertTrue(self.check("import test_review_tool_browser as trb\n"))
        self.assertTrue(self.check("def f():\n    from test_review_tool_browser import launch\n"))
        self.assertFalse(self.check("# a playwright browser is not used here\nimport unittest\n"))
        self.assertFalse(self.check("import test_review_server\n"))

    def test_the_real_browser_files(self):
        ann = G.REPO / "annotation" / "tests"
        for n in ("test_prediction_package_browser.py", "test_review_audit_fixes_browser.py",
                  "test_review_reject_lowconf_browser.py", "test_server_time_and_scope.py",
                  "test_review_tool_browser.py"):
            self.assertTrue(G.is_browser_file(ann / n), n)
        self.assertFalse(G.is_browser_file(ann / "test_review_server.py"))


class ChildEnv(unittest.TestCase):
    def test_actor_vars_are_stripped_or_set_on_purpose(self):
        base = {"CRYOEV_VIA": "claude", "CRYOEV_FOR": "someone", "KEEP": "1"}
        e = G.child_env(False, base)
        self.assertNotIn("CRYOEV_VIA", e)
        self.assertNotIn("CRYOEV_FOR", e)
        self.assertEqual(e["KEEP"], "1")
        e = G.child_env(True, {"KEEP": "1"})
        self.assertEqual((e["CRYOEV_VIA"], e["CRYOEV_FOR"]), ("claude", "pujan"))

    def test_a_real_child_sees_the_stripped_env(self):
        out = subprocess.run([sys.executable, "-c", "import os;print(os.environ.get('CRYOEV_VIA'),"
                              "os.environ.get('CRYOEV_FOR'))"], capture_output=True, text=True,
                             env=G.child_env(False, {**os.environ, "CRYOEV_VIA": "x", "CRYOEV_FOR": "y"}))
        self.assertEqual(out.stdout.split(), ["None", "None"])


class Lock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "lock"

    def tearDown(self):
        self.tmp.cleanup()

    def test_second_gate_waits_then_fails_busy(self):
        a = G.GateLock(self.path, wait_min=1, poll_s=0.05)
        self.assertTrue(a.acquire(log=lambda m: None))
        b = G.GateLock(self.path, wait_min=0.01, poll_s=0.05)            # the cap: 0.6 s
        t0 = time.time()
        self.assertFalse(b.acquire(log=lambda m: None))
        self.assertGreaterEqual(time.time() - t0, 0.5)
        self.assertLess(time.time() - t0, 10)
        a.release()
        self.assertFalse(self.path.exists())
        self.assertTrue(b.acquire(log=lambda m: None))                   # free again
        b.release()

    def test_waiter_gets_it_when_the_holder_releases(self):
        a = G.GateLock(self.path, wait_min=1, poll_s=0.05)
        a.acquire(log=lambda m: None)
        threading.Timer(0.4, a.release).start()
        b = G.GateLock(self.path, wait_min=1, poll_s=0.05)
        self.assertTrue(b.acquire(log=lambda m: None))
        b.release()

    def test_two_concurrent_takeovers_of_a_stale_lock_have_one_winner(self):
        for _ in range(10):
            self.path.mkdir()
            (self.path / "owner.json").write_text(json.dumps({"pid": 2 ** 22 + 12345}), encoding="utf-8")
            locks = [G.GateLock(self.path, wait_min=0.01) for _ in range(2)]
            barrier, got = threading.Barrier(2), []

            def go(lk):
                barrier.wait()
                got.append(lk.try_acquire())
            ts = [threading.Thread(target=go, args=(lk,)) for lk in locks]
            [t.start() for t in ts]
            [t.join() for t in ts]
            self.assertEqual(sorted(got), [False, True], got)
            owner = locks[got.index(True)]
            self.assertEqual(owner.owner()["pid"], os.getpid())
            owner.release()
            self.assertEqual([p.name for p in self.path.parent.iterdir()], [])     # no stale.* leftovers

    def test_a_dead_owner_is_stale(self):
        self.path.mkdir()
        (self.path / "owner.json").write_text(json.dumps({"pid": 2 ** 22 + 12345}), encoding="utf-8")
        g = G.GateLock(self.path, wait_min=0.01, poll_s=0.05)
        self.assertTrue(g.is_stale())
        self.assertTrue(g.acquire(log=lambda m: None))
        self.assertEqual(g.owner()["pid"], os.getpid())
        g.release()

    def test_an_old_heartbeat_is_stale_and_a_live_one_is_not(self):
        self.path.mkdir()
        (self.path / "owner.json").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
        hb = self.path / "heartbeat"
        hb.write_text("x", encoding="utf-8")
        g = G.GateLock(self.path, wait_min=0.01, stale_s=60)
        self.assertFalse(g.is_stale())                                   # owner alive, heartbeat fresh
        old = time.time() - 3600
        os.utime(hb, (old, old))
        self.assertTrue(g.is_stale())


class Timings(unittest.TestCase):
    def test_past_times_is_the_median_and_skips_failures(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "timings.jsonl"
            rows = [("u/a", 10, True), ("u/a", 12, True), ("u/a", 300, True), ("u/a", 999, False), ("u/b", 5, True)]
            p.write_text("\n".join(json.dumps({"unit": u, "seconds": s, "ok": ok}) for u, s, ok in rows) + "\nbad\n",
                         encoding="utf-8")
            self.assertEqual(G.past_times(p), {"u/a": 12, "u/b": 5})

    def test_record_keeps_retries_and_the_failure_tail(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "timings.jsonl"
            r = {"unit": "u/a", "seconds": 3.0, "tests": 2, "ok": True, "flaky": True, "retries": 1,
                 "retry_seconds": 2.0, "first_fail_tail": "AssertionError: boom", "output": ""}
            bad = {"unit": "u/b", "seconds": 1.0, "tests": 1, "ok": False, "output": "x" * 2000}
            G.record([r, bad], "full", True, path=p)
            rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
            self.assertEqual((rows[0]["retries"], rows[0]["retry_seconds"], rows[0]["fail_tail"]),
                             (1, 2.0, "AssertionError: boom"))
            self.assertTrue(rows[0]["agent_env"])
            self.assertFalse(rows[1]["ok"])
            self.assertEqual(len(rows[1]["fail_tail"]), 600)


class Run(unittest.TestCase):
    def test_a_real_unit_passes_and_counts_its_tests(self):
        r = G.run_unit("results", "test_p2.py", timeout=300)
        self.assertTrue(r["ok"], r["output"][-500:])
        self.assertGreater(r["tests"], 0)


if __name__ == "__main__":
    unittest.main()
