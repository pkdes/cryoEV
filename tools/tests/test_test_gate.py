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


class BrowserSkip(unittest.TestCase):
    """Round 11 #6 (QC pass 1): --affected skips the @browser CLASSES (CJ_GATE_SKIP_BROWSER=1) only when every changed
    file is non-UI; no file is skipped."""

    def test_known_answers(self):
        for files in (["registry/store.py"], ["jobs/sync.py", "docs/TOOLS.md"], ["training/x.py", "inference/y.py"],
                      ["data_utils/oslock.py"], ["tools/services.py"], ["cryoev/common.py"],
                      ["registry/importers.py", ".claude/settings.json"], ["registry\\store.py"]):
            self.assertTrue(G.ui_untouched(files), files)
        for files in (["registry/history_page.py"], ["tools/test_gate.py"], ["tools/tests/test_test_gate.py"],
                      ["registry/store.py", "portal/modules/jobs.py"], ["annotation/review_tool/index.html"],
                      ["proxy.py"], ["README.md"], ["results/compute.py"], [], [".claude/x"]):
            self.assertFalse(G.ui_untouched(files), files)

    def test_backslashes_normalised(self):
        self.assertEqual(G.affected_suites(["registry\\store.py"]), G.affected_suites(["registry/store.py"]))
        self.assertEqual(G.affected_suites(["portal\\app.py"])[1], [])

    def test_unknown_files_never_skip(self):
        """docs/x.py is under docs/ (non-UI) but matches no suite rule: unknown -> everything runs, nothing skips."""
        files = ["docs/x.py"]
        self.assertTrue(G.ui_untouched(files))
        self.assertEqual(G.affected_suites(files)[1], files)
        self.assertFalse(G.skip_mode("affected", files))
        _, skip, _ = G.select(G.affected_suites(files)[0], "affected", files)
        self.assertEqual(skip, set())

    def test_control_a_predict_parts_change_skips_nothing(self):
        files = ["portal/predict_parts/uploads.py"]
        units, skip, _ = G.select(G.affected_suites(files)[0], "affected", files)
        self.assertEqual(skip, set())
        self.assertTrue(any(b for _, _, b in units))

    def test_control_unit_classes_still_run(self):
        """--files jobs/predict_set.py: test_predict_module runs (skip env set), and its unit classes are not @browser
        while its Browser class is."""
        files = ["jobs/predict_set.py"]
        units, skip, _ = G.select(G.affected_suites(files)[0], "affected", files)
        self.assertIn(("portal", "test_predict_module.py", False), units)
        self.assertIn("portal/test_predict_module.py", skip)
        self.assertEqual(G.browser_marks(G.REPO / "portal" / "tests" / "test_predict_module.py"), (1, 0))
        env = G.child_env(skip_browser=True, base={})
        r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "portal/tests", "-p",
                            "test_predict_module.py", "-k", "StatusRules", "-k", "Browser", "-v"], cwd=G.REPO,
                           env={**os.environ, **env}, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertRegex(r.stderr, r"StatusRules\.\w+\)[^\n]*\n?[^\n]*\.\.\. ok")       # a unit class ran
        self.assertIn("skipped 'browser test: no UI file changed (gate)'", r.stderr)

    def test_every_file_runs_and_no_lane_in_skip_mode(self):
        files = ["registry/store.py"]
        units, skip, _ = G.select(set(G.SUITES), "affected", files)
        self.assertEqual(len(units), len(G.units_for(set(G.SUITES))))     # no file dropped
        self.assertFalse(any(b for _, _, b in units))                       # no browser runs: no lane needed
        n = sum(G.browser_marks(G.REPO / G.SUITES[s] / f)[0] for s, f, _ in G.units_for(set(G.SUITES)))
        self.assertGreaterEqual(n, 30)
        self.assertIn(f"{n} browser classes skipped (no UI change)", G.skip_note(units, skip))

    def test_every_browser_file_marks_its_browser_classes(self):
        for s, n, b in G.units_for(set(G.SUITES)):
            if b:
                self.assertNotEqual(G.browser_marks(G.REPO / G.SUITES[s] / n), (0, 0), f"{s}/{n} has no @browser")

    def test_a_changed_browser_test_file_runs_in_full(self):
        files = ["registry/tests/test_jobs.py"]
        units, skip, _ = G.select({"registry"}, "affected", files)
        self.assertIn(("registry", "test_jobs.py", True), units)
        self.assertNotIn("registry/test_jobs.py", skip)
        self.assertIn("registry/test_history_page.py", skip)

    def test_full_and_suites_never_skip(self):
        files = ["registry/store.py"]
        for mode in ("full", "suites"):
            _, skip, _ = G.select(set(G.SUITES), mode, files)
            self.assertEqual(skip, set(), mode)

    def test_child_env_sets_the_switch_only_on_request(self):
        self.assertNotIn(G.SKIP_BROWSER_ENV, G.child_env(base={G.SKIP_BROWSER_ENV: "1"}))
        self.assertEqual(G.child_env(base={}, skip_browser=True)[G.SKIP_BROWSER_ENV], "1")

    def test_timings_prefer_skip_mode_rows(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "t.jsonl"
            f.write_text("".join(json.dumps(r) + "\n" for r in [
                {"unit": "a/x", "seconds": 100}, {"unit": "a/x", "seconds": 10, "skip_browser": True},
                {"unit": "a/y", "seconds": 5}]), encoding="utf-8")
            self.assertEqual(G.past_times(f), {"a/x": 100.0, "a/y": 5.0})
            self.assertEqual(G.past_times(f, skip_browser=True), {"a/x": 10.0, "a/y": 5.0})

    def test_plan_runs_nothing_and_prints_the_skip(self):
        r = subprocess.run([sys.executable, str(G.REPO / "tools" / "test_gate.py"), "--plan", "--files",
                            "jobs/sync.py"], cwd=G.REPO, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("browser classes skipped (no UI change)", r.stdout)
        self.assertIn("estimated", r.stdout)

    def test_estimate_known_answer(self):
        units = [("a", "x", False), ("a", "y", True), ("a", "z", True)]
        times = {"a/x": 10.0, "a/y": 5.0, "a/z": 5.0}
        self.assertEqual(G.estimate_wall(units, times, jobs=3, browser_jobs=1), 10.0)   # y then z on the one lane
        self.assertEqual(G.estimate_wall(units, times, jobs=1, browser_jobs=2), 20.0)   # one worker: serial


if __name__ == "__main__":
    unittest.main()
