"""jobs/predict_local.py waiting and chaining (fake predictor only, never the real GPU), in temp roots.

Known answers: two sets started together -> the second shows waiting "another analysis is running" with its own
waiter pid and a refreshing `updated`, then runs to done by itself; a busy GPU hook cleared after a few checks -> the
set runs; three queued sets complete from one start (chain); "Start again" on a waiting set starts no second run
(the predictor ran exactly once per set, counted from its argv log). Control for the duplicate check: the same set
runs once it is free.
"""
import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobtest as T  # noqa: E402
from jobtest import C, run_job  # noqa: E402
from test_predict_set import FAKE_PRED, GOLDEN_PRED, PredictRoots, assert_contract_layout  # noqa: E402

import predict_local as PL  # noqa: E402
from data_utils.oslock import acquire_lock  # noqa: E402

FAST = dict(wait=60, wait_poll_s=0.3, poll_s=0.1)


class QueueTests(PredictRoots, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.argv_file = self.tmp / "argv.jsonl"
        os.environ.update(T.hooks_env(train_cmd=None, gpu=T.LAPTOP_GPU, predict_cmd=FAKE_PRED),
                          FAKE_PRED_FROM=str(GOLDEN_PRED), FAKE_PRED_ARGV=str(self.argv_file), FAKE_PRED_DELAY="0.8")

    def status(self, ps):
        for _ in range(20):                            # a reader can meet the file mid-replace on Windows
            try:
                return C.read_json(ps / "status.json")
            except (OSError, ValueError):
                time.sleep(0.05)
        return C.read_json(ps / "status.json")

    def wait_for(self, cond, timeout=90, what="condition"):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return
            time.sleep(0.1)
        self.fail(f"timed out waiting for {what}")

    def runs_per_set(self):
        """How often the predictor ran for each set's images folder (one argv line per run)."""
        out = {}
        if self.argv_file.exists():
            for ln in self.argv_file.read_text(encoding="utf-8").splitlines():
                argv = json.loads(ln)
                ps = Path(argv[argv.index("--images") + 1]).parent.name
                out[ps] = out.get(ps, 0) + 1
        return out

    def done(self, *sets):
        return all(self.status(ps)["state"] in ("done", "failed") for ps in sets)

    def test_two_started_together_second_waits_then_runs(self):
        a, b = self.make(), self.make()
        PL.start_detached(a, self.root, self.reg, **FAST, chain=False)
        self.wait_for(lambda: self.status(a)["state"] == "running", what="a running")
        pid_b = PL.start_detached(b, self.root, self.reg, **FAST, chain=False)
        beats = set()
        while self.status(a)["state"] == "running":
            st = self.status(b)
            if st["waiting"] == PL.WAIT_LOCK:
                self.assertEqual((st["state"], st["pid"]), ("queued", pid_b))
                beats.add(st["updated"])
            time.sleep(0.1)
        self.assertGreaterEqual(len(beats), 2, f"`updated` not refreshed while waiting: {beats}")
        self.wait_for(lambda: self.done(a, b), what="both done")
        for ps in (a, b):
            assert_contract_layout(self, ps)
            self.assertIsNone(self.status(ps)["waiting"])
        self.assertEqual(self.runs_per_set(), {a.name: 1, b.name: 1})

    def test_gpu_busy_clears_then_runs(self):
        a = self.make()
        hooks_file = self.tmp / "hooks.json"
        hooks_file.write_text(json.dumps({"gpu": {"total_mb": 4096, "apps": [[4242, "python.exe", 900]]}}))
        os.environ[run_job.HOOKS_FILE_ENV] = str(hooks_file)
        box = {}
        t = threading.Thread(target=lambda: box.update(r=PL.run(a.name, self.root, self.reg, wait=30,
                                                                 wait_poll_s=0.1, poll_s=0.1)))
        t.start()
        self.wait_for(lambda: str(self.status(a)["waiting"]).startswith("gpu busy: pid 4242"), 20, "gpu busy note")
        self.assertEqual((self.status(a)["state"], self.status(a)["pid"]), ("queued", os.getpid()))
        time.sleep(0.5)                                # a few checks while busy
        self.assertEqual(self.runs_per_set(), {}, "ran while the GPU was busy")
        hooks_file.write_text(json.dumps({"gpu": {"total_mb": 4096, "apps": []}}))   # the labmate is done
        t.join(60)
        self.assertEqual(box["r"]["exit"], PL.EXIT_DONE)
        assert_contract_layout(self, a)
        self.assertEqual(self.runs_per_set(), {a.name: 1})

    def test_chain_three_sets_one_start(self):
        sets = [self.make() for _ in range(3)]
        PL.start_detached(sets[0], self.root, self.reg, **FAST)          # chain on (the default)
        self.wait_for(lambda: self.done(*sets), 120, "all three done")
        for ps in sets:
            assert_contract_layout(self, ps)
        self.assertEqual(self.runs_per_set(), {ps.name: 1 for ps in sets})
        self.assertIsNone(PL.next_queued(self.root))

    def test_start_again_starts_no_second_run(self):
        a = self.make()
        lock = acquire_lock(self.root / PL.LOCK_FILE)  # another analysis holds the root
        try:
            pid = PL.start_detached(a, self.root, self.reg, **FAST)
            self.wait_for(lambda: self.status(a)["waiting"] == PL.WAIT_LOCK and self.status(a)["pid"] == pid,
                          30, "the first waiter")
            with self.assertRaises(PL.Refused) as cm:  # "Start again", in this process
                PL.run(a.name, self.root, self.reg, wait=5, wait_poll_s=0.1)
            self.assertEqual(cm.exception.code, PL.EXIT_LOCKED)
            self.assertIn("already waits for it or runs it", str(cm.exception))
            PL.start_detached(a, self.root, self.reg, **FAST)               # and as the portal would
            time.sleep(1.5)
            self.assertEqual(self.status(a)["pid"], pid, "the second waiter took the set")
            self.assertEqual(self.runs_per_set(), {})
        finally:
            lock.close()
        self.wait_for(lambda: self.done(a), what="a done")            # control: free again, it runs, once
        time.sleep(1.0)
        assert_contract_layout(self, a)
        self.assertEqual(self.runs_per_set(), {a.name: 1})


if __name__ == "__main__":
    unittest.main()
