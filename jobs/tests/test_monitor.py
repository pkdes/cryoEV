"""jobs/monitor.py on temp queue folders built in code (no real share, no real registry).

Known answers: cumulative times 30, 42.5, 55, 67.5 -> median 12.5 s and ETA (300 - 4) x 12.5 = 3700 s; best epoch 2 +
patience 3 at epoch 4 -> bound 12.5 s; batch 8 of 16 -> halved; a 10-min-old heartbeat -> stale; 20 min without an
epoch -> hung; a missing share -> offline; a cancelled DONE.json -> recent finished. Each comes with its control
(fresh heartbeat, fresh epoch, a reachable share next to the missing one). Controls on the monitor itself: the share's
sha256 manifest is unchanged by a run; a job runs to done while monitors read it every 0.2 s; a second monitor exits 3.
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobtest as T  # noqa: E402
from jobtest import C  # noqa: E402

import monitor as M  # noqa: E402

MONITOR = [sys.executable, str(T.JOBS / "monitor.py")]


def manifest(root: Path) -> dict:
    """{relative path: sha256} for every file, plus every folder (None): the share's whole state."""
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return out


class MonitorTests(T.TempRoots, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.lap = self.tmp / "lap"                      # queue_rows reads <laptop>/jobs/pulled.jsonl: a temp one
        self.lap.mkdir()
        os.environ.update(CRYOEV_LAPTOP_ROOT=str(self.lap))
        os.environ.pop("CRYOEV_REGISTRY", None)
        self.out = self.tmp / "reg" / "live" / "live.json"

    def once(self, *shares, out=None, env=None, servers=""):
        """servers="" by default: the tests never probe this machine's real review / history ports."""
        argv = [*MONITOR, *[x for s in shares for x in ("--share", str(s))], "--once", "--out", str(out or self.out),
                "--servers", servers]
        r = subprocess.run(argv, capture_output=True, text=True, env={**os.environ, **(env or {})}, timeout=120)
        return r

    def snap(self, *shares):
        r = self.once(*shares)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(self.out.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ known answers
    def test_eta_known_answer(self):
        rows = M.read_results(b"epoch,time\n1,30\n2,42.5\n3,55\n4,67.5\n")
        self.assertEqual(M.epoch_median(rows), 12.5)                 # epoch 1 (30 s, setup) skipped
        self.assertEqual(M.eta(4, 300, 12.5)["eta_s"], 3700)
        T.running_fixture(self.share, progress={"best_epoch": 2}, patience=3)
        r = M.running_job(self.share, "job_20261104_live")
        self.assertEqual((r["epoch"], r["epochs"], r["epoch_median_s"]), (4, 300, 12.5))
        self.assertEqual(r["eta_s"], 3700)
        self.assertEqual(r["eta_patience_bound_s"], 12.5)            # (2 + 3 - 4) x 12.5
        self.assertEqual(r["phase"], "training")
        # controls: patience 300 makes the bound (3750 s) larger than the ETA, so none; a non-positive step is skipped
        self.assertIsNone(M.eta(4, 300, 12.5, best_epoch=2, patience=300)["eta_patience_bound_s"])
        self.assertEqual(M.epoch_median(M.read_results(b"epoch,time\n1,30\n2,42.5\n3,40\n4,52.5\n")), 12.5)
        # the job's time cap: 1 h, 3500 s used -> at most 100 s left
        capped = M.eta(4, 300, 12.5, max_hours=1, elapsed_s=3500)
        self.assertEqual((capped["eta_s"], capped["eta_capped"]), (100, True))

    def test_batch_halved_and_older_run(self):
        T.running_fixture(self.share, progress={"batch": 8, "batch_asked": 16})
        r = M.running_job(self.share, "job_20261104_live")
        self.assertEqual((r["batch"], r["batch_asked"], r["batch_halved"]), (8, 16, True))
        self.assertEqual(r["gpu_reserved_gb"], 31.4)
        T.running_fixture(self.share, job="job_20261104_same", progress={"batch": 16, "batch_asked": 16})
        self.assertFalse(M.running_job(self.share, "job_20261104_same")["batch_halved"])   # control
        T.running_fixture(self.share, job="job_20261104_old")          # no progress.json: an older run
        r = M.running_job(self.share, "job_20261104_old")
        self.assertIsNone(r["batch"])
        self.assertEqual((r["batch_asked"], r["batch_note"]), (16, "unknown (older run)"))
        self.assertEqual((r["epoch"], r["epochs"], r["eta_s"]), (4, 300, 3700))   # results.csv + args.yaml
        self.assertEqual(r["map50_95"], 0.2)

    def test_watcher_stale(self):
        T.watch_fixture(self.share, age_s=600)
        w = self.snap(self.share)["queues"][0]["watcher"]
        self.assertEqual((w["display"], w["stale"]), ("stale", True))
        T.watch_fixture(self.share, age_s=10)                          # control: a fresh heartbeat
        w = self.snap(self.share)["queues"][0]["watcher"]
        self.assertEqual((w["display"], w["stale"]), ("alive", False))
        T.watch_fixture(self.share, state="waiting", detail="GPU busy for 3 min: pid 1 python.exe 900 MB")
        self.assertEqual(self.snap(self.share)["queues"][0]["watcher"]["display"], "waiting for GPU")

    def test_hung(self):
        T.running_fixture(self.share, progress={}, age_s=1200)
        r = M.running_job(self.share, "job_20261104_live")
        self.assertEqual(r["hung_after_s"], 600)                       # max(5 x 12.5, 600)
        self.assertTrue(r["hung"])
        T.running_fixture(self.share, job="job_20261104_fresh", progress={}, age_s=30)
        self.assertFalse(M.running_job(self.share, "job_20261104_fresh")["hung"])          # control

    def test_offline_and_loop_continues(self):
        T.watch_fixture(self.share, job="job_20261104_live")
        T.running_fixture(self.share, progress={})
        missing = self.tmp / "no_such_share"
        s = self.snap(missing, self.share)                             # --once returned 0 (snap checks)
        q0, q1 = s["queues"]
        self.assertEqual((q0["reach"], q0["running"]), ("offline", None))
        self.assertEqual(q1["reach"], "ok")                            # control: the next queue is still read
        self.assertEqual(q1["running"]["job"], "job_20261104_live")
        self.assertTrue(q1["running"]["watcher_on_it"])

    def test_access_denied_and_timeout(self):
        with mock.patch.object(M.os, "stat", side_effect=PermissionError(13, "Access is denied")):
            self.assertEqual(M.reach(self.share)[0], "access denied")
        self.assertEqual(M.reach(self.share)[0], "ok")                 # control
        gate = threading.Event()
        with mock.patch.object(M, "read_queue", side_effect=lambda share: (gate.wait(5), {"reach": "ok"})[1]):
            rd = M.Reader(timeout=0.2)
            self.assertIn("no answer", rd.read(self.share)["detail"])
            self.assertIn("previous read", rd.read(self.share)["detail"])   # never two reads stuck on one share
            gate.set()
            time.sleep(0.3)
            self.assertEqual(rd.read(self.share)["reach"], "ok")

    def test_cancelled_in_recent_finished(self):
        T.done_fixture(self.share, "job_20261104_cxl", status="cancelled", reason="cancelled by pujan (x): test",
                       ended="2026-11-04T11:00:00")
        T.done_fixture(self.share, "job_20261104_ok", ended="2026-11-04T09:00:00")
        f = self.snap(self.share)["queues"][0]["finished"]
        self.assertEqual([x["job"] for x in f], ["job_20261104_cxl", "job_20261104_ok"])   # newest first
        self.assertEqual((f[0]["status"], f[0]["reason"]), ("cancelled", "cancelled by pujan (x): test"))
        self.assertEqual((f[1]["metrics"]["mask_map50_95"], f[1]["best_epoch"]), (0.32, 140))

    # ------------------------------------------------------------ run folder details
    def test_latest_run_mapped_from_desktop_path(self):
        run = T.running_fixture(self.share, progress={})
        newer = run.parent / "2026-11-05_000000_y8n_640_4b_10ep_Run_2"   # newer by name, not the one LATEST_RUN names
        newer.mkdir()
        self.assertEqual(M.find_run(self.share, "job_20261104_live"), run)
        (self.share / C.WORK / "job_20261104_live" / "LATEST_RUN.txt").unlink()
        self.assertEqual(M.find_run(self.share, "job_20261104_live"), newer)                # fallback: newest

    def test_phases_and_partial_line(self):
        T.running_fixture(self.share, csv=False)
        self.assertEqual(M.running_job(self.share, "job_20261104_live")["phase"], "setting up")
        T.running_fixture(self.share, job="job_20261104_fin", progress={"phase": "final_eval"})
        self.assertEqual(M.running_job(self.share, "job_20261104_fin")["phase"], "finishing")
        T.running_fixture(self.share, job="job_20261104_last", times=(30, 42.5), epochs=2)
        self.assertEqual(M.running_job(self.share, "job_20261104_last")["phase"], "finishing")
        T.running_fixture(self.share, job="job_20261104_tail", tail="5,80,0.4,0.6")       # epoch 5 half written
        r = M.running_job(self.share, "job_20261104_tail")
        self.assertEqual((r["epoch"], r["series"]["epoch"]), (4, [1, 2, 3, 4]))
        self.assertEqual(r["series"]["box"], [2.0, 1.0, 2 / 3, 0.5])
        self.assertEqual(r["series"]["map50_95"], [0.05, 0.1, 0.15, 0.2])

    def test_cancel_requested(self):
        T.running_fixture(self.share, progress={})
        (self.share / C.Q / "job_20261104_live" / C.CANCEL).write_text("{}", encoding="utf-8")
        self.assertTrue(self.snap(self.share)["queues"][0]["running"]["cancel_requested"])

    # ------------------------------------------------------------ local server probes
    def test_server_probes(self):
        """A live dummy server answering 403 (as the review server does without its key) and one answering 404 (as
        http.server does for /api/ping) are up; a port nobody listens on and one that accepts but never answers are
        down, within the time bound. The snapshot carries them, in --servers order."""
        import http.server
        import socket

        class Answer(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(403 if self.server.server_address[1] == live else 404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *a):
                pass
        servers = []
        for _ in range(2):
            srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Answer)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            self.addCleanup(srv.server_close)
            self.addCleanup(srv.shutdown)
            servers.append(srv)
        live, live404 = (s.server_address[1] for s in servers)
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
        s.close()                                                      # nothing listens on `dead` any more
        mute = socket.socket()                                         # listens, never accepts or answers
        mute.bind(("127.0.0.1", 0))
        mute.listen(1)
        self.addCleanup(mute.close)
        hang = mute.getsockname()[1]
        t0 = time.monotonic()
        r = M.probe_servers([("review", live), ("history", live404), ("sandbox", dead), ("mute", hang)])
        self.assertLess(time.monotonic() - t0, M.PROBE_TOTAL_S + 0.5)
        self.assertEqual([(x["name"], x["port"], x["up"]) for x in r],
                         [("review", live, True), ("history", live404, True), ("sandbox", dead, False),
                          ("mute", hang, False)])
        self.assertIsInstance(r[0]["ms"], int)
        t0 = time.monotonic()                                          # the total bound holds whatever the timeout
        self.assertFalse(M.probe_servers([("mute", hang)], timeout=5, total=0.5)[0]["up"])
        self.assertLess(time.monotonic() - t0, 1.5)
        # through the CLI: the snapshot has servers; the default list is the laptop's three
        snap = self.once(self.share, servers=f"review={live},sandbox={dead}")
        self.assertEqual(snap.returncode, 0, snap.stderr)
        self.assertIn(f"servers: review=127.0.0.1:{live}, sandbox=127.0.0.1:{dead}", snap.stdout)
        got = json.loads(self.out.read_text(encoding="utf-8"))["servers"]
        self.assertEqual([(x["name"], x["port"], x["up"]) for x in got], [("review", live, True), ("sandbox", dead, False)])
        self.assertEqual(M.parse_servers(M.SERVERS), [("review", 8766), ("sandbox", 8765), ("history", 8768)])
        self.assertEqual(M.parse_servers(""), [])
        self.assertEqual(json.loads(self.out.read_text(encoding="utf-8"))["queues"][0]["reach"], "ok")
        for bad in ("review", "review=x", "=8766", "review=70000"):
            with self.assertRaises(ValueError):
                M.parse_servers(bad)
        self.assertEqual(self.once(self.share, servers="review").returncode, 2)   # argparse error

    # ------------------------------------------------------------ controls on the monitor itself
    def test_share_unchanged(self):
        """(a) sha256 manifest of the share is identical before and after a run."""
        T.watch_fixture(self.share, job="job_20261104_live")
        T.running_fixture(self.share, progress={"batch": 8, "batch_asked": 16})
        T.done_fixture(self.share, "job_20261104_cxl", status="cancelled", reason="test")
        before = manifest(self.share)
        self.assertGreater(len(before), 15)
        s = self.snap(self.share)
        self.assertEqual(s["queues"][0]["running"]["epoch"], 4)       # it really read the run
        self.assertEqual(manifest(self.share), before)
        self.assertFalse(any("monitor" in k or k.endswith(".tmp") for k in manifest(self.share)))

    def test_job_runs_to_done_under_monitoring(self):
        """(b) run_job.py with the fake trainer (progress.json by os.replace with train_yolo.py's retry) while a monitor
        reads the queue every 0.2 s: the job ends done, no .stale_ folder, no lost progress.json write, and the monitor
        saw it running. A refused first try is possible (any open handle, shared or not, makes a replace onto the
        file fail for that moment on NTFS); the writer's retry covers it, as in the real trainer."""
        zp = T.make_bundle(self.bundles)
        T.queue(self.share, "job_20261104_conc", zp)
        env = {**os.environ, **T.hooks_env(), "FAKE_EPOCHS": "20", "FAKE_EPOCH_S": "0.25"}
        runner = subprocess.Popen([sys.executable, str(T.JOBS / "run_job.py"), "--queue-dir", str(self.share)],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        seen, n = set(), 0
        while runner.poll() is None:
            r = self.once(self.share)
            self.assertEqual(r.returncode, 0, r.stderr)
            n += 1
            q = json.loads(self.out.read_text(encoding="utf-8"))["queues"][0]
            if q["running"] and q["running"]["epoch"]:
                seen.add(q["running"]["epoch"])
            time.sleep(0.2)
        log = runner.communicate()[0]
        self.assertEqual(runner.returncode, 0, log)
        done = self.done("job_20261104_conc")
        self.assertEqual(done["status"], "done", done)
        dd = self.share / C.D
        self.assertEqual([p.name for p in dd.iterdir() if ".stale_" in p.name or p.name.startswith(".tmp_")], [])
        run = dd / "job_20261104_conc" / "runs" / done["run"]
        errs = run / "fake_replace_errors.txt"
        self.assertFalse(errs.exists(), errs.exists() and errs.read_text(encoding="utf-8"))
        retries = run / "fake_replace_retries.txt"
        print(f"[test_monitor] {n} monitor reads; progress.json tries refused then retried: "
              f"{len(retries.read_text(encoding='utf-8').splitlines()) if retries.exists() else 0} of 20 writes")
        self.assertEqual(C.read_json(run / "progress.json")["epoch"], 20)
        self.assertGreaterEqual(n, 5)
        self.assertGreaterEqual(len(seen), 2, f"monitor saw epochs {sorted(seen)} in {n} reads")
        q = self.snap(self.share)["queues"][0]
        self.assertEqual((q["running"], q["finished"][0]["job"]), (None, "job_20261104_conc"))

    def test_second_monitor_exits_3(self):
        """(c) one monitor per output file: with monitor.lock held, a second exits 3; with it free, 0 (control)."""
        self.assertEqual(self.once(self.share).returncode, 0)
        from data_utils.oslock import acquire_lock
        lock = acquire_lock(self.out.parent / M.LOCK_FILE, 0)
        try:
            r = self.once(self.share)
            self.assertEqual(r.returncode, 3, r.stderr)
            self.assertIn("another monitor", r.stderr)
        finally:
            lock.close()
        self.assertEqual(self.once(self.share).returncode, 0)


class PortalWatchTests(unittest.TestCase):
    """--portal: a fake /healthz server; the sender is mocked (no network); the topic never appears in a log or a
    message."""
    TOPIC = "sekrit-topic-value-123"
    OK = ("200", b'{"ok": true}')

    def setUp(self):
        import http.server

        outer = self
        outer.mode = self.OK

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                status, body = outer.mode
                self.send_response(int(status))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)
        self.port = self.srv.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/healthz"
        self.sent, self.logs = [], []
        self.saved = os.environ.get("CRYOEV_NTFY_TOPIC")
        os.environ["CRYOEV_NTFY_TOPIC"] = self.TOPIC
        self.addCleanup(self.restore)
        self.w = M.PortalWatch([self.url], send=lambda t, m, p=3: self.sent.append((t, m, p)), log=self.logs.append)

    def restore(self):
        if self.saved is None:
            os.environ.pop("CRYOEV_NTFY_TOPIC", None)
        else:
            os.environ["CRYOEV_NTFY_TOPIC"] = self.saved

    def test_up_down_stays_down_recovers(self):
        r = self.w.poll()
        self.assertEqual((r[0]["up"], r[0]["port"], r[0]["fails"]), (True, self.port, 0))
        self.assertEqual(self.sent, [])
        self.mode = ("401", b'{"ok": true}')                     # 401 is down, whatever the body
        self.w.poll()
        self.assertEqual(self.sent, [])                          # one failure: no ntfy
        r = self.w.poll()
        self.assertFalse(r[0]["up"])
        self.assertEqual(len(self.sent), 1)                      # two: one ntfy
        self.assertIn(f":{self.port}", self.sent[0][1])
        self.assertNotIn("127.0.0.1", self.sent[0][1])
        for _ in range(3):
            self.w.poll()
        self.assertEqual(len(self.sent), 1)                      # stays down: no repeat
        self.mode = self.OK
        self.w.poll()
        self.assertEqual(len(self.sent), 2)                      # recovery: one ntfy
        self.assertIn("recovered", self.sent[1][0])
        self.w.poll()
        self.assertEqual(len(self.sent), 2)
        for s in [*self.logs, *[x for m in self.sent for x in m[:2]]]:
            self.assertNotIn(self.TOPIC, str(s))

    def test_single_blip_never_alerts(self):
        self.w.poll()
        self.mode = ("500", b"")
        self.w.poll()
        self.mode = self.OK
        self.w.poll()
        self.mode = ("500", b"")
        self.w.poll()
        self.assertEqual(self.sent, [])                          # failures were never two in a row

    def test_not_ok_body_and_dead_port_are_down(self):
        self.mode = ("200", b'{"ok": false}')
        self.assertFalse(M.portal_check(self.url)[0])
        self.mode = self.OK
        self.assertTrue(M.portal_check(self.url)[0])             # control
        self.srv.shutdown()
        self.srv.server_close()
        self.assertFalse(M.portal_check(self.url, timeout=1)[0])

    def test_no_topic_logs_once_and_sends_nothing(self):
        os.environ.pop("CRYOEV_NTFY_TOPIC", None)
        self.mode = ("404", b"")
        for _ in range(4):
            self.w.poll()
        self.mode = self.OK
        self.w.poll()
        self.assertEqual(self.sent, [])
        self.assertEqual(sum("ntfy not configured" in x for x in self.logs), 1)

    def test_default_sender_goes_through_notify_without_network(self):
        from unittest import mock
        w = M.PortalWatch([self.url], log=self.logs.append)
        self.mode = ("404", b"")
        with mock.patch("urllib.request.urlopen") as uo:
            w.poll()
            w.poll()
        self.assertEqual(uo.call_count, 1)
        self.assertNotIn(self.TOPIC, " ".join(self.logs))

    def test_snapshot_has_portals_only_when_asked(self):
        rd = M.Reader()
        self.assertNotIn("portals", M.snapshot([], rd, 30))
        self.assertEqual(M.snapshot([], rd, 30, (), self.w)["portals"][0]["port"], self.port)


if __name__ == "__main__":
    unittest.main()
