"""tools/services.py on a temp config with dummy services: tiny python http servers on free 127.0.0.1 ports.

Known answers: a foreign server on a configured port is reported "port busy (not ours)" and is still alive
after stop; a pid file naming an unrelated live process (this test runner; a foreign server with a wrong
start time) never gets it killed; a stale pid file is replaced; disabled entries never start.
Never touches the real %USERPROFILE%/.cryoev (config, run dir and logs are all in a temp folder).
"""
import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import services as S  # noqa: E402

PY = sys.executable


sys.path.insert(0, str(REPO / "tests"))
from cryoev_testkit import free_ports  # noqa: E402  (shared helpers, round 10)


def http_argv(port):
    return [PY, "-m", "http.server", str(port), "--bind", "127.0.0.1"]


def wait_for(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return cond()


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rc = fn(*a, **kw)
    return rc, out.getvalue()


class ServicesTests(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)  # started services outlive their Popen on purpose
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.tmp.name)
        # a server this tool did not start, on a configured port. The six ports are distinct (held together), but
        # the OS can still give one away between picking and binding, so the foreign server is retried on new ports.
        for _ in range(5):
            self.p = dict(zip(("web1", "web2", "off", "busy", "crash", "tree"), free_ports(6)))
            self.foreign = subprocess.Popen(http_argv(self.p["busy"]), stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL)
            if wait_for(lambda: S.listening("127.0.0.1", self.p["busy"])):
                break
            self.foreign.kill()
        self.assertTrue(S.listening("127.0.0.1", self.p["busy"]), "foreign server did not start")
        child = f"import subprocess,sys,time; subprocess.Popen({http_argv(self.p['tree'])!r}); time.sleep(600)"
        cfg = {"run_dir": str(root / "run"), "services": [
            {"name": "web1", "port": self.p["web1"], "cwd": str(root), "argv": http_argv(self.p["web1"]),
             "log": str(root / "logs" / "web1.log")},
            {"name": "web2", "port": self.p["web2"], "cwd": str(root), "argv": http_argv(self.p["web2"])},
            {"name": "worker", "port": None, "cwd": str(root), "argv": [PY, "-c", "import time; time.sleep(600)"]},
            {"name": "off", "enabled": False, "port": self.p["off"], "cwd": str(root),
             "argv": http_argv(self.p["off"])},
            {"name": "busy", "port": self.p["busy"], "cwd": str(root), "argv": http_argv(self.p["busy"])},
            {"name": "crash", "port": self.p["crash"], "cwd": str(root), "argv": [PY, "-c", "import sys; sys.exit(3)"]},
            {"name": "tree", "port": self.p["tree"], "cwd": str(root), "argv": [PY, "-c", child]},
        ]}
        self.cfg_path = root / "services.json"
        self.cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        self.cfg = S.load_config(self.cfg_path)
        self.by = {s.name: s for s in self.cfg.services}

    def tearDown(self):
        quiet(S.run_actions, self.cfg, self.cfg.services, "stop", 5)
        self.foreign.kill()
        self.foreign.wait()
        self.tmp.cleanup()

    def start(self, *names, wait=15):
        return quiet(S.run_actions, self.cfg, [self.by[n] for n in names], "start", wait)

    def stop(self, *names):
        return quiet(S.run_actions, self.cfg, [self.by[n] for n in names], "stop", 5)

    def status(self):
        return {r["name"]: r for r in (S.service_status(self.cfg, s) for s in self.cfg.services)}

    def test_start_all_status_and_log(self):
        rc, out = quiet(S.main, ["--config", str(self.cfg_path), "start", "all"])
        self.assertEqual(rc, 1, out)  # busy + crash
        st = self.status()
        for n in ("web1", "web2"):
            self.assertEqual(st[n]["state"], "running", out)
            self.assertTrue(st[n]["listening"])
            self.assertTrue(st[n]["health"].startswith("http "), st[n]["health"])  # http.server: 404 on /api/ping
            self.assertIsNotNone(st[n]["uptime_s"])
        self.assertEqual(st["worker"]["state"], "running")
        self.assertEqual(st["worker"]["health"], "-")
        self.assertEqual(st["off"]["state"], "disabled")
        self.assertFalse(st["off"]["listening"])
        self.assertFalse(S.pid_path(self.cfg, "off").exists())
        self.assertEqual(st["busy"]["state"], "port busy (not ours)")
        self.assertFalse(S.pid_path(self.cfg, "busy").exists())
        self.assertIn("port busy (not ours)", out)
        self.assertIn("exited with code 3", out)
        self.assertFalse(S.pid_path(self.cfg, "crash").exists())
        log = self.by["web1"].log
        self.assertTrue(wait_for(lambda: "GET /api/ping" in log.read_text(errors="replace")), log.read_text())
        self.assertIn("services.py start web1", log.read_text())
        self.assertTrue((self.cfg.run_dir.parent / "logs" / "web2.log").exists())  # default log location
        rc, out = quiet(S.main, ["--config", str(self.cfg_path), "status", "--json"])
        self.assertEqual(rc, 0)
        self.assertEqual({r["name"] for r in json.loads(out)}, set(self.by))

    def test_start_again_is_noop(self):
        self.start("web1")
        pid = S.read_pid(self.cfg, "web1")["pid"]
        rc, out = self.start("web1")
        self.assertEqual(rc, 0)
        self.assertIn("already running", out)
        self.assertEqual(S.read_pid(self.cfg, "web1")["pid"], pid)

    def test_stop_only_ours(self):
        self.start("web1", "web2", "worker")
        worker_pid = S.read_pid(self.cfg, "worker")["pid"]
        rc, out = self.stop("web1")
        self.assertEqual(rc, 0, out)
        self.assertFalse(S.listening("127.0.0.1", self.p["web1"]))
        self.assertFalse(S.pid_path(self.cfg, "web1").exists())
        self.assertTrue(S.listening("127.0.0.1", self.p["web2"]))
        rc, out = quiet(S.run_actions, self.cfg, self.cfg.services, "stop", 5)
        self.assertEqual(rc, 1, out)  # busy is reported, not killed
        self.assertIn("port busy (not ours)", out)
        self.assertIsNone(self.foreign.poll(), "foreign server was killed")
        self.assertTrue(S.listening("127.0.0.1", self.p["busy"]))
        self.assertFalse(S.listening("127.0.0.1", self.p["web2"]))
        with S.Proc(worker_pid) as p:
            self.assertFalse(p.alive())

    def test_stop_kills_children_of_ours(self):
        rc, out = self.start("tree")
        self.assertEqual(rc, 0, out)
        self.assertTrue(S.listening("127.0.0.1", self.p["tree"]))
        rc, out = self.stop("tree")
        self.assertEqual(rc, 0, out)
        self.assertTrue(wait_for(lambda: not S.listening("127.0.0.1", self.p["tree"]), 5), "child still listening")

    def test_control_unrelated_pid_never_killed(self):
        # pid file naming this test runner, with web1's command line: must not be killed
        S.write_pid(self.cfg, "web1", {"pid": os.getpid(), "ctime": 1, "cmdline": S.cmdline_of(self.by["web1"].argv)})
        self.assertEqual(self.status()["web1"]["state"], "stale pid file")
        rc, out = self.stop("web1")
        self.assertIn("left alone", out)
        self.assertFalse(S.pid_path(self.cfg, "web1").exists())
        # pid file naming the foreign server with its exact command line but a different start time (pid reuse)
        S.write_pid(self.cfg, "busy", {"pid": self.foreign.pid, "ctime": 1, "cmdline": S.cmdline_of(self.foreign.args)})
        verdict, _ = S.identify(S.read_pid(self.cfg, "busy"))
        self.assertEqual(verdict, "foreign")
        rc, out = self.stop("busy")
        self.assertIn("left alone", out)
        time.sleep(0.5)
        self.assertIsNone(self.foreign.poll(), "foreign server was killed")
        # sanity for the control: the same check does recognise a process this tool started
        self.start("web1")
        verdict, p = S.identify(S.read_pid(self.cfg, "web1"))
        p and p.close()
        self.assertEqual(verdict, "ours")

    def test_stale_pid_file(self):
        dead = subprocess.Popen([PY, "-c", "pass"])
        dead.wait()
        S.write_pid(self.cfg, "web1", {"pid": dead.pid, "ctime": 1, "cmdline": "x"})
        self.assertEqual(self.status()["web1"]["state"], "stale pid file")
        rc, out = self.start("web1")
        self.assertEqual(rc, 0, out)
        self.assertIn("stale pid file", out)
        self.assertEqual(self.status()["web1"]["state"], "running")
        S.pid_path(self.cfg, "web2").write_text("not json")
        rc, out = self.stop("web2")
        self.assertEqual(rc, 0, out)
        self.assertFalse(S.pid_path(self.cfg, "web2").exists())

    def test_disabled_skipped(self):
        rc, out = self.start("off")
        self.assertEqual(rc, 0)
        self.assertIn("disabled", out)
        self.assertFalse(S.pid_path(self.cfg, "off").exists())
        self.assertFalse(S.listening("127.0.0.1", self.p["off"]))

    def test_config_checks(self):
        root = Path(self.tmp.name)
        bad = root / "bad.json"
        for services, msg in [([{"name": "a", "argv": ["x"]}, {"name": "a", "argv": ["y"]}], "duplicate"),
                              ([{"name": "a", "argv": ["x"], "port": 70000}], "port"),
                              ([{"name": "a b", "argv": ["x"]}], "name"),
                              ([{"name": "a", "argv": "x"}], "argv"),
                              ([{"name": "a", "argv": ["x"], "port": 1}, {"name": "b", "argv": ["y"], "port": 1}],
                               "both enabled")]:
            bad.write_text(json.dumps({"services": services}), encoding="utf-8")
            with self.assertRaises(S.ConfigError) as cm:
                S.load_config(bad)
            self.assertIn(msg, str(cm.exception))
        # PowerShell 5.1 `>` writes UTF-16: still readable; placeholders expanded
        ok = root / "u16.json"
        ok.write_text(json.dumps({"services": [{"name": "a", "argv": ["{python}", "x"], "cwd": "{repo}"}]}),
                      encoding="utf-16")
        c = S.load_config(ok)
        self.assertEqual(c.services[0].argv[0], S.python_exe())
        self.assertEqual(c.services[0].cwd, S.REPO)
        rc, out = quiet(S.main, ["--config", str(self.cfg_path), "start", "nope"])
        self.assertEqual(rc, 2)

    def test_example_config(self):
        rc, out = quiet(S.main, ["example-config"])
        self.assertEqual(rc, 0)
        names = [s["name"] for s in json.loads(out)["services"]]
        self.assertEqual(names, ["review", "sandbox", "history", "monitor", "portal"])
        dest = Path(self.tmp.name) / "ex" / "services.json"
        self.assertEqual(quiet(S.main, ["example-config", "--out", str(dest)])[0], 0)
        cfg = S.load_config(dest)
        self.assertFalse({s.name: s for s in cfg.services}["portal"].enabled)
        self.assertEqual({s.name: s for s in cfg.services}["portal"].health, "/healthz")
        self.assertIsNone({s.name: s for s in cfg.services}["review"].health)   # unset = today's behaviour
        self.assertEqual(quiet(S.main, ["example-config", "--out", str(dest)])[0], 2)  # never overwrites


class HealthTests(unittest.TestCase):
    """health(): with a path, up = 200 + JSON ok:true; without one, any HTTP answer is up (the control)."""

    def serve(self, status, body):
        import http.server
        import threading

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return srv.server_address[1]

    def test_healthz_ok_is_up(self):
        self.assertEqual(S.health("127.0.0.1", self.serve(200, b'{"ok": true}'), path="/healthz"), "ok")

    def test_healthz_not_ok_is_not_up(self):
        for status, body in ((401, b'{"ok": true}'), (404, b"nope"), (200, b'{"ok": false}'), (200, b"<html>"),
                             (200, b"[]"), (200, b'{"ok": "yes"}')):
            self.assertNotEqual(S.health("127.0.0.1", self.serve(status, body), path="/healthz"), "ok", (status, body))

    def test_no_health_option_any_answer_is_up(self):             # control: today's behaviour for review servers
        for status in (200, 403, 404):
            self.assertEqual(S.health("127.0.0.1", self.serve(status, b"x")), f"http {status}")

    def test_dead_port_down(self):
        with socket.socket() as held:                 # bound, never listening, and held: nobody else can take it
            held.bind(("127.0.0.1", 0))
            self.assertEqual(S.health("127.0.0.1", held.getsockname()[1], path="/healthz"), "down")

    def test_config_health_validated(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            for h, ok in (("/healthz", True), ("healthz", False), (5, False), ("/a b", False), ("/a\nb", False)):
                f = Path(d) / "c.json"
                f.write_text(json.dumps({"run_dir": str(Path(d) / "run"), "services": [
                    {"name": "p", "port": 1234, "argv": ["x"], "health": h}]}), encoding="utf-8")
                if ok:
                    self.assertEqual(S.load_config(f).services[0].health, "/healthz")
                else:
                    with self.assertRaises(S.ConfigError):
                        S.load_config(f)


if __name__ == "__main__":
    unittest.main()
