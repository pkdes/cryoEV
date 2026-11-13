"""
One writer per review package (round 9, R2): annotation/multi_review_server.py and annotation/review_server.py hold an
OS lock per package (<root>/.review_locks/<pkg>.lock) for as long as they run, and refuse a package whose lock another
process holds; the multi server also serves lab_ (compare) packages and never serves an excluded package (--exclude /
$CJ_REVIEW_EXCLUDE); multi_review_server.main() strips CRYOEV_VIA / CRYOEV_FOR (registry/actor.strip_env), not at import.

Real processes (servers in subprocesses) on temp packages built by make_fixture: the in-process write lock
(review_server.py's threading.Lock) can't see another process, so two servers on one package used to both accept
saves, each merging against the edits file as it last read it.

  python -m unittest annotation.tests.test_review_lock -v
"""
import http.client
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import make_fixture  # noqa: E402
from annotation import multi_review_server as mrs  # noqa: E402
from annotation import review_server as rs  # noqa: E402
from test_review_server import free_port  # noqa: E402

KEY = "lock-test-key-1"
PKG = "lab_compare_fixture"
FILE = "fixC_old.png"


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CRYOEV_", "CJ_"))}
    env.update(extra)
    return env


def build_pkg(root: Path, name=PKG, key=KEY):
    pkg = root / name
    make_fixture.build(pkg)
    (pkg / "review_package.json").write_text(json.dumps({"id": name, "kind": "compare", "owner": "pujan",
                                                         "reviewers": ["au1"], "n_images": 4}), encoding="utf-8")
    (pkg / ".review_key").write_text(key, encoding="utf-8")
    return pkg


def request(port, method, path, body=None, key=KEY):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"X-Review-Key": key} if key else {}
    c.request(method, path, body=json.dumps(body).encode() if body is not None else None, headers=h)
    r = c.getresponse()
    raw = r.read()
    c.close()
    return r.status, raw


def edit(v, t):
    return {"t": t, "dec": {"g301": {"v": v, "t": t, "who": "au1", "pending": True}}}


class Proc:
    """A server in a subprocess; .port once it answers HTTP, or .code if it exited."""

    def __init__(self, argv, tmp: Path, env=None):
        self.port = free_port()
        self.logf = tmp / f"srv_{self.port}.log"
        self._log = open(self.logf, "wb")
        self.p = subprocess.Popen([sys.executable, *argv, "--port", str(self.port)], cwd=REPO, env=env or clean_env(),
                                  stdout=self._log, stderr=subprocess.STDOUT)
        for _ in range(300):
            if self.p.poll() is not None:
                break
            try:
                request(self.port, "GET", "/", key=None)
                break
            except OSError:
                time.sleep(0.05)
        self.code = self.p.poll()

    def log(self):
        self._log.flush()
        return self.logf.read_text(encoding="utf-8", errors="replace")

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            self.p.wait(10)
        self._log.close()


class TwoServersOneLock(unittest.TestCase):
    def setUp(self):
        for k in [k for k in os.environ if k.startswith("CRYOEV_") or k == mrs.EXCLUDE_ENV]:
            self.addCleanup(os.environ.__setitem__, k, os.environ.pop(k))
        self.tmp = Path(tempfile.mkdtemp(prefix="review_lock_"))
        self.root = self.tmp / "review"
        self.root.mkdir()
        self.pkg = build_pkg(self.root)
        self.procs = []
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def tearDown(self):
        for p in self.procs:
            p.stop()

    def multi(self, *args, env=None):
        p = Proc(["-m", "annotation.multi_review_server", "--root", str(self.root), *args], self.tmp, env)
        self.procs.append(p)
        self.assertIsNone(p.code, p.log())
        return p

    def single(self):
        p = Proc(["annotation/review_server.py", "--package", str(self.pkg), "--key", KEY], self.tmp)
        self.procs.append(p)
        return p

    def saved(self):
        f = self.pkg / "edits" / f"{FILE}.json"
        return json.loads(f.read_text(encoding="utf-8"))["edit"]["dec"]["g301"]["v"] if f.exists() else None

    def test_two_multi_servers_one_package(self):
        a, b = self.multi(), self.multi()
        self.assertEqual(request(a.port, "GET", f"/{PKG}/api/ping")[0], 200)   # a lab_ package is served; a holds it
        st, body = request(b.port, "GET", f"/{PKG}/api/ping")
        self.assertEqual(st, 409, body)
        self.assertIn(b"another review server", body)
        st, _ = request(b.port, "PUT", f"/{PKG}/api/edits?f={quote(FILE)}", edit("keep:MV", 1000))
        self.assertEqual(st, 409)
        self.assertIsNone(self.saved())                                       # b wrote nothing
        st, _ = request(a.port, "PUT", f"/{PKG}/api/edits?f={quote(FILE)}", edit("keep:VLDL", 2000))
        self.assertEqual(st, 200)
        self.assertEqual(self.saved(), "keep:VLDL")
        self.assertEqual(request(b.port, "GET", f"/{PKG}/api/edits")[0], 409)  # every request, not only the first
        st, _ = request(b.port, "GET", f"/{PKG}/api/ping", key="wrong")       # a wrong key: still the unknown answer
        self.assertEqual(st, 403)
        a.stop()                                                              # the lock dies with its process
        self.assertEqual(request(b.port, "GET", f"/{PKG}/api/ping")[0], 200)
        st, _ = request(b.port, "PUT", f"/{PKG}/api/edits?f={quote(FILE)}", edit("keep:EV", 3000))
        self.assertEqual(st, 200)
        self.assertEqual(self.saved(), "keep:EV")
        self.assertTrue((self.root / rs.LOCK_DIR / f"{PKG}.lock").is_file())
        self.assertFalse(any(self.pkg.glob("*.lock")))                        # never inside the package

    def test_single_and_multi_exclude_each_other(self):
        s = self.single()
        self.assertIsNone(s.code, s.log())
        self.assertEqual(request(s.port, "GET", "/api/ping")[0], 200)
        m = self.multi()
        self.assertEqual(request(m.port, "GET", f"/{PKG}/api/ping")[0], 409)
        s.stop()
        self.assertEqual(request(m.port, "GET", f"/{PKG}/api/ping")[0], 200)  # now the multi server holds it
        s2 = self.single()
        self.assertEqual(s2.code, 2, s2.log())                                # refuses to start
        self.assertIn("open in another review server", s2.log())

    def test_two_single_servers(self):
        s1, s2 = self.single(), self.single()
        self.assertIsNone(s1.code, s1.log())
        self.assertEqual(s2.code, 2, s2.log())

    def test_exclude_arg_and_env(self):
        m = self.multi("--exclude", str(self.pkg))
        self.assertEqual(request(m.port, "GET", f"/{PKG}/api/ping")[0], 403)   # as an unknown id
        self.assertFalse((self.root / rs.LOCK_DIR / f"{PKG}.lock").exists())  # never locked, never touched
        self.assertFalse((self.pkg / "edits").exists())
        self.assertIn(f"Never served (--exclude / ${mrs.EXCLUDE_ENV}): {PKG.lower()}", m.log())
        m2 = self.multi(env=clean_env(**{mrs.EXCLUDE_ENV: f"other_pkg, {PKG}"}))
        self.assertEqual(request(m2.port, "GET", f"/{PKG}/api/ping")[0], 403)
        m3 = self.multi("--prefix", "pred_")                                   # an explicit prefix: lab_ not served
        self.assertEqual(request(m3.port, "GET", f"/{PKG}/api/ping")[0], 403)
        m4 = self.multi()
        self.assertEqual(request(m4.port, "GET", f"/{PKG}/api/ping")[0], 200)

    def test_main_strips_agent_env(self):
        m = self.multi(env=clean_env(CRYOEV_VIA="claude", CRYOEV_FOR="pujan"))
        self.assertIn("[OK] Removed CRYOEV_VIA, CRYOEV_FOR from this server's environment", m.log())
        m2 = self.multi()
        self.assertNotIn("Removed", m2.log())


class InProcess(unittest.TestCase):
    """make_server in this process: excluded ids, ids(), the lock released on server_close, and that importing the
    module never strips the env (only main() does)."""

    def setUp(self):
        for k in [k for k in os.environ if k.startswith("CRYOEV_") or k == mrs.EXCLUDE_ENV]:
            self.addCleanup(os.environ.__setitem__, k, os.environ.pop(k))
        self.tmp = Path(tempfile.mkdtemp(prefix="review_lock_ip_"))
        self.root = self.tmp / "review"
        self.root.mkdir()
        self.pkg = build_pkg(self.root)
        build_pkg(self.root, "pred_ps_20261112_abcdef")
        build_pkg(self.root, "other_pkg")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def serve(self, **kw):
        srv = mrs.make_server(self.root, "127.0.0.1", 0, people=[], **kw)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def stop(self, srv):
        srv.shutdown()
        srv.server_close()

    def test_ids_and_exclude(self):
        srv = self.serve(exclude=["LAB_compare_fixture"])
        try:
            self.assertEqual(srv.RequestHandlerClass.packages.ids(), ["pred_ps_20261112_abcdef"])
            self.assertEqual(request(srv.server_address[1], "GET", f"/{PKG}/api/ping")[0], 403)
        finally:
            self.stop(srv)
        os.environ[mrs.EXCLUDE_ENV] = "pred_ps_20261112_abcdef"
        srv = self.serve()
        try:
            self.assertEqual(srv.RequestHandlerClass.packages.ids(), [PKG])
        finally:
            self.stop(srv)

    def test_lock_released_on_close(self):
        srv = self.serve()
        port = srv.server_address[1]
        self.assertEqual(request(port, "GET", f"/{PKG}/api/ping")[0], 200)
        self.stop(srv)
        r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
                            "from annotation import review_server as rs; print(rs.hold_package(sys.argv[2]))",
                            str(REPO), str(self.pkg)], capture_output=True, text=True, env=clean_env(), timeout=60)
        self.assertEqual(r.stdout.strip(), "True", r.stderr)

    def test_import_keeps_env(self):
        os.environ["CRYOEV_VIA"] = "claude"
        importlib.reload(mrs)
        self.assertEqual(os.environ.get("CRYOEV_VIA"), "claude")
        os.environ.pop("CRYOEV_VIA")

    def test_excluded_names(self):
        self.assertEqual(mrs.excluded_names([str(self.pkg) + os.sep, " a "], {mrs.EXCLUDE_ENV: "B, ,c"}),
                         frozenset({PKG, "a", "b", "c"}))
        self.assertEqual(mrs.excluded_names([], {}), frozenset())

    def test_one_write_lock_per_package_in_a_process(self):
        h1 = rs.make_handler(self.pkg, KEY, set(), "x")
        self.assertIs(rs._write_lock(self.pkg / "edits"), rs._write_lock(Path(str(self.pkg).upper()) / "edits")
                      if os.name == "nt" else rs._write_lock(self.pkg / "edits"))
        self.assertTrue(h1)


if __name__ == "__main__":
    unittest.main()
