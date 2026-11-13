"""
Shared helpers for the CryoEV portal tests: a temp roster / state dir / registry (never the real ones), the portal
under a real uvicorn in a thread, a dummy upstream that records what reaches it, and raw HTTP helpers.
"""
from __future__ import annotations

import hashlib
import http.client
import http.server
import json
import shutil
import socket
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "tests") not in sys.path:
    sys.path.append(str(REPO / "tests"))             # appended: never shadows a portal test module

import cryoev_testkit as TK  # noqa: E402  (shared helpers: free_port, env, fixture registry)
from cryoev_testkit import free_port  # noqa: E402,F401  (T.free_port, used by many portal tests)

# The portal runs with CRYOEV_VIA / CRYOEV_FOR stripped (portal.app main(), services.child_env). Tests start the
# app without main(), so drop them here too: results must not depend on whether the test run is an agent session.
# A test that needs them sets them explicitly (mock.patch.dict). Also the agent API's switches (round 10 #60): the
# kill switch and the shared token store. Tests use Env.tokens (a temp store) and set CJ_API explicitly where they
# test it.
TK.strip_environ(drop=("CJ_API", "CJ_API_WRITE", "CJ_AGENT_TOKENS"))

import uvicorn  # noqa: E402

from portal import app as APP  # noqa: E402
from portal import auth as A  # noqa: E402

UPKEY = "upstream-review-key-for-tests"
USER = "tester"


def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def live_doc(servers=(("review", True), ("sandbox", False), ("history", True)), age_s=5, queues=None) -> dict:
    """A live.json shaped like jobs/monitor.py's snapshot (schema live/1)."""
    t = datetime.now(timezone.utc).timestamp() - age_s
    return {"schema": "live/1", "generated": "", "interval_s": 30,
            "generated_utc": datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds"),
            "monitor": {"machine": "home-laptop", "pid": 1},
            "servers": [{"name": n, "port": 1, "up": up, "ms": 3 if up else None} for n, up in servers],
            "queues": queues if queues is not None else [
                {"share": "\\\\gpu-desktop\\cj", "reach": "ok", "detail": None,
                 "watcher": {"display": "alive", "state": "running", "detail": None, "job": "job_20261105_demo",
                             "age_s": 40, "stale": False, "machine": "gpu-desktop"},
                 "jobs": [], "running": {"job": "job_20261105_demo", "phase": "training", "epoch": 12, "epochs": 300,
                                         "eta_s": 7380, "elapsed_s": 900}, "finished": []}]}


class Env:
    """Temp roster (one admin, USER), state dir, registry folder and agent token store."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_portal_"))
        self.roster = self.tmp / "roster.json"
        self.state = self.tmp / "state"
        self.registry = self.tmp / "registry"
        self.tokens = self.tmp / "agent_tokens.json"
        (self.registry / "live").mkdir(parents=True)
        self.key = A.add_key(self.roster, USER, "admin")

    def config(self, **kw) -> APP.Config:
        base = dict(registry=self.registry, roster=self.roster, state_dir=self.state,
                    review_upstream=f"http://127.0.0.1:{free_port()}", review_key=UPKEY, tokens=self.tokens)
        base.update(kw)
        return APP.Config(**base)

    def write_live(self, doc):
        (self.registry / "live" / "live.json").write_text(json.dumps(doc), encoding="utf-8")

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class Portal:
    """create_app(cfg) under uvicorn in a thread, configured as main() runs it (access log and proxy headers off).
    log_config=None so uvicorn leaves logging alone and the tests can capture every record."""

    def __init__(self, cfg: APP.Config):
        self.app = APP.create_app(cfg)
        self.port = free_port()
        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port, access_log=False,
                                                    proxy_headers=False, server_header=False, log_config=None,
                                                    log_level="warning", lifespan="on"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(200):
            if self.server.started:
                break
            time.sleep(0.025)
        else:
            raise RuntimeError("portal did not start")
        self.origin = f"http://127.0.0.1:{self.port}"

    def stop(self):
        self.server.should_exit = True
        self.thread.join(10)


def request(port, method, path, headers=None, body=None, timeout=15):
    """-> (status, [(name, value)], body). http.client sends `path` as given (no normalisation)."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        c.request(method, path, body=body, headers=dict(headers or {}))
        r = c.getresponse()
        return r.status, r.getheaders(), r.read()
    finally:
        c.close()


def hget(headers, name):
    vals = [v for k, v in headers if k.lower() == name.lower()]
    return vals[0] if vals else None


def hall(headers, name):
    return [v for k, v in headers if k.lower() == name.lower()]


def raw(port, data: bytes, timeout=15) -> bytes:
    """Send bytes as they are (for targets http.client refuses: NUL, backslash) and read the reply."""
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    try:
        s.sendall(data)
        out = b""
        while True:
            try:
                b = s.recv(65536)
            except (ConnectionResetError, ConnectionAbortedError, socket.timeout):
                break
            if not b:
                break
            out += b
            if b"\r\n\r\n" in out and b"content-length: 0" in out.lower():
                break
        return out
    finally:
        s.close()


def raw_status(resp: bytes) -> int:
    try:
        return int(resp.split(b" ", 2)[1])
    except (IndexError, ValueError):
        return 0


def login(port, key) -> str:
    """The session token from a key link (asserts the redirect)."""
    st, h, _ = request(port, "GET", f"/login?k={key}")
    assert st == 303, st
    sc = hget(h, "set-cookie")
    return sc.split(";", 1)[0].split("=", 1)[1]


GATE_DETAIL_KEYS = {"status", "from", "to", "n", "via"}      # via: a bearer request's agent (round 10 #60)


def remove_file(path, wait_s=5.0) -> None:
    """Delete a file a test reset needs gone, retrying while the Gate's audit worker (another thread) still has it
    open: on Windows an open handle makes unlink fail with PermissionError for a moment."""
    end = time.monotonic() + wait_s
    while True:
        try:
            Path(path).unlink(missing_ok=True)
            return
        except PermissionError:
            if time.monotonic() > end:
                raise
            time.sleep(0.05)


def gate_line(rec) -> bool:
    """A line from the Gate's audit catch-all (portal/app.py, round 9 #17): a sign-in, or detail {"status": n} (plus
    a counter line's from / to / n). Module tests that count their own lines filter these out."""
    if not isinstance(rec, dict):
        return False
    d = rec.get("detail")
    return rec.get("action") == "signed in" or (isinstance(d, dict) and "status" in d and set(d) <= GATE_DETAIL_KEYS)


def module_lines(path) -> list:
    """The audit file's lines minus the Gate's (gate_line). A line the Gate's worker is still writing is skipped."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for x in p.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(x)
        except ValueError:
            continue
        if not gate_line(rec):
            out.append(rec)
    return out


def cookie(token) -> dict:
    return {"Cookie": f"{A.COOKIE}={token}"}


def bearer(token) -> dict:
    """The agent API's header (portal/app.py BEARER_ROUTES)."""
    return {"Authorization": f"Bearer {token}"}


class Upstream:
    """A dummy review server that records every request (method, path, headers, body md5) and answers:
      /range        1024 bytes, honours Range (206 + Content-Range)
      /big          5 MB of deterministic bytes
      /setcookie    Set-Cookie, Connection: close, x-hop (named in Connection), Keep-Alive, X-Keep
      /redir-abs    302 to http://127.0.0.1:<own port>/x;  /redir-rel  302 to /y
      /slow         sleeps 3 s
      anything else 200 JSON echo of the request"""

    DATA = bytes(range(256)) * 4
    BIG = (b"0123456789abcdef" * 65536) * 5

    def __init__(self):
        self.seen = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _rec(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                rec = {"method": self.command, "path": self.path, "headers": list(self.headers.items()),
                       "body_len": len(body), "body_md5": md5(body)}
                outer.seen.append(rec)
                return rec

            def _send(self, code, body=b"", headers=()):
                self.send_response(code)
                for k, v in headers:
                    self.send_header(k, v)
                if not any(k.lower() == "content-length" for k, _ in headers):
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def handle_any(self):
                rec = self._rec()
                p = self.path.split("?", 1)[0]
                if p == "/range":
                    rng = self.headers.get("Range")
                    if rng and rng.startswith("bytes="):
                        a, b = rng[6:].split("-")
                        a, b = int(a), int(b)
                        return self._send(206, outer.DATA[a:b + 1], [("Content-Range", f"bytes {a}-{b}/1024"),
                                                                    ("Accept-Ranges", "bytes")])
                    return self._send(200, outer.DATA, [("Accept-Ranges", "bytes"),
                                                        ("Content-Type", "application/octet-stream")])
                if p == "/big":
                    return self._send(200, outer.BIG, [("Content-Type", "application/octet-stream")])
                if p == "/setcookie":
                    return self._send(200, b"ok", [("Set-Cookie", "review_key_8766=secret; Path=/"),
                                                   ("Set-Cookie", "other=1"), ("Connection", "x-hop"),
                                                   ("X-Hop", "1"), ("Keep-Alive", "timeout=5"), ("X-Keep", "1")])
                if p == "/redir-abs":
                    return self._send(302, b"", [("Location", f"http://127.0.0.1:{outer.port}/x")])
                if p == "/redir-rel":
                    return self._send(302, b"", [("Location", "/y")])
                if p == "/slow":
                    time.sleep(3)
                body = json.dumps(rec).encode()
                return self._send(200, body, [("Content-Type", "application/json")])

            do_GET = do_HEAD = do_PUT = do_POST = do_DELETE = do_PATCH = handle_any

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.srv.handle_error = lambda *a: None          # the portal hanging up on /slow is expected
        self.port = self.srv.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def last(self):
        return self.seen[-1] if self.seen else None

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()


class ReviewUpstream:
    """The real annotation/review_server.py handler on a temp copy of the test fixture package, key in .review_key
    (as the real package keeps it)."""

    def __init__(self, key="fixture-review-key"):
        sys.path.insert(0, str(REPO / "annotation"))
        sys.path.insert(0, str(REPO / "annotation" / "tests"))
        import make_fixture
        import review_server as rs
        self.dir = Path(tempfile.mkdtemp(prefix="cj_review_"))
        self.pkg = self.dir / "pkg"
        make_fixture.build(self.pkg)
        (self.pkg / ".review_key").write_text(key, encoding="utf-8")
        self.key = key
        data = json.loads((self.pkg / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
        self.files = [im["file"] for im in data["images"]]
        handler = rs.make_handler(self.pkg, key, set(self.files), data.get("id", ""), rs.load_people(), False)
        handler.log_message = lambda *a: None
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()
        shutil.rmtree(self.dir, ignore_errors=True)
