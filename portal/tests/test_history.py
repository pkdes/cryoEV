"""
History through the shell: allowlist, include injection (file on disk untouched), live.json fresh, and logs without
keys. history.html is built by registry/history_page.py from registry/samples into a temp registry folder.

  python -m unittest discover -s portal/tests -v
"""
import json
import logging
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import app as APP  # noqa: E402

from registry import history_page as hp  # noqa: E402


class History(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        reg = cls.env.registry
        hp.build([T.REPO / "registry" / "samples"], reg / "history.html")
        for name in ("model.jsonl", "job.jsonl", "live/monitor.lock", "lists/a.txt", "secret.txt"):
            (reg / name).parent.mkdir(parents=True, exist_ok=True)
            (reg / name).write_text("do not serve\n", encoding="utf-8")
        cls.env.write_live(T.live_doc())
        cls.up = T.Upstream()
        cls.p = T.Portal(cls.env.config(review_upstream=cls.up.url))
        cls.token = T.login(cls.p.port, cls.env.key)

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.up.stop()
        cls.env.cleanup()

    def get(self, path):
        return T.request(self.p.port, "GET", path, T.cookie(self.token))

    def test_history_html_served_with_includes_file_untouched(self):
        path = self.env.registry / "history.html"
        disk = path.read_bytes()
        before = T.md5(disk)
        st, h, body = self.get("/history/history.html")
        self.assertEqual(st, 200)
        self.assertEqual(T.md5(path.read_bytes()), before)                  # on disk: unchanged
        self.assertNotIn(b"/cj/header.js", disk)
        self.assertEqual(body.count(APP.INCLUDES), 1)
        i = body.index(APP.INCLUDES)
        self.assertEqual(body[i + len(APP.INCLUDES):i + len(APP.INCLUDES) + 7].lower(), b"</head>")
        self.assertEqual(body.replace(APP.INCLUDES, b"", 1), disk)           # nothing else changed
        self.assertEqual(int(T.hget(h, "content-length")), len(body))      # recomputed
        self.assertTrue(T.hget(h, "content-type").startswith("text/html"))
        self.assertIn(b'<link rel="stylesheet" href="/cj/cj.css">', body)
        self.assertIn(b'<script src="/cj/header.js" defer></script>', body)
        st, _, _ = self.get("/history/history.html")                         # served twice: still untouched
        self.assertEqual(T.md5(path.read_bytes()), before)

    def test_allowlist_only(self):
        st, _, body = self.get("/history/live/live.json")
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(body)["schema"], "live/1")
        for p in ("/history/model.jsonl", "/history/job.jsonl", "/history/live/monitor.lock", "/history/lists/a.txt",
                  "/history/secret.txt", "/history/live/", "/history/lists/", "/history/live", "/history/History.html",
                  "/history/history.html/", "/history/history.html.bak", "/history/%2e%2e/registry/model.jsonl",
                  "/history/live/../model.jsonl", "/history/./history.html", "/history/index.html"):
            st, _, body = self.get(p)
            self.assertIn(st, (400, 404), p)
            self.assertNotIn(b"do not serve", body, p)
        st, h, _ = self.get("/history/")                                     # the folder: a redirect, no listing
        self.assertEqual((st, T.hget(h, "location")), (307, "history.html"))

    def test_live_json_fresh(self):
        self.env.write_live({"schema": "live/1", "n": 1})
        st, h, body = self.get("/history/live/live.json")
        self.assertEqual((st, json.loads(body)["n"]), (200, 1))
        self.assertEqual(T.hget(h, "cache-control"), "no-store")
        self.env.write_live({"schema": "live/1", "n": 2})
        st, _, body = self.get("/history/live/live.json")
        self.assertEqual(json.loads(body)["n"], 2)
        self.env.write_live(T.live_doc())

    def test_missing_files_404(self):
        env = T.Env()
        p = T.Portal(env.config(registry=env.tmp / "no_registry"))
        try:
            tok = T.login(p.port, env.key)
            self.assertEqual(T.request(p.port, "GET", "/history/history.html", T.cookie(tok))[0], 404)
            self.assertEqual(T.request(p.port, "GET", "/history/live/live.json", T.cookie(tok))[0], 404)
            st, _, body = T.request(p.port, "GET", "/", T.cookie(tok))      # home still renders
            self.assertEqual(st, 200)
            self.assertIn(b"No live snapshot", body)
        finally:
            p.stop()
            env.cleanup()


class Logs(unittest.TestCase):
    """Every log record while keys travel in query strings: no key, no session id; `k=***` shows it was redacted."""

    def test_no_key_in_logs(self):
        env = T.Env()
        up = T.Upstream()
        records = []

        class Grab(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())
        grab = Grab(level=logging.DEBUG)
        root = logging.getLogger()
        old = root.level
        root.addHandler(grab)
        root.setLevel(logging.DEBUG)
        p = T.Portal(env.config(review_upstream=up.url))
        try:
            token = T.login(p.port, env.key)
            T.request(p.port, "GET", "/login?k=wrong-key-zz9", {})
            T.request(p.port, "GET", "/review/echo?f=1&k=stale-key-qq7", T.cookie(token))
            T.request(p.port, "GET", "/?k=home-key-xx1", T.cookie(token))
            T.request(p.port, "GET", "/review/echo?%6B=enc-key-ww3", T.cookie(token))
            time.sleep(0.2)
        finally:
            p.stop()
            up.stop()
            root.removeHandler(grab)
            root.setLevel(old)
            env.cleanup()
        text = "\n".join(records)
        self.assertTrue(any("cj.access" in r or "/login" in r for r in records), records)   # control: logging ran
        self.assertIn("/login?k=***", text)
        for secret in (env.key, token, "wrong-key-zz9", "stale-key-qq7", "home-key-xx1", "enc-key-ww3"):
            self.assertNotIn(secret, text)
        self.assertEqual(APP.redact_query("a=1&k=x&b=2&%6b=y"), "a=1&k=***&b=2&k=***")

    def test_main_runs_uvicorn_without_access_log_or_proxy_headers(self):
        env = T.Env()
        try:
            with mock.patch("uvicorn.run") as run, mock.patch("logging.basicConfig"), \
                    mock.patch("builtins.print"):
                APP.main(["--registry", str(env.registry), "--roster", str(env.roster), "--state-dir",
                          str(env.state), "--review-package", str(env.tmp / "none")])
            kw = run.call_args.kwargs
            self.assertEqual((kw["host"], kw["port"]), ("127.0.0.1", 8770))
            self.assertIs(kw["access_log"], False)
            self.assertIs(kw["proxy_headers"], False)
        finally:
            env.cleanup()

    def test_startup_lists_steps_without_secrets(self):
        env = T.Env()
        try:
            cfg = env.config()
            lines = "\n".join(APP.startup_lines(cfg, "127.0.0.1", 8770))
            for word in ("deny by default", "/healthz", "Origin check", "sandbox: not proxied", "k= redacted"):
                self.assertIn(word, lines)
            self.assertNotIn(T.UPKEY, lines)
            self.assertNotIn(env.key, lines)
        finally:
            env.cleanup()


if __name__ == "__main__":
    unittest.main()
