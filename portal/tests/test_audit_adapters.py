"""
Round 11 #3: the admin / jobs / train `audit` wrappers are thin adapters over portal/audit.py. Same names and
positional signatures (existing callers and monkeypatches, e.g. J.audit and TR.PA.audit, keep working); with a request
they go through PA.event (a bearer request's agent rides along as detail.via); without one through PA.audit; and both
raise OSError when the line is not written (the write-ahead callers rely on it). Temp state only.
python -m unittest portal.tests.test_audit_adapters -v
"""
import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from portal import audit as PA  # noqa: E402
from portal.modules import admin as ADM  # noqa: E402
from portal.modules import jobs as J  # noqa: E402
from portal.modules import train as TR  # noqa: E402

BEARER = {"cj_user": {"id": "ann"}, "cj_auth": "bearer", "cj_via": "claude"}
BROWSER = {"cj_user": {"id": "ann"}, "cj_auth": "key"}
P = {"dataset": "d", "base": "b", "preset": "p", "where": "gpu", "epochs": 1, "imgsz": 640, "seed": 0, "max_hours": 1}


class Adapters(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = SimpleNamespace(state_dir=Path(self.tmp.name))

    def lines(self):
        f = PA.audit_path(self.cfg.state_dir)
        return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []

    def calls(self, request):
        """One call per adapter, in their existing positional forms (+ request)."""
        return [lambda: ADM.audit(SimpleNamespace(cfg=self.cfg), "ann", "role_changed", "au", {"to": "basic"},
                                  request=request),
                lambda: J.audit(self.cfg, "ann", "cancel", "job_1", "why", "requested", request=request),
                lambda: TR.audit(self.cfg, "ann", "job_2", P, "requested", None, "gpu_queue", request=request)]

    def test_without_request_unchanged(self):
        rec = ADM.audit(SimpleNamespace(cfg=self.cfg), "ann", "role_changed", "au", {"to": "basic"})
        self.assertEqual((rec["actor"], rec["target"], rec["detail"]), ("ann", "au", {"to": "basic"}))   # returned
        J.audit(self.cfg, "ann", "cancel", "job_1", "why", "requested")
        TR.audit(self.cfg, "ann", "job_2", P, "requested", None, "gpu_queue")
        got = self.lines()
        self.assertEqual([(x["actor"], x["action"], x["target"]) for x in got],
                         [("ann", "role_changed", "au"), ("ann", "cancel", "job_1"), ("ann", "train_queue", "job_2")])
        self.assertEqual(got[1]["detail"], {"reason": "why", "result": "requested", "phase": "requested"})
        self.assertEqual(got[2]["detail"]["phase"], "requested")
        self.assertFalse(any("via" in (x["detail"] or {}) for x in got))

    def test_with_request_via_event(self):
        for call in self.calls(BEARER):
            call()
        for call in self.calls(BROWSER):                      # control: a browser request carries no via
            call()
        got = self.lines()
        self.assertEqual([x["actor"] for x in got], ["ann"] * 6)
        self.assertEqual([x["detail"].get("via") for x in got], ["claude"] * 3 + [None] * 3)
        self.assertEqual(got[1]["detail"], {"reason": "why", "result": "requested", "phase": "requested",
                                            "via": "claude"})

    def test_unwritten_line_raises_both_ways(self):
        with mock.patch.object(PA, "audit", side_effect=OSError("disk full")):
            for request in (None, BEARER):
                for call in self.calls(request):
                    logs = self.assertLogs("cj.audit", "ERROR") if request else contextlib.nullcontext()
                    with self.subTest(request=bool(request)), logs, self.assertRaises(OSError):
                        call()
        self.assertEqual(self.lines(), [])


if __name__ == "__main__":
    unittest.main()
