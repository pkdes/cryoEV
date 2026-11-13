"""
Round 11 #3: portal.audit.event, the one audit helper for modules. The actor comes from the request (cj_user), a
bearer request's agent rides along as detail.via, and it never raises. Temp state only.
python -m unittest portal.tests.test_audit_event -v
"""
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from portal import audit as PA  # noqa: E402


class FakeRequest:
    def __init__(self, scope):
        self.scope = scope


def lines(state) -> list:
    f = PA.audit_path(state)
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()] if f.exists() else []


class Event(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)

    def test_browser_request_actor_and_no_via(self):
        req = FakeRequest({"cj_user": {"id": "alyssia"}, "cj_auth": "access"})
        self.assertTrue(PA.event(self.state, req, "feedback", "#3", {"x": 1}))
        (rec,) = lines(self.state)
        self.assertEqual((rec["actor"], rec["action"], rec["target"], rec["detail"]), ("alyssia", "feedback", "#3",
                                                                                      {"x": 1}))

    def test_bearer_request_carries_via(self):
        scope = {"cj_user": {"id": "alyssia"}, "cj_auth": "bearer", "cj_via": "claude"}
        self.assertTrue(PA.event(self.state, scope, "predict.create", "ps1"))
        (rec,) = lines(self.state)
        self.assertEqual(rec["detail"], {"via": "claude"})
        self.assertEqual(PA.normalise(rec)["detail"]["via"], "claude")

    def test_via_ignored_without_bearer(self):
        """A stray cj_via on a non-bearer request is never trusted (only the Gate's verified token sets bearer)."""
        PA.event(self.state, {"cj_user": {"id": "a"}, "cj_auth": "key", "cj_via": "claude"}, "x", None)
        self.assertIsNone(lines(self.state)[0]["detail"])

    def test_non_dict_detail_is_kept_with_via(self):
        scope = {"cj_user": {"id": "a"}, "cj_auth": "bearer", "cj_via": "bot"}
        PA.event(self.state, scope, "x", None, "plain")
        self.assertEqual(lines(self.state)[0]["detail"], {"detail": "plain", "via": "bot"})

    def test_explicit_actor_and_none_request(self):
        PA.event(self.state, None, "cli.thing", "t", actor="cli")
        self.assertEqual(lines(self.state)[0]["actor"], "cli")
        PA.event(self.state, None, "anon", "t")
        self.assertEqual(lines(self.state)[1]["actor"], "-")

    def test_never_raises(self):
        lg = logging.getLogger("test.audit.event")
        with self.assertLogs(lg, "ERROR"):
            self.assertFalse(PA.event(None, None, "x", "t", log_to=lg))
        with mock.patch.object(PA, "audit", side_effect=OSError("locked")), self.assertLogs("cj.audit", "ERROR"):
            self.assertFalse(PA.event(self.state, None, "x", "t"))
        self.assertEqual(lines(self.state), [])

    def test_caller_detail_not_mutated(self):
        d = {"a": 1}
        PA.event(self.state, {"cj_user": {"id": "a"}, "cj_auth": "bearer", "cj_via": "bot"}, "x", "t", d)
        self.assertEqual(d, {"a": 1})


if __name__ == "__main__":
    unittest.main()
