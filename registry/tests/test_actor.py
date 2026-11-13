"""registry/actor.py: who + via attribution ("claude for pujan"). Tests pass an explicit environ dict, so they behave
the same in a Claude session (CRYOEV_VIA set) and in the user's shell."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from registry import actor as A  # noqa: E402


class Actor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.people = Path(self.tmp.name) / "people.json"
        self.people.write_text(json.dumps({"people": [{"id": "pujan"}, {"id": "au1"}]}), "utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_via_unset_and_set(self):
        self.assertIsNone(A.via({}))
        self.assertIsNone(A.via({"CRYOEV_VIA": "  "}))
        self.assertEqual(A.via({"CRYOEV_VIA": "claude"}), "claude")

    def test_bad_via_is_an_error(self):
        for bad in ("Claude", "claude for pujan", "1bot", "x" * 40, "a/b"):
            with self.assertRaises(A.ActorError):
                A.via({"CRYOEV_VIA": bad})

    def test_who_precedence_flag_env_default(self):
        env = {"CRYOEV_FOR": "pujan"}
        self.assertEqual(A.who("au1", "x", env, self.people), "au1")        # explicit flag wins over the env
        self.assertEqual(A.who(None, "x", env, self.people), "pujan")      # then the env
        self.assertEqual(A.who(None, "x", {}, self.people), "x")           # then the old default
        with self.assertRaises(A.ActorError):
            A.who(None, "x", {"CRYOEV_FOR": "nobody"}, self.people)

    def test_current(self):
        self.assertEqual(A.current(None, "pujan", {}, self.people), {"who": "pujan"})
        self.assertEqual(A.current(None, None, {"CRYOEV_VIA": "claude", "CRYOEV_FOR": "pujan"}, self.people),
                         {"who": "pujan", "via": "claude"})

    def test_label(self):
        self.assertEqual(A.label("pujan", "claude"), "claude for pujan")
        self.assertEqual(A.label("pujan", None), "pujan")
        self.assertEqual(A.label(None, "claude"), "claude")
        self.assertEqual(A.label(None, None), "")
        self.assertEqual(A.label_of({"status_by": "pujan", "via": "claude"}), "claude for pujan")
        self.assertEqual(A.label_of({"created_by": "pujan", "created_via": "claude"}), "claude for pujan")
        self.assertTrue(A.is_agent({"via": "claude"}))
        self.assertFalse(A.is_agent({"who": "pujan"}))

    def test_strip_env(self):
        env = {"CRYOEV_VIA": "claude", "CRYOEV_FOR": "pujan", "PATH": "x"}
        self.assertEqual(sorted(A.strip_env(env)), ["CRYOEV_FOR", "CRYOEV_VIA"])
        self.assertEqual(env, {"PATH": "x"})
        self.assertEqual(A.strip_env(env), [])

    def test_strip_env_default_is_os_environ(self):
        with mock.patch.dict(os.environ, {"CRYOEV_VIA": "claude"}, clear=False):
            self.assertEqual(A.strip_env(), ["CRYOEV_VIA"])
            self.assertNotIn("CRYOEV_VIA", os.environ)

    def test_real_people_json_has_pujan(self):
        self.assertEqual(A.who(None, None, {"CRYOEV_FOR": "pujan"}), "pujan")


if __name__ == "__main__":
    unittest.main()
