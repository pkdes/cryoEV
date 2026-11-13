"""data_utils/actor.py VIA_RE: anchored at the very end (\\Z, not $), so a trailing newline is refused (round 10 QC)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_utils import actor  # noqa: E402


class ViaRe(unittest.TestCase):
    def test_good_names_pass(self):                       # control
        for v in ("claude", "codex", "a", "claude-code_2", "a" + "b" * 31):
            self.assertTrue(actor.VIA_RE.match(v), v)

    def test_trailing_newline_and_others_refused(self):
        for v in ("claude\n", "claude\r\n", "Claude", "9claude", "", "a" * 33, "claude x", "claude\t"):
            self.assertIsNone(actor.VIA_RE.match(v), repr(v))

    def test_env_value_with_an_inner_newline_is_refused(self):
        with self.assertRaises(actor.ActorError):
            actor.via({actor.ENV_VIA: "claude\nx"})
        self.assertEqual(actor.via({actor.ENV_VIA: "claude"}), "claude")


class OwnerVia(unittest.TestCase):
    """owner -> owner_via (round 11 #65): an agent-made prediction set shows "claude for X" on every later line."""

    def test_owner_pairs_with_owner_via_after_a_status_line(self):
        create = {"id": "ps_1", "owner": "alyssia", "owner_via": "claude", "via": "claude", "status": "queued"}
        later = {"id": "ps_1", "owner": "alyssia", "owner_via": "claude", "status": "done"}   # no line via
        for rec in (create, later):
            self.assertEqual(actor.label_of(rec, "owner"), "claude for alyssia")
            self.assertEqual(actor.via_of(rec, "owner"), "claude")
            self.assertTrue(actor.is_agent(rec))
        person = {"id": "ps_2", "owner": "alyssia", "status": "done"}             # control: a person's own set
        self.assertEqual(actor.label_of(person, "owner"), "alyssia")
        self.assertIsNone(actor.via_of(person, "owner"))
        self.assertFalse(actor.is_agent(person))
        self.assertEqual(actor.label_of({"owner": "a", "via": "codex"}, "owner"), "codex for a")   # older lines
        # the other pairs are unchanged
        self.assertEqual(actor.label_of({"status_by": "a", "status_via": "claude", "via": "x"}, "who"),
                         "claude for a")
        self.assertEqual(actor.label_of({"who": "a", "via": "claude", "owner_via": "zzz"}), "claude for a")


if __name__ == "__main__":
    unittest.main()
