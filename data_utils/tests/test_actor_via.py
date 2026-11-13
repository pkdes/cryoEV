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


if __name__ == "__main__":
    unittest.main()
