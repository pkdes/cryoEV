"""T4 (round 10): a label set that a newer one replaced links forward to it ("Replaced by")."""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from portal.modules import data as D  # noqa: E402


class ReplacedBy(unittest.TestCase):
    def test_reverse_of_supersedes(self):
        recs = {"label_version": {"lv_a": {"id": "lv_a", "status": "superseded"},
                                  "lv_b": {"id": "lv_b", "supersedes": "lv_a"},
                                  "lv_c": {"id": "lv_c", "supersedes": "lv_b"}}}
        self.assertEqual(D.replaced_by(recs, "lv_a"), ["lv_b"])
        self.assertEqual(D.replaced_by(recs, "lv_b"), ["lv_c"])
        self.assertEqual(D.replaced_by(recs, "lv_c"), [])            # control: the newest has no replacement
        self.assertEqual(D.replaced_by({}, "lv_a"), [])


if __name__ == "__main__":
    unittest.main()


class PreviewLink(unittest.TestCase):
    """V4 (round 10): the label-set page's "Waiting on it" rows link to the admin-only apply preview."""

    def test_link_rendered_only_when_set(self):
        row = {"id": "lab_x", "title": "Lab X", "n": 3, "url": None}
        html = D.render_used_by({"datasets": [], "waiting": [dict(row, preview="/apply/#p-lab_x")]})
        self.assertIn('href="/apply/#p-lab_x"', html)
        self.assertIn("Preview apply", html)
        html = D.render_used_by({"datasets": [], "waiting": [dict(row, preview=None)]})   # non-admin / compare / pred_
        self.assertNotIn("Preview apply", html)
