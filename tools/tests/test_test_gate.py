"""tools/test_gate.py: which suites a change runs (known answers), and that a run reports pass/fail."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_gate as G  # noqa: E402


class Affected(unittest.TestCase):
    def test_known_answers(self):
        self.assertEqual(G.affected_suites(["portal/modules/gallery.py"])[0], {"tests", "portal"})
        self.assertEqual(G.affected_suites(["docs/TOOLS.md", "README.md"])[0], {"tests"})
        self.assertEqual(G.affected_suites(["results/compute.py"])[0], {"tests", "results", "portal"})
        self.assertEqual(G.affected_suites(["jobs/convert_raw.py"])[0], {"tests", "jobs", "portal", "tools"})

    def test_shared_code_and_unknown_files_run_everything(self):
        self.assertEqual(G.affected_suites(["data_utils/oslock.py"])[0], set(G.SUITES))
        suites, unknown = G.affected_suites(["setup.cfg"])
        self.assertEqual(suites, set(G.SUITES))
        self.assertEqual(unknown, ["setup.cfg"])

    def test_every_suite_dir_exists(self):
        for name, d in G.SUITES.items():
            self.assertTrue((G.REPO / d).is_dir(), name)


class Run(unittest.TestCase):
    def test_a_real_unit_passes_and_counts_its_tests(self):
        r = G.run_unit("results", "test_p2.py", timeout=300)
        self.assertTrue(r["ok"], r["output"][-500:])
        self.assertGreater(r["tests"], 0)


if __name__ == "__main__":
    unittest.main()
