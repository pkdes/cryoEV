"""tests/cryoev_testkit.py (round 11 #4): clean_env / strip_environ / env_patch and write_registry, known answers."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cryoev_testkit as TK  # noqa: E402

BASE = {"CRYOEV_VIA": "claude", "cryoev_for": "pujan", "CRYOEV_LAPTOP_ROOT": "x", "CJ_API": "off", "CJ_RAW_ROOT": "r",
        "PATH": "p"}


class Env(unittest.TestCase):
    def test_default_strips_the_actor_vars_only(self):
        self.assertEqual(TK.clean_env(BASE), {"CRYOEV_LAPTOP_ROOT": "x", "CJ_API": "off", "CJ_RAW_ROOT": "r",
                                              "PATH": "p"})

    def test_strip_all_drop_and_extra(self):
        self.assertEqual(TK.clean_env(BASE, strip_all=True), {"CJ_API": "off", "CJ_RAW_ROOT": "r", "PATH": "p"})
        self.assertEqual(TK.clean_env(BASE, strip_all=True, drop=("cj_api",)), {"CJ_RAW_ROOT": "r", "PATH": "p"})
        self.assertEqual(TK.clean_env(BASE, strip_all=True, drop=("CJ_*",), CRYOEV_REGISTRY="g"),
                         {"PATH": "p", "CRYOEV_REGISTRY": "g"})

    def test_env_patch_restores_and_strip_environ_reports(self):
        with TK.env_patch(CRYOEV_VIA="claude", CRYOEV_X="1"):           # extra is set after the strip
            with TK.env_patch(strip_all=True, A_T="1"):
                self.assertNotIn("CRYOEV_X", os.environ)
                self.assertEqual(os.environ["A_T"], "1")
            self.assertEqual(os.environ["CRYOEV_X"], "1")
            self.assertNotIn("A_T", os.environ)
            self.assertEqual(TK.strip_environ(), {"CRYOEV_VIA": "claude"})
            self.assertEqual(os.environ["CRYOEV_X"], "1")


class WriteRegistry(unittest.TestCase):
    def test_records_and_samples(self):
        with tempfile.TemporaryDirectory() as t:
            root = TK.write_registry(Path(t) / "reg", {"job": [{"id": "a"}, {"id": "b"}]})
            self.assertEqual([json.loads(x) for x in (root / "job.jsonl").read_text(encoding="utf-8").splitlines()],
                             [{"id": "a"}, {"id": "b"}])
            TK.write_registry(root, {"job": [{"id": "c"}]})                    # rewritten whole without samples
            self.assertEqual((root / "job.jsonl").read_text(encoding="utf-8").splitlines(), ['{"id": "c"}'])
            s = TK.write_registry(Path(t) / "s", {"job": [{"id": "z"}]}, samples=True)
            names = {p.name for p in s.glob("*.jsonl")}
            self.assertEqual(names, {p.name for p in TK.SAMPLES.glob("*.jsonl")} - TK.NOT_REGISTRY)
            sample_job = (TK.SAMPLES / "job.jsonl").read_bytes()
            self.assertTrue((s / "job.jsonl").read_bytes().startswith(sample_job))
            self.assertEqual((s / "job.jsonl").read_text(encoding="utf-8").splitlines()[-1], '{"id": "z"}')


if __name__ == "__main__":
    unittest.main()
