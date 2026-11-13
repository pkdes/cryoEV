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
    def test_dict_records_rewrite_without_samples(self):
        with tempfile.TemporaryDirectory() as t:
            root = TK.write_registry(Path(t) / "reg", {"job": [{"id": "a"}, {"id": "b"}]}, samples=False)
            self.assertEqual([json.loads(x) for x in (root / "job.jsonl").read_text(encoding="utf-8").splitlines()],
                             [{"id": "a"}, {"id": "b"}])
            TK.write_registry(root, {"job": [{"id": "c"}]}, samples=False)        # rewritten whole
            self.assertEqual((root / "job.jsonl").read_bytes(), b'{"id": "c"}' + bytes([10]))   # LF only
            TK.write_registry(root, {"job": [{"id": "é"}]}, samples=False)        # ensure_ascii=False
            self.assertEqual((root / "job.jsonl").read_text(encoding="utf-8"), '{"id": "é"}' + chr(10))

    def test_list_records_with_samples_and_skip(self):
        with tempfile.TemporaryDirectory() as t:
            recs = [{"schema": "job/1", "id": "z"}, {"schema": "newent/1", "id": "n"}]
            s = TK.write_registry(Path(t) / "s", recs, samples=True)
            self.assertEqual({p.name for p in s.glob("*.jsonl")},
                             {p.name for p in TK.SAMPLES.glob("*.jsonl")} | {"newent.jsonl"})
            sample = (TK.SAMPLES / "job.jsonl").read_text(encoding="utf-8").splitlines()
            got = (s / "job.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(got, sample + ['{"schema": "job/1", "id": "z"}'])
            self.assertNotIn(bytes([13]), (s / "job.jsonl").read_bytes())
            k = TK.write_registry(Path(t) / "k", None, samples=True, skip=("label_change.jsonl",))
            self.assertFalse((k / "label_change.jsonl").exists())

    def test_samples_is_required(self):
        with tempfile.TemporaryDirectory() as t, self.assertRaises(TypeError):
            TK.write_registry(Path(t), {"job": []})


if __name__ == "__main__":
    unittest.main()
