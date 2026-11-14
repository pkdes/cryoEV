"""python -m cryoev schema register (round 14, contract C2): dry run writes nothing; --do writes the definition once
and one label_schema/1 line; a second register and an invalid schema are refused. Temp platform only."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cli_fixture as F  # noqa: E402

LAM = F.REPO / "labels" / "schemas" / "ls_lamellarity_v1.json"


class SchemaRegister(F.Platform):
    def test_dry_run_writes_nothing(self):
        before = self.snapshot()
        rc, out, err = self.cli("schema", "register", LAM)
        self.assertEqual(rc, 0, err)
        self.assertIn("ls_lamellarity_v1 (v1, maker pujan): valid", out)
        self.assertIn("DRY RUN", out)
        self.assertEqual(self.snapshot(), before)

    def test_do_once_then_refused(self):
        rc, out, err = self.cli("schema", "register", LAM, "--do", "--who", "pujan")
        self.assertEqual(rc, 0, err + out)
        line = [json.loads(x) for x in (self.reg / "label_schema.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["id"] for r in line], ["ls_lamellarity_v1"])
        self.assertEqual(line[0]["def"], "defs/label_schema/ls_lamellarity_v1.json")
        written = json.loads((self.reg / line[0]["def"]).read_text(encoding="utf-8"))
        self.assertEqual(written, json.loads(LAM.read_text(encoding="utf-8")))
        acts = [a for a in self.action_log() if a["action"] == "label schema registered"]
        self.assertEqual([a["detail"]["phase"] for a in acts], ["requested", "result"])
        from labels import schemas as S
        self.assertEqual(S.load("ls_lamellarity_v1", self.reg)["id"], "ls_lamellarity_v1")
        before = self.snapshot()
        rc, out, err = self.cli("schema", "register", LAM)
        self.assertEqual(rc, 2)
        self.assertIn("already registered (same content)", out)
        self.assertEqual(self.snapshot(), before)

    def test_invalid_refused(self):
        doc = json.loads(LAM.read_text(encoding="utf-8"))
        doc["object_attrs"][3]["key"] = "i"                      # reserved by the object card
        bad = self.tmp / "ls_bad_v1.json"
        bad.write_text(json.dumps(doc), encoding="utf-8")
        before = self.snapshot()
        rc, out, err = self.cli("schema", "register", bad, "--do", "--who", "pujan")
        self.assertEqual(rc, 2)
        self.assertIn("reserved", err)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
