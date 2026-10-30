"""registry.py / importers.py: append/current/history, refusals, guard, production rule, idempotent import."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REG))
import importers  # noqa: E402
import registry  # noqa: E402


def sample(entity):
    return json.loads((REG / "samples" / f"{entity}.jsonl").read_text(encoding="utf-8").splitlines()[0])


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "registry"

    def tearDown(self):
        self.tmp.cleanup()

    def test_append_current_history(self):
        run = sample("run")
        registry.append(run, root=self.root)
        registry.append(dict(run, status="failed", logged="2026-10-30T13:00:00"), root=self.root)
        registry.append(sample("pull"), root=self.root)
        cur = registry.current(root=self.root)
        self.assertEqual(cur[run["id"]]["status"], "failed")            # last line per id wins
        self.assertEqual([h["status"] for h in registry.history(run["id"], root=self.root)], ["finished", "failed"])
        self.assertEqual(set(registry.current("pull", root=self.root)), {sample("pull")["id"]})
        self.assertTrue((self.root / "run.jsonl").exists() and (self.root / "pull.jsonl").exists())

    def test_missing_trailing_newline_is_fixed(self):
        registry.append(sample("run"), root=self.root)
        p = self.root / "run.jsonl"
        p.write_bytes(p.read_bytes().rstrip(b"\n") + b'{"schema": "run/1", "id": "cut')   # line cut short
        registry.append(dict(sample("run"), status="failed"), root=self.root)
        self.assertEqual(registry.current(root=self.root)[sample("run")["id"]]["status"], "failed")

    def test_invalid_record_refused(self):
        bad = dict(sample("run"))
        del bad["dataset"]
        for rec in (bad, dict(sample("run"), status="nonsense"), {"schema": "nope/1", "id": "x"},
                    dict(sample("image"), session="20261012")):
            with self.assertRaises(registry.RegistryError):
                registry.append(rec, root=self.root)
        self.assertFalse((self.root / "run.jsonl").exists())

    def test_guard(self):
        for root in (Path.home() / "Box" / "registry", Path.home() / "Desktop" / "Cryo-image-processing" / "reg",
                     "\\\\localhost\\C$\\x\\registry"):
            with self.assertRaises(registry.RegistryError):
                registry.append(sample("run"), root=root)
        self.assertFalse((Path.home() / "Box" / "registry").exists())

    def test_production_rule(self):
        m = dict(sample("model"), status="registered")
        registry.append(m, root=self.root)
        with self.assertRaises(registry.RegistryError):                 # not via set-status
            registry.append(dict(m, status="production", status_by="pujan"), root=self.root)
        with self.assertRaises(registry.RegistryError):                 # wrong person
            registry.set_status(m["id"], "production", "au1", root=self.root)
        with self.assertRaises(registry.RegistryError):                 # not on the roster
            registry.set_status(m["id"], "candidate", "nobody", root=self.root)
        registry.set_status(m["id"], "production", "pujan", root=self.root)
        cur = registry.current(root=self.root)[m["id"]]
        self.assertEqual((cur["status"], cur["status_by"]), ("production", "pujan"))
        registry.append(dict(cur, note="re-append keeps production"), root=self.root)   # already production: ok

    def test_pointer_record_derives_from_owner(self):
        own = Path(self.tmp.name) / "batches.jsonl"
        ic = sample("inference_check")
        own.write_text(json.dumps({k: v for k, v in ic.items() if k not in ("schema", "id")}) + "\n", encoding="utf-8")
        line = registry.append({"schema": "inference_check/1", "id": ic["id"], "owner": f"{own}#{ic['id']}"},
                               root=self.root)
        self.assertNotIn("model", line)                                   # stored line is a pointer
        self.assertEqual(registry.current(root=self.root)[ic["id"]]["model"], ic["model"])
        with open(own, "a", encoding="utf-8") as f:                      # owner moves on: derived at read time
            f.write(json.dumps(dict({k: v for k, v in ic.items() if k not in ("schema", "id")}, status="scored")) + "\n")
        self.assertEqual(registry.current(root=self.root)[ic["id"]]["status"], "scored")
        registry.set_status(ic["id"], "reviewed", "pujan", root=self.root)  # explicit beats derived
        self.assertEqual(registry.current(root=self.root)[ic["id"]]["status"], "reviewed")


@unittest.skipUnless(importers.find_sibling("CryoAI-laptop"), "CryoAI-laptop not found")
class TestImportRerun(unittest.TestCase):
    def test_rerun_appends_nothing(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "registry"
            first = importers.run(root)
            self.assertGreater(sum(first["appended"].values()), 0)
            sizes = {p.name: p.stat().st_size for p in root.glob("*.jsonl")}
            second = importers.run(root)
            self.assertEqual(sum(second["appended"].values()), 0, second["appended"])
            self.assertEqual(sizes, {p.name: p.stat().st_size for p in root.glob("*.jsonl")})
            cur = registry.current(root=root)
            self.assertEqual(cur["v5_evnonev_y8m_20261029"]["status"], "production")
            self.assertEqual(cur["lv_20261030_0846"]["status"], "superseded")
            self.assertEqual(cur["lv_20261030_0909"]["supersedes"], "lv_20261030_0846")
            self.assertFalse(any(str(k).startswith("SANDBOX") for k in cur))
            self.assertFalse(any("20261012" in json.dumps(v) for v in cur.values()))   # lipo: no records
            self.assertIn("allLayer_run14_20260713", cur["v0_alllayer_y8n_20260713"]["aliases"])


if __name__ == "__main__":
    unittest.main()
