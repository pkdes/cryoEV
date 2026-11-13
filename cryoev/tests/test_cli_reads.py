"""cryoev reads: status, find, show (KNOWN-ANSWER control against registry/registry.py show), lineage, and the
blind-audit rule of `review progress` (counts only; `_work_*` and decision contents refused while open)."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cli_fixture as F  # noqa: E402


class Reads(F.Platform):
    def test_show_known_answer(self):
        """`python -m cryoev show <id> --json` == the record `registry/registry.py show <id>` prints (both run as
        the commands an agent types, on the same fixture registry)."""
        env = dict(os.environ)
        for rid in (F.PS_ID, "v5_evnonev_y8m_20261029", "lv_20261030_0846"):
            with self.subTest(rid=rid):
                a = subprocess.run([sys.executable, "-m", "cryoev", "show", rid, "--json"], cwd=F.REPO, env=env,
                                   capture_output=True, text=True, encoding="utf-8")
                b = subprocess.run([sys.executable, "registry/registry.py", "show", rid], cwd=F.REPO, env=env,
                                   capture_output=True, text=True, encoding="utf-8")
                self.assertEqual((a.returncode, b.returncode), (0, 0), a.stderr + b.stderr)
                rec_b = json.loads(b.stdout.rstrip("\n").rsplit("\n", 1)[0])     # minus "(N line(s); ...)"
                self.assertEqual(json.loads(a.stdout), rec_b)
                self.assertEqual(rec_b["id"], rid)
        a = subprocess.run([sys.executable, "-m", "cryoev", "show", "no_such_id", "--json"], cwd=F.REPO, env=env,
                           capture_output=True, text=True)
        self.assertEqual(a.returncode, 1)

    def test_find(self):
        rc, out, err = self.cli("find", "v5_evnonev", "--json")
        self.assertEqual(rc, 0, err)
        ids = {m["id"] for m in json.loads(out)["matches"]}
        self.assertIn("v5_evnonev_y8m_20261029", ids)
        rc, out, err = self.cli("find", "ps_", "--entity", "prediction_set")
        self.assertIn(F.PS_ID, out)

    def test_status(self):
        rc, out, err = self.cli("status", "--json")
        self.assertEqual(rc, 0, err)
        s = json.loads(out)
        self.assertEqual(s["services"]["rows"], [{"name": "demo", "enabled": False, "state": "disabled", "port": None,
                                                  "health": "-"}])
        self.assertEqual(s["feedback"]["open"], 1)
        pkg = [r for r in s["review_packages"]["rows"] if r["id"] == F.PKG]
        self.assertEqual(pkg, [{"id": F.PKG, "kind": "review", "status": "in_progress", "cards": 4, "decided": 1,
                                "images": 2, "reviewed": 0}])
        self.assertFalse(any(r["id"].startswith("_work") for r in s["review_packages"]["rows"]))
        self.assertNotIn(F.REVIEWER, out)                          # counts only, never who decided
        self.assertIn("live.json", s["desktop_jobs"]["problem"])   # no monitor snapshot in the fixture
        rc, out, err = self.cli("status", "--share", self.share, "--json")
        self.assertEqual(json.loads(out)["desktop_jobs"]["queues"][0]["jobs"], [])
        rc, out, err = self.cli("status")
        self.assertEqual(rc, 0, err)
        self.assertIn("cards 1/4", out)

    def test_lineage(self):
        rc, out, err = self.cli("lineage", F.MODEL, "--json")
        self.assertEqual(rc, 0, err)
        t = json.loads(out)
        self.assertEqual((t["id"], t["root"]["type"]), (F.MODEL, "model"))
        self.assertEqual(t["root"]["parents"][0]["type"], "run")
        self.assertNotIn("thumbs", json.dumps(t["root"]))
        rc, out, err = self.cli("lineage", "no_such_id")
        self.assertEqual(rc, 1)


class ReviewProgress(F.Platform):
    def test_counts_only(self):
        rc, out, err = self.cli("review", "progress", F.PKG, "--json")
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(out), {"package": F.PKG, "status": "in_progress", "cards": 4, "decided": 1,
                                           "images": 2, "reviewed": 0, "applied_by": None})
        self.assertNotIn(F.REVIEWER, out + err)
        self.assertNotIn("keep", out + err)

    def test_blind_audit_refusals(self):
        for argv in (["review", "progress", f"_work_{F.PKG}"], ["review", "progress", f"{F.PKG}/../_work_{F.PKG}"],
                     ["review", "progress", F.PKG, "--decisions"]):
            with self.subTest(argv=argv):
                rc, out, err = self.cli(*argv)
                self.assertEqual(rc, 3, err)
                self.assertNotIn("decoys", out)
                self.assertNotIn(F.REVIEWER, out + err)
        rc, out, err = self.cli("review", "progress", str(self.lap / "review" / F.PKG))   # a path, not a name
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
