"""importers.py carries the portal's upload-session keys (`conversions`, `details`; portal/upload_sessions.py) forward
when it rebuilds a session record, in both orders, and holds the upload-session lock while it writes.

- Order 1: the portal made the session first (an upload of a date the lab had not imported); the importer then builds
  the lab record for that id: lab fields win, the upload keys stay; a rerun appends 0.
- Order 2: the importer first; the portal merges its keys into the lab line; a rerun appends 0 (nothing dropped); a
  changed lab value appends a line that still has the upload keys.
- Control: with the carry-forward switched off (UPLOAD_KEYS = ()) the same run drops the keys, so the checks above
  can see a loss.
- Every registry file is only appended to (its earlier bytes are a prefix of the later ones).
Temp folders only.
"""
import ast
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REG))
import importers  # noqa: E402
import registry  # noqa: E402

SID = "20991105"
CONV = {"id": f"{SID}_a1b2c3", "uploaded_by": "adv", "grids": [1, 2], "sample_labels": {"1": "Up A", "2": "Up B"},
        "tiles_per_grid": {"1": 2, "2": 1}, "n_tiles": 3, "acquisition": {"instrument": "Glacios",
                                                                          "magnification": [45000]}}
DETAILS = {"grid_ids": {"1": "2602"}, "sample_display": {"2": "Shown B"}, "prep_notes": "glow 25 mA\nblot 3 s",
           "by": "adv", "at": "2099-11-06T10:00:00"}


def lab_record(n_mdoc=4):
    return {"schema": "session/1", "id": SID, "date": "2099-11-05", "date_basis": "folder", "status": "processed",
            "local_folder": f"proc/{SID}", "sample_labels": {"1": "Lab A", "2": "Lab B"}, "instrument": "Glacios",
            "magnification": [11000, 45000], "n_mdoc": n_mdoc}


def portal_record():
    """What portal/upload_sessions.register_conversion + save_details append for a session the lab never imported."""
    return {"schema": "session/1", "id": SID, "date": "2099-11-05", "status": "processed",
            "instrument": "Glacios", "magnification": [45000], "sample_labels": {"1": "Up A", "2": "Up B"},
            "conversions": [CONV], "details": DETAILS, "recorded_by": "portal/upload_sessions.py details"}


class UploadKeys(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.root = t / "registry"
        for d in ("lap", "cai", "proc"):
            (t / d).mkdir()
        self.dirs = dict(laptop=t / "lap", cache=t / "cai", processing=t / "proc")

    def tearDown(self):
        self.tmp.cleanup()

    def importer_writes(self, rec):
        imp = importers.Importer(self.root, **self.dirs)
        imp.out = [("session", dict(rec))]
        return imp.write()

    def current(self):
        return registry.current("session", self.root)[SID]

    def snap(self):
        return {p.name: p.read_bytes() for p in self.root.glob("*.jsonl")}

    def assert_appended_only(self, before):
        after = self.snap()
        for name, data in before.items():
            self.assertTrue(after[name].startswith(data), f"{name} was rewritten, not appended to")

    def test_keys_match_the_portal(self):
        src = (REG.parent / "portal" / "upload_sessions.py").read_text(encoding="utf-8")
        keys = re.search(r"^UPLOAD_KEYS = (\([^)]*\))", src, re.M).group(1)
        self.assertEqual(ast.literal_eval(keys), importers.UPLOAD_KEYS)
        self.assertEqual(ast.literal_eval(re.search(r"^LOCK = (\"[^\"]+\")", src, re.M).group(1)),
                         importers.UPLOAD_LOCK)

    def test_portal_first_then_importer(self):
        registry.append(portal_record(), self.root)
        before = self.snap()
        res = self.importer_writes(lab_record())
        self.assertEqual(res["session"], 1)
        cur = self.current()
        self.assertEqual(cur["conversions"], [CONV])
        self.assertEqual(cur["details"], DETAILS)
        self.assertEqual(cur["sample_labels"], {"1": "Lab A", "2": "Lab B"})       # the lab's own fields win
        self.assertEqual((cur["local_folder"], cur["n_mdoc"]), (f"proc/{SID}", 4))
        self.assertEqual(self.importer_writes(lab_record())["session"], 0)        # rerun: nothing new
        self.assert_appended_only(before)

    def test_importer_first_then_portal_then_importer(self):
        self.assertEqual(self.importer_writes(lab_record())["session"], 1)
        lab = self.current()
        merged = {k: v for k, v in lab.items() if k != "logged"}
        merged.update(conversions=[CONV], details=DETAILS, recorded_by="portal/upload_sessions.py")
        registry.append(merged, self.root)                                        # the portal's merge-append
        before = self.snap()
        self.assertEqual(self.importer_writes(lab_record())["session"], 0)        # carried: equal, not re-appended
        self.assertEqual(self.current()["conversions"], [CONV])
        self.assertEqual(self.importer_writes(lab_record(n_mdoc=6))["session"], 1)   # a real lab change
        cur = self.current()
        self.assertEqual((cur["n_mdoc"], cur["conversions"], cur["details"]), (6, [CONV], DETAILS))
        self.assert_appended_only(before)

    def test_control_without_carry_forward_keys_are_lost(self):
        registry.append(portal_record(), self.root)
        with mock.patch.object(importers, "UPLOAD_KEYS", ()):
            self.importer_writes(lab_record())
        cur = self.current()
        self.assertNotIn("conversions", cur)
        self.assertNotIn("details", cur)

    def test_write_waits_for_the_upload_lock(self):
        self.root.mkdir(parents=True)
        held = registry.acquire_lock(self.root / importers.UPLOAD_LOCK)
        self.assertIsNotNone(held)
        try:
            with mock.patch.object(importers, "UPLOAD_LOCK_WAIT", 0.2):
                with self.assertRaises(registry.RegistryError) as cm:
                    self.importer_writes(lab_record())
            self.assertIn("registry busy", str(cm.exception))
            self.assertFalse((self.root / "session.jsonl").exists())             # nothing written meanwhile
            imp = importers.Importer(self.root, **self.dirs)                       # a dry run needs no lock
            imp.out = [("session", lab_record())]
            self.assertEqual(imp.write(dry_run=True)["session"], 1)
        finally:
            held.close()
        self.assertEqual(self.importer_writes(lab_record())["session"], 1)

    def test_records_validate(self):
        registry.append(portal_record(), self.root)
        self.importer_writes(lab_record())
        for line in (self.root / "session.jsonl").read_text(encoding="utf-8").splitlines():
            self.assertEqual(registry.check_record(json.loads(line), registry.schemas()), [])
        bad = dict(portal_record(), details=dict(DETAILS, extra="x"))                # details is a closed object
        self.assertTrue(registry.check_record(dict(bad, logged="2099-11-06T10:00:00"), registry.schemas()))


if __name__ == "__main__":
    unittest.main()
