"""Apply (round 13): a label-set revision (compare package, purpose "revise" in its own review_package.json, no
_work_ folder) is eligible; an audit is not; neither is a compare without a purpose (pre-round-13: an audit), nor a
revise whose purpose is forged next to a _work_ folder, nor one whose purpose comes only from the registry line.
Control: test_applyreview's label-set package stays eligible in the same world.

  python -m unittest portal.tests.test_applyreview_revise -v
"""
import json
import os
import sys
import unittest
from unittest import mock

from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_applyreview as TA  # noqa: E402  (module import: its TestCases are not collected twice)
from portal.modules import applyreview as M  # noqa: E402


class ReviseEligible(unittest.TestCase):
    def setUp(self):
        self.w = TA.World()
        self.addCleanup(self.w.cleanup)
        p = mock.patch.dict(os.environ, {"CRYOEV_SERVICES_CONFIG": str(self.w.services)})
        p.start()
        self.addCleanup(p.stop)
        base = str(self.w.base)
        mk = lambda name, meta: TA.make_package(self.w.review, name, self.w.base, meta=meta)  # noqa: E731
        mk("lab_revise_ok", {"kind": "compare", "purpose": "revise", "base": base, "owner": "pujan"})
        mk("lab_audit_x", {"kind": "compare", "purpose": "audit", "base": base, "owner": "pujan"})
        mk("lab_cmp_old", {"kind": "compare", "base": base, "owner": "pujan"})
        mk("lab_forged", {"kind": "compare", "purpose": "revise", "base": base, "owner": "pujan"})
        (self.w.review / "_work_lab_forged").mkdir()
        mk("lab_regonly", {"kind": "compare", "base": base, "owner": "pujan"})
        for name, extra, meta_text in (("lab_halfbuilt", {"kind": "compare"}, None),          # json missing
                                       ("lab_portal_bad", {"built_by": "portal"}, "{not json"),  # json unreadable
                                       ("lab_portal_ok", {"built_by": "portal"}, None)):
            pkg = TA.make_package(self.w.review, name, self.w.base)
            d = json.loads((pkg / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
            (pkg / "data.js").write_text("window.REVIEW_DATA = " + json.dumps({**d, **extra}) + ";",
                                         encoding="utf-8")
            if meta_text:
                (pkg / "review_package.json").write_text(meta_text, encoding="utf-8")
        (self.w.review / "lab_portal_ok" / "review_package.json").write_text(json.dumps(   # control: json present
            {"kind": "compare", "purpose": "revise", "built_by": "portal", "base": base, "owner": "pujan"}),
            encoding="utf-8")
        import shutil
        shutil.rmtree(self.w.review.parent / "_build", ignore_errors=True)
        self.records = {"review_package": {"lab_regonly": {"id": "lab_regonly", "kind": "compare",
                                                           "purpose": "revise"}}}

    def rows(self):
        served, problem = M.served_packages()
        return {r["id"]: r for r in M.packages(self.w.cfg(), self.records, served, problem)}

    def test_revise_eligible_audits_not(self):
        rows = self.rows()
        self.assertTrue(rows["lab_ok"]["eligible"], rows["lab_ok"]["why"])            # control (label-set package)
        self.assertTrue(rows["lab_revise_ok"]["eligible"], rows["lab_revise_ok"]["why"])
        self.assertTrue(rows["lab_portal_ok"]["eligible"], rows["lab_portal_ok"]["why"])
        for pkg in ("lab_audit_x", "lab_cmp_old", "lab_forged", "lab_regonly", "cmp_audit", "lab_halfbuilt",
                    "lab_portal_bad"):
            self.assertNotIn(pkg, rows, pkg)

    def test_unbacked_compare_excluded(self):
        """QC 3: data.js says compare / portal-built while the package's own json is missing or unreadable."""
        cfg = self.w.cfg()
        for pkg in ("lab_halfbuilt", "lab_portal_bad"):
            self.assertTrue(M.unbacked_compare(self.w.review, pkg), pkg)
            self.assertTrue(M.excluded(cfg, pkg, self.records), pkg)
        for pkg in ("lab_ok", "lab_portal_ok", "lab_revise_ok"):           # controls
            self.assertFalse(M.unbacked_compare(self.w.review, pkg), pkg)
            self.assertFalse(M.excluded(cfg, pkg, self.records), pkg)

    def test_serve_time_recheck(self):
        cfg = self.w.cfg()
        self.assertFalse(M.excluded(cfg, "lab_revise_ok", self.records))
        for pkg in ("lab_audit_x", "lab_cmp_old", "lab_forged", "lab_regonly"):
            self.assertTrue(M.excluded(cfg, pkg, self.records), pkg)
        work = self.w.review / "_work_lab_revise_ok"                      # a _work_ folder appearing later
        work.mkdir()
        try:
            self.assertTrue(M.excluded(cfg, "lab_revise_ok", self.records))
            self.assertNotIn("lab_revise_ok", self.rows())
        finally:
            work.rmdir()
        meta = self.w.review / "lab_revise_ok" / "review_package.json"   # the purpose changed to audit on disk
        old = meta.read_bytes()
        meta.write_text(json.dumps({"kind": "compare", "purpose": "audit"}), encoding="utf-8")
        try:
            self.assertTrue(M.excluded(cfg, "lab_revise_ok", self.records))
        finally:
            meta.write_bytes(old)
        self.assertFalse(M.excluded(cfg, "lab_revise_ok", self.records))  # control: back

    def test_is_revise_fails_closed(self):
        root = self.w.review
        self.assertTrue(M.is_revise(root, "lab_revise_ok"))
        self.assertFalse(M.is_revise(root, "nope"))
        (root / "lab_bad").mkdir()
        (root / "lab_bad" / "review_package.json").write_text("{bad", encoding="utf-8")
        self.assertFalse(M.is_revise(root, "lab_bad"))
        self.assertFalse(M.is_revise(None, "lab_revise_ok"))


if __name__ == "__main__":
    unittest.main()
