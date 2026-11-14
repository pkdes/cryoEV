"""compare_package round 13: --purpose revise|audit, refusing leftovers, cleanup on failure, an atomic
review_package.json written last, built_by, the audit owner rule and nothing blind on stdout.

Fixture: test_compare_package's one-image world (known answer without decoys: 1 FP card + 1 FN card).
Controls: the default purpose is audit (earlier command lines unchanged); an audit with decoys 0 still has no
"cards" in review_package.json; a revise's real counts equal the known answer.

  python -m unittest annotation.tests.test_compare_purpose -v
"""
import argparse
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import compare_package as CP  # noqa: E402
from test_compare_package import CompareKnownAnswer  # noqa: E402


class Purpose(CompareKnownAnswer):
    # the fixture only: the inherited tests run in test_compare_package
    test_cards_known_answer_without_decoys = test_decoys_known_answer_and_key_not_served = None
    test_extra_models_union_known_answer = test_source_title = test_reviewers_are_portal_ids = None

    def ns(self, name, **kw):
        a = dict(labels=self.labels, images_root=[self.t / "imgs"], select_csv=None, where=None, split_csv=None,
                 b_model=None, b_preds=self.t / "preds", b_coco=None, b_label="B", imgsz=1440, nms=0.7, max_det=1000,
                 device="cpu", pred_classes="EV,non-EV", class_groups=CP.B.CLASS_GROUPS, flag_conf=0.55, decoys=None,
                 seed=0, owner="tester", reviewer=None, reuse_key=None, index_html=None, width=1440, quality=72,
                 force=False, out=self.t / "review" / name, source_title=None)
        a.update(kw)
        return argparse.Namespace(**a)

    def run_build(self, name, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out, work = CP.build(self.ns(name, **kw))
        return out, work, buf.getvalue()

    def meta(self, out):
        return json.loads((out / "review_package.json").read_text(encoding="utf-8"))

    def data(self, out):
        return json.loads((out / "data.js").read_text(encoding="utf-8").split("=", 1)[1].rstrip().rstrip(";"))

    # ---------------------------------------------------------------- revise
    def test_revise_no_decoys_no_work_folder(self):
        wd = self.t / "state" / "work_rev"
        out, work, printed = self.run_build("lab_rev_a", purpose="revise", work_dir=wd)
        meta = self.meta(out)
        self.assertEqual((meta["kind"], meta["purpose"]), ("compare", "revise"))
        self.assertEqual(meta["cards"], {"fp": 1, "fn": 1})            # the known answer, no decoys added
        d = self.data(out)
        self.assertEqual(sum(1 for p in d["images"][0]["pr"] if p.get("u")), 1)
        self.assertEqual(sum(1 for g in d["images"][0]["gt"] if g.get("u")), 1)
        self.assertFalse((self.t / "review" / "_work_lab_rev_a").exists())  # nothing next to it in the review root
        self.assertFalse(wd.exists())                                  # scratch deleted after the build
        self.assertEqual(sorted(p.name for p in (self.t / "review").iterdir()), ["lab_rev_a"])
        self.assertIn("[phase] writing", printed)

    def test_revise_without_work_dir_uses_a_temp_folder(self):
        out, work, _ = self.run_build("lab_rev_tmp", purpose="revise")
        self.assertFalse(work.exists())
        self.assertFalse(CP.inside(work, self.t / "review"))
        self.assertEqual(sorted(p.name for p in (self.t / "review").iterdir()), ["lab_rev_tmp"])

    def test_revise_refuses_decoys(self):
        for bad in (0.1, 1.0):
            with self.subTest(decoys=bad), self.assertRaises(SystemExit) as cm:
                self.run_build("lab_rev_dec", purpose="revise", decoys=bad)
            self.assertIn("no decoys", str(cm.exception))
        self.assertFalse((self.t / "review").exists())                 # refused before any work
        out, _, _ = self.run_build("lab_rev_dec0", purpose="revise", decoys=0)   # 0 is accepted (control)
        self.assertEqual(self.meta(out)["purpose"], "revise")

    def test_revise_work_dir_inside_review_root_or_existing_refused(self):
        (self.t / "review").mkdir()
        with self.assertRaises(SystemExit) as cm:
            self.run_build("lab_rev_in", purpose="revise", work_dir=self.t / "review" / "scratch")
        self.assertIn("outside the review root", str(cm.exception))
        (self.t / "old_work").mkdir()
        with self.assertRaises(SystemExit) as cm:
            self.run_build("lab_rev_old", purpose="revise", work_dir=self.t / "old_work")
        self.assertIn("already exists", str(cm.exception))
        self.assertEqual(list((self.t / "review").iterdir()), [])

    def test_revise_refused_when_a_work_folder_exists_even_with_force(self):
        (self.t / "review" / "_work_lab_rev_w").mkdir(parents=True)
        for force in (False, True):
            with self.subTest(force=force), self.assertRaises(SystemExit):
                self.run_build("lab_rev_w", purpose="revise", force=force)
        self.assertFalse((self.t / "review" / "lab_rev_w").exists())

    # ---------------------------------------------------------------- audit
    def test_audit_work_dir_holds_the_key_elsewhere(self):
        """r13 fixer 12: an audit with --work-dir writes its answer key there (no _work_<name> next to the package);
        an existing --work-dir or one inside the package is refused."""
        wd = self.t / "stage" / "w"
        out, work, _ = self.run_build("lab_aud_wd", purpose="audit", work_dir=wd, seed=7)
        self.assertEqual(work, wd)
        self.assertEqual(json.loads((wd / "decoys.json").read_text(encoding="utf-8"))["seed"], 7)
        self.assertFalse((self.t / "review" / "_work_lab_aud_wd").exists())
        with self.assertRaises(SystemExit) as cm:
            self.run_build("lab_aud_wd2", purpose="audit", work_dir=wd)
        self.assertIn("already exists", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            self.run_build("lab_aud_wd3", purpose="audit", work_dir=self.t / "review" / "lab_aud_wd3" / "w")
        self.assertIn("inside the package", str(cm.exception))

    def test_audit_counts_only_in_work(self):
        out, work, printed = self.run_build("lab_aud_a", decoys=1.0)          # default purpose: audit
        meta = self.meta(out)
        self.assertEqual(meta["purpose"], "audit")
        self.assertNotIn("cards", meta)
        key = json.loads((work / "decoys.json").read_text(encoding="utf-8"))
        n_dec = key["counts"]["decoys"]
        self.assertGreater(n_dec["accept"] + n_dec["reject"], 0)
        self.assertEqual(key["counts"]["real"], {"fp": 1, "fn": 1})
        served = (out / "data.js").read_text(encoding="utf-8")
        for text in (json.dumps(meta), served, printed):
            self.assertNotIn("decoys.json", text)
            self.assertNotIn("accept", text)
            self.assertNotIn(str(work), text)
        self.assertNotIn("FP 1", printed)                              # real counts (shown - real = decoys) too
        self.assertFalse(any(out.rglob("decoys.json")))

    def test_audit_owner_cannot_review(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_build("lab_aud_own", reviewer=["au1", "tester"])
        self.assertIn("owner can't review", str(cm.exception))
        self.assertFalse((self.t / "review").exists())
        out, _, _ = self.run_build("lab_aud_ok", reviewer=["au1"])          # control: someone else is fine
        self.assertEqual(self.meta(out)["reviewers"], ["au1"])
        out, _, _ = self.run_build("lab_rev_own", purpose="revise", reviewer=["tester"])  # a revise: no blind rule
        self.assertEqual(self.meta(out)["reviewers"], ["tester"])
        self.assertEqual(CP.check_reviewers(["a"], "a", "revise"), ["a"])

    # ---------------------------------------------------------------- both
    def test_existing_out_or_work_refused(self):
        (self.t / "review" / "lab_left").mkdir(parents=True)               # a crash leftover, empty
        (self.t / "review" / "lab_left" / "data.js").write_text("x", encoding="utf-8")
        with self.assertRaises(SystemExit) as cm:
            self.run_build("lab_left")
        self.assertIn("already exists", str(cm.exception))
        self.assertEqual((self.t / "review" / "lab_left" / "data.js").read_text(encoding="utf-8"), "x")  # untouched
        (self.t / "review" / "_work_lab_w2").mkdir()
        with self.assertRaises(SystemExit):
            self.run_build("lab_w2")
        self.assertFalse((self.t / "review" / "lab_w2").exists())
        self.assertEqual(list((self.t / "review" / "_work_lab_w2").iterdir()), [])

    def test_force_still_rebuilds_in_place(self):
        out, _, _ = self.run_build("lab_force")
        out2, _, _ = self.run_build("lab_force", force=True)
        self.assertEqual(out, out2)
        self.assertEqual(self.meta(out2)["id"], "lab_force")

    def test_failure_removes_partial_folders(self):
        for purpose in ("audit", "revise"):
            with self.subTest(purpose=purpose):
                name = f"lab_fail_{purpose}"
                wd = self.t / "state" / f"w_{purpose}" if purpose == "revise" else None
                with mock.patch.object(CP.B, "write_packages", side_effect=RuntimeError("disk full")):
                    with self.assertRaises(RuntimeError):
                        self.run_build(name, purpose=purpose, work_dir=wd)
                self.assertFalse((self.t / "review" / name).exists())
                self.assertFalse((self.t / "review" / f"_work_{name}").exists())
                if wd:
                    self.assertFalse(wd.exists())

    def test_no_images_refused_and_cleaned(self):
        (self.t / "empty").mkdir()
        with self.assertRaises(SystemExit) as cm:
            self.run_build("lab_noimg", images_root=[self.t / "empty"])
        self.assertIn("nothing to build", str(cm.exception))
        self.assertEqual(list((self.t / "review").iterdir()) if (self.t / "review").exists() else [], [])

    def test_review_package_json_last_and_atomic(self):
        seen = {}
        real = CP.write_json_atomic

        def spy(path, obj):
            out = path.parent
            seen["before"] = sorted(p.name for p in out.iterdir())
            seen["work"] = sorted(p.name for p in (out.parent / f"_work_{out.name}").iterdir())
            real(path, obj)

        with mock.patch.object(CP, "write_json_atomic", spy):
            out, _, _ = self.run_build("lab_last")
        self.assertNotIn("review_package.json", seen["before"])
        self.assertTrue({"data.js", "index.html", "images", ".review_key"} <= set(seen["before"]))
        self.assertIn("decoys.json", seen["work"])                     # the key is complete before the package is
        self.assertEqual([p.name for p in out.iterdir() if p.name.endswith(".tmp")], [])

    def test_built_by_portal(self):
        out, _, _ = self.run_build("lab_portal", built_by="portal")
        self.assertEqual(self.meta(out)["built_by"], "portal")
        self.assertEqual(self.data(out)["built_by"], "portal")
        out, _, _ = self.run_build("lab_cli")                              # control: a CLI build has neither
        self.assertNotIn("built_by", self.meta(out))
        self.assertNotIn("built_by", self.data(out))

    def test_force_rebuild_keeps_built_by(self):
        out, _, _ = self.run_build("lab_keep", built_by="portal")
        out, _, _ = self.run_build("lab_keep", force=True)                   # no --built-by: carried over
        self.assertEqual((self.meta(out)["built_by"], self.data(out)["built_by"]), ("portal", "portal"))
        (out / "review_package.json").unlink()                                # only data.js says so: still kept
        out, _, _ = self.run_build("lab_keep", force=True)
        self.assertEqual(self.meta(out)["built_by"], "portal")
        out, _, _ = self.run_build("lab_keep", force=True, built_by="none")  # explicit override: dropped
        self.assertNotIn("built_by", self.meta(out))
        self.assertNotIn("built_by", self.data(out))
        out, _, _ = self.run_build("lab_cli2")                                 # control: a CLI package stays CLI
        out, _, _ = self.run_build("lab_cli2", force=True)
        self.assertNotIn("built_by", self.meta(out))

    def test_atomic_write_retries_permission_error(self):
        real, calls = CP.os.replace, []

        def flaky(a, b):
            calls.append(1)
            if len(calls) < 3:
                raise PermissionError("held by a reader")
            return real(a, b)

        p = self.t / "x.json"
        with mock.patch.object(CP.os, "replace", flaky), mock.patch.object(CP.time, "sleep"):
            CP.write_json_atomic(p, {"a": 1})
        self.assertEqual((len(calls), json.loads(p.read_text(encoding="utf-8"))), (3, {"a": 1}))
        with mock.patch.object(CP.os, "replace", side_effect=PermissionError("held")), \
                mock.patch.object(CP.time, "sleep"), self.assertRaises(PermissionError):
            CP.write_json_atomic(self.t / "y.json", {"a": 2})
        self.assertEqual([q.name for q in self.t.glob(".y.json*")], [])     # no temp file left behind

    def test_cli_flags(self):
        argv = ["compare_package.py", "--labels", "l.json", "--images-root", "i", "--owner", "o", "--out", "x"]
        for extra, want in (([], ("audit", None, None, None)),
                            (["--purpose", "revise", "--work-dir", "w", "--built-by", "portal"],
                             ("revise", None, Path("w"), "portal")),
                            (["--decoys", "0.2"], ("audit", 0.2, None, None))):
            with mock.patch.object(CP, "build") as b, mock.patch.object(sys, "argv", argv + extra):
                CP.main()
            a = b.call_args[0][0]
            self.assertEqual((a.purpose, a.decoys, a.work_dir, a.built_by), want)
        with mock.patch.object(sys, "argv", argv + ["--built-by", "someone"]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            CP.main()


if __name__ == "__main__":
    unittest.main()
