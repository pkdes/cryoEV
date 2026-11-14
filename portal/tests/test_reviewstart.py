"""Start a review (round 13, B4): portal/review_build.py and portal/modules/reviewstart.py, in a throwaway world
(review_build_fixture: temp laptop, temp registry, temp export). Every build here compares a label set with another
label set (--b-coco: no GPU); a real model build runs only with CJ_TEST_GPU_BUILD=1 (outside the gate).

Known answers and controls:
  - inputs: the fixture label set's 3 images are all found (one through the registry's duplicate_of alias), splits
    from the export's split file; a missing image / a different width:height refuses with the counts (control: put
    back -> ok); the link: dataset made from the lv, a roboflow_export naming it, the base chain, ambiguity and no
    link each give the stated answer;
  - a revise build: done, purpose revise, no _work_ folder, its scratch removed, an allowlisted registry line (no
    cards, path, reviewers or decoy numbers); an audit: the _work_ folder holds the key, nothing else shows counts;
  - one build at a time (the runner's lock, a live status), a stale status reads as stopped; the runner refuses
    while another holds the lock; the GPU wait (lock held, GPU busy, no GPU, free);
  - pages: admin sees (a) + both forms + the coverage line; a basic user sees (a) only, gets 403 on the build and 404
    on progress pages; the build POST needs the form token and Origin, writes one audit line without decoy numbers.

  python -m unittest portal.tests.test_reviewstart -v
"""
import json
import os
import re
import shutil
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.parse import urlencode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cjtest as T  # noqa: E402
import review_build_fixture as F  # noqa: E402
from data_utils.oslock import acquire_lock  # noqa: E402
from portal import app as APP  # noqa: E402
from portal import auth as A  # noqa: E402
from portal import review_build as RB  # noqa: E402
from registry import store  # noqa: E402

B_LV = {"kind": "label_version", "id": F.LV_BASE}


def world(tmp):
    w = F.build(tmp, owner=T.USER)
    cfg = SimpleNamespace(review_root=w["review"], registry=w["registry"], state_dir=str(Path(tmp) / "state"),
                          roster=None)
    return w, cfg


def wait_done(rr, name, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        st = RB.read_status(rr, name)
        if st and st["phase"] in RB.TERMINAL:
            return st
        time.sleep(0.2)
    raise AssertionError(f"{name} did not finish: {RB.read_status(rr, name)}")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory(prefix="cj_rs_")))
        self.w, self.cfg = world(self.tmp)
        p = mock.patch.dict(os.environ, {"CJ_PREDICT_ROOT": str(self.w["predict"])})
        p.start()
        self.addCleanup(p.stop)
        RB._cov_cache.clear()

    def build(self, purpose="revise", reviewers=(), b=B_LV):
        name = RB.start(self.cfg, T.USER, purpose, F.LV, b, list(reviewers), spawn=False)
        rc = RB.run(Path(self.cfg.state_dir) / RB.STATE_SUB / name / "spec.json")
        return name, rc, RB.read_status(self.w["review"], name)


class Inputs(Base):
    def test_coverage_known_answer(self):
        cov = RB.coverage(self.cfg, F.LV, use_cache=False)
        self.assertTrue(cov["ok"], cov.get("problem"))
        self.assertEqual((cov["found"], cov["total"], cov["aliased"]), (3, 3, 1))
        self.assertEqual({k: v[1] for k, v in cov["files"].items()},
                         {"img_a_png": "train", "img_b_png": "valid", "img_c_png": "test"})   # the split file's
        self.assertTrue(cov["files"]["img_c_png"][0].endswith("S1__img_c_png.rf.0003.jpg"))
        self.assertEqual(RB.coverage_line(cov), "Images found 3 of 3 (1 through the registry's duplicate_of link) · "
                                                "splits from roboflow - fixture")

    def test_missing_and_mismatched_refused(self):
        img = self.w["export"] / "valid" / "img_b_png.rf.0002.jpg"
        moved = self.tmp / "away.jpg"
        shutil.move(img, moved)
        cov = RB.coverage(self.cfg, F.LV, use_cache=False)
        self.assertFalse(cov["ok"])
        self.assertEqual(cov["problem"], "Images found 2 of 3: 1 missing")
        F.picture(img, w=400, h=400)                                # back, but square: the ratio differs
        cov = RB.coverage(self.cfg, F.LV, use_cache=False)
        self.assertEqual((cov["ok"], cov["missing"], cov["mismatched"]), (False, 0, 1))
        F.picture(img, w=404, h=302)                                # within 1 %: accepted (control)
        self.assertTrue(RB.coverage(self.cfg, F.LV, use_cache=False)["ok"])
        with self.assertRaises(RB.BuildError):                      # start() refuses with the counts
            F.picture(img, w=400, h=400)
            RB.start(self.cfg, T.USER, "revise", F.LV, B_LV, spawn=False)
        self.assertFalse(any(p.name.startswith("lab_") for p in self.w["review"].iterdir()))

    def test_export_link_rules(self):
        reg = self.w["registry"]
        lroot = self.w["labels"]
        self.assertIn("dataset lv_20990101_0101_ds", RB.find_export(F.LV, lroot, reg)["how"])
        with self.assertRaises(RB.BuildError) as cm:                # the base: chain ends at a pull, no export
            RB.find_export(F.LV_BASE, lroot, reg)
        self.assertIn("ends at roboflow_pull_20990101_0000", str(cm.exception))
        child = "lv_20990102_0101"                                  # a newer lv on top: found through its base
        (lroot / child).mkdir()
        (lroot / child / "version.json").write_text(json.dumps({"base": F.LV}), encoding="utf-8")
        got = RB.find_export(child, lroot, reg)
        self.assertEqual((got["export"]["id"], got["chain"]), (F.EXPORT, [child, F.LV]))
        F.jline(reg / "roboflow_export.jsonl", {"schema": "roboflow_export/1", "id": "rf direct", "path": "x",
                                                "label_version": child})
        self.assertEqual(RB.find_export(child, lroot, reg)["export"]["id"], "rf direct")   # the direct link first
        F.jline(reg / "roboflow_export.jsonl", {"schema": "roboflow_export/1", "id": "rf other", "path": "y",
                                                "label_version": child})
        with self.assertRaises(RB.BuildError) as cm:
            RB.find_export(child, lroot, reg)
        self.assertIn("refusing to guess", str(cm.exception))
        with self.assertRaises(RB.BuildError):
            RB.find_export(F.LV, lroot, None)

    def test_aliases_both_ways_and_clusters(self):
        al = RB.image_aliases({"a": {"schema": "image/1", "duplicate_of": "b"}, "c": {"schema": "image/1"},
                               "d": {"schema": "image/1", "duplicate_of": "e"},
                               "f": {"schema": "image/1", "duplicate_of": "e"}})
        self.assertEqual(al, {"a": {"a", "b"}, "b": {"a", "b"}, "d": {"d", "e", "f"}, "e": {"d", "e", "f"},
                              "f": {"d", "e", "f"}})
        F.jline(self.w["registry"] / "image.jsonl", {"schema": "image/1", "id": "T2__img_c_png",   # a third name
                                                     "duplicate_of": "img_c_png"})
        cov = RB.coverage(self.cfg, F.LV, use_cache=False)          # a 3-name cluster: refused, even with 1 on disk
        self.assertEqual((cov["ok"], cov["ambiguous"], cov["ambiguous_ex"]), (False, 1, ["img_c_png"]))
        self.assertIn("1 ambiguous", cov["problem"])

    def test_alias_needs_the_exact_pixel_size(self):
        img = self.w["export"] / "train" / "S1__img_c_png.rf.0003.jpg"
        F.picture(img, w=404, h=302)                                # same ratio within 1 %, other pixel size
        cov = RB.coverage(self.cfg, F.LV, use_cache=False)
        self.assertEqual((cov["ok"], cov["mismatched"], cov["mismatched_ex"]), (False, 1, ["img_c_png"]))
        self.assertIn("1 with a different size", cov["problem"])
        F.picture(self.w["export"] / "train" / "img_a_png.rf.0001.jpg", w=404, h=302)  # control: own key, ratio ok
        F.picture(img)
        self.assertTrue(RB.coverage(self.cfg, F.LV, use_cache=False)["ok"])

    def test_stem_twice_in_the_export_is_ambiguous(self):
        F.picture(self.w["export"] / "valid" / "img_a_png.rf.0099.jpg")   # img_a in train AND valid
        cov = RB.coverage(self.cfg, F.LV, use_cache=False)
        self.assertEqual((cov["ok"], cov["found"], cov["ambiguous"]), (False, 2, 1))
        self.assertEqual(cov["problem"], "Images found 2 of 3: 1 ambiguous (in the export more than once, or more "
                                         "than 2 linked names)")
        F.picture(self.w["export"] / "test" / "S1__img_c_png.rf.0098.jpg")  # an alias's stem twice: ambiguous too
        self.assertEqual(RB.coverage(self.cfg, F.LV, use_cache=False)["ambiguous"], 2)

    def test_unusable_keys_refused(self):
        labels = self.w["labels"] / F.LV / "labels.coco.json"
        coco = json.loads(labels.read_text(encoding="utf-8"))
        for i, name in enumerate(("..evil_png", "a b_png", "_x_png")):
            coco["images"].append({"id": 90 + i, "file_name": f"x{i}.png", "width": F.W, "height": F.H,
                                   "extra": {"name": name}})
        labels.write_text(json.dumps(coco), encoding="utf-8")
        cov = RB.coverage(self.cfg, F.LV, use_cache=False)
        self.assertEqual((cov["ok"], cov["bad_key"], cov["found"], cov["total"]), (False, 3, 3, 6))
        self.assertIn("3 with a name that can't be used", cov["problem"])

    def test_validate(self):
        with self.assertRaises(RB.BuildError) as cm:
            RB.validate(self.cfg, "audit", F.LV, B_LV, [], T.USER)
        self.assertIn("at least one reviewer", str(cm.exception))
        with self.assertRaises(RB.BuildError) as cm:
            RB.validate(self.cfg, "audit", F.LV, B_LV, ["alyssia", T.USER], T.USER)
        self.assertIn("can't review it", str(cm.exception))
        for bad in (("x", F.LV, B_LV, []), ("revise", "lv_19990101_0000", B_LV, []),
                    ("revise", F.LV, {"kind": "label_version", "id": F.LV}, []),
                    ("revise", F.LV, {"kind": "model", "id": "nope"}, []),
                    ("revise", F.LV, B_LV, ["Not An Id"])):
            with self.subTest(bad=bad), self.assertRaises(RB.BuildError):
                RB.validate(self.cfg, *bad, T.USER)
        bd, revs = RB.validate(self.cfg, "audit", F.LV, B_LV, ["alyssia", "alyssia"], T.USER)
        self.assertEqual((bd["kind"], revs), ("label_version", ["alyssia"]))
        with self.assertRaises(RB.BuildError):                      # not on the roster
            RB.start(self.cfg, T.USER, "audit", F.LV, B_LV, ["alyssia"], roster_ids={"bas"}, spawn=False)

    def test_name_fits_the_portal_route(self):
        for p in RB.PURPOSES:
            n = RB.new_name(p, F.LV)
            self.assertRegex(n, RB.NAME_RE)
            self.assertRegex(n, APP.PRED_ID_RE)


class Builds(Base):
    def test_revise_end_to_end(self):
        name, rc, st = self.build("revise")
        self.assertEqual((rc, st["phase"], st["reason"]), (0, "done", None), st)
        rr = self.w["review"]
        self.assertEqual(sorted(p.name for p in rr.iterdir()), [".building", name])        # no _work_ folder
        meta = json.loads((rr / name / "review_package.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["purpose"], meta["built_by"], meta["n_images"], meta["owner"]),
                         ("revise", "portal", 3, T.USER))
        sdir = Path(self.cfg.state_dir) / RB.STATE_SUB / name
        self.assertFalse((sdir / "work").exists())                  # scratch removed
        self.assertFalse((sdir / "images").exists())                # staged links removed
        self.assertIn("[phase] writing", (sdir / "build.log").read_text(encoding="utf-8"))  # raw log: state only
        line = store.current("review_package", self.w["registry"])[name]
        self.assertTrue(set(line) <= set(RB.ALLOWED_LINE_KEYS) | {"logged"}, line)
        self.assertEqual((line["kind"], line["purpose"], line["base"], line["base_kind"], line["b_label_version"],
                          line["recorded_by"]), ("compare", "revise", F.LV, "label_version", F.LV_BASE,
                                                 "portal/review_build.py"))
        for k in ("cards", "path", "reviewers"):
            self.assertNotIn(k, line)
        raw = json.loads(RB.status_path(rr, name).read_text(encoding="utf-8"))
        self.assertTrue(set(raw) <= set(RB.STATUS_KEYS), raw)
        self.assertEqual(raw["detail"], "Images found 3 of 3 (1 through the registry's duplicate_of link) · "
                                        "sizes match")

    def test_audit_end_to_end_nothing_blind_outside_work(self):
        name, rc, st = self.build("audit", reviewers=["alyssia"])
        self.assertEqual((rc, st["phase"]), (0, "done"), st)
        rr = self.w["review"]
        work = rr / f"_work_{name}"
        key = json.loads((work / "decoys.json").read_text(encoding="utf-8"))
        self.assertGreater(len(key["decoys"]), 0)
        meta = json.loads((rr / name / "review_package.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["purpose"], meta["reviewers"]), ("audit", ["alyssia"]))
        self.assertNotIn("cards", meta)
        line = store.current("review_package", self.w["registry"])[name]
        texts = [json.dumps(meta), json.dumps(line), RB.status_path(rr, name).read_text(encoding="utf-8"),
                 (rr / name / "data.js").read_text(encoding="utf-8")]
        for t in texts:
            self.assertNotIn("decoy", t)
            self.assertNotIn("_work_", t)
        n = key["counts"]["decoys"]
        self.assertNotIn(f'"accept": {n["accept"]}', texts[1] + texts[2])

    def test_apply_takes_the_revise_not_the_audit(self):
        """QC 1 end to end: a portal-built revise is eligible for Apply once an image is reviewed; the audit isn't."""
        from portal.modules import applyreview as AP
        rev, _, _ = self.build("revise")
        aud, _, _ = self.build("audit", reviewers=["alyssia"])
        for name in (rev, aud):
            (self.w["review"] / name / "edits").mkdir()
            (self.w["review"] / name / "edits" / "img_a.png.json").write_text(json.dumps(
                {"file": "img_a.png", "pkg": name, "fmt": 1, "edit": {"reviewed": True, "dec": {}}}), encoding="utf-8")
        rows = {r["id"]: r for r in AP.packages(self.cfg, {}, set(), None)}
        self.assertTrue(rows[rev]["eligible"], rows[rev]["why"])
        self.assertEqual(rows[rev]["base_lv"], F.LV)
        self.assertNotIn(aud, rows)

    def test_built_in_staging_then_published(self):
        """QC 3: nothing listable in the review root until the build is complete and checked; _work_ moves first."""
        seen, real = {}, RB.publish

        def spy(rr, name, purpose):
            seen["root"] = sorted(p.name for p in Path(rr).iterdir())
            seen["staged"] = sorted(p.name for p in RB.staging_dir(rr, name).iterdir())
            real(rr, name, purpose)
            seen["after"] = sorted(p.name for p in Path(rr).iterdir())

        with mock.patch.object(RB, "publish", spy):
            name, rc, st = self.build("audit", reviewers=["alyssia"])
        self.assertEqual(rc, 0, st)
        self.assertEqual(seen["root"], [".building"])
        self.assertEqual(seen["staged"], [f"_work_{name}", name])
        self.assertEqual(seen["after"], [".building", f"_work_{name}", name])
        self.assertFalse(RB.staging_dir(self.w["review"], name).exists())
        meta = json.loads((self.w["review"] / name / "review_package.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(meta["path"]), (self.w["review"] / name).resolve())   # its final place, not staging

    def test_compare_runs_in_the_runner_process(self):
        """QC 1: no child process for compare_package (a killed runner can't orphan GPU work)."""
        import subprocess
        with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("no child process")):
            name, rc, st = self.build("revise")
        self.assertEqual((rc, st["phase"]), (0, "done"), st)
        log = (Path(self.cfg.state_dir) / RB.STATE_SUB / name / "build.log").read_text(encoding="utf-8")
        self.assertIn("[phase] writing", log)
        self.assertIn(f"[OK] {name}", log)

    def test_sweep_of_killed_builds(self):
        """QC 3 + 4: at runner start, other non-final statuses become stopped; leftovers of NAME_RE builds go;
        everything else (another package, its _work_, a finished portal build) stays."""
        rr = self.w["review"]
        dead, dead2, done = ("lab_revise_lv_20990101_0101_20990101_0000_aaaa",
                             "lab_audit_lv_20990101_0101_20990101_0000_bbbb",
                             "lab_revise_lv_20990101_0101_20990101_0000_cccc")
        (rr / dead).mkdir()                                         # killed mid-publish: no review_package.json
        (rr / dead / "data.js").write_text("x", encoding="utf-8")
        (rr / f"_work_{dead2}").mkdir()                             # its package folder is gone
        (rr / done).mkdir()                                         # a finished portal build: kept
        (rr / done / "review_package.json").write_text(json.dumps({"id": done, "built_by": "portal"}),
                                                       encoding="utf-8")
        (rr / f"_work_{done}").mkdir()
        for other in ("lab_compare_live_audit", "_work_lab_compare_live_audit", "pred_ps_20990101_cccccc"):
            (rr / other).mkdir()                                    # never a portal name: never touched
        RB.write_status(rr, dead2, phase="predicting", started=RB._now(), pid=999999)
        RB.staging_dir(rr, dead2).mkdir(parents=True)
        name, rc, st = self.build("revise")
        self.assertEqual(rc, 0, st)
        left = sorted(p.name for p in rr.iterdir())
        self.assertEqual(left, sorted([".building", name, done, f"_work_{done}", "lab_compare_live_audit",
                                       "_work_lab_compare_live_audit", "pred_ps_20990101_cccccc"]))
        raw = json.loads(RB.status_path(rr, dead2).read_text(encoding="utf-8"))
        self.assertEqual((raw["phase"], raw["reason"], raw["failed_at"]), ("failed", "stopped", "predicting"))
        self.assertFalse(RB.staging_dir(rr, dead2).exists())
        self.assertIn("swept leftovers", (Path(self.cfg.state_dir) / RB.STATE_SUB / name / "build.log")
                      .read_text(encoding="utf-8"))

    def test_check_failure_removes_the_build(self):
        """QC 11a: a complete build that fails the final check is never published or registered."""
        real = RB.run_compare

        def tamper(args, log, say):
            rc = real(args, log, say)
            out = Path(args[args.index("--out") + 1])
            meta = json.loads((out / "review_package.json").read_text(encoding="utf-8"))
            (out / "review_package.json").write_text(json.dumps(dict(meta, n_images=99)), encoding="utf-8")
            return rc

        with mock.patch.object(RB, "run_compare", tamper):
            name, rc, st = self.build("audit", reviewers=["alyssia"])
        self.assertEqual((rc, st["phase"], st["reason"]), (1, "failed", "check"))
        self.assertEqual(sorted(p.name for p in self.w["review"].iterdir()), [".building"])
        self.assertFalse(RB.staging_dir(self.w["review"], name).exists())
        self.assertNotIn(name, store.current("review_package", self.w["registry"]))

    def test_model_build_frees_predict_lock_and_chains(self):
        """QC 11c: after a model build the predict lock is released and the next queued prediction set started."""
        from jobs import predict_local as PL
        model = {"id": "fake_model", "production": True, "weights": "x.pt", "imgsz": 1440, "classes": "EV,non-EV"}
        lockf = self.w["predict"] / PL.LOCK_FILE
        calls = []

        def compare(args, log, say):
            calls.append(("compare", acquire_lock(lockf, 0) is None))   # the lock is held while it runs
            return 1                                                    # (it fails: no real YOLO here)

        with mock.patch.object(RB, "models", return_value=[model]), \
                mock.patch.object(RB, "wait_for_gpu", lambda spec, say: (acquire_lock(lockf, 0), None)), \
                mock.patch.object(RB, "run_compare", compare), \
                mock.patch.object(PL, "next_queued", return_value=self.w["predict"] / "ps_next"), \
                mock.patch.object(PL, "start_detached", side_effect=lambda *a, **k: calls.append(("next", a[0].name))):
            name, rc, st = self.build("revise", b={"kind": "model", "id": "fake_model"})
        self.assertEqual((rc, st["phase"]), (1, "failed"))
        self.assertEqual(calls, [("compare", True), ("next", "ps_next")])
        free = acquire_lock(lockf, 0)
        self.assertIsNotNone(free)                                      # released
        free.close()

    def test_apply_preview_lands_on_the_label_set_ids(self):
        """QC 11d: the Apply preview (apply_review_edits.make_version) on a portal-built revise: a "del" on label
        g<id> removes exactly that annotation id, a "keep:MV" reclasses exactly that id, every other id is kept."""
        from portal.modules import applyreview as AP
        name, rc, _ = self.build("revise")
        pkg = self.w["review"] / name
        data = json.loads((pkg / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
        im = next(r for r in data["images"] if r["file"] == "img_a.png")
        g_del, g_mv = [g["id"] for g in im["gt"]][:2]
        (pkg / "edits").mkdir()
        (pkg / "edits" / "img_a.png.json").write_text(json.dumps(
            {"file": "img_a.png", "pkg": name, "fmt": 1,
             "edit": {"dec": {f"g{g_del}": {"v": "del", "t": 5, "who": "pujan"},
                              f"g{g_mv}": {"v": "keep:MV", "t": 5, "who": "pujan"}},
                      "t": 5, "modified": True, "reviewed": True, "reviewedT": 5}}), encoding="utf-8")
        row = next(r for r in AP.packages(self.cfg, {}, set(), None) if r["id"] == name)
        self.assertTrue(row["eligible"], row["why"])
        ts, info = AP.run_preview(self.cfg, row, "pujan")
        self.assertFalse(info.get("error"), info)
        base = json.loads((self.w["labels"] / F.LV / "labels.coco.json").read_text(encoding="utf-8"))
        out = json.loads((AP.scratch_root(self.cfg) / name / ts / "labels" / info["lv"] / "labels.coco.json")
                         .read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in out["categories"]}
        got = {a["id"]: cats[a["category_id"]] for a in out["annotations"]}
        want = {a["id"]: "EV" for a in base["annotations"] if a["id"] != g_del}
        want[g_mv] = "MV"
        self.assertEqual(got, want)

    def test_failed_build_cleans_up(self):
        name = RB.start(self.cfg, T.USER, "revise", F.LV, B_LV, spawn=False)
        os.remove(self.w["labels"] / F.LV_BASE / "labels.coco.json")    # B vanishes: compare_package fails
        rc = RB.run(Path(self.cfg.state_dir) / RB.STATE_SUB / name / "spec.json")
        st = RB.read_status(self.w["review"], name)
        self.assertEqual((rc, st["phase"], st["reason"], st["failed_at"]), (1, "failed", "build", "writing"))
        self.assertEqual(sorted(p.name for p in self.w["review"].iterdir()), [".building"])
        self.assertFalse((Path(self.cfg.state_dir) / RB.STATE_SUB / name / "work").exists())
        self.assertNotIn(name, store.current("review_package", self.w["registry"]))

    def test_registry_line_failure_still_done(self):
        with mock.patch.object(RB, "register", side_effect=RuntimeError("registry busy")):
            name, rc, st = self.build("revise")
        self.assertEqual((rc, st["phase"], st["reason"]), (0, "done", "registry"))


class OneAtATime(Base):
    def test_lock_and_live_status_refuse(self):
        rr = self.w["review"]
        (rr / RB.BUILDING).mkdir()
        held = acquire_lock(rr / RB.BUILDING / RB.BUILD_LOCK, 0)
        try:
            with self.assertRaises(RB.BuildError) as cm:
                RB.start(self.cfg, T.USER, "revise", F.LV, B_LV, spawn=False)
            self.assertIn("one at a time", str(cm.exception))
        finally:
            held.close()
        name = RB.start(self.cfg, T.USER, "revise", F.LV, B_LV, spawn=False)   # control: free -> starts
        with self.assertRaises(RB.BuildError):                     # still starting (no runner pid yet): busy
            RB.start(self.cfg, T.USER, "revise", F.LV, B_LV, spawn=False)
        RB.write_status(rr, name, pid=os.getpid())                 # a live runner (this process) with the lock
        held = acquire_lock(rr / RB.BUILDING / RB.BUILD_LOCK, 0)
        try:
            with self.assertRaises(RB.BuildError):
                RB.start(self.cfg, T.USER, "revise", F.LV, B_LV, spawn=False)
            self.assertEqual(RB.read_status(rr, name)["phase"], "checking")
        finally:
            held.close()
        st = RB.read_status(rr, name)                              # lock free: no runner, whatever the pid says
        self.assertEqual((st["phase"], st["reason"], st["failed_at"]), ("failed", "stopped", "checking"))
        RB.write_status(rr, name, pid=999999)                      # a dead pid: stopped too
        self.assertIsNone(RB.active_build(rr))
        RB.start(self.cfg, T.USER, "revise", F.LV, B_LV, spawn=False)

    def test_runner_refuses_when_another_holds_the_lock(self):
        name = RB.start(self.cfg, T.USER, "revise", F.LV, B_LV, spawn=False)
        held = acquire_lock(self.w["review"] / RB.BUILDING / RB.BUILD_LOCK, 0)
        try:
            with mock.patch.object(RB, "acquire_lock", wraps=lambda p, wait=0: acquire_lock(p, 0)):
                rc = RB.run(Path(self.cfg.state_dir) / RB.STATE_SUB / name / "spec.json")
        finally:
            held.close()
        st = RB.read_status(self.w["review"], name)
        self.assertEqual((rc, st["phase"], st["reason"]), (3, "failed", "busy"))
        self.assertFalse((self.w["review"] / name).exists())

    def test_status_allowlist(self):
        with self.assertRaises(ValueError):
            RB.write_status(self.w["review"], "lab_revise_lv_20990101_0101_20990101_0000_abcd", phase="checking",
                            decoys=3)
        with self.assertRaises(ValueError):
            RB.write_status(self.w["review"], "lab_revise_lv_20990101_0101_20990101_0000_abcd", phase="secret")
        self.assertIsNone(RB.read_status(self.w["review"], "../etc"))


class GpuWait(Base):
    def spec(self):
        return {"predict_root": str(self.w["predict"]), "wait_s": 5, "wait_poll_s": 1}

    def run_wait(self, gpu, hold=False, ticks=20):
        from jobs import predict_local as PL
        said, clock = [], [0.0]
        held = acquire_lock(self.w["predict"] / PL.LOCK_FILE, 0) if hold else None
        try:
            with mock.patch.object(PL, "gpu_check", side_effect=gpu):
                lock, why = RB.wait_for_gpu(self.spec(), lambda **f: said.append(f["reason"]),
                                            sleep=lambda s: clock.__setitem__(0, clock[0] + s),
                                            clock=lambda: clock[0])
        finally:
            if held:
                held.close()
        return lock, why, said

    def test_lock_held_then_timeout(self):
        lock, why, said = self.run_wait(lambda a: ("cuda", "", None), hold=True)
        self.assertEqual((lock, why, said), (None, "gpu_timeout", ["lock"]))

    def test_busy_missing_free(self):
        self.assertEqual(self.run_wait(lambda a: (None, "gpu busy: pid 1", "busy"))[1:], ("gpu_timeout", ["gpu_busy"]))
        self.assertEqual(self.run_wait(lambda a: (None, "none", "missing"))[1:], ("gpu_missing", []))
        lock, why, said = self.run_wait(lambda a: ("cuda", "", None))
        try:
            self.assertEqual((why, said), (None, []))
            from jobs import predict_local as PL
            self.assertIsNone(acquire_lock(self.w["predict"] / PL.LOCK_FILE, 0))   # held for the build
        finally:
            lock.close()


class Detached(Base):
    def test_real_detached_runner(self):
        """spawn_runner for real: the runner is its own process (python -m jobs.review_build_run) and finishes."""
        name = RB.start(self.cfg, T.USER, "revise", F.LV, B_LV)
        st = wait_done(self.w["review"], name)
        self.assertEqual(st["phase"], "done", st)
        self.assertNotEqual(st["pid"], os.getpid())
        end = time.time() + 60                                      # let it exit before the temp folder goes
        while RB._pid_alive(st["pid"]) and time.time() < end:
            time.sleep(0.2)


@unittest.skipUnless(os.environ.get("CJ_TEST_GPU_BUILD") == "1", "real GPU build: CJ_TEST_GPU_BUILD=1 only")
class RealModel(Base):
    def test_model_build(self):
        ms = RB.models(self.cfg)
        self.assertTrue(ms)
        name, rc, st = self.build("revise", b={"kind": "model", "id": ms[0]["id"]})
        self.assertEqual((rc, st["phase"]), (0, "done"), st)


class Pages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.w = F.build(cls.env.tmp, owner=T.USER)
        cls.envp = mock.patch.dict(os.environ, {"CJ_PREDICT_ROOT": str(cls.w["predict"])})
        cls.envp.start()
        cls.keys = {T.USER: cls.env.key, "bas": A.add_key(cls.env.roster, "bas", "basic"),
                    "alyssia": A.add_key(cls.env.roster, "alyssia", "advanced")}
        cls.p = T.Portal(cls.env.config(review_root=cls.w["review"], registry=cls.w["registry"]))
        cls.tok = {u: T.login(cls.p.port, k) for u, k in cls.keys.items()}
        cls.threads = []

        def fake_spawn(spec_path):                                 # the runner in a thread of this process
            t = threading.Thread(target=RB.run, args=(spec_path,), daemon=True)
            t.start()
            cls.threads.append(t)
            return 0
        cls.spawn = mock.patch.object(RB, "spawn_runner", fake_spawn)
        cls.spawn.start()

    @classmethod
    def tearDownClass(cls):
        for t in cls.threads:
            t.join(60)
        cls.spawn.stop()
        cls.p.stop()
        cls.envp.stop()
        cls.env.cleanup()

    def get(self, user, path):
        return T.request(self.p.port, "GET", path, T.cookie(self.tok[user]) if user else {})

    def post(self, user, path, fields, origin=True):
        h = {**T.cookie(self.tok[user]), "Content-Type": "application/x-www-form-urlencoded"}
        if origin:
            h["Origin"] = self.p.origin
        return T.request(self.p.port, "POST", path, h, urlencode(fields, doseq=True).encode())

    def token(self, user=T.USER):
        st, _, body = self.get(user, "/reviews/start/")
        self.assertEqual(st, 200)
        return re.search(r'name="t" value="([0-9a-f]+)"', body.decode()).group(1)

    def test_1_admin_page(self):
        st, _, body = self.get(T.USER, "/reviews/start/")
        page = body.decode()
        self.assertEqual(st, 200)
        self.assertIn("Revise a label set", page)
        self.assertIn("Blind audit", page)
        self.assertIn("Images found 3 of 3", page)
        self.assertIn('action="/predict/ps_20990101_aaaaaa/review"', page)       # (a): own finished set
        self.assertNotIn("ps_20990101_bbbbbb", page)                              # someone else's: never
        self.assertIn(f'value="lv:{F.LV_BASE}"', page)
        self.assertIn("tester (owner, can't review)", page)
        self.assertIn('value="alyssia"', page)
        self.assertEqual(self.get(T.USER, "/reviews/start")[0], 307)              # the module's redirect

    def test_2_basic_user_sees_a_only(self):
        st, _, body = self.get("bas", "/reviews/start/")
        page = body.decode()
        self.assertEqual(st, 200)
        self.assertIn("Finished prediction sets without a review", page)
        for hidden in ("Revise a label set", "Blind audit", F.LV, F.LV_BASE, "Images found", 'name="t"',
                       "ps_20990101_aaaaaa"):
            self.assertNotIn(hidden, page)
        st, _, body = self.post("bas", "/reviews/start/build", {"t": "x", "purpose": "revise", "lv": F.LV,
                                                                "b": f"lv:{F.LV_BASE}"})
        self.assertEqual(st, 403)
        self.assertEqual(self.get("bas", "/reviews/start/lab_revise_lv_20990101_0101_20990101_0000_abcd")[0], 404)

    def test_3_post_needs_token_and_origin(self):
        f = {"purpose": "revise", "lv": F.LV, "b": f"lv:{F.LV_BASE}"}
        self.assertEqual(self.post(T.USER, "/reviews/start/build", dict(f, t="0" * 40))[0], 403)
        self.assertEqual(self.post(T.USER, "/reviews/start/build", dict(f, t=self.token()), origin=False)[0], 403)
        st, _, body = self.post(T.USER, "/reviews/start/build", dict(f, t=self.token(), b="model:nope"))
        self.assertEqual(st, 400)
        self.assertIn("pick a model", body.decode())
        st, _, body = self.post(T.USER, "/reviews/start/build", dict(f, t=self.token(), purpose="audit"))
        self.assertEqual(st, 400)                                                   # an audit without reviewers
        st, _, body = self.post(T.USER, "/reviews/start/build", dict(f, t=self.token(), purpose="audit",
                                                                     reviewer=["alyssia", T.USER]))
        self.assertIn("can&#x27;t review it", body.decode())
        self.assertFalse(any(p.name.startswith("lab_") for p in self.w["review"].iterdir()))

    def test_4_build_progress_and_audit_line(self):
        f = {"purpose": "revise", "lv": F.LV, "b": f"lv:{F.LV_BASE}", "t": self.token()}
        st, h, _ = self.post(T.USER, "/reviews/start/build", f)
        self.assertEqual(st, 303)
        loc = T.hget(h, "location")
        name = loc.rsplit("/", 1)[1]
        self.assertRegex(name, RB.NAME_RE)
        st = wait_done(self.w["review"], name)
        self.assertEqual(st["phase"], "done", st)
        st, _, body = self.get(T.USER, loc)
        page = body.decode()
        self.assertEqual(st, 200)
        self.assertIn(f'href="/reviews/go/{name}"', page)
        self.assertNotIn('http-equiv="refresh"', page)
        self.assertIn("Images found 3 of 3", page)
        lines = [r for r in T.module_lines(self.env.state / "audit" / "portal.jsonl")
                 if r.get("action") == "review build started"]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["target"], name)
        self.assertEqual(lines[0]["detail"], {"purpose": "revise", "lv": F.LV, "b": f"label_version:{F.LV_BASE}",
                                              "reviewers": 0})
        # a build in progress: the page reloads itself and shows the step it is on
        rr = self.w["review"]
        busy = "lab_audit_lv_20990101_0101_20990101_0000_beef"
        RB.write_status(rr, busy, purpose="audit", lv=F.LV, b_kind="model", b_id="v5_evnonev_y8m_20261029",
                        phase="waiting", reason="gpu_busy", detail="Images found 3 of 3 · aspect ratios match",
                        started=RB._now(), pid=os.getpid(), by=T.USER)
        held = acquire_lock(rr / RB.BUILDING / RB.BUILD_LOCK, 0)       # as its runner would
        try:
            page = self.get(T.USER, f"/reviews/start/{busy}")[2].decode()
            self.assertIn('http-equiv="refresh"', page)
            self.assertIn("The GPU is busy", page)
            start = self.get(T.USER, "/reviews/start/")[2].decode()
            self.assertIn("A review is being built now", start)                    # one at a time, shown
            self.assertEqual(self.post(T.USER, "/reviews/start/build", dict(f, t=self.token()))[0], 409)
        finally:
            held.close()
            RB.write_status(rr, busy, phase="failed", reason="stopped")

    def test_5_check_images_is_its_own_get_form(self):
        """QC 9: the label-set picker is a GET form without the form token; the build forms carry the lv hidden."""
        page = self.get(T.USER, "/reviews/start/")[2].decode()
        get_form = re.search(r'<form method="get"[^>]*>(.*?)</form>', page, re.S).group(1)
        self.assertNotIn('name="t"', get_form)
        self.assertIn('name="lv"', get_form)
        self.assertEqual(len(re.findall(r'<input type="hidden" name="lv" value="lv_20990101_0101">', page)), 2)
        page = self.get(T.USER, f"/reviews/start/?lv={F.LV_BASE}")[2].decode()
        self.assertIn("no registered Roboflow export is linked", page)             # the base has no export
        self.assertEqual(len(re.findall(r'type="submit" disabled', page)), 2)       # nothing to build from it


if __name__ == "__main__":
    unittest.main()
