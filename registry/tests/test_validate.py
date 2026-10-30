"""validate.py on a synthetic artifact tree with known answers, report-path refusals, read-only checks."""
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REG))
import validate  # noqa: E402

LIPO = "grid6_1_r0c0_t00_jpg"          # stands in for an excluded-session image; must never be named in output
PULL = "roboflow_pull_20990101_0000"
RUN = "2099-01-01_000000_y8m_1440_8b_300ep_Run_1"
DS = "roboflow_20990101_test_S1"
EXPORT = "roboflow - 20990101 test S1"
SHA0 = "0" * 64


def w(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def coco(names, with_extra=True):
    ims = []
    for n, (name, rid) in enumerate(names, 1):
        im = {"id": n, "file_name": name, "width": 10, "height": 10}
        if with_extra:
            im["extra"] = {"name": validate.ids.image_key(name), "roboflow_id": rid, "split": "train"}
        ims.append(im)
    return json.dumps({"images": ims, "annotations": [], "categories": []})


def build_fixture(root):
    lap, cache, proc, repo = (root / x for x in ("CryoAI-laptop", "CryoAI", "proc", "repo"))
    # repo: one model from RUN, weights present; one legacy model with no run
    w(repo / "models/models.yaml", f"models:\n  v9_test_y8m_20990101:\n    weights: models/v9/best.pt\n"
      f"    train_data: {EXPORT} (2 images)\n    source_run: training outputs/{DS}/runs/{RUN}\n"
      f"  oldmodel:\n    weights: models/old/best.pt\n")
    w(repo / "models/v9/best.pt", "x")
    # cache: export (2 kept + 1 excluded), split file, run + dataset.yaml
    exp = cache / "training images" / EXPORT
    w(exp / "train/_annotations.coco.json",
      coco([("a-one_png.rf.0a.jpg", None), ("b-two_png.rf.0b.jpg", None), (f"{LIPO}.rf.0c.jpg", None)], False))
    w(exp / "split_assignment.csv", "image,batch,split\na-one_png,20990101 A,train\nb-two_png,20990101 A,train\n"
                                    f"{LIPO},20261012 lipo+VLDL,train\n")
    w(exp / "SPLIT_REPORT.md", "- Source: `roboflow - 20990101 test`, 3 images\n")
    w(cache / "training images" / "roboflow - 20980101 empty" / "train" / ".keep", "")
    w(cache / "training outputs" / DS / "dataset.yaml", "train: train\n")
    w(cache / "training outputs" / DS / "runs" / RUN / "args.yaml", "imgsz: 1440\n")
    # laptop: pull with 2 images, tracking, packages (one duplicated id), lv, batches
    w(lap / PULL / "current.coco.json", coco([("a one.png", "A" * 20), ("b two.png", "B" * 20)]))
    w(lap / PULL / "roboflow_state.json", "[]")
    w(lap / "tracking/v2/prelipo_tracking.csv",
      "stem,session_date,session_date_basis,grid_id,grid,roboflow_id\n"
      f"a-one_png,2099-01-01,tracker,2601,1,{'A' * 20}\nb-two_png,,,,,{'B' * 20}\n")
    dj = lambda pid: "window.REVIEW_DATA = " + json.dumps(
        {"id": pid, "source": f"C:\\x\\{PULL}\\current.coco.json", "run": "v9_test_y8m_20990101 (F)", "images": []}) + ";"
    w(lap / "review/pkgA/data.js", dj("pkgA"))
    w(lap / "review/pkgA/edits/a one.png.json", json.dumps({"file": "a one.png", "pkg": "pkgA", "fmt": 1, "edit": {}}))
    w(lap / "review/pkgB/data.js", dj("pkgB"))
    w(lap / "review/SANDBOX_pkgB/data.js", dj("pkgB"))
    w(lap / "review/_control/package/data.js", dj("package"))
    lv = lap / "labels/lv_20990101_0000"
    w(lv / "version.json", json.dumps({"id": "lv_20990101_0000", "base": PULL, "pkg": "pkgA", "edits_dir": "e",
                                       "n_images_applied": 1, "counts_by_op": {"reclass": 1}, "class_groups_sha256": SHA0,
                                       "people_sha256": SHA0, "tool": "apply_review_edits.py abc", "created": "2099-01-01T00:00:00"}))
    row = {"schema": "label_change/1", "label_version": "lv_20990101_0000", "base": PULL, "pkg": "pkgA",
           "image": "a-one_png", "dec_key": "g1", "op": "reclass", "who": "pujan", "status": "applied"}
    w(lv / "changes.jsonl", json.dumps(row) + "\n" + json.dumps(dict(row, op="explode")) + "\n")
    lv2 = lap / "labels/lv_20990101_0001"
    w(lv2 / "version.json", json.dumps({"id": "lv_20990101_0001", "base": PULL, "pkg": "pkgZ"}))
    w(lv2 / "SUPERSEDED.md", "x")
    ic = "20990101_v9_test_y8m_20990101_20990101-000000"
    line = {"batch_id": ic, "created": "2099-01-01T00:00:00", "session": "20990101", "tiles_dir": "t",
            "model": "v9_test_y8m_20990101", "predictions_dir": "p", "n": 2, "seed": 1, "strata": {}, "status": "staged",
            "logged": "2099-01-01T00:00:01"}
    lipo_ic = dict(line, batch_id="20261012_v9_test_y8m_20990101_20990101-000000", session="20261012")
    w(lap / "inference_checks/batches.jsonl", "".join(json.dumps(x) + "\n" for x in
                                                      (line, dict(line, status="scored"), lipo_ic)))
    (lap / "inference_checks" / ic).mkdir(parents=True)
    # processing: master + one session MANIFEST + an excluded session folder
    w(proc / "imaged_grids_master.csv", "session_date,session_grid,inventory_id\n2099-01-01,1,2601\n2026-10-12,6,2700\n")
    w(proc / "20990101/5. Model + labeling/model_input_MANIFEST.csv", "output_name,grid,nm_per_px\ntile_a.png,1,1.4\n")
    w(proc / "20261012/5. Model + labeling/model_input_MANIFEST.csv", f"output_name,grid\n{LIPO[:-4]}.jpg,6\n")
    return lap, cache, proc, repo


def tree_digest(root):
    h = hashlib.md5()
    for p in sorted(Path(root).rglob("*")):
        h.update(str(p.relative_to(root)).encode())
        if p.is_file():
            h.update(p.read_bytes())
            h.update(str(p.stat().st_mtime_ns).encode())
    return h.hexdigest()


class TestFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        lap, cache, proc, repo = build_fixture(cls.root)
        cls.before = tree_digest(cls.root)
        cls.scan = validate.Scan(lap, cache, proc, repo).run()
        cls.rep = cls.scan.summary()
        cls.after = tree_digest(cls.root)
        cls.e = cls.rep["entities"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_read_only(self):
        self.assertEqual(self.before, self.after)

    def test_known_counts(self):
        e = self.e
        self.assertEqual((e["pull"]["found"], e["pull"]["linked"]), (1, 1))
        self.assertEqual(e["roboflow_image"]["linked"], 2)
        self.assertEqual(e["model"]["found"], 2)
        self.assertEqual(e["model"]["linked"], 1)                      # oldmodel: no source_run
        self.assertEqual(e["run"]["found"], 1)
        self.assertEqual(e["run"]["missing"], {"no job bundle (pre-bundle run)": 1})
        self.assertEqual(e["dataset"]["found"], 1)
        self.assertEqual(e["roboflow_export"]["found"], 2)
        self.assertEqual(e["roboflow_export"]["linked"], 1)            # the empty one has no COCO
        self.assertEqual(e["roboflow_export"]["excluded"], 1)

    def test_packages(self):
        p = self.e["review_package"]
        self.assertEqual(p["found"], 3)                                 # _control skipped
        self.assertEqual(p["linked"], 1)                                # pkgA only
        self.assertEqual(p["missing"]["data.js id shared by several folders"], 2)
        self.assertEqual(p["missing"]["folder name != data.js id"], 1)

    def test_label_versions(self):
        lv = self.e["label_version"]
        self.assertEqual((lv["found"], lv["linked"]), (2, 1))
        self.assertIn("pkg not found", lv["missing"])
        self.assertIn("version.json does not fit label_version/1", lv["missing"])
        ch = [r for k, r in self.rep["jsonl"].items() if k.endswith("changes.jsonl")][0]
        self.assertEqual((ch["lines"], ch["valid"]), (2, 1))           # op "explode" rejected

    def test_inference_checks_last_line_and_exclusion(self):
        ic = self.e["inference_check"]
        self.assertEqual((ic["found"], ic["linked"], ic["excluded"]), (1, 1, 1))
        bj = [r for k, r in self.rep["jsonl"].items() if k.endswith("batches.jsonl")][0]
        self.assertEqual((bj["lines"], bj["valid"], bj["legacy_mapped"], bj["excluded"]), (2, 2, 2, 1))

    def test_images(self):
        im = self.e["image"]
        # a-one_png, b-two_png (pull/tracking, keys normalised), tile_a_png (MANIFEST)
        self.assertEqual(im["found"], 3)
        self.assertEqual(im["linked"], 2)                               # a-one_png (tracker) + tile_a_png (folder)
        self.assertEqual(im["missing"]["no session date"], 1)
        self.assertNotIn("tile_a_png", im["examples"].get("no inventory Grid#", []))   # slot 1 -> 2601 via master

    def test_excluded_session_never_named(self):
        text = json.dumps(self.rep)
        for needle in (LIPO, "grid6_1", "20261012", "2700"):
            self.assertNotIn(needle, text)

    def test_cli_report_written_outside_protected(self):
        out = self.root / "out" / "report.json"
        out.parent.mkdir()
        lap, cache, proc, repo = (self.root / x for x in ("CryoAI-laptop", "CryoAI", "proc", "repo"))
        args = ["--laptop", str(lap), "--cache", str(cache), "--processing", str(proc), "--repo", str(repo),
                "--report", str(out)]
        with redirect_stdout(io.StringIO()) as buf:
            rc = validate.main(args)
        self.assertEqual(rc, 1)                                          # the bad changes.jsonl row
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["schema"], "validate_report/1")
        self.assertIn("review_package", buf.getvalue())
        with redirect_stdout(io.StringIO()):                             # overwriting its own report is allowed
            validate.main(args)


class TestReportRefusal(unittest.TestCase):
    def test_real_protected_roots_refused(self):
        for p in (r"C:\Users\pujan\Box\x\r.json", r"c:\users\PUJAN\box\r.json", r"C:\Users\pujan\Box",
                  r"C:\Users\pujan\Desktop\Cryo-image-processing\r.json",
                  r"C:\Users\pujan\Desktop\Cryo-image-processing\20261027\..\r.json",
                  r"C:\Users\pujan\Desktop\elsewhere\..\Cryo-image-processing\r.json"):
            with self.assertRaises(ValueError, msg=p):
                validate.check_report_path(p)

    def test_cli_refuses_before_scanning(self):
        with self.assertRaises(ValueError):
            validate.main(["--report", r"C:\Users\pujan\Box\r.json", "--laptop", "nonexistent"])

    def test_neighbour_names_allowed_and_non_report_files_refused(self):
        with tempfile.TemporaryDirectory() as t:
            prot = Path(t) / "Box"
            ok = Path(t) / "Box2" / "r.json"          # a sibling whose name merely starts with Box
            self.assertEqual(validate.check_report_path(ok, [prot]), Path(os.path.realpath(ok)))
            with self.assertRaises(ValueError):
                validate.check_report_path(prot / "r.json", [prot])
            other = Path(t) / "data.json"
            other.write_text('{"x": 1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                validate.check_report_path(other, [prot])


@unittest.skipUnless(validate.find_sibling("CryoAI-laptop"), "real artifacts not present")
class TestRealArtifacts(unittest.TestCase):
    def test_real_scan_is_read_only_and_finds_core_artifacts(self):
        lap = validate.find_sibling("CryoAI-laptop")
        watch = [p for p in lap.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".csv", ".js")]
        before = {p: p.stat().st_mtime_ns for p in watch}
        rep = validate.Scan(lap, validate.find_sibling("CryoAI"), validate.PROTECTED[1]).run().summary()
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in watch})
        self.assertGreaterEqual(rep["entities"]["pull"]["linked"], 1)
        self.assertGreaterEqual(rep["entities"]["label_version"]["found"], 1)
        self.assertNotIn("grid6_", json.dumps(rep))


if __name__ == "__main__":
    unittest.main()
