"""Tests for inference/inference_check.py, the web_qc Roboflow classes and training/make_job_bundle.py.

    python -m unittest tests.test_inference_check -v        (from the repo root)

The real-session test reads C:/Users/pujan/Desktop/Cryo-image-processing/20261027 (read-only)
and is skipped when that folder is absent. Everything is written to temp dirs.
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from inference import inference_check as ic  # noqa: E402
from tests.qc_fixture import make_fake_tiles, write_reviews  # noqa: E402
from training import make_job_bundle as mjb  # noqa: E402

SESSION = Path(r"C:\Users\pujan\Desktop\Cryo-image-processing\20261027\5. Model + labeling")
ROBOFLOW = REPO / "annotation" / "web_qc" / "classes_roboflow.yaml"


def fake_tiles(grids=(("1", 4), ("2", 6), ("3", 1)), per_montage=9, defocus=True):
    """Tiles of 3x3 montages; grid 1 has no defocus, the others have a spread."""
    tiles, rng = [], random.Random(0)
    for g, n_mont in grids:
        for m in range(n_mont):
            d = None if (g == "1" or not defocus) else round(-2 - 0.5 * m - rng.random(), 2)
            for k in range(per_montage):
                pos = f"r{k // 3}c{k % 3}"
                tiles.append({"name": f"g{g}__grid{g}_{m}_{pos}_t0{k}.png", "grid": g,
                              "montage": f"grid{g}_{m}", "position": pos, "defocus_um": d})
    ic.assign_defocus_bins(tiles, 3)
    return tiles


LEVELS = ("grid", "defocus_bin")


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        raise AssertionError(f"{args} failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    return r.stdout


class TestSampler(unittest.TestCase):
    def test_seed_reproducible_and_order_independent(self):
        tiles = fake_tiles()
        a = [t["name"] for t in ic.stratified_sample(tiles, 20, seed=7, levels=LEVELS)]
        shuffled = tiles[:]
        random.Random(3).shuffle(shuffled)
        b = [t["name"] for t in ic.stratified_sample(shuffled, 20, seed=7, levels=LEVELS)]
        c = [t["name"] for t in ic.stratified_sample(tiles, 20, seed=8, levels=LEVELS)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_every_stratum_when_n_at_least_strata(self):
        tiles = fake_tiles()
        strata = {ic.stratum(t, LEVELS) for t in tiles}
        self.assertGreaterEqual(len(strata), 5)   # 1/na, 2/dz*, 3/dz*
        for n in range(len(strata), len(strata) + 12):
            for seed in range(5):
                got = {ic.stratum(t, LEVELS) for t in ic.stratified_sample(tiles, n, seed, LEVELS)}
                self.assertEqual(got, strata, f"n={n} seed={seed}")

    def test_no_duplicates_and_caps_at_pool(self):
        tiles = fake_tiles()
        for n in (1, 10, 50, len(tiles), len(tiles) + 5):
            names = [t["name"] for t in ic.stratified_sample(tiles, n, 1, LEVELS)]
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(len(names), min(n, len(tiles)))

    def test_spread_over_grids_montages_positions(self):
        tiles = fake_tiles()
        pick = ic.stratified_sample(tiles, 18, 2, LEVELS)
        grids = Counter(t["grid"] for t in pick)
        self.assertEqual(grids, {"1": 6, "2": 6, "3": 6})   # equal shares, whatever the grid size
        two = ic.stratified_sample(tiles, 30, 2, LEVELS)    # grid 3 has only 9 tiles: capped
        self.assertEqual(Counter(t["grid"] for t in two)["3"], 9)
        # no montage gets a 2nd tile while another montage of its stratum has none
        for s in {ic.stratum(t, LEVELS) for t in pick}:
            in_s = [t for t in tiles if ic.stratum(t, LEVELS) == s]
            used = Counter(t["montage"] for t in pick if ic.stratum(t, LEVELS) == s)
            all_m = {t["montage"] for t in in_s}
            if max(used.values()) > 1:
                self.assertEqual(set(used), all_m, s)
        # one tile per montage here -> positions all differ within a grid where possible
        self.assertGreater(len({t["position"] for t in pick}), 5)

    def test_never_all_from_one_montage(self):
        tiles = fake_tiles(grids=(("1", 3),), defocus=False)
        ic.assign_defocus_bins(tiles, 3)
        pick = ic.stratified_sample(tiles, 3, 0, ("grid",))
        self.assertEqual(len({t["montage"] for t in pick}), 3)

    def test_defocus_bins(self):
        tiles = [{"defocus_um": v} for v in (-6, -5, -4, -3, -2, -1, None)]
        bins = ic.assign_defocus_bins(tiles, 3)
        self.assertEqual(bins, {"dz1": [-6, -5], "dz2": [-4, -3], "dz3": [-2, -1]})
        self.assertEqual([t["defocus_bin"] for t in tiles],
                         ["dz1", "dz1", "dz2", "dz2", "dz3", "dz3", "na"])

    def test_registered_conf(self):
        reg = ic.registry()
        self.assertEqual(ic.registered_conf(reg["v5_evnonev_y8m_20261029"]), (0.5, "models.yaml note"))
        self.assertEqual(ic.registered_conf(reg["v4_y8m_20260924"])[0], 0.5)


@unittest.skipUnless(SESSION.exists(), "real 20261027 session not on this PC")
class TestStageReal(unittest.TestCase):
    def test_stage_existing_predictions(self):
        preds = SESSION / "predictions" / "v4_y8m_20260924" / "predictions"
        with tempfile.TemporaryDirectory() as tmp:
            out = run("inference/inference_check.py", "stage", "--tiles", SESSION / "model_input",
                      "--model", "v4_y8m_20260924", "--predictions-dir", preds, "--n", 24,
                      "--seed", 1, "--out", tmp, "--batch-id", "t")
            self.assertIn("python -m annotation.web_qc.server", out)
            b = Path(tmp) / "t"
            batch = json.loads((b / "BATCH.json").read_text(encoding="utf-8"))
            rows = ic.read_csv(b / "sample.csv")
            self.assertEqual(len(rows), 24)
            self.assertEqual(len(list((b / "images").iterdir())), 24)
            self.assertEqual(len(list((b / "predictions").glob("*.txt"))), 24)
            self.assertEqual(set(batch["strata"]), {"grid=1/defocus_bin=na", "grid=2/defocus_bin=dz1",
                                                    "grid=2/defocus_bin=dz2", "grid=2/defocus_bin=dz3"})
            self.assertEqual(sum(batch["strata"].values()), 24)
            self.assertGreaterEqual(batch["n_montages"], 20)
            self.assertTrue(batch["field_table"])
            img = next((b / "images").iterdir())
            self.assertTrue(os.path.samefile(img, SESSION / "model_input" / img.name)
                            or batch["images"].get("copied"))
            log = (Path(tmp) / "batches.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(log), 1)
            self.assertTrue(set(ic.LOG_FIELDS) <= set(json.loads(log[0])))
            # web_qc seeds the staged tile with the Roboflow classes
            from annotation.web_qc.server import Store, load_classes
            store = Store(sorted((b / "images").iterdir()), b / "predictions", b / "review_probe",
                          load_classes(ROBOFLOW), 0.5)
            objs = store.get(img.name)["objects"]
            self.assertTrue(objs and {o["cls"] for o in objs} <= {"EV", "MV", "inner", "non-EV"})

    def test_refuses_out_inside_session(self):
        r = subprocess.run([sys.executable, "inference/inference_check.py", "stage", "--tiles",
                            str(SESSION / "model_input"), "--model", "v4_y8m_20260924",
                            "--predictions-dir", str(SESSION / "predictions" / "v4_y8m_20260924"),
                            "--out", str(SESSION / "_must_not_exist")],
                           cwd=REPO, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((SESSION / "_must_not_exist").exists())


class TestScore(unittest.TestCase):
    """Known answer (tests/qc_fixture.py): per image at conf 0.5, 1 TP, 1 FP, 1 FN, plus one
    non-EV model outline that the reviewer kept as non-EV and must not count as an EV FP."""

    def test_score_fake_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, preds = make_fake_tiles(Path(tmp) / "src")
            run("inference/inference_check.py", "stage", "--tiles", images, "--model",
                "v5_evnonev_y8m_20261029", "--predictions-dir", preds, "--n", 2,
                "--out", Path(tmp) / "out", "--batch-id", "fake", "--session", "fake")
            b = Path(tmp) / "out" / "fake"
            write_reviews(b / "images", b / "predictions", b / "review", ROBOFLOW)
            run("inference/inference_check.py", "score", b, "--reviewer", "tester")

            by_t = {r["conf_cut"]: r for r in ic.read_csv(b / "review/metrics/by_threshold.csv")}
            at = by_t["0.5"]
            self.assertEqual((int(at["tp"]), int(at["fp"]), int(at["fn"])), (2, 2, 2))
            result = (b / "RESULT.md").read_text(encoding="utf-8")
            self.assertIn("| 0.500 | 0.500 | 0.500 | 2 | 2 | 2 |", result)
            self.assertIn("| non-EV | no | 2 | 0 | 0 | 2 |", result)       # kept as seeded
            self.assertIn("| EV | yes | 2 | 0 | 2 | 4 |", result)          # 2 kept + 2 drawn
            self.assertTrue((b / "review/export/_annotations.coco.json").exists())
            coco = json.loads((b / "review/export/_annotations.coco.json").read_text())
            self.assertEqual([c["name"] for c in coco["categories"]],
                             ["EV", "Low conf EV", "MV", "inner", "Incomplete", "Overlapping", "VLDL", "non-EV"])
            log = [json.loads(l) for l in (Path(tmp) / "out/batches.jsonl").read_text().splitlines()]
            self.assertEqual([l["status"] for l in log], ["staged", "scored"])
            self.assertEqual(log[1]["reviewer"], "tester")
            self.assertAlmostEqual(log[1]["metrics"]["f1"], 0.5)
            self.assertIs(log[1]["partial"], False)
            self.assertEqual(log[1]["n_done"], 2)
            self.assertFalse(any(os.path.samefile(p, images / p.name) for p in (b / "images").iterdir()))

    def test_default_classes_unchanged(self):
        """classes.yaml (no model_class_to_class): the non-EV outline is seeded and scored as EV."""
        with tempfile.TemporaryDirectory() as tmp:
            images, preds = make_fake_tiles(Path(tmp))
            write_reviews(images, preds, Path(tmp) / "review", REPO / "annotation/web_qc/classes.yaml")
            run("-m", "annotation.web_qc.export", "--images-dir", images, "--predictions-dir", preds,
                "--review-dir", Path(tmp) / "review")
            at = {r["conf_cut"]: r for r in ic.read_csv(
                Path(tmp) / "review/metrics/by_threshold.csv")}["0.5"]
            self.assertEqual((int(at["tp"]), int(at["fp"]), int(at["fn"])), (4, 2, 2))


def stage_fake(tmp, *extra, grid_per_image=False, batch_id="fake"):
    """Stage the fixture tiles into <tmp>/out/<batch_id>; return (batch dir, out root, completed process)."""
    images, preds = make_fake_tiles(Path(tmp) / "src", grid_per_image=grid_per_image)
    args = ["inference/inference_check.py", "stage", "--tiles", images, "--model", "v5_evnonev_y8m_20261029",
            "--n", 2, "--out", Path(tmp) / "out", "--batch-id", batch_id, "--session", "fake", *extra]
    if "--device" not in extra:
        args += ["--predictions-dir", preds]
    r = subprocess.run([sys.executable, *map(str, args)], cwd=REPO, capture_output=True, text=True)
    return Path(tmp) / "out" / batch_id, Path(tmp) / "out", r


def read_lines(out):
    return (out / "batches.jsonl").read_text(encoding="utf-8").splitlines()


class TestScorePartial(unittest.TestCase):
    def test_zero_done_is_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, out, r = stage_fake(tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            write_reviews(b / "images", b / "predictions", b / "review", ROBOFLOW)
            for p in (b / "review").glob("*.json"):
                d = json.loads(p.read_text(encoding="utf-8"))
                d["done"] = False
                p.write_text(json.dumps(d), encoding="utf-8")
            r = subprocess.run([sys.executable, "inference/inference_check.py", "score", str(b)],
                               cwd=REPO, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("no images marked Done", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertEqual([json.loads(l)["status"] for l in read_lines(out)], ["staged"])

    def test_partial_batch_keeps_empty_strata(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, out, r = stage_fake(tmp, grid_per_image=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            write_reviews(b / "images", b / "predictions", b / "review", ROBOFLOW)
            p = next((b / "review").glob("g2_*.json"))
            d = json.loads(p.read_text(encoding="utf-8"))
            d["done"] = False
            p.write_text(json.dumps(d), encoding="utf-8")
            run("inference/inference_check.py", "score", b)
            last = json.loads(read_lines(out)[-1])
            self.assertEqual((last["status"], last["partial"], last["n_done"], last["n"]), ("scored", True, 1, 2))
            self.assertEqual(last["metrics"]["by_stratum"]["grid=2"], {"sampled": 1, "n_done": 0})
            result = (b / "RESULT.md").read_text(encoding="utf-8")
            self.assertIn("| grid=2 | 1 | 0 | – | – | – | – | – | – |", result)
            self.assertIn("| grid=1 | 1 | 1 | 0.50 | 0.50 | 0.50 | 1 | 1 | 1 |", result)
            self.assertIn("**partial**", result)


class TestLogRobustness(unittest.TestCase):
    def test_truncated_line_then_log_and_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, out, r = stage_fake(tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            log = out / "batches.jsonl"
            raw = log.read_bytes()
            log.write_bytes(raw[:-15])                       # cut the line short, newline lost
            run("inference/inference_check.py", "log", b, "--status", "reviewed", "--reviewer", "amy")
            lines = read_lines(out)
            self.assertEqual(len(lines), 2)                  # new record on its own line
            self.assertEqual(json.loads(lines[1])["status"], "reviewed")
            r = subprocess.run([sys.executable, "inference/inference_check.py", "list", "--out", str(out)],
                               cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("line 1 is not a valid record", r.stderr)
            self.assertRegex(r.stdout, r"fake\s+reviewed\s+n=2\s+amy")


class TestGuards(unittest.TestCase):
    def test_protected_roots_any_depth(self):
        tiles = Path(tempfile.gettempdir()) / "tiles"
        home = Path.home()
        for bad in (home / "Box", home / "Box" / "lab" / "x",
                    home / "Desktop" / "Cryo-image-processing",
                    home / "Desktop" / "Cryo-image-processing" / "20261027" / "a" / "b",
                    home / "Desktop" / "Cryo-image-processing" / "notes"):
            self.assertIsNotNone(ic.out_root_problem(bad, tiles), bad)
        if os.name == "nt":   # case-insensitive file system
            self.assertIsNotNone(ic.out_root_problem(home / "desktop" / "CRYO-image-processing" / "x", tiles))
        self.assertIsNone(ic.out_root_problem(home / "Boxes" / "x", tiles))       # prefix only, not under
        self.assertIsNone(ic.out_root_problem(Path(tempfile.gettempdir()) / "ic", tiles))

    def test_session_folder_and_custom_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            tiles = Path(tmp) / "20261027" / "5. Model + labeling" / "model_input"
            self.assertIn("session folder", ic.out_root_problem(Path(tmp) / "20261027" / "q", tiles))
            self.assertIsNone(ic.out_root_problem(Path(tmp) / "elsewhere", tiles))
            self.assertIsNotNone(ic.out_root_problem(Path(tmp) / "p" / "x", tiles, protected=[Path(tmp) / "p"]))

    def test_refuses_protected_out_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path.home() / "Box" / "_inference_check_must_not_exist"
            images, preds = make_fake_tiles(Path(tmp) / "src")
            r = subprocess.run([sys.executable, "inference/inference_check.py", "stage", "--tiles", str(images),
                                "--predictions-dir", str(preds), "--out", str(bad)],
                               cwd=REPO, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("read-only", r.stderr)
            self.assertFalse(bad.exists())

    def test_path_length_guard(self):
        self.assertEqual(ic.longest_output_path(Path("C:/o/b"), "m", ["ab", "abcd"]),
                         len(str(Path("C:/o/b/p/m/predictions"))) + 4 + 5)
        if os.name != "nt":
            self.skipTest("260-character limit applies on Windows only")
        with tempfile.TemporaryDirectory() as tmp:
            b, out, r = stage_fake(tmp, batch_id="b" * 200)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("Windows limit 260", r.stderr)
            self.assertFalse(out.exists() and any(out.iterdir()))   # nothing created, nothing logged


class TestManifestWarnings(unittest.TestCase):
    def test_warns_on_unusable_manifest_and_names(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "plain_tile.png").write_bytes(b"x")
            (d / "MANIFEST.csv").write_text("foo,grid\nbar,1\n", encoding="utf-8")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                t = ic.tile_table(d, d / "MANIFEST.csv", None)
            msg = err.getvalue()
            self.assertIn("none of the name columns", msg)
            self.assertIn("grid not in MANIFEST or file name", msg)
            self.assertEqual(t[0]["grid"], "na")


class TestStageFailure(unittest.TestCase):
    def test_predict_failure_removes_batch_and_logs_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, out, r = stage_fake(tmp, "--device", "no_such_device")
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse(b.exists())
            lines = [json.loads(l) for l in read_lines(out)]
            self.assertEqual([l["status"] for l in lines], ["failed"])
            self.assertIn("CalledProcessError", lines[0]["error"])
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["batches.jsonl"])


class TestJobBundle(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "my_dataset"
            for s in ("train", "val", "test"):
                (ds / "images" / s).mkdir(parents=True)
                (ds / "labels" / s).mkdir(parents=True)
                (ds / "images" / s / f"{s}_0.png").write_bytes(b"\x89PNG fake " + s.encode())
                (ds / "labels" / s / f"{s}_0.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n")
            (ds / "dataset.yaml").write_text(
                "# YOLO dataset config\npath: C:/Users/ML-2619/Desktop/CryoAI/training outputs/my_dataset\n"
                "train: images/train\nval: C:/Users/ML-2619/Desktop/CryoAI/training outputs/my_dataset/images/val\n"
                "test: images/test\n\nnames:\n  0: EV\n  1: non-EV\n", encoding="utf-8")
            out = run("training/make_job_bundle.py", "--dataset-dir", ds, "--out", Path(tmp) / "out",
                      "--", "--single-run", "--model-size", "m", "--batch", "8", "--rect", "0",
                      "--weights", "models/v4_y8m_20260924/best.pt")
            self.assertIn("path: C:/Users/ML-2619", out)
            zips = list((Path(tmp) / "out").glob("*.zip"))
            self.assertEqual(len(zips), 1)
            self.assertTrue(zips[0].with_name(zips[0].stem + "_README.md").exists())

            desk = Path(tmp) / "desktop" / "CryoAI" / "training outputs"
            with zipfile.ZipFile(zips[0]) as z:
                z.extractall(desk)
            y = desk / "my_dataset" / "dataset.yaml"
            text = y.read_text(encoding="utf-8")
            self.assertNotIn("\npath:", text)
            self.assertNotIn("ML-2619/Desktop/CryoAI/training outputs/my_dataset/images", text)
            resolved = mjb.resolve_like_ultralytics(y)
            self.assertEqual(set(resolved), {"train", "val", "test"})
            for s, p in resolved.items():
                self.assertEqual(p, (desk / "my_dataset" / "images" / s).resolve())
                self.assertTrue(any(p.iterdir()))
                self.assertTrue((desk / "my_dataset" / "labels" / s / f"{s}_0.txt").exists())
            job = json.loads((desk / "my_dataset" / "JOB.json").read_text())
            self.assertEqual(job["run_experiments_args"][:2], ["--dataset", "my_dataset"])
            self.assertIn("models/v4_y8m_20260924/best.pt", job["run_experiments_args"])
            readme = (desk / "my_dataset" / "README.md").read_text(encoding="utf-8")
            self.assertIn('run_experiments.py --dataset "my_dataset" --single-run', readme)

    def test_rejects_unknown_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "d"
            (ds / "images").mkdir(parents=True)
            (ds / "labels").mkdir()
            (ds / "dataset.yaml").write_text("train: images/train\nval: images/val\nnames:\n  0: EV\n")
            r = subprocess.run([sys.executable, "training/make_job_bundle.py", "--dataset-dir", str(ds),
                                "--out", str(Path(tmp) / "o"), "--", "--single-run", "--epoch", "5"],
                               cwd=REPO, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("--epoch", r.stderr)

    def _dataset(self, tmp):
        ds = Path(tmp) / "d"
        for sp in ("train", "val"):
            (ds / "images" / sp).mkdir(parents=True)
            (ds / "labels" / sp).mkdir(parents=True)
            (ds / "images" / sp / "a.png").write_bytes(b"x")
        (ds / "dataset.yaml").write_text("train: images/train\nval: images/val\nnames:\n  0: EV\n")
        return ds

    def test_weights_equals_form_is_packed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = self._dataset(tmp)
            w = Path(tmp) / "w.pt"
            w.write_bytes(b"weights")
            run("training/make_job_bundle.py", "--dataset-dir", ds, "--out", Path(tmp) / "o",
                "--", "--single-run", f"--weights={w}")
            z = zipfile.ZipFile(next((Path(tmp) / "o").glob("*.zip")))
            self.assertIn("d/weights/w.pt", z.namelist())
            job = json.loads(z.read("d/JOB.json"))
            self.assertEqual(job["run_experiments_args"][-2:], ["--weights", "../CryoAI/training outputs/d/weights/w.pt"])
            self.assertIn("8.4.x", z.read("d/README.md").decode("utf-8"))
            z.close()

    def test_weights_directory_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = self._dataset(tmp)
            r = subprocess.run([sys.executable, "training/make_job_bundle.py", "--dataset-dir", str(ds),
                                "--out", str(Path(tmp) / "o"), "--", "--weights", str(Path(tmp))],
                               cwd=REPO, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("is a folder", r.stderr)
            self.assertFalse(list((Path(tmp) / "o").glob("*.zip")))


if __name__ == "__main__":
    unittest.main()
