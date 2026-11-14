"""jobs/predict_set.py adopt (round 14, contract C6): an outside prediction run registered as a prediction set.

A synthetic run (temp) with rectangles, so every mask area is known without the code under test: line 0 an EV at
conf 0.9, line 1 an EV at conf 0.3 (below the run's conf: no particle), line 2 a non-EV, line 3 an EV, line 4 blank.
The run's particle ids are then 0 -> line 0, 1 -> line 3 (not line 1: the known answer that tells a right mapping
from particle_id == line), 2 -> line 2 (non-EVs after the EVs).

  python -m unittest discover -s jobs/tests -p test_predict_adopt.py -v
"""
import contextlib
import csv
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from jobs import predict_set as P  # noqa: E402
from registry import store  # noqa: E402

MODEL = "v5_evnonev_y8m_20261029"
W, H = 200, 150                     # the run's model-input size; the source images are 2x
RECTS = [(0, 0.9, (10, 10, 59, 39)), (0, 0.3, (100, 20, 129, 49)), (1, 0.8, (150, 100, 189, 139)),
         (0, 0.7, (70, 60, 109, 119))]


def rect_line(k, conf, r):
    x0, y0, x1, y1 = r
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    # +0.5 px: int(c * W) truncates back onto the integer corner whatever the float rounding
    return f"{k} {conf} " + " ".join(f"{(x + 0.5) / W:.6f} {(y + 0.5) / H:.6f}" for x, y in pts)


def area(r):
    x0, y0, x1, y1 = r
    return (x1 - x0 + 1) * (y1 - y0 + 1)


def rcsv(path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def snapshot(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


class Adopt(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="adopt_"))
        t = self.tmp
        self.run, self.imgs, self.root, self.reg = t / "run", t / "imgs", t / "predict", t / "registry"
        for d in (self.run / "data" / "pa" / "SetA", self.imgs, self.root, self.reg):
            d.mkdir(parents=True)
        man = {"run_id": "20991231_0000", "config": {
            "model": {"id": MODEL, "classes": {"0": "ev", "1": "non_ev"}, "imgsz": W},
            "params": {"conf": 0.5}, "pixel_size": {"nm_per_px_5760": 0.3681},
            "applications": {"Set A": {"path": "elsewhere"}, "exclude_images": {"Set A": ["c.png"]}}}}
        (self.run / "run_manifest.json").write_text(json.dumps(man), encoding="utf-8")
        d2 = [{"set": "Set A", "image_id": n, "file": f"{n}.png", "width_px": W, "height_px": H,
               "nm_per_px": round(0.3681 * 5760 / W, 4)} for n in ("a", "b")]
        d2.append({"set": "Set A", "image_id": "c", "file": "c.png", "width_px": "", "height_px": "", "nm_per_px": ""})
        self._csv(self.run / "data" / "D2_images.csv", d2)
        lines = [rect_line(k, c, r) for k, c, r in RECTS] + [""]
        d1 = []
        for n in ("a", "b"):
            (self.run / "data" / "pa" / "SetA" / f"{n}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            Image.new("L", (2 * W, 2 * H), 128).save(self.imgs / f"{n}.png")
            for pid, r in enumerate([RECTS[0][2], RECTS[3][2], RECTS[2][2]]):
                d1.append({"set": "Set A", "image_id": n, "particle_id": pid, "mask_area_px": area(r)})
        self._csv(self.run / "data" / "D1_particles.csv", d1)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def adopt(self, dry=True, **kw):
        args = dict(sets=None, root=self.root, registry_root=self.reg, dry_run=dry, via=None)
        args.update(kw)
        with contextlib.redirect_stdout(io.StringIO()):
            return P.adopt(self.run, MODEL, [self.imgs], "synthetic run", "pujan", "pujan", **args)

    def test_mapping_known_answer(self):
        parts = P.run_particle_lines(self.run / "data" / "pa" / "SetA" / "a.txt", W, H, 0.5, {0: "ev", 1: "non_ev"})
        self.assertEqual([(p[0], p[1]) for p in parts], [(0, 0), (1, 3), (2, 2)])
        self.assertEqual([p[4] for p in parts], [area(RECTS[0][2]), area(RECTS[3][2]), area(RECTS[2][2])])

    def test_dry_run_writes_nothing(self):
        before = snapshot(self.tmp)
        s = self.adopt()
        self.assertEqual(snapshot(self.tmp), before)
        self.assertEqual((s["n_images"], s["n_particles"], s["n_area_match"]), (2, 6, 6))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = P.main(["adopt", "--from", str(self.run), "--model", MODEL, "--images", str(self.imgs), "--name", "x",
                         "--owner", "pujan", "--root", str(self.root), "--registry", str(self.reg)])
        self.assertEqual(rc, 0)
        self.assertIn("DRY RUN, nothing written", out.getvalue())
        self.assertEqual(snapshot(self.tmp), before)

    def test_do_temp_root(self):
        run_before = snapshot(self.run) | {f"img/{k}": v for k, v in snapshot(self.imgs).items()}
        ps = self.adopt(dry=False)
        self.assertEqual(snapshot(self.run) | {f"img/{k}": v for k, v in snapshot(self.imgs).items()}, run_before)
        req = json.loads((ps / "request.json").read_text(encoding="utf-8"))
        self.assertEqual(req["source"]["kind"], "adopted")
        self.assertEqual(req["source"]["run"], "20991231_0000")
        self.assertEqual(json.loads((ps / "status.json").read_text(encoding="utf-8"))["state"], "done")
        rows = rcsv(ps / "images.csv")
        self.assertEqual([(r["image"], r["orig_name"], r["sample_label"]) for r in rows],
                         [("0001", "a.png", "Set A"), ("0002", "b.png", "Set A")])
        self.assertAlmostEqual(float(rows[0]["nm_per_px"]), 0.3681 * 5760 / (2 * W))
        for r, n in zip(rows, ("a", "b")):
            self.assertEqual((ps / "p" / MODEL / "predictions" / f"{r['image']}.txt").read_bytes(),
                             (self.run / "data" / "pa" / "SetA" / f"{n}.txt").read_bytes())
            self.assertEqual((ps / r["file"]).read_bytes(), (self.imgs / f"{n}.png").read_bytes())
        pl = rcsv(ps / "adopted" / "particle_lines.csv")
        self.assertEqual([(r["image"], r["particle_id"], r["line"], r["area_match"]) for r in pl][:3],
                         [("0001", "0", "0", "1"), ("0001", "1", "3", "1"), ("0001", "2", "2", "1")])
        files = rcsv(ps / "adopted" / "source_files.csv")
        self.assertEqual(len(files), 5)
        rec = store.current("prediction_set", self.reg)[ps.name]
        self.assertEqual((rec["status"], rec["source"]["kind"], rec["n_images"]), ("done", "adopted", 2))
        self.assertEqual(store.check_record(rec, store.schemas()), [])
        # the object set builder maps a run particle id to its line through particle_lines.csv
        from annotation import object_set as OS
        items = self.tmp / "items.csv"
        items.write_text("image_id,particle_id\na,1\nb,2\n", encoding="utf-8")
        plan = OS.plan(ps, items, "ls_lamellarity_v1", image_col="image_id", particle_col="particle_id")
        self.assertEqual(plan["keys"], [("0001", 3), ("0002", 2)])

    def test_refusals(self):
        # a run particle whose area doesn't match: reported by the dry run, refused by --do
        d1 = rcsv(self.run / "data" / "D1_particles.csv")
        d1[1]["mask_area_px"] = "7"
        self._csv(self.run / "data" / "D1_particles.csv", d1)
        s = self.adopt()
        self.assertEqual((s["n_particles"], s["n_area_match"]), (6, 5))
        self.assertEqual(s["area_mismatch"], [("a", 1)])
        with self.assertRaises(P.PredictSetError):
            self.adopt(dry=False)
        self.assertEqual([p.name for p in self.root.iterdir() if not p.name.startswith(".")], [])
        with self.assertRaises(P.PredictSetError) as cm:
            P.adopt(self.run, "no_such_model", [self.imgs], "x", "pujan", "pujan", root=self.root,
                    registry_root=self.reg)
        self.assertIn("not the run's model", str(cm.exception))
        (self.imgs / "sub").mkdir()
        shutil.copy(self.imgs / "a.png", self.imgs / "sub" / "a.png")
        with self.assertRaises(P.PredictSetError) as cm:
            self.adopt()
        self.assertIn("candidate images", str(cm.exception))
        shutil.rmtree(self.imgs / "sub")
        (self.imgs / "b.png").unlink()
        with self.assertRaises(P.PredictSetError) as cm:
            self.adopt()
        self.assertIn("image not found", str(cm.exception))
        Image.new("L", (2 * W, 2 * W), 128).save(self.imgs / "b.png")          # wrong shape
        with self.assertRaises(P.PredictSetError) as cm:
            self.adopt()
        self.assertIn("shape", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
