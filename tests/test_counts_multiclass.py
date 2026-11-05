"""Multi-class EV counting (models.yaml `classes:`) and load_predictions_from_model(max_det=...).

    python -m unittest tests.test_counts_multiclass -v        (from the repo root)

Known answer, hand-counted (make_v5_fixture): image A has a 3-layer EV stack, a standalone EV,
and a VLDL (class 1) outline enclosing an EV; image B has a VLDL alone, an EV, and one EV and
one VLDL below conf 0.5. With the EV class only:
  A: 5 EV outlines -> 3 EVs (1 multilayer, 2 inner layers), 1 non-EV
  B: 1 EV outline  -> 1 EV, 1 non-EV (the conf-0.3 VLDL is below the floor)
Without the filter the VLDL in A would be 'multilayer' and its EV 'inner-layer' (5 EVs, 2 multilayer).
No GPU or weights: YOLO is stubbed. Everything is written to temp dirs.
"""

import csv
import math
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data_utils import config  # noqa: E402
from inference import model_classes as mc  # noqa: E402

V5 = "v5_evnonev_y8m_20261029"
V4 = "v4_y8m_20260924"
SIZE = 400

# (class, conf, cx, cy, r) per image; centres and radii in px of a 400 x 400 image
FIXTURE = {
    "grid1_1_a.png": [(0, 0.95, 100, 100, 60), (0, 0.90, 100, 100, 40), (0, 0.85, 100, 100, 20),
                      (0, 0.80, 300, 300, 30),
                      (1, 0.90, 300, 100, 45), (0, 0.75, 300, 100, 25)],
    "grid1_2_b.png": [(1, 0.90, 100, 300, 30), (0, 0.80, 300, 300, 35),
                      (0, 0.30, 100, 100, 20), (1, 0.30, 250, 120, 25)],
}
WANT = {  # image -> (EV outlines >= 0.5, EVs, multilayer, inner, non-EV >= 0.5)
    "grid1_1_a.png": (5, 3, 1, 2, 1),
    "grid1_2_b.png": (1, 1, 0, 0, 1),
}


def make_v5_fixture(root: Path, model_id: str = V5):
    """Images + <root>/<model_id>/predictions/*.txt in the predict_models.py layout."""
    from PIL import Image
    images, preds = root / "images", root / model_id / "predictions"
    images.mkdir(parents=True, exist_ok=True)
    preds.mkdir(parents=True, exist_ok=True)
    for name, objs in FIXTURE.items():
        Image.new("L", (SIZE, SIZE), 128).save(images / name)
        lines = []
        for cls, conf, cx, cy, r in objs:
            pts = [(cx + r * math.cos(2 * math.pi * k / 64), cy + r * math.sin(2 * math.pi * k / 64))
                   for k in range(64)]
            lines.append(f"{cls} {conf:.4f} " + " ".join(f"{x / SIZE:.6f} {y / SIZE:.6f}" for x, y in pts))
        (preds / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return images, preds


class ModelClassesTest(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(mc.model_classes(V5), {0: "EV", 1: "non-EV"})
        self.assertTrue(mc.is_multiclass(V5))
        self.assertEqual(mc.ev_class_ids(V5), {0})
        self.assertEqual(mc.model_classes(V4), {})            # single-class: no classes: key
        self.assertIsNone(mc.ev_class_ids(V4))
        with self.assertRaises(KeyError):
            mc.model_classes("no_such_model")

    def test_resolve_model_id(self):
        self.assertEqual(mc.resolve_model_id(None, Path("x") / V5 / "predictions"), V5)
        self.assertEqual(mc.resolve_model_id(V4, Path("x") / V5 / "predictions"), V4)   # --model wins
        self.assertIsNone(mc.resolve_model_id(None, Path("x") / "run_7" / "predictions"))


@unittest.skipIf(config.cryoai_root(required=False) is None, "add_layer_roles needs the CryoAI folder")
class MulticlassPipelineTest(unittest.TestCase):
    """size_profile_from_predictions -> add_layer_roles on the known-answer fixture."""

    def run_pipeline(self, tmp: Path, model_args):
        images, preds = make_v5_fixture(tmp)
        morph = tmp / "morph"
        for cmd in ([sys.executable, "-m", "inference.size_profile_from_predictions", "--predictions-dir",
                     str(preds), "--images-dir", str(images), "--output-dir", str(morph),
                     "--pixel-size", "1.0", "--no-plot"] + model_args,
                    [sys.executable, "-m", "analysis.add_layer_roles", "--morphology-csv",
                     str(morph / "morphology_all.csv"), "--predictions-dir", str(preds),
                     "--images-dir", str(images)] + model_args):
            r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with open(morph / "morphology_all_roles.csv", newline="", encoding="utf-8") as f:
            roles = list(csv.DictReader(f))
        with open(morph / "per_image_summary.csv", newline="", encoding="utf-8") as f:
            per_image = {r["image"]: r for r in csv.DictReader(f)}
        return roles, per_image

    def check(self, roles, per_image):
        for name, (n_out, n_ev, n_ml, n_inner, n_nonev) in WANT.items():
            rs = [r for r in roles if r["image"] == name]
            self.assertEqual(len(rs), n_out, name)
            self.assertEqual(sum(r["role"] != "inner-layer" for r in rs), n_ev, name)
            self.assertEqual(sum(r["role"] == "multilayer" for r in rs), n_ml, name)
            self.assertEqual(sum(r["role"] == "inner-layer" for r in rs), n_inner, name)
            self.assertEqual(int(per_image[name]["n_nonev_accepted"]), n_nonev, name)

    def test_explicit_model(self):
        with tempfile.TemporaryDirectory() as d:
            self.check(*self.run_pipeline(Path(d), ["--model", V5]))

    def test_model_from_predictions_folder(self):
        with tempfile.TemporaryDirectory() as d:
            self.check(*self.run_pipeline(Path(d), []))

    def test_single_class_keeps_everything(self):
        """Control: the same files read as a single-class model count the VLDL as an EV."""
        with tempfile.TemporaryDirectory() as d:
            roles, per_image = self.run_pipeline(Path(d), ["--model", V4])
        a = [r for r in roles if r["image"] == "grid1_1_a.png"]
        self.assertEqual(len(a), 6)
        self.assertEqual(sum(r["role"] == "multilayer" for r in a), 2)   # the VLDL became 'multilayer'
        self.assertNotIn("n_nonev_accepted", per_image["grid1_1_a.png"])  # single-class columns unchanged


class _Arr:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Box:
    def __init__(self, conf, cls):
        self.conf, self.cls = _Arr([conf]), _Arr([cls])


class _FakeYOLO:
    instances = []

    def __init__(self, path):
        self.path, self.calls, self.names = path, [], {0: "EV"}
        _FakeYOLO.instances.append(self)

    def predict(self, source, **kw):
        self.calls.append(kw)
        m = np.zeros((32, 32), np.float32)
        m[8:24, 8:24] = 1
        return [types.SimpleNamespace(masks=types.SimpleNamespace(data=[_Arr(m)]), boxes=[_Box(0.9, 0)])]


class LoadPredictionsFromModelTest(unittest.TestCase):
    """max_det reaches predict(), weights load once, and calculate_segmentation_metrics no longer
    raises TypeError (it passed max_det= to a function that did not take it)."""

    def setUp(self):
        self._saved = sys.modules.get("ultralytics")
        sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=_FakeYOLO)
        from training import train_yolo
        self.ty = train_yolo
        train_yolo._YOLO_CACHE.clear()
        _FakeYOLO.instances.clear()
        self.tmp = tempfile.TemporaryDirectory()
        from PIL import Image
        self.img = Path(self.tmp.name) / "img" / "a.png"
        self.img.parent.mkdir()
        Image.new("L", (64, 64), 0).save(self.img)
        (Path(self.tmp.name) / "lbl").mkdir()
        (Path(self.tmp.name) / "lbl" / "a.txt").write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")

    def tearDown(self):
        self.ty._YOLO_CACHE.clear()
        if self._saved is None:
            sys.modules.pop("ultralytics", None)
        else:
            sys.modules["ultralytics"] = self._saved
        self.tmp.cleanup()

    def test_max_det_and_cache(self):
        ty = self.ty
        masks, confs = ty.load_predictions_from_model("w.pt", str(self.img), 64, 0.25, 0.7, "cpu", max_det=1000)
        self.assertEqual((len(masks), confs), (1, [0.9]))
        self.assertEqual(masks[0].shape, (64, 64))
        # hitl_annotator's keyword call, without max_det: the module default (300)
        ty.load_predictions_from_model(model_path="w.pt", img_path=str(self.img), imgsz=64, conf=0.25,
                                       iou=0.7, device="cpu")
        self.assertEqual(len(_FakeYOLO.instances), 1)                  # one load for both calls
        self.assertEqual([c["max_det"] for c in _FakeYOLO.instances[0].calls], [1000, 300])

    def test_calculate_segmentation_metrics(self):
        d = Path(self.tmp.name)
        m = self.ty.calculate_segmentation_metrics("w.pt", str(d / "img"), str(d / "lbl"), 64, 0.25, 0.7,
                                                   "cpu", 0.5, max_det=500)
        self.assertEqual(_FakeYOLO.instances[0].calls[0]["max_det"], 500)
        self.assertEqual(m["object_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
