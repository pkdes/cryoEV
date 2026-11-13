"""Stand-in for training/train_yolo.validate_yolo_model in the eval-job tests (run_job hooks eval_validate =
"<this file>:validate"). Loaded by the --eval-child process; no torch / Ultralytics.

Records what it was called with, and what it found on disk, to $FAKE_VAL_RECORD (JSON): the kwargs, the weights'
md5, the dataset.yaml text, the test images under the yaml's folder, and whether train_yolo / ultralytics were
imported in this process. $FAKE_VAL_FAIL=1 raises instead (a val that crashes). Returns an object shaped like
Ultralytics SegmentMetrics: .seg / .box (map, map50, ap_class_index, class_result(i)), .names.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

MASK = {0: 0.70, 1: 0.50}          # per-class mask mAP50-95 (EV, non-EV)
BOX = {0: 0.80, 1: 0.60}


class _Metric:
    def __init__(self, per, map50):
        self._per, self.map50 = per, map50
        self.ap_class_index = sorted(per)
        self.map = sum(per.values()) / len(per)

    def class_result(self, i):
        c = self.ap_class_index[i]
        return (0.9, 0.8, min(1.0, self._per[c] + 0.2), self._per[c])


class _Results:
    def __init__(self):
        self.seg, self.box = _Metric(MASK, 0.905), _Metric(BOX, 0.95)
        self.names = {0: "EV", 1: "non-EV"}


def validate(model_path, data_yaml, imgsz=1024, batch_size=16, device="0", split="val", project=None, name="val",
             overlap_mask=None):
    if os.environ.get("FAKE_VAL_FAIL") == "1":
        raise RuntimeError("CUDA out of memory (fake)")
    yaml_path = Path(data_yaml)
    test_dir = yaml_path.parent / "images" / split
    rec = {"model_path": str(model_path), "data_yaml": str(data_yaml), "imgsz": imgsz, "batch_size": batch_size,
           "device": device, "split": split, "project": project, "name": name, "overlap_mask": overlap_mask,
           "weights_md5": hashlib.md5(Path(model_path).read_bytes()).hexdigest(),
           "yaml": yaml_path.read_text(encoding="utf-8"),
           "test_images": sorted(p.name for p in test_dir.iterdir()) if test_dir.is_dir() else None,
           "train_yolo_imported": "train_yolo" in sys.modules, "ultralytics_imported": "ultralytics" in sys.modules}
    out = Path(project) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "confusion_matrix.png").write_bytes(b"png")          # as Ultralytics' plots
    if os.environ.get("FAKE_VAL_RECORD"):
        Path(os.environ["FAKE_VAL_RECORD"]).write_text(json.dumps(rec), encoding="utf-8")
    return _Results()
