import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_yolo import visualize_predictions_with_matching

ROOT = Path(__file__).parent.parent.parent  # Carney CryoEV/
DATASET = ROOT / "training outputs" / "roboflow20260604_stratified"
VAL_IMGS   = str(DATASET / "images" / "val")
VAL_LABELS = str(DATASET / "labels" / "val")

RUNS = [
    {
        "name": "Run1_640_300ep",
        "model": str(DATASET / "runs" / "2026-06-04_213516_y8_640_4b_300ep_Run_1" / "weights" / "best.pt"),
        "imgsz": 640,
    },
    {
        "name": "Run2_768_300ep",
        "model": str(DATASET / "runs" / "2026-06-05_004849_y8_768_2b_300ep_Run_2" / "weights" / "best.pt"),
        "imgsz": 768,
    },
]

for run in RUNS:
    out = str(DATASET / "overlays" / run["name"])
    print(f"\n=== {run['name']} ===")
    visualize_predictions_with_matching(
        model_path=run["model"],
        source_dir=VAL_IMGS,
        label_dir=VAL_LABELS,
        output_dir=out,
        imgsz=run["imgsz"],
        conf=0.25,
        iou=0.7,
        device="0",
        match_threshold=0.5,
        class_names=["EV", "multilayer EV"],
    )
