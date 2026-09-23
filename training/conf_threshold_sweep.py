"""
Confidence-threshold sweep on an already-trained model -- no retraining needed.

Re-runs calculate_per_class_segmentation_metrics() at several conf values on
the val split to see how much recall improves (and precision degrades) as
the threshold is lowered, since the plan is to lean toward recall for "EV"
and clean up precision in a later post-hoc pass rather than at inference time.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_yolo import calculate_per_class_segmentation_metrics

DATASET = Path(__file__).parent.parent.parent / "training outputs" / "roboflow_20260713_all_layer_singleclass"
MODEL_PATH = DATASET / "runs" / "2026-07-13_144254_y8_1440_8b_300ep_Run_10" / "weights" / "best.pt"
IMGSZ = 1440
CONF_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]
OUT_CSV = DATASET / "conf_threshold_sweep_run10.csv"


def main():
    rows = []
    for split in ("val", "train"):
        img_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split
        for conf in CONF_VALUES:
            print(f"\n=== split={split} conf={conf:.2f} ===")
            metrics = calculate_per_class_segmentation_metrics(
                model_path=str(MODEL_PATH),
                img_dir=str(img_dir),
                label_dir=str(label_dir),
                imgsz=IMGSZ, conf=conf, iou=0.7, device='0',
                class_names=["EV"],
            )
            m = metrics["EV"]
            print(f"  P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
                  f"TP={m['tp']} FP={m['fp']} FN={m['fn']}")
            rows.append({
                "split": split, "conf": conf,
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                "n_gt": m["n_gt"], "n_pred": m["n_pred"],
            })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] Saved to {OUT_CSV}")

    print("\n=== Summary (val) ===")
    print(f"{'conf':>6} {'P':>7} {'R':>7} {'F1':>7} {'TP':>4} {'FP':>4} {'FN':>4}")
    for r in rows:
        if r["split"] != "val":
            continue
        print(f"{r['conf']:>6.2f} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} "
              f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4}")


if __name__ == "__main__":
    main()
