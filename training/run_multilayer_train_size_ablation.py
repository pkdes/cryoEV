"""
Run train-size ablation experiments for the 2-class (Spherical/Multilayer) model.

Design:
- Keep test split fixed for fair comparison.
- Build one fixed validation split from train+valid pool.
- Vary only train subset size across runs.
- Train/evaluate each run and save aggregate + per-class metrics.
"""

import csv
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from training.train_yolo import (
    prepare_yolo_dataset,
    create_yolo_yaml,
    train_yolo_segmentation,
    validate_yolo_model,
    calculate_segmentation_metrics,
    calculate_per_class_segmentation_metrics,
    save_metrics_to_file,
    save_per_class_metrics,
)


@dataclass
class Sample:
    stem: str
    image_path: Path
    label_path: Path
    n_objects: int
    n_multilayer: int


def load_split_samples(split_dir: Path) -> List[Sample]:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    samples: List[Sample] = []
    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        n_objects = 0
        n_multilayer = 0
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    n_objects += 1
                    cls = int(line.split()[0])
                    if cls == 1:
                        n_multilayer += 1
        samples.append(
            Sample(
                stem=stem,
                image_path=img_path,
                label_path=label_path,
                n_objects=n_objects,
                n_multilayer=n_multilayer,
            )
        )
    return samples


def stratified_train_val_split(pool: List[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)
    with_multilayer = [s for s in pool if s.n_multilayer > 0]
    without_multilayer = [s for s in pool if s.n_multilayer == 0]
    rng.shuffle(with_multilayer)
    rng.shuffle(without_multilayer)

    def split_group(items: List[Sample]) -> Tuple[List[Sample], List[Sample]]:
        n_val = max(1, int(round(len(items) * val_ratio))) if items else 0
        return items[n_val:], items[:n_val]

    train_a, val_a = split_group(with_multilayer)
    train_b, val_b = split_group(without_multilayer)

    train_pool = train_a + train_b
    val_pool = val_a + val_b
    rng.shuffle(train_pool)
    rng.shuffle(val_pool)
    return train_pool, val_pool


def write_split(samples: List[Sample], split_root: Path, split_name: str) -> Dict[str, int]:
    out_images = split_root / split_name / "images"
    out_labels = split_root / split_name / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = {"n_images": 0, "n_objects": 0, "n_multilayer": 0}
    for s in samples:
        shutil.copy2(s.image_path, out_images / s.image_path.name)
        if s.label_path.exists():
            shutil.copy2(s.label_path, out_labels / s.label_path.name)
        stats["n_images"] += 1
        stats["n_objects"] += s.n_objects
        stats["n_multilayer"] += s.n_multilayer
    return stats


def main() -> None:
    source_root = Path(r"C:\Users\ML-2619\Desktop\Pujan Cryo\cryo-ev pipeline\data\experiments\CryoAI.v4i.yolov8\collapsed_2class")
    train_src = source_root / "train"
    valid_src = source_root / "valid"
    test_src = source_root / "test"

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ablation_root = source_root / "ablations" / f"multilayer_train_size_{run_stamp}"
    split_root = ablation_root / "splits"
    runs_root = ablation_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    # Fixed design settings
    train_fracs = [0.60, 0.80, 1.00]
    val_ratio = 0.20
    seed = 42
    class_names = ["Spherical", "Multilayer"]

    # Training settings (aligned with your best long-run config)
    epochs = 300
    imgsz = 1024
    batch_size = 16
    patience = 50
    device = "0"
    use_yolov11 = True
    model_size = "n"

    conf_threshold = 0.35
    iou_threshold = 0.50
    match_threshold = 0.30

    base_pool = load_split_samples(train_src) + load_split_samples(valid_src)
    test_samples = load_split_samples(test_src)
    train_pool, fixed_val = stratified_train_val_split(base_pool, val_ratio=val_ratio, seed=seed)

    print("=" * 80)
    print("MULTILAYER TRAIN-SIZE ABLATION")
    print("=" * 80)
    print(f"Pool images: {len(base_pool)} | train_pool: {len(train_pool)} | fixed_val: {len(fixed_val)} | fixed_test: {len(test_samples)}")

    summary_rows: List[Dict] = []
    summary_csv = ablation_root / "ablation_summary.csv"

    for frac in train_fracs:
        n_train = max(1, int(round(len(train_pool) * frac)))
        train_subset = train_pool[:n_train]
        tag = f"tr{int(frac * 100):02d}"
        this_split = split_root / tag

        split_stats_train = write_split(train_subset, this_split, "train")
        split_stats_val = write_split(fixed_val, this_split, "valid")
        split_stats_test = write_split(test_samples, this_split, "test")

        # Build YOLO dataset copy for this run
        yolo_dataset = ablation_root / "datasets" / tag
        yolo_dataset.mkdir(parents=True, exist_ok=True)
        prepare_yolo_dataset(str(this_split / "train"), str(yolo_dataset), "train")
        prepare_yolo_dataset(str(this_split / "valid"), str(yolo_dataset), "val")
        prepare_yolo_dataset(str(this_split / "test"), str(yolo_dataset), "test")
        yaml_path = yolo_dataset / "dataset.yaml"
        create_yolo_yaml(str(yolo_dataset), str(yaml_path), class_names)

        exp_name = f"{run_stamp}_2class_{tag}_{imgsz}_ep{epochs}"
        print(f"\n--- Training run: {exp_name} ---")
        results = train_yolo_segmentation(
            data_yaml=str(yaml_path),
            model_size=model_size,
            epochs=epochs,
            imgsz=imgsz,
            batch_size=batch_size,
            device=device,
            project=str(runs_root),
            name=exp_name,
            patience=patience,
            use_v11=use_yolov11,
        )

        run_dir = Path(results.save_dir)
        best_model = run_dir / "weights" / "best.pt"

        val_results = validate_yolo_model(
            str(best_model),
            str(yaml_path),
            imgsz=imgsz,
            batch_size=batch_size,
            device=device,
            split="val",
            project=str(run_dir),
            name="validation",
        )

        inf_dir = run_dir / "inference"
        inf_dir.mkdir(parents=True, exist_ok=True)

        val_metrics = calculate_segmentation_metrics(
            model_path=str(best_model),
            img_dir=str(yolo_dataset / "images" / "val"),
            label_dir=str(yolo_dataset / "labels" / "val"),
            imgsz=imgsz,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            match_threshold=match_threshold,
        )
        test_metrics = calculate_segmentation_metrics(
            model_path=str(best_model),
            img_dir=str(yolo_dataset / "images" / "test"),
            label_dir=str(yolo_dataset / "labels" / "test"),
            imgsz=imgsz,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            match_threshold=match_threshold,
        )

        save_metrics_to_file(
            val_metrics,
            inf_dir / "metrics_val.txt",
            "val",
            {
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "match_threshold": match_threshold,
            },
        )
        save_metrics_to_file(
            test_metrics,
            inf_dir / "metrics_test.txt",
            "test",
            {
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "match_threshold": match_threshold,
            },
        )

        val_pc = calculate_per_class_segmentation_metrics(
            model_path=str(best_model),
            img_dir=str(yolo_dataset / "images" / "val"),
            label_dir=str(yolo_dataset / "labels" / "val"),
            imgsz=imgsz,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            class_names=class_names,
            match_threshold=match_threshold,
        )
        test_pc = calculate_per_class_segmentation_metrics(
            model_path=str(best_model),
            img_dir=str(yolo_dataset / "images" / "test"),
            label_dir=str(yolo_dataset / "labels" / "test"),
            imgsz=imgsz,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            class_names=class_names,
            match_threshold=match_threshold,
        )
        save_per_class_metrics(val_pc, inf_dir / "metrics_per_class_val.txt", "val")
        save_per_class_metrics(test_pc, inf_dir / "metrics_per_class_test.txt", "test")

        summary_rows.append(
            {
                "run_name": run_dir.name,
                "train_frac": frac,
                "train_images": split_stats_train["n_images"],
                "train_multilayer_instances": split_stats_train["n_multilayer"],
                "val_images": split_stats_val["n_images"],
                "val_multilayer_instances": split_stats_val["n_multilayer"],
                "test_images": split_stats_test["n_images"],
                "test_multilayer_instances": split_stats_test["n_multilayer"],
                "box_map50": float(val_results.box.map50),
                "mask_map50": float(val_results.seg.map50),
                "test_precision": test_metrics["object_precision"],
                "test_recall": test_metrics["object_recall"],
                "test_f1": test_metrics["object_f1"],
                "test_spherical_f1": test_pc["Spherical"]["f1"],
                "test_multilayer_f1": test_pc["Multilayer"]["f1"],
                "best_model": str(best_model),
            }
        )

        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        print(
            f"Completed {run_dir.name} | test F1={test_metrics['object_f1']:.4f} | "
            f"multilayer F1={test_pc['Multilayer']['f1']:.4f}"
        )

    print("\n" + "=" * 80)
    print("Ablation complete")
    print(f"Summary CSV: {summary_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
