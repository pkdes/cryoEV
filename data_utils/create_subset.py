"""
Create a training data subset from a full YOLO dataset while preserving
the original val and test sets. Used for data scaling experiments.

Usage:
    python create_subset.py --source <dataset_root> --fraction 0.25 --seed 42
    python create_subset.py --source <dataset_root> --fraction 0.50 --seed 42
"""

import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


def create_subset(source_root: str, fraction: float, seed: int = 42, target_root: str = None):
    """
    Create a training data subset by randomly sampling `fraction` of training images.
    Val and test sets are copied in full.
    The subset preserves the class ratio (stratified by multilayer EV).
    """
    source = Path(source_root)
    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")

    # Determine output path
    pct = int(fraction * 100)
    if target_root is None:
        target = source.parent / f"{source.name}_{pct}pct"
    else:
        target = Path(target_root)

    print(f"Source: {source}")
    print(f"Target: {target}")
    print(f"Fraction: {fraction} ({pct}%)")
    print()

    # --- Val and test: copy in full ---
    for split in ['val', 'test']:
        for subdir in ['images', 'labels']:
            src_dir = source / subdir / split
            dst_dir = target / subdir / split
            if src_dir.exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                for fpath in src_dir.iterdir():
                    shutil.copy2(str(fpath), str(dst_dir / fpath.name))
                print(f"  Copied {split}/{subdir}: {len(list(dst_dir.iterdir()))} files")

    # --- Train: select subset preserving class ratio ---
    train_labels_dir = source / "labels" / "train"
    train_images_dir = source / "images" / "train"

    # Group training images by whether they contain multilayer EV
    images_with_rare = []
    images_without_rare = []

    for label_path in sorted(train_labels_dir.glob("*.txt")):
        # Check if contains multilayer EV (class 1)
        has_rare = False
        with open(label_path) as f:
            for line in f:
                if line.strip().startswith("1"):
                    has_rare = True
                    break

        # Corresponding image
        stem = label_path.stem
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = train_images_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            print(f"  Warning: no image found for {label_path.name}, skipping")
            continue

        if has_rare:
            images_with_rare.append((label_path, image_path))
        else:
            images_without_rare.append((label_path, image_path))

    n_rare = len(images_with_rare)
    n_common = len(images_without_rare)
    n_total = n_rare + n_common
    n_target = max(int(n_total * fraction), 1)

    print(f"\nTraining images: {n_total} total ({n_rare} with multilayer EV, {n_common} without)")

    # Stratified sampling: keep same rare/common ratio in subset
    n_rare_target = max(int(n_rare * fraction), 1)
    n_common_target = n_target - n_rare_target
    if n_common_target < 0:
        n_common_target = 0
    if n_common_target > n_common:
        n_common_target = n_common
        n_rare_target = n_target - n_common_target

    random.seed(seed)
    random.shuffle(images_with_rare)
    random.shuffle(images_without_rare)

    selected_rare = images_with_rare[:n_rare_target]
    selected_common = images_without_rare[:n_common_target]

    selected = selected_rare + selected_common
    random.shuffle(selected)  # mix them up

    # Copy selected training files
    target_train_img = target / "images" / "train"
    target_train_lbl = target / "labels" / "train"
    target_train_img.mkdir(parents=True, exist_ok=True)
    target_train_lbl.mkdir(parents=True, exist_ok=True)

    for label_path, image_path in selected:
        shutil.copy2(str(image_path), str(target_train_img / image_path.name))
        shutil.copy2(str(label_path), str(target_train_lbl / label_path.name))

    print(f"  Selected train: {len(selected)} images ({n_rare_target} with multilayer EV, {n_common_target} without)")

    # --- Verify counts ---
    print(f"\nVerification:")
    for split in ['train', 'val', 'test']:
        n_labels = len(list((target / "labels" / split).glob("*.txt")))
        n_imgs = len(list((target / "images" / split).glob("*")))
        counts = {0: 0, 1: 0}
        for lp in (target / "labels" / split).glob("*.txt"):
            with open(lp) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0])
                        if cls in counts:
                            counts[cls] += 1
        print(f"  {split}: {n_imgs} images, EV={counts[0]}, multilayer EV={counts[1]}")

    # --- Create dataset.yaml ---
    yaml_content = f"""# YOLO dataset config — {pct}% training subset of {source.name}
# Generated by create_subset.py (fraction={fraction}, seed={seed})

path: {target.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: EV
  1: multilayer EV
"""
    yaml_path = target / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\n[OK] Dataset created at: {target}")
    print(f"[OK] Config: {yaml_path}")
    return str(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create training data subset for scaling experiments")
    parser.add_argument("--source", required=True, help="Source dataset root (must have images/ and labels/)")
    parser.add_argument("--fraction", type=float, required=True, help="Fraction of training data to keep (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target", default=None, help="Output path (auto-generated if not provided)")
    args = parser.parse_args()

    create_subset(
        source_root=args.source,
        fraction=args.fraction,
        seed=args.seed,
        target_root=args.target
    )