"""
Convert COCO JSON annotations to YOLO polygon format with class remapping.
Splits data into train/val/test sets.

Class remapping (for Roboflow 20260604 dataset):
  EV (cat 2) + Incomplete (cat 3) + Low conf EV (cat 4) + Overlapping (cat 6) -> class 0 "EV"
  MV (cat 5)        -> class 1 "multilayer EV"
  items (cat 0), EP in EV (cat 1), VLDL (cat 7), non-EV (cat 8) -> dropped
"""

import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ---------------------------------------------------------------------------
# Class remapping table
# ---------------------------------------------------------------------------
CATEGORY_MAP = {
    0: None,    # "items" - ignored (no annotations)
    1: None,    # "EP in EV" - dropped (only 7 instances, ambiguous)
    2: 0,       # "EV" -> class 0 "EV"
    3: 0,       # "Incomplete" -> class 0 "EV" (merge into EV)
    4: 0,       # "Low conf EV" -> class 0 "EV" (merge into EV)
    5: 1,       # "MV" -> class 1 "multilayer EV"
    6: 0,       # "Overlapping" -> class 0 "EV" (merge into EV)
    7: None,    # "VLDL" - dropped (contaminant, all in train only)
    8: None,    # "non-EV" - dropped (too rare, user request)
}

CLASS_NAMES = ["EV", "multilayer EV"]


def coco_to_yolo_segmentation(coco_path: str, output_root: str,
                               train_ratio: float = 0.8, val_ratio: float = 0.1,
                               seed: int = 42, stratified: bool = True):
    """
    Convert a COCO JSON segmentation dataset into YOLO polygon format
    with train/val/test splits and class remapping.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio >= 0, "train_ratio + val_ratio must be <= 1.0"

    output_root = Path(output_root)
    coco_path = Path(coco_path)

    # Load COCO JSON
    with open(coco_path, 'r') as f:
        coco = json.load(f)

    print(f"Loaded COCO JSON: {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations, "
          f"{len(coco['categories'])} categories")

    cat_id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}

    # Group annotations by image_id
    anns_by_image = defaultdict(list)
    for ann in coco['annotations']:
        anns_by_image[ann['image_id']].append(ann)

    # Collect images with at least one valid annotation
    usable_images = []
    total_annotations_remapped = defaultdict(int)
    annotation_stack = defaultdict(lambda: defaultdict(int))
    # Track which images contain multilayer EV (class 1) for stratified splitting
    images_with_rare = []  # images that contain multilayer EV
    images_without_rare = []  # images with only EV

    for img in coco['images']:
        img_id = img['id']
        anns = anns_by_image.get(img_id, [])

        has_valid = False
        has_rare = False
        for ann in anns:
            new_cat = CATEGORY_MAP.get(ann['category_id'])
            orig_name = cat_id_to_name.get(ann['category_id'], f"cat_{ann['category_id']}")
            if new_cat is not None:
                has_valid = True
                total_annotations_remapped[new_cat] += 1
                annotation_stack[orig_name][new_cat] += 1
                if new_cat == 1:
                    has_rare = True
            else:
                annotation_stack[orig_name][None] += 1

        if has_valid:
            usable_images.append(img)
            if has_rare:
                images_with_rare.append(img)
            else:
                images_without_rare.append(img)

    print(f"\nClass remapping:")
    for orig_cat, new_id in sorted(CATEGORY_MAP.items()):
        name = cat_id_to_name.get(orig_cat, f"cat_{orig_cat}")
        if new_id is not None:
            print(f"  {name:>15s} (id={orig_cat}) -> class {new_id} ({CLASS_NAMES[new_id]})")
        else:
            print(f"  {name:>15s} (id={orig_cat}) -> DROPPED")

    print(f"\nAnnotation movement:")
    for orig_name, dests in sorted(annotation_stack.items()):
        for dest_id, count in sorted(dests.items()):
            if dest_id is None:
                print(f"  {orig_name:>15s}: {count} -> DROPPED")
            else:
                print(f"  {orig_name:>15s}: {count} -> {CLASS_NAMES[dest_id]}")

    # Split according to strategy
    random.seed(seed)
    n_total = len(usable_images)
    n_val_target = max(int(n_total * val_ratio), 1)
    n_test_target = max(int(n_total * (1.0 - train_ratio - val_ratio)), 1) if (1.0 - train_ratio - val_ratio) > 0 else 0

    if stratified:
        # Stratified: put ~50% of rare images into val to boost multilayer EV counts
        random.shuffle(images_with_rare)
        random.shuffle(images_without_rare)

        n_rare_for_val = max(int(len(images_with_rare) * 0.50), 1)
        val_rare = images_with_rare[:n_rare_for_val]
        remaining_rare = images_with_rare[n_rare_for_val:]

        val_remaining_needed = n_val_target - len(val_rare)
        val_common = images_without_rare[:max(val_remaining_needed, 0)]
        val_images = val_rare + val_common
        remaining_common = images_without_rare[max(val_remaining_needed, 0):]

        remaining_all = remaining_rare + remaining_common
        random.shuffle(remaining_all)
        test_images = remaining_all[:n_test_target]
        train_images = remaining_all[n_test_target:]
    else:
        # Random shuffle: simple 80/10/10
        random.shuffle(usable_images)
        train_images = usable_images[:n_total - n_val_target - n_test_target]
        val_images = usable_images[n_total - n_val_target - n_test_target:n_total - n_test_target]
        test_images = usable_images[n_total - n_test_target:]

    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images,
    }

    print(f"\nSplit: {n_total} images total")
    print(f"  train: {len(train_images)}")
    print(f"  val:   {len(val_images)}")
    print(f"  test:  {len(test_images)}")

    # Process each split
    coco_source_dir = coco_path.parent

    for split_name, split_imgs in splits.items():
        img_dir = output_root / 'images' / split_name
        label_dir = output_root / 'labels' / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        n_images = 0
        n_labels = 0
        n_instances = 0

        for img in tqdm(split_imgs, desc=f"Processing {split_name}"):
            # Copy image
            src_path = coco_source_dir / img['file_name']
            if not src_path.exists():
                print(f"  Warning: image not found: {src_path}")
                continue

            dst_path = img_dir / img['file_name']
            shutil.copy2(src_path, dst_path)
            n_images += 1

            # Create YOLO polygon label file
            img_w = img['width']
            img_h = img['height']
            anns = anns_by_image.get(img['id'], [])

            label_lines = []
            for ann in anns:
                new_cat = CATEGORY_MAP.get(ann['category_id'])
                if new_cat is None:
                    continue

                seg = ann['segmentation']
                if not seg or len(seg) == 0:
                    continue

                # Take the first polygon
                if isinstance(seg[0], list):
                    polygon = seg[0]
                else:
                    polygon = seg

                # Convert to normalized YOLO polygon format
                coords = []
                for i in range(0, len(polygon), 2):
                    x = float(polygon[i]) / img_w
                    y = float(polygon[i + 1]) / img_h
                    x = max(0.0, min(1.0, x))
                    y = max(0.0, min(1.0, y))
                    coords.extend([f"{x:.6f}", f"{y:.6f}"])

                if len(coords) >= 6:
                    line = f"{new_cat} " + " ".join(coords)
                    label_lines.append(line)
                    n_instances += 1

            if label_lines:
                label_path = label_dir / f"{img['file_name'].rsplit('.', 1)[0]}.txt"
                label_path.write_text("\n".join(label_lines) + "\n", encoding='utf-8')
                n_labels += 1

        print(f"  [OK] {split_name}: {n_images} images, {n_labels} with labels, "
              f"{n_instances} instances")

    print(f"\n[OK] Dataset ready at: {output_root}")
    return str(output_root)


def merge_coco_files(coco_paths: list) -> dict:
    """
    Merge multiple COCO JSON files into one, handling ID collisions
    and deduplicating images by filename (first occurrence wins).
    """
    merged = {
        "info": {}, "licenses": [], "categories": [],
        "images": [], "annotations": []
    }
    img_id_offset = 0
    ann_id_offset = 0
    cat_ids_seen = set()
    seen_filenames = set()

    for fpath in coco_paths:
        with open(fpath, 'r') as f:
            coco = json.load(f)

        # Merge categories (first file wins)
        for cat in coco.get("categories", []):
            if cat["id"] not in cat_ids_seen:
                cat_ids_seen.add(cat["id"])
                merged["categories"].append(cat)

        # Build filename -> original image_id mapping for this file
        fname_to_orig_id = {}
        for img in coco.get("images", []):
            fname = img["file_name"]
            if fname not in seen_filenames:
                seen_filenames.add(fname)
                new_img = dict(img)
                new_img["id"] += img_id_offset
                merged["images"].append(new_img)
                fname_to_orig_id[img["id"]] = new_img["id"]
            else:
                existing_img = next(im for im in merged["images"] if im["file_name"] == fname)
                fname_to_orig_id[img["id"]] = existing_img["id"]

        # Merge annotations with shifted image + annotation IDs
        for ann in coco.get("annotations", []):
            if ann["image_id"] not in fname_to_orig_id:
                continue
            new_ann = dict(ann)
            new_ann["image_id"] = fname_to_orig_id[ann["image_id"]]
            new_ann["id"] += ann_id_offset
            merged["annotations"].append(new_ann)

        img_id_offset = max(im["id"] for im in merged["images"]) + 1 if merged["images"] else 0
        ann_id_offset = max(an["id"] for an in merged["annotations"]) + 1 if merged["annotations"] else 0

    return merged


def generate_dataset(output_suffix: str = "", stratified: bool = False):
    """
    Generate YOLO dataset from merged Roboflow COCO files.
    If stratified=True, puts ~50% of multilayer EV images into val.
    output_suffix is appended to the output folder name (e.g. '_stratified').
    """
    base = Path(r"C:\Users\pujan\Desktop\cryoEV\Carney CryoEV\training images\roboflow20260604")
    output_root = Path(r"C:\Users\pujan\Desktop\cryoEV\Carney CryoEV\training outputs") / f"roboflow20260604{output_suffix}"

    # Step 1: Flat image directory (reuse existing if already copied)
    flat_dir = base / "_all_images"
    if not flat_dir.exists() or len(list(flat_dir.glob("*"))) == 0:
        flat_dir.mkdir(exist_ok=True)
        import os
        seen_fnames = set()
        for subdir in ['train', 'test', 'valid']:
            for fname in os.listdir(base / subdir):
                if fname == '_annotations.coco.json':
                    continue
                if fname not in seen_fnames:
                    shutil.copy2(base / subdir / fname, flat_dir / fname)
                    seen_fnames.add(fname)
        print(f"Copied {len(seen_fnames)} unique images to {flat_dir}")
    else:
        print(f"Reusing existing flat image directory: {flat_dir}")

    # Step 2: Merge COCO files (with dedup)
    coco_paths = [
        str(base / "train" / "_annotations.coco.json"),
        str(base / "test" / "_annotations.coco.json"),
        str(base / "valid" / "_annotations.coco.json"),
    ]
    merged = merge_coco_files(coco_paths)
    merged_path = flat_dir / "_annotations_merged.coco.json"
    with open(merged_path, 'w') as f:
        json.dump(merged, f)
    print(f"Merged {len(merged['images'])} images, {len(merged['annotations'])} annotations")

    # Step 3: Convert
    if stratified:
        print("\n>> Using STRATIFIED split (multilayer EV boosted in val)")
    else:
        print("\n>> Using RANDOM 80/10/10 split")

    coco_to_yolo_segmentation(
        coco_path=str(merged_path),
        output_root=str(output_root),
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
        stratified=stratified,
    )
    return str(output_root)


if __name__ == '__main__':
    import sys
    mode = "stratified" if "--stratified" in sys.argv else "random"
    
    if mode == "stratified":
        generate_dataset(output_suffix="_stratified", stratified=True)
    else:
        # Recreate original random-split dataset
        import shutil, os
        orig = Path(r"C:\Users\pujan\Desktop\cryoEV\Carney CryoEV\training outputs\roboflow20260604")
        
        # Backup existing version (which may be stratified) before regenerating
        if orig.exists():
            backup = Path(str(orig) + "_backup")
            if not backup.exists():
                shutil.copytree(str(orig), str(backup))
                print(f"[backed up current roboflow20260604 -> {backup}]")
            shutil.rmtree(str(orig))
            print(f"[cleared {orig} to regenerate random split]")

        generate_dataset(output_suffix="", stratified=False)