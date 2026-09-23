import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from train_yolo import load_gt_masks_and_classes_from_labels

ROOT = Path(__file__).parent.parent.parent
DATASET = ROOT / "training outputs" / "roboflow_20260713_all_layer_singleclass"
OUT_DIR = DATASET / "overlays" / "GT_labels"

CLASS_NAMES = ["EV"]
CLASS_COLORS = [np.array([0, 180, 255])]  # blue
ALPHA = 0.25  # light fill only -- outline (below) carries the main signal for nested/overlapping masks
OUTLINE_THICKNESS = 3

OUT_DIR.mkdir(parents=True, exist_ok=True)


def _draw_mask(overlay: np.ndarray, mask: np.ndarray, color: np.ndarray) -> None:
    """Light alpha fill + a thick contour outline, so overlapping/nested masks stay legible."""
    overlay[mask] = overlay[mask] * (1 - ALPHA) + color * ALPHA
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color.tolist(), thickness=OUTLINE_THICKNESS)

for split in ["train", "val"]:
    imgs_dir = DATASET / "images" / split
    labels_dir = DATASET / "labels" / split
    split_out = OUT_DIR / split
    split_out.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(imgs_dir.glob("*"))
    for img_path in tqdm(img_paths, desc=f"GT overlays ({split})"):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]

        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_masks, gt_classes = load_gt_masks_and_classes_from_labels(label_path, w, h)

        overlay = img.copy().astype(float)
        for mask, cid in zip(gt_masks, gt_classes):
            _draw_mask(overlay, mask, CLASS_COLORS[cid])

        fig, ax = plt.subplots(1, 1, figsize=(12, 8.5))
        ax.imshow(overlay.astype(np.uint8))
        ax.set_title(f"{img_path.name}   EV instances={len(gt_masks)}", fontsize=11, fontweight="bold")
        ax.axis("off")

        patches = [Patch(facecolor=CLASS_COLORS[0] / 255, alpha=0.8, label="EV (all layers collapsed)")]
        ax.legend(handles=patches, loc="upper right", fontsize=10, framealpha=0.7)

        plt.tight_layout()
        plt.savefig(split_out / f"{img_path.stem}_gt.png", dpi=150, bbox_inches="tight")
        plt.close()

print(f"[OK] GT overlays saved to {OUT_DIR}")
