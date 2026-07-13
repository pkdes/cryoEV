import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from train_yolo import load_gt_masks_and_classes_from_labels

ROOT = Path(__file__).parent.parent.parent
DATASET = ROOT / "training outputs" / "roboflow20260604_stratified"
VAL_IMGS   = DATASET / "images" / "val"
VAL_LABELS = DATASET / "labels" / "val"
OUT_DIR    = DATASET / "overlays" / "GT_labels"

CLASS_NAMES  = ["EV", "multilayer EV"]
CLASS_COLORS = [np.array([0, 180, 255]), np.array([255, 80, 0])]  # blue, orange
ALPHA = 0.5

OUT_DIR.mkdir(parents=True, exist_ok=True)

img_paths = sorted(VAL_IMGS.glob("*"))
for img_path in tqdm(img_paths, desc="GT overlays"):
    if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        continue

    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]

    label_path = VAL_LABELS / f"{img_path.stem}.txt"
    gt_masks, gt_classes = load_gt_masks_and_classes_from_labels(label_path, w, h)

    overlay = img.copy().astype(float)
    counts = [0] * len(CLASS_NAMES)
    for mask, cid in zip(gt_masks, gt_classes):
        color = CLASS_COLORS[cid]
        overlay[mask] = overlay[mask] * (1 - ALPHA) + color * ALPHA
        counts[cid] += 1

    legend_str = "  ".join(f"{CLASS_NAMES[i]}={counts[i]}" for i in range(len(CLASS_NAMES)))
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(overlay.astype(np.uint8))
    ax.set_title(f"{img_path.name}   {legend_str}", fontsize=11, fontweight="bold")
    ax.axis("off")

    patches = [Patch(facecolor=CLASS_COLORS[i]/255, alpha=0.8, label=CLASS_NAMES[i])
               for i in range(len(CLASS_NAMES))]
    ax.legend(handles=patches, loc="upper right", fontsize=10, framealpha=0.7)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{img_path.stem}_gt.png", dpi=150, bbox_inches="tight")
    plt.close()

print(f"[OK] GT overlays saved to {OUT_DIR}")
