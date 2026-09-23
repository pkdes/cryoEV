"""Step 1: extract one fixed-size crop per ground-truth EV object.

Geometry only -- the held-out labels (annotator category, containment role, layer
count, cell line) are RECORDED in the manifest but never used to decide anything
here. They exist so evaluate.py can score clusters post-hoc.

Two pixel scales are present in the source data: every *Jul30* micrograph is
5760x4092, everything else is 1440x1024. The Jul30 images are downsampled 4x so
that one manifest pixel means the same physical distance everywhere -- otherwise
DINO would trivially separate the two scales and we would be clustering
magnification rather than morphology.

Usage:
    python embedding/crops.py [--source-name "..."] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.multilayer_containment import classify_roles, polygon_to_mask
from data_utils.prepare_all_layer_singleclass import CATEGORY_NAME_TO_CLASS
from embedding.paths import (
    CONTEXT_FACTOR, CROP_SIZE, DEFAULT_SOURCE_NAME, MIN_WINDOW_PX,
    OUT, QC, REFERENCE_WIDTH, ROOT, SPLITS, Progress, parse_name, variant,
)


def extract_window(img: np.ndarray, cx: float, cy: float, side: float):
    """Cut a square window around (cx, cy), SHIFTED INWARD to stay on real pixels.

    A window 2x the object's long side overhangs the micrograph border for objects
    near the edge. Padding those regions (replicate or reflect) invents a synthetic,
    high-contrast texture that DINO will happily cluster on -- and since edge objects
    are unevenly distributed across grids, that artifact would masquerade as exactly
    the acquisition-batch effect this experiment is trying to detect.

    So: slide the window back inside the image instead. The object ends up off-center
    but fully visible on genuine data, which costs a ViT essentially nothing. Only a
    window larger than the micrograph itself needs padding, and that gets
    BORDER_REFLECT_101 rather than streaky replicate.

    Returns (crop resized to CROP_SIZE, fraction of the window that was padding,
    pixels the window had to be shifted).
    """
    h, w = img.shape[:2]
    side_i = int(round(side))
    x0, y0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))

    # slide inward; np.clip is a no-op when the window already fits
    sx0 = int(np.clip(x0, 0, max(0, w - side_i)))
    sy0 = int(np.clip(y0, 0, max(0, h - side_i)))
    shift = float(abs(sx0 - x0) + abs(sy0 - y0))

    x1, y1 = sx0 + side_i, sy0 + side_i
    sub = img[max(0, sy0):min(h, y1), max(0, sx0):min(w, x1)]
    if sub.size == 0:
        sub = np.zeros((1, 1), dtype=img.dtype)

    # only reachable when the window is bigger than the micrograph
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    if pad_r or pad_b:
        sub = cv2.copyMakeBorder(sub, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT_101)

    win_area = float(side_i * side_i)
    inside = max(0, min(h, y1) - max(0, sy0)) * max(0, min(w, x1) - max(0, sx0))
    pad_frac = max(0.0, (win_area - inside) / max(1.0, win_area))

    interp = cv2.INTER_AREA if sub.shape[0] > CROP_SIZE else cv2.INTER_CUBIC
    crop = cv2.resize(sub.astype(np.float32), (CROP_SIZE, CROP_SIZE), interpolation=interp)
    return crop, pad_frac, shift


def apply_polygon_mask(crop, mask_crop, dilate_px: int):
    """Blank everything outside the object outline, filling with the object's own
    median intensity.

    The baseline run showed 3 of 4 clusters sorting on what the vesicle SITS ON --
    carbon-film edge, clean ice, cluttered field -- rather than on the vesicle. DINO
    was trained on photographs, where background is usually the most informative
    thing in frame; it has no way to know that here the background is the part to
    ignore. Masking removes that information outright.

    Fill is the median of the object's own interior, not zero: a black fill would
    introduce a hard high-contrast silhouette edge, which is just a different strong
    artifact to cluster on. The mask is dilated a few pixels first because the
    annotator drew the polygon ON the membrane -- cutting exactly at the outline
    would clip the very structure we want the model to see.
    """
    m = mask_crop > 0.5
    if dilate_px > 0 and m.any():
        k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        m = cv2.dilate(m.astype(np.uint8), k).astype(bool)
    if not m.any():
        return crop, 0.0
    return np.where(m, crop, float(np.median(crop[m]))).astype(np.float32), float(m.mean())


def process_split(split_dir: str, source_root: Path, rows: list, prog: Progress,
                  stats: dict, V: dict, mask_objects: bool, dilate_px: int,
                  window_px: int = 0):
    coco_path = source_root / split_dir / "_annotations.coco.json"
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_image = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    for img_meta in tqdm(coco["images"], desc=f"crops [{split_dir}]"):
        anns = anns_by_image.get(img_meta["id"], [])
        if not anns:
            continue

        src = source_root / split_dir / img_meta["file_name"]
        raw = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print(f"  [WARN] unreadable: {src}")
            continue

        # --- normalize pixel scale ------------------------------------------------
        scale = img_meta["width"] / REFERENCE_WIDTH   # 4.0 for Jul30, else 1.0
        if scale != 1.0:
            raw = cv2.resize(
                raw,
                (round(img_meta["width"] / scale), round(img_meta["height"] / scale)),
                interpolation=cv2.INTER_AREA,
            )
        H, W = raw.shape
        img16 = raw.astype(np.float32) * 257.0   # 8-bit source -> 16-bit headroom

        # --- containment roles, computed on the DOWNSAMPLED masks -----------------
        # find_containment() is O(n^2) full-frame boolean ANDs; at native 24 MP this
        # would dominate runtime. Its thresholds are area *ratios*, so rasterizing
        # after the downsample changes no verdict.
        kept, masks = [], []
        for a in anns:
            name = cat_name.get(a["category_id"], "cat_%s" % a["category_id"])
            if name not in CATEGORY_NAME_TO_CLASS:
                raise ValueError(
                    "Unmapped category %r in %s -- add it to CATEGORY_NAME_TO_CLASS "
                    "in data_utils/prepare_all_layer_singleclass.py" % (name, coco_path)
                )
            seg = a.get("segmentation")
            if not seg:
                continue
            poly = seg[0] if isinstance(seg[0], list) else seg
            m = polygon_to_mask([[c / scale for c in poly]], W, H)
            if m.sum() == 0:
                continue
            kept.append((a, name))
            masks.append(m)

        if not masks:
            continue
        role, layer_count = classify_roles(masks)

        info = parse_name(img_meta["file_name"])
        if not info["parsed"]:
            stats["unparsed"] += len(kept)

        for idx, (a, name) in enumerate(kept):
            bx, by, bw, bh = [v / scale for v in a["bbox"]]
            long_side = max(bw, bh)
            if window_px:
                # Fixed NATIVE window: identical physical area for every object and
                # no per-object rescaling, so the resize-sharpness cue that tracked
                # object size (rho 0.78) in the context-window design disappears.
                side = float(window_px)
            else:
                side = max(MIN_WINDOW_PX, CONTEXT_FACTOR * long_side)
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            crop, pad_frac, shift = extract_window(img16, cx, cy, side)

            object_frac = np.nan
            if mask_objects:
                # identical window on the object's own mask, so the two line up exactly
                mcrop, _, _ = extract_window(masks[idx].astype(np.float32), cx, cy, side)
                crop, object_frac = apply_polygon_mask(crop, mcrop, dilate_px)

            object_id = "%s_%s" % (split_dir, a["id"])
            cv2.imwrite(str(V["crops"] / (object_id + ".png")),
                        np.clip(crop, 0, 65535).astype(np.uint16))

            rows.append({
                "object_id": object_id, "coco_ann_id": a["id"],
                "image_file": img_meta["file_name"], "split": split_dir,
                "session": info["session"], "grid": info["grid"],
                "session_grid": info["session"] + "/" + info["grid"],
                "square": info["square"], "hole": info["hole"],
                "session_parsed": info["parsed"], "cell_line": info["cell_line"],
                "native_width": img_meta["width"], "scale_factor": scale,
                "bbox_x": bx, "bbox_y": by, "bbox_w": bw, "bbox_h": bh,
                "long_side_px": long_side, "area_px": float(masks[idx].sum()),
                "window_side_px": side, "pad_frac": pad_frac,
                "window_shift_px": shift, "object_frac": object_frac,
                # --- held out below: recorded, never an input ---
                "category": name,
                "role": role[idx],
                "layer_count": layer_count[idx],
            })
            stats["by_category"][name] += 1
            prog.update()


def build_qc_montage(df: pd.DataFrame, out_path: Path, crops_dir: Path):
    """5x5 montage deliberately spanning both pixel scales, size extremes, categories."""
    rng = np.random.default_rng(0)
    cats = sorted(df["category"].unique())
    picks = []
    for i in range(25):
        sub = df[df["category"] == cats[i % len(cats)]]
        # alternate native scales so a 4x scale error shows up side by side
        want = 5760 if i % 2 else 1440
        s2 = sub[sub["native_width"] == want]
        sub = s2 if len(s2) else sub
        picks.append(sub.iloc[int(rng.integers(0, len(sub)))])

    tiles = []
    for r in picks:
        c = cv2.imread(str(crops_dir / (r["object_id"] + ".png")), cv2.IMREAD_UNCHANGED)
        c = cv2.normalize(c.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(c, (0, CROP_SIZE - 34), (CROP_SIZE, CROP_SIZE), (0, 0, 0), -1)
        cv2.putText(c, str(r["session"])[:20], (3, CROP_SIZE - 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(c, "%s %.0fpx @%d" % (r["category"], r["long_side_px"], r["native_width"]),
                    (3, CROP_SIZE - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (120, 255, 120), 1, cv2.LINE_AA)
        cv2.rectangle(c, (0, 0), (CROP_SIZE - 1, CROP_SIZE - 1), (70, 70, 70), 1)
        tiles.append(c)

    grid = np.vstack([np.hstack(tiles[i * 5:(i + 1) * 5]) for i in range(5)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    ap.add_argument("--variant", default="",
                    help="name this run (e.g. 'masked') to build it alongside the baseline")
    ap.add_argument("--mask-objects", action="store_true",
                    help="blank everything outside the object polygon (removes field context)")
    ap.add_argument("--window-px", type=int, default=0,
                    help="fixed NATIVE window side in px (0 = scale to 2x the bbox). "
                         "A fixed window resamples nothing, removing the sharpness-vs-size "
                         "artifact, at the cost of small objects filling little of the frame")
    ap.add_argument("--mask-dilate", type=int, default=3,
                    help="pixels to dilate the mask so the membrane itself is not clipped")
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    args = ap.parse_args()

    V = variant(args.variant)
    manifest_path = V["manifest"]
    if manifest_path.exists() and not args.force:
        sys.exit("[ABORT] %s exists. Re-run with --force to rebuild." % manifest_path)

    source_root = ROOT / "training images" / args.source_name
    V["crops"].mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)

    total = 0
    for s in SPLITS:
        with open(source_root / s / "_annotations.coco.json", "r", encoding="utf-8") as f:
            total += len(json.load(f)["annotations"])

    print("Source: %s\nOutput: %s\nAnnotations to process: %d\n" % (source_root, OUT, total))

    rows: list = []
    stats = {"by_category": Counter(), "unparsed": 0}
    prog = Progress("crops%s" % V["sfx"], total, every=50)
    for sp in SPLITS:
        process_split(sp, source_root, rows, prog, stats, V,
                      args.mask_objects, args.mask_dilate, args.window_px)
    prog.finish()

    df = pd.DataFrame(rows)
    df.to_parquet(manifest_path, index=False)

    print("\n[OK] %d crops -> %s" % (len(df), V["crops"]))
    print("[OK] manifest -> %s" % manifest_path)
    print("\nHeld-out category counts (evaluation only):")
    for k, v in sorted(stats["by_category"].items(), key=lambda kv: -kv[1]):
        print("  %14s: %d" % (k, v))
    print("\nRole counts (classify_roles, geometry only):")
    for k, v in df["role"].value_counts().items():
        print("  %14s: %d" % (k, v))
    print("\nAcquisition groups: %d" % df["session_grid"].nunique())
    print("Objects whose session prefix Roboflow trimmed: %d / %d"
          % (stats["unparsed"], len(df)))
    print("Crops needing any padding (window > micrograph): %d"
          % int((df["pad_frac"] > 0).sum()))
    print("Crops whose window was shifted inward to avoid padding: %d"
          % int((df["window_shift_px"] > 0).sum()))
    print("\nlong_side_px pct [1,25,50,75,99]: %s"
          % np.percentile(df["long_side_px"], [1, 25, 50, 75, 99]).round(1))

    if args.mask_objects:
        print("Object pixels retained per crop (median): %.1f%%"
              % (100 * df["object_frac"].median()))

    qc_path = QC / ("crops_montage_qc%s.png" % V["sfx"])
    build_qc_montage(df, qc_path, V["crops"])
    print("\n[GATE B] review -> %s" % qc_path)


if __name__ == "__main__":
    main()
