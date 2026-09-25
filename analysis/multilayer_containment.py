"""
Multilayer EV detection via polygon containment -- validated on ground truth.

Idea: rather than a dedicated "multilayer EV" model class, treat every
membrane layer as its own "EV" polygon (already how the 20260713 all-layer
dataset is labeled) and determine multilayer status *after* detection: an
outer polygon that fully contains one or more other polygons is a multilayer
EV, and the number of contained polygons is its layer count.

This validates that idea purely on ground truth, before ever touching a
trained model: the pre-collapse COCO source still has the annotator's own
`MV` (outer, multilayer) and `inner` (inner-layer) category judgments, which
are exactly the labels a geometry-only containment test should reproduce.

Containment rule: polygon B is "contained in" polygon A if
  area(A ∩ B) / area(B) >= CONTAINMENT_THRESHOLD
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent.parent / "CryoAI"
DEFAULT_SOURCE_NAME = "roboflow - 20260713 all layer label"
SOURCE_ROOT = ROOT / "training images" / DEFAULT_SOURCE_NAME
SPLITS = ["train", "valid"]

CONTAINMENT_THRESHOLD = 0.9


def polygon_to_mask(segmentation, width: int, height: int) -> np.ndarray:
    """Rasterize a COCO polygon (absolute pixel coords) into a boolean mask."""
    poly = segmentation[0] if isinstance(segmentation[0], list) else segmentation
    pts = np.array([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def load_image_annotations(coco_path: Path):
    """Yield (image_dict, [ (ann_id, category_name, mask) ]) per image."""
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    for img in coco["images"]:
        w, h = img["width"], img["height"]
        entries = []
        for ann in anns_by_image.get(img["id"], []):
            seg = ann.get("segmentation")
            if not seg:
                continue
            mask = polygon_to_mask(seg, w, h)
            if mask.sum() == 0:
                continue
            entries.append((ann["id"], cat_id_to_name.get(ann["category_id"], "?"), mask))
        yield img, entries


def pairwise_overlaps(masks):
    """Mask areas and pairwise intersection pixel counts -- the only mask work
    find_containment() needs, so it can be computed once and reused across
    threshold/ratio settings.

    Only pairs whose bounding boxes overlap are intersected, and only inside
    the shared box; every other pair has intersection 0, which can never pass
    a containment_threshold > 0. Returns (areas, inter) with inter a dict
    {(i, j): pixels} holding both orderings of each overlapping pair.
    """
    n = len(masks)
    areas = np.array([int(m.sum()) for m in masks], dtype=np.int64)
    boxes = np.zeros((n, 4), dtype=np.int64)  # y0, y1, x0, x1 (exclusive ends)
    for k, m in enumerate(masks):
        if areas[k]:
            rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
            boxes[k] = (rows[0], rows[-1] + 1, cols[0], cols[-1] + 1)
    y0 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.minimum(boxes[:, None, 1], boxes[None, :, 1])
    x0 = np.maximum(boxes[:, None, 2], boxes[None, :, 2])
    x1 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    nonzero = areas > 0
    overlap = (y1 > y0) & (x1 > x0) & nonzero[:, None] & nonzero[None, :]
    inter = {}
    for i, j in zip(*np.nonzero(np.triu(overlap, k=1))):
        sl = (slice(y0[i, j], y1[i, j]), slice(x0[i, j], x1[i, j]))
        v = int(np.logical_and(masks[i][sl], masks[j][sl]).sum())
        inter[(i, j)] = inter[(j, i)] = v
    return areas, inter


def find_containment(entries, containment_threshold: float = CONTAINMENT_THRESHOLD, max_size_ratio: float = None,
                     overlaps=None):
    """For each polygon, find the set of ann_ids it directly contains.

    B counts as contained in A if area(A∩B)/area(B) >= containment_threshold,
    and (if max_size_ratio is set) area(B)/area(A) <= max_size_ratio -- the
    size-ratio filter guards against near-duplicate predictions of the same
    object (B almost as big as A) being mistaken for a true inner layer.

    `overlaps` is an optional precomputed pairwise_overlaps() of the same masks,
    for callers that evaluate many threshold/ratio settings on one image.
    """
    areas, inter = overlaps if overlaps is not None else pairwise_overlaps([m for _, _, m in entries])
    contains = defaultdict(set)  # ann_id -> set of ann_ids it contains
    for (i, j), v in inter.items():  # i = candidate container, j = candidate contained
        if max_size_ratio is not None and areas[j] / areas[i] > max_size_ratio:
            continue
        if v / areas[j] >= containment_threshold:
            contains[entries[i][0]].add(entries[j][0])
    return contains


def classify_roles(masks, containment_threshold: float = CONTAINMENT_THRESHOLD, max_size_ratio: float = None,
                   overlaps=None):
    """
    Assign each mask a role purely from geometry -- shared by ground truth AND
    predictions, so both sides of any comparison use the identical rule. This
    mirrors the annotator's own 2-tier scheme (MV = outermost boundary only,
    inner = everything else nested inside at ANY depth): being contained by
    something takes priority over containing something, so a middle layer of
    a 3+-layer stack is 'inner-layer' (matching how it was actually labeled),
    not 'multilayer'. Only the true outermost polygon (no parent) that also
    contains >=1 other polygon is 'multilayer'.
      'inner-layer'  if it IS contained by something else (checked FIRST,
                      regardless of whether it also contains something itself)
      'multilayer'   else if it contains >=1 other mask (and has no parent --
                      this is the true outermost boundary of the stack)
      'standalone'   otherwise

    `overlaps`: optional precomputed pairwise_overlaps(masks) (see find_containment).
    Returns (role: dict[idx -> str], layer_count: dict[idx -> int]).
    """
    entries = [(i, "x", m) for i, m in enumerate(masks)]
    contains = find_containment(entries, containment_threshold, max_size_ratio, overlaps)
    contained_by_someone = set()
    for parent, children in contains.items():
        contained_by_someone |= children

    role, layer_count = {}, {}
    for i in range(len(masks)):
        n_contained = len(contains.get(i, set()))
        if i in contained_by_someone:
            role[i] = "inner-layer"
            layer_count[i] = 0
        elif n_contained > 0:
            role[i] = "multilayer"
            layer_count[i] = n_contained
        else:
            role[i] = "standalone"
            layer_count[i] = 0
    return role, layer_count


def process_split(coco_path: Path) -> dict:
    """Accumulate GT containment-validation stats for one COCO split file."""
    stats = {
        "total_images": 0, "total_mv_polys": 0, "total_inner_polys": 0,
        "tp": 0, "fp": 0, "fn": 0, "mv_with_contained": 0,
        "role_vs_category": defaultdict(lambda: defaultdict(int)),
        "per_image_report": [], "categories_seen": set(),
    }
    if not coco_path.exists():
        return stats

    for img, entries in load_image_annotations(coco_path):
        stats["total_images"] += 1
        id_to_cat = {aid: cat for aid, cat, _ in entries}
        stats["categories_seen"] |= {cat for _, cat, _ in entries}
        mv_ids = {aid for aid, cat, _ in entries if cat == "MV"}
        inner_ids = {aid for aid, cat, _ in entries if cat == "inner"}
        stats["total_mv_polys"] += len(mv_ids)
        stats["total_inner_polys"] += len(inner_ids)

        if not entries:
            continue

        contains = find_containment(entries)

        # Union of everything contained within any MV polygon on this image
        geo_inner_ids = set()
        for mv_id in mv_ids:
            contained = contains.get(mv_id, set())
            if contained:
                stats["mv_with_contained"] += 1
            geo_inner_ids |= contained

        image_tp = len(geo_inner_ids & inner_ids)
        image_fp = len(geo_inner_ids - inner_ids)
        image_fn = len(inner_ids - geo_inner_ids)
        stats["tp"] += image_tp
        stats["fp"] += image_fp
        stats["fn"] += image_fn

        # Geometric role per polygon, cross-tabbed against annotator category
        # (works regardless of whether MV/inner category names are present --
        # it's tabulated against whatever the raw annotator category is).
        masks = [m for _, _, m in entries]
        ids = [aid for aid, _, _ in entries]
        role, _ = classify_roles(masks)
        for idx, aid in enumerate(ids):
            cat = id_to_cat[aid]
            stats["role_vs_category"][cat][role[idx]] += 1

        if mv_ids or inner_ids or geo_inner_ids:
            cat_breakdown = defaultdict(int)
            for aid in geo_inner_ids:
                cat_breakdown[id_to_cat.get(aid, "?")] += 1
            stats["per_image_report"].append({
                "image": img["file_name"],
                "n_mv": len(mv_ids),
                "n_inner_labeled": len(inner_ids),
                "n_geo_contained": len(geo_inner_ids),
                "tp": image_tp, "fp": image_fp, "fn": image_fn,
                "geo_contained_categories": dict(cat_breakdown),
            })
    return stats


def merge_stats(split_stats: list) -> dict:
    merged = {
        "total_images": sum(s["total_images"] for s in split_stats),
        "total_mv_polys": sum(s["total_mv_polys"] for s in split_stats),
        "total_inner_polys": sum(s["total_inner_polys"] for s in split_stats),
        "tp": sum(s["tp"] for s in split_stats),
        "fp": sum(s["fp"] for s in split_stats),
        "fn": sum(s["fn"] for s in split_stats),
        "mv_with_contained": sum(s["mv_with_contained"] for s in split_stats),
        "role_vs_category": defaultdict(lambda: defaultdict(int)),
        "per_image_report": [r for s in split_stats for r in s["per_image_report"]],
        "categories_seen": set().union(*[s["categories_seen"] for s in split_stats]) if split_stats else set(),
    }
    for s in split_stats:
        for cat, counts in s["role_vs_category"].items():
            for role, n in counts.items():
                merged["role_vs_category"][cat][role] += n
    return merged


def print_report(label: str, stats: dict, has_mv_inner_labels: bool):
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("=" * 70)
    print(f"MULTILAYER CONTAINMENT VALIDATION -- {label.upper()} (ground truth only, no model)")
    print("=" * 70)
    print(f"Images processed:              {stats['total_images']}")

    if has_mv_inner_labels:
        print(f"Total MV polygons (annotator): {stats['total_mv_polys']}")
        print(f"Total inner polygons (annotator): {stats['total_inner_polys']}")
        print(f"MV polygons with >=1 geometrically contained polygon: {stats['mv_with_contained']} / {stats['total_mv_polys']}")
        print()
        print(f"Containment threshold: {CONTAINMENT_THRESHOLD}")
        print("-- 'Is this inside an MV' check (original, narrower question) --")
        print(f"TP={tp} FP={fp} FN={fn}")
        print(f"Precision (geo-contained -> labeled 'inner'): {precision:.4f}")
        print(f"Recall    (labeled 'inner' -> geo-contained):  {recall:.4f}")
        print(f"F1: {f1:.4f}")
    else:
        print("[WARN] 'MV'/'inner' categories not present in this export -- "
              "skipping annotator MV/inner cross-check (P/R/F1 above is not meaningful without them).")

    print()
    print("-- Geometric role (multilayer/inner-layer/standalone) vs. annotator category --")
    role_vs_category = stats["role_vs_category"]
    all_roles = sorted({r for cats in role_vs_category.values() for r in cats})
    header = f"{'category':>15}" + "".join(f"{r:>13}" for r in all_roles) + f"{'total':>8}"
    print(header)
    for cat in sorted(role_vs_category):
        counts = role_vs_category[cat]
        total = sum(counts.values())
        row = f"{cat:>15}" + "".join(f"{counts.get(r, 0):>13}" for r in all_roles) + f"{total:>8}"
        print(row)
    if has_mv_inner_labels:
        n_inner_also_multilayer = role_vs_category.get("inner", {}).get("multilayer", 0)
        print(f"\n'inner'-labeled polygons that are ALSO geometrically multilayer "
              f"(middle layers of a 3+-layer stack): {n_inner_also_multilayer} / {stats['total_inner_polys']}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-name", default=DEFAULT_SOURCE_NAME,
                         help="Folder name under 'training images/' holding the raw Roboflow COCO export")
    args = parser.parse_args()
    source_root = ROOT / "training images" / args.source_name

    splits = ["train", "valid"]
    if (source_root / "test" / "_annotations.coco.json").exists():
        splits.append("test")

    per_split_stats = {}
    for split in splits:
        coco_path = source_root / split / "_annotations.coco.json"
        per_split_stats[split] = process_split(coco_path)

    combined = merge_stats(list(per_split_stats.values()))
    has_mv_inner_labels = {"MV", "inner"}.issubset(combined["categories_seen"])
    if not has_mv_inner_labels:
        print(f"[WARN] categories seen across this dataset: {sorted(combined['categories_seen'])}")
        print("[WARN] 'MV' and/or 'inner' not both present -- annotator cross-check will be skipped per split.\n")

    split_label = {"train": "TRAIN", "valid": "VALID", "test": "TEST"}
    for split in splits:
        print_report(split_label.get(split, split.upper()), per_split_stats[split], has_mv_inner_labels)

    print_report("COMBINED", combined, has_mv_inner_labels)

    print("Per-image detail (combined, images with any MV/inner/geo-contained polygons):")
    for r in combined["per_image_report"]:
        print(f"  {r['image']}: MV={r['n_mv']} inner_labeled={r['n_inner_labeled']} "
              f"geo_contained={r['n_geo_contained']} TP={r['tp']} FP={r['fp']} FN={r['fn']} "
              f"categories={r['geo_contained_categories']}")


if __name__ == "__main__":
    main()
