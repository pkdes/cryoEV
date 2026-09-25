"""
Build an offline, browser-based label review package from a Roboflow COCO export
plus a training run's cached predictions.

The package is a folder (and a .zip of it) holding `index.html`, `data.js` and
downscaled grayscale JPEGs. Opening index.html in any browser needs no Python,
no install and no network. The reviewer edits the existing annotations with
the model's predictions drawn as an overlay, then exports a JSON of edits.
annotation/apply_review_edits.py merges that JSON back into a new COCO export.

Each image is flagged by where labels and predictions disagree (IoU 0.5 Hungarian
match, same as the training evaluation, on masks rasterized at 1/8 scale for speed):
  - labels with no matching prediction   (possibly spurious / wrong label)
  - predictions with no matching label   (possibly missed object)
Images are listed most-flagged first.

Usage:
  python annotation/build_review_package.py \
      --source-name "roboflow - 20260924 cryoai v4" \
      --dataset roboflow_20260924_cryoai_v4_singleclass \
      --model-run-dir 2026-09-24_172246_y8m_1440_16b_300ep_Run_2
"""
import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
from train_yolo import load_prediction_polygons, match_objects_hungarian  # noqa: E402

ROOT = REPO.parent / "CryoAI"
TEMPLATE = Path(__file__).resolve().parent / "review_tool" / "index.html"
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}  # coco split -> yolo split
NOT_MATCHED = {"non-EV", "items-nfaR"}  # dropped from training, so never flagged
MATCH_SCALE = 1 / 8


def rasterize(ring_norm, w, h):
    """Normalized flat ring -> bool mask at MATCH_SCALE of the original image size."""
    mw, mh = max(1, round(w * MATCH_SCALE)), max(1, round(h * MATCH_SCALE))
    pts = np.round(np.asarray(ring_norm, dtype=np.float64).reshape(-1, 2) * [mw, mh]).astype(np.int32)
    mask = np.zeros((mh, mw), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-name", required=True, help="Folder under 'training images/' (Roboflow COCO export)")
    ap.add_argument("--dataset", required=True, help="Folder under 'training outputs/' holding the run")
    ap.add_argument("--model-run-dir", required=True, help="Run folder under '<dataset>/runs/' whose predictions to show")
    ap.add_argument("--output-name", default=None, help="Folder under 'annotation review/' (default: <source>_review_<date>)")
    ap.add_argument("--width", type=int, default=1440, help="Display width of the packaged images")
    ap.add_argument("--quality", type=int, default=72, help="JPEG quality of the packaged images")
    ap.add_argument("--part-size", type=int, default=40,
                    help="Also split into self-contained parts of this many images (0 = no split); "
                         "40 keeps each zip under ~30 MB")
    args = ap.parse_args()

    source_root = ROOT / "training images" / args.source_name
    run_dir = ROOT / "training outputs" / args.dataset / "runs" / args.model_run_dir
    name = args.output_name or f"{args.source_name.replace(' ', '_')}_review_{datetime.now():%Y%m%d}"
    out = ROOT / "annotation review" / name
    (out / "images").mkdir(parents=True, exist_ok=True)

    categories, images = None, []
    for coco_split, yolo_split in SPLIT_MAP.items():
        coco_path = source_root / coco_split / "_annotations.coco.json"
        if not coco_path.exists():
            continue
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        cat_names = {c["id"]: c["name"] for c in coco["categories"]}
        if categories is None:
            categories = [c["name"] for c in coco["categories"] if c["name"] != "items-nfaR"]
        anns_by_img = {}
        for a in coco["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)

        for img in tqdm(coco["images"], desc=f"Packaging {coco_split}"):
            w, h = img["width"], img["height"]
            gt = []
            for a in anns_by_img.get(img["id"], []):
                rings = [[round(v / (w if k % 2 == 0 else h), 5) for k, v in enumerate(seg)]
                         for seg in a.get("segmentation", []) if isinstance(seg, list) and len(seg) >= 6]
                if rings:
                    gt.append({"id": a["id"], "c": cat_names.get(a["category_id"], "EV"), "p": rings})

            stem = Path(img["file_name"]).stem
            polys, confs = load_prediction_polygons(run_dir / "predictions" / yolo_split / f"{stem}.txt", 1, 1)
            pr = [{"s": round(c, 3), "p": [[round(float(v), 5) for v in poly.reshape(-1)]]}
                  for poly, c in zip(polys, confs) if len(poly) >= 3]

            # Disagreement flags (labels vs predictions), same matcher as training evaluation.
            gt_idx = [i for i, g in enumerate(gt) if g["c"] not in NOT_MATCHED]
            matches, unmatched_p, unmatched_g = match_objects_hungarian(
                [rasterize(p["p"][0], w, h) for p in pr], [rasterize(gt[i]["p"][0], w, h) for i in gt_idx], 0.5)
            for p in unmatched_p:
                pr[p]["u"] = 1
            for g in unmatched_g:
                gt[gt_idx[g]]["u"] = 1

            im = Image.open(source_root / coco_split / img["file_name"]).convert("L")
            if im.width > args.width:
                im = im.resize((args.width, round(im.height * args.width / im.width)), Image.LANCZOS)
            jpg = f"images/{stem}.jpg"
            im.save(out / jpg, "JPEG", quality=args.quality)

            images.append({"file": img["file_name"], "split": coco_split, "w": w, "h": h, "src": jpg,
                           "gt": gt, "pr": pr, "fg": len(unmatched_g), "fp": len(unmatched_p)})

    images.sort(key=lambda r: -(r["fg"] + r["fp"]))
    data = {"id": name, "source": args.source_name, "run": args.model_run_dir,
            "created": datetime.now().isoformat(timespec="seconds"), "categories": categories, "images": images}
    write_packages(out, data, args.part_size)
    print(f"[OK] {len(images)} images, {sum(len(r['gt']) for r in images)} labels, "
          f"{sum(len(r['pr']) for r in images)} predictions")
    print(f"[OK] Flags: {sum(r['fg'] for r in images)} unmatched labels, "
          f"{sum(r['fp'] for r in images)} unmatched predictions")


def write_packages(out: Path, data: dict, part_size: int = 0):
    """Write index.html + data.js next to out/images and zip it. With part_size > 0, also split the
    (flag-sorted) images into self-contained parts <out>_partN/ + .zip, each small enough to send or
    download on its own; the most-flagged images land in part 1."""
    def write(folder: Path, d: dict):
        (folder / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(d, separators=(",", ":")) + ";\n",
                                        encoding="utf-8")
        shutil.copy(TEMPLATE, folder / "index.html")
        zip_path = folder.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:  # JPEGs don't compress further
            for f in sorted(folder.rglob("*")):
                if f.is_file():
                    z.write(f, Path(folder.name) / f.relative_to(folder))
        print(f"[OK] {folder.name}: {len(d['images'])} images, zip {zip_path.stat().st_size / 2**20:.0f} MiB")

    write(out, data)
    if part_size <= 0:
        return
    chunks = [data["images"][k:k + part_size] for k in range(0, len(data["images"]), part_size)]
    for n, chunk in enumerate(chunks, 1):
        part = out.parent / f"{out.name}_part{n}"
        (part / "images").mkdir(parents=True, exist_ok=True)
        for r in chunk:
            shutil.copy(out / r["src"], part / r["src"])
        write(part, {**data, "id": f"{data['id']}_part{n}", "images": chunk})

if __name__ == "__main__":
    main()
