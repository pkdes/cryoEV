"""Evaluate segmentation robustness to noise and contrast perturbations.

This script materializes perturbed copies of a held-out image set, runs the
existing YOLO segmentation model, and compares predictions against the existing
YOLO polygon ground-truth labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from training.train_yolo import (
    load_gt_masks_from_labels,
    load_predictions_from_model,
    match_objects_hungarian,
)


VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _parse_float_list(raw: str) -> list[float]:
    raw = raw.strip()
    if not raw or raw.lower() in {"none", "null", "off"}:
        return []
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def _iter_images(images_dir: Path) -> list[Path]:
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def _load_rgb_image(image_path: Path) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"))


def _save_rgb_image(image: np.ndarray, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(image_path)


def _save_condition_preview(images_dir: Path, output_path: Path, max_images: int = 9, thumb_size: int = 256) -> None:
    image_paths = _iter_images(images_dir)[:max_images]
    if not image_paths:
        return

    n = len(image_paths)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    canvas = np.full((n_rows * thumb_size, n_cols * thumb_size, 3), 255, dtype=np.uint8)

    for idx, image_path in enumerate(image_paths):
        image = _load_rgb_image(image_path)
        h, w = image.shape[:2]
        scale = min(thumb_size / max(h, 1), thumb_size / max(w, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        row = idx // n_cols
        col = idx % n_cols
        y0 = row * thumb_size + (thumb_size - new_h) // 2
        x0 = col * thumb_size + (thumb_size - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

        label = image_path.name[:28]
        cv2.putText(
            canvas,
            label,
            (col * thumb_size + 8, row * thumb_size + thumb_size - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _apply_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, size=image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _apply_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    adjusted = (image.astype(np.float32) - 127.5) * alpha + 127.5
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _apply_poisson_noise(image: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    # Lower scale increases noise strength after re-scaling back to uint8.
    safe_scale = max(scale, 1e-6)
    lam = np.clip(image.astype(np.float32) * safe_scale, 0.0, None)
    noisy = rng.poisson(lam).astype(np.float32) / safe_scale
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    k = max(3, int(math.ceil(sigma * 6)) | 1)
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)


def _apply_sharpen(image: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return image
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0, sigmaY=1.0)
    sharp = cv2.addWeighted(image.astype(np.float32), 1.0 + amount, blur.astype(np.float32), -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _build_conditions(
    noise_sigmas: Iterable[float],
    contrast_alphas: Iterable[float],
    poisson_scales: Iterable[float],
    blur_sigmas: Iterable[float],
    sharpen_amounts: Iterable[float],
    include_contrast_sharpen: bool,
) -> list[dict]:
    conditions = [{"name": "baseline", "kind": "baseline", "value": 0.0}]
    for sigma in noise_sigmas:
        conditions.append({
            "name": f"noise_sigma_{int(round(sigma)):03d}",
            "kind": "gaussian_noise",
            "value": float(sigma),
        })
    for alpha in contrast_alphas:
        safe_alpha = str(alpha).replace(".", "p")
        conditions.append({
            "name": f"contrast_alpha_{safe_alpha}",
            "kind": "contrast",
            "value": float(alpha),
        })
    for scale in poisson_scales:
        safe_scale = str(scale).replace(".", "p")
        conditions.append({
            "name": f"poisson_scale_{safe_scale}",
            "kind": "poisson_noise",
            "value": float(scale),
        })
    for sigma in blur_sigmas:
        safe_sigma = str(sigma).replace(".", "p")
        conditions.append({
            "name": f"gaussian_blur_sigma_{safe_sigma}",
            "kind": "gaussian_blur",
            "value": float(sigma),
        })
    for amount in sharpen_amounts:
        safe_amt = str(amount).replace(".", "p")
        conditions.append({
            "name": f"sharpen_amount_{safe_amt}",
            "kind": "sharpen",
            "value": float(amount),
        })

    if include_contrast_sharpen:
        for alpha in contrast_alphas:
            for amount in sharpen_amounts:
                safe_alpha = str(alpha).replace(".", "p")
                safe_amt = str(amount).replace(".", "p")
                conditions.append({
                    "name": f"lower_defocus_c{safe_alpha}_s{safe_amt}",
                    "kind": "contrast_sharpen",
                    "value": {
                        "contrast_alpha": float(alpha),
                        "sharpen_amount": float(amount),
                    },
                })
    return conditions


def _materialize_condition_images(
    src_images_dir: Path,
    dst_images_dir: Path,
    condition: dict,
    seed: int,
) -> int:
    if dst_images_dir.exists():
        shutil.rmtree(dst_images_dir)
    dst_images_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _iter_images(src_images_dir)
    if condition["kind"] == "baseline":
        for src in image_paths:
            shutil.copy2(src, dst_images_dir / src.name)
        return len(image_paths)

    for idx, src in enumerate(image_paths):
        image = _load_rgb_image(src)
        rng = np.random.default_rng(seed + idx)
        if condition["kind"] == "gaussian_noise":
            image = _apply_gaussian_noise(image, sigma=condition["value"], rng=rng)
        elif condition["kind"] == "contrast":
            image = _apply_contrast(image, alpha=condition["value"])
        elif condition["kind"] == "poisson_noise":
            image = _apply_poisson_noise(image, scale=condition["value"], rng=rng)
        elif condition["kind"] == "gaussian_blur":
            image = _apply_gaussian_blur(image, sigma=condition["value"])
        elif condition["kind"] == "sharpen":
            image = _apply_sharpen(image, amount=condition["value"])
        elif condition["kind"] == "contrast_sharpen":
            image = _apply_contrast(image, alpha=condition["value"]["contrast_alpha"])
            image = _apply_sharpen(image, amount=condition["value"]["sharpen_amount"])
        else:
            raise ValueError(f"Unknown condition kind: {condition['kind']}")
        _save_rgb_image(image, dst_images_dir / src.name)

    return len(image_paths)


def _make_detection_overlay(
    image_rgb: np.ndarray,
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    match_threshold: float,
) -> tuple[np.ndarray, int, int, int, float, float, float, float]:
    matches, unmatched_preds, unmatched_gts = match_objects_hungarian(
        pred_masks, gt_masks, iou_threshold=match_threshold
    )

    overlay = image_rgb.copy().astype(np.float32)
    for pred_idx, _ in matches:
        mask = pred_masks[pred_idx]
        overlay[mask] = overlay[mask] * 0.4 + np.array([0, 255, 0], dtype=np.float32) * 0.6
    for pred_idx in unmatched_preds:
        mask = pred_masks[pred_idx]
        overlay[mask] = overlay[mask] * 0.4 + np.array([255, 255, 0], dtype=np.float32) * 0.6
    for gt_idx in unmatched_gts:
        mask = gt_masks[gt_idx]
        overlay[mask] = overlay[mask] * 0.4 + np.array([255, 0, 0], dtype=np.float32) * 0.6

    tp = len(matches)
    fp = len(unmatched_preds)
    fn = len(unmatched_gts)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    matched_ious = []
    for pred_idx, gt_idx in matches:
        inter = (pred_masks[pred_idx] & gt_masks[gt_idx]).sum()
        union = (pred_masks[pred_idx] | gt_masks[gt_idx]).sum()
        matched_ious.append(float(inter / union) if union > 0 else 0.0)
    avg_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return overlay.astype(np.uint8), tp, fp, fn, precision, recall, f1, avg_iou


def _compute_image_quality_metrics(image_rgb: np.ndarray, gt_masks: list[np.ndarray]) -> tuple[float, float]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    if gt_masks:
        gt_combined = np.logical_or.reduce(gt_masks)
    else:
        gt_combined = np.zeros(gray.shape, dtype=bool)

    # SNR proxy: mean vesicle intensity over std of a low-signal background patch.
    signal_mean = float(np.mean(gray[gt_combined])) if np.any(gt_combined) else float(np.mean(gray))
    patch_size = min(64, gray.shape[0], gray.shape[1])
    candidates = [
        gray[0:patch_size, 0:patch_size],
        gray[0:patch_size, -patch_size:],
        gray[-patch_size:, 0:patch_size],
        gray[-patch_size:, -patch_size:],
    ]

    bg_patch = min(candidates, key=lambda p: float(np.std(p)))
    bg_std = float(np.std(bg_patch))
    snr = signal_mean / bg_std if bg_std > 1e-6 else float("nan")

    # Contrast proxy: mean Sobel gradient magnitude on vesicle boundary band.
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    if np.any(gt_combined):
        gt_u8 = gt_combined.astype(np.uint8)
        boundary = cv2.morphologyEx(gt_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
        sobel_boundary = float(np.mean(grad_mag[boundary])) if np.any(boundary) else float(np.mean(grad_mag))
    else:
        sobel_boundary = float(np.mean(grad_mag))

    return snr, sobel_boundary


def _save_comparison_panel(
    original_rgb: np.ndarray,
    modified_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    output_path: Path,
    condition_name: str,
    image_name: str,
    precision: float,
    recall: float,
    f1: float,
    avg_iou: float,
    snr: float,
    sobel_boundary: float,
    tp: int,
    fp: int,
    fn: int,
) -> None:
    h = max(original_rgb.shape[0], modified_rgb.shape[0], overlay_rgb.shape[0])

    def _fit_h(img: np.ndarray, target_h: int) -> np.ndarray:
        if img.shape[0] == target_h:
            return img
        scale = target_h / max(img.shape[0], 1)
        w = max(1, int(round(img.shape[1] * scale)))
        return cv2.resize(img, (w, target_h), interpolation=cv2.INTER_AREA)

    o = _fit_h(original_rgb, h)
    m = _fit_h(modified_rgb, h)
    d = _fit_h(overlay_rgb, h)
    panel = cv2.hconcat([o, m, d])

    footer_h = 70
    footer = np.full((footer_h, panel.shape[1], 3), 245, dtype=np.uint8)
    text1 = f"{image_name} | {condition_name} | TP={tp} FP={fp} FN={fn}"
    text2 = (
        f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f} "
        f"Matched-IoU={avg_iou:.3f} SNR={snr:.2f} SobelBoundary={sobel_boundary:.2f}"
    )
    cv2.putText(footer, text1, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(footer, text2, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)

    out = cv2.vconcat([panel, footer])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def _calculate_per_image_metrics(
    model_path: str,
    original_img_dir: Path,
    img_dir: Path,
    label_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    match_threshold: float,
    condition_name: str,
    panel_output_dir: Path,
) -> list[dict]:
    rows: list[dict] = []

    for img_path in _iter_images(img_dir):
        img = Image.open(img_path)
        w, h = img.size
        modified_rgb = np.array(img.convert("RGB"))
        original_rgb = _load_rgb_image(original_img_dir / img_path.name)

        pred_masks, confidences = load_predictions_from_model(
            model_path, str(img_path), imgsz, conf, iou, device
        )
        gt_masks = load_gt_masks_from_labels(label_dir / f"{img_path.stem}.txt", w, h)

        overlay_rgb, tp, fp, fn, precision, recall, f1, avg_iou = _make_detection_overlay(
            modified_rgb, pred_masks, gt_masks, match_threshold
        )
        snr, sobel_boundary = _compute_image_quality_metrics(modified_rgb, gt_masks)

        if pred_masks:
            pred_combined = np.logical_or.reduce(pred_masks)
        else:
            pred_combined = np.zeros((h, w), dtype=bool)
        if gt_masks:
            gt_combined = np.logical_or.reduce(gt_masks)
        else:
            gt_combined = np.zeros((h, w), dtype=bool)
        semantic_intersection = float((pred_combined & gt_combined).sum())
        semantic_union = float((pred_combined | gt_combined).sum())
        semantic_iou = semantic_intersection / semantic_union if semantic_union > 0 else 0.0

        _save_comparison_panel(
            original_rgb=original_rgb,
            modified_rgb=modified_rgb,
            overlay_rgb=overlay_rgb,
            output_path=panel_output_dir / f"{img_path.stem}_comparison.png",
            condition_name=condition_name,
            image_name=img_path.name,
            precision=precision,
            recall=recall,
            f1=f1,
            avg_iou=avg_iou,
            snr=snr,
            sobel_boundary=sobel_boundary,
            tp=tp,
            fp=fp,
            fn=fn,
        )

        rows.append(
            {
                "condition": condition_name,
                "image": img_path.name,
                "n_pred": len(pred_masks),
                "n_gt": len(gt_masks),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "avg_iou_matched": avg_iou,
                "semantic_iou": semantic_iou,
                "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
                "snr": snr,
                "sobel_boundary": sobel_boundary,
            }
        )

    return rows


def _summarize_rows(condition: dict, rows: list[dict]) -> dict:
    n_images = len(rows)
    tp = int(sum(r["tp"] for r in rows))
    fp = int(sum(r["fp"] for r in rows))
    fn = int(sum(r["fn"] for r in rows))
    total_gt = int(sum(r["n_gt"] for r in rows))
    total_pred = int(sum(r["n_pred"] for r in rows))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou_matched = float(np.mean([r["avg_iou_matched"] for r in rows])) if rows else 0.0
    semantic_iou = float(np.mean([r["semantic_iou"] for r in rows])) if rows else 0.0

    return {
        "condition": condition["name"],
        "kind": condition["kind"],
        "value": json.dumps(condition["value"]) if isinstance(condition["value"], dict) else condition["value"],
        "n_images": n_images,
        "object_precision": precision,
        "object_recall": recall,
        "object_f1": f1,
        "avg_iou_matched": avg_iou_matched,
        "semantic_iou": semantic_iou,
        "total_gt_objects": total_gt,
        "total_pred_objects": total_pred,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "mean_snr": float(np.mean([r["snr"] for r in rows])) if rows else float("nan"),
        "mean_sobel_boundary": float(np.mean([r["sobel_boundary"] for r in rows])) if rows else float("nan"),
    }


def run_experiment(
    model_path: str,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    noise_sigmas: list[float],
    contrast_alphas: list[float],
    poisson_scales: list[float],
    blur_sigmas: list[float],
    sharpen_amounts: list[float],
    include_contrast_sharpen: bool,
    imgsz: int,
    conf: float,
    iou: float,
    match_threshold: float,
    device: str,
    seed: int,
    break_f1_drop: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    working_dir = output_dir / "generated_images"
    previews_root = output_dir / "previews"
    comparisons_root = output_dir / "comparison_panels"
    metrics_root = output_dir / "metrics"
    working_dir.mkdir(parents=True, exist_ok=True)
    previews_root.mkdir(parents=True, exist_ok=True)
    comparisons_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)

    conditions = _build_conditions(
        noise_sigmas=noise_sigmas,
        contrast_alphas=contrast_alphas,
        poisson_scales=poisson_scales,
        blur_sigmas=blur_sigmas,
        sharpen_amounts=sharpen_amounts,
        include_contrast_sharpen=include_contrast_sharpen,
    )
    summary_rows: list[dict] = []
    per_image_rows: list[dict] = []

    for condition in conditions:
        condition_name = condition["name"]
        condition_images_dir = working_dir / condition_name
        _materialize_condition_images(images_dir, condition_images_dir, condition, seed=seed)
        _save_condition_preview(
            images_dir=condition_images_dir,
            output_path=previews_root / f"{condition_name}_preview.png",
        )
        per_image_rows.extend(
            _calculate_per_image_metrics(
                model_path=model_path,
                original_img_dir=images_dir,
                img_dir=condition_images_dir,
                label_dir=labels_dir,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                match_threshold=match_threshold,
                condition_name=condition_name,
                panel_output_dir=comparisons_root / condition_name,
            )
        )
        condition_rows = [r for r in per_image_rows if r["condition"] == condition_name]
        summary_rows.append(_summarize_rows(condition=condition, rows=condition_rows))

    summary_csv = metrics_root / "condition_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    baseline_row = next((r for r in summary_rows if r["condition"] == "baseline"), None)
    deltas = []
    if baseline_row is not None:
        base_f1 = float(baseline_row["object_f1"])
        base_iou = float(baseline_row["avg_iou_matched"])
        for row in summary_rows:
            current_f1 = float(row["object_f1"])
            current_iou = float(row["avg_iou_matched"])
            f1_drop = base_f1 - current_f1
            deltas.append(
                {
                    "condition": row["condition"],
                    "kind": row["kind"],
                    "value": row["value"],
                    "object_f1": current_f1,
                    "avg_iou_matched": current_iou,
                    "delta_f1_vs_baseline": current_f1 - base_f1,
                    "delta_iou_vs_baseline": current_iou - base_iou,
                    "is_breakpoint": bool(f1_drop >= break_f1_drop),
                    "break_threshold_f1_drop": break_f1_drop,
                }
            )

    if deltas:
        delta_csv = metrics_root / "performance_delta_vs_baseline.csv"
        with open(delta_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(deltas[0].keys()))
            writer.writeheader()
            writer.writerows(deltas)

    per_image_csv = metrics_root / "per_image_metrics.csv"
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    manifest = {
        "model_path": model_path,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "generated_images_dir": str(working_dir),
        "previews_dir": str(previews_root),
        "comparison_panels_dir": str(comparisons_root),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "match_threshold": match_threshold,
        "device": device,
        "seed": seed,
        "conditions": conditions,
        "summary_csv": str(summary_csv),
        "per_image_csv": str(per_image_csv),
        "delta_csv": str(metrics_root / "performance_delta_vs_baseline.csv"),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate robustness to noise and contrast perturbations.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--noise-sigmas", type=str, default="8,16,24")
    parser.add_argument("--contrast-alphas", type=str, default="0.7,0.85,1.15,1.3")
    parser.add_argument("--poisson-scales", type=str, default="none")
    parser.add_argument("--blur-sigmas", type=str, default="none")
    parser.add_argument("--sharpen-amounts", type=str, default="none")
    parser.add_argument("--include-contrast-sharpen", action="store_true")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--break-f1-drop", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_experiment(
        model_path=args.model_path,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        noise_sigmas=_parse_float_list(args.noise_sigmas),
        contrast_alphas=_parse_float_list(args.contrast_alphas),
        poisson_scales=_parse_float_list(args.poisson_scales),
        blur_sigmas=_parse_float_list(args.blur_sigmas),
        sharpen_amounts=_parse_float_list(args.sharpen_amounts),
        include_contrast_sharpen=bool(args.include_contrast_sharpen),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        match_threshold=args.match_threshold,
        device=args.device,
        seed=args.seed,
        break_f1_drop=args.break_f1_drop,
    )


if __name__ == "__main__":
    main()