"""
Benchmark registered EV models (models/models.yaml): size, compute, memory and runtime.

Per model and device, times the same per-image call the pipeline uses
(training.train_yolo.load_predictions_with_classes_from_model: read image, predict at the
registered imgsz with retina masks, pull masks back as numpy), so the numbers are what a
real `predict_models.py` run costs per image, excluding overlay drawing.

Writes <out>/benchmark.csv and prints a table. GFLOPs are for one square imgsz x imgsz pass
(ultralytics convention); rect inputs at 1440x1024 cost about 0.7x that.

Usage:
  python inference/benchmark_models.py --images <dir> --out <dir> \
      [--models v4_y8n_20260924,v4_y8m_20260924] [--devices cuda,cpu] [--n-gpu 50] [--n-cpu 10]
"""

import argparse
import csv
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml

from training.train_yolo import _YOLO_CACHE, load_predictions_with_classes_from_model

REPO = Path(__file__).resolve().parent.parent
REGISTRY = REPO / "models" / "models.yaml"
WARMUP = 2


def model_stats(weights, imgsz):
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    m = YOLO(weights)
    params = sum(p.numel() for p in m.model.parameters())
    return params, get_flops(m.model, imgsz)


def time_device(weights, images, imgsz, device):
    _YOLO_CACHE.pop(weights, None)  # fresh load per device
    for p in images[:WARMUP]:
        load_predictions_with_classes_from_model(weights, str(p), imgsz, 0.25, 0.7, device, max_det=1000)
    if device != "cpu":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    secs, n_det = [], 0
    for p in images[WARMUP:]:
        t0 = time.perf_counter()
        masks, _, _ = load_predictions_with_classes_from_model(weights, str(p), imgsz, 0.25, 0.7, device, max_det=1000)
        if device != "cpu":
            torch.cuda.synchronize()
        secs.append(time.perf_counter() - t0)
        n_det += len(masks)
    peak = torch.cuda.max_memory_reserved() / 2**30 if device != "cpu" else None
    _YOLO_CACHE.pop(weights, None)
    return secs, peak, n_det


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", default="v4_y8n_20260924,v4_y8m_20260924")
    ap.add_argument("--devices", default="cuda,cpu")
    ap.add_argument("--n-gpu", type=int, default=50, help="images timed on GPU")
    ap.add_argument("--n-cpu", type=int, default=10, help="images timed on CPU (slow)")
    args = ap.parse_args()

    registry = yaml.safe_load(REGISTRY.read_text())["models"]
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"GPU: {gpu}\nCPU: {platform.processor()} ({os.cpu_count()} threads, torch uses {torch.get_num_threads()})"
          f"\nImages: {len(images)} in {args.images}\n", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mid in args.models.split(","):
        spec = registry[mid]
        weights = str(REPO / spec["weights"])
        params, gflops = model_stats(weights, spec["imgsz"])
        for dev in args.devices.split(","):
            n = args.n_cpu if dev == "cpu" else args.n_gpu
            print(f"[{mid}] {dev}: timing {min(n, len(images) - WARMUP)} images ...", flush=True)
            secs, peak, n_det = time_device(weights, images[:n + WARMUP], spec["imgsz"], dev)
            rows.append(dict(model_id=mid, arch=spec["arch"].split(",")[0], params_M=round(params / 1e6, 2),
                             gflops_at_imgsz=round(gflops, 1), weights_MB=round(Path(weights).stat().st_size / 1e6, 1),
                             imgsz=spec["imgsz"], device=gpu if dev != "cpu" else f"CPU {os.cpu_count()} threads",
                             images_timed=len(secs), median_s_per_image=round(statistics.median(secs), 3),
                             mean_s_per_image=round(statistics.mean(secs), 3), max_s_per_image=round(max(secs), 3),
                             images_per_hour=round(3600 / statistics.mean(secs)),
                             peak_vram_GB=round(peak, 2) if peak is not None else "",
                             detections_per_image=round(n_det / len(secs), 1)))
            print(f"    median {rows[-1]['median_s_per_image']} s/image, peak VRAM {rows[-1]['peak_vram_GB']} GB", flush=True)

    with open(args.out / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'model':<18}{'device':<24}{'params':>8}{'GFLOPs':>8}{'s/img':>8}{'img/h':>8}{'VRAM GB':>9}")
    for r in rows:
        print(f"{r['model_id']:<18}{r['device'][:23]:<24}{r['params_M']:>7}M{r['gflops_at_imgsz']:>8}"
              f"{r['median_s_per_image']:>8}{r['images_per_hour']:>8}{str(r['peak_vram_GB']):>9}")
    print(f"\n-> {args.out / 'benchmark.csv'}")


if __name__ == "__main__":
    main()
