"""
inference/perf_log.py
=====================
Lightweight performance telemetry for cryoEV inference sessions.

Records, per image:
  - wall-clock inference time
  - raw detection count
  - input image dimensions
  - model / parameter settings used

Records, per session (once):
  - software versions (Python, PyTorch, Ultralytics, OpenCV)
  - hardware snapshot (OS, CPU model + core counts, total RAM, GPU + VRAM)
  - run identity (session ID, model path, device, imgsz, conf, iou)

All rows land in a single ``performance_log.csv`` file placed in the
``output_dir`` you supply.  Every row carries the full session context so
the file stays self-contained and can be loaded from any downstream tool
without needing a separate metadata file.

Example usage
-------------
From `run_annotation_session`::

    from inference.perf_log import PerformanceLogger

    perf = PerformanceLogger(output_dir, session_id=run_name)
    perf.log_session_start(model_path=model_path, n_images=len(image_paths),
                           device=device, imgsz=imgsz, conf=conf, iou=iou)

    t0 = time.perf_counter()
    pred_masks, confidences = load_predictions_from_model(...)
    inference_s = time.perf_counter() - t0

    perf.log_image(image_path=image_path, n_raw_detections=len(pred_masks),
                   inference_time_s=inference_s, image=image)

    perf.log_session_end(total_time_s=time.perf_counter() - session_t0,
                         n_processed=processed_count)
"""

import csv
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  Version / hardware helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_software_info() -> Dict[str, str]:
    """Collect software version strings.  Missing packages return 'n/a'."""
    import sys

    info: Dict[str, str] = {
        "python_version": sys.version.split()[0],
        "os": platform.platform(),
    }

    for pkg, attr in [
        ("torch", "torch"),
        ("ultralytics", "ultralytics"),
        ("cv2", "cv2"),
        ("numpy", "numpy"),
    ]:
        try:
            mod = __import__(pkg)
            info[f"{attr}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"{attr}_version"] = "n/a"

    return info


def _get_hardware_info() -> Dict[str, Any]:
    """Collect CPU, RAM, and GPU details."""
    info: Dict[str, Any] = {}

    # ── CPU ──────────────────────────────────────────────────────────────────
    info["cpu_model"] = platform.processor() or "unknown"
    try:
        import psutil

        info["cpu_cores_physical"] = psutil.cpu_count(logical=False) or "n/a"
        info["cpu_cores_logical"] = psutil.cpu_count(logical=True) or "n/a"
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["ram_available_gb"] = round(psutil.virtual_memory().available / 1e9, 1)
    except ImportError:
        info["cpu_cores_physical"] = "n/a"
        info["cpu_cores_logical"] = "n/a"
        info["ram_total_gb"] = "n/a"
        info["ram_available_gb"] = "n/a"

    # ── GPU ──────────────────────────────────────────────────────────────────
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_gb"] = round(props.total_memory / 1e9, 1)
            info["gpu_cuda_capability"] = f"{props.major}.{props.minor}"
        else:
            info["gpu_name"] = "none (CPU only)"
            info["gpu_vram_gb"] = 0
            info["gpu_cuda_capability"] = "n/a"
    except Exception:
        info["gpu_name"] = "unknown"
        info["gpu_vram_gb"] = "n/a"
        info["gpu_cuda_capability"] = "n/a"

    return info


# ──────────────────────────────────────────────────────────────────────────────
# CSV schema
# ──────────────────────────────────────────────────────────────────────────────

_FIELDNAMES: List[str] = [
    # identity
    "timestamp",
    "session_id",
    "row_type",                # "session_start" | "image" | "session_end"
    # image-level
    "image_name",
    "image_width",
    "image_height",
    "inference_time_s",
    "n_raw_detections",
    # session totals (filled on session_end row)
    "total_session_time_s",
    "n_images_submitted",
    "n_images_processed",
    # run config
    "model_path",
    "device",
    "imgsz",
    "conf",
    "iou",
    # software
    "python_version",
    "torch_version",
    "ultralytics_version",
    "cv2_version",
    "os",
    # hardware
    "cpu_model",
    "cpu_cores_physical",
    "cpu_cores_logical",
    "ram_total_gb",
    "ram_available_gb",
    "gpu_name",
    "gpu_vram_gb",
    "gpu_cuda_capability",
]


class PerformanceLogger:
    """
    Append-safe CSV telemetry logger for inference sessions.

    Parameters
    ----------
    output_dir:
        Directory where ``performance_log.csv`` will be written.
        Created if it does not exist yet.
    session_id:
        Human-readable run identifier (e.g. ``"session_20260430_153000"``).
    log_name:
        File name of the CSV (default ``"performance_log.csv"``).
    """

    def __init__(
        self,
        output_dir: Path,
        session_id: str,
        log_name: str = "performance_log.csv",
    ) -> None:
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._csv = self._out / log_name
        self._session_id = session_id
        self._sw = _get_software_info()
        self._hw = _get_hardware_info()
        self._context: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def log_session_start(
        self,
        model_path: "str | Path",
        n_images: int,
        device: str = "cpu",
        imgsz: int = 1024,
        conf: float = 0.25,
        iou: float = 0.7,
    ) -> None:
        """Write the session-start sentinel row with all context columns filled."""
        self._context = {
            "model_path": str(model_path),
            "device": device,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "n_images_submitted": n_images,
        }
        self._write_row(
            row_type="session_start",
            overrides={"n_images_submitted": n_images},
        )

    def log_image(
        self,
        image_path: "str | Path",
        n_raw_detections: int,
        inference_time_s: float,
        image: "Optional[np.ndarray]" = None,
    ) -> None:
        """Write one row for a single processed image."""
        ip = Path(image_path)
        h, w = (int(image.shape[0]), int(image.shape[1])) if image is not None else ("n/a", "n/a")
        self._write_row(
            row_type="image",
            overrides={
                "image_name": ip.name,
                "image_width": w,
                "image_height": h,
                "inference_time_s": round(inference_time_s, 4),
                "n_raw_detections": n_raw_detections,
            },
        )

    def log_session_end(
        self,
        total_time_s: float,
        n_processed: int,
    ) -> None:
        """Write the session-end summary row."""
        self._write_row(
            row_type="session_end",
            overrides={
                "total_session_time_s": round(total_time_s, 2),
                "n_images_processed": n_processed,
                "n_images_submitted": self._context.get("n_images_submitted", "n/a"),
            },
        )
        print(
            f"[perf] session '{self._session_id}' — "
            f"{n_processed} images in {total_time_s:.1f}s  "
            f"({total_time_s / max(n_processed, 1):.2f}s/image) — "
            f"log: {self._csv}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_base_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {name: "" for name in _FIELDNAMES}
        row.update(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            session_id=self._session_id,
        )
        row.update(self._sw)
        row.update(self._hw)
        row.update(self._context)
        return row

    def _write_row(
        self,
        row_type: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        row = self._build_base_row()
        row["row_type"] = row_type
        if overrides:
            row.update(overrides)

        file_exists = self._csv.exists()
        with self._csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in _FIELDNAMES})
