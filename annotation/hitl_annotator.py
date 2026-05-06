import argparse
import csv
import hashlib
import importlib
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon, Rectangle as MplRectangle
from matplotlib.widgets import Button, PolygonSelector, RectangleSelector
from scipy.optimize import linear_sum_assignment

from analysis.morphology import (
    analyze_instances,
    save_morphology_csv,
    draw_ellipses_on_image,
    plot_morphology_distributions,
)
from annotation.io import (
    copy_image_to_dataset,
    mask_to_polygon,
    polygons_to_masks,
    refine_polygon_from_box,
    write_yolo_box_labels,
    write_yolo_polygon_labels,
)
from inference.perf_log import PerformanceLogger
from training.train_yolo import load_predictions_from_model


DEFAULT_MODEL_PATH = (
    r"C:\Users\ML-2619\Desktop\Pujan Cryo\cryo-ev pipeline\Model Training by Yifei"
    r"\round_2\results_yolov8_heavy_augmentation\training\vesicle_instance_seg_v2\weights"
    r"\best.pt"
)


def _default_output_dir_for(input_images: Path | None) -> Path:
    if input_images is None:
        return Path.cwd() / "hitl_annotation_output"

    base_name = input_images.stem if input_images.is_file() else input_images.name
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base_name).strip("_")
    safe_name = safe_name or "annotation_session"
    return input_images.parent / f"{safe_name}_hitl_output"


def _pick_paths_with_dialogs(
    input_images: Path | None,
    output_dir: Path | None,
    model_path: str | None,
) -> tuple[Path, Path, str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("Tkinter is unavailable, so the picker UI cannot be opened.") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected_input = str(input_images) if input_images else filedialog.askdirectory(
        title="Select input image folder for annotation"
    )
    if not selected_input:
        raise SystemExit("No input image folder selected.")

    default_output = str(output_dir or _default_output_dir_for(Path(selected_input)))
    selected_output = filedialog.askdirectory(
        title="Select output folder for reviewed labels and stats",
        initialdir=str(Path(default_output).parent if Path(default_output).parent.exists() else Path.cwd()),
        mustexist=False,
    )
    if not selected_output:
        selected_output = default_output

    selected_model = str(model_path) if model_path else filedialog.askopenfilename(
        title="Select model weights (.pt)",
        filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
    )
    if not selected_model:
        selected_model = filedialog.askdirectory(title="Or select a folder containing best.pt")
    if not selected_model:
        raise SystemExit("No model weights selected.")

    root.destroy()
    return Path(selected_input), Path(selected_output), selected_model


def _resolve_runtime_paths(args: argparse.Namespace) -> tuple[List[Path], Path, Path]:
    # args.input_images is now a list[Path] or None
    input_images: List[Path] | None = args.input_images
    first = input_images[0] if input_images else None
    output_dir = args.output_dir or _default_output_dir_for(first)
    model_path_raw = args.model_path

    if args.gui or not input_images or args.output_dir is None:
        single, output_dir, model_path_raw = _pick_paths_with_dialogs(
            input_images=first,
            output_dir=args.output_dir,
            model_path=model_path_raw,
        )
        if not input_images:
            input_images = [single]

    if not input_images:
        raise SystemExit("Provide --input-images or use --gui.")

    model_path = _resolve_model_path(model_path_raw)
    return [Path(p) for p in input_images], Path(output_dir), model_path


def _resolve_model_path(model_path: str) -> Path:
    candidate = Path(model_path)
    if candidate.is_dir():
        candidate = candidate / "best.pt"
    if not candidate.exists():
        raise FileNotFoundError(f"Model weights not found: {candidate}")
    return candidate


def _collect_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def _collect_images_multi(input_paths: List[Path]) -> List[Path]:
    seen: set = set()
    result: List[Path] = []
    for p in input_paths:
        for img in _collect_images(p):
            if img not in seen:
                seen.add(img)
                result.append(img)
    return result


def _draw_polygons(image: np.ndarray, polygons: Iterable[np.ndarray], color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.int32)
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=thickness)

    return canvas


def _clean_polygon(polygon: np.ndarray) -> np.ndarray | None:
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 3:
        return None

    filtered = [poly[0]]
    for point in poly[1:]:
        if np.linalg.norm(point - filtered[-1]) >= 1.0:
            filtered.append(point)
    poly = np.asarray(filtered, dtype=np.float32)

    if poly.shape[0] >= 2 and np.linalg.norm(poly[0] - poly[-1]) < 1.0:
        poly = poly[:-1]

    rounded_unique = np.unique(np.round(poly, 1), axis=0)
    if poly.shape[0] < 3 or rounded_unique.shape[0] < 3:
        return None

    if abs(cv2.contourArea(poly.astype(np.float32))) < 4.0:
        return None

    return poly


# ---------------------------------------------------------------------------
# Image resolution helpers
# ---------------------------------------------------------------------------

# Images wider/taller than this multiple of imgsz trigger the resize prompt.
_RESCALE_THRESHOLD = 1.5


def _prompt_rescale(image_path: Path, orig_w: int, orig_h: int, target: int) -> bool:
    """Ask the user whether to downscale a high-resolution image before inference.

    Returns True if the user agrees to rescale, False to keep the original.
    """
    msg = (
        f"High-resolution image detected:\n"
        f"  {image_path.name}  ({orig_w} × {orig_h} px)\n\n"
        f"The model was trained on ~{target} px images.  Running inference on the\n"
        f"full-resolution image may produce poor detections because the vesicles\n"
        f"appear larger in pixel units than the model expects.\n\n"
        f"Downscale to {target} px before inference?\n"
        f"(Annotations and saved images will also use the downscaled size.)"
    )
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        answer = messagebox.askyesno("Resolution mismatch", msg, default="yes")
        root.destroy()
        return bool(answer)
    except Exception:
        # Fallback to terminal prompt if Tkinter is unavailable.
        print(f"\n[WARNING] {msg}")
        resp = input("Downscale? [Y/n]: ").strip().lower()
        return resp in ("", "y", "yes")


def _rescale_image_for_inference(
    image: np.ndarray,
    image_path: Path,
    imgsz: int,
    _rescale_cache: dict,
) -> tuple[np.ndarray, Path, float]:
    """Return (image, path_for_inference, scale_factor).

    If the image exceeds _RESCALE_THRESHOLD * imgsz on either axis, the user is
    prompted once per unique image path.  On agreement the image is downscaled
    and written to a temp file; scale_factor < 1 means coordinates output by
    inference must be multiplied by 1/scale_factor to map back to original size.
    On refusal scale_factor == 1.0 and the original image + path are returned.
    _rescale_cache maps (str(image_path), imgsz) -> bool so the dialog only
    appears once per path across repeated calls.
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)
    threshold = int(_RESCALE_THRESHOLD * imgsz)
    if max_dim <= threshold:
        return image, image_path, 1.0

    cache_key = (str(image_path), imgsz)
    if cache_key not in _rescale_cache:
        _rescale_cache[cache_key] = _prompt_rescale(image_path, w, h, imgsz)
    if not _rescale_cache[cache_key]:
        return image, image_path, 1.0

    scale = imgsz / max_dim
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    suffix = image_path.suffix or ".png"
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=suffix, prefix="cryo_rescale_")
    import os
    os.close(tmp_fd)
    cv2.imwrite(tmp_name, resized)
    print(f"  [rescale] {w}×{h} → {new_w}×{new_h} (scale={scale:.3f}). Temp file: {tmp_name}")
    return resized, Path(tmp_name), scale


def _polygon_to_box(polygon: np.ndarray) -> np.ndarray:
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    x0 = float(np.min(poly[:, 0]))
    y0 = float(np.min(poly[:, 1]))
    x1 = float(np.max(poly[:, 0]))
    y1 = float(np.max(poly[:, 1]))
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def _clean_box(box: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray | None:
    x0, y0, x1, y1 = [float(v) for v in np.asarray(box, dtype=np.float32).reshape(4)]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))

    h, w = image_shape[:2]
    x0 = float(np.clip(x0, 0, w - 1))
    x1 = float(np.clip(x1, 0, w - 1))
    y0 = float(np.clip(y0, 0, h - 1))
    y1 = float(np.clip(y1, 0, h - 1))

    if (x1 - x0) < 4 or (y1 - y0) < 4:
        return None
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def _box_to_polygon(box: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in np.asarray(box, dtype=np.float32).reshape(4)]
    return np.asarray(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        dtype=np.float32,
    )


def _prepare_display_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            arr = arr[..., 0]

    gray = arr.astype(np.float32)
    lo = float(np.percentile(gray, 1))
    hi = float(np.percentile(gray, 99))
    if hi <= lo:
        lo = float(gray.min())
        hi = float(gray.max()) if float(gray.max()) > lo else lo + 1.0
    return np.clip((gray - lo) / (hi - lo), 0.0, 1.0)


def _safe_shapes_data_to_polygons(data: Iterable[np.ndarray]) -> List[np.ndarray]:
    polygons: List[np.ndarray] = []
    for shape in data:
        poly = np.asarray(shape, dtype=np.float32)
        if poly.ndim != 2 or poly.shape[0] < 3:
            continue
        yx = poly[:, :2]
        cleaned = _clean_polygon(np.stack([yx[:, 1], yx[:, 0]], axis=1))
        if cleaned is not None:
            polygons.append(cleaned)
    return polygons


def _xy_polygons_to_napari(polygons: Iterable[np.ndarray]) -> List[np.ndarray]:
    napari_polygons: List[np.ndarray] = []
    for polygon in polygons:
        cleaned = _clean_polygon(np.asarray(polygon, dtype=np.float32))
        if cleaned is None:
            continue
        napari_polygons.append(np.stack([cleaned[:, 1], cleaned[:, 0]], axis=1))
    return napari_polygons


class SimplePolygonEditor:
    def __init__(
        self,
        image: np.ndarray,
        init_polygons_xy: List[np.ndarray],
        image_name: str,
        init_confidences: List[float] | None = None,
        class_id: int = 0,
        region_rows: int = 2,
        region_cols: int = 4,
        review_mode: str = "polygon",
    ):
        self.image = image
        self.image_name = image_name
        self.display_image = _prepare_display_image(image)
        self.selected_idx: int | None = None
        self.action = "skip"
        self.mode = "navigate"
        self.selector = None
        self.pending_replace_idx: int | None = None
        self.overlay_artists: List[object] = []
        self.buttons: List[Button] = []
        self.press_event: dict | None = None
        self.is_panning = False
        self.next_object_id = 1
        self.region_rows = max(1, int(region_rows))
        self.region_cols = max(1, int(region_cols))
        self.review_mode = review_mode
        self.region_index = 0
        self.region_bounds: List[tuple[float, float, float, float]] = []
        self.dragging_vertex: tuple[int, int] | None = None
        self.pan_start: tuple | None = None  # (x_disp, y_disp, xlim, ylim)

        # --- Custom polygon drawing state (replaces PolygonSelector) ---
        self._poly_drawing: bool = False
        self._poly_pts: List[tuple] = []
        self._poly_is_freehand: bool = False
        self._poly_drag_active: bool = False
        self._poly_drag_start: Tuple[float, float] = (0.0, 0.0)
        self._poly_last_pt: Tuple[float, float] = (0.0, 0.0)
        self._poly_preview_artists: List[object] = []

        self.fig = None
        self.ax_overlay = None
        self.ax_raw = None
        self.ax_list = None
        self.status_text = None
        self.region_text = None

        confidences = init_confidences or [None] * len(init_polygons_xy)
        self.objects: List[Dict] = []
        for polygon, confidence in zip(init_polygons_xy, confidences):
            cleaned = _clean_polygon(polygon)
            if cleaned is None:
                continue
            self.objects.append(
                {
                    "object_id": self.next_object_id,
                    "polygon": cleaned,
                    "box": _polygon_to_box(cleaned),
                    "kept": True,
                    "class_id": class_id,
                    "confidence": confidence,
                    "source": "auto",
                }
            )
            self.next_object_id += 1

    def run(self) -> Tuple[str, List[np.ndarray]]:
        self.fig = plt.figure(figsize=(24, 12))
        grid = self.fig.add_gridspec(1, 3, width_ratios=[1.7, 1.7, 0.5], wspace=0.025)
        self.ax_overlay = self.fig.add_subplot(grid[0, 0])
        self.ax_raw = self.fig.add_subplot(grid[0, 1], sharex=self.ax_overlay, sharey=self.ax_overlay)
        self.ax_list = self.fig.add_subplot(grid[0, 2])
        plt.subplots_adjust(bottom=0.16, left=0.02, right=0.985, top=0.95)

        try:
            self.fig.canvas.manager.set_window_title(f"cryoEV HITL - {self.image_name}")
        except Exception:
            pass

        self.ax_overlay.imshow(self.display_image, cmap="gray", vmin=0, vmax=1, origin="upper", interpolation="nearest")
        self.ax_raw.imshow(self.display_image, cmap="gray", vmin=0, vmax=1, origin="upper", interpolation="nearest")

        self.ax_overlay.set_title("Polygon review" if self.review_mode == "polygon" else "Box review")
        self.ax_raw.set_title("Raw image")
        self.ax_list.set_title("Detected objects")
        self.ax_list.axis("off")
        for ax in (self.ax_overlay, self.ax_raw):
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("black")

        status_message = (
            "Click an object or list entry to select it. Drag highlighted polygon vertices to refine the shape."
            if self.review_mode == "polygon"
            else "Click an object or list entry to select it. Use Prev/Next Region to move across the image."
        )
        self.status_text = self.fig.text(
            0.02,
            0.02,
            status_message,
            fontsize=9,
        )
        self.region_text = self.fig.text(0.72, 0.02, "", fontsize=9, fontweight="bold")

        self._build_regions()
        self._add_buttons()
        self._set_region_view(self.region_index)
        self._refresh_overlay()

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        print("\nSimple review controls:")
        if self.review_mode == "polygon":
            print("  - Left: image with bold polygon overlay")
            print("  - Drag the highlighted polygon vertices to refine the contour")
            print("  - Use Add Polygon or Redraw Polygon for larger changes")
            print("  - Buttons: Prev Region, Next Region, Add Polygon, Redraw Polygon, Keep, Reject, Delete, Save & Next, Skip, Quit")
        else:
            print("  - Left: image with bold box overlay")
            print("  - Buttons: Prev Region, Next Region, Add Box, Adjust Box, Keep, Reject, Delete, Save & Next, Skip, Quit")
        print("  - Middle: raw image only")
        print("  - Right: clickable object list with class / confidence / keep-reject state")
        print("  - Use Prev Region / Next Region to move through the default 8 review panes")
        print("  - Mouse wheel zooms, middle-mouse drag pans within the current pane\n")

        plt.show()
        kept_polygons = [obj["polygon"] for obj in self.objects if obj["kept"]]
        return self.action, kept_polygons

    def _add_buttons(self) -> None:
        edit_add_label = "Add Polygon" if self.review_mode == "polygon" else "Add Box"
        edit_adjust_label = "Redraw Polygon" if self.review_mode == "polygon" else "Adjust Box"
        button_defs = [
            ("\u25c4 Prev", self._prev_region),
            ("Next \u25ba", self._next_region),
            ("Full View [f]", self._full_image_view),
            (edit_add_label, self._start_add),
            (edit_adjust_label, self._start_replace),
            ("Keep", self._keep_selected),
            ("Reject", self._reject_selected),
            ("Delete", self._delete_selected),
            ("Save & Next", self._save_and_next),
            ("Skip", self._skip),
            ("Quit", self._quit),
        ]
        x = 0.02
        n_btns = len(button_defs)
        gap = 0.006
        width = (0.978 - x - gap * (n_btns - 1)) / n_btns
        for label, callback in button_defs:
            ax_btn = self.fig.add_axes([x, 0.08, width, 0.06])
            btn = Button(ax_btn, label)
            btn.on_clicked(callback)
            self.buttons.append(btn)
            x += width + gap

    def _set_status(self, message: str) -> None:
        if self.status_text is not None:
            self.status_text.set_text(message)
        if self.region_text is not None and self.region_bounds:
            n = len(self.region_bounds)
            if n >= 3 and self.region_index == 0:
                label = "Overview (start)"
            elif n >= 3 and self.region_index == n - 1:
                label = "Overview (end)"
            else:
                # sub-region number within the non-overview entries
                sub = self.region_index if n < 3 else self.region_index
                n_sub = n if n < 3 else n - 2
                label = f"Region {sub}/{n_sub}  ({self.region_cols}\u00d7{self.region_rows})"
            self.region_text.set_text(label)
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def _format_object_row(self, idx: int, obj: Dict) -> str:
        conf = obj.get("confidence")
        conf_text = f"{conf:.2f}" if conf is not None else "manual"
        class_text = f"cls {obj.get('class_id', 0)}"
        status = "KEEP" if obj.get("kept", True) else "REJECT"
        return f"#{obj['object_id']:03d}  {class_text:<6}  {conf_text:<6}  {status:<6}"

    def _refresh_list(self) -> None:
        if self.ax_list is None:
            return

        self.ax_list.clear()
        self.ax_list.set_title("Detected objects")
        self.ax_list.set_xlim(0, 1)
        self.ax_list.set_ylim(0, max(len(self.objects) + 1, 2))
        self.ax_list.invert_yaxis()
        self.ax_list.axis("off")
        self.ax_list.text(0.02, 0.4, "ID        Class   Conf    State", fontsize=9, fontweight="bold")

        for idx, obj in enumerate(self.objects, start=1):
            color = "green" if obj.get("kept", True) else "firebrick"
            bbox = None
            if self.selected_idx == idx - 1:
                bbox = dict(facecolor="khaki", edgecolor="goldenrod", boxstyle="round,pad=0.2")
            self.ax_list.text(
                0.02,
                idx,
                self._format_object_row(idx - 1, obj),
                fontsize=8.5,
                color=color,
                bbox=bbox,
            )

    def _refresh_overlay(self) -> None:
        for artist in self.overlay_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.overlay_artists = []

        if self.ax_overlay is None:
            return

        for idx, obj in enumerate(self.objects):
            polygon = np.asarray(obj["polygon"], dtype=np.float32)
            x0, y0, x1, y1 = obj["box"]
            is_selected = idx == self.selected_idx
            is_kept = obj.get("kept", True)
            color = "cyan" if is_selected else ("magenta" if is_kept else "red")
            linewidth = 3.6 if is_selected else 2.4
            linestyle = "-" if is_kept else "--"

            if self.review_mode == "polygon":
                patch = MplPolygon(
                    polygon,
                    closed=True,
                    fill=False,
                    edgecolor=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    alpha=0.98,
                    joinstyle="round",
                )
                cx = float(np.mean(polygon[:, 0]))
                cy = float(np.mean(polygon[:, 1]))
            else:
                patch = MplRectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    alpha=0.98,
                )
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0

            self.ax_overlay.add_patch(patch)
            self.overlay_artists.append(patch)

            label = self.ax_overlay.text(
                cx,
                cy,
                str(obj["object_id"]),
                color=color,
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.0),
            )
            self.overlay_artists.append(label)

            if is_selected:
                points = polygon if self.review_mode == "polygon" else np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
                scatter = self.ax_overlay.scatter(points[:, 0], points[:, 1], s=34, c=color, edgecolors="black", linewidths=0.5)
                self.overlay_artists.append(scatter)

        self._refresh_list()
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def _find_object_at(self, x: float, y: float) -> int | None:
        if self.review_mode == "polygon":
            best_idx = None
            best_distance = -np.inf
            for idx, obj in enumerate(self.objects):
                contour = np.asarray(obj["polygon"], dtype=np.float32).reshape(-1, 1, 2)
                distance = cv2.pointPolygonTest(contour, (float(x), float(y)), True)
                if distance > best_distance:
                    best_distance = distance
                    best_idx = idx
            if best_idx is not None and best_distance >= -8.0:
                return best_idx
            return None

        containing: list[tuple[float, int]] = []
        nearby: list[tuple[float, int]] = []
        for idx, obj in enumerate(self.objects):
            x0, y0, x1, y1 = obj["box"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                area = max((x1 - x0) * (y1 - y0), 1.0)
                containing.append((area, idx))
            else:
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                dist = (cx - x) ** 2 + (cy - y) ** 2
                nearby.append((dist, idx))
        if containing:
            return min(containing)[1]
        if nearby and min(nearby)[0] < 2500:
            return min(nearby)[1]
        return None

    def _find_vertex_at(self, x: float, y: float) -> int | None:
        if self.selected_idx is None or self.review_mode != "polygon":
            return None
        polygon = np.asarray(self.objects[self.selected_idx]["polygon"], dtype=np.float32)
        if polygon.size == 0:
            return None
        dists = np.linalg.norm(polygon - np.asarray([x, y], dtype=np.float32), axis=1)
        idx = int(np.argmin(dists))
        x0, x1 = self.ax_overlay.get_xlim()
        y0, y1 = self.ax_overlay.get_ylim()
        threshold = max(6.0, 0.02 * max(abs(x1 - x0), abs(y1 - y0)))
        if float(dists[idx]) <= threshold:
            return idx
        return None

    def _select_object(self, idx: int | None, center: bool = False) -> None:
        self.selected_idx = idx
        if idx is None:
            self._set_status("No object selected.")
        else:
            obj = self.objects[idx]
            conf = obj.get("confidence")
            conf_text = f"{conf:.3f}" if conf is not None else "manual"
            status = "kept" if obj.get("kept", True) else "rejected"
            extra = " Drag visible vertices to refine." if self.review_mode == "polygon" else ""
            self._set_status(
                f"Selected object #{obj['object_id']} | class={obj.get('class_id', 0)} | confidence={conf_text} | {status}.{extra}"
            )
            if center:
                self._center_on_object(idx)
        self._refresh_overlay()

    def _center_on_object(self, idx: int) -> None:
        if self.review_mode == "polygon":
            polygon = np.asarray(self.objects[idx]["polygon"], dtype=np.float32)
            x_min, y_min = polygon.min(axis=0)
            x_max, y_max = polygon.max(axis=0)
        else:
            x_min, y_min, x_max, y_max = self.objects[idx]["box"]
        pad = max(25.0, 0.5 * max(x_max - x_min, y_max - y_min))
        xlim = (x_min - pad, x_max + pad)
        ylim = (y_max + pad, y_min - pad)
        self.ax_overlay.set_xlim(*xlim)
        self.ax_overlay.set_ylim(*ylim)
        self.ax_raw.set_xlim(*xlim)
        self.ax_raw.set_ylim(*ylim)
        self.fig.canvas.draw_idle()

    def _on_press(self, event) -> None:
        # --- Custom polygon drawing (add / replace mode) ---------------
        if (
            self.mode in ("add", "replace")
            and self.review_mode == "polygon"
            and event.inaxes == self.ax_overlay
            and event.button == 1
            and event.xdata is not None
            and event.ydata is not None
        ):
            if event.dblclick and not self._poly_is_freehand:
                # Double-click finalises the click-by-click polygon
                self._finish_custom_polygon()
            else:
                # Record press start; vertex/freehand decided on motion/release
                self._poly_drag_active = True
                self._poly_drag_start = (float(event.xdata), float(event.ydata))
                self._poly_last_pt = (float(event.xdata), float(event.ydata))
            return

        if event.inaxes == self.ax_list and event.button == 1 and event.ydata is not None:
            row_idx = int(round(event.ydata)) - 1
            if 0 <= row_idx < len(self.objects):
                self._select_object(row_idx, center=True)
            return

        # Middle-mouse button: start pan
        if event.button == 2 and event.inaxes in (self.ax_overlay, self.ax_raw):
            self.pan_start = (
                event.x, event.y,
                self.ax_overlay.get_xlim(),
                self.ax_overlay.get_ylim(),
            )
            return

        if self.mode != "navigate" or self.selector is not None:
            return
        if event.inaxes not in (self.ax_overlay, self.ax_raw):
            return
        if event.xdata is None or event.ydata is None or event.button != 1:
            return

        if self.review_mode == "polygon" and event.inaxes == self.ax_overlay and self.selected_idx is not None:
            vertex_idx = self._find_vertex_at(float(event.xdata), float(event.ydata))
            if vertex_idx is not None:
                self.dragging_vertex = (self.selected_idx, vertex_idx)
                self._set_status(
                    f"Dragging vertex {vertex_idx + 1} of object #{self.objects[self.selected_idx]['object_id']}."
                )
                return

        idx = self._find_object_at(float(event.xdata), float(event.ydata))
        self._select_object(idx, center=False)

    def _on_motion(self, event) -> None:
        # --- Freehand polygon drag detection ---------------------------
        if self._poly_drag_active and self.mode in ("add", "replace") and self.review_mode == "polygon":
            if event.inaxes == self.ax_overlay and event.xdata is not None and event.ydata is not None:
                x, y = float(event.xdata), float(event.ydata)
                if not self._poly_is_freehand:
                    # Measure drag distance in screen pixels
                    ax = self.ax_overlay
                    x0, x1 = ax.get_xlim()
                    y0, y1 = ax.get_ylim()
                    fig_w_px, fig_h_px = self.fig.get_size_inches() * self.fig.dpi
                    ax_bbox = ax.get_position()
                    scale_x = (ax_bbox.width * fig_w_px) / max(abs(x1 - x0), 1)
                    scale_y = (ax_bbox.height * fig_h_px) / max(abs(y1 - y0), 1)
                    sx, sy = self._poly_drag_start
                    dist_px = ((x - sx) * scale_x) ** 2 + ((y - sy) * scale_y) ** 2
                    if dist_px > 64:  # > 8-pixel threshold
                        self._poly_is_freehand = True
                        if not self._poly_drawing:
                            self._poly_drawing = True
                            self._poly_pts = [self._poly_drag_start]
                        self._set_status(
                            "Freehand mode: hold and drag to draw polygon outline. Release to finish.  ESC to cancel."
                        )
                if self._poly_is_freehand:
                    lx, ly = self._poly_last_pt
                    if (x - lx) ** 2 + (y - ly) ** 2 > 4:
                        self._poly_pts.append((x, y))
                        self._poly_last_pt = (x, y)
                        self._update_poly_preview()
            return

        # Middle-mouse pan
        if self.pan_start is not None:
            px, py, (x0, x1), (y0, y1) = self.pan_start
            # Convert pixel delta to data-space units
            ax = self.ax_overlay
            fig_w, fig_h = self.fig.get_size_inches() * self.fig.dpi
            ax_bbox = ax.get_position()
            data_w = (x1 - x0) / (ax_bbox.width * fig_w) if ax_bbox.width > 0 else 1
            data_h = (y1 - y0) / (ax_bbox.height * fig_h) if ax_bbox.height > 0 else 1
            dx = (event.x - px) * data_w
            dy = (event.y - py) * data_h
            self.ax_overlay.set_xlim(x0 - dx, x1 - dx)
            self.ax_overlay.set_ylim(y0 - dy, y1 - dy)
            self.ax_raw.set_xlim(x0 - dx, x1 - dx)
            self.ax_raw.set_ylim(y0 - dy, y1 - dy)
            self.fig.canvas.draw_idle()
            return

        if self.dragging_vertex is None or self.review_mode != "polygon":
            return
        if event.inaxes != self.ax_overlay or event.xdata is None or event.ydata is None:
            return

        obj_idx, vertex_idx = self.dragging_vertex
        polygon = np.asarray(self.objects[obj_idx]["polygon"], dtype=np.float32).copy()
        h, w = self.image.shape[:2]
        polygon[vertex_idx, 0] = float(np.clip(event.xdata, 0, w - 1))
        polygon[vertex_idx, 1] = float(np.clip(event.ydata, 0, h - 1))

        if abs(cv2.contourArea(polygon.astype(np.float32))) >= 4.0:
            self.objects[obj_idx]["polygon"] = polygon
            self.objects[obj_idx]["box"] = _polygon_to_box(polygon)
            self._refresh_overlay()

    def _on_release(self, event) -> None:
        # --- Polygon drawing release -----------------------------------
        if self._poly_drag_active and event.button == 1:
            self._poly_drag_active = False
            if self._poly_is_freehand:
                # Freehand complete: close and save
                if event.xdata is not None and event.ydata is not None and event.inaxes == self.ax_overlay:
                    self._poly_pts.append((float(event.xdata), float(event.ydata)))
                self._finish_custom_polygon()
            else:
                # Plain click: add one vertex, wait for double-click to finish
                if event.xdata is not None and event.ydata is not None and event.inaxes == self.ax_overlay:
                    x, y = float(event.xdata), float(event.ydata)
                    if not self._poly_drawing:
                        self._poly_drawing = True
                        self._poly_pts = []
                    self._poly_pts.append((x, y))
                    self._update_poly_preview()
                    self._set_status(
                        f"{len(self._poly_pts)} vertex/vertices placed. "
                        "Click to add more, double-click to finish, ESC to cancel."
                    )
            return

        if event.button == 2:
            self.pan_start = None
            return
        if self.dragging_vertex is None:
            return
        obj_idx, _vertex_idx = self.dragging_vertex
        self.dragging_vertex = None
        self._set_status(f"Updated polygon for object #{self.objects[obj_idx]['object_id']}.")
        self._refresh_overlay()

    def _on_scroll(self, event) -> None:
        if event.inaxes not in (self.ax_overlay, self.ax_raw):
            return
        if event.xdata is None or event.ydata is None:
            return

        direction_up = getattr(event, "button", None) == "up" or getattr(event, "step", 0) > 0
        scale = 1 / 1.15 if direction_up else 1.15
        xdata = float(event.xdata)
        ydata = float(event.ydata)
        x0, x1 = self.ax_overlay.get_xlim()
        y0, y1 = self.ax_overlay.get_ylim()

        new_width = (x1 - x0) * scale
        new_height = (y1 - y0) * scale
        relx = (x1 - xdata) / (x1 - x0) if x1 != x0 else 0.5
        rely = (y1 - ydata) / (y1 - y0) if y1 != y0 else 0.5

        new_xlim = (xdata - new_width * (1 - relx), xdata + new_width * relx)
        new_ylim = (ydata - new_height * (1 - rely), ydata + new_height * rely)
        self.ax_overlay.set_xlim(*new_xlim)
        self.ax_overlay.set_ylim(*new_ylim)
        self.ax_raw.set_xlim(*new_xlim)
        self.ax_raw.set_ylim(*new_ylim)
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key == "s":
            self._save_and_next(None)
        elif event.key == "k":
            self._skip(None)
        elif event.key == "q":
            self._quit(None)
        elif event.key == "r":
            self._reject_selected(None)
        elif event.key == "a":
            self._keep_selected(None)
        elif event.key in ("right", "]"):
            self._next_region(None)
        elif event.key in ("left", "["):
            self._prev_region(None)
        elif event.key == "f":
            self._full_image_view(None)
        elif event.key == "escape":
            if self.mode in ("add", "replace") and self.review_mode == "polygon":
                self._cancel_poly_draw()

    def _build_regions(self) -> None:
        h, w = self.image.shape[:2]
        x_edges = np.linspace(0, w, self.region_cols + 1)
        y_edges = np.linspace(0, h, self.region_rows + 1)
        overlap_x = 0.06 * w / self.region_cols
        overlap_y = 0.06 * h / self.region_rows
        # Full-image overview at start
        self.region_bounds = [(0.0, float(w), 0.0, float(h))]
        for row in range(self.region_rows):
            for col in range(self.region_cols):
                x0 = max(0.0, x_edges[col] - overlap_x)
                x1 = min(float(w), x_edges[col + 1] + overlap_x)
                y0 = max(0.0, y_edges[row] - overlap_y)
                y1 = min(float(h), y_edges[row + 1] + overlap_y)
                self.region_bounds.append((x0, x1, y0, y1))
        # Full-image overview at end
        self.region_bounds.append((0.0, float(w), 0.0, float(h)))

    def _set_region_view(self, index: int) -> None:
        if not self.region_bounds:
            return
        self.region_index = index % len(self.region_bounds)
        x0, x1, y0, y1 = self.region_bounds[self.region_index]
        self.ax_overlay.set_xlim(x0, x1)
        self.ax_overlay.set_ylim(y1, y0)
        self.ax_raw.set_xlim(x0, x1)
        self.ax_raw.set_ylim(y1, y0)
        n = len(self.region_bounds)
        if n >= 3 and self.region_index == 0:
            self._set_status(
                "Full image overview (start) \u2014 scan for any obvious issues, then press Next [\u2192] to step through sub-regions.  "
                "[f] = full view at any time."
            )
        elif n >= 3 and self.region_index == n - 1:
            self._set_status(
                "Full image overview (end) \u2014 review all annotations, then Save & Next [s] to proceed.  "
                "Press Next [\u2192] to auto-save."
            )
        else:
            self._set_status(self.status_text.get_text() if self.status_text is not None else "")

    def _full_image_view(self, _event=None) -> None:
        if self.ax_overlay is None:
            return
        h, w = self.image.shape[:2]
        self.ax_overlay.set_xlim(0, w)
        self.ax_overlay.set_ylim(h, 0)
        self.ax_raw.set_xlim(0, w)
        self.ax_raw.set_ylim(h, 0)
        self._set_status(
            "Full image view.  Use Prev/Next [\u2190 \u2192] to return to sub-regions,  [f] to come back here."
        )
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def _prev_region(self, _event) -> None:
        self._set_region_view(self.region_index - 1)

    def _next_region(self, _event) -> None:
        if self.region_bounds and self.region_index >= len(self.region_bounds) - 1:
            # Already at end overview — auto-save and move on
            self._save_and_next(None)
            return
        self._set_region_view(self.region_index + 1)

    def _disconnect_selector(self) -> None:
        if self.selector is not None:
            try:
                self.selector.set_active(False)
            except Exception:
                pass
            try:
                self.selector.disconnect_events()
            except Exception:
                pass
            self.selector = None
        self.mode = "navigate"

    def _start_polygon_selector(self, message: str) -> None:
        if self.selector is not None:
            return
        self._set_status(message)
        self.selector = PolygonSelector(
            self.ax_overlay,
            self._finish_polygon,
            useblit=False,
        )

    def _start_box_selector(self, message: str) -> None:
        if self.selector is not None:
            return
        self._set_status(message)
        self.selector = RectangleSelector(
            self.ax_overlay,
            self._finish_box,
            useblit=False,
            button=[1],
            minspanx=4,
            minspany=4,
            spancoords="pixels",
            interactive=False,
        )
        self.selector.set_active(True)

    def _start_add(self, _event) -> None:
        self.pending_replace_idx = None
        self.mode = "add"
        if self.review_mode == "polygon":
            self._poly_drawing = False
            self._poly_pts = []
            self._poly_is_freehand = False
            self._poly_drag_active = False
            self._set_status(
                "Add Polygon: click to place vertices one by one \u2192 double-click to finish.  "
                "OR click-and-drag for freehand drawing.  ESC to cancel."
            )
        else:
            self._start_box_selector(
                "Add Box mode: click-drag a box around the EV on the left image."
            )

    def _start_replace(self, _event) -> None:
        if self.selected_idx is None:
            self._set_status(
                "Select an object first, then click Redraw Polygon."
                if self.review_mode == "polygon"
                else "Select an object first, then click Adjust Box."
            )
            return
        self.pending_replace_idx = self.selected_idx
        self.mode = "replace"
        if self.review_mode == "polygon":
            self._poly_drawing = False
            self._poly_pts = []
            self._poly_is_freehand = False
            self._poly_drag_active = False
            self._set_status(
                "Redraw Polygon: click to place vertices \u2192 double-click to finish.  "
                "OR click-and-drag for freehand.  ESC to cancel."
            )
        else:
            self._start_box_selector(
                "Adjust Box mode: click-drag the new box for the selected object on the left image."
            )

    # ------------------------------------------------------------------
    # Custom polygon drawing (click-to-add vertices + freehand drag)
    # ------------------------------------------------------------------

    def _update_poly_preview(self) -> None:
        for a in self._poly_preview_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._poly_preview_artists = []
        if not self._poly_pts or self.ax_overlay is None:
            if self.fig is not None:
                self.fig.canvas.draw_idle()
            return
        pts = np.asarray(self._poly_pts, dtype=np.float32)
        if len(pts) >= 2:
            line, = self.ax_overlay.plot(
                pts[:, 0], pts[:, 1], color="yellow", linewidth=1.5, alpha=0.85, linestyle="-", zorder=6
            )
            self._poly_preview_artists.append(line)
        if len(pts) >= 3:
            close, = self.ax_overlay.plot(
                [pts[-1, 0], pts[0, 0]], [pts[-1, 1], pts[0, 1]],
                color="yellow", linewidth=1.0, alpha=0.5, linestyle="--", zorder=6,
            )
            self._poly_preview_artists.append(close)
        sc = self.ax_overlay.scatter(pts[:, 0], pts[:, 1], s=18, c="yellow", zorder=7, edgecolors="none")
        self._poly_preview_artists.append(sc)
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def _clear_poly_preview(self) -> None:
        for a in self._poly_preview_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._poly_preview_artists = []
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def _cancel_poly_draw(self) -> None:
        self._poly_drawing = False
        self._poly_is_freehand = False
        self._poly_drag_active = False
        self._poly_pts = []
        self.pending_replace_idx = None
        self._clear_poly_preview()
        self.mode = "navigate"
        self._set_status("Drawing cancelled.")

    def _finish_custom_polygon(self) -> None:
        pts = self._poly_pts[:]
        replace_idx = self.pending_replace_idx
        self._poly_drawing = False
        self._poly_is_freehand = False
        self._poly_drag_active = False
        self._poly_pts = []
        self._clear_poly_preview()
        self.mode = "navigate"
        self.pending_replace_idx = None

        polygon = _clean_polygon(np.asarray(pts, dtype=np.float32)) if len(pts) >= 3 else None
        if polygon is None:
            self._set_status("Polygon too small or too few points — no change saved.")
            self._refresh_overlay()
            return

        box = _polygon_to_box(polygon)
        if replace_idx is not None and 0 <= replace_idx < len(self.objects):
            self.objects[replace_idx]["polygon"] = polygon
            self.objects[replace_idx]["box"] = box
            self.objects[replace_idx]["kept"] = True
            self.selected_idx = replace_idx
            self._set_status(f"Redrew polygon for object #{self.objects[replace_idx]['object_id']}.")
        else:
            self.objects.append({
                "object_id": self.next_object_id,
                "polygon": polygon,
                "box": box,
                "kept": True,
                "class_id": 0,
                "confidence": None,
                "source": "manual-polygon",
            })
            self.next_object_id += 1
            self.selected_idx = len(self.objects) - 1
            self._set_status(f"Added polygon for object #{self.objects[self.selected_idx]['object_id']}.")
        self._refresh_overlay()

    def _finish_polygon(self, verts: List[Tuple[float, float]]) -> None:
        polygon = _clean_polygon(np.asarray(verts, dtype=np.float32))
        replace_idx = self.pending_replace_idx
        self.pending_replace_idx = None
        self._disconnect_selector()

        if polygon is None:
            self._set_status("Polygon was too small or invalid; no change was saved.")
            self._refresh_overlay()
            return

        box = _polygon_to_box(polygon)
        if replace_idx is not None and 0 <= replace_idx < len(self.objects):
            self.objects[replace_idx]["polygon"] = polygon
            self.objects[replace_idx]["box"] = box
            self.objects[replace_idx]["kept"] = True
            self.selected_idx = replace_idx
            self._set_status(f"Redrew polygon for object #{self.objects[replace_idx]['object_id']}.")
        else:
            self.objects.append(
                {
                    "object_id": self.next_object_id,
                    "polygon": polygon,
                    "box": box,
                    "kept": True,
                    "class_id": 0,
                    "confidence": None,
                    "source": "manual-polygon",
                }
            )
            self.next_object_id += 1
            self.selected_idx = len(self.objects) - 1
            self._set_status(f"Added polygon for object #{self.objects[self.selected_idx]['object_id']}.")
        self._refresh_overlay()

    def _finish_box(self, eclick, erelease) -> None:
        box = _clean_box(
            np.asarray([eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata], dtype=np.float32),
            self.image.shape[:2],
        )
        replace_idx = self.pending_replace_idx
        self.pending_replace_idx = None
        self._disconnect_selector()

        if box is None:
            self._set_status("Box was too small or invalid; no change was saved.")
            self._refresh_overlay()
            return

        polygon = _box_to_polygon(box)

        if replace_idx is not None and 0 <= replace_idx < len(self.objects):
            self.objects[replace_idx]["box"] = box
            self.objects[replace_idx]["polygon"] = polygon
            self.objects[replace_idx]["kept"] = True
            self.selected_idx = replace_idx
            self._set_status(f"Adjusted box for object #{self.objects[replace_idx]['object_id']}.")
        else:
            self.objects.append(
                {
                    "object_id": self.next_object_id,
                    "polygon": polygon,
                    "box": box,
                    "kept": True,
                    "class_id": 0,
                    "confidence": None,
                    "source": "manual-box",
                }
            )
            self.next_object_id += 1
            self.selected_idx = len(self.objects) - 1
            self._set_status(f"Added box for object #{self.objects[self.selected_idx]['object_id']}.")
        self._refresh_overlay()

    def _keep_selected(self, _event) -> None:
        if self.selected_idx is None:
            self._set_status("Select an object first, then click Keep.")
            return
        self.objects[self.selected_idx]["kept"] = True
        self._set_status(f"Marked object #{self.objects[self.selected_idx]['object_id']} as KEEP.")
        self._refresh_overlay()

    def _reject_selected(self, _event) -> None:
        if self.selected_idx is None:
            self._set_status("Select an object first, then click Reject.")
            return
        self.objects[self.selected_idx]["kept"] = False
        self._set_status(f"Marked object #{self.objects[self.selected_idx]['object_id']} as REJECT.")
        self._refresh_overlay()

    def _delete_selected(self, _event) -> None:
        if self.selected_idx is None:
            self._set_status("Select an object first, then click Delete.")
            return
        removed_id = self.objects[self.selected_idx]["object_id"]
        self.objects.pop(self.selected_idx)
        self.selected_idx = None
        self._set_status(f"Deleted object #{removed_id}.")
        self._refresh_overlay()

    def _save_and_next(self, _event) -> None:
        self.action = "save"
        plt.close(self.fig)

    def _skip(self, _event) -> None:
        self.action = "skip"
        plt.close(self.fig)

    def _quit(self, _event) -> None:
        self.action = "quit"
        plt.close(self.fig)


def _run_simple_editor(
    image: np.ndarray,
    init_polygons_xy: List[np.ndarray],
    image_name: str,
    init_confidences: List[float] | None = None,
    class_id: int = 0,
    region_rows: int = 2,
    region_cols: int = 4,
    review_mode: str = "polygon",
) -> Tuple[str, List[np.ndarray]]:
    editor = SimplePolygonEditor(
        image=image,
        init_polygons_xy=init_polygons_xy,
        image_name=image_name,
        init_confidences=init_confidences,
        class_id=class_id,
        region_rows=region_rows,
        region_cols=region_cols,
        review_mode=review_mode,
    )
    return editor.run()


def _run_napari_editor(image: np.ndarray, init_polygons_xy: List[np.ndarray], image_name: str) -> Tuple[str, List[np.ndarray]]:
    try:
        napari = importlib.import_module("napari")
    except ImportError as exc:
        raise ImportError(
            "napari is required for the rich polygon editor. Install with: pip install napari pyqt5"
        ) from exc

    state = {"action": "skip"}

    viewer = napari.Viewer(title=f"cryoEV HITL - {image_name}")
    viewer.add_image(_prepare_display_image(image), name="image", colormap="gray")
    shapes = viewer.add_shapes(
        _xy_polygons_to_napari(init_polygons_xy),
        shape_type="polygon",
        edge_color="lime",
        face_color=[0, 1, 0, 0.12],
        edge_width=1.5,
        name="EV polygons",
    )

    print("\nNapari controls:")
    print("  - Edit vertices directly in Shapes layer")
    print("  - Add polygon: Shapes layer toolbar -> Add polygon")
    print("  - Delete polygon: select shape then Backspace/Delete")
    print("  - Press S to save and continue")
    print("  - Press K to skip image")
    print("  - Press Q to quit session\n")

    @viewer.bind_key("s")
    def _save_and_next(_viewer):
        state["action"] = "save"
        _viewer.close()

    @viewer.bind_key("k")
    def _skip(_viewer):
        state["action"] = "skip"
        _viewer.close()

    @viewer.bind_key("q")
    def _quit(_viewer):
        state["action"] = "quit"
        _viewer.close()

    napari.run()

    polygons_xy = _safe_shapes_data_to_polygons(shapes.data)
    return state["action"], polygons_xy


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def _evaluate_predictions(auto_masks: List[np.ndarray], reviewed_masks: List[np.ndarray], iou_threshold: float = 0.5) -> Dict[str, float]:
    n_pred = len(auto_masks)
    n_gt = len(reviewed_masks)

    if n_pred == 0 and n_gt == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "mean_matched_iou": 1.0,
            "pred_count": 0,
            "reviewed_count": 0,
            "count_error": 0,
        }

    if n_pred == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": n_gt,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_matched_iou": 0.0,
            "pred_count": 0,
            "reviewed_count": n_gt,
            "count_error": -n_gt,
        }

    if n_gt == 0:
        return {
            "tp": 0,
            "fp": n_pred,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_matched_iou": 0.0,
            "pred_count": n_pred,
            "reviewed_count": 0,
            "count_error": n_pred,
        }

    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i, pred_mask in enumerate(auto_masks):
        for j, gt_mask in enumerate(reviewed_masks):
            iou_matrix[i, j] = _iou(pred_mask, gt_mask)

    pred_idx, gt_idx = linear_sum_assignment(-iou_matrix)

    matched_ious = []
    for i, j in zip(pred_idx, gt_idx):
        if iou_matrix[i, j] >= iou_threshold:
            matched_ious.append(float(iou_matrix[i, j]))

    tp = len(matched_ious)
    fp = n_pred - tp
    fn = n_gt - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "pred_count": n_pred,
        "reviewed_count": n_gt,
        "count_error": n_pred - n_gt,
    }


def _save_csv(records: List[Dict], output_path: Path) -> None:
    if not records:
        return

    fieldnames: List[str] = []
    seen = set()
    for record in records:
        for key in record.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _image_key(image_path: Path) -> str:
    try:
        return image_path.resolve().as_posix().lower()
    except OSError:
        return image_path.absolute().as_posix().lower()


def _image_fingerprint(image_path: Path) -> str:
    hasher = hashlib.sha1()
    try:
        with image_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return hashlib.sha1(_image_key(image_path).encode("utf-8")).hexdigest()


def _output_name_for(image_path: Path) -> str:
    safe_stem = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in image_path.stem
    ).strip("_")
    safe_stem = safe_stem or "image"
    return f"{safe_stem}__{_image_fingerprint(image_path)[:12]}"


def _load_progress_index(progress_csv: Path) -> Dict[str, Dict[str, str]]:
    progress_index: Dict[str, Dict[str, str]] = {}
    if not progress_csv.exists():
        return progress_index

    try:
        with progress_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get("image_key") or row.get("source_path") or "").strip().lower()
                if key:
                    progress_index[key] = row
    except Exception as exc:
        print(f"Warning: could not read progress log {progress_csv}: {exc}")

    return progress_index


def _append_progress_record(progress_csv: Path, record: Dict) -> None:
    fieldnames = [
        "timestamp",
        "run_name",
        "image",
        "image_name",
        "image_key",
        "image_fingerprint",
        "output_name",
        "source_path",
        "status",
        "action",
        "n_auto",
        "n_reviewed",
        "n_boxes",
    ]
    progress_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = progress_csv.exists()
    with progress_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({name: record.get(name, "") for name in fieldnames})


def _find_previously_annotated(image_paths: List[Path], output_dir: Path) -> List[Path]:
    progress_index = _load_progress_index(output_dir / "annotation_progress.csv")
    saved_rows = [
        row for row in progress_index.values()
        if (row.get("status") or "").strip().lower() == "saved"
    ]
    saved_keys = {
        (row.get("image_key") or "").strip().lower()
        for row in saved_rows
        if (row.get("image_key") or "").strip()
    }
    saved_fingerprints = {
        (row.get("image_fingerprint") or "").strip().lower()
        for row in saved_rows
        if (row.get("image_fingerprint") or "").strip()
    }

    completed: List[Path] = []
    for image_path in image_paths:
        image_key = _image_key(image_path)
        image_fingerprint = _image_fingerprint(image_path).lower()
        if image_key in saved_keys or image_fingerprint in saved_fingerprints:
            completed.append(image_path)
    return completed


def _prompt_for_resume_mode(already_annotated: List[Path], total_images: int, resume_mode: str) -> str:
    if not already_annotated:
        return "reannotate"
    if resume_mode != "ask":
        return resume_mode

    preview = "\n".join(f"• {path.name}" for path in already_annotated[:12])
    more = ""
    if len(already_annotated) > 12:
        more = f"\n… and {len(already_annotated) - 12} more."

    message = (
        f"Found {len(already_annotated)} previously annotated image(s) out of {total_images}.\n\n"
        "Yes = skip those completed images and continue where you left off.\n"
        "No = reannotate all images from the folder(s).\n"
        "Cancel = abort this run.\n\n"
        "Already annotated examples:\n"
        f"{preview}{more}"
    )

    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        choice = messagebox.askyesnocancel("Resume annotation session?", message)
        root.destroy()

        if choice is None:
            raise SystemExit("Annotation session cancelled by user.")
        return "skip" if choice else "reannotate"
    except Exception:
        print(message)
        while True:
            response = input("Enter [s]kip completed / [r]eannotate all / [c]ancel: ").strip().lower()
            if response in {"s", "skip"}:
                return "skip"
            if response in {"r", "reannotate", "redo"}:
                return "reannotate"
            if response in {"c", "cancel", "q", "quit"}:
                raise SystemExit("Annotation session cancelled by user.")
            print("Please enter s, r, or c.")


def run_annotation_session(
    input_images: List[Path] | Path,
    output_dir: Path,
    model_path: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    class_id: int,
    epsilon_ratio: float,
    pixel_size: float | None,
    save_auto_labels: bool,
    ui: str = "simple",
    region_rows: int = 2,
    region_cols: int = 4,
    review_mode: str = "polygon",
    resume_mode: str = "ask",
) -> None:
    if isinstance(input_images, Path):
        input_images = [input_images]
    image_paths = _collect_images_multi(input_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {input_images}")

    already_annotated = _find_previously_annotated(image_paths, output_dir)
    selected_resume_mode = _prompt_for_resume_mode(already_annotated, len(image_paths), resume_mode)
    if already_annotated and selected_resume_mode == "skip":
        completed_keys = {_image_key(path) for path in already_annotated}
        image_paths = [path for path in image_paths if _image_key(path) not in completed_keys]
        print(
            f"Skipping {len(already_annotated)} previously annotated image(s). "
            f"{len(image_paths)} image(s) remaining in this session."
        )
        if not image_paths:
            print("All selected images are already annotated. Nothing to do.")
            return
    elif already_annotated:
        print(
            f"Reannotating all {len(image_paths)} image(s), including "
            f"{len(already_annotated)} previously completed item(s)."
        )

    run_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    run_dir = output_dir / run_name
    progress_csv = output_dir / "annotation_progress.csv"

    reviewed_images_dir = run_dir / "reviewed" / "images"
    reviewed_labels_dir = run_dir / "reviewed" / "labels"
    reviewed_detect_labels_dir = run_dir / "reviewed_detect" / "labels"
    reviewed_detect_overlays_dir = run_dir / "reviewed_detect" / "overlays"
    auto_labels_dir = run_dir / "auto" / "labels"
    overlays_dir = run_dir / "reviewed" / "overlays"
    stats_dir = run_dir / "stats"

    _rescale_cache: dict = {}

    session_records: List[Dict] = []
    eval_records: List[Dict] = []
    morphology_records_all: List[Dict] = []

    perf = PerformanceLogger(output_dir=output_dir, session_id=run_name)
    perf.log_session_start(
        model_path=model_path,
        n_images=len(image_paths),
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )
    session_t0 = time.perf_counter()
    n_processed = 0

    print(f"Loaded {len(image_paths)} images for HITL annotation.")
    print(f"Using model: {model_path}\n")

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] {image_path.name}")

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("  Skipped: cannot read image")
            error_record = {
                "image": image_path.name,
                "source_path": str(image_path),
                "status": "read_error",
                "action": "skip",
                "n_auto": 0,
                "n_reviewed": 0,
            }
            session_records.append(error_record)
            _append_progress_record(
                progress_csv,
                {
                    **error_record,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "run_name": run_name,
                    "image_name": image_path.name,
                    "image_key": _image_key(image_path),
                    "image_fingerprint": _image_fingerprint(image_path),
                    "output_name": "",
                    "n_boxes": 0,
                },
            )
            continue

        image, inf_path, inf_scale = _rescale_image_for_inference(
            image, image_path, imgsz, _rescale_cache
        )
        _t_inf0 = time.perf_counter()
        pred_masks, confidences = load_predictions_from_model(
            model_path=str(model_path),
            img_path=str(inf_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
        )
        _inference_time_s = time.perf_counter() - _t_inf0
        # Clean up temp file created by rescaling.
        if inf_path != image_path and inf_path.exists():
            try:
                inf_path.unlink()
            except OSError:
                pass

        auto_polygons_xy: List[np.ndarray] = []
        auto_confidences: List[float] = []
        for mask, confidence in zip(pred_masks, confidences):
            polygons = mask_to_polygon(mask, epsilon_ratio=epsilon_ratio)
            for polygon in polygons:
                auto_polygons_xy.append(polygon)
                auto_confidences.append(float(confidence))

        if save_auto_labels:
            write_yolo_polygon_labels(
                auto_labels_dir / f"{image_path.stem}.txt",
                polygons=auto_polygons_xy,
                width=image.shape[1],
                height=image.shape[0],
                class_id=class_id,
            )

        if ui == "napari":
            action, reviewed_polygons_xy = _run_napari_editor(
                image=image,
                init_polygons_xy=auto_polygons_xy,
                image_name=image_path.name,
            )
        else:
            action, reviewed_polygons_xy = _run_simple_editor(
                image=image,
                init_polygons_xy=auto_polygons_xy,
                image_name=image_path.name,
                init_confidences=auto_confidences,
                class_id=class_id,
                region_rows=region_rows,
                region_cols=region_cols,
                review_mode=review_mode,
            )

        perf.log_image(
            image_path=image_path,
            n_raw_detections=len(pred_masks),
            inference_time_s=_inference_time_s,
            image=image,
        )
        n_processed += 1

        if action == "quit":
            print("  Session ended by user.")
            break

        if action == "skip":
            print("  Skipped without saving.")
            skipped_record = {
                "image": image_path.name,
                "source_path": str(image_path),
                "status": "skipped",
                "action": action,
                "n_auto": len(auto_polygons_xy),
                "n_reviewed": len(reviewed_polygons_xy),
                "n_boxes": len(reviewed_polygons_xy),
            }
            session_records.append(skipped_record)
            _append_progress_record(
                progress_csv,
                {
                    **skipped_record,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "run_name": run_name,
                    "image_name": image_path.name,
                    "image_key": _image_key(image_path),
                    "image_fingerprint": _image_fingerprint(image_path),
                    "output_name": _output_name_for(image_path),
                },
            )
            continue

        output_name = _output_name_for(image_path)
        reviewed_label_path = reviewed_labels_dir / f"{output_name}.txt"
        reviewed_detect_label_path = reviewed_detect_labels_dir / f"{output_name}.txt"
        reviewed_image_path = reviewed_images_dir / f"{output_name}{image_path.suffix}"

        refined_polygons_xy: List[np.ndarray] = []
        reviewed_boxes_xy: List[np.ndarray] = []
        for polygon in reviewed_polygons_xy:
            box = _polygon_to_box(polygon)
            reviewed_boxes_xy.append(box)
            if review_mode == "box":
                refined_polygons_xy.append(
                    refine_polygon_from_box(
                        image=image,
                        box=box,
                        initial_polygon=polygon,
                        epsilon_ratio=max(epsilon_ratio, 0.01),
                    )
                )
            else:
                refined_polygons_xy.append(np.asarray(polygon, dtype=np.float32))

        copy_image_to_dataset(image_path, reviewed_image_path)
        write_yolo_polygon_labels(
            reviewed_label_path,
            polygons=refined_polygons_xy,
            width=image.shape[1],
            height=image.shape[0],
            class_id=class_id,
        )
        write_yolo_box_labels(
            reviewed_detect_label_path,
            boxes=reviewed_boxes_xy,
            width=image.shape[1],
            height=image.shape[0],
            class_id=class_id,
        )

        auto_masks = polygons_to_masks(auto_polygons_xy, image.shape[:2])
        reviewed_masks = polygons_to_masks(refined_polygons_xy, image.shape[:2])

        eval_metrics = _evaluate_predictions(auto_masks, reviewed_masks)
        eval_metrics["image"] = image_path.name
        eval_records.append(eval_metrics)

        morph_records = analyze_instances(reviewed_masks, pixel_size=pixel_size)
        for rec in morph_records:
            rec["image"] = image_path.name
        morphology_records_all.extend(morph_records)

        # Model-detection overlay (auto predictions only, yellow).
        overlay_auto = _draw_polygons(image, auto_polygons_xy, color=(255, 255, 0), thickness=1)
        reviewed_detect_overlays_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(reviewed_detect_overlays_dir / f"{output_name}_overlay.png"), overlay_auto)

        # Reviewed overlay: auto (yellow) + user-corrected (white) on top.
        overlay_reviewed = _draw_polygons(overlay_auto, refined_polygons_xy, color=(255, 255, 255), thickness=2)
        overlays_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlays_dir / f"{output_name}_overlay.png"), overlay_reviewed)

        # Ellipse overlay — fitted ellipses drawn on top of the reviewed polygon overlay
        ellipse_vis = draw_ellipses_on_image(
            image, reviewed_masks, morph_records, color=(0, 255, 255), thickness=2
        )
        cv2.imwrite(
            str(overlays_dir / f"{output_name}_ellipses.png"),
            cv2.cvtColor(ellipse_vis, cv2.COLOR_RGB2BGR)
        )

        saved_record = {
            "image": image_path.name,
            "image_id": _image_fingerprint(image_path)[:12],
            "output_name": output_name,
            "source_path": str(image_path),
            "status": "saved",
            "action": action,
            "n_auto": len(auto_polygons_xy),
            "n_reviewed": len(refined_polygons_xy),
            "n_boxes": len(reviewed_boxes_xy),
        }
        session_records.append(saved_record)
        _append_progress_record(
            progress_csv,
            {
                **saved_record,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "run_name": run_name,
                "image_name": image_path.name,
                "image_key": _image_key(image_path),
                "image_fingerprint": _image_fingerprint(image_path),
            },
        )

        save_label = "Saved hand-corrected polygons + detect boxes" if review_mode == "polygon" else "Saved refined masks + detect boxes"
        print(
            f"  {save_label} | "
            f"auto={len(auto_polygons_xy)} reviewed={len(refined_polygons_xy)} boxes={len(reviewed_boxes_xy)} "
            f"F1={eval_metrics['f1']:.3f} meanIoU={eval_metrics['mean_matched_iou']:.3f}"
        )

    perf.log_session_end(
        total_time_s=time.perf_counter() - session_t0,
        n_processed=n_processed,
    )

    _save_csv(session_records, stats_dir / "annotation_session.csv")
    _save_csv(eval_records, stats_dir / "model_vs_review_metrics.csv")

    if morphology_records_all:
        save_morphology_csv(morphology_records_all, str(stats_dir / "morphology_reviewed_all.csv"))
        plot_morphology_distributions(
            morphology_records_all,
            pixel_size=pixel_size,
            save_path=str(stats_dir / "morphology_distributions.png"),
        )

    print("\nAnnotation session complete.")
    print(f"Run folder: {run_dir}")
    print(f"Session log: {stats_dir / 'annotation_session.csv'}")
    print(f"Metrics: {stats_dir / 'model_vs_review_metrics.csv'}")
    print(f"Perf log:  {output_dir / 'performance_log.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rich human-in-the-loop polygon annotation for cryoEV")
    parser.add_argument("--input-images", type=Path, nargs="+", help="One or more image files or folders to annotate (combined into a single session)")
    parser.add_argument("--output-dir", type=Path, help="Output folder for reviewed dataset and stats")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to YOLO weights file or folder containing best.pt")
    parser.add_argument("--gui", action="store_true", help="Open folder/file picker dialogs for input, output, and model paths")
    parser.add_argument("--ui", choices=["simple", "napari"], default="simple", help="Annotation interface to use (default: simple side-by-side reviewer)")
    parser.add_argument("--review-mode", choices=["polygon", "box"], default="polygon", help="Review geometry to edit (default: polygon for true hand-corrected masks)")
    parser.add_argument("--region-rows", type=int, default=2, help="Number of review rows for pane-based navigation (default: 2)")
    parser.add_argument("--region-cols", type=int, default=4, help="Number of review columns for pane-based navigation (default: 4)")
    parser.add_argument("--imgsz", type=int, default=1024, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device: cpu or cuda")
    parser.add_argument("--class-id", type=int, default=0, help="Class id written in reviewed YOLO labels")
    parser.add_argument("--epsilon-ratio", type=float, default=0.005, help="Polygon simplification ratio for predicted masks")
    parser.add_argument("--pixel-size", type=float, default=None, help="Physical size per pixel (e.g., nm/px)")
    parser.add_argument("--save-auto-labels", action="store_true", help="Also save raw model predictions as YOLO labels")
    parser.add_argument(
        "--resume-mode",
        choices=["ask", "skip", "reannotate"],
        default="ask",
        help="If prior annotations exist in the output folder: ask, skip completed images, or reannotate all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_images_list, output_dir, model_path = _resolve_runtime_paths(args)
    run_annotation_session(
        input_images=input_images_list,
        output_dir=output_dir,
        model_path=model_path,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        class_id=args.class_id,
        epsilon_ratio=args.epsilon_ratio,
        pixel_size=args.pixel_size,
        save_auto_labels=args.save_auto_labels,
        ui=args.ui,
        region_rows=args.region_rows,
        region_cols=args.region_cols,
        review_mode=args.review_mode,
        resume_mode=args.resume_mode,
    )


if __name__ == "__main__":
    main()
