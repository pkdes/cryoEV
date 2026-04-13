import shutil
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np


def polygons_to_yolo_lines(polygons: Iterable[np.ndarray], width: int, height: int, class_id: int = 0) -> List[str]:
    """Convert polygons in pixel coordinates to YOLO polygon label lines."""
    lines: List[str] = []
    for polygon in polygons:
        poly = np.asarray(polygon, dtype=np.float32)
        if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
            continue

        coords: List[str] = []
        for x, y in poly:
            x_norm = float(np.clip(x / max(width, 1), 0.0, 1.0))
            y_norm = float(np.clip(y / max(height, 1), 0.0, 1.0))
            coords.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])

        lines.append(" ".join([str(class_id), *coords]))

    return lines


def write_yolo_polygon_labels(label_path: Path, polygons: Iterable[np.ndarray], width: int, height: int, class_id: int = 0) -> None:
    """Write polygons to YOLO polygon format label file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = polygons_to_yolo_lines(polygons, width=width, height=height, class_id=class_id)
    content = "\n".join(lines)
    if content:
        content += "\n"
    label_path.write_text(content, encoding="utf-8")


def boxes_to_yolo_lines(boxes: Iterable[np.ndarray], width: int, height: int, class_id: int = 0) -> List[str]:
    """Convert boxes in pixel coordinates to YOLO detection label lines."""
    lines: List[str] = []
    for box in boxes:
        x0, y0, x1, y1 = [float(v) for v in np.asarray(box, dtype=np.float32).reshape(4)]
        bw = max(x1 - x0, 1.0)
        bh = max(y1 - y0, 1.0)
        cx = x0 + bw / 2.0
        cy = y0 + bh / 2.0
        line = " ".join(
            [
                str(class_id),
                f"{np.clip(cx / max(width, 1), 0.0, 1.0):.6f}",
                f"{np.clip(cy / max(height, 1), 0.0, 1.0):.6f}",
                f"{np.clip(bw / max(width, 1), 0.0, 1.0):.6f}",
                f"{np.clip(bh / max(height, 1), 0.0, 1.0):.6f}",
            ]
        )
        lines.append(line)
    return lines


def write_yolo_box_labels(label_path: Path, boxes: Iterable[np.ndarray], width: int, height: int, class_id: int = 0) -> None:
    """Write boxes to YOLO detection format label file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = boxes_to_yolo_lines(boxes, width=width, height=height, class_id=class_id)
    content = "\n".join(lines)
    if content:
        content += "\n"
    label_path.write_text(content, encoding="utf-8")


def read_yolo_polygon_labels(label_path: Path, width: int, height: int) -> List[np.ndarray]:
    """Read YOLO polygon label file and return polygons in pixel coordinates."""
    if not label_path.exists():
        return []

    polygons: List[np.ndarray] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) < 7:
            continue
        coords = tokens[1:]
        if len(coords) % 2 != 0:
            continue

        points = []
        for i in range(0, len(coords), 2):
            try:
                x = float(coords[i]) * width
                y = float(coords[i + 1]) * height
            except ValueError:
                points = []
                break
            points.append([x, y])

        if len(points) >= 3:
            polygons.append(np.asarray(points, dtype=np.float32))

    return polygons


def mask_to_polygon(mask: np.ndarray, epsilon_ratio: float = 0.005) -> List[np.ndarray]:
    """Extract one or more polygons from a binary mask."""
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    polygons: List[np.ndarray] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = epsilon_ratio * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        pts = approx.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] >= 3:
            polygons.append(pts)
    return polygons


def polygons_to_masks(polygons: Iterable[np.ndarray], image_shape: tuple[int, int]) -> List[np.ndarray]:
    """Rasterize polygons into a list of boolean masks."""
    h, w = image_shape
    masks: List[np.ndarray] = []
    for polygon in polygons:
        poly = np.asarray(polygon, dtype=np.int32)
        if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
            continue
        canvas = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(canvas, [poly.reshape(-1, 1, 2)], color=1)
        masks.append(canvas.astype(bool))
    return masks


def refine_polygon_from_box(
    image: np.ndarray,
    box: np.ndarray,
    initial_polygon: np.ndarray | None = None,
    epsilon_ratio: float = 0.01,
) -> np.ndarray:
    """Refine a reviewed box into a tighter polygon using local threshold/edge cues.

    Falls back to the provided polygon (or the box corners) if no stable contour is found.
    """
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            arr = arr[..., 0]

    x0, y0, x1, y1 = [float(v) for v in np.asarray(box, dtype=np.float32).reshape(4)]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))

    h, w = arr.shape[:2]
    pad = 3
    xi0 = max(int(np.floor(x0)) - pad, 0)
    yi0 = max(int(np.floor(y0)) - pad, 0)
    xi1 = min(int(np.ceil(x1)) + pad, w - 1)
    yi1 = min(int(np.ceil(y1)) + pad, h - 1)

    roi = arr[yi0:yi1 + 1, xi0:xi1 + 1]
    if roi.size == 0:
        if initial_polygon is not None:
            return np.asarray(initial_polygon, dtype=np.float32)
        return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)

    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    roi_norm = cv2.normalize(roi_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, dark_mask = cv2.threshold(roi_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, bright_mask = cv2.threshold(roi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge_mask = cv2.Canny(roi_norm, 30, 120)

    kernel = np.ones((3, 3), dtype=np.uint8)
    candidates = []
    for candidate in (dark_mask, bright_mask, edge_mask):
        closed = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=1)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        candidates.append(opened)

    init_mask = None
    if initial_polygon is not None:
        poly = np.asarray(initial_polygon, dtype=np.float32).reshape(-1, 2)
        if poly.shape[0] >= 3:
            init_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            shifted = poly.copy()
            shifted[:, 0] -= xi0
            shifted[:, 1] -= yi0
            cv2.fillPoly(init_mask, [shifted.astype(np.int32).reshape(-1, 1, 2)], 1)

    roi_area = float(roi.shape[0] * roi.shape[1])
    box_center = np.asarray([(x0 + x1) / 2.0 - xi0, (y0 + y1) / 2.0 - yi0], dtype=np.float32)
    max_center_dist = max(roi.shape[:2]) + 1e-6

    best_score = -1e9
    best_polygon = None

    for candidate in candidates:
        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max(9.0, 0.01 * roi_area):
                continue
            if area > 0.98 * roi_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True).reshape(-1, 2).astype(np.float32)
            if approx.shape[0] < 3:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
            else:
                cx, cy = np.mean(approx[:, 0]), np.mean(approx[:, 1])

            center_dist = np.linalg.norm(np.asarray([cx, cy], dtype=np.float32) - box_center)
            center_score = 1.0 - min(center_dist / max_center_dist, 1.0)
            area_score = area / roi_area

            overlap_score = 0.0
            if init_mask is not None:
                contour_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.fillPoly(contour_mask, [approx.astype(np.int32).reshape(-1, 1, 2)], 1)
                inter = np.logical_and(contour_mask > 0, init_mask > 0).sum()
                union = np.logical_or(contour_mask > 0, init_mask > 0).sum()
                overlap_score = float(inter / union) if union else 0.0

            score = (2.0 * overlap_score) + (1.0 * area_score) + (0.75 * center_score)
            if score > best_score:
                best_score = score
                best_polygon = approx.copy()

    if best_polygon is None:
        if initial_polygon is not None:
            return np.asarray(initial_polygon, dtype=np.float32)
        return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)

    best_polygon[:, 0] += xi0
    best_polygon[:, 1] += yi0
    return best_polygon.astype(np.float32)


def copy_image_to_dataset(src_image_path: Path, dst_image_path: Path) -> None:
    """Copy source image into reviewed dataset image folder."""
    dst_image_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_image_path, dst_image_path)
