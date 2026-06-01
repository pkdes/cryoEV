"""
Quick Yes/No Image Triage
=========================
Fast keyboard-only reviewer to classify images as keep/reject.

Typical workflow:
1) Review images and save decisions to CSV.
2) Apply rejects to move bad images into a `rejects` subfolder.

Your cross-grid comparison pipeline already skips any image path under a
folder named `rejects`, so this is a quick pre-filter step.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class Decision:
    image_path: str
    decision: str
    timestamp: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick yes/no triage for cryoEV images.")
    parser.add_argument("--input-dir", required=True, help="Folder containing images.")
    parser.add_argument(
        "--decisions-csv",
        default=None,
        help="Decision CSV path (default: <input-dir>/image_triage_decisions.csv).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input-dir recursively.",
    )
    parser.add_argument(
        "--review-all",
        action="store_true",
        help="Review all images, even if already in decisions CSV.",
    )
    parser.add_argument(
        "--apply-rejects",
        action="store_true",
        help="Move rejected images into <input-dir>/rejects using decisions CSV.",
    )
    parser.add_argument(
        "--rejects-dir",
        default=None,
        help="Destination folder for rejected images (default: <input-dir>/rejects).",
    )
    return parser.parse_args()


def _is_reject_path(path: Path) -> bool:
    return any(part.lower() == "rejects" for part in path.parts)


def collect_images(input_dir: Path, recursive: bool) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    images: List[Path] = []
    for p in sorted(iterator):
        if not p.is_file():
            continue
        if _is_reject_path(p):
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        images.append(p)
    return images


def load_decisions(csv_path: Path) -> Dict[str, Decision]:
    if not csv_path.exists():
        return {}

    decisions: Dict[str, Decision] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = str(row.get("image_path", "")).strip()
            decision = str(row.get("decision", "")).strip().lower()
            timestamp = str(row.get("timestamp", "")).strip()
            if not image_path or decision not in {"keep", "reject"}:
                continue
            decisions[image_path] = Decision(
                image_path=image_path,
                decision=decision,
                timestamp=timestamp,
            )
    return decisions


def save_decisions(csv_path: Path, decisions: Dict[str, Decision]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(decisions.values(), key=lambda d: d.image_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "decision", "timestamp"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_path": row.image_path,
                    "decision": row.decision,
                    "timestamp": row.timestamp,
                }
            )


def _prepare_display_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    gray = arr.astype(np.float32)
    lo = float(np.percentile(gray, 1))
    hi = float(np.percentile(gray, 99))
    if hi <= lo:
        lo = float(np.min(gray))
        hi = float(np.max(gray))
        if hi <= lo:
            hi = lo + 1.0
    norm = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255).astype(np.uint8)


def _render_frame(image: np.ndarray, header_lines: List[str]) -> np.ndarray:
    display = _prepare_display_u8(image)
    canvas = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    y = 26
    for line in header_lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return canvas


def review_images(images: List[Path], csv_path: Path, review_all: bool) -> None:
    decisions = load_decisions(csv_path)
    queue = images if review_all else [p for p in images if str(p) not in decisions]

    if not queue:
        print("No images to review.")
        return

    print(f"Reviewing {len(queue)} image(s).")
    print("Keys: Y=keep, N=reject, B=back, Q=quit")

    idx = 0
    history: List[str] = []
    window_name = "Quick Image Triage"

    while 0 <= idx < len(queue):
        img_path = queue[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[warn] unreadable image: {img_path}")
            idx += 1
            continue

        previous = decisions.get(str(img_path))
        previous_text = previous.decision if previous else "undecided"
        frame = _render_frame(
            img,
            [
                f"{idx + 1}/{len(queue)}   {img_path.name}",
                f"Current: {previous_text}",
                "Y keep | N reject | B back | Q quit",
            ],
        )
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord("b"):
            if history:
                prev_path = history.pop()
                idx = max(queue.index(Path(prev_path)) if Path(prev_path) in queue else idx - 1, 0)
            else:
                idx = max(idx - 1, 0)
            continue

        if key in (ord("y"), ord("n")):
            decision = "keep" if key == ord("y") else "reject"
            decisions[str(img_path)] = Decision(
                image_path=str(img_path),
                decision=decision,
                timestamp=datetime.now().isoformat(timespec="seconds"),
            )
            history.append(str(img_path))
            save_decisions(csv_path, decisions)
            idx += 1
            continue

    cv2.destroyAllWindows()
    keep_count = sum(1 for d in decisions.values() if d.decision == "keep")
    reject_count = sum(1 for d in decisions.values() if d.decision == "reject")
    print(f"Saved decisions: {csv_path}")
    print(f"Totals -> keep: {keep_count}, reject: {reject_count}, all-decided: {len(decisions)}")


def apply_rejects(input_dir: Path, csv_path: Path, rejects_dir: Path) -> None:
    decisions = load_decisions(csv_path)
    rejects = [d for d in decisions.values() if d.decision == "reject"]
    if not rejects:
        print("No rejected images found in decisions CSV.")
        return

    moved = 0
    missing = 0
    rejects_dir.mkdir(parents=True, exist_ok=True)

    for d in rejects:
        src = Path(d.image_path)
        if not src.exists():
            missing += 1
            continue

        try:
            rel = src.relative_to(input_dir)
            dst = rejects_dir / rel
        except ValueError:
            dst = rejects_dir / src.name

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            i = 1
            while True:
                candidate = dst.with_name(f"{stem}_{i}{suffix}")
                if not candidate.exists():
                    dst = candidate
                    break
                i += 1

        shutil.move(str(src), str(dst))
        moved += 1

    print(f"Applied rejects to: {rejects_dir}")
    print(f"Moved: {moved}, Missing: {missing}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input folder not found: {input_dir}")

    decisions_csv = Path(args.decisions_csv) if args.decisions_csv else input_dir / "image_triage_decisions.csv"
    rejects_dir = Path(args.rejects_dir) if args.rejects_dir else input_dir / "rejects"

    if args.apply_rejects:
        apply_rejects(input_dir=input_dir, csv_path=decisions_csv, rejects_dir=rejects_dir)
        return

    images = collect_images(input_dir=input_dir, recursive=args.recursive)
    if not images:
        raise SystemExit("No images found for review.")
    review_images(images=images, csv_path=decisions_csv, review_all=args.review_all)


if __name__ == "__main__":
    main()
