#!/usr/bin/env python3
"""Harvest cryo-EM images from nested "raw image(s)" folders into one traceable dataset.

This is useful when images live under many condition-specific subfolders but need to be
collected into a single folder for annotation or model training.

Example
-------
python -m data_utils.harvest_raw_images \
    --source-root "C:\\path\\to\\DATA FOR THE PAPER" \
    --dest-root "C:\\path\\to\\raw images\\Sima_paper_harvest_20260408"

What it creates
---------------
<dest-root>/
  images/                 # flat folder for annotation/training inputs
  source_manifest.csv     # one row per copied image with full original path + hash
  raw_folder_summary.csv  # counts by source raw-image folder
  README.txt              # provenance notes and rerun command
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
RAW_IMAGE_NAME_TOKENS = ("raw image", "raw images")


@dataclass
class HarvestRecord:
    source_path: str
    source_rel_path: str
    raw_folder_rel_path: str
    condition_rel_path: str
    original_filename: str
    harvested_filename: str
    harvested_path: str
    sha256: str
    size_bytes: int
    status: str


def sanitize_component(text: str) -> str:
    """Turn a path component into a filesystem-safe, readable token."""
    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "_", text.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "item"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_raw_image_dir_name(name: str) -> bool:
    """Return True for folders whose names include a raw-image marker."""
    normalized = re.sub(r"\s+", " ", name.strip().lower())
    return any(token in normalized for token in RAW_IMAGE_NAME_TOKENS)



def iter_raw_image_dirs(source_root: Path) -> Iterable[Path]:
    for current_root, dirnames, _ in os.walk(source_root):
        current_path = Path(current_root)
        if is_raw_image_dir_name(current_path.name):
            yield current_path
            dirnames[:] = []  # no need to recurse deeper inside the matched raw-image folder


def build_traceable_name(source_file: Path, source_root: Path) -> str:
    rel = source_file.relative_to(source_root)
    context_parts = [sanitize_component(part) for part in rel.parts[:-1]]
    prefix = "__".join(context_parts) if context_parts else "root"
    stem = sanitize_component(source_file.stem)
    return f"{prefix}__{stem}{source_file.suffix.lower()}"


def unique_destination_path(destination_dir: Path, filename: str, source_hash: str) -> tuple[Path, str]:
    """Return a collision-safe alternative path for a filename that is already taken."""
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    short_hash = source_hash[:8]
    dedup_name = f"{stem}__{short_hash}{suffix}"
    dedup_path = destination_dir / dedup_name
    if not dedup_path.exists():
        return dedup_path, dedup_name

    counter = 2
    while True:
        numbered = f"{stem}__{short_hash}_{counter}{suffix}"
        numbered_path = destination_dir / numbered
        if not numbered_path.exists():
            return numbered_path, numbered
        counter += 1


def collect_images(raw_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def harvest_images(source_root: Path, dest_root: Path, dry_run: bool = False) -> tuple[list[HarvestRecord], Counter]:
    source_root = source_root.resolve()
    dest_root = dest_root.resolve()
    images_dir = dest_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    raw_dirs = sorted(iter_raw_image_dirs(source_root))
    if not raw_dirs:
        raise FileNotFoundError(f"No folders containing 'raw image' were found under: {source_root}")

    records: list[HarvestRecord] = []
    summary: Counter = Counter()

    for raw_dir in raw_dirs:
        raw_rel = raw_dir.relative_to(source_root)
        raw_folder_key = raw_rel.as_posix()
        files = collect_images(raw_dir)
        summary[raw_folder_key] += len(files)

        for source_file in files:
            source_hash = sha256_file(source_file)
            suggested_name = build_traceable_name(source_file, source_root)
            suggested_path = images_dir / suggested_name

            if suggested_path.exists():
                if sha256_file(suggested_path) == source_hash:
                    dst_path, dst_name = suggested_path, suggested_name
                    status = "already-present"
                else:
                    dst_path, dst_name = unique_destination_path(images_dir, suggested_name, source_hash)
                    status = "name-collision"
            else:
                dst_path, dst_name = suggested_path, suggested_name
                status = "copied"

            if not dry_run and status in {"copied", "name-collision"}:
                shutil.copy2(source_file, dst_path)

            condition_parts = list(raw_rel.parts[:-1])
            condition_rel = Path(*condition_parts).as_posix() if condition_parts else "."

            records.append(
                HarvestRecord(
                    source_path=str(source_file),
                    source_rel_path=source_file.relative_to(source_root).as_posix(),
                    raw_folder_rel_path=raw_folder_key,
                    condition_rel_path=condition_rel,
                    original_filename=source_file.name,
                    harvested_filename=dst_name,
                    harvested_path=str(dst_path),
                    sha256=source_hash,
                    size_bytes=source_file.stat().st_size,
                    status="would-copy" if dry_run else status,
                )
            )

    return records, summary


def write_manifest(dest_root: Path, records: list[HarvestRecord], summary: Counter, args: argparse.Namespace) -> None:
    manifest_path = dest_root / "source_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else list(HarvestRecord.__annotations__.keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    summary_path = dest_root / "raw_folder_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["raw_folder_rel_path", "image_count"])
        writer.writeheader()
        for raw_folder, count in sorted(summary.items()):
            writer.writerow({"raw_folder_rel_path": raw_folder, "image_count": count})

    readme_path = dest_root / "README.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    readme_text = f"""Traceable cryo-EM image harvest
Generated: {timestamp}

Source root:
{args.source_root}

Destination root:
{args.dest_root}

Copy mode:
- Images are flattened into ./images for easy annotation input.
- Filenames are renamed to include their source folder context.
- Full provenance is recorded in source_manifest.csv.
- Counts by original raw-image folder are recorded in raw_folder_summary.csv.

Command used:
python -m data_utils.harvest_raw_images --source-root \"{args.source_root}\" --dest-root \"{args.dest_root}\"{' --dry-run' if args.dry_run else ''}
"""
    readme_path.write_text(readme_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="Root directory to scan recursively for raw-image folders")
    parser.add_argument("--dest-root", required=True, type=Path, help="Destination directory for harvested images and provenance files")
    parser.add_argument("--dry-run", action="store_true", help="Scan and generate manifests without copying files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    dest_root = Path(args.dest_root)

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    dest_root.mkdir(parents=True, exist_ok=True)
    records, summary = harvest_images(source_root=source_root, dest_root=dest_root, dry_run=args.dry_run)
    write_manifest(dest_root=dest_root, records=records, summary=summary, args=args)

    copied = sum(1 for r in records if r.status == "copied")
    already_present = sum(1 for r in records if r.status == "already-present")
    print(f"Found {len(summary)} raw-image folder(s) and {len(records)} image(s).")
    print(f"Copied: {copied}")
    if already_present:
        print(f"Already present: {already_present}")
    print(f"Manifest written to: {dest_root / 'source_manifest.csv'}")

    if summary:
        print("\nPer-folder counts:")
        for raw_folder, count in sorted(summary.items()):
            print(f"  - {raw_folder}: {count}")


if __name__ == "__main__":
    main()
