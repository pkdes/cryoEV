"""Windows long-path safe access to prediction files (TODO #54).

Windows without LongPathsEnabled fails `Path.exists()` / `open()` on a path over 260 characters, and
`train_yolo.load_predictions_yolo_format()` reads a path it can't see as "no detections". A path over
LONG_PATH characters is given the `\\\\?\\` prefix (`\\\\?\\UNC\\` for a share), which lifts that limit;
shorter paths and other systems are returned unchanged, so outputs on normal paths stay byte-identical.
"""
from __future__ import annotations

import os
from pathlib import Path

LONG_PATH = 240           # below MAX_PATH (260) with room for a file name appended later


def long_path(path) -> Path:
    """`path`, absolute and `\\\\?\\`-prefixed when it is over LONG_PATH characters on Windows; else as given."""
    path = Path(path)
    if os.name != "nt":
        return path
    s = os.path.abspath(str(path))                 # normalizes / and .. (the \\?\ form takes the path literally)
    if len(s) <= LONG_PATH or s.startswith("\\\\?\\"):
        return path
    return Path("\\\\?\\UNC\\" + s[2:] if s.startswith("\\\\") else "\\\\?\\" + s)


def prediction_file(predictions_dir, stem, image=None) -> Path:
    """<predictions_dir>/<stem>.txt, long-path safe. predict_models.py / run_experiments.py write one file per image
    (empty = no detections), so a missing file is an error, not zero detections: FileNotFoundError naming the
    image and the path length."""
    txt = Path(predictions_dir) / f"{stem}.txt"
    p = long_path(txt)
    if not p.is_file():
        n = len(os.path.abspath(str(txt)))
        raise FileNotFoundError(
            f"no predictions file for image {image or stem}: {txt} ({n} characters"
            + (", over the Windows 260-character limit" if n >= 260 else "") + "). The predictions folder has "
            "one file per image (empty = no detections); rerun inference/predict_models.py for it, or check "
            "--predictions-dir / --images-dir")
    return p
