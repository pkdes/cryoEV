"""Shared paths, filename parsing, and progress reporting for the embedding workflow.

This is a separate, INFERENCE-ONLY workflow from the YOLO training pipeline: it tests
whether frozen self-supervised embeddings cluster EV objects without using labels.
Labels are held out and used only for post-hoc evaluation.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

# Same convention as data_utils/prepare_all_layer_singleclass.py
ROOT = Path(__file__).parent.parent.parent / "CryoAI"
OUT = ROOT / "embedding outputs"
QC = OUT / "qc"
CROPS = OUT / "crops"

DEFAULT_SOURCE_NAME = "roboflow - 20260824 cryoai v3"
# Roboflow split dir -> canonical split name
SPLITS = {"train": "train", "valid": "valid", "test": "test"}

CROP_SIZE = 224
CONTEXT_FACTOR = 2.0      # window side = CONTEXT_FACTOR * bbox long side
MIN_WINDOW_PX = 48        # floor, so tiny objects aren't pure interpolation
REFERENCE_WIDTH = 1440    # the majority pixel scale; 5760-wide images get downsampled 4x


# --------------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------------
# Roboflow mangles names as: <original>_{png,jpg}.rf.<32-hex>.jpg
_RF_SUFFIX = re.compile(r"_(?:png|jpg|jpeg|tif|tiff)\.rf\.[0-9a-f]+\.(?:jpg|png)$", re.I)

# session -> cell line. Sessions whose prefix Roboflow trimmed can't be resolved.
CELL_LINE = {
    "20260126-MDA-MB-KH": "MDA-MB-231",
    "20260420-MDA-MB-RM": "MDA-MB-231",
    "20260506-HEK293-holey-carbon": "HEK293",
    "20260512-A549-EV": "A549",
    "20260512-PMSC-EV": "PMSC",
    "Jul30": "unknown",
}

_SQUARE = re.compile(r"_square(\d+)")
_HOLE = re.compile(r"_hole(\d+)")
_JUL30_GRID = re.compile(r"^Grid(\d+)_", re.I)


def strip_roboflow(file_name: str) -> str:
    """Recover the original filename stem from a Roboflow-exported name."""
    return _RF_SUFFIX.sub("", file_name)


def parse_name(file_name: str) -> dict:
    """Parse session / grid / square / hole out of a micrograph filename.

    Returns a dict with keys: stem, session, grid, square, hole, cell_line, parsed.
    `parsed` is False when the session prefix is absent (Roboflow trimmed it on some
    files, e.g. '1_square118_hole13_0_hm'). Those are bucketed as 'unknown' rather
    than silently folded into a real session -- crops.py reports the count.
    """
    stem = strip_roboflow(file_name)
    square = _SQUARE.search(stem)
    hole = _HOLE.search(stem)

    parts = stem.split("__")
    head = parts[0]

    session, grid, parsed = "unknown", "unknown", False

    if "Jul30" in stem:
        # The 5760x4092 set: 'Grid3_0012_Jul30', 'Grid4_73_0001_Jul30', or bare '0004_Jul30'
        session, parsed = "Jul30", True
        m = _JUL30_GRID.match(stem)
        grid = f"Grid{m.group(1)}" if m else "Grid_unlabeled"
    elif head in CELL_LINE:
        session, parsed = head, True
        grid = parts[1] if len(parts) > 2 else (parts[1] if len(parts) > 1 else "grid_unlabeled")
        # '20260506-HEK293-holey-carbon__grid10__10_square312...' -> grid10
        grid = grid.split("_square")[0] or "grid_unlabeled"
    else:
        # Session prefix trimmed by Roboflow. Keep the leading token as a grid-ish
        # hint but do NOT invent a session.
        grid = head.split("_square")[0] or "unknown"

    return {
        "stem": stem,
        "session": session,
        "grid": grid,
        "square": int(square.group(1)) if square else -1,
        "hole": int(hole.group(1)) if hole else -1,
        "cell_line": CELL_LINE.get(session, "unknown"),
        "parsed": parsed,
    }


def session_grid(info: dict) -> str:
    """Batch key used for the acquisition-batch-effect plot."""
    return f"{info['session']}/{info['grid']}"


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------
# A variant is a parallel run of the whole workflow under a name suffix, so an
# alternative (e.g. polygon-masked crops) can be built and scored side by side
# with the baseline instead of overwriting it. "" is the baseline.

def variant(name: str = "", data_from: str = None) -> dict:
    """Resolve the paths for one variant.

    `data_from` points the INPUT paths (crops, manifest, normalized array) at a
    different variant while outputs still go under `name`. That lets two runs share
    one expensive crop set when they differ only downstream -- e.g. native-window
    crops embedded once at 224 and once at 448.
    """
    sfx = ("_" + name) if name else ""
    dsfx = sfx if data_from is None else (("_" + data_from) if data_from else "")
    return {
        "name": name or "baseline",
        "sfx": sfx,
        "data_sfx": dsfx,
        "crops": OUT / ("crops" + dsfx),
        "manifest": OUT / ("manifest%s.parquet" % dsfx),
        "norm": OUT / ("crops_norm%s.npy" % dsfx),
        "norm_meta": OUT / ("normalize_meta%s.json" % dsfx),
        "emb": OUT / ("embeddings%s.npy" % sfx),
        "emb_meta": OUT / ("embed_meta%s.json" % sfx),
        "clusters": OUT / ("clusters%s.parquet" % sfx),
        "cluster_meta": OUT / ("cluster_meta%s.json" % sfx),
        "eval": OUT / ("evaluation" + sfx),
    }


def umap_path(sfx: str, nn: int):
    return OUT / ("umap%s_nn%d.npy" % (sfx, nn))


# --------------------------------------------------------------------------
# Progress reporting (consumed by monitor.py)
# --------------------------------------------------------------------------
_PROGRESS = OUT / "progress.json"


class Progress:
    """Writes embedding outputs/progress.json so monitor.py can show a live ETA.

    Decoupled on purpose: the monitor works whether the step is in the foreground,
    backgrounded, or already finished.
    """

    def __init__(self, step: str, total: int, every: int = 25):
        self.step, self.total, self.every = step, total, max(1, every)
        self.done, self.started = 0, time.time()
        OUT.mkdir(parents=True, exist_ok=True)
        self._write("running")

    def _write(self, state: str):
        _PROGRESS.write_text(json.dumps({
            "step": self.step, "done": self.done, "total": self.total,
            "started_at": self.started, "updated_at": time.time(), "state": state,
        }), encoding="utf-8")

    def update(self, n: int = 1):
        self.done += n
        if self.done % self.every == 0 or self.done >= self.total:
            self._write("running")

    def finish(self):
        self.done = self.total
        self._write("done")
