"""
A prediction review package with the folders behind it (round 12: image info and sample / montage groups), built in a
temp folder for the tests and the screenshots:

  <root>/predict/raw_tiles/<CONV>/            a portal conversion: convert.json (recipe, params_effective), acquisition.json
                                              (per montage defocus ...), montages/layout.json (row / col), tiles/MANIFEST.csv
  <root>/predict/<PS>/                        images.csv (orig_name = the tile names), request.json (source.dir = the tiles),
                                              synthetic PNGs and prediction .txt files
  <root>/review/pred_<PS>/                    built by annotation/prediction_package.build

Samples and montages (package order = images.csv order):
  0001-0004  EV-013-2  grid10_1  r0c0 r0c1 r1c0 r1c1   defocus -4.48
  0005-0006  EV-013-2  grid10_2  r0c0 r0c1             defocus -2.21
  0007-0008  HEK       grid6_1   r0c0 r0c1             defocus -5.63
  0009       HEK       grid6_9   r0c0                  not in MANIFEST / layout / acquisition: montage and tile from
                                                       its name only, defocus not recorded
Everything is deterministic. Used by test_review_meta_groups.py and test_review_meta_groups_browser.py.
"""
import contextlib
import csv
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from annotation import prediction_package as pp  # noqa: E402

PS = "ps_20261113_a1b2c3"
PKG = f"pred_{PS}"
CONV = "20261106-9_abc123"
MODEL = "v5_evnonev_y8m_20261029"
W, H = 720, 512
TILES = [  # image, grid, sample, montage, row, col, piece
    ("0001", 10, "EV-013-2", "grid10_1", 0, 0, 2), ("0002", 10, "EV-013-2", "grid10_1", 0, 1, 5),
    ("0003", 10, "EV-013-2", "grid10_1", 1, 0, 1), ("0004", 10, "EV-013-2", "grid10_1", 1, 1, 4),
    ("0005", 10, "EV-013-2", "grid10_2", 0, 0, 0), ("0006", 10, "EV-013-2", "grid10_2", 0, 1, 3),
    ("0007", 6, "HEK", "grid6_1", 0, 0, 0), ("0008", 6, "HEK", "grid6_1", 0, 1, 3),
    ("0009", 6, "HEK", "grid6_9", 0, 0, 0),
]
ACQ = {"grid10_1": -4.4799, "grid10_2": -2.2061, "grid6_1": -5.632}
IN_MANIFEST = {"0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008"}


def tile_name(grid, sample, montage, row, col, piece):
    """export_detector_tiles.training_name, as the processing repo writes it."""
    return f"g{grid}_{sample}__{montage}_r{row}c{col}_t{piece:02d}__denoised_nobar_14p72Apx.png"


def _circle(cx, cy, r, n=24):
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        out += [(cx + r * math.cos(a)) / W, (cy + r * math.sin(a)) / H]
    return out


def _image(seed):
    rng = np.random.default_rng(seed)
    im = rng.normal(140, 14, (H, W))
    yy, xx = np.mgrid[:H, :W]
    circles = [(150 + 37 * seed % 300, 140, 60), (420, 300 - 9 * seed, 48), (600, 120 + 11 * seed, 34)]
    for cx, cy, r in circles:
        d = np.hypot(xx - cx, yy - cy)
        im[np.abs(d - r) < 2.2] -= 75
    return Image.fromarray(np.clip(im, 0, 255).astype(np.uint8), "L"), circles


def write_conversion(root: Path) -> Path:
    conv = root / "predict" / "raw_tiles" / CONV
    (conv / "tiles").mkdir(parents=True)
    (conv / "montages").mkdir()
    with open(conv / "tiles" / "MANIFEST.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["output_name", "source_file", "grid", "sample_label", "montage", "nm_per_px", "md5"])
        for image, grid, sample, montage, row, col, piece in TILES:
            if image in IN_MANIFEST:
                w.writerow([tile_name(grid, sample, montage, row, col, piece), f"{montage}_r{row}c{col}_t{piece:02d}.jpg",
                            grid, sample, montage, "1.47240", "0" * 32])
    (conv / "convert.json").write_text(json.dumps({
        "schema": "cj_convert/1", "id": CONV, "owner": "pujan", "session": "20261106-9", "recipe": "20261027",
        "gpu": True, "inputs": ["C:\\somewhere\\input\\grid10-position1.idoc"],
        "params_effective": {"clip_lo_pct": 1.0, "clip_hi_pct": 99.0, "bandpass_nm": 2.0, "denoise": True},
        "params_overrides": {}, "variant": "denoised"}, indent=1), encoding="utf-8")
    (conv / "acquisition.json").write_text(json.dumps({
        "schema": "acquisition/1",
        "montages": [{"montage": m, "mag": 11000, "pixel_A": 7.362, "defocus_um": d, "dose": 3.6239,
                      "acquired": "2026-11-06T13:11:24", "exposure_s": 2.996, "n_pieces": 9} for m, d in ACQ.items()],
        "session": {"instrument": "Glacios", "magnification": [11000], "pixel_size_a": [7.362],
                    "defocus_um": {"min": -5.632, "median": -4.48, "max": -2.2061}}}, indent=1), encoding="utf-8")
    lay = {}
    for image, grid, sample, montage, row, col, piece in TILES:
        if image in IN_MANIFEST:
            lay.setdefault(montage, []).append({"output_name": tile_name(grid, sample, montage, row, col, piece),
                                                "row": row, "col": col, "x": col * 640, "y": row * 450, "w": 714, "h": 507})
    (conv / "montages" / "layout.json").write_text(json.dumps({"schema": "montage_layout/1", "montages": [
        {"name": m, "file": f"{m}_stitched.jpg", "tiles": t} for m, t in lay.items()]}), encoding="utf-8")
    return conv


def write_prediction_set(root: Path, conv: Path) -> Path:
    ps = root / "predict" / PS
    (ps / "images").mkdir(parents=True)
    pred = ps / "p" / MODEL / "predictions"
    pred.mkdir(parents=True)
    rows = []
    for n, (image, grid, sample, montage, row, col, piece) in enumerate(TILES, 1):
        im, circles = _image(n)
        im.save(ps / "images" / f"{image}.png")
        lines = []
        for k, (cx, cy, r) in enumerate(circles):
            conf = (0.91, 0.72, 0.38)[k]
            lines.append(f"0 {conf:.2f} " + " ".join(f"{v:.6f}" for v in _circle(cx, cy, r)))
        (pred / f"{image}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        rows.append([image, f"images/{image}.png", tile_name(grid, sample, montage, row, col, piece), sample, "1.4724",
                     "pujan", "session:20261106-9", "0" * 32])
    with open(ps / "images.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "file", "orig_name", "sample_label", "nm_per_px", "nm_confirmed_by", "source", "md5"])
        w.writerows(rows)
    (ps / "request.json").write_text(json.dumps({
        "ps": PS, "owner": "pujan", "model": MODEL, "conf": 0.25, "max_det": 1000, "imgsz": None,
        "source": {"kind": "session", "session": "20261106-9", "dir": str(conv / "tiles")},
        "created": "2026-11-13T09:00:00"}, indent=1), encoding="utf-8")
    (ps / "status.json").write_text(json.dumps({"state": "done", "n_done": len(TILES), "n_total": len(TILES)}),
                                    encoding="utf-8")
    return ps


def build(root) -> dict:
    """The whole fixture under root -> {"root", "conv", "ps", "review", "pkg"}."""
    root = Path(root)
    conv = write_conversion(root)
    ps = write_prediction_set(root, conv)
    review = root / "review"
    review.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):   # the builder's progress lines
        pkg = pp.build(ps, review)
    return {"root": root, "conv": conv, "ps": ps, "review": review, "pkg": pkg}


if __name__ == "__main__":
    import tempfile
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="metafix_"))))
