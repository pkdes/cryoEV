"""
Write the synthetic review package used by the review tool tests: three small generated PNGs and a
data.js in the build_review_package.py format.

  fixA_new: has the optional flag fields (pr[k].c, x, lc, fx, flc) that newer packages carry
  fixB_old, fixC_old: the older format without them (as in the September cryoai_v4_review package)
Object ids and outlines are fixed; annotation/tests/test_review_tool_browser.py relies on them.

Categories are in the old, alphabetical export order and lack VLDL, so the tests also cover the
tool's fixed class order. Deterministic: rerunning gives identical files.

Usage (also gives a package to try the tool on by hand):
  python annotation/tests/make_fixture.py
  python annotation/review_server.py --package annotation/tests/fixture_review
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "review_tool" / "index.html"
W, H = 360, 256
CATEGORIES = ["EV", "Incomplete", "Low conf EV", "MV", "Overlapping", "inner", "non-EV"]  # old export order


def ring(cx, cy, r, n=20):
    """Circle as a flat normalized ring, rounded like build_review_package.py."""
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        out += [round((cx + r * math.cos(a)) / W, 5), round((cy + r * math.sin(a)) / H, 5)]
    return out


def image(seed, circles):
    rng = np.random.default_rng(seed)
    im = rng.normal(150, 12, (H, W))
    yy, xx = np.mgrid[:H, :W]
    for cx, cy, r in circles:
        d = np.hypot(xx - cx, yy - cy)
        im[np.abs(d - r) < 1.6] -= 70  # membrane
    return Image.fromarray(np.clip(im, 0, 255).astype(np.uint8), "L")


def build(out: Path):
    (out / "images").mkdir(parents=True, exist_ok=True)
    a = [(70, 70, 28), (70, 70, 16), (200, 80, 30), (290, 170, 24), (120, 190, 22), (230, 200, 12)]
    b = [(90, 120, 35), (250, 110, 26), (180, 210, 14), (300, 210, 18)]
    c = [(120, 100, 30), (260, 150, 22)]
    # Package order is B, A, C (fg + fp: 3, 2, 0, as build_review_package.py sorts); with fx + flc, A has the
    # most flags (4), so "most flags" and "package order" differ.
    images = [
        {"file": "fixB_old.png", "split": "valid", "w": W, "h": H, "src": "images/fixB_old.png",
         "gt": [{"id": 201, "c": "EV", "p": [ring(*b[0])]},
                {"id": 202, "c": "Low conf EV", "p": [ring(*b[1])], "u": 1},
                {"id": 203, "c": "MV", "p": [ring(*b[3])], "u": 1}],
         "pr": [{"s": 0.88, "p": [ring(*b[0])]},
                {"s": 0.52, "p": [ring(*b[2])], "u": 1}],
         "fg": 2, "fp": 1},
        {"file": "fixA_new.png", "split": "train", "w": W, "h": H, "src": "images/fixA_new.png",
         "gt": [{"id": 101, "c": "EV", "p": [ring(*a[0])]},
                {"id": 102, "c": "inner", "p": [ring(*a[1])]},
                {"id": 103, "c": "MV", "p": [ring(*a[2])], "x": True},
                {"id": 104, "c": "Low conf EV", "p": [ring(*a[3])], "lc": True, "u": 1},
                {"id": 105, "c": "EV", "p": [ring(*a[4])], "u": 1},
                {"id": 106, "c": "VLDL", "p": [ring(*a[5])]}],
         "pr": [{"s": 0.91, "p": [ring(*a[0])], "c": "EV"},
                {"s": 0.84, "p": [ring(*a[1])], "c": "inner"},
                {"s": 0.77, "p": [ring(*a[2])], "c": "EV", "x": True},
                {"s": 0.66, "p": [ring(*a[5])], "c": "VLDL"}],
         "fg": 2, "fp": 0, "fx": 1, "flc": 1},
        {"file": "fixC_old.png", "split": "test", "w": W, "h": H, "src": "images/fixC_old.png",
         "gt": [{"id": 301, "c": "EV", "p": [ring(*c[0])]},
                {"id": 302, "c": "MV", "p": [ring(*c[1])]}],
         "pr": [{"s": 0.93, "p": [ring(*c[0])]}, {"s": 0.81, "p": [ring(*c[1])]}],
         "fg": 0, "fp": 0},
    ]
    for k, (im, circles) in enumerate(zip(images, (b, a, c))):
        image(k, circles).save(out / im["src"], optimize=True)
    data = {"id": "fixture_review", "source": "synthetic", "run": "synthetic predictions",
            "created": "2026-10-29T00:00:00", "categories": CATEGORIES, "images": images}
    (out / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    shutil.copy(TEMPLATE, out / "index.html")  # not committed: always the current template
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "fixture_review")
    args = ap.parse_args()
    data = build(args.out)
    print(f"[OK] {args.out}: {len(data['images'])} images")


if __name__ == "__main__":
    main()
