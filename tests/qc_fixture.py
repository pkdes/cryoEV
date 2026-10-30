"""Synthetic web_qc review folder: images, cached predictions, review JSONs.

Built through annotation.web_qc.server.seed_objects, so the JSONs have the
format the browser tool saves. Known answer at conf >= 0.5 (EV truth = accepted
EV-type outlines + drawn):

  per image: EV 0.90 kept (TP), EV 0.60 rejected (FP), non-EV 0.80 (model id 1)
  accepted as non-EV, EV 0.30 left pending (below the cut), one EV drawn (FN).
  -> 2 images: tp 2, fn 2, and fp 2 + the non-EV outline when it is scored as EV.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 400, 300


def circle(cx, cy, r, n=24):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(float((cx + r * np.cos(a)) / W), float((cy + r * np.sin(a)) / H)) for a in t]


def pred_line(cls, conf, poly):
    return " ".join([str(cls), f"{conf:.4f}"] + [f"{v:.6f}" for xy in poly for v in xy])


# (model class id, conf, centre x, centre y, radius, reviewer status, reviewer class or None)
OBJECTS = [
    (0, 0.90, 80, 80, 30, "accepted", None),      # kept as seeded (EV)
    (0, 0.60, 220, 80, 25, "rejected", None),
    (1, 0.80, 320, 200, 28, "accepted", None),    # seeded class kept (non-EV or EV)
    (0, 0.30, 120, 220, 20, "pending", None),
]
DRAWN = (250, 230, 22)


def make_fake_tiles(root: Path, n_images: int = 2, grid_per_image: bool = False):
    """Write images/ and predictions/ under root; return both paths.

    grid_per_image: image k is on grid k+1 (one stratum per image) instead of all on grid 1.
    """
    images, preds = root / "images", root / "predictions"
    for d in (images, preds):
        d.mkdir(parents=True, exist_ok=True)
    for k in range(n_images):
        g = k + 1 if grid_per_image else 1
        name = f"g{g}_fake__grid{g}_{k + 1}_r0c{k}_t0{k}__nobar.png"
        rng = np.random.default_rng(k)
        Image.fromarray(rng.integers(90, 160, (H, W), dtype=np.uint8)).save(images / name)
        (preds / (Path(name).stem + ".txt")).write_text(
            "\n".join(pred_line(c, p, circle(x, y, r)) for c, p, x, y, r, _, _ in OBJECTS) + "\n",
            encoding="utf-8")
    return images, preds


def make_fake_review(root: Path, classes_path: Path, n_images: int = 2):
    """Write images/, predictions/, review/ under root; return the three paths."""
    images, preds = make_fake_tiles(root, n_images)
    review = root / "review"
    write_reviews(images, preds, review, classes_path)
    return images, preds, review


def write_reviews(images: Path, preds: Path, review: Path, classes_path: Path):
    """Review JSONs for every image, as a reviewer would leave them (all Done)."""
    from annotation.web_qc.server import load_classes, seed_objects

    cfg = load_classes(classes_path)
    review.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in images.iterdir())
    for name in names:
        txt = preds / (Path(name).stem + ".txt")
        objs = seed_objects(txt, W, H, cfg, role_conf=0.5)
        for o, (_, _, _, _, _, status, cls) in zip(objs, OBJECTS):
            o["status"] = status
            if cls:
                o["cls"] = cls
        x, y, r = DRAWN
        objs.append({"id": "h0", "poly": [list(p) for p in circle(x, y, r)], "conf": None,
                     "model_class": None, "role": None, "cls": cfg["role_to_class"]["standalone"],
                     "status": "accepted", "source": "human"})
        doc = {"image": name, "width": W, "height": H, "version": 1, "done": True,
               "updated": "2026-10-29T12:00:00", "conf_cut": 0.25, "objects": objs}
        (review / (Path(name).stem + ".json")).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    (review / "images_list.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
