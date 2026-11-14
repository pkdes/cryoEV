"""A throwaway world for Start a review (round 13, B4): temp CryoAI-laptop (labels, review, predict), temp registry and
a temp Roboflow export folder. Never the real data, the real registry or ~/.cryoev.

  labels/lv_20990101_0101   current label set, 3 images (img_a, img_b, img_c 400x300), made on lv_20990101_0000
  labels/lv_20990101_0000   its base (superseded), same images, other outlines: B for a label set vs label set build
  export "roboflow - fixture" (registered, path = <tmp>/export) with train/ and valid/ images, a split CSV; img_c is
  on disk only under its session-prefixed name "S1__img_c_png" and the registry's image record img_c_png says
  duplicate_of S1__img_c_png (the alias link)
  dataset "lv_20990101_0101_ds" (registered): label_version lv_20990101_0101, export "roboflow - fixture"
  predict/ps_20990101_aaaaaa: done, no package (Start (a)); ps_20990101_bbbbbb: done, other owner
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

LV, LV_BASE = "lv_20990101_0101", "lv_20990101_0000"
EXPORT = "roboflow - fixture"
W, H = 400, 300
CATS = [{"id": 1, "name": "EV"}, {"id": 2, "name": "Low conf EV"}, {"id": 3, "name": "non-EV"}]


def ring(cx, cy, r, n=16):
    return [round(v, 2) for k in range(n) for v in (cx + r * math.cos(2 * math.pi * k / n),
                                                   cy + r * math.sin(2 * math.pi * k / n))]


def coco(shift=0):
    ims, anns, aid = [], [], 1
    for i, stem in enumerate(("img_a", "img_b", "img_c")):
        ims.append({"id": i + 1, "file_name": f"{stem}.png", "width": W, "height": H,
                    "extra": {"name": f"{stem}_png", "split": "train"}})
        for cx, cy in ((100, 100), (300, 100 + shift), (200, 220)):
            anns.append({"id": aid, "image_id": i + 1, "category_id": 1, "segmentation": [ring(cx, cy, 20)]})
            aid += 1
        if shift:                       # label set B adds one more and moves one: real disagreements
            anns.append({"id": aid, "image_id": i + 1, "category_id": 1, "segmentation": [ring(330, 250, 18)]})
            aid += 1
    return {"images": ims, "categories": CATS, "annotations": anns}


def picture(path, w=W, h=H):
    im = Image.new("L", (w, h), 120)
    d = ImageDraw.Draw(im)
    for cx, cy in ((100, 100), (300, 100), (200, 220)):
        d.ellipse((cx - 20, cy - 20, cx + 20, cy + 20), outline=40, width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def jline(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def build(tmp: Path, owner="pujan") -> dict:
    """-> {laptop, labels, review, predict, registry, export}."""
    tmp = Path(tmp)
    lap = tmp / "laptop"
    labels, review, predict, reg, export = lap / "labels", lap / "review", lap / "predict", tmp / "registry", tmp / "export"
    for d in (review, predict):
        d.mkdir(parents=True, exist_ok=True)
    for lv, shift, extra in ((LV, 0, {"base": LV_BASE, "supersedes": LV_BASE, "pkg": "lab_fixture_20990101",
                                      "n_annotations_out": 9}),
                             (LV_BASE, 30, {"base": "roboflow_pull_20990101_0000", "n_annotations_out": 12})):
        (labels / lv).mkdir(parents=True, exist_ok=True)
        (labels / lv / "labels.coco.json").write_text(json.dumps(coco(shift)), encoding="utf-8")
        (labels / lv / "version.json").write_text(json.dumps({"id": lv, **extra}), encoding="utf-8")
    picture(export / "train" / "img_a_png.rf.0001.jpg")
    picture(export / "valid" / "img_b_png.rf.0002.jpg")
    picture(export / "train" / "S1__img_c_png.rf.0003.jpg")
    (export / "split_assignment.csv").write_text("image,split\nimg_a_png,train\nimg_b_png,valid\nS1__img_c_png,test\n",
                                                 encoding="utf-8")
    (reg / "live").mkdir(parents=True, exist_ok=True)
    jline(reg / "roboflow_export.jsonl", {"schema": "roboflow_export/1", "id": EXPORT, "path": str(export),
                                          "split_file": "split_assignment.csv", "logged": "2099-01-01T00:00:00"})
    jline(reg / "dataset.jsonl", {"schema": "dataset/1", "id": f"{LV}_ds", "label_version": LV, "export": EXPORT,
                                  "source": {"kind": "label_version", "id": LV}, "logged": "2099-01-01T00:00:00"})
    jline(reg / "image.jsonl", {"schema": "image/1", "id": "img_c_png", "duplicate_of": "S1__img_c_png",
                                "logged": "2099-01-01T00:00:00"})
    for ps, who in (("ps_20990101_aaaaaa", owner), ("ps_20990101_bbbbbb", "someone")):
        d = predict / ps
        d.mkdir(parents=True, exist_ok=True)
        (d / "request.json").write_text(json.dumps({"ps": ps, "owner": who, "model": "v5_evnonev_y8m_20261029",
                                                    "created": "2099-01-01T10:00:00",
                                                    "name": f"Fixture set {ps[-6:]}",
                                                    **({"owner_via": "claude"} if who == owner else {})}),
                                        encoding="utf-8")
        (d / "status.json").write_text(json.dumps({"state": "done", "n_done": 4, "n_total": 4}), encoding="utf-8")
    return {"laptop": lap, "labels": labels, "review": review, "predict": predict, "registry": reg, "export": export}
