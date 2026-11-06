"""EV counts and sizes per sample and per image for one prediction_set (BUILD_CONTRACTS.md section 4).

    from results import compute, write_results
    res = compute(ps_dir)                       # model predictions at conf >= 0.5
    res = compute(ps_dir, "reviewed.json")      # reviewed polygons (contract section 3); a dict also works
    write_results(res, ps_dir / "results")      # results.json + per_image.csv + per_sample.csv

The rule is the existing counting chain (size_profile_from_predictions -> add_layer_roles ->
summarize_detections), called, not re-implemented:
  * EV class    inference.model_classes.ev_class_ids (class 0 for a multi-class model; every class for a
                single-class one), applied BEFORE layer roles, so an EV inside a non-EV outline stays an EV;
  * outlines    model predictions: training.train_yolo.load_predictions_yolo_format at conf >= 0.5;
                reviewed polygons: analysis.multilayer_containment.polygon_to_mask (same int truncation);
  * roles       analysis.add_layer_roles.roles_for (containment 0.99, size ratio 0.75): 'inner-layer'
                outlines are not EVs, a 'multilayer' outline is one multilayer EV;
  * diameter    analysis.morphology.analyze_instances equivalent_diameter (area-equivalent, nm via the
                image's nm_per_px); an outline that rasterizes to nothing is dropped, as in the chain;
  * non-EV      accepted outlines of the other classes, counted separately (as summarize_detections n_nonev).

Reviewed images (reviewed.json `reviewed: true`): every object listed is accepted (a reviewer may keep a
0.25-0.50 candidate or draw a new outline). A set `reviewer_class` wins over `cls`: EV / MV / inner -> EV
class, nonEV -> non-EV. pct_multilayer_reviewer = share of the EVs on reviewed images that the reviewer
classed MV; null when the sample has no reviewed image. Unreviewed images keep conf >= 0.5.

A sample whose images differ in tile size or nm/px, or lack a confirmed nm/px, is refused (left out of
`samples`, reason in `refused`), never mixed or defaulted.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from data_utils import config
from data_utils.longpath import long_path, prediction_file
from inference.model_classes import ev_class_ids, model_classes

SCHEMA = "results/1"
CONF = 0.5
MULTILAYER_RULE = "geometry: outline containing another outline"
REVIEWER_EV = {"EV", "MV", "inner"}
REVIEWER_NONEV = {"nonEV"}
SAMPLE_FIELDS = ["n_images", "n_reviewed", "ev_total", "ev_per_image_mean", "diam_nm_median", "diam_nm_q1",
                 "diam_nm_q3", "pct_multilayer_geometry", "pct_multilayer_reviewer", "inner_layers_excluded",
                 "nonev_total"]
IMAGE_FIELDS = ["sample_label", "reviewed", "nm_per_px", "ev", "multilayer", "inner", "nonev"]
IMAGES_CSV_COLUMNS = {"image", "file", "sample_label", "nm_per_px", "nm_confirmed_by"}


# ---- the existing chain, imported in one place ------------------------------------------------------------

def _chain():
    """(load_predictions_yolo_format, polygon_to_mask, roles_for, analyze_instances) from the counting chain.

    analysis.add_layer_roles takes its validated containment pair (0.99 / 0.75) from analysis.layer_count_stats,
    whose module body resolves the CryoAI training folder for its own CLI defaults and raises ConfigError on a
    host without one. Counting never reads that folder, so on that error the import is retried with
    $CRYOEV_CRYOAI_ROOT pointed at the repo for the duration of the import only (the thresholds stay the
    chain's own, not a copy)."""
    from training.train_yolo import load_predictions_yolo_format
    from analysis.multilayer_containment import polygon_to_mask
    from analysis.morphology import analyze_instances
    try:
        from analysis.add_layer_roles import roles_for
    except config.ConfigError:
        old = os.environ.get("CRYOEV_CRYOAI_ROOT")
        os.environ["CRYOEV_CRYOAI_ROOT"] = str(config.REPO)
        try:
            from analysis.add_layer_roles import roles_for
        finally:
            if old is None:
                os.environ.pop("CRYOEV_CRYOAI_ROOT", None)
            else:
                os.environ["CRYOEV_CRYOAI_ROOT"] = old
    return load_predictions_yolo_format, polygon_to_mask, roles_for, analyze_instances


# ---- inputs -----------------------------------------------------------------------------------------------

@dataclass
class _Obj:
    mask: np.ndarray
    conf: float | None
    cls: int
    reviewer_class: str | None = None


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_images_csv(ps_dir: Path) -> list[dict]:
    path = ps_dir / "images.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    missing = IMAGES_CSV_COLUMNS - set(rows[0] if rows else IMAGES_CSV_COLUMNS)
    if not rows or missing:
        raise ValueError(f"{path}: {'no images' if not rows else 'missing columns ' + str(sorted(missing))}")
    seen = set()
    for r in rows:
        if r["image"] in seen:
            raise ValueError(f"{path}: image {r['image']} listed twice")
        seen.add(r["image"])
        if not (r["sample_label"] or "").strip():
            raise ValueError(f"{path}: image {r['image']} has no sample_label")
    return sorted(rows, key=lambda r: r["image"])


def _nm_per_px(row: dict) -> float | None:
    """The image's nm/px, or None when it is missing, not a positive number, or not confirmed by anyone."""
    try:
        nm = float(row["nm_per_px"])
    except (TypeError, ValueError):
        return None
    if not (nm > 0 and math.isfinite(nm)) or not (row.get("nm_confirmed_by") or "").strip():
        return None
    return nm


def _image_size(ps_dir: Path, rel: str) -> tuple[int, int]:
    path = (ps_dir / rel).resolve()
    if ps_dir.resolve() not in path.parents:
        raise ValueError(f"images.csv file {rel!r} is outside {ps_dir}")
    with Image.open(long_path(path)) as im:
        return im.size


def _is_ev(o: _Obj, ev_ids: set[int] | None) -> bool:
    if o.reviewer_class is not None:
        if o.reviewer_class in REVIEWER_EV:
            return True
        if o.reviewer_class in REVIEWER_NONEV:
            return False
        raise ValueError(f"unknown reviewer_class {o.reviewer_class!r}")
    return ev_ids is None or o.cls in ev_ids


def _objects_from_predictions(pred_dir: Path, image: str, w: int, h: int, load) -> list[_Obj]:
    masks, confs, classes = load(prediction_file(pred_dir, image, image), w, h)   # missing file = error
    return [_Obj(m, c, k) for m, c, k in zip(masks, confs, classes) if c >= CONF]


def _objects_from_reviewed(entry: dict, image: str, w: int, h: int, to_mask) -> list[_Obj]:
    out = []
    for n, o in enumerate(entry.get("objects") or []):
        poly = o.get("poly") or []
        if len(poly) < 3:
            raise ValueError(f"reviewed image {image} object {n}: polygon has {len(poly)} point(s)")
        conf = o.get("conf")
        if not entry.get("reviewed") and conf is not None and conf < CONF:
            continue   # unreviewed image: model predictions at conf >= 0.5 only (contract section 3)
        mask = to_mask([float(c) for xy in poly for c in xy], w, h)
        out.append(_Obj(mask, conf, int(o["cls"]), o.get("reviewer_class")))
    return out


# ---- per image --------------------------------------------------------------------------------------------

def _measure(objs: list[_Obj], nm_per_px: float | None, ev_ids, roles_for, analyze_instances) -> dict:
    """Counts for one image's accepted outlines, plus the EV diameters (nm) and reviewer-MV count."""
    ev_objs = [o for o in objs if _is_ev(o, ev_ids)]
    masks = [o.mask for o in ev_objs]
    role, _ = roles_for(masks)
    recs = analyze_instances(masks, pixel_size=nm_per_px)   # drops an outline that rasterizes to nothing
    ev = ml = inner = mv = 0
    diam = []
    for r in recs:
        i = r["object_id"]
        if role[i] == "inner-layer":
            inner += 1
            continue
        ev += 1
        diam.append(float(r["equivalent_diameter"]))
        ml += role[i] == "multilayer"
        mv += ev_objs[i].reviewer_class == "MV"
    return {"ev": ev, "multilayer": ml, "inner": inner, "nonev": len(objs) - len(ev_objs),
            "_diam": diam, "_mv": mv}


# ---- per sample -------------------------------------------------------------------------------------------

def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _sample(imgs: list[dict]) -> dict:
    ev_total = sum(i["ev"] for i in imgs)
    ml = sum(i["multilayer"] for i in imgs)
    d = np.array([x for i in imgs for x in i["_diam"]], dtype=float)
    q1, med, q3 = (np.percentile(d, [25, 50, 75]) if len(d) else (None, None, None))
    rev = [i for i in imgs if i["reviewed"]]
    ev_rev = sum(i["ev"] for i in rev)
    return {
        "n_images": len(imgs),
        "n_reviewed": len(rev),
        "ev_total": ev_total,
        "ev_per_image_mean": _r(ev_total / len(imgs)),
        "diam_nm_median": _r(med),
        "diam_nm_q1": _r(q1),
        "diam_nm_q3": _r(q3),
        "pct_multilayer_geometry": _r(100.0 * ml / ev_total) if ev_total else None,
        "pct_multilayer_reviewer": _r(100.0 * sum(i["_mv"] for i in rev) / ev_rev) if ev_rev else None,
        "inner_layers_excluded": sum(i["inner"] for i in imgs),
        "nonev_total": sum(i["nonev"] for i in imgs),
    }


def _refusal(label: str, imgs: list[dict]) -> str | None:
    """Why this sample can't be summarised as one population, or None."""
    no_nm = [i["_image"] for i in imgs if i["nm_per_px"] is None]
    if no_nm:
        return (f"sample {label!r}: no confirmed nm_per_px for image(s) {', '.join(no_nm)} "
                f"(images.csv needs nm_per_px and nm_confirmed_by)")
    why = []
    nms = sorted({i["nm_per_px"] for i in imgs})
    if len(nms) > 1:
        why.append(f"nm/px {', '.join(f'{v:g}' for v in nms)}")
    sizes = sorted({i["_size"] for i in imgs})
    if len(sizes) > 1:
        why.append(f"tile sizes {', '.join(f'{w}x{h}' for w, h in sizes)}")
    if why:
        return f"sample {label!r} mixes {' and '.join(why)}; split it into one sample per scale"
    return None


# ---- entry points -----------------------------------------------------------------------------------------

def compute(ps_dir, reviewed_json=None) -> dict:
    """results/1 dict for the prediction_set folder `ps_dir` (contract section 4). `reviewed_json`: None (model
    predictions at conf >= 0.5), a path to reviewed.json, or the dict reviewed_polygons() returns."""
    ps_dir = Path(ps_dir)
    request = _read_json(ps_dir / "request.json")
    ps_id, model = request["ps"], request["model"]
    reviewed = reviewed_json if isinstance(reviewed_json, dict) or reviewed_json is None \
        else _read_json(Path(reviewed_json))
    if reviewed is not None:
        if reviewed.get("schema") != "reviewed/1":
            raise ValueError(f"reviewed json schema is {reviewed.get('schema')!r}, expected 'reviewed/1'")
        for key, want in (("prediction_set", ps_id), ("model", model)):
            if reviewed.get(key) != want:
                raise ValueError(f"reviewed json {key} {reviewed.get(key)!r} != this prediction set's {want!r}")

    ev_ids = ev_class_ids(model)
    rule_classes = sorted(ev_ids) if ev_ids is not None else (sorted(model_classes(model)) or [0])
    result = {"schema": SCHEMA, "prediction_set": ps_id, "model": model,
              "computed": datetime.now().astimezone().isoformat(timespec="seconds"),
              "rule": {"conf": CONF, "ev_classes": rule_classes, "multilayer": MULTILAYER_RULE},
              "samples": {}, "images": {}, "refused": None}

    status_path = ps_dir / "status.json"
    if status_path.is_file():
        status = _read_json(status_path)
        if status.get("state") != "done":
            result["refused"] = (f"prediction set is {status.get('state')!r} "
                                 f"({status.get('n_done')}/{status.get('n_total')} images); count when done")
            return result

    rows = _read_images_csv(ps_dir)
    if reviewed is not None:
        listed, have = {r["image"] for r in rows}, set(reviewed.get("images") or {})
        if listed != have:
            raise ValueError(f"reviewed json images differ from images.csv: missing {sorted(listed - have)}, "
                             f"extra {sorted(have - listed)}")

    load, to_mask, roles_for, analyze_instances = _chain()
    pred_dir = ps_dir / "p" / model / "predictions"
    per_image = {}
    for r in rows:
        image = r["image"]
        w, h = _image_size(ps_dir, r["file"])
        nm = _nm_per_px(r)
        if reviewed is None:
            is_rev, objs = False, _objects_from_predictions(pred_dir, image, w, h, load)
        else:
            entry = reviewed["images"][image]
            is_rev, objs = bool(entry.get("reviewed")), _objects_from_reviewed(entry, image, w, h, to_mask)
        m = _measure(objs, nm, ev_ids, roles_for, analyze_instances)
        per_image[image] = {"sample_label": r["sample_label"], "reviewed": is_rev, "nm_per_px": nm,
                            **{k: m[k] for k in ("ev", "multilayer", "inner", "nonev")},
                            "_diam": m["_diam"], "_mv": m["_mv"], "_size": (w, h), "_image": image}

    by_sample = {}
    for img in per_image.values():
        by_sample.setdefault(img["sample_label"], []).append(img)
    refusals = []
    for label, imgs in by_sample.items():
        why = _refusal(label, imgs)
        if why:
            refusals.append(why)
        else:
            result["samples"][label] = _sample(imgs)
    result["images"] = {k: {f: v[f] for f in IMAGE_FIELDS} for k, v in per_image.items()}
    result["refused"] = "; ".join(refusals) or None
    return result


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, path)


def _csv_text(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows([["" if v is None else v for v in row] for row in rows])
    return buf.getvalue()


def write_results(result: dict, out_dir) -> Path:
    """results.json, per_image.csv and per_sample.csv under `out_dir` (each replaced atomically)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(out_dir / "results.json", json.dumps(result, indent=1) + "\n")
    _atomic_write(out_dir / "per_image.csv", _csv_text(
        ["image"] + IMAGE_FIELDS, [[k] + [v[f] for f in IMAGE_FIELDS] for k, v in result["images"].items()]))
    _atomic_write(out_dir / "per_sample.csv", _csv_text(
        ["sample_label"] + SAMPLE_FIELDS, [[k] + [v[f] for f in SAMPLE_FIELDS] for k, v in result["samples"].items()]))
    return out_dir


def compute_and_write(ps_dir, reviewed_json=None, out_dir=None) -> tuple[dict, Path]:
    """compute() then write_results() to `out_dir` (default <ps_dir>/results)."""
    res = compute(ps_dir, reviewed_json)
    return res, write_results(res, out_dir or Path(ps_dir) / "results")
