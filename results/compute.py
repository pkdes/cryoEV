"""EV counts and sizes per sample and per image for one prediction_set (BUILD_CONTRACTS.md section 4).

    from results import compute, write_results
    res = compute(ps_dir)                       # model predictions at conf >= 0.5
    res = compute(ps_dir, "reviewed.json")      # with reviewed images (contract section 3); a dict also works
    write_results(res, ps_dir / "results")      # results.json + per_image.csv + per_sample.csv

compute() reads and rasterizes every outline, so a big set takes minutes: callers run it in a background
job or thread (or `python -m results`), and pages read the cached <ps>/results/results.json.

The rule is the existing counting chain (size_profile_from_predictions -> add_layer_roles ->
summarize_detections), called, not re-implemented:
  * EV class    inference.model_classes.ev_class_ids on each outline's `cls` (class 0 for a multi-class
                model; every class for a single-class one), applied BEFORE layer roles, so an EV inside a
                non-EV outline stays an EV;
  * outlines    model predictions: training.train_yolo.load_predictions_yolo_format at conf >= 0.5;
                reviewed polygons: analysis.multilayer_containment.polygon_to_mask (same int truncation);
  * roles       analysis.add_layer_roles.roles_for (containment 0.99, size ratio 0.75): 'inner-layer'
                outlines are not EVs, a 'multilayer' outline is one multilayer EV;
  * diameter    area-equivalent, sqrt(4 * mask pixels / pi) * nm_per_px: analysis.morphology's
                equivalent_diameter formula, without the contour / hull / ellipse work it also does.
                An outline analyze_instances would drop (no pixels, or a zero-length contour) is dropped;
                the test suite checks the diameters equal analyze_instances' on golden and real sessions;
  * non-EV      accepted outlines of the other classes, counted separately (as summarize_detections n_nonev).

reviewed.json images with `edited: true` or `reviewed: true` (a missing `edited` key counts as equal to
`reviewed`): the objects listed are the outlines (a reviewer may delete, redraw, keep a 0.25-0.50 candidate
or add one), so deletions and redraws move the geometry numbers. Only `reviewed: true` images count toward
n_reviewed / reviewed_images and the reviewer columns; an edited-but-not-reviewed image (e.g. the phone card
flow) moves the geometry numbers only. `reviewer_label` (raw) is not read; `reviewer_class` is.
`reviewer_class` labels NEVER move the geometry headline; they feed only the reviewer columns:
reviewer_ev (EV + MV calls), reviewer_mv, reviewer_inner, reviewer_nonev (nonEV calls on EV-class outlines),
pct_multilayer_reviewer = reviewer_mv / reviewer_ev, and reviewer_geometry_disagree: EV/MV/inner calls whose
geometry role is not standalone / multilayer / inner-layer respectively (a call on a non-EV-class outline
always disagrees), plus nonEV calls on EV-class outlines (geometry counts them as an EV or an inner layer:
the model-grading signal). A nonEV call on a non-EV-class outline agrees.
Per image, `edited` is reviewed.json's `edited` flag (a missing key = `reviewed`; false without reviewed.json),
so a page can show "edited but not marked reviewed".
Images neither edited nor reviewed always come from the prediction .txt at conf >= 0.5, never from
reviewed.json, so editing one image cannot shift another.

Unsure outlines (additive, results/1 unchanged). A candidate is a prediction with CAND_CONF <= conf < CONF; it is
NOT counted until a reviewer accepts it. Per sample `n_unsure_open` = candidates nobody has decided, by image:
  * untouched (neither edited nor reviewed)  all its candidates (reviewed.json's `n_candidates_open` when given,
                                             else counted from the prediction .txt);
  * edited, not Reviewed                     reviewed.json's `n_candidates_open` (no accept or reject decision);
  * Reviewed                                 0 (marking an image Reviewed closes its remaining candidates).
Without reviewed.json (results computed without a review package) every image is untouched, so they are counted
from the predictions at CAND_CONF <= conf < CONF. `n_unsure_open` is null when an edited, not Reviewed image comes
from a reviewed.json written before `n_candidates_open` existed (unknown, not zero). `n_checked` = images edited or
reviewed. `diam_hist` = {"bin_nm": 10, "max_nm": 500, "counts": [51]}: counts of the sample's per-EV diameters
(the same ones as the median), bin k = [10k, 10k+10) nm, the last bin = 500 nm and above; sum(counts) == ev_total.
`diam_hist` is in results.json only, not in SAMPLE_FIELDS / per_sample.csv.

Rejected images and low confidence (additive, results/1 unchanged; BUILD_CONTRACTS.md section 14). reviewed.json
images may carry `rejected: true`, `reject_reason`, `note`; objects `low_conf: true`, `note` (read with .get). ONE rule:
rejected images are dropped from a sample before the refusal check and every number (EV, non-EV, inner, reviewer %,
nm/px min / max, n_images = counted images, refused_unsure, diam_hist). They stay in `images` / per_image.csv with
`rejected`, `reject_reason`, `note`. Per sample `n_rejected` and `n_low_conf` (counted EVs that carry low conf: a
low-conf non-EV or inner layer is not counted). A sample whose every image is rejected has all numbers null,
n_images 0, n_rejected k and `all_rejected: true` (JSON only). CSV text cells starting with = + - @, tab or CR get a
' prefix (`guard_cell`).

Nullable fields: per sample diam_nm_median / diam_nm_q1 / diam_nm_q3 and pct_multilayer_geometry are null
when the sample has no EV; pct_multilayer_reviewer is null with no reviewed image or no reviewer EV/MV call;
per image nm_per_px is null when images.csv has no confirmed nm/px (that sample is then refused); `refused`
is null unless something was refused.

A sample whose images differ in tile size, or whose nm/px values spread (max - min) by more than NM_TOLERANCE
(1 %) of their median, or that lack a confirmed nm/px, is refused (left out of `samples`, reason in `refused`),
never mixed or defaulted. Within the tolerance the sample counts as one scale (contract section 8): each image
keeps and uses its OWN nm/px (diameters are converted per image) and the sample reports the range used
(nm_per_px_min, nm_per_px_max). A set whose status.json is missing or not "done" is refused whole.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import statistics
import threading
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
CAND_CONF = 0.25      # = annotation.prediction_package.CAND_CONF (a test pins both equal): lowest "unsure" confidence
HIST_BIN_NM = 10      # diam_hist: 10 nm bins from 0 to HIST_MAX_NM, then one overflow bin
HIST_MAX_NM = 500
MULTILAYER_RULE = "geometry: outline containing another outline"
NM_TOLERANCE = 0.01   # nm/px spread (max - min) <= this x median = one scale; tests pin the other copies equal
NM_EPS = 1e-12        # float slack on that comparison: exactly 1 % is accepted (pinned equal too)
REVIEWER_CLASSES = {"EV", "MV", "inner", "nonEV"}
REVIEWER_ROLE = {"EV": "standalone", "MV": "multilayer", "inner": "inner-layer"}   # agreeing geometry role
SAMPLE_FIELDS = ["n_images", "n_reviewed", "ev_total", "ev_per_image_mean", "diam_nm_median", "diam_nm_q1",
                 "diam_nm_q3", "pct_multilayer_geometry", "inner_layers_excluded", "nonev_total",
                 "reviewed_images", "ev_reviewed_geometry", "reviewer_ev", "reviewer_mv", "reviewer_inner",
                 "reviewer_nonev", "pct_multilayer_reviewer", "reviewer_geometry_disagree", "nm_per_px_min",
                 "nm_per_px_max", "n_rejected", "n_low_conf"]
IMAGE_FIELDS = ["sample_label", "reviewed", "edited", "nm_per_px", "ev", "multilayer", "inner", "nonev",
                "rejected", "reject_reason", "note"]
IMAGES_CSV_COLUMNS = {"image", "file", "sample_label", "nm_per_px", "nm_confirmed_by"}
_TINY = 16   # below this many pixels, ask analysis.morphology whether the chain would keep the outline


# ---- the existing chain, imported once ---------------------------------------------------------------------

_CHAIN = None
_CHAIN_LOCK = threading.Lock()


def _chain():
    """(load_predictions_yolo_format, polygon_to_mask, roles_for, compute_morphology) from the counting chain,
    imported once per process under a lock.

    analysis.add_layer_roles takes its validated containment pair (0.99 / 0.75) from analysis.layer_count_stats,
    whose module body resolves the CryoAI training folder for its own CLI defaults and raises ConfigError on a
    host without one. Counting never reads that folder, so on that error the import is retried with
    $CRYOEV_CRYOAI_ROOT pointed at the repo, inside the lock and restored before the lock is released; it
    happens at most once per process (the result is cached). The thresholds stay the chain's own."""
    global _CHAIN
    with _CHAIN_LOCK:
        if _CHAIN is None:
            from training.train_yolo import load_predictions_yolo_format
            from analysis.multilayer_containment import polygon_to_mask
            from analysis.morphology import compute_morphology
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
            _CHAIN = (load_predictions_yolo_format, polygon_to_mask, roles_for, compute_morphology)
        return _CHAIN


# ---- inputs -----------------------------------------------------------------------------------------------

@dataclass
class _Obj:
    mask: np.ndarray
    cls: int
    reviewer_class: str | None = None
    low_conf: bool = False


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
        r["sample_label"] = (r["sample_label"] or "").strip()
        if not r["sample_label"]:
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


def _split_predictions(pred_dir: Path, image: str, w: int, h: int, load) -> tuple[list[_Obj], int]:
    """(outlines at conf >= CONF, number of candidates CAND_CONF <= conf < CONF) from the prediction .txt."""
    masks, confs, classes = load(prediction_file(pred_dir, image, image), w, h)   # missing file = error
    objs = [_Obj(m, k) for m, c, k in zip(masks, confs, classes) if c >= CONF]
    return objs, sum(1 for c in confs if CAND_CONF <= c < CONF)


def _objects_from_predictions(pred_dir: Path, image: str, w: int, h: int, load) -> list[_Obj]:
    return _split_predictions(pred_dir, image, w, h, load)[0]


def _objects_from_reviewed(entry: dict, image: str, w: int, h: int, to_mask) -> list[_Obj]:
    """A reviewed or edited image's outlines: every object listed (no conf floor; human outlines have conf
    null)."""
    out = []
    for n, o in enumerate(entry.get("objects") or []):
        poly = o.get("poly") or []
        if len(poly) < 3:
            raise ValueError(f"reviewed image {image} object {n}: polygon has {len(poly)} point(s)")
        rc = o.get("reviewer_class")
        if rc is not None and rc not in REVIEWER_CLASSES:
            raise ValueError(f"reviewed image {image} object {n}: unknown reviewer_class {rc!r}")
        out.append(_Obj(to_mask([float(c) for xy in poly for c in xy], w, h), int(o["cls"]), rc,
                        bool(o.get("low_conf"))))
    return out


# ---- per image --------------------------------------------------------------------------------------------

def _diameter(mask: np.ndarray, nm_per_px: float | None, compute_morphology) -> float | None:
    """analysis.morphology's equivalent_diameter for this mask, or None where analyze_instances drops it."""
    area_px = float(mask.sum())
    if area_px == 0 or (area_px < _TINY and compute_morphology(mask) is None):
        return None
    return float(np.sqrt(4 * area_px / np.pi) * (nm_per_px if nm_per_px is not None else 1.0))


def _measure(objs: list[_Obj], nm_per_px: float | None, ev_ids, roles_for, compute_morphology) -> dict:
    """Geometry counts for one image's accepted outlines (classes from `cls` only), the EV diameters (nm),
    and the reviewer-label tallies (zero unless some object carries a reviewer_class)."""
    ev_idx = [k for k, o in enumerate(objs) if ev_ids is None or o.cls in ev_ids]
    role, _ = roles_for([objs[k].mask for k in ev_idx])
    geo_role = {k: role[j] for j, k in enumerate(ev_idx)}
    ev = ml = inner = low = 0
    diam = []
    for k in ev_idx:
        d = _diameter(objs[k].mask, nm_per_px, compute_morphology)
        if d is None:
            continue
        if geo_role[k] == "inner-layer":
            inner += 1
            continue
        ev += 1
        diam.append(d)
        ml += geo_role[k] == "multilayer"
        low += bool(objs[k].low_conf)          # counted EVs only: a low-conf non-EV or inner layer is not here
    calls = [(o.reviewer_class, geo_role.get(k)) for k, o in enumerate(objs) if o.reviewer_class is not None]
    return {"ev": ev, "multilayer": ml, "inner": inner, "nonev": len(objs) - len(ev_idx), "_diam": diam, "_low_conf": low,
            "_rev_ev": sum(rc in ("EV", "MV") for rc, _ in calls),
            "_rev_mv": sum(rc == "MV" for rc, _ in calls),
            "_rev_inner": sum(rc == "inner" for rc, _ in calls),
            "_rev_nonev": sum(rc == "nonEV" and g is not None for rc, g in calls),
            "_disagree": sum((rc in REVIEWER_ROLE and g != REVIEWER_ROLE[rc]) or (rc == "nonEV" and g is not None)
                             for rc, g in calls)}


# ---- per sample -------------------------------------------------------------------------------------------

def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def diam_hist(diams) -> dict:
    """10 nm bins over 0-500 nm plus an overflow bin (500 nm and above): 51 counts."""
    counts = [0] * (HIST_MAX_NM // HIST_BIN_NM + 1)
    for d in diams:
        counts[min(int(d // HIST_BIN_NM), len(counts) - 1)] += 1
    return {"bin_nm": HIST_BIN_NM, "max_nm": HIST_MAX_NM, "counts": counts}


def _sample(imgs: list[dict], n_rejected: int = 0) -> dict:
    ev_total = sum(i["ev"] for i in imgs)
    ml = sum(i["multilayer"] for i in imgs)
    d = np.array([x for i in imgs for x in i["_diam"]], dtype=float)
    q1, med, q3 = (np.percentile(d, [25, 50, 75]) if len(d) else (None, None, None))
    rev = [i for i in imgs if i["reviewed"]]
    rev_ev, rev_mv = sum(i["_rev_ev"] for i in rev), sum(i["_rev_mv"] for i in rev)
    return {
        "n_images": len(imgs),
        "n_reviewed": len(rev),
        "ev_total": ev_total,
        "ev_per_image_mean": _r(ev_total / len(imgs)),
        "diam_nm_median": _r(med),
        "diam_nm_q1": _r(q1),
        "diam_nm_q3": _r(q3),
        "pct_multilayer_geometry": _r(100.0 * ml / ev_total) if ev_total else None,
        "inner_layers_excluded": sum(i["inner"] for i in imgs),
        "nonev_total": sum(i["nonev"] for i in imgs),
        "reviewed_images": len(rev),
        "ev_reviewed_geometry": sum(i["ev"] for i in rev),
        "reviewer_ev": rev_ev,
        "reviewer_mv": rev_mv,
        "reviewer_inner": sum(i["_rev_inner"] for i in rev),
        "reviewer_nonev": sum(i["_rev_nonev"] for i in rev),
        "pct_multilayer_reviewer": _r(100.0 * rev_mv / rev_ev) if rev_ev else None,
        "reviewer_geometry_disagree": sum(i["_disagree"] for i in rev),
        "nm_per_px_min": min(i["nm_per_px"] for i in imgs),
        "nm_per_px_max": max(i["nm_per_px"] for i in imgs),
        "n_rejected": n_rejected,
        "n_low_conf": sum(i["_low_conf"] for i in imgs),
        "n_checked": sum(1 for i in imgs if i["reviewed"] or i["edited"]),
        "n_unsure_open": None if any(i["_unsure"] is None for i in imgs) else sum(i["_unsure"] for i in imgs),
        "diam_hist": diam_hist(d),
    }


def _all_rejected(n_rejected: int) -> dict:
    """A sample whose every image was rejected: nothing is counted, so every number is null (not a crash on
    min / max / a division by zero)."""
    out = {f: None for f in SAMPLE_FIELDS}
    out.update(n_images=0, n_reviewed=0, reviewed_images=0, n_rejected=n_rejected, n_low_conf=0, n_checked=0,
               n_unsure_open=0, diam_hist=diam_hist([]), all_rejected=True)
    return out


def nm_spread(values) -> float:
    """(max - min) / median of nm/px values, one per image (0.0 for a single value)."""
    vals = [float(v) for v in values]
    return (max(vals) - min(vals)) / statistics.median(vals)


def one_scale(values) -> bool:
    """True when these nm/px values (one per image of one sample) count as one scale: spread <= NM_TOLERANCE
    (+ NM_EPS, so exactly 1 % is not lost to float rounding)."""
    return nm_spread(values) <= NM_TOLERANCE + NM_EPS


def _refusal(label: str, imgs: list[dict]) -> str | None:
    """Why this sample can't be summarised as one population, or None."""
    no_nm = [i["_image"] for i in imgs if i["nm_per_px"] is None]
    if no_nm:
        return (f"sample {label!r}: no confirmed nm_per_px for image(s) {', '.join(no_nm)} "
                f"(images.csv needs nm_per_px and nm_confirmed_by)")
    why = []
    per_image = [i["nm_per_px"] for i in imgs]
    if not one_scale(per_image):
        why.append(f"nm/px {', '.join(f'{v:g}' for v in sorted(set(per_image)))} (spread "
                   f"{100 * nm_spread(per_image):.2g} % of the median, over the {100 * NM_TOLERANCE:g} % tolerance)")
    sizes = sorted({i["_size"] for i in imgs})
    if len(sizes) > 1:
        why.append(f"tile sizes {', '.join(f'{w}x{h}' for w, h in sizes)}")
    if why:
        return f"sample {label!r} mixes {' and '.join(why)}; split it into one sample per scale"
    return None


# ---- entry points -----------------------------------------------------------------------------------------

def compute(ps_dir, reviewed_json=None) -> dict:
    """results/1 dict for the prediction_set folder `ps_dir` (contract section 4). `reviewed_json`: None (model
    predictions at conf >= 0.5), a path to reviewed.json, or the dict reviewed_polygons() returns. Slow on big
    sets: call it from a background job or thread."""
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
    status = _read_json(status_path) if status_path.is_file() else None
    if status is None or status.get("state") != "done":
        result["refused"] = ("set not finished: no status.json" if status is None else
                             f"set not finished: status {status.get('state')!r} "
                             f"({status.get('n_done')}/{status.get('n_total')} images); count when done")
        return result

    rows = _read_images_csv(ps_dir)
    if reviewed is not None:
        listed, have = {r["image"] for r in rows}, set(reviewed.get("images") or {})
        if listed != have:
            raise ValueError(f"reviewed json images differ from images.csv: missing {sorted(listed - have)}, "
                             f"extra {sorted(have - listed)}")

    load, to_mask, roles_for, compute_morphology = _chain()
    pred_dir = ps_dir / "p" / model / "predictions"
    per_image = {}
    for r in rows:
        image = r["image"]
        nm = _nm_per_px(r)
        entry = reviewed["images"][image] if reviewed is not None else None
        is_rev = bool(entry and entry.get("reviewed"))
        is_edited = bool(entry and entry.get("edited", entry.get("reviewed")))   # no key: same as reviewed
        rejected = bool(entry and entry.get("rejected"))
        note = (entry.get("note") if entry else None) or None
        try:
            w, h = _image_size(ps_dir, r["file"])
            if is_rev or is_edited:
                objs, n_cand = _objects_from_reviewed(entry, image, w, h, to_mask), None
            else:
                objs, n_cand = _split_predictions(pred_dir, image, w, h, load)
            # unsure candidates still open: Reviewed 0; edited = the package's own count (None = unknown, an old
            # reviewed.json); untouched = all candidates (the package's count when it gives one, else the .txt's)
            if is_rev:
                unsure = 0
            elif is_edited:
                unsure = entry.get("n_candidates_open")
            else:
                unsure = entry["n_candidates_open"] if entry and entry.get("n_candidates_open") is not None else n_cand
            m = _measure(objs, nm, ev_ids, roles_for, compute_morphology)
        except Exception as e:  # noqa: BLE001
            if not rejected:
                raise                       # a counted image that can't be read is a real error
            # a rejected image is not counted, so a broken file or a missing .txt must not fail the whole set:
            # it is recorded on its own row (note) and left out of every number
            w = h = None
            unsure = 0
            m = {"ev": 0, "multilayer": 0, "inner": 0, "nonev": 0, "_diam": [], "_low_conf": 0, "_rev_ev": 0,
                 "_rev_mv": 0, "_rev_inner": 0, "_rev_nonev": 0, "_disagree": 0}
            note = (note + " | " if note else "") + f"not read ({type(e).__name__}: {e})"
        if rejected:
            unsure = 0                      # a rejected image has nothing left to decide
        per_image[image] = {"sample_label": r["sample_label"], "reviewed": is_rev, "edited": is_edited,
                            "nm_per_px": nm, "rejected": rejected,
                            "reject_reason": (entry.get("reject_reason") if rejected else None),
                            "note": note,
                            **m, "_unsure": unsure, "_size": (w, h), "_image": image}

    by_sample = {}
    for img in per_image.values():
        by_sample.setdefault(img["sample_label"], []).append(img)
    refusals = []
    for label, all_imgs in by_sample.items():
        # ONE rule: rejected images leave the sample before the refusal check and every number (a rejected
        # image with a bad nm/px or a different tile size neither refuses the sample nor moves a mean)
        imgs = [i for i in all_imgs if not i["rejected"]]
        n_rej = len(all_imgs) - len(imgs)
        if not imgs:
            result["samples"][label] = _all_rejected(n_rej)
            continue
        why = _refusal(label, imgs)
        if why:
            refusals.append(why)
            # additive: a refused sample is not counted, but its open unsure outlines still count as work to do
            result.setdefault("refused_unsure", {})[label] = (
                None if any(i["_unsure"] is None for i in imgs) else sum(i["_unsure"] for i in imgs))
        else:
            result["samples"][label] = _sample(imgs, n_rej)
    result["images"] = {k: {f: v[f] for f in IMAGE_FIELDS} for k, v in per_image.items()}
    result["refused"] = "; ".join(refusals) or None
    return result


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, path)


def guard_cell(text: str) -> str:
    """CSV formula guard: a cell a spreadsheet would read as a formula (starts with = + - @, a tab or a CR) gets a
    ' prefix. Notes and sample labels are free text. (portal/modules/results.py has the same rule; a test pins
    them equal.)"""
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return guard_cell(v) if isinstance(v, str) else v


def _csv_text(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows([[_cell(v) for v in row] for row in rows])
    return buf.getvalue()


def write_results(result: dict, out_dir) -> Path:
    """per_image.csv, per_sample.csv, then results.json under `out_dir` (each replaced atomically). results.json is
    written LAST: it is the commit marker (readers trust the CSVs only next to a results.json at least as new).
    Stray *.tmp files from an interrupted earlier write are removed first."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stray in out_dir.glob("*.tmp"):
        stray.unlink(missing_ok=True)
    _atomic_write(out_dir / "per_image.csv", _csv_text(
        ["image"] + IMAGE_FIELDS, [[k] + [v[f] for f in IMAGE_FIELDS] for k, v in result["images"].items()]))
    _atomic_write(out_dir / "per_sample.csv", _csv_text(
        ["sample_label"] + SAMPLE_FIELDS, [[k] + [v[f] for f in SAMPLE_FIELDS] for k, v in result["samples"].items()]))
    _atomic_write(out_dir / "results.json", json.dumps(result, indent=1) + "\n")
    return out_dir


def compute_and_write(ps_dir, reviewed_json=None, out_dir=None) -> tuple[dict, Path]:
    """compute() then write_results() to `out_dir` (default <ps_dir>/results). Run it off the request path."""
    res = compute(ps_dir, reviewed_json)
    return res, write_results(res, out_dir or Path(ps_dir) / "results")
