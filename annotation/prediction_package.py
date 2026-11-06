"""
Prediction review packages (BUILD_CONTRACTS.md section 3): a user's prediction set becomes a review package for the
existing review tool, and the review comes back as reviewed.json for the results engine.

  build      <CryoAI-laptop>/predict/<ps>/ -> <review root>/pred_<ps>/  (index.html, data.js, images/, sidecars)
             - labels     = predictions at conf >= 0.50 (a stub COCO, labels.coco.json: EV = 0, non-EV = 1).
                            A label nobody touches stays present: the reviewer only removes or fixes.
             - candidates = predictions at 0.25 <= conf < 0.50, shown as unmatched predictions (u = 1), so the
                            cards offer each one for accept / reject.
  reviewed   <pkg>/edits/ -> <pkg>/reviewed.json ("reviewed/1"), through review_server.effective_labels.
             - An image marked reviewed: its labels as the tool shows them.
             - Any other image: the model predictions at conf >= 0.50, edits or not (contract section 3).
  spotcheck  a seeded, representative subset to review first: round-robin over the sample labels, each pick
             weighted by 1 + its number of 0.25-0.50 candidates. Written to <pkg>/spotcheck.json.

Served by annotation/multi_review_server.py (port 8767) at /<pkg id>/.

  python -m annotation.prediction_package build --ps <CryoAI-laptop>/predict/<ps> [--review-root <dir>] [--owner <id>]
  python -m annotation.prediction_package reviewed --pkg <review root>/pred_<ps>
  python -m annotation.prediction_package spotcheck --pkg <review root>/pred_<ps> [--n 10] [--seed 0]
"""
import argparse
import csv
import json
import math
import random
import re
import secrets
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from annotation import build_review_package as brp  # noqa: E402
from annotation import review_server as rs  # noqa: E402

LABEL_CONF = 0.50          # predictions at or above: the package's labels
CAND_CONF = 0.25           # predictions at or above (and below LABEL_CONF): candidates to accept / reject
PRED_CLASSES = ["EV", "non-EV"]          # model class id -> name (v5_evnonev models)
EXTRA_CLASSES = ["MV", "inner"]          # offered in the tool without the "new class" mark
REVIEWER_CLASS = {"EV": "EV", "MV": "MV", "inner": "inner", "non-EV": "nonEV"}
META = "prediction_package.json"
RECORD = "review_package.json"
PS_RE = re.compile(r"^ps_\d{8}_[0-9a-f]{6}$")
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]*$")
OWNER_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
DEFAULT_SPOT = 10
TOOL = "annotation/prediction_package.py"


def package_id(ps_id: str) -> str:
    return f"pred_{ps_id}"


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _write_json(p: Path, obj, indent=1):
    tmp = Path(p).with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=indent) + "\n", encoding="utf-8")
    tmp.replace(p)


def read_datajs(pkg: Path) -> dict:
    return json.loads((Path(pkg) / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))


def _display_label(s: str) -> str:
    """The sample label as data.js `split`: the page puts split into HTML unescaped, and a label can come from an
    upload, so only [A-Za-z0-9 _.-] survive (the real label stays in the sidecar and reviewed/results)."""
    return re.sub(r"[^A-Za-z0-9 _.-]", "_", str(s or ""))[:40] or "-"


def _ps_inputs(ps_dir: Path):
    """request.json, images.csv rows and the predictions folder of one prediction set, checked."""
    ps_dir = Path(ps_dir)
    req = _read_json(ps_dir / "request.json")
    ps_id, model = req.get("ps") or ps_dir.name, req.get("model") or ""
    if not PS_RE.match(ps_id):
        raise ValueError(f"bad prediction set id {ps_id!r}")
    if not MODEL_RE.match(model):
        raise ValueError(f"bad model id {model!r}")
    st = ps_dir / "status.json"
    if st.exists() and _read_json(st).get("state") != "done":
        raise ValueError(f"{ps_id}: status is {_read_json(st).get('state')!r}, not done")
    with open(ps_dir / "images.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if not IMAGE_RE.match(r.get("image") or ""):
            raise ValueError(f"{ps_id}: bad image name {r.get('image')!r} in images.csv")
    return req, ps_id, model, rows, ps_dir / "p" / model / "predictions"


def _predictions(pred_dir: Path, image: str):
    """One image's predictions as build_review_package.load_predictions gives them ({s, sr, c, k, p}).
    A missing file is an error: predict_models writes one per image, empty when nothing was found."""
    path = pred_dir / f"{image}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"no predictions file for image {image}: {path}")
    out = [p for p in brp.load_predictions(path, PRED_CLASSES) if len(p["p"][0]) >= 6]
    for p in out:
        if p["k"] not in (0, 1):
            raise ValueError(f"{path}: class {p['k']} is not EV (0) / non-EV (1)")
    return out


def _flat_px(ring, w, h):
    return [round(v * (w if i % 2 == 0 else h), 3) for i, v in enumerate(ring)]


def build(ps_dir, review_root, owner=None, width=1440, quality=72) -> Path:
    """<review_root>/pred_<ps>/ from one prediction set. Returns the package folder.

    Rebuilding a package keeps its .review_key; a package that already has saved edits is refused (its label ids
    are what the edits point at)."""
    req, ps_id, model, rows, pred_dir = _ps_inputs(ps_dir)
    ps_dir = Path(ps_dir)
    owner = owner or req.get("owner")
    if not OWNER_RE.match(owner or ""):
        raise ValueError(f"bad owner {owner!r}")
    pid = package_id(ps_id)
    out = Path(review_root) / pid
    if (out / "edits").is_dir() and any((out / "edits").glob("*.json")):
        raise FileExistsError(f"{out} has saved edits: not rebuilt")
    (out / "images").mkdir(parents=True, exist_ok=True)
    print(f"[steps] labels = predictions at conf >= {LABEL_CONF:.2f}; candidates = {CAND_CONF:.2f} <= conf < "
          f"{LABEL_CONF:.2f} (u = 1); images grayscale JPEG <= {width} px, q{quality}; no flag matching")

    coco = {"info": {"description": f"stub from prediction set {ps_id}, model {model}, conf >= {LABEL_CONF}"},
            "categories": [{"id": k, "name": n} for k, n in enumerate(PRED_CLASSES)], "images": [], "annotations": []}
    meta_images, images, ann_id = {}, [], 0
    for n, r in enumerate(rows, 1):
        image = r["image"]
        src = (ps_dir / (r.get("file") or f"images/{image}.png")).resolve()
        if src.parent != (ps_dir / "images").resolve():
            raise ValueError(f"{ps_id}: image {image} is not in images/: {r.get('file')!r}")
        with Image.open(src) as im0:
            w, h = im0.size
            im = im0.convert("L")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        jpg = f"images/{image}.jpg"
        im.save(out / jpg, "JPEG", quality=quality)
        preds = _predictions(pred_dir, image)
        labels = [p for p in preds if p["sr"] >= LABEL_CONF]
        cands = [p for p in preds if CAND_CONF <= p["sr"] < LABEL_CONF]

        coco["images"].append({"id": n, "file_name": Path(src).name, "width": w, "height": h})
        anns, lab_meta = [], []
        for p in labels:
            ann_id += 1
            seg = _flat_px(p["p"][0], w, h)
            xs, ys = seg[0::2], seg[1::2]
            anns.append({"id": ann_id, "image_id": n, "category_id": p["k"], "segmentation": [seg], "iscrowd": 0,
                         "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], "score": p["sr"]})
            lab_meta.append({"id": ann_id, "cls": p["k"], "conf": p["sr"]})
        coco["annotations"] += anns
        gt = brp.gt_items(anns, dict(enumerate(PRED_CLASSES)), w, h)
        pr = [{"s": p["s"], "c": p["c"], "p": p["p"], "u": 1} for p in cands]
        file = Path(src).name
        images.append({"file": file, "split": _display_label(r.get("sample_label")), "w": w, "h": h, "src": jpg,
                       "gt": gt, "pr": pr, "fg": 0, "fp": len(pr)})
        meta_images[image] = {"file": file, "w": w, "h": h, "sample_label": r.get("sample_label"),
                              "labels": lab_meta, "candidates": [{"k": k, "cls": p["k"], "conf": p["sr"]}
                                                                 for k, p in enumerate(cands)]}

    built = _now()
    data = {"id": pid, "kind": "prediction", "prediction_set": ps_id, "model": model, "owner": owner,
            "source": f"prediction_set:{ps_id}", "run": model, "created": built,
            "categories": PRED_CLASSES + EXTRA_CLASSES, "images": images}
    brp.write_packages(out, data, part_size=0, zip_it=False)
    _write_json(out / "labels.coco.json", coco, indent=None)
    _write_json(out / META, {"schema": "prediction_package/1", "package": pid, "prediction_set": ps_id,
                             "model": model, "owner": owner, "label_conf": LABEL_CONF, "cand_conf": CAND_CONF,
                             "built": built, "ps_dir": str(ps_dir.resolve()), "images": meta_images})
    _write_json(out / RECORD, registry_record(out, data, built, str(pred_dir.resolve())))
    key = out / ".review_key"
    if not key.exists() or not key.read_text(encoding="utf-8").strip():
        key.write_text(secrets.token_urlsafe(18), encoding="utf-8")
    print(f"[OK] {pid}: {len(images)} images, {sum(len(i['gt']) for i in images)} labels, "
          f"{sum(len(i['pr']) for i in images)} candidates -> {out}")
    return out


def registry_record(pkg: Path, data: dict, built: str, predictions_dir=None) -> dict:
    """The registry review_package line for this package (registry/schemas/review_package.json)."""
    return {"schema": "review_package/1", "id": data["id"], "logged": built, "recorded_by": TOOL,
            "path": str(Path(pkg).resolve()), "kind": "prediction", "parent": None,
            "base": data["prediction_set"], "base_kind": "prediction_set",
            "model": data["model"], "run": None, "predictions_dir": predictions_dir, "flag_conf": LABEL_CONF,
            "n_images": len(data["images"]), "built": built, "tool": TOOL, "status": "built",
            "owner": data["owner"], "prediction_set": data["prediction_set"]}


# ---------------------------------------------------------------- reviewed.json
def _class_map():
    g = brp.load_class_groups()["label_group"]
    return lambda c: 0 if g.get(c) in ("ev_type", "low_conf") else 1 if g.get(c) == "non_ev_type" else None


def _ring_px(rings, w, h):
    """The object's outline in pixels: its largest ring (prediction and drawn outlines have one)."""
    def area(r):
        x, y = r[0::2], r[1::2]
        return abs(sum(x[i] * y[i - 1] - x[i - 1] * y[i] for i in range(len(x))))
    r = max(rings, key=area)
    return [[round(r[i] * w, 3), round(r[i + 1] * h, 3)] for i in range(0, len(r), 2)]


def reviewed_polygons(pkg_dir, write=True) -> dict:
    """reviewed/1 for one prediction package (BUILD_CONTRACTS.md section 3), also written to <pkg>/reviewed.json.

    reviewed image: effective_labels(image, its edit record), each object with
        cls             0 for the EV-type classes (EV, MV, inner, Incomplete, Overlapping, Low conf EV), 1 for
                        non-EV / VLDL (annotation/class_groups.yaml); any other class keeps the model's class
        reviewer_class  EV / MV / inner / nonEV from the class the image stands at, else null
        source          "model" for an outline as the model drew it; "human" when drawn or reshaped by a reviewer
        conf            the model's confidence for a label or accepted candidate (reshaped too); null when drawn
    reviewed_by = the reviewers (who) of the image's decisions: the tool does not record who pressed Reviewed.
    any other image: the model predictions at conf >= 0.50, source "model", reviewer_class null."""
    pkg = Path(pkg_dir)
    meta, data = _read_json(pkg / META), read_datajs(pkg)
    by_file = {i["file"]: i for i in data["images"]}
    cls_of = _class_map()
    out = {"schema": "reviewed/1", "package": meta["package"], "prediction_set": meta["prediction_set"],
           "model": meta["model"], "images": {}}
    for image, info in meta["images"].items():
        im, w, h = by_file[info["file"]], info["w"], info["h"]
        lab = {g["id"]: g for g in im["gt"]}
        conf = {m["id"]: m["conf"] for m in info["labels"]}
        rec_path = pkg / "edits" / f"{info['file']}.json"
        edit = _read_json(rec_path).get("edit") if rec_path.is_file() else None
        if not (edit and edit.get("reviewed")):
            out["images"][image] = {"reviewed": False, "reviewed_by": [], "objects": [
                {"cls": m["cls"], "poly": _ring_px(lab[m["id"]]["p"], w, h), "conf": m["conf"], "source": "model",
                 "reviewer_class": None} for m in info["labels"]]}
            continue
        objs = []
        for g in rs.effective_labels(im, edit):
            src, model_cls, c_conf, orig = g.get("src") or "", None, None, None
            if g.get("id") is not None and g["id"] in lab:
                model_cls = next(m["cls"] for m in info["labels"] if m["id"] == g["id"])
                c_conf, orig = conf[g["id"]], lab[g["id"]]["p"]
            elif src.startswith("p") and src[1:].isdigit() and int(src[1:]) < len(im["pr"]):
                k = int(src[1:])
                model_cls, c_conf, orig = info["candidates"][k]["cls"], info["candidates"][k]["conf"], im["pr"][k]["p"]
            cls = cls_of(g["c"])
            objs.append({"cls": cls if cls is not None else (model_cls if model_cls is not None else 0),
                         "poly": _ring_px(g["p"], w, h), "conf": c_conf,
                         "source": "model" if orig is not None and g["p"] == orig else "human",
                         "reviewer_class": REVIEWER_CLASS.get(g["c"])})
        who = sorted({d.get("who") for d in rs._migrate(edit)["dec"].values() if d.get("who")})
        out["images"][image] = {"reviewed": True, "reviewed_by": who, "objects": objs}
    if write:
        _write_json(pkg / "reviewed.json", out)
    return out


# ---------------------------------------------------------------- spot check
def _spot_rows(src: Path):
    """[(image, sample_label, n_candidates, n_labels)] from a package (its sidecar) or a prediction set."""
    if (src / META).is_file():
        meta = _read_json(src / META)
        return [(i, m["sample_label"], len(m["candidates"]), len(m["labels"])) for i, m in meta["images"].items()], meta
    _, ps_id, model, rows, pred_dir = _ps_inputs(src)
    out = []
    for r in rows:
        preds = _predictions(pred_dir, r["image"])
        out.append((r["image"], r.get("sample_label"), sum(1 for p in preds if CAND_CONF <= p["sr"] < LABEL_CONF),
                    sum(1 for p in preds if p["sr"] >= LABEL_CONF)))
    return out, {"package": None, "prediction_set": ps_id, "model": model}


def spot_check(ps_dir_or_pkg, n=DEFAULT_SPOT, seed=0, write=True) -> list:
    """A representative subset of image ids to review first, deterministic for a seed.

    Round-robin over the sample labels (sorted), so every sample is covered once n >= the number of samples; within
    a sample, a seeded weighted draw without replacement (Efraimidis-Spirakis), weight 1 + the image's number of
    candidates (0.25 <= conf < 0.50), so images with many uncertain objects come up more often. Given a package,
    also writes <pkg>/spotcheck.json."""
    src = Path(ps_dir_or_pkg)
    rows, meta = _spot_rows(src)
    rng = random.Random(seed)
    by_sample = {}
    for row in sorted(rows):
        key = rng.random() ** (1.0 / (1 + row[2]))
        by_sample.setdefault(row[1], []).append((key, row))
    queues = {s: [r for _, r in sorted(v, key=lambda kv: -kv[0])] for s, v in by_sample.items()}
    picked, n = [], min(max(0, int(n)), len(rows))
    while len(picked) < n:
        for s in sorted(queues, key=str):
            if queues[s] and len(picked) < n:
                picked.append(queues[s].pop(0))
    if write and (src / META).is_file():
        _write_json(src / "spotcheck.json", {
            "schema": "spotcheck/1", "package": meta["package"], "prediction_set": meta["prediction_set"],
            "seed": seed, "n": len(picked), "made": _now(),
            "rule": "round-robin over sample labels; weighted draw, weight 1 + candidates (0.25 <= conf < 0.50)",
            "images": [{"image": i, "sample_label": s, "n_candidates": c, "n_labels": l} for i, s, c, l in picked]})
    return [r[0] for r in picked]


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="prediction set -> review package")
    b.add_argument("--ps", required=True, type=Path, help="<CryoAI-laptop>/predict/<ps_id>")
    b.add_argument("--review-root", type=Path, default=None, help="default: <CryoAI-laptop>/review")
    b.add_argument("--owner", default=None, help="portal person id (default: request.json owner)")
    r = sub.add_parser("reviewed", help="package edits -> <pkg>/reviewed.json")
    r.add_argument("--pkg", required=True, type=Path)
    s = sub.add_parser("spotcheck", help="suggest images to review first")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--pkg", type=Path)
    g.add_argument("--ps", type=Path)
    s.add_argument("--n", type=int, default=DEFAULT_SPOT)
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        if a.review_root is None:
            from data_utils import config
            a.review_root = config.laptop_root() / "review"
        pkg = build(a.ps, a.review_root, a.owner)
        print(f"[OK] serve: python -m annotation.multi_review_server --root \"{Path(a.review_root)}\"; "
              f"open :8767/{pkg.name}/?k=<key in {pkg / '.review_key'}>")
    elif a.cmd == "reviewed":
        d = reviewed_polygons(a.pkg)
        n_rev = sum(1 for v in d["images"].values() if v["reviewed"])
        print(f"[OK] {a.pkg / 'reviewed.json'}: {n_rev} of {len(d['images'])} images reviewed, "
              f"{sum(len(v['objects']) for v in d['images'].values())} objects")
    else:
        ids = spot_check(a.pkg or a.ps, a.n, a.seed)
        print(json.dumps(ids))


if __name__ == "__main__":
    main()
