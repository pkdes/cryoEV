"""
Prediction review packages (BUILD_CONTRACTS.md section 3): a user's prediction set becomes a review package for the
existing review tool, and the review comes back as reviewed.json for the results engine.

  build      <CryoAI-laptop>/predict/<ps>/ -> <review root>/pred_<ps>/  (index.html, data.js, images/, sidecars)
             - labels     = predictions at conf >= 0.50 (a stub COCO, labels.coco.json: EV = 0, non-EV = 1).
                            A label nobody touches stays present: the reviewer only removes or fixes.
             - candidates = predictions at 0.25 <= conf < 0.50, shown as unmatched predictions (u = 1), so the
                            cards offer each one for accept / reject.
             Built in a temp folder and renamed into place, so a half-built package is never served.
  reviewed   <pkg>/edits/ -> <pkg>/reviewed.json ("reviewed/1" + the section 3 amendment), through
             review_server.effective_labels.
             - An image that is edited (any decision) or marked Reviewed: its labels as the tool shows them.
             - Any other image: the model predictions at conf >= 0.50.
  spotcheck  a seeded subset to review first. representative (default): uniform per sample, proportional, >= 1 per
             sample; prioritised: weighted by 1 + the image's 0.25-0.50 candidates. Written to <pkg>/spotcheck.json.

Served by annotation/multi_review_server.py (port 8767) at /<pkg id>/.

  python -m annotation.prediction_package build --ps <CryoAI-laptop>/predict/<ps> [--review-root <dir>] [--owner <id>]
  python -m annotation.prediction_package reviewed --pkg <review root>/pred_<ps>
  python -m annotation.prediction_package spotcheck --pkg <review root>/pred_<ps> [--n 10] [--seed 0]
      [--mode representative|prioritised]
"""
import argparse
import csv
import json
import math
import random
import re
import secrets
import shutil
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
# the tool's other classes, listed in data.js categories so none is marked "(new class)" in a prediction package
EXTRA_CLASSES = ["Low conf EV", "MV", "inner", "Incomplete", "Overlapping", "VLDL"]
# reviewer label -> reviewer_class (BUILD_CONTRACTS.md section 3 amendment); any other label -> null
REVIEWER_CLASS = {"EV": "EV", "MV": "MV", "inner": "inner", "Low conf EV": "EV", "Incomplete": "EV",
                  "Overlapping": "EV", "VLDL": "nonEV", "non-EV": "nonEV"}
META = "prediction_package.json"
RECORD = "review_package.json"
PS_RE = re.compile(r"^ps_\d{8}_[0-9a-f]{6}$")
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]*$")
OWNER_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
# A portal roster id (a reviewer the hub's Assign panel can add, e.g. "j.doe"): a copy of portal/auth.py ID_RE, used
# with fullmatch. annotation never imports portal, so keep the two in step by hand.
REVIEWER_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,39}")
IMAGE_RE = re.compile(r"^\d{4,}$")                                 # server-made names: <nnnn>
FILE_RE = re.compile(r"^images/(\d{4,})\.(png|jpg|jpeg)$")
DEFAULT_SPOT = 10
META_LOCK_DIR = ".meta_locks"      # = portal/modules/reviews.py LOCK_DIR (the reviewers write)
META_LOCK_WAIT_S = 5.0
SPOT_MODES = {
    "representative": "uniform random per sample, proportional to sample size, >= 1 per sample when n >= samples; "
                      "n < samples: n samples at random, one image each",
    "prioritised": "round-robin over sample labels; weighted draw, weight 1 + candidates (0.25 <= conf < 0.50)"}
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
        m = FILE_RE.match(r.get("file") or "")
        if not IMAGE_RE.match(r.get("image") or "") or not m or m.group(1) != r["image"]:
            raise ValueError(f"{ps_id}: images.csv row image={r.get('image')!r} file={r.get('file')!r}: "
                             "expected image <nnnn>, file images/<nnnn>.png|jpg|jpeg")
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


def _has_edits(pkg: Path) -> bool:
    return (pkg / "edits").is_dir() and any((pkg / "edits").glob("*.json"))


def build(ps_dir, review_root, owner=None, width=1440, quality=72, lock_wait=META_LOCK_WAIT_S) -> Path:
    """<review_root>/pred_<ps>/ from one prediction set. Returns the package folder.

    Built in a temp folder (.building_*, never served) and renamed into place when complete, so the multi server
    never sees half a package. Rebuilding keeps the .review_key and the `reviewers` list of review_package.json
    (round 13: the Reviews hub assigns them); a package that already has saved edits is refused (its label ids are
    what the edits point at). The reviewers are read and the folder swapped under the OS lock
    <review_root>/.meta_locks/<pkg>.lock (the Reviews hub's reviewers write takes it too); a lock still held after
    lock_wait seconds raises TimeoutError and leaves the old package as it was."""
    req, ps_id, model, rows, pred_dir = _ps_inputs(ps_dir)
    owner = owner or req.get("owner")
    if not OWNER_RE.match(owner or ""):
        raise ValueError(f"bad owner {owner!r}")
    pid = package_id(ps_id)
    final = Path(review_root) / pid
    if _has_edits(final):
        raise FileExistsError(f"{final} has saved edits: not rebuilt")
    old_key = (final / ".review_key").read_text(encoding="utf-8").strip() if (final / ".review_key").is_file() else ""
    tmp = Path(review_root) / f".building_{pid}_{secrets.token_hex(4)}"
    try:
        _build_into(tmp, final, Path(ps_dir), req, ps_id, model, rows, pred_dir, owner, width, quality,
                    old_key or secrets.token_urlsafe(18))
        # the Reviews hub's reviewers POST takes the same OS lock: no assignment is lost between the read and the swap
        lock = meta_lock(review_root, pid, lock_wait)
        gone = None
        try:
            keep_reviewers(final, tmp)
            if final.exists():
                gone = Path(review_root) / f".old_{pid}_{secrets.token_hex(4)}"
                final.rename(gone)
                if _has_edits(gone):  # an edit was saved since the check above: keep that package
                    gone.rename(final)
                    raise FileExistsError(f"{final} has saved edits: not rebuilt")
                tmp.rename(final)
            else:
                tmp.rename(final)
        finally:
            lock.close()
        if gone is not None:
            shutil.rmtree(gone, ignore_errors=True)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"[OK] {pid} -> {final}")
    return final


def meta_lock(review_root, pid: str, wait: float):
    """The OS lock <review_root>/.meta_locks/<pid>.lock (data_utils.oslock, the same file as the Reviews hub's
    reviewers write) -> an open handle (close() releases it). TimeoutError when another holder keeps it > wait s."""
    from data_utils.oslock import acquire_lock
    locks = Path(review_root) / META_LOCK_DIR
    locks.mkdir(exist_ok=True)
    lk = acquire_lock(locks / f"{pid}.lock", wait=wait)
    if lk is None:
        raise TimeoutError(f"{pid}: its reviewers are being changed (lock held over {wait:g} s); not rebuilt, "
                           "try again")
    return lk


def old_reviewers(final: Path) -> list:
    """The `reviewers` of an existing package's review_package.json (portal ids only), [] when none or unreadable."""
    try:
        doc = _read_json(Path(final) / RECORD) if (Path(final) / RECORD).is_file() else {}
    except (OSError, ValueError):
        return []
    revs = doc.get("reviewers") if isinstance(doc, dict) else None
    if not isinstance(revs, list):
        return []
    return list(dict.fromkeys(r for r in revs if isinstance(r, str) and REVIEWER_ID_RE.fullmatch(r)))


def keep_reviewers(final: Path, tmp: Path) -> None:
    """Carry the old package's reviewers into the freshly built one (its review_package.json, still unserved)."""
    revs = old_reviewers(final)
    if not revs:
        return
    rec = _read_json(Path(tmp) / RECORD)
    rec["reviewers"] = revs
    _write_json(Path(tmp) / RECORD, rec)


def _build_into(out, final, ps_dir, req, ps_id, model, rows, pred_dir, owner, width, quality, key):
    (out / "images").mkdir(parents=True)
    print(f"[steps] labels = predictions at conf >= {LABEL_CONF:.2f}; candidates = {CAND_CONF:.2f} <= conf < "
          f"{LABEL_CONF:.2f} (u = 1); images grayscale JPEG <= {width} px, q{quality}; no flag matching")

    coco = {"info": {"description": f"stub from prediction set {ps_id}, model {model}, conf >= {LABEL_CONF}"},
            "categories": [{"id": k, "name": n} for k, n in enumerate(PRED_CLASSES)], "images": [], "annotations": []}
    meta_images, images, ann_id = {}, [], 0
    for n, r in enumerate(rows, 1):
        image = r["image"]
        src = ps_dir / r["file"]  # images/<nnnn>.<ext>, checked by _ps_inputs (a link is fine: "copied or linked")
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
    pid = final.name
    data = {"id": pid, "kind": "prediction", "prediction_set": ps_id, "model": model, "owner": owner,
            "source": f"prediction_set:{ps_id}", "run": model, "created": built, "cand_conf": CAND_CONF,
            "label_conf": LABEL_CONF,
            "categories": PRED_CLASSES + EXTRA_CLASSES, "images": images}
    brp.write_packages(out, data, part_size=0, zip_it=False)
    _write_json(out / "labels.coco.json", coco, indent=None)
    _write_json(out / META, {"schema": "prediction_package/1", "package": pid, "prediction_set": ps_id,
                             "model": model, "owner": owner, "label_conf": LABEL_CONF, "cand_conf": CAND_CONF,
                             "built": built, "ps_dir": str(ps_dir.resolve()), "images": meta_images})
    (out / ".review_key").write_text(key, encoding="utf-8")
    _write_json(out / RECORD, registry_record(final, data, built, str(pred_dir.resolve())))  # last: "complete"
    print(f"[OK] {pid}: {len(images)} images, {sum(len(i['gt']) for i in images)} labels, "
          f"{sum(len(i['pr']) for i in images)} candidates")


def registry_record(pkg: Path, data: dict, built: str, predictions_dir=None) -> dict:
    """The registry review_package line for this package (registry/schemas/review_package.json)."""
    return {"schema": "review_package/1", "id": data["id"], "logged": built, "recorded_by": TOOL,
            "path": str(Path(pkg).resolve()), "kind": "prediction", "parent": None,
            "base": data["prediction_set"], "base_kind": "prediction_set",
            "model": data["model"], "run": None, "predictions_dir": predictions_dir, "flag_conf": LABEL_CONF,
            "n_images": len(data["images"]), "built": built, "tool": TOOL, "status": "built",
            "owner": data["owner"], "prediction_set": data["prediction_set"]}


# ---------------------------------------------------------------- reviewed.json
def is_edited(edit) -> bool:
    """Any saved object decision (g / p / n, and t:<objkey> low-conf tags: without them a tag would never reach the
    results), or an edited label list (the Editor's gt + gtT). The image-level i:note and i:reject don't count: a
    note or a rejection alone leaves the image's objects on the model path (rejection is read by is_rejected)."""
    e = rs._migrate(edit) if edit else {"dec": {}}
    return any(not k.startswith("i:") for k in e["dec"]) or (bool(e.get("gtT")) and e.get("gt") is not None)


def is_rejected(edit) -> bool:
    """The image is rejected (i:reject set): it drops out of results; no training effect."""
    return bool(rs.image_flags(edit).get("rejected"))


def _edit_record(pkg: Path, file: str):
    p = Path(pkg) / "edits" / f"{file}.json"
    return _read_json(p).get("edit") if p.is_file() else None


def _rejected_file(pkg: Path, file) -> bool:
    """is_rejected for one package image; an unreadable record counts as not rejected (one bad file never stops a
    spot check)."""
    if not file:
        return False
    try:
        return is_rejected(_edit_record(pkg, file))
    except (OSError, ValueError, AttributeError, TypeError):
        return False


def _ring_px(rings, w, h):
    """The object's outline in pixels: its largest ring (prediction and drawn outlines have one)."""
    def area(r):
        x, y = r[0::2], r[1::2]
        return abs(sum(x[i] * y[i - 1] - x[i - 1] * y[i] for i in range(len(x))))
    r = max(rings, key=area)
    return [[round(r[i] * w, 3), round(r[i + 1] * h, 3)] for i in range(0, len(r), 2)]


def candidates_open(n_candidates: int, edit) -> int:
    """Candidates (data.js pr items, decision keys "p<k>") with neither an accept ("add:<cls>") nor a reject
    ("no") decision in this edit record. A decision with an empty v was undone: the candidate is open again."""
    dec = rs._migrate(edit)["dec"] if edit else {}
    return sum(1 for k in range(n_candidates) if not (dec.get(f"p{k}") or {}).get("v"))


def reviewed_polygons(pkg_dir, write=True) -> dict:
    """reviewed/1 for one prediction package (BUILD_CONTRACTS.md section 3 + amendment), also written to
    <pkg>/reviewed.json.

    Per image: reviewed = the explicit Reviewed flag; edited = any saved decision or edited label list.
    edited or reviewed: the labels as the tool shows them (review_server.effective_labels), each object with
        reviewer_label  the class the object stands at on this image: the reviewer's pick, or the model's class
                        name for an object nobody changed
        reviewer_class  REVIEWER_CLASS[reviewer_label]; null for any other label (counted in unknown_labels)
        cls             0 for reviewer_class EV / MV / inner, 1 for nonEV; with reviewer_class null, the model
                        object's class (0 for a drawn outline)
        source          "model" for an outline as the model drew it; "human" when drawn or reshaped by a reviewer
        conf            the model's confidence for a label or accepted candidate (reshaped too); null when drawn
    reviewed_by = authors of decisions: the who of the image's decisions. The tool does not record who pressed
    Reviewed, so an image marked Reviewed with no decisions has [].
    neither: the model predictions at conf >= 0.50, source "model", reviewer_label / reviewer_class null.
    Every image also has n_candidates_open (additive to reviewed/1): its candidates (cand_conf <= conf < label_conf,
    data.js pr items with u = 1) with no accept and no reject decision; all of them for an untouched image.
    Round 7, additive and each key only when set (readers use .get): per image rejected (true) + reject_reason
    (i:reject; such an image has n_candidates_open 0) and note (i:note); per object (edited / reviewed images)
    low_conf (true: its t: tag, or the Low conf EV class, review_server.is_low_conf) and note (its t: note).
    Round 12, additive: per image reject_scope ("sample" | "montage") when a group action rejected it; top level
    sample_groups (sample_groups()) when the package has group notes or group rejects."""
    pkg = Path(pkg_dir)
    meta, data = _read_json(pkg / META), read_datajs(pkg)
    by_file = {i["file"]: i for i in data["images"]}
    out = {"schema": "reviewed/1", "package": meta["package"], "prediction_set": meta["prediction_set"],
           "model": meta["model"], "unknown_labels": 0, "images": {}}
    for image, info in meta["images"].items():
        im, w, h = by_file[info["file"]], info["w"], info["h"]
        lab = {g["id"]: g for g in im["gt"]}
        lab_meta = {m["id"]: m for m in info["labels"]}
        edit = _edit_record(pkg, info["file"])
        reviewed, edited = bool(edit and edit.get("reviewed")), is_edited(edit)
        flags = rs.image_flags(edit)
        if not (reviewed or edited):
            out["images"][image] = _image_flags({"reviewed": False, "edited": False, "reviewed_by": [],
                                                 "n_candidates_open": len(im["pr"]), "objects": [
                {"cls": m["cls"], "poly": _ring_px(lab[m["id"]]["p"], w, h), "conf": m["conf"], "source": "model",
                 "reviewer_label": None, "reviewer_class": None} for m in info["labels"]]}, flags)
            continue
        dec = rs._migrate(edit)["dec"]
        objs = []
        for g in rs.effective_labels(im, edit):
            src, model_cls, c_conf, orig = g.get("src") or "", None, None, None
            if g.get("id") is not None and g["id"] in lab_meta:
                model_cls, c_conf, orig = lab_meta[g["id"]]["cls"], lab_meta[g["id"]]["conf"], lab[g["id"]]["p"]
            elif src.startswith("p") and src[1:].isdigit() and int(src[1:]) < len(im["pr"]):
                k = int(src[1:])
                model_cls, c_conf, orig = info["candidates"][k]["cls"], info["candidates"][k]["conf"], im["pr"][k]["p"]
            rc = REVIEWER_CLASS.get(g["c"])
            if rc is None:
                out["unknown_labels"] += 1
            obj = {"cls": (1 if rc == "nonEV" else 0) if rc else (model_cls if model_cls is not None else 0),
                   "poly": _ring_px(g["p"], w, h), "conf": c_conf,
                   "source": "model" if orig is not None and g["p"] == orig else "human",
                   "reviewer_label": g["c"], "reviewer_class": rc}
            tag_v, note = rs.tag_of(dec, rs.object_key(g))
            if rs.is_low_conf(g["c"], tag_v):
                obj["low_conf"] = True
            if note:
                obj["note"] = note
            objs.append(obj)
        who = sorted({d.get("who") for d in dec.values() if d.get("who")})
        out["images"][image] = _image_flags({"reviewed": reviewed, "edited": edited, "reviewed_by": who,
                                             "n_candidates_open": candidates_open(len(im["pr"]), edit),
                                             "objects": objs}, flags)
    sg = sample_groups(pkg, data, meta)
    if sg:
        out["sample_groups"] = sg
    if write:
        _write_json(pkg / "reviewed.json", out)
    return out


def sample_groups(pkg: Path, data: dict, meta: dict) -> dict:
    """Round 12, additive: the package's group notes and group rejects (edits/_groups.json, annotation/review_groups.py)
    by real sample label, {} without any: {<sample_label>: {note?, reject?, montages? {<montage>: {note?, reject?}}}}.
    A group is keyed by data.js split (what the reviewer picked); its images' sample labels say where it goes. The
    group reject itself is on every image already (rejected / reject_scope): this only carries the group's words."""
    from annotation import review_groups as rg
    try:
        doc = rg.load(Path(pkg) / "edits")
    except rg.GroupsUnreadable:
        return {}
    split_of = {i["file"]: i.get("split") for i in data["images"]}
    labels = {}
    for m in meta["images"].values():
        labels.setdefault(split_of.get(m.get("file")), set()).add(m.get("sample_label"))
    out = {}
    for gid, g in sorted(doc.get("groups", {}).items()):
        parsed = rg.parse_group_id(gid)
        if not parsed or not isinstance(g, dict):
            continue
        kind, split, montage = parsed
        words = {k: g[k] for k in ("note", "reject") if isinstance(g.get(k), dict)}
        if not words:
            continue
        for label in sorted(x for x in labels.get(split, ()) if isinstance(x, str)):
            dst = out.setdefault(label, {})
            if kind == "sample":
                dst.update(words)
            else:
                dst.setdefault("montages", {})[montage] = words
    return out


def _image_flags(entry: dict, flags: dict) -> dict:
    """One reviewed/1 image entry plus its image-level state (rs.image_flags), keys only when set, after the
    existing keys. A rejected image has no open candidates: nobody needs to decide them."""
    if flags.get("rejected"):
        entry["n_candidates_open"] = 0
        entry["rejected"], entry["reject_reason"] = True, flags["reject_reason"]
        if flags.get("reject_scope"):  # round 12: rejected by a sample / montage group action
            entry["reject_scope"] = flags["reject_scope"]
    if flags.get("note"):
        entry["note"] = flags["note"]
    return entry


# ---------------------------------------------------------------- spot check
def _spot_rows(src: Path):
    """[(image, sample_label, n_candidates, n_labels)] from a package (its sidecar) or a prediction set. A package's
    rejected images (i:reject) are left out: they drop out of results, so there is nothing to spot-check."""
    if (src / META).is_file():
        meta = _read_json(src / META)
        return [(i, m["sample_label"], len(m["candidates"]), len(m["labels"])) for i, m in meta["images"].items()
                if not _rejected_file(src, m.get("file"))], meta
    _, ps_id, model, rows, pred_dir = _ps_inputs(src)
    out = []
    for r in rows:
        preds = _predictions(pred_dir, r["image"])
        out.append((r["image"], r.get("sample_label"), sum(1 for p in preds if CAND_CONF <= p["sr"] < LABEL_CONF),
                    sum(1 for p in preds if p["sr"] >= LABEL_CONF)))
    return out, {"package": None, "prediction_set": ps_id, "model": model}


def _quotas(n, sizes, rng):
    """n split over samples in proportion to their sizes: >= 1 each (n >= the number of samples), never above a
    size. Ties go to a seeded random order of the samples."""
    total, order = sum(sizes.values()), list(sizes)
    rng.shuffle(order)
    exact = {s: n * sizes[s] / total for s in order}
    q = {s: min(sizes[s], max(1, math.floor(exact[s]))) for s in order}
    while sum(q.values()) > n:
        s = max((s for s in order if q[s] > 1), key=lambda s: q[s] - exact[s])
        q[s] -= 1
    while sum(q.values()) < n:
        s = max((s for s in order if q[s] < sizes[s]), key=lambda s: exact[s] - q[s])
        q[s] += 1
    return q


def spot_check(ps_dir_or_pkg, n=DEFAULT_SPOT, seed=0, write=True, mode="representative") -> list:
    """A subset of image ids to review first, deterministic for a seed (SPOT_MODES).

    representative (default): a uniform random draw per sample; n split in proportion to sample size, >= 1 per
        sample when n >= the number of samples; when n < samples, n samples chosen at random, one image each.
    prioritised: round-robin over the sample labels; within a sample a seeded weighted draw without replacement
        (Efraimidis-Spirakis), weight 1 + the image's number of candidates (0.25 <= conf < 0.50).
    Given a package, also writes <pkg>/spotcheck.json, labelled with the mode and its rule."""
    if mode not in SPOT_MODES:
        raise ValueError(f"mode must be one of {sorted(SPOT_MODES)}")
    src = Path(ps_dir_or_pkg)
    rows, meta = _spot_rows(src)
    rng = random.Random(seed)
    n = min(max(0, int(n)), len(rows))
    by_sample = {}
    for row in sorted(rows):
        by_sample.setdefault(row[1], []).append(row)
    samples = sorted(by_sample, key=str)
    picked = []
    if mode == "representative":
        if n < len(samples):
            picked = [rng.choice(by_sample[s]) for s in rng.sample(samples, n)]
        elif n:
            q = _quotas(n, {s: len(by_sample[s]) for s in samples}, rng)
            for s in samples:
                picked += rng.sample(by_sample[s], q[s])
    else:
        queues = {}
        for s in samples:
            keyed = [(rng.random() ** (1.0 / (1 + row[2])), row) for row in by_sample[s]]
            queues[s] = [r for _, r in sorted(keyed, key=lambda kv: -kv[0])]
        while len(picked) < n:
            for s in samples:
                if queues[s] and len(picked) < n:
                    picked.append(queues[s].pop(0))
    if write and (src / META).is_file():
        _write_json(src / "spotcheck.json", {
            "schema": "spotcheck/1", "package": meta["package"], "prediction_set": meta["prediction_set"],
            "mode": mode, "rule": SPOT_MODES[mode], "seed": seed, "n": len(picked), "made": _now(),
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
    s.add_argument("--mode", choices=sorted(SPOT_MODES), default="representative")
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
        n_ed = sum(1 for v in d["images"].values() if v["edited"])
        print(f"[OK] {a.pkg / 'reviewed.json'}: {n_rev} of {len(d['images'])} images reviewed, {n_ed} edited, "
              f"{d['unknown_labels']} unknown labels, {sum(len(v['objects']) for v in d['images'].values())} objects")
    else:
        ids = spot_check(a.pkg or a.ps, a.n, a.seed, mode=a.mode)
        print(json.dumps(ids))


if __name__ == "__main__":
    main()
