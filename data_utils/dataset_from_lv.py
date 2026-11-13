"""
Build a YOLO training dataset from a reviewed label version (`<CryoAI-laptop>/labels/lv_*/labels.coco.json`),
keeping the train/valid/test split of an existing Roboflow export.

    python data_utils/dataset_from_lv.py --lv lv_20261030_0909 \
        --split-from "roboflow - 20261028 cryoai v5 S1" --output-name lv_20261030_0909_v5S1_evnonev \
        --class-scheme ev_nonev --export-only drop [--register]

What it writes (`<CryoAI>/training outputs/<output-name>/`, or under --output-root; refuses an existing folder):
  images/<split>/, labels/<split>/, dataset.yaml   exactly what prepare_all_layer_singleclass.py writes for
                                                   the export: same file names (the export's), same class
                                                   mapping (--class-scheme), same label formatting. The
                                                   conversion IS prepare's convert_split(), fed a per-split
                                                   COCO whose outlines come from the lv.
  provenance.json   {"dataset_record": the dataset/1 registry record, "details": names, warnings}
  image_list.tsv    `stem\\tsplit` lines, the same text as <registry>/lists/<dataset>.tsv

Split: by image identity (registry/ids.image_key of the lv image's `extra.name` / file name vs the export file
name), never re-split. An image keeps the export split folder it sits in (train / valid -> val / test). If the
export has a split_assignment.csv, its `split` column must agree with the folders.
  - lv images not in the export are reported (with the split file's reason, e.g. DROPPED-duplicate, and any
    review edits they had in changes.jsonl) and left out.
  - export images not in the lv: the build refuses unless --export-only says what to do (no default):
    `drop` leaves them out; `export-labels` keeps them with the export's own labels (then an unedited lv gives
    exactly prepare's dataset). The choice is in provenance.json and the record's note.
Pixels come from the export's image files; outlines are normalized by the lv image's width/height. A resized
export copy (same aspect ratio) is reported; another aspect ratio (padded / cropped) is refused. Every planned
image must be copied; the list, counts and hashes come from the files written.

Registry: the dataset/1 record is printed. --register (only for an lv_* with version.json labels_sha256; the id is
checked before building) appends it (registry.append, schema-checked) and writes <registry>/lists/<output-name>.tsv.
--register-existing FOLDER registers a folder built earlier. Without them nothing outside the dataset folder is
written. registry/importers.py keeps such a dataset's own record (provenance.json) and only derives its status.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REG_DIR = REPO / "registry"
sys.path.insert(0, str(REPO))
from data_utils import config  # noqa: E402
from data_utils import prepare_all_layer_singleclass as prep  # noqa: E402
from data_utils.path_guard import protected_problem  # noqa: E402

LV_RE = re.compile(r"^lv_\d{8}_\d{4}$")
DATASET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")    # registry/ids.py "dataset"
SPLIT_ORDER = {"train": 0, "valid": 1, "test": 2}
YOLO_SPLIT = {"train": "train", "valid": "val", "test": "test"}   # prepare's mapping
PROVENANCE = "provenance.json"
IMAGE_LIST = "image_list.tsv"
MAX_PATH = 260                    # Windows without LongPathsEnabled (prepare and Ultralytics both fail past it)
EXPORT_ONLY = ("drop", "export-labels")   # --export-only: what to do with export images the lv doesn't have
ASPECT_TOL = 0.01                 # lv vs export aspect ratio: within 1 % = a resize (warn), else padded/cropped (refuse)


class BuildError(RuntimeError):
    pass


def _load(name, path):
    """A jobs/ module by file (jobs/ modules import each other by bare name, so their folder goes on sys.path).
    registry modules are imported as the package (`from registry import ...`), never through this."""
    m = sys.modules.get(name)
    if m is not None and getattr(m, "__file__", None) and Path(m.__file__).resolve() == Path(path).resolve():
        return m
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


from registry import ids  # noqa: E402


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    return sha256_bytes(Path(p).read_bytes())


def list_text(pairs):
    """{(stem, split)} -> sorted `stem\\tsplit` lines; same text as registry/importers.list_text."""
    rows = sorted(set(pairs), key=lambda x: (SPLIT_ORDER.get(x[1], 9), x[1], x[0]))
    return "".join(f"{stem}\t{split}\n" for stem, split in rows)


def git_sha():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@contextmanager
def class_scheme(name):
    """prepare's convert_split reads module globals; set them as its main() does, and put them back."""
    if name not in prep.CLASS_SCHEMES:
        raise BuildError(f"unknown class scheme {name!r} (have {sorted(prep.CLASS_SCHEMES)})")
    old = prep.CATEGORY_NAME_TO_CLASS, prep.CLASS_NAMES
    prep.CATEGORY_NAME_TO_CLASS, prep.CLASS_NAMES = prep.CLASS_SCHEMES[name]
    try:
        yield prep.CLASS_NAMES
    finally:
        prep.CATEGORY_NAME_TO_CLASS, prep.CLASS_NAMES = old


# ---------------------------------------------------------------- inputs
def resolve_lv(lv):
    """lv id (under <CryoAI-laptop>/labels/), an lv folder, or a labels COCO file -> (folder, coco path)."""
    p = Path(lv)
    if LV_RE.match(str(lv)) and not p.exists():
        p = config.laptop_root() / "labels" / str(lv)
    if p.is_file():
        return p.parent, p
    if (p / "labels.coco.json").is_file():
        return p, p / "labels.coco.json"
    raise BuildError(f"no label version at {p} (want an lv id, an lv folder or a labels.coco.json)")


def resolve_export(split_from):
    """Export folder, or its name under <CryoAI>/training images/."""
    p = Path(split_from)
    if not p.is_dir():
        p = config.cryoai_root() / "training images" / str(split_from)
    if not (p / "train" / "_annotations.coco.json").is_file():
        raise BuildError(f"{p} is not a Roboflow COCO export (no train/_annotations.coco.json)")
    return p


def export_splits(export_dir):
    """prepare's split list: train, valid, and test when the export has one."""
    out = ["train", "valid"]
    if (export_dir / "test" / "_annotations.coco.json").exists():
        out.append("test")
    return out


def read_export(export_dir):
    """-> ({image key: (split, export image record)}, {split: COCO}). Refuses a key in two places."""
    where, cocos = {}, {}
    for split in export_splits(export_dir):
        c = json.loads((export_dir / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        cocos[split] = c
        for im in c["images"]:
            key = ids.image_key(im["file_name"])
            if key in where:
                raise BuildError(f"export image key {key} appears twice ({where[key][0]}, {split}): "
                                 f"the split is ambiguous")
            where[key] = (split, im)
    return where, cocos


def read_split_file(export_dir):
    """split_assignment.csv (make_balanced_split.py) -> (path, {image key: split column}, {image key: batch})."""
    f = export_dir / "split_assignment.csv"
    if not f.is_file():
        return None, {}, {}
    with open(f, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("image")]
    return (f, {ids.image_key(r["image"]): (r.get("split") or "").strip() for r in rows},
            {ids.image_key(r["image"]): (r.get("batch") or "").strip() for r in rows})


def lv_image_key(im):
    return ids.image_key((im.get("extra") or {}).get("name") or im["file_name"])


def excluded_sessions():
    """registry/validate.py EXCLUDED_SESSIONS (read from the file: validate has heavier imports)."""
    m = re.search(r"^EXCLUDED_SESSIONS\s*=\s*\{([^}]*)\}", (REG_DIR / "validate.py").read_text(encoding="utf-8"), re.M)
    return set(re.findall(r'"(\d{8})"', m.group(1))) if m else set()


# ---------------------------------------------------------------- build
def plan(lv_coco, where, cocos, split_rows, export_only_mode):
    """Match lv images to the export, in the export's image order. -> (per split [(kind, key, lv image | None,
    export image)], not_in_export [{image, reason}], export_only {split: n}, stretched [keys]). kind is "lv", or
    "export" for an export image not in the lv when export_only_mode is export-labels."""
    lv = {}
    for im in lv_coco["images"]:
        key = lv_image_key(im)
        if key in lv:
            raise BuildError(f"label version has image key {key} twice")
        lv[key] = im
    missing = [{"image": k, "reason": f"split file: {split_rows[k]}" if split_rows.get(k) else "not in the export"}
               for k in lv if k not in where]
    per_split, export_only, stretched, letterbox = defaultdict(list), Counter(), [], []
    for split, c in cocos.items():
        for eim in c["images"]:
            key = ids.image_key(eim["file_name"])
            if key not in lv:
                export_only[split] += 1
                if export_only_mode == "export-labels":
                    per_split[split].append(("export", key, None, eim))
                continue
            if split_rows and split_rows.get(key) != split:
                raise BuildError(f"{key}: export folder {split} but split_assignment.csv says {split_rows.get(key)!r}")
            im = lv[key]
            lw, lh, ew, eh = int(im["width"]), int(im["height"]), int(eim["width"]), int(eim["height"])
            if (lw, lh) != (ew, eh):
                (letterbox if abs(lw * eh - lh * ew) > ASPECT_TOL * lh * ew else stretched).append(key)
            per_split[split].append(("lv", key, im, eim))
    if letterbox:
        raise BuildError(f"{len(letterbox)} image(s) whose export copy has another aspect ratio than the lv's (padded "
                         f"or cropped?): outlines can't be mapped, e.g. {letterbox[:3]}")
    return per_split, missing, dict(export_only), stretched


def split_coco(lv_coco, entries, anns_by_image, export_coco):
    """A COCO in the export's naming (file names, image ids): lv outlines and lv image sizes for lv images, the
    export's own outlines and sizes for export-labelled ones. Categories are merged by name."""
    cats = [dict(c) for c in lv_coco["categories"]]
    by_name = {c["name"]: c["id"] for c in cats}
    ecat, e_anns = {}, defaultdict(list)
    if any(kind == "export" for kind, *_ in entries):
        nxt = max([c["id"] for c in cats] + [0]) + 1
        for c in export_coco["categories"]:
            if c["name"] not in by_name:
                by_name[c["name"]] = nxt
                cats.append({"id": nxt, "name": c["name"]})
                nxt += 1
            ecat[c["id"]] = by_name[c["name"]]
        for a in export_coco["annotations"]:
            e_anns[a["image_id"]].append(a)
    images, anns = [], []
    for kind, _key, lim, eim in entries:
        src = lim if kind == "lv" else eim
        images.append({"id": eim["id"], "file_name": eim["file_name"],
                       "width": int(src["width"]), "height": int(src["height"])})
        for a in (anns_by_image.get(lim["id"], []) if kind == "lv" else e_anns.get(eim["id"], [])):
            anns.append({"id": len(anns) + 1, "image_id": eim["id"],
                         "category_id": a["category_id"] if kind == "lv" else ecat[a["category_id"]],
                         "segmentation": a.get("segmentation")})
    return {"images": images, "annotations": anns, "categories": cats}


def label_hashes(out):
    """Label files {rel: md5} -> labels_sha256; images+labels -> dataset_sha256 (= the bundle pack key)."""
    dc = _load("dataset_cache", REPO / "jobs" / "dataset_cache.py")
    md5s = {p.relative_to(out).as_posix(): dc.C.md5_file(p)
            for d in dc.DATA_DIRS for p in sorted((out / d).rglob("*")) if p.is_file()}
    labels = {k: v for k, v in md5s.items() if k.startswith("labels/")}
    lines = "".join(f"{rel}\t{labels[rel]}\n" for rel in sorted(labels))
    return sha256_bytes(lines.encode("utf-8")), dc.content_sha256(md5s), len(labels)


def written_pairs(out, splits):
    """(image key, export split) for every image file actually in images/<split>/."""
    return {(ids.image_key(p.name), split) for split in splits
            for p in ((out / "images" / YOLO_SPLIT[split]).iterdir() if (out / "images" / YOLO_SPLIT[split]).is_dir()
                      else ()) if p.is_file()}


def changed_images(lv_dir):
    """changes.jsonl -> Counter(image key) of applied change rows (empty without a change log)."""
    f = lv_dir / "changes.jsonl"
    c = Counter()
    if f.is_file():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("image") and r.get("status", "applied") == "applied" and r.get("op") not in ("keep", "reject"):
                c[ids.image_key(r["image"])] += 1
    return c


def read_lv(lv, allow_superseded=False):
    """-> (folder, coco path, version.json dict, lv id). Refuses a superseded lv and labels edited after the fact."""
    lv_dir, lv_path = resolve_lv(lv)
    version = json.loads((lv_dir / "version.json").read_text(encoding="utf-8")) \
        if (lv_dir / "version.json").is_file() else {}
    lv_id = version.get("id") or lv_dir.name
    if (lv_dir / "SUPERSEDED.md").exists() and not allow_superseded:
        raise BuildError(f"{lv_id} is superseded ({lv_dir / 'SUPERSEDED.md'}); use its successor, or "
                         f"--allow-superseded")
    return lv_dir, lv_path, version, lv_id


def check_registrable_lv(lv_dir, lv_path, version):
    """--register needs a real label version: an lv_* folder whose version.json names it and pins labels_sha256."""
    if not (LV_RE.match(lv_dir.name) and version.get("id") == lv_dir.name and version.get("labels_sha256")
            and lv_path.name == "labels.coco.json"):
        raise BuildError(f"--register needs a label version (an lv_* folder with version.json id and labels_sha256); "
                         f"{lv_path} is not one (a bare COCO or a pull is never registered)")


def build(lv, split_from, output_name, output_root=None, scheme="ev", allow_superseded=False, export_only=None,
          log=print):
    """Write the dataset. -> provenance dict {"dataset_record": ..., "details": ...}. Raises BuildError.
    export_only: None (refuse when the export has images the lv lacks), "drop" or "export-labels"."""
    if not DATASET_RE.match(output_name or ""):
        raise BuildError(f"--output-name {output_name!r} is not a dataset id ({DATASET_RE.pattern})")
    if export_only not in (None, *EXPORT_ONLY):
        raise BuildError(f"--export-only must be one of {EXPORT_ONLY}")
    lv_dir, lv_path, version, lv_id = read_lv(lv, allow_superseded)
    export_dir = resolve_export(split_from)
    out = Path(output_root or config.cryoai_root() / "training outputs") / output_name
    problem = protected_problem(out)
    if problem:
        raise BuildError(f"refused: output {out} {problem}")
    if out.exists():
        raise BuildError(f"{out} exists; datasets are never overwritten (pick another --output-name)")
    lv_bytes = lv_path.read_bytes()
    lv_sha = sha256_bytes(lv_bytes)
    if version.get("labels_sha256") and version["labels_sha256"] != lv_sha:
        raise BuildError(f"{lv_path} sha256 {lv_sha[:12]} != version.json labels_sha256 "
                         f"{version['labels_sha256'][:12]}: the labels changed after the version was made")
    lv_coco = json.loads(lv_bytes.decode("utf-8"))
    where, cocos = read_export(export_dir)
    split_file, split_rows, batches = read_split_file(export_dir)
    per_split, missing, n_export_only, stretched = plan(lv_coco, where, cocos, split_rows, export_only)
    if sum(n_export_only.values()) and export_only is None:
        raise BuildError(f"{sum(n_export_only.values())} export image(s) are not in {lv_id} ({n_export_only}). "
                         f"Choose --export-only drop (leave them out) or --export-only export-labels (keep them "
                         f"with the export's own labels)")
    with class_scheme(scheme):
        used = {c["name"] for c in lv_coco["categories"]}
        if export_only == "export-labels":
            used |= {c["name"] for c in cocos[next(iter(cocos))]["categories"]}
        unmapped = sorted(used - set(prep.CATEGORY_NAME_TO_CLASS))
    if unmapped:
        raise BuildError(f"categories {unmapped} are not in class scheme {scheme!r} "
                         f"(prepare_all_layer_singleclass.CATEGORY_NAME_TO_CLASS)")
    longest = max((str(out / "images" / YOLO_SPLIT[s] / e[3]["file_name"]) for s, es in per_split.items()
                   for e in es), key=len, default="")
    if os.name == "nt" and len(longest) >= MAX_PATH:
        raise BuildError(f"an output file path would be {len(longest)} characters (Windows limit {MAX_PATH}; "
                         f"training could not read it either): use a shorter --output-root / --output-name")

    log(f"Label version: {lv_id}  ({lv_path})")
    log(f"Split from:    {export_dir.name}  ({export_dir})")
    log(f"Output:        {out}")
    ctx = dict(out=out, lv_coco=lv_coco, lv_dir=lv_dir, lv_id=lv_id, lv_sha=lv_sha, version=version,
               export_dir=export_dir, cocos=cocos, where=where, split_file=split_file, batches=batches,
               scheme=scheme, per_split=per_split, missing=missing, n_export_only=n_export_only,
               export_only=export_only, stretched=stretched, output_name=output_name, log=log)
    try:
        return _write(**ctx)
    except BaseException:
        if out.exists():                    # this run created it (it didn't exist above): no half-built dataset
            shutil.rmtree(out, ignore_errors=True)
        raise


def _write(out, lv_coco, lv_dir, lv_id, lv_sha, version, export_dir, cocos, where, split_file, batches, scheme,
           per_split, missing, n_export_only, export_only, stretched, output_name, log):
    anns_by_image = defaultdict(list)
    for a in lv_coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)
    used = {lim["id"] for es in per_split.values() for kind, _, lim, _ in es if kind == "lv"}
    multi_ring = sum(1 for a in lv_coco["annotations"] if a["image_id"] in used and isinstance(a.get("segmentation"), list)
                     and len(a["segmentation"]) > 1 and isinstance(a["segmentation"][0], list))
    no_outline = sum(1 for a in lv_coco["annotations"] if a["image_id"] in used and not a.get("segmentation"))
    stats = {}
    with class_scheme(scheme) as names, tempfile.TemporaryDirectory(prefix="dataset_from_lv_") as tmp:
        names = list(names)
        cmap = prep.CATEGORY_NAME_TO_CLASS
        log(f"Class scheme:  {scheme} -> {names}\n")
        for split in export_splits(export_dir):
            entries = per_split.get(split, [])
            coco_path = Path(tmp) / f"{split}.coco.json"
            coco_path.write_text(json.dumps(split_coco(lv_coco, entries, anns_by_image, cocos[split])),
                                 encoding="utf-8")
            dst = YOLO_SPLIT[split]
            log(f"=== {split} -> {dst} ===")
            s = prep.convert_split(coco_path, export_dir / split, out / "images" / dst, out / "labels" / dst)
            if s["n_images"] != len(entries):
                gone = [e[3]["file_name"] for e in entries if not (out / "images" / dst / e[3]["file_name"]).is_file()]
                raise BuildError(f"{split}: {len(entries) - s['n_images']} image(s) not copied (missing from "
                                 f"{export_dir / split}), e.g. {gone[:3]}")
            stats[split] = s
            log(f"  images: {s['n_images']}, labeled: {s['n_labels']}, instances: {s['n_instances']}, "
                f"dropped(non-EV etc.): {s['n_dropped']}, export images not in the lv: {n_export_only.get(split, 0)}"
                + (" (kept, export labels)" if export_only == "export-labels" and n_export_only.get(split) else ""))
        prep.write_dataset_yaml(out, has_test="test" in stats)
        cats = lv_coco["categories"] + (cocos[next(iter(cocos))]["categories"] if export_only == "export-labels" else [])
        class_map = {c["name"]: (names[cmap[c["name"]]] if cmap[c["name"]] is not None else None) for c in cats}

    pairs = written_pairs(out, stats)               # what was written, not what was planned
    planned = {(e[1], s) for s, es in per_split.items() for e in es}
    if pairs != planned:
        raise BuildError(f"images written differ from the plan: {sorted(pairs ^ planned)[:3]}")
    text = list_text(pairs)
    (out / IMAGE_LIST).write_bytes(text.encode("utf-8"))
    labels_sha, dataset_sha, n_label_files = label_hashes(out)
    class_counts = {}
    for split, s in stats.items():
        c = Counter()
        for dests in s["movement"].values():
            for dest, n in dests.items():
                if dest != "DROPPED":
                    c[dest] += n
        class_counts[split] = {k: c[k] for k in names}
    source_counts = {split: {orig: dict(sorted(d.items())) for orig, d in sorted(s["movement"].items())}
                     for split, s in stats.items()}
    excl = excluded_sessions()
    excluded = {k for k, _ in pairs if batches.get(k, "")[:8] in excl}
    images_lines = sorted(f"{split}\t{stem}\n" for stem, split in pairs if stem not in excluded)
    export_lines = sorted(f"{split}\t{k}\n" for k, (split, _) in where.items())
    labels_from = {split: dict(Counter(e[0] for e in es)) for split, es in per_split.items()}
    created = datetime.now().isoformat(timespec="seconds")
    eo_text = {"drop": "left out", "export-labels": "kept with the export's own labels"}.get(export_only)
    record = {
        "schema": "dataset/1", "id": output_name, "created": created,
        "source": {"kind": "label_version", "id": lv_id, "path": str(lv_dir)},
        "label_version": lv_id if LV_RE.match(lv_id) else None,
        "export": export_dir.name,
        "split": {"method": "export split folders, matched by image key (no re-split)",
                  "file": str(split_file) if split_file else str(export_dir),
                  "sha256": sha256_file(split_file) if split_file else None,
                  "sha256_gap": None if split_file else "no split file (export split folders)",
                  "export_images_sha256": sha256_bytes("".join(export_lines).encode("utf-8"))},
        "counts": {split: {"images": s["n_images"], "instances": s["n_instances"],
                           "labeled": s["n_labels"], "dropped": s["n_dropped"]} for split, s in stats.items()},
        "classes": names, "class_scheme": scheme, "class_map": class_map,
        "class_counts": class_counts,
        "images_sha256": sha256_bytes("".join(images_lines).encode("utf-8")),
        "images_sha256_method": "importer definition: sorted `split\\tstem` lines of the dataset's images, "
                                f"excluded-session images ({len(excluded)}) left out; image_list.sha256 covers all",
        "dataset_dir": str(out),
        "image_list": {"file": f"lists/{output_name}.tsv", "sha256": sha256_bytes(text.encode("utf-8")),
                       "method": "images written, matched to the export's split folders by image key",
                       "n": len(pairs)},
        "counts_trained": {split: s["n_images"] for split, s in stats.items()},
        "labels": {"lv_labels_sha256": lv_sha, "base": version.get("base"), "base_sha256": version.get("base_sha256"),
                   "label_files_sha256": labels_sha, "n_label_files": n_label_files,
                   "dataset_sha256": dataset_sha, "labels_from": labels_from},
        "not_in_export": len(missing),
        "export_only": {"choice": export_only, "n": {k: n_export_only[k] for k in
                                                     sorted(n_export_only, key=lambda k: SPLIT_ORDER.get(k, 9))}},
        "status": "frozen",
        "inputs": [{"id": lv_id, "role": "source", "entity": "label_version"},
                   {"id": export_dir.name, "role": "images", "entity": "roboflow_export"}],
        "recorded_by": f"dataset_from_lv.py {git_sha()}",
        "note": None,
    }
    warnings = []
    if missing:
        warnings.append(f"{len(missing)} lv image(s) not in the export's split folders, left out (details.not_in_export)")
    changed = changed_images(lv_dir)
    edited_dropped = {m["image"]: changed[m["image"]] for m in missing if changed.get(m["image"])}
    if edited_dropped:
        warnings.append(f"{len(edited_dropped)} of those have review edits in changes.jsonl that this dataset loses: "
                        f"{edited_dropped}")
    if sum(n_export_only.values()):
        warnings.append(f"export-only: {sum(n_export_only.values())} export image(s) not in the lv, {eo_text} "
                        f"(--export-only {export_only}): {n_export_only}")
    if stretched:
        warnings.append(f"{len(stretched)} image(s) whose lv size differs from the export's (same aspect ratio); "
                        f"outlines normalized by the lv size")
    if multi_ring:
        warnings.append(f"{multi_ring} lv object(s) with several rings: only the first ring is written "
                        f"(as convert_split does for exports)")
    if no_outline:
        warnings.append(f"{no_outline} lv object(s) with no outline, skipped (as convert_split does)")
    if version and "review_scope" not in version:
        warnings.append(f"{lv_id} has no review_scope: it applied every edit, reviewed or not (LABEL_CHANGES.md)")
    if (lv_dir / "SUPERSEDED.md").exists():
        warnings.append(f"{lv_id} is superseded (built with --allow-superseded)")
    record["note"] = "; ".join(warnings) or None
    details = {"not_in_export": missing, "edited_but_not_in_export": edited_dropped, "size_stretched": stretched,
               "warnings": warnings, "export_only_choice": export_only,
               "source_category_counts": source_counts,
               "images_without_labels": {split: s["images_without_labels"] for split, s in stats.items()},
               "lv_version": {k: version.get(k) for k in ("id", "base", "pkg", "tool", "created", "review_scope",
                                                         "counts_by_op", "supersedes", "labels_sha256") if k in version},
               "command": " ".join(sys.argv) if sys.argv and sys.argv[0].endswith("dataset_from_lv.py") else None}
    prov = {"schema": "dataset_provenance/1", "dataset_record": record, "details": details}
    (out / PROVENANCE).write_bytes((json.dumps(prov, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))

    log("")
    for w in warnings:
        log(f"[WARN] {w}")
    for m in missing:
        log(f"  not in export: {m['image']}  ({m['reason']})")
    total = sum(s["n_images"] for s in stats.values())
    log(f"[OK] Dataset ready at: {out}")
    log(f"[OK] Total: {total} images, {sum(s['n_instances'] for s in stats.values())} instances")
    for split, s in stats.items():
        log(f"  {YOLO_SPLIT[split]}: {s['n_images']} images, {s['n_instances']} instances  {class_counts[split]}  "
            f"labels from {labels_from.get(split, {})}")
    return prov


# ---------------------------------------------------------------- registry
def registry_lines(record):
    return json.dumps({k: v for k, v in record.items() if v is not None}, ensure_ascii=False)


def _reg():
    from registry import store
    return store


def check_id_free(dataset_id, root=None):
    """Before building: the id is not in the registry and its list file doesn't exist. -> registry root."""
    reg = _reg()
    try:
        root = Path(root or reg.default_root())
        if dataset_id in reg.current(root=root):
            raise BuildError(f"registry {root} already has id {dataset_id}")
    except reg.RegistryError as e:
        raise BuildError(f"registry refused: {e}")
    if (root / "lists" / f"{dataset_id}.tsv").exists():
        raise BuildError(f"{root / 'lists' / (dataset_id + '.tsv')} already exists")
    return root


def register(prov, dataset_dir, root=None):
    """Append the dataset/1 record and write <registry>/lists/<id>.tsv. Refuses an id already in the registry and
    a record that isn't from a label version with pinned labels."""
    reg = _reg()
    rec = {k: v for k, v in prov["dataset_record"].items() if v is not None}
    lab = rec.get("labels") or {}
    lvv = (prov.get("details") or {}).get("lv_version") or {}
    if not (LV_RE.match(str(rec.get("label_version"))) and lab.get("lv_labels_sha256")
            and lvv.get("labels_sha256") == lab["lv_labels_sha256"]):
        raise BuildError("only a dataset built from a label version (lv_* with version.json labels_sha256) is "
                         "registered; this one came from a bare COCO or a pull")
    root = check_id_free(rec["id"], root)
    data = (Path(dataset_dir) / IMAGE_LIST).read_bytes()
    if sha256_bytes(data) != rec["image_list"]["sha256"]:
        raise BuildError(f"{Path(dataset_dir) / IMAGE_LIST} doesn't match the record's image_list.sha256")
    lst = root / rec["image_list"]["file"]
    try:
        reg._guard(root)
        reg._guard(lst)
        reg._check(dict(rec, logged=reg.now()), root)          # schema check before anything is written
        # check_id_free above is only the early refusal: the id, the list file and the line are settled under the
        # registry lock, so of two racing registers exactly one wins and the list file is the winner's
        return reg.append(rec, root=root, new_only=True, files={rec["image_list"]["file"]: data})
    except reg.RegistryError as e:
        raise BuildError(f"registry refused: {e}")


def register_existing(folder, root=None):
    """Register a dataset folder built earlier: its provenance.json record, after checking the folder is unchanged."""
    folder = Path(folder).resolve()
    p = folder / PROVENANCE
    if not p.is_file():
        raise BuildError(f"{folder} has no {PROVENANCE}: not built by dataset_from_lv.py")
    prov = json.loads(p.read_text(encoding="utf-8"))
    rec = prov.get("dataset_record") or {}
    if prov.get("schema") != "dataset_provenance/1" or rec.get("id") != folder.name:
        raise BuildError(f"{p} is not a dataset_provenance/1 file for {folder.name}")
    _, dataset_sha, _ = label_hashes(folder)
    if dataset_sha != (rec.get("labels") or {}).get("dataset_sha256"):
        raise BuildError(f"{folder} images/labels changed since it was built (dataset_sha256 differs)")
    prov = dict(prov, dataset_record=dict(rec, dataset_dir=str(folder)))
    return register(prov, folder, root)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lv", help="lv id (under <CryoAI-laptop>/labels/), lv folder or labels COCO")
    ap.add_argument("--split-from", help="Roboflow export folder, or its name under <CryoAI>/training images/")
    ap.add_argument("--output-name", help="dataset id = folder name under training outputs/")
    ap.add_argument("--output-root", type=Path,
                    help="folder to create the dataset in (default <CryoAI>/training outputs)")
    ap.add_argument("--class-scheme", choices=sorted(prep.CLASS_SCHEMES), default="ev",
                    help="as prepare_all_layer_singleclass.py --class-scheme")
    ap.add_argument("--export-only", choices=EXPORT_ONLY,
                    help="export images the lv doesn't have: drop them, or keep them with the export's labels. "
                         "Required when there are any (no default)")
    ap.add_argument("--allow-superseded", action="store_true", help="build from an lv that has SUPERSEDED.md")
    ap.add_argument("--register", action="store_true",
                    help="append the dataset/1 record to the registry (default: print what would be added)")
    ap.add_argument("--register-existing", type=Path, metavar="FOLDER",
                    help="register a dataset folder built earlier (nothing is built)")
    ap.add_argument("--registry", type=Path, help="registry folder (default $CRYOEV_REGISTRY / <CryoAI-laptop>/registry)")
    a = ap.parse_args(argv)
    try:
        if a.register_existing:
            line = register_existing(a.register_existing, a.registry)
            print(f"[OK] registered dataset {line['id']} (+ {line['image_list']['file']})")
            return 0
        if not (a.lv and a.split_from and a.output_name):
            ap.error("--lv, --split-from and --output-name are required (or --register-existing FOLDER)")
        if a.register:                                   # everything --register needs, before building
            lv_dir, lv_path, version, _ = read_lv(a.lv, a.allow_superseded)
            check_registrable_lv(lv_dir, lv_path, version)
            check_id_free(a.output_name, a.registry)
        prov = build(a.lv, a.split_from, a.output_name, a.output_root, a.class_scheme, a.allow_superseded,
                     a.export_only)
        rec = prov["dataset_record"]
        if a.register:
            line = register(prov, rec["dataset_dir"], a.registry)
            print(f"\n[OK] registered dataset {line['id']} (+ {rec['image_list']['file']})")
        else:
            print("\nRegistry record that --register would append to dataset.jsonl "
                  f"(and lists/{rec['id']}.tsv = {IMAGE_LIST}):")
            print(registry_lines(rec))
    except (BuildError, config.ConfigError) as e:
        sys.exit(f"[ERROR] {e}")
    return 0


if __name__ == "__main__":
    main()
