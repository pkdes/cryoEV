"""
Build a YOLO training dataset from a reviewed label version (`<CryoAI-laptop>/labels/lv_*/labels.coco.json`),
keeping the train/valid/test split of an existing Roboflow export.

    python data_utils/dataset_from_lv.py --lv lv_20261030_0909 \
        --split-from "roboflow - 20261028 cryoai v5 S1" --output-name lv_20261030_0909_v5S1_evnonev \
        --class-scheme ev_nonev [--register]

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
  - lv images not in the export are reported (with the split file's reason, e.g. DROPPED-duplicate) and left out.
  - export images not in the lv are counted and left out (no lv labels for them).
Pixels come from the export's image files; outlines are normalized by the lv image's width/height (a size that
differs from the export's is reported).

Registry: the dataset/1 record is printed. --register appends it (registry.append, schema-checked) and writes
<registry>/lists/<output-name>.tsv. Without --register nothing outside the dataset folder is written.
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


class BuildError(RuntimeError):
    pass


def _load(name, path):
    """A registry/ or jobs/ module by file (those folders aren't packages; their own imports use sys.path)."""
    m = sys.modules.get(name)
    if m is not None and getattr(m, "__file__", None) and Path(m.__file__).resolve() == Path(path).resolve():
        return m
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ids = _load("ids", REG_DIR / "ids.py")


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
    """split_assignment.csv (make_balanced_split.py) -> (path, {image key: split column}) or (None, {})."""
    f = export_dir / "split_assignment.csv"
    if not f.is_file():
        return None, {}
    with open(f, newline="", encoding="utf-8-sig") as fh:
        rows = {ids.image_key(r["image"]): (r.get("split") or "").strip() for r in csv.DictReader(fh) if r.get("image")}
    return f, rows


def lv_image_key(im):
    return ids.image_key((im.get("extra") or {}).get("name") or im["file_name"])


# ---------------------------------------------------------------- build
def plan(lv_coco, where, split_rows):
    """Match lv images to export splits. -> (per split [(lv image, export image)], not_in_export [{image, reason}],
    export_only Counter(split), size_mismatch [image keys])."""
    per_split = defaultdict(list)
    seen, missing, size_mismatch = set(), [], []
    for im in lv_coco["images"]:
        key = lv_image_key(im)
        if key in seen:
            raise BuildError(f"label version has image key {key} twice")
        seen.add(key)
        if key not in where:
            sf = split_rows.get(key)
            reason = f"split file: {sf}" if sf else "not in the export"
            missing.append({"image": key, "reason": reason})
            continue
        split, eim = where[key]
        sf = split_rows.get(key)
        if split_rows and sf != split:
            raise BuildError(f"{key}: export folder {split} but split_assignment.csv says {sf!r}")
        if (int(im["width"]), int(im["height"])) != (int(eim["width"]), int(eim["height"])):
            size_mismatch.append(key)
        per_split[split].append((im, eim))
    export_only = Counter(split for key, (split, _) in where.items() if key not in seen)
    return per_split, missing, export_only, size_mismatch


def split_coco(lv_coco, pairs, anns_by_image):
    """A COCO in the export's naming (file names, image ids) holding the lv's outlines and lv image sizes."""
    images, anns = [], []
    for lim, eim in pairs:
        images.append({"id": eim["id"], "file_name": eim["file_name"],
                       "width": int(lim["width"]), "height": int(lim["height"])})
        for a in anns_by_image.get(lim["id"], []):
            anns.append({"id": len(anns) + 1, "image_id": eim["id"], "category_id": a["category_id"],
                         "segmentation": a.get("segmentation")})
    return {"images": images, "annotations": anns, "categories": lv_coco["categories"]}


def label_hashes(out):
    """Label files {rel: md5} -> labels_sha256; images+labels -> dataset_sha256 (= the bundle pack key)."""
    dc = _load("dataset_cache", REPO / "jobs" / "dataset_cache.py")
    md5s = {p.relative_to(out).as_posix(): dc.C.md5_file(p)
            for d in dc.DATA_DIRS for p in sorted((out / d).rglob("*")) if p.is_file()}
    labels = {k: v for k, v in md5s.items() if k.startswith("labels/")}
    lines = "".join(f"{rel}\t{labels[rel]}\n" for rel in sorted(labels))
    return sha256_bytes(lines.encode("utf-8")), dc.content_sha256(md5s), len(labels)


def build(lv, split_from, output_name, output_root=None, scheme="ev", allow_superseded=False, log=print):
    """Write the dataset. -> provenance dict {"dataset_record": ..., "details": ...}. Raises BuildError."""
    if not DATASET_RE.match(output_name or ""):
        raise BuildError(f"--output-name {output_name!r} is not a dataset id ({DATASET_RE.pattern})")
    lv_dir, lv_path = resolve_lv(lv)
    export_dir = resolve_export(split_from)
    out = Path(output_root or config.cryoai_root() / "training outputs") / output_name
    problem = protected_problem(out)
    if problem:
        raise BuildError(f"refused: output {out} {problem}")
    if out.exists():
        raise BuildError(f"{out} exists; datasets are never overwritten (pick another --output-name)")

    version = json.loads((lv_dir / "version.json").read_text(encoding="utf-8")) \
        if (lv_dir / "version.json").is_file() else {}
    lv_id = version.get("id") or lv_dir.name
    if (lv_dir / "SUPERSEDED.md").exists() and not allow_superseded:
        raise BuildError(f"{lv_id} is superseded ({lv_dir / 'SUPERSEDED.md'}); use its successor, or "
                         f"--allow-superseded")
    lv_bytes = lv_path.read_bytes()
    lv_sha = sha256_bytes(lv_bytes)
    if version.get("labels_sha256") and version["labels_sha256"] != lv_sha:
        raise BuildError(f"{lv_path} sha256 {lv_sha[:12]} != version.json labels_sha256 "
                         f"{version['labels_sha256'][:12]}: the labels changed after the version was made")
    lv_coco = json.loads(lv_bytes.decode("utf-8"))
    where, _ = read_export(export_dir)
    split_file, split_rows = read_split_file(export_dir)
    per_split, missing, export_only, size_mismatch = plan(lv_coco, where, split_rows)
    longest = max((str(out / "images" / YOLO_SPLIT[s] / eim["file_name"]) for s, ps in per_split.items()
                   for _, eim in ps), key=len, default="")
    if os.name == "nt" and len(longest) >= MAX_PATH:
        raise BuildError(f"an output file path would be {len(longest)} characters (Windows limit {MAX_PATH}; "
                         f"training could not read it either): use a shorter --output-root / --output-name")

    log(f"Label version: {lv_id}  ({lv_path})")
    log(f"Split from:    {export_dir.name}  ({export_dir})")
    log(f"Output:        {out}")
    try:
        return _write(out, lv_coco, lv_dir, lv_id, lv_sha, version, export_dir, split_file, scheme, per_split,
                      missing, export_only, size_mismatch, where, output_name, log)
    except BaseException:
        if out.exists():                    # this run created it (it didn't exist above): no half-built dataset
            shutil.rmtree(out, ignore_errors=True)
        raise


def _write(out, lv_coco, lv_dir, lv_id, lv_sha, version, export_dir, split_file, scheme, per_split, missing,
           export_only, size_mismatch, where, output_name, log):
    anns_by_image = defaultdict(list)
    for a in lv_coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)
    used = {lim["id"] for pairs in per_split.values() for lim, _ in pairs}
    multi_ring = sum(1 for a in lv_coco["annotations"] if a["image_id"] in used and isinstance(a.get("segmentation"), list)
                     and len(a["segmentation"]) > 1 and isinstance(a["segmentation"][0], list))
    no_outline = sum(1 for a in lv_coco["annotations"] if a["image_id"] in used and not a.get("segmentation"))
    stats = {}
    with class_scheme(scheme) as names, tempfile.TemporaryDirectory(prefix="dataset_from_lv_") as tmp:
        names = list(names)
        cmap = prep.CATEGORY_NAME_TO_CLASS
        unmapped = sorted({c["name"] for c in lv_coco["categories"]} - set(cmap))
        if unmapped:
            raise BuildError(f"lv categories {unmapped} are not in class scheme {scheme!r} "
                             f"(prepare_all_layer_singleclass.CATEGORY_NAME_TO_CLASS)")
        log(f"Class scheme:  {scheme} -> {names}\n")
        for split in export_splits(export_dir):
            coco_path = Path(tmp) / f"{split}.coco.json"
            coco_path.write_text(json.dumps(split_coco(lv_coco, per_split.get(split, []), anns_by_image)),
                                 encoding="utf-8")
            dst = YOLO_SPLIT[split]
            log(f"=== {split} -> {dst} ===")
            s = prep.convert_split(coco_path, export_dir / split, out / "images" / dst, out / "labels" / dst)
            stats[split] = s
            log(f"  images: {s['n_images']}, labeled: {s['n_labels']}, instances: {s['n_instances']}, "
                f"dropped(non-EV etc.): {s['n_dropped']}, export images not in the lv: {export_only.get(split, 0)}")
        prep.write_dataset_yaml(out, has_test="test" in stats)
        class_map = {c["name"]: (names[cmap[c["name"]]] if cmap[c["name"]] is not None else None)
                     for c in lv_coco["categories"]}

    pairs = {(lv_image_key(lim), split) for split, ps in per_split.items() for lim, _ in ps}
    text = list_text(pairs)
    (out / IMAGE_LIST).write_text(text, encoding="utf-8")
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
    images_lines = sorted(f"{split}\t{stem}\n" for stem, split in pairs)
    export_lines = sorted(f"{split}\t{k}\n" for k, (split, _) in where.items())
    created = datetime.now().isoformat(timespec="seconds")
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
        "dataset_dir": str(out),
        "image_list": {"file": f"lists/{output_name}.tsv", "sha256": sha256_bytes(text.encode("utf-8")),
                       "method": "lv images matched to the export's split folders", "n": len(pairs)},
        "counts_trained": {split: s["n_images"] for split, s in stats.items()},
        "labels": {"lv_labels_sha256": lv_sha, "base": version.get("base"), "base_sha256": version.get("base_sha256"),
                   "label_files_sha256": labels_sha, "n_label_files": n_label_files,
                   "dataset_sha256": dataset_sha},
        "not_in_export": len(missing),
        "export_only": {k: export_only[k] for k in sorted(export_only, key=lambda k: SPLIT_ORDER.get(k, 9))},
        "status": "frozen",
        "inputs": [{"id": lv_id, "role": "source", "entity": "label_version"},
                   {"id": export_dir.name, "role": "images", "entity": "roboflow_export"}],
        "recorded_by": f"dataset_from_lv.py {git_sha()}",
        "note": None,
    }
    warnings = []
    if missing:
        warnings.append(f"{len(missing)} lv image(s) not in the export's split folders, left out (details.not_in_export)")
    if sum(export_only.values()):
        warnings.append(f"{sum(export_only.values())} export image(s) not in the lv, left out: {dict(export_only)}")
    if size_mismatch:
        warnings.append(f"{len(size_mismatch)} image(s) whose lv size differs from the export's; outlines normalized "
                        f"by the lv size")
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
    details = {"not_in_export": missing, "size_mismatch": size_mismatch, "warnings": warnings,
               "source_category_counts": source_counts,
               "images_without_labels": {split: s["images_without_labels"] for split, s in stats.items()},
               "lv_version": {k: version.get(k) for k in ("id", "base", "pkg", "tool", "created", "review_scope",
                                                         "counts_by_op", "supersedes", "labels_sha256") if k in version},
               "command": " ".join(sys.argv) if sys.argv and sys.argv[0].endswith("dataset_from_lv.py") else None}
    prov = {"schema": "dataset_provenance/1", "dataset_record": record, "details": details}
    (out / PROVENANCE).write_text(json.dumps(prov, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    log("")
    for w in warnings:
        log(f"[WARN] {w}")
    for m in missing:
        log(f"  not in export: {m['image']}  ({m['reason']})")
    total = sum(s["n_images"] for s in stats.values())
    log(f"[OK] Dataset ready at: {out}")
    log(f"[OK] Total: {total} images, {sum(s['n_instances'] for s in stats.values())} instances")
    for split, s in stats.items():
        log(f"  {YOLO_SPLIT[split]}: {s['n_images']} images, {s['n_instances']} instances  {class_counts[split]}")
    return prov


# ---------------------------------------------------------------- registry
def registry_lines(record):
    return json.dumps({k: v for k, v in record.items() if v is not None}, ensure_ascii=False)


def register(prov, dataset_dir, root=None):
    """Append the dataset/1 record and write <registry>/lists/<id>.tsv. Refuses an id already in the registry."""
    reg = _load("registry", REG_DIR / "registry.py")
    root = Path(root or reg.default_root())
    rec = {k: v for k, v in prov["dataset_record"].items() if v is not None}
    if rec["id"] in reg.current(root=root):
        raise BuildError(f"registry already has id {rec['id']}")
    text = (Path(dataset_dir) / IMAGE_LIST).read_text(encoding="utf-8")
    lst = root / rec["image_list"]["file"]
    reg._guard(root)
    reg._guard(lst)
    if lst.exists() and lst.read_text(encoding="utf-8") != text:
        raise BuildError(f"{lst} exists with other content")
    reg._check(dict(rec, logged=reg.now()), root)          # schema check before anything is written
    lst.parent.mkdir(parents=True, exist_ok=True)
    lst.write_text(text, encoding="utf-8")
    return reg.append(rec, root=root)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lv", required=True, help="lv id (under <CryoAI-laptop>/labels/), lv folder or labels COCO")
    ap.add_argument("--split-from", required=True,
                    help="Roboflow export folder, or its name under <CryoAI>/training images/")
    ap.add_argument("--output-name", required=True, help="dataset id = folder name under training outputs/")
    ap.add_argument("--output-root", type=Path,
                    help="folder to create the dataset in (default <CryoAI>/training outputs)")
    ap.add_argument("--class-scheme", choices=sorted(prep.CLASS_SCHEMES), default="ev",
                    help="as prepare_all_layer_singleclass.py --class-scheme")
    ap.add_argument("--allow-superseded", action="store_true", help="build from an lv that has SUPERSEDED.md")
    ap.add_argument("--register", action="store_true",
                    help="append the dataset/1 record to the registry (default: print what would be added)")
    ap.add_argument("--registry", type=Path, help="registry folder (default $CRYOEV_REGISTRY / <CryoAI-laptop>/registry)")
    a = ap.parse_args(argv)
    try:
        prov = build(a.lv, a.split_from, a.output_name, a.output_root, a.class_scheme, a.allow_superseded)
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
