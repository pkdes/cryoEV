"""
Backfill per-image tracking for Roboflow images that predate the MANIFEST
system (the pre-lipo set), in the same columns as a session MANIFEST where
they apply, plus each image's history across the dated Roboflow exports.

Images are keyed by stem: the Roboflow name with its extension joined by "_"
(`x_0_hm.png` -> `x_0_hm_png`), which is also what every export's file name
starts with before `.rf.<hash>`. Pixel md5 is not used: Roboflow re-encodes.
The export also rewrites spaces: `20260420 - MDA-MB RM__x.png` exports as
`20260420-MDA-MB-RM__x_png...`, so names are matched on rf_key(), which turns
every run of spaces and hyphens into one hyphen on both sides.

    python annotation/backfill_tracking.py --rf-images rf_all_images.json \
        --exports-root ".../CryoAI/training images" \
        --current "roboflow - 20261028 cryoai v5 S1" --history "roboflow - 20260924 cryoai v4" \
        --exclude-batch "20261012 lipo+VLDL" --out prelipo_tracking.csv         [--images-out prelipo_images.csv] [--session-map session_map.csv] [--editors <pull>/editors.csv]

--rf-images is a JSON list from Roboflow search (id, name, split, tags, created).
The pinned set is every image in --history (here v4's 170), plus nothing else.
--session-map is a CSV keyed by batch (the S1 split_assignment.csv batch) with
session_date, session_date_basis, grid_id, sample_label, tracker_note, filled by
hand from the lab's grid tracker where the mapping is unambiguous (blank
otherwise). session_date_basis says whether the date is from the tracker or
inferred (e.g. a year worked out from the Roboflow upload date). The scope grid
(`grid`) comes from the stem. --editors is roboflow_pull_ids.py's editors.csv;
any annotator field it holds goes into `annotator`.
--history-review-data takes the --history export's images, splits and label
counts from a review package data.js built from it (build_review_package.py),
for when that export's COCO files are not available locally.
Columns that MANIFEST tiles have but these images can't (roboflow_batch,
split_role) are kept and left blank; label_source is "unknown (pre-MANIFEST)".
An "in: <export>" cell is "unread" when that export (or one of its split files)
could not be read, so absence there is unknown rather than "not in it".
--images-out writes the short list (stem, roboflow_id, S1 split, duplicate_of)
that roboflow_pull_ids.py takes.
Read-only: nothing is written to Roboflow or to the exports.
"""
import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

SPLITS = ("train", "valid", "test")
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+\.\w+$")
GRID_PATTERNS = [re.compile(p) for p in (r"__grid(.+?)__",           # <session>__grid4__4_square...
                                          r"^(?:.*__)?(\d+)_square",  # <session>__3_square..., bare 4_square...
                                          r"^grid-(\d+)_",             # grid-2_square...
                                          r"^Grid(\d+)_")]             # Grid3_0019_Jul30


def stem_of_export(file_name):
    return RF_SUFFIX.sub("", Path(file_name).name)


def stem_of_rf(name):
    p = Path(name)
    return f"{p.stem}_{p.suffix[1:]}" if p.suffix else p.name


def rf_key(stem):
    """Match key for a Roboflow name's stem vs an export stem (the export turns ' - ' and ' ' into '-')."""
    return re.sub(r"[\s-]+", "-", stem)


def grid_of(stem):
    """Scope grid / cassette position as written in the stem (blank when the stem doesn't say)."""
    for p in GRID_PATTERNS:
        m = p.search(stem)
        if m:
            return m.group(1)
    return ""


def read_csv(path, key):
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {r[key]: r for r in csv.DictReader(fh)}


def review_index(data_js):
    """export_index() from a review package's data.js (its gt lists are the export's labels)."""
    d = json.loads(Path(data_js).read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
    out = {}
    for im in d["images"]:
        c = {}
        for g in im["gt"]:
            c[g["c"]] = c.get(g["c"], 0) + 1
        out[stem_of_export(im["file"])] = (im["split"], sum(c.values()), c)
    return out


def export_index(folder):
    """({stem: (split, n_annotations, {class: n})}, [split folders whose COCO file is missing or unreadable])."""
    out, unread = {}, []
    files = {f.parent.name: f for f in folder.glob("*/_annotations.coco.json")}
    unread += [s for s in SPLITS if (folder / s).is_dir() and s not in files]
    for name, f in sorted(files.items()):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            print(f"  [skip] unreadable {f.parent.parent.name}/{f.parent.name}: {type(e).__name__}")
            unread.append(name)
            continue
        cats = {c["id"]: c["name"] for c in j["categories"]}
        per = {}
        for a in j["annotations"]:
            per.setdefault(a["image_id"], {}).setdefault(cats[a["category_id"]], 0)
            per[a["image_id"]][cats[a["category_id"]]] += 1
        for im in j["images"]:
            c = per.get(im["id"], {})
            out[stem_of_export(im["file_name"])] = (f.parent.name, sum(c.values()), c)
    return out, unread


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rf-images", required=True, type=Path)
    ap.add_argument("--exports-root", required=True, type=Path)
    ap.add_argument("--current", required=True, help="export folder holding split_assignment.csv")
    ap.add_argument("--history", required=True, help="export folder whose images are the pinned set")
    ap.add_argument("--exclude-batch", action="append", default=[])
    ap.add_argument("--project", default="carney-lab-image-annotation/cryoai-20260611-current-version")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--history-review-data", type=Path, help="data.js of a review package built from --history")
    ap.add_argument("--images-out", type=Path, help="also write stem, roboflow_id, s1_split, duplicate_of")
    ap.add_argument("--session-map", type=Path, help="CSV: batch, session_date, grid_id, sample_label, tracker_note")
    ap.add_argument("--editors", type=Path, help="roboflow_pull_ids.py editors.csv (stem, editor_fields)")
    args = ap.parse_args()
    sessions = read_csv(args.session_map, "batch")
    editors = read_csv(args.editors, "stem")

    exports = sorted(d for d in args.exports_root.iterdir()
                     if d.is_dir() and (any(d.glob("*/_annotations.coco.json")) or any((d / s).is_dir() for s in SPLITS)))
    idx, unread = {}, {}
    for d in exports:
        idx[d.name], unread[d.name] = export_index(d)
        if unread[d.name]:
            print(f"  {d.name}: unread splits {unread[d.name]}")
    if args.history_review_data:
        idx[args.history], unread[args.history] = review_index(args.history_review_data), []
        print(f"  {args.history}: {len(idx[args.history])} images from {args.history_review_data}")
    pinned = sorted(idx[args.history])
    with open(args.exports_root / args.current / "split_assignment.csv", newline="", encoding="utf-8") as fh:
        assign = {r["image"]: r for r in csv.DictReader(fh)}
    dup_of = {}
    report = (args.exports_root / args.current / "SPLIT_REPORT.md").read_text(encoding="utf-8")
    for kept, dropped in re.findall(r"`([^`]+)` \(\w+\) <- `([^`]+)`", report):
        dup_of[dropped] = kept

    rf = {}
    for r in json.loads(args.rf_images.read_text(encoding="utf-8")):
        rf.setdefault(rf_key(stem_of_rf(r["name"])), []).append(r)

    rows, missing, multiple = [], [], []
    for s in pinned:
        a = assign.get(s) or assign.get(dup_of.get(s, ""), {})
        if a.get("batch") in args.exclude_batch:
            continue
        hits = rf.get(rf_key(s), [])
        if not hits:
            missing.append(s)
        elif len(hits) > 1:
            multiple.append((s, [h["id"] for h in hits]))
        sm = sessions.get(a.get("batch", ""), {})
        r = hits[0] if len(hits) == 1 else {}
        cur = idx[args.current].get(s)
        row = {
            "stem": s,
            "output_name": r.get("name", ""),
            "session": a.get("batch", ""),
            "session_date": sm.get("session_date", ""),
            "session_date_basis": sm.get("session_date_basis", ""),
            "grid": grid_of(s),
            "grid_id": sm.get("grid_id", ""),
            "sample_label": sm.get("sample_label", ""),
            "tracker_note": sm.get("tracker_note", ""),
            "output_size": f"{r['width']}x{r['height']}" if r else "",
            "roboflow_project": args.project,
            "roboflow_batch": r.get("batch", ""),
            "split_role": "",
            "roboflow_id": r.get("id", "" if not hits else f"AMBIGUOUS x{len(hits)}"),
            "roboflow_split": r.get("split", ""),
            "roboflow_tags": ";".join(r.get("tags") or []),
            "uploaded_at": datetime.fromtimestamp(r["created"] / 1000, timezone.utc).isoformat(timespec="seconds")
                           if r.get("created") else "",
            "label_source": "unknown (pre-MANIFEST)",
            "annotator": (editors.get(s) or {}).get("editor_fields", ""),
            "duplicate_of": dup_of.get(s, ""),
            "current_split": cur[0] if cur else ("dropped (duplicate)" if s in dup_of else ""),
            "s1_split": a.get("split", "") if s not in dup_of else "",
            "current_n_labels": cur[1] if cur else "",
            "current_classes": ";".join(f"{k}:{v}" for k, v in sorted(cur[2].items())) if cur else "",
        }
        for d in exports:
            e = idx[d.name].get(s)
            row[f"in: {d.name}"] = f"{e[0]} ({e[1]})" if e else ("unread" if unread.get(d.name) else "")
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if args.images_out:
        with open(args.images_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["stem", "roboflow_id", "roboflow_name", "s1_split", "duplicate_of"])
            w.writeheader()
            w.writerows({"stem": r["stem"], "roboflow_id": r["roboflow_id"], "roboflow_name": r["output_name"],
                         "s1_split": r["s1_split"], "duplicate_of": r["duplicate_of"]} for r in rows)
        print(f"  image list -> {args.images_out}")
    print(f"{len(rows)} pinned images -> {args.out}")
    print(f"  exports scanned: {', '.join(d.name for d in exports)}")
    print(f"  not found in Roboflow by name: {len(missing)}" + (f" {missing}" if missing else ""))
    print(f"  found more than once in Roboflow: {len(multiple)}" + (f" {multiple}" if multiple else ""))
    print(f"  excluded by batch {args.exclude_batch}: {len(pinned) - len(rows)}")
    print(f"  duplicates (dropped from {args.current}): {sum(1 for r in rows if r['duplicate_of'])}")


if __name__ == "__main__":
    main()
