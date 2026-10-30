"""
Save what Roboflow currently holds for a pinned list of image ids: every image's
outlines, tags and split, exactly as returned by the API. The by-id twin of
roboflow_snapshot.py, for images that predate the MANIFEST system (e.g. the
pre-lipo set listed by backfill_tracking.py). Read-only: only project.image(id)
is called; nothing in Roboflow is changed.

Writes, under --out (refuses to write into an existing folder):
  roboflow_state.json   one record per image (stem, roboflow_id, saved_at, the raw
                        image info minus embeddings/urls; info has split, tags,
                        lastSet, metadata, annotation)
  current.coco.json     the outlines as COCO (pixel coords of the image as
                        annotated; file_name = Roboflow name, extra.name = stem,
                        extra.split = Roboflow split, extra.roboflow_id; classes
                        by name)
  editors.csv           per image, any annotator / last-editor field the API
                        returned (blank when it returns none)
  diff_vs_export.csv    with --compare-export: per image, label count and class
                        counts now vs in that export (matched by stem)

    python annotation/roboflow_pull_ids.py --ids ".../tracking/prelipo_images.csv" \
        --workspace WS --project PROJ --out ".../roboflow_pull_YYYYMMDD_HHMM"         [--compare-export ".../training images/roboflow - 20261028 cryoai v5 S1"]

--ids is a CSV with columns stem and roboflow_id (rows without an id are skipped).
The API key is read from ROBOFLOW_API_KEY.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Per-image fields that could name who labeled or last edited an image. The
# project API returned none of these for this project on 2026-10-29 (only
# `owner`, the uploader, in search results), but record them if they appear.
EDITOR_KEYS = ("owner", "labeler", "labeledBy", "annotator", "createdBy", "updatedBy", "lastEditedBy",
               "lastSetBy", "user", "reviewer", "approvedBy")


def user_env(name):
    """A Windows user variable set with setx after this shell started (copied from roboflow_upload.py)."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, name)[0]
    except Exception:
        return None


def editor_fields(info):
    """Any annotator-like scalar on the image or its annotation record."""
    out = {}
    for where, d in (("", info), ("annotation.", info.get("annotation") or {})):
        for k in EDITOR_KEYS:
            if d.get(k) not in (None, "", [], {}):
                out[where + k] = d[k]
    return out


def export_counts(folder):
    """{stem: (split, Counter(class))} for every */_annotations.coco.json in an export folder."""
    out = {}
    for f in sorted(Path(folder).glob("*/_annotations.coco.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in j["categories"]}
        per = {}
        for a in j["annotations"]:
            per.setdefault(a["image_id"], Counter())[cats[a["category_id"]]] += 1
        for im in j["images"]:
            out[re.sub(r"\.rf\.[0-9a-f]+\.\w+$", "", im["file_name"])] = (f.parent.name, per.get(im["id"], Counter()))
    return out


def write_diff(state, folder, path):
    """Images whose label count or class counts differ between now and the export."""
    exp = export_counts(folder)
    rows = []
    for s in state:
        now = Counter(b["label"] for b in (s["info"].get("annotation") or {}).get("boxes") or [])
        split, then = exp.get(s["stem"], ("", None))
        diff = "not in export" if then is None else             ";".join(f"{k}:{then.get(k, 0)}->{now.get(k, 0)}" for k in sorted(set(now) | set(then))
                     if then.get(k, 0) != now.get(k, 0))
        rows.append({"stem": s["stem"], "export_split": split, "rf_split": s["info"].get("split", ""),
                     "n_export": sum(then.values()) if then is not None else "", "n_now": sum(now.values()),
                     "changed": bool(diff), "class_changes": diff})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def box_points(b):
    """A Roboflow box as a polygon; boxes without points become their rectangle."""
    pts = b.get("points")
    if pts:
        return [[float(x), float(y)] for x, y in pts]
    x, y, bw, bh = (float(b[k]) for k in ("x", "y", "width", "height"))
    return [[x - bw / 2, y - bh / 2], [x + bw / 2, y - bh / 2], [x + bw / 2, y + bh / 2], [x - bw / 2, y + bh / 2]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True, type=Path, help="CSV with stem, roboflow_id")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--compare-export", type=Path, help="export folder to diff label counts against")
    args = ap.parse_args()

    if args.out.exists():
        sys.exit(f"{args.out} exists; pulls are never overwritten")
    with open(args.ids, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("roboflow_id") and not r["roboflow_id"].startswith("AMBIGUOUS")]
    key = os.environ.get("ROBOFLOW_API_KEY") or user_env("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("set ROBOFLOW_API_KEY")
    from roboflow import Roboflow
    project = Roboflow(api_key=key).workspace(args.workspace).project(args.project)

    stamp = datetime.now().isoformat(timespec="seconds")
    state, images, anns, cats, editors = [], [], [], {}, []
    for i, r in enumerate(rows, 1):
        info = project.image(r["roboflow_id"])
        info.pop("embedding", None)
        info.pop("urls", None)
        state.append({"stem": r["stem"], "roboflow_id": r["roboflow_id"], "saved_at": stamp, "info": info})
        a = info.get("annotation") or {}
        boxes = a.get("boxes") or []
        ed = editor_fields(info)
        editors.append({"stem": r["stem"], "roboflow_id": r["roboflow_id"], "lastSet": info.get("lastSet", ""),
                        "editor_fields": json.dumps(ed) if ed else ""})
        labels = sorted({b["label"] for b in boxes})
        print(f"  {i:3d} {r['stem'][:60]:60s} {info.get('split', '?'):5s} {len(boxes):4d} outlines  {labels}")
        images.append({"id": i, "file_name": info.get("name", ""),
                       "width": int(a.get("width") or info.get("width") or 0),
                       "height": int(a.get("height") or info.get("height") or 0),
                       "extra": {"name": r["stem"], "split": info.get("split", ""), "roboflow_id": r["roboflow_id"]}})
        for b in boxes:
            cid = cats.setdefault(b["label"], len(cats) + 1)
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cid,
                         "segmentation": [[v for p in box_points(b) for v in p]], "iscrowd": 0})

    args.out.mkdir(parents=True)
    (args.out / "roboflow_state.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
    coco = {"info": {"source": f"{args.workspace}/{args.project}", "saved_at": stamp, "ids_from": str(args.ids)},
            "images": images, "annotations": anns,
            "categories": [{"id": c, "name": n} for n, c in cats.items()]}
    (args.out / "current.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    with open(args.out / "editors.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(editors[0]))
        w.writeheader()
        w.writerows(editors)
    n_ed = sum(1 for e in editors if e["editor_fields"])
    print(f"{len(state)} images saved ({len(anns)} outlines, classes {list(cats)}) -> {args.out}")
    print(f"annotator / last-editor fields returned for {n_ed} of {len(editors)} images")
    if args.compare_export:
        rows = write_diff(state, args.compare_export, args.out / "diff_vs_export.csv")
        ch = [r for r in rows if r["changed"]]
        print(f"vs {args.compare_export.name}: {len(ch)} of {len(rows)} images changed")
        for r in ch:
            print(f"  {r['stem'][:60]:60s} {r['n_export']!s:>4} -> {r['n_now']:<4} {r['class_changes']}")


if __name__ == "__main__":
    main()
