"""
Save what Roboflow currently holds for the tiles in a MANIFEST: every image's
outlines, tags and split, exactly as returned by the API. Run it before
deleting or replacing images, so review edits made in Roboflow are not lost,
and before reviewing a labeler's work, so their version can be compared later.

Writes, under --out (refuses to write into an existing folder):
  roboflow_state.json   one record per tile (source_file, roboflow_id, job, the
                        raw image info minus embeddings/urls; info has split,
                        lastSet, metadata)
  current.coco.json     the outlines as COCO (pixel coords of the image as
                        annotated; extra.name = the MANIFEST source_file;
                        classes by name), so roboflow_relabel.py --coco can put
                        them on another copy of the same fields.
  jobs.json             with --jobs: each job's details and image ids
  control.csv           with --control-labels: per tile, how many outlines still
                        match the model pre-annotation (IoU >= 0.98)

    python annotation/roboflow_snapshot.py --manifest ".../3d.../MANIFEST.csv" --grids 6 \
        --workspace WS --project PROJ --out ".../snapshot_YYYYMMDD" [--roles AU-eval]
        [--jobs JOB1,JOB2] [--control-labels DIR] [--mark-split valid [--mark-limit N]]

--mark-split S tags the saved tiles whose split is S in this project (metadata
review_stage=au_saved, au_saved_at, au_snapshot; tag AU-saved). It runs only
after the files are written. Without it nothing in Roboflow is changed.

The API key is read from ROBOFLOW_API_KEY.
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roboflow_upload import read_manifest, user_env  # noqa: E402

MARK_TAG = "AU-saved"
MATCH_IOU = 0.98


def job_image_ids(project, job_id):
    ids, after = [], None
    while True:
        page = project.get_annotation_job_images(job_id, limit=200, after=after)
        ids += page.get("imageIds") or []
        after = page.get("nextPageToken")
        if not after:
            return ids


def polygon_iou(p, q):
    """IoU of two polygons, rasterized inside their joint bounding box."""
    pts = np.vstack([p, q])
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 1
    w, h = x1 - x0, y1 - y0
    a, b = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    cv2.fillPoly(a, [np.round(p - (x0, y0)).astype(np.int32)], 1)
    cv2.fillPoly(b, [np.round(q - (x0, y0)).astype(np.int32)], 1)
    inter, union = np.logical_and(a, b).sum(), np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def control(boxes, label_file, width, height):
    """How many model outlines are still in Roboflow unchanged (IoU >= MATCH_IOU)."""
    model = []
    for line in label_file.read_text().split("\n"):
        v = line.split()
        if len(v) > 6:
            model.append(np.array(v[1:], float).reshape(-1, 2) * (width, height))
    rf = [np.array(b["points"], float) for b in boxes if b.get("points")]
    boxes_rf = [(q.min(0), q.max(0)) for q in rf]
    same = 0
    for p in model:
        lo, hi = p.min(0), p.max(0)
        for q, (qlo, qhi) in zip(rf, boxes_rf):
            if (qlo < hi).all() and (lo < qhi).all() and polygon_iou(p, q) >= MATCH_IOU:
                same += 1
                break
    return len(model), len(rf), same


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--grids", required=True, type=lambda s: {g.strip() for g in s.split(",")})
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--roles", default=None, type=lambda s: set(s.split(",")),
                    help="only these split_roles go into current.coco.json (state.json keeps all)")
    ap.add_argument("--jobs", default=None, type=lambda s: [j.strip() for j in s.split(",")],
                    help="only tiles in these annotation jobs")
    ap.add_argument("--control-labels", default=None, type=Path,
                    help="YOLO-seg pre-annotation folder; files matched by source_file stem")
    ap.add_argument("--mark-split", default=None, help="mark saved tiles whose split is this (e.g. valid)")
    ap.add_argument("--mark-limit", default=None, type=int, help="mark at most N tiles (test first)")
    args = ap.parse_args()

    if args.out.exists():
        sys.exit(f"{args.out} exists; snapshots are never overwritten")
    rows, _f = read_manifest(args.manifest)
    rows = [r for r in rows if r["grid"] in args.grids and r["roboflow_id"]]
    key = os.environ.get("ROBOFLOW_API_KEY") or user_env("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("set ROBOFLOW_API_KEY")
    from roboflow import Roboflow
    project = Roboflow(api_key=key).workspace(args.workspace).project(args.project)

    job_of, jobs = {}, {}
    if args.jobs:
        for j in args.jobs:
            ids = job_image_ids(project, j)
            jobs[j] = {"info": project.get_annotation_job(j), "image_ids": ids}
            job_of.update({i: j for i in ids})
        rows = [r for r in rows if r["roboflow_id"] in job_of]
        counts = ", ".join(f"{j}: {len(v['image_ids'])}" for j, v in jobs.items())
        print(f"jobs {counts} -> {len(rows)} MANIFEST tiles")

    stamp = datetime.now().isoformat(timespec="seconds")
    state, images, anns, cats, ctrl = [], [], [], {}, []
    for i, r in enumerate(rows, 1):
        info = project.image(r["roboflow_id"])
        info.pop("embedding", None)
        info.pop("urls", None)
        state.append({"source_file": r["source_file"], "split_role": r["split_role"],
                      "roboflow_id": r["roboflow_id"], "job": job_of.get(r["roboflow_id"]),
                      "saved_at": stamp, "info": info})
        a = info.get("annotation") or {}
        boxes = a.get("boxes") or []
        labels = sorted({b["label"] for b in boxes})
        print(f"  {r['source_file']:24s} {info.get('split', '?'):5s} {len(boxes):4d} outlines  {labels}")
        w, h = (int(v) for v in r["output_size"].split("x"))
        width, height = int(a.get("width") or w), int(a.get("height") or h)
        if args.control_labels:
            stem = Path(r["source_file"]).stem
            lab = next(args.control_labels.glob(f"*__{stem}__*.txt"), None)
            if lab:
                n_model, n_rf, same = control(boxes, lab, width, height)
                ctrl.append({"source_file": r["source_file"], "split": info.get("split"),
                             "model": n_model, "roboflow": n_rf, "unchanged": same,
                             "classes": ";".join(labels)})
        if args.roles and r["split_role"] not in args.roles:
            continue
        images.append({"id": i, "file_name": r["output_name"], "width": width,
                       "height": height, "extra": {"name": r["source_file"]}})
        for b in boxes:
            cid = cats.setdefault(b["label"], len(cats) + 1)
            pts = b.get("points")
            if not pts:
                x, y, bw, bh = (float(b[k]) for k in ("x", "y", "width", "height"))
                pts = [[x - bw / 2, y - bh / 2], [x + bw / 2, y - bh / 2], [x + bw / 2, y + bh / 2], [x - bw / 2, y + bh / 2]]
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cid,
                         "segmentation": [[float(v) for p in pts for v in p]], "iscrowd": 0})

    args.out.mkdir(parents=True)
    (args.out / "roboflow_state.json").write_text(json.dumps(state, indent=1))
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": c, "name": n} for n, c in cats.items()]}
    (args.out / "current.coco.json").write_text(json.dumps(coco))
    if jobs:
        (args.out / "jobs.json").write_text(json.dumps(jobs, indent=1))
    if ctrl:
        with open(args.out / "control.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(ctrl[0]))
            wr.writeheader()
            wr.writerows(ctrl)
        print(f"control: outlines still matching the model (IoU >= {MATCH_IOU})")
        for c in ctrl:
            print(f"  {c['source_file']:24s} {c['split']:5s} model {c['model']:4d}  roboflow {c['roboflow']:4d}  "
                  f"unchanged {c['unchanged']:4d}")
    print(f"{len(state)} tiles saved; {len(images)} in current.coco.json "
          f"({len(anns)} outlines, classes {list(cats)}) -> {args.out}")

    if args.mark_split:
        written = json.loads((args.out / "roboflow_state.json").read_text())
        todo = [s for s in written if s["info"].get("split") == args.mark_split
                and MARK_TAG not in (s["info"].get("tags") or [])][:args.mark_limit]
        for s in todo:
            project.update_image_metadata(image_id=s["roboflow_id"],
                                          metadata={"review_stage": "au_saved", "au_saved_at": stamp,
                                                    "au_snapshot": args.out.name},
                                          add_tags=[MARK_TAG])
            print(f"  marked {s['source_file']}")
        print(f"{len(todo)} tiles marked review_stage=au_saved")


if __name__ == "__main__":
    main()
