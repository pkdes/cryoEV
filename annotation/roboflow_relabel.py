"""
Put a human-labeled COCO export onto the matching bar-free tiles already in
Roboflow, replacing their model pre-annotations, and record it.

The export's images are the delivered scalebar JPGs (`extra.name` = the
MANIFEST's source_file); the bar-free tile is the same field at 1/factor scale,
so polygons are divided by the MANIFEST factor. Labels keep the labeler's
classes. Rows get label_source=<source> and split_role=<role>.

    python annotation/roboflow_relabel.py --coco ".../_annotations.coco.json" \
        --manifest ".../3d.../MANIFEST.csv" --source AU-job3 --out ".../AU_job3" \
        [--preview 2] [--dry-run] [--limit 1] --workspace WS --project PROJ

Writes <out>/labels_<A/px>Apx/<tile stem>.txt (YOLO-seg) and, with --preview,
<out>/preview/*.png. The API key is read from ROBOFLOW_API_KEY.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roboflow_upload import git_commit, read_manifest, user_env, write_manifest  # noqa: E402

LOG_FIELDS = ["date", "workspace", "project", "batch", "n_uploaded", "n_preannotated", "n_blind",
              "model", "conf_cut", "cryoEV_commit", "manifest", "note"]
COLORS = [(0, 255, 255), (255, 0, 255), (0, 165, 255), (0, 255, 0), (255, 255, 0)]


def convert(coco, rows, out_dir):
    """One YOLO-seg file per matched tile; returns [(row, label_path, Counter)]."""
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    used = sorted({a["category_id"] for a in coco["annotations"]})
    names = [cats[i] for i in used]
    cls = {cid: k for k, cid in enumerate(used)}
    by_src = {r["source_file"]: r for r in rows}
    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for im in coco["images"]:
        name = (im.get("extra") or {}).get("name") or im["file_name"]
        r = by_src.get(name)
        if r is None:
            sys.exit(f"no MANIFEST row for {name}")
        w, h = im["width"], im["height"]
        ow, oh = (int(v) for v in r["output_size"].split("x"))
        f = float(r["factor"])
        if abs(w / f - ow) > 1 or abs(h / f - oh) > 1:
            sys.exit(f"{name}: {w}x{h} / {f} != {r['output_size']}")
        lines, n = [], Counter()
        for a in anns.get(im["id"], []):
            for poly in a["segmentation"]:
                # x/f/ow == x/w up to rounding: normalize by the source size
                xy = [min(max(v / (w if i % 2 == 0 else h), 0.0), 1.0) for i, v in enumerate(poly)]
                lines.append(" ".join([str(cls[a["category_id"]])] + [f"{v:.6f}" for v in xy]))
                n[cats[a["category_id"]]] += 1
        lab = out_dir / (Path(r["output_name"]).stem + ".txt")
        lab.write_text("\n".join(lines) + "\n")
        done.append((r, lab, n))
    return done, names


def preview(img_path, lab, names, dst):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    for ln in lab.read_text().splitlines():
        p = ln.split()
        pts = (np.array(p[1:], float).reshape(-1, 2) * [w, h]).round().astype(np.int32)
        cv2.polylines(img, [pts], True, COLORS[int(p[0]) % len(COLORS)], 1, cv2.LINE_AA)
    for k, nm in enumerate(names):
        cv2.putText(img, nm, (10, 25 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[k % len(COLORS)], 2)
    cv2.imwrite(str(dst), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--source", required=True, help="label_source value and Roboflow tag, e.g. AU-job3")
    ap.add_argument("--role", default=None, help="split_role to set (default: <labeler>-eval)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workspace")
    ap.add_argument("--project")
    ap.add_argument("--drop-tag", default="v4_y8m_20260924-preannot@0.50",
                    help="pre-annotation tag to remove from relabeled images")
    ap.add_argument("--preview", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    role = args.role or args.source.split("-")[0] + "-eval"

    rows, fields = read_manifest(args.manifest)
    if "label_source" not in fields:
        fields.append("label_source")
    for r in rows:
        r.setdefault("label_source", "")
        if not r["label_source"] and r["split_role"] == "pre-annotated":
            r["label_source"] = "v4m@0.50"
    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    apx = rows[0]["output_a_px"].replace(".", "p")[:5]
    done, names = convert(coco, rows, args.out / f"labels_{apx}Apx")

    total = Counter()
    for r, lab, n in done:
        total += n
        print(f"  {r['source_file']:24s} {sum(n.values()):4d}  {dict(sorted(n.items()))}  "
              f"[{r['split_role']}, {r['label_source'] or '-'}]")
    n_in = sum(len(a["segmentation"]) for a in coco["annotations"])
    print(f"outlines: {n_in} in COCO, {sum(total.values())} written; labelmap {dict(enumerate(names))}")
    if n_in != sum(total.values()):
        sys.exit("outline count mismatch")

    if args.preview:
        pv = args.out / "preview"
        pv.mkdir(exist_ok=True)
        for r, lab, _ in done[:args.preview]:
            preview(args.manifest.parent / r["output_name"], lab, names, pv / (lab.stem + ".png"))
        print(f"previews -> {pv}")

    pending = [d for d in done if d[0]["label_source"] != args.source]
    if args.limit:
        pending = pending[:args.limit]
    print(f"already relabeled: {len(done) - len([d for d in done if d[0]['label_source'] != args.source])}; "
          f"this run: {len(pending)}")
    if args.dry_run or not pending:
        return
    if not (args.workspace and args.project):
        sys.exit("--workspace and --project are needed to upload")

    key = os.environ.get("ROBOFLOW_API_KEY") or user_env("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("set ROBOFLOW_API_KEY")
    from roboflow import Roboflow
    project = Roboflow(api_key=key).workspace(args.workspace).project(args.project)

    up = 0
    try:
        for r, lab, n in pending:
            if not r["roboflow_id"]:
                sys.exit(f"{r['output_name']} is not in Roboflow yet")
            resp = project.single_upload(
                image_id=r["roboflow_id"], annotation_path=str(lab),
                annotation_labelmap=dict(enumerate(names)), annotation_overwrite=True,
                num_retry_uploads=3)
            ann = resp.get("annotation") or {}
            if not ann.get("success"):
                sys.exit(f"annotation upload failed for {r['output_name']}: {resp}")
            project.update_image_metadata(image_id=r["roboflow_id"],
                                          metadata={"split_role": role, "label_source": args.source},
                                          add_tags=[args.source], remove_tags=[args.drop_tag])
            r.update(label_source=args.source, split_role=role,
                     uploaded_at=datetime.now().isoformat(timespec="seconds"))
            up += 1
            print(f"  relabeled {r['output_name']} ({sum(n.values())} outlines) -> {r['roboflow_id']}")
    finally:
        write_manifest(args.manifest, rows, fields)
        if up:
            log = args.manifest.resolve().parents[2] / "5. Model + labeling" / "ROBOFLOW_UPLOADS.csv"
            with open(log, "a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=LOG_FIELDS).writerow({
                    "date": datetime.now().isoformat(timespec="seconds"), "workspace": args.workspace,
                    "project": args.project, "batch": pending[0][0]["roboflow_batch"],
                    "n_uploaded": up, "n_preannotated": 0, "n_blind": 0, "model": args.source,
                    "conf_cut": "", "cryoEV_commit": git_commit(), "manifest": str(args.manifest),
                    "note": f"relabel: labels replaced with {args.source} ({role}). {args.note}".strip()})
        print(f"relabeled {up}/{len(pending)}; manifest updated")


if __name__ == "__main__":
    main()
