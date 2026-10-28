"""
Save what Roboflow currently holds for the tiles in a MANIFEST: every image's
outlines, tags and split, exactly as returned by the API. Run it before
deleting or replacing images, so review edits made in Roboflow are not lost.

Writes, under --out:
  roboflow_state.json   one record per tile (source_file, roboflow_id, the raw
                        image info minus embeddings/urls)
  current.coco.json     the outlines as COCO (pixel coords of the image as
                        annotated; extra.name = the MANIFEST source_file), so
                        roboflow_relabel.py --coco can put them on another copy
                        of the same fields.

    python annotation/roboflow_snapshot.py --manifest ".../3d.../MANIFEST.csv" --grids 6 \
        --workspace WS --project PROJ --out ".../snapshot_YYYYMMDD" [--roles AU-eval]

The API key is read from ROBOFLOW_API_KEY.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roboflow_upload import read_manifest, user_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--grids", required=True, type=lambda s: {g.strip() for g in s.split(",")})
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--roles", default=None, type=lambda s: set(s.split(",")),
                    help="only these split_roles go into current.coco.json (state.json keeps all)")
    args = ap.parse_args()

    rows, _f = read_manifest(args.manifest)
    rows = [r for r in rows if r["grid"] in args.grids and r["roboflow_id"]]
    key = os.environ.get("ROBOFLOW_API_KEY") or user_env("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("set ROBOFLOW_API_KEY")
    from roboflow import Roboflow
    project = Roboflow(api_key=key).workspace(args.workspace).project(args.project)

    args.out.mkdir(parents=True, exist_ok=True)
    state, images, anns, cats = [], [], [], {}
    for i, r in enumerate(rows, 1):
        info = project.image(r["roboflow_id"])
        info.pop("embedding", None)
        info.pop("urls", None)
        state.append({"source_file": r["source_file"], "split_role": r["split_role"],
                      "roboflow_id": r["roboflow_id"], "info": info})
        a = info.get("annotation") or {}
        boxes = a.get("boxes") or []
        print(f"  {r['source_file']:24s} {len(boxes):4d} outlines  tags {info.get('tags')}")
        if args.roles and r["split_role"] not in args.roles:
            continue
        images.append({"id": i, "file_name": r["output_name"], "width": int(a.get("width") or 0),
                       "height": int(a.get("height") or 0), "extra": {"name": r["source_file"]}})
        for b in boxes:
            cid = cats.setdefault(b["label"], len(cats) + 1)
            pts = b.get("points")
            if not pts:
                x, y, w, h = (float(b[k]) for k in ("x", "y", "width", "height"))
                pts = [[x - w / 2, y - h / 2], [x + w / 2, y - h / 2], [x + w / 2, y + h / 2], [x - w / 2, y + h / 2]]
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cid,
                         "segmentation": [[float(v) for p in pts for v in p]], "iscrowd": 0})

    (args.out / "roboflow_state.json").write_text(json.dumps(state, indent=1))
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": c, "name": n} for n, c in cats.items()]}
    (args.out / "current.coco.json").write_text(json.dumps(coco))
    print(f"{len(state)} tiles saved; {len(images)} in current.coco.json "
          f"({len(anns)} outlines, classes {list(cats)}) -> {args.out}")


if __name__ == "__main__":
    main()
