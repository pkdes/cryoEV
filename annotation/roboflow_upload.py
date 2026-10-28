"""
Upload bar-free training tiles to a Roboflow project, with model pre-annotations
on most of them and a blind hold-out on the rest, and record every upload.

Reads the MANIFEST.csv that Cryo-image-processing's export_detector_tiles.py
writes next to the tiles, and writes back into it, per tile:
  split_role        pre-annotated | blind-eval   (assigned once, never redrawn)
  roboflow_split    train (pre-annotated) | test (blind-eval); re-splits must keep
                    blind-eval out of training -- the tag and metadata say which
  roboflow_project  roboflow_batch  roboflow_id  uploaded_at
so "what is in Roboflow?" is answered from the manifest. Rows that already have
a roboflow_id are skipped, so a re-run only uploads what is missing. Each run
also appends one line to <work>/ROBOFLOW_UPLOADS.csv.

Why a blind hold-out: labelers accept near-right outlines, so labels started
from predictions are biased toward the model. The blind tiles go up WITHOUT
predictions and are labeled from scratch; scoring the frozen predictions
against them (analysis/evaluate_coco_export.py) is the honest number.

Pre-annotations are the frozen predict_models.py output (<pred-dir>/<stem>.txt,
`class conf x1 y1 ...`, normalized) cut at --conf-cut, conf column dropped:
a valid YOLO-seg label. Written to <work>/preannot_labels_<cut>/ and kept.

The API key is read from ROBOFLOW_API_KEY and never written anywhere.

    python annotation/roboflow_upload.py --manifest ".../3d.../MANIFEST.csv" \
        --pred-dir ".../predictions/v4_y8m_20260924/predictions" --grids 6 \
        --workspace WS --project PROJ --dry-run
    ... --limit 2 --note "test"      # then check in Roboflow
    ...                              # the rest
"""
import argparse
import csv
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RF_FIELDS = ["split_role", "roboflow_project", "roboflow_batch", "roboflow_split", "roboflow_id",
             "uploaded_at"]
LOG_FIELDS = ["date", "workspace", "project", "batch", "n_uploaded", "n_preannotated", "n_blind",
              "model", "conf_cut", "cryoEV_commit", "manifest", "note"]


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys()) + [f for f in RF_FIELDS if f not in rows[0]]
    for r in rows:
        for f in RF_FIELDS:
            if r.get(f) is None:
                r[f] = ""
    return rows, fields


def write_manifest(path, rows, fields):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def assign_roles(rows, frac, seed):
    """Blind-eval picks spread over montages; only rows with no role yet."""
    todo = [r for r in rows if not r["split_role"]]
    if not todo:
        return 0
    n_blind = round(frac * len(todo))
    by_mont = defaultdict(list)
    for r in todo:
        by_mont[r["montage"]].append(r)
    rng = random.Random(seed)
    for lst in by_mont.values():
        rng.shuffle(lst)
    picked, monts = [], sorted(by_mont)
    while len(picked) < n_blind:
        for m in monts:
            if by_mont[m] and len(picked) < n_blind:
                picked.append(by_mont[m].pop())
    ids = {id(r) for r in picked}
    for r in todo:
        r["split_role"] = "blind-eval" if id(r) in ids else "pre-annotated"
    return len(todo)


def yolo_label(pred_txt, cut, dst):
    lines = []
    for ln in pred_txt.read_text().splitlines():
        p = ln.split()
        if len(p) >= 8 and float(p[1]) >= cut:
            lines.append(" ".join(["0"] + p[2:]))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def user_env(name):
    """A Windows user variable set with setx after this shell started."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, name)[0]
    except Exception:
        return None


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--pred-dir", required=True, type=Path, help="frozen predict_models.py predictions/")
    ap.add_argument("--model", default="v4_y8m_20260924")
    ap.add_argument("--grids", required=True, type=lambda s: {g.strip() for g in s.split(",")})
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--batch", default=None, help="default: <session>_nobar_<A/px>")
    ap.add_argument("--conf-cut", type=float, default=0.50)
    ap.add_argument("--blind-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--work", type=Path, default=None, help="default: <manifest>/../../../5. Model + labeling")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    rows, fields = read_manifest(args.manifest)
    session = args.manifest.resolve().parents[2].name
    work = args.work or args.manifest.resolve().parents[2] / "5. Model + labeling"
    selected = [r for r in rows if r["grid"] in args.grids]
    if not selected:
        sys.exit(f"no manifest rows for grids {sorted(args.grids)}")
    apx = selected[0]["output_a_px"].replace(".", "p")[:5]
    batch = args.batch or f"{session}_nobar_{apx}Apx"
    newly = assign_roles(selected, args.blind_frac, args.seed)
    if newly and not args.dry_run:
        write_manifest(args.manifest, rows, fields)

    pending = [r for r in selected if not r["roboflow_id"]]
    if args.limit:
        # a test upload takes one of each role
        pre = [r for r in pending if r["split_role"] == "pre-annotated"]
        bl = [r for r in pending if r["split_role"] == "blind-eval"]
        pending = (pre[:max(1, args.limit - 1)] + bl[:1])[:args.limit]

    per_mont = defaultdict(lambda: [0, 0])
    for r in selected:
        per_mont[r["montage"]][r["split_role"] == "blind-eval"] += 1
    print(f"Session {session}, grids {sorted(args.grids)}: {len(selected)} tiles "
          f"({sum(r['split_role'] == 'blind-eval' for r in selected)} blind-eval)")
    for m, (pre, bl) in sorted(per_mont.items()):
        print(f"  {m:14s} pre-annotated {pre:2d}  blind {bl}")
    print(f"Already uploaded: {len(selected) - len([r for r in selected if not r['roboflow_id']])}; "
          f"this run: {len(pending)} -> {args.workspace}/{args.project}, batch '{batch}'")

    lab_dir = work / f"preannot_labels_{str(args.conf_cut).replace('.', 'p')}"
    lab_dir.mkdir(parents=True, exist_ok=True)
    img_dir = args.manifest.parent
    plan = []
    for r in pending:
        img = img_dir / r["output_name"]
        pred = args.pred_dir / (Path(r["output_name"]).stem + ".txt")
        if not img.exists():
            sys.exit(f"missing image {img}")
        lab, n = None, 0
        if r["split_role"] == "pre-annotated":
            if not pred.exists():
                sys.exit(f"missing prediction {pred}")
            lab = lab_dir / pred.name
            n = yolo_label(pred, args.conf_cut, lab)
        tags = [session, f"g{r['grid']}", r["sample_label"].replace(" ", ""), f"{apx}Apx",
                f"{args.model}-preannot@{args.conf_cut:.2f}" if lab else r["split_role"]]
        plan.append((r, img, lab, n, tags))
        if args.dry_run:
            print(f"  DRY {r['split_role']:13s} {n:4d} outlines  {img.name}")

    if args.dry_run or not plan:
        print("dry run, nothing uploaded" if args.dry_run else "nothing to upload")
        return

    key = os.environ.get("ROBOFLOW_API_KEY") or user_env("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("set ROBOFLOW_API_KEY")
    from roboflow import Roboflow
    project = Roboflow(api_key=key).workspace(args.workspace).project(args.project)

    done = 0
    try:
        for r, img, lab, n, tags in plan:
            split = "test" if r["split_role"] == "blind-eval" else "train"
            resp = project.single_upload(
                image_path=str(img), annotation_path=str(lab) if lab else None,
                annotation_labelmap={0: "EV"} if lab else None,
                batch_name=batch, tag_names=tags, num_retry_uploads=3, split=split,
                metadata={"session": session, "grid": r["grid"], "montage": r["montage"],
                          "a_px": r["output_a_px"], "split_role": r["split_role"]})
            image_id = (resp.get("image") or {}).get("id")
            if not image_id:
                sys.exit(f"no image id for {img.name}: {resp}")
            r.update(roboflow_project=f"{args.workspace}/{args.project}", roboflow_batch=batch,
                     roboflow_split=split, roboflow_id=image_id, uploaded_at=datetime.now().isoformat(timespec="seconds"))
            done += 1
            print(f"  up {r['split_role']:13s} {n:4d} outlines  {img.name} -> {image_id}")
    finally:
        write_manifest(args.manifest, rows, fields)
        if done:
            log = work / "ROBOFLOW_UPLOADS.csv"
            new = not log.exists()
            with open(log, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
                if new:
                    w.writeheader()
                ups = [p for p in plan[:done]]
                w.writerow({"date": datetime.now().isoformat(timespec="seconds"),
                            "workspace": args.workspace, "project": args.project, "batch": batch,
                            "n_uploaded": done,
                            "n_preannotated": sum(1 for p in ups if p[2]),
                            "n_blind": sum(1 for p in ups if not p[2]),
                            "model": args.model, "conf_cut": args.conf_cut,
                            "cryoEV_commit": git_commit(), "manifest": str(args.manifest),
                            "note": args.note})
        print(f"uploaded {done}/{len(plan)}; manifest updated")


if __name__ == "__main__":
    main()
