"""Stand-in for inference/predict_models.py in the predict tests (no GPU, no torch). Same CLI and output layout:
<out>/<model>/predictions/<stem>.txt (one per image, as it goes) and <out>/summary.csv.

FAKE_PRED       = ok (default) | fail_after:<n> (exit 1 after n images) | no_summary (exit 0, no summary.csv)
                  | skip_one (exit 0, the last image has no file)
FAKE_PRED_DELAY = seconds per image (default 0)
FAKE_PRED_FROM  = folder of <stem>.txt to copy (the golden predictions); else one fixed line per image
FAKE_PRED_ARGV  = a file: this run's argv is appended to it as one JSON line
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--models", required=True)
for f in ("--conf", "--max-det", "--imgsz", "--device", "--iou"):
    ap.add_argument(f)
ap.add_argument("--no-overlays", action="store_true")
a = ap.parse_args()
if os.environ.get("FAKE_PRED_ARGV"):
    with open(os.environ["FAKE_PRED_ARGV"], "a", encoding="utf-8") as f:
        f.write(json.dumps(sys.argv[1:]) + "\n")
mode = os.environ.get("FAKE_PRED", "ok")
delay = float(os.environ.get("FAKE_PRED_DELAY") or 0)
src = Path(os.environ["FAKE_PRED_FROM"]) if os.environ.get("FAKE_PRED_FROM") else None
images = sorted(p for p in a.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
pred = a.out / a.models / "predictions"
pred.mkdir(parents=True, exist_ok=True)
rows = []
for i, img in enumerate(images, 1):
    if mode.startswith("fail_after:") and i > int(mode.split(":")[1]):
        print(f"fake predictor: CUDA error: out of memory at image {img.name}", flush=True)
        sys.exit(1)
    if mode == "skip_one" and i == len(images):
        break
    time.sleep(delay)
    if src is not None and (src / f"{img.stem}.txt").is_file():
        shutil.copyfile(src / f"{img.stem}.txt", pred / f"{img.stem}.txt")
    else:
        (pred / f"{img.stem}.txt").write_text("0 0.9000 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8")
    lines = [ln for ln in (pred / f"{img.stem}.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    confs = [float(ln.split()[1]) for ln in lines]
    rows.append(f"{img.name},{a.models},{len(lines)},{round(sum(confs) / len(confs), 4) if confs else ''}")
    print(f"fake predictor: {img.name} {len(lines)} detections", flush=True)
if mode != "no_summary":
    (a.out / "summary.csv").write_text("image,model_id,n_detections,mean_conf\n" + "".join(r + "\n" for r in rows),
                                       encoding="utf-8")
print(f"Done. {len(images)} images x 1 models -> {a.out}")
