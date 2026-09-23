"""Step 3: frozen ViT embedding of every crop. Inference only -- no training.

The model is never fine-tuned and never sees a label. We take the CLS token, which
is the model's own summary of the whole crop.

Default is facebook/dinov2-large (ungated). facebook/dinov3-vitl16-pretrain-lvd1689m
is gated: accept its license on the HF model page and set HF_TOKEN, then pass
--model facebook/dinov3-vitl16-pretrain-lvd1689m. Crops and normalization are
model-independent, so switching costs only this step onward.

Usage:
    python embedding/embed.py [--model ...] [--batch-size 64] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from embedding.paths import CROP_SIZE, OUT, QC, Progress, variant

DEFAULT_MODEL = "facebook/dinov2-large"
GATED_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def knn_qc(df, emb, out_path: Path, crops_dir: Path, n_query: int = 5, k: int = 8):
    """Query crops beside their nearest neighbours in embedding space (gate D).

    If the neighbours are not visually similar, the embedding is meaningless and
    clustering it would be wasted effort. Cheap to check, expensive to skip.
    """
    x = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(0)
    # spread queries across the size range so we see both small and large objects
    order = np.argsort(df["long_side_px"].to_numpy())
    queries = [int(order[int(q)]) for q in
               np.linspace(0.08, 0.95, n_query) * (len(order) - 1)]

    def tile(i, label, color):
        c = cv2.imread(str(crops_dir / (df.iloc[i]["object_id"] + ".png")), cv2.IMREAD_UNCHANGED)
        c = cv2.normalize(c.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        c = cv2.cvtColor(cv2.resize(c, (128, 128)), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(c, (0, 110), (128, 128), (0, 0, 0), -1)
        cv2.putText(c, label, (2, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
        cv2.rectangle(c, (0, 0), (127, 127), color, 2)
        return c

    rows = []
    for qi in queries:
        sims = x @ x[qi]
        nbrs = np.argsort(-sims)[1:k + 1]
        r = df.iloc[qi]
        tiles = [tile(qi, "%s %.0fpx" % (r["category"], r["long_side_px"]), (0, 200, 255))]
        tiles += [tile(int(j), "%s %.2f" % (df.iloc[int(j)]["category"], sims[j]),
                       (120, 255, 120)) for j in nbrs]
        rows.append(np.hstack(tiles))

    grid = np.vstack(rows)
    banner = np.zeros((34, grid.shape[1], 3), np.uint8)
    cv2.putText(banner, "GATE D -- col 1 = query (orange), cols 2-9 = 8 nearest neighbours "
                "by cosine similarity. Neighbours should look like the query.",
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([banner, grid]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--variant", default="")
    ap.add_argument("--data-from", default=None,
                    help="read crops/manifest from another variant (outputs stay under --variant)")
    ap.add_argument("--input-size", type=int, default=224,
                    help="resolution fed to the ViT. Upsampling here applies the SAME "
                         "factor to every crop, so it adds no size-dependent sharpness; "
                         "it just gives small objects more patches to be seen in")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    V = variant(args.variant, args.data_from)
    out_npy = V["emb"]
    if out_npy.exists() and not args.force:
        sys.exit("[ABORT] %s exists -- this is the expensive step. Re-run with --force."
                 % out_npy)

    df = pd.read_parquet(V["manifest"]).reset_index(drop=True)
    norm = np.load(V["norm"])
    assert len(norm) == len(df), "crops_norm.npy (%d) and manifest (%d) disagree" % (
        len(norm), len(df))

    from transformers import AutoModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Variant: %s" % V["name"])
    print("Model  : %s" % args.model)
    print("Device : %s (%s)" % (dev, torch.cuda.get_device_name(0) if dev == "cuda" else "-"))

    t_load = time.time()
    try:
        model = AutoModel.from_pretrained(args.model, dtype=torch.float16).to(dev).eval()
    except Exception as e:
        if args.model == GATED_MODEL:
            sys.exit("[ABORT] %s is gated: accept its license on the HF model page and set "
                     "HF_TOKEN, or use the default %s.\n  %s"
                     % (GATED_MODEL, DEFAULT_MODEL, e))
        raise
    load_s = time.time() - t_load
    print("loaded in %.1fs\n" % load_s)

    mean, std = IMAGENET_MEAN.to(dev).half(), IMAGENET_STD.to(dev).half()

    embs = []
    prog = Progress("embed%s" % V["sfx"], len(df), every=64)
    t0 = time.time()
    with torch.inference_mode():
        for s in tqdm(range(0, len(norm), args.batch_size), desc="embed"):
            batch = torch.from_numpy(norm[s:s + args.batch_size]).to(dev).half()
            # grayscale -> 3 identical channels, then ImageNet standardization
            batch = batch.unsqueeze(1)
            if args.input_size != batch.shape[-1]:
                batch = torch.nn.functional.interpolate(
                    batch, size=(args.input_size, args.input_size),
                    mode="bilinear", align_corners=False)
            batch = batch.repeat(1, 3, 1, 1)
            batch = (batch - mean) / std
            out = model(pixel_values=batch).last_hidden_state[:, 0]   # CLS token
            embs.append(out.float().cpu().numpy())
            prog.update(len(batch))
    prog.finish()
    wall = time.time() - t0

    emb = np.concatenate(embs, axis=0)
    assert emb.shape[0] == len(df), "embedding rows != manifest rows"
    assert np.isfinite(emb).all(), "non-finite values in embeddings"
    np.save(out_npy, emb)

    meta = {
        "model": args.model, "fallback_from_gated": args.model != GATED_MODEL,
        "dtype": "float16 compute / float32 stored", "batch_size": args.batch_size,
        "token": "CLS", "shape": list(emb.shape), "device": dev,
        "input_size": args.input_size, "data_from": args.data_from,
        "load_seconds": round(load_s, 1), "inference_seconds": round(wall, 1),
        "crops_per_second": round(len(df) / wall, 1),
        "torch": torch.__version__,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if dev == "cuda" else None,
    }
    import transformers
    meta["transformers"] = transformers.__version__
    V["emb_meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n[OK] %s  %s" % (out_npy, emb.shape))
    print("[OK] %s" % V["emb_meta"])
    print("  inference %.1fs (%.0f crops/s), peak VRAM %s GB"
          % (wall, len(df) / wall, meta["peak_vram_gb"]))

    x = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, len(x), (2000, 2))
    print("\nSanity: self-similarity %.4f (want 1.0), random-pair mean %.3f (want << 1)"
          % (float(x[0] @ x[0]), float(np.mean(np.sum(x[pairs[:, 0]] * x[pairs[:, 1]], axis=1)))))

    qc_path = QC / ("embed_knn_qc%s.png" % V["sfx"])
    knn_qc(df, emb, qc_path, V["crops"])
    print("\n[GATE D] review -> %s" % qc_path)


if __name__ == "__main__":
    main()
