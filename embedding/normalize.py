"""Step 2: contrast-normalize every crop against ONE reference micrograph.

The normalization is locked to a single reference image's percentiles and applied
identically to all crops. It is deliberately NOT per-crop: per-crop normalization
would rescale every object to fill the same intensity range, erasing exactly the
contrast and density differences that might carry the morphological signal we are
looking for.

Writes crops_norm.npy as [N, 224, 224] float16 -- one channel, not three. The
3-channel replication and ImageNet mean/std that the ViT wants are applied on the
GPU in embed.py; storing three identical copies on disk would triple the file for
nothing.

Usage:
    python embedding/normalize.py [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from embedding.paths import (
    CROP_SIZE, DEFAULT_SOURCE_NAME, OUT, QC, REFERENCE_WIDTH, ROOT,
    Progress, parse_name, variant,
)

P_LOW, P_HIGH = 1.0, 99.0


def pick_reference(df: pd.DataFrame, source_root: Path) -> Path:
    """Deterministic: first 1440-wide train micrograph by sorted filename.

    Chosen from the majority pixel scale so the reference is representative of most
    of the data rather than of the 4x-downsampled Jul30 set.
    """
    cand = sorted(df[(df["split"] == "train") & (df["native_width"] == REFERENCE_WIDTH)]
                  ["image_file"].unique())
    if not cand:
        cand = sorted(df[df["split"] == "train"]["image_file"].unique())
    return source_root / "train" / cand[0]


def build_qc(df, norm, raw_lo, raw_hi, out_path: Path, crops_dir: Path):
    """Before/after tiles plus per-session intensity histograms (gate C)."""
    rng = np.random.default_rng(0)
    cats = sorted(df["category"].unique())
    idx = []
    for i in range(8):
        sub = df[df["category"] == cats[i % len(cats)]]
        want = 5760 if i % 2 else 1440
        s2 = sub[sub["native_width"] == want]
        sub = s2 if len(s2) else sub
        idx.append(int(sub.index[int(rng.integers(0, len(sub)))]))

    sessions = sorted(df["session_grid"].unique())
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(3, 8, height_ratios=[1, 1, 1.3], hspace=0.35, wspace=0.12)

    for col, i in enumerate(idx):
        r = df.loc[i]
        raw = cv2.imread(str(crops_dir / (r["object_id"] + ".png")), cv2.IMREAD_UNCHANGED)

        ax = fig.add_subplot(gs[0, col])
        ax.imshow(raw, cmap="gray")
        ax.set_title("%s\n%s" % (str(r["session"])[:16], r["category"]), fontsize=7)
        ax.axis("off")
        if col == 0:
            ax.text(-0.22, 0.5, "BEFORE\n(raw 16-bit)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, rotation=90)

        ax = fig.add_subplot(gs[1, col])
        ax.imshow(norm[i], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if col == 0:
            ax.text(-0.22, 0.5, "AFTER\n(shared ref)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, rotation=90)

    ax = fig.add_subplot(gs[2, :])
    cmap = plt.get_cmap("tab20")
    for k, s in enumerate(sessions):
        vals = norm[df.index[df["session_grid"] == s].to_numpy()]
        if len(vals) == 0:
            continue
        sample = vals[np.random.default_rng(k).integers(0, len(vals), min(len(vals), 200))]
        ax.hist(sample.ravel(), bins=80, range=(-0.15, 1.15), histtype="step",
                density=True, label="%s (n=%d)" % (s, len(vals)), color=cmap(k % 20), lw=1.3)
    ax.set_title("Per-acquisition-group intensity after shared-reference normalization\n"
                 "These should NOT all overlap perfectly -- identical curves would mean "
                 "per-crop normalization leaked in and erased the batch signal.", fontsize=10)
    ax.set_xlabel("normalized intensity (0 = ref p%g, 1 = ref p%g)" % (P_LOW, P_HIGH))
    ax.legend(fontsize=6, ncol=5)

    fig.suptitle("Gate C -- normalization QC   |   reference raw range [%.0f, %.0f]"
                 % (raw_lo, raw_hi), fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    ap.add_argument("--variant", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    V = variant(args.variant)
    out_npy = V["norm"]
    if out_npy.exists() and not args.force:
        sys.exit("[ABORT] %s exists. Re-run with --force." % out_npy)

    df = pd.read_parquet(V["manifest"]).reset_index(drop=True)
    source_root = ROOT / "training images" / args.source_name

    ref_path = pick_reference(df, source_root)
    ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    if ref is None:
        sys.exit("[ABORT] could not read reference %s" % ref_path)
    ref16 = ref.astype(np.float32) * 257.0
    lo, hi = np.percentile(ref16, [P_LOW, P_HIGH])
    if hi <= lo:
        sys.exit("[ABORT] degenerate reference percentiles: %s %s" % (lo, hi))

    print("Reference micrograph : %s" % ref_path.name)
    print("Reference p%g / p%g   : %.1f / %.1f (16-bit scale)" % (P_LOW, P_HIGH, lo, hi))
    print("Applying this SINGLE mapping to all %d crops.\n" % len(df))

    norm = np.zeros((len(df), CROP_SIZE, CROP_SIZE), dtype=np.float16)
    prog = Progress("normalize%s" % V["sfx"], len(df), every=100)
    n_clipped = 0
    for i, oid in enumerate(tqdm(df["object_id"], desc="normalize")):
        c = cv2.imread(str(V["crops"] / (oid + ".png")), cv2.IMREAD_UNCHANGED).astype(np.float32)
        scaled = (c - lo) / (hi - lo)
        n_clipped += int(((scaled < 0) | (scaled > 1)).mean() > 0.5)
        # Clip generously, not to [0,1]: values outside the reference range are real
        # signal (denser/emptier fields), and hard-clipping them would flatten exactly
        # the contrast differences we want to keep.
        norm[i] = np.clip(scaled, -0.25, 1.25).astype(np.float16)
        prog.update()
    prog.finish()

    np.save(out_npy, norm)
    meta = {
        "reference_image": ref_path.name,
        "reference_split": "train",
        "percentile_low": P_LOW, "percentile_high": P_HIGH,
        "raw_low": float(lo), "raw_high": float(hi),
        "clip_range": [-0.25, 1.25],
        "n_crops": int(len(df)), "shape": list(norm.shape), "dtype": "float16",
        "per_crop_normalization": False,
    }
    V["norm_meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n[OK] %s  %s %s (%.0f MB)"
          % (out_npy, norm.shape, norm.dtype, out_npy.stat().st_size / 1e6))
    print("[OK] %s" % V["norm_meta"])
    print("\nNormalized intensity by acquisition group (mean +/- sd):")
    for s in sorted(df["session_grid"].unique()):
        v = norm[df.index[df["session_grid"] == s].to_numpy()].astype(np.float32)
        print("  %-34s n=%-5d mean=%6.3f sd=%5.3f" % (s, len(v), v.mean(), v.std()))
    print("\nCrops mostly outside the reference range: %d / %d" % (n_clipped, len(df)))

    qc_path = QC / ("normalize_qc%s.png" % V["sfx"])
    build_qc(df, norm.astype(np.float32), lo, hi, qc_path, V["crops"])
    print("\n[GATE C] review -> %s" % qc_path)


if __name__ == "__main__":
    main()
