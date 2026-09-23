"""Step 4: UMAP -> HDBSCAN sweep over the cached embeddings.

Reads embeddings.npy only -- never touches images, never sees a label. Saves BOTH
the reduced coordinates and the cluster assignments, so re-clustering with different
hyperparameters costs seconds and never re-runs the expensive embedding step.

Usage:
    python embedding/cluster.py [--neighbors 15 30 50] [--min-cluster-size 15 30 50 100]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from embedding.paths import OUT, Progress, umap_path, variant

PCA_DIMS = 50
SEED = 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--neighbors", type=int, nargs="+", default=[15, 30, 50])
    ap.add_argument("--min-cluster-size", type=int, nargs="+", default=[15, 30, 50, 100])
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--variant", default="")
    ap.add_argument("--data-from", default=None,
                    help="read crops/manifest from another variant")
    args = ap.parse_args()

    V = variant(args.variant, args.data_from)

    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    import umap

    df = pd.read_parquet(V["manifest"]).reset_index(drop=True)
    emb = np.load(V["emb"])
    assert len(emb) == len(df), "embeddings (%d) and manifest (%d) disagree" % (len(emb), len(df))
    print("embeddings: %s\n" % (emb.shape,))

    # L2-normalize, then PCA -- the standard UMAP pre-step; makes the sweep fast and
    # takes the cosine geometry of the CLS space seriously.
    x = normalize(emb)
    n_comp = min(PCA_DIMS, x.shape[0], x.shape[1])
    pca = PCA(n_components=n_comp, random_state=SEED)
    xp = pca.fit_transform(x)
    print("PCA %d dims, explained variance %.1f%%"
          % (n_comp, 100 * pca.explained_variance_ratio_.sum()))

    out = {"object_id": df["object_id"]}
    summary = []
    prog = Progress("cluster%s" % V["sfx"], len(args.neighbors) * (1 + len(args.min_cluster_size)), every=1)

    for nn in args.neighbors:
        t0 = time.time()
        coords = umap.UMAP(n_neighbors=nn, min_dist=args.min_dist, metric="cosine",
                           n_components=2, random_state=SEED).fit_transform(xp)
        np.save(umap_path(V["sfx"], nn), coords)
        print("\nUMAP nn=%-3d  %.1fs  -> %s"
              % (nn, time.time() - t0, umap_path(V["sfx"], nn).name))
        prog.update()

        for mcs in args.min_cluster_size:
            lab = HDBSCAN(min_cluster_size=mcs, min_samples=None).fit_predict(coords)
            key = "nn%d_mcs%d" % (nn, mcs)
            out[key] = lab
            n_clu = int(len(set(lab)) - (1 if -1 in lab else 0))
            noise = float((lab == -1).mean())
            sizes = np.bincount(lab[lab >= 0]) if n_clu else np.array([0])
            print("  mcs=%-4d clusters=%-3d noise=%4.1f%%  sizes[min/med/max]=%d/%d/%d"
                  % (mcs, n_clu, 100 * noise, sizes.min(), int(np.median(sizes)), sizes.max()))
            summary.append({"key": key, "n_neighbors": nn, "min_cluster_size": mcs,
                            "n_clusters": n_clu, "noise_frac": round(noise, 4)})
            prog.update()
    prog.finish()

    pd.DataFrame(out).to_parquet(V["clusters"], index=False)
    V["cluster_meta"].write_text(json.dumps({
        "pca_dims": n_comp, "seed": SEED, "min_dist": args.min_dist,
        "metric": "cosine", "configs": summary,
    }, indent=2), encoding="utf-8")

    print("\n[OK] %s  (%d configs)" % (V["clusters"], len(summary)))
    print("[OK] umap_nn*.npy coordinates cached -- re-clustering needs no re-embedding")


if __name__ == "__main__":
    main()
