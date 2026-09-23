"""Step 6: what does DINOv2 know that a ruler does not?

Every object already has a cheap geometric description -- diameter, area, elongation,
fill. If the embedding's cluster structure is fully explained by those, then DINOv2 is
an expensive caliper and the honest figure caption is "vesicles cluster by size". If
structure remains after geometry is removed, THAT residual is the finding.

Size is treated as real biological signal throughout, not as a confound. The question
is not "is it size?" but "what is here besides size?".

Two analyses:

(b) PROBE -- cross-validated logistic regression predicting held-out role from
    geometry alone / embedding alone / both. The gap between geometry and both is the
    incremental value of the embedding, as one number.

(c) RESIDUAL -- regress the embedding on geometry, keep the out-of-fold residual, then
    re-run the same UMAP + HDBSCAN on it. Clusters that survive are structure DINOv2
    sees independently of how big and how round the object is.

Out-of-fold predictions are used for the residual so it cannot be an overfitting
artifact.

Usage:
    python embedding/geometry_residual.py [--variant native] [--neighbors 30]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import SplineTransformer, StandardScaler, normalize
from sklearn.pipeline import make_pipeline

from embedding.paths import OUT, Progress, variant

SEED = 0
PCA_DIMS = 50


def geometry_matrix(df: pd.DataFrame):
    """Cheap, ruler-measurable descriptors, all already in the manifest.

    Diameter and area are the dominant pair (and near-redundant -- area ~ d^2), so
    both go in on a log scale. Aspect and fill add shape independent of scale.
    Splines let the fit be non-linear, so we are not crediting the embedding for
    structure a curved relationship with size would have explained.
    """
    d = df["long_side_px"].to_numpy(float)
    a = np.maximum(df["area_px"].to_numpy(float), 1.0)
    bw = np.maximum(df["bbox_w"].to_numpy(float), 1e-6)
    bh = np.maximum(df["bbox_h"].to_numpy(float), 1e-6)
    feats = np.column_stack([
        np.log(np.maximum(d, 1.0)),                  # diameter
        np.log(a),                                   # area
        np.log(np.maximum(bw, bh) / np.minimum(bw, bh)),   # elongation
        a / (bw * bh),                               # fill of bounding box
    ])
    names = ["log_diameter", "log_area", "log_aspect", "fill"]
    return feats, names


def probe(X, y, name, seed=SEED):
    """Cross-validated balanced accuracy of predicting y from X."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0,
                                           class_weight="balanced"))
    pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
    return balanced_accuracy_score(y, pred)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="")
    ap.add_argument("--data-from", default=None)
    ap.add_argument("--neighbors", type=int, default=30)
    ap.add_argument("--min-cluster-size", type=int, nargs="+", default=[30, 50])
    args = ap.parse_args()

    import umap

    V = variant(args.variant, args.data_from)
    out_dir = OUT / ("geometry_residual" + V["sfx"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(V["manifest"]).reset_index(drop=True)
    emb = np.load(V["emb"])
    assert len(emb) == len(df)

    G, gnames = geometry_matrix(df)
    E = PCA(n_components=PCA_DIMS, random_state=SEED).fit_transform(normalize(emb))

    print("variant   : %s" % V["name"])
    print("objects   : %d   embedding PCs: %d   geometry features: %s\n"
          % (len(df), E.shape[1], ", ".join(gnames)))

    # ---------------- (b) probes ---------------------------------------------
    print("=== (b) can each description predict the held-out labels? ===")
    print("cross-validated balanced accuracy; chance = 1/n_classes\n")
    probes = {}
    tasks = [
        ("role (3-class)", df["role"].to_numpy(), None),
        ("multilayer vs rest", (df["role"] == "multilayer").to_numpy(), None),
        ("inner-layer vs standalone", None,
         df["role"].isin(["inner-layer", "standalone"]).to_numpy()),
    ]
    rows = []
    for tname, y, sub in tasks:
        if sub is not None:
            y = (df.loc[sub, "role"] == "inner-layer").to_numpy()
            Gs, Es = G[sub], E[sub]
        else:
            Gs, Es = G, E
        n_cls = len(np.unique(y))
        r = {
            "task": tname, "n": int(len(y)), "chance": round(1.0 / n_cls, 3),
            "geometry": round(probe(Gs, y, "geom"), 3),
            "dinov2": round(probe(Es, y, "emb"), 3),
            "both": round(probe(np.hstack([Gs, Es]), y, "both"), 3),
        }
        r["gain_over_geometry"] = round(r["both"] - r["geometry"], 3)
        rows.append(r)
        print("%-28s n=%-5d chance %.3f | geometry %.3f | DINOv2 %.3f | both %.3f "
              "| gain %+.3f" % (tname, r["n"], r["chance"], r["geometry"],
                                r["dinov2"], r["both"], r["gain_over_geometry"]))
    probes = pd.DataFrame(rows)
    probes.to_csv(out_dir / "probe_scores.csv", index=False)

    # ---------------- (c) residual -------------------------------------------
    print("\n=== (c) remove everything geometry can explain, re-cluster the rest ===")
    spl = make_pipeline(StandardScaler(),
                        SplineTransformer(n_knots=6, degree=3, include_bias=False))
    Gx = spl.fit_transform(G)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 25))
    # out-of-fold fit, so the residual cannot be an overfitting artifact
    fitted = cross_val_predict(ridge, Gx, E, cv=KFold(5, shuffle=True,
                                                      random_state=SEED))
    R = E - fitted

    ss_tot = ((E - E.mean(0)) ** 2).sum()
    r2 = 1 - ((E - fitted) ** 2).sum() / ss_tot
    print("geometry explains %.1f%% of the embedding's variance (out-of-fold R2)"
          % (100 * r2))
    print("remaining %.1f%% is what DINOv2 sees beyond diameter/area/shape\n"
          % (100 * (1 - r2)))

    res_rows = []
    coords_cache = {}
    for tag, X in (("original", E), ("residual", R)):
        co = umap.UMAP(n_neighbors=args.neighbors, min_dist=0.1, metric="cosine",
                       n_components=2, random_state=SEED).fit_transform(X)
        coords_cache[tag] = co
        np.save(out_dir / ("umap_%s.npy" % tag), co)
        for mcs in args.min_cluster_size:
            lab = HDBSCAN(min_cluster_size=mcs).fit_predict(co)
            keep = lab >= 0
            nclu = len({c for c in lab[keep]})
            if nclu < 2:
                continue
            big = np.bincount(lab[keep]).max() / max(1, keep.sum())
            rec = {
                "space": tag, "min_cluster_size": mcs, "n_clusters": nclu,
                "largest_cluster_frac": round(float(big), 3),
                "noise_frac": round(float((~keep).mean()), 3),
                "role_ARI": round(adjusted_rand_score(df["role"][keep], lab[keep]), 4),
                "role_NMI": round(normalized_mutual_info_score(df["role"][keep], lab[keep]), 4),
                "category_NMI": round(normalized_mutual_info_score(
                    df["category"][keep], lab[keep]), 4),
                "batch_ARI": round(adjusted_rand_score(
                    df["session_grid"][keep], lab[keep]), 4),
            }
            res_rows.append(rec)
            print("%-9s mcs=%-4d clusters=%-3d big=%.2f noise=%.2f | role ARI %.3f "
                  "NMI %.3f | cat NMI %.3f | batch ARI %.3f"
                  % (tag, mcs, nclu, big, rec["noise_frac"], rec["role_ARI"],
                     rec["role_NMI"], rec["category_NMI"], rec["batch_ARI"]))
    res = pd.DataFrame(res_rows)
    res.to_csv(out_dir / "residual_scores.csv", index=False)

    # ---------------- figure --------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    for r, tag in enumerate(["original", "residual"]):
        co = coords_cache[tag]
        for c, (col, title) in enumerate([
                ("role", "held-out containment role"),
                ("session_grid", "acquisition batch"),
                (None, "object diameter (log px)")]):
            ax = axes[r, c]
            if col is None:
                sc = ax.scatter(co[:, 0], co[:, 1], s=3,
                                c=np.log(df["long_side_px"]), cmap="viridis",
                                linewidths=0)
                plt.colorbar(sc, ax=ax, fraction=0.046)
            else:
                vals = df[col].astype(str).to_numpy()
                cmap = plt.get_cmap("tab20")
                for i, v in enumerate(sorted(set(vals))):
                    m = vals == v
                    ax.scatter(co[m, 0], co[m, 1], s=3, color=cmap(i % 20),
                               label=str(v)[:20], linewidths=0)
                if len(set(vals)) <= 12:
                    ax.legend(fontsize=6, markerscale=3)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("%s -- %s" % (tag.upper(), title), fontsize=10)
    fig.suptitle("%s: embedding before and after removing what geometry explains "
                 "(geometry R2 = %.1f%%)" % (V["name"], 100 * r2), fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "residual_umap.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    (out_dir / "meta.json").write_text(json.dumps({
        "variant": V["name"], "geometry_features": gnames,
        "geometry_R2_out_of_fold": round(float(r2), 4),
        "n_objects": int(len(df)), "n_neighbors": args.neighbors,
    }, indent=2), encoding="utf-8")

    print("\n[OK] %s" % out_dir)


if __name__ == "__main__":
    main()
