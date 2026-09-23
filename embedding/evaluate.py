"""Step 5: score the clusters against held-out labels -- the decision step.

This is the ONLY place labels are read. Two plots are drawn from identical UMAP
coordinates: one colored by acquisition batch, one by held-out label. That pairing
is what separates "the embedding found biology" from "the embedding found the
microscope session".

Decision criteria (from the experiment plan):
  * clusters track held-out labels (ARI well above the permutation baseline)
        -> the approach works; proceed to feature extraction and figure design
  * clusters track acquisition session instead
        -> batch effect, not failure; try per-session standardization first
  * clusters track neither
        -> frozen embeddings insufficient; consider continued pretraining / MAE

Usage:
    python embedding/evaluate.py [--montage-config nn30_mcs30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import mutual_info_score

from embedding.paths import OUT, Progress, umap_path, variant

# EVAL is resolved per-variant inside main()
N_PERM = 100
# held-out label columns, in reporting order
LABELS = ["category", "role", "layer_count", "cell_line"]
BATCH = "session_grid"


def scores(labels: np.ndarray, truth: np.ndarray, rng) -> dict:
    """ARI/NMI plus a permutation baseline, so 'above chance' is a number."""
    ari = adjusted_rand_score(truth, labels)
    nmi = normalized_mutual_info_score(truth, labels)
    perm = np.array([adjusted_rand_score(rng.permutation(truth), labels)
                     for _ in range(N_PERM)])
    permn = np.array([normalized_mutual_info_score(rng.permutation(truth), labels)
                      for _ in range(N_PERM)])
    sd = perm.std()
    return {
        "ARI": ari, "NMI": nmi,
        "ARI_chance": perm.mean(), "ARI_chance_sd": sd,
        "NMI_chance": permn.mean(),
        # how many permutation sds above chance -- the "meaningfully above" number
        "ARI_z": (ari - perm.mean()) / sd if sd > 1e-12 else np.nan,
    }


def paired_plot(coords, df, lab, key, out_path):
    """The core figure: same coordinates, colored by batch vs by held-out label."""
    keep = lab >= 0
    panels = [(BATCH, "colored by ACQUISITION BATCH\n(structure here = batch effect)"),
              ("category", "colored by HELD-OUT annotator category"),
              ("role", "colored by HELD-OUT containment role"),
              (None, "HDBSCAN clusters (%d found, %.0f%% noise)"
                     % (len(set(lab[keep])), 100 * (~keep).mean()))]

    fig, axes = plt.subplots(1, 4, figsize=(26, 6.6))
    for ax, (col, title) in zip(axes, panels):
        vals = lab if col is None else df[col].astype(str).to_numpy()
        cats = sorted(set(vals[keep])) if col is None else sorted(set(vals))
        cmap = plt.get_cmap("tab20")
        if col is None:
            ax.scatter(coords[~keep, 0], coords[~keep, 1], s=2, c="#dddddd",
                       label="noise", linewidths=0)
        for i, c in enumerate(cats):
            m = (vals == c) & (keep if col is None else np.ones(len(vals), bool))
            ax.scatter(coords[m, 0], coords[m, 1], s=3, color=cmap(i % 20),
                       label=str(c)[:22], linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if len(cats) <= 22:
            ax.legend(fontsize=5.5, markerscale=3, ncol=2, loc="best", framealpha=0.85)

    fig.suptitle("%s  --  identical UMAP coordinates in all four panels" % key, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=115, bbox_inches="tight")
    plt.close(fig)


def cluster_montage(df, lab, key, out_path, crops_dir, per_cluster=16, max_clusters=12):
    """A grid of member crops per cluster -- the fastest way to see what a cluster IS."""
    rng = np.random.default_rng(0)
    ids = [c for c in sorted(set(lab)) if c >= 0]
    sizes = {c: int((lab == c).sum()) for c in ids}
    ids = sorted(ids, key=lambda c: -sizes[c])[:max_clusters]
    if not ids:
        return

    rows = []
    for c in ids:
        idx = np.where(lab == c)[0]
        pick = rng.choice(idx, min(per_cluster, len(idx)), replace=False)
        tiles = []
        head = np.zeros((96, 150, 3), np.uint8)
        top = df.iloc[idx]["category"].value_counts()
        cv2.putText(head, "cluster %d" % c, (6, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(head, "n=%d" % sizes[c], (6, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(head, "%s %.0f%%" % (top.index[0][:12], 100 * top.iloc[0] / sizes[c]),
                    (6, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(head, "med %.0fpx" % df.iloc[idx]["long_side_px"].median(),
                    (6, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 200, 255), 1, cv2.LINE_AA)
        tiles.append(head)
        for i in pick:
            im = cv2.imread(str(crops_dir / (df.iloc[int(i)]["object_id"] + ".png")),
                            cv2.IMREAD_UNCHANGED)
            im = cv2.normalize(im.astype(np.float32), None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
            tiles.append(cv2.cvtColor(cv2.resize(im, (96, 96)), cv2.COLOR_GRAY2BGR))
        while len(tiles) < per_cluster + 1:
            tiles.append(np.zeros((96, 96, 3), np.uint8))
        rows.append(np.hstack(tiles))

    grid = np.vstack(rows)
    banner = np.zeros((30, grid.shape[1], 3), np.uint8)
    cv2.putText(banner, "%s -- %d largest clusters, random members. Green = dominant "
                "held-out category (for reading only; never an input)." % (key, len(ids)),
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), np.vstack([banner, grid]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--montage-config", default=None,
                    help="cluster key to montage; default = best ARI vs category")
    ap.add_argument("--variant", default="")
    ap.add_argument("--data-from", default=None,
                    help="read crops/manifest from another variant")
    ap.add_argument("--compare-to", default=None,
                    help="another variant's scores.csv to tabulate side by side "
                         "(use 'baseline' for the unmasked run)")
    args = ap.parse_args()

    V = variant(args.variant, args.data_from)
    EVAL = V["eval"]
    EVAL.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(V["manifest"]).reset_index(drop=True)
    cl = pd.read_parquet(V["clusters"])
    assert (cl["object_id"].to_numpy() == df["object_id"].to_numpy()).all(), \
        "clusters.parquet is not row-aligned with manifest.parquet"

    keys = [c for c in cl.columns if c != "object_id"]
    rng = np.random.default_rng(0)
    rows = []

    prog = Progress("evaluate%s" % V["sfx"], len(keys), every=1)
    for key in keys:
        lab = cl[key].to_numpy()
        nn = int(key.split("_")[0][2:])
        coords = np.load(umap_path(V["sfx"], nn))
        keep = lab >= 0

        for scope, mask in (("all", np.ones(len(lab), bool)), ("no_noise", keep)):
            if mask.sum() < 10 or len(set(lab[mask])) < 2:
                continue
            rec = {"config": key, "scope": scope, "n": int(mask.sum()),
                   "n_clusters": int(len({c for c in lab[mask] if c >= 0})),
                   "noise_frac": round(float((lab == -1).mean()), 4)}
            for col in LABELS + [BATCH]:
                s = scores(lab[mask], df[col].astype(str).to_numpy()[mask], rng)
                for k, v in s.items():
                    rec["%s_%s" % (col, k)] = round(float(v), 4)
            # --- is this just a size axis? -----------------------------------
            size_bin = pd.qcut(df["long_side_px"], 8, labels=False, duplicates="drop")
            rec["size_MI"] = round(float(mutual_info_score(
                size_bin.to_numpy()[mask], lab[mask])), 4)

            # The decisive control: hold object size constant and ask whether the
            # role signal survives. Bin into size sextiles, score ARI inside each,
            # average. If this collapses relative to the unconditional ARI, the
            # "biology" was really a size axis all along.
            band = pd.qcut(df["long_side_px"], 6, labels=False,
                           duplicates="drop").to_numpy()[mask]
            r_masked, l_masked = df["role"].to_numpy()[mask], lab[mask]
            per_band = [adjusted_rand_score(r_masked[band == b], l_masked[band == b])
                        for b in np.unique(band)
                        if len(set(l_masked[band == b])) > 1
                        and len(set(r_masked[band == b])) > 1]
            rec["role_ARI_size_controlled"] = round(float(np.mean(per_band)), 4)                 if per_band else np.nan

            # degenerate partitions (one cluster swallowing nearly everything) can
            # post a flattering ARI off a couple of tiny pure pockets -- flag them
            # so the verdict below does not get chosen by one.
            pos = l_masked[l_masked >= 0]
            rec["largest_cluster_frac"] = round(
                float(np.bincount(pos).max() / len(pos)), 4) if len(pos) else 1.0
            rows.append(rec)

        paired_plot(coords, df, lab, key, EVAL / ("umap_%s.png" % key))
        prog.update()
    prog.finish()

    sc = pd.DataFrame(rows)
    sc.to_csv(EVAL / "scores.csv", index=False)

    nn_ = sc[sc["scope"] == "no_noise"].copy()
    # Only non-degenerate partitions are eligible to define the verdict: at least 3
    # clusters, and no single cluster holding more than 80% of the objects.
    ok = nn_[(nn_["largest_cluster_frac"] <= 0.80) & (nn_["n_clusters"] >= 3)]
    if ok.empty:
        print("[WARN] every config is degenerate; falling back to the full set")
        ok = nn_
    n_degen = len(nn_) - len(ok)
    best_cat = ok.loc[ok["category_ARI"].idxmax()]
    best_role = ok.loc[ok["role_ARI"].idxmax()]
    best_batch = ok.loc[ok["session_grid_ARI"].idxmax()]

    mkey = args.montage_config or best_cat["config"]
    cluster_montage(df, cl[mkey].to_numpy(), mkey, EVAL / ("montage_%s.png" % mkey), V["crops"])

    # ---- verdict -------------------------------------------------------------
    cat_ari, role_ari = best_cat["category_ARI"], best_role["role_ARI"]
    sc_ari = best_role["role_ARI_size_controlled"]
    bat_ari = best_batch["session_grid_ARI"]
    bio_ari, bio_z = max(cat_ari, role_ari), max(best_cat["category_ARI_z"],
                                                 best_role["role_ARI_z"])
    size_ok = not np.isnan(sc_ari) and sc_ari >= 0.7 * role_ari
    if bio_ari > bat_ari and bio_z > 3:
        verdict = ("WORKS -- clusters track held-out biology (ARI %.3f, %.1f sd above "
                   "chance) far more strongly than acquisition batch (ARI %.3f). "
                   "Size-controlled role ARI %.3f vs unconditional %.3f: the signal %s "
                   "a size artifact. Proceed to feature extraction and figure design."
                   % (bio_ari, bio_z, bat_ari, sc_ari, role_ari,
                      "is NOT" if size_ok else "MAY BE"))
    elif bat_ari > bio_ari and bat_ari > 0.05:
        verdict = ("BATCH EFFECT -- clusters track acquisition session (ARI %.3f) more "
                   "than biology (ARI %.3f). Not a failure: try per-session "
                   "standardization before abandoning." % (bat_ari, bio_ari))
    else:
        verdict = ("NEITHER -- clusters track neither biology (ARI %.3f) nor batch "
                   "(ARI %.3f) meaningfully. Frozen embeddings look insufficient; "
                   "consider continued pretraining or the MAE route."
                   % (bio_ari, bat_ari))

    lines = [
        "# Embedding clustering -- results brief", "",
        "Frozen self-supervised embeddings, no labels used as input. %d objects, %d "
        "acquisition groups." % (len(df), df[BATCH].nunique()), "",
        "## Verdict", "", verdict, "",
        "## Best configurations (noise excluded)", "",
        "| vs | config | clusters | ARI | chance | sd above | NMI |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r, p in (("held-out category", best_cat, "category"),
                       ("containment role", best_role, "role"),
                       ("acquisition batch", best_batch, "session_grid")):
        lines.append("| %s | %s | %d | %.3f | %.3f | %.1f | %.3f |" % (
            name, r["config"], r["n_clusters"], r["%s_ARI" % p],
            r["%s_ARI_chance" % p], r["%s_ARI_z" % p], r["%s_NMI" % p]))
    lines += ["", "## Size confound", "",
              "Cluster-vs-diameter mutual information at the best-category config: "
              "**%.3f**. The decisive control is size-controlled role ARI (ARI scored "
              "inside size sextiles, then averaged): **%.3f** against an unconditional "
              "**%.3f**. A collapse here would mean the clusters were a size axis; "
              "parity means the morphology signal is real. %d degenerate config(s) were "
              "excluded from the verdict."
              % (best_cat["size_MI"], sc_ari, role_ari, n_degen), "",
              "## Artifacts", "",
              "- `scores.csv` -- every config, every label, both noise scopes",
              "- `umap_<config>.png` -- 4 panels from identical coordinates",
              "- `montage_%s.png` -- what the clusters actually contain" % mkey, ""]
    (EVAL / "RESULTS_BRIEF.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 78)
    print(verdict)
    print("=" * 78)
    print("\nbest vs category : %-12s ARI %.3f (%.1f sd above chance)  NMI %.3f"
          % (best_cat["config"], cat_ari, best_cat["category_ARI_z"], best_cat["category_NMI"]))
    print("best vs role     : %-12s ARI %.3f (%.1f sd)  NMI %.3f"
          % (best_role["config"], role_ari, best_role["role_ARI_z"], best_role["role_NMI"]))
    print("best vs BATCH    : %-12s ARI %.3f (%.1f sd)  NMI %.3f"
          % (best_batch["config"], bat_ari, best_batch["session_grid_ARI_z"],
             best_batch["session_grid_NMI"]))
    print("size MI at best-category config    : %.3f" % best_cat["size_MI"])
    print("role ARI size-controlled vs uncond : %.3f vs %.3f  (%s)"
          % (sc_ari, role_ari, "size-independent" if size_ok else "SIZE-CONFOUNDED"))
    print("degenerate configs excluded        : %d" % n_degen)

    # ---- side-by-side against another variant --------------------------------
    if args.compare_to is not None:
        other_name = "" if args.compare_to == "baseline" else args.compare_to
        other_csv = variant(other_name)["eval"] / "scores.csv"
        if not other_csv.exists():
            print("\n[WARN] no scores.csv for variant %r -- skipping comparison"
                  % args.compare_to)
        else:
            o = pd.read_csv(other_csv)
            o = o[(o["scope"] == "no_noise") & (o["largest_cluster_frac"] <= 0.80)
                  & (o["n_clusters"] >= 3)]
            ob_role = o.loc[o["role_ARI"].idxmax()]
            ob_cat = o.loc[o["category_ARI"].idxmax()]
            ob_bat = o.loc[o["session_grid_ARI"].idxmax()]

            cmp_lines = [
                "", "## %s vs %s" % (V["name"], args.compare_to), "",
                "| metric | %s | %s | change |" % (args.compare_to, V["name"]),
                "|---|---|---|---|",
            ]
            hdr = "%-34s %10s %10s %10s" % ("metric", args.compare_to, V["name"], "change")
            print("\n" + hdr)
            print("-" * len(hdr))
            for label, a, b in (
                ("role ARI", ob_role["role_ARI"], role_ari),
                ("role ARI (size-controlled)",
                 ob_role["role_ARI_size_controlled"], sc_ari),
                ("role NMI", ob_role["role_NMI"], best_role["role_NMI"]),
                ("category ARI", ob_cat["category_ARI"], cat_ari),
                ("BATCH ARI (want ~0)", ob_bat["session_grid_ARI"], bat_ari),
                ("size MI (want low)", ob_cat["size_MI"], best_cat["size_MI"]),
            ):
                d = b - a
                print("%-34s %10.3f %10.3f %+10.3f" % (label, a, b, d))
                cmp_lines.append("| %s | %.3f | %.3f | %+.3f |" % (label, a, b, d))
            (EVAL / "RESULTS_BRIEF.md").write_text(
                "\n".join(lines + cmp_lines) + "\n", encoding="utf-8")

    print("\n[GATE E] review -> %s" % EVAL)


if __name__ == "__main__":
    main()
