"""
Re-split a Roboflow COCO export so every acquisition session (batch) is spread
across train/val/test in about the same proportions, with duplicates of one
micrograph never landing in two splits.

Batch = acquisition session, read from the filename (BATCH_RULES below; the
first matching rule wins). Images no rule matches are reported, not guessed.

Duplicate guard: Roboflow projects accumulate re-uploads of the same micrograph
under different names (e.g. "20260512-A549-EV__grid4__4_square17_hole25" and
"4_square17_hole25"). Copies are found by identity key (name after the last
"__") plus a 256-bit difference hash, and only one copy is kept (the
session-prefixed one unless --keep-bare). Lipo/VLDL tiles of one montage abut
but do not overlap (checked 2026-10-28: edge-strip correlation <= 0.25), so
they split per tile; --group-montage keeps a whole montage in one split.

Two modes:
  --history <v4 export>  S1 "comparable": an old image may go to test only if
                         it (and every copy of it) was in that export's test,
                         so a model trained on it has a leak-free test set.
  (omitted)              S2 "fresh": no history constraint.

  --propose   write split_assignment.csv + SPLIT_REPORT.md only, under
              'secondary analyses/split_proposals/<output-name>/'
  (default)   also write a Roboflow-layout export under 'training images/'
              that prepare_all_layer_singleclass.py reads unchanged.

    python data_utils/make_balanced_split.py --source-name "roboflow - 20261028 cryoai v5" \\
        --history "roboflow - 20260924 cryoai v4" --output-name "roboflow - 20261028 cryoai v5 S1" --propose
"""

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent.parent / "CryoAI"
SPLITS = ["train", "valid", "test"]

# (regex on the original filename, session label, sub-batch group in the regex or None)
BATCH_RULES = [
    (r"^g6_.*lipo_VLDL__(grid6_\d+)_r\dc\d_t\d+", "20261012 lipo+VLDL"),
    (r"^(grid6_\d+)_r\dc\d_t\d+", "20261012 lipo+VLDL"),
    (r"^20260126-MDA-MB-KH__", "20260126 MDA-MB-KH"),
    (r"^20260420-MDA-MB-RM__", "20260420 MDA-MB-RM"),
    (r"^20260506-HEK293-holey-carbon__", "20260506 HEK293 holey carbon"),
    (r"^20260512-A549-EV__", "20260512 A549"),
    (r"^4_square", "20260512 A549"),            # bare copies of the A549 grid4 images
    (r"^20260512-PMSC-EV__", "20260512 PMSC"),
    (r"_Jul30", "Jul30 (5760 px)"),
    (r"^[123]_square", "UNKNOWN bare 1_/2_/3_"),
    (r"^grid-\d_square", "UNKNOWN grid-N_"),
]
# Instance categories used to balance class mix across splits within a batch.
BALANCE_CATS = ["VLDL", "Low conf EV", "non-EV", "MV"]


def orig_name(file_name):
    return re.sub(r"\.rf\.[0-9A-Za-z]+\.\w+$", "", file_name)


def batch_of(name):
    for pattern, label in BATCH_RULES:
        m = re.search(pattern, name)
        if m:
            return label, (m.group(1) if m.groups() else None)
    return "UNMATCHED", None


def identity_key(name):
    name = re.sub(r"_(png|jpg)$", "", name)
    m = re.search(r"(grid6_\d+_r\dc\d_t\d+)", name)
    return m.group(1) if m else name.split("__")[-1]


def dhash(path, n=16):
    im = Image.open(path).convert("L").resize((n + 1, n), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def load_export(root):
    """One record per image across all splits of a Roboflow COCO export."""
    recs, cocos = [], {}
    for s in SPLITS:
        coco_path = root / s / "_annotations.coco.json"
        if not coco_path.exists():
            continue
        coco = json.load(open(coco_path, encoding="utf-8"))
        cocos[s] = coco
        cat_name = {c["id"]: c["name"] for c in coco["categories"]}
        counts = defaultdict(Counter)
        for a in coco["annotations"]:
            counts[a["image_id"]][cat_name[a["category_id"]]] += 1
        for im in coco["images"]:
            o = orig_name(im["file_name"])
            recs.append(dict(src_split=s, image=im, name=o, counts=counts[im["id"]],
                             path=root / s / im["file_name"]))
    return recs, cocos


def find_duplicates(recs, max_hamming):
    """Union-find over identity key and near-identical dhash."""
    parent = list(range(len(recs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_key = defaultdict(list)
    for i, r in enumerate(recs):
        by_key[identity_key(r["name"])].append(i)
    for idx in by_key.values():
        for j in idx[1:]:
            parent[find(j)] = find(idx[0])
    hashes = np.stack([r["hash"] for r in recs])
    for i in range(len(recs)):
        d = (hashes[i + 1:] != hashes[i]).sum(axis=1)
        for j in np.nonzero(d <= max_hamming)[0]:
            parent[find(i + 1 + j)] = find(i)
    clusters = defaultdict(list)
    for i in range(len(recs)):
        clusters[find(i)].append(i)
    return list(clusters.values())


def targets(n, ratios):
    if n < 3:
        return {"train": n, "valid": 0, "test": 0}
    n_val = max(1, round(n * ratios[1]))
    n_test = max(1, round(n * ratios[2]))
    return {"train": n - n_val - n_test, "valid": n_val, "test": n_test}


def assign_batch(units, ratios, rng, trials):
    """units: list of dicts with n, counts, test_ok. Random-restart greedy fill;
    keeps the assignment closest to the image targets, then to the class mix."""
    n = sum(u["n"] for u in units)
    tgt = targets(n, ratios)
    n_ok = sum(u["n"] for u in units if u["test_ok"])
    tgt["test"] = min(tgt["test"], n_ok)
    tgt["train"] = n - tgt["valid"] - tgt["test"]
    total = Counter()
    for u in units:
        total.update(u["counts"])
    best, best_score = None, None
    for _ in range(trials):
        order = units[:]
        rng.shuffle(order)
        fill = {"train": 0, "valid": 0, "test": 0}
        assign = {}
        for u in sorted(order, key=lambda u: not u["test_ok"]):     # eligible units first
            for s in (["test", "valid", "train"] if u["test_ok"] else ["valid", "train"]):
                if s == "train" or fill[s] + u["n"] <= tgt[s]:
                    assign[u["id"]] = s
                    fill[s] += u["n"]
                    break
        score = sum(abs(fill[s] - tgt[s]) for s in SPLITS) * 10.0
        for c in BALANCE_CATS:
            if total[c] == 0:
                continue
            for s in SPLITS:
                share = sum(u["counts"][c] for u in units if assign[u["id"]] == s) / total[c]
                score += abs(share - fill[s] / n)
        if best_score is None or score < best_score:
            best, best_score = dict(assign), score
    return best, tgt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-name", required=True, help="Roboflow export folder under 'training images/'")
    ap.add_argument("--output-name", required=True, help="New export folder under 'training images/'")
    ap.add_argument("--history", help="Previous export: S1 mode, test only from its test split")
    ap.add_argument("--ratios", default="0.6,0.15,0.25")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--max-hamming", type=int, default=20, help="dhash bits (of 256) for 'same image'")
    ap.add_argument("--keep-bare", action="store_true", help="Keep the bare-named copy of a duplicate")
    ap.add_argument("--propose", action="store_true", help="Report only; don't write the export")
    ap.add_argument("--group-montage", action="store_true", help="Keep all tiles of a montage in one split")
    args = ap.parse_args()
    ratios = [float(x) for x in args.ratios.split(",")]
    src = ROOT / "training images" / args.source_name
    out = ROOT / "training images" / args.output_name

    recs, cocos = load_export(src)
    for r in recs:
        r["hash"] = dhash(r["path"])
        r["batch"], r["montage"] = batch_of(r["name"])

    hist_split = {}
    if args.history:
        hrecs, _ = load_export(ROOT / "training images" / args.history)
        for r in hrecs:
            hist_split.setdefault(identity_key(r["name"]), set()).add(r["src_split"])

    # Duplicates: keep one copy per cluster.
    dropped, dup_notes = set(), []
    for cl in find_duplicates(recs, args.max_hamming):
        if len(cl) == 1:
            continue
        prefixed = [i for i in cl if "__" in recs[i]["name"]]
        bare = [i for i in cl if i not in prefixed]
        pref = (bare if args.keep_bare else prefixed) or cl
        keep = sorted(pref, key=lambda i: -sum(recs[i]["counts"].values()))[0]
        for i in cl:
            recs[i]["cluster"] = cl
            if i != keep:
                dropped.add(i)
        dup_notes.append((recs[keep], [recs[i] for i in cl if i != keep]))

    def test_ok(r):
        if not args.history:
            return True
        names = [recs[i]["name"] for i in r.get("cluster", [])] or [r["name"]]
        seen = set().union(*(hist_split.get(identity_key(n), set()) for n in names))
        return not seen or seen == {"test"}

    # Units: one per kept image, or one per montage with --group-montage.
    units_by_batch = defaultdict(dict)
    for i, r in enumerate(recs):
        if i in dropped:
            continue
        uid = f"{r['batch']}|{r['montage']}" if args.group_montage and r["montage"] else r["name"]
        u = units_by_batch[r["batch"]].setdefault(uid, dict(id=uid, n=0, counts=Counter(), test_ok=True, members=[]))
        u["n"] += 1
        u["counts"].update(r["counts"])
        u["test_ok"] &= test_ok(r)
        u["members"].append(i)

    rng = random.Random(args.seed)
    split_of, batch_rows = {}, []
    for batch in sorted(units_by_batch):
        units = list(units_by_batch[batch].values())
        assign, tgt = assign_batch(units, ratios, rng, args.trials)
        fill = Counter()
        for u in units:
            for i in u["members"]:
                split_of[i] = assign[u["id"]]
            fill[assign[u["id"]]] += u["n"]
        n_ok = sum(u["n"] for u in units if u["test_ok"])
        batch_rows.append((batch, sum(u["n"] for u in units), len(units), fill, n_ok))

    # Report.
    out_report = out if not args.propose else ROOT / "secondary analyses" / "split_proposals" / args.output_name
    out_report.mkdir(parents=True, exist_ok=True)
    with open(out_report / "split_assignment.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "batch", "montage", "split", "source_split", "history_split", "test_eligible",
                    *BALANCE_CATS, "n_objects"])
        for i, r in enumerate(recs):
            hs = "/".join(sorted(hist_split.get(identity_key(r["name"]), []))) or "new"
            w.writerow([r["name"], r["batch"], r["montage"] or "", "DROPPED-duplicate" if i in dropped else split_of[i],
                        r["src_split"], hs if args.history else "", test_ok(r),
                        *[r["counts"][c] for c in BALANCE_CATS], sum(r["counts"].values())])

    cls = defaultdict(Counter)
    for i, s in split_of.items():
        cls[s].update(recs[i]["counts"])
    lines = [f"# Split report: {args.output_name}", "",
             f"- Source: `{args.source_name}`, {len(recs)} images, {len(dropped)} duplicate copies dropped",
             f"- Mode: {'S1 comparable (history: `' + args.history + '`)' if args.history else 'S2 fresh'}; "
             f"ratios {args.ratios}; seed {args.seed}", "",
             "| Batch | Images | Units | Train | Val | Test | Test-eligible |", "|---|--:|--:|--:|--:|--:|--:|"]
    tot = Counter()
    for batch, n, nu, fill, n_ok in batch_rows:
        tot.update(fill)
        lines.append(f"| {batch} | {n} | {nu} | {fill['train']} | {fill['valid']} | {fill['test']} | {n_ok} |")
    lines.append(f"| **Total** | {sum(tot.values())} | | {tot['train']} | {tot['valid']} | {tot['test']} | |")
    cats = sorted({c for s in cls for c in cls[s]})
    lines += ["", "| Category | Train | Val | Test |", "|---|--:|--:|--:|"]
    lines += [f"| {c} | {cls['train'][c]} | {cls['valid'][c]} | {cls['test'][c]} |" for c in cats]
    lines += ["", "## Duplicates (kept <- dropped)", ""]
    for keep, drops in dup_notes:
        lines.append(f"- `{keep['name']}` ({keep['src_split']}) <- " +
                     ", ".join(f"`{d['name']}` ({d['src_split']})" for d in drops))
    (out_report / "SPLIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out_report}")
    if args.propose:
        return

    # Build a Roboflow-layout export holding only the kept images, re-split.
    for s in SPLITS:
        (out / s).mkdir(parents=True, exist_ok=True)
    merged_cats = next(iter(cocos.values()))["categories"]
    for s in SPLITS:
        images, anns = [], []
        for i, r in enumerate(recs):
            if split_of.get(i) != s:
                continue
            im = dict(r["image"], id=len(images))
            src_coco = cocos[r["src_split"]]
            src_id = r["image"]["id"]
            for a in src_coco["annotations"]:
                if a["image_id"] == src_id:
                    anns.append(dict(a, id=len(anns), image_id=im["id"]))
            images.append(im)
            shutil.copy2(r["path"], out / s / r["image"]["file_name"])
        coco = dict(next(iter(cocos.values())), images=images, annotations=anns, categories=merged_cats)
        json.dump(coco, open(out / s / "_annotations.coco.json", "w", encoding="utf-8"))
    print(f"Wrote export {out}")


if __name__ == "__main__":
    main()
