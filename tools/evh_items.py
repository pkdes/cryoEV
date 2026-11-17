"""
eVHarmony items CSV (round 17): the candidates of one prediction set -> an items CSV for
`python -m annotation.object_set build --line-col line --first-items <gold> --app evharmony`.

    python tools/evh_items.py --ps <CryoAI-laptop>/predict/<ps> --out <items.csv> [--lo 0.25] [--hi 0.5]
        [--gold 40] [--seed 17]

Items: every polygon line of every image's prediction .txt with lo <= conf < hi (any class), read with
annotation/object_set.read_lines (the object key of contract C5: image, 0-based line). Columns: image, line, conf,
gold. The first --gold rows are a confidence-stratified pick (the candidates sorted by conf, cut into --gold equal
bins, one seeded pick per bin), marked gold=1; the rest follow in (image, line) order. Build with --first-items
<gold>, so the gold rows are the first cards for every reviewer; they become known answers once the set's owner
answers them (tools/evh_score.py scores everyone else against those answers). Read-only on the prediction set;
the same inputs and seed give the same bytes.
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from annotation import object_set as O  # noqa: E402
from annotation import prediction_package as pp  # noqa: E402

COLUMNS = ["image", "line", "conf", "gold"]


def candidates(ps_dir, lo=0.25, hi=0.5) -> list:
    """[(image, line, conf)] of every prediction line with lo <= conf < hi, in (image, line) order."""
    _req, _ps, _model, rows, pred_dir = pp._ps_inputs(Path(ps_dir))
    out = []
    for r in sorted(rows, key=lambda r: r["image"]):
        f = pred_dir / f"{r['image']}.txt"
        if not f.is_file():
            continue
        for o in O.read_lines(f):
            if o is not None and lo <= o["sr"] < hi:
                out.append((r["image"], o["line"], o["sr"]))
    return out


def stratified(cands: list, n: int, seed: int) -> list:
    """n items, one seeded pick from each of n equal-count bins of the candidates sorted by conf."""
    if n <= 0:
        return []
    if n >= len(cands):
        return list(cands)
    rng = random.Random(seed)
    srt = sorted(cands, key=lambda c: (c[2], c[0], c[1]))
    picks = []
    for k in range(n):
        a, b = k * len(srt) // n, (k + 1) * len(srt) // n
        picks.append(srt[rng.randrange(a, b)])
    return picks


def items_csv(cands: list, gold: int, seed: int) -> str:
    g = stratified(cands, gold, seed)
    gs = set(g)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    for c in g:
        w.writerow([c[0], c[1], f"{c[2]:.4f}", 1])
    for c in cands:
        if c not in gs:
            w.writerow([c[0], c[1], f"{c[2]:.4f}", 0])
    return buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ps", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lo", type=float, default=0.25)
    ap.add_argument("--hi", type=float, default=0.5)
    ap.add_argument("--gold", type=int, default=40)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args(argv)
    cands = candidates(a.ps, a.lo, a.hi)
    if not cands:
        print(f"[evh_items] no candidates with {a.lo} <= conf < {a.hi}", file=sys.stderr)
        return 2
    text = items_csv(cands, a.gold, a.seed)
    a.out.write_text(text, encoding="utf-8", newline="")
    ng = min(a.gold, len(cands))
    print(f"[OK] {a.out}: {len(cands)} candidates ({a.lo} <= conf < {a.hi}) on "
          f"{len({c[0] for c in cands})} images; first {ng} rows = gold (conf-stratified, seed {a.seed}). "
          f"Build with --first-items {ng} --app evharmony")
    return 0


if __name__ == "__main__":
    sys.exit(main())
