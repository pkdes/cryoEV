"""
eVHarmony scoring (round 17): each reviewer of an eVHarmony object set against the gold reviewer's answers.

    python -m annotation.object_set export --pkg <pred_obj_evh_*> --out evh.csv
    python tools/evh_score.py --csv evh.csv --gold <gold reviewer id> [--out scores.csv] [--min-gold 20] [--min-agree 0.8]

Reads the export's match@<rid> and consent@<rid> columns (schema tools/evh/ls_evh_match_v1.json). Gold items = the
rows where the gold reviewer answered yes or no (their "unsure" rows are not gold). Per other reviewer:
  n_gold     gold items they answered (any answer)
  n_unsure   of those, answered "unsure": excluded from agreement, reported
  n_scored   n_gold - n_unsure
  agree_pct  matches / n_scored x 100 (empty when n_scored is 0)
  consent    "yes" only when their consent cells say yes and none says no; absent or no = "no"
  pass       agree_pct >= --min-agree x 100 AND n_scored >= --min-gold (the owner's OK still decides what counts;
             a reviewer without consent = "yes" is never used, pass or not)
Prints a table; --out writes it as CSV. Read-only.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

ATTR, CONSENT = "match", "consent"
SCORED = ("yes", "no")
COLUMNS = ["reviewer", "n_gold", "n_unsure", "n_scored", "n_agree", "agree_pct", "consent", "pass"]


def _v(t) -> str:
    t = (t or "").strip()
    return t[1:] if t.startswith("'") else t          # object_set's formula guard


def read_export(text: str) -> tuple:
    """-> (rows, reviewers): rows as dicts; reviewers = the ids with a match@<rid> column."""
    rd = csv.DictReader(io.StringIO(text))
    cols = rd.fieldnames or []
    revs = [c.split("@", 1)[1] for c in cols if c.startswith(ATTR + "@")]
    if not revs:
        raise ValueError(f"no {ATTR}@<reviewer> columns: is this an eVHarmony set's export?")
    return list(rd), revs


def score(rows, reviewers, gold: str, min_gold: int = 20, min_agree: float = 0.8) -> list:
    if gold not in reviewers:
        raise ValueError(f"gold reviewer {gold!r} has no {ATTR}@{gold} column (reviewers: {', '.join(reviewers)})")
    golds = [(r, _v(r.get(f"{ATTR}@{gold}"))) for r in rows]
    golds = [(r, g) for r, g in golds if g in SCORED]
    out = []
    for rid in sorted(x for x in reviewers if x != gold):
        n_gold = n_unsure = n_agree = 0
        for r, g in golds:
            a = _v(r.get(f"{ATTR}@{rid}"))
            if not a:
                continue
            n_gold += 1
            if a == "unsure":
                n_unsure += 1
            elif a == g:
                n_agree += 1
        n_scored = n_gold - n_unsure
        pct = round(100.0 * n_agree / n_scored, 1) if n_scored else None
        cons = {_v(r.get(f"{CONSENT}@{rid}")) for r in rows} - {""}
        consent = "yes" if cons == {"yes"} else "no"
        ok = pct is not None and pct >= 100.0 * min_agree and n_scored >= min_gold
        out.append({"reviewer": rid, "n_gold": n_gold, "n_unsure": n_unsure, "n_scored": n_scored, "n_agree": n_agree,
                    "agree_pct": pct, "consent": consent, "pass": ok})
    return out


def to_csv(res: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    for r in res:
        w.writerow(["" if r[c] is None else ("1" if r[c] is True else "0" if r[c] is False else r[c]) for c in COLUMNS])
    return buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path, help="object_set export of the eVHarmony set")
    ap.add_argument("--gold", required=True, help="the gold reviewer's id (the set's owner)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--min-gold", type=int, default=20)
    ap.add_argument("--min-agree", type=float, default=0.8)
    a = ap.parse_args(argv)
    try:
        rows, revs = read_export(a.csv.read_text(encoding="utf-8-sig"))
        res = score(rows, revs, a.gold, a.min_gold, a.min_agree)
    except (ValueError, OSError) as e:
        print(f"[evh_score] {e}", file=sys.stderr)
        return 2
    n_gold = sum(1 for r in rows if _v(r.get(f"{ATTR}@{a.gold}")) in SCORED)
    print(f"gold: {n_gold} items answered yes/no by {a.gold}; pass = >= {a.min_agree:.0%} agreement on >= "
          f"{a.min_gold} scored gold (unsure excluded)")
    for r in res:
        print(f"  {r['reviewer']:<20} gold {r['n_gold']:>3}  unsure {r['n_unsure']:>3}  agree "
              f"{'-' if r['agree_pct'] is None else str(r['agree_pct']) + ' %':>8}  consent {r['consent']:<3}  "
              f"{'PASS' if r['pass'] else 'fail'}")
    if a.out:
        a.out.write_text(to_csv(res), encoding="utf-8", newline="")
        print(f"[OK] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
