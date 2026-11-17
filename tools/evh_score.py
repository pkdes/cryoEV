"""
eVHarmony scoring (round 17): each reviewer of an eVHarmony object set against the gold reviewer's answers.

    python -m annotation.object_set export --pkg <pred_obj_evh_*> --out evh.csv
    python tools/evh_score.py --csv evh.csv --gold <gold reviewer id> [--decoys <items>_decoys.csv] [--out scores.csv]
        [--min-gold 20] [--min-agree 0.8]

Reads the export's match@<rid> and consent@<rid> columns (schema tools/evh/ls_evh_match_v1.json). Gold items = the
rows where the gold reviewer answered yes or no (their "unsure" and "dense" rows are not gold), plus (r17c) every
decoy of --decoys (tools/evh_items.py's off-package list, matched on the export's image + line): a known "no", no
owner answer needed (a decoy the owner answered "yes" stays "no" and is counted as a conflict). Per other reviewer:
  n_gold     gold items they answered (any answer)
  n_unsure   of those, answered "unsure": excluded from agreement, reported
  n_dense    of those, answered "dense" (v2 "Small & dense"): excluded from agreement, reported
  n_decoys   decoys they answered (any answer); decoy_no = how many of those they called "no"
  n_scored   n_gold - n_unsure - n_dense
  agree_pct  matches / n_scored x 100 (empty when n_scored is 0)
  consent    "yes" only when their consent cells say yes and none says no; absent or no = "no"
  pass       agree_pct >= --min-agree x 100 AND n_scored >= --min-gold (the owner's OK still decides what counts;
             a reviewer without consent = "yes" is never used, pass or not)
  kind_*     (v2) their "kind" answers on items they called yes: single / ml / mv / inner (every reviewer, the gold
             one too, in the printed kind table)
Prints a table; --out writes it as CSV. Read-only.

--control (r17c, before a go-live): scores three made-up reviewers on the same gold (nothing is written):
ctl_always_yes and ctl_always_no must FAIL and ctl_replay (the gold answers, decoys as no) must PASS; exit 3 if
not (the gold is unbalanced or too small: fix it before anyone is scored).
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

ATTR, CONSENT, KIND = "match", "consent", "kind"
SCORED = ("yes", "no")
KINDS = ("single", "ml", "mv", "inner")
COLUMNS = ["reviewer", "n_gold", "n_unsure", "n_scored", "n_agree", "agree_pct", "consent", "pass", "n_dense",
           "n_decoys", "decoy_no"] + [f"kind_{k}" for k in KINDS]


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


def read_decoys(text: str) -> set:
    """{(image, line)} of tools/evh_items.py's <items>_decoys.csv."""
    return {(r["image"].strip(), int(r["line"])) for r in csv.DictReader(io.StringIO(text))}


def _key(r) -> tuple:
    try:
        return (_v(r.get("image")), int(_v(r.get("line"))))
    except (TypeError, ValueError):
        return (_v(r.get("image")), None)


def gold_rows(rows, gold: str, decoys=frozenset()) -> tuple:
    """-> ([(row, truth)], n_decoys found, n_conflicts): the owner's yes / no answers, and every decoy as "no"."""
    out, nd, nc = [], 0, 0
    for r in rows:
        g = _v(r.get(f"{ATTR}@{gold}"))
        if _key(r) in decoys:
            nd += 1
            nc += g == "yes"
            out.append((r, "no"))
        elif g in SCORED:
            out.append((r, g))
    return out, nd, nc


def kinds(rows, rid: str) -> dict:
    out = {k: 0 for k in KINDS}
    for r in rows:
        k = _v(r.get(f"{KIND}@{rid}"))
        if k in out and _v(r.get(f"{ATTR}@{rid}")) == "yes":
            out[k] += 1
    return out


def score(rows, reviewers, gold: str, min_gold: int = 20, min_agree: float = 0.8, decoys=frozenset()) -> list:
    if gold not in reviewers:
        raise ValueError(f"gold reviewer {gold!r} has no {ATTR}@{gold} column (reviewers: {', '.join(reviewers)})")
    golds, _nd, _nc = gold_rows(rows, gold, decoys)
    out = []
    for rid in sorted(x for x in reviewers if x != gold):
        n_gold = n_unsure = n_dense = n_agree = n_dec = dec_no = 0
        for r, g in golds:
            a = _v(r.get(f"{ATTR}@{rid}"))
            if not a:
                continue
            n_gold += 1
            if _key(r) in decoys:
                n_dec += 1
                dec_no += a == "no"
            if a == "unsure":
                n_unsure += 1
            elif a == "dense":
                n_dense += 1
            elif a == g:
                n_agree += 1
        n_scored = n_gold - n_unsure - n_dense
        pct = round(100.0 * n_agree / n_scored, 1) if n_scored else None
        cons = {_v(r.get(f"{CONSENT}@{rid}")) for r in rows} - {""}
        consent = "yes" if cons == {"yes"} else "no"
        ok = pct is not None and pct >= 100.0 * min_agree and n_scored >= min_gold
        out.append({"reviewer": rid, "n_gold": n_gold, "n_unsure": n_unsure, "n_scored": n_scored, "n_agree": n_agree,
                    "agree_pct": pct, "consent": consent, "pass": ok, "n_dense": n_dense, "n_decoys": n_dec,
                    "decoy_no": dec_no, **{f"kind_{k}": v for k, v in kinds(rows, rid).items()}})
    return out


CTL = ("ctl_always_yes", "ctl_always_no", "ctl_replay")


def control(rows, gold: str, min_gold: int = 20, min_agree: float = 0.8, decoys=frozenset()) -> tuple:
    """-> (results of the three made-up reviewers, ok): always-yes / always-no must fail, the replay must pass."""
    golds, _nd, _nc = gold_rows(rows, gold, decoys)
    truth = {id(r): g for r, g in golds}
    rows2 = []
    for r in rows:
        x = {k: v for k, v in r.items() if k.split("@", 1)[-1] in (gold,) or "@" not in k}
        x[f"{ATTR}@ctl_always_yes"], x[f"{ATTR}@ctl_always_no"] = "yes", "no"
        x[f"{ATTR}@ctl_replay"] = truth.get(id(r), "")
        rows2.append(x)
    res = {r["reviewer"]: r for r in score(rows2, [gold, *CTL], gold, min_gold, min_agree, decoys)}
    ok = not res["ctl_always_yes"]["pass"] and not res["ctl_always_no"]["pass"] and res["ctl_replay"]["pass"]
    return [res[c] for c in CTL], ok


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
    ap.add_argument("--decoys", type=Path, help="r17c: tools/evh_items.py's <items>_decoys.csv (known Nopes)")
    ap.add_argument("--control", action="store_true", help="r17c: always-yes / always-no must fail, a replay pass")
    ap.add_argument("--min-gold", type=int, default=20)
    ap.add_argument("--min-agree", type=float, default=0.8)
    a = ap.parse_args(argv)
    try:
        rows, revs = read_export(a.csv.read_text(encoding="utf-8-sig"))
        dec = read_decoys(a.decoys.read_text(encoding="utf-8-sig")) if a.decoys else frozenset()
        res = score(rows, revs, a.gold, a.min_gold, a.min_agree, dec)
    except (ValueError, OSError, KeyError) as e:
        print(f"[evh_score] {e}", file=sys.stderr)
        return 2
    golds, nd, nc = gold_rows(rows, a.gold, dec)
    ny = sum(1 for _r, g in golds if g == "yes")
    print(f"gold: {len(golds)} items ({ny} yes, {len(golds) - ny} no): {len(golds) - nd} answered yes/no by {a.gold}"
          + (f" + {nd} decoys (known no; {len(dec) - nd} decoys not in the export; {nc} the owner called yes)"
             if a.decoys else "")
          + f"; pass = >= {a.min_agree:.0%} agreement on >= {a.min_gold} scored gold (unsure / dense excluded)")
    for r in res:
        print(f"  {r['reviewer']:<20} gold {r['n_gold']:>3}  unsure {r['n_unsure']:>3}  dense {r['n_dense']:>3}  agree "
              f"{'-' if r['agree_pct'] is None else str(r['agree_pct']) + ' %':>8}  decoys {r['decoy_no']}/{r['n_decoys']} no  "
              f"consent {r['consent']:<3}  {'PASS' if r['pass'] else 'fail'}")
    if rows and any(f"{KIND}@{x}" in rows[0] for x in revs):
        print("kind answers (on items each called yes): " + ", ".join(KINDS))
        for rid in revs:
            k = kinds(rows, rid)
            print(f"  {rid:<20} " + "  ".join(f"{n} {k[n]}" for n in KINDS))
    if a.out:
        a.out.write_text(to_csv(res), encoding="utf-8", newline="")
        print(f"[OK] {a.out}")
    if a.control:
        ctl, ok = control(rows, a.gold, a.min_gold, a.min_agree, dec)
        print("control (made-up reviewers on this gold):")
        for r in ctl:
            print(f"  {r['reviewer']:<20} scored {r['n_scored']:>3}  agree {r['agree_pct']} %  "
                  f"{'PASS' if r['pass'] else 'fail'}")
        print("[control OK] always-yes and always-no fail, the replay passes" if ok else
              "[control FAILED] the gold can't tell a lazy player from a careful one: do not score with it")
        return 0 if ok else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
