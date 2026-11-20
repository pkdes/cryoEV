"""
eVHarmony scoreboard (round 17p): the set's answers -> D.scoreboard in its data.js, which the page shows behind a 🏆
button. The page sees only its reviewer's own answers (the server filters the rest), so the board is computed here.

    python tools/evh_scoreboard.py <pred_obj_*> [--exclude ID,...] [--hide NAME,...] [--top 20] [--do]
        [--every SECONDS]

Board = {t, rows: [{name, swipes}], others}:
  swipes   a reviewer's `match` answers with a value of the schema's options, on the set's items (practice cards are
           never saved; follow-ups like kind / why / outline_off never count; an undone answer has no value)
  listed   consent = yes (and none says no, on any item: evh_score's rule) AND a dating name (`nick`, schema v4)
  others   reviewers with >= 1 swipe, consent not "no", who are not listed (no name, or not answered consent yet)
  never    the set's owner (D.owner), --exclude ids, any id ending "-test", and anyone whose consent says "no" (they
           were told their answers won't count): not listed, not counted
Names: control and format characters (bidi, zero-width) removed, trimmed, at most 24 characters; a name holding "@"
or equal to a reviewer id, or listed in --hide (case-insensitive), is not shown: that reviewer counts in `others`.
Ids and emails never go into data.js. Rows: most swipes first, then name.

Writes (only with --do) the `scoreboard` key of <pkg>/data.js and nothing else (every other byte stays; under the
package's .meta_locks lock; temp file + os.replace), and only when rows or others changed (t = that change's time):
each data.js write makes the review server reload the set. Without --do it prints the board (names included: for the
set's owner to check before the first --do). --every N: repeat every N seconds (a service, ~/.cryoev/services.json);
with --do it never prints names. Reads the answers with annotation/object_set.table (the export's reader).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from annotation import object_set as O  # noqa: E402
from annotation import prediction_package as pp  # noqa: E402

NAME_MAX = 24
KEY = "scoreboard"


def clean_name(v) -> str:
    if not isinstance(v, str):
        return ""
    s = "".join(ch for ch in v if unicodedata.category(ch) not in ("Cc", "Cf", "Cs", "Co", "Cn"))
    return " ".join(s.split())[:NAME_MAX].strip()


def board(pkg, exclude=(), hide=(), top=20) -> dict:
    """{rows, others} from the set's answers (no time stamp)."""
    pkg = Path(pkg)
    tab = O.table(pkg)
    data = O._split_datajs((pkg / "data.js").read_bytes())[0]
    opts = {o["v"] for a in tab["schema"].get("object_attrs") or [] if a["id"] == "match" for o in a.get("options") or []}
    rids = sorted({c.split("@", 1)[1] for c in tab["columns"] if c.startswith("match@")})
    skip = {str(data.get("owner") or "")} | set(exclude)
    hide = {h.strip().lower() for h in hide if h.strip()}
    lower_ids = {r.lower() for r in rids}
    rows, others = [], 0
    for r in rids:
        if r in skip or r.endswith("-test"):
            continue
        n = sum(1 for row in tab["rows"] if row.get(f"match@{r}") in opts)
        cons = {row.get(f"consent@{r}") for row in tab["rows"]} - {None}
        if "no" in cons or n == 0:
            continue
        names = [clean_name(row.get(f"nick@{r}")) for row in tab["rows"] if row.get(f"nick@{r}")]
        name = names[0] if names else ""
        ok = (cons == {"yes"} and name and "@" not in name and name.lower() not in lower_ids
              and name.lower() not in hide)
        if ok:
            rows.append({"name": name, "swipes": n})
        else:
            others += 1
    rows.sort(key=lambda x: (-x["swipes"], x["name"].lower()))
    return {"rows": rows[:top], "others": others + max(0, len(rows) - top)}


def write(pkg, b: dict, lock_wait: float = 5.0) -> bool:
    """Write D.scoreboard when rows / others changed; -> True when written."""
    pkg = Path(pkg)
    lk = pp.meta_lock(pkg.parent, pkg.name, lock_wait)
    try:
        djs = pkg / "data.js"
        data, kw, tail = O._split_datajs(djs.read_bytes())
        old = data.get(KEY) or {}
        if old.get("rows") == b["rows"] and old.get("others") == b["others"]:
            return False
        before = {k: v for k, v in data.items() if k != KEY}
        data[KEY] = {"t": int(time.time() * 1000), "rows": b["rows"], "others": b["others"]}
        if {k: v for k, v in data.items() if k != KEY} != before:
            raise ValueError("internal: data.js would change beyond `scoreboard`; refused")
        O._write_atomic(djs, O.DATAJS_PREFIX + json.dumps(data, **kw).encode("utf-8") + tail)
        return True
    finally:
        lk.close()


def run_once(a) -> int:
    data = O._split_datajs((a.pkg / "data.js").read_bytes())[0]
    if data.get("app") != "evharmony" or data.get("kind") != "objects":
        print(f"[evh_scoreboard] {a.pkg.name}: not an eVHarmony object set", file=sys.stderr)
        return 2
    b = board(a.pkg, _list(a.exclude), _list(a.hide), a.top)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if not a.do:
        print(f"{a.pkg.name}: {len(b['rows'])} listed, {b['others']} others (dry run; add --do)")
        for i, r in enumerate(b["rows"], 1):
            print(f"  {i:>2}. {r['name']:<24} {r['swipes']}")
        return 0
    w = write(a.pkg, b)
    print(f"{stamp} {a.pkg.name}: {len(b['rows'])} listed, {b['others']} others, "
          f"{'written' if w else 'unchanged'}", flush=True)
    return 0


def _list(v) -> list:
    return [x.strip() for x in (v or "").split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkg", type=Path)
    ap.add_argument("--exclude", default="", help="reviewer ids never listed or counted (comma-separated)")
    ap.add_argument("--hide", default="", help="dating names never shown (comma-separated, case-insensitive)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--every", type=float, default=0, help="repeat every N seconds (with --do: a service)")
    ap.add_argument("--do", action="store_true")
    a = ap.parse_args(argv)
    if not a.every:
        return run_once(a)
    while True:
        try:
            run_once(a)
        except Exception as e:  # noqa: BLE001  a service keeps going (a busy lock, a file mid-replace)
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [evh_scoreboard] {type(e).__name__}: {e}", flush=True)
        time.sleep(max(30.0, a.every))


if __name__ == "__main__":
    sys.exit(main())
