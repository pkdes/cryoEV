"""
eVHarmony practice cards (round 17d): move a few items of an eVHarmony object set out of its queue into D.practice.

    python tools/evh_practice.py <pred_obj_*> --pick "IMAGE:LINE=ANSWER[/KIND][:NOTE]" [--pick ...] [--do]

e.g. --pick "0006:103=yes/single:A closed round ring" --pick "0022:17=no:The edge of the carbon film". An object is
(prediction-set image id, prediction line), as in items.json (contract C5). ANSWER is a match option of the set's
schema (yes / no / unsure), KIND (only after yes) a `kind` option (v3: single / multilayer); the NOTE is one plain
line (<= 200 characters) shown as feedback when a reviewer redoes the practice cards.

The page (annotation/review_tool/index.html) shows the practice cards, in the order picked, to a reviewer who has not
answered consent yet; their answers stay in page memory (never saved), then the reveal asks for consent. Use objects
whose answer is known: the owner's clear yes / no answers, or a decoy (tools/evh_items.py --decoys: a known no).

Writes (only with --do; refused when the set has practice cards already: rebuild it instead):
  <pkg>/data.js         the picked items leave D.items; D.practice = [{id, image, obj, crop[, also], answer[, kind]
                        [, note]}] (only `items` and `practice` change; every other byte of data.js stays; under the
                        package's .meta_locks lock). The server accepts answers only on D.items objects, so a practice
                        card can't be saved even by hand.
  <pkg>/items.json      the picked items move from `items` to `practice` (with answer / kind / note): the export
                        (object_set export) has no row for them, so they are never scored gold.
  <pkg>/review_package.json  n_items = the items left; n_practice = how many moved.
Refused: the set's FIRST item (it holds every reviewer's consent answer), an object that isn't an item, a repeat,
an answer or kind the schema doesn't have, a set that isn't an eVHarmony object set. A dry run unless --do.
Answers already given on a picked item (e.g. carried by tools/evh_carry.py) stay in edits/ but leave the export.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from annotation import object_meta as om  # noqa: E402
from annotation import object_set as O  # noqa: E402
from annotation import prediction_package as pp  # noqa: E402

PICK_RE = re.compile(r"^\s*([A-Za-z0-9_.-]{1,80}):(\d{1,6})\s*=\s*([a-z_]{1,20})(?:/([a-z_]{1,20}))?(?::(.*))?$", re.S)
NOTE_MAX = 200
MAX_PRACTICE = 10


def parse_pick(t: str) -> dict:
    m = PICK_RE.match(t or "")
    if not m:
        raise ValueError(f"--pick {t!r}: expected IMAGE:LINE=ANSWER[/KIND][:NOTE]")
    note = " ".join((m.group(5) or "").split())
    if len(note) > NOTE_MAX or re.search(r"[\x00-\x1f\x7f<>]", note):
        raise ValueError(f"--pick {t!r}: the note must be one plain line of <= {NOTE_MAX} characters, no < or >")
    return {"image": m.group(1), "line": int(m.group(2)), "answer": m.group(3), "kind": m.group(4) or "", "note": note}


def _opts(schema: dict, attr: str) -> list:
    for a in schema.get("object_attrs") or []:
        if a.get("id") == attr:
            return [str(o.get("v")) for o in a.get("options") or []]
    return []


def plan(pkg, picks: list) -> dict:
    pkg = Path(pkg)
    data, _kw, _tail = O._split_datajs((pkg / "data.js").read_bytes())
    if data.get("kind") != "objects" or data.get("app") != "evharmony" or data.get("mode") == "tree":
        raise ValueError(f"{pkg.name}: not an eVHarmony object set (data.js app {data.get('app')!r})")
    if data.get("practice"):
        raise ValueError(f"{pkg.name}: already has {len(data['practice'])} practice cards; rebuild the set to change them")
    if not 1 <= len(picks) <= MAX_PRACTICE:
        raise ValueError(f"give 1-{MAX_PRACTICE} --pick")
    schema = O.read_schema(pkg)
    answers, kinds = _opts(schema, "match"), _opts(schema, "kind")
    items = O.read_items(pkg)
    its = items["items"]
    pub = data.get("items") or []
    if [(i["id"], i["obj"]) for i in its] != [(i["id"], i["obj"]) for i in pub]:
        raise ValueError(f"{pkg.name}: items.json and data.js items differ; refused")
    at = {(i["image"], int(i["line"])): n for n, i in enumerate(its)}
    out, seen = [], set()
    for p in picks:
        key = (p["image"], p["line"])
        if key in seen:
            raise ValueError(f"{key[0]}:{key[1]} picked twice")
        seen.add(key)
        if key not in at:
            raise ValueError(f"{key[0]}:{key[1]} is not an item of {pkg.name}")
        n = at[key]
        if n == 0:
            raise ValueError(f"{key[0]}:{key[1]} is the set's first item (it holds the consent answers); pick another")
        if p["answer"] not in answers:
            raise ValueError(f"{key[0]}:{key[1]}: answer {p['answer']!r} is not one of {answers}")
        if p["kind"] and (p["answer"] != "yes" or p["kind"] not in kinds):
            raise ValueError(f"{key[0]}:{key[1]}: kind {p['kind']!r} needs answer yes and one of {kinds or 'none'}")
        out.append({**p, "n": n, "item": its[n], "pub": pub[n]})
    return {"pkg": pkg, "picks": out, "n_items": len(its), "first_items": items.get("first_items") or 0,
            "answered": _answered(pkg, data, [p["pub"]["id"] for p in out])}


def _answered(pkg, data, ids) -> dict:
    """{item id: [attr@rid, ...]} decisions already on the picked items (they leave the export)."""
    dec = om.decisions(pkg, data)
    return {i: sorted(k for k, d in dec.get(i, {}).items() if isinstance(d, dict) and d.get("v") is not None) for i in ids}


def _extra(p) -> dict:
    return {"answer": p["answer"], **({"kind": p["kind"]} if p["kind"] else {}), **({"note": p["note"]} if p["note"] else {})}


def write(P, lock_wait: float = 5.0):
    pkg = Path(P["pkg"])
    gone = {p["n"] for p in P["picks"]}
    lk = pp.meta_lock(pkg.parent, pkg.name, lock_wait)
    try:
        djs = pkg / "data.js"
        raw = djs.read_bytes()
        data, kw, tail = O._split_datajs(raw)
        before = dict(data)
        pub = data["items"]
        data["items"] = [it for n, it in enumerate(pub) if n not in gone]
        data["practice"] = [{**p["pub"], **_extra(p)} for p in P["picks"]]
        rest, b2 = dict(data), dict(before)
        for k in ("items", "practice"):
            rest.pop(k, None)
            b2.pop(k, None)
        if rest != b2:
            raise ValueError("internal: data.js would change beyond `items` / `practice`; refused")
        ip = pkg / O.ITEMS
        items = json.loads(ip.read_text(encoding="utf-8"))
        its = items["items"]
        items["items"] = [it for n, it in enumerate(its) if n not in gone]
        items["practice"] = [{**p["item"], **_extra(p)} for p in P["picks"]]
        if items.get("first_items"):
            items["first_items"] -= sum(1 for n in gone if n < items["first_items"])
        rp = pkg / pp.RECORD
        rtext = rp.read_text(encoding="utf-8")
        rec = json.loads(rtext)
        rec["n_items"], rec["n_practice"] = len(items["items"]), len(items["practice"])
        O._write_atomic(djs, O.DATAJS_PREFIX + json.dumps(data, **kw).encode("utf-8") + tail)
        O._write_atomic(ip, (json.dumps(items, indent=1) + "\n").encode("utf-8"))
        O._write_atomic(rp, (json.dumps(rec, indent=1) + ("\n" if rtext.endswith("\n") else ""))
                        .encode("utf-8"))
    finally:
        lk.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkg", type=Path)
    ap.add_argument("--pick", action="append", required=True, help='"IMAGE:LINE=ANSWER[/KIND][:NOTE]" (repeatable)')
    ap.add_argument("--do", action="store_true")
    a = ap.parse_args(argv)
    try:
        P = plan(a.pkg, [parse_pick(t) for t in a.pick])
    except (ValueError, OSError, KeyError) as e:
        print(f"[evh_practice] {e}", file=sys.stderr)
        return 2
    print(f"{a.pkg.name}: {len(P['picks'])} practice cards out of {P['n_items']} items "
          f"({P['n_items'] - len(P['picks'])} left in the queue)")
    for p in P["picks"]:
        ans = P["answered"].get(p["pub"]["id"]) or []
        print(f"  {p['image']}:{p['line']} item {p['pub']['id']} (queue #{p['n'] + 1}) = {p['answer']}"
              + (f"/{p['kind']}" if p["kind"] else "") + (f"  {p['note']}" if p["note"] else "")
              + (f"  [answers already here, leaving the export: {', '.join(ans)}]" if ans else ""))
    if not a.do:
        print("[dry run] nothing written; add --do")
        return 0
    write(P)
    print(f"[OK] {a.pkg / 'data.js'} practice, {O.ITEMS} practice, {pp.RECORD} n_items / n_practice")
    return 0


if __name__ == "__main__":
    sys.exit(main())
