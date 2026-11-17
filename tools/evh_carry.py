"""
eVHarmony answer carry-over (round 17c): one reviewer's answers in an old object set -> the same objects of a new one.

    python tools/evh_carry.py --from <old pred_obj_*> --to <new pred_obj_*> --who <reviewer id> [--attrs match,consent]
        [--do]

An object is the same when its (prediction-set image id, prediction line) is the same (contract C5; items.json of
both sets), so the new set must be built on the same prediction set or a derived copy that keeps every line where it
was (tools/evh_items.py --derive only appends decoy lines). Per carried answer the old decision {v, t, who, server_t}
is copied unchanged under the new object's key a:<new obj>:<attr>@<who>. `consent` is a per-reviewer answer kept on
the set's FIRST item (the page's reveal), so it goes to the new set's items[0] object, whatever object held it before.

Only into a key the new set does not have yet (an answer already there wins); values the new set's schema.json
refuses (review_server's check) are skipped and counted. Writes edits/<file>.json in review_server's record format
({file, pkg, fmt, edit, rx}, rx = the carry's time) and appends the carried rows to edits/_attr_log.jsonl (the
history). Decisions have no field for provenance (review_server.ATTR_FIELDS), so it goes in <new>/carried_answers.json
(server-only: never served). Run it before the set is served, or while nobody answers (no write lock is shared with
a running review server). A dry run unless --do. Read-only on --from.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from annotation import object_meta as om  # noqa: E402
from annotation import object_set as O  # noqa: E402
from annotation import review_server as rs  # noqa: E402
from labels import schemas as S  # noqa: E402

PROV = "carried_answers.json"
PER_ITEM0 = ("consent",)           # answered once per reviewer, on the set's first item


def answers(pkg, who: str, attrs=("match", "consent")) -> dict:
    """{(image, line, attr): decision} of one reviewer in an object set (decisions with v null are left out)."""
    pkg = Path(pkg)
    items = O.read_items(pkg)["items"]
    dec = om.decisions(pkg)
    out = {}
    for it in items:
        for a in attrs:
            d = dec.get(it["id"], {}).get(f"{a}@{who}")
            if isinstance(d, dict) and d.get("v") is not None:
                out[(it["image"], int(it["line"]), a)] = d
    return out


def answered_objects(pkg, who: str, attr: str = "match") -> set:
    """{(image, line)} the reviewer answered (any value) in an object set."""
    return {(img, ln) for (img, ln, a) in answers(pkg, who, (attr,)) if a == attr}


def plan(src, dst, who: str, attrs=("match", "consent")) -> dict:
    src, dst = Path(src), Path(dst)
    old = answers(src, who, attrs)
    new_items = O.read_items(dst)["items"]
    schema = O.read_schema(dst)
    obj_of = {(it["image"], int(it["line"])): it["obj"] for it in new_items}
    data = om.read_datajs(dst)
    file_of = {Path(i["file"]).stem: i["file"] for i in data["images"]}
    have = om.decisions(dst, data)
    id_of = {(it["image"], it["obj"]): it["id"] for it in new_items}
    first = new_items[0] if new_items else None
    put, missing, bad, exists = {}, [], [], []
    for (img, ln, a), d in sorted(old.items()):
        if a in PER_ITEM0:
            if first is None:
                missing.append((img, ln, a))
                continue
            img2, obj = first["image"], first["obj"]
        else:
            img2, obj = img, obj_of.get((img, ln))
            if obj is None:
                missing.append((img, ln, a))
                continue
        at = S.attr(schema, a)
        if at is None or not S.value_ok(at, d.get("v")):
            bad.append((img, ln, a, d.get("v")))
            continue
        if f"{a}@{who}" in have.get(id_of[(img2, obj)], {}):
            exists.append((img, ln, a))
            continue
        clean = {k: v for k, v in d.items() if k in rs.ATTR_FIELDS and k != "pending"}
        put.setdefault(file_of[img2], {})[f"a:{obj}:{a}@{who}"] = (clean, (img, ln, a))
    counts = {}
    for f in put.values():
        for (d, (_i, _l, a)) in f.values():
            counts[(a, d["v"])] = counts.get((a, d["v"]), 0) + 1
    return {"src": src, "dst": dst, "who": who, "old": old, "put": put, "missing": missing, "bad": bad,
            "exists": exists, "counts": counts}


def write(P) -> int:
    dst, edits = Path(P["dst"]), Path(P["dst"]) / "edits"
    edits.mkdir(exist_ok=True)
    pkg_id = dst.name
    now = int(time.time() * 1000)
    n = 0
    prov = []
    for file, keys in sorted(P["put"].items()):
        target = edits / f"{file}.json"
        saved = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        cur = saved.get("edit")
        dec = {k: d for k, (d, _src) in keys.items()}
        incoming = {"reviewed": False, "modified": False, "dec": dec, "t": max(d["t"] for d in dec.values())}
        merged = rs.merge_edits(cur, incoming, now)
        rx = rs.stamp_received(saved.get("rx") or {}, cur, merged, now)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps({"file": file, "pkg": pkg_id, "fmt": rs.RECORD_FMT, "edit": merged, "rx": rx}),
                       encoding="utf-8")
        tmp.replace(target)
        rs.append_attr_log(edits, file, list(dec), merged, cur)
        for k, (d, (img, ln, a)) in keys.items():
            prov.append({"key": k, "file": file, "from_image": img, "from_line": ln, "attr": a, "v": d["v"]})
            n += 1
    pf = dst / PROV
    old = json.loads(pf.read_text(encoding="utf-8")) if pf.is_file() else {"schema": "evh_carry/1", "runs": []}
    old["runs"].append({"t": now, "from": Path(P["src"]).name, "who": P["who"], "n": n, "tool": "tools/evh_carry.py",
                        "answers": prov})
    pf.write_text(json.dumps(old, indent=1), encoding="utf-8")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True, type=Path)
    ap.add_argument("--to", dest="dst", required=True, type=Path)
    ap.add_argument("--who", required=True)
    ap.add_argument("--attrs", default="match,consent")
    ap.add_argument("--do", action="store_true")
    a = ap.parse_args(argv)
    attrs = tuple(x for x in a.attrs.split(",") if x)
    try:
        P = plan(a.src, a.dst, a.who, attrs)
    except (OSError, ValueError, KeyError) as e:
        print(f"[evh_carry] {e}", file=sys.stderr)
        return 2
    vals = {}
    for (_i, _l, at), d in P["old"].items():
        vals[(at, d["v"])] = vals.get((at, d["v"]), 0) + 1
    print(f"{a.who} in {a.src.name}: {len(P['old'])} answers " + ", ".join(f"{k[0]}={k[1]}: {v}" for k, v in sorted(vals.items())))
    n = sum(len(k) for k in P["put"].values())
    print(f"to carry into {a.dst.name}: {n} (" + ", ".join(f"{k[0]}={k[1]}: {v}" for k, v in sorted(P['counts'].items()))
          + f"); not in the new set: {len(P['missing'])}; refused by its schema: {len(P['bad'])}; "
            f"already answered there: {len(P['exists'])}")
    if not a.do:
        print("[dry run] nothing written; add --do")
        return 0
    if not n:
        print("[OK] nothing to carry; nothing written")
        return 0
    w = write(P)
    print(f"[OK] carried {w} answers; provenance in {a.dst / PROV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
