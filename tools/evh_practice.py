"""
eVHarmony practice cards (round 17d): a few cards with known answers, shown before the real queue of an eVHarmony
object set (data.js D.practice), from the set's own items or from another review package.

    python tools/evh_practice.py <pred_obj_*> --pick SPEC [--pick SPEC ...] [--from-pkg <other review package>] [--do]

SPEC, in the order the cards show:
  IMAGE:LINE=ANSWER[/KIND][:NOTE]        an item of this set (prediction-set image id, prediction line; contract C5):
                                         it leaves the queue (D.items, items.json items) and the export, so it is never
                                         scored gold
  @IMAGE:OBJ=ANSWER[/KIND][:NOTE]        object OBJ (g<id> / p<k>) of image IMAGE of --from-pkg (r17d v2)
  @IMAGE@X,Y,R=ANSWER[/KIND][:NOTE]      a circle on image IMAGE of --from-pkg: centre X, Y and radius R as fractions of
                                         the image width / height / width (e.g. a decoy outline on ice or carbon)
e.g. --pick "@0001:g2=yes/multilayer:Two rings, one inside the other" --pick "@0055@0.65,0.145,0.02=no:Ice".
ANSWER is a match option of the set's schema (yes / no / unsure), KIND (only after yes) a `kind` option (v3: single /
multilayer); NOTE is one plain line (<= 200 characters) shown as feedback when a reviewer redoes the practice cards.

--from-pkg is only read. For each picked image its base JPEG (and its 2x image, and the picked item's native crop
when the source has them) is copied into this set's images/ as px_<random>.jpg / .hi.jpg / .c.jpg (no source name);
the card's image record (size, the one outline, no sample label) goes inside its D.practice entry (`im`), never into
D.images or D.items, so it is no item of the queue and the server refuses any answer on it.

The page (annotation/review_tool/index.html) shows the practice cards to a reviewer who has not answered consent yet;
their answers stay in page memory (never saved), then the reveal asks for consent.

Writes (only with --do; refused when the set has practice cards already: rebuild it instead):
  <pkg>/data.js         D.practice = [{id, image, obj, crop, answer[, kind][, note][, chi, chi_box][, im]}]; own items
                        leave D.items. Only `items` and `practice` change; every other byte stays (.meta_locks lock).
  <pkg>/items.json      own items move from `items` to `practice`; from --from-pkg: {from, image, obj, ring} provenance
  <pkg>/images/px_*     the copied images (from --from-pkg)
  <pkg>/review_package.json  n_items = the items left; n_practice
Refused: the set's FIRST item (it holds every reviewer's consent answer), an object that isn't an item / object, a
repeat, an answer or kind the schema doesn't have, a set that isn't an eVHarmony object set. A dry run unless --do.
Answers already given on a picked own item (e.g. carried by tools/evh_carry.py) stay in edits/ but leave the export.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import secrets
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from annotation import object_meta as om  # noqa: E402
from annotation import object_set as O  # noqa: E402
from annotation import prediction_package as pp  # noqa: E402

TAIL = r"\s*=\s*([a-z_]{1,20})(?:/([a-z_]{1,20}))?(?::(.*))?$"
PICK_RE = re.compile(r"^\s*([A-Za-z0-9_.-]{1,80}):(\d{1,6})" + TAIL, re.S)
EXT_OBJ_RE = re.compile(r"^\s*@([A-Za-z0-9_.-]{1,80}):([gp]\d{1,6})" + TAIL, re.S)
EXT_RING_RE = re.compile(r"^\s*@([A-Za-z0-9_.-]{1,80})@([0-9.]+),([0-9.]+),([0-9.]+)" + TAIL, re.S)
NOTE_MAX = 200
MAX_PRACTICE = 10
RING_N = 24


def _note(t, g):
    note = " ".join((g or "").split())
    if len(note) > NOTE_MAX or re.search(r"[\x00-\x1f\x7f<>]", note):
        raise ValueError(f"--pick {t!r}: the note must be one plain line of <= {NOTE_MAX} characters, no < or >")
    return note


def parse_pick(t: str) -> dict:
    m = EXT_RING_RE.match(t or "")
    if m:
        x, y, r = (float(v) for v in m.group(2, 3, 4))
        if not (0 < x < 1 and 0 < y < 1 and 0 < r < 0.5):
            raise ValueError(f"--pick {t!r}: X, Y must be in (0, 1) and R in (0, 0.5)")
        return {"ext": True, "image": m.group(1), "obj": None, "circle": (x, y, r), "answer": m.group(5),
                "kind": m.group(6) or "", "note": _note(t, m.group(7))}
    m = EXT_OBJ_RE.match(t or "")
    if m:
        return {"ext": True, "image": m.group(1), "obj": m.group(2), "circle": None, "answer": m.group(3),
                "kind": m.group(4) or "", "note": _note(t, m.group(5))}
    m = PICK_RE.match(t or "")
    if not m:
        raise ValueError(f"--pick {t!r}: expected IMAGE:LINE=, @IMAGE:OBJ= or @IMAGE@X,Y,R= then ANSWER[/KIND][:NOTE]")
    return {"ext": False, "image": m.group(1), "line": int(m.group(2)), "answer": m.group(3), "kind": m.group(4) or "",
            "note": _note(t, m.group(5))}


def _opts(schema: dict, attr: str) -> list:
    for a in schema.get("object_attrs") or []:
        if a.get("id") == attr:
            return [str(o.get("v")) for o in a.get("options") or []]
    return []


def _circle(x, y, r, w, h) -> list:
    """A normalized ring of RING_N vertices: centre (x, y), radius r x width (round in pixels)."""
    out = []
    for k in range(RING_N):
        a = 2 * math.pi * k / RING_N
        out += [round(x + r * math.cos(a), 6), round(y + r * w / h * math.sin(a), 6)]
    return out


def _crop(ring) -> list:
    xs, ys = ring[0::2], ring[1::2]
    mx, my, d = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(max(xs) - min(xs), max(ys) - min(ys)) * 1.5
    return [round(max(0, mx - d), 5), round(max(0, my - d), 5), round(min(1, mx + d), 5), round(min(1, my + d), 5)]


def plan_ext(src, p) -> dict:
    """One --from-pkg pick -> {im (source image record), ring, src files, chi}; read only."""
    src = Path(src)
    sd = om.read_datajs(src)
    by = {}
    for im in sd.get("images") or []:
        by[im["file"]] = im
        by[Path(im["file"]).stem] = im
    im = by.get(p["image"])
    if im is None:
        raise ValueError(f"@{p['image']}: not an image of {src.name}")
    if not (src / im["src"]).is_file():
        raise ValueError(f"@{p['image']}: {im['src']} not found in {src.name}")
    chi = None
    if p["obj"]:
        o = p["obj"]
        rings = None
        if o.startswith("g"):
            g = next((g for g in im.get("gt") or [] if str(g.get("id")) == o[1:]), None)
            rings = g and g.get("p")
        else:
            k = int(o[1:])
            prs = im.get("pr") or []
            rings = prs[k].get("p") if k < len(prs) else None
        if not rings:
            raise ValueError(f"@{p['image']}:{o}: no such object in {src.name}")
        ring = rings[0]
        it = next((i for i in sd.get("items") or [] if Path(str(i.get("image"))).stem == Path(im["file"]).stem
                   and i.get("obj") == o), None)
        if it and it.get("chi") and it.get("chi_box") and (src / it["chi"]).is_file():
            chi = {"src": it["chi"], "box": it["chi_box"]}
    else:
        ring = _circle(*p["circle"], im["w"], im["h"])
    return {"im": im, "ring": ring, "chi": chi}


def plan(pkg, picks: list, src=None) -> dict:
    pkg = Path(pkg)
    data, _kw, _tail = O._split_datajs((pkg / "data.js").read_bytes())
    if data.get("kind") != "objects" or data.get("app") != "evharmony" or data.get("mode") == "tree":
        raise ValueError(f"{pkg.name}: not an eVHarmony object set (data.js app {data.get('app')!r})")
    if data.get("practice"):
        raise ValueError(f"{pkg.name}: already has {len(data['practice'])} practice cards; rebuild the set to change them")
    if not 1 <= len(picks) <= MAX_PRACTICE:
        raise ValueError(f"give 1-{MAX_PRACTICE} --pick")
    if any(p["ext"] for p in picks):
        if src is None:
            raise ValueError("a @ pick needs --from-pkg")
        if Path(src).resolve() == pkg.resolve():
            raise ValueError("--from-pkg is this set: pick its items as IMAGE:LINE")
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
        key = ("@" + p["image"], p["obj"] or p["circle"]) if p["ext"] else (p["image"], p["line"])
        name = f"@{p['image']}:{p['obj'] or p['circle']}" if p["ext"] else f"{p['image']}:{p['line']}"
        if key in seen:
            raise ValueError(f"{name} picked twice")
        seen.add(key)
        if p["answer"] not in answers:
            raise ValueError(f"{name}: answer {p['answer']!r} is not one of {answers}")
        if p["kind"] and (p["answer"] != "yes" or p["kind"] not in kinds):
            raise ValueError(f"{name}: kind {p['kind']!r} needs answer yes and one of {kinds or 'none'}")
        if p["ext"]:
            out.append({**p, "name": name, **plan_ext(src, p)})
            continue
        if key not in at:
            raise ValueError(f"{name} is not an item of {pkg.name}")
        n = at[key]
        if n == 0:
            raise ValueError(f"{name} is the set's first item (it holds the consent answers); pick another")
        out.append({**p, "name": name, "n": n, "item": its[n], "pub": pub[n]})
    own = [p for p in out if not p["ext"]]
    return {"pkg": pkg, "src": Path(src) if src else None, "picks": out, "n_items": len(its),
            "answered": _answered(pkg, data, [p["pub"]["id"] for p in own])}


def _answered(pkg, data, ids) -> dict:
    """{item id: [attr@rid, ...]} decisions already on the picked items (they leave the export)."""
    dec = om.decisions(pkg, data)
    return {i: sorted(k for k, d in dec.get(i, {}).items() if isinstance(d, dict) and d.get("v") is not None) for i in ids}


def _extra(p) -> dict:
    return {"answer": p["answer"], **({"kind": p["kind"]} if p["kind"] else {}), **({"note": p["note"]} if p["note"] else {})}


def _copy_ext(pkg: Path, src: Path, p, used: set) -> tuple:
    """Copy one external pick's files into <pkg>/images as px_<hex>*; -> (D.practice entry, items.json entry)."""
    while True:
        h = secrets.token_hex(4)
        if h not in used and not (pkg / "images" / f"px_{h}.jpg").exists():
            used.add(h)
            break
    im = p["im"]
    shutil.copyfile(src / im["src"], pkg / "images" / f"px_{h}.jpg")
    rec = {"file": f"px_{h}.jpg", "split": "", "w": im["w"], "h": im["h"], "src": f"images/px_{h}.jpg",
           "gt": [{"id": 1, "c": "EV", "p": [p["ring"]]}], "pr": [], "fg": 0, "fp": 0}
    if im.get("hi") and (src / im["hi"]).is_file():
        shutil.copyfile(src / im["hi"], pkg / "images" / f"px_{h}.hi.jpg")
        rec.update({"hi": f"images/px_{h}.hi.jpg", "hi_w": im.get("hi_w"), "hi_h": im.get("hi_h")})
    ent = {"id": h[:5], "image": f"px_{h}", "obj": "g1", "crop": _crop(p["ring"]), **_extra(p)}
    if p["chi"]:
        shutil.copyfile(src / p["chi"]["src"], pkg / "images" / f"px_{h}.c.jpg")
        ent.update({"chi": f"images/px_{h}.c.jpg", "chi_box": p["chi"]["box"]})
    ent["im"] = rec
    prov = {"from": src.name, "image": p["image"], "obj": p["obj"], "circle": p["circle"], "copied_as": f"px_{h}",
            **_extra(p)}
    return ent, prov


def write(P, lock_wait: float = 5.0):
    pkg = Path(P["pkg"])
    gone = {p["n"] for p in P["picks"] if not p["ext"]}
    used = set()
    ext = {id(p): _copy_ext(pkg, Path(P["src"]), p, used) for p in P["picks"] if p["ext"]}
    lk = pp.meta_lock(pkg.parent, pkg.name, lock_wait)
    try:
        djs = pkg / "data.js"
        raw = djs.read_bytes()
        data, kw, tail = O._split_datajs(raw)
        before = dict(data)
        pub = data["items"]
        data["items"] = [it for n, it in enumerate(pub) if n not in gone]
        data["practice"] = [ext[id(p)][0] if p["ext"] else {**p["pub"], **_extra(p)} for p in P["picks"]]
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
        items["practice"] = [ext[id(p)][1] if p["ext"] else {**p["item"], **_extra(p)} for p in P["picks"]]
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
    ap.add_argument("--pick", action="append", required=True, help="SPEC (repeatable, in card order)")
    ap.add_argument("--from-pkg", type=Path, help="r17d v2: the review package @ picks come from (read only)")
    ap.add_argument("--do", action="store_true")
    a = ap.parse_args(argv)
    try:
        P = plan(a.pkg, [parse_pick(t) for t in a.pick], a.from_pkg)
    except (ValueError, OSError, KeyError) as e:
        print(f"[evh_practice] {e}", file=sys.stderr)
        return 2
    n_own = sum(1 for p in P["picks"] if not p["ext"])
    print(f"{a.pkg.name}: {len(P['picks'])} practice cards ({n_own} from its {P['n_items']} items, "
          f"{len(P['picks']) - n_own} from {P['src'].name if P['src'] else '-'})")
    for p in P["picks"]:
        tail = f" = {p['answer']}" + (f"/{p['kind']}" if p["kind"] else "") + (f"  {p['note']}" if p["note"] else "")
        if p["ext"]:
            print(f"  {p['name']}{tail}" + ("  [+ native crop]" if p["chi"] else ""))
            continue
        ans = P["answered"].get(p["pub"]["id"]) or []
        print(f"  {p['name']} item {p['pub']['id']} (queue #{p['n'] + 1}){tail}"
              + (f"  [answers already here, leaving the export: {', '.join(ans)}]" if ans else ""))
    if not a.do:
        print("[dry run] nothing written; add --do")
        return 0
    write(P)
    print(f"[OK] {a.pkg / 'data.js'} practice, {O.ITEMS} practice, {pp.RECORD} n_items / n_practice")
    return 0


if __name__ == "__main__":
    sys.exit(main())
