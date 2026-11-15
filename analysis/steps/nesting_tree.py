"""nesting_tree v1: the per-layer ML / MV tree (rule "tree v2", PKD 11-14 FINAL) on top of nesting v1.

Ported (copied, not imported) from the ISEVxTech 2026 poster's scripts/nesting_tree.py (sha256
ea68994c11e9a1c3a28bece4e8e57bb92c39b1065d6c40e2267f1887c92c9225): `unit` / `cls` / `notation` and the per-pair
tests, with the poster's module globals turned into PARAMS passed in. The polar indices are analysis/steps/polar.py
(a word-for-word copy of the poster's nesting_polar_probe.py metrics).

Rule tree v2 (params): a child is a LAMELLA of its parent when gap_median <= g_ml_nm, offset_idx <= offset_max,
shape_idx <= shape_max and gap_min >= gap_min_nm; if several children pass, the largest (area_ratio) is the lamella.
A VESICLE UNIT = an outline plus its chain of lamellae (layers = chain length). Every other child of any outline in
the chain is an INNER VESICLE of the unit and is classified the same way, recursively. "Wide" = concentric (same
offset / shape test) and gap_min >= gap_min_nm but gap_median > g_ml_nm: an inner vesicle tagged "c:" (counted as
a layer only with wide_as_layer, which tree v2 leaves off).
Call per unit: ML<n> (n >= 2 layers), MV (1 layer holding inner vesicles), s (single), with the inner units' calls
in brackets, sorted: MV[ML2,s] = one membrane holding an ML2 vesicle and a single vesicle.

The tree is nesting v1's: each EV outline's parent is its `parent_line` (the smallest container, always a direct
parent); the per-pair gap stats are nesting v1's `children` entries (median / min in nm, area_ratio). An outline
directly inside two outlines that are not nested is placed under its smallest container only (the summary counts
such outlines). A child whose polar profile is None (an outline over fewer than 24 of 72 angle bins) is not
concentric, so it is an inner vesicle (the summary counts them).

Columns per object (blank for a non-EV outline or one with no pixels):
  unit           the unit's id = the unit's first outline's object ref (its package key g<id> / p<k> / n<uid>,
                 else L<line>, else #<index in the image's accepted outlines>)
  parent_unit    the id of the unit that holds this unit as an inner vesicle (blank for an outermost unit)
  role           outer (first outline of an outermost unit) / lamella / inner_vesicle (first outline of an inner unit)
  layer_index    1 for a unit's first outline, 2.. down its lamella chain
  vesicle_depth  0 for an outermost unit, +1 per inner-vesicle level
  offset_idx     polar a1/a0 of this outline vs its parent (3 dp; blank for a root or a None profile)
  shape_idx      polar a2/a0 (3 dp; same blanks)
  wide           1 for an inner unit that is concentric but wider than a lamella ("c:"), else 0
  call           the unit's call (e.g. MV[ML2,s]) on the unit's first outline; blank on lamella rows
With no nm/px the rule cannot run: unit, parent_unit, role, layer_index, vesicle_depth, wide and call are blank
(offset_idx / shape_idx are ratios and are still given).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from analysis.steps import Step, is_ev, obj_mask

PARAMS = {"rule": "tree_v2", "g_ml_nm": 16, "offset_max": 0.65, "shape_max": 0.35, "gap_min_nm": 1.5,
          "wide_as_layer": False}
COLUMNS = ("unit", "parent_unit", "role", "layer_index", "vesicle_depth", "offset_idx", "shape_idx", "wide", "call")


# ---- the poster's rule, params passed in (nesting_tree.py: concentric / is_lamella / is_wide / unit / cls / notation)

def concentric(e, p, wide=False):
    """tree v2 (polar mode): offset_idx <= offset_max and shape_idx <= shape_max. No polar profile -> not concentric."""
    if e.get("offset_idx") is None or e.get("shape_idx") is None:
        return False
    return e["offset_idx"] <= p["offset_max"] and e["shape_idx"] <= p["shape_max"]


def is_lamella(e, p):
    if p["wide_as_layer"] and is_wide(e, p):
        return True
    return e["gap_median_nm"] <= p["g_ml_nm"] and concentric(e, p) and e["gap_min_nm"] >= p["gap_min_nm"]


def is_wide(e, p):
    """concentric but wider than a lamella: even gap all round, > g_ml_nm."""
    return e["gap_median_nm"] > p["g_ml_nm"] and concentric(e, p, wide=True) and e["gap_min_nm"] >= p["gap_min_nm"]


def unit(node, kids, p):
    """kids: node -> list of edge dicts (child, area_ratio, gap stats). Returns {layers, inner: [units], chain}."""
    chain, inner_roots, cur = [node], [], node
    while True:
        es = kids.get(cur, [])
        lam = [e for e in es if is_lamella(e, p)]
        nxt = max(lam, key=lambda e: e["area_ratio"])["child"] if lam else None
        inner_roots += [(e["child"], is_wide(e, p)) for e in es if e["child"] != nxt]
        if nxt is None:
            break
        chain.append(nxt)
        cur = nxt
    return {"root": node, "layers": len(chain), "chain": chain,
            "inner": [dict(unit(c, kids, p), wide=w) for c, w in inner_roots]}


def cls(u):
    if u["layers"] >= 2:
        return "ML+MV" if u["inner"] else "ML"
    return "MV" if u["inner"] else "single"


def notation(u):
    head = f"ML{u['layers']}" if u["layers"] >= 2 else ("MV" if u["inner"] else "s")
    if not u["inner"]:
        return head
    return head + "[" + ",".join(sorted(("c:" if v.get("wide") else "") + notation(v) for v in u["inner"])) + "]"


def walk(u, depth=0):
    yield u, depth
    for v in u["inner"]:
        yield from walk(v, depth + 1)


def build(edges, p):
    """edges: list of dicts with parent, child + gap / polar stats (one image). Returns top-level units (the poster's
    build: roots = parents that are nobody's child, in sorted order)."""
    kids, childs = defaultdict(list), set()
    for e in edges:
        kids[e["parent"]].append(e)
        childs.add(e["child"])
    return [unit(r, kids, p) for r in sorted(set(kids) - childs)]


def class_of_call(call):
    """'MV[ML2,s]' -> 'MV' (the unit class from its call): single / ML / MV / ML+MV; None for a blank call."""
    if not call:
        return None
    head, inner = call.split("[", 1)[0], "[" in call
    if head.startswith("ML"):
        return "ML+MV" if inner else "ML"
    return "MV" if inner else "single"


# ---- the step ------------------------------------------------------------------------------------------------

def ref(o, k):
    """An object's id in this step's columns: its package key, else L<line>, else #<index>."""
    if o.get("key"):
        return str(o["key"])
    return f"L{o['line']}" if o.get("line") is not None else f"#{k}"


def polar_pair(par, ch, nm):
    """offset_idx / shape_idx (3 dp, the probe's rounding) for crops par (parent) and ch (child) on one canvas, as
    nesting_polar_probe.py builds it (the two masks' union box + 3 px); (None, None) when metrics() is None."""
    import numpy as np

    from analysis.steps import polar
    y0, x0 = min(par.y0, ch.y0) - 3, min(par.x0, ch.x0) - 3
    y1 = max(par.y0 + par.m.shape[0], ch.y0 + ch.m.shape[0]) + 3
    x1 = max(par.x0 + par.m.shape[1], ch.x0 + ch.m.shape[1]) + 3
    Pm = np.zeros((y1 - y0, x1 - x0), np.uint8)
    Cm = np.zeros_like(Pm)
    Pm[par.y0 - y0:par.y0 - y0 + par.m.shape[0], par.x0 - x0:par.x0 - x0 + par.m.shape[1]] = par.m
    Cm[ch.y0 - y0:ch.y0 - y0 + ch.m.shape[0], ch.x0 - x0:ch.x0 - x0 + ch.m.shape[1]] = ch.m
    m = polar.metrics(Pm, Cm, 1.0 if nm is None else float(nm))
    return (m["offset_idx"], m["shape_idx"]) if m else (None, None)


def structure(ctx, objs, prior, crops):
    """{child index: (parent index, nesting children entry)} for every contained EV outline. From nesting v1's
    output (parent_line + children) when every EV outline has its own line; else (a drawn / reshaped outline has no
    line, so lines cannot name parents) nesting v1's nest() is re-run on the crops (the same code)."""
    from analysis.steps import nesting as N
    idx = [k for k, c in enumerate(crops) if c is not None]
    lines = [objs[k].get("line") for k in idx]
    nest_out = (prior or {}).get("nesting")
    if nest_out is not None and None not in lines and len(set(lines)) == len(lines):
        by_line = {objs[k]["line"]: k for k in idx}
        out = {}
        for k in idx:
            pl = nest_out["parent_line"][k]
            if pl is None:
                continue
            par = by_line[pl]
            entry = next(c for c in json.loads(nest_out["children"][par]) if c["child"] == objs[k]["line"])
            out[k] = (par, entry)
        return out
    res = N.nest(crops, ctx["H"], ctx["W"], ctx.get("nm_per_px"))
    out = {}
    for k in idx:
        par = res[k]["parent"]
        if par is not None:
            out[k] = (par, res[par]["children"][res[par]["direct"].index(k)])
    return out


def run_image(ctx, objs, prior, params=None):
    from analysis.steps import nesting as N
    p = dict(PARAMS if params is None else params)
    nm = ctx.get("nm_per_px")
    crops = [N.crop(obj_mask(ctx, o)) if is_ev(ctx, o) else None for o in objs]
    out = {c: [None] * len(objs) for c in COLUMNS}
    par_of = structure(ctx, objs, prior, crops)
    edges = []
    for k, (par, entry) in sorted(par_of.items()):
        off, shp = polar_pair(crops[par], crops[k], nm)
        out["offset_idx"][k], out["shape_idx"][k] = off, shp
        edges.append({"parent": par, "child": k, "area_ratio": entry["area_ratio"],
                      "gap_median_nm": entry.get("gap_median_nm"), "gap_min_nm": entry.get("gap_min_nm"),
                      "offset_idx": off, "shape_idx": shp})
    if nm is None:
        return out                                   # no nm/px: no gap in nm, so no calls
    edges.sort(key=lambda e: (e["parent"], e["child"]))

    def emit(v, vdepth, parent_unit, role_first):
        uid = ref(objs[v["root"]], v["root"])
        for li, m in enumerate(v["chain"]):
            out["unit"][m], out["parent_unit"][m] = uid, parent_unit
            out["role"][m] = role_first if li == 0 else "lamella"
            out["layer_index"][m], out["vesicle_depth"][m] = li + 1, vdepth
            out["wide"][m] = int(bool(v.get("wide"))) if li == 0 else 0
            out["call"][m] = notation(v) if li == 0 else None
        for w in v["inner"]:
            emit(w, vdepth + 1, uid, "inner_vesicle")

    for u in build(edges, p):
        emit(u, 0, None, "outer")
    for k, c in enumerate(crops):                    # an EV outline in no pair: a single, outermost unit
        if c is not None and out["unit"][k] is None and k not in par_of:
            emit({"root": k, "layers": 1, "chain": [k], "inner": []}, 0, None, "outer")
    return out


def counters(rows) -> dict:
    """Over objects.csv-like rows (with image, nesting.children and this step's columns, un-prefixed): outlines
    directly inside two or more outlines (placed under the smallest only), and children with no polar profile."""
    seen = Counter()
    for r in rows:
        ch = r.get("nesting.children")
        if ch:
            for c in json.loads(ch):
                seen[(r.get("image"), c.get("child"))] += 1
    two = sum(1 for v in seen.values() if v > 1)
    no_polar = sum(1 for r in rows if r.get("role") in ("lamella", "inner_vesicle") and r.get("offset_idx") in (None, ""))
    return {"two_parents": two, "no_polar": no_polar}


def summary(rows) -> str:
    roots = [r for r in rows if r.get("role") == "outer" and r.get("call") not in (None, "")]
    if not roots:
        return "No EV outlines classified (no nm/px, or no EV outlines)."
    n = Counter(class_of_call(r["call"]) for r in roots)
    inner = [r for r in rows if r.get("role") == "inner_vesicle" and r.get("call") not in (None, "")]
    ni = Counter(class_of_call(r["call"]) for r in inner)
    c = counters(rows)
    out = (f"Outermost EVs: {n['ML']} ML, {n['MV']} MV, {n['ML+MV']} ML+MV, {n['single']} single; "
           f"inner vesicles: {len(inner)} ({ni['ML'] + ni['ML+MV']} multilamellar)")
    if c["two_parents"] or c["no_polar"]:
        out += (f"; {c['two_parents']} outlines sit directly inside two outlines (placed under the smaller), "
                f"{c['no_polar']} children had no polar profile (inner vesicles)")
    return out


NESTING_TREE_V1 = Step("nesting_tree", 1, needs=("nesting",), pins={"nesting": 1}, columns=COLUMNS, params=PARAMS,
                       fn=run_image, sources=("analysis/steps/nesting_tree.py", "analysis/steps/polar.py",
                                              "analysis/steps/nesting.py"),
                       title="Lamellarity tree",
                       description="ML / MV call for every EV and every vesicle inside it (rule tree v2: a layer "
                                   "is a child with median gap <= 16 nm, offset <= 0.65, shape <= 0.35, min gap "
                                   ">= 1.5 nm; every other child is an inner vesicle). Adds Nesting.",
                       summary=summary)
