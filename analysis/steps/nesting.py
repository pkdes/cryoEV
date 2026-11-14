"""nesting v1: what each EV outline contains, and the edge-to-edge gap to each direct child.

Ported (copied, not imported) from the ISEVxTech 2026 poster's lamellarity label set script (gap_stats and the
direct-children rule), so the numbers match the lamellarity labelling items. NOT the older centroid rule.

Among the image's EV-class outlines with at least one pixel:
  contained   j is contained in i when j has fewer pixels than i and >= 99 % of j's pixels are inside i
              (no size-ratio filter: tight layers the roles 0.75 rule drops are counted here)
  direct      a contained j that is not contained in another of i's contained outlines
  gap         per direct child: the parent's distance transform (cv2.DIST_L2, mask 5) sampled at every pixel of the
              child's external contour (CHAIN_APPROX_NONE), minus 0.5 px, times nm/px; median / min / p90 / CV
Columns per object (blank for a non-EV outline or one with no pixels):
  n_contained        outlines contained in this one (any depth)
  n_direct_children  of those, the direct ones
  d_outer_nm         area-equivalent diameter (nm, 1 dp) of an outermost outline (contained in none); blank inside
  children           JSON list, one per direct child: child (its prediction line), d_child_nm, area_ratio,
                     gap_median_nm, gap_min_nm, gap_p90_nm, gap_cv (rounded as the poster's items.csv)
nm values are blank (null in the JSON) when the image has no nm/px.
"""
from __future__ import annotations

import json
from collections import Counter

from analysis.steps import Step, is_ev, obj_mask

PARAMS = {"containment": 0.99, "size_ratio": None, "gap": "distanceTransform L2 5, contour - 0.5 px"}


class _O:
    """A mask as a crop: y0, x0, m (bool), like the poster's figs Obj."""
    __slots__ = ("y0", "x0", "m", "area")

    def __init__(self, y0, x0, m):
        self.y0, self.x0, self.m = y0, x0, m
        self.area = int(m.sum())

    def box(self):
        return self.y0, self.y0 + self.m.shape[0], self.x0, self.x0 + self.m.shape[1]


def crop(mask):
    """Full-image bool mask -> _O over its pixel box, or None with no pixel."""
    import numpy as np
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    return _O(y0, x0, np.asarray(mask[y0:y1, x0:x1], bool))


def inter(a, b):
    import numpy as np
    ay0, ay1, ax0, ax1 = a.box()
    by0, by1, bx0, bx1 = b.box()
    y0, y1, x0, x1 = max(ay0, by0), min(ay1, by1), max(ax0, bx0), min(ax1, bx1)
    if y1 <= y0 or x1 <= x0:
        return 0
    return int(np.logical_and(a.m[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0],
                              b.m[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]).sum())


def inside(j, i, thr=PARAMS["containment"]) -> bool:
    """j contained in i: fewer pixels and >= thr of j's pixels inside i."""
    return j is not i and j.area < i.area and inter(j, i) / max(j.area, 1) >= thr


def gap_stats(par, ch, H, W):
    """Edge-to-edge gap samples (px) from the child's external contour to the parent's edge (poster gap_stats)."""
    import cv2
    import numpy as np
    P_ = np.zeros((H + 2, W + 2), np.uint8)
    C_ = np.zeros((H + 2, W + 2), np.uint8)
    P_[par.y0 + 1:par.y0 + 1 + par.m.shape[0], par.x0 + 1:par.x0 + 1 + par.m.shape[1]] = par.m
    C_[ch.y0 + 1:ch.y0 + 1 + ch.m.shape[0], ch.x0 + 1:ch.x0 + 1 + ch.m.shape[1]] = ch.m
    ys, xs = np.nonzero(P_)
    sy, sx = slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)
    Pp, Cp = np.pad(P_[sy, sx], 2), np.pad(C_[sy, sx], 2)
    dt = cv2.distanceTransform(Pp, cv2.DIST_L2, 5)
    cs, _ = cv2.findContours(Cp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack([c[:, 0, :] for c in cs])
    g = dt[pts[:, 1], pts[:, 0]] - 0.5
    return g


def child_stats(par, ch, H, W, nm):
    """One direct child's entry in `children` (the poster's rounding); nm None -> nm values None."""
    import numpy as np
    out = {"area_ratio": round(ch.area / par.area, 3)}
    if nm is None:
        out.update(d_child_nm=None, gap_median_nm=None, gap_min_nm=None, gap_p90_nm=None, gap_cv=None)
        return out
    gg = gap_stats(par, ch, H, W) * float(nm)
    out.update(d_child_nm=round(2 * np.sqrt(ch.area / np.pi) * float(nm), 1),
               gap_median_nm=round(float(np.median(gg)), 2), gap_min_nm=round(float(gg.min()), 2),
               gap_p90_nm=round(float(np.percentile(gg, 90)), 2),
               gap_cv=round(float(np.std(gg) / max(np.mean(gg), 1e-6)), 3))
    return out


def nest(objs_crop, H, W, nm):
    """[_O | None] (one per object, None = skip) -> per index {n_contained, n_direct_children, d_outer_nm,
    direct: [indices], children: [stats dicts without 'child']} (None for a skipped index)."""
    import numpy as np
    ids = [k for k, o in enumerate(objs_crop) if o is not None]
    kids = {i: [j for j in ids if inside(objs_crop[j], objs_crop[i])] for i in ids}
    contained = {j for v in kids.values() for j in v}
    out = [None] * len(objs_crop)
    for i in ids:
        o, ks = objs_crop[i], kids[i]
        direct = [j for j in ks if not any(j != m and inside(objs_crop[j], objs_crop[m]) for m in ks)]
        out[i] = {"n_contained": len(ks), "n_direct_children": len(direct),
                  "d_outer_nm": (round(float(2 * np.sqrt(o.area / np.pi) * float(nm)), 1)
                                 if nm is not None and i not in contained else None),
                  "direct": direct, "children": [child_stats(o, objs_crop[j], H, W, nm) for j in direct]}
    return out


def run_image(ctx, objs, prior):
    crops = [crop(obj_mask(ctx, o)) if is_ev(ctx, o) else None for o in objs]
    res = nest(crops, ctx["H"], ctx["W"], ctx.get("nm_per_px"))
    out = {c: [None] * len(objs) for c in NESTING_V1.columns}
    for k, r in enumerate(res):
        if r is None:
            continue
        out["n_contained"][k] = r["n_contained"]
        out["n_direct_children"][k] = r["n_direct_children"]
        out["d_outer_nm"][k] = r["d_outer_nm"]
        out["children"][k] = json.dumps([{"child": objs[j].get("line"), **st}
                                         for j, st in zip(r["direct"], r["children"])])
    return out


def summary(rows) -> str:
    n = Counter(int(r["n_direct_children"]) for r in rows
                if r.get("n_direct_children") not in (None, "") and r.get("d_outer_nm") not in (None, ""))
    nested = sum(v for k, v in n.items() if k > 0)
    if not n:
        return "No EV outlines measured."
    return (f"{nested} outermost EVs contain another outline "
            f"({n[1]} with 1 direct child, {nested - n[1]} with 2 or more); {n[0]} contain nothing")


NESTING_V1 = Step("nesting", 1, needs=(), columns=("n_contained", "n_direct_children", "d_outer_nm", "children"),
                  params=PARAMS, fn=run_image, sources=("analysis/steps/nesting.py",),
                  title="Nesting",
                  description="What each EV outline contains (99 % of the inner outline's area inside, no size "
                              "limit), its direct children, and the edge-to-edge gap to each.",
                  summary=summary)
