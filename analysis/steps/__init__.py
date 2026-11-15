"""Analysis steps: per-object measurements that run after prediction, inside Results (round 14, contract C5).

A step is pure: `run_image(ctx, objs, prior) -> {column: [value per obj]}`.
  ctx    {"image": stem, "nm_per_px": float | None, "ev_ids": set | None (None = every class is EV), "H", "W"}
  objs   [{"line": int | None, "cls": int, "conf": float | None, "poly_px": [[x, y], ...] | None, "mask": bool HxW,
          "key": str | None}]  (key = the review package's object key, g<id> / p<k> / n<uid>, when known)
         the image's ACCEPTED outlines (conf >= 0.5 on an untouched image, every listed object on a reviewed /
         edited one), in prediction-file order; `mask` is the outline rasterised exactly as results.compute counts it
  prior  {step name: {column: [values]}} of the steps that ran before this one on the same image
The values go to <ps>/results/objects.csv as `<step>.<column>`; the step's name, version, params and code sha are
stamped in <ps>/results/steps.json. There is no registry line: the stamp is the record.

    from analysis.steps import STEPS, DEFAULT_RECIPE, validate_recipe
    steps = validate_recipe([("roles", 1), ("nesting", 1)])     # -> [Step, Step]; ValueError on a bad recipe

Steps: roles v1, nesting v1, nesting_tree v1 (round 15: the ML / MV tree on nesting v1, pinned).
DEFAULT_RECIPE = [("roles", 1)]: what results.compute has always done. results.json gains a "steps" key only when the
recipe is not the default, so the default results are byte-identical to the counts before steps existed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class Step:
    """One analysis step (name, version) with the columns it adds per object.

    needs     step names that must come EARLIER in the recipe (their output is in `prior`)
    pins      {needed step name: version} when the step is built on one version of a needed step (round 15)
    params    the fixed settings of this version (a new setting = a new version), stamped in steps.json
    sources   repo-relative files whose text makes up code_sha (the step module and what it calls)
    summary   optional fn(rows) -> short text for the Results page: rows = [{col: value}] of the counted objects"""

    def __init__(self, name, version, needs, columns, params, runs_on="cpu", *, fn, sources=(), title="",
                 description="", summary=None, pins=None):
        self.name, self.version = str(name), int(version)
        self.needs, self.columns = tuple(needs), tuple(columns)
        self.pins = {str(k): int(v) for k, v in (pins or {}).items()}
        self.params, self.runs_on = dict(params), runs_on
        self._fn, self.sources = fn, tuple(sources)
        self.title, self.description, self._summary = title or self.name, description, summary

    def run_image(self, ctx, objs, prior) -> dict:
        out = self._fn(ctx, objs, prior)
        if set(out) != set(self.columns) or any(len(out[c]) != len(objs) for c in self.columns):
            raise ValueError(f"step {self.name} v{self.version}: returned columns {sorted(out)} with lengths "
                             f"{[len(v) for v in out.values()]}, expected {list(self.columns)} x {len(objs)}")
        return out

    def summary(self, rows) -> str:
        return self._summary(rows) if self._summary else ""

    def code_sha(self) -> str:
        """sha256 over the source files (line endings normalised, so a CRLF checkout gives the same sha)."""
        h = hashlib.sha256()
        for rel in self.sources:
            h.update(rel.encode("utf-8") + b"\0")
            h.update((REPO / rel).read_bytes().replace(b"\r\n", b"\n") + b"\0")
        return h.hexdigest()

    def stamp(self) -> dict:
        return {"step": self.name, "version": self.version, "params": dict(self.params), "code_sha": self.code_sha()}

    @property
    def key(self) -> tuple:
        return (self.name, self.version)

    def __repr__(self):
        return f"Step({self.name!r}, {self.version})"


def obj_mask(ctx, o):
    """The object's mask: o["mask"], else its poly_px rasterised like results.compute's prediction outlines
    (int truncation, cv2.fillPoly on an H x W canvas)."""
    if o.get("mask") is not None:
        return o["mask"]
    import cv2
    import numpy as np
    m = np.zeros((ctx["H"], ctx["W"]), np.uint8)
    pts = np.array([(int(x), int(y)) for x, y in (o.get("poly_px") or [])], np.int32)
    if len(pts):
        cv2.fillPoly(m, [pts], 1)
    return m > 0


def is_ev(ctx, o) -> bool:
    ev_ids = ctx.get("ev_ids")
    return ev_ids is None or o["cls"] in ev_ids


from analysis.steps import nesting as _nesting, nesting_tree as _tree, roles as _roles  # noqa: E402  (they import Step)

STEPS = {s.key: s for s in (_roles.ROLES_V1, _nesting.NESTING_V1, _tree.NESTING_TREE_V1)}
DEFAULT_RECIPE = [("roles", 1)]
REQUIRED = ("roles",)            # results.compute counts with the roles step's output


def _entry(e) -> tuple:
    if isinstance(e, dict):
        name, version = e.get("step"), e.get("version")
    elif isinstance(e, (list, tuple)) and len(e) == 2:
        name, version = e
    else:
        raise ValueError(f"recipe entry {e!r}: expected (step, version) or {{'step', 'version'}}")
    try:
        return str(name), int(version)
    except (TypeError, ValueError):
        raise ValueError(f"recipe entry {e!r}: version must be a whole number") from None


def validate_recipe(recipe=None, steps=None, require=REQUIRED) -> list:
    """The recipe as [Step], in order. None = DEFAULT_RECIPE. ValueError on an unknown step or version, a step listed
    twice, a step whose `needs` are not earlier in the recipe (or are there in another version than the step's
    `pins`), or a missing required step (`require`)."""
    steps = STEPS if steps is None else steps
    entries = [_entry(e) for e in (DEFAULT_RECIPE if recipe is None else recipe)]
    out, seen = [], {}
    for name, version in entries:
        st = steps.get((name, version))
        if st is None:
            have = sorted(f"{n} v{v}" for n, v in steps)
            raise ValueError(f"unknown step {name} v{version} (known: {', '.join(have)})")
        if name in seen:
            raise ValueError(f"step {name} is listed twice")
        missing = [n for n in st.needs if n not in seen]
        if missing:
            raise ValueError(f"step {name} v{version} needs {', '.join(missing)} earlier in the recipe")
        wrong = [f"{n} v{v}" for n, v in st.pins.items() if seen.get(n, v) != v]
        if wrong:
            raise ValueError(f"step {name} v{version} is built on {', '.join(wrong)}")
        seen[name] = version
        out.append(st)
    for name in require:
        if name not in seen:
            raise ValueError(f"the recipe must include {name} (the counts use it)")
    return out


def with_needs(keys, steps=None) -> list:
    """[(name, version)] plus every step they need (transitively; the pinned version, else the lowest known), in
    STEPS order: what the portal's recipe form turns a ticked step into (picking nesting_tree adds nesting)."""
    steps = STEPS if steps is None else steps
    want, todo = set(), [tuple(k) for k in keys]
    while todo:
        k = todo.pop()
        if k in want:
            continue
        want.add(k)
        st = steps.get(k)
        for n in (st.needs if st else ()):
            if any(w[0] == n for w in want):
                continue
            v = st.pins.get(n) or min((kv for kn, kv in steps if kn == n), default=None)
            if v is not None:
                todo.append((n, v))
    return [k for k in steps if k in want]


def is_default(steps) -> bool:
    return [s.key for s in steps] == [tuple(e) for e in DEFAULT_RECIPE]


def recipe_doc(steps) -> list:
    """steps.json `recipe`: [{step, version, params, code_sha}]."""
    return [s.stamp() for s in steps]


__all__ = ["Step", "STEPS", "DEFAULT_RECIPE", "REQUIRED", "validate_recipe", "with_needs", "is_default", "recipe_doc",
           "obj_mask", "is_ev"]
