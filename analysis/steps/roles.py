"""roles v1: the multilayer role of each EV outline, analysis.add_layer_roles.roles_for (the counting rule).

Columns per object: role (standalone | multilayer | inner-layer; blank for a non-EV outline) and layer_count (the
outlines a multilayer EV contains, 0 otherwise). Roles are assigned among the image's EV-class outlines only, in
prediction-file order, exactly as results.compute always has: containment 0.99, size ratio 0.75
(analysis.layer_count_stats, the validated pair; a test pins PARAMS equal to them).
"""
from __future__ import annotations

import os
import threading
from collections import Counter

from analysis.steps import Step, is_ev, obj_mask

PARAMS = {"containment": 0.99, "size_ratio": 0.75}

_FN = None
_LOCK = threading.Lock()


def load_roles_for():
    """analysis.add_layer_roles.roles_for, imported once per process under a lock.

    analysis.add_layer_roles takes its containment pair from analysis.layer_count_stats, whose module body resolves
    the CryoAI training folder for its own CLI defaults and raises ConfigError on a host without one. Roles never
    read that folder, so on that error the import is retried with $CRYOEV_CRYOAI_ROOT pointed at the repo, inside
    the lock and restored before the lock is released (at most once per process). The thresholds stay the chain's."""
    global _FN
    with _LOCK:
        if _FN is None:
            from data_utils import config
            try:
                from analysis.add_layer_roles import roles_for
            except config.ConfigError:
                old = os.environ.get("CRYOEV_CRYOAI_ROOT")
                os.environ["CRYOEV_CRYOAI_ROOT"] = str(config.REPO)
                try:
                    from analysis.add_layer_roles import roles_for
                finally:
                    if old is None:
                        os.environ.pop("CRYOEV_CRYOAI_ROOT", None)
                    else:
                        os.environ["CRYOEV_CRYOAI_ROOT"] = old
            _FN = roles_for
        return _FN


def run_image(ctx, objs, prior):
    idx = [k for k, o in enumerate(objs) if is_ev(ctx, o)]
    role, layers = load_roles_for()([obj_mask(ctx, objs[k]) for k in idx])
    out = {"role": [None] * len(objs), "layer_count": [None] * len(objs)}
    for j, k in enumerate(idx):
        out["role"][k], out["layer_count"][k] = role[j], int(layers[j])
    return out


def summary(rows) -> str:
    n = Counter(r.get("role") for r in rows if r.get("role"))
    ev = n["standalone"] + n["multilayer"]
    if not ev and not n["inner-layer"]:
        return "No EV outlines counted."
    pct = f" ({100 * n['multilayer'] / ev:.1f}% of EVs)" if ev else ""
    return (f"{n['multilayer']} multilayer{pct}, {n['standalone']} single, "
            f"{n['inner-layer']} inner layers (not counted as EVs)")


ROLES_V1 = Step("roles", 1, needs=(), columns=("role", "layer_count"), params=PARAMS, fn=run_image,
                sources=("analysis/steps/roles.py", "analysis/add_layer_roles.py",
                         "analysis/multilayer_containment.py", "analysis/layer_count_stats.py"),
                title="Layer roles",
                description="Which EV outlines are multilayer (contain another EV outline), single, or an inner "
                            "layer of another EV (not counted). Containment 99 %, inner/outer size ratio up to 0.75.",
                summary=summary)
