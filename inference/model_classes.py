"""Which predicted class ids are EVs, per registered model (models/models.yaml `classes:`).

Multi-class models (v5 `ev_nonev`: 0 = EV, 1 = non-EV such as VLDL) write every
class to the prediction files. EV counts, sizes and layer roles must use the EV
class only, or a VLDL particle is counted as an EV and an EV inside a VLDL
outline reads as "multilayer".

A model entry may carry `classes: {0: EV, 1: non-EV}`. No `classes:` key, or a
single class, means single-class: every prediction is an EV (the old
behaviour, unchanged). Multi-class <=> more than one class.

    from inference.model_classes import ev_class_ids
    keep = ev_class_ids("v5_evnonev_y8m_20261029")   # {0}; None for a single-class model

Reads YAML only (no torch), so processing scripts can use it too.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REGISTRY = Path(__file__).resolve().parent.parent / "models" / "models.yaml"
EV_NAME = "EV"


def model_classes(model_id: str, registry: Path = REGISTRY) -> dict[int, str]:
    """{class_id: name} from the model's `classes:` map; {} when it has none."""
    models = yaml.safe_load(Path(registry).read_text(encoding="utf-8"))["models"]
    if model_id not in models:
        raise KeyError(f"{model_id!r} not in {registry}; known: {sorted(models)}")
    return {int(k): str(v) for k, v in (models[model_id].get("classes") or {}).items()}


def is_multiclass(model_id: str, registry: Path = REGISTRY) -> bool:
    return len(model_classes(model_id, registry)) > 1


def ev_class_ids(model_id: str, registry: Path = REGISTRY) -> set[int] | None:
    """Class ids that are EVs for a multi-class model; None for a single-class one (keep all)."""
    classes = model_classes(model_id, registry)
    if len(classes) <= 1:
        return None
    ev = {k for k, v in classes.items() if v == EV_NAME}
    if not ev:
        raise ValueError(f"{model_id}: multi-class but no class named {EV_NAME!r} in {classes}")
    return ev


def describe(model_id: str, registry: Path = REGISTRY) -> str:
    """One log line naming the class filter, for a pipeline's start-up banner."""
    classes = model_classes(model_id, registry)
    keep = ev_class_ids(model_id, registry)
    if keep is None:
        return f"classes: {model_id} is single-class; every prediction is an EV"
    return (f"classes: {model_id} is multi-class {classes}; EV analysis keeps class "
            f"{sorted(keep)}, other classes are excluded (reported separately where supported)")


def resolve_model_id(model: str | None, predictions_dir: Path | None,
                     registry: Path = REGISTRY) -> str | None:
    """--model if given, else the predictions folder's parent name when it is a registered id
    (predict_models.py writes <out>/<model_id>/predictions/). None when neither is known."""
    if model:
        return model
    if predictions_dir is not None:
        models = yaml.safe_load(Path(registry).read_text(encoding="utf-8"))["models"]
        name = Path(predictions_dir).resolve().parent.name
        if name in models:
            return name
    return None


def ev_classes_for(model: str | None, predictions_dir: Path | None,
                   registry: Path = REGISTRY) -> set[int] | None:
    """EV class ids to keep (None = keep every prediction), printing the start-up log line.
    size_profile_from_predictions and add_layer_roles both call this, so they filter the same masks."""
    model_id = resolve_model_id(model, predictions_dir, registry)
    if model_id is None:
        print("classes: model unknown (no --model, and the predictions folder's parent is not a "
              "registered model_id): every prediction is treated as an EV")
        return None
    print(describe(model_id, registry))
    return ev_class_ids(model_id, registry)
