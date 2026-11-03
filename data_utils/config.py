"""
Where the data folders are. Standard library only; nothing is resolved or raised at import.

    CryoAI-laptop   registry, review packages, labels, job bundles   $CRYOEV_LAPTOP_ROOT
    CryoAI          training images and outputs                       $CRYOEV_CRYOAI_ROOT
    registry        <CryoAI-laptop>/registry                          $CRYOEV_REGISTRY
    Cryo-image-processing (optional, read-only datasets)              $CRYOEV_PROCESSING_ROOT

Without the env var, each root is the first `<ancestor>/<name>` folder above the repo, so a nested
worktree finds the same folders as the main checkout. A missing required root raises ConfigError when
its function is called (required=False returns None instead).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LAPTOP, CRYOAI, PROCESSING = "CryoAI-laptop", "CryoAI", "Cryo-image-processing"


class ConfigError(RuntimeError):
    pass


def find_sibling(name, start=None):
    """First `<ancestor>/<name>` folder above `start` (default: above the repo, never inside it, so a stray
    `<repo>/CryoAI` can't win and get committed; the checkout may be a nested worktree)."""
    start = Path(start) if start else REPO.parent
    for p in [start, *start.parents]:
        if (p / name).is_dir():
            return p / name
    return None


def _root(env, name, required):
    val = os.environ.get(env)
    if val:
        if Path(val).is_dir():
            return Path(val)
        if required:
            raise ConfigError(f"${env} is {val}, which is not a folder")
        return None
    p = find_sibling(name)
    if p is None and required:
        raise ConfigError(f"{name} folder not found above {REPO}; set ${env} to its path")
    return p


def laptop_root(required=True) -> Path | None:
    return _root("CRYOEV_LAPTOP_ROOT", LAPTOP, required)


def cryoai_root(required=True) -> Path | None:
    return _root("CRYOEV_CRYOAI_ROOT", CRYOAI, required)


def processing_root() -> Path | None:
    """Optional: None when not found."""
    return _root("CRYOEV_PROCESSING_ROOT", PROCESSING, False)


def registry_root(required=True) -> Path | None:
    """$CRYOEV_REGISTRY as given (may not exist yet), else <CryoAI-laptop>/registry. Never a guessed folder."""
    env = os.environ.get("CRYOEV_REGISTRY")
    if env:
        return Path(env)
    lap = laptop_root(required)
    return lap / "registry" if lap else None


def protected_roots() -> tuple:
    """Never written: the processing datasets (env and found copies) and Box."""
    roots = [p for p in (processing_root(), find_sibling(PROCESSING), Path.home() / "Desktop" / PROCESSING) if p]
    return tuple(dict.fromkeys(roots)) + (Path.home() / "Box",)
