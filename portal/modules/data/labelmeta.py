"""
Label schemas and class maps (round 14, contracts C1 and C2), read only: one loader for the Data pages
(/data/classmap/<cm>, /data/schema/<schema>), the dataset page's class map link and Start a review's schema picker.

Where they come from, first hit wins (nothing here writes anything):
  class maps    labels.classmaps (B1: list_all() / load(id)); else the files themselves, the repo's built-ins
                labels/classmaps/<id>.json and the user maps <CryoAI-laptop>/classmaps/<id>.json
  label schemas labels.schemas (B3: list_all() / load(id), when it has them); else the registry: the label_schema
                pointer lines (<registry>/label_schema.jsonl) and each definition written once under
                <registry>/defs/label_schema/<id>.json; else the repo's labels/schemas/<id>.json
A file is read only under its own folder, by an id that matches the contract's pattern (never a path from a
request), at most MAX_BYTES. Every shape check here is for display only: the owners' modules validate on write.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from portal.modules.data.records import _dict, _list, _text, configured_root, is_contained

REPO = Path(__file__).resolve().parents[3]
CM_ID_RE = re.compile(r"^cm_[a-z0-9][a-z0-9_]{0,60}_v\d{1,4}\Z")
SCHEMA_ID_RE = re.compile(r"^ls_[a-z0-9][a-z0-9_]{0,60}_v\d{1,4}\Z")
MAX_BYTES = 512_000
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}\Z")


def _read(path: Path, root: Path):
    """A small JSON object under `root`, else None."""
    try:
        if not is_contained(path, root) or not path.is_file() or path.stat().st_size > MAX_BYTES:
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _owner_module(name):
    try:
        import importlib
        return importlib.import_module(name)
    except Exception:                                   # noqa: BLE001  the owner's module is not on this branch yet
        return None


# ---------------------------------------------------------------- class maps (C1)
def classmap_dirs() -> list:
    out = [REPO / "labels" / "classmaps"]
    lap = configured_root("laptop")
    if lap:
        out.append(Path(lap) / "classmaps")
    return out


def load_classmap(cm_id):
    """The class map dict for an id, or None."""
    if not isinstance(cm_id, str) or not CM_ID_RE.match(cm_id):
        return None
    mod = _owner_module("labels.classmaps")
    if mod is not None and hasattr(mod, "load"):
        try:
            cm = mod.load(cm_id)
            if isinstance(cm, dict) and cm.get("id") == cm_id:
                return cm
        except Exception:                               # noqa: BLE001  unknown id / unreadable: try the files
            pass
    for d in classmap_dirs():
        cm = _read(d / f"{cm_id}.json", d)
        if cm and cm.get("id") == cm_id:
            return cm
    return None


def list_classmaps() -> list:
    """Every class map, by id ({id: dict}); built-ins and user maps."""
    found: dict = {}
    mod = _owner_module("labels.classmaps")
    if mod is not None and hasattr(mod, "list_all"):
        try:
            for cm in mod.list_all() or []:
                cm = cm if isinstance(cm, dict) else load_classmap(cm)
                if isinstance(cm, dict) and isinstance(cm.get("id"), str) and CM_ID_RE.match(cm["id"]):
                    found.setdefault(cm["id"], cm)
        except Exception:                               # noqa: BLE001
            pass
    for d in classmap_dirs():
        try:
            names = sorted(p.stem for p in d.glob("cm_*.json"))
        except OSError:
            continue
        for stem in names:
            if stem not in found and CM_ID_RE.match(stem):
                cm = _read(d / f"{stem}.json", d)
                if cm and cm.get("id") == stem:
                    found[stem] = cm
    return found


def classmap_view(cm: dict) -> dict:
    """{targets: [[training class, [label classes]]], dropped: [label classes]} in the map's class order."""
    order = [str(c) for c in _list(cm.get("classes")) if isinstance(c, str)]
    by: dict = {}
    dropped = []
    for src, dst in _dict(cm.get("map")).items():
        if dst in (None, ""):
            dropped.append(str(src))
        else:
            by.setdefault(str(dst), []).append(str(src))
    targets = [t for t in order if t in by] + [t for t in by if t not in order] + \
        [t for t in order if t not in by]
    return {"targets": [[t, by.get(t, [])] for t in targets], "dropped": dropped}


# ---------------------------------------------------------------- label schemas (C2)
def schema_dirs(registry_root) -> list:
    out = []
    if registry_root:
        out.append(Path(registry_root) / "defs" / "label_schema")
    out.append(REPO / "labels" / "schemas")
    return out


def schema_lines(registry_root) -> dict:
    """{id: the registry's current label_schema line} (last line per id); {} when there are none."""
    if not registry_root:
        return {}
    out: dict = {}
    try:
        from registry import store
        out = store.current("label_schema", Path(registry_root)) or {}
    except Exception:                                   # noqa: BLE001  the kind isn't registered on this branch yet
        p = Path(registry_root) / "label_schema.jsonl"
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and isinstance(rec.get("id"), str):
                    out[rec["id"]] = {**out.get(rec["id"], {}), **rec}
        except OSError:
            return {}
    return {k: v for k, v in out.items() if isinstance(v, dict) and isinstance(k, str) and SCHEMA_ID_RE.match(k)}


def load_schema(schema_id, registry_root=None):
    """The schema definition dict for an id, or None."""
    if not isinstance(schema_id, str) or not SCHEMA_ID_RE.match(schema_id):
        return None
    mod = _owner_module("labels.schemas")
    if mod is not None and hasattr(mod, "load"):
        try:
            s = mod.load(schema_id)
            if isinstance(s, dict) and s.get("id") == schema_id:
                return s
        except Exception:                               # noqa: BLE001
            pass
    for d in schema_dirs(registry_root):
        s = _read(d / f"{schema_id}.json", d)
        if s and s.get("id") == schema_id:
            return s
    return None


def list_schemas(registry_root=None) -> dict:
    """{id: definition} for every schema the registry lists or the repo ships."""
    ids = set(schema_lines(registry_root))
    mod = _owner_module("labels.schemas")
    if mod is not None and hasattr(mod, "list_all"):
        try:
            for s in mod.list_all() or []:
                sid = s.get("id") if isinstance(s, dict) else s
                if isinstance(sid, str):
                    ids.add(sid)
        except Exception:                               # noqa: BLE001
            pass
    for d in schema_dirs(registry_root):
        try:
            ids.update(p.stem for p in d.glob("ls_*.json"))
        except OSError:
            continue
    out = {}
    for sid in sorted(i for i in ids if SCHEMA_ID_RE.match(i)):
        s = load_schema(sid, registry_root)
        if s:
            out[sid] = s
    return out


def schema_rows(schema: dict) -> list:
    """[{label, kind, required, choices: [[label, key]]}] for the schema's object attributes (display only)."""
    rows = []
    for a in _list(schema.get("object_attrs")):
        a = _dict(a)
        if not _text(a.get("id")):
            continue
        kind = _text(a.get("kind")) or "?"
        if kind in ("choice", "ordinal"):
            choices = [[str(_dict(o).get("label") if _dict(o).get("label") is not None else _dict(o).get("v")),
                        _text(_dict(o).get("key")) or ""] for o in _list(a.get("options"))]
        elif kind == "bool":
            choices = [["yes / no", _text(a.get("key")) or ""]]
        elif kind == "text":
            choices = [[f"text, up to {a.get('max') if isinstance(a.get('max'), int) else 300} characters",
                        _text(a.get("key")) or ""]]
        else:
            choices = []
        rows.append({"label": str(a.get("label") or a["id"]), "id": a["id"], "kind": kind,
                     "required": a.get("required") is True, "choices": choices})
    return rows


def version_chain(items: dict, item_id: str) -> list:
    """The ids of every version in item_id's chain (supersedes links both ways), oldest first, by version."""
    up = {k: _text(_dict(v).get("supersedes")) for k, v in items.items()}
    down: dict = {}
    for k, s in up.items():
        if s:
            down.setdefault(s, []).append(k)
    seen, todo = set(), [item_id]
    while todo and len(seen) < 50:
        k = todo.pop()
        if k in seen:
            continue
        seen.add(k)
        if up.get(k):
            todo.append(up[k])
        todo.extend(down.get(k, []))

    def ver(k):
        v = _dict(items.get(k)).get("version")
        return (v if isinstance(v, int) and not isinstance(v, bool) else 0, k)
    return sorted(seen, key=ver)
