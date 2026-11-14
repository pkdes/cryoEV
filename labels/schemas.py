"""Label schemas (round 14, contract C2): what reviewers label with, as a versioned registry item with a maker.

    labels/schemas/<id>.json        built-ins (ls_legacy_v1 = today's review tool list, ls_lamellarity_v1)
    <registry>/defs/label_schema/<id>.json   registered definitions (written once, `python -m cryoev schema register`)

Shape ("label_schema/1"):
  {"schema": "label_schema/1", "id": "ls_<name>_v<N>", "name", "version": N, "maker": <person id>, "description",
   "classes": [{"name", "color": "#rrggbb", "key"?, "label"?}],
   "object_attrs": [{"id", "label", "kind": choice|ordinal|bool|text, "required"?, "key"?, "max"? (text),
                     "options"?: [{"v", "label"?, "key"?}] (choice / ordinal)}],
   "image_attrs": [same as object_attrs], "tags": [{"id", "label", "key"?}],
   "supersedes": null | <ls id>, "class_mapping": null | {<old class>: <new class> | null}}

validate(doc) -> [problems] (empty = valid):
  - attribute ids ^[a-z][a-z0-9_]{0,15}$, never "class"; ids unique within object_attrs / image_attrs / tags;
  - colours ^#[0-9a-fA-F]{6}$; labels 1-40 characters, plain text (no control characters, no < > &; the page
    escapes them anyway);
  - keys: one character [a-z0-9] (case-insensitive), never reserved by the object card (RESERVED_KEYS:
    ArrowLeft, ArrowRight, Enter, Backspace, i), never used twice within a scope: the classes and tags (the Editor's
    picker) are one scope, the object attributes (the object card) one, the image attributes one;
  - choice / ordinal: >= 2 options, values unique strings of 1-40 characters; text: max 1-300;
  - a new version (supersedes) carries a class_mapping whose keys are the old schema's classes (checked when the
    old one is loadable) and whose targets are this schema's classes or null.
load(id, registry_root=None) -> dict: a built-in, else the registered definition; validated.
to_page(doc) -> the part the review page gets (data.js D.schema): labels filled in, no maker / description.
migrate(cls, doc) -> the class name under doc for a class of the schema doc supersedes (class_mapping).
value_ok(attr, v) -> True when v is a valid value of that attribute (the review server's check, contract C3).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SCHEMA = "label_schema/1"
HERE = Path(__file__).resolve().parent
BUILTIN_DIR = HERE / "schemas"
DEFS_DIR = "defs/label_schema"                       # inside the registry folder
ID_RE = re.compile(r"^ls_[a-z0-9][a-z0-9_]{0,40}_v([1-9][0-9]{0,3})$")
ATTR_RE = re.compile(r"^[a-z][a-z0-9_]{0,15}$")
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
PERSON_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
KEY_RE = re.compile(r"^[a-z0-9]$")
RESERVED_KEYS = ("arrowleft", "arrowright", "enter", "backspace", "i")   # the object card's own keys (lower case)
KINDS = ("choice", "ordinal", "bool", "text")
TEXT_MAX = 300
LABEL_MAX = 40
DESC_MAX = 1000
TOP_KEYS = {"schema", "id", "name", "version", "maker", "description", "classes", "object_attrs", "image_attrs",
            "tags", "supersedes", "class_mapping"}
ATTR_KEYS = {"id", "label", "kind", "required", "key", "max", "options"}


class SchemaError(ValueError):
    pass


def _label_ok(s) -> bool:
    return (isinstance(s, str) and 1 <= len(s) <= LABEL_MAX and s == s.strip()
            and not any(ord(c) < 32 or c in "<>&\x7f" for c in s))


def _key_problem(key, where, scope: dict) -> str | None:
    if key is None:
        return None
    if not isinstance(key, str):
        return f"{where}: key must be a string"
    k = key.lower()
    if k in RESERVED_KEYS:
        return f"{where}: key {key!r} is reserved by the object card ({', '.join(RESERVED_KEYS)})"
    if not KEY_RE.match(k):
        return f"{where}: key {key!r} must be one letter or digit"
    if k in scope:
        return f"{where}: key {key!r} is already used by {scope[k]}"
    scope[k] = where
    return None


def _attrs_problems(attrs, field: str) -> list:
    out, ids, keys = [], set(), {}
    if not isinstance(attrs, list):
        return [f"{field} must be a list"]
    for n, a in enumerate(attrs):
        where = f"{field}[{n}]"
        if not isinstance(a, dict):
            out.append(f"{where} must be an object")
            continue
        extra = set(a) - ATTR_KEYS
        if extra:
            out.append(f"{where}: unknown fields {sorted(extra)}")
        aid = a.get("id")
        if not isinstance(aid, str) or not ATTR_RE.match(aid) or aid == "class":
            out.append(f"{where}: id {aid!r} must match {ATTR_RE.pattern} and not be 'class'")
        elif aid in ids:
            out.append(f"{where}: id {aid!r} used twice")
        else:
            ids.add(aid)
            where = f"{field}.{aid}"
        if not _label_ok(a.get("label")):
            out.append(f"{where}: label must be 1-{LABEL_MAX} plain characters")
        kind = a.get("kind")
        if kind not in KINDS:
            out.append(f"{where}: kind {kind!r} must be one of {KINDS}")
        if "required" in a and not isinstance(a["required"], bool):
            out.append(f"{where}: required must be true or false")
        p = _key_problem(a.get("key"), where, keys)
        if p:
            out.append(p)
        opts = a.get("options")
        if kind in ("choice", "ordinal"):
            if a.get("key") is not None:
                out.append(f"{where}: a {kind} attribute has keys on its options, not on itself")
            if not isinstance(opts, list) or len(opts) < 2:
                out.append(f"{where}: a {kind} attribute needs >= 2 options")
                continue
            vals = set()
            for m, o in enumerate(opts):
                ow = f"{where}.options[{m}]"
                if not isinstance(o, dict) or set(o) - {"v", "label", "key"}:
                    out.append(f"{ow}: an option is {{v, label?, key?}}")
                    continue
                v = o.get("v")
                if not _label_ok(v):
                    out.append(f"{ow}: v must be 1-{LABEL_MAX} plain characters")
                elif v in vals:
                    out.append(f"{ow}: value {v!r} used twice")
                else:
                    vals.add(v)
                if "label" in o and not _label_ok(o["label"]):
                    out.append(f"{ow}: label must be 1-{LABEL_MAX} plain characters")
                p = _key_problem(o.get("key"), f"{ow} ({v})", keys)
                if p:
                    out.append(p)
        elif opts is not None:
            out.append(f"{where}: options only for choice / ordinal")
        if kind == "text":
            mx = a.get("max", TEXT_MAX)
            if isinstance(mx, bool) or not isinstance(mx, int) or not 1 <= mx <= TEXT_MAX:
                out.append(f"{where}: max must be a whole number 1-{TEXT_MAX}")
        elif "max" in a:
            out.append(f"{where}: max only for text")
    return out


def validate(doc, previous=None) -> list:
    """Problems with a label schema document ([] = valid). previous: the schema `supersedes` names, when loaded."""
    if not isinstance(doc, dict):
        return ["a label schema is a JSON object"]
    out = []
    extra = set(doc) - TOP_KEYS
    if extra:
        out.append(f"unknown fields {sorted(extra)}")
    if doc.get("schema") != SCHEMA:
        out.append(f"schema must be {SCHEMA!r}")
    sid = doc.get("id")
    m = ID_RE.match(sid) if isinstance(sid, str) else None
    if not m:
        out.append(f"id {sid!r} must be ls_<name>_v<N>")
    ver = doc.get("version")
    if isinstance(ver, bool) or not isinstance(ver, int) or ver < 1:
        out.append("version must be a whole number >= 1")
    elif m and int(m.group(1)) != ver:
        out.append(f"version {ver} does not match the id's _v{m.group(1)}")
    if not isinstance(doc.get("name"), str) or not ATTR_RE.match(doc.get("name") or ""):
        out.append(f"name {doc.get('name')!r} must match {ATTR_RE.pattern}")
    if not isinstance(doc.get("maker"), str) or not PERSON_RE.match(doc.get("maker") or ""):
        out.append(f"maker {doc.get('maker')!r} must be a person id")
    desc = doc.get("description", "")
    if not isinstance(desc, str) or len(desc) > DESC_MAX or any(ord(c) < 32 and c not in "\n" for c in desc):
        out.append(f"description must be text of at most {DESC_MAX} characters")
    classes = doc.get("classes")
    names, picker = [], {}
    if not isinstance(classes, list) or not classes:
        out.append("classes must be a non-empty list")
        classes = []
    for n, c in enumerate(classes):
        where = f"classes[{n}]"
        if not isinstance(c, dict) or set(c) - {"name", "color", "key", "label"}:
            out.append(f"{where}: a class is {{name, color, key?, label?}}")
            continue
        if not _label_ok(c.get("name")):
            out.append(f"{where}: name must be 1-{LABEL_MAX} plain characters")
        elif c["name"] in names:
            out.append(f"{where}: class {c['name']!r} listed twice")
        else:
            names.append(c["name"])
            where = f"class {c['name']}"
        if not isinstance(c.get("color"), str) or not COLOR_RE.match(c["color"]):
            out.append(f"{where}: color {c.get('color')!r} must be #rrggbb")
        if "label" in c and not _label_ok(c["label"]):
            out.append(f"{where}: label must be 1-{LABEL_MAX} plain characters")
        p = _key_problem(c.get("key"), where, picker)
        if p:
            out.append(p)
    tags = doc.get("tags", [])
    if not isinstance(tags, list):
        out.append("tags must be a list")
        tags = []
    tag_ids = set()
    for n, t in enumerate(tags):
        where = f"tags[{n}]"
        if not isinstance(t, dict) or set(t) - {"id", "label", "key"}:
            out.append(f"{where}: a tag is {{id, label, key?}}")
            continue
        if not isinstance(t.get("id"), str) or not ATTR_RE.match(t["id"]) or t["id"] in tag_ids:
            out.append(f"{where}: id {t.get('id')!r} must match {ATTR_RE.pattern}, once")
        tag_ids.add(t.get("id"))
        if not _label_ok(t.get("label")):
            out.append(f"{where}: label must be 1-{LABEL_MAX} plain characters")
        p = _key_problem(t.get("key"), f"tag {t.get('id')}", picker)
        if p:
            out.append(p)
    out += _attrs_problems(doc.get("object_attrs", []), "object_attrs")
    out += _attrs_problems(doc.get("image_attrs", []), "image_attrs")
    sup, cmap = doc.get("supersedes"), doc.get("class_mapping")
    if sup is not None:
        if not isinstance(sup, str) or not ID_RE.match(sup) or sup == sid:
            out.append(f"supersedes {sup!r} must be another schema id")
        if not isinstance(cmap, dict):
            out.append("a schema that supersedes another needs class_mapping {old class: new class | null}")
    if cmap is not None:
        if not isinstance(cmap, dict):
            out.append("class_mapping must be an object or null")
        else:
            for k, v in cmap.items():
                if v is not None and v not in names:
                    out.append(f"class_mapping {k!r} -> {v!r}: not a class of this schema")
            if previous is not None:
                old = {c.get("name") for c in previous.get("classes") or [] if isinstance(c, dict)}
                missing = sorted(old - set(cmap) - set(names))
                if missing:
                    out.append(f"class_mapping leaves out old classes {missing} (map them, or to null)")
    return out


def canonical_bytes(doc: dict) -> bytes:
    """The bytes a definition is written once as (registry defs/, sha256): sorted keys, indent 1, newline."""
    return (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def sha256(doc: dict) -> str:
    return hashlib.sha256(canonical_bytes(doc)).hexdigest()


def read_file(path) -> dict:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SchemaError(f"{path}: unreadable ({type(e).__name__}: {e})") from e
    return doc


def builtin_ids() -> list:
    return sorted(p.stem for p in BUILTIN_DIR.glob("ls_*.json"))


def _find(sid: str, registry_root=None) -> Path | None:
    if not isinstance(sid, str) or not ID_RE.match(sid):
        raise SchemaError(f"{sid!r} is not a label schema id")
    p = BUILTIN_DIR / f"{sid}.json"
    if p.is_file():
        return p
    if registry_root is not None:
        q = Path(registry_root) / DEFS_DIR / f"{sid}.json"
        if q.is_file():
            return q
    return None


def load(sid: str, registry_root=None, _depth=0) -> dict:
    """A label schema by id (a built-in, else <registry_root>/defs/label_schema/<id>.json), validated."""
    p = _find(sid, registry_root)
    if p is None:
        raise SchemaError(f"label schema {sid} not found (built-ins: {', '.join(builtin_ids())})")
    doc = read_file(p)
    if doc.get("id") != sid:
        raise SchemaError(f"{p}: id {doc.get('id')!r} != file name {sid}")
    prev = None
    if doc.get("supersedes") and _depth < 20:
        try:
            prev = load(doc["supersedes"], registry_root, _depth + 1)
        except SchemaError:
            prev = None
    probs = validate(doc, prev)
    if probs:
        raise SchemaError(f"{sid}: invalid: " + "; ".join(probs[:6]))
    return doc


def load_file(path, registry_root=None) -> dict:
    """A schema file (not yet registered), validated (with its predecessor when that one is loadable)."""
    doc = read_file(path)
    prev = None
    if isinstance(doc, dict) and doc.get("supersedes"):
        try:
            prev = load(doc["supersedes"], registry_root)
        except SchemaError:
            prev = None
    probs = validate(doc, prev)
    if probs:
        raise SchemaError(f"{path}: invalid: " + "; ".join(probs[:8]))
    return doc


def _page_attr(a: dict) -> dict:
    out = {"id": a["id"], "label": a["label"], "kind": a["kind"], "required": bool(a.get("required", False))}
    if a.get("key"):
        out["key"] = a["key"].lower()
    if a["kind"] in ("choice", "ordinal"):
        out["options"] = [{"v": o["v"], "label": o.get("label") or o["v"],
                           **({"key": o["key"].lower()} if o.get("key") else {})} for o in a["options"]]
    if a["kind"] == "text":
        out["max"] = int(a.get("max", TEXT_MAX))
    return out


def to_page(doc: dict) -> dict:
    """data.js D.schema: what the review page renders (labels filled in, keys lower case)."""
    return {"id": doc["id"], "version": doc["version"], "name": doc["name"],
            "classes": [{"name": c["name"], "label": c.get("label") or c["name"], "color": c["color"].lower(),
                         **({"key": c["key"].lower()} if c.get("key") else {})} for c in doc["classes"]],
            "object_attrs": [_page_attr(a) for a in doc.get("object_attrs") or []],
            "image_attrs": [_page_attr(a) for a in doc.get("image_attrs") or []],
            "tags": [{"id": t["id"], "label": t["label"], **({"key": t["key"].lower()} if t.get("key") else {})}
                     for t in doc.get("tags") or []]}


def attr(doc: dict, aid: str, image=False) -> dict | None:
    for a in doc.get("image_attrs" if image else "object_attrs") or []:
        if isinstance(a, dict) and a.get("id") == aid:
            return a
    return None


def value_ok(a: dict, v) -> bool:
    """A value for attribute a (None = undone, always fine): an option's v; true / false; text within max."""
    if v is None:
        return True
    k = a.get("kind")
    if k in ("choice", "ordinal"):
        return isinstance(v, str) and any(o.get("v") == v for o in a.get("options") or [])
    if k == "bool":
        return isinstance(v, bool)
    if k == "text":
        if not isinstance(v, str) or len(v) > int(a.get("max", TEXT_MAX)):
            return False
        try:
            v.encode("utf-8")
        except UnicodeEncodeError:
            return False
        return True
    return False


def migrate(cls: str, doc: dict):
    """A class name of the schema `doc` supersedes -> its name under doc (class_mapping; null = dropped -> None).
    A name doc lists itself, or one the mapping doesn't name, is returned unchanged."""
    cmap = doc.get("class_mapping") or {}
    if cls in cmap:
        return cmap[cls]
    return cls


def registry_record(doc: dict, logged: str, recorded_by: str = "labels/schemas.py") -> tuple:
    """(the label_schema/1 registry line, {defs path: bytes}) for store.append(record, files=...)."""
    rel = f"{DEFS_DIR}/{doc['id']}.json"
    rec = {"schema": "label_schema/1", "id": doc["id"], "logged": logged, "recorded_by": recorded_by,
           "name": doc["name"], "version": doc["version"], "maker": doc["maker"],
           "description": doc.get("description") or "", "supersedes": doc.get("supersedes"),
           "def": rel, "def_sha256": sha256(doc), "n_classes": len(doc["classes"]),
           "n_object_attrs": len(doc.get("object_attrs") or []), "status": "active"}
    return rec, {rel: canonical_bytes(doc)}
