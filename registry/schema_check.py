"""Minimal JSON Schema (draft 2020-12) checker for registry/schemas/*.json.

`jsonschema` is not installed in the lab envs, so this covers the subset the schemas use. Any other
keyword raises UnsupportedKeyword instead of being silently ignored. If `jsonschema` is
installed, the tests also cross-check against it.
"""
import json
import re
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
ANNOTATIONS = {"$schema", "$id", "title", "description"}
SUPPORTED = ANNOTATIONS | {"type", "properties", "required", "additionalProperties", "items", "enum", "const",
                           "pattern", "minimum", "uniqueItems"}
TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict,
         "array": list, "null": type(None)}


class UnsupportedKeyword(ValueError):
    pass


def _is_type(v, t):
    if t in ("integer", "number") and isinstance(v, bool):
        return False
    if t == "integer" and isinstance(v, float):
        return v.is_integer()
    return isinstance(v, TYPES[t])


def errors(value, schema, path="$"):
    """List of 'path: message' strings; empty when valid."""
    bad = set(schema) - SUPPORTED
    if bad:
        raise UnsupportedKeyword(f"{path}: unsupported keyword(s) {sorted(bad)}")
    out = []
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(value, t) for t in types):
            return [f"{path}: expected {'/'.join(types)}, got {type(value).__name__}"]
    if "const" in schema and value != schema["const"]:
        out.append(f"{path}: must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, str) and "pattern" in schema and not re.search(schema["pattern"], value):
        out.append(f"{path}: {value!r} does not match {schema['pattern']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and "minimum" in schema \
            and value < schema["minimum"]:
        out.append(f"{path}: {value} < minimum {schema['minimum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                out.append(f"{path}: missing required '{k}'")
        extra = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                out += errors(v, props[k], f"{path}.{k}")
            elif extra is False:
                out.append(f"{path}: unexpected field '{k}'")
            elif isinstance(extra, dict):
                out += errors(v, extra, f"{path}.{k}")
    if isinstance(value, list):
        if "items" in schema:
            for n, v in enumerate(value):
                out += errors(v, schema["items"], f"{path}[{n}]")
        if schema.get("uniqueItems"):
            seen = [json.dumps(v, sort_keys=True) for v in value]
            if len(seen) != len(set(seen)):
                out.append(f"{path}: items not unique")
    return out


def walk(schema, path="$"):
    """Raise UnsupportedKeyword for any keyword anywhere in `schema` (checked once, at load)."""
    bad = set(schema) - SUPPORTED
    if bad:
        raise UnsupportedKeyword(f"{path}: unsupported keyword(s) {sorted(bad)}")
    for k, sub in (schema.get("properties") or {}).items():
        walk(sub, f"{path}.{k}")
    for key in ("items", "additionalProperties"):
        if isinstance(schema.get(key), dict):
            walk(schema[key], f"{path}[{key}]")


def load_schemas(schema_dir=SCHEMA_DIR):
    """{'image/1': schema, ...} keyed by each schema's const `schema` value. Every keyword is checked here."""
    out = {}
    for p in sorted(Path(schema_dir).glob("*.json")):
        sch = json.loads(p.read_text(encoding="utf-8"))
        walk(sch, p.name)
        out[sch["properties"]["schema"]["const"]] = sch
    return out


def check_record(rec, schemas):
    """Validate one record by its own `schema` field."""
    if not isinstance(rec, dict):
        return ["$: record is not an object"]
    name = rec.get("schema")
    if name not in schemas:
        return [f"$.schema: unknown schema {name!r}"]
    return errors(rec, schemas[name])
