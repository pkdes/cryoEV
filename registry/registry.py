"""Registry-lite: append-only JSONL, one file per entity (docs/TRACKING_CONTRACT.md).

    <CryoAI-laptop>/registry/<entity>.jsonl     one record per line; the LAST line per id (file order) is current

    python registry/registry.py import [--dry-run]
    python registry/registry.py list [--entity model] [--status production]
    python registry/registry.py show <id>
    python registry/registry.py set-status <id> <status> --who <people.json id> [--note ...]

API (piece 7 reads through these): append(record), current() -> {id: record}, history(id) -> [record, ...].
inference_check and label_version are pointer records (`owner` = batches.jsonl / version.json): the registry
stores the id, status lines and anything that differs from the owner file; the rest is read from the owner
file at read time. Explicit status lines (`status_by`, from set-status) beat derived statuses.
Writes go through data_utils.path_guard (never under Box or Cryo-image-processing). Session 20261012 is refused.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from data_utils.path_guard import protected_problem  # noqa: E402
from schema_check import check_record, load_schemas  # noqa: E402
from validate import ENTITIES, EXCLUDED_SESSIONS, find_sibling, read_json  # noqa: E402

PEOPLE = REPO / "annotation" / "people.json"
POINTER = {"inference_check", "label_version"}
ENVELOPE = {"schema", "id", "logged", "recorded_by", "note", "owner", "status_by"}
PRODUCTION_WHO = "pujan"
_SCHEMAS = None


class RegistryError(ValueError):
    pass


def default_root():
    env = os.environ.get("CRYOEV_REGISTRY")
    if env:
        return Path(env)
    lap = find_sibling("CryoAI-laptop", REPO)
    return (lap or Path(r"C:\Users\pujan\Desktop\cryoEV pujrog\CryoAI-laptop")) / "registry"


def schemas():
    global _SCHEMAS
    if _SCHEMAS is None:
        _SCHEMAS = load_schemas()
    return _SCHEMAS


def roster(path=PEOPLE):
    return {p["id"] for p in read_json(path)["people"]}


def now():
    return datetime.now().isoformat(timespec="seconds")


def entity_of(rec):
    e = str(rec.get("schema", "")).split("/")[0]
    if e not in ENTITIES:
        raise RegistryError(f"refused: unknown entity schema {rec.get('schema')!r}")
    return e


# ---------------------------------------------------------------- owner files (pointer records)
def _owner(entity, owner):
    """Fields derived from the owner file at read time: `<version.json>` or `<batches.jsonl>#<batch_id>`."""
    if not owner:
        return {}
    path, _, key = str(owner).partition("#")
    if not Path(path).exists():
        return {}
    if entity == "label_version":
        return read_json(path)
    last = {}
    with open(path, encoding="utf-8") as fh:            # batches.jsonl: last line for the batch, file order
        for line in fh:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("batch_id") == key:
                last = o
    return {k: v for k, v in last.items() if k != "logged"}


def derive(entity, stored):
    """Stored line -> full record (owner file underneath, stored fields on top)."""
    if entity not in POINTER:
        return dict(stored)
    base = _owner(entity, stored.get("owner"))
    out = {**base, **stored}
    if "status" not in stored and "status" in base:
        out["status"] = base["status"]
    return out


def compact(entity, rec):
    """Full record -> what gets stored (pointer types keep only what differs from the owner file)."""
    if entity not in POINTER:
        return rec
    base = _owner(entity, rec.get("owner"))
    return {k: v for k, v in rec.items() if k in ENVELOPE or k == "status" and (rec.get("status_by") or
            base.get("status") != v) or base.get(k) != v}


# ---------------------------------------------------------------- read
def _lines(root, entity):
    p = Path(root) / f"{entity}.jsonl"
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue                            # a line cut short is skipped, not fatal
    return out


def current(entity=None, root=None):
    """{id: latest record} (last line per id in file order), owner fields derived for pointer records."""
    root = root or default_root()
    out = {}
    for e in ([entity] if entity else ENTITIES):
        for rec in _lines(root, e):
            out[rec["id"]] = rec
    return {k: derive(entity_of(v), v) for k, v in out.items()}


def history(id, root=None):
    """Every line for `id`, oldest first, owner fields derived."""
    root = root or default_root()
    return [derive(e, r) for e in ENTITIES for r in _lines(root, e) if r.get("id") == id]


# ---------------------------------------------------------------- write
def _guard(path):
    problem = protected_problem(path)
    if problem:
        raise RegistryError(f"refused: {problem}")


def _check(rec, root, via_set_status=False):
    e = entity_of(rec)
    full = derive(e, rec)
    errs = check_record(full, schemas())
    if errs:
        raise RegistryError(f"refused {rec.get('id')}: " + "; ".join(errs[:4]))
    if str(full.get("session") or "")[:8] in EXCLUDED_SESSIONS or str(rec.get("id"))[:8] in EXCLUDED_SESSIONS:
        raise RegistryError("refused: excluded session (counted only, no records)")
    if rec.get("status_by") and rec["status_by"] not in roster():
        raise RegistryError(f"refused: status_by {rec['status_by']!r} is not in people.json")
    if e == "model" and full.get("status") == "production":
        prev = current("model", root).get(rec["id"], {})
        kept = prev.get("status") == "production" and prev.get("status_by") == PRODUCTION_WHO \
            and rec.get("status_by") == PRODUCTION_WHO
        if not (kept or via_set_status and rec.get("status_by") == PRODUCTION_WHO):
            raise RegistryError(f"refused: only `set-status <id> production --who {PRODUCTION_WHO}` sets production")
    return e


def append(record, root=None, _via_set_status=False):
    """Validate, then append one line to <root>/<entity>.jsonl. Returns the stored line."""
    root = Path(root or default_root())
    rec = dict(record)
    rec.setdefault("logged", now())
    e = _check(rec, root, _via_set_status)
    path = root / f"{e}.jsonl"
    _guard(root)
    _guard(path)
    root.mkdir(parents=True, exist_ok=True)
    line = compact(e, rec)
    lead = ""
    if path.exists() and path.stat().st_size:          # a line cut short must not swallow the next one
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            lead = "" if f.read(1) == b"\n" else "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(lead + json.dumps(line, ensure_ascii=False) + "\n")
    return line


def set_status(id, status, who, note=None, root=None):
    if who not in roster():
        raise RegistryError(f"refused: --who {who!r} is not in people.json")
    cur = current(root=root).get(id)
    if cur is None:
        raise RegistryError(f"unknown id {id}")
    e = entity_of(cur)
    if e == "model" and status == "production" and who != PRODUCTION_WHO:
        raise RegistryError(f"refused: only {PRODUCTION_WHO} may set production")
    rec = {k: v for k, v in cur.items()} if e not in POINTER else \
        {k: v for k, v in _lines_last(root, e, id).items()}
    rec.update(status=status, status_by=who, logged=now(), recorded_by="registry.py set-status")
    if note:
        rec["note"] = note
    return append(rec, root, _via_set_status=True)


def _lines_last(root, entity, id):
    last = {}
    for r in _lines(root or default_root(), entity):
        if r.get("id") == id:
            last = r
    return last


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None, help="registry folder (default <CryoAI-laptop>/registry)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    im = sub.add_parser("import")
    im.add_argument("--dry-run", action="store_true")
    ls = sub.add_parser("list")
    ls.add_argument("--entity")
    ls.add_argument("--status")
    sh = sub.add_parser("show")
    sh.add_argument("id")
    ss = sub.add_parser("set-status")
    ss.add_argument("id")
    ss.add_argument("status")
    ss.add_argument("--who", required=True)
    ss.add_argument("--note")
    a = ap.parse_args(argv)
    root = a.root or default_root()
    try:
        if a.cmd == "import":
            import importers
            res = importers.run(root, dry_run=a.dry_run)
            importers.print_result(res)
        elif a.cmd == "list":
            for i, r in sorted(current(a.entity, root).items(), key=lambda kv: (kv[1]["schema"], kv[0])):
                if not a.status or r.get("status") == a.status:
                    print(f"{r['schema']:20} {r.get('status') or '-':12} {i}")
        elif a.cmd == "show":
            h = history(a.id, root)
            if not h:
                print(f"unknown id {a.id}")
                return 1
            print(json.dumps(h[-1], indent=1, ensure_ascii=False, default=str))
            print(f"({len(h)} line(s); statuses: {' -> '.join(str(x.get('status')) for x in h)})")
        elif a.cmd == "set-status":
            line = set_status(a.id, a.status, a.who, a.note, root)
            print(f"appended: {line['id']} status={line.get('status')} by {line.get('status_by')}")
    except RegistryError as e:
        print(e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
