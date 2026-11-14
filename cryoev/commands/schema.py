"""schema: label schemas in the registry (round 14, contract C2).

  python -m cryoev schema register <file> [--do]     validate a label schema file and register it

Dry run: the file is validated (labels/schemas.validate: ids, colours, labels, reserved and duplicate keys, the
class mapping of a new version) and the registry line + definition path that --do would write are printed.
--do: one label_schema/1 line in <registry>/label_schema.jsonl, with the definition written once to
<registry>/defs/label_schema/<id>.json under the registry lock (store.append(files=...)). Registering the same id
again with identical content is refused as already registered; with other content it is refused (a new version
needs a new id with supersedes + class_mapping). A file using a built-in's id (labels/schemas/<id>.json) whose content
differs from the built-in is refused too (the built-in would win every load anyway).
"""
from __future__ import annotations

from pathlib import Path

from cryoev import common as CM


def add(sub):
    p = sub.add_parser("schema", help="label schemas: register (dry run unless --do)")
    ss = p.add_subparsers(dest="schema_cmd", required=True, metavar="action")
    r = ss.add_parser("register", help="validate a label schema file and register it")
    r.add_argument("file", type=Path, help="label schema JSON (e.g. labels/schemas/ls_lamellarity_v1.json)")
    CM.write_flags(r)
    r.set_defaults(func=run_register)


def _schemas():
    from labels import schemas
    return schemas


def run_register(a) -> int:
    S = _schemas()
    root = CM.registry_root()
    reg = CM.registry()
    try:
        doc = S.load_file(a.file, registry_root=root)
    except S.SchemaError as e:
        raise CM.CliError(f"refused: {e}")
    rec, files = S.registry_record(doc, reg.now(), recorded_by="cryoev schema register")
    rel = next(iter(files))

    def problem():
        bi = S.BUILTIN_DIR / f"{doc['id']}.json"
        if bi.is_file():                    # a built-in id: only that very definition may be registered under it
            try:
                same_bi = S.canonical_bytes(S.read_file(bi)) == files[rel]
            except S.SchemaError:
                same_bi = False
            if not same_bi:
                return (f"{doc['id']} is a built-in id (labels/schemas/{doc['id']}.json) and this file's content "
                        "differs: give it a new id")
        cur = reg.current("label_schema", root).get(doc["id"])
        defp = Path(root) / rel
        if cur is not None or defp.exists():
            same = defp.is_file() and defp.read_bytes() == files[rel]
            return (f"{doc['id']} is already registered" + (" (same content)" if same else
                    " with OTHER content: a new version needs a new id with supersedes + class_mapping"))
        if doc.get("supersedes") and doc["supersedes"] not in reg.current("label_schema", root) \
                and doc["supersedes"] not in S.builtin_ids():
            return f"supersedes {doc['supersedes']} is neither registered nor a built-in"
        return None

    def plan():
        why = problem()
        print(f"label schema {doc['id']} (v{doc['version']}, maker {doc['maker']}): valid; "
              f"{len(doc['classes'])} classes, {len(doc.get('object_attrs') or [])} object attributes, "
              f"{len(doc.get('image_attrs') or [])} image attributes, sha256 {rec['def_sha256'][:12]}")
        if why:
            print(f"refused: {why}")
            return 2
        errs = reg.check_record(rec, reg.schemas())
        if errs:
            print("refused: registry line invalid: " + "; ".join(errs[:4]))
            return 2
        print(f"would write {Path(root) / rel} (once) and append a label_schema/1 line to "
              f"{Path(root) / 'label_schema.jsonl'}")
        return 0

    def act(who, via):
        why = problem()
        if why:
            raise CM.CliError(f"refused: {why}")
        line = reg.append(rec, root, new_only=True, files=files)
        print(f"registered {doc['id']} -> {Path(root) / rel}")
        return 0, {"id": line["id"], "def_sha256": rec["def_sha256"]}

    return CM.run_write(a, "label schema registered", doc["id"], {"file": Path(a.file).name}, plan, act)
