"""
Class maps (round 14, contract C1): which label-set classes become which training classes.

A class map is a JSON file `class_map/1`:
    {"schema": "class_map/1", "id": "cm_ev_nonev_v1", "name": "ev_nonev", "version": 1, "maker": "builtin",
     "description": "...", "source_schema": "ls_legacy_v1", "classes": ["EV", "non-EV"],
     "map": {"<source class>": "<target class>" | null}, "ev_classes": ["EV"], "supersedes": null, "rename": null}
`null` drops the source class. A source class that is missing from the map is an error where it is used
(dataset_from_lv raises BuildError), never a silent drop.

Where maps live:
  built-ins   <repo>/labels/classmaps/<id>.json, generated from prepare_all_layer_singleclass.CLASS_SCHEMES
              (`cm_ev_v1`, `cm_ev_nonev_v1`, `cm_ev_nonev_undet_v1`; data_utils/tests/test_classmaps.py proves
              to_scheme(map) == the scheme tuple exactly)
  user maps   <CryoAI-laptop>/classmaps/<id>.json (data_utils/config.laptop_root), written once: the same id
              with other content is refused, and a built-in id can't be reused.

API: load(id | path), builtin(), list_all(), to_scheme(cm), ev_ids(cm), sha256(cm), labels_sha(dataset_dir),
check_compat(model_classes, schema_or_map), validate(cm), write_user(cm).

    python -m labels.classmaps list
    python -m labels.classmaps show cm_ev_nonev_v1
    python -m labels.classmaps add my_map.json [--do]      # admin: write a user map (dry run without --do)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILTIN_DIR = Path(__file__).resolve().parent / "classmaps"
SCHEMA = "class_map/1"
ID_RE = re.compile(r"^cm_[a-z0-9][a-z0-9_]*_v\d+$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$")
KEYS = ("schema", "id", "name", "version", "maker", "description", "source_schema", "classes", "map", "ev_classes",
        "supersedes", "rename")
EV_NAME = "EV"                       # the EV class of a label schema without ev_classes (inference/model_classes.py)
BUILTIN_SCHEMES = {"cm_ev_v1": "ev", "cm_ev_nonev_v1": "ev_nonev", "cm_ev_nonev_undet_v1": "ev_nonev_undet"}
BUILTIN_TEXT = {
    "ev": "Single class: every vesicle layer is EV; non-EV, VLDL and the root placeholder are dropped.",
    "ev_nonev": "EV + non-EV: vesicle layers are EV; VLDL and non-EV are non-EV.",
    "ev_nonev_undet": "EV + non-EV + Undetermined: as ev_nonev, with Low conf EV as its own class.",
}


class ClassMapError(ValueError):
    pass


def _repo_on_path():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------- read / check
def canonical(cm: dict) -> bytes:
    """The bytes sha256() hashes: sorted keys, no whitespace, UTF-8 (independent of file formatting)."""
    return json.dumps(cm, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256(cm: dict) -> str:
    return hashlib.sha256(canonical(cm)).hexdigest()


def validate(cm) -> dict:
    """-> cm. Raises ClassMapError with the first problem."""
    if not isinstance(cm, dict):
        raise ClassMapError("a class map is a JSON object")
    if cm.get("schema") != SCHEMA:
        raise ClassMapError(f"schema must be {SCHEMA!r}")
    extra = sorted(set(cm) - set(KEYS))
    if extra:
        raise ClassMapError(f"unknown keys {extra}")
    if not isinstance(cm.get("id"), str) or not ID_RE.match(cm["id"]):
        raise ClassMapError(f"id {cm.get('id')!r} must match {ID_RE.pattern}")
    if not isinstance(cm.get("name"), str) or not NAME_RE.match(cm["name"]):
        raise ClassMapError(f"name {cm.get('name')!r} must match {NAME_RE.pattern}")
    if not isinstance(cm.get("version"), int) or isinstance(cm.get("version"), bool) or cm["version"] < 1:
        raise ClassMapError("version must be an integer >= 1")
    if not cm["id"].endswith(f"_v{cm['version']}"):
        raise ClassMapError(f"id {cm['id']} must end with _v{cm['version']}")
    for k in ("maker", "source_schema"):
        if not isinstance(cm.get(k), str) or not cm[k]:
            raise ClassMapError(f"{k} must be a non-empty string")
    if not isinstance(cm.get("description", ""), str):
        raise ClassMapError("description must be a string")
    classes = cm.get("classes")
    if (not isinstance(classes, list) or not classes or not all(isinstance(c, str) and c for c in classes)
            or len(set(classes)) != len(classes)):
        raise ClassMapError("classes must be a non-empty list of distinct names")
    m = cm.get("map")
    if not isinstance(m, dict) or not m:
        raise ClassMapError("map must be a non-empty object {source class: target class or null}")
    for src, dst in m.items():
        if not isinstance(src, str) or not src:
            raise ClassMapError("map keys must be source class names")
        if dst is not None and dst not in classes:
            raise ClassMapError(f"map[{src!r}] = {dst!r} is not one of classes {classes}")
    unused = [c for c in classes if c not in set(m.values())]
    if unused:
        raise ClassMapError(f"classes {unused} get no source class")
    ev = cm.get("ev_classes")
    if not isinstance(ev, list) or not all(isinstance(c, str) and c in classes for c in ev):
        raise ClassMapError("ev_classes must be a list of names from classes")
    for k in ("supersedes", "rename"):
        if k not in cm:
            raise ClassMapError(f"missing key {k!r} (null if none)")
    if cm["supersedes"] is not None and (not isinstance(cm["supersedes"], str) or not ID_RE.match(cm["supersedes"])):
        raise ClassMapError("supersedes must be a class map id or null")
    if cm["rename"] is not None and not isinstance(cm["rename"], dict):
        raise ClassMapError("rename must be an object or null")
    return cm


def _read(path: Path) -> dict:
    try:
        return validate(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError) as e:
        if isinstance(e, ClassMapError):
            raise ClassMapError(f"{path}: {e}")
        raise ClassMapError(f"{path}: not readable as a class map ({type(e).__name__}: {e})")


def user_dir(required: bool = False) -> Path | None:
    """<CryoAI-laptop>/classmaps (config.laptop_root; $CRYOEV_LAPTOP_ROOT); None when the root is not found."""
    _repo_on_path()
    from data_utils import config
    lap = config.laptop_root(required=required)
    return lap / "classmaps" if lap else None


def builtin() -> dict:
    """{id: map} of the repo's built-in maps."""
    out = {}
    for p in sorted(BUILTIN_DIR.glob("cm_*.json")):
        cm = _read(p)
        if cm["id"] != p.stem:
            raise ClassMapError(f"{p}: id {cm['id']} != file name")
        out[cm["id"]] = cm
    return out


def user_maps(root: Path | None = None) -> dict:
    d = Path(root) if root else user_dir()
    out = {}
    if d and d.is_dir():
        for p in sorted(d.glob("cm_*.json")):
            cm = _read(p)
            if cm["id"] != p.stem:
                raise ClassMapError(f"{p}: id {cm['id']} != file name")
            out[cm["id"]] = cm
    return out


def list_all(user_root: Path | None = None) -> list:
    """Built-ins first, then user maps; each a dict with `_origin` ("builtin" | "user") and `_sha256` added (copies)."""
    b = builtin()
    out = [dict(cm, _origin="builtin", _sha256=sha256(cm)) for cm in b.values()]
    out += [dict(cm, _origin="user", _sha256=sha256(cm)) for i, cm in user_maps(user_root).items() if i not in b]
    return out


def load(ref, expect_sha256: str | None = None, user_root: Path | None = None) -> dict:
    """A map by id (built-in, else user map) or by file path. expect_sha256 (a dataset's recorded class_map_sha256)
    catches a map edited after the dataset was built."""
    if isinstance(ref, dict):
        cm = validate(ref)
    else:
        s = str(ref)
        p = Path(s)
        if ID_RE.match(s) and not p.is_file():
            f = BUILTIN_DIR / f"{s}.json"
            if not f.is_file():
                d = Path(user_root) if user_root else user_dir()
                f = d / f"{s}.json" if d else None
            if f is None or not f.is_file():
                raise ClassMapError(f"no class map {s!r} (built-ins: {sorted(builtin())})")
            cm = _read(f)
            if cm["id"] != s:
                raise ClassMapError(f"{f}: id {cm['id']} != {s}")
        elif p.is_file():
            cm = _read(p)
        else:
            raise ClassMapError(f"{s!r} is neither a class map id nor a file")
    if expect_sha256 and sha256(cm) != expect_sha256:
        raise ClassMapError(f"class map {cm['id']} sha256 {sha256(cm)[:12]} != recorded {expect_sha256[:12]}: "
                            f"the map changed after it was used")
    return cm


def write_user(cm: dict, root: Path | None = None, do: bool = False) -> tuple:
    """Write a user map once. -> (path, "written" | "would write" | "exists, identical"). Refuses a built-in id and
    an existing id with other content."""
    validate(cm)
    if cm["id"] in builtin() or (BUILTIN_DIR / f"{cm['id']}.json").exists():
        raise ClassMapError(f"{cm['id']} is a built-in id")
    d = Path(root) if root else user_dir(required=True)
    p = d / f"{cm['id']}.json"
    if p.exists():
        old = _read(p)
        if sha256(old) == sha256(cm):
            return p, "exists, identical"
        raise ClassMapError(f"{p} exists with other content: maps are written once (make _v{cm['version'] + 1})")
    if not do:
        return p, "would write"
    d.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_bytes(json.dumps(cm, indent=1, ensure_ascii=False).encode("utf-8") + b"\n")
    try:
        os.link(tmp, p)                                       # fails if p appeared meanwhile: never overwrite
    finally:
        tmp.unlink(missing_ok=True)
    return p, "written"


# ---------------------------------------------------------------- use
def to_scheme(cm: dict) -> tuple:
    """-> (CATEGORY_NAME_TO_CLASS, CLASS_NAMES) as prepare_all_layer_singleclass.CLASS_SCHEMES holds them."""
    classes = list(cm["classes"])
    idx = {c: i for i, c in enumerate(classes)}
    return {src: (None if dst is None else idx[dst]) for src, dst in cm["map"].items()}, classes


def ev_ids(cm: dict) -> set:
    """Target class ids that are EVs."""
    return {i for i, c in enumerate(cm["classes"]) if c in set(cm.get("ev_classes") or ())}


def _target_classes(t) -> tuple:
    """(class names, EV names, label) of a class map, a label schema (classes [{name}]) or a plain list of names."""
    if isinstance(t, dict):
        cl = [c.get("name") if isinstance(c, dict) else c for c in (t.get("classes") or [])]
        ev = t.get("ev_classes")
        if ev is None:
            ev = [EV_NAME] if EV_NAME in cl else []
        return [c for c in cl if isinstance(c, str)], list(ev), str(t.get("id") or "the target")
    cl = [str(c) for c in (t or [])]
    return cl, [EV_NAME] if EV_NAME in cl else [], "the target"


def check_compat(model_classes, schema_or_map) -> list:
    """Problems (plain sentences; [] = compatible) between a model's classes and a label schema / class map.

    model_classes: {id: name} (inference.model_classes.model_classes; {} or one class = a single-class model whose
    one class is an EV) or a list of names. Checks: every class of a multi-class model has a class of the same name in
    the target (a prediction review package refuses otherwise), and at least one model class resolves to an EV class
    of the target (Results refuses otherwise)."""
    if isinstance(model_classes, dict):
        names = [str(v) for _, v in sorted(model_classes.items(), key=lambda kv: int(kv[0]))]
    else:
        names = [str(v) for v in (model_classes or [])]
    cl, ev, label = _target_classes(schema_or_map)
    probs = []
    if len(names) <= 1:                                   # single-class model: every prediction is an EV
        if not ev:
            probs.append(f"{label} has no EV class for the single-class model's predictions")
        return probs
    missing = [n for n in names if n not in cl]
    if missing:
        probs.append(f"model class(es) {missing} have no class in {label} (classes {cl})")
    if not any(n in ev for n in names):
        probs.append(f"no model class resolves to an EV class of {label} (model {names}, EV classes {ev})")
    return probs


# ---------------------------------------------------------------- dataset identity
YAML_NAMES = ("dataset.yaml", "data.yaml")
YAML_SKIP = ("path:", "# YOLO dataset config for")   # the lines that name the folder (prepare's write_dataset_yaml)


def yaml_norm(text: str) -> str:
    return "".join(ln + "\n" for ln in text.splitlines() if not ln.startswith(YAML_SKIP))


def labels_sha_of(label_files: dict, yaml_text: str | None, image_files: dict) -> str:
    """label_files {"labels/<split>/<name>.txt": bytes}, the dataset yaml text, image files {"images/<split>/<name>":
    size in bytes}. -> sha256 over sorted `L\\t<rel>\\t<sha256 of the text with LF line ends>` lines (prepare writes
    labels in text mode: CRLF on Windows), one `Y\\t<sha256 of the yaml without its folder lines>` line and sorted
    `I\\t<rel>\\t<size>` lines. A copy of a dataset in another folder has the same value; two datasets whose images
    have the same names but other files (other sizes) do not (QC r14 #17; sizes, not content hashes, so a preview
    never reads every image)."""
    if not isinstance(image_files, dict):
        raise TypeError("image_files must be {rel path: size in bytes}")
    crlf, lf = b"\r\n", b"\n"
    lines = [f"L\t{rel}\t{hashlib.sha256(b.replace(crlf, lf)).hexdigest()}\n" for rel, b in sorted(label_files.items())]
    lines.append(f"Y\t{hashlib.sha256(yaml_norm(yaml_text or '').encode('utf-8')).hexdigest()}\n")
    lines += [f"I\t{rel}\t{int(image_files[rel])}\n" for rel in sorted(image_files)]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def labels_sha(dataset_dir) -> str:
    """The "identical dataset" hash of a YOLO dataset folder: its label files, its dataset.yaml (folder lines left
    out) and its image files' names and sizes. Raises ClassMapError when the folder has no labels/ or yaml."""
    d = Path(dataset_dir)
    lab = d / "labels"
    y = next((d / n for n in YAML_NAMES if (d / n).is_file()), None)
    if not lab.is_dir() or y is None:
        raise ClassMapError(f"{d} is not a YOLO dataset folder (labels/ and dataset.yaml)")
    files = {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(lab.rglob("*.txt")) if p.is_file()}
    imgs = {p.relative_to(d).as_posix(): p.stat().st_size for p in sorted((d / "images").rglob("*")) if p.is_file()} \
        if (d / "images").is_dir() else {}
    return labels_sha_of(files, y.read_text(encoding="utf-8"), imgs)


# ---------------------------------------------------------------- built-in generator
def generate_builtins() -> dict:
    """{id: map} generated from prepare_all_layer_singleclass.CLASS_SCHEMES (the source of truth for the files in
    labels/classmaps/; data_utils/tests/test_classmaps.py checks the files equal this)."""
    _repo_on_path()
    from data_utils import prepare_all_layer_singleclass as prep
    out = {}
    for cid, scheme in BUILTIN_SCHEMES.items():
        mapping, names = prep.CLASS_SCHEMES[scheme]
        out[cid] = {"schema": SCHEMA, "id": cid, "name": scheme, "version": 1, "maker": "builtin",
                    "description": BUILTIN_TEXT[scheme], "source_schema": "ls_legacy_v1", "classes": list(names),
                    "map": {src: (None if i is None else names[i]) for src, i in mapping.items()},
                    "ev_classes": [EV_NAME], "supersedes": None, "rename": None}
    return out


def builtin_file_bytes(cm: dict) -> bytes:
    return json.dumps(cm, indent=1, ensure_ascii=False).encode("utf-8") + b"\n"


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m labels.classmaps", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="every map: id, origin, classes")
    s = sub.add_parser("show", help="one map as JSON, with its sha256")
    s.add_argument("ref")
    a = sub.add_parser("add", help="write a user map to <CryoAI-laptop>/classmaps/ (dry run without --do)")
    a.add_argument("file", type=Path)
    a.add_argument("--do", action="store_true")
    g = sub.add_parser("write-builtins", help="regenerate labels/classmaps/*.json from CLASS_SCHEMES (repo files)")
    g.add_argument("--do", action="store_true")
    ns = ap.parse_args(argv)
    try:
        if ns.cmd == "list":
            for cm in list_all():
                print(f"{cm['id']:28s} {cm['_origin']:8s} {cm['classes']}  ev={cm['ev_classes']}")
        elif ns.cmd == "show":
            cm = load(ns.ref)
            print(json.dumps(dict(cm, _sha256=sha256(cm)), indent=1, ensure_ascii=False))
        elif ns.cmd == "add":
            cm = json.loads(ns.file.read_text(encoding="utf-8"))
            p, what = write_user(cm, do=ns.do)
            print(f"{what}: {p}" + ("" if ns.do or what != "would write" else "  (dry run; add --do)"))
        elif ns.cmd == "write-builtins":
            for cid, cm in generate_builtins().items():
                p = BUILTIN_DIR / f"{cid}.json"
                same = p.is_file() and p.read_bytes() == builtin_file_bytes(cm)
                if not same and ns.do:
                    p.write_bytes(builtin_file_bytes(cm))
                print(f"{'same' if same else ('written' if ns.do else 'would write')}: {p}")
    except (ClassMapError, OSError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
