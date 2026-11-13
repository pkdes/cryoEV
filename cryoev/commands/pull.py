"""pull labels|predictions|results|dataset <id> --out DIR: copy the files a registry record points to into
<DIR>/<id>/, with a sha256 manifest. Read-only on the data; a dry run (the file list and size) unless --do.

  labels <lv id>         the label version's folder (the folder of its version.json: labels.coco.json, version.json,
                         changes.jsonl, conflicts.csv)
  predictions <ps id>    the prediction set's p/ (predictions + summary.csv), images.csv, request.json, status.json;
                         --with-images adds images/ (the tiles)
  results <ps id>        the prediction set's results/ (results.json, per_image.csv, per_sample.csv; made by
                         `python -m results --ps <set folder>`)
  dataset <dataset id>   the prepared dataset folder (registry dataset_dir: images/, labels/, dataset.yaml, ...)

--out is refused under a protected root (data_utils/path_guard.py: Box, Cryo-image-processing) and inside the data
folders or the repo; <DIR>/<id> must not exist yet (nothing is overwritten). Each copy is checked against its
source's sha256; <DIR>/<id>/MANIFEST.sha256 lists `sha256  relative path` for every file, and pull.json says what
was pulled, from which record, when and by whom.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from cryoev import common as CM

KINDS = ("labels", "predictions", "results", "dataset")
MANIFEST, INFO = "MANIFEST.sha256", "pull.json"
PS_FILES = ("images.csv", "request.json", "status.json")


def add(sub):
    p = sub.add_parser("pull", help="copy a label version / predictions / results / dataset out, with a sha256 "
                                    "manifest (dry run unless --do)")
    p.add_argument("kind", choices=KINDS)
    p.add_argument("id")
    p.add_argument("--out", type=Path, required=True, help="a folder outside the data folders and the repo")
    p.add_argument("--with-images", action="store_true", help="predictions: the tiles too")
    CM.write_flags(p)
    p.set_defaults(func=run)


def _record(entity, rid):
    rec = CM.registry().current(entity, CM.registry_root()).get(rid)
    if rec is None:
        raise CM.CliError(f"no {entity} record {rid!r} in the registry (python -m cryoev find {rid})")
    return rec


def _tree(folder: Path, rel=None) -> list:
    """[(source file, path relative to the pull)] for every file under folder (sorted)."""
    base = Path(rel) if rel else Path()
    return [(p, (base / p.relative_to(folder)).as_posix()) for p in sorted(folder.rglob("*")) if p.is_file()]


def _ps_folder(rid) -> Path:
    from jobs import predict_set as PS
    rec = _record("prediction_set", rid)
    folder = PS.resolve_record_path(rec.get("path"))
    if folder is None or not folder.is_dir():
        raise CM.CliError(f"prediction set {rid}: folder {rec.get('path')} not found")
    return folder


def plan_files(kind, rid, with_images=False) -> tuple:
    """-> (source folder, [(source file, relative path)]). Raises CliError when the record or its files are missing."""
    if kind == "labels":
        rec = _record("label_version", rid)
        if not rec.get("owner"):
            raise CM.CliError(f"label version {rid} has no version.json (owner) in the registry")
        folder = Path(rec["owner"]).parent
        if not folder.is_dir():
            raise CM.CliError(f"label version {rid}: folder {folder} not found")
        return folder, _tree(folder)
    if kind in ("predictions", "results"):
        folder = _ps_folder(rid)
        if kind == "results":
            res = folder / "results"
            if not res.is_dir():
                raise CM.CliError(f"{rid} has no results yet; compute them with `python -m results --ps {folder}`")
            return folder, _tree(res, "results")
        if not (folder / "p").is_dir():
            st = {}
            try:
                st = json.loads((folder / "status.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            raise CM.CliError(f"{rid} has no predictions yet (status {st.get('state', 'unknown')})")
        files = [(folder / n, n) for n in PS_FILES if (folder / n).is_file()] + _tree(folder / "p", "p")
        if with_images:
            files += _tree(folder / "images", "images")
        return folder, files
    rec = _record("dataset", rid)
    folder = Path(str(rec.get("dataset_dir") or ""))
    if not rec.get("dataset_dir") or not folder.is_dir():
        raise CM.CliError(f"dataset {rid}: dataset_dir {rec.get('dataset_dir')!r} not found")
    return folder, _tree(folder)


def run(a) -> int:
    out = Path(a.out)
    problem = CM.out_problem(out)
    if problem:
        raise CM.CliError(f"refused: --out {problem}")
    dest = out / a.id
    if Path(a.id).name != a.id or a.id in (".", ".."):
        raise CM.CliError(f"refused: {a.id!r} is not a plain id")
    if dest.exists():
        raise CM.CliError(f"refused: {dest} already exists (nothing is overwritten; pick another --out)")
    src, files = plan_files(a.kind, a.id, a.with_images)
    if not files:
        raise CM.CliError(f"nothing to pull: {src} has no files")
    total = sum(p.stat().st_size for p, _ in files)
    args = {"kind": a.kind, "id": a.id, "with_images": bool(a.with_images)}

    def plan():
        print(f"would copy {len(files)} file(s), {total / 1e6:.1f} MB, from {src}\n  to {dest}")
        for _, rel in files[:15]:
            print(f"  {rel}")
        if len(files) > 15:
            print(f"  ... and {len(files) - 15} more")
        print(f"then write {dest / MANIFEST} (sha256 of each file, checked against its source) and {dest / INFO}")
        return 0

    def act(who, via):
        dest.mkdir(parents=True)
        lines = []
        for p, rel in files:
            h = CM.sha256(p)
            d = dest / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d)
            if CM.sha256(d) != h:
                raise CM.CliError(f"copy of {rel} differs from its source (sha256); {dest} is incomplete")
            lines.append(f"{h}  {rel}")
        (dest / MANIFEST).write_text("".join(ln + "\n" for ln in lines), encoding="utf-8", newline="\n")
        info = {"schema": "cryoev_pull/1", "kind": a.kind, "id": a.id, "source": str(src), "t": CM.utc_now(),
                "who": who, "via": via, "n_files": len(files), "bytes": total, "manifest": MANIFEST}
        (dest / INFO).write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"pulled {len(files)} file(s), {total / 1e6:.1f} MB -> {dest} (manifest {MANIFEST})")
        return 0, {"n_files": len(files), "bytes": total}

    return CM.run_write(a, f"pull {a.kind}", a.id, args, plan, act)
