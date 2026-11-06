#!/usr/bin/env python3
"""Raw cryo-EM montages -> model-ready tiles (the lab's "3d. Tiles no scalebar - 14.7 A-px PNG" folders).

    python -m jobs.convert_raw --in <files or dirs> --out <new dir> [--recipe default|<YYYYMMDD>]
        [--sample-key <sample_key.csv>] [--gpu] [--processing-repo <dir>]

Inputs: MRC stacks with their `.mrc.mdoc` sidecar (pass the .mrc, the .mdoc, or a folder), or SerialEM TIFF montage
tiles with their `.idoc` (pass the .idoc, any of its .tif tiles, or a folder). A whole montage is always converted.
Folders are searched one level deep, by the processing repo's own discovery layouts; sidecars no layout recognises
are skipped there with a warning (whole-grid atlases, for one). A sidecar named directly is always converted.

Output: <out>/*.png + <out>/MANIFEST.csv (output_name, ..., nm_per_px, ...), self-contained, so it can be a
Predict source folder unchanged. <out> must be new or empty.

NOTHING ABOUT THE IMAGE IS DONE HERE. Every pixel comes from the lab's processing repo (setting CJ_PROCESSING_REPO,
or --processing-repo), imported, never copied:
  * export_detector_tiles.export_dataset: the bar-free tile export that wrote the delivered 3c/3d folders (render,
    INTER_AREA downsample to TARGET_A_PX, file names, MANIFEST.csv). It renders through
    cryoem_core.pipeline.render_variant_tiles, so the denoised pass inherits the clean pass's clip levels;
  * the recipe is a session config, datasets/process_<recipe>.py, read as-is (denoise, contrast, dtype, sampling).
    "default" is 20261012, the template the lab's WORKFLOW.md says to copy;
  * cryoem_core.discovery layouts name the montages (grid1_1, ...) exactly as the session pipeline does.
This module only picks the montages, refuses what it cannot vouch for, points the export at a scratch folder inside
<out>, and moves the 3d tiles + MANIFEST.csv into <out>. The native-resolution copies the export also writes are
deleted.

Refused (ConvertError, CLI exit 2), nothing left in <out>: no repo; an unknown recipe; an input that is not a
montage; a sidecar without PixelSpacing (nm/px is never defaulted -- cryoem_core's own fallback is never reached);
an MRC stack or idoc missing; two montages that would get the same name; <out> not empty; no tile written.
Montages coarser than 14.724 A/px are skipped by the export (no upsampling) and listed in warnings.

Compute is CPU unless --gpu: the delivered sets up to 2026-10 were CPU renders, and a GPU render differs by +-1 grey
level on a few pixels (the lab's gotcha 31). Slow (seconds to a minute per montage): run it in a worker process.
stdout carries only the final JSON result; the pipeline's progress goes to stderr.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ENV = "CJ_PROCESSING_REPO"
DEFAULT_RECIPE = "20261012"          # WORKFLOW.md: "20261012 is the template to copy"
ENTRY_POINT = "export_detector_tiles.export_dataset"
WORK = ".convert_work"
MANIFEST = "MANIFEST.csv"
RECIPE_RE = re.compile(r"^[A-Za-z0-9_]+$")


class ConvertError(Exception):
    """A refusal: the message says what is missing. Nothing is left in the output folder."""


# ---------------------------------------------------------------- the processing repo
def processing_repo(path=None) -> Path:
    raw = path or os.environ.get(REPO_ENV)
    if not raw:
        raise ConvertError(f"no processing repo: set {REPO_ENV} (or pass --processing-repo)")
    repo = Path(raw)
    missing = [p for p in ("cryoem_core/__init__.py", "export_detector_tiles.py", "datasets")
               if not (repo / p).exists()]
    if missing:
        raise ConvertError(f"{repo} is not the processing repo: missing {', '.join(missing)}")
    return repo


def import_pipeline(repo: Path):
    """(export_detector_tiles, cryoem_core) from the repo. Bytecode writes are switched off first: the repo is
    read-only to us, and importing would otherwise drop .pyc files into it."""
    sys.dont_write_bytecode = True
    r = str(repo.resolve())
    if r not in sys.path:
        sys.path.insert(0, r)
    try:
        import cryoem_core  # noqa: F401
        import export_detector_tiles as edt
    except ImportError as e:
        raise ConvertError(f"processing repo not importable: {type(e).__name__}: {e}") from e
    for mod in (sys.modules["cryoem_core"], edt):
        if not os.path.normcase(str(Path(mod.__file__).resolve())).startswith(os.path.normcase(r) + os.sep):
            raise ConvertError(f"{mod.__name__} is already imported from {mod.__file__}, not from {repo}; "
                               f"run each repo in its own process")
    for name in ("export_dataset", "TARGET_A_PX", "SMALL_DIR", "DATASET_FIELDS"):
        if not hasattr(edt, name):
            raise ConvertError(f"processing repo has no export_detector_tiles.{name}")
    return edt, sys.modules["cryoem_core"]


def load_recipe(repo: Path, recipe: str):
    """The DatasetConfig of datasets/process_<name>.py, loaded by path (the folder is not a package, and a pip
    'datasets' would shadow it). The module's own __main__ guard keeps it from running anything."""
    name = DEFAULT_RECIPE if recipe in (None, "", "default") else str(recipe)
    if not RECIPE_RE.match(name):
        raise ConvertError(f"recipe {recipe!r} is not a session name")
    path = repo / "datasets" / f"process_{name}.py"
    if not path.is_file():
        raise ConvertError(f"unknown recipe {recipe!r}: no datasets/process_{name}.py in the processing repo")
    spec = importlib.util.spec_from_file_location(f"_cj_recipe_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = getattr(mod, "DATASET", None)
    if cfg is None:
        raise ConvertError(f"recipe {name}: datasets/process_{name}.py has no DATASET")
    return name, cfg


# ---------------------------------------------------------------- montages
class _FixedLayout:
    """A discovery layout that returns the montages picked here (export_dataset asks cfg.layout.find(raw_dir))."""

    def __init__(self, montages):
        self.montages = list(montages)

    def find(self, _dataset_dir=None):
        return list(self.montages)


def _is_sidecar(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".mdoc") or n.endswith(".idoc")


def _sidecar_for(p: Path, core) -> Path:
    n = p.name.lower()
    if _is_sidecar(p):
        return p
    if n.endswith(".mrc"):
        side = p.with_name(p.name + ".mdoc")
        if not side.is_file():
            raise ConvertError(f"{p.name}: no {side.name} beside it (nm/px and tile positions come from it)")
        return side
    if n.endswith((".tif", ".tiff")):
        for idoc in sorted(p.parent.glob("*.idoc")):
            _g, pieces = core.metadata.parse_metadata(idoc)
            if any(pc.get("_file") == p.name for pc in pieces):
                return idoc
        raise ConvertError(f"{p.name}: no .idoc in its folder lists this tile")
    raise ConvertError(f"{p.name}: not a raw montage input (.mrc, .mdoc, .idoc, .tif)")


def _layouts_for(side: Path, core):
    d = core.discovery
    if side.name.lower().endswith(".mrc.mdoc"):
        return [d.FlatGridPositionMrcLayout()]
    if side.name.lower().endswith(".idoc"):
        return [d.FlatGridPositionLayout(), d.FlatGridSquareHoleLayout()]
    return []


def _key(p: Path) -> str:
    return os.path.normcase(str(p.resolve()))


def find_montages(inputs, core):
    """[(MontageRef, warnings)] for the inputs: layout names where a layout knows the sidecar, else a name from the
    file stem (with a warning). Folder inputs take only what a layout recognises."""
    warnings, chosen, named = [], {}, []
    cache = {}

    def from_layouts(side):
        for lay in _layouts_for(side, core):
            k = (_key(side.parent), type(lay).__name__)
            if k not in cache:
                with contextlib.redirect_stdout(sys.stderr):
                    cache[k] = {_key(m.idoc_path): m for m in lay.find(side.parent)}
            m = cache[k].get(_key(side))
            if m is not None:
                return m
        return None

    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            sides = sorted(list(p.glob("*.mdoc")) + list(p.glob("*.idoc")))
            for side in sides:
                m = from_layouts(side)
                if m is None:
                    warnings.append(f"skipped {side.name}: not a montage of a known layout (atlas?); "
                                    f"name the file to convert it anyway")
                else:
                    chosen.setdefault(_key(side), m)
        elif p.is_file():
            named.append(_sidecar_for(p, core))
        else:
            raise ConvertError(f"input not found: {p}")

    for side in named:
        if _key(side) in chosen:
            continue
        m = from_layouts(side)
        if m is None:
            stem = side.name[: -len(".mdoc")] if side.name.lower().endswith(".mdoc") else side.stem
            stem = stem[: -len(".mrc")] if stem.lower().endswith(".mrc") else stem
            g = re.search(r"grid\s*(\d+)", stem, re.IGNORECASE)
            r = re.findall(r"(\d+)", stem)
            base = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_") or "montage"
            m = core.discovery.MontageRef(int(g.group(1)) if g else 0, int(r[-1]) if r else 0, side, base)
            warnings.append(f"{side.name}: no layout knows this name; tiles are named {base}_r*c*_t*")
        chosen[_key(side)] = m

    montages = sorted(chosen.values(), key=lambda m: (m.grid, m.region, m.basename))
    seen = {}
    for m in montages:
        if m.basename in seen:
            raise ConvertError(f"{m.idoc_path.name} and {seen[m.basename].name} would both be named {m.basename}; "
                               f"convert them separately")
        seen[m.basename] = m.idoc_path
    return montages, warnings


def check_montage(m, core, target_a_px):
    """Pixel size from the sidecar (refused if absent), the placement convention, and what the pipeline would only
    warn about. Returns (A/px, warnings)."""
    gp, pieces = core.metadata.parse_metadata(m.idoc_path)
    ps_raw = gp.get("PixelSpacing")
    try:
        ps = float(ps_raw)
    except (TypeError, ValueError):
        ps = None
    if not ps or ps <= 0:
        raise ConvertError(f"{m.idoc_path.name}: no PixelSpacing in the sidecar header; nm/px is never defaulted")
    if not pieces:
        raise ConvertError(f"{m.idoc_path.name}: the sidecar lists no tiles")
    warnings = []
    tiff = any(pc.get("_file") for pc in pieces)
    if tiff:
        lost = [pc["_file"] for pc in pieces if pc.get("_file") and not (m.idoc_path.parent / pc["_file"]).is_file()]
        if len(lost) == sum(1 for pc in pieces if pc.get("_file")):
            raise ConvertError(f"{m.idoc_path.name}: none of its TIFF tiles are beside it")
        if lost:
            warnings.append(f"{m.basename}: {len(lost)} TIFF tile(s) missing, skipped by the pipeline: "
                            f"{', '.join(lost[:5])}")
    else:
        mrc = core.stitching.mrc_path_for(m, gp)
        if not mrc.is_file():
            raise ConvertError(f"{m.idoc_path.name}: no MRC stack beside it (looked for {mrc.name})")
    # cryoem_core: "yflip" for idoc stage coords, "direct" for MRC image coords. Only the r/c in tile names
    # depend on it; the pixels do not.
    m.placement = m.placement or ("yflip" if tiff else "direct")
    if target_a_px / ps < 1:
        warnings.append(f"{m.basename}: {ps:.3f} A/px is coarser than {target_a_px} A/px; skipped (no upsampling)")
    return ps, warnings


# ---------------------------------------------------------------- convert
def _read_manifest(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def convert(inputs, out_dir, *, recipe="default", sample_key=None, gpu=False, processing_repo_dir=None,
            log=None) -> dict:
    """Raw montages -> <out_dir>/*.png + MANIFEST.csv. Returns {"out_dir", "recipe", "entry_point", "compute",
    "montages", "n_tiles", "tiles": [{"file", "nm_per_px", "source_file", "montage", "sample_label", "size"}],
    "nm_per_px": [distinct values], "warnings"}. Raises ConvertError on a refusal (out_dir left as it was).
    The pipeline's console output goes to `log` (default stderr)."""
    log = log or sys.stderr
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ConvertError(f"output folder is not empty: {out_dir}")
    repo = processing_repo(processing_repo_dir)
    edt, core = import_pipeline(repo)
    import cryoem_core.discovery  # noqa: F401  (submodules used below, by attribute)
    import cryoem_core.metadata  # noqa: F401
    import cryoem_core.stitching  # noqa: F401
    recipe_name, base_cfg = load_recipe(repo, recipe)
    if not inputs:
        raise ConvertError("no inputs")
    if sample_key is not None and not Path(sample_key).is_file():
        raise ConvertError(f"sample key not found: {sample_key}")

    montages, warnings = find_montages(inputs, core)
    if not montages:
        raise ConvertError("no montages found in the inputs")
    target = float(edt.TARGET_A_PX)
    for m in montages:
        _ps, w = check_montage(m, core, target)
        warnings += w
    if sample_key is None:
        warnings.append("no sample key: sample_label is 'Grid <n>' (pass --sample-key for the session's labels)")

    made_out = not out_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / WORK
    try:
        work.mkdir()
        if sample_key is not None:
            shutil.copyfile(sample_key, work / "sample_key.csv")
        # The recipe as-is; only where it reads from and writes to changes. Outputs land in <out>/.convert_work.
        cfg = dataclasses.replace(base_cfg, dataset_dir=work, raw_subdir=None, notes_subdir=None,
                                  layout=_FixedLayout(montages), use_gpu=bool(gpu),
                                  name=f"convert_raw[{recipe_name}]")
        args = argparse.Namespace(out=str(work / "native"), grids=None, limit=0, gpu=bool(gpu))
        small = work / (edt.SMALL_DIR % ("%.1f" % target))
        with contextlib.redirect_stdout(log):
            print(f"convert_raw: {len(montages)} montage(s), recipe {recipe_name}, via {ENTRY_POINT}")
            try:
                rc = edt.export_dataset(args, cfg)
            except SystemExit as e:          # render_ref stops the run with SystemExit(message)
                raise ConvertError(f"pipeline stopped: {e}") from e
        if rc not in (0, None) or not (small / MANIFEST).is_file():
            raise ConvertError(f"pipeline returned {rc} and wrote no {MANIFEST}")
        rows = _read_manifest(small / MANIFEST)
        if not rows:
            raise ConvertError("no tile written (every montage skipped)")
        for r in rows:
            if not (small / r["output_name"]).is_file() or not r.get("nm_per_px"):
                raise ConvertError(f"pipeline MANIFEST row without its tile or nm/px: {r.get('output_name')}")
        for r in rows:
            os.replace(small / r["output_name"], out_dir / r["output_name"])
        os.replace(small / MANIFEST, out_dir / MANIFEST)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        if made_out:
            shutil.rmtree(out_dir, ignore_errors=True)
        else:
            for p in list(out_dir.iterdir()):
                (shutil.rmtree if p.is_dir() else os.remove)(p)
        raise
    shutil.rmtree(work, ignore_errors=True)

    tiles = [{"file": r["output_name"], "nm_per_px": float(r["nm_per_px"]), "source_file": r["source_file"],
              "montage": r["montage"], "sample_label": r["sample_label"], "size": r["output_size"]} for r in rows]
    return {"out_dir": str(out_dir), "recipe": recipe_name, "entry_point": ENTRY_POINT,
            "compute": "GPU" if gpu else "CPU", "montages": [m.basename for m in montages],
            "n_tiles": len(tiles), "tiles": tiles,
            "nm_per_px": sorted({t["nm_per_px"] for t in tiles}), "warnings": warnings}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, type=Path,
                    help=".mrc / .mrc.mdoc / .idoc / .tif files, or folders of them")
    ap.add_argument("--out", required=True, type=Path, help="new or empty folder for tiles + MANIFEST.csv")
    ap.add_argument("--recipe", default="default", help="default (= 20261012) or a session name with a "
                                                         "datasets/process_<name>.py in the processing repo")
    ap.add_argument("--sample-key", type=Path, help="the session's sample_key.csv (#,Label)")
    ap.add_argument("--gpu", action="store_true", help="GPU denoise/percentiles (+-1 grey level vs CPU)")
    ap.add_argument("--processing-repo", type=Path, help=f"overrides ${REPO_ENV}")
    a = ap.parse_args(argv)
    try:
        res = convert(a.inputs, a.out, recipe=a.recipe, sample_key=a.sample_key, gpu=a.gpu,
                      processing_repo_dir=a.processing_repo)
    except ConvertError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    for w in res["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    print(json.dumps({"ok": True, **res}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
