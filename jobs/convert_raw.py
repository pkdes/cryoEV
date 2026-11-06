#!/usr/bin/env python3
"""Raw cryo-EM montages -> model-ready tiles (the lab's "3d. Tiles no scalebar - 14.7 A-px PNG" folders).

    python -m jobs.convert_raw --in <files or dirs> --out <new dir> [--recipe default|<YYYYMMDD>]
        [--sample-key <sample_key.csv>] [--gpu] [--processing-repo <dir>]

Inputs: MRC stacks with their `.mrc.mdoc` sidecar (pass the .mrc, the .mdoc, or a folder), or SerialEM TIFF montage
tiles with their `.idoc` (pass the .idoc, any of its .tif tiles, or a folder). A whole montage is always converted.
Montages are found and named by the recipe's own discovery layout when it fits the input format (so a session's
ALIASES apply: 20261027's `grid2-45krerun-1` is `grid2_104_45krerun_1`, as delivered), else by the processing repo's
standard flat layouts. Folders are searched one level deep; a sidecar no layout recognises is skipped there with a
warning (whole-grid atlases, for one). A sidecar named directly is always converted, named from its stem if need be.

Output: <out>/*.png + <out>/MANIFEST.csv (output_name, ..., nm_per_px, ...), self-contained, so it can be a
Predict source folder unchanged. <out> must be new or empty.

NOTHING ABOUT THE IMAGE IS DONE HERE. Every pixel comes from the lab's processing repo (setting CJ_PROCESSING_REPO,
or --processing-repo), imported, never copied:
  * export_detector_tiles.export_dataset: the bar-free tile export that wrote the delivered 3c/3d folders (render,
    INTER_AREA downsample to TARGET_A_PX, file names, MANIFEST.csv). It renders through
    cryoem_core.pipeline.render_variant_tiles, so the denoised pass inherits the clean pass's clip levels;
  * the recipe is a session config, datasets/process_<recipe>.py, used as-is (denoise, contrast, dtype, sampling).
    "default" is 20261012, the template the lab's WORKFLOW.md says to copy. A recipe is checked statically before it
    is run: one `DATASET = DatasetConfig(layout=<a session layout>(...))`, imports from cryoem_core and a few plain
    modules only, nothing else at top level. Old full-script processors (torch at import, prints) and combined sets
    (ManifestLayout) are refused without being executed;
  * cryoem_core.discovery layouts name the montages (grid1_1, ...) exactly as the session pipeline does.
This module only picks the montages, refuses what it cannot vouch for, points the export at <out>/.convert_work, and
moves the 3d tiles + MANIFEST.csv into <out>. The native-resolution copies the export also writes are deleted.

Refused (ConvertError; CLI exit 2, {"ok": false, "kind": "refused"}), nothing left in <out>: no repo; an unknown or
non-session recipe; an input that is not a montage; a sidecar without a finite, positive PixelSpacing (nm/px is never
defaulted: cryoem_core's own fallback is never reached); an MRC stack or idoc missing; two montages that would get the
same name; <out> not empty; a montage whose estimated peak memory (~14 bytes per montage pixel) exceeds the RAM free
now; --gpu with the GPU busy or missing (the same check as jobs/predict_local.py); with the default recipe, a sampling
outside 0.5x-2x of the recipe session's own (name --recipe to override); no tile written. Montages coarser than
14.724 A/px are dropped before rendering (no upsampling) and listed in warnings. Any other error: CLI exit 1,
{"ok": false, "kind": "failed"}. stdout is always one JSON object; the pipeline's progress goes to stderr, and its
"Warning:" lines are also returned in `warnings`.

Compute is CPU unless --gpu. Match the delivered set when reproducing one: 20261027 was rendered on the CPU, 20261012
on the GPU, and the two differ by +-1 grey level on <0.01% of pixels (the lab's gotcha 31). The device the core really
used is returned in `device`; a GPU request that fell back to the CPU is a warning.

Run it as a one-shot subprocess: it is slow (about 25-30 s per montage on the CPU), and it imports the processing repo
into its own interpreter (sys.path, module cache), so one process converts with one repo. A run that is killed leaves
<out>/.convert_work behind; delete it (or <out>) before trying again.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import dataclasses
import importlib.util
import io
import json
import math
import os
import re
import shutil
import sys
import traceback
from pathlib import Path

REPO_ENV = "CJ_PROCESSING_REPO"
DEFAULT_RECIPE = "20261012"          # WORKFLOW.md: "20261012 is the template to copy"
ENTRY_POINT = "export_detector_tiles.export_dataset"
WORK = ".convert_work"
MANIFEST = "MANIFEST.csv"
RECIPE_RE = re.compile(r"^[A-Za-z0-9_]+$")
BYTES_PER_MONTAGE_PX = 14            # peak working set of one montage render, per montage pixel (estimate)
SAMPLING_RANGE = (0.5, 2.0)          # default recipe: A/px allowed relative to the recipe session's own
SESSION_LAYOUTS = {"FlatGridPositionMrcLayout", "FlatGridPositionLayout", "FlatGridRegionLayout",
                   "FlatGridSquareHoleLayout", "GridSubfolderLayout", "GridSubfolderAreaLayout"}
RECIPE_IMPORTS = {"__future__", "sys", "os", "pathlib", "numpy", "math", "re", "dataclasses", "collections",
                  "typing", "cryoem_core"}
WARN_RE = re.compile(r"warning|skipped", re.IGNORECASE)


class ConvertError(Exception):
    """A refusal: the message says what is missing. Nothing is left in the output folder."""

    def __init__(self, msg, warnings=None):
        super().__init__(msg)
        self.warnings = list(warnings or [])


class _Tee(io.TextIOBase):
    """Forwards the pipeline's console output to `log` and keeps its warning lines."""

    def __init__(self, log):
        self.log, self.buf, self.caught = log, "", []

    def writable(self):
        return True

    def write(self, s):
        self.log.write(s)
        self.buf += s
        *lines, self.buf = self.buf.split("\n")
        self.caught += [ln.strip() for ln in lines if WARN_RE.search(ln)]
        return len(s)

    def flush(self):
        self.log.flush()


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
        import cryoem_core.device  # noqa: F401
        import cryoem_core.discovery  # noqa: F401
        import cryoem_core.metadata  # noqa: F401
        import cryoem_core.stitching  # noqa: F401
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


def _call_name(node):
    f = node.func if isinstance(node, ast.Call) else None
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def check_recipe_source(path: Path, name: str):
    """Static check of a recipe file; raises ConvertError, never executes it. A session recipe is docstring,
    imports (RECIPE_IMPORTS), sys.path.insert(...), plain assignments, one DATASET = DatasetConfig(layout=<session
    layout>(...)), and the `if __name__ == '__main__':` block."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    except (SyntaxError, UnicodeDecodeError, ValueError) as e:
        raise ConvertError(f"recipe {name}: not readable Python ({type(e).__name__})") from e
    bad, dataset = [], None
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Expr) and i == 0 and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                    else [node.module or ""] if node.level == 0 else ["." * node.level])
            bad += [f"import {m}" for m in mods if m.split(".")[0] not in RECIPE_IMPORTS]
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and \
                ast.unparse(node.value.func) == "sys.path.insert":
            continue
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "DATASET" for t in targets):
                dataset = node.value
        elif isinstance(node, ast.If) and ast.unparse(node.test).replace('"', "'") == "__name__ == '__main__'":
            continue
        else:
            bad.append(f"line {node.lineno}: {type(node).__name__}")
    if bad:
        raise ConvertError(f"recipe {name} is not a session config (refused without running it): "
                           f"{'; '.join(bad[:4])}")
    if _call_name(dataset) != "DatasetConfig":
        raise ConvertError(f"recipe {name} has no DATASET = DatasetConfig(...) (refused without running it)")
    layout = next((k.value for k in dataset.keywords if k.arg == "layout"), None)
    if _call_name(layout) not in SESSION_LAYOUTS:
        raise ConvertError(f"recipe {name}: layout {_call_name(layout) or '?'} is not a session layout "
                           f"(a combined set, not a recipe; refused without running it)")


def load_recipe(repo: Path, recipe: str, core, log):
    """(name, defaulted, DatasetConfig) from datasets/process_<name>.py: checked statically, then loaded by path
    (the folder is not a package, and a pip 'datasets' would shadow it) with its prints sent to `log`."""
    defaulted = recipe in (None, "", "default")
    name = DEFAULT_RECIPE if defaulted else str(recipe)
    if not RECIPE_RE.match(name):
        raise ConvertError(f"recipe {recipe!r} is not a session name")
    path = repo / "datasets" / f"process_{name}.py"
    if not path.is_file():
        raise ConvertError(f"unknown recipe {recipe!r}: no datasets/process_{name}.py in the processing repo")
    check_recipe_source(path, name)
    spec = importlib.util.spec_from_file_location(f"_cj_recipe_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(log):
        spec.loader.exec_module(mod)
    cfg = getattr(mod, "DATASET", None)
    if not isinstance(cfg, core.DatasetConfig):
        raise ConvertError(f"recipe {name}: DATASET is not a cryoem_core DatasetConfig")
    return name, defaulted, cfg


# ---------------------------------------------------------------- montages
class _FixedLayout:
    """A discovery layout that returns the montages picked here (export_dataset asks cfg.layout.find(raw_dir))."""

    def __init__(self, montages):
        self.montages = list(montages)

    def find(self, _dataset_dir=None):
        return list(self.montages)


def _is_mrc_sidecar(p: Path) -> bool:
    return p.name.lower().endswith(".mrc.mdoc")


def _sidecar_for(p: Path, core) -> Path:
    n = p.name.lower()
    if n.endswith((".mdoc", ".idoc")):
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


def _layouts_for(side: Path, core, recipe_layout):
    """The recipe's own layout first when it is a flat layout for this format (its ALIASES included), then the
    repo's standard flat layouts for the format."""
    d = core.discovery
    if _is_mrc_sidecar(side):
        std = [d.FlatGridPositionMrcLayout()]
    elif side.name.lower().endswith(".idoc"):
        std = [d.FlatGridPositionLayout(), d.FlatGridSquareHoleLayout()]
    else:
        return []
    flat = tuple(type(x) for x in std) + ((d.FlatGridRegionLayout,) if not _is_mrc_sidecar(side) else ())
    return ([recipe_layout] if isinstance(recipe_layout, flat) else []) + std


def _key(p: Path) -> str:
    return os.path.normcase(str(p.resolve()))


def find_montages(inputs, core, recipe_layout=None):
    """(montages, warnings): layout names where a layout knows the sidecar, else a name from the file stem (with a
    warning). Folder inputs take only what a layout recognises."""
    warnings, chosen, named = [], {}, []
    cache = {}

    def from_layouts(side):
        for lay in _layouts_for(side, core, recipe_layout):
            k = (_key(side.parent), id(lay) if lay is recipe_layout else type(lay).__name__)
            if k not in cache:
                cache[k] = {_key(m.idoc_path): m for m in lay.find(side.parent)}
            m = cache[k].get(_key(side))
            if m is not None:
                return m
        return None

    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for side in sorted(list(p.glob("*.mdoc")) + list(p.glob("*.idoc"))):
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
                               f"convert them separately", warnings)
        seen[m.basename] = m.idoc_path
    return montages, warnings


def sidecar_a_px(path: Path, core):
    """PixelSpacing from the sidecar header as a finite positive float, else None."""
    gp, _pieces = core.metadata.parse_metadata(path)
    try:
        ps = float(gp.get("PixelSpacing"))
    except (TypeError, ValueError):
        return None
    return ps if math.isfinite(ps) and ps > 0 else None


def check_montage(m, core):
    """(A/px, estimated peak bytes, warnings) for one montage, or ConvertError. Sets the placement convention."""
    ps = sidecar_a_px(m.idoc_path, core)
    if ps is None:
        raise ConvertError(f"{m.idoc_path.name}: no finite, positive PixelSpacing in the sidecar header; "
                           f"nm/px is never defaulted")
    gp, pieces = core.metadata.parse_metadata(m.idoc_path)
    if not pieces:
        raise ConvertError(f"{m.idoc_path.name}: the sidecar lists no tiles")
    warnings = []
    tiff = any(pc.get("_file") for pc in pieces)
    if tiff:
        files = [pc["_file"] for pc in pieces if pc.get("_file")]
        lost = [f for f in files if not (m.idoc_path.parent / f).is_file()]
        if len(lost) == len(files):
            raise ConvertError(f"{m.idoc_path.name}: none of its TIFF tiles are beside it")
        if lost:
            warnings.append(f"{m.basename}: {len(lost)} TIFF tile(s) missing, skipped by the pipeline: "
                            f"{', '.join(lost[:5])}")
        n = len(files)
    else:
        mrc = core.stitching.mrc_path_for(m, gp)
        if not mrc.is_file():
            raise ConvertError(f"{m.idoc_path.name}: no MRC stack beside it (looked for {mrc.name})")
        n = sum(1 for pc in pieces if "PieceCoordinates" in pc)
    try:
        w, h = (int(v) for v in gp.get("ImageSize", "").split()[:2])
    except ValueError:
        raise ConvertError(f"{m.idoc_path.name}: no ImageSize in the sidecar header") from None
    # cryoem_core: "yflip" for idoc stage coords, "direct" for MRC image coords. Only the r/c in tile names
    # depend on it; the pixels do not.
    m.placement = m.placement or ("yflip" if tiff else "direct")
    return ps, w * h * max(n, 1) * BYTES_PER_MONTAGE_PX, warnings


def available_ram():
    """Bytes of RAM free now, or None when it cannot be read."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def recipe_sampling(cfg, core):
    """The A/px values of the recipe's own session, from its sidecars (read-only); else the recipe's
    contrast_level_sampling_A; else an empty set."""
    found = set()
    try:
        raw = Path(cfg.raw_dir)
        if raw.is_dir():
            with contextlib.redirect_stdout(io.StringIO()):
                for m in cfg.layout.find(raw):
                    ps = sidecar_a_px(Path(m.idoc_path), core)
                    if ps:
                        found.add(round(ps, 4))
    except Exception:
        found = set()
    if not found and getattr(cfg, "contrast_level_sampling_A", None):
        found.add(float(cfg.contrast_level_sampling_A))
    return found


def gpu_check():
    """jobs/predict_local.py's GPU check: (ok, note)."""
    if __package__:
        from . import predict_local as PL
    else:
        here = Path(__file__).resolve().parent
        for p in (here, here.parent):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        import predict_local as PL
    dev, note, _kind = PL.gpu_check(allow_cpu=False)
    return dev == "cuda", note


# ---------------------------------------------------------------- convert
def _read_manifest(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def convert(inputs, out_dir, *, recipe="default", sample_key=None, gpu=False, processing_repo_dir=None,
            log=None) -> dict:
    """Raw montages -> <out_dir>/*.png + MANIFEST.csv. Returns {"out_dir", "recipe", "recipe_defaulted",
    "entry_point", "compute", "device", "montages", "a_px": {montage: A/px}, "n_tiles", "tiles": [{"file",
    "nm_per_px", "source_file", "montage", "sample_label", "size"}], "nm_per_px": [distinct], "warnings"}.
    Raises ConvertError on a refusal (out_dir left as it was, .warnings carries what was found so far).
    The pipeline's console output goes to `log` (default stderr)."""
    tee = _Tee(log or sys.stderr)
    warnings = []
    try:
        return _convert(inputs, Path(out_dir), recipe, sample_key, bool(gpu), processing_repo_dir, tee, warnings)
    except ConvertError as e:
        e.warnings = list(dict.fromkeys(warnings + e.warnings + tee.caught))
        raise


def _convert(inputs, out_dir, recipe, sample_key, gpu, processing_repo_dir, tee, warnings):
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ConvertError(f"output folder is not empty: {out_dir}")
    if not inputs:
        raise ConvertError("no inputs")
    repo = processing_repo(processing_repo_dir)
    edt, core = import_pipeline(repo)
    recipe_name, defaulted, base_cfg = load_recipe(repo, recipe, core, tee)
    if sample_key is not None and not Path(sample_key).is_file():
        raise ConvertError(f"sample key not found: {sample_key}")

    with contextlib.redirect_stdout(tee):
        montages, w = find_montages(inputs, core, base_cfg.layout)
    warnings += w
    if not montages:
        raise ConvertError("no montages found in the inputs")
    target = float(edt.TARGET_A_PX)
    a_px, peak, keep = {}, {}, []
    for m in montages:
        ps, need, w = check_montage(m, core)
        warnings += w
        a_px[m.basename], peak[m.basename] = ps, need
        if target / ps < 1:
            warnings.append(f"{m.basename}: {ps:.3f} A/px is coarser than {target} A/px; skipped (no upsampling)")
        else:
            keep.append(m)
    if not keep:
        raise ConvertError(f"every montage is coarser than {target} A/px; nothing to convert")
    montages = keep

    # The recipe was tuned on its own session's sampling and format: say when the input is not that.
    ref = recipe_sampling(base_cfg, core)
    rec_mrc = isinstance(base_cfg.layout, core.discovery.FlatGridPositionMrcLayout)
    for m in montages:
        ps = a_px[m.basename]
        if ref and not any(abs(ps - r) <= 0.01 * r for r in ref):
            msg = (f"{m.basename}: {ps:.3f} A/px is not a sampling of recipe {recipe_name}'s session "
                   f"({', '.join('%g' % r for r in sorted(ref))} A/px)")
            if defaulted and not (SAMPLING_RANGE[0] * min(ref) <= ps <= SAMPLING_RANGE[1] * max(ref)):
                raise ConvertError(msg + "; outside the default recipe's range: name a --recipe to convert it",
                                   warnings)
            warnings.append(msg)
        if _is_mrc_sidecar(m.idoc_path) != rec_mrc:
            warnings.append(f"{m.basename}: recipe {recipe_name} is for "
                            f"{'MRC' if rec_mrc else 'idoc + TIFF'} montages, this one is not")
    if not ref:
        warnings.append(f"recipe {recipe_name}: its session's sampling is unknown here; A/px not compared")
    if sample_key is None:
        warnings.append("no sample key: sample_label is 'Grid <n>' (pass --sample-key for the session's labels)")

    free = available_ram()
    worst = max(montages, key=lambda m: peak[m.basename])
    if free is None:
        warnings.append("free RAM unreadable; memory not checked")
    elif peak[worst.basename] > free:
        raise ConvertError(f"{worst.basename}: needs about {peak[worst.basename] / 2**30:.1f} GB, "
                           f"{free / 2**30:.1f} GB free now")
    if gpu:
        ok, note = gpu_check()
        if not ok:
            raise ConvertError(f"--gpu: {note}")

    made_out = not out_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / WORK
    try:
        work.mkdir()
        if sample_key is not None:
            shutil.copyfile(sample_key, work / "sample_key.csv")
        # The recipe as-is; only where it reads from and writes to changes. Outputs land in <out>/.convert_work.
        cfg = dataclasses.replace(base_cfg, dataset_dir=work, raw_subdir=None, notes_subdir=None,
                                  layout=_FixedLayout(montages), use_gpu=gpu,
                                  name=f"convert_raw[{recipe_name}]")
        args = argparse.Namespace(out=str(work / "native"), grids=None, limit=0, gpu=gpu)
        small = work / (edt.SMALL_DIR % ("%.1f" % target))
        with contextlib.redirect_stdout(tee):
            print(f"convert_raw: {len(montages)} montage(s), recipe {recipe_name}, via {ENTRY_POINT}")
            try:
                rc = edt.export_dataset(args, cfg)
            except SystemExit as e:          # render_ref stops the run with SystemExit(message)
                raise ConvertError(f"pipeline stopped: {e}") from e
        if rc not in (0, None) or not (small / MANIFEST).is_file():
            raise ConvertError(f"pipeline returned {rc} and wrote no {MANIFEST}")
        rows = _read_manifest(small / MANIFEST)
        if not rows:
            raise ConvertError("no tile written")
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

    device = "CPU (NumPy)"
    if gpu:
        device = core.device.describe_device(True)
        if not device.startswith("GPU"):
            warnings.append(f"--gpu: the core found no usable GPU and ran on the CPU ({device}); "
                            f"MANIFEST.csv still says GPU")
    warnings[:] = list(dict.fromkeys(warnings + tee.caught))
    tiles =[{"file": r["output_name"], "nm_per_px": float(r["nm_per_px"]), "source_file": r["source_file"],
              "montage": r["montage"], "sample_label": r["sample_label"], "size": r["output_size"]} for r in rows]
    return {"out_dir": str(out_dir), "recipe": recipe_name, "recipe_defaulted": defaulted,
            "entry_point": ENTRY_POINT, "compute": "GPU" if device.startswith("GPU") else "CPU", "device": device,
            "montages": [m.basename for m in montages], "a_px": {m.basename: a_px[m.basename] for m in montages},
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
        print(json.dumps({"ok": False, "kind": "refused", "error": str(e), "warnings": e.warnings}))
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "kind": "failed", "error": f"{type(e).__name__}: {e}", "warnings": []}))
        return 1
    for w in res["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    print(json.dumps({"ok": True, **res}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
