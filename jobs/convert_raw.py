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
defaulted: cryoem_core's own fallback is never reached); an MRC stack or idoc missing; a sidecar whose [Image = ...]
names a tile that is not a plain file name inside its own folder (an absolute, UNC or '..' name; checked before any
tile is opened); two montages that would get the
same name; <out> not empty; a montage whose estimated peak memory (~20 bytes per montage pixel, x2 for a float64 recipe) exceeds the RAM free
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

MONTAGES AND ACQUISITION (round 7, BUILD_CONTRACTS.md section 13), both AFTER the tiles are final and outside the
cleanup that empties <out> on a failure, so neither can fail the conversion or touch a tile:
    --montages-out <dir>     a separate stitch pass: <dir>/<montage>_stitched.jpg, layout.json, stitch.json
    --acquisition-out <file> acquisition.json (per montage + session summary)
  The stitch pass reproduces cryoem_core.pipeline.run_convert's composite step by calling the same functions in the
  same order (load_tiles; _variant_tiles; stitch_composite for the clean variant first, only for its clip levels, then
  the denoised variant with those levels when denoise_match_contrast; the last variant not in skip_variants is the
  one written; downsample_and_save at min(max_long_edge, 2000) px and the recipe's JPEG quality; stitch_composite
  draws the composite scalebar from the pieces' magnification). Placement is the montage's own (yflip for idoc
  stage coordinates, direct for MRC). It never writes into the tiles folder; it builds in .<dir>.stitch_work beside
  <dir> and swaps it in by rename. A montage that fails is listed in stitch.json (state "failed") and in `warnings`;
  the others are still written -- unless <dir> already holds a finished set (stitch.json done): a pass with any
  failure then leaves it untouched (the result's stitch doc says kept_previous). stitch.json and the result carry
  "processing" {commit, recipe, recipe_md5}; --stitch-only warns when they differ from the conversion's. layout.json boxes each tile in montage-JPG pixels, keyed by its MANIFEST output_name (the same
  geometry as assemble_canvas: tile_offset with the loaded tile's own size). acquisition.json comes from the cryoEV
  registry importer's file_summary / session_acquisition / scope_model (scope model only: never the raw header,
  a serial, a site or a path).
    --stitch-only <conv dir> backfill for a finished portal conversion (<dir>/convert.json + tiles/MANIFEST.csv and
                             its raw inputs still there): writes <dir>/montages and <dir>/acquisition.json only.
  Each montage prints a progress line ("stitch [i/n] ...") to stderr.

CONVERSION SETTINGS (round 7 phase 2, BUILD_CONTRACTS.md section 14): read, preview, override.
    describe --recipe <session> [--json]         the recipe's real values (read-only): denoise (describe_denoise),
                                                  clip lo/hi %, level sampling, pixel targets, where it came from
    describe --conv <raw_tiles/cid> [--json]      what a finished conversion used: convert.json params_effective
                                                  (recorded by the conversion), else the recipe's values now
    preview --conv <raw_tiles/cid> --tile <output_name> [--params JSON] --out <dir> [--gpu] [--json]
        before.png (the conversion's own settings: the recipe + convert.json params_overrides, i.e. its tile) and
        after.png (that base + params; values are absolute; the same settings = one render, copied) of one tile,
        rendered from the kept raw input
        by the conversion's own entry point (export_dataset -> render_ref: the clean pass first, its clip levels
        handed to the denoised pass, the INTER_AREA downsample, the PNG save), with load_tiles narrowed to that one
        tile (tiles are rendered independently: per-tile levels, per-tile denoise). Writes only into --out. CPU
        unless --gpu (a GPU conversion's tiles differ by +-1 grey level from a CPU preview: gotcha 31).
    --params JSON  (on a conversion) overrides the recipe by dataclasses.replace; absent = the recipe untouched,
        byte for byte. Keys (anything else, or out of range: refused, exit 2):
            clip_lo_pct 0-5 -> contrast_lo_pct     clip_hi_pct 95-100 -> contrast_hi_pct
            bandpass_nm 0.5-10 -> denoise_min_nm (bandpass recipes with denoise on after the overrides only;
                                  below denoise_max_nm; reported null whenever denoise is off)
            denoise true|false -> denoise (false also renders the clean variant even when skip_variants names it:
                                  otherwise no variant would be left to write)
        The result carries params_effective (all four, after the overrides) and params_overrides; a portal
        conversion (--out <dir>/tiles beside <dir>/convert.json) also records both, and the tiles' variant, in
        convert.json -- overrides that can't be recorded fail the conversion (exit 1) -- and --stitch-only
        re-applies convert.json params_overrides so the montages match the tiles. tiles[] and convert.json carry
        "variant" ("denoised" | "clean"); the file names stay the processing repo's ("..._denoised_nobar_...").
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import dataclasses
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO_ENV = "CJ_PROCESSING_REPO"
DEFAULT_RECIPE = "20261012"          # WORKFLOW.md: "20261012 is the template to copy"
ENTRY_POINT = "export_detector_tiles.export_dataset"
WORK = ".convert_work"
MANIFEST = "MANIFEST.csv"
RECIPE_RE = re.compile(r"^[A-Za-z0-9_]+$")
# Peak working set per montage piece pixel (tile w x h x pieces), float32 recipes; float64 ones x2. Measured
# 2026-11-09 on 20261012 grid1-pos1 (9 x 5760x4092, composite 16128x11124, CPU): tile export 3.52 GB (16.6 B/px),
# the stitch pass 5.06 GB (25.6 B/px), one after the other in one process.
# With a margin over the measurement:
BYTES_PER_MONTAGE_PX = 20            # the tile export: a montage above the RAM free now refuses the conversion
STITCH_BYTES_PER_PX = 30             # the stitch pass: a montage above the RAM free now is not stitched (warning)
SAMPLING_RANGE = (0.5, 2.0)          # default recipe: A/px allowed relative to the recipe session's own
SESSION_LAYOUTS = {"FlatGridPositionMrcLayout", "FlatGridPositionLayout", "FlatGridRegionLayout",
                   "FlatGridSquareHoleLayout", "GridSubfolderLayout", "GridSubfolderAreaLayout"}
RECIPE_IMPORTS = {"__future__", "sys", "os", "pathlib", "numpy", "math", "re", "dataclasses", "collections",
                  "typing", "cryoem_core"}
WARN_RE = re.compile(r"warning|skipped", re.IGNORECASE)
MONTAGE_MAX_EDGE = 2000              # composites: long edge cap (the recipes' own max_long_edge is 2000 too)
STITCH_JSON, LAYOUT_JSON, ACQ_JSON = "stitch.json", "layout.json", "acquisition.json"
LAYOUT_SCHEMA, ACQ_SCHEMA = "montage_layout/1", "acquisition/1"
STITCH_WORK = ".stitch_work"
CONVERT_JSON, CONV_TILES, CONV_MONTAGES = "convert.json", "tiles", "montages"   # a portal conversion folder
CONV_INPUT = "input"
# --params: key -> (DatasetConfig field, lo, hi); "denoise" is a bool (BUILD_CONTRACTS.md section 14)
PARAM_RANGES = {"clip_lo_pct": ("contrast_lo_pct", 0.0, 5.0), "clip_hi_pct": ("contrast_hi_pct", 95.0, 100.0),
                "bandpass_nm": ("denoise_min_nm", 0.5, 10.0)}
PARAM_KEYS = (*PARAM_RANGES, "denoise")
PREVIEW_WORK = ".preview_work"


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


def repo_commit(repo: Path) -> str | None:
    """The processing repo's HEAD commit, read from its .git files (no git is run); None when it can't be read."""
    try:
        git = Path(repo) / ".git"
        if git.is_file():                                    # a worktree: "gitdir: <path>"
            git = Path(git.read_text(encoding="utf-8").split(":", 1)[1].strip())
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head[:40] or None
        ref = head.split(":", 1)[1].strip()
        common = git
        if (git / "commondir").is_file():
            common = (git / (git / "commondir").read_text(encoding="utf-8").strip()).resolve()
        for base in (git, common):
            if (base / ref).is_file():
                return (base / ref).read_text(encoding="utf-8").strip()[:40] or None
        for line in (common / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split()[0][:40]
    except (OSError, IndexError, ValueError):
        pass
    return None


def processing_info(repo: Path, recipe_name: str) -> dict:
    """What rendered the pixels: {"commit": the repo's HEAD or None, "recipe", "recipe_md5"} (recorded with the
    tiles and in stitch.json, compared by --stitch-only)."""
    p = Path(repo) / "datasets" / f"process_{recipe_name}.py"
    try:
        rmd5 = hashlib.md5(p.read_bytes()).hexdigest()
    except OSError:
        rmd5 = None
    return {"commit": repo_commit(repo), "recipe": recipe_name, "recipe_md5": rmd5}


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


# ---------------------------------------------------------------- --params (overrides on the recipe)
def parse_params(raw) -> dict | None:
    """--params JSON (a str, or an already-parsed dict) -> the validated overrides, or None when absent.
    ConvertError (exit 2) for bad JSON, a non-object, an unknown key, a wrong type or a value out of range."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as e:
            raise ConvertError(f"--params: not valid JSON ({e})") from None
    if not isinstance(raw, dict):
        raise ConvertError("--params: must be a JSON object, e.g. {\"clip_hi_pct\": 99.5}")
    unknown = sorted(str(k) for k in raw if k not in PARAM_KEYS)
    if unknown:
        raise ConvertError(f"--params: unknown key(s) {', '.join(unknown)[:200]}; allowed: {', '.join(PARAM_KEYS)}")
    out = {}
    for k, v in raw.items():
        if k == "denoise":
            if not isinstance(v, bool):
                raise ConvertError("--params: denoise must be true or false")
            out[k] = v
            continue
        _field, lo, hi = PARAM_RANGES[k]
        # (a range comparison is False for NaN and +-inf, and safe for an int too big for a float)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            raise ConvertError(f"--params: {k} must be a number from {lo:g} to {hi:g} (got {str(v)[:40]})")
        out[k] = float(v)
    return out


def apply_params(cfg, params):
    """The recipe with the overrides, by dataclasses.replace. No params: cfg itself, untouched."""
    if not params:
        return cfg
    ch = {PARAM_RANGES[k][0]: v for k, v in params.items() if k in PARAM_RANGES}
    if "bandpass_nm" in params:
        if not params.get("denoise", cfg.denoise):
            raise ConvertError("--params: bandpass_nm needs denoise on (it sets the denoise filter's cutoff); "
                               "leave it out, or add \"denoise\": true")
        if cfg.denoise_method != "bandpass":
            raise ConvertError(f"--params: bandpass_nm needs a bandpass recipe; this one denoises with "
                               f"{cfg.denoise_method}")
        if cfg.denoise_max_nm and params["bandpass_nm"] >= cfg.denoise_max_nm:
            raise ConvertError(f"--params: bandpass_nm must be below the recipe's denoise_max_nm "
                               f"({cfg.denoise_max_nm:g} nm)")
    if "denoise" in params:
        ch["denoise"] = params["denoise"]
        if not params["denoise"]:            # only the clean variant is left: it must be written
            ch["skip_variants"] = tuple(v for v in cfg.skip_variants if v != "clean")
    return dataclasses.replace(cfg, **ch)


def effective_params(cfg) -> dict:
    """The four tunable values as the recipe (+ overrides) will render them; bandpass_nm is None for a median
    recipe or with denoise off (and a field a config does not have is None)."""
    def num(name):
        v = getattr(cfg, name, None)
        return None if v is None else float(v)
    denoise = bool(getattr(cfg, "denoise", False))
    band = denoise and getattr(cfg, "denoise_method", None) == "bandpass"
    return {"clip_lo_pct": num("contrast_lo_pct"), "clip_hi_pct": num("contrast_hi_pct"),
            "bandpass_nm": num("denoise_min_nm") if band else None, "denoise": denoise}


def record_params(conv_json: Path, params, effective, variant=None) -> bool:
    """Read -> merge -> atomic write of params_effective / params_overrides / variant into a portal conversion's
    convert.json. -> written (False: unreadable or not an object); OSError when the write fails after retries."""
    try:
        meta = json.loads(conv_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(meta, dict):
        return False
    meta["params_effective"] = effective
    meta["params_overrides"] = dict(params or {})
    meta["variant"] = variant
    _write_json(conv_json, meta)
    return True


def record_or_fail(conv_json: Path, params, effective, variant):
    """record_params for a conversion. Overrides that can't be recorded fail the conversion (RuntimeError: a later
    --stitch-only would otherwise stitch with the recipe's settings, not the tiles'); without overrides a miss is
    only a warning. -> warning or None."""
    try:
        ok, why = record_params(conv_json, params, effective, variant), "unreadable or not a JSON object"
    except OSError as ex:
        ok, why = False, f"{type(ex).__name__}: {ex}"
    if ok:
        return None
    if params:
        raise RuntimeError(f"the overrides {json.dumps(params)} could not be recorded in {CONVERT_JSON} ({why}); "
                           f"the conversion is failed so its montages can't be made with other settings")
    return f"{CONVERT_JSON}: params_effective not recorded ({why})"


def _portal_convert_json(out_dir: Path):
    """<dir>/convert.json when out_dir is a portal conversion's <dir>/tiles, else None."""
    p = out_dir.parent / CONVERT_JSON
    return p if out_dir.name == CONV_TILES and p.is_file() else None


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


def tile_ref_problem(side: Path, name) -> str | None:
    """A tile a sidecar names ([Image = <name>]) must be a plain file name beside the sidecar: no drive, no folder,
    no UNC share, not '..'. Checked by its spelling first (a UNC or absolute name is never touched, so no network or
    other folder is reached), then its resolved path must be inside the sidecar's own folder (a link out is refused).
    -> why not, or None."""
    name = "" if name is None else str(name)
    if (not name or name in (".", "..") or any(c in name for c in "\\/:")
            or any(ord(c) < 32 for c in name)):
        return f"tile name {name[:80]!r} is not a plain file name beside it (refused: no paths in a sidecar)"
    try:
        here = Path(side).parent.resolve()
        where = (here / name).resolve()
    except (OSError, RuntimeError, ValueError):
        return f"tile name {name[:80]!r} can't be resolved beside it"
    if os.path.normcase(str(where.parent)) != os.path.normcase(str(here)):
        return f"tile {name[:80]!r} leads outside the sidecar's folder (refused)"
    return None


def dtype_factor(cfg) -> float:
    """The memory estimates are for float32 recipes: x2 for a float64 one (cryoem_core's own default)."""
    try:
        import numpy as np
        return max(1.0, np.dtype(getattr(cfg, "dtype", None) or np.float64).itemsize / 4)
    except Exception:
        return 2.0


def check_montage(m, core):
    """(A/px, estimated peak bytes of the tile export for a float32 recipe, warnings) for one montage, or
    ConvertError. Sets the placement convention."""
    ps = sidecar_a_px(m.idoc_path, core)
    if ps is None:
        raise ConvertError(f"{m.idoc_path.name}: no finite, positive PixelSpacing in the sidecar header; "
                           f"nm/px is never defaulted")
    gp, pieces = core.metadata.parse_metadata(m.idoc_path)
    if not pieces:
        raise ConvertError(f"{m.idoc_path.name}: the sidecar lists no tiles")
    warnings = []
    tiff = any(pc.get("_file") for pc in pieces)
    for pc in pieces:                       # before any tile is looked at: never open a file outside the folder
        if pc.get("_section") == "Image" or "_file" in pc:
            why = tile_ref_problem(m.idoc_path, pc.get("_file"))
            if why:
                raise ConvertError(f"{m.idoc_path.name}: {why}")
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
            log=None, montages_out=None, acquisition_out=None, params=None) -> dict:
    """Raw montages -> <out_dir>/*.png + MANIFEST.csv. Returns {"out_dir", "recipe", "recipe_defaulted",
    "entry_point", "compute", "device", "montages", "a_px": {montage: A/px}, "n_tiles", "tiles": [{"file",
    "nm_per_px", "source_file", "montage", "sample_label", "size"}], "nm_per_px": [distinct], "warnings",
    "stitch": stitch.json | None, "acquisition": written, "params_effective", "params_overrides"}.
    params: the --params overrides (JSON text or dict; validated first, before anything is touched); None = the
    recipe as-is, byte for byte.
    Raises ConvertError on a refusal (out_dir left as it was, .warnings carries what was found so far).
    montages_out / acquisition_out: the stitch pass and acquisition.json, after the tiles are final (module
    docstring); their failures are warnings, never a failed conversion.
    The pipeline's console output goes to `log` (default stderr)."""
    tee = _Tee(log or sys.stderr)
    warnings = []
    try:
        params = parse_params(params)
        return _convert(inputs, Path(out_dir), recipe, sample_key, bool(gpu), processing_repo_dir, tee, warnings,
                        montages_out, acquisition_out, params)
    except ConvertError as e:
        e.warnings = list(dict.fromkeys(warnings + e.warnings + tee.caught))
        raise


def _convert(inputs, out_dir, recipe, sample_key, gpu, processing_repo_dir, tee, warnings, montages_out=None,
             acquisition_out=None, params=None):
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ConvertError(f"output folder is not empty: {out_dir}")
    if not inputs:
        raise ConvertError("no inputs")
    repo = processing_repo(processing_repo_dir)
    edt, core = import_pipeline(repo)
    recipe_name, defaulted, base_cfg = load_recipe(repo, recipe, core, tee)
    apply_params(base_cfg, params)               # an override the recipe can't take is refused before any work
    if sample_key is not None and not Path(sample_key).is_file():
        raise ConvertError(f"sample key not found: {sample_key}")

    with contextlib.redirect_stdout(tee):
        montages, w = find_montages(inputs, core, base_cfg.layout)
    warnings += w
    if not montages:
        raise ConvertError("no montages found in the inputs")
    if montages_out is not None and _overlaps(Path(montages_out), out_dir):
        raise ConvertError("--montages-out must not be the tiles folder, inside it, or hold it")
    found = list(montages)                       # every montage found, coarse ones too (acquisition's session)
    target = float(edt.TARGET_A_PX)
    a_px, peak, keep = {}, {}, []
    for m in montages:
        ps, need, w = check_montage(m, core)
        warnings += w
        a_px[m.basename], peak[m.basename] = ps, int(need * dtype_factor(base_cfg))
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
    work = out_dir / WORK
    cfg = apply_params(run_cfg(base_cfg, work, montages, gpu, f"convert_raw[{recipe_name}]"), params)
    effective = effective_params(cfg)
    variant = tile_variant(cfg)
    conv_json = _portal_convert_json(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        work.mkdir()
        if sample_key is not None:
            shutil.copyfile(sample_key, work / "sample_key.csv")
        args = argparse.Namespace(out=str(work / "native"), grids=None, limit=0, gpu=gpu)
        small = work / (edt.SMALL_DIR % ("%.1f" % target))
        with contextlib.redirect_stdout(tee):
            print(f"convert_raw: {len(montages)} montage(s), recipe {recipe_name}, via {ENTRY_POINT}")
            if params:
                print(f"convert_raw: --params {json.dumps(params)} -> {json.dumps(effective)}")
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
        if conv_json is not None:                # overrides that can't be recorded fail the conversion
            w = record_or_fail(conv_json, params, effective, variant)
            if w:
                warnings.append(w)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        if made_out:
            shutil.rmtree(out_dir, ignore_errors=True)
        else:
            for p in list(out_dir.iterdir()):
                (shutil.rmtree if p.is_dir() else os.remove)(p)
        raise
    shutil.rmtree(work, ignore_errors=True)

    # The tiles are final. Separate passes from here on: a failure is a warning, never a failed conversion.
    stitch = None
    if montages_out is not None:
        stitch, w = stitch_pass(core, cfg, montages, rows, Path(montages_out), out_dir, tee, need=stitch_need(peak),
                                processing=processing_info(repo, recipe_name))
        warnings += w
    acq_written = False
    if acquisition_out is not None:
        acq_written, w = acquisition_pass(core, found, montages, Path(acquisition_out))
        warnings += w

    device = "CPU (NumPy)"
    if gpu:
        device = core.device.describe_device(True)
        if not device.startswith("GPU"):
            warnings.append(f"--gpu: the core found no usable GPU and ran on the CPU ({device}); "
                            f"MANIFEST.csv still says GPU")
    warnings[:] = list(dict.fromkeys(warnings + tee.caught))
    tiles =[{"file": r["output_name"], "nm_per_px": float(r["nm_per_px"]), "source_file": r["source_file"],
              "montage": r["montage"], "sample_label": r["sample_label"], "size": r["output_size"],
              "variant": variant} for r in rows]
    return {"out_dir": str(out_dir), "recipe": recipe_name, "recipe_defaulted": defaulted,
            "entry_point": ENTRY_POINT, "compute": "GPU" if device.startswith("GPU") else "CPU", "device": device,
            "montages": [m.basename for m in montages], "a_px": {m.basename: a_px[m.basename] for m in montages},
            "n_tiles": len(tiles), "tiles": tiles,
            "nm_per_px": sorted({t["nm_per_px"] for t in tiles}), "warnings": warnings,
            "stitch": stitch, "acquisition": acq_written, "processing": processing_info(repo, recipe_name),
            "params_effective": effective, "params_overrides": dict(params or {})}


def run_cfg(base_cfg, work: Path, montages, gpu, name):
    """The recipe as-is; only where it reads from and writes to changes (outputs land in `work`)."""
    return dataclasses.replace(base_cfg, dataset_dir=work, raw_subdir=None, notes_subdir=None,
                               layout=_FixedLayout(montages), use_gpu=gpu, name=name)


# ---------------------------------------------------------------- montages (after the tiles are final)
def _now():
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, doc):
    """Atomic: a reader never sees half a file (os.replace retried: on Windows a reader may hold the target)."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for i in range(6):
        try:
            return os.replace(tmp, path)
        except PermissionError:
            if i == 5:
                raise
            time.sleep(0.1 * 2 ** i)


def _overlaps(a: Path, b: Path) -> bool:
    """a and b are the same folder, or one holds the other."""
    ra, rb = (os.path.normcase(str(Path(x).resolve())) for x in (a, b))
    return ra == rb or ra.startswith(rb + os.sep) or rb.startswith(ra + os.sep)


def stitch_need(peak: dict) -> dict:
    """The tile export's estimates (check_montage x dtype_factor) -> the stitch pass's."""
    return {k: v * STITCH_BYTES_PER_PX // BYTES_PER_MONTAGE_PX for k, v in peak.items()}


def tile_variant(cfg):
    """The variant the tiles are ("denoised" | "clean"): render_ref delivers the last one not in skip_variants, as
    composite_variant. The file names say "denoised" either way (the processing repo's naming). None if unknown."""
    try:
        return composite_variant(cfg)
    except (AttributeError, RuntimeError, TypeError):
        return None


def composite_variant(cfg) -> str:
    """The variant whose composite run_convert writes last (the delivered 3b folder): the last one configured that
    is not in skip_variants. Every variant is still rendered, in order, for its clip levels."""
    kept = [v for v, _m, _p in cfg.output_dirs() if v not in cfg.skip_variants]
    if not kept:
        raise RuntimeError("every variant is in skip_variants: run_convert writes no composite")
    return kept[-1]


def scalebar_drawn(ps, mag, canvas_w, imaging):
    """(nm) of the bar stitch_composite draws on a canvas_w-wide composite: get_scalebar(mag, SCALEBAR_LENGTHS_BIG),
    shortened as draw_scalebar_big does when it would span over 40 % of the width."""
    nm, _label = imaging.get_scalebar(mag, imaging.SCALEBAR_LENGTHS_BIG)
    if nm is None:
        return None
    px_nm = ps * 0.1
    if int(nm / px_nm) > canvas_w * 0.4:
        nm = int(int(canvas_w * 0.4) * px_nm)
        nm = round(nm / 100) * 100 if nm >= 1000 else round(nm / 50) * 50
    return nm


def stitch_montage(cfg, m, mrows, dest_dir: Path) -> dict:
    """One montage's composite, as run_convert makes it (module docstring), saved as
    <dest_dir>/<basename>_stitched.jpg. -> its layout.json entry."""
    from cryoem_core import imaging, pipeline, stitching
    from PIL import Image

    placement = m.placement or cfg.placement
    _g, tiles, params = stitching.load_tiles(m, dtype=cfg.dtype, placement=placement)
    if not tiles:
        raise RuntimeError("no tiles loaded")
    ps, mag = params[0], params[1]
    # the canvas geometry, taken before stitch_composite frees the pixels (assemble_canvas: the loaded tile's size)
    geom = [{"src": pipeline._tile_filename(m, t), "row": int(t["row"]), "col": int(t["col"]), "x": t["x"],
             "y": t["y"], "h": int(t["data"].shape[0]), "w": int(t["data"].shape[1])} for t in tiles]
    target = composite_variant(cfg)
    out = dest_dir / f"{m.basename}_stitched.jpg"
    comp_levels, canvas = None, None
    for variant, var_tiles in pipeline._variant_tiles(cfg, tiles, ps):
        reuse = cfg.denoise_match_contrast and variant != "clean"
        composite, used = stitching.stitch_composite(
            var_tiles, params, placement, prefer_gpu=cfg.use_gpu, free_tiles=cfg.free_tiles,
            lo_pct=cfg.contrast_lo_pct, hi_pct=cfg.contrast_hi_pct,
            levels=comp_levels if reuse else None, with_levels=True, level_bin=pipeline.level_bin(cfg, ps))
        if comp_levels is None:
            comp_levels = used
        if composite is not None and variant == target:
            canvas = composite.size
            imaging.downsample_and_save(composite, out, min(cfg.max_long_edge, MONTAGE_MAX_EDGE), cfg.jpeg_quality)
        composite = None                         # not held while the next variant is stitched
    del tiles
    if canvas is None:
        raise RuntimeError(f"the {target} composite came out empty")
    with Image.open(out) as im:
        width, height = im.size
    cw, ch = canvas
    sx, sy = width / cw, height / ch
    min_x, min_y = min(g["x"] for g in geom), min(g["y"] for g in geom)
    by_src = {r.get("source_file"): r for r in mrows}
    boxes = []
    for g in sorted(geom, key=lambda g: (g["row"], g["col"], g["src"])):
        r = by_src.get(g["src"])
        if r is None:                            # a tile the export did not write (empty): no MANIFEST name
            continue
        ox, oy = stitching.tile_offset(g["x"], g["y"], min_x, min_y, ch, g["h"], placement)
        x0, y0 = max(0, round(ox * sx)), max(0, round(oy * sy))
        x1, y1 = min(width, round((ox + g["w"]) * sx)), min(height, round((oy + g["h"]) * sy))
        boxes.append({"output_name": r["output_name"], "row": g["row"], "col": g["col"], "x": x0, "y": y0,
                      "w": max(0, x1 - x0), "h": max(0, y1 - y0)})
    label = next((r.get("sample_label") for r in mrows if r.get("sample_label")), None) or f"Grid {m.grid}"
    return {"name": m.basename, "file": out.name, "width": width, "height": height,
            "nm_per_px": round(ps * 0.1 * cw / width, 5), "scalebar_nm": scalebar_drawn(ps, mag, cw, imaging),
            "grid": int(m.grid), "sample_label": label, "tiles": boxes}


def _stitch_done(dest: Path) -> bool:
    """dest holds a finished montage set (its stitch.json says done)."""
    try:
        return json.loads((dest / STITCH_JSON).read_text(encoding="utf-8")).get("state") == "done"
    except (OSError, ValueError, AttributeError):
        return False


MONTAGES_IN_USE = "montages in use; previous set kept — try again"
RENAME_WAITS = (0.5, 1, 2, 3, 4)         # a folder rename retried over ~10 s (Windows: a file in it held open)


def _rename_retry(src: Path, dst: Path, waits=RENAME_WAITS) -> bool:
    """os.replace(src, dst), retried on PermissionError (WinError 5: a reader holds a file in it). -> done."""
    for w in (*waits, None):
        try:
            os.replace(src, dst)
            return True
        except PermissionError:
            if w is None:
                return False
            time.sleep(w)
    return False


def stitch_dirs(dest: Path) -> tuple:
    """(work, old) beside dest: .<dest>.stitch_work (built here) and .<dest>_old (the set being replaced)."""
    return dest.parent / f".{dest.name}{STITCH_WORK}", dest.parent / f".{dest.name}_old"


def stitch_pass(core, cfg, montages, rows, dest: Path, tiles_dir: Path, log, need=None, processing=None):
    """Every montage's composite + layout.json + stitch.json into `dest`, built in .<dest>.stitch_work beside it and
    swapped in by rename (dest -> .<dest>_old, work -> dest, the old one removed). Never raises: -> (stitch doc,
    warnings). A montage whose estimated peak memory exceeds the RAM free now is not attempted (a failed montage);
    the others still are. A good set is never destroyed: when this pass has any failure and dest already holds a
    finished set (stitch.json done), dest is left untouched and the failure is only returned (doc kept_previous).
    A swap cut short (dest missing, .<dest>_old there) is renamed back first."""
    warns, t_start = [], time.time()
    work, old = stitch_dirs(dest)
    try:
        if any(_overlaps(x, tiles_dir) for x in (dest, work, old)):
            raise RuntimeError("the montages folder must not be the tiles folder, inside it, or hold it")
        if not dest.exists() and old.exists():
            os.replace(old, dest)                # an earlier swap was cut short: the previous set comes back
        shutil.rmtree(old, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        by_m = {}
        for r in rows:
            by_m.setdefault(r.get("montage"), []).append(r)
        entries, failed = [], []
        with contextlib.redirect_stdout(log):
            print(f"stitch: {len(montages)} montage(s) -> {dest.name}/ (composite as run_convert: "
                  f"{composite_variant(cfg)} variant, clip levels from the clean pass, "
                  f"<= {min(cfg.max_long_edge, MONTAGE_MAX_EDGE)} px, JPEG q{cfg.jpeg_quality}, scalebar)", flush=True)
            if available_ram() is None:
                print("stitch: free RAM unreadable (psutil missing?): the memory guard is off", flush=True)
            for i, m in enumerate(montages, 1):
                t0 = time.time()
                try:
                    free = available_ram()
                    want = (need or {}).get(m.basename)
                    if free is not None and want and want > free:
                        raise MemoryError(f"needs about {want / 2**30:.1f} GB, {free / 2**30:.1f} GB free now")
                    e = stitch_montage(cfg, m, by_m.get(m.basename, []), work)
                    entries.append(e)
                    print(f"stitch [{i}/{len(montages)}] {m.basename}: {e['width']}x{e['height']} px, "
                          f"{len(e['tiles'])} tiles boxed, {time.time() - t0:.1f} s", flush=True)
                except Exception as ex:          # one montage: recorded, the next one still runs
                    failed.append(f"{m.basename}: {type(ex).__name__}: {ex}")
                    print(f"stitch [{i}/{len(montages)}] {m.basename}: FAILED {type(ex).__name__}: {ex}", flush=True)
        doc = {"state": "failed" if failed else "done", "error": "; ".join(failed)[:4000] or None,
               "n": len(entries), "t": _now(), "processing": processing}
        if failed and _stitch_done(dest):
            shutil.rmtree(work, ignore_errors=True)
            doc["kept_previous"] = True          # the finished set in dest stays exactly as it was
        else:
            _write_json(work / LAYOUT_JSON, {"schema": LAYOUT_SCHEMA, "montages": entries})
            _write_json(work / STITCH_JSON, doc)
            if dest.exists() and not _rename_retry(dest, old):
                # a viewer (the gallery) holds a montage open: the previous set stays, nothing half-swapped
                shutil.rmtree(work, ignore_errors=True)
                doc = {**doc, "state": "failed", "kept_previous": True, "error": MONTAGES_IN_USE}
            else:
                if not _rename_retry(work, dest):
                    raise OSError("the new montages could not be moved into place")
                shutil.rmtree(old, ignore_errors=True)
    except Exception as ex:                      # the pass itself; a finished set in dest is never touched
        shutil.rmtree(work, ignore_errors=True)
        doc = {"state": "failed", "error": f"{type(ex).__name__}: {ex}"[:4000], "n": 0, "t": _now(),
               "processing": processing}
        try:
            if not _overlaps(dest, tiles_dir):
                if not dest.exists() and old.exists():
                    os.replace(old, dest)
                if _stitch_done(dest):
                    doc["kept_previous"] = True
                else:                            # not a finished set: replaced by this failure's record
                    shutil.rmtree(dest, ignore_errors=True)
                    dest.mkdir(parents=True, exist_ok=True)
                    _write_json(dest / STITCH_JSON, doc)
        except OSError:
            pass
    if doc.get("error") == MONTAGES_IN_USE:
        warns.append(f"montages: {MONTAGES_IN_USE}; the tiles are not affected")
    elif doc["state"] == "failed":
        kept = "; the previous montages are kept" if doc.get("kept_previous") else ""
        warns.append(f"montages: stitching failed for some or all ({doc['error']}){kept}; "
                     f"the tiles are not affected")
    print(f"stitch: {doc['state']}, {doc['n']} montage(s), {time.time() - t_start:.1f} s", file=log, flush=True)
    return doc, warns


# ---------------------------------------------------------------- acquisition.json
def load_importers():
    """cryoEV's registry/importers.py (the same file_summary / session_acquisition / scope_model the registry uses),
    imported as the package module registry.importers (the repo root goes on sys.path when this runs as a script)."""
    repo = str(Path(__file__).resolve().parent.parent)
    if repo not in sys.path:
        sys.path.append(repo)
    from registry import importers
    return importers


def _median(values, nd=4):
    return round(statistics.median(values), nd) if values else None


def montage_acquisition(imp, core, m) -> tuple:
    """(acquisition.json montage entry, (sidecar name, file_summary)). Scalars: the montage's magnification and
    pixel size (its imaging pieces have one each; the first if not), median defocus (um), median dose (> 0 only;
    0 = uncalibrated), the first piece's time, median exposure (s), placed pieces."""
    gp, pieces = core.metadata.parse_metadata(m.idoc_path)
    s = imp.file_summary(gp, pieces)
    min_mag = getattr(imp, "IMAGING_MIN_MAG", 0)

    def num(v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        return x if math.isfinite(x) else None
    ims = [p for p in pieces if p.get("_section") in ("ZValue", "Image")
           and (num(p.get("Magnification")) or min_mag) >= min_mag]
    exposure = [v for v in (num(p.get("ExposureTime")) for p in ims) if v is not None]
    entry = {"montage": m.basename, "mag": (s.get("mag") or [None])[0], "pixel_A": (s.get("px") or [None])[0],
             "defocus_um": _median(s.get("defocus") or []),
             "dose": _median([d for d in s.get("dose") or [] if d > 0]), "acquired": s.get("first"),
             "exposure_s": _median(exposure), "n_pieces": len(core.stitching.placed_pieces(pieces))}
    return entry, (m.idoc_path.name, s)


def acquisition_pass(core, found, converted, dest: Path) -> tuple:
    """acquisition.json: one entry per converted montage, the session summary over every montage found (coarse
    ones too; session_acquisition sets low-mag images apart), the scope model. Never raises: -> (written,
    warnings)."""
    try:
        imp = load_importers()
        names = {m.basename for m in converted}
        entries, summaries = [], []
        for m in found:
            e, s = montage_acquisition(imp, core, m)
            summaries.append(s)
            if m.basename in names:
                entries.append(e)
        scope = next((s.get("scope") for _n, s in summaries if s.get("scope")), None)
        doc = {"schema": ACQ_SCHEMA, "montages": entries, "session": imp.session_acquisition(summaries),
               "scope_model": scope}
        dest.parent.mkdir(parents=True, exist_ok=True)
        _write_json(dest, doc)
        return True, []
    except Exception as ex:
        return False, [f"acquisition.json not written: {type(ex).__name__}: {ex}"]


# ---------------------------------------------------------------- --stitch-only (backfill)
def stitch_only(conv_dir, *, processing_repo_dir=None, log=None) -> dict:
    """Montages + acquisition.json for a finished portal conversion (module docstring). Its tiles are only read.
    -> {"mode": "stitch", "out_dir", "recipe", "compute", "device", "montages", "stitch", "acquisition",
    "warnings"}; ConvertError when it can't (no MANIFEST, raw inputs gone, ...)."""
    tee = _Tee(log or sys.stderr)
    warnings = []
    try:
        return _stitch_only(Path(conv_dir), processing_repo_dir, tee, warnings)
    except ConvertError as e:
        e.warnings = list(dict.fromkeys(warnings + e.warnings + tee.caught))
        raise


def _stitch_only(conv, processing_repo_dir, tee, warnings):
    try:
        meta = json.loads((conv / CONVERT_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConvertError(f"not a conversion folder (no readable {CONVERT_JSON})") from e
    tiles = conv / CONV_TILES
    if not (tiles / MANIFEST).is_file():
        raise ConvertError(f"no {CONV_TILES}/{MANIFEST}: the conversion has not finished")
    rows = _read_manifest(tiles / MANIFEST)
    inputs = [Path(p) for p in meta.get("inputs") or []]
    if not inputs or not all(p.is_file() for p in inputs):
        raise ConvertError("the raw inputs are gone (removed after the keep time?): nothing to stitch from")
    repo = processing_repo(processing_repo_dir)
    _edt, core = import_pipeline(repo)
    recipe_name, _defaulted, base_cfg = load_recipe(repo, meta.get("recipe") or "default", core, tee)
    with contextlib.redirect_stdout(tee):
        found, w = find_montages(inputs, core, base_cfg.layout)
    warnings += w
    peak = {}
    for m in found:
        _ps, need, _w = check_montage(m, core)                    # placement + the sidecar path checks
        peak[m.basename] = int(need * dtype_factor(base_cfg))
    names = {r.get("montage") for r in rows}
    montages = [m for m in found if m.basename in names]
    lost = sorted(n for n in names - {m.basename for m in montages} if n)
    if lost:
        warnings.append(f"in MANIFEST but not found in the raw inputs: {', '.join(lost)}")
    if not montages:
        raise ConvertError("none of the converted montages is in the raw inputs", warnings)
    gpu = bool(meta.get("gpu"))
    if gpu:
        ok, note = gpu_check()
        if not ok:
            gpu = False
            warnings.append(f"the conversion used the GPU, which is not free now ({note}); stitched on the CPU")
    # the overrides the tiles were made with (convert.json params_overrides): the montages must match them
    overrides = parse_params(meta.get("params_overrides") or None)
    cfg = apply_params(dataclasses.replace(base_cfg, use_gpu=gpu, name=f"convert_raw[{recipe_name}] stitch-only"),
                       overrides)
    info = processing_info(repo, recipe_name)
    try:                                         # what made the tiles (the portal keeps convert_raw's result)
        before = (json.loads((conv / "status.json").read_text(encoding="utf-8")).get("result") or {}).get("processing")
    except (OSError, ValueError, AttributeError):
        before = None
    if isinstance(before, dict):
        diff = [k for k in ("commit", "recipe", "recipe_md5") if before.get(k) and info.get(k) != before.get(k)]
        if diff:
            warnings.append("the processing repo changed since the tiles were made (" + ", ".join(
                f"{k} {before.get(k)} -> {info.get(k)}" for k in diff) + "): the montages may not match them")
    stitch, w = stitch_pass(core, cfg, montages, rows, conv / CONV_MONTAGES, tiles, tee, need=stitch_need(peak),
                            processing=info)
    warnings += w
    acq, w = acquisition_pass(core, found, montages, conv / ACQ_JSON)
    warnings += w
    device = core.device.describe_device(True) if gpu else "CPU (NumPy)"
    warnings[:] = list(dict.fromkeys(warnings + tee.caught))
    return {"mode": "stitch", "out_dir": str(conv / CONV_MONTAGES), "recipe": recipe_name,
            "compute": "GPU" if device.startswith("GPU") else "CPU", "device": device,
            "montages": [m.basename for m in montages], "stitch": stitch, "acquisition": acq, "processing": info,
            "params_effective": effective_params(cfg), "params_overrides": dict(overrides or {}),
            "warnings": warnings}


# ---------------------------------------------------------------- describe (read-only)
def session_pixel(cfg, core) -> tuple:
    """(A/px values, camera binnings) of the recipe's own session, from its sidecars (read-only); empty sets when
    the session folder is not here."""
    a_px, binning = set(), set()
    try:
        raw = Path(cfg.raw_dir)
        if raw.is_dir():
            with contextlib.redirect_stdout(io.StringIO()):
                for m in cfg.layout.find(raw):
                    gp, pieces = core.metadata.parse_metadata(Path(m.idoc_path))
                    try:
                        ps = float(gp.get("PixelSpacing"))
                        if math.isfinite(ps) and ps > 0:
                            a_px.add(round(ps, 4))
                    except (TypeError, ValueError):
                        pass
                    for pc in [gp, *pieces]:
                        try:
                            binning.add(int(float(pc.get("Binning"))))
                            break
                        except (TypeError, ValueError):
                            continue
    except Exception:
        return set(), set()
    return a_px, binning


def describe_cfg(cfg, core, edt, name, defaulted, repo) -> dict:
    """The shape `describe` returns for one (recipe + overrides) config."""
    import numpy as np
    pipeline = sys.modules.get("cryoem_core.pipeline")
    a_px, binning = session_pixel(cfg, core)
    if not a_px and getattr(cfg, "contrast_level_sampling_A", None):
        a_px = {float(cfg.contrast_level_sampling_A)}
    ps_ref = min(a_px) if a_px else None
    text = None
    if pipeline is not None and hasattr(pipeline, "describe_denoise"):
        try:
            text = pipeline.describe_denoise(cfg, ps_ref)
        except Exception:                        # a median recipe needs the sampling to size its kernel
            text = None
    band = cfg.denoise_method == "bandpass"
    on_band = band and bool(cfg.denoise)          # a cutoff is only reported when the filter runs
    sampling = getattr(cfg, "contrast_level_sampling_A", None)
    target = float(edt.TARGET_A_PX)
    info = processing_info(repo, name)
    return {
        "recipe": name,
        "denoise": {"on": bool(cfg.denoise), "method": cfg.denoise_method,
                    "bandpass_nm": float(cfg.denoise_min_nm) if on_band else None,
                    "max_nm": float(cfg.denoise_max_nm) if on_band and cfg.denoise_max_nm else None,
                    "median_target_nm": None if band else float(cfg.denoise_target_nm),
                    "match_contrast": bool(cfg.denoise_match_contrast), "text": text},
        "contrast": {"clip_lo_pct": float(cfg.contrast_lo_pct), "clip_hi_pct": float(cfg.contrast_hi_pct),
                     "sampling": float(sampling) if sampling else None,
                     "sampling_text": (f"clip levels measured at {float(sampling):g} A/px (finer data block-averaged"
                                       f" first)" if sampling else "clip levels measured on the data as acquired"),
                     "levels_from": ("the clean pass (denoise_match_contrast)"
                                     if cfg.denoise and cfg.denoise_match_contrast else "its own pass")},
        "pixel": {"target_A_px": target, "target_nm_px": round(target * 0.1, 5),
                  "session_A_px": sorted(a_px), "binning": sorted(binning),
                  "downsample": {f"{a:g}": round(target / a, 4) for a in sorted(a_px)}, "resample": "INTER_AREA",
                  "dtype": np.dtype(cfg.dtype or np.float64).name},
        "variant": tile_variant(cfg),
        "params": effective_params(cfg),
        "params_allowed": {**{k: [lo, hi] for k, (_f, lo, hi) in PARAM_RANGES.items()}, "denoise": "bool"},
        "source": {"recipe_file": f"datasets/process_{name}.py", "recipe_md5": info["recipe_md5"],
                   "commit": info["commit"], "defaulted": bool(defaulted), "layout": type(cfg.layout).__name__,
                   "checked_by": "check_recipe_source (static, never executed unchecked) + load_recipe",
                   "entry_point": ENTRY_POINT},
    }


def describe_recipe(recipe="default", *, params=None, processing_repo_dir=None, log=None) -> dict:
    """`describe --recipe`: the recipe's real values (+ params, when given). Read-only: the recipe file is checked
    statically, then loaded; no bytecode is written; nothing else is opened but its session's sidecars."""
    tee = _Tee(log or sys.stderr)
    params = parse_params(params)
    repo = processing_repo(processing_repo_dir)
    edt, core = import_pipeline(repo)
    name, defaulted, cfg = load_recipe(repo, recipe, core, tee)
    return describe_cfg(apply_params(cfg, params), core, edt, name, defaulted, repo)


def _read_conv(conv: Path) -> dict:
    try:
        meta = json.loads((conv / CONVERT_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConvertError(f"not a conversion folder (no readable {CONVERT_JSON})") from e
    if not isinstance(meta, dict):
        raise ConvertError(f"not a conversion folder ({CONVERT_JSON} is not an object)")
    return meta


def describe_conv(conv_dir, *, processing_repo_dir=None, log=None) -> dict:
    """`describe --conv`: the settings a conversion used. convert.json params_effective when the conversion
    recorded it ("recorded": true); else the recipe's values now ("recorded": false). The recipe blocks are the
    recipe now + the recorded overrides; a warning says when that no longer gives the recorded values."""
    conv = Path(conv_dir)
    meta = _read_conv(conv)
    recorded = meta.get("params_effective") if isinstance(meta.get("params_effective"), dict) else None
    overrides = meta.get("params_overrides") if isinstance(meta.get("params_overrides"), dict) else {}
    try:
        result = (json.loads((conv / "status.json").read_text(encoding="utf-8")).get("result") or {})
    except (OSError, ValueError, AttributeError):
        result = {}
    result = result if isinstance(result, dict) else {}
    warnings = []
    try:
        d = describe_recipe(meta.get("recipe") or "default", params=overrides or None,
                            processing_repo_dir=processing_repo_dir, log=log)
    except ConvertError as e:
        if recorded is None:
            raise
        d = {"recipe": meta.get("recipe"), "denoise": None, "contrast": None, "pixel": None, "variant": None,
             "params": None, "source": None}
        warnings.append(f"the recipe can't be read now ({e}); only the recorded values are shown")
    if recorded is not None and d.get("params") is not None and d["params"] != recorded:
        warnings.append("the recipe now gives other values than the conversion recorded: params_effective is what "
                        "the tiles were made with")
    before = result.get("processing") if isinstance(result.get("processing"), dict) else {}
    now = d.get("source") or {}
    if before.get("recipe_md5") and now.get("recipe_md5") and before["recipe_md5"] != now["recipe_md5"]:
        warnings.append("the recipe file changed since this conversion")
    if recorded is not None and meta.get("variant"):
        d["variant"] = meta["variant"]              # what the conversion recorded
    d.update({"conv": meta.get("id") or conv.name, "recorded": recorded is not None,
              "params_effective": recorded if recorded is not None else d.get("params"),
              "params_overrides": overrides, "gpu": bool(meta.get("gpu")), "compute": result.get("compute"),
              "processing": before or None, "warnings": warnings})
    return d


# ---------------------------------------------------------------- preview (one tile, before / after)
@contextlib.contextmanager
def _only_tile(edt, mont, source_file):
    """export_detector_tiles' load_tiles narrowed to one tile of `mont` (its MANIFEST source_file). Each tile is
    rendered on its own (its own clip levels, its own denoise), so the narrowed render is the same pixels."""
    from cryoem_core import pipeline
    orig = edt.load_tiles

    def load_one(m, *a, **kw):
        g, tiles, params = orig(m, *a, **kw)
        if getattr(m, "basename", None) == mont.basename:
            tiles = [t for t in tiles if pipeline._tile_filename(m, t) == source_file]
        return g, tiles, params
    edt.load_tiles = load_one
    try:
        yield
    finally:
        edt.load_tiles = orig


def clear_stale_preview_work(out: Path):
    """Remove .preview_work_<pid> folders in `out` left by a preview that was killed (its process is gone); a
    running preview's folder is left alone."""
    for d in out.glob(f"{PREVIEW_WORK}_*"):
        pid = d.name[len(PREVIEW_WORK) + 1:]
        if not d.is_dir() or pid == str(os.getpid()):
            continue
        try:
            import psutil
            alive = pid.isdigit() and psutil.pid_exists(int(pid))
        except ImportError:
            alive = False
        if not alive:
            shutil.rmtree(d, ignore_errors=True)


def _kept(conv: Path, p) -> Path:
    """A recorded input path -> its kept copy in <conv>/input when there (the folder may have moved), else as is."""
    p = Path(p)
    k = conv / CONV_INPUT / p.name
    return k if k.is_file() else p


def render_tile(edt, cfg, mont, source_file, work: Path, sample_key, gpu, log) -> tuple:
    """One tile through the conversion's own entry point (export_dataset, as _convert calls it) into `work`.
    -> (PNG path, its MANIFEST row)."""
    work.mkdir(parents=True)
    if sample_key is not None:
        shutil.copyfile(sample_key, work / "sample_key.csv")
    args = argparse.Namespace(out=str(work / "native"), grids=None, limit=0, gpu=gpu)
    small = work / (edt.SMALL_DIR % ("%.1f" % float(edt.TARGET_A_PX)))
    with contextlib.redirect_stdout(log), _only_tile(edt, mont, source_file):
        try:
            rc = edt.export_dataset(args, cfg)
        except SystemExit as e:
            raise ConvertError(f"pipeline stopped: {e}") from e
    rows = _read_manifest(small / MANIFEST) if (small / MANIFEST).is_file() else []
    hit = [r for r in rows if r.get("source_file") == source_file]
    if rc not in (0, None) or len(hit) != 1 or not (small / hit[0]["output_name"]).is_file():
        raise ConvertError(f"the pipeline did not render {source_file} (returned {rc})")
    return small / hit[0]["output_name"], hit[0]


def preview(conv_dir, tile, out_dir, *, params=None, gpu=False, processing_repo_dir=None, log=None) -> dict:
    """`preview`: before.png (the conversion's own settings: the recipe + convert.json params_overrides) and
    after.png (that base + params; the values are absolute) of one tile of a conversion, from its kept raw input,
    into out_dir only (module docstring). -> {"before", "after", "after_reused", "params_effective",
    "params_conversion", "params_recipe", "params_overrides", "params_conversion_overrides", "tile", "montage",
    "source_file", "recipe", "compute", "device", "warnings", "seconds"}. ConvertError on a refusal (nothing
    written)."""
    tee = _Tee(log or sys.stderr)
    warnings = []
    try:
        params = parse_params(params)
        return _preview(Path(conv_dir), str(tile), Path(out_dir), params, bool(gpu), processing_repo_dir, tee,
                        warnings)
    except ConvertError as e:
        e.warnings = list(dict.fromkeys(warnings + e.warnings + tee.caught))
        raise


def _preview(conv, tile, out, params, gpu, processing_repo_dir, tee, warnings):
    t0 = time.time()
    meta = _read_conv(conv)
    if not tile or any(c in tile for c in "\\/:") or tile in (".", ".."):
        raise ConvertError("--tile: a MANIFEST output_name, not a path")
    if not (conv / CONV_TILES / MANIFEST).is_file():
        raise ConvertError(f"no {CONV_TILES}/{MANIFEST}: the conversion has not finished")
    row = next((r for r in _read_manifest(conv / CONV_TILES / MANIFEST) if r.get("output_name") == tile), None)
    if row is None:
        raise ConvertError(f"no tile {tile[:120]!r} in this conversion's {MANIFEST}")
    if _overlaps(out, conv):
        raise ConvertError("--out must be outside the conversion folder (a preview never writes into it)")
    if out.exists() and not out.is_dir():
        raise ConvertError("--out is a file, not a folder")
    inputs = [_kept(conv, p) for p in meta.get("inputs") or []]
    if not inputs or not all(p.is_file() for p in inputs):
        raise ConvertError("the raw inputs are gone (removed after the keep time?): nothing to preview from")
    repo = processing_repo(processing_repo_dir)
    edt, core = import_pipeline(repo)
    recipe_name, _defaulted, recipe_cfg = load_recipe(repo, meta.get("recipe") or "default", core, tee)
    # the base: the conversion's own settings (recipe + its recorded overrides), i.e. the tile the user sees
    conv_overrides = parse_params(meta.get("params_overrides") or None)
    base_cfg = apply_params(recipe_cfg, conv_overrides)
    apply_params(base_cfg, params)               # refused before any work
    with contextlib.redirect_stdout(tee):
        found, w = find_montages(inputs, core, base_cfg.layout)
    mont = next((m for m in found if m.basename == row.get("montage")), None)
    if mont is None:
        raise ConvertError(f"montage {row.get('montage')!r} is not in the raw inputs", w)
    _ps, need, w = check_montage(mont, core)
    warnings += w
    need = int(need * dtype_factor(base_cfg))
    free = available_ram()
    if free is not None and need > free:
        raise ConvertError(f"{mont.basename}: needs about {need / 2**30:.1f} GB, {free / 2**30:.1f} GB free now")
    if gpu:
        ok, note = gpu_check()
        if not ok:
            raise ConvertError(f"--gpu: {note}")
    sk = meta.get("sample_key")
    sk = _kept(conv, sk) if sk else None
    if sk is not None and not sk.is_file():
        warnings.append("the conversion's sample key is gone")
        sk = None
    used = "GPU" if gpu else "CPU"
    if row.get("compute") and row["compute"] != used:
        warnings.append(f"the tile was rendered on the {row['compute']}, the preview on the {used}: +-1 grey level "
                        f"on a few pixels (gotcha 31)")
    out.mkdir(parents=True, exist_ok=True)
    clear_stale_preview_work(out)
    work = out / f"{PREVIEW_WORK}_{os.getpid()}"
    name = f"convert_raw[{recipe_name}] preview"
    shutil.rmtree(work, ignore_errors=True)
    try:
        work.mkdir()
        cfgs = {"before": run_cfg(base_cfg, work / "before", [mont], gpu, name),
                "after": apply_params(run_cfg(base_cfg, work / "after", [mont], gpu, name), params)}
        # the same settings: after is before (one render)
        reuse = dataclasses.replace(cfgs["after"], dataset_dir=cfgs["before"].dataset_dir,
                                    layout=cfgs["before"].layout) == cfgs["before"]
        with contextlib.redirect_stdout(tee):
            print(f"preview: {tile} ({mont.basename} {row.get('source_file')}), recipe {recipe_name} + the "
                  f"conversion's overrides {json.dumps(conv_overrides or {})}, params {json.dumps(params or {})}"
                  f"{' (= the base: one render)' if reuse else ''}, via {ENTRY_POINT} ({used})", flush=True)
        made = {}
        for label, c in cfgs.items():
            if label == "after" and reuse:
                made[label] = work / "after.png"
                shutil.copyfile(made["before"], made[label])
                continue
            png, _r = render_tile(edt, c, mont, row.get("source_file"), Path(c.dataset_dir), sk, gpu, tee)
            made[label] = png
        for label, png in made.items():
            if not _rename_retry(png, out / f"{label}.png", waits=(0.2, 0.5, 1)):
                raise ConvertError(f"{label}.png is held open by another program; try again")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    device = core.device.describe_device(True) if gpu else "CPU (NumPy)"
    if sk is None:
        warnings.append("no sample key: labels are 'Grid <n>' (the pixels are the same)")
    # the pipeline's lines about its own (deleted) work folder say nothing to the caller
    warnings[:] = list(dict.fromkeys(warnings + [w for w in tee.caught if PREVIEW_WORK not in w]))
    return {"conv": meta.get("id") or conv.name, "tile": tile, "montage": mont.basename,
            "source_file": row.get("source_file"), "before": str(out / "before.png"),
            "after": str(out / "after.png"), "after_reused": reuse,
            "params_effective": effective_params(cfgs["after"]), "params_conversion": effective_params(cfgs["before"]),
            "params_recipe": effective_params(recipe_cfg), "params_overrides": dict(params or {}),
            "params_conversion_overrides": dict(conv_overrides or {}),
            "recipe": recipe_name, "compute": "GPU" if device.startswith("GPU") else "CPU", "device": device,
            "processing": processing_info(repo, recipe_name), "warnings": warnings,
            "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- CLI
def _emit(res, as_json) -> int:
    for w in res.get("warnings") or []:
        print(f"warning: {w}", file=sys.stderr)
    print(json.dumps({"ok": True, **res}) if as_json else json.dumps({"ok": True, **res}, indent=1))
    return 0


def _refused(e) -> int:
    print(json.dumps({"ok": False, "kind": "refused", "error": str(e), "warnings": getattr(e, "warnings", [])}))
    print(f"REFUSED: {e}", file=sys.stderr)
    return 2


def _failed(e) -> int:
    traceback.print_exc(file=sys.stderr)
    print(json.dumps({"ok": False, "kind": "failed", "error": f"{type(e).__name__}: {e}", "warnings": []}))
    return 1


class _JsonArgs(argparse.ArgumentParser):
    """A usage error is a refusal in JSON on stdout (exit 2), like every other refusal."""

    def error(self, message):
        self.print_usage(sys.stderr)
        raise SystemExit(_refused(ConvertError(f"{self.prog}: {message}")))


def main_sub(argv) -> int:
    """`describe ...` / `preview ...` (module docstring)."""
    cmd = argv[0]
    ap = _JsonArgs(prog=f"convert_raw {cmd}")
    ap.add_argument("--processing-repo", type=Path, help=f"overrides ${REPO_ENV}")
    ap.add_argument("--json", action="store_true", help="one JSON object on one line (default: indented)")
    if cmd == "describe":
        g = ap.add_mutually_exclusive_group(required=True)
        g.add_argument("--recipe", help="default or a session name")
        g.add_argument("--conv", type=Path, help="a conversion folder raw_tiles/<cid>")
    else:
        ap.add_argument("--conv", type=Path, required=True, help="a conversion folder raw_tiles/<cid>")
        ap.add_argument("--tile", required=True, help="the tile's MANIFEST output_name")
        ap.add_argument("--params", help="JSON overrides for after.png (keys: " + ", ".join(PARAM_KEYS) + ")")
        ap.add_argument("--out", type=Path, required=True, help="where before.png / after.png go (outside --conv)")
        ap.add_argument("--gpu", action="store_true", help="render on the GPU (default CPU)")
    a = ap.parse_args(argv[1:])
    try:
        if cmd == "describe" and a.recipe is not None:
            res = describe_recipe(a.recipe, processing_repo_dir=a.processing_repo)
        elif cmd == "describe":
            res = describe_conv(a.conv, processing_repo_dir=a.processing_repo)
        else:
            res = preview(a.conv, a.tile, a.out, params=a.params, gpu=a.gpu, processing_repo_dir=a.processing_repo)
    except ConvertError as e:
        return _refused(e)
    except Exception as e:
        return _failed(e)
    return _emit({"mode": cmd, **res}, a.json)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("describe", "preview"):
        return main_sub(argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inputs", nargs="+", type=Path,
                    help=".mrc / .mrc.mdoc / .idoc / .tif files, or folders of them")
    ap.add_argument("--out", type=Path, help="new or empty folder for tiles + MANIFEST.csv")
    ap.add_argument("--recipe", default="default", help="default (= 20261012) or a session name with a "
                                                         "datasets/process_<name>.py in the processing repo")
    ap.add_argument("--sample-key", type=Path, help="the session's sample_key.csv (#,Label)")
    ap.add_argument("--gpu", action="store_true", help="GPU denoise/percentiles (+-1 grey level vs CPU)")
    ap.add_argument("--processing-repo", type=Path, help=f"overrides ${REPO_ENV}")
    ap.add_argument("--montages-out", type=Path, help="also stitch the montages into this folder (not in --out)")
    ap.add_argument("--acquisition-out", type=Path, help="also write acquisition.json here")
    ap.add_argument("--stitch-only", type=Path, metavar="CONV_DIR",
                    help="backfill a finished portal conversion: <dir>/montages + <dir>/acquisition.json only")
    ap.add_argument("--params", help="JSON overrides on the recipe (keys: " + ", ".join(PARAM_KEYS) + "); "
                                     "absent = the recipe as-is")
    a = ap.parse_args(argv)
    if a.stitch_only is None and (not a.inputs or a.out is None):
        ap.error("--in and --out are required (or --stitch-only <conversion folder>)")
    try:
        if a.stitch_only is not None:
            if a.params is not None:
                raise ConvertError("--stitch-only takes the conversion's own overrides (convert.json "
                                   "params_overrides); --params is refused")
            res = stitch_only(a.stitch_only, processing_repo_dir=a.processing_repo)
        else:
            res = convert(a.inputs, a.out, recipe=a.recipe, sample_key=a.sample_key, gpu=a.gpu,
                          processing_repo_dir=a.processing_repo, montages_out=a.montages_out,
                          acquisition_out=a.acquisition_out, params=a.params)
    except ConvertError as e:
        return _refused(e)
    except Exception as e:
        return _failed(e)
    for w in res["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    print(json.dumps({"ok": True, **res}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
