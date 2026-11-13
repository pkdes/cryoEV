#!/usr/bin/env python3
"""
Package a prepared YOLO dataset as a training job for the shared GPU desktop.

Writes, in <out>:
  ds_<sha12>.zip (+ .md5)          the dataset pack: images/, labels/, MANIFEST.json. Reused when a pack with the
                                   same content hash is already there (seed repeats of the same data).
  <dataset>__job_<timestamp>.zip   the thin job zip: <dataset>/dataset.yaml, JOB.json, README.md, weights/ if packed
  <same>_README.md, <zip>.md5      the exact commands for whoever runs it by hand; md5 of the zip
JOB.json carries `bundle_id` (= the zip stem) and the pack reference (`dataset_sha256`, `dataset_pack`,
`dataset_pack_md5`); jobs/sync.py push uploads a pack only once per queue (docs/JOBS.md, "Dataset cache").
--single-zip writes the old single zip instead (images and labels inside <dataset>/, no pack).
--eval --weights <pt> --dataset <registry dataset id | folder> [--model NAME]: an eval job instead (Ultralytics val on
the test split, imgsz 1440, overlap_mask=False; docs/JOBS.md, "Eval jobs"): the same dataset pack, and a thin zip
<dataset>__eval_<timestamp>.zip with dataset.yaml, JOB.json (job_type eval) and the weights unless tracked in the repo.

    python training/make_job_bundle.py --dataset-dir <folder with dataset.yaml, images/, labels/> \
        [--out <dir>] [--dataset-name NAME] [--single-zip] -- --single-run --model-size m --batch 8 --rect 0 ...

From Python: make_bundle(dataset_dir, exp_args, out, dataset_name=None, single_zip=False, allow_dirty=False) -> dict
writes the same files (main() only parses the arguments and prints; jobs/sync.py queue_train calls it).

Everything after `--` goes to training/run_experiments.py unchanged, except --dataset (set by the
bundle) and --weights (a file outside the repo's models/ is packed into the zip and re-pointed).

dataset.yaml: prepare_all_layer_singleclass.py writes `path: <absolute local folder>`, which is
wrong on any other PC. The bundle comments `path:` out and makes train/val/test relative.
Ultralytics then uses the yaml's own folder as the root (data/utils.py check_det_dataset: a
relative `path:` would resolve against the CWD instead, so it is removed, not relativized).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from data_utils import config  # noqa: E402
from jobs import dataset_cache  # noqa: E402

SPLIT_KEYS = ("train", "val", "test")
STORED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pt", ".zip"}   # already compressed


def run_experiments_flags() -> set:
    src = (REPO / "training" / "run_experiments.py").read_text(encoding="utf-8")
    return set(re.findall(r'add_argument\(\s*"(--[\w-]+)"', src))


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    """Full sha (the desktop resolves it exactly; a short one can become ambiguous)."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


CODE_DIRS = ("training", "jobs", "data_utils")   # the code a bundle's run uses


def uncommitted_code() -> list:
    """Modified tracked files under CODE_DIRS: the bundle's commit would not be the code that was tested."""
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", *CODE_DIRS], cwd=REPO,
                           capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line[3:] for line in r.stdout.splitlines() if line.strip()]


def tracked(path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(REPO)
    except ValueError:
        return False
    r = subprocess.run(["git", "ls-files", "--error-unmatch", rel.as_posix()], cwd=REPO,
                       capture_output=True, text=True)
    return r.returncode == 0


def relative_split(value: str, root: Path, old_root: Path | None) -> str:
    """A split entry as a path relative to the dataset folder."""
    p = Path(value)
    if not p.is_absolute():
        return p.as_posix()
    for base in [b for b in (old_root, root) if b is not None]:
        try:
            return p.relative_to(base).as_posix()
        except ValueError:
            pass
    # Absolute and from another PC: keep the part from images/ on (the YOLO layout).
    parts = p.parts
    if "images" in parts:
        return Path(*parts[parts.index("images"):]).as_posix()
    raise SystemExit(f"dataset.yaml split {value!r} is outside the dataset folder")


def rewrite_dataset_yaml(text: str, root: Path) -> tuple:
    """(new text, list of changes). Keeps comments and names; edits path/train/val/test lines."""
    data = yaml.safe_load(text)
    old_root = Path(data["path"]) if data.get("path") else None
    out, changes = [], []
    for line in text.splitlines():
        m = re.match(r"^(path|train|val|test)\s*:\s*(.*?)\s*$", line)
        if not m or not m.group(2):
            out.append(line)
            continue
        key, val = m.group(1), m.group(2).strip("'\"")
        if key == "path":
            out.append(f"# path: {val}   # removed by make_job_bundle: Ultralytics uses this file's folder")
            changes.append(f"path: {val} -> (removed)")
        else:
            rel = relative_split(val, root, old_root)
            out.append(f"{key}: {rel}")
            if rel != val:
                changes.append(f"{key}: {val} -> {rel}")
    return "\n".join(out) + "\n", changes


def resolve_like_ultralytics(yaml_path: Path) -> dict:
    """{split: absolute path} as Ultralytics resolves them when `path:` is absent."""
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(data["path"]) if data.get("path") else yaml_path.parent
    return {k: (root / data[k]).resolve() for k in SPLIT_KEYS if data.get(k)}


def split_counts(root: Path) -> dict:
    out = {}
    for s in SPLIT_KEYS:
        imgs = root / "images" / s
        if imgs.is_dir():
            labels = root / "labels" / s
            out[s] = {"images": sum(1 for p in imgs.iterdir() if p.is_file()),
                      "labels": sum(1 for _ in labels.glob("*.txt")) if labels.is_dir() else 0}
    return out


def ultralytics_version() -> str:
    try:
        from importlib.metadata import version
        return version("ultralytics")
    except Exception:   # not installed on the bundling PC
        return "unknown"


def readme(name, zip_name, exp_args, changes, counts, commit, weights_note, created,
           ultra="unknown", pack=None) -> str:
    args = " ".join(f'"{a}"' if " " in a else a for a in exp_args)
    rows = "\n".join(f"| {s} | {c['images']} | {c['labels']} |" for s, c in counts.items())
    chg = "\n".join(f"  - `{c}`" for c in changes) or "  - none needed"
    unzip_pack = "" if not pack else f"""
   This job's images and labels are in the dataset pack `{pack}` (next to this zip). Unzip it into the
   dataset folder the job zip just created:
   ```powershell
   Expand-Archive "{pack}" -DestinationPath "<parent>\\CryoAI\\training outputs\\{name}"
   Remove-Item "<parent>\\CryoAI\\training outputs\\{name}\\MANIFEST.json"   # the pack's md5 list, not training data
   ```"""
    return f"""# Training job: `{name}`

Built {created} from cryoEV commit `{commit}` by `training/make_job_bundle.py`.
Nothing here runs by itself; please run the steps below by hand.

| Split | Images | Label files |
|---|---:|---:|
{rows}

## On the GPU desktop (PowerShell)
The repo is `<parent>\\cryoEV`, next to the data folder `<parent>\\CryoAI`.

1. Unzip into the training data folder (creates `CryoAI\\training outputs\\{name}\\`):
   ```powershell
   Expand-Archive "{zip_name}" -DestinationPath "<parent>\\CryoAI\\training outputs"
   ```
   If that folder already exists, stop and ask: don't overwrite another dataset.{unzip_pack}
2. Update the repo to commit `{commit}` or later. Check the remote name first (`git remote -v`;
   on the desktop it is usually `CarneyMLdesktop`):
   ```powershell
   cd "<parent>\\cryoEV"
   git fetch <remote>; git merge --ff-only <remote>/main
   git merge-base --is-ancestor {commit} HEAD; if ($?) {{ "ok" }} else {{ "commit {commit} missing" }}
   ```
   If it says missing, the commit hasn't been pushed yet: stop and ask.
3. Check the Ultralytics version (bundled on {ultra}):
   ```powershell
   python -c "import ultralytics; print(ultralytics.__version__)"
   ```
   It must be 8.4.x. The rewritten `dataset.yaml` has no `path:` line and relies on 8.4's fallback
   to the yaml's own folder; another version may look for the images elsewhere. If it isn't 8.4.x, stop and ask.
4. Train in the background, with a log you can follow:
   ```powershell
   cd "<parent>\\cryoEV"
   Start-Process python -ArgumentList '-u training/run_experiments.py --dataset "{name}" {args}' `
       -RedirectStandardOutput "..\\CryoAI\\training outputs\\{name}\\train.log" `
       -RedirectStandardError "..\\CryoAI\\training outputs\\{name}\\train.err.log" -NoNewWindow
   Get-Content "..\\CryoAI\\training outputs\\{name}\\train.log" -Wait -Tail 20
   ```
   Don't start a second training or evaluation job on the GPU while this one runs (out-of-memory
   risk; see the cryoEV README gotchas).{weights_note}

## Send back
- The run folder `CryoAI\\training outputs\\{name}\\runs\\<run name>\\`, zipped. The run name is
  in `CryoAI\\LATEST_RUN.txt` (`training_dir:`).
- `train.log` and `train.err.log` from `CryoAI\\training outputs\\{name}\\`.
- The new last row of `CryoAI\\EXPERIMENT_LOG.csv`.

## What the bundle changed
`dataset.yaml` was rewritten so it works from any folder:
{chg}
With no `path:` line, Ultralytics uses the folder that holds `dataset.yaml` as the dataset root.
Images and labels are unchanged (md5 list in `JOB.json`).
"""


def make_bundle(dataset_dir: Path, exp_args: list, out: Path | None, dataset_name: str | None = None,
                single_zip: bool = False, allow_dirty: bool = False) -> dict:
    """Write one bundle: the CLI's work without the printing (main() is a thin wrapper around this).
    exp_args: the run_experiments.py arguments (what follows `--` on the command line); a relative --weights path is
    resolved against the current folder, as on the CLI. out None = <CryoAI-laptop>/job_bundles.
    Refusals raise SystemExit, as the CLI always has (the checks are the CLI's, in the same order).
    -> {zip, md5 (of the zip), md5_path, readme, pack (None with single_zip), pack_reused, pack_bytes,
        dataset_sha256 (None with single_zip), job_json (the JOB.json dict), name, exp (as written), changes, n_files}."""
    dirty = [] if allow_dirty else uncommitted_code()
    if dirty:
        raise SystemExit(f"uncommitted changes in {', '.join(dirty[:5])}: commit (and push, for the desktop) first, "
                         f"so the bundle's commit is the code that runs")
    if out is None:
        try:
            out = config.laptop_root() / "job_bundles"
        except config.ConfigError as e:
            raise SystemExit(f"{e} (or pass --out)")
    out = Path(out)

    root = Path(dataset_dir).resolve()
    for need in ("dataset.yaml", "images", "labels"):
        if not (root / need).exists():
            raise SystemExit(f"{root} has no {need}")
    name = dataset_name or root.name
    exp = list(exp_args)
    # --flag=value -> --flag value, so --weights=PATH is found (and packed) like --weights PATH.
    exp = [y for x in exp for y in (x.split("=", 1) if x.startswith("--") and "=" in x else [x])]
    known = run_experiments_flags()
    flags = [x.split("=")[0] for x in exp if x.startswith("--")]
    bad = [f for f in flags if f not in known]
    if bad:
        raise SystemExit(f"run_experiments.py has no {bad}; it takes {sorted(known)}")
    if "--dataset" in flags:
        raise SystemExit("leave --dataset out: the bundle sets it to the dataset name")

    created = time.strftime("%Y-%m-%d %H:%M")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / f"{name}__job_{stamp}.zip"
    text, changes = rewrite_dataset_yaml((root / "dataset.yaml").read_text(encoding="utf-8"), root)

    extra, weights_note = {}, ""
    if "--weights" in exp:
        i = exp.index("--weights") + 1
        if i >= len(exp) or exp[i].startswith("--"):
            raise SystemExit("--weights needs a file path")
        w = Path(exp[i])
        w_abs = w if w.is_absolute() else (Path.cwd() / w)
        if w_abs.is_dir():
            raise SystemExit(f"--weights {w} is a folder; pass the .pt file")
        if tracked(w_abs):   # in the repo: the desktop's own copy is used
            exp[i] = w_abs.resolve().relative_to(REPO).as_posix()
        elif w_abs.is_file():
            extra[f"{name}/weights/{w.name}"] = w_abs
            exp[i] = f"../CryoAI/training outputs/{name}/weights/{w.name}"
            weights_note = (f"\n\n   `--weights` points at `weights\\{w.name}` inside this bundle "
                            f"(packed from `{w_abs}`).")
        else:
            raise SystemExit(f"--weights {w} not found")

    files = sorted(p for d in ("images", "labels") for p in (root / d).rglob("*") if p.is_file())
    md5, pack_ref, pack, reused = {}, {}, None, False
    if not single_zip:                                  # the dataset pack first: the job zip names it
        pack, sha, pack_md5, md5, reused = dataset_cache.write_pack(out, root, files)
        pack_ref = {"dataset_sha256": sha, "dataset_pack": pack.name, "dataset_pack_md5": pack_md5,
                    "dataset_pack_bytes": pack.stat().st_size}
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(f"{name}/dataset.yaml", text, compress_type=zipfile.ZIP_DEFLATED)
        for p in files if single_zip else ():
            rel = p.relative_to(root).as_posix()
            data = p.read_bytes()
            md5[rel] = hashlib.md5(data).hexdigest()
            z.writestr(zipfile.ZipInfo.from_file(p, f"{name}/{rel}"), data,
                       compress_type=zipfile.ZIP_STORED if p.suffix.lower() in STORED else zipfile.ZIP_DEFLATED)
        for arc, p in extra.items():
            z.write(p, arc, compress_type=zipfile.ZIP_STORED)
            md5[arc.split("/", 1)[1]] = hashlib.md5(p.read_bytes()).hexdigest()
        counts = split_counts(root)
        commit = git_commit()
        ultra = ultralytics_version()
        body = readme(name, zip_path.name, exp, changes, counts, commit, weights_note, created, ultra,
                      pack.name if pack else None)
        job = {"bundle_id": zip_path.stem,
               "dataset_name": name, "source_dir": str(root), "created": created, "cryoEV_commit": commit,
               "run_experiments_args": ["--dataset", name] + exp, "dataset_yaml_changes": changes,
               "ultralytics_bundled_on": ultra, "splits": counts, "md5": md5, **pack_ref}
        z.writestr(f"{name}/JOB.json", json.dumps(job, indent=1), compress_type=zipfile.ZIP_DEFLATED)
        z.writestr(f"{name}/README.md", body, compress_type=zipfile.ZIP_DEFLATED)
    readme_path = zip_path.with_name(zip_path.stem + "_README.md")
    readme_path.write_text(body, encoding="utf-8")
    zip_md5 = md5_file(zip_path)
    md5_path = zip_path.with_name(zip_path.name + ".md5")          # sidecar: jobs/sync.py and run_job.py check it
    md5_path.write_text(f"{zip_md5}  {zip_path.name}\n", encoding="utf-8")
    return {"zip": zip_path, "md5": zip_md5, "md5_path": md5_path, "readme": readme_path, "pack": pack,
            "pack_reused": reused, "pack_bytes": pack_ref.get("dataset_pack_bytes"),
            "dataset_sha256": pack_ref.get("dataset_sha256"), "job_json": job, "name": name, "exp": exp,
            "changes": changes, "n_files": len(files)}


# ---------------------------------------------------------------- eval bundles (docs/JOBS.md, "Eval jobs")
EVAL_SPLIT, EVAL_IMGSZ, EVAL_BATCH = "test", 1440, 8   # the settled scoring (models.yaml test_mask_map50_95)
MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def resolve_dataset(ref, registry_root=None) -> tuple:
    """--dataset: a folder holding dataset.yaml, or a registry dataset id (its dataset_dir, else
    <CryoAI>/training outputs/<id>). -> (folder, name: the id, else the folder's name). Raises SystemExit."""
    p = Path(str(ref))
    if (p / "dataset.yaml").is_file():
        return p.resolve(), p.resolve().name
    rec = {}
    try:
        from registry import store as registry
        rec = registry.current("dataset", Path(registry_root) if registry_root else config.registry_root()).get(
            str(ref)) or {}
    except Exception as e:                              # no registry here: the CryoAI folder below still works
        print(f"[bundle] registry not read ({type(e).__name__}: {e})", file=sys.stderr)
    ddir = rec.get("dataset_dir")
    if not ddir:
        try:
            ddir = config.cryoai_root() / "training outputs" / str(ref)
        except config.ConfigError as e:
            raise SystemExit(f"dataset {ref!r}: not a folder with dataset.yaml, and {e}")
    if not (Path(ddir) / "dataset.yaml").is_file():
        raise SystemExit(f"dataset {ref!r}: {ddir} has no dataset.yaml")
    return Path(ddir).resolve(), str(ref)


def default_model_name(weights: Path) -> str:
    """models/<id>/best.pt -> <id>; runs/<run>/weights/best.pt -> <run>; else the file's stem."""
    w = Path(weights)
    if w.parent.name == "weights" and w.parent.parent.name:
        return w.parent.parent.name
    if w.parent.parent.name == "models":
        return w.parent.name
    return w.stem


def make_eval_bundle(dataset_dir: Path, weights: Path, out: Path | None, dataset_name: str | None = None,
                     model: str | None = None, imgsz: int = EVAL_IMGSZ, batch: int = EVAL_BATCH,
                     allow_dirty: bool = False) -> dict:
    """An eval job (jobs/run_job.py: validate_yolo_model on the test split, imgsz 1440, overlap_mask=False): the
    same labelled dataset pack as a training bundle of this folder (same content hash: reused, here and on the
    share), and a thin zip <dataset>/dataset.yaml (rewritten as for training), JOB.json (job_type eval, model,
    weights, weights_in, weights_md5, split, imgsz, batch, n_images), README.md, and weights/<file> unless the
    weights are tracked in this repo (then the desktop's own copy at the same commit is used, md5-checked).
    Refusals raise SystemExit: uncommitted code, no dataset.yaml / images / labels, no test split or no test
    images, weights missing or not a .pt file, a bad model name.
    -> as make_bundle, plus model, test_images (file names of images/test)."""
    dirty = [] if allow_dirty else uncommitted_code()
    if dirty:
        raise SystemExit(f"uncommitted changes in {', '.join(dirty[:5])}: commit (and push, for the desktop) first, "
                         f"so the bundle's commit is the code that runs")
    if out is None:
        try:
            out = config.laptop_root() / "job_bundles"
        except config.ConfigError as e:
            raise SystemExit(f"{e} (or pass --out)")
    out = Path(out)
    root = Path(dataset_dir).resolve()
    for need in ("dataset.yaml", "images", "labels"):
        if not (root / need).exists():
            raise SystemExit(f"{root} has no {need}")
    name = dataset_name or root.name
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", name):
        raise SystemExit(f"dataset name {name!r}: letters, digits, _ . - only")
    w = Path(weights)
    w_abs = (w if w.is_absolute() else Path.cwd() / w).resolve()
    if not w_abs.is_file() or w_abs.suffix.lower() != ".pt":
        raise SystemExit(f"--weights {w}: not a .pt file")
    model = model or default_model_name(w_abs)
    if not MODEL_NAME.match(model):
        raise SystemExit(f"--model {model!r}: letters, digits, _ . - only (it names testmap_<model>.json)")
    text, changes = rewrite_dataset_yaml((root / "dataset.yaml").read_text(encoding="utf-8"), root)
    if not (yaml.safe_load(text) or {}).get(EVAL_SPLIT):
        raise SystemExit(f"{root / 'dataset.yaml'} has no {EVAL_SPLIT}: split (eval jobs score the test split)")
    tests = sorted(p.name for p in (root / "images" / EVAL_SPLIT).iterdir() if p.is_file()) \
        if (root / "images" / EVAL_SPLIT).is_dir() else []
    if not tests:
        raise SystemExit(f"{root} has no images/{EVAL_SPLIT}/ images")

    created = time.strftime("%Y-%m-%d %H:%M")
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / f"{name}__eval_{time.strftime('%Y%m%d-%H%M%S')}.zip"
    files = sorted(p for d in ("images", "labels") for p in (root / d).rglob("*") if p.is_file())
    pack, sha, pack_md5, md5, reused = dataset_cache.write_pack(out, root, files)   # = the training bundle's pack
    pack_ref = {"dataset_sha256": sha, "dataset_pack": pack.name, "dataset_pack_md5": pack_md5,
                "dataset_pack_bytes": pack.stat().st_size}
    w_md5 = md5_file(w_abs)
    if tracked(w_abs):                                  # in the repo: the desktop's copy at this commit
        w_rel, w_in, packed = w_abs.relative_to(REPO).as_posix(), "repo", None
    else:
        w_rel, w_in, packed = f"weights/{w_abs.name}", "bundle", w_abs
    commit, ultra, counts = git_commit(), ultralytics_version(), split_counts(root)
    job = {"bundle_id": zip_path.stem, "job_type": "eval", "dataset_name": name, "source_dir": str(root),
           "created": created, "cryoEV_commit": commit, "model": model, "weights": w_rel, "weights_in": w_in,
           "weights_md5": w_md5, "weights_source": str(w_abs), "split": EVAL_SPLIT, "imgsz": int(imgsz),
           "batch": int(batch), "overlap_mask": False, "n_images": len(tests), "run_experiments_args": [],
           "dataset_yaml_changes": changes, "ultralytics_bundled_on": ultra, "splits": counts, "md5": md5,
           **pack_ref}
    body = (f"# Eval job: `{model}` on `{name}` ({EVAL_SPLIT}, {len(tests)} images)\n\n"
            f"Built {created} from cryoEV commit `{commit}` by `training/make_job_bundle.py --eval`.\n"
            f"Run by `jobs/run_job.py` on the queue (docs/JOBS.md, \"Eval jobs\"): Ultralytics val, split "
            f"{EVAL_SPLIT}, imgsz {imgsz}, overlap_mask=False, into `metrics.json`.\n\n"
            f"Weights: `{w_rel}` ({'packed in this zip' if packed else 'the repo copy at this commit'}; md5 "
            f"`{w_md5}`). Images and labels: the dataset pack `{pack.name}` (the same pack as a training bundle of "
            f"this dataset).\n")
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(f"{name}/dataset.yaml", text, compress_type=zipfile.ZIP_DEFLATED)
        if packed:
            z.write(packed, f"{name}/{w_rel}", compress_type=zipfile.ZIP_STORED)
        z.writestr(f"{name}/JOB.json", json.dumps(job, indent=1), compress_type=zipfile.ZIP_DEFLATED)
        z.writestr(f"{name}/README.md", body, compress_type=zipfile.ZIP_DEFLATED)
    readme_path = zip_path.with_name(zip_path.stem + "_README.md")
    readme_path.write_text(body, encoding="utf-8")
    zip_md5 = md5_file(zip_path)
    md5_path = zip_path.with_name(zip_path.name + ".md5")
    md5_path.write_text(f"{zip_md5}  {zip_path.name}\n", encoding="utf-8")
    return {"zip": zip_path, "md5": zip_md5, "md5_path": md5_path, "readme": readme_path, "pack": pack,
            "pack_reused": reused, "pack_bytes": pack_ref["dataset_pack_bytes"], "dataset_sha256": sha,
            "job_json": job, "name": name, "exp": [], "changes": changes, "n_files": len(files), "model": model,
            "test_images": tests}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path,
                    help="prepared YOLO dataset: dataset.yaml, images/<split>/, labels/<split>/")
    ap.add_argument("--eval", action="store_true", help="an eval job instead (needs --weights and --dataset): "
                    "Ultralytics val on the test split, imgsz 1440, overlap_mask=False (docs/JOBS.md, Eval jobs)")
    ap.add_argument("--weights", type=Path, help="--eval: the model's .pt file")
    ap.add_argument("--dataset", help="--eval: a registry dataset id or a dataset folder (or use --dataset-dir)")
    ap.add_argument("--model", help="--eval: the model name in metrics.json (default: models/<id>/ or the run "
                    "folder's name)")
    ap.add_argument("--imgsz", type=int, help=f"--eval: default {EVAL_IMGSZ}")
    ap.add_argument("--dataset-name", help="folder name on the desktop (default: --dataset-dir's name)")
    ap.add_argument("--out", type=Path, help="default <CryoAI-laptop>/job_bundles")
    ap.add_argument("--single-zip", action="store_true",
                    help="one zip with the images and labels inside (the old format; no dataset pack)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help=f"bundle even with uncommitted changes under {', '.join(CODE_DIRS)} (tests only)")
    ap.add_argument("exp_args", nargs=argparse.REMAINDER,
                    help="after --: arguments for training/run_experiments.py")
    a = ap.parse_args()
    if a.eval:
        if not a.weights or not (a.dataset or a.dataset_dir):
            ap.error("--eval needs --weights and --dataset (or --dataset-dir)")
        if a.exp_args[1:] if a.exp_args[:1] == ["--"] else a.exp_args:
            ap.error("--eval takes no run_experiments arguments after --")
        ddir, dname = (a.dataset_dir.resolve(), a.dataset_name) if a.dataset_dir else resolve_dataset(a.dataset)
        b = make_eval_bundle(ddir, a.weights, a.out, a.dataset_name or dname, a.model, a.imgsz or EVAL_IMGSZ,
                             allow_dirty=a.allow_dirty)
        print(f"{b['pack']}  ({b['pack_bytes'] / 1e6:.1f} MB, {b['n_files']} files, "
              f"{'reused: same content hash' if b['pack_reused'] else 'new'}; dataset_sha256 {b['dataset_sha256'][:12]})")
        print(f"{b['zip']}  (eval: {b['model']} on {b['name']} test, {len(b['test_images'])} images; weights "
              f"{b['job_json']['weights_in']})")
        print(f"{b['md5_path']}  (md5 {b['md5']})")
        print("  queue it: python jobs/sync.py push-eval --bundle <the zip> --share <queue> "
              "(needs CRYOEV_EVAL_JOBS=1 here and on the desktop)")
        return
    if not a.dataset_dir:
        ap.error("--dataset-dir is required (or --eval with --weights and --dataset)")
    if a.weights or a.dataset or a.model or a.imgsz:   # never silently dropped from a training bundle
        ap.error("--weights / --dataset / --model / --imgsz before -- are --eval options; for training put "
                 "run_experiments arguments after --")
    b = make_bundle(a.dataset_dir, a.exp_args[1:] if a.exp_args[:1] == ["--"] else a.exp_args, a.out,
                    a.dataset_name, a.single_zip, a.allow_dirty)
    pack, zip_path, name, exp = b["pack"], b["zip"], b["name"], b["exp"]
    if pack:
        print(f"{pack}  ({pack.stat().st_size / 1e6:.1f} MB, {b['n_files']} files, "
              f"{'reused: same content hash' if b['pack_reused'] else 'new'}; "
              f"dataset_sha256 {b['dataset_sha256'][:12]})")
        print(f"{zip_path}  ({zip_path.stat().st_size / 1e6:.2f} MB, thin: refers to {pack.name})")
    else:
        print(f"{zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB, {b['n_files']} files)")
    print(f"{b['readme']}")
    print(f"{b['md5_path']}  (md5 {b['md5']})")
    for c in b["changes"]:
        print(f"  dataset.yaml {c}")
    print("  run on desktop: python training/run_experiments.py --dataset "
          f"\"{name}\" " + " ".join(f'"{x}"' if " " in x else x for x in exp))


if __name__ == "__main__":
    main()
