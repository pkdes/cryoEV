#!/usr/bin/env python3
"""
Package a prepared YOLO dataset as a training job for the shared GPU desktop.

Writes <out>/<dataset>__job_<timestamp>.zip and, next to it, <same>_README.md: the exact
commands for whoever runs it there, and what to send back. Nothing on the desktop is automated.

    python training/make_job_bundle.py --dataset-dir <folder with dataset.yaml, images/, labels/> \
        [--out <dir>] [--dataset-name NAME] -- --single-run --model-size m --batch 8 --rect 0 ...

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
DEFAULT_OUT = REPO.parent / "CryoAI-laptop" / "job_bundles"
SPLIT_KEYS = ("train", "val", "test")
STORED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pt", ".zip"}   # already compressed


def run_experiments_flags() -> set:
    src = (REPO / "training" / "run_experiments.py").read_text(encoding="utf-8")
    return set(re.findall(r'add_argument\(\s*"(--[\w-]+)"', src))


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


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
           ultra="unknown") -> str:
    args = " ".join(f'"{a}"' if " " in a else a for a in exp_args)
    rows = "\n".join(f"| {s} | {c['images']} | {c['labels']} |" for s, c in counts.items())
    chg = "\n".join(f"  - `{c}`" for c in changes) or "  - none needed"
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
   If that folder already exists, stop and ask: don't overwrite another dataset.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, type=Path,
                    help="prepared YOLO dataset: dataset.yaml, images/<split>/, labels/<split>/")
    ap.add_argument("--dataset-name", help="folder name on the desktop (default: --dataset-dir's name)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"default {DEFAULT_OUT}")
    ap.add_argument("exp_args", nargs=argparse.REMAINDER,
                    help="after --: arguments for training/run_experiments.py")
    a = ap.parse_args()

    root = a.dataset_dir.resolve()
    for need in ("dataset.yaml", "images", "labels"):
        if not (root / need).exists():
            raise SystemExit(f"{root} has no {need}")
    name = a.dataset_name or root.name
    exp = a.exp_args[1:] if a.exp_args[:1] == ["--"] else a.exp_args
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
    a.out.mkdir(parents=True, exist_ok=True)
    zip_path = a.out / f"{name}__job_{stamp}.zip"
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
    md5 = {}
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(f"{name}/dataset.yaml", text, compress_type=zipfile.ZIP_DEFLATED)
        for p in files:
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
        body = readme(name, zip_path.name, exp, changes, counts, commit, weights_note, created, ultra)
        job = {"dataset_name": name, "source_dir": str(root), "created": created, "cryoEV_commit": commit,
               "run_experiments_args": ["--dataset", name] + exp, "dataset_yaml_changes": changes,
               "ultralytics_bundled_on": ultra, "splits": counts, "md5": md5}
        z.writestr(f"{name}/JOB.json", json.dumps(job, indent=1), compress_type=zipfile.ZIP_DEFLATED)
        z.writestr(f"{name}/README.md", body, compress_type=zipfile.ZIP_DEFLATED)
    readme_path = zip_path.with_name(zip_path.stem + "_README.md")
    readme_path.write_text(body, encoding="utf-8")

    print(f"{zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB, {len(files)} files)")
    print(f"{readme_path}")
    for c in changes:
        print(f"  dataset.yaml {c}")
    print("  run on desktop: python training/run_experiments.py --dataset "
          f"\"{name}\" " + " ".join(f'"{x}"' if " " in x else x for x in exp))


if __name__ == "__main__":
    main()
