#!/usr/bin/env python3
"""
Inference check: score a model on a new imaging session from a small human-reviewed sample.

  stage  tiles folder -> stratified sample -> <out>/<batch_id>/ (images/, predictions/,
         sample.csv, BATCH.json); prints the web_qc command for the review
  score  web_qc.export on the batch's review -> review/export (corrected labels, YOLO + COCO),
         review/metrics, and a one-page RESULT.md
  log    append a status line by hand (e.g. --status reviewed --reviewer NAME)
  list   latest status of every batch in <out>/batches.jsonl

Usage (from the repo root):
  python inference/inference_check.py stage --tiles <dir> [--model v5_evnonev_y8m_20261029]
      [--predictions-dir <model>/predictions | --device cuda] [--n 24 --seed 1]
      [--manifest MANIFEST.csv] [--field-table field_table.csv|none] [--out <root>]
  python -m annotation.web_qc.server ...          (exact command printed by stage)
  python inference/inference_check.py score <batch_dir> [--reviewer NAME]

Sampling: greedy, seeded. Each pick is the tile that best balances, in order: an uncovered
grid x defocus stratum, grid, defocus bin within grid, montage, position in montage. So with
n >= the number of strata every stratum gets a tile, grids get equal shares (capped by what
they have), and no montage gets a second tile before every montage in its stratum has one.
Defocus bins are session rank thirds of mdoc_defocus_um (MANIFEST column or --field-table).

Without --predictions-dir, predict_models.py runs on the sampled tiles only.
<out>/batches.jsonl is append-only; see docs/INFERENCE_CHECK.md for its fields.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
REGISTRY = REPO / "models" / "models.yaml"
CLASSES = {"roboflow": REPO / "annotation" / "web_qc" / "classes_roboflow.yaml",
           "default": REPO / "annotation" / "web_qc" / "classes.yaml"}
DEFAULT_MODEL = "v5_evnonev_y8m_20261029"
DEFAULT_OUT = REPO.parent / "CryoAI-laptop" / "inference_checks"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}   # as web_qc.server
DEFOCUS_COLUMNS = ("mdoc_defocus_um", "defocus_um", "defocus")
NAME_COLUMNS = ("output_name", "image", "file_name", "filename", "name")
LOG_FIELDS = ["batch_id", "created", "session", "tiles_dir", "model", "predictions_dir",
              "n", "seed", "strata", "status", "reviewer", "cryoEV_commit"]
STATUSES = ("staged", "reviewed", "scored", "failed")
sys.path.insert(0, str(REPO))
# Raw data and synced storage: a batch is never written under these (any depth). The check (prefixes, UNC,
# trailing spaces, links) lives in data_utils/path_guard.py, shared with annotation/apply_review_edits.py.
from data_utils.path_guard import PROTECTED_ROOTS, protected_problem  # noqa: E402


def warn(msg: str):
    print(f"WARNING: {msg}", file=sys.stderr)


def _under(path: Path, root: Path) -> bool:
    p, r = os.path.normcase(str(path)), os.path.normcase(str(root))
    return p == r or p.startswith(r.rstrip("\\/") + os.sep)


def out_root_problem(root: Path, tiles_dir: Path, protected=PROTECTED_ROOTS) -> str | None:
    """Why --out may not be used, or None. Refuses protected roots and the tiles' session folder."""
    problem = protected_problem(root, protected)
    if problem:
        return f"--out {problem}; stage elsewhere"
    root = Path(os.path.abspath(root))
    tiles_dir = Path(os.path.abspath(tiles_dir))
    session = next((p for p in [tiles_dir, *tiles_dir.parents] if re.match(r"\d{8}", p.name)), tiles_dir)
    if protected_problem(root, [session]):
        return f"--out {root} is inside the session folder {session}; stage elsewhere"
    return None


def longest_output_path(bdir: Path, model: str, stems) -> int:
    """Length of the longest path stage + predict_models will write (p/<model>/predictions/<stem>.txt)."""
    return len(str(bdir / "p" / model / "predictions")) + max(len(s) for s in stems) + 5


# ---------------------------------------------------------------- tile table

def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def find_manifest(tiles_dir: Path) -> Path | None:
    for p in (tiles_dir / "MANIFEST.csv", tiles_dir.parent / f"{tiles_dir.name}_MANIFEST.csv"):
        if p.exists():
            return p
    return None


def find_field_table(tiles_dir: Path) -> Path | None:
    p = tiles_dir.parent / "sampling" / "field_table.csv"
    return p if p.exists() else None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def tile_table(tiles_dir: Path, manifest: Path | None, field_table: Path | None) -> list:
    """One dict per image: name, grid, montage, position, defocus_um (or None), nm_per_px."""
    images = sorted(p for p in tiles_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    man = {}
    if manifest:
        rows = read_csv(manifest)
        key = next((c for c in NAME_COLUMNS if rows and c in rows[0]), None)
        man = {r[key]: r for r in rows} if key else {}
        if not key:
            warn(f"{manifest.name} has none of the name columns {NAME_COLUMNS}; ignoring it")
        elif not any(p.name in man for p in images):
            warn(f"{manifest.name}: no {key} matches an image in {tiles_dir}; ignoring it")
    field = {}
    if field_table:
        for r in read_csv(field_table):
            for k in ("output_name", "tile"):
                if r.get(k):
                    field[r[k]] = r

    out = []
    for p in images:
        m = man.get(p.name, {})
        src = m.get("source_file", "")
        g = re.search(r"grid(\d+)", p.name)
        mont = re.search(r"(grid\d+_\d+)_r\d+c\d+", p.name)
        pos = re.search(r"_r(\d+)c(\d+)", p.name) or re.search(r"_r(\d+)c(\d+)", src)
        defocus = next((_num(m[c]) for c in DEFOCUS_COLUMNS if _num(m.get(c)) is not None), None)
        if defocus is None and field:
            keys = [p.name, p.stem, Path(src).stem] + p.stem.split("__")
            f = next((field[k] for k in keys if k in field), {})
            defocus = next((_num(f[c]) for c in DEFOCUS_COLUMNS if _num(f.get(c)) is not None), None)
        out.append({
            "name": p.name, "path": p,
            "grid": str(m.get("grid") or (g.group(1) if g else "na")),
            "montage": m.get("montage") or (mont.group(1) if mont else p.stem),
            "position": f"r{pos.group(1)}c{pos.group(2)}" if pos else "na",
            "defocus_um": defocus,
            "nm_per_px": _num(m.get("nm_per_px")),
        })
    for what, bad in (("grid", lambda t: t["grid"] == "na"),
                      ("montage", lambda t: t["montage"] == t["path"].stem),
                      ("position", lambda t: t["position"] == "na")):
        n = sum(map(bad, out))
        if n:
            warn(f"{n} of {len(out)} tile(s): {what} not in MANIFEST or file name "
                 f"(e.g. {next(t['name'] for t in out if bad(t))}); sampled as unknown")
    return out


def assign_defocus_bins(tiles: list, k: int) -> dict:
    """Label tiles dz1..dzk by rank of defocus in the session ('na' if unknown).

    dz1 = most negative mdoc defocus. Equal values share a bin. Returns {bin: [min, max]}.
    """
    vals = sorted(t["defocus_um"] for t in tiles if t["defocus_um"] is not None)
    if k < 1 or len(set(vals)) < 2:
        for t in tiles:
            t["defocus_bin"] = "na" if t["defocus_um"] is None else "all"
        return {}
    ranges = {}
    for t in tiles:
        d = t["defocus_um"]
        if d is None:
            t["defocus_bin"] = "na"
            continue
        t["defocus_bin"] = b = f"dz{1 + bisect.bisect_left(vals, d) * k // len(vals)}"
        lo, hi = ranges.get(b, (d, d))
        ranges[b] = [min(lo, d), max(hi, d)]
    return dict(sorted(ranges.items()))


# ---------------------------------------------------------------- sampler

def stratum(t: dict, levels) -> str:
    return "/".join(f"{lv}={t[lv]}" for lv in levels)


def stratified_sample(tiles: list, n: int, seed: int, levels=("grid", "defocus_bin")) -> list:
    """Pick n tiles, seeded; balance levels (outer first), then montage, then position.

    Deterministic for a given seed and tile set, whatever the input order.
    """
    rng = random.Random(seed)
    pool = sorted(tiles, key=lambda t: t["name"])
    tie = {t["name"]: rng.random() for t in pool}
    n_level, n_montage, n_position = Counter(), Counter(), Counter()
    picked = []

    def prefixes(t):
        return [tuple(t[lv] for lv in levels[:i + 1]) for i in range(len(levels))]

    def key(t):
        pre = prefixes(t)
        return ((n_level[pre[-1]] > 0,)                       # uncovered leaf stratum first
                + tuple(n_level[p] for p in pre)               # grid, then defocus within grid
                + (n_montage[t["montage"]], n_position[t["position"]], tie[t["name"]]))

    while pool and len(picked) < n:
        best = min(pool, key=key)
        pool.remove(best)
        picked.append(best)
        n_level.update(prefixes(best))
        n_montage[best["montage"]] += 1
        n_position[best["position"]] += 1
    return picked


# ---------------------------------------------------------------- helpers

def registry() -> dict:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["models"]


def registered_conf(spec: dict) -> tuple:
    """(conf, where it came from): a `conf` key, else 'use conf X' in note, else the best-F1 conf."""
    if spec.get("conf") is not None:
        return float(spec["conf"]), "models.yaml conf"
    m = re.search(r"use conf (\d*\.\d+)", str(spec.get("note", "")))
    if m:
        return float(m.group(1)), "models.yaml note"
    m = re.search(r"best F1 [\d.]+ at conf (\d*\.\d+)", str(spec.get("test_detection", "")))
    if m:
        return float(m.group(1)), "models.yaml test_detection (best F1)"
    return 0.5, "default (none registered)"


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def guess_session(tiles_dir: Path) -> str:
    for part in [tiles_dir, *tiles_dir.parents]:
        m = re.match(r"\d{8}(?:-\d+)?(?!\d)", part.name)  # 20260827-2 is its own session
        if m:
            return m.group(0)
    return tiles_dir.name


def resolve_predictions_dir(p: Path) -> Path:
    """Accept <model>/predictions or its parent <model>/."""
    if not any(p.glob("*.txt")) and (p / "predictions").is_dir():
        return p / "predictions"
    return p


def append_log(root: Path, batch: dict, status: str, reviewer=None, **extra):
    assert status in STATUSES, status
    line = {f: batch.get(f) for f in LOG_FIELDS}
    line.update(status=status, reviewer=reviewer, cryoEV_commit=git_commit(),
                logged=time.strftime("%Y-%m-%dT%H:%M:%S"), **extra)
    path = root / "batches.jsonl"
    # A line cut short (crash, sync conflict) must not swallow the next one.
    lead = ""
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            lead = "" if f.read(1) == b"\n" else "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(lead + json.dumps(line) + "\n")
    return line


def read_log(path: Path) -> list:
    """Parsed lines of batches.jsonl; unparseable lines are skipped with a warning."""
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            r["batch_id"]
        except (ValueError, KeyError, TypeError):
            warn(f"{path.name} line {i} is not a valid record; skipped: {line[:80]!r}")
            continue
        out.append(r)
    return out


def link_or_copy(src: Path, dst: Path) -> str:
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def count_preds(txt: Path, conf: float) -> tuple:
    rows = [l.split() for l in txt.read_text().splitlines()] if txt.exists() else []
    rows = [r for r in rows if len(r) >= 8]
    return len(rows), sum(float(r[1]) >= conf for r in rows)


def q(p) -> str:
    return f'"{p}"'


def load_batch(arg: str, out: Path) -> tuple:
    bdir = Path(arg)
    if not (bdir / "BATCH.json").exists():
        bdir = out / arg
    if not (bdir / "BATCH.json").exists():
        raise SystemExit(f"no BATCH.json in {arg} or {out / arg}")
    return bdir, json.loads((bdir / "BATCH.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- stage

def cmd_stage(a):
    reg = registry()
    if a.model not in reg:
        raise SystemExit(f"unknown --model {a.model}; registered: {list(reg)}")
    classes = CLASSES.get(a.classes) or Path(a.classes).resolve()
    if not classes.exists():
        raise SystemExit(f"--classes: no file {classes}")
    tiles_dir = a.tiles.resolve()
    manifest = a.manifest or find_manifest(tiles_dir)
    if a.field_table and str(a.field_table).lower() == "none":
        field_table = None
    else:
        field_table = a.field_table or find_field_table(tiles_dir)
    tiles = tile_table(tiles_dir, manifest, field_table)
    if not tiles:
        raise SystemExit(f"no images in {tiles_dir}")

    pred_src = None
    n_missing = 0
    if a.predictions_dir:
        pred_src = resolve_predictions_dir(a.predictions_dir.resolve())
        others = [m for m in reg if m != a.model and m in pred_src.parts]
        if others:
            raise SystemExit(f"--predictions-dir looks like {others[0]}'s output; pass --model {others[0]}")
        have = {p.stem for p in pred_src.glob("*.txt")}
        n_missing = sum(t["path"].stem not in have for t in tiles)
        tiles = [t for t in tiles if t["path"].stem in have]
        if not tiles:
            raise SystemExit(f"no prediction files in {pred_src} match the tiles")

    edges = assign_defocus_bins(tiles, a.defocus_bins)
    levels = ("grid", "defocus_bin") if edges else ("grid",)
    picked = stratified_sample(tiles, a.n, a.seed, levels)
    strata = dict(sorted(Counter(stratum(t, levels) for t in picked).items()))
    all_strata = sorted({stratum(t, levels) for t in tiles})

    conf, conf_src = (a.conf, "--conf") if a.conf is not None else registered_conf(reg[a.model])
    session = a.session or guess_session(tiles_dir)
    created = time.strftime("%Y-%m-%dT%H:%M:%S")
    batch_id = a.batch_id or f"{session}_{a.model}_{time.strftime('%Y%m%d-%H%M%S')}"
    root = a.out.resolve()
    problem = out_root_problem(root, tiles_dir)
    if problem:
        raise SystemExit(problem)
    bdir = root / batch_id
    if bdir.exists():
        raise SystemExit(f"{bdir} exists; pass another --batch-id")
    # Windows MAX_PATH (260) without long-path support: predict_models then fails mid-run.
    longest = longest_output_path(bdir, a.model, [t["path"].stem for t in picked])
    if os.name == "nt" and longest > 259:
        raise SystemExit(f"output paths would reach {longest} characters (Windows limit 260): "
                         f"use a shorter --out or --batch-id")
    stub = {"batch_id": batch_id, "created": created, "session": session, "tiles_dir": str(tiles_dir),
            "model": a.model, "predictions_dir": str(pred_src) if pred_src else None,
            "n": len(picked), "seed": a.seed, "strata": strata}
    root.mkdir(parents=True, exist_ok=True)
    bdir.mkdir()   # did not exist (checked above): on failure everything under it is ours to remove
    try:
        return _stage_into(a, bdir, root, picked, tiles, levels, strata, all_strata, edges, conf,
                           conf_src, classes, manifest, field_table, pred_src, n_missing, stub)
    except BaseException as exc:
        shutil.rmtree(bdir, ignore_errors=True)
        append_log(root, stub, "failed", a.reviewer, error=f"{type(exc).__name__}: {exc}"[:500])
        warn(f"staging failed; removed {bdir} and logged status 'failed'")
        raise


def _stage_into(a, bdir, root, picked, tiles, levels, strata, all_strata, edges, conf, conf_src,
                classes, manifest, field_table, pred_src, n_missing, stub):
    batch_id, created, session, tiles_dir = (stub[k] for k in ("batch_id", "created", "session", "tiles_dir"))
    tiles_dir = Path(tiles_dir)
    (bdir / "images").mkdir()
    (bdir / "predictions").mkdir()
    (bdir / "review").mkdir()

    # Copies by default: a hard link IS the raw file, so an in-place edit would change the source.
    put = link_or_copy if a.link else (lambda s, d: shutil.copy2(s, d) and "copied")
    modes = Counter(put(t["path"], bdir / "images" / t["name"]) for t in picked)
    if pred_src:
        pred_mode = "existing"
        for t in picked:
            shutil.copy2(pred_src / (t["path"].stem + ".txt"), bdir / "predictions")
    else:
        pred_mode = "predict_models"
        run_out = bdir / "p"   # short: Windows path limit
        cmd = [sys.executable, str(REPO / "inference" / "predict_models.py"), "--images",
               str(bdir / "images"), "--out", str(run_out), "--models", a.model,
               "--device", a.device, "--conf", str(a.pred_conf)]
        print("running:", " ".join(q(c) if " " in c else c for c in cmd))
        subprocess.run(cmd, cwd=REPO, check=True)
        pred_src = run_out / a.model / "predictions"
        for t in picked:   # an image with no detections still gets an (empty) file
            src = pred_src / (t["path"].stem + ".txt")
            dst = bdir / "predictions" / src.name
            shutil.copy2(src, dst) if src.exists() else dst.write_text("", encoding="utf-8")

    rows = []
    for i, t in enumerate(picked, 1):
        n_all, n_conf = count_preds(bdir / "predictions" / (t["path"].stem + ".txt"), conf)
        rows.append({"order": i, "image": t["name"], "grid": t["grid"], "montage": t["montage"],
                     "position": t["position"],
                     "defocus_um": "" if t["defocus_um"] is None else t["defocus_um"],
                     "defocus_bin": t["defocus_bin"], "stratum": stratum(t, levels),
                     "n_pred": n_all, f"n_pred_conf{conf:g}": n_conf, "source": str(t["path"])})
    with open(bdir / "sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    px = {t["nm_per_px"] for t in picked}
    nm_per_px = px.pop() if len(px) == 1 and None not in px else None
    review_cmd = (f"python -m annotation.web_qc.server --images-dir {q(bdir / 'images')} "
                  f"--predictions-dir {q(bdir / 'predictions')} --review-dir {q(bdir / 'review')} "
                  f"--classes {q(classes)}")
    score_cmd = f"python inference/inference_check.py score {q(bdir)} --reviewer <name>"
    batch = {
        "batch_id": batch_id, "created": created, "session": session,
        "tiles_dir": str(tiles_dir), "manifest": str(manifest) if manifest else None,
        "field_table": str(field_table) if field_table else None,
        "model": a.model, "model_conf": conf, "model_conf_source": conf_src,
        "predictions_dir": str(pred_src), "predictions_mode": pred_mode,
        "n": len(picked), "n_requested": a.n, "seed": a.seed,
        "n_candidates": len(tiles), "n_tiles_without_predictions": n_missing,
        "levels": list(levels) + ["montage", "position"], "defocus_bins_um": edges,
        "strata": strata, "strata_empty": [s for s in all_strata if s not in strata],
        "n_montages": len({t["montage"] for t in picked}),
        "images": dict(modes), "classes": str(classes), "nm_per_px": nm_per_px,
        "reviewer": a.reviewer, "cryoEV_commit": git_commit(),
        "commands": {"review": review_cmd, "score": score_cmd},
    }
    (bdir / "BATCH.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    append_log(root, batch, "staged", a.reviewer)

    print(f"\nbatch {batch_id}: {len(picked)} of {len(tiles)} tiles "
          f"({batch['n_montages']} montages), seed {a.seed}  ->  {bdir}")
    if n_missing:
        print(f"  {n_missing} tile(s) had no prediction file and were not eligible")
    print("  strata: " + ", ".join(f"{k}: {v}" for k, v in strata.items()))
    if batch["strata_empty"]:
        print(f"  not sampled (n too small): {', '.join(batch['strata_empty'])}")
    if edges:
        print("  defocus bins (um): " + ", ".join(f"{b} {lo:.2f}..{hi:.2f}" for b, (lo, hi) in edges.items()))
    print(f"  model {a.model}, scored at conf {conf} ({conf_src}); images {dict(modes)}")
    print(f"\nReview (from {REPO}):\n  cd {q(REPO)}\n  {review_cmd}\n\nThen score:\n  {score_cmd}\n")
    return bdir


# ---------------------------------------------------------------- score

def read_docs(review: Path) -> list:
    """Review JSONs marked Done, via web_qc's own loader (same images and order as the export)."""
    sys.path.insert(0, str(REPO))
    from annotation.web_qc.export import load_done
    return load_done(review)


def class_counts(docs: list, conf: float) -> tuple:
    """(reviewer-class rows, model-class rows) for RESULT.md."""
    truth = defaultdict(Counter)
    model = defaultdict(Counter)
    for d in docs:
        for o in d["objects"]:
            if o["source"] == "human":
                if o["status"] == "accepted":
                    truth[o["cls"]]["drawn"] += 1
                continue
            if o["status"] == "accepted":
                truth[o["cls"]]["kept" if o["cls"] == o["model_class"] else "reclassed in"] += 1
            if o["conf"] is not None and o["conf"] >= conf:
                c = model[o["model_class"]]
                c["n"] += 1
                c[o["status"] if o["status"] != "accepted"
                  else ("kept" if o["cls"] == o["model_class"] else "reclassed")] += 1
    return truth, model


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if (tp + fp) and (tp + fn) and p + r else 0.0
    return p, r, f


def write_result(bdir: Path, batch: dict, conf: float, iou: float, reviewer, cfg: dict) -> dict:
    review = bdir / "review"
    docs = read_docs(review)
    by_t = read_csv(review / "metrics" / "by_threshold.csv")
    at = next((r for r in by_t if abs(float(r["conf_cut"]) - conf) < 1e-9), None)
    floor = max([d.get("conf_cut") or 0.0 for d in docs] + [0.0])
    sample = {r["image"]: r for r in read_csv(bdir / "sample.csv")}
    is_ev = {c["name"]: c["is_ev"] for c in cfg["classes"]}

    L = [f"# Inference check `{batch['batch_id']}`", "",
         "| | |", "|---|---|",
         f"| Session | {batch['session']} (`{batch['tiles_dir']}`) |",
         f"| Model | `{batch['model']}`, scored at conf {conf} ({batch['model_conf_source']}) |",
         f"| Sample | {batch['n']} of {batch['n_candidates']} tiles, seed {batch['seed']}, "
         f"{batch['n_montages']} montages; by {' > '.join(batch['levels'])} |",
         f"| Reviewed (Done) | {len(docs)} of {batch['n']}" + (f", by {reviewer}" if reviewer else "")
         + (" (**partial**: scores cover the Done tiles only)" if len(docs) < batch["n"] else "") + " |",
         f"| Reviewed down to | conf {floor:.2f} (outlines below were never judged) |",
         f"| Scored | {time.strftime('%Y-%m-%d %H:%M')}, cryoEV {git_commit()} |", ""]

    metrics = {"conf": conf, "n_done": len(docs)}
    L += [f"## EV detection at conf >= {conf} (mask IoU >= {iou})", ""]
    if at:
        metrics.update(precision=float(at["precision"]), recall=float(at["recall"]),
                       f1=float(at["f1"]))
        L += ["| Precision | Recall | F1 | TP | FP | FN |", "|---:|---:|---:|---:|---:|---:|",
              f"| {float(at['precision']):.3f} | {float(at['recall']):.3f} | {float(at['f1']):.3f} "
              f"| {at['tp']} | {at['fp']} | {at['fn']} |", ""]
    else:
        L += [f"Not available: the reviewer worked down to conf {floor:.2f}, above {conf}.", ""]
    L += ["Truth = accepted outlines of an EV-type class + drawn outlines. Non-EV model "
          "outputs (model_class_to_class) are not EV predictions." if cfg.get("model_class_to_class")
          else "Truth = accepted outlines of an EV-type class + drawn outlines.", ""]
    pick = [r for r in by_t if float(r["conf_cut"]) in (0.25, 0.4, 0.5, 0.6, 0.75)]
    if pick:
        L += ["| conf | P | R | F1 |", "|---:|---:|---:|---:|"]
        L += [f"| {float(r['conf_cut']):.2f} | {float(r['precision']):.3f} | "
              f"{float(r['recall']):.3f} | {float(r['f1']):.3f} |" for r in pick]
        L.append("")

    # Every sampled stratum gets a row, also those with no Done tile yet (n_done 0).
    by = defaultdict(Counter)
    for r in sample.values():
        by[r["stratum"]]["sampled"] += 1
    done_names = {d["image"] for d in docs}
    for nm in done_names:
        by[sample.get(nm, {}).get("stratum", "?")]["n_done"] += 1
    per_image = review / "metrics" / "per_image.csv"
    if per_image.exists() and per_image.stat().st_size:
        for r in read_csv(per_image):
            s = sample.get(r["image"], {}).get("stratum", "?")
            by[s].update({k: int(r[k]) for k in ("tp", "fp", "fn")})
    L += [f"## By stratum (conf >= {conf})", "",
          "| Stratum | Sampled | Done | P | R | F1 | TP | FP | FN |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, c in sorted(by.items()):
        if c["n_done"] and at:
            p, r, f = prf(c["tp"], c["fp"], c["fn"])
            cells = f"{p:.2f} | {r:.2f} | {f:.2f} | {c['tp']} | {c['fp']} | {c['fn']}"
        else:
            cells = "– | – | – | – | – | –"
        L.append(f"| {s} | {c['sampled']} | {c['n_done']} | {cells} |")
    L.append("")
    metrics["by_stratum"] = {s: {"sampled": c["sampled"], "n_done": c["n_done"]} for s, c in sorted(by.items())}

    truth, model = class_counts(docs, conf)
    L += ["## Per-class counts (Done tiles)", "", "Reviewer's classes (the corrected labels):", "",
          "| Class | EV-type | Kept from model | Reclassed into | Drawn | Total |",
          "|---|:-:|---:|---:|---:|---:|"]
    for c in cfg["classes"]:
        t = truth.get(c["name"], Counter())
        tot = t["kept"] + t["reclassed in"] + t["drawn"]
        if tot:
            L.append(f"| {c['name']} | {'yes' if is_ev[c['name']] else 'no'} | {t['kept']} | "
                     f"{t['reclassed in']} | {t['drawn']} | {tot} |")
    L += ["", f"Model outlines at conf >= {conf}, by the class they started as:", "",
          "| Model class | n | Kept | Reclassed | Rejected | Pending |", "|---|---:|---:|---:|---:|---:|"]
    for name, c in sorted(model.items(), key=lambda kv: -kv[1]["n"]):
        L.append(f"| {name} | {c['n']} | {c['kept']} | {c['reclassed']} | {c['rejected']} | {c['pending']} |")
    metrics["per_class"] = {k: sum(v.values()) for k, v in truth.items()}

    L += ["", "## Outputs", "",
          "- Corrected labels: `review/export/` (`labels/` YOLO-seg, `_annotations.coco.json`, "
          f"class ids from `{Path(batch['classes']).name}`). Keep them with this batch as held-out "
          "test data for this session; add to training only through a planned split.",
          "- Full report: `review/metrics/qc_report.html`; tables in `review/metrics/*.csv`.",
          "- Sample: `sample.csv`; batch settings: `BATCH.json`; log: `../batches.jsonl`.", ""]
    (bdir / "RESULT.md").write_text("\n".join(L), encoding="utf-8")
    return metrics


def cmd_score(a):
    bdir, batch = load_batch(a.batch, a.out)
    review = bdir / "review"
    if not (review / "images_list.txt").exists():
        raise SystemExit(f"no review in {review} yet. Review first:\n  {batch['commands']['review']}")
    n_done = len(read_docs(review))
    if not n_done:
        raise SystemExit(f"no images marked Done in {review} yet (0 of {batch['n']}); "
                         "press Done on reviewed images in web_qc, then score again")
    conf = a.conf if a.conf is not None else batch["model_conf"]
    cmd = [sys.executable, "-m", "annotation.web_qc.export", "--images-dir", str(bdir / "images"),
           "--predictions-dir", str(bdir / "predictions"), "--review-dir", str(review),
           "--classes", batch["classes"], "--analysis-conf", str(conf), "--iou", str(a.iou)]
    px = a.pixel_size or batch.get("nm_per_px")
    if px:
        cmd += ["--pixel-size", str(px)]
    subprocess.run(cmd, cwd=REPO, check=True)

    sys.path.insert(0, str(REPO))
    from annotation.web_qc.server import load_classes
    reviewer = a.reviewer or batch.get("reviewer")
    metrics = write_result(bdir, batch, conf, a.iou, reviewer, load_classes(Path(batch["classes"])))
    partial = n_done < batch["n"]
    append_log(bdir.parent, batch, "scored", reviewer, partial=partial, n_done=n_done,
               metrics=metrics, result=str(bdir / "RESULT.md"))
    if partial:
        warn(f"partial: {n_done} of {batch['n']} images Done; logged as scored with partial=true")
    print(f"\n-> {bdir / 'RESULT.md'}")


def cmd_log(a):
    bdir, batch = load_batch(a.batch, a.out)
    line = append_log(bdir.parent, batch, a.status, a.reviewer, **({"note": a.note} if a.note else {}))
    print(json.dumps(line))


def cmd_list(a):
    path = a.out / "batches.jsonl"
    if not path.exists():
        raise SystemExit(f"no {path}")
    last = {r["batch_id"]: r for r in read_log(path)}
    for r in last.values():
        m = r.get("metrics") or {}
        f1 = f"F1 {m['f1']:.3f} @ {m['conf']}" if "f1" in m else ""
        part = f" ({r.get('n_done')}/{r.get('n')} done)" if r.get("partial") else ""
        print(f"{r['batch_id']:<55} {r.get('status'):<9} n={r.get('n')!s:<4} "
              f"{r.get('reviewer') or '':<10} {f1}{part}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"batch root (default {DEFAULT_OUT})")

    s = sub.add_parser("stage", help="sample tiles and stage a review batch")
    s.add_argument("--tiles", required=True, type=Path, help="folder of model-input tiles (8-bit PNG/JPG)")
    s.add_argument("--model", default=DEFAULT_MODEL, help="model_id from models/models.yaml")
    s.add_argument("--predictions-dir", type=Path, help="existing <model_id>/predictions; else run predict_models.py")
    s.add_argument("--n", type=int, default=24)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--manifest", type=Path, help="tile MANIFEST.csv (default: <tiles>/MANIFEST.csv or <tiles>_MANIFEST.csv)")
    s.add_argument("--field-table", help="CSV with tile/output_name + mdoc_defocus_um "
                                         "(default: <tiles>/../sampling/field_table.csv if present; 'none' = off)")
    s.add_argument("--defocus-bins", type=int, default=3, help="equal-count bins of defocus (0 = ignore defocus)")
    s.add_argument("--classes", default="roboflow",
                   help="'roboflow' (lab Roboflow names, default), 'default' (web_qc classes.yaml) or a path")
    s.add_argument("--conf", type=float, help="scoring conf (default: the model's registered conf)")
    s.add_argument("--device", default="cuda", help="for predict_models.py when no --predictions-dir")
    s.add_argument("--pred-conf", type=float, default=0.25, help="predict_models.py --conf")
    s.add_argument("--session", help="session name (default: first 8-digit folder name above --tiles)")
    s.add_argument("--batch-id")
    s.add_argument("--reviewer")
    s.add_argument("--link", action="store_true",
                   help="hard-link images instead of copying. WARNING: a hard link is the raw file; "
                        "editing a staged image in place changes the source data")
    common(s)
    s.set_defaults(func=cmd_stage)

    s = sub.add_parser("score", help="export the review, score the model, write RESULT.md")
    s.add_argument("batch", help="batch folder, or batch_id under --out")
    s.add_argument("--reviewer")
    s.add_argument("--conf", type=float, help="override the batch's model_conf")
    s.add_argument("--iou", type=float, default=0.5)
    s.add_argument("--pixel-size", type=float, help="nm/px (default: the MANIFEST nm_per_px, if one value)")
    common(s)
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("log", help="append a status line to batches.jsonl")
    s.add_argument("batch")
    s.add_argument("--status", required=True, choices=STATUSES)
    s.add_argument("--reviewer")
    s.add_argument("--note")
    common(s)
    s.set_defaults(func=cmd_log)

    s = sub.add_parser("list", help="latest status per batch")
    common(s)
    s.set_defaults(func=cmd_list)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
