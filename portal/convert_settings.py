"""
Conversion settings for the portal (round 7 phase 2, piece CU): the logic behind the settings readout and the
"Adjust settings" panel. No HTML here (portal/modules/predict.py renders), and the processing code is NEVER imported:
everything goes through `python -m jobs.convert_raw describe | preview` in a child process (docs/JOBS.md,
"Conversion settings").

    clean_params(raw)                  -> {key: number | bool}: the four user-adjustable settings, checked against the same
                                          ranges as the CLI (a refusal here means the CLI is never called)
    peek_recipe(repo, name)            -> the cached document or None (never runs the child)
    describe_recipe(cmd, repo, name)   -> (doc, err): `describe --recipe`, cached per recipe-file md5 (+ a TTL)
    describe_conv(cmd, repo, d)        -> (doc, err): `describe --conv`, cached per convert.json / status.json mtime
    run_preview(cmd, repo, d, tile, params, out_dir) -> (doc, err): `preview`, ONE at a time per portal process
    PreviewBusy                        raised when a preview is already rendering (the caller answers 429)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent        # the cryoEV checkout: `python -m jobs.convert_raw` runs from here
RANGES = {"clip_lo_pct": (0.0, 5.0), "clip_hi_pct": (95.0, 100.0), "bandpass_nm": (0.5, 10.0)}
KEYS = tuple(RANGES) + ("denoise",)
LABELS = {"clip_lo_pct": "Clip low %", "clip_hi_pct": "Clip high %", "bandpass_nm": "Bandpass nm",
          "denoise": "Denoise"}
NUM_RE = re.compile(r"^-?\d{1,6}(\.\d{1,6})?$")
DESCRIBE_TIMEOUT_S = 90
PREVIEW_TIMEOUT_S = 240
DESCRIBE_TTL_S = 6 * 3600
FAIL_TTL_S = 30
CONV_TTL_S = 600


class ParamError(ValueError):
    """The settings were refused (never reaches the CLI)."""


class PreviewBusy(RuntimeError):
    """A preview is already being rendered by this portal process."""


def _num(key, v) -> float:
    if isinstance(v, bool) or v is None:
        raise ParamError(f"{LABELS[key]}: a number is needed.")
    if isinstance(v, str):
        if not NUM_RE.match(v.strip()):
            raise ParamError(f"{LABELS[key]}: “{v[:20]}” is not a number.")
        v = float(v.strip())
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ParamError(f"{LABELS[key]}: a finite number is needed.")
    lo, hi = RANGES[key]
    if not lo <= v <= hi:
        raise ParamError(f"{LABELS[key]}: {v:g} is outside {lo:g} to {hi:g}.")
    v = round(float(v), 4)                      # 99.5 stays 99.5, never 99.5000001; 99 stays 99
    return int(v) if v == int(v) else v


def _bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "on", "yes", "1"):
        return True
    if isinstance(v, str) and v.strip().lower() in ("false", "off", "no", "0"):
        return False
    raise ParamError("Denoise: on or off is needed.")


def clean_params(raw) -> dict:
    """The user's settings -> a clean dict of the allowed keys only. Unknown keys, wrong types, NaN and out-of-range
    values raise ParamError. Absent keys stay absent (no override)."""
    if not isinstance(raw, dict):
        raise ParamError("The settings must be an object.")
    bad = sorted(str(k)[:30] for k in raw if k not in KEYS)
    if bad:
        raise ParamError("Unknown setting: " + ", ".join(bad))
    out = {}
    for k in KEYS:
        if k in raw and raw[k] is not None and raw[k] != "":
            out[k] = _bool(raw[k]) if k == "denoise" else _num(k, raw[k])
    if out.get("denoise") is False and "bandpass_nm" in out:      # the CLI refuses it too (exit 2)
        raise ParamError("Bandpass nm: there is no cutoff with denoise off.")
    return out


def params_json(params: dict) -> str:
    """The --params argument: compact, stable key order."""
    return json.dumps({k: params[k] for k in KEYS if k in params}, separators=(",", ":"))


def diff(recipe: dict, eff: dict) -> list:
    """[(key, recipe value, used value)] where the used value is not the recipe's."""
    out = []
    for k in KEYS:
        a, b = (recipe or {}).get(k), (eff or {}).get(k)
        if b is None and a is None:
            continue
        same = (a == b) if (isinstance(a, bool) or isinstance(b, bool)) else (
            a is not None and b is not None and abs(float(a) - float(b)) < 1e-9)
        if not same:
            out.append((k, a, b))
    return out


# ---------------------------------------------------------------- the child process
def _env(repo):
    from jobs import convert_raw as CR
    return {**os.environ, CR.REPO_ENV: str(repo), "PYTHONIOENCODING": "utf-8"}


def run_cli(cmd: list, repo, args: list, timeout: int, cwd) -> tuple:
    """-> (doc, err). cmd is the converter command (python -m jobs.convert_raw); one JSON object on stdout."""
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        p = subprocess.run(list(cmd) + [str(a) for a in args], cwd=str(cwd), stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=timeout, env=_env(repo), **kw)
    except subprocess.TimeoutExpired:
        return None, f"it took longer than {timeout} s and was stopped"
    except OSError as e:
        return None, f"could not start: {type(e).__name__}: {e}"
    text = p.stdout.decode("utf-8", "replace").strip()
    doc = None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                doc = json.loads(line)
            except ValueError:
                doc = None
            break
    if isinstance(doc, dict) and doc.get("ok") and p.returncode == 0:
        return doc, None
    if isinstance(doc, dict):
        return None, str(doc.get("error") or "refused")
    tail = p.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no output"]
    return None, f"exit {p.returncode}: {tail[0][:300]}"


_DESC: dict = {}
_DESC_LOCK = threading.Lock()
_DESC_SEM = threading.Semaphore(2)


def _md5(p: Path) -> str:
    try:
        return hashlib.md5(p.read_bytes()).hexdigest()
    except OSError:
        return ""


def _cached(key, ttl_ok, fetch):
    now = time.time()
    with _DESC_LOCK:
        hit = _DESC.get(key)
    if hit and now - hit[0] < (ttl_ok if hit[1] is not None else FAIL_TTL_S):
        return hit[1], hit[2]
    with _DESC_SEM:
        doc, err = fetch()
    with _DESC_LOCK:
        if len(_DESC) > 500:
            _DESC.clear()
        _DESC[key] = (time.time(), doc, err)
    return doc, err


def recipe_file(repo, name) -> Path:
    return Path(repo) / "datasets" / f"process_{name}.py"


def describe_recipe(cmd, repo, name, file_name=None) -> tuple:
    """`describe --recipe <name> --json`. file_name: the recipe file's session name (for "default": convert_raw's
    DEFAULT_RECIPE), used for the cache key (its md5)."""
    key = ("recipe", os.path.normcase(str(repo)), name, _md5(recipe_file(repo, file_name or name)))
    return _cached(key, DESCRIBE_TTL_S, lambda: run_cli(
        cmd, repo, ["describe", "--recipe", name, "--json"], DESCRIBE_TIMEOUT_S, HOME))


def peek_recipe(repo, name, file_name=None):
    """The cached `describe --recipe` document, or None (never runs the child: for pages that must not wait)."""
    key = ("recipe", os.path.normcase(str(repo)), name, _md5(recipe_file(repo, file_name or name)))
    with _DESC_LOCK:
        hit = _DESC.get(key)
    return hit[1] if hit and hit[1] is not None and time.time() - hit[0] < DESCRIBE_TTL_S else None


def describe_conv(cmd, repo, d: Path) -> tuple:
    """`describe --conv <d> --json`, cached until the conversion's convert.json / status.json change."""
    def mt(n):
        try:
            return (d / n).stat().st_mtime_ns
        except OSError:
            return 0
    key = ("conv", os.path.normcase(str(d)), mt("convert.json"), mt("status.json"))
    return _cached(key, CONV_TTL_S, lambda: run_cli(
        cmd, repo, ["describe", "--conv", str(d), "--json"], DESCRIBE_TIMEOUT_S, HOME))


def clear_cache():
    with _DESC_LOCK:
        _DESC.clear()


# ---------------------------------------------------------------- preview: one at a time
_PREVIEW = threading.Lock()


def preview_busy() -> bool:
    return _PREVIEW.locked()


def run_preview(cmd, repo, d: Path, tile: str, params: dict, out_dir: Path) -> tuple:
    """`preview --conv <d> --tile <t> --params <json> --out <out_dir> --json` (CPU). Refuses to start a second one."""
    if not _PREVIEW.acquire(blocking=False):
        raise PreviewBusy()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        args = ["preview", "--conv", str(d), "--tile", tile, "--out", str(out_dir), "--json"]
        if params:
            args += ["--params", params_json(params)]
        return run_cli(cmd, repo, args, PREVIEW_TIMEOUT_S, HOME)
    finally:
        _PREVIEW.release()
