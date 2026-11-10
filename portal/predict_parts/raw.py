"""
Predict, raw conversion inputs: the processing repo and its recipes, the raw form fields, the GPU check, raw
session folders (admin; read only) and the raw-file content checks (MRC / TIFF magic, .mdoc / .idoc sidecars).

Split out of portal/modules/predict.py (round 7, P7; no behaviour change); predict.py re-exports every name.
Names the tests patch are used from other parts only as <PART>.<name> (portal/predict_parts/__init__.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

from jobs import convert_raw as CR
from jobs import predict_set as PS
from portal import predict_sources as PSRC
from portal import upload_sessions as US

from portal.predict_parts.common import (GPU_CHECK_S, NOTES_SUB, RAW_ROLES, RAW_SCAN_DEPTH, RAW_SCAN_MAX,
                                         RAW_SKIP_DIR, RECIPE_TTL_S, REPO, Refusal, SAMPLE_KEY, TEST_HOOKS, _setting)


# ---------------------------------------------------------------- raw conversion: settings, recipes, checks
def test_hooks() -> dict:
    """jobs/run_job.py's test hooks (only when CRYOEV_JOBS_TESTING=1), read here without importing run_job (it puts
    jobs/ and training/ on sys.path, which the portal process must not get)."""
    if os.environ.get(TEST_HOOKS[0]) != "1":
        return {}
    try:
        h = json.loads(os.environ.get(TEST_HOOKS[1]) or "{}")
        f = os.environ.get(TEST_HOOKS[2])
        if f and Path(f).is_file():
            h.update(json.loads(Path(f).read_text(encoding="utf-8")))
        return h if isinstance(h, dict) else {}
    except (ValueError, OSError):
        return {}


def processing_repo(cfg) -> tuple:
    """-> (repo Path, None) or (None, why): CJ_PROCESSING_REPO, checked the way jobs/convert_raw.py checks it."""
    v = _setting(cfg, CR.REPO_ENV)
    if not v:
        return None, f"{CR.REPO_ENV} is not set: raw conversion is off"
    try:
        return CR.processing_repo(str(v)), None
    except CR.ConvertError as e:
        return None, str(e)


_RECIPES: dict = {}
_RECIPE_LOCK = threading.Lock()


def recipes(repo) -> list:
    """The processing repo's session recipes (datasets/process_<name>.py) that convert_raw's static check accepts,
    newest name first. Read only and never executed; cached RECIPE_TTL_S."""
    d = Path(repo) / "datasets"
    key = os.path.normcase(str(d))
    with _RECIPE_LOCK:
        hit = _RECIPES.get(key)
        if hit and 0 <= time.monotonic() - hit[0] < RECIPE_TTL_S:
            return list(hit[1])
    names = []
    try:
        files = sorted(d.glob("process_*.py"))
    except OSError:
        files = []
    for p in files:
        name = p.stem[len("process_"):]
        if not CR.RECIPE_RE.match(name):
            continue
        try:
            CR.check_recipe_source(p, name)
        except (CR.ConvertError, OSError):
            continue
        names.append(name)
    names.sort(reverse=True)
    with _RECIPE_LOCK:
        _RECIPES.clear()
        _RECIPES[key] = (time.monotonic(), names)
    return list(names)


def session_recipe(session, names) -> str | None:
    """The recipe named after this session (20260827-2 -> 20260827_2), when the repo has one."""
    want = str(session or "").replace("-", "_")
    return want if want in names else None


def valid_session_id(s) -> bool:
    return isinstance(s, str) and PSRC.session_of(s) == s and bool(PS.SESSION_RE.match(s))


def raw_fields(cfg, user, f: dict, session=None) -> dict:
    """The raw form's fields -> {session, recipe, gpu, repo}. Raises Refusal (403 for a basic user, 503 when the
    processing repo is not set up, else 400 listing every problem). session: fixed by a picked session folder."""
    if user.get("role") not in RAW_ROLES:
        raise Refusal("Converting raw images is for advanced users and the admin.", 403)
    repo, why = processing_repo(cfg)
    if repo is None:
        raise Refusal("Raw conversion is not set up on this server. Ask the admin.", 503, tech=why)
    problems = []
    sid = session if session is not None else str(f.get("session") or "").strip()
    if not valid_session_id(sid):
        problems.append("Session: the imaging date as YYYYMMDD, or YYYYMMDD-n for a later session that day "
                        "(e.g. 20261027 or 20260827-2).")
    recipe = str(f.get("recipe") or "default").strip()
    if recipe != "default" and recipe not in recipes(repo):
        problems.append("Recipe: pick one from the list.")
    samples, bad = US.form_samples(f)                  # the upload form's sample per grid (blank: none)
    problems += bad
    if problems:
        raise Refusal(f"Please fix ({len(problems)}):\n" + "\n".join(problems))
    return {"session": sid, "recipe": recipe, "gpu": f.get("gpu") == "yes", "repo": repo, "samples": samples}


def gpu_check() -> tuple:
    """jobs/convert_raw.gpu_check() (= jobs/predict_local.py's) in a short child process, so the portal never imports
    jobs/run_job.py. -> (ok, note)."""
    code = "import json; from jobs import convert_raw as CR; print(json.dumps(list(CR.gpu_check())))"
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                           timeout=GPU_CHECK_S, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        ok, note = json.loads(r.stdout.strip().splitlines()[-1])
        return bool(ok), str(note)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as e:
        return False, f"GPU check failed: {type(e).__name__}: {e}"


def check_gpu(fields: dict):
    if fields.get("gpu"):
        ok, note = gpu_check()
        if not ok:
            raise Refusal("The GPU can’t be used now. Untick “Use the GPU” to convert on the CPU, or try later.",
                          409, tech=note)


# ---------------------------------------------------------------- raw session folders (admin; read only)
def raw_sessions(cfg) -> list:
    """Session folders under the scan roots with no model-ready tiles yet (predict_sources' "no model-ready tiles"):
    [{key, folder, root, dir, session, label}]."""
    out = []
    for s in PSRC.discover(cfg)["skipped"]:
        if not str(s.get("reason") or "").startswith("no model-ready tiles"):
            continue
        sid = PSRC.session_of(s["folder"])
        if not sid:
            continue
        d = Path(s["root"]) / s["folder"]
        out.append({"key": hashlib.sha1(os.path.normcase(str(d)).encode("utf-8")).hexdigest()[:10],
                    "folder": s["folder"], "root": Path(s["root"]), "dir": d, "session": sid,
                    "label": PSRC.default_label(sid)})
    out.sort(key=lambda s: (s["session"][:8], s["session"][8:]), reverse=True)
    return out


def raw_montages(s: dict) -> dict:
    """{"montages": [{rel, name, kind, bytes, files, atlas}], "sample_key": Path | None, "truncated": bool} for one
    session folder: .mrc.mdoc (with its .mrc beside it) and .idoc files, RAW_SCAN_DEPTH folders deep (processed,
    gallery and model folders skipped). Reads only; follows no link out of the scan root."""
    rr = PSRC.real(s["root"])
    found, seen = [], {"n": 0, "cut": False}

    def walk(p: Path, depth: int):
        try:
            entries = sorted(os.scandir(p), key=lambda e: e.name.lower())
        except OSError:
            return
        for e in entries:
            seen["n"] += 1
            if seen["n"] > RAW_SCAN_MAX:
                seen["cut"] = True
                return
            try:
                if e.is_dir(follow_symlinks=False):
                    if depth < RAW_SCAN_DEPTH and not RAW_SKIP_DIR.match(e.name) and PSRC.inside(e.path, rr):
                        walk(Path(e.path), depth + 1)
                    continue
                if not e.is_file(follow_symlinks=False):
                    continue
            except OSError:
                continue
            low, path = e.name.lower(), Path(e.path)
            if low.endswith(".mrc.mdoc"):
                mrc = path.with_name(e.name[:-len(".mdoc")])
                if not (mrc.is_file() and PSRC.inside(mrc, rr)):
                    continue
                kind, size = "MRC", mrc.stat().st_size
            elif low.endswith(".idoc"):
                kind, size = "TIFF", None
            else:
                continue
            if not PSRC.inside(path, rr):
                continue
            found.append({"rel": path.relative_to(s["dir"]).as_posix(), "name": e.name, "kind": kind, "bytes": size,
                          "atlas": "atlas" in low})
    walk(s["dir"], 0)
    key = None
    for cand in (s["dir"] / NOTES_SUB / SAMPLE_KEY, s["dir"] / SAMPLE_KEY):
        if cand.is_file() and PSRC.inside(cand, rr):
            key = cand
            break
    return {"montages": found, "sample_key": key, "truncated": seen["cut"]}


# ---------------------------------------------------------------- raw file content checks (MRC / TIFF, sidecars)
MRC_MAGIC = b"MAP"
TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


SIDECAR_EXT = (".idoc", ".mdoc")


def sidecar_images(path: Path) -> list:
    """Every [Image = <name>] of a SerialEM .idoc / .mrc.mdoc, read exactly as cryoem_core.metadata.parse_metadata
    reads it (a line starting with "[" and holding "=": the section type before the first "=", the name between the
    first and second "=", "]" and spaces trimmed). Latin-1, so every byte is kept and no path character is lost."""
    out = []
    with open(path, "rb") as f:
        text = f.read().decode("latin-1")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and "=" in line:
            if line.split("=")[0].strip()[1:].strip() == "Image":
                out.append(line.split("=")[1].strip().rstrip("]").strip())
    return out


def image_ref_problem(name: str) -> str | None:
    """A tile name in a sidecar must be a bare file name: no \\ / : (no drive, folder or UNC share), not . or ..,
    not empty, no control characters."""
    if not name or name in (".", "..") or any(c in name for c in "\\/:") or any(ord(c) < 32 for c in name):
        return f"names a tile {name[:80]!r} that is not a plain file name (no paths allowed)"
    return None


def sidecar_problems(path: Path, declared=None) -> list:
    """Problems with the tiles a sidecar names: each must be a bare file name and, for an upload (declared = the
    uploaded names), one of the uploaded files."""
    out = []
    for name in sidecar_images(path):
        why = image_ref_problem(name)
        if why is None and declared is not None and name not in declared:
            why = f"names a tile {name[:80]!r} that was not uploaded with it"
        if why:
            out.append(f"{path.name}: {why}")
    return list(dict.fromkeys(out))


def raw_problem(path: Path) -> str | None:
    """A raw upload file checked by its content: an MRC stack, a SerialEM sidecar, a TIFF tile, a sample key."""
    n = path.name.lower()
    with open(path, "rb") as f:
        head = f.read(65536)
    if n.endswith(".mrc"):
        if len(head) < 1024 or head[208:211] != MRC_MAGIC:
            return f"{path.name}: not an MRC file (by content)"
        nx, ny, nz = struct.unpack("<3i", head[:12])
        if min(nx, ny, nz) < 1:
            return f"{path.name}: an MRC file with no image in it"
    elif n.endswith((".mdoc", ".idoc")):
        if b"\x00" in head or b"PixelSpacing" not in head or not (b"[ZValue" in head or b"[Image" in head):
            return f"{path.name}: not a SerialEM {n.rsplit('.', 1)[-1]} file (by content: no PixelSpacing and tiles)"
        bad = sidecar_problems(path)                     # a path in [Image = ...]: refused at once
        if bad:
            return bad[0]
    elif n.endswith((".tif", ".tiff")):
        if head[:4] not in TIFF_MAGIC:
            return f"{path.name}: not a TIFF (by content)"
    elif n.endswith(".csv"):
        try:
            first = head.decode("utf-8-sig").splitlines()[0]
        except (UnicodeDecodeError, IndexError):
            return f"{path.name}: not a sample key (a text .csv with a #,Label header)"
        if b"\x00" in head or "label" not in first.lower():
            return f"{path.name}: not a sample key (a text .csv with a #,Label header)"
    return None


def file_problem(kind: str, path: Path) -> str | None:
    if kind == "tiles":
        try:
            PS.inspect_image(path)
        except (ValueError, OSError) as e:
            return f"{path.name}: {e}"
        return None
    return raw_problem(path)
