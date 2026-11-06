"""Dataset cache for training jobs (docs/JOBS.md, "Dataset cache"). Standard library only.

A job bundle no longer has to carry its dataset. make_job_bundle.py writes two files:
  ds_<sha12>.zip   the dataset pack: images/, labels/ and MANIFEST.json ({schema, dataset_sha256, files: {rel: md5}})
  <name>__job_<stamp>.zip   the thin job zip: <name>/dataset.yaml, JOB.json, README.md, weights/ (if packed)
JOB.json names the pack: `dataset_sha256` (the content key), `dataset_pack` (file name), `dataset_pack_md5`.

dataset_sha256 = sha256 over the sorted lines f"{rel}\\t{md5}\\n" of images/ and labels/ only: it depends on the
files' content and paths, not on zip bytes, timestamps or the dataset's name. Seed repeats of the same data share it.

On the queue (sync.py push writes it, the runner only reads it; nobody deletes automatically):
  <queue>/ds/.incoming/<sha>[.<time>]/   pack upload in progress (never read by the runner)
  <queue>/ds/<sha>/                      ds_<sha12>.zip, ds_<sha12>.zip.md5, READY (written last)

The runner (run_job.py) extracts the pack into work/<job>/training outputs/<name>/, then the thin zip over it,
and checks every file against MANIFEST.json. MANIFEST.json itself is never written into the job's folder, so the
trainer sees exactly the tree a single-zip (fat) bundle gives. Each job gets its own copy: the trainer writes
inside the dataset folder (runs/, labels/*.cache), so nothing may link into the shared cache.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

DS = "ds"                         # <queue>/ds/<sha>/
MANIFEST = "MANIFEST.json"        # in the pack, at its root
PACK_SCHEMA = "dataset_pack/1"
DATA_DIRS = ("images", "labels")
STORED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pt", ".zip"}   # already compressed
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class PackError(Exception):
    """A pack is missing or doesn't match the job (run_job.py turns it into Blocked)."""


def content_sha256(md5s: dict) -> str:
    """The dataset's content key: sha256 over sorted f"{rel}\\t{md5}\\n", images/ and labels/ files only."""
    lines = "".join(f"{rel}\t{md5s[rel]}\n" for rel in sorted(md5s) if rel.split("/", 1)[0] in DATA_DIRS)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def pack_name(sha: str) -> str:
    return f"ds_{sha[:12]}.zip"


def is_thin(job: dict) -> bool:
    return bool(job.get("dataset_sha256"))


def valid_sha(sha) -> bool:
    return isinstance(sha, str) and bool(SHA_RE.match(sha))


# ---------------------------------------------------------------- laptop: write a pack
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def write_pack(out_dir: Path, root: Path, files: list, fixed_time=False) -> tuple:
    """Write (or reuse) <out_dir>/ds_<sha12>.zip + .md5 for `files` (absolute paths under images/ and labels/ of
    `root`). -> (pack path, sha256, pack md5, {rel: md5}, reused). An existing pack with the same name is reused
    only if its sidecar md5 and its MANIFEST match; otherwise SystemExit (never overwritten).
    fixed_time=True (predict packs, rebuilt for every push): every member dated 1980-01-01, so the same files always
    give the same pack bytes and md5, and a second set with the same tiles reuses ds/<sha>/ on the share."""
    out_dir = Path(out_dir)
    md5s = {}
    for p in files:
        md5s[p.relative_to(root).as_posix()] = C.md5_file(p)
    sha = content_sha256(md5s)
    pack = out_dir / pack_name(sha)
    side_path = Path(str(pack) + ".md5")
    if pack.exists():
        side, got = C.sidecar_md5(pack), C.md5_file(pack)
        try:
            man = read_manifest(pack)
        except PackError as e:
            raise SystemExit(f"{pack} exists but {e}; move it aside and bundle again")
        if side != got or man.get("dataset_sha256") != sha or man.get("files") != md5s:
            raise SystemExit(f"{pack} exists but doesn't match this dataset (md5 {got}, sidecar {side}); "
                             f"move it aside and bundle again")
        return pack, sha, got, md5s, True
    tmp = pack.with_name(pack.name + ".part")
    manifest = {"schema": PACK_SCHEMA, "dataset_sha256": sha, "files": md5s}
    with zipfile.ZipFile(tmp, "w") as z:
        for p in sorted(files):
            rel = p.relative_to(root).as_posix()
            kind = zipfile.ZIP_STORED if p.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
            if fixed_time:
                z.writestr(zipfile.ZipInfo(rel, FIXED_ZIP_TIME), p.read_bytes(), compress_type=kind)
            else:
                z.write(p, rel, compress_type=kind)
        man = json.dumps(manifest, indent=1, sort_keys=True)
        if fixed_time:
            z.writestr(zipfile.ZipInfo(MANIFEST, FIXED_ZIP_TIME), man, compress_type=zipfile.ZIP_DEFLATED)
        else:
            z.writestr(MANIFEST, man, compress_type=zipfile.ZIP_DEFLATED)
    tmp.replace(pack)
    got = C.md5_file(pack)
    side_path.write_text(f"{got}  {pack.name}\n", encoding="utf-8")
    return pack, sha, got, md5s, False


def read_manifest(pack: Path) -> dict:
    try:
        with zipfile.ZipFile(pack) as z:
            return json.loads(z.read(MANIFEST))
    except (KeyError, ValueError, zipfile.BadZipFile, OSError) as e:
        raise PackError(f"dataset pack {Path(pack).name} has no readable {MANIFEST}: {e}")


# ---------------------------------------------------------------- runner: find, check, extract
def pack_path(root: Path, job: dict) -> Path:
    """<queue>/ds/<sha>/<dataset_pack> for a thin job. PackError if it isn't there (READY missing)."""
    sha, name = job.get("dataset_sha256"), job.get("dataset_pack")
    if not valid_sha(sha) or not name or Path(name).name != name or name != pack_name(sha):
        raise PackError(f"JOB.json dataset reference is malformed (dataset_sha256 {sha!r}, dataset_pack {name!r})")
    d = Path(root) / DS / sha
    if not (d / C.READY).is_file() or not (d / name).is_file():
        raise PackError(f"dataset pack missing: {DS}/{sha[:12]}... has no READY or no {name} "
                        f"(on the laptop: `sync.py push-pack --bundle <job zip>`, then `sync.py retry`)")
    return d / name


def check_pack(root: Path, job: dict, spec: dict) -> tuple:
    """For a thin job: the pack exists, its md5 matches JOB.json, job.json and its sidecar, and its MANIFEST matches
    dataset_sha256 and its members. -> (pack path, {rel: md5}). Raises PackError."""
    pack = pack_path(root, job)
    want = str(job.get("dataset_pack_md5") or "").lower()
    got = C.md5_file(pack)
    side = C.sidecar_md5(pack)
    queued = str(spec.get("dataset_pack_md5") or want).lower()
    if not want or got != want or got != queued or (side and side != got):
        raise PackError(f"dataset pack md5 mismatch: {pack.name} is {got}, JOB.json {want or None}, "
                        f"job.json {queued or None}, sidecar {side}")
    man = read_manifest(pack)
    files = man.get("files") if isinstance(man.get("files"), dict) else None
    if not files or man.get("dataset_sha256") != job["dataset_sha256"] or content_sha256(files) != job["dataset_sha256"]:
        raise PackError(f"dataset pack {pack.name}: MANIFEST doesn't match dataset_sha256 {job['dataset_sha256'][:12]}")
    with zipfile.ZipFile(pack) as z:
        members = {n for n in z.namelist() if not n.endswith("/")} - {MANIFEST}
    if members != set(files):
        extra, gone = sorted(members - set(files))[:3], sorted(set(files) - members)[:3]
        raise PackError(f"dataset pack {pack.name}: members differ from MANIFEST (extra {extra}, missing {gone})")
    if any(rel.split("/", 1)[0] not in DATA_DIRS or ".." in rel.split("/") for rel in files):
        raise PackError(f"dataset pack {pack.name}: a MANIFEST path is outside images/ and labels/")
    return pack, files


def extract_pack(pack: Path, dest: Path, files: dict):
    """Extract the pack's data files (not MANIFEST.json) into dest (the job's training outputs/<name>/)."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(pack) as z:
        z.extractall(dest, members=sorted(files))


def verify_tree(dest: Path, files: dict):
    """Every MANIFEST file is in dest with its md5. Raises PackError on the first few mismatches."""
    bad = []
    for rel in sorted(files):
        p = dest / rel
        if not p.is_file() or C.md5_file(p) != files[rel]:
            bad.append(rel)
    if bad:
        raise PackError(f"dataset files differ from the pack's MANIFEST: {len(bad)} file(s), e.g. {bad[:3]}")


# ---------------------------------------------------------------- queue listing (sync.py status)
def cache_entries(share: Path) -> list:
    """One dict per <share>/ds/<sha>/: sha, ready, bytes, refs = the jobs in q/ naming it that may still run (no
    DONE.json, or one that isn't done/cancelled: a blocked job can be retried)."""
    share = Path(share)
    refs = {}
    qroot = share / C.Q
    for d in sorted(qroot.iterdir()) if qroot.is_dir() else []:
        if d.name.startswith(".") or not (d / C.JOB_FILE).is_file():
            continue
        try:
            done = share / C.D / d.name / C.DONE
            if done.is_file() and C.read_json_shared(done).get("status") in ("done", "cancelled"):
                continue
            sha = C.read_json_shared(d / C.JOB_FILE).get("dataset_sha256")
        except (OSError, ValueError):
            continue
        if sha:
            refs.setdefault(sha, []).append(d.name)
    out = []
    root = share / DS
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if d.name.startswith(".") or not d.is_dir():
            continue
        out.append({"sha": d.name, "ready": (d / C.READY).is_file(),
                    "bytes": sum(p.stat().st_size for p in d.iterdir() if p.is_file()),
                    "refs": refs.get(d.name, [])})
    return out
