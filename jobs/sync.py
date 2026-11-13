#!/usr/bin/env python3
"""Laptop side of the training-job queue (docs/JOBS.md): push bundles to the desktop's queue share, pull results.

    python jobs/sync.py push <job id> --bundle <zip> --share \\\\desktop\\cj
    python jobs/sync.py pull --share \\\\desktop\\cj [--who pujan] [--all-weights]
    python jobs/sync.py status --share \\\\desktop\\cj
    python jobs/sync.py retry <job id> --share \\\\desktop\\cj
    python jobs/sync.py push-pack --bundle <zip> --share ...   (only a thin bundle's dataset pack, e.g. after pruning)
    python jobs/sync.py release <job id> --share ...      (a job pushed with --hold)
    python jobs/sync.py stop|start --share ...            (pause / resume the desktop watcher, jobs/watch.py)
    python jobs/sync.py cancel <job id> --share ... [--reason "..."]   (stop a queued or running job)
    python jobs/sync.py local <job id> --bundle <zip> [--max-hours H] [--allow-big]   (train on this machine)

push: the job must be a `train` job in the registry (`registry.py job new --type train --where desktop ...`).
  Copies the zip (+ .md5) to q/.incoming/<job>/, checks the copy's md5, writes job.json, renames the folder to
  q/<job>/ and writes READY last. A thin bundle (JOB.json has dataset_sha256) first gets its dataset pack
  (ds_<sha12>.zip, next to the zip) onto the share as ds/<sha>/, via ds/.incoming/, only if ds/<sha>/ isn't there
  yet; an existing ds/<sha>/ with another pack md5 is refused (docs/JOBS.md, "Dataset cache").
pull: for each d/<job>/ with DONE.json not pulled yet, copies runs/<run>/ to <CryoAI>/training outputs/<dataset>/
  runs/<run>/ (new folders only, never overwrites; staged under <dataset>/.pull_tmp/), runs the importer
  (`registry.py import`) and `registry.py set-status <job> done|blocked --reason`, then logs the pull in
  <CryoAI-laptop>/jobs/pulled.jsonl. Pulling again adds nothing. weights/epoch*.pt (save_period checkpoints,
  ~209 MB each) stay on the share unless --all-weights; best.pt and last.pt always come; the ledger line's
  not_pulled = {files, bytes, paths} lists what was left.
retry: moves d/<job>/ to d/.retired/<job>_<time>/ so the runner picks the job again (nothing is deleted).
queue_train(job, dataset_dir, exp_args, share, max_hours) (Python only; the portal's Train page): make_job_bundle's
  make_bundle, then push, in one call; refusals listed on the function.
push-eval (--bundle <zip from make_job_bundle.py --eval> | --weights <pt> --dataset <id>) --share ... [--job ...]: an
  eval job (Ultralytics val, test split, overlap_mask=False); pull lands its e/ in <CryoAI-laptop>/eval/<job>/ and
  sets the registry job's status, never copying run folders or running the importer. Off unless $CRYOEV_EVAL_JOBS=1
  (docs/JOBS.md, "Eval jobs").
push-predict --ps <ps id> --share ...: a prediction set as a predict job (images-only dataset pack + thin zip); pull
  lands its p/ in <CryoAI-laptop>/predict/<ps>/ and updates the set's status.json and registry record. Off unless
  $CRYOEV_PREDICT_JOBS=1 (docs/JOBS.md, "Predict jobs").
local: push to a folder on this machine (default C:\\cj_local), run jobs/run_job.py on it, then pull.
Copies use robocopy /E /Z /R:2 /W:5 when the share is a UNC path (never /MIR, /PURGE or /MOV*; exit >= 8 =
failure), shutil for a local folder. The registry is written only through registry.py, and only by pull.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "registry"))
import actor  # noqa: E402  (registry/actor.py: who + via)
import common as C  # noqa: E402
import dataset_cache  # noqa: E402
from data_utils import config  # noqa: E402
from data_utils.path_guard import protected_problem  # noqa: E402

REGISTRY_PY = REPO / "registry" / "registry.py"
DEFAULT_WHO = "pujan"                            # the old default; actor.who(): --who flag > $CRYOEV_FOR > this
LEDGER = Path("jobs") / "pulled.jsonl"          # under <CryoAI-laptop>
PULL_TMP = ".pull_tmp"                           # <CryoAI>/training outputs/<dataset>/.pull_tmp/<run> while copying
EPOCH_CKPT = "epoch*.pt"                         # runs/<run>/weights/: left on the share by pull unless --all-weights
LOCAL_QUEUE = Path("C:/cj_local")                # `local`: the laptop's own queue (short: Windows path limit)


class SyncError(RuntimeError):
    pass


def registry_job(job_id, root=None):
    """The job's current registry record (read only), or None. root: the registry folder (default the configured one)."""
    import registry
    return registry.current("job", Path(root) if root else config.registry_root()).get(job_id)


# ---------------------------------------------------------------- push
def git(*args, timeout=60):
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=timeout, env=env)


def commit_on_origin(commit):
    """None when `commit` is on origin/main after a fetch (the desktop watcher trains with main), else why not."""
    try:
        r = git("fetch", "origin", "main", timeout=120)
        if r.returncode:
            return f"git fetch failed ({r.stderr.strip()[:200]}); can't check that the code is on GitHub"
        if git("merge-base", "--is-ancestor", str(commit), "origin/main").returncode:
            return f"bundle commit {commit} is not on origin/main: push the repo first (the desktop trains with main)"
    except (OSError, subprocess.SubprocessError) as e:
        return f"git failed: {e}"
    return None


def push(job_id, bundle, share, copier="auto", check_registry=True, check_origin=True, max_hours=None, hold=False,
         job_type="train", extra=None):
    """Queue a bundle. job_type / extra are for push_predict (a predict job's job.json gets type predict and
    extra = {ps, model, n_images}); a train job's job.json is the same as ever."""
    import ids
    bundle, share = Path(bundle), Path(share)
    if not share.is_dir():                         # never create a queue root: a typo would make a stray folder
        raise SyncError(f"share {share} not found: a desktop share is \\\\host\\cj (two leading backslashes), "
                        f"and the queue root must already exist")
    if not ids.is_id("job", job_id) or len(job_id) > C.MAX_JOB_ID:
        raise SyncError(f"{job_id!r} is not a job id of <= {C.MAX_JOB_ID} characters (job_<YYYYMMDD>_<slug>)")
    if check_registry:
        rec = registry_job(job_id)
        if rec is None or rec.get("type") != job_type:
            raise SyncError(f"{job_id} is not a {job_type} job in the registry; create it with "
                            f"`registry.py job new --type {job_type} --where desktop ...`")
    if not bundle.is_file():
        raise SyncError(f"no bundle {bundle}")
    md5 = C.md5_file(bundle)
    side = C.sidecar_md5(bundle)
    if side and side != md5:
        raise SyncError(f"md5 mismatch: {bundle.name} is {md5}, its .md5 sidecar says {side}")
    with zipfile.ZipFile(bundle) as z:
        names = [n for n in z.namelist() if n.count("/") == 1 and n.endswith("/JOB.json")]
        if len(names) != 1:
            raise SyncError(f"{bundle.name} has no <dataset>/JOB.json (make it with training/make_job_bundle.py)")
        job = json.loads(z.read(names[0]))
    if check_origin:
        why = commit_on_origin(job.get("cryoEV_commit"))
        if why:
            raise SyncError(why)
    q = share / C.Q
    if (q / job_id).exists() or (share / C.D / job_id).exists():
        raise SyncError(f"{job_id} is already in the queue (q/ or d/); use a new job, or `retry` for d/")
    pack = {}
    if dataset_cache.is_thin(job):                     # the dataset goes once per queue, before the job names it
        pack = push_pack(job, bundle, share, copier)
    inc = q / C.INCOMING / job_id
    files = [bundle.name] + ([bundle.name + ".md5"] if side else [])
    C.copy_files(bundle.parent, inc, files, copier)
    if C.md5_file(inc / bundle.name) != md5:
        raise SyncError(f"md5 of the copy in {inc} differs; left there, push again")
    C.write_json(inc / C.JOB_FILE, {"schema": "queue_job/1", "job": job_id, "type": job_type,
                                    "bundle_id": job.get("bundle_id") or bundle.stem, "zip": bundle.name,
                                    "zip_md5": md5, "dataset": job.get("dataset_name"),
                                    "pushed": C.now(), "pushed_from": platform.node(), "max_hours": max_hours,
                                    **pack, **(extra or {})})
    inc.rename(q / job_id)
    if not hold:
        (q / job_id / C.READY).write_text(C.now() + "\n", encoding="utf-8")  # last
    return q / job_id


def push_bundle_pack(bundle, share, copier="auto") -> Path:
    """`push-pack`: only the dataset pack of a thin bundle (e.g. after ds/<sha>/ was pruned while a job still needed
    it: then `retry` that job). -> <share>/ds/<sha>."""
    bundle, share = Path(bundle), Path(share)
    if not share.is_dir():
        raise SyncError(f"share {share} not found")
    try:
        with zipfile.ZipFile(bundle) as z:
            names = [n for n in z.namelist() if n.count("/") == 1 and n.endswith("/JOB.json")]
            job = json.loads(z.read(names[0])) if len(names) == 1 else {}
    except (OSError, zipfile.BadZipFile, ValueError) as e:
        raise SyncError(f"can't read {bundle}: {e}")
    if not dataset_cache.is_thin(job):
        raise SyncError(f"{bundle.name} is not a thin bundle (its JOB.json names no dataset pack)")
    push_pack(job, bundle, share, copier)
    return share / dataset_cache.DS / job["dataset_sha256"]


def push_pack(job, bundle, share, copier="auto") -> dict:
    """Make sure <share>/ds/<sha>/ holds the job's dataset pack (from next to the bundle), uploading it only if the
    share doesn't have it yet. -> the job.json fields naming it. Never deletes or overwrites on the share."""
    sha, name = job.get("dataset_sha256"), job.get("dataset_pack")
    if not dataset_cache.valid_sha(sha) or name != dataset_cache.pack_name(sha):
        raise SyncError(f"{bundle.name}: JOB.json dataset reference is malformed (dataset_sha256 {sha!r}, "
                        f"dataset_pack {name!r}); rebundle")
    local_pack = bundle.parent / name
    if not local_pack.is_file():
        raise SyncError(f"dataset pack {name} not found next to {bundle.name} (make_job_bundle.py writes both)")
    md5 = C.md5_file(local_pack)
    side = C.sidecar_md5(local_pack)
    if md5 != str(job.get("dataset_pack_md5", "")).lower() or (side and side != md5):
        raise SyncError(f"md5 mismatch: {name} is {md5}, JOB.json says {job.get('dataset_pack_md5')}, "
                        f"its .md5 sidecar {side}")
    fields = {"dataset_sha256": sha, "dataset_pack": name, "dataset_pack_md5": md5}
    ds = Path(share) / dataset_cache.DS
    dest = ds / sha
    if dest.exists():
        theirs = C.sidecar_md5(dest / name)
        if theirs != md5:
            raise SyncError(f"same dataset hash, different pack: {dataset_cache.DS}/{sha[:12]} on the share has md5 "
                            f"{theirs}, {name} here is {md5}; check both by hand (nothing was copied)")
        if not (dest / C.READY).is_file():             # renamed after its md5 check, then interrupted before READY
            if not (dest / name).is_file() or (dest / name).stat().st_size != local_pack.stat().st_size:
                raise SyncError(f"{dest} has no READY and its pack is missing or the wrong size; check it by hand")
            (dest / C.READY).write_text(C.now() + "\n", encoding="utf-8")
            print(f"[push] dataset pack {sha[:12]}: completed an interrupted upload (READY written)")
        else:
            print(f"[push] dataset pack {sha[:12]}: already on the share, not uploaded")
        return fields
    inc = ds / C.INCOMING / sha
    if inc.exists():                                   # an interrupted upload: left as is, a new folder instead
        inc = ds / C.INCOMING / f"{sha}.{time.strftime('%Y%m%d-%H%M%S')}"
        n = 1
        while inc.exists():
            inc, n = ds / C.INCOMING / f"{sha}.{time.strftime('%Y%m%d-%H%M%S')}_{n}", n + 1
    print(f"[push] dataset pack {sha[:12]}: uploading {name} ({local_pack.stat().st_size / 1e6:.1f} MB)", flush=True)
    C.copy_files(local_pack.parent, inc, [name] + ([name + ".md5"] if side else []), copier)
    if not side:
        (inc / (name + ".md5")).write_text(f"{md5}  {name}\n", encoding="utf-8")
    if C.md5_file(inc / name) != md5:
        raise SyncError(f"md5 of the pack copy in {inc} differs; left there, push again")
    inc.rename(dest)
    (dest / C.READY).write_text(C.now() + "\n", encoding="utf-8")   # last
    return fields


def repo_weights(exp_args) -> list:
    """exp_args with a relative --weights path that exists under the repo made absolute (the portal writes
    models/<id>/best.pt; make_bundle resolves relative paths against the current folder, as the CLI does)."""
    exp = [y for x in map(str, exp_args) for y in (x.split("=", 1) if x.startswith("--weights=") else [x])]
    if "--weights" in exp:
        i = exp.index("--weights") + 1
        if i < len(exp) and not Path(exp[i]).is_absolute() and (REPO / exp[i]).is_file():
            exp[i] = str(REPO / exp[i])
    return exp


def queue_train(job_id, dataset_dir, exp_args, share, max_hours, hold=False, check_origin=True, *, out=None,
                registry_root=None, expect_sha256=None, copier="auto", allow_dirty=False) -> Path:
    """One call from a registry train job to the GPU queue: training/make_job_bundle.make_bundle, then push (the
    same functions as `make_job_bundle.py` + `sync.py push`). docs/JOBS.md, "queue_train".
    - dataset_dir, exp_args: as make_job_bundle.py --dataset-dir ... -- <exp_args>; a relative --weights path is
      taken from the repo. out: the bundle folder (default <CryoAI-laptop>/job_bundles, as the CLI).
    - registry_root: the registry holding job_id (default the configured one). expect_sha256: the dataset_sha256 the
      registry recorded for this dataset (dataset_from_lv.py writes labels.dataset_sha256), when there is one.
    Refused with SyncError, nothing written to the share: the share missing (never created); a bad job id, or not a
    train job in the registry; the job already on the share (q/ or d/); the bundle refused (uncommitted code,
    no dataset.yaml, an unknown run_experiments flag, ...); the bundle's dataset_sha256 differing from expect_sha256;
    the commit not on origin/main (check_origin); the share's ds/<sha>/ holding another pack (push_pack).
    The checks that need no bundle run before it is made. -> q/<job> (READY written unless hold)."""
    import ids
    from training import make_job_bundle as MB
    share = Path(share)
    if not share.is_dir():                         # as push: never create a queue root
        raise SyncError(f"share {share} not found: a desktop share is \\\\host\\cj (two leading backslashes), "
                        f"and the queue root must already exist")
    if not ids.is_id("job", job_id) or len(job_id) > C.MAX_JOB_ID:
        raise SyncError(f"{job_id!r} is not a job id of <= {C.MAX_JOB_ID} characters (job_<YYYYMMDD>_<slug>)")
    rec = registry_job(job_id, registry_root)
    if rec is None or rec.get("type") != "train":
        raise SyncError(f"{job_id} is not a train job in the registry; create it with "
                        f"`registry.py job new --type train --where desktop ...`")
    if (share / C.Q / job_id).exists() or (share / C.D / job_id).exists():
        raise SyncError(f"{job_id} is already in the queue (q/ or d/); use a new job, or `retry` for d/")
    try:
        b = MB.make_bundle(Path(dataset_dir), repo_weights(exp_args), out, allow_dirty=allow_dirty)
    except SystemExit as e:                        # make_job_bundle refuses with SystemExit (its CLI contract)
        raise SyncError(f"bundle refused: {e.code}") from None
    if expect_sha256 and b["dataset_sha256"] != expect_sha256:
        raise SyncError(f"dataset sha mismatch: {Path(dataset_dir).name} hashes to {str(b['dataset_sha256'])[:12]}, "
                        f"the registry recorded {str(expect_sha256)[:12]}: the folder changed since it was recorded "
                        f"(nothing was sent; the bundle stays in {b['zip'].parent})")
    return push(job_id, b["zip"], share, copier, check_registry=False, check_origin=check_origin,
                max_hours=max_hours, hold=hold)


PREDICT_LOCK = "predict.lock"                    # <predict root>/predict.lock, held by jobs/predict_local.py
PREDICT_HOURS = 2.0                              # push-predict's default --max-hours (a 1000-tile set takes minutes)


def push_predict(ps, share, job_id=None, copier="auto", check_origin=True, max_hours=PREDICT_HOURS, hold=False):
    """Queue a prediction set for the desktop (docs/JOBS.md, "Predict jobs"; needs $CRYOEV_PREDICT_JOBS=1).
    - the registry job: `job_id`, else job_<today>_<ps id> (type predict, where desktop), made if new;
    - an images-only dataset pack (dataset_cache: images/ + MANIFEST, no labels/) and a thin zip
      <ps>__job_<stamp>.zip = <ps>/JOB.json (job_type predict, model, conf, max_det, imgsz, the pack), request.json,
      images.csv; both go up like a train job's (push, push_pack);
    - the set's status.json and registry record: queued, where desktop, job.
    Holds <predict root>/predict.lock (never while predict_local runs). The registry job is created only after the
    push succeeded; if that then fails, q/<job>/CANCEL is written (nothing is deleted) and SyncError raised.
    -> q/<job>. Refused: the flag off, a done or running set, a set already queued as a job, a missing image."""
    import predict_set as PS
    from data_utils.oslock import acquire_lock
    if not C.predict_enabled():
        raise SyncError(f"predict jobs are off: set {C.PREDICT_ENV}=1 (on this machine, and on the desktop runner "
                        f"before it can run them)")
    share = Path(share)
    if not share.is_dir():
        raise SyncError(f"share {share} not found: a desktop share is \\\\host\\cj (two leading backslashes)")
    reg = config.registry_root()
    try:
        ps_dir = PS.find(ps, registry_root=reg)
    except PS.PredictSetError as e:
        raise SyncError(str(e))
    lock = acquire_lock(ps_dir.parent / PREDICT_LOCK, 0)   # never while predict_local runs a set of this root
    if lock is None:
        raise SyncError(f"a local prediction run holds {ps_dir.parent / PREDICT_LOCK}; push when it has finished")
    try:
        return _push_predict_locked(ps_dir, reg, share, job_id, copier, check_origin, max_hours, hold)
    finally:
        lock.close()


def _push_predict_locked(ps_dir, reg, share, job_id, copier, check_origin, max_hours, hold):
    import tempfile
    import ids
    import predict_set as PS
    import registry
    me = actor.current(None, default=None)             # the new job's who + via; a bad env fails before the push
    try:
        req, st, rows = PS.read_request(ps_dir), PS.read_status(ps_dir), PS.read_images(ps_dir)
    except (OSError, ValueError) as e:
        raise SyncError(str(e))
    ps_id = req["ps"]
    state = st.get("state")
    if state == "done" or (ps_dir / C.PS_PRED).exists():
        raise SyncError(f"{ps_id} is already done")
    if state == "running" or (st.get("job") and state == "queued"):
        raise SyncError(f"{ps_id} is {state}" + (f" as {st['job']}" if st.get("job") else " on the laptop"))
    files = [ps_dir / r["file"] for r in rows]
    missing = [r["file"] for r in rows if not (ps_dir / r["file"]).is_file()]
    if not rows or missing:
        raise SyncError(f"{ps_id}: images missing from the set: {missing[:3] or 'none listed'}")
    commit = git("rev-parse", "HEAD").stdout.strip() or None
    if check_origin:
        why = commit_on_origin(commit)
        if why:
            raise SyncError(why)
    jid = job_id or ids.job_id(time.strftime("%Y%m%d"), ps_id)
    cur = registry_job(jid)
    if cur is None and job_id:
        raise SyncError(f"{job_id} is not in the registry (leave --job out to make job_<today>_{ps_id})")
    if cur is not None and cur.get("type") != "predict":
        raise SyncError(f"{jid} is a {cur.get('type')} job in the registry, not predict; pass another --job")
    with tempfile.TemporaryDirectory(prefix="cj_pp_") as td:
        td = Path(td)
        pack, sha, pack_md5, md5s, _ = dataset_cache.write_pack(td, ps_dir, sorted(files), fixed_time=True)
        zp = td / f"{ps_id}__job_{time.strftime('%Y%m%d-%H%M%S')}.zip"
        job = {"bundle_id": zp.stem, "dataset_name": ps_id, "job_type": "predict", "ps": ps_id,
               "model": req["model"], "conf": req.get("conf", 0.25), "max_det": req.get("max_det", 1000),
               "imgsz": req.get("imgsz"), "n_images": len(rows), "cryoEV_commit": commit, "run_experiments_args": [],
               "md5": md5s, "dataset_sha256": sha, "dataset_pack": pack.name, "dataset_pack_md5": pack_md5}
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr(f"{ps_id}/JOB.json", json.dumps(job, indent=1))
            for name in (PS.REQUEST, PS.IMAGES_CSV):
                z.write(ps_dir / name, f"{ps_id}/{name}")
        Path(str(zp) + ".md5").write_text(f"{C.md5_file(zp)}  {zp.name}\n", encoding="utf-8")
        # the registry job only once the push has succeeded: a refused or failed push leaves no registry line
        q = push(jid, zp, share, copier, check_registry=False, check_origin=False, max_hours=max_hours, hold=hold,
                 job_type="predict", extra={"ps": ps_id, "model": req["model"], "n_images": len(rows)})
    if cur is None:
        try:
            with tempfile.TemporaryDirectory(prefix="cj_pl_") as td:
                lst = Path(td) / "images.txt"
                lst.write_text("".join(r["image"] + "\n" for r in rows), encoding="utf-8")
                registry.new_job("predict", "desktop", ps_id, images=lst, root=reg, date=jid[4:12],
                                 note=f"prediction set {ps_id}", who=me["who"], via=me.get("via"))
        except Exception as e:                         # roll back without deleting: the runner skips a CANCELled job
            try:
                cancel(jid, share, reason=f"push-predict: the registry job was not created ({e})")
            except Exception as ce:                    # never hides the first error: both are reported
                raise SyncError(f"{jid} was queued but its registry job was not created ({e}), and writing its "
                                f"CANCEL failed too ({type(ce).__name__}: {ce}): cancel it by hand "
                                f"(`sync.py cancel {jid}`) before the runner picks it") from e
            raise SyncError(f"{jid} was queued but its registry job was not created ({e}); CANCEL written, so the "
                            f"runner skips it. Fix the registry, then push again with a new job (or --job)") from e
    PS.write_status(ps_dir, state="queued", n_done=0, n_total=len(rows), where="desktop", job=jid, error=None,
                    pid=None, waiting=None)
    try:
        PS.update_record(ps_id, reg, recorded_by="jobs/sync.py push-predict", status="queued", where="desktop",
                         job=jid)
    except Exception as e:
        print(f"[push-predict] registry record of {ps_id} not updated: {e}", file=sys.stderr)
    return q


EVAL_LOCK = Path("jobs") / "eval.lock"           # <CryoAI-laptop>/jobs/eval.lock: one push-eval at a time
EVAL_HOURS = 1.0                                 # push-eval's default --max-hours (a 39-image test split: minutes)
EVAL_DIR = "eval"                                # <CryoAI-laptop>/eval/<job>/: a pulled eval job's results


def push_eval(share, bundle=None, weights=None, dataset=None, job_id=None, model=None, copier="auto",
              check_origin=True, max_hours=EVAL_HOURS, hold=False, out=None, allow_dirty=False):
    """Queue an eval job for the desktop (docs/JOBS.md, "Eval jobs"; needs $CRYOEV_EVAL_JOBS=1). Either `bundle` (a
    zip from make_job_bundle.py --eval) or `weights` + `dataset` (a registry dataset id or folder: the bundle is made
    here, into `out`, default <CryoAI-laptop>/job_bundles, reusing the dataset's pack).
    - the registry job: `job_id`, else job_<today>_eval_<model> (type eval, where desktop; its list = the test
      images), made only after the push succeeded; if that then fails, q/<job>/CANCEL is written (nothing deleted);
    - push as a train job's (the dataset pack once per queue, then the thin zip), job.json type eval.
    Holds <CryoAI-laptop>/jobs/eval.lock for the whole push. -> q/<job>. Refused: the flag off, no share, neither or
    both of bundle and weights+dataset, a bundle that is not an eval bundle, the job id not an eval job."""
    from data_utils.oslock import acquire_lock
    if not C.eval_enabled():
        raise SyncError(f"eval jobs are off: set {C.EVAL_ENV}=1 (on this machine, and on the desktop runner "
                        f"before it can run them)")
    share = Path(share)
    if not share.is_dir():
        raise SyncError(f"share {share} not found: a desktop share is \\\\host\\cj (two leading backslashes)")
    if (bundle is None) == (weights is None or dataset is None):
        raise SyncError("give either --bundle (make_job_bundle.py --eval) or --weights and --dataset")
    lock_path = config.laptop_root() / EVAL_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(lock_path, 0)
    if lock is None:
        raise SyncError(f"another push-eval holds {lock_path}; push when it has finished")
    try:
        return _push_eval_locked(share, bundle, weights, dataset, job_id, model, copier, check_origin, max_hours,
                                 hold, out, allow_dirty)
    finally:
        lock.close()


def _push_eval_locked(share, bundle, weights, dataset, job_id, model, copier, check_origin, max_hours, hold, out,
                      allow_dirty):
    import tempfile
    import ids
    import registry
    me = actor.current(None, default=None)             # the new job's who + via; a bad env fails before the push
    reg = config.registry_root()
    if bundle is None:
        from training import make_job_bundle as MB
        try:
            ddir, dname = MB.resolve_dataset(dataset, reg)
            b = MB.make_eval_bundle(ddir, Path(weights), out, dname, model, allow_dirty=allow_dirty)
        except SystemExit as e:                        # make_job_bundle refuses with SystemExit (its CLI contract)
            raise SyncError(f"bundle refused: {e.code}") from None
        bundle = b["zip"]
    bundle = Path(bundle)
    try:
        with zipfile.ZipFile(bundle) as z:
            names = [n for n in z.namelist() if n.count("/") == 1 and n.endswith("/JOB.json")]
            job = json.loads(z.read(names[0])) if len(names) == 1 else {}
    except (OSError, zipfile.BadZipFile, ValueError) as e:
        raise SyncError(f"can't read {bundle}: {e}")
    if C.job_kind(job) != "eval":
        raise SyncError(f"{bundle.name} is not an eval bundle (JOB.json job_type {job.get('job_type')!r}); make it "
                        f"with make_job_bundle.py --eval")
    if model and job.get("model") != model:
        raise SyncError(f"{bundle.name} scores model {job.get('model')!r}, not {model!r}")
    jid = job_id or ids.job_id(time.strftime("%Y%m%d"), f"eval_{job.get('model')}")
    if len(jid) > C.MAX_JOB_ID:
        raise SyncError(f"{jid} is longer than {C.MAX_JOB_ID} characters; pass a shorter --job")
    cur = registry_job(jid)
    if cur is None and job_id:
        raise SyncError(f"{job_id} is not in the registry (leave --job out to make job_<today>_eval_<model>)")
    if cur is not None and cur.get("type") != "eval":
        raise SyncError(f"{jid} is a {cur.get('type')} job in the registry, not eval; pass another --job")
    q = push(jid, bundle, share, copier, check_registry=False, check_origin=check_origin, max_hours=max_hours,
             hold=hold, job_type="eval", extra={"model": job.get("model"), "split": job.get("split"),
                                                "n_images": job.get("n_images")})
    if cur is None:                                    # the registry job only once the push has succeeded
        try:
            with tempfile.TemporaryDirectory(prefix="cj_el_") as td:
                lst = Path(td) / "images.txt"
                files = sorted(Path(k).name for k in job.get("md5") or {}   # the scored images (the pack's md5s)
                               if k.startswith(f"images/{job.get('split') or 'test'}/"))
                lst.write_text("".join(f + "\n" for f in files), encoding="utf-8")
                registry.new_job("eval", "desktop", jid.split("_", 2)[2], images=lst, root=reg, date=jid[4:12],
                                 note=f"eval {job.get('model')} on {job.get('dataset_name')} "
                                      f"{job.get('split')}", who=me["who"], via=me.get("via"))
        except Exception as e:                         # roll back without deleting: the runner skips a CANCELled job
            try:
                cancel(jid, share, reason=f"push-eval: the registry job was not created ({e})")
            except Exception as ce:
                raise SyncError(f"{jid} was queued but its registry job was not created ({e}), and writing its "
                                f"CANCEL failed too ({type(ce).__name__}: {ce}): cancel it by hand "
                                f"(`sync.py cancel {jid}`) before the runner picks it") from e
            raise SyncError(f"{jid} was queued but its registry job was not created ({e}); CANCEL written, so the "
                            f"runner skips it. Fix the registry, then push again with a new job (or --job)") from e
    return q


def release(job_id, share):
    """Write READY for a job pushed with --hold."""
    qd = Path(share) / C.Q / job_id
    if not (qd / C.JOB_FILE).is_file():
        raise SyncError(f"{job_id} is not in {qd.parent}")
    if (qd / C.READY).exists():
        raise SyncError(f"{job_id} is already released")
    (qd / C.READY).write_text(C.now() + "\n", encoding="utf-8")
    return qd


def stop(share, who=None):
    """The watcher finishes its current job, then pauses (it stays running). who: actor.who() (flag > env > pujan)."""
    who = actor.who(who, default=DEFAULT_WHO)
    p = Path(share) / C.STOP
    p.write_text(f"{C.now()} {who} {platform.node()}\n", encoding="utf-8")
    return p


def start(share):
    """Lift a stop: STOP is renamed into .retired/ (nothing on the share is deleted)."""
    share = Path(share)
    if not (share / C.STOP).exists():
        print("[start] no STOP file: the watcher isn't paused by a stop")
        return None
    dst = share / C.RETIRED / f"STOP_{time.strftime('%Y%m%d-%H%M%S')}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    (share / C.STOP).rename(dst)
    return dst


def watch_state(share):
    """watch.json plus the heartbeat age in seconds, or None."""
    try:
        w = C.read_json_shared(Path(share) / C.WATCH_JSON)
    except (OSError, ValueError):
        return None
    from datetime import datetime, timezone
    try:
        if w.get("heartbeat_utc"):                     # both machines' clocks compared in UTC
            age = datetime.now(timezone.utc) - datetime.fromisoformat(w["heartbeat_utc"])
        else:
            age = datetime.now() - datetime.fromisoformat(w["heartbeat"])
        w["age_s"] = round(age.total_seconds())
    except (KeyError, ValueError, TypeError):
        w["age_s"] = None
    return w


def cancel(job_id, share, who=None, reason=None):
    """Ask the runner to stop a queued or running job: writes q/<job>/CANCEL (a new file; nothing is deleted).
    A running job is stopped within ~10 s (its partial run is kept); a queued one is never started.
    who: actor.who() (flag > $CRYOEV_FOR > pujan); the CANCEL file is not stamped with via."""
    who = actor.who(who, default=DEFAULT_WHO)
    qd = Path(share) / C.Q / job_id
    if not (qd / C.JOB_FILE).is_file():
        raise SyncError(f"{job_id} is not in {qd.parent}")
    if (Path(share) / C.D / job_id / C.DONE).is_file():
        raise SyncError(f"{job_id} has already finished (d/{job_id}/DONE.json)")
    C.write_json(qd / C.CANCEL, {"job": job_id, "who": who, "when": C.now(), "from": platform.node(),
                                 "reason": reason or "cancelled from the laptop"})
    return qd / C.CANCEL


# ---------------------------------------------------------------- pull
def ledger_path():
    return config.laptop_root() / LEDGER


def ledger(path):
    out = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(line)
                out.add((o["job"], o["done_sha256"]))
            except (ValueError, KeyError):
                continue
    return out


def guard(path):
    problem = protected_problem(path)
    if problem:
        raise SyncError(f"refused: {problem}")


def left_on_share(rd, all_weights=False) -> list:
    """Files of run folder rd that pull doesn't copy: weights/epoch*.pt (Ultralytics save_period checkpoints,
    ~209 MB each, most of a run's size) unless all_weights. best.pt and last.pt are always copied."""
    w = rd / "weights"
    if all_weights or not w.is_dir():
        return []
    return sorted(p for p in w.glob(EPOCH_CKPT) if p.is_file())


def copy_run(rd, part, copier, exclude=()):
    """C.copy_tree(rd, part) without the files in `exclude` (full paths under rd); same copy rules."""
    if not exclude:
        C.copy_tree(rd, part, copier)
        return
    if C.use_robocopy(copier, rd, part):
        argv = C.robocopy_argv(rd, part) + ["/XF", *(str(p) for p in exclude)]   # /XF takes full paths
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode > C.ROBOCOPY_OK_MAX:
            raise OSError(f"robocopy {rd} -> {part} failed (exit {r.returncode}): {r.stdout[-400:]}")
        return
    skip = {os.path.normcase(os.path.abspath(p)) for p in exclude}
    shutil.copytree(rd, part, dirs_exist_ok=True, ignore=lambda d, names: [
        n for n in names if os.path.normcase(os.path.abspath(os.path.join(d, n))) in skip])


def copy_runs(jd, dataset, copier, all_weights=False):
    """d/<job>/runs/<run>/ -> <CryoAI>/training outputs/<dataset>/runs/<run>/ (new only), without
    weights/epoch*.pt unless all_weights. -> (copied, skipped, not_pulled {files, bytes, paths})."""
    src = jd / "runs"
    not_pulled = {"files": 0, "bytes": 0, "paths": []}
    if not src.is_dir():
        return [], [], not_pulled
    ds = config.cryoai_root() / "training outputs" / dataset
    copied, skipped = [], []
    for rd in sorted(p for p in src.iterdir() if p.is_dir()):
        dest = ds / "runs" / rd.name
        if dest.exists():
            skipped.append(rd.name)                     # never overwrite an existing run folder
            continue
        part = ds / PULL_TMP / rd.name
        guard(part)
        guard(dest)
        part.parent.mkdir(parents=True, exist_ok=True)
        leave = left_on_share(rd, all_weights)
        copy_run(rd, part, copier, leave)               # raises on failure; the partial copy stays in .pull_tmp
        dest.parent.mkdir(parents=True, exist_ok=True)
        part.rename(dest)
        copied.append(rd.name)
        not_pulled["files"] += len(leave)
        not_pulled["bytes"] += sum(p.stat().st_size for p in leave)
        not_pulled["paths"] += [p.relative_to(src).as_posix() for p in leave]
    return copied, skipped, not_pulled


PULL_TMP_PRED = ".pull_tmp_p"                    # <ps>/.pull_tmp_p while a predict job's p/ is copied


def pull_predict(jd, done, copier="auto"):
    """A done predict job's results -> its prediction set (<CryoAI-laptop>/predict/<ps>/): d/<job>/p/ to <ps>/p/
    (staged in <ps>/.pull_tmp_p, then renamed; an existing p/ is kept, never overwritten), and d/<job>/predict.log
    appended to <ps>/predict.log. A blocked or cancelled job copies nothing (its partial p/ stays on the share).
    -> (copied, skipped, not_pulled) like copy_runs."""
    import predict_set as PS
    none = {"files": 0, "bytes": 0, "paths": []}
    try:
        ps_dir = PS.find(str(done.get("ps")), registry_root=config.registry_root())
    except PS.PredictSetError as e:
        raise SyncError(str(e))
    if done.get("status") != "done":
        return [], [], none
    dest, part = ps_dir / C.PS_PRED, ps_dir / PULL_TMP_PRED
    if dest.exists():
        return [], [C.PS_PRED], none
    guard(part)
    guard(dest)
    C.copy_tree(jd / C.PS_PRED, part, copier)          # raises on failure; the partial copy stays in .pull_tmp_p
    part.rename(dest)
    if (jd / C.PREDICT_LOG).is_file():
        with open(ps_dir / C.PREDICT_LOG, "ab") as f:
            f.write(f"\n=== {C.now()} pulled from {jd} ({done.get('machine')})\n".encode("utf-8")
                    + C.read_shared(jd / C.PREDICT_LOG))
    return [C.PS_PRED], [], none


def pull_eval(jd, done, copier="auto"):
    """A done eval job's results -> <CryoAI-laptop>/eval/<job>/: d/<job>/e/ (metrics.json, val/) and eval.log, plus
    testmap_<model>.json (= metrics.json, the name analysis/compare_eval_runs.py --eval-dir reads). Staged in
    eval/.pull_tmp_<job>, then renamed; an existing eval/<job>/ is kept, never overwritten. Never touches training
    outputs and never runs the importer (pull). A blocked or cancelled job copies nothing (its e/ stays on the
    share). -> (copied, skipped, not_pulled) like copy_runs."""
    none = {"files": 0, "bytes": 0, "paths": []}
    if done.get("status") != "done":
        return [], [], none
    root = config.laptop_root() / EVAL_DIR
    dest, part = root / jd.name, root / f".pull_tmp_{jd.name}"
    if dest.exists():
        return [], [f"{EVAL_DIR}/{jd.name}"], none
    guard(part)
    guard(dest)
    if not (jd / C.EVAL_OUT / C.METRICS).is_file():
        raise SyncError(f"{jd.name}: done but d/{jd.name}/{C.EVAL_OUT}/{C.METRICS} is missing")
    C.copy_tree(jd / C.EVAL_OUT, part, copier)         # raises on failure; the partial copy stays in .pull_tmp_<job>
    if (jd / C.EVAL_LOG).is_file():
        (part / C.EVAL_LOG).write_bytes(C.read_shared(jd / C.EVAL_LOG))
    model = str(done.get("model") or "model")
    (part / f"testmap_{model}.json").write_bytes((part / C.METRICS).read_bytes())
    part.rename(dest)
    return [f"{EVAL_DIR}/{jd.name}"], [], none


def predict_pulled(job_id, done, target, reason):
    """After a predict job's pull: its set's status.json (done, or failed with the reason) and registry record
    (done | cancelled | failed). Raises SyncError."""
    import predict_set as PS
    reg = config.registry_root()
    try:
        ps_dir = PS.find(str(done.get("ps")), registry_root=reg)
        fields = {"n_total": done["n_total"]} if done.get("n_total") is not None else {}
        PS.write_status(ps_dir, state="done" if target == "done" else "failed", n_done=done.get("n_done") or 0,
                        where="desktop", job=job_id, error=None if target == "done" else (reason or target), **fields)
        PS.update_record(done["ps"], reg, recorded_by="jobs/sync.py pull", where="desktop", job=job_id,
                         status={"done": "done", "cancelled": "cancelled"}.get(target, "failed"))
    except (PS.PredictSetError, OSError, ValueError) as e:
        raise SyncError(f"prediction set {done.get('ps')} not updated: {e}")


def registry_cmd(*args):
    """registry.py as a child process. It inherits this process's env, so it reads $CRYOEV_VIA itself and stamps
    `via` once (nothing here passes it); the person always goes as an explicit --who."""
    r = subprocess.run([sys.executable, str(REGISTRY_PY), *args], cwd=REPO, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip())
    if r.returncode:
        raise SyncError(f"registry.py {' '.join(args[:2])} failed (exit {r.returncode})")
    return r.stdout


def pull(share, who=None, copier="auto", all_weights=False):
    """-> list of {job, status, copied, skipped, not_pulled}. Raises SyncError after logging what succeeded.
    weights/epoch*.pt stay on the share unless all_weights; not_pulled = {files, bytes, paths} (also in the ledger).
    who: the set-status person, actor.who() (flag > $CRYOEV_FOR > pujan)."""
    who = actor.current(who, default=DEFAULT_WHO)["who"]   # also checks $CRYOEV_VIA before anything is copied
    share = Path(share)
    dd = share / C.D
    led_path = ledger_path()
    seen = ledger(led_path)
    todo = []
    for jd in sorted(dd.iterdir()) if dd.is_dir() else []:
        if jd.name.startswith(".") or not jd.is_dir() or not (jd / C.DONE).is_file():
            continue
        raw = (jd / C.DONE).read_bytes()
        key = (jd.name, hashlib.sha256(raw).hexdigest())
        if key in seen:
            continue
        todo.append((jd, json.loads(raw), key[1]))
    if not todo:
        print("[pull] nothing new")
        return []
    pulled, errors = [], []
    for jd, done, sha in todo:
        kind = C.job_kind(done)
        if kind not in C.JOB_TYPES and done.get("status") == "done":   # a newer runner's type: never as training
            errors.append(f"{jd.name}: unknown job_type {kind!r}; not pulled (update this checkout)")
            continue
        try:
            if kind not in C.JOB_TYPES:                # refused by the runner (blocked): its status only, no files
                copied, skipped, left = [], [], {"files": 0, "bytes": 0, "paths": []}
            elif kind == "predict":                      # results go to the prediction set, not training outputs
                copied, skipped, left = pull_predict(jd, done, copier)
            elif kind == "eval":                       # results go to <CryoAI-laptop>/eval/<job>/; no run folders
                copied, skipped, left = pull_eval(jd, done, copier)
            else:
                copied, skipped, left = (copy_runs(jd, done["dataset"], copier, all_weights) if done.get("dataset")
                                         else ([], [], {"files": 0, "bytes": 0, "paths": []}))
            pulled.append((jd.name, done, sha, copied, skipped, left))
            print(f"[pull] {jd.name}: {done['status']}; copied {copied or 'no runs'}"
                  + (f"; kept existing {skipped}" if skipped else "")
                  + (f"; left {left['files']} epoch checkpoint(s) ({left['bytes'] / 1e6:.0f} MB) on the share "
                     f"(--all-weights copies them)" if left["files"] else ""))
        except (OSError, SyncError) as e:
            errors.append(f"{jd.name}: copy failed: {e}")
    if any(C.job_kind(p[1]) == "train" for p in pulled):   # the importer reads run folders: train jobs only
        registry_cmd("import")
    out = []
    for job_id, done, sha, copied, skipped, left in pulled:
        target = done["status"] if done["status"] in ("done", "cancelled") else "blocked"
        reason = done.get("reason") or ("blocked on the runner" if target == "blocked" else None)
        kind = C.job_kind(done)
        predict = kind == "predict"
        cur = registry_job(job_id)
        if cur is None:
            errors.append(f"{job_id}: not in the registry; status not set")
        elif cur.get("status") != target or (target == "blocked" and cur.get("blocked_reason") != reason):
            args = ["set-status", job_id, target, "--who", who]
            if target == "blocked":
                args += ["--reason", reason[:300]]
            elif target == "cancelled":
                args += ["--note", (reason or "cancelled")[:300]]
            elif predict:
                args += ["--note", f"{done.get('ps')}: {done.get('n_done')}/{done.get('n_total')} images "
                                   f"predicted ({done.get('machine')})"]
            elif kind == "eval":
                m = (done.get("metrics") or {}).get("mask_map50_95")
                args += ["--note", f"eval {done.get('model')} on {done.get('dataset')} {done.get('split')}: mask "
                                   f"mAP50-95 {m:.3f} ({done.get('machine')})" if isinstance(m, (int, float)) else
                         f"eval {done.get('model')} on {done.get('dataset')} ({done.get('machine')})"]
            else:
                args += ["--note", f"run {done.get('run')} ({done.get('machine')}"
                                   + (", --allow-big" if done.get("allow_big") else "") + ")"]
            try:
                registry_cmd(*args)
            except SyncError as e:
                errors.append(f"{job_id}: {e}")
                continue
        if predict:
            try:
                predict_pulled(job_id, done, target, reason)
            except SyncError as e:
                errors.append(f"{job_id}: {e}")
                continue
        guard(led_path)
        led_path.parent.mkdir(parents=True, exist_ok=True)
        with open(led_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"job": job_id, "done_sha256": sha, "status": target, "reason": reason,
                                "copied": copied, "skipped": skipped, "all_weights": all_weights,
                                "not_pulled": left, "pulled": C.now()}) + "\n")
        out.append({"job": job_id, "status": target, "copied": copied, "skipped": skipped, "not_pulled": left})
    if errors:
        raise SyncError("; ".join(errors))
    return out


# ---------------------------------------------------------------- status / retry
def queue_rows(share) -> list:
    """One dict per job in q/: job, state (done / blocked / cancelled / running / queued / held), reason, pulled,
    cancel_requested, done (DONE.json or None). Reads only small files, with the shared reader."""
    share = Path(share)
    seen = {j for j, _ in ledger(ledger_path())} if config.laptop_root(required=False) else set()
    rows = []
    for d in sorted((share / C.Q).iterdir()) if (share / C.Q).is_dir() else []:
        if d.name.startswith(".") or not d.is_dir():
            continue
        row = {"job": d.name, "state": None, "reason": None, "pulled": d.name in seen, "done": None,
               "cancel_requested": (d / C.CANCEL).is_file()}
        if (share / C.D / d.name / C.DONE).is_file():
            try:
                row["done"] = C.read_json_shared(share / C.D / d.name / C.DONE)
                row["state"], row["reason"] = row["done"].get("status"), row["done"].get("reason")
            except (OSError, ValueError) as e:
                row["state"], row["reason"] = "unknown", f"DONE.json unreadable: {e}"
        elif (share / C.D / f".tmp_{d.name}").is_dir():
            row["state"] = "running"
        elif (d / C.READY).is_file():
            row["state"] = "queued"
        elif (d / C.JOB_FILE).is_file():
            row["state"] = "held"
        else:
            row["state"] = "incomplete"
        rows.append(row)
    return rows


def status(share):
    share = Path(share)
    rows = []
    for r in queue_rows(share):
        what = {"running": "running (or stopped mid-job)", "held": "held (`release` starts it)",
                "incomplete": "no READY (push incomplete)"}.get(r["state"], r["state"])
        if r["done"] is not None:
            what += (f" ({r['reason']})" if r["reason"] else "") + (" pulled" if r["pulled"] else " not pulled")
        elif r["cancel_requested"]:
            what += " (cancel requested)"
        rows.append((r["job"], what))
    for name, what in rows:
        print(f"{name:50} {what}")
    cache = dataset_cache.cache_entries(share)
    if cache:
        print(f"dataset cache ({dataset_cache.DS}/): {len(cache)} pack(s), "
              f"{sum(e['bytes'] for e in cache) / 1e6:.1f} MB")
        for e in cache:
            print(f"  {e['sha'][:12]}  {e['bytes'] / 1e6:9.1f} MB" + ("" if e["ready"] else "  (no READY)")
                  + (f"  needed by {len(e['refs'])} job(s) not done/cancelled: {', '.join(e['refs'][:3])}"
                     if e["refs"] else "  no job in q/ still needs it"))
    w = watch_state(share)
    if w:
        print(f"watcher: {w.get('state')}" + (f" ({w.get('detail')})" if w.get("detail") else "")
              + (f", job {w['job']}" if w.get("job") else "") + f", heartbeat {w.get('age_s')} s ago"
              + ("  ** no heartbeat for 5 min: is the watcher running? **"
                 if w.get("age_s") is not None and w["age_s"] > 300 else ""))
    elif (share / C.STOP).exists():
        print("watcher: no watch.json; STOP is set")
    return rows


def retry(job_id, share, force=False):
    """Move d/<job> aside so the runner picks the job again. Refuses a bad id (e.g. a live .tmp_ folder)
    and a done job that was never pulled (its run would never be imported) unless force."""
    import ids
    share = Path(share)
    if not ids.is_id("job", job_id):
        raise SyncError(f"{job_id!r} is not a job id")
    src = share / C.D / job_id
    if not src.is_dir():
        raise SyncError(f"no d/{job_id} to retry")
    done = src / C.DONE
    if not force and done.is_file() and C.read_json(done).get("status") == "done":
        pulled = {j for j, _ in ledger(ledger_path())} if config.laptop_root(required=False) else set()
        if job_id not in pulled:
            raise SyncError(f"{job_id} is done but not pulled yet; pull it first (or --force)")
    dst = share / C.D / C.RETIRED / f"{job_id}_{time.strftime('%Y%m%d-%H%M%S')}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    ready = share / C.Q / job_id / C.READY
    if ready.exists():                                 # fresh mtime: a retried job goes to the back of the queue
        ready.write_text(C.now() + "\n", encoding="utf-8")
    print(f"[retry] d/{job_id} -> d/{C.RETIRED}/{dst.name}; the runner will pick {job_id} again")
    return dst


# ---------------------------------------------------------------- local (the laptop trains its own job)
def local(job_id, bundle, queue=LOCAL_QUEUE, allow_big=False, who=None, runner_argv=None, max_hours=None,
          all_weights=False):
    """push to a local queue folder, run jobs/run_job.py on it for this job, pull. -> pull() output ([] when
    the runner refused to start). The same code paths as the desktop; only the folder is local."""
    from run_job import is_network_path
    queue = Path(queue)
    who = actor.current(who, default=DEFAULT_WHO)["who"]  # before anything runs: a bad CRYOEV_FOR / _VIA fails here
    if not max_hours:
        raise SyncError("local needs --max-hours (the runner stops the trainer then; the partial run is kept)")
    if is_network_path(queue):
        raise SyncError(f"{queue} is a network folder; `local` uses a folder on this machine (default {LOCAL_QUEUE})")
    (queue / C.Q).mkdir(parents=True, exist_ok=True)
    push(job_id, bundle, queue, copier="shutil", check_origin=False)   # same machine: no GitHub needed
    argv = (runner_argv or [sys.executable, str(HERE / "run_job.py")]) + ["--queue-dir", str(queue), "--job", job_id]
    if allow_big:
        argv.append("--allow-big")
    if max_hours:
        argv += ["--max-hours", str(max_hours)]
    rc = subprocess.run(argv, cwd=REPO).returncode
    if rc in (2, 3):
        raise SyncError(f"run_job.py exit {rc} ({'refused at start' if rc == 2 else 'another runner holds the lock'});"
                        f" {job_id} stays queued in {queue}: fix it, then `run_job.py --queue-dir {queue}` and `pull`")
    return pull(queue, who, copier="shutil", all_weights=all_weights)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("local")
    lp.add_argument("job_id")
    lp.add_argument("--bundle", required=True, type=Path, help="zip from training/make_job_bundle.py")
    lp.add_argument("--queue", type=Path, default=LOCAL_QUEUE, help=f"local queue folder (default {LOCAL_QUEUE})")
    lp.add_argument("--allow-big", action="store_true", help="y8m/l/x or imgsz > 640 on a small GPU (needs --max-hours)")
    lp.add_argument("--max-hours", type=float, help="the runner stops the trainer after this many hours")
    lp.add_argument("--who", default=None, help="people.json id for the set-status lines (default: $CRYOEV_FOR, "
                    "else pujan)")
    lp.add_argument("--all-weights", action="store_true", help="also copy weights/epoch*.pt when pulling")
    pp = sub.add_parser("push-predict", help=f"queue a prediction set for the desktop (needs {C.PREDICT_ENV}=1)")
    pp.add_argument("--ps", required=True, help="prediction set id or folder")
    pp.add_argument("--share", required=True, type=Path)
    pp.add_argument("--job", help="an existing registry predict job (default: job_<today>_<ps id>, made if new)")
    pp.add_argument("--copier", choices=["auto", "robocopy", "shutil"], default="auto")
    pp.add_argument("--max-hours", type=float, default=PREDICT_HOURS)
    pp.add_argument("--hold", action="store_true")
    ep = sub.add_parser("push-eval", help=f"queue an eval job for the desktop (needs {C.EVAL_ENV}=1)")
    ep.add_argument("--share", required=True, type=Path)
    ep.add_argument("--bundle", type=Path, help="a zip from make_job_bundle.py --eval")
    ep.add_argument("--weights", type=Path, help="or: the model's .pt file (with --dataset; the bundle is made here)")
    ep.add_argument("--dataset", help="with --weights: a registry dataset id or a dataset folder")
    ep.add_argument("--model", help="the model name in metrics.json (default: models/<id>/ or the run folder)")
    ep.add_argument("--job", help="an existing registry eval job (default: job_<today>_eval_<model>, made if new)")
    ep.add_argument("--copier", choices=["auto", "robocopy", "shutil"], default="auto")
    ep.add_argument("--max-hours", type=float, default=EVAL_HOURS)
    ep.add_argument("--hold", action="store_true")
    for name in ("push", "push-pack", "pull", "status", "retry", "release", "stop", "start", "cancel"):
        p = sub.add_parser(name)
        if name in ("push", "retry", "release", "cancel"):
            p.add_argument("job_id")
        if name == "cancel":
            p.add_argument("--reason", help="why (shown on the Live tab and in the registry)")
            p.add_argument("--who", default=None, help="default: $CRYOEV_FOR, else pujan")
        p.add_argument("--share", required=True, type=Path, help="queue root on the desktop, e.g. \\\\desktop\\cj")
        if name in ("push", "push-pack", "pull"):
            p.add_argument("--copier", choices=["auto", "robocopy", "shutil"], default="auto")
        if name in ("push", "push-pack"):
            p.add_argument("--bundle", required=True, type=Path, help="zip from training/make_job_bundle.py")
        if name == "push":
            p.add_argument("--max-hours", type=float, help="the runner stops the trainer after this many hours "
                           "(default on the watcher: 48)")
            p.add_argument("--hold", action="store_true", help="queue without READY; `release` starts it")
        if name in ("pull", "stop"):
            p.add_argument("--who", default=None, help="people.json id for the set-status lines (default: "
                           "$CRYOEV_FOR, else pujan)")
        if name == "pull":
            p.add_argument("--all-weights", action="store_true", help="also copy runs/<run>/weights/epoch*.pt "
                           "(~209 MB each; by default they stay on the share)")
        if name == "retry":
            p.add_argument("--force", action="store_true", help="also retry a done job that was never pulled")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "local":
            local(a.job_id, a.bundle, a.queue, a.allow_big, a.who, max_hours=a.max_hours, all_weights=a.all_weights)
        elif a.cmd == "push":
            q = push(a.job_id, a.bundle, a.share, a.copier, max_hours=a.max_hours, hold=a.hold)
            print(f"[push] {'held (run `release` to start it)' if a.hold else 'queued'}: {q}")
        elif a.cmd == "push-predict":
            q = push_predict(a.ps, a.share, a.job, a.copier, max_hours=a.max_hours, hold=a.hold)
            print(f"[push-predict] {'held' if a.hold else 'queued'}: {q}")
        elif a.cmd == "push-eval":
            q = push_eval(a.share, a.bundle, a.weights, a.dataset, a.job, a.model, a.copier, max_hours=a.max_hours,
                          hold=a.hold)
            print(f"[push-eval] {'held' if a.hold else 'queued'}: {q}")
        elif a.cmd == "push-pack":
            print(f"[push-pack] {push_bundle_pack(a.bundle, a.share, a.copier)}")
        elif a.cmd == "cancel":
            print(f"[cancel] wrote {cancel(a.job_id, a.share, a.who, a.reason)}: the runner stops it within ~10 s "
                  f"(a queued job is never started); then `pull` records it")
        elif a.cmd == "release":
            print(f"[release] {release(a.job_id, a.share)} is ready")
        elif a.cmd == "stop":
            print(f"[stop] wrote {stop(a.share, a.who)}: the watcher finishes its current job, then pauses")
        elif a.cmd == "start":
            dst = start(a.share)
            if dst:
                print(f"[start] STOP moved to {dst}: the watcher resumes within a minute")
        elif a.cmd == "pull":
            pull(a.share, a.who, a.copier, a.all_weights)
        elif a.cmd == "status":
            status(a.share)
        else:
            retry(a.job_id, a.share, a.force)
    except (SyncError, config.ConfigError, OSError, actor.ActorError) as e:
        print(f"[{a.cmd}] {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
