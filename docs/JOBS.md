# Training jobs: laptop queue → desktop runner

The laptop is home base: it owns the registry, makes bundles, pushes them and pulls results.
The A6000 desktop is only sometimes on. It runs one queued job when someone starts `jobs/run_job.py` by hand, and **never writes the registry**.
Transport is a folder on the desktop, shared over Tailscale (SMB).
Tracking rules: [TRACKING_CONTRACT.md](TRACKING_CONTRACT.md). Producers write owner files (`run.json`, `DONE.json`); the laptop's importer reads them.

## Commands
Laptop (PowerShell, in the repo):
```powershell
# 1. A train job in the registry (job/1; the list is the dataset's images)
python registry/registry.py job new --type train --where desktop --name v6_y8m --images <list.txt> --who pujan
# 2. A bundle: zip + _README.md + .md5 sidecar; JOB.json has bundle_id and the commit
python training/make_job_bundle.py --dataset-dir <prepared dataset> -- --single-run --model-size m --batch 8 --imgsz 1440 --workers 4
# 3. Queue it on the desktop (READY is written last)
python jobs/sync.py push job_20261102_v6_y8m --bundle <zip> --share \\<desktop>\cj
python jobs/sync.py status --share \\<desktop>\cj
# 5. Bring results home: run folders -> <CryoAI>/training outputs/<dataset>/runs/, importer, job done|blocked
python jobs/sync.py pull --share \\<desktop>\cj
```
Desktop (in the repo, at the bundle's commit):
```powershell
# 4. Run the oldest ready job; --wait 600 waits for another runner's lock instead of exiting
python jobs/run_job.py --queue-dir C:\cj
```
- **Runner checks.** It checks the zip md5 (job.json and the `.md5` sidecar), that the repo commit equals JOB.json's `cryoEV_commit`, Ultralytics 8.4.x, and that the longest unzipped path is under 240 characters.
  It also checks that no other process is on the GPU: any python process, or one using ≥ 256 MB. Graphics apps listed with no memory are ignored.
  It needs free disk ≥ 10 GB + 3× the zip, and keeps the machine awake while the job runs.
- A failed check or run ends **blocked** with a reason. A job is **done** only when the run's `run.json` says `finished` **and** the trainer exited 0 (a later step can fail after `run.json` is written).
- The GPU-busy check runs before the old `work\<job>\` is cleared, so an orphaned trainer's files are never deleted under it. If a final rename fails (antivirus, an open Explorer window), it is retried; if it still fails, the runner exits non-zero and the results stay in `d\.tmp_<job>\`.
- One ntfy push at the end, only if `CRYOEV_NTFY_TOPIC` is set.
- **Exit codes.** `run_job.py`: 0 = done or nothing ready, 1 = blocked, 2 = refused at start (queue untouched), 3 = the lock is held. `sync.py`: 2 = refused or failed.
- **Smoke test.** `jobs/tests/test_smoke.py` (real GPU, y8n at 320). The fixture is `<CryoAI-laptop>\fixtures\smoke\` (`tiny\` + `yolov8n-seg.pt`):
  `set CRYOEV_JOBS_SMOKE_DATA=...\fixtures\smoke\tiny` and `set CRYOEV_JOBS_SMOKE_WEIGHTS=...\fixtures\smoke\yolov8n-seg.pt`.

## Training on the laptop
When the desktop isn't available. Same bundle, runner and pull; the queue is a local folder (`C:\cj_local`).
```powershell
python registry/registry.py job new --type train --where laptop --name v5_y8s_640 --images <list.txt> --who pujan
python training/make_job_bundle.py --dataset-dir <prepared dataset> -- --single-run --model-size s --imgsz 640 --batch 4 --workers 2 --rect 0
python jobs/sync.py local job_20261103_v5_y8s_640 --bundle <zip> --max-hours 4   # push + run + pull, one command
```
Rules on a GPU under 8 GB (the laptop: RTX 3050 Ti, 4 GB, 15 GB RAM):

| Rule | Why |
|---|---|
| `--max-hours <H>` always (exit 2 without it) | A dataloader worker that dies of WinError 1455 hangs the trainer with no error (11-03: 1 h 25 min) |
| `--workers` ≤ 2, and free commit memory (RAM + page file) ≥ 4 GB + 2 GB per worker at start, or blocked | Each worker loads its own torch/CUDA; the hang happened at 2.3 GB free |
| y8m/l/x or imgsz > 640: blocked "desktop only", unless `--allow-big` | Measured below: y8m at 640 fits, nothing at 960 does |
| Refuses (exit 2, job stays queued): a network queue folder, or running on battery | The laptop never takes the desktop's jobs |
| `--allow-big` is recorded in DONE.json and the registry note | Visible on the history page |

- **Set once:** NVIDIA Control Panel → Manage 3D settings → Program settings → python.exe → "CUDA - Sysmem Fallback Policy" = "Prefer No Sysmem Fallback". Without it, too big runs 10–50× slower instead of failing "oom".
- **Comparing with v5 F5:** v5 F5 is y8m, imgsz 1440, batch 8, `rect` off, from v4_y8m weights (`models/models.yaml`). Pass `--rect 0` to match. A 640 model isn't directly comparable; compare on the test split (val is inflated).
- **`--max-hours` (any machine), not Ultralytics `--time`.** `--time` overrides `--epochs` and stretches or shrinks the epoch count to fill the time. `--max-hours` stops the trainer (and its workers) at H hours: blocked, `run.json` status `time_cap`, `weights/last.pt` kept, imported as a failed run.

**What fits on the laptop** (measured 11-03: v5 S1, 114 train images, 3 epochs, `--rect 0`, through the runner):

| Model | imgsz | Batch | Workers | Peak GPU (reserved, of 4 GB) | s / epoch | 300 epochs |
|---|---|---|---|---|---|---|
| y8n | 640 | 8 | 2 | 3.5 GB | 9 | ~45 min |
| y8n | 640 | 8 | 0 | 3.2 GB | 24 | ~2 h |
| y8s | 640 | 4 | 0 | 2.8 GB | 28 | ~2.3 h |
| y8m | 640 | 2 | 0 | 3.6 GB | 33 | ~2.8 h |
| y8m | 640 | 4 | 0 | 3.6 GB | 31 | ~2.6 h |
| y8n | 960 | 4 | 0 | 3.0 GB, then out of memory 37% into epoch 2 (a dense batch) | 37 | — |
| y8s | 960 | 2 → 1 | 0 | out of memory in epoch 1, even after halving to 1 | — | — |

- 640 is the safe ceiling; 960 is borderline (y8n batch 2 and y8m/s at 960 untested). Ultralytics halves the batch only in epoch 1, so a later spike fails the run.
- The NVIDIA GPU was empty at each start (the display runs on the AMD iGPU): the GPU limits are real. System RAM was short (2–6 GB commit free), which forced 0 workers; 2 workers are ~2.6× faster when the memory check allows it (close Box, browsers, extra windows).
- Not tried yet: batch 1–2 at 960; 640 tiles cut from the native images (keeps the 1440 pixel scale; needs a tiler that cuts labels too); `mask_ratio` 8; a frozen backbone when fine-tuning.
- Set `--max-hours` ~1.5× the table (early stopping often ends sooner). The desktop's 1440 runs can't be done here.
- **Tests.** `python -m unittest discover -s jobs/tests` uses temp roots and a fake trainer, so no GPU is needed.

## Queue folder
```
C:\cj\q\.incoming\<job>\   push in progress (the runner never looks here)
C:\cj\q\<job>\             <bundle>.zip, <bundle>.zip.md5, job.json, READY      laptop writes; READY last
C:\cj\work\<job>\          unzipped bundle + training output                     runner; deleted when done
C:\cj\d\.tmp_<job>\        results being written                                 runner
C:\cj\d\<job>\             run.json, runs\<run>\, status.json, train.log, DONE.json (last)
C:\cj\d\.retired\          earlier attempts moved aside by `sync.py retry`
C:\cj\runner.lock          OS lock (msvcrt) held for the whole job; runner.lock.info says who
```
- Keep the root and the names short. Windows' 260-character limit once silently dropped images during unzip.
  Use `C:\cj`, short dataset names, and job names like `v6_y8m`. Job ids over 48 characters are refused.
- The runner deletes only inside its own `work\<job>\`. The laptop never deletes on the share: `retry` renames the folder, and copies never use `/MIR`, `/PURGE` or `/MOV*`.
- **Copying.** Copies to a `\\host\share` path use `robocopy /E /Z /R:2 /W:5`; an exit code ≥ 8 is a failure. A local folder uses shutil.
- **Pull.**
  - It copies each run into `<dataset>\.pull_tmp\<run>` first, then renames it into `runs\<run>`.
  - An existing run folder is never overwritten (it is reported as kept).
  - Pulled jobs are logged in `<CryoAI-laptop>\jobs\pulled.jsonl`, keyed by the DONE.json hash, so a second pull adds nothing.
  - It writes the registry only through `registry.py import` and `registry.py set-status <job> done|blocked --reason ... --who pujan`.

## Desktop setup checklist (user, admin; once)
1. **IT.** Check that the department allows Tailscale and a file share on this PC.
2. **Tailscale.** Install it and sign in. In the tailnet ACL, allow TCP 445 to the desktop **only from `pujrog`** (the laptop).
3. **Firewall.** Add an inbound rule: TCP 445 from `100.64.0.0/10` only (the tailnet), on all profiles.
   Then open the built-in **"File and Printer Sharing (SMB-In)"** rules and make sure none of them allows 445 from any address. Disable them or scope them to `100.64.0.0/10`.
   Check with `Get-NetFirewallRule -DisplayGroup "File and Printer Sharing" | Get-NetFirewallAddressFilter`.
4. **Share.** Create the queue root, e.g. `C:\cj`. Share it to **the user's own account only**; remove Everyone and Users. Turn on SMB encryption:
   ```powershell
   New-SmbShare -Name cj -Path C:\cj -FullAccess "$env:COMPUTERNAME\<user>" -EncryptData $true
   ```
   The runner runs as that user, never as admin or SYSTEM.
5. **Repo and venv.** Clone cryoEV and create a venv. Check out **the same commit** as the bundles (`git fetch; git checkout <commit>`); the runner blocks "commit mismatch" otherwise.
   Install `requirements.txt`, with Ultralytics 8.4.x and CUDA torch.
6. **Page file.** Set a fixed page file on C: of 48–64 GB. Until then, bundle with `--batch 8 --workers 4` or fewer (README gotcha: Windows commit memory, `ArrayMemoryError` / WinError 1455 show up as run status `oom`).
7. **Power and updates.** No sleep on AC power (`powercfg /change standby-timeout-ac 0`). Set Windows Update active hours to cover training time, so it doesn't reboot mid-run.
8. **Env vars** (user scope, `setx`):
   - `CRYOEV_NTFY_TOPIC`: the same secret topic as the laptop, for the end-of-job push.
   - `CRYOEV_CRYOAI_ROOT`: not needed for jobs, because the runner sets it to `work\<job>` for each run. Set it only for hand runs.
   - **Long paths.** Either enable long paths (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`, admin, reboot) or keep the short root `C:\cj`. The runner's 240-character check applies either way.

## Recovery
- **A stale `d\.tmp_<job>\`.** The runner was killed or the PC slept.
  - Just start the runner again. The lock was released when the process died. The runner renames the old folder to `.tmp_<job>.stale_<time>` and runs the job again from a clean `work\<job>\`.
  - Delete the `.stale_` folders by hand when you no longer need them.
  - A training process orphaned by the kill still holds the GPU, and the runner blocks "GPU busy" until it exits. Find it with `nvidia-smi` and end it.
- **A failed copy.**
  - A failed `push` leaves `q\.incoming\<job>\`, which the runner ignores. Push again: it copies into the same folder, checks the md5, then renames it.
  - A failed `pull` leaves `<dataset>\.pull_tmp\<run>\`. Nothing is logged and no status is set, so run `pull` again: robocopy `/Z` resumes into the same folder.
- **A stale lock.** There isn't really one: the OS lock dies with the process. `runner.lock` and `runner.lock.info` staying on disk are normal.
  If `run_job.py` exits 3 ("another runner holds the lock"), a runner really is running. Check Task Manager for `python ... run_job.py`.
- **Blocked jobs.** `sync.py pull` sets the registry job to blocked, with the reason.
  - Fix the cause: check out the commit, free the GPU, use a shorter name, or move the job to the desktop.
  - Then run `python jobs/sync.py retry <job> --share ...`. It moves `d\<job>\` to `d\.retired\`, the next runner picks the job again, and the next pull sets it to done.
  - A bad bundle (md5, path length, "desktop only" on the laptop) needs a new bundle and a new job instead.
