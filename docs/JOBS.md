# Training jobs: laptop queue → desktop watcher

The laptop is home base: it owns the registry, makes bundles, pushes them and pulls results.
The A6000 desktop (shared lab PC) runs `jobs/watch.py`, started once. It runs every queued job, one at a time, with the **latest GitHub main**, and **never writes the registry**.
Transport is a folder on the desktop, shared over Tailscale (SMB).
Tracking rules: [TRACKING_CONTRACT.md](TRACKING_CONTRACT.md). Producers write owner files (`run.json`, `DONE.json`); the laptop's importer reads them.

## Commands
Laptop (PowerShell, in the repo):
```powershell
# 0. Code changes the job needs must be on GitHub main first (the desktop trains with main): commit + push
# 1. A train job in the registry (job/1; the list is the dataset's images)
python registry/registry.py job new --type train --where desktop --name v6_y8m --images <list.txt> --who pujan
# 2. A bundle: dataset pack ds_<sha12>.zip (reused if already there) + thin job zip + _README.md + .md5 sidecars;
#    JOB.json has bundle_id, the full commit and the pack's hash (refused with uncommitted code). --single-zip: the old one zip
python training/make_job_bundle.py --dataset-dir <prepared dataset> -- --single-run --model-size m --batch 8 --imgsz 1440 --workers 4
# 3. Queue it (refused if its commit isn't on origin/main); uploads the dataset pack only if the share doesn't have it yet.
#    --max-hours caps the run (default 48); --hold queues without starting
python jobs/sync.py push job_20261102_v6_y8m --bundle <zip> --share \\<desktop>\cj --max-hours 24
python jobs/sync.py release job_20261102_v6_y8m --share \\<desktop>\cj      # only after --hold
python jobs/sync.py status --share \\<desktop>\cj                            # jobs, dataset cache, watcher state and heartbeat
# 4. Bring results home: run folders -> <CryoAI>/training outputs/<dataset>/runs/, importer, job done|blocked
#    (without weights/epoch*.pt; --all-weights copies those too)
python jobs/sync.py pull --share \\<desktop>\cj
# Pause / resume the watcher (it finishes the current job first; STOP is renamed, never deleted)
python jobs/sync.py stop  --share \\<desktop>\cj
python jobs/sync.py start --share \\<desktop>\cj
# Cancel a queued or running job (stops it within ~10 s, keeps the partial run; registry: cancelled after pull)
python jobs/sync.py cancel job_20261102_v6_y8m --share \\<desktop>\cj --reason "wrong dataset"
```
Desktop (runner clone `C:\cj_repo`, venv `C:\cj_venv`; setup below):
```powershell
C:\cj_venv\Scripts\python C:\cj_repo\jobs\watch.py --queue-dir C:\cj
```

**Watcher** (`jobs/watch.py`). Every minute it writes `C:\cj\watch.json` (state, job, heartbeat) and `watch.log`, then:

| Situation | What it does |
|---|---|
| `STOP` exists (`sync.py stop`) | Paused. A running job is never interrupted |
| GPU busy (a labmate: any python, ≥ 256 MB, or > 2 GB in use in total) | Waits; the job stays queued. One ntfy after 2 h |
| Before each job | Holding `runner.lock`: `git fetch` + `merge --ff-only origin/main` (only main, only fast-forward) |
| Clone not on main, modified tracked files, untracked `.py`, `requirements.txt` changed since the venv stamp, disk < 10 GB | **Paused + one ntfy. No job is blocked**; it resumes by itself once fixed |
| Runs the job | `run_job.py --job X --max-hours <job's, default 48>` as a child; the run records `runner_commit` |
| Child exit 2 (refused) | Writes `STOP` (paused) + ntfy: needs a person |
| Idle, every 10 min (`--update-every`) and at start | Updates the clone; if its own code changed (`watch.py`, `common.py`, `run_job.py`, `dataset_cache.py`, `oslock.py`) it exits 75 and the supervisor loop in `watch.py` starts it again from the new file. No restart by hand after a push; a running job is never interrupted |

- **Code version.** Each job trains with the clone's main at that moment. DONE.json keeps both `cryoEV_commit` (where the bundle was made) and `runner_commit` (the code that trained). A bundle from code newer than the desktop's main is blocked "commit mismatch".
- **After a requirements change on main:** at the desktop, `C:\cj_venv\Scripts\pip install -r C:\cj_repo\requirements.txt`, then `C:\cj_venv\Scripts\python C:\cj_repo\jobs\watch.py --stamp-venv`.
- **Runner checks** (`run_job.py`, per job): zip md5 (job.json and the `.md5` sidecar); for a thin bundle, the dataset pack (below); the bundle's commit is in the runner's history; Ultralytics 8.4.x; longest unzipped path < 240 characters; no `download:` in `dataset.yaml` (Ultralytics would run it); free disk ≥ 10 GB + 3× (the zip + its dataset pack). It keeps the machine awake while the job runs.
- A failed check or run ends **blocked** with a reason. A job is **done** only when the run's `run.json` says `finished` **and** the trainer exited 0.
- A job interrupted twice (reboots: `d\.tmp_<job>.stale_*`) is blocked "killed twice" instead of run a third time.
- The GPU-busy check runs before the old `work\<job>\` is cleared, so an orphaned trainer's files are never deleted under it. If a final rename fails (antivirus, an open Explorer window), it is retried; if it still fails, the runner exits non-zero and the results stay in `d\.tmp_<job>\`.
- One ntfy push at the end of each job, only if `CRYOEV_NTFY_TOPIC` is set.
- **Exit codes.** `run_job.py`: 0 = done or nothing ready, 1 = blocked, 2 = refused at start, 3 = the lock is held, 4 = GPU busy (2–4 leave the queue untouched). `watch.py`: 3 = another watcher is running. `sync.py`: 2 = refused or failed.
- **Smoke test.** `jobs/tests/test_smoke.py` (real GPU, y8n at 320). The fixture is `<CryoAI-laptop>\fixtures\smoke\` (`tiny\` + `yolov8n-seg.pt`):
  `set CRYOEV_JOBS_SMOKE_DATA=...\fixtures\smoke\tiny` and `set CRYOEV_JOBS_SMOKE_WEIGHTS=...\fixtures\smoke\yolov8n-seg.pt`.
- **Tests.** `python -m unittest discover -s jobs/tests`: temp roots, a fake trainer, and a temp git origin for the watcher (`test_watch.py`); no GPU needed. `test_monitor.py` covers the live monitor.

## Live monitor
The history page's **Live** tab shows the queue as it runs. On the laptop, start the monitor next to the history server, which serves the registry folder on the tailnet. The page fetches `live/live.json` next to `history.html`:
```powershell
python -m http.server 8768 --directory <CryoAI-laptop>\registry        # the history server (already running)
python jobs/monitor.py --share \\gpu-desktop\cj                       # writes <registry>\live\live.json every 30 s
```
- `--share` is repeatable (one card per queue); `--every 30`; `--once` writes one snapshot; `--out` changes the file. One monitor per output file (`monitor.lock`; a second exits 3).
- Read only on the share: small files through `common.read_shared`, nothing under `d\.tmp_*`, never the registry `*.jsonl`. Each queue is read in a thread and given up after 20 s.
- Opened as a file (`file://`), the page fetches nothing and shows how to start the monitor; open it from the server.
- **Server probes.** Each cycle the monitor also probes the laptop's servers on 127.0.0.1 with `GET /api/ping`: `review` 8766, `sandbox` 8765, `history` 8768 (`--servers review=8766,sandbox=8765,history=8768` is the default; `--servers ""` turns it off).
  - Any HTTP answer is **up**, 403 and 404 included. Refused, or no answer in 1 s, is **down**.
  - All the probes run at once in their own threads, apart from the queue readers, and are given up after 3 s. live.json gets `servers: [{name, port, up, ms}]`.
- **Portal uptime probes** (`--portal URL`, repeatable; default none, so the desktop's monitor is unchanged). Example: `--portal http://127.0.0.1:8770/healthz`.
  - Each cycle the monitor GETs the URL. **Up** = HTTP 200 and a JSON body with `"ok": true` (what the portal's `/healthz` returns); a 401/404, another body, a refusal or no answer in 3 s is a failed check.
  - **Down** after 2 failed checks in a row: ONE ntfy, and ONE more when it answers again. No repeats while it stays down. State is kept in memory, so a monitor restart while the portal is down sends one more down alert.
  - The sender is `training/notify.py`'s, with the topic from env `CRYOEV_NTFY_TOPIC` (never printed, logged or put in a message). Unset: `ntfy not configured` is logged once and nothing is sent. The message names the port (`portal :8770`), not the host.
  - live.json gets `portals: [{name, port, up, fails, ms}]`, only when `--portal` is given.
- **Servers bar** (history page header, under the title): Review, Sandbox and History links, each with a dot from the snapshot: green up, amber down, grey unknown (stale or no snapshot).
  - The links are built from the page's own address. On a `tailscale serve` port (443, 8443, 8445) they use the same scheme and host: Review `/`, Sandbox `:8443/`, History `:8445/history.html`. Opened on another port (e.g. 127.0.0.1:8768) they use `http:` and the local ports.
  - Opened as a file, it shows the names only, with no links.
  - The browser never contacts another port; the status comes only from `live.json`.
  - A first visit to the review needs its `?k=` link once per device (ask the lab). No key is in the page.

| Shown | Meaning |
|---|---|
| Queue **offline** / **access denied** | The share can't be reached (Tailscale, desktop off, no answer in 20 s) / the login has no access |
| Watcher **alive** | Heartbeat < 5 min old, state idle or running |
| **paused** / **waiting for GPU** / **waiting** | `STOP` or a machine problem (detail says) / a labmate uses the GPU / `runner.lock` held |
| **stale** / **no watcher** / **stopped** | No heartbeat for 5 min (watcher not running) / no `watch.json` / the watcher exited |
| Phase **setting up** / **training** / **finishing** | No `results.csv` yet / epochs running / final validation (`progress.json`), the last epoch done, or patience ran out, and no `DONE.json` yet |
| **ETA** | Epochs left × the median of the last ≤ 5 epoch durations (epoch 1 skipped), capped at the job's `max_hours`. **≤ X**: Ultralytics stops at best epoch + patience unless val improves |
| **batch 8 of 16 asked** | Ultralytics halved the batch after an out-of-memory in epoch 1 (`progress.json`); runs from before `progress.json` show "unknown (older run)" |
| **Hung?** | Running, but no new epoch for max(5 × median epoch, 10 min) |
| **Stale** banner | The snapshot is older than 3 intervals: the monitor stopped; nothing is shown as running |

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

## One call: `queue_train` and the portal's Train page
Steps 2 and 3 above as one function, for code (the portal) rather than people:
```python
from training.make_job_bundle import make_bundle   # step 2: the CLI's work without printing -> dict
import sync                                         # jobs/ on sys.path, as the scripts do
sync.queue_train(job_id, dataset_dir, exp_args, share, max_hours, hold=False, check_origin=True,
                 *, out=None, registry_root=None, expect_sha256=None, copier="auto")   # -> <share>/q/<job>
```
- **`make_bundle(dataset_dir, exp_args, out, dataset_name=None, single_zip=False, allow_dirty=False)`** writes the same files as `make_job_bundle.py` and returns `{zip, md5, md5_path, readme, pack, pack_reused, pack_bytes, dataset_sha256, job_json, name, exp, changes, n_files}`. `main()` only parses the arguments and prints. Refusals stay `SystemExit`, as on the CLI. `out=None` means `<CryoAI-laptop>\job_bundles`.
  - Control (`tests/test_make_bundle.py`): with the clock frozen, the CLI from the commit before the refactor, the CLI now and `make_bundle()` write byte-identical files (job zip, pack, sidecars, README) and the same stdout.
- **`queue_train`** = `make_bundle` + `push`, the same functions as the two commands. A relative `--weights` is taken from the repo (the portal writes `models/<id>/best.pt`). `expect_sha256` is the registry's `labels.dataset_sha256` when the dataset has one (`dataset_from_lv.py` records it).
- **Refused** (`SyncError`; nothing on the share):

  | Refusal | When |
  |---|---|
  | share not found | before anything; never created (as `push`) |
  | bad id, or not a `train` job in the registry (`registry_root`) | before bundling |
  | job already on the share (`q\<job>` or `d\<job>`) | before bundling |
  | bundle refused (uncommitted code, no `dataset.yaml`, an unknown flag, ...) | bundling (`bundle refused: ...`) |
  | dataset sha mismatch: the bundle's `dataset_sha256` ≠ `expect_sha256` (the folder changed since it was recorded) | after bundling; the bundle stays in `out` |
  | commit not on origin/main (`check_origin`) | `push` |
  | `ds\<sha>\` on the share holds another pack ("same dataset hash, different pack") | `push_pack` |
- Control (`jobs/tests/test_queue_train.py`): `make_job_bundle.py` + `sync.py push` (the CLI path) and `queue_train` on two temp shares, clock frozen, give byte-identical share trees with exactly one `q\<job>\`.
- **Portal (Train).** Only with `CJ_TRAIN_QUEUE=on` (default off: Queue writes nothing). For a **desktop** job, when a queue share is known, Queue does all three steps:
  - The share comes from the Config attribute `train_share`, else `$CJ_TRAIN_SHARE` (both taken as given), else the one share `live.json` names. Only that live.json share must be a UNC path (`\\host\share`; never `\\?\` or `\\.\`) or be listed in `$CJ_JOBS_LOCAL_SHARES`, the Jobs page's rule.
  - **Admins only** by default. Another role may send only if it is listed in `CJ_TRAIN_SEND_ROLES` (comma separated, e.g. `admin,advanced`; Config `train_send_roles`). Everyone else gets the preview without a Queue button, and Queue answers 409 "an admin must send this one", with nothing written.
  - Before anything is written, it checks that:
    - the share can be reached;
    - the dataset folder is on the laptop (the registry's `dataset_dir`, else `<CryoAI>\training outputs\<dataset>`);
    - the job id isn't on the share yet;
    - no code under `training/`, `jobs/` or `data_utils/` is uncommitted;
    - HEAD is on origin/main (a git fetch).

    A failed check: 409, nothing written. The dataset sha is checked only in the send, because it needs an md5 of every image.
  - Then three audit lines: `requested` (write-ahead, `send: gpu_queue`), then `registry.new_job` and `result`, then `queue_train` in a worker thread (one send at a time) and `send` (`ok` / `failed: <error>`).
  - A failed send marks the job **blocked** with the plain reason `not sent to the GPU queue (portal Train); details in the audit log`, through `registry.set_status`; the job schema has no `failed`. The error itself is never written to the registry, because the Jobs page shows `blocked_reason` to non-admins. It goes only into the audit's `send` line and the admin's Train page.
  - The send's state is kept in the portal process. The Train page shows it (sending / on the queue / failed) until the portal restarts. After a restart, a send with `requested` + `result ok` and no `send` line shows as "outcome unknown". The Jobs page (registry status) is the durable view.
  - A laptop job, or no share known, works as before: the registry job, then the commands (admin).
- **Recovery after a failed send** (admin; the Train page lists these commands under "Send it by hand"). Fix the cause from the audit's `send` line, then:
  ```powershell
  python training/make_job_bundle.py --dataset-dir "<dataset folder>" -- <the job's run_experiments args>
  python jobs/sync.py push <job> --bundle <zip from the line above> --share \\<desktop>\cj --max-hours <H>
  python registry/registry.py set-status <job> ready --who <your id>      # blocked -> ready; pull sets done later
  ```
  If the job should not run at all, leave it blocked or `set-status <job> cancelled`.

## Predict jobs
A **prediction set** is one batch of a user's tiles run through one registry model (`BUILD_CONTRACTS.md` section 2). It lives in `<CryoAI-laptop>\predict\<ps id>\`; review packages and results read it there.
```powershell
# 1. Make the set (status queued; a prediction_set record in the registry). --only <list> picks files from a big folder
python -m jobs.predict_set create --from-dir "<3d dir>" --manifest "<3d dir>\MANIFEST.csv" --session 20261027 `
    --model v5_evnonev_y8m_20261029 --owner pujan --sample-label-from manifest:sample_label --nm-from-manifest --nm-confirmed-by pujan
# 2a. Laptop: predict here (GPU), status.json updated as it goes
python -m jobs.predict_local --ps ps_20261105_abc123
# 2b. Desktop (only with CRYOEV_PREDICT_JOBS=1 on both machines): queue it, then pull as usual
python jobs/sync.py push-predict --ps ps_20261105_abc123 --share \\<desktop>\cj
python jobs/sync.py pull --share \\<desktop>\cj
```
- **Set folder:** `images\<nnnn>.png|.jpg` (never the client's file name; `images.csv` keeps it as `orig_name`), `images.csv`, `request.json`, `status.json` (`state` queued / running / done / failed, `n_done`, `n_total`, `where`, `job`, `error`, `updated`, `pid`, `waiting`), `p\<model>\predictions\<nnnn>.txt` + `p\<model>\summary.csv`, and `predict.log`.
  - `p\` appears only when the set is complete: the laptop writes into `.p_tmp\` and renames it; pull stages in `.pull_tmp_p\`.
  - Tiles are copied, and `images.csv` records each one's `md5`. `--link` hard-links instead (same drive only, never from a protected folder such as Cryo-image-processing or Box).
  - `images.csv` holds raw values (it is machine-read). A sample label or original file name starting with `= + - @` is refused at create, because a spreadsheet would run it as a formula.
  - `status.json` is written atomically, read-modify-write under `<root>\.locks\<ps>.status.lock`. While running, `updated` is refreshed every poll and `pid` is the predictor's: a stale `updated` with `running` means a crash, not a model still loading.
- **Create refuses** (nothing written, every problem listed): no images or more than 200; a file that isn't an 8-bit PNG/JPEG by content; a side over 8192 px (read from the header, before decoding); nm/px missing for any image; no `--nm-confirmed-by`; mixed tile sizes or nm/px within one sample label; an unknown model; `--only` entries that aren't plain file names; a bad `conf` / `max_det` / `imgsz`. `MANIFEST.csv` and OS files (`desktop.ini`) in the folder are skipped.
- **`predict_local`** runs `inference/predict_models.py --no-overlays` at the request's `conf` / `max_det` / `imgsz` (null = the model's registered size). One run at a time (`<root>\predict.lock`, exit 3).
  - The GPU must be free (the runner's rule: any python process, ≥ 256 MB, or > 2 GB in total; this process and its parent excluded).
    - Busy: the set stays **queued** with `waiting: "gpu busy: ..."`, exit 4.
    - No NVIDIA GPU: failed, exit 4.
    - `--allow-cpu` runs on the CPU instead.
  - Any error while the predictor runs (a status write, Ctrl+C, a bug) kills the predictor and its children and marks the set failed.
  - The final `.p_tmp` → `p` rename is retried with backoff; if it never succeeds, the set is failed.
  - A failed set can run again; a done one is refused (exit 2). It prints the time and peak GPU memory.
  - **Waiting** (`--wait S`; `start_detached` uses 6 h): instead of exit 3 / 4, it waits for the lock or a free GPU, checking every `--wait-poll-s` (30 s).
    - While waiting, `status.json` stays queued with `waiting` ("another analysis is running" / "gpu busy: ...") and `pid` = the waiter, and `updated` is refreshed at every check. A stale `updated` therefore means the waiter died.
    - At the deadline it gives up (exit 3 / 4).
  - **One run per set:** the set is claimed under its status lock and checked again under it right before running. A second waiter ("Start again") for a set that a live process waits on or runs exits 3 at once.
  - **Chain** (`--chain`, set by `start_detached`): after a run, with the lock released, it starts the oldest queued laptop set of the root that has no job and no live waiter. That run chains the next, so one start drains the queue.
  - **Portal:** `predict_local.start_detached(ps_dir) -> pid` starts a run detached (no console window; waits; chains; its own output goes to `<root>\.logs\<ps>.out`).
    - Call `predict_set.create()` from a worker thread: it reads, md5s and decodes every tile.
    - Imported as `jobs.*`, neither module changes `sys.path`; the registry is loaded by path as `cj_registry` (as `portal/app.py` does).
- **Desktop path.**
  - `push-predict` holds `<root>\predict.lock`, so it never pushes while a local run is going.
  - It creates the registry job `job_<today>_<ps id>` (`--type predict --where desktop`) only after the push succeeds. If that registry write then fails, it writes `q\<job>\CANCEL` (nothing is deleted).
  - It sends an images-only dataset pack (`ds\<sha>\`, the same cache; no `labels\`). The pack has fixed zip times, so the same tiles always give the same pack and a second set reuses `ds\<sha>\`. It also sends a thin zip `<ps>\JOB.json` (`job_type: predict`) + `request.json` + `images.csv`.
  - `run_job.py` keeps the md5, pack, commit, Ultralytics, path, disk and GPU-busy checks and skips the training size and worker rules. It runs `predict_models` into `work\<job>\p\`.
  - The job is done only with exit 0, a prediction file per image and `summary.csv`. `d\<job>\` gets `p\`, `predict.log`, `status.json`, `DONE.json` (+ `job_type`, `ps`, `model`, `n_done`, `n_total`). Default `--max-hours` 2.
  - `pull` copies a done job's `p\` into the set (never over an existing `p\`) and sets the set's `status.json` and record. Blocked → failed with the reason, cancelled → cancelled; nothing is copied for those. The registry importer runs only when a train job was pulled.
- **Flag.** With `CRYOEV_PREDICT_JOBS` unset, `push-predict` refuses and `run_job.py` / the watcher leave predict jobs queued, untouched. Train jobs behave the same either way. `watch.json`, and the Live view's `watcher`, has `predict_jobs: true|false`, so the Jobs page can explain a waiting predict job.
- **Live tab.** A running predict job shows as `{"kind": "predict", "n_done", "n_total", "eta_s", "hung"}`.
  - `n_done` counts the prediction files in `work\<job>\p\`, which are listed, never opened.
  - `hung` means no new file for max(10 × the time per image, 10 min).
  - Every `jobs[]` entry has `ready_at` (READY's time; `retry` refreshes it).

## Raw montages to tiles: `jobs/convert_raw.py`
Turns raw montages into the model-ready tiles (the lab's `3d. Tiles no scalebar - 14.7 A-px PNG`). The output folder is a Predict source as it is.
```powershell
python -m jobs.convert_raw --in <files or dirs> --out <new dir> [--recipe default|<YYYYMMDD>] [--sample-key <sample_key.csv>] [--gpu] [--processing-repo <dir>]
```
- **Inputs.** An MRC stack with its `.mrc.mdoc`, or SerialEM TIFF tiles with their `.idoc`. Pass the file, its sidecar or a folder (searched one level deep). A whole montage is always converted.
- **Output.** `<out>\*.png` + `MANIFEST.csv` (`output_name`, ..., `nm_per_px`, ...). `<out>` must be new or empty.
- **No pixel work here.** Every pixel comes from the lab's processing repo, imported, never copied. Its location is `CJ_PROCESSING_REPO` (or `--processing-repo`). The repo's tile export renders the tiles; the recipe is a session config `datasets/process_<recipe>.py`, used as is. `default` is 20261012.
- **Result.** stdout is always one JSON object; the pipeline's progress goes to stderr.

  | Exit | stdout |
  |---|---|
  | 0 | `{"ok": true, out_dir, recipe, recipe_defaulted, entry_point, compute, device, montages, a_px, n_tiles, tiles: [{file, nm_per_px, source_file, montage, sample_label, size}], nm_per_px, warnings}` |
  | 2 | `{"ok": false, "kind": "refused", "error", "warnings"}`: nothing left in `<out>` |
  | 1 | `{"ok": false, "kind": "failed", "error", "warnings"}` |
- **Refused:**
  - no processing repo;
  - an unknown or non-session recipe (checked statically, never executed);
  - an input that isn't a montage, or a sidecar without a finite, positive PixelSpacing (nm/px is never defaulted);
  - a missing stack or idoc, or two montages that would get the same name;
  - `<out>` not empty;
  - a montage whose estimated peak memory (~14 bytes per montage pixel) is more than the RAM free;
  - `--gpu` with the GPU busy or missing;
  - with the default recipe, a sampling outside 0.5–2× the recipe session's own (name `--recipe` to override);
  - no tile written.
  
  Montages coarser than 14.724 Å/px are dropped (no upsampling) and listed in `warnings`.
- **One-shot subprocess.** Run it as its own process, one per conversion. It is slow (~25–30 s per montage on the CPU) and imports the processing repo into its interpreter, so one process converts with one repo. The portal's Predict page is to start it detached (wave 4, W1), with `CJ_PROCESSING_REPO` set, into a folder under the predict root whose name starts with the session id.
  - A killed run leaves `<out>\.convert_work`: delete it (or `<out>`) before trying again.
- **CPU vs GPU.** Compute is CPU unless `--gpu`. To reproduce a delivered set, use its compute: GPU and CPU differ by ±1 grey level on < 0.01 % of pixels. `device` says what really ran.
- **Tests** (`jobs/tests/test_convert_raw.py`):
  - A stub processing repo checks the plumbing.
  - With `CJ_PROCESSING_REPO` set, synthetic MRC and TIFF montages go through the real pipeline. Known answer: tile size = source / (14.724 / PixelSpacing), nm/px 1.4724, an 8-bit greyscale PNG.
  - **Known-answer control:** with `CJ_CONVERT_CONTROL_DIR` set to a session that already has its 3d tiles, one montage (`CJ_CONVERT_CONTROL_MONTAGE`, default `grid1-pos1`) is converted again. It must be md5-identical to the delivered tiles, with the same nm/px, on the compute the delivered MANIFEST names. The test only reads that folder.

## Queue folder
```
C:\cj\q\.incoming\<job>\   push in progress (the runner never looks here)
C:\cj\q\<job>\             <bundle>.zip, <bundle>.zip.md5, job.json, READY      laptop writes; READY last
C:\cj\ds\.incoming\<sha>\  dataset pack upload in progress                       laptop
C:\cj\ds\<sha>\            ds_<sha12>.zip, ds_<sha12>.zip.md5, READY             laptop writes; READY last; shared by jobs
C:\cj\work\<job>\          unzipped bundle + training output                     runner; deleted when done
C:\cj\d\.tmp_<job>\        results being written                                 runner
C:\cj\d\<job>\             run.json, runs\<run>\, status.json, train.log, DONE.json (last)
C:\cj\d\.retired\          earlier attempts moved aside by `sync.py retry`
C:\cj\runner.lock          OS lock (msvcrt) held for the whole job; runner.lock.info says who
C:\cj\watch.lock           OS lock held by the one watcher
C:\cj\watch.json, watch.log  watcher state + heartbeat (read by `sync.py status`), its log
C:\cj\STOP                 pause request (`sync.py stop`); `start` renames it to .retired\STOP_<time>
```
- Keep the root and the names short. Windows' 260-character limit once silently dropped images during unzip.
  Use `C:\cj`, short dataset names, and job names like `v6_y8m`. Job ids over 48 characters are refused.
- The runner deletes only inside its own `work\<job>\`. The laptop never deletes on the share: `retry` renames the folder, and copies never use `/MIR`, `/PURGE` or `/MOV*`.
- **Copying.** Copies to a `\\host\share` path use `robocopy /E /Z /R:2 /W:5`; an exit code ≥ 8 is a failure. A local folder uses shutil.
- **Pull.**
  - It copies each run into `<dataset>\.pull_tmp\<run>` first, then renames it into `runs\<run>`.
  - An existing run folder is never overwritten (it is reported as kept).
  - It leaves `runs\<run>\weights\epoch*.pt` on the share: the `save_period` checkpoints are ~209 MB each, about 2.4 of a run's 2.7 GB. `best.pt` and `last.pt` always come (`last.pt` is all a failed or cancelled run has). `--all-weights` copies everything, as before.
  - Pulled jobs are logged in `<CryoAI-laptop>\jobs\pulled.jsonl`, keyed by the DONE.json hash, so a second pull adds nothing. Each line records `all_weights` and `not_pulled` (`files`, `bytes`, `paths` like `<run>/weights/epoch50.pt`). An epoch checkpoint needed later is copied by hand from `d\<job>\runs\<run>\weights\`; pulling again with `--all-weights` doesn't fetch it (the job is in the ledger and the run folder already exists).
  - The registry importer reads `run.json` and never the weights, so the registry records are the same either way.
  - It writes the registry only through `registry.py import` and `registry.py set-status <job> done|blocked --reason ... --who pujan`.

## Dataset cache
The link to `gpu-desktop` is slow (~1 MB/s), and a full dataset is ~360 MB. Each dataset goes to the queue once, keyed by its content; a job carries only a thin zip that names it.

- **Bundle** (`make_job_bundle.py`, the default). Two files in `--out`:
  - `ds_<sha12>.zip` + `.md5`: the dataset pack (`images/`, `labels/`, `MANIFEST.json` = schema, `dataset_sha256`, `files` {rel: md5}). If a pack with that name is already in `--out`, it is reused (its sidecar md5 and MANIFEST must match; otherwise the bundle is refused, never overwritten).
  - `<name>__job_<stamp>.zip` + `.md5`: `<name>/dataset.yaml`, `JOB.json`, `README.md`, `weights/` if packed (KB, not MB).
  - JOB.json: `dataset_sha256` = sha256 over the sorted lines `rel<TAB>md5` of images/ and labels/ only. It depends only on the files' content and paths (not on zip bytes, timestamps or the dataset name), so seed repeats and renamed copies share one pack. Also `dataset_pack` (file name), `dataset_pack_md5`, `dataset_pack_bytes`, and `md5` (per file, as before).
  - `--single-zip` writes the old single zip (images and labels inside `<name>/`, no pack). The runner still takes it.
- **Push** (`sync.py push` / `local`). Before the job itself:
  - `ds\<sha>\READY` missing: copies the pack + `.md5` to `ds\.incoming\<sha>\`, checks the copy's md5, renames it to `ds\<sha>\`, writes READY last.
  - `ds\<sha>\` there: nothing is copied. Its `.md5` must equal the local pack's md5, or the push is refused ("same dataset hash, different pack"; check both by hand). A `ds\<sha>\` with a matching `.md5` but no READY (a push interrupted after the rename) gets its READY.
  - A stale `ds\.incoming\<sha>\` (an interrupted upload) is left alone; the upload goes to a new `ds\.incoming\<sha>.<time>\`. Delete stale ones by hand.
  - job.json records `dataset_sha256`, `dataset_pack`, `dataset_pack_md5`.
  - `sync.py push-pack --bundle <job zip> --share ...` uploads only the pack (after a prune, see below).
- **Run** (`run_job.py`, `jobs/dataset_cache.py`). For a thin job: `ds\<sha>\READY` and the pack must exist ("dataset pack missing"); the pack's md5 must equal JOB.json, job.json and its sidecar; MANIFEST must hash to `dataset_sha256` and list exactly the pack's files. Then it extracts the pack into `work\<job>\training outputs\<name>\`, the thin zip over it, and checks every file's md5 against MANIFEST. Any failure: blocked, with the reason.
  - Each job gets its **own copy**: the trainer writes inside the dataset folder (`runs\`, `labels\*.cache`, `CAMPAIGN_RESULTS_BRIEF.md`), so nothing links into `ds\`.
  - `MANIFEST.json` is never written into the job's folder: the trainer sees exactly the tree a `--single-zip` bundle gives (tested byte for byte, `jobs/tests/test_dataset_cache.py`).
  - The runner never writes or deletes in `ds\`.
- **Pruning** (by hand, on `gpu-desktop`; nothing prunes automatically). `sync.py status` lists each pack (sha12, size, total) and the jobs in `q\` that may still need it (no DONE.json, or blocked: a blocked job can be retried).
  - Delete `C:\cj\ds\<sha>` only when status says "no job in q/ still needs it". To check one by hand:
    ```powershell
    Select-String -Path C:\cj\q\*\job.json -Pattern '<full sha>' | ForEach-Object { Split-Path (Split-Path $_.Path) -Leaf }   # jobs naming it
    Get-ChildItem C:\cj\d\<job>\DONE.json                                                                                 # done/cancelled = not needed
    ```
  - A job that needs a pruned pack is blocked "dataset pack missing": `sync.py push-pack --bundle <its job zip> --share ...`, then `sync.py retry <job>`.

## Desktop setup checklist (user, admin; once)
Done on `gpu-desktop` 2026-11-04. "Admin window": PowerShell → Run as administrator (prompt `C:\WINDOWS\system32>`); "normal window": plain PowerShell.
1. **IT.** Check that the department allows Tailscale, a file share and (if needed, step 5) a share-only local account.
2. **Tailscale.** Install it and sign in with the laptop's account. `tailscale status` lists both machines.
   Optional: a tailnet ACL allowing TCP 445 to the desktop only from `home-laptop`. With only your own devices on the tailnet it can wait; a wrong ACL can cut off the phone.
3. **Firewall** (admin). Allow 445 from the tailnet only, then check the built-in rules (on `gpu-desktop` only the Public SMB-In is on, scoped to `LocalSubnet`: left as is):
   ```powershell
   New-NetFirewallRule -DisplayName "cryoEV share from Tailscale" -Direction Inbound -Protocol TCP -LocalPort 445 -RemoteAddress 100.64.0.0/10 -Action Allow -Profile Any
   Get-NetFirewallRule -DisplayName "File and Printer Sharing (SMB-In)" | ForEach-Object { [pscustomobject]@{ Profile = $_.Profile; Enabled = $_.Enabled; Remote = (($_ | Get-NetFirewallAddressFilter).RemoteAddress -join ',') } } | Format-Table
   ```
4. **Folders, locked to you** (admin). New folders under `C:\` let every signed-in user modify them:
   ```powershell
   $me = whoami
   mkdir C:\cj\q -Force
   foreach ($d in 'C:\cj','C:\cj_repo','C:\cj_venv') {
     New-Item -ItemType Directory -Force $d | Out-Null
     icacls $d /inheritance:r /grant:r "${me}:(OI)(CI)F" "Administrators:(OI)(CI)F" "SYSTEM:(OI)(CI)F"
   }
   icacls C:\cj      # 3 lines: SYSTEM, Administrators, you, all (OI)(CI)(F)
   ```
   On a **shared lab login** (e.g. `lab-login`), everyone who uses that login has the same access (accepted 11-04).
5. **Share** (admin), encrypted, your account only:
   ```powershell
   $me = whoami
   New-SmbShare -Name cj -Path C:\cj -FullAccess $me -EncryptData $true
   ```
   **If that login has no password:** Windows refuses network logins without one, so add a share-only account (not in Users: no desk sign-in, reaches `C:\cj` only):
   ```powershell
   $pw = Read-Host "Password for the share account (12+ characters)" -AsSecureString
   New-LocalUser -Name queue-share -Password $pw -PasswordNeverExpires -UserMayNotChangePassword -Description "cryoEV job queue share (laptop only)"
   icacls C:\cj /grant "queue-share:(OI)(CI)M"
   Grant-SmbShareAccess -Name cj -AccountName queue-share -AccessRight Change -Force
   ```
   Undo: `Remove-LocalUser queue-share; Revoke-SmbShareAccess -Name cj -AccountName queue-share -Force`.
   **On the laptop**, once: `cmdkey /add:gpu-desktop /user:GPU-DESKTOP\queue-share /pass` (asks for the password).
6. **Runner clone and venv** (normal window; the venv lives outside the clone). No `py` launcher on `gpu-desktop`: the venv is made from Anaconda's Python 3.11.
   ```powershell
   git clone https://github.com/pkdes/cryoEV C:\cj_repo
   C:\ProgramData\anaconda3\python.exe -m venv C:\cj_venv          # or: py -3.11 -m venv C:\cj_venv
   C:\cj_venv\Scripts\python -m pip install --upgrade pip
   C:\cj_venv\Scripts\pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118   # first: a plain install is CPU-only
   C:\cj_venv\Scripts\pip install -r C:\cj_repo\requirements.txt ultralytics==8.4.94
   C:\cj_venv\Scripts\python -c "import torch, ultralytics; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), ultralytics.__version__)"   # True NVIDIA RTX A6000 8.4.94
   C:\cj_venv\Scripts\python C:\cj_repo\jobs\watch.py --stamp-venv
   ```
   Never edit files in `C:\cj_repo`: the watcher pauses on modified or extra `.py` files.
7. **Page file (optional).** A fixed 32–64 GB page file on C: needs that much free disk plus ~10 GB for jobs. Skipped on `gpu-desktop` (67 GB free, 11-04): bundle with `--batch 8 --workers 4` or fewer; a run ending `oom` with WinError 1455 / `ArrayMemoryError` is the signal to clean up disk and set it.
8. **Power and updates** (admin): `powercfg /change standby-timeout-ac 0`; Windows Update active hours over training time (may be IT-managed).
9. **ntfy** (normal window): `setx CRYOEV_NTFY_TOPIC <secret topic>`, the same value on the laptop; subscribe in the ntfy app. Never write the topic into the repo (it's public). New windows only see it.
10. **Start the watcher** in a new normal window and leave it open:
    ```powershell
    C:\cj_venv\Scripts\python C:\cj_repo\jobs\watch.py --queue-dir C:\cj
    ```
    A terminal survives a locked screen and switching user, but **stops on sign-out or reboot**.
11. **Scheduled task** (survives sign-out and reboots). Run as yourself with S4U ("do not store password"), at startup, no time limit:
    ```powershell
    $a = New-ScheduledTaskAction -Execute 'C:\cj_venv\Scripts\python.exe' -Argument 'C:\cj_repo\jobs\watch.py --queue-dir C:\cj' -WorkingDirectory 'C:\cj_repo'
    $t = New-ScheduledTaskTrigger -AtStartup
    $p = New-ScheduledTaskPrincipal -UserId (whoami) -LogonType S4U -RunLevel Limited
    $s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName cryoEV-watch -Action $a -Trigger $t -Principal $p -Settings $s
    ```
    Then, with no terminal watcher running: `Start-ScheduledTask cryoEV-watch`; from the laptop `sync.py status` shows the watcher alive. Push one smoke job to confirm CUDA and `CRYOEV_NTFY_TOPIC` work under the task.
- **Paths from Git Bash:** run `sync.py` from PowerShell, or check the share prints as `\\host\cj`. A path with one leading backslash is a local folder; `push` now refuses a share root that doesn't exist (11-04: one did, and made `C:\gpu-desktop\cj`).
- **Sharing the GPU.** The watcher waits while anyone else uses the GPU, but a job that has started keeps it until it ends. `sync.py stop` prevents new jobs only.

## Recovery
- **A stale `d\.tmp_<job>\`.** The runner was killed or the PC slept.
  - Just start the runner again. The lock was released when the process died. The runner renames the old folder to `.tmp_<job>.stale_<time>` and runs the job again from a clean `work\<job>\`.
  - Delete the `.stale_` folders by hand when you no longer need them.
  - A training process orphaned by the kill still holds the GPU, and the runner blocks "GPU busy" until it exits. Find it with `nvidia-smi` and end it.
- **A failed copy.**
  - A failed `push` leaves `q\.incoming\<job>\`, which the runner ignores. Push again: it copies into the same folder, checks the md5, then renames it.
  - A failed pack upload leaves `ds\.incoming\<sha>[.<time>]\` (ignored by the runner). Push again: it uploads into a new folder; delete the stale one by hand.
  - A failed `pull` leaves `<dataset>\.pull_tmp\<run>\`. Nothing is logged and no status is set, so run `pull` again: robocopy `/Z` resumes into the same folder.
- **A stale lock.** There isn't really one: the OS lock dies with the process. `runner.lock` and `runner.lock.info` staying on disk are normal.
  If `run_job.py` exits 3 ("another runner holds the lock"), a runner really is running. Check Task Manager for `python ... run_job.py`.
- **The watcher is paused** (`sync.py status`): the detail says why. STOP → `sync.py start`. A machine problem (clone not clean, venv stamp, disk, fetch) clears by itself once fixed at the desktop.
- **No heartbeat for 5 minutes** (`status` warns): the watcher isn't running (sign-out, reboot without the task, crash). Start it again; `watch.log` has its last lines.
- **Blocked jobs.** `sync.py pull` sets the registry job to blocked, with the reason.
  - Fix the cause: push the code ("commit mismatch"), use a shorter name, rebundle, or move the job to the desktop.
  - Then run `python jobs/sync.py retry <job> --share ...`. It moves `d\<job>\` to `d\.retired\`, the next runner picks the job again, and the next pull sets it to done.
  - A bad bundle (md5, path length, "desktop only" on the laptop) needs a new bundle and a new job instead.
  - "dataset pack missing": `sync.py push-pack --bundle <job zip>`, then `retry`. A pack md5 or MANIFEST mismatch (a damaged copy on the share): at `gpu-desktop`, rename `ds\<sha>\` aside, then `push-pack` and `retry`. If the laptop's pack is damaged too, `make_job_bundle.py` refuses to reuse it: move it out of `--out`, rebundle, and push a new job.
