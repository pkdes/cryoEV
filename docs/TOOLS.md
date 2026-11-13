# Repo tools (`tools/`)

Tests: `python -m unittest discover -s tools/tests` (temp git repos, temp folders and dummy servers on free local ports; never touches real data or the running servers).

## Pre-push secret scan: `tools/prepush_scan.py` + `.githooks/pre-push`

This repo is public, and pushing to `main` deploys code to the lab GPU desktop. The hook blocks a push whose new commits add a secret.

- **Enable once per clone:** `git config core.hooksPath .githooks`. Hooks aren't enabled by cloning; each clone has to opt in.
- **What it scans:** the lines **added** by each pushed commit, commit by commit, so a key that is added and then deleted in a later commit is still caught. New binary files are listed as a warning and don't block the push.
- **Range:** `<remote sha>..<local sha>`. For a new branch it uses `<remote>/main..<local sha>`, and if that doesn't exist, the commits that aren't on any remote.
- **Rules:**
  - Built-in key patterns: Roboflow keys, GitHub tokens, `sk-` style keys, AWS access keys, private key blocks, and a generic `key/secret/token/password = "<long value>"`.
  - Private literals: real names, hosts, emails and the like. They are kept **outside the repo** in `%USERPROFILE%\.cryoev\private_patterns.txt`. Override with `--patterns <file>` or env `CRYOEV_PRIVATE_PATTERNS`. One case-insensitive literal per line, `#` for comments, and a `re:` prefix for a regex. If the file is missing, the scan prints a warning and runs the key patterns only.
  - The notification topic, from env `CRYOEV_NTFY_TOPIC` if set. Its value is never printed.
- **Output:** `file:line  <rule>  <first 3 chars>…`. The exit code is 1 on any hit, which blocks the push.
- **Manual runs:**
  - `python tools/prepush_scan.py --range origin/main..HEAD`
  - `--tree HEAD` audits every tracked file.
  - `--self-test`
- **Override:** `git push --no-verify`. Only the user makes that call, after reading the hits.
- **Python:** the hook uses `$CRYOEV_PYTHON` if set, otherwise `python`, `python3` or `py -3`.
- **Test fixtures:** build fake keys at runtime (e.g. `"rf_" + "x" * 20`), so the repo never contains a string that matches.

## Daily state backup: `tools/backup_state.py`

The laptop data folder holds the only copy of a few MB of text state: the registry `*.jsonl`, review edits, label versions, the job ledger, tracking files and logs. This tool mirrors that state into a clone of a **private** backup repo.

- **What gets copied:** the whitelist is the `INCLUDE` list at the top of the script.
- **What never gets copied:** `.review_key` and any `*key*` file, `registry/live/**`, `*.html`, images, `*.pt`, `*.lock`, and anything over `--max-mb` (default 20).
- **Where it goes:** files are copied byte for byte to `<repo>/state/<same path>`. Files deleted at the source are removed from the mirror.
- **Manifest and line endings:** it writes `<repo>/MANIFEST.md5` (`md5  path`, sorted by path) and sets `.gitattributes` to `* -text`, so that git never changes line endings.
- **Commits:** it commits only when something changed (`backup <UTC>: N changed`). It pushes only with `--push`. On any failure the exit code is non-zero.
- **Portal audit logs:** `<portal state dir>/audit/portal.jsonl` lives outside `--source`. By default the backup takes it from every existing `~/.cryoev/portal*` dir (live and preview) and copies it to `<repo>/state/portal_audit/<state dir name>/portal.jsonl`. Other files in the state dir (sessions) are never copied. Audit logs have their own size cap (`AUDIT_MAX_MB`, 200), independent of `--max-mb`. If one is over the cap, a normal run prints a `WARNING ... NOT backed up` line on stderr and drops it from the mirror.
- **`--portal-state DIR` (repeatable) REPLACES the defaults.** If you pass it, only the dirs you name are backed up, so list the live and preview dirs both. Use the same flags for `--verify`.
- **Source folder:** read-only. The script never writes there.

```
python tools/backup_state.py --source <laptop data dir> --dry-run                      # what would be included
python tools/backup_state.py --source <laptop data dir> --repo <backup clone> --push   # daily run
python tools/backup_state.py --source <laptop data dir> --verify <fresh clone>         # restore drill
```

- **Restore drill:** `--verify` checks every mirror file's md5 against the manifest and against the source. It also reports whitelisted source files that are missing from the backup. Files that changed after the last backup show up as differences, which is expected. The live audit log is appended to continuously, so `--verify` usually reports it as differing from the source; that is expected, not corruption.
- **Setup, done once by the user:**
  1. Create the private repo and clone it locally.
  2. Set `user.name` and `user.email` in that clone.
  3. Make sure `git push` works without a prompt (credential manager).
- **Scheduled task:** create it as the user, **not SYSTEM**, so git credentials resolve. Put the command in a small `.bat` (plain UTF-8, no BOM), because `/TR` is limited to 261 characters.

```
schtasks /Create /SC DAILY /ST 21:00 /TN cryoai-state-backup /TR "\"<path>\backup_state.bat\""
```

Here `backup_state.bat` contains `"<python.exe>" "<repo>\tools\backup_state.py" --source "<laptop data dir>" --repo "<backup clone>" --push >> "<backup clone>\..\backup_state.log" 2>&1`. If a run fails, Task Scheduler shows the non-zero "Last Run Result".

### Conversion state in the text backup
`INCLUDE` also lists, under `predict/raw_tiles/<cid>/`: `convert.json`, `status.json`, `session.json`, `acquisition.json`, `mirror.json`, `montages/layout.json`, `montages/stitch.json`, `gallery/**/*.jsonl` and `input/sample_key.csv` (exempted from the `*key*` name rule, since it is a lab table, not a secret). Tiles, montage and gallery images, `*.html`, raw uploads and any other file stay out. Paths are relative to `--source`, so **if `CJ_RAW_ROOT` ever moves outside `--source`, these files are silently missed**: point `--source` at a parent of it or add a second run.

## Nightly conversion mirror: `tools/mirror_conversions.py`
Copies each **finished** conversion (`status.json` state `done`) from `<raw_tiles>` to the desktop share ("queue-share"), verified, and never deletes anything there:

```
python tools/mirror_conversions.py --source <raw_tiles dir> --dest <share conversions dir> [--min-free-gb 20] [--dry-run] [--json]
```

- **`--dest` comes from the caller** (the scheduled `.bat`, outside the repo), never from the repo.
- **Copied:** everything in the conversion folder (tiles, montages, gallery, `MANIFEST.csv`, `convert.json`, `status.json`, `acquisition.json`, logs). **Not copied: `input/`**, because raw uploads are large and temporary (the portal deletes them after the retention period).
- **No robocopy.** One copier for local and UNC paths (robocopy hit the command-line length limit above ~500 tiles and its file list acts as a filter over subfolders). Each file goes to `<name>.mirror-part`, the part's md5 is checked against the source, then `os.replace` puts it in place. Nothing on the share is ever deleted; files only there survive. `*.lock` files are skipped; long dest paths get the `\\?\` prefix.
- **Verified:** each tile's md5 against the `MANIFEST` md5 column (if no row has an md5, `mirror.json` says `tiles_unverified: true`), and every file copied in this run against its source. `<dest_rel>/mirror_verified.json` on the share lists the md5 of every file. `<conv>/mirror.json` is written atomically: `dest_rel`, `mirrored_at`, `n_files`, `bytes`, `md5_ok`, `error`, `last_ok_at`, `last_ok_dest_rel` (kept on error records), `fingerprint`, `previous`. `dest_rel` is a folder name only. Limit: the read-back goes through the SMB client, which may serve cached data, so it proves the copy path, not the platters.
- **Changed since the last mirror:** if any file differs from what the share holds (real triggers: a CX re-stitch changes `montages/stitch.json` and the montage images; a re-convert changes tiles), the whole conversion goes to a new folder `<cid>__<YYYYMMDD-HHMMSS>`, built as `<name>.partial` and renamed after verify; the old folder is untouched. Updated in place instead: `gallery/**/*.jsonl` (append only, when the share copy is a prefix) and the latest-wins files `status.json`, `session.json`, `convert.log`, `convert.out` (overwritten via part+rename, left out of the fingerprint, so S's re-registration never makes a dated folder). A crashed run leaves only `*.mirror-part` / `*.partial`, which the next run overwrites or resumes.
- **Unchanged conversions** are skipped from a size+mtime fingerprint; inside a changed conversion, files with equal size and mtime (within 2 s) are not copied again.
- **Guards:** the share must be reachable and writable (checked once, before any conversion, otherwise exit 1 and no `mirror.json` is touched); `<source>/.mirror.lock` stops two runs at once (exit 1, "already running"); an error in one conversion is recorded and the next one still runs; below `--min-free-gb` (default 20) on the share it stops that conversion and records the error.
- Exit 0 on success or nothing to do, 1 on any failure; one summary line per conversion.

Add it to the nightly `.bat` after the backup line. Keep the first failing exit code and write a dated header line to the log each run:

```
echo ==== %DATE% %TIME% >> "<backup clone>\..\mirror_conversions.log"
"<python.exe>" "<repo>\tools\backup_state.py" --source "<laptop data dir>" --repo "<backup clone>" --push >> "<backup clone>\..\backup_state.log" 2>&1
set RC1=%ERRORLEVEL%
"<python.exe>" "<repo>\tools\mirror_conversions.py" --source "<laptop data dir>\predict\raw_tiles" --dest "<share conversions dir>" >> "<backup clone>\..\mirror_conversions.log" 2>&1
set RC2=%ERRORLEVEL%
if not %RC1%==0 exit /b %RC1%
exit /b %RC2%
```

## Laptop services: `tools/services.py` + `tools/register_services_task.ps1`

The review server, review sandbox, history server and job monitor (and later the portal) used to run as children of a shell session and died with it. `services.py` starts them detached, keeps pid files and logs, and a logon task starts them after a reboot.

- **Config:** JSON **outside the repo**, because it holds real paths: `%USERPROFILE%\.cryoev\services.json` (override `--config` or env `CRYOEV_SERVICES_CONFIG`).
  - One entry per service: `name`, `argv`, `cwd`, `port` (null for the monitor), `enabled`, `log`; optional `host` (health check, default 127.0.0.1), `health` (a path starting with `/`, see `status` below) and `env`.
  - `{python}` = the python running `services.py`, `{repo}` = this checkout; `%VARS%` and `~` are expanded.
  - `python tools/services.py example-config --out %USERPROFILE%\.cryoev\services.json` writes a template with placeholders (`<CryoAI-laptop>`, `<pkg>`) and never overwrites. The portal entry is there, `enabled: false`.
  - The config can be written with PowerShell's `>` (UTF-16) or as UTF-8; both load.
- **Commands:**
  - `status` (`--json`): name, enabled, state, pid, port, listening, health, uptime, log. Health is `GET /api/ping` unless the service has a `health` path (the portal entry uses `"health": "/healthz"`): then up = HTTP 200 and a JSON body with `"ok": true` (shown as `ok`; anything else is `http <status>` or `bad body`). Without `health`, any HTTP answer is up (`http 403` for a review server without its key, `http 404` for the history server); `tcp` if the port accepts but doesn't speak HTTP.
  - `start [name ...|all]` (default all): enabled entries only. It skips a service whose pid file names a live process of ours, and one whose port something else already holds: **"port busy (not ours)"**, left alone, exit 1. The service starts detached with no console window, out of the caller's job object when Windows allows it. It waits up to `--wait` (15 s) for the port and reports a quick exit with the log's last lines.
  - `stop [name ...|all]`: kills only a process whose pid, command line **and** creation time match its pid file, plus that process's children (e.g. the real python behind a venv launcher). A hand-started server, a reused pid or any other python is never touched; a stale pid file is just removed.
  - `restart [name ...|all]`.
- **Files:** pid files `%USERPROFILE%\.cryoev\run\<name>.pid` (config `run_dir`). Logs: the entry's `log`, default `%USERPROFILE%\.cryoev\logs\<name>.log`, appended with a header per start and moved to `<log>.1` when over 10 MB. The logon task's own output goes to `%USERPROFILE%\.cryoev\logs\services.log`.
- **Stop is a hard kill** (Windows has no clean stop for a windowless process). The review server writes edits with tmp + rename, so a kill mid-save loses at most that one save.
- **Migrating from hand-started servers.** Stop the old ones first: `services.py` won't touch them, and while they hold the ports `start` only reports "port busy (not ours)".
  1. `python tools/services.py example-config --out %USERPROFILE%\.cryoev\services.json`, then fill in the real paths (same `--package` folders and ports as today).
  2. `python tools/services.py status` shows each running server as "port busy (not ours)".
  3. Stop the old processes: close their shells, or end the `python` processes on 8765, 8766 and 8768 and the `monitor.py` process. Find them with `Get-NetTCPConnection -LocalPort 8766 -State Listen` (OwningProcess) and `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ft ProcessId,CommandLine`.
  4. `python tools/services.py start all`, then `status`: all running, health `http ...`. The review links and keys are unchanged, because the key is kept in `<package>/.review_key`.
- **Logon task**, one task `cryoev-services` (current user, Interactive, no admin, StartWhenAvailable, 30 s delay after logon, no time limit). Run it from the main checkout:
  - Register: `powershell -ExecutionPolicy Bypass -File tools\register_services_task.ps1` (`-Python <python.exe>` when PATH has the wrong one). It uses `pythonw.exe` so that no window opens. Running it again replaces the task.
  - Run it once now: `Start-ScheduledTask cryoev-services`.
  - Remove it: `... register_services_task.ps1 -Unregister`. Running services are left as they are.
  - It runs at logon only: after a sign-out the services stop, and the next logon starts them again.
- **Switching review packages:** edit `--package` in the config, then `services.py restart review`.

## Test gate (`tools/test_gate.py`)
- **Default (`--affected`):** runs only the suites the changed files touch (vs `origin/main`, uncommitted edits included), plus the quick repo-wide `tests` suite.
  - Edges follow who imports whom: `registry/` also runs annotation and tools; `results/` also jobs; `training/` and `inference/` also annotation and results; `portal/` also annotation; `tools/` also portal.
  - Shared code (`data_utils/`), an unrecognized code file, `CHANGELOG.md` and `requirements*.txt` run everything (those last two are not doc-only: tests parse the changelog and the pins set the environment).
  - **Browser skip (round 11):** in `--affected`, when every changed file is non-UI (`registry/` except `history_page.py`, `jobs/`, `training/`, `inference/`, `data_utils/`, `docs/`, `tools/` except the gate and its test, `cryoev/`), the 24 pure-browser files are skipped and the gate prints "N browser files skipped (no UI change)". The 3 mixed files (`test_predict_convert`, `test_data_module`, `test_admin_module`) and a changed test file always run. Any other change (`portal/`, `annotation/`, `results/`, root files) runs them all.
- **`--plan [--files a,b]`:** prints what a gate would run (suites, files, browser files skipped) and an estimated wall time from the past medians; runs nothing and takes no lock. `--files` replaces the git diff, to try a change set.
- **`--full`:** every suite, never skips. Run `--full --agent-env` as well before every go-live (the default run has the actor env stripped; this one proves nothing depends on that).
- **`--suites a,b`:** named suites.
- **One gate at a time:** a lock directory `~/.cryoev/gate/lock/` is shared by every worktree. A second gate waits up to `--lock-wait` minutes (default 45), then fails `gate busy`. A lock whose owner process is gone, or whose heartbeat (touched every 30 s) is older than 10 minutes, is stale and is taken over.
- **Environment:** children run with `CRYOEV_VIA` and `CRYOEV_FOR` removed, because the registry and jobs writers default their actor fields from them. `--agent-env` runs with `CRYOEV_VIA=claude CRYOEV_FOR=pujan` on purpose; a suite that passes only under one of the two has a hidden env dependency.
- **How it runs:**
  - Each test file is one unit. Units run in parallel (`--jobs`, default CPU/2), slowest first (median of past passing runs).
  - Browser files share a lane of 2. A file is a browser file if it imports `playwright` or a browser helper module (`test_review_tool_browser`); a comment that says "playwright" does not count.
  - A file that fails is re-run alone once. If it then passes it is reported as **FLAKY**, never as a plain pass.
  - The pre-push scan runs at the end.
- **Timings:** `~/.cryoev/gate/timings.jsonl` gets one line per file per run: wall seconds, test count, ok, flaky, retries, retry seconds, `agent_env`, and the last 600 characters of a failure (also for a flaky first attempt). `--times` lists the slowest files (median).
- **Shared test helpers:** `tests/cryoev_testkit.py` (append `<repo>/tests` to `sys.path`; portal tests get it as `cjtest.TK`, and `cjtest.free_port` is the testkit's). Use these instead of writing a local copy (round 11 #4):
  - ports: `free_port()`, `free_ports(n)` (distinct), `bind_retry`;
  - environment: `clean_env(base=None, strip_all=False, drop=(), **extra)` returns a copy without `CRYOEV_VIA`/`CRYOEV_FOR` (`strip_all=True`: every `CRYOEV_*`; `drop`: more names or `"CJ_*"` prefixes); `env_patch(...)` applies the same to `os.environ` and restores it (`with`, `self.enterContext`); `strip_environ(...)` removes them for the rest of the process; `no_actor_env()`;
  - `write_registry(root, {entity: [records]}, samples=False)`: a fixture registry folder (`samples=True` starts from `registry/samples/*.jsonl`);
  - `wait_until`, `process_lock(name)` (cross-process mutex, e.g. for `subst` drive letters).
- **Importing the registry in tests:** `from registry import store` (or `importers`, `validate`, ...) with the repo root on `sys.path`; never `registry/` itself (see [`ARCHITECTURE.md`](ARCHITECTURE.md)). `tests/test_registry_package.py` enforces it.
- **Writing timing-proof tests:** never assert a wall-clock bound on something load can slow. Block the slow part on an `Event` the test controls and assert the caller returned while it was still blocked; poll with a generous ceiling instead of a fixed `sleep`.

## Agent CLI: `python -m cryoev` (package `cryoev/`, not under `tools/`)
- **What:** one entry point for agents (and people) to read the platform and make the few writes agents may make: `status`, `find` / `show` / `lineage`, `review progress` (counts only), `feedback list|status|reply`, `pull labels|predictions|results|dataset`, `register`, `predict new`. Thin wrappers over the registry, `jobs/predict_set.py`, `portal/feedback_store.py`, `tools/services.py` and the portal's read helpers.
- **Writes:** a dry run unless `--do`; each `--do` appends a `requested` and a `result` line to `<CryoAI-laptop>/logs/agent_actions.jsonl` (the Admin page's "laptop CLI (unverified)" door). Never-automated actions are refused (exit 3).
- **Guide:** [`AGENTS.md`](AGENTS.md). **Tests:** `cryoev/tests` (temp roots only; the `cryoev` suite of the test gate).
