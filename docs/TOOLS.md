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

## Laptop services: `tools/services.py` + `tools/register_services_task.ps1`

The review server, review sandbox, history server and job monitor (and later the portal) used to run as children of a shell session and died with it. `services.py` starts them detached, keeps pid files and logs, and a logon task starts them after a reboot.

- **Config:** JSON **outside the repo**, because it holds real paths: `%USERPROFILE%\.cryoev\services.json` (override `--config` or env `CRYOEV_SERVICES_CONFIG`).
  - One entry per service: `name`, `argv`, `cwd`, `port` (null for the monitor), `enabled`, `log`; optional `host` (health check, default 127.0.0.1) and `env`.
  - `{python}` = the python running `services.py`, `{repo}` = this checkout; `%VARS%` and `~` are expanded.
  - `python tools/services.py example-config --out %USERPROFILE%\.cryoev\services.json` writes a template with placeholders (`<CryoAI-laptop>`, `<pkg>`) and never overwrites. The portal entry is there, `enabled: false`.
  - The config can be written with PowerShell's `>` (UTF-16) or as UTF-8; both load.
- **Commands:**
  - `status` (`--json`): name, enabled, state, pid, port, listening, health, uptime, log. Health is `GET /api/ping`: any HTTP answer is up (`http 403` for a review server without its key, `http 404` for the history server); `tcp` if the port accepts but doesn't speak HTTP.
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
