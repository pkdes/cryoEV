# Repo tools (`tools/`)

Tests: `python -m unittest discover -s tools/tests` (temp git repos only; never touches real data).

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
- **Source folder:** read-only. The script never writes there.

```
python tools/backup_state.py --source <laptop data dir> --dry-run                      # what would be included
python tools/backup_state.py --source <laptop data dir> --repo <backup clone> --push   # daily run
python tools/backup_state.py --source <laptop data dir> --verify <fresh clone>         # restore drill
```

- **Restore drill:** `--verify` checks every mirror file's md5 against the manifest and against the source. It also reports whitelisted source files that are missing from the backup. Files that changed after the last backup show up as differences, which is expected.
- **Setup, done once by the user:**
  1. Create the private repo and clone it locally.
  2. Set `user.name` and `user.email` in that clone.
  3. Make sure `git push` works without a prompt (credential manager).
- **Scheduled task:** create it as the user, **not SYSTEM**, so git credentials resolve. Put the command in a small `.bat` (plain UTF-8, no BOM), because `/TR` is limited to 261 characters.

```
schtasks /Create /SC DAILY /ST 21:00 /TN cryoai-state-backup /TR "\"<path>\backup_state.bat\""
```

Here `backup_state.bat` contains `"<python.exe>" "<repo>\tools\backup_state.py" --source "<laptop data dir>" --repo "<backup clone>" --push >> "<backup clone>\..\backup_state.log" 2>&1`. If a run fails, Task Scheduler shows the non-zero "Last Run Result".
