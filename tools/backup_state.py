"""Mirror the laptop data folder's irreplaceable text state into a git clone and commit it (daily backup).

  python tools/backup_state.py --source <CryoAI-laptop> --repo <local clone> [--push] [--max-mb 20] [--dry-run]
  python tools/backup_state.py --source <CryoAI-laptop> --verify <clone>      # restore drill

Only files matching INCLUDE (and none of EXCLUDE, and <= --max-mb) are copied, byte for byte, to
<repo>/state/<same relative path>. Mirror files whose source is gone are removed (inside <repo>/state only).
The portal audit log (<portal state dir>/audit/portal.jsonl) lives outside --source (default state dirs:
~/.cryoev/portal*, i.e. live and preview; override with repeated --portal-state DIR) and is mirrored to
<repo>/state/portal_audit/<state dir name>/portal.jsonl with its own size cap (AUDIT_MAX_MB, 200); a skipped audit log prints a warning.
<repo>/MANIFEST.md5 lists `md5  relpath`, sorted by path. <repo>/.gitattributes is set to `* -text` so git never
rewrites line endings (a fresh clone must be byte-identical). Then `git add -A`, a commit only if anything
changed (`backup <UTC>: N changed`), and `git push` with --push. The source folder is only ever read.
Exit 0 on success (including "nothing changed"), non-zero on any failure.
"""
import argparse
import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

INCLUDE = [
    "registry/**/*.jsonl",
    "registry/**/*.json",
    "registry/**/*.txt",
    "registry/**/*.csv",
    "review/*/edits/**/*.json",
    "review/*/*.json",          # package metadata
    "labels/**/*.json",
    "labels/**/*.csv",
    "labels/**/*.txt",
    "september_edits/**",
    "jobs/**",
    "tracking/**",
    "logs/**",
    "portal_audit/*/portal.jsonl",   # portal audit logs, copied from the portal state dir(s), not from --source
]
EXCLUDE_PATHS = ["registry/live/**",                       # regenerated every 30 s
                 "registry/cache/**"]                      # derived parse caches (safe to delete, rebuilt on import)
EXCLUDE_NAMES = [".review_key", "*key*", "*.html", "*.htm",  # secrets, generated pages
                 "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.mrc", "*.pt", "*.lock"]
AUDIT_MAX_MB = 200                  # own cap for portal audit logs (append only, grow forever); ignores --max-mb
MIRROR_DIR = "state"
MANIFEST = "MANIFEST.md5"
GITATTRIBUTES = "* -text\n"


def _glob_rx(pattern):
    """Glob -> regex on posix relpaths: `**/` = zero or more dirs, `**` = anything, `*` = within a segment."""
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(out + r"\Z", re.IGNORECASE)


INCLUDE_RX = [_glob_rx(p) for p in INCLUDE]
EXCLUDE_RX = [_glob_rx(p) for p in EXCLUDE_PATHS]


def classify(rel, size, max_bytes):
    """'include' or the exclusion reason for one posix relpath."""
    if not any(rx.match(rel) for rx in INCLUDE_RX):
        return "not-whitelisted"
    if any(rx.match(rel) for rx in EXCLUDE_RX):
        return "excluded-path"
    name = rel.rsplit("/", 1)[-1].lower()
    if any(fnmatch.fnmatchcase(name, p) for p in EXCLUDE_NAMES):
        return "excluded-name"
    if size > max_bytes:
        return "too-big"
    return "include"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_portal_dirs():
    """Portal state dirs that exist under ~/.cryoev (live `portal`, preview `portal*`)."""
    base = Path.home() / ".cryoev"
    return sorted(d for d in base.glob("portal*") if d.is_dir()) if base.is_dir() else []


def portal_audit_files(portal_dirs):
    """{mirror relpath: Path} for each state dir's audit/portal.jsonl. Read-only."""
    out = {}
    for d in portal_dirs or []:
        f = Path(d) / "audit" / "portal.jsonl"
        if not f.is_file():
            continue
        label = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(d).name) or "portal"
        rel, i = f"portal_audit/{label}/portal.jsonl", 2
        while rel in out:
            rel, i = f"portal_audit/{label}_{i}/portal.jsonl", i + 1
        out[rel] = f
    return out


def select(source, max_mb, portal_dirs=()):
    """({relpath: Path}, {reason: [relpath]}) for the source tree plus portal audit logs. Read-only."""
    source = Path(source)
    max_bytes = int(max_mb * 1024 * 1024)
    chosen, skipped = {}, {}
    for dirpath, dirnames, filenames in os.walk(source):
        dirnames.sort()
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            rel = p.relative_to(source).as_posix()
            if not any(rx.match(rel) for rx in INCLUDE_RX):
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            why = classify(rel, size, max_bytes)
            if why == "include":
                chosen[rel] = p
            else:
                skipped.setdefault(why, []).append(rel)
    for rel, p in portal_audit_files(portal_dirs).items():
        try:
            why = classify(rel, p.stat().st_size, int(AUDIT_MAX_MB * 1024 * 1024))
        except OSError:
            continue
        if why == "include":
            chosen[rel] = p
        else:
            skipped.setdefault(why, []).append(rel)
            print(f"backup_state: WARNING portal audit log NOT backed up ({why}): {p}", file=sys.stderr)
    return chosen, skipped


def git(repo, *args, check=True):
    r = subprocess.run(["git", "-c", "core.quotepath=off", "-c", "core.autocrlf=false", *args],
                       cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
    return r


def mirror_files(mirror):
    if not mirror.is_dir():
        return {}
    return {p.relative_to(mirror).as_posix(): p for p in mirror.rglob("*") if p.is_file()}


def read_manifest(path):
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, rel = line.split("  ", 1)
                out[rel] = digest
    return out


def check_paths(source, repo):
    s, r = source.resolve(), repo.resolve()
    if s == r or s in r.parents or r in s.parents:
        raise RuntimeError("--repo and --source must not contain each other")


def backup(source, repo, max_mb=20, push=False, dry_run=False, remote="origin", portal_dirs=None):
    source = Path(source)
    if not source.is_dir():
        raise RuntimeError(f"source not found: {source}")
    if portal_dirs is None:
        portal_dirs = default_portal_dirs()
    chosen, skipped = select(source, max_mb, portal_dirs)
    total = sum(p.stat().st_size for p in chosen.values())
    if dry_run:
        print(f"dry-run: would mirror {len(chosen)} files, {total / 1e6:.2f} MB")
        for why in sorted(skipped):
            print(f"  skipped {why}: {len(skipped[why])}")
            for rel in skipped[why][:5]:
                print(f"    {rel}")
        if repo is not None and (Path(repo) / MIRROR_DIR).is_dir():
            have = mirror_files(Path(repo) / MIRROR_DIR)
            new = [r for r in chosen if r not in have]
            gone = [r for r in have if r not in chosen]
            changed = [r for r in chosen if r in have and md5(chosen[r]) != md5(have[r])]
            print(f"  vs mirror: {len(new)} new, {len(changed)} changed, {len(gone)} to remove")
        return 0

    repo = Path(repo)
    check_paths(source, repo)
    if git(repo, "rev-parse", "--is-inside-work-tree", check=False).returncode != 0:
        raise RuntimeError(f"not a git clone: {repo}")
    mirror = repo / MIRROR_DIR
    have = mirror_files(mirror)

    manifest = []
    for rel in sorted(chosen):
        src, dst = chosen[rel], mirror / rel
        digest = md5(src)
        if rel not in have or md5(have[rel]) != digest:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            if md5(dst) != digest:  # file changed mid-copy: retry once from the new content
                shutil.copyfile(src, dst)
                digest = md5(dst)
        manifest.append(f"{digest}  {rel}")
    for rel, p in have.items():
        if rel not in chosen:
            p.unlink()
    for d in sorted((p for p in mirror.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    (repo / MANIFEST).write_text("\n".join(manifest) + "\n", encoding="utf-8", newline="\n")
    ga = repo / ".gitattributes"
    if not ga.exists() or ga.read_text(encoding="utf-8") != GITATTRIBUTES:
        ga.write_text(GITATTRIBUTES, encoding="utf-8", newline="\n")

    git(repo, "add", "-A")
    staged = [ln for ln in git(repo, "diff", "--cached", "--name-only").stdout.splitlines() if ln]
    n_state = sum(1 for ln in staged if ln.startswith(MIRROR_DIR + "/"))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    if staged:
        git(repo, "commit", "-q", "-m", f"backup {stamp}: {n_state} changed")
    sha = git(repo, "rev-parse", "--short", "HEAD", check=False).stdout.strip() or "-"
    pushed = ""
    if push:
        r = git(repo, "push", "-q", remote, "HEAD", check=False)
        if r.returncode != 0:
            raise RuntimeError(f"push failed: {r.stderr.strip()}")
        pushed = ", pushed"
    what = f"committed {sha}" if staged else f"no changes (HEAD {sha})"
    print(f"backup_state: {len(chosen)} files {total / 1e6:.2f} MB, {n_state} changed, {what}{pushed}")
    return 0


def verify(source, clone, max_mb=20, portal_dirs=None):
    """Restore drill: every mirror file vs the source and vs MANIFEST.md5. Returns the problem count."""
    source, clone = Path(source), Path(clone)
    mirror = mirror_files(clone / MIRROR_DIR)
    manifest = read_manifest(clone / MANIFEST)
    if portal_dirs is None:
        portal_dirs = default_portal_dirs()
    audit = portal_audit_files(portal_dirs)
    problems = []
    for rel in sorted(set(mirror) | set(manifest)):
        if rel not in mirror:
            problems.append(f"missing from mirror (in manifest): {rel}")
            continue
        got = md5(mirror[rel])
        if manifest.get(rel) != got:
            problems.append(f"manifest mismatch: {rel}")
        src = audit.get(rel, source / rel)
        if not src.is_file():
            problems.append(f"missing from source: {rel}")
        elif md5(src) != got:
            problems.append(f"differs from source (changed since backup?): {rel}")
    chosen, _ = select(source, max_mb, portal_dirs)
    for rel in sorted(set(chosen) - set(mirror)):
        problems.append(f"in source, not in backup: {rel}")
    for p in problems:
        print("  " + p)
    print(f"verify: {len(mirror)} mirror files, {len(manifest)} manifest rows, {len(problems)} problem(s)")
    return len(problems)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, help="laptop data folder (read only)")
    ap.add_argument("--repo", help="local clone of the private backup repo")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--max-mb", type=float, default=20)
    ap.add_argument("--portal-state", action="append", type=Path, metavar="DIR",
                    help="portal state dir whose audit/portal.jsonl is included (repeatable; "
                         "default: every ~/.cryoev/portal* dir)")
    ap.add_argument("--dry-run", action="store_true", help="report what would be mirrored; write nothing")
    ap.add_argument("--verify", metavar="CLONE", help="restore drill against a clone (e.g. a fresh clone)")
    a = ap.parse_args(argv)
    try:
        if a.verify:
            return 1 if verify(a.source, a.verify, a.max_mb, a.portal_state) else 0
        if not a.repo and not a.dry_run:
            ap.error("--repo is required (or use --dry-run / --verify)")
        return backup(a.source, a.repo, a.max_mb, a.push, a.dry_run, a.remote, a.portal_state)
    except (RuntimeError, OSError) as e:
        print(f"backup_state: FAILED {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
