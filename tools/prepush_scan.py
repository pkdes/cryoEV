"""Scan the lines a push would add for secrets and private literals. Used by .githooks/pre-push.

Scans every commit in the range separately (`git log -p -U0`), so a secret added in one commit and removed
in a later one is still caught: it would be in the pushed history. Only ADDED lines are checked. New
binary files are listed as a warning and never fail the scan.

Rules:
  * built-in key patterns (Roboflow, GitHub, sk- style, AWS, private key blocks, generic key = "long value");
  * private literals from a pattern file kept OUTSIDE the repo: --patterns, else env CRYOEV_PRIVATE_PATTERNS,
    else %USERPROFILE%/.cryoev/private_patterns.txt. One case-insensitive literal per line, `#` comments,
    `re:` prefix for a regex. Missing file: warning, key patterns still run;
  * env CRYOEV_NTFY_TOPIC, if set (literal). Its value is never printed.

Usage:
  python tools/prepush_scan.py --range origin/main..HEAD
  python tools/prepush_scan.py --new-ref <sha>        # commits not on any remote (new branch, no origin/main)
  python tools/prepush_scan.py --tree HEAD            # every tracked file at a commit (audit)
  python tools/prepush_scan.py --self-test
Output per hit: `file:line  <rule>  <masked snippet>`. Exit 1 on any hit, 0 if clean, 2 on a git error.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

KEY_RULES = [
    ("roboflow-key", r"(?:\bapi_key|ROBOFLOW_API_KEY)\s*[:=]\s*['\"]?([A-Za-z0-9]{20,})"),
    ("roboflow-rf-key", r"\b(rf_[A-Za-z0-9]{16,})"),
    ("github-token", r"\b((?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    ("sk-key", r"\b(sk-[A-Za-z0-9_-]{20,})"),
    ("aws-access-key", r"\b(AKIA[0-9A-Z]{16})\b"),
    ("private-key-block", r"(-----BEGIN [A-Z ]*PRIVATE KEY-----)"),
    ("cryoev-agent-token", r"\b(cev_[A-Za-z0-9_-]{43})"),     # portal agent API tokens (portal/auth.py, round 10)
    ("generic-secret", r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]([^'\"\s]{12,})['\"]"),
]
KEY_RULES = [(name, re.compile(rx)) for name, rx in KEY_RULES]

ZERO = "0" * 40
COMMIT_MARK = "\x01commit "


def default_patterns_path():
    env = os.environ.get("CRYOEV_PRIVATE_PATTERNS")
    if env:
        return Path(env)
    return Path(os.environ.get("USERPROFILE") or Path.home()) / ".cryoev" / "private_patterns.txt"


def load_private_rules(path):
    """[(rule_name, compiled_regex)] from the pattern file; None if the file is missing."""
    path = Path(path)
    if not path.is_file():
        return None
    rules = []
    for n, raw in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("re:"):
            try:
                rx = re.compile(line[3:].strip(), re.IGNORECASE)
            except re.error as e:
                print(f"prepush_scan: WARNING bad regex on pattern line {n}: {e}", file=sys.stderr)
                continue
        else:
            rx = re.compile(re.escape(line), re.IGNORECASE)
        rules.append((f"private-pattern#{n}", rx))
    return rules


def build_rules(patterns_path, quiet=False):
    rules = list(KEY_RULES)
    private = load_private_rules(patterns_path)
    if private is None:
        if not quiet:
            print(f"prepush_scan: WARNING private patterns file not found ({patterns_path}); "
                  "scanning key patterns only", file=sys.stderr)
    else:
        rules += private
    topic = os.environ.get("CRYOEV_NTFY_TOPIC", "").strip()
    if topic:
        rules.append(("ntfy-topic", re.compile(re.escape(topic), re.IGNORECASE)))
    return rules


def mask(rule, value):
    if rule == "ntfy-topic":
        return "<ntfy topic>"
    return value[:3] + "…"


def scan_line(text, rules):
    """[(rule_name, masked)] for one line."""
    hits = []
    for name, rx in rules:
        m = rx.search(text)
        if m:
            value = m.group(1) if m.groups() and m.group(1) else m.group(0)
            hits.append((name, mask(name, value)))
    return hits


def _unquote(path):
    path = path.strip()
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape").encode(
            "latin-1", "replace").decode("utf-8", "replace")
    return path


def parse_patch(text, rules):
    """Parse `git log -p -U0 --cc` output. Returns (hits, binary_new_files).

    hits: [(commit, file, line_no, rule, masked)]. Normal diffs have a 1-char prefix per line; combined
    (merge) diffs have one column per parent, and a line counts as added only when every column is '+'
    (new relative to all parents, i.e. introduced by the merge itself)."""
    hits, binaries = [], []
    commit, cur_file, new_file = "", None, False
    nparents, line_no, in_hunk = 1, 0, False
    for raw in text.split("\n"):
        if raw.startswith(COMMIT_MARK):
            commit = raw[len(COMMIT_MARK):].strip()[:10]
            cur_file, new_file, in_hunk = None, False, False
            continue
        if raw.startswith("diff --git ") or raw.startswith("diff --cc ") or raw.startswith("diff --combined "):
            cur_file, new_file, in_hunk = None, False, False
            if raw.startswith("diff --cc ") or raw.startswith("diff --combined "):
                cur_file = _unquote(raw.split(" ", 2)[2])
            continue
        if raw.startswith("@@"):  # hunk header (added/removed body lines always carry a +/- prefix)
            at = len(raw) - len(raw.lstrip("@"))
            nparents = max(1, at - 1)
            m = re.search(r"\+(\d+)", raw)
            line_no = int(m.group(1)) if m else 0
            in_hunk = True
            continue
        if not in_hunk:  # file header lines; inside a hunk a body line may itself look like "+++ x"
            if raw.startswith("new file mode"):
                new_file = True
            elif raw.startswith("+++ "):
                target = raw[4:]
                if target.strip() != "/dev/null":
                    target = _unquote(target)
                    cur_file = target[2:] if target.startswith("b/") else target
            elif raw.startswith("Binary files "):
                m = re.match(r"Binary files (.+) and (.+) differ", raw)
                if m and (new_file or m.group(1).strip() == "/dev/null"):
                    b = _unquote(m.group(2))
                    binaries.append(b[2:] if b.startswith("b/") else b)
            continue
        if cur_file is None or len(raw) < nparents:
            continue
        prefix, body = raw[:nparents], raw[nparents:]
        if set(prefix) <= {"+", "-", " "} and prefix.strip():
            if "-" in prefix:
                continue  # removed line: not in the result, no line number
            if prefix == "+" * nparents:
                for rule, masked in scan_line(body, rules):
                    hits.append((commit, cur_file, line_no, rule, masked))
            line_no += 1
    return hits, binaries


def git(args, cwd=None):
    r = subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=cwd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.decode('utf-8', 'replace').strip()}")
    return r.stdout.decode("utf-8", "replace")


def patch_for(revs, cwd=None):
    return git(["log", "-p", "-U0", "--cc", "--no-renames", "--no-color", "--no-ext-diff", "--no-textconv",
                f"--format={COMMIT_MARK}%H", *revs], cwd=cwd)


def scan_tree(rev, rules, cwd=None):
    """Every tracked text file at <rev> (audit mode). Returns (hits, binaries)."""
    hits, binaries = [], []
    files = [f for f in git(["ls-tree", "-r", "-z", "--name-only", rev], cwd=cwd).split("\0") if f]
    for f in files:
        r = subprocess.run(["git", "cat-file", "-p", f"{rev}:{f}"], cwd=cwd, capture_output=True)
        data = r.stdout
        if b"\0" in data[:8000]:
            continue
        for n, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            for rule, masked in scan_line(line, rules):
                hits.append((rev[:10], f, n, rule, masked))
    return hits, binaries


def report(hits, binaries):
    for b in binaries:
        print(f"prepush_scan: warning: new binary file {b} (not scanned)")
    seen = set()
    for commit, f, n, rule, masked in hits:
        key = (f, n, rule, masked)
        if key in seen:
            continue
        seen.add(key)
        print(f"{f}:{n}  {rule}  {masked}  (commit {commit})")
    if hits:
        print(f"prepush_scan: BLOCKED, {len(seen)} hit(s). Remove the value, rewrite the commit(s), "
              "or (user's call only) git push --no-verify.")
        return 1
    print("prepush_scan: clean")
    return 0


def self_test():
    rules = list(KEY_RULES)
    fake = {
        "roboflow-rf-key": "x = '" + "rf_" + "A1b2" * 5 + "'",
        "github-token": "t " + "ghp" + "_" + "Z9" * 18,
        "sk-key": "k " + "sk-" + "ant-" + "q" * 24,
        "aws-access-key": "AK" + "IA" + "Q" * 16,
        "private-key-block": "-----BEGIN " + "RSA PRIVATE KEY-----",
        "generic-secret": "pass" + "word = '" + "w" * 14 + "'",
        "roboflow-key": "ROBOFLOW_" + "API_KEY=" + "a" * 24,
        "cryoev-agent-token": "Authorization: Bearer " + "cev" + "_" + "Q7-" * 14 + "z",
    }
    ok = True
    for rule, line in fake.items():
        got = [r for r, _ in scan_line(line, rules)]
        if rule not in got:
            print(f"self-test FAIL: {rule} not detected")
            ok = False
    for clean in ["password: (ask the user)", "the share needs a password:", "set the api key in env",
                  "api_key = os.environ['ROBOFLOW_API_KEY']", "token = get_token()"]:
        got = scan_line(clean, rules)
        if got:
            print(f"self-test FAIL: false hit on control line: {got}")
            ok = False
    print("self-test", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--range", help="commit range, e.g. origin/main..HEAD")
    g.add_argument("--new-ref", help="scan commits reachable from this sha and not on any remote")
    g.add_argument("--tree", help="scan every tracked file at this commit (audit)")
    g.add_argument("--self-test", action="store_true")
    ap.add_argument("--patterns", help="private pattern file (default: env CRYOEV_PRIVATE_PATTERNS, "
                                       "else %%USERPROFILE%%/.cryoev/private_patterns.txt)")
    a = ap.parse_args(argv)
    try:  # piped (git hook): emit UTF-8 for Git Bash / mintty; a real console handles Unicode itself
        if not sys.stdout.isatty():
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        else:
            sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    if a.self_test:
        return self_test()
    rules = build_rules(Path(a.patterns) if a.patterns else default_patterns_path())
    try:
        if a.tree:
            hits, binaries = scan_tree(a.tree, rules)
        else:
            revs = [a.range] if a.range else [a.new_ref, "--not", "--remotes"]
            hits, binaries = parse_patch(patch_for(revs), rules)
    except RuntimeError as e:
        print(f"prepush_scan: ERROR {e}", file=sys.stderr)
        return 2
    return report(hits, binaries)


if __name__ == "__main__":
    sys.exit(main())
