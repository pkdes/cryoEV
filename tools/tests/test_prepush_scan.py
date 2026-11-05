"""tools/prepush_scan.py and .githooks/pre-push, on temp git repos.

Every fake secret is assembled at runtime ("rf_" + "x" * 20 style) so this file never contains a string
the scanner would match. USERPROFILE / CRYOEV_* env vars point at temp paths: the real private pattern
file and ntfy topic are never read. Controls: a doc line `password: (ask the user)` and prose "api key"
must pass (known answer: clean).
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "prepush_scan.py"
HOOKS = REPO / ".githooks"
sys.path.insert(0, str(REPO / "tools"))
import prepush_scan  # noqa: E402


def fake_rf_key():
    return "rf_" + "x" * 20


def fake_github():
    return "gh" + "p_" + "Ab3" * 12


def run(args, cwd, env=None, inp=None):
    return subprocess.run(args, cwd=cwd, env=env, input=inp, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


class TempRepo:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "core.autocrlf", "false")
        self.commit("README.md", "start\n", "init")

    def git(self, *args):
        r = run(["git", *args], self.root)
        if r.returncode != 0:
            raise AssertionError(f"git {args}: {r.stderr}")
        return r.stdout.strip()

    def commit(self, name, text, msg="c", binary=None):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if binary is not None:
            p.write_bytes(binary)
        else:
            p.write_text(text, encoding="utf-8", newline="\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", msg)
        return self.git("rev-parse", "HEAD")


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.repo = TempRepo(t / "repo")
        self.base = self.repo.git("rev-parse", "HEAD")
        self.home = t / "home"
        self.home.mkdir()
        self.env = {k: v for k, v in os.environ.items()
                    if k not in ("CRYOEV_NTFY_TOPIC", "CRYOEV_PRIVATE_PATTERNS")}
        self.env["USERPROFILE"] = str(self.home)
        self.env["HOME"] = str(self.home)
        self.env["PYTHONIOENCODING"] = "utf-8"

    def tearDown(self):
        self.tmp.cleanup()

    def scan(self, *extra, env=None):
        return run([sys.executable, str(SCRIPT), "--range", f"{self.base}..HEAD", *extra],
                   self.repo.root, env or self.env)

    def test_planted_key_blocked_and_masked(self):
        key = fake_rf_key()
        self.repo.commit("cfg.py", f"x = 1\nRF = '{key}'\n")
        r = self.scan()
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("cfg.py:2  roboflow-rf-key  rf_…", r.stdout)
        self.assertNotIn(key, r.stdout + r.stderr)

    def test_github_token_blocked(self):
        self.repo.commit("a.txt", "token here " + fake_github() + "\n")
        self.assertEqual(self.scan().returncode, 1)

    def test_key_added_then_removed_still_blocked(self):
        self.repo.commit("cfg.py", f"RF = '{fake_rf_key()}'\n", "add")
        self.repo.commit("cfg.py", "RF = None\n", "remove")
        r = self.scan()
        self.assertEqual(r.returncode, 1, "a secret in pushed history must block even if removed later")

    def test_private_literal_from_patterns_file(self):
        pat = self.home / "pats.txt"
        literal = "Zq" + "uuxvillE"
        pat.write_text("# comment\n" + literal.lower() + "\nre:" + "Hostq" + r"\d{3}" + "\n", encoding="utf-8")
        self.repo.commit("notes.md", f"ran on {literal} today\n")
        r = self.scan("--patterns", str(pat))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("private-pattern#2", r.stdout)
        self.assertNotIn(literal, r.stdout)
        # regex line, and the env var route
        self.repo.commit("notes2.md", "box " + "Hostq" + "123\n")
        env = dict(self.env, CRYOEV_PRIVATE_PATTERNS=str(pat))
        r = self.scan(env=env)
        self.assertIn("private-pattern#3", r.stdout)

    def test_ntfy_topic_blocked_never_printed(self):
        topic = "tq-" + "abc" + "987xyz"
        self.repo.commit("n.py", f"url = 'https://example.invalid/{topic}'\n")
        env = dict(self.env, CRYOEV_NTFY_TOPIC=topic)
        r = self.scan(env=env)
        self.assertEqual(r.returncode, 1)
        self.assertIn("ntfy-topic  <ntfy topic>", r.stdout)
        self.assertNotIn(topic, r.stdout + r.stderr)
        self.assertNotIn(topic[:3], r.stdout.replace("ntfy", ""))

    def test_clean_diff_passes(self):
        self.repo.commit("ok.py", "def f():\n    return 42\n")
        r = self.scan()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("clean", r.stdout)

    def test_control_doc_lines_pass(self):
        """Known answer: past false hits and prose must not be flagged."""
        text = ("password: (ask the user)\n"
                "the share needs a password:\n"
                "Set the api key in the environment, never in code.\n"
                "api_key = os.environ['ROBOFLOW_API_KEY']\n"
                "token = get_token()\n")
        self.repo.commit("docs.md", text)
        r = self.scan()
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_missing_patterns_file_warns_and_still_catches_key(self):
        self.repo.commit("cfg.py", f"K = '{fake_rf_key()}'\n")
        r = self.scan("--patterns", str(self.home / "nope.txt"))
        self.assertIn("private patterns file not found", r.stderr)
        self.assertEqual(r.returncode, 1)

    def test_binary_new_file_warns_only(self):
        self.repo.commit("img.png", None, binary=b"\x89PNG\x00\x01" + b"\x00" * 50)
        r = self.scan()
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("new binary file img.png", r.stdout)

    def test_new_ref_mode_without_remote(self):
        sha = self.repo.commit("cfg.py", f"K = '{fake_rf_key()}'\n")
        r = run([sys.executable, str(SCRIPT), "--new-ref", sha], self.repo.root, self.env)
        self.assertEqual(r.returncode, 1)

    def test_self_test(self):
        r = run([sys.executable, str(SCRIPT), "--self-test"], self.repo.root, self.env)
        self.assertEqual(r.returncode, 0, r.stdout)


class ParseTests(unittest.TestCase):
    def test_added_line_that_looks_like_header(self):
        patch = ("\x01commit " + "a" * 40 + "\n\ndiff --git a/f b/f\n--- a/f\n+++ b/f\n"
                 "@@ -0,0 +1,2 @@\n+++ " + fake_rf_key() + "\n+ok\n")
        hits, _ = prepush_scan.parse_patch(patch, prepush_scan.KEY_RULES)
        self.assertEqual([(h[1], h[2], h[3]) for h in hits], [("f", 1, "roboflow-rf-key")])

    def test_repo_tool_files_contain_no_matchable_string(self):
        """The repo must pass its own scan: no literal fake keys in tools/, .githooks/, docs/TOOLS.md."""
        files = [p for p in (REPO / "tools").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        files += [p for p in HOOKS.rglob("*") if p.is_file()]
        if (REPO / "docs" / "TOOLS.md").exists():
            files.append(REPO / "docs" / "TOOLS.md")
        for p in files:
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                self.assertEqual(prepush_scan.scan_line(line, prepush_scan.KEY_RULES), [], f"{p}:{n}")


class HookEndToEndTests(unittest.TestCase):
    """Temp bare remote + clone with core.hooksPath = this repo's .githooks; real `git push`."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.remote = t / "remote.git"
        run(["git", "init", "-q", "--bare", str(self.remote)], t)
        self.repo = TempRepo(t / "work")
        self.repo.git("remote", "add", "origin", str(self.remote))
        self.repo.git("config", "core.hooksPath", str(HOOKS))
        home = t / "home"
        home.mkdir()
        self.env = {k: v for k, v in os.environ.items()
                    if k not in ("CRYOEV_NTFY_TOPIC", "CRYOEV_PRIVATE_PATTERNS")}
        self.env.update(USERPROFILE=str(home), HOME=str(home), CRYOEV_PYTHON=sys.executable,
                        PYTHONIOENCODING="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def push(self, *args):
        return run(["git", "push", "origin", *args], self.repo.root, self.env)

    def remote_head(self, branch="main"):
        r = run(["git", "rev-parse", "--verify", "-q", f"refs/heads/{branch}"], self.remote)
        return r.stdout.strip()

    def test_push_clean_then_blocked_then_clean(self):
        r = self.push("main")  # empty remote, no origin/main: --new-ref path
        self.assertEqual(r.returncode, 0, r.stderr)
        first = self.repo.git("rev-parse", "HEAD")
        self.assertEqual(self.remote_head(), first)

        self.repo.commit("secret.py", f"K = '{fake_rf_key()}'\n", "oops")
        r = self.push("main")  # remote sha known: --range path
        self.assertNotEqual(r.returncode, 0, r.stderr)
        self.assertIn("BLOCKED", r.stderr + r.stdout)
        self.assertEqual(self.remote_head(), first, "remote must be unchanged after a blocked push")

        # new branch with the secret: zero remote sha, origin/main exists
        self.repo.git("checkout", "-q", "-b", "feature")
        r = self.push("feature")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self.remote_head("feature"), "")

        self.repo.git("checkout", "-q", "main")
        self.repo.git("reset", "-q", "--hard", first)
        self.repo.commit("ok.py", "x = 1\n", "clean")
        r = self.push("main")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.remote_head(), self.repo.git("rev-parse", "HEAD"))


if __name__ == "__main__":
    unittest.main()
