"""tools/backup_state.py on a temp source tree and a temp git clone (bare remote + clone).

Known answers: a second run with nothing changed makes no commit; changing one edit file makes a commit
with exactly one changed path under state/; `.review_key` never appears in any commit.
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import backup_state as B  # noqa: E402

INCLUDED = {
    "registry/job.jsonl": "{\"id\": 1}\n",
    "registry/lists/job_a.txt": "img1\r\nimg2\r\n",           # CRLF must survive byte for byte
    "registry/sub/meta.json": "{}",
    "registry/table.csv": "a,b\n1,2\n",
    "review/pkg1/edits/img_a.png.json": "{\"edits\": []}\n",
    "review/pkg1/edits/deep/img_b.json": "{\"edits\": [1]}\n",
    "review/pkg1/package.json": "{\"name\": \"pkg1\"}\n",
    "labels/lv_1/version.json": "{\"v\": 1}\n",
    "labels/lv_1/conflicts.csv": "x\n",
    "labels/lv_1/notes.txt": "n\n",
    "september_edits/seed/one.png.json": "{}\n",
    "september_edits/resolution.csv": "r\n",
    "jobs/pulled.jsonl": "{}\n",
    "tracking/v2/map.csv": "m\n",
    "logs/run.log": "log line\n",
}
EXCLUDED = {
    "review/pkg1/.review_key": "SECRETVALUE\n",
    "review/pkg1/edits/api_key.json": "{}\n",
    "registry/live/live.json": "{}\n",
    "registry/history.html": "<html>",
    "review/pkg1/images/a.png": "PNG",
    "review/pkg1/index.html": "<html>",
    "review/pkg1/data.js": "var x;",
    "logs/shot.jpg": "JPG",
    "jobs/model.pt": "PT",
    "predictions/x.txt": "not whitelisted\n",
    "review/top.json": "not in a package folder\n",
}


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def git(cwd, *args):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        raise AssertionError(f"git {args}: {r.stderr}")
    return r.stdout


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.src = t / "src"
        for rel, text in {**INCLUDED, **EXCLUDED}.items():
            p = self.src / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(text.encode())
        big = self.src / "logs" / "huge.log"
        big.write_bytes(b"x" * (2 * 1024 * 1024))  # over --max-mb 1
        self.remote = t / "remote.git"
        git(t, "init", "-q", "--bare", "-b", "main", str(self.remote))
        self.clone = t / "clone"
        git(t, "clone", "-q", str(self.remote), str(self.clone))
        for k, v in (("user.name", "test"), ("user.email", "test@example.invalid")):
            git(self.clone, "config", k, v)
        self.src_snapshot = self.snapshot(self.src)
        # never read the real ~/.cryoev: no portal dirs unless a test passes them
        self._p = mock.patch.object(B, "default_portal_dirs", return_value=[])
        self._p.start()

    def tearDown(self):
        self._p.stop()
        # never write into the source: byte-identical before/after every test
        self.assertEqual(self.snapshot(self.src), self.src_snapshot)
        self.tmp.cleanup()

    @staticmethod
    def snapshot(folder):
        return {p.relative_to(folder).as_posix(): md5(p) for p in sorted(folder.rglob("*")) if p.is_file()}

    def run_backup(self, push=True, portal=None):
        extra = [x for d in (portal or []) for x in ("--portal-state", str(d))]
        return B.main(["--source", str(self.src), "--repo", str(self.clone), "--max-mb", "1"]
                      + extra + (["--push"] if push else []))

    def commits(self):
        return int(git(self.clone, "rev-list", "--count", "HEAD").strip())

    def mirrored(self):
        m = self.clone / "state"
        return {p.relative_to(m).as_posix() for p in m.rglob("*") if p.is_file()}

    def test_whitelist_manifest_and_idempotent(self):
        self.assertEqual(self.run_backup(), 0)
        self.assertEqual(self.mirrored(), set(INCLUDED))
        manifest = B.read_manifest(self.clone / "MANIFEST.md5")
        self.assertEqual(set(manifest), set(INCLUDED))
        for rel, digest in manifest.items():
            self.assertEqual(digest, md5(self.src / rel), rel)
        rels = [ln.split("  ", 1)[1] for ln in (self.clone / "MANIFEST.md5").read_text().splitlines()]
        self.assertEqual(rels, sorted(rels), "manifest sorted by path")
        n = self.commits()
        self.assertEqual(self.run_backup(), 0)
        self.assertEqual(self.commits(), n, "second run with no change must not commit")

    def test_one_change_one_path(self):
        self.run_backup()
        (self.src / "review/pkg1/edits/img_a.png.json").write_text("{\"edits\": [42]}\n")
        self.src_snapshot = self.snapshot(self.src)
        self.assertEqual(self.run_backup(), 0)
        names = [ln for ln in git(self.clone, "show", "--name-only", "--format=", "HEAD").splitlines() if ln]
        self.assertEqual([n for n in names if n.startswith("state/")], ["state/review/pkg1/edits/img_a.png.json"])
        self.assertEqual(set(names) - {"state/review/pkg1/edits/img_a.png.json"}, {"MANIFEST.md5"})
        self.assertIn(": 1 changed", git(self.clone, "log", "-1", "--format=%s"))

    def test_deleted_source_file_removed_from_mirror(self):
        self.run_backup()
        (self.src / "tracking/v2/map.csv").unlink()
        self.src_snapshot = self.snapshot(self.src)
        self.run_backup()
        self.assertNotIn("tracking/v2/map.csv", self.mirrored())
        self.assertFalse((self.clone / "state/tracking").exists(), "empty mirror dirs are pruned")

    def test_verify_on_fresh_clone(self):
        self.run_backup(push=True)
        fresh = Path(self.tmp.name) / "fresh"
        git(self.tmp.name, "clone", "-q", str(self.remote), str(fresh))
        self.assertEqual(B.verify(self.src, fresh, max_mb=1), 0)
        # control: a corrupted mirror file must be reported
        (fresh / "state/registry/job.jsonl").write_text("tampered\n")
        self.assertGreater(B.verify(self.src, fresh, max_mb=1), 0)

    def test_review_key_never_committed(self):
        self.run_backup()
        (self.src / "labels/lv_1/version.json").write_text("{\"v\": 2}\n")
        self.src_snapshot = self.snapshot(self.src)
        self.run_backup()
        log = git(self.clone, "log", "--all", "--name-only", "--format=")
        self.assertNotIn(".review_key", log)
        self.assertNotIn("api_key", log)
        self.assertNotIn("huge.log", log)
        self.assertNotIn("SECRETVALUE", git(self.clone, "log", "--all", "-p"))

    def test_dry_run_writes_nothing(self):
        before = self.snapshot(self.clone)
        self.assertEqual(B.main(["--source", str(self.src), "--repo", str(self.clone), "--dry-run"]), 0)
        self.assertEqual(self.snapshot(self.clone), before)

    def test_failure_exits_nonzero(self):
        not_repo = Path(self.tmp.name) / "plain"
        not_repo.mkdir()
        os.environ.setdefault("GIT_CEILING_DIRECTORIES", str(Path(self.tmp.name)))
        try:
            self.assertNotEqual(B.main(["--source", str(self.src), "--repo", str(not_repo)]), 0)
        finally:
            if os.environ.get("GIT_CEILING_DIRECTORIES") == str(Path(self.tmp.name)):
                del os.environ["GIT_CEILING_DIRECTORIES"]
        self.assertNotEqual(B.main(["--source", str(self.src / "missing"), "--repo", str(self.clone)]), 0)

    def test_portal_audit_included_rest_identical(self):
        self.run_backup(push=False)
        control = self.mirrored()
        self.assertEqual(control, set(INCLUDED))
        t = Path(self.tmp.name)
        live, prev = t / "portal", t / "portal_preview"
        for d, text in ((live, "{\"a\": 1}\n"), (prev, "{\"a\": 2}\n")):
            (d / "audit").mkdir(parents=True)
            (d / "audit" / "portal.jsonl").write_text(text)
            (d / "sessions.json").write_text("SESSIONSECRET")   # siblings must not be copied
        snap = self.snapshot(live)
        with mock.patch.object(B, "default_portal_dirs", return_value=[live, prev]):
            self.assertEqual(self.run_backup(push=False), 0)
            self.assertEqual(B.verify(self.src, self.clone, 1), 0)
        self.assertEqual(self.mirrored() - control,
                         {"portal_audit/portal/portal.jsonl", "portal_audit/portal_preview/portal.jsonl"})
        self.assertEqual(control - self.mirrored(), set())
        self.assertEqual((self.clone / "state/portal_audit/portal/portal.jsonl").read_text(), "{\"a\": 1}\n")
        self.assertNotIn("SESSIONSECRET", git(self.clone, "log", "--all", "-p"))
        self.assertEqual(self.snapshot(live), snap, "portal state is only read")
        # live audit log gone -> dropped from the mirror, the rest unchanged
        (live / "audit" / "portal.jsonl").unlink()
        self.run_backup(push=False, portal=[prev])
        self.assertEqual(self.mirrored() - control, {"portal_audit/portal_preview/portal.jsonl"})

    def test_portal_audit_own_cap_and_warning(self):
        import contextlib
        import io
        d = Path(self.tmp.name) / "portal"
        (d / "audit").mkdir(parents=True)
        log = d / "audit" / "portal.jsonl"
        log.write_bytes(b"x" * (2 * 1024 * 1024))     # over --max-mb 1, under the audit cap
        self.assertEqual(self.run_backup(push=False, portal=[d]), 0)
        self.assertIn("portal_audit/portal/portal.jsonl", self.mirrored(), "own, larger cap")
        with mock.patch.object(B, "AUDIT_MAX_MB", 1):   # now over the audit cap
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(self.run_backup(push=False, portal=[d]), 0)   # normal run, not dry-run
            self.assertIn("WARNING portal audit log NOT backed up (too-big)", err.getvalue())
            self.assertNotIn("portal_audit/portal/portal.jsonl", self.mirrored())
        # control: no warning when everything fits
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.run_backup(push=False, portal=[d])
        self.assertNotIn("WARNING", err.getvalue())

    def test_portal_state_flag(self):
        d = Path(self.tmp.name) / "ps"
        (d / "audit").mkdir(parents=True)
        (d / "audit" / "portal.jsonl").write_text("{}\n")
        self.assertEqual(self.run_backup(push=False, portal=[d]), 0)
        self.assertIn("portal_audit/ps/portal.jsonl", self.mirrored())


class ConversionStateTests(unittest.TestCase):
    """predict/raw_tiles/<cid>/ text state is included; tiles, montage images, html, input/ uploads are not."""

    def test_select_conversion_files(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d)
            c = "predict/raw_tiles/20261106-1_abc"
            want = ["convert.json", "status.json", "acquisition.json", "mirror.json", "session.json", "montages/layout.json",
                    "montages/stitch.json", "gallery/events.jsonl", "gallery/sub/more.jsonl", "input/sample_key.csv"]
            skip = ["tiles/t1.png", "tiles/MANIFEST.csv", "montages/m_stitched.jpg", "gallery/index.html",
                    "gallery/thumb.jpg", "input/a.mrc", "input/a.mrc.mdoc", "input/other.csv", "convert.log",
                    "input/api_key.csv", "gallery/key.jsonl", "montages/other.json"]
            for rel in want + skip:
                p = src / c / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
            chosen, _ = B.select(src, 20, portal_dirs=())
            got = sorted(r[len(c) + 1:] for r in chosen if r.startswith(c))
            self.assertEqual(got, sorted(want))


if __name__ == "__main__":
    unittest.main()
