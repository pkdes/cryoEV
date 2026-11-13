"""jobs/sync.py: push layout and refusals, robocopy argv, pull into temp CryoAI + a temp registry (env roots).

Never touches the real registry or CryoAI: CRYOEV_LAPTOP_ROOT / CRYOEV_CRYOAI_ROOT point at temp folders and
CRYOEV_PROCESSING_ROOT at a missing one. Known answer: pulling twice leaves every registry file byte-identical.
"""
import hashlib
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobtest as T  # noqa: E402
from jobtest import C  # noqa: E402

from registry import store as registry  # noqa: E402
import sync  # noqa: E402

JOB = "job_20261102_t1"


def snapshot(folder):
    return {str(p.relative_to(folder)): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(Path(folder).rglob("*")) if p.is_file()}


class RobocopyTests(unittest.TestCase):
    def test_argv_never_mirrors_or_moves(self):
        calls = []

        def fake_run(argv, **kw):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 1, "", "")
        with mock.patch.object(C.subprocess, "run", fake_run):
            C.copy_tree("C:/a", "\\\\host\\share\\b", "auto")
            C.copy_files("C:/a", "C:/b", ["x.zip", "x.zip.md5"], "robocopy")
        self.assertEqual(len(calls), 2)
        for argv in calls:
            self.assertEqual(argv[0], "robocopy")
            up = [a.upper() for a in argv]
            for bad in ("/MIR", "/PURGE", "/MOV", "/MOVE"):
                self.assertFalse(any(a.startswith(bad) for a in up), (bad, argv))
            for need in ("/E", "/Z", "/R:2", "/W:5"):
                self.assertIn(need, up)
        with self.assertRaises(ValueError):
            C.robocopy_argv("a", "b", ["/MIR"])

    def test_exit_code_8_is_failure(self):
        with mock.patch.object(C.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 8, "", "")):
            with self.assertRaises(OSError):
                C.copy_tree("C:/a", "C:/b", "robocopy")

    def test_auto_uses_shutil_for_local_folders(self):
        self.assertFalse(C.use_robocopy("auto", "C:/a", "D:/b"))
        self.assertEqual(C.use_robocopy("auto", "\\\\desk\\cj"), os.name == "nt")

    @unittest.skipUnless(os.name == "nt", "robocopy is Windows only")
    def test_real_robocopy_local(self):
        import tempfile
        import shutil
        tmp = Path(tempfile.mkdtemp(prefix="cj_rc_"))
        try:
            (tmp / "a" / "sub").mkdir(parents=True)
            (tmp / "a" / "sub" / "f.txt").write_text("hi")
            (tmp / "b").mkdir()
            (tmp / "b" / "keep.txt").write_text("keep")
            C.copy_tree(tmp / "a", tmp / "b", "robocopy")       # exit 1 (files copied) = success
            self.assertEqual((tmp / "b" / "sub" / "f.txt").read_text(), "hi")
            self.assertEqual((tmp / "b" / "keep.txt").read_text(), "keep", "no purge")
            self.assertTrue((tmp / "a" / "sub" / "f.txt").exists(), "no move")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class SyncTests(T.TempRoots, unittest.TestCase):
    def setUp(self):
        super().setUp()
        origin = mock.patch.object(sync, "commit_on_origin", lambda commit: None)   # tested in OriginTests
        origin.start()
        self.addCleanup(origin.stop)
        self.lap, self.cai = self.tmp / "lap", self.tmp / "cai"
        self.reg = self.lap / "registry"
        self.reg.mkdir(parents=True)
        (self.cai / "training outputs").mkdir(parents=True)
        (self.cai / "training images").mkdir()          # validate.Scan expects both
        os.environ.update(CRYOEV_LAPTOP_ROOT=str(self.lap), CRYOEV_CRYOAI_ROOT=str(self.cai),
                          CRYOEV_PROCESSING_ROOT=str(self.tmp / "no_processing"))
        os.environ.pop("CRYOEV_REGISTRY", None)
        lst = self.tmp / "imgs.txt"
        lst.write_text("a_png\nb_png\n", encoding="utf-8")
        registry.new_job("train", "desktop", "t1", images=lst, who="pujan", root=self.reg, date="20261102")
        registry.new_job("annotate", "app", "not train", images=lst, root=self.reg, date="20261102")
        self.zp = T.make_bundle(self.bundles, args=["--model-size", "n", "--imgsz", "320"])

    def test_push_layout_and_refusals(self):
        q = sync.push(JOB, self.zp, self.share, copier="shutil")
        self.assertEqual(sorted(p.name for p in q.iterdir()),
                         sorted([self.zp.name, self.zp.name + ".md5", C.JOB_FILE, C.READY]))
        spec = C.read_json(q / C.JOB_FILE)
        self.assertEqual((spec["zip_md5"], spec["bundle_id"]), (C.md5_file(self.zp), self.zp.stem))
        self.assertFalse((self.share / C.Q / C.INCOMING / JOB).exists())
        with self.assertRaisesRegex(sync.SyncError, "already in the queue"):
            sync.push(JOB, self.zp, self.share, copier="shutil")
        with self.assertRaisesRegex(sync.SyncError, "not a train job"):
            sync.push("job_20261102_not_train", self.zp, self.share, copier="shutil")
        with self.assertRaisesRegex(sync.SyncError, "not a train job"):
            sync.push("job_20261102_unknown", self.zp, self.share, copier="shutil")
        bad = T.make_bundle(self.tmp / "b2")
        Path(str(bad) + ".md5").write_text("0" * 32 + "  x.zip\n")
        with self.assertRaisesRegex(sync.SyncError, "md5 mismatch"):
            (self.tmp / "s2").mkdir()
            sync.push(JOB, bad, self.tmp / "s2", copier="shutil", check_registry=False)

    def test_push_refuses_a_missing_share(self):
        stray = self.tmp / "gpu-desktop" / "cj"                 # what a UNC path missing one backslash becomes
        with self.assertRaisesRegex(sync.SyncError, "not found"):
            sync.push(JOB, self.zp, stray, copier="shutil")
        self.assertFalse((self.tmp / "gpu-desktop").exists(), "no stray folder created")

    def run_and_pull(self, **env):
        sync.push(JOB, self.zp, self.share, copier="shutil")
        with mock.patch.dict(os.environ, env):
            self.run_job()
        return sync.pull(self.share, copier="shutil")

    def test_retry_refuses_done_job_not_pulled(self):
        sync.push(JOB, self.zp, self.share, copier="shutil")
        self.assertEqual(self.run_job(), 0)
        with self.assertRaises(sync.SyncError):
            sync.retry(JOB, self.share)
        self.assertTrue((self.share / C.D / JOB).is_dir(), "a done, unpulled result is not moved")
        sync.retry(JOB, self.share, force=True)
        self.assertFalse((self.share / C.D / JOB).exists())

    def test_pull_twice_adds_nothing(self):
        out = self.run_and_pull()
        self.assertEqual(out[0]["status"], "done")
        run = out[0]["copied"][0]
        dest = self.cai / "training outputs" / "ds_tiny" / "runs" / run
        self.assertTrue((dest / "run.json").is_file())
        self.assertFalse((self.cai / "training outputs" / "ds_tiny" / sync.PULL_TMP / run).exists())
        cur = registry.current(root=self.reg)
        self.assertEqual(cur[JOB]["status"], "done")
        self.assertEqual(cur[run]["status"], "finished")                       # from run.json
        self.assertEqual(cur[run]["job_bundle"], self.zp.stem)                 # from run.json, not "pre-bundle"
        self.assertEqual(cur[run]["config"]["imgsz"], 320)
        before = snapshot(self.lap)
        self.assertEqual(sync.pull(self.share, copier="shutil"), [])
        self.assertEqual(snapshot(self.lap), before, "second pull changed the registry or ledger")
        self.assertEqual(sum(1 for _ in open(self.reg / "job.jsonl", encoding="utf-8")),
                         3, "job lines: 2 x job new + 1 set-status")

    def test_pull_never_overwrites_a_run_folder(self):
        sync.push(JOB, self.zp, self.share, copier="shutil")
        self.run_job()
        run = self.done(JOB)["run"]
        dest = self.cai / "training outputs" / "ds_tiny" / "runs" / run
        dest.mkdir(parents=True)
        (dest / "mine.txt").write_text("keep")
        out = sync.pull(self.share, copier="shutil")
        self.assertEqual((out[0]["copied"], out[0]["skipped"]), ([], [run]))
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ["mine.txt"])

    def test_cancelled_job_pulls_as_cancelled(self):
        sync.push(JOB, self.zp, self.share, copier="shutil")
        sync.cancel(JOB, self.share, reason="wrong dataset")
        self.assertEqual(sync.status(self.share), [(JOB, "queued (cancel requested)")])
        self.assertEqual(self.run_job(), 1)
        out = sync.pull(self.share, copier="shutil")
        self.assertEqual(out[0]["status"], "cancelled")
        cur = registry.current(root=self.reg)[JOB]
        self.assertEqual(cur["status"], "cancelled")
        self.assertIn("wrong dataset", cur.get("note") or "")
        rows = sync.queue_rows(self.share)
        self.assertEqual((rows[0]["state"], rows[0]["pulled"]), ("cancelled", True))

    def test_blocked_job_pulls_as_blocked_then_retry(self):
        out = self.run_and_pull(FAKE_TRAINER="error")
        self.assertEqual(out[0]["status"], "blocked")
        cur = registry.current(root=self.reg)
        self.assertEqual(cur[JOB]["status"], "blocked")
        self.assertIn("run error", cur[JOB]["blocked_reason"])
        run = out[0]["copied"][0]
        self.assertEqual(cur[run]["status"], "failed")
        self.assertIn("run.json status error", cur[run]["note"])
        with self.assertRaises(sync.SyncError):          # not a job id (e.g. the runner's live .tmp_ folder)
            sync.retry(f".tmp_{JOB}", self.share)
        sync.retry(JOB, self.share)
        self.assertFalse((self.share / C.D / JOB).exists())
        self.assertEqual(len(list((self.share / C.D / C.RETIRED).iterdir())), 1)
        self.assertEqual(self.run_job(), 0)
        out = sync.pull(self.share, copier="shutil")
        self.assertEqual(out[0]["status"], "done")
        self.assertEqual(registry.current(root=self.reg)[JOB]["status"], "done")

    def test_local_push_run_pull(self):
        lq = self.tmp / "lq"
        with self.assertRaisesRegex(sync.SyncError, "needs --max-hours"):
            sync.local(JOB, self.zp, lq, allow_big=True)
        with self.assertRaisesRegex(sync.SyncError, "needs --max-hours"):
            sync.local(JOB, self.zp, lq)
        with self.assertRaisesRegex(sync.SyncError, "network folder"):
            sync.local(JOB, self.zp, "\\\\desk\\cj", max_hours=1)
        with mock.patch.dict(os.environ, T.hooks_env(gpu=T.LAPTOP_GPU, on_battery=True)):    # refused at start
            with self.assertRaisesRegex(sync.SyncError, "exit 2"):
                sync.local(JOB, self.zp, lq, max_hours=1)
        self.assertEqual(sync.status(lq), [(JOB, "queued")])                                 # still queued
        self.assertFalse((self.lap / "jobs" / "pulled.jsonl").exists())                     # nothing pulled
        os.environ.update(T.hooks_env(gpu=T.DESKTOP_GPU))
        rc = sync.subprocess.run([sys.executable, str(T.JOBS / "run_job.py"), "--queue-dir", str(lq)]).returncode
        self.assertEqual(rc, 0)
        out = sync.pull(lq, copier="shutil")
        self.assertEqual(out[0]["status"], "done")
        self.assertEqual(registry.current(root=self.reg)[JOB]["status"], "done")
        # one command, from scratch: a second job end to end
        registry.new_job("train", "laptop", "t2", images=self.tmp / "imgs.txt", who="pujan", root=self.reg,
                         date="20261102")
        zp2 = T.make_bundle(self.tmp / "b3", args=["--model-size", "n", "--imgsz", "320"], stamp="20261102-120009")
        out = sync.local("job_20261102_t2", zp2, lq, max_hours=1)
        self.assertEqual(out[0]["status"], "done")
        self.assertEqual(registry.current(root=self.reg)["job_20261102_t2"]["status"], "done")

    def test_status_lists_jobs(self):
        sync.push(JOB, self.zp, self.share, copier="shutil")
        self.assertEqual(sync.status(self.share), [(JOB, "queued")])
        self.run_job()
        self.assertEqual(sync.status(self.share), [(JOB, "done not pulled")])


if __name__ == "__main__":
    unittest.main()
