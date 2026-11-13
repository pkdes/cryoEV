"""jobs/sync.py queue_train: make_bundle + push in one call (docs/JOBS.md, "queue_train").

Known-answer control: the CLI path a person runs today (`make_job_bundle.py --dataset-dir ... -- <args>`, then
`sync.py push <job> --bundle <zip> --share ...`) and queue_train(), each on its own temp share with the clock frozen,
leave byte-identical share trees: exactly one job folder q/<job>/ (zip, .md5, job.json, READY) and one ds/<sha>/
(pack, .md5, READY). Then the refusals, each leaving the share byte-identical.

Temp roots only (CRYOEV_LAPTOP_ROOT / CRYOEV_CRYOAI_ROOT / a temp registry); commit_on_origin is replaced (no git
fetch); allow_dirty for the bundle (the code under test may be uncommitted). Nothing real is queued.

    python -m unittest discover -s jobs/tests -p "test_queue_train*" -v
"""
import contextlib
import hashlib
import io
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobtest as T  # noqa: E402
from jobtest import C  # noqa: E402

from registry import store as registry  # noqa: E402
import sync  # noqa: E402
from training import make_job_bundle as mjb  # noqa: E402

JOB = "job_20261105_qt1"
FIXED = time.mktime((2026, 11, 5, 21, 30, 15, 0, 0, -1))
ARGS = ["--single-run", "--model-size", "n", "--imgsz", "320", "--epochs", "3", "--weights",
        "models/v4_y8n_20260924/best.pt"]


@contextlib.contextmanager
def frozen_clock():
    """time.* and common.now() pinned (bundle stamp, zip member times, job.json `pushed`, READY)."""
    lt, sf = time.localtime, time.strftime
    with mock.patch.object(time, "time", lambda: FIXED), \
            mock.patch.object(time, "localtime", lambda secs=None: lt(FIXED if secs is None else secs)), \
            mock.patch.object(time, "strftime", lambda fmt, t=None: sf(fmt, lt(FIXED) if t is None else t)), \
            mock.patch.object(C, "now", lambda: "2026-11-05T21:30:15"):
        yield


def snapshot(folder):
    return {str(p.relative_to(folder)): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(Path(folder).rglob("*")) if p.is_file()}


class QueueTrain(T.TempRoots, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.origin_calls = []
        origin = mock.patch.object(sync, "commit_on_origin",
                                   lambda commit: self.origin_calls.append(commit) or None)
        origin.start()
        self.addCleanup(origin.stop)
        dirty = mock.patch.object(mjb, "uncommitted_code", lambda: [])   # the code under test may be uncommitted
        dirty.start()
        self.addCleanup(dirty.stop)
        self.lap, self.cai = self.tmp / "lap", self.tmp / "cai"
        self.reg = self.lap / "registry"
        self.reg.mkdir(parents=True)
        (self.cai / "training outputs").mkdir(parents=True)
        os.environ.update(CRYOEV_LAPTOP_ROOT=str(self.lap), CRYOEV_CRYOAI_ROOT=str(self.cai),
                          CRYOEV_PROCESSING_ROOT=str(self.tmp / "no_processing"))
        os.environ.pop("CRYOEV_REGISTRY", None)
        lst = self.tmp / "imgs.txt"
        lst.write_text("a_png\nb_png\n", encoding="utf-8")
        registry.new_job("train", "desktop", "qt1", images=lst, who="pujan", root=self.reg, date="20261105")
        registry.new_job("annotate", "app", "not_train", images=lst, root=self.reg, date="20261105")
        self.ds = self.cai / "training outputs" / "qt_ds"
        for s in ("train", "val"):
            (self.ds / "images" / s).mkdir(parents=True)
            (self.ds / "labels" / s).mkdir(parents=True)
            (self.ds / "images" / s / f"{s}_0.png").write_bytes(b"\x89PNG " + s.encode())
            (self.ds / "labels" / s / f"{s}_0.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
        (self.ds / "dataset.yaml").write_text("path: C:/somewhere/qt_ds\ntrain: images/train\nval: images/val\n"
                                              "names:\n  0: EV\n", encoding="utf-8")
        self.out = self.tmp / "bundles"

    def qt(self, share=None, **kw):
        kw.setdefault("out", self.out)
        return sync.queue_train(kw.pop("job", JOB), self.ds, ARGS, share or self.share, 24.0, copier="shutil", **kw)

    def test_equals_the_cli_push_path(self):
        """The control: make_job_bundle.py (CLI) + `sync.py push` (CLI) vs queue_train, same frozen clock."""
        ctl_share, ctl_out = self.tmp / "ctl_share", self.tmp / "ctl_bundles"
        (ctl_share / C.Q).mkdir(parents=True)
        cwd = os.getcwd()
        os.chdir(T.REPO)                                   # the CLI takes the relative --weights from the repo
        try:
            with frozen_clock():
                with mock.patch.object(sys, "argv", ["make_job_bundle.py", "--dataset-dir", str(self.ds), "--out",
                                                     str(ctl_out), "--", *ARGS]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    mjb.main()
                zips = list(ctl_out.glob("*__job_*.zip"))
                self.assertEqual(len(zips), 1)
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = sync.main(["push", JOB, "--bundle", str(zips[0]), "--share", str(ctl_share), "--copier",
                                    "shutil", "--max-hours", "24"])
                self.assertEqual(rc, 0)
        finally:
            os.chdir(cwd)
        os.chdir(self.tmp)                                 # queue_train must not depend on the current folder
        try:
            with frozen_clock(), contextlib.redirect_stdout(io.StringIO()):
                q = self.qt()
        finally:
            os.chdir(cwd)
        self.assertEqual(q, self.share / C.Q / JOB)
        got, want = snapshot(self.share), snapshot(ctl_share)
        self.assertEqual(got, want)                        # byte for byte, every file on the share
        jobs = [p.name for p in (self.share / C.Q).iterdir() if p.is_dir() and not p.name.startswith(".")]
        self.assertEqual(jobs, [JOB])                      # exactly one job folder
        self.assertEqual(sorted(p.name for p in q.iterdir()),
                         sorted([zips[0].name, zips[0].name + ".md5", C.JOB_FILE, C.READY]))
        spec = C.read_json(q / C.JOB_FILE)
        self.assertEqual((spec["job"], spec["type"], spec["max_hours"]), (JOB, "train", 24.0))
        self.assertEqual([p.name for p in (self.share / "ds").iterdir() if not p.name.startswith(".")],
                         [spec["dataset_sha256"]])          # one dataset pack
        self.assertEqual(snapshot(self.out), snapshot(ctl_out))   # and the local bundle folder too
        self.assertEqual(len(self.origin_calls), 2)               # the origin check ran on both paths

    def test_hold_and_registry_root(self):
        os.environ["CRYOEV_REGISTRY"] = str(self.tmp / "elsewhere")   # the configured registry has no such job
        with self.assertRaisesRegex(sync.SyncError, "not a train job"):
            self.qt()
        with contextlib.redirect_stdout(io.StringIO()):
            q = self.qt(registry_root=self.reg, hold=True)
        self.assertTrue((q / C.JOB_FILE).is_file())
        self.assertFalse((q / C.READY).exists())

    def test_refusals_leave_the_share_unchanged(self):
        before = snapshot(self.share)
        stray = self.tmp / "gpu-desktop" / "cj"
        with self.assertRaisesRegex(sync.SyncError, "not found"):
            self.qt(share=stray)
        self.assertFalse((self.tmp / "gpu-desktop").exists(), "no stray folder created")
        for job, why in (("job_20261105_not_train", "not a train job"), ("job_20261105_unknown", "not a train job"),
                         ("not a job", "not a job id")):
            with self.subTest(job=job):
                with self.assertRaisesRegex(sync.SyncError, why):
                    self.qt(job=job)
        self.assertFalse(self.out.exists(), "nothing bundled before the cheap checks pass")
        # the dataset sha the registry recorded differs: refused before the share is touched
        with self.assertRaisesRegex(sync.SyncError, "dataset sha mismatch"):
            self.qt(expect_sha256="0" * 64)
        # a bundle refusal (uncommitted code) is a SyncError, not SystemExit
        with mock.patch.object(mjb, "uncommitted_code", lambda: ["jobs/sync.py"]):
            with self.assertRaisesRegex(sync.SyncError, "bundle refused: uncommitted changes"):
                self.qt()
        # the commit is not on origin/main
        with mock.patch.object(sync, "commit_on_origin", lambda c: "bundle commit x is not on origin/main"):
            with self.assertRaisesRegex(sync.SyncError, "not on origin/main"):
                self.qt()
        self.assertEqual(snapshot(self.share), before)

    def test_existing_job_on_the_share(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.qt()
        before, out_before = snapshot(self.share), snapshot(self.out)
        with self.assertRaisesRegex(sync.SyncError, "already in the queue"):
            self.qt()
        self.assertEqual(snapshot(self.share), before)
        self.assertEqual(snapshot(self.out), out_before, "refused before bundling")
        (self.share / C.Q / JOB).rename(self.tmp / "aside")           # a finished job in d/ counts too
        (self.share / C.D / JOB).mkdir(parents=True)
        with self.assertRaisesRegex(sync.SyncError, "already in the queue"):
            self.qt()

    def test_share_holds_another_pack_for_the_same_sha(self):
        """dataset sha mismatch on the share: ds/<sha>/ with another pack md5 is refused by push_pack."""
        with contextlib.redirect_stdout(io.StringIO()):
            res = mjb.make_bundle(self.ds, ["--single-run"], self.tmp / "probe", allow_dirty=True)
        dest = self.share / "ds" / res["dataset_sha256"]
        dest.mkdir(parents=True)
        (dest / res["pack"].name).write_bytes(b"another pack")
        (dest / (res["pack"].name + ".md5")).write_text("0" * 32 + "  x\n", encoding="utf-8")
        (dest / C.READY).write_text("x\n", encoding="utf-8")
        before = snapshot(self.share)
        with self.assertRaisesRegex(sync.SyncError, "same dataset hash, different pack"):
            self.qt()
        self.assertEqual(snapshot(self.share), before)
        self.assertFalse((self.share / C.Q / JOB).exists())

    def test_expected_sha_matches(self):
        with contextlib.redirect_stdout(io.StringIO()):
            res = mjb.make_bundle(self.ds, ["--single-run"], self.tmp / "probe", allow_dirty=True)
            q = self.qt(expect_sha256=res["dataset_sha256"])
        self.assertEqual(C.read_json(q / C.JOB_FILE)["dataset_sha256"], res["dataset_sha256"])


if __name__ == "__main__":
    unittest.main()
