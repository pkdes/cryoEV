"""jobs/sync.py attribution (round 9): every `who` goes through actor.who() -- flag > $CRYOEV_FOR > "pujan" -- and the
registry lines it writes carry `via` exactly once (registry.py, run as a child, reads $CRYOEV_VIA itself).
Each test sets CRYOEV_VIA / CRYOEV_FOR explicitly; TempRoots' env patch is undone after each test."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobtest as T  # noqa: E402
from jobtest import C  # noqa: E402

sys.path.insert(0, str(T.REPO / "registry"))
import registry  # noqa: E402
import sync  # noqa: E402

JOB = "job_20261102_t1"
IDENTITY = ("CRYOEV_VIA", "CRYOEV_FOR")


class WhoPrecedence(T.TempRoots, unittest.TestCase):
    def setUp(self):
        super().setUp()
        for k in IDENTITY:
            os.environ.pop(k, None)                   # inside TempRoots' patch: restored after the test
        self.share.joinpath(C.Q, JOB).mkdir(parents=True)
        (self.share / C.Q / JOB / C.JOB_FILE).write_text("{}", encoding="utf-8")

    def stop_who(self, flag=None):
        return sync.stop(self.share, flag).read_text(encoding="utf-8").split()[1]

    def test_stop_flag_env_default(self):
        self.assertEqual(self.stop_who(), "pujan")                                  # default
        with mock.patch.dict(os.environ, {"CRYOEV_FOR": "au1", "CRYOEV_VIA": "claude"}):
            self.assertEqual(self.stop_who(), "au1")                                # env beats the default
            self.assertEqual(self.stop_who("pujan"), "pujan")                       # the flag beats the env
            self.assertNotIn("claude", (self.share / C.STOP).read_text())           # STOP is not stamped
        with mock.patch.dict(os.environ, {"CRYOEV_FOR": "nobody"}):
            with self.assertRaises(sync.actor.ActorError):
                sync.stop(self.share)

    def test_cancel_flag_env_default(self):
        def who(**kw):
            p = sync.cancel(JOB, self.share, **kw)
            out = json.loads(p.read_text(encoding="utf-8"))
            p.unlink()
            return out
        self.assertEqual(who()["who"], "pujan")
        with mock.patch.dict(os.environ, {"CRYOEV_FOR": "au1", "CRYOEV_VIA": "claude"}):
            self.assertEqual(who()["who"], "au1")
            c = who(who="pujan")
            self.assertEqual(c["who"], "pujan")
            self.assertNotIn("via", c)                                               # CANCEL is not stamped

    def test_cli_flag_default_is_none_so_env_counts(self):
        with mock.patch.dict(os.environ, {"CRYOEV_FOR": "au1"}):
            self.assertEqual(sync.main(["stop", "--share", str(self.share)]), 0)
            self.assertEqual((self.share / C.STOP).read_text().split()[1], "au1")
            self.assertEqual(sync.main(["stop", "--share", str(self.share), "--who", "pujan"]), 0)
            self.assertEqual((self.share / C.STOP).read_text().split()[1], "pujan")
        with mock.patch.dict(os.environ, {"CRYOEV_FOR": "nobody"}):
            self.assertEqual(sync.main(["stop", "--share", str(self.share)]), 2)  # a clean refusal, no traceback


class PullStampsOnce(T.TempRoots, unittest.TestCase):
    """pull -> `registry.py set-status` child: --who resolved here, via read by the child from the inherited env."""

    def setUp(self):
        super().setUp()
        for k in IDENTITY:
            os.environ.pop(k, None)
        self.lap = self.tmp / "lap"
        self.reg = self.lap / "registry"
        self.reg.mkdir(parents=True)
        for sub in ("training outputs", "training images"):     # validate.Scan expects both (pull runs import)
            (self.tmp / "cai" / sub).mkdir(parents=True)
        os.environ.update(CRYOEV_LAPTOP_ROOT=str(self.lap), CRYOEV_REGISTRY=str(self.reg),
                          CRYOEV_CRYOAI_ROOT=str(self.tmp / "cai"), CRYOEV_PROCESSING_ROOT=str(self.tmp / "np"))
        lst = self.tmp / "imgs.txt"
        lst.write_text("a_png\n", encoding="utf-8")
        registry.new_job("predict", "desktop", "t1", images=lst, root=self.reg, date="20261102")
        d = self.share / C.D / JOB
        d.mkdir(parents=True)
        C.write_json(d / C.DONE, {"job": JOB, "status": "cancelled", "reason": "test"})   # no runs, no import

    def test_agent_pull_set_status_line(self):
        with mock.patch.dict(os.environ, {"CRYOEV_FOR": "pujan", "CRYOEV_VIA": "claude"}):
            out = sync.pull(self.share, copier="shutil")
        self.assertEqual(out[0]["status"], "cancelled")
        raw = (self.reg / "job.jsonl").read_text(encoding="utf-8").splitlines()
        last = json.loads(raw[-1])
        self.assertEqual((last["status"], last["status_by"], last["via"]), ("cancelled", "pujan", "claude"))
        self.assertEqual(raw[-1].count('"via"'), 1)
        self.assertNotIn("via", json.loads(raw[0]))                                # job new: a person (no env)

    def test_person_pull_has_no_via(self):
        out = sync.pull(self.share, who="au1", copier="shutil")
        self.assertEqual(out[0]["status"], "cancelled")
        last = json.loads((self.reg / "job.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(last["status_by"], "au1")
        self.assertNotIn("via", last)


if __name__ == "__main__":
    unittest.main()
