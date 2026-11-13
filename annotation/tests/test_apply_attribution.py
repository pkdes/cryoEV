"""apply_review_edits.py version.json attribution (round 9): `created_by` (--who > $CRYOEV_FOR > null) and
`created_via` ($CRYOEV_VIA, else null). Never a key named `via` (the registry merges version.json under its later
label_version lines, where `via` means the writer of that line). Each test sets the env explicitly.

Fixture: test_apply_review_edits.build_synthetic (temp dir); people = the repo's annotation/people.json."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_apply_review_edits import build_synthetic, read_out, run  # noqa: E402

IDENTITY = ("CRYOEV_VIA", "CRYOEV_FOR")


class VersionJsonAttribution(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k not in IDENTITY}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.tmp = Path(tempfile.mkdtemp(prefix="apply_attr_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.base, self.pkg, self.edits = build_synthetic(self.tmp / "in")
        self.out_root = self.tmp / "labels"
        self.n = 0

    def version(self, extra=()):
        self.n += 1
        _, out, _ = run(self.base, self.pkg, self.edits, self.out_root, stamp=f"20990101_00{self.n:02d}", extra=extra)
        return read_out(out)[3]

    def test_person_no_env(self):
        v = self.version()
        self.assertEqual((v["created_by"], v["created_via"]), (None, None))
        self.assertNotIn("via", v)
        v = self.version(["--who", "au1"])
        self.assertEqual((v["created_by"], v["created_via"]), ("au1", None))

    def test_agent_env_and_flag(self):
        with mock.patch.dict(os.environ, {"CRYOEV_VIA": "claude", "CRYOEV_FOR": "pujan"}):
            v = self.version()
            self.assertEqual((v["created_by"], v["created_via"]), ("pujan", "claude"))
            self.assertNotIn("via", v)
            v = self.version(["--who", "au1"])                                  # the flag beats CRYOEV_FOR
            self.assertEqual((v["created_by"], v["created_via"]), ("au1", "claude"))

    def test_refusals_write_nothing(self):
        for env, extra in (({}, ["--who", "nobody"]), ({"CRYOEV_FOR": "nobody"}, []),
                           ({"CRYOEV_VIA": "Claude for pujan"}, [])):
            with mock.patch.dict(os.environ, env), self.assertRaises(SystemExit):
                self.version(extra)
        self.assertFalse(self.out_root.exists())


if __name__ == "__main__":
    unittest.main()
