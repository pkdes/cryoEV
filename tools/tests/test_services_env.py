"""services.py child environment (round 9): CRYOEV_VIA / CRYOEV_FOR never reach a service.

Known answer: started from an "agent session" (both variables set, and an entry that even sets one itself), the child
process sees neither; control: an ordinary entry variable and PYTHONUNBUFFERED still arrive, so the env dump works.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import services as S  # noqa: E402

AGENT_ENV = {"CRYOEV_VIA": "claude", "CRYOEV_FOR": "pujan"}
DUMP = ("import json,os,sys; json.dump({k: v for k, v in os.environ.items() if k.startswith(('CRYOEV_', 'SVC_'))"
        " or k == 'PYTHONUNBUFFERED'}, open(sys.argv[1], 'w'))")


class ChildEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def svc(self, env):
        out = self.root / "env.json"
        return S.Service(name="envdump", argv=[sys.executable, "-c", DUMP, str(out)], cwd=self.root,
                         log=self.root / "envdump.log", env=env), out

    def test_child_env_drops_identity_vars(self):
        svc, _ = self.svc({"SVC_X": "1", "CRYOEV_FOR": "au1"})
        env = S.child_env(svc, dict(AGENT_ENV, PATH="p", CRYOEV_LAPTOP_ROOT="lap"))
        self.assertNotIn("CRYOEV_VIA", env)
        self.assertNotIn("CRYOEV_FOR", env)                  # even when the entry's own env sets it
        self.assertEqual((env["SVC_X"], env["PYTHONUNBUFFERED"], env["CRYOEV_LAPTOP_ROOT"], env["PATH"]),
                         ("1", "1", "lap", "p"))               # control: everything else passes through

    def test_spawned_child_has_no_identity_vars(self):
        svc, out = self.svc({"SVC_X": "1"})
        with mock.patch.dict(os.environ, AGENT_ENV):
            self.assertEqual(os.environ["CRYOEV_VIA"], "claude")   # the parent does have them
            p = S.spawn(svc)
            self.assertEqual(os.environ.get("CRYOEV_VIA"), "claude")   # only the child's copy is cleaned
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            raise
        seen = json.loads(out.read_text())
        self.assertNotIn("CRYOEV_VIA", seen)
        self.assertNotIn("CRYOEV_FOR", seen)
        self.assertEqual((seen.get("SVC_X"), seen.get("PYTHONUNBUFFERED")), ("1", "1"))   # control


if __name__ == "__main__":
    unittest.main()
