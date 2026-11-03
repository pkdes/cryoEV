"""Tests for data_utils/config.py: env overrides, a missing root raises on call, never on import.

    python -m unittest tests.test_config -v        (from the repo root)

"No siblings" is simulated with a temp folder (config.REPO patched, or a copy of data_utils/ and
registry/ under %TEMP%); the real data folders are never moved or written.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data_utils import config  # noqa: E402


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CRYOEV_")}
    env.update(extra)
    return env


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        if any(config.find_sibling(n, self.tmp) for n in (config.LAPTOP, config.CRYOAI, config.PROCESSING)):
            self.skipTest(f"a data folder sits above {self.tmp}")
        self.nowhere = mock.patch.object(config, "REPO", self.tmp)   # sibling search finds nothing

    def test_env_override(self):
        lap, cai, proc = (self.tmp / n for n in ("lap", "cai", "proc"))
        for p in (lap, cai, proc):
            p.mkdir()
        env = clean_env(CRYOEV_LAPTOP_ROOT=str(lap), CRYOEV_CRYOAI_ROOT=str(cai), CRYOEV_PROCESSING_ROOT=str(proc))
        with mock.patch.dict(os.environ, env, clear=True), self.nowhere:
            self.assertEqual(config.laptop_root(), lap)
            self.assertEqual(config.cryoai_root(), cai)
            self.assertEqual(config.processing_root(), proc)
            self.assertEqual(config.registry_root(), lap / "registry")
            self.assertIn(proc, config.protected_roots())
            self.assertEqual(config.protected_roots()[-1], Path.home() / "Box")
        with mock.patch.dict(os.environ, clean_env(CRYOEV_REGISTRY=str(self.tmp / "reg")), clear=True), self.nowhere:
            self.assertEqual(config.registry_root(), self.tmp / "reg")   # as given, may not exist yet

    def test_missing_root_raises_on_call(self):
        with mock.patch.dict(os.environ, clean_env(), clear=True), self.nowhere:
            for f in (config.laptop_root, config.cryoai_root, config.registry_root):
                with self.assertRaises(config.ConfigError):
                    f()
                self.assertIsNone(f(required=False))
            self.assertIsNone(config.processing_root())
            self.assertEqual(config.protected_roots(), (Path.home() / "Box",))
        with mock.patch.dict(os.environ, clean_env(CRYOEV_LAPTOP_ROOT=str(self.tmp / "missing")), clear=True):
            with self.assertRaises(config.ConfigError):
                config.laptop_root()

    def test_found_from_repo(self):
        """Control with a known answer: the real checkout (or a worktree) finds its sibling folders."""
        lap = config.find_sibling(config.LAPTOP)
        if lap is None:
            self.skipTest("CryoAI-laptop not found above the repo")
        with mock.patch.dict(os.environ, clean_env(), clear=True):
            self.assertEqual(config.laptop_root(), lap)
            self.assertEqual(config.registry_root(), lap / "registry")

    def test_no_raise_on_import(self):
        """A copy of data_utils/ and registry/ with no data folders above it: imports work, use raises."""
        for d in ("data_utils", "registry"):
            shutil.copytree(REPO / d, self.tmp / d, ignore=shutil.ignore_patterns("tests", "__pycache__", "samples"))
        code = ("import sys; sys.path.insert(0, 'registry'); "
                "import data_utils.config, data_utils.path_guard, history_page, importers, validate, registry; "
                "assert importers.PROCESSING is None\n"
                "try:\n    registry.default_root()\nexcept data_utils.config.ConfigError as e:\n    print('RAISED', e)")
        r = subprocess.run([sys.executable, "-c", code], cwd=self.tmp, env=clean_env(),
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("RAISED CryoAI-laptop folder not found", r.stdout)


if __name__ == "__main__":
    unittest.main()
