"""
Round 11 #1 control: a portal process holds ONE module object per registry file after the Data (list and Lab graph),
Jobs and Train pages have read the registry and Apply, Predict and Admin were opened (before: registry/registry.py was
loaded by path 3 times, as cj_data_registry, cj_jobs_registry and cj_train_registry, each with its own LOCK_WAIT,
caches and patches).

Counted from the garbage collector, not sys.modules: a by-path loader keeps its module without registering it.
Temp world only (cjtest.Env); never the real registry.
"""
import gc
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402

REG_DIR = (T.REPO / "registry").resolve()


def registry_modules() -> dict:
    """{registry file name: [module names]} over every live module object."""
    out = {}
    for o in gc.get_objects():
        if isinstance(o, types.ModuleType):
            f = getattr(o, "__file__", None)
            if f and Path(f).resolve().parent == REG_DIR:
                out.setdefault(Path(f).name, []).append(o.__name__)
    return out


class OneRegistryModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cj_onemod_"))
        cls.saved = {k: os.environ.get(k) for k in ("CRYOEV_LAPTOP_ROOT", "CRYOEV_PROCESSING_ROOT")}
        for k, d in (("CRYOEV_LAPTOP_ROOT", "lap"), ("CRYOEV_PROCESSING_ROOT", "proc")):
            (cls.tmp / d).mkdir()
            os.environ[k] = str(cls.tmp / d)
        cls.env = T.Env()
        for f in (T.REPO / "registry" / "samples").glob("*.jsonl"):
            (cls.env.registry / f.name).write_bytes(f.read_bytes())
        cls.portal = T.Portal(cls.env.config())
        cls.cookie = T.cookie(T.login(cls.portal.port, cls.env.key))

    @classmethod
    def tearDownClass(cls):
        cls.portal.stop()
        cls.env.cleanup()
        for k, v in cls.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_one_module_object_per_registry_file(self):
        for path in ("/data/", "/data/lineage/graph", "/jobs/", "/train/"):
            st, _, _ = T.request(self.portal.port, "GET", path, self.cookie)
            self.assertEqual(st, 200, path)
        for path in ("/apply/", "/predict/", "/admin/"):       # pages whose modules may load the registry too
            st, _, _ = T.request(self.portal.port, "GET", path, self.cookie)
            self.assertLess(st, 500, path)
        gc.collect()
        mods = registry_modules()
        self.assertIn("store.py", mods, "control: the pages must have read the registry")
        self.assertEqual(mods["store.py"], ["registry.store"])
        self.assertIn("registry", sys.modules)
        self.assertTrue(hasattr(sys.modules["registry"], "__path__"), "registry must be the package")
        self.assertIn("history_page.py", mods, "control: the Lab graph must have loaded the history page module")
        for f, names in mods.items():
            self.assertLessEqual(len(names), 1, f"{f}: {names}")
            for n in names:
                self.assertEqual(n, "registry" if f == "__init__.py" else f"registry.{Path(f).stem}", json.dumps(mods))

    def test_control_the_counter_sees_a_by_path_copy(self):
        """Known answer: a by-path load of store.py is a second object, and the counter finds it."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("cj_onemod_copy", REG_DIR / "store.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertIn("cj_onemod_copy", registry_modules().get("store.py", []))
        del mod, spec
        gc.collect()


if __name__ == "__main__":
    unittest.main()
