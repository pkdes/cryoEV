"""
Patch guard for the split Data module (portal/modules/data/: __init__ re-exports + records, details, lineage, graphs,
render, routes), built on test_predict_patch_guard's lookup reader.

Before round 11 data.py was one module, so `D.X = v` or `mock.patch.object(D, "X")` reached every lookup of X. Now
the package only RE-EXPORTS X: setting it on the package changes that one binding, which outside modules read
(`DATA.read_registry` in jobs.py) but the parts never do, so a test would silently stop testing what it says. For every
assignment (`M.X = v`, `M.X += v`, `setattr` / `monkeypatch.setattr`) and patch (`mock.patch.object`, `patch.multiple`,
`mock.patch("portal.modules.data...X")`) in portal/tests whose target is the package or one of its parts:

  1. the package, a re-exported X, assigned: always flagged (set it on the part that defines it);
  2. the package, a re-exported X, patched: flagged unless the same test function also patches X on every part
     that binds X and looks it up bare (so the patches together reach what the one data.py patch did);
  3. a part: test_predict_patch_guard's rules (X is looked up there by some part's function; no other part binds X
     and looks it up bare).

Known-answer controls: a patch of a re-exported name on the package really does not reach the part, and synthetic test
files with each bad pattern are flagged while the good ones pass.

  python -m unittest portal.tests.test_data_patch_guard -v
"""
import ast
import importlib
import pkgutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_predict_patch_guard as PG  # noqa: E402

PKG = "portal.modules.data"


def family() -> dict:
    """{module name: PG.Member} for the package and each of its parts."""
    pkg = importlib.import_module(PKG)
    out = {}
    for n in [PKG] + [f"{PKG}.{i.name}" for i in pkgutil.iter_modules(pkg.__path__)]:
        m = importlib.import_module(n)
        out[n] = PG.Member(n, m, Path(m.__file__).read_text(encoding="utf-8"))
    return out


def reexports() -> set:
    """Names __init__ imports from its parts."""
    pkg = importlib.import_module(PKG)
    tree = ast.parse(Path(pkg.__file__).read_text(encoding="utf-8"))
    return {a.asname or a.name for node in tree.body if isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith(PKG + ".") for a in node.names}


# ---------------------------------------------------------------- what the tests assign and patch
def _aliases(tree) -> dict:
    """Every import of a portal module in a test file (any depth), and `X = importlib.import_module("portal...")`."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("portal"):
            for a in node.names:
                try:
                    out[a.asname or a.name] = importlib.import_module(f"{node.module}.{a.name}")
                except ImportError:
                    pass
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("portal") and a.asname:
                    out[a.asname] = importlib.import_module(a.name)
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
              and node.value.func.attr == "import_module" and node.value.args
              and isinstance(node.value.args[0], ast.Constant) and str(node.value.args[0].value).startswith("portal")):
            out[node.targets[0].id] = importlib.import_module(node.value.args[0].value)
    return out


def _resolve(expr, aliases):
    """-> the module object for a Name / dotted Attribute rooted in an alias, when it is the package or a part."""
    chain = []
    while isinstance(expr, ast.Attribute):
        chain.append(expr.attr)
        expr = expr.value
    if not isinstance(expr, ast.Name) or expr.id not in aliases:
        return None
    obj = aliases[expr.id]
    for a in reversed(chain):
        obj = getattr(obj, a, None)
    name = getattr(obj, "__name__", None)
    return obj if isinstance(name, str) and (name == PKG or name.startswith(PKG + ".")) and \
        type(obj).__name__ == "module" else None


def _is_setattr(func) -> bool:
    return (isinstance(func, ast.Name) and func.id == "setattr") or \
        (isinstance(func, ast.Attribute) and func.attr == "setattr")


def collect(paths) -> list:
    """[(module, name, "assign" | "patch", "file:line", enclosing function node or None)]."""
    out = []
    for path in paths:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        aliases = _aliases(tree)
        parents = {}
        for node in ast.walk(tree):
            for ch in ast.iter_child_nodes(node):
                parents[ch] = node

        def fn_of(node):
            while node in parents and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node = parents[node]
            return node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None

        for node in ast.walk(tree):
            where = f"{Path(path).name}:{getattr(node, 'lineno', '?')}"
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                    for tt in (t.elts if isinstance(t, ast.Tuple) else [t]):
                        if isinstance(tt, ast.Attribute):
                            obj = _resolve(tt.value, aliases)
                            if obj is not None:
                                out.append((obj, tt.attr, "assign", where, fn_of(node)))
                continue
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if _is_setattr(node.func) and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                obj = _resolve(node.args[0], aliases)
                if obj is not None:
                    out.append((obj, node.args[1].value, "assign", where, fn_of(node)))
            elif PG._is_patch(node.func, "object") and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                obj = _resolve(node.args[0], aliases)
                if obj is not None:
                    out.append((obj, node.args[1].value, "patch", where, fn_of(node)))
            elif PG._is_patch(node.func, "multiple"):
                obj = _resolve(node.args[0], aliases)
                if obj is not None:
                    out += [(obj, k.arg, "patch", where, fn_of(node)) for k in node.keywords if k.arg]
            elif PG._is_patch(node.func, None) and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str) and node.args[0].value.startswith(PKG + "."):
                mod, _, attr = node.args[0].value.rpartition(".")
                obj = sys.modules.get(mod) or importlib.import_module(mod)
                out.append((obj, attr, "patch", where, fn_of(node)))
    return out


def problems(fam: dict, items: list, exported: set) -> list:
    out = []
    pkg = fam[PKG].module
    patched_in = {}                                   # id(function node) -> {(id(module), name)}
    for obj, name, how, _w, fn in items:
        if how == "patch":
            patched_in.setdefault(id(fn), set()).add((id(obj), name))
    for obj, name, how, where, fn in items:
        if obj is pkg:
            if name not in exported:
                continue                              # defined in __init__ itself (ROLES): one binding, nothing to miss
            if how == "assign":
                out.append(f"{where}: {name} is assigned on the package {PKG}, a re-export the parts never read: "
                           f"set it on the part that defines it")
                continue
            bare = sorted(n for n, m in fam.items() if n != PKG and name in vars(m.module) and name in m.names)
            have = patched_in.get(id(fn), set())
            missing = [n for n in bare if (id(fam[n].module), name) not in have]
            if missing:
                out.append(f"{where}: {name} is patched on the package {PKG} only, but {', '.join(missing)} "
                           f"look(s) it up bare (the patch misses it): patch it on that part too")
        else:
            out += PG.problems(fam, [(obj, name, where)])
    return out


def test_files() -> list:
    return sorted(HERE.glob("test_*.py"))


# ---------------------------------------------------------------- the checks
class RealTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fam = family()
        cls.exported = reexports()
        cls.items = collect(test_files())

    def test_no_patch_or_assignment_misses(self):
        self.assertEqual(problems(self.fam, self.items, self.exported), [])

    def test_the_collector_sees_the_known_sites(self):
        """Control: an empty or partial list would pass vacuously."""
        got = [(o.__name__, n, h) for o, n, h, _w, _f in self.items]
        self.assertEqual(got.count((f"{PKG}.lineage", "LINEAGE_MAX_DEPTH", "assign")), 4)     # test_data_module
        self.assertGreaterEqual(got.count((PKG, "read_registry", "patch")), 1)                 # test_vis
        self.assertGreaterEqual(got.count((f"{PKG}.routes", "read_registry", "patch")), 1)

    def test_the_family_is_complete(self):
        self.assertEqual(sorted(self.fam), [PKG] + [f"{PKG}.{p}" for p in
                                                    ("details", "graphs", "lineage", "records", "render", "routes")])
        for n, m in self.fam.items():
            if n != PKG:
                self.assertTrue(m.names, n)
        self.assertGreaterEqual(len(self.exported), 60)
        self.assertIn("read_registry", self.exported)
        self.assertNotIn("ROLES", self.exported)

    def test_the_pre_split_tests_are_flagged(self):
        """Negative control on real files: test_data_module / test_vis as they were before the re-targeting."""
        dm = (HERE / "test_data_module.py").read_text(encoding="utf-8")
        vis = (HERE / "test_vis.py").read_text(encoding="utf-8")
        self.assertEqual(dm.count("DL.LINEAGE_MAX_DEPTH = "), 4)
        self.assertIn('mock.patch.object(DR, "read_registry", fake)', vis)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "test_data_module.py").write_text(dm.replace("DL.LINEAGE_MAX_DEPTH", "D.LINEAGE_MAX_DEPTH"),
                                                           encoding="utf-8")
            (Path(tmp) / "test_vis.py").write_text(vis.replace(', mock.patch.object(DR, "read_registry", fake)', ""),
                                                   encoding="utf-8")
            got = problems(self.fam, collect(sorted(Path(tmp).glob("*.py"))), self.exported)
        self.assertEqual(sum("LINEAGE_MAX_DEPTH is assigned on the package" in g for g in got), 4, got)
        self.assertEqual(sum("read_registry is patched on the package" in g and "routes" in g for g in got), 1, got)
        self.assertEqual(len(got), 5, got)

    def test_a_package_patch_really_misses_the_part(self):
        """Known answer (why the guard exists): patching the re-export leaves the part's own binding alone."""
        pkg, lineage = importlib.import_module(PKG), importlib.import_module(f"{PKG}.lineage")
        before = lineage.LINEAGE_MAX_DEPTH
        with mock.patch.object(pkg, "LINEAGE_MAX_DEPTH", before + 1):
            self.assertEqual(lineage.LINEAGE_MAX_DEPTH, before)
        with mock.patch.object(lineage, "LINEAGE_MAX_DEPTH", before + 1):
            self.assertEqual(lineage.LINEAGE_MAX_DEPTH, before + 1)


class Synthetic(unittest.TestCase):
    """Each pattern on the real package, in a throwaway test file, with a known answer."""

    HEAD = ("import importlib\nfrom unittest import mock\nfrom portal.modules import data as D\n"
            "from portal.modules.data import lineage as DL\nDR = importlib.import_module('portal.modules.data.routes')\n"
            "DREC = importlib.import_module('portal.modules.data.records')\n")

    @classmethod
    def setUpClass(cls):
        cls.fam, cls.exported = family(), reexports()

    def check(self, body):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_fixture_data_patches.py"
            path.write_text(self.HEAD + body, encoding="utf-8")
            items = collect([path])
        self.assertTrue(items, "the collector saw nothing")
        return problems(self.fam, items, self.exported)

    def test_assignment_on_the_package_is_flagged(self):
        for body in ("def t():\n    D.LINEAGE_MAX_DEPTH = 2\n",
                     "def t():\n    D.LINEAGE_MAX_DEPTH += 1\n",
                     "def t():\n    setattr(D, 'LINEAGE_MAX_DEPTH', 2)\n",
                     "def t(monkeypatch):\n    monkeypatch.setattr(D, 'read_registry', None)\n"):
            with self.subTest(body=body):
                got = self.check(body)
                self.assertEqual(len(got), 1, got)
                self.assertIn("is assigned on the package", got[0])

    def test_assignment_on_the_defining_part_passes(self):
        self.assertEqual(self.check("def t():\n    DL.LINEAGE_MAX_DEPTH = 2\n"), [])

    def test_package_only_patch_is_flagged(self):
        for body in ("def t():\n    with mock.patch.object(D, 'read_registry', None):\n        pass\n",
                     "def t():\n    with mock.patch('portal.modules.data.read_registry'):\n        pass\n",
                     "def t():\n    with mock.patch.multiple(D, LINEAGE_MAX_DEPTH=1):\n        pass\n"):
            with self.subTest(body=body):
                got = self.check(body)
                self.assertEqual(len(got), 1, got)
                self.assertIn("patched on the package", got[0])

    def test_package_patch_paired_with_the_part_passes(self):
        """test_vis's pattern: jobs.py reads DATA.read_registry on the package, the routes part reads its own."""
        body = ("def t():\n    with mock.patch.object(D, 'read_registry', None), "
                "mock.patch.object(DR, 'read_registry', None):\n        pass\n")
        self.assertEqual(self.check(body), [])

    def test_patch_on_a_part_that_does_not_look_it_up_is_flagged(self):
        """records defines read_registry, but only routes calls it (with its own binding)."""
        got = self.check("def t():\n    with mock.patch.object(DREC, 'read_registry', None):\n        pass\n")
        self.assertTrue(got)
        self.assertTrue(any("routes binds its own read_registry" in g for g in got), got)

    def test_name_defined_in_init_is_fine(self):
        self.assertEqual(self.check("def t():\n    with mock.patch.object(D, 'ROLES', ()):\n        pass\n"), [])


if __name__ == "__main__":
    unittest.main()
