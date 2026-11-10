"""
Patch guard for the split Predict module (portal/modules/predict.py + portal/predict_parts/*.py).

Tests patch module globals (`mock.patch.object(M, "name")`, `mock.patch.multiple(M, ...)`, `mock.patch("mod.name")`).
A patch only bites when the code under test looks `name` up in M at call time. After the split a function that
moved to another part and kept its own binding (`from portal.predict_parts.x import name`) would silently miss the
patch, so for every such patch in portal/tests whose target is reached through the Predict family:

  1. reached: `name` is looked up inside some function of the family, either as a global of M itself or as
     `<alias of M>.name` (dis: LOAD_GLOBAL, or LOAD_GLOBAL alias + LOAD_ATTR / LOAD_METHOD name);
  2. no shadow: no other family module binds `name` as a global AND looks it up bare in its functions;
  3. a collaborator patched through the family (`P.PS.create`, `P.secrets.token_hex`): no family module has
     from-imported that object under `name` and looks it up bare.

Negative controls: synthetic modules (a from-import shadow, an unreached name, a from-imported collaborator) and a
deliberately broken copy of a real part must each be flagged; the real tree must be clean.

  python -m unittest portal.tests.test_predict_patch_guard -v
"""
import ast
import dis
import importlib
import pkgutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

MAIN = "portal.modules.predict"
PARTS = "portal.predict_parts"
MISSING = object()


# ---------------------------------------------------------------- what the family's functions look up
def lookups(source: str, filename: str) -> tuple:
    """-> (names loaded as globals inside functions, {(global, attribute)} loaded inside functions)."""
    names, attrs = set(), set()

    def walk(co, in_fn):
        prev = None
        for ins in dis.get_instructions(co):
            if in_fn and ins.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
                names.add(ins.argval)
            if (in_fn and ins.opname in ("LOAD_ATTR", "LOAD_METHOD") and prev is not None
                    and prev.opname == "LOAD_GLOBAL"):
                attrs.add((prev.argval, ins.argval))
            prev = ins
        for c in co.co_consts:
            if isinstance(c, types.CodeType):
                walk(c, True)

    walk(compile(source, filename, "exec"), False)
    return names, attrs


class Member:
    def __init__(self, name, module, source):
        self.name, self.module = name, module
        self.names, self.attrs = lookups(source, name)

    def aliases_of(self, target) -> set:
        return {k for k, v in vars(self.module).items() if v is target}


def family() -> dict:
    """{module name: Member} for predict.py and every portal/predict_parts module."""
    out = {}
    pkg = importlib.import_module(PARTS)
    names = [MAIN] + [f"{PARTS}.{i.name}" for i in pkgutil.iter_modules(pkg.__path__)]
    for n in names:
        m = importlib.import_module(n)
        out[n] = Member(n, m, Path(m.__file__).read_text(encoding="utf-8"))
    return out


def problems(fam: dict, patches: list) -> list:
    """patches: [(target object, attribute, where)] -> the rule breaks, as text."""
    by_id = {id(m.module): n for n, m in fam.items()}
    out = []
    for target, attr, where in patches:
        tname = by_id.get(id(target))
        if tname is not None:                                    # a family module itself
            reached = attr in fam[tname].names or any(
                (a, attr) in m.attrs for m in fam.values() for a in m.aliases_of(target))
            if not reached:
                out.append(f"{where}: {tname}.{attr} is patched but no family function looks it up there")
            for n, m in fam.items():
                if n != tname and attr in vars(m.module) and attr in m.names:
                    out.append(f"{where}: {attr} is patched on {tname} but {n} binds its own {attr} and looks it "
                               f"up bare (the patch misses it): patch the part that defines it, and reach it "
                               f"from the other parts as <PART>.{attr}")
        else:                                                    # a collaborator reached through the family
            want = getattr(target, attr, MISSING)
            for n, m in fam.items():
                if attr in m.names and want is not MISSING and vars(m.module).get(attr, MISSING) is want:
                    out.append(f"{where}: {getattr(target, '__name__', target)}.{attr} is patched but {n} "
                               f"from-imported it and looks it up bare (the patch misses it)")
    return out


# ---------------------------------------------------------------- the patches in portal/tests
def _aliases(tree) -> dict:
    """Module-level imports of portal modules in a test file: {local name: module object}."""
    out = {}
    for node in tree.body:
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
    return out


def _resolve(expr, aliases):
    """-> (object, rooted in the family) for P / P.PS / UP.secrets; (None, False) when not resolvable."""
    chain = []
    while isinstance(expr, ast.Attribute):
        chain.append(expr.attr)
        expr = expr.value
    if not isinstance(expr, ast.Name) or expr.id not in aliases:
        return None, False
    obj = aliases[expr.id]
    rooted = obj.__name__ == MAIN or obj.__name__.startswith(PARTS)
    for a in reversed(chain):
        obj = getattr(obj, a, None)
        if obj is None:
            return None, False
    return obj, rooted


def _is_patch(func, kind) -> bool:
    """func is <...>.patch.<kind> (kind: object / multiple) or, kind None, <...>.patch / patch."""
    if kind is None:
        return (isinstance(func, ast.Attribute) and func.attr == "patch") or \
            (isinstance(func, ast.Name) and func.id == "patch")
    return isinstance(func, ast.Attribute) and func.attr == kind and _is_patch(func.value, None)


def collect(paths) -> list:
    """[(target, attribute, "file:line")] for every patch whose target is reached through the Predict family."""
    out = []
    for path in paths:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        aliases = _aliases(tree)
        parents = {}
        for node in ast.walk(tree):
            for ch in ast.iter_child_nodes(node):
                parents[ch] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            where = f"{Path(path).name}:{node.lineno}"
            if _is_patch(node.func, "object") and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                obj, rooted = _resolve(node.args[0], aliases)
                if rooted:
                    out.append((obj, node.args[1].value, where))
            elif _is_patch(node.func, "multiple"):
                obj, rooted = _resolve(node.args[0], aliases)
                if not rooted:
                    continue
                keys = {k.arg for k in node.keywords if k.arg}
                fn = node
                while fn in parents and not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fn = parents[fn]
                for k in node.keywords:
                    if k.arg is None:                          # **{...}: every str key of the dicts in the function
                        for d in ast.walk(fn):
                            if isinstance(d, ast.Dict):
                                keys |= {c.value for c in d.keys if isinstance(c, ast.Constant)
                                         and isinstance(c.value, str) and hasattr(obj, c.value)}
                out += [(obj, k, where) for k in sorted(keys)]
            elif _is_patch(node.func, None) and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str) \
                    and node.args[0].value.startswith((MAIN + ".", PARTS + ".")):
                mod, _, attr = node.args[0].value.rpartition(".")
                out.append((importlib.import_module(mod), attr, where))
    return out


def test_files() -> list:
    return sorted(p for p in HERE.glob("test_*.py"))


# ---------------------------------------------------------------- the checks
class RealTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fam = family()
        cls.patches = collect(test_files())

    def test_no_patch_misses(self):
        self.assertEqual(problems(self.fam, self.patches), [])

    def test_the_collector_sees_the_patches(self):
        """Control: an empty or partial patch list would pass vacuously."""
        self.assertGreaterEqual(len(self.patches), 60)
        got = {(getattr(t, "__name__", "?"), a) for t, a, _ in self.patches}
        for want in [(f"{PARTS}.disk", "disk_free"), ("jobs.predict_set", "create"), (f"{PARTS}.common", "MAX_FILES"),
                     (f"{PARTS}.common", "CHUNK_BYTES"), (f"{PARTS}.common", "REFRESH_S"),
                     (f"{PARTS}.create", "check_active"), ("secrets", "token_hex")]:
            self.assertIn(want, got)

    def test_the_family_is_complete(self):
        self.assertIn(MAIN, self.fam)
        self.assertGreaterEqual(len(self.fam), 8)
        for m in self.fam.values():
            self.assertTrue(m.names, m.name)

    def test_broken_copy_of_a_real_part_is_flagged(self):
        """Negative control on the real code: uploads.py with CHUNK_BYTES from-imported and used bare."""
        name = f"{PARTS}.uploads"
        src = Path(self.fam[name].module.__file__).read_text(encoding="utf-8")
        self.assertIn("COMMON.CHUNK_BYTES", src)
        bad = src.replace("COMMON.CHUNK_BYTES", "CHUNK_BYTES").replace(
            "from portal.predict_parts import ", "from portal.predict_parts.common import CHUNK_BYTES\n"
                                                "from portal.predict_parts import ", 1)
        mod = types.ModuleType(name + "_broken")
        exec(compile(bad, name + "_broken", "exec"), mod.__dict__)
        fam = dict(self.fam)
        fam[name] = Member(name, mod, bad)
        got = problems(fam, self.patches)
        n_patches = sum(1 for t, a, _ in self.patches if a == "CHUNK_BYTES")
        self.assertGreaterEqual(n_patches, 1)
        self.assertEqual(sum(f"{name} binds its own CHUNK_BYTES" in g for g in got), n_patches, got)
        self.assertTrue(all("CHUNK_BYTES" in g for g in got), got)          # nothing else is flagged


class Synthetic(unittest.TestCase):
    """The rules on tiny modules with known answers."""

    def make(self, name, src):
        m = types.ModuleType(name)
        exec(compile(src, name, "exec"), m.__dict__)
        sys.modules[name] = m
        return Member(name, m, src)

    def setUp(self):
        p = mock.patch.dict(sys.modules)
        p.start()
        self.addCleanup(p.stop)
        self.ext = self.make("p7g_ext", "def create():\n    return 1\n")
        self.a = self.make("p7g_a", "X = 1\nY = 2\ndef foo():\n    return 1\ndef use():\n    return foo() + X\n")
        self.ref = self.make("p7g_ref", "import p7g_a as A\nimport p7g_ext as E\n"
                                        "def go():\n    return A.foo() + A.X + E.create()\n")
        self.shadow = self.make("p7g_shadow", "from p7g_a import foo\ndef go():\n    return foo()\n")
        self.ext_shadow = self.make("p7g_extshadow", "from p7g_ext import create\ndef go():\n    return create()\n")

    def fam(self, *members):
        return {m.name: m for m in members}

    def test_clean_family_passes(self):
        fam = self.fam(self.a, self.ref)
        self.assertEqual(problems(fam, [(self.a.module, "foo", "t"), (self.a.module, "X", "t"),
                                        (self.ext.module, "create", "t")]), [])

    def test_from_import_shadow_is_flagged(self):
        got = problems(self.fam(self.a, self.ref, self.shadow), [(self.a.module, "foo", "t")])
        self.assertEqual(len(got), 1)
        self.assertIn("p7g_shadow binds its own foo", got[0])

    def test_unreached_name_is_flagged(self):
        got = problems(self.fam(self.a, self.ref), [(self.a.module, "Y", "t")])
        self.assertEqual(len(got), 1)
        self.assertIn("no family function looks it up", got[0])

    def test_reached_only_through_an_alias_counts(self):
        """X is read by use() in a; foo by go() in ref only through A.foo: both reached."""
        b = self.make("p7g_b", "def unrelated():\n    return 0\n")
        fam = self.fam(b, self.ref)
        self.assertEqual(problems({**fam, "p7g_a": self.a}, [(self.a.module, "foo", "t")]), [])

    def test_from_imported_collaborator_is_flagged(self):
        got = problems(self.fam(self.a, self.ref, self.ext_shadow), [(self.ext.module, "create", "t")])
        self.assertEqual(len(got), 1)
        self.assertIn("p7g_extshadow from-imported it", got[0])

    def test_collector_reads_object_multiple_and_string_patches(self):
        src = ("from unittest import mock\nfrom portal.modules import predict as P\n"
               "def t():\n"
               "    with mock.patch.object(P, 'kick'), mock.patch.object(P.PS, 'create'):\n        pass\n"
               "    patch = {'REFRESH_S': 1, 'nope_not_a_name': 2}\n"
               "    with mock.patch.multiple(P, **patch), mock.patch.multiple(P, STALE_S=3):\n        pass\n"
               "    with mock.patch('portal.modules.predict.task'):\n        pass\n"
               "    with mock.patch.object(old, 'check_model'):\n        pass\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_fixture_patches.py"
            path.write_text(src, encoding="utf-8")
            got = sorted((getattr(t, "__name__", "?"), a) for t, a, _ in collect([path]))
        self.assertEqual(got, [("jobs.predict_set", "create"), (MAIN, "REFRESH_S"), (MAIN, "STALE_S"), (MAIN, "kick"),
                               (MAIN, "task")])


if __name__ == "__main__":
    unittest.main()
