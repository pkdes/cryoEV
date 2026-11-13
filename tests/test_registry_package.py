"""
Round 11 #1: `registry` is a real package, imported one way (`from registry import store`, or the module needed), with
one module object per file in a process.

Guards:
  1. Static: no .py file in the repo loads a registry file by path (spec_from_file_location, or a generic loader
     given a registry path), puts registry/ on sys.path (a "registry" path, REG / REG_DIR, a registry script's own
     folder), or imports a registry module by its bare name (that only works with registry/ on sys.path, where
     registry/registry.py, the script shim, shadows the package). Code strings for child processes are lines too.
  2. Every registry script with a __main__ block carries the preamble that drops its own folder from sys.path.
  3. Each such script runs by path (--help) from the repo, from registry/ and from a temp folder, and resolves the
     package (registry/__init__.py), not the shim.
  4. One object: every accessor the code uses (portal Train/Jobs, jobs/predict_set, the CLI, dataset_from_lv,
     importers, convert_raw) returns the same registry.store / registry.importers module.
  5. Patch guard (portal/tests/test_predict_patch_guard's rules over the registry modules): every patch of a registry
     module in the test suites (mock.patch.object / multiple / "registry.x.name", and `R.name = ...`) targets a name
     a registry function looks up through that module, and no other registry module from-imports it and uses it bare.

Controls (known answers): each forbidden pattern on a synthetic line is flagged; registry/ on sys.path really
shadows the package (so the preamble is load-bearing); the collector sees the known patches; a broken copy of
importers that from-imports a patched store name is flagged.
"""
from __future__ import annotations

import ast
import importlib
import os
import re
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "portal" / "tests"))
import test_predict_patch_guard as PG  # noqa: E402  (lookups / Member / problems: the same rules)
sys.path.append(str(REPO / "tests"))
import cryoev_testkit as TK  # noqa: E402  (shared helpers)

SKIP_DIRS = {".git", ".r10wt", ".r11wt", "__pycache__", "node_modules", ".venv", "venv"}
ME = Path(__file__).resolve()
# path (or folder prefix) -> why it may still match. Keep this short.
ALLOW = {
    "registry/tests/test_actor.py": "loads the actor shim by path on purpose (the shim's by-path promise)",
    "portal/tests/test_registry_one_module.py": "its control loads store.py by path on purpose (a known second copy)",
}
# a registry path: "registry" / ..., a registry/<file>.py string, or a variable named for the folder (REG, REG_DIR,
# reg_dir, registry_dir, REGISTRY_DIR; REG.parent is the repo root and fine)
REG_PATH = (r"""(["']registry["']\s*/|\b(?i:(?:reg|registry)_?dir)\b|\bREG\b(?!\.parent)|registry[\\/]+\w+\.py)""")
MODS = r"(registry|importers|validate|history_page|ids|schema_check|actor)"
RULES = {
    "by-path loader": re.compile(r"(spec_from_file_location|SourceFileLoader|run_path)\s*\(.*" + REG_PATH +
                                 r"|\b\w*load\w*\s*\(\s*[\"'][^\"']*[\"']\s*,.*" + REG_PATH),
    # loading tricks around the shadowing: popping / setting / reading a bare registry name in sys.modules
    "sys.modules registry name": re.compile(r"sys\.modules(\.(pop|get|setdefault)\s*\(|\s*\[)\s*[\"']" + MODS +
                                            r"[\"']"),
    "registry/ on sys.path": re.compile(r"sys\.path\.(insert|append)\s*\(.*(" + REG_PATH +
                                        r"""|["'][^"']*registry["'])"""),
    "bare registry import": re.compile(
        r"""(^\s*|[;"']\s*)(import (registry|importers|validate|history_page|ids|schema_check|actor)\s*($|[#;,"']| as )"""
        r"""|from (importers|validate|history_page|ids|schema_check) import"""
        r"""|from registry import registry\b|import registry\.registry\b|from registry\.registry import)"""),
}
PREAMBLE = [
    'if __name__ == "__main__":     # run by path: registry/ off sys.path (it would shadow the package), the repo on it',
    "    import os as _os",
    "    _here = _os.path.dirname(_os.path.abspath(__file__))",
    '    sys.path[:] = [p for p in sys.path if _os.path.normcase(_os.path.abspath(p or ".")) != _os.path.normcase(_here)]',
    "    sys.path.insert(0, _os.path.dirname(_here))",
]
OWN_INSERT = re.compile(r"sys\.path\.(insert|append)\s*\((?!.*\b(_ROOT|REPO)\b)")   # the repo root is fine


def allowed(rel: str) -> bool:
    return any(rel == k or (k.endswith("/") and rel.startswith(k)) for k in ALLOW)


def py_files():
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f.endswith(".py"):
                p = Path(root) / f
                if p.resolve() != ME:
                    yield p


def scan_text(rel: str, text: str) -> list:
    """-> ["rel:line: rule: text"] for every forbidden line."""
    out = []
    own = rel.startswith("registry/") and not rel.startswith("registry/tests/")
    for i, line in enumerate(text.splitlines(), 1):
        for name, rx in RULES.items():
            if rx.search(line):
                out.append(f"{rel}:{i}: {name}: {line.strip()[:120]}")
        if own and OWN_INSERT.search(line) and line not in PREAMBLE:
            out.append(f"{rel}:{i}: sys.path edit in a registry module: {line.strip()[:120]}")
    return out


def main_scripts() -> list:
    return sorted(p for p in (REPO / "registry").glob("*.py")
                  if re.search(r"^if __name__ == .__main__.:", p.read_text(encoding="utf-8"), re.M))


def child_env() -> dict:
    return TK.clean_env(strip_all=True, PYTHONDONTWRITEBYTECODE="1")


class StaticGuard(unittest.TestCase):
    def test_no_registry_loader_insert_or_bare_import(self):
        bad = []
        for p in py_files():
            rel = p.relative_to(REPO).as_posix()
            if not allowed(rel):
                bad += scan_text(rel, p.read_text(encoding="utf-8", errors="ignore"))
        self.assertEqual(bad, [], "use `from registry import store` (or the module you need); see docs/ARCHITECTURE.md")

    def test_allowlist_entries_still_match(self):
        """A stale entry would hide a later regression: each allowlisted file that exists must still need it."""
        for k in ALLOW:
            paths = [REPO / k] if not k.endswith("/") else sorted((REPO / k).rglob("*.py"))
            hits = [h for p in paths if p.is_file()
                    for h in scan_text(p.relative_to(REPO).as_posix(), p.read_text(encoding="utf-8"))]
            if any(p.is_file() for p in paths):
                self.assertTrue(hits, f"{k} no longer needs its allowlist entry: remove it")

    def test_control_each_pattern_is_flagged(self):
        lines = {
            'spec = importlib.util.spec_from_file_location("cj", REPO / "registry" / "registry.py")': "by-path loader",
            'ids = _load("ids", REG_DIR / "ids.py")': "by-path loader",
            'return CM.load("cj_history_page", "registry/history_page.py")': "by-path loader",
            'sys.path.insert(0, str(REPO / "registry"))': "registry/ on sys.path",
            "sys.path.insert(0, str(REG))": "registry/ on sys.path",
            "code = \"import sys; sys.path.insert(0, 'registry'); import registry\"": "registry/ on sys.path",
            "import registry  # noqa: E402": "bare registry import",
            "    import importers": "bare registry import",
            "import history_page as hp": "bare registry import",
            "from validate import ENTITIES": "bare registry import",
            "from registry import registry as REG": "bare registry import",
            # the r11-apply style (applyreview.registry_modules before the merge): a reg_dir variable, a _load_path
            # helper, and sys.modules juggling so `import registry` inside importers finds the shim
            "        reg_dir = REPO / \"registry\"": None,
            "            sys.path.insert(0, str(reg_dir))": "registry/ on sys.path",
            '                imp = _load_path("importers", reg_dir / "importers.py")': "by-path loader",
            '            hp = _load_path("cj_apply_history_page", reg_dir / "history_page.py")': "by-path loader",
            '                sys.modules.pop("registry", None)': "sys.modules registry name",
            '                sys.modules["registry"] = saved_reg': "sys.modules registry name",
            '            imp = sys.modules.get("importers")': "sys.modules registry name",
            "    sys.path.append(str(REGISTRY_DIR))": "registry/ on sys.path",
        }
        for line, rule in lines.items():
            if rule is None:                      # context only (a plain assignment is not itself a load)
                continue
            got = scan_text("x/test_x.py", line)
            self.assertTrue(got and any(f": {rule}:" in g for g in got), (line, got))
        self.assertTrue(scan_text("registry/x.py", "sys.path.insert(0, str(HERE))"))
        for ok in ("from registry import store as registry", "from registry import importers, ids",
                   "sys.path.insert(0, str(REG.parent))", "sys.path.insert(0, str(REPO))",
                   "from data_utils import actor", 'spec_from_file_location("x", REPO / "jobs" / "sync.py")',
                   "rows = json.loads((REG / 'samples' / 'x.jsonl').read_text())",
                   "self.assertNotIn('registry', sys.modules)", "m = json.load(fh)",
                   'd = yaml.safe_load(REGISTRY.read_text())'):
            self.assertEqual(scan_text("x/test_x.py", ok), [], ok)
        self.assertEqual(scan_text("registry/x.py", "\n".join(PREAMBLE)), [])

    def test_every_registry_script_has_the_preamble(self):
        scripts = main_scripts()
        self.assertGreaterEqual({p.name for p in scripts}, {"registry.py", "store.py", "validate.py", "history_page.py"})
        for p in scripts:
            text = p.read_text(encoding="utf-8")
            self.assertIn("\n".join(PREAMBLE), text, p.name)
            first_pkg_import = re.search(r"^from registry", text, re.M).start()
            self.assertLess(text.find(PREAMBLE[0]), first_pkg_import, f"{p.name}: preamble after the package import")

    def test_package_layout(self):
        self.assertTrue((REPO / "registry" / "__init__.py").is_file())
        self.assertEqual((REPO / "registry" / "__init__.py").read_text(encoding="utf-8").strip(), "")


class ByPath(unittest.TestCase):
    """`python registry/<script>.py ...` keeps working (callers and docs keep these paths) and never shadows."""

    def run_py(self, args, cwd):
        return subprocess.run([sys.executable, *args], cwd=cwd, env=child_env(), capture_output=True, text=True,
                              timeout=120)

    def test_each_script_runs_by_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            for p in main_scripts():
                for cwd in (REPO, REPO / "registry", Path(tmp)):
                    with self.subTest(script=p.name, cwd=cwd.name):
                        r = self.run_py([str(p), "--help"], cwd)
                        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                        self.assertIn("usage:", r.stdout)

    def test_by_path_resolves_the_package(self):
        """As Python runs a script (its folder first on sys.path): `registry` must be the package afterwards."""
        code = ("import runpy, sys, os\n"
                "p = sys.argv[1]; sys.argv = [p, '--help']; sys.path.insert(0, os.path.dirname(p))\n"
                "try:\n    runpy.run_path(p, run_name='__main__')\nexcept SystemExit:\n    pass\n"
                "m = sys.modules['registry']\n"
                "print('PKG', os.path.basename(m.__file__), hasattr(m, '__path__'), "
                "os.path.dirname(p) in sys.path)\n")
        for p in main_scripts():
            with self.subTest(script=p.name):
                r = self.run_py(["-c", code, str(p)], REPO / "registry")
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                self.assertIn("PKG __init__.py True False", r.stdout)

    def test_control_registry_dir_on_sys_path_shadows(self):
        """Known answer: with registry/ first on sys.path, `registry` is the shim and the package import fails."""
        code = f"import sys; sys.path.insert(0, {str(REPO / 'registry')!r}); from registry import store"
        r = self.run_py(["-c", code], REPO)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("registry", r.stderr)

    def test_store_refusal_is_rc2_whichever_way_it_runs(self):
        """QC: store.py run as __main__ used to define its own RegistryError, so a refusal raised inside
        registry.store (via importers) escaped main() as a traceback. All three entry points now run the package's
        store.main(): the FORMAT min_reader refusal is a message and exit 2."""
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "FORMAT.json").write_text('{"min_reader": 99}', encoding="utf-8")
            for args in ([str(REPO / "registry" / "store.py")], ["-m", "registry.store"],
                         [str(REPO / "registry" / "registry.py")]):
                with self.subTest(args=args[-1]):
                    r = self.run_py([*args, "--root", tmp, "import"], REPO)
                    self.assertEqual(r.returncode, 2, r.stderr[-2000:])
                    self.assertIn("needs reader version 99", r.stderr)
                    self.assertNotIn("Traceback", r.stderr)

    def test_shim_runs_store_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.run_py([str(REPO / "registry" / "registry.py"), "--root", tmp, "list"], REPO)
            self.assertEqual((r.returncode, r.stdout, r.stderr), (0, "", ""))


class OneModule(unittest.TestCase):
    def test_every_accessor_returns_the_package_module(self):
        from registry import importers, store
        from cryoev import common as CM
        from cryoev.commands import register as REGCMD
        from jobs import predict_set as PS
        from portal.modules import jobs as JB, train as TR
        from data_utils import dataset_from_lv as DFL
        self.assertIs(importers.registry, store)
        self.assertIs(TR.registry_module(), store)
        self.assertIs(JB._registry_module(), store)
        self.assertIs(PS.registry_module(), store)
        self.assertIs(CM.registry(), store)
        self.assertIs(DFL._reg(), store)
        self.assertIs(REGCMD._importers(), importers)
        self.assertIs(REGCMD._history(), importlib.import_module("registry.history_page"))
        self.assertIs(DFL.ids, importlib.import_module("registry.ids"))
        self.assertTrue(JB.valid_job_id("job_20261113_x"))
        self.assertIs(JB._ids_mod, importlib.import_module("registry.ids"))

    def test_convert_raw_uses_the_package_importers(self):
        code = ("import sys; sys.path.insert(0, 'jobs'); import convert_raw as CR; from registry import importers; "
                "assert CR.load_importers() is importers; print('same')")
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=child_env(), capture_output=True, text=True,
                           timeout=120)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "same"), r.stderr[-2000:])

    def test_shim_loaded_by_path_shares_store_state(self):
        """portal/modules/data.py loads registry/registry.py by path until the merge: its functions are store's own."""
        import importlib.util
        from registry import store
        spec = importlib.util.spec_from_file_location("shim_by_path_t", REPO / "registry" / "registry.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertIs(mod.current, store.current)
        self.assertIs(mod.RegistryError, store.RegistryError)


# ---------------------------------------------------------------- patch guard over the registry modules
FAMILY = ["registry.store", "registry.importers", "registry.validate", "registry.history_page", "registry.ids",
          "registry.schema_check"]
ACCESSORS = {"registry_module", "_registry_module", "_reg"}          # functions returning registry.store (OneModule)
SUITE_DIRS = ["jobs/tests", "tests", "registry/tests", "results/tests", "tools/tests", "annotation/tests",
              "data_utils/tests", "portal/tests", "cryoev/tests"]


def family() -> dict:
    out = {}
    for n in FAMILY:
        m = importlib.import_module(n)
        out[n] = PG.Member(n, m, Path(m.__file__).read_text(encoding="utf-8"))
    return out


def _aliases(tree) -> dict:
    """Names bound to a registry module anywhere in a test file: `from registry import x [as y]`,
    `import registry.x as y`, and `y = <...>.registry_module()` style accessors (registry.store)."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "registry":
            for a in node.names:
                if f"registry.{a.name}" in FAMILY:
                    out[a.asname or a.name] = importlib.import_module(f"registry.{a.name}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in FAMILY and a.asname:
                    out[a.asname] = importlib.import_module(a.name)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            f = node.value.func
            fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if fname in ACCESSORS:
                out[node.targets[0].id] = importlib.import_module("registry.store")
    return out


def _resolve(expr, aliases):
    chain = []
    while isinstance(expr, ast.Attribute):
        chain.append(expr.attr)
        expr = expr.value
    if not isinstance(expr, ast.Name) or expr.id not in aliases:
        return None
    obj = aliases[expr.id]
    for a in reversed(chain):
        obj = getattr(obj, a, None)
        if obj is None:
            return None
    return obj


def collect(paths) -> list:
    """[(target, attribute, "file:line")] for every patch / attribute assignment on a registry module (or an object
    reached through one)."""
    out = []
    for path in paths:
        try:
            tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        aliases = _aliases(tree)
        for node in ast.walk(tree):
            where = f"{Path(path).name}:{getattr(node, 'lineno', 0)}"
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Attribute):
                        obj = _resolve(t.value, aliases)
                        if obj is not None:
                            out.append((obj, t.attr, where))
                continue
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if PG._is_patch(node.func, "object") and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                obj = _resolve(node.args[0], aliases)
                if obj is not None:
                    out.append((obj, node.args[1].value, where))
            elif PG._is_patch(node.func, "multiple"):
                obj = _resolve(node.args[0], aliases)
                if obj is not None:
                    out += [(obj, k.arg, where) for k in node.keywords if k.arg]
            elif PG._is_patch(node.func, None) and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str) and node.args[0].value.startswith("registry."):
                mod, _, attr = node.args[0].value.rpartition(".")
                out.append((importlib.import_module(mod), attr, where))
    return out


def test_files() -> list:
    return sorted(p for d in SUITE_DIRS for p in (REPO / d).glob("test*.py"))


class Callers:
    """A pseudo family member: the repo's own (non-test) lookups `<alias>.name` of registry modules, wherever the alias
    is bound. dis sees a function-level `from registry import importers` as a local, so store.main's
    `importers.run(...)` is only found this way."""

    def __init__(self):
        self.name, self.names, self.attrs = "<repo code>", set(), set()
        self.module = types.ModuleType("_registry_callers")
        for p in py_files():
            rel = p.relative_to(REPO).as_posix()
            if "/tests/" in f"/{rel}" or allowed(rel):
                continue
            try:
                tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            aliases = _aliases(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
                    key = "_m_" + aliases[node.value.id].__name__.replace(".", "_")
                    setattr(self.module, key, aliases[node.value.id])
                    self.attrs.add((key, node.attr))

    def aliases_of(self, target) -> set:
        return {k for k, v in vars(self.module).items() if v is target}


class PatchGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fam = family()
        cls.fam["<repo code>"] = Callers()
        cls.patches = collect(test_files())

    def test_no_patch_misses(self):
        self.assertEqual(PG.problems(self.fam, self.patches), [])

    def test_the_collector_sees_the_patches(self):
        """Control: an empty or partial list would pass vacuously (QC 2: LOCK_WAIT, new_job, R.roster)."""
        got = {(getattr(t, "__name__", "?"), a) for t, a, _ in self.patches}
        for want in [("registry.store", "LOCK_WAIT"), ("registry.store", "new_job"), ("registry.store", "roster"),
                     ("registry.importers", "UPLOAD_LOCK_WAIT"), ("registry.importers", "run")]:
            self.assertIn(want, got)

    def test_broken_copy_of_importers_is_flagged(self):
        """Negative control: importers with store's LOCK_WAIT from-imported and used bare misses a store patch."""
        name = "registry.importers"
        src = Path(self.fam[name].module.__file__).read_text(encoding="utf-8")
        bad = src.replace("from registry import ids\n", "from registry import ids\nfrom registry.store import LOCK_WAIT\n",
                          1) + "\n\ndef _wait_broken():\n    return LOCK_WAIT\n"
        mod = types.ModuleType(name + "_broken")
        exec(compile(bad, name + "_broken", "exec"), mod.__dict__)
        fam = dict(self.fam)
        fam[name] = PG.Member(name, mod, bad)
        got = PG.problems(fam, self.patches)
        self.assertTrue(got and all("LOCK_WAIT" in g and "registry.importers binds its own" in g for g in got), got)


if __name__ == "__main__":
    unittest.main()
