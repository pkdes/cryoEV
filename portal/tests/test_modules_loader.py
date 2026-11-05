"""Module plug-in loader + role rules (BUILD_CONTRACTS.md section 1)."""
import sys
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from starlette.responses import PlainTextResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

from portal import modules as M  # noqa: E402


def fake(name, prefix, roles, paths):
    mod = types.ModuleType(f"portal.modules.{name}")
    mod.PREFIX, mod.ROLES = prefix, roles

    async def ok(request):
        return PlainTextResponse("ok")
    mod.routes = lambda ctx: [Route(p, ok) for p in paths]
    sys.modules[f"portal.modules.{name}"] = mod
    return name


class Loader(unittest.TestCase):
    def tearDown(self):
        for k in [k for k in sys.modules if k.startswith("portal.modules.zz")]:
            del sys.modules[k]

    def test_load_and_rules(self):
        a = fake("zza", "/zz", ("admin", "advanced"), ["/zz/", "/zz/api/x"])
        b = fake("zzb", "/zz-basic", ("admin", "advanced", "basic"), ["/zz-basic/"])
        routes, rules = M.load(M.Ctx(cfg=None), names=[a, b])
        paths = [r.path for r in routes]
        self.assertIn("/zz", paths)                     # redirect added
        self.assertIn("/zz/api/x", paths)
        self.assertTrue(M.allowed(rules, "/zz/api/x", "advanced"))
        self.assertFalse(M.allowed(rules, "/zz/", "basic"))           # control: role not listed -> denied
        self.assertFalse(M.allowed(rules, "/zz", "basic"))
        self.assertTrue(M.allowed(rules, "/zz-basic/", "basic"))      # prefix match is per path segment
        self.assertTrue(M.allowed(rules, "/history/history.html", "basic"))   # core routes are not module rules

    def test_bad_declarations_fail_loudly(self):
        cases = [("zzc", "/Bad", ("admin",), ["/Bad/"]), ("zzd", "/review", ("admin",), ["/review/x"]),
                 ("zze", "/zze", ("root",), ["/zze/"]), ("zzf", "/zzf", ("admin",), ["/elsewhere/"]),
                 ("zzg", "/zzg", (), ["/zzg/"])]
        for name, prefix, roles, paths in cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    M.load(M.Ctx(cfg=None), names=[fake(name, prefix, roles, paths)])

    def test_can_see(self):
        self.assertTrue(M.Ctx.can_see({"id": "a", "role": "admin"}, "b"))
        self.assertTrue(M.Ctx.can_see({"id": "b", "role": "basic"}, "b"))
        self.assertFalse(M.Ctx.can_see({"id": "c", "role": "basic"}, "b"))
        self.assertFalse(M.Ctx.can_see({"id": "c", "role": "advanced"}, None))


if __name__ == "__main__":
    unittest.main()
