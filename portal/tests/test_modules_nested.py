"""The module loader's nested PREFIX (round 13: /reviews/start under /reviews): allowed one segment under another
loaded module's PREFIX, its routes first (so the parent's patterns never shadow it), its own ROLES deciding under it.
Refused: a nested prefix without its parent module, or under a reserved core prefix, or two segments deep.
Control: the real modules load, and /reviews/start/ is routed to reviewstart, before /reviews' routes.

  python -m unittest portal.tests.test_modules_nested -v
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_modules_loader import fake  # noqa: E402
from portal import modules as M  # noqa: E402


class Nested(unittest.TestCase):
    def tearDown(self):
        for k in [k for k in sys.modules if k.startswith("portal.modules.zz")]:
            del sys.modules[k]

    def test_nested_under_a_parent(self):
        parent = fake("zzp", "/zzp", ("admin", "advanced", "basic"), ["/zzp/", "/zzp/{pkg}"])
        child = fake("zzq", "/zzp/kid", ("admin",), ["/zzp/kid/", "/zzp/kid/{name}"])
        routes, rules = M.load(M.Ctx(cfg=None), names=[parent, child])
        paths = [r.path for r in routes]
        self.assertLess(paths.index("/zzp/kid"), paths.index("/zzp/{pkg}"))   # the child's redirect wins
        self.assertLess(paths.index("/zzp/kid/{name}"), paths.index("/zzp/"))
        self.assertTrue(M.allowed(rules, "/zzp/x", "basic"))
        self.assertFalse(M.allowed(rules, "/zzp/kid/", "basic"))                # its own ROLES under it
        self.assertTrue(M.allowed(rules, "/zzp/kid/", "admin"))
        self.assertTrue(M.allowed(rules, "/zzp/kidney", "basic"))               # per path segment

    def test_refused(self):
        cases = [("zzr", "/zzr/kid", ["/zzr/kid/"], []),                          # no parent module
                 ("zzs", "/review/kid", ["/review/kid/"], []),                    # under a core prefix
                 ("zzt", "/zzt/a/b", ["/zzt/a/b/"], ["zzu"])]                     # two segments deep
        fake("zzu", "/zzt", ("admin",), ["/zzt/"])
        for name, prefix, paths, extra in cases:
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                M.load(M.Ctx(cfg=None), names=extra + [fake(name, prefix, ("admin",), paths)])

    def test_roles_within_parent_and_no_parent_route_inside(self):
        parent = fake("zzv", "/zzv", ("admin", "advanced"), ["/zzv/"])
        wider = fake("zzw", "/zzv/kid", ("admin", "basic"), ["/zzv/kid/"])     # basic: the parent keeps it out
        with self.assertRaises(ValueError) as cm:
            M.load(M.Ctx(cfg=None), names=[parent, wider])
        self.assertIn("ROLES must be within", str(cm.exception))
        for paths in (["/zzx/", "/zzx/kid"], ["/zzx/", "/zzx/kid/x"]):            # the parent routes into it
            with self.subTest(paths=paths):
                par = fake("zzx", "/zzx", ("admin",), paths)
                kid = fake("zzy", "/zzx/kid", ("admin",), ["/zzx/kid/"])
                with self.assertRaises(ValueError) as cm:
                    M.load(M.Ctx(cfg=None), names=[par, kid])
                self.assertIn("fall inside", str(cm.exception))
        par = fake("zzx", "/zzx", ("admin",), ["/zzx/", "/zzx/kidney"])           # control: another segment is fine
        kid = fake("zzy", "/zzx/kid", ("admin",), ["/zzx/kid/"])
        M.load(M.Ctx(cfg=None), names=[par, kid])

    def test_real_modules(self):
        import cjtest as T
        env = T.Env()
        self.addCleanup(env.cleanup)
        routes, rules = M.load(M.Ctx(cfg=env.config()))
        paths = [getattr(r, "path", None) for r in routes]
        self.assertIn("/reviews/start/", paths)
        self.assertLess(paths.index("/reviews/start/{name}"), paths.index("/reviews/"))
        self.assertTrue(M.allowed(rules, "/reviews/start/", "basic"))


if __name__ == "__main__":
    unittest.main()
