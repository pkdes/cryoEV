"""
Results page, round 14 (contract C5): the "Analysis steps" box, the objects.csv download and its blind rule, and the
admin-only POST /results/<ps>/recipe.

The golden prediction set (tests/fixtures/cj_golden) is copied into a temp predict root and computed with the real
engine; the registry is the temp one from cjtest.Env with hand-written review_package lines (an open object set on
one prediction set). Temp roster / registry / predict root only; never the real ones.

  python -m unittest discover -s portal/tests -p test_results_steps.py
"""
import io
import json
import shutil
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import results as R  # noqa: E402
from portal.modules.data import records as DREC  # noqa: E402
from results.compute import compute_and_write  # noqa: E402

GOLDEN = T.REPO / "tests" / "fixtures" / "cj_golden"
GPS = "ps_20261105_abc123"
OWNER = "pujan"
DONE = "ps_20261114_b2a001"          # computed, default recipe
OLD = "ps_20261114_b2a002"           # results.json without steps.json (computed before round 14)
BLIND = "ps_20261114_b2a003"         # an open object set builds on it
TODO = "ps_20261114_b2a004"          # not computed (the recipe route runs the real engine on it)


def make_set(root: Path, ps_id: str, owner=OWNER) -> Path:
    d = root / ps_id
    shutil.copytree(GOLDEN / "predict" / GPS, d)
    req = json.loads((d / "request.json").read_text(encoding="utf-8"))
    req.update(ps=ps_id, owner=owner)
    (d / "request.json").write_text(json.dumps(req, indent=1), encoding="utf-8")
    return d


def pkg_line(pid, base, kind="objects", status="built"):
    return {"schema": "review_package/1", "id": pid, "kind": kind, "base": base, "status": status,
            "base_kind": "prediction_set", "created": "2026-11-14T10:00:00"}


def write_registry(root: Path, lines) -> None:
    (root / "review_package.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    DREC._registry_cache.clear()


class Helpers(unittest.TestCase):
    """open_object_sets / objects_restricted / may_get_objects on a fixture registry (no portal)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_steps_reg_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(DREC._registry_cache.clear)
        self.cfg = SimpleNamespace(registry=self.tmp)
        self.ps = {"id": "ps_20261114_aaaaaa", "request": {"owner": "own"}}

    def test_open_set_detection(self):
        write_registry(self.tmp, [pkg_line("pred_obj_a_0001", self.ps["id"]),                       # open
                                  pkg_line("pred_obj_b_0002", self.ps["id"], status="applied"),     # closed
                                  pkg_line("pred_obj_c_0003", "ps_20261114_bbbbbb"),                # other set
                                  pkg_line("pred_ps_20261114_aaaaaa", self.ps["id"], kind="prediction"),
                                  {**pkg_line("pred_obj_d_0004", self.ps["id"]), "status": "in_review"}])
        self.assertEqual(R.open_object_sets(self.cfg, self.ps["id"]), ["pred_obj_a_0001", "pred_obj_d_0004"])
        self.assertTrue(R.objects_restricted(self.cfg, self.ps))

    def test_closing_the_set_lifts_the_restriction(self):
        write_registry(self.tmp, [pkg_line("pred_obj_a_0001", self.ps["id"]),
                                  pkg_line("pred_obj_a_0001", self.ps["id"], status="reviewed")])   # last line wins
        self.assertEqual(R.open_object_sets(self.cfg, self.ps["id"]), [])
        self.assertFalse(R.objects_restricted(self.cfg, self.ps))

    def test_who_gets_objects(self):
        write_registry(self.tmp, [pkg_line("pred_obj_a_0001", self.ps["id"])])
        self.assertTrue(R.may_get_objects(self.cfg, self.ps, {"id": "boss", "role": "admin"}))
        self.assertTrue(R.may_get_objects(self.cfg, self.ps, {"id": "own", "role": "basic"}))
        self.assertFalse(R.may_get_objects(self.cfg, self.ps, {"id": "rev", "role": "advanced"}))
        self.assertNotIn("objects.csv", R.files_for(self.cfg, self.ps, {"id": "rev", "role": "advanced"}))
        self.assertIn("objects.csv", R.files_for(self.cfg, self.ps, {"id": "own", "role": "basic"}))
        write_registry(self.tmp, [])                                                # control: no set
        self.assertTrue(R.may_get_objects(self.cfg, self.ps, {"id": "rev", "role": "advanced"}))

    def test_fail_closed_and_no_registry(self):
        self.assertTrue(R.objects_restricted(SimpleNamespace(registry=self.tmp / "missing"), self.ps))
        self.assertFalse(R.objects_restricted(SimpleNamespace(registry=None), self.ps))

    def test_parse_recipe_form(self):
        self.assertEqual(R.parse_recipe_form([]), [("roles", 1)])
        self.assertEqual(R.parse_recipe_form(["nesting:1"]), [("roles", 1), ("nesting", 1)])
        for bad in (["roles:1"], ["nesting:9"], ["x"], ["nesting:one"]):
            with self.assertRaises(ValueError, msg=bad):
                R.parse_recipe_form(bad)


class Page(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.tmp = Path(tempfile.mkdtemp(prefix="cj_steps_"))
        cls.root, cls.review = cls.tmp / "predict", cls.tmp / "review"
        cls.root.mkdir()
        cls.review.mkdir()
        compute_and_write(make_set(cls.root, DONE))
        compute_and_write(make_set(cls.root, BLIND))
        compute_and_write(make_set(cls.root, OLD))
        (cls.root / OLD / "results" / "steps.json").unlink()
        (cls.root / OLD / "results" / "objects.csv").unlink()
        make_set(cls.root, TODO)
        write_registry(cls.env.registry, [pkg_line("pred_obj_lamel_ab12", BLIND)])
        cls.keys = {"admin": cls.env.key, "owner": A.add_key(cls.env.roster, OWNER, "basic")}
        cfg = cls.env.config()
        cfg.predict_root, cfg.review_root = cls.root, cls.review
        cls.cfg = cfg
        cls.p = T.Portal(cfg)
        cls.tok = {k: T.login(cls.p.port, v) for k, v in cls.keys.items()}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        DREC._registry_cache.clear()

    def setUp(self):
        R._jobs.clear()
        self.addCleanup(R._jobs.clear)

    def get(self, path, who="admin"):
        return T.request(self.p.port, "GET", path, T.cookie(self.tok[who]))

    def post(self, path, who="admin", body=b"", origin=True):
        h = T.cookie(self.tok[who])
        h["Content-Type"] = "application/x-www-form-urlencoded"
        if origin:
            h["Origin"] = self.p.origin
        return T.request(self.p.port, "POST", path, h, body=body)

    def page(self, ps, who="admin") -> str:
        st, _, body = self.get(f"/results/{ps}/", who)
        self.assertEqual(st, 200, body[:300])
        return body.decode("utf-8")

    def wait_done(self, ps, timeout=60):
        """Until the set's worker is gone (the POST registered it before answering 303)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with R._jobs_lock:
                js = [j for k, j in R._jobs.items() if Path(k).name == ps]
            if not js:
                st, _, body = self.get(f"/results/{ps}/api/status")
                self.assertEqual(json.loads(body)["state"], "ready")
                return
            if js[0]["state"] == "error":
                self.fail(js[0])
            time.sleep(0.1)
        self.fail("compute never finished")

    def test_panel_for_owner(self):
        page = self.page(DONE, "owner")
        self.assertIn('id="cj-results-steps"', page)
        self.assertIn("Analysis steps", page)
        self.assertIn('data-step="roles"', page)
        self.assertIn("Layer roles", page)
        self.assertIn(">v1<", page)
        self.assertIn("containment 0.99 · size ratio 0.75", page)
        self.assertIn("roles.role, roles.layer_count", page)
        self.assertIn("Recipe: roles v1 (default)", page)
        self.assertIn(f'href="/results/{DONE}/download/objects.csv"', page)
        self.assertIn("stamped with step versions", page)
        self.assertNotIn("Change recipe (admin)", page)                     # owner, not admin
        self.assertIn("Change recipe (admin)", self.page(DONE, "admin"))

    def test_old_results_say_update(self):
        page = self.page(OLD)
        self.assertIn("computed before analysis steps were recorded", page)
        self.assertNotIn(f"/results/{OLD}/download/objects.csv", page)

    def test_objects_csv_download(self):
        st, h, body = self.get(f"/results/{DONE}/download/objects.csv", "owner")
        self.assertEqual(st, 200)
        self.assertEqual(T.hget(h, "content-disposition"), f'attachment; filename="{DONE}_objects.csv"')
        disk = (self.root / DONE / "results" / "objects.csv").read_bytes()
        self.assertTrue(body.endswith(disk))
        head = body[:len(body) - len(disk)].decode()
        self.assertIn("# steps=roles v1 (steps.json)", head)
        st, _, body = self.get(f"/results/{DONE}/download/steps.json", "owner")
        self.assertEqual((st, json.loads(body)["schema"]), (200, "steps/1"))

    def test_blind_set_restricts_objects(self):
        """Owner and admin still get objects.csv while the object set is open; anyone else who can see the set does
        not (simulated by a viewer may_get_objects refuses): the page, the file and both zips leave it out."""
        self.assertIn(f"/results/{BLIND}/download/objects.csv", self.page(BLIND, "owner"))
        self.assertEqual(self.get(f"/results/{BLIND}/download/objects.csv", "admin")[0], 200)
        real = R.may_get_objects
        R.may_get_objects = lambda cfg, ps, user: real(cfg, ps, dict(user, id="someone_else", role="basic"))
        self.addCleanup(setattr, R, "may_get_objects", real)
        page = self.page(BLIND, "owner")
        self.assertNotIn(f"/results/{BLIND}/download/objects.csv", page)
        self.assertIn("blind object review is open", page)
        st, _, body = self.get(f"/results/{BLIND}/download/objects.csv", "owner")
        self.assertEqual(st, 403)
        st, _, body = self.get(f"/results/{BLIND}/download/results_{BLIND}.zip", "owner")
        self.assertEqual(st, 200)
        self.assertNotIn("objects.csv", zipfile.ZipFile(io.BytesIO(body)).namelist())
        st, _, body = self.get(f"/results/{BLIND}/download/everything_{BLIND}.zip", "owner")
        self.assertEqual(st, 200)
        names = zipfile.ZipFile(io.BytesIO(body)).namelist()
        self.assertIn("results/results.json", names)
        self.assertNotIn("results/objects.csv", names)
        # control: the same viewer on a set with no object set gets it
        self.assertEqual(self.get(f"/results/{DONE}/download/objects.csv", "owner")[0], 200)

    def test_recipe_admin_only_and_checked(self):
        self.assertEqual(self.post(f"/results/{DONE}/recipe", "owner", b"step=nesting%3A1")[0], 403)
        self.assertEqual(self.post(f"/results/{DONE}/recipe", "admin", b"step=nesting%3A1", origin=False)[0], 403)
        self.assertEqual(self.post(f"/results/{DONE}/recipe", "admin", b"step=bogus%3A1")[0], 400)
        self.assertEqual(self.post(f"/results/{DONE}/recipe", "admin", b"step=roles%3A1")[0], 400)
        calls = []

        def spy(ps_dir, reviewed, recipe=None):
            calls.append(recipe)
            raise RuntimeError("not run in this test")
        self.addCleanup(setattr, R, "COMPUTE", R.COMPUTE)
        R.COMPUTE = spy
        st, h, _ = self.post(f"/results/{DONE}/recipe", "admin", b"step=nesting%3A1")
        self.assertEqual(st, 303)
        self.assertEqual(T.hget(h, "location"), f"/results/{DONE}/")
        t0 = time.time()
        while not calls and time.time() - t0 < 10:
            time.sleep(0.05)
        self.assertEqual(calls, [[("roles", 1), ("nesting", 1)]])

    def test_recipe_runs_the_engine_and_update_keeps_it(self):
        st, _, _ = self.post(f"/results/{TODO}/recipe", "admin", b"step=nesting%3A1")
        self.assertEqual(st, 303)
        self.wait_done(TODO)
        rdir = self.root / TODO / "results"
        res = json.loads((rdir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(res["steps"], [{"step": "roles", "version": 1}, {"step": "nesting", "version": 1}])
        doc = json.loads((rdir / "steps.json").read_text(encoding="utf-8"))
        self.assertEqual([e["step"] for e in doc["recipe"]], ["roles", "nesting"])
        page = self.page(TODO)
        self.assertIn('data-step="nesting"', page)
        self.assertIn('value="nesting:1" checked', page)
        # a plain Update keeps the recipe
        self.assertEqual(R.current_recipe({"dir": self.root / TODO}), [("roles", 1), ("nesting", 1)])
        st, _, _ = self.post(f"/results/{TODO}/compute", "admin")
        self.assertEqual(st, 303)
        self.wait_done(TODO)
        self.assertIn("steps", json.loads((rdir / "results.json").read_text(encoding="utf-8")))
        # and back to the default: no "steps" key
        self.post(f"/results/{TODO}/recipe", "admin", b"")
        self.wait_done(TODO)
        self.assertNotIn("steps", json.loads((rdir / "results.json").read_text(encoding="utf-8")))
        self.assertIsNone(R.current_recipe({"dir": self.root / TODO}))


if __name__ == "__main__":
    unittest.main()
