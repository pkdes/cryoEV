"""
Train > Advanced > Collapse classes (round 14, class maps; portal/modules/train.py).

GET /train/collapse previews the per-class counts and writes nothing (registry, outputs and audit byte-identical);
when the collapsed labels equal a registered dataset (labels_sha) it offers "Use existing dataset". POST builds and
registers <dataset>__<map> in a worker thread (dataset_from_lv.derive), one build per target, with write-ahead audit
lines; basic users get 403, cross-origin POSTs are refused, non-admins see no paths.

Known answer: collapsing the ev_nonev_undet fixture dataset with cm_ev_nonev_v1 must produce the ev_nonev fixture
dataset's labels_sha (the reuse case). Temp roster / state / registry / CryoAI / CryoAI-laptop only (cjtest.Env +
the dataset_from_lv fixture).

    python -m unittest discover -s portal/tests -p test_train_collapse.py
"""
import hashlib
import json
import os
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import audit as PA  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import train as TR  # noqa: E402

sys.path.insert(0, str(T.REPO / "data_utils" / "tests"))
import test_dataset_from_lv as F  # noqa: E402
from data_utils import dataset_from_lv as D  # noqa: E402

FORM = "application/x-www-form-urlencoded"
UNDET, EVNONEV, ROBO = "lv_ds_undet", "lv_ds_evnonev", "ds_robo"
UNDET2 = "lv_ds_undet2"                     # a second ev_nonev_undet dataset: the one the build test collapses
TARGET = f"{UNDET2}__cm_ev_nonev_v1"


def snap(*dirs) -> dict:
    out = {}
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.is_file():
                out[str(p)] = hashlib.md5(p.read_bytes()).hexdigest()
    return out


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.data = cls.env.tmp / "data"                     # CryoAI and CryoAI-laptop roots of the fixture
        cls.data.mkdir()
        cls.export = F.make_export(cls.data)
        F.write_lv(cls.data / "labels", "lv_20990101_0000", F.make_pull(), review_scope="reviewed_only")
        cls.envp = mock.patch.dict(os.environ, {"CRYOEV_LAPTOP_ROOT": str(cls.data),
                                                "CRYOEV_CRYOAI_ROOT": str(cls.data),
                                                "CRYOEV_REGISTRY": str(cls.env.registry)})
        cls.envp.start()
        cls.out = cls.env.tmp / "outputs"
        for name, scheme in ((UNDET, "ev_nonev_undet"), (UNDET2, "ev_nonev_undet"), (EVNONEV, "ev_nonev")):
            prov = D.build("lv_20990101_0000", F.EXPORT, name, cls.out, scheme, export_only="export-labels",
                           log=lambda *a: None)
            D.register(prov, cls.out / name, cls.env.registry)
        T.TK.write_registry(cls.env.registry, {"dataset": [
            *[json.loads(x) for x in (cls.env.registry / "dataset.jsonl").read_text(encoding="utf-8").splitlines()],
            {"schema": "dataset/1", "id": ROBO, "created": "2026-10-28T00:00:00", "logged": "2026-10-30T09:00:00",
             "source": {"kind": "roboflow_export", "id": "rf"}, "split": {"method": "x"}, "counts": {},
             "classes": ["EV"], "images_sha256": None, "status": "trained"}]}, samples=False)
        cls.adv_key = A.add_key(cls.env.roster, "adv", "advanced")
        cls.bas_key = A.add_key(cls.env.roster, "bas", "basic")
        cls.saved_ttl = TR.REGISTRY_TTL_S
        TR.REGISTRY_TTL_S = 0
        cfg = cls.env.config()
        cfg.dataset_output_root = cls.out
        cls.p = T.Portal(cfg)
        cls.tok = {"admin": T.login(cls.p.port, cls.env.key), "advanced": T.login(cls.p.port, cls.adv_key),
                   "basic": T.login(cls.p.port, cls.bas_key)}

    @classmethod
    def tearDownClass(cls):
        TR.wait_collapses(60)
        cls.p.stop()
        TR.REGISTRY_TTL_S = cls.saved_ttl
        TR._reg_cache.clear()
        cls.envp.stop()
        cls.env.cleanup()

    def setUp(self):
        TR._reg_cache.clear()
        with TR._collapse_lock:
            TR._collapses.clear()
            TR._target_locks.clear()
        getattr(TR, "_preview_cache", {}).clear()

    def get(self, path, role="admin"):
        return T.request(self.p.port, "GET", path, T.cookie(self.tok[role]))

    def post(self, path, fields, role="admin", origin=True):
        h = {**T.cookie(self.tok[role]), "Content-Type": FORM}
        if origin:
            h["Origin"] = self.p.origin
        return T.request(self.p.port, "POST", path, h, urlencode(fields).encode())

    def audits(self):
        return [x for x in T.module_lines(PA.audit_path(self.env.state)) if x.get("action") == TR.COLLAPSE_ACTION]

    def files(self):
        return snap(self.env.registry, self.out)


class Collapse(Base):
    def test_basic_is_refused(self):
        before = self.files()
        for path in ("/train/collapse?dataset=x&cm=cm_ev_v1", "/train/api/collapse?target=x"):
            self.assertEqual(self.get(path, "basic")[0], 403, path)
        self.assertEqual(self.post("/train/collapse", {"dataset": UNDET, "cm": "cm_ev_nonev_v1"}, "basic")[0], 403)
        self.assertEqual(self.post("/train/collapse", {"dataset": UNDET, "cm": "cm_ev_nonev_v1"}, origin=False)[0],
                         403)                                                      # cross-origin (no Origin)
        self.assertEqual(self.files(), before)

    def test_page_offers_only_label_set_datasets(self):
        st, _, body = self.get("/train/")
        self.assertEqual(st, 200)
        t = body.decode()
        self.assertIn('id="cj-train-adv"', t)
        self.assertIn("Collapse classes", t)
        self.assertRegex(t, rf'<option value="{ROBO}" disabled>{ROBO} \(not built from a label set')
        self.assertRegex(t, rf'<option value="{UNDET}"( selected)?>{UNDET}</option>')
        self.assertIn('<option value="cm_ev_nonev_v1">', t)

    def test_preview_writes_nothing_and_offers_reuse(self):
        before, n_audit = self.files(), len(self.audits())
        st, _, body = self.get(f"/train/collapse?dataset={UNDET}&cm=cm_ev_nonev_v1")
        t = body.decode()
        self.assertEqual(st, 200, t[-2000:])
        self.assertIn('id="cj-train-collapse-counts"', t)
        self.assertIn(f"An identical dataset already exists: {EVNONEV}", t)
        self.assertIn(f'href="/train/?dataset={EVNONEV}#cj-train-h-new">Use existing dataset', t)
        self.assertIn("Make collapsed copy", t)
        # counts: train now (undet record) -> after (= the ev_nonev record)
        ev_now = D.registered(self.env.registry)[UNDET]["class_counts"]["train"]["EV"]
        ev_after = D.registered(self.env.registry)[EVNONEV]["class_counts"]["train"]["EV"]
        self.assertIn(f'<td data-k="Train now">{ev_now}</td><td data-k="Train after"><b>{ev_after}</b>', t)
        self.assertIn("merged or dropped", t)                                     # Undetermined
        self.assertEqual(self.files(), before)
        self.assertEqual(len(self.audits()), n_audit)
        st, _, body = self.get(f"/train/?dataset={EVNONEV}")                  # "Use existing dataset" preselects
        self.assertIn(f'<option value="{EVNONEV}" selected>', body.decode())

    def test_preview_refusals(self):
        self.assertEqual(self.get(f"/train/collapse?dataset={ROBO}&cm=cm_ev_v1")[0], 400)
        self.assertEqual(self.get(f"/train/collapse?dataset={UNDET}&cm=cm_nope_v1")[0], 400)
        self.assertEqual(self.get("/train/collapse?dataset=%3Cb%3E&cm=cm_ev_v1")[0], 400)

    def test_build_registers_the_copy(self):
        st, _, body = self.post("/train/collapse", {"dataset": UNDET2, "cm": "cm_ev_nonev_v1"}, "advanced")
        self.assertEqual(st, 202, body.decode()[-2000:])
        self.assertIn('id="cj-train-collapse-status"', body.decode())
        self.assertTrue(TR.wait_collapses(120))
        rec = D.registered(self.env.registry)[TARGET]
        self.assertEqual((rec["parent_dataset"], rec["class_map_id"]), (UNDET2, "cm_ev_nonev_v1"))
        self.assertEqual(D.labels_sha(self.out / TARGET), D.labels_sha(self.out / EVNONEV))      # the known answer
        self.assertEqual([(a["actor"], a["target"], a["detail"]["phase"], a["detail"].get("result"))
                          for a in self.audits()], [("adv", TARGET, "requested", None), ("adv", TARGET, "result", "ok")])
        st, _, body = self.get(f"/train/api/collapse?target={TARGET}", "advanced")
        j = json.loads(body)
        self.assertEqual((st, j["state"], j["dataset"]), (200, "done", UNDET2))
        self.assertNotIn("dataset_dir", j)                                         # no paths for non-admins
        self.assertIn("dataset_dir", json.loads(self.get(f"/train/api/collapse?target={TARGET}")[2]))
        before = self.files()
        st, _, body = self.post("/train/collapse", {"dataset": UNDET2, "cm": "cm_ev_nonev_v1"})
        self.assertEqual(st, 409)
        self.assertIn("already in the registry", body.decode())
        self.assertEqual(self.files(), before)
        TR._collapses.clear()                                     # after a restart: the registry says it exists
        st, _, body = self.get(f"/train/collapse?dataset={UNDET2}&cm=cm_ev_nonev_v1", "advanced")
        self.assertIn("is already in the registry", body.decode())

    def test_one_build_per_target(self):
        lock = threading.Lock()
        lock.acquire()
        with TR._collapse_lock:
            TR._target_locks["lv_ds_evnonev__cm_ev_v1"] = lock                # a build of that target is running
        before, n = self.files(), len(self.audits())
        st, _, body = self.post("/train/collapse", {"dataset": EVNONEV, "cm": "cm_ev_v1"})
        self.assertEqual(st, 409)
        self.assertIn("being built already", body.decode())
        self.assertEqual((self.files(), len(self.audits())), (before, n))
        lock.release()

    def test_audit_unwritable_means_nothing_built(self):
        before = self.files()
        with mock.patch.object(TR.PA, "event", return_value=False):
            st, _, body = self.post("/train/collapse", {"dataset": EVNONEV, "cm": "cm_ev_v1"})
        self.assertEqual(st, 500)
        self.assertIn("audit log", body.decode())
        self.assertTrue(TR.wait_collapses(10))
        self.assertEqual(self.files(), before)
        self.assertTrue(TR._target_locks["lv_ds_evnonev__cm_ev_v1"].acquire(blocking=False))   # released again

    def test_other_users_build_is_hidden(self):
        TR._collapses["x__cm_ev_v1"] = {"target": "x__cm_ev_v1", "dataset": "x", "cm": "cm_ev_v1", "uid": "someone",
                                        "state": "failed", "error": "C:\\secret\\path boom", "dataset_dir": None}
        self.assertEqual(self.get("/train/api/collapse?target=x__cm_ev_v1", "advanced")[0], 404)
        j = json.loads(self.get("/train/api/collapse?target=x__cm_ev_v1")[2])
        self.assertIn("secret", j["error"])                                        # admins see the error

    # ---- round 14 QC fixes (items 16, 19)
    def test_low_disk_space_refuses_before_audit(self):
        before, n = self.files(), len(self.audits())
        Usage = type("U", (), {"total": 10, "used": 10, "free": 0})
        with mock.patch.object(TR.shutil, "disk_usage", return_value=Usage()):
            st, _, body = self.post("/train/collapse", {"dataset": EVNONEV, "cm": "cm_ev_v1"})
        self.assertEqual(st, 409, body.decode()[-1500:])
        self.assertIn("free space", body.decode())
        self.assertEqual((self.files(), len(self.audits())), (before, n))      # no audit line, nothing written
        self.assertNotIn("lv_ds_evnonev__cm_ev_v1", TR._collapses)
        self.assertTrue(TR._target_locks.get("lv_ds_evnonev__cm_ev_v1", threading.Lock()).acquire(blocking=False))

    def test_need_bytes_counts_parent_images(self):
        rec = D.registered(self.env.registry)[EVNONEV]
        size = sum(p.stat().st_size for p in (Path(rec["dataset_dir"]) / "images").rglob("*") if p.is_file())
        self.assertGreater(size, 0)
        self.assertEqual(TR.collapse_need_bytes(rec), int(size * 1.5))

    def test_preview_is_cached_and_locked_per_key(self):
        real = D.derive_preview
        calls = []

        def spy(*a, **kw):
            calls.append(a[:2])
            return real(*a, **kw)
        with mock.patch.object(D, "derive_preview", side_effect=spy):
            for _ in range(2):
                st, _, body = self.get(f"/train/collapse?dataset={UNDET}&cm=cm_ev_v1")
                self.assertEqual(st, 200, body.decode()[-1500:])
        self.assertEqual(len(calls), 1)                                         # second view from the cache
        self.assertIn((UNDET, "cm_ev_v1"), TR._preview_locks)

    def test_preview_says_folder_exists_not_registered(self):
        d = self.out / f"{EVNONEV}__cm_ev_v1"
        d.mkdir()
        try:
            st, _, body = self.get(f"/train/collapse?dataset={EVNONEV}&cm=cm_ev_v1")
            t = body.decode()
            self.assertEqual(st, 200, t[-1500:])
            self.assertIn("folder exists, not registered", t)
            self.assertNotIn("Make collapsed copy", t)
        finally:
            d.rmdir()


if __name__ == "__main__":
    unittest.main()
