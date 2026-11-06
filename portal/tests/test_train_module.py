"""
Train module (portal/modules/train.py): roles through the Gate (basic 403), form validation, the queue setting
CJ_TRAIN_QUEUE (off: the preview is shown and the temp registry and temp queue share stay byte-identical, no audit
line; on: exactly one job, equal to what the CLI `registry.py job new` writes on a copy of the same registry, which is
the control), the preview JSON equal to the line written, the audit lines, escaping, the live snapshot without paths
for non-admins, and the pages in a browser (Playwright, Edge channel: 390 and 1280 px, light and dark, no horizontal
scroll, same-origin requests only, no console errors). Screenshots go to $CJ_SHOT_DIR when set (else a temp folder,
removed afterwards).

Temp roster / state dir / registry / queue share only (cjtest.Env); never the real ones. Nothing is ever pushed.

  python -m unittest discover -s portal/tests -v
"""
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import audit as PA  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import train as TR  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

FORM = "application/x-www-form-urlencoded"
JSON_RE = re.compile(r'<pre class="cj-train-pre" id="cj-train-json">(.*?)</pre>', re.S)
PROD = "v5_evnonev_y8m_20261029"          # real repo weights (models/<id>/best.pt) so the weights check passes
SMALL = "v4_y8n_20260924"
EVIL_EXPORT = "rf export <b>bold</b> & \"q\""
EVIL_MODEL = "m_<i>it</i>"
TODAY = datetime.now().strftime("%Y%m%d")
REG_PY = T.REPO / "registry" / "registry.py"
DATED_RUN = "2026-10-01_101010_y8n_640_4b_3ep_Run_7"
PRED_JOB = "job_20261105_pred_other"
UNKNOWN_JOB = "job_20261105_not_in_registry"


def jl(path: Path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_registry(root: Path):
    """A small registry: two datasets (one with an image list), three models (production first), runs, train jobs."""
    lst = "".join(f"img_{i:04d}_png\t{s}\n" for i, s in enumerate(["train"] * 6 + ["valid"] * 2 + ["test"] * 2))
    (root / "lists").mkdir(parents=True, exist_ok=True)
    (root / "lists" / "ds_test_v1.tsv").write_text(lst, encoding="utf-8", newline="\n")
    sha = hashlib.sha256(lst.encode()).hexdigest()
    jl(root / "dataset.jsonl", [
        {"schema": "dataset/1", "id": "ds_test_v1", "created": "2026-10-28T00:00:00", "logged": "2026-10-30T09:00:00",
         "source": {"kind": "roboflow_export", "id": EVIL_EXPORT}, "export": EVIL_EXPORT,
         "split": {"method": "split file"}, "counts": {"train": {"images": 6}, "valid": {"images": 2}},
         "counts_trained": {"train": 6, "valid": 2, "test": 2}, "classes": ["EV"], "images_sha256": None,
         "status": "trained", "image_list": {"file": "lists/ds_test_v1.tsv", "sha256": sha, "n": 10}},
        {"schema": "dataset/1", "id": "ds_lv_nolist", "created": "2026-11-01T00:00:00",
         "logged": "2026-11-01T09:00:00", "source": {"kind": "label_version", "id": "lv_20261101_0900"},
         "label_version": "lv_20261101_0900", "split": {"method": "x"}, "counts": {}, "classes": [],
         "images_sha256": None, "status": "created"},
        {"schema": "dataset/1", "id": "ds_badsha", "created": "2026-09-01T00:00:00", "logged": "2026-09-01T09:00:00",
         "source": {"kind": "roboflow_export", "id": "x"}, "split": {}, "counts": {}, "classes": [],
         "images_sha256": None, "status": "trained",
         "image_list": {"file": "lists/ds_test_v1.tsv", "sha256": "0" * 64, "n": 10}},
        {"schema": "dataset/1", "id": "ds_badn", "created": "2026-07-01T00:00:00", "logged": "2026-07-01T09:00:00",
         "source": {"kind": "roboflow_export", "id": "z"}, "split": {}, "counts": {}, "classes": [],
         "images_sha256": None, "status": "trained",
         "image_list": {"file": "lists/ds_test_v1.tsv", "sha256": sha, "n": 11}},
        {"schema": "dataset/1", "id": "ds_escape", "created": "2026-08-01T00:00:00", "logged": "2026-08-01T09:00:00",
         "source": {"kind": "roboflow_export", "id": "y"}, "split": {}, "counts": {}, "classes": [],
         "images_sha256": None, "status": "trained", "image_list": {"file": "../outside.tsv", "n": 1}}])
    jl(root / "run.jsonl", [
        {"schema": "run/1", "id": "run_prod", "dataset": "ds_test_v1", "status": "finished",
         "started": "2026-10-29T05:48:10", "config": {"arch": "yolov8m-seg", "imgsz": 1440, "batch": 8,
                                                     "epochs": 300, "patience": 300, "rect": False},
         "metrics": {"mask_map50": 0.905, "mask_map50_95": 0.622}, "logged": "2026-10-30T09:00:00"},
        {"schema": "run/1", "id": "run_seed1", "dataset": "ds_test_v1", "status": "finished",
         "started": "2026-11-04T20:54:41", "machine": "gpu-desktop",
         "config": {"arch": "yolov8m-seg", "imgsz": 1440, "batch": 8, "epochs": 300, "seed": 1},
         "metrics": {"mask_map50": 0.903, "mask_map50_95": 0.618}, "logged": "2026-11-05T10:58:35"},
        {"schema": "run/1", "id": "run_small", "dataset": "ds_test_v1", "status": "finished",
         "started": "2026-09-24T13:42:21", "config": {"arch": "yolov8n-seg", "imgsz": 1440, "batch": 24, "epochs": 300},
         "logged": "2026-10-30T09:00:00"},
        {"schema": "run/1", "id": DATED_RUN, "dataset": "ds_test_v1", "status": "finished",
         "config": {"arch": "yolov8n-seg", "imgsz": 640, "batch": 4, "epochs": 3},
         "metrics": {"mask_map50": 0.5, "mask_map50_95": 0.3}, "logged": "2026-10-30T09:00:00"},
        {"schema": "run/1", "id": "run_nodate", "dataset": "ds_test_v1", "status": "finished",
         "config": {"arch": "yolov8n-seg", "imgsz": 640}, "logged": "2026-06-30T09:00:00"}])
    jl(root / "model.jsonl", [
        {"schema": "model/1", "id": SMALL, "run": "run_small", "dataset": "ds_test_v1",
         "weights": f"models/{SMALL}/best.pt", "status": "registered", "logged": "2026-10-30T09:00:00"},
        {"schema": "model/1", "id": PROD, "run": "run_prod", "dataset": "ds_test_v1",
         "weights": f"models/{PROD}/best.pt", "status": "production", "logged": "2026-10-30T09:00:01"},
        {"schema": "model/1", "id": EVIL_MODEL, "run": None, "weights": "models/none/best.pt",
         "status": "registered", "logged": "2026-10-30T09:00:02"}])
    jl(root / "job.jsonl", [
        {"schema": "job/1", "id": "job_20261104_demo_seed1", "logged": "2026-11-05T10:58:35", "name": "demo_seed1",
         "type": "train", "where": "desktop", "status": "done", "n_images": 10, "who": None, "inputs": [],
         "outputs": [], "note": "run run_seed1 (gpu-desktop)"},
        {"schema": "job/1", "id": "job_20261103_vialive", "logged": "2026-11-03T10:00:00", "name": "vialive",
         "type": "train", "where": "desktop", "status": "done", "n_images": 10, "who": None, "inputs": [],
         "outputs": []},
        {"schema": "job/1", "id": "job_20261102_nolink", "logged": "2026-11-02T10:00:00", "name": "nolink",
         "type": "train", "where": "desktop", "status": "done", "n_images": 10, "who": None, "inputs": [],
         "outputs": []},
        {"schema": "job/1", "id": PRED_JOB, "logged": "2026-11-05T09:00:00", "name": "pred_other",
         "type": "predict", "where": "desktop", "status": "in_progress", "n_images": 3, "who": "someone_else",
         "inputs": [], "outputs": []},
        {"schema": "job/1", "id": f"job_{TODAY}_taken", "logged": "2026-11-05T10:00:00", "name": "taken",
         "type": "train", "where": "desktop", "status": "ready", "n_images": 10, "who": None, "inputs": [],
         "outputs": []}])


def snap(*dirs) -> dict:
    """{path: md5} of every file under the folders (a byte-identity check)."""
    out = {}
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.is_file():
                out[str(p)] = hashlib.md5(p.read_bytes()).hexdigest()
    return out


def lines(path: Path) -> list:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def live_doc(share: Path) -> dict:
    d = T.live_doc()
    q = d["queues"][0]
    q["share"] = str(share)
    q["watcher"]["detail"] = "C:\\cj\\secret\\path"
    q["running"].update({"job": f"job_{TODAY}_live</script><b>x", "run": "2026-11-05_120000_y8m_1440_8b_300ep_Run_9",
                         "batch": 8, "map50": 0.81, "map50_95": 0.52,
                         "series": {"epoch": list(range(1, 13)), "box": [1.2 - i * .05 for i in range(12)],
                                    "seg": [2.0 - i * .07 for i in range(12)], "cls": [1.0 - i * .04 for i in range(12)],
                                    "dfl": [1.1 - i * .03 for i in range(12)],
                                    "map50": [.3 + i * .04 for i in range(12)],
                                    "map50_95": [.2 + i * .027 for i in range(12)]}})
    q["jobs"] = [{"job": "job_20261104_demo_seed1", "state": "done", "reason": None, "pulled": True},
                 {"job": "job_20261103_broken", "state": "blocked", "reason": "C:\\cj\\work\\x failed", "pulled": True}]
    q["finished"] = [{"job": "job_20261104_demo_seed1", "status": "done", "ended": "2026-11-04T21:49:42",
                      "run": "run_seed1", "pulled": True, "metrics": {"mask_map50_95": 0.618, "mask_map50": 0.903},
                      "best_epoch": 140, "reason": None},
                     {"job": "job_20261103_vialive", "status": "done", "ended": "2026-10-01T11:00:00",
                      "run": DATED_RUN, "pulled": True, "metrics": {"mask_map50_95": 0.3}}]
    return d


class Base(unittest.TestCase):
    """A portal on a temp Env (admin `tester`, advanced `adv`, basic `bas`) with the small registry and a temp queue
    share named in live.json. CJ_TRAIN_QUEUE is set per test and restored."""

    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.share = cls.env.tmp / "share"
        for d in ("q", "d", "ds"):
            (cls.share / d).mkdir(parents=True)
        (cls.share / "watch.json").write_text('{"state": "idle"}', encoding="utf-8")
        build_registry(cls.env.registry)
        cls.env.write_live(live_doc(cls.share))
        cls.adv_key = A.add_key(cls.env.roster, "adv", "advanced")
        cls.bas_key = A.add_key(cls.env.roster, "bas", "basic")
        cls.saved_env = os.environ.get(TR.SETTING)
        cls.saved_ttl = TR.REGISTRY_TTL_S
        TR.REGISTRY_TTL_S = 0
        cls.p = T.Portal(cls.env.config())
        cls.tok = {"admin": T.login(cls.p.port, cls.env.key), "advanced": T.login(cls.p.port, cls.adv_key),
                   "basic": T.login(cls.p.port, cls.bas_key)}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        TR.REGISTRY_TTL_S = cls.saved_ttl
        TR._reg_cache.clear()
        if cls.saved_env is None:
            os.environ.pop(TR.SETTING, None)
        else:
            os.environ[TR.SETTING] = cls.saved_env
        cls.env.cleanup()

    def setUp(self):
        os.environ.pop(TR.SETTING, None)
        TR._reg_cache.clear()

    def get(self, path, role="admin"):
        return T.request(self.p.port, "GET", path, T.cookie(self.tok[role]))

    def post(self, path, fields, role="admin", origin=True):
        h = {**T.cookie(self.tok[role]), "Content-Type": FORM}
        if origin:
            h["Origin"] = self.p.origin if origin is True else origin
        return T.request(self.p.port, "POST", path, h, urlencode(fields).encode())

    def good(self, **kw) -> dict:
        f = {"dataset": "ds_test_v1", "base": PROD, "preset": "f5", "name": "", "epochs": "", "imgsz": "",
             "seed": "", "max_hours": ""}
        f.update(kw)
        return f

    def audit_lines(self):
        return lines(PA.audit_path(self.env.state))


class Roles(Base):
    PATHS = ("/train/", "/train/train.js", "/train/api/options", "/train/api/runs", "/train/api/live.json")

    def test_basic_gets_403_everywhere(self):
        before = snap(self.env.registry, self.share)
        for path in self.PATHS:
            with self.subTest(path=path):
                st, _, _ = self.get(path, "basic")
                self.assertEqual(st, 403)
        os.environ[TR.SETTING] = "on"
        for path in ("/train/check", "/train/queue"):
            st, _, _ = self.post(path, self.good(name="basic_try"), "basic")
            self.assertEqual(st, 403, path)
        self.assertEqual(snap(self.env.registry, self.share), before)
        self.assertEqual(self.audit_lines(), [])

    def test_admin_and_advanced_open_it(self):
        for role in ("admin", "advanced"):
            for path in self.PATHS:
                with self.subTest(role=role, path=path):
                    st, h, _ = self.get(path, role)
                    self.assertEqual(st, 200)
        st, h, _ = self.get("/train")
        self.assertEqual(st, 307)
        self.assertEqual(T.hget(h, "location"), "/train/")

    def test_cross_origin_post_refused(self):
        os.environ[TR.SETTING] = "on"
        before = snap(self.env.registry, self.share)
        for origin in ("https://evil.example", None):
            st, _, _ = self.post("/train/queue", self.good(name="xorigin"), origin=origin or False)
            self.assertEqual(st, 403, origin)
        self.assertEqual(snap(self.env.registry, self.share), before)
        self.assertEqual(self.audit_lines(), [])

    def test_no_paths_for_advanced(self):
        _, _, body = self.get("/train/api/live.json", "advanced")
        doc = json.loads(body)
        text = body.decode()
        self.assertEqual(doc["queues"][0]["share"], "GPU computer 1")
        self.assertNotIn(str(self.share).replace("\\", "\\\\"), text)
        self.assertNotIn("secret", text)
        self.assertNotIn("gpu-desktop", text)
        self.assertNotIn("C:\\\\cj", text)
        _, _, page = self.get("/train/", "advanced")
        self.assertNotIn(str(self.env.registry), page.decode())
        self.assertNotIn("models/", page.decode())                # weights paths are admin-only
        self.assertNotIn("gpu-desktop", page.decode())
        _, _, adm = self.get("/train/api/live.json")
        self.assertEqual(json.loads(adm)["queues"][0]["share"], str(self.share))
        _, _, opts = self.get("/train/api/options", "advanced")
        self.assertNotIn("weights", json.loads(opts)["models"][0])

    def private_doc(self):
        """Someone else's predict job running, queued-done and finished, plus a queue job not in the registry."""
        d = live_doc(self.share)
        q = d["queues"][0]
        q["running"] = {"job": PRED_JOB, "kind": "predict", "n_done": 1, "n_total": 3}
        q["watcher"]["job"] = PRED_JOB
        q["jobs"] = q["jobs"] + [{"job": PRED_JOB, "state": "running"}, {"job": UNKNOWN_JOB, "state": "queued"}]
        q["finished"] = q["finished"] + [{"job": PRED_JOB, "status": "done", "ended": "2026-11-05T09:30:00"},
                                         {"job": UNKNOWN_JOB, "status": "blocked", "reason": "x"}]
        return d

    def test_live_hides_other_jobs_from_non_admins(self):
        self.env.write_live(self.private_doc())
        try:
            for role, path in (("advanced", "/train/api/live.json"), ("advanced", "/train/"),
                               ("admin", "/train/api/live.json"), ("admin", "/train/")):
                with self.subTest(role=role, path=path):
                    st, _, body = self.get(path, role)
                    self.assertEqual(st, 200)
                    text = body.decode()
                    for hidden in (PRED_JOB, UNKNOWN_JOB):
                        if role == "admin":                         # the control: admins see every job
                            self.assertIn(hidden, text)
                        else:
                            self.assertNotIn(hidden, text)
                    self.assertIn("job_20261104_demo_seed1", text)  # a train job: shown to both
            _, _, body = self.get("/train/api/live.json", "advanced")
            q = json.loads(body)["queues"][0]
            self.assertIsNone(q["running"])
            self.assertIsNone(q["watcher"]["job"])
            self.assertEqual(sorted(q["watcher"]), ["age_s", "display", "job", "stale", "state"])
            self.assertEqual([j["job"] for j in q["jobs"]], ["job_20261104_demo_seed1"])
            self.assertEqual([f["job"] for f in q["finished"]], ["job_20261104_demo_seed1", "job_20261103_vialive"])
            _, _, page = self.get("/train/", "advanced")
            self.assertIn(b'data-cj-live="compact"', page)
            self.assertIn(b"No job running.", page)
            _, _, page = self.get("/train/")
            self.assertIn(b'data-cj-live="full"', page)
            # the registry can't be read: no job at all for non-admins (fail closed); admins unchanged
            doc = self.private_doc()
            out = TR.live_for(doc, False, {}, False)
            self.assertEqual(out["queues"][0]["jobs"] + out["queues"][0]["finished"], [])
            self.assertIsNone(out["queues"][0]["watcher"]["job"])
            self.assertEqual(len(TR.live_for(doc, True, {}, False)["queues"][0]["jobs"]), 4)
        finally:
            self.env.write_live(live_doc(self.share))


class Validation(Base):
    def plan(self, role="admin", **kw):
        return TR.plan(self.env.config(), self.good(**kw), {"id": "tester", "role": role})

    def test_good_plan_and_defaults_from_production_run(self):
        errs, p = self.plan()
        self.assertEqual(errs, [])
        self.assertEqual((p["epochs"], p["imgsz"], p["seed"], p["max_hours"]), (300, 1440, 42, 48.0))
        self.assertEqual(p["job"], f"job_{TODAY}_f5_m1440_seed42")
        self.assertEqual(p["where"], "desktop")
        self.assertEqual(p["run_experiments_args"],
                         ["--single-run", "--model-size", "m", "--batch", "8", "--imgsz", "1440", "--epochs", "300",
                          "--patience", "300", "--rect", "0", "--workers", "4", "--seed", "42",
                          "--weights", f"models/{PROD}/best.pt"])
        self.assertEqual(p["record"]["n_images"], 10)
        self.assertIsNone(p["record"]["who"])                       # "tester" is not a people.json id
        self.assertEqual(TR.presets({})["f5"]["seed"], 42)          # no production run: the documented values

    def test_options_production_first(self):
        _, _, body = self.get("/train/api/options")
        doc = json.loads(body)
        self.assertEqual(doc["queue"], "off")
        self.assertEqual(doc["models"][0]["id"], PROD)
        self.assertEqual(doc["models"][0]["status"], "production")
        ds = {d["id"]: d for d in doc["datasets"]}
        self.assertEqual(ds["ds_test_v1"]["counts"], {"train": 6, "valid": 2, "test": 2})
        self.assertTrue(ds["ds_test_v1"]["usable"])
        self.assertFalse(ds["ds_lv_nolist"]["usable"])
        self.assertEqual(ds["ds_lv_nolist"]["source"], "label version lv_20261101_0900")
        self.assertFalse(ds["ds_escape"]["usable"])                 # a list outside the registry is never read
        self.assertFalse(ds["ds_badsha"]["usable"])
        self.assertFalse(ds["ds_badn"]["usable"])
        self.assertEqual(ds["ds_badn"]["problem"], "its image list doesn't match its recorded size")

    def test_errors(self):
        cases = [
            (dict(dataset=""), "Pick a dataset."),
            (dict(dataset="nope"), "That dataset is not in the registry."),
            (dict(dataset="ds_lv_nolist"), "no image list"),
            (dict(dataset="ds_escape"), "no image list"),
            (dict(dataset="ds_badsha"), "has changed since it was recorded"),
            (dict(base=""), "Pick a base model."),
            (dict(base="nope"), "not in the registry"),
            (dict(base=EVIL_MODEL), "weights file is not in the repo"),
            (dict(preset="huge"), "Pick a preset."),
            (dict(epochs="abc"), "Epochs must be a whole number."),
            (dict(epochs="0"), "Epochs must be between"),
            (dict(imgsz="1000"), "multiple of 32"),
            (dict(imgsz="4096"), "Image size must be between"),
            (dict(seed="-1"), "Seed must be between"),
            (dict(max_hours="nan"), "Max hours must be a number"),
            (dict(max_hours="100"), "Max hours must be a number"),
            (dict(name="a/b"), "The name may use"),
            (dict(name="x" * 36), "The name may use"),
            (dict(name="taken"), "already exists today"),
            (dict(preset="laptop"), "too big for the laptop"),            # y8m base on the laptop
            (dict(preset="laptop", base=TR.YOLO, imgsz="1024"), "Image size 1024 is too big"),
            (dict(preset="laptop", base=TR.YOLO, epochs="301"), "Epochs can be at most 300 on the laptop."),
            (dict(preset="laptop", base=TR.YOLO, max_hours="12.5"), "Max hours can be at most 12 on the laptop."),
            (dict(dataset="ds_badn"), "doesn't hold the number of images recorded"),
        ]
        for kw, want in cases:
            with self.subTest(**kw):
                errs, p = self.plan(**kw)
                self.assertIsNone(p)
                self.assertTrue(any(want in e for e in errs), errs)
        errs, p = self.plan(preset="laptop", base=TR.YOLO)                  # the laptop recipe from YOLO weights
        self.assertEqual(errs, [])
        self.assertEqual((p["size"], p["imgsz"], p["where"], p["max_hours"]), ("s", 640, "laptop", 4.0))
        self.assertNotIn("--weights", p["run_experiments_args"])
        errs, p = self.plan(preset="laptop", base=TR.YOLO, epochs="300", max_hours="12")
        self.assertEqual(errs, [])
        errs, p = self.plan(preset="laptop", base=SMALL)
        self.assertEqual(errs, [])
        self.assertEqual(p["size"], "n")

    def test_preview_note_hides_weights_path_from_non_admins(self):
        st, _, body = self.post("/train/check", self.good(name="note_adv"), "advanced")
        self.assertEqual(st, 200)
        note = json.loads(html.unescape(JSON_RE.search(body.decode()).group(1)))["note"]
        self.assertIn("--weights (the base model's file)", note)
        self.assertNotIn("models/", body.decode())
        st, _, body = self.post("/train/check", self.good(name="note_adm"))
        note = json.loads(html.unescape(JSON_RE.search(body.decode()).group(1)))["note"]
        self.assertIn(f"--weights models/{PROD}/best.pt", note)
        errs, p = TR.plan(self.env.config(), self.good(name="note_adv"), {"id": "adv", "role": "advanced"})
        self.assertIn(f"--weights models/{PROD}/best.pt", p["record"]["note"])     # what Queue stores: the full note

    def test_raw_errors_admin_only(self):
        errs, _ = self.plan(role="advanced", base=EVIL_MODEL)
        self.assertFalse(any("models/" in e for e in errs), errs)
        errs, _ = self.plan(role="admin", base=EVIL_MODEL)
        self.assertTrue(any("models/none/best.pt" in e for e in errs), errs)

    def test_who_is_a_people_id_only(self):
        R = TR.registry_module()
        saved = R.roster
        R.roster = lambda path=None: {"tester"}
        try:
            _, p = self.plan()
        finally:
            R.roster = saved
        self.assertEqual(p["record"]["who"], "tester")

    def test_http_400_on_errors(self):
        st, _, body = self.post("/train/check", self.good(epochs="abc"))
        self.assertEqual(st, 400)
        self.assertIn(b"Epochs must be a whole number.", body)
        st, _, _ = T.request(self.p.port, "POST", "/train/check",
                             {**T.cookie(self.tok["admin"]), "Origin": self.p.origin, "Content-Type": "text/plain"},
                             b"dataset=x")
        self.assertEqual(st, 400)
        st, _, _ = self.post("/train/check", {"name": "x" * 9000})
        self.assertEqual(st, 400)

    def test_setting_values(self):
        for v, want in (("on", True), ("ON ", True), ("off", False), ("1", False), ("true", False), ("", False)):
            os.environ[TR.SETTING] = v
            self.assertIs(TR.queue_on(types.SimpleNamespace()), want, v)
        os.environ[TR.SETTING] = "on"
        self.assertFalse(TR.queue_on(types.SimpleNamespace(train_queue="off")))     # the cfg attribute wins
        os.environ.pop(TR.SETTING)
        self.assertTrue(TR.queue_on(types.SimpleNamespace(train_queue="on")))
        self.assertFalse(TR.queue_on(types.SimpleNamespace()))                       # default off

    def test_max_job_id_matches_jobs_common(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("cj_t_common", T.REPO / "jobs" / "common.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(TR.MAX_JOB_ID, mod.MAX_JOB_ID)
        self.assertEqual(len(f"job_{TODAY}_") + TR.NAME_MAX, TR.MAX_JOB_ID)


class QueueOff(Base):
    def test_post_writes_nothing(self):
        before = snap(self.env.registry, self.share)
        st, _, body = self.post("/train/check", self.good(name="off_check"))
        self.assertEqual(st, 200)
        text = body.decode()
        self.assertIn("Preview only: this is exactly what would be queued.", text)
        self.assertNotIn('action="/train/queue"', text)                      # no Queue button while off
        st, _, body = self.post("/train/queue", self.good(name="off_queue"))
        self.assertEqual(st, 409)
        self.assertIn("Preview only: this is exactly what would be queued.", body.decode())
        self.assertIn("Queueing is off, so nothing was written.", body.decode())
        st, _, _ = self.post("/train/queue", self.good(name="off_queue", epochs="abc"))
        self.assertEqual(st, 409)
        self.assertEqual(snap(self.env.registry, self.share), before)
        self.assertFalse(PA.audit_path(self.env.state).exists())
        self.assertFalse((self.env.registry / "lists" / f"job_{TODAY}_off_queue.txt").exists())


class QueueOn(Base):
    def test_one_job_equal_to_the_cli_control(self):
        os.environ[TR.SETTING] = "on"
        name = "on_ctl"
        jid = f"job_{TODAY}_{name}"
        ctl = self.env.tmp / "control_registry"
        shutil.copytree(self.env.registry, ctl)
        share_before = snap(self.share)
        reg_before = lines(self.env.registry / "job.jsonl")
        # the preview first (Check): its JSON is what Queue must write
        st, _, body = self.post("/train/check", self.good(name=name))
        self.assertEqual(st, 200)
        text = body.decode()
        self.assertIn('action="/train/queue"', text)
        preview = json.loads(html.unescape(JSON_RE.search(text).group(1)))
        # queue
        st, _, body = self.post("/train/queue", self.good(name=name))
        self.assertEqual(st, 200, body[:600])
        self.assertIn(f"Job <b>{jid}</b> was created in the registry", body.decode())
        after = lines(self.env.registry / "job.jsonl")
        self.assertEqual(len(after), len(reg_before) + 1)
        self.assertEqual(after[:-1], reg_before)
        made = after[-1]
        self.assertEqual(made["id"], jid)
        self.assertEqual(snap(self.share), share_before)                   # nothing goes to a queue share
        # the control: the CLI on a copy of the same registry, same arguments
        r = subprocess.run([sys.executable, str(REG_PY), "--root", str(ctl), "job", "new", "--type", "train",
                            "--where", "desktop", "--name", name, "--images", str(ctl / "lists" / "ds_test_v1.tsv"),
                            "--note", made["note"]], capture_output=True, text=True, cwd=T.REPO)
        self.assertEqual(r.returncode, 0, r.stderr)
        cli = lines(ctl / "job.jsonl")[-1]
        strip = lambda d: {k: v for k, v in d.items() if k != "logged"}  # noqa: E731
        self.assertEqual(strip(made), strip(cli))
        self.assertEqual(list(made), list(cli))                            # same key order too
        self.assertEqual(preview.pop("logged"), "(the time it is written)")
        self.assertEqual(preview, strip(made))
        a, b = snap(self.env.registry), snap(ctl)
        rel = lambda s, root: {str(Path(k).relative_to(root)): v for k, v in s.items()  # noqa: E731
                               if not k.endswith(".lock")}
        a, b = rel(a, self.env.registry), rel(b, ctl)
        self.assertEqual(set(a), set(b))
        self.assertLessEqual({k for k in a if a[k] != b[k]}, {"job.jsonl"})  # at most `logged` differs
        self.assertEqual((self.env.registry / "lists" / f"{jid}.txt").read_bytes(),
                         (ctl / "lists" / f"{jid}.txt").read_bytes())
        # the audit: requested, then ok
        au = [x for x in self.audit_lines() if x["target"] == jid]
        self.assertEqual([x["detail"]["phase"] for x in au], ["requested", "result"])
        self.assertEqual(au[-1]["detail"]["result"], "ok")
        self.assertEqual(au[-1]["actor"], "tester")
        self.assertEqual(au[-1]["action"], "train_queue")
        self.assertEqual(au[-1]["detail"]["dataset"], "ds_test_v1")
        audit_text = PA.audit_path(self.env.state).read_text(encoding="utf-8")
        self.assertNotIn(T.A.COOKIE, audit_text)
        # the same job again: refused, nothing more written
        before = snap(self.env.registry, self.share)
        st, _, body = self.post("/train/queue", self.good(name=name))
        self.assertEqual(st, 400)
        self.assertIn("already exists today", body.decode())
        self.assertEqual(snap(self.env.registry, self.share), before)

    def test_advanced_queues_without_paths(self):
        os.environ[TR.SETTING] = "on"
        st, _, body = self.post("/train/queue", self.good(name="adv_one"), "advanced")
        self.assertEqual(st, 200)
        text = body.decode()
        self.assertIn("An admin sends it to the GPU computer", text)
        self.assertNotIn("make_job_bundle", text)
        self.assertNotIn(str(self.share), text)
        st, _, body = self.post("/train/queue", self.good(name="adm_one"))
        self.assertIn("make_job_bundle.py", body.decode())
        self.assertIn(html.escape(str(self.share)), body.decode())

    def test_audit_unwritable_means_no_job(self):
        os.environ[TR.SETTING] = "on"
        saved = TR.PA.audit

        def boom(*a, **k):
            raise OSError("disk full")
        TR.PA.audit = boom
        before = snap(self.env.registry, self.share)
        try:
            st, _, body = self.post("/train/queue", self.good(name="no_audit"))
        finally:
            TR.PA.audit = saved
        self.assertEqual(st, 500)
        self.assertIn("nothing was queued", body.decode())
        self.assertEqual(snap(self.env.registry, self.share), before)


class Escaping(Base):
    def test_registry_and_form_values_escaped(self):
        _, _, body = self.get("/train/")
        text = body.decode()
        self.assertNotIn("<b>bold</b>", text)
        self.assertIn("&lt;b&gt;bold&lt;/b&gt;", text)
        self.assertNotIn("<i>it</i>", text)
        self.assertIn("m_&lt;i&gt;it&lt;/i&gt;", text)
        self.assertNotIn("</script><b>x", text)                         # live job id inside the JSON block
        self.assertIn("\\u003c/script\\u003e", text)
        evil = '"><script>alert(1)</script>'
        st, _, body = self.post("/train/check", self.good(name=evil, epochs=evil))
        self.assertEqual(st, 400)
        text = body.decode()
        self.assertNotIn("<script>alert(1)</script>", text)
        self.assertIn("&quot;&gt;&lt;script&gt;", text)

    def test_past_runs_link_job_run_dataset_model(self):
        _, _, body = self.get("/train/api/runs")
        rows = json.loads(body)["runs"]
        seed = [r for r in rows if r["job"] == "job_20261104_demo_seed1"][0]
        self.assertEqual(seed["run"], "run_seed1")
        self.assertEqual(seed["mask_map50_95"], 0.618)
        self.assertEqual(seed["dataset"], "ds_test_v1")
        prod = [r for r in rows if r["run"] == "run_prod"][0]
        self.assertIsNone(prod["job"])
        self.assertEqual(prod["models"], [{"id": PROD, "status": "production"}])
        self.assertEqual(rows[0]["when"][:10], "2026-11-05")                # newest first (today's ready job)
        via = [r for r in rows if r["job"] == "job_20261103_vialive"][0]
        self.assertEqual(via["run"], DATED_RUN)                             # linked through live.json finished[]
        self.assertEqual((via["when"], via["when_kind"]), ("2026-10-01T10:10:10", "started"))   # from the run id
        self.assertEqual(via["mask_map50_95"], 0.3)
        nolink = [r for r in rows if r["job"] == "job_20261102_nolink"][0]
        self.assertTrue(nolink["run_not_linked"])
        self.assertIsNone(nolink["run"])
        nodate = [r for r in rows if r["run"] == "run_nodate"][0]
        self.assertEqual((nodate["when"], nodate["when_kind"]), ("2026-06-30T09:00:00", "recorded"))
        _, _, page = self.get("/train/")
        self.assertIn(b"run not linked", page)
        self.assertIn(b'<span class="cj-mut">recorded</span>', page)
        _, _, body = self.get("/train/api/runs", "advanced")
        self.assertNotIn("gpu-desktop", body.decode())


def launch(p):
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:
            continue
    return None


class Browser(Base):
    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        super().setUpClass()
        cls.own_shots = not os.environ.get("CJ_SHOT_DIR")
        cls.shots = Path(os.environ.get("CJ_SHOT_DIR") or tempfile.mkdtemp(prefix="cj_shots_"))
        cls.shots.mkdir(parents=True, exist_ok=True)
        cls.pw = sync_playwright().start()
        cls.browser = launch(cls.pw)
        if cls.browser is None:
            cls.tearDownClass()
            raise unittest.SkipTest("no Edge / Chrome for Playwright")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "browser", None):
            cls.browser.close()
        if getattr(cls, "pw", None):
            cls.pw.stop()
        super().tearDownClass()
        if cls.own_shots:
            shutil.rmtree(cls.shots, ignore_errors=True)

    def session(self, width, scheme, key=None):
        ctx = self.browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
        origin = self.p.origin
        rec = {"requests": [], "foreign": [], "errors": [], "console": []}
        ctx.route("**/*", lambda r: (r.continue_() if r.request.url.startswith(origin + "/")
                                     else (rec["foreign"].append(r.request.url), r.abort())))
        page = ctx.new_page()
        page.on("request", lambda r: rec["requests"].append(r.url))
        page.on("pageerror", lambda e: rec["errors"].append(str(e)))
        page.on("console", lambda m: m.type == "error" and rec["console"].append(m.text))
        page.goto(f"{origin}/login?k={key or self.env.key}")
        self.assertEqual(page.url, origin + "/")
        return ctx, page, rec

    def check(self, page, rec, scheme):
        sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
        self.assertLessEqual(sw, iw, "horizontal scroll")
        bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
        self.assertEqual(bg, "rgb(15, 21, 27)" if scheme == "dark" else "rgb(243, 245, 247)")
        self.assertEqual(rec["foreign"], [])
        self.assertTrue(all(u.startswith(self.p.origin + "/") for u in rec["requests"]))
        self.assertEqual(rec["errors"], [])
        self.assertEqual(rec["console"], [])

    def test_pages_sizes_and_schemes(self):
        before = snap(self.env.registry, self.share)
        for scheme in ("light", "dark"):
            for width in (390, 1280):
                with self.subTest(width=width, scheme=scheme):
                    ctx, page, rec = self.session(width, scheme)
                    try:
                        page.goto(self.p.origin + "/train/")
                        page.wait_for_load_state("networkidle")
                        page.wait_for_selector("#cj-bar")
                        page.wait_for_selector("#cj-train-live .cj-lv-run")          # CJLive drew the running job
                        self.assertEqual(page.locator(".cj-train-runs tbody tr").count(), 7)
                        self.check(page, rec, scheme)
                        page.screenshot(path=str(self.shots / f"train_{width}_{scheme}.png"), full_page=True)
                        page.select_option("#cj-train-preset", "laptop")
                        self.assertEqual(page.get_attribute("#cj-train-imgsz", "placeholder"), "640")
                        page.select_option("#cj-train-preset", "f5")
                        page.fill("#cj-train-seed", "7")
                        page.click(".cj-train-form button[type=submit]")
                        page.wait_for_selector("#cj-train-sum")
                        self.assertIn("Preview only: this is exactly what would be queued.",
                                      page.text_content("#cj-train-sum"))
                        self.assertIn(f"job_{TODAY}_f5_m1440_seed7", page.text_content("#cj-train-json"))
                        self.check(page, rec, scheme)
                        page.screenshot(path=str(self.shots / f"train_preview_{width}_{scheme}.png"), full_page=True)
                    finally:
                        ctx.close()
        self.assertEqual(snap(self.env.registry, self.share), before)

    def test_advanced_compact_live(self):
        doc = live_doc(self.share)
        doc["queues"][0]["running"]["job"] = "job_20261104_demo_seed1"      # a registry train job: shown
        self.env.write_live(doc)
        try:
            for scheme in ("light", "dark"):
                ctx, page, rec = self.session(390, scheme, self.adv_key)
                try:
                    page.goto(self.p.origin + "/train/")
                    page.wait_for_load_state("networkidle")
                    page.wait_for_selector("#cj-train-live .cj-lv-snap")
                    live = page.text_content("#cj-train-live")
                    for word in ("Watcher", "heartbeat", "Queue", "q/"):
                        self.assertNotIn(word, live)
                    self.check(page, rec, scheme)
                    page.screenshot(path=str(self.shots / f"train_advanced_390_{scheme}.png"), full_page=True)
                finally:
                    ctx.close()
        finally:
            self.env.write_live(live_doc(self.share))


if __name__ == "__main__":
    unittest.main()
