"""Eval jobs on the queue (#45; docs/JOBS.md, "Eval jobs"): make_job_bundle --eval -> push-eval -> run_job (the
--eval-child with a stubbed validate_yolo_model, tests/fake_validate.py) -> pull into <CryoAI-laptop>/eval/<job>/.

Known answers: the stub's metrics come back unchanged in metrics.json and DONE.json; the child sees the bundle's own
weights (same md5) and the dataset's test images; it is called with overlap_mask=False, split test, imgsz 1440.
Controls: a train job in the same queue runs and pulls exactly as before (same DONE.json keys, the importer runs);
with the flag off the eval job stays queued, untouched; the import gate's control shows the block is real.
"""
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobtest as T  # noqa: E402
from jobtest import C, REPO, run_job  # noqa: E402

import sync  # noqa: E402  (puts registry/ on sys.path)
from registry import store as registry  # noqa: E402
from training import make_job_bundle as MB  # noqa: E402

FAKE_VAL = Path(__file__).resolve().parent / "fake_validate.py"
TRAIN = "job_20261112_t1"
EVAL = "job_20261112_eval_m1"
TRAIN_DONE_KEYS = {"job", "status", "reason", "dataset", "bundle_id", "run", "run_status", "runs", "machine", "ended",
                   "allow_big", "max_hours", "time_capped", "runner_commit", "cryoEV_commit"}
EVAL_DONE_KEYS = TRAIN_DONE_KEYS | {"job_type", "model", "split", "imgsz", "n_images", "metrics"}
YAML = "path: C:/elsewhere/ds_eval\nnames:\n  0: EV\n  1: non-EV\ntrain: images/train\nval: images/val\ntest: images/test\n"
FILES = {"images/train/a.png": b"pa", "labels/train/a.txt": b"0 0 0 1 1 1 0",
         "images/val/b.png": b"pb", "labels/val/b.txt": b"0 0 0 1 1 1 0",
         "images/test/c.png": b"pc", "labels/test/c.txt": b"0 0 0 1 1 1 0",
         "images/test/d.png": b"pd", "labels/test/d.txt": b"1 0 0 1 1 1 0"}


def tree(root: Path) -> dict:
    """{relative path: md5} of every file under root."""
    return {p.relative_to(root).as_posix(): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()} if root.exists() else {}


class EvalRoots(T.TempRoots):
    """Temp laptop root (registry), CryoAI root with a labelled dataset ds_eval (train / val / test), weights outside
    the repo, the eval flag on, agent identity env cleared."""

    def setUp(self):
        super().setUp()
        for k in ("CRYOEV_VIA", "CRYOEV_FOR", "CRYOEV_REGISTRY", C.PREDICT_ENV, "FAKE_VAL_FAIL", "FAKE_VAL_USED"):
            os.environ.pop(k, None)
        self.lap = self.tmp / "lap"
        self.reg = self.lap / "registry"
        self.reg.mkdir(parents=True)
        self.cai = self.tmp / "cai"
        self.ds = self.cai / "training outputs" / "ds_eval"
        for rel, data in FILES.items():
            (self.ds / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.ds / rel).write_bytes(data)
        (self.ds / "dataset.yaml").write_text(YAML, encoding="utf-8")
        (self.cai / "training images").mkdir()
        self.weights = self.tmp / "runs" / "r_m1" / "weights" / "best.pt"
        self.weights.parent.mkdir(parents=True)
        self.weights.write_bytes(b"fake weights " * 100)
        self.record = self.tmp / "val_record.json"
        os.environ.update(CRYOEV_LAPTOP_ROOT=str(self.lap), CRYOEV_CRYOAI_ROOT=str(self.cai),
                          CRYOEV_PROCESSING_ROOT=str(self.tmp / "no_processing"), FAKE_VAL_RECORD=str(self.record),
                          **{C.EVAL_ENV: "1"})

    def run_job(self, *extra, **hook_kw):
        hook_kw.setdefault("eval_validate", f"{FAKE_VAL}:validate")
        return super().run_job(*extra, **hook_kw)

    def eval_bundle(self, **kw):
        kw.setdefault("model", "m1")
        return MB.make_eval_bundle(self.ds, self.weights, self.bundles, allow_dirty=True, **kw)

    def queue_eval(self, job=EVAL, b=None):
        b = b or self.eval_bundle()
        return sync.push(job, b["zip"], self.share, copier="shutil", check_registry=False, check_origin=False,
                         job_type="eval", extra={"model": b["model"], "split": "test", "n_images": 2})

    def queue_train(self):
        return T.queue(self.share, TRAIN, T.make_bundle(self.bundles, args=["--model-size", "n", "--imgsz", "320"]))


class BundleTests(EvalRoots, unittest.TestCase):
    def test_eval_bundle_contents_and_pack_reuse(self):
        b = self.eval_bundle()
        with zipfile.ZipFile(b["zip"]) as z:
            self.assertEqual(sorted(z.namelist()), ["ds_eval/JOB.json", "ds_eval/README.md", "ds_eval/dataset.yaml",
                                                    "ds_eval/weights/best.pt"])
            job = json.loads(z.read("ds_eval/JOB.json"))
            text = z.read("ds_eval/dataset.yaml").decode()
            self.assertEqual(z.read("ds_eval/weights/best.pt"), self.weights.read_bytes())
        self.assertNotRegex(text, r"(?m)^path:")              # rewritten as for training
        self.assertIn("test: images/test", text)
        self.assertEqual({k: job[k] for k in ("job_type", "model", "weights", "weights_in", "split", "imgsz",
                                              "overlap_mask", "n_images", "run_experiments_args")},
                         {"job_type": "eval", "model": "m1", "weights": "weights/best.pt", "weights_in": "bundle",
                          "split": "test", "imgsz": 1440, "overlap_mask": False, "n_images": 2,
                          "run_experiments_args": []})
        self.assertEqual(job["weights_md5"], hashlib.md5(self.weights.read_bytes()).hexdigest())
        self.assertEqual(b["test_images"], ["c.png", "d.png"])
        self.assertEqual(sorted(job["md5"]), sorted(FILES))   # the full labelled pack, every split
        t = MB.make_bundle(self.ds, [], self.bundles, allow_dirty=True)   # a training bundle of the same folder
        self.assertEqual(t["dataset_sha256"], b["dataset_sha256"])
        self.assertEqual(t["pack"], b["pack"])
        self.assertTrue(t["pack_reused"], "the training bundle should reuse the eval bundle's pack")
        self.assertEqual(MB.default_model_name(self.weights), "r_m1")
        self.assertEqual(MB.default_model_name(REPO / "models" / "v5_x" / "best.pt"), "v5_x")

    def test_repo_weights_are_not_packed(self):
        tracked = sorted((p for p in (REPO / "models").glob("*/best.pt") if MB.tracked(p)),
                         key=lambda p: p.stat().st_size)
        if not tracked:
            self.skipTest("no tracked models/*/best.pt in this checkout")
        b = MB.make_eval_bundle(self.ds, tracked[0], self.bundles, allow_dirty=True)
        j = b["job_json"]
        self.assertEqual((j["weights_in"], j["weights"], j["model"]),
                         ("repo", tracked[0].relative_to(REPO).as_posix(), tracked[0].parent.name))
        with zipfile.ZipFile(b["zip"]) as z:
            self.assertFalse(any("weights/" in n for n in z.namelist()))

    def test_refusals(self):
        (self.ds / "dataset.yaml").write_text(YAML.replace("test: images/test\n", ""), encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "no test: split"):
            self.eval_bundle()
        (self.ds / "dataset.yaml").write_text(YAML, encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "not a .pt file"):
            MB.make_eval_bundle(self.ds, self.tmp / "missing.pt", self.bundles, allow_dirty=True)
        with self.assertRaisesRegex(SystemExit, "--model"):
            self.eval_bundle(model="bad name/..")
        for rel in ("images/test/c.png", "images/test/d.png"):
            (self.ds / rel).unlink()
        with self.assertRaisesRegex(SystemExit, "no images/test"):
            self.eval_bundle()

    def test_resolve_dataset(self):
        self.assertEqual(MB.resolve_dataset(self.ds), (self.ds.resolve(), "ds_eval"))
        self.assertEqual(MB.resolve_dataset("ds_eval", self.reg), (self.ds.resolve(), "ds_eval"))   # CryoAI fallback
        with self.assertRaisesRegex(SystemExit, "no dataset.yaml"):
            MB.resolve_dataset("ds_nope", self.reg)

    def test_cli_keeps_eval_flags_out_of_training(self):
        r = subprocess.run([sys.executable, str(REPO / "training" / "make_job_bundle.py"), "--dataset-dir",
                            str(self.ds), "--out", str(self.bundles), "--allow-dirty", "--weights", str(self.weights)],
                           capture_output=True, text=True, cwd=REPO)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("are --eval options", r.stderr)
        self.assertFalse(self.bundles.exists() and any(self.bundles.iterdir()))


class RunnerTests(EvalRoots, unittest.TestCase):
    def test_eval_and_train_jobs_end_to_end(self):
        tq = self.queue_train()
        b = self.eval_bundle()
        q = self.queue_eval(b=b)
        self.assertEqual(C.read_json(q / C.JOB_FILE)["type"], "eval")
        self.assertEqual(C.read_json(tq / C.JOB_FILE)["type"], "train")
        os.utime(q / C.READY, (time.time() + 5, time.time() + 5))   # the train job first, then the eval job
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.run_job(), 0)                # nothing left
        tdone = self.done(TRAIN)                           # control: the train job as before
        self.assertEqual(set(tdone), TRAIN_DONE_KEYS)
        self.assertEqual(tdone["status"], "done", tdone)
        done = self.done(EVAL)
        self.assertEqual(set(done), EVAL_DONE_KEYS)
        self.assertEqual((done["status"], done["job_type"], done["model"], done["split"], done["imgsz"],
                          done["n_images"], done["dataset"]), ("done", "eval", "m1", "test", 1440, 2, "ds_eval"), done)
        self.assertEqual(done["metrics"], {"mask_map50_95": 0.6, "mask_map50": 0.905, "box_map50_95": 0.7,
                                           "box_map50": 0.95})
        d = self.share / C.D / EVAL
        self.assertEqual(sorted(p.name for p in d.iterdir()), sorted([C.EVAL_OUT, C.EVAL_LOG, C.STATUS, C.DONE]))
        self.assertTrue((d / C.EVAL_OUT / "val" / "confusion_matrix.png").is_file())
        self.assertFalse((self.share / C.WORK / EVAL).exists(), "work folder kept after done")
        m = C.read_json(d / C.EVAL_OUT / C.METRICS)       # the testmap_<model>.json shape compare_eval_runs reads
        self.assertEqual((m["mask_map50_95"], m["box_map50_95"], m["per_class_mask_map50_95"], m["overlap_mask"]),
                         (0.6, 0.7, {"EV": 0.7, "non-EV": 0.5}, False))
        pc = ", ".join(f"{k} {v:.3f}" for k, v in m.get("per_class_mask_map50_95", {}).items())
        self.assertEqual(f"| m1 | {m['mask_map50_95']:.3f} | {m['box_map50_95']:.3f} | {pc} |",
                         "| m1 | 0.600 | 0.700 | EV 0.700, non-EV 0.500 |")
        rec = json.loads(self.record.read_text(encoding="utf-8"))   # what the child's validate was given
        self.assertEqual((rec["overlap_mask"], rec["split"], rec["imgsz"], rec["batch_size"], rec["device"]),
                         (False, "test", 1440, 8, "0"))
        self.assertEqual(rec["weights_md5"], hashlib.md5(self.weights.read_bytes()).hexdigest())   # round trip
        self.assertEqual(rec["test_images"], ["c.png", "d.png"])
        self.assertIn("test: images/test", rec["yaml"])
        self.assertFalse(rec["train_yolo_imported"] or rec["ultralytics_imported"], rec)

    def test_flag_off_leaves_eval_queued(self):
        self.queue_train()
        q = self.queue_eval()
        os.environ.pop(C.EVAL_ENV)
        before = tree(q)
        self.assertEqual(self.run_job(), 0)                # the train job runs
        self.assertEqual(self.done(TRAIN)["status"], "done")
        self.assertEqual(self.run_job(), 0)                # "nothing ready": the eval job waits
        self.assertFalse((self.share / C.D / EVAL).exists())
        self.assertEqual(tree(q), before)
        self.assertIsNone(run_job.pick(self.share))
        os.environ[C.EVAL_ENV] = "1"
        self.assertEqual(run_job.pick(self.share), q)

    def test_unknown_job_type_refused(self):
        zp = T.make_bundle(self.bundles, name="ds_x")
        T.rewrite_zip(zp, lambda m: m.__setitem__("ds_x/JOB.json", json.dumps(
            {**json.loads(m["ds_x/JOB.json"]), "job_type": "frobnicate"}).encode()))
        sync.push("job_20261112_odd", zp, self.share, copier="shutil", check_registry=False, check_origin=False,
                  job_type="frobnicate")
        self.assertEqual(self.run_job(), 1)
        done = self.done("job_20261112_odd")
        self.assertEqual((done["status"], done["job_type"]), ("blocked", "frobnicate"))
        self.assertIn("unknown job type", done["reason"])
        self.assertFalse((self.share / C.D / "job_20261112_odd" / "runs").exists())   # nothing was run
        self.assertEqual(sorted(p.name for p in (self.share / C.D / "job_20261112_odd").iterdir()),
                         sorted([C.STATUS, C.DONE]))

    def test_type_mismatch_refused(self):
        b = self.eval_bundle()                             # an eval bundle queued as a train job
        sync.push(EVAL, b["zip"], self.share, copier="shutil", check_registry=False, check_origin=False)
        self.assertEqual(self.run_job(), 1)
        self.assertIn("job type mismatch", self.done(EVAL)["reason"])
        self.assertFalse(self.record.exists(), "validate ran")

    def test_failed_val_is_blocked(self):
        self.queue_eval()
        os.environ["FAKE_VAL_FAIL"] = "1"
        self.assertEqual(self.run_job(), 1)
        done = self.done(EVAL)
        self.assertEqual((done["status"], done["job_type"], done["model"], done["metrics"]),
                         ("blocked", "eval", "m1", None))
        self.assertIn("eval exit 1", done["reason"])
        self.assertIn("CUDA out of memory (fake)", done["reason"])
        self.assertTrue((self.share / C.D / EVAL / C.EVAL_LOG).is_file())

    def test_overlap_mask_used_must_be_false(self):
        """The check reads what the validator ran with, not the kwarg: True or unreadable blocks the job."""
        for used, shown in (("true", "True"), ("none", "'unknown'")):
            job = f"job_20261112_eval_{used}"
            self.queue_eval(job)
            os.environ["FAKE_VAL_USED"] = used
            try:
                self.assertEqual(self.run_job("--job", job), 1)
            finally:
                os.environ.pop("FAKE_VAL_USED")
            done = self.done(job)
            self.assertEqual((done["status"], done["metrics"]), ("blocked", None))
            self.assertIn(f"not scored with overlap_mask=False (the validator used {shown})", done["reason"])
            self.assertFalse(json.loads(self.record.read_text(encoding="utf-8"))["overlap_mask"])   # kwarg was False

    def test_weights_changed_after_bundling_blocked(self):
        b = self.eval_bundle()
        T.rewrite_zip(b["zip"], lambda m: m.__setitem__("ds_eval/weights/best.pt", b"other weights"))
        self.queue_eval(b=b)
        self.assertEqual(self.run_job(), 1)
        self.assertIn("weights md5", self.done(EVAL)["reason"])
        self.assertFalse(self.record.exists(), "validate ran")


class ChildTests(EvalRoots, unittest.TestCase):
    """eval_child in this process, with validate_yolo_model monkeypatched (no subprocess, no hooks)."""

    def test_eval_child_with_patched_validate(self):
        sys.path.insert(0, str(FAKE_VAL.parent))
        import fake_validate
        calls = []

        def validate(*a, **kw):
            calls.append((a, kw))
            res = fake_validate._Results()
            res.overlap_mask_used = kw["overlap_mask"]   # as validate_yolo_model's on_val_end callback reports it
            return res
        out = self.tmp / "e"
        req = {"weights": str(self.weights), "data": str(self.ds / "dataset.yaml"), "split": "test", "imgsz": 1440,
               "batch": 8, "device": "0", "out": str(out), "model": "m1", "dataset": "ds_eval", "n_images": 2,
               "weights_md5": "x"}
        (self.tmp / "req.json").write_text(json.dumps(req), encoding="utf-8")
        with mock.patch.object(run_job, "load_validate", return_value=validate):
            self.assertEqual(run_job.eval_child(self.tmp / "req.json"), 0)
        (a, kw), = calls
        self.assertEqual(a, (str(self.weights), str(self.ds / "dataset.yaml")))
        self.assertEqual(kw, {"imgsz": 1440, "batch_size": 8, "device": "0", "split": "test", "project": str(out),
                              "name": "val", "overlap_mask": False})
        m = C.read_json(out / C.METRICS)
        self.assertEqual((m["schema"], m["model"], m["mask_map50_95"], m["per_class_box_map50_95"], m["overlap_mask"]),
                         ("eval_metrics/1", "m1", 0.6, {"EV": 0.8, "non-EV": 0.6}, False))
        self.assertEqual(run_job.metrics_from_val(fake_validate._Results(), req)["overlap_mask"], "unknown")

    def test_validate_entry_takes_overlap_mask(self):
        """train_yolo.validate_yolo_model passes overlap_mask=False through, over the checkpoint's train_args (read
        from source: importing train_yolo needs the training stack)."""
        import ast
        src = (REPO / "training" / "train_yolo.py").read_text(encoding="utf-8")
        fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "validate_yolo_model")
        self.assertIn("overlap_mask", [a.arg for a in fn.args.args])
        body = ast.get_source_segment(src, fn)
        self.assertIn("if overlap_mask is not None:\n        val_kwargs['overlap_mask'] = bool(overlap_mask)", body)
        self.assertIn('model.add_callback("on_val_end"', body)      # what it actually ran with
        self.assertIn('results.overlap_mask_used = used.get("overlap_mask")', body)

    @unittest.skipUnless(os.environ.get("CRYOEV_REAL_EVAL_TEST") == "1",
                         "a real Ultralytics val: set CRYOEV_REAL_EVAL_TEST=1, CRYOEV_REAL_EVAL_WEIGHTS=<pt> and "
                         "CRYOEV_REAL_EVAL_DATA=<dataset.yaml> (GPU; never in the gate)")
    def test_real_val(self):
        out = self.tmp / "e"
        req = {"weights": os.environ["CRYOEV_REAL_EVAL_WEIGHTS"], "data": os.environ["CRYOEV_REAL_EVAL_DATA"],
               "split": "test", "imgsz": 1440, "batch": 4, "device": os.environ.get("CRYOEV_REAL_EVAL_DEVICE", "0"),
               "out": str(out), "model": "real", "dataset": "real", "n_images": None, "weights_md5": None}
        (self.tmp / "req.json").write_text(json.dumps(req), encoding="utf-8")
        r = subprocess.run([sys.executable, str(REPO / "jobs" / "run_job.py"), "--eval-child",
                            str(self.tmp / "req.json")], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        m = C.read_json(out / C.METRICS)
        self.assertGreater(m["mask_map50_95"], 0)
        print(f"\n[real val] {m}")


class PullTests(EvalRoots, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        real = sync.registry_cmd

        def rec(*args):
            self.calls.append(args)
            return real(*args)
        p = mock.patch.object(sync, "registry_cmd", rec)
        p.start()
        self.addCleanup(p.stop)
        lst = self.tmp / "imgs.txt"
        lst.write_text("c_png\nd_png\n", encoding="utf-8")
        registry.new_job("eval", "desktop", "eval_m1", images=lst, root=self.reg, date="20261112")

    def run_eval(self, fail=False):
        self.queue_eval()
        if fail:
            os.environ["FAKE_VAL_FAIL"] = "1"
        self.assertEqual(self.run_job(), 1 if fail else 0)
        os.environ.pop("FAKE_VAL_FAIL", None)

    def test_eval_pull_never_imports_or_touches_training_outputs(self):
        self.run_eval()
        before = tree(self.cai)
        with mock.patch.object(sync, "copy_runs", side_effect=AssertionError("copy_runs called for an eval job")):
            out = sync.pull(self.share, copier="shutil")
        self.assertEqual(out, [{"job": EVAL, "status": "done", "copied": [f"eval/{EVAL}"], "skipped": [],
                                "not_pulled": {"files": 0, "bytes": 0, "paths": []}}])
        self.assertNotIn(("import",), self.calls)
        self.assertEqual([c[:3] for c in self.calls], [("set-status", EVAL, "done")])
        self.assertIn("mask mAP50-95 0.600", self.calls[0][-1])
        self.assertEqual(tree(self.cai), before, "training outputs changed")
        dest = self.lap / "eval" / EVAL
        self.assertEqual(sorted(p.name for p in dest.iterdir()),
                         sorted([C.METRICS, "testmap_m1.json", C.EVAL_LOG, "val"]))
        self.assertEqual((dest / "testmap_m1.json").read_bytes(), (dest / C.METRICS).read_bytes())
        self.assertEqual(registry.current("job", self.reg)[EVAL]["status"], "done")
        self.assertEqual(sync.pull(self.share, copier="shutil"), [])      # pulling again adds nothing

    def test_train_pull_still_imports(self):
        """Control: the same recorder sees the importer when a train job is pulled with the eval job."""
        lst = self.tmp / "t.txt"
        lst.write_text("a_png\n", encoding="utf-8")
        registry.new_job("train", "desktop", "t1", images=lst, root=self.reg, date="20261112")
        self.queue_train()
        self.run_eval()
        self.assertEqual(self.run_job(), 0)
        sync.pull(self.share, copier="shutil")
        self.assertEqual(self.calls.count(("import",)), 1)
        self.assertTrue((self.cai / "training outputs" / "ds_tiny" / "runs").is_dir())
        self.assertTrue((self.lap / "eval" / EVAL / C.METRICS).is_file())

    def test_unknown_type_pull_is_never_training(self):
        """A done job of a type this checkout doesn't know is not pulled at all (no copy, no importer, no ledger)."""
        d = self.share / C.D / "job_20261112_odd"
        d.mkdir(parents=True)
        (d / "runs" / "r1").mkdir(parents=True)
        (d / C.DONE).write_text(json.dumps({"job": d.name, "status": "done", "job_type": "frobnicate",
                                            "dataset": "ds_eval"}), encoding="utf-8")
        before = tree(self.cai)
        with self.assertRaisesRegex(sync.SyncError, "unknown job_type 'frobnicate'"):
            sync.pull(self.share, copier="shutil")
        self.assertEqual(self.calls, [])
        self.assertEqual(tree(self.cai), before)
        self.assertFalse(sync.ledger_path().exists())

    def test_blocked_eval_pull_copies_nothing(self):
        self.run_eval(fail=True)
        before = tree(self.cai)
        out = sync.pull(self.share, copier="shutil")
        self.assertEqual((out[0]["status"], out[0]["copied"]), ("blocked", []))
        self.assertNotIn(("import",), self.calls)
        self.assertEqual(tree(self.cai), before)
        self.assertFalse((self.lap / "eval" / EVAL).exists())
        self.assertEqual(registry.current("job", self.reg)[EVAL]["status"], "blocked")


class PushEvalTests(EvalRoots, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.watch(True)

    def watch(self, eval_jobs):
        """The share's watch.json as the desktop watcher writes it (eval_jobs = its CRYOEV_EVAL_JOBS)."""
        T.watch_fixture(self.share, state="idle")
        w = C.read_json(self.share / C.WATCH_JSON)
        if eval_jobs is not None:
            w["eval_jobs"] = eval_jobs
        C.write_json(self.share / C.WATCH_JSON, w)

    def push(self, **kw):
        kw.setdefault("weights", self.weights)
        kw.setdefault("dataset", "ds_eval")
        return sync.push_eval(self.share, model="m1", copier="shutil", check_origin=False, out=self.bundles,
                              allow_dirty=True, **kw)

    def test_flag_off_refused(self):
        os.environ.pop(C.EVAL_ENV)
        with self.assertRaisesRegex(sync.SyncError, "eval jobs are off"):
            self.push()
        self.assertEqual(list((self.share / C.Q).iterdir()), [])

    def test_push_eval_queues_and_makes_the_registry_job(self):
        q = self.push()
        jid = f"job_{time.strftime('%Y%m%d')}_eval_m1"
        self.assertEqual(q.name, jid)
        spec = C.read_json(q / C.JOB_FILE)
        self.assertEqual((spec["type"], spec["model"], spec["split"], spec["n_images"], spec["max_hours"]),
                         ("eval", "m1", "test", 2, sync.EVAL_HOURS))
        self.assertTrue((q / C.READY).is_file())
        rec = registry.current("job", self.reg)[jid]
        self.assertEqual((rec["type"], rec["where"], rec["n_images"], rec["progress"]["unit"]),
                         ("eval", "desktop", 2, "images scored"))
        with self.assertRaisesRegex(sync.SyncError, "already in the queue"):
            self.push()

    def test_desktop_flag_from_watch_json(self):
        """on: queued (test_push_eval_queues...); off or missing key: refused; unreadable: refused unless forced."""
        for value in (False, None):
            self.watch(value)
            with self.assertRaisesRegex(sync.SyncError, r"the desktop has eval jobs off \(CRYOEV_EVAL_JOBS\); set it "
                                                        r"there first"):
                self.push()
        (self.share / C.WATCH_JSON).write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(sync.SyncError, "--force-unwatched"):
            self.push()
        (self.share / C.WATCH_JSON).unlink()
        with self.assertRaisesRegex(sync.SyncError, "--force-unwatched"):
            self.push()
        self.assertEqual(list((self.share / C.Q).iterdir()), [])
        q = self.push(force_unwatched=True)
        self.assertEqual(C.read_json(q / C.JOB_FILE)["type"], "eval")

    def test_bundle_or_weights_not_both(self):
        b = self.eval_bundle()
        with self.assertRaisesRegex(sync.SyncError, "either --bundle"):
            self.push(bundle=b["zip"])
        t = MB.make_bundle(self.ds, [], self.bundles, allow_dirty=True)
        with self.assertRaisesRegex(sync.SyncError, "not an eval bundle"):
            self.push(bundle=t["zip"], weights=None, dataset=None)

    def test_lock_held_refused(self):
        from data_utils.oslock import acquire_lock
        (self.lap / "jobs").mkdir()
        lock = acquire_lock(self.lap / sync.EVAL_LOCK, 0)
        try:
            with self.assertRaisesRegex(sync.SyncError, "another push-eval"):
                self.push()
        finally:
            lock.close()
        self.assertEqual(list((self.share / C.Q).iterdir()), [])


class MonitorTests(EvalRoots, unittest.TestCase):
    def test_finished_and_running_eval(self):
        import monitor as M
        self.queue_eval()
        self.assertEqual(self.run_job(), 0)
        f = M.read_queue(self.share)["finished"][0]
        self.assertEqual((f["job"], f["kind"], f["model"], f["status"], f["metrics"]["mask_map50_95"]),
                         (EVAL, "eval", "m1", "done", 0.6))
        job2 = "job_20261112_eval_m2"
        self.queue_eval(job2, self.eval_bundle(model="m2"))
        (self.share / C.D / f".tmp_{job2}").mkdir()      # running, as run_job leaves it mid-job
        (self.share / C.WORK / job2).mkdir(parents=True)
        (self.share / C.WORK / job2 / C.EVAL_LOG).write_text("x", encoding="utf-8")
        r = M.read_queue(self.share)["running"]
        self.assertEqual((r["job"], r["kind"], r["model"], r["n_total"], r["n_done"], r["hung"]),
                         (job2, "eval", "m2", 2, None, False))
        self.assertGreaterEqual(r["elapsed_s"], 0)
        self.assertTrue(M.running_eval(self.share, job2, {}, now=time.time() + M.EVAL_HUNG_S + 5)["hung"])


class ImportGate(unittest.TestCase):
    """The desktop watcher restarts on code changes and imports run_job: neither may need torch / Ultralytics."""
    PRELUDE = "import sys; sys.modules['ultralytics'] = None; sys.modules['torch'] = None\n"

    def py(self, code):
        env = {k: v for k, v in os.environ.items() if not k.startswith("CRYOEV_JOBS_TEST")}
        return subprocess.run([sys.executable, "-c", self.PRELUDE + textwrap.dedent(code)], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=120)

    def test_watch_and_run_job_import_without_ultralytics(self):
        r = self.py("""
            import jobs.watch, jobs.run_job
            import run_job
            assert run_job.eval_child and run_job.load_validate
            bad = sorted(m for m in ('train_yolo', 'ultralytics', 'torch', 'cv2') if sys.modules.get(m) is not None)
            print('imported:', bad)
            """)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("imported: []", r.stdout)

    def test_control_the_block_is_real(self):
        r = self.py("import ultralytics\n")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("import of ultralytics halted", r.stderr)


if __name__ == "__main__":
    unittest.main()
