"""Predict jobs on the queue (desktop path, behind $CRYOEV_PREDICT_JOBS): push-predict -> run_job -> pull, with the
jobtest harness (temp share, temp laptop root and registry, fake predictor), plus the images-only dataset pack and
the live monitor's predict view.

Known answers: the pulled set equals the contract layout and its predictions are byte-identical to the golden ones.
Controls: a train job in the same queue runs exactly as before (same DONE.json keys, same d/<job>/ files, pulled
into training outputs); with the flag off the predict job stays queued, untouched, while the train job runs; a
running train job still shows no `kind` on the Live view.
"""
import json
import os
import subprocess
import sys
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobtest as T  # noqa: E402
from jobtest import C, run_job  # noqa: E402
from test_predict_set import FAKE_PRED, GOLDEN, GOLDEN_PRED, MODEL, PredictRoots, assert_contract_layout  # noqa: E402

import dataset_cache  # noqa: E402
import monitor as M  # noqa: E402
import registry  # noqa: E402
import sync  # noqa: E402

TRAIN = "job_20261102_t1"
TRAIN_DONE_KEYS = {"job", "status", "reason", "dataset", "bundle_id", "run", "run_status", "runs", "machine", "ended",
                   "allow_big", "max_hours", "time_capped", "runner_commit", "cryoEV_commit"}


class PackTests(T.TempRoots, unittest.TestCase):
    def test_images_only_pack(self):
        src = self.tmp / "ps"
        (src / "images").mkdir(parents=True)
        files = []
        for f in sorted((GOLDEN / "images").iterdir()):
            (src / "images" / f.name).write_bytes(f.read_bytes())
            files.append(src / "images" / f.name)
        (self.tmp / "out").mkdir()
        pack, sha, md5, md5s, reused = dataset_cache.write_pack(self.tmp / "out", src, files)
        self.assertEqual(sorted(md5s), ["images/0001.png", "images/0002.png", "images/0003.png"])
        self.assertEqual(sha, dataset_cache.content_sha256(md5s))
        job = {"dataset_sha256": sha, "dataset_pack": pack.name, "dataset_pack_md5": md5}
        (self.share / dataset_cache.DS / sha).mkdir(parents=True)
        for p in (pack, Path(str(pack) + ".md5")):
            (self.share / dataset_cache.DS / sha / p.name).write_bytes(p.read_bytes())
        (self.share / dataset_cache.DS / sha / C.READY).write_text("x")
        got, files_ok = dataset_cache.check_pack(self.share, job, {"dataset_pack_md5": md5})
        self.assertEqual(files_ok, md5s)
        dest = self.tmp / "x"
        dataset_cache.extract_pack(got, dest, files_ok)
        dataset_cache.verify_tree(dest, files_ok)
        self.assertEqual(sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()), sorted(md5s))
        self.assertEqual(dataset_cache.write_pack(self.tmp / "out", src, files)[4], True, "same images: reused")


class DesktopTests(PredictRoots, unittest.TestCase):
    def setUp(self):
        super().setUp()
        origin = mock.patch.object(sync, "commit_on_origin", lambda commit: None)
        origin.start()
        self.addCleanup(origin.stop)
        os.environ.update(FAKE_PRED_FROM=str(GOLDEN_PRED), **{C.PREDICT_ENV: "1"})
        self.ps = self.make()

    def run_job(self, *extra, **hook_kw):
        hook_kw.setdefault("predict_cmd", FAKE_PRED)
        return super().run_job(*extra, **hook_kw)

    def push_train(self):
        lst = self.tmp / "imgs.txt"
        lst.write_text("a_png\nb_png\n", encoding="utf-8")
        registry.new_job("train", "desktop", "t1", images=lst, who="pujan", root=self.reg, date="20261102")
        zp = T.make_bundle(self.bundles, args=["--model-size", "n", "--imgsz", "320"])
        return sync.push(TRAIN, zp, self.share, copier="shutil")

    def test_push_run_pull_with_a_train_job(self):
        tq = self.push_train()
        q = sync.push_predict(self.ps.name, self.share, copier="shutil", check_origin=False)
        jid = q.name
        self.assertEqual(jid, f"job_{time.strftime('%Y%m%d')}_{self.ps.name}")
        spec = C.read_json(q / C.JOB_FILE)
        self.assertEqual((spec["type"], spec["ps"], spec["model"], spec["n_images"], spec["max_hours"]),
                         ("predict", self.ps.name, MODEL, 3, sync.PREDICT_HOURS))
        self.assertEqual(C.read_json(tq / C.JOB_FILE)["type"], "train")
        with zipfile.ZipFile(q / spec["zip"]) as z:
            self.assertEqual(sorted(z.namelist()), sorted(f"{self.ps.name}/{n}" for n in
                                                          ("JOB.json", "request.json", "images.csv")))
        st = C.read_json(self.ps / "status.json")
        self.assertEqual((st["state"], st["where"], st["job"]), ("queued", "desktop", jid))
        rec = registry.current(root=self.reg)
        self.assertEqual((rec[self.ps.name]["where"], rec[self.ps.name]["job"]), ("desktop", jid))
        self.assertEqual((rec[jid]["type"], rec[jid]["where"], rec[jid]["n_images"]), ("predict", "desktop", 3))
        with self.assertRaisesRegex(sync.SyncError, "queued as"):
            sync.push_predict(self.ps.name, self.share, copier="shutil", check_origin=False)
        self.assertEqual(self.run_job(), 0)                # the oldest READY first, then the other
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.run_job(), 0)                # nothing left
        tdone = self.done(TRAIN)                           # control: the train job as before
        self.assertEqual(set(tdone), TRAIN_DONE_KEYS)
        self.assertEqual(tdone["status"], "done", tdone)
        self.assertEqual(sorted(p.name for p in (self.share / C.D / TRAIN).iterdir()),
                         sorted(["runs", "run.json", C.STATUS, "train.log", C.DONE]))
        pdone = self.done(jid)
        self.assertEqual(set(pdone), TRAIN_DONE_KEYS | {"job_type", "ps", "model", "n_done", "n_total"})
        self.assertEqual((pdone["status"], pdone["n_done"], pdone["n_total"], pdone["ps"]),
                         ("done", 3, 3, self.ps.name), pdone)
        d = self.share / C.D / jid
        self.assertEqual(sorted(p.name for p in d.iterdir()), sorted(["p", C.STATUS, C.PREDICT_LOG, C.DONE]))
        self.assertFalse((self.share / C.WORK / jid).exists(), "work folder kept after done")
        out = sync.pull(self.share, copier="shutil")
        self.assertEqual({o["job"]: o["status"] for o in out}, {TRAIN: "done", jid: "done"})
        assert_contract_layout(self, self.ps)
        for f in GOLDEN_PRED.iterdir():
            self.assertEqual((self.ps / "p" / MODEL / "predictions" / f.name).read_bytes(), f.read_bytes())
        st = C.read_json(self.ps / "status.json")
        self.assertEqual((st["state"], st["n_done"], st["where"], st["job"]), ("done", 3, "desktop", jid))
        cur = registry.current(root=self.reg)
        self.assertEqual((cur[jid]["status"], cur[self.ps.name]["status"], cur[TRAIN]["status"]),
                         ("done", "done", "done"))
        run = tdone["run"]
        self.assertTrue((self.tmp / "cai" / "training outputs" / "ds_tiny" / "runs" / run / "run.json").is_file())
        self.assertEqual(cur[run]["status"], "finished")   # the importer ran for the train job
        before = {p: p.read_bytes() for p in self.lap.rglob("*") if p.is_file()}
        self.assertEqual(sync.pull(self.share, copier="shutil"), [])
        self.assertEqual({p: p.read_bytes() for p in self.lap.rglob("*") if p.is_file()}, before)

    def test_flag_off_leaves_predict_jobs_queued(self):
        q = sync.push_predict(self.ps.name, self.share, copier="shutil", check_origin=False)
        self.push_train()
        os.environ.pop(C.PREDICT_ENV)
        ps2 = self.make()
        with self.assertRaisesRegex(sync.SyncError, "predict jobs are off"):
            sync.push_predict(ps2.name, self.share, copier="shutil", check_origin=False)
        before = sorted(p.name for p in q.iterdir())
        self.assertEqual(self.run_job(), 0)                # runs the train job only
        self.assertEqual(self.run_job(), 0)                # "nothing ready"
        self.assertEqual(self.done(TRAIN)["status"], "done")
        self.assertFalse((self.share / C.D / q.name).exists())
        self.assertEqual(sorted(p.name for p in q.iterdir()), before)
        os.environ[C.PREDICT_ENV] = "1"                    # control: on again, it runs
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.done(q.name)["status"], "done")

    def test_blocked_predict_pulls_as_failed(self):
        q = sync.push_predict(self.ps.name, self.share, copier="shutil", check_origin=False)
        with mock.patch.dict(os.environ, FAKE_PRED="fail_after:1"):
            self.assertEqual(self.run_job(), 1)
        done = self.done(q.name)
        self.assertEqual((done["status"], done["n_done"], done["n_total"]), ("blocked", 1, 3))
        self.assertIn("out of memory", done["reason"])
        out = sync.pull(self.share, copier="shutil")
        self.assertEqual((out[0]["status"], out[0]["copied"]), ("blocked", []))
        st = C.read_json(self.ps / "status.json")
        self.assertEqual((st["state"], st["n_done"]), ("failed", 1))
        self.assertIn("out of memory", st["error"])
        self.assertFalse((self.ps / "p").exists(), "a blocked job's partial p/ stays on the share")
        cur = registry.current(root=self.reg)
        self.assertEqual((cur[q.name]["status"], cur[self.ps.name]["status"]), ("blocked", "failed"))
        sync.retry(q.name, self.share)                     # control: retried, it runs and pulls to done
        self.assertEqual(self.run_job(), 0)
        sync.pull(self.share, copier="shutil")
        assert_contract_layout(self, self.ps)

    def test_gpu_busy_predict_is_blocked_and_recognised(self):
        q = sync.push_predict(self.ps.name, self.share, copier="shutil", check_origin=False)
        busy = {"total_mb": 49140, "apps": [[7, "python.exe", 900]]}
        self.assertEqual(self.run_job(gpu=busy), run_job.EXIT_GPU_BUSY)   # before it is touched: stays queued
        self.assertFalse((self.share / C.D / q.name).exists())

    def test_live_view_of_a_running_predict_job(self):
        q = sync.push_predict(self.ps.name, self.share, copier="shutil", check_origin=False)
        env = {**os.environ, **T.hooks_env(predict_cmd=FAKE_PRED), "FAKE_PRED_DELAY": "1.0"}
        p = subprocess.Popen([sys.executable, str(T.JOBS / "run_job.py"), "--queue-dir", str(self.share)], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        seen = []
        while p.poll() is None:
            r = M.read_queue(self.share)
            if r.get("running"):
                seen.append(r["running"])
            time.sleep(0.2)
        self.assertEqual(p.returncode, 0)
        mid = [s for s in seen if s.get("kind") == "predict" and 0 < s["n_done"] < 3]
        self.assertTrue(mid, seen[-3:])
        self.assertEqual((mid[0]["n_total"], mid[0]["ps"], mid[0]["hung"]), (3, self.ps.name, False))
        self.assertIsNotNone(mid[-1]["eta_s"])
        jobs = M.read_queue(self.share)["jobs"]
        self.assertEqual(jobs[0]["job"], q.name)
        self.assertRegex(jobs[0]["ready_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")

    def test_live_view_fixture_hung_and_train_control(self):
        job = "job_20261105_ps_x"
        q = self.share / C.Q / job
        q.mkdir(parents=True)
        (q / C.JOB_FILE).write_text(json.dumps({"type": "predict", "model": MODEL, "n_images": 3, "ps": "ps_x"}))
        (q / C.READY).write_text("x")
        (self.share / C.D / f".tmp_{job}").mkdir(parents=True)
        pred = self.share / C.WORK / job / "p" / MODEL / "predictions"
        pred.mkdir(parents=True)
        log = self.share / C.WORK / job / C.PREDICT_LOG
        log.write_text("x")
        for s in ("0001", "0002"):
            (pred / f"{s}.txt").write_text("")
        now = time.time()
        r = M.read_queue(self.share, now)["running"]
        self.assertEqual({k: r[k] for k in ("kind", "n_done", "n_total", "hung")},
                         {"kind": "predict", "n_done": 2, "n_total": 3, "hung": False})
        for f in pred.iterdir():                           # no new file for 20 minutes: hung
            os.utime(f, (now - 1200, now - 1200))
        self.assertTrue(M.read_queue(self.share, now)["running"]["hung"])
        share2 = self.tmp / "s2"                           # control: a running train job has no kind
        T.running_fixture(share2)
        r2 = M.read_queue(share2)["running"]
        self.assertNotIn("kind", r2)
        self.assertEqual(r2["epoch"], 4)


if __name__ == "__main__":
    unittest.main()
