"""sync.py pull leaves runs/<run>/weights/epoch*.pt on the share unless --all-weights (TODO #52).

A fake finished job (d/<job>/runs/<run>/ with best.pt, last.pt, epoch0/50/100.pt, CSVs, run.json) is pulled into
temp laptop/CryoAI roots, never the real ones. Known answers:
- control: --all-weights copies the run folder exactly as before (every file, byte-identical to the share);
- default: the same minus weights/epoch*.pt, and the registry import gives the same records as --all-weights;
- the ledger line lists the files left (count, bytes, paths); a run without best.pt still gets last.pt.
"""
import hashlib
import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobtest as T  # noqa: E402
from jobtest import C  # noqa: E402

sys.path.insert(0, str(T.REPO / "registry"))
import registry  # noqa: E402
import sync  # noqa: E402

JOB = "job_20261105_w1"
RUN = "2026-11-05_120000_y8n_320_8b_100ep_Run_1"
DS = "ds_w"
EPOCHS = {"epoch0.pt": 3000, "epoch50.pt": 3001, "epoch100.pt": 3002}
VOLATILE = ("logged",)                       # when registry.py wrote the line (import, set-status, job new)


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(Path(folder).rglob("*")) if p.is_file()}


def fake_job(share, best=True, status="done"):
    """q/<job>/ + d/<job>/ as run_job.py leaves a finished job, with a full results tree. -> the run folder."""
    share = Path(share)
    (share / C.Q / JOB).mkdir(parents=True, exist_ok=True)
    (share / C.Q / JOB / C.JOB_FILE).write_text(json.dumps({"job": JOB}), encoding="utf-8")
    d = share / C.D / JOB
    run = d / "runs" / RUN
    (run / "weights").mkdir(parents=True)
    (run / "predictions" / "test").mkdir(parents=True)
    for name, size in EPOCHS.items():
        (run / "weights" / name).write_bytes(name.encode() * (size // len(name)) + b"x" * (size % len(name)))
    if best:
        (run / "weights" / "best.pt").write_bytes(b"best weights")
    (run / "weights" / "last.pt").write_bytes(b"last weights")
    (run / "results.csv").write_text("epoch,metrics/mAP50-95(M)\n1,0.1\n2,0.2\n", encoding="utf-8")
    (run / "per_class_metrics_test.txt").write_text("EV 0.5\n", encoding="utf-8")
    (run / "args.yaml").write_text("epochs: 100\nimgsz: 320\nbatch: 8\n", encoding="utf-8")
    (run / "predictions" / "test" / "a.txt").write_text("0 0.9 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
    (run / "epoch_notes.pt.txt").write_text("not a checkpoint", encoding="utf-8")   # outside weights/: copied
    (run / "run.json").write_text(json.dumps({
        "schema": "run/1", "status": "finished" if best else "error", "error": None if best else "cancelled",
        "config": {"arch": "yolov8n-seg", "imgsz": 320, "batch": 8, "epochs": 100},
        "machine": "fake-desktop", "metrics": {"mask_map50": .61}}), encoding="utf-8")
    (d / C.DONE).write_text(json.dumps({"job": JOB, "status": status, "reason": None if status == "done" else "x",
                                        "run": RUN, "dataset": DS, "machine": "fake-desktop"}), encoding="utf-8")
    return run


def stable(cur):
    """registry.current() without the import's own timestamps."""
    return {k: {f: v for f, v in r.items() if f not in VOLATILE} for k, r in cur.items()}


class PullWeightsTests(T.TempRoots, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.n = 0

    def roots(self):
        """Fresh temp laptop + CryoAI roots (env) with the train job in the registry. -> (laptop, cryoai)."""
        self.n += 1
        lap, cai = self.tmp / f"lap{self.n}", self.tmp / f"cai{self.n}"
        (lap / "registry").mkdir(parents=True)
        (cai / "training outputs").mkdir(parents=True)
        (cai / "training images").mkdir()
        os.environ.update(CRYOEV_LAPTOP_ROOT=str(lap), CRYOEV_CRYOAI_ROOT=str(cai),
                          CRYOEV_PROCESSING_ROOT=str(self.tmp / "no_processing"))
        os.environ.pop("CRYOEV_REGISTRY", None)
        lst = self.tmp / "imgs.txt"
        lst.write_text("a_png\n", encoding="utf-8")
        registry.new_job("train", "desktop", "w1", images=lst, who="pujan", root=lap / "registry", date="20261105")
        return lap, cai

    def pull_into_fresh_roots(self, copier="shutil", **kw):
        lap, cai = self.roots()
        out = sync.pull(self.share, copier=copier, **kw)
        dest = cai / "training outputs" / DS / "runs" / RUN
        led = [json.loads(x) for x in (lap / "jobs" / "pulled.jsonl").read_text(encoding="utf-8").splitlines()]
        return out, dest, led, registry.current(root=lap / "registry")

    def check(self, copier):
        src = fake_job(self.share)
        on_share = snapshot(src)
        out_all, dest_all, led_all, reg_all = self.pull_into_fresh_roots(copier, all_weights=True)
        out, dest, led, reg = self.pull_into_fresh_roots(copier)
        # control: --all-weights = the old behaviour, every file byte-identical
        self.assertEqual(snapshot(dest_all), on_share)
        self.assertEqual(out_all[0]["not_pulled"], {"files": 0, "bytes": 0, "paths": []})
        # default: everything but weights/epoch*.pt, byte-identical
        want = {k: v for k, v in on_share.items() if not (k.startswith("weights/epoch") and k.endswith(".pt"))}
        self.assertEqual(len(on_share) - len(want), len(EPOCHS))
        self.assertEqual(snapshot(dest), want)
        self.assertTrue((dest / "weights" / "best.pt").is_file() and (dest / "weights" / "last.pt").is_file())
        # what was left, in the result and the ledger
        left = {"files": 3, "bytes": sum(EPOCHS.values()),
                "paths": sorted(f"{RUN}/weights/{n}" for n in EPOCHS)}
        self.assertEqual(out[0]["not_pulled"], left)
        self.assertEqual((led[0]["not_pulled"], led[0]["all_weights"]), (left, False))
        self.assertEqual((led_all[0]["not_pulled"]["files"], led_all[0]["all_weights"]), (0, True))
        # the registry import sees the same thing either way
        self.assertEqual(stable(reg), stable(reg_all))
        self.assertEqual(reg[RUN]["status"], "finished")
        self.assertEqual(reg[JOB]["status"], "done")
        # nothing was removed from the share, and nothing is left in .pull_tmp
        self.assertEqual(snapshot(src), on_share)
        self.assertEqual(list((dest.parent.parent / sync.PULL_TMP).iterdir()), [])

    def test_default_skips_epoch_checkpoints_shutil(self):
        self.check("shutil")

    @unittest.skipUnless(os.name == "nt", "robocopy is Windows only")
    def test_default_skips_epoch_checkpoints_robocopy(self):
        self.check("robocopy")

    def test_run_without_best_still_gets_last(self):
        fake_job(self.share, best=False, status="cancelled")
        out, dest, led, reg = self.pull_into_fresh_roots()
        self.assertEqual(out[0]["status"], "cancelled")
        self.assertEqual(sorted(p.name for p in (dest / "weights").iterdir()), ["last.pt"])
        self.assertEqual(led[0]["not_pulled"]["files"], 3)
        self.assertEqual(reg[RUN]["status"], "failed")

    def test_cli_flag(self):
        seen = []
        with mock.patch.object(sync, "pull", lambda *a: seen.append(a)):
            sync.main(["pull", "--share", str(self.share)])
            sync.main(["pull", "--share", str(self.share), "--all-weights"])
        self.assertEqual([a[3] for a in seen], [False, True])

    def test_left_on_share_only_matches_weights_epochs(self):
        run = fake_job(self.share)
        (run / "weights" / "epochs.txt").write_text("x")             # not epoch*.pt
        self.assertEqual([p.name for p in sync.left_on_share(run)], ["epoch0.pt", "epoch100.pt", "epoch50.pt"])
        self.assertEqual(sync.left_on_share(run, all_weights=True), [])
        shutil.rmtree(run / "weights")
        self.assertEqual(sync.left_on_share(run), [])


if __name__ == "__main__":
    unittest.main()
