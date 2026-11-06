"""training/make_job_bundle.py: make_bundle() (the callable that jobs/sync.py queue_train uses) writes exactly what the
CLI writes, and the CLI after the refactor writes exactly what it wrote before it.

Known-answer control: the CLI as it was before make_bundle existed (git object BEFORE_REFACTOR, run from its source)
and the CLI now, on the same temp dataset with the clock frozen, give byte-identical files (job zip, dataset pack,
.md5 sidecars, _README.md) and the same stdout. make_bundle() is then held to the same bytes. With the clock frozen
the timestamps (bundle stamp, `created`, zip member times) are equal too, so the comparison is byte for byte.
If the old git object is missing (a shallow clone), only the old-vs-new part is skipped.

Temp folders only; --allow-dirty / allow_dirty=True (the worktree may have uncommitted code under test).

    python -m unittest discover -s tests -p "test_make_bundle*" -v
"""
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from training import make_job_bundle as mjb  # noqa: E402

BEFORE_REFACTOR = "43bfbb0aaee3709982b10d362e4fdf5e9ad25ae5"   # main just before make_bundle() was factored out
SCRIPT = REPO / "training" / "make_job_bundle.py"
FIXED = time.mktime((2026, 11, 5, 21, 30, 15, 0, 0, -1))
TRACKED_W = REPO / "models" / "v4_y8n_20260924" / "best.pt"


@contextlib.contextmanager
def frozen_clock():
    """time.time / localtime / strftime pinned to FIXED (zip member times, the bundle stamp and `created`)."""
    lt, sf = time.localtime, time.strftime
    with mock.patch.object(time, "time", lambda: FIXED), \
            mock.patch.object(time, "localtime", lambda secs=None: lt(FIXED if secs is None else secs)), \
            mock.patch.object(time, "strftime", lambda fmt, t=None: sf(fmt, lt(FIXED) if t is None else t)):
        yield


def make_dataset(root: Path) -> Path:
    ds = root / "tiny_ds"
    for s in ("train", "val", "test"):
        (ds / "images" / s).mkdir(parents=True)
        (ds / "labels" / s).mkdir(parents=True)
        (ds / "images" / s / f"{s}_0.png").write_bytes(b"\x89PNG fake " + s.encode())
        (ds / "images" / s / f"{s}_1.jpg").write_bytes(b"\xff\xd8 fake " + s.encode())
        (ds / "labels" / s / f"{s}_0.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
    (ds / "dataset.yaml").write_text(
        "# YOLO dataset config\npath: C:/elsewhere/tiny_ds\ntrain: images/train\n"
        "val: C:/elsewhere/tiny_ds/images/val\ntest: images/test\n\nnames:\n  0: EV\n  1: non-EV\n", encoding="utf-8")
    return ds


def snap(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def old_source() -> str | None:
    r = subprocess.run(["git", "show", f"{BEFORE_REFACTOR}:training/make_job_bundle.py"], cwd=REPO,
                       capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 and "def main" in r.stdout else None


def run_cli(main, argv) -> str:
    out = io.StringIO()
    with mock.patch.object(sys, "argv", ["make_job_bundle.py", *argv]), contextlib.redirect_stdout(out):
        main()
    return out.getvalue()


def old_main(src: str):
    """The pre-refactor script's main(), compiled from its source with this file's path as __file__ (so REPO and
    the imports resolve as they did when it was the script)."""
    ns = {"__file__": str(SCRIPT), "__name__": "cj_make_job_bundle_before"}
    exec(compile(src, str(SCRIPT), "exec"), ns)
    return ns["main"]


class CliIdentity(unittest.TestCase):
    """Three variants: thin with a repo (tracked) --weights, --single-zip with a packed weights file given as
    --weights=PATH, and --dataset-name."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_mb_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.ds = make_dataset(self.tmp)
        self.w = self.tmp / "my_weights.pt"
        self.w.write_bytes(b"not really weights")
        self.variants = {
            "thin": ([], ["--single-run", "--model-size", "m", "--batch", "8", "--rect", "0",
                          "--weights", str(TRACKED_W)], {}),
            "single": (["--single-zip"], ["--single-run", "--model-size", "n", f"--weights={self.w}"],
                       {"single_zip": True}),
            "named": (["--dataset-name", "renamed_ds"], ["--single-run", "--seed", "7"],
                      {"dataset_name": "renamed_ds"}),
        }

    def cli(self, main, out: Path, flags, exp) -> str:
        with frozen_clock():
            return run_cli(main, ["--allow-dirty", "--dataset-dir", str(self.ds), "--out", str(out), *flags,
                                  "--", *exp])

    def test_cli_now_equals_cli_before(self):
        """The control: the refactor changed nothing the CLI writes or prints."""
        src = old_source()
        if src is None:
            self.skipTest(f"git object {BEFORE_REFACTOR[:12]} not available (shallow clone?)")
        before = old_main(src)
        for v, (flags, exp, _) in self.variants.items():
            with self.subTest(variant=v):
                a, b = self.tmp / f"old_{v}", self.tmp / f"new_{v}"
                out_a = self.cli(before, a, flags, exp)
                out_b = self.cli(mjb.main, b, flags, exp)
                self.assertEqual(out_a.replace(str(a), "<OUT>"), out_b.replace(str(b), "<OUT>"))
                sa, sb = snap(a), snap(b)
                self.assertEqual(sorted(sa), sorted(sb))
                self.assertGreaterEqual(len(sa), 3 if v == "single" else 5)
                for name in sa:
                    self.assertEqual(sa[name], sb[name], name)   # byte for byte, every file

    def test_make_bundle_equals_cli(self):
        for v, (flags, exp, kw) in self.variants.items():
            with self.subTest(variant=v):
                a, b = self.tmp / f"cli_{v}", self.tmp / f"fn_{v}"
                self.cli(mjb.main, a, flags, exp)
                with frozen_clock():
                    res = mjb.make_bundle(self.ds, exp, b, allow_dirty=True, **kw)
                sa, sb = snap(a), snap(b)
                self.assertEqual(sorted(sa), sorted(sb))
                for name in sa:
                    self.assertEqual(sa[name], sb[name], name)
                # the returned dict describes what was written
                self.assertEqual(res["zip"].parent, b)
                self.assertEqual(res["md5"], hashlib.md5(res["zip"].read_bytes()).hexdigest())
                self.assertTrue(res["md5_path"].read_text().startswith(res["md5"]))
                self.assertTrue(res["readme"].is_file())
                with zipfile.ZipFile(res["zip"]) as z:
                    job = json.loads(z.read(f"{res['name']}/JOB.json"))
                self.assertEqual(job, res["job_json"])
                self.assertEqual(job["bundle_id"], res["zip"].stem)
                self.assertEqual(job["run_experiments_args"], ["--dataset", res["name"]] + res["exp"])
                if v == "single":
                    self.assertIsNone(res["pack"])
                    self.assertIsNone(res["dataset_sha256"])
                    self.assertEqual(job["run_experiments_args"][-2:],
                                     ["--weights", "../CryoAI/training outputs/tiny_ds/weights/my_weights.pt"])
                else:
                    self.assertEqual(res["pack"].name, job["dataset_pack"])
                    self.assertEqual(res["dataset_sha256"], job["dataset_sha256"])
                    self.assertEqual(res["pack_bytes"], res["pack"].stat().st_size)
                    self.assertFalse(res["pack_reused"])
                if v == "thin":                                    # a repo weights file: the desktop's own copy
                    self.assertIn("models/v4_y8n_20260924/best.pt", res["exp"])
                if v == "named":
                    self.assertEqual(res["name"], "renamed_ds")

    def test_second_bundle_reuses_the_pack(self):
        out = self.tmp / "o"
        r1 = mjb.make_bundle(self.ds, ["--single-run"], out, allow_dirty=True)
        time.sleep(1.1)                                            # a new stamp, so a second job zip
        r2 = mjb.make_bundle(self.ds, ["--single-run"], out, allow_dirty=True)
        self.assertNotEqual(r1["zip"], r2["zip"])
        self.assertEqual(r1["pack"], r2["pack"])
        self.assertTrue(r2["pack_reused"])

    def test_refusals_are_systemexit_and_write_nothing(self):
        out = self.tmp / "never"
        with mock.patch.object(mjb, "uncommitted_code", lambda: ["jobs/sync.py"]):
            with self.assertRaisesRegex(SystemExit, "uncommitted changes in jobs/sync.py"):
                mjb.make_bundle(self.ds, ["--single-run"], out)
        with self.assertRaisesRegex(SystemExit, "has no"):
            mjb.make_bundle(self.tmp / "missing", ["--single-run"], out, allow_dirty=True)
        with self.assertRaisesRegex(SystemExit, "--epoch"):
            mjb.make_bundle(self.ds, ["--single-run", "--epoch", "5"], out, allow_dirty=True)
        with self.assertRaisesRegex(SystemExit, "leave --dataset out"):
            mjb.make_bundle(self.ds, ["--dataset", "x"], out, allow_dirty=True)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
