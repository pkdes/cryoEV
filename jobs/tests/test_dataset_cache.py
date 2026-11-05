"""The dataset cache (jobs/dataset_cache.py, docs/JOBS.md "Dataset cache"): make_job_bundle's pack + thin zip,
sync.py push uploading a pack once per queue, run_job.py extracting and checking it. Temp roots, fake trainer.

Known answer: for one dataset, the folder the trainer finds is byte-identical for a fat (--single-zip) and a thin
bundle, and its images/labels/dataset.yaml equal the source files (the control: independent of fat vs thin).
"""
import contextlib
import hashlib
import io
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
from jobtest import C  # noqa: E402

import dataset_cache as DC  # noqa: E402
import sync  # noqa: E402

ARGS = ["--single-run", "--model-size", "n", "--imgsz", "320"]
SRC = {"images/train/a.png": b"\x89PNG a", "images/train/b.png": b"\x89PNG b", "images/val/c.png": b"\x89PNG c",
       "labels/train/a.txt": b"0 0.1 0.1 0.2 0.1 0.2 0.2\n", "labels/train/b.txt": b"0 0.3 0.3 0.4 0.3 0.4 0.4\n",
       "labels/val/c.txt": b"0 0.5 0.5 0.6 0.5 0.6 0.6\n"}


def md5(b):
    return hashlib.md5(b).hexdigest()


def dataset_dir(parent, name="dsq"):
    ds = Path(parent) / name
    for rel, data in SRC.items():
        (ds / rel).parent.mkdir(parents=True, exist_ok=True)
        (ds / rel).write_bytes(data)
    (ds / "dataset.yaml").write_text(f"path: C:/elsewhere/{name}\ntrain: images/train\nval: images/val\n"
                                     f"names:\n  0: EV\n", encoding="utf-8")
    return ds


def bundle(ds, out, *flags, name=None):
    """training/make_job_bundle.py as a user runs it. -> (job zip, stdout)."""
    before = set(Path(out).glob("*__job_*.zip")) if Path(out).is_dir() else set()
    argv = [sys.executable, "training/make_job_bundle.py", "--allow-dirty", "--dataset-dir", str(ds), "--out", str(out),
            *flags, *(["--dataset-name", name] if name else []), "--", *ARGS]
    r = subprocess.run(argv, cwd=T.REPO, capture_output=True, text=True)
    if r.returncode:
        raise AssertionError(r.stdout + r.stderr)
    new = set(Path(out).glob("*__job_*.zip")) - before
    assert len(new) == 1, (new, r.stdout)
    return new.pop(), r.stdout


def job_json(zp):
    with zipfile.ZipFile(zp) as z:
        return json.loads(next(z.read(n) for n in z.namelist() if n.endswith("/JOB.json")))


def push(share, job_id, zp):
    return sync.push(job_id, zp, share, copier="shutil", check_registry=False, check_origin=False)


class BundleFormatTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        import shutil
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_dc_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ds = dataset_dir(self.tmp)

    def test_single_zip_keeps_the_old_layout(self):
        zp, _ = bundle(self.ds, self.tmp / "o", "--single-zip")
        self.assertEqual([p.name for p in (self.tmp / "o").glob("*.zip")], [zp.name], "no pack written")
        with zipfile.ZipFile(zp) as z:
            names = set(z.namelist())
        self.assertEqual(names, {f"dsq/{r}" for r in SRC} | {"dsq/dataset.yaml", "dsq/JOB.json", "dsq/README.md"})
        job = job_json(zp)
        self.assertNotIn("dataset_sha256", job)
        self.assertEqual(job["md5"], {r: md5(b) for r, b in SRC.items()})
        self.assertTrue(Path(str(zp) + ".md5").is_file())

    def test_thin_bundle_and_pack(self):
        zp, out = bundle(self.ds, self.tmp / "o")
        job = job_json(zp)
        # the content key, computed here from the source files (known answer)
        want = hashlib.sha256("".join(f"{r}\t{md5(SRC[r])}\n" for r in sorted(SRC)).encode()).hexdigest()
        self.assertEqual(job["dataset_sha256"], want)
        pack = self.tmp / "o" / f"ds_{want[:12]}.zip"
        self.assertEqual(job["dataset_pack"], pack.name)
        self.assertEqual(job["dataset_pack_md5"], C.md5_file(pack))
        self.assertEqual(C.sidecar_md5(pack), C.md5_file(pack))
        self.assertEqual(job["md5"], {r: md5(b) for r, b in SRC.items()})
        with zipfile.ZipFile(zp) as z:
            self.assertEqual(set(z.namelist()), {"dsq/dataset.yaml", "dsq/JOB.json", "dsq/README.md"})
            self.assertIn(pack.name, z.read("dsq/README.md").decode("utf-8"))
        with zipfile.ZipFile(pack) as z:
            self.assertEqual(set(z.namelist()), set(SRC) | {DC.MANIFEST})
            man = json.loads(z.read(DC.MANIFEST))
            self.assertEqual(man["files"], {r: md5(b) for r, b in SRC.items()})
            self.assertEqual(man["dataset_sha256"], want)
            self.assertEqual({r: z.read(r) for r in SRC}, SRC)
        self.assertIn("new", out)
        # same data, another name: the pack is reused (same file, untouched); the key ignores the name
        mtime = pack.stat().st_mtime_ns
        zp2, out2 = bundle(self.ds, self.tmp / "o", name="other")
        self.assertEqual(job_json(zp2)["dataset_sha256"], want)
        self.assertEqual(pack.stat().st_mtime_ns, mtime)
        self.assertIn("reused", out2)
        self.assertEqual(len(list((self.tmp / "o").glob("ds_*.zip"))), 1)
        # one label changed: another key, another pack
        (self.ds / "labels/val/c.txt").write_bytes(b"0 0.5 0.5 0.6 0.5 0.6 0.7\n")
        zp3, _ = bundle(self.ds, self.tmp / "o", name="changed")
        self.assertNotEqual(job_json(zp3)["dataset_sha256"], want)
        self.assertEqual(len(list((self.tmp / "o").glob("ds_*.zip"))), 2)

    def test_a_mismatching_pack_in_out_is_not_reused(self):
        zp, _ = bundle(self.ds, self.tmp / "o")
        pack = self.tmp / "o" / job_json(zp)["dataset_pack"]
        Path(str(pack) + ".md5").write_text("0" * 32 + "  x\n")
        with self.assertRaisesRegex(AssertionError, "doesn't match this dataset"):
            bundle(self.ds, self.tmp / "o", name="again")


class ThinQueueTests(T.TempRoots, unittest.TestCase):
    def test_pack_pushed_once_and_both_jobs_run(self):
        ds = dataset_dir(self.tmp)
        zp1, _ = bundle(ds, self.bundles)
        time.sleep(1.1)                                         # the bundle name has a 1 s timestamp
        zp2, _ = bundle(ds, self.bundles)                       # a seed repeat: same data, same name
        sha = job_json(zp1)["dataset_sha256"]
        self.assertEqual(job_json(zp2)["dataset_sha256"], sha)
        copies = []
        real = C.copy_files

        def spy(src, dst, names, copier="auto"):
            copies.append(list(names))
            return real(src, dst, names, copier)
        with mock.patch.object(C, "copy_files", spy):
            push(self.share, "job_20261105_a", zp1)
            push(self.share, "job_20261105_b", zp2)
        pack = f"ds_{sha[:12]}.zip"
        self.assertEqual(copies, [[pack, pack + ".md5"], [zp1.name, zp1.name + ".md5"], [zp2.name, zp2.name + ".md5"]],
                         "the pack is copied once")
        ds_root = self.share / DC.DS
        self.assertEqual(sorted(p.name for p in ds_root.iterdir()), [C.INCOMING, sha])
        self.assertEqual(list((ds_root / C.INCOMING).iterdir()), [])
        self.assertEqual(sorted(p.name for p in (ds_root / sha).iterdir()), sorted([pack, pack + ".md5", C.READY]))
        for j, zp in (("job_20261105_a", zp1), ("job_20261105_b", zp2)):
            spec = C.read_json(self.share / C.Q / j / C.JOB_FILE)
            self.assertEqual((spec["dataset_sha256"], spec["dataset_pack"]), (sha, pack))
            self.assertEqual(sorted(p.name for p in (self.share / C.Q / j).iterdir()),
                             sorted([zp.name, zp.name + ".md5", C.JOB_FILE, C.READY]))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            sync.status(self.share)
        self.assertIn(f"{sha[:12]}", out.getvalue())
        self.assertIn("needed by 2 job(s)", out.getvalue())
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.run_job(), 0)
        for j in ("job_20261105_a", "job_20261105_b"):
            self.assertEqual(self.done(j)["status"], "done", self.done(j))
            self.assertEqual(C.read_json(self.share / C.D / j / C.STATUS)["dataset_sha256"], sha)
        self.assertTrue((ds_root / sha / C.READY).is_file(), "the runner leaves the cache alone")
        self.assertEqual([e["refs"] for e in DC.cache_entries(self.share)], [[]], "done jobs no longer need it")

    def test_known_answer_fat_and_thin_trees_identical(self):
        ds = dataset_dir(self.tmp)
        fat, _ = bundle(ds, self.bundles / "fat", "--single-zip")
        thin, _ = bundle(ds, self.bundles / "thin")
        push(self.share, "job_20261105_fat", fat)
        push(self.share, "job_20261105_thin", thin)
        snaps = self.tmp / "snaps"
        snaps.mkdir()
        with mock.patch.dict(os.environ, {"FAKE_SNAPSHOT": str(snaps)}):
            self.assertEqual(self.run_job("--job", "job_20261105_fat"), 0)
            self.assertEqual(self.run_job("--job", "job_20261105_thin"), 0)
        f = json.loads((snaps / "job_20261105_fat.json").read_text())
        t = json.loads((snaps / "job_20261105_thin.json").read_text())
        per_job = {"JOB.json", "README.md"}                   # bundle_id and the pack name differ by design
        self.assertEqual(set(f), set(t))
        self.assertNotIn(DC.MANIFEST, t)
        self.assertEqual({k: v for k, v in f.items() if k not in per_job},
                         {k: v for k, v in t.items() if k not in per_job})
        # control (known answer, independent of fat vs thin): the data equals the source files
        self.assertEqual({k: v for k, v in t.items() if k.split("/")[0] in ("images", "labels")},
                         {r: md5(b) for r, b in SRC.items()})
        self.assertEqual(set(t) - set(SRC), {"dataset.yaml"} | per_job)

    # ---------------------------------------------------------------- refusals
    def thin_job(self, job="job_20261105_t", **kw):
        zp, pack = T.make_thin_bundle(self.bundles, args=ARGS[1:], **kw)
        push(self.share, job, zp)
        self.zp = zp
        return job, self.share / DC.DS / job_json(zp)["dataset_sha256"] / pack.name

    def test_thin_job_runs(self):                               # the control for the refusals below
        job, _ = self.thin_job()
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.done(job)["status"], "done", self.done(job))

    def test_corrupted_pack_is_blocked_by_md5(self):
        job, pack = self.thin_job()
        data = bytearray(pack.read_bytes())
        data[len(data) // 2] ^= 0xFF
        pack.write_bytes(bytes(data))
        self.assertEqual(self.run_job(), 1)
        self.assertIn("dataset pack md5 mismatch", self.done(job)["reason"])
        self.assertFalse((self.share / C.WORK / job).exists(), "nothing extracted")

    def test_tampered_file_with_fixed_md5s_is_blocked_by_manifest(self):
        zp, pack = T.make_thin_bundle(self.bundles, args=ARGS[1:])
        new_md5 = T.rewrite_zip(pack, lambda m: m.update({"images/train/a.png": b"tampered"}))

        def fix(m):                                             # every md5 consistent with the tampered pack
            name = next(n for n in m if n.endswith("/JOB.json"))
            j = json.loads(m[name])
            j["dataset_pack_md5"] = new_md5
            m[name] = json.dumps(j).encode()
        T.rewrite_zip(zp, fix)
        push(self.share, "job_20261105_t", zp)
        self.assertEqual(self.run_job(), 1)
        reason = self.done("job_20261105_t")["reason"]
        self.assertIn("differ from the pack's MANIFEST", reason)
        self.assertIn("images/train/a.png", reason)

    def test_missing_pack_is_blocked_then_push_pack_and_retry(self):
        job, pack = self.thin_job()
        pack.parent.rename(pack.parent.with_name("moved_aside"))          # pruned while the job still needed it
        self.assertEqual(self.run_job(), 1)
        self.assertIn("dataset pack missing", self.done(job)["reason"])
        self.assertEqual(sync.main(["push-pack", "--bundle", str(self.zp), "--share", str(self.share),
                                    "--copier", "shutil"]), 0)
        self.assertTrue((pack.parent / C.READY).is_file())
        sync.retry(job, self.share)
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.done(job)["status"], "done", self.done(job))
        fat = T.make_bundle(self.tmp / "fat")
        self.assertEqual(sync.main(["push-pack", "--bundle", str(fat), "--share", str(self.share)]), 2,
                         "a fat bundle has no pack")

    def test_pack_without_ready_is_blocked(self):
        job, pack = self.thin_job()
        (pack.parent / C.READY).rename(pack.parent / "READY.off")
        self.assertEqual(self.run_job(), 1)
        self.assertIn("dataset pack missing", self.done(job)["reason"])

    def test_push_refuses_same_hash_different_pack(self):
        job, pack = self.thin_job()
        before = sorted(str(p.relative_to(self.share)) for p in self.share.rglob("*"))
        Path(str(pack) + ".md5").write_text("0" * 32 + f"  {pack.name}\n")   # the share holds "another" pack
        zp2, _ = T.make_thin_bundle(self.tmp / "b2", args=ARGS[1:], stamp="20261102-120001")
        with self.assertRaisesRegex(sync.SyncError, "same dataset hash, different pack"):
            push(self.share, "job_20261105_u", zp2)
        self.assertEqual(sorted(str(p.relative_to(self.share)) for p in self.share.rglob("*")), before,
                         "nothing copied or deleted")

    def test_push_refuses_a_bundle_without_its_pack(self):
        zp, pack = T.make_thin_bundle(self.bundles, args=ARGS[1:])
        pack.rename(pack.with_name("elsewhere.zip"))
        with self.assertRaisesRegex(sync.SyncError, "not found next to"):
            push(self.share, "job_20261105_t", zp)
        self.assertFalse((self.share / C.Q / "job_20261105_t").exists())
        self.assertFalse((self.share / DC.DS).exists())

    def test_stale_incoming_is_kept_and_a_new_one_used(self):
        zp, pack = T.make_thin_bundle(self.bundles, args=ARGS[1:])
        sha = job_json(zp)["dataset_sha256"]
        stale = self.share / DC.DS / C.INCOMING / sha
        stale.mkdir(parents=True)
        (stale / pack.name).write_bytes(b"half a pack")
        push(self.share, "job_20261105_t", zp)
        self.assertEqual((stale / pack.name).read_bytes(), b"half a pack", "never deleted")
        self.assertTrue((self.share / DC.DS / sha / C.READY).is_file())
        self.assertEqual(C.md5_file(self.share / DC.DS / sha / pack.name), C.md5_file(pack))
        self.assertEqual(self.run_job(), 0)


if __name__ == "__main__":
    unittest.main()
