"""jobs/predict_set.py: create a prediction set from the golden tiles into temp roots (BUILD_CONTRACTS.md section 2).

Known answers: the golden set (tests/fixtures/cj_golden/predict/ps_20261105_abc123) built again from its own tiles
gives the same images.csv rows (all but `source`, whose session is "golden") and the same request.json / status.json
keys. Each refusal comes with a control that passes: a 200-image set next to the 201 refused, the same mixed tile
sizes accepted once they sit in two sample labels, a JPEG named .png accepted (content, not extension).
Never touches the real registry or CryoAI-laptop (CRYOEV_LAPTOP_ROOT points at a temp folder). Helpers here are
shared by test_predict_local.py and test_predict_desktop.py.
"""
import csv
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobtest as T  # noqa: E402
from jobtest import C  # noqa: E402

sys.path.insert(0, str(T.REPO / "registry"))
import predict_set as PS  # noqa: E402
import registry  # noqa: E402

GOLDEN = T.REPO / "tests" / "fixtures" / "cj_golden" / "predict" / "ps_20261105_abc123"
MODEL = "v5_evnonev_y8m_20261029"
GOLDEN_PRED = GOLDEN / "p" / MODEL / "predictions"
FAKE_PRED = [sys.executable, str(T.TESTS / "fake_predictor.py")]
STATUS_KEYS = {"state", "n_done", "n_total", "where", "job", "error", "updated"}
REQUEST_KEYS = {"ps", "owner", "model", "conf", "max_det", "imgsz", "source", "created"}


def png(path, size=(64, 48), mode="L", value=128):
    from PIL import Image
    Image.new(mode, size, value).save(path)
    return path


def assert_contract_layout(tc, ps_dir: Path, model=MODEL, done=True):
    """The prediction_set folder exactly as BUILD_CONTRACTS.md section 2 (+ predict.log, the predictor's log)."""
    allowed = {"images", "images.csv", "request.json", "status.json", C.PREDICT_LOG} | ({C.PS_PRED} if done else set())
    names = {p.name for p in ps_dir.iterdir()}
    tc.assertTrue(names <= allowed, f"unexpected entries {names - allowed}")
    tc.assertTrue({"images", "images.csv", "request.json", "status.json"} <= names)
    imgs = sorted(p.name for p in (ps_dir / "images").iterdir())
    tc.assertTrue(all(len(n) == 8 and n[:4].isdigit() and n[4:] in (".png", ".jpg") for n in imgs), imgs)
    with open(ps_dir / "images.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tc.assertEqual(list(rows[0]), PS.CSV_COLUMNS)
    tc.assertEqual([r["file"] for r in rows], [f"images/{n}" for n in imgs])
    tc.assertEqual(set(C.read_json(ps_dir / "request.json")), REQUEST_KEYS)
    st = C.read_json(ps_dir / "status.json")
    tc.assertEqual(set(st), STATUS_KEYS)
    if done:
        tc.assertEqual((st["state"], st["n_done"], st["n_total"], st["error"]), ("done", len(imgs), len(imgs), None))
        tc.assertEqual({p.name for p in (ps_dir / C.PS_PRED).iterdir()}, {model})
        tc.assertEqual({p.name for p in (ps_dir / C.PS_PRED / model).iterdir()}, {"predictions", "summary.csv"})
        tc.assertEqual(sorted(p.name for p in (ps_dir / C.PS_PRED / model / "predictions").iterdir()),
                       [n[:4] + ".txt" for n in imgs])
    return rows


class PredictRoots(T.TempRoots):
    """Temp laptop root (registry + predict/), and the golden tiles under their original names in src/ with a
    MANIFEST.csv (sample S1, S1, S2; 1.4724 nm/px)."""

    def setUp(self):
        super().setUp()
        self.lap = self.tmp / "lap"
        self.reg, self.root = self.lap / "registry", self.lap / "predict"
        self.reg.mkdir(parents=True)
        (self.tmp / "cai" / "training outputs").mkdir(parents=True)
        (self.tmp / "cai" / "training images").mkdir()
        os.environ.update(CRYOEV_LAPTOP_ROOT=str(self.lap), CRYOEV_CRYOAI_ROOT=str(self.tmp / "cai"),
                          CRYOEV_PROCESSING_ROOT=str(self.tmp / "no_processing"))
        os.environ.pop("CRYOEV_REGISTRY", None)
        self.src = self.tmp / "src"
        self.src.mkdir()
        with open(GOLDEN / "images.csv", newline="", encoding="utf-8") as f:
            self.golden_rows = list(csv.DictReader(f))
        for r in self.golden_rows:
            shutil.copyfile(GOLDEN / r["file"], self.src / r["orig_name"])
        self.manifest = self.src / "MANIFEST.csv"
        self.manifest.write_text("output_name,sample_label,nm_per_px\n" + "".join(
            f"{r['orig_name']},{r['sample_label']},{r['nm_per_px']}\n" for r in self.golden_rows), encoding="utf-8")

    def make(self, src=None, **kw):
        args = dict(model=MODEL, owner="pujan", nm_confirmed_by="pujan", sample_label_column="manifest:sample_label",
                    nm_from_manifest=True, manifest=self.manifest, registry_root=self.reg, root=self.root)
        args.update(kw)
        return PS.create(src or self.src, **args)

    def assert_nothing_written(self):
        self.assertFalse(self.root.exists() and any(p.name.startswith("ps_") for p in self.root.iterdir()))
        self.assertFalse((self.root / PS.INCOMING).exists() and any((self.root / PS.INCOMING).iterdir()))
        self.assertFalse((self.reg / "prediction_set.jsonl").exists())


class CreateTests(PredictRoots, unittest.TestCase):
    def test_golden_rebuilt(self):
        ps = self.make(session="20261027")
        self.assertRegex(ps.name, r"^ps_\d{8}_[0-9a-f]{6}$")
        rows = assert_contract_layout(self, ps, done=False)
        keep = ["image", "file", "orig_name", "sample_label", "nm_per_px", "nm_confirmed_by"]
        self.assertEqual([{k: r[k] for k in keep} for r in rows], [{k: r[k] for k in keep} for r in self.golden_rows])
        self.assertEqual({r["source"] for r in rows}, {"session:20261027"})
        for r in rows:                                 # same bytes as the golden tiles; hard links on one drive
            self.assertEqual((ps / r["file"]).read_bytes(), (GOLDEN / r["file"]).read_bytes())
            self.assertGreaterEqual(os.stat(ps / r["file"]).st_nlink, 2)
        self.assertEqual(set(C.read_json(GOLDEN / "request.json")), REQUEST_KEYS)   # the golden agrees
        self.assertEqual(set(C.read_json(GOLDEN / "status.json")), STATUS_KEYS)
        req = C.read_json(ps / "request.json")
        self.assertEqual((req["ps"], req["model"], req["conf"], req["max_det"], req["imgsz"], req["owner"]),
                         (ps.name, MODEL, 0.25, 1000, None, "pujan"))
        self.assertEqual(req["source"], {"kind": "session", "session": "20261027", "dir": str(self.src.resolve())})
        st = C.read_json(ps / "status.json")
        self.assertEqual((st["state"], st["n_done"], st["n_total"], st["where"], st["job"]), ("queued", 0, 3, "laptop", None))
        rec = registry.current("prediction_set", self.reg)[ps.name]
        self.assertEqual((rec["status"], rec["n_images"], rec["model"], rec["path"]),
                         ("queued", 3, MODEL, f"<CryoAI-laptop>/predict/{ps.name}"))
        self.assertEqual(PS.find(ps.name, registry_root=self.reg), ps)            # via the record's path
        self.assertFalse((self.root / PS.INCOMING / ps.name).exists())

    def test_upload_copy_and_content_extension(self):
        jpg = self.src / "golden_0002.png"                 # JPEG bytes under a .png name: stored as .jpg
        from PIL import Image
        Image.open(GOLDEN / "images" / "0002.png").save(jpg, format="JPEG")
        ps = self.make(sample_label="EV prep A", sample_label_column=None, nm_per_px=1.4724, nm_from_manifest=False,
                       manifest=None, link=False)
        rows = assert_contract_layout(self, ps, done=False)
        self.assertEqual([r["file"] for r in rows], ["images/0001.png", "images/0002.jpg", "images/0003.png"])
        self.assertEqual({r["source"] for r in rows} | {r["sample_label"] for r in rows}, {"upload", "EV prep A"})
        self.assertEqual(os.stat(ps / "images" / "0001.png").st_nlink, 1, "link=False copies")
        self.assertEqual(C.read_json(ps / "request.json")["source"], {"kind": "upload"})
        src = str(self.src.resolve()).lower()              # the source sits under a protected root (processing)
        with mock.patch.object(PS, "protected_problem",
                               lambda p: "protected" if str(Path(p).resolve()).lower().startswith(src) else None):
            ps2 = self.make(nm_per_px=1.4724, nm_from_manifest=False)
        self.assertEqual(os.stat(ps2 / "images" / "0001.png").st_nlink, 1, "a protected source is never linked")

    def test_refusals_write_nothing(self):
        cases = {
            "nm/px is required": dict(nm_from_manifest=False),
            "nm-confirmed-by is required": dict(nm_confirmed_by=None),
            "unknown model": dict(model="v9_nope_20991231"),
            "owner": dict(owner="Not An Id"),
        }
        for msg, kw in cases.items():
            with self.subTest(msg), self.assertRaisesRegex(PS.PredictSetError, msg):
                self.make(**kw)
        self.assert_nothing_written()
        bad = self.tmp / "nm_missing.csv"                  # one image without nm/px in the manifest
        bad.write_text(self.manifest.read_text(encoding="utf-8").replace(",S2,1.4724", ",S2,"), encoding="utf-8")
        with self.assertRaisesRegex(PS.PredictSetError, "golden_0003.png: nm/px missing"):
            self.make(manifest=bad)
        self.assert_nothing_written()

    def test_content_refusals(self):
        for name, writer, msg in [
                ("text.png", lambda p: p.write_text("not an image"), "not a PNG or JPEG"),
                ("deep.png", lambda p: png(p, mode="I;16", value=4000), "16-bit PNG"),
                ("bits.png", lambda p: png(p, mode="1", value=1), "1-bit PNG"),
                ("notes.txt", lambda p: p.write_text("hello"), "not a PNG or JPEG")]:
            with self.subTest(name):
                d = self.tmp / f"c_{name.replace('.', '_')}"
                shutil.copytree(self.src, d)
                writer(d / name)
                with self.assertRaisesRegex(PS.PredictSetError, f"{name}: {msg}"):
                    self.make(src=d, sample_label="S", sample_label_column=None, nm_per_px=1.4724,
                              nm_from_manifest=False, manifest=None)
        self.assert_nothing_written()

    def test_mixed_sizes_refused_in_one_sample_only(self):
        d = self.tmp / "mixed"
        d.mkdir()
        for i in range(2):
            png(d / f"a{i}.png", (64, 48))
            png(d / f"b{i}.png", (48, 64))
        kw = dict(nm_per_px=1.0, nm_from_manifest=False, manifest=None, sample_label_column=None)
        with self.assertRaisesRegex(PS.PredictSetError, r"sample 'S': mixed tile sizes 64x48 x2 .*48x64 x2"):
            self.make(src=d, sample_label="S", **kw)
        self.assert_nothing_written()
        man = d / "MANIFEST.csv"                           # control: one size per sample label is fine
        man.write_text("output_name,sample_label\na0.png,A\na1.png,A\nb0.png,B\nb1.png,B\n", encoding="utf-8")
        ps = self.make(src=d, manifest=man, sample_label_column="manifest:sample_label",
                       **{k: v for k, v in kw.items() if k not in ("manifest", "sample_label_column")})
        self.assertEqual(C.read_json(ps / "status.json")["n_total"], 4)

    def test_more_than_200_refused(self):
        d = self.tmp / "many"
        d.mkdir()
        for i in range(201):
            png(d / f"t{i:03d}.png", (8, 8))
        kw = dict(sample_label="S", sample_label_column=None, nm_per_px=1.0, nm_from_manifest=False, manifest=None)
        with self.assertRaisesRegex(PS.PredictSetError, "201 files .* at most 200"):
            self.make(src=d, **kw)
        self.assert_nothing_written()
        (d / "t200.png").unlink()                          # control: 200 is allowed
        ps = self.make(src=d, **kw)
        self.assertEqual(sorted(p.name for p in (ps / "images").iterdir())[-1], "0200.png")

    def test_cli(self):
        env = {**os.environ}
        base = [sys.executable, "-m", "jobs.predict_set", "create", "--from-dir", str(self.src), "--model", MODEL,
                "--owner", "pujan", "--manifest", str(self.manifest), "--sample-label-from", "manifest:sample_label",
                "--nm-from-manifest", "--registry", str(self.reg), "--root", str(self.root)]
        r = subprocess.run(base, cwd=T.REPO, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("--nm-confirmed-by is required", r.stderr)
        self.assert_nothing_written()
        r = subprocess.run(base + ["--nm-confirmed-by", "pujan"], cwd=T.REPO, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        ps_id = r.stdout.strip().splitlines()[-1]
        self.assertTrue((self.root / ps_id / "status.json").is_file())
        self.assertEqual(registry.current("prediction_set", self.reg)[ps_id]["status"], "queued")


if __name__ == "__main__":
    unittest.main()
