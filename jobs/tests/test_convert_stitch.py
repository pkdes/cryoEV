"""jobs/convert_raw.py round 7: the stitch pass (montages/ + layout.json + stitch.json), acquisition.json and
--stitch-only (BUILD_CONTRACTS.md section 13).

  * Stub (always runs): a stitch that fails is a warning (montages/stitch.json "failed"), never a failed conversion,
    and the tiles are exactly those of a run without stitching; --montages-out inside the tiles is refused;
    acquisition.json has the schema and no header, serial, site or path; --stitch-only refuses without a MANIFEST
    or once the raw inputs are gone; the CLI stays one JSON object.
  * Real (needs CJ_PROCESSING_REPO): known answer -- the montage JPG is md5-identical to the one
    cryoem_core.pipeline.run_convert itself writes for the same synthetic montage and recipe (MRC + direct, idoc +
    TIFF + yflip), with a negative control (the other placement differs) so the comparison can fail; layout.json
    boxes every MANIFEST tile inside the image, row 0 / col 0 at the top left; tiles md5-identical with and without
    stitching; --stitch-only reproduces the conversion's own montage byte for byte.
The delivered-session control (a lab 3b JPG re-rendered) needs the lab's raw data and is run by hand (see the
round 7 report); the synthetic run_convert comparison above is its always-available stand-in.
"""
import dataclasses
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import test_convert_raw as T  # noqa: E402

CR = T.CR
REPO = T.REPO


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def tree_md5(d):
    return {p.relative_to(d).as_posix(): md5(p) for p in sorted(Path(d).rglob("*")) if p.is_file()}


STUB = dict(T.STUB_FILES)
STUB["cryoem_core/stitching.py"] = T.STUB_FILES["cryoem_core/stitching.py"] + """
        def placed_pieces(pieces):
            return [(i, 0, 0, p) for i, p in enumerate(pieces) if 'PieceCoordinates' in p]
        """
# a header line as SerialEM writes it: scope, serial, site (must never reach acquisition.json)
HEAD = "[T = SerialEM: Glacios 1234567 at Some Site   06-Nov-26  07:48:56 ]\n\n"
MDOC = ("PixelSpacing = 7.362\nImageSize = 16 12\nVoltage = 200\n\n" + HEAD +
        "[ZValue = 0]\nPieceCoordinates = 0 0 0\nMagnification = 11000\nDefocus = -4.5\nExposureDose = 3.6\n"
        "ExposureTime = 3\nDateTime = 06-Nov-2026  07:48:56\n")


class Stub(unittest.TestCase):
    def setUp(self):
        T.purge_pipeline()
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_stitch_stub_"))
        self.repo = self.tmp / "processing"
        for rel, text in STUB.items():
            p = self.repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
        (self.repo / "session").mkdir()
        (self.repo / "session" / "grid1-pos1.mrc.mdoc").write_text(MDOC, encoding="utf-8")
        self.raw = self.tmp / "raw"
        self.raw.mkdir()
        for stem in ("grid1-pos1", "grid2-pos3"):
            (self.raw / f"{stem}.mrc").write_bytes(b"\0" * 16)
            (self.raw / f"{stem}.mrc.mdoc").write_text(MDOC, encoding="utf-8")
        self.syspath = list(sys.path)

    def tearDown(self):
        T.purge_pipeline()
        sys.path[:] = self.syspath
        shutil.rmtree(self.tmp, ignore_errors=True)

    def convert(self, out, **kw):
        return CR.convert([self.raw], out, processing_repo_dir=self.repo, log=io.StringIO(), **kw)

    def test_stitch_failure_is_a_warning_and_tiles_unchanged(self):
        plain = self.convert(self.tmp / "a" / "tiles")
        conv = self.tmp / "b"
        res = self.convert(conv / "tiles", montages_out=conv / "montages", acquisition_out=conv / "acquisition.json")
        self.assertEqual(res["n_tiles"], plain["n_tiles"])
        self.assertEqual(tree_md5(conv / "tiles"), tree_md5(self.tmp / "a" / "tiles"))   # tiles untouched
        self.assertEqual(res["stitch"]["state"], "failed")              # the stub core has no pipeline module
        self.assertEqual(json.loads((conv / "montages" / "stitch.json").read_text(encoding="utf-8")), res["stitch"])
        self.assertEqual(sorted(p.name for p in (conv / "montages").iterdir()), ["stitch.json"])
        self.assertTrue(any(w.startswith("montages: stitching failed") for w in res["warnings"]), res["warnings"])
        self.assertFalse((conv / ".stitch_work").exists())
        self.assertIsNone(plain["stitch"])
        self.assertFalse(plain["acquisition"])

    def test_acquisition_json_schema_and_nothing_private(self):
        conv = self.tmp / "c"
        res = self.convert(conv / "tiles", acquisition_out=conv / "acquisition.json")
        self.assertTrue(res["acquisition"], res["warnings"])
        text = (conv / "acquisition.json").read_text(encoding="utf-8")
        doc = json.loads(text)
        self.assertEqual(doc["schema"], "acquisition/1")
        self.assertEqual([m["montage"] for m in doc["montages"]], ["grid1_1", "grid2_3"])
        self.assertEqual(set(doc["montages"][0]), {"montage", "mag", "pixel_A", "defocus_um", "dose", "acquired",
                                                  "exposure_s", "n_pieces"})
        m = doc["montages"][0]
        self.assertEqual((m["mag"], m["pixel_A"], m["defocus_um"], m["dose"], m["exposure_s"], m["n_pieces"]),
                         (11000, 7.362, -4.5, 3.6, 3.0, 1))
        self.assertEqual(m["acquired"], "2026-11-06T07:48:56")
        self.assertEqual(doc["session"]["n_mdoc"], 2)
        for bad in ("1234567", "Some Site", "SerialEM", str(self.raw), "raw", ".mdoc", "\\\\"):
            self.assertNotIn(bad, text)

    def test_montages_out_overlapping_the_tiles_refused(self):
        out = self.tmp / "o" / "tiles"
        for mo in (out, out / "m", self.tmp / "o" / "tiles" / ".."):
            with self.subTest(mo=mo), self.assertRaisesRegex(CR.ConvertError, "must not be the tiles folder"):
                self.convert(out, montages_out=mo)
            self.assertFalse(out.exists())

    def test_stitch_only_refusals(self):
        conv = self.tmp / "conv"
        conv.mkdir()
        with self.assertRaisesRegex(CR.ConvertError, "not a conversion folder"):
            CR.stitch_only(conv, processing_repo_dir=self.repo, log=io.StringIO())
        (conv / "convert.json").write_text(json.dumps({"inputs": [str(self.raw / "grid1-pos1.mrc.mdoc")],
                                                       "recipe": "default"}), encoding="utf-8")
        with self.assertRaisesRegex(CR.ConvertError, "has not finished"):
            CR.stitch_only(conv, processing_repo_dir=self.repo, log=io.StringIO())
        self.convert(conv / "tiles")
        before = tree_md5(conv / "tiles")
        res = CR.stitch_only(conv, processing_repo_dir=self.repo, log=io.StringIO())
        self.assertEqual((res["mode"], res["stitch"]["state"]), ("stitch", "failed"))   # stub: no pipeline module
        self.assertTrue(res["acquisition"])
        self.assertEqual(tree_md5(conv / "tiles"), before)
        shutil.rmtree(self.raw)
        with self.assertRaisesRegex(CR.ConvertError, "raw inputs are gone"):
            CR.stitch_only(conv, processing_repo_dir=self.repo, log=io.StringIO())

    def stitched_conv(self):
        """A finished conversion folder whose montages/ is a done set (the stub core cannot stitch, so it is made
        by hand), with convert.json and a status.json result naming what made the tiles."""
        conv = self.tmp / "conv"
        self.convert(conv / "tiles")
        (conv / "convert.json").write_text(json.dumps({"inputs": [str(self.raw / "grid1-pos1.mrc.mdoc")],
                                                       "recipe": "default"}), encoding="utf-8")
        good = conv / "montages"
        good.mkdir()
        (good / "grid1_1_stitched.jpg").write_bytes(b"good jpg")
        (good / "layout.json").write_text('{"schema": "montage_layout/1", "montages": []}', encoding="utf-8")
        (good / "stitch.json").write_text('{"state": "done", "error": null, "n": 1}', encoding="utf-8")
        return conv

    def test_a_failed_pass_never_destroys_a_done_set(self):
        conv = self.stitched_conv()
        before = tree_md5(conv / "montages")
        res = CR.stitch_only(conv, processing_repo_dir=self.repo, log=io.StringIO())   # stub: every montage fails
        self.assertEqual((res["stitch"]["state"], res["stitch"].get("kept_previous")), ("failed", True))
        self.assertEqual(tree_md5(conv / "montages"), before)
        self.assertTrue(any("previous montages are kept" in w for w in res["warnings"]), res["warnings"])
        self.assertEqual(sorted(p.name for p in conv.iterdir() if p.name.startswith(".")), [])
        # a swap cut short (montages/ renamed away, the new set never moved in): put back, then kept
        os.replace(conv / "montages", conv / ".montages_old")
        res = CR.stitch_only(conv, processing_repo_dir=self.repo, log=io.StringIO())
        self.assertTrue(res["stitch"].get("kept_previous"))
        self.assertEqual(tree_md5(conv / "montages"), before)
        self.assertFalse((conv / ".montages_old").exists())
        # control: a set that is not done is replaced by the failed pass's own record
        (conv / "montages" / "stitch.json").write_text('{"state": "failed"}', encoding="utf-8")
        res = CR.stitch_only(conv, processing_repo_dir=self.repo, log=io.StringIO())
        self.assertFalse(res["stitch"].get("kept_previous"))
        self.assertEqual(sorted(p.name for p in (conv / "montages").iterdir()), ["stitch.json"])

    def test_processing_recorded_and_a_change_warned(self):
        conv = self.stitched_conv()
        res = CR.stitch_only(conv, processing_repo_dir=self.repo, log=io.StringIO())
        self.assertEqual(res["processing"]["recipe"], "20261012")
        self.assertEqual(res["processing"]["recipe_md5"],
                         md5(self.repo / "datasets" / "process_20261012.py"))
        self.assertEqual(res["stitch"]["processing"], res["processing"])
        self.assertFalse(any("processing repo changed" in w for w in res["warnings"]))
        (conv / "status.json").write_text(json.dumps({"result": {"processing": {
            "commit": "0" * 40, "recipe": "20261012", "recipe_md5": "f" * 32}}}), encoding="utf-8")
        res = CR.stitch_only(conv, processing_repo_dir=self.repo, log=io.StringIO())
        w = [x for x in res["warnings"] if "processing repo changed" in x]
        self.assertTrue(w and "recipe_md5" in w[0], res["warnings"])
        plain = self.convert(self.tmp / "p")
        self.assertEqual(plain["processing"]["recipe"], "20261012")

    def test_cli_json(self):
        env = {**os.environ, CR.REPO_ENV: str(self.repo)}
        run = lambda *a: subprocess.run([sys.executable, "-m", "jobs.convert_raw", *map(str, a)], cwd=REPO,  # noqa
                                        env=env, capture_output=True, text=True, timeout=120)
        conv = self.tmp / "cli"
        p = run("--in", self.raw, "--out", conv / "tiles", "--montages-out", conv / "montages",
                "--acquisition-out", conv / "acquisition.json")
        self.assertEqual(p.returncode, 0, p.stderr)
        res = json.loads(p.stdout)
        self.assertTrue(res["ok"])
        self.assertEqual(res["stitch"]["state"], "failed")
        self.assertTrue(res["acquisition"])
        p = run("--stitch-only", self.tmp / "nowhere")
        self.assertEqual((p.returncode, json.loads(p.stdout)["kind"]), (2, "refused"))
        p = run("--out", self.tmp / "x")                                  # neither --in nor --stitch-only
        self.assertEqual(p.returncode, 2)


@unittest.skipUnless(os.environ.get(CR.REPO_ENV), f"{CR.REPO_ENV} not set: real-pipeline stitch tests skipped")
class Real(unittest.TestCase):
    def setUp(self):
        T.purge_pipeline()
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_stitch_real_"))
        self.raw = self.tmp / "raw"
        self.raw.mkdir()
        self.syspath = list(sys.path)

    def tearDown(self):
        T.purge_pipeline()
        sys.path[:] = self.syspath
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_convert_reference(self, recipe, placement=None):
        """The composite cryoem_core.pipeline.run_convert writes for self.raw with this recipe (all outputs in a temp
        dataset folder; no notes, so no git call)."""
        repo = CR.processing_repo()
        _edt, core = CR.import_pipeline(repo)
        _n, _d, cfg = CR.load_recipe(repo, recipe, core, io.StringIO())
        import cryoem_core.pipeline as pl
        ds = self.tmp / f"ds_{recipe}_{placement}"
        (ds / "notes").mkdir(parents=True)
        cfg = dataclasses.replace(cfg, dataset_dir=ds, raw_subdir=None, notes_subdir="notes", use_gpu=False,
                                  placement=placement or cfg.placement)
        for p in self.raw.iterdir():
            shutil.copyfile(p, ds / p.name)
        with mock.patch("sys.stdout", io.StringIO()):
            pl.run_convert(cfg)
        return cfg.montages_denoised_dir if cfg.denoise else cfg.montages_dir

    def check(self, recipe, montage, other_placement, canvas_w=None):
        conv = self.tmp / "conv"
        res = CR.convert([self.raw], conv / "tiles", recipe=recipe, log=io.StringIO(),
                         montages_out=conv / "montages", acquisition_out=conv / "acquisition.json")
        self.assertEqual(res["stitch"]["state"], "done", res["stitch"])
        mine = conv / "montages" / f"{montage}_stitched.jpg"
        ref = self.run_convert_reference(recipe) / f"{montage}_stitched.jpg"
        self.assertEqual(md5(mine), md5(ref), "stitch pass != run_convert's composite")
        flipped = self.run_convert_reference(recipe, other_placement) / f"{montage}_stitched.jpg"
        self.assertNotEqual(md5(mine), md5(flipped), "negative control: the comparison cannot tell placements apart")
        # layout: every MANIFEST tile boxed inside the image, top-left tile at 0,0
        lay = json.loads((conv / "montages" / "layout.json").read_text(encoding="utf-8"))
        self.assertEqual(lay["schema"], "montage_layout/1")
        (entry,) = lay["montages"]
        self.assertEqual((entry["name"], entry["file"], entry["grid"]), (montage, mine.name, int(montage[4])))
        from PIL import Image
        with Image.open(mine) as im:
            self.assertEqual((entry["width"], entry["height"]), im.size)
        names = [r["output_name"] for r in T.read_manifest(conv / "tiles")]
        self.assertEqual(sorted(t["output_name"] for t in entry["tiles"]), sorted(names))
        for t in entry["tiles"]:
            self.assertTrue(0 <= t["x"] < t["x"] + t["w"] <= entry["width"], t)
            self.assertTrue(0 <= t["y"] < t["y"] + t["h"] <= entry["height"], t)
        first = next(t for t in entry["tiles"] if (t["row"], t["col"]) == (0, 0))
        self.assertEqual((first["x"], first["y"]), (0, 0))
        if canvas_w is None:                                      # under 2000 px: not downsampled
            self.assertEqual(entry["nm_per_px"], 0.7362)
        else:
            self.assertEqual(entry["width"], 2000)
            self.assertEqual(entry["nm_per_px"], round(0.7362 * canvas_w / 2000, 5))
        self.assertIsNotNone(entry["scalebar_nm"])
        acq = (conv / "acquisition.json").read_text(encoding="utf-8")
        self.assertEqual([m["montage"] for m in json.loads(acq)["montages"]], [montage])
        self.assertNotIn(str(self.raw), acq)
        self.assertNotIn(str(self.raw).replace("\\", "\\\\"), acq)
        # tiles: md5-identical to a conversion without the stitch pass
        plain = self.tmp / "plain"
        CR.convert([self.raw], plain, recipe=recipe, log=io.StringIO())
        self.assertEqual(tree_md5(plain), tree_md5(conv / "tiles"))
        return conv, res

    def test_mrc_montage_matches_run_convert(self):
        T.write_mrc_montage(self.raw, grid=(2, 5))              # 5 x (512 - 48) + 48 = 2368 px wide: downsampled
        self.check("20261012", "grid1_1", "yflip", canvas_w=2368)

    def test_idoc_tiff_montage_matches_run_convert(self):
        if not (Path(os.environ[CR.REPO_ENV]) / "datasets" / "process_20261007.py").is_file():
            self.skipTest("no 20261007 recipe")
        T.write_idoc_montage(self.raw, grid=(3, 2))
        self.check("20261007", "grid3_1", "direct")

    def test_stitch_only_reproduces_the_conversion(self):
        T.write_mrc_montage(self.raw, grid=(2, 2))
        conv, res = self.check("20261012", "grid1_1", "yflip")
        first = tree_md5(conv / "montages")
        acq = (conv / "acquisition.json").read_bytes()
        tiles = tree_md5(conv / "tiles")
        (conv / "convert.json").write_text(json.dumps({"inputs": [str(self.raw / "grid1-pos1.mrc.mdoc")],
                                                       "recipe": "20261012", "gpu": False}), encoding="utf-8")
        shutil.rmtree(conv / "montages")
        (conv / "acquisition.json").unlink()
        log = io.StringIO()
        again = CR.stitch_only(conv, log=log)
        self.assertEqual(again["stitch"]["state"], "done")
        self.assertIn("stitch [1/1] grid1_1:", log.getvalue())               # a progress line per montage
        now = tree_md5(conv / "montages")
        for k in ("grid1_1_stitched.jpg", "layout.json"):
            self.assertEqual(now[k], first[k], k)
        self.assertEqual(json.loads((conv / "acquisition.json").read_bytes()), json.loads(acq))
        self.assertEqual(tree_md5(conv / "tiles"), tiles)
        # a later backfill short of RAM (every montage refused by the guard) leaves the done set as it was
        with mock.patch.object(CR, "available_ram", return_value=1):
            low = CR.stitch_only(conv, log=io.StringIO())
        self.assertEqual((low["stitch"]["state"], low["stitch"].get("kept_previous")), ("failed", True))
        self.assertIn("MemoryError", low["stitch"]["error"])
        self.assertEqual(tree_md5(conv / "montages"), now)


if __name__ == "__main__":
    unittest.main()
