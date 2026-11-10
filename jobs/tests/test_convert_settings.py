"""jobs/convert_raw.py round 7 phase 2: describe, preview and --params (BUILD_CONTRACTS.md section 14).

  * Stub (always runs): a stand-in processing repo whose tile export encodes the contrast / denoise settings it was
    given into the tile's grey value, so a setting that reaches the render shows in the bytes.
      - --params: unknown keys, wrong types, out-of-range values and bad JSON are refused (exit 2, nothing
        written); the recipe-specific refusals (bandpass on a median recipe, at or above denoise_max_nm);
      - no --params = byte-identical tiles to --params {} and to --params holding the recipe's own values; a
        changed value changes the bytes (so the comparison can fail);
      - params_effective / params_overrides land in a portal conversion's convert.json (read -> merge -> write,
        other keys kept), and only there; overrides that can't be recorded fail the conversion (no overrides: a
        warning); denoise false still writes a tile when skip_variants names "clean", variant "clean";
      - bandpass_nm needs denoise on after the overrides, and is reported null with denoise off;
      - describe --recipe: the shape and values, read-only (the processing repo is unchanged); describe --conv:
        recorded vs not, a warning when the recipe now differs;
      - preview refusals (an --out that is a file too), stale .preview_work_<pid> cleanup; --stitch-only refuses
        --params.
  * Real (needs CJ_PROCESSING_REPO): a synthetic MRC montage through the real pipeline.
      - golden md5s made by MAIN's code (hard-coded below): no --params reproduces them;
      - preview known answers: before.png = the conversion's own tile (its own overrides included); --params equal
        to the base reuses it; clip_hi_pct changed gives a different after.png, byte-identical to the tile of a
        conversion run with that --params; params are absolute over the conversion's base;
      - denoise false differs from the default and equals the clean variant of a normal run (export_dataset called
        directly); bandpass_nm 4 differs from both.
"""
import argparse
import contextlib
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
import unittest.mock
from pathlib import Path

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
STUB["cryoem_core/__init__.py"] = """
    from . import device, discovery, metadata, stitching
    from dataclasses import dataclass, field
    from pathlib import Path

    @dataclass
    class DatasetConfig:
        dataset_dir: Path
        layout: object
        raw_subdir: str = None
        notes_subdir: str = None
        use_gpu: bool = False
        contrast_level_sampling_A: float = None
        name: str = field(default='')
        skip_variants: tuple = ()
        contrast_lo_pct: float = 1.0
        contrast_hi_pct: float = 99.0
        denoise: bool = False
        denoise_method: str = 'bandpass'
        denoise_min_nm: float = 4.0
        denoise_max_nm: float = None
        denoise_target_nm: float = 2.0
        denoise_match_contrast: bool = True
        dtype: object = None

        @property
        def raw_dir(self):
            return Path(self.dataset_dir) / self.raw_subdir if self.raw_subdir else Path(self.dataset_dir)

        @property
        def sample_key_csv(self):
            return Path(self.dataset_dir) / 'sample_key.csv'

        def output_dirs(self):
            return [('clean', None, None)] + ([('denoised', None, None)] if self.denoise else [])
    """
# The tile export: one tile per montage whose grey value encodes the settings it rendered with.
STUB["export_detector_tiles.py"] = """
    import csv
    from pathlib import Path
    from PIL import Image

    TARGET_A_PX = 14.724
    SMALL_DIR = '3d. Tiles no scalebar - %s A-px PNG'
    DATASET_FIELDS = ['output_name', 'source_file', 'grid', 'sample_label', 'montage', 'output_size', 'nm_per_px',
                      'compute']

    def grey(cfg):
        v = cfg.contrast_lo_pct * 10 + cfg.contrast_hi_pct
        if cfg.denoise:
            v += 3 * cfg.denoise_min_nm
        return int(round(v * 7)) % 256

    def export_dataset(args, cfg):
        kept = [v for v, _m, _p in cfg.output_dirs() if v not in cfg.skip_variants]
        if not kept:
            raise SystemExit('every variant is in skip_variants')
        native = Path(args.out)
        small = native.parent / (SMALL_DIR % ('%.1f' % TARGET_A_PX))
        native.mkdir(parents=True)
        small.mkdir(parents=True)
        rows = []
        for m in cfg.layout.find(cfg.raw_dir):
            name = 'g%d_Grid_%d__%s_r0c0_t00__denoised_nobar_14p72Apx.png' % (m.grid, m.grid, m.basename)
            Image.new('L', (8, 6), grey(cfg)).save(small / name)
            rows.append({'output_name': name, 'source_file': m.basename + '_r0c0_t00.jpg', 'grid': m.grid,
                         'sample_label': 'Grid %d' % m.grid, 'montage': m.basename, 'output_size': '8x6',
                         'nm_per_px': '%.5f' % (TARGET_A_PX * 0.1), 'compute': 'CPU'})
        with open(small / 'MANIFEST.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=DATASET_FIELDS)
            w.writeheader()
            w.writerows(rows)
        return 0
    """
_RECIPE = """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cryoem_core import DatasetConfig, discovery
    DATASET = DatasetConfig(dataset_dir=Path(__file__).parent.parent / 'session',
                            layout=discovery.FlatGridPositionMrcLayout(), {extra})
    """
STUB["datasets/process_dn.py"] = _RECIPE.format(
    extra="denoise=True, denoise_min_nm=2.0, skip_variants=('clean',), contrast_level_sampling_A=7.362")
STUB["datasets/process_med.py"] = _RECIPE.format(extra="denoise=True, denoise_method='median'")
STUB["datasets/process_band.py"] = _RECIPE.format(extra="denoise=True, denoise_min_nm=2.0, denoise_max_nm=3.0")

BAD_PARAMS = [
    ('{"bogus": 1}', "unknown key"),
    ('{"contrast_hi_pct": 99}', "unknown key"),                 # the DatasetConfig name is not a key
    ('{"clip_lo_pct": -0.1}', "clip_lo_pct must be a number from 0 to 5"),
    ('{"clip_lo_pct": 5.01}', "clip_lo_pct must be"),
    ('{"clip_hi_pct": 94.9}', "clip_hi_pct must be a number from 95 to 100"),
    ('{"clip_hi_pct": 100.5}', "clip_hi_pct must be"),
    ('{"bandpass_nm": 0.4}', "bandpass_nm must be a number from 0.5 to 10"),
    ('{"bandpass_nm": 11}', "bandpass_nm must be"),
    ('{"clip_hi_pct": "99"}', "clip_hi_pct must be"),
    ('{"clip_hi_pct": true}', "clip_hi_pct must be"),
    ('{"clip_hi_pct": NaN}', "clip_hi_pct must be"),
    ('{"clip_hi_pct": Infinity}', "clip_hi_pct must be"),
    ('{"clip_hi_pct": null}', "clip_hi_pct must be"),
    ('{"clip_hi_pct": -Infinity}', "clip_hi_pct must be"),
    ('{"clip_hi_pct": 1' + "0" * 400 + '}', "clip_hi_pct must be"),   # an int too big for a float
    ('{"denoise": 1}', "denoise must be true or false"),
    ('{"denoise": "false"}', "denoise must be true or false"),
    ('[1, 2]', "must be a JSON object"),
    ('99.5', "must be a JSON object"),
    ('{clip_hi_pct: 99}', "not valid JSON"),
]


class Stub(unittest.TestCase):
    def setUp(self):
        T.purge_pipeline()
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_settings_stub_"))
        self.repo = self.tmp / "processing"
        for rel, text in STUB.items():
            p = self.repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
        (self.repo / "session").mkdir()
        (self.repo / "session" / "grid1-pos1.mrc.mdoc").write_text(
            "PixelSpacing = 7.362\nImageSize = 16 12\n\n[ZValue = 0]\nPieceCoordinates = 0 0 0\nBinning = 2\n",
            encoding="utf-8")
        self.syspath = list(sys.path)

    def tearDown(self):
        T.purge_pipeline()
        sys.path[:] = self.syspath
        shutil.rmtree(self.tmp, ignore_errors=True)

    def conv_folder(self, name="20261106-1_abc123", recipe="dn"):
        """A portal-shaped conversion folder: convert.json + input/ (one stub MRC montage)."""
        d = self.tmp / "raw_tiles" / name
        (d / "input").mkdir(parents=True)
        (d / "input" / "grid1-pos1.mrc").write_bytes(b"\0" * 16)
        (d / "input" / "grid1-pos1.mrc.mdoc").write_text(T.MDOC.format(head="PixelSpacing = 7.362\n"),
                                                          encoding="utf-8")
        meta = {"schema": "cj_convert/1", "id": name, "owner": "u1", "session": "20261106-1", "recipe": recipe,
                "gpu": False, "inputs": [str(d / "input" / "grid1-pos1.mrc.mdoc")], "sample_key": None,
                "created": "2026-11-09T21:00:00"}
        (d / "convert.json").write_text(json.dumps(meta), encoding="utf-8")
        return d

    def convert(self, d, params=None, recipe="dn", out=None):
        return CR.convert([d / "input" / "grid1-pos1.mrc.mdoc"], out or d / "tiles", recipe=recipe, params=params,
                          processing_repo_dir=self.repo, log=io.StringIO())

    def cli(self, *args):
        env = {k: v for k, v in os.environ.items() if k != CR.REPO_ENV}
        env[CR.REPO_ENV] = str(self.repo)
        p = subprocess.run([sys.executable, "-m", "jobs.convert_raw", *map(str, args)], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=120)
        return p.returncode, json.loads(p.stdout), p.stderr

    # ------------------------------------------------------------ --params validation
    def test_parse_params_refuses_keys_types_and_ranges(self):
        for raw, msg in BAD_PARAMS:
            with self.subTest(raw=raw), self.assertRaisesRegex(CR.ConvertError, msg):
                CR.parse_params(raw)
        self.assertIsNone(CR.parse_params(None))
        self.assertEqual(CR.parse_params("{}"), {})
        ok = CR.parse_params('{"clip_lo_pct": 0, "clip_hi_pct": 100, "bandpass_nm": 0.5, "denoise": false}')
        self.assertEqual(ok, {"clip_lo_pct": 0.0, "clip_hi_pct": 100.0, "bandpass_nm": 0.5, "denoise": False})
        self.assertEqual(CR.parse_params({"clip_lo_pct": 5, "bandpass_nm": 10}), {"clip_lo_pct": 5.0,
                                                                                 "bandpass_nm": 10.0})

    def test_cli_bad_params_exit_2_and_nothing_written(self):
        d = self.conv_folder()
        for raw, msg in BAD_PARAMS[:3] + [('{"denoise": 1}', "denoise")]:
            with self.subTest(raw=raw):
                code, res, _err = self.cli("--in", d / "input", "--out", d / "tiles", "--recipe", "dn",
                                           "--params", raw)
                self.assertEqual((code, res["ok"], res["kind"]), (2, False, "refused"))
                self.assertIn(msg, res["error"])
                self.assertFalse((d / "tiles").exists())
                code, res, _err = self.cli("preview", "--conv", d, "--tile", "x.png", "--params", raw,
                                           "--out", self.tmp / "pv", "--json")
                self.assertEqual((code, res["kind"]), (2, "refused"))
                self.assertIn(msg, res["error"])
                self.assertFalse((self.tmp / "pv").exists())
        self.assertNotIn("params_effective", json.loads((d / "convert.json").read_text(encoding="utf-8")))

    def test_recipe_specific_refusals_before_any_work(self):
        d = self.conv_folder()
        for recipe, params, msg in (("med", {"bandpass_nm": 3}, "needs a bandpass recipe"),
                                    ("band", {"bandpass_nm": 3}, "below the recipe's denoise_max_nm")):
            with self.subTest(recipe=recipe), self.assertRaisesRegex(CR.ConvertError, msg):
                self.convert(d, params, recipe=recipe)
            self.assertFalse((d / "tiles").exists())
        self.convert(d, {"bandpass_nm": 2.5}, recipe="band")               # below max: fine

    def test_bandpass_needs_denoise_on_after_the_overrides(self):
        d = self.conv_folder()
        # 20261012 (stub) has denoise off; dn has it on
        for recipe, params in (("20261012", {"bandpass_nm": 3}), ("dn", {"denoise": False, "bandpass_nm": 3})):
            with self.subTest(recipe=recipe), self.assertRaisesRegex(CR.ConvertError, "needs denoise on"):
                self.convert(d, params, recipe=recipe)
            self.assertFalse((d / "tiles").exists())
        res = self.convert(d, {"denoise": True, "bandpass_nm": 3}, recipe="20261012", out=self.tmp / "on")
        self.assertEqual(res["params_effective"]["bandpass_nm"], 3.0)
        res = self.convert(d, None, recipe="20261012", out=self.tmp / "off")
        self.assertEqual(res["params_effective"], {"clip_lo_pct": 1.0, "clip_hi_pct": 99.0, "bandpass_nm": None,
                                                   "denoise": False})
        res = self.convert(d, {"denoise": False}, recipe="dn", out=self.tmp / "dnoff")
        self.assertIsNone(res["params_effective"]["bandpass_nm"])
        desc = CR.describe_recipe("20261012", processing_repo_dir=self.repo, log=io.StringIO())
        self.assertIsNone(desc["denoise"]["bandpass_nm"])
        self.assertIsNone(desc["params"]["bandpass_nm"])
        self.assertEqual(desc["variant"], "clean")

    # ------------------------------------------------------------ no --params = the recipe, byte for byte
    def test_no_params_is_byte_identical(self):
        d = self.conv_folder()
        base = self.convert(d, None, out=self.tmp / "a")
        same = {"clip_lo_pct": 1, "clip_hi_pct": 99, "bandpass_nm": 2, "denoise": True}   # dn's own values
        self.assertEqual(base["params_effective"], {k: (v if k == "denoise" else float(v)) for k, v in same.items()})
        self.assertEqual(base["params_overrides"], {})
        self.convert(d, "{}", out=self.tmp / "b")
        self.convert(d, same, out=self.tmp / "c")
        self.convert(d, {"clip_hi_pct": 99.5}, out=self.tmp / "x")
        a = tree_md5(self.tmp / "a")
        self.assertEqual(a, tree_md5(self.tmp / "b"))
        self.assertEqual(a, tree_md5(self.tmp / "c"))
        changed = tree_md5(self.tmp / "x")
        self.assertEqual(sorted(changed), sorted(a))
        self.assertNotEqual(changed, a)                      # the control: a setting that reaches the render shows

    def test_denoise_off_writes_the_clean_variant(self):
        d = self.conv_folder()
        res = self.convert(d, {"denoise": False})           # dn skips "clean": without the fix, no variant is left
        self.assertEqual(res["n_tiles"], 1)
        self.assertFalse(res["params_effective"]["denoise"])
        self.assertEqual([t["variant"] for t in res["tiles"]], ["clean"])
        self.assertEqual(json.loads((d / "convert.json").read_text(encoding="utf-8"))["variant"], "clean")
        self.assertIn("denoised_nobar", res["tiles"][0]["file"])     # names stay the processing repo's
        res = self.convert(d, None, out=self.tmp / "plain")
        self.assertEqual([t["variant"] for t in res["tiles"]], ["denoised"])
        cfg = CR.apply_params(CR.load_recipe(self.repo, "dn", CR.import_pipeline(self.repo)[1], io.StringIO())[2],
                              {"denoise": False})
        self.assertEqual(cfg.skip_variants, ())
        self.assertFalse(cfg.denoise)

    # ------------------------------------------------------------ convert.json
    def test_params_effective_recorded_in_convert_json(self):
        d = self.conv_folder()
        before = json.loads((d / "convert.json").read_text(encoding="utf-8"))
        res = self.convert(d, '{"clip_hi_pct": 99.5, "bandpass_nm": 3}')
        meta = json.loads((d / "convert.json").read_text(encoding="utf-8"))
        want = {"clip_lo_pct": 1.0, "clip_hi_pct": 99.5, "bandpass_nm": 3.0, "denoise": True}
        self.assertEqual(meta["params_effective"], want)
        self.assertEqual(meta["params_overrides"], {"clip_hi_pct": 99.5, "bandpass_nm": 3.0})
        self.assertEqual(res["params_effective"], want)
        self.assertEqual(meta["variant"], "denoised")
        self.assertEqual({k: meta[k] for k in before}, before)        # merged: every other key kept as it was
        # no --params: the recipe's values are recorded, overrides {}
        d2 = self.conv_folder("20261106-1_def456")
        self.convert(d2)
        meta2 = json.loads((d2 / "convert.json").read_text(encoding="utf-8"))
        self.assertEqual(meta2["params_overrides"], {})
        self.assertEqual(meta2["params_effective"]["clip_hi_pct"], 99.0)
        # not a portal folder (--out not <dir>/tiles beside convert.json): nothing is written outside --out
        loose = self.tmp / "loose"
        res = self.convert(d2, {"clip_hi_pct": 99.5}, out=loose / "out")
        self.assertEqual(res["params_overrides"], {"clip_hi_pct": 99.5})
        self.assertEqual(sorted(p.name for p in loose.iterdir()), ["out"])

    def test_overrides_that_cant_be_recorded_fail_the_conversion(self):
        held = unittest.mock.patch.object(CR, "_write_json", side_effect=PermissionError("held open"))
        d = self.conv_folder()
        with held, self.assertRaisesRegex(RuntimeError, "could not be recorded.*PermissionError"):
            self.convert(d, {"clip_hi_pct": 99.5})
        self.assertFalse((d / "tiles").exists())                   # failed: no tiles left behind
        with held:                                                 # no overrides: a warning only
            res = self.convert(d, None)
        self.assertTrue(any("params_effective not recorded" in w for w in res["warnings"]), res["warnings"])
        d2 = self.conv_folder("20261106-1_bad000")
        (d2 / "convert.json").write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "could not be recorded.*unreadable"):
            self.convert(d2, {"denoise": False})
        self.assertFalse((d2 / "tiles").exists())
        res = self.convert(d2, None)
        self.assertTrue(any("params_effective not recorded" in w for w in res["warnings"]), res["warnings"])
        code, res, _ = self.cli("--in", d / "input", "--out", self.tmp / "x" / "tiles", "--recipe", "dn")
        self.assertEqual(code, 0)                                  # not a portal folder: nothing to record

    def test_clear_stale_preview_work(self):
        out = self.tmp / "pv"
        dead, live, mine = (out / f"{CR.PREVIEW_WORK}_{n}" for n in (999999999, os.getppid(), os.getpid()))
        for x in (dead, live, mine):
            x.mkdir(parents=True)
        (out / "before.png").write_bytes(b"x")
        CR.clear_stale_preview_work(out)
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(["before.png", live.name, mine.name]))

    def test_cli_params_round_trip(self):
        d = self.conv_folder()
        code, res, err = self.cli("--in", d / "input", "--out", d / "tiles", "--recipe", "dn",
                                  "--params", '{"clip_lo_pct": 0.5}')
        self.assertEqual(code, 0, err)
        self.assertEqual(res["params_effective"]["clip_lo_pct"], 0.5)
        self.assertEqual(json.loads((d / "convert.json").read_text(encoding="utf-8"))["params_overrides"],
                         {"clip_lo_pct": 0.5})
        code, res, _ = self.cli("--stitch-only", d, "--params", "{}")
        self.assertEqual((code, res["kind"]), (2, "refused"))
        self.assertIn("--params is refused", res["error"])

    # ------------------------------------------------------------ describe
    def test_describe_recipe_shape_and_read_only(self):
        repo_before = tree_md5(self.repo)
        code, res, err = self.cli("describe", "--recipe", "dn", "--json")
        self.assertEqual(code, 0, err)
        for k in ("recipe", "denoise", "contrast", "pixel", "source", "params", "params_allowed"):
            self.assertIn(k, res)
        self.assertEqual((res["ok"], res["mode"], res["recipe"]), (True, "describe", "dn"))
        self.assertEqual(res["contrast"]["clip_lo_pct"], 1.0)
        self.assertEqual(res["contrast"]["clip_hi_pct"], 99.0)
        self.assertEqual(res["contrast"]["sampling"], 7.362)
        self.assertEqual({k: res["denoise"][k] for k in ("on", "method", "bandpass_nm", "match_contrast")},
                         {"on": True, "method": "bandpass", "bandpass_nm": 2.0, "match_contrast": True})
        self.assertEqual(res["variant"], "denoised")
        self.assertEqual(res["pixel"]["target_A_px"], 14.724)
        self.assertEqual(res["pixel"]["session_A_px"], [7.362])
        self.assertEqual(res["pixel"]["binning"], [2])
        self.assertEqual(res["pixel"]["downsample"], {"7.362": 2.0})
        self.assertEqual(res["params_allowed"], {"clip_lo_pct": [0.0, 5.0], "clip_hi_pct": [95.0, 100.0],
                                                 "bandpass_nm": [0.5, 10.0], "denoise": "bool"})
        self.assertEqual(res["source"]["recipe_md5"], md5(self.repo / "datasets" / "process_dn.py"))
        self.assertEqual(res["source"]["layout"], "FlatGridPositionMrcLayout")
        self.assertEqual(tree_md5(self.repo), repo_before)          # read-only: no bytecode, nothing written
        code, res, _ = self.cli("describe", "--recipe", "default")
        self.assertEqual((code, res["recipe"], res["source"]["defaulted"]), (0, "20261012", True))
        for args, msg in ((("--recipe", "oldstyle"), "not a session config"), (("--recipe", "nope"), "unknown"),
                          ((), "one of the arguments"), (("--conv", self.tmp / "nowhere"), "not a conversion")):
            with self.subTest(args=args):
                code, res, _ = self.cli("describe", *args, "--json")
                self.assertEqual((code, res["kind"]), (2, "refused"))
                self.assertIn(msg, res["error"])
        self.assertFalse((self.repo / "datasets" / "RAN_oldstyle").exists())

    def test_describe_conv_recorded_or_not(self):
        d = self.conv_folder()
        res = CR.describe_conv(d, processing_repo_dir=self.repo, log=io.StringIO())   # before any conversion
        self.assertFalse(res["recorded"])
        self.assertEqual(res["params_effective"]["clip_hi_pct"], 99.0)
        self.convert(d, {"clip_hi_pct": 99.5})
        res = CR.describe_conv(d, processing_repo_dir=self.repo, log=io.StringIO())
        self.assertTrue(res["recorded"])
        self.assertEqual(res["params_effective"]["clip_hi_pct"], 99.5)
        self.assertEqual(res["contrast"]["clip_hi_pct"], 99.5)       # the blocks carry the recorded overrides
        self.assertEqual(res["params_overrides"], {"clip_hi_pct": 99.5})
        self.assertEqual(res["warnings"], [])
        p = self.repo / "datasets" / "process_dn.py"                 # the recipe changes after the conversion
        p.write_text(p.read_text(encoding="utf-8").replace("denoise_min_nm=2.0", "denoise_min_nm=3.0"),
                     encoding="utf-8")
        T.purge_pipeline()
        res = CR.describe_conv(d, processing_repo_dir=self.repo, log=io.StringIO())
        self.assertEqual(res["params_effective"]["bandpass_nm"], 2.0)   # what the tiles were made with
        self.assertTrue(any("other values" in w for w in res["warnings"]), res["warnings"])

    # ------------------------------------------------------------ preview refusals
    def test_preview_refusals(self):
        d = self.conv_folder()
        self.convert(d)
        tile = T.read_manifest(d / "tiles")[0]["output_name"]
        cases = [((d, "nope.png", self.tmp / "pv"), "no tile 'nope.png'"),
                 ((d, "../x.png", self.tmp / "pv"), "not a path"),
                 ((d, tile, d / "preview"), "outside the conversion folder"),
                 ((d, tile, d / "tiles"), "outside the conversion folder"),
                 ((self.tmp / "nowhere", tile, self.tmp / "pv"), "not a conversion folder"),
                 ((d, tile, self.tmp / "afile"), "is a file")]
        (self.tmp / "afile").write_bytes(b"x")
        for (conv, t, out), msg in cases:
            with self.subTest(msg=msg), self.assertRaisesRegex(CR.ConvertError, msg):
                CR.preview(conv, t, out, processing_repo_dir=self.repo, log=io.StringIO())
        self.assertFalse((self.tmp / "pv").exists())
        self.assertFalse((d / "preview").exists())
        shutil.rmtree(d / "input")
        with self.assertRaisesRegex(CR.ConvertError, "raw inputs are gone"):
            CR.preview(d, tile, self.tmp / "pv", processing_repo_dir=self.repo, log=io.StringIO())


# ---------------------------------------------------------------- the real processing repo
@unittest.skipUnless(os.environ.get(CR.REPO_ENV), f"{CR.REPO_ENV} not set: real-pipeline tests skipped")
class Real(unittest.TestCase):
    RECIPE = "20261012"

    def setUp(self):
        T.purge_pipeline()
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_settings_real_"))
        self.syspath = list(sys.path)

    def tearDown(self):
        T.purge_pipeline()
        sys.path[:] = self.syspath
        shutil.rmtree(self.tmp, ignore_errors=True)

    def conv(self, name, params=None):
        d = self.tmp / "raw_tiles" / name
        (d / "input").mkdir(parents=True)
        T.write_mrc_montage(d / "input")
        meta = {"schema": "cj_convert/1", "id": name, "recipe": self.RECIPE, "gpu": False,
                "inputs": [str(d / "input" / "grid1-pos1.mrc.mdoc")], "sample_key": None}
        (d / "convert.json").write_text(json.dumps(meta), encoding="utf-8")
        res = CR.convert([d / "input" / "grid1-pos1.mrc.mdoc"], d / "tiles", recipe=self.RECIPE, params=params,
                         log=io.StringIO())
        return d, res

    def clean_variant_md5(self, d) -> dict:
        """source_file -> md5 of the CLEAN variant of a normal run of the recipe (denoise on, both variants
        rendered, the denoised one skipped): export_dataset called directly, outside convert_raw."""
        repo = Path(os.environ[CR.REPO_ENV])
        edt, core = CR.import_pipeline(repo)
        _n, _df, base = CR.load_recipe(repo, self.RECIPE, core, io.StringIO())
        self.assertTrue(base.denoise, "the recipe must denoise for this control")
        ms, _w = CR.find_montages([d / "input" / "grid1-pos1.mrc.mdoc"], core, base.layout)
        for m in ms:
            CR.check_montage(m, core)                     # sets the placement, as the conversion does
        work = self.tmp / "clean_ref"
        cfg = dataclasses.replace(CR.run_cfg(base, work, ms, False, "clean ref"), skip_variants=("denoised",))
        args = argparse.Namespace(out=str(work / "native"), grids=None, limit=0, gpu=False)
        with contextlib.redirect_stdout(io.StringIO()):
            edt.export_dataset(args, cfg)
        small = work / (edt.SMALL_DIR % ("%.1f" % float(edt.TARGET_A_PX)))
        return {r["source_file"]: md5(small / r["output_name"]) for r in T.read_manifest(small)}

    def test_no_params_matches_main_golden(self):
        """Tiles of the synthetic montage as MAIN's code (00a1921, before --params existed) wrote them."""
        repo = Path(os.environ[CR.REPO_ENV])
        if CR.processing_info(repo, self.RECIPE)["recipe_md5"] != GOLDEN_RECIPE_MD5:
            self.skipTest(f"recipe {self.RECIPE} changed since the goldens were made: regenerate them")
        d, res = self.conv("c_golden")
        self.assertEqual(res["compute"], "CPU")
        got = {r["output_name"]: md5(d / "tiles" / r["output_name"]) for r in T.read_manifest(d / "tiles")}
        self.assertEqual(got, GOLDEN_TILES)
        self.assertEqual(md5(d / "tiles" / "MANIFEST.csv"), GOLDEN_MANIFEST)

    def test_preview_known_answer_and_params_identity(self):
        d, res = self.conv("c_plain")
        own = res["params_effective"]                         # the recipe's own values
        self.assertEqual(own, CR.describe_recipe(self.RECIPE, log=io.StringIO())["params"])
        rows = T.read_manifest(d / "tiles")
        tile = rows[len(rows) // 2]["output_name"]
        want = md5(d / "tiles" / tile)

        pv = CR.preview(d, tile, self.tmp / "pv_same", params=json.dumps(own), log=io.StringIO())
        self.assertEqual(pv["params_effective"], own)
        self.assertTrue(pv["after_reused"])                   # the same settings as the base: one render
        self.assertEqual(md5(pv["before"]), want, "preview of the conversion's own settings != its tile")
        self.assertEqual(md5(pv["after"]), want)
        self.assertEqual(sorted(p.name for p in (self.tmp / "pv_same").iterdir()), ["after.png", "before.png"])

        changed = {"clip_hi_pct": 99.5 if own["clip_hi_pct"] != 99.5 else 99.0}
        pv2 = CR.preview(d, tile, self.tmp / "pv_hi", params=changed, log=io.StringIO())
        self.assertFalse(pv2["after_reused"])
        self.assertEqual(md5(pv2["before"]), want)
        self.assertNotEqual(md5(pv2["after"]), want)          # the control: the comparison can fail

        d2, res2 = self.conv("c_hi", params=changed)          # the real conversion with the same override
        hi = md5(d2 / "tiles" / tile)
        self.assertEqual(md5(pv2["after"]), hi)
        self.assertEqual(json.loads((d2 / "convert.json").read_text(encoding="utf-8"))["params_overrides"],
                         {"clip_hi_pct": float(changed["clip_hi_pct"])})

        # S2: the base is the conversion's own settings. No params: before = its own (overridden) tile.
        pv3 = CR.preview(d2, tile, self.tmp / "pv_base", log=io.StringIO())
        self.assertEqual(md5(pv3["before"]), hi)
        self.assertEqual(md5(pv3["after"]), hi)
        self.assertEqual(pv3["params_conversion"]["clip_hi_pct"], float(changed["clip_hi_pct"]))
        self.assertEqual(pv3["params_recipe"], own)
        self.assertEqual(pv3["params_conversion_overrides"], {"clip_hi_pct": float(changed["clip_hi_pct"])})
        # params are absolute over that base: the recipe's value again renders the plain conversion's tile
        pv4 = CR.preview(d2, tile, self.tmp / "pv_back", params={"clip_hi_pct": own["clip_hi_pct"]},
                         log=io.StringIO())
        self.assertFalse(pv4["after_reused"])
        self.assertEqual((md5(pv4["before"]), md5(pv4["after"])), (hi, want))

        d3, _ = self.conv("c_own", params=own)                # recipe-valued --params = no --params, byte for byte
        self.assertEqual(tree_md5(d3 / "tiles"), tree_md5(d / "tiles"))

    def test_denoise_off_and_bandpass_change_the_pixels(self):
        d, res = self.conv("c_plain")
        plain = {r["source_file"]: md5(d / "tiles" / r["output_name"]) for r in T.read_manifest(d / "tiles")}
        self.assertEqual({t["variant"] for t in res["tiles"]}, {"denoised"})

        dn, res_dn = self.conv("c_nodn", params={"denoise": False})
        nodn = {r["source_file"]: md5(dn / "tiles" / r["output_name"]) for r in T.read_manifest(dn / "tiles")}
        self.assertEqual({t["variant"] for t in res_dn["tiles"]}, {"clean"})
        self.assertIsNone(res_dn["params_effective"]["bandpass_nm"])
        self.assertTrue(all(nodn[k] != plain[k] for k in plain), "denoise off left a tile unchanged")
        self.assertEqual(nodn, self.clean_variant_md5(d))     # = the clean variant of a normal run

        bp, res_bp = self.conv("c_bp4", params={"bandpass_nm": 4})
        bp4 = {r["source_file"]: md5(bp / "tiles" / r["output_name"]) for r in T.read_manifest(bp / "tiles")}
        self.assertEqual(res_bp["params_effective"]["bandpass_nm"], 4.0)
        self.assertTrue(all(bp4[k] != plain[k] and bp4[k] != nodn[k] for k in plain))


# Made by MAIN's convert_raw (00a1921) on test_convert_raw.write_mrc_montage()'s defaults, recipe 20261012, CPU,
# processing repo 199dfe65 (2026-11-09). They pin pre-change output: compare against these, never regenerate them
# from this branch's code. Other numpy / OpenCV / mrcfile builds may differ; the test skips when the recipe changed.
GOLDEN_RECIPE_MD5 = "21915787bb97fafe5a5a27dc92fcde4b"
GOLDEN_TILES = {
    "g1_Grid_1__grid1_1_r0c0_t00__denoised_nobar_14p72Apx.png": "ca27b3da0153c8214265ab3ef402ebf3",
    "g1_Grid_1__grid1_1_r0c1_t02__denoised_nobar_14p72Apx.png": "ce45dc7c110a918922eeeec8dab219da",
    "g1_Grid_1__grid1_1_r1c0_t01__denoised_nobar_14p72Apx.png": "e412c752708f8643b2c3ec631cd7a532",
    "g1_Grid_1__grid1_1_r1c1_t03__denoised_nobar_14p72Apx.png": "c6f4ed76f65d9f352a6c2b22c2cbf0d4",
}
GOLDEN_MANIFEST = "2586ca91b5cb0704e9d2ff29217afded"


if __name__ == "__main__":
    unittest.main()
