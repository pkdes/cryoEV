"""jobs/convert_raw.py round 7 phase 2: describe, preview and --params (BUILD_CONTRACTS.md section 14).

  * Stub (always runs): a stand-in processing repo whose tile export encodes the contrast / denoise settings it was
    given into the tile's grey value, so a setting that reaches the render shows in the bytes.
      - --params: unknown keys, wrong types, out-of-range values and bad JSON are refused (exit 2, nothing
        written); the recipe-specific refusals (bandpass on a median recipe, at or above denoise_max_nm);
      - no --params = byte-identical tiles to --params {} and to --params holding the recipe's own values; a
        changed value changes the bytes (so the comparison can fail);
      - params_effective / params_overrides land in a portal conversion's convert.json (read -> merge -> write,
        other keys kept), and only there; denoise false still writes a tile when skip_variants names "clean";
      - describe --recipe: the shape and values, read-only (the processing repo is unchanged); describe --conv:
        recorded vs not, a warning when the recipe now differs;
      - preview refusals; --stitch-only refuses --params.
  * Real (needs CJ_PROCESSING_REPO): a synthetic MRC montage through the real pipeline. Known answer: preview with
    --params equal to the recipe's own values gives an after.png (and before.png) byte-identical to the tile the
    conversion wrote; clip_hi_pct 99.5 gives a different after.png, byte-identical to the tile of a conversion run
    with that --params; no --params = recipe-valued --params, byte for byte.
"""
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
                 ((self.tmp / "nowhere", tile, self.tmp / "pv"), "not a conversion folder")]
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

    def test_preview_known_answer_and_params_identity(self):
        d, res = self.conv("c_plain")
        own = res["params_effective"]                         # the recipe's own values
        self.assertEqual(own, CR.describe_recipe(self.RECIPE, log=io.StringIO())["params"])
        rows = T.read_manifest(d / "tiles")
        tile = rows[len(rows) // 2]["output_name"]
        want = md5(d / "tiles" / tile)

        pv = CR.preview(d, tile, self.tmp / "pv_same", params=json.dumps(own), log=io.StringIO())
        self.assertEqual(pv["params_effective"], own)
        self.assertEqual(md5(pv["after"]), want, "preview with the recipe's own values != the conversion's tile")
        self.assertEqual(md5(pv["before"]), want)
        self.assertEqual(sorted(p.name for p in (self.tmp / "pv_same").iterdir()), ["after.png", "before.png"])

        changed = {"clip_hi_pct": 99.5 if own["clip_hi_pct"] != 99.5 else 99.0}
        pv2 = CR.preview(d, tile, self.tmp / "pv_hi", params=changed, log=io.StringIO())
        self.assertEqual(md5(pv2["before"]), want)
        self.assertNotEqual(md5(pv2["after"]), want)          # the control: the comparison can fail

        d2, res2 = self.conv("c_hi", params=changed)          # the real conversion with the same override
        self.assertEqual(md5(pv2["after"]), md5(d2 / "tiles" / tile))
        self.assertEqual(json.loads((d2 / "convert.json").read_text(encoding="utf-8"))["params_overrides"],
                         {"clip_hi_pct": float(changed["clip_hi_pct"])})

        d3, _ = self.conv("c_own", params=own)                # recipe-valued --params = no --params, byte for byte
        self.assertEqual(tree_md5(d3 / "tiles"), tree_md5(d / "tiles"))


if __name__ == "__main__":
    unittest.main()
