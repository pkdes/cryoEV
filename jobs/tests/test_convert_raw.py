"""jobs/convert_raw.py: raw montages -> 3d tiles + MANIFEST.csv through the lab's processing repo.

Three layers:
  * StubRepo (always runs): a stand-in processing repo in a temp folder, so the plumbing is tested without the lab's
    code -- tiles and MANIFEST.csv end up in <out> and nothing else does, refusals leave nothing behind, nm/px comes
    only from the sidecar, the CLI speaks JSON on stdout with exit 2 on a refusal.
  * RealRepo (needs CJ_PROCESSING_REPO): synthetic MRC+mdoc and idoc+TIFF montages through the real pipeline.
    Known answer: tile size = source size / (14.724 / PixelSpacing), nm/px = 1.4724, and an 8-bit greyscale PNG.
  * Control (needs CJ_PROCESSING_REPO and CJ_CONVERT_CONTROL_DIR = a session folder that already has its 3d tiles):
    one montage (CJ_CONVERT_CONTROL_MONTAGE, default grid1-pos1) converted again must be md5-identical to the
    delivered 3d tiles, with the same nm/px, rendered on the compute the delivered MANIFEST names. Read-only there.
"""
import csv
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

import numpy as np
from PIL import Image

TESTS = Path(__file__).resolve().parent
JOBS = TESTS.parent
REPO = JOBS.parent
sys.path.insert(0, str(JOBS))
import convert_raw as CR  # noqa: E402

PIPELINE_MODULES = ("cryoem_core", "export_detector_tiles")


def purge_pipeline():
    """Forget any processing repo imported by an earlier test (stub and real must not share a process cache)."""
    for k in list(sys.modules):
        if k.split(".")[0] in PIPELINE_MODULES:
            del sys.modules[k]


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def read_manifest(d):
    with open(Path(d) / "MANIFEST.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_mrc_montage(folder, stem="grid1-pos1", ps=7.362, tile=(384, 512), grid=(2, 2), overlap=48,
                      pixel_spacing_line=True, seed=0):
    """A synthetic MRC stack + .mrc.mdoc in SerialEM's form: [ZValue = n] blocks with PieceCoordinates."""
    import mrcfile
    h, w = tile
    rng = np.random.default_rng(seed)
    n = grid[0] * grid[1]
    data = (3000 + 200 * rng.standard_normal((n, h, w))).astype(np.int16)
    yy, xx = np.mgrid[:h, :w]
    for z in range(n):                        # a few dark discs, so contrast has something to find
        for cy, cx in ((h // 3, w // 3), (2 * h // 3, 2 * w // 3)):
            data[z][(yy - cy) ** 2 + (xx - cx) ** 2 < 30 ** 2] -= 600
    with mrcfile.new(str(folder / f"{stem}.mrc"), overwrite=True) as m:
        m.set_data(data)
    step = (w - overlap, h - overlap)
    lines = []
    if pixel_spacing_line:
        lines.append(f"PixelSpacing = {ps}")
    lines += [f"PieceSpacing = {step[0]} {step[1]}", f"ImageFile = {stem}", f"ImageSize = {w} {h}",
              "Montage = 1", "DataMode = 1", ""]
    z = 0
    for c in range(grid[1]):
        for r in range(grid[0]):
            lines += [f"[ZValue = {z}]", f"PieceCoordinates = {c * step[0]} {r * step[1]} 0",
                      "Magnification = 11000", ""]
            z += 1
    (folder / f"{stem}.mrc.mdoc").write_text("\n".join(lines), encoding="utf-8")
    return folder / f"{stem}.mrc"


def write_idoc_montage(folder, stem="grid3-1", ps=7.362, tile=(384, 512), grid=(2, 2), overlap=48, seed=1):
    """A synthetic SerialEM idoc + one 16-bit TIFF per piece, named <stem><piece:04d>.tif."""
    h, w = tile
    rng = np.random.default_rng(seed)
    step = (w - overlap, h - overlap)
    lines = [f"PixelSpacing = {ps}", f"PieceSpacing = {step[0]} {step[1]}", f"ImageSize = {w} {h}", ""]
    i = 0
    for c in range(grid[1]):
        for r in range(grid[0]):
            name = f"{stem}{i:04d}.tif"
            arr = (3000 + 200 * rng.standard_normal((h, w))).clip(0, 65535).astype(np.uint16)
            Image.fromarray(arr).save(folder / name)
            lines += [f"[Image = {name}]", f"PieceCoordinates = {c * step[0]} {r * step[1]} 0",
                      "Magnification = 11000", ""]
            i += 1
    (folder / f"{stem}.idoc").write_text("\n".join(lines), encoding="utf-8")
    return folder / f"{stem}.idoc"


# ---------------------------------------------------------------- a stand-in processing repo
STUB_FILES = {
    "cryoem_core/__init__.py": """
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

            @property
            def raw_dir(self):
                return Path(self.dataset_dir) / self.raw_subdir if self.raw_subdir else Path(self.dataset_dir)

            @property
            def sample_key_csv(self):
                return Path(self.dataset_dir) / 'sample_key.csv'
        """,
    "cryoem_core/device.py": """
        def describe_device(prefer_gpu=True):
            return 'CPU (NumPy)'
        """,
    "cryoem_core/metadata.py": """
        def parse_metadata(path):
            g, pieces, cur = {}, [], None
            with open(path) as fh:
                lines = fh.read().splitlines()
            for line in lines:
                line = line.strip()
                if line.startswith('[') and '=' in line:
                    kind = line[1:].split('=')[0].strip()
                    cur = {'_section': kind}
                    if kind == 'Image':
                        cur['_file'] = line.split('=')[1].strip().rstrip(']').strip()
                    pieces.append(cur)
                elif '=' in line:
                    k, v = (s.strip() for s in line.split('=', 1))
                    (cur if cur is not None else g).setdefault(k, v)
            return g, pieces
        """,
    "cryoem_core/discovery.py": """
        import re
        from dataclasses import dataclass
        from pathlib import Path

        @dataclass
        class MontageRef:
            grid: int
            region: int
            idoc_path: Path
            basename: str
            suffix: str = ''
            placement: str = None

        class FlatGridPositionMrcLayout:
            def __init__(self, aliases=None):
                self.aliases = aliases or {}

            def find(self, d):
                out = []
                for p in sorted(Path(d).glob('*.mrc.mdoc')):
                    stem = p.name[:-len('.mrc.mdoc')]
                    m = re.match(r'^grid(\\d+)-pos(\\d+)$', stem)
                    if m:
                        out.append(MontageRef(int(m[1]), int(m[2]), p, 'grid%s_%s' % (m[1], m[2])))
                    elif stem in self.aliases:
                        region, suffix = self.aliases[stem]
                        g = int(re.match(r'^grid(\\d+)', stem)[1])
                        out.append(MontageRef(g, region, p, 'grid%d_%d_%s' % (g, region, suffix), suffix))
                return out

        class FlatGridPositionLayout:
            def find(self, d):
                return []

        class FlatGridRegionLayout(FlatGridPositionLayout):
            pass

        class FlatGridSquareHoleLayout(FlatGridPositionLayout):
            pass
        """,
    "cryoem_core/stitching.py": """
        def mrc_path_for(mont, global_params):
            return mont.idoc_path.with_suffix('')
        """,
    "export_detector_tiles.py": """
        import csv
        from pathlib import Path
        from PIL import Image
        from cryoem_core.metadata import parse_metadata

        TARGET_A_PX = 14.724
        SMALL_DIR = '3d. Tiles no scalebar - %s A-px PNG'
        DATASET_FIELDS = ['output_name', 'source_file', 'grid', 'sample_label', 'montage', 'output_size',
                          'nm_per_px']

        def export_dataset(args, cfg):
            native = Path(args.out)
            small = native.parent / (SMALL_DIR % ('%.1f' % TARGET_A_PX))
            native.mkdir(parents=True)
            small.mkdir(parents=True)
            rows = []
            print('  Warning: missing tile stub0001.tif, skipping')
            for m in cfg.layout.find(cfg.raw_dir):
                if 'boom' in m.basename:
                    raise SystemExit('%s: no tiles loaded' % m.basename)
                if 'crash' in m.basename:
                    raise RuntimeError('stub crash')
                if float(parse_metadata(m.idoc_path)[0]['PixelSpacing']) > TARGET_A_PX:
                    raise SystemExit('a coarse montage reached the render')
                name = 'g%d_Grid_%d__%s_r0c0_t00__denoised_nobar_14p72Apx.png' % (m.grid, m.grid, m.basename)
                Image.new('L', (8, 6), 128).save(small / name)
                Image.new('L', (16, 12), 128).save(native / name)
                rows.append({'output_name': name, 'source_file': m.basename + '_r0c0_t00.jpg', 'grid': m.grid,
                             'sample_label': 'Grid %d' % m.grid, 'montage': m.basename, 'output_size': '8x6',
                             'nm_per_px': '%.5f' % (TARGET_A_PX * 0.1)})
            for d in (native, small):
                with open(d / 'MANIFEST.csv', 'w', newline='', encoding='utf-8') as fh:
                    w = csv.DictWriter(fh, fieldnames=DATASET_FIELDS)
                    w.writeheader()
                    w.writerows(rows)
            return 0
        """,
    # The session recipe: its own sampling is read from <stub repo>/session (one sidecar at 7.362 A/px).
    "datasets/process_20261012.py": """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from cryoem_core import DatasetConfig, discovery
        ALIASES = {'grid2-45krerun-1': (104, '45krerun_1')}
        DATASET = DatasetConfig(dataset_dir=Path(__file__).parent.parent / 'session',
                                layout=discovery.FlatGridPositionMrcLayout(aliases=ALIASES))
        if __name__ == '__main__':
            print('never')
        """,
    # An old full-script processor: must be refused without running (it would leave a marker file).
    "datasets/process_oldstyle.py": """
        import torch
        from pathlib import Path
        Path(__file__).with_name('RAN_oldstyle').write_text('x')
        print('[GPU] Using device')
        """,
    # A combined set, not a session: ManifestLayout.
    "datasets/process_combo.py": """
        from pathlib import Path
        from cryoem_core import DatasetConfig, ManifestLayout
        RAN = Path(__file__).with_name('RAN_combo').write_text('x')
        DATASET = DatasetConfig(dataset_dir='x', layout=ManifestLayout('m.csv'))
        """,
}

MDOC = "{head}ImageSize = 16 12\n\n[ZValue = 0]\nPieceCoordinates = 0 0 0\n"


class StubRepo(unittest.TestCase):
    def setUp(self):
        purge_pipeline()
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_convert_stub_"))
        self.repo = self.tmp / "processing"
        for rel, text in STUB_FILES.items():
            p = self.repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
        (self.repo / "session").mkdir()
        (self.repo / "session" / "grid1-pos1.mrc.mdoc").write_text(MDOC.format(head="PixelSpacing = 7.362\n"))
        self.raw = self.tmp / "raw"
        self.raw.mkdir()
        self.syspath = list(sys.path)

    def tearDown(self):
        purge_pipeline()
        sys.path[:] = self.syspath
        shutil.rmtree(self.tmp, ignore_errors=True)

    def stub_mdoc(self, stem, ps="7.362"):
        (self.raw / f"{stem}.mrc").write_bytes(b"\0" * 16)
        head = f"PixelSpacing = {ps}\n" if ps is not None else ""
        (self.raw / f"{stem}.mrc.mdoc").write_text(MDOC.format(head=head), encoding="utf-8")
        return self.raw / f"{stem}.mrc"

    def convert(self, inputs, out, **kw):
        return CR.convert(inputs, out, processing_repo_dir=self.repo, log=io.StringIO(), **kw)

    def cli(self, *args, env_extra=None, repo=True):
        env = {k: v for k, v in os.environ.items() if k != CR.REPO_ENV}
        if repo:
            env[CR.REPO_ENV] = str(self.repo)
        env.update(env_extra or {})
        p = subprocess.run([sys.executable, "-m", "jobs.convert_raw", *map(str, args)], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=120)
        return p.returncode, json.loads(p.stdout), p.stderr

    def test_output_is_self_contained(self):
        self.stub_mdoc("grid1-pos1")
        self.stub_mdoc("grid2-pos3")
        (self.raw / "grid1-atlas.mrc.mdoc").write_text("PixelSpacing = 900\n", encoding="utf-8")
        out = self.tmp / "out"
        res = self.convert([self.raw], out)
        self.assertEqual(res["montages"], ["grid1_1", "grid2_3"])
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         sorted([t["file"] for t in res["tiles"]] + ["MANIFEST.csv"]))
        self.assertEqual(res["nm_per_px"], [1.4724])
        self.assertEqual(res["a_px"], {"grid1_1": 7.362, "grid2_3": 7.362})
        self.assertTrue(res["recipe_defaulted"])
        self.assertEqual((res["compute"], res["device"]), ("CPU", "CPU (NumPy)"))
        self.assertEqual([r["output_name"] for r in read_manifest(out)], [t["file"] for t in res["tiles"]])
        self.assertTrue(any("grid1-atlas" in w for w in res["warnings"]), res["warnings"])
        self.assertIn("Warning: missing tile stub0001.tif, skipping", res["warnings"])   # the pipeline's own line
        self.assertFalse(any("not a sampling" in w for w in res["warnings"]), res["warnings"])
        self.assertEqual(res["entry_point"], "export_detector_tiles.export_dataset")
        self.assertEqual(res["recipe"], "20261012")

    def test_recipe_layout_names_aliased_montages(self):
        self.stub_mdoc("grid2-45krerun-1")
        res = self.convert([self.raw], self.tmp / "out")
        self.assertEqual(res["montages"], ["grid2_104_45krerun_1"])

    def test_file_input_with_unknown_name_is_named_from_its_stem(self):
        mrc = self.stub_mdoc("Sample A_area-7")
        res = self.convert([mrc], self.tmp / "out")
        self.assertEqual(res["montages"], ["Sample_A_area_7"])
        self.assertTrue(any("no layout knows" in w for w in res["warnings"]))

    def test_pixel_spacing_missing_or_not_finite_is_refused(self):
        for ps in (None, "nan", "inf", "-1", "0", "abc"):
            with self.subTest(ps=ps):
                shutil.rmtree(self.raw)
                self.raw.mkdir()
                mrc = self.stub_mdoc("grid1-pos1", ps=ps)
                out = self.tmp / "out"
                with self.assertRaisesRegex(CR.ConvertError, "PixelSpacing"):
                    self.convert([mrc], out)
                self.assertFalse(out.exists())

    def test_coarse_montage_dropped_before_rendering(self):
        good, coarse = self.stub_mdoc("grid1-pos1"), self.stub_mdoc("grid1-pos2", ps="30")
        res = self.convert([good, coarse], self.tmp / "out", recipe="20261012")
        self.assertEqual(res["montages"], ["grid1_1"])
        self.assertTrue(any("grid1_2" in w and "coarser" in w for w in res["warnings"]), res["warnings"])
        with self.assertRaisesRegex(CR.ConvertError, "every montage is coarser"):
            self.convert([coarse], self.tmp / "out2")

    def test_default_recipe_sampling_range(self):
        far = self.stub_mdoc("grid1-pos1", ps="1.0")          # < 0.5 x 7.362
        with self.assertRaisesRegex(CR.ConvertError, "outside the default recipe's range") as cm:
            self.convert([far], self.tmp / "out")
        self.assertIsInstance(cm.exception.warnings, list)
        res = self.convert([far], self.tmp / "out", recipe="20261012")   # named: allowed, with a warning
        self.assertFalse(res["recipe_defaulted"])
        self.assertTrue(any("not a sampling of recipe 20261012" in w for w in res["warnings"]), res["warnings"])
        near = self.stub_mdoc("grid1-pos2", ps="5.0")          # in range but not the session's: warning only
        res = self.convert([near], self.tmp / "out2")
        self.assertTrue(any("5.000 A/px is not a sampling" in w for w in res["warnings"]), res["warnings"])

    def test_old_and_combined_recipes_refused_without_running(self):
        mrc = self.stub_mdoc("grid1-pos1")
        for recipe, msg in (("oldstyle", "not a session config"), ("combo", "not a session layout")):
            with self.subTest(recipe=recipe):
                log = io.StringIO()
                with self.assertRaisesRegex(CR.ConvertError, msg):
                    CR.convert([mrc], self.tmp / "out", recipe=recipe, processing_repo_dir=self.repo, log=log)
                self.assertFalse((self.repo / "datasets" / f"RAN_{recipe}").exists())

    def test_refusals(self):
        mrc = self.stub_mdoc("grid1-pos1")
        full = self.tmp / "full"
        full.mkdir()
        (full / "keep.txt").write_text("x")
        cases = [
            (dict(inputs=[mrc], out=full), "not empty"),
            (dict(inputs=[self.raw / "nope.mrc"], out=self.tmp / "o1"), "not found"),
            (dict(inputs=[mrc], out=self.tmp / "o2", recipe="19990101"), "unknown recipe"),
            (dict(inputs=[mrc], out=self.tmp / "o3", recipe="../x"), "not a session name"),
            (dict(inputs=[], out=self.tmp / "o4"), "no inputs"),
        ]
        (self.raw / "notes.txt").write_text("x")
        cases.append((dict(inputs=[self.raw / "notes.txt"], out=self.tmp / "o5"), "not a raw montage input"))
        (self.raw / "lonely.mrc").write_bytes(b"\0")
        cases.append((dict(inputs=[self.raw / "lonely.mrc"], out=self.tmp / "o6"), "no lonely.mrc.mdoc"))
        for kw, msg in cases:
            with self.subTest(msg=msg), self.assertRaisesRegex(CR.ConvertError, msg):
                self.convert(kw.pop("inputs"), kw.pop("out"), **kw)
        self.assertEqual([p.name for p in full.iterdir()], ["keep.txt"])
        self.assertFalse(any((self.tmp / f"o{i}").exists() for i in range(1, 7)))

    def test_memory_estimate_refuses(self):
        mrc = self.stub_mdoc("grid1-pos1")                     # 16 x 12 x 1 piece x 17 B x 2 (float64) = 6528 B
        with unittest.mock.patch.object(CR, "available_ram", return_value=2000):
            with self.assertRaisesRegex(CR.ConvertError, "GB free now"):
                self.convert([mrc], self.tmp / "out")
        with unittest.mock.patch.object(CR, "available_ram", return_value=None):
            res = self.convert([mrc], self.tmp / "out")
        self.assertIn("free RAM unreadable; memory not checked", res["warnings"])

    def test_gpu_busy_refused_and_fallback_recorded(self):
        mrc = self.stub_mdoc("grid1-pos1")
        hooks = lambda gpu: {"CRYOEV_JOBS_TESTING": "1", "CRYOEV_JOBS_TEST_HOOKS": json.dumps({"gpu": gpu})}  # noqa
        busy = {"total_mb": 4096, "apps": [[4242, "C:\\Python\\python.exe", 900]]}
        with unittest.mock.patch.dict(os.environ, hooks(busy)):
            with self.assertRaisesRegex(CR.ConvertError, "--gpu: gpu busy"):
                self.convert([mrc], self.tmp / "out", gpu=True)
        self.assertFalse((self.tmp / "out").exists())
        with unittest.mock.patch.dict(os.environ, hooks({"total_mb": 4096, "apps": []})):
            res = self.convert([mrc], self.tmp / "out", gpu=True)   # the stub core has no GPU: falls back
        self.assertEqual(res["compute"], "CPU")
        self.assertTrue(any("ran on the CPU" in w for w in res["warnings"]), res["warnings"])

    def test_sidecar_tile_names_stay_in_the_folder(self):
        """[Image = ...] naming an absolute, UNC or '..' path (in an .idoc or an .mrc.mdoc) is refused before any
        tile is opened; a plain name beside the sidecar passes (control)."""
        secret = self.tmp / "secret.tif"
        Image.fromarray(np.zeros((12, 16), np.uint16)).save(secret)
        head = "PixelSpacing = 7.362\nImageSize = 16 12\n\n"
        bad = {"absolute": str(secret), "unc": r"\\gpu-desktop\share\x.tif", "dotdot": "../secret.tif",
               "dotdot-back": r"..\secret.tif", "drive-relative": "C:secret.tif", "empty": ""}
        for why, ref in bad.items():
            for side_name in ("grid9-1.idoc", "grid9-pos1.mrc.mdoc"):
                with self.subTest(why=why, sidecar=side_name):
                    shutil.rmtree(self.raw)
                    self.raw.mkdir()
                    if side_name.endswith(".mrc.mdoc"):
                        (self.raw / side_name[:-len(".mdoc")]).write_bytes(b"\0" * 16)
                    side = self.raw / side_name
                    side.write_text(head + f"[Image = {ref}]\nPieceCoordinates = 0 0 0\n", encoding="utf-8")
                    out = self.tmp / "out"
                    core = self._core()
                    m = CR.find_montages([side], core)[0][0]
                    # no tile is looked at (a UNC name would reach the network): any file check fails the test
                    with unittest.mock.patch.object(Path, "is_file", side_effect=AssertionError("tile opened")), \
                            unittest.mock.patch.object(Path, "exists", side_effect=AssertionError("tile opened")):
                        with self.assertRaisesRegex(CR.ConvertError, "not a plain file name|outside"):
                            CR.check_montage(m, core)
                    with self.assertRaisesRegex(CR.ConvertError, "not a plain file name|outside"):
                        self.convert([side], out)
                    self.assertFalse(out.exists())
        # control: a plain tile name beside the idoc passes the check and converts
        shutil.rmtree(self.raw)
        self.raw.mkdir()
        Image.fromarray(np.zeros((12, 16), np.uint16)).save(self.raw / "grid9-10000.tif")
        (self.raw / "grid9-1.idoc").write_text(head + "[Image = grid9-10000.tif]\nPieceCoordinates = 0 0 0\n",
                                               encoding="utf-8")
        m = CR.find_montages([self.raw / "grid9-1.idoc"], self._core())[0][0]
        self.assertEqual(CR.check_montage(m, self._core())[0], 7.362)
        self.assertEqual(CR.tile_ref_problem(self.raw / "grid9-1.idoc", "grid9-10000.tif"), None)

    def _core(self):
        _edt, core = CR.import_pipeline(self.repo)
        return core

    def test_missing_repo_is_refused(self):
        with self.assertRaisesRegex(CR.ConvertError, "processing repo"):
            CR.convert([self.raw], self.tmp / "o", processing_repo_dir=self.tmp / "nowhere")
        env = {k: v for k, v in os.environ.items() if k != CR.REPO_ENV}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(CR.ConvertError, CR.REPO_ENV):
                CR.convert([self.raw], self.tmp / "o")

    def test_pipeline_stop_cleans_up(self):
        mrc = self.stub_mdoc("grid1-pos1")
        out = self.tmp / "out"
        out.mkdir()                                     # an existing empty folder stays, empty
        boom = self.stub_mdoc("boom")                  # the stub pipeline stops on this one, as render_ref can
        with self.assertRaisesRegex(CR.ConvertError, "pipeline stopped"):
            self.convert([mrc, boom], out)
        self.assertEqual(list(out.iterdir()), [])

    def test_cli_always_json(self):
        self.stub_mdoc("grid1-pos1")
        code, res, err = self.cli("--in", self.raw, "--out", self.tmp / "cli")
        self.assertEqual(code, 0, err)
        self.assertTrue(res["ok"])
        self.assertEqual(res["n_tiles"], 1)
        code, res, err = self.cli("--in", self.raw, "--out", self.tmp / "c2", repo=False)
        self.assertEqual((code, res["ok"], res["kind"]), (2, False, "refused"))
        self.assertIn(CR.REPO_ENV, err)
        code, res, _ = self.cli("--in", self.raw, "--out", self.tmp / "c3", "--recipe", "oldstyle")
        self.assertEqual((code, res["kind"]), (2, "refused"))
        nan = self.tmp / "nan"
        nan.mkdir()
        (nan / "grid1-pos1.mrc").write_bytes(b"\0")
        (nan / "grid1-pos1.mrc.mdoc").write_text(MDOC.format(head="PixelSpacing = nan\n"))
        (nan / "grid1-atlas.mrc.mdoc").write_text("PixelSpacing = 900\n")
        code, res, _ = self.cli("--in", nan, "--out", self.tmp / "c4")
        self.assertEqual((code, res["kind"]), (2, "refused"))
        self.assertIn("PixelSpacing", res["error"])
        self.assertTrue(any("grid1-atlas" in w for w in res["warnings"]), res)   # warnings kept on a refusal
        crash = self.stub_mdoc("crash")
        code, res, err = self.cli("--in", crash, "--out", self.tmp / "c5")
        self.assertEqual((code, res["ok"], res["kind"]), (1, False, "failed"))
        self.assertIn("RuntimeError: stub crash", res["error"])
        self.assertFalse((self.tmp / "c5").exists())


# ---------------------------------------------------------------- the real processing repo
@unittest.skipUnless(os.environ.get(CR.REPO_ENV), f"{CR.REPO_ENV} not set: real-pipeline tests skipped")
class RealRepo(unittest.TestCase):
    def setUp(self):
        purge_pipeline()
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_convert_real_"))
        self.raw = self.tmp / "raw"
        self.raw.mkdir()
        self.syspath = list(sys.path)

    def tearDown(self):
        purge_pipeline()
        sys.path[:] = self.syspath
        shutil.rmtree(self.tmp, ignore_errors=True)

    def check_tiles(self, out, res, n, src_hw, ps):
        factor = 14.724 / ps
        want = (int(round(src_hw[1] / factor)), int(round(src_hw[0] / factor)))
        self.assertEqual(res["n_tiles"], n)
        self.assertEqual(res["nm_per_px"], [round(ps * factor * 0.1, 5)])
        rows = read_manifest(out)
        self.assertEqual(len(rows), n)
        for r in rows:
            with Image.open(out / r["output_name"]) as im:
                self.assertEqual((im.mode, im.size), ("L", want))
            self.assertEqual(r["md5"], md5(out / r["output_name"]))
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted([r["output_name"] for r in rows]
                                                                       + ["MANIFEST.csv"]))

    def test_synthetic_mrc_montage(self):
        mrc = write_mrc_montage(self.raw)
        out = self.tmp / "out"
        res = CR.convert([mrc], out, log=io.StringIO())
        self.check_tiles(out, res, 4, (384, 512), 7.362)
        self.assertEqual(res["montages"], ["grid1_1"])
        self.assertEqual(res["a_px"], {"grid1_1": 7.362})
        self.assertTrue(all(t["file"].startswith("g1_Grid_1__grid1_1_r") for t in res["tiles"]))

    def test_synthetic_idoc_tiff_montage(self):
        write_idoc_montage(self.raw)
        out = self.tmp / "out"
        res = CR.convert([self.raw / "grid3-10000.tif"], out, log=io.StringIO())
        self.check_tiles(out, res, 4, (384, 512), 7.362)
        self.assertEqual(res["montages"], ["grid3_1"])
        self.assertTrue(any("is for MRC montages" in w for w in res["warnings"]), res["warnings"])

    def test_recipe_aliases_name_the_montage(self):
        if not (Path(os.environ[CR.REPO_ENV]) / "datasets" / "process_20261027.py").is_file():
            self.skipTest("no 20261027 recipe")
        write_mrc_montage(self.raw, stem="grid2-45krerun-1")
        res = CR.convert([self.raw], self.tmp / "out", recipe="20261027", log=io.StringIO())
        self.assertEqual(res["montages"], ["grid2_104_45krerun_1"])

    def test_no_pixel_spacing_refused_before_the_pipeline_defaults_it(self):
        mrc = write_mrc_montage(self.raw, pixel_spacing_line=False)
        out = self.tmp / "out"
        with self.assertRaisesRegex(CR.ConvertError, "PixelSpacing"):
            CR.convert([mrc], out, log=io.StringIO())
        self.assertFalse(out.exists())

    def test_old_full_script_and_combined_recipes_refused(self):
        mrc = write_mrc_montage(self.raw)
        for recipe in ("20260601", "evcore"):
            if not (Path(os.environ[CR.REPO_ENV]) / "datasets" / f"process_{recipe}.py").is_file():
                continue
            with self.subTest(recipe=recipe):
                with self.assertRaisesRegex(CR.ConvertError, "refused without running it"):
                    CR.convert([mrc], self.tmp / "out", recipe=recipe, log=io.StringIO())
                self.assertFalse((self.tmp / "out").exists())

    def test_cli_corrupt_mrc_and_nan_are_json(self):
        bad = write_mrc_montage(self.raw)
        bad.write_bytes(os.urandom(4096))                      # header and data garbage, sidecar intact
        for args in (["--in", bad], ["--in", bad, "--recipe", "no_such_recipe"]):
            p = subprocess.run([sys.executable, "-m", "jobs.convert_raw", *map(str, args), "--out",
                                str(self.tmp / "o")], cwd=REPO, capture_output=True, text=True, timeout=300)
            res = json.loads(p.stdout)
            self.assertFalse(res["ok"])
            self.assertIn(p.returncode, (1, 2))
            self.assertEqual(res["kind"], {1: "failed", 2: "refused"}[p.returncode])
            self.assertFalse((self.tmp / "o").exists())


@unittest.skipUnless(os.environ.get(CR.REPO_ENV) and os.environ.get("CJ_CONVERT_CONTROL_DIR"),
                     "CJ_PROCESSING_REPO / CJ_CONVERT_CONTROL_DIR not set: known-answer control skipped")
class Control(unittest.TestCase):
    """The delivered 3d tiles of one montage, reproduced from raw. Read-only on the session folder."""

    def test_delivered_tiles_reproduced(self):
        purge_pipeline()
        session = Path(os.environ["CJ_CONVERT_CONTROL_DIR"])
        stem = os.environ.get("CJ_CONVERT_CONTROL_MONTAGE", "grid1-pos1")
        delivered = next(session.glob("3. Processed images/3d. Tiles no scalebar*"))
        ref = {r["output_name"]: r for r in read_manifest(delivered)}
        tmp = Path(tempfile.mkdtemp(prefix="cj_convert_ctl_"))
        try:
            out = tmp / "out"
            mine = [r for r in ref.values() if r["montage"] == stem.replace("-pos", "_")]
            self.assertTrue(mine, f"no delivered 3d tiles for {stem}")
            gpu = {r["compute"] for r in mine} == {"GPU"}
            res = CR.convert([session / "2. Raw images" / f"{stem}.mrc"], out, recipe=session.name,
                             sample_key=session / "1. Notes + reports" / "sample_key.csv", gpu=gpu,
                             log=io.StringIO())
            self.assertEqual(res["n_tiles"], len(mine))
            for r in read_manifest(out):
                d = ref[r["output_name"]]
                self.assertEqual(md5(out / r["output_name"]), d["md5"], r["output_name"])
                self.assertEqual((r["nm_per_px"], r["output_size"]), (d["nm_per_px"], d["output_size"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            purge_pipeline()


if __name__ == "__main__":
    unittest.main()
