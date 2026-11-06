"""
CJ_RAW_ROOT: the whole raw pipeline (raw-upload .portal staging + raw_tiles/<id>/...) on another folder or drive
(portal/modules/predict.py raw_base / raw_root / raw_roots / tmp_dirs; admin.disks).

  * Unset: every W1 test in test_predict_convert.py passes unchanged (they run with it unset).
  * Set to a second temp folder: raw upload -> convert -> tiles -> prediction set works, tiles are md5-identical to
    the unset case, and nothing raw is written under the predict root.
  * No os.replace ever crosses the two roots (a mocked EXDEV proves it), also on a `subst` drive letter (Windows).
  * Old conversions under <predict root>/raw_tiles stay listed; retention and the stale-scratch sweep cover both roots.
  * The 2x-free check and the upload holds read the raw root's drive; Admin > Health shows both drives.

Temp folders only.   python -m unittest portal.tests.test_predict_rawroot -v
"""
import errno
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_predict_convert as TC  # noqa: E402
from portal import app as APP  # noqa: E402
from portal import predict_sources as PSRC  # noqa: E402
from portal.modules import admin as ADM  # noqa: E402
from portal.modules import predict as P  # noqa: E402

TestClient, World, md5 = TC.TestClient, TC.World, TC.md5
Usage = namedtuple("Usage", "total used free")


def under(p, root) -> bool:
    try:
        return os.path.commonpath([os.path.normcase(os.path.abspath(p)), os.path.normcase(os.path.abspath(root))]) \
            == os.path.normcase(os.path.abspath(root))
    except ValueError:                                  # another drive
        return False


def tile_md5s(d: Path) -> dict:
    return {p.name: md5(p) for p in sorted(d.glob("*.png"))}


class RawBase(TC.Base):
    """TC.Base with CJ_RAW_ROOT set to a second temp folder."""
    raw_dir_name = None

    @classmethod
    def make_raw_dir(cls) -> Path:
        cls.raw_tmp = Path(tempfile.mkdtemp(prefix="cj_rawroot_"))
        return cls.raw_tmp / "rawdrive"

    @classmethod
    def setUpClass(cls):
        if TestClient is None:
            raise unittest.SkipTest("starlette TestClient (httpx) not installed")
        raw = cls.make_raw_dir()
        raw.mkdir(parents=True, exist_ok=True)
        cls.w = World(raw_root=raw)
        cls.tc = TestClient(APP.create_app(cls.w.cfg), base_url=TC.ORIGIN, follow_redirects=False)
        cls.tc.__enter__()
        cls.tok = {}
        for who, key in cls.w.keys.items():
            r = cls.tc.get(f"/login?k={key}")
            assert r.status_code == 303, r.status_code
            cls.tok[who] = r.headers["set-cookie"].split(";", 1)[0].split("=", 1)[1]
        cls.tc.cookies.clear()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(cls.raw_tmp, ignore_errors=True)

    def tearDown(self):
        super().tearDown()
        tmp = self.w.raw_base / P.PORTAL_TMP
        for p in (list(tmp.iterdir()) if tmp.exists() else []):
            if P.UPLOAD_RE.match(p.name):
                P.drop_upload(p, p.name)

    # the conversions of both roots
    def conv_folder(self, cid) -> Path:
        for root in (self.w.rawtiles, self.w.root / P.RAW_TILES):
            if (root / cid / "convert.json").is_file():
                return root / cid
        return self.w.rawtiles / cid

    def status(self, cid) -> dict:
        return TC.json.loads((self.conv_folder(cid) / "status.json").read_text(encoding="utf-8"))

    def src_key(self, cid) -> str:
        return PSRC.source_key(self.conv_folder(cid) / P.CONVERT_TILES)

    def converted_old(self, who="advanced") -> str:
        """A conversion made with CJ_RAW_ROOT unset: it lands under <predict root>/raw_tiles (an 'old' one)."""
        saved = os.environ.pop("CJ_RAW_ROOT")
        try:
            return self.converted(who)
        finally:
            os.environ["CJ_RAW_ROOT"] = saved

    def under_predict_root(self, name_part) -> list:
        return [p for p in self.w.root.rglob("*") if name_part in p.name]

    @staticmethod
    def eexdev_replace(roots):
        """os.replace that fails with EXDEV when src and dst are not under the same one of `roots`."""
        real = os.replace

        def fake(src, dst, *a, **k):
            home = lambda p: next((i for i, r in enumerate(roots) if under(p, r)), None)  # noqa: E731
            if home(src) != home(dst) or home(src) is None:
                raise OSError(errno.EXDEV, "Invalid cross-device link", str(src))
            fake.calls.append((str(src), str(dst)))
            return real(src, dst, *a, **k)
        fake.calls = []
        return fake


class RawRoot(RawBase):
    def test_flow_set_made_bytes_identical_and_nothing_raw_under_predict_root(self):
        self.assertEqual(P.raw_base(None), self.w.raw_base)
        self.assertNotEqual(P.raw_base(None), self.w.root)
        cid = self.converted("advanced")
        d = self.w.rawtiles / cid
        self.assertTrue((d / "input" / "grid1-pos1.mrc").is_file())
        self.assertTrue((d / "convert.json").is_file() and (d / "status.json").is_file())
        tiles = tile_md5s(d / "tiles")
        self.assertGreaterEqual(len(tiles), 3)
        # nothing raw under the predict root: no raw_tiles, no raw file, no raw staging left
        self.assertFalse((self.w.root / P.RAW_TILES).exists())
        for part in ("grid1-pos1", "convert.json", "convert.log", ".convert_work"):
            self.assertEqual(self.under_predict_root(part), [], part)
        self.assertEqual(sorted(p.name for p in (self.w.raw_base / P.PORTAL_TMP).iterdir()), [])
        # a prediction set from the conversion: copies (md5-equal), never links across the roots
        path = f"/predict/new/session/{self.src_key(cid)}/"
        form = self.page(path, "advanced")
        sig = TC.re.search(r'name="sig" value="([0-9a-f]+)"', form).group(1)
        r = self.form(path, {"sig": sig, "samples": ["g0", "g1"], "nm_g0": "1.4724", "nm_g1": "1.4724",
                             "how": "all", "model": TC.MODEL, "confirm": "yes"}, "advanced")
        self.assertEqual(r.status_code, 303, r.text[:600])
        ps = self.w.root / r.headers["location"].strip("/").split("/")[-1]
        imgs = [p for p in (ps / "images").rglob("*") if p.is_file()]
        self.assertEqual(len(imgs), len(tiles))
        self.assertEqual(sorted(md5(p) for p in imgs), sorted(tiles.values()))
        self.assertTrue(all(os.stat(p).st_nlink == 1 for p in imgs))
        TC.wait_for(lambda: TC.json.loads((ps / "status.json").read_text(encoding="utf-8"))["state"]
                    in ("done", "failed"), 60, what="the fake predictor")
        # the same conversion with CJ_RAW_ROOT unset: byte-identical tiles
        old = self.converted_old()
        self.assertTrue(self.conv_folder(old).is_relative_to(self.w.root / P.RAW_TILES))
        self.assertEqual(tile_md5s(self.conv_folder(old) / "tiles"), tiles)

    def test_old_conversions_under_the_predict_root_stay_listed_and_usable(self):
        new, old = self.converted("advanced"), self.converted_old("advanced")
        other = self.converted_old("adv2")
        self.assertIn(new, self.page("/predict/convert/%s/" % new, "advanced"))
        self.assertIn(old, self.page("/predict/convert/%s/" % old, "advanced"))
        self.assertEqual(self.get("/predict/convert/%s/" % other, "advanced").status_code, 404)    # owners as now
        self.assertEqual(self.get("/predict/convert/%s/" % other, "admin").status_code, 200)
        self.assertEqual(self.get("/predict/convert/%s/" % old, "adv2").status_code, 404)
        cfg = self.w.cfg
        mine = {t[0].name for t in P.list_conversions(cfg, {"id": "adv"}, lambda u, o: o == u["id"])}
        self.assertTrue({new, old} <= mine and other not in mine)
        keys = {s["conversion"]: s["dir"] for s in P.converted_sources(cfg, {"id": "adv"}, lambda u, o: o == u["id"])}
        self.assertEqual(keys[old], self.w.root / P.RAW_TILES / old / "tiles")
        self.assertEqual(keys[new], self.w.rawtiles / new / "tiles")
        page = self.page("/predict/new/", "advanced")
        self.assertIn(f'href="/predict/new/session/{self.src_key(old)}/"', page)
        self.assertIn(f'href="/predict/new/session/{self.src_key(new)}/"', page)
        self.assertEqual(self.get(f"/predict/new/session/{self.src_key(old)}/", "advanced").status_code, 200)

    def test_retention_and_stale_scratch_cover_both_roots(self):
        new, old = self.converted("advanced"), self.converted_old("advanced")
        for c in (new, old):
            self.assertTrue((self.conv_folder(c) / "input").is_dir())
            st = self.status(c)
            st["finished"] = "2026-01-01T00:00:00"
            P.PS.write_json_atomic(self.conv_folder(c) / "status.json", st)
        os.environ["CJ_RAW_KEEP_DAYS"] = "1"
        try:
            P.kick(None)
        finally:
            os.environ.pop("CJ_RAW_KEEP_DAYS")
        for c in (new, old):
            self.assertFalse((self.conv_folder(c) / "input").exists(), c)
            self.assertTrue((self.conv_folder(c) / "tiles" / "MANIFEST.csv").is_file())
        # stale scratch: both .portal folders
        tmp_dirs = P.tmp_dirs(None)
        self.assertEqual(len(tmp_dirs), 2)
        for t in tmp_dirs:
            (t / "up_stale").mkdir(parents=True, exist_ok=True)
            (t / "up_stale" / "f").write_bytes(b"x")
            os.utime(t / "up_stale", (time.time() - 7200,) * 2)
        self.assertEqual(P.sweep_scratch(None, max_age_s=3600), 2)
        for t in tmp_dirs:
            self.assertFalse((t / "up_stale").exists())

    def test_raw_upload_stages_under_the_raw_root_tiles_upload_under_the_predict_root(self):
        st, _ = self.upload("advanced", "raw", {"session": "20261101"}, self.raw_files(), create=False)
        self.assertTrue((self.w.raw_base / P.PORTAL_TMP / st["upload"]).is_dir())
        self.assertFalse((self.w.root / P.PORTAL_TMP / st["upload"]).exists())
        st2, _ = self.upload("advanced", "tiles", {"sample_label": "S", "nm_per_px": "1.4724", "confirm": "yes", "model": TC.MODEL}, [("a.png", TC.tile_bytes(1))], create=False)
        self.assertTrue((self.w.root / P.PORTAL_TMP / st2["upload"]).is_dir())
        self.assertFalse((self.w.raw_base / P.PORTAL_TMP / st2["upload"]).exists())
        # a later piece / create finds either one; the stale sweep drops either one with its hold
        self.assertIsNotNone(P.upload_dir(None, st["upload"]))
        self.assertIsNotNone(P.upload_dir(None, st2["upload"]))
        self.assertEqual(P.sweep_scratch(None, max_age_s=-1), 2)
        self.assertEqual(P._RESERVED, {})

    def test_disk_checks_read_the_raw_roots_drive(self):
        seen = []

        def free(p):
            seen.append(Path(p))
            return 10 ** 13
        with mock.patch.object(P, "disk_free", side_effect=free):
            self.assertEqual(self.pick_session().status_code, 303)
            self.converted("advanced")
        self.assertTrue(seen)
        self.assertTrue(all(under(p, self.w.raw_base) for p in seen), [str(p) for p in seen])
        # and it refuses by the raw drive's free space
        with mock.patch.object(P, "disk_free", return_value=P.FREE_MARGIN_BYTES + 10):
            r = self.pick_session()
        self.assertEqual(r.status_code, 507)

    def test_no_replace_crosses_the_roots_exdev(self):
        fake = self.eexdev_replace([self.w.raw_base, self.w.root])
        with mock.patch.object(os, "replace", side_effect=fake):
            cid = self.converted("advanced")
        self.assertEqual(self.status(cid)["state"], "done")
        moves = [c for c in fake.calls if "input" in c[1] and os.path.basename(c[0]) == "files"]
        self.assertEqual(len(moves), 1, fake.calls)
        self.assertTrue(under(moves[0][0], self.w.raw_base) and under(moves[0][1], self.w.raw_base))
        # control: the mock does refuse a cross-root replace
        with mock.patch.object(os, "replace", side_effect=fake):
            self.w.root.mkdir(parents=True, exist_ok=True)
            a = self.w.root / "x_control"
            a.write_bytes(b"1")
            with self.assertRaises(OSError) as cm:
                os.replace(a, self.w.raw_base / "x_control")
            self.assertEqual(cm.exception.errno, errno.EXDEV)
        a.unlink()

    def test_health_shows_both_drives_and_warns_under_20_gb(self):
        cfg = self.w.cfg
        rows = {r["label"]: r for r in ADM.disks(cfg)["rows"]}
        self.assertEqual(rows["Predict sets"]["path"], str(self.w.root))
        self.assertEqual(rows["Raw images"]["path"], str(self.w.raw_base))
        self.assertTrue(all(isinstance(r["free_gb"], float) for r in rows.values()))
        quiet = {"ok": False, "rows": [], "problem": "x"}
        with mock.patch.object(ADM, "services_status", return_value=quiet), \
                mock.patch.object(ADM, "backup_commit", return_value={"configured": None, "problem": "x"}), \
                mock.patch.object(ADM, "backup_task", return_value={"found": False, "problem": "x"}):
            fake = lambda p: Usage(10 ** 12, 0, (5 if str(self.w.raw_base) in str(p) else 500) * 2 ** 30)  # noqa: E731
            with mock.patch("shutil.disk_usage", side_effect=fake):
                h = ADM._check_health(cfg)
        html_ = ADM.render_health(h)
        self.assertIn("Disk space", html_)
        self.assertIn("Predict sets", html_)
        self.assertIn("Raw images", html_)
        self.assertIn("500.0 GB free", html_)
        self.assertIn("5.0 GB free", html_)
        self.assertEqual({r["label"]: r["warn"] for r in h["disks"]["rows"]},
                         {"Predict sets": False, "Raw images": True})
        self.assertIn("under 20 GB", html_)


def free_letter():
    for L in "ZYXWVUTSRQPONM":
        if not os.path.exists(f"{L}:\\"):
            return L
    return None


@unittest.skipUnless(os.name == "nt" and free_letter(), "needs Windows and a free drive letter for subst")
class RawRootOnSubstDrive(RawBase):
    """CJ_RAW_ROOT on a second drive letter (subst of a temp folder; removed afterwards)."""

    @classmethod
    def make_raw_dir(cls) -> Path:
        cls.raw_tmp = Path(tempfile.mkdtemp(prefix="cj_rawroot_"))
        cls.letter = free_letter()
        subprocess.run(["subst", f"{cls.letter}:", str(cls.raw_tmp)], check=True)
        return Path(f"{cls.letter}:\\rawdrive")

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            subprocess.run(["subst", f"{cls.letter}:", "/d"], check=False)
            shutil.rmtree(cls.raw_tmp, ignore_errors=True)

    def test_flow_on_the_other_drive_letter(self):
        self.assertNotEqual(os.path.splitdrive(str(self.w.raw_base))[0].upper(),
                            os.path.splitdrive(str(self.w.root))[0].upper())
        fake = self.eexdev_replace([self.w.raw_base, self.w.root])
        with mock.patch.object(os, "replace", side_effect=fake):
            cid = self.converted("advanced")
        d = self.w.rawtiles / cid
        self.assertEqual(self.status(cid)["state"], "done")
        self.assertTrue((d / "tiles" / "MANIFEST.csv").is_file())
        self.assertEqual((self.raw_tmp / "rawdrive" / P.RAW_TILES / cid).is_dir(), True)   # really on that folder
        self.assertFalse((self.w.root / P.RAW_TILES).exists())
        key = self.src_key(cid)
        path = f"/predict/new/session/{key}/"
        sig = TC.re.search(r'name="sig" value="([0-9a-f]+)"', self.page(path, "advanced")).group(1)
        r = self.form(path, {"sig": sig, "samples": ["g0", "g1"], "nm_g0": "1.4724", "nm_g1": "1.4724",
                             "how": "all", "model": TC.MODEL, "confirm": "yes"}, "advanced")
        self.assertEqual(r.status_code, 303, r.text[:600])
        ps = self.w.root / r.headers["location"].strip("/").split("/")[-1]
        imgs = [p for p in (ps / "images").rglob("*") if p.is_file()]
        self.assertEqual(sorted(md5(p) for p in imgs), sorted(tile_md5s(d / "tiles").values()))
        self.assertTrue(all(os.stat(p).st_nlink == 1 for p in imgs))
        TC.wait_for(lambda: TC.json.loads((ps / "status.json").read_text(encoding="utf-8"))["state"]
                    in ("done", "failed"), 60, what="the fake predictor")


if __name__ == "__main__":
    unittest.main()
