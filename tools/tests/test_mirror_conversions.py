"""tools/mirror_conversions.py in temp dirs only (never the live laptop folders, ~/.cryoev or a share)."""
import hashlib
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import mirror_conversions as M  # noqa: E402

TILES = {"t1.png": b"tile-one" * 50, "t2.png": b"tile-two" * 70}


def make_conv(src, cid="20261106-1_abc", state="done", tiles=None, extra=None):
    conv = Path(src) / cid
    (conv / "tiles").mkdir(parents=True, exist_ok=True)
    (conv / "input").mkdir(exist_ok=True)
    (conv / "montages").mkdir(exist_ok=True)
    tiles = tiles or TILES
    rows = ["output_name,md5"]
    for n, b in tiles.items():
        (conv / "tiles" / n).write_bytes(b)
        rows.append(f"{n},{hashlib.md5(b).hexdigest()}")
    (conv / "tiles" / "MANIFEST.csv").write_text("\n".join(rows) + "\n")
    (conv / "convert.json").write_text(json.dumps({"id": cid, "recipe": "r1"}))
    (conv / "status.json").write_text(json.dumps({"state": state}))
    (conv / "acquisition.json").write_text("{}")
    (conv / "montages" / "layout.json").write_text("{}")
    (conv / "input" / "raw.mrc").write_bytes(b"RAW" * 1000)
    for rel, b in (extra or {}).items():
        p = conv / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    return conv


class MirrorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = Path(self.tmp.name)
        self.src, self.dest = t / "raw_tiles", t / "share"
        self.src.mkdir()
        self.dest.mkdir()

    def run_m(self, **kw):
        kw.setdefault("min_free_gb", 0)
        return M.run(self.src, self.dest, **kw)

    def test_copies_everything_but_input_and_records(self):
        conv = make_conv(self.src, extra={"gallery/events.jsonl": b"{}\n"})
        (res,) = self.run_m()
        d = self.dest / conv.name
        self.assertTrue(res["md5_ok"] and not res["error"])
        self.assertTrue((d / "tiles" / "t1.png").is_file() and (d / "gallery" / "events.jsonl").is_file())
        self.assertTrue((d / "status.json").is_file() and (d / "convert.json").is_file())
        self.assertFalse((d / "input").exists())
        mj = json.loads((conv / "mirror.json").read_text())
        self.assertEqual(mj["dest_rel"], conv.name)
        self.assertTrue(mj["md5_ok"] and mj["error"] is None)
        self.assertEqual(mj["n_files"], 8)
        self.assertFalse(list(conv.glob("*.tmp")))

    def test_idempotent_second_run_copies_nothing(self):
        make_conv(self.src)
        self.run_m()
        with mock.patch.object(M, "copy_one") as cf:
            (res,) = self.run_m()
        cf.assert_not_called()
        self.assertIn("skipped", res["action"])

    def test_unfinished_skipped(self):
        make_conv(self.src, "a_run", state="running")
        make_conv(self.src, "b_fail", state="failed")
        self.assertEqual(self.run_m(), [])
        self.assertEqual(list(self.dest.iterdir()), [])

    def test_never_deletes_on_dest(self):
        conv = make_conv(self.src)
        self.run_m()
        extra = self.dest / conv.name / "tiles" / "only_on_dest.png"
        extra.write_bytes(b"keep me")
        (conv / "tiles" / "t1.png").unlink()      # source loses a tile: the dest copy must stay
        (conv / "tiles" / "MANIFEST.csv").write_text(
            "output_name,md5\nt2.png,%s\n" % hashlib.md5(TILES["t2.png"]).hexdigest())
        self.run_m()
        self.assertEqual(extra.read_bytes(), b"keep me")
        self.assertTrue((self.dest / conv.name / "tiles" / "t1.png").is_file())

    def test_changed_conversion_goes_to_new_dated_folder(self):
        conv = make_conv(self.src)
        self.run_m()
        old = {p.relative_to(self.dest / conv.name): p.read_bytes()
               for p in (self.dest / conv.name).rglob("*") if p.is_file()}
        make_conv(self.src, tiles={"t1.png": b"retried" * 40, "t2.png": TILES["t2.png"]})
        (conv / "convert.json").write_text(json.dumps({"id": conv.name, "recipe": "r2"}))
        (res,) = self.run_m(now=datetime(2026, 11, 9, 21, 0, 5))
        self.assertTrue(res["md5_ok"])
        self.assertEqual(res["dest_rel"], f"{conv.name}__20261109-210005")
        new = self.dest / res["dest_rel"]
        self.assertEqual((new / "tiles" / "t1.png").read_bytes(), b"retried" * 40)
        now_old = {p.relative_to(self.dest / conv.name): p.read_bytes()
                   for p in (self.dest / conv.name).rglob("*") if p.is_file()}
        self.assertEqual(old, now_old)
        mj = json.loads((conv / "mirror.json").read_text())
        self.assertEqual(mj["dest_rel"], res["dest_rel"])
        self.assertEqual(mj["previous"], [conv.name])

    def test_gallery_jsonl_growth_updates_in_place(self):
        conv = make_conv(self.src, extra={"gallery/events.jsonl": b'{"a":1}\n'})
        self.run_m()
        (conv / "gallery" / "events.jsonl").write_bytes(b'{"a":1}\n{"a":2}\n')
        (res,) = self.run_m()
        self.assertEqual(res["dest_rel"], conv.name)
        self.assertEqual((self.dest / conv.name / "gallery" / "events.jsonl").read_bytes(), b'{"a":1}\n{"a":2}\n')
        self.assertEqual([p.name for p in self.dest.iterdir()], [conv.name])

    def test_md5_mismatch_fails(self):
        conv = make_conv(self.src)
        real = M.shutil.copy2

        def corrupt(src, dst, *a, **k):
            r = real(src, dst, *a, **k)
            if Path(src).name == "t1.png":
                b = bytearray(Path(dst).read_bytes())
                b[3] ^= 0xFF
                Path(dst).write_bytes(bytes(b))
            return r
        with mock.patch.object(M.shutil, "copy2", corrupt):
            (res,) = self.run_m()
        self.assertFalse(res["md5_ok"])
        self.assertIn("md5 mismatch", res["error"])
        mj = json.loads((conv / "mirror.json").read_text())
        self.assertFalse(mj["md5_ok"])
        self.assertFalse(list(self.dest.rglob("t1.png")))      # corrupt part never put in place
        (res2,) = self.run_m()                                  # next clean run completes
        self.assertTrue(res2["md5_ok"])

    def test_cli_exit_codes(self):
        make_conv(self.src)
        args = ["--source", str(self.src), "--dest", str(self.dest), "--min-free-gb", "0"]
        self.assertEqual(M.main(args), 0)
        self.assertEqual(M.main(args), 0)             # nothing to do
        make_conv(self.src, "bad_src", tiles={"t1.png": b"zzz"})
        (self.src / "bad_src" / "tiles" / "t1.png").write_bytes(b"corrupted source")
        self.assertEqual(M.main(args), 1)
        self.assertEqual(M.main(["--source", str(self.src / "nope"), "--dest", str(self.dest)]), 1)

    def test_source_tile_not_matching_manifest(self):
        conv = make_conv(self.src)
        (conv / "tiles" / "t2.png").write_bytes(b"changed")
        (res,) = self.run_m()
        self.assertFalse(res["md5_ok"])
        self.assertIn("MANIFEST", res["error"])
        self.assertEqual(list(self.dest.iterdir()), [])

    def test_min_free_floor(self):
        conv = make_conv(self.src)
        (res,) = self.run_m(min_free_gb=10 ** 9)
        self.assertFalse(res["md5_ok"])
        self.assertIn("floor", res["error"])
        self.assertEqual(list(self.dest.iterdir()), [])
        self.assertFalse(json.loads((conv / "mirror.json").read_text())["md5_ok"])

    def test_dry_run_writes_nothing(self):
        conv = make_conv(self.src)
        (res,) = self.run_m(dry_run=True)
        self.assertEqual(res["copied"], 7)
        self.assertEqual(list(self.dest.iterdir()), [])
        self.assertFalse((conv / "mirror.json").exists())

    def test_in_place_keeps_dest_only_and_jsonl_growth(self):
        conv = make_conv(self.src, extra={"gallery/events.jsonl": b'{"a":1}\n'})
        self.run_m()
        keep = self.dest / conv.name / "gallery" / "dest_only.txt"
        keep.write_bytes(b"mine")
        (conv / "gallery" / "events.jsonl").write_bytes(b'{"a":1}\n{"a":2}\n')
        (res,) = self.run_m()
        self.assertEqual(res["dest_rel"], conv.name)
        self.assertEqual(keep.read_bytes(), b"mine")
        self.assertEqual((self.dest / conv.name / "gallery" / "events.jsonl").read_bytes(), b'{"a":1}\n{"a":2}\n')
        vj = json.loads((self.dest / conv.name / "mirror_verified.json").read_text())
        self.assertIn("tiles/t1.png", vj["files"])

    def test_latest_wins_files_overwrite_in_place(self):
        conv = make_conv(self.src, extra={"session.json": b"{}", "convert.log": b"a"})
        self.run_m()
        (conv / "status.json").write_text(json.dumps({"state": "done", "registered": True}))
        (conv / "session.json").write_text('{"s": 1}')
        (conv / "convert.log").write_text("a longer log")
        (res,) = self.run_m()
        self.assertIn("skipped", res["action"])        # not in the fingerprint: nothing to do
        # a different run that does update (a gallery event) carries the latest-wins files along in place
        (conv / "gallery").mkdir()
        (conv / "gallery" / "events.jsonl").write_bytes(b"{}\n")
        (res,) = self.run_m()
        self.assertEqual(res["dest_rel"], conv.name)
        d = self.dest / conv.name
        self.assertEqual(json.loads((d / "session.json").read_text()), {"s": 1})
        self.assertEqual((d / "convert.log").read_text(), "a longer log")
        self.assertEqual([p.name for p in self.dest.iterdir()], [conv.name])

    def test_crash_mid_copy_then_clean_completion(self):
        conv = make_conv(self.src)
        real = M.shutil.copy2
        n = {"i": 0}

        def crash(src, dst, *a, **k):
            n["i"] += 1
            if n["i"] == 3:
                Path(dst).write_bytes(b"half")
                raise OSError("share dropped")
            return real(src, dst, *a, **k)
        with mock.patch.object(M.shutil, "copy2", crash):
            (res,) = self.run_m()
        self.assertFalse(res["md5_ok"])
        names = [p.name for p in self.dest.iterdir()]
        self.assertEqual(names, [conv.name + ".partial"])              # never a half-built final folder
        self.assertTrue(list((self.dest / names[0]).rglob("*.mirror-part")))
        (res2,) = self.run_m()
        self.assertTrue(res2["md5_ok"])
        self.assertEqual([p.name for p in self.dest.iterdir()], [conv.name])
        self.assertFalse(list(self.dest.rglob("*.mirror-part")))

    def test_offline_share_exit_1_and_mirror_json_untouched(self):
        conv = make_conv(self.src)
        self.run_m()
        before = (conv / "mirror.json").read_bytes()
        gone = self.dest.parent / "offline"
        args = ["--source", str(self.src), "--dest", str(gone), "--min-free-gb", "0"]
        self.assertEqual(M.main(args), 1)
        self.assertEqual((conv / "mirror.json").read_bytes(), before)

    def test_dated_folder_already_exists(self):
        conv = make_conv(self.src)
        self.run_m()
        make_conv(self.src, tiles={"t1.png": b"retried" * 40, "t2.png": TILES["t2.png"]})
        (self.dest / f"{conv.name}__20261109-210005").mkdir()
        (res,) = self.run_m(now=datetime(2026, 11, 9, 21, 0, 5))
        self.assertFalse(res["md5_ok"])
        self.assertIn("already exists", res["error"])
        rec = json.loads((conv / "mirror.json").read_text())
        self.assertEqual(rec["last_ok_dest_rel"], conv.name)
        self.assertTrue(rec["last_ok_at"])

    def test_one_failing_conversion_does_not_stop_the_next(self):
        make_conv(self.src, "a_first")
        make_conv(self.src, "b_second")
        real = M.copy_one

        def boom(src, dst):
            if "a_first" in str(src):
                raise ValueError("unexpected")
            return real(src, dst)
        with mock.patch.object(M, "copy_one", boom):
            res = self.run_m()
        self.assertEqual([r["md5_ok"] for r in res], [False, True])

    def test_run_lock(self):
        make_conv(self.src)
        lock = M.acquire_lock(self.src / ".mirror.lock", wait=0)
        try:
            args = ["--source", str(self.src), "--dest", str(self.dest), "--min-free-gb", "0"]
            self.assertEqual(M.main(args), 1)
            self.assertEqual(list(self.dest.iterdir()), [])
        finally:
            lock.close()

    def test_700_tiles(self):
        tiles = {f"tile_{i:04d}.png": f"tile{i}".encode() * 5 for i in range(700)}
        make_conv(self.src, tiles=tiles)
        (res,) = self.run_m()
        self.assertTrue(res["md5_ok"], res["error"])
        self.assertEqual(len(list((self.dest / res["dest_rel"] / "tiles").glob("tile_*.png"))), 700)
        self.assertFalse((self.dest / res["dest_rel"] / "input").exists())

    def test_tiles_unverified_flagged_and_lock_files_skipped(self):
        conv = make_conv(self.src, extra={"x.lock": b"l"})
        (conv / "tiles" / "MANIFEST.csv").write_text("output_name,md5\nt1.png,\nt2.png,\n")
        (res,) = self.run_m()
        self.assertTrue(res["md5_ok"])
        self.assertTrue(json.loads((conv / "mirror.json").read_text())["tiles_unverified"])
        self.assertFalse((self.dest / conv.name / "x.lock").exists())


if __name__ == "__main__":
    unittest.main()
