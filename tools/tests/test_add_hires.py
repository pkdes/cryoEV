"""tools/add_hires.py (round 16a stage 2, TODO #85): additive 2x images with a known-answer alignment control."""
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from annotation import object_set as OS
from tools import add_hires as AH


def texture(seed, w=1200, h=852):
    rng = np.random.default_rng(seed)
    a = rng.normal(128, 40, (h // 8, w // 8))
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return im


class AddHires(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ps = self.tmp / "ps"
        (self.ps / "images").mkdir(parents=True)
        self.pkg = self.tmp / "review" / "pkg"
        (self.pkg / "images").mkdir(parents=True)
        (self.pkg / "edits").mkdir()
        rows, imgs = [], []
        for i in range(1, 4):
            stem = f"{i:04d}"
            big = texture(i)
            big.save(self.ps / "images" / f"{stem}.jpg", quality=95)
            big.resize((360, 256), Image.LANCZOS).save(self.pkg / "images" / f"{stem}.jpg", quality=72)
            rows.append({"image": stem, "file": f"images/{stem}.jpg", "orig_name": f"S{i}.jpg"})
            imgs.append({"file": f"{stem}.jpg", "src": f"images/{stem}.jpg", "w": 1200, "h": 852})
        with open(self.ps / "images.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["image", "file", "orig_name"])
            w.writeheader()
            w.writerows(rows)
        self.data = {"id": "pkg", "kind": "objects", "prediction_set": "ps", "images": imgs}
        (self.pkg / "data.js").write_bytes(OS.DATAJS_PREFIX + json.dumps(self.data, separators=(",", ":")).encode() + b";\n")
        (self.pkg / "edits" / "0001.jpg.json").write_text('{"file":"0001.jpg"}', encoding="utf-8")
        (self.pkg / "items.json").write_text("[]", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dry_run_writes_nothing_and_control_separates(self):
        before = sorted(p.name for p in self.pkg.rglob("*"))
        P = AH.plan(self.pkg, ps_dir=self.ps, width=720)
        self.assertTrue(P["control"]["ok"], P["control"]["why"])
        self.assertEqual([r["status"] for r in P["rows"]], ["pass"] * 3)
        self.assertTrue(all(1.5 <= s["shift"] <= 2.5 for s in P["control"]["shift_checks"]))
        self.assertLess(max(n["ncc"] for n in P["control"]["negatives"]), P["control"]["threshold"])
        self.assertEqual(before, sorted(p.name for p in self.pkg.rglob("*")))

    def test_do_is_additive(self):
        edit = (self.pkg / "edits" / "0001.jpg.json").read_bytes()
        out = AH.write(AH.plan(self.pkg, ps_dir=self.ps, width=720))
        self.assertEqual(out["written"], 3)
        data, _, _ = OS._split_datajs((self.pkg / "data.js").read_bytes())
        self.assertEqual(data["images"][0]["hi"], "images/0001.hi.jpg")
        self.assertEqual(data["images"][0]["hi_w"], 720)
        for im in data["images"]:
            for k in AH.HI_KEYS:
                im.pop(k)
        self.assertEqual(data, self.data)                               # nothing but the hi keys changed
        self.assertEqual((self.pkg / "edits" / "0001.jpg.json").read_bytes(), edit)
        with Image.open(self.pkg / "images" / "0001.hi.jpg") as im:
            self.assertEqual((im.size, im.mode), ((720, 511), "L"))
            self.assertFalse(im.info.get("exif"))
        self.assertTrue((self.pkg / "hires_manifest.json").is_file())

    def test_wrong_source_fails_alignment(self):
        """negative: image 1's package JPEG replaced by image 2's -> 0001 fails, the others pass (control)."""
        shutil.copy(self.pkg / "images" / "0002.jpg", self.pkg / "images" / "0001.jpg")
        P = AH.plan(self.pkg, ps_dir=self.ps, width=720)
        st = {r["file"]: r["status"] for r in P["rows"]}
        self.assertEqual(st["0001.jpg"], "fail")
        self.assertEqual(st["0003.jpg"], "pass")

    def test_shifted_source_fails(self):
        big = Image.open(self.ps / "images" / "0002.jpg")
        np_big = np.roll(np.asarray(big), 24, axis=1)                   # 24 px at 1200 = 7.2 px at 360
        Image.fromarray(np_big).save(self.ps / "images" / "0002.jpg", quality=95)
        st = {r["file"]: r["status"] for r in AH.plan(self.pkg, ps_dir=self.ps, width=720)["rows"]}
        self.assertEqual(st["0002.jpg"], "fail")
        self.assertEqual(st["0001.jpg"], "pass")                       # control

    def test_small_source_is_skipped(self):
        Image.open(self.ps / "images" / "0003.jpg").resize((400, 284)).save(self.ps / "images" / "0003.jpg")
        st = {r["file"]: r["status"] for r in AH.plan(self.pkg, ps_dir=self.ps, width=720)["rows"]}
        self.assertEqual(st["0003.jpg"], "skip")

    def test_source_matches_orig_name_and_refuses_box(self):
        src = self.tmp / "orig"
        src.mkdir()
        for i in range(1, 4):
            shutil.copy(self.ps / "images" / f"{i:04d}.jpg", src / f"s{i}.JPG")
        P = AH.plan(self.pkg, ps_dir=self.ps, source=src, width=720)
        self.assertEqual([r["status"] for r in P["rows"]], ["pass"] * 3)
        with self.assertRaises(AH.HiresError):
            AH.sources(self.pkg, self.data, self.ps, Path.home() / "Box" / "x")


if __name__ == "__main__":
    unittest.main()
