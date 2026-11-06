"""Raw imaging import (2026-11-05): grid fields from imaged_grids_master.csv, session acquisition summary from the
raw .mdoc/.idoc (canonical cryoem_core parse_metadata), image montage / source_file, the `YYYYMMDD-N` grid fix.

- Synthetic tree with known answers: a MontSection, an atlas and a dose-0 search map that would change the
  defocus / mag answers if counted (controls); the excluded session's mdoc is never parsed; the processing tree
  and the parser folder are unchanged afterwards; the per-file cache re-parses only a changed file.
- A temp copy of the real registry (skipped when not present): 20260827-2 tiles all get their grid; grids match
  master rows read here with csv; 20261027 instrument / mag / px / defocus match an independent regex read of
  the raw .mdoc (no parse_metadata); nothing removed from any existing record; re-import adds 0; the page builds.
  Real sample labels and scope names are read from the workspace at test time, never written here.
"""
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REG))
import history_page as hp  # noqa: E402
import importers  # noqa: E402
import registry  # noqa: E402
import validate  # noqa: E402

PARSER = importers.load_metadata_parser()
PROC = importers.PROCESSING
LAPTOP = importers.find_sibling("CryoAI-laptop")
REAL_REG = LAPTOP / "registry" if LAPTOP else None
SCOPE = "Scope 0000 at Example Lab"
RAW = importers.RAW_DIR


def w(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def header(day="02-Jan-99"):
    return (f"PixelSpacing = 7.362\nVoltage = 200\nImageSize = 2880 2046\nMontage = 1\n\n"
            f"[T = SerialEM: {SCOPE}                   {day}  07:00:00    ]\n\n"
            "[T =     Tilt axis angle = 0.0, binning = 2  spot = 5  camera = 0]\n\n")


def zsec(n, mag, px, defocus, dose, t, target=-3, binning=2, tag="ZValue"):
    return (f"[{tag} = {n}]\nMagnification = {mag}\nPixelSpacing = {px}\nDefocus = {defocus}\nExposureDose = {dose}\n"
            f"TargetDefocus = {target}\nBinning = {binning}\nDateTime = 02-Jan-2099  {t}\n\n")


def tree_digest(root):
    h = hashlib.md5()
    for p in sorted(Path(root).rglob("*")):
        h.update(str(p.relative_to(root)).encode())
        if p.is_file():
            h.update(p.read_bytes())
            h.update(str(p.stat().st_mtime_ns).encode())
    return h.hexdigest()


def build_proc(proc):
    w(proc / "imaged_grids_master.csv",
      "session_date,session_grid,inventory_id,sample_label,project,prep_date,grid_type,replicate,glow_discharge,"
      "storage,n_montages,key_source,flags,notes\n"
      '2099-01-01,1,9901,Sample A,PRJ,2098-12-30,Type X,2,"25 mA, 45 s",box 1 / slot 1,3,'
      "20990101/1. Notes + reports/sample_key.csv,,first look\n"
      "2099-01-01,5,9905,Sample E,,bad date,Type Y,1,,box 2,4,20990101-2/1. Notes + reports/sample_key.csv,"
      "check id,second session\n"
      "2099-01-05,2,9901,Sample A (renamed later),PRJ,2098-12-30,Type X,2,,box 1 / slot 1,10,20990105/sample_key.csv,,re-image\n"
      "2026-10-12,3,9912,Excluded,,,,,,,1,20261012/x.csv,,\n")
    w(proc / "20990101/1. Notes + reports/sample_key.csv", "#,Label\n1,Sample A\n")
    w(proc / "20990101/5. Model + labeling/model_input_MANIFEST.csv",
      "output_name,source_file,grid,montage,nm_per_px\ntile_a.png,grid1_1_r0c0_t00.jpg,1,grid1_1,1.4\n")
    s2 = proc / "20990101-2"
    w(s2 / "1. Notes + reports/sample_key.csv", "#,Label\n5,Sample E\n")
    w(s2 / "5. Model + labeling/model_input_MANIFEST.csv",
      "output_name,source_file,grid,montage,nm_per_px\n"
      "e1.png,grid5_1_r0c0_t00.jpg,5,grid5_1,0.67\ne2.png,grid5_1_r0c1_t01.jpg,5,grid5_1,0.67\n")
    raw = s2 / RAW
    # idoc: 2 images at 45k (target -2)
    w(raw / "grid5/grid5-area1.idoc",
      header().replace("7.362", "6.674") + "".join(
          zsec(0, 45000, 6.674, d, 16.0, t, target=-2, tag="Image").replace("[Image = 0]", f"[Image = grid5-area1000{i}.tif]")
          for i, (d, t) in enumerate([(-8.0, "07:00:00"), (-4.0, "07:00:10")])))
    # mdoc montage: 3 images at 11k + a MontSection whose -50 must not count
    w(raw / "grid5-pos1.mrc.mdoc",
      header() + zsec(0, 11000, 7.362, -5.0, 3.6, "07:10:00") + zsec(1, 11000, 7.362, -3.0, 3.6, "07:10:05")
      + zsec(2, 11000, 7.362, -2.0, 3.6, "08:00:00") + zsec(0, 11000, 7.362, -50.0, 3.6, "07:10:00", tag="MontSection"))
    # atlas (counts for dates/scope only) and a dose-0 search map (defocus -100: left out of the stats)
    w(raw / "grid5-atlas.mrc.mdoc", header() + zsec(0, 210, 2739.6, 0.0, 0.0, "06:50:00", binning=4))
    w(raw / "grid5-search.mrc.mdoc", header() + zsec(0, 2600, 30.1, -100.0, 0.0, "06:55:00"))
    w(proc / "20261012" / RAW / "never.mrc.mdoc", header() + zsec(0, 11000, 7.0, -1.0, 1.0, "07:00:00"))


class CountingParser:
    def __init__(self):
        self.paths = []

    def __call__(self, path):
        self.paths.append(path)
        return PARSER(path)


@unittest.skipUnless(PARSER, "canonical parser cryoem_core/metadata.py not found")
class SyntheticRaw(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.proc, cls.root = t / "proc", t / "registry"
        (t / "CryoAI-laptop").mkdir()
        (t / "CryoAI" / "training images").mkdir(parents=True)
        (t / "CryoAI" / "training outputs").mkdir(parents=True)
        cls.lap, cls.cache = t / "CryoAI-laptop", t / "CryoAI"
        build_proc(cls.proc)
        cls.before = tree_digest(cls.proc)
        cls.parser = CountingParser()
        cls.res = importers.run(cls.root, laptop=cls.lap, cache=cls.cache, processing=cls.proc, parser=cls.parser)
        cls.after = tree_digest(cls.proc)
        cls.cur = registry.current(root=cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def imp(self, parser):
        return importers.Importer(self.root, laptop=self.lap, cache=self.cache, processing=self.proc, parser=parser).scan()

    def test_processing_tree_read_only(self):
        self.assertEqual(self.before, self.after)
        self.assertFalse(list(self.proc.rglob("__pycache__")))

    def test_excluded_session_never_parsed(self):
        self.assertEqual(len(self.parser.paths), 4)
        self.assertFalse([p for p in self.parser.paths if "20261012" in p])
        self.assertNotIn("9912", self.cur)
        self.assertNotIn("20261012", json.dumps(self.cur))

    def test_session_acquisition_known_answers(self):
        s = self.cur["20990101-2"]
        self.assertEqual(s["instrument"], SCOPE)
        self.assertEqual(s["magnification"], [11000, 45000])            # 210 (atlas) and 2600 (dose 0) left out
        self.assertEqual(s["pixel_size_a"], [6.674, 7.362])
        # defocus of the 5 imaging images: -8 -5 -4 -3 -2; MontSection -50 and search -100 would change min
        self.assertEqual(s["defocus_um"], {"min": -8.0, "median": -4.0, "max": -2.0})
        self.assertEqual(s["target_defocus_um"], [-3, -2])
        self.assertEqual(s["dose"], {"min": 3.6, "median": 3.6, "max": 16.0})
        self.assertEqual((s["voltage_kv"], s["binning"]), ([200], [2]))  # atlas binning 4 left out
        self.assertEqual(s["acq_dates"], {"first": "2099-01-02T06:50:00", "last": "2099-01-02T08:00:00"})
        self.assertEqual(s["n_mdoc"], 4)
        self.assertNotIn("acquisition_gap", s)
        self.assertEqual(s["sample_labels"], {"5": "Sample E"})
        self.assertEqual(s["local_folder"], "proc/20990101-2")
        self.assertEqual(s["session_grids"], {"5": "9905"})

    def test_session_gaps(self):
        self.assertIn("no `2. Raw images/` folder", self.cur["20990101"]["acquisition_gap"])
        self.assertEqual(self.cur["20990101"]["sample_labels"], {"1": "Sample A"})
        self.assertIn("no local session folder", self.cur["20990105"]["acquisition_gap"])
        self.assertNotIn("local_folder", self.cur["20990105"])

    def test_grid_fields(self):
        g = self.cur["9901"]
        self.assertEqual((g["sample_label"], g["project"], g["prep_date"], g["grid_type"], g["replicate"]),
                         ("Sample A", "PRJ", "2098-12-30", "Type X", 2))   # earliest session's value
        self.assertEqual((g["glow_discharge"], g["storage"], g["n_montages"]), ("25 mA, 45 s", "box 1 / slot 1", 13))
        self.assertEqual(g["notes"], "20990101: first look; 20990105: re-image")
        self.assertNotIn("flags", g)
        e = self.cur["9905"]
        self.assertEqual((e["notes"], e["flags"], e["n_montages"]), ("second session", "check id", 4))
        self.assertNotIn("prep_date", e)                                 # "bad date" is not written

    def test_second_session_grid_fix_and_manifest_fields(self):
        for k in ("20990101-2__e1_png", "20990101-2__e2_png"):
            self.assertEqual(self.cur[k]["grid"], "9905")
            self.assertEqual(self.cur[k]["montage"], "grid5_1")
        a = self.cur["20990101__tile_a_png"]
        self.assertEqual((a["grid"], a["montage"], a["source_file"]), ("9901", "grid1_1", "grid1_1_r0c0_t00.jpg"))

    def test_control_date_key_alone_misses_second_session(self):
        """The bug's mechanism: the master keys rows by date, so (folder, slot) needs key_source."""
        self.assertEqual(validate.master_folder("20990101", "20990101-2/1. Notes + reports/sample_key.csv"), "20990101-2")
        self.assertEqual(validate.master_folder("20990101", "20990101/x.csv + 20990101/y.csv"), "20990101")
        self.assertEqual(validate.master_folder("20990101", "20990102-2/x.csv"), "20990101")   # other day: ignored

    def test_cache_parses_only_changed_files(self):
        self.assertTrue((self.root / importers.MDOC_CACHE).is_file())
        p2 = CountingParser()
        imp = self.imp(p2)
        self.assertEqual((p2.paths, imp.mdoc_cache.hits), ([], 4))
        self.assertEqual(sum(imp.write(dry_run=True).values()), 0)          # same records from the cache
        f = self.proc / "20990101-2" / RAW / "grid5-search.mrc.mdoc"
        try:
            f.write_text(f.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            p3 = CountingParser()
            self.imp(p3)
            self.assertEqual([Path(x).name for x in p3.paths], ["grid5-search.mrc.mdoc"])
        finally:
            f.write_text(f.read_text(encoding="utf-8")[:-1], encoding="utf-8")

    def test_reimport_adds_nothing(self):
        res = importers.run(self.root, laptop=self.lap, cache=self.cache, processing=self.proc, parser=CountingParser())
        self.assertEqual(sum(res["appended"].values()), 0, res["appended"])

    def test_records_validate(self):
        sch = registry.schemas()
        for r in self.cur.values():
            self.assertEqual(validate.check_record(r, sch), [], r["id"])


class Helpers(unittest.TestCase):
    def test_acq_time_and_scope(self):
        self.assertEqual(importers.acq_time("27-Oct-2026  16:27:27"), "2026-10-27T16:27:27")
        self.assertEqual(importers.acq_time("27-Oct-26 16:27:27"), "2026-10-27T16:27:27")
        self.assertIsNone(importers.acq_time("yesterday"))
        self.assertEqual(importers.scope_name(f"SerialEM: {SCOPE}        27-Oct-26  16:27:16    ]"), SCOPE)

    def test_file_summary_without_dose_keeps_all_images(self):
        s = importers.file_summary({"Voltage": "300"}, [{"_section": "ZValue", "Magnification": "5000", "Defocus": "-1"}])
        self.assertEqual((s["mag"], s["defocus"], s["kv"], s["n_imaging"]), ([5000], [-1.0], [300.0], 1))

    @unittest.skipUnless(PARSER and PROC, "canonical parser not found")
    def test_parser_load_writes_no_bytecode(self):
        d = PROC / importers.PARSER_REL.parent
        snap = lambda: {str(p): p.stat().st_mtime_ns for p in d.rglob("*")}   # noqa: E731
        before = snap()
        importers.load_metadata_parser(PROC)
        self.assertEqual(before, snap())


# ---------------------------------------------------------------- real data (temp copy of the live registry)
def regex_mdoc(path):
    """Independent read of a SerialEM .mdoc (no parse_metadata): scope + per-[ZValue] key values."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    scope = re.search(r"^\[T = SerialEM:\s*(.*?)\s+\d{1,2}-[A-Za-z]{3}-\d{2,4}\s+\d", text, re.M).group(1)
    secs = []
    for block in re.split(r"^\[", text, flags=re.M)[1:]:
        if block.startswith("ZValue"):
            secs.append(dict(re.findall(r"^(\w+) = (.*?)\s*$", block, re.M)))
    return scope, secs


@unittest.skipUnless(REAL_REG and REAL_REG.is_dir() and PROC and PARSER, "real registry / processing root not found")
class RealRegistryCopy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "registry"
        cls.root.mkdir()
        for f in REAL_REG.glob("*.jsonl"):                      # never append to the real registry
            shutil.copy2(f, cls.root / f.name)
        if (REAL_REG / "lists").is_dir():
            shutil.copytree(REAL_REG / "lists", cls.root / "lists")
        cls.real_sizes = {f.name: f.stat().st_size for f in REAL_REG.glob("*.jsonl")}
        cls.old = registry.current(root=cls.root)
        cls.first = importers.run(cls.root)
        cls.second = importers.run(cls.root)
        cls.cur = registry.current(root=cls.root)
        cls.page = Path(cls.tmp.name) / "page" / "h.html"
        hp.build([cls.root], cls.page)
        with open(PROC / "imaged_grids_master.csv", newline="", encoding="utf-8-sig") as fh:
            cls.master = list(csv.DictReader(fh))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_real_registry_untouched(self):
        self.assertEqual(self.real_sizes, {f.name: f.stat().st_size for f in REAL_REG.glob("*.jsonl")})

    def test_reimport_adds_nothing(self):
        self.assertEqual(sum(self.second["appended"].values()), 0, self.second["appended"])

    def test_second_session_images_all_get_their_grid(self):
        mf = PROC / "20260827-2" / "5. Model + labeling" / "model_input_MANIFEST.csv"
        with open(mf, newline="", encoding="utf-8-sig") as fh:
            n = sum(1 for _ in csv.DictReader(fh))
        slot_inv = {r["session_grid"]: r["inventory_id"] for r in self.master
                    if r["key_source"].startswith("20260827-2/") and r["inventory_id"]}
        ims = [r for r in self.cur.values() if r["schema"] == "image/1" and r["session"] == "20260827-2"]
        self.assertEqual(len(ims), n)
        self.assertEqual([r["id"] for r in ims if r["grid"] is None], [])
        for r in ims:
            self.assertEqual(r["grid"], slot_inv[r["session_grid"]], r["id"])
            self.assertTrue(r["montage"] and r["source_file"], r["id"])

    def test_grid_labels_match_master(self):
        for gid in ("2694", "26109", "26162", "26161"):                  # hand-checked rows (one grid row each)
            row = next(r for r in self.master if r["inventory_id"] == gid)
            g = self.cur[gid]
            self.assertEqual(g["sample_label"], row["sample_label"], gid)
            self.assertEqual(g.get("prep_date"), row["prep_date"] or None, gid)
            self.assertEqual(g.get("storage"), row["storage"] or None, gid)
            self.assertEqual(g.get("n_montages"), int(row["n_montages"]) if row["n_montages"] else None, gid)
        rows = [r for r in self.master if r["inventory_id"] == "26103"]        # imaged in two sessions
        self.assertEqual(len(rows), 2)
        self.assertEqual(self.cur["26103"]["n_montages"], sum(int(r["n_montages"]) for r in rows))

    def test_20261027_acquisition_matches_raw_mdoc(self):
        s = self.cur["20261027"]
        raw = PROC / "20261027" / RAW
        scope, secs = regex_mdoc(raw / "grid2-pos1.mrc.mdoc")
        self.assertEqual(s["instrument"], scope)
        self.assertIn(int(secs[0]["Magnification"]), s["magnification"])
        self.assertIn(float(secs[0]["PixelSpacing"]), s["pixel_size_a"])
        d = float(secs[0]["Defocus"])
        self.assertTrue(s["defocus_um"]["min"] <= d <= s["defocus_um"]["max"])
        # control: session median recomputed here from every non-atlas .mdoc, dose > 0, by regex
        vals, mags = [], set()
        for f in sorted(raw.glob("*.mdoc")):
            if "atlas" in f.name.lower():
                continue
            for z in regex_mdoc(f)[1]:
                if float(z.get("ExposureDose", 0)) > 0:
                    vals.append(float(z["Defocus"]))
                    mags.add(int(z["Magnification"]))
        self.assertEqual(s["defocus_um"]["median"], round(statistics.median(vals), 3))
        self.assertEqual(s["magnification"], sorted(mags))
        self.assertEqual(s["n_mdoc"], len(list(raw.glob("*.mdoc"))))

    def test_additive_only(self):
        """Every existing field keeps its value; the one allowed change is grid null -> Grid# (20260827-2)."""
        changed = []
        for i, old in self.old.items():
            new = self.cur[i]
            for k, v in old.items():
                if k in ("logged", "recorded_by"):
                    continue
                self.assertIn(k, new, f"{i}: field {k} removed")
                if new[k] != v:
                    changed.append((i, k, v, new[k]))
        for i, k, v, nv in changed:
            self.assertEqual((k, v, self.cur[i]["session"]), ("grid", None, "20260827-2"), i)
            self.assertRegex(nv, r"^\d{4,}$")

    def test_page_builds_and_excluded_session_absent(self):
        html = self.page.read_text(encoding="utf-8")
        self.assertIn("acq_dates", html)
        self.assertNotIn("20261012/", html)


if __name__ == "__main__":
    unittest.main()
