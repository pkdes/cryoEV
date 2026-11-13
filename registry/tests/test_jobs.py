"""Contract v2 (jobs and inputs): dataset membership, revise jobs, `job new`, the Queue and the Model page.

- Counts control (known answer): the real v5 S1 export gives 114 / 26 / 39 trained (models.yaml), and the list
  stems equal the CSV `split` rows minus DROPPED. A planted CSV makes `source_split` or a DROPPED row change
  the answer, so using the wrong column fails.
- Revise-job progress equals a count made here from fixture edits/ (wrapper and bare forms; parts and SANDBOX
  skipped; a package that is `applied` does not make its job done).
- `job new`: id collision refused; ready -> in_progress -> done moves lanes in the page model.
- A temp copy of the real registry: import adds the v2 fields, re-import adds 0 lines, lv inputs, the page.
  JOBS_TEST_SHOTS=<folder> keeps 390 px and 1280 px screenshots of that page (Playwright + Edge/Chrome).
"""
import contextlib
import csv
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REG.parent))                   # the repo root: `from registry import ...`
from registry import history_page as hp  # noqa: E402
from registry import ids  # noqa: E402
from registry import importers  # noqa: E402
from registry import store as registry  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

V5S1 = "roboflow - 20261028 cryoai v5 S1"
V5DS = "roboflow_20261028_cryoai_v5_S1_evnonev"
V5 = "v5_evnonev_y8m_20261029"
CACHE = importers.find_sibling("CryoAI")
LAPTOP = importers.find_sibling("CryoAI-laptop")
REAL_CSV = CACHE / "training images" / V5S1 / "split_assignment.csv" if CACHE else None
REAL_REG = LAPTOP / "registry" if LAPTOP else None
SHOTS = os.environ.get("JOBS_TEST_SHOTS")
PULL = "roboflow_pull_20990101_0000"


sys.path.append(str(Path(__file__).resolve().parents[2] / "tests"))
from cryoev_testkit import browser  # noqa: E402  (the gate skips these when no UI file changed)


def lines_by_file(root):
    return {p.name: p.read_bytes().count(b"\n") for p in Path(root).glob("*.jsonl")}


# ---------------------------------------------------------------- dataset membership
class PlantedSplitCsv(unittest.TestCase):
    """A CSV where `source_split` and the DROPPED rows would change the answer if used."""
    ROWS = [  # image, batch, split, source_split
        ("a.png", "20260101 x", "train", "test"),
        ("b.png", "20260101 x", "train", "train"),
        ("c.png", "20260101 x", "valid", "train"),
        ("d.png", "20260101 x", "test", "valid"),
        ("e.png", "20260101 x", "DROPPED-duplicate", "train"),
        ("f.png", "20260101 x", "DROPPED-duplicate", "test"),
        ("g.png", "20261012 lipo", "test", "train"),
    ]
    EXPECTED = {"train": 2, "valid": 1, "test": 2}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.cache = t / "CryoAI"
        exp = self.cache / "training images" / V5S1
        exp.mkdir(parents=True)
        with open(exp / "split_assignment.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "batch", "montage", "split", "source_split", "history_split"])
            for im, b, s, ss in self.ROWS:
                w.writerow([im, b, "", s, ss, ss])
        self.csv = exp / "split_assignment.csv"
        (self.cache / "training outputs").mkdir()
        self.laptop = t / "CryoAI-laptop"
        self.laptop.mkdir()
        self.root = t / "registry"

    def tearDown(self):
        self.tmp.cleanup()

    def test_wrong_column_or_dropped_rows_would_change_the_answer(self):   # the control is discriminating
        src = Counter(ss for _, _, s, ss in self.ROWS)
        with_dropped = Counter(s for _, _, s, _ in self.ROWS)
        self.assertNotEqual({k: src[k] for k in self.EXPECTED}, self.EXPECTED)
        self.assertNotEqual(dict(with_dropped), self.EXPECTED)

    def test_split_membership_uses_split_without_dropped(self):
        rows = importers.split_membership(self.csv)
        self.assertEqual(sorted((s, sp) for s, sp, _ in rows),
                         [("a_png", "train"), ("b_png", "train"), ("c_png", "valid"), ("d_png", "test"), ("g_png", "test")])
        self.assertEqual(importers.split_counts({(s, sp) for s, sp, _ in rows}), self.EXPECTED)

    def test_importer_dataset_record(self):
        imp = importers.Importer(self.root, laptop=self.laptop, cache=self.cache, processing=None).scan()
        ds = {r["id"]: r for e, r in imp.out if e == "dataset"}
        self.assertIn(V5DS, ds)                                  # from models.yaml source_run
        d = ds[V5DS]
        self.assertEqual(d["counts_trained"], self.EXPECTED)
        self.assertEqual(d["counts_listed"], {})                 # no images registered in this fixture
        self.assertEqual(d["not_listed"], {"excluded_session": 1, "not_in_registry": 4})
        self.assertEqual(d["inputs"], [{"id": V5S1, "role": "source", "entity": "roboflow_export"}])
        text = imp.lists[d["image_list"]["file"]]
        self.assertEqual(text, "a_png\ttrain\nb_png\ttrain\nc_png\tvalid\nd_png\ttest\ng_png\ttest\n")
        unknown = [r for r in ds.values() if r["id"] != V5DS]       # datasets without data: unknown, never 0
        self.assertTrue(unknown)
        for r in unknown:
            self.assertIsNone(r["counts_trained"])
            self.assertIsNone(r["image_list"])
            self.assertIsNone(r["counts"]["train"]["images"])


@unittest.skipUnless(REAL_CSV and REAL_CSV.exists(), "real v5 S1 split_assignment.csv not found")
class RealSplitControl(unittest.TestCase):
    def test_v5_s1_trained_counts_known_answer(self):
        rows = importers.split_membership(REAL_CSV)
        pairs = {(s, sp) for s, sp, _ in rows}
        self.assertEqual(importers.split_counts(pairs), {"train": 114, "valid": 26, "test": 39})   # models.yaml
        with open(REAL_CSV, newline="", encoding="utf-8-sig") as f:
            want = {(ids.image_key(r["image"]), r["split"]) for r in csv.DictReader(f)
                    if not r["split"].startswith("DROPPED")}
        self.assertEqual(pairs, want)
        self.assertEqual(len(pairs), 179)


# ---------------------------------------------------------------- revise jobs
def write_package(laptop, folder, pkg_id, n_images, edits, created="2099-01-02T10:00:00"):
    d = laptop / "review" / folder
    (d / "edits").mkdir(parents=True)
    data = {"id": pkg_id, "source": str(laptop / PULL / "current.coco.json"), "run": f"{V5} (F5)",
            "created": created, "images": [{"file": f"img{i}.png"} for i in range(n_images)]}
    (d / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
    for name, rec in edits.items():
        (d / "edits" / f"{name}.json").write_text(json.dumps(rec), encoding="utf-8")


def wrap(file, edit, pkg="pkg_a"):
    return {"file": file, "pkg": pkg, "fmt": 1, "edit": edit, "rx": {}}


class ReviseJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.laptop, self.cache, self.root = t / "CryoAI-laptop", t / "CryoAI", t / "registry"
        self.cache.mkdir()
        dec = {"g1": {"v": "del", "who": "Pujan"}}
        write_package(self.laptop, "pkg_a", "pkg_a", 10, {
            "img0.png": wrap("img0.png", {"dec": dec, "reviewed": True}),
            "img1.png": wrap("img1.png", {"dec": dec, "reviewed": False}),
            "img2.png": {"dec": dec, "reviewed": True, "t": 1},                 # old bare form
            "img3.png": {"dec": {}, "reviewed": False},                         # bare, untouched
            "img4.png": wrap("img4.png", {"dec": {}, "reviewed": True}),
        })
        write_package(self.laptop, "pkg_a_part1", "pkg_a_part1", 5,
                      {"img0.png": wrap("img0.png", {"dec": dec, "reviewed": True}, "pkg_a_part1")})
        write_package(self.laptop, "SANDBOX_pkg_a", "pkg_a", 10,
                      {"img0.png": wrap("img0.png", {"dec": dec, "reviewed": True})})
        write_package(self.laptop, "pkg_b", "pkg_b", 4, {"x.png": wrap("x.png", {"dec": {}, "reviewed": False}, "pkg_b")},
                      created="2099-01-03T10:00:00")
        for sub in ("training images", "training outputs"):
            (self.cache / sub).mkdir()
        lv = self.laptop / "labels" / "lv_20990105_0900"                      # pkg_a has been applied
        lv.mkdir(parents=True)
        (lv / "version.json").write_text(json.dumps({
            "id": lv.name, "base": PULL, "pkg": "pkg_a", "edits_dir": "x", "n_images_applied": 3,
            "counts_by_op": {}, "class_groups_sha256": "0" * 64, "people_sha256": "0" * 64,
            "tool": "test", "created": "2099-01-05T09:00:00"}), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def expected_reviewed(self, folder):                 # count made here, independent of importers.edit_body
        n = 0
        for f in (self.laptop / "review" / folder / "edits").glob("*.json"):
            o = json.loads(f.read_text(encoding="utf-8"))
            e = o["edit"] if "edit" in o else o
            n += e.get("reviewed") is True
        return n

    def run_import(self):
        return importers.run(self.root, laptop=self.laptop, cache=self.cache, processing=None)

    def test_label_version_review_scope(self):
        """A partial ship (apply_review_edits.py reviewed_only) after the older lv: the record carries review_scope,
        n_reviewed, n_package_images, n_held_back and supersedes; the older lv (no scope fields) stays valid and is
        now superseded. Schema-checked on append (registry.append refuses an invalid record)."""
        lv2 = self.laptop / "labels" / "lv_20990106_0900"
        lv2.mkdir(parents=True)
        (lv2 / "version.json").write_text(json.dumps({
            "id": lv2.name, "base": PULL, "pkg": "pkg_a", "edits_dir": "x", "n_images_applied": 1,
            "counts_by_op": {}, "class_groups_sha256": "0" * 64, "people_sha256": "0" * 64, "tool": "test",
            "created": "2099-01-06T09:00:00", "review_scope": "reviewed_only", "n_package_images": 10, "n_reviewed": 3,
            "reviewed_images": ["img0_png", "img2_png", "img4_png"], "n_held_back": 1, "held_back_images": ["img1_png"],
            "supersedes": "lv_20990105_0900"}), encoding="utf-8")
        res = self.run_import()
        self.assertEqual([k for k in res["refused"] if k.startswith("label_version")], [])
        cur = registry.current("label_version", self.root)
        new, old = cur["lv_20990106_0900"], cur["lv_20990105_0900"]
        self.assertEqual((new["review_scope"], new["n_reviewed"], new["n_package_images"], new["n_held_back"],
                          new["supersedes"], new["status"]), ("reviewed_only", 3, 10, 1, "lv_20990105_0900", "created"))
        self.assertEqual(new["reviewed_images"], ["img0_png", "img2_png", "img4_png"])
        self.assertEqual(old["status"], "superseded")
        self.assertNotIn("review_scope", old)
        schemas = importers.registry.load_schemas()
        for rec in (new, old):
            self.assertEqual(importers.registry.check_record(rec, schemas), [], rec["id"])
        bad = dict(new, review_scope="everything")
        self.assertTrue(importers.registry.check_record(bad, schemas))          # control: a wrong scope fails
        self.assertEqual(sum(self.run_import()["appended"].values()), 0)        # re-import: 0 lines

    def test_progress_status_and_skips(self):
        self.run_import()
        cur = registry.current("job", self.root)
        self.assertEqual(sorted(cur), ["job_20990102_pkg_a", "job_20990103_pkg_b"])   # no part, no SANDBOX
        a, b = cur["job_20990102_pkg_a"], cur["job_20990103_pkg_b"]
        self.assertEqual(self.expected_reviewed("pkg_a"), 3)
        self.assertEqual(a["progress"], {"done": self.expected_reviewed("pkg_a"), "total": 10, "unit": "images reviewed"})
        self.assertEqual((a["type"], a["where"], a["package"], a["n_images"]), ("revise", "app", "pkg_a", 10))
        self.assertEqual(a["status"], "in_progress")                 # applied package, but NOT done
        self.assertEqual(registry.current("review_package", self.root)["pkg_a"]["status"], "applied")
        self.assertEqual(a["outputs"], ["lv_20990105_0900"])
        self.assertEqual(a["who"], "pujan")                          # alias "Pujan" -> people.json id
        self.assertEqual((b["status"], b["progress"]["done"]), ("ready", 0))
        self.assertEqual(registry.current("label_version", self.root)["lv_20990105_0900"]["inputs"],
                         [{"id": PULL, "role": "base", "entity": "pull"},
                          {"id": "pkg_a", "role": "edits", "entity": "review_package"}])
        # re-import: 0 lines; explicit set-status wins and survives a re-import with new progress
        before = lines_by_file(self.root)
        self.assertEqual(sum(self.run_import()["appended"].values()), 0)
        self.assertEqual(before, lines_by_file(self.root))
        registry.set_status("job_20990102_pkg_a", "done", "pujan", root=self.root)
        p = self.laptop / "review" / "pkg_a" / "edits" / "img1.png.json"
        p.write_text(json.dumps(wrap("img1.png", {"dec": {}, "reviewed": True})), encoding="utf-8")
        self.assertEqual(self.run_import()["appended"]["job"], 1)
        a = registry.current("job", self.root)["job_20990102_pkg_a"]
        self.assertEqual((a["status"], a["status_by"], a["progress"]["done"]), ("done", "pujan", 4))
        self.assertEqual(sum(self.run_import()["appended"].values()), 0)


# ---------------------------------------------------------------- job new + lanes
class JobNew(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "registry"
        self.lst = Path(self.tmp.name) / "batch.txt"
        self.lst.write_text("# blind tiles\n20990101 - x__a.png\ttrain\n20990101-x__b_png\n\n20990101-x__a_png\n",
                            encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def page(self):
        cur, hist, stats = hp.read_registry([self.root])
        return hp.build_model(cur, hist, stats, [self.root])

    def lane(self, m, key):
        for L in m["queue"]:
            if key in L["open"]:
                return (L["type"], "open")
            if key in L["done"]:
                return (L["type"], "done")
        return None

    def test_new_set_status_lanes(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = registry.main(["--root", str(self.root), "job", "new", "--type", "annotate", "--where", "roboflow",
                                "--name", "Blind tiles 2099", "--images", str(self.lst), "--who", "pujan"])
        self.assertEqual(rc, 0)
        self.assertIn("2 images", out.getvalue())
        jobs = registry.current("job", self.root)
        self.assertEqual(len(jobs), 1)
        jid, j = next(iter(jobs.items()))
        self.assertRegex(jid, r"^job_\d{8}_blind_tiles_2099$")
        self.assertEqual((j["status"], j["n_images"], j["progress"]), ("ready", 2, {"done": 0, "total": 2,
                                                                                    "unit": "images labeled"}))
        lf = self.root / j["image_list"]["file"]
        self.assertEqual(lf.read_text(encoding="utf-8"), "20990101-x__a_png\n20990101-x__b_png\n")
        self.assertEqual(importers.sha256_file(lf), j["image_list"]["sha256"])
        k = f"job|{jid}"
        m = self.page()
        self.assertEqual(self.lane(m, k), ("annotate", "open"))
        self.assertEqual(m["recs"][k]["job"]["next"], "Start")
        registry.set_status(jid, "in_progress", "pujan", root=self.root)
        m = self.page()
        self.assertEqual((self.lane(m, k), m["recs"][k]["job"]["next"]), (("annotate", "open"), "Label in roboflow"))
        with self.assertRaises(registry.RegistryError):
            registry.set_status(jid, "blocked", "pujan", root=self.root)          # blocked needs a reason
        registry.set_status(jid, "blocked", "pujan", root=self.root, reason="tiles not uploaded")
        m = self.page()
        self.assertEqual(m["recs"][k]["job"]["blocked"], "tiles not uploaded")
        registry.set_status(jid, "done", "pujan", root=self.root)
        m = self.page()
        self.assertEqual(self.lane(m, k), ("annotate", "done"))
        self.assertNotIn("blocked_reason", registry.current("job", self.root)[jid])
        for bad in ("nobody",):
            with self.assertRaises(registry.RegistryError):
                registry.set_status(jid, "ready", bad, root=self.root)

    def test_id_collision_refused(self):
        registry.new_job("check", "laptop", "Same name", images=self.lst, root=self.root, date="20990101")
        with self.assertRaises(registry.RegistryError):
            registry.new_job("check", "laptop", "same  NAME", images=self.lst, root=self.root, date="20990101")
        ds = json.loads((REG / "samples" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()[0])
        registry.append(dict(ds, id="job_20990102_other"), root=self.root)          # another entity owns the id
        with self.assertRaises(registry.RegistryError):
            registry.new_job("train", "desktop", "other", images=self.lst, root=self.root, date="20990102")
        with self.assertRaises(registry.RegistryError):
            registry.new_job("train", "desktop", "x", images=self.lst, root=self.root, who="nobody")
        with self.assertRaises(registry.RegistryError):
            registry.new_job("train", "desktop", "y", session="20261012", root=self.root)
        self.assertEqual(len(registry.current("job", self.root)), 1)
        self.assertFalse((self.root / "lists" / "job_20990102_other.txt").exists())


# ---------------------------------------------------------------- real registry, temp copy
@unittest.skipUnless(REAL_REG and REAL_REG.is_dir() and CACHE, "real registry / CryoAI cache not found")
class RealRegistryCopy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "registry"
        cls.root.mkdir()
        for f in REAL_REG.glob("*.jsonl"):                   # never append to the real registry
            shutil.copy2(f, cls.root / f.name)
        cls.real_sizes = {f.name: f.stat().st_size for f in REAL_REG.glob("*.jsonl")}
        cls.real_lists = sorted(p.name for p in (REAL_REG / "lists").glob("*")) if (REAL_REG / "lists").is_dir() else None
        cls.first = importers.run(cls.root)
        cls.before = lines_by_file(cls.root)
        cls.second = importers.run(cls.root)
        cls.cur = registry.current(root=cls.root)
        cls.page_path = Path(cls.tmp.name) / "page" / "history.html"
        cls.model = hp.build([cls.root], cls.page_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_real_registry_untouched(self):
        self.assertEqual(self.real_sizes, {f.name: f.stat().st_size for f in REAL_REG.glob("*.jsonl")})
        now = sorted(p.name for p in (REAL_REG / "lists").glob("*")) if (REAL_REG / "lists").is_dir() else None
        self.assertEqual(self.real_lists, now)
        self.assertTrue((self.root / "lists" / f"{V5DS}.tsv").is_file())

    def test_reimport_adds_nothing(self):
        self.assertEqual(sum(self.second["appended"].values()), 0, self.second["appended"])
        self.assertEqual(self.before, lines_by_file(self.root))

    def test_v5_dataset_counts(self):
        d = self.cur[V5DS]
        self.assertEqual(d["counts_trained"], {"train": 114, "valid": 26, "test": 39})
        self.assertEqual(d["counts_listed"], {"train": 103, "valid": 23, "test": 34})
        self.assertEqual(d["not_listed"], {"excluded_session": 19})
        lf = self.root / d["image_list"]["file"]
        self.assertEqual(importers.sha256_file(lf), d["image_list"]["sha256"])
        got = {tuple(x.split("\t")) for x in lf.read_text(encoding="utf-8").splitlines()}
        self.assertEqual(Counter(s for _, s in got), Counter({"train": 114, "valid": 26, "test": 39}))
        v4 = self.cur["roboflow_20260924_cryoai_v4_singleclass"]
        self.assertIsNone(v4["counts_trained"])                                    # unknown, never 0
        self.assertIsNone(v4["counts"]["train"]["images"])

    def test_inputs(self):
        self.assertEqual(self.cur["lv_20261030_0909"]["inputs"],
                         [{"id": "roboflow_pull_20261029_1811", "role": "base", "entity": "pull"},
                          {"id": "prelipo_v5F5_review_v2", "role": "edits", "entity": "review_package"}])
        self.assertEqual(self.cur[V5]["inputs"],
                         [{"id": V5DS, "role": "train_data", "entity": "dataset"},
                          {"id": "v4_y8m_20260924", "role": "init_weights", "entity": "model"}])
        self.assertEqual(self.cur[V5DS]["inputs"], [{"id": V5S1, "role": "source", "entity": "roboflow_export"}])

    def test_revise_job_v2(self):
        jobs = {j["package"]: j for j in registry.current("job", self.root).values() if j.get("package")}
        j = jobs["prelipo_v5F5_review_v2"]
        n = 0
        for f in (LAPTOP / "review" / "prelipo_v5F5_review_v2" / "edits").glob("*.json"):
            o = json.loads(f.read_text(encoding="utf-8"))
            n += (o["edit"] if "edit" in o else o).get("reviewed") is True
        self.assertEqual(j["progress"]["done"], n)
        self.assertEqual(j["progress"]["total"], j["n_images"])
        self.assertNotEqual(j["status"], "done")
        self.assertFalse(any("_part" in p or p.upper().startswith("SANDBOX") for p in jobs))

    def test_model_page(self):
        mp = self.model["recs"][f"model|{V5}"]["mp"]
        self.assertEqual(mp["badge"], "production")
        self.assertEqual([(s["split"], s["trained"], s["listed"]) for s in mp["splits"]],
                         [("train", 114, 103), ("valid", 26, 23), ("test", 39, 34)])
        self.assertEqual(mp["note"], "+19 trained on, not listed (20261012 lipo, excluded from registry)")
        L = self.model["lists"][mp["list"]]
        self.assertEqual((len(L["rows"]), L["hidden"], L["problem"]), (160, 19, None))
        self.assertEqual(mp["init"], ["v4_y8m_20260924", "model|v4_y8m_20260924"])
        v4 = self.model["recs"]["model|v4_y8m_20260924"]["mp"]
        self.assertTrue(all(s["trained"] is None for s in v4["splits"]))
        html = self.page_path.read_text(encoding="utf-8")
        lf = self.root / self.cur[V5DS]["image_list"]["file"]
        hidden = [x.split("\t")[0] for x in lf.read_text(encoding="utf-8").splitlines()
                  if f"image|{x.split(chr(9))[0]}" not in self.model["recs"]]
        self.assertEqual(len(hidden), 19)
        for stem in hidden:                                   # excluded-session image names are not embedded
            self.assertNotIn(stem, html)

    @browser
    def test_browser(self):
        if sync_playwright is None:
            self.skipTest("playwright not installed")
        with sync_playwright() as p:
            browser = None
            for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
                try:
                    browser = p.chromium.launch(headless=True, **kw)
                    break
                except Exception:
                    continue
            if browser is None:
                self.skipTest("no Edge / Chrome for Playwright")
            try:
                for width in (390, 1280):
                    ctx = browser.new_context(viewport={"width": width, "height": 900})
                    ctx.route("**/*", lambda r: r.continue_() if r.request.url.startswith("file:") else r.abort())
                    page = ctx.new_page()
                    errors = []
                    page.on("pageerror", lambda e: errors.append(str(e)))
                    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
                    page.goto(self.page_path.as_uri())
                    page.wait_for_selector("#queue .card")
                    self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), width)
                    q = page.inner_text("#queue")
                    for t in ("Annotate", "Revise", "Train", "Check", "images reviewed", "Next:"):
                        self.assertIn(t, q)
                    if SHOTS:
                        Path(SHOTS).mkdir(parents=True, exist_ok=True)
                        page.screenshot(path=str(Path(SHOTS) / f"queue_{width}.png"), full_page=width > 400)
                    page.click(f"#models .rec[data-k='model|{V5}']")
                    d = page.inner_text("#detail")
                    for t in ("Model page", "production", "114", "+19 trained on, not listed", V5DS, "v4_y8m_20260924"):
                        self.assertIn(t, d)
                    page.click("#detail .imgs summary")
                    self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), width)
                    if SHOTS:
                        page.screenshot(path=str(Path(SHOTS) / f"model_{width}.png"), full_page=width > 400)
                    self.assertEqual(errors, [])
                    ctx.close()
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
