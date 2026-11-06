"""
Data module (portal/modules/data.py): roles through the Gate, every page's fields from a temp registry built from
registry/samples/*.jsonl plus fixture rows (a session with acquisition fields, models, a prediction set per owner,
packages that point outside the known roots), search by id and sample label, the JSON API matching the pages,
thumbnails (a real JPEG of at most 240 px, disk-cached, private + nosniff) and their path safety (an id that resolves
outside the known roots, traversal tries, bad indexes, links out), privacy of prediction sets, the registry cache,
and the pages in a browser (Playwright, Edge channel: 390 and 1280 px, light and dark, no horizontal scroll,
same-origin requests only, no console errors), and the Lineage tab (a temp chain model -> run -> dataset -> label
version -> sessions, unknown parents, cycles, hidden prediction sets, escaping, the pages in a browser).
Screenshots go to $CJ_SHOT_DIR when set.

Temp roster / registry / data folders only (cjtest.Env); never the real ones. The two data roots are temp folders
named through CRYOEV_LAPTOP_ROOT / CRYOEV_PROCESSING_ROOT.

  python -m unittest discover -s portal/tests -v
"""
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import data as D  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

REPO = Path(__file__).resolve().parent.parent.parent
SAMPLES = REPO / "registry" / "samples"
XSS = '<img src=x onerror=alert(1)> label'
JPEG = b"\xff\xd8\xff"


def write_img(path: Path, size=(640, 480), color=(90, 120, 160), fmt=None):
    from PIL import Image, ImageDraw
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color)
    ImageDraw.Draw(im).ellipse((50, 50, size[0] - 50, size[1] - 50), outline=(250, 250, 250), width=6)
    im.save(path, fmt or ("PNG" if path.suffix.lower() == ".png" else "JPEG"))


def fixture_records() -> dict:
    """Extra registry rows on top of the samples: entity -> [record]."""
    sess = {"schema": "session/1", "id": "20261104", "logged": "2026-11-05T09:00:00", "date": "2026-11-04",
            "date_basis": "folder", "instrument": "Glacios", "magnification": [11000, 45000],
            "pixel_size_a": [1.8, 7.36], "defocus_um": {"min": -9.5, "median": -3.5, "max": -1.5},
            "dose": {"min": 3.5, "median": 3.6, "max": 60.1}, "voltage_kv": [200],
            "acq_dates": {"first": "2026-11-04T10:05:44", "last": "2026-11-04T18:16:12"}, "n_mdoc": 12,
            "n_imaging": 10, "n_other": 2, "other_magnification": [2600],
            "sample_labels": {"1": "Fixture <b>EV</b> prep", "2": XSS}, "session_grids": {"1": "2602", "2": None},
            "local_folder": "Cryo-image-processing/20261104", "status": "processed"}
    prod = {"schema": "model/1", "id": "v9_prod_20261101", "logged": "2026-11-02T10:00:00", "run": "r9",
            "weights": "models/v9/best.pt", "imgsz": 1440, "classes": ["EV", "non-EV"], "conf": 0.5,
            "init_from": "v5_evnonev_y8m_20261029", "status": "production",
            "dataset": "roboflow_20261028_cryoai_v5_S1_evnonev",
            "metrics": {"mask_map50": 0.612, "mask_map50_95": 0.3214, "per_class": {"EV": 0.7}}}
    pkgs = [{"schema": "review_package/1", "id": "pkg_ok", "logged": "2026-11-01T10:00:00", "path": "x/review/pkg_ok",
             "kind": "review", "base": "b", "base_kind": "pull", "n_images": 3, "built": "2026-11-01T10:00:00",
             "status": "in_review"},
            {"schema": "review_package/1", "id": "pkg_outside", "logged": "2026-11-01T10:00:00", "path": "@OUTSIDE@",
             "kind": "review", "base": "b", "base_kind": "pull", "n_images": 1, "built": "2026-11-01T10:00:00",
             "status": "in_review"},
            {"schema": "review_package/1", "id": "pkg_escape", "logged": "2026-11-01T10:00:00",
             "path": "x/review/pkg_escape",
             "kind": "review", "base": "b", "base_kind": "pull", "n_images": 3, "built": "2026-11-01T10:00:00",
             "status": "in_review"},
            {"schema": "review_package/1", "id": "pkg_owned", "logged": "2026-11-01T10:00:00", "path": "x",
             "kind": "review", "owner": "otherperson", "base": "b", "base_kind": "pull", "n_images": 2,
             "built": "2026-11-01T10:00:00", "status": "in_review"}]
    labels = [{"schema": "label_version/1", "id": "lv_priv", "logged": "2026-11-02T00:00:00",
               "pkg": "pred_ps_20261105_cccccc",
               "status": "active", "n_images_applied": 1},
              {"schema": "label_version/1", "id": "lv_owned", "logged": "2026-11-02T00:00:00", "pkg": "pkg_owned",
               "status": "active", "n_images_applied": 1}]
    ps = [{"schema": "prediction_set/1", "id": "ps_20261105_aaaaaa", "logged": "2026-11-05T10:00:00",
           "created": "2026-11-05T10:00:00", "owner": "adv", "model": "v9_prod_20261101", "path": "p",
           "source": {"kind": "upload"}, "n_images": 2, "status": "done", "conf": 0.25},
          {"schema": "prediction_set/1", "id": "ps_20261105_bbbbbb", "logged": "2026-11-05T11:00:00",
           "created": "2026-11-05T11:00:00", "owner": "otherperson", "model": "v9_prod_20261101", "path": "p",
           "source": {"kind": "upload"}, "n_images": 2, "status": "done", "conf": 0.25}]
    grid = {"schema": "grid/1", "id": "2603", "logged": "2026-11-05T09:00:00", "sample_label": "Second grid sample",
            "grid_type": "Gold grid", "prep_date": "2026-10-06", "n_montages": 4}
    return {"session": [sess], "model": [prod], "review_package": pkgs, "prediction_set": ps, "grid": [grid],
            "label_version": labels}


def write_registry(root: Path, outside: Path):
    root.mkdir(parents=True, exist_ok=True)
    extra = fixture_records()
    for f in SAMPLES.glob("*.jsonl"):
        lines = f.read_text(encoding="utf-8").splitlines()
        ent = f.stem
        for rec in extra.get(ent, []):
            lines.append(json.dumps(rec).replace("@OUTSIDE@", str(outside / "pkg_outside").replace("\\", "\\\\")))
        (root / f.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    for ent, recs in extra.items():
        if not (root / f"{ent}.jsonl").exists():
            (root / f"{ent}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    D._registry_cache.clear()
    D._files_cache.clear()
    D._package_cache.clear()
    D._owner_cache.clear()


def build_data(lap: Path, proc: Path, outside: Path):
    for i in range(14):                                  # session 20261104: 14 images on slot 1 + 2 on slot 2
        write_img(proc / "20261104" / D.PROCESSED / "3a. Individual tiles" / f"grid1_{i:02d}_r0c0_t00.jpg")
    for i in range(2):
        write_img(proc / "20261104" / D.PROCESSED / "3a. Individual tiles" / f"grid2_{i:02d}_r0c0_t00.jpg")
    for i in range(30):                                  # a grid with 30 images: 2 pages of 24
        write_img(proc / "20261110" / D.PROCESSED / "3a. Individual tiles" / f"grid1_{i:02d}.jpg", (320, 240))
    pk = lap / "review" / "pkg_ok"
    names = []
    for i in range(3):
        write_img(pk / "images" / f"i{i}.png", (900, 700), fmt="PNG")
        names.append({"file": f"i{i}.png", "src": f"images/i{i}.png"})
    (pk / "data.js").write_text("window.REVIEW_DATA = " + json.dumps({"id": "pkg_ok", "images": names}) + ";",
                                encoding="utf-8")
    pe = lap / "review" / "pkg_escape"                   # data.js tries to reach a file outside the package
    write_img(pe / "images" / "ok.png")
    write_img(lap / "secret_in_laptop.png")
    write_img(outside / "secret.png")
    (pe / "data.js").write_text("window.REVIEW_DATA = " + json.dumps({"id": "pkg_escape", "images": [
        {"src": "images/ok.png"}, {"src": "../../secret_in_laptop.png"}, {"src": "../../../outside/secret.png"},
        {"src": str(outside / "secret.png")}, {"src": "images/nope.png"},
        {"src": "images/ok.txt"}]}) + ";", encoding="utf-8")
    write_img(outside / "pkg_outside" / "images" / "o.png")
    (outside / "pkg_outside" / "data.js").write_text("window.REVIEW_DATA = " + json.dumps({"images": [
        {"src": "images/o.png"}]}) + ";", encoding="utf-8")
    for ps in ("ps_20261105_aaaaaa", "ps_20261105_bbbbbb"):
        for i in range(2):
            write_img(lap / "predict" / ps / "images" / f"{i + 1:04d}.png", (500, 400))
    build_private_packages(lap)
    lv = lap / "review" / "prelipo_v5F5_review_v2"      # the sample label version's package
    for i in range(2):
        write_img(lv / "images" / f"s{i}.jpg")
    (lv / "data.js").write_text("window.REVIEW_DATA = " + json.dumps({"images": [
        {"src": "images/s0.jpg"}, {"src": "images/s1.jpg"}]}) + ";", encoding="utf-8")


def make_package(folder: Path, n=2, **meta):
    """A review package folder: images + data.js (+ review_package.json when meta is given)."""
    names = []
    for i in range(n):
        write_img(folder / "images" / f"i{i}.png", (300, 200))
        names.append({"src": f"images/i{i}.png"})
    (folder / "data.js").write_text("window.REVIEW_DATA = " + json.dumps({"images": names}) + ";", encoding="utf-8")
    if meta:
        (folder / "review_package.json").write_text(json.dumps(meta), encoding="utf-8")


def build_private_packages(lap: Path):
    """Prediction packages that are on disk but NOT in the registry, and one with a registry owner."""
    make_package(lap / "review" / "pkg_owned")
    make_package(lap / "review" / "pred_ps_20261105_cccccc")                         # no owner anywhere
    make_package(lap / "review" / "pred_ps_20261105_dddddd", owner="adv")             # owner in review_package.json
    make_package(lap / "review" / "pred_ps_20261105_eeeeee")                         # owner only in request.json
    (lap / "predict" / "ps_20261105_eeeeee").mkdir(parents=True)
    (lap / "predict" / "ps_20261105_eeeeee" / "request.json").write_text(json.dumps({"owner": "adv"}), encoding="utf-8")


def launch(p):
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:
            continue
    return None


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.adv_key = A.add_key(cls.env.roster, "adv", "advanced")
        cls.bas_key = A.add_key(cls.env.roster, "bas", "basic")
        cls.lap, cls.proc, cls.outside = (cls.env.tmp / "CryoAI-laptop", cls.env.tmp / "Cryo-image-processing",
                                          cls.env.tmp / "outside")
        for d in (cls.lap, cls.proc, cls.outside):
            d.mkdir()
        cls.saved_env = {k: os.environ.get(k) for k in ("CRYOEV_LAPTOP_ROOT", "CRYOEV_PROCESSING_ROOT")}
        os.environ["CRYOEV_LAPTOP_ROOT"], os.environ["CRYOEV_PROCESSING_ROOT"] = str(cls.lap), str(cls.proc)
        build_data(cls.lap, cls.proc, cls.outside)
        write_registry(cls.env.registry, cls.outside)
        cls.env.write_live(T.live_doc())                      # header.js polls it for the nav dots
        cls.p = T.Portal(cls.env.config())
        cls.tok = {"admin": T.login(cls.p.port, cls.env.key), "advanced": T.login(cls.p.port, cls.adv_key),
                   "basic": T.login(cls.p.port, cls.bas_key)}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        for k, v in cls.saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cls.env.cleanup()

    def get(self, path, role="admin"):
        return T.request(self.p.port, "GET", path, T.cookie(self.tok[role]))

    def page(self, path, role="admin", status=200) -> str:
        st, h, body = self.get(path, role)
        self.assertEqual(st, status, path)
        self.assertEqual(T.hget(h, "cache-control"), "no-store")
        return body.decode("utf-8")

    def api(self, path, role="admin", status=200):
        st, h, body = self.get(path, role)
        self.assertEqual(st, status, path)
        self.assertTrue(T.hget(h, "content-type").startswith("application/json"))
        return json.loads(body)


def main_html(page: str) -> str:
    body = page.split('<main class="cj-main">', 1)[1].split("</main>", 1)[0]
    return re.sub(r"<style>.*?</style>", "", body, flags=re.S)


class Roles(Base):
    def test_module_declaration(self):
        self.assertEqual((D.PREFIX, D.ROLES), ("/data", ("admin", "advanced")))
        st, h, _ = self.get("/data")
        self.assertEqual((st, T.hget(h, "location")), (307, "/data/"))

    def test_admin_and_advanced_in_basic_403(self):
        paths = ("/data/", "/data/search?q=x", "/data/list/sessions", "/data/session/20261104", "/data/grid/2602",
                 "/data/model/v9_prod_20261101", "/data/api/overview", "/data/thumb/session/20261104/0.jpg")
        for role in ("admin", "advanced"):
            for p in paths:
                with self.subTest(role=role, path=p):
                    self.assertEqual(self.get(p, role)[0], 200)
        for p in paths + ("/data",):
            with self.subTest(role="basic", path=p):
                st, _, body = self.get(p, "basic")
                self.assertEqual(st, 403)
                self.assertNotIn(b"20261104", body)
        self.assertEqual(self.get("/", "basic")[0], 200)          # control: basic is signed in, just not here
        self.assertEqual(T.request(self.p.port, "GET", "/data/api/overview")[0], 401)


class Pages(Base):
    def test_overview(self):
        t = main_html(self.page("/data/"))
        for word in ("Sessions", "Image sets", "Label versions", "Datasets", "Models", "Session 20261104",
                     "Session 20261027", "v9_prod_20261101", "Production", 'name="q"'):
            self.assertIn(word, t)
        ov = self.api("/data/api/overview")
        c = ov["counts"]
        self.assertEqual((c["sessions"], c["labels"], c["datasets"], c["models"], c["grids"]), (2, 3, 1, 2, 2))
        # 5 packages + 3 prediction sets + 3 pred_ folders not in the registry + 1 export
        self.assertEqual(c["images"], 5 + 3 + 3 + 1)
        models = next(x for x in ov["cards"] if x["kind"] == "models")["rows"]
        self.assertEqual(models[0]["id"], "v9_prod_20261101")     # production first
        self.assertTrue(models[0]["hl"])
        self.assertIn("cj-data-hl", main_html(self.page("/data/model/v9_prod_20261101")))
        self.assertTrue(all(len(x["rows"]) <= D.OVERVIEW_N for x in ov["cards"]))

    def test_session_page_fields(self):
        t = main_html(self.page("/data/session/20261104"))
        for want in ("Glacios", "11,000× and 45,000×", "1.8 Å/px and 7.36 Å/px",
                     "median -3.5 µm (range -9.5 to -1.5)", "median 3.6 e/Å² (range 3.5 to 60.1)",
                     "200 kV", "2026-11-04 10:05 to 2026-11-04 18:16", "10", "2 files below 5,000× (2,600×)",
                     "Grid 2602", "Example carbon grid", "14 images (processed files)", "Processed images", "16"):
            self.assertIn(want, t)
        self.assertEqual(t.count('class="cj-data-img"'), D.STRIP_N)
        self.assertNotIn("<b>EV</b>", t)                          # escaped
        self.assertIn("Fixture &lt;b&gt;EV&lt;/b&gt; prep", t)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt; label", t)
        self.assertNotIn("<img src=x", t)
        self.assertIn("grid number not recorded", t)
        d = self.api("/data/api/session/20261104")
        self.assertEqual((d["n_images"], len(d["thumbs"])), (16, 12))
        self.assertEqual(d["acquisition"]["magnification"], [11000, 45000])
        self.assertEqual([g["n_images"] for g in d["grids"]], [14, 2])

    def test_session_all_images_paged_view(self):
        strip = main_html(self.page("/data/session/20261104"))
        self.assertIn("See all 16 images", strip)
        self.assertIn("/data/session/20261104?page=1", strip)
        full = main_html(self.page("/data/session/20261104?page=1"))
        self.assertEqual(full.count('class="cj-data-img"'), 16)
        self.assertNotIn("See all", full)
        self.assertEqual(self.api("/data/api/session/20261104?page=1")["view"], "paged")

    def test_plain_wording(self):
        self.assertEqual((D.plural(1, "image"), D.plural(0, "image"), D.plural(2, "imaging file")),
                         ("1 image", "0 images", "2 imaging files"))
        self.assertEqual(D.status_word("in_review"), "In review")
        self.assertEqual(D.metric_name("mask_map50_95", "EV"), "Mask mAP50-95 (EV)")
        grids = [r["id"] for r in self.api("/data/api/list/grids")["rows"]]
        self.assertEqual(grids, sorted(grids, key=int, reverse=True))              # numeric, newest first

    def test_sample_session_with_sparse_data(self):
        t = main_html(self.page("/data/session/20261027"))        # in the registry, no images on disk
        self.assertIn("Example sample 1", t)
        self.assertIn("No images found", t)

    def test_grid_page_and_paging(self):
        t = main_html(self.page("/data/grid/2602"))
        for want in ("Example sample A", "Example carbon grid", "2026-10-05", "box 1 / slot 1", "Session 20261104"):
            self.assertIn(want, t)
        self.assertEqual(t.count('class="cj-data-img"'), 14)
        # a grid reaching 30 images over two pages
        from portal.modules import data as dm
        rec = {"schema": "session/1", "id": "20261110", "logged": "2026-11-06T00:00:00", "status": "processed",
               "session_grids": {"1": "2603"}, "sample_labels": {"1": "Second grid sample"}}
        with open(self.env.registry / "session.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.addCleanup(self._drop_last_line, "session.jsonl")
        dm._registry_cache.clear()
        p1 = main_html(self.page("/data/grid/2603"))
        self.assertEqual(p1.count('class="cj-data-img"'), 24)
        self.assertIn("Showing 1-24 of 30", p1)
        self.assertIn("?page=2", p1)
        p2 = main_html(self.page("/data/grid/2603?page=2"))
        self.assertEqual(p2.count('class="cj-data-img"'), 6)
        self.assertIn("Showing 25-30 of 30", p2)
        self.assertEqual(main_html(self.page("/data/grid/2603?page=99")).count('class="cj-data-img"'), 6)   # clamped
        self.assertEqual(main_html(self.page("/data/grid/2603?page=abc")).count('class="cj-data-img"'), 24)

    def _drop_last_line(self, name):
        p = self.env.registry / name
        lines = p.read_text(encoding="utf-8").splitlines()[:-1]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        D._registry_cache.clear()

    def test_model_page(self):
        t = main_html(self.page("/data/model/v9_prod_20261101"))
        for want in ("Production", "Mask mAP50", "Mask mAP50-95", "0.612", "Score by class (EV)", "1440 px",
                     "Started from",
                     "v5_evnonev_y8m_20261029", "roboflow_20261028_cryoai_v5_S1_evnonev"):
            self.assertIn(want, t)
        t2 = main_html(self.page("/data/model/v5_evnonev_y8m_20261029"))
        self.assertIn("Candidate", t2)
        self.assertNotIn(">Production<", t2)
        self.assertIn("Checks on new images", t2)

    def test_dataset_labels_and_images_pages(self):
        t = main_html(self.page("/data/dataset/roboflow_20261028_cryoai_v5_S1_evnonev"))
        self.assertIn("114/26/39 train/valid/test", t)
        self.assertIn("MV as EV", t)
        self.assertIn("v5_evnonev_y8m_20261029", t)               # models trained on it
        self.assertIn("No example images are linked", t)
        t = main_html(self.page("/data/labels/lv_20261030_0846"))
        self.assertIn("includes edits not yet checked by a second person", t)
        self.assertIn("prelipo_v5F5_review_v2", t)
        self.assertEqual(t.count('class="cj-data-img"'), 2)
        t = main_html(self.page("/data/images/pkg_ok"))
        self.assertIn("Review package", t)
        self.assertEqual(t.count('class="cj-data-img"'), 3)
        t = main_html(self.page("/data/images/ps_20261105_aaaaaa", "advanced"))
        self.assertIn("Prediction set", t)
        self.assertEqual(t.count('class="cj-data-img"'), 2)
        t = main_html(self.page("/data/images/roboflow - 20261028 cryoai v5 S1".replace(" ", "%20")))
        self.assertIn("Roboflow export", t)
        self.assertIn("Thumbnails are not shown for Roboflow exports", t)

    def test_unknown_is_404(self):
        for p in ("/data/session/99999999", "/data/grid/nope", "/data/images/nope", "/data/list/nope",
                  "/data/session/20261104%2F..", "/data/session/%3Cscript%3E"):
            with self.subTest(p=p):
                st, _, body = self.get(p)
                self.assertEqual(st, 404)
                self.assertNotIn(b"<script>", body)
        self.api("/data/api/session/99999999", status=404)
        self.api("/data/api/nope/x", status=404)

    def test_search(self):
        t = main_html(self.page("/data/search?q=" + "Example%20sample%20A"))
        self.assertIn("Grid 2602", t)
        self.assertIn("Session 20261104", t)                      # the session that imaged that grid
        t = main_html(self.page("/data/search?q=example%20sample%201"))   # a session slot label, any case
        self.assertIn("Session 20261027", t)
        r = self.api("/data/api/search?q=20261104")
        kinds = {g["kind"]: [x["id"] for x in g["rows"]] for g in r["groups"]}
        self.assertEqual(kinds["sessions"], ["20261104"])
        r = self.api("/data/api/search?q=v9_prod")
        self.assertEqual([g["kind"] for g in r["groups"]], ["models"])
        self.assertIn("Nothing found", main_html(self.page("/data/search?q=zzzzqqq")))
        self.assertIn("Type an id", main_html(self.page("/data/search")))
        t = main_html(self.page("/data/search?q=" + "%22%3E%3Cscript%3Ealert(1)%3C/script%3E"))
        self.assertNotIn("<script>", t)
        self.assertIn("&lt;script&gt;", t)
        self.assertIn("<b>", "<b>")                                # control for the escape checks above

    def test_api_matches_page(self):
        for kind, eid in (("session", "20261104"), ("grid", "2602"), ("model", "v9_prod_20261101"),
                          ("dataset", "roboflow_20261028_cryoai_v5_S1_evnonev"), ("labels", "lv_20261030_0846"),
                          ("images", "pkg_ok")):
            with self.subTest(kind=kind):
                d = self.api(f"/data/api/{kind}/{eid}")
                self.assertEqual((d["schema"], d["kind"], d["id"]), ("cj_data/1", kind, eid))
                page = main_html(self.page(f"/data/{kind}/{eid}"))
                for label, value, _ in d["fields"]:
                    self.assertIn(D.esc(label), page)
                    self.assertIn(D.esc(value), page)
                self.assertEqual(page.count('class="cj-data-img"'), len(d["thumbs"]))
        rows = self.api("/data/api/list/models")["rows"]
        self.assertEqual({r["id"] for r in rows}, {"v9_prod_20261101", "v5_evnonev_y8m_20261029"})


class Privacy(Base):
    def test_prediction_sets_belong_to_their_owner(self):
        mine, theirs = "ps_20261105_aaaaaa", "ps_20261105_bbbbbb"
        self.assertEqual(self.get(f"/data/images/{mine}", "advanced")[0], 200)
        self.assertEqual(self.get(f"/data/images/{theirs}", "advanced")[0], 404)
        self.assertEqual(self.get(f"/data/images/{theirs}", "admin")[0], 200)
        self.assertEqual(self.get(f"/data/thumb/images/{theirs}/0.jpg", "advanced")[0], 404)
        self.assertEqual(self.get(f"/data/thumb/images/{theirs}/0.jpg", "admin")[0], 200)
        self.assertEqual(self.get(f"/data/thumb/images/{mine}/0.jpg", "advanced")[0], 200)
        for url in ("/data/api/list/images", "/data/api/search?q=ps_2026", "/data/api/overview"):
            with self.subTest(url=url):
                body = self.get(url, "advanced")[2].decode()
                self.assertIn(mine, body) if url != "/data/api/overview" else None
                self.assertNotIn(theirs, body)
                self.assertIn(theirs, self.get(url, "admin")[2].decode()) if url != "/data/api/overview" else None
        self.assertEqual(self.get(f"/data/api/images/{theirs}", "advanced")[0], 404)


    def test_unregistered_prediction_packages_fail_closed(self):
        none_id, json_id, req_id = "pred_ps_20261105_cccccc", "pred_ps_20261105_dddddd", "pred_ps_20261105_eeeeee"
        for pid, adv_sees in ((none_id, False), (json_id, True), (req_id, True)):
            with self.subTest(pid=pid):
                for role, expect in (("advanced", adv_sees), ("admin", True)):
                    self.assertEqual(self.get(f"/data/images/{pid}", role)[0], 200 if expect else 404, (pid, role))
                    self.assertEqual(self.get(f"/data/thumb/images/{pid}/0.jpg", role)[0], 200 if expect else 404)
                    self.assertEqual(self.get(f"/data/api/images/{pid}", role)[0], 200 if expect else 404)
                    for url in ("/data/api/list/images", f"/data/api/search?q={pid}", "/data/api/overview"):
                        body = json.loads(self.get(url, role)[2])
                        shown = pid in json.dumps(body.get("groups", body.get("rows", [])))   # not the echoed query
                        if url.endswith("overview"):
                            continue                                  # the overview lists only the 5 newest
                        self.assertEqual(shown, expect, (pid, role, url))
                    self.assertEqual(f'/data/images/{pid}' in self.page("/data/list/images", role), expect)
        self.assertNotIn(none_id.encode(), self.get(f"/data/images/{none_id}", "advanced")[2])    # nothing leaks
        self.assertEqual(self.get("/data/images/pred_ps_20261105_ffffff", "admin")[0], 404)   # control: absent

    def test_open_in_review_link_only_for_allowed(self):
        t = main_html(self.page("/data/images/pred_ps_20261105_dddddd", "advanced"))
        self.assertIn('href="/review/pred_ps_20261105_dddddd/"', t)
        self.assertNotIn("/review/", main_html(self.page("/data/images/pkg_ok")))

    def test_owned_review_package(self):
        self.assertEqual(self.get("/data/images/pkg_owned", "advanced")[0], 404)
        self.assertEqual(self.get("/data/thumb/images/pkg_owned/0.jpg", "advanced")[0], 404)
        self.assertEqual(self.get("/data/images/pkg_owned", "admin")[0], 200)
        self.assertEqual(self.get("/data/thumb/images/pkg_owned/0.jpg", "admin")[0], 200)
        self.assertNotIn(b"pkg_owned", self.get("/data/api/list/images", "advanced")[2])
        self.assertIn(b"pkg_owned", self.get("/data/api/list/images", "admin")[2])

    def test_label_version_hides_a_private_package(self):
        for lv, pkg in (("lv_priv", "pred_ps_20261105_cccccc"), ("lv_owned", "pkg_owned")):
            with self.subTest(lv=lv):
                t = main_html(self.page(f"/data/labels/{lv}", "advanced"))
                self.assertNotIn(pkg, t)
                self.assertEqual(t.count('class="cj-data-img"'), 0)
                self.assertNotIn(pkg, self.get(f"/data/api/labels/{lv}", "advanced")[2].decode())
                self.assertEqual(self.get(f"/data/thumb/labels/{lv}/0.jpg", "advanced")[0], 404)
                ta = main_html(self.page(f"/data/labels/{lv}", "admin"))
                self.assertIn(pkg, ta)
                self.assertEqual(ta.count('class="cj-data-img"'), 2)
                self.assertEqual(self.get(f"/data/thumb/labels/{lv}/0.jpg", "admin")[0], 200)


class Thumbnails(Base):
    def thumb(self, path, role="admin"):
        return self.get(path, role)

    def test_jpeg_headers_and_cache(self):
        st, h, body = self.thumb("/data/thumb/session/20261104/0.jpg")
        self.assertEqual(st, 200)
        self.assertEqual(T.hget(h, "content-type"), "image/jpeg")
        self.assertEqual(T.hget(h, "x-content-type-options"), "nosniff")
        self.assertTrue(T.hget(h, "cache-control").startswith("private"))
        self.assertTrue(body.startswith(JPEG))
        from PIL import Image
        import io
        with Image.open(io.BytesIO(body)) as im:
            self.assertLessEqual(max(im.size), D.THUMB_PX)
            self.assertEqual(im.size, (240, 180))
        cached = list((self.env.state / "data_thumbs").rglob("*.jpg"))
        self.assertTrue(cached)
        before = len(cached)
        self.assertEqual(self.thumb("/data/thumb/session/20261104/0.jpg")[2], body)        # served from the cache
        self.assertEqual(len(list((self.env.state / "data_thumbs").rglob("*.jpg"))), before)
        st, h, body = self.thumb("/data/thumb/images/pkg_ok/1.jpg")                       # a PNG source
        self.assertEqual((st, body[:3]), (200, JPEG))

    def test_version_in_url_and_cache_header(self):
        d = self.api("/data/api/session/20261104")
        url = d["thumbs"][0]["url"]
        self.assertRegex(url, r"\.jpg\?v=[0-9a-f]{12}$")
        st, h, _ = self.get(url)
        self.assertEqual((st, T.hget(h, "cache-control")), (200, "private, max-age=86400"))
        st, h, _ = self.get(url.split("?")[0] + "?v=stale")
        self.assertEqual((st, T.hget(h, "cache-control")), (200, "private, max-age=60"))

    def test_16_bit_input(self):
        from PIL import Image
        f = self.proc / "20261104" / D.PROCESSED / "3a. Individual tiles" / "grid2_99_sixteen.png"
        im = Image.new("I;16", (400, 300), 1000)
        im.paste(30000, (100, 100, 300, 200))
        im.save(f)
        self.addCleanup(D._files_cache.clear)
        self.addCleanup(f.unlink)
        D._files_cache.clear()
        idx = [p.name for p in D.session_images("20261104")].index("grid2_99_sixteen.png")
        st, h, body = self.get(f"/data/thumb/session/20261104/{idx}.jpg")
        self.assertEqual((st, body[:3]), (200, JPEG))
        import io
        from PIL import Image as I
        with I.open(io.BytesIO(body)) as out:
            self.assertEqual(out.mode, "L")
            self.assertGreater(out.getextrema()[1] - out.getextrema()[0], 100)       # stretched, not black

    def test_missing_processing_root(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {"CRYOEV_PROCESSING_ROOT": str(self.env.tmp / "no_such_root")}):
            D._files_cache.clear()
            t = main_html(self.page("/data/session/20261104"))
            self.assertIn("No images found", t)
            self.assertEqual(self.get("/data/thumb/session/20261104/0.jpg")[0], 404)
            self.assertEqual(self.api("/data/api/session/20261104")["n_images"], 0)
        D._files_cache.clear()

    def test_text_check_before_disk(self):
        touched = []
        real = Path.is_dir
        from unittest import mock
        with mock.patch.object(Path, "is_dir", lambda self_: (touched.append(str(self_)), real(self_))[1]):
            D._files_cache.clear()
            self.assertEqual(D.scan_images(self.outside / "pkg_outside" / "images", self.lap / "review"), [])
            self.assertEqual(D.scan_images(Path("//no-such-host/share/x"), self.lap / "review"), [])
        self.assertEqual(touched, [])                                    # refused as text, nothing was opened

    def test_source_is_untouched(self):
        f = self.proc / "20261104" / D.PROCESSED / "3a. Individual tiles" / "grid1_00_r0c0_t00.jpg"
        before = (T.md5(f.read_bytes()), f.stat().st_mtime_ns)
        self.thumb("/data/thumb/session/20261104/0.jpg")
        self.assertEqual((T.md5(f.read_bytes()), f.stat().st_mtime_ns), before)

    def test_out_of_range_and_bad_parts(self):
        for p in ("/data/thumb/session/20261104/16.jpg", "/data/thumb/session/20261104/99999.jpg",
                  "/data/thumb/session/20261104/abc.jpg", "/data/thumb/session/20261104/-1.jpg",
                  "/data/thumb/session/20261104/1.png", "/data/thumb/model/v9_prod_20261101/0.jpg",
                  "/data/thumb/nope/20261104/0.jpg", "/data/thumb/session/99999999/0.jpg",
                  "/data/thumb/session/20261104/00000000000.jpg"):
            with self.subTest(p=p):
                self.assertEqual(self.thumb(p)[0], 404)

    def test_outside_roots_refused(self):
        # a package whose registry path points outside the known roots, with real images there: nothing is served
        self.assertTrue((self.outside / "pkg_outside" / "images" / "o.png").is_file())     # control: the file exists
        t = main_html(self.page("/data/images/pkg_outside"))
        self.assertEqual(t.count('class="cj-data-img"'), 0)
        self.assertEqual(self.thumb("/data/thumb/images/pkg_outside/0.jpg")[0], 404)
        # data.js naming files above / beside the package: only the in-package image is listed
        d = self.api("/data/api/images/pkg_escape")
        self.assertEqual(d["n_images"], 1)
        self.assertEqual(self.thumb("/data/thumb/images/pkg_escape/0.jpg")[0], 200)        # control: the good one works
        for i in range(1, 6):
            self.assertEqual(self.thumb(f"/data/thumb/images/pkg_escape/{i}.jpg")[0], 404)

    def test_traversal_attempts(self):
        paths = ["/data/thumb/session/%2e%2e/%2e%2e/x/0.jpg", "/data/thumb/session/..%2f..%2fx/0.jpg",
                 "/data/thumb/session/20261104%2f..%2f..%2fsecret/0.jpg", "/data/thumb/images/..%5c..%5cx/0.jpg",
                 "/data/thumb/images/pkg_escape/..%2f..%2fsecret_in_laptop/0.jpg", "/data/thumb/images/%2e%2e/0.jpg",
                 "/data/thumb/images/pkg_ok/0.jpg/../../../x.jpg", "/data/thumb/images/pkg_ok/0.jpg%00.png",
                 "/data/thumb/images/C:%5cWindows%5cwin.ini/0.jpg", "/data/thumb/images/%2fetc%2fpasswd/0.jpg",
                 "/data/thumb/session/20261104/0.jpg?path=../../x", "/data/thumb/../../etc/passwd"]
        for p in paths:
            with self.subTest(p=p):
                st, _, body = self.thumb(p)
                if "?path=" in p:
                    self.assertEqual(st, 200)                      # the query is ignored: same image as /0.jpg
                    self.assertTrue(body.startswith(JPEG))
                else:
                    self.assertIn(st, (400, 404))
                    self.assertFalse(body.startswith(JPEG))
        cookie = f"{A.COOKIE}={self.tok['admin']}".encode()
        for raw_path in (b"GET /data/thumb/images/x\\..\\y/0.jpg HTTP/1.1",
                         b"GET /data/thumb/images/x\x00/0.jpg HTTP/1.1"):
            resp = T.raw(self.p.port, raw_path + b"\r\nHost: x\r\nCookie: " + cookie + b"\r\n\r\n")
            self.assertNotEqual(T.raw_status(resp), 200)

    def test_link_out_of_roots_refused(self):
        link = self.proc / "20261104" / D.PROCESSED / "3a. Individual tiles" / "grid1_zz_link.jpg"
        try:
            os.symlink(self.outside / "secret.png", link)
        except (OSError, NotImplementedError):
            self.skipTest("cannot create symlinks here")
        self.addCleanup(link.unlink)
        D._files_cache.clear()
        d = self.api("/data/api/session/20261104")
        self.assertEqual(d["n_images"], 16)                       # the link is not listed

    def test_roots_helper(self):
        roots = D.known_roots()
        self.assertEqual(len(roots), 2)
        self.assertTrue(D.inside(self.lap / "review" / "pkg_ok" / "data.js", roots))
        self.assertFalse(D.inside(self.outside / "secret.png", roots))
        self.assertFalse(D.inside(self.lap / ".." / "outside" / "secret.png", roots))
        self.assertFalse(D.inside(self.lap.parent / "CryoAI-laptop-evil" / "x.png", roots))   # prefix is not a parent


class Cache(unittest.TestCase):
    def test_one_read_per_ttl(self):
        tmp = Path(tempfile.mkdtemp(prefix="cj_data_reg_"))
        try:
            write_registry(tmp, tmp / "o")
            a, e1 = D.read_registry(tmp, now=100.0)
            self.assertIsNone(e1)
            (tmp / "model.jsonl").write_text("", encoding="utf-8")
            b, _ = D.read_registry(tmp, now=105.0)
            self.assertIs(a, b)                                    # served from the cache
            c, _ = D.read_registry(tmp, now=111.0)                 # past 10 s: re-read
            self.assertIsNot(a, c)
            self.assertEqual(c["model"], {})
        finally:
            D._registry_cache.clear()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_registry(self):
        S, err = D.read_registry(Path(tempfile.gettempdir()) / "cj_no_such_registry_dir")
        self.assertTrue(err)
        self.assertEqual(S["session"], {})
        D._registry_cache.clear()


class Browser(Base):
    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        super().setUpClass()
        cls.own_shots = not os.environ.get("CJ_SHOT_DIR")
        cls.shots = Path(os.environ.get("CJ_SHOT_DIR") or tempfile.mkdtemp(prefix="cj_shots_"))
        cls.shots.mkdir(parents=True, exist_ok=True)
        cls.pw = sync_playwright().start()
        cls.browser = launch(cls.pw)
        if cls.browser is None:
            cls.tearDownClass()
            raise unittest.SkipTest("no Edge / Chrome for Playwright")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "browser", None):
            cls.browser.close()
        if getattr(cls, "pw", None):
            cls.pw.stop()
        super().tearDownClass()
        if cls.own_shots:
            shutil.rmtree(cls.shots, ignore_errors=True)

    def session(self, width, scheme, key=None):
        ctx = self.browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
        origin = self.p.origin
        rec = {"requests": [], "foreign": [], "errors": [], "console": []}
        ctx.route("**/*", lambda r: (r.continue_() if r.request.url.startswith(origin + "/")
                                     else (rec["foreign"].append(r.request.url), r.abort())))
        page = ctx.new_page()
        page.on("request", lambda r: rec["requests"].append(r.url))
        page.on("pageerror", lambda e: rec["errors"].append(str(e)))
        page.on("console", lambda m: m.type == "error" and rec["console"].append(m.text))
        page.goto(f"{origin}/login?k={key or self.env.key}")
        self.assertEqual(page.url, origin + "/")
        return ctx, page, rec

    def check_clean(self, rec):
        self.assertEqual(rec["foreign"], [])
        self.assertTrue(rec["requests"])
        self.assertTrue(all(u.startswith(self.p.origin + "/") for u in rec["requests"]),
                        [u for u in rec["requests"] if not u.startswith(self.p.origin + "/")])
        self.assertEqual(rec["errors"], [])
        self.assertEqual(rec["console"], [])

    def test_pages_sizes_and_schemes(self):
        pages = [("overview", "/data/"), ("session", "/data/session/20261104"), ("grid", "/data/grid/2603"),
                 ("model", "/data/model/v9_prod_20261101"), ("search", "/data/search?q=example"),
                 ("images", "/data/images/pkg_ok")]
        with open(self.env.registry / "session.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"schema": "session/1", "id": "20261110", "logged": "2026-11-06T00:00:00",
                                 "status": "processed", "session_grids": {"1": "2603"}}) + "\n")
        self.addCleanup(lambda: (self.env.registry / "session.jsonl").write_text(
            "\n".join((self.env.registry / "session.jsonl").read_text(encoding="utf-8").splitlines()[:-1]) + "\n",
            encoding="utf-8"))
        D._registry_cache.clear()
        for scheme in ("light", "dark"):
            for width in (390, 1280):
                ctx, page, rec = self.session(width, scheme)
                try:
                    for name, path in pages:
                        with self.subTest(page=name, width=width, scheme=scheme):
                            page.goto(self.p.origin + path)
                            page.wait_for_load_state("networkidle")
                            page.wait_for_selector("#cj-bar")
                            sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
                            self.assertLessEqual(sw, iw, "horizontal scroll")
                            page.evaluate("Promise.all(Array.from(document.images).map(i => { i.loading = 'eager'; "
                                          "return i.complete ? 1 : "
                                          "new Promise(r => { i.onload = i.onerror = r; }); }))")
                            bad = page.evaluate("Array.from(document.images).filter(i => !i.naturalWidth).length")
                            self.assertEqual(bad, 0, "broken thumbnail")
                            bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
                            self.assertEqual(bg, "rgb(15, 21, 27)" if scheme == "dark" else "rgb(243, 245, 247)")
                            self.assertEqual(page.locator("script:not([src])").count(), 0)
                            page.screenshot(path=str(self.shots / f"data_{name}_{width}_{scheme}.png"), full_page=True)
                    self.check_clean(rec)
                finally:
                    ctx.close()

    def test_search_form_and_advanced_user(self):
        ctx, page, rec = self.session(390, "dark", key=self.adv_key)
        try:
            page.goto(self.p.origin + "/data/")
            page.fill("input[name=q]", "Example sample A")
            page.click("button[type=submit]")
            page.wait_for_load_state("networkidle")
            self.assertIn("/data/search?q=", page.url)
            self.assertIn("Grid 2602", page.text_content("main"))
            page.click("text=Grid 2602")
            page.wait_for_load_state("networkidle")
            self.assertIn("Example carbon grid", page.text_content("main"))
            page.goto(self.p.origin + "/data/images/ps_20261105_bbbbbb")
            self.assertIn("Not found", page.text_content("main"))
            self.check_clean({**rec, "console": [c for c in rec["console"] if "404" not in c]})
        finally:
            ctx.close()


LIN_BAD = '<img src=x onerror=alert(1)>'


def lineage_records() -> dict:
    """Chain rows added on top of the fixture registry: entity -> [record]."""
    def rec(entity, **kw):
        return {"schema": f"{entity}/1", "logged": "2026-11-05T12:00:00", **kw}
    return {
        "model": [rec("model", id="m_chain", run="r_chain", dataset="ds_chain", status="candidate", imgsz=1440,
                      classes=["EV"], weights="models/m_chain/best.pt"),
                  rec("model", id="m_orphan", status="candidate"),
                  rec("model", id="m_ghost", run="r_ghost", dataset="ds_ghost", status="candidate"),
                  rec("model", id="m_xss", run="r_xss", status="candidate"),
                  rec("model", id="m_mine", dataset="ds_mine", status="candidate"),
                  rec("model", id="m_theirs", dataset="ds_theirs", status="candidate"),
                  rec("model", id="m_loop", dataset="ds_loop", status="candidate")],
        "run": [rec("run", id="r_chain", dataset="ds_chain", status="finished",
                    config={"arch": "yolov8m-seg", "epochs": 300, "imgsz": 1440}, machine="ML desktop",
                    metrics={"mask_map50": 0.9}),
                rec("run", id="r_xss", dataset=LIN_BAD, status="finished")],
        "dataset": [rec("dataset", id="ds_chain", label_version="lv_chain", status="built",
                        counts={"train": {"images": 5}, "valid": {"images": 2}}, classes=["EV"]),
                    rec("dataset", id="ds_mine", source={"kind": "prediction_set", "id": "ps_20261105_aaaaaa"},
                        status="built"),
                    rec("dataset", id="ds_theirs", source={"kind": "prediction_set", "id": "ps_20261105_bbbbbb"},
                        status="built"),
                    rec("dataset", id="ds_loop", label_version="lv_loop", status="built")],
        "label_version": [rec("label_version", id="lv_chain", pkg="pkg_chain", base="pull_chain", status="active",
                              n_images_applied=2),
                          rec("label_version", id="lv_loop", base="ds_loop", status="active")],
        "review_package": [rec("review_package", id="pkg_chain", path="x/review/pkg_chain", kind="prediction",
                               owner="adv", prediction_set="ps_20261105_aaaaaa", n_images=2, status="in_review")],
        "pull": [rec("pull", id="pull_chain", created="2026-11-03T10:00", n_images=2, path="x/pull_chain")],
        "roboflow_image": [rec("roboflow_image", id="rf1", name="t1.jpg", first_seen_pull="pull_chain"),
                           rec("roboflow_image", id="rf2", name="t2.jpg", first_seen_pull="pull_chain")],
        "image": [rec("image", id="t1", rf_name="t1.jpg", session="20261104", session_grid="1", grid=None),
                  rec("image", id="t2", rf_name="t2.jpg", session="20261104", session_grid="2", grid=None)],
        "roboflow_export": [rec("roboflow_export", id="cyc a", derived_from="cyc b", classes=[]),
                            rec("roboflow_export", id="cyc b", derived_from="cyc a", classes=[]),
                            rec("roboflow_export", id="plain exp", classes=[])],
        "prediction_set": [
            rec("prediction_set", id="ps_20261105_aaaaaa", created="2026-11-05T10:00:00", owner="adv",
                model="v9_prod_20261101", path="p", source={"kind": "session", "session": "20261104"}, n_images=2,
                status="done", conf=0.25),
            rec("prediction_set", id="ps_20261105_bbbbbb", created="2026-11-05T11:00:00", owner="otherperson",
                model="v9_prod_20261101", path="p", source={"kind": "session", "session": "20261104"}, n_images=2,
                status="done", conf=0.25)],
        "grid": [rec("grid", id="2699", sample_label=LIN_BAD, grid_type="Gold grid")],
    }


def add_lineage_records(registry: Path):
    for entity, recs in lineage_records().items():
        with open(registry / f"{entity}.jsonl", "a", encoding="utf-8") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in recs))
    sess = {"schema": "session/1", "id": "20261111", "logged": "2026-11-05T12:00:00", "status": "processed",
            "session_grids": {"1": "2699"}}
    with open(registry / "session.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(sess) + "\n")
    D._registry_cache.clear()
    D._files_cache.clear()
    D._package_cache.clear()
    D._owner_cache.clear()


def walk(node):
    yield node
    for parent in node["parents"]:
        yield from walk(parent)


def by_type(tree, ntype):
    return [n for n in walk(tree["root"]) if n["type"] == ntype]


class LineageBase(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        add_lineage_records(cls.env.registry)


class Lineage(LineageBase):
    def tree(self, kind, eid, role="admin"):
        return self.api(f"/data/api/lineage/{kind}/{quote(eid, safe='')}", role)

    def test_full_chain_model_to_session(self):
        t = self.tree("model", "m_chain")
        root = t["root"]
        self.assertEqual((t["schema"], root["type"], root["id"]), ("cj_data/1", "model", "m_chain"))
        chain = [root]
        while chain[-1]["parents"]:
            chain.append(chain[-1]["parents"][0])
        self.assertEqual([n["type"] for n in chain[:4]], ["model", "run", "dataset", "label"])
        lv = chain[3]
        kinds = {p["type"]: p for p in lv["parents"]}
        self.assertEqual(set(kinds), {"pull", "images"})
        ps = kinds["images"]["parents"][0]                           # review package -> prediction set -> session
        self.assertEqual((ps["type"], ps["id"]), ("images", "ps_20261105_aaaaaa"))
        self.assertEqual([p["id"] for p in ps["parents"]], ["20261104"])
        grid = kinds["pull"]["parents"][0]                           # download -> grid -> session
        self.assertEqual((grid["type"], grid["id"]), ("grid", "2602"))
        self.assertIn("20261104", [p["id"] for p in grid["parents"]])
        session = next(n for n in walk(root) if n["type"] == "session" and n["id"] == "20261104")
        shown = {f[0]: f[1] for f in session["fields"]}
        self.assertEqual(shown["Instrument"], "Glacios")
        self.assertEqual(shown["Magnification"], "11,000× and 45,000×")
        self.assertEqual(shown["Pixel size"], "1.8 Å/px and 7.36 Å/px")
        self.assertIn("-3.5 µm", shown["Defocus"])
        self.assertIn("3.6 e/Å²", shown["Dose"])
        self.assertTrue(session["thumbs"] and len(session["thumbs"]["thumbs"]) == D.LINEAGE_THUMBS)
        self.assertEqual(t["summary"]["hidden"], 0)
        self.assertEqual(self.get(session["thumbs"]["thumbs"][0]["url"])[0], 200)
        page = main_html(self.page("/data/lineage/model/m_chain/"))
        for want in ("Lineage of m_chain", "Training run", "Dataset", "Label version", "Roboflow download", "Grid 2602",
                     "Session 20261104", "Glacios", "Made from"):
            self.assertIn(want, page)
        self.assertGreaterEqual(page.count('class="cj-data-img"'), 4)

    def test_every_kind_has_a_lineage_page_and_api(self):
        for kind, eid in (("session", "20261104"), ("grid", "2602"), ("images", "ps_20261105_aaaaaa"),
                          ("labels", "lv_chain"), ("dataset", "ds_chain"), ("model", "m_chain"),
                          ("images", "roboflow - 20261028 cryoai v5 S1")):
            with self.subTest(kind=kind):
                self.assertEqual(self.tree(kind, eid)["root"]["id"], eid)
                self.assertIn("Lineage of", self.page(f"/data/lineage/{kind}/{quote(eid, safe='')}/"))
        self.assertEqual(self.get("/data/lineage/model/m_chain")[0], 200)       # no trailing slash works too
        for bad in ("/data/lineage/model/nope/", "/data/lineage/bogus/x/", "/data/lineage/run/r_chain/"):
            self.assertEqual(self.get(bad)[0], 404, bad)
        self.assertEqual(self.get("/data/api/lineage/model/nope")[0], 404)
        self.assertEqual(self.get("/data/lineage/model/m_chain/", "basic")[0], 403)
        self.assertEqual(self.get("/data/api/lineage/model/m_chain", "basic")[0], 403)

    def test_link_on_every_catalog_item(self):
        for path in ("/data/list/sessions", "/data/list/grids", "/data/list/images", "/data/list/labels",
                     "/data/list/datasets", "/data/list/models", "/data/search?q=m_chain"):
            with self.subTest(path=path):
                t = main_html(self.page(path))
                n_items = t.count('class="cj-data-row"')
                self.assertGreater(n_items, 0)
                self.assertEqual(t.count(">Lineage</a>"), n_items)
        self.assertIn('href="/data/lineage/model/m_chain/"', self.page("/data/list/models"))
        self.assertIn('href="/data/lineage/model/m_chain/"', self.page("/data/model/m_chain"))
        for row in self.api("/data/api/list/models")["rows"]:
            self.assertEqual(row["lineage"], f"/data/lineage/model/{row['id']}/")

    def test_missing_parents_are_unknown(self):
        t = self.tree("model", "m_orphan")
        kinds = [(n["type"], n["status"]) for n in t["root"]["parents"]]
        self.assertEqual(kinds, [("run", "unknown"), ("dataset", "unknown")])
        self.assertEqual(t["summary"]["unknown"], 2)
        page = main_html(self.page("/data/lineage/model/m_orphan/"))
        self.assertIn("Unknown", page)
        self.assertIn("does not say which training run", page)
        # recorded but absent: the id is quoted as recorded, nothing is filled in
        g = self.tree("model", "m_ghost")
        self.assertEqual([(n["type"], n["status"], n.get("recorded")) for n in g["root"]["parents"]],
                         [("run", "unknown", "r_ghost"), ("dataset", "unknown", "ds_ghost")])
        self.assertTrue(all(n["id"] is None for n in g["root"]["parents"]))
        # a session has no parents and is not "unknown"
        s = self.tree("session", "20261104")
        self.assertEqual((s["root"]["parents"], s["summary"]["unknown"]), ([], 0))
        # an export made from an export the registry does not hold
        e = self.tree("images", "roboflow - 20261028 cryoai v5 S1")
        self.assertEqual([(n["status"], n.get("recorded")) for n in e["root"]["parents"]],
                         [("unknown", "roboflow - 20261028 cryoai v5")])
        # an export that records no source at all
        e = self.tree("images", "plain exp")
        self.assertEqual([(n["status"], n["id"]) for n in e["root"]["parents"]], [("unknown", None)])

    def test_cycle_terminates(self):
        for kind, eid in (("images", "cyc a"), ("model", "m_loop")):
            with self.subTest(root=eid):
                t = self.tree(kind, eid)
                loops = [n for n in walk(t["root"]) if n["status"] == "loop"]
                self.assertEqual(len(loops), 1)
                self.assertEqual(t["summary"]["loop"], 1)
                self.assertLessEqual(t["summary"]["steps"], 5)
                self.assertIn("links this step back", main_html(self.page(
                    f"/data/lineage/{kind}/{eid.replace(' ', '%20')}/")))

    def test_depth_and_size_caps(self):
        old = D.LINEAGE_MAX_DEPTH
        D.LINEAGE_MAX_DEPTH = 2
        try:
            t = self.tree("model", "m_chain")
        finally:
            D.LINEAGE_MAX_DEPTH = old
        cut = [n for n in walk(t["root"]) if n["status"] == "cut"]
        self.assertTrue(cut)
        self.assertEqual([(n["type"], n["id"], n["parents"]) for n in cut], [("dataset", "ds_chain", [])])
        self.assertEqual((len(list(walk(t["root"]))), t["summary"]["cut"]), (3, 1))

    def test_other_users_prediction_set_is_hidden(self):
        mine, theirs = "ps_20261105_aaaaaa", "ps_20261105_bbbbbb"
        # control: the owner and admin see the chain through their own set
        for role in ("advanced", "admin"):
            with self.subTest(role=role):
                t = self.tree("model", "m_mine", role)
                ps = by_type(t, "images")[0]
                self.assertEqual((ps["status"], ps["id"]), ("ok", mine))
                self.assertEqual(t["summary"]["hidden"], 0)
        self.assertEqual(by_type(self.tree("model", "m_mine", "admin"), "session")[0]["id"], "20261104")
        # someone else's set: admin sees it, an advanced user sees "not visible to you" and no id
        self.assertEqual(by_type(self.tree("model", "m_theirs", "admin"), "images")[0]["id"], theirs)
        for url, kind in ((f"/data/api/lineage/model/m_theirs", "api"), ("/data/lineage/model/m_theirs/", "page")):
            with self.subTest(view=kind):
                body = self.get(url, "advanced")[2].decode()
                self.assertNotIn(theirs, body)
                self.assertNotIn("otherperson", body)
                self.assertIn("Not visible to you", body)
        t = self.tree("model", "m_theirs", "advanced")
        hidden = [n for n in walk(t["root"]) if n["status"] == "hidden"]
        self.assertEqual(len(hidden), 1)
        self.assertEqual((hidden[0]["id"], hidden[0]["url"], hidden[0]["fields"], hidden[0]["parents"],
                          hidden[0]["thumbs"]), (None, None, [], [], None))
        self.assertEqual(t["summary"]["hidden"], 1)
        # the root itself being hidden is a plain 404, the same as an id that does not exist
        self.assertEqual(self.get(f"/data/api/lineage/images/{theirs}", "advanced")[0], 404)
        self.assertEqual(self.get(f"/data/lineage/images/{theirs}/", "advanced")[0], 404)
        self.assertEqual(self.get(f"/data/api/lineage/images/{theirs}", "admin")[0], 200)
        # unregistered prediction packages with no known owner fail closed
        self.assertEqual(self.get("/data/api/lineage/images/pred_ps_20261105_cccccc", "advanced")[0], 404)
        self.assertEqual(self.get("/data/api/lineage/images/pred_ps_20261105_cccccc", "admin")[0], 200)

    def test_hidden_package_under_a_label_version(self):
        # lv_owned's package belongs to someone else: the label version shows, the package does not
        t = self.tree("labels", "lv_owned", "advanced")
        self.assertEqual([n["status"] for n in t["root"]["parents"]], ["hidden"])
        self.assertNotIn("pkg_owned", json.dumps(t))
        self.assertEqual(t["root"]["thumbs"], None)
        ta = self.tree("labels", "lv_owned", "admin")
        self.assertEqual([n["id"] for n in ta["root"]["parents"]], ["pkg_owned"])

    def test_paths_for_admins_only(self):
        adm, adv = json.dumps(self.tree("model", "m_chain")), json.dumps(self.tree("model", "m_chain", "advanced"))
        self.assertIn("models/m_chain/best.pt", adm)
        self.assertNotIn("models/m_chain/best.pt", adv)
        self.assertIn("Cryo-image-processing/20261104", json.dumps(self.tree("session", "20261104")))
        self.assertNotIn("Cryo-image-processing/20261104", json.dumps(self.tree("session", "20261104", "advanced")))
        self.assertNotIn("x/review/pkg_chain", json.dumps(self.tree("images", "pkg_chain", "advanced")))

    def test_escaping(self):
        raw = LIN_BAD
        esc_text = "&lt;img src=x onerror=alert(1)&gt;"
        pages = (self.page("/data/lineage/model/m_xss/"), self.page("/data/lineage/grid/2699/"))
        for page in pages:
            t = main_html(page)
            self.assertNotIn(raw, t)
            self.assertIn(esc_text, t)
        self.assertIn("Recorded as " + esc_text, main_html(pages[0]))      # an unknown parent id is quoted, escaped
        self.assertIn("Sample", main_html(pages[1]))
        data = self.tree("grid", "2699")                                    # JSON carries the raw text as data
        self.assertEqual([f[1] for f in data["root"]["fields"] if f[0] == "Sample"], [raw])
        # an id with markup in the URL is escaped in the breadcrumb of the 404 and never echoed raw
        st, _, body = self.get("/data/lineage/model/%3Cb%3Ex%3C%2Fb%3E/")
        self.assertEqual(st, 404)
        self.assertNotIn(b"<b>x</b>", body)

    def test_reads_do_not_write_the_registry(self):
        before = {f.name: f.read_bytes() for f in self.env.registry.glob("*.jsonl")}
        self.page("/data/lineage/model/m_chain/")
        self.tree("images", "ps_20261105_aaaaaa")
        after = {f.name: f.read_bytes() for f in self.env.registry.glob("*.jsonl")}
        self.assertEqual(before, after)


class LineageBrowser(Browser):
    test_pages_sizes_and_schemes = None
    test_search_form_and_advanced_user = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        add_lineage_records(cls.env.registry)

    def test_lineage_pages_at_two_widths(self):
        paths = (("model", "/data/lineage/model/m_chain/"), ("orphan", "/data/lineage/model/m_orphan/"),
                 ("cycle", "/data/lineage/images/cyc%20a/"), ("session", "/data/lineage/session/20261104/"),
                 ("catalog", "/data/list/models"))
        for scheme in ("light", "dark"):
            for width in (390, 1280):
                ctx, page, rec = self.session(width, scheme)
                try:
                    for name, path in paths:
                        with self.subTest(page=name, width=width, scheme=scheme):
                            page.goto(self.p.origin + path)
                            page.wait_for_load_state("networkidle")
                            page.wait_for_selector("#cj-bar")
                            sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
                            self.assertLessEqual(sw, iw, "horizontal scroll")
                            page.evaluate("Promise.all(Array.from(document.images).map(i => { i.loading = 'eager'; "
                                          "return i.complete ? 1 : "
                                          "new Promise(r => { i.onload = i.onerror = r; }); }))")
                            self.assertEqual(page.evaluate("Array.from(document.images).filter(i => !i.naturalWidth)"
                                                           ".length"), 0, "broken thumbnail")
                            self.assertEqual(page.locator("script:not([src])").count(), 0)
                            if name == "model":
                                self.assertIn("Session 20261104", page.text_content("main"))
                                self.assertGreaterEqual(page.locator("main img.cj-data-img").count(), 4)
                            page.screenshot(path=str(self.shots / f"lineage_{name}_{width}_{scheme}.png"),
                                            full_page=True)
                    self.check_clean(rec)
                finally:
                    ctx.close()

    def test_catalog_link_leads_to_lineage(self):
        ctx, page, rec = self.session(390, "light", key=self.adv_key)
        try:
            page.goto(self.p.origin + "/data/list/models")
            page.click('a[href="/data/lineage/model/m_chain/"]')
            page.wait_for_load_state("networkidle")
            self.assertIn("Lineage of m_chain", page.text_content("main"))
            page.goto(self.p.origin + "/data/lineage/model/m_theirs/")
            self.assertIn("Not visible to you", page.text_content("main"))
            self.assertNotIn("bbbbbb", page.content())
            self.check_clean(rec)
        finally:
            ctx.close()


if __name__ == "__main__":
    unittest.main()
