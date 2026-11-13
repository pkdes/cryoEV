"""
Data module, round 9 (#14, #15, #16, attribution), on a temp registry with the SAME STRUCTURE as the live known answer
(dataset lv_20261111_1409_v5S1_evnonev_undet <- export "roboflow - 20261028 cryoai v5 S1"):

  - #15 known answer: the export's image list has 160 rows; the registry's image records put them in sessions
    20250730: 44, 20260126: 7, 20260420: 9, 20260506: 20, 20260512: 22, and 58 have no session ("Session unknown");
    sum 160. The dataset lists 179: 160 from the export plus 19 that are in no image record (the dataset note says
    why). Grids where known (20260420: 2602; 20260512: 2642 and 2643). Four older exports with no image list (counts
    {}) say "Image list not recorded".
  - #16 lanes: Imaging sessions | Labels: automatic | Labels: human | Labels: revised | Datasets | Models; Roboflow
    exports are human label sets "via Roboflow"; review packages sit in the revised lane (dashed in the browser);
    session -> export edges carry the image count; arrows are drawn; at 390 px the lanes stack.
  - #14: classes as a bulleted list; the class map in a collapsed <details>, one line per training class and
    "dropped: ..." for classes mapped to nothing.
  - attribution: who_label() is registry/actor.py's label ("claude for pujan").

Temp folders only (cjtest.Env for the served part); CRYOEV_* cleared in setUp and restored after.

  python -m unittest portal.tests.test_data_r9 -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_data_graph as TG  # noqa: E402
from portal.modules import data as D  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

DS = "lv_20261111_1409_v5S1_evnonev_undet"
EXP = "roboflow - 20261028 cryoai v5 S1"
LV, PKG, PKG_OPEN, MODEL = "lv_20261111_1409", "lab_compare_v5_cell_20261110", "audit_in_progress", "m_c1_20261111"
SESSIONS = {"20250730": 44, "20260126": 7, "20260420": 9, "20260506": 20, "20260512": 22}
GRIDS = {"20260420": {"2602": 9}, "20260512": {"2642": 16, "2643": 6}}
N_UNKNOWN, N_EXTRA = 58, 19
OLD_EXPORTS = ("roboflow - 20260529", "roboflow - 20260713 all layer label", "roboflow - 20260824 cryoai v3",
               "roboflow20260604")
NOTE = ("10 lv image(s) not in the export's split folders, left out (details.not_in_export); export-only: 19 export "
        "image(s) not in the lv, kept with the export's own labels (--export-only export-labels)")
CLASS_MAP = {"EV": "EV", "Low conf EV": "Undetermined", "Incomplete": "EV", "MV": "EV", "inner": "EV",
             "non-EV": "non-EV", "Overlapping": "EV", "items-nfaR": None, "VLDL": "non-EV"}
ENV_KEYS = ("CRYOEV_VIA", "CRYOEV_FOR", "CRYOEV_REGISTRY", "CRYOEV_CRYOAI_ROOT", "CRYOEV_LAPTOP_ROOT",
            "CRYOEV_PROCESSING_ROOT")


def rec(entity, rid, **kw):
    return {"schema": entity + "/1", "id": rid, "logged": "2026-11-11T10:00:00", **kw}


def fixture(root: Path) -> None:
    """Write <entity>.jsonl files and the two image lists under `root` (a registry folder)."""
    images, export_rows = [], []
    for sid, n in SESSIONS.items():
        grids = [g for g, k in GRIDS.get(sid, {}).items() for _ in range(k)]
        for i in range(n):
            stem = f"{sid}_tile{i:03d}_png"
            images.append(rec("image", stem, session=sid, grid=grids[i] if i < len(grids) else None, rf_id=None))
            export_rows.append(stem)
    for i in range(N_UNKNOWN):                          # pre-manifest images: an image record, no session
        stem = f"{i:04d}_Jul30_jpg"
        images.append(rec("image", stem, session=None, grid=None, rf_id=None))
        export_rows.append(stem)
    lists = root / "lists" / "exports"
    lists.mkdir(parents=True, exist_ok=True)
    (lists / "s1.tsv").write_text("".join(f"{s}\ttrain\n" for s in sorted(export_rows)), encoding="utf-8")
    ds_rows = sorted(export_rows + [f"g6_extra_{i:02d}_png" for i in range(N_EXTRA)])   # in no image record
    (root / "lists" / "ds.tsv").write_text("".join(f"{s}\ttrain\n" for s in ds_rows), encoding="utf-8")
    rows = {
        "image": images,
        "session": [rec("session", sid, date=f"{sid[:4]}-{sid[4:6]}-{sid[6:]}", date_basis="folder",
                        status="processed") for sid in SESSIONS],
        "grid": [rec("grid", g, sample="EV prep") for g in ("2602", "2642", "2643")],
        "roboflow_export": [rec("roboflow_export", EXP, path="x", counts={"train": {"images": 103}, "valid":
                                {"images": 23}, "test": {"images": 34}},
                                image_list={"file": "lists/exports/s1.tsv", "sha256": None, "n": 160})]
        + [rec("roboflow_export", e, path="x", counts={}) for e in OLD_EXPORTS],
        "review_package": [rec("review_package", PKG, path="r", kind="review", base=EXP, base_kind="export",
                               n_images=43, built="2026-11-10T10:00:00", status="applied", model="m_v5_20261029"),
                           rec("review_package", PKG_OPEN, path="r2", kind="review", base=LV, base_kind="pull",
                               n_images=39, built="2026-11-11T20:00:00", status="in_review",
                               model="m_v5_20261029")],
        "label_version": [rec("label_version", LV, status="created", pkg=PKG, n_images_applied=43,
                              review_scope="all_edits",
                              inputs=[{"id": EXP, "role": "base", "entity": "roboflow_export"},
                                      {"id": PKG, "role": "edits", "entity": "review_package"}])],
        "dataset": [rec("dataset", DS, created="2026-11-11T14:10:20", status="frozen",
                        source={"kind": "label_version", "id": LV}, label_version=LV, export=EXP,
                        classes=["EV", "non-EV", "Undetermined"], class_map=CLASS_MAP,
                        counts={"train": {"images": 114}, "valid": {"images": 26}, "test": {"images": 39}},
                        counts_trained={"train": 114, "valid": 26, "test": 39},
                        image_list={"file": "lists/ds.tsv", "sha256": None, "n": 179}, note=NOTE,
                        inputs=[{"id": LV, "role": "source", "entity": "label_version"},
                                {"id": EXP, "role": "images", "entity": "roboflow_export"}])],
        "model": [rec("model", "m_v5_20261029", run=None, weights="w.pt", imgsz=1440, status="production",
                      inputs=[{"id": EXP, "role": "train_data", "entity": "roboflow_export"}]),
                  rec("model", MODEL, run=None, weights="w.pt", imgsz=1440, status="candidate",
                      inputs=[{"id": DS, "role": "train_data", "entity": "dataset"}])],
    }
    root.mkdir(parents=True, exist_ok=True)
    for ent, recs in rows.items():
        (root / f"{ent}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    TG.clear_caches()


def walk(node):
    yield node
    for p in node["parents"]:
        yield from walk(p)


class CleanEnv:
    @classmethod
    def clear_env(cls):
        cls.saved = {k: os.environ.pop(k) for k in ENV_KEYS if k in os.environ}

    @classmethod
    def restore_env(cls):
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(cls.saved)


class Function(CleanEnv, TG.Roots, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clear_env()
        cls.tmp = Path(tempfile.mkdtemp(prefix="cj_r9_"))
        cls.set_roots(cls.tmp)
        cls.reg = cls.tmp / "registry"
        fixture(cls.reg)
        cls.records, err = D.read_registry(cls.reg)
        assert err is None, err

    @classmethod
    def tearDownClass(cls):
        cls.restore_roots()
        cls.restore_env()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        TG.clear_caches()

    def setUp(self):
        for k in ("CRYOEV_VIA", "CRYOEV_FOR"):          # never stamped from the test runner's own shell
            os.environ.pop(k, None)

    def tree(self, kind, eid):
        return D.lineage_tree(self.records, TG.ADMIN, kind, eid, self.reg)

    # ---------------------------------------------------------------- #15 known answer
    def test_export_list_reaches_sessions_with_counts(self):
        t = self.tree("dataset", DS)
        exp = next(n for n in t["root"]["parents"] if n["type"] == "export" and n["id"] == EXP)
        got = {n["id"]: n["images"] for n in exp["parents"] if n["type"] == "session" and n["status"] == "ok"}
        self.assertEqual(got, SESSIONS)
        unknown = [n for n in exp["parents"] if n["title"] == "Session unknown"]
        self.assertEqual([(n["status"], n["images"]) for n in unknown], [("unknown", N_UNKNOWN)])
        self.assertEqual(sum(got.values()) + unknown[0]["images"], 160)      # = the export's image list
        self.assertIn(("Image list", "160 images", None), [tuple(f) for f in exp["fields"]])
        self.assertTrue(exp["note"].startswith("102 of 160 images traced to sessions 20250730"), exp["note"])

    def test_dataset_explains_179_vs_160(self):
        root = self.tree("dataset", DS)["root"]
        self.assertEqual(root["note"], f"179 images in this dataset: 160 are in the image list of export {EXP}, "
                                       "19 are not (none of them is in the registry's image records). Its imaging "
                                       "sessions are traced through that export.")
        self.assertIn(NOTE, {f[1] for f in root["fields"] if f[0] == "Note"})
        # its sessions show once, under the export: not a second time straight under the dataset
        self.assertFalse([n for n in root["parents"] if n["type"] == "session"])

    def test_grids_where_known(self):
        exp = next(n for n in self.tree("images", EXP)["root"]["parents"] if n["id"] == "20260512")
        fields = {f[0]: f[1] for f in exp["fields"]}
        self.assertEqual(fields["Grids"], "Grid 2642 (16), Grid 2643 (6)")
        self.assertEqual(fields["Images in this list"], "22")
        s0730 = next(n for n in self.tree("images", EXP)["root"]["parents"] if n["id"] == "20250730")
        self.assertNotIn("Grids", {f[0] for f in s0730["fields"]})           # no grid recorded: none invented

    def test_old_exports_say_image_list_not_recorded(self):
        for e in OLD_EXPORTS:
            with self.subTest(export=e):
                root = self.tree("images", e)["root"]
                self.assertEqual([(n["title"], n["status"]) for n in root["parents"]],
                                 [("Image list not recorded", "unknown")])
                self.assertIn(("Image list", "not recorded", None), [tuple(f) for f in root["fields"]])
        # control: the export WITH a list never says it
        self.assertNotIn("Image list not recorded", json.dumps(self.tree("images", EXP)))

    def test_lineage_page_shows_the_numbers(self):
        html = D.render_lineage(self.tree("dataset", DS), None, D.graph_focus_key(self.records, "dataset", DS))
        for want in ("Session unknown", "58 images", "44 images", "22 images", "Grid 2642 (16), Grid 2643 (6)",
                     "179 images in this dataset"):
            self.assertIn(want, html)

    # ---------------------------------------------------------------- #16 lanes (graph data)
    def lab(self, focus=None):
        return D.lab_graph(self.records, TG.ADMIN, self.reg, focus)

    def test_lanes_and_session_edges(self):
        g = self.lab()
        self.assertEqual(g["cols"], ["Imaging sessions", "Labels: automatic", "Labels: human", "Labels: revised",
                                     "Datasets", "Models"])
        by = {(n["kind"], n["title"]): i for i, n in enumerate(g["nodes"])}
        nodes = g["nodes"]
        exp = by[("export", EXP)]
        roles = {nodes[a]["title"]: role for a, b, role in g["edges"] if b == exp and nodes[a]["kind"] == "session"}
        self.assertEqual({k: v.split(" · ")[0] for k, v in roles.items()},
                         {**{s: f"{n} images" for s, n in SESSIONS.items()}, "Session unknown": "58 images"})
        self.assertEqual(roles["20260512"], "22 images · grid 2642: 16, grid 2643: 6")
        unk = nodes[by[("session", "Session unknown")]]
        self.assertEqual((unk["col"], unk["url"], unk["sub"]), (0, None, "not in the image records"))
        self.assertEqual(nodes[by[("session", "20250730")]]["url"], "/data/lineage/session/20250730/")
        self.assertEqual((nodes[exp]["col"], nodes[exp]["maker"]), (2, "export · via Roboflow"))
        lv, pkg, pkg2 = (nodes[by[k]] for k in (("label version", LV), ("package", PKG), ("package", PKG_OPEN)))
        self.assertEqual((lv["col"], lv["maker"], pkg["col"], pkg2["col"]), (3, "labels · revised", 3, 3))
        self.assertIn("→ " + LV, pkg["sub"])                                # applied: names the label set it made
        self.assertEqual(nodes[by[("dataset", DS)]]["col"], 4)
        self.assertEqual(nodes[by[("model", MODEL)]]["col"], 5)
        self.assertIn((exp, by[("package", PKG)], "base"), [tuple(e) for e in g["edges"]])  # export -> review
        for e in OLD_EXPORTS:
            self.assertIn("image list not recorded", nodes[by[("export", e)]]["sub"])
        self.assertTrue(all(set(n) == set(D.GRAPH_NODE_KEYS) for n in nodes))

    def test_item_graph_of_the_dataset_reaches_sessions(self):
        g = D.item_graph(self.lab("dataset|" + DS))
        titles = {n["title"] for n in g["nodes"] if n["kind"] == "session"}
        self.assertEqual(titles, set(SESSIONS) | {"Session unknown"})

    # ---------------------------------------------------------------- #14
    def test_classes_list_and_collapsed_class_map(self):
        d = D.detail(self.records, "dataset", DS, TG.ADMIN)
        self.assertEqual({f[0]: f[1] for f in d["fields"]}["Classes"], ["EV", "non-EV", "Undetermined"])
        self.assertEqual(d["class_map"], {"lines": [["EV", ["EV", "Incomplete", "MV", "inner", "Overlapping"]],
                                                    ["non-EV", ["non-EV", "VLDL"]],
                                                    ["Undetermined", ["Low conf EV"]]],
                                          "dropped": ["items-nfaR"]})
        html = D.render_detail(d, None)
        self.assertIn('<ul class="cj-data-vlist"><li>EV</li><li>non-EV</li><li>Undetermined</li></ul>', html)
        self.assertIn('<details class="cj-card cj-data-cmap"><summary>Class mapping (from the source labels)'
                      '</summary>', html)                       # collapsed: no `open`
        self.assertIn("<li><b>EV</b> ← EV, Incomplete, MV, inner, Overlapping</li>", html)
        self.assertIn('<li class="cj-mut">dropped: items-nfaR</li>', html)
        self.assertNotIn("Merged classes", html)
        row = next(r for r in D.dataset_rows(self.records) if r["id"] == DS)
        self.assertIn("3 classes: EV, non-EV, Undetermined", row["sub"])

    def test_class_map_escaped(self):
        d = D.class_mapping({"classes": ["<b>x</b>"], "class_map": {"<i>y</i>": "<b>x</b>"}})
        html = D.render_class_map(d)
        self.assertNotIn("<i>y</i>", html)
        self.assertIn("&lt;i&gt;y&lt;/i&gt;", html)

    # ---------------------------------------------------------------- attribution
    def test_who_label(self):
        self.assertEqual(D.who_label("pujan", "claude"), "claude for pujan")
        self.assertEqual(D.who_label("pujan", None), "pujan")
        self.assertEqual(D.who_label(None, None), "")


class Served(CleanEnv, TG.Base):
    @classmethod
    def setUpClass(cls):
        cls.clear_env()
        saved = TG.write_registry
        TG.write_registry = lambda root, with_hidden_ps=True: fixture(root)
        try:
            super().setUpClass()
        finally:
            TG.write_registry = saved

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.restore_env()

    def test_lineage_page_and_api(self):
        st, _, body = self.get("/data/lineage/dataset/" + quote(DS, safe="") + "/")
        self.assertEqual(st, 200)
        page = body.decode()
        self.assertIn("Session unknown", page)
        self.assertIn("44 images", page)
        st, _, body = self.get("/data/api/lineage/graph?scope=item&focus=" + quote("dataset|" + DS, safe=""),
                               "advanced")
        g = json.loads(body)
        self.assertIn("Session unknown", {n["title"] for n in g["nodes"]})


@unittest.skipIf(sync_playwright is None, "playwright not installed")
class Browser(Served):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.pw = sync_playwright().start()
        cls.browser = TG.launch(cls.pw)
        if cls.browser is None:
            cls.pw.stop()
            super().tearDownClass()
            raise unittest.SkipTest("no Edge / Chrome for Playwright")

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        super().tearDownClass()

    def open_graph(self, width):
        ctx = self.browser.new_context(viewport={"width": width, "height": 900})
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{self.p.origin}/login?k={self.adv_key}")
        page.goto(self.p.origin + "/data/lineage/dataset/" + quote(DS, safe="") + "/")
        page.wait_for_load_state("networkidle")
        page.click('[data-view="graph"]')
        page.wait_for_selector(".cj-data-graph-node")
        page.wait_for_load_state("networkidle")
        if os.environ.get("CJ_SHOT_DIR"):
            page.screenshot(path=str(Path(os.environ["CJ_SHOT_DIR"]) / f"r9_lanes_{width}.png"), full_page=True)
        return ctx, page, errors

    def test_lanes_arrows_and_counts_wide(self):
        ctx, page, errors = self.open_graph(1280)
        try:
            heads = page.locator(".cj-data-graph-head").all_text_contents()
            self.assertEqual(heads[0], "Imaging sessions")
            self.assertNotIn("Roboflow labels", heads)
            self.assertIn("Labels: revised", heads)
            n_edges = page.locator("svg#cj-data-graph-svg g").count()
            self.assertGreater(n_edges, 6)
            self.assertEqual(page.locator("svg#cj-data-graph-svg path.cj-data-graph-a").count(), n_edges)  # arrows
            counts = page.locator("svg text.cj-data-graph-count").all_text_contents()
            self.assertIn("44 images", counts)
            self.assertIn("58 images", counts)
            dashed = page.eval_on_selector(".cj-data-graph-between", "e => getComputedStyle(e).borderTopStyle")
            self.assertEqual(dashed, "dashed")
            tops = page.eval_on_selector_all(".cj-data-graph-head", "es => es.map(e => e.offsetTop)")
            self.assertEqual(len(set(tops)), 1)                     # side by side: one row of lane headings
            self.assertEqual(errors, [])
        finally:
            ctx.close()

    def test_lanes_stack_on_a_phone(self):
        ctx, page, errors = self.open_graph(390)
        try:
            sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
            self.assertLessEqual(sw, iw)
            wrap = page.evaluate("(w => [w.scrollWidth, w.clientWidth])(document.getElementById('cj-data-graph-wrap'))")
            self.assertLessEqual(wrap[0], wrap[1], "stacked lanes need no sideways scroll")
            tops = page.eval_on_selector_all(".cj-data-graph-head", "es => es.map(e => e.offsetTop)")
            self.assertEqual(tops, sorted(tops))
            self.assertEqual(len(set(tops)), len(tops))             # one lane under the other
            n_edges = page.locator("svg#cj-data-graph-svg g").count()
            self.assertGreater(n_edges, 6)
            self.assertEqual(page.locator("svg#cj-data-graph-svg path.cj-data-graph-a").count(), n_edges)  # arrows
            counts = page.locator("svg text.cj-data-graph-count").all_text_contents()
            self.assertEqual(sorted(counts), sorted(["44", "7", "9", "20", "22", "58"]))   # beside each session
            self.assertEqual(errors, [])
        finally:
            ctx.close()


if __name__ == "__main__":
    unittest.main()
