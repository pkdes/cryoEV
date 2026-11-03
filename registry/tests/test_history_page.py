"""history_page.py: samples + a synthetic registry (repeated status lines, superseded label version, a
partial status line, a gap, an id with <script>), escaping, no external URLs, the output guard, and (when
Playwright + Edge/Chrome exist) the page itself in a browser. HISTORY_TEST_SHOTS=<folder> keeps a 390 px
and a 1280 px screenshot."""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REG))
import history_page as hp  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

SAMPLES = REG / "samples"
EVIL = "pkg_<script>alert(1)</script>&x"
SHOTS = os.environ.get("HISTORY_TEST_SHOTS")


def synthetic_lines():
    P, OLD, NEW = "roboflow_pull_20990101_0000", "lv_20990101_0900", "lv_20990102_0900"
    DS, RUN, MOD = "ds_test_S1", "2099-01-03_000000_y8m_1440_8b_300ep_Run_99", "v9_Test_Y8M_20990103"
    L = [
        {"schema": "pull/1", "id": P, "logged": "2099-01-01T00:00:00", "path": "p", "project": "x",
         "created": "2099-01-01T00:00:00", "n_images": 1, "coco_sha256": "0" * 64},
        {"schema": "label_version/1", "id": OLD, "logged": "2099-01-01T09:00:00", "base": P, "pkg": EVIL,
         "status": "created"},
        {"schema": "review_package/1", "id": EVIL, "logged": "2099-01-01T08:00:00", "path": "r", "kind": "review",
         "base": P, "base_kind": "pull", "n_images": 3, "built": "2099-01-01T08:00:00", "status": "built"},
        {"schema": "label_version/1", "id": OLD, "logged": "2099-01-01T10:00:00", "base": P, "pkg": EVIL,
         "status": "qc_passed"},
        {"schema": "label_version/1", "id": NEW, "logged": "2099-01-02T09:00:00", "base": P, "supersedes": OLD,
         "status": "created"},
        {"schema": "label_version/1", "id": NEW, "logged": "2099-01-02T10:00:00", "base": P, "supersedes": OLD,
         "status": "qc_passed"},
        {"schema": "label_version/1", "id": NEW, "logged": "2099-01-02T11:00:00", "base": P, "supersedes": OLD,
         "status": "pushed", "pushed_to": "rf"},
        {"schema": "review_package/1", "id": EVIL, "logged": "2099-01-01T12:00:00", "path": "r", "kind": "review",
         "base": P, "base_kind": "pull", "n_images": 3, "built": "2099-01-01T08:00:00", "status": "applied"},
        {"schema": "dataset/1", "id": DS, "logged": "2099-01-02T12:00:00", "created": "2099-01-02T12:00:00",
         "source": {"kind": "label_version", "id": NEW}, "split": {}, "counts": {}, "classes": [],
         "images_sha256": None, "images_sha256_gap": "split list not local", "status": "frozen"},
        {"schema": "run/1", "id": RUN, "logged": "2099-01-03T00:00:00", "dataset": DS, "job_bundle": None,
         "job_bundle_gap": "pre-bundle run", "config": {}, "status": "finished"},
        {"schema": "model/1", "id": MOD, "logged": "2099-01-03T01:00:00", "run": RUN, "weights": "w.pt",
         "imgsz": 1440, "status": "candidate"},
        {"schema": "model/1", "id": MOD, "logged": "2099-01-04T01:00:00", "run": RUN, "weights": "w.pt",
         "imgsz": 1440, "status": "production"},
        {"schema": "inference_check/1", "id": "ic_1", "batch_id": "ic_1", "created": "2099-01-04T02:00:00",
         "session": "20990104", "model": MOD, "n": 5, "status": "staged", "logged": "2099-01-04T02:00:00"},
        {"schema": "inference_check/1", "id": "ic_1", "logged": "2099-01-05T00:00:00", "status": "scored"},
        {"schema": "image/1", "id": "20990104__tile_0001_png", "logged": "2099-01-04T00:00:00", "session": "20990104",
         "grid": None, "rf_id": None, "copy_of": "20990104__tile_0000_png"},
        {"schema": "label_change/1", "label_version": NEW, "image": "x"},
    ]
    return L, dict(P=P, OLD=OLD, NEW=NEW, DS=DS, RUN=RUN, MOD=MOD)


def find_node(sections, key, path=()):
    for sec in sections:
        for n in sec["nodes"]:
            r = _find(n, key, (sec["title"],))
            if r:
                return r
    return None


def _find(n, key, path):
    path = path + (n["k"],)
    if n["k"] == key:
        return path
    for c in n["c"]:
        r = _find(c, key, path)
        if r:
            return r
    return None


class HistoryPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        lines, cls.ids = synthetic_lines()
        cls.syn = t / "registry.jsonl"
        cls.syn.write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json\n", encoding="utf-8")
        cls.out_s = t / "samples.html"
        cls.out_x = t / "synthetic.html"
        cls.m_s = hp.build([SAMPLES], cls.out_s)
        cls.m_x = hp.build([cls.syn, SAMPLES], cls.out_x)
        cls.html_s = cls.out_s.read_text(encoding="utf-8")
        cls.html_x = cls.out_x.read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_samples_key_text(self):
        for f in SAMPLES.glob("*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if "id" in rec:
                    self.assertIn(hp.safe_json(rec["id"])[1:-1], self.html_s, rec["id"])
        for text in ("Registry history", "<h2>Queue</h2>", "History tree", "Search id", "prefers-color-scheme"):
            self.assertIn(text, self.html_s)
        self.assertNotIn("status board", self.html_s)
        q = {L["type"]: L for L in self.m_s["queue"]}               # samples: 1 revise in_progress, 1 annotate blocked
        self.assertEqual([L["title"] for L in self.m_s["queue"]], ["Annotate", "Revise", "Train", "Check"])
        self.assertEqual(q["revise"]["open"], ["job|job_20261102_prelipo_v5f5_review_v2"])
        self.assertEqual(q["annotate"]["open"], ["job|job_20261102_annotate_20261027_blind"])
        card = self.m_s["recs"]["job|job_20261102_annotate_20261027_blind"]["job"]
        self.assertEqual((card["next"], card["blocked"]), ("tiles not uploaded yet", "tiles not uploaded yet"))
        self.assertEqual(self.m_s["recs"]["job|job_20261102_prelipo_v5f5_review_v2"]["job"]["next"],
                         "Review in app (port 8766)")
        self.assertEqual(self.m_s["stats"]["label_change_rows"], 1)
        self.assertEqual(len(self.m_s["recs"]), 15)  # 13 entity samples + 2 job samples (job/1)
        # samples: pull -> lv; export -> dataset -> run -> model -> inference check; package under model
        S = self.m_s["sections"]
        self.assertEqual(find_node(S, "model|v5_evnonev_y8m_20261029")[-3:],
                         ("dataset|roboflow_20261028_cryoai_v5_S1_evnonev",
                          "run|2026-10-29_054810_y8m_1440_8b_300ep_Run_15", "model|v5_evnonev_y8m_20261029"))
        self.assertEqual(find_node(S, "label_version|lv_20261030_0846")[:2], ("Roboflow pulls", "pull|roboflow_pull_20261029_1811"))
        self.assertIn("model|v5_evnonev_y8m_20261029",
                      find_node(S, "inference_check|20261027_v5_evnonev_y8m_20261029_20261030-090300"))

    def test_synthetic_history_and_tree(self):
        i, R, S = self.ids, self.m_x["recs"], self.m_x["sections"]
        old, new = R["label_version|" + i["OLD"]], R["label_version|" + i["NEW"]]
        self.assertTrue(old["dim"])                     # derived from NEW.supersedes, though its status is qc_passed
        self.assertEqual(old["supby"], [i["NEW"]])
        self.assertEqual(new["sup"], i["OLD"])
        self.assertEqual([h["status"] for h in new["hist"]], ["created", "qc_passed", "pushed"])
        self.assertEqual(new["s"], "pushed")
        self.assertEqual(new["lane"], "pushed")
        pkg = R["review_package|" + EVIL]
        self.assertEqual(pkg["s"], "applied")           # last line in file order, though its `logged` is earlier
        self.assertEqual(find_node(S, "review_package|" + EVIL)[-2:], ("label_version|" + i["OLD"], "review_package|" + EVIL))
        self.assertEqual(find_node(S, "model|" + i["MOD"])[1:],
                         ("pull|" + i["P"], "label_version|" + i["NEW"], "dataset|" + i["DS"], "run|" + i["RUN"], "model|" + i["MOD"]))
        self.assertEqual(self.m_x["production"], ["model|" + i["MOD"]])
        ic = R["inference_check|ic_1"]                  # partial status line: keeps its model link, takes the new status
        self.assertEqual((ic["s"], ic["lane"], ic["raw"]["model"]), ("scored", "scored", i["MOD"]))
        self.assertEqual(find_node(S, "inference_check|ic_1")[-2], "model|" + i["MOD"])
        self.assertEqual(R["dataset|" + i["DS"]]["gaps"], {"images_sha256_gap": "split list not local"})
        self.assertEqual(R["image|20990104__tile_0001_png"]["copy_of"], "20990104__tile_0000_png")
        self.assertEqual(self.m_x["stats"]["bad_json"], 1)
        self.assertIn(i["MOD"], self.html_x)            # uppercase id kept as is

    def test_escaping(self):
        self.assertNotIn("<script>alert", self.html_x)
        self.assertNotIn("alert(1)</script>", self.html_x)
        self.assertEqual(self.html_x.count("<script"), 2)   # the JSON data block and the page script
        self.assertEqual(self.html_x.count("</script>"), 2)
        self.assertIn("\\u003cscript\\u003ealert(1)\\u003c/script\\u003e\\u0026x", self.html_x)
        self.assertNotIn("innerHTML", self.html_x)
        self.assertNotIn("insertAdjacentHTML", self.html_x)
        self.assertNotIn("document.write", self.html_x)

    def test_no_external_url(self):
        for html in (self.html_s, self.html_x):
            self.assertIsNone(re.search(r"(?i)https?://|//cdn|\bsrc\s*=|<link\b|@import|url\(", html))

    def test_output_guard(self):
        for bad in (Path.home() / "Box" / "history_test.html",
                    Path.home() / "Desktop" / "Cryo-image-processing" / "history_test.html"):
            with self.assertRaises(SystemExit):
                hp.build([SAMPLES], bad)
            self.assertFalse(bad.exists())
        with self.assertRaises(SystemExit):
            hp.build([self.syn], self.syn)               # never overwrite the input


def graph_index(m):
    """-> (node by key, set of (parent key, child key, role))."""
    G = m["graph"]
    nodes = {n[0]: n for n in G["nodes"]}
    edges = {(G["nodes"][a][0], G["nodes"][b][0], role) for a, b, role in G["edges"]}
    return nodes, edges


def lookup_refs(m, name):
    """-> {entity key: note or None} for one image in the lookup index (None when the image is not indexed)."""
    L = m["lk"]
    for n, refs in L["i"]:
        if n == name:
            return {L["e"][r[0]] if isinstance(r, list) else L["e"][r]: (r[1] if isinstance(r, list) else None)
                    for r in refs}
    return None


class GraphLookup(unittest.TestCase):
    """A folder registry with lists/, an lv change log next to its version.json, a package data.js and a job list."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        reg = t / "registry"
        (reg / "lists").mkdir(parents=True)
        lv_dir, pkg_dir = t / "labels" / "lv_20990102_0900", t / "review" / "pkg_new"
        lv_dir.mkdir(parents=True)
        pkg_dir.mkdir(parents=True)
        (lv_dir / "version.json").write_text(json.dumps({"id": "lv_20990102_0900", "n_images_applied": 2}), encoding="utf-8")
        (lv_dir / "changes.jsonl").write_text("".join(json.dumps({"schema": "label_change/1", "label_version": "lv_20990102_0900",
                                                                  "image": im}) + "\n" for im in ("img_a_png", "img_a_png", "img_b_png")),
                                              encoding="utf-8")
        (pkg_dir / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(
            {"id": "pkg_new", "images": [{"file": "img_a.png"}, {"file": "img_c.png"}]}) + ";\n", encoding="utf-8")
        (reg / "lists" / "ds_1.tsv").write_text("img_a_png\ttrain\nimg_b_png\tvalid\nimg_ghost_png\ttest\n", encoding="utf-8")
        (reg / "lists" / "job_20990103_annotate_b.txt").write_text("img_b_png\n", encoding="utf-8")
        recs = {
            "pull": [{"schema": "pull/1", "id": "roboflow_pull_20990101_0000"}],
            "roboflow_export": [{"schema": "roboflow_export/1", "id": "roboflow - 20990101 x"}],
            "label_version": [
                {"schema": "label_version/1", "id": "lv_20990101_0900", "status": "created",
                 "inputs": [{"id": "roboflow_pull_20990101_0000", "role": "base", "entity": "pull"}]},
                {"schema": "label_version/1", "id": "lv_20990102_0900", "status": "created", "supersedes": "lv_20990101_0900",
                 "owner": str(lv_dir / "version.json"),
                 "inputs": [{"id": "roboflow_pull_20990101_0000", "role": "base", "entity": "pull"},
                            {"id": "pkg_new", "role": "edits", "entity": "review_package"}]}],
            "label_change": [{"schema": "label_change/1", "label_version": "lv_20990101_0900", "image": "img_b_png"}],
            "review_package": [
                {"schema": "review_package/1", "id": "pkg_old", "model": "v9_m_20990101", "status": "superseded"},
                {"schema": "review_package/1", "id": "pkg_old_part1", "parent": "pkg_old", "model": "v9_m_20990101",
                 "status": "superseded"},
                {"schema": "review_package/1", "id": "pkg_new", "model": "v9_m_20990101", "path": str(pkg_dir),
                 "status": "applied"}],
            "dataset": [{"schema": "dataset/1", "id": "ds_1", "status": "trained",
                         "inputs": [{"id": "roboflow - 20990101 x", "role": "source", "entity": "roboflow_export"}],
                         "image_list": {"file": "lists/ds_1.tsv", "sha256": None},
                         "counts_trained": {"train": 1, "valid": 1, "test": 1}}],
            "model": [{"schema": "model/1", "id": "v9_m_20990101", "status": "production",
                       "inputs": [{"id": "ds_1", "role": "train_data", "entity": "dataset"}]}],
            "job": [{"schema": "job/1", "id": "job_20990102_revise_new", "type": "revise", "status": "in_progress",
                     "package": "pkg_new", "outputs": ["lv_20990102_0900"],
                     "inputs": [{"id": "roboflow_pull_20990101_0000", "role": "base"},
                                {"id": "v9_m_20990101", "role": "predictions"}]},
                    {"schema": "job/1", "id": "job_20990103_annotate_b", "type": "annotate", "status": "cancelled",
                     "image_list": {"file": "lists/job_20990103_annotate_b.txt", "sha256": None}, "inputs": []}],
            "image": [{"schema": "image/1", "id": "img_a_png", "session": "20990101", "grid": "9901"},
                      {"schema": "image/1", "id": "img_b_png", "session": "20990101", "grid": None},
                      {"schema": "image/1", "id": "img_c_png", "session": "20990102", "grid": None}],
        }
        for ent, lines in recs.items():
            (reg / f"{ent}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
        cls.m = hp.build([reg], t / "out" / "history.html")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_graph(self):
        nodes, edges = graph_index(self.m)
        self.assertEqual(self.m["graph"]["cols"][0], "Roboflow labels")
        for e in (("roboflow_export|roboflow - 20990101 x", "dataset|ds_1", "source"),
                  ("dataset|ds_1", "model|v9_m_20990101", "train_data"),
                  ("model|v9_m_20990101", "review_package|pkg_new", "predictions"),
                  ("review_package|pkg_new", "label_version|lv_20990102_0900", "edits"),        # back-edge
                  ("model|v9_m_20990101", "job|job_20990102_revise_new", "predictions"),
                  ("job|job_20990102_revise_new", "label_version|lv_20990102_0900", "output"),  # back-edge
                  ("pull|roboflow_pull_20990101_0000", "label_version|lv_20990102_0900", "base")):
            self.assertIn(e, edges)
        self.assertFalse(any(p == "pull|roboflow_pull_20990101_0000" and c.startswith("job|") for p, c, _ in edges))
        self.assertFalse(any(c == "review_package|pkg_old_part1" for _, c, _ in edges))   # parts sit under their package
        hidden = {k for k, n in nodes.items() if n[2]}
        self.assertEqual(hidden, {"label_version|lv_20990101_0900", "review_package|pkg_old",
                                  "review_package|pkg_old_part1", "job|job_20990103_annotate_b"})
        keys = [n[0] for n in self.m["graph"]["nodes"]]
        self.assertEqual(nodes["review_package|pkg_old_part1"][3], keys.index("review_package|pkg_old"))
        self.assertTrue(nodes["dataset|ds_1"][4].startswith("1/1/1 train/valid/test"))
        self.assertTrue(self.m["recs"]["model|v9_m_20990101"]["prod"])

    def test_lookup(self):
        a = lookup_refs(self.m, "img_a_png")
        self.assertEqual(a, {"dataset|ds_1": "train", "label_version|lv_20990102_0900": 2,
                             "review_package|pkg_new": None, "job|job_20990102_revise_new": None})
        b = lookup_refs(self.m, "img_b_png")      # lv_20990101_0900: registry label_change rows (it has no change log)
        self.assertEqual(b, {"dataset|ds_1": "valid", "label_version|lv_20990102_0900": 1,
                             "label_version|lv_20990101_0900": 1, "job|job_20990103_annotate_b": None})
        self.assertEqual(lookup_refs(self.m, "img_c_png"), {"review_package|pkg_new": None,
                                                            "job|job_20990102_revise_new": None})
        self.assertIsNone(lookup_refs(self.m, "img_ghost_png"))     # in a list, not in the registry: counted only
        self.assertEqual(self.m["lk"]["hidden"], 1)
        self.assertNotIn("img_ghost_png", json.dumps(self.m["lk"]))
        self.assertEqual(self.m["lk"]["sources"]["change_log"], ["lv_20990102_0900"])
        self.assertEqual(self.m["lk"]["sources"]["no_image_list"], ["pkg_old", "pkg_old_part1"])


REAL = hp.config.registry_root(required=False)


@unittest.skipUnless(REAL and (REAL / "model.jsonl").is_file(), "real registry not found (CryoAI-laptop/registry)")
class RealRegistryKnownAnswers(unittest.TestCase):
    """Known answers on the real registry (read-only; the page goes to a temp folder)."""
    V5, DS, EXP = "model|v5_evnonev_y8m_20261029", "dataset|roboflow_20261028_cryoai_v5_S1_evnonev", \
        "roboflow_export|roboflow - 20261028 cryoai v5 S1"
    PKG2, JOB2, LV = "review_package|prelipo_v5F5_review_v2", "job|job_20261029_prelipo_v5f5_review_v2", \
        "label_version|lv_20261030_0909"

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.m = hp.build([REAL], Path(cls.tmp.name) / "history.html")
        cls.nodes, cls.edges = graph_index(cls.m)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_v5_chain(self):
        self.assertIn((self.EXP, self.DS, "source"), self.edges)
        self.assertIn((self.DS, self.V5, "train_data"), self.edges)
        self.assertTrue(self.nodes[self.DS][4].startswith("114/26/39"))
        self.assertTrue(self.m["recs"][self.V5]["prod"])
        self.assertEqual([self.nodes[k][2] for k in (self.EXP, self.DS, self.V5)], [0, 0, 0])

    def test_revise_back_edge(self):
        for e in ((self.V5, self.PKG2, "predictions"), (self.PKG2, self.LV, "edits"),
                  (self.V5, self.JOB2, "predictions"), (self.JOB2, self.LV, "output")):
            self.assertIn(e, self.edges)
        col = {k: n[1] for k, n in self.nodes.items()}
        self.assertGreater(col[self.PKG2], col[self.LV])           # drawn right to left: a back-edge

    def test_history_adds_superseded_v1_packages(self):
        hidden = {k for k, n in self.nodes.items() if n[2] and k.startswith("review_package|")}
        v1 = {"review_package|prelipo_v5F5_review"} | {f"review_package|prelipo_v5F5_review_part{i}" for i in range(1, 5)}
        self.assertEqual(hidden, v1)
        self.assertEqual(self.nodes[self.PKG2][2], 0)

    def test_lookup(self):
        rows = (REAL / "lists" / "roboflow_20261028_cryoai_v5_S1_evnonev.tsv").read_text(encoding="utf-8").splitlines()
        stem, split = rows[0].split("\t")[:2]
        self.assertEqual(lookup_refs(self.m, stem)[self.DS], split)
        changed = [n for n, _ in self.m["lk"]["i"] if self.LV in (lookup_refs(self.m, n) or {})]
        self.assertEqual(len(changed), 28)                          # version.json n_images_applied
        self.assertIsInstance(lookup_refs(self.m, changed[0])[self.LV], int)
        fake = "zz_no_such_image_20991231_png"                      # control: a made-up name
        self.assertIsNone(lookup_refs(self.m, fake))
        self.assertFalse(any("no_such_image" in n for n, _ in self.m["lk"]["i"]))


def launch(p):
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:
            continue
    return None


class Browser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        cls.pw = sync_playwright().start()
        cls.browser = launch(cls.pw)
        if cls.browser is None:
            cls.pw.stop()
            raise unittest.SkipTest("no Edge / Chrome for Playwright")
        cls.tmp = tempfile.TemporaryDirectory()
        lines, cls.ids = synthetic_lines()
        syn = Path(cls.tmp.name) / "registry.jsonl"
        syn.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
        cls.page_path = Path(cls.tmp.name) / "history.html"
        hp.build([syn, SAMPLES], cls.page_path)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.tmp.cleanup()

    def open(self, width):
        ctx = self.browser.new_context(viewport={"width": width, "height": 900})
        ctx.route("**/*", lambda route: route.continue_() if route.request.url.startswith("file:") else route.abort())
        page = ctx.new_page()
        errors, requests, dialogs = [], [], []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: requests.append(r.url))
        page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
        page.goto(self.page_path.as_uri())
        page.wait_for_selector("#tree .rec")
        return ctx, page, errors, requests, dialogs

    def test_phone_and_desktop(self):
        for width in (390, 1280):
            ctx, page, errors, requests, dialogs = self.open(width)
            try:
                self.assertEqual(errors, [])
                self.assertEqual(dialogs, [])
                self.assertTrue(all(u.startswith("file:") for u in requests), requests)
                self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), width)
                self.assertIn("PRODUCTION", page.inner_text("#prodline"))
                self.assertIn(EVIL, page.inner_text("#tree"))          # shown as text, not run
                page.click(f"#tree .rec[data-k='label_version|{self.ids['NEW']}']")
                d = page.inner_text("#detail")
                for t in ("Status history (3 lines)", "created", "qc_passed", "pushed", "supersedes", self.ids["OLD"]):
                    self.assertIn(t, d)
                self.assertTrue(page.is_visible("#detail"))
                if SHOTS:
                    Path(SHOTS).mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(Path(SHOTS) / f"history_{width}.png"), full_page=width > 400)
                page.fill("#fq", "lv_2099")
                vis = page.eval_on_selector_all("#tree li:not(.hide) > * > .rec, #tree li:not(.hide) > details > summary > .rec",
                                                "els => els.filter(e => e.offsetParent).map(e => e.dataset.k)")
                self.assertIn(f"label_version|{self.ids['OLD']}", vis)
                self.assertNotIn(f"model|{self.ids['MOD']}", vis)
                page.fill("#fq", "")
                page.select_option("#fe", "job")
                self.assertEqual(page.eval_on_selector_all("#queue .card:not(.hide)", "els => els.length"), 2)
                page.select_option("#fe", "model")
                self.assertEqual(page.eval_on_selector_all("#queue .card:not(.hide)", "els => els.length"), 0)
            finally:
                ctx.close()

    def test_graph_and_lookup(self):
        v5 = "model|v5_evnonev_y8m_20261029"                      # v5 F5, from the samples
        for width in (390, 1280):
            ctx, page, errors, requests, dialogs = self.open(width)
            try:
                page.click("#t-graph")
                self.assertTrue(page.is_visible("#graph"))
                self.assertFalse(page.is_visible("#tree"))
                n_default = page.eval_on_selector_all("#gcanvas .gnode", "els => els.length")
                page.click(f".gnode[data-k='{v5}']")
                self.assertIn("v5_evnonev_y8m_20261029", page.inner_text("#detail"))
                self.assertIn("★ production", page.inner_text(f".gnode[data-k='model|{self.ids['MOD']}']"))
                if width < 900:
                    page.click("#detail .close")
                    self.assertTrue(page.evaluate("(() => {const g = document.getElementById('gwrap');"
                                                  " return g.scrollWidth > g.clientWidth;})()"))   # scrolls inside the box
                page.check("#ghist")
                n_hist = page.eval_on_selector_all("#gcanvas .gnode", "els => els.length")
                self.assertGreater(n_hist, n_default)
                self.assertGreater(page.eval_on_selector_all("#gcanvas .gnode.dim", "els => els.length"), 0)
                self.assertGreater(page.eval_on_selector_all("#gsvg path.back", "els => els.length"), 0)  # pkg -> lv
                self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), width)
                page.click("#t-lookup")
                page.fill("#lq", "square143_hole73")
                card = page.inner_text("#lres .lcard")
                self.assertIn("20260126-MDA-MB-KH__2_square143_hole73_0_hm_png", card)
                self.assertIn("20260126", card)                            # its session
                page.fill("#lq", "20260126 - MDA-MB KH__2_square143_hole73_0_hm.png")   # Roboflow-style name
                self.assertEqual(page.eval_on_selector_all("#lres .lcard", "els => els.length"), 1)
                page.fill("#lq", "zz_no_such_image")                        # control: a made-up name
                self.assertIn("No match", page.inner_text("#lres"))
                self.assertEqual(page.eval_on_selector_all("#lres .lcard", "els => els.length"), 0)
                self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), width)
                self.assertEqual(errors, [])
                self.assertTrue(all(u.startswith("file:") for u in requests), requests)
            finally:
                ctx.close()


if __name__ == "__main__":
    unittest.main()
