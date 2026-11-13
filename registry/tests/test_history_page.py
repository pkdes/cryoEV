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
sys.path.insert(0, str(REG.parent))                   # the repo root: `from registry import ...`
sys.path.append(str(REG.parent / "tests"))
import cryoev_testkit as TK  # noqa: E402  (shared helpers)
from registry import history_page as hp  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

SAMPLES = REG / "samples"
EVIL = "pkg_<script>alert(1)</script>&x"
SHOTS = os.environ.get("HISTORY_TEST_SHOTS")


sys.path.append(str(Path(__file__).resolve().parents[2] / "tests"))
from cryoev_testkit import browser  # noqa: E402  (the gate skips these when no UI file changed)


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
        self.assertEqual(len(self.m_s["recs"]), 16)  # 14 entity samples (incl. prediction_set) + 2 job samples
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

    def test_live_tab(self):
        for t in ('id="t-live"', '>Live</button>', '<div id="live" class="pane hide">', "live/live.json",
                  "cache:'no-store'", "location.protocol.indexOf('http')===0", "jobs/monitor.py --share"):
            self.assertIn(t, self.html_s)

    def test_servers_bar(self):
        self.assertEqual([(x["name"], x["label"], x["public_port"], x["local_port"], x["path"]) for x in self.m_s["servers"]],
                         [("review", "Review", 443, 8766, "/"), ("sandbox", "Sandbox", 8443, 8765, "/"),
                          ("history", "History", 8445, 8768, "/history.html")])
        head = self.html_s[self.html_s.index("<h1>Registry history</h1>"):self.html_s.index('<div class="meta" id="gen">')]
        self.assertIn('<nav class="srvbar" id="srv" aria-label="Servers"></nav>', head)   # right after the title
        self.assertNotRegex(self.html_s, r"\?k=[A-Za-z0-9_-]{6,}")                       # no key, ever

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
            "roboflow_export": [{"schema": "roboflow_export/1", "id": "roboflow - 20990101 x"},
                                {"schema": "roboflow_export/1", "id": "roboflow - 20980101 unused"}],
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
        TK.write_registry(reg, recs, samples=False)
        cls.m = hp.build([reg], t / "out" / "history.html")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_graph(self):
        nodes, edges = graph_index(self.m)
        self.assertEqual(self.m["graph"]["cols"], ["Imaging sessions", "Labels: automatic", "Labels: human",
                                                  "Labels: revised", "Datasets", "Models"])
        for e in (("roboflow_export|roboflow - 20990101 x", "dataset|ds_1", "source"),
                  ("dataset|ds_1", "model|v9_m_20990101", "train_data"),
                  ("model|v9_m_20990101", "review_package|pkg_new", "predictions"),
                  ("review_package|pkg_new", "label_version|lv_20990102_0900", "edits"),        # same lane
                  ("model|v9_m_20990101", "job|job_20990102_revise_new", "predictions"),
                  ("job|job_20990102_revise_new", "label_version|lv_20990102_0900", "output"),  # same lane
                  ("pull|roboflow_pull_20990101_0000", "label_version|lv_20990102_0900", "base")):
            self.assertIn(e, edges)
        self.assertFalse(any(p == "pull|roboflow_pull_20990101_0000" and c.startswith("job|") for p, c, _ in edges))
        self.assertFalse(any(c == "review_package|pkg_old_part1" for _, c, _ in edges))   # parts sit under their package
        hidden = {k for k, n in nodes.items() if n[2]}
        self.assertEqual(hidden, {"label_version|lv_20990101_0900", "review_package|pkg_old",
                                  "review_package|pkg_old_part1", "job|job_20990103_annotate_b",
                                  "roboflow_export|roboflow - 20980101 unused"})          # nothing built from it
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



def full_refs(m, name):
    """-> {entity key: full ref list} for one image (the 3-item [n, rows, "reviewed"] form included)."""
    L = m["lk"]
    for n, refs in L["i"]:
        if n == name:
            return {L["e"][r[0] if isinstance(r, list) else r]: (r if isinstance(r, list) else None) for r in refs}
    return None


def scope_registry(t: Path):
    """Three label versions of one package: lv_..0900 has no review_scope (made before 2026-11-04), lv_..1000 is
    all_edits, lv_..1100 is reviewed_only with 1 of 3 images reviewed (img_c, approved with no changes) and img_b
    held back. Known answers: chips amber / amber / "verified 1/3"; img_c "reviewed" in lv_..1100 only."""
    reg = t / "registry"
    reg.mkdir(parents=True)
    vers = {
        "lv_20990101_0900": {"n_images_applied": 2},
        "lv_20990101_1000": {"n_images_applied": 2, "review_scope": "all_edits", "n_reviewed": 1, "n_package_images": 3,
                             "reviewed_images": ["img_c_png"], "n_held_back": 0, "held_back_images": [],
                             "supersedes": "lv_20990101_0900"},
        "lv_20990101_1100": {"n_images_applied": 0, "review_scope": "reviewed_only", "n_reviewed": 1,
                             "n_package_images": 3, "reviewed_images": ["img_c_png"], "n_held_back": 1,
                             "held_back_images": ["img_b_png"], "supersedes": "lv_20990101_1000"},
    }
    lines = []
    for lv, v in vers.items():
        d = t / "labels" / lv
        d.mkdir(parents=True)
        (d / "version.json").write_text(json.dumps({"id": lv, "pkg": "pkg_s", **v}), encoding="utf-8")
        rows = [] if v.get("review_scope") == "reviewed_only" else ["img_a_png", "img_b_png"]
        (d / "changes.jsonl").write_text("".join(json.dumps({"schema": "label_change/1", "label_version": lv,
                                                             "image": im}) + "\n" for im in rows), encoding="utf-8")
        lines.append({"schema": "label_version/1", "id": lv, "status": "created", "owner": str(d / "version.json"),
                      **({"supersedes": v["supersedes"]} if "supersedes" in v else {})})
    (reg / "label_version.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    (reg / "image.jsonl").write_text("".join(json.dumps({"schema": "image/1", "id": f"img_{c}_png", "session": "20990101"})
                                             + "\n" for c in "abc"), encoding="utf-8")
    return reg


class ReviewScopeChips(unittest.TestCase):
    OLD, ALL, REV = "label_version|lv_20990101_0900", "label_version|lv_20990101_1000", "label_version|lv_20990101_1100"

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.out = t / "out" / "history.html"
        cls.m = hp.build([scope_registry(t)], cls.out)
        cls.html = cls.out.read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_chips(self):
        R = self.m["recs"]
        self.assertEqual(R[self.REV]["ver"], ["verified 1/3", "ver"])
        self.assertEqual(R[self.ALL]["ver"], ["all edits incl. unreviewed", "unrev"])
        self.assertEqual(R[self.OLD]["ver"], ["all edits incl. unreviewed", "unrev"])     # no scope field
        self.assertIsNone(R["image|img_a_png"]["ver"])
        nodes, _ = graph_index(self.m)
        self.assertEqual(nodes[self.REV][4], "verified 1/3 · 0 images changed")
        self.assertEqual(nodes[self.OLD][4], "all edits incl. unreviewed · 2 images changed")
        self.assertEqual((R[self.REV]["sup"], R[self.ALL]["supby"]), ("lv_20990101_1000", ["lv_20990101_1100"]))
        self.assertTrue(R[self.ALL]["dim"])                    # superseded by the newer partial ship

    def test_lookup_reviewed(self):
        c = full_refs(self.m, "img_c_png")
        e = self.m["lk"]["e"]
        self.assertEqual(c, {self.ALL: [e.index(self.ALL), None, "reviewed"], self.REV: [e.index(self.REV), None, "reviewed"]})
        a = full_refs(self.m, "img_a_png")                     # changed, not reviewed: the plain row count
        self.assertEqual(a, {self.OLD: [e.index(self.OLD), 1], self.ALL: [e.index(self.ALL), 1]})
        self.assertEqual(lookup_refs(self.m, "img_c_png"), {self.ALL: None, self.REV: None})

    def test_page_rules(self):
        self.assertNotIn("innerHTML", self.html)
        self.assertIsNone(re.search(r"(?i)https?://", self.html))

    @browser
    @unittest.skipIf(sync_playwright is None, "playwright not installed")
    def test_browser(self):
        with sync_playwright() as pw:
            browser = launch(pw)
            if browser is None:
                self.skipTest("no Edge / Chrome for Playwright")
            try:
                for width in (390, 1280):
                    page = browser.new_page(viewport={"width": width, "height": 900})
                    errors = []
                    page.on("pageerror", lambda e: errors.append(str(e)))
                    page.goto(self.out.as_uri())
                    page.wait_for_selector("#tree .rec")
                    row = f"#tree .rec[data-k='{self.REV}']"
                    self.assertEqual(page.inner_text(row + " .st.ver"), "verified 1/3")
                    self.assertEqual(page.inner_text(f"#tree .rec[data-k='{self.OLD}'] .st.unrev"), "all edits incl. unreviewed")
                    amber = page.eval_on_selector(f"#tree .rec[data-k='{self.OLD}'] .st.unrev", "e => getComputedStyle(e).color")
                    plain = page.eval_on_selector(f"#tree .rec[data-k='{self.OLD}'] .rid", "e => getComputedStyle(e).color")
                    self.assertNotEqual(amber, plain)
                    page.click(row)
                    d = page.inner_text("#detail")
                    self.assertIn("verified 1/3", d)
                    self.assertIn("1 edited image held back (not reviewed)", d)
                    if width < 900:
                        page.click("#detail .close")
                    page.click(f"#tree .rec[data-k='{self.OLD}']")
                    self.assertIn("every edit applied", page.inner_text("#detail"))
                    if width < 900:
                        page.click("#detail .close")
                    page.click("#t-graph")
                    page.check("#ghist")
                    self.assertIn("verified 1/3", page.inner_text(f".gnode[data-k='{self.REV}'] .gsub"))
                    self.assertEqual(page.eval_on_selector_all(f".gnode[data-k='{self.OLD}'] .gsub.unrev", "e => e.length"), 1)
                    page.click("#t-lookup")
                    page.fill("#lq", "img_c")
                    card = page.inner_text("#lres .lcard")
                    self.assertIn("lv_20990101_1100 (reviewed)", card)
                    self.assertIn("changed or reviewed it", card)
                    page.fill("#lq", "img_a")
                    self.assertNotIn("reviewed)", page.inner_text("#lres .lcard"))
                    self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), width)
                    self.assertEqual(errors, [])
                    page.close()
            finally:
                browser.close()


R9_SESSIONS = {"20250730": 44, "20260126": 7, "20260420": 9, "20260506": 20, "20260512": 22}
R9_EXP, R9_OLD, R9_LV, R9_JOB = "roboflow - 20261028 cryoai v5 S1", "roboflow - 20260529", "lv_20261111_1409", \
    "job_20261111_revise_cell"


class R9LanesAndAttribution(unittest.TestCase):
    """Round 9 (#15, #16, attribution) on a folder registry with the live known answer's structure: the export's
    image list has 160 rows -> sessions 44 / 7 / 9 / 20 / 22 and 58 with no session ("Session unknown"); an older
    export with no list; a label version and a job written by an agent (via), one later re-stamped by a person."""

    @classmethod
    def setUpClass(cls):
        cls.saved = TK.strip_environ(strip_all=True)                 # CRYOEV_*-wide; restored in tearDownClass
        cls.tmp = tempfile.TemporaryDirectory()
        reg = Path(cls.tmp.name) / "registry"
        (reg / "lists").mkdir(parents=True)
        images, stems = [], []
        for sid, n in R9_SESSIONS.items():
            for i in range(n):
                images.append({"schema": "image/1", "id": f"{sid}_t{i}_png", "session": sid,
                               "grid": "2642" if sid == "20260512" and i < 16 else None})
                stems.append(f"{sid}_t{i}_png")
        for i in range(58):
            images.append({"schema": "image/1", "id": f"{i:04d}_Jul30_jpg", "session": None, "grid": None})
            stems.append(f"{i:04d}_Jul30_jpg")
        (reg / "lists" / "s1.tsv").write_text("".join(f"{x}\ttrain\n" for x in stems), encoding="utf-8")
        lines = {
            "image": images,
            "session": [{"schema": "session/1", "id": sid, "date": f"{sid[:4]}-{sid[4:6]}-{sid[6:]}",
                         "status": "processed"} for sid in list(R9_SESSIONS) + ["20261111"]],   # last: in no list
            "roboflow_export": [{"schema": "roboflow_export/1", "id": R9_EXP, "counts": {"train": {"images": 160}},
                                 "image_list": {"file": "lists/s1.tsv", "sha256": None}},
                                {"schema": "roboflow_export/1", "id": R9_OLD, "counts": {},
                                 "note": "local placeholder: no COCO in the cache"}],
            "review_package": [{"schema": "review_package/1", "id": "pkg_cell", "base": R9_EXP, "base_kind": "export",
                                "status": "applied", "model": "m_v5"},
                               {"schema": "review_package/1", "id": "pkg_audit", "base": R9_EXP,
                                "base_kind": "export", "status": "in_review", "model": "m_v5"}],
            "label_version": [{"schema": "label_version/1", "id": R9_LV, "status": "created", "pkg": "pkg_cell",
                               "who": "pujan", "via": "claude", "base": R9_EXP},
                              {"schema": "label_version/1", "id": "lv_20261111_1500", "status": "created",
                               "who": "pujan", "via": "claude", "base": R9_EXP},
                              {"schema": "label_version/1", "id": "lv_20261111_1500", "status": "qc_passed",
                               "who": "pujan", "status_by": "person-b", "base": R9_EXP},    # a person's line
                              {"schema": "label_version/1", "id": "lv_20261111_1600", "status": "created",
                               "who": "person-b", "base": R9_EXP},                              # a person made it
                              {"schema": "label_version/1", "id": "lv_20261111_1600", "status": "qc_passed",
                               "who": "person-b", "status_by": "pujan", "status_via": "claude",  # agent set-status
                               "base": R9_EXP}],
            "dataset": [{"schema": "dataset/1", "id": "ds_cell", "status": "frozen",
                         "inputs": [{"id": R9_LV, "role": "source", "entity": "label_version"},
                                    {"id": R9_EXP, "role": "images", "entity": "roboflow_export"}]},
                        {"schema": "dataset/1", "id": "ds_old", "status": "frozen",
                         "inputs": [{"id": R9_OLD, "role": "source", "entity": "roboflow_export"}]}],
            "model": [{"schema": "model/1", "id": "m_v5", "status": "production",
                       "inputs": [{"id": R9_EXP, "role": "train_data", "entity": "roboflow_export"}]},
                      {"schema": "model/1", "id": "m_c1", "status": "candidate",
                       "inputs": [{"id": "ds_cell", "role": "train_data", "entity": "dataset"}]}],
            "job": [{"schema": "job/1", "id": R9_JOB, "type": "revise", "status": "in_progress", "who": "pujan",
                     "via": "claude", "package": "pkg_audit", "outputs": [],
                     "inputs": [{"id": R9_EXP, "role": "base"}]}],
        }
        TK.write_registry(reg, lines, samples=False)
        cls.out = Path(cls.tmp.name) / "out" / "history.html"
        cls.m = hp.build([reg], cls.out)
        cls.nodes, cls.edges = graph_index(cls.m)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        os.environ.update(cls.saved)

    def test_lanes(self):
        self.assertEqual(self.m["graph"]["cols"], ["Imaging sessions", "Labels: automatic", "Labels: human",
                                                  "Labels: revised", "Datasets", "Models"])
        col = {k: n[1] for k, n in self.nodes.items()}
        exp = "roboflow_export|" + R9_EXP
        self.assertEqual((col[exp], self.nodes[exp][5]), (2, "export · via Roboflow"))   # no Roboflow lane
        self.assertEqual((col["label_version|" + R9_LV], self.nodes["label_version|" + R9_LV][5]),
                         (3, "labels · revised"))
        self.assertEqual(col["label_version|lv_20261111_1500"], 2)                       # no review: human
        self.assertEqual((col["review_package|pkg_cell"], col["review_package|pkg_audit"]), (3, 3))
        self.assertEqual(col["job|" + R9_JOB], 3)
        self.assertEqual((col["dataset|ds_cell"], col["model|m_c1"]), (4, 5))
        self.assertIn("→ " + R9_LV, self.nodes["review_package|pkg_cell"][4])           # applied: names its lv
        self.assertIn((exp, "review_package|pkg_cell", "base"), self.edges)              # export -> review -> lv
        self.assertFalse([e for e in self.edges if e[1] == "job|" + R9_JOB and e[2] == "base"])

    def test_export_reaches_sessions_with_counts(self):
        exp = "roboflow_export|" + R9_EXP
        got = {p.split("|", 1)[1]: role for p, c, role in self.edges if c == exp and p.startswith("session|")}
        self.assertEqual({k: v.split(" · ")[0] for k, v in got.items()},
                         {**{s: f"{n} images" for s, n in R9_SESSIONS.items()}, "?": "58 images"})
        self.assertEqual(got["20260512"], "22 images · grid 2642: 16")
        self.assertEqual(sum(int(v.split()[0]) for v in got.values()), 160)
        self.assertEqual(self.nodes[hp.UNKNOWN_SESSION][1:5], [0, 0, -1, "not in the image records"])
        self.assertNotIn("session|20261111", self.nodes)                               # in no export list: not drawn
        self.assertNotIn(hp.UNKNOWN_SESSION, self.m["recs"])                            # a node, never a record
        old = "roboflow_export|" + R9_OLD
        self.assertEqual(self.nodes[old][4], "image list not recorded · local placeholder: no COCO in the cache")
        self.assertFalse([e for e in self.edges if e[1] == old and e[0].startswith("session|")])

    def test_agent_attribution(self):
        r = self.m["recs"]["label_version|" + R9_LV]
        self.assertEqual((r["who"]["who"], r["aiw"]), ("claude for pujan", ["who"]))
        self.assertEqual([(h["by"], h["ai"]) for h in r["hist"]], [("claude for pujan", True)])
        later = self.m["recs"]["label_version|lv_20261111_1500"]     # last line by a person: no agent credit
        self.assertEqual((later["who"]["who"], later["who"]["status_by"], later["aiw"]),
                         ("pujan", "person-b", []))
        self.assertEqual([h["ai"] for h in later["hist"]], [True, False])
        self.assertEqual(later["hist"][1]["by"], None)                # no recorded_by / tool on that line
        job = self.m["recs"]["job|" + R9_JOB]["job"]
        self.assertEqual((job["who"], job["ai"]), ("claude for pujan", True))
        self.assertEqual(hp.who_fields({"who": "pujan"}, None), ({"who": "pujan"}, []))     # control: a person

    def test_status_via_credits_the_status_only(self):
        """An agent's set-status line carries status_via (no via): its status row and status_by read "claude for
        pujan" + AI; the record's maker (who) stays the person's."""
        r = self.m["recs"]["label_version|lv_20261111_1600"]
        self.assertEqual([(h["status"], h["by"], h["ai"]) for h in r["hist"]],
                         [("created", None, False), ("qc_passed", "claude for pujan", True)])
        self.assertEqual((r["who"]["who"], r["who"]["status_by"], r["aiw"]), ("person-b", "claude for pujan",
                                                                             ["status_by"]))
        self.assertIsNone(hp.line_via({"status_via": "claude"}))                  # not for a non-status field
        self.assertEqual(hp.line_via({"status_via": "claude"}, status=True), "claude")

    @browser
    @unittest.skipIf(sync_playwright is None, "playwright not installed")
    def test_browser_ai_tag_lanes_and_arrows(self):
        with sync_playwright() as pw:
            browser = launch(pw)
            if browser is None:
                self.skipTest("no Edge / Chrome for Playwright")
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(self.out.as_uri())
                page.wait_for_selector("#tree .rec")
                page.click(f"#tree .rec[data-k='label_version|{R9_LV}']")
                self.assertEqual(page.inner_text("#detail dd .ai"), "AI")
                self.assertIn("claude for pujan", page.inner_text("#detail"))
                page.click("#t-graph")
                page.check("#ghist")
                heads = page.locator(".ghead").all_text_contents()
                self.assertEqual(heads[0], "Imaging sessions")
                self.assertNotIn("Roboflow labels", heads)
                n = page.locator("#gsvg g").count()
                self.assertGreater(n, 8)
                self.assertEqual(page.locator("#gsvg path.ga").count(), n)           # one arrow per edge
                self.assertIn("44 images", page.locator("#gsvg text.gcount").all_text_contents())
                self.assertEqual(page.eval_on_selector(".gnode.btw", "e => getComputedStyle(e).borderTopStyle"),
                                 "dashed")
                self.assertIn("in progress", page.inner_text(".gnode[data-k='review_package|pkg_audit']"))
                self.assertEqual(page.inner_text(".gnode.unk .gid"), "Session unknown")
                self.assertEqual(errors, [])
            finally:
                browser.close()


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

    def test_production_ancestors_shown(self):
        """Every ancestor of the production model is shown by default (v4 0924 via v4_y8m's init weights)."""
        par = {}
        for p, c, _ in self.edges:
            par.setdefault(c, []).append(p)
        anc, todo = set(), [self.V5]
        while todo:
            for p in par.get(todo.pop(), []):
                if p not in anc:
                    anc.add(p)
                    todo.append(p)
        self.assertIn("roboflow_export|roboflow - 20260924 cryoai v4", anc)
        self.assertEqual([k for k in anc if self.nodes[k][2]], [])
        unused = {k for k, n in self.nodes.items() if n[2] and k.startswith("roboflow_export|")}
        self.assertTrue(unused and not unused & anc)

    def test_revise_back_edge(self):
        for e in ((self.V5, self.PKG2, "predictions"), (self.PKG2, self.LV, "edits"),
                  (self.V5, self.JOB2, "predictions"), (self.JOB2, self.LV, "output")):
            self.assertIn(e, self.edges)
        col = {k: n[1] for k, n in self.nodes.items()}
        self.assertEqual(col[self.PKG2], col[self.LV])             # the review sits in the lane it produces
        self.assertGreater(col[self.V5], col[self.PKG2])           # model -> review: drawn right to left

    def test_history_adds_superseded_v1_packages(self):
        hidden = {k for k, n in self.nodes.items() if n[2] and k.startswith("review_package|")}
        v1 = {"review_package|prelipo_v5F5_review"} | {f"review_package|prelipo_v5F5_review_part{i}" for i in range(1, 5)}
        self.assertEqual(hidden, v1)
        self.assertEqual(self.nodes[self.PKG2][2], 0)

    def test_lv_0909_flagged_all_edits(self):
        """lv_20261030_0909 has no review_scope (made before 2026-11-04, every edit applied): amber, not "verified"."""
        self.assertEqual(self.m["recs"][self.LV]["ver"], ["all edits incl. unreviewed", "unrev"])
        self.assertTrue(self.nodes[self.LV][4].startswith("all edits incl. unreviewed"))

    def test_lookup(self):
        rows = (REAL / "lists" / "roboflow_20261028_cryoai_v5_S1_evnonev.tsv").read_text(encoding="utf-8").splitlines()
        stem, split = rows[0].split("\t")[:2]
        self.assertEqual(lookup_refs(self.m, stem)[self.DS], split)
        changed = [n for n, _ in self.m["lk"]["i"] if self.LV in (lookup_refs(self.m, n) or {})]
        applied = json.loads((REAL.parent / "labels" / self.LV.split("|")[1] / "version.json")
                             .read_text(encoding="utf-8"))["n_images_applied"]
        self.assertEqual(len(changed), applied)                     # version.json n_images_applied (28 today)
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


@browser
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
                bar = page.inner_text("#srv")                           # as a file: names only, no links, no dots
                for t in ("Review", "Sandbox", "History", "open via the history server for links"):
                    self.assertIn(t, bar)
                self.assertEqual(page.eval_on_selector_all("#srv a, #srv .chip", "els => els.length"), 0)
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

    def test_live_from_file_makes_no_request(self):
        ctx, page, errors, requests, dialogs = self.open(390)
        try:
            page.click("#t-live")
            self.assertTrue(page.is_visible("#live"))
            note = page.inner_text("#lbody")
            self.assertIn("python jobs/monitor.py --share \\\\gpu-desktop\\cj", note)
            self.assertIn("served by the history server", note)
            page.wait_for_load_state("networkidle")      # a request, if the page made one, has been issued by now
            self.assertEqual(errors, [])
            self.assertTrue(requests and all(u.startswith("file:") for u in requests), requests)
            self.assertFalse(any("live.json" in u for u in requests))
        finally:
            ctx.close()


@browser
class LiveServed(unittest.TestCase):
    """The Live tab served over a localhost http.server, with live/live.json written by jobs/monitor.py --once from
    temp queues built in code (jobs/tests/jobtest.py): queue A has a job at epoch 4 of 300 with batch 8 of 16
    (ETA 3700 s = 1 h 2 min), queue B one with best epoch 2 + patience 3 (bound 12.5 s), a stale heartbeat and a
    cancelled job. A second folder holds the same snapshot with its time set 10 min back: the stale banner."""

    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        import functools
        import http.server
        import subprocess
        import threading
        sys.path.insert(0, str(REG.parent / "jobs" / "tests"))
        import jobtest as T
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        a, b = t / "qa", t / "qb"
        (a / "q").mkdir(parents=True)
        (b / "q").mkdir(parents=True)
        T.watch_fixture(a, job="job_20261104_live")
        T.running_fixture(a, progress={"batch": 8, "batch_asked": 16}, patience=300)
        T.watch_fixture(b, job="job_20261104_pat", age_s=600)
        T.running_fixture(b, job="job_20261104_pat", progress={"best_epoch": 2}, patience=3)
        T.done_fixture(b, "job_20261104_cxl", status="cancelled", reason="cancelled by pujan: test")
        lap = t / "lap"
        lap.mkdir()
        env = {k: v for k, v in os.environ.items() if k != "CRYOEV_REGISTRY"}
        env["CRYOEV_LAPTOP_ROOT"] = str(lap)
        cls.live, cls.stale = t / "live_site", t / "stale_site"
        cls.servers = []
        for site in (cls.live, cls.stale):
            site.mkdir()

            class Quiet(http.server.SimpleHTTPRequestHandler):
                def log_message(self, *a):
                    pass
            srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(site)))
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            cls.servers.append(srv)
        import socket
        # "down" fixture: a port that is bound but never listens, and stays bound until teardown. (It used to be
        # bound-then-closed, so another process could take the port and make "sandbox" read as up.)
        cls.dead_sock = socket.socket()
        cls.dead_sock.bind(("127.0.0.1", 0))
        dead = cls.dead_sock.getsockname()[1]
        # the Servers bar's fixture: review and history = the two test servers (up), sandbox = a closed port (down)
        cls.ports = {"review": cls.servers[0].server_address[1], "sandbox": dead,
                     "history": cls.servers[1].server_address[1]}
        r = subprocess.run([sys.executable, str(REG.parent / "jobs" / "monitor.py"), "--share", str(a), "--share",
                            str(b), "--once", "--out", str(cls.live / "live" / "live.json"),
                            "--servers", ",".join(f"{k}={v}" for k, v in cls.ports.items())],
                           capture_output=True, text=True, env=env, timeout=120)
        assert r.returncode == 0, r.stderr
        hp.build([SAMPLES], cls.live / "history.html")
        snap = json.loads((cls.live / "live" / "live.json").read_text(encoding="utf-8"))
        import datetime as dt
        snap["generated_utc"] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=10)).isoformat()
        (cls.stale / "live").mkdir()
        (cls.stale / "live" / "live.json").write_text(json.dumps(snap), encoding="utf-8")
        hp.build([SAMPLES], cls.stale / "history.html")
        cls.pw = sync_playwright().start()
        cls.browser = launch(cls.pw)
        if cls.browser is None:
            cls.tearDownClass()
            raise unittest.SkipTest("no Edge / Chrome for Playwright")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "browser", None):
            cls.browser.close()
        cls.pw.stop()
        for s in cls.servers:
            s.shutdown()
            s.server_close()
        cls.dead_sock.close()
        cls.tmp.cleanup()

    def open(self, srv, width):
        base = f"http://127.0.0.1:{srv.server_address[1]}/"
        ctx = self.browser.new_context(viewport={"width": width, "height": 900})
        ctx.route("**/*", lambda route: route.continue_() if route.request.url.startswith(base) else route.abort())
        page = ctx.new_page()
        errors, requests = [], []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: requests.append(r.url))
        page.goto(base + "history.html")
        page.click("#t-live")
        return ctx, page, errors, requests, base

    def test_live_tab(self):
        for width in (390, 1280):
            ctx, page, errors, requests, base = self.open(self.servers[0], width)
            try:
                page.wait_for_selector("#lbody .lrun .lchart svg")
                text = page.inner_text("#lbody")
                for t in ("Epoch 4 / 300", "1 h 2 min", "batch 8 of 16 asked", "31.4 GB", "≤ 13 s",
                          "job_20261104_cxl", "cancelled by pujan: test", "stale", "alive", "Updated"):
                    self.assertIn(t, text)
                self.assertNotIn("Stale: this snapshot", text)
                runs = page.query_selector_all("#lbody .lrun")
                self.assertEqual(len(runs), 2)
                charts = runs[0].query_selector_all(".lchart svg")
                self.assertEqual(len(charts), 2)                         # train losses, val mask mAP
                self.assertEqual(len(charts[0].query_selector_all("path.lline")), 4)
                self.assertEqual(len(charts[1].query_selector_all("path.lline")), 2)
                charts[1].hover()
                tip = runs[0].query_selector_all(".ltip")[1].inner_text()
                self.assertIn("Epoch", tip)
                self.assertIn("mAP50-95", tip)
                self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), width)
                if SHOTS:
                    Path(SHOTS).mkdir(parents=True, exist_ok=True)
                    page.mouse.move(0, 0)
                    page.screenshot(path=str(Path(SHOTS) / f"live_{width}.png"), full_page=True)
                self.assertEqual(errors, [])
                self.assertIn(base + "live/live.json", requests)
                self.assertTrue(all(u.startswith(base) for u in requests), requests)
            finally:
                ctx.close()

    def bar(self, page):
        """{name: [href or None, status class or None]} from the Servers bar."""
        return page.evaluate("""() => Object.fromEntries([...document.querySelectorAll('#srv .srvi')].map(w => {
            const a = w.querySelector('a'), c = w.querySelector('.chip');
            return [w.dataset.name, [a ? a.getAttribute('href') : null, c ? c.className.split(' ').pop() : null]];}))""")

    def fresh(self):
        """The live site's snapshot, its time set to now (a slow run must not make it stale), as JSON text."""
        import datetime as dt
        f = self.live / "live" / "live.json"
        snap = json.loads(f.read_text(encoding="utf-8"))
        snap["generated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        f.write_text(json.dumps(snap), encoding="utf-8")
        return json.dumps(snap)

    def test_servers_bar_dots(self):
        """The fixture snapshot: review and history up, sandbox down; the page itself only asks its own origin."""
        snap = json.loads(self.fresh())
        self.assertEqual({x["name"]: x["up"] for x in snap["servers"]}, {"review": True, "sandbox": False, "history": True})
        for width in (390, 1280):
            ctx, page, errors, requests, base = self.open(self.servers[0], width)
            try:
                page.wait_for_selector("#srv .chip.down")
                b = self.bar(page)
                self.assertEqual({k: v[1] for k, v in b.items()}, {"review": "up", "sandbox": "down", "history": "up"})
                # opened on 127.0.0.1 at a port that is not a tailnet one: plain http to each local port
                self.assertEqual({k: v[0] for k, v in b.items()},
                                 {"review": "http:" + "//127.0.0.1:8766/", "sandbox": "http:" + "//127.0.0.1:8765/",
                                  "history": "http:" + "//127.0.0.1:8768/history.html"})
                self.assertIn("First time on a device: open the review link with its ?k= key (ask Pujan)",
                              page.inner_text("#srv"))
                self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), width)
                self.assertEqual(errors, [])
                self.assertTrue(all(u.startswith(base) for u in requests), requests)   # no probe from the browser
            finally:
                ctx.close()

    def test_servers_bar_stale_is_grey(self):
        ctx, page, errors, requests, base = self.open(self.servers[1], 390)
        try:
            page.wait_for_selector("#lbody .lstale")
            self.assertEqual({k: v[1] for k, v in self.bar(page).items()},
                             {"review": "unk", "sandbox": "unk", "history": "unk"})
            self.assertIn("unknown", page.inner_text("#srv"))
            self.assertEqual(errors, [])
        finally:
            ctx.close()

    def open_as(self, origin, width):
        """The page as if served at `origin` (tailscale serve's https ports, or the history server's own port):
        every request is answered here from the live site, so nothing reaches a real server on this machine."""
        ctx = self.browser.new_context(viewport={"width": width, "height": 900})
        files = {"/history.html": ("text/html", (self.live / "history.html").read_bytes()),
                 "/live/live.json": ("application/json", self.fresh().encode())}

        def answer(route):
            u = route.request.url
            path = u[len(origin):].split("?")[0] if u.startswith(origin + "/") else None
            if path in files:
                route.fulfill(status=200, content_type=files[path][0], body=files[path][1])
            elif path is not None:
                route.fulfill(status=404, body="")
            else:
                route.abort()
        ctx.route("**/*", answer)
        page = ctx.new_page()
        errors, requests = [], []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: requests.append(r.url))
        page.goto(origin + "/history.html")
        page.wait_for_selector("#srv .chip.down")
        return ctx, page, errors, requests

    def test_servers_bar_hrefs(self):
        S, H = "https:" + "//", "http:" + "//"
        tail = {"review": "/", "sandbox": "/", "history": "/history.html"}
        public = {"review": S + "home-laptop.example", "sandbox": S + "home-laptop.example:8443",
                  "history": S + "home-laptop.example:8445"}
        cases = {S + "home-laptop.example:8445": public, S + "home-laptop.example": public,
                 H + "127.0.0.1:8768": {"review": H + "127.0.0.1:8766", "sandbox": H + "127.0.0.1:8765",
                                        "history": H + "127.0.0.1:8768"},
                 H + "localhost:8768": {"review": H + "localhost:8766", "sandbox": H + "localhost:8765",
                                        "history": H + "localhost:8768"}}
        for origin, want in cases.items():
            for width in (390, 1280):
                with self.subTest(origin=origin, width=width):
                    ctx, page, errors, requests = self.open_as(origin, width)
                    try:
                        self.assertEqual({k: v[0] for k, v in self.bar(page).items()},
                                         {k: v + tail[k] for k, v in want.items()})
                        self.assertTrue(all(u.startswith(origin + "/") for u in requests), requests)
                        self.assertIn(origin + "/live/live.json", requests)
                        self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), width)
                        self.assertEqual(errors, [])
                        if SHOTS and origin.endswith(":8445"):
                            Path(SHOTS).mkdir(parents=True, exist_ok=True)
                            page.screenshot(path=str(Path(SHOTS) / f"servers_bar_{width}.png"))
                    finally:
                        ctx.close()

    def test_stale_banner(self):
        ctx, page, errors, requests, base = self.open(self.servers[1], 390)
        try:
            page.wait_for_selector("#lbody .lstale")
            self.assertIn("Stale: this snapshot is 10 min old", page.inner_text("#lbody .lstale"))
            self.assertEqual(len(page.query_selector_all("#lbody .lrun")), 0)   # nothing shown as running
            self.assertIn("unknown (stale)", page.inner_text("#lbody"))
            if SHOTS:
                page.screenshot(path=str(Path(SHOTS) / "live_stale_390.png"), full_page=True)
            self.assertEqual(errors, [])
        finally:
            ctx.close()


if __name__ == "__main__":
    unittest.main()
