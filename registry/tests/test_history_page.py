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
        for text in ("Registry history", "Queue / status board", "History tree", "Search id", "prefers-color-scheme"):
            self.assertIn(text, self.html_s)
        self.assertEqual(self.m_s["stats"]["label_change_rows"], 1)
        self.assertEqual(len(self.m_s["recs"]), 13)
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
                page.select_option("#fe", "model")
                self.assertEqual(page.eval_on_selector_all("#board li:not(.hide)", "els => els.length"), 2)
            finally:
                ctx.close()


if __name__ == "__main__":
    unittest.main()
