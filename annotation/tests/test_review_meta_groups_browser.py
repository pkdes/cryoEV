"""
Browser tests (Playwright, headless Edge or Chrome; skipped without one) for round 12 in annotation/review_tool/index.html:
the Image info panel (#20), the sample switcher and sort, and sample / montage group actions (#21). The page is served
by the multi server in this process over the meta_fixture package (temp folder).

  python -m unittest annotation.tests.test_review_meta_groups_browser
  REVIEW_TEST_SHOTS=<dir> keeps screenshots (editor + info panel, switcher, reject dialog, rejected sample, phone).
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import meta_fixture as MF  # noqa: E402
import test_review_tool_browser as trb  # noqa: E402  (module: its TestCases must not be collected twice)
from annotation import multi_review_server as mrs  # noqa: E402
from annotation import review_groups as rg  # noqa: E402
sys.path.append(str(REPO / "tests"))
import cryoev_testkit as TK  # noqa: E402
from cryoev_testkit import browser  # noqa: E402

SHOTS = os.environ.get("REVIEW_TEST_SHOTS")
DESK = {"viewport": {"width": 1400, "height": 900}}
PHONE = {"viewport": {"width": 390, "height": 844}, "is_mobile": True, "has_touch": True, "device_scale_factor": 2}
EV, HEK = "EV-013-2", "HEK"


def shot(page, name):
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(SHOTS) / f"{name}.png"))


@browser
class MetaGroupsBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if trb.sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        cls.pw = trb.sync_playwright().start()
        cls.browser = trb.launch(cls.pw)
        if cls.browser is None:
            cls.pw.stop()
            raise unittest.SkipTest("no Edge / Chrome for Playwright")

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="r12ui_"))
        with TK.no_actor_env():
            self.fx = MF.build(self.tmp)
        self.pkg = self.fx["pkg"]
        self.key = (self.pkg / ".review_key").read_text(encoding="utf-8").strip()
        self.srv = mrs.make_server(self.fx["review"], "127.0.0.1", 0, people=[])
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}/{MF.PKG}/"
        self.errors = []

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def open(self, query="&who=tester&mode=editor", setup=None, wait_meta=True, **ctx):
        context = self.browser.new_context(**(ctx or DESK))
        self.addCleanup(context.close)
        page = context.new_page()
        page.on("pageerror", lambda e: self.errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and "404" not in m.text and self.errors.append(m.text))
        page.route("**/favicon.ico", lambda r: r.fulfill(status=204, body=""))  # the multi server 403s ids it doesn't serve
        if setup:
            setup(page)
        page.goto(self.base + "?k=" + self.key + query)
        page.wait_for_function("window.__review && __review.state().server && __review.state().imgReady", timeout=20000)
        if wait_meta:
            page.wait_for_function("__review.meta() !== null", timeout=20000)
        return page

    def go(self, page, file):
        i = page.evaluate(f"__review.D.images.findIndex(im => im.file === {json.dumps(file)})")
        page.locator("#items .it", has=page.locator(f'.nm[title="{file}"]')).click()
        page.wait_for_function(f"__review.state().cur === {i}")

    def info(self, page):
        """The Image info panel as {label: value}."""
        return page.evaluate("""() => { const o = {}, dl = document.getElementById('infoBox');
            for (const dt of dl.querySelectorAll('dt')) o[dt.textContent] = dt.nextElementSibling.textContent; return o; }""")

    def saved_reject(self, file):
        p = self.pkg / "edits" / f"{file}.json"
        d = json.loads(p.read_text(encoding="utf-8"))["edit"]["dec"].get("i:reject") if p.exists() else None
        return (d.get("v"), d.get("grp")) if d and d.get("v") else (None, None)

    # ------------------------------------------------------------------ #20 image info
    def test_info_panel_and_quick_check_line(self):
        page = self.open()
        self.go(page, "0002.png")
        i = self.info(page)
        self.assertEqual((i["Sample"], i["Montage"], i["Tile"], i["Defocus"]),
                         (EV, "grid10_1", "row 0, col 1 (piece 5)", "-4.48 µm"))
        self.assertEqual(i["Magnification"], "11,000×")
        self.assertIn("recipe 20261027", i["Processing"])
        self.assertIn("contrast clip 1–99 %", i["Processing"])
        self.assertIn("bandpass 2 nm", i["Processing"])
        self.assertEqual(i["Conversion"], MF.CONV)
        self.assertFalse(page.locator("#infoSec").is_hidden())
        shot(page, "r12_editor_info_panel")
        self.go(page, "0009.png")      # name only: montage and tile, the rest "not recorded"
        i = self.info(page)
        self.assertEqual((i["Montage"], i["Defocus"], i["Dose"]), ("grid6_9", "not recorded", "not recorded"))
        self.assertEqual(page.locator("#infoBox dd.nr").count(), 5)   # defocus, magnification, dose, exposure, acquired
        page.click("#toTri")
        page.wait_for_function("!document.getElementById('thMeta').hidden")
        self.assertRegex(page.inner_text("#thMeta"), r"^grid\d+_\d+ · row \d, col \d")
        self.assertEqual(self.errors, [])

    # ------------------------------------------------------------------ the sample switcher
    def test_sample_switcher_and_sort(self):
        page = self.open()
        opts = page.evaluate("[...document.getElementById('fSplit').options].map(o => [o.value, o.textContent])")
        self.assertEqual(opts, [["", "All samples"], [EV, f"{EV} · 6"], [HEK, f"{HEK} · 3"]])
        names = lambda: page.locator("#items .nm").evaluate_all("els => els.map(e => e.title)")  # noqa: E731
        self.assertEqual(len(names()), 9)
        page.select_option("#fSplit", HEK)
        self.assertEqual(sorted(names()), ["0007.png", "0008.png", "0009.png"])
        page.wait_for_function("__review.D.images[__review.state().cur].split === 'HEK'")   # moved into the sample
        page.keyboard.press("n")
        self.assertEqual(page.evaluate("__review.D.images[__review.state().cur].split"), HEK)
        page.click("#smpNext")                          # ◀ / ▶ Sample wrap around
        page.wait_for_function("document.getElementById('fSplit').value === 'EV-013-2'")
        self.assertEqual(len(names()), 6)
        page.select_option("#fSplit", "")
        page.select_option("#fSort", "sample")
        heads = page.locator("#items .gh").all_inner_texts()
        self.assertEqual([h.strip() for h in heads], [f"Sample {EV} · 6", "Montage grid10_1 · 4", "Montage grid10_2 · 2",
                                                      f"Sample {HEK} · 3", "Montage grid6_1 · 2", "Montage grid6_9 · 1"])
        self.assertEqual(names(), [f"{i:04d}.png" for i in range(1, 10)])
        shot(page, "r12_sample_switcher_sorted")
        page.select_option("#fSort", "flags")           # the other sorts: no headers, as before
        self.assertEqual(page.locator("#items .gh").count(), 0)
        self.assertEqual(self.errors, [])

    # ------------------------------------------------------------------ #21 group reject / undo / comment
    def test_reject_sample_marks_and_undo_restores(self):
        page = self.open()
        # 0002 rejected by hand first (the image dialog), with its own reason
        self.go(page, "0002.png")
        page.click("#imgEdit")
        page.click("#rejChips button:has-text('empty')")
        page.click("#imgSave")
        page.wait_for_function("__review.state().dirty === 0")
        self.assertEqual(self.saved_reject("0002.png"), ("reject:empty", None))
        self.go(page, "0001.png")
        page.click("#grpRej_sample")
        self.assertFalse(page.locator("#grpDlg").is_hidden())
        self.assertTrue(page.locator("#grpOk").is_disabled())           # a reason is required
        page.click("#grpReasons button:has-text('thick ice')")
        page.fill("#grpNote", "whole grid icy")
        self.assertEqual(page.inner_text("#grpCount"), "5 images of 6 will be rejected; 1 image already rejected stays as it is.")
        self.assertEqual(page.inner_text("#grpOk"), "Reject 5 images")
        shot(page, "r12_reject_sample_dialog")
        page.click("#grpOk")
        page.wait_for_function("document.getElementById('grpDlg').hidden")
        gid = rg.group_id("sample", EV)
        for i in range(1, 7):
            f = f"{i:04d}.png"
            self.assertEqual(self.saved_reject(f), ("reject:empty", None) if f == "0002.png" else ("reject:thick ice", gid))
        self.assertEqual(self.saved_reject("0007.png"), (None, None))
        # marked: the header tag, the side panel, the switcher, each row
        self.assertEqual(page.inner_text("#rejTag"), "Rejected (sample): thick ice")
        self.assertIn("Sample rejected: thick ice", page.inner_text("#grpBox"))
        self.assertIn("whole grid icy", page.inner_text("#grpBox"))
        self.assertIn(f"{EV} · 6 · rejected", page.evaluate("[...document.getElementById('fSplit').options].map(o => o.textContent).join('|')"))
        page.select_option("#fSort", "sample")
        self.assertIn("rej", page.locator("#items .gh").first.get_attribute("class"))
        self.assertEqual(page.locator("#items .rejb").count(), 6 + 1)    # 6 rows + the sample header
        self.assertFalse(page.locator("#grpUndo_sample").is_hidden())
        shot(page, "r12_rejected_sample_marked")
        # undo: only the group's five come back; 0002 keeps its own rejection
        page.click("#grpUndo_sample")
        self.assertIn("Undo for 5 images", page.inner_text("#grpOk"))
        page.click("#grpOk")
        page.wait_for_function("document.getElementById('grpDlg').hidden")
        for i in range(1, 7):
            f = f"{i:04d}.png"
            self.assertEqual(self.saved_reject(f), ("reject:empty", None) if f == "0002.png" else (None, None))
        self.assertTrue(page.locator("#rejTag").is_hidden())
        self.assertFalse(page.locator("#grpRej_sample").is_hidden())
        self.assertEqual(self.errors, [])

    def test_comment_and_montage_reject_seen_by_another_reviewer(self):
        a = self.open("&who=ann&mode=editor")
        b = self.open("&who=bob&mode=editor")
        self.go(a, "0007.png")
        a.click("#grpNote_montage")
        a.fill("#grpNote", "drifted")
        a.click("#grpOk")
        a.wait_for_function("document.getElementById('grpDlg').hidden")
        a.click("#grpRej_montage")
        a.click("#grpReasons button:has-text('out of focus / drift')")
        self.assertEqual(a.inner_text("#grpCount"), "2 images of 2 will be rejected.")
        a.click("#grpOk")
        a.wait_for_function("document.getElementById('grpDlg').hidden")
        self.assertEqual(self.saved_reject("0007.png")[1], rg.group_id("montage", HEK, "grid6_1"))
        self.assertEqual(self.saved_reject("0009.png"), (None, None))      # another montage of the sample
        b.evaluate("__review.pull().then(() => __review.render())")
        self.go(b, "0008.png")
        b.wait_for_function("document.getElementById('grpBox').textContent.includes('Montage rejected')")
        self.assertIn("drifted", b.inner_text("#grpBox"))
        self.assertIn("ann", b.inner_text("#grpBox"))
        self.assertEqual(b.inner_text("#rejTag"), "Rejected (montage): out of focus / drift")
        self.assertEqual(self.errors, [])

    def test_phone_width(self):
        page = self.open(**PHONE)
        page.click("#sideBtn")
        page.wait_for_function("document.getElementById('side').classList.contains('show')")
        self.assertFalse(page.locator("#infoSec").is_hidden())
        self.assertFalse(page.locator("#grpSec").is_hidden())
        over = page.evaluate("""() => [...document.querySelectorAll('#side, #grpBox, #infoBox')].filter(e =>
            e.scrollWidth > e.clientWidth + 1).map(e => e.id)""")
        self.assertEqual(over, [])
        box = page.locator("#side").bounding_box()
        self.assertLessEqual(box["x"] + box["width"], 390.5)
        page.locator("#infoSec").scroll_into_view_if_needed()
        shot(page, "r12_phone_side_panel")
        page.click("#grpRej_sample")
        dlg = page.locator("#grpDlg .box").bounding_box()
        self.assertLessEqual(dlg["x"] + dlg["width"], 390.5)
        self.assertGreaterEqual(dlg["x"], 0)
        page.click("#grpReasons button:has-text('empty')")
        shot(page, "r12_phone_reject_dialog")
        page.click("#grpCancel")
        self.assertTrue(page.locator("#grpDlg").is_hidden())
        self.assertEqual(self.saved_reject("0001.png"), (None, None))       # cancel writes nothing
        self.assertEqual(self.errors, [])

    def test_older_server_hides_the_new_controls(self):
        """Control for the caps gate: a server on older code (caps i, t only; no /api/meta or /api/group)."""
        def old(page):
            page.route("**/api/ping", lambda r: r.fulfill(status=200, content_type="application/json",
                                                          body=json.dumps({"ok": True, "caps": ["i", "t"]})))
        page = self.open(setup=old, wait_meta=False)
        page.wait_for_timeout(500)
        self.assertIsNone(page.evaluate("__review.meta()"))
        self.assertTrue(page.locator("#infoSec").is_hidden())
        self.assertEqual(page.locator("#grpRej_sample, #grpNote_sample, #grpUndo_sample").count(), 0)
        self.assertFalse(page.locator("#smpNext").is_hidden())             # navigation needs no server feature
        self.assertEqual(self.errors, [])


if __name__ == "__main__":
    unittest.main()
