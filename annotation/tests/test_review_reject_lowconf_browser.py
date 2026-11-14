"""
Browser tests (Playwright, headless Edge or Chrome; skipped without one) for round 7 R1 in
annotation/review_tool/index.html: image reject / note and the object Low conf tag (decisions i:reject, i:note,
t:<objkey>; docs/REVIEW_TOOL.md "Reject, notes, low conf").

- caps gate: the page against the HEAD (00a1921) review_server.py: no caps in /api/ping, so no reject / note / Low
  conf control shows, L does nothing and no i: / t: key is ever sent (that server answers 400 to them);
- reject + note: the dialog, the red badge and ✎ in the image list, status rejected + the filter + counts, the
  header tag; Quick check skips the rejected image's cards and gets them back when it is un-rejected;
- XSS strings in the image note and an object note stay text (list, side panel, card bio);
- "Accept as low conf" (Editor and Quick check): add:<cls> + t:p<k> in one step, undone together (Z, ↺);
- the Low conf toggle and note on a label; turning it off on a Low conf EV label writes keep:EV too;
- export -> import into an empty browser: the new decisions round-trip;
- the page's labels (with low_conf / note) equal review_server.effective_labels on what the server saved.
- REVIEW_TEST_SHOTS=<folder>: screenshots r1_*.png at 1280 px (desktop) and 390 / 360 px (phone), light and dark.

  python -m unittest discover -s annotation/tests -p "test_review_reject_lowconf_browser.py" -v
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_review_server as trs  # noqa: E402
import test_review_tool_browser as trb  # noqa: E402  (module: its TestCases must not be collected twice)
from test_review_server import LiveServer, rs  # noqa: E402

HEAD = "00a1921"
SHOTS = os.environ.get("REVIEW_TEST_SHOTS")
XSS = '<img src=x onerror="window.__xss=1"><script>window.__xss=2</script>&amp;'
DESK = {"viewport": {"width": 1280, "height": 800}}
PHONE = {"viewport": {"width": 390, "height": 844}, "is_mobile": True, "has_touch": True, "device_scale_factor": 2}
CAP_CONTROLS = ("#imgSec", "#helpL", "#aLow", "#triImgBtn", "#lowBtn", "#accLowBtn")


sys.path.append(str(Path(__file__).resolve().parents[2] / "tests"))
from cryoev_testkit import browser  # noqa: E402  (the gate skips these when no UI file changed)


def shot(page, name):
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(SHOTS) / f"r1_{name}.png"))


def as_prediction_package(context, page):
    """The prediction-package words and pills (kind, cand_conf), as test_prediction_package_browser sets them."""
    page.add_init_script("Object.defineProperty(window, 'REVIEW_DATA', {configurable: true, get() { return this.__rd; }, "
                         "set(d) { Object.assign(d, {kind: 'prediction', cand_conf: 0.3, label_conf: 0.5}); this.__rd = d; }});")


@browser
class Base(trb.Base):
    def tri_go(self, page, file, key):
        q = self.state(page)["tri"]["queue"]
        n = next(k for k, c in enumerate(q) if self.srv.data["images"][c["i"]]["file"] == file and c["key"] == key)
        page.evaluate(f"__review.triGo({n})")
        return n

    def edge_click(self, page, file, oid):
        """Click inside label oid near its rim (a click at the centre of fixA 101 would pick the inner 102)."""
        g = next(g for g in self.image(file)["gt"] if g["id"] == oid)
        r = g["p"][0]
        xs, ys = r[0::2], r[1::2]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        page.mouse.click(*self.scr(page, cx + (max(xs) - cx) * 0.8, cy))
        page.wait_for_function(f"__review.state().sel && __review.state().sel.key === 'g{oid}'")

    def card_files(self, page, n=25):
        """The images of the cards the queue shows, skipping n times (advance wraps around)."""
        seen = set()
        for _ in range(n):
            st = self.state(page)["tri"]
            if st["pos"] >= len(st["queue"]):
                break
            seen.add(self.srv.data["images"][st["queue"][st["pos"]]["i"]]["file"])
            page.evaluate("document.getElementById('aSkip').click()")
            time.sleep(0.3)
        return seen


class OldServerCapsGateTests(Base):
    """The new page against the server code before round 7 (git show 00a1921): every new control hidden."""

    @classmethod
    def make_server(cls):
        try:
            src = subprocess.run(["git", "show", f"{HEAD}:annotation/review_server.py"], cwd=HERE, capture_output=True,
                                 text=True, check=True, encoding="utf-8").stdout
        except (OSError, subprocess.CalledProcessError):
            raise unittest.SkipTest(f"CONTROL NOT RUN: no git history (git show {HEAD} failed)")
        cls._old_dir = Path(tempfile.mkdtemp(prefix="old_server_"))
        script = cls._old_dir / "review_server.py"
        script.write_text(src, encoding="utf-8")
        was, trs.SERVER = trs.SERVER, script
        try:
            return LiveServer()  # serves the fixture with the CURRENT index.html (make_fixture copies the template)
        finally:
            trs.SERVER = was

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(cls._old_dir, ignore_errors=True)

    def test_old_server_refuses_and_the_page_hides_the_controls(self):
        self.assertEqual(self.srv.request("GET", "/api/ping")[:2], (200, {"ok": True}))  # control: no caps
        code = self.srv.put("fixC_old.png", {"t": 1, "dec": {"i:reject": {"v": "reject:empty", "t": 1, "who": "x", "pending": True}}})[0]
        self.assertEqual(code, 400)  # what the gate protects against
        self.assertEqual(self.srv.put("fixC_old.png", {"t": 1, "dec": {"t:g301": {"v": "lowconf", "t": 1, "who": "x"}}})[0], 400)
        sent = []
        page = self.open("&who=tester&mode=editor", setup=lambda c, p: p.on(
            "request", lambda r: r.method == "PUT" and sent.append(r.post_data or "")), **DESK)
        self.assertEqual(page.evaluate("__review.caps()"), [])
        for el in CAP_CONTROLS:
            self.assertEqual(page.locator(el).count() == 0 or page.locator(el).is_hidden(), True, el)
        self.assertEqual(page.locator("#flagLegend .sw").count(), 5)  # no Low conf tag entry
        self.assertTrue(page.evaluate("document.querySelector('#fStat option[value=rejected]').hidden"))
        self.go(page, "fixA_new.png")
        self.edge_click(page, "fixA_new.png", 101)
        self.assertEqual(page.locator("#lowBtn").count(), 0)
        page.keyboard.press("l")  # L: nothing
        page.keyboard.press("3")  # an old edit still saves
        self.synced(page)
        self.assertEqual(self.saved_dec("fixA_new.png")["g101"]["v"], "keep:MV")
        page.click("#toTri")
        self.assertTrue(page.locator("#aLow").is_hidden() and page.locator("#triImgBtn").is_hidden())
        page.keyboard.press("l")
        time.sleep(0.5)
        self.assertTrue(sent)
        self.assertFalse(any('"i:' in b or '"t:' in b for b in sent), sent)
        self.assertEqual(page.evaluate("getComputedStyle(document.getElementById('banner')).display"), "none")  # no 400
        shot(page, "oldserver_cards_hidden_1280_light")


class RejectNoteTests(Base):
    def test_reject_with_note_badge_filter_and_quick_check(self):
        f = "fixB_old.png"
        page = self.open("&who=tester&mode=editor", **DESK)
        self.assertEqual(sorted(page.evaluate("__review.caps()")), ["a", "g", "i", "m", "t"])  # round 12 added m, g; round 14 a
        page.click("#toTri")
        self.assertIn(f, self.card_files(page))  # control: its cards are in the queue before
        page.click("#triExit")
        self.go(page, f)
        self.assertFalse(page.locator("#imgSec").is_hidden())
        self.assertEqual(page.inner_text("#imgBox .st"), "Not rejected")
        page.click("#imgEdit")
        self.assertFalse(page.locator("#imgDlg").is_hidden())
        self.assertEqual(page.locator("#rejChips button").all_inner_texts(),
                         ["Not rejected", *rs.REJECT_REASONS])
        page.click("#rejChips button:has-text('thick ice')")
        page.fill("#imgNote", XSS)
        page.keyboard.press("r")  # typing in the dialog: no hotkeys (R would mark reviewed)
        shot(page, "reject_dialog_1280_light")
        page.click("#imgSave")
        self.synced(page)
        d = self.saved_dec(f)
        self.assertEqual((d["i:reject"]["v"], d["i:reject"]["who"], d["i:note"]["v"]), ("reject:thick ice", "tester", XSS + "r"))
        self.assertNotIn("reviewed", [k for k, v in (self.srv.request("GET", "/api/edits")[1][f]).items() if v is True])
        row = page.locator("#items .it", has=page.locator(f'.nm[title="{f}"]'))
        self.assertEqual(row.locator(".rejb").inner_text(), "Rejected")
        self.assertEqual(row.locator(".rejb").get_attribute("title"), "Rejected: thick ice")
        self.assertEqual(row.locator(".notem").get_attribute("title"), XSS + "r")
        self.assertEqual(row.locator(".ok").inner_text(), "✕")
        self.assertEqual(page.inner_text("#rejTag"), "Rejected: thick ice")
        self.assertEqual(page.inner_text("#imgNoteShow"), "✎ " + XSS + "r")
        self.assertIsNone(page.evaluate("window.__xss"))
        self.assertEqual(page.locator("#items img, #side img, #imgBox script").count(), 0)
        self.assertIn("1 rejected", page.inner_text("#qSum"))
        page.select_option("#fStat", "rejected")
        self.assertEqual(page.locator("#items .nm").evaluate_all("els => els.map(e => e.title)"), [f])
        page.select_option("#fStat", "todo")
        self.assertNotIn(f, page.locator("#items .nm").evaluate_all("els => els.map(e => e.title)"))
        page.select_option("#fStat", "")
        shot(page, "reject_editor_1280_light")
        # Quick check skips the rejected image; un-rejecting brings its cards back
        page.click("#toTri")
        files = self.card_files(page)
        self.assertTrue(files)
        self.assertNotIn(f, files)
        page.click("#triExit")
        self.go(page, f)
        page.click("#imgEdit")
        page.click("#rejChips button:has-text('Not rejected')")
        page.click("#imgSave")
        self.synced(page)
        self.assertIsNone(self.saved_dec(f)["i:reject"]["v"])
        self.assertTrue(page.locator("#rejTag").is_hidden())
        self.assertEqual(self.saved_dec(f)["i:note"]["v"], XSS + "r")  # the note is its own key
        page.click("#toTri")
        self.assertIn(f, self.card_files(page))
        self.check_js_equals_python(page)


class ImageDialogRaceTests(Base):
    """M1: saving the image dialog writes only what this reviewer changed in it. Another device's reject that arrived
    (pull) while the dialog was open survives a save that only changed the note."""

    def test_save_keeps_a_reject_that_arrived_while_open(self):
        f = "fixB_old.png"
        page = self.open("&who=tester&mode=editor", **DESK)
        self.go(page, f)
        page.click("#imgEdit")  # opened: not rejected, no note
        code = self.srv.put(f, {"t": 1, "dec": {"i:reject": {"v": "reject:thick ice", "t": 1, "who": "bob", "pending": True},
                                               "i:note": {"v": "B note", "t": 1, "who": "bob", "pending": True}}})[0]
        self.assertEqual(code, 200)
        page.evaluate("__review.pull()")
        page.wait_for_function(f"(__review.edits()[{json.dumps(f)}].dec['i:reject'] || {{}}).v === 'reject:thick ice'")
        page.fill("#imgNote", "A note")  # A changes only the note
        page.click("#imgSave")
        self.synced(page)
        d = self.saved_dec(f)
        self.assertEqual((d["i:reject"]["v"], d["i:reject"]["who"]), ("reject:thick ice", "bob"))  # untouched by A
        self.assertEqual((d["i:note"]["v"], d["i:note"]["who"]), ("A note", "tester"))  # A's own change: newest wins
        # and the other way round: A opens, B changes the note, A only un-rejects -> B's note stays
        page.click("#imgEdit")
        self.srv.put(f, {"t": 2, "dec": {"i:note": {"v": "B again", "t": 2, "who": "bob", "pending": True}}})
        page.evaluate("__review.pull()")
        page.wait_for_function(f"__review.edits()[{json.dumps(f)}].dec['i:note'].v === 'B again'")
        page.click("#rejChips button:has-text('Not rejected')")
        page.click("#imgSave")
        self.synced(page)
        d = self.saved_dec(f)
        self.assertEqual((d["i:reject"]["v"], d["i:note"]["v"], d["i:note"]["who"]), (None, "B again", "bob"))


class LowConfTests(Base):
    def test_accept_as_low_conf_in_the_editor_undone_together(self):
        f = "fixB_old.png"
        page = self.open("&who=ann&mode=editor", **DESK)
        self.go(page, f)
        self.click_obj(page, self.image(f)["pr"][1]["p"][0])
        page.wait_for_function("__review.state().sel && __review.state().sel.kind === 'pr'")
        self.assertIn("Accept as low conf", page.inner_text("#accLowBtn"))
        page.click("#accLowBtn")
        self.synced(page)
        d = self.saved_dec(f)
        self.assertEqual((d["p1"]["v"], d["t:p1"]["v"], d["t:p1"]["who"]), ("add:EV", "lowconf", "ann"))
        g = self.obj(page, f, src="p1")
        self.assertEqual((g["c"], g.get("low_conf")), ("EV", True))
        self.assertIn("on", page.get_attribute("#lowBtn", "class"))  # the accepted object is selected, tag on
        shot(page, "lowconf_editor_1280_light")
        self.check_js_equals_python(page)
        page.keyboard.press("z")
        self.synced(page)
        d = self.saved_dec(f)
        self.assertEqual((d["p1"]["v"], d["t:p1"]["v"]), (None, None))  # both reverted by one Z
        self.assertIsNone(self.obj(page, f, src="p1"))
        self.check_js_equals_python(page)

    def test_toggle_and_note_on_labels_and_the_low_conf_ev_alias(self):
        f = "fixA_new.png"
        page = self.open("&who=ann&mode=editor", color_scheme="dark", **DESK)
        self.go(page, f)
        self.edge_click(page, f, 105)
        page.keyboard.press("l")
        self.synced(page)
        self.assertEqual(self.saved_dec(f)["t:g105"]["v"], "lowconf")
        page.fill("#objNote", XSS)
        page.press("#objNote", "Enter")
        self.synced(page)
        t = self.saved_dec(f)["t:g105"]
        self.assertEqual((t["v"], t["note"]), ("lowconf", XSS))
        self.assertEqual(page.input_value("#objNote"), XSS)
        self.assertIsNone(page.evaluate("window.__xss"))
        shot(page, "lowconf_label_note_1280_dark")
        page.click("#lowBtn")  # off: the note stays, as "plain"
        self.synced(page)
        t = self.saved_dec(f)["t:g105"]
        self.assertEqual((t["v"], t["note"]), ("plain", XSS))
        self.assertNotIn("low_conf", self.obj(page, f, 105))
        # S1: the Low conf EV alias picked while the tag is note-only: the tag becomes lowconf, the note kept, one step
        page.keyboard.press("2")
        self.synced(page)
        d = self.saved_dec(f)
        self.assertEqual((d["g105"]["v"], d["t:g105"]["v"], d["t:g105"]["note"]), ("keep:Low conf EV", "lowconf", XSS))
        self.assertIn("on", page.get_attribute("#lowBtn", "class"))
        page.keyboard.press("z")
        self.synced(page)
        d = self.saved_dec(f)
        self.assertEqual((d["t:g105"]["v"], self.obj(page, f, 105)["c"]), ("plain", "EV"))
        # a Low conf EV label reads as low conf; turning it off makes it EV
        self.click_obj(page, next(g for g in self.image(f)["gt"] if g["id"] == 104)["p"][0])
        page.wait_for_function("__review.state().sel && __review.state().sel.key === 'g104'")
        self.assertIn("on", page.get_attribute("#lowBtn", "class"))
        page.click("#lowBtn")
        self.synced(page)
        d = self.saved_dec(f)
        self.assertEqual((d["g104"]["v"], d["t:g104"]["v"]), ("keep:EV", None))
        self.assertEqual(self.obj(page, f, 104)["c"], "EV")
        page.keyboard.press("z")  # one step: class and tag back
        self.synced(page)
        self.assertEqual(self.obj(page, f, 104)["c"], "Low conf EV")
        self.check_js_equals_python(page)
        labels = {g.get("id"): g for g in rs.effective_labels(self.image(f), self.srv.request("GET", "/api/edits")[1][f])}
        self.assertEqual(labels[105].get("note"), XSS)

    def test_quick_check_accept_as_low_conf_and_rewind(self):
        f = "fixB_old.png"
        page = self.open("&who=phone&mode=cards", setup=as_prediction_package, **PHONE)
        self.tri_go(page, f, "p1")
        self.assertFalse(page.locator("#aLow").is_hidden())
        self.assertIn("Low conf", page.inner_text("#aLow"))
        self.assertIn("L accept as low conf", page.inner_text("#triKeys"))
        self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 390)
        shot(page, "quickcheck_lowconf_390_light")
        page.tap("#aLow")  # the card flies off first, then the decisions are made
        page.wait_for_function("((__review.edits()['fixB_old.png'] || {}).dec || {})['t:p1'] && __review.edits()['fixB_old.png'].dec['t:p1'].v === 'lowconf'")
        self.synced(page)
        d = self.saved_dec(f)
        self.assertEqual((d["p1"]["v"], d["t:p1"]["v"]), ("add:EV", "lowconf"))
        page.tap("#triUndo")
        self.synced(page)
        d = self.saved_dec(f)
        self.assertEqual((d["p1"]["v"], d["t:p1"]["v"]), (None, None))
        # a label card: L toggles the tag and the bio says so (note as text)
        self.tri_go(page, f, "g203")
        page.keyboard.press("l")
        self.synced(page)
        self.assertEqual(self.saved_dec(f)["t:g203"]["v"], "lowconf")
        self.assertIn("on", page.get_attribute("#aLow", "class"))
        self.assertEqual(page.inner_text("#triLow"), "◐ Low conf")
        self.check_js_equals_python(page)


class ExportImportTests(Base):
    def test_round_trip(self):
        f = "fixC_old.png"
        page = self.open("&who=ann&mode=editor", **DESK)
        self.go(page, f)
        page.click("#imgEdit")
        page.click("#rejChips button:has-text('empty')")
        page.fill("#imgNote", "nothing here")
        page.click("#imgSave")
        self.click_obj(page, self.image(f)["gt"][1]["p"][0])
        page.keyboard.press("l")
        self.synced(page)
        with page.expect_download() as dl:
            page.click("#export")
        exported = json.loads(Path(dl.value.path()).read_text(encoding="utf-8"))
        dec = exported["images"][f]["decisions"]
        self.assertEqual({k: dec[k]["v"] for k in ("i:reject", "i:note", "t:g302")},
                         {"i:reject": "reject:empty", "i:note": "nothing here", "t:g302": "lowconf"})
        # import into a browser whose server has nothing (GET /api/edits -> {}; uploads only echoed)
        def empty_server(context, p):
            p.route(re.compile(r".*/api/edits(\?.*)?$"), lambda r: r.fulfill(json={}) if r.request.method == "GET"
                    else r.fulfill(json={"edit": json.loads(r.request.post_data)}))  # an echo: nothing reaches the server
        fresh = self.open("&who=bob&mode=editor", setup=empty_server, **DESK)
        self.assertEqual(fresh.evaluate("Object.keys(__review.edits())"), [])
        fresh.set_input_files("#importFile", dl.value.path())
        fresh.wait_for_function(f"(__review.edits()[{json.dumps(f)}] || {{}}).dec")
        got = fresh.evaluate(f"__review.edits()[{json.dumps(f)}].dec")
        strip = lambda d: {k: {x: y for x, y in v.items() if x not in ("pending", "server_t")} for k, v in d.items()}
        self.assertEqual(strip(got), strip(dec))
        self.assertEqual(fresh.evaluate(f"__review.statusOf(__review.D.images.find(i => i.file === {json.dumps(f)}))"), "rejected")


class ScreenshotTests(Base):
    """Phone 390 / 360 and desktop 1280, light and dark: the reject dialog, the note, the Low conf controls."""

    def test_screens(self):
        f = "fixD_new.png"
        page = self.open("&who=shots&mode=editor", **DESK)
        self.go(page, f)
        page.click("#imgEdit")
        page.click("#rejChips button:has-text('contaminated')")
        page.fill("#imgNote", "Ice crystals over the left half; vesicles there are not countable.")
        page.click("#imgSave")
        self.click_obj(page, self.image(f)["gt"][0]["p"][0])
        page.keyboard.press("1")  # EV ...
        page.keyboard.press("l")  # ... with the Low conf tag: the amber dots and "low conf"
        page.fill("#objNote", "faint membrane")
        page.press("#objNote", "Enter")
        self.synced(page)
        for scheme in ("light", "dark"):
            p = self.open("&who=shots&mode=editor", color_scheme=scheme, **DESK)
            self.go(p, f)
            p.mouse.click(*self.scr(p, 200 / 360, 130 / 256))
            p.wait_for_function("__review.state().sel")
            shot(p, f"editor_rejected_1280_{scheme}")
            for w in (390, 360):
                ph = self.open("&who=shots&mode=cards", setup=as_prediction_package, color_scheme=scheme,
                               **dict(PHONE, viewport={"width": w, "height": 844}))
                self.tri_go(ph, "fixB_old.png", "p1")
                self.assertLessEqual(ph.evaluate("document.documentElement.scrollWidth"), w)
                shot(ph, f"quickcheck_{w}_{scheme}")
                ph.tap("#triImgBtn")
                self.assertFalse(ph.locator("#imgDlg").is_hidden())
                box = ph.locator("#imgDlg .box").bounding_box()
                self.assertLessEqual(box["x"] + box["width"], w)
                shot(ph, f"image_dialog_{w}_{scheme}")
                ph.click("#imgCancel")
                ph.select_option("#triMode", "cls")
                shot(ph, f"classreview_{w}_{scheme}")
                ed = self.open("&who=shots&mode=editor", color_scheme=scheme, **dict(PHONE, viewport={"width": w, "height": 844}))
                ed.evaluate(f"document.querySelector('#items .nm[title=\"{f}\"]').click()")  # the list is a drawer here
                ed.wait_for_function(f"__review.D.images[__review.state().cur].file === {json.dumps(f)}")
                ed.click("#sideBtn")
                shot(ed, f"editor_side_{w}_{scheme}")
                self.assertLessEqual(ed.evaluate("document.documentElement.scrollWidth"), w)


if __name__ == "__main__":
    unittest.main()
