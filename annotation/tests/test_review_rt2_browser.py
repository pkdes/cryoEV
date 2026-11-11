"""
Browser tests (Playwright; skipped without a browser) for the review-tool polish in annotation/review_tool/index.html:
#6 Quick check Back, #7 closer zoom + wheel zoom, #10 Redraw outline button + banner, #13 fill toggle + faded old
outline, #8 "Saved" state + Show original. Control: the saved records of a standard accept / reshape flow equal
those the page at main (8ff2601) writes (REVIEW_TOOL_BASE_INDEX=<an index.html> overrides that page).

  python -m unittest discover -s annotation/tests -p "test_review_rt2_browser.py" -v
"""
import os
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_review_tool_browser as trb  # noqa: E402
from test_review_server import LiveServer  # noqa: E402
from test_review_audit_fixes_browser import with_fields  # noqa: E402

DESK = {"viewport": {"width": 1280, "height": 800}}
F = "fixB_old.png"
PIX = """([px, py]) => { const c = document.getElementById('cv'), b = c.getBoundingClientRect(), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  const x = px - b.left, y = py - b.top; let best = 0; for (let j = -2; j <= 2; j++) for (let i = -2; i <= 2; i++) { const o = ((Math.round(y) + j) * c.width + Math.round(x) + i) * 4;
  best = Math.max(best, Math.min(d[o], d[o+1], d[o+2])); } return best; }"""
RGBA = "([x, y]) => { const b = document.getElementById('cv').getBoundingClientRect(); return Array.from(document.getElementById('cv').getContext('2d').getImageData(Math.round(x - b.left), Math.round(y - b.top), 1, 1).data); }"
YELLOW = """() => { const c = document.getElementById('cv'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0;
  for (let i = 0; i < d.length; i += 4) if (d[i] > 220 && d[i+1] > 180 && d[i+2] < 90) n++; return n; }"""


class BackTests(trb.CardBase):
    def test_back_returns_to_the_previous_card_and_keeps_other_decisions(self):
        page = self.open("&who=back&mode=classes", setup=with_fields(kind="prediction", cand_conf=0.3, label_conf=0.5), **DESK)
        self.wait_card(page)
        self.assertTrue(page.locator("#triBack").is_disabled())
        self.assertIn("B back", page.inner_text("#triKeys"))  # the key legend (prediction packages show it)
        keys = []
        for _ in range(2):  # pick a class on two cards, moving on after each
            c, pos = self.card(page), self.state(page)["tri"]["pos"]
            keys.append((self.file_of(c), c["key"]))
            page.locator("#triCls button:not(.cur)").first.click()
            self.synced(page)
            page.keyboard.press("ArrowRight")
            page.wait_for_function(f"__review.state().tri.pos !== {pos}")
        pos2 = self.state(page)["tri"]["pos"]
        d0, d1 = self.saved_dec(keys[0][0])[keys[0][1]], self.saved_dec(keys[1][0])[keys[1][1]]
        page.keyboard.press("b")
        page.wait_for_function(f"__review.state().tri.pos < {pos2}")
        c = self.card(page)
        self.assertEqual((self.file_of(c), c["key"]), keys[1])
        self.assertIn("✓", page.inner_text("#triBio"))  # its pick is shown, and the decided card is not skipped
        time.sleep(0.5)
        self.assertEqual(self.card(page)["key"], keys[1][1])
        page.locator("#triCls button:not(.cur)").nth(1).click()  # change the pick: only this decision changes
        self.synced(page)
        self.assertNotEqual(self.saved_dec(keys[1][0])[keys[1][1]]["v"], d1["v"])
        self.assertEqual(self.saved_dec(keys[0][0])[keys[0][1]], d0)
        pos1 = self.state(page)["tri"]["pos"]
        page.keyboard.press("Backspace")
        page.wait_for_function(f"__review.state().tri.pos < {pos1}")
        c = self.card(page)
        self.assertEqual((self.file_of(c), c["key"]), keys[0])
        self.check_js_equals_python(page)

    def test_back_then_adjust_the_outline(self):
        page = self.open("&who=backadj&mode=cards", **DESK)
        self.wait_card(page)
        first = self.card(page)
        page.keyboard.press("n")  # skip: Back returns
        page.wait_for_function("__review.state().tri.pos > 0")
        page.locator("#triBack").click()
        page.wait_for_function("__review.state().tri.pos === 0")
        self.assertEqual(self.card(page)["key"], first["key"])
        page.keyboard.press("e")  # the existing Adjust
        self.assertTrue(page.locator("#triAdjBar").is_visible())
        self.assertEqual(page.inner_text("#adjRedraw"), "✏️ Redraw outline")


class ZoomTests(trb.CardBase):
    def test_closest_step_and_wheel(self):
        page = self.open("&who=zoom&mode=cards", **DESK)
        self.wait_card(page)
        s = lambda: self.state(page)["tri"]
        self.assertEqual(s()["zoom"], 0)
        s0 = s()["xfs"]
        seen = []
        for _ in range(4):
            page.keyboard.press(" ")
            seen.append((s()["zoom"], s()["xfs"]))
        self.assertEqual([z for z, _ in seen], [1, 2, 3, 0])
        self.assertGreater(seen[2][1], s0)  # closest = a larger scale than the close-up
        self.assertAlmostEqual(seen[3][1], s0, places=6)
        box = page.locator("#tcv").bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + 200)
        page.mouse.wheel(0, -300)
        page.wait_for_function("__review.state().tri.wk > 1")
        self.assertGreater(s()["xfs"], s0)
        for _ in range(4):
            page.mouse.wheel(0, 300)
            time.sleep(0.15)
        page.wait_for_function("__review.state().tri.wk < 1")
        page.keyboard.press(" ")  # a zoom step resets the wheel zoom
        self.assertEqual(s()["wk"], 1)


class RedrawEditorTests(trb.Base):
    def select(self, page, oid=302, f="fixC_old.png"):
        self.go(page, f)
        o = self.obj(page, f, oid)
        self.click_obj(page, o["p"][0])
        return o

    def test_button_banner_and_fade(self):
        page = self.open(**DESK)
        o = self.select(page)
        self.assertEqual(self.state(page)["sel"]["key"], "g302")  # (tests share a server: each edits its own object)
        btn = page.locator("#redrawBtn")
        self.assertTrue(btn.is_visible())
        self.assertIn("Redraw outline", btn.inner_text())
        self.assertTrue(page.locator("#redrawBan").is_hidden())
        vx, vy = self.scr(page, o["p"][0][0], o["p"][0][1])
        before = page.evaluate(PIX, [vx, vy])
        btn.click()
        st = self.state(page)
        self.assertEqual((st["mode"], st["redrawKey"]), ("draw", "g302"))
        self.assertEqual(page.inner_text("#redrawBan"), "Replacing the outline of MV … (Esc cancels)")
        self.assertLess(page.evaluate(PIX, [vx, vy]), before - 40)  # the old outline is faded
        page.keyboard.press("Escape")
        self.assertTrue(page.locator("#redrawBan").is_hidden())
        page.locator("#redrawBtn").click()  # the button does what E does: keeps class and id
        self.stroke(page, 0.70, 0.55, 30)
        after = self.obj(page, "fixC_old.png", 302)
        self.assertEqual((after["id"], after["c"]), (302, "MV"))
        self.assertNotEqual(after["p"], o["p"])
        self.assertTrue(page.locator("#redrawBan").is_hidden())
        self.check_js_equals_python(page)

    def test_fill_toggle_default_off_and_remembered(self):
        page = self.open(**DESK)
        o = self.select(page, 301)
        self.assertFalse(page.locator("#showFill").is_checked())
        r = o["p"][0]
        sx, sy = self.scr(page, sum(r[0::2]) / (len(r) / 2), sum(r[1::2]) / (len(r) / 2))
        off = page.evaluate(RGBA, [sx, sy])
        page.locator("#showFill").check()
        on = page.evaluate(RGBA, [sx, sy])
        self.assertNotEqual(off, on)
        self.assertGreater(on[2] - on[0], off[2] - off[0])  # bluer with the tint
        page.reload()
        page.wait_for_function("window.__review && __review.state().imgReady")
        self.assertTrue(page.locator("#showFill").is_checked())

    def test_saved_state_and_show_original(self):
        page = self.open(**DESK)
        self.select(page, 202, "fixB_old.png")
        self.assertEqual(page.inner_text("#savedSt"), "")  # unchanged outline: nothing to confirm
        page.keyboard.press("e")
        self.stroke(page, 0.70, 0.55, 30)
        self.synced(page)
        page.wait_for_function("document.getElementById('savedSt').textContent === 'Saved ✓'")
        page.route("**/api/edits*", lambda r: r.abort())  # pending while the server is unreachable
        page.keyboard.press("e")
        self.stroke(page, 0.72, 0.57, 25)
        page.wait_for_function("document.getElementById('savedSt').textContent === 'Saving…'")
        page.unroute("**/api/edits*")
        page.wait_for_function("document.getElementById('savedSt').textContent === 'Saved ✓'", timeout=30000)
        a = page.evaluate(YELLOW)  # Show original: the base outline, dashed yellow
        page.locator("#showOrig").check()
        self.assertGreater(page.evaluate(YELLOW), a + 30)
        page.locator("#showOrig").uncheck()
        self.assertAlmostEqual(page.evaluate(YELLOW), a, delta=15)


class RecordsUnchangedControl(unittest.TestCase):
    """A standard accept + reshape + class flow writes the same decisions (v, p, who) as the page at main."""

    def run_flow(self, index):
        class T(trb.Base):
            @classmethod
            def make_server(cls):
                srv = LiveServer()
                if index:
                    shutil.copy(index, srv.pkg / "index.html")
                return srv

            def runTest(self):
                page = self.open(**DESK)
                self.go(page, F)
                ring = self.image(F)["pr"][1]["p"][0]
                trb.EditAcceptTests.zoom_on(self, page, ring)
                self.click_obj(page, ring)
                page.keyboard.press("a")
                x, y = self.scr(page, ring[22], ring[23])
                page.mouse.move(x, y); page.mouse.down(); page.mouse.move(x + 20, y - 20, steps=5); page.mouse.up()
                time.sleep(0.6)
                page.keyboard.press("2")
                self.synced(page)
                self.result = {k: {"v": d["v"], "p": d.get("p"), "who": d["who"]} for k, d in self.saved_dec(F).items()}

        T.setUpClass()
        try:
            t = T()
            t.runTest()
            return t.result
        finally:
            T.tearDownClass()

    def test_same_records_as_main(self):
        if trb.sync_playwright is None:
            self.skipTest("playwright not installed")
        base = os.environ.get("REVIEW_TOOL_BASE_INDEX")
        if not base:
            out = subprocess.run(["git", "-C", str(HERE), "show", "8ff2601:annotation/review_tool/index.html"],
                                 capture_output=True, check=True)
            base = str(HERE / "_main_index.html")
            Path(base).write_bytes(out.stdout)
            self.addCleanup(lambda: Path(base).unlink())
        new, old = self.run_flow(None), self.run_flow(base)
        self.assertTrue(old and "p1" in old)
        self.assertEqual(new, old)


if __name__ == "__main__":
    unittest.main()
