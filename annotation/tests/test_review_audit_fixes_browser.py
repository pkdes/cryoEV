"""
Browser tests (Playwright, headless Edge or Chrome; skipped without one) for the label-audit fixes in
annotation/review_tool/index.html:

- #9  Editor: a prediction accepted and then reshaped (enlarged) keeps its outline when it is clicked again and a
      class chip or L is used. The click picks the accepted label (hitTest skips predictions whose label is shown),
      and acceptPred / acceptLow reuse the accepted outline (Labels layer hidden: the click picks the prediction);
- #12 Quick check: Low conf on an undecided card also decides it (a label card keep:<cls>; a September prediction
      card add:<cls> with its own outline), so Accept moves it out of the queue for good; one rewind undoes both;
- #11 the Editor canvas follows its stage when only the header's height changes (ResizeObserver);
- V3  the "Working on" bar, the tab title (D.title || D.id) and the Reviewed button's tooltip.

Control: REVIEW_TOOL_INDEX=<an index.html> serves that page instead of the current one. With the page before these
fixes (git show 17ba3e8:annotation/review_tool/index.html) every test here except the compare-package ones fails.

  python -m unittest discover -s annotation/tests -p "test_review_audit_fixes_browser.py" -v
"""
import json
import math
import os
import shutil
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_review_tool_browser as trb  # noqa: E402  (module: its TestCases must not be collected twice)
from test_review_server import LiveServer  # noqa: E402

OVERRIDE = os.environ.get("REVIEW_TOOL_INDEX")
DESK = {"viewport": {"width": 1280, "height": 800}}
XSS = '<img src=x onerror="window.__xss=1"><script>window.__xss=2</script>&amp;'
LAB_TAIL = "Your decisions are saved in this package. They become labels when an admin applies them to a new label set."
PRED_TAIL = "Your decisions update this analysis' results."
REVIEWED_TIP = "Marks this image as done. It does not apply anything to a label set by itself."


sys.path.append(str(Path(__file__).resolve().parents[2] / "tests"))
from cryoev_testkit import browser  # noqa: E402  (the gate skips these when no UI file changed)


def with_fields(**kw):
    """data.js as served, plus these fields (as a builder would write them)."""
    def setup(context, page):
        page.add_init_script("Object.defineProperty(window, 'REVIEW_DATA', {configurable: true, "
                             "get() { return this.__rd; }, set(d) { Object.assign(d, %s); this.__rd = d; }});" % json.dumps(kw))
    return setup


def area(r):
    return abs(sum(r[i] * r[(i + 3) % len(r)] - r[(i + 2) % len(r)] * r[i + 1] for i in range(0, len(r), 2))) / 2


@browser
class Base(trb.Base):
    @classmethod
    def make_server(cls):
        srv = LiveServer()
        if OVERRIDE:  # control run: the page under test is another index.html (the server reads it per request)
            shutil.copy(OVERRIDE, srv.pkg / "index.html")
        return srv


class CardBase(Base, trb.CardBase):
    def left(self, page):
        return int(page.inner_text("#triProg").split(" of ")[0])

    def here(self, page):
        c = self.card(page)
        return self.file_of(c), c["key"]


# ---------- #9: a reshaped accept survives a click + class pick / L ----------
class ReshapeBase(Base):
    F = "fixB_old.png"  # p1: a prediction with no label; this old-format package has no predicted class (EV)

    def enlarge(self, page):
        """Accept p1 (A), then drag four upper-half vertices out to 1.6x the radius. -> (prediction ring, saved ring)"""
        self.go(page, self.F)
        ring = self.image(self.F)["pr"][1]["p"][0]
        trb.EditAcceptTests.zoom_on(self, page, ring)
        self.click_obj(page, ring)
        self.assertEqual(self.state(page)["sel"], {"kind": "pr", "i": 1})
        page.keyboard.press("a")
        self.assertEqual(self.state(page)["sel"], {"kind": "gt", "key": "p1"})
        n = len(ring) // 2
        cx, cy = sum(ring[0::2]) / n, sum(ring[1::2]) / n
        for k in (11, 13, 15, 17):  # upper half (the class chips cover the bottom of the stage)
            x, y = ring[2 * k], ring[2 * k + 1]
            page.mouse.move(*self.scr(page, x, y)); page.mouse.down()
            page.mouse.move(*self.scr(page, cx + (x - cx) * 1.6, cy + (y - cy) * 1.6), steps=5); page.mouse.up()
            time.sleep(0.6)  # the next press must not count as a double-click
        self.synced(page)
        d = self.saved_dec(self.F)["p1"]
        self.assertEqual(d["v"], "add:EV")
        big = d["p"][0]
        self.assertGreater(area(big), area(ring) * 1.1)  # enlarged: before the fix the smaller prediction won the click
        page.keyboard.press("Escape")
        self.assertIsNone(self.state(page)["sel"])
        return ring, d["p"]

    def after(self, page, big, cls, low=True):
        self.synced(page)
        d = self.saved_dec(self.F)
        self.assertEqual((d["p1"]["v"], d["p1"].get("p")), ("add:" + cls, big))
        if low:
            self.assertEqual(d["t:p1"]["v"], "lowconf")
        self.assertEqual(self.obj(page, self.F, src="p1")["p"], big)
        self.check_js_equals_python(page)


class ReshapeChipThenLTests(ReshapeBase):
    def test_click_chip_and_l_keep_the_enlarged_outline(self):
        page = self.open(**DESK)
        ring, big = self.enlarge(page)
        self.click_obj(page, ring)  # inside both outlines
        self.assertEqual(self.state(page)["sel"], {"kind": "gt", "key": "p1"})  # the accepted label, not the prediction
        page.locator("#chipBar button", has_text="3 MV").click()
        page.keyboard.press("l")
        self.after(page, big, "MV")


class ReshapeLTests(ReshapeBase):
    def test_click_and_l_keep_the_enlarged_outline(self):
        page = self.open(**DESK)
        ring, big = self.enlarge(page)
        self.click_obj(page, ring)
        page.keyboard.press("l")
        self.after(page, big, "EV")


class ReshapeLabelsHiddenTests(ReshapeBase):
    """Labels layer off: the click can only pick the prediction, so acceptPred / acceptLow themselves must keep it."""

    def test_accept_paths_reuse_the_accepted_outline(self):
        page = self.open(**DESK)
        ring, big = self.enlarge(page)
        page.uncheck("#showGt")
        self.click_obj(page, ring)
        self.assertEqual(self.state(page)["sel"], {"kind": "pr", "i": 1})
        page.keyboard.press("3")  # setCat -> acceptPred(MV)
        self.after(page, big, "MV", low=False)
        page.keyboard.press("Escape")
        self.click_obj(page, ring)
        self.assertEqual(self.state(page)["sel"], {"kind": "pr", "i": 1})
        page.keyboard.press("l")  # acceptLow
        self.after(page, big, "MV")
        page.keyboard.press("Escape")
        self.click_obj(page, ring)
        page.keyboard.press("a")  # A: acceptPred, the predicted class (none here: the current class, MV)
        self.after(page, big, "MV")
        page.check("#showGt")


# ---------- #12: Low conf on an undecided card decides it ----------
class QuickLowTests(CardBase):
    F = "fixB_old.png"
    P1 = [[round((180 + 19 * math.cos(2 * math.pi * k / 12)) / 360, 5), round((214 + 15 * math.sin(2 * math.pi * k / 12)) / 256, 5)][j]
          for k in range(12) for j in (0, 1)]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        t = int(time.time() * 1000)  # p1 carries a September decision with its own outline: a card until confirmed
        seed = {"v": "add:EV", "p": [cls.P1], "t": t, "who": "Pujan", "seed": "september:old_b.png:p4"}
        assert cls.srv.put(cls.F, {"t": t, "dec": {"p1": seed}})[0] == 200

    def tag_accept_and_check(self, page, key):
        """L on the card, then Accept: the card leaves and never comes back. -> (left before, the L card's pos)"""
        self.go_card(page, self.F, key)
        n0, pos = self.left(page), self.state(page)["tri"]["pos"]
        page.keyboard.press("l")
        self.synced(page)
        self.assertEqual(self.saved_dec(self.F)["t:" + key]["v"], "lowconf")
        self.assertEqual(self.left(page), n0 - 1)  # resolved: no longer counted as left
        page.keyboard.press("a")  # the card was picked on: Accept only moves on
        page.wait_for_function(f"__review.state().tri.pos !== {pos}")
        for _ in range(len(self.state(page)["tri"]["queue"]) + 1):  # through the queue (it wraps): not seen again
            if self.state(page)["tri"]["pos"] >= len(self.state(page)["tri"]["queue"]):
                break
            self.assertNotEqual(self.here(page), (self.F, key))
            page.evaluate("document.getElementById('aSkip').click()")
            time.sleep(0.35)
        return n0, pos

    def test_label_card(self):
        page = self.open("&who=lowtag&mode=cards", **DESK)
        self.wait_card(page)
        n0, pos = self.tag_accept_and_check(page, "g203")  # MV, no matching prediction, undecided
        d = self.saved_dec(self.F)
        self.assertEqual((d["g203"]["v"], d["g203"].get("p"), d["g203"]["who"]), ("keep:MV", None, "lowtag"))
        page.keyboard.press("z")  # one rewind: the tag and the decision together, back on the card
        page.wait_for_function(f"__review.state().tri.pos === {pos}")
        self.synced(page)
        d = self.saved_dec(self.F)
        self.assertEqual((d["g203"]["v"], d["t:g203"]["v"]), (None, None))
        self.assertEqual(self.left(page), n0)
        self.check_js_equals_python(page)

    def test_seeded_prediction_card(self):
        page = self.open("&who=lowtag2&mode=cards", **DESK)
        self.wait_card(page)
        self.tag_accept_and_check(page, "p1")
        d = self.saved_dec(self.F)["p1"]
        self.assertEqual((d["v"], d["p"], d["who"], "seed" in d), ("add:EV", [self.P1], "lowtag2", False))  # outline kept
        page.keyboard.press("z")
        self.synced(page)
        d = self.saved_dec(self.F)
        self.assertEqual(d["t:p1"]["v"], None)
        self.assertEqual((d["p1"]["v"], d["p1"]["p"]), ("add:EV", [self.P1]))  # the value before (undo writes it fresh)
        self.check_js_equals_python(page)


class QuickLowSeedTests(CardBase):
    """#12 controls: L never rewrites a class decision, and a September n/g seed's outline survives it."""
    F = "fixC_old.png"
    RING = lambda cx, cy, r: [v for k in range(12) for v in (round((cx + r * math.cos(2 * math.pi * k / 12)) / 360, 5),
                                                             round((cy + r * math.sin(2 * math.pi * k / 12)) / 256, 5))]
    N1 = [RING(60, 200, 15)]
    G1 = [RING(120, 100, 30)]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        t = int(time.time() * 1000)
        seeds = {"nsept1": {"v": "new:EV", "p": cls.N1, "t": t, "who": "Pujan", "seed": "september:old_c.png:n1"},
                 "g301": {"v": "keep:non-EV", "p": cls.G1, "t": t, "who": "Pujan", "seed": "september:old_c.png:g9"}}
        assert cls.srv.put(cls.F, {"t": t, "dec": seeds})[0] == 200

    def test_seeded_n_and_g_keep_their_outline(self):
        page = self.open("&who=lowseed&mode=cards", **DESK)
        self.wait_card(page)
        for key, want_v, want_p in (("nsept1", "new:EV", self.N1), ("g301", "keep:non-EV", self.G1)):
            self.go_card(page, self.F, key)
            page.keyboard.press("l")
            self.synced(page)
            d = self.saved_dec(self.F)
            self.assertEqual(d["t:" + key]["v"], "lowconf")
            self.assertEqual((d[key]["v"], d[key]["p"], d[key]["who"], "seed" in d[key]), (want_v, want_p, "lowseed", False))

    def test_l_on_a_decided_card_writes_only_the_tag(self):
        page = self.open("&who=lowctl&mode=classes", **DESK)
        self.wait_card(page)
        f, key = "fixB_old.png", "g202"
        self.go_card(page, f, key)
        page.locator("#triCls button:has-text('MV')").first.click()  # a class decision first
        self.synced(page)
        before = dict(self.saved_dec(f)[key])
        self.assertEqual((before["v"], before["who"]), ("keep:MV", "lowctl"))
        time.sleep(0.05)
        page.keyboard.press("l")
        self.synced(page)
        d = self.saved_dec(f)
        self.assertEqual(d["t:" + key]["v"], "lowconf")
        self.assertEqual(d[key], before)  # v, t and who untouched


# ---------- #11: the canvas follows the stage when only the header grows ----------
class StageResizeTests(Base):
    SIZE = "(() => { const c = document.getElementById('cv'), r = c.getBoundingClientRect(); return {w: c.width, h: c.height, cw: r.width, ch: r.height, dpr: devicePixelRatio}; })()"

    def test_taller_header_resizes_the_canvas(self):
        page = self.open(device_scale_factor=2, **DESK)
        f = "fixC_old.png"
        self.go(page, f)
        s0 = page.evaluate(self.SIZE)
        self.assertEqual((s0["w"], s0["h"]), (int(s0["cw"] * 2), int(s0["ch"] * 2)))  # control: right at the start
        page.evaluate("const d = document.createElement('div'); d.id = 'tallBanner'; d.style.cssText = 'flex:1 1 100%;height:140px';"
                      "d.textContent = 'banner'; document.querySelector('header').appendChild(d)")
        time.sleep(0.5)
        s1 = page.evaluate(self.SIZE)
        self.assertLessEqual(s1["ch"], s0["ch"] - 140)  # the stage got shorter (140 px + the header gap); same window
        self.assertEqual((s1["w"], s1["h"]), (int(s1["cw"] * s1["dpr"]), int(s1["ch"] * s1["dpr"])))
        g = self.image(f)["gt"][1]  # 302
        ring = g["p"][0]
        n = len(ring) // 2
        cx, cy = sum(ring[0::2]) / n, sum(ring[1::2]) / n
        self.click_obj(page, ring)
        self.assertEqual(self.state(page)["sel"], {"kind": "gt", "key": "g302"})
        page.evaluate("(() => { const f = document.getElementById('showFill'); if (!f.checked) f.click(); })()")  # fill is off by default
        # what is shown at the clicked point is the selected outline's blue fill (the image itself is grey)
        x, y = self.scr(page, cx, cy)
        box = page.locator("#cv").bounding_box()
        r, gg, b = page.evaluate("([x, y]) => { const c = document.getElementById('cv'), k = c.getBoundingClientRect();"
                                 "return [...c.getContext('2d').getImageData(Math.round(x * c.width / k.width), Math.round(y * c.height / k.height), 1, 1).data].slice(0, 3); }",
                                 [x - box["x"], y - box["y"]])
        self.assertGreater(b - r, 20, (r, gg, b))


# ---------- V3: where this goes ----------
class WhereBarTests(Base):
    def bar(self, page):
        return page.evaluate("document.getElementById('whereBar').textContent")

    def test_labeling_package(self):
        page = self.open(**DESK)
        self.assertEqual(self.bar(page), f"Working on: synthetic · 4 images · {LAB_TAIL}")  # fixture: source "synthetic"
        self.assertTrue(page.locator("#whereBar").is_visible())
        self.assertEqual(page.title(), "fixture_review")  # D.title || D.id
        self.assertEqual(page.get_attribute("#reviewed", "title"), REVIEWED_TIP)
        page.click("#toTri")
        self.assertIn(page.title(), ("eVHarmony", "Gridr", "cryonder"))  # the cards keep the brand
        page.click("#triExit")
        self.assertEqual(page.title(), "fixture_review")

    def test_compare_package_folder_names_never_a_path(self):
        for src, want in ((r"C:\Users\someone\CryoAI-laptop\labels\lv_20261030_0909\labels.coco.json", "lv_20261030_0909"),
                          ("/home/someone/CryoAI-laptop/roboflow_pull_20261029_1811/current.coco.json", "roboflow_pull_20261029_1811"),
                          (r"D:\data\lv_20261101_1200", "lv_20261101_1200")):
            page = self.open(setup=with_fields(kind="compare", source=src, title="Cell compare v5"), **DESK)
            t = self.bar(page)
            self.assertEqual(t, f"Working on: {want} · 4 images · {LAB_TAIL}")
            for bad in ("Users", "home", "CryoAI", ":\\", "decoy"):
                self.assertNotIn(bad, t)
            self.assertEqual(page.title(), "Cell compare v5")

    def test_source_title_is_text(self):
        page = self.open(setup=with_fields(kind="compare", source="x/lv_1/labels.coco.json", source_title=XSS, title=XSS), **DESK)
        self.assertEqual(page.evaluate("document.querySelector('#whereBar b').textContent"), XSS)
        self.assertEqual(page.title(), XSS)
        self.assertIsNone(page.evaluate("window.__xss"))

    def test_source_name_skips_a_split_folder(self):
        for src in ("/data/roboflow - X/train/_annotations.coco.json", r"D:\data\roboflow - X\valid\_annotations.coco.json"):
            page = self.open(setup=with_fields(kind="compare", source=src), **DESK)
            t = self.bar(page)
            self.assertEqual(t, f"Working on: roboflow - X · 4 images · {LAB_TAIL}")
            self.assertNotIn("train", t)

    def test_prediction_package(self):
        page = self.open(setup=with_fields(kind="prediction", cand_conf=0.3, label_conf=0.5,
                                           source="prediction_set:ps_20261110_f5d3a1"), **DESK)
        self.assertEqual(self.bar(page), f"Working on: ps_20261110_f5d3a1 · 4 images · {PRED_TAIL}")
        self.assertEqual(page.title(), "fixture_review")


if __name__ == "__main__":
    unittest.main()
