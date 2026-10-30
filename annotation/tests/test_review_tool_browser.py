"""
Browser tests for annotation/review_tool/index.html, served by review_server.py on the fixture package.
Headless Edge or Chrome through Playwright (pip install playwright; no browser download needed when Edge
is installed). Skipped when neither is available.

Covers: script syntax, classes and hotkeys, queue view, reclass / accept / freehand / tap-vertex / redraw /
vertex drag in the Editor, the phone class review at 400 px with touch, and that the page's labels equal
review_server.effective_labels() on what it saved (also for the real September edits, read-only).

  python -m unittest discover -s annotation/tests -v
"""
import json
import math
import os
import re
import shutil
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_review_server import KEY, LiveServer, load_data, real_package, rs  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

TEMPLATE = HERE.parent / "review_tool" / "index.html"
CLASS_ORDER = ["EV", "Low conf EV", "MV", "inner", "Incomplete", "Overlapping", "VLDL", "non-EV"]
SHOTS = os.environ.get("REVIEW_TEST_SHOTS")  # a folder: keep phone screenshots for a look by eye


def launch(p):
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:  # browser not installed: try the next
            continue
    return None


def norm(labels):
    """Order-free comparable labels (the browser hands back 0 where the file says 0.0)."""
    return sorted((g.get("id") or 0, g.get("src") or "", g["c"], json.dumps([[float(v) for v in r] for r in g["p"]]))
                  for g in labels)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        cls.pw = sync_playwright().start()
        cls.browser = launch(cls.pw)
        if cls.browser is None:
            cls.pw.stop()
            raise unittest.SkipTest("no Edge / Chrome for Playwright")
        cls.srv = cls.make_server()

    @classmethod
    def make_server(cls):
        return LiveServer()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.srv.close()

    def open(self, query="&who=tester&mode=editor", **ctx):
        context = self.browser.new_context(**ctx)
        self.addCleanup(context.close)
        page = context.new_page()
        self.errors = []
        page.on("pageerror", lambda e: self.errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and "404" not in m.text and self.errors.append(m.text))
        page.goto(self.srv.url + query)
        page.wait_for_function("window.__review && __review.state().server && __review.state().imgReady", timeout=15000)
        return page

    def state(self, page):
        return page.evaluate("__review.state()")

    def synced(self, page):
        page.wait_for_function("__review.state().dirty === 0 && document.getElementById('sync').textContent.includes('saved')",
                               timeout=15000)

    def check_js_equals_python(self, page):
        """The page's labels for every image equal review_server.effective_labels() on the server's saved record."""
        self.synced(page)
        saved = self.srv.request("GET", "/api/edits")[1]
        js = page.evaluate("() => Object.fromEntries(__review.D.images.map(im => [im.file, __review.effGt(im)]))")
        for im in self.srv.data["images"]:
            py = rs.effective_labels(im, saved[im["file"]]) if im["file"] in saved else im["gt"]
            self.assertEqual(norm(js[im["file"]]), norm(py), im["file"])

    def go(self, page, file):
        page.click(f'#items .nm[title="{file}"]')
        page.wait_for_function(f"__review.state().imgReady && __review.D.images[__review.state().cur].file === {json.dumps(file)}")

    def scr(self, page, x, y):
        """Normalized image point -> page coordinates on the Editor canvas."""
        st, box = self.state(page), page.locator("#cv").bounding_box()
        v = st["view"]
        return box["x"] + x * st["W"] * v["s"] + v["ox"], box["y"] + y * st["H"] * v["s"] + v["oy"]

    def obj(self, page, file, oid=None, src=None):
        labels = page.evaluate(f"__review.effGt(__review.D.images.find(im => im.file === {json.dumps(file)}))")
        return next((g for g in labels if (oid is not None and g["id"] == oid) or (src and g.get("src") == src)), None)

    def click_obj(self, page, ring):
        cx, cy = sum(ring[0::2]) / (len(ring) / 2), sum(ring[1::2]) / (len(ring) / 2)
        page.mouse.click(*self.scr(page, cx, cy))

    def stroke(self, page, cx, cy, r, n=36):
        """A freehand stroke around a circle (normalized centre, radius in image px)."""
        st = self.state(page)
        pts = [self.scr(page, cx + r * math.cos(2 * math.pi * k / n) / st["W"], cy + r * math.sin(2 * math.pi * k / n) / st["H"])
               for k in range(n + 1)]
        page.mouse.move(*pts[0]); page.mouse.down()
        for p in pts[1:]:
            page.mouse.move(*p, steps=2)
        page.mouse.up()

    def saved_dec(self, file):
        return self.srv.request("GET", "/api/edits")[1].get(file, {}).get("dec", {})

    def image(self, file):
        return next(im for im in self.srv.data["images"] if im["file"] == file)

    def tearDown(self):
        self.assertEqual(getattr(self, "errors", []), [])


class EditorTests(Base):
    def test_script_syntax(self):
        """No node on this PC, so the inline scripts are parsed by the browser's own compiler."""
        page = self.open()
        html = TEMPLATE.read_text(encoding="utf-8")
        scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
        self.assertGreaterEqual(len(scripts), 2)
        for s in scripts:
            self.assertIsNone(page.evaluate("s => { try { new Function(s); return null; } catch (e) { return String(e); } }", s))

    def test_classes_follow_roboflow_order(self):
        page = self.open()
        self.assertEqual(page.evaluate("__review.CATS"), CLASS_ORDER)  # fixture's data.js lists them alphabetically, no VLDL
        self.assertEqual(page.locator("#chipBar button").all_inner_texts(), [f"{k + 1} {c}" for k, c in enumerate(CLASS_ORDER)])
        self.assertEqual(page.locator("#flagLegend .sw").count(), 5)

    def test_queue_view(self):
        page = self.open()
        names = lambda: page.locator("#items .nm").evaluate_all("els => els.map(e => e.title)")
        self.assertEqual(names(), ["fixA_new.png", "fixB_old.png", "fixC_old.png"])  # most flags first (A: fx + flc)
        page.select_option("#fSort", "pkg")
        self.assertEqual(names(), ["fixB_old.png", "fixA_new.png", "fixC_old.png"])
        page.select_option("#fSplit", "valid")
        self.assertEqual(names(), ["fixB_old.png"])
        page.select_option("#fSplit", "")
        row_a = page.locator("#items .it", has=page.locator('.nm[title="fixA_new.png"]'))
        self.assertIn("1 class disagreements", row_a.locator(".fl").get_attribute("title"))
        # claims by someone else show in the row
        self.srv.request("POST", "/api/claim?c=fixC_old.png%7Cp0&who=bob", b"")
        page.wait_for_function("[...document.querySelectorAll('#items .it')].some(r => r.textContent.includes('bob'))", timeout=12000)
        # status filters (other tests may have edited images already: compare counts before / after)
        status = lambda: page.evaluate("Object.fromEntries(__review.D.images.map(im => [im.file, "
                                       "(document.querySelector(`#items .it .nm[title='${im.file}']`) || {}).previousElementSibling?.title]))")
        st0 = status()
        page.select_option("#fStat", "reviewed")
        n_rev = len(names())
        page.select_option("#fStat", "")
        target = next(f for f, s in st0.items() if s != "reviewed")
        self.go(page, target)
        page.keyboard.press("r")  # mark reviewed
        page.select_option("#fStat", "reviewed")
        self.assertEqual(len(names()), n_rev + 1)
        self.assertIn(target, names())
        page.select_option("#fStat", "todo")
        self.assertNotIn(target, names())
        page.select_option("#fStat", "")
        self.check_js_equals_python(page)

    def test_reclass_by_key_and_chip(self):
        page = self.open()
        f = "fixA_new.png"
        self.go(page, f)
        self.click_obj(page, self.image(f)["gt"][2]["p"][0])  # 103, MV, flagged x
        self.assertEqual(self.state(page)["sel"]["kind"], "gt")
        self.assertIn("class disagrees", page.inner_text("#selBox"))
        page.keyboard.press("1")
        self.assertEqual(self.obj(page, f, 103)["c"], "EV")
        page.click("#chipBar button:has-text('Incomplete')")
        self.assertEqual(self.obj(page, f, 103)["c"], "Incomplete")
        self.assertEqual(self.state(page)["curCls"], "Incomplete")
        self.synced(page)
        d = self.saved_dec(f)["g103"]
        self.assertEqual(d["v"], "keep:Incomplete")
        self.assertNotIn("p", d)  # outline unchanged: no polygon carried
        self.check_js_equals_python(page)

    def test_prediction_class_and_accept(self):
        page = self.open()
        f = "fixA_new.png"
        self.go(page, f)
        page.uncheck("#showGt")
        self.click_obj(page, self.image(f)["pr"][2]["p"][0])
        txt = page.inner_text("#selBox")
        self.assertIn("Prediction EV", txt)
        self.assertIn("class disagrees", txt)
        # old format: no class, still works
        self.go(page, "fixB_old.png")
        self.click_obj(page, self.image("fixB_old.png")["pr"][1]["p"][0])
        txt = page.inner_text("#selBox")
        self.assertRegex(txt, r"^Prediction, confidence 0\.52")
        page.keyboard.press("7")  # VLDL: with a prediction selected, accepts it as that class
        self.synced(page)
        self.assertEqual(self.saved_dec("fixB_old.png")["p1"]["v"], "add:VLDL")
        self.check_js_equals_python(page)

    def test_freehand_and_tap_vertex_new_shapes(self):
        page = self.open()
        f = "fixC_old.png"
        self.go(page, f)
        page.keyboard.press("8")  # nothing selected: sets the class for new shapes
        page.keyboard.press("d")
        self.stroke(page, 0.2, 0.75, 15)
        page.keyboard.press("v")
        for x, y in [(0.8, 0.1), (0.9, 0.1), (0.9, 0.25), (0.8, 0.25)]:
            page.mouse.click(*self.scr(page, x, y))
        page.keyboard.press("Enter")
        self.synced(page)
        new = {k: d for k, d in self.saved_dec(f).items() if k[0] == "n"}
        self.assertEqual(len(new), 2)
        for d in new.values():
            self.assertEqual(d["v"], "new:non-EV")
            self.assertTrue(rs._valid_rings(d["p"]))
        rings = sorted(len(d["p"][0]) // 2 for d in new.values())
        self.assertEqual(rings[0], 4)       # the tap-vertex square
        self.assertGreater(rings[1], 6)     # the freehand circle, simplified
        self.assertLess(rings[1], 37)
        self.check_js_equals_python(page)

    def test_redraw_keeps_class_and_id(self):
        page = self.open()
        f = "fixC_old.png"
        self.go(page, f)
        before = self.obj(page, f, 302)
        self.click_obj(page, before["p"][0])
        page.keyboard.press("e")
        st = self.state(page)
        self.assertEqual((st["mode"], st["redrawIdx"] is not None), ("draw", True))
        self.stroke(page, 0.70, 0.55, 30)
        after = self.obj(page, f, 302)
        self.assertEqual((after["id"], after["c"]), (302, "MV"))
        self.assertNotEqual(after["p"], before["p"])
        self.assertEqual(self.state(page)["mode"], "select")
        self.synced(page)
        d = self.saved_dec(f)["g302"]
        self.assertEqual((d["v"], d["p"]), ("keep:MV", after["p"]))
        self.check_js_equals_python(page)

    def test_vertex_drag(self):
        page = self.open()
        f = "fixB_old.png"
        self.go(page, f)
        ring = self.image(f)["gt"][0]["p"][0]  # 201
        self.click_obj(page, ring)
        x, y = self.scr(page, ring[0], ring[1])
        page.mouse.move(x, y); page.mouse.down(); page.mouse.move(x + 25, y, steps=5); page.mouse.up()
        self.synced(page)
        d = self.saved_dec(f)["g201"]
        self.assertEqual(d["v"], "keep:EV")
        self.assertGreater(d["p"][0][0], ring[0])
        self.check_js_equals_python(page)


class PhoneTests(Base):
    PHONE = {"viewport": {"width": 400, "height": 820}, "is_mobile": True, "has_touch": True, "device_scale_factor": 2}

    def test_class_review_at_400px(self):
        page = self.open("&who=phone", **self.PHONE)
        self.assertFalse(page.locator("#tri").is_hidden())  # narrow screens open the cards
        page.select_option("#triMode", "cls")
        self.assertEqual(page.locator("#triCls button").count(), 8)
        for el in ("#triAdj", "#aNope", "#triChips"):
            self.assertTrue(page.locator(el).is_hidden(), el)
        q = self.state(page)["tri"]["queue"]
        cases = sorted((self.srv.data["images"][c["i"]]["file"], c["key"]) for c in q)
        self.assertEqual(cases, [("fixA_new.png", "g103"), ("fixA_new.png", "g104"), ("fixB_old.png", "g202")])
        self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 400)
        box = page.locator("#triCls button").first.bounding_box()
        self.assertGreaterEqual(box["height"], 44)  # thumb-sized
        if SHOTS:
            page.screenshot(path=str(Path(SHOTS) / "phone_class_review.png"))
        st = self.state(page)
        c0 = st["tri"]["queue"][st["tri"]["pos"]]
        page.tap("#triCls button:has-text('MV')")
        page.wait_for_function(f"__review.state().tri.pos !== {st['tri']['pos']}")
        f0 = self.srv.data["images"][c0["i"]]["file"]
        self.synced(page)
        self.assertEqual(self.saved_dec(f0)[c0["key"]]["v"], "keep:MV")
        # swipe left = next, no decision
        st = self.state(page)
        c1 = st["tri"]["queue"][st["tri"]["pos"]]
        card = page.locator("#tcv").bounding_box()
        x, y = card["x"] + card["width"] / 2, card["y"] + card["height"] / 2
        page.mouse.move(x, y); page.mouse.down(); page.mouse.move(x - 160, y, steps=8); page.mouse.up()
        page.wait_for_function(f"__review.state().tri.pos !== {st['tri']['pos']}")
        time.sleep(0.5)
        self.assertNotIn(c1["key"], self.saved_dec(self.srv.data["images"][c1["i"]]["file"]))
        self.check_js_equals_python(page)

    def test_triage_still_works_on_phone(self):
        page = self.open("&who=phone2&mode=cards", **self.PHONE)
        self.assertEqual(self.state(page)["tri"]["mode"], "tri")
        self.assertFalse(page.locator("#aNope").is_hidden())
        self.assertGreater(len(self.state(page)["tri"]["queue"]), 0)
        page.tap("#aSkip")
        self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 400)
        if SHOTS:
            page.screenshot(path=str(Path(SHOTS) / "phone_triage.png"))


class RealPackageTests(Base):
    """The September package's data.js + saved edits, copied (never written in place)."""

    @classmethod
    def make_server(cls):
        cls.real = real_package()
        if not cls.real:
            raise unittest.SkipTest("real cryoai_v4_review package not found")

        def build(pkg):
            pkg.mkdir(parents=True)
            shutil.copy(cls.real / "data.js", pkg / "data.js")
            shutil.copytree(cls.real / "edits", pkg / "edits")
            shutil.copy(TEMPLATE, pkg / "index.html")
        return LiveServer(build)

    def test_september_edits_render_like_python(self):
        page = self.browser.new_page()
        self.addCleanup(page.close)
        self.errors = []
        page.on("pageerror", lambda e: self.errors.append(str(e)))
        page.goto(self.srv.url + "&who=tester&mode=editor")
        page.wait_for_function("window.__review && __review.state().server", timeout=20000)
        self.synced(page)
        saved = self.srv.request("GET", "/api/edits")[1]
        self.assertGreaterEqual(len(saved), 33)
        data = {im["file"]: im for im in load_data(self.real)["images"]}
        js = page.evaluate("files => Object.fromEntries(files.map(f => [f, __review.effGt(__review.D.images.find(im => im.file === f))]))",
                           list(saved))
        for f, rec in saved.items():
            self.assertEqual(norm(js[f]), norm(rs.effective_labels(data[f], rec)), f)
        self.assertEqual(page.locator("#items .it").count(), len(data))


if __name__ == "__main__":
    unittest.main()
