"""
Browser tests (Playwright, headless Edge or Chrome; skipped without one) for round 16a Stage 1 in
annotation/review_tool/index.html:
  - P1 viewer contrast (display only): the same edits save the same PUT bodies and export with contrast off and with
    Auto + brightness / contrast / gamma on (control: the contrast really was applied); Auto stretches what is on
    screen (a faint image gets brighter, a zoomed-in region gets its own levels) and Reset restores the image exactly;
    settings are per reviewer and remembered; a page that can't read pixels (a tainted canvas) falls back to
    ctx.filter without errors and hides Auto and gamma; the object and tree cards draw through it too;
  - #24: on a Quick check card, after a class pick, Accept's label and title name the picked class;
  - #25a: a mouse drag on the card pans the image (no decision, no zoom click, no peek); touch and pen still swipe;
  - #25b: Tab / Shift+Tab in the Editor select and centre the next / previous object; not while a control has focus.

  python -m unittest discover -s annotation/tests -p test_review_r16a_browser.py -v
"""
import io
import json
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fixture  # noqa: E402
import object_card_fixture as OF  # noqa: E402
import test_review_tool_browser as trb  # noqa: E402
from test_object_card_browser import _Browser  # noqa: E402
from test_review_server import LiveServer  # noqa: E402
from test_review_tool_browser import touch_swipe  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2] / "tests"))
from cryoev_testkit import browser  # noqa: E402

DESK = {"viewport": {"width": 1280, "height": 800}}
FAINT = "fixA_new.png"  # its image is replaced by a faint one (route) where a test needs it
RGBA = ("([x, y]) => { const c = document.getElementById('cv'), b = c.getBoundingClientRect(), d = window.devicePixelRatio || 1; "
        "return Array.from(c.getContext('2d').getImageData(Math.round((x - b.left) * d), Math.round((y - b.top) * d), 1, 1).data); }")
VOLATILE = {"t", "gtT", "reviewedT", "server_t", "exported"}  # clock values: differ between any two runs


def faint_png(w=360, h=256):
    """A dark, low-contrast image: left half ~25, right half ~45 (grey levels), a little noise."""
    rng = np.random.default_rng(3)
    a = np.where(np.arange(w)[None, :] < w // 2, 25.0, 45.0) + rng.normal(0, 2, (h, w))
    buf = io.BytesIO()
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L").save(buf, "PNG")
    return buf.getvalue()


def serve_faint(context, page):
    body = faint_png()
    page.route(re.compile(r".*/images/" + re.escape(FAINT) + r"(\?.*)?$"),
               lambda r: r.fulfill(status=200, content_type="image/png", body=body))


def settings(who, **vc):
    """Contrast settings stored for reviewer `who` before the page loads (what the panel would have saved)."""
    js = "try { localStorage.setItem(%s, %s); } catch (e) {}" % (json.dumps("cryoev-review:contrast:" + who), json.dumps(json.dumps(vc)))
    return lambda context, page: page.add_init_script(js)


def both(*fns):
    def setup(context, page):
        for f in fns:
            f(context, page)
    return setup


def strip(o):
    """A record or export with the clock values zeroed, so two runs of the same edits compare equal."""
    if isinstance(o, dict):
        return {k: (0 if k in VOLATILE else strip(v)) for k, v in o.items()}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def set_range(page, sel, v):
    page.evaluate("([s, v]) => { const e = document.querySelector(s); e.value = String(v); e.dispatchEvent(new Event('input', {bubbles: true})); }", [sel, v])


@browser
class ContrastTests(trb.Base):
    def edits_and_bodies(self, srv, setup, who):
        """The same Editor edits on a fresh package; the last PUT body per image and the export, clock values zeroed."""
        self.srv = srv
        bodies = {}

        def grab(context, page):
            def on(req):
                if req.method == "PUT" and "/api/edits" in req.url:
                    bodies[re.search(r"[?&]f=([^&]+)", req.url).group(1)] = json.loads(req.post_data)
            page.on("request", on)
        page = self.open(f"&who={who}&mode=editor", setup=both(grab, *( [setup] if setup else [])), accept_downloads=True, **DESK)
        f = "fixB_old.png"
        self.go(page, f)
        st = self.state(page)
        self.assertEqual((st["W"], st["H"]), (360, 256))  # the base image's natural size, contrast or not
        self.click_obj(page, self.image(f)["pr"][1]["p"][0])
        page.keyboard.press("a")  # accept the prediction
        self.click_obj(page, self.image(f)["gt"][1]["p"][0])
        page.keyboard.press("3")  # reclass label 202 as MV
        page.keyboard.press("r")  # reviewed
        self.synced(page)
        vc = self.state(page)["vc"]
        with page.expect_download() as dl:
            page.click("#export")
        exp = json.loads(Path(dl.value.path()).read_text(encoding="utf-8"))
        exp.pop("package", None)
        return {k: strip(v) for k, v in bodies.items()}, strip(exp), vc, page

    def test_contrast_on_vs_off_same_put_bodies_and_export(self):
        srv_on = LiveServer()
        self.addCleanup(srv_on.close)
        base_srv = self.srv
        off, exp_off, vc_off, _ = self.edits_and_bodies(base_srv, None, "vcsame")
        on, exp_on, vc_on, page = self.edits_and_bodies(srv_on, settings("vcsame", b=20, c=30, g=1.4, auto=True, smooth=True), "vcsame")
        self.srv = base_srv
        # control: contrast was off in the first run and really applied (a lookup table, Auto levels) in the second
        self.assertIsNone(vc_off["last"])
        self.assertTrue(vc_on["auto"] and vc_on["last"] and "lo" in vc_on["last"] and vc_on["last"]["smooth"], vc_on)
        self.assertTrue(page.evaluate("document.getElementById('vcBtn').classList.contains('on')"))
        self.assertEqual(sorted(off), ["fixB_old.png"])
        self.assertEqual(on, off)
        self.assertEqual(exp_on, exp_off)
        d = off["fixB_old.png"]["dec"]
        self.assertEqual({k: v["v"] for k, v in d.items()}, {"p1": "add:EV", "g202": "keep:MV"})

    def pick_px(self, page, x, y):
        b = page.locator("#cv").bounding_box()
        return page.evaluate(RGBA, [b["x"] + x, b["y"] + y])

    def test_auto_stretches_the_visible_region_and_reset_restores(self):
        page = self.open("&who=vcauto&mode=editor", setup=serve_faint, **DESK)
        self.go(page, FAINT)
        st = self.state(page)
        right = (st["view"]["ox"] + 0.75 * st["W"] * st["view"]["s"], st["view"]["oy"] + 0.3 * st["H"] * st["view"]["s"])
        before = self.pick_px(page, *right)
        self.assertLess(before[0], 60)  # faint and dark, as served
        page.click("#vcBtn")
        self.assertTrue(page.is_visible("#vcPop"))
        self.assertTrue(page.is_visible("#vcAuto") and page.is_visible("#vcG"))
        page.click("#vcAuto")
        after = self.pick_px(page, *right)
        self.assertGreater(after[0], before[0] + 100)  # the stretch: ~45 of 18..52 shows near white
        lv = self.state(page)["vc"]["last"]
        self.assertLess(lv["lo"], 25)
        self.assertGreater(lv["hi"], 45)
        self.assertEqual(page.inner_text("#vcAuto"), "Auto: on")
        # zoomed into the left half (~25 only): Auto takes that region's own levels
        left = (st["view"]["ox"] + 0.2 * st["W"] * st["view"]["s"], st["view"]["oy"] + 0.5 * st["H"] * st["view"]["s"])
        b = page.locator("#cv").bounding_box()
        page.mouse.move(b["x"] + left[0], b["y"] + left[1])
        for _ in range(6):
            page.mouse.wheel(0, -400)
            time.sleep(0.05)
        page.wait_for_function("__review.state().vc.last.hi < 40")
        page.evaluate("document.activeElement.blur()")  # keys in the panel stay in the panel
        page.keyboard.press("f")  # fit again
        # brightness and gamma add on top; Reset restores the delivered image exactly
        set_range(page, "#vcB", 40)
        set_range(page, "#vcG", 2)
        self.assertEqual(self.state(page)["vc"]["b"], 40)
        page.click("#vcReset")
        self.assertEqual({k: self.state(page)["vc"][k] for k in ("b", "c", "g", "auto")}, {"b": 0, "c": 0, "g": 1, "auto": False})
        self.assertEqual(self.pick_px(page, *right), before)
        self.assertFalse(page.evaluate("document.getElementById('vcBtn').classList.contains('on')"))
        # remembered per reviewer: Auto on again, reload -> on; another reviewer -> off
        page.click("#vcAuto")
        page.keyboard.press("Escape")
        self.assertTrue(page.is_hidden("#vcPop"))
        page.reload()
        page.wait_for_function("window.__review && __review.state().imgReady")
        self.assertTrue(self.state(page)["vc"]["auto"])
        page2 = self.open("&who=vcother&mode=editor", **DESK)
        self.assertFalse(self.state(page2)["vc"]["auto"])

    def test_tainted_canvas_falls_back_without_errors(self):
        # an offscreen canvas can't be read (as on a file:// page); the visible canvases still can (for this test)
        taint = ("(() => { const g = CanvasRenderingContext2D.prototype.getImageData; CanvasRenderingContext2D.prototype.getImageData = "
                 "function (...a) { if (!this.canvas.isConnected) throw new DOMException('tainted', 'SecurityError'); return g.apply(this, a); }; })();")
        page = self.open("&who=vctaint&mode=editor", setup=both(serve_faint, lambda c, p: p.add_init_script(taint)), **DESK)
        self.go(page, FAINT)
        st = self.state(page)
        right = (st["view"]["ox"] + 0.75 * st["W"] * st["view"]["s"], st["view"]["oy"] + 0.3 * st["H"] * st["view"]["s"])
        before = self.pick_px(page, *right)
        page.click("#vcBtn")
        page.click("#vcAuto")  # the first read fails: fallback, Auto and gamma hidden, the note shown
        vc = self.state(page)["vc"]
        self.assertTrue(vc["taint"])
        self.assertTrue(page.is_hidden("#vcAuto") and page.is_hidden("#vcG"))
        self.assertTrue(page.is_visible("#vcTaint"))
        set_range(page, "#vcB", 60)
        self.assertEqual(self.state(page)["vc"]["last"], {"filter": True, "smooth": 0})
        self.assertGreater(self.pick_px(page, *right)[0], before[0] + 10)  # ctx.filter brightness applied
        page.click("#toTri")  # the cards draw through the same path
        page.wait_for_function("__review.state().tri.ready")
        page.keyboard.press(" ")
        page.click("#triExit")
        # tearDown: no page errors

    def test_editor_tab_cycles_objects(self):
        page = self.open("&who=tab&mode=editor", **DESK)
        f = "fixC_old.png"
        self.go(page, f)
        page.evaluate("document.activeElement && document.activeElement.blur()")
        seen = []
        for _ in range(5):
            page.keyboard.press("Tab")
            s = self.state(page)["sel"]
            seen.append(s["key"] if s["kind"] == "gt" else "p%d" % s["i"])
        self.assertEqual(seen, ["g301", "p0", "g302", "p1", "g301"])  # reading order of centres, then round again
        page.keyboard.press("Shift+Tab")
        self.assertEqual(self.state(page)["sel"]["i"], 1)
        # centred like focusObj: the object's centre is the stage's centre
        ring = self.image(f)["pr"][1]["p"][0]
        cx, cy = sum(ring[0::2]) / (len(ring) / 2), sum(ring[1::2]) / (len(ring) / 2)
        x, y = self.scr(page, cx, cy)
        b = page.locator("#stage").bounding_box()
        self.assertAlmostEqual(x, b["x"] + b["width"] / 2, delta=2)
        self.assertAlmostEqual(y, b["y"] + b["height"] / 2, delta=2)
        # a focused control keeps Tab for focus moves: the selection stays
        page.focus("#mSel")
        page.keyboard.press("Tab")
        self.assertEqual(self.state(page)["sel"]["i"], 1)
        self.assertNotEqual(page.evaluate("document.activeElement.id"), "mSel")


@browser
class CardTests(trb.CardBase):
    def test_24_accept_names_the_picked_class(self):
        for fun in ("1", "0"):
            with self.subTest(fun=fun):
                page = self.open("&who=acc24&mode=cards", setup=lambda c, p, fun=fun: p.add_init_script(
                    "try { localStorage.setItem('cryoev-review:fun', '%s'); } catch (e) {}" % fun), **self.PHONE)
                self.wait_card(page)
                f = "fixB_old.png"
                self.go_card(page, f, "p1")
                txt = lambda: " ".join(page.inner_text("#aYes").split())
                self.assertTrue(txt().endswith("as EV"), txt())
                self.assertIn("add as EV", page.get_attribute("#aYes", "title"))
                page.locator("#triChips button", has_text=re.compile("^MV$")).tap()
                page.wait_for_function("__review.state().tri.picked")
                self.assertTrue(txt().endswith("as MV"), txt())
                title = page.get_attribute("#aYes", "title")
                self.assertIn("MV", title)
                self.assertNotIn("EV", title.replace("MV", ""))
                self.shot(page, f"r16a_24_fun{fun}.png")
                pos = self.state(page)["tri"]["pos"]
                page.tap("#aYes")  # Accept keeps the pick and moves on
                page.wait_for_function(f"__review.state().tri.pos !== {pos}")
                self.synced(page)
                self.assertEqual(self.saved_dec(f)["p1"]["v"], "add:MV")
                page.keyboard.press("z")  # rewind for the next round
                self.synced(page)
                page.close()

    def test_25a_mouse_drag_pans_touch_and_pen_swipe(self):
        page = self.open("&who=pan&mode=cards", **self.DESK)
        self.wait_card(page)
        self.assertTrue(self.state(page)["fun"])  # fun mode on (the default): the pan does not depend on it
        st = self.state(page)["tri"]
        pos, x0, z0 = st["pos"], st["xf"]["ox"], st["zoom"]
        box = page.locator("#tcv").bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 3
        page.mouse.move(x, y); page.mouse.down(); page.mouse.move(x + 160, y + 30, steps=8)
        time.sleep(0.5)  # longer than the 350 ms hold: a moving mouse never peeks
        self.assertFalse(page.evaluate("document.getElementById('triEye').classList.contains('off')"))
        page.mouse.up()
        time.sleep(0.4)
        st = self.state(page)["tri"]
        self.assertEqual((st["pos"], st["zoom"]), (pos, z0))  # no decision, no zoom click
        self.assertAlmostEqual(st["xf"]["ox"], x0 + 160, delta=1)
        c = self.card(page)
        self.assertNotIn(c["key"], self.saved_dec(self.file_of(c)))
        page.mouse.click(x, y)  # a plain click still steps the zoom (and the pan resets with it)
        st = self.state(page)["tri"]
        self.assertEqual(st["zoom"], (z0 + 1) % 4)
        self.assertEqual(st["wk"], 1)
        for kind in ("touch", "pen"):  # touch and pen: swipe right decides
            pos = self.state(page)["tri"]["pos"]
            c = self.card(page)
            touch_swipe(page, x, y, 160, kind=kind)
            page.wait_for_function(f"__review.state().tri.pos !== {pos}")
            self.synced(page)
            self.assertIn(c["key"], self.saved_dec(self.file_of(c)), kind)


@browser
class ObjectCardContrastTests(_Browser):
    def test_object_and_tree_cards_draw_with_contrast(self):
        from test_tree_card_browser import READY
        for tree in (False, True):
            d = Path(tempfile.mkdtemp(prefix="vc_obj_")) / ("pred_obj_tree_fixture_ab12" if tree else "pred_obj_fixture_ab12")
            OF.build_tree(d) if tree else OF.build(d)
            srv = OF.FakeServer(d).start()
            self.addCleanup(srv.close)
            pg = self.page(srv.url + "?who=rev1")
            pg.wait_for_function(READY, timeout=15000)
            pg.wait_for_function("__review.obj().drawn.length > 0")
            pg.click("#oVc")
            self.assertTrue(pg.is_visible("#vcPop"))
            pg.click("#vcAuto")
            vc = pg.evaluate("__review.state().vc")
            self.assertTrue(vc["auto"] and vc["last"] and "lo" in vc["last"], vc)
            pg.keyboard.press("Escape")
            if tree:  # tree cards keep Tab for their nodes
                n0 = pg.evaluate("__review.obj().tree.node")
                pg.evaluate("document.activeElement && document.activeElement.blur()")
                pg.keyboard.press("Tab")
                self.assertEqual(pg.evaluate("__review.obj().tree.node"), n0 + 1)
            self.assertEqual(self.errors, [])


# ---------------------------------------------------------------- stage 2: the 2x ("hi") images and Smooth
NKEY = re.compile(r'"n[0-9a-z]{6,}"')  # a drawn shape's random key


def last_puts(page_bodies):
    return {f: json.loads(NKEY.sub('"nNEW"', json.dumps(strip(b), sort_keys=True))) for f, b in page_bodies.items()}


def grab_puts(bodies):
    def setup(context, page):
        def on(req):
            if req.method == "PUT" and "/api/edits" in req.url:
                bodies[re.search(r"[?&]f=([^&]+)", req.url).group(1)] = json.loads(req.post_data)
        page.on("request", on)
    return setup


def meta_cap(nm):
    """/api/ping lists the m cap and /api/meta gives every image nm_per_px = nm (what review_meta reads from images.csv)."""
    def setup(context, page):
        def ping(r):
            j = r.fetch().json()
            r.fulfill(status=200, content_type="application/json", body=json.dumps(dict(j, caps=sorted(set(j.get("caps", [])) | {"m"}))))
        page.route(re.compile(r".*/api/ping(\?.*)?$"), ping)
        page.route(re.compile(r".*/api/meta(\?.*)?$"), lambda r: r.fulfill(status=200, content_type="application/json", body=json.dumps(
            {"package": {}, "images": {f: {"nm_per_px": nm} for f in ("fixA_new.png", "fixB_old.png", "fixC_old.png", "fixD_new.png")}})))
    return setup


@browser
class HiEditorTests(trb.Base):
    @classmethod
    def make_server(cls):
        return LiveServer(build=make_fixture.build_hi)  # fixA / fixB with a 2x image, fixC / fixD without (a mixed set)

    def flow(self, page):
        """The same Editor edits: accept a prediction, drag one of its vertices, reclass it, then a freehand shape."""
        f = "fixB_old.png"
        self.go(page, f)
        ring = self.image(f)["pr"][1]["p"][0]
        trb.EditAcceptTests.zoom_on(self, page, ring)
        self.click_obj(page, ring)
        page.keyboard.press("a")
        x, y = self.scr(page, ring[22], ring[23])
        page.mouse.move(x, y); page.mouse.down(); page.mouse.move(x + 20, y - 20, steps=5); page.mouse.up()
        time.sleep(0.6)
        page.keyboard.press("2")
        page.keyboard.press("f")
        page.keyboard.press("d")
        self.stroke(page, 0.3, 0.3, 20)
        self.synced(page)

    def test_put_bodies_same_with_and_without_hi(self):
        runs = {}
        for name, build in (("hi", None), ("plain", make_fixture.build)):
            srv = self.srv if build is None else LiveServer(build=build)
            if build is not None:
                self.addCleanup(srv.close)
            base, self.srv = self.srv, srv
            bodies = {}
            page = self.open("&who=hiput&mode=editor", setup=grab_puts(bodies), **DESK)
            self.flow(page)
            runs[name] = (last_puts(bodies), self.state(page)["hi"], self.state(page)["W"])
            self.srv = base
        # control: the 2x image really was drawn in one run (and not in the other), at the base image's W
        self.assertEqual(runs["hi"][1]["editor"], "images/fixB_old.hi.jpg")
        self.assertEqual(runs["hi"][1]["level"], 720)
        self.assertIsNone(runs["plain"][1]["editor"])
        self.assertEqual(runs["hi"][2], runs["plain"][2])
        self.assertEqual(runs["hi"][0], runs["plain"][0])
        dec = runs["hi"][0]["fixB_old.png"]["dec"]
        self.assertEqual(dec["p1"]["v"], "add:Low conf EV")
        self.assertIn("nNEW", dec)

    def test_mixed_set_lazy_and_nm_per_px(self):
        hi_req = []

        def log(context, page):
            page.on("request", lambda r: ".hi." in r.url and hi_req.append(r.url.split("/")[-1].split("?")[0]))
        page = self.open("&who=hinm&mode=editor", setup=both(log, meta_cap(1.47)), **DESK)
        page.wait_for_function("__review.meta() !== null")
        info = lambda: page.evaluate("(() => { for (const dt of document.querySelectorAll('#infoBox dt')) "
                                     "if (dt.textContent === 'Pixel size') return dt.nextElementSibling.textContent; })()")
        self.go(page, "fixC_old.png")  # no hi: the 1x level
        page.mouse.move(600, 400)
        page.mouse.wheel(0, -600)
        time.sleep(0.3)
        self.assertIsNone(self.state(page)["hi"]["editor"])
        self.assertEqual(info(), "1.47 nm/px here")
        self.go(page, "fixB_old.png")  # hi: drawn once loaded; nm/px of the 2x image
        page.wait_for_function("__review.state().hi.editor === 'images/fixB_old.hi.jpg'")
        self.assertEqual(info(), "0.735 nm/px here · sharper image")
        self.assertEqual(self.state(page)["W"], 360)  # pixel rules: the base image
        self.go(page, "fixC_old.png")
        self.assertEqual(info(), "1.47 nm/px here")
        # never preloaded: only the images opened asked for their 2x image (fixA, the first one, and fixB)
        self.assertTrue(set(hi_req) <= {"fixA_new.hi.jpg", "fixB_old.hi.jpg"}, hi_req)
        self.assertIn("fixB_old.hi.jpg", hi_req)

    def test_old_data_js_and_redraws_reuse_the_table(self):
        page = self.open("&who=hiold&mode=editor", setup=settings("hiold", auto=True, smooth=True), **DESK)
        self.assertEqual(self.state(page)["vc"]["smooth"], 1)  # r16a's saved "smooth": true reads as strength 1.0
        self.go(page, "fixD_new.png")  # an image without hi keys, as in every data.js before stage 2
        self.assertIsNone(self.state(page)["hi"]["editor"])
        page.wait_for_function("__review.state().vc.builds > 0")
        n = self.state(page)["vc"]["builds"]
        for _ in range(10):  # redraws (a save, a pull) of the same view: no new lookup table or histogram
            page.evaluate("__review.render()")
        self.assertEqual(self.state(page)["vc"]["builds"], n)

    def test_smooth_toggle_and_hidden_when_unsupported(self):
        page = self.open("&who=smo&mode=editor", **DESK)
        page.click("#vcBtn")
        self.assertTrue(page.is_visible("#vcS"))
        self.assertEqual(page.inner_text("#vcSo"), "off")
        set_range(page, "#vcS", 2)  # r16c: a strength in full-res px: sigma = 2 x (px per base px) / 4 in the drawn image
        st, k = self.state(page)["vc"], self.state(page)["hi"]["level"] / 360  # fixA: the 2x image once loaded (k = 2)
        self.assertEqual((st["smooth"], st["last"]["sigma"]), (2, 2 * k / 4))
        self.assertEqual(page.inner_text("#vcSo"), "2.00")
        page.click("#vcReset")
        self.assertFalse(self.state(page)["vc"]["smooth"])
        # a page that can't read pixels and has no canvas filter (Safari < 18 on a file:// copy): no Smooth
        nofilter = ("Object.defineProperty(CanvasRenderingContext2D.prototype, 'filter', {configurable: true, get() { return undefined; }, set(v) {}});"
                    "(() => { const g = CanvasRenderingContext2D.prototype.getImageData; CanvasRenderingContext2D.prototype.getImageData = "
                    "function (...a) { if (!this.canvas.isConnected) throw new DOMException('tainted', 'SecurityError'); return g.apply(this, a); }; })();")
        page2 = self.open("&who=smo2&mode=editor", setup=lambda c, p: p.add_init_script(nofilter), **DESK)
        page2.click("#vcBtn")
        set_range(page2, "#vcB", 30)  # the first non-default draw finds the canvas tainted
        st = self.state(page2)["vc"]
        self.assertTrue(st["taint"] and not st["smoothOk"])
        self.assertTrue(page2.is_hidden("#vcSRow"))
        # no canvas filter but readable pixels: the separable kernel in the page, so the slider stays
        page3 = self.open("&who=smo3&mode=editor", setup=lambda c, p: p.add_init_script(nofilter.split(";(() =>")[0] + ";"), **DESK)
        page3.click("#vcBtn")
        self.assertTrue(page3.is_visible("#vcS"))
        set_range(page3, "#vcS", 3)
        st = self.state(page3)["vc"]
        k = self.state(page3)["hi"]["level"] / 360
        self.assertTrue(not st["taint"] and st["last"]["sigma"] == 3 * k / 4 and st["builds"] > 0, st)

    def test_quick_check_card_draws_hi(self):
        page = self.open("&who=hicard&mode=cards", **trb.PhoneTests.PHONE)
        page.wait_for_function("__review.state().tri.ready")
        st = self.state(page)["tri"]
        im = self.srv.data["images"][st["queue"][st["pos"]]["i"]]
        self.assertEqual(im["file"], "fixA_new.png")  # most flags first: an image with hi
        page.wait_for_function("__review.state().hi.tri === 'images/fixA_new.hi.jpg'")
        self.assertEqual(self.state(page)["tri"]["xf"]["Wi"], 360)  # Wi / Hi: the base image's


@browser
class HiObjectCardTests(_Browser):
    def answers(self, hi):
        d = Path(tempfile.mkdtemp(prefix="hi_obj_")) / "pred_obj_fixture_ab12"
        OF.build(d, hi=hi)
        srv = OF.FakeServer(d).start()
        self.addCleanup(srv.close)
        pg = self.page(srv.url + "?who=rev1", width=390, height=844)
        pg.wait_for_function("window.__review && __review.obj() && __review.state().server", timeout=15000)
        seen = []
        for _ in range(6):
            it = self.obj(pg)["item"]
            if it["image"] == "imgA" and hi:
                pg.wait_for_function("__review.obj().hi === 'images/imgA.hi.jpg'")
                seen.append(it["id"])
            for k in ("2", "s", "z"):
                pg.keyboard.press(k)
            pg.keyboard.press("ArrowRight")
        pg.wait_for_function("__review.state().dirty === 0", timeout=10000)
        self.assertNotIn("nm/px", pg.inner_text("#obj"))  # a blind set: no nm/px or scale readout on its cards
        self.assertEqual(self.errors, [])
        return {f: strip(r) for f, r in srv.records.items()}, seen

    def test_answers_same_with_and_without_hi(self):
        plain, _ = self.answers(())
        hi, seen = self.answers(("imgA",))
        self.assertTrue(seen)  # control: the 2x image was drawn on imgA's cards
        self.assertEqual(hi, plain)
        self.assertTrue(any(k.startswith("a:") for r in hi.values() for k in r["dec"]))


if __name__ == "__main__":
    unittest.main()
