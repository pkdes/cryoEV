"""
Browser tests (Playwright, headless Edge or Chrome; skipped without one) for the round-9 review tool changes in
annotation/review_tool/index.html (held branch r9-review):

- #19 Back goes several cards back: a second tap / key during the card's fly-off animation used to decide the NEXT
      card (never shown) and push it to Back's history. Now a busy guard ignores input during the fly-off, and
      Back's history holds case ids;
- R1  Fun mode (default on; off = "Quick check", no pop-up, no quips, "Reject ✕ · Low conf · Accept ✓"): the saved
      decisions are byte-identical either way;
- #18 the side list of the Quick check cards: names, states, click = jump (as Back);
- #7  pinch zoom on the Quick check card (the editor's two-pointer math, shared);
- #10 the multi-point vertex eraser in the card's adjust mode (box / shift-click, Delete, >= 3 points, undo).
- #10 the same eraser in the full Editor (Shift-drag box / shift-click / click, Delete or the button, >= 3 points,
      Z), saved exactly like a right-click vertex delete.

Control: REVIEW_TOOL_INDEX=<an index.html> serves that page instead of the current one (the #19 repro fails with the
page from main 00dab61: git show 00dab61:annotation/review_tool/index.html).

  python -m unittest discover -s annotation/tests -p "test_review_r9_browser.py" -v
"""
import datetime
import json
import os
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_review_tool_browser as trb  # noqa: E402,F401  (module: its TestCases must not be collected twice)
sys.path.append(str(HERE.parent.parent / "tests"))
from cryoev_testkit import browser  # noqa: E402  (the gate skips these when no UI file changed)
from test_review_audit_fixes_browser import CardBase as _AuditCardBase, with_fields  # noqa: E402
from test_review_server import LiveServer  # noqa: E402

DESK = {"viewport": {"width": 1280, "height": 800}}
PHONE = {"viewport": {"width": 400, "height": 820}, "is_mobile": True, "has_touch": True, "device_scale_factor": 2}
PRED = with_fields(kind="prediction", cand_conf=0.3, label_conf=0.5)
FUN_OFF = "try { localStorage.setItem('cryoev-review:fun', '0'); } catch (e) {}"
NO_CAPTURE = "Element.prototype.setPointerCapture = function () {};"  # synthetic pointers have nothing to capture
BRANDS = ("eVHarmony", "Gridr", "cryonder")


def clear_cryoev_env():
    """Drop CRYOEV_* from this process (the review server subprocesses inherit it); -> what to put back."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("CRYOEV_")}
    for k in saved:
        os.environ.pop(k, None)
    return saved


def setups(*fns):
    def setup(context, page):
        for f in fns:
            f(context, page)
    return setup


def init(js):
    return lambda context, page: page.add_init_script(js)


def ptr(page, sel, typ, pid, x, y):
    """A synthetic touch pointer event on `sel` at page coordinates."""
    page.evaluate("([sel, typ, pid, x, y]) => document.querySelector(sel).dispatchEvent(new PointerEvent(typ, "
                  "{pointerId: pid, pointerType: 'touch', clientX: x, clientY: y, bubbles: true, cancelable: true, "
                  "isPrimary: pid === 1}))", [sel, typ, pid, x, y])


def pinch(page, sel, mx, my, d0, d1, steps=6):
    """Two fingers on `sel`, horizontally d0 apart about (mx, my), spread (or squeezed) to d1 apart."""
    ptr(page, sel, "pointerdown", 1, mx - d0 / 2, my)
    ptr(page, sel, "pointerdown", 2, mx + d0 / 2, my)
    for k in range(1, steps + 1):
        d = d0 + (d1 - d0) * k / steps
        ptr(page, sel, "pointermove", 1, mx - d / 2, my)
        ptr(page, sel, "pointermove", 2, mx + d / 2, my)
    ptr(page, sel, "pointerup", 1, mx - d1 / 2, my)
    ptr(page, sel, "pointerup", 2, mx + d1 / 2, my)


@browser
class CardBase(_AuditCardBase):
    @classmethod
    def setUpClass(cls):
        cls._env = clear_cryoev_env()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        os.environ.update(cls._env)

    def setUp(self):
        clear_cryoev_env()

    def queue_keys(self, page):
        return [(self.file_of(c), c["key"]) for c in self.state(page)["tri"]["queue"]]

    def decided_keys(self, page):
        saved = self.srv.request("GET", "/api/edits")[1]
        return [(f, k) for f, k in self.queue_keys(page) if ((saved.get(f) or {}).get("dec") or {}).get(k, {}).get("v")]

    def idle(self, page, cond="true"):
        page.wait_for_function(f"(() => {{ const t = __review.state().tri; return !t.busy && ({cond}); }})()")


@browser
class Back19Tests(CardBase):
    """#19 reproduction. With the page before the fix (REVIEW_TOOL_INDEX=<main's index.html>) both tests fail: the
    second key decides the next card unseen, and Back lands on that card instead of the one just decided."""

    def check_double(self, page, press):
        self.wait_card(page)
        first, before = self.here(page), set(self.decided_keys(page))
        press()  # two inputs inside the 240 ms fly-off
        time.sleep(0.3)
        self.idle(page)  # the fly-off is over (its decision runs in the same task)
        self.synced(page)
        new = [k for k in self.decided_keys(page) if k not in before]
        self.assertEqual(new, [first], "only the card on screen may be decided")
        shown = self.here(page)
        page.keyboard.press("b")
        page.wait_for_function("__review.state().tri.ready")
        self.assertEqual(self.here(page), first, "Back returns to exactly the last card")
        page.keyboard.press("b")  # nothing older: Back stays put
        time.sleep(0.2)
        self.assertEqual(self.here(page), first)
        self.assertNotEqual(shown, first)

    def test_double_key_during_fly_off(self):
        page = self.open("&who=r19k&mode=cards", setup=PRED, **DESK)
        # two keydowns in one task: the second is always inside the 240 ms fly-off, however loaded the machine is
        self.check_double(page, lambda: page.evaluate(
            "() => { for (let i = 0; i < 2; i++) document.dispatchEvent(new KeyboardEvent('keydown', {key: 'x', bubbles: true})); }"))

    def test_double_click_during_fly_off(self):
        page = self.open("&who=r19c&mode=cards", setup=PRED, **DESK)
        self.wait_card(page)
        # the second click lands while the card flies off (same task: always inside the fly-off)
        self.check_double(page, lambda: page.evaluate("() => { const b = document.getElementById('aYes'); b.click(); b.click(); }"))

    def test_editor_exit_during_fly_off(self):
        """Editor (triExit) during the fly-off is ignored: the card's own decision lands, the cards stay open. And if
        the cards are hidden meanwhile by another route, the decision never runs off screen."""
        page = self.open("&who=r19e&mode=cards", setup=PRED, **DESK)
        self.wait_card(page)
        first, before = self.here(page), set(self.decided_keys(page))
        page.evaluate("() => { document.getElementById('aNope').click(); document.getElementById('triExit').click(); }")
        time.sleep(0.3)
        self.idle(page)  # the fly-off is over (its decision runs in the same task)
        self.synced(page)
        self.assertTrue(page.locator("#tri").is_visible())
        self.assertEqual([k for k in self.decided_keys(page) if k not in before], [first])
        second, before = self.here(page), set(self.decided_keys(page))
        page.evaluate("() => { document.getElementById('aNope').click(); document.getElementById('tri').hidden = true; }")
        time.sleep(0.3)
        self.idle(page)  # the fly-off is over (its decision runs in the same task)
        self.synced(page)
        self.assertEqual([k for k in self.decided_keys(page) if k not in before], [], second)

    def test_history_holds_case_ids(self):
        page = self.open("&who=r19h&mode=cards", setup=PRED, **DESK)
        self.wait_card(page)
        first = self.here(page)
        page.keyboard.press("s")
        self.idle(page, "t.pos > 0")
        self.assertEqual(self.state(page)["tri"]["histIds"], ["|".join(first)])


@browser
class FunModeTests(CardBase):
    def test_default_on_switch_and_plain_texts(self):
        page = self.open("&who=fun1&mode=cards", setup=PRED, **DESK)
        self.wait_card(page)
        self.assertTrue(self.state(page)["fun"])
        self.assertTrue(page.locator("#logo").is_visible())
        self.assertTrue(page.locator("#logoPlain").is_hidden())
        self.assertIn(page.inner_text("#logoName"), BRANDS)
        self.assertTrue(page.locator("#triBio .quip").is_visible())
        page.click("#triSet")
        self.assertTrue(page.is_checked("#funMode"))
        page.click("#funMode")
        page.keyboard.press("Escape")
        self.assertTrue(page.locator("#setPop").is_hidden())
        self.check_plain(page)
        page.reload()  # remembered per viewer
        page.wait_for_function("window.__review && __review.state().tri.ready")
        self.check_plain(page)
        page.click("#aYes")  # no pop-up
        time.sleep(0.15)
        self.assertTrue(page.locator("#triMatch").is_hidden())
        self.idle(page)
        page.click("#triSet")
        page.click("#funMode")  # back on
        page.click("#setClose")
        self.assertTrue(self.state(page)["fun"])
        self.assertTrue(page.locator("#logo").is_visible())
        self.assertEqual(page.evaluate("localStorage.getItem('cryoev-review:fun')"), "1")

    def check_plain(self, page):
        self.assertFalse(self.state(page)["fun"])
        self.assertTrue(page.evaluate("document.body.classList.contains('plain')"))
        self.assertTrue(page.locator("#logo").is_hidden())
        self.assertEqual(page.inner_text("#logoPlain"), "Quick check")
        self.assertEqual(page.title(), "Quick check")
        self.assertTrue(page.locator("#triBio .quip").is_hidden())
        self.assertNotIn("singles", page.inner_text("#triProg"))
        texts = [" ".join(page.inner_text(b).split()) for b in ("#aNope", "#aLow", "#aYes")]
        self.assertEqual(texts[0], "Reject ✕")
        self.assertTrue(texts[1].startswith("Low conf"), texts[1])
        self.assertTrue(texts[2].startswith("Accept ✓"), texts[2])
        xs = [page.locator(b).bounding_box()["x"] for b in ("#aNope", "#aLow", "#aYes")]
        self.assertEqual(xs, sorted(xs))  # Reject ✕ · Low conf · Accept ✓
        self.assertNotIn("♥", page.inner_text("#triActions"))

    def test_editor_button(self):
        page = self.open("&who=fun2&mode=editor", setup=setups(PRED, init(FUN_OFF)), **DESK)
        self.assertEqual(" ".join(page.inner_text("#toTri").split()), "Quick check")
        page = self.open("&who=fun2&mode=editor", setup=PRED, **DESK)
        self.assertIn(page.inner_text("#toTri").split()[-1], BRANDS)

    def test_decisions_byte_identical_either_way(self):
        """One flow (accept, reject, low conf, a class chip + Next, Back + re-decide, an adjust with erased points)
        with Fun mode on and off, each on a fresh server, the clock fixed: the saved records are byte-identical
        (server_t, the server's own receive time, set aside)."""
        def fixed(context, page):
            page.clock.set_fixed_time(datetime.datetime(2026, 11, 12, 12, 0, 0))
        out = []
        for fun in (True, False):
            srv = LiveServer()
            self.addCleanup(srv.close)
            setup = setups(PRED, fixed) if fun else setups(PRED, fixed, init(FUN_OFF))
            page = self.open(url=srv.url + "&who=same&mode=cards", setup=setup, **DESK)
            self.assertEqual(self.state(page)["fun"], fun)
            self.flow(page)
            recs = srv.request("GET", "/api/edits")[1]
            for rec in recs.values():
                for d in rec.get("dec", {}).values():
                    d.pop("server_t", None)
            out.append(json.dumps(recs, sort_keys=True))
        self.assertGreater(len(json.loads(out[0])), 1)
        self.assertEqual(out[0], out[1])

    def flow(self, page):
        self.wait_card(page)

        def step(fn):
            pos = self.state(page)["tri"]["pos"]
            fn()
            self.idle(page, f"t.pos !== {pos}")
            self.synced(page)
        step(lambda: page.click("#aYes"))
        step(lambda: page.click("#aNope"))
        step(lambda: page.keyboard.press("l"))  # an undecided prediction: accepted as low conf
        page.locator("#triChips button").first.click()
        page.wait_for_function("__review.state().tri.picked")
        step(lambda: page.click("#aNext"))
        step(lambda: page.keyboard.press("b"))
        step(lambda: page.click("#aNope"))
        page.keyboard.press("e")
        st = self.state(page)["tri"]
        box = page.locator("#tcv").bounding_box()
        xf, r = st["xf"], st["adj"]["ring"]
        for k in (0, 2):
            x, y = box["x"] + r[2 * k] * xf["Wi"] * xf["s"] + xf["ox"], box["y"] + r[2 * k + 1] * xf["Hi"] * xf["s"] + xf["oy"]
            page.keyboard.down("Shift")
            page.mouse.click(x, y)
            page.keyboard.up("Shift")
        page.keyboard.press("Delete")
        step(lambda: page.keyboard.press("Enter"))


@browser
class SideListTests(CardBase):
    def items(self, page):
        return page.evaluate("[...document.querySelectorAll('#triItems button')].map(b => "
                             "[b.querySelector('.nm').textContent, b.querySelector('.st').textContent])")

    def test_desktop_list_states_and_jump(self):
        page = self.open("&who=list1&mode=cards", setup=PRED, **DESK)
        self.wait_card(page)
        self.assertTrue(page.locator("#triList").is_visible())
        self.assertTrue(page.locator("#triListBtn").is_hidden())
        q = self.queue_keys(page)
        it = self.items(page)
        self.assertEqual(len(it), len(q))
        self.assertEqual(it[0][1], "on screen")
        self.assertTrue(it[0][0].startswith("fixA_new · obj 1"), it[0])
        self.assertEqual([s for _, s in it[1:]], ["open"] * (len(q) - 1))
        first = self.here(page)
        page.click("#aYes")
        self.idle(page, "t.pos === 1")
        page.click("#aNope")
        self.idle(page, "t.pos === 2")
        self.synced(page)
        # fixA g104 is a Low conf EV label: kept, it reads "low conf"; g105 rejected
        self.assertEqual([s for _, s in self.items(page)[:3]], ["low conf", "rejected", "on screen"])
        d0 = self.saved_dec(first[0])[first[1]]
        third = self.here(page)
        page.locator("#triItems button").nth(0).click()  # back to the first card: its decision stays, editable
        page.wait_for_function("__review.state().tri.pos === 0")
        self.assertEqual(self.here(page), first)
        self.assertIn("✓", page.inner_text("#triBio"))
        time.sleep(0.4)
        self.assertEqual(self.here(page), first)  # a decided card reached from the list is not skipped
        self.assertEqual(self.saved_dec(first[0])[first[1]], d0)
        self.assertEqual(self.items(page)[2][1], "open")
        page.keyboard.press("b")  # Back returns to where the jump came from
        page.wait_for_function("__review.state().tri.pos === 2")
        self.assertEqual(self.here(page), third)
        page.locator("#triItems button").nth(1).click()
        page.wait_for_function("__review.state().tri.pos === 1")
        page.click("#aYes")  # re-decide: only this card changes
        self.idle(page, "t.pos !== 1")
        self.synced(page)
        self.assertEqual(self.saved_dec(first[0])[first[1]], d0)
        self.assertEqual(self.items(page)[1][1], "accepted")
        self.check_js_equals_python(page)

    def test_phone_drawer(self):
        page = self.open("&who=list2&mode=cards", setup=PRED, **PHONE)
        self.wait_card(page)
        self.assertTrue(page.locator("#triList").is_hidden())
        page.tap("#triListBtn")
        self.assertTrue(page.locator("#triList").is_visible())
        pos = self.state(page)["tri"]["pos"]  # (the class's server: an earlier test may have decided cards)
        self.assertEqual(self.items(page)[pos][1], "on screen")
        to = (pos + 2) % len(self.items(page))
        page.locator("#triItems button").nth(to).tap()
        page.wait_for_function(f"__review.state().tri.pos === {to}")
        self.assertTrue(page.locator("#triList").is_hidden())
        self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 400)


@browser
class PinchTests(CardBase):
    def test_card_pinch_zooms_about_the_fingers(self):
        page = self.open("&who=pin1&mode=cards", setup=setups(PRED, init(NO_CAPTURE)), **PHONE)
        self.wait_card(page)
        first, st0 = self.here(page), self.state(page)["tri"]
        box = page.locator("#tcv").bounding_box()
        mx, my = box["x"] + box["width"] / 2, box["y"] + box["height"] * 0.3
        xf0 = st0["xf"]
        img_pt = ((mx - box["x"] - xf0["ox"]) / xf0["s"], (my - box["y"] - xf0["oy"]) / xf0["s"])
        pinch(page, "#tcv", mx, my, 60, 120)
        st = self.state(page)["tri"]
        self.assertAlmostEqual(st["wk"], 2, places=3)
        self.assertAlmostEqual(st["xf"]["s"], xf0["s"] * 2, places=4)
        xf = st["xf"]  # the image point under the fingers' midpoint stays under it
        self.assertAlmostEqual((mx - box["x"] - xf["ox"]) / xf["s"], img_pt[0], places=3)
        self.assertAlmostEqual((my - box["y"] - xf["oy"]) / xf["s"], img_pt[1], places=3)
        self.assertEqual(self.here(page), first)  # no swipe, no decision
        self.assertEqual(st["zoom"], st0["zoom"])
        pinch(page, "#tcv", mx, my, 200, 10)  # clamped like the wheel zoom
        self.assertAlmostEqual(self.state(page)["tri"]["wk"], 0.5, places=3)
        self.assertIsNone(self.saved_dec(first[0]).get(first[1]))

    def test_editor_pinch_unchanged(self):
        page = self.open("&who=pin2&mode=editor", setup=init(NO_CAPTURE), **DESK)
        v0, box = self.state(page)["view"], page.locator("#cv").bounding_box()
        pinch(page, "#cv", box["x"] + 300, box["y"] + 200, 50, 150)
        v = self.state(page)["view"]
        self.assertAlmostEqual(v["s"], v0["s"] * 3, places=4)
        self.assertAlmostEqual((300 - v["ox"]) / v["s"], (300 - v0["ox"]) / v0["s"], places=3)
        self.assertAlmostEqual((200 - v["oy"]) / v["s"], (200 - v0["oy"]) / v0["s"], places=3)


@browser
class EraserTests(CardBase):
    def scr(self, page, k):
        st, box = self.state(page)["tri"], page.locator("#tcv").bounding_box()
        xf, r = st["xf"], st["adj"]["ring"]
        return box["x"] + r[2 * k] * xf["Wi"] * xf["s"] + xf["ox"], box["y"] + r[2 * k + 1] * xf["Hi"] * xf["s"] + xf["oy"]

    def adj(self, page):
        return self.state(page)["tri"]["adj"]

    def shift_click(self, page, xy):
        page.keyboard.down("Shift")
        page.mouse.click(*xy)
        page.keyboard.up("Shift")

    def test_box_shift_delete_undo_and_floor(self):
        page = self.open("&who=erase&mode=cards", setup=PRED, **DESK)
        self.wait_card(page)
        card = self.here(page)
        page.keyboard.press("e")
        a0 = self.adj(page)
        n = len(a0["ring"]) // 2
        self.assertGreaterEqual(n, 10)
        pts = [self.scr(page, k) for k in range(n)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2
        # a box from beyond the right edge over the right half of the outline
        x0, y0, x1, y1 = max(xs) + 40, min(ys) - 40, cx + 1, max(ys) + 40
        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move(x1, y1, steps=6)
        page.mouse.up()
        want = sorted(k for k, (x, y) in enumerate(pts) if x1 <= x <= x0 and y0 <= y <= y1)
        self.assertTrue(0 < len(want) < n - 3, want)
        self.assertEqual(self.adj(page)["sel"], want)
        self.assertEqual(page.inner_text("#adjDel"), f"Delete {len(want)} points")
        extra = next(k for k in range(n) if k not in want)
        self.shift_click(page, pts[extra])  # shift-click adds one
        sel = sorted(want + [extra])
        self.assertEqual(self.adj(page)["sel"], sel)
        page.keyboard.press("Delete")
        a = self.adj(page)
        keep = [v for k in range(n) if k not in sel for v in a0["ring"][2 * k:2 * k + 2]]
        self.assertEqual(a["ring"], keep)
        self.assertEqual((a["sel"], a["undo"]), ([], 1))
        page.keyboard.press("z")  # undo the delete
        self.assertEqual(self.adj(page)["ring"], a0["ring"])
        page.mouse.move(min(xs) - 40, min(ys) - 40)
        page.mouse.down()
        page.mouse.move(max(xs) + 40, max(ys) + 40, steps=6)
        page.mouse.up()
        self.assertEqual(len(self.adj(page)["sel"]), n)
        page.click("#adjDel")  # every point: refused, the outline stays
        self.assertIn("at least 3 points", self.adj(page)["msg"])
        self.assertIn("at least 3 points", page.inner_text("#triZoom"))
        self.assertEqual(self.adj(page)["ring"], a0["ring"])
        page.mouse.click(*pts[0])  # a tap on a point selects just that one
        self.assertEqual(self.adj(page)["sel"], [0])
        for k in range(1, n - 3):  # n - 3 selected: 3 points stay, still an outline
            self.shift_click(page, pts[k])
        page.keyboard.press("Delete")
        self.assertEqual(len(self.adj(page)["ring"]), 6)
        page.keyboard.press("Enter")  # save the shape: a decision with a 3-point outline the server accepts
        self.idle(page, "t.pos !== 0")
        self.synced(page)
        d = self.saved_dec(card[0])[card[1]]
        self.assertEqual(d["p"][0], [round(v * 1e5) / 1e5 for v in a0["ring"][2 * (n - 3):]])
        self.check_js_equals_python(page)
        page.keyboard.press("z")  # the card's rewind undoes the saved shape
        page.wait_for_function("__review.state().tri.pos === 0")
        self.synced(page)
        self.assertFalse(self.saved_dec(card[0])[card[1]].get("v"))



@browser
class EditorEraserTests(CardBase):
    F = "fixA_new.png"  # label 103 (MV): 20 points, in the upper half of the image (the class chips cover the bottom)

    def pts(self, page, ring):
        return [self.scr(page, ring[2 * k], ring[2 * k + 1]) for k in range(len(ring) // 2)]

    def vsel(self, page):
        return sorted(int(k.split(":")[1]) // 2 for k in self.state(page)["vsel"])

    def shift_box(self, page, x0, y0, x1, y1):
        page.keyboard.down("Shift")
        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move(x1, y1, steps=6)
        page.mouse.up()
        page.keyboard.up("Shift")

    def shift_click(self, page, xy):
        page.keyboard.down("Shift")
        page.mouse.click(*xy)
        page.keyboard.up("Shift")

    def select103(self, page):
        ring = self.image(self.F)["gt"][2]["p"][0]
        self.click_obj(page, ring)
        self.assertEqual(self.state(page)["sel"]["kind"], "gt")
        return ring

    @staticmethod
    def bare(d):
        return {k: v for k, v in d.items() if k not in ("t", "server_t", "pending")}

    def test_box_shift_delete_floor_undo(self):
        page = self.open("&who=erase3&mode=editor", **DESK)
        self.go(page, self.F)
        ring = self.select103(page)
        n = len(ring) // 2
        pts = self.pts(page, ring)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        x0, y0, x1, y1 = max(xs) + 30, min(ys) - 30, cx + 1, cy  # Shift-drag a box over the upper-right quarter
        self.shift_box(page, x0, y0, x1, y1)
        want = sorted(k for k, (x, y) in enumerate(pts) if x1 <= x <= x0 and y0 <= y <= y1)
        self.assertTrue(0 < len(want) < n - 3, want)
        self.assertEqual(self.vsel(page), want)
        self.assertEqual(self.state(page)["sel"]["kind"], "gt")  # the label stays selected
        self.assertEqual(page.inner_text("#vDel"), f"Delete {len(want)} points")
        extra = next(k for k in range(n) if k not in want and pts[k][1] < cy)
        self.shift_click(page, pts[extra])  # shift-click adds one, and toggles it off again
        self.assertEqual(self.vsel(page), sorted(want + [extra]))
        self.shift_click(page, pts[extra])
        self.assertEqual(self.vsel(page), want)
        self.shift_click(page, pts[extra])
        sel = sorted(want + [extra])
        self.assertIsNone(self.saved_dec(self.F).get("g103"))  # selecting points is not an edit
        page.keyboard.press("Delete")
        self.synced(page)
        d = self.saved_dec(self.F)["g103"]
        keep = [v for k in range(n) if k not in sel for v in ring[2 * k:2 * k + 2]]
        self.assertEqual((d["v"], d["p"]), ("keep:MV", [keep]))
        self.assertEqual(self.state(page)["vsel"], [])
        self.assertTrue(page.locator("#vDel").is_hidden())
        self.assertIsNotNone(self.obj(page, self.F, oid=103))  # Delete took the points, not the label
        self.check_js_equals_python(page)
        page.keyboard.press("z")  # undo restores the outline
        self.synced(page)
        self.assertEqual(self.obj(page, self.F, oid=103)["p"], [ring])
        self.assertIsNone(self.saved_dec(self.F)["g103"]["v"])
        # every point: refused with a message, nothing changes
        ring = self.select103(page)
        self.shift_box(page, min(xs) - 30, min(ys) - 30, max(xs) + 30, max(ys) + 30)
        self.assertEqual(len(self.vsel(page)), n)
        page.click("#vDel")
        self.assertIn("at least 3 points", page.inner_text("#hint"))
        self.assertEqual(self.obj(page, self.F, oid=103)["p"], [ring])
        page.keyboard.press("Escape")  # Esc clears the points first, the label stays selected
        self.assertEqual((self.state(page)["vsel"], self.state(page)["sel"]["kind"]), ([], "gt"))
        top = sorted(range(n), key=lambda k: pts[k][1])  # upper points first (clear of the class chips)
        page.mouse.click(*pts[top[0]])  # a click on a point selects just that one
        self.assertEqual(self.vsel(page), [top[0]])
        for k in top[1:n - 3]:  # n - 3 selected: 3 points stay
            self.shift_click(page, pts[k])
        page.click("#vDel")
        self.synced(page)
        stay = sorted(top[n - 3:])
        self.assertEqual(self.saved_dec(self.F)["g103"]["p"], [[v for k in stay for v in ring[2 * k:2 * k + 2]]])
        self.check_js_equals_python(page)
        page.keyboard.press("Delete")  # no points selected: Delete deletes the label, as before
        self.synced(page)
        self.assertEqual(self.saved_dec(self.F)["g103"]["v"], "del")

    def test_record_equals_right_click_delete_and_drag_shape(self):
        f = "fixC_old.png"  # its own image: the class's server is shared with the test above
        page = self.open("&who=erase4&mode=editor", **DESK)
        self.go(page, f)
        ring = self.image(f)["gt"][0]["p"][0]  # 301
        self.click_obj(page, ring)
        pts = self.pts(page, ring)
        top = sorted(range(len(pts)), key=lambda k: pts[k][1])[:3]
        page.mouse.click(*pts[top[0]])
        for k in top[1:]:
            self.shift_click(page, pts[k])
        page.keyboard.press("Backspace")
        self.synced(page)
        erased = self.saved_dec(f)["g301"]
        page.keyboard.press("z")
        self.synced(page)
        self.assertEqual(self.obj(page, f, oid=301)["p"], [ring])
        self.click_obj(page, ring)
        for k in sorted(top, reverse=True):  # the same points, one right-click each (highest index first)
            page.mouse.click(*pts[k], button="right")
            time.sleep(0.6)
        self.synced(page)
        clicked = self.saved_dec(f)["g301"]
        self.assertEqual(self.bare(erased), self.bare(clicked))
        self.assertEqual(sorted(erased), sorted(clicked))
        self.assertEqual(len(erased["p"][0]), len(ring) - 6)
        # a vertex drag on another label saves a record of the same shape
        r302 = self.image(f)["gt"][1]["p"][0]
        self.click_obj(page, r302)
        top = min(range(len(r302) // 2), key=lambda k: r302[2 * k + 1])
        x, y = self.scr(page, r302[2 * top], r302[2 * top + 1])
        page.mouse.move(x, y); page.mouse.down(); page.mouse.move(x, y - 20, steps=5); page.mouse.up()
        self.synced(page)
        dragged = self.saved_dec(f)["g302"]
        self.assertEqual(sorted(dragged), sorted(erased))
        self.assertEqual(dragged["v"].split(":")[0], erased["v"].split(":")[0])
        self.assertNotEqual(dragged["p"], [r302])
        self.check_js_equals_python(page)

if __name__ == "__main__":
    unittest.main()
