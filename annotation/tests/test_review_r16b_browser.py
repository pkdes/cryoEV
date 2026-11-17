"""
Browser tests (Playwright, headless Edge or Chrome; skipped without one) for round 16b: native-resolution item crops on
object / tree cards (data.js items[].chi, chi_box, chi_w, chi_h; tools/add_hires.py), annotation/review_tool/index.html.
  - answers save the same records with the crop drawn as without it (control: it was drawn);
  - the crop lands where the base image shows the same spot: a marker at a known normalized point, white in the base
    image and black in the crop, is found at the same screen point within 1 px;
  - r17b: Smooth's sigma is in the image's finest px: / 4 only when the image has hi or the item chi (those sets
    unchanged), else base px; on a base-only card Smooth visibly lowers the canvas pixels' std dev.

  python -m unittest discover -s annotation/tests -p test_review_r16b_browser.py -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import object_card_fixture as OF  # noqa: E402
from test_object_card_browser import _Browser  # noqa: E402
from test_review_r16a_browser import strip  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2] / "tests"))
from cryoev_testkit import browser  # noqa: E402

K = 4  # the "original" is 4x the package image (5760 vs 1440 in the ML / MV set)
READY = "window.__review && __review.obj() && __review.state().server"


def chi_box(crop):
    cx, cy, hw, hh = (crop[0] + crop[2]) / 2, (crop[1] + crop[3]) / 2, (crop[2] - crop[0]) * 0.75, (crop[3] - crop[1]) * 0.75
    return [max(0.0, cx - hw), max(0.0, cy - hh), min(1.0, cx + hw), min(1.0, cy + hh)]


def add_chi(folder, data, original, only=None):
    """items[].chi / chi_box / chi_w / chi_h, as add_hires writes them: crops of `original` ({stem: PIL image at K x})."""
    for it in data["items"]:
        if it["image"] not in original or (only and it["id"] not in only):
            continue
        im, box = original[it["image"]], chi_box(it["crop"])
        W, H = im.size
        c = im.crop((round(box[0] * W), round(box[1] * H), round(box[2] * W), round(box[3] * H)))
        rel = f"images/c_{it['id']}.jpg"
        c.convert("L").save(folder / rel, quality=90)
        it.update(chi=rel, chi_box=[round(box[0] * W) / W, round(box[1] * H) / H, round(box[2] * W) / W, round(box[3] * H) / H],
                  chi_w=c.size[0], chi_h=c.size[1])
    (folder / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(data) + ";\n", encoding="utf-8")


@browser
class ChiTests(_Browser):
    def serve_set(self, chi, marker=None):
        d = Path(tempfile.mkdtemp(prefix="chi_")) / "pred_obj_fixture_ab12"
        data = OF.build(d)
        if marker:  # base: grey with a white marker; original: grey with a black marker at the same normalized point
            it = data["items"][0]
            mx, my = it["crop"][0] + (it["crop"][2] - it["crop"][0]) * 0.65, it["crop"][1] + (it["crop"][3] - it["crop"][1]) * 0.4
            base = Image.new("L", (640, 480), 110)
            base.paste(255, (round(mx * 640) - 2, round(my * 480) - 2, round(mx * 640) + 2, round(my * 480) + 2))
            base.convert("RGB").save(d / "images" / f"{it['image']}.png")
            orig = Image.new("L", (640 * K, 480 * K), 110)
            px, py = round(mx * 640) * K, round(my * 480) * K  # the same area as the base marker's 4 x 4 px
            orig.paste(0, (px - 2 * K, py - 2 * K, px + 2 * K, py + 2 * K))
            originals = {it["image"]: orig}
        else:
            originals = {}
            for stem, circ in (("imgA", [(200, 220, 110), (200, 220, 86), (200, 220, 64), (470, 200, 90), (440, 180, 28), (500, 225, 24)]),
                               ("imgB", [(160, 160, 70), (420, 300, 95), (420, 300, 72), (560, 110, 40)])):
                p = d / "images" / f"_{stem}_orig.png"
                OF._png(p, 640 * K, 480 * K, [(x * K, y * K, r * K) for x, y, r in circ])
                originals[stem] = Image.open(p).convert("L")
                p.unlink()
        if chi:
            add_chi(d, data, originals)
        srv = OF.FakeServer(d).start()
        self.addCleanup(srv.close)
        pg = self.page(srv.url + "?who=rev1", width=1280, height=800)
        pg.wait_for_function(READY, timeout=15000)
        return data, srv, pg

    def test_answers_same_with_and_without_chi(self):
        out = {}
        for chi in (False, True):
            data, srv, pg = self.serve_set(chi)
            drawn = 0
            for _ in range(len(data["items"])):
                if chi:
                    want = data["items"][self.obj(pg)["pos"]].get("chi")
                    if want:
                        pg.wait_for_function(f"__review.obj().chi === {json.dumps(want)}")
                        drawn += 1
                for k in ("2", "s", "z"):
                    pg.keyboard.press(k)
                pg.keyboard.press("ArrowRight")
            pg.wait_for_function("__review.state().dirty === 0", timeout=10000)
            self.assertEqual(self.errors, [])
            out[chi] = ({f: strip(r) for f, r in srv.records.items()}, drawn)
        self.assertGreater(out[True][1], 0)  # control: crops were drawn
        self.assertEqual(out[True][0], out[False][0])
        self.assertTrue(any(k.startswith("a:") for r in out[True][0].values() for k in r["dec"]))

    def marker_at(self, pg, dark):
        """Screen (css px, card canvas) centroid of the marker's pixels: near black (crop) or near white (base)."""
        return pg.evaluate("""dark => { const c = document.getElementById('ocv'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data,
            r = window.devicePixelRatio || 1; let sx = 0, sy = 0, n = 0;
            for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) { const v = d[(y * c.width + x) * 4];
              if (dark ? v < 40 : v > 220) { sx += x; sy += y; n++; } }
            return n ? [sx / n / r, sy / n / r, n] : null; }""", dark)

    def test_crop_lands_on_the_base_image(self):
        _, _, pg0 = self.serve_set(False, marker=True)
        pg0.keyboard.press("i")  # outlines off
        pg0.wait_for_function("!__review.obj().outlines")
        base = self.marker_at(pg0, False)
        data, _, pg1 = self.serve_set(True, marker=True)
        pg1.wait_for_function(f"__review.obj().chi === {json.dumps(data['items'][0]['chi'])}")
        pg1.keyboard.press("i")
        pg1.wait_for_function("!__review.obj().outlines")
        crop = self.marker_at(pg1, True)
        self.assertIsNotNone(base)
        self.assertIsNotNone(crop)
        self.assertIsNone(self.marker_at(pg1, False))  # the base marker is under the crop
        self.assertAlmostEqual(crop[0], base[0], delta=1)
        self.assertAlmostEqual(crop[1], base[1], delta=1)
        # an item without chi (or with a bad path) draws the base image only
        bad = pg1.evaluate("(() => { const it = __review.obj().item; return it.chi; })()")
        self.assertTrue(bad.startswith("images/c_"))
        self.assertEqual(self.errors, [])


SMOOTH2 = ("try { localStorage.setItem('cryoev-review:contrast:rev1', "
           "JSON.stringify({b: 0, c: 0, g: 1, auto: false, smooth: 2})); } catch (e) {}")
SD = """() => { const c = document.getElementById('ocv'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let s = 0, q = 0, n = 0; for (let i = 0; i < d.length; i += 4) { s += d[i]; q += d[i] * d[i]; n++; }
    const m = s / n; return Math.sqrt(q / n - m * m); }"""


@browser
class SmoothLevelTests(_Browser):
    """r17b: Smooth's divisor (FULL_K) applies only when a finer level (hi / chi) exists."""

    def serve_smooth(self, *, hi=(), chi=False):
        d = Path(tempfile.mkdtemp(prefix="sm_")) / "pred_obj_fixture_ab12"
        data = OF.build(d, hi=hi)
        if chi:
            originals = {}
            for stem in ("imgA", "imgB"):
                p = d / "images" / f"_{stem}_orig.png"
                OF._png(p, 640 * K, 480 * K, [])
                originals[stem] = Image.open(p).convert("L")
                p.unlink()
            add_chi(d, data, originals)
        srv = OF.FakeServer(d).start()
        self.addCleanup(srv.close)
        ctx = self.browser.new_context(viewport={"width": 1280, "height": 800})
        self.addCleanup(ctx.close)
        ctx.add_init_script(SMOOTH2)
        pg = ctx.new_page()
        self.errors = []
        pg.on("pageerror", lambda e: self.errors.append(str(e)))
        pg.goto(srv.url + "?who=rev1")
        pg.wait_for_function(READY, timeout=15000)
        pg.wait_for_function("__review.state().vc.last && __review.state().vc.last.sigma != null", timeout=10000)
        return data, pg

    def test_sigma_rule(self):
        _, pg = self.serve_smooth()
        got = pg.evaluate("[[2, 1, false], [3, 1, false], [2, 1, true], [2, 2, true], [2, 4, true], [0, 1, false]]"
                          ".map(a => __review.vcSigma(...a))")
        self.assertEqual(got, [2, 3, 0.5, 1, 2, 0])

    def test_sigma_base_only_hi_and_chi(self):
        _, pg = self.serve_smooth()                       # base-only: sigma = Smooth in base px (was 0.5)
        self.assertEqual(pg.evaluate("__review.state().vc.last.sigma"), 2)
        _, pg = self.serve_smooth(hi=("imgA", "imgB"))    # the 2x image: unchanged, 2 x 2 / 4
        pg.wait_for_function("__review.obj().hi", timeout=10000)
        pg.wait_for_function("__review.state().vc.last.w === 1280", timeout=10000)
        self.assertEqual(pg.evaluate("__review.state().vc.last.sigma"), 1)
        data, pg = self.serve_smooth(chi=True)            # the item's crop: unchanged, 2 x k / 4
        pg.wait_for_function("__review.obj().chi", timeout=10000)
        it = data["items"][self.obj(pg)["pos"]]
        pg.wait_for_function(f"__review.state().vc.last.w === {it['chi_w']}", timeout=10000)
        k = it["chi_w"] / ((it["chi_box"][2] - it["chi_box"][0]) * 640)
        self.assertAlmostEqual(pg.evaluate("__review.state().vc.last.sigma"), 2 * k / 4, places=6)
        self.assertEqual(self.errors, [])

    def test_smooth_visible_on_base_only_card(self):
        _, pg = self.serve_smooth()
        pg.keyboard.press("i")
        pg.wait_for_function("!__review.obj().outlines")
        sd = {}
        for v in (0, 2):
            pg.evaluate("v => { const s = document.getElementById('vcS'); s.value = v; "
                        "s.dispatchEvent(new Event('input')); }", v)
            pg.wait_for_function(f"__review.state().vc.smooth === {v}")
            pg.wait_for_timeout(150)
            sd[v] = pg.evaluate(SD)
            shots = os.environ.get("R17B_SHOTS")
            if shots and v == 2:
                pg.screenshot(path=str(Path(shots) / "r17b_smooth2_base_only.png"))
        self.assertGreater(sd[0], 5, sd)                  # control: the card shows the noisy image
        self.assertLess(sd[2], sd[0] * 0.7, sd)
        self.assertEqual(self.errors, [])


if __name__ == "__main__":
    unittest.main()
