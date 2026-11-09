"""
Gallery module (BUILD_CONTRACTS.md section 13): events.jsonl fold + validation, concurrent appends, access (an advanced
non-owner gets the same 404 as for a cid that doesn't exist, on every route; owner and admin are the controls), escaping
(pages and the standalone report), the streamed zips, the report's counts on a known answer, with and without
montages, and a Playwright pass at 390 / 1280 px, light and dark.

Temp roster / state / raw root only (never the live raw_tiles).   python -m unittest portal.tests.test_gallery -v
"""
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import app as APP  # noqa: E402
from portal import auth as A  # noqa: E402
from portal import zipstream as Z  # noqa: E402
from portal.modules import gallery as G  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except Exception:                                            # noqa: BLE001
    sync_playwright = None

XSS = '<img src=x onerror=alert(1)>'
XNOTE = '<script>alert("note")</script>'
CID_A, CID_B, CID_C, CID_X = "20261106-1_aaaaaa", "20261107_bbbbbb", "20261108_cccccc", "20261109_dddddd"
MISSING = "20261110_eeeeee"
NAMES = [f"g1_Grid_1__m1_r{r}c{c}_t{r * 3 + c:02d}__denoised.png" for r in range(3) for c in range(3)]


def tile_png(i: int) -> bytes:
    from PIL import Image
    im = Image.new("L", (120, 90), 40 + 20 * i)
    for x in range(0, 120, 8):                                # something to look at
        for y in range(90):
            im.putpixel((x, y), 200)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def montage_jpg() -> bytes:
    from PIL import Image
    b = io.BytesIO()
    Image.new("L", (360, 270), 90).save(b, "JPEG")
    return b.getvalue()


def make_conv(raw: Path, cid: str, owner: str, *, montage=True, sample="Sample one", n=9, mname="m1") -> Path:
    d = raw / "raw_tiles" / cid
    (d / "tiles").mkdir(parents=True)
    rows = ["output_name,source_file,grid,sample_label,montage,source_size,output_size,factor,source_a_px,output_a_px,"
            "nm_per_px,compute,md5"]
    csvq = lambda s: '"' + s.replace('"', '""') + '"'  # noqa: E731
    for i, name in enumerate(NAMES[:n]):
        (d / "tiles" / name).write_bytes(tile_png(i))
        rows.append(f"{name},src{i}.jpg,1,{csvq(sample)},{mname},2880x2046,120x90,2.0,7.362,14.724,1.4724,GPU,0")
    (d / "tiles" / "MANIFEST.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (d / "convert.json").write_text(json.dumps({"schema": "cj_convert/1", "id": cid, "owner": owner, "session": cid[:cid.index("_")],
                                                "recipe": "20261027", "created": "2026-11-06T12:28:38"}), encoding="utf-8")
    (d / "status.json").write_text(json.dumps({"state": "done", "n_tiles": n}), encoding="utf-8")
    (d / "acquisition.json").write_text(json.dumps({
        "schema": "acquisition/1", "scope_model": "Test scope",
        "montages": [{"montage": mname, "mag": 1900, "pixel_A": 7.362, "defocus_um": -50.0, "dose": 1.2,
                      "acquired": "2026-11-06T10:15:00", "exposure_s": 0.5, "n_pieces": n}],
        "session": {"n_montages": 1, "pixel_A": 7.362}}), encoding="utf-8")
    if montage:
        (d / "montages").mkdir()
        (d / "montages" / f"{mname}_stitched.jpg").write_bytes(montage_jpg())
        tiles = [{"output_name": name, "row": r, "col": c, "x": c * 120, "y": r * 90, "w": 120, "h": 90}
                 for i, name in enumerate(NAMES[:n]) for r, c in [divmod(i, 3)]]
        (d / "montages" / "layout.json").write_text(json.dumps({"schema": "montage_layout/1", "montages": [{
            "name": mname, "file": f"{mname}_stitched.jpg", "width": 360, "height": 270, "nm_per_px": 4.4, "scalebar_nm": 200,
            "grid": "1", "sample_label": sample, "tiles": tiles}]}), encoding="utf-8")
        (d / "montages" / "stitch.json").write_text(json.dumps({"state": "done", "error": None, "n": 1, "t": "x"}),
                                                    encoding="utf-8")
    return d


def mark(d: Path, who="alyssia"):
    """The known answer: 9 tiles = 4 keep, 2 reject (thick ice, empty), 1 maybe, 2 unmarked."""
    c = G.Conv(d, json.loads((d / "convert.json").read_text()), {})
    for i in range(4):
        G.append_event(c, who, NAMES[i], "flag", "keep")
    G.append_event(c, who, NAMES[4], "flag", "reject")
    G.append_event(c, who, NAMES[4], "tags", ["thick ice"])
    G.append_event(c, who, NAMES[5], "flag", "reject")
    G.append_event(c, who, NAMES[5], "tags", ["empty"])
    G.append_event(c, who, NAMES[6], "flag", "maybe")
    G.append_event(c, who, NAMES[0], "stars", 5)
    G.append_event(c, who, NAMES[0], "note", "nice field")
    return c


class World:
    """Temp env, raw root with conversions, three people (owner and other are advanced), the portal under uvicorn."""

    def __init__(self):
        self.saved = {k: os.environ.get(k) for k in ("CJ_RAW_ROOT", "CJ_PREDICT_ROOT", "CJ_REVIEW_ROOT")}
        self.env = T.Env()
        self.keys = {"admin": self.env.key, "owner": A.add_key(self.env.roster, "alyssia", "advanced"),
                     "other": A.add_key(self.env.roster, "bob", "advanced"), "basic": A.add_key(self.env.roster, "bas", "basic")}
        self.raw = self.env.tmp / "raw"
        self.predict = self.env.tmp / "predict"
        for p in (self.raw, self.predict, self.env.tmp / "review"):
            p.mkdir(parents=True)
        os.environ.update(CJ_RAW_ROOT=str(self.raw), CJ_PREDICT_ROOT=str(self.predict), CJ_REVIEW_ROOT=str(self.env.tmp / "review"))
        self.env.write_live(T.live_doc())
        self.a = make_conv(self.raw, CID_A, "alyssia")
        self.b = make_conv(self.raw, CID_B, "alyssia", montage=False, sample="Sample two")
        self.c = make_conv(self.raw, CID_C, "bob", sample="Bob's sample")
        self.x = make_conv(self.raw, CID_X, "alyssia", sample=XSS)
        self.portal = T.Portal(self.env.config())
        self.tok = {w: T.login(self.portal.port, k) for w, k in self.keys.items()}

    def req(self, who, method, path, body=None, origin=True, extra=None):
        h = T.cookie(self.tok[who]) if who else {}
        if origin and method != "GET":
            h["Origin"] = self.portal.origin
        h.update(extra or {})
        return T.request(self.portal.port, method, path, h, body)

    def get(self, who, path):
        return self.req(who, "GET", path)

    def post(self, who, cid, doc, **kw):
        extra = {"X-Requested-With": "gallery", "Content-Type": "application/json"}
        extra.update(kw.pop("extra", {}))
        return self.req(who, "POST", f"/gallery/{cid}/api/event", json.dumps(doc).encode(), extra=extra, **kw)

    def close(self):
        self.portal.stop()
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.env.cleanup()


# ---------------------------------------------------------------- events: fold, validation, concurrency
class Events(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_gal_"))
        self.d = make_conv(self.tmp, CID_A, "alyssia")
        self.c = G.Conv(self.d, json.loads((self.d / "convert.json").read_text()), {})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_latest_wins_and_cleared_flag(self):
        k = NAMES[0]
        G.append_event(self.c, "u1", k, "flag", "keep")
        G.append_event(self.c, "u2", k, "flag", "reject")
        e = G.state_of(self.c)[k]
        self.assertEqual((e["flag"], e["by"]["flag"]["who"]), ("reject", "u2"))
        G.append_event(self.c, "u1", k, "flag", "")                 # cleared
        self.assertNotIn(k, G.state_of(self.c))
        G.append_event(self.c, "u1", k, "stars", 3)
        G.append_event(self.c, "u1", k, "tags", ["empty", "thick ice"])
        G.append_event(self.c, "u1", k, "note", "hello")
        e = G.state_of(self.c)[k]
        self.assertEqual((e["flag"], e["stars"], e["tags"], e["note"]), ("", 3, ["thick ice", "empty"], "hello"))
        G.append_event(self.c, "u1", k, "stars", 0)
        G.append_event(self.c, "u1", k, "tags", [])
        self.assertEqual(G.state_of(self.c)[k]["stars"], 0)
        lines = (self.d / "gallery" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 8)                              # append only: nothing rewritten
        rec = json.loads(lines[0])
        self.assertEqual(set(rec), {"k", "f", "v", "who", "t"})
        self.assertRegex(rec["t"], r"^\d{4}-\d\d-\d\dT")

    def test_invalid_refused(self):
        k = NAMES[0]
        bad = [("not-a-tile.png", "flag", "keep"), (k, "color", "red"), (k, "flag", "great"), (k, "flag", None),
               (k, "stars", 6), (k, "stars", -1), (k, "stars", "3"), (k, "stars", True), (k, "stars", 2.5),
               (k, "tags", ["made up reason"]), (k, "tags", "empty"), (k, "tags", [1]),
               (k, "note", "x" * 1001), (k, "note", 5), ("../convert.json", "note", "x")]
        for key, f, v in bad:
            with self.subTest(k=key, f=f, v=v):
                with self.assertRaises(ValueError):
                    G.append_event(self.c, "u", key, f, v)
        self.assertFalse((self.d / "gallery" / "events.jsonl").exists())      # nothing was written
        G.append_event(self.c, "u", "m1", "flag", "maybe")                    # a montage name is a valid key
        G.append_event(self.c, "u", k, "note", "x" * 1000)                    # the limit itself is fine
        # a hand-edited file can't break the fold: bad lines, wrong keys and wrong values are skipped
        with open(self.d / "gallery" / "events.jsonl", "a", encoding="utf-8") as f:
            f.write('not json\n{"k": "zzz", "f": "flag", "v": "keep"}\n{"k": "%s", "f": "flag", "v": "bogus"}\n' % k)
            f.write('{"k": "%s", "f": "stars", "v": 4, "who": "u", "t": "x"}' % k)       # a last line without newline
        st = G.state_of(self.c)
        self.assertEqual(st[k]["stars"], 4)
        self.assertEqual(st["m1"]["flag"], "maybe")
        G.append_event(self.c, "u", k, "stars", 2)                            # must not glue onto the cut line
        self.assertEqual(G.state_of(self.c)[k]["stars"], 2)

    def test_concurrent_appends_lose_nothing(self):
        errs = []

        def work(t):
            try:
                for i in range(25):
                    G.append_event(self.c, f"u{t}", NAMES[t], "note", f"t{t}-{i}")
            except Exception as e:                                           # noqa: BLE001
                errs.append(e)
        ths = [threading.Thread(target=work, args=(t,)) for t in range(8)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        self.assertEqual(errs, [])
        lines = (self.d / "gallery" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 200)
        recs = [json.loads(x) for x in lines]                                # every line whole
        for t in range(8):
            self.assertEqual([r["v"] for r in recs if r["k"] == NAMES[t]], [f"t{t}-{i}" for i in range(25)])
        self.assertEqual(G.state_of(self.c)[NAMES[3]]["note"], "t3-24")


# ---------------------------------------------------------------- zipstream
class ZipStream(unittest.TestCase):
    def test_contents_streamed_and_zip64(self):
        tmp = Path(tempfile.mkdtemp(prefix="cj_zip_"))
        try:
            big = tmp / "big.png"
            blk = os.urandom(1 << 20)
            with open(big, "wb") as f:
                for _ in range(24):
                    f.write(blk)

            def gen():
                for i in range(10):
                    yield bytes([i]) * 100000
            chunks = list(Z.stream_zip([("a/big.png", big), ("notes.txt", b"hello" * 100), ("gen.bin", gen()),
                                        ("a/big.png", b"dup")]))
            self.assertGreater(len(chunks), 20)                              # streamed in pieces ...
            self.assertLess(max(map(len, chunks)), 3 * (1 << 20))            # ... never whole files (24 MB)
            z = zipfile.ZipFile(io.BytesIO(b"".join(chunks)))
            self.assertIsNone(z.testzip())
            self.assertEqual(z.namelist(), ["a/big.png", "notes.txt", "gen.bin", "a/big (2).png"])
            self.assertEqual(z.read("a/big.png"), big.read_bytes())
            self.assertEqual(z.read("notes.txt"), b"hello" * 100)
            self.assertEqual(len(z.read("gen.bin")), 1000000)
            infos = {i.filename: i for i in z.infolist()}
            self.assertEqual(infos["a/big.png"].compress_type, zipfile.ZIP_STORED)       # images stored
            self.assertEqual(infos["notes.txt"].compress_type, zipfile.ZIP_DEFLATED)
            self.assertTrue(infos["gen.bin"].extract_version >= 45)                      # unknown size -> zip64
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_names_and_early_errors(self):
        for raw, want in [("../../etc/passwd", "etc/passwd"), ("C:\\Windows\\x.png", "C_/Windows/x.png"),
                          ("/abs/name.png", "abs/name.png"), ("a/./b/../c.png", "a/b/c.png"), ("con", "_con"),
                          ("we<ir>d|na?me.png", "we_ir_d_na_me.png"), ("", "file"), ("..", "file"), ("a\x00b.png", "a_b.png"),
                          ("x" * 300 + ".png", "x" * 116 + ".png")]:
            with self.subTest(raw=raw):
                self.assertEqual(Z.safe_arcname(raw), want)
        with self.assertRaises(FileNotFoundError):                           # before the first byte
            Z.stream_zip([("x.png", Path(tempfile.gettempdir()) / "cj_no_such_file_xyz.png")])


# ---------------------------------------------------------------- the portal
class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = World()

    @classmethod
    def tearDownClass(cls):
        cls.w.close()

    def tearDown(self):
        with G._ZIPS_LOCK:
            G._ZIPS.clear()


class Access(Base):
    def routes(self, cid):
        n, mname = NAMES[0], "m1"
        return [("GET", f"/gallery/{cid}/"), ("GET", f"/gallery/{cid}/report"), ("GET", f"/gallery/{cid}/report.zip"),
                ("GET", f"/gallery/{cid}/images.zip"), ("GET", f"/gallery/{cid}/api/state"),
                ("GET", f"/gallery/{cid}/t/{n}"), ("GET", f"/gallery/{cid}/ts/{n}"), ("GET", f"/gallery/{cid}/rep/{n}"),
                ("GET", f"/gallery/{cid}/m/{mname}"), ("GET", f"/gallery/{cid}/ms/{mname}"), ("GET", f"/gallery/{cid}/mp/{mname}"),
                ("GET", f"/gallery/compare?c={cid}"), ("GET", f"/gallery/compare?c={CID_B}&c={cid}"),
                ("GET", f"/gallery/compare/report?c={cid}"), ("GET", f"/gallery/compare/report.zip?c={cid}"),
                ("GET", f"/gallery/compare/images.zip?c={cid}"), ("POST", f"/gallery/{cid}/api/event")]

    def call(self, who, method, path):
        if method == "POST":
            return self.w.post(who, path.split("/")[2], {"k": NAMES[0], "f": "stars", "v": 1})
        return self.w.get(who, path)

    def test_non_owner_gets_the_same_404_as_a_missing_cid(self):
        for method, path in self.routes(CID_A):
            with self.subTest(path=path):
                st, _, body = self.call("other", method, path)
                miss_path = path.replace(CID_A, MISSING)
                mst, _, mbody = self.call("other", method, miss_path)
                self.assertEqual((st, mst), (404, 404), path)
                self.assertEqual(body, mbody, "foreign and missing differ: " + path)
                self.assertNotIn(b"alyssia", body)
                for who in ("owner", "admin"):                                # controls: owner and admin get through
                    cst, _, _ = self.call(who, method, path)
                    self.assertEqual(cst, 200, f"{who} {path}")
        # compare with ONE foreign cid among the owner's: the whole set is refused
        st, _, body = self.w.get("owner", f"/gallery/compare?c={CID_A}&c={CID_C}")
        st2, _, body2 = self.w.get("owner", f"/gallery/compare?c={CID_A}&c={MISSING}")
        self.assertEqual((st, st2, body), (404, 404, body2))
        # no events were written by the refused POSTs
        self.assertFalse((self.w.c / "gallery" / "events.jsonl").exists())

    def test_roles_and_traversal(self):
        self.assertEqual(self.w.get("basic", "/gallery/")[0], 403)                       # control: role
        self.assertEqual(self.w.get(None, "/gallery/")[0] in (302, 303, 401, 403), True)
        for name in ("convert.json", "..%2Fconvert.json", "MANIFEST.csv", "%2e%2e%2f%2e%2e%2fstatus.json", "x.png"):
            for kind in ("t", "ts", "rep", "m", "ms", "mp"):
                st, _, _ = self.w.get("owner", f"/gallery/{CID_A}/{kind}/{name}")
                self.assertEqual(st, 404, f"{kind}/{name}")

    def test_post_needs_origin_and_header(self):
        doc = {"k": NAMES[0], "f": "stars", "v": 2}
        self.assertEqual(self.w.post("owner", CID_A, doc, origin=False)[0], 403)         # the Gate
        h = {"Content-Type": "application/json", "Origin": self.w.portal.origin}
        st, _, _ = T.request(self.w.portal.port, "POST", f"/gallery/{CID_A}/api/event", {**T.cookie(self.w.tok["owner"]), **h},
                             json.dumps(doc).encode())
        self.assertEqual(st, 403)                                                         # no X-Requested-With
        self.assertEqual(self.w.post("owner", CID_A, doc)[0], 200)                        # control

    def test_index_lists_mine_admin_all(self):
        st, _, body = self.w.get("owner", "/gallery/")
        page = body.decode()
        self.assertIn(CID_A, page)
        self.assertIn(CID_B, page)
        self.assertNotIn(CID_C, page)
        self.assertNotIn(CID_A, self.w.get("other", "/gallery/")[2].decode())
        self.assertIn(CID_C, self.w.get("other", "/gallery/")[2].decode())
        adm = self.w.get("admin", "/gallery/")[2].decode()
        self.assertNotIn(CID_C, adm)                                                      # admin default: Mine
        self.assertIn(CID_C, self.w.get("admin", "/gallery/?all=1")[2].decode())
        self.assertNotIn(CID_C, self.w.get("owner", "/gallery/?all=1")[2].decode())
        self.assertIn("Not backed up yet", page)


class Events_http(Base):
    def test_post_and_state(self):
        w, k = self.w, NAMES[2]
        st, _, body = w.post("owner", CID_B, {"k": k, "f": "flag", "v": "keep"})
        self.assertEqual(st, 200, body)
        self.assertEqual(json.loads(body)["entry"]["flag"], "keep")
        st, _, body = w.get("owner", f"/gallery/{CID_B}/api/state")
        self.assertEqual(json.loads(body)["state"][k]["flag"], "keep")
        self.assertEqual(json.loads(body)["state"][k]["by"]["flag"]["who"], "alyssia")      # server-side identity
        # the admin may mark the owner's conversion; the stamp is the admin's
        w.post("admin", CID_B, {"k": k, "f": "note", "v": "from admin"})
        self.assertEqual(json.loads(w.get("owner", f"/gallery/{CID_B}/api/state")[2])["state"][k]["by"]["note"]["who"], "tester")
        for doc in ({"k": "nope.png", "f": "flag", "v": "keep"}, {"k": k, "f": "tags", "v": ["bad reason"]},
                    {"k": k, "f": "stars", "v": 9}, {"k": k, "f": "note", "v": "x" * 1001}, {"k": k}, [1, 2], "str"):
            with self.subTest(doc=doc):
                st, _, body = w.post("owner", CID_B, doc)
                self.assertEqual(st, 400, body)
        st, _, _ = w.req("owner", "POST", f"/gallery/{CID_B}/api/event", b"x" * 20000,
                         extra={"X-Requested-With": "gallery", "Content-Type": "application/json"})
        self.assertEqual(st, 413)
        st, _, _ = w.req("owner", "POST", f"/gallery/{CID_B}/api/event", b"{not json",
                         extra={"X-Requested-With": "gallery", "Content-Type": "application/json"})
        self.assertEqual(st, 400)


class Pages(Base):
    def page(self, who, path):
        st, _, body = self.w.get(who, path)
        self.assertEqual(st, 200, path)
        return body.decode("utf-8")

    def test_with_montages_boxes_link_the_right_tile(self):
        page = self.page("owner", f"/gallery/{CID_A}/")
        boxes = re.findall(r'<a class="cj-gallery-box" data-cid="[^"]+" data-k="([^"]+)" href="([^"]+)" style="([^"]+)"', page)
        self.assertEqual(len(boxes), 9)
        for i, (k, href, style) in enumerate(boxes):
            self.assertEqual(k, NAMES[i])
            self.assertEqual(href, f"/gallery/{CID_A}/t/{NAMES[i]}")
            r, c = divmod(i, 3)
            self.assertIn(f"left:{100 * c * 120 / 360:.3f}%;top:{100 * r * 90 / 270:.3f}%;width:33.333%;height:33.333%", style)
            self.assertEqual(self.w.get("owner", href)[2], tile_png(i))                  # the link serves THAT tile
        self.assertEqual(len(re.findall(r'class="cj-gallery-tile"', page)), 9)
        self.assertIn("Sample one", page)
        self.assertIn('id="cj-gallery"', page)

    def test_without_montages(self):
        page = self.page("owner", f"/gallery/{CID_B}/")
        self.assertNotIn("cj-gallery-stage", page)
        self.assertEqual(len(re.findall(r'class="cj-gallery-tile"', page)), 9)           # grouped by montage from MANIFEST
        self.assertIn('data-m="m1"', page)
        rep = self.page("owner", f"/gallery/{CID_B}/report")
        self.assertIn("No stitched montages", rep)
        self.assertIn('data-col="n">9<', rep)

    def test_page_weight_and_no_inline_script(self):
        for path in (f"/gallery/{CID_A}/", f"/gallery/{CID_B}/", "/gallery/", f"/gallery/{CID_A}/report",
                     f"/gallery/compare?c={CID_A}&c={CID_B}"):
            page = self.page("owner", path)
            for src in re.findall(r'<img[^>]+src="([^"]+)"', page):
                self.assertRegex(src, r"/(ts|mp|ms|rep)/", f"{path}: a full-size image is loaded at page load: {src}")
            for m in re.finditer(r"<script\b([^>]*)>", page):
                self.assertIn("src=", m.group(1), "inline script in " + path)
            self.assertNotRegex(page, r'\son[a-z]+=')
        self.assertEqual(self.w.get("owner", "/gallery/gallery.js")[0], 200)

    def test_compare_side_by_side(self):
        page = self.page("owner", f"/gallery/compare?c={CID_A}&c={CID_B}")
        self.assertEqual(page.count('class="cj-gallery-conv"'), 2)
        self.assertIn(f'data-cids="{CID_A} {CID_B}"', page)
        rep = self.page("owner", f"/gallery/compare/report?c={CID_A}&c={CID_B}")
        self.assertEqual(rep.count('class="cj-gallery-rconv"'), 2)

    def test_xss_inert(self):
        c = mark(self.w.x)
        G.append_event(c, "alyssia", NAMES[1], "note", XNOTE)
        G.append_event(c, "alyssia", NAMES[1], "stars", 4)
        for path in (f"/gallery/{CID_X}/", f"/gallery/{CID_X}/report", "/gallery/", f"/gallery/compare?c={CID_X}",
                     f"/gallery/compare/report?c={CID_X}"):
            page = self.page("owner", path)
            self.assertNotIn(XSS, page, path)
            self.assertNotIn("<script>alert", page, path)
            self.assertNotIn("<img src=x", page)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", self.page("owner", f"/gallery/{CID_X}/report"))
        self.assertIn("&lt;script&gt;alert(&quot;note&quot;)", self.page("owner", f"/gallery/{CID_X}/report"))
        st, _, zbody = self.w.get("owner", f"/gallery/{CID_X}/report.zip")
        z = zipfile.ZipFile(io.BytesIO(zbody))
        doc = z.read("report.html").decode("utf-8")
        self.assertNotIn(XSS, doc)
        self.assertNotIn("<script", doc.lower())
        self.assertIn("&lt;script&gt;alert(&quot;note&quot;)", doc)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", doc)
        # JSON from the API is data, served as JSON
        st, h, body = self.w.get("owner", f"/gallery/{CID_X}/api/state")
        self.assertIn("application/json", T.hget(h, "content-type"))


class Report(Base):
    def test_known_answer_counts(self):
        c = mark(self.w.a)
        data = G.report_data([c], {c.cid: G.state_of(c)})
        tot = data[0]["total"]
        self.assertEqual((tot["n"], tot["keep"], tot["maybe"], tot["reject"], tot["unmarked"]), (9, 4, 1, 2, 2))
        self.assertEqual(dict(tot["reasons"]), {"thick ice": 1, "empty": 1})
        g = data[0]["grids"][0]
        self.assertEqual((g["tally"]["n"], g["montages"][0]["tally"]["keep"]), (9, 4))
        self.assertEqual([s["k"] for s in data[0]["stars"]], [NAMES[0]])
        st, _, body = self.w.get("owner", f"/gallery/{CID_A}/report")
        page = body.decode()
        row = re.search(r'<tr class="cj-gallery-tot">(.*?)</tr>', page, re.S).group(1)
        self.assertEqual(re.findall(r'data-col="(\w+)">(\d+)<', row),
                         [("n", "9"), ("keep", "4"), ("maybe", "1"), ("reject", "2"), ("unmarked", "2")])
        self.assertIn("empty 1, thick ice 1", row)
        # acquisition: both pixel sizes, mag, defocus, dose, time
        for want in ("7.362", "14.72", "1900", "-50", "1.2", "2026-11-06 10:15", "Test scope"):
            self.assertIn(want, page)
        self.assertIn("nice field", page)
        self.assertIn(f"/gallery/{CID_A}/rep/", page)                                     # the starred tile
        self.assertIn("report.zip", page)

    def test_report_zip_standalone(self):
        mark(self.w.a)
        st, h, body = self.w.get("owner", f"/gallery/{CID_A}/report.zip")
        self.assertEqual(st, 200)
        self.assertIn("application/zip", T.hget(h, "content-type"))
        self.assertIn("attachment", T.hget(h, "content-disposition"))
        z = zipfile.ZipFile(io.BytesIO(body))
        self.assertIsNone(z.testzip())
        names = z.namelist()
        self.assertEqual(names[0], "report.html")
        doc = z.read("report.html").decode("utf-8")
        self.assertNotIn("<script", doc.lower())
        self.assertNotRegex(doc, r"https?://")
        self.assertNotIn("/gallery/", doc)                                                # relative paths only
        refs = re.findall(r'<img src="([^"]+)"', doc)
        self.assertEqual(len(refs), 2)                                                    # starred tile + the montage
        for r in refs:
            self.assertIn(r, names)
            self.assertFalse(r.startswith("/"))
        self.assertTrue(z.read(refs[0]).startswith(b"\xff\xd8"))
        self.assertIn('data-col="keep">4<', doc)
        self.assertIn("<style>", doc)

    def test_images_zip_streamed_and_complete(self):
        port = self.w.portal.port
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
        c.request("GET", f"/gallery/{CID_A}/images.zip", headers=T.cookie(self.w.tok["owner"]))
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        self.assertIsNone(r.getheader("Content-Length"))                                  # streamed, not buffered
        self.assertEqual((r.getheader("Transfer-Encoding") or "").lower(), "chunked")
        body = r.read()
        c.close()
        z = zipfile.ZipFile(io.BytesIO(body))
        self.assertIsNone(z.testzip())
        names = set(z.namelist())
        self.assertEqual({f"tiles/{n}" for n in NAMES} | {"montages/m1_stitched.jpg", "MANIFEST.csv", "acquisition.json",
                                                          "montages/layout.json", "gallery_verdicts.csv"}, names)
        for i, n in enumerate(NAMES):
            self.assertEqual(z.read(f"tiles/{n}"), tile_png(i))
        self.assertEqual(z.read("MANIFEST.csv"), (self.w.a / "tiles" / "MANIFEST.csv").read_bytes())
        # compare: one folder per conversion
        st, _, body = self.w.get("owner", f"/gallery/compare/images.zip?c={CID_A}&c={CID_B}")
        names = zipfile.ZipFile(io.BytesIO(body)).namelist()
        self.assertTrue(any(n.startswith(CID_A + "/tiles/") for n in names) and any(n.startswith(CID_B + "/tiles/") for n in names))

    def test_two_downloads_per_person(self):
        s1, s2 = G.take_slot("alyssia"), G.take_slot("alyssia")
        self.assertTrue(s1 and s2)
        self.assertIsNone(G.take_slot("alyssia"))
        st, h, _ = self.w.get("owner", f"/gallery/{CID_A}/images.zip")
        self.assertEqual(st, 429)
        self.assertIsNotNone(T.hget(h, "retry-after"))
        self.assertEqual(self.w.get("admin", f"/gallery/{CID_A}/images.zip")[0], 200)     # another person is not held up
        s1.release()
        self.assertEqual(self.w.get("owner", f"/gallery/{CID_A}/images.zip")[0], 200)     # and a freed slot works
        st, _, _ = self.w.get("owner", f"/gallery/{CID_A}/images.zip")
        self.assertEqual(st, 200)                                                          # finished streams freed theirs

    def test_scalebar_helper(self):
        self.assertEqual(G.nice_bar(1.4724, 1440)[0], 500)                                # 500 nm = 340 px of 1440 (<25%)
        self.assertEqual(G.nice_bar(1.4724, 1440)[1], 500 / 1.4724)
        self.assertEqual(G.nice_bar(0.01, 100), (None, None))


# ---------------------------------------------------------------- browser
def launch(p):
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:                                                                 # noqa: BLE001
            continue
    return None


class Browser(Base):
    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        super().setUpClass()
        cls.shots = Path(os.environ.get("CJ_SHOT_DIR") or tempfile.mkdtemp(prefix="cj_gal_shots_"))
        cls.shots.mkdir(parents=True, exist_ok=True)
        cls.pw = sync_playwright().start()
        cls.browser = launch(cls.pw)
        if cls.browser is None:
            cls.tearDownClass()
            raise unittest.SkipTest("no Edge / Chrome for Playwright")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "browser", None):
            cls.browser.close()
        if getattr(cls, "pw", None):
            cls.pw.stop()
        super().tearDownClass()

    def test_pages_and_keyboard(self):
        mark(self.w.a)
        origin = self.w.portal.origin
        for scheme in ("light", "dark"):
            for width in (390, 1280):
                with self.subTest(width=width, scheme=scheme):
                    ctx = self.browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
                    foreign, errors = [], []
                    ctx.route("**/*", lambda r: (r.continue_() if r.request.url.startswith(origin + "/")
                                                 else (foreign.append(r.request.url), r.abort())))
                    page = ctx.new_page()
                    page.on("pageerror", lambda e: errors.append(str(e)))
                    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
                    page.goto(f"{origin}/login?k={self.w.keys['owner']}")
                    tag = f"{scheme}-{width}"
                    for name, path in (("index", "/gallery/"), ("gallery", f"/gallery/{CID_A}/"),
                                       ("report", f"/gallery/{CID_A}/report"), ("compare", f"/gallery/compare?c={CID_A}&c={CID_B}")):
                        page.goto(origin + path)
                        page.wait_for_load_state("networkidle")
                        sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
                        self.assertLessEqual(sw, iw, f"horizontal scroll on {name}")
                        page.screenshot(path=str(self.shots / f"gallery-{name}-{tag}.png"), full_page=(name != "gallery"))
                    page.goto(f"{origin}/gallery/{CID_A}/")
                    page.wait_for_selector(".cj-gallery-tile[data-flag]")
                    self.assertEqual(page.locator('.cj-gallery-tile[data-flag="keep"]').count(), 4,
                                     self.w.get("owner", f"/gallery/{CID_A}/api/state")[2])       # painted from the state
                    first = page.locator(".cj-gallery-tile").first
                    first.click()
                    view = page.locator(".cj-gallery-view")
                    self.assertTrue(view.is_visible())
                    page.wait_for_function("document.querySelector('.cj-gallery-vwrap img').naturalWidth > 0")
                    self.assertTrue(page.locator(".cj-gallery-sbar").is_visible())
                    page.screenshot(path=str(self.shots / f"gallery-viewer-{tag}.png"))
                    page.keyboard.press("ArrowRight")
                    self.assertIn("3 of 10 shown", page.locator(".cj-gallery-vside .cj-mut").inner_text())
                    page.keyboard.press("r")
                    page.wait_for_function("document.querySelector('.cj-gallery-tile:nth-child(2)').dataset.flag === 'reject'")
                    page.keyboard.press("t")
                    page.keyboard.press("Space")                                                            # toggles the focused reason
                    page.wait_for_function("document.querySelector('.cj-gallery-chip[aria-pressed=true]') !== null")
                    page.keyboard.press("n")
                    page.keyboard.type("seen in browser")
                    page.keyboard.press("Escape")                                                           # leaves the note, keeps the view
                    self.assertTrue(view.is_visible())
                    page.keyboard.press("3")
                    page.keyboard.press("Escape")
                    self.assertFalse(view.is_visible())
                    page.wait_for_function("document.querySelector('.cj-gallery-tile:nth-child(2)').dataset.stars === '3'")
                    for _ in range(60):                                      # the saves are in flight: wait for the server
                        st = json.loads(self.w.get("owner", f"/gallery/{CID_A}/api/state")[2])["state"][NAMES[1]]
                        if (st["flag"], st["stars"], st["note"]) == ("reject", 3, "seen in browser") and st["tags"]:
                            break
                        page.wait_for_timeout(100)
                    self.assertEqual((st["flag"], st["stars"], st["tags"], st["note"]), ("reject", 3, ["thick ice"], "seen in browser"))
                    page.select_option('[data-filter="flag"]', "keep")
                    self.assertEqual(page.locator(".cj-gallery-tile:visible").count(), 3)               # NAMES[1] is reject now
                    page.select_option('[data-filter="flag"]', "")
                    page.select_option('[data-filter="tag"]', "thick ice")
                    self.assertEqual(page.locator(".cj-gallery-tile:visible").count(), 2)                  # NAMES[1] and NAMES[4]
                    # undo for the next round
                    for f, v in (("flag", "keep"), ("stars", 0), ("tags", []), ("note", "")):
                        self.w.post("owner", CID_A, {"k": NAMES[1], "f": f, "v": v})
                    self.assertEqual(foreign, [])
                    self.assertEqual(errors, [])
                    ctx.close()


if __name__ == "__main__":
    unittest.main()
