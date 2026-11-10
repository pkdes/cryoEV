"""
Conversion settings in the portal (round 7 phase 2, piece CU): the recipe readout on the raw form, the settings on the
conversion page, the Adjust panel (one-tile preview, re-convert as a NEW conversion with --params), the tiles zip.

  * Stub child: portal.convert_settings.run_cli is replaced, so no processing code runs. The tests check what the
    portal ASKS the CLI (argv) and what it renders from the answer.
  * Known answer (needs CJ_PROCESSING_REPO and a demo conversion folder in CJ_CONVSET_DEMO): a scratch copy of a real
    conversion goes through the portal's preview handler with the recipe's own values; after.png must be
    byte-identical to the conversion's existing tile (no float drift on the way: 99 stays 99). Screenshots at 1280 /
    390 px, light and dark, go to $CJ_SHOT_DIR when set.

  python -m unittest portal.tests.test_convert_settings_ui -v
"""
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import convert_settings as CS  # noqa: E402
from portal import predict_sources as PSRC  # noqa: E402
from portal.modules import predict as P  # noqa: E402
import test_predict_convert as TPC  # noqa: E402

XSS = TPC.XSS
CID = "20261101_a1b2c3"
OWNER = "adv"
PNG = TPC.png16()


def describe_doc(recipe="20261027", ov=None, text="Gaussian bandpass, keeps structures >2 nm", recorded=True):
    rec = {"clip_lo_pct": 1.0, "clip_hi_pct": 99.0, "bandpass_nm": 2.0, "denoise": True}
    eff = {**rec, **(ov or {})}
    return {"ok": True, "mode": "describe", "recipe": recipe,
            "denoise": {"on": True, "method": "bandpass", "bandpass_nm": 2.0, "max_nm": None, "median_target_nm": None,
                        "match_contrast": True, "text": text},
            "contrast": {"clip_lo_pct": 1.0, "clip_hi_pct": 99.0, "sampling": 7.362,
                         "sampling_text": "clip levels measured at 7.362 A/px (finer data block-averaged first)",
                         "levels_from": "the clean pass (denoise_match_contrast)"},
            "pixel": {"target_A_px": 14.724, "target_nm_px": 1.4724}, "variant": "denoised", "params": rec,
            "params_allowed": {}, "source": {"recipe_file": f"datasets/process_{recipe}.py", "recipe_md5": "cec10c3b98b"},
            "params_effective": eff, "params_overrides": ov or {}, "recorded": recorded, "warnings": []}


class Stub:
    """A fake run_cli: records the args, answers describe / preview."""

    def __init__(self):
        self.calls = []
        self.gate = None                  # an Event the preview waits for
        self.started = threading.Event()
        self.text = "Gaussian bandpass, keeps structures >2 nm"

    def __call__(self, cmd, repo, args, timeout, cwd):
        self.calls.append([str(a) for a in args])
        if args[0] == "describe":
            if "--conv" in args:
                meta = json.loads((Path(args[args.index("--conv") + 1]) / "convert.json").read_text(encoding="utf-8"))
                return describe_doc(ov=meta.get("params"), text=self.text), None
            return describe_doc(args[args.index("--recipe") + 1], text=self.text), None
        out = Path(args[args.index("--out") + 1])
        self.started.set()
        if self.gate is not None:
            self.gate.wait(30)
        out.mkdir(parents=True, exist_ok=True)
        (out / "before.png").write_bytes(PNG)
        (out / "after.png").write_bytes(PNG)
        ov = json.loads(args[args.index("--params") + 1]) if "--params" in args else {}
        d = describe_doc(ov=ov)
        return {"ok": True, "mode": "preview", "params_effective": d["params_effective"], "params_recipe": d["params"],
                "params_conversion": d["params"], "params_overrides": ov, "compute": "CPU", "warnings": [],
                "seconds": 1.0}, None

    def of(self, kind):
        return [c for c in self.calls if c[0] == kind]


class Base(TPC.Base):
    """The World / TestClient of test_predict_convert, with a finished conversion made by hand."""

    def setUp(self):
        super().setUp()
        CS.clear_cache()
        self.stub = Stub()
        self.patch = mock.patch.object(CS, "run_cli", self.stub)
        self.patch.start()
        # the fake converter understands --params (the test_predict_convert one does not)
        self.w.fake.write_text(TPC.FAKE_CONVERT.replace('ap.add_argument("--stitch-only", type=Path)',
                                                        'ap.add_argument("--stitch-only", type=Path)\n'
                                                        'ap.add_argument("--params")'), encoding="utf-8")

    def tearDown(self):
        self.patch.stop()
        super().tearDown()
        for c in self.convs():
            shutil.rmtree(self.w.root / P.RAW_TILES / c, ignore_errors=True)
        shutil.rmtree(self.w.root / P.PORTAL_TMP / P.CPREV, ignore_errors=True)
        self.w.argv_log.unlink(missing_ok=True)

    def make_done(self, cid=CID, owner=OWNER, params=None, keep_input=True):
        d = self.w.root / P.RAW_TILES / cid
        (d / "tiles").mkdir(parents=True)
        names = ["g1_t__grid1_1_r0c0_t00__denoised_nobar_14p72Apx.png", "g1_t__grid1_1_r0c1_t01__denoised_nobar_14p72Apx.png"]
        for n in names:
            (d / "tiles" / n).write_bytes(PNG)
        with open(d / "tiles" / "MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["output_name", "source_file", "grid", "sample_label", "montage", "nm_per_px", "compute"])
            for n in names:
                w.writerow([n, "x.jpg", 1, "Grid 1", "grid1_1", "1.47240", "CPU"])
        inp = d / "input"
        inp.mkdir()
        (inp / "grid1-pos1.mrc").write_bytes(TPC.mrc_bytes())
        (inp / "grid1-pos1.mrc.mdoc").write_text(TPC.MDOC, encoding="utf-8")
        meta = {"schema": "cj_convert/1", "id": cid, "owner": owner, "session": cid.split("_")[0], "recipe": "20261027",
                "gpu": False, "source": {"kind": "upload", "files": ["grid1-pos1.mrc", "grid1-pos1.mrc.mdoc"]},
                "inputs": [str(inp / "grid1-pos1.mrc.mdoc")], "sample_key": None, "created": "2026-11-09T10:00:00",
                "input_bytes": 5000}
        if params:
            meta["params"] = params
        (d / "convert.json").write_text(json.dumps(meta), encoding="utf-8")
        st = {"state": "done", "finished": "2026-11-09T10:01:00", "warnings": [], "n_tiles": 2,
              "result": {"ok": True, "nm_per_px": [1.4724], "montages": ["grid1_1"], "compute": "CPU"}}
        (d / "status.json").write_text(json.dumps(st), encoding="utf-8")
        if not keep_input:
            shutil.rmtree(inp)
            st["inputs_removed"] = "2026-11-09T12:00:00"
            (d / "status.json").write_text(json.dumps(st), encoding="utf-8")
        return d

    def settings(self, cid=CID, who="advanced"):
        return self.get(f"/predict/convert/{cid}/settings", who)

    def preview(self, doc, cid=CID, who="advanced"):
        return self.jpost(f"/predict/convert/{cid}/preview", doc, who)


class Readout(Base):
    def test_raw_form_readout_from_describe(self):
        html = self.page("/predict/new/raw/", "advanced")
        self.assertIn("data-cj-recipe-readout", html)
        self.assertEqual(self.stub.of("describe"), [])                 # the page itself never runs the child
        r = self.get("/predict/api/recipe/20261027", "advanced")
        self.assertEqual(r.status_code, 200)
        h = r.json()["html"]
        for want in ("Gaussian bandpass, keeps structures &gt;2 nm", "1 % / 99 %", "1.4724 nm/px", "7.362 A/px",
                     "Tile variant"):
            self.assertIn(want, h)
        self.assertEqual(r.json()["summary"], "bandpass 2 nm · clip 1–99 %")
        self.assertEqual(self.stub.of("describe")[0], ["describe", "--recipe", "20261027", "--json"])
        self.get("/predict/api/recipe/20261027", "advanced")             # cached: one child for two requests
        self.assertEqual(len(self.stub.of("describe")), 1)
        self.get("/predict/api/recipe/default", "advanced")              # the chosen one is "default"
        html = self.page("/predict/new/raw/", "advanced")                # now the page carries the values
        self.assertIn("1 % / 99 %", html)
        self.assertIn("20261027 · bandpass 2 nm · clip 1–99 %", html)

    def test_recipe_api_gates(self):
        self.assertEqual(self.get("/predict/api/recipe/20261027", "basic").status_code, 403)
        self.assertEqual(self.get("/predict/api/recipe/nope", "advanced").status_code, 404)
        self.assertEqual(self.get("/predict/api/recipe/..%2f..%2fx", "advanced").status_code, 404)
        self.assertEqual(self.stub.calls, [])

    def test_recipe_names_and_values_are_escaped(self):
        self.stub.text = XSS
        with mock.patch.object(P, "recipes", return_value=[XSS, "20261027"]):
            html = self.page("/predict/new/raw/", "advanced")
            self.assertNotIn(XSS, html)
            self.assertIn("&lt;img src=x onerror=alert(1)&gt;", html)
            self.assertEqual(self.get("/predict/api/recipe/" + "%3Cimg", "advanced").status_code, 404)
        h = self.get("/predict/api/recipe/20261027", "advanced").json()["html"]
        self.assertNotIn(XSS, h)

    def test_conversion_page_values_and_custom(self):
        self.make_done()
        self.make_done("20261101_ffffff", params={"clip_hi_pct": 99.5, "denoise": False})
        j = self.settings().json()
        self.assertIn("1 % / 99 %", j["html"])
        self.assertNotIn("custom settings", j["html"])
        j = self.settings("20261101_ffffff").json()
        self.assertIn("custom settings", j["html"])
        self.assertIn("Differences from the recipe", j["html"])
        self.assertIn("<td>Clip high %</td><td>99</td><td><b>99.5</b></td>", j["html"])
        self.assertIn("<td>Denoise</td><td>on</td><td><b>off</b></td>", j["html"])
        self.assertEqual(self.stub.of("describe")[0][:2], ["describe", "--conv"])
        page = self.page(f"/predict/convert/{CID}/", "advanced")
        self.assertIn('id="cj-convset"', page)
        self.assertIn("Download tiles (zip)", page)


class Gates(Base):
    def test_not_yours_is_the_same_404(self):
        self.make_done()
        ghost = "20261101_000000"
        for who in ("adv2", "basic"):
            for method, path in (("get", "settings"), ("get", "tiles.zip"), ("get", "preview/aaaaaaaaaaaa/after.png")):
                a = self.tc.get(f"/predict/convert/{CID}/{path}", headers=self.h(who, origin=False))
                b = self.tc.get(f"/predict/convert/{ghost}/{path}", headers=self.h(who, origin=False))
                self.assertEqual((a.status_code, a.text), (b.status_code, b.text), (who, path))
                self.assertEqual(a.status_code, 404)
            a = self.preview({"tile": "x", "params": {}}, CID, who)
            b = self.preview({"tile": "x", "params": {}}, ghost, who)
            self.assertEqual((a.status_code, a.text), (b.status_code, b.text))
            self.assertEqual(a.status_code, 404)
            a = self.form(f"/predict/convert/{CID}/reconvert", {"clip_hi_pct": "99.5"}, who)
            b = self.form(f"/predict/convert/{ghost}/reconvert", {"clip_hi_pct": "99.5"}, who)
            self.assertEqual((a.status_code, a.text), (b.status_code, b.text))
            self.assertEqual(a.status_code, 404)
        self.assertEqual(self.stub.of("preview"), [])
        self.assertEqual(self.convs(), [CID])

    def test_tiles_zip_owner_and_admin_only(self):
        d = self.make_done()
        r = self.tc.get(f"/predict/convert/{CID}/tiles.zip", headers=self.h("advanced", origin=False))
        self.assertEqual(r.status_code, 200)
        names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
        self.assertEqual(names[0], f"{CID}/MANIFEST.csv")
        self.assertEqual(len(names), 3)
        self.assertEqual(self.tc.get(f"/predict/convert/{CID}/tiles.zip", headers=self.h("admin", origin=False)).status_code, 200)
        self.assertEqual(self.tc.get(f"/predict/convert/{CID}/tiles.zip", headers=self.h("adv2", origin=False)).status_code, 404)
        from portal.modules import gallery as GAL
        slots = [GAL.take_slot(OWNER) for _ in range(GAL.MAX_ZIPS)]
        try:
            r = self.tc.get(f"/predict/convert/{CID}/tiles.zip", headers=self.h("advanced", origin=False))
            self.assertEqual(r.status_code, 429)
        finally:
            for s in slots:
                s and s.release()
        self.assertTrue(d.exists())


class Preview(Base):
    GOOD = {"clip_lo_pct": "1", "clip_hi_pct": "99", "bandpass_nm": "2", "denoise": "true"}

    def tile(self):
        return P.tile_names(self.w.root / P.RAW_TILES / CID)[0]

    def test_preview_passes_values_exactly(self):
        self.make_done()
        r = self.preview({"tile": self.tile(), "params": self.GOOD})
        self.assertEqual(r.status_code, 200, r.text)
        j = r.json()
        a = self.stub.of("preview")[0]
        self.assertEqual(a[a.index("--params") + 1], '{"clip_lo_pct":1,"clip_hi_pct":99,"bandpass_nm":2,"denoise":true}')
        self.assertEqual(a[a.index("--tile") + 1], self.tile())
        self.assertNotIn("--gpu", a)
        img = self.tc.get(j["after"], headers=self.h("advanced", origin=False))
        self.assertEqual((img.status_code, img.content), (200, PNG))
        self.assertEqual(self.tc.get(j["after"], headers=self.h("adv2", origin=False)).status_code, 404)
        # decimals are not re-formatted: 99.5 stays 99.5
        self.preview({"tile": self.tile(), "params": {**self.GOOD, "clip_hi_pct": "99.5", "clip_lo_pct": 0.3}})
        a = self.stub.of("preview")[1]
        self.assertIn('"clip_lo_pct":0.3,"clip_hi_pct":99.5', a[a.index("--params") + 1])

    def test_refused_params_never_reach_the_cli(self):
        self.make_done()
        t = self.tile()
        bad = [{"clip_hi_pct": "120"}, {"clip_hi_pct": 94.9}, {"clip_lo_pct": -1}, {"clip_lo_pct": "5.1"},
               {"bandpass_nm": "0.4"}, {"bandpass_nm": 11}, {"bandpass_nm": True}, {"clip_hi_pct": "NaN"},
               {"clip_hi_pct": "1e3"}, {"clip_hi_pct": None, "evil": 1}, {"denoise": "maybe"}, {"rm": "-rf"},
               {"denoise": "false", "bandpass_nm": "2"}, "text", [1]]
        for params in bad:
            r = self.preview({"tile": t, "params": params})
            self.assertEqual(r.status_code, 400, (params, r.text))
        self.assertEqual(self.preview({"tile": "../../x.png", "params": {}}).status_code, 400)
        self.assertEqual(self.preview({"tile": t + "x", "params": {}}).status_code, 400)
        r = self.tc.post(f"/predict/convert/{CID}/preview", content=b"{not json",
                         headers=self.h("advanced", **{"Content-Type": "application/json"}))
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.stub.of("preview"), [])
        # the same rules on the re-convert form
        for form in ({"clip_hi_pct": "120"}, {"bandpass_nm": "99"}, {"clip_lo_pct": "x"}):
            r = self.form(f"/predict/convert/{CID}/reconvert", {"denoise": "true", **form}, "advanced")
            self.assertEqual(r.status_code, 400)
        self.assertEqual(self.convs(), [CID])

    def test_one_preview_at_a_time(self):
        self.make_done()
        self.stub.gate = threading.Event()
        out = {}
        th = threading.Thread(target=lambda: out.setdefault("r", self.preview({"tile": self.tile(), "params": self.GOOD})))
        th.start()
        try:
            self.assertTrue(self.stub.started.wait(20))
            r2 = self.preview({"tile": self.tile(), "params": self.GOOD}, who="admin")
            self.assertEqual(r2.status_code, 429, r2.text)
            self.assertEqual(r2.headers.get("retry-after"), "10")
        finally:
            self.stub.gate.set()
            th.join(30)
        self.assertEqual(out["r"].status_code, 200)
        self.assertEqual(len(self.stub.of("preview")), 1)
        self.assertEqual(self.preview({"tile": self.tile(), "params": self.GOOD}).status_code, 200)   # free again

    def test_failed_preview_is_reported_and_cleaned(self):
        self.make_done()
        with mock.patch.object(CS, "run_cli", return_value=(None, "not enough memory")):
            r = self.preview({"tile": self.tile(), "params": self.GOOD})
        self.assertEqual(r.status_code, 502)
        self.assertIn("not enough memory", r.json()["error"])
        left = [p for p in (self.w.root / P.PORTAL_TMP / P.CPREV).rglob("*") if p.is_file()]
        self.assertEqual(left, [])

    def test_swept_input_disables_everything(self):
        self.make_done(keep_input=False)
        html = self.settings().json()["html"]
        self.assertIn('data-disabled="1"', html)
        self.assertIn("The raw files were removed", html)
        self.assertNotIn("data-cj-preview", html)
        r = self.preview({"tile": self.tile(), "params": self.GOOD})
        self.assertEqual(r.status_code, 409)
        self.assertIn("removed", r.json()["error"])
        r = self.form(f"/predict/convert/{CID}/reconvert", {"clip_hi_pct": "99.5", "denoise": "true"}, "advanced")
        self.assertEqual(r.status_code, 409)
        self.assertEqual((self.stub.of("preview"), self.convs()), ([], [CID]))

    def test_panel_present_when_kept(self):
        self.make_done()
        html = self.settings().json()["html"]
        for want in ('data-cj-slider', 'name="clip_lo_pct"', 'name="clip_hi_pct"', 'name="bandpass_nm"',
                     'data-cj-denoise', "data-cj-tile", "Re-convert with these settings"):
            self.assertIn(want, html)
        self.assertNotIn("data-disabled", html)
        # the admin sees it for someone else's conversion too
        self.assertIn("data-cj-slider", self.settings(who="admin").json()["html"])


class Reconvert(Base):
    def test_new_id_with_params_and_old_untouched(self):
        d = self.make_done()
        before = TPC.tree(d)
        r = self.form(f"/predict/convert/{CID}/reconvert",
                      {"clip_lo_pct": "1", "clip_hi_pct": "99.5", "bandpass_nm": "2", "denoise": "true"}, "advanced")
        self.assertEqual(r.status_code, 303, r.text[:400])
        new = TPC.CID_RE.match(r.headers["location"]).group(1)
        self.assertNotEqual(new, CID)
        self.w.settle()
        meta = json.loads((self.w.root / P.RAW_TILES / new / "convert.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["params"], {"clip_hi_pct": 99.5})            # only what differs from the recipe
        self.assertEqual((meta["derived_from"], meta["owner"], meta["recipe"]), (CID, OWNER, "20261027"))
        self.assertEqual(TPC.tree(d), before)                              # the old one: not a byte changed
        self.assertTrue((self.w.root / P.RAW_TILES / new / "input" / "grid1-pos1.mrc").is_file())
        calls = [json.loads(x)["argv"] for x in self.w.argv_log.read_text(encoding="utf-8").splitlines()]
        argv = [c for c in calls if "--in" in c][-1]
        self.assertEqual(argv[argv.index("--params") + 1], '{"clip_hi_pct":99.5}')
        self.assertEqual(argv[argv.index("--recipe") + 1], "20261027")
        page = self.settings(new).json()["html"]
        self.assertIn("custom settings", page)
        self.assertIn("Re-converted from", page)
        self.assertIn("<td>Clip high %</td><td>99</td><td><b>99.5</b></td>", page)
        self.assertNotIn("--gpu", argv)

    def test_denoise_off_and_nothing_to_change(self):
        self.make_done()
        r = self.form(f"/predict/convert/{CID}/reconvert", {"clip_lo_pct": "1", "clip_hi_pct": "99", "bandpass_nm": "2",
                                                            "denoise": "true"}, "advanced")
        self.assertEqual(r.status_code, 400)                               # the recipe's own values
        self.assertEqual(self.convs(), [CID])
        r = self.form(f"/predict/convert/{CID}/reconvert", {"clip_lo_pct": "1", "clip_hi_pct": "99"}, "advanced")
        self.assertEqual(r.status_code, 303)                               # denoise off (box not ticked)
        self.w.settle()
        new = [c for c in self.convs() if c != CID][0]
        meta = json.loads((self.w.root / P.RAW_TILES / new / "convert.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["params"], {"denoise": False})

    def test_basic_and_other_users_refused(self):
        self.make_done()
        self.assertEqual(self.form(f"/predict/convert/{CID}/reconvert", {"clip_hi_pct": "99.5"}, "basic").status_code, 404)
        self.assertEqual(self.form(f"/predict/convert/{CID}/reconvert", {"clip_hi_pct": "99.5"}, "adv2").status_code, 404)


class ParamUnits(unittest.TestCase):
    def test_numbers_do_not_drift(self):
        self.assertEqual(CS.params_json(CS.clean_params({"clip_hi_pct": "99"})), '{"clip_hi_pct":99}')
        self.assertEqual(CS.params_json(CS.clean_params({"clip_hi_pct": 99.0, "clip_lo_pct": "0.1"})),
                         '{"clip_lo_pct":0.1,"clip_hi_pct":99}')
        self.assertEqual(CS.clean_params({"bandpass_nm": 0.1 + 0.2 + 0.2}), {"bandpass_nm": 0.5})
        self.assertEqual(CS.clean_params({"denoise": "off"}), {"denoise": False})

    def test_refusals(self):
        for bad in ({"x": 1}, {"clip_hi_pct": float("nan")}, {"clip_hi_pct": float("inf")}, {"clip_hi_pct": True},
                    {"clip_hi_pct": "99,5"}, {"denoise": 2}, {"denoise": False, "bandpass_nm": 2}, []):
            with self.assertRaises(CS.ParamError, msg=str(bad)):
                CS.clean_params(bad)


# ------------------------------------------------------------------ known answer + screenshots (real CLI, scratch copy)
DEMO = os.environ.get("CJ_CONVSET_DEMO")
REPO_ENV = os.environ.get("CJ_PROCESSING_REPO")


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


@unittest.skipUnless(DEMO and REPO_ENV and Path(DEMO).is_dir(), "needs CJ_PROCESSING_REPO and CJ_CONVSET_DEMO (a demo conversion)")
class KnownAnswer(unittest.TestCase):
    """A scratch copy of a real conversion through the portal's preview handler: after.png (the recipe's own values,
    sent as the sliders send them) is byte-identical to the conversion's existing tile."""

    @classmethod
    def setUpClass(cls):
        cls.scratch = Path(tempfile.mkdtemp(prefix="cj_cs_ka_"))
        cls.w = TPC.World(real_repo=REPO_ENV, raw_root=cls.scratch)
        cls.cid = Path(DEMO).name
        cls.d = cls.scratch / "raw_tiles" / cls.cid
        shutil.copytree(DEMO, cls.d)
        meta = json.loads((cls.d / "convert.json").read_text(encoding="utf-8"))
        meta["owner"] = OWNER
        meta["inputs"] = [str(cls.d / "input" / Path(p).name) for p in meta["inputs"]]
        (cls.d / "convert.json").write_text(json.dumps(meta), encoding="utf-8")
        cls.tc = None
        from starlette.testclient import TestClient
        from portal import app as APP
        cls.tc = TestClient(APP.create_app(cls.w.cfg), base_url=TPC.ORIGIN, follow_redirects=False)
        cls.tc.__enter__()
        r = cls.tc.get(f"/login?k={cls.w.keys['advanced']}")
        cls.tok = r.headers["set-cookie"].split(";", 1)[0].split("=", 1)[1]
        cls.tc.cookies.clear()

    @classmethod
    def tearDownClass(cls):
        cls.tc.__exit__(None, None, None)
        cls.w.close()
        shutil.rmtree(cls.scratch, ignore_errors=True)

    def hdr(self, **kw):
        from portal import auth as A
        return {"Cookie": f"{A.COOKIE}={self.tok}", "Origin": TPC.ORIGIN, **kw}

    def test_preview_equals_the_conversions_tile(self):
        CS.clear_cache()
        seen = []
        real = CS.run_cli

        def spy(cmd, repo, args, timeout, cwd):
            seen.append([str(a) for a in args])
            return real(cmd, repo, args, timeout, cwd)
        j = self.tc.get(f"/predict/convert/{self.cid}/settings", headers=self.hdr()).json()
        self.assertIn("nm/px", j["html"])
        doc, err = CS.describe_conv(TPC.P._converter_cmd(), Path(REPO_ENV), self.d)
        self.assertIsNotNone(doc, err)
        rec = doc["params_effective"]
        form = {k: f"{rec[k]:g}" for k in ("clip_lo_pct", "clip_hi_pct")}
        form["denoise"] = "true" if rec["denoise"] else "false"
        if rec.get("bandpass_nm") is not None and rec["denoise"]:
            form["bandpass_nm"] = f"{rec['bandpass_nm']:g}"
        names = P.tile_names(self.d)
        tile = names[0]
        with mock.patch.object(CS, "run_cli", spy):
            r = self.tc.post(f"/predict/convert/{self.cid}/preview", content=json.dumps({"tile": tile, "params": form}).encode(),
                             headers=self.hdr(**{"Content-Type": "application/json"}))
        self.assertEqual(r.status_code, 200, r.text)
        jj = r.json()
        args = seen[0]
        sent = args[args.index("--params") + 1]
        print("\n  form values      :", form)
        print("  --params argv    :", sent)
        self.assertEqual(json.loads(sent), {k: (v == "true" if k == "denoise" else json.loads(v)) for k, v in form.items()})
        for v in ("0000", "0001", "9999"):
            self.assertNotIn(v, sent)                                     # no float drift (99 -> 99.0000001)
        after = self.tc.get(jj["after"], headers=self.hdr()).content
        want = (self.d / "tiles" / tile).read_bytes()
        print("  after.png md5    :", hashlib.md5(after).hexdigest())
        print("  conversion's tile:", hashlib.md5(want).hexdigest(), tile)
        print("  effective        :", jj["effective"], "compute", jj["compute"], "warnings", jj["warnings"])
        if (json.loads((self.d / "status.json").read_text(encoding="utf-8")).get("result") or {}).get("compute") == "CPU":
            self.assertEqual(hashlib.md5(after).hexdigest(), hashlib.md5(want).hexdigest())
        else:
            # a GPU conversion: the portal previews on the CPU (+-1 grey level on a few pixels, gotcha 31). So: (a) the
            # portal's after.png is byte-identical to the CLI run by hand with the same values (nothing lost on the way
            # through the handler), and (b) it is within +-1 grey level of the tile on < 0.01 % of the pixels.
            import numpy as np
            from PIL import Image
            out = self.scratch / "direct"
            doc, err = real(TPC.P._converter_cmd(), Path(REPO_ENV), ["preview", "--conv", str(self.d), "--tile", tile,
                            "--params", sent, "--out", str(out), "--json"], 300, CS.HOME)
            self.assertIsNotNone(doc, err)
            print("  direct CLI after :", md5(out / "after.png"))
            self.assertEqual(hashlib.md5(after).hexdigest(), md5(out / "after.png"))
            a = np.asarray(Image.open(io.BytesIO(after))).astype(int)
            b = np.asarray(Image.open(io.BytesIO(want))).astype(int)
            diff = np.abs(a - b)
            print("  vs GPU tile      : max |diff| %d, %.5f %% of pixels differ" % (diff.max(), 100.0 * (diff > 0).mean()))
            self.assertLessEqual(int(diff.max()), 1)
            self.assertLess(float((diff > 0).mean()), 1e-4)
        before = self.tc.get(jj["before"], headers=self.hdr()).content
        print("  before.png md5   :", hashlib.md5(before).hexdigest())
        self.assertEqual(md5(Path(DEMO) / "tiles" / tile), hashlib.md5(want).hexdigest())      # the original: untouched


if __name__ == "__main__":
    unittest.main()
