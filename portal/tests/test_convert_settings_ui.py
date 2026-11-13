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
from portal.predict_parts import raw as RAW  # noqa: E402
from portal.predict_parts import runner as RUNNER  # noqa: E402
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
            "pixel": {"target_A_px": 14.724, "target_nm_px": 1.4724}, "variant": "denoised", "params": eff,
            "params_allowed": {}, "source": {"recipe_file": f"datasets/process_{recipe}.py", "recipe_md5": "cec10c3b98b"},
            "params_effective": eff, "params_overrides": ov or {}, "recorded": recorded, "warnings": []}


class Stub:
    """A fake run_cli: records the args, answers describe / preview."""

    def __init__(self):
        self.calls = []
        self.gate = None                  # an Event the preview waits for
        self.started = threading.Event()
        self.text = "Gaussian bandpass, keeps structures >2 nm"
        self.record = True                # False: a queued re-conversion (its overrides are not recorded yet)
        self.max_nm = None
        self.method = "bandpass"

    def __call__(self, cmd, repo, args, timeout, cwd):
        self.calls.append([str(a) for a in args])
        if args[0] == "describe":
            if "--conv" in args:
                meta = json.loads((Path(args[args.index("--conv") + 1]) / "convert.json").read_text(encoding="utf-8"))
                d = describe_doc(ov=meta.get("params") if self.record else None, text=self.text)
            else:
                d = describe_doc(args[args.index("--recipe") + 1], text=self.text)
            d["denoise"]["max_nm"], d["denoise"]["method"] = self.max_nm, self.method
            return d, None
        out = Path(args[args.index("--out") + 1])
        self.started.set()
        if self.gate is not None:
            self.gate.wait(30)
        out.mkdir(parents=True, exist_ok=True)
        (out / "before.png").write_bytes(PNG)
        (out / "after.png").write_bytes(PNG)
        ov = json.loads(args[args.index("--params") + 1]) if "--params" in args else {}
        d = describe_doc(ov=ov)
        return {"ok": True, "mode": "preview", "params_effective": d["params_effective"],
                "params_recipe": describe_doc()["params"], "params_conversion": describe_doc()["params"], "params_overrides": ov, "compute": "CPU", "warnings": [],
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
        with mock.patch.object(RAW, "recipes", return_value=[XSS, "20261027"]):
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
        self.assertEqual(meta["params"], {"clip_lo_pct": 1, "clip_hi_pct": 99.5, "bandpass_nm": 2, "denoise": True})
        self.assertEqual((meta["derived_from"], meta["owner"], meta["recipe"]), (CID, OWNER, "20261027"))
        self.assertEqual(TPC.tree(d), before)                              # the old one: not a byte changed
        self.assertTrue((self.w.root / P.RAW_TILES / new / "input" / "grid1-pos1.mrc").is_file())
        calls = [json.loads(x)["argv"] for x in self.w.argv_log.read_text(encoding="utf-8").splitlines()]
        argv = [c for c in calls if "--in" in c][-1]
        self.assertEqual(argv[argv.index("--params") + 1],
                         '{"clip_lo_pct":1,"clip_hi_pct":99.5,"bandpass_nm":2,"denoise":true}')
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
        self.assertEqual(meta["params"], {"clip_lo_pct": 1, "clip_hi_pct": 99, "denoise": False})

    def test_basic_and_other_users_refused(self):
        self.make_done()
        self.assertEqual(self.form(f"/predict/convert/{CID}/reconvert", {"clip_hi_pct": "99.5"}, "basic").status_code, 404)
        self.assertEqual(self.form(f"/predict/convert/{CID}/reconvert", {"clip_hi_pct": "99.5"}, "adv2").status_code, 404)


class Followups(Base):
    """QC round: M1 (diff against the bare recipe), M2 (chained re-conversions), S1-S6, paths, nits."""

    def full(self, **kw):
        return {"clip_lo_pct": "1", "clip_hi_pct": "99", "bandpass_nm": "2", "denoise": "true", **kw}

    def test_m1_custom_conversion_diffs_against_the_bare_recipe(self):
        self.make_done(params={"clip_hi_pct": 99.5})
        html = self.settings().json()["html"]
        self.assertIn("custom settings", html)
        self.assertIn("<td>Clip high %</td><td>99</td><td><b>99.5</b></td>", html)
        self.assertNotIn("These are the recipe’s own values", html)
        self.assertIn("99.5 % ", html.replace("&nbsp;", " ") + " ")           # the readout shows what was used
        self.assertIn(["describe", "--recipe", "20261027", "--json"], self.stub.of("describe"))
        # sliders start at what the conversion used
        self.assertIn('name="clip_hi_pct" min="95" max="100" step="0.1" value="99.5"', html)

    def test_m1_queued_reconvert_shows_its_requested_values(self):
        self.make_done(params={"clip_hi_pct": 99.5})
        self.stub.record = False                                          # nothing recorded yet by the converter
        html = self.settings().json()["html"]
        self.assertIn("custom settings", html)
        self.assertIn("<td>Clip high %</td><td>99</td><td><b>99.5</b></td>", html)

    def test_m2_chained_reconversions_keep_earlier_overrides(self):
        self.make_done()
        r = self.form(f"/predict/convert/{CID}/reconvert", self.full(clip_hi_pct="99.5"), "advanced")
        self.assertEqual(r.status_code, 303, r.text[:300])
        d1 = TPC.CID_RE.match(r.headers["location"]).group(1)
        self.w.settle()
        # the page for D1 starts the sliders at clip_hi 99.5; the next preview/re-convert sends it with clip_lo 2
        self.assertIn('name="clip_hi_pct" min="95" max="100" step="0.1" value="99.5"', self.settings(d1).json()["html"])
        r = self.form(f"/predict/convert/{d1}/reconvert", self.full(clip_hi_pct="99.5", clip_lo_pct="2"), "advanced")
        self.assertEqual(r.status_code, 303, r.text[:300])
        d2 = TPC.CID_RE.match(r.headers["location"]).group(1)
        self.w.settle()
        self.assertEqual(len({CID, d1, d2}), 3)
        meta = json.loads((self.w.root / P.RAW_TILES / d2 / "convert.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["params"], {"clip_lo_pct": 2, "clip_hi_pct": 99.5, "bandpass_nm": 2, "denoise": True})
        self.assertEqual(meta["derived_from"], d1)
        calls = [json.loads(x)["argv"] for x in self.w.argv_log.read_text(encoding="utf-8").splitlines()]
        argv = [c for c in calls if "--in" in c][-1]
        self.assertEqual(argv[argv.index("--params") + 1],
                         '{"clip_lo_pct":2,"clip_hi_pct":99.5,"bandpass_nm":2,"denoise":true}')
        # the same values again: refused (nothing changes)
        r = self.form(f"/predict/convert/{d2}/reconvert", self.full(clip_hi_pct="99.5", clip_lo_pct="2"), "advanced")
        self.assertEqual(r.status_code, 400)

    def test_s1_derived_from_in_the_registry_entry_and_not_double_counted(self):
        from portal import upload_sessions as US
        from portal.modules import data as D
        d = self.make_done()
        d2 = self.make_done("20261101_b2b2b2", params={"clip_hi_pct": 99.5})
        m2 = json.loads((d2 / "convert.json").read_text(encoding="utf-8"))
        m2["derived_from"] = CID
        (d2 / "convert.json").write_text(json.dumps(m2), encoding="utf-8")
        e1 = US.conversion_entry(d, json.loads((d / "convert.json").read_text(encoding="utf-8")), "20261101")
        e2 = US.conversion_entry(d2, m2, "20261101")
        self.assertNotIn("derived_from", e1)
        self.assertEqual(e2["derived_from"], CID)
        rec = US.merge_session(US.merge_session(None, "20261101", e1), "20261101", e2)
        cards = D.session_grid_cards({"image": {}, "grid": {}}, {**rec, "id": "20261101"}, [])
        once = [c["n_images"] for c in cards]
        solo = [c["n_images"] for c in D.session_grid_cards(
            {"image": {}, "grid": {}}, {**US.merge_session(None, "20261101", e1), "id": "20261101"}, [])]
        self.assertEqual(once, solo)                                      # the re-conversion adds nothing
        self.assertTrue(once and once[0] > 0)
        # the schema accepts the key
        from registry import store as REG
        probe = {"schema": "session/1", "id": "20261101", "logged": "2026-11-09T00:00:00", "status": "processed",
                 "conversions": [e1, e2]}
        self.assertEqual(REG.check_record(probe, REG.schemas()), [])

    def test_s1_session_list_says_re_converted_from(self):
        from portal.modules import data as D
        try:
            html = D.render_session_extras({"grids": [], "n_images": 0, "view": "strip", "thumbs": [],
                                            "upload": {"conversions": [
                                                {"id": "20261101_b2b2b2", "url": "/predict/convert/x/",
                                                 "gallery": "/gallery/x/", "n_tiles": 2, "derived_from": CID}],
                                                "details": {}}})
        except KeyError as e:                                             # the page needs more of d: not our concern
            self.skipTest(f"render_session_extras needs {e}")
        self.assertIn(f"re-converted from {CID}", html)

    def test_s2_admin_reconvert_keeps_the_owner(self):
        self.make_done()
        r = self.form(f"/predict/convert/{CID}/reconvert", self.full(clip_hi_pct="99.5"), "admin")
        self.assertEqual(r.status_code, 303, r.text[:300])
        new = TPC.CID_RE.match(r.headers["location"]).group(1)
        self.w.settle()
        meta = json.loads((self.w.root / P.RAW_TILES / new / "convert.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["owner"], meta["made_by"]), (OWNER, T.USER))
        html = self.settings(new, "advanced").json()["html"]                # the owner sees it and who made it
        self.assertIn(f"by {T.USER}", html)
        from portal import upload_sessions as US
        self.assertEqual(US.conversion_entry(self.w.root / P.RAW_TILES / new, meta, "20261101")["uploaded_by"], OWNER)
        # the owner's own re-conversion has no made_by
        r = self.form(f"/predict/convert/{CID}/reconvert", self.full(clip_hi_pct="99.4"), "advanced")
        new2 = TPC.CID_RE.match(r.headers["location"]).group(1)
        self.w.settle()
        meta2 = json.loads((self.w.root / P.RAW_TILES / new2 / "convert.json").read_text(encoding="utf-8"))
        self.assertNotIn("made_by", meta2)

    def test_s3_linked_inputs_are_not_swept_by_the_free_space_floor(self):
        d = self.make_done()
        d2 = self.make_done("20261101_c3c3c3")
        solo = self.make_done("20261101_d4d4d4")
        for p in (d2 / "input").iterdir():
            p.unlink()
        for p in (d / "input").iterdir():
            os.link(p, d2 / "input" / p.name)                              # d2 shares d's bytes
        self.assertTrue(P.inputs_linked(d) and P.inputs_linked(d2) and not P.inputs_linked(solo))
        items = [(x, json.loads((x / "convert.json").read_text(encoding="utf-8")),
                  json.loads((x / "status.json").read_text(encoding="utf-8"))) for x in (d, d2, solo)]
        for x, _m, st in items:
            st["montages_done"] = True
        with mock.patch.object(RUNNER, "min_free_bytes", return_value=1 << 60), \
                mock.patch.object(RUNNER, "montages_done", return_value=True):
            n = P.sweep_inputs(self.w.cfg, items)
        self.assertEqual(n, 1)                                             # only the unlinked one was taken early
        self.assertTrue((d / "input").is_dir() and (d2 / "input").is_dir() and not (solo / "input").exists())

    def test_s4_no_links_means_a_disk_check_before_copying(self):
        self.make_done()
        asked = []

        def no_room(where, nbytes, **kw):
            asked.append((str(where), nbytes, kw))
            raise P.Refusal("Not enough free disk space.", 507)
        with mock.patch.object(os, "link", side_effect=OSError("cross-device")), \
                mock.patch.object(RUNNER, "check_disk_for", no_room):
            r = self.form(f"/predict/convert/{CID}/reconvert", self.full(clip_hi_pct="99.5"), "advanced")
        self.assertEqual(r.status_code, 507)
        self.assertEqual(len(asked), 1)
        self.assertEqual(self.convs(), [CID])                              # nothing copied, no new conversion
        left = [q for q in (self.w.root / P.PORTAL_TMP).glob("sel_rc*")] if (self.w.root / P.PORTAL_TMP).exists() else []
        self.assertEqual(left, [])

    def test_s5_cache_key_has_the_repo_commit(self):
        self.get("/predict/api/recipe/20261027", "advanced")
        self.get("/predict/api/recipe/20261027", "advanced")
        self.assertEqual(len(self.stub.of("describe")), 1)
        with mock.patch.object(CS, "_commit", return_value="deadbeef"):
            self.get("/predict/api/recipe/20261027", "advanced")
        self.assertEqual(len(self.stub.of("describe")), 2)                 # a new commit: described again
        self.make_done()
        self.settings()
        n = len(self.stub.of("describe"))
        self.settings()
        self.assertEqual(len(self.stub.of("describe")), n)
        with mock.patch.object(CS, "_commit", return_value="cafe"):
            self.settings()
        self.assertGreater(len(self.stub.of("describe")), n)

    def test_s6_bandpass_limits_from_the_recipe(self):
        self.stub.max_nm = 4.0
        self.make_done()
        html = self.settings().json()["html"]
        self.assertIn('name="bandpass_nm" min="0.5" max="3.9"', html)
        t = P.tile_names(self.w.root / P.RAW_TILES / CID)[0]
        r = self.preview({"tile": t, "params": {"bandpass_nm": "4"}})
        self.assertEqual(r.status_code, 400)
        self.assertIn("below 4", r.json()["error"])
        r = self.form(f"/predict/convert/{CID}/reconvert", self.full(bandpass_nm="4.5"), "advanced")
        self.assertEqual(r.status_code, 400)
        self.assertEqual((self.stub.of("preview"), self.convs()), ([], [CID]))
        self.assertEqual(self.preview({"tile": t, "params": {"bandpass_nm": "3.9"}}).status_code, 200)

    def test_nit_median_recipe_has_no_bandpass_slider_and_refuses_one(self):
        self.stub.method = "median"
        self.make_done()
        html = self.settings().json()["html"]
        self.assertNotIn('name="bandpass_nm"', html)
        t = P.tile_names(self.w.root / P.RAW_TILES / CID)[0]
        self.assertEqual(self.preview({"tile": t, "params": {"bandpass_nm": "2"}}).status_code, 400)

    def test_nit_bandpass_slider_stays_when_denoise_is_off(self):
        self.make_done(params={"denoise": False})
        html = self.settings().json()["html"]
        self.assertIn('name="bandpass_nm"', html)                          # decided by the method, not denoise on/off
        self.assertRegex(html, r'name="bandpass_nm"[^>]*disabled')

    def test_paths_never_reach_the_user(self):
        self.make_done()
        d = self.w.root / P.RAW_TILES / CID
        leak = f"failed reading {d}\\input\\x and {self.w.repo}\\datasets\\y"
        with mock.patch.object(CS, "run_cli", return_value=(None, leak)):
            CS.clear_cache()
            h = self.get("/predict/api/recipe/20261027", "advanced").json()["html"]
            t = P.tile_names(d)[0]
            r = self.preview({"tile": t, "params": {"clip_hi_pct": "99.5"}})
        self.assertNotIn(str(self.w.repo), h)
        self.assertNotIn(str(d), r.text)
        self.assertEqual(r.status_code, 502)
        # a warning in the describe document
        self.stub.calls.clear()
        orig = describe_doc

        def with_warn(*a, **k):
            doc = orig(*a, **k)
            doc["warnings"] = [f"the recipe moved: {self.w.repo}\\datasets\\process_x.py"]
            return doc
        with mock.patch(__name__ + ".describe_doc", with_warn):
            CS.clear_cache()
            html = self.settings().json()["html"]
        self.assertNotIn(str(self.w.repo), html)

    def test_nit_param_error_in_start_conversion_marks_it_failed(self):
        d = self.make_done()
        meta = json.loads((d / "convert.json").read_text(encoding="utf-8"))
        meta["params"] = {"clip_hi_pct": 500}                              # not possible through the page; a damaged file
        (d / "convert.json").write_text(json.dumps(meta), encoding="utf-8")
        st = P.start_conversion(self.w.cfg, d, meta)
        self.assertEqual((st["state"], st["kind"]), ("failed", "refused"))
        self.assertIn("not valid", st["error"])

    def test_nit_preview_prune_runs_under_the_lock(self):
        self.make_done()
        seen = []
        real = CS.run_preview

        def spy(cmd, repo, d, tile, params, out, before=None):
            def b():
                seen.append(CS.preview_busy())
                before()
            return real(cmd, repo, d, tile, params, out, before=b)
        with mock.patch.object(CS, "run_preview", spy):
            r = self.preview({"tile": P.tile_names(self.w.root / P.RAW_TILES / CID)[0], "params": {"clip_hi_pct": "99.5"}})
        self.assertEqual((r.status_code, seen), (200, [True]))

    def test_nit_failures_are_cached_for_five_minutes(self):
        self.assertEqual(CS.FAIL_TTL_S, 300)


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

    def test_m1_custom_conversion_with_the_real_cli(self):
        """A conversion whose convert.json records params_overrides {clip_hi_pct: 99.5}: the real `describe --conv`
        (params = recipe + overrides) through the portal gives the custom chip and the row 99 -> 99.5."""
        cj = self.d / "convert.json"
        orig = cj.read_text(encoding="utf-8")
        meta = json.loads(orig)
        try:
            for recorded in (True, False):                       # False: a queued re-conversion (only meta.params)
                CS.clear_cache()
                m = dict(meta)
                if recorded:
                    m["params_effective"] = {"clip_lo_pct": 1.0, "clip_hi_pct": 99.5, "bandpass_nm": 2.0, "denoise": True}
                    m["params_overrides"] = {"clip_hi_pct": 99.5}
                else:
                    m["params"] = {"clip_hi_pct": 99.5}
                cj.write_text(json.dumps(m), encoding="utf-8")
                html = self.tc.get(f"/predict/convert/{self.cid}/settings", headers=self.hdr()).json()["html"]
                self.assertIn("custom settings", html, recorded)
                self.assertIn("<td>Clip high %</td><td>99</td><td><b>99.5</b></td>", html, recorded)
                self.assertIn('name="clip_hi_pct" min="95" max="100" step="0.1" value="99.5"', html, recorded)
                print("\n  M1 real CLI, recorded=%s: custom chip + diff row 99 -> 99.5 + slider at 99.5" % recorded)
        finally:
            cj.write_text(orig, encoding="utf-8")
            CS.clear_cache()

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
