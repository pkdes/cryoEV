"""
Round 12 (#20 image metadata, #21 sample / montage groups): annotation/review_meta.py, annotation/review_groups.py and
their routes in review_server.py (GET /api/meta, GET /api/groups, POST /api/group), served here by the multi server in
this process on a free port over a temp fixture (meta_fixture.py: a prediction package with its prediction set and
conversion folders). Nothing outside temp folders is read or written (the registry fallback reads a temp registry).

  python -m unittest annotation.tests.test_review_meta_groups      (or discover -s annotation/tests)

- parse_tile_name: the processing repo's tile names (export_detector_tiles.training_name), the delivered stem, a
  Roboflow copy; it agrees with MANIFEST.csv montage for every fixture row; odd names give {}.
- package_meta: per image defocus by montage, tile from layout.json, recipe params; a tile missing from MANIFEST /
  layout / acquisition gets montage and row / col from its name and no defocus; missing / broken files leave fields
  out, never raise; no local path in the answer; an upload set: note, provenance (script name only), source manifest;
  the registry's session summary for a set made from a tile folder; a lab package: split only.
- routes: /api/meta and /api/groups need the key (403); /api/ping caps; meta never 500 (broken folder).
- group semantics: reject / undo / comment, individually rejected images survive undo, nested sample and montage
  rejects fall back to each other, a reason change, claims by others reported, portal identity stamps who,
  bad bodies 400, the page's echo of a grp decision accepted (grp only on i:reject).
- concurrency: two reviewers' group rejects at once plus image saves in flight: nothing lost.
- edits/_groups.json is no image record for any reader: review_server /api/edits, apply_review_edits.load_records,
  registry importers.package_progress and validate, portal titles / reviews counters, compare_package.strip_decoys,
  prediction_package.reviewed_polygons; counts with and without it are equal.
- results: a group-rejected sample is dropped by results.compute (all_rejected), its group note and reject listed.
"""
import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock  # noqa: F401  (unittest.mock.patch below)
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import meta_fixture as MF  # noqa: E402
import make_fixture  # noqa: E402
from annotation import multi_review_server as mrs  # noqa: E402
from annotation import prediction_package as pp  # noqa: E402
from annotation import review_groups as rg  # noqa: E402
from annotation import review_meta as rm  # noqa: E402
from annotation import review_server as rs  # noqa: E402
sys.path.append(str(REPO / "tests"))
import cryoev_testkit as TK  # noqa: E402

EV, HEK = "EV-013-2", "HEK"
EV_FILES = [f"{i:04d}.png" for i in range(1, 7)]
HEK_FILES = ["0007.png", "0008.png", "0009.png"]
G10_1 = ["0001.png", "0002.png", "0003.png", "0004.png"]


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="r12meta_"))
        with TK.no_actor_env():
            self.fx = MF.build(self.tmp)
        self.pkg = self.fx["pkg"]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------- names
class TileNameTests(unittest.TestCase):
    def test_processing_repo_names(self):
        self.assertEqual(rm.parse_tile_name("g10_EV-013-2__grid10_1_r0c0_t02__denoised_nobar_14p72Apx.png"),
                         {"montage": "grid10_1", "row": 0, "col": 0, "piece": 2, "grid": 10})
        # the label's own underscores (training_name collapses runs of other characters into one "_")
        self.assertEqual(rm.parse_tile_name("g1_holey_carbon_identity_unknown__grid1_10_r2c11_t07__denoised_nobar_14p72Apx.png"),
                         {"montage": "grid1_10", "row": 2, "col": 11, "piece": 7, "grid": 1})
        self.assertEqual(rm.parse_tile_name("grid3_2_r1c0_t04.jpg"), {"montage": "grid3_2", "row": 1, "col": 0, "piece": 4})
        self.assertEqual(rm.parse_tile_name("g2_A__grid2_1_r1c2_t05__denoised_nobar_14p72Apx_png.rf.0a1b2c.jpg"),
                         {"montage": "grid2_1", "row": 1, "col": 2, "piece": 5, "grid": 2})
        self.assertEqual(rm.parse_tile_name("g2_A__sq_r3c1__labeling_G3p5nm_7p36Apx.png")["montage"], "sq")

    def test_odd_names(self):
        for n in ("0001.png", "", None, "HEK__raw__x0000_y0000.png", "IMG_4522~2.jpg", "a__b__c.png", 7):
            self.assertEqual(rm.parse_tile_name(n), {}, n)

    def test_agrees_with_manifest(self):
        """Control: for every row the converter's MANIFEST wrote, the name gives the same montage."""
        for image, grid, sample, montage, row, col, piece in MF.TILES:
            got = rm.parse_tile_name(MF.tile_name(grid, sample, montage, row, col, piece))
            self.assertEqual((got["montage"], got["row"], got["col"], got["grid"]), (montage, row, col, grid))


# ---------------------------------------------------------------- package_meta
class MetaTests(FixtureCase):
    def test_full_conversion(self):
        m = rm.package_meta(self.pkg)
        P, im = m["package"], m["images"]
        self.assertEqual(m["schema"], "review_meta/1")
        self.assertEqual((P["kind"], P["prediction_set"], P["session"], P["conversion"], P["source_kind"]),
                         ("prediction", MF.PS, "20261106-9", MF.CONV, "session"))
        self.assertEqual(P["recipe"], {"id": "20261027", "variant": "denoised", "gpu": True, "params": {
            "clip_lo_pct": 1.0, "clip_hi_pct": 99.0, "bandpass_nm": 2.0, "denoise": True}})
        self.assertEqual(P["session_acquisition"]["defocus_um"]["median"], -4.48)
        self.assertEqual(sorted(im), [f"{i:04d}.png" for i in range(1, 10)])
        a = im["0002.png"]
        self.assertEqual({k: a[k] for k in ("split", "sample", "montage", "grid", "tile", "defocus_um", "mag", "pixel_A",
                                            "nm_per_px", "dose", "exposure_s", "acquired")},
                         {"split": EV, "sample": EV, "montage": "grid10_1", "grid": 10,
                          "tile": {"row": 0, "col": 1, "piece": 5}, "defocus_um": -4.4799, "mag": 11000, "pixel_A": 7.362,
                          "nm_per_px": 1.4724, "dose": 3.6239, "exposure_s": 2.996, "acquired": "2026-11-06T13:11:24"})
        self.assertEqual(im["0005.png"]["defocus_um"], -2.2061)
        self.assertEqual(im["0007.png"]["montage"], "grid6_1")
        # 0009: not in MANIFEST, layout or acquisition: montage and row / col from its name, nothing else invented
        n = im["0009.png"]
        self.assertEqual((n["montage"], n["tile"]), ("grid6_9", {"row": 0, "col": 0, "piece": 0}))
        for k in ("defocus_um", "mag", "pixel_A", "dose", "acquired"):
            self.assertNotIn(k, n)
        # names only: no local folder reaches the page
        dump = json.dumps(m)
        for p in (str(self.tmp), str(self.tmp).replace("\\", "\\\\"), "C:\\\\somewhere", "input"):
            self.assertNotIn(p, dump)

    def test_layout_row_col_win_over_the_name(self):
        """layout.json is what the stitcher placed: when it says another row / col than the name, it wins."""
        lay = json.loads((self.fx["conv"] / "montages" / "layout.json").read_text(encoding="utf-8"))
        lay["montages"][0]["tiles"][1]["row"] = 7
        (self.fx["conv"] / "montages" / "layout.json").write_text(json.dumps(lay), encoding="utf-8")
        self.assertEqual(rm.package_meta(self.pkg)["images"]["0002.png"]["tile"], {"row": 7, "col": 1, "piece": 5})

    def test_missing_files_leave_fields_out(self):
        (self.fx["conv"] / "acquisition.json").unlink()
        m = rm.package_meta(self.pkg)
        self.assertNotIn("defocus_um", m["images"]["0001.png"])
        self.assertNotIn("session_acquisition", m["package"])
        self.assertEqual(m["images"]["0001.png"]["montage"], "grid10_1")
        (self.fx["conv"] / "convert.json").unlink()   # no longer a conversion: the folder's name, no recipe
        m = rm.package_meta(self.pkg)
        self.assertNotIn("recipe", m["package"])
        self.assertEqual(m["package"]["source_folder"], "tiles")
        (self.fx["conv"] / "tiles" / "MANIFEST.csv").unlink()
        self.assertEqual(rm.package_meta(self.pkg)["images"]["0003.png"]["tile"], {"row": 1, "col": 0, "piece": 1})
        shutil.rmtree(self.fx["ps"])   # no prediction set at all: sample and split from the package itself
        m = rm.package_meta(self.pkg)
        self.assertEqual(m["images"]["0001.png"]["sample"], EV)
        self.assertNotIn("nm_per_px", m["images"]["0001.png"])
        self.assertNotIn("error", m["package"])

    def test_broken_files_never_raise(self):
        (self.fx["conv"] / "acquisition.json").write_text("{not json", encoding="utf-8")
        (self.fx["conv"] / "montages" / "layout.json").write_text("[1, 2]", encoding="utf-8")
        (self.fx["ps"] / "request.json").write_text(json.dumps({"source": {"kind": "session", "dir": 5}}), encoding="utf-8")
        m = rm.package_meta(self.pkg)
        self.assertEqual(m["images"]["0001.png"]["montage"], "grid10_1")    # the name still gives it
        self.assertNotIn("defocus_um", m["images"]["0001.png"])
        (self.pkg / rm.META).write_text("{", encoding="utf-8")
        m = rm.package_meta(self.pkg)
        self.assertEqual(m["images"]["0001.png"]["split"], EV)
        (self.pkg / "data.js").write_text("window.REVIEW_DATA = nonsense", encoding="utf-8")
        self.assertEqual(rm.package_meta(self.pkg)["images"], {})

    def test_cache_follows_the_files(self):
        self.assertEqual(rm.package_meta(self.pkg)["images"]["0001.png"]["defocus_um"], -4.4799)
        acq = json.loads((self.fx["conv"] / "acquisition.json").read_text(encoding="utf-8"))
        acq["montages"][0]["defocus_um"] = -1.25
        time.sleep(0.02)
        (self.fx["conv"] / "acquisition.json").write_text(json.dumps(acq), encoding="utf-8")
        self.assertEqual(rm.package_meta(self.pkg)["images"]["0001.png"]["defocus_um"], -1.25)

    def test_upload_set_with_note_provenance_and_manifest(self):
        req = json.loads((self.fx["ps"] / "request.json").read_text(encoding="utf-8"))
        req["source"] = {"kind": "upload"}
        req.update(name="2x test", note="2x-downsampled tiles", source_manifest="source_manifest.csv",
                   provenance={"steps": ["2x downsample", "1440 px tiles"], "derived_from": [{"kind": "prediction_set",
                               "id": "ps_20261112_fb4539"}], "script": "C:\\Users\\x\\work\\prep_variants.py",
                               "tiling": {"tile_px": 1440, "overlap_frac": 0.5}})
        (self.fx["ps"] / "request.json").write_text(json.dumps(req), encoding="utf-8")
        name = MF.tile_name(10, EV, "grid10_1", 0, 0, 2)
        (self.fx["ps"] / "source_manifest.csv").write_text(
            "output_name,nm_per_px,sample_label,parent_image,parent_file,x0,y0,tile_px\n"
            f"{name},0.7362,{EV},HEK_view_1,HEK_view_1.png,720,0,1440\n", encoding="utf-8")
        m = rm.package_meta(self.pkg)
        P = m["package"]
        self.assertEqual((P["source_kind"], P["name"], P["note"]), ("upload", "2x test", "2x-downsampled tiles"))
        self.assertEqual(P["provenance"]["script"], "prep_variants.py")
        self.assertEqual(P["provenance"]["steps"], ["2x downsample", "1440 px tiles"])
        self.assertNotIn("conversion", P)
        a = m["images"]["0001.png"]
        self.assertEqual(a["parent_image"], "HEK_view_1")
        self.assertEqual(a["montage"], "grid10_1")   # its name still says the montage; the parent is listed too

    def test_registry_session_summary(self):
        """A set made from the processing repo's own tile folder (no conversion): the registry's session record."""
        (self.fx["conv"] / "convert.json").unlink()
        (self.fx["conv"] / "acquisition.json").unlink()
        reg = TK.write_registry(self.tmp / "registry", {"session": [{"schema": "session/1", "id": "20261106-9",
                                "defocus_um": {"min": -8.4, "median": -4.7, "max": -2.3}, "instrument": "Glacios"}]},
                                samples=False)
        with TK.env_patch(CRYOEV_REGISTRY=str(reg)):
            P = rm.package_meta(self.pkg)["package"]
        self.assertEqual(P["session_acquisition"], {"instrument": "Glacios",
                                                    "defocus_um": {"min": -8.4, "median": -4.7, "max": -2.3}})
        with TK.env_patch(CRYOEV_REGISTRY=str(self.tmp / "no_registry_here")):
            self.assertNotIn("session_acquisition", rm.package_meta(self.pkg)["package"])

    def test_lab_package_split_only(self):
        lab = self.tmp / "lab"
        make_fixture.build(lab)
        m = rm.package_meta(lab)
        self.assertEqual(m["package"]["kind"], "label")
        self.assertEqual(m["images"]["fixA_new.png"], {"split": "train"})


# ---------------------------------------------------------------- the routes, on the multi server
class ServerCase(FixtureCase):
    def setUp(self):
        super().setUp()
        self.key = (self.pkg / ".review_key").read_text(encoding="utf-8").strip()
        self.srv = mrs.make_server(self.fx["review"], "127.0.0.1", 0, people=[])
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        super().tearDown()

    def req(self, method, path, body=None, key="__own__", headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        h = dict(headers or {})
        if key:
            h["X-Review-Key"] = self.key if key == "__own__" else key
        data = body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else None
        c.request(method, f"/{MF.PKG}{path}", body=data, headers=h)
        r = c.getresponse()
        raw = r.read()
        c.close()
        try:
            return r.status, json.loads(raw)
        except ValueError:
            return r.status, raw

    def group(self, who="tester", **body):
        return self.req("POST", f"/api/group?who={quote(who)}", body)

    def saved(self, file):
        p = self.pkg / "edits" / f"{file}.json"
        return json.loads(p.read_text(encoding="utf-8"))["edit"] if p.exists() else None

    def rej(self, file):
        e = self.saved(file)
        d = rs._migrate(e)["dec"].get("i:reject") if e else None
        return (d.get("v"), d.get("grp")) if d and d.get("v") else (None, None)

    def put_reject(self, file, reason, who="ann"):
        rec = {"dec": {"i:reject": {"v": f"reject:{reason}" if reason else None, "t": int(time.time() * 1000),
                                     "who": who, "pending": True}}, "t": int(time.time() * 1000)}
        st, ans = self.req("PUT", "/api/edits?f=" + quote(file), rec)
        self.assertEqual(st, 200, ans)


class RouteTests(ServerCase):
    def test_ping_meta_groups_need_the_key(self):
        self.assertEqual(self.req("GET", "/api/ping"), (200, {"ok": True, "caps": ["i", "t", "m", "g"]}))
        for path in ("/api/meta", "/api/groups"):
            self.assertEqual(self.req("GET", path, key="wrong")[0], 403)
            self.assertEqual(self.req("GET", path, key=None)[0], 403)
        self.assertEqual(self.req("POST", "/api/group?who=x", {"action": "comment"}, key="wrong")[0], 403)
        st, m = self.req("GET", "/api/meta")
        self.assertEqual(st, 200)
        self.assertEqual(m["images"]["0001.png"]["defocus_um"], -4.4799)
        self.assertEqual(self.req("GET", "/api/groups"), (200, {"schema": rg.SCHEMA, "groups": {}, "log": []}))

    def test_meta_never_500(self):
        with unittest.mock.patch.object(rm, "package_meta", side_effect=RuntimeError("boom")):
            st, m = self.req("GET", "/api/meta")
        self.assertEqual(st, 200)
        self.assertEqual(m["images"], {})
        self.assertIn("metadata unavailable", m["package"]["error"])
        (self.pkg / "edits").mkdir(exist_ok=True)
        (self.pkg / "edits" / rg.GROUPS_FILE).write_text("{oops", encoding="utf-8")
        st, g = self.req("GET", "/api/groups")
        self.assertEqual((st, g["groups"]), (200, {}))
        self.assertIn("unreadable", g["error"])
        st, ans = self.group(action="comment", kind="sample", sample=EV, note="x")   # never written over
        self.assertEqual(st, 409)
        self.assertEqual((self.pkg / "edits" / rg.GROUPS_FILE).read_text(encoding="utf-8"), "{oops")

    def test_bad_bodies(self):
        bad = [({"action": "reject", "kind": "sample", "sample": EV}, "reason required"),
               ({"action": "reject", "kind": "sample", "sample": EV, "reason": "meh"}, "reason required"),
               ({"action": "comment", "kind": "sample", "sample": EV, "reason": "empty"}, "reason only with reject"),
               ({"action": "comment", "kind": "montage", "sample": EV}, "montage required"),
               ({"action": "comment", "kind": "sample", "sample": EV, "montage": "grid10_1"}, "montage required"),
               ({"action": "comment", "kind": "sample", "sample": EV, "extra": 1}, "bad body"),
               ({"action": "drop", "kind": "sample", "sample": EV}, "bad action"),
               ({"action": "comment", "kind": "grid", "sample": EV}, "bad action"),
               ({"action": "comment", "kind": "sample", "sample": EV, "note": "x" * 1001}, "bad note"),
               ({"action": "undo", "kind": "sample", "sample": EV, "note": "x"}, "no note with undo"),
               ({"action": "comment", "kind": "sample"}, "sample required"),
               ([1, 2], "bad body")]
        for body, why in bad:
            st, ans = self.group(**body) if isinstance(body, dict) else self.req("POST", "/api/group?who=t", body)
            self.assertEqual(st, 400, body)
            self.assertIn(why, ans["error"], body)
        self.assertEqual(self.req("POST", "/api/group", {"action": "comment", "kind": "sample", "sample": EV})[0], 400)
        self.assertEqual(self.req("POST", "/api/group?who=%3Cb%3E", {"action": "comment", "kind": "sample",
                                                                      "sample": EV})[0], 400)
        self.assertEqual(self.req("POST", "/api/group?who=t", b"{NaN}")[0], 400)
        st, ans = self.group(action="comment", kind="sample", sample="nope", note="x")
        self.assertEqual((st, ans["error"]), (404, "no images in this group"))
        self.assertFalse((self.pkg / "edits" / rg.GROUPS_FILE).exists())   # nothing written by a refusal

    def test_no_group_actions_on_a_lab_package(self):
        """A lab / compare package (no prediction set, split = train / valid): the route refuses, nothing written."""
        from http.server import ThreadingHTTPServer
        lab = self.tmp / "lab"
        make_fixture.build(lab)
        files = {im["file"] for im in pp.read_datajs(lab)["images"]}
        srv = ThreadingHTTPServer(("127.0.0.1", 0), rs.make_handler(lab, "k1", files, "lab", []))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
            c.request("POST", "/api/group?who=t", body=json.dumps({"action": "reject", "kind": "sample", "sample": "train",
                                                                   "reason": "empty"}), headers={"X-Review-Key": "k1"})
            r = c.getresponse()
            self.assertEqual((r.status, json.loads(r.read())["error"]), (404, "group actions are for prediction packages"))
            c.request("GET", "/api/meta", headers={"X-Review-Key": "k1"})
            r = c.getresponse()
            self.assertEqual(json.loads(r.read())["images"]["fixB_old.png"], {"split": "valid"})
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(sorted(p.name for p in (lab / "edits").iterdir()), [])

    def test_echo_of_a_grp_decision(self):
        """The page PUTs whole records back, group-rejected ones included: grp is valid on i:reject only."""
        self.assertEqual(self.group(action="reject", kind="sample", sample=HEK, reason="empty")[0], 200)
        e = self.saved("0007.png")
        self.assertEqual(self.req("PUT", "/api/edits?f=0007.png", e)[0], 200)
        e2 = json.loads(json.dumps(e))
        e2["dec"]["i:note"] = {"v": "n", "t": 1, "who": "x", "grp": e["dec"]["i:reject"]["grp"]}
        self.assertEqual(self.req("PUT", "/api/edits?f=0007.png", e2)[0], 400)
        e2 = json.loads(json.dumps(e))
        e2["dec"]["i:reject"]["grp"] = "x" * (rs.GRP_MAX + 1)
        self.assertEqual(self.req("PUT", "/api/edits?f=0007.png", e2)[0], 400)
        self.assertTrue(rs.valid_edit(e))
        self.assertEqual(rs.GRP_MAX, rg.GRP_MAX)


class GroupTests(ServerCase):
    def test_reject_then_undo_keeps_individual_rejects(self):
        self.put_reject("0002.png", "empty")                     # rejected by hand before the group action
        st, ans = self.group(action="reject", kind="sample", sample=EV, reason="thick ice", note="whole grid icy")
        self.assertEqual(st, 200, ans)
        self.assertEqual((ans["n_images"], ans["n_changed"], ans["kept_individual"]), (6, 5, 1))
        gid = rg.group_id("sample", EV)
        for f in EV_FILES:
            self.assertEqual(self.rej(f), ("reject:empty", None) if f == "0002.png" else ("reject:thick ice", gid), f)
            if f != "0002.png":
                d = self.saved(f)["dec"]["i:reject"]
                self.assertEqual(d["who"], "tester")
                self.assertIsInstance(d["server_t"], int)
                self.assertNotIn("pending", d)
        for f in HEK_FILES:
            self.assertEqual(self.rej(f), (None, None))
        flags = rs.image_flags(self.saved("0001.png"))
        self.assertEqual(flags, {"rejected": True, "reject_reason": "thick ice", "reject_scope": "sample",
                                 "reject_group": gid})
        self.assertEqual(rs.image_flags(self.saved("0002.png")), {"rejected": True, "reject_reason": "empty"})
        g = self.req("GET", "/api/groups")[1]["groups"][gid]
        self.assertEqual({k: g["reject"][k] for k in ("reason", "note", "who", "n_images", "n_rejected")},
                         {"reason": "thick ice", "note": "whole grid icy", "who": "tester", "n_images": 6, "n_rejected": 5})
        self.assertEqual(sorted(ans["edits"]), sorted(set(EV_FILES) - {"0002.png"}))
        # undo: only the five this group rejected; 0002 keeps its own rejection
        st, ans = self.group(action="undo", kind="sample", sample=EV, who="lead")
        self.assertEqual((st, ans["n_changed"]), (200, 5))
        for f in EV_FILES:
            self.assertEqual(self.rej(f), ("reject:empty", None) if f == "0002.png" else (None, None), f)
        self.assertEqual(self.saved("0001.png")["dec"]["i:reject"]["who"], "lead")
        self.assertNotIn(gid, self.req("GET", "/api/groups")[1]["groups"])
        log = self.req("GET", "/api/groups")[1]["log"]
        self.assertEqual([(x["action"], x["who"]) for x in log], [("reject", "tester"), ("undo", "lead")])
        # results drop rejected images: is_rejected agrees
        self.assertTrue(pp.is_rejected(self.saved("0002.png")))
        self.assertFalse(pp.is_rejected(self.saved("0001.png")))

    def test_a_later_individual_change_is_not_undone(self):
        self.group(action="reject", kind="sample", sample=EV, reason="thick ice")
        self.put_reject("0003.png", "contaminated", who="ann")     # re-reasoned by hand: now its own
        self.put_reject("0004.png", None, who="ann")               # un-rejected by hand
        st, ans = self.group(action="undo", kind="sample", sample=EV)
        self.assertEqual(ans["n_changed"], 4)
        self.assertEqual(self.rej("0003.png"), ("reject:contaminated", None))
        self.assertEqual(self.rej("0004.png"), (None, None))

    def test_montage_inside_a_rejected_sample(self):
        gs, gm = rg.group_id("sample", EV), rg.group_id("montage", EV, "grid10_1")
        # montage first, then the sample: the montage's images stay the montage's
        self.assertEqual(self.group(action="reject", kind="montage", sample=EV, montage="grid10_1", reason="empty")[1]["n_changed"], 4)
        ans = self.group(action="reject", kind="sample", sample=EV, reason="thick ice")[1]
        self.assertEqual((ans["n_changed"], ans["kept_other_group"]), (2, 4))
        # undo the montage: its images fall back to the sample reject, still on
        ans = self.group(action="undo", kind="montage", sample=EV, montage="grid10_1")[1]
        self.assertEqual((ans["n_changed"], ans["kept_other_group"]), (4, 4))
        for f in EV_FILES:
            self.assertEqual(self.rej(f), ("reject:thick ice", gs), f)
        # sample reject on, montage again: nothing new to reject (all rejected by the sample)
        self.assertEqual(self.group(action="reject", kind="montage", sample=EV, montage="grid10_1", reason="empty")[1]["n_changed"], 0)
        # undo the sample: the montage reject is on, so its four go to it; the other two are cleared
        ans = self.group(action="undo", kind="sample", sample=EV)[1]
        self.assertEqual(ans["n_changed"], 6)
        for f in EV_FILES:
            self.assertEqual(self.rej(f), ("reject:empty", gm) if f in G10_1 else (None, None), f)
        self.assertEqual(self.group(action="undo", kind="montage", sample=EV, montage="grid10_1")[1]["n_changed"], 4)
        self.assertTrue(all(self.rej(f) == (None, None) for f in EV_FILES))
        self.assertEqual(self.req("GET", "/api/groups")[1]["groups"], {})

    def test_unreadable_record_changes_nothing(self):
        self.put_reject("0001.png", "empty")
        (self.pkg / "edits" / "0008.png.json").write_text("{broken", encoding="utf-8")
        before = {f.name: f.read_bytes() for f in (self.pkg / "edits").iterdir()}
        st, ans = self.group(action="reject", kind="sample", sample=HEK, reason="empty")
        self.assertEqual(st, 409, ans)
        self.assertIn("nothing was changed", ans["error"])
        self.assertEqual({f.name: f.read_bytes() for f in (self.pkg / "edits").iterdir()}, before)

    def test_reason_change_and_repeat(self):
        self.group(action="reject", kind="sample", sample=HEK, reason="empty")
        ans = self.group(action="reject", kind="sample", sample=HEK, reason="empty")[1]   # the same again: no-op
        self.assertEqual((ans["n_changed"], ans["already"]), (0, 3))
        ans = self.group(action="reject", kind="sample", sample=HEK, reason="aggregates")[1]
        self.assertEqual(ans["n_changed"], 3)
        self.assertTrue(all(self.rej(f)[0] == "reject:aggregates" for f in HEK_FILES))

    def test_comment(self):
        gid = rg.group_id("montage", HEK, "grid6_1")
        st, ans = self.group(action="comment", kind="montage", sample=HEK, montage="grid6_1", note="  drifted  ")
        self.assertEqual((st, ans["n_changed"]), (200, 0))
        n = ans["groups"][gid]["note"]
        self.assertEqual((n["text"], n["who"]), ("drifted", "tester"))
        self.assertFalse(any((self.pkg / "edits" / f"{f}.json").exists() for f in HEK_FILES))  # no image touched
        ans = self.group(action="comment", kind="montage", sample=HEK, montage="grid6_1", note="")[1]
        self.assertNotIn(gid, ans["groups"])
        self.assertEqual([x["action"] for x in ans["log"]], ["comment", "comment"])

    def test_claimed_images_are_changed_and_reported(self):
        self.assertEqual(self.req("POST", "/api/claim?c=0008.png%7Ceditor&who=bob", b"")[0], 200)
        self.assertEqual(self.req("POST", "/api/claim?c=0007.png%7Cp0&who=tester", b"")[0], 200)
        ans = self.group(action="reject", kind="sample", sample=HEK, reason="empty")[1]
        self.assertEqual(ans["claimed_by_others"], [{"file": "0008.png", "who": "bob"}])
        self.assertEqual(self.rej("0008.png")[0], "reject:empty")

    def test_portal_identity_is_the_reviewer(self):
        h = {mrs.IDENTITY_HEADER: "alyssia"}
        st, ans = self.req("POST", "/api/group?who=someone_else", {"action": "reject", "kind": "sample", "sample": HEK,
                                                                     "reason": "empty"}, headers=h)
        self.assertEqual(st, 200, ans)
        self.assertEqual(self.saved("0007.png")["dec"]["i:reject"]["who"], "alyssia")
        self.assertEqual(ans["groups"][rg.group_id("sample", HEK)]["reject"]["who"], "alyssia")

    def test_concurrent_group_actions_and_saves(self):
        """Two reviewers reject the two samples at once while notes are saved on every image: under the package
        write lock nothing is lost (both group entries, every reject, every note)."""
        errors = []

        def run(fn):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def note(f):
            rec = {"dec": {"i:note": {"v": f"note {f}", "t": int(time.time() * 1000), "who": "carol", "pending": True}},
                   "t": int(time.time() * 1000)}
            st, ans = self.req("PUT", "/api/edits?f=" + quote(f), rec)
            assert st == 200, ans
        jobs = [lambda: self.group(who="ann", action="reject", kind="sample", sample=EV, reason="empty"),
                lambda: self.group(who="bob", action="reject", kind="sample", sample=HEK, reason="aggregates")]
        jobs += [lambda f=f: note(f) for f in EV_FILES + HEK_FILES]
        threads = [threading.Thread(target=run, args=(j,)) for j in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(errors, [])
        groups = self.req("GET", "/api/groups")[1]["groups"]
        self.assertEqual(sorted(groups), sorted([rg.group_id("sample", EV), rg.group_id("sample", HEK)]))
        for f in EV_FILES + HEK_FILES:
            dec = self.saved(f)["dec"]
            self.assertEqual(dec["i:reject"]["v"], "reject:empty" if f in EV_FILES else "reject:aggregates", f)
            self.assertEqual(dec["i:note"]["v"], f"note {f}", f)

    def test_lock_is_the_package_write_lock(self):
        """Control for the concurrency test: a group action waits for the lock a PUT holds."""
        lock = rs._write_lock(self.pkg / "edits")
        done = threading.Event()
        lock.acquire()
        try:
            t = threading.Thread(target=lambda: (self.group(action="comment", kind="sample", sample=EV, note="x"),
                                                 done.set()))
            t.start()
            self.assertFalse(done.wait(0.6))     # blocked on the write lock
        finally:
            lock.release()
        self.assertTrue(done.wait(10))
        t.join(10)


# ---------------------------------------------------------------- edits/_groups.json is no image record
class GroupsFileIgnoredTests(ServerCase):
    def counts(self):
        from annotation import apply_review_edits as are
        from portal import titles
        from portal.modules import reviews
        from registry import importers
        titles._cache.clear()
        reviews._cache.clear()
        reviews._cards.clear()
        edits = self.pkg / "edits"
        prog = titles.package_progress(self.pkg)
        return {"api": sorted(self.req("GET", "/api/edits")[1]),
                "load_records": [f.name for f, _ in are.load_records(edits)],
                "record_files": [f.name for f in rg.record_files(edits)],
                "registry": importers.package_progress(edits)[:3],
                "titles": {k: prog[k] for k in ("cards", "decided", "images", "reviewed")},
                "reviews": (reviews.edit_stats(self.pkg)[0], reviews.card_progress(self.pkg)),
                "reviewed": pp.reviewed_polygons(self.pkg, write=False)["images"]}

    def test_counts_unchanged(self):
        self.put_reject("0001.png", "empty")
        rec = {"dec": {"p0": {"v": "no", "t": 1, "who": "ann", "pending": True}}, "reviewed": True, "reviewedT": 5, "t": 5}
        self.assertEqual(self.req("PUT", "/api/edits?f=0008.png", rec)[0], 200)
        before = self.counts()
        self.assertFalse((self.pkg / "edits" / rg.GROUPS_FILE).exists())
        self.assertEqual(self.group(action="comment", kind="sample", sample=HEK, note="all fine")[0], 200)
        self.assertTrue((self.pkg / "edits" / rg.GROUPS_FILE).exists())
        after = self.counts()
        self.assertEqual(after, before)
        self.assertEqual(before["registry"], (2, 1, 2))
        self.assertEqual(before["titles"]["reviewed"], 1)
        # control: a plain glob does see the file, so the counts above would move without the skip
        self.assertEqual(len(list((self.pkg / "edits").glob("*.json"))), len(before["record_files"]) + 1)

    def test_strip_decoys_and_validate(self):
        from annotation import compare_package as cp
        from registry import validate
        self.group(action="comment", kind="sample", sample=HEK, note="x")
        self.put_reject("0001.png", "empty")
        work = self.tmp / "work"
        work.mkdir()
        (work / "decoys.json").write_text(json.dumps({"decoys": []}), encoding="utf-8")
        st = cp.strip_decoys(self.pkg / "edits", work, self.tmp / "out")
        self.assertEqual((st["files"], st["changed"]), (1, []))
        self.assertTrue((self.tmp / "out" / rg.GROUPS_FILE).is_file())       # carried along, byte for byte
        self.assertEqual(cp.edits_sha256(self.pkg / "edits"), cp.edits_sha256(self.tmp / "out"))
        self.assertEqual(validate.GROUPS_FILE, rg.GROUPS_FILE)            # registry/ keeps its own copy of the name
        self.assertFalse(validate.is_record_file(self.pkg / "edits" / rg.GROUPS_FILE))


# ---------------------------------------------------------------- results
class ResultsTests(ServerCase):
    def test_group_rejected_sample_dropped(self):
        try:
            from results import compute  # the function (results/__init__.py)
        except Exception as e:  # noqa: BLE001  the counting chain needs its heavy deps
            self.skipTest(f"results chain unavailable: {e}")
        self.group(action="reject", kind="sample", sample=HEK, reason="empty", note="no grid")
        self.group(action="comment", kind="sample", sample=EV, note="fine")
        reviewed = pp.reviewed_polygons(self.pkg, write=False)
        self.assertEqual(reviewed["images"]["0007"]["reject_scope"], "sample")
        res = compute(self.fx["ps"], reviewed)
        self.assertTrue(res["samples"][HEK]["all_rejected"])
        self.assertEqual(res["samples"][HEK]["n_rejected"], 3)
        self.assertEqual(res["samples"][HEK]["group"]["reject"]["reason"], "empty")
        self.assertEqual(res["samples"][HEK]["group"]["reject"]["note"], "no grid")
        self.assertEqual(res["samples"][EV]["group"]["note"]["text"], "fine")
        self.assertEqual(res["samples"][EV]["n_rejected"], 0)
        self.assertGreater(res["samples"][EV]["n_images"], 0)


if __name__ == "__main__":
    unittest.main()
