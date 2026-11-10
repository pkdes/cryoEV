"""
Round 7 R1: image reject / note and the object Low conf tag, as decisions on keys of their own (i:reject, i:note,
t:<objkey>; docs/REVIEW_TOOL.md "Reject, notes, low conf"). Server side and downstream; the page is tested in
test_review_reject_lowconf_browser.py.

- valid_edit: exactly the new keys, reasons from the list, notes within their caps, no outline, no bare strings;
  /api/ping caps; the HEAD (00a1921) valid_edit refuses the new keys (the reason for the caps gate).
- merge: two devices with different i: / t: decisions on one image merge per key, the newest received wins; a class
  change on another device leaves the t: tag intact.
- effective_labels passes low_conf / note through; is_edited ignores i: keys but counts t:; is_rejected; reviews
  _is_checked; spot check skips rejected images.
- reviewed_polygons: rejected / reject_reason / note per image (n_candidates_open 0 when rejected), low_conf / note
  per object, each key only when set.
- apply_review_edits: i: / t: keys never counted as ops (a note "no" is not a removal); THE ONE COCO RULE (EV + tag ->
  Low conf EV with dec_key t:<key>; MV + tag -> attributes.low_conf; note -> attributes.note); version.json
  rejected_images / low_conf; images never dropped.

  python -m unittest discover -s annotation/tests -p "test_review_reject_lowconf.py" -v
"""
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import test_apply_review_edits as tar  # noqa: E402  (module import: its TestCases are not collected twice)
from test_review_server import LiveServer  # noqa: E402
import test_prediction_package as tpp  # noqa: E402

from annotation import prediction_package as pp  # noqa: E402
from annotation import review_server as rs  # noqa: E402
import apply_review_edits as ar  # noqa: E402  (path set up by test_apply_review_edits)

HEAD = "00a1921"  # main before round 7 Phase 2: the old server
XSS = '<img src=x onerror="window.__xss=1"><script>window.__xss=2</script>&amp;"\''


def dec(v, t=1000, who="ann", **kw):
    return {"v": v, "t": t, "who": who, **kw}


class ValidEditTests(unittest.TestCase):
    def ok(self, d):
        return rs.valid_edit({"t": 1, "dec": d})

    def test_new_keys_accepted(self):
        for d in ({"i:reject": dec("reject:thick ice")}, {"i:reject": dec("reject:out of focus / drift")},
                  {"i:reject": dec(None)}, {"i:note": dec("x" * rs.IMAGE_NOTE_MAX)}, {"i:note": dec(None)},
                  {"i:note": dec(XSS)}, {"t:g12": dec("lowconf")}, {"t:p0": dec("plain", note="y" * rs.TAG_NOTE_MAX)},
                  {"t:nl0abc12": dec(None)}, {"t:g1": dec("lowconf", note=XSS, pending=True)},
                  {"t:g1": dec("lowconf", server_t=5)}):
            self.assertTrue(self.ok(d), d)
        for r in rs.REJECT_REASONS:
            self.assertTrue(self.ok({"i:reject": dec("reject:" + r)}), r)

    def test_bad_shapes_refused(self):
        bad = [
            {"i:reject": dec("reject:blurry")}, {"i:reject": dec("thick ice")}, {"i:reject": "reject:empty"},
            {"i:reject": dec("reject:empty", note="n")}, {"i:note": dec("x" * (rs.IMAGE_NOTE_MAX + 1))},
            {"i:note": dec(5)}, {"i:note": dec("a", p=[[0, 0, 1, 0, 1, 1]])}, {"i:other": dec("x")},
            {"t:g1": dec("maybe")}, {"t:g1": dec("lowconf", note="z" * (rs.TAG_NOTE_MAX + 1))},
            {"t:g1": dec("lowconf", note=7)}, {"t:g1": "lowconf"}, {"t:x1": dec("lowconf")}, {"t:g1\n": dec("lowconf")},
            {"t:gA": dec("lowconf")}, {"t:nABC": dec("lowconf")}, {"t:g1": dec("lowconf", who="<b>")},
            {"t:g1": dec("lowconf", pending="yes")}, {"t:g1": dec("lowconf", t=float("inf"))},
            {"t:": dec("lowconf")}, {"t:g١": dec("lowconf")},  # a non-ASCII digit
        ]
        for d in bad:
            self.assertFalse(self.ok(d), d)

    def test_crafted_state_decisions_refused(self):
        """S2: only the decision fields, UTF-8 encodable strings (a lone surrogate would crash every later writer),
        a bounded key."""
        lone = json.loads('"x\\ud800y"')
        self.assertRaises(UnicodeEncodeError, lone.encode, "utf-8")  # control: it really is unencodable
        for d in ({"t:g1": dec("lowconf", seed="s")}, {"i:note": dec("n", extra=1)}, {"t:g1": dec("lowconf", p=None)},
                  {"i:note": dec(lone)}, {"t:g1": dec("plain", note=lone)}, {"t:n" + "a" * 62: dec("lowconf")}):
            self.assertFalse(self.ok(d), d)
        self.assertTrue(self.ok({"t:g" + "1" * 61: dec("lowconf")}))  # 64 characters: the cap

    def test_old_keys_unchanged(self):
        self.assertTrue(self.ok({"g1": dec("keep:EV"), "p0": "no", "nabc": dec("new:EV", p=[[0, 0, 1, 0, 1, 1]])}))
        self.assertFalse(self.ok({"x1": dec("keep:EV")}))

    def test_head_server_refuses_the_new_keys(self):
        """The control for the caps gate: the server code before round 7 answers 400 to every new key."""
        try:
            src = subprocess.run(["git", "show", f"{HEAD}:annotation/review_server.py"], cwd=REPO, capture_output=True,
                                 text=True, check=True, encoding="utf-8").stdout
        except (OSError, subprocess.CalledProcessError):
            self.skipTest(f"CONTROL NOT RUN: no git history (git show {HEAD} failed)")
        old = types.ModuleType("review_server_head")
        old.__file__ = str(REPO / "annotation" / "review_server.py")  # people.json is looked up beside it
        exec(compile(src, "review_server_head.py", "exec"), old.__dict__)
        self.assertTrue(old.valid_edit({"t": 1, "dec": {"g1": dec("keep:EV")}}))  # control: an old key passes
        for d in ({"i:reject": dec("reject:empty")}, {"i:note": dec("n")}, {"t:g1": dec("lowconf")}):
            self.assertFalse(old.valid_edit({"t": 1, "dec": d}), d)
            self.assertTrue(rs.valid_edit({"t": 1, "dec": d}), d)


class ServerMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = LiveServer()

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()

    def saved(self, file):
        return self.srv.request("GET", "/api/edits")[1][file]

    def test_ping_caps(self):
        self.assertEqual(self.srv.request("GET", "/api/ping")[:2], (200, {"ok": True, "caps": ["i", "t"]}))

    def test_two_devices_merge_per_key_newest_wins(self):
        f = "fixC_old.png"
        a1 = {"t": 1, "dec": {"i:reject": dec("reject:empty", 10, "ann", pending=True),
                              "t:g301": dec("lowconf", 10, "ann", note="A's note", pending=True)}}
        self.assertEqual(self.srv.put(f, a1)[0], 200)
        # device B, offline since before A's save: its own new decisions on other keys, and a later reject reason
        b1 = {"t": 1, "dec": {"i:note": dec("from B", 5, "bob", pending=True),
                              "t:g302": dec("plain", 5, "bob", note="B's note", pending=True),
                              "i:reject": dec("reject:thick ice", 5, "bob", pending=True)}}
        self.assertEqual(self.srv.put(f, b1)[0], 200)
        d = self.saved(f)["dec"]
        self.assertEqual({k: (d[k]["v"], d[k]["who"], d[k].get("note")) for k in d}, {
            "i:reject": ("reject:thick ice", "bob", None),  # received last: wins although its client t is older
            "t:g301": ("lowconf", "ann", "A's note"), "i:note": ("from B", "bob", None),
            "t:g302": ("plain", "bob", "B's note")})
        self.assertTrue(all(isinstance(x.get("server_t"), int) and "pending" not in x for x in d.values()))
        # A re-sends its stale copy (no pending: an echo) -> nothing changes
        echo = {"t": 2, "dec": {"i:reject": dec("reject:empty", 10, "ann", server_t=d["i:reject"]["server_t"] - 1)}}
        self.assertEqual(self.srv.put(f, echo)[0], 200)
        self.assertEqual(self.saved(f)["dec"]["i:reject"]["v"], "reject:thick ice")
        # A un-rejects (pending): newest wins again, the note and tags are untouched
        self.srv.put(f, {"t": 3, "dec": {"i:reject": dec(None, 11, "ann", pending=True)}})
        d2 = self.saved(f)["dec"]
        self.assertIsNone(d2["i:reject"]["v"])
        self.assertEqual({k: d2[k] for k in ("t:g301", "t:g302", "i:note")}, {k: d[k] for k in ("t:g301", "t:g302", "i:note")})

    def test_class_change_on_another_device_keeps_the_tag(self):
        f = "fixB_old.png"
        self.srv.put(f, {"t": 1, "dec": {"t:g201": dec("lowconf", 10, "ann", note="thin wall", pending=True)}})
        self.srv.put(f, {"t": 1, "dec": {"g201": dec("keep:MV", 20, "bob", pending=True)}})
        rec = self.saved(f)
        self.assertEqual((rec["dec"]["g201"]["v"], rec["dec"]["t:g201"]["v"], rec["dec"]["t:g201"]["note"]),
                         ("keep:MV", "lowconf", "thin wall"))
        im = next(i for i in self.srv.data["images"] if i["file"] == f)
        g = next(o for o in rs.effective_labels(im, rec) if o.get("id") == 201)
        self.assertEqual((g["c"], g.get("low_conf"), g.get("note")), ("MV", True, "thin wall"))

    def test_bad_new_key_is_a_400_and_nothing_is_stored(self):
        f = "fixD_new.png"
        before = self.srv.request("GET", "/api/edits")[1].get(f)
        for raw in (b'{"t": 1, "dec": {"i:note": {"v": "a\\ud800", "t": 1, "who": "x"}}}',
                    b'{"t": 1, "dec": {"t:g401": {"v": "lowconf", "t": 1, "who": "x", "note": "\\udfff"}}}',
                    b'{"t": 1, "dec": {"t:g401": {"v": "lowconf", "t": 1, "who": "x", "big": "' + b"z" * 5000 + b'"}}}'):
            self.assertEqual(self.srv.put(f, raw)[0], 400, raw[:80])
        for d in ({"i:reject": dec("reject:nope", pending=True)}, {"i:note": dec("x" * 1001, pending=True)},
                  {"t:g401": dec("lowconf", note="y" * 301, pending=True)}):
            self.assertEqual(self.srv.put(f, {"t": 1, "dec": d})[0], 400, d)
        self.assertEqual(self.srv.request("GET", "/api/edits")[1].get(f), before)


class HelperTests(unittest.TestCase):
    IM = {"gt": [{"id": 1, "c": "EV", "p": [[0, 0, .1, 0, .1, .1]]}, {"id": 2, "c": "Low conf EV", "p": [[.2, .2, .3, .2, .3, .3]]}],
          "pr": [{"s": .4, "p": [[.5, .5, .6, .5, .6, .6]]}]}

    def test_effective_labels_pass_through_and_untouched_without_tags(self):
        plain = {"dec": {"g1": dec("keep:MV")}, "t": 1}
        out = rs.effective_labels(self.IM, plain)
        self.assertFalse(any("low_conf" in o or "note" in o for o in out))
        e = {"dec": {"p0": dec("add:EV"), "t:p0": dec("lowconf", note="n"), "t:g2": dec("plain", note="  "),
                     "t:g1": dec(None, note="undone")}, "t": 1}
        out = {o.get("src") or o["id"]: o for o in rs.effective_labels(self.IM, e)}
        self.assertEqual((out["p0"].get("low_conf"), out["p0"].get("note")), (True, "n"))
        self.assertEqual({k for k in out[1]} | {k for k in out[2]}, {"id", "c", "p"})  # undone / blank note: nothing
        self.assertEqual(rs.tag_of(rs._migrate(e)["dec"], "g2"), ("plain", None))
        self.assertTrue(rs.is_low_conf("Low conf EV", None))
        self.assertTrue(rs.is_low_conf("Low conf EV", "plain"))  # the class: off only by turning it into EV
        self.assertFalse(rs.is_low_conf("EV", "plain"))
        self.assertTrue(rs.is_low_conf("MV", "lowconf"))

    def test_is_edited_ignores_image_keys_counts_tags(self):
        self.assertFalse(pp.is_edited({"dec": {"i:note": dec("hello")}}))
        self.assertFalse(pp.is_edited({"dec": {"i:reject": dec("reject:empty"), "i:note": dec("x")}}))
        self.assertTrue(pp.is_edited({"dec": {"t:g1": dec("lowconf")}}))
        self.assertTrue(pp.is_edited({"dec": {"i:note": dec("x"), "g1": dec("keep:EV")}}))
        self.assertTrue(pp.is_rejected({"dec": {"i:reject": dec("reject:empty")}}))
        self.assertFalse(pp.is_rejected({"dec": {"i:reject": dec(None)}}))
        self.assertFalse(pp.is_rejected(None))
        self.assertEqual(rs.image_flags({"dec": {"i:reject": dec("reject:other"), "i:note": dec("n")}}),
                         {"rejected": True, "reject_reason": "other", "note": "n"})
        self.assertEqual(rs.image_flags({"dec": {"g1": dec("del")}}), {})

    def test_reviews_is_checked(self):
        from portal.modules import reviews
        self.assertTrue(reviews._is_checked({"edit": {"dec": {"i:reject": dec("reject:empty")}}}))
        self.assertFalse(reviews._is_checked({"edit": {"dec": {"i:note": dec("just a note")}}}))
        self.assertTrue(reviews._is_checked({"edit": {"dec": {"t:g1": dec("lowconf")}}}))


class ReviewedPolygonsTests(tpp.Tmp):
    def test_new_fields_only_when_set(self):
        pkg = self.build()
        data = pp.read_datajs(pkg)
        a, b, c = data["images"]
        before = pp.reviewed_polygons(pkg, write=False)
        ga = [g["id"] for g in a["gt"]]
        edits = {
            a["file"]: {"dec": {f"g{ga[0]}": dec("keep:EV"), f"t:g{ga[0]}": dec("lowconf", note=XSS),
                                f"g{ga[1]}": dec("keep:Low conf EV"),
                                f"g{ga[2]}": dec("keep:Low conf EV"), f"t:g{ga[2]}": dec("plain", note="off"),
                                f"t:g{ga[3]}": dec("plain", note="just a note"), "i:note": dec("image note")}, "t": 1},
            b["file"]: {"dec": {"i:reject": dec("reject:crystalline ice"), "i:note": dec("ice")}, "t": 1},
            c["file"]: {"dec": {"i:note": dec("only a note")}, "t": 1},
        }
        (pkg / "edits").mkdir(exist_ok=True)
        for f, e in edits.items():
            (pkg / "edits" / f"{f}.json").write_text(json.dumps({"file": f, "edit": e}), encoding="utf-8")
        got = pp.reviewed_polygons(pkg, write=False)
        ia, ib, ic = (got["images"][k] for k in ("0001", "0002", "0003"))
        self.assertEqual([(o["reviewer_label"], o.get("low_conf"), o.get("note")) for o in ia["objects"]],
                         [("EV", True, XSS), ("Low conf EV", True, None), ("Low conf EV", True, "off"),
                          (ia["objects"][3]["reviewer_label"], None, "just a note")])
        self.assertEqual((ia["edited"], ia["note"], "rejected" in ia), (True, "image note", False))
        # 0002: rejected + note, no object decision -> model path, no open candidates, nothing else changed
        want_b = dict(before["images"]["0002"], n_candidates_open=0, rejected=True, reject_reason="crystalline ice", note="ice")
        self.assertEqual(ib, want_b)
        self.assertEqual(list(ib)[:5], list(before["images"]["0002"])[:5])  # the old keys first, in their order
        # 0003: a note alone: still the model path, unedited
        self.assertEqual(ic, dict(before["images"]["0003"], note="only a note"))
        # an unrejected, untouched image carries none of the new keys
        for im in before["images"].values():
            self.assertFalse({"rejected", "reject_reason", "note"} & set(im))
            self.assertFalse(any({"low_conf", "note"} & set(o) for o in im["objects"]))
        # the spot check leaves the rejected image out
        picks = {i for s in range(20) for i in pp.spot_check(pkg, 3, seed=s, write=False)}
        self.assertNotIn("0002", picks)
        self.assertEqual(picks, {"0001", "0003"})


class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="apply_r1_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.base, self.pkg, self.edits = tar.build_synthetic(self.tmp / "in")

    def add(self, file, decs):
        p = self.edits / f"{file}.json"
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["edit"]["dec"].update(decs)
        p.write_text(json.dumps(rec), encoding="utf-8")

    def test_note_no_is_not_a_removal_and_image_keys_change_nothing(self):
        _, out0, _ = tar.run(self.base, self.pkg, self.edits, self.tmp / "l0")
        coco0, rows0, conf0, ver0 = tar.read_out(out0)
        self.add("a.png", {"i:note": dec("no"), "i:reject": dec("reject:empty")})
        self.add("b.png", {"i:note": dec("del")})
        self.add("c.png", {"i:note": dec("no")})
        _, out1, _ = tar.run(self.base, self.pkg, self.edits, self.tmp / "l1")
        coco1, rows1, conf1, ver1 = tar.read_out(out1)
        self.assertEqual((coco1, rows1, conf1), (coco0, rows0, conf0))
        for k in ("skipped", "counts_by_op", "n_images_applied", "n_records_without_changes"):
            self.assertEqual(ver1[k], ver0[k], k)
        self.assertEqual(ver1["rejected_images"], [{"image": "a_png", "reason": "empty", "reviewed": True, "who_raw": "ann",
                                                    "who": "unknown", "note": "no"}])
        self.assertEqual((ver0["rejected_images"], ver0["object_notes"], ver1["object_notes"]), ([], [], []))
        self.assertEqual(ver1["low_conf"], ver0["low_conf"])  # p0 add:Low conf EV, in both
        self.assertEqual(len(coco1["images"]), 3)  # never dropped
        # image_ops directly: the i: / t: keys are not counted as superseded / undone / removed_again
        data = tar.ar.read_data_js(self.pkg / "data.js")
        im = next(i for i in data["images"] if i["file"] == "c.png")
        rec = {"edit": {"dec": {"i:note": dec("no"), "i:reject": dec(None), "t:g1": dec(None), "t:p0": dec("plain", note="no")},
                        "t": 1}}
        ops, sk = ar.image_ops(im, rec, {}, tar.W, tar.H)
        self.assertEqual((ops, sk), ([], Counter()))

    def test_one_coco_rule(self):
        self.add("a.png", {"t:g6": dec("lowconf", 2000, "Pujan"),            # base EV, no class decision -> Low conf EV
                           "t:g2": dec("lowconf", 2000, "Pujan", note="thin"),  # keep:MV -> MV + attributes
                           "t:p1": dec("lowconf", 2000, "Pujan"),            # accepted as EV (add:EV) -> Low conf EV
                           "t:g7": dec("plain", 2000, "Pujan", note=XSS)})   # untouched MV, a note only
        _, out, _ = tar.run(self.base, self.pkg, self.edits, self.tmp / "l")
        coco, rows, conflicts, ver = tar.read_out(out)
        cat = {c["id"]: c["name"] for c in coco["categories"]}
        anns = {a["id"]: a for a in coco["annotations"]}
        self.assertEqual((cat[anns[6]["category_id"]], anns[6].get("attributes")), ("Low conf EV", None))
        self.assertEqual((cat[anns[2]["category_id"]], anns[2]["attributes"]), ("MV", {"low_conf": True, "note": "thin"}))
        self.assertEqual((cat[anns[7]["category_id"]], anns[7]["attributes"]), ("MV", {"note": XSS}))
        p1 = next(r for r in rows if r["dec_key"] == "p1")
        self.assertEqual((p1["op"], p1["cls_to"], cat[anns[p1["ann_new"]]["category_id"]]), ("add", "Low conf EV", "Low conf EV"))
        r6 = next(r for r in rows if r["ann_old"] == 6)
        self.assertEqual((r6["dec_key"], r6["op"], r6["cls_from"], r6["cls_to"], r6["who_raw"], r6["status"]),
                         ("t:g6", "reclass", "EV", "Low conf EV", "Pujan", "applied"))
        self.assertFalse(any(r["dec_key"].startswith("i:") for r in rows))
        # low_conf = what is low conf (tag, or the Low conf EV class: p0 add:Low conf EV); note-only -> object_notes
        self.assertEqual({(x["key"], x["cls"], x["tagged"], x.get("note")) for x in ver["low_conf"]},
                         {("g6", "Low conf EV", True, None), ("g2", "MV", True, "thin"),
                          ("p1", "Low conf EV", True, None), ("p0", "Low conf EV", False, None)})
        self.assertEqual(ver["n_low_conf"], 4)
        self.assertEqual([(x["key"], x["cls"], x["tagged"], x["note"]) for x in ver["object_notes"]], [("g7", "MV", False, XSS)])
        self.assertEqual(ver["n_annotations_out"], json.loads(self.base.read_text())["annotations"].__len__() - 1 + 4)

    def test_tag_only_record_counts_as_a_change(self):
        """A record whose only decision is a t: tag (modified false) is applied: otherwise the tag never reaches COCO."""
        only = self.tmp / "only"
        only.mkdir()
        (only / "b.png.json").write_text(json.dumps({"file": "b.png", "edit": {
            "dec": {"t:g8": dec("lowconf")}, "t": 1, "modified": False, "reviewed": True, "reviewedT": 1}}), encoding="utf-8")
        _, out, _ = tar.run(self.base, self.pkg, only, self.tmp / "l", all_edits=False)
        coco, rows, _, ver = tar.read_out(out)
        cat = {c["id"]: c["name"] for c in coco["categories"]}
        self.assertEqual(cat[{a["id"]: a for a in coco["annotations"]}[8]["category_id"]], "Low conf EV")
        self.assertEqual([(r["dec_key"], r["op"]) for r in rows], [("t:g8", "reclass")])


if __name__ == "__main__":
    unittest.main()
