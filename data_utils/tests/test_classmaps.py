"""labels/classmaps.py (round 14, class maps) and the models.yaml `class_map:` in inference/model_classes.py.

Known answer (control): the built-in maps are generated from prepare_all_layer_singleclass.CLASS_SCHEMES, and
to_scheme(map) gives back exactly the scheme's (dict, list), key order included. If that fails, nothing built with
--class-map can be trusted. Temp folders only; the user-map root is a temp folder ($CRYOEV_LAPTOP_ROOT).

    python -m unittest discover -s data_utils/tests -p test_classmaps.py
"""
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from data_utils import prepare_all_layer_singleclass as prep  # noqa: E402
from inference import model_classes as MC  # noqa: E402
from labels import classmaps as CM  # noqa: E402

V5 = "v5_evnonev_y8m_20261029"


def user_map(**kw):
    cm = copy.deepcopy(CM.builtin()["cm_ev_nonev_v1"])
    cm.update({"id": "cm_test_merge_v1", "name": "test_merge", "maker": "tester", "description": "test"})
    cm.update(kw)
    return cm


class TestBuiltins(unittest.TestCase):
    def test_files_are_generated_from_class_schemes(self):
        gen = CM.generate_builtins()
        self.assertEqual(sorted(gen), sorted(CM.BUILTIN_SCHEMES))
        for cid, cm in gen.items():
            f = CM.BUILTIN_DIR / f"{cid}.json"
            self.assertEqual(f.read_bytes().replace(b"\r\n", b"\n"), CM.builtin_file_bytes(cm), cid)
        self.assertEqual(CM.builtin(), gen)

    def test_to_scheme_is_the_scheme_exactly(self):
        for cid, scheme in CM.BUILTIN_SCHEMES.items():
            d, names = CM.to_scheme(CM.load(cid))
            want_d, want_names = prep.CLASS_SCHEMES[scheme]
            self.assertEqual((d, names), (want_d, want_names), cid)
            self.assertEqual(list(d.items()), list(want_d.items()), cid)          # same order, same None drops
            self.assertEqual(type(names), type(want_names))
        self.assertEqual(CM.to_scheme(CM.load("cm_ev_v1"))[0]["VLDL"], None)
        self.assertEqual(CM.to_scheme(CM.load("cm_ev_nonev_undet_v1"))[0]["Low conf EV"], 2)

    def test_ev_ids_and_sha(self):
        self.assertEqual(CM.ev_ids(CM.load("cm_ev_nonev_undet_v1")), {0})
        cm = CM.load("cm_ev_nonev_v1")
        self.assertEqual(CM.sha256(cm), CM.sha256(json.loads(json.dumps(cm, indent=4))))   # formatting-independent
        self.assertEqual(len(CM.sha256(cm)), 64)


class TestValidation(unittest.TestCase):
    def bad(self, msg, **kw):
        with self.assertRaisesRegex(CM.ClassMapError, msg):
            CM.validate(user_map(**kw))

    def test_refusals(self):
        CM.validate(user_map())
        self.bad("schema", schema="class_map/2")
        self.bad("id", id="Bad id")
        self.bad("_v1", version=2)
        self.bad("not one of classes", map={"EV": "EV", "VLDL": "Lipo", "non-EV": "non-EV"})
        self.bad("get no source class", classes=["EV", "non-EV", "Other"])
        self.bad("ev_classes", ev_classes=["Lipo"])
        self.bad("distinct", classes=["EV", "EV"])
        self.bad("unknown keys", extra=1)
        m = user_map()
        del m["supersedes"]
        with self.assertRaisesRegex(CM.ClassMapError, "supersedes"):
            CM.validate(m)


class TestUserMaps(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cm_")
        self.lap = Path(self.tmp.name)
        p = mock.patch.dict(os.environ, {"CRYOEV_LAPTOP_ROOT": str(self.lap)})
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_written_once(self):
        cm = user_map()
        path, what = CM.write_user(cm)                                    # dry run: nothing written
        self.assertEqual((path, what), (self.lap / "classmaps" / "cm_test_merge_v1.json", "would write"))
        self.assertFalse(path.exists())
        self.assertEqual(CM.write_user(cm, do=True)[1], "written")
        self.assertEqual(CM.write_user(cm, do=True)[1], "exists, identical")
        with self.assertRaisesRegex(CM.ClassMapError, "written once"):
            CM.write_user(dict(cm, description="changed"), do=True)
        with self.assertRaisesRegex(CM.ClassMapError, "built-in"):
            CM.write_user(dict(CM.load("cm_ev_v1"), maker="me"), do=True)
        self.assertEqual(CM.load("cm_test_merge_v1"), cm)
        self.assertEqual([m["id"] for m in CM.list_all() if m["_origin"] == "user"], ["cm_test_merge_v1"])
        with self.assertRaisesRegex(CM.ClassMapError, "changed after"):
            CM.load("cm_test_merge_v1", expect_sha256="0" * 64)
        self.assertEqual(CM.load(str(path)), cm)                              # by file

    def test_cli_add_is_a_dry_run_without_do(self):
        f = self.lap / "in.json"
        f.write_text(json.dumps(user_map()), encoding="utf-8")
        self.assertEqual(CM.main(["add", str(f)]), 0)
        self.assertFalse((self.lap / "classmaps").exists())
        self.assertEqual(CM.main(["add", str(f), "--do"]), 0)
        self.assertTrue((self.lap / "classmaps" / "cm_test_merge_v1.json").is_file())

    def test_unknown_id(self):
        with self.assertRaisesRegex(CM.ClassMapError, "no class map"):
            CM.load("cm_nothing_v1")


class TestCompat(unittest.TestCase):
    def test_check_compat(self):
        v5 = MC.model_classes(V5)
        self.assertEqual(CM.check_compat(v5, CM.load("cm_ev_nonev_v1")), [])
        self.assertEqual(CM.check_compat(v5, CM.load("cm_ev_nonev_undet_v1")), [])
        probs = CM.check_compat(v5, CM.load("cm_ev_v1"))
        self.assertEqual(len(probs), 1)
        self.assertIn("non-EV", probs[0])
        schema = {"id": "ls_x_v1", "classes": [{"name": "EV"}, {"name": "non-EV"}]}       # label schema shape
        self.assertEqual(CM.check_compat(v5, schema), [])
        self.assertEqual(len(CM.check_compat({0: "Lipo", 1: "non-EV"}, schema)), 2)      # missing + no EV
        self.assertEqual(CM.check_compat({}, schema), [])                                 # single-class
        self.assertEqual(CM.check_compat({0: "vesicle"}, schema), [])
        self.assertEqual(len(CM.check_compat({}, {"classes": [{"name": "Lipo"}]})), 1)
        self.assertEqual(CM.check_compat(["EV", "non-EV"], ["EV", "non-EV"]), [])


class TestLabelsSha(unittest.TestCase):
    def test_folder_independent_and_sensitive(self):
        with tempfile.TemporaryDirectory(prefix="cm_") as t:
            a = Path(t) / "a" / "ds1"
            (a / "labels" / "train").mkdir(parents=True)
            (a / "images" / "train").mkdir(parents=True)
            (a / "labels" / "train" / "x.txt").write_bytes(b"0 0.1 0.2 0.3 0.4 0.5 0.6\n")
            (a / "images" / "train" / "x.jpg").write_bytes(b"img")
            (a / "dataset.yaml").write_text(f"# YOLO dataset config for ds1 - 1 class(es): ['EV']\npath: {a}\n"
                                            "train: images/train\nnc: 1\nnames:\n  0: EV\n", encoding="utf-8")
            sha = CM.labels_sha(a)
            b = Path(t) / "b" / "other"
            shutil.copytree(a, b)
            (b / "dataset.yaml").write_text((b / "dataset.yaml").read_text(encoding="utf-8")
                                            .replace(str(a), str(b)).replace("ds1", "other"), encoding="utf-8")
            (b / "labels" / "train" / "x.txt").write_bytes(b"0 0.1 0.2 0.3 0.4 0.5 0.6\r\n")       # CRLF = LF
            self.assertEqual(CM.labels_sha(b), sha)
            (b / "images" / "train" / "y.jpg").write_bytes(b"img")
            self.assertNotEqual(CM.labels_sha(b), sha)
            (b / "images" / "train" / "y.jpg").unlink()
            (b / "labels" / "train" / "x.txt").write_bytes(b"1 0.1 0.2 0.3 0.4 0.5 0.6\n")
            self.assertNotEqual(CM.labels_sha(b), sha)
            with self.assertRaises(CM.ClassMapError):
                CM.labels_sha(Path(t))

    def test_same_names_other_image_bytes_differ(self):
        """QC 17: two datasets with the same file names and labels but different image files must not match."""
        with tempfile.TemporaryDirectory(prefix="cm_") as t:
            a = Path(t) / "a"
            (a / "labels" / "train").mkdir(parents=True)
            (a / "images" / "train").mkdir(parents=True)
            (a / "labels" / "train" / "x.txt").write_bytes(b"0 0.1 0.2 0.3 0.4 0.5 0.6\n")
            (a / "images" / "train" / "x.jpg").write_bytes(b"img")
            (a / "dataset.yaml").write_text("train: images/train\nnc: 1\nnames:\n  0: EV\n", encoding="utf-8")
            b = Path(t) / "b"
            shutil.copytree(a, b)
            self.assertEqual(CM.labels_sha(b), CM.labels_sha(a))                   # control: a copy matches
            (b / "images" / "train" / "x.jpg").write_bytes(b"other pixels")
            self.assertNotEqual(CM.labels_sha(b), CM.labels_sha(a))
            self.assertEqual(CM.labels_sha_of({}, "", {"images/train/x.jpg": 3}),
                             CM.labels_sha_of({}, "", {"images/train/x.jpg": 3}))
            self.assertNotEqual(CM.labels_sha_of({}, "", {"images/train/x.jpg": 3}),
                                CM.labels_sha_of({}, "", {"images/train/x.jpg": 4}))


class TestModelClasses(unittest.TestCase):
    def reg(self, models):
        t = tempfile.TemporaryDirectory(prefix="cm_")
        self.addCleanup(t.cleanup)
        p = Path(t.name) / "models.yaml"
        import yaml
        p.write_text(yaml.safe_dump({"models": models}), encoding="utf-8")
        return p

    def test_real_registry_unchanged(self):
        self.assertEqual(MC.model_class_map(V5), "cm_ev_nonev_v1")
        self.assertEqual(MC.ev_class_ids(V5), {0})
        self.assertIsNone(MC.ev_class_ids("v4_y8m_20260924"))
        self.assertEqual(MC.describe("v4_y8m_20260924"),
                         "classes: v4_y8m_20260924 is single-class; every prediction is an EV")

    def test_map_rules(self):
        r = self.reg({"multi": {"classes": {0: "non-EV", 1: "EV"}, "class_map": "cm_ev_nonev_v1"},
                      "plain": {"classes": {0: "non-EV", 1: "EV"}},
                      "single": {"classes": {0: "EV"}, "class_map": "cm_ev_nonev_v1"},
                      "none": {"class_map": "cm_ev_nonev_v1"},
                      "bad": {"classes": {0: "Lipo", 1: "EV"}, "class_map": "cm_ev_nonev_v1"},
                      "noev": {"classes": {0: "non-EV", 1: "Undetermined"}, "class_map": "cm_ev_nonev_undet_v1"}})
        self.assertEqual(MC.ev_class_ids("multi", r), {1})
        self.assertEqual(MC.ev_class_ids("plain", r), {1})
        self.assertIsNone(MC.ev_class_ids("single", r))                  # single-class: None, map or not
        self.assertIsNone(MC.ev_class_ids("none", r))
        with self.assertRaisesRegex(ValueError, "not in its class map"):
            MC.ev_class_ids("bad", r)
        with self.assertRaisesRegex(ValueError, "no class"):
            MC.ev_class_ids("noev", r)


if __name__ == "__main__":
    unittest.main()
