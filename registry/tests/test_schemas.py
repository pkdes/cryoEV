"""Schemas: every entity has a sample that validates; broken copies fail (control); ids helpers."""
import copy
import json
import sys
import unittest
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REG))
import ids  # noqa: E402
from schema_check import UnsupportedKeyword, check_record, errors, load_schemas  # noqa: E402

ENTITIES = ["image", "session", "grid", "roboflow_image", "pull", "roboflow_export", "label_version",
            "review_package", "inference_check", "dataset", "job_bundle", "run", "model"]


def samples():
    out = {}
    for p in sorted((REG / "samples").glob("*.jsonl")):
        out[p.stem] = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return out


class TestSchemas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schemas = load_schemas()
        cls.samples = samples()

    def test_one_schema_and_sample_per_entity(self):
        for e in ENTITIES + ["label_change"]:
            self.assertIn(f"{e}/1", self.schemas, e)
            self.assertTrue(self.samples.get(e), f"no sample for {e}")
            sch = json.loads((REG / "schemas" / f"{e}.json").read_text(encoding="utf-8"))
            self.assertEqual(sch["$schema"], "https://json-schema.org/draft/2020-12/schema")

    def test_samples_validate(self):
        for e, recs in self.samples.items():
            for r in recs:
                self.assertEqual(check_record(r, self.schemas), [], e)
                self.assertEqual(r["schema"], f"{e}/1")

    def test_envelope_required_on_entities(self):
        for e in ENTITIES:
            req = self.schemas[f"{e}/1"]["required"]
            for k in ("schema", "id", "logged"):
                self.assertIn(k, req, e)

    def test_control_broken_records_fail(self):
        """Known-bad copies must fail; if any passes, the checker is broken."""
        for e in ENTITIES:
            good = self.samples[e][0]
            for k in self.schemas[f"{e}/1"]["required"]:
                bad = copy.deepcopy(good)
                del bad[k]
                self.assertTrue(check_record(bad, self.schemas), f"{e}: missing {k} passed")
            bad = dict(good, id="has space/and slash")
            self.assertTrue(check_record(bad, self.schemas), f"{e}: bad id passed")
            if "status" in good:
                self.assertTrue(check_record(dict(good, status="nonsense"), self.schemas), f"{e}: bad status passed")
            self.assertTrue(check_record(dict(good, schema=f"{e}/2"), self.schemas))
            self.assertTrue(check_record(dict(good, logged="yesterday"), self.schemas))

    def test_label_change_v1_fields_untouched(self):
        """label_change/1 keeps exactly the LABEL_CHANGES.md v1 field set (owned by apply_review_edits.py)."""
        v1 = {"schema", "label_version", "base", "pkg", "image", "rf_id", "dec_key", "op", "ann_old", "ann_new",
              "cls_from", "cls_to", "who_raw", "who", "who_auth", "t", "rx", "seed", "reviewed", "status", "reason"}
        self.assertEqual(set(self.schemas["label_change/1"]["properties"]), v1)
        self.assertTrue(self.schemas["label_change/1"]["additionalProperties"])

    def test_checker_basics(self):
        self.assertEqual(errors(1, {"type": "integer"}), [])
        self.assertTrue(errors(True, {"type": "integer"}))       # bool is not an integer
        self.assertTrue(errors(1.5, {"type": "integer"}))
        self.assertEqual(errors(None, {"type": ["string", "null"]}), [])
        self.assertTrue(errors([1, 1], {"type": "array", "uniqueItems": True}))
        self.assertTrue(errors({"a": 1}, {"type": "object", "additionalProperties": False}))
        with self.assertRaises(UnsupportedKeyword):
            errors("x", {"type": "string", "format": "date-time"})

    def test_cross_check_with_jsonschema_if_installed(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        for e, recs in self.samples.items():
            jsonschema.Draft202012Validator(self.schemas[f"{e}/1"]).validate(recs[0])


class TestIds(unittest.TestCase):
    def test_image_key_forms_agree(self):
        rf_name = "20260126 - MDA-MB KH__2_square143_hole73_0_hm.png"
        export = "20260126-MDA-MB-KH__2_square143_hole73_0_hm_png.rf.74ae79bbab6ae84e406b44b9cdb29da4.jpg"
        key = "20260126-MDA-MB-KH__2_square143_hole73_0_hm_png"
        self.assertEqual(ids.image_key(rf_name), key)
        self.assertEqual(ids.image_key(export), key)
        self.assertEqual(ids.image_key(key), key)                     # idempotent
        self.assertEqual(ids.image_key("0004_Jul30.jpg"), "0004_Jul30_jpg")
        self.assertTrue(ids.is_id("stem", key))
        self.assertFalse(ids.is_id("stem", rf_name))                  # raw names with spaces are not keys

    def test_parsers(self):
        r = ids.parse_run("2026-10-29_054810_y8m_1440_8b_300ep_Run_15")
        self.assertEqual((r["size"], r["imgsz"], r["batch"], r["epochs"], r["num"]), ("m", 1440, 8, 300, 15))
        self.assertIsNone(ids.parse_run("2026-10-29_054810_probe_y8m_b8"))
        self.assertTrue(ids.is_id("run", "2026-10-29_054810_probe_y8m_b8"))
        self.assertEqual(ids.bundle_dataset("ds_x__job_20261030-140000"), "ds_x")
        self.assertIsNone(ids.bundle_dataset("ds_x"))
        self.assertEqual(ids.package_parent("prelipo_v5F5_review_v2_part3"), "prelipo_v5F5_review_v2")
        self.assertEqual(ids.session_of_ic("20261027_v5_evnonev_y8m_20261029_20261030-090300"), "20261027")
        self.assertTrue(ids.is_id("session", "20260827-2"))
        self.assertTrue(ids.is_id("model_new", "v5_evnonev_y8m_20261029"))
        self.assertFalse(ids.is_id("model_new", "yifei_202512"))


if __name__ == "__main__":
    unittest.main()
