"""Round 14: the label_schema registry kind (contract C2) and object sets in the importer / validator (C4).

- both built-in schemas give a valid label_schema/1 line; store.append(files=...) writes the definition once and
  refuses other content under the same path; the line reads back through current();
- validate.Scan checks the repo's built-in schema files (label_schema entity);
- the importer skips an object set (data.js kind "objects", or kind objects in review_package.json) entirely: no
  review_package line, no revise job; a CLI lab package on the same registry still gets both (control).

  python -m unittest registry.tests.test_label_schema_registry -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REG.parent))
from labels import schemas as S  # noqa: E402
from registry import importers  # noqa: E402
from registry import store  # noqa: E402
from registry import validate  # noqa: E402
from registry.schema_check import check_record  # noqa: E402

LV = "lv_20991231_0101"
CLI, OBJ, OBJ_MID = "lab_compare_cli_20991231", "pred_obj_lamellarity_ab12", "pred_obj_lamellarity_cd34"


def w(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def datajs(pid, source, **extra):
    return "window.REVIEW_DATA = " + json.dumps({"id": pid, "kind": "compare", "source": source,
                                                 "run": "v5_evnonev_y8m_20261029", "created": "2099-12-31T10:00:00",
                                                 "images": [{"file": "0001.png"}], **extra}) + ";"


class LabelSchemaKind(unittest.TestCase):
    def test_lines_valid_and_written_once(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "registry"
            for sid in S.builtin_ids():
                rec, files = S.registry_record(S.load(sid), "2099-12-31T10:00:00")
                self.assertEqual(check_record(rec, store.schemas()), [], sid)
                store.append(rec, root, new_only=True, files=files, via=None)
                rel = next(iter(files))
                self.assertEqual((root / rel).read_bytes(), files[rel])
                self.assertEqual(S.load(sid, root), S.load(sid))
            cur = store.current("label_schema", root)
            self.assertEqual(sorted(cur), S.builtin_ids())
            self.assertEqual(cur["ls_lamellarity_v1"]["maker"], "pujan")
            # the same path with other content is refused before any line is written
            rec, files = S.registry_record(S.load("ls_lamellarity_v1"), "2099-12-31T10:00:01")
            rec["id"] = "ls_other_v1"
            files = {next(iter(files)): b"{}"}
            with self.assertRaises(store.RegistryError):
                store.append(rec, root, files=files, via=None)
            self.assertEqual(len((root / "label_schema.jsonl").read_text(encoding="utf-8").splitlines()), 2)

    def test_bad_line_refused(self):
        rec, _ = S.registry_record(S.load("ls_lamellarity_v1"), "2099-12-31T10:00:00")
        self.assertTrue(check_record({**rec, "id": "lamellarity"}, store.schemas()))
        self.assertTrue(check_record({**rec, "def_sha256": "x"}, store.schemas()))

    def test_validate_scans_builtins(self):
        s = validate.Scan(None, None, None)
        s.scan_label_schemas()
        d = s.t["label_schema"].as_dict()
        self.assertEqual((d["found"], d["identified"], d["linked"]), (2, 2, 2))
        self.assertIn("label_schema", validate.ENTITIES)


class ObjectSetSkipped(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.lap, self.cache, self.root = t / "CryoAI-laptop", t / "CryoAI", t / "registry"
        (self.cache / "training images").mkdir(parents=True)
        (self.cache / "training outputs").mkdir(parents=True)
        labels = str(self.lap / "labels" / LV / "labels.coco.json")
        rev = self.lap / "review"
        w(rev / CLI / "data.js", datajs(CLI, labels))
        w(rev / CLI / "review_package.json", json.dumps({"id": CLI, "kind": "compare", "owner": "pujan"}))
        # an object set whose data.js names a non-prediction source, so only the objects rule can skip it
        w(rev / OBJ / "data.js", datajs(OBJ, labels, kind="objects"))
        w(rev / OBJ / "review_package.json", json.dumps({"id": OBJ, "kind": "objects", "owner": "pujan"}))
        w(rev / OBJ_MID / "data.js", datajs(OBJ_MID, labels))                     # data.js says compare, the record objects
        w(rev / OBJ_MID / "review_package.json", json.dumps({"id": OBJ_MID, "kind": "objects", "owner": "pujan"}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_skipped_no_revise_job(self):
        res = importers.run(self.root, dry_run=False, laptop=self.lap, cache=self.cache, processing=None, via=None)
        self.assertEqual(res["excluded"]["review_package (object set, own line)"], 2)
        cur = store.current(root=self.root)
        self.assertIn(CLI, cur)                                  # control: the lab package as before
        self.assertTrue(any(r.get("schema") == "job/1" and CLI in (r.get("name") or "") for r in cur.values()))
        for pid in (OBJ, OBJ_MID):
            self.assertNotIn(pid, cur)
            self.assertFalse(any(pid in json.dumps(r) for r in cur.values()))
        self.assertTrue(importers.object_set(self.lap / "review" / OBJ))
        self.assertFalse(importers.object_set(self.lap / "review" / CLI))


if __name__ == "__main__":
    unittest.main()
