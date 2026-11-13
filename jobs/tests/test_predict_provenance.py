"""jobs/predict_set.py round 11b: --note, --provenance and source_manifest.csv.

Known answers: a set made with a note, a provenance file and a manifest keeps both in request.json and in the registry
record, and both are still on the record after update_record writes later status lines; source_manifest.csv is the
manifest byte for byte. Each refusal sits next to a control that passes (the same file made valid), and every refusal
writes nothing. A set made without them has none of the new keys (existing sets are unchanged). Temp roots only.
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobtest as T  # noqa: E402
from jobtest import C  # noqa: E402
from test_predict_set import MODEL, PredictRoots, assert_contract_layout  # noqa: E402

import predict_set as PS  # noqa: E402
from registry import store as registry  # noqa: E402

NOTE = "Native-resolution tiles (0.368 nm/px, no downsampling) cut from the 44 Jul30 training images.\nTest of v5."
PROV = {"derived_from": [{"kind": "dataset", "id": "ds_jul30_full"}, {"kind": "model", "id": MODEL}],
        "source_images": "44 Jul30 full-resolution training images",
        "steps": ["cut 1440 px tiles at native resolution", "20% overlap", "8-bit PNG, no contrast change"],
        "tiling": {"tile_px": 1440, "overlap_px": 288, "overlap_frac": 0.2, "nm_per_px": 0.368, "downsample": 1},
        "script": "tools/cut_native_tiles.py", "git": "e69366f"}


class ProvenanceTests(PredictRoots, unittest.TestCase):
    def prov_file(self, obj, name="provenance.json", raw=None):
        p = self.tmp / name
        p.write_text(raw if raw is not None else json.dumps(obj), encoding="utf-8")
        return p

    def test_stored_in_request_and_record_and_carried_forward(self):
        ps = self.make(note=NOTE, provenance=self.prov_file(PROV))
        assert_contract_layout(self, ps, done=False)
        req = C.read_json(ps / "request.json")
        self.assertEqual((req["note"], req["provenance"], req["source_manifest"]), (NOTE, PROV, "source_manifest.csv"))
        self.assertEqual((ps / "source_manifest.csv").read_bytes(), self.manifest.read_bytes())   # byte-identical
        rec = registry.current("prediction_set", self.reg)[ps.name]
        self.assertEqual((rec["note"], rec["provenance"]), (NOTE, PROV))
        self.assertNotIn("source_manifest", rec)                        # request.json only
        # later status lines (update_record's **cur) keep both data fields
        PS.update_record(ps.name, registry_root=self.reg, status="running", recorded_by="test")
        PS.update_record(ps.name, registry_root=self.reg, status="done", results="x/results.json", recorded_by="test")
        lines = [json.loads(ln) for ln in (self.reg / "prediction_set.jsonl").read_text(encoding="utf-8").splitlines()]
        mine = [ln for ln in lines if ln["id"] == ps.name]
        self.assertEqual([ln["status"] for ln in mine], ["queued", "running", "done"])
        for ln in mine:
            self.assertEqual((ln["note"], ln["provenance"]), (NOTE, PROV))
        self.assertEqual(registry.current("prediction_set", self.reg)[ps.name]["provenance"], PROV)

    def test_absent_means_no_new_keys(self):
        d = self.tmp / "plain"
        d.mkdir()
        from test_predict_set import png
        png(d / "a.png", (8, 8))
        ps = self.make(src=d, sample_label="S", sample_label_column=None, nm_per_px=1.0, nm_from_manifest=False,
                       manifest=None)
        req = C.read_json(ps / "request.json")
        self.assertFalse({"note", "provenance", "source_manifest"} & set(req))
        self.assertFalse((ps / "source_manifest.csv").exists())
        rec = registry.current("prediction_set", self.reg)[ps.name]
        self.assertFalse({"note", "provenance"} & set(rec))
        ps2 = self.make()                                   # a manifest alone: copied, no note/provenance
        self.assertEqual((ps2 / "source_manifest.csv").read_bytes(), self.manifest.read_bytes())
        self.assertFalse({"note", "provenance"} & set(C.read_json(ps2 / "request.json")))

    def test_dry_run_validates_and_writes_nothing(self):
        plan = self.make(note=NOTE, provenance=self.prov_file(PROV), dry_run=True)
        self.assertEqual((plan["note"], plan["provenance"]), (NOTE, PROV))
        self.assertIn("source_manifest.csv", plan["source_manifest"])
        self.assertEqual(plan["record"]["provenance"], PROV)
        self.assert_nothing_written()
        with self.assertRaises(PS.PredictSetError):
            self.make(note="=1+1", provenance=self.prov_file(PROV), dry_run=True)
        self.assert_nothing_written()

    def test_provenance_refusals_write_nothing(self):
        big = dict(PROV, steps=["x" * 200] * 15, source_images="y" * 300)
        cases = {
            "unknown key": dict(PROV, colour="blue"),
            "unknown nested key": dict(PROV, tiling={"tile_px": 1440, "stride": 5}),
            "derived_from extra key": dict(PROV, derived_from=[{"kind": "dataset", "id": "a", "url": "x"}]),
            "bad kind": dict(PROV, derived_from=[{"kind": "folder", "id": "a"}]),
            "id too long": dict(PROV, derived_from=[{"kind": "other", "id": "a" * 81}]),
            "too many derived": dict(PROV, derived_from=[{"kind": "other", "id": str(i)} for i in range(11)]),
            "steps not a list": dict(PROV, steps="cut tiles"),
            "too many steps": dict(PROV, steps=["s"] * 16),
            "step too long": dict(PROV, steps=["s" * 201]),
            "step not a string": dict(PROV, steps=[5]),
            "control char": dict(PROV, script="a\tb"),
            "empty tiling": dict(PROV, tiling={}),
            "tile_px float": dict(PROV, tiling={"tile_px": 1440.0}),
            "tile_px bool": dict(PROV, tiling={"tile_px": True}),
            "overlap_frac 1": dict(PROV, tiling={"overlap_frac": 1}),
            "nm_per_px string": dict(PROV, tiling={"nm_per_px": "0.368"}),
            "downsample 0": dict(PROV, tiling={"downsample": 0}),
            "git not hex": dict(PROV, git="main"),
            "git too long": dict(PROV, git="a" * 41),
            "source_images too long": dict(PROV, source_images="s" * 301),
            "not an object": [PROV],
            "empty": {},
        }
        for why, obj in cases.items():
            with self.subTest(why=why):
                with self.assertRaises(PS.PredictSetError) as cm:
                    self.make(note=NOTE, provenance=self.prov_file(obj))
                self.assertIn("provenance", str(cm.exception))
                self.assert_nothing_written()
        raw = {"duplicate key": '{"git": "abc", "git": "def"}', "NaN": '{"tiling": {"nm_per_px": NaN}}',
               "not JSON": "{steps: [}", "oversize": json.dumps(dict(big, script=" " * 200)) + " " * 16384}
        for why, text in raw.items():
            with self.subTest(why=why):
                with self.assertRaises(PS.PredictSetError):
                    self.make(provenance=self.prov_file(None, raw=text))
                self.assert_nothing_written()
        with self.assertRaises(PS.PredictSetError):
            self.make(provenance=self.tmp / "missing.json")
        self.assert_nothing_written()
        # controls: the limits themselves pass (10 derived, 15 steps of 200, 300-character source), with a BOM
        edge = dict(big, derived_from=[{"kind": "other", "id": "i" * 80}] * 10, git="A" * 40)
        p = self.tmp / "edge.json"
        p.write_bytes(b"\xef\xbb\xbf" + json.dumps(edge).encode("utf-8"))
        ps = self.make(provenance=p)
        self.assertEqual(C.read_json(ps / "request.json")["provenance"], edge)
        self.assertEqual(registry.current("prediction_set", self.reg)[ps.name]["provenance"], edge)

    def test_note_rules(self):
        for bad in ("x" * 501, "a\x07b", "a\tb", "a‮b", "line\n" * 10 + "eleven", "=HYPERLINK(1)", "+1",
                    "-x", "@x", "   "):
            with self.subTest(note=bad[:20]):
                with self.assertRaises(PS.PredictSetError) as cm:
                    self.make(note=bad)
                self.assertIn("note", str(cm.exception))
                self.assert_nothing_written()
        ps = self.make(note=" two\r\nlines " + "y" * 480)          # control: CRLF -> LF, trimmed, at the cap
        self.assertEqual(C.read_json(ps / "request.json")["note"], "two\nlines " + "y" * 480)

    def test_cli_options(self):
        prov = self.prov_file(PROV)
        base = [sys.executable, "-m", "jobs.predict_set", "create", "--from-dir", str(self.src), "--model", MODEL,
                "--owner", "pujan", "--manifest", str(self.manifest), "--sample-label-from", "manifest:sample_label",
                "--nm-from-manifest", "--nm-confirmed-by", "pujan", "--registry", str(self.reg), "--root",
                str(self.root), "--note", NOTE, "--provenance", str(prov)]
        r = subprocess.run(base + ["--dry-run"], cwd=T.REPO, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("DRY RUN, nothing written", r.stdout)
        self.assert_nothing_written()
        self.assertIn("--note", PS.create_options())
        self.assertIn("--provenance", PS.create_options())
        r = subprocess.run(base, cwd=T.REPO, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        ps_id = r.stdout.strip().splitlines()[-1]
        self.assertEqual(registry.current("prediction_set", self.reg)[ps_id]["note"], NOTE)


if __name__ == "__main__":
    unittest.main()
