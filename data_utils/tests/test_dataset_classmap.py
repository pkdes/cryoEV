"""dataset_from_lv.py with class maps (round 14): --class-map vs --class-scheme, preview (no writes), derive.

Known answers (controls):
  - --class-scheme X and --class-map cm_X_v1 write byte-identical label files, dataset.yaml and image_list.tsv, and
    provenance differs only in class_map_id / class_map_sha256 and the created / command fields.
  - preview() (dry run) gives the same counts and labels_sha as the folder build() writes.
  - collapsing the ev_nonev_undet dataset with cm_ev_nonev_v1 gives the ev_nonev dataset's labels_sha (the reuse case).
Synthetic fixture (test_dataset_from_lv.py); temp folders and a temp registry only.

    python -m unittest discover -s data_utils/tests -p test_dataset_classmap.py
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_dataset_from_lv as F  # noqa: E402
from test_dataset_from_lv import D, REPO  # noqa: E402
from labels import classmaps as CM  # noqa: E402


def files(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*"))
            if p.is_file() and p.name != D.PROVENANCE}


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(("+" if k not in a else "-") + f"{path}/{k}")
            else:
                out += diff(a[k], b[k], f"{path}/{k}")
        return out
    return [] if a == b else [f"~{path}"]


class TestSchemeVsMap(F.Fixture):
    def test_byte_identical_and_only_new_keys(self):
        for scheme, cm in (("ev", "cm_ev_v1"), ("ev_nonev", "cm_ev_nonev_v1"),
                           ("ev_nonev_undet", "cm_ev_nonev_undet_v1")):
            for eo in ("drop", "export-labels"):
                out = self.root / f"o_{scheme}_{eo}"
                pa, _ = F.build(self.base_lv, self.export, "ds", out, scheme, eo)
                (out / "ds").rename(out / "ds_scheme")
                pb, log = F.build(self.base_lv, self.export, "ds", out, export_only=eo, class_map=cm)
                fa, fb = files(out / "ds_scheme"), files(out / "ds")
                self.assertGreater(sum(k.startswith("labels/") for k in fa), 2)
                self.assertEqual(fa, fb, (scheme, eo))                    # labels, dataset.yaml, image list, images
                d = [x for x in diff(pa, pb) if x not in ("~/dataset_record/created", "~/details/command")]
                self.assertEqual(d, ["+/dataset_record/class_map_id", "+/dataset_record/class_map_sha256"])
                rec = pb["dataset_record"]
                self.assertEqual((rec["class_map_id"], rec["class_scheme"]), (cm, scheme))
                self.assertEqual(rec["class_map_sha256"], CM.sha256(CM.load(cm)))
                self.assertNotIn("class_map_id", pa["dataset_record"])
                self.assertIn(f"Class map:     {cm}", log)

    def test_map_missing_a_class_is_refused(self):
        cm = dict(CM.load("cm_ev_nonev_v1"), id="cm_short_v1", name="short")
        cm["map"] = {k: v for k, v in cm["map"].items() if k != "MV"}
        with self.assertRaisesRegex(D.BuildError, r"\['MV'\] are not in class map cm_short_v1"):
            F.build(self.base_lv, self.export, "ds", self.root / "out", class_map=cm)
        self.assertFalse((self.root / "out" / "ds").exists())

    def test_cli_class_map(self):
        env = dict(os.environ, CRYOEV_REGISTRY=str(self.root / "reg"))
        base = [sys.executable, str(REPO / "data_utils" / "dataset_from_lv.py"), "--lv", str(self.base_lv),
                "--split-from", str(self.export), "--output-root", str(self.root / "cli"), "--export-only", "drop"]
        r = subprocess.run(base + ["--output-name", "a", "--class-map", "cm_ev_nonev_v1", "--class-scheme", "ev"],
                           capture_output=True, text=True, env=env, cwd=REPO)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("alternatives", r.stderr)
        r = subprocess.run(base + ["--output-name", "a", "--class-map", "cm_ev_nonev_v1"],
                           capture_output=True, text=True, env=env, cwd=REPO)
        self.assertEqual(r.returncode, 0, r.stderr)
        rec = json.loads((self.root / "cli" / "a" / D.PROVENANCE).read_text(encoding="utf-8"))["dataset_record"]
        self.assertEqual(rec["class_map_id"], "cm_ev_nonev_v1")
        self.assertFalse((self.root / "reg").exists())                      # no --register: no registry


class TestPreview(F.Fixture):
    def test_preview_equals_build_and_writes_nothing(self):
        for kw in ({"scheme": "ev"}, {"class_map": "cm_ev_nonev_v1"}, {"class_map": "cm_ev_nonev_undet_v1"}):
            for eo in ("drop", "export-labels"):
                before = sorted(p for p in self.root.rglob("*"))
                pv = D.preview(str(self.base_lv), str(self.export), eo, **kw)
                self.assertEqual(sorted(p for p in self.root.rglob("*")), before)          # nothing written here
                out = self.root / f"o_{len(before)}"
                prov, _ = F.build(self.base_lv, self.export, "ds", out, export_only=eo, **kw)
                rec = prov["dataset_record"]
                self.assertEqual(pv["class_counts"], rec["class_counts"], (kw, eo))
                self.assertEqual(pv["counts"], rec["counts"], (kw, eo))
                self.assertEqual(pv["labels_sha"], CM.labels_sha(out / "ds"), (kw, eo))


    def test_preview_does_not_wait_for_a_running_build(self):
        import threading
        held, release = threading.Event(), threading.Event()

        def build_running():                    # what a build does: prepare's globals swapped, lock held
            with D.use_tables(D.prep.CLASS_SCHEMES["ev"]):
                held.set()
                release.wait(30)
        t = threading.Thread(target=build_running)
        t.start()
        try:
            self.assertTrue(held.wait(10))
            pv = D.preview(str(self.base_lv), str(self.export), "drop", class_map="cm_ev_nonev_v1")
            self.assertEqual(pv["classes"], ["EV", "non-EV"])
            self.assertEqual(pv["class_counts"]["train"], {"EV": 4, "non-EV": 1})
        finally:
            release.set()
            t.join(10)


class TestDerive(F.Fixture):
    def setUp(self):
        super().setUp()
        env = mock.patch.dict(os.environ, {"CRYOEV_LAPTOP_ROOT": str(self.root), "CRYOEV_CRYOAI_ROOT": str(self.root),
                                           "CRYOEV_REGISTRY": str(self.root / "registry")})
        env.start()
        self.addCleanup(env.stop)
        self.reg = self.root / "registry"
        self.out = self.root / "outputs"
        for name, scheme in (("lv_ds_undet", "ev_nonev_undet"), ("lv_ds_evnonev", "ev_nonev"), ("lv_ds_ev", "ev")):
            prov, _ = F.build(self.base_lv, self.export, name, self.out, scheme, "export-labels")
            D.register(prov, self.out / name, self.reg)

    def test_preview_finds_the_identical_dataset(self):
        pv = D.derive_preview("lv_ds_undet", "cm_ev_nonev_v1", self.reg)
        self.assertEqual(pv["identical"], ["lv_ds_evnonev"])
        self.assertEqual(pv["labels_sha"], D.labels_sha(self.out / "lv_ds_evnonev"))
        self.assertNotEqual(pv["labels_sha"], D.labels_sha(self.out / "lv_ds_undet"))
        self.assertEqual(pv["target"], "lv_ds_undet__cm_ev_nonev_v1")
        self.assertFalse(pv["target_exists"])
        self.assertEqual(pv["before_classes"], ["EV", "non-EV", "Undetermined"])
        self.assertEqual(pv["identical"], D.derive_preview("lv_ds_ev", "cm_ev_nonev_v1", self.reg)["identical"])
        self.assertEqual(D.derive_preview("lv_ds_undet", "cm_ev_v1", self.reg)["identical"], ["lv_ds_ev"])

    def test_derive_builds_and_registers_the_copy(self):
        prov = D.derive("lv_ds_undet", "cm_ev_nonev_v1", self.reg, self.out, register_it=True, log=lambda *a: None)
        rec = prov["dataset_record"]
        self.assertEqual(rec["id"], "lv_ds_undet__cm_ev_nonev_v1")
        self.assertEqual((rec["parent_dataset"], rec["class_map_id"]), ("lv_ds_undet", "cm_ev_nonev_v1"))
        self.assertIn({"id": "lv_ds_undet", "role": "base", "entity": "dataset"}, rec["inputs"])
        self.assertEqual(rec["export_only"]["choice"], "export-labels")       # the parent's choice
        self.assertEqual(D.labels_sha(self.out / rec["id"]), D.labels_sha(self.out / "lv_ds_evnonev"))
        from registry import store
        cur = store.current("dataset", self.reg)
        self.assertEqual(cur[rec["id"]]["parent_dataset"], "lv_ds_undet")
        with self.assertRaisesRegex(D.BuildError, "already has"):
            D.derive("lv_ds_undet", "cm_ev_nonev_v1", self.reg, self.out, register_it=True, log=lambda *a: None)

    def test_only_label_set_datasets(self):
        self.assertIsNone(D.derive_problem(D.registered(self.reg)["lv_ds_ev"]))
        self.assertIn("label set", D.derive_problem({"id": "x", "source": {"kind": "roboflow_export", "id": "r"}}))
        self.assertIn("not on this computer", D.derive_problem(
            {"id": "x", "label_version": "lv_20990101_0000", "source": {"kind": "label_version"}, "export": "e",
             "dataset_dir": str(self.root / "nowhere")}))
        with self.assertRaisesRegex(D.BuildError, "no dataset"):
            D.derive("missing", "cm_ev_v1", self.reg, self.out)

    def test_changed_lv_is_refused(self):
        lv = self.base_lv / "labels.coco.json"
        lv.write_bytes(lv.read_bytes() + b" ")
        with self.assertRaisesRegex(D.BuildError, "changed"):
            D.derive_preview("lv_ds_undet", "cm_ev_nonev_v1", self.reg)

    # ---- round 14 QC fixes (items 17-19)
    def test_parent_drift_is_refused(self):
        """derive_preview / derive first preview the parent with its own classes: that must equal the parent folder's
        labels_sha (the known answer). A label file edited in the parent folder breaks it: refused, nothing written."""
        pv = D.derive_preview("lv_ds_undet", "cm_ev_nonev_v1", self.reg)             # control: unchanged parent passes
        self.assertEqual(pv["parent_check"], "ok")
        lab = next((self.out / "lv_ds_undet" / "labels" / "train").glob("*.txt"))
        lab.write_bytes(lab.read_bytes() + b"0 0.1 0.1 0.2 0.1 0.2 0.2\n")
        with self.assertRaisesRegex(D.BuildError, "does not match a rebuild"):
            D.derive_preview("lv_ds_undet", "cm_ev_nonev_v1", self.reg)
        with self.assertRaisesRegex(D.BuildError, "does not match a rebuild"):
            D.derive("lv_ds_undet", "cm_ev_nonev_v1", self.reg, self.out, register_it=True, log=lambda *a: None)
        self.assertFalse((self.out / "lv_ds_undet__cm_ev_nonev_v1").exists())

    def test_failed_register_leaves_no_folder(self):
        with mock.patch.object(D, "register", side_effect=D.BuildError("registry refused: boom")):
            with self.assertRaisesRegex(D.BuildError, "boom"):
                D.derive("lv_ds_undet", "cm_ev_nonev_v1", self.reg, self.out, register_it=True, log=lambda *a: None)
        self.assertFalse((self.out / "lv_ds_undet__cm_ev_nonev_v1").exists())
        self.assertEqual([p.name for p in self.out.iterdir() if p.name.startswith(".")], [])    # no temp left

    def test_derive_builds_in_temp_then_renames(self):
        seen = []
        real = D.build

        def spy(*a, **kw):
            seen.append(Path(a[3]))
            return real(*a, **kw)
        with mock.patch.object(D, "build", side_effect=spy):
            prov = D.derive("lv_ds_undet", "cm_ev_v1", self.reg, self.out, register_it=True, log=lambda *a: None)
        final = self.out / "lv_ds_undet__cm_ev_v1"
        self.assertNotEqual(seen[-1], self.out)                                    # built under a temp root
        self.assertEqual(seen[-1].parent, self.out)
        self.assertFalse(seen[-1].exists())                                        # temp root removed
        self.assertEqual(prov["dataset_record"]["dataset_dir"], str(final))
        disk = json.loads((final / D.PROVENANCE).read_text(encoding="utf-8"))["dataset_record"]
        self.assertEqual(disk["dataset_dir"], str(final))
        self.assertIn(f"path: {final.as_posix()}\n", (final / "dataset.yaml").read_text(encoding="utf-8"))
        from registry import store
        self.assertEqual(store.current("dataset", self.reg)[final.name]["dataset_dir"], str(final))
        self.assertEqual(D.labels_sha(final), D.labels_sha(self.out / "lv_ds_ev"))     # known answer still holds

    def test_preview_flags_unregistered_folder(self):
        pv = D.derive_preview("lv_ds_undet", "cm_ev_v1", self.reg, output_root=self.out)
        self.assertFalse(pv["target_dir_exists"])
        (self.out / "lv_ds_undet__cm_ev_v1").mkdir()
        pv = D.derive_preview("lv_ds_undet", "cm_ev_v1", self.reg, output_root=self.out)
        self.assertTrue(pv["target_dir_exists"])
        self.assertFalse(pv["target_exists"])

    def test_superseded_lv_is_flagged_in_the_picker(self):
        rec = D.registered(self.reg)["lv_ds_undet"]
        self.assertIsNone(D.derive_problem(rec))
        (self.base_lv / "SUPERSEDED.md").write_text("use lv_x", encoding="utf-8")
        self.assertIn("superseded", D.derive_problem(rec))
        with self.assertRaisesRegex(D.BuildError, "superseded"):
            D.derive_preview("lv_ds_undet", "cm_ev_v1", self.reg)


if __name__ == "__main__":
    unittest.main()
