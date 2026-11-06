"""data_utils/dataset_from_lv.py: lv labels + an export's split -> the dataset prepare_all_layer_singleclass.py makes.

Known answer (control): an lv with no edits (the base pull the export was made from) gives label files byte-identical
to prepare_all_layer_singleclass.py run unchanged on the export, and the same images. If that fails, the method is
wrong and nothing else here counts. Synthetic fixture always; the real pull + v5 S1 export too when they are local.

    python -m unittest discover -s data_utils/tests
"""
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from data_utils import config  # noqa: E402
from data_utils import dataset_from_lv as D  # noqa: E402

EXPORT = "roboflow - 20990101 fixture"
# Roboflow export categories (id order differs from the pull's on purpose: classes are matched by name)
EXPORT_CATS = [{"id": 0, "name": "items-nfaR", "supercategory": "none"}, {"id": 1, "name": "EV", "supercategory": "items-nfaR"},
               {"id": 2, "name": "Low conf EV", "supercategory": "items-nfaR"}, {"id": 3, "name": "MV", "supercategory": "items-nfaR"},
               {"id": 4, "name": "VLDL", "supercategory": "items-nfaR"}, {"id": 5, "name": "non-EV", "supercategory": "items-nfaR"}]
PULL_CATS = [{"id": 1, "name": "EV"}, {"id": 2, "name": "non-EV"}, {"id": 3, "name": "MV"}, {"id": 4, "name": "Low conf EV"}]
# stem -> (export split, Roboflow split in the pull, w, h, [(class, polygon px)])
IMAGES = {
    "a1_png": ("train", "train", 1440, 1024, [("EV", [10, 20, 110, 20.5, 110, 120, 10, 120]),
                                             ("MV", [300.25, 300, 400, 300, 350, 410.75]),
                                             ("non-EV", [500, 500, 600, 500, 600, 600])]),
    "a2_png": ("train", "valid", 1440, 1024, [("Low conf EV", [1, 1, 50, 1, 50, 50, 1, 50]),
                                             ("EV", [700, 700, 800, 700, 800, 800])]),
    "a3_jpg": ("train", "train", 5760, 4092, []),                       # no labels: image only, no .txt
    "b1_png": ("valid", "valid", 1440, 1024, [("EV", [1439, 1023, 1500, 1023, 1500, 1100])]),   # clamped to 1
    "c1_png": ("test", "train", 2880, 2046, [("EV", [100, 100, 200, 100, 200, 200]),            # pull says train
                                            ("MV", [5, 5, 15, 5, 15, 15, 5, 15])]),
}
EXPORT_ONLY = {"x9_png": ("train", 1440, 1024, [("VLDL", [10, 10, 20, 10, 20, 20])])}           # not in the lv
LV_ONLY = {"a2dup_png": ("train", 1440, 1024, [("EV", [1, 1, 50, 1, 50, 50])]),                # DROPPED-duplicate
           "orphan_png": ("test", 1440, 1024, [("EV", [3, 3, 9, 3, 9, 9])])}                     # nowhere in the export


def hexname(stem):
    return hashlib.md5(stem.encode()).hexdigest()


def make_export(root):
    """<root>/training images/<EXPORT>/{train,valid,test}/ + split_assignment.csv, like make_balanced_split.py."""
    d = root / "training images" / EXPORT
    cid = {c["name"]: c["id"] for c in EXPORT_CATS}
    per = {s: {"images": [], "annotations": [], "categories": EXPORT_CATS} for s in ("train", "valid", "test")}
    rows = ["image,batch,montage,split,source_split"]
    items = [(k, v[0], v[2], v[3], v[4]) for k, v in IMAGES.items()] + [(k, *v) for k, v in EXPORT_ONLY.items()]
    for stem, split, w, h, objs in items:
        c = per[split]
        fn = f"{stem}.rf.{hexname(stem)}.jpg"
        iid = len(c["images"])
        c["images"].append({"id": iid, "license": 1, "file_name": fn, "height": h, "width": w,
                            "extra": {"name": stem.replace("_png", ".png")}})
        for cls, poly in objs:
            c["annotations"].append({"id": len(c["annotations"]), "image_id": iid, "category_id": cid[cls],
                                     "bbox": [0, 0, 1, 1], "area": 1, "segmentation": [poly], "iscrowd": 0})
        (d / split).mkdir(parents=True, exist_ok=True)
        (d / split / fn).write_bytes(b"\xff\xd8 fake jpeg " + stem.encode() * 50)
        rows.append(f"{stem},batch,,{split},{split}")
    rows.append("a2dup_png,batch,,DROPPED-duplicate,train")
    for s, c in per.items():
        (d / s / "_annotations.coco.json").write_text(json.dumps(c), encoding="utf-8")
    (d / "split_assignment.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return d


def make_pull():
    """The base pull's COCO (roboflow_pull_ids.py form): pull image ids, float coords, its own category ids."""
    cid = {c["name"]: c["id"] for c in PULL_CATS}
    images, anns = [], []
    items = [(k, v[1], v[2], v[3], v[4]) for k, v in IMAGES.items()] + [(k, *v) for k, v in LV_ONLY.items()]
    for i, (stem, rf_split, w, h, objs) in enumerate(items, 1):
        images.append({"id": i, "file_name": stem.replace("_png", ".png").replace("_jpg", ".jpg"), "width": w,
                       "height": h, "extra": {"name": stem, "split": rf_split, "roboflow_id": f"RF{i:018d}"}})
        for cls, poly in objs:
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cid[cls],
                         "segmentation": [[float(v) for v in poly]], "iscrowd": 0})
    return {"info": {"source": "fixture"}, "images": images, "annotations": anns, "categories": PULL_CATS}


def write_lv(parent, lv_id, coco, **version):
    d = parent / lv_id
    d.mkdir(parents=True)
    b = json.dumps(coco).encode("utf-8")
    (d / "labels.coco.json").write_bytes(b)
    (d / "version.json").write_text(json.dumps({"id": lv_id, "base": "roboflow_pull_20990101_0000",
                                                "labels_sha256": hashlib.sha256(b).hexdigest(), **version}),
                                    encoding="utf-8")
    return d


def run_prepare(cryoai_root, out_dir, scheme, source_name=EXPORT):
    """prepare_all_layer_singleclass.py exactly as a user runs it (an absolute --output-name lands in out_dir)."""
    env = dict(os.environ, CRYOEV_CRYOAI_ROOT=str(cryoai_root))
    r = subprocess.run([sys.executable, str(REPO / "data_utils" / "prepare_all_layer_singleclass.py"),
                        "--source-name", source_name, "--output-name", str(out_dir), "--class-scheme", scheme],
                       env=env, capture_output=True, text=True, cwd=REPO)
    if r.returncode:
        raise AssertionError(r.stdout[-2000:] + r.stderr[-2000:])
    return out_dir


def build(lv, export, name, out_root, scheme="ev", **kw):
    buf = io.StringIO()
    prov = D.build(str(lv), str(export), name, out_root, scheme, log=lambda *a: print(*a, file=buf), **kw)
    return prov, buf.getvalue()


def compare(prep_dir, lv_dir):
    """-> per split (byte-identical label files, files whose polygons differ, lv label files, lv images,
    images md5-differing, prepare images not in the lv build)."""
    out = {}
    for s in ("train", "val", "test"):
        li = {p.name for p in (lv_dir / "images" / s).iterdir()} if (lv_dir / "images" / s).is_dir() else set()
        pi = {p.name for p in (prep_dir / "images" / s).iterdir()} if (prep_dir / "images" / s).is_dir() else set()
        ll = sorted(p.name for p in (lv_dir / "labels" / s).glob("*.txt"))
        stems = {Path(n).stem for n in li}
        pl = {p.name for p in (prep_dir / "labels" / s).glob("*.txt") if Path(p.name).stem in stems}
        byte = sum((prep_dir / "labels" / s / n).read_bytes() == (lv_dir / "labels" / s / n).read_bytes()
                   for n in ll if n in pl)
        poly_diff = [n for n in sorted(set(ll) | pl) if n not in pl or n not in ll or
                     sorted((prep_dir / "labels" / s / n).read_text().splitlines()) !=
                     sorted((lv_dir / "labels" / s / n).read_text().splitlines())]
        img_diff = [n for n in li if n not in pi or
                    (prep_dir / "images" / s / n).read_bytes() != (lv_dir / "images" / s / n).read_bytes()]
        out[s] = {"byte": byte, "poly_diff": poly_diff, "files": len(ll), "images": len(li), "img_diff": img_diff,
                  "export_only": len(pi - li),
                  "lines": sum(len((lv_dir / "labels" / s / n).read_text().splitlines()) for n in ll)}
    return out


def yaml_body(d):
    """dataset.yaml without the lines that name the folder."""
    return [ln for ln in (d / "dataset.yaml").read_text(encoding="utf-8").splitlines()
            if not ln.startswith(("path:", "# YOLO dataset config for"))]


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="dfl_")
        self.root = Path(self.tmp.name)
        self.export = make_export(self.root)
        self.base = make_pull()
        self.lvs = self.root / "labels"
        self.base_lv = write_lv(self.lvs, "lv_20990101_0000", self.base, review_scope="reviewed_only")

    def tearDown(self):
        self.tmp.cleanup()


class TestKnownAnswer(Fixture):
    """The control: unedited base -> prepare's output, byte for byte."""

    def check_control(self, scheme):
        prep_dir = run_prepare(self.root, self.root / f"prep_{scheme}", scheme)
        prov, _ = build(self.base_lv, self.export, f"lv_{scheme}", self.root / "out", scheme)
        lv_dir = self.root / "out" / f"lv_{scheme}"
        cmp = compare(prep_dir, lv_dir)
        files = sum(c["files"] for c in cmp.values())
        self.assertGreaterEqual(files, 4, cmp)                                   # not a trivial pass
        for s, c in cmp.items():
            self.assertEqual(c["poly_diff"], [], (scheme, s))
            self.assertEqual(c["byte"], c["files"], (scheme, s))                 # byte-identical label files
            self.assertEqual(c["img_diff"], [], (scheme, s))
        self.assertEqual({s: c["export_only"] for s, c in cmp.items()}, {"train": 1, "val": 0, "test": 0})
        self.assertEqual(yaml_body(prep_dir), yaml_body(lv_dir))
        return prov, cmp

    def test_control_single_class(self):
        prov, cmp = self.check_control("ev")
        rec = prov["dataset_record"]
        self.assertEqual(rec["counts"]["train"]["images"], 3)
        self.assertEqual(rec["counts"]["train"]["instances"], 4)            # non-EV dropped in "ev"
        self.assertEqual(rec["class_counts"], {"train": {"EV": 4}, "valid": {"EV": 1}, "test": {"EV": 2}})

    def test_control_ev_nonev(self):
        prov, _ = self.check_control("ev_nonev")
        self.assertEqual(prov["dataset_record"]["class_counts"]["train"], {"EV": 4, "non-EV": 1})

    def test_split_from_export_not_from_pull(self):
        build(self.base_lv, self.export, "ds", self.root / "out")
        ds = self.root / "out" / "ds"
        self.assertTrue((ds / "images" / "test" / f"c1_png.rf.{hexname('c1_png')}.jpg").exists())   # pull: train
        self.assertTrue((ds / "images" / "train" / f"a2_png.rf.{hexname('a2_png')}.jpg").exists())  # pull: valid
        self.assertEqual((ds / D.IMAGE_LIST).read_text(encoding="utf-8"),
                         "a1_png\ttrain\na2_png\ttrain\na3_jpg\ttrain\nb1_png\tvalid\nc1_png\ttest\n")


class TestEdits(Fixture):
    def test_deleted_polygon_and_class_change_show_up(self):
        edited = copy.deepcopy(self.base)
        pid = {c["name"]: c["id"] for c in PULL_CATS}
        img = {im["extra"]["name"]: im["id"] for im in edited["images"]}
        anns = edited["annotations"]
        gone = next(a for a in anns if a["image_id"] == img["a1_png"] and a["category_id"] == pid["MV"])
        anns.remove(gone)
        flip = next(a for a in anns if a["image_id"] == img["c1_png"] and a["category_id"] == pid["EV"])
        flip["category_id"] = pid["non-EV"]
        lv = write_lv(self.lvs, "lv_20990101_0100", edited, review_scope="reviewed_only")
        build(self.base_lv, self.export, "base", self.root / "out", "ev_nonev")
        prov, _ = build(lv, self.export, "edited", self.root / "out", "ev_nonev")
        B, E = self.root / "out" / "base" / "labels", self.root / "out" / "edited" / "labels"
        a1, c1 = f"train/a1_png.rf.{hexname('a1_png')}.txt", f"test/c1_png.rf.{hexname('c1_png')}.txt"
        b_a1, e_a1 = (B / a1).read_text().splitlines(), (E / a1).read_text().splitlines()
        self.assertEqual(len(e_a1), len(b_a1) - 1)
        mv_line = "0 " + " ".join(f"{v:.6f}" for x, y in [(300.25, 300), (400, 300), (350, 410.75)]
                                  for v in (x / 1440, y / 1024))
        self.assertIn(mv_line, b_a1)
        self.assertNotIn(mv_line, e_a1)
        b_c1, e_c1 = (B / c1).read_text().splitlines(), (E / c1).read_text().splitlines()
        self.assertEqual(b_c1[0][:2], "0 ")
        self.assertEqual(e_c1[0], "1 " + b_c1[0][2:])                        # same outline, class 0 -> 1
        self.assertEqual(e_c1[1:], b_c1[1:])
        others = sorted(p.relative_to(B).as_posix() for p in B.rglob("*.txt"))
        for rel in others:
            if rel not in (a1, c1):
                self.assertEqual((B / rel).read_bytes(), (E / rel).read_bytes(), rel)
        rec = prov["dataset_record"]
        self.assertEqual(rec["class_counts"]["test"]["non-EV"], 1)
        self.assertEqual(rec["counts"]["train"]["instances"], 4)              # 5 in the base build


class TestReporting(Fixture):
    def test_images_missing_from_split_source_are_reported(self):
        prov, log = build(self.base_lv, self.export, "ds", self.root / "out")
        det, rec = prov["details"], prov["dataset_record"]
        self.assertEqual(sorted((m["image"], m["reason"]) for m in det["not_in_export"]),
                         [("a2dup_png", "split file: DROPPED-duplicate"), ("orphan_png", "not in the export")])
        self.assertEqual(rec["not_in_export"], 2)
        self.assertEqual(rec["export_only"], {"train": 1})
        self.assertIn("orphan_png", log)
        self.assertIn("a2dup_png", log)
        self.assertIn("2 lv image(s) not in the export", rec["note"])
        on_disk = json.loads((self.root / "out" / "ds" / D.PROVENANCE).read_text(encoding="utf-8"))
        self.assertEqual(on_disk["details"]["not_in_export"], det["not_in_export"])
        self.assertEqual(on_disk["dataset_record"]["id"], "ds")

    def test_provenance_hashes(self):
        prov, _ = build(self.base_lv, self.export, "ds", self.root / "out")
        rec, ds = prov["dataset_record"], self.root / "out" / "ds"
        lab = rec["labels"]
        self.assertEqual(lab["lv_labels_sha256"], hashlib.sha256((self.base_lv / "labels.coco.json").read_bytes()).hexdigest())
        sys.path.insert(0, str(REPO / "jobs"))
        dc = D._load("dataset_cache", REPO / "jobs" / "dataset_cache.py")
        md5s = {p.relative_to(ds).as_posix(): dc.C.md5_file(p) for d in ("images", "labels")
                for p in (ds / d).rglob("*") if p.is_file()}
        self.assertEqual(lab["dataset_sha256"], dc.content_sha256(md5s))
        self.assertEqual(lab["n_label_files"], sum(1 for k in md5s if k.startswith("labels/")))
        self.assertEqual(rec["images_sha256"], hashlib.sha256("".join(sorted(
            f"{s}\t{k}\n" for k, s in [("a1_png", "train"), ("a2_png", "train"), ("a3_jpg", "train"),
                                       ("b1_png", "valid"), ("c1_png", "test")])).encode()).hexdigest())
        self.assertEqual(rec["split"]["sha256"], hashlib.sha256((self.export / "split_assignment.csv").read_bytes()).hexdigest())
        self.assertEqual(rec["counts_trained"], {"train": 3, "valid": 1, "test": 1})

    def test_list_text_matches_importer(self):
        sys.path.insert(0, str(REPO / "registry"))
        importers = D._load("importers", REPO / "registry" / "importers.py")
        pairs = {("b", "test"), ("a", "train"), ("c", "valid"), ("z", "train")}
        self.assertEqual(D.list_text(pairs), importers.list_text(pairs))


class TestRefusals(Fixture):
    def test_existing_output_is_never_overwritten(self):
        (self.root / "out" / "ds").mkdir(parents=True)
        with self.assertRaisesRegex(D.BuildError, "never overwritten"):
            build(self.base_lv, self.export, "ds", self.root / "out")

    def test_superseded_lv(self):
        (self.base_lv / "SUPERSEDED.md").write_text("old", encoding="utf-8")
        with self.assertRaisesRegex(D.BuildError, "superseded"):
            build(self.base_lv, self.export, "ds", self.root / "out")
        self.assertFalse((self.root / "out" / "ds").exists())
        prov, _ = build(self.base_lv, self.export, "ds", self.root / "out", allow_superseded=True)
        self.assertIn("superseded", prov["dataset_record"]["note"])

    def test_labels_changed_after_version(self):
        p = self.base_lv / "labels.coco.json"
        p.write_bytes(p.read_bytes() + b" ")
        with self.assertRaisesRegex(D.BuildError, "changed after the version"):
            build(self.base_lv, self.export, "ds", self.root / "out")

    def test_split_file_disagrees_with_folders(self):
        f = self.export / "split_assignment.csv"
        f.write_text(f.read_text(encoding="utf-8").replace("c1_png,batch,,test", "c1_png,batch,,train"), encoding="utf-8")
        with self.assertRaisesRegex(D.BuildError, "c1_png"):
            build(self.base_lv, self.export, "ds", self.root / "out")

    def test_failed_build_leaves_no_folder(self):
        bad = copy.deepcopy(self.base)
        bad["categories"].append({"id": 9, "name": "Mystery"})
        lv = write_lv(self.lvs, "lv_20990101_0200", bad)
        with self.assertRaisesRegex(D.BuildError, "Mystery"):
            build(lv, self.export, "ds", self.root / "out")
        self.assertFalse((self.root / "out" / "ds").exists())


class TestRegistry(Fixture):
    def test_register_into_temp_registry(self):
        reg_root = self.root / "registry"
        prov, _ = build(self.base_lv, self.export, "lv_ds_fixture", self.root / "out", "ev_nonev")
        line = D.register(prov, self.root / "out" / "lv_ds_fixture", reg_root)
        reg = D._load("registry", REPO / "registry" / "registry.py")
        cur = reg.current("dataset", reg_root)["lv_ds_fixture"]
        self.assertEqual(cur["source"], {"kind": "label_version", "id": "lv_20990101_0000",
                                         "path": str(self.base_lv)})
        self.assertEqual(cur["label_version"], "lv_20990101_0000")
        self.assertEqual(cur["status"], "frozen")
        self.assertEqual(cur["classes"], ["EV", "non-EV"])
        self.assertEqual(line["id"], "lv_ds_fixture")
        self.assertEqual((reg_root / "lists" / "lv_ds_fixture.tsv").read_bytes(),
                         (self.root / "out" / "lv_ds_fixture" / D.IMAGE_LIST).read_bytes())
        with self.assertRaisesRegex(D.BuildError, "already has"):
            D.register(prov, self.root / "out" / "lv_ds_fixture", reg_root)
        self.assertEqual(len((reg_root / "dataset.jsonl").read_text(encoding="utf-8").splitlines()), 1)

    def test_cli_without_register_writes_no_registry(self):
        reg_root = self.root / "registry"
        env = dict(os.environ, CRYOEV_REGISTRY=str(reg_root), CRYOEV_LAPTOP_ROOT=str(self.root))
        r = subprocess.run([sys.executable, str(REPO / "data_utils" / "dataset_from_lv.py"), "--lv", "lv_20990101_0000",
                            "--split-from", str(self.export), "--output-name", "ds", "--output-root",
                            str(self.root / "out")], env=env, capture_output=True, text=True, cwd=REPO)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("--register would append", r.stdout)
        rec = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(rec["schema"], "dataset/1")
        self.assertFalse(reg_root.exists())
        r = subprocess.run([sys.executable, str(REPO / "data_utils" / "dataset_from_lv.py"), "--lv", "lv_20990101_0000",
                            "--split-from", str(self.export), "--output-name", "ds2", "--output-root",
                            str(self.root / "out"), "--register"], env=env, capture_output=True, text=True, cwd=REPO)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((reg_root / "dataset.jsonl").is_file())


class TestBundleAcceptsIt(Fixture):
    def test_make_job_bundle_and_dataset_cache(self):
        """training/make_job_bundle.py takes the folder unchanged; its pack key is the recorded dataset_sha256."""
        prov, _ = build(self.base_lv, self.export, "lvds", self.root / "out")
        out = self.root / "bundles"
        r = subprocess.run([sys.executable, "training/make_job_bundle.py", "--allow-dirty", "--dataset-dir",
                            str(self.root / "out" / "lvds"), "--out", str(out), "--", "--single-run", "--model-size",
                            "n", "--imgsz", "320"], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        zp = next(out.glob("lvds__job_*.zip"))
        with zipfile.ZipFile(zp) as z:
            job = json.loads(z.read("lvds/JOB.json"))
        self.assertEqual(job["dataset_sha256"], prov["dataset_record"]["labels"]["dataset_sha256"])


# ---------------------------------------------------------------- real data (only where it is local)
REAL_EXPORT_NAME = "roboflow - 20261028 cryoai v5 S1"
REAL_PULL_NAME = "roboflow_pull_20261029_1811"


def _real():
    lap, cai = config.laptop_root(required=False), config.cryoai_root(required=False)
    if not (lap and cai):
        return None
    pull = lap / REAL_PULL_NAME / "current.coco.json"
    exp = cai / "training images" / REAL_EXPORT_NAME
    return (cai, pull, exp) if pull.is_file() and (exp / "train" / "_annotations.coco.json").is_file() else None


@unittest.skipUnless(_real(), "real pull / v5 S1 export not local")
class TestKnownAnswerRealData(unittest.TestCase):
    """The pull the reviewed lvs are based on vs prepare on the export it was compared to (diff_vs_export.csv).
    No unedited lv exists, so the control lv is built in a temp folder from the base pull, unchanged.
    Reads only; everything is written under a temp folder."""

    def test_base_pull_equals_prepare_on_export(self):
        cai, pull, exp = _real()
        with tempfile.TemporaryDirectory(prefix="dfl_") as t:
            t = Path(t)
            lv = t / "lv_control"
            lv.mkdir()
            (lv / "labels.coco.json").write_bytes(pull.read_bytes())
            prep_dir = run_prepare(cai, t / "prep", "ev_nonev", REAL_EXPORT_NAME)
            prov, _ = build(lv, exp, "fromlv", t, "ev_nonev")
            cmp = compare(prep_dir, t / "fromlv")
            for s, c in cmp.items():
                self.assertEqual(c["poly_diff"], [], s)
                self.assertEqual(c["byte"], c["files"], s)
                self.assertEqual(c["img_diff"], [], s)
            self.assertEqual({s: c["images"] for s, c in cmp.items()}, {"train": 103, "val": 23, "test": 34})
            self.assertEqual(sum(c["files"] for c in cmp.values()), 149)
            self.assertEqual(sum(c["lines"] for c in cmp.values()), 3528)
            self.assertEqual(prov["dataset_record"]["not_in_export"], 10)
            self.assertTrue(all(m["reason"] == "split file: DROPPED-duplicate" for m in prov["details"]["not_in_export"]))
            self.assertEqual(yaml_body(prep_dir), yaml_body(t / "fromlv"))


if __name__ == "__main__":
    unittest.main()
