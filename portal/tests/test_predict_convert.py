"""
Predict: raw conversion and the one-file-per-request upload (portal/modules/predict.py).

  * Gating: raw upload for admin + advanced, raw session pick for admin only (basic: 403 everywhere).
  * Recipes: the processing repo's session recipes that jobs/convert_raw.py's static check accepts; default + warning.
  * Lifecycle with a fake converter (written to the temp folder, run through the jobs test hooks' convert_cmd):
    queued -> running -> done / failed, one at a time, convert JSON + warnings in status.json, the session folder
    never written (tree snapshot), GPU only when asked and free.
  * The converted folder as a Predict source: owner and admin only; sample groups, the nm/px tick, the 200-tile cap.
  * Upload API: one file (or piece) per request, resume, refusals (caps, disk, content, fields, owner, Origin), a
    session never finished (no create) swept with its disk hold; the no-JavaScript form (small sets only).
  * Escaping, and the pages in a browser (Playwright, Edge channel; 390 and 1280 px) with the JS upload driven for
    real. Screenshots go to $CJ_SHOT_DIR when set (else a temp folder, removed afterwards).
  * Known answer (needs CJ_PROCESSING_REPO and CJ_CONVERT_CONTROL_DIR, as jobs/tests/test_convert_raw.py's Control):
    the raw montage uploaded through the portal (and, a second time, picked from a temp copy of its session folder)
    converts to tiles md5-identical to the delivered 3d tiles. The control folder and the processing repo are only
    read.

Temp roster / registry / predict root / scan root / processing repo stub only; never the real ones.

  python -m unittest portal.tests.test_predict_convert -v
"""
import csv
import hashlib
import html
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import quote, urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import app as APP  # noqa: E402
from portal import auth as A  # noqa: E402
from portal import predict_sources as PSRC  # noqa: E402
from portal.modules import predict as P  # noqa: E402

try:
    from starlette.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None
try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

REPO = T.REPO
GOLDEN = REPO / "tests" / "fixtures" / "cj_golden" / "predict" / "ps_20261105_abc123"
MODEL = "v5_evnonev_y8m_20261029"
FAKE_PRED = [sys.executable, str(REPO / "jobs" / "tests" / "fake_predictor.py")]
XSS = '<img src=x onerror=alert(1)>'
ORIGIN = "http://127.0.0.1:8771"
ENV_KEYS = ("CJ_PREDICT_SOURCES", "CJ_PREDICT_ROOT", "CJ_REVIEW_ROOT", "CJ_PREDICT_SCAN_ROOTS", "CJ_PROCESSING_REPO",
            "CRYOEV_LAPTOP_ROOT", "CRYOEV_CRYOAI_ROOT", "CRYOEV_PROCESSING_ROOT", "CRYOEV_REGISTRY",
            "CRYOEV_JOBS_TESTING", "CRYOEV_JOBS_TEST_HOOKS", "CRYOEV_JOBS_TEST_HOOKS_FILE", "FAKE_PRED_FROM",
            "FAKE_PRED_DELAY", "FAKE_CONVERT_MODE", "FAKE_CONVERT_N", "FAKE_CONVERT_GATE", "FAKE_CONVERT_ARGV",
            "FAKE_CONVERT_WARN", "CJ_RAW_ROOT", "CJ_RAW_KEEP_DAYS")
HOOKS_OK = {"gpu": {"total_mb": 4096, "apps": []}, "predict_cmd": FAKE_PRED}
CID_RE = re.compile(r"^/predict/convert/((\d{8}(?:-\d+)?)_[0-9a-f]{6})/$")

FAKE_CONVERT = r'''
"""Stand-in for `python -m jobs.convert_raw` (same CLI, same stdout JSON). Env: FAKE_CONVERT_MODE ok|refuse|crash,
FAKE_CONVERT_N tiles (3), FAKE_CONVERT_GATE (wait until this file exists), FAKE_CONVERT_ARGV (append argv as JSON),
FAKE_CONVERT_WARN (one warning), FAKE_GOLDEN (folder of 0001-0003.png)."""
import argparse, csv, json, os, shutil, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inputs", nargs="+", required=True)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--recipe", default="default")
ap.add_argument("--sample-key")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--processing-repo")
a = ap.parse_args()
if os.environ.get("FAKE_CONVERT_ARGV"):
    with open(os.environ["FAKE_CONVERT_ARGV"], "a", encoding="utf-8") as f:
        f.write(json.dumps({"argv": sys.argv[1:], "env_repo": os.environ.get("CJ_PROCESSING_REPO")}) + "\n")
gate = os.environ.get("FAKE_CONVERT_GATE")
t0 = time.time()
while gate and not Path(gate).exists() and time.time() - t0 < 120:
    time.sleep(0.05)
mode = os.environ.get("FAKE_CONVERT_MODE", "ok")
if mode == "crash":
    print("Traceback (most recent call last):\n  boom", file=sys.stderr)
    sys.exit(1)
warn = os.environ.get("FAKE_CONVERT_WARN") or "grid1_1: 7.362 A/px is not a sampling of recipe default's session"
if mode == "refuse":
    print(json.dumps({"ok": False, "kind": "refused", "warnings": [warn],
                      "error": "grid1-pos1.mrc.mdoc: no finite, positive PixelSpacing " + os.environ.get("FAKE_XSS", "")}))
    sys.exit(2)
if a.out.exists() and any(a.out.iterdir()):
    print(json.dumps({"ok": False, "kind": "refused", "error": "output folder is not empty", "warnings": []}))
    sys.exit(2)
labels = {}
if a.sample_key:
    with open(a.sample_key, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            labels[int(r["#"])] = r["Label"]
a.out.mkdir(parents=True, exist_ok=True)
gold = Path(os.environ["FAKE_GOLDEN"])
n = int(os.environ.get("FAKE_CONVERT_N") or 3)
rows, tiles = [], []
for i in range(n):
    g = 1 + i % 2
    lab = labels.get(g, f"Grid {g}")
    name = f"g{g}_t__grid{g}_1_r0c{i}_t00__denoised_nobar_14p72Apx.png"
    shutil.copyfile(gold / f"000{1 + i % 3}.png", a.out / name)
    rows.append({"output_name": name, "source_file": f"grid{g}_1_r0c{i}_t00.jpg", "grid": g, "sample_label": lab,
                 "montage": f"grid{g}_1", "source_size": "800x800", "output_size": "400x400", "factor": "2.0000",
                 "source_a_px": "7.3620", "output_a_px": "14.7240", "nm_per_px": "1.47240",
                 "compute": "GPU" if a.gpu else "CPU", "md5": ""})
    tiles.append({"file": name, "nm_per_px": 1.4724, "source_file": rows[-1]["source_file"],
                  "montage": rows[-1]["montage"], "sample_label": lab, "size": "400x400"})
with open(a.out / "MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("convert_raw: fake pipeline progress", file=sys.stderr)
print(json.dumps({"ok": True, "out_dir": str(a.out), "recipe": "20261012" if a.recipe == "default" else a.recipe,
                  "recipe_defaulted": a.recipe == "default", "entry_point": "fake", "compute": "CPU",
                  "device": "CPU (NumPy)", "montages": sorted({r["montage"] for r in rows}), "a_px": {},
                  "n_tiles": n, "tiles": tiles, "nm_per_px": [1.4724], "warnings": [warn]}))
'''

RECIPE_OK = '''"""A session config (stub)."""
import sys
from cryoem_core import DatasetConfig
from cryoem_core.discovery import FlatGridPositionMrcLayout
DATASET = DatasetConfig(layout=FlatGridPositionMrcLayout())
'''
RECIPE_OLD = '''"""An old full-script processor."""
print("would run at import")
DATASET = None
'''


def mrc_bytes(n=4096) -> bytes:
    """Enough of an MRC2014 file for the content check: nx ny nz > 0 and 'MAP ' at byte 208."""
    import struct
    head = bytearray(1024)
    head[0:12] = struct.pack("<3i", 64, 64, 1)
    head[208:212] = b"MAP "
    return bytes(head) + b"\x01" * (n - 1024)


MDOC = ("PixelSpacing = 7.362\nImageFile = grid1-pos1\nImageSize = 64 64\nMontage = 1\n\n"
        "[ZValue = 0]\nPieceCoordinates = 0 0 0\n")
SAMPLE_KEY = f"#,Label\n1,EV prep A\n2,{XSS}\n"


def md5(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def tree(d: Path) -> dict:
    """{relative path: (size, mtime_ns, md5)} of every file: proof a folder was not written."""
    out = {}
    for p in sorted(Path(d).rglob("*")):
        if p.is_file():
            st = p.stat()
            out[p.relative_to(d).as_posix()] = (st.st_size, st.st_mtime_ns, md5(p))
    return out


def png16() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("I;16", (64, 64), 1000).save(buf, "PNG")
    return buf.getvalue()


def tile_bytes(i) -> bytes:
    return (GOLDEN / "images" / f"000{i}.png").read_bytes()


def wait_for(fn, timeout=60, every=0.1, what="condition"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(every)
    raise AssertionError(f"timed out waiting for {what}")


class World:
    """The temp world every class here uses: env, roster (admin tester, adv, adv2, bas), registry, predict root, scan
    root with raw session folders, a stub processing repo and the fake converter."""

    def __init__(self, real_repo=None, raw_root=None):
        self.saved = {k: os.environ.get(k) for k in ENV_KEYS}
        self.env = T.Env()
        tmp = self.tmp = self.env.tmp
        self.keys = {"admin": self.env.key, "advanced": A.add_key(self.env.roster, "adv", "advanced"),
                     "adv2": A.add_key(self.env.roster, "adv2", "advanced"),
                     "basic": A.add_key(self.env.roster, "bas", "basic")}
        self.lap = tmp / "lap"
        self.root, self.review = self.lap / "predict", self.lap / "review"
        (tmp / "cai" / "training outputs").mkdir(parents=True)
        (tmp / "cai" / "training images").mkdir()
        self.env.write_live(T.live_doc())                     # the admin's bar reads it (a 404 would repeat)
        (self.env.registry / "model.jsonl").write_text(json.dumps(
            {"schema": "model/1", "id": MODEL, "status": "production", "logged": "2026-11-05T00:00:00"}) + "\n",
            encoding="utf-8")
        # the stub processing repo (what convert_raw.processing_repo checks + recipes for the static check)
        self.repo = tmp / "repo"
        (self.repo / "cryoem_core").mkdir(parents=True)
        (self.repo / "cryoem_core" / "__init__.py").write_text("", encoding="utf-8")
        (self.repo / "export_detector_tiles.py").write_text("", encoding="utf-8")
        (self.repo / "datasets").mkdir()
        for name in ("20261012", "20261027", "20261101", "20260827_2"):
            (self.repo / "datasets" / f"process_{name}.py").write_text(RECIPE_OK, encoding="utf-8")
        (self.repo / "datasets" / "process_old.py").write_text(RECIPE_OLD, encoding="utf-8")
        # the scan root: 20261101 raw (MRC + atlas + sample key), 20261103 raw (idoc), 20261027 processed (3d)
        self.scan = tmp / "scan"
        s1 = self.scan / "20261101" / "2. Raw images"
        s1.mkdir(parents=True)
        (s1 / "grid1-pos1.mrc").write_bytes(mrc_bytes())
        (s1 / "grid1-pos1.mrc.mdoc").write_text(MDOC, encoding="utf-8")
        (s1 / "grid1-atlas.mrc").write_bytes(mrc_bytes())
        (s1 / "grid1-atlas.mrc.mdoc").write_text(MDOC, encoding="utf-8")
        (s1 / "orphan.mrc.mdoc").write_text(MDOC, encoding="utf-8")            # no .mrc beside it: not listed
        (self.scan / "20261101" / "1. Notes + reports").mkdir()
        (self.scan / "20261101" / "1. Notes + reports" / "sample_key.csv").write_text(SAMPLE_KEY, encoding="utf-8")
        (self.scan / "20261103").mkdir()
        (self.scan / "20261103" / "grid3-1.idoc").write_text(MDOC.replace("[ZValue = 0]", "[Image = grid3-10000.tif]"),
                                                             encoding="utf-8")
        # 20261104: a raw session whose idoc names a file outside its folder (must never reach the converter)
        (self.scan / "20261104").mkdir()
        (self.scan / "20261104" / "grid4-1.idoc").write_text(
            MDOC.replace("[ZValue = 0]", "[Image = ..\\..\\secret.tif]"), encoding="utf-8")
        d3 = self.scan / "20261027" / "3. Processed images" / "3d. Tiles no scalebar - 14.7 A-px PNG"
        d3.mkdir(parents=True)
        shutil.copyfile(GOLDEN / "images" / "0001.png", d3 / "t1.png")
        (d3 / "MANIFEST.csv").write_text("output_name,sample_label,nm_per_px\nt1.png,S,1.4724\n", encoding="utf-8")
        # the fake converter
        self.fake = tmp / "fake_convert.py"
        self.fake.write_text(FAKE_CONVERT, encoding="utf-8")
        self.argv_log = tmp / "convert_argv.jsonl"
        self.gate = tmp / "gate"
        os.environ.update(CJ_PREDICT_ROOT=str(self.root), CJ_REVIEW_ROOT=str(self.review),
                          CJ_PREDICT_SCAN_ROOTS=str(self.scan), CJ_PROCESSING_REPO=str(real_repo or self.repo),
                          CRYOEV_LAPTOP_ROOT=str(self.lap), CRYOEV_CRYOAI_ROOT=str(tmp / "cai"),
                          CRYOEV_PROCESSING_ROOT=str(tmp / "no_processing"), FAKE_GOLDEN=str(GOLDEN / "images"),
                          FAKE_PRED_FROM=str(GOLDEN / "p" / MODEL / "predictions"),
                          FAKE_CONVERT_ARGV=str(self.argv_log))
        self.raw_base = Path(raw_root) if raw_root else self.root       # CJ_RAW_ROOT unset = the predict root
        self.rawtiles = self.raw_base / "raw_tiles"
        os.environ.pop("CJ_RAW_ROOT", None)
        if raw_root:
            os.environ["CJ_RAW_ROOT"] = str(raw_root)
        for k in ("CJ_PREDICT_SOURCES", "CRYOEV_REGISTRY", "FAKE_PRED_DELAY", "FAKE_CONVERT_MODE", "FAKE_CONVERT_N",
                  "FAKE_CONVERT_GATE", "FAKE_CONVERT_WARN", "CRYOEV_JOBS_TEST_HOOKS_FILE"):
            os.environ.pop(k, None)
        if real_repo:
            for k in ("CRYOEV_JOBS_TESTING", "CRYOEV_JOBS_TEST_HOOKS"):
                os.environ.pop(k, None)
        else:
            self.hooks(HOOKS_OK)
        man = json.loads((REPO / "portal" / "modules.json").read_text(encoding="utf-8"))
        for m in man["modules"]:
            if m["id"] == "predict":
                m.update(route="/predict/", status="on")
        self.manifest = tmp / "modules.json"
        self.manifest.write_text(json.dumps(man), encoding="utf-8")
        self.cfg = self.env.config(manifest=self.manifest, review_root=self.review)
        PSRC.clear_cache()

    def hooks(self, h):
        os.environ["CRYOEV_JOBS_TESTING"] = "1"
        os.environ["CRYOEV_JOBS_TEST_HOOKS"] = json.dumps({**h, "convert_cmd": [sys.executable, str(self.fake)]})

    def settle(self, timeout=60):
        """Wait until no conversion is queued or running (the watcher thread done), so nothing outlives the env."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            busy = [1 for _, _, st in P.all_conversions(None) if st.get("state") in ("queued", "running")]
            with P._WATCH_LOCK:
                watching = bool(P._WATCH)
            if not busy and not watching:
                return
            self.gate.touch()
            time.sleep(0.2)

    def close(self):
        self.settle()
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        PSRC.clear_cache()
        time.sleep(0.3)
        self.env.cleanup()


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TestClient is None:
            raise unittest.SkipTest("starlette TestClient (httpx) not installed")
        cls.w = World()
        cls.tc = TestClient(APP.create_app(cls.w.cfg), base_url=ORIGIN, follow_redirects=False)
        cls.tc.__enter__()
        cls.tok = {}
        for who, key in cls.w.keys.items():
            r = cls.tc.get(f"/login?k={key}")
            assert r.status_code == 303, r.status_code
            cls.tok[who] = r.headers["set-cookie"].split(";", 1)[0].split("=", 1)[1]
        cls.tc.cookies.clear()

    @classmethod
    def tearDownClass(cls):
        cls.tc.__exit__(None, None, None)
        cls.w.close()

    def setUp(self):
        PSRC.clear_cache()
        P._CREATES.clear()
        P._RECIPES.clear()
        self.w.hooks(HOOKS_OK)
        for k in ("FAKE_CONVERT_MODE", "FAKE_CONVERT_N", "FAKE_CONVERT_GATE", "FAKE_CONVERT_WARN"):
            os.environ.pop(k, None)
        self.sets0 = self.sets()                       # the classes share one predict root: compare with this

    def tearDown(self):
        self.w.settle()
        self.tc.cookies.clear()
        with P._UP_RES_LOCK:
            leftover = list(P._UP_RES)
        tmp = self.w.root / P.PORTAL_TMP
        for p in (list(tmp.iterdir()) if tmp.exists() else []):
            if P.UPLOAD_RE.match(p.name):
                P.drop_upload(p, p.name)
        self.assertEqual(P._RESERVED, {}, f"disk holds left: {P._RESERVED} (uploads {leftover})")

    # ---- http
    def h(self, who, origin=True, **extra):
        out = {"Cookie": f"{A.COOKIE}={self.tok[who]}", **extra}
        if origin:
            out["Origin"] = ORIGIN
        return out

    def get(self, path, who="admin"):
        return self.tc.get(path, headers=self.h(who, origin=False))

    def page(self, path, who="admin", status=200) -> str:
        r = self.get(path, who)
        self.assertEqual(r.status_code, status, r.text[:600])
        return r.text

    def form(self, path, data, who="admin", origin=True):
        return self.tc.post(path, content=urlencode(data, doseq=True).encode(),
                            headers=self.h(who, origin, **{"Content-Type": "application/x-www-form-urlencoded"}))

    def jpost(self, path, doc, who="admin", origin=True):
        return self.tc.post(path, content=json.dumps(doc).encode(),
                            headers=self.h(who, origin, **{"Content-Type": "application/json"}))

    def piece(self, uid, name, offset, data, who="admin"):
        return self.tc.post(f"/predict/api/upload/{uid}/file?name={quote(name)}&offset={offset}", content=data,
                            headers=self.h(who, **{"Content-Type": "application/octet-stream"}))

    def upload(self, who, kind, fields, files, chunk=None, create=True, expect=200):
        """The page's script, request by request. -> (start response json, create response or None)."""
        r = self.jpost("/predict/api/upload", {"kind": kind, "fields": fields,
                                               "files": [{"name": n, "size": len(b)} for n, b in files]}, who)
        self.assertEqual(r.status_code, expect, r.text[:600])
        st = r.json()
        if expect != 200:
            return st, None
        step = chunk or st["chunk"]
        for name, data in files:
            off = 0
            while off < len(data):
                part = data[off:off + step]
                rr = self.piece(st["upload"], name, off, part, who)
                self.assertEqual(rr.status_code, 200, rr.text[:600])
                off += len(part)
        if not create:
            return st, None
        return st, self.jpost(f"/predict/api/upload/{st['upload']}/create", {}, who)

    # ---- helpers
    def follow(self, loc, who="advanced", timeout=120) -> str:
        """A set is made in the background: /predict/preparing/<id>/ -> /predict/<ps>/ once it is done."""
        if not loc.startswith("/predict/preparing/"):
            return loc
        tid = loc.rstrip("/").split("/")[-1]
        wait_for(lambda: self.get(f"/predict/api/preparing/{tid}", who).json().get("state") in ("done", "failed"),
                 timeout, what="the set being prepared")
        j = self.get(f"/predict/api/preparing/{tid}", who).json()
        self.assertEqual(j["state"], "done", j)
        return j["href"]

    def sets(self) -> list:
        return sorted(p.name for p in self.w.root.iterdir() if p.name.startswith("ps_")) if self.w.root.exists() \
            else []

    def convs(self) -> list:
        d = self.w.root / P.RAW_TILES
        return sorted(p.name for p in d.iterdir() if P.RAW_ID_RE.match(p.name)) if d.exists() else []

    def status(self, cid) -> dict:
        return json.loads((self.w.root / P.RAW_TILES / cid / "status.json").read_text(encoding="utf-8"))

    def wait_conv(self, cid, states=("done", "failed"), timeout=60) -> dict:
        def check():
            try:
                st = self.status(cid)
            except (OSError, ValueError):
                return None
            return st if st.get("state") in states else None
        return wait_for(check, timeout, what=f"{cid} in {states}")

    def raw_key(self, folder="20261101"):
        return next(s["key"] for s in P.raw_sessions(None) if s["folder"] == folder)

    def pick_session(self, who="admin", **kw):
        key = self.raw_key()
        data = {"montages": ["2. Raw images/grid1-pos1.mrc.mdoc"], "recipe": "default"}
        data.update(kw)
        return self.form(f"/predict/new/raw/sessions/{key}/", {k: v for k, v in data.items() if v is not None}, who)

    def started(self, r) -> str:
        self.assertEqual(r.status_code, 303, r.text[:800])
        m = CID_RE.match(r.headers["location"])
        self.assertIsNotNone(m, r.headers["location"])
        return m.group(1)

    def argv_lines(self) -> list:
        try:
            return [json.loads(ln) for ln in self.w.argv_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except OSError:
            return []

    def raw_files(self, stem="grid1-pos1", key=True):
        out = [(f"{stem}.mrc", mrc_bytes()), (f"{stem}.mrc.mdoc", MDOC.encode())]
        if key:
            out.append(("sample_key.csv", SAMPLE_KEY.encode()))
        return out

    def converted(self, who="advanced", session="20260827-2", recipe="20260827_2") -> str:
        """An upload conversion by `who`, run to done. -> its id."""
        st, r = self.upload(who, "raw", {"session": session, "recipe": recipe}, self.raw_files())
        self.assertEqual(r.status_code, 200, r.text[:600])
        cid = CID_RE.match(r.json()["href"]).group(1)
        self.assertEqual(self.wait_conv(cid)["state"], "done", self.status(cid))
        return cid

    def src_key(self, cid) -> str:
        return PSRC.source_key(self.w.root / P.RAW_TILES / cid / P.CONVERT_TILES)


class Gating(Base):
    def test_basic_gets_403_everywhere_raw(self):
        key = self.raw_key()
        for method, path in (("GET", "/predict/new/raw/"), ("POST", "/predict/new/raw/"),
                             ("GET", "/predict/new/raw/sessions/"), ("GET", f"/predict/new/raw/sessions/{key}/"),
                             ("POST", f"/predict/new/raw/sessions/{key}/")):
            with self.subTest(method=method, path=path):
                r = self.get(path, "basic") if method == "GET" else self.form(path, {"recipe": "default"}, "basic")
                self.assertEqual(r.status_code, 403, r.text[:300])
        st, _ = self.upload("basic", "raw", {"session": "20261101"}, self.raw_files(), expect=403)
        self.assertIn("advanced users and the admin", st["error"])
        self.assertTrue(st["dropped"])
        page = self.page("/predict/new/", "basic")
        self.assertNotIn("/predict/new/raw/", page)
        self.assertEqual(self.convs(), [])

    def test_advanced_uploads_but_does_not_pick_sessions(self):
        page = self.page("/predict/new/", "advanced")
        self.assertIn('href="/predict/new/raw/"', page)
        self.assertNotIn('href="/predict/new/raw/sessions/"', page)
        self.assertIn('data-cj-upload="raw"', self.page("/predict/new/raw/", "advanced"))
        key = self.raw_key()
        for path in ("/predict/new/raw/sessions/", f"/predict/new/raw/sessions/{key}/"):
            self.assertEqual(self.get(path, "advanced").status_code, 403)
        self.assertEqual(self.pick_session("advanced").status_code, 403)
        self.assertEqual(self.convs(), [])

    def test_admin_sees_raw_sessions_without_3d_only(self):
        page = self.page("/predict/new/raw/sessions/")
        self.assertIn("Session 1 Nov 2026", page)
        self.assertIn("Session 3 Nov 2026", page)
        self.assertNotIn("Session 27 Oct 2026", page)                     # it has 3d tiles: a normal source
        one = self.page(f"/predict/new/raw/sessions/{self.raw_key()}/")
        self.assertIn('value="2. Raw images/grid1-pos1.mrc.mdoc" checked', one)
        self.assertIn('value="2. Raw images/grid1-atlas.mrc.mdoc">', one)      # listed, not ticked
        self.assertNotIn("orphan", one)                                        # no .mrc beside it
        self.assertIn("1. Notes + reports/sample_key.csv", one)
        self.assertIn("This session has its own recipe: <b>20261101</b>", one)
        idoc = self.page(f"/predict/new/raw/sessions/{self.raw_key('20261103')}/")
        self.assertIn("grid3-1.idoc", idoc)
        self.assertIn("No sample_key.csv found", idoc)
        self.assertEqual(self.get("/predict/new/raw/sessions/0123456789/").status_code, 404)

    def test_origin_needed_and_not_set_up(self):
        r = self.jpost("/predict/api/upload", {"kind": "tiles"}, "admin", origin=False)
        self.assertEqual(r.status_code, 403)
        with mock.patch.dict(os.environ, {"CJ_PROCESSING_REPO": ""}):
            page = self.page("/predict/new/raw/", "advanced")
            self.assertIn("Raw conversion is not set up on this server", page)
            self.assertNotIn("CJ_PROCESSING_REPO", page)                        # the setting: admin only
            self.assertIn("CJ_PROCESSING_REPO", self.page("/predict/new/raw/", "admin"))
            st, _ = self.upload("advanced", "raw", {"session": "20261101"}, self.raw_files(), expect=503)
            self.assertIn("not set up", st["error"])


class Recipes(Base):
    def test_valid_recipes_only_default_first_and_warned(self):
        self.assertEqual(P.recipes(self.w.repo), ["20261101", "20261027", "20261012", "20260827_2"])
        page = self.page("/predict/new/raw/", "advanced")
        opts = re.findall(r'<option value="([^"]+)"', page)
        self.assertEqual(opts, ["default", "20261101", "20261027", "20261012", "20260827_2"])
        self.assertIn('<option value="default" selected>default (20261012, the lab’s template)', page)
        self.assertIn("“default” is the lab’s template (20261012)", page)
        self.assertNotIn("old", opts)

    def test_session_id_and_recipe_checked_before_any_byte(self):
        for fields, why in (({"session": "2026-11-01"}, "Session: the imaging date"),
                            ({"session": "20261340"}, "Session: the imaging date"),
                            ({"session": "20261101 x"}, "Session: the imaging date"),
                            ({"session": "20261101", "recipe": "old"}, "Recipe: pick one from the list"),
                            ({"session": "20261101", "recipe": XSS}, "Recipe: pick one from the list")):
            with self.subTest(fields=fields):
                st, _ = self.upload("advanced", "raw", fields, self.raw_files(), expect=400)
                self.assertIn(why, st["error"])
                self.assertNotIn(XSS, json.dumps(st))
        st, _ = self.upload("advanced", "raw", {"session": "20260827-2"}, self.raw_files(), create=False)
        self.assertTrue(P.UPLOAD_RE.match(st["upload"]))
        self.assertEqual(self.convs(), [])


class Lifecycle(Base):
    def test_session_pick_queue_run_done_and_folder_untouched(self):
        before = tree(self.w.scan / "20261101")
        os.environ["FAKE_CONVERT_GATE"] = str(self.w.gate)
        self.w.gate.unlink(missing_ok=True)
        first = self.started(self.pick_session())
        self.assertTrue(first.startswith("20261101_"))
        st = wait_for(lambda: self.status(first).get("state") == "running" and self.status(first), 20,
                      what="running")
        self.assertTrue(st["pid"])
        self.assertIn("Converting", self.page(f"/predict/convert/{first}/"))
        # a second one waits: one conversion at a time
        second = self.started(self.pick_session(recipe="20261101"))
        self.assertEqual(self.status(second)["state"], "queued")
        page = self.page(f"/predict/convert/{second}/")
        self.assertIn("Waiting to start", page)
        self.assertIn('id="cj-predict-conv"', page)
        self.assertIn('data-live="1"', page)
        self.w.gate.touch()
        done = self.wait_conv(first)
        self.assertEqual(done["state"], "done", done)
        self.assertEqual(self.wait_conv(second)["state"], "done")
        # status.json: the convert JSON and the warnings
        self.assertTrue(done["result"]["ok"])
        self.assertEqual(done["n_tiles"], 3)
        self.assertEqual(done["warnings"], ["grid1_1: 7.362 A/px is not a sampling of recipe default's session"])
        self.assertTrue(done["started"] and done["finished"] and done["pid"] is None)
        meta = json.loads((self.w.root / P.RAW_TILES / first / "convert.json").read_text(encoding="utf-8"))
        raw = self.w.scan / "20261101"
        self.assertEqual((meta["owner"], meta["session"], meta["recipe"], meta["gpu"]),
                         ("tester", "20261101", "default", False))
        self.assertEqual(meta["inputs"], [str(raw / "2. Raw images" / "grid1-pos1.mrc.mdoc")])
        self.assertEqual(meta["sample_key"], str(raw / "1. Notes + reports" / "sample_key.csv"))
        # the converter's command line
        call = next(c for c in self.argv_lines() if str(self.w.root / P.RAW_TILES / first) in " ".join(c["argv"]))
        a = call["argv"]
        self.assertEqual(a[a.index("--out") + 1], str(self.w.root / P.RAW_TILES / first / "tiles"))
        self.assertEqual(a[a.index("--recipe") + 1], "default")
        self.assertEqual(a[a.index("--processing-repo") + 1], str(self.w.repo))
        self.assertEqual(call["env_repo"], str(self.w.repo))
        self.assertIn("--sample-key", a)
        self.assertNotIn("--gpu", a)
        # done page: the warning and the way on
        page = self.page(f"/predict/convert/{first}/")
        self.assertIn("A/px is not a sampling", page)
        self.assertIn(f'href="/predict/new/session/{self.src_key(first)}/">Analyse these tiles', page)
        self.assertIn("Converted with the default recipe", page)
        api = self.get(f"/predict/api/convert/{first}?html=1").json()
        self.assertEqual((api["schema"], api["state"], api["live"], api["n_tiles"]),
                         ("cj_predict_convert/1", "done", False, 3))
        # read only: the session folder is exactly as it was
        self.assertEqual(tree(self.w.scan / "20261101"), before)
        self.assertIn("20261101", self.page("/predict/"))                     # listed under raw conversions

    def test_refused_and_crashed(self):
        os.environ["FAKE_CONVERT_MODE"] = "refuse"
        os.environ["FAKE_XSS"] = XSS
        try:
            cid = self.started(self.pick_session())
            st = self.wait_conv(cid)
        finally:
            os.environ.pop("FAKE_XSS", None)
        self.assertEqual((st["state"], st["kind"]), ("failed", "refused"))
        self.assertIn("PixelSpacing", st["error"])
        self.assertEqual(len(st["warnings"]), 1)
        page = self.page(f"/predict/convert/{cid}/")
        self.assertIn("Not converted: grid1-pos1.mrc.mdoc: no finite, positive PixelSpacing", page)
        self.assertIn(html.escape(XSS), page)
        self.assertNotIn(XSS, page)
        self.assertNotIn("Analyse these tiles", page)
        os.environ["FAKE_CONVERT_MODE"] = "crash"
        cid = self.started(self.pick_session())
        st = self.wait_conv(cid)
        self.assertEqual((st["state"], st["kind"], st["error"]),
                         ("failed", "failed", "The converter stopped without a result."))
        self.assertIn("Traceback", self.page(f"/predict/convert/{cid}/"))     # the log: admin only
        self.assertNotIn(cid, [s.get("conversion") for s in P.converted_sources(None, {"id": "tester", "role": "admin"},
                                                                               lambda u, o: True)])

    def test_gpu_only_when_asked_and_free(self):
        self.w.hooks({**HOOKS_OK, "gpu": {"total_mb": 4096, "apps": [[4242, "C:\\Python\\python.exe", 900]]}})
        r = self.pick_session(gpu="yes")
        self.assertEqual(r.status_code, 409, r.text[:500])
        self.assertIn("The GPU can’t be used now", r.text)
        self.assertIn("gpu busy", r.text)                                       # the reason: admin line
        self.assertEqual(self.convs(), [])
        self.w.hooks(HOOKS_OK)
        cid = self.started(self.pick_session(gpu="yes"))
        self.wait_conv(cid)
        call = next(c for c in self.argv_lines() if str(self.w.root / P.RAW_TILES / cid) in " ".join(c["argv"]))
        self.assertIn("--gpu", call["argv"])
        cid = self.started(self.pick_session())
        self.wait_conv(cid)
        call = next(c for c in self.argv_lines() if str(self.w.root / P.RAW_TILES / cid) in " ".join(c["argv"]))
        self.assertNotIn("--gpu", call["argv"])

    def test_bad_pick_refused(self):
        r = self.pick_session(montages=["../../etc/passwd"])
        self.assertEqual(r.status_code, 400)
        self.assertIn("Tick at least one montage", r.text)
        r = self.pick_session(montages=["2. Raw images/grid1-pos1.mrc.mdoc", "2. Raw images/gone.mrc.mdoc"])
        self.assertEqual(r.status_code, 400)
        self.assertIn("changed since this page was opened", r.text)
        self.assertEqual(self.convs(), [])


class Visibility(Base):
    def test_owner_and_admin_only_then_the_normal_flow(self):
        cid = self.converted("advanced")
        self.assertTrue(cid.startswith("20260827-2_"))
        key = self.src_key(cid)
        path = f"/predict/new/session/{key}/"
        for who, st in (("advanced", 200), ("admin", 200), ("adv2", 404), ("basic", 404)):
            with self.subTest(who=who):
                self.assertEqual(self.get(path, who).status_code, st)
                self.assertEqual(self.get(f"/predict/convert/{cid}/", who).status_code, st)
                self.assertEqual(self.get(f"/predict/api/convert/{cid}", who).status_code, st)
                listed = f'href="{path}"' in self.page("/predict/new/", who)
                self.assertEqual(listed, st == 200)
        self.assertIn("converted by adv", self.page("/predict/new/", "admin"))
        self.assertNotIn("converted by", self.page("/predict/new/", "advanced"))
        # the uploaded raw files are kept CJ_RAW_KEEP_DAYS (default 7) after the end; the tiles stay
        d = self.w.root / P.RAW_TILES / cid
        self.assertTrue((d / "input" / "grid1-pos1.mrc").is_file())
        self.assertIn("The uploaded raw files are kept until", self.page(f"/predict/convert/{cid}/", "advanced"))
        self.assertEqual(len(list((d / "tiles").glob("*.png"))), 3)
        # the normal Predict flow: sample groups from the key (escaped), nm/px shown, the tick required
        form = self.page(path, "advanced")
        self.assertIn("Session 20260827-2", form)
        self.assertIn("EV prep A", form)
        self.assertIn(html.escape(XSS), form)
        self.assertNotIn(XSS, form)
        self.assertIn('value="1.4724"', form)
        sig = re.search(r'name="sig" value="([0-9a-f]+)"', form).group(1)
        data = {"sig": sig, "samples": ["g0", "g1"], "nm_g0": "1.4724", "nm_g1": "1.4724", "how": "all",
                "model": MODEL}
        r = self.form(path, data, "advanced")
        self.assertEqual(r.status_code, 400)
        self.assertIn("I confirm the pixel size", r.text)
        self.assertEqual(self.sets(), self.sets0)
        r = self.form(path, {**data, "confirm": "yes"}, "advanced")
        self.assertEqual(r.status_code, 303, r.text[:800])
        ps = self.follow(r.headers["location"]).strip("/").split("/")[-1]
        req = json.loads((self.w.root / ps / "request.json").read_text(encoding="utf-8"))
        self.assertEqual((req["owner"], req["source"]["kind"], req["source"]["session"]),
                         ("adv", "session", "20260827-2"))
        rows = list(csv.DictReader(open(self.w.root / ps / "images.csv", newline="", encoding="utf-8")))
        self.assertEqual({(r_["nm_per_px"], r_["nm_confirmed_by"]) for r_ in rows}, {("1.4724", "adv")})
        self.assertEqual(sorted({r_["sample_label"] for r_ in rows}), sorted(["EV prep A", XSS]))
        set_page = self.page(f"/predict/{ps}/", "advanced")
        self.assertIn("Session 20260827-2 (Session 27 Aug 2026 -2 · converted", set_page)
        wait_for(lambda: json.loads((self.w.root / ps / "status.json").read_text(encoding="utf-8"))["state"]
                 in ("done", "failed"), 60, what="the fake predictor")

    def test_tile_cap_applies(self):
        self.assertEqual(P.MAX_FILES, 1000)
        os.environ["FAKE_CONVERT_N"] = "5"
        cid = self.converted("advanced", session="20261101", recipe="20261101")
        path = f"/predict/new/session/{self.src_key(cid)}/"
        with mock.patch.object(P, "MAX_FILES", 4):
            return self._tile_cap(path)

    def _tile_cap(self, path):
        form = self.page(path, "advanced")
        self.assertIn("At most 4 tiles per analysis", form)
        sig = re.search(r'name="sig" value="([0-9a-f]+)"', form).group(1)
        r = self.form(path, {"sig": sig, "samples": ["g0", "g1"], "nm_g0": "1.4724", "nm_g1": "1.4724",
                             "how": "all", "confirm": "yes", "model": MODEL}, "advanced")
        self.assertEqual(r.status_code, 400)
        self.assertIn("5 tiles chosen", r.text)
        self.assertEqual(self.sets(), self.sets0)


class Upload(Base):
    def fields(self, **kw):
        f = {"sample_label": "Up " + XSS, "nm_per_px": "1.4724", "confirm": "yes", "model": MODEL}
        f.update(kw)
        return {k: v for k, v in f.items() if v is not None}

    def tiles(self, n=2):
        return [(f"my tile {i}.png", tile_bytes(i)) for i in range(1, n + 1)]

    def scratch_empty(self):
        tmp = self.w.root / P.PORTAL_TMP
        return not (tmp.exists() and any(tmp.iterdir()))

    def test_one_piece_per_request_to_a_set(self):
        with mock.patch.object(P, "CHUNK_BYTES", 700):
            st, r = self.upload("basic", "tiles", self.fields(), self.tiles(3), chunk=700)
        self.assertEqual(st["chunk"], 700)
        self.assertEqual(r.status_code, 200, r.text[:600])
        ps = self.follow(r.json()["href"], "basic").strip("/").split("/")[-1]
        self.assertEqual(sorted(set(self.sets()) - set(self.sets0)), [ps])
        rows = list(csv.DictReader(open(self.w.root / ps / "images.csv", newline="", encoding="utf-8")))
        self.assertEqual([(r_["orig_name"], r_["nm_confirmed_by"], r_["sample_label"]) for r_ in rows],
                         [(f"my tile {i}.png", "bas", "Up " + XSS) for i in (1, 2, 3)])
        self.assertEqual([r_["md5"] for r_ in rows], [hashlib.md5(tile_bytes(i)).hexdigest() for i in (1, 2, 3)])
        self.assertTrue(self.scratch_empty())
        page = self.page(f"/predict/{ps}/", "basic")
        self.assertIn("Up &lt;img src=x", page)
        self.assertNotIn(XSS, page)
        wait_for(lambda: json.loads((self.w.root / ps / "status.json").read_text(encoding="utf-8"))["state"]
                 in ("done", "failed"), 60, what="the fake predictor")

    def test_resume_after_a_dropped_piece(self):
        files = self.tiles(2)
        with mock.patch.object(P, "CHUNK_BYTES", 1000):
            r = self.jpost("/predict/api/upload", {"kind": "tiles", "fields": self.fields(),
                                                   "files": [{"name": n, "size": len(b)} for n, b in files]}, "basic")
            st = r.json()
            uid = st["upload"]
            name, data = files[0]
            self.assertEqual(self.piece(uid, name, 0, data[:1000], "basic").status_code, 200)
            # the next piece "lost": the page asks again with a stale offset and is told where it stands
            r = self.piece(uid, name, 0, data[:1000], "basic")
            self.assertEqual(r.status_code, 409)
            self.assertEqual(r.json()["received"][name], 1000)
            # a reload: the same files resume in the same session
            r = self.jpost("/predict/api/upload", {"kind": "tiles", "fields": self.fields(), "resume": uid,
                                                   "files": [{"name": n, "size": len(b)} for n, b in files]}, "basic")
            self.assertEqual(r.json()["upload"], uid)
            self.assertEqual(r.json()["received"][name], 1000)
            # create too early: kept, says what is missing
            r = self.jpost(f"/predict/api/upload/{uid}/create", {}, "basic")
            self.assertEqual(r.status_code, 409)
            self.assertIn("not sent yet", r.json()["error"])
            self.assertFalse(r.json()["dropped"])
            for n, b in files:
                off = self.get(f"/predict/api/upload/{uid}", "basic").json()["received"][n]
                while off < len(b):
                    rr = self.piece(uid, n, off, b[off:off + 1000], "basic")
                    self.assertEqual(rr.status_code, 200, rr.text)
                    off = rr.json()["received"][n]
            r = self.jpost(f"/predict/api/upload/{uid}/create", {}, "basic")
        self.assertEqual(r.status_code, 200, r.text)
        ps = self.follow(r.json()["href"], "basic").strip("/").split("/")[-1]
        self.assertEqual(len(list(csv.DictReader(open(self.w.root / ps / "images.csv", encoding="utf-8")))), 2)
        self.assertTrue(self.scratch_empty())
        wait_for(lambda: json.loads((self.w.root / ps / "status.json").read_text(encoding="utf-8"))["state"]
                 in ("done", "failed"), 60, what="the fake predictor")

    def test_refusals_keep_nothing(self):
        before = self.sets()
        cases = [
            ("caps: count", {"MAX_FILES": 2}, self.fields(), self.tiles(3), 413, "More than 2 files"),
            ("caps: size", {"MAX_FILE_BYTES": 1000}, self.fields(), self.tiles(1), 413, "is over"),
            ("no tick", {}, self.fields(confirm=None), self.tiles(1), 400, "I confirm the pixel size"),
            ("nm", {}, self.fields(nm_per_px="abc"), self.tiles(1), 400, "enter the pixel size"),
            ("label", {}, self.fields(sample_label=""), self.tiles(1), 400, "Sample name"),
            ("name", {}, self.fields(), [("a:b.png", b"x")], 400, "File name not accepted"),
            ("name xss", {}, self.fields(), [(XSS + ".png", b"x")], 400, "File name not accepted"),
            ("twice", {}, self.fields(), self.tiles(1) + [("MY TILE 1.png", b"x")], 400, "Two files are named"),
            ("none", {}, self.fields(), [], 400, "Choose at least one image"),
        ]
        for why, patch, fields, files, status, text in cases:
            with self.subTest(why=why), mock.patch.multiple(P, **(patch or {"REFRESH_S": P.REFRESH_S})):
                st, _ = self.upload("basic", "tiles", fields, files, expect=status)
                self.assertIn(text, st["error"])                       # JSON text: the page shows it as text
                self.assertTrue(st["dropped"])
        # content: checked when each file is complete; the session is removed
        for why, data, text in (("not an image", b"MZ\x90\x00 not an image" * 10, "not a PNG or JPEG"),
                                ("16-bit", png16(), "16-bit PNG")):
            with self.subTest(why=why):
                r = self.jpost("/predict/api/upload", {"kind": "tiles", "fields": self.fields(),
                                                       "files": [{"name": "x.png", "size": len(data)}]}, "basic")
                uid = r.json()["upload"]
                rr = self.piece(uid, "x.png", 0, data, "basic")
                self.assertEqual(rr.status_code, 400)
                self.assertIn(text, rr.json()["error"])
                self.assertNotIn(str(self.w.root), rr.text)
                self.assertEqual(self.get(f"/predict/api/upload/{uid}", "basic").status_code, 404)
        # a piece bigger than allowed, or past the declared size
        for why, piece_len, chunk, text, status in (("too big", 1500, 1000, "at most", 413),
                                                    ("past the end", 10, 10 ** 6, "more bytes than its size", 400)):
            with self.subTest(why=why), mock.patch.object(P, "CHUNK_BYTES", chunk):
                data = tile_bytes(1)
                r = self.jpost("/predict/api/upload", {"kind": "tiles", "fields": self.fields(),
                                                       "files": [{"name": "t.png", "size": len(data)}]}, "basic")
                uid = r.json()["upload"]
                off = 0 if why == "too big" else len(data) - 5
                if off:
                    self.assertEqual(self.piece(uid, "t.png", 0, data[:off], "basic").status_code, 200)
                rr = self.piece(uid, "t.png", off, data[off:off + piece_len] + b"\0" * 20, "basic")
                self.assertEqual(rr.status_code, status, rr.text)
                self.assertIn(text, rr.json()["error"])
        self.assertEqual(self.sets(), before)
        self.assertTrue(self.scratch_empty())

    def test_disk_checked_at_start_and_at_each_piece(self):
        margin = P.FREE_MARGIN_BYTES
        files = self.tiles(2)
        with mock.patch.object(P, "disk_free", return_value=margin + 10):      # the declared total does not fit
            st, _ = self.upload("basic", "tiles", self.fields(), files, expect=507)
        self.assertIn("Not enough disk space", st["error"])
        r = self.jpost("/predict/api/upload", {"kind": "tiles", "fields": self.fields(),
                                               "files": [{"name": n, "size": len(b)} for n, b in files]}, "basic")
        uid = r.json()["upload"]
        held = sum(len(b) for _, b in files)
        self.assertEqual(sum(P._RESERVED.values()), held)                     # the whole declared size is held
        name, data = files[0]
        self.assertEqual(self.piece(uid, name, 0, data, "basic").status_code, 200)
        self.assertEqual(sum(P._RESERVED.values()), held - len(data))
        with mock.patch.object(P, "disk_free", return_value=margin - 1):       # the disk filled up meanwhile
            rr = self.piece(uid, files[1][0], 0, files[1][1], "basic")
        self.assertEqual(rr.status_code, 507)
        self.assertTrue(rr.json()["dropped"])
        self.assertEqual(P._RESERVED, {})
        self.assertTrue(self.scratch_empty())
        self.assertEqual(self.sets(), self.sets0)

    def test_no_create_is_swept_with_its_hold(self):
        st, _ = self.upload("basic", "tiles", self.fields(), self.tiles(2), create=False)
        uid = st["upload"]
        self.assertTrue(P._RESERVED)
        self.assertEqual(P.sweep_scratch(None), 0)                              # still fresh: kept
        self.assertEqual(self.get(f"/predict/api/upload/{uid}", "basic").status_code, 200)
        self.assertEqual(P.sweep_scratch(None, max_age_s=-1), 1)                # unused past the limit: removed
        self.assertEqual(self.get(f"/predict/api/upload/{uid}", "basic").status_code, 404)
        self.assertEqual(P._RESERVED, {})
        self.assertEqual(self.sets(), self.sets0)
        self.assertTrue(self.scratch_empty())

    def test_only_the_owner_reaches_a_session(self):
        st, _ = self.upload("basic", "tiles", self.fields(), self.tiles(1), create=False)
        uid = st["upload"]
        for who in ("admin", "advanced"):
            self.assertEqual(self.get(f"/predict/api/upload/{uid}", who).status_code, 404)
            self.assertEqual(self.piece(uid, "my tile 1.png", 0, b"x", who).status_code, 404)
            self.assertEqual(self.jpost(f"/predict/api/upload/{uid}/create", {}, who).status_code, 404)
        r = self.tc.post(f"/predict/api/upload/{uid}/create", headers=self.h("basic", origin=False))
        self.assertEqual(r.status_code, 403)                                    # no Origin: the Gate refuses
        self.assertEqual(self.jpost(f"/predict/api/upload/{uid}/cancel", {}, "basic").json(), {"ok": True})
        self.assertEqual(self.sets(), self.sets0)

    def test_raw_upload_checks(self):
        cases = [
            ("mrc without mdoc", [("a.mrc", mrc_bytes())], 400, "a.mrc.mdoc is missing"),
            ("mdoc without mrc", [("a.mrc.mdoc", MDOC.encode())], 400, "its a.mrc is missing"),
            ("tif without idoc", [("t.tif", b"II*\x00" + b"\0" * 20)], 400, "TIFF tiles need"),
            ("not raw", [("a.exe", b"MZ")], 400, "not a raw montage file"),
            ("two keys", self.raw_files() + [("k2.csv", b"#,Label\n")], 400, "at most one sample key"),
            ("sidecar cap", [("a.mrc", mrc_bytes()), ("a.mrc.mdoc", b"x" * (P.RAW_SIDE_MAX_BYTES + 1))], 413,
             "is over"),
        ]
        for why, files, status, text in cases:
            with self.subTest(why=why):
                st, _ = self.upload("advanced", "raw", {"session": "20261101"}, files, expect=status)
                self.assertIn(text, st["error"])
        for why, files, text in (("fake mrc", [("a.mrc", b"\0" * 2048), ("a.mrc.mdoc", MDOC.encode())],
                                  "a.mrc: not an MRC file"),
                                 ("fake mdoc", [("a.mrc", mrc_bytes()), ("a.mrc.mdoc", b"hello")],
                                  "not a SerialEM mdoc file")):
            with self.subTest(why=why):
                r = self.jpost("/predict/api/upload", {"kind": "raw", "fields": {"session": "20261101"},
                                                       "files": [{"name": n, "size": len(b)} for n, b in files]},
                               "advanced")
                uid = r.json()["upload"]
                errs = [self.piece(uid, n, 0, b, "advanced") for n, b in files]
                bad = [e for e in errs if e.status_code != 200]
                self.assertEqual(bad[0].status_code, 400)
                self.assertIn(text, bad[0].json()["error"])
        self.assertEqual(self.convs(), [])
        self.assertTrue(self.scratch_empty())


class NoJS(Base):
    def multipart(self, fields, files):
        from uuid import uuid4
        b = "cjb" + uuid4().hex
        out = io.BytesIO()
        for k, v in fields.items():
            out.write(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        for name, data in files:
            out.write(f'--{b}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
                      f'Content-Type: application/octet-stream\r\n\r\n'.encode())
            out.write(data + b"\r\n")
        out.write(f"--{b}--\r\n".encode())
        return f"multipart/form-data; boundary={b}", out.getvalue()

    def test_small_plain_form_works_big_is_refused_with_why(self):
        fields = {"sample_label": "Plain", "nm_per_px": "1.4724", "confirm": "yes", "model": MODEL}
        files = [("a.png", tile_bytes(1)), ("b.png", tile_bytes(2))]
        ctype, body = self.multipart(fields, files)
        r = self.tc.post("/predict/new/upload/", content=body, headers=self.h("basic", **{"Content-Type": ctype}))
        self.assertEqual(r.status_code, 303, r.text[:500])
        ps = r.headers["location"].strip("/").split("/")[-1]
        with mock.patch.object(P, "NOJS_MAX_BYTES", 1000):
            r = self.tc.post("/predict/new/upload/", content=body, headers=self.h("basic", **{"Content-Type": ctype}))
        self.assertEqual(r.status_code, 413)
        self.assertIn("Without JavaScript, one upload is at most 0 MB in total", r.text)
        self.assertIn("Turn JavaScript on", r.text)
        self.assertEqual(sorted(set(self.sets()) - set(self.sets0)), [ps])
        page = self.page("/predict/new/upload/", "basic")
        self.assertIn("<noscript>", page)
        self.assertIn('action="/predict/new/upload/"', page)
        self.assertIn('data-cj-upload="tiles"', page)
        r = self.form("/predict/new/raw/", {"session": "20261101"}, "advanced")
        self.assertEqual(r.status_code, 400)
        self.assertIn("needs JavaScript", r.text)
        wait_for(lambda: json.loads((self.w.root / ps / "status.json").read_text(encoding="utf-8"))["state"]
                 in ("done", "failed"), 60, what="the fake predictor")

    def test_no_inline_script_on_the_new_pages(self):
        cid = self.started(self.pick_session())
        self.wait_conv(cid)
        for path in ("/predict/new/raw/", "/predict/new/raw/sessions/", f"/predict/new/raw/sessions/{self.raw_key()}/",
                     f"/predict/convert/{cid}/", "/predict/"):
            page = self.page(path)
            self.assertIn('<script src="/predict/predict.js" defer></script>', page)
            self.assertEqual(re.findall(r"<script(?![^>]*\ssrc=)[^>]*>", page), [], path)
            self.assertEqual(re.findall(r"<[a-z][^>]*\son[a-z]+\s*=", page), [], path)
            self.assertNotIn(XSS, page)


IDOC_HEAD = "PixelSpacing = 7.362\nImageSize = 64 64\nMontage = 1\n\n"
TIF = b"II*\x00" + b"\0" * 60


def idoc_with(*names) -> bytes:
    return (IDOC_HEAD + "".join(f"[Image = {n}]\nPieceCoordinates = 0 0 0\n\n" for n in names)).encode("latin-1")


class SidecarPaths(Base):
    """QC MUST: an uploaded (or picked) sidecar's [Image = ...] must be a bare name of an uploaded file; an absolute,
    UNC or '..' name never reaches the converter."""

    def raw_up(self, files, expect_piece=200, expect_create=200):
        r = self.jpost("/predict/api/upload", {"kind": "raw", "fields": {"session": "20261101"},
                                               "files": [{"name": n, "size": len(b)} for n, b in files]}, "advanced")
        self.assertEqual(r.status_code, 200, r.text)
        uid = r.json()["upload"]
        codes = [self.piece(uid, n, 0, b, "advanced") for n, b in files]
        bad = [c for c in codes if c.status_code != 200]
        if expect_piece != 200:
            self.assertEqual(bad[0].status_code, expect_piece, bad[0].text)
            return bad[0].json()
        self.assertEqual(bad, [])
        c = self.jpost(f"/predict/api/upload/{uid}/create", {}, "advanced")
        self.assertEqual(c.status_code, expect_create, c.text)
        return c.json()

    def test_paths_in_image_entries_refused(self):
        secret = str(self.w.tmp / "secret.tif")
        cases = {"absolute": secret, "unc": r"\\gpu-desktop\share\x.tif", "dotdot": r"..\..\x.tif",
                 "dotdot-slash": "../x.tif", "drive": "C:x.tif", "empty": ""}
        for why, ref in cases.items():
            for side in ("grid3-1.idoc", "grid1-pos1.mrc.mdoc"):
                with self.subTest(why=why, sidecar=side):
                    files = [(side, idoc_with("t1.tif", ref))]
                    files.append(("grid1-pos1.mrc", mrc_bytes()) if side.endswith(".mrc.mdoc") else ("t1.tif", TIF))
                    j = self.raw_up(files, expect_piece=400)
                    self.assertIn("not a plain file name", j["error"])
                    self.assertTrue(j["dropped"])
        # a bare name that was not uploaded: refused at create, nothing kept
        j = self.raw_up([("grid3-1.idoc", idoc_with("t1.tif", "t2.tif")), ("t1.tif", TIF)], expect_create=400)
        self.assertIn("'t2.tif' that was not uploaded", j["error"])
        self.assertEqual(self.convs(), [])
        self.assertEqual(self.argv_lines(), [])                        # the converter never ran
        # control: a valid idoc with its tiles converts
        j = self.raw_up([("grid3-1.idoc", idoc_with("t1.tif", "t2.tif")), ("t1.tif", TIF), ("t2.tif", TIF)])
        cid = CID_RE.match(j["href"]).group(1)
        self.assertEqual(self.wait_conv(cid)["state"], "done")

    def test_session_pick_with_a_path_refused(self):
        key = self.raw_key("20261104")
        before = self.convs()
        r = self.form(f"/predict/new/raw/sessions/{key}/", {"montages": ["grid4-1.idoc"], "recipe": "default"})
        self.assertEqual(r.status_code, 400, r.text[:500])
        self.assertIn("not a plain file name", r.text)
        self.assertEqual(self.convs(), before)

    def test_sidecar_reader_matches_the_core(self):
        p = self.w.tmp / "x.idoc"
        p.write_bytes(b"PixelSpacing = 1\n[T = SerialEM: x]\n[Image = a.tif]\n  [Image=b.tif ]  \n"
                      b"[ZValue = 0]\n[Image = c=d.tif]\n[MontSection = 0]\n")
        self.assertEqual(P.sidecar_images(p), ["a.tif", "b.tif", "c"])   # cut at the 2nd "=", as parse_metadata


class Stuck(Base):
    """QC SHOULD 2: Stop / Retry (admin), a maximum runtime, and a reused pid is not the converter."""

    def gated(self):
        os.environ["FAKE_CONVERT_GATE"] = str(self.w.gate)
        self.w.gate.unlink(missing_ok=True)
        cid = self.started(self.pick_session())
        wait_for(lambda: self.status(cid).get("state") == "running", 20, what="running")
        return cid

    def test_stop_and_retry(self):
        cid = self.gated()
        st = self.status(cid)
        self.assertTrue(P.proc_alive(st))
        for who in ("advanced", "basic"):
            self.assertEqual(self.form(f"/predict/convert/{cid}/stop", {}, who).status_code, 403)
        self.assertIn('action="/predict/convert/%s/stop"' % cid, self.page(f"/predict/convert/{cid}/"))
        r = self.form(f"/predict/convert/{cid}/stop", {})
        self.assertEqual(r.status_code, 303)
        st2 = self.status(cid)
        self.assertEqual((st2["state"], st2["kind"], st2["error"]), ("failed", "stopped", "Stopped by the admin."))
        wait_for(lambda: not P.PL.pid_alive(st["pid"]), 15, what="the converter killed")
        page = self.page(f"/predict/convert/{cid}/")
        self.assertIn("Stopped: Stopped by the admin.", page)
        self.assertIn('action="/predict/convert/%s/retry"' % cid, page)
        self.w.gate.touch()
        self.assertEqual(self.form(f"/predict/convert/{cid}/retry", {}, "advanced").status_code, 403)
        self.assertEqual(self.form(f"/predict/convert/{cid}/retry", {}).status_code, 303)
        st3 = self.wait_conv(cid)
        self.assertEqual((st3["state"], st3["retries"]), ("done", 1))
        r = self.form(f"/predict/convert/{cid}/retry", {})                 # done: nothing to retry
        self.assertEqual(r.status_code, 409)

    def test_max_runtime_and_silent_log(self):
        for patch, text in (({"CONVERT_MAX_S": 0}, "It ran for over"),
                            ({"CONVERT_STALE_S": -1}, "No progress for")):
            with self.subTest(patch=patch):
                cid = self.gated()
                pid = self.status(cid)["pid"]
                with mock.patch.multiple(P, **patch):
                    P.kick(None)
                st = self.status(cid)
                self.assertEqual((st["state"], st["kind"]), ("failed", "timeout"))
                self.assertIn(text, st["error"])
                wait_for(lambda: not P.PL.pid_alive(pid), 15, what="the converter killed")
        self.assertIsNone(P.overdue(self.w.root, {"started": P.now_iso()}))

    def test_a_reused_pid_is_not_the_converter(self):
        me = os.getpid()                                                # alive, but not a converter
        made = P.proc_created(me)
        self.assertIsNotNone(made)
        self.assertTrue(P.proc_alive({"pid": me, "pid_created": made}))  # control: the right start time
        self.assertFalse(P.proc_alive({"pid": me, "pid_created": made - 3600}))
        self.assertFalse(P.proc_alive({"pid": me, "started": "2020-01-01T00:00:00"}))
        d = self.w.root / P.RAW_TILES / "20261101_abcdef"
        d.mkdir(parents=True)
        PS_ = P.PS
        PS_.write_json_atomic(d / "convert.json", {"schema": "cj_convert/1", "id": d.name, "owner": "tester",
                                                   "session": "20261101", "recipe": "default", "gpu": False,
                                                   "source": {"kind": "session"}, "inputs": [], "created": P.now_iso()})
        PS_.write_json_atomic(d / "status.json", {"state": "running", "pid": me, "pid_created": made - 3600,
                                                  "started": P.now_iso()})
        P.kick(None)
        st = self.status(d.name)
        self.assertEqual((st["state"], st["error"]), ("failed", "The converter stopped without a result."))
        self.assertTrue(P.PL.pid_alive(me))                             # never killed: it was not the converter


class Retention(Base):
    """QC SHOULD 3: uploaded raw inputs kept CJ_RAW_KEEP_DAYS after the end (done or failed), the admin's delete,
    and at most 2 conversions per person queued or running."""

    def tearDown(self):
        os.environ.pop("CJ_RAW_KEEP_DAYS", None)
        super().tearDown()

    def test_kept_then_swept_and_deleted_on_demand(self):
        cid = self.converted("advanced", session="20261101", recipe="20261101")
        d = self.w.root / P.RAW_TILES / cid
        self.assertTrue((d / "input").is_dir())
        # admin: delete now (advanced may not)
        self.assertEqual(self.form(f"/predict/convert/{cid}/inputs", {}, "advanced").status_code, 403)
        self.assertIn('action="/predict/convert/%s/inputs"' % cid, self.page(f"/predict/convert/{cid}/"))
        self.assertEqual(self.form(f"/predict/convert/{cid}/inputs", {}).status_code, 303)
        self.assertFalse((d / "input").exists())
        self.assertTrue(self.status(cid)["inputs_removed"])
        self.assertTrue((d / "tiles" / "MANIFEST.csv").is_file())
        # older than the keep time: swept (done and failed alike)
        os.environ["FAKE_CONVERT_MODE"] = "refuse"
        failed = self.converted_any("advanced")
        os.environ.pop("FAKE_CONVERT_MODE")
        ok = self.converted("advanced", session="20261101", recipe="20261101")
        for c in (failed, ok):
            self.assertTrue((self.w.root / P.RAW_TILES / c / "input").is_dir())
            st = self.status(c)
            st["finished"] = "2026-01-01T00:00:00"
            P.PS.write_json_atomic(self.w.root / P.RAW_TILES / c / "status.json", st)
        os.environ["CJ_RAW_KEEP_DAYS"] = "7"
        P.kick(None)
        for c in (failed, ok):
            self.assertFalse((self.w.root / P.RAW_TILES / c / "input").exists(), c)

    def converted_any(self, who) -> str:
        st, r = self.upload(who, "raw", {"session": "20261101", "recipe": "20261101"}, self.raw_files())
        cid = CID_RE.match(r.json()["href"]).group(1)
        self.wait_conv(cid)
        return cid

    def test_keep_zero_removes_at_the_end(self):
        os.environ["CJ_RAW_KEEP_DAYS"] = "0"
        ok = self.converted_any("advanced")
        os.environ["FAKE_CONVERT_MODE"] = "refuse"
        failed = self.converted_any("advanced")
        for c in (ok, failed):
            wait_for(lambda: not (self.w.root / P.RAW_TILES / c / "input").exists(), 10, what=f"{c} inputs gone")
        self.assertEqual((self.status(ok)["state"], self.status(failed)["state"]), ("done", "failed"))
        self.assertEqual(P.keep_days(None), 0.0)
        os.environ["CJ_RAW_KEEP_DAYS"] = "nonsense"
        self.assertEqual(P.keep_days(None), 7.0)

    def test_two_per_person(self):
        os.environ["FAKE_CONVERT_GATE"] = str(self.w.gate)
        self.w.gate.unlink(missing_ok=True)
        files = self.raw_files()
        for _ in range(2):
            st, r = self.upload("advanced", "raw", {"session": "20261101"}, files)
            self.assertEqual(r.status_code, 200, r.text)
        st, _ = self.upload("advanced", "raw", {"session": "20261101"}, files, expect=429)
        self.assertIn("already have 2 conversions", st["error"])
        self.upload("adv2", "raw", {"session": "20261101"}, files, create=False)    # someone else: fine
        for _ in range(2):
            self.assertEqual(self.pick_session().status_code, 303)
        r = self.pick_session()
        self.assertEqual(r.status_code, 429)
        self.assertIn("already have 2 conversions", r.text)
        self.w.gate.touch()

    def test_disk_checked_before_converting(self):
        with mock.patch.object(P, "disk_free", return_value=P.FREE_MARGIN_BYTES + 10):
            r = self.pick_session()
        self.assertEqual(r.status_code, 507)
        self.assertIn("Not enough disk space on the server to convert", r.text)
        self.assertEqual(self.convs(), [])


class UploadRules(Base):
    """QC SHOULD 4-6: bad piece requests keep the session; name rules; the converted-source cache."""

    def test_bad_requests_keep_the_session(self):
        data = tile_bytes(1)
        r = self.jpost("/predict/api/upload", {"kind": "tiles", "fields": {"sample_label": "S", "nm_per_px": "1.4724",
                                                                           "confirm": "yes", "model": MODEL},
                                               "files": [{"name": "t.png", "size": len(data)}]}, "basic")
        uid = r.json()["upload"]
        for path, status, text in ((f"/predict/api/upload/{uid}/file?name=other.png&offset=0", 400,
                                    "is not one of the files"),
                                   (f"/predict/api/upload/{uid}/file?name=t.png&offset=abc", 400, "offset")):
            rr = self.tc.post(path, content=data[:100], headers=self.h("basic", **{
                "Content-Type": "application/octet-stream"}))
            self.assertEqual(rr.status_code, status, rr.text)
            self.assertIn(text, rr.json()["error"])
            self.assertFalse(rr.json()["dropped"])
            self.assertEqual(self.get(f"/predict/api/upload/{uid}", "basic").status_code, 200)
        self.assertEqual(self.piece(uid, "t.png", 50, data[:10], "basic").status_code, 409)
        self.assertEqual(self.piece(uid, "t.png", 0, data, "basic").status_code, 200)        # still usable
        self.jpost(f"/predict/api/upload/{uid}/cancel", {}, "basic")

    def test_names_by_kind(self):
        fields = {"sample_label": "S", "nm_per_px": "1.4724", "confirm": "yes", "model": MODEL}
        for kind, f, names, text in (
                ("tiles", fields, ["x.txt"], "not a PNG or JPEG file name"),
                ("tiles", fields, ["x.tif"], "not a PNG or JPEG file name"),
                ("tiles", fields, ["a.png", "a.png.part"], ".part"),
                ("raw", {"session": "20261101"}, ["a.mrc", "a.mrc.mdoc", "a.mrc.part"], ".part"),
                ("raw", {"session": "20261101"}, ["a.png"], "not a raw montage file")):
            with self.subTest(names=names):
                who = "basic" if kind == "tiles" else "advanced"
                st, _ = self.upload(who, kind, f, [(n, b"x" * 10) for n in names], expect=400)
                self.assertIn(text, st["error"])

    def test_converted_sources_cached_by_status_mtime(self):
        cid = self.converted("advanced", session="20261101", recipe="20261101")
        admin = {"id": "tester", "role": "admin"}
        calls = []
        real = P.PSRC.check_folder

        def counting(d, *a, **k):
            calls.append(Path(d).parent.name)
            return real(d, *a, **k)
        P._SRC_CACHE.clear()
        with mock.patch.object(P.PSRC, "check_folder", side_effect=counting):
            P.converted_sources(None, admin, lambda u, o: True)
            n = calls.count(cid)
            P.converted_sources(None, admin, lambda u, o: True)
            self.assertEqual(calls.count(cid), n)                        # cached
            time.sleep(0.05)
            P.write_conv_status(self.w.root / P.RAW_TILES / cid, note="touched")
            P.converted_sources(None, admin, lambda u, o: True)
            self.assertEqual(calls.count(cid), n + 1)                    # status changed: checked again
        self.assertEqual(n, 1)


def launch(p):
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:
            continue
    return None


class Browser(unittest.TestCase):
    """The real portal (uvicorn in a thread) and Edge: the pages at 390 and 1280 px, and the JS upload for real."""

    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        cls.w = World()
        cls.p = T.Portal(cls.w.cfg)
        cls.own_shots = not os.environ.get("CJ_SHOT_DIR")
        cls.shots = Path(os.environ.get("CJ_SHOT_DIR") or tempfile.mkdtemp(prefix="cj_shots_"))
        cls.shots.mkdir(parents=True, exist_ok=True)
        cls.pw = sync_playwright().start()
        cls.browser = launch(cls.pw)
        if cls.browser is None:
            cls.tearDownClass()
            raise unittest.SkipTest("no Edge / Chrome for Playwright")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "browser", None):
            cls.browser.close()
        if getattr(cls, "pw", None):
            cls.pw.stop()
        cls.p.stop()
        cls.w.close()
        if cls.own_shots:
            shutil.rmtree(cls.shots, ignore_errors=True)

    def session(self, width, who="admin", scheme="light"):
        ctx = self.browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
        origin = self.p.origin
        rec = {"requests": [], "foreign": [], "errors": [], "console": [], "bad": []}
        ctx.route("**/*", lambda r: (r.continue_() if r.request.url.startswith(origin + "/")
                                     else (rec["foreign"].append(r.request.url), r.abort())))
        page = ctx.new_page()
        page.on("request", lambda r: rec["requests"].append((r.method, r.url, r.post_data_buffer)))
        page.on("response", lambda r: r.status >= 400 and rec["bad"].append((r.status, r.request.method, r.url)))
        page.on("pageerror", lambda e: rec["errors"].append(str(e)))
        page.on("console", lambda m: m.type == "error" and rec["console"].append(m.text))
        page.goto(f"{origin}/login?k={self.w.keys[who]}")
        self.assertEqual(page.url, origin + "/")
        return ctx, page, rec

    def check_clean(self, rec):
        self.assertEqual(rec["bad"][:5], [])
        self.assertEqual(rec["foreign"], [])
        self.assertTrue(all(u.startswith(self.p.origin + "/") for _, u, _ in rec["requests"]))
        self.assertEqual(rec["errors"], [])
        self.assertEqual(rec["console"], [])

    def shot(self, page, name):
        page.screenshot(path=str(self.shots / f"{name}.png"), full_page=True)

    def wait_done(self, cid):
        def st():
            try:
                return json.loads((self.w.root / P.RAW_TILES / cid / "status.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
        wait_for(lambda: st().get("state") in ("done", "failed"), 60, what=cid)
        return st()

    def test_pages_and_js_uploads(self):
        PSRC.clear_cache()
        key = next(s["key"] for s in P.raw_sessions(None) if s["folder"] == "20261101")
        for width in (390, 1280):
            ctx, page, rec = self.session(width)
            try:
                # a raw upload through the page's script, one file (piece) per request, then the conversion page
                with mock.patch.object(P, "CHUNK_BYTES", 1500):
                    page.goto(self.p.origin + "/predict/new/raw/")
                    page.fill('input[name="session"]', "20260827-2")
                    page.select_option('select[name="recipe"]', "20260827_2")
                    files = self.w.tmp / f"up{width}"
                    files.mkdir()
                    (files / "grid1-pos1.mrc").write_bytes(mrc_bytes())
                    (files / "grid1-pos1.mrc.mdoc").write_text(MDOC, encoding="utf-8")
                    (files / "sample_key.csv").write_text(SAMPLE_KEY, encoding="utf-8")
                    page.set_input_files('input[name="files"]', [str(files / n) for n in
                                                                  ("grid1-pos1.mrc", "grid1-pos1.mrc.mdoc",
                                                                   "sample_key.csv")])
                    self.shot(page, f"predict_raw_upload_{width}")
                    page.click('button[type="submit"]')
                    page.wait_for_url(re.compile(r".*/predict/convert/20260827-2_[0-9a-f]{6}/$"), timeout=30000)
                cid = page.url.rstrip("/").split("/")[-1]
                pieces = [u for m, u, _ in rec["requests"] if m == "POST" and "/file?" in u]
                self.assertGreaterEqual(len(pieces), 4)                         # the 4 KB .mrc went in 3 pieces
                self.assertTrue(all(len(b or b"") <= 1500 for m, u, b in rec["requests"]
                                    if m == "POST" and "/file?" in u))
                self.assertFalse([u for m, u, _ in rec["requests"] if m == "POST" and u.endswith("/new/raw/")])
                self.assertEqual(self.wait_done(cid)["state"], "done")
                page.wait_for_selector("text=Analyse these tiles", timeout=20000)   # the page refreshed itself
                self.shot(page, f"predict_convert_done_{width}")
                # a tile upload the same way
                with mock.patch.object(P, "CHUNK_BYTES", 1000):
                    page.goto(self.p.origin + "/predict/new/upload/")
                    page.fill('input[name="sample_label"]', "Browser " + XSS)
                    page.fill('input[name="nm_per_px"]', "1.4724")
                    page.check('input[name="confirm"]')
                    page.set_input_files('input[name="files"]', [str(GOLDEN / "images" / f"000{i}.png")
                                                                 for i in (1, 2)])
                    page.click('button[type="submit"]')
                    page.wait_for_url(re.compile(r".*/predict/ps_\d{8}_[0-9a-f]{6}/$"), timeout=30000)
                self.assertFalse([u for m, u, _ in rec["requests"] if m == "POST" and u.endswith("/new/upload/")])
                self.assertEqual(page.locator("main img").count(), 0)
                # every page: no sideways scroll, nothing foreign, no errors
                for name, path in (("new", "/predict/new/"), ("raw_sessions", "/predict/new/raw/sessions/"),
                                   ("raw_session", f"/predict/new/raw/sessions/{key}/"),
                                   ("convert", f"/predict/convert/{cid}/"), ("upload", "/predict/new/upload/"),
                                   ("list", "/predict/")):
                    with self.subTest(width=width, page=name):
                        page.goto(self.p.origin + path)
                        page.wait_for_load_state("networkidle")
                        page.wait_for_selector("#cj-bar")
                        sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
                        self.assertLessEqual(sw, iw, "horizontal scroll")
                        self.assertEqual(page.locator("main img").count(), 0)
                        self.shot(page, f"predict_{name}_{width}")
                self.check_clean(rec)
            finally:
                ctx.close()

    def test_refusal_shown_in_the_page(self):
        ctx, page, rec = self.session(390, "basic")
        try:
            page.goto(self.p.origin + "/predict/new/upload/")
            page.fill('input[name="sample_label"]', "S")
            page.fill('input[name="nm_per_px"]', "1.4724")
            page.check('input[name="confirm"]')
            bad = self.w.tmp / "bad.png"
            bad.write_bytes(b"MZ not an image at all" * 10)
            page.set_input_files('input[name="files"]', [str(bad)])
            page.click('button[type="submit"]')
            box = page.wait_for_selector("[data-cj-upload-msg]:not([hidden])", timeout=15000)
            self.assertIn("not a PNG or JPEG", box.text_content())
            self.assertTrue(page.is_enabled('button[type="submit"]'))
            self.assertEqual(page.locator("main img").count(), 0)
            self.shot(page, "predict_upload_refused_390")
            # the refused piece is a 400 the browser logs; anything else is not expected
            self.assertEqual(rec["foreign"], [])
            self.assertEqual(rec["errors"], [])
            self.assertEqual([m for m in rec["console"] if "status of 400" not in m], [])
        finally:
            ctx.close()


@unittest.skipUnless(os.environ.get("CJ_PROCESSING_REPO") and os.environ.get("CJ_CONVERT_CONTROL_DIR"),
                     "CJ_PROCESSING_REPO / CJ_CONVERT_CONTROL_DIR not set: known-answer control skipped")
class KnownAnswer(unittest.TestCase):
    """The delivered 3d tiles of one montage, reproduced from raw through the portal (as jobs/tests/test_convert_raw.py
    Control does it directly). The control session folder and the processing repo are only read."""

    @classmethod
    def setUpClass(cls):
        if TestClient is None:
            raise unittest.SkipTest("starlette TestClient (httpx) not installed")
        cls.session = Path(os.environ["CJ_CONVERT_CONTROL_DIR"])
        cls.stem = os.environ.get("CJ_CONVERT_CONTROL_MONTAGE", "grid1-pos1")
        cls.real_repo = os.environ["CJ_PROCESSING_REPO"]
        cls.delivered = next(cls.session.glob("3. Processed images/3d. Tiles no scalebar*"))
        with open(cls.delivered / "MANIFEST.csv", newline="", encoding="utf-8") as f:
            cls.ref = {r["output_name"]: r for r in csv.DictReader(f)}
        cls.mine = {n: r for n, r in cls.ref.items() if r["montage"] == cls.stem.replace("-pos", "_")}
        cls.raw = [cls.session / "2. Raw images" / f"{cls.stem}.mrc",
                   cls.session / "2. Raw images" / f"{cls.stem}.mrc.mdoc",
                   cls.session / "1. Notes + reports" / "sample_key.csv"]
        cls.before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in cls.raw}
        # CJ_TEST_RAW_ROOT=1: the same known answer with CJ_RAW_ROOT set to a second temp folder
        tmp_raw = Path(tempfile.mkdtemp(prefix="cj_rawroot_")) if os.environ.get("CJ_TEST_RAW_ROOT") else None
        cls._tmp_raw = tmp_raw
        cls.w = World(real_repo=cls.real_repo, raw_root=tmp_raw)
        cls.tc = TestClient(APP.create_app(cls.w.cfg), base_url=ORIGIN, follow_redirects=False)
        cls.tc.__enter__()
        cls.tok = {}
        for who in ("admin", "adv2"):
            r = cls.tc.get(f"/login?k={cls.w.keys[who]}")
            cls.tok[who] = r.headers["set-cookie"].split(";", 1)[0].split("=", 1)[1]
        cls.tc.cookies.clear()

    @classmethod
    def tearDownClass(cls):
        cls.tc.__exit__(None, None, None)
        cls.w.close()
        if cls._tmp_raw:
            shutil.rmtree(cls._tmp_raw, ignore_errors=True)

    def h(self, who="admin", **extra):
        return {"Cookie": f"{A.COOKIE}={self.tok[who]}", "Origin": ORIGIN, **extra}

    def check_tiles(self, cid):
        st = wait_for(lambda: (lambda s: s if s.get("state") in ("done", "failed") else None)(
            self.tc.get(f"/predict/api/convert/{cid}", headers=self.h()).json()), 900, every=2,
            what=f"conversion {cid}")
        self.assertEqual(st["state"], "done", (self.w.rawtiles / cid / "convert.log").read_text(
            encoding="utf-8", errors="replace")[-3000:])
        out = self.w.rawtiles / cid / "tiles"
        with open(out / "MANIFEST.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(st["n_tiles"], len(self.mine))
        self.assertEqual(sorted(r["output_name"] for r in rows), sorted(self.mine))
        same = 0
        for r in rows:
            d = self.mine[r["output_name"]]
            got = md5(out / r["output_name"])
            self.assertEqual(got, d["md5"], r["output_name"])
            self.assertEqual(got, md5(self.delivered / r["output_name"]), r["output_name"])
            self.assertEqual((r["nm_per_px"], r["output_size"], r["sample_label"]),
                             (d["nm_per_px"], d["output_size"], d["sample_label"]))
            same += 1
        print(f"\n[known answer] {cid}: {same}/{len(self.mine)} tiles md5-identical to the delivered 3d tiles "
              f"({self.stem}, recipe {self.session.name}, CPU)", file=sys.stderr)
        # a Predict source for the owner (admin here) only
        key = PSRC.source_key(out)
        self.assertEqual(self.tc.get(f"/predict/new/session/{key}/", headers=self.h()).status_code, 200)
        self.assertEqual(self.tc.get(f"/predict/new/session/{key}/", headers=self.h("adv2")).status_code, 404)
        return st

    def test_upload_through_the_portal(self):
        files = [(p.name, p) for p in self.raw]
        r = self.tc.post("/predict/api/upload", headers=self.h(**{"Content-Type": "application/json"}),
                         content=json.dumps({"kind": "raw", "fields": {"session": self.session.name,
                                                                       "recipe": self.session.name},
                                             "files": [{"name": n, "size": p.stat().st_size} for n, p in files]}))
        self.assertEqual(r.status_code, 200, r.text)
        st = r.json()
        self.assertEqual(st["chunk"], P.CHUNK_BYTES)
        uid, n_req = st["upload"], 0
        for name, p in files:
            with open(p, "rb") as f:                                    # read only
                off = 0
                while True:
                    data = f.read(st["chunk"])
                    if not data:
                        break
                    rr = self.tc.post(f"/predict/api/upload/{uid}/file?name={quote(name)}&offset={off}",
                                      content=data, headers=self.h(**{"Content-Type": "application/octet-stream"}))
                    self.assertEqual(rr.status_code, 200, rr.text)
                    off += len(data)
                    n_req += 1
        self.assertGreater(n_req, len(files))                            # the .mrc is over one request's cap
        r = self.tc.post(f"/predict/api/upload/{uid}/create", headers=self.h(**{"Content-Type": "application/json"}),
                         content=b"{}")
        self.assertEqual(r.status_code, 200, r.text)
        cid = CID_RE.match(r.json()["href"]).group(1)
        self.assertTrue(cid.startswith(self.session.name + "_"))
        self.check_tiles(cid)
        self.assertEqual({str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in self.raw}, self.before)

    def test_session_pick_through_the_portal(self):
        # a temp copy of the session's raw montage (the real folder has its 3d tiles, so it is not a raw session)
        scan = self.w.tmp / "scan_known"
        os.environ["CJ_PREDICT_SCAN_ROOTS"] = str(scan)                  # World.close() puts it back
        copy = scan / self.session.name
        (copy / "2. Raw images").mkdir(parents=True)
        (copy / "1. Notes + reports").mkdir()
        for p in self.raw[:2]:
            shutil.copyfile(p, copy / "2. Raw images" / p.name)
        shutil.copyfile(self.raw[2], copy / "1. Notes + reports" / "sample_key.csv")
        PSRC.clear_cache()
        key = next(s["key"] for s in P.raw_sessions(None) if s["folder"] == self.session.name)
        snap = tree(copy)
        r = self.tc.post(f"/predict/new/raw/sessions/{key}/",
                         content=urlencode({"montages": [f"2. Raw images/{self.stem}.mrc.mdoc"],
                                            "recipe": self.session.name}, doseq=True).encode(),
                         headers=self.h(**{"Content-Type": "application/x-www-form-urlencoded"}))
        self.assertEqual(r.status_code, 303, r.text[:800])
        cid = CID_RE.match(r.headers["location"]).group(1)
        self.check_tiles(cid)
        self.assertEqual(tree(copy), snap)                                # the picked folder was only read
        self.assertEqual({str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in self.raw}, self.before)


if __name__ == "__main__":
    unittest.main()
