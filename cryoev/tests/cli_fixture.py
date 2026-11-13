"""A temp platform for the cryoev CLI tests: laptop root (registry = registry/samples, the golden prediction set,
a lab review package with its `_work_` folder, logs/), CryoAI root, an empty processing root, a home folder, a
queue share, a services config and a feedback file. Every CRYOEV_* / CJ_* variable from the caller's env is
cleared first; tests set CRYOEV_VIA / CRYOEV_FOR themselves. Never touches real data."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cryoev import __main__ as CLI  # noqa: E402
sys.path.append(str(REPO / "tests"))
import cryoev_testkit as TK  # noqa: E402  (shared helpers)

GOLDEN = REPO / "tests" / "fixtures" / "cj_golden" / "predict" / "ps_20261105_abc123"
PS_ID = "ps_20261105_abc123"                     # = registry/samples/prediction_set.jsonl
MODEL = "v5_evnonev_y8m_20261029"
PKG = "lab_compare_cli_20261112"           # a package id the importer recognises (dated)
REVIEWER = "au1"                                  # who made the fixture decision: must never be printed


class Platform(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cryoev_cli_"))
        t = self.tmp
        self.lap, self.cai, self.home = t / "lap", t / "cai", t / "home"
        self.reg, self.share, self.out = self.lap / "registry", t / "share", t / "out"
        for d in (self.reg, self.lap / "logs", self.lap / "predict", self.lap / "review", self.cai / "training outputs",
                  self.cai / "training images", t / "noproc", self.home, self.share / "q"):
            d.mkdir(parents=True, exist_ok=True)
        TK.write_registry(self.reg, samples=True, skip=("label_change.jsonl",))   # its rows have no id
        shutil.copytree(GOLDEN, self.lap / "predict" / PS_ID)
        self.feedback = self.lap / "logs" / "portal_feedback.jsonl"
        self.services = t / "services.json"
        self.services.write_text(json.dumps({"run_dir": str(t / "run"), "services": [
            {"name": "demo", "enabled": False, "port": None, "argv": ["python", "-c", "pass"]}]}), encoding="utf-8")
        env = TK.clean_env(strip_all=True, drop=("CJ_*",))   # CRYOEV_* and CJ_* wide: the CLI reads both
        env.update(CRYOEV_LAPTOP_ROOT=str(self.lap), CRYOEV_REGISTRY=str(self.reg), CRYOEV_CRYOAI_ROOT=str(self.cai),
                   CRYOEV_PROCESSING_ROOT=str(t / "noproc"), CRYOEV_SERVICES_CONFIG=str(self.services),
                   CJ_FEEDBACK_FILE=str(self.feedback), USERPROFILE=str(self.home), HOME=str(self.home))
        self.env = mock.patch.dict(os.environ, env, clear=True)
        self.env.start()
        from portal import feedback_store as FS
        FS.create(self.feedback, "pujan", "bug", "the gallery button does nothing")
        self._package()
        self._tiles()

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _package(self):
        pdir = self.lap / "review" / PKG
        (pdir / "edits").mkdir(parents=True)
        data = {"id": PKG, "source": "lv_20261030_0846/labels.coco.json", "images": [
            {"file": "a.png", "gt": [{"id": 1, "u": 1}, {"id": 2, "u": 1}], "pr": [{"u": 1}]},
            {"file": "b.png", "gt": [{"id": 3, "u": 1}], "pr": []}]}
        (pdir / "data.js").write_text("window.DATA = " + json.dumps(data) + ";", encoding="utf-8")
        (pdir / "edits" / "a.png.json").write_text(json.dumps({"file": "a.png", "edit": {
            "dec": {"g1": {"v": "keep", "who": REVIEWER, "t": 1}}, "reviewed": False}}), encoding="utf-8")
        work = self.lap / "review" / f"_work_{PKG}"
        work.mkdir()
        (work / "decoys.json").write_text(json.dumps({"a.png": ["g2"]}), encoding="utf-8")

    def _tiles(self):
        """The golden tiles under their original names in src/ with a MANIFEST.csv (as jobs/tests/test_predict_set)."""
        self.src = self.tmp / "src"
        self.src.mkdir()
        with open(GOLDEN / "images.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            shutil.copyfile(GOLDEN / r["file"], self.src / r["orig_name"])
        (self.src / "MANIFEST.csv").write_text("output_name,sample_label,nm_per_px\n" + "".join(
            f"{r['orig_name']},{r['sample_label']},{r['nm_per_px']}\n" for r in rows), encoding="utf-8")
        self.predict_args = ["--from-dir", str(self.src), "--manifest", str(self.src / "MANIFEST.csv"),
                             "--model", MODEL, "--owner", "pujan", "--sample-label-from", "manifest:sample_label",
                             "--nm-from-manifest", "--nm-confirmed-by", "pujan"]

    # ---- helpers
    def cli(self, *argv) -> tuple:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = CLI.main([str(a) for a in argv])
        return rc, out.getvalue(), err.getvalue()

    def snapshot(self) -> dict:
        """sha256 of every file under the temp platform (laptop root + registry, --out, action log, queue, home)."""
        return {p.relative_to(self.tmp).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(self.tmp.rglob("*")) if p.is_file()}

    def action_log(self) -> list:
        p = self.lap / "logs" / "agent_actions.jsonl"
        return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []

    def as_agent(self):
        return mock.patch.dict(os.environ, {"CRYOEV_VIA": "claude", "CRYOEV_FOR": "pujan"})
