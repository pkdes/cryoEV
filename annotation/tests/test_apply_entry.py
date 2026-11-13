"""
apply_review_edits.py callable entry (round 10, V4 preview): make_version() is what main() runs, with an explicit
out_root and who, no sys.exit (refusals raise ApplyError). The CLI must stay byte-identical:

  - GOLDEN: the CLI's outputs (labels.coco.json, changes.jsonl, conflicts.csv, version.json), its stdout, its return
    value and its refusal texts, for fixed inputs, hashed BEFORE the refactor (r10-platform 1b3d43c) and pinned here.
    The time (created) and git sha (tool) are frozen; temp paths are replaced by <TMP>. If the output contract changes
    on purpose, recompute with `python annotation/tests/test_apply_entry.py --print-golden` and say why in the commit.
  - the entry and the CLI write the same bytes for the same inputs;
  - the entry refuses with ApplyError (never SystemExit) and never writes on a refusal.

Fixture: test_apply_review_edits.build_synthetic (temp dir); people = a temp roster (not the repo's, so a roster edit
never moves the golden).
"""
import contextlib
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from datetime import datetime as _dt
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_apply_review_edits as tar  # noqa: E402  (module import: its TestCases are not collected twice)
import apply_review_edits as ar  # noqa: E402  (path set up by test_apply_review_edits)
sys.path.append(str(HERE.parent.parent / "tests"))
import cryoev_testkit as TK  # noqa: E402  (shared helpers)

STAMP = "20990101_0000"
PEOPLE = {"people": [{"id": "pujan", "name": "Pujan", "aliases": ["Puj", "September review"]},
                     {"id": "au1", "name": "AU 1", "aliases": []}]}

# sha256 per scenario, computed with the pre-refactor CLI (r10-platform 1b3d43c)
GOLDEN = {  # recomputed 2026-11-13 with class_groups_sha256 masked; pre-refactor code (1b3d43c) gives the same 8 digests

    "all_edits": "9fdc3e1e7b5b6cd396d44820e367a8486f9d875b381d138da8127da8d6ebf18a",
    "dry_run": "cf4b8a8238b833d400f29e9965a88e05c3bdaedc6d2ec02b9a2b4703288b751c",
    "refuse_exists": "76b9dbea0f214cd0370a22dda61290a9db107d1e1a38572a1ad3a1a008ce75bc",
    "refuse_source": "472f14086be522b894fa623b49512c879e63e2af4e92ae7da16e9de6366d76bf",
    "refuse_stamp": "64e1837520499ed913938a011b0d7b3cb278db722dc2395e13ba7f8692a29451",
    "refuse_who": "a788af461213788254bcdc02370bee36fa12063c0e109270a566e2831890e3c9",
    "reviewed_only_who": "4303a52d97fe2fa792f1841e4998dc507e01c95305767c893f91d771043511f4",
    "source_named": "422ad28aa9ac53d158c2f4bacec9925f3873ffefcd93057d083eab35f9641090",
}


class _FrozenDT(_dt):
    @classmethod
    def now(cls, tz=None):
        return _dt(2099, 1, 1, 12, 0, 0)


def _patches(stack):
    stack.enter_context(mock.patch.object(ar, "git_sha", lambda: "abc1234"))
    stack.enter_context(mock.patch.object(ar, "datetime", _FrozenDT))
    stack.enter_context(TK.env_patch())                   # the actor variables removed, restored after


class World:
    """Temp inputs: base, package (data.js), edits, people; a fresh out-root per run."""

    def __init__(self, source=False):
        self.tmp = Path(tempfile.mkdtemp(prefix="apply_entry_")).resolve()
        self.base, self.pkg, self.edits = tar.build_synthetic(self.tmp / "in")
        if source:  # the data.js names its base (what a real package does)
            p = self.pkg / "data.js"
            d = json.loads(p.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
            d["source"] = str(self.base)
            p.write_text("window.REVIEW_DATA = " + json.dumps(d) + ";\n", encoding="utf-8")
        self.people = self.tmp / "people.json"
        self.people.write_text(json.dumps(PEOPLE), encoding="utf-8")

    def norm(self, text: str) -> str:
        for s in (json.dumps(str(self.tmp))[1:-1], str(self.tmp)):
            text = text.replace(s, "<TMP>")
        return text.replace("\\\\", "/").replace("\\", "/")  # the same digest on any OS

    def cli(self, out_root, *extra):
        """main(argv) -> (return value or None, stdout, exit text or None)."""
        buf, ret, exit_text = io.StringIO(), None, None
        argv = ["--base", str(self.base), "--package", str(self.pkg), "--edits", str(self.edits),
                "--out-root", str(out_root), "--stamp", STAMP, "--people", str(self.people), *extra]
        with contextlib.ExitStack() as st:
            _patches(st)
            with contextlib.redirect_stdout(buf):
                try:
                    ret = ar.main(argv)
                except SystemExit as e:
                    exit_text = str(e.code)
        return ret, buf.getvalue(), exit_text

    def files(self, out: Path) -> dict:
        """Output files, normalised. version.json's class_groups_sha256 hashes annotation/class_groups.yaml's raw bytes,
        which differ by checkout (core.autocrlf: LF in one tree, CRLF in another) though git's content is the same; it is
        checked against the real file here and then masked, so the digest doesn't depend on the checkout."""
        out_files = {}
        for p in sorted(Path(out).glob("*")):
            if not p.is_file():
                continue
            text = p.read_text(encoding="utf-8")
            if p.name == "version.json":
                v = json.loads(text)
                assert v.get("class_groups_sha256") == ar.sha256_file(ar.CLASS_GROUPS), "class_groups_sha256 is not the file's"
                text = re.sub(r'"class_groups_sha256": "[0-9a-f]{64}"', '"class_groups_sha256": "<CG>"', text)
            out_files[p.name] = self.norm(text)
        return out_files

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def mask_cg(v):
    """The return value carries class_groups_sha256 too: check it is the real file's, then mask it (see World.files)."""
    if isinstance(v, dict) and "class_groups_sha256" in v:
        assert v["class_groups_sha256"] == ar.sha256_file(ar.CLASS_GROUPS), "class_groups_sha256 is not the file's"
        v = {**v, "class_groups_sha256": "<CG>"}
    return v


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def scenarios() -> dict:
    """name -> canonical result (files, stdout, return, exit text), each from a fresh temp world."""
    out = {}

    def one(name, extra=(), source=False, prep=None):
        w = World(source)
        try:
            root = w.tmp / "labels"
            if prep:
                prep(w, root)
            ret, log, exit_text = w.cli(root, *extra)
            lv = root / f"lv_{STAMP}"
            files = w.files(lv) if lv.is_dir() and not (prep and exit_text) else {}
            out[name] = {"files": files, "stdout": w.norm(log), "exit": w.norm(exit_text) if exit_text else None,
                         "ret": mask_cg(json.loads(w.norm(json.dumps(ret, sort_keys=True)))) if ret is not None else None}
        finally:
            w.cleanup()

    one("reviewed_only_who", ("--who", "pujan"))
    one("all_edits", ("--all-edits",))
    one("source_named", ("--who", "au1"), source=True)
    one("dry_run", ("--dry-run",))
    one("refuse_who", ("--who", "alyssia"))
    one("refuse_exists", (), prep=lambda w, root: (root / f"lv_{STAMP}").mkdir(parents=True))
    one("refuse_stamp", ("--stamp", "2099"))

    def other_source(w, root):
        p = w.pkg / "data.js"
        d = json.loads(p.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
        d["source"] = str(w.tmp / "elsewhere.coco.json")
        p.write_text("window.REVIEW_DATA = " + json.dumps(d) + ";\n", encoding="utf-8")
    one("refuse_source", (), prep=other_source)
    return out


class Golden(unittest.TestCase):
    def test_cli_byte_identical_to_pre_refactor(self):
        got = {k: digest(v) for k, v in scenarios().items()}
        self.assertEqual(sorted(got), sorted(GOLDEN))
        for k in GOLDEN:
            self.assertEqual(got[k], GOLDEN[k], f"CLI output changed for scenario {k}")


class Entry(unittest.TestCase):
    def setUp(self):
        self.w = World(source=True)
        self.addCleanup(self.w.cleanup)

    def entry(self, out_root, **kw):
        args = dict(out_root=out_root, who="pujan", people_path=self.w.people, edits_dir=self.w.edits,
                    stamp=STAMP, log=lambda s: None)
        args.update(kw)
        with contextlib.ExitStack() as st:
            _patches(st)
            with contextlib.redirect_stdout(io.StringIO()):
                return ar.make_version(self.w.base, self.w.pkg, **args)

    def test_entry_and_cli_write_the_same_bytes(self):
        ret_cli, _, err = self.w.cli(self.w.tmp / "cli", "--who", "pujan")
        self.assertIsNone(err)
        ret_entry = self.entry(self.w.tmp / "entry")
        a, b = self.w.tmp / "cli" / f"lv_{STAMP}", self.w.tmp / "entry" / f"lv_{STAMP}"
        names = sorted(p.name for p in a.iterdir())
        self.assertEqual(names, ["changes.jsonl", "conflicts.csv", "labels.coco.json", "version.json"])
        self.assertEqual(names, sorted(p.name for p in b.iterdir()))
        for n in names:
            self.assertEqual((a / n).read_bytes(), (b / n).read_bytes(), n)
        self.assertEqual(ret_cli, ret_entry)

    def test_refusals_raise_apply_error_and_write_nothing(self):
        root = self.w.tmp / "labels"
        cases = [dict(who="alyssia"), dict(stamp="2099"), dict(scope="nope")]
        for kw in cases:
            with self.subTest(**kw):
                with self.assertRaises(ar.ApplyError) as cm:
                    self.entry(root, **kw)
                self.assertNotIsInstance(cm.exception, SystemExit)
                self.assertTrue(str(cm.exception).startswith("[ERROR] "), str(cm.exception))
                self.assertFalse(root.exists())
        # a refusal inside apply() (data.js names another base) also comes out as ApplyError
        p = self.w.pkg / "data.js"
        d = json.loads(p.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
        d["source"] = str(self.w.tmp / "elsewhere.coco.json")
        p.write_text("window.REVIEW_DATA = " + json.dumps(d) + ";\n", encoding="utf-8")
        with self.assertRaises(ar.ApplyError) as cm:
            self.entry(root)
        self.assertIn("data.js was built from", str(cm.exception))
        self.assertFalse(root.exists())

    def test_explicit_who_and_out_root_no_env(self):
        import os
        with mock.patch.dict(os.environ, {"CRYOEV_FOR": "au1"}):
            ver = self.entry(self.w.tmp / "x", who=None)
        self.assertIsNone(ver["created_by"])          # the entry never falls back to $CRYOEV_FOR: who is explicit
        self.assertTrue((self.w.tmp / "x" / f"lv_{STAMP}" / "version.json").is_file())


if __name__ == "__main__":
    if "--print-golden" in sys.argv:
        for k, v in sorted(scenarios().items()):
            print(f'    "{k}": "{digest(v)}",')
    else:
        unittest.main()
