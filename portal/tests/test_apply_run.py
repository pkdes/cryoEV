"""
The Apply button (round 11, #67; portal/modules/applyreview.py run_apply / run_retry, POST /apply/run), fixtures only:
test_applyreview's temp CryoAI-laptop (a label set, review packages on it, a temp services config) and a temp registry.
Never the real data or ~/.cryoev.

  - happy path: the label set exists (the previewed bytes exactly), is recorded in the registry and the history page;
    NOTHING else pending is appended (control: another unregistered label set and a production restatement that a
    full import WOULD append); nothing is written outside the temp roots;
  - every refusal writes nothing: typed name, no / failed preview, unmapped who, not eligible, decisions changed after
    the preview, the frozen copy changed, already applied, another label set on (or replacing) the same base;
  - fault injection mid-apply: no lv_ folder, no partial; a killed run's leftover partial blocks while young (it may
    be a CLI run) and is cleaned on the next run once older than PARTIAL_MIN_AGE_S;
  - a crash after the rename but before the apply_runs line: refused as "made here but not recorded", Retry records
    it; a run folder the log names is never removed; a decisions-copy I/O error is a clean failure; a package
    rebuilt since the preview (data.js sha256) is refused; registry_modules is a cached return in a request;
  - an import failure (and a history failure) -> Retry records it, once; a second Retry has nothing to do;
  - a concurrent press (the OS lock held by another process; the in-process lock) is refused; control: free -> runs;
  - HTTP: non-admin 403, bearer refused, stale form token 403, no Origin 403; the typed-name confirm, the result page
    with Retry, one self-logged audit line.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock
from urllib.parse import urlencode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cjtest as T  # noqa: E402
import test_applyreview as TAR  # noqa: E402  (module import: its TestCases are not collected twice)
from portal import auth as A  # noqa: E402
from portal.modules import applyreview as M  # noqa: E402
from annotation import apply_review_edits as AR  # noqa: E402  (the module object the portal uses)

LV = TAR.LV                                     # the base label set: lv_20261101_1200
NOW = datetime(2026, 11, 13, 10, 50)
NEW = "lv_20261113_1050"
OTHER = "lv_20261101_0900"                      # pending: an unregistered label set on another base


def tree(root: Path) -> dict:
    return {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def jsonl(path: Path) -> list:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


class World(TAR.World):
    """test_applyreview's world plus pending registry work: an unregistered label set on another base, and a
    production model that is not production now (a full import would restate it)."""

    def __init__(self):
        super().__init__()
        self.labels = self.lap / "labels"
        self.reg = self.env.registry
        other = self.labels / OTHER
        other.mkdir()
        shutil.copy(self.base, other / "labels.coco.json")
        (other / "version.json").write_text(json.dumps({"id": OTHER, "base": "roboflow_pull_20261101_0800",
                                                        "pkg": "lab_elsewhere", "supersedes": None}), encoding="utf-8")
        prod = M.registry_modules().importers.PRODUCTION[0]
        (self.reg / "model.jsonl").write_text(json.dumps({"schema": "model/1", "id": prod, "status": "candidate",
                                                          "logged": "2026-11-01T00:00:00"}) + "\n", encoding="utf-8")

    def lvs(self) -> list:
        return sorted(p.name for p in self.labels.iterdir())


class Base(unittest.TestCase):
    def setUp(self):
        self.w = World()
        self.addCleanup(self.w.cleanup)
        p = mock.patch.dict(os.environ, {"CRYOEV_SERVICES_CONFIG": str(self.w.services)})
        p.start()
        self.addCleanup(p.stop)
        self.cfg = self.w.cfg()

    def row(self, pkg="lab_ok"):
        served, problem = M.served_packages()
        return next((r for r in M.packages(self.cfg, {}, served, problem) if r["id"] == pkg), None)

    def preview(self, pkg="lab_ok"):
        ts, info = M.run_preview(self.cfg, self.row(pkg), "pujan")
        self.assertNotIn("error", info, info.get("error"))
        return ts

    def apply(self, ts, typed="lab_ok", who="pujan", pkg="lab_ok", now=NOW):
        return M.run_apply(self.cfg, lambda: self.row(pkg), pkg, ts, typed, who, {}, now)

    def assert_nothing_written(self, before_labels, before_reg):
        self.assertEqual(self.w.lvs(), before_labels)
        self.assertEqual(tree(self.w.reg), before_reg)
        self.assertFalse(M.runs_log(self.cfg).exists())
        self.assertFalse((Path(self.cfg.state_dir) / M.APPLY_DIR / NEW).exists())


class HappyPath(Base):
    def test_apply_records_only_this_label_set(self):
        ts = self.preview()
        imps = M.registry_modules().importers
        # control: a full label-version import WOULD append the other pending label set, and the full import would
        # restate production; the targeted one must do neither
        full = imps.Importer(self.w.reg, laptop=self.w.lap, processing=None, via=None)
        full.label_versions()
        self.assertIn(OTHER, [r["id"] for _, r in full.out])
        self.assertEqual(full.write(dry_run=True)["label_version"], 1)
        cur = M.registry_modules().registry.current(root=self.w.reg)
        self.assertTrue(full._restate_production(imps.PRODUCTION[0], imps.PRODUCTION[1], cur))
        model_before = (self.w.reg / "model.jsonl").read_bytes()
        before = tree(self.w.tmp)
        self.assertIn(f"laptop/labels/{LV}/labels.coco.json", before)               # control: the hash sees them

        status, rec = self.apply(ts)
        self.assertEqual(status, "done", rec)
        self.assertEqual((rec["lv"], rec["steps"]["apply"], rec["steps"]["import"], rec["steps"]["history"]),
                         (NEW, "ok", "ok", "ok"), rec)
        self.assertEqual(rec["steps"]["appended"], 1)
        out = self.w.labels / NEW
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         ["changes.jsonl", "conflicts.csv", "labels.coco.json", "version.json"])
        ver = json.loads((out / "version.json").read_text(encoding="utf-8"))
        run_edits = Path(self.cfg.state_dir) / M.APPLY_DIR / NEW / "edits"
        self.assertEqual((ver["base"], ver["pkg"], ver["created_by"], ver["created_via"], ver["review_scope"]),
                         (LV, "lab_ok", "pujan", None, "reviewed_only"))
        self.assertEqual(Path(ver["edits_dir"]), run_edits.resolve())                # the kept frozen copy
        self.assertEqual(M.edits_sha256(run_edits), rec["edits_sha256"])
        scratch = M.scratch_root(self.cfg) / "lab_ok" / ts / "labels"
        prev_lv = json.loads((M.scratch_root(self.cfg) / "lab_ok" / ts / "preview.json").read_text("utf-8"))["lv"]
        self.assertEqual((out / "labels.coco.json").read_bytes(),                   # exactly what was previewed
                         (scratch / prev_lv / "labels.coco.json").read_bytes())
        self.assertFalse([p for p in self.w.labels.iterdir() if p.name.startswith(".partial_")])
        # the registry: one label_version line (this lv, no via), nothing else appended, no production restatement
        lines = jsonl(self.w.reg / "label_version.jsonl")
        self.assertEqual([x["id"] for x in lines], [NEW])
        self.assertNotIn("via", lines[0])
        self.assertEqual((self.w.reg / "model.jsonl").read_bytes(), model_before)
        self.assertIn(NEW, M.registry_modules().registry.current("label_version", root=self.w.reg))
        self.assertIn(NEW, (self.w.reg / "history.html").read_text(encoding="utf-8"))
        after = tree(self.w.tmp)
        changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
        allowed = re.compile(r"(laptop/labels/%s/[^/]+|registry/(label_version\.jsonl|history\.html|\.registry\.lock|"
                             r"\.upload_session\.lock|FORMAT\.json)|state/.+)" % NEW)
        self.assertTrue(changed)
        self.assertEqual(sorted(k for k in changed if not allowed.fullmatch(k)), [])  # nothing outside these
        self.assertEqual([r["lv"] for r in jsonl(M.runs_log(self.cfg))], [NEW])
        # the same decisions again: refused (already applied), nothing more written
        lv_before, reg_before = self.w.lvs(), tree(self.w.reg)
        status, d = self.apply(ts, now=datetime(2026, 11, 13, 11, 0))
        self.assertEqual(status, "refused")
        self.assertIn("already applied as " + NEW, d["why"])
        self.assertEqual((self.w.lvs(), tree(self.w.reg)), (lv_before, reg_before))


class Refusals(Base):
    def test_each_refusal_writes_nothing(self):
        ts = self.preview()
        lv0, reg0 = self.w.lvs(), tree(self.w.reg)
        bad_preview = M.scratch_root(self.cfg) / "lab_ok" / "20261112T000000Z"
        bad_preview.mkdir(parents=True)
        (bad_preview / "preview.json").write_text(json.dumps({"error": "x"}), encoding="utf-8")
        cases = [("typed name", dict(ts=ts, typed="lab_okx"), "typed name"),
                 ("empty name", dict(ts=ts, typed=""), "typed name"),
                 ("no preview", dict(ts="20261111T000000Z"), "no such preview"),
                 ("failed preview", dict(ts="20261112T000000Z"), "preview failed"),
                 ("unmapped who", dict(ts=ts, who=None), "people.json"),
                 ("not eligible", dict(ts=ts, pkg="lab_served", typed="lab_served"), "not ready"),
                 ("compare package", dict(ts=ts, pkg="cmp_audit", typed="cmp_audit"), "not ready")]
        for name, kw, want in cases:
            with self.subTest(name):
                status, d = self.apply(**kw)
                self.assertEqual(status, "refused")
                self.assertIn(want, d["why"])
                self.assert_nothing_written(lv0, reg0)
        # decisions changed after the preview
        rec = self.w.review / "lab_ok" / "edits" / "c.png.json"
        old = rec.read_bytes()
        rec.write_bytes(old.replace(b'"reviewed": true', b'"reviewed": false'))
        status, d = self.apply(ts)
        self.assertEqual(status, "refused")
        self.assertIn("decisions changed since this preview", d["why"])
        self.assert_nothing_written(lv0, reg0)
        rec.write_bytes(old)
        # the preview's frozen copy changed on disk
        frozen = M.scratch_root(self.cfg) / "lab_ok" / ts / "edits" / "c.png.json"
        fold = frozen.read_bytes()
        frozen.write_bytes(fold + b" ")
        status, d = self.apply(ts)
        self.assertIn("frozen decisions copy", d["why"])
        self.assert_nothing_written(lv0, reg0)
        frozen.write_bytes(fold)
        # another label set builds on, or replaces, the same base
        for key in ("base", "supersedes"):
            with self.subTest(key):
                fork = self.w.labels / "lv_20261102_0000"
                fork.mkdir()
                (fork / "version.json").write_text(json.dumps({"base": "roboflow_pull_x", "supersedes": None,
                                                               key: LV}), encoding="utf-8")
                status, d = self.apply(ts)
                self.assertEqual(status, "refused")
                self.assertIn("lv_20261102_0000 already builds on (or replaces) " + LV, d["why"])
                shutil.rmtree(fork)
                self.assert_nothing_written(lv0, reg0)
        # control: with everything back, the apply runs
        self.assertEqual(self.apply(ts)[0], "done")

    def test_same_minute_label_set_exists(self):
        ts = self.preview()
        (self.w.labels / NEW).mkdir()                                       # made by hand this minute (no version)
        status, d = self.apply(ts)
        self.assertEqual(status, "refused")
        self.assertIn("try again in a minute", d["why"])


class Faults(Base):
    def test_fault_mid_apply_leaves_no_label_set(self):
        ts = self.preview()
        lv0, reg0 = self.w.lvs(), tree(self.w.reg)
        seen = []

        def boom(conflicts):                                                # labels.coco.json already written
            seen.append(sorted(p.name for p in self.w.labels.iterdir()))
            raise OSError("disk full")
        with mock.patch.object(AR, "conflicts_csv", boom):
            status, d = self.apply(ts)
        self.assertEqual(status, "failed")
        self.assertIn("disk full", d["why"])
        self.assertIn(".partial_" + NEW, seen[0])                           # it was being written ...
        self.assertNotIn(NEW, seen[0])                                      # ... never visible as the lv_ folder
        self.assert_nothing_written(lv0, reg0)                              # and the partial is gone
        # a killed run (its cleanup never ran) leaves the partial: the next run cleans it under the lock
        with mock.patch.object(AR, "conflicts_csv", boom), \
                mock.patch.object(AR, "shutil", mock.Mock(rmtree=lambda *a, **k: None)):
            self.assertEqual(self.apply(ts)[0], "failed")
        part = self.w.labels / (".partial_" + NEW)
        self.assertIn(part.name, self.w.lvs())
        self.assertNotIn(NEW, self.w.lvs())
        # a YOUNG partial may be a CLI apply still writing (the CLI takes no lock): it blocks and is left alone
        status, d = self.apply(ts)
        self.assertEqual(status, "refused")
        self.assertIn(f"another apply seems to be running (partial {part.name}, 0 min old)", d["why"])
        self.assertTrue(part.is_dir())
        self.assertFalse((Path(self.cfg.state_dir) / M.APPLY_DIR / NEW).exists())
        old = part.stat().st_mtime - M.PARTIAL_MIN_AGE_S - 60                # older than the limit: a dead run's
        os.utime(part, (old, old))
        status, rec = self.apply(ts)
        self.assertEqual(status, "done", rec)
        self.assertEqual(self.w.lvs(), sorted([*lv0, NEW]))                 # partial cleaned, lv made

    def test_crash_after_rename_then_retry(self):
        """A crash between the rename and the apply_runs line: the next press says so and offers Retry, which
        records it (import + history + its "apply" line); then the (pkg, sha) rule applies as usual."""
        ts = self.preview()

        class Crash(BaseException):
            pass
        with mock.patch.object(M, "record_steps", side_effect=Crash()), self.assertRaises(Crash):
            self.apply(ts)
        self.assertIn(NEW, self.w.lvs())
        self.assertFalse(M.runs_log(self.cfg).exists())
        self.assertFalse((self.w.reg / "label_version.jsonl").exists())
        status, d = self.apply(ts, now=datetime(2026, 11, 13, 11, 0))
        self.assertEqual((status, d.get("unrecorded")), ("refused", NEW))
        self.assertIn(f"{NEW} was made here but not recorded", d["why"])
        page = M.render_confirm("lab_ok", ts, {}, None, "pujan", "tok", d["why"], d["unrecorded"])
        self.assertIn(f'name="op" value="retry"><input type="hidden" name="lv" value="{NEW}"', page)
        self.assertEqual(self.w.lvs().count("lv_20261113_1100"), 0)
        status, rec = M.run_retry(self.cfg, NEW)
        self.assertEqual((status, rec["op"], rec.get("recovered"), rec["steps"]["import"], rec["steps"]["history"]),
                         ("done", "apply", True, "ok", "ok"))
        self.assertEqual([x["id"] for x in jsonl(self.w.reg / "label_version.jsonl")], [NEW])
        self.assertEqual([(r["op"], r["lv"], r["pkg"]) for r in jsonl(M.runs_log(self.cfg))],
                         [("apply", NEW, "lab_ok")])
        status, d = self.apply(ts, now=datetime(2026, 11, 13, 11, 0))
        self.assertIn("already applied as " + NEW, d["why"])
        # control: an lv on the same base that this page did NOT make is the plain refusal, with no Retry
        self.assertEqual(M.run_retry(self.cfg, OTHER)[0], "refused")
        self.assertFalse(M.made_here(self.cfg, OTHER))

    def test_run_dir_named_by_the_log_is_never_removed(self):
        ts = self.preview()
        run_dir = Path(self.cfg.state_dir) / M.APPLY_DIR / NEW
        (run_dir / "edits").mkdir(parents=True)
        (run_dir / "keep.txt").write_text("named", encoding="utf-8")
        M.append_run(self.cfg, {"op": "retry", "pkg": "lab_other", "lv": NEW, "steps": {}})
        status, d = self.apply(ts)
        self.assertEqual(status, "refused")
        self.assertIn("is named by the apply log", d["why"])
        self.assertEqual((run_dir / "keep.txt").read_text(encoding="utf-8"), "named")
        self.assertNotIn(NEW, self.w.lvs())

    def test_decisions_copy_io_failure_is_a_clean_failure(self):
        ts = self.preview()
        lv0, reg0 = self.w.lvs(), tree(self.w.reg)
        (Path(self.cfg.state_dir) / M.APPLY_DIR).write_text("a file where the folder goes", encoding="utf-8")
        status, d = self.apply(ts)
        self.assertEqual(status, "failed")
        self.assertIn("the decisions copy could not be made", d["why"])
        self.assertEqual((self.w.lvs(), tree(self.w.reg)), (lv0, reg0))
        (Path(self.cfg.state_dir) / M.APPLY_DIR).unlink()
        self.assertEqual(self.apply(ts)[0], "done")                         # control: the folder can be made

    def test_package_rebuilt_since_preview(self):
        ts = self.preview()
        info = json.loads((M.scratch_root(self.cfg) / "lab_ok" / ts / "preview.json").read_text(encoding="utf-8"))
        data_js = self.w.review / "lab_ok" / "data.js"
        self.assertEqual(info["data_js_sha256"], hashlib.sha256(data_js.read_bytes()).hexdigest())
        lv0, reg0 = self.w.lvs(), tree(self.w.reg)
        old = data_js.read_bytes()
        data_js.write_bytes(old + b"\n")                                    # rebuilt: same decisions, other data.js
        status, d = self.apply(ts)
        self.assertEqual(status, "refused")
        self.assertIn("the package was rebuilt since the preview", d["why"])
        self.assert_nothing_written(lv0, reg0)
        data_js.write_bytes(old)
        self.assertEqual(self.apply(ts)[0], "done")                         # control: unchanged -> it runs

    def test_registry_modules_cached_and_not_retried_in_a_request(self):
        mods = M.registry_modules()
        self.assertIs(M.registry_modules(), mods)                           # a cached return
        with mock.patch.object(M, "_reg_mods", None), mock.patch.object(M, "_reg_error", "ImportError: x"), \
                mock.patch.object(M, "_load_path", side_effect=AssertionError("loaded in a request")):
            with self.assertRaises(RuntimeError):
                M.registry_modules()
            steps = M.record_steps(self.cfg, NEW, self.w.labels)
        self.assertIn("could not be loaded at startup", steps["import"])

    def test_cli_entry_refuses_an_existing_partial(self):
        """make_version without the portal's lock: an existing partial (another run, or a killed one) is refused."""
        row = self.row()
        part = self.w.labels / (".partial_" + NEW)
        part.mkdir()
        with self.assertRaises(AR.ApplyError) as cm:
            AR.make_version(row["base"], row["dir"], row["dir"] / "edits", out_root=self.w.labels, who="pujan",
                            stamp=NEW[3:], log=lambda s: None)
        self.assertIn("exists (another run", str(cm.exception))
        self.assertFalse((self.w.labels / NEW).exists())

    def test_import_failure_then_retry(self):
        ts = self.preview()
        mods = M.registry_modules()
        with mock.patch.object(mods.importers, "import_label_versions", side_effect=RuntimeError("registry busy")):
            status, rec = self.apply(ts)
        self.assertEqual(status, "done")
        self.assertIn("failed (RuntimeError: registry busy)", rec["steps"]["import"])
        self.assertEqual(rec["steps"]["history"], "not run (the import failed)")
        self.assertTrue((self.w.labels / NEW / "version.json").is_file())    # the label set stays
        self.assertFalse((self.w.reg / "label_version.jsonl").exists())
        self.assertFalse(M.recorded(M.run_state(self.cfg, NEW)))
        status, rec = M.run_retry(self.cfg, NEW)
        self.assertEqual((status, rec["op"], rec["steps"]["import"], rec["steps"]["history"]), ("done", "retry", "ok",
                                                                                               "ok"))
        self.assertEqual([x["id"] for x in jsonl(self.w.reg / "label_version.jsonl")], [NEW])
        self.assertTrue(M.recorded(M.run_state(self.cfg, NEW)))
        status, d = M.run_retry(self.cfg, NEW)                              # nothing left to do
        self.assertEqual(status, "refused")
        self.assertIn("nothing to retry", d["why"])
        self.assertEqual(M.run_retry(self.cfg, "lv_19990101_0000")[0], "refused")
        self.assertEqual(M.run_retry(self.cfg, "../x")[0], "refused")

    def test_history_failure_then_retry_is_idempotent(self):
        ts = self.preview()
        with mock.patch.object(M.registry_modules().history, "build", side_effect=SystemExit("refusing to write")):
            status, rec = self.apply(ts)
        self.assertEqual((rec["steps"]["import"], rec["steps"]["history"][:6]), ("ok", "failed"))
        status, rec = M.run_retry(self.cfg, NEW)
        self.assertEqual((status, rec["steps"]["appended"], rec["steps"]["history"]), ("done", 0, "ok"))
        self.assertEqual([x["id"] for x in jsonl(self.w.reg / "label_version.jsonl")], [NEW])   # still one line


class Lock(Base):
    def test_concurrent_press_refused(self):
        ts = self.preview()
        lv0, reg0 = self.w.lvs(), tree(self.w.reg)
        state = Path(self.cfg.state_dir)
        holder = subprocess.Popen(
            [sys.executable, "-c", "import sys; from data_utils.oslock import acquire_lock; "
             "l = acquire_lock(sys.argv[1]); print('held' if l else 'no', flush=True); sys.stdin.read()",
             str(state / M.APPLY_LOCK)], cwd=str(T.REPO), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(holder.stdout.readline().strip(), "held")
            self.assertEqual(self.apply(ts), ("busy", {}))                   # another process holds the OS lock
            self.assertEqual(M.run_retry(self.cfg, NEW), ("busy", {}))
        finally:
            holder.stdin.close()
            holder.wait(10)
            holder.stdout.close()
        self.assert_nothing_written(lv0, reg0)
        self.assertTrue(M._applying.acquire(blocking=False))                 # another press in this process
        try:
            self.assertEqual(self.apply(ts), ("busy", {}))
        finally:
            M._applying.release()
        self.assert_nothing_written(lv0, reg0)
        self.assertEqual(self.apply(ts)[0], "done")                          # control: both free -> it runs
        self.assertFalse(M._applying.locked())


class Portal(unittest.TestCase):
    """The route: admin + form token + Origin; never bearer; the typed-name confirm; the result page and Retry."""

    @classmethod
    def setUpClass(cls):
        cls.w = World()
        cls.envp = mock.patch.dict(os.environ, {"CRYOEV_SERVICES_CONFIG": str(cls.w.services)})
        cls.envp.start()
        cls.p = T.Portal(cls.w.cfg())
        cls.tok = {u: T.login(cls.p.port, k) for u, k in cls.w.keys.items()}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.envp.stop()
        cls.w.cleanup()

    def get(self, user, path):
        return T.request(self.p.port, "GET", path, T.cookie(self.tok[user]))

    def post(self, user, path, fields, origin=True, headers=None):
        h = {**T.cookie(self.tok[user]), "Content-Type": "application/x-www-form-urlencoded", **(headers or {})}
        if origin:
            h["Origin"] = self.p.origin
        return T.request(self.p.port, "POST", path, h, urlencode(fields).encode())

    def token_on(self, user, path):
        m = re.search(r'name="t" value="([0-9a-f]+)"', self.get(user, path)[2].decode())
        return m.group(1) if m else ""

    def lvs(self):
        return sorted(p.name for p in (self.w.lap / "labels").iterdir())

    def test_1_refused_callers_write_nothing(self):
        lv0 = self.lvs()
        fields = {"op": "apply", "pkg": "lab_ok", "ts": "20261112T000000Z", "name": "lab_ok"}
        st, _, _ = self.post("adv", "/apply/run", {**fields, "t": "x"})                 # non-admin
        self.assertEqual(st, 403)
        self.assertEqual(self.get("adv", "/apply/done/lv_20261113_1050")[0], 403)
        tok, _ = A.create_token(self.w.env.tokens, "pujan", "claude", 30, roster=self.w.env.roster)
        st, _, _ = T.request(self.p.port, "POST", "/apply/run", {**T.bearer(tok), "Origin": self.p.origin,
                             "Content-Type": "application/x-www-form-urlencoded"}, urlencode(fields).encode())
        self.assertIn(st, (401, 403))                                                    # bearer: never
        st, _, _ = T.request(self.p.port, "GET", "/apply/done/lv_20261113_1050", T.bearer(tok))
        self.assertIn(st, (401, 403))
        st, _, _ = self.post("pujan", "/apply/run", {**fields, "t": "stale"})          # stale form token
        self.assertEqual(st, 403)
        t = self.token_on("pujan", "/apply/")
        st, _, body = self.post("pujan", "/apply/run", {**fields, "t": t}, origin=False)  # no Origin (the Gate)
        self.assertEqual(st, 403)
        st, _, body = self.post("pujan", "/apply/run", {**fields, "t": t, "op": "nope"})
        self.assertEqual(st, 400)
        self.assertEqual(self.lvs(), lv0)
        self.assertFalse(M.runs_log(self.w.cfg()).exists())

    def test_2_confirm_apply_retry(self):
        t = self.token_on("pujan", "/apply/")
        st, h, _ = self.post("pujan", "/apply/preview", {"t": t, "pkg": "lab_ok"})
        self.assertEqual(st, 303)
        run_page = T.hget(h, "location")
        ts = run_page.rsplit("/", 1)[1]
        page = self.get("pujan", run_page)[2].decode()
        self.assertIn("Type the package name to confirm", page)
        self.assertIn("Apply and register", page)
        self.assertIn("decisions unchanged since", page)
        self.assertIn("Changes: +", page)
        self.assertEqual(page.count('name="op" value="apply"'), 1)
        t = self.token_on("pujan", run_page)
        fields = {"t": t, "op": "apply", "pkg": "lab_ok", "ts": ts}
        lv0 = self.lvs()
        st, _, body = self.post("pujan", "/apply/run", {**fields, "name": "lab_o"})      # the typed name is checked
        self.assertEqual(st, 409)
        self.assertIn("typed name", body.decode())
        self.assertEqual(self.lvs(), lv0)
        audit = Path(self.w.env.state) / "audit" / "portal.jsonl"
        n_audit = len(jsonl(audit))
        mods = M.registry_modules()
        with mock.patch.object(mods.importers, "import_label_versions", side_effect=RuntimeError("registry busy")):
            st, h, _ = self.post("pujan", "/apply/run", {**fields, "name": "lab_ok"})
        self.assertEqual(st, 303)
        done = T.hget(h, "location")
        lv = re.fullmatch(r"/apply/done/(lv_\d{8}_\d{4})", done).group(1)
        self.assertEqual(sorted(set(self.lvs()) - set(lv0)), [lv])
        page = self.get("pujan", done)[2].decode()
        self.assertIn("created, not yet registered", page)
        self.assertIn("Retry register", page)
        self.assertIn("cryoev register", page)
        st, h, _ = self.post("pujan", "/apply/run", {"t": self.token_on("pujan", done), "op": "retry", "lv": lv})
        self.assertEqual((st, T.hget(h, "location")), (303, done))
        page = self.get("pujan", done)[2].decode()
        self.assertIn("Applied &#10003;", page.replace("✓", "&#10003;"))
        self.assertIn("registered &middot; history updated", page)
        self.assertNotIn("Retry register", page)
        lines = [x for x in jsonl(audit)[n_audit:] if x["action"] in ("apply", "apply retry")]
        self.assertEqual([(x["actor"], x["action"], x["target"]) for x in lines],
                         [("pujan", "apply", "lab_ok"), ("pujan", "apply retry", "lab_ok")])
        self.assertEqual((lines[0]["detail"]["lv"], lines[1]["detail"]["import"]), (lv, "ok"))
        # the preview page now says why it can't be applied again; pressing anyway is refused
        page = self.get("pujan", run_page)[2].decode()
        self.assertIn("Not available: these decisions were already applied as " + lv, page)
        st, _, body = self.post("pujan", "/apply/run", {**fields, "t": self.token_on("pujan", "/apply/"),
                                                         "name": "lab_ok"})
        self.assertEqual(st, 409)
        self.assertIn("already applied", body.decode())


if __name__ == "__main__":
    unittest.main()
