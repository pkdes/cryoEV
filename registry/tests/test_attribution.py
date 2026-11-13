"""Round 9 attribution in the registry: `via` on lines, never carried onto a person's line, never re-appended by the
importer, never promoting a model. Every test sets CRYOEV_VIA / CRYOEV_FOR explicitly (or clears them), so the result
is the same in a Claude Code session and in the user's own shell.

Known-answer controls:
- import twice under CRYOEV_VIA=claude -> the second run appends 0; then with the env unset -> still 0. Control: with
  `via` taken out of importers.IGNORE, the same rerun WOULD re-append (so the check is what saves it);
- an agent's line, then a person's set-status (no env) -> the new line has no `via`;
- production: refused with via, allowed for pujan with no via.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REG = Path(__file__).resolve().parents[1]
REPO = REG.parent
sys.path.insert(0, str(REG))
sys.path.insert(0, str(REPO / "data_utils" / "tests"))
import importers  # noqa: E402
import registry  # noqa: E402
from test_dataset_from_lv import D, make_export, make_pull, write_lv  # noqa: E402

AGENT = {"CRYOEV_VIA": "claude", "CRYOEV_FOR": "pujan"}
DS, PLAIN = "lvds_fixture", "plain_export_ds"


def sample(entity):
    return json.loads((REG / "samples" / f"{entity}.jsonl").read_text(encoding="utf-8").splitlines()[0])


def no_identity_env():
    """os.environ without CRYOEV_VIA / CRYOEV_FOR (whatever the session that runs the tests has)."""
    return mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k not in AGENT}, clear=True)


def lines(root, entity):
    p = Path(root) / f"{entity}.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = registry.main(argv)
    return rc, out.getvalue() + err.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self.env = no_identity_env()
        self.env.start()
        self.addCleanup(self.env.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "registry"


class AppendVia(Base):
    def test_via_set_and_inherited_via_dropped(self):
        run = sample("run")
        line = registry.append(run, root=self.root, via="claude")
        self.assertEqual(line["via"], "claude")
        line = registry.append(dict(line, status="failed"), root=self.root)   # copied from the agent's line
        self.assertNotIn("via", line)
        self.assertNotIn("via", lines(self.root, "run")[-1])
        line = registry.append(dict(run, via="claude"), root=self.root, via="bot2")   # replaced, not merged
        self.assertEqual(line["via"], "bot2")
        with self.assertRaises(registry.RegistryError):
            registry.append(run, root=self.root, via="Claude for pujan")

    def test_agent_line_then_human_set_status_has_no_via(self):
        run = sample("run")
        with mock.patch.dict(os.environ, AGENT):
            rc, out = cli(["--root", str(self.root), "set-status", run["id"], "finished"])
        self.assertEqual(rc, 2, out)                                          # unknown id yet: refused, nothing written
        registry.append(run, root=self.root, via="claude")                    # the agent's line
        with mock.patch.dict(os.environ, AGENT):                               # agent set-status, no --who: CRYOEV_FOR
            rc, out = cli(["--root", str(self.root), "set-status", run["id"], "failed"])
        self.assertEqual(rc, 0, out)
        self.assertIn("by claude for pujan", out)
        last = lines(self.root, "run")[-1]
        self.assertEqual((last["status_by"], last["via"]), ("pujan", "claude"))
        rc, out = cli(["--root", str(self.root), "set-status", run["id"], "finished", "--who", "au1"])   # a person
        self.assertEqual(rc, 0, out)
        last = lines(self.root, "run")[-1]
        self.assertEqual(last["status_by"], "au1")
        self.assertNotIn("via", last)                                         # known answer: not carried over
        self.assertNotIn("via", registry.current(root=self.root)[run["id"]])

    def test_set_status_who_flag_beats_env_and_neither_is_an_error(self):
        run = sample("run")
        registry.append(run, root=self.root)
        with mock.patch.dict(os.environ, AGENT):
            rc, out = cli(["--root", str(self.root), "set-status", run["id"], "failed", "--who", "au1"])
        self.assertEqual(rc, 0, out)
        last = lines(self.root, "run")[-1]
        self.assertEqual((last["status_by"], last["via"]), ("au1", "claude"))  # au1's decision, run by the agent
        n = len(lines(self.root, "run"))
        rc, out = cli(["--root", str(self.root), "set-status", run["id"], "finished"])   # no flag, no env
        self.assertEqual(rc, 2)
        self.assertIn("--who", out)
        with mock.patch.dict(os.environ, {"CRYOEV_FOR": "nobody"}):
            self.assertEqual(cli(["--root", str(self.root), "set-status", run["id"], "finished"])[0], 2)
        with mock.patch.dict(os.environ, {"CRYOEV_VIA": "Not An Id"}):
            self.assertEqual(cli(["--root", str(self.root), "set-status", run["id"], "finished", "--who",
                                  "pujan"])[0], 2)
        self.assertEqual(len(lines(self.root, "run")), n)                     # none of the refusals wrote

    def test_job_new_who_and_via(self):
        lst = Path(self.tmp.name) / "imgs.txt"
        lst.write_text("a_png\nb_png\n", encoding="utf-8")
        base = ["--root", str(self.root), "job", "new", "--type", "annotate", "--where", "app", "--images", str(lst)]
        with mock.patch.dict(os.environ, AGENT):
            self.assertEqual(cli(base + ["--name", "agent"])[0], 0)
            self.assertEqual(cli(base + ["--name", "flag", "--who", "au1"])[0], 0)
        self.assertEqual(cli(base + ["--name", "person"])[0], 0)
        by = {r["name"]: r for r in lines(self.root, "job")}
        self.assertEqual((by["agent"]["who"], by["agent"]["via"]), ("pujan", "claude"))
        self.assertEqual((by["flag"]["who"], by["flag"]["via"]), ("au1", "claude"))
        self.assertEqual(by["person"]["who"], None)                           # the old default: none
        self.assertNotIn("via", by["person"])

    def test_pointer_record_keeps_via_and_created_via_is_separate(self):
        lv = sample("label_version")
        own = Path(self.tmp.name) / "version.json"
        own.write_text(json.dumps(dict({k: v for k, v in lv.items() if k not in ("schema", "logged", "recorded_by")},
                                       created_by="pujan", created_via="claude")), encoding="utf-8")
        line = registry.append({"schema": "label_version/1", "id": lv["id"], "owner": str(own)}, root=self.root,
                               via="claude")
        self.assertEqual(line["via"], "claude")                               # ENVELOPE: kept on a pointer line
        registry.set_status(lv["id"], "superseded", "au1", root=self.root)    # a person, later
        cur = registry.current(root=self.root)[lv["id"]]
        self.assertNotIn("via", cur)
        self.assertEqual((cur["created_by"], cur["created_via"], cur["status_by"]), ("pujan", "claude", "au1"))


class StatusVia(Base):
    """status_via travels with status_by: an agent-set status keeps "claude for pujan" after later import lines (written
    by anyone); a person's new status clears it. The importer is driven with a fixed scan output (one changed field)."""

    def import_run(self, rec, via=None):
        def scan(imp):
            imp.out = [("run", dict(rec))]
            return imp
        t = Path(self.tmp.name)
        with mock.patch.object(importers.Importer, "scan", scan):
            return sum(importers.run(self.root, laptop=t, cache=t, processing=None, via=via)["appended"].values())

    def test_agent_status_survives_imports_and_a_person_clears_it(self):
        import actor as A                                                     # registry/actor.py (REG on sys.path)
        run = sample("run")
        registry.append(run, root=self.root)                                  # a person's line
        with mock.patch.dict(os.environ, AGENT):
            self.assertEqual(cli(["--root", str(self.root), "set-status", run["id"], "failed"])[0], 0)
        last = lines(self.root, "run")[-1]
        self.assertEqual((last["status_by"], last["status_via"], last["via"]), ("pujan", "claude", "claude"))
        # a person's import that changes another field: a new line, no via, the agent-set status kept with its agent
        self.assertEqual(self.import_run(dict(run, machine="laptop")), 1)
        last = lines(self.root, "run")[-1]
        self.assertEqual((last["machine"], last["status"], last["status_by"], last["status_via"]),
                         ("laptop", "failed", "pujan", "claude"))
        self.assertNotIn("via", last)
        cur = registry.current(root=self.root)[run["id"]]
        self.assertEqual(A.label_of(cur, who_key="status_by"), "claude for pujan")
        # no-change control in both env states: 0 appended
        self.assertEqual(self.import_run(dict(run, machine="laptop")), 0)
        self.assertEqual(self.import_run(dict(run, machine="laptop"), via="claude"), 0)
        # a person's new status clears status_via (set-status's own copy carries it otherwise)
        self.assertEqual(cli(["--root", str(self.root), "set-status", run["id"], "finished", "--who", "au1"])[0], 0)
        last = lines(self.root, "run")[-1]
        self.assertEqual(last["status_by"], "au1")
        self.assertNotIn("status_via", last)
        self.assertNotIn("via", last)
        self.assertEqual(A.label_of(registry.current(root=self.root)[run["id"]], who_key="status_by"), "au1")
        # and the importer then carries no status_via either
        self.assertEqual(self.import_run(dict(run, machine="desktop")), 1)
        self.assertNotIn("status_via", lines(self.root, "run")[-1])

    def test_agent_status_kept_through_set_status_copy(self):
        run = sample("run")
        registry.append(run, root=self.root)
        registry.set_status(run["id"], "failed", "pujan", root=self.root, via="claude")
        registry.set_status(run["id"], "finished", "pujan", root=self.root, via="claude")   # agent again: kept
        self.assertEqual(lines(self.root, "run")[-1]["status_via"], "claude")

    def test_append_keeps_status_via(self):
        line = registry.append(dict(sample("run"), status_by="pujan", status_via="claude", via="claude"),
                               root=self.root)
        self.assertEqual((line["status_via"], line.get("via")), ("claude", None))   # only the top-level via is dropped
        with self.assertRaises(registry.RegistryError):
            registry.append(dict(sample("run"), status_by="pujan", status_via="Claude!"), root=self.root)


class Production(Base):
    def setUp(self):
        super().setUp()
        self.m = dict(sample("model"), status="registered")
        registry.append(self.m, root=self.root)

    def test_refused_with_via_allowed_for_pujan_without(self):
        mid = self.m["id"]
        with self.assertRaisesRegex(registry.RegistryError, "agent"):
            registry.set_status(mid, "production", "pujan", root=self.root, via="claude")
        with mock.patch.dict(os.environ, AGENT):                               # the CLI under an agent session
            rc, out = cli(["--root", str(self.root), "set-status", mid, "production", "--who", "pujan"])
        self.assertEqual(rc, 2, out)
        self.assertIn("never sets production", out)
        self.assertEqual(registry.current(root=self.root)[mid]["status"], "registered")
        rc, out = cli(["--root", str(self.root), "set-status", mid, "production", "--who", "pujan"])   # control
        self.assertEqual(rc, 0, out)
        cur = registry.current(root=self.root)[mid]
        self.assertEqual((cur["status"], cur["status_by"]), ("production", "pujan"))
        self.assertNotIn("via", cur)
        # re-appending an already-production model (e.g. an agent's import) is not a promotion: allowed
        line = registry.append(dict(cur, note="alias added"), root=self.root, via="claude")
        self.assertEqual((line["status"], line["via"]), ("production", "claude"))

    def test_agent_append_of_production_refused(self):
        with self.assertRaisesRegex(registry.RegistryError, "agent"):
            registry.append(dict(self.m, status="production", status_by="pujan"), root=self.root,
                            _via_set_status=True, via="claude")


class ImportTwice(Base):
    """The blocker (QC B1): the importer's no-change check must ignore `via`, or every agent-run import re-appends
    every record (append-only, irreversible). Fixture: test_import_lv_dataset's (lv dataset + plain export dataset)."""

    def setUp(self):
        super().setUp()
        t = Path(self.tmp.name)
        self.lap, self.cache = t / "CryoAI-laptop", t / "CryoAI"
        self.lap.mkdir()
        export = make_export(self.cache)
        lv = write_lv(t / "lvs", "lv_20990101_0000", make_pull(), review_scope="reviewed_only")
        outs = self.cache / "training outputs"
        D.build(str(lv), str(export), DS, outs, "ev_nonev", export_only="drop", log=lambda *a: None)
        (outs / PLAIN).mkdir()
        (outs / PLAIN / "dataset.yaml").write_text("nc: 1\n", encoding="utf-8")

    def run_cli(self, *extra):
        """`registry.py import` (the CLI path that reads the env), with the importer pointed at the temp fixture."""
        real = importers.run

        def run(root, dry_run=False, **kw):
            return real(root, dry_run, laptop=self.lap, cache=self.cache, processing=None, **kw)
        with mock.patch.object(importers, "run", run):
            return cli(["--root", str(self.root), "import", *extra])

    def total(self):
        return sum(len(lines(self.root, e)) for e in registry.ENTITIES)

    def appended(self, out):
        return int(out.split("appended")[1].split(":")[1].split()[0])

    def test_agent_import_twice_then_person_appends_nothing(self):
        with mock.patch.dict(os.environ, AGENT):
            rc, out = self.run_cli()
            self.assertEqual(rc, 0, out)
            first = self.total()
            self.assertGreater(first, 0)
            every = [r for e in registry.ENTITIES for r in lines(self.root, e)]
            restated = [r for r in every if r.get("recorded_by") == "registry.py set-status"]
            # the user's production decision (importers.PRODUCTION) is restated without via; every other line is stamped
            self.assertEqual([(r["id"], r["status"], r.get("via")) for r in restated],
                             [(importers.PRODUCTION[0], "production", None)])
            self.assertTrue(all(r.get("via") == "claude" for r in every if r not in restated))
            rc, out = self.run_cli()
            self.assertEqual((rc, self.appended(out)), (0, 0), out)
        self.assertEqual(self.total(), first)
        rc, out = self.run_cli()                                              # env unset: a person's import
        self.assertEqual((rc, self.appended(out)), (0, 0), out)
        self.assertEqual(self.total(), first)
        # control: without `via` in IGNORE the same rerun would re-append every record (dry run: nothing written)
        with mock.patch.object(importers, "IGNORE", {"logged", "recorded_by"}):
            rc, out = self.run_cli("--dry-run")
        self.assertEqual(rc, 0, out)
        self.assertGreater(self.appended(out), 0, "control: the trap is real, so the 0s above mean something")
        self.assertEqual(self.total(), first)

    def test_lv_dataset_carried_forward_drops_the_agents_via(self):
        """importers.lv_dataset copies the current line (`dict(old)`): a person's later change must not carry the
        agent's via."""
        with mock.patch.dict(os.environ, AGENT):
            self.assertEqual(self.run_cli()[0], 0)
        (self.cache / "training outputs" / DS / "runs" / "2099-01-02_120000_y8n_320_2b_1ep_Run_1").mkdir(parents=True)
        rc, out = self.run_cli()                                              # a person's import: frozen -> trained
        self.assertEqual(rc, 0, out)
        mine = [r for r in lines(self.root, "dataset") if r["id"] == DS]
        self.assertEqual([(r["status"], r.get("via")) for r in mine], [("frozen", "claude"), ("trained", None)])

    def test_person_import_then_agent_import_appends_nothing(self):
        rc, out = self.run_cli()
        self.assertEqual(rc, 0, out)
        first = self.total()
        self.assertFalse(any("via" in r for e in registry.ENTITIES for r in lines(self.root, e)))
        with mock.patch.dict(os.environ, AGENT):
            rc, out = self.run_cli()
        self.assertEqual((rc, self.appended(out)), (0, 0), out)
        self.assertEqual(self.total(), first)


if __name__ == "__main__":
    unittest.main()
