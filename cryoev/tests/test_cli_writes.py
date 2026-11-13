"""cryoev writes: dry runs write nothing; --do writes the record and the two action-log lines; refusals.

Controls: the dry runs are checked by hashing every file of the temp platform before and after (the same test then
shows --do does change it, so the hash check can see a write); the never-automated refusals sit next to a look-alike
read that is allowed (`find push` is a search, not a push)."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cli_fixture as F  # noqa: E402

from data_utils import path_guard  # noqa: E402


class DryRun(F.Platform):
    def test_dry_runs_write_nothing(self):
        before = self.snapshot()
        with self.as_agent():
            runs = [("feedback", "status", 1, "Done"), ("feedback", "reply", 1, "fixed in round 9"),
                    ("pull", "predictions", F.PS_ID, "--out", self.out, "--with-images"),
                    ("pull", "results", F.PS_ID, "--out", self.out),
                    ("register",),
                    ("predict", "new", *self.predict_args)]
            results = {r[:2]: self.cli(*r) for r in runs}
        self.assertEqual(self.snapshot(), before, "a dry run wrote something")
        self.assertFalse(self.out.exists())
        self.assertFalse((self.home / ".cryoev").exists())
        for key, (rc, out, err) in results.items():
            if key == ("pull", "results"):           # the golden set has no results/: refused, still nothing written
                self.assertEqual(rc, 2, err)
                self.assertIn("no results yet", err)
                continue
            self.assertEqual(rc, 0, f"{key}: {out}\n{err}")
            self.assertIn("DRY RUN: nothing was written", out, key)
            self.assertIn("recorded as: claude for pujan", out, key)
        self.assertIn("[predict_set] DRY RUN, nothing written", results[("predict", "new")][1])
        self.assertIn("appended (dry run)", results[("register",)][1])
        self.assertIn("regenerates", results[("register",)][1])
        # the samples' production model reads "candidate": the real import would restate the user's decision
        self.assertIn(f"would restate production for {F.MODEL} as recorded by the user", results[("register",)][1])
        self.assertIn("model (set-status production)=1", results[("register",)][1])
        # control: the same pull with --do changes the snapshot (the hash check sees writes)
        with self.as_agent():
            rc, out, err = self.cli("pull", "predictions", F.PS_ID, "--out", self.out, "--do")
        self.assertEqual(rc, 0, err)
        self.assertNotEqual(self.snapshot(), before)

    def test_predict_set_dry_run_refusal_writes_nothing(self):
        before = self.snapshot()
        rc, out, err = self.cli("predict", "new", *self.predict_args[:-2])      # no --nm-confirmed-by
        self.assertEqual(rc, 2)
        self.assertIn("nm-confirmed-by", err)
        self.assertEqual(self.snapshot(), before)


class Do(F.Platform):
    def test_feedback_status_do(self):
        old = self.feedback.read_bytes()
        with self.as_agent():
            rc, out, err = self.cli("feedback", "status", 1, "done", "--do")
        self.assertEqual(rc, 0, err)
        ev = json.loads(self.feedback.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual({k: ev[k] for k in ("type", "id", "who", "via", "status")},
                         {"type": "status", "id": 1, "who": "pujan", "via": "claude", "status": "Done"})
        backups = list((self.home / ".cryoev" / "backups" / "feedback").glob("*.jsonl"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), old)           # the file as it was before the write
        log = self.action_log()
        self.assertEqual([x["detail"]["phase"] for x in log], ["requested", "result"])
        for x in log:
            self.assertEqual((x["actor"], x["action"], x["target"], x["detail"]["via"]),
                             ("pujan", "feedback status", "feedback #1", "claude"))
            self.assertEqual(x["detail"]["args"], {"id": 1, "status": "Done"})
            self.assertEqual(set(x), {"t", "actor", "action", "target", "detail"})
        self.assertEqual(log[1]["detail"]["result"]["rc"], 0)
        from portal import feedback_store as FS
        self.assertEqual(FS.read_all(self.feedback)[1]["status"], "Done")
        # the Admin page reads this file as the unverified CLI door and shows "claude for pujan" + AI
        from portal import audit as PA
        self.assertEqual(PA.door_label(self.lap / "logs" / "agent_actions.jsonl"), PA.CLI_DOOR)
        from portal.modules import admin
        rows, _ = PA.read_file(self.lap / "logs" / "agent_actions.jsonl")
        r = admin.audit_row(dict(rows[0], door=PA.CLI_DOOR))
        self.assertEqual((r["name"], r["ai"], r["unverified"], r["kind"]), ("claude for pujan", True, True, "feedback"))

    def test_feedback_reply_do_without_agent(self):
        rc, out, err = self.cli("feedback", "reply", 1, "thanks, fixed", "--do", "--who", "pujan")
        self.assertEqual(rc, 0, err)
        ev = json.loads(self.feedback.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual((ev["type"], ev["reply"], ev["who"]), ("reply", "thanks, fixed", "pujan"))
        self.assertNotIn("via", ev)                           # a person acting directly: no agent stamped
        log = self.action_log()
        self.assertEqual(len(log), 2)
        self.assertIsNone(log[0]["detail"]["via"])
        self.assertEqual(log[0]["detail"]["args"], {"id": 1, "chars": 13})     # the text itself is never logged
        self.assertNotIn("thanks", json.dumps(log))

    def test_do_needs_a_person(self):
        rc, out, err = self.cli("feedback", "status", 1, "Done", "--do")
        self.assertEqual(rc, 2)
        self.assertIn("CRYOEV_FOR", err)
        self.assertEqual(self.action_log(), [])

    def test_bad_via_refused(self):
        with mock.patch.dict(os.environ, {"CRYOEV_VIA": "Claude Code", "CRYOEV_FOR": "pujan"}):
            rc, out, err = self.cli("feedback", "status", 1, "Done", "--do")
        self.assertEqual(rc, 2)
        self.assertIn("CRYOEV_VIA", err)
        self.assertEqual(self.action_log(), [])

    def test_predict_new_do(self):
        with self.as_agent():
            rc, out, err = self.cli("predict", "new", *self.predict_args, "--do")
        self.assertEqual(rc, 0, err)
        import re
        ps = next(ln.strip() for ln in out.splitlines() if re.fullmatch(r"ps_\d{8}_[0-9a-f]{6}", ln.strip()))
        self.assertTrue((self.lap / "predict" / ps / "request.json").is_file())
        lines = [json.loads(ln) for ln in (self.reg / "prediction_set.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual((lines[-1]["id"], lines[-1]["status"], lines[-1]["owner"]), (ps, "queued", "pujan"))
        log = self.action_log()
        self.assertEqual([(x["action"], x["detail"]["phase"], x["detail"]["via"]) for x in log],
                         [("prediction set create", "requested", "claude"), ("prediction set create", "result", "claude")])
        self.assertEqual(log[1]["detail"]["result"], {"rc": 0, "ps": ps})

    def test_pull_predictions_do(self):
        with self.as_agent():
            rc, out, err = self.cli("pull", "predictions", F.PS_ID, "--out", self.out, "--do")
        self.assertEqual(rc, 0, err)
        dest = self.out / F.PS_ID
        man = [ln.split("  ", 1) for ln in (dest / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()]
        self.assertIn("p/v5_evnonev_y8m_20261029/summary.csv", [m[1] for m in man])
        self.assertNotIn("images/0001.png", [m[1] for m in man])             # tiles only with --with-images
        for h, rel in man:                                                    # every copy = its source
            import hashlib
            self.assertEqual(hashlib.sha256((F.GOLDEN / rel).read_bytes()).hexdigest(), h)
            self.assertEqual(hashlib.sha256((dest / rel).read_bytes()).hexdigest(), h)
        info = json.loads((dest / "pull.json").read_text(encoding="utf-8"))
        self.assertEqual((info["who"], info["via"], info["n_files"]), ("pujan", "claude", len(man)))
        rc, out, err = self.cli("pull", "predictions", F.PS_ID, "--out", self.out, "--do", "--who", "pujan")
        self.assertEqual(rc, 2)                                               # never over an earlier pull
        self.assertIn("already exists", err)


class Refusals(F.Platform):
    def test_never_automated(self):
        before = self.snapshot()
        for argv in (["roboflow", "upload"], ["upload", "x"], ["relabel"], ["promote", F.MODEL],
                     ["set-production", F.MODEL], ["push"], ["git", "push"], ["predict", "push", F.PS_ID],
                     ["sync", "push"], ["train", "send"], ["box", "pull"], ["model", "production", F.MODEL]):
            with self.subTest(argv=argv), self.as_agent():
                rc, out, err = self.cli(*argv, "--do")
                self.assertEqual(rc, 3, argv)
                self.assertIn("never automated", err)
                self.assertIn("docs/AGENTS.md", err)
        self.assertEqual(self.snapshot(), before)
        rc, out, err = self.cli("find", "push")          # control: a search for the word is a read, not a push
        self.assertEqual(rc, 1, err)                     # (no id contains it)
        self.assertIn("0 match", out)

    def test_out_under_protected_root(self):
        box = path_guard.PROTECTED_ROOTS[-1] / "cryoev_cli_test_out"     # ~/Box, as the guard knows it
        for out in (box, self.lap / "pulled", self.reg / "x", self.cai / "x", F.REPO / "pulled_here"):
            with self.subTest(out=out), self.as_agent():
                rc, o, err = self.cli("pull", "predictions", F.PS_ID, "--out", out, "--do")
                self.assertEqual(rc, 2, out)
                self.assertIn("refused: --out", err)
                self.assertFalse(Path(out).exists())
        self.assertEqual(self.action_log(), [])          # refused before the requested line

    def test_box_paths_refused_for_predict(self):
        box = path_guard.PROTECTED_ROOTS[-1] / "tiles"
        args = list(self.predict_args)
        args[1] = str(box)
        rc, out, err = self.cli("predict", "new", *args)
        self.assertEqual(rc, 3)
        self.assertIn("Box", err)


class FeedbackStoreVia(F.Platform):
    def test_via_optional_and_checked(self):
        from portal import feedback_store as FS
        self.assertNotIn("via", FS.set_status(self.feedback, 1, "pujan", "Planned"))     # the portal: unchanged
        self.assertEqual(FS.set_reply(self.feedback, 1, "pujan", "ok", via="claude")["via"], "claude")
        self.assertEqual(FS.create(self.feedback, "pujan", "feature", "x", via="claude")["via"],
                         "claude")
        with self.assertRaises(FS.Invalid):
            FS.set_status(self.feedback, 1, "pujan", "Done", via="Claude!")
        self.assertEqual(FS.read_all(self.feedback)[1]["status"], "Planned")    # fold ignores via


class Round9QC(F.Platform):
    """QC rulings on r9-cli: exact predict_set options only, Box on every value, pull sources guarded, realpath
    containment, register --do."""

    def box(self):
        return path_guard.PROTECTED_ROOTS[-1] / "tiles"          # ~/Box as the guard knows it (never created)

    def test_predict_abbreviations_and_box_values(self):
        before = self.snapshot()
        base = self.predict_args[2:]                               # everything but --from-dir <src>
        cases = [(["--from", str(self.box())] + base, 2), (["--from-d=" + str(self.box())] + base, 2),
                 (self.predict_args[:2] + ["--man", str(self.box())] + base[2:], 2),
                 (["--from-dir=" + str(self.box())] + base, 3), (["--from-dir", str(self.box())] + base, 3),
                 (self.predict_args + ["--root", str(self.box())], 3),
                 (self.predict_args + ["--dry"], 2), (self.predict_args + ["--dry-run=1"], 2),
                 (self.predict_args + ["--help"], 2), (self.predict_args + [str(self.box())], 3)]
        for args, code in cases:
            with self.subTest(args=args[:3]), self.as_agent():
                rc, out, err = self.cli("predict", "new", *args, "--do")
                self.assertEqual(rc, code, err)
                self.assertIn("refused", err)
        self.assertEqual(self.snapshot(), before)                  # nothing written, nothing logged
        # control: the exact options pass, and a user's own --dry-run is dropped (the CLI decides): still a dry run
        rc, out, err = self.cli("predict", "new", *self.predict_args, "--dry-run")
        self.assertEqual(rc, 0, err)
        self.assertEqual(out.count("--dry-run"), 1)
        self.assertEqual(self.snapshot(), before)

    def test_predict_note_and_provenance(self):
        """Round 11b: --note and --provenance are exact create options; --provenance and --manifest are read sources
        (Box refused, a protected processing folder allowed for reading); --note is text (Box refused, no write-path
        check); abbreviations refused. The dry run writes nothing; --do stores both."""
        from cryoev import common as CM
        from cryoev.commands import predict as P
        prov = self.tmp / "provenance.json"
        prov.write_text(json.dumps({"steps": ["cut 1440 px tiles at native resolution"],
                                    "tiling": {"tile_px": 1440, "overlap_frac": 0.2, "nm_per_px": 0.368,
                                               "downsample": 1}}), encoding="utf-8")
        extra = ["--note", "native tiles, no downsampling", "--provenance", str(prov)]
        self.assertEqual(P.check_args(self.predict_args + extra), self.predict_args + extra)
        processing = path_guard.PROTECTED_ROOTS[0] / "x" / "provenance.json"      # a read source: allowed
        self.assertEqual(P.check_args(["--provenance", str(processing)]), ["--provenance", str(processing)])
        note = f"tiles from {path_guard.PROTECTED_ROOTS[0] / 'x'}"                   # text: no write-path check
        self.assertEqual(P.check_args(["--note", note]), ["--note", note])
        with self.assertRaises(CM.CliError):                                          # control: a write flag is
            P.check_args(["--root", str(processing)])                                 # still checked
        for key in ("--provenance", "--manifest", "--note"):
            with self.subTest(key=key), self.assertRaises(CM.Refused):
                P.check_args([key, str(self.box() / "provenance.json")])
        for abbrev in ("--prov", "--not"):
            with self.subTest(abbrev=abbrev), self.assertRaises(CM.CliError):
                P.check_args([abbrev, "x"])
        before = self.snapshot()
        rc, out, err = self.cli("predict", "new", *self.predict_args, *extra)
        self.assertEqual(rc, 0, err)
        self.assertIn('"note": "native tiles, no downsampling"', out)
        self.assertEqual(self.snapshot(), before)
        bad = self.tmp / "bad.json"
        bad.write_text('{"colour": "blue"}', encoding="utf-8")
        rc, out, err = self.cli("predict", "new", *self.predict_args, "--provenance", str(bad))
        self.assertEqual(rc, 2, err)
        self.assertIn("unknown key 'colour'", err)
        with self.as_agent():
            rc, out, err = self.cli("predict", "new", *self.predict_args, *extra, "--do")
        self.assertEqual(rc, 0, err)
        rec = [json.loads(ln) for ln in (self.reg / "prediction_set.jsonl").read_text(encoding="utf-8").splitlines()][-1]
        self.assertEqual(rec["note"], "native tiles, no downsampling")
        self.assertEqual(rec["provenance"]["tiling"]["tile_px"], 1440)
        ps_dir = self.lap / "predict" / rec["id"]
        self.assertEqual((ps_dir / "source_manifest.csv").read_bytes(), (self.src / "MANIFEST.csv").read_bytes())

    def test_pull_source_under_box_refused(self):
        lv_dir = self.home / "Box" / "labels" / "lv_20261112_0000"         # this test's home: Path.home()/Box
        lv_dir.mkdir(parents=True)
        (lv_dir / "version.json").write_text("{}", encoding="utf-8")
        with open(self.reg / "label_version.jsonl", "a", encoding="utf-8") as f:   # fixture setup, not the CLI
            f.write(json.dumps({"schema": "label_version/1", "id": "lv_20261112_0000",
                                "owner": str(lv_dir / "version.json"), "logged": "2026-11-12T00:00:00"}) + "\n")
        with self.as_agent():
            rc, out, err = self.cli("pull", "labels", "lv_20261112_0000", "--out", self.out)
        self.assertEqual(rc, 3, err)
        self.assertIn("Box", err)
        self.assertFalse(self.out.exists())

    @unittest.skipUnless(os.name == "nt", "junctions are Windows-only")
    def test_out_through_a_junction_into_the_data_refused(self):
        import _winapi
        link = self.tmp / "looks_outside"
        _winapi.CreateJunction(str(self.lap), str(link))
        try:
            with self.as_agent():
                rc, out, err = self.cli("pull", "predictions", F.PS_ID, "--out", link / "pulls", "--do")
            self.assertEqual(rc, 2, err)
            self.assertIn("CryoAI-laptop data folder", err)
            self.assertFalse((self.lap / "pulls").exists())
        finally:
            os.rmdir(link)

    def test_register_do(self):
        page = self.reg / "history.html"
        self.assertFalse(page.exists())
        with self.as_agent():
            rc, out, err = self.cli("register", "--do")
        self.assertEqual(rc, 0, err)
        self.assertIn("history page regenerated", out)
        self.assertTrue(page.is_file() and page.stat().st_size > 1000)
        log = self.action_log()
        self.assertEqual([(x["action"], x["detail"]["phase"], x["actor"], x["detail"]["via"]) for x in log],
                         [("registry import", "requested", "pujan", "claude"), ("registry import", "result", "pujan",
                                                                                "claude")])
        self.assertEqual(log[1]["detail"]["result"]["history_page"], "regenerated")
        self.assertEqual(log[1]["detail"]["result"]["rc"], 0)
        from cryoev import common as CM
        model = CM.registry().current("model", self.reg)[F.MODEL]
        self.assertEqual((model["status"], model["status_by"]), ("production", "pujan"))   # restated, user's
        self.assertNotIn("status_via", model)


if __name__ == "__main__":
    unittest.main()
