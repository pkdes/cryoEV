"""
Upload sessions (portal/upload_sessions.py; round 7 S): a raw upload's conversion -> a registry session/1 record.

  * register_conversion: idempotent; only uploads, only when done, only with CJ_UPLOAD_SESSIONS=on (off: no registry
    byte and no file written); merged into an existing lab session without changing a lab field (known answer, with
    a new session as the control that the merge does fill an empty record); an excluded session is returned and
    written to <conv>/session.json, never raised; acquisition.json read through the schema only (junk keys, paths and
    bad values dropped); registry files only ever appended to.
  * The importer and the portal in both orders (registry/importers.py carry-forward of `conversions` / `details`).
  * save_details: validation, merge (posted rows replace, blank removes, others stay), no-op when unchanged, audit
    line without the typed text, refused off / for a session with no upload.
  * Through the portal (the fake converter of test_predict_convert): the upload form's sample per grid becomes
    input/sample_key.csv (`#,Label`) before conversion and reaches MANIFEST.csv; the table and a .csv are either /
    or; Data -> Sessions shows the "Uploaded" chip, acquisition, samples, notes; conversion / Gallery links and
    thumbnails only for the owner and the admin (an advanced non-owner gets none: control = the admin); the details
    form on the conversion page and on the Data page (uploader / admin only); every user text escaped.

Temp folders only: never the live registry or ~/.cryoev.

  python -m unittest discover -s portal/tests -p test_upload_sessions.py -v
"""
import csv
import html
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cjtest as T  # noqa: E402
import test_predict_convert as TPC  # noqa: E402
from portal import upload_sessions as US  # noqa: E402
from portal.modules import data as D  # noqa: E402

REPO = T.REPO
sys.path.insert(0, str(REPO / "registry"))
import importers  # noqa: E402

XSS = '<img src=x onerror=alert(1)>'
NOTE_XSS = '<script>alert("n")</script> & notes'
LAB_SID, NEW_SID = "20991105", "20991106"
ACQ = {"schema": "acquisition/1",
       "montages": [{"montage": "grid1_1", "mag": 45000, "pixel_A": 1.8}, {"montage": "grid2_1", "mag": 45000}],
       "session": {"instrument": "Krios", "magnification": [45000], "pixel_size_a": [1.8],
                   "defocus_um": {"min": -3.0, "median": -2.0, "max": -1.0}, "n_mdoc": 2, "n_imaging": 2,
                   "magnification_bad": "x", "local_path": "C:\\secret\\raw", "voltage_kv": "two hundred"},
       "scope_model": "Krios"}
LAB = {"schema": "session/1", "id": LAB_SID, "logged": "2099-11-06T10:00:00", "date": "2099-11-05",
       "date_basis": "folder", "status": "processed", "local_folder": f"proc/{LAB_SID}",
       "sample_labels": {"1": "Lab A", "2": "Lab B"}, "session_grids": {"1": "2602"}, "instrument": "Glacios",
       "magnification": [11000, 45000], "n_mdoc": 4, "recorded_by": "backfill:importers abc"}


def make_conv(base: Path, cid: str, owner="adv", kind="upload", state="done", labels=None, acq=None) -> Path:
    """A finished conversion folder as predict.py + jobs/convert_raw.py leave it (tiles + MANIFEST, json files)."""
    d = base / "raw_tiles" / cid
    (d / "tiles").mkdir(parents=True)
    labels = labels or {1: "Up A", 2: XSS}
    rows = []
    for i, g in enumerate((1, 1, 2)):
        name = f"g{g}_x__grid{g}_1_r0c{i}_t00.png"
        T_IMG.save(d / "tiles" / name)
        rows.append({"output_name": name, "grid": g, "sample_label": labels.get(g, f"Grid {g}"), "montage": f"grid{g}_1"})
    with open(d / "tiles" / "MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (d / "convert.json").write_text(json.dumps(
        {"schema": "cj_convert/1", "id": cid, "owner": owner, "session": cid.split("_")[0], "recipe": "default",
         "gpu": False, "source": {"kind": kind, "files": []}, "inputs": [], "sample_key": None,
         "created": "2099-11-06T09:00:00"}), encoding="utf-8")
    (d / "status.json").write_text(json.dumps({"state": state, "n_tiles": 3}), encoding="utf-8")
    if acq is not None:
        (d / "acquisition.json").write_text(json.dumps(acq), encoding="utf-8")
    return d


class _Img:
    def save(self, path):
        from PIL import Image
        Image.new("L", (64, 48), 128).save(path)


T_IMG = _Img()


def snap(reg: Path) -> dict:
    return {p.name: p.read_bytes() for p in reg.glob("*")} if reg.exists() else {}


class Unit(unittest.TestCase):
    def setUp(self):
        self.saved = os.environ.pop(US.FLAG, None)
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_us_"))
        self.base, self.reg, self.state = self.tmp / "raw", self.tmp / "registry", self.tmp / "state"
        self.cfg = SimpleNamespace(registry=self.reg, state_dir=self.state, upload_sessions="on",
                                   raw_root=str(self.base), cj_predict_root=str(self.tmp / "predict"))

    def tearDown(self):
        os.environ.pop(US.FLAG, None)
        if self.saved is not None:
            os.environ[US.FLAG] = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cur(self, sid):
        return US.current_record(self.cfg, sid)

    def lines(self):
        p = self.reg / "session.jsonl"
        return p.read_text(encoding="utf-8").splitlines() if p.exists() else []

    def add_lab(self):
        self.reg.mkdir(parents=True, exist_ok=True)
        (self.reg / "session.jsonl").write_text(json.dumps(LAB) + "\n", encoding="utf-8")

    # ---- register_conversion
    def test_flag_off_writes_nothing(self):
        d = make_conv(self.base, f"{NEW_SID}_aaaaaa", acq=ACQ)
        for off in ("off", "", None):
            self.cfg.upload_sessions = off
            self.assertEqual(US.register_conversion(d, self.cfg)["ok"], False)
        self.assertFalse(self.reg.exists())
        self.assertFalse((d / US.SESSION_JSON).exists())
        self.cfg.upload_sessions = None
        os.environ[US.FLAG] = "on"                                   # the env switch works too
        self.assertTrue(US.register_conversion(d, self.cfg)["ok"])

    def test_new_session_registered_once(self):
        d = make_conv(self.base, f"{NEW_SID}_aaaaaa", acq=ACQ)
        r1 = US.register_conversion(d, self.cfg)
        self.assertEqual((r1["ok"], r1["appended"], r1["session"]), (True, True, NEW_SID))
        before = snap(self.reg)
        r2 = US.register_conversion(d, self.cfg)
        self.assertEqual((r2["ok"], r2["appended"]), (True, False))
        self.assertEqual(snap(self.reg), before)                    # idempotent: not a byte more
        rec = self.cur(NEW_SID)
        self.assertEqual(rec["status"], "processed")
        self.assertEqual(rec["date"], "2099-11-06")
        self.assertEqual(rec["sample_labels"], {"1": "Up A", "2": XSS})
        self.assertEqual((rec["instrument"], rec["magnification"], rec["n_mdoc"]), ("Krios", [45000], 2))
        self.assertNotIn("owner", rec)
        (c,) = rec["conversions"]
        self.assertEqual((c["id"], c["uploaded_by"], c["grids"], c["tiles_per_grid"], c["n_tiles"], c["n_montages"]),
                         (d.name, "adv", [1, 2], {"1": 2, "2": 1}, 3, 2))
        local = json.loads((d / US.SESSION_JSON).read_text(encoding="utf-8"))
        self.assertEqual((local["session"], local["registered"], local["error"]), (NEW_SID, True, None))
        self.assertEqual(US.register_conversion(d, self.cfg)["appended"], False)

    def test_acquisition_through_the_schema_only(self):
        d = make_conv(self.base, f"{NEW_SID}_aaaaaa", acq=ACQ)
        US.register_conversion(d, self.cfg)
        text = (self.reg / "session.jsonl").read_text(encoding="utf-8")
        for bad in ("secret", "magnification_bad", "two hundred", "local_path"):
            self.assertNotIn(bad, text)
        acq = self.cur(NEW_SID)["conversions"][0]["acquisition"]
        self.assertEqual(set(acq), {"instrument", "magnification", "pixel_size_a", "defocus_um", "n_mdoc",
                                    "n_imaging"})
        self.assertNotIn("voltage_kv", acq)                          # a value the schema refuses is dropped
        d2 = make_conv(self.base, f"{NEW_SID}_bbbbbb")                # no acquisition.json: registered without
        self.assertTrue(US.register_conversion(d2, self.cfg)["ok"])
        self.assertNotIn("acquisition", self.cur(NEW_SID)["conversions"][1])

    def test_merge_keeps_every_lab_field(self):
        """Known answer: the lab line's fields come out exactly as they went in; the upload only adds its keys. The
        control (a new session id) shows the same merge does fill an empty record."""
        self.add_lab()
        d = make_conv(self.base, f"{LAB_SID}_cccccc", acq=ACQ)
        self.assertTrue(US.register_conversion(d, self.cfg)["appended"])
        rec = self.cur(LAB_SID)
        for k, v in LAB.items():
            if k not in ("logged", "recorded_by"):
                self.assertEqual(rec[k], v, k)
        self.assertEqual(set(rec) - set(LAB), {"conversions"})
        self.assertEqual(rec["conversions"][0]["acquisition"]["instrument"], "Krios")   # kept with the upload only
        ctl = make_conv(self.base, f"{NEW_SID}_dddddd", acq=ACQ)
        US.register_conversion(ctl, self.cfg)
        self.assertEqual((self.cur(NEW_SID)["instrument"], self.cur(NEW_SID)["sample_labels"]["1"]), ("Krios", "Up A"))
        # a second upload of the same day joins the first, sorted by id; each registers once
        d2 = make_conv(self.base, f"{LAB_SID}_aaaaaa", owner="adv2")
        US.register_conversion(d2, self.cfg)
        self.assertEqual([c["id"] for c in self.cur(LAB_SID)["conversions"]], [d2.name, d.name])
        self.assertEqual(US.register_conversion(d, self.cfg)["appended"], False)

    def test_only_done_uploads(self):
        folder = make_conv(self.base, f"{NEW_SID}_aaaaaa", kind="folder")
        self.assertEqual(US.register_conversion(folder, self.cfg),
                         {"ok": True, "skipped": "not an upload (only uploads become sessions)", "appended": False})
        running = make_conv(self.base, f"{NEW_SID}_bbbbbb", state="running")
        res = US.register_conversion(running, self.cfg)
        self.assertEqual((res["ok"], res["retry"]), (False, True))
        bad = make_conv(self.base, "20991107_eeeeee")
        meta = json.loads((bad / "convert.json").read_text(encoding="utf-8"))
        (bad / "convert.json").write_text(json.dumps(dict(meta, session="20991108")), encoding="utf-8")
        self.assertIn("not valid", US.register_conversion(bad, self.cfg)["error"])
        self.assertFalse(self.reg.exists())

    def test_excluded_session_recorded_not_raised(self):
        d = make_conv(self.base, "20261012_abcdef", acq=ACQ)
        res = US.register_conversion(d, self.cfg)
        self.assertEqual((res["ok"], res["retry"]), (False, False))
        self.assertIn("excluded session", res["error"])
        local = json.loads((d / US.SESSION_JSON).read_text(encoding="utf-8"))
        self.assertEqual(local["registered"], False)
        self.assertIn("excluded session", local["error"])
        self.assertEqual(self.lines(), [])
        self.assertEqual(US.conversion_details_html(self.cfg, {"id": "adv", "role": "advanced"}, d,
                                                    json.loads((d / "convert.json").read_text(encoding="utf-8")),
                                                    {"state": "done"}).count("Not added to Data"), 1)

    def test_registry_only_appended(self):
        self.add_lab()
        files = [make_conv(self.base, f"{LAB_SID}_{h * 6}") for h in "abc"]
        for d in files:
            before = snap(self.reg)
            US.register_conversion(d, self.cfg)
            after = snap(self.reg)
            for name, data in before.items():
                if name.endswith(".jsonl"):
                    self.assertTrue(after[name].startswith(data), name)
        before = snap(self.reg)
        self.assertTrue(US.save_details(LAB_SID, {"prep_notes": "x"}, "adv", self.cfg)["ok"])
        self.assertTrue(snap(self.reg)["session.jsonl"].startswith(before["session.jsonl"]))

    def test_importer_and_portal_both_orders(self):
        """registry/importers.py carries conversions + details forward; the portal never replaces lab fields."""
        dirs = {k: self.tmp / k for k in ("lap", "cai", "proc")}
        for p in dirs.values():
            p.mkdir()

        def importer(rec):
            imp = importers.Importer(self.reg, laptop=dirs["lap"], cache=dirs["cai"], processing=dirs["proc"])
            imp.out = [("session", {k: v for k, v in rec.items() if k not in ("logged", "recorded_by")})]
            return imp.write()["session"]

        # importer first, then the portal, then the importer again (unchanged: 0 lines; changed: keys kept)
        self.assertEqual(importer(LAB), 1)
        d = make_conv(self.base, f"{LAB_SID}_aaaaaa", acq=ACQ)
        self.assertTrue(US.register_conversion(d, self.cfg)["appended"])
        self.assertTrue(US.save_details(LAB_SID, {"prep_notes": "blot 3 s", "grid_id_2": "2603"}, "adv",
                                        self.cfg)["appended"])
        self.assertEqual(importer(LAB), 0)
        self.assertEqual(importer(dict(LAB, n_mdoc=9)), 1)
        rec = self.cur(LAB_SID)
        self.assertEqual((rec["n_mdoc"], rec["conversions"][0]["id"], rec["details"]["prep_notes"],
                          rec["details"]["grid_ids"]), (9, d.name, "blot 3 s", {"2": "2603"}))
        # portal first (a date the lab had not imported), then the importer builds the lab record
        d2 = make_conv(self.base, f"{NEW_SID}_aaaaaa", acq=ACQ)
        US.register_conversion(d2, self.cfg)
        self.assertEqual(self.cur(NEW_SID)["instrument"], "Krios")
        self.assertEqual(importer(dict(LAB, id=NEW_SID, date="2099-11-06")), 1)
        rec = self.cur(NEW_SID)
        self.assertEqual((rec["instrument"], rec["sample_labels"], rec["conversions"][0]["id"]),
                         ("Glacios", {"1": "Lab A", "2": "Lab B"}, d2.name))
        self.assertEqual(US.register_conversion(d2, self.cfg)["appended"], False)     # and the portal agrees

    def test_keys_match_the_importer(self):
        self.assertEqual(US.UPLOAD_KEYS, importers.UPLOAD_KEYS)
        self.assertEqual(US.LOCK, importers.UPLOAD_LOCK)

    # ---- save_details
    def test_details_validated_merged_and_audited(self):
        d = make_conv(self.base, f"{NEW_SID}_aaaaaa")
        self.assertIn("no uploaded conversion", US.save_details(NEW_SID, {"prep_notes": "x"}, "adv", self.cfg)["error"])
        US.register_conversion(d, self.cfg)
        bad = US.save_details(NEW_SID, {"prep_notes": "x" * 2001, "grid_id_7": "1", "display_1": "y" * 81},
                              "adv", self.cfg)
        self.assertFalse(bad["ok"])
        for want in ("Prep notes: at most 2000", "Grid 7 is not a grid of this session", "Grid 1 sample name"):
            self.assertIn(want, bad["error"])
        n = len(self.lines())
        ok = US.save_details(NEW_SID, {"prep_notes": NOTE_XSS + "\r\nline 2\x07", "grid_id_1": " 2602 ",
                                       "display_2": "Shown " + XSS}, "adv", self.cfg)
        self.assertEqual(ok, {"ok": True, "appended": True})
        det = self.cur(NEW_SID)["details"]
        self.assertEqual((det["prep_notes"], det["grid_ids"], det["sample_display"], det["by"]),
                         (NOTE_XSS + "\nline 2", {"1": "2602"}, {"2": "Shown " + XSS}, "adv"))
        self.assertEqual(US.save_details(NEW_SID, {"grid_id_1": "2602"}, "adv", self.cfg)["appended"], False)
        US.save_details(NEW_SID, {"grid_id_1": "", "grid_id_2": "2603"}, "admin1", self.cfg)   # blank removes
        det = self.cur(NEW_SID)["details"]
        self.assertEqual((det["grid_ids"], det["sample_display"], det["prep_notes"], det["by"]),
                         ({"2": "2603"}, {"2": "Shown " + XSS}, NOTE_XSS + "\nline 2", "admin1"))
        self.assertEqual(len(self.lines()), n + 2)
        audit = (self.state / "audit" / "portal.jsonl").read_text(encoding="utf-8")
        self.assertEqual(audit.count('"session_details"'), 2)
        self.assertNotIn("script", audit)                                         # no typed text in the log
        self.assertNotIn("2602", audit)
        self.assertEqual(US.effective_labels(self.cur(NEW_SID)), {"1": "Up A", "2": "Shown " + XSS})
        self.cfg.upload_sessions = "off"
        self.assertEqual(US.save_details(NEW_SID, {"prep_notes": "y"}, "adv", self.cfg)["status"], 403)

    def test_details_html_escaped(self):
        d = make_conv(self.base, f"{NEW_SID}_aaaaaa", labels={1: XSS})
        US.register_conversion(d, self.cfg)
        US.save_details(NEW_SID, {"prep_notes": NOTE_XSS, "display_2": XSS, "grid_id_1": '"><b>x'}, "adv", self.cfg)
        rec = self.cur(NEW_SID)
        out = US.details_form(rec, "/x", error=XSS) + US.details_view(rec)
        for raw in (XSS, "<script>", '"><b>x'):
            self.assertNotIn(raw, out)
        self.assertIn(html.escape(XSS), out)
        self.assertIn(html.escape(NOTE_XSS), out)

    # ---- the upload form's samples
    def test_form_samples_and_key(self):
        samples, bad = US.form_samples({"sample_1": "  EV  prep A ", "sample_2": "", "sample_10": "=cmd()",
                                        "sample_3": "Åbc", "sample_4": "x" * 61, "session": "1", "sample_5": XSS})
        self.assertEqual(samples, {1: "EV prep A", 5: XSS})
        self.assertEqual([b.split(":")[0] for b in bad], ["Sample for grid 3", "Sample for grid 4",
                                                          "Sample for grid 10"])
        text = US.sample_key_text({2: 'B, "two"', 1: "A"})
        self.assertEqual(text, '#,Label\n1,A\n2,"B, ""two"""\n')
        # read back exactly as cryoem_core.metadata.load_sample_key does
        rows = {int(r["#"].strip()): r["Label"].strip() for r in csv.DictReader(text.splitlines())}
        self.assertEqual(rows, {1: "A", 2: 'B, "two"'})


class Portal(TPC.Base):
    """Through the portal: the upload form, the conversion page, Data."""

    @classmethod
    def setUpClass(cls):
        cls.flag = os.environ.get(US.FLAG)
        os.environ[US.FLAG] = "on"
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if cls.flag is None:
            os.environ.pop(US.FLAG, None)
        else:
            os.environ[US.FLAG] = cls.flag

    def fresh(self):
        D._registry_cache.clear()
        D._files_cache.clear()

    def upload_with_samples(self, session, samples, who="advanced"):
        fields = {"session": session, "recipe": "default", **{f"sample_{g}": v for g, v in samples.items()}}
        st, r = self.upload(who, "raw", fields, self.raw_files(key=False))
        self.assertEqual(r.status_code, 200, r.text[:600])
        cid = TPC.CID_RE.match(r.json()["href"]).group(1)
        self.assertEqual(self.wait_conv(cid)["state"], "done", self.status(cid))
        return cid, self.w.rawtiles / cid

    def test_form_table_to_registry_to_data(self):
        page = self.page("/predict/new/raw/", "advanced")
        self.assertIn('name="sample_1"', page)
        self.assertIn('name="sample_12"', page)
        self.assertIn("Data → Sessions", page)
        # either the table or a key file; a label the lab parser can't read is refused before any byte
        both = {"session": "20991102", "recipe": "default", "sample_1": "A"}
        st, _ = self.upload("advanced", "raw", both, self.raw_files(key=True), expect=400)
        self.assertIn("not both", st["error"])
        st, _ = self.upload("advanced", "raw", {**both, "sample_1": "5 µm"}, self.raw_files(key=False), expect=400)
        self.assertIn("Sample for grid 1", st["error"])
        # the table -> input/sample_key.csv (#,Label) -> the converter's --sample-key -> MANIFEST labels
        cid, d = self.upload_with_samples("20991102", {1: "EV prep form", 2: XSS})
        self.assertEqual((d / "input" / "sample_key.csv").read_text(encoding="ascii"),
                         f"#,Label\n1,EV prep form\n2,{XSS}\n")
        self.assertTrue(any("--sample-key" in ln["argv"] and str(d / "input" / "sample_key.csv") in ln["argv"]
                            for ln in self.argv_lines()))
        with open(d / "tiles" / "MANIFEST.csv", newline="", encoding="utf-8") as f:
            self.assertEqual({r["sample_label"] for r in csv.DictReader(f)}, {"EV prep form", XSS})
        # blank table: today's behaviour (no key, "Grid n")
        cid0, d0 = self.upload_with_samples("20991103", {})
        self.assertFalse((d0 / "input" / "sample_key.csv").exists())
        self.assertNotIn("--sample-key", self.argv_lines()[-1]["argv"])

        # the conversion page before registration: no form yet, a plain note
        conv_page = self.page(f"/predict/convert/{cid}/", "advanced")
        self.assertIn("Adding this upload to Data", conv_page)
        self.assertNotIn('id="cj-us-details"', conv_page)
        (d / "acquisition.json").write_text(json.dumps(ACQ), encoding="utf-8")
        self.assertEqual(US.register_conversion(d, self.w.cfg)["appended"], True)
        self.check_conversion_page(cid, d)
        self.check_data_pages(cid, d)

    def check_conversion_page(self, cid, d):
        conv_page = self.page(f"/predict/convert/{cid}/", "advanced")
        self.assertIn('id="cj-us-details"', conv_page)
        self.assertIn(f'action="/predict/convert/{cid}/details"', conv_page)
        self.assertIn(html.escape(XSS), conv_page)
        self.assertNotIn(XSS, conv_page)
        api = self.get(f"/predict/api/convert/{cid}?html=1", "advanced").json()
        self.assertIn("cj-us-details", api["html"]["result"])
        data = {"prep_notes": NOTE_XSS, "display_1": "Shown one", "grid_id_2": "2603"}
        self.assertEqual(self.form(f"/predict/convert/{cid}/details", data, "adv2").status_code, 404)
        self.assertEqual(self.form(f"/predict/convert/{cid}/details", data, "basic").status_code, 404)
        self.assertEqual(self.form(f"/predict/convert/{cid}/details", data, "advanced", origin=False).status_code,
                         403)
        r = self.form(f"/predict/convert/{cid}/details", data, "advanced")
        self.assertEqual(r.status_code, 303, r.text[:400])
        self.assertIn("saved=1", r.headers["location"])
        self.assertIn("Saved.", self.page(r.headers["location"].split("#")[0], "advanced"))
        r = self.form(f"/predict/convert/{cid}/details", {"prep_notes": "x" * 2001}, "advanced")
        self.assertEqual(r.status_code, 400)
        self.assertIn("at most 2000", r.text)
        det = US.current_record(self.w.cfg, "20991102")["details"]
        self.assertEqual((det["prep_notes"], det["sample_display"], det["grid_ids"], det["by"]),
                         (NOTE_XSS, {"1": "Shown one"}, {"2": "2603"}, "adv"))

    def check_data_pages(self, cid, d):
        self.fresh()
        listing = self.page("/data/list/sessions", "adv2")
        row = listing.split('href="/data/session/20991102"', 1)[1].split("</li>", 1)[0]
        self.assertIn(">Uploaded<", row)
        self.assertIn("Krios", row)
        self.assertIn("Shown one", row)                                 # the display name wins in the list
        self.assertNotIn(XSS, listing)
        pages = {who: self.page("/data/session/20991102", who) for who in ("advanced", "adv2", "admin")}
        for who, page in pages.items():
            with self.subTest(who=who):
                self.assertIn("Krios", page)                            # acquisition from the upload
                self.assertIn("the uploaded raw files", page)
                self.assertIn(html.escape(NOTE_XSS), page)              # details: every admin / advanced user
                self.assertIn("2603", page)
                self.assertIn(html.escape(XSS), page)
                for raw in (XSS, "<script>"):
                    self.assertNotIn(raw, page)
                mine = who != "adv2"
                self.assertEqual(f'href="/predict/convert/{cid}/"' in page, mine)
                self.assertEqual(f'href="/gallery/{cid}/"' in page, mine)
                self.assertEqual(cid in page, mine)                      # not even the id for a non-owner
                self.assertEqual('class="cj-data-img"' in page, mine)    # thumbnails from the conversion
                self.assertEqual('id="cj-us-details"' in page, mine)     # the form: uploader / admin
        api = self.get("/data/api/session/20991102", "adv2").json()
        self.assertNotIn(cid, json.dumps(api))
        self.assertEqual(api["upload"]["conversions"], [])
        self.assertEqual(api["upload"]["details"]["prep_notes"], NOTE_XSS)
        self.assertIn(cid, json.dumps(self.get("/data/api/session/20991102", "admin").json()))
        # thumbnails: owner and admin 200 (a JPEG), the advanced non-owner 404
        for who, st in (("advanced", 200), ("admin", 200), ("adv2", 404)):
            r = self.get("/data/thumb/session/20991102/0.jpg", who)
            self.assertEqual(r.status_code, st, who)
            if st == 200:
                self.assertTrue(r.content.startswith(b"\xff\xd8\xff"))
        self.assertEqual(self.get("/data/session/20991102", "basic").status_code, 403)
        # the Data details form: uploader / admin only; basic never reaches Data
        path = "/data/session/20991102/details"
        self.assertEqual(self.form(path, {"prep_notes": "z"}, "adv2").status_code, 404)
        self.assertEqual(self.form(path, {"prep_notes": "z"}, "basic").status_code, 403)
        r = self.form(path, {"prep_notes": "from Data", "display_2": ""}, "advanced")
        self.assertEqual(r.status_code, 303, r.text[:400])
        self.assertIn("Saved.", self.page(r.headers["location"].split("#")[0], "advanced"))
        r = self.form(path, {"grid_id_9": "1"}, "admin")
        self.assertEqual(r.status_code, 400)
        self.assertIn("Grid 9 is not a grid of this session", r.text)
        det = US.current_record(self.w.cfg, "20991102")["details"]
        self.assertEqual((det["prep_notes"], det["sample_display"]), ("from Data", {"1": "Shown one"}))
        # switched off: no form, and a post is refused (the registry is not written)
        before = (self.w.env.registry / "session.jsonl").read_bytes()
        with mock.patch.dict(os.environ, {US.FLAG: "off"}):
            self.fresh()
            self.assertNotIn('id="cj-us-details"', self.page("/data/session/20991102", "admin"))
            self.assertNotIn('id="cj-us-details"', self.page(f"/predict/convert/{cid}/", "admin"))
            self.assertEqual(self.form(path, {"prep_notes": "off"}, "admin").status_code, 403)
            self.assertEqual(US.register_conversion(d, self.w.cfg)["ok"], False)
        self.assertEqual((self.w.env.registry / "session.jsonl").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
