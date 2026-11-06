"""
Predict, round 6 piece P1 (BUILD_CONTRACTS.md section 12): the 1000-image cap, set creation in the background (a
progress page, no duplicate on a retry, a failed create keeps the upload), hard-linked uploads, the 20 GB / free-disk
limits, the time estimate, analysis names (clean_name, display_name) and rename. Temp roots only: same fixtures as
test_predict_module.py (its Base class), the fake predictor, never the real GPU or data.
"""
import hashlib
import html
import json
import os
import re
import shutil
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
import test_predict_module as TM  # noqa: E402
from portal.modules import names as NM  # noqa: E402
from portal.modules import predict as P  # noqa: E402

GOLDEN, MODEL, XSS = TM.GOLDEN, TM.MODEL, TM.XSS


class Pure(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(P.PS.MAX_IMAGES, 1000)
        self.assertEqual(P.MAX_FILES, 1000)
        self.assertEqual(P.TILES_MAX_TOTAL_BYTES, 20 * 1024 ** 3)

    def test_display_name_never_holds_the_source_dir(self):
        req = {"source": {"kind": "session", "session": "20261028", "dir": r"C:\Users\someone\secret\tiles"},
               "created": "2026-11-06T14:03:00"}
        got = NM.display_name(req)
        self.assertEqual(got, "Session 20261028 · 6 Nov 2026, 14:03")
        for bad in ("secret", "Users", "someone", "tiles"):
            self.assertNotIn(bad, got)
        self.assertEqual(NM.display_name({**req, "name": "My run"}), "My run")
        self.assertEqual(NM.display_name({**req, "name": "  "}), got)           # empty = the default
        self.assertEqual(NM.display_name({**req, "name": 7}), got)
        self.assertEqual(NM.display_name({"source": {"kind": "upload"}, "created": "2026-11-06T14:03:00"}),
                         "Uploaded images · 6 Nov 2026, 14:03")
        self.assertEqual(NM.display_name(req, "EV prep"), "Session 20261028 (EV prep) · 6 Nov 2026, 14:03")

    def test_set_title_label_comes_from_the_configured_label_not_the_path(self):
        d = Path(os.getcwd())
        req = {"source": {"kind": "session", "session": "1", "dir": str(d)}, "created": "2026-11-06T14:03:00"}
        got = P.set_title(req, [{"label": "Prep A", "dir": d, "problem": None}])
        self.assertEqual(got, "Session 1 (Prep A) · 6 Nov 2026, 14:03")
        self.assertNotIn(str(d), got)


class Estimate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__import__("tempfile").mkdtemp(prefix="cj_est_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.cfg = object()
        p = mock.patch.object(P, "predict_root", lambda cfg: self.tmp)
        p.start()
        self.addCleanup(p.stop)
        P._EST_CACHE.clear()
        self.addCleanup(P._EST_CACHE.clear)

    def finished(self, i, n, seconds, where="laptop", state="done"):
        d = self.tmp / f"ps_20261106_{i:06x}"
        d.mkdir()
        t0 = 1_800_000_000
        iso = lambda t: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))  # noqa: E731
        (d / "status.json").write_text(json.dumps(
            {"state": state, "n_done": n, "n_total": n, "where": where, "updated": iso(t0 + seconds)}), "utf-8")
        (d / "predict.log").write_text(f"old noise\n\n=== {iso(t0)} predict_local ps on cpu: argv\nrest\n"
                                       f"=== {iso(t0 + seconds)} done: x\n", "utf-8")
        return d

    def test_fit_known_answer(self):
        for i, n in enumerate((10, 50, 120), 1):
            self.finished(i, n, 10 + 0.2 * n)
        a, b, fitted = P.estimate_line(self.cfg)
        self.assertTrue(fitted)
        self.assertAlmostEqual(a, 10, delta=0.01)
        self.assertAlmostEqual(b, 0.2, delta=0.0001)
        self.assertAlmostEqual(a + b * 1000, 210, delta=0.2)
        self.assertEqual(P.estimate_text(a, b, 1000), "about 4 min once it starts")
        self.assertEqual(P.estimate_text(a, b, 1000, 2), "about 4 min once it starts · 2 analyses ahead of you")
        self.assertEqual(P.estimate_text(a, b, 3, 1), "about 1 min once it starts · 1 analysis ahead of you")
        self.assertEqual(P.estimate_text(a, b, 0), "")

    def test_fallback_with_fewer_than_three_distinct(self):
        self.finished(1, 10, 20)
        self.finished(2, 50, 40)
        self.assertEqual(P.estimate_line(self.cfg), (*P.EST_FALLBACK, False))
        P._EST_CACHE.clear()
        self.finished(3, 50, 41)                             # a third run, but only two distinct counts
        self.assertEqual(P.estimate_line(self.cfg), (*P.EST_FALLBACK, False))
        a, b, _ = P.estimate_line(self.cfg)
        self.assertEqual(P.estimate_text(a, b, 1000), "about 3 min once it starts")      # 12 + 170 s

    def test_only_finished_laptop_runs_count_and_cache_for_ten_minutes(self):
        for i, n in enumerate((10, 50, 120), 1):
            self.finished(i, n, 10 + 0.2 * n)
        self.finished(9, 200, 5, where="desktop")
        self.finished(10, 300, 5, state="failed")
        t = time.time()
        first = P.estimate_line(self.cfg, now=t)
        self.assertTrue(first[2])
        shutil.rmtree(next(self.tmp.iterdir()))              # gone from disk, still cached
        self.assertEqual(P.estimate_line(self.cfg, now=t + 599), first)
        self.assertNotEqual(P.estimate_line(self.cfg, now=t + 601), first)

    def test_queue_ahead(self):
        for i, s in enumerate(("queued", "running", "done", "failed"), 1):
            d = self.tmp / f"ps_20261106_{i:06x}"
            d.mkdir()
            (d / "status.json").write_text(json.dumps({"state": s}), "utf-8")
        self.assertEqual(P.queue_ahead(self.cfg), 2)


class P1(TM.Base):
    def setUp(self):
        self.addCleanup(P._CREATES.clear)

    # ---- helpers
    def jpost(self, path, doc, who="basic"):
        return self.post(path, json.dumps(doc).encode(), who, ctype="application/json")

    def jget(self, path, who="basic"):
        st, _, body = self.get(path, who)
        return st, json.loads(body)

    def raw_submit(self, who="basic", **kw):
        st, h, body = self.post(f"/predict/new/session/{self.key_a}/", self.form_a(who, **kw), who)
        return st, T.hget(h, "location"), body.decode("utf-8")

    def wait_task(self, tid, who="basic", states=("done", "failed"), timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            st, j = self.jget(f"/predict/api/preparing/{tid}", who)
            if j.get("state") in states:
                return j
            time.sleep(0.05)
        self.fail(f"{tid} still running")

    def stage_upload(self, n=2, who="basic", **fields):
        """The page's script by hand: start, send every file. -> (upload id, [(name, bytes)])."""
        files = [(f"my tile {i}.png", (GOLDEN / "images" / f"000{i}.png").read_bytes()) for i in range(1, n + 1)]
        f = {"sample_label": "Up S", "nm_per_px": "1.4724", "confirm": "yes", "model": MODEL}
        f.update(fields)
        st, _, body = self.jpost("/predict/api/upload", {"kind": "tiles", "fields": f,
                                                         "files": [{"name": a, "size": len(b)} for a, b in files]}, who)
        self.assertEqual(st, 200, body[:400])
        uid = json.loads(body)["upload"]
        for name, data in files:
            st, _, body = self.post(f"/predict/api/upload/{uid}/file?name={name.replace(' ', '%20')}&offset=0", data,
                                    who, ctype="application/octet-stream")
            self.assertEqual(st, 200, body[:400])
        return uid, files

    def staging(self, uid):
        return self.root / P.PORTAL_TMP / uid

    # ---- the cap, both layers
    def test_cap_n_allowed_n_plus_one_refused_both_layers(self):
        def decl(k):
            return [{"name": f"t{i}.png", "size": 10} for i in range(k)]
        with mock.patch.object(P, "MAX_FILES", 3):
            self.assertEqual(len(P.declared_files("tiles", decl(3))), 3)
            with self.assertRaises(P.Refusal) as cm:
                P.declared_files("tiles", decl(4))
            self.assertEqual(cm.exception.status, 413)
            self.assertIn("More than 3 files", cm.exception.msg)
        # the portal layer open (1000), the jobs layer at 2: the pick of 3 tiles is refused when the set is made
        before = self.sets()
        with mock.patch.object(P.PS, "MAX_IMAGES", 2):
            st, loc, _ = self.raw_submit()
            self.assertEqual(st, 303)
            j = self.wait_task(loc.rstrip("/").split("/")[-1])
        self.assertEqual(j["state"], "failed")
        self.assertEqual(self.sets(), before)
        st, page = self.get(loc, "basic")[0], self.get(loc, "basic")[2].decode("utf-8")
        self.assertEqual(st, 200)
        self.assertIn("at most 2 images", page)
        self.assertIn("Back to the form", page)

    # ---- the background create
    def test_background_create_progress_dedupe_reload(self):
        real, gate, seen = P.PS.create, threading.Event(), []

        def slow(*a, **k):
            k["progress"](2, 5, "placing")
            seen.append(k.get("link"))
            gate.wait(30)
            return real(*a, **k)
        before = set(self.sets())
        with mock.patch.object(P.PS, "create", slow):
            t0 = time.time()
            st, loc, _ = self.raw_submit()
            self.assertLess(time.time() - t0, 5)
            self.assertEqual(st, 303)
            self.assertRegex(loc, r"^/predict/preparing/sc_[0-9a-f]{12}/$")
            tid = loc.rstrip("/").split("/")[-1]
            for _ in range(100):
                _, j = self.jget(f"/predict/api/preparing/{tid}?html=1")
                if j.get("total"):
                    break
                time.sleep(0.05)
            self.assertEqual((j["state"], j["done"], j["total"]), ("running", 2, 5))
            self.assertEqual(j["text"], "Preparing your images: 2 of 5")
            self.assertIn("Preparing your images: 2 of 5", j["html"]["status"])
            # a reload of the progress page works, twice
            for _ in range(2):
                st, _, body = self.get(loc, "basic")
                self.assertEqual(st, 200)
                self.assertIn("Preparing your images: 2 of 5", body.decode("utf-8"))
            # the same submit again while it runs: the same task, no second create
            st2, loc2, _ = self.raw_submit()
            self.assertEqual((st2, loc2), (303, loc))
            self.assertEqual(self.get(loc, "other")[0], 404)           # someone else's task is not found
            self.assertEqual(self.get(f"/predict/api/preparing/{tid}", "other")[0], 404)
            gate.set()
            j = self.wait_task(tid)
        self.assertEqual(j["state"], "done")
        new = set(self.sets()) - before
        self.assertEqual(len(new), 1, "a duplicate set was made")
        self.assertEqual(j["href"], f"/predict/{new.pop()}/")
        self.assertEqual(seen, [False])                                # session tiles are copied, not linked
        self.assertEqual(len(seen), 1)
        self.assertEqual(self.get(loc, "basic")[0], 303)               # a finished task: straight to the analysis
        self.wait_state(j["href"].strip("/").split("/")[-1])

    def test_unknown_task_is_404(self):
        for tid in ("sc_000000000000", "us_000000000000", "../x", "nope"):
            self.assertEqual(self.get(f"/predict/preparing/{tid}/", "basic")[0], 404, tid)

    # ---- uploads: hard links, a failed create keeps the upload
    def test_upload_is_hard_linked_and_staging_removed(self):
        uid, files = self.stage_upload(2)
        real, probe = P.PS.create, {}

        def watch(*a, **k):
            out = real(*a, **k)
            for i, (name, _) in enumerate(files, 1):
                a_, b_ = self.staging(uid) / "files" / name, out / "images" / f"{i:04d}.png"
                probe[name] = (os.stat(a_).st_ino == os.stat(b_).st_ino, os.stat(a_).st_nlink)
            probe["link"] = k.get("link")
            return out
        before = set(self.sets())
        with mock.patch.object(P.PS, "create", watch):
            st, _, body = self.jpost(f"/predict/api/upload/{uid}/create", {})
            self.assertEqual(st, 200, body)
            href = json.loads(body)["href"]
            self.assertEqual(href, f"/predict/preparing/{uid}/")
            j = self.wait_task(uid)
        self.assertEqual(j["state"], "done")
        self.assertTrue(probe["link"])
        self.assertTrue(all(v == (True, 2) for k, v in probe.items() if k != "link"), probe)
        ps = (set(self.sets()) - before).pop()
        self.assertFalse(self.staging(uid).exists())                    # staging names gone ...
        for i, (_, data) in enumerate(files, 1):                        # ... the set's files remain, bytes identical
            got = (self.root / ps / "images" / f"{i:04d}.png").read_bytes()
            self.assertEqual(hashlib.md5(got).hexdigest(), hashlib.md5(data).hexdigest())
        self.wait_state(ps)

    def test_failed_create_keeps_the_upload_and_retry_works(self):
        uid, files = self.stage_upload(2)
        before = self.sets()
        with mock.patch.object(P.PS, "create", side_effect=P.PS.PredictSetError("refused: boom")):
            st, _, body = self.jpost(f"/predict/api/upload/{uid}/create", {})
            self.assertEqual(st, 200, body)
            j = self.wait_task(uid)
        self.assertEqual(j["state"], "failed")
        self.assertEqual(self.sets(), before)
        self.assertTrue((self.staging(uid) / "files" / files[0][0]).is_file())          # the upload is kept
        self.assertEqual(len(list((self.staging(uid) / "files").iterdir())), 2)
        page = self.get(f"/predict/preparing/{uid}/", "basic")[2].decode("utf-8")
        self.assertIn("boom", page)
        self.assertIn("Try again", page)
        # a failure half way (one tile linked, then the disk goes) leaves the staged names alone
        real_place, calls = P.PS._place, []

        def flaky(src, dst, md5, link):
            if calls:
                raise OSError("disk went away")
            calls.append(src)
            return real_place(src, dst, md5, link)
        with mock.patch.object(P.PS, "_place", flaky):
            self.assertEqual(self.post(f"/predict/preparing/{uid}/retry", {}, "basic")[0], 303)
            j = self.wait_task(uid)
        self.assertEqual(j["state"], "failed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.sets(), before)
        self.assertEqual(len(list((self.staging(uid) / "files").iterdir())), 2)
        self.assertEqual([p.name for p in (self.root / ".incoming").iterdir()] if (self.root / ".incoming").exists()
                         else [], [])
        self.assertEqual(self.post(f"/predict/preparing/{uid}/retry", {}, "other")[0], 404)
        # try again: it works from the kept upload, and then the staging is gone
        self.assertEqual(self.post(f"/predict/preparing/{uid}/retry", {}, "basic")[0], 303)
        j = self.wait_task(uid)
        self.assertEqual(j["state"], "done")
        self.assertEqual(len(set(self.sets()) - set(before)), 1)
        self.assertFalse(self.staging(uid).exists())
        self.wait_state(j["href"].strip("/").split("/")[-1])

    def test_pressing_create_twice_is_one_task(self):
        uid, _ = self.stage_upload(2)
        real, gate = P.PS.create, threading.Event()

        def slow(*a, **k):
            gate.wait(30)
            return real(*a, **k)
        before = set(self.sets())
        with mock.patch.object(P.PS, "create", slow):
            hrefs = [json.loads(self.jpost(f"/predict/api/upload/{uid}/create", {})[2])["href"] for _ in range(2)]
            self.assertEqual(hrefs[0], hrefs[1])
            gate.set()
            self.assertEqual(self.wait_task(uid)["state"], "done")
        self.assertEqual(len(set(self.sets()) - before), 1)
        self.wait_state((set(self.sets()) - before).pop())

    # ---- limits
    def test_total_upload_bytes_cap(self):
        files = [{"name": "a.png", "size": 600}, {"name": "b.png", "size": 600}]
        with mock.patch.object(P, "TILES_MAX_TOTAL_BYTES", 1000):
            with self.assertRaises(P.Refusal) as cm:
                P.declared_files("tiles", files)
            self.assertEqual(cm.exception.status, 413)
            self.assertIn("Upload fewer images", cm.exception.msg)
            self.assertEqual(sum(P.declared_files("tiles", [{"name": "a.png", "size": 1000}]).values()), 1000)
            f = {"sample_label": "S", "nm_per_px": "1.4", "confirm": "yes", "model": MODEL}
            st, _, body = self.jpost("/predict/api/upload", {"kind": "tiles", "fields": f, "files": files})
            self.assertEqual(st, 413, body)

    def test_free_disk_check_copies_vs_links(self):
        src = self.src_a
        need = P.tile_bytes(src)
        self.assertGreater(need, 0)
        with mock.patch.object(P, "disk_free", return_value=int(need * 1.1) - 1):
            with self.assertRaises(P.Refusal) as cm:
                P.check_create_disk(self.root, src, None, False)
            self.assertEqual(cm.exception.status, 507)
            self.assertIn("Not enough disk space", cm.exception.msg)
            P.check_create_disk(self.root, src, None, True)              # same drive + link: nothing is copied
        with mock.patch.object(P, "disk_free", return_value=int(need * 1.1) + 1):
            P.check_create_disk(self.root, src, None, False)             # control: 110 % fits

    def test_session_pick_refused_when_disk_is_short(self):
        before = self.sets()
        with mock.patch.object(P, "disk_free", return_value=10):
            st, loc, _ = self.raw_submit()
            self.assertEqual(st, 303)
            j = self.wait_task(loc.rstrip("/").split("/")[-1])
        self.assertEqual(j["state"], "failed")
        self.assertEqual(self.sets(), before)
        self.assertIn("Not enough disk space", self.get(loc, "basic")[2].decode("utf-8"))

    # ---- the estimate on the forms
    def test_forms_show_the_estimate_and_name(self):
        page = self.page(f"/predict/new/session/{self.key_a}/", "basic")
        self.assertRegex(page, r'id="cj-predict-est" data-a="[\d.]+" data-b="[\d.]+" data-ahead="\d+"')
        self.assertIn("min once it starts", page)
        self.assertIn('data-max="1000"', page)
        self.assertRegex(page, r'<input name="name" maxlength="80" value="Session 20261027')
        up = self.page("/predict/new/upload/", "basic")
        self.assertIn('id="cj-predict-est"', up)
        self.assertRegex(up, r'<input name="name" maxlength="80" value="Uploaded images · ')
        js = self.get("/predict/predict.js")[2].decode("utf-8")
        self.assertNotIn("|| 200", js)
        self.assertIn("dataset.max", js)

    def test_queue_line_when_busy(self):
        made = [self.fake_set("bas2", s) for s in ("queued", "running")]
        for ps in made:
            self.addCleanup(shutil.rmtree, self.root / ps, True)
        P._EST_CACHE.clear()
        page = self.page(f"/predict/new/session/{self.key_a}/", "admin")
        self.assertRegex(page, r"min once it starts · \d+ analyses ahead of you")

    # ---- names
    def test_name_is_cleaned_escaped_and_shown_everywhere(self):
        before = set(self.sets())
        bad = "  <b>x</b> " + XSS + "\u202e\u200b\u2066 end\x07 " + "y" * 90
        st, loc, _ = self.raw_submit(name=bad)
        self.assertEqual(st, 303)
        ps = self.wait_task(loc.rstrip("/").split("/")[-1])["href"].strip("/").split("/")[-1]
        req = json.loads((self.root / ps / "request.json").read_text(encoding="utf-8"))
        name = req["name"]
        self.assertEqual(len(name), 80)
        for ch in "\u202e\u200b\u2066\x07":
            self.assertNotIn(ch, name)
        self.assertTrue(name.startswith("<b>x</b> <img src=x onerror=alert(1)> end"))
        page = self.page(f"/predict/{ps}/", "basic")
        self.assertIn(html.escape(name[:40]), page)
        self.assertNotIn(XSS, page)
        self.assertNotIn("<b>x</b>", page)
        self.assertIn(f"<title>{html.escape(name)}", page)
        mine = self.page("/predict/", "basic")
        self.assertIn(html.escape(name[:40]), mine)
        self.assertNotIn(XSS, mine)
        item = next(i for i in json.loads(self.get("/predict/api/mine", "basic")[2])["items"] if i["ps"] == ps)
        self.assertEqual(item["title"], name)
        self.wait_state(ps)
        self.assertEqual(len(set(self.sets()) - before), 1)

    def test_no_name_keeps_request_json_without_it_and_default_title(self):
        st, loc, _ = self.raw_submit(name="")
        ps = self.wait_task(loc.rstrip("/").split("/")[-1])["href"].strip("/").split("/")[-1]
        req = json.loads((self.root / ps / "request.json").read_text(encoding="utf-8"))
        self.assertNotIn("name", req)
        page = self.page(f"/predict/{ps}/", "basic")
        self.assertRegex(page, r"<h1>Session 20261027 \(EV prep A [^<]*\) · \d")
        self.assertNotIn(str(self.src_a), page)
        self.wait_state(ps)

    # ---- rename
    def rename(self, ps, name, who="basic"):
        return self.post(f"/predict/{ps}/rename", {"name": name}, who)

    def test_rename_owner_admin_other_and_method(self):
        ps = self.fake_set("bas", "done")
        self.addCleanup(shutil.rmtree, self.root / ps, True)
        page = self.page(f"/predict/{ps}/", "basic")
        self.assertIn(f'action="/predict/{ps}/rename"', page)
        with mock.patch.object(P, "audit_event") as audit:
            st, h, _ = self.rename(ps, "Owner  <i>name</i>")
            self.assertEqual((st, T.hget(h, "location")), (303, f"/predict/{ps}/"))
            self.assertEqual(audit.call_args.args[2:4], ("predict.rename", ps))
        req = json.loads((self.root / ps / "request.json").read_text(encoding="utf-8"))
        self.assertEqual(req["name"], "Owner  <i>name</i>")
        self.assertEqual(req["owner"], "bas")                                  # nothing else changed
        self.assertIn("<h1>Owner  &lt;i&gt;name&lt;/i&gt;</h1>", self.page(f"/predict/{ps}/", "basic"))
        self.assertEqual(self.rename(ps, "Admin did it", "admin")[0], 303)
        self.assertEqual(json.loads((self.root / ps / "request.json").read_text("utf-8"))["name"], "Admin did it")
        self.assertEqual(self.rename(ps, "Nope", "other")[0], 404)
        self.assertEqual(self.get(f"/predict/{ps}/rename", "basic")[0], 405)
        self.assertEqual(self.rename("ps_20260101_000000", "x")[0], 404)
        self.assertEqual(self.post(f"/predict/{ps}/rename", {"name": "x"}, "basic", origin=False)[0], 403)
        self.assertEqual(json.loads((self.root / ps / "request.json").read_text("utf-8"))["name"], "Admin did it")
        self.assertEqual(self.rename(ps, " \u200b ")[0], 303)                  # empty: back to the default
        self.assertNotIn("name", json.loads((self.root / ps / "request.json").read_text("utf-8")))

    def test_concurrent_renames_leave_valid_json(self):
        ps = self.fake_set("bas", "done")
        self.addCleanup(shutil.rmtree, self.root / ps, True)
        d = self.root / ps
        errs = []

        def go(i):
            try:
                for k in range(15):
                    P.rename_set(self.cfg(), {"id": "bas"}, d, f"name {i} {k}")
                    json.loads((d / "request.json").read_text(encoding="utf-8"))
            except Exception as e:      # noqa: BLE001
                errs.append(e)
        ts = [threading.Thread(target=go, args=(i,)) for i in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(errs, [])
        req = json.loads((d / "request.json").read_text(encoding="utf-8"))
        self.assertRegex(req["name"], r"^name \d \d+$")
        self.assertEqual((req["owner"], req["ps"]), ("bas", ps))
        self.assertEqual([p.name for p in d.iterdir() if p.name.endswith(".tmp")], [])


if __name__ == "__main__":
    unittest.main()
