"""
portal/listing.py (round 13, #22): facets for every source shape (session, upload, a derived_from chain, a dataset
root, a cycle, a depth cap, another person's private parent hidden through records.ref_hidden), parse_query's strict
fallbacks, every filter / sort / group combination on a fixture set checked against a plain reference, the sample
grouping rule, hub rows without facets, and the escaped GET form and list HTML.

Temp predict root + empty fixture registry only (CRYOEV_LAPTOP_ROOT points at the temp folder, so ref_hidden's
request.json owner lookup never reads the real CryoAI-laptop).

  python -m unittest portal.tests.test_listing -v
"""
import csv
import itertools
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import listing as L  # noqa: E402
from portal import names as NM  # noqa: E402
from portal.modules.data import records as REC  # noqa: E402

XSS = '<img src=x onerror=alert(1)>'


def wjson(p: Path, doc):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")


def make_set(root: Path, ps, owner="me", *, created="2026-11-10T10:00:00", name=None, session=None, via=None,
             model="v5", parents=(), samples=("S1",), n_images=None):
    req = {"ps": ps, "owner": owner, "model": model, "created": created,
           "source": {"kind": "session", "session": session} if session else {"kind": "upload"}}
    if name:
        req["name"] = name
    if via:
        req["owner_via"] = via
    if parents:
        req["provenance"] = {"derived_from": [{"kind": k, "id": i} for k, i in parents]}
    d = root / ps
    wjson(d / "request.json", req)
    n = n_images if n_images is not None else len(samples)
    with open(d / "images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "file", "sample_label", "nm_per_px"])
        for k in range(n):
            w.writerow([f"{k + 1:04d}", f"images/{k + 1:04d}.png", samples[k % len(samples)] if samples else "", "1.47"])
    return d, req


def can_see(user, owner):
    return user.get("role") == "admin" or (owner is not None and owner == user.get("id"))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_listing_"))
        self.root = self.tmp / "predict"
        self.root.mkdir()
        patch = T.TK.env_patch(strip_all=True, CRYOEV_LAPTOP_ROOT=str(self.tmp))
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(shutil.rmtree, self.tmp, True)
        REC._owner_cache.clear()
        self.records = {e: {} for e in REC.ENTITIES}

    def hidden(self, who="me", role="basic"):
        viewer = REC.Viewer(can_see, {"id": who, "role": role})
        return lambda kind, rid: REC.ref_hidden(self.records, viewer, rid, kind)

    def fac(self, ps, who="me", role="basic"):
        d = self.root / ps
        req = json.loads((d / "request.json").read_text(encoding="utf-8"))
        return L.facets(d, req, hidden=self.hidden(who, role))


class Facets(Base):
    def test_session_and_upload(self):
        make_set(self.root, "ps_20261110_aaaaa1", session="20261106-2", via="claude", model="v5_m")
        make_set(self.root, "ps_20261112_aaaaa2", created="2026-11-12T16:52:37")
        f = self.fac("ps_20261110_aaaaa1")
        self.assertEqual((f["session"], f["person"], f["via"], f["model"]), ("20261106-2", "me", "claude", "v5_m"))
        self.assertAlmostEqual(f["created"], datetime(2026, 11, 10, 10, 0).timestamp())
        g = self.fac("ps_20261112_aaaaa2")
        self.assertEqual(g["session"], "upload · 2026-11-12")
        self.assertIsNone(g["via"])
        self.assertEqual(set(f), {"person", "via", "created", "session", "source", "model", "samples"})

    def test_no_chain_is_its_own_name(self):
        make_set(self.root, "ps_20261110_aaaaa1", name="Poster spot-check: HEK")
        self.assertEqual(self.fac("ps_20261110_aaaaa1")["source"], "Poster spot-check: HEK")
        _, req = make_set(self.root, "ps_20261110_aaaaa3")
        self.assertEqual(self.fac("ps_20261110_aaaaa3")["source"], NM.display_name(req))

    def test_derived_chain_walks_to_its_root(self):
        make_set(self.root, "ps_20261110_00000a", name="Root A")
        make_set(self.root, "ps_20261111_00000b", name="B", parents=[("model", "v5"), ("prediction_set",
                                                                                         "ps_20261110_00000a")])
        make_set(self.root, "ps_20261112_00000c", name="C", parents=[("prediction_set", "ps_20261111_00000b")])
        for ps in ("ps_20261110_00000a", "ps_20261111_00000b", "ps_20261112_00000c"):
            self.assertEqual(self.fac(ps)["source"], "Root A", ps)

    def test_dataset_root(self):
        make_set(self.root, "ps_20261110_00000a", name="Jul30 control",
                 parents=[("dataset", "roboflow_20261028_v5")])
        make_set(self.root, "ps_20261111_00000b", name="B", parents=[("prediction_set", "ps_20261110_00000a")])
        self.assertEqual(self.fac("ps_20261110_00000a")["source"], "roboflow_20261028_v5 (dataset)")
        self.assertEqual(self.fac("ps_20261111_00000b")["source"], "roboflow_20261028_v5 (dataset)")

    def test_only_prediction_set_and_dataset_links_are_followed(self):
        make_set(self.root, "ps_20261110_00000a", name="A", parents=[("session", "20261106"), ("label_version", "lv1"),
                                                                     ("other", "x"), ("model", "v5")])
        self.assertEqual(self.fac("ps_20261110_00000a")["source"], "A")

    def test_cycle_lands_in_one_group(self):
        make_set(self.root, "ps_20261110_00000a", name="A", parents=[("prediction_set", "ps_20261111_00000b")])
        make_set(self.root, "ps_20261111_00000b", name="B", parents=[("prediction_set", "ps_20261110_00000a")])
        make_set(self.root, "ps_20261112_00000c", name="C", parents=[("prediction_set", "ps_20261112_00000c")])
        self.assertEqual(self.fac("ps_20261110_00000a")["source"], "A")
        self.assertEqual(self.fac("ps_20261111_00000b")["source"], "A")       # smallest id of the cycle names it
        self.assertEqual(self.fac("ps_20261112_00000c")["source"], "C")       # a self-link

    def test_depth_cap(self):
        ids = [f"ps_20261110_{k:06x}" for k in range(L.MAX_DEPTH + 5)]
        make_set(self.root, ids[0], name="deep root")
        for k in range(1, len(ids)):
            make_set(self.root, ids[k], name=f"n{k}", parents=[("prediction_set", ids[k - 1])])
        t0 = time.monotonic()
        self.assertEqual(self.fac(ids[-1])["source"], f"n{len(ids) - 1 - L.MAX_DEPTH}")
        self.assertEqual(self.fac(ids[L.MAX_DEPTH])["source"], "deep root")
        self.assertLess(time.monotonic() - t0, 5)

    def test_missing_parent_folder(self):
        make_set(self.root, "ps_20261111_00000b", name="B", parents=[("prediction_set", "ps_20261110_dead00")])
        self.assertEqual(self.fac("ps_20261111_00000b", role="admin")["source"], "ps_20261110_dead00")
        # a basic user can't see an unknown owner (ref_hidden fails closed): the id is never quoted
        self.assertEqual(self.fac("ps_20261111_00000b")["source"], "B")

    def test_other_persons_private_parent_is_hidden(self):
        make_set(self.root, "ps_20261110_0ff0e0", owner="other", name="Their secret set",
                 parents=[("dataset", "ds_secret")])
        make_set(self.root, "ps_20261111_00000b", name="Mine B", parents=[("prediction_set", "ps_20261110_0ff0e0")])
        make_set(self.root, "ps_20261112_00000c", name="Mine C", parents=[("prediction_set", "ps_20261111_00000b")])
        for ps, want in (("ps_20261111_00000b", "Mine B"), ("ps_20261112_00000c", "Mine B")):
            f = self.fac(ps)
            self.assertEqual(f["source"], want)
            blob = json.dumps(f)
            for leak in ("0ff0e0", "Their secret", "ds_secret", "other"):
                self.assertNotIn(leak, blob)
        self.assertEqual(self.fac("ps_20261112_00000c", role="admin")["source"], "ds_secret (dataset)")  # control
        self.assertEqual(self.fac("ps_20261112_00000c", who="other")["source"], "Mine C")   # can't see mine either

    def test_registry_record_owner_is_used(self):
        self.records["prediction_set"]["ps_20261110_0ff0e0"] = {"id": "ps_20261110_0ff0e0", "owner": "other"}
        make_set(self.root, "ps_20261110_0ff0e0", owner="me", name="On disk says me")   # the registry wins
        make_set(self.root, "ps_20261111_00000b", name="B", parents=[("prediction_set", "ps_20261110_0ff0e0")])
        self.assertEqual(self.fac("ps_20261111_00000b")["source"], "B")
        self.assertEqual(self.fac("ps_20261111_00000b", who="other")["source"], "On disk says me")

    def test_hidden_none_fails_closed(self):
        make_set(self.root, "ps_20261110_00000a", name="A")
        d, req = make_set(self.root, "ps_20261111_00000b", name="B", parents=[("prediction_set", "ps_20261110_00000a")])
        self.assertEqual(L.facets(d, req)["source"], "B")

    def test_hidden_none_hides_private_looking_keys(self):
        d, req = make_set(self.root, "ps_20261110_00000a", session="ps_20261109_abcdef", model="pred_ps_20261109_abcdef")
        f = L.facets(d, req)
        self.assertEqual((f["session"], f["model"]), (L.HIDDEN_LABEL, L.HIDDEN_LABEL))
        self.assertNotIn("abcdef", json.dumps([f["session"], f["model"]]))   # (source = the set's own name)
        d, req = make_set(self.root, "ps_20261110_00000b", session="20261106-2", model="v5_evnonev_y8m_20261029")
        f = L.facets(d, req)                                                      # control: ordinary keys shown
        self.assertEqual((f["session"], f["model"]), ("20261106-2", "v5_evnonev_y8m_20261029"))
        for v in ("ps_20261110_abcdef", "PRED_x", "see pred_ps_1", "x/ps_20261110_abcdef"):
            self.assertEqual(L._checked(None, "session", v), L.HIDDEN_LABEL, v)
        for v in ("20261106", "v5", "upload · 2026-11-13", "caps_20261110_abcdefx"):
            self.assertEqual(L._checked(None, "session", v), v, v)
        self.assertEqual(L._checked(lambda k, i: False, "session", "ps_20261110_abcdef"), "ps_20261110_abcdef")

    def test_a_broken_hidden_check_fails_closed(self):
        make_set(self.root, "ps_20261110_00000a", name="A")
        d, req = make_set(self.root, "ps_20261111_00000b", name="B", parents=[("prediction_set", "ps_20261110_00000a")],
                          session="20261106")

        def boom(kind, rid):
            raise RuntimeError("x")
        f = L.facets(d, req, hidden=boom)
        self.assertEqual((f["source"], f["session"], f["model"]), ("B", L.HIDDEN_LABEL, L.HIDDEN_LABEL))

    def test_samples_cached_by_mtime(self):
        d, req = make_set(self.root, "ps_20261110_00000a", samples=("HEK", "GFP", "HEK"), n_images=5)
        self.assertEqual(L.facets(d, req)["samples"], ["HEK", "GFP"])
        p = d / "images.csv"
        p.write_text("image,file,sample_label\n0001,images/0001.png,EV only\n", encoding="utf-8")
        st = p.stat()
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        self.assertEqual(L.facets(d, req)["samples"], ["EV only"])
        p.unlink()
        self.assertEqual(L.facets(d, req)["samples"], [])

    def test_bad_request_fields(self):
        d = self.root / "ps_20261110_00000a"
        d.mkdir()
        f = L.facets(d, {"ps": d.name, "created": "not a date", "source": "x", "owner": 5,
                         "provenance": {"derived_from": "nope"}}, hidden=lambda k, i: False)
        self.assertEqual((f["created"], f["session"], f["person"], f["samples"]), (None, "upload · ?", None, []))


class Query(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(L.parse_query({}), L.ListQuery())
        self.assertEqual(L.parse_query({}, default_status="open").status, "open")
        self.assertEqual(L.parse_query(None), L.ListQuery())

    def test_strict_fallbacks(self):
        q = L.parse_query({"person": "<x>", "model": "a b", "status": "Open!", "sort": "evil", "group": "owner",
                           "from": "2026-13-01", "to": "11/12/2026", "q": "a\x00b"}, default_status="open")
        self.assertEqual(q, L.ListQuery(status="open"))
        q = L.parse_query({"person": "alyssia", "model": "v5_evnonev_y8m_20261029", "status": "ready",
                           "sort": "size", "group": "source", "from": "2026-11-01", "to": "2026-11-13",
                           "q": "  HEK  ", "agents": "0"})
        self.assertEqual(q, L.ListQuery("alyssia", False, "2026-11-01", "2026-11-13", "v5_evnonev_y8m_20261029",
                                        "ready", "HEK", "size", "source"))
        self.assertEqual(len(L.parse_query({"q": "x" * 500}).q), L.Q_MAX)

    def test_agents_checkbox_pairs(self):
        from starlette.datastructures import QueryParams
        self.assertTrue(L.parse_query(QueryParams("")).agents)                       # no form: default on
        self.assertFalse(L.parse_query(QueryParams("agents=0")).agents)              # unticked
        self.assertTrue(L.parse_query(QueryParams("agents=0&agents=1")).agents)      # ticked
        self.assertTrue(L.parse_query(QueryParams("agents=junk")).agents)

    def test_to_query_roundtrip(self):
        from starlette.datastructures import QueryParams
        for q in (L.ListQuery(), L.ListQuery(status="open"), L.ListQuery("pujan", False, "2026-11-01", "", "v5",
                                                                          "any", "a&b=c", "name", "sample")):
            for default in ("any", "open"):
                self.assertEqual(L.parse_query(QueryParams(q.to_query()), default_status=default), q)
        self.assertEqual(L.ListQuery().to_query(), "status=any")


ROWS_SPEC = [  # id, owner, via, created, model, status, n_images, session, source, samples, title
    ("ps_20261113_000001", "pujan", "claude", "2026-11-13T11:43:59", "v5", "ready", 26, "upload · 2026-11-13", "SKOV",
     ["SKOV-3"], "Sima SKOV-3 re-rendered"),
    ("ps_20261113_000002", "pujan", None, "2026-11-13T10:40:12", "v5", "ready", 910, "upload · 2026-11-13", "SKOV",
     ["SKOV-3 native"], "v5 native-res SKOV"),
    ("ps_20261112_000003", "pujan", None, "2026-11-12T16:52:37", "v4", "none", 32, "upload · 2026-11-12", "HEK",
     ["HEK293"], "Poster spot-check: HEK"),
    ("ps_20261110_000004", "alyssia", None, "2026-11-10T09:47:31", "v5", "error", 677, "20261106-2",
     "Session 20261106-2", ["EV-013-2", "14K2-A", "14K2-B", "HEK293"], "Session 20261106-2"),
    ("ps_20261106_000005", "pujan-web", None, "2026-11-06T11:24:55", "v4", "ready", 200, "20261012", "Session 20261012",
     ["26119 - LHD", "26120 - SHD", "26121 - SLD"], "Session 20261012"),
    ("ps_20261105_000006", "pujan", "claude", "2026-11-05T20:27:09", "v5", "waiting", 20, "20261012",
     "Session 20261012", [], "Uploaded images · 5 Nov"),
]


def spec_rows():
    out = []
    for i, o, v, c, m, s, n, se, so, sa, t in ROWS_SPEC:
        out.append({"id": i, "title": t, "status": s, "n_images": n,
                    "facets": {"person": o, "via": v, "created": datetime.fromisoformat(c).timestamp(),
                               "session": se, "source": so, "model": m, "samples": sa}})
    return out


def reference(rows, q):
    """A plain, independent reading of the filter / sort rules."""
    out = []
    for r in rows:
        f = r["facets"]
        day = datetime.fromtimestamp(f["created"]).strftime("%Y-%m-%d")
        if q.person != "any" and f["person"] != q.person:
            continue
        if not q.agents and f["via"]:
            continue
        if q.date_from and day < q.date_from or q.date_to and day > q.date_to:
            continue
        if q.model != "any" and f["model"] != q.model:
            continue
        if q.status == "open" and r["status"] in ("finished", "applied"):
            continue
        if q.status not in ("any", "open") and r["status"] != q.status:
            continue
        if q.q and q.q.lower() not in (r["title"] + " " + r["id"]).lower():
            continue
        out.append(r)
    if q.sort == "name":
        out.sort(key=lambda r: r["title"].lower())
    elif q.sort == "size":
        out.sort(key=lambda r: -r["n_images"])
    else:
        out.sort(key=lambda r: -r["facets"]["created"])
    return out


class ApplyGroup(unittest.TestCase):
    def test_every_combination(self):
        rows = spec_rows()
        n = 0
        for person, agents, status, model, sort, grp, dates, text in itertools.product(
                ("any", "pujan", "alyssia", "nobody"), (True, False), ("any", "ready", "error", "open"),
                ("any", "v5", "v4"), L.SORTS, L.GROUPS, (("", ""), ("2026-11-10", ""), ("", "2026-11-12"),
                                                         ("2026-11-12", "2026-11-12")), ("", "hek", "PS_2026111")):
            q = L.ListQuery(person, agents, dates[0], dates[1], model, status, text, sort, grp)
            got = L.apply(rows, q)
            want = reference(rows, q)
            self.assertEqual([r["id"] for r in got], [r["id"] for r in want], q)
            groups = L.group(got, grp)
            if grp == "none":
                self.assertEqual(groups, [("", got)])
            else:
                flat = [r["id"] for _, rs in groups for r in rs]
                self.assertEqual(sorted(set(flat)), sorted(r["id"] for r in got), q)   # every row in some group
                for label, rs in groups:
                    self.assertEqual([r["id"] for r in rs], [r["id"] for r in got if r in rs])   # order kept
            n += 1
        self.assertGreater(n, 5000)

    def test_group_keys(self):
        rows = L.apply(spec_rows(), L.ListQuery())
        self.assertEqual([k for k, _ in L.group(rows, "person")], ["pujan", "alyssia", "pujan-web"])
        self.assertEqual(dict((k, len(v)) for k, v in L.group(rows, "session")),
                         {"upload · 2026-11-13": 2, "upload · 2026-11-12": 1, "20261106-2": 1, "20261012": 2})
        self.assertEqual(dict((k, len(v)) for k, v in L.group(rows, "source"))["SKOV"], 2)
        self.assertEqual([k for k, _ in L.group(rows, "model")], ["v5", "v4"])

    def test_sample_rule(self):
        rows = L.apply(spec_rows(), L.ListQuery())
        g = dict(L.group(rows, "sample"))
        self.assertEqual([r["id"] for r in g[L.MIXED]], ["ps_20261110_000004"])          # 4 samples: mixed
        for s in ("26119 - LHD", "26120 - SHD", "26121 - SLD"):                           # exactly 3: each group
            self.assertEqual([r["id"] for r in g[s]], ["ps_20261106_000005"])
        self.assertEqual([r["id"] for r in g["HEK293"]], ["ps_20261112_000003"])         # not the mixed one
        self.assertEqual([r["id"] for r in g["No sample label"]], ["ps_20261105_000006"])

    def test_hub_rows_without_facets(self):
        rows = [{"id": "lab_a", "title": "Label set A", "status": "applied", "owner": "pujan", "via": None,
                 "last": 200.0, "n_images": 4, "facets": {}},
                {"id": "lab_b", "title": "Audit B", "status": "in_progress", "owner": None, "via": None,
                 "last": 300.0, "n_images": None, "facets": {}},
                {"id": "pred_x", "title": "Pred X", "status": "not_started", "owner": "alyssia", "via": "claude",
                 "last": 100.0, "n_images": 9}]                                    # no facets key at all
        self.assertEqual([r["id"] for r in L.apply(rows, L.ListQuery(status="open"))], ["lab_b", "pred_x"])
        self.assertEqual([r["id"] for r in L.apply(rows, L.ListQuery(person="pujan"))], ["lab_a"])
        self.assertEqual([r["id"] for r in L.apply(rows, L.ListQuery(agents=False))], ["lab_b", "lab_a"])
        self.assertEqual([r["id"] for r in L.apply(rows, L.ListQuery(sort="size"))], ["pred_x", "lab_a", "lab_b"])
        self.assertEqual([k for k, _ in L.group(rows, "person")], ["pujan", "Unknown", "alyssia"])
        self.assertEqual([k for k, _ in L.group(rows, "sample")], ["No sample label"])


class Html(unittest.TestCase):
    def test_controls(self):
        q = L.ListQuery("alyssia", False, "2026-11-01", "2026-11-13", "v5", "ready", XSS, "size", "source")
        h = L.controls_html(q, action="/results/", people=["pujan", "alyssia", XSS], models=["v5", "v4"],
                            statuses=[("ready", "Results ready"), ("none", "Not computed yet")])
        self.assertNotIn(XSS, h)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", h)
        for cid in ("cj-list-form", "cj-list-person", "cj-list-agents", "cj-list-from", "cj-list-to", "cj-list-model",
                    "cj-list-status", "cj-list-q", "cj-list-sort", "cj-list-sort-newest", "cj-list-sort-name",
                    "cj-list-sort-size", "cj-list-group", "cj-list-go", "cj-list-reset"):
            self.assertEqual(h.count(f'id="{cid}"'), 1, cid)
        self.assertIn('method="get" action="/results/"', h)
        self.assertIn('<option value="alyssia" selected>', h)
        self.assertIn('<option value="v5" selected>', h)
        self.assertIn('<option value="ready" selected>Results ready</option>', h)
        self.assertIn('<option value="source" selected>Source images</option>', h)
        self.assertIn('id="cj-list-sort-size" name="sort" value="size" checked', h)
        self.assertIn('value="2026-11-01"', h)
        self.assertRegex(h, r'<input type="hidden" name="agents" value="0"><input type="checkbox" id="cj-list-agents" '
                            r'name="agents" value="1">')                       # unticked: no "checked"
        d = L.controls_html(L.ListQuery(), action="/predict/", people=[], models=[], statuses=[], show_group=False)
        self.assertIn('id="cj-list-agents" name="agents" value="1" checked', d)   # default: on
        for absent in ("cj-list-person", "cj-list-model", "cj-list-status", "cj-list-group"):
            self.assertNotIn(f'id="{absent}"', d)
        self.assertNotIn("<script", h + d)

    def test_rows_html(self):
        rows = []
        for r in spec_rows():
            f = r["facets"]
            rows.append({**r, "title": r["title"] + (XSS if r["id"].endswith("1") else ""), "href": f"/x/{r['id']}/",
                         "open_href": f"/x/{r['id']}/", "owner": f["person"], "via": f["via"], "model": f["model"],
                         "samples": f["samples"], "created": f["created"], "status_html": "<i>s</i>"})
        q = L.ListQuery(group="session")
        got = L.apply(rows, q)
        h = L.rows_html(L.group(got, "session"), "session", shown=len(got), total=len(rows))
        self.assertNotIn(XSS, h)
        self.assertEqual(h.count("<details class=\"cj-list-group\" open"), 4)
        self.assertIn("· 2 sets", h)
        self.assertIn("936 images", h)                                     # 26 + 910 in upload · 2026-11-13
        self.assertIn("claude for pujan", h)
        self.assertIn('title="EV-013-2; 14K2-A; 14K2-B; HEK293"', h)
        self.assertIn('id="cj-list-count">6 of 6 sets', h)
        h2 = L.rows_html(L.group(got, "source"), "source", shown=len(got), total=len(rows))
        self.assertIn("Source: SKOV", h2)
        flat = L.rows_html([("", got)], "none", shown=len(got), total=len(rows))
        self.assertNotIn("<details", flat)
        self.assertEqual(flat.count('<li class="cj-list-row'), 6)
        empty = L.rows_html([("", [])], "none", shown=0, total=6, empty_text="Nothing <here>")
        self.assertIn("Nothing &lt;here&gt;", empty)

    def test_choices_only_from_given_rows(self):
        people, models, n_agent = L.choices(spec_rows())
        self.assertEqual(people, ["alyssia", "pujan", "pujan-web"])
        self.assertEqual(models, ["v4", "v5"])
        self.assertEqual(n_agent, 2)


if __name__ == "__main__":
    unittest.main()
