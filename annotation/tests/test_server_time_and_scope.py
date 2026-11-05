"""
Server-time merge (TODO #53) and the sandbox's own localStorage key.

  python -m unittest discover -s annotation/tests -v

Python (live server on the fixture, and the record functions):
  - a decision the server receives later wins, whatever the client clocks say (a device a day ahead used to win);
  - records saved before server_t existed (no server_t, no pending) merge byte-identical to the merge before it
    (HEAD_merge_edits below is a frozen copy of it), from the page's PUTs and from an old cached page's;
  - every saved decision's who goes through review_server._request_identity (today: the name picker's value).
Browser (Playwright, Edge or Chrome; skipped without one):
  - the Editor on a device whose clock runs a day ahead doesn't override another reviewer's later save;
  - the page's mergeEdit() equals merge_edits() on stamped / pending / old entries;
  - a sandbox page keeps its drafts under its own key, so a real review on the same origin never uploads them;
    a real package keeps the exact old key ("cryoev-review:" + id), so in-progress drafts are picked up.
Fixtures are made here; no real package is read.
"""
import copy
import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fixture  # noqa: E402
from test_review_server import KEY, LiveServer, rs  # noqa: E402

DAY = 86_400_000
T0 = 1_761_000_000_000  # ms: when the older records below were "saved"


# ---- frozen copy of merge_edits / _migrate before server_t (commit 8e9f9f6): the byte-identity reference ----
def HEAD_migrate(e):
    e = dict(e or {})
    e["dec"] = {k: (v if isinstance(v, dict) else {"v": v, "t": e.get("t", 0), "who": ""})
                for k, v in (e.get("dec") or {}).items()}
    if e.get("gt") is not None and not e.get("gtT"):
        if e.get("modified"):
            e["gtT"] = e.get("t", 0) or 1
        else:
            e.pop("gt", None)
    return e


def HEAD_merge_edits(a, b):
    if not a:
        return HEAD_migrate(b)
    a, b = HEAD_migrate(a), HEAD_migrate(b)
    out = {"dec": {}, "t": max(a.get("t", 0), b.get("t", 0)), "modified": bool(a.get("modified") or b.get("modified"))}
    for k in dict.fromkeys([*a["dec"], *b["dec"]]):
        x, y = a["dec"].get(k), b["dec"].get(k)
        out["dec"][k] = y if x is None else x if y is None else (y if y.get("t", 0) > x.get("t", 0) else x)
    if a.get("gtT") or b.get("gtT"):
        s = b if (b.get("gtT") or 0) > (a.get("gtT") or 0) else a
        out["gt"], out["gtT"] = s.get("gt"), s.get("gtT")
    ra, rb = a.get("reviewedT", 0) or 0, b.get("reviewedT", 0) or 0
    out["reviewed"] = bool(b.get("reviewed")) if rb > ra else bool(a.get("reviewed"))
    out["reviewedT"] = max(ra, rb)
    return out


def old_versions():
    """Edit records in every format saved before server_t, per image: several reviewers' copies of one image, with
    older, newer and equal t on shared keys. Shapes follow docs/REVIEW_TOOL.md and the September / R1 records:
    bare strings, {v, t, who}, + p, + seed, null, new:, a polygon list without and with per-item t, gt without gtT."""
    R = make_fixture.ring
    gtA = [{"id": 101, "c": "EV", "p": [R(70, 70, 28)]}, {"id": 102, "c": "inner", "p": [R(70, 70, 16)]},
           {"id": 103, "c": "MV", "p": [R(200, 80, 30)]}, {"id": 104, "c": "MV", "p": [R(290, 170, 24)]}]
    moved = [round(v + 0.01, 5) for v in R(200, 80, 30)]
    return {
        "fixB_old.png": [
            {"dec": {"p1": "add:MV", "g202": "del", "g201": "keep:inner"}, "t": T0, "modified": True, "reviewed": False},
            {"dec": {"g201": {"v": "keep:EV", "t": T0 + 50, "who": "ann"}, "g202": {"v": None, "t": T0 - 10, "who": "bob"},
                     "p0": {"v": "no", "t": T0 + 5, "who": "ann"}},
             "t": T0 + 50, "modified": True, "reviewed": True, "reviewedT": T0 + 60},
            {"dec": {"g201": {"v": "keep:MV", "t": T0 + 50, "who": "cat"},          # tie with ann: the saved copy stays
                     "p1": {"v": "add:VLDL", "t": T0 + 80, "who": "cat", "p": [R(180, 210, 15)]}},
             "t": T0 + 80, "reviewed": False, "reviewedT": T0 + 70},
        ],
        "fixA_new.png": [
            {"dec": {"g103": {"v": "keep:EV", "t": T0 + 10, "who": "old"}, "p3": {"v": "no", "t": T0 + 10, "who": "old"},
                     "p2": {"v": "add:MV", "t": T0 + 10, "who": "", "p": [moved]}},
             "gt": gtA, "gtT": T0 + 5, "modified": True, "reviewed": True, "reviewedT": T0 + 20, "t": T0 + 20},
            {"dec": {"g103": {"v": "keep:non-EV", "t": T0 + 30, "who": "Pujan", "seed": "september:v4_x.png:g7"},
                     "g104": {"v": "keep:Low conf EV", "t": T0 + 30, "who": "September review",
                              "seed": "september:v4_x.png:g8"},
                     "nk1": {"v": "new:EV", "t": T0 + 40, "who": "au1", "p": [R(300, 40, 12)]}},
             "gt": [dict(g, t=T0 + 40) for g in gtA[:3]] + [{"id": None, "c": "EV", "p": [R(300, 40, 12)], "src": "nk1",
                                                             "t": T0 + 40}],
             "gtT": T0 + 40, "modified": True, "t": T0 + 40},
            {"dec": {"nk1": {"v": "del", "t": T0 + 45, "who": "bob"}, "g103": {"v": "keep:MV", "t": T0 + 20, "who": "bob"}},
             "t": T0 + 45, "modified": True, "reviewed": False, "reviewedT": T0 + 45},
        ],
        "fixC_old.png": [
            {"dec": {"p0": {"v": None, "t": T0, "who": "x"},
                     "nold1": {"v": "new:inner", "t": T0, "who": "x", "p": [R(60, 200, 15)]}}, "t": T0, "modified": True},
            {"dec": {}, "gt": [{"id": 301, "c": "MV", "p": [R(120, 100, 30)]}], "modified": True, "t": T0 + 7},   # gtT = t
            {"dec": {"g302": "keep:EV"}, "gt": [{"id": 302, "c": "EV", "p": [R(260, 150, 22)]}], "t": T0 + 9},  # list ignored
        ],
    }


def dumps(o):
    return json.dumps(o)


class OldFormatMergeIsUnchanged(unittest.TestCase):
    """Records without server_t merge byte-identical to the merge before it, with or without a server receive time."""

    def test_pairs_and_chains_byte_identical(self):
        n = 0
        for file, vers in old_versions().items():
            cands = [None, *vers]
            for a in cands:
                for b in vers:
                    want = dumps(HEAD_merge_edits(copy.deepcopy(a), copy.deepcopy(b)))
                    for now in (None, T0 + 10 * DAY):
                        with self.subTest(file=file, a=vers.index(a) if a else None, b=vers.index(b), now=now):
                            a0, b0 = copy.deepcopy(a), copy.deepcopy(b)
                            self.assertEqual(dumps(rs.merge_edits(a0, b0, now)), want)
                            self.assertEqual((a0, b0), (a, b))  # inputs untouched
                            n += 1
            for order in (vers, vers[::-1], vers[1:] + vers[:1]):  # a server receiving the copies one after another
                old = new = None
                for v in order:
                    old, new = HEAD_merge_edits(old, copy.deepcopy(v)), rs.merge_edits(new, copy.deepcopy(v), T0 + DAY)
                self.assertEqual(dumps(new), dumps(old), file)
                self.assertNotIn("server_t", dumps(new))
                self.assertEqual(dumps(rs.merge_edits(new, copy.deepcopy(new), T0 + DAY)), dumps(new))  # re-sent copy
        self.assertGreater(n, 50)

    def test_effective_labels_unchanged_on_old_records(self):
        """The new label rule (a decision with server_t applies over its list item) never fires on old records:
        labels equal those with that rule switched off. Control: it does fire on a stamped record."""
        data = {im["file"]: im for im in make_fixture.build(Path(tempfile.mkdtemp()) / "pkg")["images"]}

        def both(im, edit):
            new = rs.effective_labels(im, edit)
            orig = rs._stamped
            rs._stamped = lambda d: False  # the rule before server_t
            try:
                return new, rs.effective_labels(im, edit)
            finally:
                rs._stamped = orig
        for file, vers in old_versions().items():
            for order in (vers, vers[::-1]):
                m = None
                for v in order:
                    m = rs.merge_edits(m, copy.deepcopy(v), T0 + DAY)
                    new, old = both(data[file], m)
                    self.assertEqual(new, old, file)
        im = data["fixA_new.png"]
        lst = [dict(g, t=T0 + DAY) for g in im["gt"]]
        stamped = {"t": T0, "gt": lst, "gtT": T0 + DAY, "modified": True,
                   "dec": {"g101": {"v": "keep:MV", "t": T0, "who": "b", "server_t": T0 + 5}}}
        new, old = both(im, stamped)
        self.assertEqual(({g["id"]: g["c"] for g in new}[101], {g["id"]: g["c"] for g in old}[101]), ("MV", "EV"))


def write_old_package(pkg):
    """The fixture with the old records already saved: one image in the bare {file, edit} form (September), the others
    in the {file, pkg, fmt, edit, rx} wrapper, as review_server.py wrote them before server_t."""
    make_fixture.build(pkg)
    (pkg / "edits").mkdir(exist_ok=True)
    for file, vers in old_versions().items():
        m = HEAD_merge_edits(HEAD_merge_edits(None, vers[0]), vers[0])  # as the server stores a record after a PUT
        rec = ({"file": file, "edit": vers[0]} if file == "fixB_old.png" else
               {"file": file, "pkg": "fixture_review", "fmt": 1, "edit": m, "rx": {k: T0 for k in m["dec"]}})
        (pkg / "edits" / f"{file}.json").write_text(json.dumps(rec), encoding="utf-8")


class OldPackageOnTheNewServer(unittest.TestCase):
    """A package saved before server_t, served by this version: loads as before; re-sent copies change no byte; an old
    cached page's PUTs (no pending, no server_t) store exactly what the old server stored."""

    @classmethod
    def setUpClass(cls):
        cls.srv = LiveServer(write_old_package)

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()

    def files(self):
        return {p.name: p.read_bytes() for p in (self.srv.pkg / "edits").glob("*.json")}

    def test_load_resend_and_old_client_puts(self):
        before = self.files()
        served = self.srv.request("GET", "/api/edits")[1]
        self.assertEqual(served, {f: json.loads(b)["edit"] for f, b in ((json.loads(b)["file"], b) for b in before.values())})
        # the page re-sends what it pulled (wrapper-format records): not a byte changes
        for f in ("fixA_new.png", "fixC_old.png"):
            self.assertEqual(self.srv.put(f, served[f])[0], 200)
        self.assertEqual(self.files(), before)
        # an old cached page sends the other reviewers' copies: stored == the old server's merge, byte for byte
        for file, vers in old_versions().items():
            want = HEAD_merge_edits(None, vers[0])
            for v in vers[1:]:
                code, body, _ = self.srv.put(file, v)
                self.assertEqual(code, 200)
                want = HEAD_merge_edits(want, v)
                saved = json.loads((self.srv.pkg / "edits" / f"{file}.json").read_text(encoding="utf-8"))["edit"]
                self.assertEqual(dumps(saved), dumps(want), file)
                self.assertEqual(dumps(body["edit"]), dumps(want), file)
            self.assertNotIn(b"server_t", (self.srv.pkg / "edits" / f"{file}.json").read_bytes())


class ServerTimeMerge(unittest.TestCase):
    """The page marks a decision it made "pending"; the server stamps server_t (its receive time) and merges by it."""

    @classmethod
    def setUpClass(cls):
        cls.srv = LiveServer()

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()

    def saved(self, file):
        return self.srv.request("GET", "/api/edits")[1][file]

    @staticmethod
    def dec(v, t, who, **kw):
        return {"v": v, "t": t, "who": who, "pending": True, **kw}

    def test_future_client_t_does_not_win_over_a_later_save(self):
        """A's clock runs a day ahead; B saves the same label later, without having seen A's decision. B wins.
        Control: on the merge before server_t (newest client t wins) A's future t kept A's decision."""
        f, now = "fixC_old.png", int(time.time() * 1000)
        self.assertEqual(self.srv.put(f, {"t": now + DAY, "dec": {"g301": self.dec("keep:MV", now + DAY, "ann")}})[0], 200)
        a = self.saved(f)["dec"]["g301"]
        time.sleep(0.01)
        self.assertEqual(self.srv.put(f, {"t": now, "dec": {"g301": self.dec("keep:inner", now, "bob")}})[0], 200)
        b = self.saved(f)["dec"]["g301"]
        self.assertEqual((b["v"], b["who"], b["t"]), ("keep:inner", "bob", now))   # the later save wins
        self.assertEqual((a["v"], a["t"]), ("keep:MV", now + DAY))           # the client's t is kept for the record
        self.assertNotIn("pending", a)
        self.assertLess(abs(a["server_t"] - now), 60_000)                     # the server's clock, not the client's
        self.assertGreater(b["server_t"], a["server_t"])
        # A's page re-sends its (acknowledged) copy with its next change: B's decision stays
        self.assertEqual(self.srv.put(f, {"t": now + DAY + 1, "dec": {"g301": a, "p0": self.dec("no", now + DAY + 1, "ann")}})[0], 200)
        m = self.saved(f)
        self.assertEqual((m["dec"]["g301"]["v"], m["dec"]["p0"]["v"]), ("keep:inner", "no"))
        labels = {g["id"]: g for g in rs.effective_labels(make_fixture_image(self.srv, f), m)}
        self.assertEqual(labels[301]["c"], "inner")

    def test_later_save_wins_over_a_future_editor_list(self):
        """The Editor's polygon list carries per-item client t: A's list (a day ahead) must not hide B's later
        decision on that label (a decision with server_t applies over the list item)."""
        f, now = "fixB_old.png", int(time.time() * 1000)
        im = make_fixture_image(self.srv, f)
        lst = [dict(g, t=now + DAY) for g in im["gt"]]
        lst[0] = dict(lst[0], c="MV")
        self.srv.put(f, {"t": now + DAY, "modified": True, "gt": lst, "gtT": now + DAY,
                         "dec": {"g201": self.dec("keep:MV", now + DAY, "ann")}})
        time.sleep(0.01)
        self.srv.put(f, {"t": now, "modified": True, "dec": {"g201": self.dec("keep:inner", now, "bob")}})
        m = self.saved(f)
        self.assertEqual(m["gtT"], now + DAY)                                  # A's list is still the base
        self.assertEqual({g["id"]: g["c"] for g in rs.effective_labels(im, m)}[201], "inner")

    def test_old_entry_with_a_future_t_is_replaced_for_good(self):
        """An entry saved before server_t with a future t (wrong clock): a new decision replaces it with server_t
        above that t, so a stale copy of the old entry (no server_t) re-sent by anyone can't bring it back."""
        f, now = "fixA_new.png", int(time.time() * 1000)
        old = {"v": "keep:EV", "t": now + DAY, "who": "old"}
        self.srv.put(f, {"t": now + DAY, "dec": {"g105": old}})                  # an old cached page: no pending
        self.assertNotIn("server_t", self.saved(f)["dec"]["g105"])              # old-format entries are never stamped
        self.srv.put(f, {"t": now, "dec": {"g105": self.dec("del", now, "bob")}})
        new = self.saved(f)["dec"]["g105"]
        self.assertEqual((new["v"], new["server_t"]), ("del", now + DAY + 1))
        self.srv.put(f, {"t": now + DAY, "dec": {"g105": old}})                  # the stale copy again
        self.assertEqual(self.saved(f)["dec"]["g105"], new)

    def test_resent_pending_entry_is_not_restamped(self):
        """The reply to a PUT was lost, so the page sends the same pending entry again: nothing changes."""
        f, now = "fixA_new.png", int(time.time() * 1000)
        d = self.dec("keep:VLDL", now, "ann")
        self.srv.put(f, {"t": now, "dec": {"g106": d}})
        first = (self.srv.pkg / "edits" / f"{f}.json").read_bytes()
        time.sleep(0.01)
        self.srv.put(f, {"t": now, "dec": {"g106": d}})
        self.assertEqual((self.srv.pkg / "edits" / f"{f}.json").read_bytes(), first)

    def test_server_fields_validated(self):
        ok = {"v": "del", "t": 1, "who": "a"}
        self.assertTrue(rs.valid_edit({"t": 1, "dec": {"g1": dict(ok, server_t=5, pending=False)}}))
        for bad in (dict(ok, server_t="5"), dict(ok, server_t=None), dict(ok, pending=1), dict(ok, pending="yes")):
            with self.subTest(bad=bad):
                self.assertFalse(rs.valid_edit({"t": 1, "dec": {"g1": bad}}))
        self.assertEqual(self.srv.put("fixC_old.png", {"t": 1, "dec": {"g1": dict(ok, server_t=float("nan"))}})[0], 400)


def make_fixture_image(srv, file):
    return next(i for i in srv.data["images"] if i["file"] == file)


class IdentityHook(unittest.TestCase):
    """Every saved decision's who and every claim's name pass through review_server._request_identity: today it
    returns the name picker's value unchanged; the portal will replace it with the gateway's verified user."""

    def setUp(self):
        self.pkg = Path(tempfile.mkdtemp()) / "pkg"
        data = make_fixture.build(self.pkg)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), rs.make_handler(
            self.pkg, KEY, {im["file"] for im in data["images"]}, data["id"]))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)
        self.http = LiveServer.request.__get__(type("S", (), {"port": self.srv.server_address[1]})())

    def test_today_the_picker_name_is_recorded(self):
        self.assertEqual(rs._request_identity(None, "ann"), "ann")
        self.http("PUT", "/api/edits?f=fixC_old.png", {"t": 1, "dec": {"g301": {"v": "del", "t": 1, "who": "ann", "pending": True}}})
        self.assertEqual(self.http("GET", "/api/edits")[1]["fixC_old.png"]["dec"]["g301"]["who"], "ann")

    def test_one_function_decides_who(self):
        orig = rs._request_identity
        rs._request_identity = lambda handler, claimed: "gateway-user"
        self.addCleanup(setattr, rs, "_request_identity", orig)
        self.http("PUT", "/api/edits?f=fixC_old.png", {"t": 1, "dec": {
            "g301": {"v": "del", "t": 1, "who": "ann", "pending": True},       # this request's decision
            "g302": {"v": "keep:EV", "t": 1, "who": "bob"}}})                   # an older entry it carries: as it was
        dec = self.http("GET", "/api/edits")[1]["fixC_old.png"]["dec"]
        self.assertEqual((dec["g301"]["who"], dec["g302"]["who"]), ("gateway-user", "bob"))
        claims = self.http("POST", "/api/claim?c=fixC_old.png%7Cg301&who=ann", b"")[1]["claims"]
        self.assertEqual(claims, {"fixC_old.png|g301": "gateway-user"})
        # the page's own copy (who "ann") is the same decision: re-sending it changes nothing
        self.http("PUT", "/api/edits?f=fixC_old.png", {"t": 1, "dec": {"g301": {"v": "del", "t": 1, "who": "ann", "pending": True}}})
        self.assertEqual(self.http("GET", "/api/edits")[1]["fixC_old.png"]["dec"], dec)


# ---------------------------------------------------------------- browser ----------------------------------------------------------------
from test_review_tool_browser import Base  # noqa: E402  (after the Python tests: it imports Playwright lazily)

FUTURE_CLOCK = "{ const now = Date.now.bind(Date); Date.now = () => now() + %d; }" % DAY


class SkewBrowserTests(Base):
    """Two Editors on one server; A's device clock runs a day ahead."""

    def test_future_clock_device_does_not_override_a_later_save(self):
        f = "fixC_old.png"
        g301 = self.image(f)["gt"][0]["p"][0]
        b = self.open("&who=bob&mode=editor")
        a = self.open("&who=ann&mode=editor", setup=lambda ctx, pg: ctx.add_init_script(FUTURE_CLOCK))
        for p in (a, b):
            self.go(p, f)
        self.click_obj(a, g301)
        a.keyboard.press("3")                                  # A: MV, t a day ahead
        self.synced(a)
        self.click_obj(b, g301)                                # B hasn't pulled A's change (pulls every 20 s)
        b.keyboard.press("4")                                  # B: inner, later
        self.synced(b)
        dec = self.saved_dec(f)["g301"]
        self.assertEqual((dec["v"], dec["who"]), ("keep:inner", "bob"))
        a.evaluate("__review.pull()")
        self.assertEqual(self.obj(a, f, 301)["c"], "inner")   # A's page agrees, and has nothing newer to upload
        self.assertEqual(self.state(a)["dirty"], 0)
        time.sleep(1.0)
        self.assertEqual(self.saved_dec(f)["g301"]["v"], "keep:inner")
        self.check_js_equals_python(a)
        self.check_js_equals_python(b)

    def test_page_merge_matches_python(self):
        st = lambda v, t, srv, who="a": {"v": v, "t": t, "who": who, "server_t": srv}
        pe = lambda v, t, who="a": {"v": v, "t": t, "who": who, "pending": True}
        lg = lambda v, t, who="a": {"v": v, "t": t, "who": who}
        cases = [
            ({"t": 1, "dec": {"g1": st("del", 9e12, 100)}}, {"t": 2, "dec": {"g1": st("keep:EV", 5, 200)}}),   # server_t
            ({"t": 1, "dec": {"g1": st("del", 9e12, 100)}}, {"t": 2, "dec": {"g1": pe("keep:EV", 5)}}),        # pending wins
            ({"t": 1, "dec": {"g1": pe("del", 9e12)}}, {"t": 2, "dec": {"g1": st("keep:EV", 5, 200)}}),        # pending kept
            ({"t": 1, "dec": {"g1": pe("del", 7)}}, {"t": 2, "dec": {"g1": st("del", 7, 300)}}),              # ack adopted
            ({"t": 1, "dec": {"g1": st("del", 7, 300)}}, {"t": 2, "dec": {"g1": pe("del", 7, "z")}}),         # same, who aside
            ({"t": 1, "dec": {"g1": lg("del", 50)}}, {"t": 2, "dec": {"g1": st("keep:EV", 5, 40)}}),          # old t vs server_t
            ({"t": 1, "dec": {"g1": lg("del", 50)}}, {"t": 2, "dec": {"g1": st("keep:EV", 5, 60)}}),
            ({"t": 1, "dec": {"g1": pe("del", 50)}}, {"t": 2, "dec": {"g1": pe("keep:EV", 60)}}),             # two pending
            (*old_versions()["fixB_old.png"][:2],), (*old_versions()["fixA_new.png"][1:],),
        ]
        page = self.open()
        js = page.evaluate("cases => cases.map(([a, b]) => __review.mergeEdit(a, b))", [list(c) for c in cases])
        for k, (a, b) in enumerate(cases):
            self.assertEqual(js[k], json.loads(json.dumps(rs.merge_edits(copy.deepcopy(a), copy.deepcopy(b)))), k)


class StorageScopeTests(Base):
    """The SANDBOX_ copy has the real package's id. Both pages here share one origin (the sandbox flag is added to
    data.js by a route, as the sandbox server would), the case a portal on one host creates."""

    def pages(self):
        context = self.browser.new_context()
        self.addCleanup(context.close)
        self.errors = []
        return context

    def page(self, context, sandbox, block_puts=False):
        page = context.new_page()
        page.on("pageerror", lambda e: self.errors.append(str(e)))
        if sandbox:
            def add_flag(route):
                r = route.fetch()
                route.fulfill(response=r, body=r.body() + rs.sandbox_js(True))
            page.route("**/data.js*", add_flag)
        if block_puts:
            page.route("**/api/edits?f=*", lambda r: r.abort() if r.request.method == "PUT" else r.continue_())
        return page

    def load(self, page, query):
        page.goto(self.srv.url + query)
        page.wait_for_function("window.__review && __review.state().imgReady && (__review.state().server || true)")
        page.wait_for_function("__review.state().server", timeout=15000)

    def keys(self, page):
        return page.evaluate("Object.keys(localStorage).filter(k => k.startsWith('cryoev-review:' + __review.D.id)).sort()")

    def test_sandbox_drafts_stay_out_of_the_real_review(self):
        """Control: with one shared key (before), the real page loaded the sandbox draft and uploaded it."""
        f = "fixC_old.png"
        pid = self.srv.data["id"]
        ctx = self.pages()
        sb = self.page(ctx, sandbox=True, block_puts=True)        # a sandbox draft that never reached a server
        self.load(sb, "&who=sbx&mode=editor")
        self.assertTrue(sb.locator("#sandbox").is_visible())
        self.go(sb, f)
        self.click_obj(sb, self.image(f)["gt"][0]["p"][0])
        sb.keyboard.press("3")
        sb.wait_for_function("__review.state().dirty > 0")
        time.sleep(0.6)                                          # persisted (250 ms debounce)
        sb.close()
        real = self.page(ctx, sandbox=False)
        self.load(real, "&who=tester&mode=editor")
        time.sleep(1.5)
        self.assertEqual(self.state(real)["dirty"], 0)
        self.assertNotIn("g301", self.saved_dec(f))               # the sandbox draft was not uploaded
        self.assertNotEqual(self.obj(real, f, 301)["c"], "MV")
        self.assertEqual(self.keys(real), [f"cryoev-review:{pid}", f"cryoev-review:{pid}:dirty",
                                           f"cryoev-review:{pid}:sandbox", f"cryoev-review:{pid}:sandbox:dirty"])
        self.errors = [e for e in self.errors if "Failed to load resource" not in e]

    def test_real_package_keeps_the_exact_key(self):
        """A draft saved under "cryoev-review:<id>" before this change is still picked up and uploaded."""
        f, pid, t = "fixB_old.png", self.srv.data["id"], int(time.time() * 1000)
        draft = {f: {"dec": {"g203": {"v": "keep:inner", "t": t, "who": "drafty"}}, "reviewed": False, "modified": True, "t": t}}
        init = (f"localStorage.setItem('cryoev-review:{pid}', {json.dumps(json.dumps(draft))});"
                f"localStorage.setItem('cryoev-review:{pid}:dirty', {json.dumps(json.dumps([f]))});")
        ctx = self.pages()
        ctx.add_init_script(init)
        real = self.page(ctx, sandbox=False)
        self.load(real, "&who=tester&mode=editor")
        self.synced(real)
        d = self.saved_dec(f)["g203"]
        self.assertEqual((d["v"], d["who"], d["t"]), ("keep:inner", "drafty", t))
        self.assertEqual(real.evaluate("Object.keys(localStorage).filter(k => k.includes(':sandbox'))"), [])


if __name__ == "__main__":
    unittest.main()
