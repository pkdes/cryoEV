"""
The Reviews hub on a real portal (round 13, portal/modules/reviews.py): GET /reviews/, /reviews/go/<id> and the Assign
reviewers panel + POST. Temp roster / state / review / predict folders only (cjtest.Env); never CryoAI-laptop.

  - The live-audit stand-in (a lab_compare_* core package with a _work_ folder and decoy counts in its meta) leaves
    no trace for basic users: their hub, go and reviewers responses never hold its id, owner, counts, "cards",
    "decoys" or a path. The admin sees its row with counts only.
  - go: one identical 404 for unknown / not yours / unserved / core-for-non-admin; Quick check vs Editor.
  - Reviewers POST controls: form token, Origin, non-owner, roster ids (unknown, revoked), the audit-owner refusal,
    the core package refused; success writes only `reviewers` and one audit line.

  python -m unittest portal.tests.test_reviews_hub -v
"""
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
sys.path.insert(0, str(T.REPO / "annotation" / "tests"))
import make_fixture  # noqa: E402
from annotation import multi_review_server as MRS  # noqa: E402
from portal import audit as PA  # noqa: E402
from portal import auth as A  # noqa: E402
from portal import listing  # noqa: E402
from portal.modules import reviews as REV  # noqa: E402

FORM = "application/x-www-form-urlencoded"
TOKEN_RE = re.compile(r'name="t" value="([0-9a-f]{40})"')
AUDIT = "lab_compare_fixture3_20261111"   # core package (cfg.review_package): the live audit's stand-in
AUDIT2 = "lab_compare_served_20261110"    # a served blind audit owned by audowner
PA_ = "pred_ps_20261112_aaaaa1"           # bas's own
PB = "pred_ps_20261112_bbbbb2"            # other's, bas reviewer
PC = "pred_ps_20261112_ccccc3"            # other's, no reviewers
DECOYS = 7777


def fake_listing(stack: ExitStack):
    """B2's portal/listing.py may still be the contract stub on this branch: give it minimal bodies for the test,
    only when the real ones raise NotImplementedError (after the merge the real helper is tested)."""
    try:
        listing.parse_query({}, default_status="open").to_query()
        listing.controls_html(listing.ListQuery(), action="/x/", people=[], models=[], statuses=[])
        return
    except NotImplementedError:
        pass

    class Q(listing.ListQuery):
        def to_query(self):
            return urlencode([("status", self.status)] + ([("group", self.group)] if self.group != "none" else []))

    def parse_query(qp, *, default_status="any"):
        return Q(status=qp.get("status") or default_status, group=qp.get("group") or "none",
                 person=qp.get("person") or "any")

    def apply(rows, q, *, key=lambda r: r["facets"]):
        out = [r for r in rows if q.status == "any" or (q.status == "open" and r["status"] not in ("finished", "applied"))
               or r["status"] == q.status]
        return [r for r in out if q.person == "any" or key(r).get("person") == q.person]

    def group(rows, by, *, key=lambda r: r["facets"]):
        out = {}
        for r in rows:
            out.setdefault(str(key(r).get(by) or "none"), []).append(r)
        return list(out.items())

    def controls_html(q, *, action, people, models, statuses, show_group=True):
        return (f'<form id="cj-list-form" method="get" action="{action}"><select name="status">'
                + "".join(f'<option value="{v}">{t}</option>' for v, t in statuses) + "</select></form>")

    for name, fn in (("parse_query", parse_query), ("apply", apply), ("group", group),
                     ("controls_html", controls_html), ("facets", lambda *a, **k: {})):
        stack.enter_context(mock.patch.object(listing, name, fn))


def build(root: Path, name: str, meta: dict, flag_all=False) -> Path:
    pkg = root / name
    make_fixture.build(pkg)
    if flag_all:                                  # a distinctive card total for the audit (never shown to basic users)
        p = pkg / "data.js"
        d = json.loads(p.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
        for im in d["images"]:
            for g in im.get("gt") or []:
                g["u"] = 1
        p.write_text("window.REVIEW_DATA = " + json.dumps(d) + ";", encoding="utf-8")
    (pkg / "review_package.json").write_text(json.dumps(meta), encoding="utf-8")
    (pkg / ".review_key").write_text("k" * 20, encoding="utf-8")
    return pkg


class Hub(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = ExitStack()
        cls.stack.enter_context(T.TK.env_patch(drop=("CJ_*",)))
        fake_listing(cls.stack)
        cls.env = T.Env()
        cls.tmp = Path(tempfile.mkdtemp(prefix="cj_rhub_"))
        cls.review, cls.predict = cls.tmp / "review", cls.tmp / "predict"
        cls.review.mkdir()
        cls.predict.mkdir()
        cls.audit = build(cls.review, AUDIT, {
            "kind": "compare", "owner": "audowner", "reviewers": ["audrev"], "n_images": 4,
            "decoys": {"n": DECOYS, "frac": 0.1}, "base": "C:\\Users\\someone\\CryoAI-laptop\\labels\\lv_20261030_0909",
            "compare": {"a": {"coco": "C:\\Users\\someone\\x\\lv_20261030_0909\\labels.coco.json"},
                        "b": {"weights": "models\\secret_model_x\\best.pt"}}}, flag_all=True)
        (cls.review / f"_work_{AUDIT}").mkdir()
        (cls.review / f"_work_{AUDIT}" / "decoys.json").write_text(json.dumps({"n": DECOYS}), encoding="utf-8")
        cls.audit2 = build(cls.review, AUDIT2, {"kind": "compare", "owner": "audowner", "n_images": 4,
                                                "decoys": {"n": DECOYS}})
        cls.pa = build(cls.review, PA_, {"kind": "prediction", "owner": "bas", "model": "v5", "n_images": 4})
        cls.pb = build(cls.review, PB, {"kind": "prediction", "owner": "other", "reviewers": ["bas"], "model": "v5",
                                        "n_images": 4, "owner_via": "claude", "keep_me": {"x": [1, 2]}})
        cls.pc = build(cls.review, PC, {"kind": "prediction", "owner": "other", "model": "v5", "n_images": 4})
        cls.svc = cls.tmp / "services.json"
        cls.svc.write_text(json.dumps({"services": [{"name": "review", "argv": [
            "py", "annotation/review_server.py", "--package", str(cls.audit), "--port", "8766"]}]}), encoding="utf-8")
        os.environ["CRYOEV_SERVICES_CONFIG"] = str(cls.svc)
        os.environ[MRS.PREFIX_ENV] = "pred_,lab_"
        cls.keys = {"admin": cls.env.key}
        for pid, role in (("bas", "basic"), ("other", "basic"), ("nobody", "basic"), ("audowner", "basic"),
                          ("gone", "basic"), ("adv", "advanced")):
            cls.keys[pid] = A.add_key(cls.env.roster, pid, role)
        cfg = cls.env.config(review_package=cls.audit)
        cfg.review_root, cfg.predict_root = cls.review, cls.predict
        cls.cfg = cfg
        cls.p = T.Portal(cfg)
        cls.tok = {k: T.login(cls.p.port, v) for k, v in cls.keys.items()}
        A.revoke(cls.env.roster, "gone")
        cls.audit_total = REV.card_progress(cls.audit)[1]
        assert cls.audit_total not in {REV.card_progress(p)[1] for p in (cls.pa, cls.pb, cls.pc, cls.audit2)}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        cls.stack.close()

    def setUp(self):
        self.saved = {p: (p / "review_package.json").read_bytes() for p in (self.pa, self.pb, self.pc, self.audit2)}

    def tearDown(self):
        for p, b in self.saved.items():
            (p / "review_package.json").write_bytes(b)

    def get(self, path, who):
        st, h, body = T.request(self.p.port, "GET", path, T.cookie(self.tok[who]))
        return st, h, body.decode()

    def post(self, path, who, fields, origin=True, token=True):
        st, _, page = self.get(path, who)
        tok = TOKEN_RE.search(page).group(1) if token is True and st == 200 else (token or "")
        hdr = {**T.cookie(self.tok[who]), "Content-Type": FORM}
        if origin:
            hdr["Origin"] = self.p.origin if origin is True else origin
        body = urlencode([("t", tok)] + list(fields)).encode()
        st, h, out = T.request(self.p.port, "POST", path, hdr, body)
        return st, h, out.decode()

    def meta(self, pkg):
        return json.loads((pkg / "review_package.json").read_text(encoding="utf-8"))

    def main(self, html):
        return html.split("<main", 1)[-1]

    # ------------------------------------------------------------ the blind audit leaves no trace
    def assert_no_audit(self, text, who, strict=False):
        for leak in (AUDIT, "fixture3", "audowner", "audrev", str(DECOYS), "decoys", "_work_", "secret_model_x",
                     "Users", "CryoAI-laptop", f"of {self.audit_total} cards", str(self.review), "Blind audit",
                     "hidden (blind)"):
            self.assertNotIn(leak, text, f"{who}: {leak}")
        if strict:
            for leak in ("cards", "Audits", "Practice", "sandbox", ":8765", ":8443"):
                self.assertNotIn(leak, self.main(text), f"{who}: {leak}")

    def test_basic_responses_never_hold_the_audit(self):
        for who in ("bas", "nobody", "other"):
            for path in ("/reviews/", "/reviews/?show=all", "/reviews/?kind=audit", "/reviews/?status=any&group=person",
                         "/reviews/?more=1", f"/reviews/go/{AUDIT}", f"/reviews/{AUDIT}/reviewers",
                         f"/reviews/go/{AUDIT2}", "/reviews/go/sandbox"):
                st, h, text = self.get(path, who)
                self.assert_no_audit(text + json.dumps(h), f"{who} {path}", strict=who == "nobody")
            st, _, text = self.post(f"/reviews/{AUDIT}/reviewers", who, [("add", who)], token="0" * 40)
            self.assertEqual(st, 404)
            self.assert_no_audit(text, who, strict=True)
        _, _, nobody = self.get("/reviews/", "nobody")
        self.assertIn("No reviews yet. Start one from a finished prediction set, or ask an admin.", nobody)
        self.assertIn('href="/reviews/start/"', nobody)

    def test_admin_sees_the_audit_as_counts_only(self):
        st, _, text = self.get("/reviews/?show=all&status=any", "admin")
        self.assertEqual(st, 200)
        row = re.search(r'<div class="cj-rv-row">((?:(?!<div class="cj-rv-row">).)*?Open /review/.*?)</div></div>',
                        text, re.S)
        self.assertIsNotNone(row)
        row = row.group(1)
        self.assertIn('class="cj-rv-tag">blind<', row)
        self.assertEqual(row.count("hidden (blind)"), 2)                   # owner and reviewers
        self.assertIn(f"of {self.audit_total} cards", row)
        self.assertIn(f'href="/reviews/go/{AUDIT}"', row)
        self.assertNotIn("Assign", row)
        for leak in ("audowner", "audrev", str(DECOYS), "decoys", "secret_model_x", "Users", "_work_"):
            self.assertNotIn(leak, row)
        self.assertNotIn(str(DECOYS), text)
        self.assertNotIn("audrev", text)
        self.assertIn("lab network only", text)                            # the practice row
        st, h, _ = self.get(f"/reviews/go/{AUDIT}", "admin")
        self.assertEqual((st, T.hget(h, "location")), (303, "/review/?mode=cards"))

    def test_hub_rows_tabs_and_needs(self):
        st, _, text = self.get("/reviews/", "bas")
        self.assertEqual(st, 200)
        self.assertIn("<title>Reviews</title>", text)
        self.assertIn("Needs your review", text)
        self.assertIn(f'href="/reviews/go/{PA_}"', text)
        self.assertIn(f'href="/reviews/go/{PB}"', text)
        self.assertIn("claude for other", text)                            # the agent chip on the owner
        self.assertIn(f'href="/reviews/{PA_}/reviewers">Assign', text)     # own: assign
        self.assertNotIn(f'/reviews/{PB}/reviewers', text)                 # reviewer only: no assign
        self.assertNotIn(PC, text)
        self.assertNotIn("Mine</a>", text)                                 # Mine | All is admin only
        self.assertIn('>Label sets <', text)
        self.assertNotIn('>Audits <', text)
        st, _, adm = self.get("/reviews/?show=all", "admin")
        self.assertIn("Mine</a>", adm)
        self.assertIn('>Audits <', adm)
        self.assertIn(PC, adm)
        _, _, only = self.get("/reviews/?show=all&kind=label_review", "admin")
        self.assertNotIn(f"/reviews/go/{PC}", self.main(only).split("All reviews", 1)[1])

    def test_needs_more_link(self):
        rows = [{"id": f"pred_ps_20261112_{i:06x}", "kind": "prediction", "title": f"T{i}", "made_from": "", "owner": "bas",
                 "reviewers": [], "role": "yours", "status": "not_started", "cards": 3, "decided": 0, "n_images": 1,
                 "last": 100 + i, "open_url": "/review/x/", "go_url": f"/reviews/go/x{i}", "mode": "cards",
                 "needs_you": True, "can_assign": True, "blind": False, "via": None, "facets": {}} for i in range(6)]
        page = REV.render_hub(rows, user={"id": "bas", "role": "basic"}, params={})
        self.assertEqual(page.count('class="cj-rv-card"'), 4)
        self.assertIn("Show 2 more", page)
        self.assertIn("more=1", page)
        page = REV.render_hub(rows, user={"id": "bas", "role": "basic"}, params={"more": "1"})
        self.assertEqual(page.count('class="cj-rv-card"'), 6)
        self.assertIn("Show fewer", page)
        self.assertLess(page.index("T5"), page.index("T0"))                # most recent first

    # ------------------------------------------------------------ go
    def test_go_modes_and_one_404(self):
        st, h, _ = self.get(f"/reviews/go/{PA_}", "bas")
        self.assertEqual((st, T.hget(h, "location")), (303, f"/review/{PA_}/?mode=cards"))
        for file, keys in REV.package_cards(json.loads((self.pb / "data.js").read_text(encoding="utf-8")
                                                       .split("=", 1)[1].strip().rstrip(";"))).items():
            (self.pb / "edits").mkdir(exist_ok=True)
            (self.pb / "edits" / f"{file}.json").write_text(json.dumps(
                {"file": file, "edit": {"dec": {k: {"v": "keep", "t": 1} for k in keys}}}), encoding="utf-8")
        os.utime(self.pb / "edits", (time.time() + 9, time.time() + 9))
        try:
            st, h, _ = self.get(f"/reviews/go/{PB}", "bas")
            self.assertEqual((st, T.hget(h, "location")), (303, f"/review/{PB}/?mode=editor"))
        finally:
            shutil.rmtree(self.pb / "edits")
        bodies = set()
        for path in (f"/reviews/go/{PC}", f"/reviews/go/{AUDIT}", f"/reviews/go/{AUDIT2}", "/reviews/go/nope",
                     "/reviews/go/pred_ps_20261112_zzzzzz", "/reviews/go/prelipo_x", "/reviews/go/sandbox"):
            st, _, text = self.get(path, "bas")
            self.assertEqual(st, 404, path)
            bodies.add(self.main(text))
        self.assertEqual(len(bodies), 1)                                   # one identical answer

    # ------------------------------------------------------------ reviewers
    def test_panel_and_success(self):
        st, _, page = self.get(f"/reviews/{PA_}/reviewers", "bas")
        self.assertEqual(st, 200)
        self.assertIn("Reviewers for", page)
        self.assertIn('<option value="other">', page)
        self.assertNotIn('value="gone"', page)                             # revoked: not offered
        st, h, _ = self.post(f"/reviews/{PA_}/reviewers", "bas", [("add", "other"), ("add", "adv")])
        self.assertEqual(st, 303)
        self.assertTrue(T.hget(h, "location").startswith("/reviews/?f="))
        _, _, hub = self.get(T.hget(h, "location"), "bas")
        self.assertIn("added other, adv", hub)
        meta = self.meta(self.pa)
        self.assertEqual(meta["reviewers"], ["other", "adv"])
        self.assertEqual(meta["model"], "v5")
        lines = [r for r in T.module_lines(PA.audit_path(self.env.state)) if r.get("action") == "reviewers changed"]
        self.assertEqual(lines[-1]["target"], PA_)
        self.assertEqual(lines[-1]["detail"], {"added": ["other", "adv"], "removed": []})
        self.assertEqual(lines[-1]["actor"], "bas")
        st, _, _ = self.post(f"/reviews/{PA_}/reviewers", "bas", [("remove", "adv")])
        self.assertEqual(st, 303)
        self.assertEqual(self.meta(self.pa)["reviewers"], ["other"])
        _, _, theirs = self.get("/reviews/", "other")                     # the new reviewer sees it under Needs you
        self.assertIn(f'href="/reviews/go/{PA_}"', theirs.split("All reviews")[0])
        st, _, _ = self.post(f"/reviews/{PB}/reviewers", "admin", [("add", "nobody")])   # admin on anyone's
        self.assertEqual(st, 303)
        self.assertEqual(self.meta(self.pb)["keep_me"], {"x": [1, 2]})     # every other key kept

    def test_controls(self):
        path = f"/reviews/{PA_}/reviewers"
        unchanged = (self.pa / "review_package.json").read_bytes()
        cases = [
            ("no token", dict(token=None), 403),
            ("wrong token", dict(token="0" * 40), 403),
            ("bad Origin", dict(origin="http://evil.example"), 403),
            ("no Origin", dict(origin=False), 403),
        ]
        for name, kw, want in cases:
            st, _, _ = self.post(path, "bas", [("add", "other")], **kw)
            self.assertEqual(st, want, name)
        for ids, name in ((["ghost"], "unknown id"), (["gone"], "revoked id")):
            st, _, text = self.post(path, "bas", [("add", i) for i in ids])
            self.assertEqual(st, 400, name)
            self.assertIn("roster", text)
        self.assertEqual((self.pa / "review_package.json").read_bytes(), unchanged)
        # a reviewer is not an owner: the same 404 as an unknown id, nothing written
        before = (self.pb / "review_package.json").read_bytes()
        for who, p in (("bas", PB), ("other", PA_), ("nobody", PA_)):
            st, _, _ = self.get(f"/reviews/{p}/reviewers", who)
            self.assertEqual(st, 404, (who, p))
            st, _, _ = self.post(f"/reviews/{p}/reviewers", who, [("add", "nobody")], token="0" * 40)
            self.assertEqual(st, 404, (who, p))
        self.assertEqual((self.pb / "review_package.json").read_bytes(), before)
        # the audit's owner can't be a reviewer on it (refused for the admin too)
        st, _, text = self.post(f"/reviews/{AUDIT2}/reviewers", "admin", [("add", "audowner")])
        self.assertEqual(st, 400)
        self.assertIn("audit&#x27;s owner", text)
        self.assertNotIn("reviewers", self.meta(self.audit2))
        # the core package is never writable here, not even for the admin
        core_before = (self.audit / "review_package.json").read_bytes()
        self.assertEqual(self.get(f"/reviews/{AUDIT}/reviewers", "admin")[0], 404)
        st, _, _ = self.post(f"/reviews/{AUDIT}/reviewers", "admin", [("add", "nobody")], token="0" * 40)
        self.assertEqual(st, 404)
        self.assertEqual((self.audit / "review_package.json").read_bytes(), core_before)
        self.assertFalse((self.review / ".meta_locks" / f"{AUDIT}.lock").exists())
        # over the cap
        st, _, text = self.post(path, "admin", [("add", p) for p in ("bas", "other", "nobody", "audowner", "adv")])
        self.assertEqual(st, 303)
        with mock.patch.object(REV, "MAX_REVIEWERS", 5):
            st, _, text = self.post(path, "admin", [("add", "tester")])
        self.assertEqual(st, 400)
        self.assertIn("at most 5", text)


if __name__ == "__main__":
    unittest.main()
