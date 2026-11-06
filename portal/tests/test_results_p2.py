"""Round 6 / P2 on the Results pages: what is counted, unsure counts, the Update button, the diameter histogram and
the analysis display name. Golden prediction sets copied into a temp predict root (see test_results_module.py).

  python -m unittest discover -s portal/tests -p test_results_p2.py -v
"""
import json
import os
import re
import shutil
import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_results_module as M  # noqa: E402
from annotation import prediction_package as pp  # noqa: E402
from portal.modules import results as R  # noqa: E402
from results.tests import test_p2 as TP  # noqa: E402
from results.compute import compute_and_write  # noqa: E402

FRESH = "ps_20261106_a00001"        # new fields; a review package exists (review_package.json), edits older
STALE = "ps_20261106_a00002"        # new fields; an edit saved after the results
OLD = "ps_20261106_a00003"          # results.json without the round-6 fields
NAMED = "ps_20261106_a00004"        # request.json name with markup
NOEV = "ps_20261106_a00005"         # no confident outline -> no EV in any sample
NULLSET = "ps_20261106_a00006"      # results.json whose S1 n_unsure_open is null (unknown)
REAL = "ps_20261106_b00001"         # built through prediction_package.build, one image edited, one Reviewed
XSS = "<b onmouseover=alert(1)>Name & co</b>"


class Tags(HTMLParser):
    """Records every start tag with its stack of open ancestors."""

    def __init__(self):
        super().__init__()
        self.stack, self.found = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in ("br", "img", "meta", "link", "input", "line", "rect", "hr"):
            self.stack.append(tag)
        self.found.append((tag, dict(attrs), tuple(self.stack)))

    def handle_endtag(self, tag):
        if tag in self.stack:
            while self.stack and self.stack.pop() != tag:
                pass


def tags(page):
    p = Tags()
    p.feed(page)
    return p.found


class P2(M.Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        for ps in (FRESH, STALE, OLD, NAMED, NOEV, NULLSET):
            M.make_set(cls.root, ps, owner=M.OWNER)
        for ps in (FRESH, STALE, OLD, NAMED):
            compute_and_write(cls.root / ps)
        # a set whose confident outlines are all removed: every sample has 0 EVs
        for f in (cls.root / NOEV / "p").rglob("*.txt"):
            f.write_text("", encoding="utf-8")
        compute_and_write(cls.root / NOEV)
        for ps in (FRESH, STALE):
            pkg = cls.review / f"pred_{ps}"
            (pkg / "edits").mkdir(parents=True)
            (pkg / "review_package.json").write_text("{}", encoding="utf-8")
        done = (cls.root / STALE / "results" / "results.json").stat().st_mtime
        for ps in (FRESH, STALE):
            os.utime(cls.review / f"pred_{ps}" / "review_package.json", (done - 600, done - 600))
        edit = cls.review / f"pred_{STALE}" / "edits" / "0001.png.json"
        edit.write_text("{}", encoding="utf-8")
        os.utime(edit, (done + 120, done + 120))
        rj = cls.root / OLD / "results" / "results.json"          # as written before round 6
        doc = json.loads(rj.read_text(encoding="utf-8"))
        for s in doc["samples"].values():
            for k in ("n_unsure_open", "n_checked", "diam_hist"):
                s.pop(k)
        rj.write_text(json.dumps(doc), encoding="utf-8")
        # the real package path: a synthetic set -> prediction_package.build -> one saved edit -> the portal computes
        ps = TP.make_ps(cls.root.parent, {"0001": TP.A_OBJS, "0002": TP.B_OBJS, "0003": TP.C_OBJS}, ps_id=REAL)
        pkg = pp.build(ps, cls.review)
        TP.save_edit(pkg, "0002.png", {"dec": {"p0": {"v": "add:EV", "t": 5, "who": "rev1", "pending": True}}})
        TP.save_edit(pkg, "0003.png", {"reviewed": True, "dec": {}})
        # a computed set whose S1 sample has an unknown (null) unsure count
        compute_and_write(cls.root / NULLSET)
        nj = cls.root / NULLSET / "results" / "results.json"
        doc = json.loads(nj.read_text(encoding="utf-8"))
        doc["samples"]["S1"]["n_unsure_open"] = None
        nj.write_text(json.dumps(doc), encoding="utf-8")
        req = cls.root / NAMED / "request.json"
        doc = json.loads(req.read_text(encoding="utf-8"))
        doc["name"] = XSS
        req.write_text(json.dumps(doc), encoding="utf-8")

    def results(self, ps):
        return json.loads((self.root / ps / "results" / "results.json").read_text(encoding="utf-8"))


class CountedAndUnsure(P2):
    def test_header_line_and_columns(self):
        page = self.page(FRESH, "owner")
        self.assertIn("<b>Counted:</b> EV outlines the model is sure about (≥50%) plus everything kept or drawn in "
                      "review. Unsure outlines are not counted until you accept them.", page)
        res = self.results(FRESH)
        for label, s in res["samples"].items():
            c = M.cells(page, label)
            self.assertEqual(c["ev_total"][0], str(s["ev_total"]))
            self.assertEqual(c["n_unsure_open"][0], str(s["n_unsure_open"]))
            self.assertEqual(c["n_checked"][0], f"{s['n_checked']} of {s['n_images']}")
        self.assertIn("Unsure, not counted", page)

    def test_nudge_with_total_and_review_link(self):
        res = self.results(FRESH)
        total = sum(s["n_unsure_open"] for s in res["samples"].values())
        self.assertGreater(total, 0)                      # the golden set has 0.25-0.50 candidates, nobody decided
        page = self.page(FRESH, "owner")
        m = re.search(r'id="cj-results-nudge".*?</div>', page, re.S)
        self.assertIsNotNone(m)
        self.assertIn(f"{total} unsure outline{'s' if total != 1 else ''} not checked yet.", m.group(0))
        self.assertNotIn("Review these images", m.group(0))     # the honesty banner already has that button
        self.assertEqual(page.count("Review these images"), 1)
        self.assertEqual(page.count(f'href="/review/pred_{FRESH}/"'), 1)

    def test_nudge_without_package_has_no_dead_link(self):
        page = self.page(M.NOEDIT, "owner")               # no review package for this set
        m = re.search(r'id="cj-results-nudge".*?</div>', page, re.S)
        self.assertIsNotNone(m)
        self.assertNotIn("/review/", m.group(0))

    def test_no_nudge_when_nothing_is_open(self):
        page = self.page(NOEV, "owner")
        self.assertEqual(self.results(NOEV)["samples"]["S1"]["n_unsure_open"], 0)
        self.assertNotIn('id="cj-results-nudge"', page)

    def test_old_results_say_update_to_see(self):
        page = self.page(OLD, "owner")
        self.assertIn("Update to see unsure counts.", page)
        self.assertIn('id="cj-results-oldfields"', page)
        self.assertIn(f'<form class="cj-results-form" method="post" action="/results/{OLD}/compute">', page)
        self.assertNotIn('id="cj-results-nudge"', page)
        c = M.cells(page, "S1")
        self.assertEqual((c["n_unsure_open"][0], c["n_unsure_open"][1]), ("–", "Update to see unsure counts"))
        self.assertNotIn("cj-results-fig cj-results-histfig", page)       # no diam_hist in an old file: no histogram, no crash


class UpdateButton(P2):
    def test_stale_banner_has_the_form_and_button(self):
        page = self.page(STALE, "owner")
        m = re.search(r'<div[^>]*id="cj-results-stale".*?</form></div>', page, re.S)
        self.assertIsNotNone(m, "stale banner")
        self.assertIn(f'<form class="cj-results-form" method="post" action="/results/{STALE}/compute">', m.group(0))
        self.assertIn("Update results now</button>", m.group(0))
        self.assertNotIn("<script", m.group(0))             # a plain form: no JS needed

    def test_update_post_is_origin_checked_and_starts_the_compute(self):
        calls = []
        self.patch_compute(lambda ps_dir, reviewed: (calls.append(Path(ps_dir).name),
                                                     compute_and_write(ps_dir, reviewed))[1])
        self.assertEqual(self.post(f"/results/{STALE}/compute", "owner", origin="https://evil.example")[0], 403)
        self.assertEqual(calls, [])
        self.assertEqual(self.post(f"/results/{STALE}/compute", "owner")[0], 303)
        self.wait_state(STALE, ("ready",))
        self.assertEqual(calls, [STALE])

    def test_fresh_results_have_no_update_banner(self):
        self.assertNotIn('id="cj-results-stale"', self.page(FRESH, "owner"))

    def test_index_card_has_no_form_inside_a_link(self):
        idx = self.get("/results/", "owner")[2].decode()
        found = tags(idx)
        forms = [(a, anc) for tag, a, anc in found if tag == "form"]
        self.assertTrue(forms, "the out-of-date card should have its Update form")
        for attrs, ancestors in forms:
            self.assertNotIn("a", ancestors, attrs)
        self.assertEqual([a["action"] for a, _ in forms if "action" in a].count(f"/results/{STALE}/compute"), 1)
        self.assertNotIn(f"/results/{FRESH}/compute", idx)  # current results: no Update button
        self.assertIn(">Update</button>", idx)
        for tag, attrs, anc in found:                       # no link wraps a button either
            if tag == "button":
                self.assertNotIn("a", anc)

    def test_index_card_still_links_and_shows_out_of_date(self):
        idx = self.get("/results/", "owner")[2].decode()
        card = re.search(r'<li class="cj-card cj-results-set">((?:(?!</li>).)*%s(?:(?!</li>).)*)</li>' % STALE,
                         idx, re.S).group(1)
        self.assertIn(f'href="/results/{STALE}/"', card)
        self.assertIn("Out of date", card)


class DisplayName(P2):
    def test_named_set_is_shown_escaped_with_the_id_as_small_text(self):
        page = self.page(NAMED, "owner")
        esc = "&lt;b onmouseover=alert(1)&gt;Name &amp; co&lt;/b&gt;"
        self.assertIn(f'<h1 class="cj-wrap">{esc}</h1>', page)
        self.assertNotIn(XSS, page)
        self.assertIn(f'<p class="cj-mut cj-results-psid">{NAMED}</p>', page)
        idx = self.get("/results/", "owner")[2].decode()
        self.assertIn(f'<b class="cj-wrap">{esc}</b>', idx)
        self.assertNotIn(XSS, idx)
        self.assertIn(NAMED, idx)

    def test_unnamed_set_uses_the_default_name(self):
        page = self.page(FRESH, "owner")
        self.assertRegex(page, r'<h1 class="cj-wrap">Session golden · 5 Nov 2026, \d\d:\d\d</h1>')


class HistogramPage(P2):
    def test_one_svg_per_sample_with_hover_titles_and_a_table(self):
        res = self.results(FRESH)
        page = self.page(FRESH, "owner")
        self.assertEqual(page.count('class="cj-card cj-results-fig cj-results-histfig"'),
                         sum(1 for s in res["samples"].values() if s["ev_total"] > 0))
        label, s = next((k, v) for k, v in res["samples"].items() if v["ev_total"] > 0)
        fig = re.search(r'<figure[^>]*data-sample="%s".*?</figure>' % re.escape(label), page, re.S).group(0)
        counts = s["diam_hist"]["counts"]
        titles = re.findall(r"<title>([^<]*)</title>", fig)
        xmax = R.shared_xmax(res["samples"])
        self.assertEqual(len(titles), xmax // 10 + 1)       # one per bin below xmax + the folded overflow bar
        k = next(i for i, c in enumerate(counts) if c)
        self.assertEqual(titles[k], f"{k * 10}–{k * 10 + 10} nm: {counts[k]} EV{'s' if counts[k] != 1 else ''}")
        self.assertEqual(titles[-1], f"{xmax} nm or more: {sum(counts[xmax // 10:])} "
                                     f"EV{'s' if sum(counts[xmax // 10:]) != 1 else ''}")
        self.assertIn(f"≥{xmax}", fig)                      # the overflow bar's axis label
        self.assertIn("<details", fig)
        self.assertEqual(len(re.findall(r"<tr><td class=\"cj-results-l\">", fig)), sum(1 for c in counts if c))
        self.assertNotIn("<script", fig)
        folded = counts[:xmax // 10] + [sum(counts[xmax // 10:])]
        self.assertEqual(len(re.findall(r'<rect class="cj-results-bar"', fig)), sum(1 for c in folded if c))

    def test_no_histogram_for_a_sample_without_evs(self):
        page = self.page(NOEV, "owner")
        self.assertNotIn("cj-results-fig cj-results-histfig", page)

    def test_page_has_no_inline_script(self):
        page = self.page(FRESH, "owner")
        self.assertEqual(re.findall(r"<script[^>]*>", page).count("<script>"), 0)

    def test_overflow_label_for_bins_above_500(self):
        h = {"bin_nm": 10, "max_nm": 500, "counts": [0] * 50 + [3]}
        out = R.hist_chart(0, "S", {"diam_hist": h})
        self.assertIn("500 nm or more: 3 EVs", out)
        self.assertIn("490–500 nm: 0 EVs", out)

    def test_malformed_hist_is_skipped(self):
        for bad in (None, {}, {"counts": "x"}, {"bin_nm": 10, "counts": [0, 0]}, {"bin_nm": 0, "counts": [1, 1]},
                    {"bin_nm": 10, "counts": [1, -1]}):
            self.assertFalse(R.hist_ok({"diam_hist": bad}), bad)


class RealPackagePath(P2):
    def test_portal_compute_with_a_real_package_gives_unsure_two(self):
        """prediction_package.build + saved edits -> reviewed_polygons -> compute (no fake): known answer 2 / 2 / 4
        (A untouched 1 + B edited 1 open + C Reviewed 0; B and C checked; 3 confident + B's accepted candidate)."""
        self.assertEqual(self.post(f"/results/{REAL}/compute", "admin")[0], 303)
        self.wait_state(REAL, ("ready",))
        page = self.page(REAL, "admin")
        c = M.cells(page, "S1")
        self.assertEqual((c["n_unsure_open"][0], c["n_checked"][0], c["ev_total"][0]), ("2", "2 of 3", "4"))
        self.assertIn("2 unsure outlines not checked yet.", page)
        self.assertEqual(page.count("Review these images"), 1)     # honesty banner only (B is not Reviewed)
        self.assertNotIn('id="cj-results-stale"', page)


class AxisAndNotes(P2):
    @staticmethod
    def hist(**bins):
        counts = [0] * 51
        for k, v in bins.items():
            counts[int(k[1:])] = v
        return {"diam_hist": {"bin_nm": 10, "max_nm": 500, "counts": counts}}

    def test_single_480_outlier_does_not_stretch_the_axis(self):
        samples = {"S1": self.hist(b4=150, b5=50, b48=1)}                    # 201 EVs, one at ~485 nm
        self.assertEqual(R.shared_xmax(samples), 100)                         # 99th percentile bin 4 -> 100 nm
        out = R.hist_charts(samples)
        self.assertEqual(len(re.findall(r"<title>[^<]*</title>", out)), 11)    # ten 10 nm bins + the overflow bar
        self.assertIn("100 nm or more: 1 EV", out)
        self.assertIn("≥100", out)
        self.assertNotIn("≥500", out)
        self.assertIn("<td class=\"cj-results-l\">480–490 nm</td><td>1</td>", out)                                # the table still has the original bin

    def test_axis_is_shared_between_samples_and_rounded_to_50(self):
        samples = {"S1": self.hist(b4=40, b12=2), "S2": self.hist(b30=100)}  # pooled p99 bin 30 -> 310 -> 350 nm
        self.assertEqual(R.shared_xmax(samples), 350)
        out = R.hist_charts(samples)
        self.assertEqual(out.count("≥350"), 2)                                # both samples: same axis
        self.assertEqual(R.shared_xmax({"S1": self.hist(b0=3)}), 100)         # minimum 100 nm

    def test_page_note_says_which_edge_a_bin_includes(self):
        page = self.page(FRESH, "owner")
        self.assertIn("A bin includes its lower edge and excludes its upper (10 ≤ d &lt; 20 nm).", page)

    def test_footer_wording(self):
        page = self.page(FRESH, "owner")
        self.assertIn(">Update results now</button>", page)
        self.assertNotIn("Compute again", page)

    def test_null_unsure_count_renders_as_unknown_and_is_left_out_of_the_nudge(self):
        page = self.page(NULLSET, "owner")
        c = M.cells(page, "S1")
        self.assertEqual(c["n_unsure_open"][0], "–")
        self.assertIn("unknown", c["n_unsure_open"][1])
        res = self.results(NULLSET)
        rest = sum(s["n_unsure_open"] for k, s in res["samples"].items() if s["n_unsure_open"] is not None)
        if rest:
            self.assertIn(f"{rest} unsure outline", page)
        else:
            self.assertNotIn('id="cj-results-nudge"', page)

    def test_refused_sample_unsure_count_is_in_the_nudge_and_the_notice(self):
        res = self.results(M.REFUSED)                                        # S1 refused (nm/px), S2 counted
        ru = res["refused_unsure"]
        self.assertEqual(list(ru), ["S1"])
        self.assertGreater(ru["S1"], 0)
        page = self.page(M.REFUSED, "owner")
        total = ru["S1"] + sum(s["n_unsure_open"] for s in res["samples"].values())
        self.assertIn(f"{total} unsure outline", page)
        self.assertIn(f"{ru['S1']} unsure outline", re.search(r'id="cj-results-refused".*?</div>', page, re.S).group(0))

    def test_old_results_where_every_sample_lacks_the_fields(self):
        page = self.page(OLD, "owner")
        res = self.results(OLD)
        self.assertTrue(all("n_unsure_open" not in s for s in res["samples"].values()))
        for label in res["samples"]:
            c = M.cells(page, label)
            self.assertEqual(c["n_unsure_open"][1], "Update to see unsure counts")
            self.assertEqual(c["n_checked"][0], "–")
        self.assertNotIn('id="cj-results-nudge"', page)


if __name__ == "__main__":
    unittest.main()
