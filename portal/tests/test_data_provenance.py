"""Data page, round 11b: a prediction set's note and provenance (jobs/predict_set.py create --note / --provenance).

Known answers on one fixture world (test_data_module's, plus two sets): the set with a note and a provenance shows
"Note", "How the images were made" (its steps, then one tiling line) and "Derived from"; each derived_from entry is a
link only when that record exists (a model and a prediction set do; a dataset id and an "other" entry do not), and a
private set the viewer may not see is never quoted. Control: the set without them shows none of the three labels.
"""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_data_module as TM  # noqa: E402
from test_data_module import D, main_html  # noqa: E402

WITH, WITHOUT = "ps_20261113_aaa111", "ps_20261113_bbb222"
NOTE = "Native tiles <b>no</b> downsampling & overlap\nsecond line"
PROV = {"derived_from": [{"kind": "model", "id": "v9_prod_20261101"},
                         {"kind": "prediction_set", "id": "ps_20261105_aaaaaa"},
                         {"kind": "prediction_set", "id": "ps_20261105_bbbbbb"},       # adv may not see it
                         {"kind": "dataset", "id": "ds_not_registered"},
                         {"kind": "other", "id": "Jul30 <full-res>"}],
        "source_images": "44 Jul30 full-resolution training images",
        "steps": ["cut 1440 px tiles at native resolution", "8-bit PNG <unchanged>"],
        "tiling": {"tile_px": 1440, "overlap_px": 288, "overlap_frac": 0.2, "nm_per_px": 0.368, "downsample": 1},
        "script": "tools/cut_native_tiles.py", "git": "e69366f"}
TILING = "tiles 1440 px, overlap 288 px (20%), 0.368 nm/px, no downsampling"


def records():
    out = ORIGINAL()
    base = {"schema": "prediction_set/1", "logged": "2026-11-13T10:00:00", "created": "2026-11-13T10:00:00",
            "owner": "adv", "model": "v9_prod_20261101", "path": "p", "source": {"kind": "upload"}, "n_images": 0,
            "status": "queued", "conf": 0.25}
    out["prediction_set"] = out["prediction_set"] + [dict(base, id=WITH, note=NOTE, provenance=PROV),
                                                     dict(base, id=WITHOUT)]
    return out


ORIGINAL = TM.fixture_records


class Provenance(TM.Base):
    @classmethod
    def setUpClass(cls):
        with mock.patch.object(TM, "fixture_records", records):
            super().setUpClass()

    def fields(self, ps, role="admin"):
        return {f[0]: f[1:] for f in self.api(f"/data/api/images/{ps}", role)["fields"]}

    def test_three_fields_shown(self):
        f = self.fields(WITH)
        self.assertEqual(f["Note"], [NOTE, None])
        self.assertEqual(f["How the images were made"], [PROV["steps"] + [TILING], None])
        self.assertEqual(f["Derived from"][0], [
            ["v9_prod_20261101 (model)", "/data/model/v9_prod_20261101"],
            ["ps_20261105_aaaaaa (prediction set)", "/data/images/ps_20261105_aaaaaa"],
            ["ps_20261105_bbbbbb (prediction set)", "/data/images/ps_20261105_bbbbbb"],        # admin may see it
            ["ds_not_registered (dataset)", None], ["Jul30 <full-res> (other)", None]])
        labels = [x[0] for x in self.api(f"/data/api/images/{WITH}")["fields"]]
        self.assertEqual(labels[labels.index("Made by") + 1:labels.index("Made by") + 4],
                         ["Note", "How the images were made", "Derived from"])    # next to "Made by"
        page = main_html(self.page(f"/data/images/{WITH}"))
        self.assertIn(D.esc(NOTE), page)
        self.assertNotIn("<b>no</b>", page)                                         # escaped
        self.assertIn(f"<li>{D.esc('8-bit PNG <unchanged>')}</li>", page)
        self.assertIn(f"<li>{TILING}</li>", page)
        self.assertIn('<li><a href="/data/model/v9_prod_20261101">v9_prod_20261101 (model)</a></li>', page)
        self.assertIn('<li>ds_not_registered (dataset)</li>', page)                 # absent: plain text
        self.assertIn(f"<li>{D.esc('Jul30 <full-res> (other)')}</li>", page)
        self.assertEqual(page.count('href="/data/dataset/ds_not_registered"'), 0)

    def test_private_set_never_quoted(self):
        f = self.fields(WITH, "advanced")
        self.assertIn(["Not visible to you", None], f["Derived from"][0])
        self.assertIn(["ps_20261105_aaaaaa (prediction set)", "/data/images/ps_20261105_aaaaaa"], f["Derived from"][0])
        body = self.get(f"/data/images/{WITH}", "advanced")[2].decode()
        self.assertNotIn("ps_20261105_bbbbbb", body)
        self.assertNotIn("ps_20261105_bbbbbb", json.dumps(self.api(f"/data/api/images/{WITH}", "advanced")))

    def test_absent_fields_show_nothing(self):
        f = self.fields(WITHOUT)
        self.assertIn("Made by", f)                                                  # control: the set renders
        for label in ("Note", "How the images were made", "Derived from"):
            self.assertNotIn(label, f)
            self.assertNotIn(f"<dt>{label}</dt>", main_html(self.page(f"/data/images/{WITHOUT}")))

    def test_tiling_text(self):
        from portal.modules.data import details as DT
        self.assertEqual(DT.tiling_text(PROV["tiling"]), TILING)
        self.assertEqual(DT.tiling_text({"overlap_frac": 0.25, "downsample": 2}), "overlap 25%, downsampled 2x")
        self.assertIsNone(DT.tiling_text({}))
        self.assertIsNone(DT.tiling_text("junk"))
        self.assertEqual(DT.made_how({"tiling": {"tile_px": 640}}), ["tiles 640 px"])
        self.assertEqual(DT.made_how(None), [])


if __name__ == "__main__":
    unittest.main()
