"""Known answer: results.compute on a real session equals the delivered summarize_detections numbers.

A session counted by the processing repo's run_counts.py has its tiles (`5. Model + labeling/model_input`),
the cached predictions, model_input_MANIFEST.csv and the delivered `figures/detections_summary.csv`. This
builds a prediction_set view of it in a temp folder (tiles and prediction files COPIED, server-style <nnnn>
names, sample_label = grid, nm/px from the manifest), runs compute() with no edits, and checks per grid:
images, EV total, EVs per image, median / Q1 / Q3 diameter, % multilayer and inner layers excluded, at the
delivered CSV's own precision. The session folder is only read.

Skipped when the processing datasets folder is not found (data_utils.config.processing_root()).

    python -m unittest discover -s results/tests -p test_known_answer.py -v
    RESULTS_KNOWN_ANSWER=20260827-2,20261027 python -m unittest discover -s results/tests -p test_known_answer.py
"""

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import config  # noqa: E402
from results import compute  # noqa: E402
from test_compute import diameter_pairs  # noqa: E402

SESSIONS = [s.strip() for s in os.environ.get("RESULTS_KNOWN_ANSWER", "20260827-2").split(",") if s.strip()]
COUNTS = "5. Model + labeling"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_view(session_dir: Path, dst: Path) -> tuple[Path, list[dict]]:
    """A prediction_set folder (contract section 2) over the session's counted tiles; returns (ps_dir, delivered)."""
    base = session_dir / COUNTS
    model = json.loads((base / "run_counts.json").read_text(encoding="utf-8"))["model"]
    rows = read_csv(base / "model_input_MANIFEST.csv")
    ps = dst / f"ps_known_{session_dir.name}"
    pred = ps / "p" / model / "predictions"
    (ps / "images").mkdir(parents=True)
    pred.mkdir(parents=True)
    lines = ["image,file,orig_name,sample_label,nm_per_px,nm_confirmed_by,source"]
    for n, r in enumerate(rows, 1):
        image, src = f"{n:04d}", base / "model_input" / r["output_name"]
        shutil.copyfile(src, ps / "images" / f"{image}{src.suffix}")
        shutil.copyfile(base / "predictions" / model / "predictions" / f"{src.stem}.txt", pred / f"{image}.txt")
        lines.append(f"{image},images/{image}{src.suffix},{r['output_name']},grid{r['grid']},{r['nm_per_px']},"
                     f"MANIFEST,session:{session_dir.name}")
    (ps / "images.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ps / "request.json").write_text(json.dumps({
        "ps": ps.name, "owner": "test", "model": model, "conf": 0.25, "max_det": 1000, "imgsz": None,
        "source": {"kind": "session", "session": session_dir.name, "dir": "<known-answer view>"},
        "created": "2026-11-05T00:00:00"}), encoding="utf-8")
    (ps / "status.json").write_text(json.dumps({"state": "done", "n_done": len(rows), "n_total": len(rows)}),
                                    encoding="utf-8")
    return ps, read_csv(base / "figures" / "detections_summary.csv")


# (label, delivered column, results field, format of the delivered column)
COMPARE = [("images", "fields", "n_images", "%d"), ("EVs", "ev_total", "ev_total", "%d"),
           ("EV/image", "ev_mean_per_field", "ev_per_image_mean", "%.2f"),
           ("median nm", "diam_median_nm", "diam_nm_median", "%.1f"), ("Q1 nm", "diam_q1_nm", "diam_nm_q1", "%.1f"),
           ("Q3 nm", "diam_q3_nm", "diam_nm_q3", "%.1f"), ("% multilayer", "pct_multilayer", "pct_multilayer_geometry",
                                                           "%.1f"),
           ("inner excl.", "inner_layers_excluded", "inner_layers_excluded", "%d")]


class KnownAnswer(unittest.TestCase):
    def test_sessions_match_delivered_summary(self):
        root = config.processing_root()
        if root is None:
            self.skipTest("processing datasets folder not found (set CRYOEV_PROCESSING_ROOT)")
        for session in SESSIONS:
            session_dir = root / session
            with self.subTest(session=session):
                if not (session_dir / COUNTS / "figures" / "detections_summary.csv").is_file():
                    self.skipTest(f"{session}: no delivered detections_summary.csv")   # this session only
                self._session(session_dir)

    def _session(self, session_dir):
        session = session_dir.name
        with tempfile.TemporaryDirectory() as tmp:
            ps, delivered = build_view(session_dir, Path(tmp))
            n_obj = 0
            for image, theirs, mine in diameter_pairs(ps):   # results' diameter == analyze_instances'
                self.assertEqual(mine, theirs, image)
                n_obj += len(mine)
            print(f"\n{session}: diameters of {n_obj} EV-class outlines identical to "
                  f"analysis.morphology.analyze_instances")
            res = compute(ps)
            self.assertIsNone(res["refused"])
            print(f"\n{session} ({res['model']}): delivered summarize_detections / results.compute")
            print("  grid    " + "  ".join(f"{c[0]:>17}" for c in COMPARE))
            bad = []
            for row in delivered:
                s = res["samples"][f"grid{row['grid']}"]
                cells = []
                for label, col, field, fmt in COMPARE:
                    mine = "" if s[field] is None else fmt % s[field]
                    cells.append(f"{row[col]:>8}/{mine:<8}" + ("" if mine == row[col] else "!"))
                    if mine != row[col]:
                        bad.append((row["grid"], label, row[col], mine))
                print(f"  {row['grid']:<6}  " + "  ".join(f"{c:>17}" for c in cells))
            self.assertEqual(set(res["samples"]), {f"grid{r['grid']}" for r in delivered})
            self.assertEqual(bad, [], "differs from the delivered detections_summary.csv")


if __name__ == "__main__":
    unittest.main()
