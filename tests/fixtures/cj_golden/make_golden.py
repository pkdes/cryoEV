"""
Golden fixtures for the CJ basic loop (BUILD_CONTRACTS.md section 6). Re-run to regenerate; never edit outputs by hand.

    python tests/fixtures/cj_golden/make_golden.py

Scene (400 x 400 px tiles, 1.4724 nm/px, 32-vertex circles, predictions in predict_models' normalised format):
  0001 (sample S1): A r40 c0 .90 contains B r25 c0 .80 (multilayer + inner); C r30 c0 .95 standalone;
                    D r20 c0 .30 (below 0.5, ignored); E r15 c1 .70 (non-EV)
  0002 (sample S1): F r35 c0 .90; G r30 c0 .85; H r20 c1 .60 (non-EV); I r45 c1 .70 encloses J r28 c0 .90
                    (class-0 filter BEFORE roles: J is a standalone EV, not an inner layer)
  0003 (sample S2): K r50 c0 .92 contains L r35 c0 .90 (multilayer + inner); N r30 c0 .91 standalone
Reviewed scenario (reviewed.json): image 0002 reviewed by "pujan": G deleted, F reclassed by the reviewer as MV.

Expected numbers are HAND-DERIVED below (not computed by the code under test). Diameters = area-equivalent
diameter of a 32-gon = 2r * 0.99679 * nm/px; compare with +-2 %. Quartiles: numpy default (linear).
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PS = "ps_20261105_abc123"
MODEL = "v5_evnonev_y8m_20261029"
NM = 1.4724
W = H = 400
N_VERT = 32
K32 = math.sqrt((N_VERT / (2 * math.pi)) * math.sin(2 * math.pi / N_VERT))   # 0.99679

#            id  image   cx   cy   r   cls conf  sample
OBJECTS = [("A", "0001", 120, 120, 40, 0, 0.90), ("B", "0001", 120, 120, 25, 0, 0.80),
           ("C", "0001", 300, 120, 30, 0, 0.95), ("D", "0001", 120, 300, 20, 0, 0.30),
           ("E", "0001", 300, 300, 15, 1, 0.70),
           ("F", "0002", 100, 100, 35, 0, 0.90), ("G", "0002", 300, 100, 30, 0, 0.85),
           ("H", "0002", 100, 300, 20, 1, 0.60), ("I", "0002", 290, 290, 45, 1, 0.70),
           ("J", "0002", 290, 290, 28, 0, 0.90),
           ("K", "0003", 140, 140, 50, 0, 0.92), ("L", "0003", 140, 140, 35, 0, 0.90),
           ("N", "0003", 300, 300, 30, 0, 0.91)]
SAMPLE = {"0001": "S1", "0002": "S1", "0003": "S2"}


def poly(cx, cy, r):
    return [[round(cx + r * math.cos(2 * math.pi * i / N_VERT), 3), round(cy + r * math.sin(2 * math.pi * i / N_VERT), 3)]
            for i in range(N_VERT)]


def d_nm(r):
    return round(2 * r * K32 * NM, 2)


def write_png(path, circles):
    from PIL import Image, ImageDraw
    im = Image.new("L", (W, H), 150)
    dr = ImageDraw.Draw(im)
    for cx, cy, r in circles:
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=60, width=3)
    im.save(path)


def main():
    root = HERE / "predict" / PS
    pred = root / "p" / MODEL / "predictions"
    (root / "images").mkdir(parents=True, exist_ok=True)
    pred.mkdir(parents=True, exist_ok=True)
    rows = ["image,file,orig_name,sample_label,nm_per_px,nm_confirmed_by,source"]
    for img in ("0001", "0002", "0003"):
        objs = [o for o in OBJECTS if o[1] == img]
        write_png(root / "images" / f"{img}.png", [(o[2], o[3], o[4]) for o in objs])
        lines = []
        for o in objs:
            pts = " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in poly(o[2], o[3], o[4]))
            lines.append(f"{o[5]} {o[6]:.2f} {pts}")
        (pred / f"{img}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        rows.append(f"{img},images/{img}.png,golden_{img}.png,{SAMPLE[img]},{NM},pujan,session:golden")
    (root / "images.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "request.json").write_text(json.dumps({
        "ps": PS, "owner": "pujan", "model": MODEL, "conf": 0.25, "max_det": 1000, "imgsz": 400,
        "source": {"kind": "session", "session": "golden", "dir": "<fixture>"}, "created": "2026-11-05T20:00:00-08:00"},
        indent=1) + "\n", encoding="utf-8")
    (root / "status.json").write_text(json.dumps({"state": "done", "n_done": 3, "n_total": 3, "where": "laptop",
                                                  "job": None, "error": None, "updated": "2026-11-05T20:01:00-08:00"},
                                                 indent=1) + "\n", encoding="utf-8")
    (root / "p" / MODEL / "summary.csv").write_text(
        "image,model_id,n_detections,mean_conf\n" + "".join(
            f"{i}.png,{MODEL},{sum(1 for o in OBJECTS if o[1] == i)},"
            f"{sum(o[6] for o in OBJECTS if o[1] == i) / sum(1 for o in OBJECTS if o[1] == i):.4f}\n"
            for i in ("0001", "0002", "0003")), encoding="utf-8")

    def obj(o, rc=None, src="model"):
        return {"cls": o[5], "poly": poly(o[2], o[3], o[4]), "conf": o[6], "source": src, "reviewer_class": rc}
    by = {o[0]: o for o in OBJECTS}
    reviewed = {"schema": "reviewed/1", "package": f"pred_{PS}", "prediction_set": PS, "model": MODEL, "images": {
        "0001": {"reviewed": False, "reviewed_by": [], "objects": [obj(by[k]) for k in "ABCE"]},
        "0002": {"reviewed": True, "reviewed_by": ["pujan"],
                 "objects": [obj(by["F"], "MV"), obj(by["H"], "nonEV"), obj(by["I"], "nonEV"), obj(by["J"], "EV")]},
        "0003": {"reviewed": False, "reviewed_by": [], "objects": [obj(by[k]) for k in "KLN"]}}}
    (HERE / "reviewed.json").write_text(json.dumps(reviewed, indent=1) + "\n", encoding="utf-8")

    # ---- hand-derived expectations ------------------------------------------------------------------
    s1_noedit = dict(n_images=2, n_reviewed=0, ev_total=5, ev_per_image_mean=2.5,
                     diam_nm_median=d_nm(30), diam_nm_q1=d_nm(30), diam_nm_q3=d_nm(35),   # radii A40 C30 F35 G30 J28
                     pct_multilayer_geometry=20.0, pct_multilayer_reviewer=None, inner_layers_excluded=1, nonev_total=3)
    s2 = dict(n_images=1, n_reviewed=0, ev_total=2, ev_per_image_mean=2.0,
              diam_nm_median=d_nm(40), diam_nm_q1=d_nm(35), diam_nm_q3=d_nm(45),            # radii K50 N30
              pct_multilayer_geometry=50.0, pct_multilayer_reviewer=None, inner_layers_excluded=1, nonev_total=0)
    img_noedit = {"0001": dict(sample_label="S1", reviewed=False, nm_per_px=NM, ev=2, multilayer=1, inner=1, nonev=1),
                  "0002": dict(sample_label="S1", reviewed=False, nm_per_px=NM, ev=3, multilayer=0, inner=0, nonev=2),
                  "0003": dict(sample_label="S2", reviewed=False, nm_per_px=NM, ev=2, multilayer=1, inner=1, nonev=0)}
    rule = {"conf": 0.5, "ev_classes": [0], "multilayer": "geometry: outline containing another outline"}
    base = {"schema": "results/1", "prediction_set": PS, "model": MODEL, "rule": rule, "refused": None,
            "_tolerance": {"diam_nm_*": "relative 0.02", "other": "exact"}}
    (HERE / "expected_results_noedit.json").write_text(json.dumps(
        {**base, "samples": {"S1": s1_noedit, "S2": s2}, "images": img_noedit}, indent=1) + "\n", encoding="utf-8")
    s1_rev = dict(n_images=2, n_reviewed=1, ev_total=4, ev_per_image_mean=2.0,
                  diam_nm_median=d_nm(32.5), diam_nm_q1=d_nm(29.5), diam_nm_q3=d_nm(36.25),  # radii A40 C30 F35 J28
                  pct_multilayer_geometry=25.0, pct_multilayer_reviewer=50.0,                # reviewer: F=MV of {F, J}
                  inner_layers_excluded=1, nonev_total=3)
    img_rev = dict(img_noedit, **{"0002": dict(sample_label="S1", reviewed=True, nm_per_px=NM, ev=2, multilayer=0,
                                               inner=0, nonev=2)})
    (HERE / "expected_results_reviewed.json").write_text(json.dumps(
        {**base, "samples": {"S1": s1_rev, "S2": s2}, "images": img_rev,
         "_note": "pct_multilayer_reviewer = share of EVs the reviewer classed MV, over REVIEWED images only; "
                  "null for a sample with no reviewed image"}, indent=1) + "\n", encoding="utf-8")
    print("golden fixtures written under", HERE)


if __name__ == "__main__":
    main()
