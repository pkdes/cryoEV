#!/usr/bin/env python3
"""Add multilayer role + layer count to a morphology_all.csv, from cached predictions.

All-layer models (v3) draw one polygon per membrane, so a 3-layer EV is three
detections. classify_roles() -- unchanged, from multilayer_containment.py --
turns that back into EVs: 'multilayer' (outer boundary), 'inner-layer'
(nested, not an EV in its own right) and 'standalone'.

Roles are assigned among the SAME kept masks that size_profile_from_predictions
measured (confidence >= threshold, file order), so object_id joins directly.
Thresholds are the validated pair from layer_count_stats.py.

    python -m analysis.add_layer_roles \
        --morphology-csv <out>/morphology/<model>/morphology_all.csv \
        --predictions-dir <out>/predictions/<model>/predictions \
        --images-dir <tiles> --confidence-threshold 0.5

    python -m analysis.add_layer_roles --self-test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from analysis.layer_count_stats import CONTAINMENT_THRESHOLD, MAX_SIZE_RATIO
from analysis.multilayer_containment import classify_roles
from training.train_yolo import load_predictions_yolo_format


def roles_for(masks):
    return classify_roles(masks, CONTAINMENT_THRESHOLD, MAX_SIZE_RATIO)


def self_test() -> None:
    """Known answer: one 3-layer stack + 2 standalones."""
    def disc(cx, cy, r, shape=(400, 400)):
        m = np.zeros(shape, np.uint8)
        cv2.circle(m, (cx, cy), r, 1, -1)
        return m.astype(bool)

    masks = [disc(100, 100, 60), disc(100, 100, 40), disc(100, 100, 20),
             disc(300, 300, 30), disc(300, 100, 25)]
    role, layers = roles_for(masks)
    got = [role[i] for i in range(5)]
    want = ['multilayer', 'inner-layer', 'inner-layer', 'standalone', 'standalone']
    assert got == want, got
    assert layers[0] == 2, layers
    print('self-test PASS:', got, 'outer layer_count =', layers[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--morphology-csv', type=Path)
    ap.add_argument('--predictions-dir', type=Path)
    ap.add_argument('--images-dir', type=Path)
    ap.add_argument('--confidence-threshold', type=float, default=0.5)
    ap.add_argument('--output-csv', type=Path, default=None,
                    help='default: morphology_all_roles.csv beside the input')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    with args.morphology_csv.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    by_image = defaultdict(list)
    for r in rows:
        by_image[r['image_rel_path']].append(r)

    for rel, recs in by_image.items():
        image_path = args.images_dir / rel
        with Image.open(image_path) as im:
            w, h = im.size
        masks, confs, _ = load_predictions_yolo_format(
            args.predictions_dir / f'{image_path.stem}.txt', w, h)
        kept = [m for m, c in zip(masks, confs) if c >= args.confidence_threshold]
        role, layers = roles_for(kept)
        for r in recs:
            i = int(r['object_id'])
            # Same kept list, same order: confidence must agree to 4 dp.
            assert abs(float(r['confidence']) - [c for c in confs
                       if c >= args.confidence_threshold][i]) < 1e-4, (rel, i)
            r['role'] = role[i]
            r['layer_count'] = layers[i]

    out = args.output_csv or args.morphology_csv.with_name('morphology_all_roles.csv')
    with out.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    n = defaultdict(int)
    for r in rows:
        n[r['role']] += 1
    print(f'{len(rows)} objects over {len(by_image)} image(s): {dict(n)} -> {out}')


if __name__ == '__main__':
    main()
