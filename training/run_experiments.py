"""
Experiment Campaign Orchestrator — 20260713 All-Layer Single-Class Sweep
Runs 8 experiments on the collapsed single-class "EV" dataset
(roboflow_20260713_all_layer_singleclass), on an RTX A6000 (48GB VRAM).

Every discrete membrane layer (EV/Incomplete/Low conf EV/MV/inner) was
collapsed into one "EV" class; multilayer status will be determined later
via polygon containment, not by a model class. See:
  agent-documentation plan: single-class EV training on the all-layer dataset.
"""

import argparse
import os
import sys
import json
import csv
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_yolo import (
    train_yolo_segmentation, calculate_per_class_segmentation_metrics,
    save_per_class_metrics, visualize_predictions_with_matching,
    export_predictions_for_split,
)
from notify import notify_completion, notify_error, notify_progress, send_notification

CRYOAI_ROOT = Path(__file__).parent.parent.parent / "CryoAI"

# Dataset (single-class EV, all layers collapsed; existing Roboflow split preserved).
# Overridden by --dataset in the __main__ block below.
DATASET_ID = "roboflow_20260713_all_layer_singleclass"
DATASET_ROOT = CRYOAI_ROOT / "training outputs" / DATASET_ID
DATASET_YAML = DATASET_ROOT / "dataset.yaml"
PROJECT_ROOT = DATASET_ROOT / "runs"
OUTPUT_LOG = CRYOAI_ROOT / "EXPERIMENT_LOG.csv"

CLASS_NAMES = ["EV"]


def count_split_stats(dataset_root: Path, split: str) -> Tuple[int, int]:
    """Count images and label instances for a split by reading the converted dataset on disk."""
    img_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not img_dir.exists():
        return 0, 0
    n_images = len([p for p in img_dir.iterdir() if p.is_file()])
    n_instances = 0
    if label_dir.exists():
        for txt in label_dir.glob("*.txt"):
            n_instances += sum(1 for line in txt.read_text(encoding="utf-8").splitlines() if line.strip())
    return n_images, n_instances


# Dataset stats -- recomputed from disk in the __main__ block below once
# DATASET_ROOT is known (was previously hand-transcribed from the prep script's
# printed output; now derived directly so it's always correct for whatever
# --dataset is selected).
TRAIN_IMAGES = TRAIN_INSTANCES = VAL_IMAGES = VAL_INSTANCES = TEST_IMAGES = TEST_INSTANCES = 0

# Known-best config from the 20260713 campaign (Run 14, mask mAP50-95=0.6909,
# best in that campaign): overlap_mask=False + native-res rect + full batch.
# Selected via --single-run instead of re-sweeping on a new dataset.
BEST_CONFIG = {
    'run_num': 1,
    'name': 'Native-res full-batch y8n 1440/b24 AdamW rect (Run14 config)',
    'model_version': '8', 'model_size': 'n',
    'imgsz': 1440, 'batch_size': 24,
    'epochs': 300, 'patience': 75, 'workers': 8,
    'optimizer': 'AdamW', 'augmentations': 'on',
    'overlap_mask': False, 'mask_ratio': 4, 'rect': True,
}

# Runs 1-11 completed (see EXPERIMENT_LOG.csv). Run 10 (overlap_mask=False,
# mask_ratio=4, imgsz=1440, rect=True, batch=8) is the best config found so
# far: mask mAP50-95=0.684, peak GPU only 7.6GB/51.5GB -- huge headroom left.
# Run 12 (mask_ratio=1) was killed by the user: pinned at 98.6% VRAM with
# 10-50x slower iterations (shared-memory spillover), no clear benefit over
# Run 9/11 -- mask_ratio=1 is excluded from further consideration.
#
# This queue scales Run 10's config up along the two axes not yet tested at
# native resolution, plus the augmentation lever:
#   13 - batch scale-up (8 -> 16) at native res
#   14 - batch scale-up further (8 -> 24, ~full-batch given 26 train images)
#   15 - partial mask-resolution fidelity (mask_ratio 4 -> 2) at native res
#   16 - augmentation off, to directly test the augmentation lever
EXPERIMENTS = [
    {
        'run_num': 13,
        'name': 'Native-res batch scale-up y8n 1440/b16 AdamW rect',
        'model_version': '8', 'model_size': 'n',
        'imgsz': 1440, 'batch_size': 16,
        'epochs': 300, 'patience': 75, 'workers': 8,
        'optimizer': 'AdamW', 'augmentations': 'on',
        'overlap_mask': False, 'mask_ratio': 4, 'rect': True,
    },
    {
        'run_num': 14,
        'name': 'Native-res full-batch y8n 1440/b24 AdamW rect',
        'model_version': '8', 'model_size': 'n',
        'imgsz': 1440, 'batch_size': 24,
        'epochs': 300, 'patience': 75, 'workers': 8,
        'optimizer': 'AdamW', 'augmentations': 'on',
        'overlap_mask': False, 'mask_ratio': 4, 'rect': True,
    },
    {
        'run_num': 15,
        'name': 'Native-res mask_ratio=2 y8n 1440/b8 AdamW rect',
        'model_version': '8', 'model_size': 'n',
        'imgsz': 1440, 'batch_size': 8,
        'epochs': 300, 'patience': 75, 'workers': 8,
        'optimizer': 'AdamW', 'augmentations': 'on',
        'overlap_mask': False, 'mask_ratio': 2, 'rect': True,
    },
    {
        'run_num': 16,
        'name': 'Native-res no-augmentation y8n 1440/b8 AdamW rect',
        'model_version': '8', 'model_size': 'n',
        'imgsz': 1440, 'batch_size': 8,
        'epochs': 300, 'patience': 75, 'workers': 8,
        'optimizer': 'AdamW', 'augmentations': 'off',
        'overlap_mask': False, 'mask_ratio': 4, 'rect': True,
    },
]


def get_best_metrics(results_csv_path: str) -> Dict:
    """Extract best segmentation metrics from training results.csv."""
    import csv
    with open(results_csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    best = max(rows, key=lambda r: float(r['metrics/mAP50-95(M)']))
    return {
        'best_epoch': best['epoch'],
        'seg_map50': float(best['metrics/mAP50(M)']),
        'seg_map50_95': float(best['metrics/mAP50-95(M)']),
        'box_map50': float(best['metrics/mAP50(B)']),
        'box_map50_95': float(best['metrics/mAP50-95(B)']),
        'precision': float(best['metrics/precision(M)']),
        'recall': float(best['metrics/recall(M)']),
        'total_epochs': len(rows),
    }


def log_experiment_result(result: Dict, log_file: Path):
    """Append experiment result to EXPERIMENT_LOG.csv"""
    row = {
        'timestamp': result['timestamp'],
        'experiment_name': result['experiment_name'],
        'dataset_id': DATASET_ID,
        'model': result['model'],
        'classes': '|'.join(CLASS_NAMES),
        'epochs': result['epochs'],
        'imgsz': result['imgsz'],
        'batch_size': result['batch_size'],
        'patience': result.get('patience', 50),
        'train_images': TRAIN_IMAGES,
        'train_instances': TRAIN_INSTANCES,
        'val_images': VAL_IMAGES,
        'val_instances': VAL_INSTANCES,
        'test_images': TEST_IMAGES,
        'test_instances': TEST_INSTANCES,
        'box_map50': f"{result['box_map50']:.4f}",
        'box_map50_95': f"{result['box_map50_95']:.4f}",
        'mask_map50': f"{result['mask_map50']:.4f}",
        'mask_map50_95': f"{result['mask_map50_95']:.4f}",
        'best_model_path': result['best_model_path'] or '',
        'run_notes': f"{result['config_name']} | Status: {result['status']} | Time: {result['training_time_sec']:.0f}s | BestEp:{result.get('best_epoch','?')}",
        'peak_gpu_memory_gb': f"{result.get('peak_gpu_memory_gb', 0):.2f}",
        'total_training_time_sec': f"{result['training_time_sec']:.0f}",
        'avg_epoch_time_sec': f"{result['avg_epoch_time_sec']:.1f}",
        'optimizer': result.get('optimizer', 'AdamW'),
        'augmentations': result.get('augmentations', 'on'),
    }

    file_exists = log_file.exists()
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[OK] Logged to {log_file}")


def generate_campaign_summary(all_results: List[Dict], output_file: Path):
    """Generate a high-level briefing document for the sweep."""
    hp_completed = [r for r in all_results if r['status'] == 'completed']

    summary = f"""# Campaign Results Brief — {DATASET_ID}
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset**: {DATASET_ID} (1 class "EV", all membrane layers collapsed)

---

## Hyperparameter Sweep (26 train / 5 val images, 300ep, RTX A6000)

| Run | Config | Mask mAP@.5:95 | Mask mAP@.5 | Best Ep | Time (min) | Status |
|:---:|:-------|:--------------:|:-----------:|:-------:|:----------:|:------:|
"""
    for r in hp_completed:
        summary += f"| {r['run_num']} | {r['config_name'][:35]} | {r['mask_map50_95']:.4f} | {r['mask_map50']:.4f} | {r.get('best_epoch','?')} | {r['training_time_sec']/60:.1f} | {r['status']} |\n"

    failed = [r for r in all_results if r['status'] not in ('completed',)]
    if failed:
        summary += "\n### Failed/OOM Runs\n"
        for r in failed:
            summary += f"- **Run {r['run_num']}** ({r['config_name']}): {r['error']}\n"

    total_time = sum(r['training_time_sec'] for r in hp_completed) / 3600
    summary += f"""
---

**Total Campaign Duration**: ~{total_time:.1f} hours
**Total Runs**: {len(all_results)}

*Full results logged to: EXPERIMENT_LOG.csv*
"""

    with open(output_file, 'w') as f:
        f.write(summary)
    print(f"[OK] Summary saved to {output_file}")


def main():
    """Run the 8-experiment sweep on the 20260713 all-layer single-class dataset."""

    campaign_start = datetime.now()
    print(f"\n{'='*80}")
    print(f"{DATASET_ID}: {len(EXPERIMENTS)} RUN(S)")
    print(f"Started: {campaign_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    if not DATASET_YAML.exists():
        print(f"[x] Dataset YAML not found at {DATASET_YAML}")
        send_notification("Campaign Setup Error", f"Missing dataset YAML")
        return

    print(f"[OK] Dataset: {DATASET_YAML}")
    print(f"[OK] {len(EXPERIMENTS)} experiments remaining\n")

    all_results = []

    for config in EXPERIMENTS:
        run_num = config['run_num']
        notify_progress(run_num, len(EXPERIMENTS), "started")

        print(f"\n{'='*80}")
        print(f"RUN {run_num}: {config['name']}")
        print(f"{'='*80}")

        experiment_name = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_y{config['model_version']}_" \
                         f"{config['imgsz']}_{config['batch_size']}b_{config['epochs']}ep_Run_{run_num}"

        result = {
            'run_num': run_num,
            'config_name': config['name'],
            'experiment_name': experiment_name,
            'model': f"yolo{config['model_version']}-seg",
            'imgsz': config['imgsz'],
            'batch_size': config['batch_size'],
            'epochs': config['epochs'],
            'optimizer': config.get('optimizer', 'AdamW'),
            'augmentations': config.get('augmentations', 'on'),
            'timestamp': datetime.now().isoformat(),
            'status': 'running',
            'error': None,
            'training_time_sec': 0,
            'peak_gpu_memory_gb': 0.0,
            'avg_epoch_time_sec': 0.0,
            'box_map50': 0.0,
            'box_map50_95': 0.0,
            'mask_map50': 0.0,
            'mask_map50_95': 0.0,
            'best_epoch': None,
            'best_model_path': None,
        }

        # Write LATEST_RUN.txt
        latest_run_path = CRYOAI_ROOT / "LATEST_RUN.txt"
        try:
            latest_run_path.write_text(
                f"experiment_name: {experiment_name}\n"
                f"training_dir: {str(PROJECT_ROOT / experiment_name)}\n"
                f"dataset_yaml: {str(DATASET_YAML)}\n"
                f"started: {datetime.now().isoformat()}\n"
            )
        except Exception as e:
            print(f"Warning: Could not write LATEST_RUN.txt: {e}")

        try:
            augs_on = config.get('augmentations', 'on') == 'on'
            training_kwargs = {
                'data_yaml': str(DATASET_YAML),
                'model_size': config.get('model_size', 'n'),
                'epochs': config['epochs'],
                'imgsz': config['imgsz'],
                'batch_size': config['batch_size'],
                'device': '0',
                'project': str(PROJECT_ROOT),
                'name': experiment_name,
                'patience': config.get('patience', 50),
                'use_v11': config['model_version'] == '11',
                'workers': config.get('workers', 8),
                'optimizer': config.get('optimizer', 'AdamW'),
                'overlap_mask': config.get('overlap_mask', True),
                'mask_ratio': config.get('mask_ratio', 4),
                'rect': config.get('rect', False),
                'hsv_h': 0.015 if augs_on else 0,
                'hsv_s': 0.7 if augs_on else 0,
                'hsv_v': 0.4 if augs_on else 0,
                'degrees': 45.0 if augs_on else 0,
                'translate': 0.1 if augs_on else 0,
                'scale': 0.5 if augs_on else 0,
                'shear': 0.0,
                'perspective': 0.0,
                'fliplr': 0.5 if augs_on else 0,
                'flipud': 0.5 if augs_on else 0,
                'mosaic': 1.0 if augs_on else 0,
                'mixup': 0.15 if augs_on else 0,
                'copy_paste': 0.3 if augs_on else 0,
            }

            print(f"Starting training...\n")
            train_start = time.time()

            results = train_yolo_segmentation(**training_kwargs)

            train_elapsed = time.time() - train_start
            result['training_time_sec'] = train_elapsed
            result['best_model_path'] = str(results.save_dir / 'weights' / 'best.pt')

            # Extract metrics from results.csv (more reliable than results object)
            results_csv = Path(results.save_dir) / "results.csv"
            if results_csv.exists():
                metrics = get_best_metrics(str(results_csv))
                result['best_epoch'] = metrics.get('best_epoch')
                result['mask_map50'] = metrics.get('seg_map50', 0)
                result['mask_map50_95'] = metrics.get('seg_map50_95', 0)
                result['box_map50'] = metrics.get('box_map50', 0)
                result['box_map50_95'] = metrics.get('box_map50_95', 0)
            else:
                # Fallback to results object
                if hasattr(results, 'seg') and results.seg:
                    result['mask_map50'] = float(results.seg.map50)
                    result['mask_map50_95'] = float(results.seg.map)

            result['avg_epoch_time_sec'] = train_elapsed / config['epochs']
            result['status'] = 'completed'

            # Per-class metrics + prediction overlays for this run, on BOTH val
            # and train splits (per user request: seeing train-split failures
            # too helps distinguish "model can't learn this pattern" from
            # "model hasn't seen enough of this pattern" -- e.g. if the model
            # also fails on train images it saw directly, that points at the
            # nested-mask encoding rather than at generalization/overfitting).
            best_pt = Path(results.save_dir) / 'weights' / 'best.pt'
            for split_name in ("val", "train", "test"):
                if not best_pt.exists():
                    break
                if not (DATASET_ROOT / "images" / split_name).exists():
                    continue  # this dataset has no test split, etc.
                try:
                    pc_metrics = calculate_per_class_segmentation_metrics(
                        model_path=str(best_pt),
                        img_dir=str(DATASET_ROOT / f"images/{split_name}"),
                        label_dir=str(DATASET_ROOT / f"labels/{split_name}"),
                        imgsz=config['imgsz'], conf=0.25, iou=0.5, device='0',
                        class_names=CLASS_NAMES
                    )
                    save_per_class_metrics(pc_metrics, results.save_dir / f"per_class_metrics_{split_name}.txt", split_name)
                    for name, m in pc_metrics.items():
                        print(f"  [{split_name}] {name}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
                except Exception as pc_e:
                    print(f"  [WARN] Per-class metrics ({split_name}) failed: {pc_e}")

                try:
                    visualize_predictions_with_matching(
                        model_path=str(best_pt),
                        source_dir=str(DATASET_ROOT / f"images/{split_name}"),
                        label_dir=str(DATASET_ROOT / f"labels/{split_name}"),
                        output_dir=str(results.save_dir / "overlays" / split_name),
                        imgsz=config['imgsz'], conf=0.25, iou=0.7, device='0',
                        match_threshold=0.5, class_names=CLASS_NAMES,
                    )
                    print(f"  [OK] {split_name} overlays saved to {results.save_dir / 'overlays' / split_name / 'visualizations'}")
                except Exception as ov_e:
                    print(f"  [WARN] Overlay generation ({split_name}) failed: {ov_e}")

                # Cache raw prediction polygons to disk so secondary analyses (multilayer
                # containment, confidence sweeps, etc.) can reuse them without re-running
                # inference. Free: the overlay/metrics calls above already run predict()
                # per image, this just persists one more pass instead of discarding it.
                try:
                    export_predictions_for_split(
                        model_path=str(best_pt),
                        img_dir=str(DATASET_ROOT / f"images/{split_name}"),
                        output_dir=results.save_dir / "predictions" / split_name,
                        imgsz=config['imgsz'], conf=0.25, iou=0.7, device='0',
                    )
                    print(f"  [OK] {split_name} predictions cached to {results.save_dir / 'predictions' / split_name}")
                except Exception as exp_e:
                    print(f"  [WARN] Prediction export ({split_name}) failed: {exp_e}")

            print(f"\n[OK] Run {run_num} completed successfully")

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                result['error'] = f"OOM at batch={config['batch_size']}"
                result['status'] = 'oom'
            else:
                result['error'] = str(e)
                result['status'] = 'error'
            notify_error(run_num, str(e)[:200])
        except Exception as e:
            result['error'] = str(e)
            result['status'] = 'error'
            notify_error(run_num, str(e)[:200])

        all_results.append(result)
        log_experiment_result(result, OUTPUT_LOG)

        print(f"  Metrics: SEGmAP50={result['mask_map50']:.4f}, SEGmAP50-95={result['mask_map50_95']:.4f}")
        print(f"  Time: {result['training_time_sec']/60:.1f} min, Status: {result['status']}")

        if result['status'] == 'completed':
            notify_progress(run_num, len(EXPERIMENTS), "completed")

    # Summary
    campaign_end = datetime.now()
    total_hours = (campaign_end - campaign_start).total_seconds() / 3600
    completed = len([r for r in all_results if r['status'] == 'completed'])
    failed = len([r for r in all_results if r['status'] != 'completed'])

    print(f"\n{'='*80}")
    print(f"SWEEP COMPLETE")
    print(f"Ended: {campaign_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {total_hours:.1f} hours")
    print(f"Runs: {len(all_results)} ({completed} OK, {failed} Fail)")
    print(f"{'='*80}\n")

    generate_campaign_summary(all_results, DATASET_ROOT / "CAMPAIGN_RESULTS_BRIEF.md")

    notify_completion(total_runs=len(all_results), completed=completed, failed=failed, duration_hours=total_hours)
    send_notification("Sweep Complete", f"{completed} OK, {failed} failed. Review results in {DATASET_ROOT / 'CAMPAIGN_RESULTS_BRIEF.md'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_ID,
                         help="Folder name under 'training outputs/' to train against")
    parser.add_argument("--single-run", action="store_true",
                         help="Run only the known-best Run 14 config instead of the EXPERIMENTS sweep list")
    cli_args = parser.parse_args()

    DATASET_ID = cli_args.dataset
    DATASET_ROOT = CRYOAI_ROOT / "training outputs" / DATASET_ID
    DATASET_YAML = DATASET_ROOT / "dataset.yaml"
    PROJECT_ROOT = DATASET_ROOT / "runs"

    TRAIN_IMAGES, TRAIN_INSTANCES = count_split_stats(DATASET_ROOT, "train")
    VAL_IMAGES, VAL_INSTANCES = count_split_stats(DATASET_ROOT, "val")
    TEST_IMAGES, TEST_INSTANCES = count_split_stats(DATASET_ROOT, "test")

    if cli_args.single_run:
        EXPERIMENTS = [BEST_CONFIG]

    main()