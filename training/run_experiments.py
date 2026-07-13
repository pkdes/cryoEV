"""
Overnight Experiment Campaign Orchestrator — Session 3: Hyperparameter Sweep
Runs 6 experiments with different hyperparameters on 100% stratified data.
"""

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
from train_yolo import train_yolo_segmentation, calculate_per_class_segmentation_metrics, save_per_class_metrics
from notify import notify_completion, notify_error, notify_progress, send_notification

# Dataset (100% stratified)
DATASET_ROOT = Path(__file__).parent.parent.parent / "training outputs" / "roboflow20260604_stratified"
DATASET_YAML = DATASET_ROOT / "dataset.yaml"
PROJECT_ROOT = DATASET_ROOT / "runs"
OUTPUT_LOG = Path(__file__).parent.parent.parent / "EXPERIMENT_LOG.csv"

CLASS_NAMES = ["EV", "multilayer EV"]

# Dataset stats
TRAIN_IMAGES = 74
TRAIN_INSTANCES = 1037
VAL_IMAGES = 26
VAL_INSTANCES = 668
TEST_IMAGES = 10
TEST_INSTANCES = 197

# Hyperparameter sweep configs — skip Run 1 (already completed)
# Resume from Run 2
EXPERIMENTS = [
    {
        'run_num': 2,
        'name': 'Hi-Res 768/b2 AdamW 300ep',
        'model_version': '8',
        'imgsz': 768,
        'batch_size': 2,
        'epochs': 300,
        'patience': 50,
        'workers': 4,
        'optimizer': 'AdamW',
        'augmentations': 'on',
    },
    {
        'run_num': 3,
        'name': 'Max-Res 1024/b1 AdamW 300ep',
        'model_version': '8',
        'imgsz': 1024,
        'batch_size': 1,
        'epochs': 300,
        'patience': 50,
        'workers': 2,
        'optimizer': 'AdamW',
        'augmentations': 'on',
    },
    {
        'run_num': 4,
        'name': 'SGD Optimizer 640/b4 300ep',
        'model_version': '8',
        'imgsz': 640,
        'batch_size': 4,
        'epochs': 300,
        'patience': 50,
        'workers': 4,
        'optimizer': 'SGD',
        'augmentations': 'on',
    },
    {
        'run_num': 5,
        'name': 'No Augs 640/b4 AdamW 300ep',
        'model_version': '8',
        'imgsz': 640,
        'batch_size': 4,
        'epochs': 300,
        'patience': 50,
        'workers': 4,
        'optimizer': 'AdamW',
        'augmentations': 'off',
    },
    {
        'run_num': 6,
        'name': 'YOLOv11n 640/b4 AdamW 300ep',
        'model_version': '11',
        'imgsz': 640,
        'batch_size': 4,
        'epochs': 300,
        'patience': 50,
        'workers': 4,
        'optimizer': 'AdamW',
        'augmentations': 'on',
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
        'dataset_id': 'roboflow20260604_stratified',
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


def generate_campaign_summary(all_results: List[Dict], data_scaling_results: List[Dict], output_file: Path):
    """Generate a high-level briefing document combining data scaling + hyperparameter sweep."""
    hp_completed = [r for r in all_results if r['status'] == 'completed']

    summary = f"""# Campaign Results Brief — Sessions 2 & 3
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset**: roboflow20260604_stratified (2 classes, stratified split)

---

## Session 2: Data Scaling (200ep, 640/b4)

| Run | Data % | Mask mAP@.5:95 | Mask mAP@.5 | EV F1 | multilayer F1 |
|:---:|:------:|:--------------:|:-----------:|:-----:|:-------------:|
"""
    for r in data_scaling_results:
        summary += f"| {r['run_num']} | {r['config_name'][:25]} | {r['mask_map50_95']:.4f} | {r['mask_map50']:.4f} | {r.get('ev_f1',0):.3f} | {r.get('ml_f1',0):.3f} |\n"

    summary += f"""

---

## Session 3: Hyperparameter Sweep (100% data, 300ep)

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

    total_time = sum(r['training_time_sec'] for r in hp_completed + data_scaling_results) / 3600
    summary += f"""
---

**Total Campaign Duration**: ~{total_time:.1f} hours  
**Total Runs**: {len(all_results) + len(data_scaling_results)}

*Full results logged to: EXPERIMENT_LOG.csv*
"""

    with open(output_file, 'w') as f:
        f.write(summary)
    print(f"[OK] Summary saved to {output_file}")


def main():
    """Resume campaign from Run 2 (Run 1 already completed)."""
    
    campaign_start = datetime.now()
    print(f"\n{'='*80}")
    print(f"SESSION 3 RESUMED: HYPERPARAMETER SWEEP (Runs 2-6)")
    print(f"Run 1 already completed. Resuming from Run 2.")
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
        notify_progress(run_num, 6, "started")

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
        latest_run_path = Path(__file__).parent.parent.parent / "LATEST_RUN.txt"
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
                'model_size': 'n',
                'epochs': config['epochs'],
                'imgsz': config['imgsz'],
                'batch_size': config['batch_size'],
                'device': '0',
                'project': str(PROJECT_ROOT),
                'name': experiment_name,
                'patience': config.get('patience', 50),
                'use_v11': config['model_version'] == '11',
                'workers': config.get('workers', 4),
                'optimizer': config.get('optimizer', 'AdamW'),
                'hsv_h': 0.015 if augs_on else 0,
                'hsv_s': 0.7 if augs_on else 0,
                'hsv_v': 0.4 if augs_on else 0,
                'degrees': 45.0 if augs_on else 0,
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

            # Save per-class metrics for completed runs
            try:
                best_pt = Path(results.save_dir) / 'weights' / 'best.pt'
                if best_pt.exists():
                    pc_metrics = calculate_per_class_segmentation_metrics(
                        model_path=str(best_pt),
                        img_dir=str(DATASET_ROOT / "images/val"),
                        label_dir=str(DATASET_ROOT / "labels/val"),
                        imgsz=config['imgsz'], conf=0.25, iou=0.5, device='0',
                        class_names=CLASS_NAMES
                    )
                    save_per_class_metrics(pc_metrics, results.save_dir / "per_class_metrics.txt", "val")
                    for name, m in pc_metrics.items():
                        print(f"  {name}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
            except Exception as pc_e:
                print(f"  [WARN] Per-class metrics failed: {pc_e}")

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
            notify_progress(run_num, 6, "completed")

    # Summary
    campaign_end = datetime.now()
    total_hours = (campaign_end - campaign_start).total_seconds() / 3600
    completed = len([r for r in all_results if r['status'] == 'completed'])
    failed = len([r for r in all_results if r['status'] != 'completed'])

    print(f"\n{'='*80}")
    print(f"SESSION 3 COMPLETE")
    print(f"Ended: {campaign_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {total_hours:.1f} hours")
    print(f"Runs in this session: {len(all_results)} ({completed} OK, {failed} Fail)")
    print(f"{'='*80}\n")

    notify_completion(total_runs=len(all_results), completed=completed, failed=failed, duration_hours=total_hours)
    send_notification("Session 3 Complete", f"Runs 2-6 done. {completed} OK, {failed} failed. Review results in CAMPAIGN_RESULTS_BRIEF.md")


if __name__ == '__main__':
    main()