"""
Lightweight campaign monitor — check progress every N minutes.
Run manually:  python cryoEV/training/monitor_campaign.py
Or schedule with: python cryoEV/training/monitor_campaign.py --watch (loops every 5 min)
"""

import sys
import time
import csv
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from notify import send_notification

PROJECT_ROOT = Path(__file__).parent.parent.parent
LATEST_RUN_FILE = PROJECT_ROOT / "LATEST_RUN.txt"
LOG_FILE = Path(os.environ.get("TEMP", ".")) / "cline" / "background-1780645749563-1pwkskx.log"
OUTPUTS_DIR = PROJECT_ROOT / "training outputs"

RUN_NAMES = {
    1: "Baseline 640/b4 AdamW 300ep",
    2: "Hi-Res 768/b2 AdamW 300ep",
    3: "Max-Res 1024/b1 AdamW 300ep",
    4: "SGD 640/b4 300ep",
    5: "No Augs 640/b4 AdamW 300ep",
    6: "YOLOv11n 640/b4 AdamW 300ep",
}

def get_latest_run_info():
    if not LATEST_RUN_FILE.exists():
        return None
    info = {}
    for line in LATEST_RUN_FILE.read_text().strip().split('\n'):
        if ':' in line:
            k, v = line.split(':', 1)
            info[k.strip()] = v.strip()
    return info

def get_latest_results_csv():
    """Find the newest results.csv from runs dirs."""
    runs_dir = OUTPUTS_DIR / "roboflow20260604_stratified" / "runs"
    if not runs_dir.exists():
        return None, None
    run_dirs = sorted(runs_dir.iterdir(), reverse=True)
    for rd in run_dirs:
        csv_path = rd / "results.csv"
        if csv_path.exists():
            return csv_path, rd.name
    return None, None

def parse_epoch_csv(csv_path):
    """Read last few epochs from results.csv."""
    if not csv_path or not csv_path.exists():
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    last5 = rows[-5:]
    best = max(rows, key=lambda r: float(r['metrics/mAP50-95(M)']))
    return {
        'total_epochs': len(rows),
        'last5_epochs': [int(r['epoch']) for r in last5],
        'last5_map': [float(r['metrics/mAP50-95(M)']) for r in last5],
        'best_epoch': best['epoch'],
        'best_map50_95': float(best['metrics/mAP50-95(M)']),
        'best_map50': float(best['metrics/mAP50(M)']),
    }

def check_process():
    """Check if the training process is still running."""
    import subprocess
    try:
        result = subprocess.run(
            'tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH 2>nul',
            capture_output=True, text=True, shell=True
        )
        lines = [l for l in result.stdout.strip().split('\n') if l]
        # Filter for our specific process
        for line in lines:
            if 'run_experiments' in line.lower() or 'train_yolo' in line.lower():
                return True
        return len(lines) > 0
    except:
        return False

def check_recent_activity(csv_data):
    """Check if model is still improving or stalled."""
    if not csv_data:
        return "unknown"
    last5 = csv_data['last5_map']
    if len(last5) < 2:
        return "too_early"
    # If map hasn't improved in last 5 epochs and is > 30 epochs in
    if csv_data['total_epochs'] > 30 and all(last5[i] <= last5[i+1] for i in range(len(last5)-1)):
        # Check actual change
        if last5[-1] - last5[0] < 0.001:
            return "plateaued"
    return "improving"

def status_report():
    """Generate a status report string."""
    run_info = get_latest_run_info()
    csv_path, run_name = get_latest_results_csv()
    csv_data = parse_epoch_csv(csv_path)
    running = check_process()
    activity = check_recent_activity(csv_data)

    lines = []
    lines.append(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    lines.append(f"Process running: {'YES' if running else 'NO'}")
    
    if run_info:
        lines.append(f"Latest run: {run_info.get('experiment_name', '?')}")
    
    if run_name:
        # Extract run number from folder name
        parts = run_name.split('_Run_')
        run_num = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 0
        config = RUN_NAMES.get(run_num, f"Run {run_num}")
        lines.append(f"Config: {config}")
    
    if csv_data:
        last_ep = csv_data['last5_epochs'][-1] if csv_data['last5_epochs'] else 0
        last_map = csv_data['last5_map'][-1] if csv_data['last5_map'] else 0
        lines.append(f"Epoch: {last_ep}/{csv_data['total_epochs']} (best at ep {csv_data['best_epoch']})")
        lines.append(f"Mask mAP50-95: {last_map:.4f} (best: {csv_data['best_map50_95']:.4f})")
        lines.append(f"Status: {activity}")
    else:
        lines.append("Data: no results.csv yet (starting up or between runs)")
    
    if not running:
        lines.append("⚠️  WARNING: No python process detected!")
    
    return '\n'.join(lines)

def watch_loop(interval_min=5):
    """Monitor every N minutes. Alert on issues."""
    from notify import send_notification
    
    print(f"Campaign Monitor — checking every {interval_min} min")
    print(f"First check in {interval_min} min...\n")
    
    last_epoch = 0
    stalled_count = 0
    
    while True:
        time.sleep(interval_min * 60)
        
        csv_path, run_name = get_latest_results_csv()
        csv_data = parse_epoch_csv(csv_path)
        running = check_process()
        
        current_epoch = csv_data['last5_epochs'][-1] if csv_data and csv_data['last5_epochs'] else 0
        
        if not running:
            send_notification("🚨 CLINE INTERVENTION NEEDED: Campaign Halted", 
                              "Training process crashed or finished abnormally. Please check the logs and ask Cline to investigate.")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ALERT: Process stopped! Notify user to call Cline.")
            break
        
        # Check progress
        if current_epoch == last_epoch and last_epoch > 0:
            stalled_count += 1
            if stalled_count >= 3:  # No progress for 15+ min
                send_notification("🚨 CLINE INTERVENTION NEEDED: Campaign Stalled", 
                                  f"No epoch progress in {stalled_count * interval_min} min at ep {current_epoch}. Ask Cline to check.")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ALERT: Training stalled at ep {current_epoch}")
                stalled_count = 0
        else:
            stalled_count = 0
        
        last_epoch = current_epoch
        
        report = status_report()
        print(report)
        print('-' * 50)


if __name__ == '__main__':
    if '--watch' in sys.argv:
        interval = 5
        for i, arg in enumerate(sys.argv):
            if arg == '--interval' and i + 1 < len(sys.argv):
                interval = int(sys.argv[i + 1])
        watch_loop(interval)
    else:
        print(status_report())