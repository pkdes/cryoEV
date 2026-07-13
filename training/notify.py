"""
Cross-platform notification utility for campaign completion.
Sends Windows toast notifications (local) AND ntfy.sh push notifications (mobile).

ntfy topic: cryoev-training-carney2026
"""

import subprocess
import urllib.request
import json
import platform
from datetime import datetime

NTFY_TOPIC = "cryoev-training-carney2026"

_SUMMARY_SCRIPT = """
param(
    [string]$Title = "Campaign Status",
    [string]$Message = "Task completed",
    [string]$Level = "info"
)
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
$Template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$TextNodes = $Template.GetElementsByTagName("text")
$TextNodes.Item(0).AppendChild($Template.CreateTextNode($Title)) > $null
$TextNodes.Item(1).AppendChild($Template.CreateTextNode($Message)) > $null
try {
    $AppId = "Windows.PowerShell.Shell"
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($AppId).Show($Template)
} catch {
    Write-Host "[NOTIFICATION: $Title - $Message]"
}
"""


def _send_ntfy(title: str, message: str, priority: int = 3):
    """Send push notification to phone via ntfy.sh."""
    try:
        data = json.dumps({
            "topic": NTFY_TOPIC,
            "title": title,
            "message": message,
            "priority": priority,
            "tags": ["computer", "chart_with_upwards_trend"] if "complete" in title.lower() or "error" in title.lower() else ["computer"],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://ntfy.sh",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[ntfy] Could not send push: {e}")


def send_notification(title: str, message: str, level: str = "info"):
    """Send local toast + remote push notification."""
    # Local toast
    try:
        ps_cmd = [
            "powershell.exe",
            "-NoProfile", "-NonInteractive", "-Command",
            _SUMMARY_SCRIPT,
            "-Title", title,
            "-Message", message,
            "-Level", level
        ]
        subprocess.run(ps_cmd, capture_output=True, timeout=10)
    except Exception as e:
        print(f"[toast] Error: {e}")

    # Remote push
    priority = 4 if level == "error" else 3
    _send_ntfy(title, message, priority=priority)


def notify_completion(total_runs: int, completed: int, failed: int, duration_hours: float):
    """Send completion notification with run summary."""
    now = datetime.now().strftime('%H:%M')
    title = f"Campaign Complete ({completed}/{total_runs} runs)"
    message = f"OK:{completed} Fail:{failed} Time:{duration_hours:.1f}h ended {now}"
    send_notification(title, message)


def notify_error(run_num: int, error_msg: str):
    """Send error notification for a failed run."""
    title = f"ERROR - Run {run_num}"
    message = f"Run {run_num} failed: {error_msg[:120]}"
    send_notification(title, message, level="error")


def notify_progress(run_num: int, total_runs: int, status: str):
    """Send progress notification at key milestones."""
    if status == "started":
        send_notification("Campaign Progress", f"Starting Run {run_num}/{total_runs}")
    elif status == "completed":
        send_notification("Campaign Progress", f"Run {run_num}/{total_runs} OK")


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 3:
        send_notification(sys.argv[1], sys.argv[2])
    else:
        send_notification("Test Notification", "Campaign notification system is working")