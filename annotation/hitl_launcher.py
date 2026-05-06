"""
HITL Annotation Launcher
========================
Simple GUI for launching the Human-in-the-Loop annotation workflow.
Run from the repo root:

    python -m annotation.hitl_launcher

or double-click this file.
"""

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# Default model weights (change if moved)
_DEFAULT_MODEL = (
    r"C:\Users\ML-2619\Desktop\Pujan Cryo\cryo-ev pipeline\Model Training by Yifei"
    r"\round_2\results_yolov8_heavy_augmentation\training\vesicle_instance_seg_v2\weights"
    r"\best.pt"
)

# Repo root (parent of this file's package)
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_python_executable() -> str:
    """Prefer the repo virtualenv interpreter when available."""
    candidate = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _make_log_path(output_dir: str) -> Path:
    log_dir = Path(output_dir) / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime

    return log_dir / f"annotator_launch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def _build_console_command(cmd: list[str], log_path: Path) -> list[str]:
    cmdline = subprocess.list2cmdline(cmd)
    escaped_log = str(log_path).replace("'", "''")
    ps_command = (
        "$ErrorActionPreference = 'Stop'; "
        f"$log = '{escaped_log}'; "
        f"Write-Host 'Starting annotator...' -ForegroundColor Cyan; "
        f"Write-Host 'Log file: ' $log -ForegroundColor Cyan; "
        f"& {cmdline} 2>&1 | Tee-Object -FilePath $log -Append; "
        "Write-Host ''; "
        "Write-Host 'Annotator process finished. Review the log above if needed.' -ForegroundColor Yellow"
    )
    return [
        "powershell.exe",
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        ps_command,
    ]


class HitlLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CryoEV – HITL Annotation Launcher")
        self.resizable(False, False)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # ── Header ────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg="#1e3a5f")
        hdr.grid(row=0, column=0, columnspan=3, sticky="ew")
        tk.Label(
            hdr,
            text="  CryoEV  •  HITL Annotation Launcher",
            font=("Helvetica", 13, "bold"),
            bg="#1e3a5f",
            fg="white",
            pady=8,
        ).pack(side="left")

        # ── Required paths ────────────────────────────────────────────
        req = ttk.LabelFrame(self, text=" Required ", padding=8)
        req.grid(row=1, column=0, columnspan=3, sticky="ew", padx=12, pady=(10, 4))
        req.columnconfigure(1, weight=1)

        self._output_var = tk.StringVar()
        self._model_var = tk.StringVar(value=_DEFAULT_MODEL)

        # ── Multi-directory input list ────────────────────────────────
        tk.Label(req, text="Input image folder(s) *", anchor="w").grid(
            row=0, column=0, sticky="nw", padx=(4, 8), pady=(4, 0))

        list_frame = tk.Frame(req)
        list_frame.grid(row=0, column=1, sticky="ew", pady=4)
        list_frame.columnconfigure(0, weight=1)

        self._input_listbox = tk.Listbox(list_frame, height=4, width=52,
                                          selectmode=tk.EXTENDED)
        self._input_listbox.grid(row=0, column=0, sticky="ew")
        sb = ttk.Scrollbar(list_frame, orient="vertical",
                           command=self._input_listbox.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._input_listbox.configure(yscrollcommand=sb.set)

        list_btn_frame = tk.Frame(req)
        list_btn_frame.grid(row=0, column=2, sticky="n", padx=(6, 0), pady=4)
        ttk.Button(list_btn_frame, text="Add folder…",
                   command=self._add_input_dir).pack(fill="x", pady=(0, 3))
        ttk.Button(list_btn_frame, text="Remove selected",
                   command=self._remove_input_dir).pack(fill="x")

        self._path_row(req, 1, "Output folder *",
                       self._output_var, self._browse_output)
        self._path_row(req, 2, "Model weights (.pt) *",
                       self._model_var, self._browse_model, file=True)

        # ── Settings ──────────────────────────────────────────────────
        opt = ttk.LabelFrame(self, text=" Settings ", padding=8)
        opt.grid(row=2, column=0, columnspan=3, sticky="ew", padx=12, pady=4)

        # Row 0 – review mode & device
        tk.Label(opt, text="Review mode").grid(row=0, column=0, sticky="w", **pad)
        self._mode_var = tk.StringVar(value="polygon")
        ttk.Combobox(opt, textvariable=self._mode_var, values=["polygon", "box"],
                     state="readonly", width=10).grid(row=0, column=1, sticky="w", **pad)

        tk.Label(opt, text="Device").grid(row=0, column=2, sticky="w", **pad)
        self._device_var = tk.StringVar(value="cuda")
        ttk.Combobox(opt, textvariable=self._device_var, values=["cuda", "cpu"],
                     state="readonly", width=8).grid(row=0, column=3, sticky="w", **pad)

        # Row 1 – grid layout
        tk.Label(opt, text="Region rows").grid(row=1, column=0, sticky="w", **pad)
        self._rows_var = tk.IntVar(value=2)
        ttk.Spinbox(opt, from_=1, to=8, textvariable=self._rows_var,
                    width=5).grid(row=1, column=1, sticky="w", **pad)

        tk.Label(opt, text="Region cols").grid(row=1, column=2, sticky="w", **pad)
        self._cols_var = tk.IntVar(value=4)
        ttk.Spinbox(opt, from_=1, to=12, textvariable=self._cols_var,
                    width=5).grid(row=1, column=3, sticky="w", **pad)

        # Row 2 – confidence & IoU
        tk.Label(opt, text="Confidence").grid(row=2, column=0, sticky="w", **pad)
        self._conf_var = tk.StringVar(value="0.25")
        ttk.Entry(opt, textvariable=self._conf_var, width=7).grid(
            row=2, column=1, sticky="w", **pad)

        tk.Label(opt, text="IoU (NMS)").grid(row=2, column=2, sticky="w", **pad)
        self._iou_var = tk.StringVar(value="0.7")
        ttk.Entry(opt, textvariable=self._iou_var, width=7).grid(
            row=2, column=3, sticky="w", **pad)

        # Row 3 – pixel size & save auto-labels
        tk.Label(opt, text="Pixel size (nm/px)").grid(row=3, column=0, sticky="w", **pad)
        self._px_var = tk.StringVar(value="")
        ttk.Entry(opt, textvariable=self._px_var,
                  width=10).grid(row=3, column=1, sticky="w", **pad)
        tk.Label(opt, text="leave blank if unknown",
                 font=("Helvetica", 8), fg="gray").grid(row=3, column=2, columnspan=2,
                                                         sticky="w", **pad)

        self._autolabel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Save raw model predictions alongside reviewed labels",
                        variable=self._autolabel_var).grid(
            row=4, column=0, columnspan=4, sticky="w", padx=10, pady=(2, 6))

        # ── Launch button ─────────────────────────────────────────────
        btn_frame = tk.Frame(self)
        btn_frame.grid(row=3, column=0, columnspan=3, pady=(6, 12))
        ttk.Button(btn_frame, text="  Launch Annotator  ",
                   command=self._launch).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Cancel",
                   command=self.destroy).pack(side="left", padx=6)

        # ── Status bar ────────────────────────────────────────────────
        self._status = tk.StringVar(value="Fill in the required fields and click Launch.")
        tk.Label(self, textvariable=self._status, fg="gray",
                 font=("Helvetica", 9)).grid(row=4, column=0, columnspan=3,
                                              sticky="w", padx=12, pady=(0, 8))

    def _path_row(self, parent, row, label, var, command, file=False):
        tk.Label(parent, text=label, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(4, 8), pady=4)
        ttk.Entry(parent, textvariable=var, width=52).grid(
            row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse…", command=command).grid(
            row=row, column=2, padx=(6, 0), pady=4)

    # ------------------------------------------------------------------
    # Browse callbacks
    # ------------------------------------------------------------------
    def _add_input_dir(self):
        # Show file contents so the user can confirm they are picking the right folder.
        files = filedialog.askopenfilenames(
            title="Select image(s) — the parent folder will be added as the input source",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if files:
            dirs = list(dict.fromkeys(str(Path(f).parent) for f in files))
        else:
            # Fallback: plain folder picker if user dismisses the file dialog
            d = filedialog.askdirectory(title="Select input image folder directly")
            if not d:
                return
            dirs = [d]

        existing = list(self._input_listbox.get(0, tk.END))
        for d in dirs:
            if d not in existing:
                self._input_listbox.insert(tk.END, d)
                existing.append(d)
        # Auto-suggest output from first added folder
        if not self._output_var.get() and self._input_listbox.size() >= 1:
            first = self._input_listbox.get(0)
            suggested = Path(first).parent / (Path(first).name + "_hitl_output")
            self._output_var.set(str(suggested))

    def _remove_input_dir(self):
        for idx in reversed(self._input_listbox.curselection()):
            self._input_listbox.delete(idx)

    def _browse_output(self):
        initial = self._output_var.get() or str(Path.home())
        d = filedialog.askdirectory(title="Select (or create) output folder",
                                    initialdir=str(Path(initial).parent
                                                   if Path(initial).parent.exists()
                                                   else Path.home()))
        if d:
            self._output_var.set(d)

    def _browse_model(self):
        f = filedialog.askopenfilename(
            title="Select model weights file",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
            initialfile=Path(self._model_var.get()).name
            if self._model_var.get() else "",
        )
        if f:
            self._model_var.set(f)

    # ------------------------------------------------------------------
    # Validation & launch
    # ------------------------------------------------------------------
    def _validate(self) -> bool:
        input_dirs = list(self._input_listbox.get(0, tk.END))
        if not input_dirs:
            messagebox.showerror("Missing field", "Please add at least one input image folder.")
            return False
        missing = [d for d in input_dirs if not Path(d).exists()]
        if missing:
            messagebox.showerror("Not found",
                                 "These folders do not exist:\n" + "\n".join(missing))
            return False
        if not self._output_var.get():
            messagebox.showerror("Missing field", "Please select an output folder.")
            return False
        if not self._model_var.get():
            messagebox.showerror("Missing field", "Please select model weights.")
            return False
        if not Path(self._model_var.get()).exists():
            messagebox.showerror("Not found", f"Model weights not found:\n{self._model_var.get()}")
            return False
        try:
            float(self._conf_var.get())
            float(self._iou_var.get())
        except ValueError:
            messagebox.showerror("Invalid value", "Confidence and IoU must be numbers (e.g. 0.25).")
            return False
        if self._px_var.get():
            try:
                float(self._px_var.get())
            except ValueError:
                messagebox.showerror("Invalid value", "Pixel size must be a number or left blank.")
                return False
        return True

    def _launch(self):
        if not self._validate():
            return

        input_dirs = list(self._input_listbox.get(0, tk.END))
        python_executable = _resolve_python_executable()
        annotator_cmd = [
            python_executable, "-m", "annotation.hitl_annotator",
            "--ui", "simple",
            "--review-mode", self._mode_var.get(),
            "--region-rows", str(self._rows_var.get()),
            "--region-cols", str(self._cols_var.get()),
            "--input-images", *input_dirs,
            "--output-dir", self._output_var.get(),
            "--model-path", self._model_var.get(),
            "--device", self._device_var.get(),
            "--conf", self._conf_var.get(),
            "--iou", self._iou_var.get(),
        ]
        if self._px_var.get():
            annotator_cmd += ["--pixel-size", self._px_var.get()]
        if self._autolabel_var.get():
            annotator_cmd.append("--save-auto-labels")

        log_path = _make_log_path(self._output_var.get())
        console_cmd = _build_console_command(annotator_cmd, log_path)

        self._status.set("Launching annotator in a console window…")
        self.update()

        try:
            process = subprocess.Popen(
                console_cmd,
                cwd=str(_REPO_ROOT),
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
        except OSError as exc:
            self._status.set("Launch failed. See error dialog.")
            messagebox.showerror(
                "Launch failed",
                f"Could not start the annotator.\n\nPython: {python_executable}\nLog: {log_path}\n\n{exc}",
            )
            return
        messagebox.showinfo(
            "Annotator launched",
            f"Started annotator console window.\n\nPID: {process.pid}\nLog file: {log_path}",
        )
        self.after(800, self.destroy)


if __name__ == "__main__":
    app = HitlLauncher()
    app.mainloop()
