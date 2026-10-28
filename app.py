"""ClearMark - remove logos, add your own watermark, make thumbnails. Runs locally."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk

    from vwstudio.ai_fill import download_model, model_available
    from vwstudio.logo_removal import METHODS, LogoRegion, detect_logos, refine_region
    from vwstudio.media import Cancelled, FrameReader, VideoInfo, find_ffmpeg, probe, sample_frames
    from vwstudio.pipeline import Processor, export_video, render_thumbnail
    from vwstudio.settings import QUALITY, load_settings, save_settings
    from vwstudio.thumbnail import SIZES, ThumbnailSettings, best_frame_time
    from vwstudio.watermark import POSITIONS, WatermarkSettings, list_fonts
    IMPORT_ERROR = None
except ImportError as _e:  # shown in a dialog by main()
    IMPORT_ERROR = _e

VIDEO_TYPES = [("Videos", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.3gp *.wmv"), ("All files", "*.*")]
IMAGE_TYPES = [("Images", "*.png *.webp *.jpg *.jpeg"), ("All files", "*.*")]
FONT_TYPES = [("Fonts", "*.ttf *.otf *.ttc"), ("All files", "*.*")]
REMOVE_COLOR = "#ff4d4d"
SELECT_COLOR = "#ffd400"


def fmt_time(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:05.2f}"


class App(tk.Tk):
    def __init__(self, initial_video: str | None = None):
        super().__init__()
        self.title("ClearMark")
        self.ui_scale = max(1.0, self.winfo_fpixels("1i") / 96.0)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(self.px(1320), sw - self.px(40)), min(self.px(860), sh - self.px(110))
        self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2 - self.px(30))}")
        self.minsize(min(self.px(1050), w), min(self.px(640), h))

        self.settings = load_settings()
        self.info: VideoInfo | None = None
        self.reader: FrameReader | None = None
        self.frame = None                 # source frame (BGR) at self.t
        self.t = 0.0
        self.regions: list[LogoRegion] = []
        self.selected: int | None = None
        self.samples = None               # (times, frames) cached for the current video
        self.thumb_time: float | None = None
        self.best_time: float | None = None
        self.last_output: str | None = None

        self.jobs: queue.Queue = queue.Queue()
        self.busy = False
        self.cancel_evt: threading.Event | None = None
        self._processor = None
        self._fast_processor = None
        self._version = 0                 # bumps whenever removal/watermark settings change
        self._smart_cache: dict = {}      # (frame index, version) -> best-quality preview frame
        self._smart_lock = threading.Lock()
        self._smart_running = False
        self._smart_pending = None
        self._export_started = 0.0
        self._photo = None
        self._seek_job = None
        self._refresh_job = None
        self._view = (1.0, 0, 0)
        self._drag = None
        self._rubber = None

        self._build_style()
        self._build_ui()
        self._load_settings_into_ui()
        self._update_ai_status()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind("<Left>", lambda e: self._key_step(e, -1))
        self.bind("<Right>", lambda e: self._key_step(e, 1))
        self.bind("<Delete>", self._key_delete)
        self.after(50, self._poll_jobs)
        if initial_video:
            self.after(150, lambda: self.open_video(initial_video))

    # ------------------------------------------------------------------ UI construction
    def px(self, n: float) -> int:
        """Scale a 96-dpi pixel size for the current screen."""
        return int(n * self.ui_scale)

    def _build_style(self):
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 6))
        style.configure("Hint.TLabel", foreground="#666666")
        style.configure("Head.TLabel", font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        top = ttk.Frame(self, padding=(10, 8))
        top.pack(fill="x")
        ttk.Button(top, text="Open video...", style="Primary.TButton", command=self.choose_video).pack(side="left")
        self.file_lbl = ttk.Label(top, text="No video loaded - open one to start.", style="Hint.TLabel")
        self.file_lbl.pack(side="left", padx=12)

        self.status = ttk.Label(self, text="Ready", anchor="w", padding=(10, 4), relief="sunken")
        self.status.pack(side="bottom", fill="x")

        body = ttk.Frame(self, padding=(10, 0, 10, 6))
        body.pack(fill="both", expand=True)

        right = ttk.Frame(body, width=self.px(430))
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(left, bg="#16181d", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.schedule_refresh(30))
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        ctr = ttk.Frame(left, padding=(0, 6, 0, 0))
        ctr.pack(fill="x")
        ttk.Button(ctr, text="<", width=3, command=lambda: self.step(-1)).pack(side="left")
        ttk.Button(ctr, text=">", width=3, command=lambda: self.step(1)).pack(side="left", padx=(2, 8))
        self.pos = tk.DoubleVar(value=0.0)
        self.slider = ttk.Scale(ctr, from_=0, to=1, variable=self.pos, command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True)
        self.time_lbl = ttk.Label(ctr, text="00:00.00 / 00:00.00", width=20, anchor="e")
        self.time_lbl.pack(side="left", padx=(8, 0))

        opts = ttk.Frame(left, padding=(0, 4, 0, 0))
        opts.pack(fill="x")
        self.show_result = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Show result", variable=self.show_result,
                        command=self.schedule_refresh).pack(side="left")
        self.show_boxes = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Highlight logo areas", variable=self.show_boxes,
                        command=self.schedule_refresh).pack(side="left", padx=14)
        ttk.Label(opts, text="Tip: drag on the picture to mark a logo.  Left/Right keys step frames.",
                  style="Hint.TLabel").pack(side="right")

        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)
        self._build_remove_tab(nb)
        self._build_watermark_tab(nb)
        self._build_thumbnail_tab(nb)
        self._build_export_tab(nb)

    def _scale_row(self, parent, label, var, lo, hi, fmt="{:.0f}", command=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=15).pack(side="left")
        value = ttk.Label(row, width=6, anchor="e")
        value.pack(side="right")

        def changed(*_):
            value.config(text=fmt.format(var.get()))
            if command:
                command()
        ttk.Scale(row, from_=lo, to=hi, variable=var, command=lambda _v: changed()).pack(
            side="left", fill="x", expand=True, padx=4)
        value.config(text=fmt.format(var.get()))
        var.trace_add("write", lambda *_: value.config(text=fmt.format(var.get())))
        return row

    def _combo_row(self, parent, label, var, values, width=22, readonly=True):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=15).pack(side="left")
        cb = ttk.Combobox(row, textvariable=var, values=values, width=width,
                          state="readonly" if readonly else "normal")
        cb.pack(side="left", fill="x", expand=True, padx=4)
        return row, cb

    def _color_row(self, parent, label, var):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=15).pack(side="left")
        swatch = tk.Label(row, width=4, relief="groove", bg=var.get(), cursor="hand2")
        swatch.pack(side="left", padx=4)

        def pick(_e=None):
            rgb, hexval = colorchooser.askcolor(color=var.get(), parent=self, title=label)
            if hexval:
                var.set(hexval)
        swatch.bind("<Button-1>", pick)
        ttk.Button(row, text="Choose...", command=pick).pack(side="left")
        var.trace_add("write", lambda *_: swatch.config(bg=var.get()))
        return row

    def _build_remove_tab(self, nb):
        f = ttk.Frame(nb, padding=12)
        nb.add(f, text=" 1 Remove logo ")
        ttk.Label(f, text="Mark the logo or watermark you want gone: drag a box around it on the "
                          "picture, or let the app look for it.", wraplength=self.px(390)).pack(anchor="w")
        ttk.Button(f, text="Auto-detect logos", style="Primary.TButton",
                   command=self.auto_detect).pack(fill="x", pady=(10, 4))
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Button(row, text="Reuse saved boxes", command=self.reuse_regions).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Save boxes for next time", command=self.save_regions).pack(
            side="left", fill="x", expand=True, padx=(4, 0))

        ttk.Label(f, text="Logo areas", style="Head.TLabel").pack(anchor="w", pady=(14, 2))
        self.tree = ttk.Treeview(f, columns=("box", "when", "shape"), show="headings", height=6,
                                 selectmode="browse")
        for col, text, w in (("box", "Box (x, y, w x h)", 150), ("when", "When", 110), ("shape", "Shape", 90)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=w, anchor="w")
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Delete selected", command=self.delete_selected).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Clear all", command=self.clear_regions).pack(side="left", fill="x", expand=True, padx=(4, 0))

        tf = ttk.LabelFrame(f, text="Selected box is active...", padding=8)
        tf.pack(fill="x", pady=(12, 0))
        ttk.Label(tf, text="For logos that only appear part of the time or jump around, set "
                           "when each box applies using the timeline.", wraplength=self.px(370),
                  style="Hint.TLabel").pack(anchor="w")
        row = ttk.Frame(tf)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="From here", command=lambda: self.set_region_time("start")).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Until here", command=lambda: self.set_region_time("end")).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row, text="Whole video", command=lambda: self.set_region_time("all")).pack(side="left", fill="x", expand=True)

        sf = ttk.LabelFrame(f, text="How to remove", padding=8)
        sf.pack(fill="x", pady=(12, 0))
        self.method_var = tk.StringVar()
        _, cb = self._combo_row(sf, "Method", self.method_var, list(METHODS.values()))
        cb.bind("<<ComboboxSelected>>", lambda e: self._removal_changed())
        ttk.Label(sf, text="Best copies the real background from moments where the camera "
                           "uncovers it and generates the rest with AI. Slower than Fast fill.",
                  wraplength=self.px(370), style="Hint.TLabel").pack(anchor="w", pady=(2, 4))
        row = ttk.Frame(sf)
        row.pack(fill="x", pady=(0, 4))
        self.ai_lbl = ttk.Label(row, text="")
        self.ai_lbl.pack(side="left")
        self.ai_btn = ttk.Button(row, text="Download AI model (208 MB)", command=self.download_ai_model)
        self.expand_var = tk.IntVar(value=3)
        self._scale_row(sf, "Grow area (px)", self.expand_var, 0, 15, command=self._removal_changed)
        self.auto_shape_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sf, text="Find the exact logo shape inside boxes I draw",
                        variable=self.auto_shape_var).pack(anchor="w", pady=(4, 0))

    def _build_watermark_tab(self, nb):
        f = ttk.Frame(nb, padding=12)
        nb.add(f, text=" 2 My watermark ")
        self.wm_vars = {
            "text": tk.StringVar(), "font": tk.StringVar(), "text_size": tk.DoubleVar(),
            "text_color": tk.StringVar(value="#FFFFFF"), "text_opacity": tk.DoubleVar(),
            "text_outline": tk.BooleanVar(), "text_position": tk.StringVar(),
            "logo_path": tk.StringVar(), "logo_size": tk.DoubleVar(), "logo_opacity": tk.DoubleVar(),
            "logo_position": tk.StringVar(), "margin": tk.DoubleVar(),
        }
        v = self.wm_vars
        tf = ttk.LabelFrame(f, text="Text", padding=8)
        tf.pack(fill="x")
        row = ttk.Frame(tf)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Text", width=15).pack(side="left")
        ttk.Entry(row, textvariable=v["text"]).pack(side="left", fill="x", expand=True, padx=4)
        row, self.font_combo = self._combo_row(tf, "Font", v["font"], [], readonly=False)
        ttk.Button(row, text="...", width=3, command=lambda: self._browse_font(v["font"])).pack(side="left")
        self._scale_row(tf, "Size %", v["text_size"], 1.5, 15, "{:.1f}")
        self._color_row(tf, "Color", v["text_color"])
        self._scale_row(tf, "Opacity", v["text_opacity"], 0.1, 1.0, "{:.0%}")
        ttk.Checkbutton(tf, text="Dark outline (readable on any background)",
                        variable=v["text_outline"]).pack(anchor="w", pady=2)
        self._combo_row(tf, "Position", v["text_position"], POSITIONS)

        lf = ttk.LabelFrame(f, text="Logo image (PNG with transparency works best)", padding=8)
        lf.pack(fill="x", pady=(12, 0))
        row = ttk.Frame(lf)
        row.pack(fill="x", pady=2)
        ttk.Entry(row, textvariable=v["logo_path"]).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse...", command=self._browse_logo).pack(side="left", padx=(4, 0))
        ttk.Button(row, text="x", width=3, command=lambda: v["logo_path"].set("")).pack(side="left", padx=(4, 0))
        self._scale_row(lf, "Size %", v["logo_size"], 3, 60)
        self._scale_row(lf, "Opacity", v["logo_opacity"], 0.1, 1.0, "{:.0%}")
        self._combo_row(lf, "Position", v["logo_position"], POSITIONS)

        self._scale_row(f, "Edge margin %", v["margin"], 0, 15, "{:.1f}")
        ttk.Label(f, text="Your watermark settings are remembered for next time. Tick "
                          "'Show result' under the picture to see them.", wraplength=self.px(390),
                  style="Hint.TLabel").pack(anchor="w", pady=(10, 0))
        for var in v.values():
            var.trace_add("write", lambda *_: self._watermark_changed())

    def _build_thumbnail_tab(self, nb):
        f = ttk.Frame(nb, padding=12)
        nb.add(f, text=" 3 Thumbnail ")
        self.th_vars = {
            "font": tk.StringVar(), "title_size": tk.DoubleVar(), "title_color": tk.StringVar(value="#FFFFFF"),
            "outline_color": tk.StringVar(value="#000000"), "title_position": tk.StringVar(),
            "title_band": tk.BooleanVar(), "size": tk.StringVar(), "fit": tk.StringVar(),
            "enhance": tk.BooleanVar(), "include_watermark": tk.BooleanVar(),
        }
        v = self.th_vars
        ff = ttk.LabelFrame(f, text="Frame", padding=8)
        ff.pack(fill="x")
        self.thumb_lbl = ttk.Label(ff, text="Using: best frame (picked automatically)")
        self.thumb_lbl.pack(anchor="w")
        row = ttk.Frame(ff)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="Use current frame", command=self.use_current_frame).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Find best frame", command=self.find_best_frame).pack(side="left", fill="x", expand=True, padx=(4, 0))

        tf = ttk.LabelFrame(f, text="Title", padding=8)
        tf.pack(fill="x", pady=(12, 0))
        self.title_text = tk.Text(tf, height=3, wrap="word", font=("Segoe UI", 10))
        self.title_text.pack(fill="x", pady=(0, 4))
        row, self.thumb_font_combo = self._combo_row(tf, "Font", v["font"], [], readonly=False)
        ttk.Button(row, text="...", width=3, command=lambda: self._browse_font(v["font"])).pack(side="left")
        self._scale_row(tf, "Size %", v["title_size"], 5, 25, "{:.1f}")
        self._color_row(tf, "Text color", v["title_color"])
        self._color_row(tf, "Outline", v["outline_color"])
        self._combo_row(tf, "Position", v["title_position"], ["top", "center", "bottom"])
        ttk.Checkbutton(tf, text="Dark band behind the title", variable=v["title_band"]).pack(anchor="w", pady=2)

        of = ttk.LabelFrame(f, text="Output", padding=8)
        of.pack(fill="x", pady=(12, 0))
        self._combo_row(of, "Size", v["size"], list(SIZES))
        row = ttk.Frame(of)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="If shapes differ", width=15).pack(side="left")
        ttk.Radiobutton(row, text="Blurred backdrop", value="blur", variable=v["fit"]).pack(side="left")
        ttk.Radiobutton(row, text="Crop to fill", value="crop", variable=v["fit"]).pack(side="left", padx=6)
        ttk.Checkbutton(of, text="Boost colors and sharpness", variable=v["enhance"]).pack(anchor="w")
        ttk.Checkbutton(of, text="Include my watermark", variable=v["include_watermark"]).pack(anchor="w")

        row = ttk.Frame(f)
        row.pack(fill="x", pady=(12, 0))
        ttk.Button(row, text="Preview", command=self.preview_thumbnail).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Save thumbnail...", style="Primary.TButton",
                   command=self.save_thumbnail).pack(side="left", fill="x", expand=True, padx=(4, 0))

    def _build_export_tab(self, nb):
        f = ttk.Frame(nb, padding=12)
        nb.add(f, text=" 4 Export ")
        ttk.Label(f, text="Save as", style="Head.TLabel").pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(2, 8))
        self.out_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.out_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="...", width=3, command=self._browse_output).pack(side="left", padx=(4, 0))

        self.ex_vars = {"quality": tk.StringVar(), "preset": tk.StringVar(), "keep_audio": tk.BooleanVar(),
                        "embed_cover": tk.BooleanVar(), "save_thumbnail": tk.BooleanVar()}
        v = self.ex_vars
        self._combo_row(f, "Quality", v["quality"], list(QUALITY))
        self._combo_row(f, "Encoding", v["preset"], ["veryfast", "fast", "medium", "slow"])
        ttk.Label(f, text="Slower encoding = smaller file at the same quality.",
                  style="Hint.TLabel").pack(anchor="w")
        ttk.Checkbutton(f, text="Keep the original audio", variable=v["keep_audio"]).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(f, text="Embed thumbnail as the video's cover picture",
                        variable=v["embed_cover"]).pack(anchor="w")
        ttk.Checkbutton(f, text="Also save the thumbnail as a .jpg next to the video",
                        variable=v["save_thumbnail"]).pack(anchor="w")

        ttk.Label(f, text="What will happen", style="Head.TLabel").pack(anchor="w", pady=(14, 2))
        self.summary_lbl = ttk.Label(f, text="", wraplength=self.px(390), justify="left")
        self.summary_lbl.pack(anchor="w")

        self.export_btn = ttk.Button(f, text="Export video", style="Primary.TButton", command=self.export)
        self.export_btn.pack(fill="x", pady=(16, 4))
        self.progress = ttk.Progressbar(f, maximum=1000)
        self.progress.pack(fill="x")
        self.progress_lbl = ttk.Label(f, text="", style="Hint.TLabel")
        self.progress_lbl.pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(6, 0))
        self.cancel_btn = ttk.Button(row, text="Cancel", command=self.cancel_job, state="disabled")
        self.cancel_btn.pack(side="left", fill="x", expand=True)
        self.open_btn = ttk.Button(row, text="Show in folder", command=self.open_output_folder, state="disabled")
        self.open_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))
        nb.bind("<<NotebookTabChanged>>", lambda e: self._update_summary())

    # ------------------------------------------------------------------ settings <-> UI
    def _load_settings_into_ui(self):
        s = self.settings
        fonts = list(list_fonts())
        self.font_combo["values"] = fonts
        self.thumb_font_combo["values"] = fonts
        self._loading = True
        for key, var in self.wm_vars.items():
            var.set(getattr(s.watermark, key))
        for key, var in self.th_vars.items():
            var.set(getattr(s.thumbnail, key))
        self.title_text.insert("1.0", s.thumbnail.title)
        for key, var in self.ex_vars.items():
            var.set(getattr(s.export, key))
        self.method_var.set(METHODS.get(s.removal_method, METHODS["smart"]))
        self.expand_var.set(s.removal_expand)
        self.auto_shape_var.set(s.auto_shape)
        self._loading = False

    def _collect_settings(self):
        s = self.settings
        s.watermark = self._watermark_settings()
        s.thumbnail = self._thumbnail_settings()
        for key, var in self.ex_vars.items():
            setattr(s.export, key, var.get())
        s.removal_method = self._method_key()
        s.removal_expand = int(self.expand_var.get())
        s.auto_shape = bool(self.auto_shape_var.get())
        return s

    def _watermark_settings(self) -> WatermarkSettings:
        vals = {}
        for key, var in self.wm_vars.items():
            try:
                vals[key] = var.get()
            except tk.TclError:
                vals[key] = getattr(WatermarkSettings(), key)
        return WatermarkSettings(**vals)

    def _thumbnail_settings(self) -> ThumbnailSettings:
        vals = {key: var.get() for key, var in self.th_vars.items()}
        vals["title"] = self.title_text.get("1.0", "end").strip()
        return ThumbnailSettings(**vals)

    def _method_key(self) -> str:
        label = self.method_var.get()
        return next((k for k, v in METHODS.items() if v == label), "smart")

    def save_all_settings(self):
        try:
            save_settings(self._collect_settings())
        except OSError as e:
            self.set_status(f"Could not save settings: {e}")

    # ------------------------------------------------------------------ helpers
    def set_status(self, text: str):
        self.status.config(text=text)

    def _invalidate(self):
        self._processor = self._fast_processor = None
        self._version += 1
        self.schedule_refresh()
        self._update_summary()

    def _update_ai_status(self):
        if model_available():
            self.ai_lbl.config(text="AI fill: installed", style="TLabel")
            self.ai_btn.pack_forget()
        else:
            self.ai_lbl.config(text="AI fill not installed (simpler fill used)", style="Hint.TLabel")
            self.ai_btn.pack(side="right")

    def download_ai_model(self):
        def progress(done, total):
            self.jobs.put(("status", f"Downloading AI model... {done / 1e6:.0f} of {total / 1e6:.0f} MB "
                                     "(Cancel is on the Export tab)"))

        def done(_path):
            self._update_ai_status()
            self._invalidate()
            self.set_status("AI model installed. 'Best' removal now generates hidden background with AI.")
        self.run_bg("Downloading AI model", lambda: download_model(progress, self.cancel_evt), done,
                    cancellable=True)

    def _removal_changed(self):
        self._invalidate()

    def _watermark_changed(self):
        if getattr(self, "_loading", False):
            return
        if self.info is not None and not self.show_result.get():
            self.show_result.set(True)
        self._invalidate()

    def processor(self) -> Processor:
        if self._processor is None and self.info is not None:
            self._processor = self.make_processor()
        return self._processor

    def make_processor(self, method: str | None = None) -> Processor:
        return Processor(self.info, list(self.regions), method or self._method_key(),
                         int(self.expand_var.get()), self._watermark_settings())

    def fast_processor(self) -> Processor:
        if self._fast_processor is None and self.info is not None:
            self._fast_processor = self.make_processor("inpaint")
        return self._fast_processor

    def _request_smart_preview(self, key, processor: Processor, frame, t: float):
        """Build the best-quality preview in the background; newer requests replace older ones."""
        with self._smart_lock:
            self._smart_pending = (key, processor, frame, t)
            if self._smart_running:
                return
            self._smart_running = True

        def worker():
            while True:
                with self._smart_lock:
                    req, self._smart_pending = self._smart_pending, None
                    if req is None:
                        self._smart_running = False
                        return
                key, proc, frame, t = req
                try:
                    self.jobs.put(("preview", key, proc.process(frame, t)))
                except Exception as e:
                    self.jobs.put(("status", f"Preview error: {e}"))
        threading.Thread(target=worker, daemon=True).start()

    def _browse_font(self, var):
        path = filedialog.askopenfilename(parent=self, title="Choose a font file", filetypes=FONT_TYPES)
        if path:
            var.set(path)

    def _browse_logo(self):
        path = filedialog.askopenfilename(parent=self, title="Choose your logo image", filetypes=IMAGE_TYPES)
        if path:
            self.wm_vars["logo_path"].set(path)

    def _browse_output(self):
        if not self.info:
            return
        cur = self.out_var.get()
        path = filedialog.asksaveasfilename(
            parent=self, title="Save edited video as", defaultextension=".mp4",
            initialdir=os.path.dirname(cur) or None, initialfile=os.path.basename(cur),
            filetypes=[("MP4 video", "*.mp4"), ("QuickTime", "*.mov"), ("Matroska", "*.mkv")])
        if path:
            self.out_var.set(path)

    # ------------------------------------------------------------------ background jobs
    def run_bg(self, label: str, fn, on_done=None, cancellable=False):
        if self.busy:
            messagebox.showinfo("Please wait", "Another task is still running.", parent=self)
            return
        self.busy = True
        self.config(cursor="watch")
        self.set_status(label + "...")
        self.cancel_evt = threading.Event() if cancellable else None
        self.cancel_btn.config(state="normal" if cancellable else "disabled")

        def worker():
            try:
                self.jobs.put(("done", on_done, fn()))
            except Cancelled:
                self.jobs.put(("cancelled",))
            except Exception as e:  # report to the UI thread
                self.jobs.put(("error", e, traceback.format_exc()))
        threading.Thread(target=worker, daemon=True).start()

    def _poll_jobs(self):
        try:
            while True:
                item = self.jobs.get_nowait()
                kind = item[0]
                if kind == "progress":
                    done, total = item[1], max(1, item[2])
                    self.progress["value"] = 1000 * min(1.0, done / total)
                    text = f"Frame {done} of {total}"
                    elapsed = time.monotonic() - self._export_started
                    if done > 8 and elapsed > 3:
                        left = elapsed * (total - done) / done
                        text += f"  -  about {int(left // 60)} min {int(left % 60)} s left" if left >= 60 \
                            else f"  -  about {int(left)} s left"
                    self.progress_lbl.config(text=text)
                    continue
                if kind == "status":
                    self.set_status(item[1])
                    continue
                if kind == "preview":
                    key, img = item[1], item[2]
                    self._smart_cache[key] = img
                    while len(self._smart_cache) > 24:
                        self._smart_cache.pop(next(iter(self._smart_cache)))
                    if key == self._preview_key():
                        self.schedule_refresh(0)
                        if not self.busy:
                            self.set_status("Showing the best-quality result for this frame.")
                    continue
                self.busy = False
                self.config(cursor="")
                self.cancel_btn.config(state="disabled")
                if kind == "done":
                    if item[1]:
                        item[1](item[2])
                elif kind == "cancelled":
                    self.set_status("Cancelled.")
                    self.progress["value"] = 0
                    self.progress_lbl.config(text="Cancelled")
                elif kind == "error":
                    print(item[2], file=sys.stderr)
                    self.set_status(f"Error: {item[1]}")
                    messagebox.showerror("Something went wrong", str(item[1]), parent=self)
        except queue.Empty:
            pass
        self.after(50, self._poll_jobs)

    def cancel_job(self):
        if self.cancel_evt:
            self.cancel_evt.set()
            self.set_status("Cancelling...")

    def _get_samples(self, start=None, end=None):
        """Frames used for logo analysis. Runs in a worker thread."""
        info = self.info
        if self.samples is None:
            self.samples = sample_frames(info, 32)
        times, frames = self.samples
        if start is None and end is None:
            return frames
        picked = [f for t, f in zip(times, frames)
                  if (start is None or t >= start) and (end is None or t <= end)]
        if len(picked) >= 8:
            return picked
        return sample_frames(info, 16, start, end)[1]

    # ------------------------------------------------------------------ video loading / preview
    def choose_video(self):
        path = filedialog.askopenfilename(parent=self, title="Open a video", filetypes=VIDEO_TYPES)
        if path:
            self.open_video(path)

    def open_video(self, path: str):
        if self.busy:
            messagebox.showinfo("Please wait", "Another task is still running.", parent=self)
            return
        try:
            info = probe(path)
        except Exception as e:
            messagebox.showerror("Can't open video", str(e), parent=self)
            return
        if self.reader:
            self.reader.close()
        self.info, self.reader = info, FrameReader(info)
        self.regions, self.selected, self.samples = [], None, None
        self.thumb_time = self.best_time = None
        self.thumb_lbl.config(text="Using: best frame (picked automatically)")
        self.last_output = None
        self.open_btn.config(state="disabled")
        self.progress["value"] = 0
        self.progress_lbl.config(text="")
        self.slider.config(to=max(0.01, info.duration))
        stem, _ = os.path.splitext(path)
        self.out_var.set(stem + "_edited.mp4")
        audio = "with audio" if info.has_audio else "no audio"
        self.file_lbl.config(text=f"{os.path.basename(path)}   |   {info.width}x{info.height}, "
                                  f"{info.fps:.3g} fps, {fmt_time(info.duration)}, {audio}")
        self.show_result.set(False)
        self._refresh_tree()
        self._processor = self._fast_processor = None
        self._version += 1
        self._smart_cache.clear()
        self.seek(0.0)
        key = f"{info.width}x{info.height}"
        if self.settings.saved_regions.get(key):
            self.set_status(f"Loaded. You have saved logo boxes for {key} videos - "
                            "click 'Reuse saved boxes' to apply them.")
        else:
            self.set_status("Loaded. Drag a box around the logo, or click 'Auto-detect logos'.")

    def _on_slider(self, _value):
        if not self.info:
            return
        t = float(self.pos.get())
        self.time_lbl.config(text=f"{fmt_time(t)} / {fmt_time(self.info.duration)}")
        if self._seek_job:
            self.after_cancel(self._seek_job)
        self._seek_job = self.after(40, lambda: self.seek(t, move_slider=False))

    def seek(self, t: float, move_slider=True):
        if not self.info:
            return
        self._seek_job = None
        t = max(0.0, min(self.info.duration, t))
        frame = self.reader.frame_at(t)
        if frame is None:
            return
        self.t, self.frame = t, frame
        if move_slider:
            self.pos.set(t)
        self.time_lbl.config(text=f"{fmt_time(t)} / {fmt_time(self.info.duration)}")
        self.refresh_preview()

    def step(self, frames: int):
        if self.info:
            self.seek(self.t + frames / self.info.fps)

    def _key_step(self, event, frames):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text, ttk.Combobox, ttk.Scale, ttk.Treeview)):
            return
        self.step(frames)

    def _key_delete(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text, ttk.Combobox)):
            return
        self.delete_selected()

    def schedule_refresh(self, delay=60):
        if self._refresh_job:
            self.after_cancel(self._refresh_job)
        self._refresh_job = self.after(delay, self.refresh_preview)

    def refresh_preview(self):
        self._refresh_job = None
        c = self.canvas
        cw, ch = max(10, c.winfo_width()), max(10, c.winfo_height())
        c.delete("all")
        if self.frame is None:
            c.create_text(cw // 2, ch // 2, fill="#8a8f98", font=("Segoe UI", 14),
                          text="Open a video to get started\n\n(File types: mp4, mov, mkv, avi, webm...)",
                          justify="center")
            return
        img = self.frame.copy()
        showing_result = self.show_result.get()
        label = "RESULT" if showing_result else "ORIGINAL"
        if showing_result:
            try:
                proc = self.processor()
                if proc.smart:
                    key = self._preview_key()
                    if key in self._smart_cache:
                        img = self._smart_cache[key].copy()
                        label = "RESULT (best)"
                    else:
                        img = self.fast_processor().process(img, self.t)
                        label = "QUICK PREVIEW - building best result..."
                        self._request_smart_preview(key, proc, self.frame.copy(), self.t)
                else:
                    img = proc.process(img, self.t)
            except Exception as e:  # keep the preview alive on bad settings
                self.set_status(f"Preview error: {e}")
        elif self.show_boxes.get():
            self._tint_regions(img)
        h, w = img.shape[:2]
        scale = min(cw / w, ch / h)
        dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize(
            (dw, dh), Image.LANCZOS if scale < 1 else Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(pil)
        c.create_image(ox, oy, image=self._photo, anchor="nw")
        self._view = (scale, ox, oy)
        if self.show_boxes.get() and not showing_result:
            for i, r in enumerate(self.regions):
                color = SELECT_COLOR if i == self.selected else REMOVE_COLOR
                x0, y0 = ox + r.x * scale, oy + r.y * scale
                x1, y1 = ox + (r.x + r.w) * scale, oy + (r.y + r.h) * scale
                dash = () if r.active(self.t) else (3, 3)
                c.create_rectangle(x0, y0, x1, y1, outline=color, width=2, dash=dash)
                c.create_text(x0 + 3, y0 - 2, text=str(i + 1), anchor="sw", fill=color,
                              font=("Segoe UI", 10, "bold"))
        c.create_text(ox + 8, oy + 8, text=label, anchor="nw", fill="#ffffff", font=("Segoe UI", 9, "bold"))

    def _preview_key(self):
        return (int(round(self.t * self.info.fps)) if self.info else -1, self._version)

    def _tint_regions(self, img):
        h, w = img.shape[:2]
        for r in self.regions:
            rr = r.clamp(w, h)
            roi = img[rr.y:rr.y + rr.h, rr.x:rr.x + rr.w]
            m = np.full(roi.shape[:2], 255, np.uint8) if rr.mask is None else rr.mask
            sel = m > 0
            roi[sel] = (roi[sel] * 0.5 + np.array([60, 60, 255]) * 0.5).astype(np.uint8)

    # ------------------------------------------------------------------ drawing regions
    def _to_frame(self, cx, cy):
        scale, ox, oy = self._view
        return (cx - ox) / scale, (cy - oy) / scale

    def _on_press(self, e):
        if self.frame is None:
            return
        fx, fy = self._to_frame(e.x, e.y)
        hits = [i for i, r in enumerate(self.regions) if r.x <= fx <= r.x + r.w and r.y <= fy <= r.y + r.h]
        self._drag = (e.x, e.y, hits[-1] if hits else None)
        self._rubber = None

    def _on_motion(self, e):
        if not self._drag:
            return
        x0, y0, _ = self._drag
        if self._rubber is None:
            if abs(e.x - x0) + abs(e.y - y0) < 5:
                return
            self._rubber = self.canvas.create_rectangle(x0, y0, e.x, e.y, outline=SELECT_COLOR, width=2, dash=(5, 3))
        self.canvas.coords(self._rubber, x0, y0, e.x, e.y)

    def _on_release(self, e):
        if not self._drag:
            return
        x0, y0, hit = self._drag
        self._drag = None
        if self._rubber is None:
            if hit is not None:
                self.select_region(hit)
            return
        self.canvas.delete(self._rubber)
        self._rubber = None
        ax, ay = self._to_frame(x0, y0)
        bx, by = self._to_frame(e.x, e.y)
        w, h = self.info.width, self.info.height
        left, right = sorted((max(0, min(w, ax)), max(0, min(w, bx))))
        top, bottom = sorted((max(0, min(h, ay)), max(0, min(h, by))))
        if right - left < 6 or bottom - top < 6:
            return
        self.add_region(LogoRegion(int(left), int(top), int(right - left), int(bottom - top)))

    # ------------------------------------------------------------------ region management
    def add_region(self, region: LogoRegion):
        self.regions.append(region)
        idx = len(self.regions) - 1
        self.selected = idx
        self._refresh_tree()
        self._invalidate()
        if not self.auto_shape_var.get():
            self.set_status("Box added - the whole box will be filled in.")
            return

        def work():
            return refine_region(self._get_samples(region.start, region.end), region)

        def done(refined: LogoRegion):
            if idx < len(self.regions) and self.regions[idx] is region:
                self.regions[idx] = refined
                self._refresh_tree()
                self._invalidate()
            if refined.mask is None:
                self.set_status("Box added. No distinct logo shape found inside, so the whole box will be filled.")
            else:
                self.set_status("Box added. The logo's shape is highlighted in red - tick 'Show result' to check it.")
        self.run_bg("Finding the logo's shape", work, done)

    def auto_detect(self):
        if not self.info:
            return

        def done(found: list[LogoRegion]):
            new = [r for r in found if not any(self._overlaps(r, o) for o in self.regions)]
            if not new:
                if found:
                    self.set_status("Those logos are already marked.")
                else:
                    self.set_status("No logo found automatically.")
                    messagebox.showinfo("Nothing found",
                                        "No logo that stays in one place was found.\n\n"
                                        "Drag a box around it on the picture instead.", parent=self)
                return
            self.regions.extend(new)
            self.selected = len(self.regions) - 1
            self._refresh_tree()
            self.show_result.set(False)
            self._invalidate()
            self.set_status(f"Found {len(new)} logo area(s), highlighted in red. "
                            "Tick 'Show result' to see them removed.")
        self.run_bg("Scanning the video for logos", lambda: detect_logos(self._get_samples()), done)

    @staticmethod
    def _overlaps(a: LogoRegion, b: LogoRegion) -> bool:
        ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
        iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
        return ix * iy > 0.5 * min(a.w * a.h, b.w * b.h)

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(self.regions):
            if r.start is None and r.end is None:
                when = "whole video"
            else:
                when = f"{fmt_time(r.start or 0)}-{fmt_time(r.end) if r.end is not None else 'end'}"
            shape = "exact shape" if r.mask is not None else "whole box"
            self.tree.insert("", "end", iid=str(i), values=(f"{r.x}, {r.y}, {r.w} x {r.h}", when, shape))
        if self.selected is not None and self.selected < len(self.regions):
            self.tree.selection_set(str(self.selected))

    def _on_tree_select(self, _e):
        sel = self.tree.selection()
        new = int(sel[0]) if sel else None
        if new != self.selected:
            self.selected = new
            self.schedule_refresh()

    def select_region(self, idx: int):
        self.selected = idx
        self._refresh_tree()
        self.schedule_refresh()

    def delete_selected(self):
        if self.selected is None or self.selected >= len(self.regions):
            return
        del self.regions[self.selected]
        self.selected = None
        self._refresh_tree()
        self._invalidate()

    def clear_regions(self):
        if self.regions and messagebox.askyesno("Clear all", "Remove all logo boxes?", parent=self):
            self.regions, self.selected = [], None
            self._refresh_tree()
            self._invalidate()

    def set_region_time(self, which: str):
        if self.selected is None or self.selected >= len(self.regions):
            messagebox.showinfo("Pick a box", "Select a logo box in the list first.", parent=self)
            return
        r = self.regions[self.selected]
        if which == "start":
            r.start = self.t
            if r.end is not None and r.end < r.start:
                r.end = None
        elif which == "end":
            r.end = self.t
            if r.start is not None and r.start > r.end:
                r.start = None
        else:
            r.start = r.end = None
        self._refresh_tree()
        self._invalidate()

    def save_regions(self):
        if not self.info or not self.regions:
            messagebox.showinfo("Nothing to save", "Mark at least one logo first.", parent=self)
            return
        key = f"{self.info.width}x{self.info.height}"
        self.settings.saved_regions[key] = [r.to_dict() for r in self.regions]
        self.save_all_settings()
        self.set_status(f"Saved {len(self.regions)} logo box(es) for {key} videos.")

    def reuse_regions(self):
        if not self.info:
            return
        key = f"{self.info.width}x{self.info.height}"
        saved = self.settings.saved_regions.get(key)
        if not saved:
            messagebox.showinfo("Nothing saved", f"No saved logo boxes for {key} videos yet.\n\n"
                                "Mark the logo once, then click 'Save boxes for next time'.", parent=self)
            return
        self.regions = [LogoRegion.from_dict(d) for d in saved]
        self.selected = None
        self._refresh_tree()
        self._invalidate()
        self.set_status(f"Applied {len(self.regions)} saved logo box(es).")

    # ------------------------------------------------------------------ thumbnail
    def use_current_frame(self):
        if not self.info:
            return
        self.thumb_time = self.t
        self.thumb_lbl.config(text=f"Using: frame at {fmt_time(self.t)}")

    def find_best_frame(self):
        if not self.info:
            return
        info = self.info

        def done(t):
            self.best_time = self.thumb_time = t
            self.thumb_lbl.config(text=f"Using: best frame at {fmt_time(t)}")
            self.seek(t)
            self.set_status(f"Best-looking frame is at {fmt_time(t)}.")
        self.run_bg("Looking for the best frame", lambda: best_frame_time(info), done)

    def _thumbnail_job(self):
        """Returns a function (for a worker thread) that renders the thumbnail."""
        info, settings, thumb_time = self.info, self._thumbnail_settings(), self.thumb_time
        processor = self.make_processor()

        def work():
            t = thumb_time
            if t is None:
                if self.best_time is None:
                    self.best_time = best_frame_time(info)
                t = self.best_time
            return render_thumbnail(info, processor, t, settings)
        return work

    def preview_thumbnail(self):
        if not self.info:
            return

        def done(img):
            win = tk.Toplevel(self)
            win.title("Thumbnail preview")
            k = min(1.0, 900 / img.width, 640 / img.height)
            shown = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
            photo = ImageTk.PhotoImage(shown)
            lbl = ttk.Label(win, image=photo)
            lbl.image = photo
            lbl.pack(padx=10, pady=10)
            ttk.Label(win, text=f"Actual size: {img.width} x {img.height}", style="Hint.TLabel").pack(pady=(0, 10))
            self.set_status("Thumbnail preview ready.")
        self.run_bg("Rendering thumbnail", self._thumbnail_job(), done)

    def save_thumbnail(self):
        if not self.info:
            return
        stem = os.path.splitext(self.out_var.get() or self.info.path)[0]
        path = filedialog.asksaveasfilename(
            parent=self, title="Save thumbnail", defaultextension=".jpg",
            initialdir=os.path.dirname(stem), initialfile=os.path.basename(stem) + "_thumbnail.jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if not path:
            return
        job = self._thumbnail_job()

        def work():
            img = job()
            img.save(path, quality=92) if path.lower().endswith((".jpg", ".jpeg")) else img.save(path)
            return path
        self.run_bg("Saving thumbnail", work, lambda p: self.set_status(f"Thumbnail saved: {p}"))
        self.save_all_settings()

    # ------------------------------------------------------------------ export
    def _update_summary(self):
        if not hasattr(self, "summary_lbl"):
            return
        lines = []
        if self.regions:
            how = {"smart": "real background + AI fill", "inpaint": "fast fill", "blur": "blur"}[self._method_key()]
            lines.append(f"- Remove {len(self.regions)} logo area(s) ({how})")
        wm = self._watermark_settings()
        if wm.text.strip():
            lines.append(f"- Add text watermark \"{wm.text.strip()}\" ({wm.text_position})")
        if wm.logo_path and os.path.isfile(wm.logo_path):
            lines.append(f"- Add logo {os.path.basename(wm.logo_path)} ({wm.logo_position})")
        if self.ex_vars["embed_cover"].get() or self.ex_vars["save_thumbnail"].get():
            lines.append("- Make a thumbnail (settings in tab 3)")
        if not self.regions and not wm.enabled:
            lines.append("- Nothing to change yet: mark a logo (tab 1) or add your watermark (tab 2).")
        self.summary_lbl.config(text="\n".join(lines))

    def export(self):
        if not self.info:
            messagebox.showinfo("No video", "Open a video first.", parent=self)
            return
        out = self.out_var.get().strip()
        if not out:
            self._browse_output()
            out = self.out_var.get().strip()
            if not out:
                return
        if not os.path.splitext(out)[1]:
            out += ".mp4"
            self.out_var.set(out)
        if os.path.abspath(out).lower() == os.path.abspath(self.info.path).lower():
            messagebox.showerror("Pick another name", "Choose a different file name - the original "
                                 "video is never overwritten.", parent=self)
            return
        if os.path.exists(out) and not messagebox.askyesno("Replace file?", f"{os.path.basename(out)} "
                                                           "already exists. Replace it?", parent=self):
            return
        settings = self._collect_settings()
        self.save_all_settings()
        info, processor = self.info, self.make_processor()
        ex = settings.export
        need_thumb = ex.embed_cover or ex.save_thumbnail
        thumb_job = self._thumbnail_job() if need_thumb else None
        crf = QUALITY.get(ex.quality, 19)

        def progress(i, n):
            self.jobs.put(("progress", i, n))

        def work():
            cover = None
            if thumb_job:
                cover = thumb_job()
                if ex.save_thumbnail:
                    cover.save(os.path.splitext(out)[0] + "_thumbnail.jpg", quality=92)
            export_video(info, out, processor, crf=crf, preset=ex.preset, keep_audio=ex.keep_audio,
                         cover=cover if ex.embed_cover else None, progress=progress, cancel=self.cancel_evt)
            return out

        def done(path):
            self.last_output = path
            self.open_btn.config(state="normal")
            self.progress_lbl.config(text="Done!")
            size = os.path.getsize(path) / 1e6
            self.set_status(f"Saved {path} ({size:.1f} MB)")
            messagebox.showinfo("Export finished", f"Saved:\n{path}\n\n{size:.1f} MB", parent=self)

        self.progress["value"] = 0
        self.progress_lbl.config(text="Starting...")
        self._export_started = time.monotonic()
        self.run_bg("Exporting video", work, done, cancellable=True)

    def open_output_folder(self):
        if not self.last_output or not os.path.exists(self.last_output):
            return
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", "/select,", os.path.normpath(self.last_output)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", self.last_output])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(self.last_output)])

    def on_close(self):
        if self.busy and self.cancel_evt is not None:
            if not messagebox.askyesno("Export running", "An export is still running. Stop it and quit?", parent=self):
                return
            self.cancel_evt.set()
        self.save_all_settings()
        self._close_when_idle(60)

    def _close_when_idle(self, tries: int):
        # Give a cancelled export a moment to stop ffmpeg and delete its partial file.
        if self.busy and tries > 0:
            self.after(100, lambda: self._close_when_idle(tries - 1))
            return
        if self.reader:
            self.reader.close()
        self.destroy()


def main():
    if sys.platform.startswith("win"):
        try:  # crisp text on high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    if IMPORT_ERROR is not None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Missing packages", f"{IMPORT_ERROR}\n\nRun setup again (run.bat) or:\n"
                             "pip install -r requirements.txt")
        return
    app = App(sys.argv[1] if len(sys.argv) > 1 else None)
    if not find_ffmpeg():
        messagebox.showwarning("ffmpeg not found", "Previewing works, but exporting needs ffmpeg.\n\n"
                               "Install it with:  pip install imageio-ffmpeg\nor:  winget install ffmpeg",
                               parent=app)
    app.mainloop()


if __name__ == "__main__":
    main()
