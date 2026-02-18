"""
PWE Band Structure Solver GUI.

Tkinter GUI with parameter controls, shape presets, drawable dielectric canvas,
TM/TE band structure plots, and PINN inverse design.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading

from pwe import reciprocal_lattice, solve_bands, make_k_path

# Try importing PINN components (torch may not be installed)
try:
    import torch
    from pwe_torch import (reciprocal_lattice as reciprocal_lattice_t,
                           make_k_path as make_k_path_t,
                           solve_bands as solve_bands_t,
                           extract_gap)
    from pinn import InverseDesignNet, PhysicsLoss
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Geometry generators
# ---------------------------------------------------------------------------

def make_epsilon(shape, N=32, eps_rod=8.9, eps_bg=1.0, **kw):
    """Create NxN epsilon grid for a given shape centered in the unit cell."""
    xs = np.linspace(0, 1, N, endpoint=False) + 0.5 / N
    x, y = np.meshgrid(xs, xs, indexing="ij")
    cx, cy = 0.5, 0.5

    if shape == "circle":
        r = kw.get("r_over_a", 0.2)
        mask = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= r
    elif shape == "square":
        s = kw.get("s_over_a", 0.3)
        mask = (np.abs(x - cx) <= s / 2) & (np.abs(y - cy) <= s / 2)
    elif shape == "cross":
        arm_w = kw.get("arm_w", 0.1)
        arm_l = kw.get("arm_l", 0.3)
        h_arm = (np.abs(y - cy) <= arm_w / 2) & (np.abs(x - cx) <= arm_l / 2)
        v_arm = (np.abs(x - cx) <= arm_w / 2) & (np.abs(y - cy) <= arm_l / 2)
        mask = h_arm | v_arm
    elif shape == "ellipse":
        rx = kw.get("rx", 0.25)
        ry = kw.get("ry", 0.15)
        mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
    elif shape == "ring":
        r_out = kw.get("r_outer", 0.3)
        r_in = kw.get("r_inner", 0.15)
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = (dist <= r_out) & (dist >= r_in)
    else:
        mask = np.zeros((N, N), dtype=bool)

    return np.where(mask, eps_rod, eps_bg)


# Shape definitions: (display_name, shape_key, [(label, kwarg_key, default), ...])
SHAPE_DEFS = [
    ("Circle",  "circle",  [("r/a", "r_over_a", "0.2")]),
    ("Square",  "square",  [("s/a", "s_over_a", "0.3")]),
    ("Cross",   "cross",   [("arm width", "arm_w", "0.1"), ("arm length", "arm_l", "0.3")]),
    ("Ellipse", "ellipse", [("rx", "rx", "0.25"), ("ry", "ry", "0.15")]),
    ("Ring",    "ring",    [("r outer", "r_outer", "0.3"), ("r inner", "r_inner", "0.15")]),
    ("Custom",  "custom",  []),
]


# ---------------------------------------------------------------------------
# Bandgap analysis
# ---------------------------------------------------------------------------

def find_bandgaps(bands, n_bands):
    """Find gaps between consecutive bands."""
    gaps = []
    for i in range(n_bands - 1):
        top = float(np.max(bands[:, i]))
        bot = float(np.min(bands[:, i + 1]))
        gap = bot - top
        mid = 0.5 * (bot + top)
        if gap > 1e-6 and mid > 1e-6:
            gaps.append((i + 1, i + 2, gap, mid, gap / mid))
    return gaps


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------

class PWEApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PWE Band Structure Solver")
        self.root.minsize(1200, 700)

        self._solving = False
        self._pinn_training = False
        self._pinn_cancel = False
        self._draw_mode = False
        self._custom_eps = None  # user-drawn grid
        self._drawing = False    # mouse currently held
        self._preview_job = None # debounce timer id

        # --- Left panel: controls (scrollable) ---
        outer = ttk.Frame(root)
        outer.pack(side=tk.LEFT, fill=tk.Y)

        canvas_ctrl = tk.Canvas(outer, width=210, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas_ctrl.yview)
        ctrl = ttk.Frame(canvas_ctrl, padding=10)

        ctrl.bind("<Configure>",
                  lambda e: canvas_ctrl.configure(scrollregion=canvas_ctrl.bbox("all")))
        canvas_ctrl.create_window((0, 0), window=ctrl, anchor="nw")
        canvas_ctrl.configure(yscrollcommand=scrollbar.set)

        canvas_ctrl.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas_ctrl.yview_scroll(-1 * (event.delta // 120 or (1 if event.delta > 0 else -1)), "units")
        canvas_ctrl.bind_all("<MouseWheel>", _on_mousewheel)

        ttk.Label(ctrl, text="PWE Solver", font=("Helvetica", 14, "bold")).pack(anchor="w")
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=(5, 10))

        # -- Shape selector --
        ttk.Label(ctrl, text="Shape", font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.shape_var = tk.StringVar(value="Circle")
        shape_names = [s[0] for s in SHAPE_DEFS]
        shape_combo = ttk.Combobox(ctrl, textvariable=self.shape_var, values=shape_names,
                                   state="readonly", width=14)
        shape_combo.pack(fill="x", pady=(2, 5))
        shape_combo.bind("<<ComboboxSelected>>", self._on_shape_change)

        # Shape-specific params container
        self.shape_param_frame = ttk.Frame(ctrl)
        self.shape_param_frame.pack(fill="x")
        self.shape_params = {}  # key -> StringVar

        # -- Material params --
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=(8, 8))
        self.params = {}
        mat_defs = [
            ("eps rod",   "eps_rod",  "8.9"),
            ("eps bg",    "eps_bg",   "1.0"),
        ]
        for label_text, key, default in mat_defs:
            row = ttk.Frame(ctrl)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label_text, width=10).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            ttk.Entry(row, textvariable=var, width=8).pack(side=tk.LEFT, padx=(5, 0))
            self.params[key] = var

        for var in self.params.values():
            var.trace_add("write", self._schedule_preview)

        # -- Solver params --
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=(8, 8))
        solver_defs = [
            ("Grid N",    "n_grid",   "32"),
            ("PW n_max",  "n_max",    "5"),
            ("Bands",     "n_bands",  "10"),
            ("k/seg",     "n_k_seg",  "10"),
        ]
        for label_text, key, default in solver_defs:
            row = ttk.Frame(ctrl)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label_text, width=10).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            ttk.Entry(row, textvariable=var, width=8).pack(side=tk.LEFT, padx=(5, 0))
            self.params[key] = var

        self.params["n_grid"].trace_add("write", self._schedule_preview)

        self.pw_count_var = tk.StringVar(value="= 121 PWs")
        ttk.Label(ctrl, textvariable=self.pw_count_var, foreground="grey").pack(anchor="w", pady=(0, 2))
        self.params["n_max"].trace_add("write", self._update_pw_count)

        # -- Buttons --
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=10)

        btn_row = ttk.Frame(ctrl)
        btn_row.pack(fill="x", pady=(0, 5))
        self.solve_btn = ttk.Button(btn_row, text="Solve", command=self._on_solve)
        self.solve_btn.pack(side=tk.LEFT, expand=True, fill="x", padx=(0, 2))

        self.draw_var = tk.BooleanVar(value=False)
        self.draw_btn = ttk.Checkbutton(btn_row, text="Draw", variable=self.draw_var,
                                        command=self._on_draw_toggle)
        self.draw_btn.pack(side=tk.LEFT, padx=(2, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(ctrl, textvariable=self.status_var, wraplength=180).pack(anchor="w", pady=(5, 0))

        # -- Bandgap info --
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(ctrl, text="Bandgaps", font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.gap_text = tk.Text(ctrl, height=8, width=26, font=("Courier", 10), state="disabled")
        self.gap_text.pack(fill="x", pady=(5, 0))

        # ==============================================================
        # PINN Inverse Design section
        # ==============================================================
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(ctrl, text="PINN Inverse Design",
                  font=("Helvetica", 11, "bold")).pack(anchor="w")

        if not HAS_TORCH:
            ttk.Label(ctrl, text="(torch not installed)",
                      foreground="grey").pack(anchor="w")
        else:
            self.pinn_params = {}

            # Objective selector: maximize gap or target specific values
            obj_row = ttk.Frame(ctrl)
            obj_row.pack(fill="x", pady=(2, 4))
            self.pinn_maximize_var = tk.BooleanVar(value=False)
            ttk.Radiobutton(obj_row, text="Target", value=False,
                            variable=self.pinn_maximize_var,
                            command=self._on_pinn_obj_change).pack(side=tk.LEFT)
            ttk.Radiobutton(obj_row, text="Maximize gap", value=True,
                            variable=self.pinn_maximize_var,
                            command=self._on_pinn_obj_change).pack(side=tk.LEFT, padx=(6, 0))

            # Target-specific params (disabled when maximize is checked)
            self._pinn_target_frame = ttk.Frame(ctrl)
            self._pinn_target_frame.pack(fill="x")
            pinn_target_defs = [
                ("target freq", "pinn_target_freq", "0.35"),
                ("target width","pinn_target_width","0.05"),
            ]
            self._pinn_target_entries = []
            for label_text, key, default in pinn_target_defs:
                row = ttk.Frame(self._pinn_target_frame)
                row.pack(fill="x", pady=1)
                ttk.Label(row, text=label_text, width=12).pack(side=tk.LEFT)
                var = tk.StringVar(value=default)
                entry = ttk.Entry(row, textvariable=var, width=8)
                entry.pack(side=tk.LEFT, padx=(5, 0))
                self.pinn_params[key] = var
                self._pinn_target_entries.append(entry)

            pinn_defs = [
                ("band lo",     "pinn_band_lo",     "0"),
                ("band hi",     "pinn_band_hi",     "1"),
                ("steps",       "pinn_steps",       "200"),
                ("lr",          "pinn_lr",          "0.001"),
                ("latent dim",  "pinn_latent_dim",  "32"),
            ]
            for label_text, key, default in pinn_defs:
                row = ttk.Frame(ctrl)
                row.pack(fill="x", pady=1)
                ttk.Label(row, text=label_text, width=12).pack(side=tk.LEFT)
                var = tk.StringVar(value=default)
                ttk.Entry(row, textvariable=var, width=8).pack(side=tk.LEFT, padx=(5, 0))
                self.pinn_params[key] = var

            # Loss weights in a compact sub-section
            ttk.Label(ctrl, text="Loss weights", foreground="grey",
                      font=("Helvetica", 9)).pack(anchor="w", pady=(4, 0))
            weight_defs = [
                ("w gap",    "pinn_w_gap",    "1.0"),
                ("w freq",   "pinn_w_freq",   "1.0"),
                ("w binary", "pinn_w_binary", "0.1"),
                ("w bloch",  "pinn_w_bloch",  "0.01"),
            ]
            for label_text, key, default in weight_defs:
                row = ttk.Frame(ctrl)
                row.pack(fill="x", pady=1)
                ttk.Label(row, text=label_text, width=12).pack(side=tk.LEFT)
                var = tk.StringVar(value=default)
                ttk.Entry(row, textvariable=var, width=8).pack(side=tk.LEFT, padx=(5, 0))
                self.pinn_params[key] = var

            pinn_btn_row = ttk.Frame(ctrl)
            pinn_btn_row.pack(fill="x", pady=(6, 2))
            self.pinn_train_btn = ttk.Button(pinn_btn_row, text="Train PINN",
                                              command=self._on_pinn_train)
            self.pinn_train_btn.pack(side=tk.LEFT, expand=True, fill="x", padx=(0, 2))
            self.pinn_cancel_btn = ttk.Button(pinn_btn_row, text="Stop",
                                               command=self._on_pinn_cancel,
                                               state="disabled")
            self.pinn_cancel_btn.pack(side=tk.LEFT, padx=(2, 0))

            self.pinn_status_var = tk.StringVar(value="")
            ttk.Label(ctrl, textvariable=self.pinn_status_var,
                      wraplength=180, foreground="grey").pack(anchor="w", pady=(2, 0))

            # Progress bar
            self.pinn_progress = ttk.Progressbar(ctrl, mode="determinate", length=180)
            self.pinn_progress.pack(fill="x", pady=(2, 0))

        # --- Right panel: plots ---
        plot_frame = ttk.Frame(root)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax_eps = self.fig.add_subplot(1, 3, 1)
        self.ax_tm = self.fig.add_subplot(1, 3, 2)
        self.ax_te = self.fig.add_subplot(1, 3, 3)
        self.fig.subplots_adjust(left=0.06, right=0.97, wspace=0.35, top=0.92, bottom=0.10)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Drawing event hooks
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)

        # Build initial shape params and solve
        self._on_shape_change()
        self.root.after(100, self._on_solve)

    # -----------------------------------------------------------------------
    # Shape parameter UI
    # -----------------------------------------------------------------------

    def _on_shape_change(self, event=None):
        """Rebuild shape-specific parameter fields."""
        for w in self.shape_param_frame.winfo_children():
            w.destroy()
        self.shape_params.clear()

        shape_name = self.shape_var.get()
        for sdef in SHAPE_DEFS:
            if sdef[0] == shape_name:
                for label_text, key, default in sdef[2]:
                    row = ttk.Frame(self.shape_param_frame)
                    row.pack(fill="x", pady=1)
                    ttk.Label(row, text=label_text, width=10).pack(side=tk.LEFT)
                    var = tk.StringVar(value=default)
                    ttk.Entry(row, textvariable=var, width=8).pack(side=tk.LEFT, padx=(5, 0))
                    self.shape_params[key] = var
                    var.trace_add("write", self._schedule_preview)
                break

        if shape_name == "Custom":
            ttk.Label(self.shape_param_frame,
                      text="Click on dielectric plot\nto paint pixels.\nToggle Draw mode first.",
                      foreground="grey", wraplength=160).pack(anchor="w", pady=4)

        # Reset custom grid when switching away
        if shape_name != "Custom":
            self._custom_eps = None
            self.draw_var.set(False)
            self._draw_mode = False

        self._schedule_preview()

    # -----------------------------------------------------------------------
    # Live dielectric preview
    # -----------------------------------------------------------------------

    def _schedule_preview(self, *_):
        """Debounced preview update (200ms after last change)."""
        if self._draw_mode:
            return
        if self._preview_job is not None:
            self.root.after_cancel(self._preview_job)
        self._preview_job = self.root.after(200, self._update_preview)

    def _update_preview(self):
        """Regenerate and display the dielectric pattern without solving."""
        self._preview_job = None
        try:
            p = self._read_params()
            eps_grid = self._build_eps_grid(p)
        except (ValueError, KeyError):
            return
        ax = self.ax_eps
        ax.clear()
        ax.imshow(eps_grid.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdYlBu_r")
        ax.set_title("Dielectric")
        ax.set_xlabel("x/a")
        ax.set_ylabel("y/a")
        ax.set_aspect("equal")
        self.canvas.draw_idle()

    # -----------------------------------------------------------------------
    # Drawing on canvas
    # -----------------------------------------------------------------------

    def _on_draw_toggle(self):
        self._draw_mode = self.draw_var.get()
        if self._draw_mode:
            self.shape_var.set("Custom")
            self._on_shape_change()
            if self._custom_eps is None:
                N = int(self.params["n_grid"].get())
                eps_bg = float(self.params["eps_bg"].get())
                self._custom_eps = np.full((N, N), eps_bg)
                self._refresh_eps_plot()
            self.status_var.set("Draw mode: L-click=rod, R-click=bg")
        else:
            self.status_var.set("Ready")

    def _canvas_to_grid(self, event):
        """Convert matplotlib event coords to grid indices, or None."""
        if event.inaxes is not self.ax_eps:
            return None
        N = self._custom_eps.shape[0]
        ix = int(event.xdata * N)
        iy = int(event.ydata * N)
        if 0 <= ix < N and 0 <= iy < N:
            return ix, iy
        return None

    def _paint(self, event):
        """Paint a pixel on the custom eps grid."""
        if self._custom_eps is None:
            return
        pos = self._canvas_to_grid(event)
        if pos is None:
            return
        ix, iy = pos
        brush = max(1, self._custom_eps.shape[0] // 32)  # scale brush with grid
        eps_rod = float(self.params["eps_rod"].get())
        eps_bg = float(self.params["eps_bg"].get())
        val = eps_rod if event.button == 1 else eps_bg

        for di in range(-brush + 1, brush):
            for dj in range(-brush + 1, brush):
                ni, nj = ix + di, iy + dj
                N = self._custom_eps.shape[0]
                if 0 <= ni < N and 0 <= nj < N:
                    self._custom_eps[ni, nj] = val
        self._refresh_eps_plot()

    def _on_canvas_press(self, event):
        if not self._draw_mode:
            return
        self._drawing = True
        self._paint(event)

    def _on_canvas_motion(self, event):
        if not self._draw_mode or not self._drawing:
            return
        self._paint(event)

    def _on_canvas_release(self, event):
        self._drawing = False

    def _refresh_eps_plot(self):
        """Redraw just the dielectric subplot."""
        ax = self.ax_eps
        ax.clear()
        ax.imshow(self._custom_eps.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdYlBu_r")
        ax.set_title("Dielectric (draw)")
        ax.set_xlabel("x/a")
        ax.set_ylabel("y/a")
        ax.set_aspect("equal")
        self.canvas.draw_idle()

    # -----------------------------------------------------------------------
    # PWE Solver
    # -----------------------------------------------------------------------

    def _update_pw_count(self, *_):
        try:
            n = int(self.params["n_max"].get())
            self.pw_count_var.set(f"= {(2*n+1)**2} PWs")
        except ValueError:
            self.pw_count_var.set("")

    def _read_params(self):
        return {
            "eps_rod":  float(self.params["eps_rod"].get()),
            "eps_bg":   float(self.params["eps_bg"].get()),
            "n_grid":   int(self.params["n_grid"].get()),
            "n_max":    int(self.params["n_max"].get()),
            "n_bands":  int(self.params["n_bands"].get()),
            "n_k_seg":  int(self.params["n_k_seg"].get()),
        }

    def _build_eps_grid(self, p):
        """Build epsilon grid from current shape or custom drawing."""
        shape_name = self.shape_var.get()
        if shape_name == "Custom" and self._custom_eps is not None:
            # Resize if grid N changed
            if self._custom_eps.shape[0] != p["n_grid"]:
                from scipy.ndimage import zoom
                ratio = p["n_grid"] / self._custom_eps.shape[0]
                self._custom_eps = zoom(self._custom_eps, ratio, order=0)
            return self._custom_eps.copy()

        shape_key = None
        for sdef in SHAPE_DEFS:
            if sdef[0] == shape_name:
                shape_key = sdef[1]
                break
        kw = {k: float(v.get()) for k, v in self.shape_params.items()}
        return make_epsilon(shape_key, N=p["n_grid"], eps_rod=p["eps_rod"], eps_bg=p["eps_bg"], **kw)

    def _on_solve(self):
        if self._solving:
            return
        try:
            p = self._read_params()
        except ValueError:
            self.status_var.set("Invalid parameter value")
            return

        eps_grid = self._build_eps_grid(p)

        self._solving = True
        self.solve_btn.config(state="disabled")
        self.status_var.set("Solving...")

        thread = threading.Thread(target=self._solve_worker, args=(p, eps_grid), daemon=True)
        thread.start()

    def _solve_worker(self, p, eps_grid):
        try:
            g_vectors, m_indices = reciprocal_lattice(p["n_max"])
            k_points, k_dist, tick_pos, tick_labels = make_k_path(p["n_k_seg"])

            bands_tm = solve_bands(k_points, g_vectors, eps_grid, m_indices, p["n_bands"], "tm")
            bands_te = solve_bands(k_points, g_vectors, eps_grid, m_indices, p["n_bands"], "te")

            result = {
                "eps_grid": eps_grid, "bands_tm": bands_tm, "bands_te": bands_te,
                "k_dist": k_dist, "tick_pos": tick_pos, "tick_labels": tick_labels,
                "p": p,
            }
            self.root.after(0, lambda: self._update_plots(result))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {e}"))
        finally:
            self.root.after(0, self._solve_done)

    def _solve_done(self):
        self._solving = False
        self.solve_btn.config(state="normal")

    # -----------------------------------------------------------------------
    # Plotting (standard PWE)
    # -----------------------------------------------------------------------

    def _update_plots(self, r):
        p = r["p"]
        eps_grid = r["eps_grid"]
        bands_tm, bands_te = r["bands_tm"], r["bands_te"]
        k_dist = r["k_dist"]
        tick_pos, tick_labels = r["tick_pos"], r["tick_labels"]
        n_bands = p["n_bands"]

        # -- Dielectric pattern --
        ax = self.ax_eps
        ax.clear()
        ax.imshow(eps_grid.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdYlBu_r")
        ax.set_title("Dielectric")
        ax.set_xlabel("x/a")
        ax.set_ylabel("y/a")
        ax.set_aspect("equal")

        # -- TM bands --
        ax = self.ax_tm
        ax.clear()
        for i in range(n_bands):
            ax.plot(k_dist, bands_tm[:, i], color="#2563eb", linewidth=0.9)
        for tp in tick_pos:
            ax.axvline(tp, color="grey", linewidth=0.4, linestyle="--")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(k_dist[0], k_dist[-1])
        ax.set_ylim(0, 0.8)
        ax.set_title("TM")
        ax.set_ylabel("$\\omega a / 2\\pi c$")

        # -- TE bands --
        ax = self.ax_te
        ax.clear()
        for i in range(n_bands):
            ax.plot(k_dist, bands_te[:, i], color="#dc2626", linewidth=0.9)
        for tp in tick_pos:
            ax.axvline(tp, color="grey", linewidth=0.4, linestyle="--")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(k_dist[0], k_dist[-1])
        ax.set_ylim(0, 0.8)
        ax.set_title("TE")

        # -- Shade bandgaps --
        tm_gaps = find_bandgaps(bands_tm, n_bands)
        te_gaps = find_bandgaps(bands_te, n_bands)
        for _, _, gap, mid, ratio in tm_gaps:
            lo = mid - gap / 2
            self.ax_tm.axhspan(lo, lo + gap, alpha=0.12, color="#2563eb")
        for _, _, gap, mid, ratio in te_gaps:
            lo = mid - gap / 2
            self.ax_te.axhspan(lo, lo + gap, alpha=0.12, color="#dc2626")

        self.canvas.draw()

        # -- Bandgap text --
        lines = []
        lines.append("-- TM --")
        if tm_gaps:
            for b1, b2, gap, mid, ratio in tm_gaps:
                lines.append(f" {b1}-{b2}: gap={gap:.4f} ratio={ratio:.4f}")
        else:
            lines.append(" (none)")
        lines.append("")
        lines.append("-- TE --")
        if te_gaps:
            for b1, b2, gap, mid, ratio in te_gaps:
                lines.append(f" {b1}-{b2}: gap={gap:.4f} ratio={ratio:.4f}")
        else:
            lines.append(" (none)")

        self.gap_text.config(state="normal")
        self.gap_text.delete("1.0", tk.END)
        self.gap_text.insert("1.0", "\n".join(lines))
        self.gap_text.config(state="disabled")

        n_pw = (2 * p["n_max"] + 1) ** 2
        self.status_var.set(f"Done. {n_pw} PWs, {len(k_dist)} k-pts")

    # -----------------------------------------------------------------------
    # PINN Inverse Design
    # -----------------------------------------------------------------------

    def _on_pinn_obj_change(self):
        """Enable/disable target freq/width entries based on objective mode."""
        maximize = self.pinn_maximize_var.get()
        state = "disabled" if maximize else "normal"
        for entry in self._pinn_target_entries:
            entry.config(state=state)

    def _read_pinn_params(self):
        return {
            "target_freq":  float(self.pinn_params["pinn_target_freq"].get()),
            "target_width": float(self.pinn_params["pinn_target_width"].get()),
            "maximize":     self.pinn_maximize_var.get(),
            "band_lo":      int(self.pinn_params["pinn_band_lo"].get()),
            "band_hi":      int(self.pinn_params["pinn_band_hi"].get()),
            "steps":        int(self.pinn_params["pinn_steps"].get()),
            "lr":           float(self.pinn_params["pinn_lr"].get()),
            "latent_dim":   int(self.pinn_params["pinn_latent_dim"].get()),
            "w_gap":        float(self.pinn_params["pinn_w_gap"].get()),
            "w_freq":       float(self.pinn_params["pinn_w_freq"].get()),
            "w_binary":     float(self.pinn_params["pinn_w_binary"].get()),
            "w_bloch":      float(self.pinn_params["pinn_w_bloch"].get()),
        }

    def _on_pinn_train(self):
        if self._pinn_training:
            return
        try:
            pp = self._read_pinn_params()
            sp = self._read_params()
        except ValueError:
            self.pinn_status_var.set("Invalid parameter")
            return

        self._pinn_training = True
        self._pinn_cancel = False
        self.pinn_train_btn.config(state="disabled")
        self.pinn_cancel_btn.config(state="normal")
        self.pinn_progress["value"] = 0
        self.pinn_progress["maximum"] = pp["steps"]
        self.pinn_status_var.set("Training...")

        thread = threading.Thread(target=self._pinn_worker,
                                  args=(pp, sp), daemon=True)
        thread.start()

    def _on_pinn_cancel(self):
        self._pinn_cancel = True
        self.pinn_status_var.set("Stopping...")

    def _pinn_worker(self, pp, sp):
        """Run PINN inverse design training in background thread."""
        try:
            torch.manual_seed(42)
            n_grid = sp["n_grid"]
            # Force even grid for C4v tiling
            if n_grid % 2 != 0:
                n_grid += 1

            g_vectors, m_indices = reciprocal_lattice_t(sp["n_max"])
            k_points, k_dist, tick_pos, tick_labels = make_k_path_t(sp["n_k_seg"])

            model = InverseDesignNet(
                N=n_grid, latent_dim=pp["latent_dim"],
                eps_bg=sp["eps_bg"], eps_rod=sp["eps_rod"],
            ).double()

            physics_loss = PhysicsLoss(
                w_gap=pp["w_gap"], w_freq=pp["w_freq"],
                w_binary=pp["w_binary"], w_bloch=pp["w_bloch"],
                maximize=pp["maximize"],
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=pp["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, pp["steps"]
            )

            binary_schedule = np.linspace(
                pp["w_binary"] * 0.1, pp["w_binary"], pp["steps"]
            )

            history = []
            n_bands = sp["n_bands"]

            for step in range(pp["steps"]):
                if self._pinn_cancel:
                    break

                optimizer.zero_grad()
                physics_loss.w_binary = float(binary_schedule[step])

                eps_grid, eps_norm = model(
                    pp["target_freq"], pp["target_width"],
                    pp["band_lo"], pp["band_hi"],
                )
                bands = solve_bands_t(
                    k_points, g_vectors, eps_grid, m_indices, n_bands, "tm"
                )
                loss, info = physics_loss(
                    bands, eps_grid, eps_norm,
                    pp["target_freq"], pp["target_width"],
                    pp["band_lo"], pp["band_hi"],
                )

                loss.backward()

                # Skip step if gradients are NaN (degenerate eigenvalues)
                has_nan = False
                for p in model.parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        has_nan = True
                        break
                if has_nan or torch.isnan(loss):
                    optimizer.zero_grad()
                    info["step"] = step
                    info["total"] = float("nan")
                    history.append(info)
                    scheduler.step()
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                info["step"] = step
                history.append(info)

                # Update GUI periodically
                if (step + 1) % max(1, pp["steps"] // 40) == 0 or step == 0:
                    _info = dict(info)
                    _step = step + 1
                    _total = pp["steps"]
                    self.root.after(0, lambda s=_step, t=_total, i=_info:
                        self._pinn_progress_update(s, t, i))

            # Final result
            with torch.no_grad():
                eps_grid, eps_norm = model(
                    pp["target_freq"], pp["target_width"],
                    pp["band_lo"], pp["band_hi"],
                )
                bands = solve_bands_t(
                    k_points, g_vectors, eps_grid, m_indices, n_bands, "tm"
                )

            result = {
                "eps_grid": eps_grid.detach().cpu().numpy(),
                "bands": bands.detach().cpu().numpy(),
                "k_dist": k_dist,
                "tick_pos": tick_pos,
                "tick_labels": tick_labels,
                "history": history,
                "pp": pp,
                "n_bands": n_bands,
            }
            self.root.after(0, lambda: self._pinn_done(result))

        except Exception as e:
            self.root.after(0, lambda: self._pinn_error(str(e)))

    def _pinn_progress_update(self, step, total, info):
        """Update progress bar and status from main thread."""
        self.pinn_progress["value"] = step
        extra = f" ratio={info['gap_ratio']:.4f}" if "gap_ratio" in info else ""
        self.pinn_status_var.set(
            f"[{step}/{total}] loss={info['total']:.5f} "
            f"gap={info['gap_width']:.4f} mid={info['midgap']:.4f}{extra}"
        )

    def _pinn_done(self, result):
        """Display PINN results in the plot area."""
        self._pinn_training = False
        self.pinn_train_btn.config(state="normal")
        self.pinn_cancel_btn.config(state="disabled")
        self.pinn_progress["value"] = self.pinn_progress["maximum"]

        eps_grid = result["eps_grid"]
        bands = result["bands"]
        k_dist = result["k_dist"]
        tick_pos = result["tick_pos"]
        tick_labels = result["tick_labels"]
        history = result["history"]
        pp = result["pp"]
        n_bands = result["n_bands"]

        # Compute actual gap from band data (not from smooth approximation)
        band_lo = pp["band_lo"]
        band_hi = pp["band_hi"]
        floor_val = float(np.max(bands[:, band_lo]))   # top of lower band
        ceil_val = float(np.min(bands[:, band_hi]))     # bottom of upper band
        gap_width = max(0.0, ceil_val - floor_val)
        midgap = 0.5 * (ceil_val + floor_val)

        # -- Dielectric (PINN result) --
        ax = self.ax_eps
        ax.clear()
        ax.imshow(eps_grid.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdYlBu_r")
        ax.set_title("PINN unit cell")
        ax.set_xlabel("x/a")
        ax.set_ylabel("y/a")
        ax.set_aspect("equal")

        # -- TM bands (PINN) --
        ax = self.ax_tm
        ax.clear()
        for i in range(n_bands):
            ax.plot(k_dist, bands[:, i], color="#2563eb", linewidth=0.9)
        for tp in tick_pos:
            ax.axvline(tp, color="grey", linewidth=0.4, linestyle="--")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(k_dist[0], k_dist[-1])
        ax.set_ylim(0, max(0.8, midgap + gap_width + 0.1))
        ax.set_ylabel("$\\omega a / 2\\pi c$")
        ax.set_title("TM (PINN)")
        # Shade between actual band edges (floor_val to ceil_val)
        if gap_width > 0:
            ax.axhspan(floor_val, ceil_val, alpha=0.15, color="#2563eb")
        ax.axhline(pp["target_freq"], color="red", linewidth=0.7, linestyle=":")
        ax.axhline(pp["target_freq"] - pp["target_width"] / 2,
                    color="red", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axhline(pp["target_freq"] + pp["target_width"] / 2,
                    color="red", linewidth=0.5, linestyle="--", alpha=0.5)

        # -- Convergence plot (replaces TE during PINN) --
        ax = self.ax_te
        ax.clear()
        if history:
            steps = [h["step"] for h in history]
            ax.plot(steps, [h["total"] for h in history], label="total",
                    color="#1e293b", linewidth=1.0)
            ax.plot(steps, [h["loss_gap"] for h in history], label="gap",
                    color="#2563eb", linewidth=0.8, alpha=0.7)
            ax.plot(steps, [h["loss_freq"] for h in history], label="freq",
                    color="#dc2626", linewidth=0.8, alpha=0.7)
            ax.plot(steps, [h["loss_binary"] for h in history], label="binary",
                    color="#16a34a", linewidth=0.8, alpha=0.7)
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
            ax.set_title("PINN convergence")
            ax.legend(fontsize=7, loc="upper right")
            # Use log scale, but handle zeros gracefully
            all_vals = [h["total"] for h in history]
            if min(all_vals) > 0:
                ax.set_yscale("log")

        self.canvas.draw()

        # Update bandgap text
        tm_gaps = find_bandgaps(bands, n_bands)
        lines = ["-- PINN TM --"]
        if tm_gaps:
            for b1, b2, gap, mid, ratio in tm_gaps:
                lines.append(f" {b1}-{b2}: gap={gap:.4f} ratio={ratio:.4f}")
        else:
            lines.append(" (none)")
        lines.append("")
        lines.append(f"target: f={pp['target_freq']:.3f} w={pp['target_width']:.3f}")
        lines.append(f"result: f={midgap:.4f} w={gap_width:.4f}")

        self.gap_text.config(state="normal")
        self.gap_text.delete("1.0", tk.END)
        self.gap_text.insert("1.0", "\n".join(lines))
        self.gap_text.config(state="disabled")

        self.pinn_status_var.set(
            f"Done. gap={gap_width:.4f} midgap={midgap:.4f}"
        )
        self.status_var.set("PINN training complete. Press Solve for PWE.")

    def _pinn_error(self, msg):
        self._pinn_training = False
        self.pinn_train_btn.config(state="normal")
        self.pinn_cancel_btn.config(state="disabled")
        self.pinn_status_var.set(f"Error: {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = PWEApp(root)
    root.mainloop()
