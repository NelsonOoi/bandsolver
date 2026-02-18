"""
PWE Band Structure Solver GUI.

Tkinter GUI with parameter controls, shape presets, drawable dielectric canvas,
and TM/TE band structure plots.
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
        self._draw_mode = False
        self._custom_eps = None  # user-drawn grid
        self._drawing = False    # mouse currently held
        self._preview_job = None # debounce timer id

        # --- Left panel: controls ---
        ctrl = ttk.Frame(root, padding=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

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
        self.gap_text = tk.Text(ctrl, height=10, width=26, font=("Courier", 10), state="disabled")
        self.gap_text.pack(fill="x", pady=(5, 0))

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
    # Solver
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
    # Plotting
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = PWEApp(root)
    root.mainloop()
