"""
Validate hexagonal-lattice photonic and phononic PWE solvers.

Supports two geometry types:
  - rod:       Si rods in air (default)
  - snowflake: Mercedes-benz / snowflake air holes in Si

Usage:
    python validate_hex.py                                       # Si rods
    python validate_hex.py --geometry snowflake                  # snowflake sweep
    python validate_hex.py --geometry snowflake --arm-width 0.15 # fixed w, sweep r
    python validate_hex.py --n-max 5 --grid-size 32
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pwe_hex_torch import (hex_reciprocal_lattice, hex_make_k_path,
                           hex_solve_bands, make_hex_rod_epsilon,
                           make_hex_snowflake_epsilon, find_best_gap,
                           _oblique_coords, _oblique_to_cartesian, A1, A2)
from pwe_phononic_hex_torch import solve_phononic_bands


def plot_geometry(ax, eps_np, N, title):
    s1, s2 = _oblique_coords(N)
    offsets = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
               (1, 1), (-1, -1), (1, -1), (-1, 1)]
    vmin, vmax = eps_np.min(), eps_np.max()
    for di, dj in offsets:
        x, y = _oblique_to_cartesian(s1 + di, s2 + dj)
        ax.pcolormesh(x, y, eps_np, cmap="RdYlBu_r", vmin=vmin,
                      vmax=vmax, shading="auto")
    corners = np.array([[0, 0], A1, A1 + A2, A2, [0, 0]])
    ax.plot(corners[:, 0], corners[:, 1], "k-", lw=2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, A1[0] + A2[0] + 0.3)
    ax.set_ylim(-0.3, A2[1] + 0.3)
    ax.set_title(title)
    ax.set_xlabel("x/a")
    ax.set_ylabel("y/a")


def plot_bands(ax, kd, bands_np, n_bands, tp, tl, ylabel, title, gap_color):
    for b in range(n_bands):
        ax.plot(kd, bands_np[:, b], lw=1)
    for t in tp:
        ax.axvline(t, color="gray", lw=0.5, ls="--")
    ax.set_xticks(tp)
    ax.set_xticklabels(tl)
    ax.set_ylabel(ylabel)
    gw, gm, gp = find_best_gap(bands_np)
    ax.set_title(f"{title}\ngap={gw:.4f} @ bands {gp}-{gp+1}")
    if gw > 0.001:
        floor_v = np.max(bands_np[:, gp])
        ceil_v = np.min(bands_np[:, gp + 1])
        ax.axhspan(floor_v, ceil_v, alpha=0.2, color=gap_color)
    return gw, gm, gp


def run_column(axes_col, eps, s, label, kp, kd, tp, tl, g, m,
               n_bands_em, n_bands_ac, N):
    """Run all solvers and plot one column (geometry + TM + TE + phononic)."""
    eps_np = eps.numpy()
    fill = float(s.mean())

    plot_geometry(axes_col[0], eps_np, N, f"ε(r), {label}, fill={fill:.2f}")

    bands_tm = hex_solve_bands(kp, g, eps, m, n_bands_em, "tm")
    bands_tm_np = bands_tm.detach().numpy()
    gw_tm, gm_tm, gp_tm = plot_bands(
        axes_col[1], kd, bands_tm_np, n_bands_em, tp, tl,
        "Frequency (a/λ)", f"TM, {label}", "red")
    axes_col[1].set_ylim(0, max(0.8, bands_tm_np[:, min(n_bands_em - 1, 7)].max() * 1.1))

    bands_te = hex_solve_bands(kp, g, eps, m, n_bands_em, "te")
    bands_te_np = bands_te.detach().numpy()
    gw_te, gm_te, gp_te = plot_bands(
        axes_col[2], kd, bands_te_np, n_bands_em, tp, tl,
        "Frequency (a/λ)", f"TE, {label}", "orange")
    axes_col[2].set_ylim(0, max(0.8, bands_te_np[:, min(n_bands_em - 1, 7)].max() * 1.1))

    bands_ac = solve_phononic_bands(kp, g, s, m, n_bands_ac)
    bands_ac_np = bands_ac.detach().numpy()
    gw_ac, gm_ac, gp_ac = plot_bands(
        axes_col[3], kd, bands_ac_np, n_bands_ac, tp, tl,
        "Frequency (ωa/2πv_ref)", f"Phononic, {label}", "blue")

    print(f"  {label} fill={fill:.2f}: "
          f"TM gap={gw_tm:.5f} ({gp_tm}-{gp_tm+1}), "
          f"TE gap={gw_te:.5f} ({gp_te}-{gp_te+1}), "
          f"Phon gap={gw_ac:.5f} ({gp_ac}-{gp_ac+1})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["rod", "snowflake"], default="rod")
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--n-max", type=int, default=5)
    parser.add_argument("--n-bands-em", type=int, default=8)
    parser.add_argument("--n-bands-ac", type=int, default=8)
    parser.add_argument("--k-segments", type=int, default=20)
    parser.add_argument("--eps-rod", type=float, default=11.7)
    parser.add_argument("--eps-bg", type=float, default=1.0)
    parser.add_argument("--output", type=str, default=None)

    # Rod-specific
    parser.add_argument("--radii", type=float, nargs="+", default=[0.2, 0.3, 0.4])

    # Snowflake-specific
    parser.add_argument("--arm-lengths", type=float, nargs="+",
                        default=[0.15, 0.25, 0.35])
    parser.add_argument("--arm-width", type=float, default=0.10)

    args = parser.parse_args()
    if args.output is None:
        args.output = f"hex_validation_{args.geometry}.png"

    N = args.grid_size
    g, m_idx = hex_reciprocal_lattice(args.n_max)
    kp, kd, tp, tl = hex_make_k_path(args.k_segments)
    print(f"n_pw={g.shape[0]}, n_k={kp.shape[0]}, grid={N}x{N}, "
          f"geometry={args.geometry}")

    if args.geometry == "rod":
        cases = []
        for r in args.radii:
            eps = make_hex_rod_epsilon(N, r, args.eps_rod, args.eps_bg)
            s = (eps > 0.5 * (args.eps_rod + args.eps_bg)).to(torch.float64)
            cases.append((eps, s, f"r/a={r}"))
        suptitle = f"Hexagonal lattice: Si rods (ε={args.eps_rod}) in air"
    else:
        cases = []
        w = args.arm_width
        for r in args.arm_lengths:
            eps = make_hex_snowflake_epsilon(N, r, w, args.eps_rod, args.eps_bg)
            s = (eps > 0.5 * (args.eps_rod + args.eps_bg)).to(torch.float64)
            cases.append((eps, s, f"r={r}, w={w}"))
        suptitle = (f"Hexagonal lattice: snowflake holes in Si "
                    f"(ε={args.eps_rod}), w/a={w}")

    n_cols = len(cases)
    fig, axes = plt.subplots(4, n_cols, figsize=(6 * n_cols, 18))
    if n_cols == 1:
        axes = axes[:, None]

    for col, (eps, s, label) in enumerate(cases):
        run_column([axes[row, col] for row in range(4)],
                   eps, s, label, kp, kd, tp, tl, g, m_idx,
                   args.n_bands_em, args.n_bands_ac, N)

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
