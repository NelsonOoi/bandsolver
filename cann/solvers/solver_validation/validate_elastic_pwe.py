"""
Validation of the in-plane elastic PWE solver.

Three tests:
  1. Homogeneous slab (analytical: linear dispersion omega = c * |k|)
  2. Ni cylinders in Al matrix (Kushwaha et al. benchmark, moderate contrast)
  3. Convergence study: n_max sweep for both Ni/Al and Si/air cases

Produces validate_elastic_pwe.png with all panels.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from pwe_torch import reciprocal_lattice, make_k_path
from pwe_elastic_torch import (
    solve_elastic_bands, solve_elastic_bands_from_mask,
    make_material_grids, SI_PROPS, AIR_PROPS,
)


# ── Material databases ──────────────────────────────────────────────

AL_PROPS = dict(rho=2697.0, lam=54.3e9, mu=28.0e9)   # C11=110.3, C44=28.0
NI_PROPS = dict(rho=8968.0, lam=73.0e9, mu=122.0e9)   # C11=317.0, C44=122.0
AIR_CORRECTED = dict(rho=1.225, lam=1.42e5, mu=1e-2)   # physical air


def make_circle_mask(r: float, N: int = 64):
    xs = torch.linspace(-0.5, 0.5, N + 1)[:-1] + 0.5 / N
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    return ((xx ** 2 + yy ** 2) <= r ** 2).to(torch.float64)


def run_elastic(k_points, g_vectors, m_indices, mask, n_bands,
                mat_hole, mat_matrix):
    rho, lam, mu = make_material_grids(mask, mat_hole=mat_hole,
                                       mat_matrix=mat_matrix)
    with torch.no_grad():
        return solve_elastic_bands(k_points, g_vectors, rho, lam, mu,
                                   m_indices, n_bands)


# ── Test 1: homogeneous slab ────────────────────────────────────────

def test_homogeneous(ax):
    """Bands of a uniform Al slab should be linear: omega = c_L|k| and c_T|k|."""
    n_max = 3
    N = 32
    n_bands = 6
    n_per_seg = 20

    g_vectors, m_indices = reciprocal_lattice(n_max)
    k_points, k_dist, tick_pos, tick_labels = make_k_path(n_per_seg)

    mask = torch.zeros(N, N, dtype=torch.float64)
    freqs = run_elastic(k_points, g_vectors, m_indices, mask, n_bands,
                        mat_hole=AL_PROPS, mat_matrix=AL_PROPS)

    c_t = np.sqrt(AL_PROPS["mu"] / AL_PROPS["rho"])
    c_l = np.sqrt((AL_PROPS["lam"] + 2 * AL_PROPS["mu"]) / AL_PROPS["rho"])

    # Numerical diagnostic: at the X point k=(pi,0), the lowest eigenvalues
    # should be omega = c_T*pi and c_L*pi  =>  f = c_T*pi/(2pi) = c_T/2
    # normalised: f/c_t = 0.5  and  f/c_t = 0.5*c_l/c_t
    k_X_idx = n_per_seg  # approximate index of X point
    print(f"  c_T={c_t:.1f} m/s, c_L={c_l:.1f} m/s, c_L/c_T={c_l/c_t:.4f}")
    print(f"  At X: computed bands (normalised) = {freqs[k_X_idx].numpy() / c_t}")
    print(f"  Expected lowest two ≈ {0.5:.4f} and {0.5*c_l/c_t:.4f}")

    freqs_norm = freqs.numpy() / c_t

    # Analytical: all folded branches omega = c * |k+G|
    # Collect the full set of folded branches for T and L polarisations
    kpg = k_points[:, None, :] + g_vectors[None, :, :]  # (n_k, n_pw, 2)
    kpg_mag = np.linalg.norm(kpg, axis=-1)              # (n_k, n_pw)
    all_T = np.sort(kpg_mag / (2 * np.pi), axis=-1)     # normalised by c_t
    all_L = np.sort(kpg_mag * (c_l / c_t) / (2 * np.pi), axis=-1)

    # Merge T and L branches, sort per k-point, take lowest n_bands
    all_branches = np.concatenate([all_T, all_L], axis=-1)
    all_branches = np.sort(all_branches, axis=-1)[:, :n_bands]

    for b in range(n_bands):
        col = "b" if b == 0 else "b"
        ax.plot(k_dist, freqs_norm[:, b], "b-", lw=1.8,
                label="PWE" if b == 0 else None)
        ax.plot(k_dist, all_branches[:, b], "r--", lw=0.9,
                label="analytical" if b == 0 else None)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    for tp in tick_pos:
        ax.axvline(tp, color="gray", lw=0.4, ls="--")
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylim(0, 1.5)
    ax.set_ylabel(r"$\omega a / 2\pi c_T$")
    ax.set_title("Homogeneous Al (sanity)")
    ax.legend(fontsize=7, loc="upper left")


# ── Test 2: Ni/Al benchmark ─────────────────────────────────────────

def test_ni_al(ax, n_max=5):
    """Ni cylinders in Al, r/a = 0.375 — moderate contrast benchmark."""
    r = 0.375
    N = 64
    n_bands = 8
    n_per_seg = 25

    mask = make_circle_mask(r, N=N)
    g_vectors, m_indices = reciprocal_lattice(n_max)
    k_points, k_dist, tick_pos, tick_labels = make_k_path(n_per_seg)

    n_pw = len(g_vectors)
    print(f"Ni/Al: r/a={r}, n_max={n_max}, {n_pw} PWs, matrix {2*n_pw}x{2*n_pw}")

    t0 = time.time()
    freqs = run_elastic(k_points, g_vectors, m_indices, mask, n_bands,
                        mat_hole=NI_PROPS, mat_matrix=AL_PROPS)
    dt = time.time() - t0
    print(f"  solved in {dt:.1f}s")

    c_t_al = np.sqrt(AL_PROPS["mu"] / AL_PROPS["rho"])
    freqs_norm = freqs.numpy() / c_t_al

    for b in range(n_bands):
        ax.plot(k_dist, freqs_norm[:, b], "b-", lw=1.2)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    for tp in tick_pos:
        ax.axvline(tp, color="gray", lw=0.4, ls="--")
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylim(0, max(1.5, freqs_norm[:, min(5, n_bands-1)].max() * 1.15))
    ax.set_ylabel(r"$\omega a / 2\pi c_{T,Al}$")
    ax.set_title(f"Ni/Al  r/a={r}  n_max={n_max}")


# ── Test 3: Si/air band structure ───────────────────────────────────

def test_si_air(ax, n_max=5):
    """Si matrix with air holes, r/a = 0.3."""
    r = 0.3
    N = 64
    n_bands = 8
    n_per_seg = 25

    mask = make_circle_mask(r, N=N)
    g_vectors, m_indices = reciprocal_lattice(n_max)
    k_points, k_dist, tick_pos, tick_labels = make_k_path(n_per_seg)

    n_pw = len(g_vectors)
    print(f"Si/air: r/a={r}, n_max={n_max}, {n_pw} PWs, matrix {2*n_pw}x{2*n_pw}")

    t0 = time.time()
    freqs = run_elastic(k_points, g_vectors, m_indices, mask, n_bands,
                        mat_hole=AIR_CORRECTED, mat_matrix=SI_PROPS)
    dt = time.time() - t0
    print(f"  solved in {dt:.1f}s")

    c_t_si = np.sqrt(SI_PROPS["mu"] / SI_PROPS["rho"])
    freqs_norm = freqs.numpy() / c_t_si

    for b in range(n_bands):
        ax.plot(k_dist, freqs_norm[:, b], "b-", lw=1.2)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    for tp in tick_pos:
        ax.axvline(tp, color="gray", lw=0.4, ls="--")
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylim(0, max(1.5, freqs_norm[:, min(5, n_bands-1)].max() * 1.15))
    ax.set_ylabel(r"$\omega a / 2\pi c_{T,Si}$")
    ax.set_title(f"Si/air  r/a={r}  n_max={n_max}")


# ── Test 4: Convergence study ───────────────────────────────────────

def test_convergence(axes):
    """Sweep n_max for Ni/Al and Si/air, plot lowest 4 bands at X point."""
    r = 0.3
    N = 64
    n_bands = 6
    k_X = np.array([[np.pi, 0.0]])

    configs = [
        ("Ni/Al", NI_PROPS, AL_PROPS, np.sqrt(AL_PROPS["mu"] / AL_PROPS["rho"])),
        ("Si/air", AIR_CORRECTED, SI_PROPS, np.sqrt(SI_PROPS["mu"] / SI_PROPS["rho"])),
    ]

    n_max_values = [2, 3, 4, 5, 6, 7]
    mask = make_circle_mask(r, N=N)

    for ax, (label, mat_hole, mat_matrix, c_ref) in zip(axes, configs):
        results = []
        for nm in n_max_values:
            g_vectors, m_indices = reciprocal_lattice(nm)
            n_pw = len(g_vectors)
            print(f"  convergence {label}: n_max={nm}, {n_pw} PWs ...", end="")
            t0 = time.time()
            freqs = run_elastic(k_X, g_vectors, m_indices, mask, n_bands,
                                mat_hole=mat_hole, mat_matrix=mat_matrix)
            dt = time.time() - t0
            f_norm = freqs.numpy()[0] / c_ref
            results.append(f_norm)
            print(f" {dt:.1f}s  bands={np.array2string(f_norm[:4], precision=4)}")

        results = np.array(results)
        for b in range(min(4, n_bands)):
            ax.plot(n_max_values, results[:, b], "o-", ms=4, lw=1.2,
                    label=f"band {b+1}")
        ax.set_xlabel("n_max")
        ax.set_ylabel(r"$\omega a / 2\pi c_T$ at X")
        ax.set_title(f"Convergence: {label}  r/a={r}")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)


# ── Main ────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    print("=== Test 1: Homogeneous Al slab ===")
    test_homogeneous(axes[0, 0])

    print("\n=== Test 2: Ni/Al benchmark (n_max=5) ===")
    test_ni_al(axes[0, 1], n_max=5)

    print("\n=== Test 3: Si/air bands (n_max=5) ===")
    test_si_air(axes[0, 2], n_max=5)

    print("\n=== Test 4: Convergence study ===")
    test_convergence([axes[1, 0], axes[1, 1]])

    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig("elastic_pwe_validation.png", dpi=150)
    plt.close()
    print("\nSaved elastic_pwe_validation.png")


if __name__ == "__main__":
    main()
