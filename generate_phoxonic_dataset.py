"""
Generate phoxonic dataset: (geometry, TE bands, elastic bands) pairs
on random C6v silicon/air hexagonal-lattice geometries.

Usage:
    python generate_phoxonic_dataset.py --n-samples 10000 --grid-size 16 --n-max 5
    python generate_phoxonic_dataset.py --n-samples 100 --grid-size 16 --n-max 3  # quick test
"""

import argparse
import time
import numpy as np
import torch

from pwe_hex_torch import (hex_reciprocal_lattice, hex_make_k_path,
                           hex_solve_bands, generate_c6v_eps,
                           make_hex_rod_epsilon, find_best_gap,
                           A1, A2, _oblique_coords, _oblique_to_cartesian)
from pwe_phononic_hex_torch import (solve_phononic_bands, SI_MATERIALS)


def generate_parametric_geometries(N, n_rods=50, n_hex=30, n_annuli=20):
    """Parametric geometries known to produce gaps."""
    samples = []

    # Circular rods at varying r/a
    for r_over_a in np.linspace(0.1, 0.45, n_rods):
        eps = make_hex_rod_epsilon(N, r_over_a, 11.7, 1.0)
        s = (eps > 5.0).to(torch.float64)
        samples.append(s)

    # Hexagonal rods
    s1, s2 = _oblique_coords(N)
    x, y = _oblique_to_cartesian(s1, s2)
    cx = 0.5 * (A1[0] + A2[0])
    cy = 0.5 * (A1[1] + A2[1])
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    for hex_r in np.linspace(0.1, 0.4, n_hex):
        # Hexagonal mask: r_eff = r / cos(theta_fold - pi/6)
        theta_fold = theta % (np.pi / 3.0)
        r_hex = hex_r / np.maximum(np.cos(theta_fold - np.pi / 6.0), 0.5)
        s = torch.tensor((r <= r_hex).astype(np.float64))
        samples.append(s)

    # Annular rods
    for r_in, r_out in zip(np.linspace(0.05, 0.2, n_annuli),
                           np.linspace(0.2, 0.45, n_annuli)):
        s = torch.tensor(((r >= r_in) & (r <= r_out)).astype(np.float64))
        samples.append(s)

    return samples


def process_sample(s_grid, g_vectors, m_indices, k_points,
                   n_bands_em, n_bands_ac, mat_si, mat_air):
    """Run both solvers on a single geometry. Returns dict or None on failure."""
    N = s_grid.shape[0]

    eps_grid = mat_air["eps"] + (mat_si["eps"] - mat_air["eps"]) * s_grid

    try:
        bands_em = hex_solve_bands(k_points, g_vectors, eps_grid,
                                   m_indices, n_bands_em, "te")
        bands_em_np = bands_em.detach().numpy()
    except Exception:
        return None

    try:
        bands_ac = solve_phononic_bands(k_points, g_vectors, s_grid,
                                        m_indices, n_bands_ac,
                                        mat_si, mat_air)
        bands_ac_np = bands_ac.detach().numpy()
    except Exception:
        return None

    if np.any(np.isnan(bands_em_np)) or np.any(np.isnan(bands_ac_np)):
        return None

    gw_em, gm_em, gp_em = find_best_gap(bands_em_np)
    gw_ac, gm_ac, gp_ac = find_best_gap(bands_ac_np)

    fill = float(s_grid.mean())

    # FFT features: top-K magnitudes
    fft_2d = torch.fft.fft2(s_grid) / (N * N)
    fft_mags = torch.abs(fft_2d).flatten()
    fft_mags_sorted, _ = torch.sort(fft_mags, descending=True)
    top_k = min(32, fft_mags_sorted.shape[0])
    fft_feats = fft_mags_sorted[:top_k].numpy()

    return {
        "s_grid": s_grid.numpy().astype(np.float32),
        "fill_fraction": fill,
        "fft_coeffs": fft_feats.astype(np.float32),
        "bands_em": bands_em_np.astype(np.float32),
        "bands_ac": bands_ac_np.astype(np.float32),
        "best_gap_em": float(gw_em),
        "mid_em": float(gm_em),
        "pair_em": int(gp_em),
        "best_gap_ac": float(gw_ac),
        "mid_ac": float(gm_ac),
        "pair_ac": int(gp_ac),
        "has_dual_gap": bool(gw_em > 0.01 and gw_ac > 0.01),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate phoxonic dataset")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--n-max", type=int, default=5)
    parser.add_argument("--n-bands-em", type=int, default=8)
    parser.add_argument("--n-bands-ac", type=int, default=8)
    parser.add_argument("--k-segments", type=int, default=10)
    parser.add_argument("--output", type=str, default="phoxonic_dataset.pt")
    parser.add_argument("--fill-range", type=float, nargs=2, default=[0.1, 0.9])
    args = parser.parse_args()

    N = args.grid_size
    g_vectors, m_indices = hex_reciprocal_lattice(args.n_max)
    k_points, k_dist, tick_pos, tick_labels = hex_make_k_path(args.k_segments)

    mat_si = SI_MATERIALS["silicon"]
    mat_air = SI_MATERIALS["air"]

    print(f"PWE: n_pw={g_vectors.shape[0]}, n_k={k_points.shape[0]}, "
          f"grid={N}x{N}")

    # Parametric geometries
    param_geoms = generate_parametric_geometries(N)
    n_param = len(param_geoms)

    # Random C6v geometries
    n_random = max(0, args.n_samples - n_param)
    fill_lo, fill_hi = args.fill_range

    print(f"Generating {n_param} parametric + {n_random} random C6v geometries")

    dataset = []
    t0 = time.time()
    failed = 0

    for i in range(args.n_samples):
        if i < n_param:
            s = param_geoms[i]
        else:
            # Random C6v with varying fill fraction
            p = np.random.uniform(fill_lo, fill_hi)
            from pwe_hex_torch import c6v_tile
            n_indep = (N * N) // 12 + 1
            wedge = torch.bernoulli(p * torch.ones(n_indep, dtype=torch.float64))
            s = c6v_tile(wedge, N)

        result = process_sample(s, g_vectors, m_indices, k_points,
                                args.n_bands_em, args.n_bands_ac,
                                mat_si, mat_air)

        if result is None:
            failed += 1
            continue

        dataset.append(result)

        if (i + 1) % 100 == 0 or i == args.n_samples - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (args.n_samples - i - 1) / rate
            n_dual = sum(1 for d in dataset if d["has_dual_gap"])
            print(f"[{i+1}/{args.n_samples}] {rate:.1f} samples/s, "
                  f"ETA {eta:.0f}s, dual gaps: {n_dual}/{len(dataset)}, "
                  f"failed: {failed}")

    print(f"\nDone: {len(dataset)} valid samples, {failed} failed")
    n_dual = sum(1 for d in dataset if d["has_dual_gap"])
    print(f"Dual-gap samples: {n_dual} ({100*n_dual/max(1,len(dataset)):.1f}%)")

    # Save
    save_dict = {
        "samples": dataset,
        "k_points": k_points,
        "k_dist": k_dist,
        "tick_positions": tick_pos,
        "tick_labels": tick_labels,
        "g_vectors": g_vectors,
        "m_indices": m_indices,
        "grid_size": N,
        "n_max": args.n_max,
    }
    torch.save(save_dict, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
