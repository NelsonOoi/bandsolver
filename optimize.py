"""
Finite-difference inverse design for targeted photonic band gaps.

Optimizes a pixelated dielectric unit cell to achieve a specified band gap
center frequency and width using the PWE forward solver from pwe.py.

Usage:
    python optimize.py [options]

Example:
    python optimize.py --target-freq 0.35 --target-width 0.05 --band-lo 0 --band-hi 1 --steps 80
"""

import argparse
import time

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pwe import (reciprocal_lattice, solve_bands, make_k_path,
                 build_epsilon_matrix, _solve_bands_from_inv)

# Note: fd_gradient removed in favor of adjoint_gradient (eigenvalue perturbation).
# Adjoint computes exact gradients in ONE forward solve vs N*N eigensolves for FD.


# ---------------------------------------------------------------------------
# 1. Smooth min / max
# ---------------------------------------------------------------------------

def smooth_min(x, beta=-50.0):
    """Differentiable approximation to min via softmax weighting."""
    w = _softmax(beta * x)
    return float(np.sum(x * w))


def smooth_max(x, beta=50.0):
    """Differentiable approximation to max via softmax weighting."""
    w = _softmax(beta * x)
    return float(np.sum(x * w))


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


# ---------------------------------------------------------------------------
# 2. Sigmoid parameterization
# ---------------------------------------------------------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


def z_to_eps(z, eps_bg, eps_rod, blur_sigma=0.0):
    """Map unconstrained design variables to an epsilon grid."""
    zb = gaussian_filter(z, sigma=blur_sigma) if blur_sigma > 0 else z
    eps_norm = sigmoid(zb)
    eps_grid = eps_bg + (eps_rod - eps_bg) * eps_norm
    return eps_grid, eps_norm


# ---------------------------------------------------------------------------
# 3. Band gap extraction & loss
# ---------------------------------------------------------------------------

def extract_gap(bands, band_lo, band_hi, beta=50.0):
    """Extract gap width and midgap frequency between two bands."""
    floor = smooth_max(bands[:, band_lo], beta=beta)
    ceil = smooth_min(bands[:, band_hi], beta=-beta)
    return ceil - floor, 0.5 * (ceil + floor)


def _loss_from_bands(bands, band_lo, band_hi, eps_norm,
                     target_width, target_freq, w_gap, w_freq, lambda_binary):
    """Scalar loss from precomputed bands and eps_norm."""
    gap_width, midgap = extract_gap(bands, band_lo, band_hi)
    return (w_gap * (gap_width - target_width) ** 2
            + w_freq * (midgap - target_freq) ** 2
            + lambda_binary * np.mean(eps_norm * (1.0 - eps_norm)))


def compute_loss(z, target_width, target_freq, band_lo, band_hi,
                 eps_bg, eps_rod, blur_sigma,
                 g_vectors, m_indices, k_points, n_bands,
                 w_gap=1.0, w_freq=1.0, lambda_binary=0.1):
    """Full loss evaluation (used for final reporting)."""
    eps_grid, eps_norm = z_to_eps(z, eps_bg, eps_rod, blur_sigma)
    bands = solve_bands(k_points, g_vectors, eps_grid, m_indices, n_bands, "tm")
    gap_width, midgap = extract_gap(bands, band_lo, band_hi)

    loss_gap = w_gap * (gap_width - target_width) ** 2
    loss_freq = w_freq * (midgap - target_freq) ** 2
    binary_pen = lambda_binary * np.mean(eps_norm * (1.0 - eps_norm))

    return loss_gap + loss_freq + binary_pen, {
        "gap_width": gap_width, "midgap": midgap, "binary_pen": binary_pen,
        "loss_gap": loss_gap, "loss_freq": loss_freq,
    }


# ---------------------------------------------------------------------------
# 4. Adjoint gradient via eigenvalue perturbation theory
# ---------------------------------------------------------------------------

def _smooth_grad(x, beta, f_val):
    """Gradient of smooth_min/max(x) w.r.t. each x_i.

    For f(x) = sum_i x_i * softmax(beta*x)_i,
    df/dx_j = w_j * (1 + beta * (x_j - f(x)))
    """
    w = _softmax(beta * x)
    return w * (1.0 + beta * (x - f_val))


def adjoint_gradient(z, eps_bg, eps_rod, blur_sigma,
                     g_vectors, m_indices, k_points, n_bands,
                     band_lo, band_hi, target_width, target_freq,
                     w_gap, w_freq, lambda_binary, beta=50.0):
    """Exact gradient via eigenvalue perturbation theorem (one forward solve).

    Uses d(lambda_n)/d(eps_grid) = -(1/N^2) * |W_n(x,y)|^2
    where W_n is the IFFT of (eps_mat_inv @ D @ v_n) placed on the G-grid.
    Total cost: 1 eigensolve + O(n_k * n_bands) FFTs.
    """
    N = z.shape[0]

    # --- forward pass with eigenvectors ---
    eps_grid, eps_norm = z_to_eps(z, eps_bg, eps_rod, blur_sigma)
    eps_mat = build_epsilon_matrix(eps_grid, m_indices)
    eps_mat_inv = np.linalg.inv(eps_mat)
    freqs, vecs, kpg_norm = _solve_bands_from_inv(
        k_points, g_vectors, eps_mat_inv, n_bands, "tm", return_vecs=True
    )
    # freqs: (n_k, n_bands), vecs: (n_k, n_pw, n_bands), kpg_norm: (n_k, n_pw)

    # --- loss and its gradient w.r.t. frequencies ---
    floor_val = smooth_max(freqs[:, band_lo], beta=beta)
    ceil_val = smooth_min(freqs[:, band_hi], beta=-beta)
    gap_width = ceil_val - floor_val
    midgap = 0.5 * (ceil_val + floor_val)

    loss_gap = w_gap * (gap_width - target_width) ** 2
    loss_freq = w_freq * (midgap - target_freq) ** 2
    binary_pen = lambda_binary * np.mean(eps_norm * (1.0 - eps_norm))
    loss = loss_gap + loss_freq + binary_pen

    dl_dgap = 2.0 * w_gap * (gap_width - target_width)
    dl_dmid = 2.0 * w_freq * (midgap - target_freq)
    dl_dceil = dl_dgap + 0.5 * dl_dmid
    dl_dfloor = -dl_dgap + 0.5 * dl_dmid

    # gradient of smooth min/max w.r.t. freq arrays
    dceil_dfreq = _smooth_grad(freqs[:, band_hi], -beta, ceil_val)   # (n_k,)
    dfloor_dfreq = _smooth_grad(freqs[:, band_lo], beta, floor_val)  # (n_k,)

    # dl/d(freq[k, band]) — only band_lo and band_hi have nonzero gradients
    dl_dfreq = np.zeros_like(freqs)  # (n_k, n_bands)
    dl_dfreq[:, band_lo] = dl_dfloor * dfloor_dfreq
    dl_dfreq[:, band_hi] = dl_dceil * dceil_dfreq

    # freq = sqrt(lambda) / (2*pi), so dl/dlambda = dl/dfreq * 1/(4*pi*freq)
    # guard against zero freq
    safe_freqs = np.where(freqs > 1e-10, freqs, 1e-10)
    dl_dlambda = dl_dfreq / (4.0 * np.pi * safe_freqs)  # (n_k, n_bands)

    # --- eigenvalue perturbation: d(lambda_n)/d(eps_grid[x,y]) ---
    # For TM: H = D * eps_inv * D, so dH/d(eps_inv) = D * d(eps_inv) * D
    # d(eps_inv)/d(eps_grid[x,y]) = -eps_inv * d(eps_mat)/d(eps_grid[x,y]) * eps_inv
    # d(eps_mat)/d(eps_grid[x,y]) = (1/N^2) * u(x,y) * u(x,y)^H  (rank-1)
    #
    # => d(lambda_n)/d(eps_grid[x,y]) = -1/N^2 * |u^H * eps_inv * D * v_n|^2
    #
    # Compute w_n = eps_inv * D * v_n for each (k, n), then IFFT to get
    # u^H * w_n for all (x,y) simultaneously.

    dl_deps = np.zeros((N, N))  # accumulate gradient

    for ki in range(k_points.shape[0]):
        D_k = kpg_norm[ki]  # (n_pw,)
        for ni in range(n_bands):
            if abs(dl_dlambda[ki, ni]) < 1e-16:
                continue
            v = vecs[ki, :, ni]             # (n_pw,)
            w = eps_mat_inv @ (D_k * v)     # (n_pw,)

            # Place w into N*N grid and IFFT to get u^H*w at all pixels
            W_grid = np.zeros((N, N), dtype=complex)
            for a in range(m_indices.shape[0]):
                mi, mj = int(m_indices[a, 0]) % N, int(m_indices[a, 1]) % N
                W_grid[mi, mj] += w[a]
            W_real = np.fft.ifft2(W_grid) * (N * N)  # inverse DFT, undo 1/N^2

            # d(lambda)/d(eps_grid) = -(1/N^2) * |W_real|^2
            dl_deps += dl_dlambda[ki, ni] * (-1.0 / (N * N)) * np.abs(W_real) ** 2

    # --- chain through sigmoid parameterization ---
    # eps_grid = eps_bg + (eps_rod - eps_bg) * sigmoid(blur(z))
    # d(eps)/d(z) = (eps_rod - eps_bg) * sigmoid'(blur(z)) * blur_kernel
    zb = gaussian_filter(z, sigma=blur_sigma) if blur_sigma > 0 else z
    sig = sigmoid(zb)
    sig_deriv = sig * (1.0 - sig)  # sigmoid'
    deps_dz_local = (eps_rod - eps_bg) * sig_deriv  # (N, N)

    # dl/dz = dl/deps * deps/dz, where deps/dz includes blur
    dl_dz_pre = dl_deps * deps_dz_local
    if blur_sigma > 0:
        dl_dz = gaussian_filter(dl_dz_pre, sigma=blur_sigma)
    else:
        dl_dz = dl_dz_pre

    # add binary penalty gradient: d/dz [lambda * mean(s*(1-s))]
    # = lambda * mean((1-2s) * s' * blur_kernel)
    bin_grad_pre = lambda_binary * (1.0 - 2.0 * sig) * sig_deriv / (N * N)
    if blur_sigma > 0:
        dl_dz += gaussian_filter(bin_grad_pre, sigma=blur_sigma)
    else:
        dl_dz += bin_grad_pre

    info = {"gap_width": gap_width, "midgap": midgap, "binary_pen": binary_pen,
            "loss_gap": loss_gap, "loss_freq": loss_freq}
    return dl_dz, loss, info


# ---------------------------------------------------------------------------
# 6. Adam optimizer
# ---------------------------------------------------------------------------

class Adam:
    """Minimal Adam optimizer in NumPy."""

    def __init__(self, shape, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, z, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return z - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# 7. Optimization loop
# ---------------------------------------------------------------------------

def optimize(cfg):
    """Run the full optimization and save results."""
    np.random.seed(cfg.seed)

    g_vectors, m_indices = reciprocal_lattice(cfg.n_max)
    k_points, k_dist, tick_pos, tick_labels = make_k_path(cfg.n_k_seg)
    n_pw = (2 * cfg.n_max + 1) ** 2

    print(f"Grid: {cfg.n_grid}x{cfg.n_grid}  PWs: {n_pw}  "
          f"k-points: {len(k_points)}  bands: {cfg.n_bands}")
    print(f"Target: freq={cfg.target_freq:.4f}  width={cfg.target_width:.4f}  "
          f"bands {cfg.band_lo}-{cfg.band_hi}")
    print(f"Steps: {cfg.steps}  lr: {cfg.lr}  method: adjoint")
    print("-" * 60)

    z = np.random.randn(cfg.n_grid, cfg.n_grid) * 0.5
    lambda_schedule = np.linspace(cfg.lambda_binary * 0.1, cfg.lambda_binary, cfg.steps)
    optimizer = Adam((cfg.n_grid, cfg.n_grid), lr=cfg.lr)

    history = []
    for step in range(cfg.steps):
        t0 = time.time()
        lam = float(lambda_schedule[step])

        grad, loss, info = adjoint_gradient(
            z, cfg.eps_bg, cfg.eps_rod, cfg.blur_sigma,
            g_vectors, m_indices, k_points, cfg.n_bands,
            cfg.band_lo, cfg.band_hi, cfg.target_width, cfg.target_freq,
            cfg.w_gap, cfg.w_freq, lam,
        )
        z = optimizer.step(z, grad)

        elapsed = time.time() - t0
        history.append({**info, "loss": loss, "step": step, "time": elapsed})

        print(f"[{step+1:3d}/{cfg.steps}] loss={loss:.6f}  "
              f"gap={info['gap_width']:.5f}  midgap={info['midgap']:.5f}  "
              f"bin={info['binary_pen']:.5f}  ({elapsed:.1f}s)")

    # Final plot
    eps_grid, _ = z_to_eps(z, cfg.eps_bg, cfg.eps_rod, cfg.blur_sigma)
    bands = solve_bands(k_points, g_vectors, eps_grid, m_indices, cfg.n_bands, "tm")
    gap_width, midgap = extract_gap(bands, cfg.band_lo, cfg.band_hi)

    print("-" * 60)
    if midgap > 1e-6:
        print(f"Final: gap_width={gap_width:.5f}  midgap={midgap:.5f}  "
              f"gap/midgap={gap_width/midgap:.5f}")

    _plot_result(eps_grid, bands, k_dist, tick_pos, tick_labels,
                 cfg.band_lo, cfg.band_hi, gap_width, midgap, history, cfg)


def _plot_result(eps_grid, bands, k_dist, tick_pos, tick_labels,
                 band_lo, band_hi, gap_width, midgap, history, cfg):
    """Save a summary figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    im = ax.imshow(eps_grid.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdYlBu_r")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Optimized unit cell")
    ax.set_xlabel("x/a"); ax.set_ylabel("y/a"); ax.set_aspect("equal")

    ax = axes[1]
    for i in range(bands.shape[1]):
        ax.plot(k_dist, bands[:, i], color="#2563eb", linewidth=0.9)
    for tp in tick_pos:
        ax.axvline(tp, color="grey", linewidth=0.4, linestyle="--")
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels)
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylim(0, max(0.8, midgap + gap_width))
    ax.set_ylabel("$\\omega a / 2\\pi c$"); ax.set_title("TM bands")
    if gap_width > 0:
        ax.axhspan(midgap - gap_width / 2, midgap + gap_width / 2,
                    alpha=0.15, color="#2563eb")
    ax.axhline(cfg.target_freq, color="red", linewidth=0.7, linestyle=":")
    ax.axhline(cfg.target_freq - cfg.target_width / 2, color="red",
               linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(cfg.target_freq + cfg.target_width / 2, color="red",
               linewidth=0.5, linestyle="--", alpha=0.5)

    ax = axes[2]
    steps = [h["step"] for h in history]
    ax.plot(steps, [h["loss"] for h in history], label="total")
    ax.plot(steps, [h["loss_gap"] for h in history], label="gap", alpha=0.7)
    ax.plot(steps, [h["loss_freq"] for h in history], label="freq", alpha=0.7)
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Convergence")
    ax.legend(fontsize=8); ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig("optimized.png", dpi=150)
    print(f"Saved optimized.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Inverse design of photonic band gaps via FD gradients")
    p.add_argument("--target-freq", type=float, default=0.35)
    p.add_argument("--target-width", type=float, default=0.05)
    p.add_argument("--band-lo", type=int, default=0)
    p.add_argument("--band-hi", type=int, default=1)
    p.add_argument("--eps-rod", type=float, default=8.9)
    p.add_argument("--eps-bg", type=float, default=1.0)
    p.add_argument("--n-grid", type=int, default=16)
    p.add_argument("--n-max", type=int, default=5)
    p.add_argument("--n-bands", type=int, default=6)
    p.add_argument("--n-k-seg", type=int, default=8)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=1e-3,
                   help="(unused, kept for compat)")
    p.add_argument("--blur-sigma", type=float, default=0.5)
    p.add_argument("--w-gap", type=float, default=1.0)
    p.add_argument("--w-freq", type=float, default=1.0)
    p.add_argument("--lambda-binary", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    cfg = p.parse_args()
    optimize(cfg)


if __name__ == "__main__":
    main()
