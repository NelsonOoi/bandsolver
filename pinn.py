"""
Physics-Informed Neural Network for photonic band structure prediction.

Mode A: Inverse design -- NN generates epsilon from target gap specs,
        trained end-to-end through a differentiable PWE solver.
Mode B: Fast surrogate -- NN predicts band structure from epsilon grid,
        trained with data + eigenvalue residual losses.

Usage:
    python pinn.py --mode inverse --steps 500
    python pinn.py --mode surrogate --steps 2000
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
if matplotlib.get_backend() == "agg":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pwe_torch import (reciprocal_lattice, make_k_path, build_epsilon_matrix,
                        solve_bands, extract_gap, smooth_min, smooth_max)


# ===================================================================
# C4v Symmetry Tiling
# ===================================================================

def c4v_tile(octant: torch.Tensor, N: int) -> torch.Tensor:
    """Expand a triangular octant to a full NxN grid via C4v symmetry.

    The octant covers the upper-right triangle of one quadrant:
    pixels (i, j) with j >= i, for i in [0, half), j in [i, half).

    Steps:
        1. octant -> quadrant (reflect across diagonal j=i)
        2. quadrant -> full grid (reflect horizontally and vertically)

    Args:
        octant: (half, half) tensor -- upper-triangle values filled,
                lower triangle will be mirrored from transpose.
        N: full grid side length (must be even).

    Returns:
        (N, N) tensor with full C4v symmetry.
    """
    half = N // 2
    # 1. Make upper-right quadrant symmetric about diagonal
    upper = torch.triu(octant)
    quadrant = upper + upper.T - torch.diag(torch.diag(upper))

    # 2. Tile to full grid: reflect across both axes
    #    quadrant fills top-left; flip for other three quadrants
    top = torch.cat([quadrant.flip(1), quadrant], dim=1)     # (half, N)
    full = torch.cat([quadrant.flip(0).flip(1),
                      quadrant.flip(0)], dim=1)               # (half, N)
    grid = torch.cat([top, full], dim=0)                      # (N, N)
    return grid


class C4vTiler(nn.Module):
    """Differentiable module wrapping c4v_tile."""

    def __init__(self, N: int):
        super().__init__()
        self.N = N
        self.half = N // 2

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """raw: (batch, 1, half, half) or (half, half) -> (N, N)."""
        squeezed = False
        if raw.dim() == 2:
            raw = raw.unsqueeze(0).unsqueeze(0)
            squeezed = True
        B = raw.shape[0]
        out = []
        for b in range(B):
            out.append(c4v_tile(raw[b, 0], self.N))
        result = torch.stack(out)  # (B, N, N)
        if squeezed:
            return result[0]
        return result


# ===================================================================
# Physics Loss Functions
# ===================================================================

class PhysicsLoss(nn.Module):
    """Combined physics-informed loss for band structure optimization.

    Supports two gap objectives via the `maximize` flag:
        maximize=False (default): target a specific gap width and midgap freq.
        maximize=True: maximize the gap-midgap ratio (no target needed).

    Other terms:
        - binary_pen: mean(s * (1-s)) pushes toward 0/1 designs
        - bloch_pen:  boundary continuity penalty
        - residual:   eigenvalue equation residual |Hv - lv|^2
    """

    def __init__(self, w_gap=1.0, w_freq=1.0, w_binary=0.1,
                 w_bloch=0.01, w_residual=0.0, maximize=False):
        super().__init__()
        self.w_gap = w_gap
        self.w_freq = w_freq
        self.w_binary = w_binary
        self.w_bloch = w_bloch
        self.w_residual = w_residual
        self.maximize = maximize

    def forward(self, bands, eps_grid, eps_norm, target_freq, target_width,
                band_lo, band_hi, H_matrices=None, eigvecs=None, eigvals=None):
        """Compute total physics loss.

        Args:
            bands: (n_k, n_bands) frequencies.
            eps_grid: (N, N) dielectric grid.
            eps_norm: (N, N) normalized epsilon in [0,1].
            target_freq: scalar target midgap frequency (ignored if maximize=True).
            target_width: scalar target gap width (ignored if maximize=True).
            band_lo, band_hi: band indices for gap extraction.
            H_matrices: optional (n_k, n_pw, n_pw) for residual loss.
            eigvecs: optional (n_k, n_pw, n_bands) for residual loss.
            eigvals: optional (n_k, n_bands) for residual loss.

        Returns:
            total_loss, info dict.
        """
        gap_width, midgap = extract_gap(bands, band_lo, band_hi)

        if self.maximize:
            # Maximize gap-midgap ratio: loss = -gap/midgap (clamped to avoid /0)
            safe_mid = torch.clamp(midgap, min=1e-6)
            gap_ratio = gap_width / safe_mid
            loss_gap = -self.w_gap * gap_ratio
            loss_freq = torch.tensor(0.0, dtype=eps_grid.dtype)
        else:
            loss_gap = self.w_gap * (gap_width - target_width) ** 2
            loss_freq = self.w_freq * (midgap - target_freq) ** 2

        # Binary penalty
        loss_binary = self.w_binary * torch.mean(eps_norm * (1.0 - eps_norm))

        # Bloch periodicity penalty: left-right and top-bottom boundary match
        loss_bloch = torch.tensor(0.0, dtype=eps_grid.dtype)
        if self.w_bloch > 0:
            loss_bloch = self.w_bloch * (
                torch.mean((eps_grid[0, :] - eps_grid[-1, :]) ** 2) +
                torch.mean((eps_grid[:, 0] - eps_grid[:, -1]) ** 2)
            )

        # Eigenvalue residual: ||H @ v - lambda * v||^2
        loss_residual = torch.tensor(0.0, dtype=eps_grid.dtype)
        if self.w_residual > 0 and H_matrices is not None:
            Hv = torch.bmm(H_matrices, eigvecs)           # (n_k, n_pw, n_bands)
            lv = eigvals.unsqueeze(1) * eigvecs            # broadcast lambda
            loss_residual = self.w_residual * torch.mean((Hv - lv) ** 2)

        total = loss_gap + loss_freq + loss_binary + loss_bloch + loss_residual

        info = {
            "gap_width": gap_width.item(),
            "midgap": midgap.item(),
            "loss_gap": loss_gap.item(),
            "loss_freq": loss_freq.item(),
            "loss_binary": loss_binary.item(),
            "loss_bloch": loss_bloch.item(),
            "loss_residual": loss_residual.item(),
            "total": total.item(),
        }
        if self.maximize:
            safe_mid_val = max(midgap.item(), 1e-6)
            info["gap_ratio"] = gap_width.item() / safe_mid_val
        return total, info


# ===================================================================
# Mode A: Inverse Design Network
# ===================================================================

class InverseDesignNet(nn.Module):
    """Maps target gap specs to an epsilon grid via CNN decoder + C4v tiling.

    Architecture:
        MLP encoder: (target_freq, target_width, band_lo, band_hi) -> latent z
        CNN decoder: z -> (half, half) octant
        C4v tiler:   octant -> (N, N) full grid
        Softplus:    ensures epsilon > 0
    """

    def __init__(self, N=16, latent_dim=32, eps_bg=1.0, eps_rod=8.9):
        super().__init__()
        self.N = N
        self.half = N // 2
        self.eps_bg = eps_bg
        self.eps_rod = eps_rod

        # Encoder: target specs -> latent
        self.encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Decoder: latent -> octant pixel values
        # Output enough values for a half x half grid
        octant_pixels = self.half * self.half
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, octant_pixels),
        )

        self.tiler = C4vTiler(N)

    def forward(self, target_freq, target_width, band_lo, band_hi):
        """Generate epsilon grid from target specifications.

        All inputs are scalars or 1-element tensors.

        Returns:
            eps_grid: (N, N) dielectric grid.
            eps_norm: (N, N) normalized values in [0, 1].
        """
        specs = torch.stack([
            torch.as_tensor(target_freq, dtype=torch.float64),
            torch.as_tensor(target_width, dtype=torch.float64),
            torch.as_tensor(float(band_lo), dtype=torch.float64),
            torch.as_tensor(float(band_hi), dtype=torch.float64),
        ])
        z = self.encoder(specs)
        raw = self.decoder(z).view(self.half, self.half)

        # Sigmoid for [0, 1] range, then tile with C4v symmetry
        eps_norm = torch.sigmoid(raw)
        eps_norm_full = self.tiler(eps_norm)

        # Map to physical epsilon range
        eps_grid = self.eps_bg + (self.eps_rod - self.eps_bg) * eps_norm_full
        return eps_grid, eps_norm_full


# ===================================================================
# Mode B: Fast Surrogate Network
# ===================================================================

class BandSurrogate(nn.Module):
    """Predicts band frequencies from epsilon grid + k-points.

    Hard constraints baked in:
        - Reciprocity: k enters as (kx^2, ky^2, kx*ky) -- even functions.
        - Non-negative freqs: softplus output.
        - Band ordering: predict increments >= 0, cumsum for ordered bands.

    Architecture:
        Fourier encoder: eps_grid -> FFT features -> MLP -> embedding
        k encoder: (kx^2, ky^2, kx*ky) -> MLP -> embedding
        Trunk MLP: concatenated embeddings -> n_bands increments -> cumsum
    """

    def __init__(self, n_pw_features=64, n_bands=6, hidden=256):
        super().__init__()
        self.n_bands = n_bands

        # Epsilon encoder: takes flattened |FFT coefficients|
        self.eps_encoder = nn.Sequential(
            nn.Linear(n_pw_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # k encoder: takes (kx^2, ky^2, kx*ky) -- 3 even features
        self.k_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, hidden),
            nn.ReLU(),
        )

        # Trunk: combined -> band increments
        self.trunk = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_bands),
        )

    def forward(self, eps_fft_features, k_points_np):
        """Predict bands for all k-points given epsilon FFT features.

        Args:
            eps_fft_features: (n_feat,) tensor of |FFT| magnitudes.
            k_points_np: (n_k, 2) numpy array of k-points.

        Returns:
            freqs: (n_k, n_bands) predicted frequencies.
        """
        k_pts = torch.from_numpy(k_points_np).to(torch.float64)
        n_k = k_pts.shape[0]

        # Even-function encoding of k (enforces reciprocity)
        k_even = torch.stack([
            k_pts[:, 0] ** 2,
            k_pts[:, 1] ** 2,
            k_pts[:, 0] * k_pts[:, 1],
        ], dim=-1)  # (n_k, 3)

        # Encode epsilon (same for all k)
        eps_emb = self.eps_encoder(eps_fft_features)  # (hidden,)
        eps_emb = eps_emb.unsqueeze(0).expand(n_k, -1)  # (n_k, hidden)

        # Encode k
        k_emb = self.k_encoder(k_even)  # (n_k, hidden)

        # Combine and predict increments
        combined = torch.cat([eps_emb, k_emb], dim=-1)  # (n_k, 2*hidden)
        increments = F.softplus(self.trunk(combined))    # (n_k, n_bands), >= 0

        # Cumsum enforces band ordering
        freqs = torch.cumsum(increments, dim=-1)
        return freqs

    def extract_features(self, eps_grid, m_indices_np, n_features):
        """Extract Fourier features from epsilon grid for network input."""
        N = eps_grid.shape[0]
        eps_fft = torch.fft.fft2(eps_grid.to(torch.complex128)) / (N * N)
        magnitudes = torch.abs(eps_fft).flatten()
        # Take top-n features sorted by magnitude
        if magnitudes.shape[0] > n_features:
            _, idx = torch.topk(magnitudes, n_features)
            idx, _ = torch.sort(idx)  # preserve spatial order
            return magnitudes[idx]
        return magnitudes


# ===================================================================
# Training: Mode A -- Inverse Design
# ===================================================================

def train_inverse(cfg):
    """Train the inverse design network end-to-end through PWE solver."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    g_vectors, m_indices = reciprocal_lattice(cfg.n_max)
    k_points, k_dist, tick_pos, tick_labels = make_k_path(cfg.n_k_seg)

    model = InverseDesignNet(
        N=cfg.n_grid, latent_dim=cfg.latent_dim,
        eps_bg=cfg.eps_bg, eps_rod=cfg.eps_rod,
    ).double()

    physics_loss = PhysicsLoss(
        w_gap=cfg.w_gap, w_freq=cfg.w_freq, w_binary=cfg.w_binary,
        w_bloch=cfg.w_bloch, w_residual=0.0, maximize=cfg.maximize,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.steps)

    # Binary penalty warmup schedule
    binary_schedule = np.linspace(cfg.w_binary * 0.1, cfg.w_binary, cfg.steps)

    history = []
    obj = "maximize gap" if cfg.maximize else f"freq={cfg.target_freq:.4f}  width={cfg.target_width:.4f}"
    print(f"Mode A: Inverse design | Grid: {cfg.n_grid} | Steps: {cfg.steps}")
    print(f"Objective: {obj}  bands {cfg.band_lo}-{cfg.band_hi}")
    print("-" * 60)

    for step in range(cfg.steps):
        t0 = time.time()
        optimizer.zero_grad()

        # Warmup binary penalty
        physics_loss.w_binary = float(binary_schedule[step])

        # Forward: specs -> epsilon -> bands
        eps_grid, eps_norm = model(
            cfg.target_freq, cfg.target_width, cfg.band_lo, cfg.band_hi
        )

        bands = solve_bands(k_points, g_vectors, eps_grid, m_indices,
                            cfg.n_bands, "tm")

        loss, info = physics_loss(
            bands, eps_grid, eps_norm,
            cfg.target_freq, cfg.target_width, cfg.band_lo, cfg.band_hi,
        )

        loss.backward()

        # Skip step if gradients are NaN (degenerate eigenvalues)
        has_nan = any(
            p.grad is not None and torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        if has_nan or torch.isnan(loss):
            optimizer.zero_grad()
            scheduler.step()
            elapsed = time.time() - t0
            info["step"] = step
            info["time"] = elapsed
            history.append(info)
            if (step + 1) % max(1, cfg.steps // 20) == 0:
                print(f"[{step+1:4d}/{cfg.steps}] NaN detected, skipping step")
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t0
        info["step"] = step
        info["time"] = elapsed
        history.append(info)

        if (step + 1) % max(1, cfg.steps // 20) == 0 or step == 0:
            extra = f"  ratio={info['gap_ratio']:.5f}" if "gap_ratio" in info else ""
            print(f"[{step+1:4d}/{cfg.steps}] loss={info['total']:.6f}  "
                  f"gap={info['gap_width']:.5f}  midgap={info['midgap']:.5f}{extra}  "
                  f"bin={info['loss_binary']:.5f}  ({elapsed:.2f}s)")

    # Final evaluation and plot
    with torch.no_grad():
        eps_grid, eps_norm = model(
            cfg.target_freq, cfg.target_width, cfg.band_lo, cfg.band_hi
        )
        bands = solve_bands(k_points, g_vectors, eps_grid, m_indices,
                            cfg.n_bands, "tm")

    _plot_inverse(eps_grid, bands, k_dist, tick_pos, tick_labels,
                  cfg.band_lo, cfg.band_hi, history, cfg)


# ===================================================================
# Training: Mode B -- Surrogate
# ===================================================================

def train_surrogate(cfg):
    """Train the surrogate network with PWE ground truth + physics losses."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    g_vectors, m_indices = reciprocal_lattice(cfg.n_max)
    k_points, k_dist, tick_pos, tick_labels = make_k_path(cfg.n_k_seg)
    n_pw = (2 * cfg.n_max + 1) ** 2
    n_feat = min(cfg.n_fourier_features, cfg.n_grid * cfg.n_grid)

    model = BandSurrogate(
        n_pw_features=n_feat, n_bands=cfg.n_bands, hidden=cfg.hidden_dim,
    ).double()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.steps)

    history = []
    print(f"Mode B: Surrogate | Grid: {cfg.n_grid} | Steps: {cfg.steps} | "
          f"Features: {n_feat}")
    print("-" * 60)

    for step in range(cfg.steps):
        t0 = time.time()
        optimizer.zero_grad()

        # Generate random epsilon grid for training
        eps_np = _random_epsilon_grid(cfg.n_grid, cfg.eps_bg, cfg.eps_rod)
        eps_grid = torch.from_numpy(eps_np).double()

        # Ground truth from PWE solver
        with torch.no_grad():
            bands_true = solve_bands(
                k_points, g_vectors, eps_grid, m_indices, cfg.n_bands, "tm"
            )

        # Extract features and predict
        features = model.extract_features(eps_grid, m_indices, n_feat)
        bands_pred = model(features, k_points)

        # Data loss (MSE against PWE ground truth)
        loss_data = F.mse_loss(bands_pred, bands_true)

        # Variational bound: predictions should not be below ground truth
        # (Rayleigh quotient gives upper bounds, so penalize undershoot)
        violations = F.relu(bands_true - bands_pred)
        loss_variational = cfg.w_variational * torch.mean(violations ** 2)

        # Reciprocity check loss: omega(k) == omega(-k)
        # Already enforced by architecture (even-function inputs), but add
        # soft check for robustness
        k_neg = -k_points
        bands_neg = model(features, k_neg)
        bands_pos = model(features, k_points)
        loss_reciprocity = cfg.w_reciprocity * F.mse_loss(bands_pos, bands_neg)

        loss = loss_data + loss_variational + loss_reciprocity

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t0
        info = {
            "step": step,
            "loss_data": loss_data.item(),
            "loss_variational": loss_variational.item(),
            "loss_reciprocity": loss_reciprocity.item(),
            "total": loss.item(),
            "time": elapsed,
        }
        history.append(info)

        if (step + 1) % max(1, cfg.steps // 20) == 0 or step == 0:
            print(f"[{step+1:4d}/{cfg.steps}] loss={info['total']:.6f}  "
                  f"data={info['loss_data']:.6f}  var={info['loss_variational']:.6f}  "
                  f"({elapsed:.2f}s)")

    # Final evaluation plot
    with torch.no_grad():
        eps_np = _random_epsilon_grid(cfg.n_grid, cfg.eps_bg, cfg.eps_rod, seed=0)
        eps_grid = torch.from_numpy(eps_np).double()
        bands_true = solve_bands(
            k_points, g_vectors, eps_grid, m_indices, cfg.n_bands, "tm"
        )
        features = model.extract_features(eps_grid, m_indices, n_feat)
        bands_pred = model(features, k_points)

    _plot_surrogate(eps_grid, bands_true, bands_pred, k_dist, tick_pos,
                    tick_labels, history, cfg)


# ===================================================================
# Helpers
# ===================================================================

def _random_epsilon_grid(N, eps_bg, eps_rod, seed=None):
    """Generate random binary epsilon grids with circular/square features."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    xs = np.linspace(0, 1, N, endpoint=False) + 0.5 / N
    x, y = np.meshgrid(xs, xs, indexing="ij")

    # Random circle
    r = rng.uniform(0.1, 0.4)
    cx, cy = 0.5, 0.5
    mask = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= r
    return np.where(mask, eps_rod, eps_bg)


# ===================================================================
# Plotting
# ===================================================================

def _plot_inverse(eps_grid, bands, k_dist, tick_pos, tick_labels,
                  band_lo, band_hi, history, cfg):
    """Plot inverse design results."""
    eps_np = eps_grid.detach().cpu().numpy()
    bands_np = bands.detach().cpu().numpy()

    # Use actual min/max of bands for accurate gap display
    floor_val = float(np.max(bands_np[:, band_lo]))
    ceil_val = float(np.min(bands_np[:, band_hi]))
    gap_width = max(0.0, ceil_val - floor_val)
    midgap = 0.5 * (ceil_val + floor_val)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    im = ax.imshow(eps_np.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdYlBu_r")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("PINN unit cell")
    ax.set_xlabel("x/a"); ax.set_ylabel("y/a"); ax.set_aspect("equal")

    ax = axes[1]
    for i in range(bands_np.shape[1]):
        ax.plot(k_dist, bands_np[:, i], color="#2563eb", linewidth=0.9)
    for tp in tick_pos:
        ax.axvline(tp, color="grey", linewidth=0.4, linestyle="--")
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels)
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylim(0, max(0.8, midgap + gap_width + 0.1))
    ax.set_ylabel("$\\omega a / 2\\pi c$"); ax.set_title("TM bands (PINN)")
    if gap_width > 0:
        ax.axhspan(floor_val, ceil_val, alpha=0.15, color="#2563eb")
    ax.axhline(cfg.target_freq, color="red", linewidth=0.7, linestyle=":")

    ax = axes[2]
    steps = [h["step"] for h in history]
    ax.plot(steps, [h["total"] for h in history], label="total")
    ax.plot(steps, [h["loss_gap"] for h in history], label="gap", alpha=0.7)
    ax.plot(steps, [h["loss_freq"] for h in history], label="freq", alpha=0.7)
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Convergence")
    ax.legend(fontsize=8); ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig("pinn_inverse.png", dpi=150)
    print(f"Saved pinn_inverse.png")
    plt.close(fig)


def _plot_surrogate(eps_grid, bands_true, bands_pred, k_dist, tick_pos,
                    tick_labels, history, cfg):
    """Plot surrogate comparison."""
    eps_np = eps_grid.detach().cpu().numpy()
    true_np = bands_true.detach().cpu().numpy()
    pred_np = bands_pred.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    im = ax.imshow(eps_np.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdYlBu_r")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Test unit cell")
    ax.set_xlabel("x/a"); ax.set_ylabel("y/a"); ax.set_aspect("equal")

    ax = axes[1]
    for i in range(true_np.shape[1]):
        ax.plot(k_dist, true_np[:, i], color="#2563eb", linewidth=1.2,
                label="PWE" if i == 0 else None)
        ax.plot(k_dist, pred_np[:, i], color="#dc2626", linewidth=0.9,
                linestyle="--", label="PINN" if i == 0 else None)
    for tp in tick_pos:
        ax.axvline(tp, color="grey", linewidth=0.4, linestyle="--")
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels)
    ax.set_xlim(k_dist[0], k_dist[-1]); ax.set_ylim(0, 0.8)
    ax.set_ylabel("$\\omega a / 2\\pi c$"); ax.set_title("PWE vs PINN")
    ax.legend(fontsize=8)

    ax = axes[2]
    steps = [h["step"] for h in history]
    ax.plot(steps, [h["total"] for h in history], label="total")
    ax.plot(steps, [h["loss_data"] for h in history], label="data", alpha=0.7)
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Surrogate loss")
    ax.legend(fontsize=8); ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig("pinn_surrogate.png", dpi=150)
    print(f"Saved pinn_surrogate.png")
    plt.close(fig)


# ===================================================================
# CLI
# ===================================================================

def main():
    p = argparse.ArgumentParser(description="Physics-Informed NN for photonic bands")
    p.add_argument("--mode", choices=["inverse", "surrogate"], default="inverse")
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
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-fourier-features", type=int, default=64)
    p.add_argument("--maximize", action="store_true",
                   help="Maximize gap-midgap ratio instead of targeting specific values")
    p.add_argument("--w-gap", type=float, default=1.0)
    p.add_argument("--w-freq", type=float, default=1.0)
    p.add_argument("--w-binary", type=float, default=0.1)
    p.add_argument("--w-bloch", type=float, default=0.01)
    p.add_argument("--w-variational", type=float, default=0.1)
    p.add_argument("--w-reciprocity", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    cfg = p.parse_args()

    if cfg.mode == "inverse":
        train_inverse(cfg)
    else:
        train_surrogate(cfg)


if __name__ == "__main__":
    main()

