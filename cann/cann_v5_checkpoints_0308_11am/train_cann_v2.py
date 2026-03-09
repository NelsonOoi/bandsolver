"""
Training script for v2 photonic bandstructure CANNs.

Compares FourierShapeCANN vs LevelSetCANN on shapes expressible by both
parameterizations (perturbed circles). Also trains the v1 FullyConnectedCANN
as a baseline.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pwe_torch import reciprocal_lattice, solve_bands, make_k_path
from cann_v2 import (FourierShapeCANN, LevelSetCANN, FullyConnectedCANN,
                      FourierShapeRasterizer, LevelSetRasterizer, GridCANN)
from train_cann import train_model, plot_losses, print_metrics


# ---------------------------------------------------------------------------
# Hard rasterizers for PWE ground truth (high-res, not differentiable)
# ---------------------------------------------------------------------------

def make_fourier_eps_grid(coeffs: np.ndarray, N: int = 64,
                          eps_bg: float = 8.9, eps_rod: float = 1.0):
    """Rasterize a Fourier boundary shape onto an NxN grid for PWE.

    coeffs: [a_0, a_1, b_1, a_2, b_2, ...] defining
            r(theta) = a_0 + sum_n(a_n cos(n theta) + b_n sin(n theta))
    """
    n_harmonics = (len(coeffs) - 1) // 2
    xs = torch.linspace(-0.5, 0.5, N + 1)[:-1] + 0.5 / N
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    grid_r = torch.sqrt(xx**2 + yy**2)
    grid_theta = torch.atan2(yy, xx)

    r_boundary = torch.full_like(grid_theta, float(coeffs[0]))
    for n in range(1, n_harmonics + 1):
        a_n = coeffs[2 * n - 1]
        b_n = coeffs[2 * n]
        r_boundary = r_boundary + a_n * torch.cos(n * grid_theta) + b_n * torch.sin(n * grid_theta)

    mask = grid_r <= r_boundary
    grid = torch.full((N, N), eps_bg, dtype=torch.float64)
    grid[mask] = eps_rod
    return grid


def make_levelset_eps_grid(rbf_weights: np.ndarray, n_rbf_side: int = 4,
                           N: int = 64, sigma: float = 0.15,
                           eps_bg: float = 8.9, eps_rod: float = 1.0):
    """Rasterize a level-set RBF field onto an NxN grid for PWE."""
    cs = np.linspace(-0.5 + 0.5 / n_rbf_side,
                     0.5 - 0.5 / n_rbf_side, n_rbf_side)
    cy, cx = np.meshgrid(cs, cs, indexing="ij")
    centers = np.stack([cx.ravel(), cy.ravel()], axis=-1)  # (n_rbf, 2)

    xs = torch.linspace(-0.5, 0.5, N + 1)[:-1] + 0.5 / N
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    grid_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).numpy()  # (N*N, 2)

    diff = grid_xy[:, None, :] - centers[None, :, :]  # (N*N, n_rbf, 2)
    rbf_vals = np.exp(-0.5 * np.sum(diff**2, axis=-1) / sigma**2)  # (N*N, n_rbf)

    phi = rbf_vals @ rbf_weights  # (N*N,)
    mask = phi > 0

    grid = torch.full((N, N), eps_bg, dtype=torch.float64)
    grid.view(-1)[torch.from_numpy(mask)] = eps_rod
    return grid


# ---------------------------------------------------------------------------
# Shape sampling
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Standard shape generators (grid-based, not limited to star-convex)
# ---------------------------------------------------------------------------

def make_cross_eps_grid(arm_width: float, arm_length: float, N: int = 64,
                        eps_bg: float = 8.9, eps_rod: float = 1.0):
    """Cross / plus shape centred in the unit cell."""
    xs = torch.linspace(-0.5, 0.5, N + 1)[:-1] + 0.5 / N
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    h_arm = (xx.abs() <= arm_length / 2) & (yy.abs() <= arm_width / 2)
    v_arm = (yy.abs() <= arm_length / 2) & (xx.abs() <= arm_width / 2)
    grid = torch.full((N, N), eps_bg, dtype=torch.float64)
    grid[h_arm | v_arm] = eps_rod
    return grid


def make_square_eps_grid(side: float, N: int = 64,
                         eps_bg: float = 8.9, eps_rod: float = 1.0):
    """Square rod centred in the unit cell."""
    xs = torch.linspace(-0.5, 0.5, N + 1)[:-1] + 0.5 / N
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    mask = (xx.abs() <= side / 2) & (yy.abs() <= side / 2)
    grid = torch.full((N, N), eps_bg, dtype=torch.float64)
    grid[mask] = eps_rod
    return grid


def make_ellipse_eps_grid(a: float, b: float, N: int = 64,
                          eps_bg: float = 8.9, eps_rod: float = 1.0):
    """Elliptical rod with semi-axes a (x) and b (y)."""
    xs = torch.linspace(-0.5, 0.5, N + 1)[:-1] + 0.5 / N
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    mask = (xx / a) ** 2 + (yy / b) ** 2 <= 1.0
    grid = torch.full((N, N), eps_bg, dtype=torch.float64)
    grid[mask] = eps_rod
    return grid


def make_ring_eps_grid(r_outer: float, r_inner: float, N: int = 64,
                       eps_bg: float = 8.9, eps_rod: float = 1.0):
    """Annular ring rod."""
    xs = torch.linspace(-0.5, 0.5, N + 1)[:-1] + 0.5 / N
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    rr = torch.sqrt(xx ** 2 + yy ** 2)
    mask = (rr <= r_outer) & (rr >= r_inner)
    grid = torch.full((N, N), eps_bg, dtype=torch.float64)
    grid[mask] = eps_rod
    return grid


def sample_standard_shapes(n_samples: int, rng: np.random.Generator = None):
    """Sample a mix of crosses, squares, ellipses, and rings.

    Returns list of (grid, label) tuples.
    """
    if rng is None:
        rng = np.random.default_rng()

    shapes = []
    per_type = max(1, n_samples // 4)
    remainder = n_samples - 4 * per_type

    for _ in range(per_type + (1 if remainder > 0 else 0)):
        w = rng.uniform(0.06, 0.18)
        l = rng.uniform(0.3, 0.7)
        if l > 0.9:
            l = 0.9
        shapes.append((make_cross_eps_grid(w, l), "cross"))
    remainder -= 1 if remainder > 0 else 0

    for _ in range(per_type + (1 if remainder > 0 else 0)):
        s = rng.uniform(0.15, 0.6)
        shapes.append((make_square_eps_grid(s), "square"))
    remainder -= 1 if remainder > 0 else 0

    for _ in range(per_type + (1 if remainder > 0 else 0)):
        a = rng.uniform(0.1, 0.4)
        b = rng.uniform(0.1, 0.4)
        shapes.append((make_ellipse_eps_grid(a, b), "ellipse"))
    remainder -= 1 if remainder > 0 else 0

    for _ in range(per_type):
        r_out = rng.uniform(0.2, 0.45)
        r_in = rng.uniform(0.05, r_out * 0.6)
        shapes.append((make_ring_eps_grid(r_out, r_in), "ring"))

    rng.shuffle(shapes)
    return shapes[:n_samples]


def _boundary_is_valid(coeffs_single: np.ndarray, n_angles: int = 128,
                       min_r: float = 0.02) -> bool:
    """Check that r(theta) > min_r everywhere (no self-intersection)."""
    n_harmonics = (len(coeffs_single) - 1) // 2
    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    r = np.full_like(thetas, coeffs_single[0])
    for n in range(1, n_harmonics + 1):
        a_n = coeffs_single[2 * n - 1]
        b_n = coeffs_single[2 * n]
        r = r + a_n * np.cos(n * thetas) + b_n * np.sin(n * thetas)
    return np.all(r > min_r) and np.max(r) < 0.5


def sample_perturbed_circles(n_samples: int, n_harmonics: int = 2,
                             r_mean_range=(0.1, 0.4),
                             perturbation_scale: float = 0.03,
                             rng: np.random.Generator = None):
    """Sample Fourier coefficients for perturbed shapes.

    Larger perturbation_scale produces non-convex shapes. Rejects
    samples with r(theta) < 0 or r(theta) > 0.5 (self-intersecting
    or exceeding the unit cell).

    Returns (n_samples, 1 + 2*n_harmonics) array.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_coeffs = 1 + 2 * n_harmonics
    samples = []
    max_attempts = n_samples * 20

    for _ in range(max_attempts):
        if len(samples) >= n_samples:
            break
        c = np.zeros(n_coeffs)
        c[0] = rng.uniform(r_mean_range[0], r_mean_range[1])
        if n_harmonics > 0:
            c[1:] = rng.normal(0, perturbation_scale, 2 * n_harmonics)
        if _boundary_is_valid(c):
            samples.append(c)

    if len(samples) < n_samples:
        raise RuntimeError(
            f"Only generated {len(samples)}/{n_samples} valid shapes. "
            f"Reduce perturbation_scale or r_mean_range.")

    return np.array(samples)


def fourier_to_levelset_weights(coeffs: np.ndarray, n_rbf_side: int = 4,
                                sigma: float = 0.15):
    """Fit RBF level-set weights to approximate a Fourier boundary shape.

    Uses least-squares to find weights such that the RBF level-set
    field's zero crossing approximates the Fourier boundary.
    """
    n_rbf = n_rbf_side ** 2
    # Sample points on a grid
    N_fit = 32
    xs = np.linspace(-0.5, 0.5, N_fit + 1)[:-1] + 0.5 / N_fit
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    grid_r = np.sqrt(xx**2 + yy**2)
    grid_theta = np.arctan2(yy, xx)
    grid_xy = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    # RBF centers
    cs = np.linspace(-0.5 + 0.5 / n_rbf_side,
                     0.5 - 0.5 / n_rbf_side, n_rbf_side)
    cy, cx = np.meshgrid(cs, cs, indexing="ij")
    centers = np.stack([cx.ravel(), cy.ravel()], axis=-1)

    # RBF basis
    diff = grid_xy[:, None, :] - centers[None, :, :]
    rbf_vals = np.exp(-0.5 * np.sum(diff**2, axis=-1) / sigma**2)  # (N_fit^2, n_rbf)

    batch_weights = []
    for i in range(len(coeffs)):
        n_harmonics = (len(coeffs[i]) - 1) // 2
        r_boundary = np.full_like(grid_r, coeffs[i, 0])
        for n in range(1, n_harmonics + 1):
            a_n = coeffs[i, 2 * n - 1]
            b_n = coeffs[i, 2 * n]
            r_boundary += a_n * np.cos(n * grid_theta) + b_n * np.sin(n * grid_theta)

        # Target: positive inside, negative outside
        target = (r_boundary - grid_r).ravel()
        # Least-squares fit
        w, _, _, _ = np.linalg.lstsq(rbf_vals, target, rcond=None)
        batch_weights.append(w)

    return np.array(batch_weights)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_shape_dataset(coeffs_all: np.ndarray, n_max=3, n_bands=6,
                           n_per_segment=10, N=64):
    """Run PWE for each set of Fourier coefficients.

    Returns (coeffs_tensor, bands_tensor, grids_tensor, meta).
    grids_tensor: (n_samples, N, N) float32 dielectric grids used by PWE.
    """
    g_vectors, m_indices = reciprocal_lattice(n_max)
    k_points, k_dist, tick_pos, tick_labels = make_k_path(n_per_segment)
    n_k = len(k_points)

    all_bands = []
    all_grids = []
    for i, coeffs in enumerate(coeffs_all):
        eps_grid = make_fourier_eps_grid(coeffs, N=N)
        with torch.no_grad():
            bands = solve_bands(k_points, g_vectors, eps_grid,
                                m_indices, n_bands, polarization="tm")
        all_bands.append(bands.float())
        all_grids.append(eps_grid.float())
        if (i + 1) % 10 == 0:
            print(f"    PWE solve {i+1}/{len(coeffs_all)}")

    coeffs_t = torch.tensor(coeffs_all, dtype=torch.float32)
    bands_t = torch.stack(all_bands, dim=0)
    grids_t = torch.stack(all_grids, dim=0)
    meta = dict(k_dist=k_dist, tick_pos=tick_pos, tick_labels=tick_labels,
                n_k=n_k, n_bands=n_bands)
    return coeffs_t, bands_t, grids_t, meta


def generate_grid_dataset(grids, n_max=3, n_bands=6,
                          n_per_segment=10):
    """Run PWE for pre-built dielectric grids.

    Returns (grids_tensor, bands_tensor, meta).
    """
    g_vectors, m_indices = reciprocal_lattice(n_max)
    k_points, k_dist, tick_pos, tick_labels = make_k_path(n_per_segment)
    n_k = len(k_points)

    all_bands = []
    for i, eps_grid in enumerate(grids):
        with torch.no_grad():
            bands = solve_bands(k_points, g_vectors, eps_grid,
                                m_indices, n_bands, polarization="tm")
        all_bands.append(bands.float())
        if (i + 1) % 10 == 0:
            print(f"    PWE solve {i+1}/{len(grids)}")

    grids_t = torch.stack([g.float() for g in grids], dim=0)
    bands_t = torch.stack(all_bands, dim=0)
    meta = dict(k_dist=k_dist, tick_pos=tick_pos, tick_labels=tick_labels,
                n_k=n_k, n_bands=n_bands)
    return grids_t, bands_t, meta


def grid_to_levelset_weights(grids: torch.Tensor, n_rbf_side: int = 4,
                             sigma: float = 0.15,
                             eps_bg: float = 8.9, eps_rod: float = 1.0):
    """Fit RBF level-set weights from dielectric grids.

    Converts the grid to a signed-distance-like target field, then
    does least-squares fitting against the RBF basis.
    """
    N = grids.shape[-1]
    cs = np.linspace(-0.5 + 0.5 / n_rbf_side,
                     0.5 - 0.5 / n_rbf_side, n_rbf_side)
    cy, cx = np.meshgrid(cs, cs, indexing="ij")
    centers = np.stack([cx.ravel(), cy.ravel()], axis=-1)

    xs = np.linspace(-0.5, 0.5, N + 1)[:-1] + 0.5 / N
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    grid_xy = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    diff = grid_xy[:, None, :] - centers[None, :, :]
    rbf_vals = np.exp(-0.5 * np.sum(diff ** 2, axis=-1) / sigma ** 2)

    batch_weights = []
    for i in range(len(grids)):
        eps = grids[i].numpy().ravel()
        target = (eps - eps_bg) / (eps_rod - eps_bg) * 2.0 - 1.0
        w, _, _, _ = np.linalg.lstsq(rbf_vals, target, rcond=None)
        batch_weights.append(w)

    return np.array(batch_weights)


# ---------------------------------------------------------------------------
# V2 training helper (handles vector inputs instead of scalar radii)
# ---------------------------------------------------------------------------

def _get_rasterizer(model):
    """Return the rasterizer sub-module if present, else None."""
    return getattr(model, "rasterizer", None)


def train_model_v2(model, params, bands, n_epochs=2000, lr=1e-3,
                   freq_scale=None, grad_clip: float = 1.0, verbose=True):
    """Train a v2 CANN model with vector parameter input."""
    if freq_scale is None:
        freq_scale = bands.max().item()
    target = bands / freq_scale

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)
    rasterizer = _get_rasterizer(model)
    losses = []

    for epoch in range(n_epochs):
        if rasterizer is not None:
            rasterizer.set_beta(min(epoch / (n_epochs * 0.5), 1.0))

        model.train()
        pred = model(params) / freq_scale
        loss = nn.functional.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if verbose and (epoch + 1) % 500 == 0:
            print(f"  epoch {epoch+1:5d}  loss={loss.item():.6e}  "
                  f"beta={rasterizer.beta:.1f}" if rasterizer else
                  f"  epoch {epoch+1:5d}  loss={loss.item():.6e}")

    return losses, freq_scale


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_v2_comparison(models, names, params_list, bands_test, meta,
                       freq_scales, grids_test, labels_test=None,
                       save_path="cann_v2_comparison.png"):
    """Side-by-side predicted vs ground-truth for v2 models.

    Includes a geometry column showing the test shape for each row.
    grids_test: (n_test, N, N) dielectric grids for the geometry column.
    """
    n_models = len(models)
    n_test = len(bands_test)
    fig, axes = plt.subplots(n_test, n_models + 1,
                             figsize=(5 * n_models + 3, 4 * n_test),
                             squeeze=False,
                             gridspec_kw={"width_ratios": [1] + [2] * n_models})

    k_dist = meta["k_dist"]
    tick_pos = meta["tick_pos"]
    tick_labels = meta["tick_labels"]

    for i in range(n_test):
        ax = axes[i, 0]
        grid = grids_test[i]
        if isinstance(grid, torch.Tensor):
            grid = grid.numpy()
        ax.imshow(grid, origin="lower", cmap="RdBu_r",
                  extent=[-0.5, 0.5, -0.5, 0.5])
        title = labels_test[i] if labels_test else f"#{i}"
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        if i == 0:
            ax.set_xlabel("Geometry")

    # Band structure columns
    for j, (model, name, fs, params) in enumerate(
            zip(models, names, freq_scales, params_list)):
        model.eval()
        with torch.no_grad():
            pred = model(params)

        for i in range(n_test):
            ax = axes[i, j + 1]
            gt = bands_test[i].numpy()
            pr = pred[i].numpy()
            for b in range(gt.shape[-1]):
                ax.plot(k_dist, gt[:, b], "k-", lw=1.2, alpha=0.7,
                        label="PWE" if b == 0 else None)
                ax.plot(k_dist, pr[:, b], "r--", lw=1.0, alpha=0.8,
                        label=name if b == 0 else None)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels)
            ax.set_ylabel("$\\omega a / 2\\pi c$")
            ax.set_title(name, fontsize=9)
            ax.legend(fontsize=8)
            for tp in tick_pos:
                ax.axvline(tp, color="gray", lw=0.5, ls="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def print_metrics_v2(models, names, params_list, bands_test, freq_scales):
    """Per-band RMSE on test set for v2 models."""
    print("\n--- Test metrics (v2) ---")
    for model, name, fs, params in zip(models, names, freq_scales, params_list):
        model.eval()
        with torch.no_grad():
            pred = model(params)
        err = (pred - bands_test).abs()
        rmse_per_band = err.pow(2).mean(dim=(0, 1)).sqrt()
        rmse_total = err.pow(2).mean().sqrt().item()
        print(f"{name:>25s}  RMSE={rmse_total:.6f}  "
              f"per-band={[f'{v:.5f}' for v in rmse_per_band.tolist()]}")
        print(f"{'':>25s}  params={sum(p.numel() for p in model.parameters()):,}")


def plot_training_geometries(coeffs_all, n_show=8,
                             save_path="cann_v2_train_geometries.png"):
    """Visualize hard-rasterized PWE training geometries."""
    n_show = min(n_show, len(coeffs_all))
    ncols = min(n_show, 8)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                             squeeze=False)
    for i in range(n_show):
        ax = axes[i // ncols, i % ncols]
        grid = make_fourier_eps_grid(coeffs_all[i], N=64)
        ax.imshow(grid.numpy(), origin="lower", cmap="RdBu_r",
                  extent=[-0.5, 0.5, -0.5, 0.5])
        ax.set_title(f"$a_0$={coeffs_all[i, 0]:.2f}", fontsize=9)
        ax.set_aspect("equal")
    for i in range(n_show, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def plot_training_geometries_from_grids(grids, all_labels=None, n_show=8,
                                        save_path="cann_v2_train_geometries.png"):
    """Visualize training geometries from pre-built grids."""
    n_show = min(n_show, len(grids))
    ncols = min(n_show, 8)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                             squeeze=False)
    for i in range(n_show):
        ax = axes[i // ncols, i % ncols]
        g = grids[i]
        if isinstance(g, torch.Tensor):
            g = g.numpy()
        ax.imshow(g, origin="lower", cmap="RdBu_r",
                  extent=[-0.5, 0.5, -0.5, 0.5])
        title = all_labels[i] if all_labels else f"#{i}"
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
    for i in range(n_show, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def plot_rasterizer_samples(coeffs_samples, weights_samples, n_show=4,
                            save_path="cann_v2_rasterizers.png"):
    """Visualize rasterized shapes from both parameterizations."""
    fourier_rast = FourierShapeRasterizer(grid_size=32)
    levelset_rast = LevelSetRasterizer(n_rbf_side=4, grid_size=32)

    coeffs_t = torch.tensor(coeffs_samples[:n_show], dtype=torch.float32)
    weights_t = torch.tensor(weights_samples[:n_show], dtype=torch.float32)

    with torch.no_grad():
        fourier_grids = fourier_rast(coeffs_t).squeeze(1).numpy()
        levelset_grids = levelset_rast(weights_t).squeeze(1).numpy()

    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    for i in range(n_show):
        axes[0, i].imshow(fourier_grids[i], origin="lower", cmap="RdBu_r",
                          extent=[-0.5, 0.5, -0.5, 0.5])
        axes[0, i].set_title(f"Fourier #{i}")
        axes[0, i].set_aspect("equal")

        axes[1, i].imshow(levelset_grids[i], origin="lower", cmap="RdBu_r",
                          extent=[-0.5, 0.5, -0.5, 0.5])
        axes[1, i].set_title(f"LevelSet #{i}")
        axes[1, i].set_aspect("equal")

    axes[0, 0].set_ylabel("Fourier")
    axes[1, 0].set_ylabel("LevelSet")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    n_bands = 6
    n_per_segment = 10
    n_harmonics = 4
    n_coeffs = 1 + 2 * n_harmonics  # 9
    n_rbf_side = 4
    N_pwe = 64

    # --- Sample perturbed circles (Fourier-expressible) ---
    print("Sampling perturbed circles...")
    n_circle_train, n_circle_test = 60, 10
    train_coeffs = sample_perturbed_circles(
        n_circle_train, n_harmonics=n_harmonics,
        r_mean_range=(0.12, 0.38), perturbation_scale=0.08, rng=rng)
    test_coeffs = sample_perturbed_circles(
        n_circle_test, n_harmonics=n_harmonics,
        r_mean_range=(0.15, 0.35), perturbation_scale=0.06, rng=rng)

    # Rasterize circles to grids for the unified pipeline
    circle_train_grids = [make_fourier_eps_grid(c, N=N_pwe) for c in train_coeffs]
    circle_test_grids = [make_fourier_eps_grid(c, N=N_pwe) for c in test_coeffs]
    circle_test_labels = [f"circle {i}" for i in range(n_circle_test)]

    # --- Sample standard shapes (grid-only) ---
    print("Sampling standard shapes...")
    n_std_train, n_std_test = 90, 10
    std_train = sample_standard_shapes(n_std_train, rng=rng)
    std_test = sample_standard_shapes(n_std_test, rng=rng)

    std_train_grids = [g for g, _ in std_train]
    std_test_grids = [g for g, _ in std_test]
    std_test_labels = [lbl for _, lbl in std_test]

    # --- Merge into unified grid dataset ---
    all_train_grids = circle_train_grids + std_train_grids
    all_test_grids = circle_test_grids + std_test_grids
    all_test_labels = circle_test_labels + std_test_labels
    n_train = len(all_train_grids)
    n_test = len(all_test_grids)

    plot_training_geometries_from_grids(
        all_train_grids, all_labels=None, n_show=min(n_train, 16))

    # --- Generate ground truth bands ---
    print("Generating training bands...")
    grids_train_t, b_train, meta = generate_grid_dataset(
        all_train_grids, n_bands=n_bands, n_per_segment=n_per_segment)
    print("Generating test bands...")
    grids_test_t, b_test, _ = generate_grid_dataset(
        all_test_grids, n_bands=n_bands, n_per_segment=n_per_segment)

    n_k = meta["n_k"]
    print(f"  train: {n_train} samples ({n_circle_train} circles + "
          f"{n_std_train} standard), n_k={n_k}, n_bands={n_bands}")
    print(f"  test:  {n_test} samples ({n_circle_test} circles + "
          f"{n_std_test} standard)")

    # --- Fit level-set weights from grids ---
    print("Fitting level-set weights...")
    weights_train = grid_to_levelset_weights(grids_train_t, n_rbf_side=n_rbf_side)
    weights_test = grid_to_levelset_weights(grids_test_t, n_rbf_side=n_rbf_side)
    weights_train_t = torch.tensor(weights_train, dtype=torch.float32)
    weights_test_t = torch.tensor(weights_test, dtype=torch.float32)

    # Fourier coeffs only for the circle subset
    coeffs_train_t = torch.tensor(train_coeffs, dtype=torch.float32)
    coeffs_test_t = torch.tensor(test_coeffs, dtype=torch.float32)
    b_train_circles = b_train[:n_circle_train]
    b_test_circles = b_test[:n_circle_test]

    # Visualize rasterizer fidelity on circle subset
    circle_weights_train = fourier_to_levelset_weights(
        train_coeffs, n_rbf_side=n_rbf_side)
    plot_rasterizer_samples(train_coeffs, circle_weights_train, n_show=4)

    # --- Build models ---
    model_fourier = FourierShapeCANN(
        n_coeffs=n_coeffs, n_k=n_k, n_bands=n_bands, hidden=128)
    model_levelset = LevelSetCANN(
        n_rbf_side=n_rbf_side, n_k=n_k, n_bands=n_bands, hidden=128)
    model_grid = GridCANN(n_k=n_k, n_bands=n_bands, hidden=128)

    n_epochs = 9000

    # --- Train ---
    # FourierShapeCANN: circles only (can't represent crosses etc.)
    print("\nTraining FourierShapeCANN (circles only)...")
    losses_fourier, fs_fourier = train_model_v2(
        model_fourier, coeffs_train_t, b_train_circles,
        n_epochs=n_epochs, lr=1e-3)

    # LevelSet and Grid: all shapes
    print("\nTraining LevelSetCANN (all shapes)...")
    losses_levelset, fs_levelset = train_model_v2(
        model_levelset, weights_train_t, b_train, n_epochs=n_epochs, lr=1e-3)

    print("\nTraining GridCANN (all shapes)...")
    losses_grid, fs_grid = train_model_v2(
        model_grid, grids_train_t, b_train, n_epochs=n_epochs, lr=1e-3)

    # --- Evaluate on full test set (Grid + LevelSet) ---
    models_full = [model_levelset, model_grid]
    names_full = ["LevelSet", "Grid"]
    params_full = [weights_test_t, grids_test_t]
    fs_full = [fs_levelset, fs_grid]

    print_metrics_v2(models_full, names_full, params_full, b_test, fs_full)
    plot_v2_comparison(models_full, names_full, params_full, b_test, meta,
                       fs_full, grids_test_t, labels_test=all_test_labels,
                       save_path="cann_v2_comparison.png")

    # --- Evaluate on circle subset (all models including Fourier) ---
    models_circ = [model_fourier, model_levelset, model_grid]
    names_circ = ["FourierShape", "LevelSet", "Grid"]
    params_circ = [coeffs_test_t, weights_test_t[:n_circle_test],
                   grids_test_t[:n_circle_test]]
    fs_circ = [fs_fourier, fs_levelset, fs_grid]

    print("\n--- Circle-subset metrics ---")
    print_metrics_v2(models_circ, names_circ, params_circ,
                     b_test_circles, fs_circ)
    plot_v2_comparison(models_circ, names_circ, params_circ,
                       b_test_circles, meta, fs_circ,
                       grids_test_t[:n_circle_test],
                       labels_test=circle_test_labels,
                       save_path="cann_v2_comparison_circles.png")

    # --- Loss curves ---
    plot_losses([losses_fourier, losses_levelset, losses_grid],
                ["FourierShape (circles)", "LevelSet (all)", "Grid (all)"],
                save_path="cann_v2_losses.png")


if __name__ == "__main__":
    main()
