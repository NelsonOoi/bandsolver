"""
Training script for photonic bandstructure CANNs.

Generates training data via PWE solver, trains both FullyConnectedCANN and
ParallelBranchCANN, and produces side-by-side comparison plots.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

from pwe_torch import reciprocal_lattice, solve_bands, make_k_path
from cann import FullyConnectedCANN, ParallelBranchCANN


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_circle_eps_grid(r: float, N: int = 64,
                         eps_bg: float = 8.9, eps_rod: float = 1.0):
    """Create NxN dielectric grid with a circular rod of radius r (a=1)."""
    xs = torch.linspace(-0.5, 0.5, N + 1)[:-1] + 0.5 / N
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    mask = (xx ** 2 + yy ** 2) <= r ** 2
    grid = torch.full((N, N), eps_bg, dtype=torch.float64)
    grid[mask] = eps_rod
    return grid


def generate_dataset(radii, n_max=3, n_bands=6, n_per_segment=10, N=64):
    """Run PWE for each radius, return (radii_tensor, bands_tensor)."""
    g_vectors, m_indices = reciprocal_lattice(n_max)
    k_points, k_dist, tick_pos, tick_labels = make_k_path(n_per_segment)
    n_k = len(k_points)

    all_bands = []
    for r in radii:
        eps_grid = make_circle_eps_grid(r, N=N)
        with torch.no_grad():
            bands = solve_bands(k_points, g_vectors, eps_grid,
                                m_indices, n_bands, polarization="tm")
        all_bands.append(bands.float())

    radii_t = torch.tensor(radii, dtype=torch.float32)
    bands_t = torch.stack(all_bands, dim=0)  # (n_samples, n_k, n_bands)
    meta = dict(k_dist=k_dist, tick_pos=tick_pos, tick_labels=tick_labels,
                n_k=n_k, n_bands=n_bands)
    return radii_t, bands_t, meta


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, radii, bands, n_epochs=2000, lr=1e-3,
                freq_scale=None, verbose=True):
    """Train a CANN model. Returns loss history."""
    if freq_scale is None:
        freq_scale = bands.max().item()
    target = bands / freq_scale

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    losses = []

    for epoch in range(n_epochs):
        model.train()
        pred = model(radii) / freq_scale
        loss = nn.functional.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if verbose and (epoch + 1) % 500 == 0:
            print(f"  epoch {epoch+1:5d}  loss={loss.item():.6e}")

    return losses, freq_scale


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(models, names, radii_test, bands_test, meta, freq_scales,
                    save_path="cann_comparison.png"):
    """Side-by-side predicted vs ground-truth bandstructures."""
    n_models = len(models)
    n_test = len(radii_test)
    fig, axes = plt.subplots(n_test, n_models, figsize=(5 * n_models, 4 * n_test),
                             squeeze=False)

    k_dist = meta["k_dist"]
    tick_pos = meta["tick_pos"]
    tick_labels = meta["tick_labels"]

    for j, (model, name, fs) in enumerate(zip(models, names, freq_scales)):
        model.eval()
        with torch.no_grad():
            pred = model(radii_test)

        for i in range(n_test):
            ax = axes[i, j]
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
            ax.set_title(f"r = {radii_test[i].item():.3f}")
            ax.legend(fontsize=8)
            for tp in tick_pos:
                ax.axvline(tp, color="gray", lw=0.5, ls="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def plot_losses(all_losses, names, save_path="cann_losses.png"):
    """Training loss curves."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for losses, name in zip(all_losses, names):
        ax.semilogy(losses, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalized)")
    ax.legend()
    ax.set_title("Training convergence")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def print_metrics(models, names, radii_test, bands_test, freq_scales):
    """Per-band RMSE on test set."""
    print("\n--- Test metrics ---")
    for model, name, fs in zip(models, names, freq_scales):
        model.eval()
        with torch.no_grad():
            pred = model(radii_test)
        err = (pred - bands_test).abs()
        rmse_per_band = err.pow(2).mean(dim=(0, 1)).sqrt()
        rmse_total = err.pow(2).mean().sqrt().item()
        print(f"{name:>25s}  RMSE={rmse_total:.6f}  "
              f"per-band={[f'{v:.5f}' for v in rmse_per_band.tolist()]}")
        print(f"{'':>25s}  params={sum(p.numel() for p in model.parameters()):,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    n_bands = 6
    n_per_segment = 10

    # Generate data
    print("Generating training data...")
    train_radii = np.linspace(0.05, 0.45, 40)
    test_radii = np.array([0.10, 0.25, 0.38])

    r_train, b_train, meta = generate_dataset(
        train_radii, n_bands=n_bands, n_per_segment=n_per_segment)
    r_test, b_test, _ = generate_dataset(
        test_radii, n_bands=n_bands, n_per_segment=n_per_segment)

    n_k = meta["n_k"]
    print(f"  train: {len(train_radii)} samples, n_k={n_k}, n_bands={n_bands}")
    print(f"  test:  {len(test_radii)} samples")

    # Build models
    model_fc = FullyConnectedCANN(n_k=n_k, n_bands=n_bands,
                                  hidden=128, n_fourier=8)
    model_pb = ParallelBranchCANN(n_k=n_k, n_bands=n_bands,
                                  hidden=128, n_max=3)

    models = [model_fc, model_pb]
    names = ["FullyConnected", "ParallelBranch"]
    n_epochs = 3000

    # Train
    all_losses = []
    freq_scales = []
    for model, name in zip(models, names):
        print(f"\nTraining {name}...")
        losses, fs = train_model(model, r_train, b_train,
                                 n_epochs=n_epochs, lr=1e-3)
        all_losses.append(losses)
        freq_scales.append(fs)

    # Evaluate and plot
    print_metrics(models, names, r_test, b_test, freq_scales)
    plot_losses(all_losses, names)
    plot_comparison(models, names, r_test, b_test, meta, freq_scales)


if __name__ == "__main__":
    main()
