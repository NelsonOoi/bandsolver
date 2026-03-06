"""
Training script for v2.1 photonic bandstructure CANNs.

Improvements over train_cann_v2:
  - Weight decay in Adam optimizer
  - C4v symmetry augmentation (8x effective data)
  - Lightweight CNN encoder + FC backbone with dropout
  - Minibatch training for augmented dataset
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pwe_torch import make_k_path
from cann_v2 import GridCANN
from cann_v2_1 import GridCANN_v21
from train_cann_v2 import (
    sample_perturbed_circles, sample_standard_shapes,
    make_fourier_eps_grid, generate_grid_dataset,
    plot_training_geometries_from_grids,
)
from train_cann import plot_losses


# ---------------------------------------------------------------------------
# C4v symmetry augmentation
# ---------------------------------------------------------------------------

def augment_c4v(grids: torch.Tensor, bands: torch.Tensor):
    """Apply C4v symmetry group (4 rotations x 2 reflections).

    Bands on the Gamma-X-M-Gamma path of a square lattice are invariant
    under C4v, so augmented grids share the same band structure.
    """
    squeeze = grids.dim() == 4
    if squeeze:
        grids = grids.squeeze(1)

    aug_grids, aug_bands = [], []
    for k in range(4):
        g_rot = torch.rot90(grids, k, dims=(-2, -1))
        aug_grids.append(g_rot)
        aug_bands.append(bands)
        aug_grids.append(torch.flip(g_rot, dims=(-1,)))
        aug_bands.append(bands)

    out_grids = torch.cat(aug_grids, dim=0)
    out_bands = torch.cat(aug_bands, dim=0)
    if squeeze:
        out_grids = out_grids.unsqueeze(1)
    return out_grids, out_bands


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_v21(model, params, bands, n_epochs=3000, lr=1e-3,
              weight_decay=1e-4, grad_clip=1.0, batch_size=128,
              freq_scale=None, verbose=True):
    """Train with minibatching, weight decay, and gradient clipping."""
    if freq_scale is None:
        freq_scale = bands.max().item()
    target = bands / freq_scale

    n_samples = params.shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)
    rasterizer = getattr(model, "rasterizer", None)
    losses = []

    for epoch in range(n_epochs):
        if rasterizer is not None:
            rasterizer.set_beta(min(epoch / (n_epochs * 0.5), 1.0))

        model.train()
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(params[idx]) / freq_scale
            loss = nn.functional.mse_loss(pred, target[idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if verbose and (epoch + 1) % 100 == 0:
            beta_str = f"  beta={rasterizer.beta:.1f}" if rasterizer else ""
            print(f"  epoch {epoch+1:5d}  loss={avg_loss:.6e}{beta_str}")

    return losses, freq_scale


# ---------------------------------------------------------------------------
# Plotting / metrics
# ---------------------------------------------------------------------------

def print_metrics(models, names, params_list, bands_test, freq_scales):
    print("\n--- Test metrics (v2.1) ---")
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


def plot_comparison(models, names, params_list, bands_test, meta,
                    freq_scales, grids_test, labels_test=None,
                    save_path="cann_v2_1_comparison.png"):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    n_bands = 6
    n_per_segment = 10
    n_harmonics = 4
    N_pwe = 64

    # --- Sample shapes ---
    print("Sampling shapes...")
    n_circle_train, n_circle_test = 60, 10
    train_coeffs = sample_perturbed_circles(
        n_circle_train, n_harmonics=n_harmonics,
        r_mean_range=(0.12, 0.38), perturbation_scale=0.08, rng=rng)
    test_coeffs = sample_perturbed_circles(
        n_circle_test, n_harmonics=n_harmonics,
        r_mean_range=(0.15, 0.35), perturbation_scale=0.06, rng=rng)

    circle_train_grids = [make_fourier_eps_grid(c, N=N_pwe) for c in train_coeffs]
    circle_test_grids = [make_fourier_eps_grid(c, N=N_pwe) for c in test_coeffs]
    circle_test_labels = [f"circle {i}" for i in range(n_circle_test)]

    n_std_train, n_std_test = 90, 10
    std_train = sample_standard_shapes(n_std_train, rng=rng)
    std_test = sample_standard_shapes(n_std_test, rng=rng)

    std_train_grids = [g for g, _ in std_train]
    std_test_grids = [g for g, _ in std_test]
    std_test_labels = [lbl for _, lbl in std_test]

    all_train_grids = circle_train_grids + std_train_grids
    all_test_grids = circle_test_grids + std_test_grids
    all_test_labels = circle_test_labels + std_test_labels

    plot_training_geometries_from_grids(
        all_train_grids, n_show=min(len(all_train_grids), 16),
        save_path="cann_v2_1_train_geometries.png")

    # --- Generate ground truth ---
    print("Generating training bands...")
    grids_train_t, b_train, meta = generate_grid_dataset(
        all_train_grids, n_bands=n_bands, n_per_segment=n_per_segment)
    print("Generating test bands...")
    grids_test_t, b_test, _ = generate_grid_dataset(
        all_test_grids, n_bands=n_bands, n_per_segment=n_per_segment)

    n_k = meta["n_k"]
    print(f"  train: {len(all_train_grids)} samples, n_k={n_k}, n_bands={n_bands}")
    print(f"  test:  {len(all_test_grids)} samples")

    # --- C4v augmentation ---
    print("Applying C4v augmentation (8x)...")
    grids_train_aug, b_train_aug = augment_c4v(grids_train_t, b_train)
    print(f"  augmented train: {len(grids_train_aug)} samples")

    # --- Build models ---
    model_v21 = GridCANN_v21(n_k=n_k, n_bands=n_bands)
    model_v2 = GridCANN(n_k=n_k, n_bands=n_bands, hidden=128)

    n_epochs = 3000

    # --- Train ---
    print("\nTraining GridCANN v2.1 (CNN+dropout, augmented)...")
    losses_v21, fs_v21 = train_v21(
        model_v21, grids_train_aug, b_train_aug, n_epochs=n_epochs)

    print("\nTraining GridCANN v2 baseline...")
    from train_cann_v2 import train_model_v2
    losses_v2, fs_v2 = train_model_v2(
        model_v2, grids_train_t, b_train, n_epochs=n_epochs, lr=1e-3)

    # --- Evaluate ---
    models = [model_v21, model_v2]
    names = ["Grid-v2.1", "Grid-v2 baseline"]
    params = [grids_test_t, grids_test_t]
    fs = [fs_v21, fs_v2]

    print_metrics(models, names, params, b_test, fs)

    plot_comparison(models, names, params, b_test, meta, fs,
                    grids_test_t, labels_test=all_test_labels,
                    save_path="cann_v2_1_comparison.png")

    plot_losses([losses_v21, losses_v2],
                ["Grid-v2.1", "Grid-v2 baseline"],
                save_path="cann_v2_1_losses.png")


if __name__ == "__main__":
    main()
