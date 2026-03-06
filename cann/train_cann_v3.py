"""
Training script for dual-physics CANN v3.

Jointly trains photonic + phononic band prediction from geometry masks.
Compares:
  1. DualGridCANN        — shared encoder, independent backbones
  2. DualGridCANN_cross   — shared encoder, cross-gated backbones
  3. GridCANN baseline   — photonic-only (from cann_v2)
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pwe_torch import reciprocal_lattice, solve_bands, make_k_path
from pwe_elastic_torch import solve_elastic_bands_from_mask
from cann_v2 import GridCANN
from cann_v3 import DualGridCANN, DualGridCANN_cross
from train_cann_v2 import (
    sample_perturbed_circles, sample_standard_shapes,
    make_fourier_eps_grid, train_model_v2,
    plot_training_geometries_from_grids,
)
from train_cann_v2_1 import augment_c4v
from train_cann import plot_losses


# ---------------------------------------------------------------------------
# Mask convention: mask=1 is air hole, mask=0 is Si matrix
# eps_grid = 8.9 - 7.9 * mask  (Si=8.9, Air=1.0)
# ---------------------------------------------------------------------------

def eps_grid_to_mask(eps_grid: torch.Tensor) -> torch.Tensor:
    """Convert dielectric grid to soft mask (1=hole, 0=matrix)."""
    return (8.9 - eps_grid) / 7.9


# ---------------------------------------------------------------------------
# Dual-physics dataset generation
# ---------------------------------------------------------------------------

def generate_dual_dataset(grids, n_max=3, n_bands_phot=6, n_bands_phon=10,
                          n_per_segment=10, N_pwe=64):
    """Run both photonic and elastic PWE for each dielectric grid.

    Returns (grids_t, b_phot_t, b_phon_t, meta).
    """
    g_vectors, m_indices = reciprocal_lattice(n_max)
    k_points, k_dist, tick_pos, tick_labels = make_k_path(n_per_segment)
    n_k = len(k_points)

    all_phot, all_phon = [], []
    for i, eps_grid in enumerate(grids):
        with torch.no_grad():
            bands_phot = solve_bands(k_points, g_vectors, eps_grid,
                                     m_indices, n_bands_phot, polarization="tm")
            mask = eps_grid_to_mask(eps_grid)
            bands_phon = solve_elastic_bands_from_mask(
                k_points, g_vectors, mask, m_indices, n_bands_phon)
        all_phot.append(bands_phot.float())
        all_phon.append(bands_phon.float())
        if (i + 1) % 10 == 0:
            print(f"    PWE solve {i+1}/{len(grids)}")

    grids_t = torch.stack([g.float() for g in grids], dim=0)
    b_phot_t = torch.stack(all_phot, dim=0)
    b_phon_t = torch.stack(all_phon, dim=0)
    meta = dict(k_dist=k_dist, tick_pos=tick_pos, tick_labels=tick_labels,
                n_k=n_k, n_bands_phot=n_bands_phot, n_bands_phon=n_bands_phon)
    return grids_t, b_phot_t, b_phon_t, meta


# ---------------------------------------------------------------------------
# C4v augmentation (extends to dual targets)
# ---------------------------------------------------------------------------

def augment_c4v_dual(grids: torch.Tensor, b_phot: torch.Tensor,
                     b_phon: torch.Tensor):
    """C4v augmentation for dual-physics targets."""
    grids_aug, b_phot_aug = augment_c4v(grids, b_phot)
    _, b_phon_aug = augment_c4v(grids, b_phon)
    return grids_aug, b_phot_aug, b_phon_aug


# ---------------------------------------------------------------------------
# Dual training loop
# ---------------------------------------------------------------------------

def train_dual(model, grids, b_phot, b_phon, alpha=0.5, n_epochs=5000,
               lr=1e-3, grad_clip=1.0, verbose=True, resume=None):
    """Joint training on photonic + phononic targets.

    Loss = alpha * MSE_phot + (1 - alpha) * MSE_phon  (each normalized).

    If resume is a checkpoint dict, restores optimizer/scheduler/losses and
    continues from the saved epoch.
    """
    fs_phot = b_phot.max().item()
    fs_phon = b_phon.max().item()
    tgt_phot = b_phot / fs_phot
    tgt_phon = b_phon / fs_phon

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    start_epoch = 0
    losses, losses_phot, losses_phon = [], [], []

    if resume is not None:
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        start_epoch = resume["epoch"]
        losses = resume.get("losses", [])
        losses_phot = resume.get("losses_phot", [])
        losses_phon = resume.get("losses_phon", [])
        print(f"  Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, n_epochs):
        model.train()
        pred_phot, pred_phon = model(grids)
        l_phot = nn.functional.mse_loss(pred_phot / fs_phot, tgt_phot)
        l_phon = nn.functional.mse_loss(pred_phon / fs_phon, tgt_phon)
        loss = alpha * l_phot + (1 - alpha) * l_phon

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        losses_phot.append(l_phot.item())
        losses_phon.append(l_phon.item())

        if verbose and (epoch + 1) % 500 == 0:
            print(f"  epoch {epoch+1:5d}  loss={loss.item():.6e}  "
                  f"phot={l_phot.item():.6e}  phon={l_phon.item():.6e}")

    train_state = dict(
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        epoch=n_epochs,
        losses=losses, losses_phot=losses_phot, losses_phon=losses_phon,
    )
    return losses, losses_phot, losses_phon, fs_phot, fs_phon, train_state


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dual_losses(loss_data, save_path="cann_v3_losses.png"):
    """Loss curves for all models including per-physics breakdown."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.set_title("Total loss")
    for name, losses in loss_data["total"]:
        ax.semilogy(losses, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axes[1]
    ax.set_title("Photonic loss")
    for name, losses in loss_data["phot"]:
        ax.semilogy(losses, label=name)
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[2]
    ax.set_title("Phononic loss")
    for name, losses in loss_data["phon"]:
        ax.semilogy(losses, label=name)
    ax.set_xlabel("Epoch")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def plot_band_comparison(models_and_data, meta, grids_test, labels_test,
                         physics="phot", save_path="cann_v3_bands.png"):
    """Band structure comparison: model predictions vs PWE ground truth.

    models_and_data: list of (model_name, pred_bands)  -- (n_test, n_k, n_bands)
    """
    k_dist = meta["k_dist"]
    tick_pos = meta["tick_pos"]
    tick_labels = meta["tick_labels"]
    n_models = len(models_and_data)
    n_show = min(4, len(grids_test))

    fig, axes = plt.subplots(n_show, n_models + 1,
                             figsize=(5 * n_models + 3, 4 * n_show),
                             squeeze=False,
                             gridspec_kw={"width_ratios": [1] + [2] * n_models})

    for i in range(n_show):
        ax = axes[i, 0]
        g = grids_test[i].numpy() if isinstance(grids_test[i], torch.Tensor) else grids_test[i]
        ax.imshow(g, origin="lower", cmap="RdBu_r",
                  extent=[-0.5, 0.5, -0.5, 0.5])
        ax.set_title(labels_test[i] if labels_test else f"#{i}", fontsize=9)
        ax.set_aspect("equal")

    for j, (name, gt, pred) in enumerate(models_and_data):
        for i in range(n_show):
            ax = axes[i, j + 1]
            gt_i = gt[i].numpy() if isinstance(gt[i], torch.Tensor) else gt[i]
            pr_i = pred[i].numpy() if isinstance(pred[i], torch.Tensor) else pred[i]
            for b in range(gt_i.shape[-1]):
                ax.plot(k_dist, gt_i[:, b], "k-", lw=1.2, alpha=0.7,
                        label="PWE" if b == 0 else None)
                ax.plot(k_dist, pr_i[:, b], "r--", lw=1.0, alpha=0.8,
                        label=name if b == 0 else None)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels)
            ylabel = "$\\omega a / 2\\pi c$" if physics == "phot" else "$\\omega$ (rad/s $\\cdot$ a)"
            ax.set_ylabel(ylabel)
            ax.set_title(name, fontsize=9)
            ax.legend(fontsize=7)
            for tp in tick_pos:
                ax.axvline(tp, color="gray", lw=0.5, ls="--")

    plt.suptitle(f"{'Photonic' if physics == 'phot' else 'Phononic'} bands", fontsize=14)
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

    n_bands_phot = 6
    n_bands_phon = 10
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

    all_train_grids = circle_train_grids + [g for g, _ in std_train]
    all_test_grids = circle_test_grids + [g for g, _ in std_test]
    all_test_labels = circle_test_labels + [lbl for _, lbl in std_test]

    plot_training_geometries_from_grids(
        all_train_grids, n_show=min(len(all_train_grids), 16),
        save_path="cann_v3_train_geometries.png")

    # --- Generate dual ground truth ---
    print("Generating training bands (photonic + phononic)...")
    grids_train, b_phot_train, b_phon_train, meta = generate_dual_dataset(
        all_train_grids, n_bands_phot=n_bands_phot,
        n_bands_phon=n_bands_phon, n_per_segment=n_per_segment)
    print("Generating test bands (photonic + phononic)...")
    grids_test, b_phot_test, b_phon_test, _ = generate_dual_dataset(
        all_test_grids, n_bands_phot=n_bands_phot,
        n_bands_phon=n_bands_phon, n_per_segment=n_per_segment)

    n_k = meta["n_k"]
    print(f"  train: {len(all_train_grids)} samples, n_k={n_k}")
    print(f"  test:  {len(all_test_grids)} samples")

    # --- C4v augmentation ---
    print("Applying C4v augmentation (8x)...")
    grids_train_aug, b_phot_aug, b_phon_aug = augment_c4v_dual(
        grids_train, b_phot_train, b_phon_train)
    print(f"  augmented: {len(grids_train_aug)} samples")

    # --- Build models ---
    model_dual = DualGridCANN(n_k=n_k, n_bands_phot=n_bands_phot,
                              n_bands_phon=n_bands_phon, hidden=128)
    model_cross = DualGridCANN_cross(n_k=n_k, n_bands_phot=n_bands_phot,
                                     n_bands_phon=n_bands_phon, hidden=128)
    model_baseline = GridCANN(n_k=n_k, n_bands=n_bands_phot, hidden=128)

    n_epochs = 5000

    # --- Train dual models ---
    print("\nTraining DualGridCANN...")
    (l_dual, l_dual_phot, l_dual_phon,
     fs_phot_dual, fs_phon_dual, ts_dual) = train_dual(
        model_dual, grids_train_aug, b_phot_aug, b_phon_aug,
        n_epochs=n_epochs)

    print("\nTraining DualGridCANN_cross...")
    (l_cross, l_cross_phot, l_cross_phon,
     fs_phot_cross, fs_phon_cross, ts_cross) = train_dual(
        model_cross, grids_train_aug, b_phot_aug, b_phon_aug,
        n_epochs=n_epochs)

    # --- Train photonic-only baseline ---
    print("\nTraining GridCANN baseline (photonic-only)...")
    grids_phot_aug, b_phot_only_aug = augment_c4v(grids_train, b_phot_train)
    l_baseline, fs_baseline = train_model_v2(
        model_baseline, grids_phot_aug, b_phot_only_aug,
        n_epochs=n_epochs, lr=1e-3)

    # --- Save models ---
    save_dir = "cann_v3_checkpoints"
    import os
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        "model_state": model_dual.state_dict(),
        "fs_phot": fs_phot_dual, "fs_phon": fs_phon_dual,
        "meta": meta, "train_state": ts_dual,
    }, os.path.join(save_dir, "dual_grid_cann.pt"))

    torch.save({
        "model_state": model_cross.state_dict(),
        "fs_phot": fs_phot_cross, "fs_phon": fs_phon_cross,
        "meta": meta, "train_state": ts_cross,
    }, os.path.join(save_dir, "dual_grid_cann_cross.pt"))

    torch.save({
        "model_state": model_baseline.state_dict(),
        "fs_phot": fs_baseline,
        "meta": meta,
    }, os.path.join(save_dir, "grid_cann_baseline.pt"))

    print(f"\nModels saved to {save_dir}/")

    # --- Loss plots ---
    loss_data = {
        "total": [("DualGridCANN", l_dual),
                  ("DualGridCANN_cross", l_cross),
                  ("GridCANN baseline", l_baseline)],
        "phot":  [("DualGridCANN", l_dual_phot),
                  ("DualGridCANN_cross", l_cross_phot),
                  ("GridCANN baseline", l_baseline)],
        "phon":  [("DualGridCANN", l_dual_phon),
                  ("DualGridCANN_cross", l_cross_phon)],
    }
    plot_dual_losses(loss_data)

    # --- Evaluate ---
    model_dual.eval()
    model_cross.eval()
    model_baseline.eval()
    with torch.no_grad():
        pred_dual_phot, pred_dual_phon = model_dual(grids_test)
        pred_cross_phot, pred_cross_phon = model_cross(grids_test)
        pred_baseline_phot = model_baseline(grids_test)

    # Photonic comparison (all 3 models)
    plot_band_comparison(
        [("DualGridCANN", b_phot_test, pred_dual_phot),
         ("DualGridCANN_cross", b_phot_test, pred_cross_phot),
         ("GridCANN baseline", b_phot_test, pred_baseline_phot)],
        meta, grids_test, all_test_labels, physics="phot",
        save_path="cann_v3_photonic_bands.png")

    # Phononic comparison (2 dual models)
    plot_band_comparison(
        [("DualGridCANN", b_phon_test, pred_dual_phon),
         ("DualGridCANN_cross", b_phon_test, pred_cross_phon)],
        meta, grids_test, all_test_labels, physics="phon",
        save_path="cann_v3_phononic_bands.png")

    # --- Metrics ---
    print("\n--- Photonic test metrics ---")
    for name, pred in [("DualGridCANN", pred_dual_phot),
                       ("DualGridCANN_cross", pred_cross_phot),
                       ("GridCANN baseline", pred_baseline_phot)]:
        err = (pred - b_phot_test).abs()
        rmse = err.pow(2).mean().sqrt().item()
        print(f"  {name:>25s}  RMSE={rmse:.6f}")

    print("\n--- Phononic test metrics ---")
    for name, pred in [("DualGridCANN", pred_dual_phon),
                       ("DualGridCANN_cross", pred_cross_phon)]:
        err = (pred - b_phon_test).abs()
        rmse = err.pow(2).mean().sqrt().item()
        print(f"  {name:>25s}  RMSE={rmse:.6f}")

    # Parameter counts
    print("\n--- Parameter counts ---")
    for name, m in [("DualGridCANN", model_dual),
                    ("DualGridCANN_cross", model_cross),
                    ("GridCANN baseline", model_baseline)]:
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  {name:>25s}  {n_params:,}")


if __name__ == "__main__":
    main()
