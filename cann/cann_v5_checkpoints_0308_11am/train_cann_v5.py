"""
Training script for dual-physics CANN v5.

v5 changes from v4:
  - Generates larger dataset (500 train / 50 test) inline
  - Per-band standardized targets (zero mean, unit variance)
  - Mini-batch training
  - Per-k conditioned backbone (no cumsum BandOrderingHead)
  - Mask input (1=hole, 0=solid) instead of dielectric grid
  - TE polarization (correct for holes-in-Si band gaps)
  - Compares DualGridCANN_v5 vs DualGridCANN (v3 baseline)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pwe_torch import reciprocal_lattice, solve_bands, make_k_path
from fem_elastic_fast import solve_elastic_bands_fem
from cann_v5 import DualGridCANN_v5, DualGridCANN_v5_cross
from cann_v3 import DualGridCANN
from train_cann_v2 import (
    sample_perturbed_circles, sample_standard_shapes, make_fourier_eps_grid,
)
from train_cann_v3 import (
    eps_grid_to_mask,
    plot_dual_losses, plot_band_comparison,
)


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset generation (reuses generate_dataset_v3 logic, more samples)
# ---------------------------------------------------------------------------

def generate_dual_dataset(eps_grids, labels, n_bands_phot=6, n_bands_phon=10,
                          n_per_segment=10, n_max=5, mesh_res=30):
    """PWE photonic + FEM phononic ground truth for each eps grid.

    Returns masks (1=hole, 0=solid) as the geometry representation,
    plus photonic and phononic band targets.
    """
    g_vectors, m_indices = reciprocal_lattice(n_max)
    k_points, k_dist, tick_pos, tick_labels = make_k_path(n_per_segment)
    n_k = len(k_points)

    all_phot, all_phon, valid_idx = [], [], []
    for i, eps_grid in enumerate(eps_grids):
        try:
            with torch.no_grad():
                bands_phot = solve_bands(k_points, g_vectors, eps_grid,
                                         m_indices, n_bands_phot, "te")
            mask = eps_grid_to_mask(eps_grid)
            mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
            freqs_fem = solve_elastic_bands_fem(
                k_points, mask_np, n_bands_phon, mesh_res=mesh_res)
            all_phot.append(bands_phot.float())
            all_phon.append(torch.tensor(freqs_fem, dtype=torch.float32))
            valid_idx.append(i)
        except Exception as e:
            print(f"    SKIP {i} ({labels[i]}): {e}")
        if (i + 1) % 20 == 0:
            print(f"    solve {i+1}/{len(eps_grids)} ({len(valid_idx)} ok)")

    valid_masks = [eps_grid_to_mask(eps_grids[i]).float() for i in valid_idx]
    valid_labels = [labels[i] for i in valid_idx]
    masks_t = torch.stack(valid_masks, dim=0)
    b_phot_t = torch.stack(all_phot, dim=0)
    b_phon_t = torch.stack(all_phon, dim=0)
    meta = dict(k_dist=k_dist, tick_pos=tick_pos, tick_labels=tick_labels,
                n_k=n_k, n_bands_phot=n_bands_phot, n_bands_phon=n_bands_phon)
    return masks_t, b_phot_t, b_phon_t, valid_labels, meta


def mask_to_eps(mask: torch.Tensor, eps_bg: float = 8.9, eps_rod: float = 1.0):
    """Convert mask (1=hole) to dielectric grid."""
    return eps_bg - (eps_bg - eps_rod) * mask


GRID_SIZE = 32


def _resize_grid(g: torch.Tensor, N: int = GRID_SIZE) -> torch.Tensor:
    """Resize a (H, W) grid to (N, N) via bilinear interpolation."""
    if g.shape[-1] == N:
        return g
    return F.interpolate(g.unsqueeze(0).unsqueeze(0).float(),
                         size=N, mode="bilinear",
                         align_corners=False).squeeze(0).squeeze(0).to(g.dtype)


def sample_shapes(n_circles, n_standard, rng):
    """Sample n_circles perturbed circles + n_standard standard shapes.

    All grids returned at GRID_SIZE resolution.
    """
    coeffs = sample_perturbed_circles(
        n_circles, n_harmonics=4,
        r_mean_range=(0.10, 0.40), perturbation_scale=0.08, rng=rng)
    circle_grids = [make_fourier_eps_grid(c, N=GRID_SIZE) for c in coeffs]
    circle_labels = [f"circle {i}" for i in range(n_circles)]

    std = sample_standard_shapes(n_standard, rng=rng)
    std_grids = [_resize_grid(g) for g, _ in std]
    std_labels = [lbl for _, lbl in std]

    return circle_grids + std_grids, circle_labels + std_labels


# ---------------------------------------------------------------------------
# Per-band standardization
# ---------------------------------------------------------------------------

def compute_standardization(bands: torch.Tensor):
    """Compute per-(k, band) mean and std from training data.

    bands: (N, n_k, n_bands)
    Returns mean, std each of shape (1, n_k, n_bands).
    """
    mean = bands.mean(dim=0, keepdim=True)
    std = bands.std(dim=0, keepdim=True).clamp(min=1e-6)
    return mean, std


def standardize(bands, mean, std):
    return (bands - mean) / std


def unstandardize(bands, mean, std):
    return bands * std + mean


# ---------------------------------------------------------------------------
# Training loop (mini-batch, per-band standardized)
# ---------------------------------------------------------------------------

def train_v5(model, masks, b_phot, b_phon,
             n_epochs=3000, lr=1e-3, batch_size=256,
             weight_decay=1e-4, grad_clip=1.0, verbose=True,
             device=None):
    """Mini-batch training with per-band standardized targets."""
    if device is None:
        device = _get_device()

    phot_mean, phot_std = compute_standardization(b_phot)
    phon_mean, phon_std = compute_standardization(b_phon)
    tgt_phot = standardize(b_phot, phot_mean, phot_std).to(device)
    tgt_phon = standardize(b_phon, phon_mean, phon_std).to(device)
    masks = masks.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    n_samples = masks.shape[0]
    losses, losses_phot, losses_phon = [], [], []
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_samples)
        ep_loss = ep_phot = ep_phon = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            pred_phot, pred_phon = model(masks[idx])

            l_phot = nn.functional.mse_loss(pred_phot, tgt_phot[idx])
            l_phon = nn.functional.mse_loss(pred_phon, tgt_phon[idx])
            loss = l_phot + l_phon

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            ep_loss += loss.item()
            ep_phot += l_phot.item()
            ep_phon += l_phon.item()
            n_batches += 1

        scheduler.step()
        losses.append(ep_loss / n_batches)
        losses_phot.append(ep_phot / n_batches)
        losses_phon.append(ep_phon / n_batches)

        if verbose and (epoch + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  epoch {epoch+1:5d}  loss={losses[-1]:.6e}  "
                  f"phot={losses_phot[-1]:.6e}  phon={losses_phon[-1]:.6e}  "
                  f"[{elapsed:.0f}s]")

    stats = dict(phot_mean=phot_mean.cpu(), phot_std=phot_std.cpu(),
                 phon_mean=phon_mean.cpu(), phon_std=phon_std.cpu())
    return losses, losses_phot, losses_phon, stats


def train_v5_maxnorm(model, masks, b_phot, b_phon,
                     n_epochs=6000, lr=1e-3, batch_size=256,
                     weight_decay=1e-4, grad_clip=1.0, verbose=True,
                     device=None):
    """Mini-batch training with global-max normalization (for cumsum-head models)."""
    if device is None:
        device = _get_device()

    fs_phot = b_phot.max().item()
    fs_phon = b_phon.max().item()
    tgt_phot = (b_phot / fs_phot).to(device)
    tgt_phon = (b_phon / fs_phon).to(device)
    masks = masks.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    n_samples = masks.shape[0]
    losses, losses_phot, losses_phon = [], [], []
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_samples)
        ep_loss = ep_phot = ep_phon = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            pred_phot, pred_phon = model(masks[idx])

            l_phot = nn.functional.mse_loss(pred_phot / fs_phot, tgt_phot[idx])
            l_phon = nn.functional.mse_loss(pred_phon / fs_phon, tgt_phon[idx])
            loss = l_phot + l_phon

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            ep_loss += loss.item()
            ep_phot += l_phot.item()
            ep_phon += l_phon.item()
            n_batches += 1

        scheduler.step()
        losses.append(ep_loss / n_batches)
        losses_phot.append(ep_phot / n_batches)
        losses_phon.append(ep_phon / n_batches)

        if verbose and (epoch + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  epoch {epoch+1:5d}  loss={losses[-1]:.6e}  "
                  f"phot={losses_phot[-1]:.6e}  phon={losses_phon[-1]:.6e}  "
                  f"[{elapsed:.0f}s]")

    stats = dict(mode="maxnorm", fs_phot=fs_phot, fs_phon=fs_phon)
    return losses, losses_phot, losses_phon, stats


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def predict_unstd(model, masks, stats):
    """Run model on masks and convert predictions to original scale."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        p_phot, p_phon = model(masks.to(device))
    p_phot, p_phon = p_phot.cpu(), p_phon.cpu()
    if stats.get("mode") == "maxnorm":
        return p_phot, p_phon
    p_phot = unstandardize(p_phot, stats["phot_mean"], stats["phot_std"])
    p_phon = unstandardize(p_phon, stats["phon_mean"], stats["phon_std"])
    return p_phot, p_phon


def print_per_band_rmse(name, pred, gt, physics="phon"):
    n_bands = gt.shape[-1]
    per_band = ((pred - gt).pow(2).mean(dim=(0, 1))).sqrt()
    overall = (pred - gt).pow(2).mean().sqrt().item()
    band_str = "  ".join(f"b{i}={per_band[i].item():.4f}" for i in range(n_bands))
    print(f"  {name:>25s}  RMSE={overall:.6f}  [{band_str}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    n_bands_phot = 6
    n_bands_phon = 10

    dataset_path = "cann_v5_dataset.pt"

    # --- Generate or load dataset ---
    if os.path.exists(dataset_path):
        print(f"Loading {dataset_path}...")
        ds = torch.load(dataset_path, weights_only=False)
        masks_train = ds["masks_train"]
        b_phot_train = ds["b_phot_train"]
        b_phon_train = ds["b_phon_train"]
        masks_test = ds["masks_test"]
        b_phot_test = ds["b_phot_test"]
        b_phon_test = ds["b_phon_test"]
        train_labels = ds["train_labels"]
        test_labels = ds["test_labels"]
        meta = ds["meta"]
    else:
        print("Generating v5 dataset (500 train + 50 test)...")
        t0 = time.time()

        train_grids, train_labels = sample_shapes(200, 300, rng)
        test_grids, test_labels = sample_shapes(20, 30, rng)

        print(f"  Sampled {len(train_grids)} train, {len(test_grids)} test shapes")

        print("\n  Solving training set...")
        masks_train, b_phot_train, b_phon_train, train_labels, meta = \
            generate_dual_dataset(train_grids, train_labels,
                                  n_bands_phot=n_bands_phot,
                                  n_bands_phon=n_bands_phon)

        print("\n  Solving test set...")
        masks_test, b_phot_test, b_phon_test, test_labels, _ = \
            generate_dual_dataset(test_grids, test_labels,
                                  n_bands_phot=n_bands_phot,
                                  n_bands_phon=n_bands_phon)

        print(f"\n  Dataset generated in {time.time()-t0:.0f}s")
        print(f"  train: {len(masks_train)}, test: {len(masks_test)}")

        torch.save({
            "masks_train": masks_train, "b_phot_train": b_phot_train,
            "b_phon_train": b_phon_train,
            "masks_test": masks_test, "b_phot_test": b_phot_test,
            "b_phon_test": b_phon_test,
            "train_labels": train_labels, "test_labels": test_labels,
            "meta": meta,
        }, dataset_path)
        print(f"  Saved {dataset_path}")

    n_k = meta["n_k"]
    device = _get_device()
    print(f"  train: {len(masks_train)}, test: {len(masks_test)}, n_k={n_k}, device={device}")

    n_epochs = 3000

    # --- Build models ---
    model_v5 = DualGridCANN_v5(
        n_k=n_k, n_bands_phot=n_bands_phot, n_bands_phon=n_bands_phon,
        enc_dim=128)

    model_v5x = DualGridCANN_v5_cross(
        n_k=n_k, n_bands_phot=n_bands_phot, n_bands_phon=n_bands_phon,
        enc_dim=128)

    model_v3 = DualGridCANN(
        n_k=n_k, n_bands_phot=n_bands_phot, n_bands_phon=n_bands_phon,
        hidden=128)

    # --- Train v5 ---
    print(f"\nTraining DualGridCANN_v5 ({n_epochs} epochs)...")
    l_v5, l_v5_phot, l_v5_phon, stats_v5 = train_v5(
        model_v5, masks_train, b_phot_train, b_phon_train,
        n_epochs=n_epochs, device=device)

    # --- Train v5 cross ---
    print(f"\nTraining DualGridCANN_v5_cross ({n_epochs} epochs)...")
    l_v5x, l_v5x_phot, l_v5x_phon, stats_v5x = train_v5(
        model_v5x, masks_train, b_phot_train, b_phon_train,
        n_epochs=n_epochs, device=device)

    # --- Train v3 baseline (needs eps grids, convert from masks) ---
    print(f"\nTraining DualGridCANN v3 baseline ({n_epochs} epochs)...")
    eps_train = mask_to_eps(masks_train)
    l_v3, l_v3_phot, l_v3_phon, stats_v3 = train_v5_maxnorm(
        model_v3, eps_train, b_phot_train, b_phon_train,
        n_epochs=n_epochs, device=device)

    # --- Save checkpoints ---
    save_dir = "cann_v5_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    for name, model, stats in [("v5", model_v5, stats_v5),
                                ("v5_cross", model_v5x, stats_v5x),
                                ("v3_baseline", model_v3, stats_v3)]:
        model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        save_stats = {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                      for k, v in stats.items()}
        torch.save({
            "model_state": model_cpu,
            "stats": save_stats, "meta": meta,
        }, os.path.join(save_dir, f"{name}.pt"))
    print(f"\nModels saved to {save_dir}/")

    # --- Loss plots ---
    loss_data = {
        "total": [("v5", l_v5), ("v5 cross", l_v5x), ("v3 baseline", l_v3)],
        "phot":  [("v5", l_v5_phot), ("v5 cross", l_v5x_phot), ("v3 baseline", l_v3_phot)],
        "phon":  [("v5", l_v5_phon), ("v5 cross", l_v5x_phon), ("v3 baseline", l_v3_phon)],
    }
    plot_dual_losses(loss_data, save_path="cann_v5_losses.png")

    # --- Evaluate ---
    p_v5_phot, p_v5_phon = predict_unstd(model_v5, masks_test, stats_v5)
    p_v5x_phot, p_v5x_phon = predict_unstd(model_v5x, masks_test, stats_v5x)
    eps_test = mask_to_eps(masks_test)
    p_v3_phot, p_v3_phon = predict_unstd(model_v3, eps_test, stats_v3)

    # Pick diverse test samples
    shown = []
    seen_types = set()
    for i, lab in enumerate(test_labels):
        shape_type = lab.split()[0]
        if shape_type not in seen_types:
            seen_types.add(shape_type)
            shown.append(i)
        if len(shown) >= 6:
            break
    idx = torch.tensor(shown)
    show_masks = masks_test[idx]
    show_labels = [test_labels[i] for i in shown]

    plot_band_comparison(
        [("v5", b_phot_test[idx], p_v5_phot[idx]),
         ("v5 cross", b_phot_test[idx], p_v5x_phot[idx]),
         ("v3 baseline", b_phot_test[idx], p_v3_phot[idx])],
        meta, show_masks, show_labels, physics="phot",
        save_path="cann_v5_photonic_bands.png")

    plot_band_comparison(
        [("v5", b_phon_test[idx], p_v5_phon[idx]),
         ("v5 cross", b_phon_test[idx], p_v5x_phon[idx]),
         ("v3 baseline", b_phon_test[idx], p_v3_phon[idx])],
        meta, show_masks, show_labels, physics="phon",
        save_path="cann_v5_phononic_bands.png")

    # --- Metrics ---
    print("\n--- Photonic test metrics ---")
    for name, pred in [("v5", p_v5_phot), ("v5 cross", p_v5x_phot),
                        ("v3 baseline", p_v3_phot)]:
        print_per_band_rmse(name, pred, b_phot_test, "phot")

    print("\n--- Phononic test metrics ---")
    for name, pred in [("v5", p_v5_phon), ("v5 cross", p_v5x_phon),
                        ("v3 baseline", p_v3_phon)]:
        print_per_band_rmse(name, pred, b_phon_test, "phon")

    # --- Parameter counts ---
    print("\n--- Parameter counts ---")
    for name, m in [("v5", model_v5), ("v5 cross", model_v5x),
                    ("v3 baseline", model_v3)]:
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  {name:>25s}  {n_params:,}")


if __name__ == "__main__":
    main()
