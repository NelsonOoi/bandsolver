"""
Inverse design of 2D photonic/phononic crystals via surrogate optimization.

Given a band gap specification (which bands, target center frequency),
optimizes the unit cell geometry by backpropagating through a frozen
DualGridCANN_v5 surrogate.  Two methods are compared:

  1. Neural Style Transfer (NST): direct pixel-level mask optimization
     with sigmoid parameterization + total-variation regularization.
  2. Warm-start + RBF: scan training dataset for best seeds, fit RBF
     weights, refine via gradient descent in smooth RBF space.

Usage:
    python inverse_design.py
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pwe_torch import (
    reciprocal_lattice, solve_bands, make_k_path, extract_gap,
    smooth_max,
)
from fem_elastic_fast import solve_elastic_bands_fem
from cann_v5 import DualGridCANN_v5, DualGridCANN_v5_cross
from train_cann_v5 import unstandardize, mask_to_eps

GRID_SIZE = 32
N_BANDS_PHOT = 6
N_BANDS_PHON = 10


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Design specification
# ---------------------------------------------------------------------------

@dataclass
class GapSpec:
    """Specification for a target band gap.

    If band_lo and band_hi are None, automatically finds the best gap
    across all adjacent band pairs via smooth_max.
    """
    physics: str            # "phot" or "phon"
    band_lo: Optional[int] = None   # None = auto-detect
    band_hi: Optional[int] = None
    target_freq: Optional[float] = None
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Load frozen surrogate
# ---------------------------------------------------------------------------

def load_surrogate(checkpoint_path="cann_v5_checkpoints_0308_11am/v5_cross.pt",
                   device=None):
    """Load trained DualGridCANN_v5 or v5_cross, freeze, return (model, stats, meta)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"{checkpoint_path} not found. Run train_cann_v5.py first.")
    if device is None:
        device = _get_device()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    stats = ckpt["stats"]

    kwargs = dict(
        n_k=meta["n_k"],
        n_bands_phot=meta["n_bands_phot"],
        n_bands_phon=meta["n_bands_phon"],
        enc_dim=128,
    )
    is_cross = "cross" in os.path.basename(checkpoint_path)
    model_cls = DualGridCANN_v5_cross if is_cross else DualGridCANN_v5
    model = model_cls(**kwargs)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, stats, meta


# ---------------------------------------------------------------------------
# MaskRasterizer — RBF level-set producing a raw mask (no eps conversion)
# ---------------------------------------------------------------------------

class MaskRasterizer(nn.Module):
    """RBF level-set -> soft binary mask (1=hole, 0=solid).

    Same math as LevelSetRasterizer but outputs sigmoid(beta * phi) directly.
    """

    def __init__(self, n_rbf_side: int = 6, grid_size: int = 32,
                 sigma: float = 0.15,
                 beta_init: float = 5.0, beta_final: float = 40.0):
        super().__init__()
        self.n_rbf_side = n_rbf_side
        self.n_rbf = n_rbf_side ** 2
        self.grid_size = grid_size
        self.beta_init = beta_init
        self.beta_final = beta_final
        self.beta = beta_init

        cs = torch.linspace(-0.5 + 0.5 / n_rbf_side,
                            0.5 - 0.5 / n_rbf_side, n_rbf_side)
        cy, cx = torch.meshgrid(cs, cs, indexing="ij")
        centers = torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=-1)
        self.register_buffer("centers", centers)

        xs = torch.linspace(-0.5, 0.5, grid_size + 1)[:-1] + 0.5 / grid_size
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")
        grid_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        self.register_buffer("grid_xy", grid_xy)

        diff = grid_xy.unsqueeze(1) - centers.unsqueeze(0)
        rbf_vals = torch.exp(-0.5 * (diff ** 2).sum(dim=-1) / sigma ** 2)
        self.register_buffer("rbf_vals", rbf_vals)

    def set_beta(self, frac: float):
        self.beta = self.beta_init + frac * (self.beta_final - self.beta_init)

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """weights: (batch, n_rbf) -> (batch, G, G) soft mask."""
        phi = torch.einsum("gn,bn->bg", self.rbf_vals, weights)
        phi = phi.view(-1, self.grid_size, self.grid_size)
        return torch.sigmoid(self.beta * phi)

    def fit(self, mask: torch.Tensor, n_iter: int = 200, lr: float = 0.1):
        """Least-squares fit weights to reproduce a given mask.

        mask: (G, G) or (B, G, G) binary mask. Returns (B, n_rbf) weights.
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        B = mask.shape[0]
        target = mask.detach().to(self.rbf_vals.device)
        w = torch.zeros(B, self.n_rbf, device=target.device, requires_grad=True)
        opt = torch.optim.Adam([w], lr=lr)
        for _ in range(n_iter):
            pred = self.forward(w)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return w.detach()


# ---------------------------------------------------------------------------
# Shared objective
# ---------------------------------------------------------------------------

def _unstd_predict(model, mask, stats):
    """Forward through surrogate, unstandardize to physical units.

    mask: (B, G, G) soft mask with grad.
    Returns phot_bands (B, n_k, n_bands_phot), phon_bands (B, n_k, n_bands_phon).
    """
    device = next(model.parameters()).device
    pred_phot, pred_phon = model(mask.to(device))

    phot_mean = stats["phot_mean"].to(device)
    phot_std = stats["phot_std"].to(device)
    phon_mean = stats["phon_mean"].to(device)
    phon_std = stats["phon_std"].to(device)

    return (unstandardize(pred_phot, phot_mean, phot_std).clamp(min=0.0),
            unstandardize(pred_phon, phon_mean, phon_std).clamp(min=0.0))


def _best_gap_auto(bands_single, n_bands):
    """Find the best relative gap across all adjacent band pairs.

    bands_single: (n_k, n_bands) for one sample.
    Returns (best_rel_gap, best_gap_width, best_midgap, best_lo, best_hi).
    """
    rel_gaps = []
    for i in range(n_bands - 1):
        gw, mg = extract_gap(bands_single, i, i + 1)
        rg = gw / mg.clamp(min=1e-6)
        rel_gaps.append(rg)
    rel_gaps_t = torch.stack(rel_gaps)
    best_rg = smooth_max(rel_gaps_t, beta=20.0)
    best_idx = rel_gaps_t.detach().argmax().item()
    gw, mg = extract_gap(bands_single, best_idx, best_idx + 1)
    return best_rg, gw, mg, best_idx, best_idx + 1


def gap_loss(phot_bands, phon_bands, specs, alpha_freq=10.0):
    """Compute total loss from gap specifications.

    If spec.band_lo/band_hi are None, auto-detects the best gap across
    all adjacent band pairs via smooth_max.  Uses relative gap so
    photonic and phononic objectives are on the same dimensionless scale.

    Returns (total_loss, info_list).
    """
    loss = torch.tensor(0.0, device=phot_bands.device)
    info = []
    B = phot_bands.shape[0]

    for spec in specs:
        bands = phot_bands if spec.physics == "phot" else phon_bands
        n_bands = bands.shape[-1]
        auto = spec.band_lo is None

        rel_gaps, gap_widths, midgaps = [], [], []
        best_pairs = []
        for b in range(B):
            if auto:
                rg, gw, mg, lo, hi = _best_gap_auto(bands[b], n_bands)
                best_pairs.append((lo, hi))
            else:
                gw, mg = extract_gap(bands[b], spec.band_lo, spec.band_hi)
                rg = gw / mg.clamp(min=1e-6)
                best_pairs.append((spec.band_lo, spec.band_hi))
            rel_gaps.append(rg)
            gap_widths.append(gw)
            midgaps.append(mg)
        rel_gaps = torch.stack(rel_gaps)
        gap_widths = torch.stack(gap_widths)
        midgaps = torch.stack(midgaps)

        spec_loss = -rel_gaps
        if spec.target_freq is not None:
            spec_loss = spec_loss + alpha_freq * (midgaps - spec.target_freq) ** 2
        loss = loss + spec.weight * spec_loss.mean()
        info.append(dict(physics=spec.physics, gap_widths=gap_widths.detach(),
                         midgaps=midgaps.detach(), rel_gap=rel_gaps.detach(),
                         best_pairs=best_pairs))
    return loss, info


def total_variation(mask):
    """Anisotropic total variation of a (B, H, W) mask."""
    return ((mask[:, 1:, :] - mask[:, :-1, :]).abs().mean() +
            (mask[:, :, 1:] - mask[:, :, :-1]).abs().mean())


# ---------------------------------------------------------------------------
# Method 1: Neural Style Transfer (direct mask optimization)
# ---------------------------------------------------------------------------

def optimize_nst(model, stats, specs, n_starts=32, n_steps=500,
                 lr=0.05, tv_weight=0.01, device=None):
    """Optimize mask pixels directly through frozen surrogate."""
    if device is None:
        device = _get_device()
    print(f"\n{'='*60}")
    print("Method 1: Neural Style Transfer (direct mask optimization)")
    print(f"  n_starts={n_starts}, n_steps={n_steps}, lr={lr}")
    print(f"{'='*60}")

    G = GRID_SIZE
    z = torch.randn(n_starts, G, G, device=device) * 0.1
    z = nn.Parameter(z)
    optimizer = torch.optim.Adam([z], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.01)

    beta_init, beta_final = 5.0, 40.0
    best_loss = float("inf")
    best_mask = None
    t0 = time.time()

    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        beta = beta_init + frac * (beta_final - beta_init)
        mask = torch.sigmoid(beta * z)

        phot_bands, phon_bands = _unstd_predict(model, mask, stats)
        loss, info = gap_loss(phot_bands, phon_bands, specs)
        loss = loss + tv_weight * total_variation(mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 100 == 0 or step == 0:
            gap_strs = []
            for si, spec in enumerate(specs):
                rg = info[si]["rel_gap"]
                mg = info[si]["midgaps"]
                best_idx = rg.argmax().item()
                lo, hi = info[si]["best_pairs"][best_idx]
                gap_strs.append(
                    f"{spec.physics} b{lo}-{hi}: "
                    f"relgap={rg[best_idx]:.4f} mid={mg[best_idx]:.4f}")
            print(f"  step {step+1:4d}  loss={loss.item():.4f}  "
                  f"beta={beta:.1f}  {' | '.join(gap_strs)}  "
                  f"[{time.time()-t0:.1f}s]")

        with torch.no_grad():
            curr_mask = torch.sigmoid(beta_final * z)
            p, q = _unstd_predict(model, curr_mask, stats)
            _, final_info = gap_loss(p, q, specs)
            total_rg = sum(fi["rel_gap"] for fi in final_info)
            best_idx = total_rg.argmax().item()
            if total_rg[best_idx].item() > -best_loss:
                best_loss = -total_rg[best_idx].item()
                best_mask = curr_mask[best_idx].detach().cpu()

    final_mask = (best_mask > 0.5).float()
    print(f"\n  Best NST mask: total rel_gap = {-best_loss:.4f}")
    return final_mask, best_mask


# ---------------------------------------------------------------------------
# Method 2: Warm-start + RBF refinement
# ---------------------------------------------------------------------------

def _scan_dataset(model, stats, specs, dataset_path="cann_v5_dataset.pt",
                  device=None):
    """Evaluate all training masks, rank by gap quality."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"{dataset_path} not found. Run train_cann_v5.py first.")
    if device is None:
        device = _get_device()

    ds = torch.load(dataset_path, map_location="cpu", weights_only=False)
    masks = ds["masks_train"]
    n = masks.shape[0]

    model.eval()
    with torch.no_grad():
        phot, phon = _unstd_predict(model, masks, stats)
    _, info = gap_loss(phot, phon, specs)

    score = torch.zeros(n)
    for si, spec in enumerate(specs):
        rg = info[si]["rel_gap"].cpu()
        mg = info[si]["midgaps"].cpu()
        s = rg.clone()
        if spec.target_freq is not None:
            s = s - 5.0 * (mg - spec.target_freq).abs()
        score += spec.weight * s

    return masks, score


def optimize_warmstart(model, stats, specs, n_seeds=8, n_steps=500,
                       lr=0.05, device=None):
    """Warm-start from dataset, refine in RBF space."""
    if device is None:
        device = _get_device()
    print(f"\n{'='*60}")
    print("Method 2: Warm-start + RBF refinement")
    print(f"  n_seeds={n_seeds}, n_steps={n_steps}, lr={lr}")
    print(f"{'='*60}")

    masks_all, scores = _scan_dataset(model, stats, specs, device=device)
    top_k = scores.topk(n_seeds).indices
    seed_masks = masks_all[top_k]

    print(f"  Top-{n_seeds} dataset scores: "
          f"{scores[top_k].tolist()}")

    rasterizer = MaskRasterizer(n_rbf_side=6, grid_size=GRID_SIZE).to(device)

    print("  Fitting RBF weights to seed masks...")
    weights = rasterizer.fit(seed_masks.to(device), n_iter=300, lr=0.1)
    weights = nn.Parameter(weights.clone())

    optimizer = torch.optim.Adam([weights], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.01)

    best_loss = float("inf")
    best_mask = None
    t0 = time.time()

    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        rasterizer.set_beta(frac)
        mask = rasterizer(weights)

        phot_bands, phon_bands = _unstd_predict(model, mask, stats)
        loss, info = gap_loss(phot_bands, phon_bands, specs)
        loss = loss + 1e-3 * weights.pow(2).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 100 == 0 or step == 0:
            gap_strs = []
            for si, spec in enumerate(specs):
                rg = info[si]["rel_gap"]
                mg = info[si]["midgaps"]
                best_idx = rg.argmax().item()
                lo, hi = info[si]["best_pairs"][best_idx]
                gap_strs.append(
                    f"{spec.physics} b{lo}-{hi}: "
                    f"relgap={rg[best_idx]:.4f} mid={mg[best_idx]:.4f}")
            print(f"  step {step+1:4d}  loss={loss.item():.4f}  "
                  f"{' | '.join(gap_strs)}  [{time.time()-t0:.1f}s]")

        with torch.no_grad():
            rasterizer.set_beta(1.0)
            curr_mask = rasterizer(weights)
            rasterizer.set_beta(frac)
            p, q = _unstd_predict(model, curr_mask, stats)
            _, final_info = gap_loss(p, q, specs)
            total_rg = sum(fi["rel_gap"] for fi in final_info)
            best_idx = total_rg.argmax().item()
            if total_rg[best_idx].item() > -best_loss:
                best_loss = -total_rg[best_idx].item()
                best_mask = curr_mask[best_idx].detach().cpu()

    final_mask = (best_mask > 0.5).float()
    print(f"\n  Best warm-start mask: total rel_gap = {-best_loss:.4f}")
    return final_mask, best_mask


# ---------------------------------------------------------------------------
# Ground-truth verification
# ---------------------------------------------------------------------------

def verify(mask, meta, n_max=5):
    """Run PWE (TE) and FEM on a binarized mask. Returns (phot_bands, phon_bands)."""
    k_points, _, _, _ = make_k_path(n_per_segment=10)
    g_vectors, m_indices = reciprocal_lattice(n_max)

    eps_grid = mask_to_eps(mask)
    with torch.no_grad():
        phot = solve_bands(k_points, g_vectors, eps_grid.double(),
                           m_indices, N_BANDS_PHOT, "te").float()

    mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
    phon_np = solve_elastic_bands_fem(k_points, mask_np, N_BANDS_PHON,
                                      mesh_res=40)
    phon = torch.tensor(phon_np, dtype=torch.float32)

    return phot, phon


def _extract_gap_numpy(bands, band_lo, band_hi):
    """Non-differentiable gap extraction for GT bands."""
    floor = bands[:, band_lo].max()
    ceil = bands[:, band_hi].min()
    gap = ceil - floor
    midgap = 0.5 * (ceil + floor)
    return gap, midgap


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_result(results, meta, specs, save_path="inverse_design_result.png"):
    """Plot optimized geometries and band structures for each method.

    results: list of (method_name, mask, surr_phot, surr_phon, gt_phot, gt_phon)
    """
    k_dist = meta["k_dist"]
    tick_pos = meta["tick_pos"]
    tick_labels = meta["tick_labels"]

    n_methods = len(results)
    fig, axes = plt.subplots(n_methods, 3, figsize=(16, 5 * n_methods),
                             squeeze=False)

    for row, (name, mask, s_phot, s_phon, gt_phot, gt_phon) in enumerate(results):
        # Geometry
        ax = axes[row, 0]
        ax.imshow(mask.numpy(), cmap="gray_r", origin="lower",
                  extent=[-0.5, 0.5, -0.5, 0.5])
        ax.set_title(f"{name}\nGeometry (white=hole)")
        ax.set_xlabel("x/a")
        ax.set_ylabel("y/a")

        for col, (physics, gt, surr, ylabel, color) in enumerate([
            ("phot", gt_phot, s_phot, "$\\omega a / 2\\pi c$", "blue"),
            ("phon", gt_phon, s_phon, "Frequency (Hz·a)", "green"),
        ], start=1):
            ax = axes[row, col]
            gt_label = "PWE" if physics == "phot" else "FEM"
            for b in range(gt.shape[1]):
                ax.plot(k_dist, gt[:, b].numpy(), "k-", lw=1.5,
                        label=gt_label if b == 0 else None)
                ax.plot(k_dist, surr[:, b].numpy(), "r--", lw=1.0, alpha=0.7,
                        label="CANN" if b == 0 else None)
            n_b = gt.shape[1]
            best_rg, best_lo = -1e9, 0
            for i in range(n_b - 1):
                g, m = _extract_gap_numpy(gt, i, i + 1)
                rg = g / max(abs(m), 1e-9)
                if rg > best_rg:
                    best_rg, best_lo = rg, i
            gap, mid = _extract_gap_numpy(gt, best_lo, best_lo + 1)
            if gap > 0:
                lo_val = gt[:, best_lo].max().item()
                hi_val = gt[:, best_lo + 1].min().item()
                ax.axhspan(lo_val, hi_val, alpha=0.2, color=color,
                           label=f"b{best_lo}-{best_lo+1} gap={gap:.4f}")
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels)
            ax.set_ylabel(ylabel)
            phys_label = "Photonic TE" if physics == "phot" else "Phononic"
            ax.set_title(f"{name} — {phys_label}")
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    device = _get_device()
    print(f"Device: {device}")

    model, stats, meta = load_surrogate(device=device)
    print(f"Surrogate loaded: n_k={meta['n_k']}, "
          f"n_bands_phot={meta['n_bands_phot']}, "
          f"n_bands_phon={meta['n_bands_phon']}")

    specs = [
        GapSpec(physics="phot", weight=1.0),
        GapSpec(physics="phon", weight=1.0),
    ]

    print("\nDesign specs:")
    for s in specs:
        bands_str = f"bands {s.band_lo}-{s.band_hi}" if s.band_lo is not None else "auto"
        tgt = f"at freq={s.target_freq}" if s.target_freq else "(any freq)"
        print(f"  {s.physics} gap [{bands_str}] {tgt}, weight={s.weight}")

    # --- Method 1: NST ---
    mask_nst, soft_nst = optimize_nst(
        model, stats, specs, n_starts=32, n_steps=500, device=device)

    # --- Method 2: Warm-start + RBF ---
    mask_ws, soft_ws = optimize_warmstart(
        model, stats, specs, n_seeds=8, n_steps=500, device=device)

    # --- Surrogate predictions on final masks ---
    results = []
    for name, mask in [("NST", mask_nst), ("Warm-start+RBF", mask_ws)]:
        print(f"\nVerifying {name}...")
        with torch.no_grad():
            s_phot, s_phon = _unstd_predict(
                model, mask.unsqueeze(0).to(device), stats)
        s_phot, s_phon = s_phot[0].cpu(), s_phon[0].cpu()

        gt_phot, gt_phon = verify(mask, meta)

        print(f"  {name} — ground truth verification:")
        for spec in specs:
            gt_bands = gt_phot if spec.physics == "phot" else gt_phon
            surr_bands = s_phot if spec.physics == "phot" else s_phon
            label = "PWE" if spec.physics == "phot" else "FEM"
            n_b = gt_bands.shape[1]

            best_gt_rg, best_lo = -1e9, 0
            for i in range(n_b - 1):
                g, m = _extract_gap_numpy(gt_bands, i, i + 1)
                rg = g / max(abs(m), 1e-9)
                if rg > best_gt_rg:
                    best_gt_rg, best_lo = rg, i
            gap, mid = _extract_gap_numpy(gt_bands, best_lo, best_lo + 1)
            s_gap, s_mid = _extract_gap_numpy(surr_bands, best_lo, best_lo + 1)
            rg = gap / max(abs(mid), 1e-9)
            s_rg = s_gap / max(abs(s_mid), 1e-9)
            print(f"    {spec.physics} b{best_lo}-{best_lo+1}: "
                  f"CANN gap={s_gap:.4f} relgap={s_rg:.4f} | "
                  f"{label} gap={gap:.4f} relgap={rg:.4f}")

        results.append((name, mask, s_phot, s_phon, gt_phot, gt_phon))

    plot_result(results, meta, specs)
    print("\nDone.")


if __name__ == "__main__":
    main()
