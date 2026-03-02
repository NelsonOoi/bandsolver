"""
Fourier-space physics-embedded inverse design network.

Outputs Fourier coefficients of the dielectric with C4v symmetry,
builds the Toeplitz matrix as an explicit differentiable layer,
and trains multi-target on sampled frequencies.

Usage:
    python pinn_fourier.py --steps 500 --lr 1e-3 --n-grid 16 --n-max 5
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

from pwe_torch import (reciprocal_lattice, make_k_path, _solve_bands_from_inv,
                        extract_gap, smooth_min, smooth_max)


# ===================================================================
# 1. C4v Fourier Orbits
# ===================================================================

def c4v_fourier_orbits(N: int):
    """Compute independent Fourier coefficient orbits under C4v symmetry.

    The C4v group on (m1, m2) mod N consists of 8 operations:
        identity, 3 rotations (90, 180, 270), and 4 reflections.

    Since the dielectric is real, eps_hat(-G) = eps_hat(G)*, and with
    C4v symmetry all coefficients in an orbit share the same real value.

    Args:
        N: grid size (should be even).

    Returns:
        orbits: list of lists, each containing (i, j) tuples in the orbit.
        n_indep: number of independent real coefficients.
    """
    visited = set()
    orbits = []

    for m1 in range(N):
        for m2 in range(N):
            if (m1, m2) in visited:
                continue
            orbit = set()
            # C4v generators: 4 rotations x {identity, reflection}
            pairs = [
                (m1, m2), (m2, (-m1) % N),          # rot 0, 90
                ((-m1) % N, (-m2) % N), ((-m2) % N, m1),  # rot 180, 270
                (m2, m1), (m1, (-m2) % N),           # reflections
                ((-m2) % N, (-m1) % N), ((-m1) % N, m2),
            ]
            # Also add negatives (reality constraint: eps_hat(-G) = eps_hat(G)*)
            all_pairs = []
            for p in pairs:
                all_pairs.append(p)
                all_pairs.append(((-p[0]) % N, (-p[1]) % N))

            for p in all_pairs:
                orbit.add(p)

            visited.update(orbit)
            orbits.append(sorted(orbit))

    return orbits, len(orbits)


# ===================================================================
# 2. Fourier-to-Toeplitz Layer
# ===================================================================

def build_eps_mat_from_fourier(eps_fft_grid: torch.Tensor,
                               m_indices_np: np.ndarray):
    """Build Toeplitz epsilon matrix directly from Fourier coefficients.

    Same indexing as pwe_torch.build_epsilon_matrix but skips the FFT --
    the Fourier grid is the direct network output.

    Args:
        eps_fft_grid: (N, N) tensor of Fourier coefficients eps_hat(m1, m2).
        m_indices_np: (n_pw, 2) integer plane-wave indices.

    Returns:
        eps_mat: (n_pw, n_pw) complex tensor.
    """
    N = eps_fft_grid.shape[0]
    dm = m_indices_np[:, None, :] - m_indices_np[None, :, :]
    idx0 = torch.from_numpy(dm[..., 0] % N).long()
    idx1 = torch.from_numpy(dm[..., 1] % N).long()
    eps_mat = eps_fft_grid[idx0, idx1]
    return eps_mat


# ===================================================================
# 3. FourierInverseNet
# ===================================================================

class FourierInverseNet(nn.Module):
    """Maps target_freq (scalar) -> Fourier coefficients -> eps grid.

    Architecture:
        Encoder:  Linear(1->64)->ReLU->Linear(64->128)->ReLU->Linear(128->latent)
        Decoder:  Linear(latent->128)->ReLU->Linear(128->n_indep)
        C4v expand: scatter n_indep values into full N x N Fourier grid
        IFFT:     N x N Fourier grid -> real-space eps grid (for visualization)
    """

    def __init__(self, N=16, latent_dim=32, eps_bg=1.0, eps_rod=8.9):
        super().__init__()
        self.N = N
        self.eps_bg = eps_bg
        self.eps_rod = eps_rod

        # Precompute C4v orbits
        self.orbits, self.n_indep = c4v_fourier_orbits(N)

        # Build scatter indices: for each orbit, store all (i,j) members
        # orbit_map[k] -> list of (i,j) in orbit k
        flat_orbit_idx = torch.zeros(N, N, dtype=torch.long)
        for k, orb in enumerate(self.orbits):
            for (i, j) in orb:
                flat_orbit_idx[i, j] = k
        self.register_buffer("flat_orbit_idx", flat_orbit_idx)

        # Find which orbit contains (0, 0) -- the DC component
        self.dc_orbit = int(flat_orbit_idx[0, 0].item())

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_indep),
        )

    def forward(self, target_freq: torch.Tensor):
        """Generate Fourier coefficient grid and real-space grid.

        Args:
            target_freq: scalar tensor.

        Returns:
            eps_fft_grid: (N, N) complex tensor of Fourier coefficients.
            eps_grid: (N, N) real tensor from IFFT (for visualization/penalties).
        """
        x = target_freq.view(1).to(torch.float64)
        z = self.encoder(x)
        coeffs = self.decoder(z)  # (n_indep,)

        # DC component: sigmoid -> affine to [eps_bg, eps_rod]
        dc_val = torch.sigmoid(coeffs[self.dc_orbit])
        dc_val = self.eps_bg + (self.eps_rod - self.eps_bg) * dc_val

        # Build full N x N grid by scattering orbit values
        grid_flat = coeffs[self.flat_orbit_idx]  # (N, N)

        # Override DC orbit with constrained value
        dc_mask = (self.flat_orbit_idx == self.dc_orbit)
        grid_flat = torch.where(dc_mask, dc_val.expand_as(grid_flat), grid_flat)

        eps_fft_grid = grid_flat.to(torch.complex128)

        # Real-space via IFFT (for visualization and penalties)
        eps_grid = torch.fft.ifft2(eps_fft_grid * (self.N * self.N)).real

        return eps_fft_grid, eps_grid


# ===================================================================
# 4. Splitting-Aware Soft Gap Selection
# ===================================================================

def soft_gap_selection(bands: torch.Tensor, target_freq: float,
                       temperature: float = 0.05, alpha: float = 1.0,
                       beta: float = 50.0):
    """Select the most promising band gap across all consecutive pairs.

    For each consecutive pair (n, n+1):
        - split_n(k) = band_{n+1}(k) - band_n(k)
        - avg_split = mean over k
        - min_split = smooth_min over k
        - mid_n = 0.5 * mean(band_{n+1} + band_n)
        - score = -|mid - target| / temperature + alpha * avg_split

    Softmax over scores yields weights w_n for soft selection.

    Returns:
        effective_split: weighted min-split across band pairs.
        effective_mid: weighted midgap frequency.
        weights: (n_pairs,) softmax weights for diagnostics.
    """
    n_bands = bands.shape[1]
    n_pairs = n_bands - 1

    min_splits = []
    avg_splits = []
    mids = []

    for n in range(n_pairs):
        split_k = bands[:, n + 1] - bands[:, n]  # (n_k,)
        avg_s = split_k.mean()
        min_s = smooth_min(split_k, beta=-beta)
        mid = 0.5 * (bands[:, n + 1] + bands[:, n]).mean()

        avg_splits.append(avg_s)
        min_splits.append(min_s)
        mids.append(mid)

    avg_splits = torch.stack(avg_splits)
    min_splits = torch.stack(min_splits)
    mids = torch.stack(mids)

    # Score: prefer pairs close to target with large splits
    freq_dist = torch.abs(mids - target_freq)
    scores = -freq_dist / temperature + alpha * avg_splits
    scores = torch.clamp(scores, min=-80.0, max=80.0)
    weights = torch.softmax(scores, dim=0)

    effective_split = (weights * min_splits).sum()
    effective_mid = (weights * mids).sum()

    return effective_split, effective_mid, weights


# ===================================================================
# 5. Training Loop
# ===================================================================

def train_fourier_inverse(cfg):
    """Multi-target training of the Fourier inverse design network."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    g_vectors, m_indices = reciprocal_lattice(cfg.n_max)
    k_points, k_dist, tick_pos, tick_labels = make_k_path(cfg.n_k_seg)

    model = FourierInverseNet(
        N=cfg.n_grid, latent_dim=cfg.latent_dim,
        eps_bg=cfg.eps_bg, eps_rod=cfg.eps_rod,
    ).double()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.steps)

    # Binary penalty warmup
    binary_schedule = np.linspace(cfg.w_binary * 0.1, cfg.w_binary, cfg.steps)

    history = []
    print(f"Fourier Inverse Design | Grid: {cfg.n_grid} | Steps: {cfg.steps}")
    print(f"Target freq range: [{cfg.freq_lo:.3f}, {cfg.freq_hi:.3f}]")
    print(f"Independent Fourier coefficients: {model.n_indep}")
    print("-" * 60)

    for step in range(cfg.steps):
        t0 = time.time()
        optimizer.zero_grad()

        # Sample target frequency
        target_freq = cfg.freq_lo + (cfg.freq_hi - cfg.freq_lo) * np.random.rand()
        target_freq_t = torch.tensor(target_freq, dtype=torch.float64)

        w_bin = float(binary_schedule[step])

        # Forward pass (all inside autograd graph)
        eps_fft, eps_grid = model(target_freq_t)
        eps_mat = build_eps_mat_from_fourier(eps_fft, m_indices)
        eps_mat_inv = torch.linalg.inv(eps_mat)
        bands = _solve_bands_from_inv(k_points, g_vectors, eps_mat_inv,
                                      cfg.n_bands, "tm")

        # Splitting-aware gap selection
        eff_split, eff_mid, gap_weights = soft_gap_selection(
            bands, target_freq, temperature=cfg.temperature, alpha=cfg.alpha
        )

        # Loss
        loss_gap = -cfg.w_gap * eff_split
        loss_freq = cfg.w_freq * (eff_mid - target_freq) ** 2

        # Binary penalty on real-space grid
        eps_norm = (eps_grid - eps_grid.min()) / (eps_grid.max() - eps_grid.min() + 1e-8)
        loss_binary = w_bin * torch.mean(eps_norm * (1.0 - eps_norm))

        # Positivity penalty
        loss_positive = cfg.w_positive * torch.mean(F.relu(-eps_grid))

        total_loss = loss_gap + loss_freq + loss_binary + loss_positive

        total_loss.backward()

        # NaN guard
        has_nan = any(
            p.grad is not None and torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        if has_nan or torch.isnan(total_loss):
            optimizer.zero_grad()
            scheduler.step()
            elapsed = time.time() - t0
            info = {"step": step, "total": float("nan"), "gap": 0.0,
                    "mid": 0.0, "target": target_freq, "time": elapsed,
                    "nan": True}
            history.append(info)
            if (step + 1) % max(1, cfg.steps // 20) == 0:
                print(f"[{step+1:4d}/{cfg.steps}] NaN detected, skipping")
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t0
        info = {
            "step": step,
            "total": total_loss.item(),
            "loss_gap": loss_gap.item(),
            "loss_freq": loss_freq.item(),
            "loss_binary": loss_binary.item(),
            "gap": eff_split.item(),
            "mid": eff_mid.item(),
            "target": target_freq,
            "time": elapsed,
            "nan": False,
        }
        history.append(info)

        if (step + 1) % max(1, cfg.steps // 20) == 0 or step == 0:
            best_pair = int(gap_weights.argmax().item())
            print(f"[{step+1:4d}/{cfg.steps}] loss={info['total']:.5f}  "
                  f"gap={info['gap']:.5f}  mid={info['mid']:.4f}  "
                  f"tgt={target_freq:.4f}  pair={best_pair}-{best_pair+1}  "
                  f"({elapsed:.2f}s)")

    # Save checkpoint
    ckpt_path = "pinn_fourier_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": vars(cfg),
        "history": history,
        "step": cfg.steps,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    evaluate_model(model, cfg, g_vectors, m_indices, k_points, k_dist,
                   tick_pos, tick_labels, history)


# ===================================================================
# 6. Evaluation
# ===================================================================

def evaluate_model(model, cfg, g_vectors, m_indices, k_points, k_dist,
                   tick_pos, tick_labels, history):
    """Generalization sweep, interpolation demo, comparison plots."""
    test_freqs = np.linspace(cfg.freq_lo, cfg.freq_hi, 9)
    results = []

    print(f"{'target':>8s}  {'gap_width':>9s}  {'midgap':>8s}  {'best_pair':>9s}")
    print("-" * 40)

    for tf in test_freqs:
        tf_t = torch.tensor(tf, dtype=torch.float64)
        with torch.no_grad():
            eps_fft, eps_grid = model(tf_t)
            eps_mat = build_eps_mat_from_fourier(eps_fft, m_indices)
            eps_mat_inv = torch.linalg.inv(eps_mat)
            bands = _solve_bands_from_inv(k_points, g_vectors, eps_mat_inv,
                                          cfg.n_bands, "tm")

        # Find best gap across all consecutive pairs
        bands_np = bands.detach().cpu().numpy()
        best_gw, best_mid, best_pair = 0.0, 0.0, 0
        for n in range(bands_np.shape[1] - 1):
            floor_n = np.max(bands_np[:, n])
            ceil_n = np.min(bands_np[:, n + 1])
            gw = max(0.0, ceil_n - floor_n)
            if gw > best_gw:
                best_gw = gw
                best_mid = 0.5 * (ceil_n + floor_n)
                best_pair = n

        results.append({
            "target": tf,
            "gap_width": best_gw,
            "midgap": best_mid,
            "best_pair": best_pair,
            "bands": bands_np,
            "eps_grid": eps_grid.detach().cpu().numpy(),
        })
        print(f"{tf:8.4f}  {best_gw:9.5f}  {best_mid:8.4f}  {best_pair:5d}-{best_pair+1}")

    # Plots
    _plot_training(results, history, k_dist, tick_pos, tick_labels, cfg)
    _plot_evaluation_grid(results, k_dist, tick_pos, tick_labels, cfg)
    _plot_interpolation(model, cfg, m_indices, g_vectors, k_points,
                        k_dist, tick_pos, tick_labels)


# ===================================================================
# 7. Plotting
# ===================================================================

def _plot_training(results, history, k_dist, tick_pos, tick_labels, cfg):
    """3-panel training plot: eps_grid, bands with gap, loss convergence."""
    # Use middle test result for display
    mid_res = results[len(results) // 2]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: epsilon grid
    ax = axes[0]
    im = ax.imshow(mid_res["eps_grid"].T, origin="lower",
                   extent=[0, 1, 0, 1], cmap="RdYlBu_r")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"Unit cell (f_tgt={mid_res['target']:.3f})")
    ax.set_xlabel("x/a"); ax.set_ylabel("y/a"); ax.set_aspect("equal")

    # Panel 2: band structure
    ax = axes[1]
    bands_np = mid_res["bands"]
    for i in range(bands_np.shape[1]):
        ax.plot(k_dist, bands_np[:, i], color="#2563eb", linewidth=0.9)
    for tp in tick_pos:
        ax.axvline(tp, color="grey", linewidth=0.4, linestyle="--")
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels)
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylabel("$\\omega a / 2\\pi c$"); ax.set_title("TM bands")
    if mid_res["gap_width"] > 0:
        bp = mid_res["best_pair"]
        floor_v = np.max(bands_np[:, bp])
        ceil_v = np.min(bands_np[:, bp + 1])
        ax.axhspan(floor_v, ceil_v, alpha=0.15, color="#2563eb")
    ax.axhline(mid_res["target"], color="red", linewidth=0.7, linestyle=":")

    # Panel 3: loss convergence
    ax = axes[2]
    valid = [h for h in history if not h["nan"]]
    if valid:
        steps = [h["step"] for h in valid]
        ax.plot(steps, [h["total"] for h in valid], label="total", linewidth=0.8)
        ax.plot(steps, [abs(h["loss_gap"]) for h in valid],
                label="|gap|", alpha=0.7, linewidth=0.8)
        ax.plot(steps, [h["loss_freq"] for h in valid],
                label="freq", alpha=0.7, linewidth=0.8)
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Convergence")
    ax.legend(fontsize=8); ax.set_yscale("symlog", linthresh=1e-4)

    fig.tight_layout()
    fig.savefig("pinn_fourier_training.png", dpi=150)
    print(f"Saved pinn_fourier_training.png")
    plt.close(fig)


def _plot_evaluation_grid(results, k_dist, tick_pos, tick_labels, cfg):
    """3x3 grid of (geometry, bands) for different target frequencies."""
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(4 * cols * 2, 3.5 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for idx, res in enumerate(results):
        r, c = divmod(idx, cols)

        # Geometry
        ax = axes[r, c * 2]
        im = ax.imshow(res["eps_grid"].T, origin="lower",
                       extent=[0, 1, 0, 1], cmap="RdYlBu_r")
        ax.set_title(f"f={res['target']:.3f}", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

        # Bands
        ax = axes[r, c * 2 + 1]
        for i in range(res["bands"].shape[1]):
            ax.plot(k_dist, res["bands"][:, i], color="#2563eb", linewidth=0.7)
        if res["gap_width"] > 0:
            bp = res["best_pair"]
            floor_v = np.max(res["bands"][:, bp])
            ceil_v = np.min(res["bands"][:, bp + 1])
            ax.axhspan(floor_v, ceil_v, alpha=0.15, color="#2563eb")
        ax.axhline(res["target"], color="red", linewidth=0.5, linestyle=":")
        ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_xlim(k_dist[0], k_dist[-1])
        ax.set_title(f"gap={res['gap_width']:.4f}", fontsize=9)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c * 2].axis("off")
        axes[r, c * 2 + 1].axis("off")

    fig.tight_layout()
    fig.savefig("pinn_fourier_eval.png", dpi=150)
    print(f"Saved pinn_fourier_eval.png")
    plt.close(fig)


def _plot_interpolation(model, cfg, m_indices, g_vectors, k_points,
                        k_dist, tick_pos, tick_labels):
    """Row of geometries at smoothly interpolated target frequencies."""
    n_interp = 8
    freqs = np.linspace(cfg.freq_lo, cfg.freq_hi, n_interp)

    fig, axes = plt.subplots(1, n_interp, figsize=(2.5 * n_interp, 2.5))

    for i, f in enumerate(freqs):
        tf_t = torch.tensor(f, dtype=torch.float64)
        with torch.no_grad():
            _, eps_grid = model(tf_t)
        eps_np = eps_grid.detach().cpu().numpy()

        ax = axes[i]
        ax.imshow(eps_np.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdYlBu_r")
        ax.set_title(f"f={f:.3f}", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)

    fig.suptitle("Interpolation: geometry vs target frequency", fontsize=11)
    fig.tight_layout()
    fig.savefig("pinn_fourier_interp.png", dpi=150)
    print(f"Saved pinn_fourier_interp.png")
    plt.close(fig)


# ===================================================================
# 8. CLI
# ===================================================================

def load_model(checkpoint="pinn_fourier_model.pt"):
    """Load a trained model from checkpoint."""
    ckpt = torch.load(checkpoint, map_location="cpu")
    c = ckpt["cfg"]
    model = FourierInverseNet(
        N=c["n_grid"], latent_dim=c["latent_dim"],
        eps_bg=c["eps_bg"], eps_rod=c["eps_rod"],
    ).double()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def eval_from_checkpoint(cfg):
    """Load checkpoint and run evaluation/plots."""
    model, ckpt = load_model(cfg.checkpoint)
    saved_cfg = argparse.Namespace(**ckpt["cfg"])
    history = ckpt.get("history", [])

    g_vectors, m_indices = reciprocal_lattice(saved_cfg.n_max)
    k_points, k_dist, tick_pos, tick_labels = make_k_path(saved_cfg.n_k_seg)

    # Override freq range if specified
    if cfg.freq_lo is not None:
        saved_cfg.freq_lo = cfg.freq_lo
    if cfg.freq_hi is not None:
        saved_cfg.freq_hi = cfg.freq_hi

    evaluate_model(model, saved_cfg, g_vectors, m_indices, k_points, k_dist,
                   tick_pos, tick_labels, history)


def main():
    p = argparse.ArgumentParser(
        description="Fourier-space physics-embedded inverse design network"
    )
    sub = p.add_subparsers(dest="command")

    # Train subcommand
    tr = sub.add_parser("train", help="Train the model")
    tr.add_argument("--steps", type=int, default=500)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--n-grid", type=int, default=16)
    tr.add_argument("--n-max", type=int, default=5)
    tr.add_argument("--n-bands", type=int, default=6)
    tr.add_argument("--n-k-seg", type=int, default=8)
    tr.add_argument("--latent-dim", type=int, default=32)
    tr.add_argument("--eps-bg", type=float, default=1.0)
    tr.add_argument("--eps-rod", type=float, default=8.9)
    tr.add_argument("--freq-lo", type=float, default=0.25)
    tr.add_argument("--freq-hi", type=float, default=0.45)
    tr.add_argument("--w-gap", type=float, default=1.0)
    tr.add_argument("--w-freq", type=float, default=5.0)
    tr.add_argument("--w-binary", type=float, default=0.1)
    tr.add_argument("--w-positive", type=float, default=1.0)
    tr.add_argument("--temperature", type=float, default=0.05)
    tr.add_argument("--alpha", type=float, default=1.0)
    tr.add_argument("--seed", type=int, default=42)

    # Eval subcommand
    ev = sub.add_parser("eval", help="Evaluate a saved checkpoint")
    ev.add_argument("--checkpoint", type=str, default="pinn_fourier_model.pt")
    ev.add_argument("--freq-lo", type=float, default=None)
    ev.add_argument("--freq-hi", type=float, default=None)

    cfg = p.parse_args()

    if cfg.command == "eval":
        eval_from_checkpoint(cfg)
    else:
        # Default to train (also handles `train` subcommand)
        if cfg.command is None:
            # No subcommand given -- parse as flat args for backwards compat
            p2 = argparse.ArgumentParser()
            for a in tr._actions:
                if a.option_strings:
                    p2._add_action(a)
            cfg = p2.parse_args()
        train_fourier_inverse(cfg)


if __name__ == "__main__":
    main()
