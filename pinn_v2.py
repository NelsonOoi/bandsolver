"""
Real-space inverse design with multi-target training.

Combines pinn.py's constrained real-space parameterization (sigmoid -> [eps_bg, eps_rod])
with pinn_fourier.py's multi-target training loop (uniform freq sampling, soft gap
selection, periodic eval checkpoints).

Key properties:
    - Epsilon is guaranteed in [eps_bg, eps_rod] by construction (sigmoid + affine)
    - C4v symmetry via octant tiling
    - Automatic band-pair selection via soft_gap_selection
    - Fourier feature embedding on target_freq input

Usage:
    python pinn_v2.py --steps 500 --lr 1e-3
    python pinn_v2.py eval --checkpoint pinn_v2_model.pt
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
                        solve_bands, smooth_min, smooth_max)


# ===================================================================
# 1. C4v Symmetry Tiling (from pinn.py)
# ===================================================================

def c4v_tile(octant: torch.Tensor, N: int) -> torch.Tensor:
    """Expand a triangular octant to a full NxN grid via C4v symmetry."""
    half = N // 2
    upper = torch.triu(octant)
    quadrant = upper + upper.T - torch.diag(torch.diag(upper))

    top = torch.cat([quadrant.flip(1), quadrant], dim=1)
    full = torch.cat([quadrant.flip(0).flip(1),
                      quadrant.flip(0)], dim=1)
    grid = torch.cat([top, full], dim=0)
    return grid


class C4vTiler(nn.Module):
    def __init__(self, N: int):
        super().__init__()
        self.N = N
        self.half = N // 2

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        squeezed = False
        if raw.dim() == 2:
            raw = raw.unsqueeze(0).unsqueeze(0)
            squeezed = True
        B = raw.shape[0]
        out = []
        for b in range(B):
            out.append(c4v_tile(raw[b, 0], self.N))
        result = torch.stack(out)
        if squeezed:
            return result[0]
        return result


# ===================================================================
# 2. Fourier Feature Embedding
# ===================================================================

class FourierFeatureEmbedding(nn.Module):
    """Positional encoding: scalar f -> [f, sin(2πf), cos(2πf), ..., sin(2πLf), cos(2πLf)]."""

    def __init__(self, n_freqs=3):
        super().__init__()
        self.n_freqs = n_freqs
        self.out_dim = 1 + 2 * n_freqs
        freqs = 2.0 * np.pi * torch.arange(1, n_freqs + 1, dtype=torch.float64)
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(1).to(torch.float64)
        phases = x * self.freqs
        return torch.cat([x, torch.sin(phases), torch.cos(phases)])


# ===================================================================
# 3. InverseDesignNetV2
# ===================================================================

class InverseDesignNetV2(nn.Module):
    """Maps target_freq -> real-space epsilon grid with hard [eps_bg, eps_rod] constraint.

    Architecture:
        Fourier embedding: f -> (7D for n_freqs=3)
        Encoder:  embed -> 64 -> 128 -> latent
        Decoder:  latent -> 128 -> 256 -> half*half octant
        C4v tile: octant -> full NxN grid
        Sigmoid + affine: guarantees eps in [eps_bg, eps_rod]
    """

    def __init__(self, N=16, latent_dim=32, eps_bg=1.0, eps_rod=8.9, n_embed_freqs=3):
        super().__init__()
        self.N = N
        self.half = N // 2
        self.eps_bg = eps_bg
        self.eps_rod = eps_rod

        self.embedding = FourierFeatureEmbedding(n_embed_freqs)

        self.encoder = nn.Sequential(
            nn.Linear(self.embedding.out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        octant_pixels = self.half * self.half
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, octant_pixels),
        )

        self.tiler = C4vTiler(N)

    def forward(self, target_freq: torch.Tensor):
        """Generate epsilon grid from target frequency.

        Returns:
            eps_grid: (N, N) with values in [eps_bg, eps_rod].
            eps_norm: (N, N) with values in [0, 1].
        """
        x_embed = self.embedding(target_freq)
        z = self.encoder(x_embed)
        raw = self.decoder(z).view(self.half, self.half)

        eps_norm = torch.sigmoid(raw)
        eps_norm_full = self.tiler(eps_norm)

        eps_grid = self.eps_bg + (self.eps_rod - self.eps_bg) * eps_norm_full
        return eps_grid, eps_norm_full


# ===================================================================
# 4. Splitting-Aware Soft Gap Selection (from pinn_fourier.py)
# ===================================================================

def soft_gap_selection(bands: torch.Tensor, target_freq: float,
                       temperature: float = 0.05, alpha: float = 1.0,
                       beta: float = 50.0):
    """Select the most promising band gap across all consecutive pairs.

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
        split_k = bands[:, n + 1] - bands[:, n]
        avg_s = split_k.mean()
        min_s = smooth_min(split_k, beta=-beta)
        mid = 0.5 * (bands[:, n + 1] + bands[:, n]).mean()

        avg_splits.append(avg_s)
        min_splits.append(min_s)
        mids.append(mid)

    avg_splits = torch.stack(avg_splits)
    min_splits = torch.stack(min_splits)
    mids = torch.stack(mids)

    freq_dist = torch.abs(mids - target_freq)
    scores = -freq_dist / temperature + alpha * avg_splits
    scores = torch.clamp(scores, min=-80.0, max=80.0)
    weights = torch.softmax(scores, dim=0)

    effective_split = (weights * min_splits).sum()
    effective_mid = (weights * mids).sum()

    return effective_split, effective_mid, weights


# ===================================================================
# 5. Eval Checkpoint
# ===================================================================

def _run_eval_checkpoint(model, cfg, g_vectors, m_indices, k_points, n_test=5):
    """Quick eval on fixed test frequencies."""
    test_freqs = np.linspace(cfg.freq_lo, cfg.freq_hi, n_test)
    gaps, mid_errors = [], []

    for tf in test_freqs:
        tf_t = torch.tensor(tf, dtype=torch.float64)
        with torch.no_grad():
            eps_grid, _ = model(tf_t)
            bands = solve_bands(k_points, g_vectors, eps_grid, m_indices,
                                cfg.n_bands, "tm")
        bands_np = bands.detach().cpu().numpy()
        best_gw, best_mid = 0.0, 0.0
        for n in range(bands_np.shape[1] - 1):
            floor_n = np.max(bands_np[:, n])
            ceil_n = np.min(bands_np[:, n + 1])
            gw = max(0.0, ceil_n - floor_n)
            if gw > best_gw:
                best_gw = gw
                best_mid = 0.5 * (ceil_n + floor_n)
        gaps.append(best_gw)
        mid_errors.append(abs(best_mid - tf))

    return {
        "mean_gap": float(np.mean(gaps)),
        "mean_mid_err": float(np.mean(mid_errors)),
        "gaps": gaps,
        "mid_errors": mid_errors,
        "test_freqs": test_freqs.tolist(),
    }


# ===================================================================
# 6. Training Loop
# ===================================================================

def train(cfg):
    """Multi-target training with soft gap selection and periodic eval."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    g_vectors, m_indices = reciprocal_lattice(cfg.n_max)
    k_points, k_dist, tick_pos, tick_labels = make_k_path(cfg.n_k_seg)

    model = InverseDesignNetV2(
        N=cfg.n_grid, latent_dim=cfg.latent_dim,
        eps_bg=cfg.eps_bg, eps_rod=cfg.eps_rod,
        n_embed_freqs=cfg.n_embed_freqs,
    ).double()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.steps)

    binary_schedule = np.linspace(cfg.w_binary * 0.1, cfg.w_binary, cfg.steps)

    history = []
    eval_history = []
    eval_interval = max(1, cfg.steps // 10)

    print(f"Inverse Design v2 | Grid: {cfg.n_grid} | Steps: {cfg.steps}")
    print(f"Target freq range: [{cfg.freq_lo:.3f}, {cfg.freq_hi:.3f}]")
    print(f"Epsilon range: [{cfg.eps_bg:.1f}, {cfg.eps_rod:.1f}] (hard constraint)")
    print(f"Fourier embedding: {cfg.n_embed_freqs} freqs ({model.embedding.out_dim}D)")
    print(f"Eval checkpoint every {eval_interval} steps")
    print("-" * 60)

    for step in range(cfg.steps):
        t0 = time.time()
        optimizer.zero_grad()

        target_freq = cfg.freq_lo + (cfg.freq_hi - cfg.freq_lo) * np.random.rand()
        target_freq_t = torch.tensor(target_freq, dtype=torch.float64)

        w_bin = float(binary_schedule[step])

        eps_grid, eps_norm = model(target_freq_t)
        bands = solve_bands(k_points, g_vectors, eps_grid, m_indices,
                            cfg.n_bands, "tm")

        eff_split, eff_mid, gap_weights = soft_gap_selection(
            bands, target_freq, temperature=cfg.temperature, alpha=cfg.alpha
        )

        safe_mid = torch.clamp(eff_mid, min=1e-6)
        loss_gap = -cfg.w_gap * (eff_split / safe_mid)
        loss_freq = cfg.w_freq * (eff_mid - target_freq) ** 2
        loss_binary = w_bin * torch.mean(eps_norm * (1.0 - eps_norm))

        total_loss = loss_gap + loss_freq + loss_binary

        total_loss.backward()

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
            "eps_min": eps_grid.min().item(),
            "eps_max": eps_grid.max().item(),
            "time": elapsed,
            "nan": False,
        }
        history.append(info)

        if (step + 1) % max(1, cfg.steps // 20) == 0 or step == 0:
            best_pair = int(gap_weights.argmax().item())
            print(f"[{step+1:4d}/{cfg.steps}] loss={info['total']:.5f}  "
                  f"gap={info['gap']:.5f}  mid={info['mid']:.4f}  "
                  f"tgt={target_freq:.4f}  pair={best_pair}-{best_pair+1}  "
                  f"eps=[{info['eps_min']:.1f},{info['eps_max']:.1f}]  "
                  f"({elapsed:.2f}s)")

        if (step + 1) % eval_interval == 0:
            ev = _run_eval_checkpoint(model, cfg, g_vectors, m_indices, k_points)
            ev["step"] = step
            eval_history.append(ev)
            print(f"  >> EVAL  mean_gap={ev['mean_gap']:.5f}  "
                  f"mean_mid_err={ev['mean_mid_err']:.5f}")

    ckpt_path = "pinn_v2_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": vars(cfg),
        "history": history,
        "eval_history": eval_history,
        "step": cfg.steps,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    evaluate_model(model, cfg, g_vectors, m_indices, k_points, k_dist,
                   tick_pos, tick_labels, history, eval_history)


# ===================================================================
# 7. Evaluation
# ===================================================================

def evaluate_model(model, cfg, g_vectors, m_indices, k_points, k_dist,
                   tick_pos, tick_labels, history, eval_history=None):
    if eval_history is None:
        eval_history = []

    test_freqs = np.linspace(cfg.freq_lo, cfg.freq_hi, 9)
    results = []

    print(f"{'target':>8s}  {'gap_width':>9s}  {'midgap':>8s}  {'best_pair':>9s}  "
          f"{'eps_min':>7s}  {'eps_max':>7s}")
    print("-" * 58)

    for tf in test_freqs:
        tf_t = torch.tensor(tf, dtype=torch.float64)
        with torch.no_grad():
            eps_grid, eps_norm = model(tf_t)
            bands = solve_bands(k_points, g_vectors, eps_grid, m_indices,
                                cfg.n_bands, "tm")

        bands_np = bands.detach().cpu().numpy()
        eps_np = eps_grid.detach().cpu().numpy()
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
            "eps_grid": eps_np,
        })
        print(f"{tf:8.4f}  {best_gw:9.5f}  {best_mid:8.4f}  {best_pair:5d}-{best_pair+1}"
              f"  {eps_np.min():7.2f}  {eps_np.max():7.2f}")

    _plot_training(results, history, eval_history, k_dist, tick_pos, tick_labels, cfg)
    _plot_evaluation_grid(results, k_dist, tick_pos, tick_labels, cfg)
    _plot_interpolation(model, cfg, m_indices, g_vectors, k_points,
                        k_dist, tick_pos, tick_labels)


# ===================================================================
# 8. Plotting
# ===================================================================

def _plot_training(results, history, eval_history, k_dist, tick_pos, tick_labels, cfg):
    mid_res = results[len(results) // 2]

    n_panels = 4 if eval_history else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))

    ax = axes[0]
    im = ax.imshow(mid_res["eps_grid"].T, origin="lower",
                   extent=[0, 1, 0, 1], cmap="RdYlBu_r")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"Unit cell (f_tgt={mid_res['target']:.3f})")
    ax.set_xlabel("x/a"); ax.set_ylabel("y/a"); ax.set_aspect("equal")

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

    ax = axes[2]
    valid = [h for h in history if not h["nan"]]
    if valid:
        steps = [h["step"] for h in valid]
        ax.plot(steps, [h["total"] for h in valid], label="total",
                linewidth=0.4, alpha=0.5)
        window = max(1, len(valid) // 20)
        if len(valid) > window:
            totals = np.array([h["total"] for h in valid])
            rolling = np.convolve(totals, np.ones(window)/window, mode="valid")
            ax.plot(steps[window-1:], rolling, label=f"avg({window})",
                    linewidth=1.5, color="#dc2626")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Per-step loss")
    ax.legend(fontsize=8); ax.set_yscale("symlog", linthresh=1e-4)

    if eval_history:
        ax = axes[3]
        ev_steps = [e["step"] for e in eval_history]
        ax.plot(ev_steps, [e["mean_gap"] for e in eval_history],
                "o-", label="mean gap", color="#2563eb", linewidth=1.5)
        ax2 = ax.twinx()
        ax2.plot(ev_steps, [e["mean_mid_err"] for e in eval_history],
                 "s--", label="mean |mid-tgt|", color="#dc2626", linewidth=1.5)
        ax.set_xlabel("step"); ax.set_ylabel("mean gap width", color="#2563eb")
        ax2.set_ylabel("mean midgap error", color="#dc2626")
        ax.set_title("Eval checkpoints")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig("pinn_v2_training.png", dpi=150)
    print("Saved pinn_v2_training.png")
    plt.close(fig)


def _plot_evaluation_grid(results, k_dist, tick_pos, tick_labels, cfg):
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(4 * cols * 2, 3.5 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for idx, res in enumerate(results):
        r, c = divmod(idx, cols)

        ax = axes[r, c * 2]
        im = ax.imshow(res["eps_grid"].T, origin="lower",
                       extent=[0, 1, 0, 1], cmap="RdYlBu_r",
                       vmin=cfg.eps_bg, vmax=cfg.eps_rod)
        ax.set_title(f"f={res['target']:.3f}", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

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

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c * 2].axis("off")
        axes[r, c * 2 + 1].axis("off")

    fig.tight_layout()
    fig.savefig("pinn_v2_eval.png", dpi=150)
    print("Saved pinn_v2_eval.png")
    plt.close(fig)


def _plot_interpolation(model, cfg, m_indices, g_vectors, k_points,
                        k_dist, tick_pos, tick_labels):
    n_interp = 8
    freqs = np.linspace(cfg.freq_lo, cfg.freq_hi, n_interp)

    fig, axes = plt.subplots(1, n_interp, figsize=(2.5 * n_interp, 2.5))

    for i, f in enumerate(freqs):
        tf_t = torch.tensor(f, dtype=torch.float64)
        with torch.no_grad():
            eps_grid, _ = model(tf_t)
        eps_np = eps_grid.detach().cpu().numpy()

        ax = axes[i]
        ax.imshow(eps_np.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdYlBu_r",
                  vmin=cfg.eps_bg, vmax=cfg.eps_rod)
        ax.set_title(f"f={f:.3f}", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)

    fig.suptitle("Interpolation: geometry vs target frequency", fontsize=11)
    fig.tight_layout()
    fig.savefig("pinn_v2_interp.png", dpi=150)
    print("Saved pinn_v2_interp.png")
    plt.close(fig)


# ===================================================================
# 9. CLI
# ===================================================================

def load_model(checkpoint="pinn_v2_model.pt"):
    ckpt = torch.load(checkpoint, map_location="cpu")
    c = ckpt["cfg"]
    model = InverseDesignNetV2(
        N=c["n_grid"], latent_dim=c["latent_dim"],
        eps_bg=c["eps_bg"], eps_rod=c["eps_rod"],
        n_embed_freqs=c.get("n_embed_freqs", 3),
    ).double()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def eval_from_checkpoint(cfg):
    model, ckpt = load_model(cfg.checkpoint)
    saved_cfg = argparse.Namespace(**ckpt["cfg"])
    history = ckpt.get("history", [])
    eval_history = ckpt.get("eval_history", [])

    g_vectors, m_indices = reciprocal_lattice(saved_cfg.n_max)
    k_points, k_dist, tick_pos, tick_labels = make_k_path(saved_cfg.n_k_seg)

    if cfg.freq_lo is not None:
        saved_cfg.freq_lo = cfg.freq_lo
    if cfg.freq_hi is not None:
        saved_cfg.freq_hi = cfg.freq_hi

    evaluate_model(model, saved_cfg, g_vectors, m_indices, k_points, k_dist,
                   tick_pos, tick_labels, history, eval_history)


def _add_train_args(parser):
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-grid", type=int, default=32)
    parser.add_argument("--n-max", type=int, default=5)
    parser.add_argument("--n-bands", type=int, default=6)
    parser.add_argument("--n-k-seg", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--eps-bg", type=float, default=1.0)
    parser.add_argument("--eps-rod", type=float, default=8.9)
    parser.add_argument("--freq-lo", type=float, default=0.25)
    parser.add_argument("--freq-hi", type=float, default=0.45)
    parser.add_argument("--w-gap", type=float, default=1.0)
    parser.add_argument("--w-freq", type=float, default=20.0)
    parser.add_argument("--w-binary", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-embed-freqs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ("train", "eval"):
        p = argparse.ArgumentParser(description="Real-space inverse design v2")
        sub = p.add_subparsers(dest="command")
        tr = sub.add_parser("train")
        _add_train_args(tr)
        ev = sub.add_parser("eval")
        ev.add_argument("--checkpoint", type=str, default="pinn_v2_model.pt")
        ev.add_argument("--freq-lo", type=float, default=None)
        ev.add_argument("--freq-hi", type=float, default=None)
        cfg = p.parse_args()
        if cfg.command == "eval":
            eval_from_checkpoint(cfg)
        else:
            train(cfg)
    else:
        p = argparse.ArgumentParser(description="Real-space inverse design v2")
        _add_train_args(p)
        cfg = p.parse_args()
        train(cfg)


if __name__ == "__main__":
    main()
