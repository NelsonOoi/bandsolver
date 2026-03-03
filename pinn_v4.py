"""
Perturbation-aware inverse design with physics-embedded activations (v4).

The network receives a random C4v-symmetric base structure each step and
outputs a perturbation delta_eps to open a bandgap.  Encoder layers use
exact constitutive equations from PWE theory as activations; decoder layers
use learnable physics-motivated activations.  Supports TM and TE.

Usage:
    python pinn_v4.py --steps 500 --lr 1e-3
    python pinn_v4.py --steps 500 --polarization te
    python pinn_v4.py eval --checkpoint pinn_v4_model.pt
"""

import argparse
import math
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
                        solve_bands, smooth_min, smooth_max,
                        _solve_bands_from_inv, _safe_sqrt)


# ===================================================================
# 1. Physics-Embedded Activation Functions
# ===================================================================

# --- Encoder activations: exact constitutive equations ---

class InverseRuleActivation(nn.Module):
    """eta = 1 / eps(x), the constitutive relation entering the Hamiltonian.

    Maps features through sigmoid to physical epsilon range, then applies
    the inverse rule.  This is the exact relation used in both TM and TE
    eigenvalue problems.
    """

    def __init__(self, eps_lo=1.0, eps_hi=8.9):
        super().__init__()
        self.eps_lo = eps_lo
        self.eps_hi = eps_hi

    def forward(self, x):
        eps = self.eps_lo + (self.eps_hi - self.eps_lo) * torch.sigmoid(x)
        return 1.0 / eps


class TMDispersionActivation(nn.Module):
    """omega = |k+G| / sqrt(eps), the TM dispersion relation.

    The linear layer before this learns an effective |k+G| scale per feature.
    A learnable k_scale is included so the network can adapt the wavevector
    magnitude.
    """

    def __init__(self, eps_lo=1.0, eps_hi=8.9):
        super().__init__()
        self.eps_lo = eps_lo
        self.eps_hi = eps_hi
        self.k_scale = nn.Parameter(torch.ones(1, dtype=torch.float64))

    def forward(self, x):
        eps = self.eps_lo + (self.eps_hi - self.eps_lo) * torch.sigmoid(x)
        return self.k_scale.abs() / torch.sqrt(eps.clamp(min=1e-12))


class TECouplingActivation(nn.Module):
    """H_ij = (k+G_i).(k+G_j) * eta_ij, the TE Hamiltonian coupling.

    Splits features into geometric and material halves.  Output concatenates
    the geometric part with the coupled product (geom * eta), preserving
    the input dimension.
    """

    def __init__(self, eps_lo=1.0, eps_hi=8.9):
        super().__init__()
        self.eps_lo = eps_lo
        self.eps_hi = eps_hi

    def forward(self, x):
        n = x.shape[-1]
        half = n // 2
        geom = x[..., :half]
        mat = x[..., half:2 * half]
        eps = self.eps_lo + (self.eps_hi - self.eps_lo) * torch.sigmoid(mat)
        coupled = geom / eps
        parts = [geom, coupled]
        if n % 2 == 1:
            parts.append(x[..., -1:])
        return torch.cat(parts, dim=-1)


class EigFreqActivation(nn.Module):
    """omega = sqrt(lambda) / (2*pi), the eigenvalue-to-frequency map.

    Clamps input to non-negative before sqrt for numerical stability.
    """

    def forward(self, x):
        return torch.sqrt(x.clamp(min=1e-12)) / (2.0 * math.pi)


# --- Decoder activations: learnable physics-motivated ---

class BandSplitActivation(nn.Module):
    """x * tanh(alpha * x): antisymmetric, saturating.

    Mirrors perturbation-induced band splitting which is odd in the
    perturbation sign and saturates at band crossings.
    """

    def __init__(self, dim=1):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim, dtype=torch.float64))

    def forward(self, x):
        return x * torch.tanh(self.alpha * x)


class GapActivation(nn.Module):
    """softplus(x - t) - softplus(-t): dead zone then linear growth.

    Models the threshold behaviour of bandgap opening under perturbation.
    """

    def __init__(self, dim=1):
        super().__init__()
        self.threshold = nn.Parameter(0.5 * torch.ones(dim, dtype=torch.float64))

    def forward(self, x):
        t = self.threshold.abs()
        return F.softplus(x - t) - F.softplus(-t)


# ===================================================================
# 2. C4v Symmetry Tiling
# ===================================================================

def c4v_tile(octant: torch.Tensor, N: int) -> torch.Tensor:
    half = N // 2
    upper = torch.triu(octant)
    quadrant = upper + upper.T - torch.diag(torch.diag(upper))
    top = torch.cat([quadrant.flip(1), quadrant], dim=1)
    full = torch.cat([quadrant.flip(0).flip(1), quadrant.flip(0)], dim=1)
    return torch.cat([top, full], dim=0)


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
        out = [c4v_tile(raw[b, 0], self.N) for b in range(raw.shape[0])]
        result = torch.stack(out)
        return result[0] if squeezed else result


# ===================================================================
# 3. Random Base Structure Generator
# ===================================================================

def generate_random_base(N, eps_bg, eps_rod):
    """Generate a random C4v-symmetric binary epsilon grid."""
    half = N // 2
    octant = torch.bernoulli(0.5 * torch.ones(half, half, dtype=torch.float64))
    full = c4v_tile(octant, N)
    return eps_bg + (eps_rod - eps_bg) * full


# ===================================================================
# 4. Eigenvector Feature Extraction
# ===================================================================

def extract_field_features(eps_base, g_vectors, m_indices, k_points,
                           n_bands, polarization, target_freq,
                           n_field_bands=4, coarsen=4):
    """Extract spatial field intensity features from the base structure.

    Returns a flat feature vector containing:
      - coarsened |E_n|^2 for the n_field_bands bands nearest target_freq
      - top FFT magnitude coefficients of eps_base
    """
    N = eps_base.shape[0]
    with torch.no_grad():
        eps_mat = build_epsilon_matrix(eps_base, m_indices)
        eps_mat_inv = torch.linalg.inv(eps_mat)
        freqs, eigvecs, kpg_info = _solve_bands_from_inv(
            k_points, g_vectors, eps_mat_inv, n_bands, polarization,
            return_vecs=True)

    freqs_mean = freqs.mean(dim=0)
    dists = (freqs_mean - target_freq).abs()
    _, nearest_idx = torch.topk(dists, min(n_field_bands, n_bands), largest=False)
    nearest_idx = nearest_idx.sort().values

    n_pw = eigvecs.shape[1]
    n_side = int(math.isqrt(n_pw))
    c = max(1, N // coarsen)

    field_feats = []
    for bi in nearest_idx:
        evec = eigvecs[:, :, bi]
        intensity = (evec.abs() ** 2).mean(dim=0)

        if n_side * n_side == n_pw:
            field_2d = intensity.view(n_side, n_side)
            field_coarse = F.adaptive_avg_pool2d(
                field_2d.unsqueeze(0).unsqueeze(0).float(), c
            ).squeeze().to(torch.float64)
        else:
            field_coarse = intensity[:c * c].view(c, c)
        field_feats.append(field_coarse.flatten())

    if len(field_feats) < n_field_bands:
        pad_size = field_feats[0].shape[0]
        while len(field_feats) < n_field_bands:
            field_feats.append(torch.zeros(pad_size, dtype=torch.float64))

    eps_fft = torch.fft.fft2(eps_base) / (N * N)
    fft_mag = eps_fft.abs().flatten()
    top_k = min(16, fft_mag.shape[0])
    fft_top, _ = torch.topk(fft_mag, top_k)

    return torch.cat(field_feats + [fft_top]).detach()


# ===================================================================
# 5. Fourier Feature Embedding
# ===================================================================

class FourierFeatureEmbedding(nn.Module):
    def __init__(self, n_freqs=3):
        super().__init__()
        self.n_freqs = n_freqs
        self.out_dim = 1 + 2 * n_freqs
        freqs = 2.0 * np.pi * torch.arange(1, n_freqs + 1, dtype=torch.float64)
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        x = x.view(1).to(torch.float64)
        phases = x * self.freqs
        return torch.cat([x, torch.sin(phases), torch.cos(phases)])


# ===================================================================
# 6. PerturbationNet
# ===================================================================

class PerturbationNet(nn.Module):
    """Maps (target_freq, eps_base, field_features) -> eps_full via perturbation.

    Encoder uses exact constitutive equations as activations (polarization-
    dependent at layer 2).  Decoder uses learnable physics-motivated
    activations.
    """

    def __init__(self, N=16, latent_dim=32, eps_bg=1.0, eps_rod=8.9,
                 n_embed_freqs=3, n_field_features=80, polarization="tm"):
        super().__init__()
        self.N = N
        self.half = N // 2
        self.eps_bg = eps_bg
        self.eps_rod = eps_rod
        self.polarization = polarization

        self.embedding = FourierFeatureEmbedding(n_embed_freqs)
        in_dim = self.embedding.out_dim + n_field_features

        # Encoder with constitutive-equation activations
        self.enc_lin1 = nn.Linear(in_dim, 128)
        self.enc_act1 = InverseRuleActivation(eps_bg, eps_rod)

        self.enc_lin2 = nn.Linear(128, 256)
        if polarization == "tm":
            self.enc_act2 = TMDispersionActivation(eps_bg, eps_rod)
        else:
            self.enc_act2 = TECouplingActivation(eps_bg, eps_rod)

        self.enc_lin3 = nn.Linear(256, latent_dim)
        self.enc_act3 = EigFreqActivation()

        # Decoder with learnable physics-motivated activations
        octant_pixels = self.half * self.half
        self.dec_lin1 = nn.Linear(latent_dim, 256)
        self.dec_act1 = BandSplitActivation(256)

        self.dec_lin2 = nn.Linear(256, 128)
        self.dec_act2 = BandSplitActivation(128)

        self.dec_lin3 = nn.Linear(128, octant_pixels)
        self.dec_act3 = GapActivation(octant_pixels)

        self.tiler = C4vTiler(N)

    def forward(self, target_freq, eps_base, field_features):
        x_embed = self.embedding(target_freq)
        x_in = torch.cat([x_embed, field_features])

        # Encoder: constitutive activations
        h = self.enc_act1(self.enc_lin1(x_in))
        h = self.enc_act2(self.enc_lin2(h))
        z = self.enc_act3(self.enc_lin3(h))

        # Decoder: learnable activations
        h = self.dec_act1(self.dec_lin1(z))
        h = self.dec_act2(self.dec_lin2(h))
        delta_octant = self.dec_act3(self.dec_lin3(h))

        delta_octant = delta_octant.view(self.half, self.half)
        delta_full = self.tiler(delta_octant)

        # Scale delta to perturbation range and clamp
        delta_scaled = (self.eps_rod - self.eps_bg) * (2.0 * torch.sigmoid(delta_full) - 1.0)
        eps_full = torch.clamp(eps_base + delta_scaled, self.eps_bg, self.eps_rod)

        eps_norm = (eps_full - self.eps_bg) / (self.eps_rod - self.eps_bg)
        return eps_full, delta_scaled, eps_norm


# ===================================================================
# 7. Soft Gap Selection
# ===================================================================

def soft_gap_selection(bands, target_freq, temperature=0.05, alpha=1.0,
                       beta=50.0):
    n_bands = bands.shape[1]
    n_pairs = n_bands - 1

    min_splits, avg_splits, mids = [], [], []
    for n in range(n_pairs):
        split_k = bands[:, n + 1] - bands[:, n]
        avg_splits.append(split_k.mean())
        min_splits.append(smooth_min(split_k, beta=-beta))
        mids.append(0.5 * (bands[:, n + 1] + bands[:, n]).mean())

    avg_splits = torch.stack(avg_splits)
    min_splits = torch.stack(min_splits)
    mids = torch.stack(mids)

    freq_dist = torch.abs(mids - target_freq)
    scores = -freq_dist / temperature + alpha * avg_splits
    scores = torch.clamp(scores, min=-80.0, max=80.0)
    weights = torch.softmax(scores, dim=0)

    return (weights * min_splits).sum(), (weights * mids).sum(), weights


# ===================================================================
# 8. Eval Checkpoint
# ===================================================================

def _run_eval_checkpoint(model, cfg, g_vectors, m_indices, k_points,
                         n_test=5):
    bw = cfg.train_bw
    test_freqs = np.linspace(cfg.target_freq - bw, cfg.target_freq + bw,
                             n_test)
    gaps, mid_errors = [], []

    for tf in test_freqs:
        tf_t = torch.tensor(tf, dtype=torch.float64)
        with torch.no_grad():
            base = generate_random_base(cfg.n_grid, cfg.eps_bg, cfg.eps_rod)
            ff = extract_field_features(
                base, g_vectors, m_indices, k_points,
                cfg.n_bands, cfg.polarization, tf,
                n_field_bands=cfg.n_field_bands)
            eps_grid, _, _ = model(tf_t, base, ff)
            bands = solve_bands(k_points, g_vectors, eps_grid, m_indices,
                                cfg.n_bands, cfg.polarization)
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
# 9. Training Loop
# ===================================================================

def train(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    g_vectors, m_indices = reciprocal_lattice(cfg.n_max)
    k_points, k_dist, tick_pos, tick_labels = make_k_path(cfg.n_k_seg)

    # Compute field feature size from a dummy run
    dummy_base = generate_random_base(cfg.n_grid, cfg.eps_bg, cfg.eps_rod)
    dummy_ff = extract_field_features(
        dummy_base, g_vectors, m_indices, k_points,
        cfg.n_bands, cfg.polarization, 0.35,
        n_field_bands=cfg.n_field_bands)
    n_field_features = dummy_ff.shape[0]

    model = PerturbationNet(
        N=cfg.n_grid, latent_dim=cfg.latent_dim,
        eps_bg=cfg.eps_bg, eps_rod=cfg.eps_rod,
        n_embed_freqs=cfg.n_embed_freqs,
        n_field_features=n_field_features,
        polarization=cfg.polarization,
    ).double()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.steps)

    binary_schedule = np.linspace(cfg.w_binary * 0.1, cfg.w_binary, cfg.steps)
    # Perturbation curriculum: start with high reg (small deltas), decay
    if cfg.pert_curriculum:
        pert_schedule = np.linspace(cfg.w_pert * 5.0, cfg.w_pert, cfg.steps)
    else:
        pert_schedule = np.full(cfg.steps, cfg.w_pert)

    history = []
    eval_history = []
    eval_interval = max(1, cfg.steps // 10)

    freq_lo = cfg.target_freq - cfg.train_bw
    freq_hi = cfg.target_freq + cfg.train_bw

    pol_label = cfg.polarization.upper()
    print(f"Inverse Design v4 (perturbation) | Grid: {cfg.n_grid} | "
          f"Steps: {cfg.steps} | {pol_label}")
    print(f"Target freq: {cfg.target_freq:.3f}  "
          f"(training range: [{freq_lo:.3f}, {freq_hi:.3f}])")
    print(f"Gap mode: {cfg.gap_mode}")
    print(f"Epsilon range: [{cfg.eps_bg:.1f}, {cfg.eps_rod:.1f}]")
    print(f"Field features: {n_field_features}D "
          f"({cfg.n_field_bands} bands)")
    print(f"Encoder activations: InverseRule → "
          f"{'TMDispersion' if cfg.polarization == 'tm' else 'TECoupling'}"
          f" → EigFreq")
    print(f"Decoder activations: BandSplit → BandSplit → Gap")
    print(f"Perturbation curriculum: "
          f"{'on' if cfg.pert_curriculum else 'off'}")
    print(f"Eval checkpoint every {eval_interval} steps")
    print("-" * 60)

    for step in range(cfg.steps):
        t0 = time.time()
        optimizer.zero_grad()

        target_freq = freq_lo + (freq_hi - freq_lo) * np.random.rand()
        target_freq_t = torch.tensor(target_freq, dtype=torch.float64)

        w_bin = float(binary_schedule[step])
        w_pert = float(pert_schedule[step])

        eps_base = generate_random_base(cfg.n_grid, cfg.eps_bg, cfg.eps_rod)
        field_feats = extract_field_features(
            eps_base, g_vectors, m_indices, k_points,
            cfg.n_bands, cfg.polarization, target_freq,
            n_field_bands=cfg.n_field_bands)

        eps_grid, delta_eps, eps_norm = model(target_freq_t, eps_base, field_feats)
        bands = solve_bands(k_points, g_vectors, eps_grid, m_indices,
                            cfg.n_bands, cfg.polarization)

        eff_split, eff_mid, gap_weights = soft_gap_selection(
            bands, target_freq, temperature=cfg.temperature, alpha=cfg.alpha)

        if cfg.gap_mode == "width":
            loss_gap = -cfg.w_gap * eff_split
        else:
            safe_mid = torch.clamp(eff_mid, min=1e-6)
            loss_gap = -cfg.w_gap * (eff_split / safe_mid)

        loss_freq = cfg.w_freq * (eff_mid - target_freq) ** 2
        loss_binary = w_bin * torch.mean(eps_norm * (1.0 - eps_norm))
        loss_pert = w_pert * torch.mean(delta_eps ** 2)

        total_loss = loss_gap + loss_freq + loss_binary + loss_pert

        if cfg.w_field > 0:
            field_intensity = field_feats[:cfg.n_grid * cfg.n_grid].view(
                cfg.n_grid, cfg.n_grid) if field_feats.shape[0] >= cfg.n_grid ** 2 else None
            if field_intensity is not None:
                loss_field = cfg.w_field * torch.mean(
                    (delta_eps * field_intensity) ** 2)
                total_loss = total_loss + loss_field

        total_loss.backward()

        has_nan = any(
            p.grad is not None and torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        if has_nan or torch.isnan(total_loss):
            optimizer.zero_grad()
            scheduler.step()
            elapsed = time.time() - t0
            history.append({"step": step, "total": float("nan"), "gap": 0.0,
                            "mid": 0.0, "target": target_freq, "time": elapsed,
                            "nan": True, "pert_norm": 0.0})
            if (step + 1) % max(1, cfg.steps // 20) == 0:
                print(f"[{step+1:4d}/{cfg.steps}] NaN detected, skipping")
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t0
        pert_rms = float(torch.sqrt(torch.mean(delta_eps ** 2)).item())
        info = {
            "step": step,
            "total": total_loss.item(),
            "loss_gap": loss_gap.item(),
            "loss_freq": loss_freq.item(),
            "loss_binary": loss_binary.item(),
            "loss_pert": loss_pert.item(),
            "gap": eff_split.item(),
            "mid": eff_mid.item(),
            "target": target_freq,
            "eps_min": eps_grid.min().item(),
            "eps_max": eps_grid.max().item(),
            "pert_norm": pert_rms,
            "time": elapsed,
            "nan": False,
        }
        history.append(info)

        if (step + 1) % max(1, cfg.steps // 20) == 0 or step == 0:
            best_pair = int(gap_weights.argmax().item())
            print(f"[{step+1:4d}/{cfg.steps}] loss={info['total']:.5f}  "
                  f"gap={info['gap']:.5f}  mid={info['mid']:.4f}  "
                  f"tgt={target_freq:.4f}  pair={best_pair}-{best_pair+1}  "
                  f"|δε|={pert_rms:.3f}  "
                  f"eps=[{info['eps_min']:.1f},{info['eps_max']:.1f}]  "
                  f"({elapsed:.2f}s)")

        if (step + 1) % eval_interval == 0:
            ev = _run_eval_checkpoint(model, cfg, g_vectors, m_indices,
                                      k_points)
            ev["step"] = step
            eval_history.append(ev)
            print(f"  >> EVAL  mean_gap={ev['mean_gap']:.5f}  "
                  f"mean_mid_err={ev['mean_mid_err']:.5f}")

    ckpt_path = "pinn_v4_model.pt"
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
# 10. Evaluation
# ===================================================================

def evaluate_model(model, cfg, g_vectors, m_indices, k_points, k_dist,
                   tick_pos, tick_labels, history, eval_history=None):
    if eval_history is None:
        eval_history = []

    bw = cfg.train_bw
    test_freqs = np.linspace(cfg.target_freq - bw, cfg.target_freq + bw, 9)
    results = []

    pol_label = cfg.polarization.upper()
    print(f"{'target':>8s}  {'gap_width':>9s}  {'midgap':>8s}  {'best_pair':>9s}  "
          f"{'eps_min':>7s}  {'eps_max':>7s}  {'|δε|_rms':>8s}")
    print("-" * 68)

    for tf in test_freqs:
        tf_t = torch.tensor(tf, dtype=torch.float64)
        with torch.no_grad():
            base = generate_random_base(cfg.n_grid, cfg.eps_bg, cfg.eps_rod)
            ff = extract_field_features(
                base, g_vectors, m_indices, k_points,
                cfg.n_bands, cfg.polarization, tf,
                n_field_bands=cfg.n_field_bands)
            eps_grid, delta_eps, _ = model(tf_t, base, ff)
            bands = solve_bands(k_points, g_vectors, eps_grid, m_indices,
                                cfg.n_bands, cfg.polarization)

        bands_np = bands.detach().cpu().numpy()
        eps_np = eps_grid.detach().cpu().numpy()
        base_np = base.detach().cpu().numpy()
        delta_np = delta_eps.detach().cpu().numpy()
        pert_rms = float(np.sqrt(np.mean(delta_np ** 2)))

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
            "target": tf, "gap_width": best_gw, "midgap": best_mid,
            "best_pair": best_pair, "bands": bands_np, "eps_grid": eps_np,
            "base_grid": base_np, "delta_grid": delta_np, "pert_rms": pert_rms,
        })
        print(f"{tf:8.4f}  {best_gw:9.5f}  {best_mid:8.4f}  "
              f"{best_pair:5d}-{best_pair+1}  {eps_np.min():7.2f}  "
              f"{eps_np.max():7.2f}  {pert_rms:8.4f}")

    _plot_training(results, history, eval_history, k_dist, tick_pos,
                   tick_labels, cfg)
    _plot_evaluation_grid(results, k_dist, tick_pos, tick_labels, cfg)
    _plot_interpolation(model, cfg, m_indices, g_vectors, k_points,
                        k_dist, tick_pos, tick_labels)


# ===================================================================
# 11. Plotting
# ===================================================================

def _plot_training(results, history, eval_history, k_dist, tick_pos,
                   tick_labels, cfg):
    mid_res = results[len(results) // 2]
    pol_label = cfg.polarization.upper()

    n_panels = 5 if eval_history else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))

    # Panel 0: unit cell
    ax = axes[0]
    im = ax.imshow(mid_res["eps_grid"].T, origin="lower",
                   extent=[0, 1, 0, 1], cmap="RdYlBu_r")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"Unit cell (f_tgt={mid_res['target']:.3f})")
    ax.set_xlabel("x/a"); ax.set_ylabel("y/a"); ax.set_aspect("equal")

    # Panel 1: bands
    ax = axes[1]
    bands_np = mid_res["bands"]
    for i in range(bands_np.shape[1]):
        ax.plot(k_dist, bands_np[:, i], color="#2563eb", linewidth=0.9)
    for tp in tick_pos:
        ax.axvline(tp, color="grey", linewidth=0.4, linestyle="--")
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels)
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylabel("$\\omega a / 2\\pi c$")
    ax.set_title(f"{pol_label} bands")
    if mid_res["gap_width"] > 0:
        bp = mid_res["best_pair"]
        floor_v = np.max(bands_np[:, bp])
        ceil_v = np.min(bands_np[:, bp + 1])
        ax.axhspan(floor_v, ceil_v, alpha=0.15, color="#2563eb")
    ax.axhline(mid_res["target"], color="red", linewidth=0.7, linestyle=":")

    # Panel 2: loss
    ax = axes[2]
    valid = [h for h in history if not h["nan"]]
    if valid:
        steps = [h["step"] for h in valid]
        ax.plot(steps, [h["total"] for h in valid], label="total",
                linewidth=0.4, alpha=0.5)
        window = max(1, len(valid) // 20)
        if len(valid) > window:
            totals = np.array([h["total"] for h in valid])
            rolling = np.convolve(totals, np.ones(window) / window,
                                  mode="valid")
            ax.plot(steps[window - 1:], rolling, label=f"avg({window})",
                    linewidth=1.5, color="#dc2626")
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_title("Per-step loss")
    ax.legend(fontsize=8); ax.set_yscale("symlog", linthresh=1e-4)

    # Panel 3: perturbation magnitude over time
    ax = axes[3]
    if valid:
        ax.plot(steps, [h["pert_norm"] for h in valid],
                linewidth=0.6, alpha=0.6, color="#059669")
        window = max(1, len(valid) // 20)
        if len(valid) > window:
            perts = np.array([h["pert_norm"] for h in valid])
            rolling = np.convolve(perts, np.ones(window) / window,
                                  mode="valid")
            ax.plot(steps[window - 1:], rolling, linewidth=1.5,
                    color="#059669")
    ax.set_xlabel("step"); ax.set_ylabel("|δε| RMS")
    ax.set_title("Perturbation magnitude")

    # Panel 4: eval checkpoints
    if eval_history:
        ax = axes[4]
        ev_steps = [e["step"] for e in eval_history]
        ax.plot(ev_steps, [e["mean_gap"] for e in eval_history],
                "o-", label="mean gap", color="#2563eb", linewidth=1.5)
        ax2 = ax.twinx()
        ax2.plot(ev_steps, [e["mean_mid_err"] for e in eval_history],
                 "s--", label="mean |mid-tgt|", color="#dc2626",
                 linewidth=1.5)
        ax.set_xlabel("step")
        ax.set_ylabel("mean gap width", color="#2563eb")
        ax2.set_ylabel("mean midgap error", color="#dc2626")
        ax.set_title("Eval checkpoints")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig("pinn_v4_training.png", dpi=150)
    print("Saved pinn_v4_training.png")
    plt.close(fig)


def _plot_evaluation_grid(results, k_dist, tick_pos, tick_labels, cfg):
    """3x3 grid, each cell: base | delta | final + bands."""
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 4, figsize=(3.2 * cols * 4, 3.5 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for idx, res in enumerate(results):
        r, c = divmod(idx, cols)
        base_col = c * 4

        # Base
        ax = axes[r, base_col]
        ax.imshow(res["base_grid"].T, origin="lower", extent=[0, 1, 0, 1],
                  cmap="RdYlBu_r", vmin=cfg.eps_bg, vmax=cfg.eps_rod)
        ax.set_title(f"base f={res['target']:.3f}", fontsize=8)
        ax.set_aspect("equal"); ax.tick_params(labelsize=6)

        # Delta
        ax = axes[r, base_col + 1]
        vmax_d = max(abs(res["delta_grid"].min()), abs(res["delta_grid"].max()), 0.1)
        ax.imshow(res["delta_grid"].T, origin="lower", extent=[0, 1, 0, 1],
                  cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d)
        ax.set_title(f"δε |{res['pert_rms']:.3f}|", fontsize=8)
        ax.set_aspect("equal"); ax.tick_params(labelsize=6)

        # Final
        ax = axes[r, base_col + 2]
        ax.imshow(res["eps_grid"].T, origin="lower", extent=[0, 1, 0, 1],
                  cmap="RdYlBu_r", vmin=cfg.eps_bg, vmax=cfg.eps_rod)
        ax.set_title("final", fontsize=8)
        ax.set_aspect("equal"); ax.tick_params(labelsize=6)

        # Bands
        ax = axes[r, base_col + 3]
        for i in range(res["bands"].shape[1]):
            ax.plot(k_dist, res["bands"][:, i], color="#2563eb", linewidth=0.7)
        if res["gap_width"] > 0:
            bp = res["best_pair"]
            floor_v = np.max(res["bands"][:, bp])
            ceil_v = np.min(res["bands"][:, bp + 1])
            ax.axhspan(floor_v, ceil_v, alpha=0.15, color="#2563eb")
        ax.axhline(res["target"], color="red", linewidth=0.5, linestyle=":")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=6)
        ax.set_xlim(k_dist[0], k_dist[-1])
        ax.set_title(f"gap={res['gap_width']:.4f}", fontsize=8)
        ax.tick_params(labelsize=6)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        for off in range(4):
            axes[r, c * 4 + off].axis("off")

    fig.tight_layout()
    fig.savefig("pinn_v4_eval.png", dpi=150)
    print("Saved pinn_v4_eval.png")
    plt.close(fig)


def _plot_interpolation(model, cfg, m_indices, g_vectors, k_points,
                        k_dist, tick_pos, tick_labels):
    n_interp = 8
    bw = cfg.train_bw
    freqs = np.linspace(cfg.target_freq - bw, cfg.target_freq + bw, n_interp)

    fig, axes = plt.subplots(1, n_interp, figsize=(2.5 * n_interp, 2.5))

    for i, f in enumerate(freqs):
        tf_t = torch.tensor(f, dtype=torch.float64)
        with torch.no_grad():
            base = generate_random_base(cfg.n_grid, cfg.eps_bg, cfg.eps_rod)
            ff = extract_field_features(
                base, g_vectors, m_indices, k_points,
                cfg.n_bands, cfg.polarization, f,
                n_field_bands=cfg.n_field_bands)
            eps_grid, _, _ = model(tf_t, base, ff)
        eps_np = eps_grid.detach().cpu().numpy()

        ax = axes[i]
        ax.imshow(eps_np.T, origin="lower", extent=[0, 1, 0, 1],
                  cmap="RdYlBu_r", vmin=cfg.eps_bg, vmax=cfg.eps_rod)
        ax.set_title(f"f={f:.3f}", fontsize=9)
        ax.set_aspect("equal"); ax.tick_params(labelsize=6)

    fig.suptitle("Interpolation: geometry vs target frequency", fontsize=11)
    fig.tight_layout()
    fig.savefig("pinn_v4_interp.png", dpi=150)
    print("Saved pinn_v4_interp.png")
    plt.close(fig)


# ===================================================================
# 12. CLI
# ===================================================================

def load_model(checkpoint="pinn_v4_model.pt"):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    c = ckpt["cfg"]

    dummy_base = generate_random_base(c["n_grid"], c["eps_bg"], c["eps_rod"])
    g_vectors, m_indices = reciprocal_lattice(c["n_max"])
    k_points, _, _, _ = make_k_path(c["n_k_seg"])
    dummy_ff = extract_field_features(
        dummy_base, g_vectors, m_indices, k_points,
        c["n_bands"], c.get("polarization", "tm"), 0.35,
        n_field_bands=c.get("n_field_bands", 4))
    n_ff = dummy_ff.shape[0]

    model = PerturbationNet(
        N=c["n_grid"], latent_dim=c["latent_dim"],
        eps_bg=c["eps_bg"], eps_rod=c["eps_rod"],
        n_embed_freqs=c.get("n_embed_freqs", 3),
        n_field_features=n_ff,
        polarization=c.get("polarization", "tm"),
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

    if cfg.target_freq is not None:
        saved_cfg.target_freq = cfg.target_freq
    if not hasattr(saved_cfg, "train_bw"):
        saved_cfg.train_bw = 0.05

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
    parser.add_argument("--target-freq", type=float, default=0.35)
    parser.add_argument("--train-bw", type=float, default=0.05,
                        help="half-width of training sampling range around target-freq")
    parser.add_argument("--polarization", choices=["tm", "te"], default="tm")
    parser.add_argument("--gap-mode", choices=["ratio", "width"], default="width")
    parser.add_argument("--w-gap", type=float, default=1.0)
    parser.add_argument("--w-freq", type=float, default=20.0)
    parser.add_argument("--w-binary", type=float, default=0.1)
    parser.add_argument("--w-pert", type=float, default=0.1)
    parser.add_argument("--w-field", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-embed-freqs", type=int, default=3)
    parser.add_argument("--n-field-bands", type=int, default=4)
    parser.add_argument("--pert-curriculum", action="store_true")
    parser.add_argument("--seed", type=int, default=42)


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ("train", "eval"):
        p = argparse.ArgumentParser(
            description="Perturbation-aware inverse design v4")
        sub = p.add_subparsers(dest="command")
        tr = sub.add_parser("train")
        _add_train_args(tr)
        ev = sub.add_parser("eval")
        ev.add_argument("--checkpoint", type=str, default="pinn_v4_model.pt")
        ev.add_argument("--target-freq", type=float, default=None)
        cfg = p.parse_args()
        if cfg.command == "eval":
            eval_from_checkpoint(cfg)
        else:
            train(cfg)
    else:
        p = argparse.ArgumentParser(
            description="Perturbation-aware inverse design v4")
        _add_train_args(p)
        cfg = p.parse_args()
        train(cfg)


if __name__ == "__main__":
    main()
