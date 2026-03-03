"""
Constitutive Artificial Neural Networks (CANNs) for photonic bandstructure.

Two architectures:
  - FullyConnectedCANN: sequential FC layers with physics activations
  - ParallelBranchCANN: Kuhl-inspired independent Bessel branches per
    G-vector harmonic, coupled only after the constitutive layer

Shared physics-informed activations:
  - BesselActivation: J_1(x)/x  (Fourier coefficients of circular inclusion)
  - SmoothInverse:    x/(x^2+eta^2)  (matrix inversion analogue)
  - SqrtSoftplus:     sqrt(softplus(x))  (eigenvalue -> frequency)
  - Band ordering:    cumsum(softplus(delta))  (spectral ordering)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pwe_torch import reciprocal_lattice


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

class BesselActivation(nn.Module):
    """J_1(x)/x with safe limit at x=0 (-> 0.5)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        safe = x.abs().clamp(min=1e-7)
        return torch.where(
            x.abs() < 1e-7,
            torch.full_like(x, 0.5),
            torch.special.bessel_j1(safe) / safe,
        )


class SmoothInverse(nn.Module):
    """x / (x^2 + eta^2) with learnable regularization eta > 0."""

    def __init__(self, n_features: int):
        super().__init__()
        self.log_eta = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eta2 = torch.exp(self.log_eta * 2.0)
        return x / (x ** 2 + eta2)


class SqrtSoftplus(nn.Module):
    """sqrt(softplus(x)) — maps eigenvalue-scale to frequency-scale."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(F.softplus(x) + 1e-8)


# ---------------------------------------------------------------------------
# Fourier feature encoder
# ---------------------------------------------------------------------------

class FourierFeatureEncoder(nn.Module):
    """Encodes scalar input r into [r, sin(pi*r), cos(pi*r), ..., sin(n*pi*r), cos(n*pi*r)]."""

    def __init__(self, n_freqs: int = 8):
        super().__init__()
        self.n_freqs = n_freqs
        self.out_dim = 1 + 2 * n_freqs

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # r: (batch,) or (batch, 1)
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        freqs = torch.arange(1, self.n_freqs + 1, device=r.device, dtype=r.dtype)
        args = np.pi * r * freqs  # (batch, n_freqs)
        return torch.cat([r, torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# Band ordering output head (shared)
# ---------------------------------------------------------------------------

class BandOrderingHead(nn.Module):
    """Reshapes raw logits to (n_k, n_bands), enforces ordering via cumsum."""

    def __init__(self, n_k: int, n_bands: int):
        super().__init__()
        self.n_k = n_k
        self.n_bands = n_bands
        self.freq_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_k * n_bands)
        x = x.view(-1, self.n_k, self.n_bands)
        increments = F.softplus(x)
        bands = torch.cumsum(increments, dim=-1)
        return bands * self.freq_scale


# ---------------------------------------------------------------------------
# Architecture A: Fully Connected CANN
# ---------------------------------------------------------------------------

class FullyConnectedCANN(nn.Module):
    """Sequential FC layers mirroring the PWE pipeline.

    FourierFeatures -> Linear+Bessel -> Linear+SmoothInverse
    -> Linear+Softplus -> Linear -> BandOrderingHead
    """

    def __init__(self, n_k: int = 31, n_bands: int = 6,
                 hidden: int = 128, n_fourier: int = 8):
        super().__init__()
        self.encoder = FourierFeatureEncoder(n_fourier)
        in_dim = self.encoder.out_dim
        out_dim = n_k * n_bands

        self.fc1 = nn.Linear(in_dim, hidden)
        self.act1 = BesselActivation()

        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = SmoothInverse(hidden)

        self.fc3 = nn.Linear(hidden, hidden)
        # softplus activation (inline)

        self.fc_out = nn.Linear(hidden, out_dim)
        self.head = BandOrderingHead(n_k, n_bands)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = self.encoder(r)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = F.softplus(self.fc3(x))
        x = self.fc_out(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Architecture B: Parallel Branch CANN (Kuhl-inspired)
# ---------------------------------------------------------------------------

class ParallelBranchCANN(nn.Module):
    """Independent Bessel branches per G-vector harmonic, then FC coupling.

    Each branch computes w_i * J_1(alpha_i * r) / (alpha_i * r) where
    alpha_i is initialized to |G_i| from the reciprocal lattice.
    Branches are concatenated, then passed through SmoothInverse + FC layers.
    """

    def __init__(self, n_k: int = 31, n_bands: int = 6,
                 hidden: int = 128, n_max: int = 3):
        super().__init__()
        g_vectors, _ = reciprocal_lattice(n_max)
        g_norms = np.linalg.norm(g_vectors, axis=-1)
        # Deduplicate by unique |G| magnitudes (keep nonzero)
        unique_norms = np.unique(np.round(g_norms, 8))
        unique_norms = unique_norms[unique_norms > 0]
        self.n_harmonics = len(unique_norms)

        self.log_alpha = nn.Parameter(
            torch.log(torch.from_numpy(unique_norms).float())
        )
        self.branch_weights = nn.Parameter(torch.randn(self.n_harmonics) * 0.1)
        self.branch_bias = nn.Parameter(torch.zeros(self.n_harmonics))

        self.bessel = BesselActivation()

        branch_out = self.n_harmonics + 1  # +1 for raw r
        out_dim = n_k * n_bands

        self.fc1 = nn.Linear(branch_out, hidden)
        self.act1 = SmoothInverse(hidden)

        self.fc2 = nn.Linear(hidden, hidden)
        # softplus inline

        self.fc_out = nn.Linear(hidden, out_dim)
        self.head = BandOrderingHead(n_k, n_bands)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        if r.dim() == 1:
            r = r.unsqueeze(-1)  # (batch, 1)
        alpha = torch.exp(self.log_alpha)  # positive
        args = r * alpha  # (batch, n_harmonics)
        bessel_out = self.bessel(args)
        branches = self.branch_weights * bessel_out + self.branch_bias
        x = torch.cat([r, branches], dim=-1)  # (batch, n_harmonics + 1)

        x = self.act1(self.fc1(x))
        x = F.softplus(self.fc2(x))
        x = self.fc_out(x)
        return self.head(x)
