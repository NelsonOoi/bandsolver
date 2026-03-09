"""
Constitutive Artificial Neural Networks (CANNs) for photonic bandstructure v2.

New architectures with general geometry parameterization:
  - FourierShapeCANN: Fourier boundary descriptors -> learned kernel encoder -> backbone
  - LevelSetCANN:     Level-set RBF weights -> learned kernel encoder -> backbone

Shared components:
  - FourierShapeRasterizer: differentiable Fourier boundary -> soft dielectric grid
  - LevelSetRasterizer:     differentiable RBF level-set -> soft dielectric grid
  - LearnedKernelEncoder:   Conv2D kernels initialized to Fourier modes at G-vectors
  - GenericCANN:             FC backbone with SmoothInverse + softplus + BandOrderingHead

Backward-compatible re-exports of v1 classes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pwe_torch import reciprocal_lattice


# ---------------------------------------------------------------------------
# Activation functions (kept for backward compat)
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
# Band ordering output head (shared across all architectures)
# ---------------------------------------------------------------------------

class BandOrderingHead(nn.Module):
    """Reshapes raw logits to (n_k, n_bands), enforces ordering via cumsum."""

    def __init__(self, n_k: int, n_bands: int):
        super().__init__()
        self.n_k = n_k
        self.n_bands = n_bands
        self.freq_scale = nn.Parameter(torch.tensor(1.0))
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.n_k, self.n_bands)
        increments = self.softplus(x)
        bands = torch.cumsum(increments, dim=-1)
        return bands * self.freq_scale


# ---------------------------------------------------------------------------
# Legacy encoders (backward compat)
# ---------------------------------------------------------------------------

class FourierFeatureEncoder(nn.Module):
    """Encodes scalar input r into [r, sin(pi*r), cos(pi*r), ...]."""

    def __init__(self, n_freqs: int = 8):
        super().__init__()
        self.n_freqs = n_freqs
        self.out_dim = 1 + 2 * n_freqs

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        freqs = torch.arange(1, self.n_freqs + 1, device=r.device, dtype=r.dtype)
        args = np.pi * r * freqs
        return torch.cat([r, torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# Legacy architectures (backward compat)
# ---------------------------------------------------------------------------

class FullyConnectedCANN(nn.Module):
    """Sequential FC layers mirroring the PWE pipeline (v1)."""

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
        self.fc_out = nn.Linear(hidden, out_dim)
        self.head = BandOrderingHead(n_k, n_bands)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = self.encoder(r)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = F.softplus(self.fc3(x))
        x = self.fc_out(x)
        return self.head(x)


class ParallelBranchCANN(nn.Module):
    """Independent Bessel branches per G-vector harmonic (v1)."""

    def __init__(self, n_k: int = 31, n_bands: int = 6,
                 hidden: int = 128, n_max: int = 3):
        super().__init__()
        g_vectors, _ = reciprocal_lattice(n_max)
        g_norms = np.linalg.norm(g_vectors, axis=-1)
        unique_norms = np.unique(np.round(g_norms, 8))
        unique_norms = unique_norms[unique_norms > 0]
        self.n_harmonics = len(unique_norms)

        self.log_alpha = nn.Parameter(
            torch.log(torch.from_numpy(unique_norms).float()))
        self.branch_weights = nn.Parameter(torch.randn(self.n_harmonics) * 0.1)
        self.branch_bias = nn.Parameter(torch.zeros(self.n_harmonics))
        self.bessel = BesselActivation()

        branch_out = self.n_harmonics + 1
        out_dim = n_k * n_bands
        self.fc1 = nn.Linear(branch_out, hidden)
        self.act1 = SmoothInverse(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, out_dim)
        self.head = BandOrderingHead(n_k, n_bands)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        alpha = torch.exp(self.log_alpha)
        args = r * alpha
        bessel_out = self.bessel(args)
        branches = self.branch_weights * bessel_out + self.branch_bias
        x = torch.cat([r, branches], dim=-1)
        x = self.act1(self.fc1(x))
        x = F.softplus(self.fc2(x))
        x = self.fc_out(x)
        return self.head(x)


# ===================================================================
# V2 COMPONENTS
# ===================================================================

# ---------------------------------------------------------------------------
# Differentiable rasterizers
# ---------------------------------------------------------------------------

class FourierShapeRasterizer(nn.Module):
    """Differentiable rasterizer: Fourier boundary coefficients -> soft eps grid.

    Coefficients layout: [a_0, a_1, b_1, a_2, b_2, ..., a_N, b_N]
      r(theta) = a_0 + sum_n (a_n cos(n*theta) + b_n sin(n*theta))
    """

    def __init__(self, grid_size: int = 32, n_angles: int = 256,
                 beta_init: float = 5.0, beta_final: float = 40.0,
                 eps_bg: float = 8.9, eps_rod: float = 1.0):
        super().__init__()
        self.grid_size = grid_size
        self.n_angles = n_angles
        self.beta_init = beta_init
        self.beta_final = beta_final
        self.beta = beta_init
        self.eps_bg = eps_bg
        self.eps_rod = eps_rod

        xs = torch.linspace(-0.5, 0.5, grid_size + 1)[:-1] + 0.5 / grid_size
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")
        self.register_buffer("grid_x", xx)  # (G, G)
        self.register_buffer("grid_y", yy)
        self.register_buffer("grid_r", torch.sqrt(xx**2 + yy**2))
        self.register_buffer("grid_theta", torch.atan2(yy, xx))

        thetas = torch.linspace(0, 2 * np.pi, n_angles, dtype=torch.float32)
        self.register_buffer("thetas", thetas)

    def set_beta(self, frac: float):
        """Set beta by linear interpolation: frac in [0, 1]."""
        self.beta = self.beta_init + frac * (self.beta_final - self.beta_init)

    @staticmethod
    def n_coeffs_for(n_harmonics: int) -> int:
        return 1 + 2 * n_harmonics

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """coeffs: (batch, n_coeffs) -> (batch, 1, G, G) soft dielectric grid."""
        batch = coeffs.shape[0]
        n_coeffs = coeffs.shape[1]
        n_harmonics = (n_coeffs - 1) // 2

        a0 = coeffs[:, 0:1]  # (batch, 1)
        r_boundary = a0.unsqueeze(-1).expand(batch, 1, self.n_angles).squeeze(1)

        if n_harmonics > 0:
            ns = torch.arange(1, n_harmonics + 1, device=coeffs.device, dtype=coeffs.dtype)
            # a_n at indices 1, 3, 5, ... and b_n at 2, 4, 6, ...
            a_n = coeffs[:, 1::2]  # (batch, n_harmonics)
            b_n = coeffs[:, 2::2]  # (batch, n_harmonics)
            angles = ns[None, :, None] * self.thetas[None, None, :]  # (1, H, A)
            r_boundary = r_boundary + (a_n.unsqueeze(-1) * torch.cos(angles)
                                       + b_n.unsqueeze(-1) * torch.sin(angles)).sum(dim=1)

        # For each grid point, find angular boundary radius by interpolation
        # Vectorized: compute r(theta_grid) for each grid point's angle
        grid_theta = self.grid_theta.unsqueeze(0).expand(batch, -1, -1)  # (B, G, G)
        grid_theta_norm = (grid_theta % (2 * np.pi))  # [0, 2pi)

        # Evaluate r(theta) at each grid point's angle
        r_at_grid = a0.unsqueeze(-1).expand(batch, self.grid_size, self.grid_size)
        if n_harmonics > 0:
            gt = grid_theta_norm.unsqueeze(1)  # (B, 1, G, G)
            ns_view = ns[None, :, None, None]  # (1, H, 1, 1)
            r_at_grid = r_at_grid + (a_n[:, :, None, None] * torch.cos(ns_view * gt)
                                     + b_n[:, :, None, None] * torch.sin(ns_view * gt)).sum(dim=1)

        # Soft inside/outside via sigmoid
        signed_dist = r_at_grid - self.grid_r.unsqueeze(0)
        mask = torch.sigmoid(self.beta * signed_dist)

        eps_grid = self.eps_bg + (self.eps_rod - self.eps_bg) * mask
        return eps_grid.unsqueeze(1)  # (batch, 1, G, G)


class LevelSetRasterizer(nn.Module):
    """Differentiable rasterizer: RBF level-set weights -> soft eps grid.

    Places Gaussian RBFs on a coarse grid. Weights control the level-set
    field, which is thresholded via sigmoid to produce a soft mask.
    """

    def __init__(self, n_rbf_side: int = 4, grid_size: int = 32,
                 sigma: float = 0.15, beta_init: float = 5.0,
                 beta_final: float = 40.0,
                 eps_bg: float = 8.9, eps_rod: float = 1.0):
        super().__init__()
        self.n_rbf_side = n_rbf_side
        self.n_rbf = n_rbf_side ** 2
        self.grid_size = grid_size
        self.eps_bg = eps_bg
        self.eps_rod = eps_rod
        self.beta_init = beta_init
        self.beta_final = beta_final
        self.beta = beta_init

        # RBF centers on a regular grid within the unit cell
        cs = torch.linspace(-0.5 + 0.5 / n_rbf_side,
                            0.5 - 0.5 / n_rbf_side, n_rbf_side)
        cy, cx = torch.meshgrid(cs, cs, indexing="ij")
        centers = torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=-1)  # (n_rbf, 2)
        self.register_buffer("centers", centers)

        # Fine grid coordinates
        xs = torch.linspace(-0.5, 0.5, grid_size + 1)[:-1] + 0.5 / grid_size
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")
        grid_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (G*G, 2)
        self.register_buffer("grid_xy", grid_xy)

        # Precompute RBF values: (G*G, n_rbf)
        diff = grid_xy.unsqueeze(1) - centers.unsqueeze(0)  # (G*G, n_rbf, 2)
        rbf_vals = torch.exp(-0.5 * (diff ** 2).sum(dim=-1) / sigma ** 2)
        self.register_buffer("rbf_vals", rbf_vals)

    def set_beta(self, frac: float):
        """Set beta by linear interpolation: frac in [0, 1]."""
        self.beta = self.beta_init + frac * (self.beta_final - self.beta_init)

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """weights: (batch, n_rbf) -> (batch, 1, G, G) soft dielectric grid."""
        # phi(x) = sum_i w_i * G_i(x)
        phi = torch.einsum("gn,bn->bg", self.rbf_vals, weights)  # (batch, G*G)
        phi = phi.view(-1, self.grid_size, self.grid_size)

        mask = torch.sigmoid(self.beta * phi)
        eps_grid = self.eps_bg + (self.eps_rod - self.eps_bg) * mask
        return eps_grid.unsqueeze(1)  # (batch, 1, G, G)


# ---------------------------------------------------------------------------
# Learned kernel encoder
# ---------------------------------------------------------------------------

class LearnedKernelEncoder(nn.Module):
    """Conv2D kernels initialized to Fourier modes at reciprocal lattice G-vectors.

    At initialization the output equals the structure factor coefficients.
    Kernels are learnable and can adapt during training.
    """

    def __init__(self, grid_size: int = 32, n_max: int = 3):
        super().__init__()
        self.grid_size = grid_size

        g_vectors, _ = reciprocal_lattice(n_max)
        # Skip the DC component (G=0) — it's just mean epsilon
        nonzero = np.any(g_vectors != 0, axis=-1)
        g_vecs = g_vectors[nonzero]
        n_g = len(g_vecs)
        # n_kernels = 2*n_g (cos and sin for each G) + 1 (DC)
        self.n_kernels = 2 * n_g + 1
        self.out_dim = self.n_kernels

        # Build initialization kernels on the grid
        xs = torch.linspace(-0.5, 0.5, grid_size + 1)[:-1] + 0.5 / grid_size
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")
        xy = torch.stack([xx, yy], dim=-1)  # (G, G, 2)

        g_t = torch.from_numpy(g_vecs).float()  # (n_g, 2)
        # dot products: (n_g, G, G)
        gdot = torch.einsum("gd,ijd->gij", g_t, xy)

        kernels = torch.zeros(self.n_kernels, 1, grid_size, grid_size)
        # DC kernel: constant
        kernels[0, 0] = 1.0
        # Cos kernels
        kernels[1:n_g + 1, 0] = torch.cos(gdot)
        # Sin kernels
        kernels[n_g + 1:, 0] = torch.sin(gdot)

        self.weight = nn.Parameter(kernels)
        self.bias = nn.Parameter(torch.zeros(self.n_kernels))

    def forward(self, eps_grid: torch.Tensor) -> torch.Tensor:
        """eps_grid: (batch, 1, G, G) -> (batch, n_kernels)."""
        # Full-size convolution (kernel_size == input size) -> (batch, n_kernels, 1, 1)
        out = F.conv2d(eps_grid, self.weight, self.bias)
        return out.view(out.shape[0], -1)


# ---------------------------------------------------------------------------
# Generic backbone (geometry-agnostic)
# ---------------------------------------------------------------------------

class GenericCANN(nn.Module):
    """FC backbone: SiLU -> SmoothInverse -> softplus -> BandOrderingHead.

    Accepts any encoder that produces a fixed-size feature vector with
    attribute `out_dim`.
    """

    def __init__(self, in_dim: int, n_k: int = 31, n_bands: int = 6,
                 hidden: int = 128):
        super().__init__()
        out_dim = n_k * n_bands

        self.fc1 = nn.Linear(in_dim, hidden)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = SmoothInverse(hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.act3 = nn.Softplus()
        self.fc_out = nn.Linear(hidden, out_dim)
        self.head = BandOrderingHead(n_k, n_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc_out(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# V2 composite architectures
# ---------------------------------------------------------------------------

class FourierShapeCANN(nn.Module):
    """Fourier boundary descriptors -> learned kernels -> FC backbone."""

    def __init__(self, n_coeffs: int = 5, n_k: int = 31, n_bands: int = 6,
                 hidden: int = 128, grid_size: int = 32, n_max: int = 3,
                 eps_bg: float = 8.9, eps_rod: float = 1.0):
        super().__init__()
        self.rasterizer = FourierShapeRasterizer(
            grid_size=grid_size, eps_bg=eps_bg, eps_rod=eps_rod)
        self.encoder = LearnedKernelEncoder(grid_size=grid_size, n_max=n_max)
        self.backbone = GenericCANN(
            in_dim=self.encoder.out_dim, n_k=n_k, n_bands=n_bands, hidden=hidden)

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        eps_grid = self.rasterizer(coeffs)
        features = self.encoder(eps_grid)
        return self.backbone(features)


class LevelSetCANN(nn.Module):
    """RBF level-set weights -> learned kernels -> FC backbone."""

    def __init__(self, n_rbf_side: int = 4, n_k: int = 31, n_bands: int = 6,
                 hidden: int = 128, grid_size: int = 32, n_max: int = 3,
                 eps_bg: float = 8.9, eps_rod: float = 1.0):
        super().__init__()
        self.rasterizer = LevelSetRasterizer(
            n_rbf_side=n_rbf_side, grid_size=grid_size,
            eps_bg=eps_bg, eps_rod=eps_rod)
        self.encoder = LearnedKernelEncoder(grid_size=grid_size, n_max=n_max)
        self.backbone = GenericCANN(
            in_dim=self.encoder.out_dim, n_k=n_k, n_bands=n_bands, hidden=hidden)

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        eps_grid = self.rasterizer(weights)
        features = self.encoder(eps_grid)
        return self.backbone(features)


class GridCANN(nn.Module):
    """Dielectric grid -> learned kernels -> FC backbone.

    Takes the eps grid directly as input (no rasterizer), making the
    model geometry-agnostic. Input grids are downsampled to the encoder's
    grid_size if they don't match.
    """

    def __init__(self, n_k: int = 31, n_bands: int = 6,
                 hidden: int = 128, grid_size: int = 32, n_max: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.encoder = LearnedKernelEncoder(grid_size=grid_size, n_max=n_max)
        self.backbone = GenericCANN(
            in_dim=self.encoder.out_dim, n_k=n_k, n_bands=n_bands, hidden=hidden)

    def forward(self, eps_grid: torch.Tensor) -> torch.Tensor:
        """eps_grid: (batch, N, N) or (batch, 1, N, N) -> (batch, n_k, n_bands)."""
        if eps_grid.dim() == 3:
            eps_grid = eps_grid.unsqueeze(1)
        if eps_grid.shape[-1] != self.grid_size:
            eps_grid = F.interpolate(eps_grid, size=self.grid_size,
                                     mode="bilinear", align_corners=False)
        features = self.encoder(eps_grid)
        return self.backbone(features)
