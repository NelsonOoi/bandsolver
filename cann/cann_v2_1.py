"""
Constitutive Artificial Neural Networks (CANNs) for photonic bandstructure v2.1.

Improvements over v2:
  - CNNEncoder: lightweight multi-scale conv encoder (16->32->64 channels)
  - FC backbone with dropout for regularization
  - Full-output prediction (fast) with BandOrderingHead

Reuses rasterizers from cann_v2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cann_v2 import (
    SmoothInverse, BandOrderingHead,
    FourierShapeRasterizer, LevelSetRasterizer,
)


# ---------------------------------------------------------------------------
# Lightweight CNN encoder
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """Lightweight hierarchical conv encoder for dielectric grids."""

    def __init__(self, grid_size: int = 32, out_features: int = 64):
        super().__init__()
        self.grid_size = grid_size
        self.out_dim = out_features

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, out_features)

    def forward(self, eps_grid: torch.Tensor) -> torch.Tensor:
        x = self.conv(eps_grid).view(eps_grid.shape[0], -1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# FC backbone with dropout
# ---------------------------------------------------------------------------

class Backbone(nn.Module):
    """FC backbone with dropout. Predicts all (n_k * n_bands) at once."""

    def __init__(self, in_dim: int, n_k: int = 31, n_bands: int = 6,
                 hidden: int = 192, dropout: float = 0.1):
        super().__init__()
        out_dim = n_k * n_bands

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            SmoothInverse(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.head = BandOrderingHead(n_k, n_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


# ---------------------------------------------------------------------------
# Composite architectures
# ---------------------------------------------------------------------------

class GridCANN_v21(nn.Module):
    """Dielectric grid -> CNN encoder -> FC backbone."""

    def __init__(self, n_k: int = 31, n_bands: int = 6,
                 hidden: int = 192, grid_size: int = 32,
                 enc_features: int = 64, dropout: float = 0.1):
        super().__init__()
        self.grid_size = grid_size
        self.encoder = CNNEncoder(grid_size=grid_size, out_features=enc_features)
        self.backbone = Backbone(
            in_dim=enc_features, n_k=n_k, n_bands=n_bands,
            hidden=hidden, dropout=dropout)

    def forward(self, eps_grid: torch.Tensor) -> torch.Tensor:
        if eps_grid.dim() == 3:
            eps_grid = eps_grid.unsqueeze(1)
        if eps_grid.shape[-1] != self.grid_size:
            eps_grid = F.interpolate(eps_grid, size=self.grid_size,
                                     mode="bilinear", align_corners=False)
        return self.backbone(self.encoder(eps_grid))


class LevelSetCANN_v21(nn.Module):
    """RBF level-set -> rasterizer -> CNN encoder -> FC backbone."""

    def __init__(self, n_rbf_side: int = 4, n_k: int = 31, n_bands: int = 6,
                 hidden: int = 192, grid_size: int = 32,
                 enc_features: int = 64, dropout: float = 0.1,
                 eps_bg: float = 8.9, eps_rod: float = 1.0):
        super().__init__()
        self.rasterizer = LevelSetRasterizer(
            n_rbf_side=n_rbf_side, grid_size=grid_size,
            eps_bg=eps_bg, eps_rod=eps_rod)
        self.encoder = CNNEncoder(grid_size=grid_size, out_features=enc_features)
        self.backbone = Backbone(
            in_dim=enc_features, n_k=n_k, n_bands=n_bands,
            hidden=hidden, dropout=dropout)

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        eps_grid = self.rasterizer(weights)
        return self.backbone(self.encoder(eps_grid))


class FourierShapeCANN_v21(nn.Module):
    """Fourier boundary -> rasterizer -> CNN encoder -> FC backbone."""

    def __init__(self, n_coeffs: int = 5, n_k: int = 31, n_bands: int = 6,
                 hidden: int = 192, grid_size: int = 32,
                 enc_features: int = 64, dropout: float = 0.1,
                 eps_bg: float = 8.9, eps_rod: float = 1.0):
        super().__init__()
        self.rasterizer = FourierShapeRasterizer(
            grid_size=grid_size, eps_bg=eps_bg, eps_rod=eps_rod)
        self.encoder = CNNEncoder(grid_size=grid_size, out_features=enc_features)
        self.backbone = Backbone(
            in_dim=enc_features, n_k=n_k, n_bands=n_bands,
            hidden=hidden, dropout=dropout)

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        eps_grid = self.rasterizer(coeffs)
        return self.backbone(self.encoder(eps_grid))
