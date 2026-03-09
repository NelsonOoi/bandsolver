"""
Dual-physics CANNs v5: per-k conditioned, mask-input architecture.

Key changes from v4:
  - Input is a geometry mask (1=hole, 0=solid), not a dielectric grid
  - DeepCNNEncoder: hierarchical conv with residual connections
  - PerKBackbone: takes (geometry_feat, kx, ky) -> n_bands per k-point
  - No BandOrderingHead / cumsum — raw output
  - Designed for per-band standardized targets (zero mean, unit variance)

Architectures:
  - DualGridCANN_v5: shared DeepCNNEncoder on mask, independent PerKBackbone per physics
  - DualGridCANN_v5_cross: same, with per-k cross-gating between physics branches
"""

import torch
import torch.nn as nn

from cann_v2 import SmoothInverse
from cann_v3 import CrossAttentionBlock
from pwe_torch import make_k_path


def _prepare_mask(mask: torch.Tensor) -> torch.Tensor:
    """Ensure mask is (B, 1, G, G)."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return mask


# ---------------------------------------------------------------------------
# Deep CNN encoder with residual connections
# ---------------------------------------------------------------------------

class ResBlock2d(nn.Module):
    """Conv2d residual block: conv -> BN -> SiLU -> conv -> BN + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


class DeepCNNEncoder(nn.Module):
    """Hierarchical conv encoder: 1 -> 32 -> 64 -> 128 with residuals.

    Output: (batch, enc_dim) geometry embedding.
    """

    def __init__(self, grid_size: int = 32, enc_dim: int = 128):
        super().__init__()
        self.grid_size = grid_size
        self.out_dim = enc_dim

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            ResBlock2d(64),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            ResBlock2d(128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, enc_dim)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        x = self.stem(mask)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.pool(x).view(x.shape[0], -1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Per-k backbone (DeepONet-style: branch=geometry, trunk=k-point)
# ---------------------------------------------------------------------------

class PerKBackbone(nn.Module):
    """Predicts n_bands values per k-point, conditioned on geometry features.

    Input:  geo_feat (B, enc_dim),  k_points registered as buffer (n_k, 2)
    Output: (B, n_k, n_bands)

    The network is split into pre (input -> hidden) and post (hidden -> bands)
    to allow cross-gating between physics branches at the hidden level.
    """

    def __init__(self, enc_dim: int, n_k: int = 31, n_bands: int = 6,
                 hidden: int = 128):
        super().__init__()
        self.n_k = n_k
        self.n_bands = n_bands

        kpath, _, _, _ = make_k_path(n_per_segment=10)
        self.register_buffer("k_points", torch.from_numpy(kpath).float())

        in_dim = enc_dim + 2
        self.pre = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            SmoothInverse(hidden),
        )
        self.post = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_bands),
        )
        self.skip = nn.Linear(in_dim, n_bands)

    def _make_input(self, geo_feat: torch.Tensor) -> torch.Tensor:
        """Build (B, n_k, enc_dim+2) input from geometry features."""
        g = geo_feat.unsqueeze(1).expand(-1, self.n_k, -1)
        k = self.k_points.unsqueeze(0).expand(g.shape[0], -1, -1)
        return torch.cat([g, k], dim=-1)

    def forward(self, geo_feat: torch.Tensor) -> torch.Tensor:
        x = self._make_input(geo_feat)
        return self.post(self.pre(x)) + self.skip(x)


# ---------------------------------------------------------------------------
# Dual-physics v5
# ---------------------------------------------------------------------------

class DualGridCANN_v5(nn.Module):
    """Shared DeepCNNEncoder on mask -> two independent PerKBackbone heads.

    Input: mask (B, G, G) or (B, 1, G, G) with 1=hole, 0=solid.
    """

    def __init__(self, n_k: int = 31, n_bands_phot: int = 6,
                 n_bands_phon: int = 10, enc_dim: int = 128,
                 hidden_phot: int = 128, hidden_phon: int = 128,
                 grid_size: int = 32):
        super().__init__()
        self.grid_size = grid_size
        self.encoder = DeepCNNEncoder(grid_size=grid_size, enc_dim=enc_dim)
        self.phot = PerKBackbone(enc_dim, n_k, n_bands_phot, hidden_phot)
        self.phon = PerKBackbone(enc_dim, n_k, n_bands_phon, hidden_phon)

    def forward(self, mask: torch.Tensor):
        mask = _prepare_mask(mask)
        feat = self.encoder(mask)
        return self.phot(feat), self.phon(feat)


class DualGridCANN_v5_cross(nn.Module):
    """Like DualGridCANN_v5 but with per-k cross-gating between physics branches.

    Cross-gating is applied at the hidden layer between pre and post halves
    of each PerKBackbone, operating on (B, n_k, hidden) tensors.
    """

    def __init__(self, n_k: int = 31, n_bands_phot: int = 6,
                 n_bands_phon: int = 10, enc_dim: int = 128,
                 hidden_phot: int = 128, hidden_phon: int = 128,
                 grid_size: int = 32):
        super().__init__()
        self.grid_size = grid_size
        self.encoder = DeepCNNEncoder(grid_size=grid_size, enc_dim=enc_dim)
        self.phot = PerKBackbone(enc_dim, n_k, n_bands_phot, hidden_phot)
        self.phon = PerKBackbone(enc_dim, n_k, n_bands_phon, hidden_phon)
        self.cross = CrossAttentionBlock(hidden_phot)

    def forward(self, mask: torch.Tensor):
        mask = _prepare_mask(mask)
        feat = self.encoder(mask)

        x_phot = self.phot._make_input(feat)
        x_phon = self.phon._make_input(feat)

        h_phot = self.phot.pre(x_phot)
        h_phon = self.phon.pre(x_phon)
        h_phot, h_phon = self.cross(h_phot, h_phon)

        return (self.phot.post(h_phot) + self.phot.skip(x_phot),
                self.phon.post(h_phon) + self.phon.skip(x_phon))
