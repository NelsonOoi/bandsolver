"""
Dual-physics CANNs for simultaneous photonic + phononic bandstructure (v3).

Two architectures, both using a shared LearnedKernelEncoder:
  - DualGridCANN:       independent GenericCANN backbones per physics
  - DualGridCANN_cross: cross-gated backbones that exchange information

Reuses LearnedKernelEncoder, GenericCANN, BandOrderingHead, SmoothInverse
from cann_v2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cann_v2 import (
    LearnedKernelEncoder, GenericCANN, BandOrderingHead, SmoothInverse,
)


# ---------------------------------------------------------------------------
# Helper: prepare grid input
# ---------------------------------------------------------------------------

def _prepare_grid(grid: torch.Tensor, grid_size: int) -> torch.Tensor:
    if grid.dim() == 3:
        grid = grid.unsqueeze(1)
    if grid.shape[-1] != grid_size:
        grid = F.interpolate(grid, size=grid_size,
                             mode="bilinear", align_corners=False)
    return grid


# ---------------------------------------------------------------------------
# Architecture 1: independent dual heads
# ---------------------------------------------------------------------------

class DualGridCANN(nn.Module):
    """Shared encoder -> two independent GenericCANN backbones."""

    def __init__(self, n_k: int = 31, n_bands_phot: int = 6,
                 n_bands_phon: int = 10, hidden: int = 128,
                 grid_size: int = 32, n_max: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.encoder = LearnedKernelEncoder(grid_size=grid_size, n_max=n_max)
        enc_dim = self.encoder.out_dim
        self.phot = GenericCANN(enc_dim, n_k, n_bands_phot, hidden)
        self.phon = GenericCANN(enc_dim, n_k, n_bands_phon, hidden)

    def forward(self, grid: torch.Tensor):
        grid = _prepare_grid(grid, self.grid_size)
        feat = self.encoder(grid)
        return self.phot(feat), self.phon(feat)


# ---------------------------------------------------------------------------
# Cross-gating block
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """Bilinear cross-gating between two hidden vectors.

    gate_a = sigmoid(W_a @ h_b)
    h_a'  = h_a + gate_a * V_a(h_b)

    Residual connection lets the model learn to ignore the cross-signal.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.gate_a = nn.Linear(hidden, hidden)
        self.gate_b = nn.Linear(hidden, hidden)
        self.val_a = nn.Linear(hidden, hidden)
        self.val_b = nn.Linear(hidden, hidden)
        self.sigmoid_a = nn.Sigmoid()
        self.sigmoid_b = nn.Sigmoid()

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor):
        g_a = self.sigmoid_a(self.gate_a(h_b))
        g_b = self.sigmoid_b(self.gate_b(h_a))
        h_a_out = h_a + g_a * self.val_a(h_b)
        h_b_out = h_b + g_b * self.val_b(h_a)
        return h_a_out, h_b_out


# ---------------------------------------------------------------------------
# Architecture 2: dual heads with cross-attention
# ---------------------------------------------------------------------------

class DualGridCANN_cross(nn.Module):
    """Shared encoder -> split FC layers with cross-gating in the middle.

    Layer structure per head mirrors GenericCANN:
      FC + SiLU -> FC + SmoothInverse -> [cross-gating] -> FC + softplus -> out + BandOrderingHead
    """

    def __init__(self, n_k: int = 31, n_bands_phot: int = 6,
                 n_bands_phon: int = 10, hidden: int = 128,
                 grid_size: int = 32, n_max: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.encoder = LearnedKernelEncoder(grid_size=grid_size, n_max=n_max)
        enc_dim = self.encoder.out_dim

        self.phot_fc1 = nn.Linear(enc_dim, hidden)
        self.phon_fc1 = nn.Linear(enc_dim, hidden)
        self.phot_act1 = nn.SiLU()
        self.phon_act1 = nn.SiLU()
        self.phot_fc2 = nn.Linear(hidden, hidden)
        self.phon_fc2 = nn.Linear(hidden, hidden)
        self.phot_act2 = SmoothInverse(hidden)
        self.phon_act2 = SmoothInverse(hidden)

        self.cross_attn = CrossAttentionBlock(hidden)

        self.phot_fc3 = nn.Linear(hidden, hidden)
        self.phon_fc3 = nn.Linear(hidden, hidden)
        self.phot_act3 = nn.Softplus()
        self.phon_act3 = nn.Softplus()
        self.phot_out = nn.Linear(hidden, n_k * n_bands_phot)
        self.phon_out = nn.Linear(hidden, n_k * n_bands_phon)
        self.phot_head = BandOrderingHead(n_k, n_bands_phot)
        self.phon_head = BandOrderingHead(n_k, n_bands_phon)

    def forward(self, grid: torch.Tensor):
        grid = _prepare_grid(grid, self.grid_size)
        feat = self.encoder(grid)

        h_p = self.phot_act1(self.phot_fc1(feat))
        h_n = self.phon_act1(self.phon_fc1(feat))
        h_p = self.phot_act2(self.phot_fc2(h_p))
        h_n = self.phon_act2(self.phon_fc2(h_n))

        h_p, h_n = self.cross_attn(h_p, h_n)

        h_p = self.phot_act3(self.phot_fc3(h_p))
        h_n = self.phon_act3(self.phon_fc3(h_n))

        return self.phot_head(self.phot_out(h_p)), self.phon_head(self.phon_out(h_n))
