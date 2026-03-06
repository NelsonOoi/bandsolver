"""
Differentiable in-plane elastic Plane Wave Expansion (PWE) solver in PyTorch.

Solves the 2D in-plane elastic wave equation for phononic crystals on a
square lattice.  Fully differentiable w.r.t. the material property grids
(rho, lambda, mu) via FFT, Cholesky, and eigh.

Convention: mask=1 is the hole (air), mask=0 is the matrix (silicon).
Units: a = 1.
"""

import numpy as np
import torch

from pwe_torch import build_epsilon_matrix, _safe_sqrt, _EPS


# ---------------------------------------------------------------------------
# Default material properties (SI units)
# ---------------------------------------------------------------------------

SI_PROPS = dict(rho=2330.0, lam=68.4e9, mu=80.0e9)
AIR_PROPS = dict(rho=1.225, lam=1e-4, mu=1e-4)


# ---------------------------------------------------------------------------
# Mask -> material grids
# ---------------------------------------------------------------------------

def make_material_grids(mask_grid: torch.Tensor,
                        mat_hole: dict = None,
                        mat_matrix: dict = None):
    """Convert a binary/soft mask to (rho, lam, mu) grids.

    mask=1 -> hole material (default air)
    mask=0 -> matrix material (default silicon)

    Returns (rho_grid, lam_grid, mu_grid) each (N, N) float64 tensors.
    """
    if mat_hole is None:
        mat_hole = AIR_PROPS
    if mat_matrix is None:
        mat_matrix = SI_PROPS

    m = mask_grid.to(torch.float64)
    rho_grid = mat_matrix["rho"] + (mat_hole["rho"] - mat_matrix["rho"]) * m
    lam_grid = mat_matrix["lam"] + (mat_hole["lam"] - mat_matrix["lam"]) * m
    mu_grid = mat_matrix["mu"] + (mat_hole["mu"] - mat_matrix["mu"]) * m
    return rho_grid, lam_grid, mu_grid


# ---------------------------------------------------------------------------
# In-plane elastic band solver
# ---------------------------------------------------------------------------

def solve_elastic_bands(k_points_np: np.ndarray,
                        g_vectors_np: np.ndarray,
                        rho_grid: torch.Tensor,
                        lam_grid: torch.Tensor,
                        mu_grid: torch.Tensor,
                        m_indices_np: np.ndarray,
                        n_bands: int):
    """Compute in-plane elastic band structure.

    Fully differentiable w.r.t. rho_grid, lam_grid, mu_grid.

    Args:
        k_points_np: (n_k, 2) numpy wavevectors.
        g_vectors_np: (n_pw, 2) numpy G-vectors.
        rho_grid: (N, N) density grid.
        lam_grid: (N, N) first Lame parameter grid.
        mu_grid:  (N, N) shear modulus grid.
        m_indices_np: (n_pw, 2) integer indices.
        n_bands: number of lowest bands to return.

    Returns:
        freqs: (n_k, n_bands) tensor of angular frequencies (rad/s * a).
    """
    rho_mat = build_epsilon_matrix(rho_grid, m_indices_np)
    lam_2mu_mat = build_epsilon_matrix(lam_grid + 2.0 * mu_grid, m_indices_np)
    mu_mat = build_epsilon_matrix(mu_grid, m_indices_np)
    lam_mat = build_epsilon_matrix(lam_grid, m_indices_np)

    n_pw = len(g_vectors_np)
    k_pts = torch.from_numpy(k_points_np).to(torch.float64)
    g_vecs = torch.from_numpy(g_vectors_np).to(torch.float64)

    kpg = k_pts[:, None, :] + g_vecs[None, :, :]  # (n_k, n_pw, 2)
    kx = kpg[..., 0]  # (n_k, n_pw)
    ky = kpg[..., 1]

    # Stiffness sub-blocks: (n_k, n_pw, n_pw)
    Gxx = (kx[:, :, None] * lam_2mu_mat.unsqueeze(0) * kx[:, None, :]
           + ky[:, :, None] * mu_mat.unsqueeze(0) * ky[:, None, :])
    Gyy = (ky[:, :, None] * lam_2mu_mat.unsqueeze(0) * ky[:, None, :]
           + kx[:, :, None] * mu_mat.unsqueeze(0) * kx[:, None, :])
    Gxy = (kx[:, :, None] * lam_mat.unsqueeze(0) * ky[:, None, :]
           + ky[:, :, None] * mu_mat.unsqueeze(0) * kx[:, None, :])

    # Assemble full stiffness matrix (n_k, 2*n_pw, 2*n_pw)
    n_k = len(k_points_np)
    Gamma = torch.zeros(n_k, 2 * n_pw, 2 * n_pw,
                        dtype=torch.complex128, device=rho_grid.device)
    Gamma[:, :n_pw, :n_pw] = Gxx
    Gamma[:, :n_pw, n_pw:] = Gxy
    Gamma[:, n_pw:, :n_pw] = Gxy.conj().transpose(-2, -1)
    Gamma[:, n_pw:, n_pw:] = Gyy

    # Block-diagonal mass matrix
    M = torch.zeros_like(Gamma)
    M[:, :n_pw, :n_pw] = rho_mat.unsqueeze(0)
    M[:, n_pw:, n_pw:] = rho_mat.unsqueeze(0)

    # Enforce Hermitian symmetry
    Gamma = 0.5 * (Gamma + Gamma.conj().transpose(-2, -1))
    M = 0.5 * (M + M.conj().transpose(-2, -1))

    # Reduce to standard eigenproblem via Cholesky: M = L L^H
    # Then H = L^{-1} Gamma L^{-H}, solve H v = omega^2 v
    L = torch.linalg.cholesky(M)
    L_inv = torch.linalg.inv(L)
    H = L_inv @ Gamma @ L_inv.conj().transpose(-2, -1)

    H = 0.5 * (H + H.conj().transpose(-2, -1))

    jitter = torch.eye(H.shape[-1], dtype=H.dtype, device=H.device) * _EPS
    H = H + jitter

    eigvals, _ = torch.linalg.eigh(H)
    eigvals = torch.clamp(eigvals.real, min=0.0)
    freqs = _safe_sqrt(eigvals[:, :n_bands]) / (2.0 * np.pi)

    return freqs


# ---------------------------------------------------------------------------
# Convenience wrapper (mirrors pwe_torch.solve_bands)
# ---------------------------------------------------------------------------

def solve_elastic_bands_from_mask(k_points_np: np.ndarray,
                                  g_vectors_np: np.ndarray,
                                  mask_grid: torch.Tensor,
                                  m_indices_np: np.ndarray,
                                  n_bands: int,
                                  mat_hole: dict = None,
                                  mat_matrix: dict = None):
    """Compute in-plane elastic bands from a geometry mask.

    mask=1 is hole (air), mask=0 is matrix (silicon).
    Drop-in replacement for pwe_torch.solve_bands in training scripts.
    """
    rho_grid, lam_grid, mu_grid = make_material_grids(
        mask_grid, mat_hole=mat_hole, mat_matrix=mat_matrix)
    return solve_elastic_bands(k_points_np, g_vectors_np,
                               rho_grid, lam_grid, mu_grid,
                               m_indices_np, n_bands)
