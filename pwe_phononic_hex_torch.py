"""
Differentiable in-plane elastic Plane Wave Expansion solver for a 2D
hexagonal lattice.

Solves the generalized eigenvalue problem:
    Gamma * u = omega^2 * M * u
where Gamma is the 2n_pw x 2n_pw stiffness matrix and M is the mass
matrix (block-diagonal with rho Fourier coefficients).

Uses Cholesky reduction to convert to a standard eigenvalue problem:
    H_eff * v = omega^2 * v,  where H_eff = L^{-1} Gamma L^{-T}

Material system: isotropic, c12 = c11 - 2*c44.
Normalized frequency: omega * a / (2*pi*v_ref), v_ref = sqrt(c44_Si/rho_Si).

Fully differentiable through FFT, matrix assembly, Cholesky, and eigh.
"""

import numpy as np
import torch

from pwe_hex_torch import (hex_reciprocal_lattice, hex_build_fourier_matrix,
                           hex_make_k_path, smooth_min, smooth_max,
                           find_best_gap, _safe_sqrt, _EPS)

# ---------------------------------------------------------------------------
# Material constants
# ---------------------------------------------------------------------------

SI_MATERIALS = {
    "silicon": {"eps": 11.7, "rho": 2329.0, "c11": 166e9, "c44": 80e9},
    "air":     {"eps": 1.0,  "rho": 1.225,  "c11": 1e5,   "c44": 1e3},
}

# Reference velocity for normalization
V_REF = np.sqrt(SI_MATERIALS["silicon"]["c44"] / SI_MATERIALS["silicon"]["rho"])


# ---------------------------------------------------------------------------
# 1. Material Fourier matrices from binary geometry
# ---------------------------------------------------------------------------

def build_material_matrices(s_grid: torch.Tensor,
                            m_indices_np: np.ndarray,
                            mat_A: dict, mat_B: dict):
    """Build Fourier-coefficient Toeplitz matrices for elastic properties.

    s_grid: (N, N) binary geometry (1 = material A, 0 = material B)
    mat_A, mat_B: dicts with keys "rho", "c11", "c44"

    Returns:
        c11_mat, c44_mat, c12_mat, rho_mat: each (n_pw, n_pw) complex tensors
    """
    c11_grid = mat_B["c11"] + (mat_A["c11"] - mat_B["c11"]) * s_grid
    c44_grid = mat_B["c44"] + (mat_A["c44"] - mat_B["c44"]) * s_grid
    c12_grid = c11_grid - 2.0 * c44_grid
    rho_grid = mat_B["rho"] + (mat_A["rho"] - mat_B["rho"]) * s_grid

    c11_mat = hex_build_fourier_matrix(c11_grid, m_indices_np)
    c44_mat = hex_build_fourier_matrix(c44_grid, m_indices_np)
    c12_mat = hex_build_fourier_matrix(c12_grid, m_indices_np)
    rho_mat = hex_build_fourier_matrix(rho_grid, m_indices_np)

    return c11_mat, c44_mat, c12_mat, rho_mat


# ---------------------------------------------------------------------------
# 2. Phononic band solver
# ---------------------------------------------------------------------------

def solve_phononic_bands(k_points_np: np.ndarray,
                         g_vectors_np: np.ndarray,
                         s_grid: torch.Tensor,
                         m_indices_np: np.ndarray,
                         n_bands: int,
                         mat_A: dict = None,
                         mat_B: dict = None):
    """Solve in-plane elastic band structure on a hexagonal lattice.

    Args:
        k_points_np: (n_k, 2) wavevectors in Cartesian coords.
        g_vectors_np: (n_pw, 2) reciprocal lattice vectors.
        s_grid: (N, N) binary geometry on the oblique grid.
        m_indices_np: (n_pw, 2) integer indices.
        n_bands: number of bands to return.
        mat_A: material dict for s=1 (default: silicon).
        mat_B: material dict for s=0 (default: air).

    Returns:
        freqs: (n_k, n_bands) normalized frequencies omega*a/(2*pi*v_ref).
    """
    if mat_A is None:
        mat_A = SI_MATERIALS["silicon"]
    if mat_B is None:
        mat_B = SI_MATERIALS["air"]

    c11_mat, c44_mat, c12_mat, rho_mat = build_material_matrices(
        s_grid, m_indices_np, mat_A, mat_B)

    return _solve_phononic_from_mats(
        k_points_np, g_vectors_np, c11_mat, c44_mat, c12_mat, rho_mat,
        n_bands)


def _solve_phononic_from_mats(k_points_np, g_vectors_np,
                              c11_mat, c44_mat, c12_mat, rho_mat,
                              n_bands):
    """Assemble and solve the elastic eigenvalue problem.

    Gamma (2n_pw x 2n_pw):
        | Gamma_xx  Gamma_xy |
        | Gamma_yx  Gamma_yy |

    Gamma_xx[G,G'] = kgx * c11 * kgx' + kgy * c44 * kgy'
    Gamma_yy[G,G'] = kgx * c44 * kgx' + kgy * c11 * kgy'
    Gamma_xy[G,G'] = kgx * c12 * kgy' + kgy * c44 * kgx'
    Gamma_yx = Gamma_xy^T

    M = | rho  0   |
        | 0    rho |
    """
    k_pts = torch.from_numpy(k_points_np).to(torch.float64)
    g_vecs = torch.from_numpy(g_vectors_np).to(torch.float64)
    n_pw = g_vecs.shape[0]
    n_k = k_pts.shape[0]

    kpg = k_pts[:, None, :] + g_vecs[None, :, :]  # (n_k, n_pw, 2)
    kgx = kpg[:, :, 0]  # (n_k, n_pw)
    kgy = kpg[:, :, 1]

    # Assemble Gamma blocks (n_k, n_pw, n_pw)
    def _outer_mat(ki, mat, kj):
        """ki (n_k, n_pw) * mat (n_pw, n_pw) * kj (n_k, n_pw)
        -> (n_k, n_pw, n_pw)"""
        return ki[:, :, None] * mat.unsqueeze(0) * kj[:, None, :]

    G_xx = _outer_mat(kgx, c11_mat, kgx) + _outer_mat(kgy, c44_mat, kgy)
    G_yy = _outer_mat(kgx, c44_mat, kgx) + _outer_mat(kgy, c11_mat, kgy)
    G_xy = _outer_mat(kgx, c12_mat, kgy) + _outer_mat(kgy, c44_mat, kgx)

    # Full Gamma: (n_k, 2*n_pw, 2*n_pw)
    Gamma = torch.zeros(n_k, 2 * n_pw, 2 * n_pw, dtype=torch.complex128)
    Gamma[:, :n_pw, :n_pw] = G_xx
    Gamma[:, :n_pw, n_pw:] = G_xy
    Gamma[:, n_pw:, :n_pw] = G_xy.conj().transpose(-2, -1)
    Gamma[:, n_pw:, n_pw:] = G_yy

    # Enforce Hermitian
    Gamma = 0.5 * (Gamma + Gamma.conj().transpose(-2, -1))

    # Mass matrix: block diagonal with rho_mat
    M = torch.zeros(2 * n_pw, 2 * n_pw, dtype=torch.complex128)
    M[:n_pw, :n_pw] = rho_mat
    M[n_pw:, n_pw:] = rho_mat
    M = 0.5 * (M + M.conj().transpose(-2, -1))

    # Add jitter for numerical stability
    jitter = torch.eye(2 * n_pw, dtype=torch.complex128) * _EPS
    M = M + jitter
    Gamma = Gamma + jitter.unsqueeze(0)

    # Cholesky reduction: M = L L^T, H_eff = L^{-1} Gamma L^{-T}
    L = torch.linalg.cholesky(M)
    L_inv = torch.linalg.inv(L)

    # H_eff = L_inv @ Gamma @ L_inv^H for each k
    H_eff = L_inv.unsqueeze(0) @ Gamma @ L_inv.conj().transpose(-2, -1).unsqueeze(0)
    H_eff = 0.5 * (H_eff + H_eff.conj().transpose(-2, -1))

    eigvals, _ = torch.linalg.eigh(H_eff)
    eigvals = torch.clamp(eigvals.real, min=0.0)

    # omega = sqrt(eigval), normalized: omega * a / (2*pi*v_ref)
    omega = _safe_sqrt(eigvals[:, :n_bands])
    freqs = omega / (2.0 * np.pi * V_REF)

    return freqs
