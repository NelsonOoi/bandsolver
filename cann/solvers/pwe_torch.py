"""
Differentiable Plane Wave Expansion (PWE) solver in PyTorch.

Direct port of pwe.py with full autograd support through FFT, matrix
assembly, inverse, and eigh.  Units: a = 1, c = 1.
"""

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Reciprocal lattice (returns numpy -- no grad needed)
# ---------------------------------------------------------------------------

def reciprocal_lattice(n_max: int):
    """G vectors for a square lattice truncated to |m1|, |m2| <= n_max."""
    ms = np.arange(-n_max, n_max + 1)
    m1, m2 = np.meshgrid(ms, ms, indexing="ij")
    m_indices = np.stack([m1.ravel(), m2.ravel()], axis=-1)
    g_vectors = 2.0 * np.pi * m_indices.astype(np.float64)
    return g_vectors, m_indices


# ---------------------------------------------------------------------------
# 2. Epsilon matrix from FFT (differentiable)
# ---------------------------------------------------------------------------

def build_epsilon_matrix(eps_grid: torch.Tensor, m_indices_np: np.ndarray):
    """Build epsilon Fourier-coefficient matrix.

    Args:
        eps_grid: (N, N) real tensor -- dielectric grid.
        m_indices_np: (n_pw, 2) int numpy array.

    Returns:
        eps_mat: (n_pw, n_pw) complex tensor with eps_mat[i,j] = eps_hat(G_i - G_j).
    """
    N = eps_grid.shape[0]
    eps_fft = torch.fft.fft2(eps_grid.to(torch.complex128)) / (N * N)

    dm = m_indices_np[:, None, :] - m_indices_np[None, :, :]  # numpy
    idx0 = torch.from_numpy(dm[..., 0] % N).long()
    idx1 = torch.from_numpy(dm[..., 1] % N).long()
    eps_mat = eps_fft[idx0, idx1]
    return eps_mat


# ---------------------------------------------------------------------------
# 3. Band solver (differentiable)
# ---------------------------------------------------------------------------

def solve_bands(k_points_np: np.ndarray,
                g_vectors_np: np.ndarray,
                eps_grid: torch.Tensor,
                m_indices_np: np.ndarray,
                n_bands: int,
                polarization: str = "tm"):
    """Compute band structure -- fully differentiable w.r.t. eps_grid.

    Args:
        k_points_np: (n_k, 2) numpy array of wavevectors.
        g_vectors_np: (n_pw, 2) numpy array of G vectors.
        eps_grid: (N, N) torch tensor (requires_grad ok).
        m_indices_np: (n_pw, 2) numpy int array.
        n_bands: number of bands to return.
        polarization: "tm" or "te".

    Returns:
        freqs: (n_k, n_bands) tensor of normalized frequencies.
    """
    eps_mat = build_epsilon_matrix(eps_grid, m_indices_np)
    eps_mat_inv = torch.linalg.inv(eps_mat)
    return _solve_bands_from_inv(k_points_np, g_vectors_np, eps_mat_inv,
                                 n_bands, polarization)


_EPS = 1e-12  # numerical floor to avoid 0/0 and inf gradients


def _safe_sqrt(x):
    """sqrt with clamped input to avoid NaN gradients at zero."""
    return torch.sqrt(torch.clamp(x, min=_EPS))


def _solve_bands_from_inv(k_points_np, g_vectors_np, eps_mat_inv,
                          n_bands, polarization="tm", return_vecs=False):
    """Vectorized band solve from precomputed eps_mat_inv.

    Guards against NaN gradients from:
      - sqrt(0) at the Gamma point (|k+G|=0)
      - degenerate eigenvalues in eigh backward (adds small jitter)
    """
    k_pts = torch.from_numpy(k_points_np).to(torch.float64)
    g_vecs = torch.from_numpy(g_vectors_np).to(torch.float64)

    kpg = k_pts[:, None, :] + g_vecs[None, :, :]  # (n_k, n_pw, 2)

    if polarization == "tm":
        kpg_norm = _safe_sqrt(torch.sum(kpg ** 2, dim=-1))  # (n_k, n_pw)
        H = kpg_norm[:, :, None] * eps_mat_inv.unsqueeze(0) * kpg_norm[:, None, :]
    else:
        dot = torch.sum(kpg[:, :, None, :] * kpg[:, None, :, :], dim=-1)
        H = dot * eps_mat_inv.unsqueeze(0)

    # Enforce Hermitian symmetry
    H = 0.5 * (H + H.conj().transpose(-2, -1))

    # Small diagonal jitter breaks exact degeneracies that make eigh
    # backward numerically unstable (grad ~ 1/(lambda_i - lambda_j)).
    jitter = torch.eye(H.shape[-1], dtype=H.dtype) * _EPS
    H = H + jitter

    eigvals, eigvecs = torch.linalg.eigh(H)
    eigvals = torch.clamp(eigvals, min=0.0)
    freqs = _safe_sqrt(eigvals[:, :n_bands]) / (2.0 * np.pi)

    if return_vecs:
        return freqs, eigvecs[:, :, :n_bands], kpg_norm if polarization == "tm" else kpg
    return freqs


# ---------------------------------------------------------------------------
# 4. Smooth min / max (differentiable)
# ---------------------------------------------------------------------------

def smooth_min(x: torch.Tensor, beta: float = -50.0):
    """Differentiable approximation to min via softmax weighting."""
    # Clamp beta*x to avoid overflow in exp
    bx = torch.clamp(beta * x, min=-80.0, max=80.0)
    w = torch.softmax(bx, dim=0)
    return torch.sum(x * w)


def smooth_max(x: torch.Tensor, beta: float = 50.0):
    """Differentiable approximation to max via softmax weighting."""
    bx = torch.clamp(beta * x, min=-80.0, max=80.0)
    w = torch.softmax(bx, dim=0)
    return torch.sum(x * w)


def extract_gap(bands: torch.Tensor, band_lo: int, band_hi: int,
                beta: float = 50.0):
    """Extract gap width and midgap frequency between two bands."""
    floor = smooth_max(bands[:, band_lo], beta=beta)
    ceil = smooth_min(bands[:, band_hi], beta=-beta)
    return ceil - floor, 0.5 * (ceil + floor)


# ---------------------------------------------------------------------------
# 5. k-path utility (numpy, same as pwe.py)
# ---------------------------------------------------------------------------

def make_k_path(n_per_segment=10):
    """Generate k-path Gamma -> X -> M -> Gamma for a square lattice."""
    gamma = np.array([0.0, 0.0])
    x = np.array([np.pi, 0.0])
    m = np.array([np.pi, np.pi])

    segments = [(gamma, x), (x, m), (m, gamma)]
    seg_lengths = [float(np.linalg.norm(b - a)) for a, b in segments]
    total_length = sum(seg_lengths)

    n_pts = [max(2, int(round(n_per_segment * sl / (total_length / len(segments)))))
             for sl in seg_lengths]

    k_points, k_dist = [], []
    offset = 0.0

    for (a, b), n in zip(segments, n_pts):
        ts = np.linspace(0.0, 1.0, n, endpoint=False)
        seg_k = a[None, :] + ts[:, None] * (b - a)[None, :]
        seg_d = offset + ts * float(np.linalg.norm(b - a))
        k_points.append(seg_k)
        k_dist.append(seg_d)
        offset = float(seg_d[-1]) + float(np.linalg.norm(b - a)) / (n - 1) if n > 1 else offset

    k_points.append(gamma[None, :])
    k_dist.append(np.array([offset]))

    k_points = np.concatenate(k_points, axis=0)
    k_dist = np.concatenate(k_dist, axis=0)

    tick_positions = [0.0]
    cum = 0.0
    for sl in seg_lengths:
        cum += sl
        tick_positions.append(cum)
    tick_labels = ["$\\Gamma$", "X", "M", "$\\Gamma$"]

    return k_points, k_dist, tick_positions, tick_labels

