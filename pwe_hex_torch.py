"""
Differentiable Plane Wave Expansion (PWE) solver for a 2D hexagonal lattice.

Full autograd support through FFT, matrix assembly, inverse, and eigh.
Units: a = 1 (lattice constant), c = 1 (speed of light).

Hexagonal lattice:
    a1 = (1, 0),  a2 = (1/2, sqrt(3)/2)
    b1 = 2*pi*(1, -1/sqrt(3)),  b2 = 2*pi*(0, 2/sqrt(3))
    BZ path: Gamma -> M -> K -> Gamma
"""

import numpy as np
import torch

_EPS = 1e-12
_SQRT3 = np.sqrt(3.0)

# Real-space primitive vectors (a = 1)
A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, _SQRT3 / 2.0])

# Reciprocal lattice vectors
B1 = 2.0 * np.pi * np.array([1.0, -1.0 / _SQRT3])
B2 = 2.0 * np.pi * np.array([0.0, 2.0 / _SQRT3])


# ---------------------------------------------------------------------------
# 1. Reciprocal lattice
# ---------------------------------------------------------------------------

def hex_reciprocal_lattice(n_max: int):
    """G vectors for a hexagonal lattice truncated to |m1|, |m2| <= n_max.

    Returns:
        g_vectors: (n_pw, 2) float64 array of physical G vectors.
        m_indices: (n_pw, 2) int array of integer indices (m1, m2).
    """
    ms = np.arange(-n_max, n_max + 1)
    m1, m2 = np.meshgrid(ms, ms, indexing="ij")
    m_indices = np.stack([m1.ravel(), m2.ravel()], axis=-1)
    g_vectors = m_indices[:, 0:1] * B1[None, :] + m_indices[:, 1:2] * B2[None, :]
    return g_vectors.astype(np.float64), m_indices


# ---------------------------------------------------------------------------
# 2. Fourier coefficient matrix (differentiable)
# ---------------------------------------------------------------------------

def hex_build_fourier_matrix(field_grid: torch.Tensor,
                             m_indices_np: np.ndarray):
    """Build Toeplitz Fourier-coefficient matrix from a field on the
    rhombic unit cell.

    The grid is N x N in oblique coordinates (s1, s2) in [0,1)^2 where
    r = s1*a1 + s2*a2.  A standard fft2 on this grid expands in the
    reciprocal basis (b1, b2), so the (m1, m2) Fourier coefficient is
    at fft index (m1 % N, m2 % N).

    Args:
        field_grid: (N, N) real tensor on the oblique grid.
        m_indices_np: (n_pw, 2) int array.

    Returns:
        mat: (n_pw, n_pw) complex tensor with mat[i,j] = f_hat(G_i - G_j).
    """
    N = field_grid.shape[0]
    fft = torch.fft.fft2(field_grid.to(torch.complex128)) / (N * N)

    dm = m_indices_np[:, None, :] - m_indices_np[None, :, :]
    idx0 = torch.from_numpy(dm[..., 0] % N).long()
    idx1 = torch.from_numpy(dm[..., 1] % N).long()
    return fft[idx0, idx1]


# ---------------------------------------------------------------------------
# 3. Band solver (differentiable)
# ---------------------------------------------------------------------------

def _safe_sqrt(x):
    return torch.sqrt(torch.clamp(x, min=_EPS))


def hex_solve_bands(k_points_np: np.ndarray,
                    g_vectors_np: np.ndarray,
                    eps_grid: torch.Tensor,
                    m_indices_np: np.ndarray,
                    n_bands: int,
                    polarization: str = "te"):
    """Compute photonic band structure on a hexagonal lattice.

    Args:
        k_points_np: (n_k, 2) wavevectors in Cartesian coords.
        g_vectors_np: (n_pw, 2) reciprocal lattice vectors.
        eps_grid: (N, N) dielectric on the oblique grid.
        m_indices_np: (n_pw, 2) integer indices.
        n_bands: number of bands to return.
        polarization: "te" or "tm".

    Returns:
        freqs: (n_k, n_bands) normalized frequencies omega*a/(2*pi*c).
    """
    eps_mat = hex_build_fourier_matrix(eps_grid, m_indices_np)
    eps_mat_inv = torch.linalg.inv(eps_mat)
    return _hex_solve_from_inv(k_points_np, g_vectors_np, eps_mat_inv,
                               n_bands, polarization)


def _hex_solve_from_inv(k_points_np, g_vectors_np, eps_mat_inv,
                        n_bands, polarization="te", return_vecs=False):
    """Band solve from precomputed eps_mat_inv."""
    k_pts = torch.from_numpy(k_points_np).to(torch.float64)
    g_vecs = torch.from_numpy(g_vectors_np).to(torch.float64)

    kpg = k_pts[:, None, :] + g_vecs[None, :, :]  # (n_k, n_pw, 2)

    if polarization == "tm":
        kpg_norm = _safe_sqrt(torch.sum(kpg ** 2, dim=-1))
        H = kpg_norm[:, :, None] * eps_mat_inv.unsqueeze(0) * kpg_norm[:, None, :]
    else:
        dot = torch.sum(kpg[:, :, None, :] * kpg[:, None, :, :], dim=-1)
        H = dot * eps_mat_inv.unsqueeze(0)

    H = 0.5 * (H + H.conj().transpose(-2, -1))
    jitter = torch.eye(H.shape[-1], dtype=H.dtype) * _EPS
    H = H + jitter

    eigvals, eigvecs = torch.linalg.eigh(H)
    eigvals = torch.clamp(eigvals, min=0.0)
    freqs = _safe_sqrt(eigvals[:, :n_bands]) / (2.0 * np.pi)

    if return_vecs:
        return freqs, eigvecs[:, :, :n_bands], kpg
    return freqs


# ---------------------------------------------------------------------------
# 4. Smooth min / max / gap extraction (same as square solver)
# ---------------------------------------------------------------------------

def smooth_min(x: torch.Tensor, beta: float = -50.0):
    bx = torch.clamp(beta * x, min=-80.0, max=80.0)
    w = torch.softmax(bx, dim=0)
    return torch.sum(x * w)


def smooth_max(x: torch.Tensor, beta: float = 50.0):
    bx = torch.clamp(beta * x, min=-80.0, max=80.0)
    w = torch.softmax(bx, dim=0)
    return torch.sum(x * w)


def extract_gap(bands: torch.Tensor, band_lo: int, band_hi: int,
                beta: float = 50.0):
    floor = smooth_max(bands[:, band_lo], beta=beta)
    ceil = smooth_min(bands[:, band_hi], beta=-beta)
    return ceil - floor, 0.5 * (ceil + floor)


def find_best_gap(bands_np: np.ndarray):
    """Find the largest gap across all consecutive band pairs (numpy)."""
    n_bands = bands_np.shape[1]
    best_gw, best_mid, best_pair = 0.0, 0.0, 0
    for n in range(n_bands - 1):
        floor_n = np.max(bands_np[:, n])
        ceil_n = np.min(bands_np[:, n + 1])
        gw = max(0.0, ceil_n - floor_n)
        if gw > best_gw:
            best_gw = gw
            best_mid = 0.5 * (ceil_n + floor_n)
            best_pair = n
    return best_gw, best_mid, best_pair


# ---------------------------------------------------------------------------
# 5. k-path: Gamma -> M -> K -> Gamma
# ---------------------------------------------------------------------------

def hex_make_k_path(n_per_segment=10):
    """BZ path for hexagonal lattice: Gamma -> M -> K -> Gamma.

    High-symmetry points (a = 1):
        Gamma = (0, 0)
        M = (pi, pi/sqrt(3))         -- midpoint of BZ edge
        K = (4*pi/3, 0)              -- BZ corner
    """
    gamma = np.array([0.0, 0.0])
    m_pt = np.array([np.pi, np.pi / _SQRT3])
    k_pt = np.array([4.0 * np.pi / 3.0, 0.0])

    segments = [(gamma, m_pt), (m_pt, k_pt), (k_pt, gamma)]
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
        offset = (float(seg_d[-1]) + float(np.linalg.norm(b - a)) / (n - 1)
                  if n > 1 else offset)

    k_points.append(gamma[None, :])
    k_dist.append(np.array([offset]))

    k_points = np.concatenate(k_points, axis=0)
    k_dist = np.concatenate(k_dist, axis=0)

    tick_positions = [0.0]
    cum = 0.0
    for sl in seg_lengths:
        cum += sl
        tick_positions.append(cum)
    tick_labels = ["$\\Gamma$", "M", "K", "$\\Gamma$"]

    return k_points, k_dist, tick_positions, tick_labels


# ---------------------------------------------------------------------------
# 6. C6v symmetry tiling for the rhombic unit cell
# ---------------------------------------------------------------------------

def _oblique_coords(N):
    """Grid of oblique fractional coordinates (s1, s2) in [0,1)."""
    s = np.arange(N, dtype=np.float64) / N
    s1, s2 = np.meshgrid(s, s, indexing="ij")
    return s1, s2


def _oblique_to_cartesian(s1, s2):
    """Convert oblique (s1, s2) to Cartesian (x, y)."""
    x = s1 * A1[0] + s2 * A2[0]
    y = s1 * A1[1] + s2 * A2[1]
    return x, y


def make_hex_rod_epsilon(N, r_over_a, eps_rod, eps_bg):
    """Circular rod at the origin of the hexagonal unit cell."""
    s1, s2 = _oblique_coords(N)
    x, y = _oblique_to_cartesian(s1, s2)

    cx, cy = 0.5 * (A1[0] + A2[0]), 0.5 * (A1[1] + A2[1])
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    grid = np.where(dist <= r_over_a, eps_rod, eps_bg)
    return torch.tensor(grid, dtype=torch.float64)


def c6v_tile(wedge: torch.Tensor, N: int) -> torch.Tensor:
    """Expand a fundamental domain to a full NxN rhombic grid via C6v.

    The wedge covers the upper-left triangle of one sextant.  We apply
    6-fold rotation and mirrors in oblique coordinates.

    For simplicity, this implementation generates a random C6v-symmetric
    grid by reflecting through 6 symmetry operations in Cartesian space,
    then resampling onto the oblique grid.  The input 'wedge' is a flat
    vector of independent pixel values; the function maps them to a
    C6v-symmetric NxN grid.
    """
    s1, s2 = _oblique_coords(N)
    x, y = _oblique_to_cartesian(s1, s2)

    cx = 0.5 * (A1[0] + A2[0])
    cy = 0.5 * (A1[1] + A2[1])
    dx, dy = x - cx, y - cy

    # Angle from center
    theta = np.arctan2(dy, dx)
    r = np.sqrt(dx ** 2 + dy ** 2)

    # Fold into fundamental domain [0, pi/3) via C6v
    theta_fold = theta % (2 * np.pi)
    theta_fold = theta_fold % (np.pi / 3.0)
    # Mirror: fold [pi/6, pi/3) back to [0, pi/6)
    mirror_mask = theta_fold > (np.pi / 6.0)
    theta_fold[mirror_mask] = np.pi / 3.0 - theta_fold[mirror_mask]

    # Quantize (r, theta_fold) to a 1D index into the wedge
    r_max = max(np.max(r), 1e-10)
    n_r = max(int(np.sqrt(wedge.shape[0])), 1)
    n_th = max(wedge.shape[0] // n_r, 1)
    actual_size = n_r * n_th

    r_idx = np.clip((r / r_max * (n_r - 1)).astype(int), 0, n_r - 1)
    th_idx = np.clip((theta_fold / (np.pi / 6.0) * (n_th - 1)).astype(int),
                     0, n_th - 1)
    flat_idx = r_idx * n_th + th_idx
    flat_idx = np.clip(flat_idx, 0, actual_size - 1)

    flat_idx_t = torch.from_numpy(flat_idx.ravel()).long()
    grid = wedge[:actual_size][flat_idx_t].view(N, N)
    return grid


def make_hex_snowflake_epsilon(N, r_over_a, w_over_a, eps_si=11.7, eps_air=1.0):
    """Snowflake / mercedes-benz air hole in a silicon background.

    Three rounded rectangular arms at 120-degree intervals, centered in the
    hexagonal unit cell.  The background is silicon; the hole is air.

    Args:
        N: grid resolution.
        r_over_a: arm length from center to tip, as fraction of a.
        w_over_a: arm width, as fraction of a.
        eps_si: permittivity of silicon (background).
        eps_air: permittivity of air (hole).
    """
    s1, s2 = _oblique_coords(N)
    x, y = _oblique_to_cartesian(s1, s2)

    cx = 0.5 * (A1[0] + A2[0])
    cy = 0.5 * (A1[1] + A2[1])
    dx, dy = x - cx, y - cy

    hw = w_over_a / 2.0
    hole = np.zeros_like(dx, dtype=bool)

    # Central hub
    dist_center = np.sqrt(dx ** 2 + dy ** 2)
    hole |= dist_center <= hw

    # Three arms at 0, 120, 240 degrees
    for angle_deg in [0, 120, 240]:
        theta = np.radians(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Project onto arm axis (along) and perpendicular (cross)
        along = dx * cos_t + dy * sin_t
        cross = -dx * sin_t + dy * cos_t

        # Rectangular body: along in [0, r], |cross| <= w/2
        rect = (along >= 0) & (along <= r_over_a) & (np.abs(cross) <= hw)
        hole |= rect

        # Rounded tip: semicircle at arm end
        tip_x = r_over_a * cos_t
        tip_y = r_over_a * sin_t
        dist_tip = np.sqrt((dx - tip_x) ** 2 + (dy - tip_y) ** 2)
        hole |= dist_tip <= hw

    grid = np.where(hole, eps_air, eps_si)
    return torch.tensor(grid, dtype=torch.float64)


def generate_c6v_random(N, dtype=torch.float64):
    """Generate a random C6v-symmetric binary grid on the rhombic cell."""
    n_indep = (N * N) // 12 + 1
    wedge = torch.bernoulli(0.5 * torch.ones(n_indep, dtype=dtype))
    return c6v_tile(wedge, N)


def generate_c6v_eps(N, eps_bg, eps_rod):
    """Generate a random C6v-symmetric epsilon grid."""
    s = generate_c6v_random(N)
    return eps_bg + (eps_rod - eps_bg) * s, s
