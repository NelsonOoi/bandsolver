"""
Plane Wave Expansion (PWE) solver for 2D photonic crystals on a square lattice.

Units: a = 1, c = 1.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Reciprocal lattice
# ---------------------------------------------------------------------------

def reciprocal_lattice(n_max: int):
    """Return G vectors for a square lattice truncated to |m1|, |m2| <= n_max.

    Returns:
        g_vectors: (n_pw, 2) array of G = (2*pi*m1, 2*pi*m2)
        m_indices: (n_pw, 2) array of integer indices (m1, m2)
    """
    ms = np.arange(-n_max, n_max + 1)
    m1, m2 = np.meshgrid(ms, ms, indexing="ij")
    m_indices = np.stack([m1.ravel(), m2.ravel()], axis=-1)
    g_vectors = 2.0 * np.pi * m_indices
    return g_vectors, m_indices


# ---------------------------------------------------------------------------
# 2. Epsilon matrix from FFT
# ---------------------------------------------------------------------------

def build_epsilon_matrix(eps_grid, m_indices):
    """Build the epsilon Fourier-coefficient matrix from a real-space grid.

    Args:
        eps_grid: (N, N) real-space dielectric constant grid.
        m_indices: (n_pw, 2) integer plane-wave indices.

    Returns:
        eps_mat: (n_pw, n_pw) complex matrix with
                 eps_mat[i,j] = eps_hat(G_i - G_j).
    """
    N = eps_grid.shape[0]
    eps_fft = np.fft.fft2(eps_grid) / (N * N)

    dm = m_indices[:, None, :] - m_indices[None, :, :]  # (n_pw, n_pw, 2)
    idx0 = dm[..., 0] % N
    idx1 = dm[..., 1] % N
    eps_mat = eps_fft[idx0, idx1]
    return eps_mat


# ---------------------------------------------------------------------------
# 3. TM eigenproblem
# ---------------------------------------------------------------------------

def solve_tm(k, g_vectors, eps_mat_inv, n_bands):
    """Solve the TM (E_z) eigenproblem at a single k-point.

    H_TM[i,j] = |k+G_i| * eps_mat_inv[i,j] * |k+G_j|
    """
    kpg = k[None, :] + g_vectors  # (n_pw, 2)
    kpg_norm = np.sqrt(np.sum(kpg ** 2, axis=-1))  # (n_pw,)

    H = kpg_norm[:, None] * eps_mat_inv * kpg_norm[None, :]
    H = 0.5 * (H + H.conj().T)

    eigvals = np.linalg.eigh(H)[0]
    eigvals = np.clip(eigvals, 0.0, None)
    freqs = np.sqrt(eigvals[:n_bands]) / (2.0 * np.pi)
    return freqs


# ---------------------------------------------------------------------------
# 4. TE eigenproblem
# ---------------------------------------------------------------------------

def solve_te(k, g_vectors, eps_mat_inv, n_bands):
    """Solve the TE (H_z) eigenproblem at a single k-point.

    H_TE[i,j] = (k+G_i).(k+G_j) * eps_mat_inv[i,j]
    """
    kpg = k[None, :] + g_vectors  # (n_pw, 2)
    dot_matrix = np.sum(kpg[:, None, :] * kpg[None, :, :], axis=-1)  # (n_pw, n_pw)

    H = dot_matrix * eps_mat_inv
    H = 0.5 * (H + H.conj().T)

    eigvals = np.linalg.eigh(H)[0]
    eigvals = np.clip(eigvals, 0.0, None)
    freqs = np.sqrt(eigvals[:n_bands]) / (2.0 * np.pi)
    return freqs


# ---------------------------------------------------------------------------
# 5. Band structure solver
# ---------------------------------------------------------------------------

def solve_bands(k_points, g_vectors, eps_grid, m_indices, n_bands, polarization="tm"):
    """Compute band structure along a k-path.

    Returns:
        bands: (n_k, n_bands) normalized frequencies.
    """
    eps_mat = build_epsilon_matrix(eps_grid, m_indices)
    eps_mat_inv = np.linalg.inv(eps_mat)

    solver = solve_tm if polarization == "tm" else solve_te

    bands = np.zeros((k_points.shape[0], n_bands))
    for i in range(k_points.shape[0]):
        bands[i] = solver(k_points[i], g_vectors, eps_mat_inv, n_bands)
    return bands


# ---------------------------------------------------------------------------
# 6. k-path utilities
# ---------------------------------------------------------------------------

def make_k_path(n_per_segment=10):
    """Generate k-path along Gamma -> X -> M -> Gamma for a square lattice.

    Returns:
        k_points: (n_total, 2) wavevectors.
        k_dist: (n_total,) cumulative distance along path.
        tick_positions: list of distances at high-symmetry points.
        tick_labels: list of labels.
    """
    gamma = np.array([0.0, 0.0])
    x = np.array([np.pi, 0.0])
    m = np.array([np.pi, np.pi])

    segments = [(gamma, x), (x, m), (m, gamma)]
    seg_lengths = [float(np.linalg.norm(b - a)) for a, b in segments]
    total_length = sum(seg_lengths)

    n_pts = [max(2, int(round(n_per_segment * sl / (total_length / len(segments)))))
             for sl in seg_lengths]

    k_points = []
    k_dist = []
    offset = 0.0

    for (a, b), n in zip(segments, n_pts):
        ts = np.linspace(0.0, 1.0, n, endpoint=False)
        seg_k = a[None, :] + ts[:, None] * (b - a)[None, :]
        seg_d = offset + ts * float(np.linalg.norm(b - a))
        k_points.append(seg_k)
        k_dist.append(seg_d)
        offset = float(seg_d[-1]) + float(np.linalg.norm(b - a)) / (n - 1) if n > 1 else offset

    # Append final Gamma point
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
