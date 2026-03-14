"""
Fast FEM in-plane elastic phononic band structure solver.

Pure numpy/scipy — no DOLFINx, PETSc, or gmsh required.
Delaunay meshing + vectorized P1 triangle assembly.

For void inclusions (air holes), meshes only the solid region with
traction-free (natural) BCs at hole boundaries, avoiding ill-conditioning
from extreme material contrast.

Bloch-Floquet ansatz: u(x) = ũ(x) exp(i k·x)
=> K(k) ũ = ω² M ũ,  K(k) = K0 + i k_j K1_j + k_j k_l K2_jl

API-compatible with pwe_elastic_fem.solve_elastic_bands_fem.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh as dense_eigh
from scipy.spatial import Delaunay

from pwe_elastic_torch import SI_PROPS, AIR_PROPS


# ── Meshing ────────────────────────────────────────────────────────────


def _generate_mesh(mask_grid: np.ndarray, mesh_res: int,
                   is_void: bool) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate the solid region of [0,1]².

    For void holes: removes nodes inside holes and triangles whose
    centroids fall inside holes.
    For solid-solid: meshes the full unit cell.
    """
    N = mask_grid.shape[0]
    xs = np.linspace(0, 1, mesh_res + 1)
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    if is_void:
        ix = np.clip((coords[:, 0] * N).astype(int), 0, N - 1)
        iy = np.clip((coords[:, 1] * N).astype(int), 0, N - 1)
        in_solid = mask_grid[iy, ix] < 0.5
        on_boundary = ((np.abs(coords[:, 0]) < 1e-10) |
                       (np.abs(coords[:, 0] - 1) < 1e-10) |
                       (np.abs(coords[:, 1]) < 1e-10) |
                       (np.abs(coords[:, 1] - 1) < 1e-10))
        coords = coords[in_solid | on_boundary]

    tri = Delaunay(coords)
    triangles = tri.simplices

    if is_void:
        centroids = coords[triangles].mean(axis=1)
        cx = np.clip((centroids[:, 0] * N).astype(int), 0, N - 1)
        cy = np.clip((centroids[:, 1] * N).astype(int), 0, N - 1)
        triangles = triangles[mask_grid[cy, cx] < 0.5]

    return coords, triangles


# ── Bloch bilinear form coefficient tables ─────────────────────────────
#
# For 2D isotropic elasticity C_ijkl = lam*d_ij*d_kl + mu*(d_ik*d_jl + d_il*d_jk)
#
# K0:  standard stiffness (gradient × gradient)
# K1j: first-order Bloch (gradient × identity, antisymmetric part)
#      K1_j = int [ C:sym(u⊗e_j) : sym(grad v) - C:sym(grad u) : sym(v⊗e_j) ] dA
# K2jl: second-order Bloch (identity × identity)
#      K2_jl = int C:sym(u⊗e_l) : sym(v⊗e_j) dA
# M:   mass matrix


def _precompute_bloch_coefficients():
    """Precompute constant coefficient tables for K1 and K2 assembly.

    K2 coefficients: for inner(C:sym(e_p⊗e_j), sym(e_q⊗e_l))
      = lam * tr(sym(e_p⊗e_j)) * tr(sym(e_q⊗e_l)) + 2*mu * sym(e_p⊗e_j) : sym(e_q⊗e_l)
      Stored as (lam_coeff, mu_coeff) per (p,j,q,l).

    K1 coefficients: term1 and term2 are linear in shape function gradients.
      term1(p,q,j) = inner(C:sym(e_q⊗e_j), sym(dNa⊗e_p)) = sum of lam*(...) + mu*(...)
      term2(p,q,j) = inner(C:sym(dNb⊗e_q), sym(e_p⊗e_j)) = sum of lam*(...) + mu*(...)
      Each is linear in (dN_x, dN_y), stored as (lam_cx, lam_cy, mu_cx, mu_cy).
    """
    k2 = np.zeros((2, 2, 2, 2, 2))
    for p in range(2):
        for j in range(2):
            A = np.zeros((2, 2))
            A[p, j] += 0.5; A[j, p] += 0.5
            for q in range(2):
                for l in range(2):
                    B = np.zeros((2, 2))
                    B[q, l] += 0.5; B[l, q] += 0.5
                    k2[p, j, q, l, 0] = A.trace() * B.trace()
                    k2[p, j, q, l, 1] = 2.0 * np.sum(A * B)

    # K1 term1: inner(C:sym(e_q⊗e_j), sym(dNa⊗e_p))
    # = sigma[r,p]*dNa_r where sigma = C:sym(e_q⊗e_j)
    # lam part: lam * tr(A) * I[r,p] * dNa_r = lam * trA * dNa_p
    # mu part:  2*mu * A[r,p] * dNa_r
    k1_t1_lam = np.zeros((2, 2, 2, 2))
    k1_t1_mu = np.zeros((2, 2, 2, 2))
    # K1 term2: inner(C:sym(dNb⊗e_q), sym(e_p⊗e_j))
    # lam part: lam * dNb_q * trB
    # mu part:  2*mu * B[r,q] * dNb_r
    k1_t2_lam = np.zeros((2, 2, 2, 2))
    k1_t2_mu = np.zeros((2, 2, 2, 2))

    for p in range(2):
        for q in range(2):
            for j in range(2):
                A = np.zeros((2, 2))
                A[q, j] += 0.5; A[j, q] += 0.5
                k1_t1_lam[p, q, j, p] += A.trace()
                k1_t1_mu[p, q, j, 0] = 2.0 * A[0, p]
                k1_t1_mu[p, q, j, 1] = 2.0 * A[1, p]

                B = np.zeros((2, 2))
                B[p, j] += 0.5; B[j, p] += 0.5
                k1_t2_lam[p, q, j, q] += B.trace()
                k1_t2_mu[p, q, j, 0] = 2.0 * B[0, q]
                k1_t2_mu[p, q, j, 1] = 2.0 * B[1, q]

    return k2, k1_t1_lam, k1_t1_mu, k1_t2_lam, k1_t2_mu


# ── Vectorized P1 assembly ─────────────────────────────────────────────


def _assemble_all(coords: np.ndarray, triangles: np.ndarray,
                  rho: np.ndarray, lam: np.ndarray, mu: np.ndarray):
    """Assemble K0, K1x, K1y, K2xx, K2xy, K2yy, M as sparse CSR.

    DOF ordering: interleaved [u0x, u0y, u1x, u1y, ...].
    rho, lam, mu: per-element arrays (n_elem,).
    """
    n_nodes = len(coords)
    n_dof = 2 * n_nodes

    x1 = coords[triangles[:, 0]]
    x2 = coords[triangles[:, 1]]
    x3 = coords[triangles[:, 2]]

    det_J = ((x2[:, 0] - x1[:, 0]) * (x3[:, 1] - x1[:, 1]) -
             (x2[:, 1] - x1[:, 1]) * (x3[:, 0] - x1[:, 0]))
    area = 0.5 * np.abs(det_J)
    inv_det = 1.0 / det_J

    dN = np.zeros((len(triangles), 3, 2))
    dN[:, 0, 0] = (x2[:, 1] - x3[:, 1]) * inv_det
    dN[:, 0, 1] = (x3[:, 0] - x2[:, 0]) * inv_det
    dN[:, 1, 0] = (x3[:, 1] - x1[:, 1]) * inv_det
    dN[:, 1, 1] = (x1[:, 0] - x3[:, 0]) * inv_det
    dN[:, 2, 0] = (x1[:, 1] - x2[:, 1]) * inv_det
    dN[:, 2, 1] = (x2[:, 0] - x1[:, 0]) * inv_det

    lam2mu = lam + 2.0 * mu
    k2_c, k1_t1l, k1_t1m, k1_t2l, k1_t2m = _precompute_bloch_coefficients()

    def _sparse(r, c, v):
        return coo_matrix((v, (r, c)), shape=(n_dof, n_dof)).tocsr()

    R = {k: [] for k in ["K0", "K1x", "K1y", "K2xx", "K2xy", "K2yy", "M"]}
    C = {k: [] for k in R}
    V = {k: [] for k in R}

    def _add(name, r, c, v):
        R[name].append(r); C[name].append(c); V[name].append(v)

    for a in range(3):
        dNax, dNay = dN[:, a, 0], dN[:, a, 1]
        for b in range(3):
            dNbx, dNby = dN[:, b, 0], dN[:, b, 1]
            ia, ib = 2 * triangles[:, a], 2 * triangles[:, b]

            # K0
            _add("K0", ia,   ib,   (lam2mu*dNax*dNbx + mu*dNay*dNby) * area)
            _add("K0", ia,   ib+1, (lam*dNax*dNby + mu*dNay*dNbx) * area)
            _add("K0", ia+1, ib,   (lam*dNay*dNbx + mu*dNax*dNby) * area)
            _add("K0", ia+1, ib+1, (mu*dNax*dNbx + lam2mu*dNay*dNby) * area)

            # K1x, K1y
            a3 = area / 3.0
            for ji, nm in enumerate(["K1x", "K1y"]):
                for p in range(2):
                    for q in range(2):
                        t1 = (lam*(k1_t1l[p,q,ji,0]*dNax + k1_t1l[p,q,ji,1]*dNay) +
                              mu *(k1_t1m[p,q,ji,0]*dNax + k1_t1m[p,q,ji,1]*dNay))
                        t2 = (lam*(k1_t2l[p,q,ji,0]*dNbx + k1_t2l[p,q,ji,1]*dNby) +
                              mu *(k1_t2m[p,q,ji,0]*dNbx + k1_t2m[p,q,ji,1]*dNby))
                        _add(nm, ia+p, ib+q, (t1 - t2) * a3)

            # K2xx, K2xy, K2yy
            NaNb = np.where(a == b, area / 6.0, area / 12.0)
            for (jj, ll), nm in [((0,0),"K2xx"), ((0,1),"K2xy"), ((1,1),"K2yy")]:
                for p in range(2):
                    for q in range(2):
                        cl, cm = k2_c[p,jj,q,ll,0], k2_c[p,jj,q,ll,1]
                        if abs(cl) < 1e-15 and abs(cm) < 1e-15:
                            continue
                        _add(nm, ia+p, ib+q, (lam*cl + mu*cm) * NaNb)

            # M
            NaNb_m = np.where(a == b, area / 6.0, area / 12.0)
            _add("M", ia, ib, rho * NaNb_m)
            _add("M", ia+1, ib+1, rho * NaNb_m)

    return {nm: _sparse(np.concatenate(R[nm]), np.concatenate(C[nm]),
                        np.concatenate(V[nm])) for nm in R}


# ── Periodic constraint ────────────────────────────────────────────────


def _periodic_dof_pairs(coords: np.ndarray):
    """Match DOFs on opposite boundaries of [0,1]²."""
    n = len(coords)
    tol = 1e-6
    pairs = []

    left = {}
    for i in range(n):
        if abs(coords[i, 0]) < tol:
            left[round(coords[i, 1], 5)] = i
    for i in range(n):
        if abs(coords[i, 0] - 1.0) < tol:
            k = round(coords[i, 1], 5)
            if k in left:
                for c in range(2):
                    pairs.append((i*2+c, left[k]*2+c))

    bottom = {}
    for i in range(n):
        if abs(coords[i, 1]) < tol:
            bottom[round(coords[i, 0], 5)] = i
    for i in range(n):
        if abs(coords[i, 1] - 1.0) < tol:
            k = round(coords[i, 0], 5)
            if k in bottom:
                for c in range(2):
                    pairs.append((i*2+c, bottom[k]*2+c))

    return pairs


def _build_constraint_csr(pairs, n_full):
    """Sparse C (n_full x n_free) for periodic DOF elimination."""
    slave_to_master = {}
    for s, m in pairs:
        if s != m:
            target = m
            while target in slave_to_master:
                target = slave_to_master[target]
            slave_to_master[s] = target

    slaves = set(slave_to_master)
    free = sorted(set(range(n_full)) - slaves)
    free_to_col = {d: i for i, d in enumerate(free)}
    n_free = len(free)

    rows, cols, vals = [], [], []
    for i in range(n_full):
        if i in slaves:
            rows.append(i)
            cols.append(free_to_col[slave_to_master[i]])
            vals.append(1.0)
        else:
            rows.append(i)
            cols.append(free_to_col[i])
            vals.append(1.0)

    return csr_matrix((vals, (rows, cols)), shape=(n_full, n_free)), n_free


# ── Main solver ────────────────────────────────────────────────────────


def solve_elastic_bands_fem(
    k_points_np: np.ndarray,
    mask_grid: np.ndarray,
    n_bands: int,
    mat_matrix: dict = None,
    mat_hole: dict = None,
    mesh_res: int = 40,
) -> np.ndarray:
    """Compute in-plane elastic band structure using fast FEM.

    For void inclusions (mu_hole/mu_matrix < 1e-4): meshes only solid,
    traction-free BCs at hole boundaries.
    For solid-solid: meshes full unit cell with per-element properties.

    Returns (n_k, n_bands) frequencies f = omega/(2*pi).
    """
    if mat_matrix is None:
        mat_matrix = SI_PROPS
    if mat_hole is None:
        mat_hole = AIR_PROPS

    is_void = (mat_hole["mu"] / mat_matrix["mu"]) < 1e-4

    coords, triangles = _generate_mesh(mask_grid, mesh_res, is_void)

    N = mask_grid.shape[0]
    centroids = coords[triangles].mean(axis=1)
    cx = np.clip((centroids[:, 0] * N).astype(int), 0, N - 1)
    cy = np.clip((centroids[:, 1] * N).astype(int), 0, N - 1)
    m = mask_grid[cy, cx].astype(np.float64)

    rho_e = mat_matrix["rho"] + (mat_hole["rho"] - mat_matrix["rho"]) * m
    lam_e = mat_matrix["lam"] + (mat_hole["lam"] - mat_matrix["lam"]) * m
    mu_e = mat_matrix["mu"] + (mat_hole["mu"] - mat_matrix["mu"]) * m

    ops = _assemble_all(coords, triangles, rho_e, lam_e, mu_e)

    n_dof = 2 * len(coords)
    pairs = _periodic_dof_pairs(coords)
    Cmat, n_free = _build_constraint_csr(pairs, n_dof)
    Ct = Cmat.T

    red = {name: Ct @ op @ Cmat for name, op in ops.items()}

    n_k = len(k_points_np)
    freqs_all = np.zeros((n_k, n_bands))
    M_r = red["M"]

    # Strategy: for small systems use dense batched eigvalsh (Cholesky-reduced),
    # for large systems use sparse eigsh per k-point.
    if n_free <= 1200:
        # Dense path: Cholesky-reduce M once, batch all k-points
        M_d = M_r.toarray().astype(np.float64)
        # Regularize: ensure positive-definite even for near-void geometries
        min_diag = np.min(np.diag(M_d))
        if min_diag < 1e-12:
            M_d += np.eye(M_d.shape[0]) * (1e-12 - min(min_diag, 0.0))
        L = np.linalg.cholesky(M_d)
        L_inv = np.linalg.inv(L)

        def _tx(sp):
            return L_inv @ sp.toarray().astype(np.float64) @ L_inv.T

        K0   = _tx(red["K0"]);   K0   = 0.5*(K0   + K0.T)
        K1x  = _tx(red["K1x"]);  K1x  = 0.5*(K1x  - K1x.T)
        K1y  = _tx(red["K1y"]);  K1y  = 0.5*(K1y  - K1y.T)
        K2xx = _tx(red["K2xx"]); K2xx = 0.5*(K2xx + K2xx.T)
        K2xy = _tx(red["K2xy"]); K2xy = 0.5*(K2xy + K2xy.T)
        K2yy = _tx(red["K2yy"]); K2yy = 0.5*(K2yy + K2yy.T)

        kx = k_points_np[:, 0]
        ky = k_points_np[:, 1]

        H_real = (K0[None]
                  + kx[:, None, None]**2 * K2xx[None]
                  + ky[:, None, None]**2 * K2yy[None]
                  + (2.0*kx*ky)[:, None, None] * K2xy[None])
        H_imag = kx[:, None, None]*K1x[None] + ky[:, None, None]*K1y[None]

        if np.max(np.abs(H_imag)) > 1e-14 * np.max(np.abs(H_real)):
            ev = np.linalg.eigvalsh(H_real + 1j*H_imag).real
        else:
            ev = np.linalg.eigvalsh(H_real)

        ev = np.maximum(ev, 0.0)
        freqs_all = np.sqrt(ev[:, :n_bands]) / (2.0 * np.pi)
    else:
        # Sparse path: eigsh per k-point with shift-invert
        M_sp = M_r.astype(complex)
        nev = min(n_bands + 20, n_free - 1)
        for ik in range(n_k):
            kx, ky = k_points_np[ik]
            K = red["K0"].copy().astype(complex)
            if abs(kx) > 1e-14:
                K += 1j * kx * red["K1x"]
                K += kx * kx * red["K2xx"]
            if abs(ky) > 1e-14:
                K += 1j * ky * red["K1y"]
                K += ky * ky * red["K2yy"]
            if abs(kx * ky) > 1e-14:
                K += 2.0 * kx * ky * red["K2xy"]
            K = 0.5 * (K + K.conj().T)
            try:
                eigvals, _ = eigsh(K, k=nev, M=M_sp, sigma=0.0, which="LM")
                eigvals = np.sort(np.maximum(eigvals.real, 0.0))
            except Exception:
                eigvals = np.zeros(nev)
            nb = min(n_bands, len(eigvals))
            freqs_all[ik, :nb] = np.sqrt(eigvals[:nb]) / (2.0 * np.pi)

    return freqs_all
