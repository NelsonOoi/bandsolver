"""
FEM-based in-plane elastic phononic band structure solver using DOLFINx + SLEPc.

Bloch-Floquet ansatz: u(x) = ũ(x) exp(i k·x).
Substituting into the elastic wave equation and testing with v*exp(-ik·x)
gives:
    K(k) ũ = ω² M ũ
where
    K(k) = K0 + i·(k_j K1_j) + k_j k_l K2_jl
with K0, K2, M symmetric and K1 antisymmetric (all real).

Since PETSc is compiled with real scalars, we split ũ = ũ_R + i·ũ_I
and solve the equivalent real doubled system:

    [K_s,  k·A ] [ũ_R]       [M  0] [ũ_R]
    [-k·A, K_s ] [ũ_I] = ω²  [0  M] [ũ_I]

where K_s = K0 + k_j k_l K2_jl (symmetric) and A = k_j K1_j (antisymmetric).
This is real symmetric ⟹ standard GHEP in SLEPc.

Each eigenvalue appears with multiplicity 2 (one for each sign of i),
so we only need half the converged eigenvalues.
"""

from __future__ import annotations

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix

from pwe_elastic_torch import SI_PROPS, AIR_PROPS


# ── Mesh + material ────────────────────────────────────────────────


def _build_unit_cell_mesh(res: int):
    return mesh.create_unit_square(
        MPI.COMM_SELF, res, res, cell_type=mesh.CellType.quadrilateral
    )


def _map_mask_to_dg0(domain, mask_grid: np.ndarray, mat_matrix: dict, mat_hole: dict):
    """Nearest-neighbour interpolation of (N,N) mask onto DG0 fields."""
    V_dg = fem.functionspace(domain, ("DG", 0))
    rho_fn = fem.Function(V_dg)
    lam_fn = fem.Function(V_dg)
    mu_fn = fem.Function(V_dg)

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    num_cells = domain.topology.index_map(tdim).size_local

    midpoints = np.zeros((num_cells, 3))
    for c in range(num_cells):
        midpoints[c] = domain.geometry.x[domain.geometry.dofmap[c]].mean(axis=0)

    N = mask_grid.shape[0]
    ix = np.clip((midpoints[:, 0] * N).astype(int), 0, N - 1)
    iy = np.clip((midpoints[:, 1] * N).astype(int), 0, N - 1)
    m = mask_grid[iy, ix].astype(np.float64)

    rho_fn.x.array[:] = mat_matrix["rho"] + (mat_hole["rho"] - mat_matrix["rho"]) * m
    lam_fn.x.array[:] = mat_matrix["lam"] + (mat_hole["lam"] - mat_matrix["lam"]) * m
    mu_fn.x.array[:] = mat_matrix["mu"] + (mat_hole["mu"] - mat_matrix["mu"]) * m
    return rho_fn, lam_fn, mu_fn


# ── Periodic DOF pairing ───────────────────────────────────────────


def _periodic_dof_pairs(V):
    """Match scalar DOFs on opposite boundaries of [0,1]² for a vector CG1 space."""
    bs = V.dofmap.bs
    coords = V.tabulate_dof_coordinates()
    n_nodes = coords.shape[0]
    tol = 1e-8
    pairs = []

    # x-periodicity: x=1 (slave) → x=0 (master), matched by y
    left = {}
    for n in range(n_nodes):
        if abs(coords[n, 0]) < tol:
            left[round(coords[n, 1], 6)] = n
    for n in range(n_nodes):
        if abs(coords[n, 0] - 1.0) < tol:
            key = round(coords[n, 1], 6)
            if key in left:
                for c in range(bs):
                    pairs.append((n * bs + c, left[key] * bs + c))

    # y-periodicity: y=1 (slave) → y=0 (master), matched by x
    bottom = {}
    for n in range(n_nodes):
        if abs(coords[n, 1]) < tol:
            bottom[round(coords[n, 0], 6)] = n
    for n in range(n_nodes):
        if abs(coords[n, 1] - 1.0) < tol:
            key = round(coords[n, 0], 6)
            if key in bottom:
                for c in range(bs):
                    pairs.append((n * bs + c, bottom[key] * bs + c))

    return pairs


def _build_constraint_matrix(pairs, n_full):
    """Rectangular C (n_full × n_free): slave rows point to master columns."""
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

    C = PETSc.Mat().createAIJ(size=(n_full, n_free), comm=PETSc.COMM_SELF)
    C.setUp()
    for i in range(n_full):
        if i in slaves:
            C.setValue(i, free_to_col[slave_to_master[i]], 1.0)
        else:
            C.setValue(i, free_to_col[i], 1.0)
    C.assemble()
    return C, n_free


# ── Bilinear form assembly ─────────────────────────────────────────


def _assemble_operators(domain, rho_fn, lam_fn, mu_fn):
    """Assemble k-independent operators K0, K1x, K1y, K2xx, K2xy, K2yy, M.

    All matrices are real.  K0, K2, M are symmetric; K1 is antisymmetric.
    """
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    I = ufl.Identity(2)

    def C_eps(w):
        return lam_fn * ufl.div(w) * I + 2.0 * mu_fn * ufl.sym(ufl.grad(w))

    def C_sym_outer(w, e):
        s = ufl.sym(ufl.outer(w, e))
        return lam_fn * ufl.tr(s) * I + 2.0 * mu_fn * s

    e = [ufl.as_vector([1, 0]), ufl.as_vector([0, 1])]

    a_K0 = ufl.inner(C_eps(u), ufl.sym(ufl.grad(v))) * ufl.dx

    a_K1 = []
    for j in range(2):
        form = (
            ufl.inner(C_sym_outer(u, e[j]), ufl.sym(ufl.grad(v)))
            - ufl.inner(C_eps(u), ufl.sym(ufl.outer(v, e[j])))
        ) * ufl.dx
        a_K1.append(form)

    a_K2 = {}
    for j in range(2):
        for l in range(j, 2):
            form = ufl.inner(
                C_sym_outer(u, e[l]), ufl.sym(ufl.outer(v, e[j]))
            ) * ufl.dx
            a_K2[(j, l)] = form

    a_M = rho_fn * ufl.inner(u, v) * ufl.dx

    ops = {}
    for name, form in [("K0", a_K0), ("K1x", a_K1[0]), ("K1y", a_K1[1]),
                        ("K2xx", a_K2[(0, 0)]), ("K2xy", a_K2[(0, 1)]),
                        ("K2yy", a_K2[(1, 1)]), ("M", a_M)]:
        compiled = fem.form(form)
        mat = assemble_matrix(compiled)
        mat.assemble()
        ops[name] = mat

    return ops, V


# ── Doubled-system builder ─────────────────────────────────────────


def _build_doubled_system(K_s, A, M_r, n):
    """Build the 2n×2n real symmetric generalised eigenproblem.

    K_big = [K_s,   A  ]    M_big = [M_r,  0 ]
            [-A^T, K_s ]            [ 0,  M_r]

    Since A is antisymmetric (-A^T = A), the top-right and bottom-left
    blocks are consistent and K_big is symmetric.
    """
    K_big = PETSc.Mat().createAIJ(size=(2 * n, 2 * n), comm=PETSc.COMM_SELF)
    K_big.setUp()

    # Fill from K_s and A using getRow
    for i in range(n):
        # K_s block: rows i and n+i
        cols_ks, vals_ks = K_s.getRow(i)
        for c, v in zip(cols_ks, vals_ks):
            K_big.setValue(i, c, v)          # top-left
            K_big.setValue(n + i, n + c, v)  # bottom-right

        # A block: top-right = A, bottom-left = -A^T = A (since A^T = -A)
        cols_a, vals_a = A.getRow(i)
        for c, v in zip(cols_a, vals_a):
            K_big.setValue(i, n + c, v)       # top-right: A
            K_big.setValue(n + c, i, -v)      # bottom-left: -A^T

    K_big.assemble()

    # M_big is block diagonal
    M_big = PETSc.Mat().createAIJ(size=(2 * n, 2 * n), comm=PETSc.COMM_SELF)
    M_big.setUp()
    for i in range(n):
        cols_m, vals_m = M_r.getRow(i)
        for c, v in zip(cols_m, vals_m):
            M_big.setValue(i, c, v)
            M_big.setValue(n + i, n + c, v)
    M_big.assemble()

    return K_big, M_big


# ── Main solver ────────────────────────────────────────────────────


def solve_elastic_bands_fem(
    k_points_np: np.ndarray,
    mask_grid: np.ndarray,
    n_bands: int,
    mat_matrix: dict = None,
    mat_hole: dict = None,
    mesh_res: int = 40,
) -> np.ndarray:
    """Compute in-plane elastic band structure using FEM + SLEPc.

    Args:
        k_points_np: (n_k, 2) wavevectors.
        mask_grid:   (N, N) binary mask, 1=hole 0=matrix.
        n_bands:     number of lowest frequency bands.
        mat_matrix:  dict(rho, lam, mu) for the matrix material.
        mat_hole:    dict(rho, lam, mu) for the hole material.
        mesh_res:    quad elements per side.

    Returns:
        (n_k, n_bands) frequencies f = ω/(2π).
    """
    if mat_matrix is None:
        mat_matrix = SI_PROPS
    if mat_hole is None:
        mat_hole = AIR_PROPS

    domain = _build_unit_cell_mesh(mesh_res)
    rho_fn, lam_fn, mu_fn = _map_mask_to_dg0(domain, mask_grid, mat_matrix, mat_hole)
    ops, V = _assemble_operators(domain, rho_fn, lam_fn, mu_fn)

    # Periodic constraint
    pairs = _periodic_dof_pairs(V)
    n_full = V.dofmap.bs * V.tabulate_dof_coordinates().shape[0]
    C, n_free = _build_constraint_matrix(pairs, n_full)
    Ct = C.copy()
    Ct.transpose()

    def project(op):
        return Ct.matMult(op.matMult(C))

    K0r = project(ops["K0"])
    K1xr = project(ops["K1x"])
    K1yr = project(ops["K1y"])
    K2xxr = project(ops["K2xx"])
    K2xyr = project(ops["K2xy"])
    K2yyr = project(ops["K2yy"])
    Mr = project(ops["M"])

    n_k = len(k_points_np)
    freqs_all = np.zeros((n_k, n_bands))

    # At the Gamma point (k=0), K1 vanishes and K2 vanishes, so the problem
    # reduces to K0 ũ = ω² M ũ (no doubling needed).  For other k-points
    # we use the doubled real system.

    for ik in range(n_k):
        kx, ky = k_points_np[ik]
        k_mag = np.sqrt(kx**2 + ky**2)

        # K_s = K0 + kx² K2xx + 2 kx ky K2xy + ky² K2yy
        K_s = K0r.copy()
        if abs(kx) > 1e-14:
            K_s.axpy(kx * kx, K2xxr)
        if abs(kx * ky) > 1e-14:
            K_s.axpy(2.0 * kx * ky, K2xyr)
        if abs(ky) > 1e-14:
            K_s.axpy(ky * ky, K2yyr)

        if k_mag < 1e-14:
            # Gamma point: simple eigenproblem
            eigvals = _solve_eigenproblem(K_s, Mr, n_bands, n_free)
        else:
            # A = kx K1x + ky K1y  (antisymmetric)
            A = K1xr.copy()
            A.scale(kx)
            A.axpy(ky, K1yr)

            K_big, M_big = _build_doubled_system(K_s, A, Mr, n_free)
            raw_eigvals = _solve_eigenproblem(K_big, M_big, 2 * n_bands + 4, 2 * n_free)
            # Each eigenvalue is doubled — take unique values
            eigvals = _deduplicate(raw_eigvals, n_bands)

            K_big.destroy()
            M_big.destroy()
            A.destroy()

        for b in range(n_bands):
            if b < len(eigvals):
                freqs_all[ik, b] = np.sqrt(max(eigvals[b], 0.0)) / (2.0 * np.pi)

        K_s.destroy()

    for op in list(ops.values()) + [C, Ct, K0r, K1xr, K1yr, K2xxr, K2xyr, K2yyr, Mr]:
        op.destroy()

    return freqs_all


def _solve_eigenproblem(K, M, nev, n):
    """Solve K x = λ M x for smallest nev eigenvalues using SLEPc."""
    eps = SLEPc.EPS().create(PETSc.COMM_SELF)
    eps.setOperators(K, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(0.0)
    eps.setDimensions(nev=min(nev, n - 1))
    eps.setTolerances(tol=1e-10, max_it=1000)

    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)

    eps.solve()

    n_conv = eps.getConverged()
    eigvals = []
    for i in range(n_conv):
        val = eps.getEigenvalue(i).real
        if val > -1e-6:
            eigvals.append(max(val, 0.0))
    eigvals.sort()

    eps.destroy()
    return eigvals


def _deduplicate(eigvals, n_target):
    """Remove duplicate eigenvalues (from the doubling) keeping n_target unique."""
    if not eigvals:
        return []
    unique = [eigvals[0]]
    for v in eigvals[1:]:
        if len(unique) >= n_target:
            break
        if abs(v - unique[-1]) > 1e-6 * max(abs(v), 1.0):
            unique.append(v)
    return unique
