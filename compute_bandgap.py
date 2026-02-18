import numpy as np
from pwe import reciprocal_lattice, solve_bands, make_k_path
from run import make_epsilon, find_bandgaps

def compute_bandgap(geometry_params):
    """
    Compute the width of the first bandgap for a given geometry.
    geometry_params: dict with keys for shape, N, eps_rod, eps_bg, n_max, n_bands, n_k_seg, shape_kwargs
    Returns: bandgap width (float)
    """
    shape = geometry_params.get("shape", "circle")
    N = geometry_params.get("N", 32)
    eps_rod = geometry_params.get("eps_rod", 8.9)
    eps_bg = geometry_params.get("eps_bg", 1.0)
    n_max = geometry_params.get("n_max", 5)
    n_bands = geometry_params.get("n_bands", 10)
    n_k_seg = geometry_params.get("n_k_seg", 10)
    shape_kwargs = geometry_params.get("shape_kwargs", {})

    eps = make_epsilon(shape, N=N, eps_rod=eps_rod, eps_bg=eps_bg, **shape_kwargs)
    g_vectors, m_indices = reciprocal_lattice(n_max)
    k_points, *_ = make_k_path(n_k_seg)
    bands = solve_bands(k_points, g_vectors, eps, m_indices, n_bands, "tm")
    gaps = find_bandgaps(bands, n_bands)
    if not gaps:
        return 0.0
    # Return the width of the first bandgap
    return gaps[0][2]
