from compute_bandgap import compute_bandgap

if __name__ == "__main__":
    # Example: Circle geometry with default parameters
    params = {
        "shape": "circle",
        "N": 32,
        "eps_rod": 8.9,
        "eps_bg": 1.0,
        "n_max": 5,
        "n_bands": 10,
        "n_k_seg": 10,
        "shape_kwargs": {"r_over_a": 0.2}
    }
    gap = compute_bandgap(params)
    print(f"First bandgap width: {gap}")
