import csv
import random
from compute_bandgap import compute_bandgap

# Parameter ranges for random sampling
def random_params():
    return {
        "shape": "circle",
        "N": 32,
        "eps_rod": random.uniform(2.0, 12.0),
        "eps_bg": random.uniform(1.0, 3.0),
        "n_max": 5,
        "n_bands": 10,
        "n_k_seg": 10,
        "shape_kwargs": {"r_over_a": random.uniform(0.1, 0.4)}
    }

samples = []
for _ in range(30):  # Generate 30 samples
    params = random_params()
    gap = compute_bandgap(params)
    row = {
        "eps_rod": params["eps_rod"],
        "eps_bg": params["eps_bg"],
        "r_over_a": params["shape_kwargs"]["r_over_a"],
        "bandgap": gap
    }
    samples.append(row)

with open("bandgap_dataset.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["eps_rod", "eps_bg", "r_over_a", "bandgap"])
    writer.writeheader()
    writer.writerows(samples)

print("Saved bandgap_dataset.csv with 30 samples.")
