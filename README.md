# PWE Band Structure Solver

Plane Wave Expansion solver for 2D photonic crystals on a square lattice. Computes TM and TE band structures with a Tkinter GUI.

## Theory

### Maxwell's equations in periodic media

In a source-free, non-magnetic medium with spatially periodic dielectric function eps(r), Maxwell's equations combine into the master equation for the magnetic field:

$$\nabla \times \left(\frac{1}{\varepsilon(\mathbf{r})} \nabla \times \mathbf{H}\right) = \frac{\omega^2}{c^2} \mathbf{H}$$

By Bloch's theorem, the solutions are labeled by a wavevector **k** in the first Brillouin zone. For each **k**, there is a discrete set of eigenfrequencies omega_n(**k**) -- these are the photonic bands.

### 2D simplification

For a 2D photonic crystal (periodic in xy, uniform in z) with in-plane propagation (k_z = 0), the vector problem decouples into two scalar polarizations:

**TM (E-polarization):** E_z is the only nonzero E component. The wave equation is:

$$-\nabla^2 E_z = \frac{\omega^2}{c^2}\,\varepsilon(\mathbf{r})\,E_z$$

**TE (H-polarization):** H_z is the only nonzero H component. The wave equation is:

$$-\nabla \cdot \left(\frac{1}{\varepsilon(\mathbf{r})}\,\nabla H_z\right) = \frac{\omega^2}{c^2}\,H_z$$

### Plane wave expansion

Expand the field in plane waves of the reciprocal lattice. For a square lattice with constant a, the reciprocal lattice vectors are **G** = 2*pi*(m1, m2)/a. Truncating to |m1|, |m2| <= n_max gives N_pw = (2*n_max + 1)^2 basis functions.

Substituting the Bloch expansion into the wave equations and projecting onto each plane wave gives matrix eigenvalue problems.

**TM:** The Fourier-space equation is:

$$|\mathbf{k}+\mathbf{G}|^2\,e_\mathbf{G} = \frac{\omega^2}{c^2} \sum_{\mathbf{G}'} \hat{\varepsilon}(\mathbf{G}-\mathbf{G}')\,e_{\mathbf{G}'}$$

This is a generalized eigenvalue problem Theta * e = lambda * eps_mat * e, where Theta is diagonal with entries |k+G|^2 and eps_mat is the Toeplitz matrix of Fourier coefficients. To use a standard Hermitian eigensolver, substitute f = Theta^{1/2} e:

$$\Theta^{1/2}\,\boldsymbol{\varepsilon}^{-1}\,\Theta^{1/2}\,\mathbf{f} = \frac{\omega^2}{c^2}\,\mathbf{f}$$

Element-wise: `H_TM[i,j] = |k+G_i| * eps_mat_inv[i,j] * |k+G_j|`.

**TE:** The Fourier-space equation is:

$$\sum_{\mathbf{G}'} (\mathbf{k}+\mathbf{G})\cdot(\mathbf{k}+\mathbf{G}')\;\eta(\mathbf{G},\mathbf{G}')\;h_{\mathbf{G}'} = \frac{\omega^2}{c^2}\,h_\mathbf{G}$$

where eta represents the 1/eps coupling. This is already Hermitian: `H_TE[i,j] = (k+G_i).(k+G_j) * eps_mat_inv[i,j]`.

### The inverse rule

The eta matrix (representing 1/eps in Fourier space) can be computed two ways:

1. **Direct:** Fourier transform 1/eps(r) to get eta_hat(G-G').
2. **Inverse rule:** Build the epsilon matrix from Fourier coefficients of eps(r), then matrix-invert it: eps_mat^{-1}.

Option 2 converges much faster at sharp dielectric interfaces (Ho, Chan & Soukoulis 1990; Li 1996). This solver uses the inverse rule for both polarizations.

### Brillouin zone path

The irreducible Brillouin zone for a square lattice is the triangle Gamma-X-M. Band extrema (and therefore bandgap edges) occur at high-symmetry points, so sweeping k along Gamma(0,0) -> X(pi/a,0) -> M(pi/a,pi/a) -> Gamma(0,0) captures the full band structure.

### Normalized units

All quantities are in dimensionless units with a = 1, c = 1. Frequencies are reported as omega*a/(2*pi*c), which is equivalent to a/lambda. Bandgap quality is measured by the gap-midgap ratio: Delta_omega / omega_mid.

## Implementation

### Steps

1. **Discretize eps(r)** on an NxN grid over the unit cell.
2. **FFT** the grid to get Fourier coefficients eps_hat(G).
3. **Build the epsilon matrix** `eps_mat[i,j] = eps_hat(G_i - G_j)` -- a Toeplitz matrix of Fourier coefficients, truncated to `(2*n_max+1)^2` plane waves.
4. **Invert** the epsilon matrix (the "inverse rule" gives much better convergence at dielectric interfaces than directly Fourier-transforming 1/eps).
5. **Assemble the Hermitian eigenproblem** at each k-point:
   - **TM** (E_z): `H[i,j] = |k+G_i| * eps_mat_inv[i,j] * |k+G_j|`
   - **TE** (H_z): `H[i,j] = (k+G_i).(k+G_j) * eps_mat_inv[i,j]`
6. **Diagonalize** with `numpy.linalg.eigh`. Eigenvalues are (omega/c)^2; normalized frequencies are `sqrt(eig) / (2*pi)`.
7. **Sweep k** along Gamma -> X -> M -> Gamma to trace out the band structure.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `r/a` | Rod radius as fraction of lattice constant |
| `eps rod` | Dielectric constant of rod |
| `eps bg` | Dielectric constant of background |
| `Grid N` | Real-space grid resolution (NxN) |
| `PW n_max` | Plane wave cutoff; gives (2*n_max+1)^2 PWs. Keep 2*n_max < Grid N |
| `Bands` | Number of lowest bands to compute |
| `k/seg` | k-points per high-symmetry segment |

## Usage

```
conda activate iqh
python run.py
```

Adjust parameters in the left panel and click Solve. The GUI shows the dielectric pattern, TM bands, TE bands, and lists any bandgaps with gap/midgap ratios.

### Validation

Default parameters (r/a=0.2, eps=8.9 rods in air) reproduce the classic square-lattice TM bandgap between bands 1-2 (expected gap/midgap ~ 0.31).

## Code structure

### `pwe.py` -- solver library

- `reciprocal_lattice(n_max)` -- generates G vectors `(2*pi*m1, 2*pi*m2)` for all `|m1|, |m2| <= n_max`. Returns both the physical G vectors and integer index pairs, which are used for FFT lookups.

- `build_epsilon_matrix(eps_grid, m_indices)` -- takes the NxN real-space dielectric grid, computes `fft2 / N^2` to get normalized Fourier coefficients, then builds the (n_pw x n_pw) matrix where element `[i,j]` is `eps_fft[(m1_i - m1_j) % N, (m2_i - m2_j) % N]`. The modular indexing maps negative frequency differences to the correct FFT bin.

- `solve_tm(k, g_vectors, eps_mat_inv, n_bands)` -- at a single k-point, computes `|k+G|` for each plane wave, forms the Hermitian matrix `|k+G_i| * eps_mat_inv[i,j] * |k+G_j|`, symmetrizes to kill numerical asymmetry, diagonalizes with `eigh`, clips negative eigenvalues, and returns `sqrt(eig) / 2*pi` as normalized frequencies.

- `solve_te(k, ...)` -- same flow but the matrix is `(k+G_i).(k+G_j) * eps_mat_inv[i,j]` (dot product instead of norm product).

- `solve_bands(k_points, ...)` -- builds the epsilon matrix once, inverts it once, then loops over k-points calling the per-k solver. This is the main entry point.

- `make_k_path(n_per_segment)` -- constructs the Gamma(0,0) -> X(pi,0) -> M(pi,pi) -> Gamma(0,0) path. Points per segment are proportional to segment length. Returns k-points, cumulative distances (for x-axis plotting), and tick positions/labels.

### `run.py` -- GUI application

- `make_rod_epsilon(N, r_over_a, eps_rod, eps_bg)` -- fills an NxN grid: cell centers at `(i+0.5)/N`, marks points within radius `r_over_a` of center (0.5, 0.5) as rod material.

- `find_bandgaps(bands, n_bands)` -- scans consecutive band pairs, reports any where `min(band_{n+1}) > max(band_n)`.

- `PWEApp` -- Tkinter GUI class. Left panel has parameter entries with a live PW count label that updates as you type `n_max`. Clicking Solve spawns a background thread that runs `solve_bands` for both TM and TE, then posts results back to the main thread to update three matplotlib subplots (dielectric map, TM bands, TE bands) and a text box listing detected bandgaps.


## Physics-Informed Neural Network (PINN)

### Overview

`pinn.py` and `pwe_torch.py` add a physics-informed neural network for inverse design of photonic band gaps. Rather than a black-box surrogate, the NN is constrained so its outputs obey the underlying Maxwell equations by construction. The PINN is also accessible from the GUI via the "Train PINN" button.

There are two operating modes:

- **Mode A (inverse design):** The network generates an epsilon grid from a target gap specification, trained end-to-end through a differentiable PWE eigensolve. No pre-computed training data is needed -- gradients flow directly from the physics loss through the solver into the network weights.
- **Mode B (fast surrogate):** The network predicts band frequencies from an epsilon grid, trained against PWE ground truth with additional physics-based loss terms.

### Hard constraints (architectural)

These are enforced by the network structure itself -- they cannot be violated regardless of weight values.

| Constraint | Physical law | How it is enforced |
|---|---|---|
| **C4v crystal symmetry** | Square lattice has 4-fold rotational + mirror symmetry | Network outputs only one octant (upper triangle of one quadrant); `c4v_tile` mirrors and rotates it to fill the full NxN grid. The result is symmetric by construction. |
| **Positive permittivity** | eps(r) > 0 everywhere for passive dielectrics | Sigmoid activation on the decoder output maps values to (0, 1), which is then linearly mapped to (eps_bg, eps_rod). Both bounds are positive. |
| **Reciprocity / time-reversal** | omega(k) = omega(-k) for lossless media | In Mode B, wavevector k enters the network only through even functions: (kx^2, ky^2, kx*ky). This makes the output invariant under k -> -k by construction. In Mode A this is automatically satisfied by the Hermitian eigensolve. |
| **Non-negative frequencies** | omega^2 are eigenvalues of a positive-semidefinite operator | In Mode A, inherited from `torch.linalg.eigh` on a Hermitian matrix with eigenvalue clamping. In Mode B, the output uses `softplus` (always >= 0). |
| **Band ordering** | omega_1 <= omega_2 <= ... <= omega_n | In Mode B, the network predicts non-negative increments (via `softplus`), then `cumsum` produces monotonically ordered frequencies. In Mode A, `eigh` returns sorted eigenvalues. |

### Soft constraints (loss terms)

These are penalty terms in the loss function that push the solution toward physical correctness but can be partially violated during training.

| Loss term | Weight flag | Purpose |
|---|---|---|
| **Gap width MSE** | `--w-gap` | (target_width - actual_width)^2. Drives the gap toward the desired width. |
| **Midgap frequency MSE** | `--w-freq` | (target_freq - actual_midgap)^2. Centers the gap at the desired frequency. |
| **Binary penalty** | `--w-binary` | mean(s * (1 - s)) where s is the normalized epsilon in [0, 1]. Pushes the design toward binary (air or solid) rather than intermediate values, which is needed for fabricability. Ramped from 10% to full value over training. |
| **Bloch periodicity** | `--w-bloch` | MSE between opposite edges of the unit cell. Penalizes designs with boundary discontinuities. (C4v tiling already helps, but this catches residual mismatch.) |
| **Eigenvalue residual** | `--w-residual` | \|\|Hv - lambda v\|\|^2 for eigenpairs. Directly penalizes violation of the eigenvalue equation (Maxwell's equations in Fourier space). |
| **Variational bound** (Mode B) | `--w-variational` | Penalizes predicted frequencies that fall below the PWE ground truth, since the Rayleigh quotient provides an upper bound. |
| **Reciprocity check** (Mode B) | `--w-reciprocity` | MSE between omega(k) and omega(-k) predictions. Redundant with the even-function encoding but adds robustness. |

### Numerical stability

Three guards prevent NaN during training:

1. **Safe sqrt:** `torch.sqrt(clamp(x, min=1e-12))` avoids infinite gradients at the Gamma point where eigenvalues are exactly zero.
2. **Eigenvalue jitter:** A small diagonal perturbation (1e-12 * I) is added to the Hamiltonian before `eigh` to break exact degeneracies, which cause the backward pass to produce NaN (the gradient involves 1/(lambda_i - lambda_j)).
3. **Clamped softmax:** The smooth min/max functions clamp beta*x to [-80, 80] before `exp` to prevent overflow.

If NaN is still detected in gradients, that training step is skipped and the model weights are not updated.

### Adjustable parameters

#### From the GUI (PINN Inverse Design section)

| Parameter | Default | Description |
|---|---|---|
| Objective | Target | **Target**: minimise `(gap - target)^2 + (midgap - target)^2`. **Maximize gap**: maximise the gap-midgap ratio `gap/midgap` (target freq/width ignored). |
| target freq | 0.35 | Desired midgap frequency (a/lambda). Disabled in maximize mode. |
| target width | 0.05 | Desired gap width. Disabled in maximize mode. |
| band lo | 0 | Lower band index for gap (0-based) |
| band hi | 1 | Upper band index for gap |
| steps | 200 | Training iterations |
| lr | 0.001 | Adam learning rate |
| latent dim | 32 | Dimensionality of the encoder's latent space |
| w gap | 1.0 | Weight for gap width loss |
| w freq | 1.0 | Weight for midgap frequency loss |
| w binary | 0.1 | Weight for binarization penalty |
| w bloch | 0.01 | Weight for Bloch periodicity penalty |

The solver parameters (Grid N, PW n_max, Bands, k/seg) and material parameters (eps rod, eps bg) are shared with the PWE solver.

#### CLI-only (pinn.py)

| Flag | Default | Description |
|---|---|---|
| `--mode` | inverse | `inverse` (Mode A) or `surrogate` (Mode B) |
| `--maximize` | off | Flag: maximize gap-midgap ratio instead of targeting specific values |
| `--hidden-dim` | 256 | Hidden layer width for the surrogate network |
| `--n-fourier-features` | 64 | Number of FFT magnitude features for the surrogate encoder |
| `--w-variational` | 0.1 | Variational bound penalty weight (Mode B) |
| `--w-reciprocity` | 0.01 | Reciprocity check weight (Mode B) |

### Usage

```
# GUI (includes PINN section)
conda activate iqh
python run.py

# CLI -- inverse design
python pinn.py --mode inverse --steps 300 --target-freq 0.35 --target-width 0.05

# CLI -- surrogate training
python pinn.py --mode surrogate --steps 2000
```

### Code structure

#### `pwe_torch.py` -- differentiable PWE solver

PyTorch port of `pwe.py` with full autograd support. Every operation (FFT, matrix inverse, `eigh`) propagates gradients back to the epsilon grid. Key additions over the NumPy version:

- `_safe_sqrt` -- numerically stable sqrt with clamped input
- Diagonal jitter on the Hamiltonian before `eigh`
- `smooth_min` / `smooth_max` -- differentiable approximations used in the training loss (not for display)
- `extract_gap` -- differentiable gap extraction for the loss function

#### `pinn.py` -- network and training

- `c4v_tile` / `C4vTiler` -- deterministic C4v symmetry expansion from octant to full grid
- `PhysicsLoss` -- combines all soft constraint terms with configurable weights
- `InverseDesignNet` -- MLP encoder + MLP decoder + C4v tiler + sigmoid + linear eps mapping
- `BandSurrogate` -- Fourier feature encoder + even-k encoder + trunk MLP + softplus + cumsum
- `train_inverse` / `train_surrogate` -- training loops with NaN detection, gradient clipping, cosine LR schedule, and binary penalty warmup

References
 - Point of comparison: https://gyptis.gitlab.io/examples/modal/plot_phc2D.html
 