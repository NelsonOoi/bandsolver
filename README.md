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


References
 - Point of comparison: https://gyptis.gitlab.io/examples/modal/plot_phc2D.html
 