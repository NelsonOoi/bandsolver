Short description:

## 2b. Physics Enforced by the Architecture

Rather than learning arbitrary input–output mappings, the CANN embeds physical and thermodynamic constraints directly into its structure. The table below lists each constraint and the architectural mechanism that enforces it.

| Physical / Thermodynamic Constraint | How It Is Enforced |
|-------------------------------------|-------------------|
| **Bloch's theorem** — material properties are periodic in real space and therefore admit a Fourier (reciprocal-space) representation | LearnedKernelEncoder initializes Conv2D kernels to \(\cos(\mathbf{G}\cdot\mathbf{r})\) and \(\sin(\mathbf{G}\cdot\mathbf{r})\) at reciprocal lattice vectors, computing the discrete structure factor. Kernels remain learnable but start from the physically correct representation. |
| **Constitutive relation / matrix inversion** — the PWE Hamiltonian depends on \(\varepsilon^{-1}\) (photonic) or \(M^{-1}\Gamma\) (phononic), requiring inversion of a material-property matrix | SmoothInverse activation \(x/(x^2+\eta^2)\) approximates \(1/x\) with learnable regularization \(\eta\), mimicking Tikhonov-regularized matrix inversion without numerical instability at small eigenvalues. |
| **Non-negative frequencies** — eigenvalues of Hermitian eigenproblems are real and non-negative; frequencies \(\omega = \sqrt{\lambda}\) must be \(\geq 0\) | Softplus activation \(\ln(1+e^x) > 0\) on the final hidden layer and inside BandOrderingHead ensures all predicted frequencies are strictly positive. |
| **Band ordering** — eigenvalues are conventionally sorted \(\omega_1 \leq \omega_2 \leq \dots\) | BandOrderingHead predicts positive increments via softplus and applies cumulative sum: \(\omega_n = \alpha\sum_{j=1}^n \Delta_j\). Monotonicity is guaranteed by construction — no post-hoc sorting needed. |
| **C4v point-group symmetry** — the square lattice has 4-fold rotational and mirror symmetry; band structures on the \(\Gamma\text{-}X\text{-}M\text{-}\Gamma\) path are invariant under these operations | C4v data augmentation (8× per sample) teaches the network this invariance. The shared encoder processes the grid without explicitly hard-coding symmetry, but the augmented training set ensures equivariant predictions. |
| **Shared material basis** — photonic and phononic properties derive from the same spatial geometry (same inclusion shape determines both \(\varepsilon(\mathbf{r})\) and \(\rho,\lambda,\mu(\mathbf{r})\)) | A single shared LearnedKernelEncoder extracts geometry features used by both physics branches, preventing the two heads from learning inconsistent geometric representations. |
| **Positive-definite material tensors** — physical material properties (\(\varepsilon, \rho, \mu, \lambda+2\mu\)) are strictly positive | Input grids are constructed from physical material constants via linear interpolation on a mask \(\in[0,1]\), guaranteeing all material fields remain within physically valid bounds. |

These constraints reduce the effective hypothesis space of the network, improving data efficiency and ensuring that predictions remain physically plausible even for out-of-distribution geometries.

# CANN v3 — Dual-Physics Constitutive Artificial Neural Networks

## 1. Problem Overview

We seek a fast surrogate model that simultaneously predicts the **photonic** (electromagnetic) and **phononic** (elastic) band structures of 2D periodic crystals from their geometry alone. The ground truth comes from the plane-wave expansion (PWE) method, which requires assembling and diagonalizing large Hermitian matrices for every geometry — a computation too expensive for design-space exploration or inverse design. The CANN replaces this with a single forward pass through a neural network, mapping a dielectric/material grid directly to band frequencies \(\omega_n(\mathbf{k})\) along a high-symmetry path in the Brillouin zone.

## 2. Physics Assumptions

**Geometry.** A 2D square-lattice unit cell (lattice constant \(a = 1\)) composed of a silicon matrix (\(\varepsilon = 8.9\), \(\rho = 2330\) kg/m³, \(\lambda = 68.4\) GPa, \(\mu = 80.0\) GPa) with an air inclusion (\(\varepsilon = 1.0\), \(\rho \approx 0\), \(\lambda \approx 0\), \(\mu \approx 0\)). The inclusion shape is parametrized by Fourier boundary descriptors or standard geometric primitives.

**Photonic (TM polarization).** Maxwell's equations in a periodic dielectric \(\varepsilon(\mathbf{r})\) reduce to the master equation for the out-of-plane electric field \(E_z\):

\[
-\nabla \cdot \left[\frac{1}{\varepsilon(\mathbf{r})} \nabla E_z\right] = \frac{\omega^2}{c^2} E_z
\]

Expanding in plane waves \(E_z = \sum_{\mathbf{G}} c_{\mathbf{G}} \, e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}\) and projecting onto the reciprocal lattice basis yields the Hermitian eigenproblem:

\[
H^{\text{TM}}_{\mathbf{G}\mathbf{G}'}(\mathbf{k}) = |\mathbf{k}+\mathbf{G}| \; [\varepsilon^{-1}]_{\mathbf{G}-\mathbf{G}'} \; |\mathbf{k}+\mathbf{G}'|
\]

\[
H^{\text{TM}} \mathbf{c} = \frac{\omega^2}{c^2} \mathbf{c}
\]

Frequencies are reported as dimensionless \(\omega a / 2\pi c = \sqrt{\lambda_n} / 2\pi\).

**Phononic (in-plane elastic).** The elastic wave equation in a periodic medium:

\[
-\nabla \cdot \boldsymbol{\sigma}(\mathbf{u}) = \omega^2 \rho(\mathbf{r}) \, \mathbf{u}
\]

with stress \(\sigma_{ij} = \lambda \, \delta_{ij} \, \nabla \cdot \mathbf{u} + 2\mu \, \varepsilon_{ij}(\mathbf{u})\). The PWE expansion gives a generalized eigenproblem with \(2 \times 2\) block stiffness matrix:

\[
\boldsymbol{\Gamma}(\mathbf{k}) \, \mathbf{c} = \omega^2 \, \mathbf{M} \, \mathbf{c}
\]

\[
\Gamma^{xx}_{\mathbf{G}\mathbf{G}'} = (k_x+G_x) \, [\widehat{\lambda+2\mu}]_{\mathbf{G}-\mathbf{G}'} \, (k_x'+G_x') + (k_y+G_y) \, [\hat{\mu}]_{\mathbf{G}-\mathbf{G}'} \, (k_y'+G_y')
\]

\[
\Gamma^{yy}_{\mathbf{G}\mathbf{G}'} = (k_y+G_y) \, [\widehat{\lambda+2\mu}]_{\mathbf{G}-\mathbf{G}'} \, (k_y'+G_y') + (k_x+G_x) \, [\hat{\mu}]_{\mathbf{G}-\mathbf{G}'} \, (k_x'+G_x')
\]

\[
\Gamma^{xy}_{\mathbf{G}\mathbf{G}'} = (k_x+G_x) \, [\hat{\lambda}]_{\mathbf{G}-\mathbf{G}'} \, (k_y'+G_y') + (k_y+G_y) \, [\hat{\mu}]_{\mathbf{G}-\mathbf{G}'} \, (k_x'+G_x')
\]

The mass matrix is \(M_{\mathbf{G}\mathbf{G}'} = [\hat{\rho}]_{\mathbf{G}-\mathbf{G}'} \otimes \mathbf{I}_2\). Solved via Cholesky decomposition \(\mathbf{M} = \mathbf{L}\mathbf{L}^H\) to standard form \(\mathbf{L}^{-1} \boldsymbol{\Gamma} \mathbf{L}^{-H} \mathbf{v} = \omega^2 \mathbf{v}\).

**Common PWE pipeline.** Both problems share four steps:

1. **Fourier decomposition** of material fields: \(\varepsilon(\mathbf{r}) \to \hat{\varepsilon}_{\mathbf{G}}\), or \(\rho, \lambda, \mu \to \hat{\rho}_{\mathbf{G}}, \hat{\lambda}_{\mathbf{G}}, \hat{\mu}_{\mathbf{G}}\)
2. **Matrix assembly** from Fourier coefficients and wavevectors \(\mathbf{k}+\mathbf{G}\)
3. **Matrix inversion** (\(\varepsilon^{-1}\) for photonic; Cholesky \(M^{-1}\) for phononic)
4. **Eigenvalue solve** \(\to\) ordered frequencies \(\omega_1 \leq \omega_2 \leq \dots\)

The CANN architecture mirrors this: LearnedKernelEncoder \(\to\) step 1, SmoothInverse activation \(\to\) step 3, BandOrderingHead \(\to\) step 4.

**Additional assumptions.** Plane-wave truncation at \(n_\text{max} = 3\) (49 plane waves). Bands evaluated on the irreducible Brillouin zone path \(\Gamma \to X \to M \to \Gamma\) with 10 points per segment (\(N_k = 31\)). C4v symmetry of the square lattice is exploited for data augmentation.

## 3. Model Architecture & Tools

**Language & framework.** Python 3.11, PyTorch 2.x (autograd, `nn.Module`), NumPy, Matplotlib. Visualization with `torchview` + Graphviz.

### 3.1 LearnedKernelEncoder

Full-size \(G \times G\) Conv2D kernels initialized to Fourier modes at reciprocal lattice \(\mathbf{G}\)-vectors:

\[
f_0 = \sum_{i,j} \varepsilon_{ij} \cdot 1 + b_0 \quad \text{(DC component)}
\]
\[
f_{2m-1} = \sum_{i,j} \varepsilon_{ij} \cos(\mathbf{G}_m \cdot \mathbf{r}_{ij}) + b_{2m-1}, \quad m = 1, \dots, N_G
\]
\[
f_{2m} = \sum_{i,j} \varepsilon_{ij} \sin(\mathbf{G}_m \cdot \mathbf{r}_{ij}) + b_{2m}
\]

At initialization these equal the structure factor coefficients \(\hat{\varepsilon}(\mathbf{G})\). Kernels and biases are learnable. Output: \(\mathbf{f} \in \mathbb{R}^{2N_G + 1}\) (97-d for \(n_\text{max}=3\)).

### 3.2 GenericCANN Backbone

Fully connected backbone with physics-inspired activations:

| Layer | Operation | Output |
|-------|-----------|--------|
| FC1 | \(\mathbf{h} = \text{SiLU}(\mathbf{W}_1 \mathbf{f} + \mathbf{b}_1)\) | \(\mathbb{R}^{128}\) |
| FC2 | \(\mathbf{h} = \text{SmoothInverse}(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2)\) | \(\mathbb{R}^{128}\) |
| FC3 | \(\mathbf{h} = \text{Softplus}(\mathbf{W}_3 \mathbf{h} + \mathbf{b}_3)\) | \(\mathbb{R}^{128}\) |
| Out | \(\mathbf{z} = \mathbf{W}_\text{out} \mathbf{h} + \mathbf{b}_\text{out}\) | \(\mathbb{R}^{N_k \times N_b}\) |
| Head | \(\boldsymbol{\omega} = \text{BandOrderingHead}(\mathbf{z})\) | \(\mathbb{R}^{N_k \times N_b}\) |

**Activation functions:**

| Activation | Expression | Physics Motivation |
|-----------|------------|-------------------|
| SiLU | \(\text{SiLU}(x) = x \cdot \sigma(x)\) | Smooth, non-monotonic; good gradient flow |
| SmoothInverse | \(\displaystyle\frac{x_i}{x_i^2 + \eta_i^2}, \quad \eta_i = e^{\lambda_i}\) | Approximates the matrix inversion \(\varepsilon^{-1}\) in PWE; learnable regularization prevents division by zero |
| Softplus | \(\ln(1 + e^x)\) | Ensures positive values, analogous to eigenvalues being non-negative |
| Sigmoid | \(1/(1 + e^{-x})\) | Gating in \([0, 1]\) for cross-attention |

**BandOrderingHead** enforces monotonic band ordering via cumulative softplus:

\[
\Delta_n = \text{softplus}(z_n), \qquad \omega_n = \alpha \sum_{j=1}^{n} \Delta_j
\]

where \(\alpha\) is a learnable frequency scale. This guarantees \(\omega_1 \leq \omega_2 \leq \dots \leq \omega_N\) by construction, matching the physical ordering of eigenvalues.

### 3.3 Architecture A: DualGridCANN

Shared LearnedKernelEncoder with two **independent** GenericCANN backbones:

\[
\varepsilon \xrightarrow{\text{Encoder}} \mathbf{f} \xrightarrow{\text{Backbone}^p} \boldsymbol{\omega}^{\text{phot}}, \quad \mathbf{f} \xrightarrow{\text{Backbone}^n} \boldsymbol{\omega}^{\text{phon}}
\]

### 3.4 Architecture B: DualGridCANN_cross

Shared encoder with **cross-gated** backbones. After the SmoothInverse layer, a CrossAttentionBlock exchanges information:

\[
\mathbf{g}_p = \sigma(\mathbf{W}_g^p \, \mathbf{h}_n), \qquad \mathbf{g}_n = \sigma(\mathbf{W}_g^n \, \mathbf{h}_p)
\]
\[
\mathbf{h}_p' = \mathbf{h}_p + \mathbf{g}_p \odot (\mathbf{V}^p \, \mathbf{h}_n), \qquad \mathbf{h}_n' = \mathbf{h}_n + \mathbf{g}_n \odot (\mathbf{V}^n \, \mathbf{h}_p)
\]

The residual connection allows the model to learn to ignore cross-physics coupling when not useful. Full forward pass:

| Step | Operation | Output |
|------|-----------|--------|
| 1 | Grid prep | \((B, 1, 32, 32)\) |
| 2 | LearnedKernelEncoder | \((B, 97)\) |
| 3 | FC1 + SiLU (per branch) | \((B, 128)\) each |
| 4 | FC2 + SmoothInverse (per branch) | \((B, 128)\) each |
| 5 | **CrossAttentionBlock** | \((B, 128)\) each |
| 6 | FC3 + Softplus (per branch) | \((B, 128)\) each |
| 7 | Linear output | \((B, N_k \cdot N_b)\) each |
| 8 | BandOrderingHead | \((B, N_k, N_b)\) each |

## 4. Loss Function

Joint normalized MSE with a mixing coefficient \(\alpha\):

\[
\mathcal{L} = \alpha \, \mathcal{L}_{\text{phot}} + (1 - \alpha) \, \mathcal{L}_{\text{phon}}
\]

where each term is the mean squared error on normalized targets:

\[
\mathcal{L}_{\text{phot}} = \frac{1}{B \, N_k \, N_b^p} \sum_{i,j,n} \left(\frac{\hat{\omega}^p_{ijn}}{\omega^p_\text{max}} - \frac{\omega^p_{ijn}}{\omega^p_\text{max}}\right)^2
\]

\[
\mathcal{L}_{\text{phon}} = \frac{1}{B \, N_k \, N_b^n} \sum_{i,j,n} \left(\frac{\hat{\omega}^n_{ijn}}{\omega^n_\text{max}} - \frac{\omega^n_{ijn}}{\omega^n_\text{max}}\right)^2
\]

Normalization by \(\omega_\text{max}\) (the maximum frequency in the training set for each physics) ensures the two losses are on comparable scales despite photonic frequencies being \(\mathcal{O}(0.1)\) and phononic frequencies being \(\mathcal{O}(10^9)\).

Default: \(\alpha = 0.5\) (equal weighting). No explicit regularization term; implicit regularization through gradient clipping (\(\|\nabla\|_\text{max} = 1.0\)) and cosine-annealed learning rate (\(10^{-3} \to 10^{-5}\) over 5000 epochs).

## 5. Data

**Generation.** Training data is generated on-the-fly by solving both PWE eigenproblems for each geometry. No external dataset is used.

**Geometry sampling.** Two families of shapes, all rasterized on a \(64 \times 64\) grid (downsampled to \(32 \times 32\) by the encoder):

1. **Perturbed circles** (60 train, 10 test): Fourier boundary \(r(\theta) = a_0 + \sum_{n=1}^{4}(a_n \cos n\theta + b_n \sin n\theta)\) with \(a_0 \in [0.12, 0.38]\), perturbation scale \(\sigma = 0.08\) (train) / \(0.06\) (test). Samples with \(r < 0\) or \(r > 0.5\) are rejected.
2. **Standard shapes** (90 train, 10 test): randomly sized crosses, squares, ellipses, and rings with randomized dimensions.

**Material mapping.** Each grid pixel maps shape mask \(\in [0, 1]\) to material properties:

| Property | Si matrix (mask=0) | Air hole (mask=1) |
|----------|-------------------|-------------------|
| \(\varepsilon\) | 8.9 | 1.0 |
| \(\rho\) | 2330 kg/m³ | 1.225 kg/m³ |
| \(\lambda\) | 68.4 GPa | \(\approx 0\) |
| \(\mu\) | 80.0 GPa | \(\approx 0\) |

**Augmentation.** C4v symmetry of the square lattice provides an 8× augmentation (4 rotations × 2 reflections) since bands on the \(\Gamma\text{-}X\text{-}M\text{-}\Gamma\) path are invariant under these operations. Final training set: \(150 \times 8 = 1200\) samples.

**Targets.** Per geometry: \(N_k = 31\) k-points, \(N_b^p = 6\) photonic bands, \(N_b^n = 10\) phononic bands. Output tensors are \((B, 31, 6)\) and \((B, 31, 10)\).

## 6. Planned Analyses

**Architecture comparison.** Three models trained under identical conditions (5000 epochs, Adam, cosine LR):

| Model | Description |
|-------|-------------|
| DualGridCANN | Shared encoder, independent backbones |
| DualGridCANN_cross | Shared encoder, cross-gated backbones |
| GridCANN (baseline) | Photonic-only single backbone |

**Ablations and studies:**

- **Cross-physics transfer**: does the cross-gated model improve photonic accuracy by leveraging phononic information (or vice versa)?
- **\(\alpha\) sensitivity**: vary \(\alpha \in \{0.3, 0.5, 0.7\}\) to assess the effect of loss weighting on each physics.
- **Shared vs. separate encoders**: compare shared LearnedKernelEncoder against independent encoders per physics.
- **Training set size**: learning curves at 50%, 75%, 100% of data.
- **Generalization to unseen shapes**: test on shape families not in training (e.g., train on circles, test on crosses).

## 7. Evaluation Metrics

**Root Mean Squared Error (RMSE).** Primary metric, computed on unnormalized frequencies:

\[
\text{RMSE} = \sqrt{\frac{1}{N_\text{test} \, N_k \, N_b} \sum_{i=1}^{N_\text{test}} \sum_{j=1}^{N_k} \sum_{n=1}^{N_b} \left(\hat{\omega}_{ijn} - \omega_{ijn}\right)^2}
\]

RMSE penalizes large errors more heavily than MAE and is in the same units as the target frequencies, making it directly interpretable. Reported separately for photonic (\(\omega a / 2\pi c\)) and phononic (rad/s · \(a\)) bands.

**Coefficient of Determination (\(R^2\)).** Measures the fraction of variance explained:

\[
R^2 = 1 - \frac{\sum_{i,j,n} (\hat{\omega}_{ijn} - \omega_{ijn})^2}{\sum_{i,j,n} (\omega_{ijn} - \bar{\omega})^2}
\]

where \(\bar{\omega}\) is the mean over all test samples, k-points, and bands. \(R^2 = 1\) is a perfect fit; \(R^2 = 0\) means the model is no better than predicting the mean. Computed per-physics.

**Qualitative: band diagram overlay.** Side-by-side plots of predicted (dashed red) vs. PWE ground truth (solid black) band structures for selected test geometries, providing visual assessment of band crossing accuracy and gap prediction.
