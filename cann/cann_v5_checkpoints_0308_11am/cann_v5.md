# CANN v5 — Per-K Conditioned Dual-Physics Constitutive Artificial Neural Networks

## 1. Problem Overview

We build a neural network surrogate that simultaneously predicts the **photonic** (electromagnetic) and **phononic** (elastic) band structures of 2D periodic crystals from a binary geometry mask alone. The ground truth requires solving large Hermitian eigenproblems — plane-wave expansion (PWE) for photonics and finite element method (FEM) for phononics — at every wavevector along the Brillouin zone boundary. These solves are prohibitively expensive for design-space exploration or inverse design. The CANN replaces them with a single forward pass: geometry mask $\to$ $\omega_n(\mathbf{k})$ for all bands and k-points simultaneously.

**Key advance in v5.** Previous versions (v3, v4) predicted the full band structure in one monolithic output vector $(N_k \times N_b)$. v5 instead conditions each prediction on the specific wavevector $(k_x, k_y)$, following a DeepONet-style branch–trunk decomposition. This lets the network learn k-dependent physics (e.g., band crossings, avoided crossings) more faithfully.

## 2. Physics Assumptions

**Geometry.** A 2D square-lattice unit cell (lattice constant $a = 1$) composed of a silicon matrix ($\varepsilon = 8.9$, $\rho = 2330$ kg/m³, $\lambda = 68.4$ GPa, $\mu = 80.0$ GPa) with an air inclusion ($\varepsilon = 1.0$, $\rho \approx 0$, $\lambda \approx 0$, $\mu \approx 0$). The inclusion shape varies: perturbed circles (Fourier boundary descriptors), squares, ellipses, crosses, and rings.

**Photonic (TE polarization).** Maxwell's equations in a periodic dielectric $\varepsilon(\mathbf{r})$ reduce to the master equation. For TE polarization the eigenproblem is:

$$H^{\mathrm{TE}}*{\mathbf{G}\mathbf{G}'}(\mathbf{k}) = (\mathbf{k}+\mathbf{G}) \cdot [\varepsilon^{-1}]*{\mathbf{G}-\mathbf{G}'}  (\mathbf{k}+\mathbf{G}'), \qquad H^{\mathrm{TE}} \mathbf{c} = \frac{\omega^2}{c^2} \mathbf{c}$$

Frequencies are reported as dimensionless $\omega a / 2\pi c$. v5 uses TE polarization (correct for observing band gaps in air-hole-in-dielectric lattices), unlike v3 which used TM.

**Phononic (in-plane elastic, FEM).** The elastic wave equation $-\nabla \cdot \boldsymbol{\sigma}(\mathbf{u}) = \omega^2 \rho(\mathbf{r})\mathbf{u}$ with Bloch-Floquet boundary conditions $\mathbf{u}(\mathbf{r}) = \tilde{\mathbf{u}}(\mathbf{r})e^{i\mathbf{k}\cdot\mathbf{r}}$. For air-hole inclusions, the FEM solver meshes only the solid region with traction-free boundary conditions at hole boundaries, avoiding the $10^{12}$-contrast ill-conditioning that plagues PWE approaches. The resulting generalized eigenproblem:

$$\mathbf{K}(\mathbf{k})\tilde{\mathbf{u}} = \omega^2\mathbf{M}\tilde{\mathbf{u}}, \qquad \mathbf{K}(\mathbf{k}) = \mathbf{K}*0 + ik_j\mathbf{K}*{1,j} + k_jk_l\mathbf{K}_{2,jl}$$

is reduced to standard form via Cholesky decomposition and solved with `scipy.linalg.eigh`.

**Common assumptions.**

- Plane-wave truncation: $n_\mathrm{max} = 5$ (121 plane waves) for photonic PWE
- FEM mesh resolution: 30 nodes per edge
- Bands evaluated on the irreducible Brillouin zone path $\Gamma \to X \to M \to \Gamma$ with 10 points per segment ($N_k = 31$)
- C4v point-group symmetry of the square lattice

## 3. Model Architecture & Tools

**Language & framework.** Python 3.11, PyTorch 2.x (autograd, `nn.Module`), NumPy, SciPy (FEM solver), Matplotlib. Visualization with `torchview` + Graphviz.

### 3.1 DeepCNNEncoder

Hierarchical convolutional encoder with residual connections, replacing the LearnedKernelEncoder of v3. Processes the raw binary mask (1=hole, 0=solid) rather than the dielectric grid.


| Stage   | Operation                                                                | Output shape      |
| ------- | ------------------------------------------------------------------------ | ----------------- |
| Input   | Geometry mask                                                            | $(B, 1, 32, 32)$  |
| Stem    | Conv2d($1 \to 32$, $5\times5$) + BN + SiLU                               | $(B, 32, 32, 32)$ |
| Stage 1 | Conv2d($32 \to 64$, $3\times3$, stride 2) + BN + SiLU + ResBlock2d(64)   | $(B, 64, 16, 16)$ |
| Stage 2 | Conv2d($64 \to 128$, $3\times3$, stride 2) + BN + SiLU + ResBlock2d(128) | $(B, 128, 8, 8)$  |
| Pool    | AdaptiveAvgPool2d(1)                                                     | $(B, 128)$        |
| Proj    | Linear($128 \to 128$)                                                    | $(B, 128)$        |


Each **ResBlock2d** is: Conv2d + BN + SiLU + Conv2d + BN + skip connection + SiLU.

### 3.2 PerKBackbone (DeepONet-style)

Rather than predicting all $N_k \times N_b$ values at once, PerKBackbone concatenates the geometry embedding with each k-point coordinate and predicts $N_b$ values per k-point. This conditions the prediction on the specific location in the Brillouin zone.


| Layer  | Operation                                                       | Output shape    |
| ------ | --------------------------------------------------------------- | --------------- |
| Input  | Concat geometry feat + $(k_x, k_y)$                             | $(B, N_k, 130)$ |
| Pre    | Linear(130, 128) + SiLU + Linear(128, 128) + SmoothInverse(128) | $(B, N_k, 128)$ |
| Post   | Linear(128, 128) + SiLU + Linear(128, $N_b$)                    | $(B, N_k, N_b)$ |
| Skip   | Linear(130, $N_b$)                                              | $(B, N_k, N_b)$ |
| Output | Post(Pre(x)) + Skip(x)                                          | $(B, N_k, N_b)$ |


The k-points are registered as a buffer (not learned), sampled from the $\Gamma$–$X$–$M$–$\Gamma$ path.

**Physics-informed activation — SmoothInverse:** $\displaystyle f(x_i) = \frac{x_i}{x_i^2 + \eta_i^2}$, with learnable $\eta_i = e^{\lambda_i}$. Approximates the matrix inversion step ($\varepsilon^{-1}$ or $M^{-1}$) in the PWE/FEM eigensolve.

### 3.3 Architecture A: DualGridCANN_v5

Shared DeepCNNEncoder with two **independent** PerKBackbone heads:

$$\mathrm{mask} \xrightarrow{\text{DeepCNNEncoder}} \mathbf{f} \xrightarrow{\text{PerKBackbone}^{p}} \boldsymbol{\omega}^{\mathrm{phot}}, \quad \mathbf{f} \xrightarrow{\text{PerKBackbone}^{n}} \boldsymbol{\omega}^{\mathrm{phon}}$$

### 3.4 Architecture B: DualGridCANN_v5_cross

Shared encoder with **cross-gated** PerKBackbone heads. After the `pre` sub-network (at the SmoothInverse hidden layer), a CrossAttentionBlock exchanges information between branches:

$$\mathbf{g}_p = \sigma(\mathbf{W}_g^p\mathbf{h}_n), \qquad \mathbf{g}_n = \sigma(\mathbf{W}_g^n\mathbf{h}_p)$$

$$\mathbf{h}_p' = \mathbf{h}_p + \mathbf{g}_p \odot (\mathbf{V}^p\mathbf{h}_n), \qquad \mathbf{h}_n' = \mathbf{h}_n + \mathbf{g}_n \odot (\mathbf{V}^n\mathbf{h}_p)$$

The residual connection allows the model to learn to ignore cross-physics coupling when not useful. Full forward pass:


| Step | Operation                                | Output              |
| ---- | ---------------------------------------- | ------------------- |
| 1    | Mask prep                                | $(B, 1, 32, 32)$    |
| 2    | DeepCNNEncoder                           | $(B, 128)$          |
| 3    | Concat with k-points (per branch)        | $(B, 31, 130)$ each |
| 4    | Pre network + SmoothInverse (per branch) | $(B, 31, 128)$ each |
| 5    | **CrossAttentionBlock**                  | $(B, 31, 128)$ each |
| 6    | Post network (per branch)                | $(B, 31, N_b)$ each |
| 7    | + Skip connection                        | $(B, 31, N_b)$ each |


### 3.5 Physics-Informed Design

The PerKBackbone layers mirror the four-step PWE/FEM computational pipeline:

| PWE/FEM step | Network analog | Layer |
|-------------|----------------|-------|
| 1. Fourier decomposition of material fields | Feature extraction from geometry mask | DeepCNNEncoder |
| 2. Matrix assembly from Fourier coefficients and $(\mathbf{k}+\mathbf{G})$ | Mix geometry features with k-coordinates | Linear layers in `pre` |
| 3. Matrix inversion ($\varepsilon^{-1}$ or Cholesky $M^{-1}$) | Regularized eigenvalue flipping | SmoothInverse |
| 4. Eigenvalue extraction | Map to band frequencies | Linear layers in `post` |

#### 3.5.1 SmoothInverse — matrix inversion analog

$$f(x_i) = \frac{x_i}{x_i^2 + \eta_i^2}, \qquad \eta_i = e^{\lambda_i} \;\text{(learnable)}$$

**Physical derivation.** Matrix inversion acts element-wise on eigenvalues. If $A = Q\Lambda Q^{-1}$, then $A^{-1} = Q\Lambda^{-1}Q^{-1}$ with $\Lambda^{-1} = \mathrm{diag}(1/\lambda_1, 1/\lambda_2, \dots)$. Large eigenvalues become small and vice versa. The SmoothInverse reproduces this:

- For $|x_i| \gg \eta_i$: $f(x_i) \approx 1/x_i$ — the inversion regime
- For $|x_i| \ll \eta_i$: $f(x_i) \approx x_i/\eta_i^2$ — linear, regularized, no blow-up
- Peak response at $|x_i| \approx \eta_i$

In the photonic problem, the Hamiltonian requires $[\varepsilon^{-1}]_{\mathbf{G}\mathbf{G}'}$ — inversion of the dielectric matrix whose eigenvalues range from ~1 (air) to ~8.9 (silicon). In the phononic problem, the generalized eigenproblem $\Gamma\mathbf{c} = \omega^2 M\mathbf{c}$ is reduced via Cholesky $M = LL^H$ to $L^{-1}\Gamma L^{-H}\mathbf{v} = \omega^2\mathbf{v}$, requiring inversion of the mass matrix.

**Why not $1/x$?** Near-zero eigenvalues cause $1/x \to \infty$, producing numerical instability and exploding gradients. The physical system handles this through finite plane-wave truncation, but the network needs explicit regularization. The learnable $\eta_i$ serves as Tikhonov regularization — analogous to replacing $A^{-1}$ with $(A^HA + \eta^2 I)^{-1}A^H$. Each feature channel learns its own regularization scale.

The network's hidden features play the role of implicit material-matrix eigenvalues. The SmoothInverse layer flips them (large $\leftrightarrow$ small) with built-in stability, just as the physical eigenproblem requires.

#### 3.5.2 SiLU — smooth non-monotonic activation

$$\mathrm{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**Behavior:**

- $x \gg 0$: $\sigma(x) \to 1$, so $\mathrm{SiLU}(x) \approx x$ (identity)
- $x \ll 0$: $\sigma(x) \to 0$, so $\mathrm{SiLU}(x) \approx 0$ (suppressed)
- $x \approx -1.28$: dips slightly negative (minimum $\approx -0.28$), then returns to zero

**Physics motivation.** Band dispersion curves $\omega_n(\mathbf{k})$ are non-monotonic along the k-path — a band may rise from $\Gamma$ to $X$, flatten near $X$, then fall toward $M$. With ReLU ($\max(0, z)$), which is monotonic in its pre-activation, a single neuron can only produce a response that increases or stays flat. Representing a "rise then fall" requires at least two ReLU neurons with opposite-sign weights. SiLU's built-in non-monotonicity lets fewer neurons capture these shapes — important given the compact architecture (128 hidden units) and limited data (499 samples).

Additionally, band frequencies are smooth, analytic functions of $\mathbf{k}$ (the Hamiltonian matrix elements involve continuous products of wavevector components and Fourier coefficients). SiLU has continuous derivatives everywhere, unlike ReLU's discontinuous gradient at zero, better matching this smoothness and avoiding dead neurons.

#### 3.5.3 Sigmoid — cross-physics gating

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Used exclusively in the CrossAttentionBlock to produce gates $\mathbf{g} \in [0, 1]^{128}$ that control how much information flows between photonic and phononic branches. The $[0,1]$ range means each hidden feature can be fully suppressed (0) or fully passed (1), letting the model learn which aspects of one physics are informative for the other.

#### 3.5.4 Linear layers — matrix assembly and eigenvalue extraction

The linear layers $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ serve different physical roles depending on their position:

- **Pre-network Linear(130 → 128) and Linear(128 → 128):** Mix geometry features with k-coordinates. In the PWE formulation, Hamiltonian matrix elements are bilinear in wavevector components and Fourier coefficients — e.g. $H_{\mathbf{G}\mathbf{G}'}^{\mathrm{TE}} = (\mathbf{k}+\mathbf{G}) \cdot [\varepsilon^{-1}]_{\mathbf{G}-\mathbf{G}'} (\mathbf{k}+\mathbf{G}')$. A linear layer operating on the concatenated $[\mathbf{f}; k_x, k_y]$ can learn exactly these products — the weight matrix entries connecting geometry features to k-coordinates represent coupling terms.
- **Post-network Linear(128 → 128) and Linear(128 → $N_b$):** After the SmoothInverse "inversion," these map from the reduced-eigenproblem hidden space to band frequencies — analogous to reading off eigenvalues from the diagonalized Hamiltonian.
- **Skip Linear(130 → $N_b$):** Direct linear mapping from input to output, capturing first-order perturbation theory: $\delta\omega_n \propto \langle n|\delta H|n\rangle$. For small geometry variations, eigenvalue shifts are approximately linear in the perturbation. The nonlinear pre/post path then learns the higher-order corrections (band repulsion, avoided crossings, large-perturbation effects).

#### 3.5.5 Architectural physics constraints

- **Per-k conditioning (DeepONet structure)** — the backbone concatenates geometry features with $(k_x, k_y)$ and predicts $N_b$ values per k-point, mirroring the physical parameterization: for each $\mathbf{k}$, a different Hamiltonian $H(\mathbf{k})$ is assembled and solved. The PWE matrix elements depend explicitly on $(\mathbf{k}+\mathbf{G})$, so the Hamiltonian is a smooth, analytic function of $\mathbf{k}$. Conditioning on $(k_x, k_y)$ lets the network learn this dependence directly rather than memorizing a flat output vector.
- **Fixed k-path buffer** — the Brillouin zone path $\Gamma \to X \to M \to \Gamma$ is computed from `pwe_torch.make_k_path()` and registered as a non-learnable buffer, encoding the reciprocal-space geometry of the square lattice.
- **Shared encoder, independent heads** — a single DeepCNNEncoder extracts geometry features for both physics branches, reflecting the fact that photonic ($\varepsilon$) and phononic ($\rho, \lambda, \mu$) properties derive from the same spatial geometry.
- **Cross-gating (v5_cross)** — bilinear gating between branches at the hidden layer, with residual connections. Positioned after the SmoothInverse "inversion" step, where both branches are in a normalized hidden space and cross-physics correlations can be meaningfully learned. The residual lets the model ignore cross-physics coupling when unhelpful.
- **Mask input** — raw binary mask (1=hole, 0=solid) rather than derived material fields. The material mapping ($\varepsilon = 8.9 - 7.9 \cdot \mathrm{mask}$) is implicit.

#### 3.5.6 Notable omission from v3

v5 drops the **Softplus** activation and **BandOrderingHead** that enforced monotonic band ordering by construction in v3:

$$\Delta_n = \mathrm{softplus}(z_n), \qquad \omega_n = \alpha\sum_{j=1}^{n}\Delta_j$$

This guaranteed $\omega_1 \leq \omega_2 \leq \dots$ but prevented the network from representing band crossings — physically real phenomena where two bands exchange character at specific k-points. v5 uses raw output with per-band standardization, allowing crossings to be learned from data.

### 3.6 Key Differences from v3


| Aspect        | v3                                        | v5                                                              |
| ------------- | ----------------------------------------- | --------------------------------------------------------------- |
| Input         | Dielectric grid $\varepsilon(\mathbf{r})$ | Binary mask (1=hole, 0=solid)                                   |
| Encoder       | LearnedKernelEncoder (G-vector Conv2D)    | DeepCNNEncoder (hierarchical conv + residual)                   |
| Backbone      | Monolithic FC → $(N_k \times N_b)$ output | Per-k conditioned: $(k_x, k_y)$ concatenated with geometry feat |
| Band ordering | BandOrderingHead (cumsum of softplus)     | Raw output (no enforced ordering)                               |
| Normalization | Global max normalization                  | Per-(k, band) standardization (zero mean, unit variance)        |
| Photonic pol. | TM                                        | TE                                                              |
| Training      | Full-batch, 5000 epochs                   | Mini-batch (256), 3000 epochs                                   |
| Parameters    | 254,803                                   | 583,936 (v5) / 649,984 (v5 cross)                               |


### 3.7 Architecture Diagrams

**DualGridCANN_v5:**

DualGridCANN_v5

**DualGridCANN_v5_cross:**

DualGridCANN_v5_cross

## 4. Loss Function

Joint MSE on per-band standardized targets with equal weighting:

$$\mathcal{L} = \mathcal{L}*{\mathrm{phot}} + \mathcal{L}*{\mathrm{phon}}$$

where each term is the MSE on standardized targets:

$$\mathcal{L}*{\mathrm{phot}} = \frac{1}{BN_kN_b^p} \sum*{i,j,n} \left(\hat{z}^p_{ijn} - z^p_{ijn}\right)^2, \qquad z^p_{ijn} = \frac{\omega^p_{ijn} - \mu^p_{jn}}{\sigma^p_{jn}}$$

$$\mathcal{L}*{\mathrm{phon}} = \frac{1}{BN_kN_b^n} \sum*{i,j,n} \left(\hat{z}^n_{ijn} - z^n_{ijn}\right)^2, \qquad z^n_{ijn} = \frac{\omega^n_{ijn} - \mu^n_{jn}}{\sigma^n_{jn}}$$

Here $\mu_{jn}$ and $\sigma_{jn}$ are the per-(k-point, band) mean and standard deviation computed once from the training set before training begins. This standardization ensures all targets have comparable scale regardless of whether they are photonic ($\sim 0.1$) or phononic ($\sim 10^3$–$10^4$), and regardless of which band or k-point.

**Equivalence to weighted MSE.** Since standardization is a fixed linear transformation $z = (\omega - \mu)/\sigma$, minimizing the loss in standardized space is equivalent to minimizing a weighted MSE in the original frequency space:

$$\mathcal{L} = \sum_{j,n} \frac{1}{\sigma_{jn}^2}\left(\hat{\omega}_{jn} - \omega_{jn}\right)^2$$

The global minimum ($\hat{\omega} = \omega$) is the same regardless of the weighting. The standardization only changes the optimization landscape — how fast the network converges on each band. Bands with small variance $\sigma_{jn}$ (e.g., low-frequency acoustic bands) are upweighted, receiving larger gradients for the same absolute error. Bands with large variance (e.g., high-frequency optical bands) are downweighted. This prevents the gradient from being dominated by the highest-magnitude bands and ensures equal relative accuracy across all bands.

**Validity.** Dividing by $\sigma_{jn}$ is valid because $\mu_{jn}$ and $\sigma_{jn}$ are fixed scalars computed once from the training set — no different from choosing different units (e.g., GHz instead of Hz). They do not depend on the model's predictions (no data leakage) and are not recomputed per batch (no moving target).

**Regularization.** No explicit regularization term. Implicit regularization via:

- Weight decay: $10^{-4}$ (L2 penalty in Adam)
- Gradient clipping: $\nabla_{\max} = 1.0$
- Cosine-annealed learning rate: $10^{-3} \to 10^{-5}$ over 3000 epochs
- Batch normalization in the encoder

The v3 baseline is trained with global-max normalization for comparison.

## 5. Data

**Generation.** Training data is generated by solving both eigenproblems for each geometry using `pwe_torch.solve_bands()` (photonic PWE, TE) and `fem_elastic_fast.solve_elastic_bands_fem()` (phononic FEM). No external dataset; all ground truth is computed from first principles.

**Dataset size.** 500 train / 50 test geometries (499/49 after rejecting invalid shapes).

**Geometry sampling.** Two families, rasterized on a $32 \times 32$ grid:

1. **Perturbed circles** (200 train, 20 test): Fourier boundary $r(\theta) = a_0 + \sum_{n=1}^{4}(a_n \cos n\theta + b_n \sin n\theta)$ with $a_0 \in [0.10, 0.40]$, perturbation scale $\sigma = 0.08$.
2. **Standard shapes** (300 train, 30 test): randomly sized crosses, squares, ellipses, and rings.

**Material mapping.** Each pixel maps mask $\in 0, 1$ to material properties:


| Property      | Si matrix (mask=0) | Air hole (mask=1) |
| ------------- | ------------------ | ----------------- |
| $\varepsilon$ | 8.9                | 1.0               |
| $\rho$        | 2330 kg/m³         | 1.225 kg/m³       |
| $\lambda$     | 68.4 GPa           | $\approx 0$       |
| $\mu$         | 80.0 GPa           | $\approx 0$       |


**Targets.** Per geometry: $N_k = 31$ k-points along $\Gamma \to X \to M \to \Gamma$, $N_b^p = 6$ photonic bands, $N_b^n = 10$ phononic bands. Output tensors are $(B, 31, 6)$ and $(B, 31, 10)$.

**Sample band structures from the dataset:**

Photonic bands (TE, PWE ground truth) and phononic bands (FEM ground truth) for five representative test geometries:


| Geometry         | Description                                    |
| ---------------- | ---------------------------------------------- |
| Perturbed circle | Fourier boundary with 4-harmonic perturbations |
| Ring             | Annular inclusion                              |
| Ellipse          | Elongated inclusion breaking C4 symmetry       |
| Square           | Sharp-cornered inclusion                       |
| Cross            | Multi-arm inclusion                            |


## 6. Planned Analyses

**Architecture comparison.** Three models trained under identical conditions (3000 epochs, Adam, cosine LR, batch size 256):


| Model                      | Description                                                    | Parameters |
| -------------------------- | -------------------------------------------------------------- | ---------- |
| DualGridCANN_v5            | Shared DeepCNNEncoder, independent PerKBackbone heads          | 583,936    |
| DualGridCANN_v5_cross      | Shared encoder, cross-gated PerKBackbone heads                 | 649,984    |
| DualGridCANN v3 (baseline) | Shared LearnedKernelEncoder, monolithic backbones, cumsum head | 254,803    |


**Ablations and studies:**

- **Per-k conditioning vs. monolithic output**: does conditioning on $(k_x, k_y)$ improve accuracy at band crossings and avoided crossings?
- **Cross-physics transfer**: does the cross-gated model improve photonic accuracy by leveraging phononic information (or vice versa)?
- **Standardization vs. max normalization**: compare per-band standardization (v5) against global-max normalization (v3) for training stability and final accuracy
- **Encoder depth**: compare DeepCNNEncoder (hierarchical conv + residual) against LearnedKernelEncoder (physics-initialized Conv2D)
- **Band ordering**: compare raw output (v5) vs. enforced monotonicity via cumsum (v3)

## 7. Evaluation Metrics

**Root Mean Squared Error (RMSE).** Primary metric, computed on unnormalized (original-scale) frequencies:

$$\mathrm{RMSE} = \sqrt{\frac{1}{N_{\mathrm{test}}N_kN_b} \sum_{i=1}^{N_{\mathrm{test}}} \sum_{j=1}^{N_k} \sum_{n=1}^{N_b} \left(\hat{\omega}*{ijn} - \omega*{ijn}\right)^2}$$

RMSE penalizes large errors more heavily than MAE and is in the same units as the target frequencies, making it directly interpretable. Reported separately for photonic ($\omega a / 2\pi c$, dimensionless) and phononic (Hz·$a$) bands. Additionally reported **per-band** to identify which bands are hardest to predict.

**Per-band RMSE.** For each band $n$:

$$\mathrm{RMSE}*n = \sqrt{\frac{1}{N*{\mathrm{test}}N_k} \sum_{i=1}^{N_{\mathrm{test}}} \sum_{j=1}^{N_k} \left(\hat{\omega}*{ijn} - \omega*{ijn}\right)^2}$$

This reveals whether higher-frequency bands (which have more complex dispersion) are systematically harder. The overall and per-band RMSE are related by:

$$\mathrm{RMSE}_{\mathrm{overall}} = \sqrt{\frac{1}{N_b}\sum_{n=1}^{N_b} \mathrm{RMSE}_n^2}$$

**Distinction from the loss function.** Both the loss and RMSE have the same MSE form, but they measure different things. The loss operates on standardized targets $z$ (weighted by $1/\sigma_{jn}^2$), optimizing for equal **relative** accuracy across bands. The RMSE is computed on original-scale frequencies $\omega$ with uniform weighting, reporting **absolute** accuracy in physical units ($\omega a/2\pi c$ for photonic, Hz·$a$ for phononic).

**Normalized RMSE (NRMSE).** Expresses error as a fraction of the target frequency range, enabling comparison across physics:

$$\mathrm{NRMSE} = \frac{\mathrm{RMSE}}{\omega_{\max} - \omega_{\min}}$$

**Coefficient of Variation of RMSE (CV-RMSE).** Expresses error as a fraction of the mean target frequency — the standard "percentage error" interpretation:

$$\mathrm{CV\text{-}RMSE} = \frac{\mathrm{RMSE}}{\bar{\omega}}$$

CV-RMSE is always larger than NRMSE (since $\bar{\omega} < \omega_{\max} - \omega_{\min}$) and is more commonly used in the literature. Both allow direct comparison of photonic and phononic accuracy on a common dimensionless scale.

**Per-band relative RMSE.** Normalizes each band's RMSE by its own mean frequency:

$$\mathrm{rRMSE}_n = \frac{\mathrm{RMSE}_n}{\bar{\omega}_n}$$

This reveals whether the network achieves uniform relative accuracy across bands or struggles disproportionately with specific bands.

**Qualitative: band diagram overlay.** Side-by-side plots of predicted (dashed red) vs. ground truth (solid black) band structures for selected test geometries, providing visual assessment of band crossing accuracy and gap prediction.

### Training Results

**Test-set RMSE (3000 epochs):**


| Model        | Photonic RMSE ($\omega a/2\pi c$) | Phononic RMSE (Hz·$a$) |
| ------------ | --------------------------------- | ---------------------- |
| **v5**       | **0.00505**                       | **158.1**              |
| **v5 cross** | **0.00532**                       | **153.4**              |
| v3 baseline  | 0.01102                           | 886.7                  |


**Per-band photonic RMSE ($\omega a/2\pi c$):**


| Model       | b0     | b1     | b2     | b3     | b4     | b5     |
| ----------- | ------ | ------ | ------ | ------ | ------ | ------ |
| v5          | 0.0022 | 0.0039 | 0.0048 | 0.0054 | 0.0059 | 0.0068 |
| v5 cross    | 0.0023 | 0.0045 | 0.0049 | 0.0051 | 0.0066 | 0.0070 |
| v3 baseline | 0.0063 | 0.0092 | 0.0102 | 0.0111 | 0.0138 | 0.0136 |


**Per-band phononic RMSE (Hz·$a$):**


| Model       | b0    | b1    | b2    | b3    | b4    | b5    | b6    | b7    | b8     | b9     |
| ----------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ------ | ------ |
| v5          | 71.3  | 96.4  | 147.7 | 165.1 | 131.1 | 187.8 | 181.1 | 158.0 | 199.2  | 191.7  |
| v5 cross    | 62.2  | 92.0  | 164.3 | 162.2 | 122.9 | 191.6 | 169.7 | 158.7 | 186.9  | 170.4  |
| v3 baseline | 567.0 | 824.2 | 892.3 | 967.4 | 857.6 | 729.2 | 802.0 | 848.8 | 1021.3 | 1206.5 |


Both v5 architectures achieve $\sim$2× lower photonic RMSE and $\sim$5.5× lower phononic RMSE than the v3 baseline, despite the v3 baseline having enforced band ordering. The v5 cross model slightly outperforms v5 on phononic bands, suggesting modest benefit from cross-physics information exchange.

