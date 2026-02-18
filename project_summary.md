# Constitutive ANN for Photonic/Phononic Band Gap Inverse Design

## Project Summary

This document summarizes the research plan for developing a differentiable pipeline that uses neural networks to predict and inversely design photonic and phononic band gaps, as an alternative to traditional FEM solvers.

---

## 1. Background & Motivation

### Constitutive Relations
Physics governing wave propagation in periodic media involves two kinds of equations:

- **Balance/conservation laws** (Maxwell's equations, Newton's second law on a continuum) — universal, material-independent.
- **Constitutive relations** — material-specific response functions that encode how a material responds to a field:
  - Electromagnetics: **D = εE** (permittivity ε)
  - Elastodynamics: **σ = C:ε** (stiffness tensor C)

Band gap computation reduces to a generalized eigenvalue problem where the matrix entries come from constitutive tensors evaluated across the unit cell. The geometry determines *where* each material is; the constitutive tensor determines *what* each material does.

### Why Constitutive ANNs?
Constitutive ANNs (originating in solid mechanics, e.g., Linka & Kuhl) bake physical constraints into the architecture rather than using soft penalty losses. Relevant constraints include:

- **Symmetry equivariance** (C4v, C6v) via group-equivariant layers or invariant descriptors
- **Reciprocity** (ω(k) = ω(−k)) by constructing even functions of k
- **Positive-definiteness** of constitutive tensors via Cholesky parameterization (C = LLᵀ)
- **Thermodynamic consistency** by learning a strain energy potential W(ε) and obtaining stresses via autodiff
- **Causality / Kramers-Kronig** for dispersive materials (advanced)

---

## 2. Architecture Options Considered

### Option A — Full Surrogate
Network maps unit cell parameters directly to band gap properties. Common but least "constitutive."

### Option B — Learned Constitutive Law (Selected for MVP)
Network learns effective constitutive relations for a unit cell; a reduced eigenvalue problem is still solved analytically or semi-analytically. More physically grounded.

### Option C — Hybrid Operator Learning
Network learns the map from material/geometry to the discretized operator (e.g., via Fourier Neural Operator). Eigensolve performed on the learned operator.

---

## 3. Selected Approach: Differentiable PWE Inverse Design

### Core Pipeline
The MVP uses a fully differentiable Plane-Wave Expansion (PWE) pipeline in which gradients flow from a band gap objective back to design variables:

```
design variables z (N×N grid)
        │
        ▼
sigmoid mapping: ε(x,y) = ε_air + (ε_high - ε_air) · σ(z)
        │
        ▼
2D FFT → Fourier coefficients ε̂(G−G')
        │
        ▼
Build epsilon matrix, apply inverse rule, assemble Hermitian eigenproblem
        │
        ▼
eigh → eigenvalues ω²_n(k) for each k-point along Γ→X→M→Γ
        │
        ▼
Extract band gap via smooth min/max (softmax-based)
        │
        ▼
loss = −gap (or −gap/midgap)
        │
        ▼
Gradient computation → optimizer updates z
```

### Why PWE over FEM?
PWE is naturally differentiable: every step (FFT, matrix assembly, matrix inverse, `eigh`) has autodiff support. No mesh generation, element assembly, or boundary condition handling required.

### Why TM Polarization First?
TM (E out of plane) gives a simpler eigenvalue problem — a single scalar equation requiring only the Fourier coefficients of ε(x,y). TE requires inverting ε, adding an extra step.

### The Inverse Rule
The epsilon matrix is built from Fourier coefficients of ε(r), then matrix-inverted (rather than directly Fourier-transforming 1/ε). This converges much faster at sharp dielectric interfaces (Ho, Chan & Soukoulis 1990).

### Smooth Min/Max
Standard `min`/`max` produce sparse gradients. Use softmax-weighted sums instead:

```python
def smooth_min(x, beta=-50):
    return jnp.sum(x * jax.nn.softmax(beta * x))

def smooth_max(x, beta=50):
    return jnp.sum(x * jax.nn.softmax(beta * x))
```

---

## 4. Three-Week MVP Plan

### Week 1 — Forward PWE Solver ✅ (Complete)
- Discretize ε(r) on N×N grid over the unit cell
- FFT to get Fourier coefficients ε̂(G)
- Build epsilon Toeplitz matrix, apply inverse rule
- Assemble Hermitian eigenproblem for TM polarization
- Diagonalize at each k-point along Γ→X→M→Γ
- Plot band structure
- **Validate** against published results (e.g., square lattice of dielectric rods, r/a = 0.2, ε = 8.9)

**Status:** Implemented in NumPy. Agrees with published results for the dielectric rod case.

### Week 2 — Differentiable Optimization
- Wrap forward solver in a loss function (negative band gap)
- Compute gradients via finite differences (NumPy) or autodiff (JAX/PyTorch)
- Run Adam optimization from random ε grid; watch band gap open
- Add regularization: blur filter for connectivity, binary penalty to push toward 0/1 designs

**Gradient strategy (given ARM64/NumPy constraints):**
- **Immediate:** Finite differences — perturb each pixel, recompute gap. For 32×32 grid = 1024 forward solves per step, ~seconds each.
- **Next:** Port to PyTorch (good ARM64/Apple Silicon support, `torch.linalg.eigh` supports autodiff).
- **Later:** JAX (ideal but ARM64 install can be finicky).

### Week 3 — Polish & Demonstrate
- Run inverse design experiments: maximize gaps between various band pairs, target specific center frequencies
- Validate optimized designs in MPB or COMSOL
- Increase plane-wave cutoff and grid resolution to check convergence
- Produce figures: optimized unit cells, band structures, convergence curves
- **Stretch goal:** Replace pixel grid with a small CNN decoder, optimize in latent space

---

## 5. Introducing the Neural Network

### Where Autonomous Discovery Adds Value
Analysis of each pipeline step:

| Step | Value of learning? |
|---|---|
| FFT | None — exact and fast |
| Matrix assembly | None — known linear operation |
| Eigensolve | None — numerically exact |
| Band gap extraction | None — design choice |
| **ε(x,y) parameterization** | **High — this is where discoverable structure lives** |

### Level 1 — Generative Prior over Unit Cells
Train a CNN decoder (VAE or conditioned generator) on designs that produce band gaps. Optimization then occurs in a compact latent space where most directions lead to physically reasonable designs. This replaces the raw pixel parameterization — everything downstream stays the same.

### Level 2 — Learned Effective Constitutive Properties (Future Work)
Train a network to predict effective ε̂(G−G') for coarse PWE solves that match fine-resolution truth. This is learned homogenization that goes beyond analytical mixing rules. The network can discover that near resonances, effective permittivity can be negative or strongly anisotropic.

### Training the CNN Decoder

**Preferred approach — Physics-conditioned decoder (end-to-end):**

```
target gap specification (width, center freq, band indices)
        │
        ▼
decoder CNN → predicted ε(x,y)
        │
        ▼
differentiable PWE solver → actual band structure
        │
        ▼
loss = (target_gap - actual_gap)² + α·(target_midgap - midgap)² + λ·regularization
```

No pre-computed training data needed — sample random targets and latent vectors, forward through the solver, backpropagate the physics loss directly into the decoder weights.

**Regularization terms:**
- Binary penalty: `mean(ε_norm · (1 - ε_norm))` — pushes toward air/material
- Smoothness penalty — penalizes isolated pixels
- Connectivity penalty (optional) — penalizes islands

**Alternative:** Pretrain with unsupervised VAE (reconstruction loss + KL divergence) on gap-producing designs, then fine-tune with the physics loss above.

---

## 6. Extension to Hexagonal Lattices

### What Changes
- **Lattice basis vectors:** a₁ = a(1,0), a₂ = a(1/2, √3/2) with corresponding reciprocal vectors
- **Brillouin zone path:** Γ→M→K→Γ instead of Γ→X→M→Γ
- **Unit cell representation:** Either a skewed grid (natural for PWE/FFT) or a rectangular supercell

### What Stays the Same
- PWE matrix assembly (still ε̂(G−G') indexed by reciprocal lattice vectors)
- Eigensolve, loss function, optimizer, pipeline architecture
- CNN decoder structure (still outputs N×N grid, interpreted in different coordinates)

### Symmetry Enforcement
Parameterize one twelfth of the unit cell (fundamental domain of C6v) and tile by symmetry operations. Hexagonal symmetry is what enables complete (TE+TM) band gaps — the main reason to use hexagonal lattices.

---

## 7. Scope Boundaries

### In Scope
- 2D photonic crystals (TM, then TE)
- Square lattice → hexagonal lattice
- Pixelated binary unit cells (one material + air)
- Forward prediction + inverse design
- CNN decoder for one-shot inverse design

### Deferred
- 3D unit cells (eigensolve cost explosion)
- Lossy/dispersive materials (breaks Hermiticity)
- Coupled phoxonic crystals (photonic + phononic simultaneously)
- Phononic extension (swap constitutive layer ε → C_ijkl — architecturally straightforward but doubles validation effort)

---

## 8. Key Implementation Notes

- **Plane-wave count:** 11×11 (121) for prototyping, 15×15 to 21×21 for publication quality
- **Symmetry enforcement:** Parameterize one octant (square) or one twelfth (hexagonal) and tile — reduces design variables by 8× or 12×
- **Checkerboard prevention:** Convolve design variables with a Gaussian blur before sigmoid
- **Gradient validation:** Always compare autodiff/finite-diff gradients against each other before trusting the optimization
- **Normalized units:** a = 1, c = 1; frequencies as ωa/(2πc) = a/λ; gap quality as gap-midgap ratio
