"""
Constitutive relation discovery from phoxonic dataset.

Target A: MLP regression from geometry features to gap properties.
Target B: Autoencoder compressing s(r) to latent, with dual-head decoders
          predicting photonic and phononic bands.

Usage:
    # Target A: MLP regression
    python discover_constitutive.py mlp --dataset phoxonic_dataset.pt

    # Target B: Autoencoder
    python discover_constitutive.py autoencoder --dataset phoxonic_dataset.pt

    # Visualization only (from saved models)
    python discover_constitutive.py visualize --dataset phoxonic_dataset.pt

    # Symbolic regression on trained MLP features
    python discover_constitutive.py symbolic --dataset phoxonic_dataset.pt
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path):
    """Load and split the phoxonic dataset into train/val/test."""
    data = torch.load(path, weights_only=False)
    samples = data["samples"]
    N = data["grid_size"]
    n_k = data["k_points"].shape[0]

    # Extract arrays
    s_grids = np.stack([s["s_grid"] for s in samples])
    fills = np.array([s["fill_fraction"] for s in samples], dtype=np.float32)
    fft_coeffs = np.stack([s["fft_coeffs"] for s in samples])
    bands_em = np.stack([s["bands_em"] for s in samples])
    bands_ac = np.stack([s["bands_ac"] for s in samples])
    gap_em = np.array([s["best_gap_em"] for s in samples], dtype=np.float32)
    mid_em = np.array([s["mid_em"] for s in samples], dtype=np.float32)
    gap_ac = np.array([s["best_gap_ac"] for s in samples], dtype=np.float32)
    mid_ac = np.array([s["mid_ac"] for s in samples], dtype=np.float32)
    has_dual = np.array([s["has_dual_gap"] for s in samples])

    n = len(samples)
    idx = np.random.permutation(n)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    splits = {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }

    return {
        "s_grids": s_grids, "fills": fills, "fft_coeffs": fft_coeffs,
        "bands_em": bands_em, "bands_ac": bands_ac,
        "gap_em": gap_em, "mid_em": mid_em,
        "gap_ac": gap_ac, "mid_ac": mid_ac,
        "has_dual": has_dual,
        "splits": splits, "N": N, "n_k": n_k,
        "meta": data,
    }


# ---------------------------------------------------------------------------
# Target A: MLP gap regression
# ---------------------------------------------------------------------------

class GapMLP(nn.Module):
    def __init__(self, n_feat, n_out=6, hidden=128, n_layers=4):
        super().__init__()
        layers = [nn.Linear(n_feat, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_features(ds, idx):
    """Build feature matrix: [fill_fraction, fft_coeffs]."""
    fills = ds["fills"][idx, None]
    fft = ds["fft_coeffs"][idx]
    return np.hstack([fills, fft])


def build_targets(ds, idx):
    """Target: [gap_em, mid_em, gap_ac, mid_ac, gap_em/mid_em, gap_ac/mid_ac]."""
    ge = ds["gap_em"][idx, None]
    me = ds["mid_em"][idx, None]
    ga = ds["gap_ac"][idx, None]
    ma = ds["mid_ac"][idx, None]
    safe_me = np.maximum(me, 1e-6)
    safe_ma = np.maximum(ma, 1e-6)
    ratio_em = ge / safe_me
    ratio_ac = ga / safe_ma
    return np.hstack([ge, me, ga, ma, ratio_em, ratio_ac])


def train_mlp(ds, epochs=200, lr=1e-3, batch_size=256, hidden=128):
    """Train MLP to predict gap properties from geometric features."""
    splits = ds["splits"]
    X_train = torch.tensor(build_features(ds, splits["train"]), dtype=torch.float32)
    Y_train = torch.tensor(build_targets(ds, splits["train"]), dtype=torch.float32)
    X_val = torch.tensor(build_features(ds, splits["val"]), dtype=torch.float32)
    Y_val = torch.tensor(build_targets(ds, splits["val"]), dtype=torch.float32)

    # Normalize inputs
    x_mean, x_std = X_train.mean(0), X_train.std(0).clamp(min=1e-6)
    y_mean, y_std = Y_train.mean(0), Y_train.std(0).clamp(min=1e-6)
    X_train_n = (X_train - x_mean) / x_std
    X_val_n = (X_val - x_mean) / x_std
    Y_train_n = (Y_train - y_mean) / y_std
    Y_val_n = (Y_val - y_mean) / y_std

    model = GapMLP(X_train_n.shape[1], Y_train_n.shape[1], hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_val = float("inf")
    train_losses, val_losses = [], []

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(X_train_n.shape[0])
        ep_loss = 0.0
        n_batch = 0

        for i in range(0, X_train_n.shape[0], batch_size):
            batch_idx = perm[i:i + batch_size]
            xb, yb = X_train_n[batch_idx], Y_train_n[batch_idx]
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batch += 1

        scheduler.step()
        train_loss = ep_loss / max(n_batch, 1)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_n)
            val_loss = F.mse_loss(val_pred, Y_val_n).item()
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std,
            }, "phoxonic_mlp.pt")

        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep+1:3d}: train={train_loss:.4f}, val={val_loss:.4f}")

    return model, train_losses, val_losses, x_mean, x_std, y_mean, y_std


# ---------------------------------------------------------------------------
# Target B: Autoencoder with dual-head band prediction
# ---------------------------------------------------------------------------

class PhoxonicAutoencoder(nn.Module):
    """Compress geometry s(r) to a latent vector, predict both band types."""

    def __init__(self, N, n_k, n_bands_em, n_bands_ac, latent_dim=8):
        super().__init__()
        self.N = N
        self.n_k = n_k
        self.n_bands_em = n_bands_em
        self.n_bands_ac = n_bands_ac

        # Encoder: NxN -> latent
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(N * N, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, latent_dim),
        )

        # Photonic band decoder
        self.decoder_em = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, n_k * n_bands_em),
        )

        # Phononic band decoder
        self.decoder_ac = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, n_k * n_bands_ac),
        )

        # Geometry reconstruction decoder
        self.decoder_geom = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, N * N), nn.Sigmoid(),
        )

    def encode(self, s):
        return self.encoder(s)

    def forward(self, s):
        z = self.encode(s)
        bands_em = self.decoder_em(z).view(-1, self.n_k, self.n_bands_em)
        bands_ac = self.decoder_ac(z).view(-1, self.n_k, self.n_bands_ac)
        s_recon = self.decoder_geom(z).view(-1, self.N, self.N)
        return z, bands_em, bands_ac, s_recon


def train_autoencoder(ds, latent_dim=8, epochs=300, lr=1e-3, batch_size=256,
                      recon_weight=0.1):
    """Train autoencoder on the phoxonic dataset."""
    splits = ds["splits"]
    N, n_k = ds["N"], ds["n_k"]
    n_bands_em = ds["bands_em"].shape[2]
    n_bands_ac = ds["bands_ac"].shape[2]

    S_train = torch.tensor(ds["s_grids"][splits["train"]], dtype=torch.float32)
    B_em_train = torch.tensor(ds["bands_em"][splits["train"]], dtype=torch.float32)
    B_ac_train = torch.tensor(ds["bands_ac"][splits["train"]], dtype=torch.float32)

    S_val = torch.tensor(ds["s_grids"][splits["val"]], dtype=torch.float32)
    B_em_val = torch.tensor(ds["bands_em"][splits["val"]], dtype=torch.float32)
    B_ac_val = torch.tensor(ds["bands_ac"][splits["val"]], dtype=torch.float32)

    # Normalize bands
    bem_mean, bem_std = B_em_train.mean(), B_em_train.std().clamp(min=1e-6)
    bac_mean, bac_std = B_ac_train.mean(), B_ac_train.std().clamp(min=1e-6)

    model = PhoxonicAutoencoder(N, n_k, n_bands_em, n_bands_ac, latent_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_val = float("inf")
    train_losses, val_losses = [], []

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(S_train.shape[0])
        ep_loss = 0.0
        n_batch = 0

        for i in range(0, S_train.shape[0], batch_size):
            bi = perm[i:i + batch_size]
            sb, bem_b, bac_b = S_train[bi], B_em_train[bi], B_ac_train[bi]

            z, pred_em, pred_ac, s_recon = model(sb)

            loss_em = F.mse_loss(pred_em, (bem_b - bem_mean) / bem_std)
            loss_ac = F.mse_loss(pred_ac, (bac_b - bac_mean) / bac_std)
            loss_recon = F.binary_cross_entropy(s_recon, sb)

            loss = loss_em + loss_ac + recon_weight * loss_recon

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batch += 1

        scheduler.step()
        train_loss = ep_loss / max(n_batch, 1)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            z_v, pred_em_v, pred_ac_v, s_recon_v = model(S_val)
            loss_em_v = F.mse_loss(pred_em_v,
                                   (B_em_val - bem_mean) / bem_std)
            loss_ac_v = F.mse_loss(pred_ac_v,
                                   (B_ac_val - bac_mean) / bac_std)
            loss_recon_v = F.binary_cross_entropy(s_recon_v, S_val)
            val_loss = (loss_em_v + loss_ac_v +
                        recon_weight * loss_recon_v).item()
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "latent_dim": latent_dim,
                "bem_mean": bem_mean, "bem_std": bem_std,
                "bac_mean": bac_mean, "bac_std": bac_std,
            }, "phoxonic_autoencoder.pt")

        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep+1:3d}: train={train_loss:.4f}, "
                  f"val={val_loss:.4f}")

    return model, train_losses, val_losses


# ---------------------------------------------------------------------------
# Symbolic regression (optional, requires gplearn or pysindy)
# ---------------------------------------------------------------------------

def run_symbolic_regression(ds):
    """Attempt symbolic regression on the dataset features."""
    try:
        from gplearn.genetic import SymbolicRegressor
    except ImportError:
        print("gplearn not installed. Install with: pip install gplearn")
        print("Skipping symbolic regression.")
        return None

    splits = ds["splits"]
    X_train = build_features(ds, splits["train"])
    Y_train = build_targets(ds, splits["train"])

    target_names = ["gap_em", "mid_em", "gap_ac", "mid_ac",
                    "ratio_em", "ratio_ac"]

    results = {}
    for i, name in enumerate(target_names):
        y = Y_train[:, i]
        if np.std(y) < 1e-8:
            print(f"  {name}: constant target, skipping")
            continue

        sr = SymbolicRegressor(
            population_size=1000,
            generations=20,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            verbose=0,
            parsimony_coefficient=0.01,
            random_state=42,
        )
        sr.fit(X_train, y)
        score = sr.score(X_train, y)
        print(f"  {name}: R²={score:.4f}, expr={sr._program}")
        results[name] = {"score": score, "expr": str(sr._program)}

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_dataset_summary(ds, out="phoxonic_dataset_summary.png"):
    """Gap distribution scatter: gap_em vs gap_ac, colored by fill."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ge = ds["gap_em"]
    ga = ds["gap_ac"]
    fills = ds["fills"]

    sc = axes[0].scatter(ge, ga, c=fills, cmap="viridis", s=5, alpha=0.6)
    axes[0].set_xlabel("Photonic TE gap width")
    axes[0].set_ylabel("Phononic gap width")
    axes[0].set_title("Gap distribution")
    plt.colorbar(sc, ax=axes[0], label="Fill fraction")

    axes[1].hist(ge[ge > 0.001], bins=50, alpha=0.7, label="Photonic")
    axes[1].hist(ga[ga > 0.001], bins=50, alpha=0.7, label="Phononic")
    axes[1].set_xlabel("Gap width")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Gap width distributions")
    axes[1].legend()

    axes[2].hist(fills, bins=50, alpha=0.7)
    axes[2].set_xlabel("Fill fraction")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Geometry fill fractions")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_pareto(ds, out="phoxonic_pareto.png"):
    """Pareto frontier of simultaneously achievable gaps."""
    ge = ds["gap_em"]
    ga = ds["gap_ac"]

    # Find Pareto front
    mask = (ge > 0.001) & (ga > 0.001)
    ge_dual, ga_dual = ge[mask], ga[mask]

    if len(ge_dual) < 2:
        print("Too few dual-gap samples for Pareto plot")
        return

    # Sort by photonic gap and find non-dominated
    order = np.argsort(ge_dual)
    ge_s, ga_s = ge_dual[order], ga_dual[order]
    pareto_ge, pareto_ga = [ge_s[-1]], [ga_s[-1]]
    max_ga = ga_s[-1]
    for i in range(len(ge_s) - 2, -1, -1):
        if ga_s[i] > max_ga:
            max_ga = ga_s[i]
            pareto_ge.insert(0, ge_s[i])
            pareto_ga.insert(0, ga_s[i])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ge_dual, ga_dual, s=10, alpha=0.4, label="Dual-gap samples")
    ax.plot(pareto_ge, pareto_ga, "r-o", ms=4, label="Pareto frontier")
    ax.set_xlabel("Photonic TE gap width")
    ax.set_ylabel("Phononic gap width")
    ax.set_title("Pareto frontier: simultaneous gaps")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_mlp_results(ds, out="phoxonic_mlp_results.png"):
    """Scatter plots: predicted vs actual for the MLP."""
    ckpt = torch.load("phoxonic_mlp.pt", weights_only=False)
    splits = ds["splits"]

    X_test = torch.tensor(build_features(ds, splits["test"]),
                          dtype=torch.float32)
    Y_test = build_targets(ds, splits["test"])

    x_mean, x_std = ckpt["x_mean"], ckpt["x_std"]
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

    model = GapMLP(X_test.shape[1], Y_test.shape[1])
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        pred_n = model((X_test - x_mean) / x_std)
        pred = (pred_n * y_std + y_mean).numpy()

    target_names = ["gap_em", "mid_em", "gap_ac", "mid_ac",
                    "ratio_em", "ratio_ac"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, (ax, name) in enumerate(zip(axes.flat, target_names)):
        ax.scatter(Y_test[:, i], pred[:, i], s=5, alpha=0.4)
        lo = min(Y_test[:, i].min(), pred[:, i].min())
        hi = max(Y_test[:, i].max(), pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", alpha=0.5)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ss_res = np.sum((Y_test[:, i] - pred[:, i]) ** 2)
        ss_tot = np.sum((Y_test[:, i] - Y_test[:, i].mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        ax.set_title(f"{name} (R²={r2:.3f})")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_bottleneck(ds, out="phoxonic_bottleneck.png"):
    """Latent space visualization of the autoencoder."""
    ckpt = torch.load("phoxonic_autoencoder.pt", weights_only=False)
    latent_dim = ckpt["latent_dim"]
    N, n_k = ds["N"], ds["n_k"]
    n_bands_em = ds["bands_em"].shape[2]
    n_bands_ac = ds["bands_ac"].shape[2]

    model = PhoxonicAutoencoder(N, n_k, n_bands_em, n_bands_ac, latent_dim)
    model.load_state_dict(ckpt["model"])
    model.eval()

    S_all = torch.tensor(ds["s_grids"], dtype=torch.float32)
    with torch.no_grad():
        z_all = model.encode(S_all).numpy()

    fills = ds["fills"]
    ge = ds["gap_em"]
    ga = ds["gap_ac"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if latent_dim >= 2:
        sc = axes[0].scatter(z_all[:, 0], z_all[:, 1], c=fills,
                             cmap="viridis", s=5, alpha=0.5)
        axes[0].set_xlabel("z₁")
        axes[0].set_ylabel("z₂")
        axes[0].set_title("Latent space (fill fraction)")
        plt.colorbar(sc, ax=axes[0])

        sc = axes[1].scatter(z_all[:, 0], z_all[:, 1], c=ge,
                             cmap="hot", s=5, alpha=0.5)
        axes[1].set_xlabel("z₁")
        axes[1].set_ylabel("z₂")
        axes[1].set_title("Latent space (photonic gap)")
        plt.colorbar(sc, ax=axes[1])

        sc = axes[2].scatter(z_all[:, 0], z_all[:, 1], c=ga,
                             cmap="cool", s=5, alpha=0.5)
        axes[2].set_xlabel("z₁")
        axes[2].set_ylabel("z₂")
        axes[2].set_title("Latent space (phononic gap)")
        plt.colorbar(sc, ax=axes[2])
    else:
        axes[0].scatter(z_all[:, 0], fills, s=5, alpha=0.5)
        axes[0].set_xlabel("z₁")
        axes[0].set_ylabel("Fill fraction")
        axes[1].scatter(z_all[:, 0], ge, s=5, alpha=0.5)
        axes[1].set_xlabel("z₁")
        axes[1].set_ylabel("Photonic gap")
        axes[2].scatter(z_all[:, 0], ga, s=5, alpha=0.5)
        axes[2].set_xlabel("z₁")
        axes[2].set_ylabel("Phononic gap")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # Bottleneck dimensionality analysis via PCA of latent embeddings
    from sklearn.decomposition import PCA
    if latent_dim > 1:
        pca = PCA()
        pca.fit(z_all)
        var_explained = pca.explained_variance_ratio_
        cum_var = np.cumsum(var_explained)
        print(f"Latent dim={latent_dim}")
        print(f"PCA variance explained: {var_explained[:min(5, latent_dim)]}")
        print(f"Cumulative: {cum_var[:min(5, latent_dim)]}")
        dim_90 = int(np.searchsorted(cum_var, 0.9)) + 1
        print(f"Dimensions for 90% variance: {dim_90}")


def plot_training_curves(train_losses, val_losses, name, out):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{name} training")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["mlp", "autoencoder", "symbolic",
                                         "visualize"])
    parser.add_argument("--dataset", type=str, default="phoxonic_dataset.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--recon-weight", type=float, default=0.1)
    args = parser.parse_args()

    print(f"Loading {args.dataset}...")
    ds = load_dataset(args.dataset)
    n = len(ds["fills"])
    n_dual = ds["has_dual"].sum()
    print(f"  {n} samples, {n_dual} dual-gap ({100*n_dual/n:.1f}%)")

    if args.mode == "mlp":
        print("\n--- Target A: MLP gap regression ---")
        model, tl, vl, *_ = train_mlp(ds, epochs=args.epochs, lr=args.lr,
                                       batch_size=args.batch_size,
                                       hidden=args.hidden)
        plot_training_curves(tl, vl, "MLP", "phoxonic_mlp_training.png")
        plot_mlp_results(ds)

    elif args.mode == "autoencoder":
        print("\n--- Target B: Autoencoder ---")
        model, tl, vl = train_autoencoder(
            ds, latent_dim=args.latent_dim, epochs=args.epochs,
            lr=args.lr, batch_size=args.batch_size,
            recon_weight=args.recon_weight)
        plot_training_curves(tl, vl, "Autoencoder",
                             "phoxonic_ae_training.png")
        plot_bottleneck(ds)

    elif args.mode == "symbolic":
        print("\n--- Symbolic Regression ---")
        run_symbolic_regression(ds)

    elif args.mode == "visualize":
        print("\n--- Visualization ---")
        plot_dataset_summary(ds)
        plot_pareto(ds)
        if Path("phoxonic_mlp.pt").exists():
            plot_mlp_results(ds)
        if Path("phoxonic_autoencoder.pt").exists():
            plot_bottleneck(ds)


if __name__ == "__main__":
    main()
