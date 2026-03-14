"""Draw CANN v3 architecture diagrams using matplotlib."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

COLORS = {
    "input":    "#4A90D9",
    "encoder":  "#7B68EE",
    "fc":       "#2ECC71",
    "act":      "#E67E22",
    "cross":    "#E74C3C",
    "head":     "#F39C12",
    "output":   "#1ABC9C",
}


def _box(ax, x, y, w, h, text, color, fontsize=8, text_color="white"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor="black", lw=1.2,
                         zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold", zorder=4)


def _arrow(ax, x0, y0, x1, y1, color="black", style="-|>", lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=2)


def _label(ax, x, y, text, fontsize=7, color="gray"):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=color, style="italic", zorder=4)


# ── DualGridCANN (independent) ──────────────────────────────────

def draw_dual(save_path="cann_v3_dual_arch.png"):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("DualGridCANN — Independent Dual Heads", fontsize=13, pad=15)

    W, H = 1.8, 0.6

    # Input
    _box(ax, 1, 3.5, 2.0, 0.8, "ε grid\n(B,1,32,32)", COLORS["input"], fontsize=8)

    # Encoder
    _box(ax, 3.5, 3.5, 1.8, 0.8, "LearnedKernel\nEncoder", COLORS["encoder"])
    _arrow(ax, 2.0, 3.5, 2.6, 3.5)
    _label(ax, 2.3, 3.8, "cos/sin\nG·r kernels", fontsize=6)

    # Shape label
    _label(ax, 4.7, 3.5, "(B, 97)", fontsize=7, color="black")

    # Fork
    _arrow(ax, 4.4, 3.5, 5.3, 5.5)
    _arrow(ax, 4.4, 3.5, 5.3, 1.5)

    # ── Photonic branch (top) ──
    yp = 5.5
    _box(ax, 5.8, yp, W, H, "FC1 + SiLU", COLORS["fc"])
    _arrow(ax, 6.7, yp, 7.3, yp)
    _box(ax, 7.8, yp, W, H, "FC2 + Smooth\nInverse", COLORS["act"])
    _arrow(ax, 8.7, yp, 9.1, yp)
    _box(ax, 9.7, yp, W, H, "FC3 + Softplus", COLORS["fc"])

    # Photonic output
    _arrow(ax, 10.6, yp, 10.6, yp - 0.7)
    _box(ax, 10.6, yp - 1.1, W, H, "BandOrdering\nHead", COLORS["head"])
    _arrow(ax, 10.6, yp - 1.5, 10.6, yp - 2.0)
    _box(ax, 10.6, yp - 2.4, 1.6, 0.6, "ω_phot\n(B,31,6)", COLORS["output"])

    ax.text(5.8, yp + 0.6, "Photonic Branch", ha="center", fontsize=9,
            fontweight="bold", color=COLORS["fc"])

    # ── Phononic branch (bottom) ──
    yn = 1.5
    _box(ax, 5.8, yn, W, H, "FC1 + SiLU", COLORS["fc"])
    _arrow(ax, 6.7, yn, 7.3, yn)
    _box(ax, 7.8, yn, W, H, "FC2 + Smooth\nInverse", COLORS["act"])
    _arrow(ax, 8.7, yn, 9.1, yn)
    _box(ax, 9.7, yn, W, H, "FC3 + Softplus", COLORS["fc"])

    _arrow(ax, 10.6, yn, 10.6, yn + 0.7)
    _box(ax, 10.6, yn + 1.1, W, H, "BandOrdering\nHead", COLORS["head"])
    _arrow(ax, 10.6, yn + 1.5, 10.6, yn + 2.0)
    _box(ax, 10.6, yn + 2.4, 1.6, 0.6, "ω_phon\n(B,31,10)", COLORS["output"])

    ax.text(5.8, yn - 0.6, "Phononic Branch", ha="center", fontsize=9,
            fontweight="bold", color=COLORS["fc"])

    # Physics annotations
    _label(ax, 7.8, yp + 0.5, "≈ ε⁻¹ inversion", fontsize=6, color=COLORS["act"])
    _label(ax, 7.8, yn - 0.5, "≈ M⁻¹Γ inversion", fontsize=6, color=COLORS["act"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ── DualGridCANN_cross ───────────────────────────────────────────

def draw_cross(save_path="cann_v3_cross_arch.png"):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("DualGridCANN_cross — Cross-Gated Dual Heads", fontsize=13, pad=15)

    W, H = 1.8, 0.6
    yp, yn = 6.0, 2.0

    # Input
    _box(ax, 0.5, 4.0, 2.0, 0.8, "ε grid\n(B,1,32,32)", COLORS["input"])
    _arrow(ax, 1.5, 4.0, 2.2, 4.0)

    # Encoder
    _box(ax, 3.0, 4.0, 1.8, 0.8, "LearnedKernel\nEncoder", COLORS["encoder"])
    _label(ax, 3.0, 4.7, "Fourier structure\nfactor", fontsize=6)
    _label(ax, 4.2, 4.0, "(B,97)", fontsize=7, color="black")

    # Fork
    _arrow(ax, 3.9, 4.0, 4.6, yp)
    _arrow(ax, 3.9, 4.0, 4.6, yn)

    # ── Photonic (top) ──
    _box(ax, 5.2, yp, W, H, "FC1 + SiLU", COLORS["fc"])
    _arrow(ax, 6.1, yp, 6.5, yp)
    _box(ax, 7.1, yp, W, H, "FC2 + Smooth\nInverse", COLORS["act"])

    # ── Phononic (bottom) ──
    _box(ax, 5.2, yn, W, H, "FC1 + SiLU", COLORS["fc"])
    _arrow(ax, 6.1, yn, 6.5, yn)
    _box(ax, 7.1, yn, W, H, "FC2 + Smooth\nInverse", COLORS["act"])

    # ── Cross-attention ──
    cx = 8.8
    _arrow(ax, 8.0, yp, cx - 0.9, yp)
    _arrow(ax, 8.0, yn, cx - 0.9, yn)
    cross_h = (yp - yn) + 1.2
    cross_box = FancyBboxPatch((cx - 0.9, yn - 0.6), 1.8, cross_h,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS["cross"], edgecolor="black",
                                lw=1.5, alpha=0.9, zorder=3)
    ax.add_patch(cross_box)
    ax.text(cx, 4.0, "Cross\nAttention\nBlock", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold", zorder=4)
    ax.text(cx, 4.0 - 1.2, "σ(W·h) ⊙ V·h\n+ residual", ha="center", va="center",
            fontsize=6, color="#FFD0D0", zorder=4)

    # Arrows between cross-attn and branches
    _arrow(ax, 8.0, yp, cx - 0.9, yp, color=COLORS["cross"])
    _arrow(ax, 8.0, yn, cx - 0.9, yn, color=COLORS["cross"])

    _arrow(ax, cx + 0.9, yp, cx + 1.3, yp, color=COLORS["cross"])
    _arrow(ax, cx + 0.9, yn, cx + 1.3, yn, color=COLORS["cross"])

    # ── Post cross-attention ──
    x_post = 10.6
    _box(ax, x_post, yp, W, H, "FC3 + Softplus", COLORS["fc"])
    _box(ax, x_post, yn, W, H, "FC3 + Softplus", COLORS["fc"])

    # Outputs — photonic (top branch)
    x_out = 12.0
    _arrow(ax, 11.5, yp, x_out - 0.1, yp)
    _box(ax, x_out, yp, W, H, "BandOrdering\nHead", COLORS["head"])
    _arrow(ax, x_out + 0.9, yp, x_out + 1.3, yp)
    _box(ax, x_out + 1.9, yp, 1.5, 0.5, "ω_phot\n(B,31,6)", COLORS["output"])

    # Outputs — phononic (bottom branch)
    _arrow(ax, 11.5, yn, x_out - 0.1, yn)
    _box(ax, x_out, yn, W, H, "BandOrdering\nHead", COLORS["head"])
    _arrow(ax, x_out + 0.9, yn, x_out + 1.3, yn)
    _box(ax, x_out + 1.9, yn, 1.5, 0.5, "ω_phon\n(B,31,10)", COLORS["output"])

    # Branch labels
    ax.text(5.2, yp + 0.6, "Photonic Branch", ha="center", fontsize=9,
            fontweight="bold", color=COLORS["fc"])
    ax.text(5.2, yn - 0.6, "Phononic Branch", ha="center", fontsize=9,
            fontweight="bold", color=COLORS["fc"])

    # Physics annotations
    _label(ax, 7.1, yp + 0.5, "≈ ε⁻¹", fontsize=6, color=COLORS["act"])
    _label(ax, 7.1, yn - 0.5, "≈ M⁻¹Γ", fontsize=6, color=COLORS["act"])
    _label(ax, x_post, yp + 0.5, "ω ≥ 0", fontsize=6, color=COLORS["fc"])
    _label(ax, x_post, yn - 0.5, "ω ≥ 0", fontsize=6, color=COLORS["fc"])

    # Legend
    legend_items = [
        mpatches.Patch(color=COLORS["input"], label="Input"),
        mpatches.Patch(color=COLORS["encoder"], label="Fourier Encoder"),
        mpatches.Patch(color=COLORS["fc"], label="FC + Activation"),
        mpatches.Patch(color=COLORS["act"], label="SmoothInverse (≈ inversion)"),
        mpatches.Patch(color=COLORS["cross"], label="Cross-Attention"),
        mpatches.Patch(color=COLORS["head"], label="BandOrderingHead (ω₁≤ω₂≤…)"),
        mpatches.Patch(color=COLORS["output"], label="Output"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=7,
              framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    draw_dual()
    draw_cross()
