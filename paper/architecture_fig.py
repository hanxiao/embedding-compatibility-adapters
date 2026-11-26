"""Architecture diagram: two canonical zero-reindex workflows (v3 <-> v5-nano).

Top row:    Offline calibration (one SVD on 5,000 paired texts, <0.5s, no GPU)
Bottom row: Online serving for BOTH standard directions:
            Left:  v5-nano queries -> frozen v3 document index (zero re-indexing)
            Right: legacy v3 queries -> new v5-nano index (backward compatibility)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": "#ffffff",
})

INK = "#111111"
MUTED = "#64748b"
BORDER = "#cfd6df"
SOFT = "#f7f9fb"
ACC = "#b5382a"
FOREST = "#2a6f4e"
SLATE = "#3b5b7c"


def box(ax, x, y, w, h, title, sub="", fc="#ffffff", ec=BORDER, lw=1.0,
        title_col=INK, sub_col=MUTED, bold=True, r=0.14):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
    ))
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
                fontsize=9.2, fontweight="bold" if bold else "normal", color=title_col, zorder=4)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=7.6, color=sub_col, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=9.2, fontweight="bold" if bold else "normal", color=title_col, zorder=4)


def arrow(ax, x1, y1, x2, y2, col="#94a3b8", lw=1.15, rad=0.0):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=11,
                        connectionstyle=f"arc3,rad={rad}"),
        zorder=2.5,
    )


def draw(out_base: Path):
    fig, ax = plt.subplots(figsize=(10.6, 4.65), dpi=220)
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.65)
    ax.axis("off")

    # ---- Top band: Offline calibration (one SVD) ----
    ax.add_patch(FancyBboxPatch(
        (0.22, 2.78), 10.16, 1.62,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor="#fafbfc", edgecolor="#e2e8f0", linewidth=0.9, zorder=1,
    ))
    ax.text(0.45, 4.18, "OFFLINE CALIBRATION  (ONCE, NO GPU, < 0.5 s)",
            ha="left", va="center", fontsize=7.8, fontweight="bold", color=MUTED, zorder=4)

    box(ax, 0.45, 3.04, 2.05, 0.72, "5,000 paired texts", "unlabeled, zero overlap", fc="#ffffff")
    box(ax, 3.05, 3.48, 2.05, 0.44, "jina-v5-nano (768d)", fc="#fff7f6", ec="#e8bfb9", title_col=ACC)
    box(ax, 3.05, 2.92, 2.05, 0.44, "jina-v3 (1024d)", fc="#f3f6fa", ec="#c2cfde", title_col=SLATE)
    box(ax, 5.75, 3.04, 2.25, 0.72, "Closed-form SVD", r"$\tilde{X}^\top \tilde{Y} = U\Sigma V^\top$",
        fc="#ffffff", ec=INK, lw=1.1)
    box(ax, 8.55, 3.04, 1.60, 0.72, "Adapter f", r"$\mu_B + s(x-\mu_A)R$",
        fc="#f2faf5", ec=FOREST, lw=1.15, title_col=FOREST)

    arrow(ax, 2.52, 3.52, 3.03, 3.68, ACC)
    arrow(ax, 2.52, 3.24, 3.03, 3.14, SLATE)
    arrow(ax, 5.12, 3.68, 5.73, 3.48, ACC)
    arrow(ax, 5.12, 3.14, 5.73, 3.28, SLATE)
    arrow(ax, 8.02, 3.40, 8.53, 3.40, FOREST, lw=1.3)

    # ---- Bottom left: Scenario 1 (Forward: v5-nano queries -> frozen v3 index) ----
    ax.add_patch(FancyBboxPatch(
        (0.22, 0.22), 4.92, 2.26,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.95, zorder=1,
    ))
    ax.text(0.45, 2.26, "SCENARIO 1  (UPGRADE QUERIES, ZERO RE-INDEX)",
            ha="left", va="center", fontsize=7.6, fontweight="bold", color=ACC, zorder=4)

    box(ax, 0.45, 1.32, 1.30, 0.64, "New query", "jina-v5-nano", fc="#fff7f6", ec="#e8bfb9", title_col=ACC)
    box(ax, 2.08, 1.32, 1.18, 0.64, "f(q)", r"$\mathrm{v5n}\to\mathrm{v3}$", fc="#f2faf5", ec=FOREST, title_col=FOREST)
    box(ax, 0.45, 0.42, 2.81, 0.60, "Frozen v3 document index", "millions of vectors untouched",
        fc="#f3f6fa", ec=SLATE, title_col=SLATE)
    box(ax, 3.62, 0.84, 1.30, 0.72, "Cosine top-k", "nDCG@10", fc="#ffffff", ec=INK, lw=1.05)

    arrow(ax, 1.77, 1.64, 2.06, 1.64, ACC)
    arrow(ax, 3.28, 1.64, 3.60, 1.34, FOREST)
    arrow(ax, 3.28, 0.72, 3.60, 1.06, SLATE)

    # ---- Bottom right: Scenario 2 (Reverse: legacy v3 queries -> new v5-nano index) ----
    ax.add_patch(FancyBboxPatch(
        (5.46, 0.22), 4.92, 2.26,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.95, zorder=1,
    ))
    ax.text(5.69, 2.26, "SCENARIO 2  (REVERSE: LEGACY QUERIES ON NEW INDEX)",
            ha="left", va="center", fontsize=7.6, fontweight="bold", color=SLATE, zorder=4)

    box(ax, 5.69, 1.32, 1.30, 0.64, "Legacy query", "jina-v3", fc="#f3f6fa", ec="#c2cfde", title_col=SLATE)
    box(ax, 7.32, 1.32, 1.18, 0.64, "g(q)", r"$\mathrm{v3}\to\mathrm{v5n}$", fc="#f2faf5", ec=FOREST, title_col=FOREST)
    box(ax, 5.69, 0.42, 2.81, 0.60, "New v5-nano document index", "native 768d vectors",
        fc="#fff7f6", ec="#e8bfb9", title_col=ACC)
    box(ax, 8.86, 0.84, 1.30, 0.72, "Cosine top-k", "nDCG@10", fc="#ffffff", ec=INK, lw=1.05)

    arrow(ax, 7.01, 1.64, 7.30, 1.64, SLATE)
    arrow(ax, 8.52, 1.64, 8.84, 1.34, FOREST)
    arrow(ax, 8.52, 0.72, 8.84, 1.06, ACC)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print("wrote", out_base)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    draw(root / "architecture-pipeline")
