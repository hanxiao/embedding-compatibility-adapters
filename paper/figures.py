"""Publication-grade vector PDF figures for the report and high-DPI PNGs for README."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from align.data import MODELS

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#c9cfd8",
    "axes.linewidth": 0.65,
    "xtick.color": "#555f6e",
    "ytick.color": "#555f6e",
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "axes.labelsize": 8.8,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.6,
    "legend.frameon": False,
    "figure.dpi": 220,
    "savefig.bbox": "tight",
    "figure.facecolor": "#ffffff",
})

INK, GREY, LIGHT = "#111111", "#6b7280", "#e2e6ec"
ACC = "#b5382a"      # terracotta
FOREST = "#2a6f4e"   # forest
SLATE = "#3b5b7c"    # slate

CMAP = LinearSegmentedColormap.from_list(
    "eca", ["#a33a2e", "#f2e2df", "#ffffff", "#e4eee8", "#2a6f4e"]
)

ORDER = ["v5small", "v5nano", "qwen3_06b", "arctic_m", "v4", "v3", "bgem3", "e5large"]
SHORT = {
    "v5small": "v5-small",
    "v5nano": "v5-nano",
    "qwen3_06b": "Qwen3-0.6B",
    "arctic_m": "Arctic-m",
    "v4": "v4",
    "v3": "v3",
    "bgem3": "bge-m3",
    "e5large": "e5-large",
}


def load(p):
    return json.loads(Path(p).read_text())


def ratio_matrix(D, method="procrustes"):
    nat = {k: v["test_mean"] for k, v in D["native"].items()}
    models = [m for m in ORDER if m in nat]
    M = np.full((len(models), len(models)), np.nan)
    for i, s in enumerate(models):
        for j, t in enumerate(models):
            if i == j:
                continue
            k = f"{s}->{t}"
            if k in D["pairs"] and method in D["pairs"][k]:
                M[i, j] = D["pairs"][k][method]["test_mean"] / nat[s]
    return models, M


def fig_heatmap_pair(D, out_pdf, out_png=None):
    """Side-by-side 8x8 retention heatmaps: Plato's Key Affine vs Ours Orthogonal Procrustes."""
    panels = [
        ("affine", "Plato's Key, affine (theirs)"),
        ("procrustes", "Orthogonal Procrustes (ours)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.65), sharey=True)
    for ax, (meth, title) in zip(axes, panels):
        models, M = ratio_matrix(D, meth)
        im = ax.imshow(M, cmap=CMAP, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(models)), [SHORT[m] for m in models], rotation=40, ha="right")
        ax.set_title(title, fontsize=9.0, fontweight="bold", color=INK, loc="left", pad=7)
        for i in range(len(models)):
            for j in range(len(models)):
                if i == j:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#f4f6f8", zorder=2))
                    continue
                if not np.isnan(M[i, j]):
                    v = np.clip(M[i, j], 0.0, 1.0)
                    r, g, b, _ = CMAP(v)
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    ax.text(
                        j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.4,
                        color="white" if lum < 0.52 else INK, zorder=3,
                    )
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(models)), [SHORT[m] for m in models])
    axes[0].set_ylabel("source (query)")
    axes[0].set_xlabel("target (frozen index)")
    axes[1].set_xlabel("target (frozen index)")
    cb = fig.colorbar(im, ax=axes, fraction=0.026, pad=0.025)
    cb.set_label("adapted / source nDCG@10", fontsize=8.0)
    cb.ax.tick_params(labelsize=7.2, length=0)
    cb.outline.set_visible(False)
    fig.savefig(out_pdf)
    if out_png:
        fig.savefig(out_png, dpi=240)
    plt.close(fig)


def fig_pair_scatter(D, out_pdf):
    """Square-axis scatter: Plato's Key Affine vs Ours Orthogonal Procrustes over all pairs."""
    nat = {k: v["test_mean"] for k, v in D["native"].items()}
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot([0, 1.05], [0, 1.05], color=LIGHT, lw=0.9, zorder=1)
    for k, p in D["pairs"].items():
        s = k.split("->")[0]
        x = p["affine"]["test_mean"] / nat[s]
        y = p["procrustes"]["test_mean"] / nat[s]
        kappa = p["features"].get("kappa_source", 0.0)
        if kappa > 1.0:
            col, fc, sz = ACC, ACC, 30
        elif kappa > 0.8:
            col, fc, sz = SLATE, SLATE, 22
        else:
            col, fc, sz = GREY, "none", 18
        ax.scatter(x, y, s=sz, facecolor=fc, edgecolor=col, lw=0.9, zorder=3)
    ax.set_xlim(0.08, 1.05)
    ax.set_ylim(0.08, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("Plato's Key affine (adapted / source)")
    ax.set_ylabel("Orthogonal Procrustes (adapted / source)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.11, 1.01, "diagonal = parity", fontsize=7.0, color=GREY, va="top")
    ax.text(0.11, 0.94, "red: e5-large source ($\\kappa = 1.82$)", fontsize=7.0, color=ACC, va="top")
    ax.text(0.11, 0.87, "slate: v4 source ($\\kappa = 0.99$)", fontsize=7.0, color=SLATE, va="top")
    fig.savefig(out_pdf)
    plt.close(fig)


def fig_diagnostics(D, out_pdf):
    """Two-panel diagnostic figure: (left) rho vs identity score; (right) kappa vs affine gain."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.9))
    rhos, ids, ckas, kappas, gains = [], [], [], [], []
    for k, p in D["pairs"].items():
        rhos.append(p["features"]["rho"])
        ids.append(p["identity"]["test_mean"])
        ckas.append(p["features"]["cka"])
        kappas.append(p["features"]["kappa_source"])
        gains.append(p["affine"]["test_mean"] - p["procrustes"]["test_mean"])

    ax1.scatter(rhos, ids, s=22, color=SLATE, alpha=0.85, edgecolors="white", lw=0.4, zorder=3)
    r_rho = np.corrcoef(rhos, ids)[0, 1]
    ax1.axvline(0.1, color=LIGHT, lw=0.9, ls="--", zorder=1)
    ax1.set_xlabel(r"frame correlation $\rho$")
    ax1.set_ylabel("identity nDCG@10 (no adapter)")
    ax1.set_title(f"Frame alignment predicts identity ($r = {r_rho:+.2f}$)", fontsize=8.6, loc="left")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.scatter(kappas, gains, s=22, color=ACC, alpha=0.85, edgecolors="white", lw=0.4, zorder=3)
    r_kap = np.corrcoef(kappas, gains)[0, 1]
    ax2.axhline(0.0, color=LIGHT, lw=0.9, zorder=1)
    ax2.axvline(1.0, color=LIGHT, lw=0.9, ls="--", zorder=1)
    ax2.set_xlabel(r"source anisotropy $\kappa_s$")
    ax2.set_ylabel("affine gain over rotation (nDCG@10)")
    ax2.set_title(f"Anisotropy predicts affine gain ($r = {r_kap:+.2f}$)", fontsize=8.6, loc="left")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(w_pad=2.4)
    fig.savefig(out_pdf)
    plt.close(fig)


def fig_progressive_migration(D, out_pdf, out_png=None):
    """Progressive zero-downtime migration (0% -> 100%): raw cosine crash vs top-100 z-score fix."""
    dep = D["deployment"]
    fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    pcts = [0, 25, 50, 75, 100]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.0), sharey=True)

    # Left: Canonical v3 frozen index -> v5-nano queries
    v = dep["idx=v3|qry=v5nano"]["hybrid_affine"]
    y_raw = [v["uncalibrated"][str(fr)] for fr in fracs]
    y_z = [v["zscore_top100"][str(fr)] for fr in fracs]
    ax1.plot(pcts, y_raw, color=ACC, lw=1.6, marker="o", ms=4.2, ls="--")
    ax1.plot(pcts, y_z, color=FOREST, lw=1.9, marker="o", ms=4.5)
    ax1.annotate(
        f"raw dip: {y_raw[1]:.3f}", xy=(25, y_raw[1]), xytext=(36, y_raw[1] - 0.045),
        fontsize=7.4, color=ACC, arrowprops=dict(arrowstyle="->", color=ACC, lw=0.8),
    )
    ax1.text(101, y_z[-1] + 0.008, "top-100 z-score", fontsize=7.5, color=FOREST, va="bottom", ha="right")
    ax1.text(72, y_raw[3] - 0.038, "raw cosine", fontsize=7.5, color=ACC, va="top")
    ax1.set_title("jina-v3 index $\\to$ jina-v5-nano", fontsize=8.8, fontweight="bold", loc="left")
    ax1.set_xlabel("% of corpus re-indexed natively")
    ax1.set_ylabel("nDCG@10 (429 held-out queries)")
    ax1.set_xticks(pcts)
    ax1.set_ylim(0.24, 0.72)
    ax1.spines[["top", "right"]].set_visible(False)

    # Right: Mean across all 56 directed pairs
    m_raw = [np.mean([r["hybrid_affine"]["uncalibrated"][str(fr)] for r in dep.values()]) for fr in fracs]
    m_z = [np.mean([r["hybrid_affine"]["zscore_top100"][str(fr)] for r in dep.values()]) for fr in fracs]
    ax2.plot(pcts, m_raw, color=ACC, lw=1.6, marker="o", ms=4.2, ls="--")
    ax2.plot(pcts, m_z, color=FOREST, lw=1.9, marker="o", ms=4.5)
    ax2.annotate(
        f"mean dip: {m_raw[1]:.3f}", xy=(25, m_raw[1]), xytext=(36, m_raw[1] - 0.045),
        fontsize=7.4, color=ACC, arrowprops=dict(arrowstyle="->", color=ACC, lw=0.8),
    )
    ax2.text(98, m_z[-1] + 0.012, "top-100 z-score", fontsize=7.5, color=FOREST, ha="right")
    ax2.text(74, m_raw[3] - 0.068, "raw cosine", fontsize=7.5, color=ACC)
    ax2.set_title(f"Mean across all {len(dep)} directed pairs", fontsize=8.8, fontweight="bold", loc="left")
    ax2.set_xlabel("% of corpus re-indexed natively")
    ax2.set_xticks(pcts)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(out_pdf)
    if out_png:
        fig.savefig(out_png, dpi=240)
    plt.close(fig)


def fig_structure(D, out_pdf):
    """Left: 336-triple chaining (Affine vs Procrustes). Right: one-sided low-rank Affine curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 2.95))

    # Left: Chaining
    rel_a = np.array([c["chained"] / c["direct"] for c in D["chain_affine"] if c["direct"] > 0])
    rel_p = np.array([c["chained"] / c["direct"] for c in D["chain"] if c["direct"] > 0])
    bins = np.linspace(0.75, 1.05, 36)
    ax1.hist(rel_p, bins=bins, color="#d8dde4", alpha=0.65, label=f"rotation (med {np.median(rel_p):.3f})")
    ax1.hist(rel_a, bins=bins, color=FOREST, alpha=0.55, label=f"affine (med {np.median(rel_a):.3f})")
    ax1.axvline(1.0, color=LIGHT, lw=0.9)
    ax1.axvline(np.median(rel_a), color=FOREST, lw=1.2)
    ax1.set_xlabel(r"chained $A\to B\to C$ / direct $A\to C$")
    ax1.set_ylabel(f"triples (n={len(rel_a)})")
    ax1.legend(loc="upper left", fontsize=7.3)
    ax1.spines[["top", "right"]].set_visible(False)

    # Right: One-sided low-rank Affine
    lr = D["lowrank_affine"]
    ks = [16, 32, 64, 128, 256, 384, 512, 768]
    for pk, row in lr.items():
        full = D["pairs"][pk]["affine-cs"]["test_mean"]
        xs = [k for k in ks if str(k) in row]
        ys = [row[str(k)] / full for k in xs]
        ax2.plot(xs, ys, color="#dce1e8", lw=0.55, zorder=1)
    means = [
        np.mean([row[str(k)] / D["pairs"][pk]["affine-cs"]["test_mean"]
                 for pk, row in lr.items() if str(k) in row])
        for k in ks
    ]
    ax2.plot(ks, means, color=INK, lw=1.6, marker="o", ms=3.2, zorder=3)
    ax2.axvline(256, color=ACC, lw=0.85, ls="--", zorder=2)
    idx256 = ks.index(256)
    ax2.text(275, 0.14, f"k = 256 ({means[idx256]:.2f}x)", fontsize=7.4, color=ACC)
    ax2.axhline(1.0, color=LIGHT, lw=0.85, zorder=0)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(ks, [str(k) for k in ks])
    ax2.set_xlabel("principal components k per model (one-sided)")
    ax2.set_ylabel("nDCG@10 relative to full-rank affine")
    ax2.set_ylim(0.0, 1.12)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(out_pdf)
    plt.close(fig)


def fig_calsize(D, out_pdf):
    """Calibration budget curve n = 50..5000 for Affine, Procrustes, Ridge."""
    ns = [50, 100, 250, 500, 1000, 2500, 5000]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    series = [
        ("calsize_src_centred", "Source-centred (ours)", FOREST, 1.9, 0.018),
        ("calsize_affine", "Affine (Plato's Key, fixed)", ACC, 1.5, 0.004),
        ("calsize", "Orthogonal Procrustes", INK, 1.4, -0.014),
        ("calsize_ridge", "Ridge regression", SLATE, 1.2, -0.044),
    ]
    for key, label, col, lw, dy in series:
        if key not in D:
            continue
        ys = [np.mean([r[str(n)] for r in D[key].values() if str(n) in r]) for n in ns]
        ax.plot(ns, ys, color=col, lw=lw, marker="o", ms=3.5)
        y_lbl = ys[-1] + dy
        ax.plot([ns[-1] * 1.04, ns[-1] * 1.15], [ys[-1], y_lbl], color=col, lw=0.6, alpha=0.6)
        ax.text(ns[-1] * 1.20, y_lbl, f"{label} ({ys[-1]:.3f})", fontsize=7.2, color=col, va="center")
    ax.set_xscale("log")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlim(42, 42000)
    ax.set_ylim(-0.01, 0.59)
    ax.set_xlabel("paired calibration texts")
    ax.set_ylabel(f"mean nDCG@10 ({len(D['pairs'])} pairs)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="paper/results.json")
    ap.add_argument("--dir", default="paper/figs")
    ap.add_argument("--assets", default="assets")
    a = ap.parse_args()
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    ass = Path(a.assets)
    ass.mkdir(parents=True, exist_ok=True)
    root = Path(a.results).resolve().parent.parent

    D = load(a.results)
    fig_heatmap_pair(D, d / "heatmap.pdf", out_png=root / "retention-theirs-vs-ours.png")
    fig_pair_scatter(D, d / "pair_scatter.pdf")
    fig_diagnostics(D, d / "diagnostics.pdf")
    if "deployment" in D:
        fig_progressive_migration(D, d / "progressive_migration.pdf", out_png=ass / "progressive-migration.png")
    if "chain_affine" in D and "lowrank_affine" in D:
        fig_structure(D, d / "structure.pdf")
    if "calsize" in D:
        fig_calsize(D, d / "calsize.pdf")
    print("figures written to", d, "and", ass)
