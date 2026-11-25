"""Figures for the report. Light, thin, one accent."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family": "Inter",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Inter",
    "mathtext.it": "Inter:italic",
    "axes.edgecolor": "#c9c9c9",
    "axes.linewidth": 0.6,
    "xtick.color": "#6f6f6f",
    "ytick.color": "#6f6f6f",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})

INK, GREY, LIGHT, ACC = "#111111", "#8a8a8a", "#dcdcdc", "#a33a2e"
CMAP = LinearSegmentedColormap.from_list("eca", ["#a33a2e", "#f2e2df", "#ffffff",
                                                 "#e4eee8", "#2f6b4f"])
ORDER = ["v5small", "v5nano", "qwen3_06b", "v4", "v3", "e5large", "bgem3"]
SHORT = {"v5small": "v5-small", "v5nano": "v5-nano", "qwen3_06b": "Qwen3-0.6B",
         "v4": "v4", "v3": "v3", "e5large": "e5-large", "bgem3": "bge-m3"}


def load(p):
    return json.loads(Path(p).read_text())


def ratio_matrix(D, method="procrustes"):
    nat = {k: v["test_mean"] for k, v in D["native"].items()}
    M = np.full((len(ORDER), len(ORDER)), np.nan)
    for i, s in enumerate(ORDER):
        for j, t in enumerate(ORDER):
            if i == j:
                continue
            k = f"{s}->{t}"
            if k in D["pairs"] and method in D["pairs"][k]:
                M[i, j] = D["pairs"][k][method]["test_mean"] / nat[s]
    return M


def fig_heatmap(D, out):
    M = ratio_matrix(D)
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    v = 0.5
    im = ax.imshow(M, cmap=CMAP, vmin=1 - v, vmax=1 + v)
    ax.set_xticks(range(len(ORDER)), [SHORT[m] for m in ORDER], rotation=45, ha="right")
    ax.set_yticks(range(len(ORDER)), [SHORT[m] for m in ORDER])
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    ax.set_title("ours, orthogonal Procrustes", fontsize=8.5, color=INK, loc="left", pad=6)
    for i in range(len(ORDER)):
        for j in range(len(ORDER)):
            if i == j:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#f7f7f7", zorder=2))
                continue
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.4,
                        color="white" if abs(M[i, j] - 1) > 0.33 else INK, zorder=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("adapted / source", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.outline.set_visible(False)
    fig.savefig(out)
    plt.close(fig)


def fig_predictor(D, out):
    nat = {k: v["test_mean"] for k, v in D["native"].items()}
    rec = {}
    for k, p in D["pairs"].items():
        s, t = k.split("->")
        rec[(s, t)] = dict(cka=p["features"]["cka"],
                           y=p["procrustes"]["test_mean"] / nat[s],
                           expanding=nat[s] <= nat[t])
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for (s, t), r in rec.items():
        ax.scatter(r["cka"], r["y"], s=22, zorder=3,
                   facecolor="none" if r["expanding"] else ACC,
                   edgecolor=GREY if r["expanding"] else ACC, linewidth=0.8)
    # connect the two directions of the pairs with the largest asymmetry
    asym = []
    for (s, t), r in rec.items():
        if (t, s) in rec:
            asym.append((abs(r["y"] - rec[(t, s)]["y"]), s, t))
    for d, s, t in sorted(asym, reverse=True)[:4]:
        a, b = rec[(s, t)], rec[(t, s)]
        ax.annotate("", xy=(a["cka"], a["y"]), xytext=(b["cka"], b["y"]),
                    arrowprops=dict(arrowstyle="-", color="#c4c4c4", lw=0.8, zorder=1))
        ax.plot([a["cka"], b["cka"]], [a["y"], b["y"]], color="#c4c4c4", lw=0.8, zorder=1)
    ax.axhline(1.0, color=LIGHT, lw=0.8, zorder=0)
    ax.set_xlabel("linear CKA between the two spaces")
    ax.set_ylabel("adapted / source nDCG@10")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.96, "filled: compressing ($d_s>d_t$)", transform=ax.transAxes,
            fontsize=7, color=ACC, va="top")
    ax.text(0.02, 0.89, "open: expanding", transform=ax.transAxes, fontsize=7, color=GREY,
            va="top")
    fig.savefig(out)
    plt.close(fig)


def fig_lowrank(D, out):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from align.data import MODELS as _M
    lr = {k: v for k, v in D["lowrank"].items()
          if min(_M[k.split("->")[0]]["dim"], _M[k.split("->")[1]]["dim"]) >= 1024}
    ks = sorted({int(x) for r in lr.values() for x in r})
    full = {k: r[str(max(int(x) for x in r))] for k, r in lr.items()}
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for k, r in lr.items():
        xs = [x for x in ks if str(x) in r]
        ys = [r[str(x)] / full[k] for x in xs]
        ax.plot(xs, ys, color="#d8d8d8", lw=0.6, zorder=1)
    xs = [k for k in ks if all(str(k) in r for r in lr.values())]
    ys = [np.mean([r[str(k)] / full[kk] for kk, r in lr.items()]) for k in xs]
    ax.plot(xs, ys, color=INK, lw=1.4, marker="o", ms=2.5, zorder=3)
    ax.axvline(256, color=ACC, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.text(256 * 1.1, 0.06, "256", color=ACC, fontsize=7)
    ax.axhline(1.0, color=LIGHT, lw=0.8, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("principal components kept per model")
    ax.set_ylabel("nDCG@10 relative to full rank")
    ax.set_ylim(0, 1.42)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out)
    plt.close(fig)


def fig_calsize(D, out):
    cs = D["calsize"]
    ns = sorted({int(x) for r in cs.values() for x in r})
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for r in cs.values():
        ax.plot([n for n in ns if str(n) in r], [r[str(n)] for n in ns if str(n) in r],
                color="#d8d8d8", lw=0.6, zorder=1)
    med = [np.median([r[str(n)] for r in cs.values() if str(n) in r]) for n in ns]
    ax.plot(ns, med, color=INK, lw=1.4, marker="o", ms=2.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("paired calibration texts")
    ax.set_ylabel("nDCG@10")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out)
    plt.close(fig)


def fig_chain(D, out):
    ch = D["chain"]
    rel = np.array([c["chained"] / c["direct"] for c in ch if c["direct"] > 0])
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    ax.hist(rel, bins=40, color="#e8e8e8", edgecolor="#c9c9c9", linewidth=0.4)
    ax.axvline(1.0, color=LIGHT, lw=0.9)
    ax.axvline(np.median(rel), color=ACC, lw=1.1)
    ax.text(np.median(rel) - 0.012, ax.get_ylim()[1] * 0.92, f"median {np.median(rel):.3f}",
            color=ACC, fontsize=7, ha="right")
    ax.set_xlabel("chained A\u2192B\u2192C / direct A\u2192C")
    ax.set_ylabel("triples")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out)
    plt.close(fig)


def fig_main(D, out):
    """Three numbers per pair: source, adapted, target."""
    nat = {k: v["test_mean"] for k, v in D["native"].items()}
    rows = []
    for k, p in D["pairs"].items():
        s, t = k.split("->")
        if "procrustes" in p:
            rows.append((s, t, nat[s], p["procrustes"]["test_mean"], nat[t]))
    rows.sort(key=lambda r: -(r[3] / r[2]))
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    y = np.arange(len(rows))
    for i, (s, t, a, b, c) in enumerate(rows):
        ax.plot([a, c], [i, i], color=LIGHT, lw=0.7, zorder=1)
        ax.scatter([a], [i], s=9, color=GREY, zorder=3)
        ax.scatter([c], [i], s=9, color="#2f6b4f", zorder=3)
        ax.scatter([b], [i], s=14, color=ACC, zorder=4, marker="D")
    ax.set_yticks(y, [f"{SHORT[s]}\u2192{SHORT[t]}" for s, t, *_ in rows], fontsize=4.6)
    ax.set_xlabel("nDCG@10")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(out)
    plt.close(fig)


def fig_heatmap_pair(D, out, v=0.5, annotate=True, size=(7.2, 3.0)):
    """Theirs and ours on one colour scale, same size."""
    panels = [("affine", "theirs, affine (Plato's Key)"),
              ("procrustes", "ours, orthogonal Procrustes")]
    fig, axes = plt.subplots(1, 2, figsize=size, sharey=True)
    for ax, (meth, title) in zip(axes, panels):
        M = ratio_matrix(D, meth)
        im = ax.imshow(M, cmap=CMAP, vmin=1 - v, vmax=1 + v)
        ax.set_xticks(range(len(ORDER)), [SHORT[m] for m in ORDER], rotation=45, ha="right")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=6)
        for i in range(len(ORDER)):
            for j in range(len(ORDER)):
                if i == j:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#f7f7f7", zorder=2))
                    continue
                if annotate and not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.0,
                            color="white" if abs(M[i, j] - 1) > 0.33 else INK, zorder=3)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(ORDER)), [SHORT[m] for m in ORDER])
    axes[0].set_ylabel("source")
    axes[0].set_xlabel("target")
    axes[1].set_xlabel("target")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("adapted / source", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.outline.set_visible(False)
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--dir", default="figs")
    a = ap.parse_args()
    d = Path(a.dir)
    d.mkdir(exist_ok=True)
    D = load(a.results)
    fig_heatmap_pair(D, d / "heatmap.pdf")
    fig_heatmap_pair(D, d / "heatmap_clean.pdf", annotate=False, size=(6.4, 2.7), v=0.4)
    fig_predictor(D, d / "predictor.pdf")
    if "lowrank" in D:
        fig_lowrank(D, d / "lowrank.pdf")
    if "calsize" in D:
        fig_calsize(D, d / "calsize.pdf")
    if "chain" in D:
        fig_chain(D, d / "chain.pdf")
    print("figures written to", d)
