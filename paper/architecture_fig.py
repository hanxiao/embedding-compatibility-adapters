"""System architecture: two models, one adapter, one index that is never re-encoded."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "Inter",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Inter",
    "mathtext.it": "Inter:italic",
    "mathtext.bf": "Inter",
})

INK, GREY, LGREY, FILL, ACC = "#111111", "#8a8a8a", "#c9c9c9", "#f6f6f6", "#a33a2e"

fig = plt.figure(figsize=(10.0, 3.4), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 34)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, label, sub=None, dashed=False, accent=False, fs=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.55",
                                facecolor="none" if dashed else FILL,
                                edgecolor=ACC if accent else "#dedede",
                                lw=1.0 if accent else 0.7,
                                ls=(0, (2.6, 2.2)) if dashed else "-"))
    if sub:
        ax.text(x + w / 2, y + h / 2 + 0.85, label, ha="center", va="center",
                fontsize=fs, color=ACC if accent else INK)
        ax.text(x + w / 2, y + h / 2 - 1.05, sub, ha="center", va="center",
                fontsize=7.5, color=GREY)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=ACC if accent else INK)


def cards(x, y, label, n=3, w=2.2, h=1.0, gap=0.34):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * 0.16, y + i * gap), w, h, facecolor=FILL,
                               edgecolor="#dedede", lw=0.6, zorder=3 - i))
    ax.text(x + w / 2 + 0.16, y - 1.0, label, fontsize=8, color=GREY,
            ha="center", va="top")


def arrow(p1, p2, color=INK, lw=1.1, ms=8):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def route(pts, color=INK, lw=1.1, ms=8):
    if len(pts) > 2:
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
                color=color, lw=lw, solid_capstyle="butt", zorder=2)
    arrow(pts[-2], pts[-1], color=color, lw=lw, ms=ms)


def vec(x, y, n=3, w=1.35, h=1.35):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * w, y), w, h, facecolor=FILL,
                               edgecolor="#dedede", lw=0.6))


# ---------------------------------------------------------------- fit, once
cards(2.0, 27.4, "texts")
ax.plot([7.4, 9.4], [28.4, 28.4], color=INK, lw=1.1)
ax.plot([9.4, 9.4], [24.6, 32.2], color=INK, lw=1.1)
arrow((9.4, 32.2), (11.8, 32.2))
arrow((9.4, 24.6), (11.8, 24.6))

box(12.0, 29.6, 8.0, 5.2, "$A$", sub="old model")
box(12.0, 21.9, 8.0, 5.2, "$B$", sub="new model")

vec(23.0, 31.4)
ax.text(25.0, 30.9, "$X$", fontsize=10.5, color=INK, ha="center", va="top")
vec(23.0, 23.7)
ax.text(25.0, 23.2, "$Y$", fontsize=10.5, color=INK, ha="center", va="top")
arrow((20.2, 32.2), (22.8, 32.2))
arrow((20.2, 24.6), (22.8, 24.6))

box(31.0, 23.2, 7.2, 9.4, "$M = X^{\\mathsf{T}}Y$", fs=9.5)
arrow((27.2, 32.1), (30.8, 29.4))
arrow((27.2, 24.5), (30.8, 26.4))

box(41.0, 26.4, 6.4, 4.0, "SVD", fs=10)
arrow((38.4, 27.9), (40.8, 27.9))
ax.text(41.0, 31.6, "fit once", fontsize=8, color=GREY, va="bottom")

# ---------------------------------------------------------------- query, per query
cards(2.0, 13.4, "queries", n=2)
arrow((7.4, 14.4), (11.8, 14.4))

box(12.0, 11.8, 8.0, 5.2, "$A$", sub="old model")
vec(23.0, 13.5)
ax.text(24.6, 12.7, "$q$", fontsize=10.5, color=INK, ha="center", va="top")
arrow((20.2, 14.4), (22.8, 14.4))

arrow((27.2, 14.4), (47.8, 14.4))
ax.text(31.0, 15.4, "per query", fontsize=8, color=GREY, va="bottom")

box(48.0, 11.8, 7.0, 5.2, "$W$", accent=True)

arrow((55.2, 14.4), (58.8, 14.4), color=ACC, lw=1.3)
vec(59.0, 13.5)
ax.text(61.6, 12.7, "$qW$", fontsize=10.5, color=INK, ha="center", va="top")

# the fit lands on W from above
route([(47.4, 27.9), (51.5, 27.9), (51.5, 17.2)])

# ---------------------------------------------------------------- corpus, once
cards(2.0, 4.6, "corpus")
arrow((7.4, 5.6), (11.8, 5.6))

box(12.0, 3.0, 8.0, 5.2, "$B$", sub="new model")
box(23.0, 3.0, 7.4, 5.2, "index", dashed=True, fs=10)
arrow((20.2, 5.6), (22.8, 5.6))
ax.text(26.7, 9.0, "not re-encoded", fontsize=8, color=GREY, ha="center", va="bottom")

# ---------------------------------------------------------------- retrieval
ax.add_patch(Circle((74.0, 9.4), 2.6, facecolor="none", edgecolor=LGREY, lw=0.9))
ax.text(74.0, 9.4, "$\\cos$", fontsize=9, color=INK, ha="center", va="center")
route([(65.8, 14.4), (74.0, 14.4), (74.0, 12.0)])
route([(30.6, 5.6), (74.0, 5.6), (74.0, 6.8)])

arrow((76.6, 9.4), (81.4, 9.4))
box(81.8, 6.8, 13.4, 5.2, "nDCG@10", fs=10.5)

fig.savefig("architecture.png", dpi=400, facecolor="white")
fig.savefig("architecture.pdf", facecolor="white")
fig.savefig("architecture.svg", facecolor="white")
print("ok")
