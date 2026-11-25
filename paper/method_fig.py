"""Two methods, one skeleton. Top: Plato's Key (HazyResearch). Bottom: embedding compatibility adapters."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (Rectangle, Circle, FancyArrowPatch, Arc,
                                FancyBboxPatch)

plt.rcParams.update({
    "font.family": "Inter",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Inter",
    "mathtext.it": "Inter:italic",
    "mathtext.bf": "Inter",
})

INK, GREY, LGREY, HAIR, FILL, ACC = "#111111", "#8a8a8a", "#dcdcdc", "#ececec", "#f7f7f7", "#a33a2e"
ACC_FILL = "#f1e0dd"

fig = plt.figure(figsize=(9.2, 3.07), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 102)
ax.set_ylim(0, 34.5)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

CW, CH, ROWS = 1.6, 1.7, 5
LBL = 0.85          # gap between a glyph and its label


def cell(x, y, w=CW, h=CW, dashed=False):
    ax.add_patch(Rectangle((x, y), w, h,
                           facecolor="none" if dashed else FILL,
                           edgecolor="#d0d0d0" if dashed else "#dedede", lw=0.55,
                           ls=(0, (2.4, 2.4)) if dashed else "-"))


def grid(x0, y0, ncols, nrows):
    for r in range(nrows):
        for c in range(ncols):
            cell(x0 + c * CW, y0 + r * CH, CW, CH)


def arrow(p1, p2, color=INK, lw=1.2, rad=0.0, ms=9):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}"))


def route(pts, color=INK, lw=1.2, ms=9):
    if len(pts) > 2:
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
                color=color, lw=lw, solid_capstyle="butt", zorder=2)
    arrow(pts[-2], pts[-1], color=color, lw=lw, ms=ms)


def sphere(cx, cy, r=4.2):
    ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=LGREY, lw=0.9))
    a_q, a_d = 58.0, 12.0
    for a in (100, 145, 200, 255, 312):
        ax.plot(cx + r * np.cos(np.radians(a)), cy + r * np.sin(np.radians(a)),
                "o", ms=3.0, color=LGREY)
    ax.plot(cx + r * np.cos(np.radians(a_d)), cy + r * np.sin(np.radians(a_d)),
            "o", ms=4.4, color=INK)
    ax.text(cx + 3.1, cy + 0.55, "$d$", ha="center", va="center", fontsize=13.12, color=INK)
    arrow((cx, cy), (cx + r * np.cos(np.radians(a_q)), cy + r * np.sin(np.radians(a_q))))
    ax.text(cx + r * np.cos(np.radians(a_q)) + 0.58, cy + r * np.sin(np.radians(a_q)) + 0.08,
            "$\\hat{e}$", ha="center", va="center", fontsize=13.12, color=INK)
    ax.add_patch(Arc((cx, cy), 2.9, 2.9, theta1=a_d, theta2=a_q, color=ACC, lw=1.1))
    ax.text(cx + 1.5 * np.cos(np.radians(29)), cy + 1.5 * np.sin(np.radians(29)),
            "$\\theta$", ha="center", va="center", fontsize=10.62, color=ACC)


def layers(x0, cy, w=8.3, h=0.82, gap=0.36, n=5, swap=1):
    pitch = h + gap
    y0 = cy - (n * h + (n - 1) * gap) / 2
    for i in range(n):
        hi = i == swap
        ax.add_patch(FancyBboxPatch((x0, y0 + i * pitch), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.16",
                                    facecolor=ACC_FILL if hi else FILL,
                                    edgecolor=ACC if hi else "#dedede",
                                    lw=1.0 if hi else 0.55))
    return y0 + swap * pitch + h / 2, y0 + (n - 1) * pitch + h / 2


def row(cy, kind):
    gy0 = cy - ROWS * CH / 2
    grid(3.0, gy0, 3, ROWS)
    grid(16.0, gy0, 4, ROWS)
    for r in range(ROWS):
        y = gy0 + (r + 0.5) * CH
        ax.plot([7.8, 16.0], [y, y], color="#e4e4e4", lw=0.7, zorder=0)
    ax.text(5.4, gy0 - LBL, "$X$", ha="center", va="top", fontsize=14.38, color=INK)
    ax.text(19.2, gy0 - LBL, "$Y$", ha="center", va="top", fontsize=14.38, color=INK)

    arrow((23.8, cy), (27.2, cy))
    if kind == "llm":
        ax.text(25.5, cy + 0.62, "$-\\mu$", ha="center", va="bottom", fontsize=11.25, color="#7c7c7c")

    grid(28.0, cy - 3.2, 4, 4)
    ax.text(31.2, cy - 3.2 - LBL, "$M = X^{\\mathsf{T}}Y$", ha="center", va="top",
            fontsize=14.38, color=INK)

    arrow((35.4, cy), (40.4, cy))
    ax.text(37.9, cy + 0.55, "SVD", ha="center", va="bottom", fontsize=11.25, color="#7c7c7c")

    grid(41.4, cy - 3.2, 4, 4)
    ax.add_patch(FancyArrowPatch((43.1, cy - 1.4), (46.3, cy + 1.6),
                                 arrowstyle="-|>", mutation_scale=9, color=ACC,
                                 lw=1.6, shrinkA=0, shrinkB=0,
                                 connectionstyle="arc3,rad=0.55"))
    ax.text(44.6, cy - 3.2 - LBL,
            "$R = UV^{\\mathsf{T}}$" if kind == "llm" else "$W = UV^{\\mathsf{T}}$",
            ha="center", va="top", fontsize=14.38, color=INK)

    arrow((48.6, cy), (51.6, cy))
    for i in range(3):
        cell(52.2 + i * CW, cy - CW / 2)
    if kind == "llm":
        x_out = 57.6
    else:
        cell(52.2 + 3 * CW, cy - CW / 2, dashed=True)
        ax.text(52.2 + 3.5 * CW, cy, "$0$", ha="center", va="center",
                fontsize=10.62, color="#6f6f6f")
        x_out = 59.4

    arrow((x_out, cy), (62.4, cy), color=ACC, lw=1.5)
    ax.text((x_out + 62.4) / 2, cy + 0.62, "$sR$" if kind == "llm" else "$W$",
            ha="center", va="bottom", fontsize=12.50, color=ACC)
    if kind == "llm":
        ax.text((x_out + 62.4) / 2, cy - 0.62, "$-\\mu_A$", ha="center", va="top",
                fontsize=11.25, color="#7c7c7c")

    for i in range(4):
        cell(63.0 + i * CW, cy - CW / 2)
    ax.text(52.2, cy - CW / 2 - LBL + 0.45, "$z_A$" if kind == "llm" else "$e$",
            ha="left", va="top", fontsize=14.38, color=INK)
    ax.text(63.0, cy - CW / 2 - LBL + 0.45, "$\\hat{z}_B$" if kind == "llm" else "$eW$",
            ha="left", va="top", fontsize=14.38, color=INK)
    if kind == "llm":
        ax.text(52.2, cy - CW / 2 - LBL - 1.55, "France", ha="left", va="top",
                fontsize=10.00, color=GREY)

    if kind == "llm":
        y_swap, y_top = layers(73.9, cy)
        route([(69.6, cy), (72.0, cy), (72.0, y_swap), (73.8, y_swap)])
        ax.text(71.0, cy + 0.62, "$+\\mu_B$", ha="center", va="bottom",
                fontsize=11.25, color="#7c7c7c")
        route([(82.4, y_top), (85.3, y_top), (85.3, cy), (88.4, cy)])
        out = "Paris"
    else:
        sphere(78.05, cy)
        arrow((82.4, cy), (88.4, cy))
        ax.text(86.3, cy + 0.62, "$\\cos$", ha="center", va="bottom",
                fontsize=11.88, color=INK)
        out = "nDCG@10"

    ax.add_patch(FancyBboxPatch((88.6, cy - 2.4), 10.4, 4.8,
                                boxstyle="round,pad=0,rounding_size=0.6",
                                facecolor=FILL, edgecolor="#dedede", lw=0.6))
    ax.text(93.8, cy, out, ha="center", va="center", fontsize=13.12, color=INK)


def head(y, title):
    ax.text(3.0, y, title, ha="left", va="bottom", fontsize=17.50, color=INK)


head(30.6, "Plato\u2019s Key (theirs)")
row(24.95, "llm")

ax.plot([3.0, 99.0], [17.45, 17.45], color=HAIR, lw=0.6)

head(13.55, "Ours, orthogonal Procrustes")
row(8.35, "emb")

fig.savefig("compare_paper.png", dpi=400, facecolor="white")
fig.savefig("compare_paper.pdf", facecolor="white")
fig.savefig("compare_paper.svg", facecolor="white")
print("ok")
