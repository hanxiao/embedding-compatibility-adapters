"""Generate crisp 3D geometric animations (GIF) for the README.

Three animations on a pure white editorial canvas:
  1. assets/anim-bridge-3d.gif      v5-nano query space rotates onto a frozen v3 index
  2. assets/anim-anisotropy-3d.gif  Why pure rotation collapses on an anisotropic cone
                                    (e5-large, kappa=1.83) and affine (centre + rotate
                                    + scale) restores it
  3. assets/anim-chain-hub-3d.gif   Chaining (A -> B -> C) and aligning 8 models onto
                                    one shared Platonic sphere

Requires: numpy, matplotlib, scipy, ffmpeg.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

OUT_DIR = Path(__file__).resolve().parent.parent / "assets"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
})

INK = "#111111"
MUTED = "#64748b"
WIRE = "#dce1e8"
WIRE_BACK = "#eef1f5"
ACC = "#b5382a"      # terracotta
FOREST = "#2a6f4e"   # hit / target
SLATE = "#3b5b7c"    # frozen index / cluster
OCHRE = "#b8732a"    # cluster
CLUSTER_COLORS = [ACC, FOREST, SLATE, OCHRE]


def ease(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def ease_cubic(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def project_3d(pts: np.ndarray, elev_deg: float = 18.0, azim_deg: float = -24.0,
               fov: float = 4.2):
    rot = Rotation.from_euler("yx", [-azim_deg, elev_deg], degrees=True).as_matrix()
    p = pts @ rot.T
    scale = fov / (fov - p[:, 2])
    return p[:, 0] * scale, p[:, 1] * scale, p[:, 2]


def draw_sphere_wireframe(ax, cx: float, cy: float, radius: float,
                          elev: float = 18.0, azim: float = -24.0,
                          ring_color: str | None = None, lw_ring: float = 1.05,
                          fill_color: str = "#fbfcfd"):
    ax.add_patch(Circle((cx, cy), radius, facecolor=fill_color, edgecolor="#d2d8e0",
                        linewidth=0.9, zorder=1))
    if ring_color:
        ax.add_patch(Circle((cx, cy), radius * 1.055, facecolor="none",
                            edgecolor=ring_color, linewidth=lw_ring,
                            linestyle=(0, (3.2, 2.4)), alpha=0.82, zorder=1.2))

    th = np.linspace(0, 2 * np.pi, 120)
    for lat_deg, lw in ((0.0, 0.85), (35.0, 0.5), (-35.0, 0.5)):
        lat = np.deg2rad(lat_deg)
        pts = np.column_stack([
            np.cos(lat) * np.cos(th),
            np.full_like(th, np.sin(lat)),
            np.cos(lat) * np.sin(th),
        ])
        x, y, z = project_3d(pts, elev, azim)
        front = z >= 0
        for mask, col, alpha in ((~front, WIRE_BACK, 0.7), (front, WIRE, 0.95)):
            xf = np.where(mask, cx + x * radius, np.nan)
            yf = np.where(mask, cy + y * radius, np.nan)
            ax.plot(xf, yf, color=col, lw=lw, alpha=alpha, zorder=1.5 if mask is front else 1.3)

    ph = np.linspace(0, 2 * np.pi, 120)
    for lon_deg in (0.0, 60.0, 120.0):
        lon = np.deg2rad(lon_deg)
        pts = np.column_stack([
            np.cos(lon) * np.sin(ph),
            np.cos(ph),
            np.sin(lon) * np.sin(ph),
        ])
        x, y, z = project_3d(pts, elev, azim)
        front = z >= 0
        for mask, col, alpha in ((~front, WIRE_BACK, 0.55), (front, WIRE, 0.8)):
            xf = np.where(mask, cx + x * radius, np.nan)
            yf = np.where(mask, cy + y * radius, np.nan)
            ax.plot(xf, yf, color=col, lw=0.5, alpha=alpha, zorder=1.4)


def draw_points_on_sphere(ax, pts: np.ndarray, colors: list, cx: float, cy: float,
                          radius: float, elev: float = 18.0, azim: float = -24.0,
                          size: float = 34.0, alpha_scale: float = 1.0,
                          edges: list | None = None, zbase: float = 3.0):
    x, y, z = project_3d(pts, elev, azim)
    px = cx + x * radius
    py = cy + y * radius

    if edges:
        for i, j, col in edges:
            zm = 0.5 * (z[i] + z[j])
            al = (0.12 + 0.28 * np.clip((zm + 1) / 2, 0, 1)) * alpha_scale
            ax.plot([px[i], px[j]], [py[i], py[j]], color=col, lw=0.75, alpha=al, zorder=zbase - 0.5)

    order = np.argsort(z)
    for idx in order:
        depth = np.clip((z[idx] + 1.0) / 2.0, 0.0, 1.0)
        al = (0.20 + 0.78 * (depth ** 1.3)) * alpha_scale
        sz = size * (0.62 + 0.55 * depth)
        ax.scatter(
            [px[idx]], [py[idx]],
            s=sz, color=colors[idx], alpha=al,
            edgecolors="#ffffff" if depth > 0.45 else "none",
            linewidths=0.45, zorder=zbase + depth,
        )
    return px, py, z


def _vis_len(s: str) -> int:
    return len(s.replace(r"$\to$", "→").replace(r"$\approx$", "≈"))


def draw_step_pills(ax, steps: list[str], active_idx: int, aspect: float, y: float = 0.068):
    """Draw Plato's Key style numbered pills in equal-aspect coordinates (true circles!)."""
    widths = [0.13 + 0.0205 * _vis_len(s) for s in steps]
    gap = 0.045
    total = sum(widths) + gap * (len(steps) - 1)
    x0 = aspect / 2.0 - total / 2.0
    h = 0.072
    r_num = 0.022
    for i, (label, w) in enumerate(zip(steps, widths)):
        active = i == active_idx
        ec = ACC if active else "#d8dce2"
        fc = "#ffffff" if active else "#fafafa"
        lw = 1.35 if active else 0.8
        box = FancyBboxPatch(
            (x0, y - h / 2), w, h,
            boxstyle="round,pad=0.004,rounding_size=0.036",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=10,
        )
        ax.add_patch(box)
        nc_x = x0 + 0.042
        ax.add_patch(Circle(
            (nc_x, y), r_num,
            facecolor=ACC if active else "#e2e6ec",
            edgecolor="none", zorder=11,
        ))
        ax.text(
            nc_x, y - 0.001, str(i + 1),
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color="#ffffff" if active else MUTED, zorder=12,
        )
        ax.text(
            nc_x + 0.038, y - 0.001, label,
            ha="left", va="center", fontsize=9.0,
            fontweight="bold" if active else "normal",
            color=INK if active else MUTED, zorder=12,
        )
        x0 += w + gap


def encode_gif(frame_dir: Path, out_gif: Path, fps: int = 24):
    palette = frame_dir / "palette.png"
    pattern = str(frame_dir / "frame_%04d.png")
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
         "-vf", "palettegen=max_colors=192:stats_mode=diff", str(palette)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern, "-i", str(palette),
         "-lavfi", "paletteuse=dither=sierra2_4a:diff_mode=rectangle", str(out_gif)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"  wrote {out_gif.name} ({out_gif.stat().st_size / 1024:.0f} KB)")


def make_constellation(seed: int = 7, per_cluster: int = 7):
    rng = np.random.default_rng(seed)
    centers = np.array([
        [0.48, 0.56, 0.67],
        [-0.66, 0.36, 0.66],
        [0.14, -0.76, 0.63],
        [-0.32, -0.22, -0.92],
    ], dtype=np.float64)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    pts, cols, edges = [], [], []
    for ci, c in enumerate(centers):
        base = len(pts)
        for k in range(per_cluster):
            v = c + rng.normal(0, 0.14, size=3)
            v /= np.linalg.norm(v)
            pts.append(v)
            cols.append(CLUSTER_COLORS[ci])
        for k in range(1, per_cluster):
            edges.append((base, base + k, CLUSTER_COLORS[ci]))
    return np.array(pts), cols, edges, centers


# ===========================================================================
# Animation 1: Hero 3D Bridge (v5-nano queries -> frozen v3 document index)
# ===========================================================================
def render_anim_bridge():
    print("Rendering anim-bridge-3d.gif ...")
    pts_tgt, cols, edges, centers_tgt = make_constellation(seed=11, per_cluster=7)
    q_tgt = centers_tgt[0].copy()
    R0 = Rotation.from_euler("zyx", [112, -54, 68], degrees=True)
    pts_src_init = pts_tgt @ R0.as_matrix().T
    q_src_init = q_tgt @ R0.as_matrix().T
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix([np.eye(3), R0.inv().as_matrix()]))

    fps = 24
    n_frames = 156
    W_in, H_in = 10.6, 4.4
    aspect = W_in / H_in
    tmp = Path(tempfile.mkdtemp(prefix="eca_anim1_"))
    try:
        for f in range(n_frames):
            fig = plt.figure(figsize=(W_in, H_in), dpi=115)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, aspect)
            ax.set_ylim(0, 1.0)
            ax.set_aspect("equal")
            ax.axis("off")

            azim = -22.0 + 7.0 * np.sin(2 * np.pi * f / n_frames)
            elev = 17.0 + 2.5 * np.cos(2 * np.pi * f / n_frames)

            if f < 42:
                step, rot_t, phase = 0, 0.0, f / 42.0
            elif f < 96:
                step, rot_t, phase = 1, ease_cubic((f - 42) / 54.0), (f - 42) / 54.0
            elif f < 140:
                step, rot_t, phase = 2, 1.0, (f - 96) / 44.0
            else:
                step, rot_t, phase = 0, 1.0 - ease_cubic((f - 140) / 16.0), 0.0

            R_cur = slerp(rot_t).as_matrix()
            pts_src = pts_src_init @ R_cur.T
            q_src = q_src_init @ R_cur.T

            cx_l, cx_r, cy, rad = 0.56, aspect - 0.56, 0.525, 0.305
            mid_x = aspect / 2.0

            ax.text(cx_l, 0.925, "QUERY MODEL", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=ACC)
            ax.text(cx_l, 0.878, "jina-v5-nano (768d)", ha="center", va="center",
                    fontsize=9.3, color=MUTED)

            ax.text(cx_r, 0.925, "FROZEN DOCUMENT INDEX", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=SLATE)
            ax.text(cx_r, 0.878, "jina-v3 (1024d, untouched)", ha="center", va="center",
                    fontsize=9.3, color=MUTED)

            box_ec = ACC if step == 1 else (FOREST if step == 2 else "#d0d5dd")
            box_fc = "#fff7f6" if step == 1 else ("#f3faf6" if step == 2 else "#f9fafb")
            ax.add_patch(FancyBboxPatch(
                (mid_x - 0.175, cy - 0.088), 0.35, 0.176,
                boxstyle="round,pad=0.01,rounding_size=0.03",
                facecolor=box_fc, edgecolor=box_ec, linewidth=1.3 if step >= 1 else 0.9,
                zorder=5,
            ))
            ax.text(mid_x, cy + 0.025, r"$f(q) = \mu_B + s(q - \mu_A)R$",
                    ha="center", va="center", fontsize=8.5, color=INK, zorder=6)
            sub_txt = (
                "identity (no map)" if step == 0
                else ("one SVD  (no GPU)" if step == 1 else "one matmul  (<0.1 ms)")
            )
            ax.text(mid_x, cy - 0.042, sub_txt, ha="center", va="center",
                    fontsize=7.8, color=ACC if step == 1 else (FOREST if step == 2 else MUTED), zorder=6)

            arr_col = "#c8cfd8" if step == 0 else (ACC if step == 1 else FOREST)
            ax.annotate("", xy=(mid_x - 0.195, cy), xytext=(cx_l + rad + 0.03, cy),
                        arrowprops=dict(arrowstyle="-|>", color=arr_col, lw=1.25, mutation_scale=11), zorder=4)
            ax.annotate("", xy=(cx_r - rad - 0.035, cy), xytext=(mid_x + 0.195, cy),
                        arrowprops=dict(arrowstyle="-|>", color=arr_col, lw=1.25, mutation_scale=11), zorder=4)

            draw_sphere_wireframe(ax, cx_l, cy, rad, elev, azim)
            draw_sphere_wireframe(ax, cx_r, cy, rad, elev, azim, ring_color=SLATE, lw_ring=1.05)

            px_l, py_l, _ = draw_points_on_sphere(ax, pts_src, cols, cx_l, cy, rad, elev, azim,
                                                  size=35, edges=edges)
            px_r, py_r, _ = draw_points_on_sphere(ax, pts_tgt, cols, cx_r, cy, rad, elev, azim,
                                                  size=35, edges=edges)

            qx_l, qy_l, _ = project_3d(q_src[None, :], elev, azim)
            qxl, qyl = cx_l + qx_l[0] * rad, cy + qy_l[0] * rad
            ax.scatter([qxl], [qyl], s=115, facecolor="#ffffff", edgecolor=ACC, lw=2.0, zorder=8)
            ax.scatter([qxl], [qyl], s=32, facecolor=ACC, edgecolor="none", zorder=8.2)
            ax.text(qxl + 0.042, qyl + 0.028, "q", fontsize=9.5, fontweight="bold", color=ACC, zorder=9)

            if step == 0 and f < 40:
                t_fly = ease(min(1.0, (f - 4) / 22.0)) if f >= 4 else 0.0
                qx_raw, qy_raw, _ = project_3d(q_src_init[None, :], elev, azim)
                tx, ty = cx_r + qx_raw[0] * rad, cy + qy_raw[0] * rad
                if t_fly > 0:
                    cur_x = qxl + (tx - qxl) * t_fly
                    cur_y = qyl + (ty - qyl) * t_fly + 0.14 * np.sin(np.pi * t_fly)
                    ax.scatter([cur_x], [cur_y], s=70, facecolor=ACC, edgecolor="#ffffff", lw=1.2, zorder=9)
                if f >= 26:
                    ax.scatter([tx], [ty], s=135, facecolor="none", edgecolor=ACC, lw=1.6, ls="--", zorder=8.5)
                    ax.text(cx_r, 0.165, "miss: nDCG@10 = 0.004", ha="center", va="center",
                            fontsize=8.3, fontweight="bold", color=ACC,
                            bbox=dict(boxstyle="round,pad=0.28", facecolor="#fff5f4", edgecolor="#f0c8c4", lw=0.8),
                            zorder=9.5)

            if step == 1:
                al_arc = np.sin(np.pi * phase) * 0.52
                for ci in range(3):
                    idx = ci * 7
                    mx = 0.5 * (px_l[idx] + px_r[idx])
                    my = 0.5 * (py_l[idx] + py_r[idx]) + 0.15 * (1 if ci != 2 else -1)
                    ts = np.linspace(0, 1, 30)
                    bx = (1 - ts) ** 2 * px_l[idx] + 2 * (1 - ts) * ts * mx + ts ** 2 * px_r[idx]
                    by = (1 - ts) ** 2 * py_l[idx] + 2 * (1 - ts) * ts * my + ts ** 2 * py_r[idx]
                    ax.plot(bx, by, color=CLUSTER_COLORS[ci], lw=1.0, ls=":", alpha=al_arc, zorder=2.5)

            if step == 2:
                t_fly = ease(min(1.0, phase / 0.45))
                qx_t, qy_t, _ = project_3d(q_tgt[None, :], elev, azim)
                tx, ty = cx_r + qx_t[0] * rad, cy + qy_t[0] * rad
                cur_x = qxl + (tx - qxl) * t_fly
                cur_y = qyl + (ty - qyl) * t_fly + 0.14 * np.sin(np.pi * t_fly)
                ax.scatter([cur_x], [cur_y], s=85, facecolor=FOREST, edgecolor="#ffffff", lw=1.4, zorder=9)
                if phase >= 0.45:
                    pulse = 1.0 + 0.25 * np.sin(2 * np.pi * (phase - 0.45) / 0.55)
                    ax.scatter([tx], [ty], s=175 * pulse, facecolor="none",
                               edgecolor=FOREST, lw=1.8, alpha=0.88, zorder=8.6)
                    ax.text(cx_r, 0.165, "hit: 0.572 nDCG@10 (zero re-index)", ha="center", va="center",
                            fontsize=8.3, fontweight="bold", color=FOREST,
                            bbox=dict(boxstyle="round,pad=0.28", facecolor="#f2faf5", edgecolor="#bfe0cc", lw=0.8),
                            zorder=9.5)

            draw_step_pills(
                ax,
                ["Unaligned frames (ρ ≈ 0)", "Fit one SVD (no GPU)", "Query frozen v3 index"],
                active_idx=step,
                aspect=aspect,
                y=0.068,
            )
            fig.savefig(tmp / f"frame_{f:04d}.png", dpi=115)
            plt.close(fig)

        encode_gif(tmp, OUT_DIR / "anim-bridge-3d.gif", fps=fps)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# Animation 2: Anisotropy (Why Rotation Fails on e5-large & Affine Recovers It)
# ===========================================================================
def render_anim_anisotropy():
    print("Rendering anim-anisotropy-3d.gif ...")
    pts_tgt, cols, edges, _ = make_constellation(seed=23, per_cluster=6)

    # Construct anisotropic cone in R^3 where affine(centre + rotate + scale) is EXACT!
    # Start from centred target points, shrink spread by 0.34, rotate by R_sem, shift to pole mu_src
    mu_tgt = pts_tgt.mean(0)
    pts_tgt_c = pts_tgt - mu_tgt
    R_sem = Rotation.from_euler("zyx", [88, -42, 52], degrees=True).as_matrix()
    mu_src = np.array([-0.48, 0.72, 0.48])
    mu_src = 0.86 * mu_src / np.linalg.norm(mu_src)
    pts_cone = mu_src + 0.34 * (pts_tgt_c @ R_sem.T)  # sits cleanly inside tight polar cap!

    # Left: uncentred Procrustes on pts_cone -> pts_tgt (rotates cone rigidly, stays crushed!)
    U, _, Vt = np.linalg.svd(pts_cone.T @ pts_tgt, full_matrices=False)
    R_proc = U @ Vt
    if np.linalg.det(R_proc) < 0:
        U[:, -1] *= -1
        R_proc = U @ Vt
    slerp_proc = Slerp([0.0, 1.0], Rotation.from_matrix([np.eye(3), R_proc]))

    # Right: exact affine inverse (centre -> rotate R_sem -> rescale 1/0.34 -> + mu_tgt)
    slerp_aff = Slerp([0.0, 1.0], Rotation.from_matrix([np.eye(3), R_sem]))
    scale_target = 1.0 / 0.34

    fps = 24
    n_frames = 156
    W_in, H_in = 10.6, 4.4
    aspect = W_in / H_in
    tmp = Path(tempfile.mkdtemp(prefix="eca_anim2_"))
    try:
        for f in range(n_frames):
            fig = plt.figure(figsize=(W_in, H_in), dpi=115)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, aspect)
            ax.set_ylim(0, 1.0)
            ax.set_aspect("equal")
            ax.axis("off")

            azim = -24.0 + 6.0 * np.sin(2 * np.pi * f / n_frames)
            elev = 18.0

            if f < 36:
                step, t_cent, t_scale, t_rot = 0, 0.0, 0.0, 0.0
            elif f < 96:
                step = 1
                u = (f - 36) / 60.0
                t_cent = ease_cubic(min(1.0, u / 0.45))
                t_scale = ease_cubic(max(0.0, (u - 0.22) / 0.78))
                t_rot = ease_cubic(u)
            elif f < 140:
                step, t_cent, t_scale, t_rot = 2, 1.0, 1.0, 1.0
            else:
                step = 0
                u = 1.0 - ease_cubic((f - 140) / 16.0)
                t_cent = t_scale = t_rot = u

            cx_l, cx_r, cy, rad = 0.62, aspect - 0.62, 0.525, 0.305

            ax.text(cx_l, 0.925, "ROTATION ONLY  (PROCRUSTES)", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=ACC)
            ax.text(cx_l, 0.878, "cone rotates rigidly, clusters stay crushed", ha="center", va="center",
                    fontsize=9.0, color=MUTED)

            ax.text(cx_r, 0.925, "AFFINE  (PLATO'S KEY)", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=FOREST)
            ax.text(cx_r, 0.878, "centre, rotate, rescale to match target spread", ha="center", va="center",
                    fontsize=9.0, color=MUTED)

            ax.plot([aspect / 2.0, aspect / 2.0], [0.16, 0.92], color="#eceff3", lw=1.0, zorder=1)

            draw_sphere_wireframe(ax, cx_l, cy, rad, elev, azim)
            draw_sphere_wireframe(ax, cx_r, cy, rad, elev, azim)

            tx_l, ty_l, tz_l = project_3d(pts_tgt, elev, azim)
            for idx in range(len(pts_tgt)):
                al_g = 0.22 + 0.38 * np.clip((tz_l[idx] + 1) / 2, 0, 1)
                ax.scatter([cx_l + tx_l[idx] * rad], [cy + ty_l[idx] * rad],
                           s=46, facecolor="none", edgecolor=cols[idx], lw=0.75, alpha=al_g, zorder=2.2)
                ax.scatter([cx_r + tx_l[idx] * rad], [cy + ty_l[idx] * rad],
                           s=46, facecolor="none", edgecolor=cols[idx], lw=0.75, alpha=al_g, zorder=2.2)

            pts_l = pts_cone @ slerp_proc(t_rot).as_matrix()
            draw_points_on_sphere(ax, pts_l, cols, cx_l, cy, rad, elev, azim, size=32, edges=edges)

            # Right: exact smooth interpolation from pts_cone -> centred -> rotated+scaled -> pts_tgt
            cur_scale = 1.0 + t_scale * (scale_target - 1.0)
            pts_r = (pts_cone - t_cent * mu_src) @ slerp_aff(t_rot).as_matrix() * cur_scale + t_scale * mu_tgt
            draw_points_on_sphere(ax, pts_r, cols, cx_r, cy, rad, elev, azim, size=32, edges=edges)

            if step == 2:
                ax.text(cx_l, 0.168, r"e5-large $\to$ v3 :  0.207  (34%)", ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=ACC,
                        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fff5f4", edgecolor="#f0c8c4", lw=0.8),
                        zorder=9)
                ax.text(cx_r, 0.168, r"e5-large $\to$ v3 :  0.588  (96%)", ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=FOREST,
                        bbox=dict(boxstyle="round,pad=0.28", facecolor="#f2faf5", edgecolor="#bfe0cc", lw=0.8),
                        zorder=9)

            draw_step_pills(
                ax,
                ["Anisotropic cone (κ = 1.82)", "Centre & rescale spread", "Clusters lock on target"],
                active_idx=step,
                aspect=aspect,
                y=0.068,
            )
            fig.savefig(tmp / f"frame_{f:04d}.png", dpi=115)
            plt.close(fig)

        encode_gif(tmp, OUT_DIR / "anim-anisotropy-3d.gif", fps=fps)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# Animation 3: Chaining (A -> B -> C) & 8-Model Platonic Hub
# ===========================================================================
def render_anim_chain_hub():
    print("Rendering anim-chain-hub-3d.gif ...")
    pts_hub, cols, edges, _ = make_constellation(seed=5, per_cluster=5)
    rng = np.random.default_rng(101)
    model_names = [
        "v5-small", "v5-nano", "Qwen3-0.6B", "Arctic-m",
        "v4", "v3", "bge-m3", "e5-large",
    ]
    model_rots = [Rotation.random(random_state=rng) for _ in model_names]

    R_A, R_B, R_C = model_rots[1], model_rots[2], model_rots[5]
    pts_A = pts_hub @ R_A.as_matrix().T
    pts_B = pts_hub @ R_B.as_matrix().T
    pts_C = pts_hub @ R_C.as_matrix().T

    fps = 24
    n_frames = 168
    W_in, H_in = 10.6, 4.75
    aspect = W_in / H_in
    tmp = Path(tempfile.mkdtemp(prefix="eca_anim3_"))
    try:
        for f in range(n_frames):
            fig = plt.figure(figsize=(W_in, H_in), dpi=115)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, aspect)
            ax.set_ylim(0, 1.0)
            ax.set_aspect("equal")
            ax.axis("off")

            azim = -20.0 + 10.0 * np.sin(2 * np.pi * f / n_frames)
            elev = 18.0
            cx0 = aspect / 2.0

            if f < 78:
                step = 0
                u = f / 78.0
                cxs = [cx0 - 0.68, cx0, cx0 + 0.68]
                cy, rad = 0.54, 0.235
                labels = [("v5-nano", "model A", ACC),
                          ("Qwen3-0.6B", "bridge B (never run)", MUTED),
                          ("v3", "model C", SLATE)]
                for cx, (nm, sub, col), pts_m in zip(cxs, labels, [pts_A, pts_B, pts_C]):
                    ax.text(cx, 0.895, nm, ha="center", va="center", fontsize=9.5, fontweight="bold", color=col)
                    ax.text(cx, 0.848, sub, ha="center", va="center", fontsize=8.0, color=MUTED)
                    draw_sphere_wireframe(ax, cx, cy, rad, elev, azim)
                    draw_points_on_sphere(ax, pts_m, cols, cx, cy, rad, elev, azim, size=26, edges=edges)

                t1 = ease_cubic(min(1.0, u / 0.36))
                t2 = ease_cubic(min(1.0, max(0.0, (u - 0.36) / 0.36)))
                t3 = ease_cubic(min(1.0, max(0.0, (u - 0.56) / 0.30)))

                ax.annotate("", xy=(cxs[1] - rad - 0.025, cy + 0.03),
                            xytext=(cxs[0] + rad + 0.025, cy + 0.03),
                            arrowprops=dict(arrowstyle="-|>", color=ACC if t1 > 0.05 else "#d4d9e0", lw=1.4))
                ax.text(0.5 * (cxs[0] + cxs[1]), cy + 0.08, r"$W_{A\to B}$",
                        ha="center", va="center", fontsize=8.5, color=ACC if t1 > 0.05 else MUTED)

                ax.annotate("", xy=(cxs[2] - rad - 0.025, cy + 0.03),
                            xytext=(cxs[1] + rad + 0.025, cy + 0.03),
                            arrowprops=dict(arrowstyle="-|>", color=FOREST if t2 > 0.05 else "#d4d9e0", lw=1.4))
                ax.text(0.5 * (cxs[1] + cxs[2]), cy + 0.08, r"$W_{B\to C}$",
                        ha="center", va="center", fontsize=8.5, color=FOREST if t2 > 0.05 else MUTED)

                if t1 < 1.0:
                    dx = cxs[0] + (cxs[1] - cxs[0]) * t1
                    dy = cy + 0.03 + 0.11 * np.sin(np.pi * t1)
                    ax.scatter([dx], [dy], s=85, facecolor=ACC, edgecolor="#ffffff", lw=1.4, zorder=9)
                elif t2 < 1.0:
                    dx = cxs[1] + (cxs[2] - cxs[1]) * t2
                    dy = cy + 0.03 + 0.11 * np.sin(np.pi * t2)
                    ax.scatter([dx], [dy], s=85, facecolor=FOREST, edgecolor="#ffffff", lw=1.4, zorder=9)
                else:
                    ax.scatter([cxs[2]], [cy], s=140, facecolor="none", edgecolor=FOREST, lw=1.8, zorder=9)

                if t3 > 0.02:
                    ts = np.linspace(0, t3, 40)
                    bx = (1 - ts) ** 2 * cxs[0] + 2 * (1 - ts) * ts * cxs[1] + ts ** 2 * cxs[2]
                    by = (1 - ts) ** 2 * (cy - rad - 0.02) + 2 * (1 - ts) * ts * 0.16 + ts ** 2 * (cy - rad - 0.02)
                    ax.plot(bx, by, color=SLATE, lw=1.2, ls="--", alpha=0.85, zorder=4)
                    if t3 > 0.60:
                        ax.text(cxs[1], 0.185, r"chained  $\approx$  98% of direct A $\to$ C",
                                ha="center", va="center", fontsize=8.5, fontweight="bold", color=FOREST,
                                bbox=dict(boxstyle="round,pad=0.25", facecolor="#f2faf5", edgecolor="#bfe0cc", lw=0.8),
                                zorder=9)
            else:
                step = 1 if f < 118 else 2
                u = (f - 78) / 76.0
                u_align = ease_cubic(min(1.0, max(0.0, (u - 0.10) / 0.56)))
                cy0 = 0.545
                rad_hub = 0.155
                rad_sm = 0.072
                # Clean wide ellipse so top/bottom spheres have huge clearance from hub & pills!
                orbit_rx, orbit_ry = 0.72, 0.305

                draw_sphere_wireframe(ax, cx0, cy0, rad_hub, elev, azim,
                                      ring_color=FOREST if step == 2 else "#d0d5dd", lw_ring=1.2)
                if u_align > 0.15:
                    draw_points_on_sphere(ax, pts_hub, cols, cx0, cy0, rad_hub, elev, azim,
                                          size=24, alpha_scale=u_align, edges=edges)
                ax.text(cx0, cy0, "shared hub" if step == 2 else "hub",
                        ha="center", va="center", fontsize=8.0, fontweight="bold",
                        color=FOREST if step == 2 else MUTED,
                        bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff",
                                  edgecolor="#d8e8df" if step == 2 else "#e5e7eb", alpha=0.92, lw=0.7),
                        zorder=9)

                # Arrange 8 models at angles offset by 22.5 deg so NONE sits straight top/bottom!
                for i, (nm, Ri) in enumerate(zip(model_names, model_rots)):
                    ang = np.deg2rad(90.0 - 22.5 - 45.0 * i)
                    mx = cx0 + orbit_rx * np.cos(ang)
                    my = cy0 + orbit_ry * np.sin(ang)
                    if u_align > 0.02:
                        vx, vy = cx0 - mx, cy0 - my
                        d = np.hypot(vx, vy)
                        x1, y1 = mx + vx * (rad_sm + 0.015) / d, my + vy * (rad_sm + 0.015) / d
                        x2, y2 = cx0 - vx * (rad_hub + 0.02) / d, cy0 - vy * (rad_hub + 0.02) / d
                        ax.plot([x1, x1 + (x2 - x1) * u_align], [y1, y1 + (y2 - y1) * u_align],
                                color=FOREST if step == 2 else "#c5ccd6",
                                lw=0.95, alpha=0.35 + 0.45 * u_align, ls=":", zorder=2)

                    slerp_i = Slerp([0.0, 1.0], Rotation.from_matrix([Ri.as_matrix(), np.eye(3)]))
                    pts_i = pts_hub @ slerp_i(u_align).as_matrix().T
                    draw_sphere_wireframe(ax, mx, my, rad_sm, elev, azim)
                    draw_points_on_sphere(ax, pts_i, cols, mx, my, rad_sm, elev, azim, size=9)

                    # Clean outward labels with zero overlap
                    if np.cos(ang) >= 0:
                        lx, ha = mx + rad_sm + 0.026, "left"
                    else:
                        lx, ha = mx - rad_sm - 0.026, "right"
                    ly = my + 0.015 * np.sign(np.sin(ang))
                    ax.text(lx, ly, nm, ha=ha, va="center", fontsize=8.2, fontweight="bold", color=INK)

            draw_step_pills(
                ax,
                [r"Chaining A $\to$ B $\to$ C", "Align 8 models (Affine GPA)", "One shared space"],
                active_idx=step,
                aspect=aspect,
                y=0.068,
            )
            fig.savefig(tmp / f"frame_{f:04d}.png", dpi=115)
            plt.close(fig)

        encode_gif(tmp, OUT_DIR / "anim-chain-hub-3d.gif", fps=fps)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    render_anim_bridge()
    render_anim_anisotropy()
    render_anim_chain_hub()
    print("All animations rendered to", OUT_DIR)
