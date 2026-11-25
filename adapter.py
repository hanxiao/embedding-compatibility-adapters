# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
# ]
# ///
"""Embedding compatibility adapters: one SVD, no training.

Two numbers decide what you need, and both come from the same few thousand
paired texts before any map is fitted:

  frame correlation rho  mean cosine between the two models' embeddings of the
                         same text. Near zero means the spaces share no
                         coordinate frame and a map is required; above ~0.1 the
                         identity already retrieves something and a map adds
                         little.
  anisotropy kappa       ||mean|| / RMS(x - mean) for one model. A source with
                         kappa above ~1 has a mean vector larger than its own
                         spread, which breaks an uncentred rotation.

Fit affine_align when kappa is high, procrustes_align when it is low.

Usage as library:
    W = procrustes_align(X_source, X_target)          # orthogonal
    f = affine_align(X_source, X_target)              # centre + scale + rotation
    adapted = f(embeddings)
    print(diagnose(X_source, X_target))

Usage standalone (synthetic demo):  uv run adapter.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# Diagnostics, before fitting anything
# ---------------------------------------------------------------------------

def frame_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean cosine between paired embeddings after padding to a common width.

    Near 0: the two spaces are related by a rotation and nothing else, so the
    identity map retrieves nothing and a map is needed.
    Above ~0.1: the models share a coordinate frame, so part of the work is
    already done and an adapter has less to add.
    """
    d = max(X.shape[1], Y.shape[1])
    Xp = np.pad(X, ((0, 0), (0, d - X.shape[1])))
    Yp = np.pad(Y, ((0, 0), (0, d - Y.shape[1])))
    Xp = Xp / np.maximum(np.linalg.norm(Xp, axis=1, keepdims=True), 1e-12)
    Yp = Yp / np.maximum(np.linalg.norm(Yp, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.sum(Xp * Yp, axis=1)))


def anisotropy(X: np.ndarray) -> float:
    """||mean|| / RMS(x - mean). Above ~1 the mean dominates the spread."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    resid = X - mu
    rms = np.sqrt((resid ** 2).sum(1).mean())
    return float(np.linalg.norm(mu) / max(rms, 1e-12))


def diagnose(X: np.ndarray, Y: np.ndarray) -> dict:
    rho = frame_correlation(X, Y)
    kx, ky = anisotropy(X), anisotropy(Y)
    if rho > 0.1:
        advice = "frames partly aligned: try the identity map first"
    elif max(kx, ky) > 1.0:
        advice = "frames orthogonal, one model anisotropic: use affine_align"
    else:
        advice = "frames orthogonal, low anisotropy: procrustes_align is enough"
    return dict(rho=rho, kappa_source=kx, kappa_target=ky, advice=advice)


# ---------------------------------------------------------------------------
# The maps
# ---------------------------------------------------------------------------

def _procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(X, np.float64).T @ np.asarray(Y, np.float64),
                             full_matrices=False)
    return U @ Vt


def procrustes_align(X_source: np.ndarray, Y_target: np.ndarray) -> np.ndarray:
    """Orthogonal map W minimizing ||X W - Y||_F subject to W^T W = I.

    One SVD of the cross-covariance, no hyperparameter, no iteration. When the
    widths differ the result is semi-orthogonal and projects onto the smaller
    space rather than padding.
    """
    return _procrustes(X_source, Y_target)


class AffineMap:
    """z_hat = mu_y + s * (z - mu_x) @ R, the map of the Plato's Key study."""

    def __init__(self, R, s, mu_x, mu_y):
        self.R, self.s, self.mu_x, self.mu_y = R, s, mu_x, mu_y

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mu_x) @ self.R * self.s + self.mu_y
        return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)

    def __repr__(self):
        return f"AffineMap(s={self.s:.3f}, R={self.R.shape})"


def affine_align(X_source: np.ndarray, Y_target: np.ndarray) -> AffineMap:
    """Centre, rotate, rescale, re-centre.

    The scale is the ratio of the two centred Frobenius norms, not a fitted
    parameter. Without centring it is exactly 1 on L2-normalized embeddings.
    """
    X = np.asarray(X_source, np.float64)
    Y = np.asarray(Y_target, np.float64)
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    R = _procrustes(Xc, Yc)
    s = float(np.linalg.norm(Yc) / max(np.linalg.norm(Xc), 1e-12))
    return AffineMap(R, s, mx, my)


def apply_adapter(embeddings: np.ndarray, W: np.ndarray, target_dim: int) -> np.ndarray:
    """Rotate, truncate to target_dim, renormalize."""
    X = np.asarray(embeddings, dtype=np.float64)
    if X.shape[1] < W.shape[0]:
        X = np.pad(X, ((0, 0), (0, W.shape[0] - X.shape[1])))
    out = (X @ W)[:, :target_dim]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo():
    rng = np.random.default_rng(0)
    d, rank, n = 256, 32, 4000
    basis = np.linalg.qr(rng.standard_normal((d, rank)))[0]

    def make(n):
        e = rng.standard_normal((n, rank)) @ basis.T
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def shift_and_rotate(E, mu_scale):
        mu = rng.standard_normal(d)
        mu = mu_scale * mu / np.linalg.norm(mu)
        R, _ = np.linalg.qr(rng.standard_normal((d, d)))
        Z = (E + mu) @ R
        return Z / np.linalg.norm(Z, axis=1, keepdims=True)

    X = make(n)
    Y = shift_and_rotate(X, 0.0)          # rotation only
    Y2 = shift_and_rotate(X, 1.2)         # rotation plus a dominant mean

    for name, YY in (("rotation only", Y), ("rotation + mean", Y2)):
        diag = diagnose(X, YY)
        W = procrustes_align(X, YY)
        f = affine_align(X, YY)
        cos = lambda A, B: float(np.mean(np.sum(A * B, 1)))
        print(f"\n{name}")
        print(f"  rho={diag['rho']:+.3f} kappa_source={diag['kappa_source']:.2f} "
              f"kappa_target={diag['kappa_target']:.2f}")
        print(f"  identity  cosine {cos(X, YY):+.3f}")
        print(f"  procrustes cosine {cos(apply_adapter(X, W, d), YY):+.3f}")
        print(f"  affine     cosine {cos(f(X), YY):+.3f}")
        print(f"  {diag['advice']}")


if __name__ == "__main__":
    run_demo()
