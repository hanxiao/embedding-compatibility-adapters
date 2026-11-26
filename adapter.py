# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
# ]
# ///
"""Embedding compatibility adapters: one SVD, no training loop, no GPU.

Bridge two independently trained text embedding spaces without re-indexing your
document store. Two diagnostics computed on a few thousand paired texts tell you
which map to use before fitting anything:

  frame correlation rho  mean cosine between paired embeddings (padded to the
                         wider dimension). Near zero means orthogonal frames and
                         an adapter is required; above ~0.1 the models already
                         share a coordinate frame (usually a shared backbone).
  anisotropy kappa       ||mean|| / RMS(x - mean) of one model. Above ~1 the
                         mean direction dominates the spread and breaks an
                         uncentred rotation; use affine_align instead.

Standard usage (zero re-indexing):
    from adapter import diagnose, affine_align, procrustes_align, apply_adapter

    # Example 1: documents are indexed with jina-v3; search with jina-v5-nano queries
    f = affine_align(X_v5nano_cal, Y_v3_cal)
    q_in_v3 = f(q_v5nano)                  # score directly against the frozen v3 index

    # Example 2: reverse -- documents indexed with jina-v5-nano; search with legacy v3 queries
    g = affine_align(X_v3_cal, Y_v5nano_cal)
    q_in_v5nano = g(q_v3)                  # score directly against the v5-nano index

    # Persist and reload
    f.save("v5nano_to_v3.npz")
    f = AffineMap.load("v5nano_to_v3.npz")

Standalone synthetic demo:
    uv run adapter.py
"""
from pathlib import Path
from typing import Union

import numpy as np


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        n = float(np.linalg.norm(X))
        return X / max(n, 1e-12)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


# ---------------------------------------------------------------------------
# Diagnostics, before fitting anything
# ---------------------------------------------------------------------------

def frame_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean cosine between paired embeddings after zero-padding to the wider space.

    Near 0: orthogonal coordinate frames; doing nothing retrieves at chance.
    Above ~0.1: shared coordinate frame (typically a shared pretrained backbone).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    d = max(X.shape[1], Y.shape[1])
    Xp = _l2_normalize(np.pad(X, ((0, 0), (0, d - X.shape[1]))))
    Yp = _l2_normalize(np.pad(Y, ((0, 0), (0, d - Y.shape[1]))))
    return float(np.mean(np.sum(Xp * Yp, axis=1)))


def anisotropy(X: np.ndarray) -> float:
    """||mean|| / RMS(x - mean). Above ~1 the mean dominates the spread."""
    X = _l2_normalize(X)
    mu = X.mean(axis=0)
    resid = X - mu
    rms = np.sqrt((resid ** 2).sum(axis=1).mean())
    return float(np.linalg.norm(mu) / max(rms, 1e-12))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear centered kernel alignment (rotation-invariant geometric similarity)."""
    Xc = _l2_normalize(X)
    Yc = _l2_normalize(Y)
    Xc = Xc - Xc.mean(axis=0)
    Yc = Yc - Yc.mean(axis=0)
    num = np.linalg.norm(Yc.T @ Xc) ** 2
    den = np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)
    return float(num / max(den, 1e-12))


def diagnose(X: np.ndarray, Y: np.ndarray) -> dict:
    """Compute pre-fit diagnostics on paired calibration embeddings."""
    rho = frame_correlation(X, Y)
    kx = anisotropy(X)
    ky = anisotropy(Y)
    cka = linear_cka(X, Y)
    if max(kx, ky) > 0.8:
        advice = "high anisotropy detected: use affine_align (centre + rotate + scale)"
    elif rho > 0.1:
        advice = "shared coordinate frame: affine_align still improves over identity"
    else:
        advice = "orthogonal frames, low anisotropy: procrustes_align or affine_align both work"
    return dict(
        rho=rho,
        kappa_source=kx,
        kappa_target=ky,
        cka=cka,
        dim_source=int(X.shape[1]),
        dim_target=int(Y.shape[1]),
        advice=advice,
    )


# ---------------------------------------------------------------------------
# Closed-form adapters
# ---------------------------------------------------------------------------

def _svd_procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(
        np.asarray(X, dtype=np.float64).T @ np.asarray(Y, dtype=np.float64),
        full_matrices=False,
    )
    return U @ Vt


def procrustes_align(X_source: np.ndarray, Y_target: np.ndarray) -> np.ndarray:
    """Orthogonal (or semi-orthogonal) Procrustes map W minimizing ||X W - Y||_F.

    Returns W of shape (d_source, d_target) with orthonormal rows (when
    d_source <= d_target) or orthonormal columns (when d_source >= d_target).
    No hyperparameters, no iteration.
    """
    return _svd_procrustes(_l2_normalize(X_source), _l2_normalize(Y_target))


class AffineMap:
    """One-sided affine map: z_hat = norm(mu_y + s * (norm(z) - mu_x) @ R).

    Covers orthogonal Procrustes (mu_x=0, mu_y=0, s=1), Plato's Key affine
    Procrustes (centred + Frobenius scale), low-rank affine Procrustes, and
    closed-form ridge regression.
    """

    def __init__(
        self,
        R: np.ndarray,
        s: float,
        mu_x: np.ndarray,
        mu_y: np.ndarray,
        kind: str = "affine",
    ):
        self.R = np.asarray(R, dtype=np.float64)
        self.s = float(s)
        self.mu_x = np.asarray(mu_x, dtype=np.float64)
        self.mu_y = np.asarray(mu_y, dtype=np.float64)
        self.kind = kind

    @property
    def source_dim(self) -> int:
        return int(self.R.shape[0])

    @property
    def target_dim(self) -> int:
        return int(self.R.shape[1])

    def __call__(self, X: np.ndarray) -> np.ndarray:
        single = np.asarray(X).ndim == 1
        Xn = _l2_normalize(np.atleast_2d(np.asarray(X, dtype=np.float64)))
        Z = (Xn - self.mu_x) @ self.R * self.s + self.mu_y
        out = _l2_normalize(Z).astype(np.float32)
        return out[0] if single else out

    def save(self, path: Union[str, Path]) -> None:
        np.savez(
            path,
            R=self.R.astype(np.float32),
            s=np.array([self.s], dtype=np.float64),
            mu_x=self.mu_x.astype(np.float32),
            mu_y=self.mu_y.astype(np.float32),
            kind=np.array([self.kind]),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AffineMap":
        d = np.load(path, allow_pickle=False)
        kind = str(d["kind"][0]) if "kind" in d else "affine"
        return cls(d["R"], float(d["s"][0]), d["mu_x"], d["mu_y"], kind=kind)

    def __repr__(self) -> str:
        return (
            f"AffineMap(kind={self.kind!r}, shape={self.R.shape}, s={self.s:.3f})"
        )


def source_centred_align(
    X_source: np.ndarray,
    Y_target: np.ndarray,
) -> AffineMap:
    """Source-centred orthogonal Procrustes: z_hat = norm((norm(z) - mu_x) @ R_c).

    Fits the orthogonal rotation R_c on centred embeddings (X - mu_x, Y - mu_y),
    subtracts mu_x at inference to remove source anisotropy, and omits +mu_y on
    the target side to avoid injecting target-side hubness bias. Zero
    hyperparameters, no validation queries required; beats both uncentred
    Procrustes and fixed Plato's Key affine across all calibration budgets.
    """
    X = _l2_normalize(X_source)
    Y = _l2_normalize(Y_target)
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    R = _svd_procrustes(X - mx, Y - my)
    return AffineMap(R, 1.0, mx, np.zeros_like(my), kind="src-centred")


def affine_align(
    X_source: np.ndarray,
    Y_target: np.ndarray,
    centre: bool = True,
    scale: bool = True,
) -> AffineMap:
    """Centre, rotate, rescale by Frobenius spread ratio, re-centre (Plato's Key).

    Recommended default across all model pairs. Reduces to orthogonal Procrustes
    when centre=False, scale=False (or when both spaces have zero mean).
    """
    X = _l2_normalize(X_source)
    Y = _l2_normalize(Y_target)
    mx = X.mean(axis=0) if centre else np.zeros(X.shape[1], dtype=np.float64)
    my = Y.mean(axis=0) if centre else np.zeros(Y.shape[1], dtype=np.float64)
    Xc, Yc = X - mx, Y - my
    R = _svd_procrustes(Xc, Yc)
    s = (
        float(np.linalg.norm(Yc) / max(np.linalg.norm(Xc), 1e-12))
        if scale
        else 1.0
    )
    return AffineMap(R, s, mx, my, kind="affine")


def ridge_align(
    X_source: np.ndarray,
    Y_target: np.ndarray,
    lam: float = 1e-2,
    rescale: bool = True,
) -> AffineMap:
    """Closed-form centred ridge regression, optionally Frobenius-rescaled."""
    X = _l2_normalize(X_source)
    Y = _l2_normalize(Y_target)
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - mx, Y - my
    n = len(X)
    A = np.linalg.solve(
        Xc.T @ Xc / n + lam * np.eye(X.shape[1], dtype=np.float64),
        Xc.T @ Yc / n,
    )
    if rescale:
        pred = Xc @ A
        s = float(np.linalg.norm(Yc) / max(np.linalg.norm(pred), 1e-12))
    else:
        s = 1.0
    return AffineMap(A, s, mx, my, kind=f"ridge(lam={lam:g})")


def lowrank_affine_align(
    X_source: np.ndarray,
    Y_target: np.ndarray,
    rank: int = 256,
) -> AffineMap:
    """One-sided low-rank affine adapter (top-k PCA subspaces, zero corpus re-indexing).

    Projects centred source onto its top-k principal components, aligns to target's
    top-k principal components via k x k orthogonal Procrustes + Frobenius scale,
    and lifts back into full target space with target mean restored.
    """
    X = _l2_normalize(X_source)
    Y = _l2_normalize(Y_target)
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - mx, Y - my
    k = min(rank, X.shape[1], Y.shape[1], len(X) - 1)
    _, _, Vtx = np.linalg.svd(Xc, full_matrices=False)
    _, _, Vty = np.linalg.svd(Yc, full_matrices=False)
    Px = Vtx[:k].T
    Py = Vty[:k].T
    Xk, Yk = Xc @ Px, Yc @ Py
    Wk = _svd_procrustes(Xk, Yk)
    s = float(np.linalg.norm(Yk) / max(np.linalg.norm(Xk), 1e-12))
    R_eff = Px @ Wk @ Py.T
    return AffineMap(R_eff, s, mx, my, kind=f"lowrank-affine(k={k})")


def apply_adapter(
    embeddings: np.ndarray,
    W: Union[np.ndarray, AffineMap],
    target_dim: int | None = None,
) -> np.ndarray:
    """Apply either an AffineMap or a Procrustes matrix W and L2-normalize."""
    if isinstance(W, AffineMap):
        out = W(embeddings)
        if target_dim is not None and out.shape[-1] > target_dim:
            out = _l2_normalize(out[..., :target_dim]).astype(np.float32)
        return out
    single = np.asarray(embeddings).ndim == 1
    X = _l2_normalize(np.atleast_2d(np.asarray(embeddings, dtype=np.float64)))
    if X.shape[1] < W.shape[0]:
        X = np.pad(X, ((0, 0), (0, W.shape[0] - X.shape[1])))
    out = X @ W
    if target_dim is not None:
        out = out[:, :target_dim]
    out = _l2_normalize(out).astype(np.float32)
    return out[0] if single else out


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def run_demo() -> None:
    rng = np.random.default_rng(0)
    d_src, d_tgt, rank, n = 768, 1024, 64, 4000
    basis_src = np.linalg.qr(rng.standard_normal((d_src, rank)))[0]
    basis_tgt = np.linalg.qr(rng.standard_normal((d_tgt, rank)))[0]

    def make_pair(mu_scale_src: float, mu_scale_tgt: float):
        Z = rng.standard_normal((n, rank))
        X = Z @ basis_src.T
        Y = Z @ basis_tgt.T
        if mu_scale_src > 0:
            ms = rng.standard_normal(d_src)
            X = X / np.linalg.norm(X, axis=1, keepdims=True) + mu_scale_src * ms / np.linalg.norm(ms)
        if mu_scale_tgt > 0:
            mt = rng.standard_normal(d_tgt)
            Y = Y / np.linalg.norm(Y, axis=1, keepdims=True) + mu_scale_tgt * mt / np.linalg.norm(mt)
        return _l2_normalize(X), _l2_normalize(Y)

    cos = lambda A, B: float(np.mean(np.sum(A * B, axis=1)))

    for label, ms, mt in (
        ("Isotropic spaces (low kappa, like v5-nano -> v3)", 0.0, 0.0),
        ("Anisotropic source (high kappa, like e5-large -> v3)", 1.8, 0.25),
    ):
        X, Y = make_pair(ms, mt)
        diag = diagnose(X, Y)
        W = procrustes_align(X, Y)
        f = affine_align(X, Y)
        r = ridge_align(X, Y)
        print(f"\n{label}")
        print(
            f"  rho={diag['rho']:+.3f}  cka={diag['cka']:.3f}  "
            f"kappa_src={diag['kappa_source']:.2f}  kappa_tgt={diag['kappa_target']:.2f}"
        )
        print(f"  procrustes  mean cosine: {cos(apply_adapter(X, W), Y):+.4f}")
        print(f"  affine      mean cosine: {cos(f(X), Y):+.4f}  (s={f.s:.3f})")
        print(f"  ridge       mean cosine: {cos(r(X), Y):+.4f}")
        print(f"  advice: {diag['advice']}")


if __name__ == "__main__":
    run_demo()
