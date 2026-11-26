"""Unit tests for adapter.py mathematical invariants and API."""
import tempfile
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adapter import (
    AffineMap,
    affine_align,
    anisotropy,
    apply_adapter,
    diagnose,
    frame_correlation,
    linear_cka,
    lowrank_affine_align,
    procrustes_align,
    ridge_align,
    source_centred_align,
)


def _norm(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def test_procrustes_exact_rotation_all_shapes():
    rng = np.random.default_rng(42)
    for ds, dt in ((64, 64), (48, 64), (64, 48)):
        n = 500
        if ds <= dt:
            Q, _ = np.linalg.qr(rng.standard_normal((dt, ds)))
            R_true = Q[:, :ds].T  # (ds, dt), orthonormal rows
            X = _norm(rng.standard_normal((n, ds)))
            Y = _norm(X @ R_true)
        else:
            Q, _ = np.linalg.qr(rng.standard_normal((ds, dt)))
            R_true = Q[:, :dt]  # (ds, dt), orthonormal cols
            Z = _norm(rng.standard_normal((n, dt)))
            X = _norm(Z @ R_true.T)
            Y = Z
        W = procrustes_align(X, Y)
        assert W.shape == (ds, dt)
        # Check semi-orthogonality
        if ds <= dt:
            np.testing.assert_allclose(W @ W.T, np.eye(ds), atol=1e-6)
        else:
            np.testing.assert_allclose(W.T @ W, np.eye(dt), atol=1e-6)
        cos = float(np.mean(np.sum(apply_adapter(X, W) * Y, axis=1)))
        assert cos > 0.9999, f"Expected exact recovery for ({ds},{dt}), got {cos}"


def test_affine_recovers_anisotropic_shift():
    rng = np.random.default_rng(7)
    d, n = 64, 1500
    Z = _norm(rng.standard_normal((n, d)))
    mu_s = rng.standard_normal(d); mu_s = 2.0 * mu_s / np.linalg.norm(mu_s)
    mu_t = rng.standard_normal(d); mu_t = 0.2 * mu_t / np.linalg.norm(mu_t)
    R, _ = np.linalg.qr(rng.standard_normal((d, d)))
    X = _norm(Z + mu_s)
    Y = _norm(Z @ R + mu_t)
    diag = diagnose(X, Y)
    assert diag["kappa_source"] > 1.5
    assert diag["kappa_target"] < 0.5
    W = procrustes_align(X, Y)
    f = affine_align(X, Y)
    f_sc = source_centred_align(X, Y)
    cos_p = float(np.mean(np.sum(apply_adapter(X, W) * Y, axis=1)))
    cos_a = float(np.mean(np.sum(f(X) * Y, axis=1)))
    cos_sc = float(np.mean(np.sum(f_sc(X) * Y, axis=1)))
    assert cos_a > cos_p + 0.05, f"Affine ({cos_a}) should beat Procrustes ({cos_p})"
    assert cos_sc > cos_p + 0.05, f"Source-centred ({cos_sc}) should beat Procrustes ({cos_p})"
    assert cos_a > 0.98
    assert np.allclose(f_sc.mu_y, 0.0)


def test_ridge_and_lowrank_and_serialization():
    rng = np.random.default_rng(13)
    n, ds, dt, r = 600, 48, 64, 16
    B1 = np.linalg.qr(rng.standard_normal((ds, r)))[0]
    B2 = np.linalg.qr(rng.standard_normal((dt, r)))[0]
    Z = rng.standard_normal((n, r))
    X = _norm(Z @ B1.T)
    Y = _norm(Z @ B2.T)

    f_ridge = ridge_align(X, Y, lam=1e-3, rescale=True)
    f_lr = lowrank_affine_align(X, Y, rank=16)
    assert float(np.mean(np.sum(f_ridge(X) * Y, axis=1))) > 0.999
    assert float(np.mean(np.sum(f_lr(X) * Y, axis=1))) > 0.999

    # Single-vector call + save/load roundtrip
    v1 = f_lr(X[0])
    assert v1.shape == (dt,)
    np.testing.assert_allclose(np.linalg.norm(v1), 1.0, atol=1e-5)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "map.npz"
        f_lr.save(p)
        f2 = AffineMap.load(p)
        np.testing.assert_allclose(f_lr(X[:10]), f2(X[:10]), atol=1e-5)
        assert f2.kind == f_lr.kind


def test_procrustes_query_equals_corpus_for_square():
    rng = np.random.default_rng(99)
    d = 64
    X = _norm(rng.standard_normal((400, d)))
    R, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Y = _norm(X @ R + 0.05 * rng.standard_normal((400, d)))
    W_ab = procrustes_align(X, Y)
    W_ba = procrustes_align(Y, X)
    # For orthogonal W_ab, W_ba == W_ab.T, so (q @ W_ab) @ d.T == q @ (d @ W_ba).T
    q = _norm(rng.standard_normal((20, d)))
    c = _norm(rng.standard_normal((50, d)))
    s_query = apply_adapter(q, W_ab) @ c.T
    s_corpus = q @ apply_adapter(c, W_ba).T
    np.testing.assert_allclose(s_query, s_corpus, atol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("ALL TESTS PASSED")
