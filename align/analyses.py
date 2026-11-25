"""Predictors, chaining, hub alignment, low rank, calibration size."""
import numpy as np

from .adapters import fit_procrustes, fit_affine


# ------------------------------------------------------------------ predictors
def cka(X, Y):
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    num = np.linalg.norm(Yc.T @ Xc) ** 2
    den = np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)
    return float(num / den)


def neighbour_agreement(X, Y, k=10, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    Xs, Ys = X[idx], Y[idx]
    out = []
    for A in (Xs, Ys):
        S = A @ A.T
        np.fill_diagonal(S, -np.inf)
        out.append(np.argsort(-S, axis=1)[:, :k])
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(*out)]))


def spectrum_features(X, Y, k=256):
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    s = np.linalg.svd(Xc.T @ Yc, compute_uv=False)
    e = s ** 2
    tot = e.sum()
    p = e / tot
    return dict(top_energy=float(e[:k].sum() / tot),
                eff_rank=float(1.0 / np.sum(p ** 2)),
                s1_frac=float(np.sqrt(e[0] / tot)))


def fit_residual(X, Y):
    """Directional: how well the source reproduces the target, and the reverse."""
    W = fit_procrustes(X, Y).A
    rx = float(np.linalg.norm(X @ W - Y) / np.linalg.norm(Y))
    V = fit_procrustes(Y, X).A
    ry = float(np.linalg.norm(Y @ V - X) / np.linalg.norm(X))
    return rx, ry


def predict_features(X, Y):
    f = dict(cka=cka(X, Y), nagree=neighbour_agreement(X, Y),
             d_s=int(X.shape[1]), d_t=int(Y.shape[1]),
             dim_ratio=float(X.shape[1] / Y.shape[1]))
    f.update(spectrum_features(X, Y))
    f["resid_s2t"], f["resid_t2s"] = fit_residual(X, Y)
    f["resid_gap"] = f["resid_s2t"] - f["resid_t2s"]
    return f


# ------------------------------------------------------------------ chaining / hub
def eval_map(study, src, tgt, transform, split="test", corpus_transform=None):
    """transform(q_src) scored against the target corpus, optionally projected."""
    from .evaluate import analyze
    out = {}
    for t in study.tasks:
        i = study.idx(t, split)
        q = transform(study.q[t][src][i])
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        C = study.c[t][tgt]
        if corpus_transform is not None:
            C = corpus_transform(C)
        C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        out[t] = analyze(q @ C.T, study.rel[t], i)["ndcg"]
    return out


# ------------------------------------------------------------------ hub (GPA)
def fit_gpa(cal, models, iters=40, seed=0):
    """Generalized Procrustes: one semi-orthogonal map per model into a shared space."""
    rng = np.random.default_rng(seed)
    dmax = max(cal[m].shape[1] for m in models)
    R = {}
    for m in models:
        dm = cal[m].shape[1]
        Q, _ = np.linalg.qr(rng.standard_normal((dmax, dm)))
        R[m] = Q[:, :dm].T          # (dm, dmax), orthonormal rows
    for _ in range(iters):
        Z = {m: cal[m] @ R[m] for m in models}
        M = np.mean([Z[m] for m in models], axis=0)
        for m in models:
            R[m] = fit_procrustes(cal[m], M).A
    return R


# ------------------------------------------------------------------ low rank
def pca_basis(X, k):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T.astype(np.float32)
