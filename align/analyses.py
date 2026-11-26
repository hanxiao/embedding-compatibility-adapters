"""Predictors, chaining, hub alignment (Procrustes & Affine GPA), low rank, deployment."""
import numpy as np

from .adapters import fit_procrustes, fit_affine


# ------------------------------------------------------------------ predictors
def frame_correlation(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    d = max(X.shape[1], Y.shape[1])
    Xp = np.pad(X, ((0, 0), (0, d - X.shape[1])))
    Yp = np.pad(Y, ((0, 0), (0, d - Y.shape[1])))
    Xp = Xp / np.maximum(np.linalg.norm(Xp, axis=1, keepdims=True), 1e-12)
    Yp = Yp / np.maximum(np.linalg.norm(Yp, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.sum(Xp * Yp, axis=1)))


def anisotropy(X):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    resid = X - mu
    rms = np.sqrt((resid ** 2).sum(1).mean())
    return float(np.linalg.norm(mu) / max(rms, 1e-12))


def cka(X, Y):
    Xc = np.asarray(X, dtype=np.float64)
    Yc = np.asarray(Y, dtype=np.float64)
    Xc = Xc - Xc.mean(0)
    Yc = Yc - Yc.mean(0)
    num = np.linalg.norm(Yc.T @ Xc) ** 2
    den = np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)
    return float(num / max(den, 1e-12))


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
    Xc = np.asarray(X, dtype=np.float64) - np.mean(X, axis=0)
    Yc = np.asarray(Y, dtype=np.float64) - np.mean(Y, axis=0)
    s = np.linalg.svd(Xc.T @ Yc, compute_uv=False)
    e = s ** 2
    tot = e.sum()
    p = e / tot
    return dict(
        top_energy=float(e[:k].sum() / tot),
        eff_rank=float(1.0 / np.sum(p ** 2)),
        s1_frac=float(np.sqrt(e[0] / tot)),
    )


def fit_residual(X, Y):
    """Directional residuals for both uncentred Procrustes and centred Affine."""
    W = fit_procrustes(X, Y).A
    rx = float(np.linalg.norm(X @ W - Y) / np.linalg.norm(Y))
    V = fit_procrustes(Y, X).A
    ry = float(np.linalg.norm(Y @ V - X) / np.linalg.norm(X))
    return rx, ry


def predict_features(X, Y):
    f = dict(
        rho=frame_correlation(X, Y),
        kappa_source=anisotropy(X),
        kappa_target=anisotropy(Y),
        cka=cka(X, Y),
        nagree=neighbour_agreement(X, Y),
        d_s=int(X.shape[1]),
        d_t=int(Y.shape[1]),
        dim_ratio=float(X.shape[1] / Y.shape[1]),
    )
    f.update(spectrum_features(X, Y))
    f["resid_s2t"], f["resid_t2s"] = fit_residual(X, Y)
    f["resid_gap"] = f["resid_s2t"] - f["resid_t2s"]
    return f


# ------------------------------------------------------------------ evaluation helpers
def eval_map(study, src, tgt, transform, split="test", corpus_transform=None):
    """Query-side: transform(q_src) scored against target corpus (optionally transformed)."""
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


def eval_corpus_map(study, index_model, query_model, corpus_transform, split="test"):
    """Corpus-side conversion: transform frozen index_model corpus into query_model space,
    then score with native query_model queries (zero neural re-encoding)."""
    from .evaluate import analyze
    out = {}
    for t in study.tasks:
        i = study.idx(t, split)
        q = study.q[t][query_model][i]
        C = corpus_transform(study.c[t][index_model])
        C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        out[t] = analyze(q @ C.T, study.rel[t], i)["ndcg"]
    return out


def eval_hybrid_migration(study, index_model, query_model, corpus_transform,
                          fracs=(0.0, 0.25, 0.5, 0.75, 1.0), split="test", seed=0):
    """Progressive zero-downtime migration: replace frac of corpus docs with native
    query_model embeddings while the remaining (1 - frac) stay corpus-adapted from
    index_model, all searched by native query_model queries.

    Returns both 'uncalibrated' (raw cosine in one mixed index) and 'zscore_top100'
    (per-query z-score normalization of the top-100 hits from each shard before merging).
    """
    from .evaluate import analyze
    rng = np.random.default_rng(seed)
    perms = {t: rng.permutation(len(study.c[t][index_model])) for t in study.tasks}
    adapted_cache = {}
    for t in study.tasks:
        C_adp = corpus_transform(study.c[t][index_model])
        C_adp = C_adp / np.maximum(np.linalg.norm(C_adp, axis=1, keepdims=True), 1e-12)
        adapted_cache[t] = C_adp.astype(np.float32)

    out = dict(uncalibrated={}, zscore_top100={})
    for frac in fracs:
        vals_raw, vals_z = [], []
        for t in study.tasks:
            i = study.idx(t, split)
            q = study.q[t][query_model][i]
            C_nat = study.c[t][query_model]
            s_nat = q @ C_nat.T
            s_adp = q @ adapted_cache[t].T
            k_nat = int(round(frac * len(C_nat)))
            idx_nat = perms[t][:k_nat]
            idx_adp = perms[t][k_nat:]

            scores_raw = s_adp.copy()
            if k_nat > 0:
                scores_raw[:, idx_nat] = s_nat[:, idx_nat]
            vals_raw.append(analyze(scores_raw, study.rel[t], i)["ndcg"])

            if k_nat > 0 and len(idx_adp) > 0:
                kn = min(100, k_nat)
                ka = min(100, len(idx_adp))
                tn = np.partition(s_nat[:, idx_nat], -kn, axis=1)[:, -kn:]
                ta = np.partition(s_adp[:, idx_adp], -ka, axis=1)[:, -ka:]
                sn_z = (s_nat - tn.mean(1, keepdims=True)) / np.maximum(tn.std(1, keepdims=True), 1e-6)
                sa_z = (s_adp - ta.mean(1, keepdims=True)) / np.maximum(ta.std(1, keepdims=True), 1e-6)
                scores_z = sa_z.copy()
                scores_z[:, idx_nat] = sn_z[:, idx_nat]
                vals_z.append(analyze(scores_z, study.rel[t], i)["ndcg"])
            else:
                vals_z.append(vals_raw[-1])
        out["uncalibrated"][str(frac)] = float(np.mean(vals_raw))
        out["zscore_top100"][str(frac)] = float(np.mean(vals_z))
    return out


# ------------------------------------------------------------------ hubs (GPA)
def fit_gpa(cal, models, iters=40, seed=0):
    """Uncentred Generalized Procrustes: one semi-orthogonal map per model into a shared space."""
    rng = np.random.default_rng(seed)
    dmax = max(cal[m].shape[1] for m in models)
    R = {}
    for m in models:
        dm = cal[m].shape[1]
        Q, _ = np.linalg.qr(rng.standard_normal((dmax, dm)))
        R[m] = Q[:, :dm].T.astype(np.float64)
    cal64 = {m: np.asarray(cal[m], dtype=np.float64) for m in models}
    for _ in range(iters):
        M = np.mean([cal64[m] @ R[m] for m in models], axis=0)
        for m in models:
            R[m] = fit_procrustes(cal64[m], M).A.astype(np.float64)
    return {m: R[m].astype(np.float32) for m in models}


def fit_affine_gpa(cal, models, iters=40, seed=0):
    """Centred + scaled Generalized Procrustes hub (Affine GPA).

    Centres each model and scales to unit RMS spread before fitting semi-orthogonal
    maps R[m] into a shared d_max-dimensional hub. To map source s -> target t:
      z_hat = norm(mu[t] + (rms[t] / rms[s]) * (z - mu[s]) @ R[s] @ R[t]^T).
    """
    rng = np.random.default_rng(seed)
    dmax = max(cal[m].shape[1] for m in models)
    mu, rms, Z0 = {}, {}, {}
    for m in models:
        X = np.asarray(cal[m], dtype=np.float64)
        mu[m] = X.mean(0)
        Xc = X - mu[m]
        rms[m] = float(np.linalg.norm(Xc))
        Z0[m] = Xc / max(rms[m], 1e-12)
    R = {}
    for m in models:
        dm = cal[m].shape[1]
        Q, _ = np.linalg.qr(rng.standard_normal((dmax, dm)))
        R[m] = Q[:, :dm].T.astype(np.float64)
    for _ in range(iters):
        M = np.mean([Z0[m] @ R[m] for m in models], axis=0)
        for m in models:
            R[m] = fit_procrustes(Z0[m], M).A.astype(np.float64)
    return dict(
        R={m: R[m].astype(np.float32) for m in models},
        mu={m: mu[m].astype(np.float32) for m in models},
        rms=rms,
    )


def pca_basis(X, k):
    Xc = np.asarray(X, dtype=np.float64) - np.mean(X, axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T.astype(np.float32)
