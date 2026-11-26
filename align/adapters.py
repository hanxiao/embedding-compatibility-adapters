"""Adapters: every map is fitted closed form unless the name says SGD.

A map exposes transform_source(X) -> target-space vectors and
transform_target(D) -> vectors in the same space (identity for all one-sided
adapters, non-identity only for two-sided methods such as Ridge CCA).
"""
import numpy as np

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_CUDA = False
    HAS_TORCH = False


def _norm(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


class Map:
    name = "map"
    two_sided = False

    def transform_source(self, X):
        raise NotImplementedError

    def transform_target(self, D):
        return _norm(D)

    def meta(self):
        return {}


class Linear(Map):
    def __init__(self, A, name="procrustes", extra=None):
        self.A = np.asarray(A, dtype=np.float32)
        self.name = name
        self.extra = extra or {}

    def transform_source(self, X):
        return _norm(np.asarray(X, dtype=np.float32) @ self.A)

    def meta(self):
        return dict(kind=self.name, shape=list(self.A.shape), **self.extra)


class Affine(Linear):
    """z_hat = norm(mu_y + s * (z - mu_x) @ A)."""

    def __init__(self, A, s, mu_x, mu_y, name="affine", extra=None):
        super().__init__(A, name=name, extra=extra)
        self.s = float(s)
        self.mu_x = np.asarray(mu_x, dtype=np.float32)
        self.mu_y = np.asarray(mu_y, dtype=np.float32)

    def transform_source(self, X):
        Z = (np.asarray(X, dtype=np.float32) - self.mu_x) @ self.A * self.s + self.mu_y
        return _norm(Z)

    def meta(self):
        return dict(super().meta(), s=self.s)


class TwoSided(Map):
    two_sided = True

    def __init__(self, Ax, Ay, mu_x, mu_y, k, name="ridgecca"):
        self.Ax = np.asarray(Ax, dtype=np.float32)
        self.Ay = np.asarray(Ay, dtype=np.float32)
        self.mu_x = np.asarray(mu_x, dtype=np.float32)
        self.mu_y = np.asarray(mu_y, dtype=np.float32)
        self.k = int(k)
        self.name = name

    def transform_source(self, X):
        return _norm((np.asarray(X, dtype=np.float32) - self.mu_x) @ self.Ax)

    def transform_target(self, D):
        return _norm((np.asarray(D, dtype=np.float32) - self.mu_y) @ self.Ay)

    def meta(self):
        return dict(kind=self.name, k=self.k, params=int(self.Ax.size + self.Ay.size))


# --------------------------------------------------------------------------- closed-form fits

def fit_procrustes(X, Y):
    U, _, Vt = np.linalg.svd(
        np.asarray(X, np.float64).T @ np.asarray(Y, np.float64),
        full_matrices=False,
    )
    return Linear(U @ Vt, "procrustes")


def fit_affine(X, Y, centre=True, scale=True, ls_scale=False):
    """Affine orthogonal Procrustes (Plato's Key form when centre=True, scale=True).

    When ls_scale=True, uses the least-squares optimal scalar
    s_ls = trace(Xc^T Yc R^T) / ||Xc||_F^2 instead of the Frobenius ratio
    ||Yc||_F / ||Xc||_F.
    """
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    mx = X.mean(0) if centre else np.zeros(X.shape[1])
    my = Y.mean(0) if centre else np.zeros(Y.shape[1])
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    A = U @ Vt
    if ls_scale:
        s = float(S.sum() / max(np.sum(Xc ** 2), 1e-12))
        suffix = ("-c" if centre else "") + "-ls"
    elif scale:
        s = float(np.linalg.norm(Yc) / max(np.linalg.norm(Xc), 1e-12))
        suffix = ("-c" if centre else "") + "-s"
    else:
        s = 1.0
        suffix = "-c" if centre else ""
    return Affine(A, s, mx, my, name="affine" + suffix)


def fit_ridge(X, Y, lam=1e-2, rescale=True):
    """Closed-form centred ridge regression, optionally Frobenius-rescaled."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    n = len(X)
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    A = np.linalg.solve(Xc.T @ Xc / n + lam * np.eye(X.shape[1]), Xc.T @ Yc / n)
    if rescale:
        pred = Xc @ A
        s = float(np.linalg.norm(Yc) / max(np.linalg.norm(pred), 1e-12))
    else:
        s = 1.0
    tag = "ridge-s" if rescale else "ridge"
    return Affine(A, s, mx, my, name=tag, extra=dict(lam=lam, rescale=rescale))


def fit_whitened_procrustes(X, Y, lam=1e-1):
    """Regularized ZCA-whitened Procrustes, mapped back into target space."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    n = len(X)
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    ex, Ux = np.linalg.eigh(Xc.T @ Xc / n + lam * np.eye(X.shape[1]))
    ey, Uy = np.linalg.eigh(Yc.T @ Yc / n + lam * np.eye(Y.shape[1]))
    Wx = (Ux / np.sqrt(np.maximum(ex, 1e-12))) @ Ux.T
    Wy_inv = (Uy * np.sqrt(np.maximum(ey, 1e-12))) @ Uy.T
    Wy = (Uy / np.sqrt(np.maximum(ey, 1e-12))) @ Uy.T
    Xw, Yw = Xc @ Wx, Yc @ Wy
    U, _, Vt = np.linalg.svd(Xw.T @ Yw, full_matrices=False)
    A = Wx @ (U @ Vt) @ Wy_inv
    pred = Xc @ A
    s = float(np.linalg.norm(Yc) / max(np.linalg.norm(pred), 1e-12))
    return Affine(A, s, mx, my, name="whitened-proc", extra=dict(lam=lam))


def fit_lowrank_affine(X, Y, k=256):
    """One-sided low-rank affine map (Plato's Key compression, zero corpus re-indexing)."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    k = int(min(k, X.shape[1], Y.shape[1], len(X) - 1))
    _, _, Vtx = np.linalg.svd(Xc, full_matrices=False)
    _, _, Vty = np.linalg.svd(Yc, full_matrices=False)
    Px, Py = Vtx[:k].T, Vty[:k].T
    Xk, Yk = Xc @ Px, Yc @ Py
    U, _, Vt = np.linalg.svd(Xk.T @ Yk, full_matrices=False)
    Wk = U @ Vt
    s = float(np.linalg.norm(Yk) / max(np.linalg.norm(Xk), 1e-12))
    A = Px @ Wk @ Py.T
    return Affine(A, s, mx, my, name=f"lowrank-affine-{k}", extra=dict(k=k))


def fit_ridge_cca(X, Y, k=128, lam=1e-3):
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    n = X.shape[0]
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    Sxx = Xc.T @ Xc / n + lam * np.eye(X.shape[1])
    Syy = Yc.T @ Yc / n + lam * np.eye(Y.shape[1])
    Sxy = Xc.T @ Yc / n
    ex, Ux = np.linalg.eigh(Sxx)
    ey, Uy = np.linalg.eigh(Syy)
    Wx = Ux / np.sqrt(np.maximum(ex, 1e-12))
    Wy = Uy / np.sqrt(np.maximum(ey, 1e-12))
    U, _, Vt = np.linalg.svd(Wx.T @ Sxy @ Wy, full_matrices=False)
    k = int(min(k, U.shape[1]))
    Ax = (Wx @ U[:, :k]).astype(np.float32)
    Ay = (Wy @ Vt[:k].T).astype(np.float32)
    return TwoSided(Ax, Ay, mx, my, k)


def fit_identity(ds, dt, name="identity"):
    A = np.zeros((ds, dt), np.float32)
    A[:min(ds, dt), :min(ds, dt)] = np.eye(min(ds, dt))
    return Linear(A, name)


def fit_random(ds, dt, seed=0, name="random"):
    """Haar random semi-orthogonal map ds -> dt."""
    rng = np.random.default_rng(seed)
    if dt >= ds:
        Q, _ = np.linalg.qr(rng.standard_normal((dt, ds)))
        A = Q[:, :ds].T
    else:
        Q, _ = np.linalg.qr(rng.standard_normal((ds, dt)))
        A = Q[:, :dt]
    return Linear(A, name)


def fit_shuffled(X, Y, seed=0, name="shuffled"):
    rng = np.random.default_rng(seed)
    return Linear(fit_procrustes(X, Y[rng.permutation(len(Y))]).A, name)


def fit_krr_rbf(X, Y, lam=1e-2, sigma_mult=1.0, max_n=2500, seed=0):
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        sel = rng.choice(len(X), max_n, replace=False)
        X, Y = X[sel], Y[sel]
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    sq = np.sum(X ** 2, 1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)
    sigma2 = np.median(d2[d2 > 0]) * sigma_mult
    K = np.exp(-d2 / (2 * sigma2))
    alpha = np.linalg.solve(K + lam * np.eye(len(X)), Y)

    class KRR(Map):
        name = "krr-rbf"

        def transform_source(self, Xn):
            Xn = np.asarray(Xn, np.float64)
            d = np.maximum((Xn ** 2).sum(1)[:, None] + sq[None, :] - 2 * Xn @ X.T, 0)
            return _norm(np.exp(-d / (2 * sigma2)) @ alpha)

        def meta(self):
            return dict(kind="krr-rbf", n=int(len(sq)), lam=lam, sigma2=float(sigma2))

    return KRR()


# --------------------------------------------------------------------------- GPU SGD fits

def _train_torch(X, Y, mode="linear", hidden=512, epochs=250, bs=512, lr=1e-3,
                 tau=0.05, seed=0):
    """Train linear or 3-layer residual MLP initialized from affine Procrustes,
    with 10% internal calibration holdout early stopping to prevent overfitting."""
    torch.manual_seed(seed)
    dev = "cuda" if HAS_CUDA else "cpu"
    aff = fit_affine(X, Y, centre=True, scale=True)
    A0 = (aff.A * aff.s).astype(np.float32)
    b0 = (aff.mu_y - aff.mu_x @ A0).astype(np.float32)

    rng = np.random.default_rng(seed)
    perm_all = rng.permutation(len(X))
    n_val = max(250, len(X) // 10)
    tr_idx, va_idx = perm_all[n_val:], perm_all[:n_val]

    Xtr = torch.tensor(X[tr_idx], dtype=torch.float32, device=dev)
    Ytr = torch.tensor(_norm(Y[tr_idx]), dtype=torch.float32, device=dev)
    Xva = torch.tensor(X[va_idx], dtype=torch.float32, device=dev)
    Yva = torch.tensor(_norm(Y[va_idx]), dtype=torch.float32, device=dev)

    ds, dt = X.shape[1], Y.shape[1]
    if mode == "linear":
        lin = torch.nn.Linear(ds, dt).to(dev)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(A0.T, device=dev))
            lin.bias.copy_(torch.tensor(b0, device=dev))
        model = lin
    else:
        # True 3-layer residual MLP initialized at the affine solution
        class ResMLP3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.skip = torch.nn.Linear(ds, dt)
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(ds, hidden),
                    torch.nn.LayerNorm(hidden),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(hidden, hidden),
                    torch.nn.LayerNorm(hidden),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(hidden, dt),
                )
                with torch.no_grad():
                    self.skip.weight.copy_(torch.tensor(A0.T))
                    self.skip.bias.copy_(torch.tensor(b0))
                    torch.nn.init.zeros_(self.net[-1].weight)
                    torch.nn.init.zeros_(self.net[-1].bias)

            def forward(self, z):
                return self.skip(z) + self.net(z)

        model = ResMLP3().to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    n_tr = len(Xtr)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=dev)
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            if len(idx) < 16:
                continue
            z = torch.nn.functional.normalize(model(Xtr[idx]), dim=1)
            t = Ytr[idx]
            logits = z @ t.T / tau
            nce = torch.nn.functional.cross_entropy(logits, torch.arange(len(idx), device=dev))
            cos = (1.0 - (z * t).sum(dim=1)).mean()
            loss = nce + cos
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                zv = torch.nn.functional.normalize(model(Xva), dim=1)
                lv = torch.nn.functional.cross_entropy(
                    zv @ Yva.T / tau, torch.arange(len(Xva), device=dev)
                ).item()
            if lv < best_loss:
                best_loss = lv
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    model.eval()
    return model


def fit_sgd_linear(X, Y, epochs=250, seed=0):
    model = _train_torch(X, Y, mode="linear", epochs=epochs, lr=5e-4, seed=seed)
    W = model.weight.detach().cpu().numpy().T
    b = model.bias.detach().cpu().numpy()
    return Affine(W, 1.0, np.zeros(X.shape[1], np.float32), b, name="sgd-linear-infonce")


def fit_sgd_mlp(X, Y, epochs=250, hidden=512, seed=0):
    model = _train_torch(X, Y, mode="mlp", hidden=hidden, epochs=epochs, lr=5e-4, seed=seed)

    class ResMLPMap(Map):
        name = "sgd-resmlp-infonce"

        def transform_source(self, Xn):
            dev = next(model.parameters()).device
            with torch.no_grad():
                z = model(torch.tensor(np.asarray(Xn, np.float32), device=dev))
            return _norm(z.cpu().numpy())

        def meta(self):
            return dict(kind="sgd-resmlp-infonce", hidden=hidden, epochs=epochs, layers=3)

    return ResMLPMap()
