"""Adapters: every map is fitted closed form unless the name says SGD.

A map is an object with transform_source(X) -> target-space vectors and
transform_target(D) -> the same space the source lands in (identity except for
two-sided methods such as CCA).
"""
import numpy as np

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except Exception:  # pragma: no cover
    HAS_TORCH = False


def _norm(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


class Map:
    name = "map"

    def transform_source(self, X):
        raise NotImplementedError

    def transform_target(self, D):
        return _norm(np.asarray(D, dtype=np.float32))

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
    """z_hat = mu_y + s (z - mu_x) A."""

    def __init__(self, A, s, mu_x, mu_y, name="affine"):
        super().__init__(A, name=name)
        self.s, self.mu_x, self.mu_y = float(s), mu_x, mu_y

    def transform_source(self, X):
        return _norm((np.asarray(X, dtype=np.float32) - self.mu_x) @ self.A * self.s + self.mu_y)

    def meta(self):
        return dict(super().meta(), s=self.s)


class TwoSided(Map):
    def __init__(self, Ax, Ay, mu_x, mu_y, k, name="ridgecca"):
        self.Ax, self.Ay = Ax, Ay
        self.mu_x, self.mu_y = mu_x, mu_y
        self.k, self.name = k, name

    def transform_source(self, X):
        return _norm((np.asarray(X, dtype=np.float32) - self.mu_x) @ self.Ax)

    def transform_target(self, D):
        return _norm((np.asarray(D, dtype=np.float32) - self.mu_y) @ self.Ay)

    def meta(self):
        return dict(kind=self.name, k=self.k, params=int(self.Ax.size + self.Ay.size))


# --------------------------------------------------------------------------- fits

def fit_procrustes(X, Y):
    U, _, Vt = np.linalg.svd(np.asarray(X, np.float64).T @ np.asarray(Y, np.float64),
                             full_matrices=False)
    return Linear(U @ Vt, "procrustes")


def fit_affine(X, Y, centre=True, scale=True):
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    mx = X.mean(0) if centre else np.zeros(X.shape[1])
    my = Y.mean(0) if centre else np.zeros(Y.shape[1])
    Xc, Yc = X - mx, Y - my
    U, _, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    A = U @ Vt
    s = float(np.linalg.norm(Yc) / max(np.linalg.norm(Xc), 1e-12)) if scale else 1.0
    return Affine(A, s, mx.astype(np.float32), my.astype(np.float32),
                  name="affine" + ("-c" if centre else "") + ("-s" if scale else ""))


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
    U, S, Vt = np.linalg.svd(Wx.T @ Sxy @ Wy, full_matrices=False)
    k = int(min(k, U.shape[1]))
    Ax = (Wx @ U[:, :k]).astype(np.float32)
    Ay = (Wy @ Vt[:k].T).astype(np.float32)
    return TwoSided(Ax, Ay, mx.astype(np.float32), my.astype(np.float32), k)


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


def fit_krr_rbf(X, Y, lam=1e-2, sigma_mult=1.0, max_n=2000, seed=0):
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
            return _norm(np.exp(-d / (2 * sigma2)) @ alpha).astype(np.float32)

        def meta(self):
            return dict(kind="krr-rbf", n=int(len(sq)), lam=lam, sigma2=float(sigma2))

    return KRR()


# --------------------------------------------------------------------------- SGD

def _info_nce_torch(X, Y, A0, hidden=None, epochs=300, bs=512, lr=1e-3, tau=0.05, seed=0):
    torch.manual_seed(seed)
    dev = "cuda" if HAS_TORCH else "cpu"
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    Yt = torch.tensor(_norm(Y), dtype=torch.float32, device=dev)
    n, ds, dt = len(X), X.shape[1], Y.shape[1]
    if hidden:
        net = torch.nn.Sequential(
            torch.nn.Linear(ds, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, dt)
        ).to(dev)
        params = net.parameters()
        fwd = lambda z: net(z)
    else:
        W = torch.nn.Parameter(torch.tensor(A0, device=dev))
        params = [W]
        fwd = lambda z: z @ W
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 8:
                continue
            z = torch.nn.functional.normalize(fwd(Xt[idx]), dim=1)
            t = Yt[idx]
            logits = z @ t.T / tau
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(idx), device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        out = fwd(Xt).cpu().numpy().astype(np.float32)
    return out, net if hidden else W.detach().cpu().numpy()


def fit_sgd_linear(X, Y, epochs=300, seed=0):
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)
    n = len(X)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    A0 = np.linalg.solve(Xc.T @ Xc / n + 1e-2 * np.eye(X.shape[1]), Xc.T @ Yc / n)
    out, _ = _info_nce_torch(X, Y, A0.astype(np.float32), epochs=epochs, seed=seed)
    A = np.linalg.lstsq(X, out, rcond=None)[0]
    return Linear(A, "sgd-linear-infonce")


def fit_sgd_mlp(X, Y, epochs=300, hidden=512, seed=0):
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)
    n = len(X)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    A0 = np.linalg.solve(Xc.T @ Xc / n + 1e-2 * np.eye(X.shape[1]), Xc.T @ Yc / n)
    out, net = _info_nce_torch(X, Y, A0.astype(np.float32), hidden=hidden, epochs=epochs, seed=seed)

    class MLP(Map):
        name = "sgd-mlp-infonce"

        def transform_source(self, Xn):
            dev = next(net.parameters()).device
            with torch.no_grad():
                z = net(torch.tensor(np.asarray(Xn, np.float32), device=dev))
            return _norm(z.cpu().numpy())

        def meta(self):
            return dict(kind="sgd-mlp-infonce", hidden=hidden, epochs=epochs)

    return MLP()
