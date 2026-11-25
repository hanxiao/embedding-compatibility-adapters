"""The protocol: fit on calibration, pick the operating point on validation
queries, report on held-out queries. Natives and per-query hit sets come along.
"""
import time
from pathlib import Path

import numpy as np

from . import adapters as A
from . import data
from .evaluate import analyze

CANDIDATES = {
    "procrustes": lambda X, Y: [A.fit_procrustes(X, Y)],
    "affine": lambda X, Y: [A.fit_affine(X, Y, c, s)
                            for c in (True, False) for s in (True, False)],
    "ridgecca": lambda X, Y: [A.fit_ridge_cca(X, Y, k) for k in (32, 64, 128, 256, 384, 512)],
    "krr-rbf": lambda X, Y: [A.fit_krr_rbf(X, Y, lam=l, sigma_mult=s)
                             for l in (1e-3, 1e-2) for s in (0.5, 1.0)],
    "sgd-linear-infonce": lambda X, Y: [A.fit_sgd_linear(X, Y)],
    "sgd-mlp-infonce": lambda X, Y: [A.fit_sgd_mlp(X, Y)],
}

CONTROLS = {
    "random": lambda X, Y: [A.fit_random(X.shape[1], Y.shape[1], seed=0)],
    "shuffled": lambda X, Y: [A.fit_shuffled(X, Y, seed=0)],
    "identity": lambda X, Y: [A.fit_identity(X.shape[1], Y.shape[1])],
}


class Study:
    def __init__(self, cache, models=None, seed=0, tasks=None, verbose=True):
        self.cache = Path(cache)
        self.models = models or list(data.MODELS)
        self.tasks = tasks or data.TASKS
        self.verbose = verbose
        self.cal = {m: data.load_calibration(self.cache, m) for m in self.models}
        self.q, self.c, self.rel = {}, {}, {}
        self.val, self.test = {}, {}
        for t in self.tasks:
            self.rel[t] = data.qrels(t, cache_json=str(self.cache / "qrels.json"))
            self.q[t], self.c[t] = {}, {}
            for m in self.models:
                q, c = data.load_task(self.cache, t, m)
                self.q[t][m], self.c[t][m] = q, c
            n = len(self.q[t][self.models[0]])
            self.val[t], self.test[t] = data.split_queries(n, seed=seed)
            if verbose:
                print(f"  {t}: {n} queries, {self.val[t].size} val / {self.test[t].size} test",
                      flush=True)

    # ---------------------------------------------------------------- scoring
    def idx(self, task, split):
        if split == "val":
            return self.val[task]
        if split == "test":
            return self.test[task]
        return np.arange(len(self.q[task][self.models[0]]))

    def scores(self, src, tgt, adapter, task, split):
        q = self.q[task][src][self.idx(task, split)]
        d = self.c[task][tgt]
        return adapter.transform_source(q) @ adapter.transform_target(d).T

    def native(self, model, task, split="test"):
        i = self.idx(task, split)
        return self.q[task][model][i] @ self.c[task][model].T

    def eval_method(self, src, tgt, adapter, split="test"):
        return {t: analyze(self.scores(src, tgt, adapter, t, split), self.rel[t],
                           self.idx(t, split)) for t in self.tasks}

    def eval_native(self, model, split="test"):
        out = {}
        for t in self.tasks:
            i = self.idx(t, split)
            out[t] = analyze(self.q[t][model][i] @ self.c[t][model].T, self.rel[t], i)
        return out

    @staticmethod
    def mean(res):
        return float(np.mean([r["ndcg"] for r in res.values()]))

    # ---------------------------------------------------------------- selection
    def run_pair(self, src, tgt, families=None, keep_hits=True):
        X, Y = self.cal[src], self.cal[tgt]
        out = {}
        for name, fn in (families or {**CANDIDATES, **CONTROLS}).items():
            t0 = time.time()
            cands = fn(X, Y)
            vals = [self.mean(self.eval_method(src, tgt, a, "val")) for a in cands]
            ad = cands[int(np.argmax(vals))]
            res = self.eval_method(src, tgt, ad, "test")
            out[name] = dict(
                test={t: r["ndcg"] for t, r in res.items()},
                test_mean=float(np.mean([r["ndcg"] for r in res.values()])),
                val=float(max(vals)), val_candidates=[float(v) for v in vals],
                meta=ad.meta(),
            )
            if keep_hits:
                out[name]["hits"] = {t: res[t]["hits"] for t in self.tasks}
                out[name]["qids"] = {t: res[t]["qids"] for t in self.tasks}
            if self.verbose:
                print(f"    {name:20s} val={max(vals):.4f} test={out[name]['test_mean']:.4f} "
                      f"{ad.meta().get('kind', '')} ({time.time() - t0:.1f}s)", flush=True)
        return out

    # ---------------------------------------------------------------- natives
    def run_native(self, split="test"):
        out = {}
        for m in self.models:
            per_task = {t: self.eval_native(m, split)[t]["ndcg"] for t in self.tasks}
            per_task_all = {t: self.eval_native(m, "all")[t]["ndcg"] for t in self.tasks}
            out[m] = dict(test=per_task, all=per_task_all,
                          test_mean=float(np.mean(list(per_task.values()))),
                          all_mean=float(np.mean(list(per_task_all.values()))))
            if self.verbose:
                print(f"  native {m:10s} test={out[m]['test_mean']:.4f} "
                      f"all={out[m]['all_mean']:.4f}", flush=True)
        return out


def failure_stats(src_hits, tgt_hits, adp_hits):
    r"""Dark-memory analogue over aligned per-query hit sets.

    recoverable = |A & (S \ T)| / |S \ T|   docs only the source model retrieves
    loss        = |T \ A| / |T|             native target hits the adapter drops
    keep        = |A & (T \ S)| / |T \ S|   target-only hits kept
    """
    n = dict(rec=0, rec_d=0, loss=0, loss_d=0, keep=0, keep_d=0)
    for S, T, Ad in zip(src_hits, tgt_hits, adp_hits):
        for d in S - T:
            n["rec_d"] += 1
            n["rec"] += d in Ad
        for d in T:
            n["loss_d"] += 1
            n["loss"] += d not in Ad
        for d in T - S:
            n["keep_d"] += 1
            n["keep"] += d in Ad
    return n


def align_hits(src_res, tgt_res, adp_hits, adp_qids, tasks):
    """Per-task counts, aligned by query id."""
    pooled = dict(rec=0, rec_d=0, loss=0, loss_d=0, keep=0, keep_d=0)
    per_task = {}
    for t in tasks:
        S = dict(zip(src_res[t]["qids"], src_res[t]["hits"]))
        T = dict(zip(tgt_res[t]["qids"], tgt_res[t]["hits"]))
        A = dict(zip(adp_qids[t], adp_hits[t]))
        common = sorted(set(S) & set(T) & set(A))
        st = failure_stats([S[q] for q in common], [T[q] for q in common], [A[q] for q in common])
        per_task[t] = st
        for k in pooled:
            pooled[k] += st[k]
    rate = lambda a, b: float(a / b) if b else float("nan")
    return dict(per_task=per_task,
                recoverable=rate(pooled["rec"], pooled["rec_d"]), n_recoverable=pooled["rec_d"],
                loss=rate(pooled["loss"], pooled["loss_d"]), n_target_hits=pooled["loss_d"],
                keep_target_only=rate(pooled["keep"], pooled["keep_d"]),
                n_target_only=pooled["keep_d"])
