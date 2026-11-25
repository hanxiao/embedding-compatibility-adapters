"""Retrieval metrics and per-query hit sets."""
import numpy as np


def hits(scores, k=10):
    """Top-k document indices for every query row."""
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
    return idx[np.arange(len(idx))[:, None], order]


def analyze(scores, rel, qidx, k=10):
    """nDCG@k plus the set of relevant docs retrieved, aligned to qidx.

    Queries without labels are dropped, which matches the earlier pipeline.
    """
    ranked = hits(scores, k)
    vals, sets, qids = [], [], []
    for row, qi in enumerate(qidx):
        rels = rel.get(int(qi))
        if not rels:
            continue
        r = ranked[row]
        dcg = sum(1.0 / np.log2(j + 2) for j, d in enumerate(r[:k]) if d in rels)
        idcg = sum(1.0 / np.log2(j + 2) for j in range(min(len(rels), k)))
        vals.append(dcg / idcg)
        sets.append({int(d) for d in r[:k] if d in rels})
        qids.append(int(qi))
    return dict(ndcg=float(np.mean(vals)) if vals else float("nan"),
                per_query=np.array(vals, dtype=np.float64),
                hits=sets, qids=qids, ranked=ranked)
