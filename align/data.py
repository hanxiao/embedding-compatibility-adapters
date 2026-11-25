"""NanoBEIR tasks, qrels, cached embeddings for seven embedding models."""
import json
from pathlib import Path

import numpy as np

TASKS = [
    "NanoClimateFEVER", "NanoDBPedia", "NanoFEVER", "NanoFiQA2018",
    "NanoHotpotQA", "NanoMSMARCO", "NanoNFCorpus", "NanoNQ",
    "NanoQuoraRetrieval", "NanoSCIDOCS", "NanoArguAna", "NanoSciFact",
    "NanoTouche2020",
]
SYMMETRIC = {"NanoQuoraRetrieval", "NanoSCIDOCS", "NanoArguAna"}

# four jina models + three local models, one name per model everywhere
MODELS = {
    "v3": dict(dim=1024, family="jina", label="jina-v3"),
    "v4": dict(dim=2048, family="jina", label="jina-v4"),
    "v5nano": dict(dim=768, family="jina", label="jina-v5-nano"),
    "v5small": dict(dim=1024, family="jina", label="jina-v5-small"),
    "qwen3_06b": dict(dim=1024, family="qwen", label="Qwen3-Embedding-0.6B"),
    "bgem3": dict(dim=1024, family="bge", label="bge-m3"),
    "e5large": dict(dim=1024, family="e5", label="multilingual-e5-large-instruct"),
}


def norm(X):
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def _find(cache, pattern):
    hits = sorted(Path(cache).glob(pattern))
    if not hits:
        raise FileNotFoundError(f"{cache}/{pattern}")
    return hits[0]


def load_task(cache, task, model):
    tt = "text-matching" if task in SYMMETRIC else None
    q = np.load(_find(cache, f"nanobeir2_{task}_queries_{model}_{tt or 'retrieval_query'}_*.npy"))
    c = np.load(_find(cache, f"nanobeir2_{task}_corpus_{model}_{tt or 'retrieval_passage'}_*.npy"))
    return norm(q), norm(c)


def load_calibration(cache, model):
    q = np.load(Path(cache) / f"cal5k_retquery_{model}_retrieval_query_embs.npy")
    p = np.load(Path(cache) / f"cal5k_retpassage_{model}_retrieval_passage_embs.npy")
    return norm(np.vstack([q, p]))


def qrels(task, cache_json="qrels.json"):
    """{query_index: set(corpus_index)} in dataset order."""
    store = Path(cache_json)
    db = json.loads(store.read_text()) if store.exists() else {}
    if task in db:
        return {int(k): set(v) for k, v in db[task].items()}
    from datasets import load_dataset
    corpus = load_dataset("sentence-transformers/NanoBEIR-en", "corpus", split=task)
    rel = load_dataset("sentence-transformers/NanoBEIR-en", "qrels", split=task)
    qids = load_dataset("sentence-transformers/NanoBEIR-en", "queries", split=task)["_id"]
    cid2i = {cid: i for i, cid in enumerate(corpus["_id"])}
    qid2i = {qid: i for i, qid in enumerate(qids)}
    out = {}
    for row in rel:
        q, c = qid2i.get(row["query-id"]), cid2i.get(row["corpus-id"])
        if q is None or c is None:
            continue
        out.setdefault(q, set()).add(c)
    db[task] = {str(k): sorted(v) for k, v in out.items()}
    store.write_text(json.dumps(db))
    return out


def split_queries(n, seed=0, frac_val=1 / 3):
    """Item-level split, stratified by task. Same split for every pair."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * frac_val)))
    return np.sort(idx[:n_val]), np.sort(idx[n_val:])
