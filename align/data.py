"""NanoBEIR tasks, qrels, cached embeddings for eight embedding models across six families."""
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

# Eight models from six vendors/families, spanning 768 to 2048 dimensions.
MODELS = {
    "v3": dict(dim=1024, family="jina", backbone="xlmr", label="jina-embeddings-v3", short="v3"),
    "v4": dict(dim=2048, family="jina", backbone="qwen25vl", label="jina-embeddings-v4", short="v4"),
    "v5nano": dict(dim=768, family="jina", backbone="eurobert", label="jina-embeddings-v5-nano", short="v5n"),
    "v5small": dict(dim=1024, family="jina", backbone="qwen3", label="jina-embeddings-v5-small", short="v5s"),
    "qwen3_06b": dict(dim=1024, family="qwen", backbone="qwen3", label="Qwen3-Embedding-0.6B", short="q3"),
    "bgem3": dict(dim=1024, family="bge", backbone="xlmr", label="bge-m3", short="bge"),
    "e5large": dict(dim=1024, family="e5", backbone="xlmr", label="multilingual-e5-large-instruct", short="e5"),
    "arctic_m": dict(dim=768, family="snowflake", backbone="bert", label="snowflake-arctic-embed-m-v1.5", short="arc"),
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


def load_calibration(cache, model, role="mixed"):
    """Load paired AG News calibration embeddings.

    role:
      'mixed'   -> 5,000 rows (2,500 query-mode + 2,500 passage-mode)
      'query'   -> 2,500 query-mode rows
      'passage' -> 2,500 passage-mode rows
    """
    q_path = Path(cache) / f"cal5k_retquery_{model}_retrieval_query_embs.npy"
    p_path = Path(cache) / f"cal5k_retpassage_{model}_retrieval_passage_embs.npy"
    if role == "query":
        return norm(np.load(q_path))
    if role == "passage":
        return norm(np.load(p_path))
    q = np.load(q_path)
    p = np.load(p_path)
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
