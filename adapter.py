# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scikit-learn",
#     "sentence-transformers",
#     "datasets",
# ]
# ///
"""
Embedding Compatibility Adapter
================================
Bridge incompatible embedding spaces with a single SVD.

When your embedding provider deprecates a model, you don't need to re-embed
billions of documents. Train a Procrustes adapter on a small calibration set
and rotate the old embeddings into the new space.

Usage:
    uv run adapter.py --source jinaai/jina-embeddings-v3 --target jinaai/jina-embeddings-v4
    uv run adapter.py --source jinaai/jina-embeddings-v3 --target Qwen/Qwen3-Embedding-0.6B
"""

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Step 1: Load calibration data
# ---------------------------------------------------------------------------

def load_calibration_texts(n_samples: int = 5000, seed: int = 42) -> list[str]:
    """Load calibration texts from AG News dataset.

    AG News is a good calibration source: short texts, diverse topics,
    no domain bias. 5000 samples is enough - Procrustes converges fast
    because it only estimates an orthogonal matrix (d x d parameters,
    heavily constrained).
    """
    print(f"Loading {n_samples} calibration texts from AG News...")
    ds = load_dataset("ag_news", split="train")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=n_samples, replace=False)
    texts = [ds[int(i)]["text"] for i in indices]
    print(f"  Loaded {len(texts)} texts (avg {np.mean([len(t) for t in texts]):.0f} chars)")
    return texts


# ---------------------------------------------------------------------------
# Step 2: Generate embeddings
# ---------------------------------------------------------------------------

def get_cache_path(model_name: str, dataset_key: str) -> Path:
    """Deterministic cache path for embeddings."""
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.md5(f"{model_name}:{dataset_key}".encode()).hexdigest()[:12]
    safe_name = model_name.replace("/", "__")
    return cache_dir / f"{safe_name}_{key}.npy"


def encode_texts(model_name: str, texts: list[str], dataset_key: str = "calibration",
                 batch_size: int = 64) -> np.ndarray:
    """Encode texts with a sentence-transformers model, with caching."""
    cache_path = get_cache_path(model_name, dataset_key)
    if cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"  Encoding {len(texts)} texts with {model_name}...")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                              normalize_embeddings=True)
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s - shape: {embeddings.shape}")

    np.save(cache_path, embeddings)
    return embeddings


# ---------------------------------------------------------------------------
# Step 3: Train Procrustes adapter
# ---------------------------------------------------------------------------

def train_procrustes(X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
    """Train an orthogonal Procrustes adapter: find rotation W that maps source -> target.

    The Procrustes problem:
        minimize ||X_source @ W - X_target||_F
        subject to W^T W = I  (orthogonal)

    Closed-form solution via SVD:
        M = X_source^T @ X_target
        U, S, V^T = SVD(M)
        W = U @ V^T

    This works because:
    - Embedding spaces trained on similar data share geometric structure
    - An orthogonal rotation preserves norms and angles (distances, cosine similarities)
    - The constraint makes it impossible to overfit, even with few calibration samples
    - Only d x d parameters, but constrained to the orthogonal group O(d)

    When source and target have different dimensions, we zero-pad the smaller one.
    """
    d_source = X_source.shape[1]
    d_target = X_target.shape[1]
    d_max = max(d_source, d_target)

    # Zero-pad to match dimensions if needed
    if d_source < d_max:
        X_source = np.pad(X_source, ((0, 0), (0, d_max - d_source)))
    if d_target < d_max:
        X_target = np.pad(X_target, ((0, 0), (0, d_max - d_target)))

    print(f"  Training Procrustes adapter ({d_source}d -> {d_target}d, padded to {d_max}d)...")

    # Core computation: one SVD
    M = X_source.T @ X_target  # (d_max, d_max)
    U, S, Vt = np.linalg.svd(M, full_matrices=True)
    W = U @ Vt  # orthogonal rotation matrix

    # Sanity check: W should be orthogonal
    ortho_error = np.linalg.norm(W @ W.T - np.eye(d_max))
    print(f"  Orthogonality error: {ortho_error:.2e} (should be ~0)")

    # Report alignment quality
    X_adapted = X_source @ W
    residual = np.linalg.norm(X_adapted - X_target) / np.linalg.norm(X_target)
    print(f"  Relative residual: {residual:.4f}")

    return W


def apply_adapter(embeddings: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply the Procrustes rotation to embeddings."""
    d_emb = embeddings.shape[1]
    d_w = W.shape[0]

    # Zero-pad if needed
    if d_emb < d_w:
        embeddings = np.pad(embeddings, ((0, 0), (0, d_w - d_emb)))

    adapted = embeddings @ W

    # Truncate back to target dimension (columns of W that matter)
    # Target dim = number of non-padded columns in original target
    adapted = adapted[:, :W.shape[1]]

    # Re-normalize (rotation preserves norms, but padding may not)
    norms = np.linalg.norm(adapted, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    adapted = adapted / norms

    return adapted


# ---------------------------------------------------------------------------
# Step 4: Evaluate on NanoBEIR
# ---------------------------------------------------------------------------

NANOBEIR_DATASETS = [
    "climatefever", "dbpedia", "fever", "fiqa2018", "hotpotqa",
    "msmarco", "nfcorpus", "nq", "quoraretrieval", "scidocs",
    "arguana", "scifact", "touche2020",
]


def load_nanobeir_task(task_name: str) -> tuple[list[str], list[str], dict[str, set[str]]]:
    """Load a NanoBEIR task: queries, corpus, and relevance judgments."""
    ds_corpus = load_dataset(f"zeta-alpha-ai/NanoBeir-{task_name}", "corpus", split="train")
    ds_queries = load_dataset(f"zeta-alpha-ai/NanoBeir-{task_name}", "queries", split="train")
    ds_qrels = load_dataset(f"zeta-alpha-ai/NanoBeir-{task_name}", "qrels", split="train")

    corpus_texts = [doc["text"] for doc in ds_corpus]
    corpus_ids = [doc["_id"] for doc in ds_corpus]
    query_texts = [q["text"] for q in ds_queries]
    query_ids = [q["_id"] for q in ds_queries]

    # Build relevance map: query_id -> set of relevant corpus_ids
    qrels = {}
    for row in ds_qrels:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        if row.get("score", 1) > 0:
            qrels.setdefault(qid, set()).add(cid)

    # Map query_ids and corpus_ids for lookup
    return query_texts, corpus_texts, query_ids, corpus_ids, qrels


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Compute nDCG@k for a single query."""
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because positions are 1-indexed

    # Ideal DCG
    n_rel = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(query_embs: np.ndarray, corpus_embs: np.ndarray,
                       query_ids: list[str], corpus_ids: list[str],
                       qrels: dict[str, set[str]], k: int = 10) -> float:
    """Evaluate retrieval performance using cosine similarity and nDCG@k."""
    # Compute all query-corpus similarities at once
    similarities = cosine_similarity(query_embs, corpus_embs)  # (n_queries, n_corpus)

    scores = []
    for i, qid in enumerate(query_ids):
        if str(qid) not in qrels:
            continue
        # Rank corpus by similarity
        ranked_indices = np.argsort(-similarities[i])
        ranked_corpus_ids = [corpus_ids[j] for j in ranked_indices]
        score = ndcg_at_k(ranked_corpus_ids, qrels[str(qid)], k)
        scores.append(score)

    return np.mean(scores) if scores else 0.0


def run_nanobeir_evaluation(model_name: str, encode_fn, prefix: str = "") -> dict[str, float]:
    """Run NanoBEIR evaluation across all 13 tasks."""
    results = {}
    label = prefix if prefix else model_name

    for task in NANOBEIR_DATASETS:
        query_texts, corpus_texts, query_ids, corpus_ids, qrels = load_nanobeir_task(task)

        query_embs = encode_fn(query_texts, f"nanobeir_{task}_queries")
        corpus_embs = encode_fn(corpus_texts, f"nanobeir_{task}_corpus")

        score = evaluate_retrieval(query_embs, corpus_embs, query_ids, corpus_ids, qrels)
        results[task] = score
        print(f"  {label} | {task}: nDCG@10 = {score:.4f}")

    avg = np.mean(list(results.values()))
    results["avg"] = avg
    print(f"  {label} | AVERAGE: nDCG@10 = {avg:.4f}")
    return results


# ---------------------------------------------------------------------------
# Step 5: Main - put it all together
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a Procrustes adapter between two embedding models and evaluate on NanoBEIR"
    )
    parser.add_argument("--source", type=str, default="jinaai/jina-embeddings-v3",
                        help="Source model (the one being deprecated)")
    parser.add_argument("--target", type=str, default="jinaai/jina-embeddings-v4",
                        help="Target model (the replacement)")
    parser.add_argument("--n-cal", type=int, default=5000,
                        help="Number of calibration samples")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Encoding batch size")
    args = parser.parse_args()

    source_model = args.source
    target_model = args.target
    n_cal = args.n_cal

    print("=" * 70)
    print("Embedding Compatibility Adapter")
    print(f"  Source: {source_model}")
    print(f"  Target: {target_model}")
    print(f"  Calibration samples: {n_cal}")
    print("=" * 70)

    # Step 1: Load calibration data
    print("\n[Step 1] Loading calibration data")
    cal_texts = load_calibration_texts(n_cal)

    # Step 2: Generate calibration embeddings
    print("\n[Step 2] Generating calibration embeddings")
    source_cal = encode_texts(source_model, cal_texts, "calibration", args.batch_size)
    target_cal = encode_texts(target_model, cal_texts, "calibration", args.batch_size)

    # Step 3: Train Procrustes adapter
    print("\n[Step 3] Training Procrustes adapter")
    W = train_procrustes(source_cal, target_cal)
    print(f"  Adapter matrix shape: {W.shape}")

    # Save adapter
    adapter_path = Path(".cache") / "adapter_W.npy"
    np.save(adapter_path, W)
    print(f"  Saved adapter to {adapter_path}")

    # Step 4: Evaluate on NanoBEIR
    print("\n[Step 4] Evaluating on NanoBEIR (13 retrieval tasks)")

    # Native target performance
    print(f"\n--- Native target model ({target_model}) ---")
    target_results = run_nanobeir_evaluation(
        target_model,
        lambda texts, key: encode_texts(target_model, texts, key, args.batch_size),
        prefix="native-target"
    )

    # Native source performance
    print(f"\n--- Native source model ({source_model}) ---")
    source_results = run_nanobeir_evaluation(
        source_model,
        lambda texts, key: encode_texts(source_model, texts, key, args.batch_size),
        prefix="native-source"
    )

    # Adapted source -> target performance
    # For adapted evaluation: encode queries with target model, corpus with adapted source
    # This simulates the real scenario: new queries encoded with new model,
    # old corpus adapted from old embeddings
    print(f"\n--- Adapted: {source_model} corpus -> {target_model} space ---")
    d_target = target_cal.shape[1]

    def encode_adapted_corpus(texts, key):
        embs = encode_texts(source_model, texts, key, args.batch_size)
        return apply_adapter(embs, W)[:, :d_target]

    adapted_results = run_nanobeir_evaluation(
        source_model,
        lambda texts, key: (
            encode_texts(target_model, texts, key, args.batch_size)
            if "queries" in key
            else encode_adapted_corpus(texts, key)
        ),
        prefix="adapted"
    )

    # Step 5: Print comparison
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"{'Task':<20} {'Native Target':>14} {'Native Source':>14} {'Adapted':>14} {'Retention':>10}")
    print("-" * 72)
    for task in NANOBEIR_DATASETS + ["avg"]:
        native_t = target_results[task]
        native_s = source_results[task]
        adapted = adapted_results[task]
        retention = adapted / native_t * 100 if native_t > 0 else 0
        label = task.upper() if task == "avg" else task
        print(f"{label:<20} {native_t:>14.4f} {native_s:>14.4f} {adapted:>14.4f} {retention:>9.1f}%")

    print("\nRetention = adapted / native_target * 100")
    print("Values close to 100% mean the adapter preserves retrieval quality.")


if __name__ == "__main__":
    main()
