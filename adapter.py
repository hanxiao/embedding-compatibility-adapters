# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
# ]
# ///
"""
Embedding Compatibility Adapter - Procrustes Alignment

Bridge incompatible embedding spaces with a single SVD.
Takes two numpy arrays (source, target embeddings on shared calibration texts),
returns the orthogonal rotation matrix that maps source -> target space.

Usage as library:
    W = procrustes_align(X_source, X_target)
    adapted = apply_adapter(embeddings, W, target_dim)

Usage standalone (runs synthetic demo):
    uv run adapter.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core: Procrustes alignment
# ---------------------------------------------------------------------------

def procrustes_align(X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
    """Find the orthogonal matrix W that best maps X_source -> X_target.

    Solves the orthogonal Procrustes problem:
        minimize  ||X_source @ W - X_target||_F
        subject to  W^T W = I

    Solution: SVD of the cross-covariance matrix M = X_source^T @ X_target.
    W = U @ V^T from that SVD. This is the unique optimum.

    Why it works: embedding models trained on similar data learn similar geometry.
    The spaces differ mainly by rotation. Orthogonality preserves norms and angles,
    so cosine similarities transfer directly.

    Args:
        X_source: (n, d_source) calibration embeddings from source model
        X_target: (n, d_target) calibration embeddings from target model
                  (same texts, same order)

    Returns:
        W: (d_padded, d_padded) orthogonal rotation matrix
    """
    n = X_source.shape[0]
    assert X_target.shape[0] == n, "Source and target must have same number of samples"

    d_source = X_source.shape[1]
    d_target = X_target.shape[1]
    d_max = max(d_source, d_target)

    # Zero-pad the smaller dimension so both live in R^d_max
    if d_source < d_max:
        X_source = np.pad(X_source, ((0, 0), (0, d_max - d_source)))
    if d_target < d_max:
        X_target = np.pad(X_target, ((0, 0), (0, d_max - d_target)))

    # Cross-covariance matrix
    M = X_source.T @ X_target  # (d_max, d_max)

    # SVD gives the optimal rotation
    U, S, Vt = np.linalg.svd(M, full_matrices=True)
    W = U @ Vt

    # Verify orthogonality
    ortho_err = np.linalg.norm(W @ W.T - np.eye(d_max))
    print(f"  Procrustes fit: d_source={d_source}, d_target={d_target}, "
          f"padded={d_max}, ortho_error={ortho_err:.2e}")

    # Residual: how well does W align the calibration data?
    residual = np.linalg.norm(X_source @ W - X_target) / np.linalg.norm(X_target)
    print(f"  Relative residual: {residual:.4f}")

    return W


def apply_adapter(embeddings: np.ndarray, W: np.ndarray, target_dim: int) -> np.ndarray:
    """Apply Procrustes rotation and truncate/renormalize to target dimension.

    Args:
        embeddings: (n, d_source) source embeddings
        W: rotation matrix from procrustes_align
        target_dim: output dimension (d_target from original alignment)

    Returns:
        (n, target_dim) adapted embeddings, L2-normalized
    """
    d_emb = embeddings.shape[1]
    d_w = W.shape[0]

    if d_emb < d_w:
        embeddings = np.pad(embeddings, ((0, 0), (0, d_w - d_emb)))

    adapted = (embeddings @ W)[:, :target_dim]

    # Re-normalize (rotation preserves norms, but padding + truncation may not)
    norms = np.linalg.norm(adapted, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return adapted / norms


# ---------------------------------------------------------------------------
# Evaluation utilities (numpy only)
# ---------------------------------------------------------------------------

def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between rows of A and rows of B.

    Both A and B should be L2-normalized for this to be a simple matmul.
    """
    # Normalize just in case
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return A @ B.T


def ndcg_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int = 10) -> float:
    """nDCG@k for a single query."""
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, doc_id in enumerate(ranked_ids[:k])
        if doc_id in relevant_ids
    )
    n_rel = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(query_embs: np.ndarray, corpus_embs: np.ndarray,
                       qrels: dict[int, set[int]], k: int = 10) -> float:
    """Compute mean nDCG@k over all queries.

    Args:
        query_embs: (n_queries, d) query embeddings
        corpus_embs: (n_corpus, d) corpus embeddings
        qrels: mapping from query index -> set of relevant corpus indices

    Returns:
        Mean nDCG@k
    """
    sims = cosine_similarity(query_embs, corpus_embs)
    scores = []
    for qi, rel_set in qrels.items():
        ranked = np.argsort(-sims[qi]).tolist()
        scores.append(ndcg_at_k(ranked, rel_set, k))
    return np.mean(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Synthetic demo
# ---------------------------------------------------------------------------

def run_demo():
    """Demonstrate Procrustes alignment on synthetic data.

    Real embeddings live on a low-dimensional manifold (not random in R^d).
    We simulate this with embeddings drawn from a rank-32 subspace, then
    rotated + lightly noised to create the "target" space. This mirrors
    how same-family models share geometric structure.
    """
    rng = np.random.default_rng(42)

    d = 128           # same dimension for simplicity
    rank = 32         # intrinsic dimensionality (realistic for embeddings)
    noise = 0.02      # small perturbation (models trained on similar data)
    n_cal = 2000
    n_queries = 100
    n_corpus = 5000

    print("=" * 60)
    print("Procrustes Adapter - Synthetic Demo")
    print(f"  dim={d}, intrinsic_rank={rank}, noise={noise}")
    print(f"  calibration={n_cal}, queries={n_queries}, corpus={n_corpus}")
    print("=" * 60)

    # Shared low-rank basis (the "semantic structure" both models capture)
    basis = np.linalg.qr(rng.standard_normal((d, rank)))[0]  # (d, rank) orthonormal

    # Ground-truth rotation between the two spaces
    R_true, _ = np.linalg.qr(rng.standard_normal((d, d)))

    def make_embeddings(n):
        """Generate structured embeddings on a low-rank manifold."""
        coords = rng.standard_normal((n, rank))
        embs = coords @ basis.T  # project into d-dimensional space
        norms = np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
        return embs / norms

    def to_target(source):
        """Simulate target model: rotate + light noise, then normalize."""
        target = source @ R_true + rng.normal(0, noise, source.shape)
        norms = np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
        return target / norms

    # Calibration data
    cal_source = make_embeddings(n_cal)
    cal_target = to_target(cal_source)

    # Step 1: Train adapter
    print("\n[1] Training Procrustes adapter on calibration data")
    W = procrustes_align(cal_source, cal_target)

    # Step 2: Test data (disjoint from calibration)
    print("\n[2] Generating test queries and corpus")
    query_source = make_embeddings(n_queries)
    corpus_source = make_embeddings(n_corpus)
    query_target = to_target(query_source)
    corpus_target = to_target(corpus_source)

    # Relevance labels: top-5 nearest neighbors in native target space
    true_sims = cosine_similarity(query_target, corpus_target)
    qrels = {i: set(np.argsort(-true_sims[i])[:5].tolist()) for i in range(n_queries)}

    # Step 3: Evaluate
    print("\n[3] Evaluating retrieval (nDCG@10)")

    score_native = evaluate_retrieval(query_target, corpus_target, qrels)
    print(f"  Native target:    nDCG@10 = {score_native:.4f}")

    score_no_adapter = evaluate_retrieval(query_target, corpus_source, qrels)
    print(f"  No adapter:       nDCG@10 = {score_no_adapter:.4f}")

    corpus_adapted = apply_adapter(corpus_source, W, d)
    score_adapted = evaluate_retrieval(query_target, corpus_adapted, qrels)
    print(f"  With adapter:     nDCG@10 = {score_adapted:.4f}")

    retention = score_adapted / score_native * 100 if score_native > 0 else 0
    print(f"\n  Retention: {retention:.1f}% of native target quality")

    print("\n" + "=" * 60)
    print("The adapter recovers most of the native retrieval quality")
    print("from a simple orthogonal rotation trained on calibration data.")
    print("=" * 60)

    return W


if __name__ == "__main__":
    run_demo()
