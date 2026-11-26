"""Worked production example: zero-reindex migration between jina-embeddings-v3 and jina-embeddings-v5-nano.

Demonstrates the three real-world deployment patterns:
  1. Forward query-side:  documents are frozen in jina-v3; search with new jina-v5-nano queries
  2. Reverse query-side:  documents are indexed in jina-v5-nano; legacy jina-v3 clients keep searching
  3. Progressive hybrid:  re-index gradually (0% -> 25% -> 50% -> 75% -> 100%) without score collapse
                          using top-100 per-shard z-score normalization

Runs out of the box with synthetic vectors, or pass --cache ./cache to run on real
cached jina-v3 and jina-v5-nano NanoBEIR embeddings.

usage:
    python examples/v3_v5nano_migration.py
    python examples/v3_v5nano_migration.py --cache cache
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adapter import AffineMap, affine_align, diagnose


def search_topk(query_embs: np.ndarray, doc_embs: np.ndarray, k: int = 10):
    """Exact cosine top-k search (vectors assumed L2-normalized)."""
    scores = query_embs @ doc_embs.T
    idx = np.argpartition(-scores, kth=min(k, scores.shape[1]) - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
    top_idx = idx[np.arange(len(idx))[:, None], order]
    top_scores = np.take_along_axis(scores, top_idx, axis=1)
    return top_idx, top_scores


def merge_two_shards_zscore(
    q_native: np.ndarray,
    docs_native: np.ndarray,
    docs_adapted: np.ndarray,
    top_k: int = 10,
    pool_k: int = 100,
):
    """Merge hits from a newly re-indexed shard and a corpus-adapted legacy shard.

    Naive single-index cosine search drops up to 0.25 nDCG@10 at 25% migration
    because adapted vectors have a lower cosine mean and attenuated peak than
    native vectors. Z-scoring each shard's top-100 scores per query eliminates
    that dip with zero extra model calls.
    """
    s_nat = q_native @ docs_native.T
    s_adp = q_native @ docs_adapted.T
    kn = min(pool_k, s_nat.shape[1])
    ka = min(pool_k, s_adp.shape[1])
    tn = np.partition(s_nat, -kn, axis=1)[:, -kn:]
    ta = np.partition(s_adp, -ka, axis=1)[:, -ka:]
    z_nat = (s_nat - tn.mean(1, keepdims=True)) / np.maximum(tn.std(1, keepdims=True), 1e-6)
    z_adp = (s_adp - ta.mean(1, keepdims=True)) / np.maximum(ta.std(1, keepdims=True), 1e-6)
    merged = np.concatenate([z_nat, z_adp], axis=1)
    return np.argsort(-merged, axis=1)[:, :top_k]


def run_synthetic():
    rng = np.random.default_rng(0)
    # Simulate v3 (1024d) and v5-nano (768d) sharing a 128d semantic manifold
    n_cal, n_docs, n_q = 2000, 1000, 20
    B_v3 = np.linalg.qr(rng.standard_normal((1024, 128)))[0]
    B_v5 = np.linalg.qr(rng.standard_normal((768, 128)))[0]

    def enc(Z, B, noise=0.12):
        E = Z @ B.T + noise * rng.standard_normal((len(Z), B.shape[0]))
        return (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32)

    Z_cal = rng.standard_normal((n_cal, 128))
    cal_v3, cal_v5 = enc(Z_cal, B_v3), enc(Z_cal, B_v5)

    Z_docs = rng.standard_normal((n_docs, 128))
    docs_v3, docs_v5 = enc(Z_docs, B_v3), enc(Z_docs, B_v5)

    # Queries match the first n_q documents with small perturbation
    Z_q = Z_docs[:n_q] + 0.35 * rng.standard_normal((n_q, 128))
    q_v3, q_v5 = enc(Z_q, B_v3), enc(Z_q, B_v5)

    # ------------------------------------------------------------------
    # Scenario 1: Documents frozen in v3, search with new v5-nano queries
    # ------------------------------------------------------------------
    print("Scenario 1: v3 frozen index + new v5-nano queries (zero re-indexing)")
    print("  diagnostics:", diagnose(cal_v5, cal_v3))
    f_v5_to_v3 = affine_align(cal_v5, cal_v3)
    q_v5_in_v3 = f_v5_to_v3(q_v5)
    top1_fwd = search_topk(q_v5_in_v3, docs_v3, k=1)[0][:, 0]
    print(f"  Recall@1 against frozen v3 index: {np.mean(top1_fwd == np.arange(n_q)):.3f}")

    # ------------------------------------------------------------------
    # Scenario 2: Reverse -- documents indexed in v5-nano, legacy v3 queries
    # ------------------------------------------------------------------
    print("\nScenario 2: v5-nano index + legacy v3 queries")
    f_v3_to_v5 = affine_align(cal_v3, cal_v5)
    q_v3_in_v5 = f_v3_to_v5(q_v3)
    top1_rev = search_topk(q_v3_in_v5, docs_v5, k=1)[0][:, 0]
    print(f"  Recall@1 against v5-nano index:   {np.mean(top1_rev == np.arange(n_q)):.3f}")

    # ------------------------------------------------------------------
    # Scenario 3: Corpus-side conversion + progressive 50% hybrid migration
    # ------------------------------------------------------------------
    print("\nScenario 3: 50% migrated index (half native v5-nano, half adapted v3)")
    docs_v3_as_v5 = f_v3_to_v5(docs_v3[n_docs // 2 :])
    top_hybrid = merge_two_shards_zscore(q_v5, docs_v5[: n_docs // 2], docs_v3_as_v5, top_k=1)
    print(f"  Hybrid Recall@1 (z-scored merge): {np.mean(top_hybrid[:, 0] == np.arange(n_q)):.3f}")

    # Save/load check
    f_v5_to_v3.save("/tmp/v5nano_to_v3_example.npz")
    reloaded = AffineMap.load("/tmp/v5nano_to_v3_example.npz")
    assert np.allclose(reloaded(q_v5), q_v5_in_v3, atol=1e-5)
    print("\nSaved and reloaded adapter successfully:", reloaded)


def run_real(cache_dir: str):
    from align.protocol import Study
    from align import analyses

    study = Study(cache_dir, models=["v3", "v5nano"], verbose=False)
    print("Real NanoBEIR evaluation (13 tasks, 429 held-out queries):")
    print("  Native v3:     ", f"{study.mean(study.eval_native('v3', 'test')):.4f}")
    print("  Native v5-nano:", f"{study.mean(study.eval_native('v5nano', 'test')):.4f}")

    # Scenario 1: v3 frozen index, v5-nano queries
    f_fwd = affine_align(study.cal["v5nano"], study.cal["v3"])
    sc_fwd = np.mean(list(analyses.eval_map(study, "v5nano", "v3", f_fwd).values()))
    print(f"  Scenario 1 (v5-nano queries -> frozen v3 index): {sc_fwd:.4f}")

    # Scenario 2: v5-nano index, legacy v3 queries
    f_rev = affine_align(study.cal["v3"], study.cal["v5nano"])
    sc_rev = np.mean(list(analyses.eval_map(study, "v3", "v5nano", f_rev).values()))
    print(f"  Scenario 2 (legacy v3 queries -> v5-nano index): {sc_rev:.4f}")

    # Scenario 3: Corpus-side conversion of v3 docs into v5-nano space + hybrid migration
    hyb = analyses.eval_hybrid_migration(study, "v3", "v5nano", f_rev)
    print("  Scenario 3 (progressive v3 -> v5-nano re-indexing):")
    print("    uncalibrated : " + "  ".join(f"{int(float(k)*100):3d}%={v:.4f}" for k, v in hyb["uncalibrated"].items()))
    print("    zscore_top100: " + "  ".join(f"{int(float(k)*100):3d}%={v:.4f}" for k, v in hyb["zscore_top100"].items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="", help="optional path to cached NanoBEIR embeddings")
    args = ap.parse_args()
    if args.cache:
        run_real(args.cache)
    else:
        run_synthetic()
