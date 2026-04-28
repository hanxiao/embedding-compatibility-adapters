# Embedding Compatibility Adapters

Your embedding provider just deprecated their model. You have billions of documents embedded with the old model. Re-embedding costs thousands of dollars and days of compute. What do you do?

This repo shows that a simple orthogonal rotation (Procrustes alignment, one SVD) trained on just 5000 calibration samples can bridge embedding spaces with minimal quality loss - if the models are geometrically similar. No neural networks, no training loops, no GPUs. One matrix multiplication at inference time.

## How it works

Two embedding models trained on similar data learn similar geometric structures. The spaces differ mainly by a rotation. Procrustes alignment finds the optimal orthogonal matrix W that maps one space to the other:

```
minimize  ||X_source @ W - X_target||_F
subject to  W^T W = I
```

The closed-form solution is a single SVD:

```
M = X_source^T @ X_target
U, S, V^T = SVD(M)
W = U @ V^T
```

At inference time, transforming an embedding is one matrix multiplication: `e_adapted = e_source @ W`. The orthogonality constraint means W preserves norms and angles - cosine similarities in the adapted space closely match the native target space.

Why only 5000 calibration samples? Because W is a d x d orthogonal matrix, and the orthogonality constraint is so strong that there are far fewer effective parameters than d^2. The calibration data just needs to span the embedding space reasonably well.

## Quick start

```bash
# Adapt jina-embeddings-v3 corpus to work with v4 queries
uv run adapter.py --source jinaai/jina-embeddings-v3 --target jinaai/jina-embeddings-v4

# Cross-vendor: bridge Jina to Qwen
uv run adapter.py --source jinaai/jina-embeddings-v3 --target Qwen/Qwen3-Embedding-0.6B

# Custom calibration size
uv run adapter.py --source model-a --target model-b --n-cal 10000
```

No `pip install` needed. The script uses `uv` inline script metadata to resolve dependencies automatically.

## Results

We evaluated Procrustes adapters across 12 model pairs on NanoBEIR (13 retrieval tasks, nDCG@10). We also compared against CCA, Kernel Ridge Regression (KRR), SGD-trained linear maps, and MLP adapters.

**Key finding: Procrustes beats all other methods on 9 out of 12 model pairs.** MLP is consistently the worst - embedding alignment is fundamentally a linear problem. Adding nonlinearity hurts.

### Native baselines

| Model | nDCG@10 |
|---|---|
| jina-embeddings-v5-nano | 0.667 |
| jina-embeddings-v5-small | 0.671 |
| jina-embeddings-v3 | 0.634 |
| jina-embeddings-v4 | 0.647 |
| Qwen3-Embedding-0.6B | 0.584 |

### Same-family Procrustes adaptation (5000 calibration samples)

| Source -> Target | Adapted nDCG@10 | Native Target | Retention |
|---|---|---|---|
| v5-nano -> v5-small | 0.666 | 0.671 | 99.3% |
| v5-small -> v5-nano | 0.660 | 0.667 | 98.9% |
| v3 -> v5-small | 0.548 | 0.671 | 81.7% |
| v3 -> v5-nano | 0.549 | 0.667 | 82.3% |
| v4 -> v5-small | 0.586 | 0.671 | 87.3% |
| v4 -> v5-nano | 0.587 | 0.667 | 88.0% |
| v3 -> v4 | 0.540 | 0.647 | 83.5% |
| v4 -> v3 | 0.434 | 0.634 | 68.5% |

### Cross-vendor Procrustes adaptation

| Source -> Target | Adapted nDCG@10 | Native Target | Retention |
|---|---|---|---|
| v5-nano -> Qwen3 | 0.613 | 0.584 | 105.0% |
| Qwen3 -> v5-nano | 0.563 | 0.667 | 84.4% |
| v5-small -> Qwen3 | 0.614 | 0.584 | 105.1% |
| Qwen3 -> v5-small | 0.562 | 0.671 | 83.8% |
| v3 -> Qwen3 | 0.535 | 0.584 | 91.6% |
| Qwen3 -> v3 | 0.455 | 0.634 | 71.8% |
| v4 -> Qwen3 | 0.516 | 0.584 | 88.4% |
| Qwen3 -> v4 | 0.534 | 0.647 | 82.5% |

The pattern is clear. Models that share similar training data and architecture (v5-nano and v5-small) have near-identical geometry - the adapter is essentially lossless. Models from different generations or vendors have less geometric overlap, and the adapter preserves less quality. CKA (Centered Kernel Alignment) between model pairs predicts adaptation quality well: CKA > 0.9 means near-lossless, CKA < 0.8 means significant degradation.

## When to use this

**Use Procrustes adaptation when:**
- Your embedding provider deprecated a model and you have a large corpus to migrate
- You want to test a new model without re-embedding everything
- The source and target models are from the same family or trained on similar data
- You need a quick bridge while planning a full re-embedding

**Re-embed instead when:**
- CKA similarity between source and target is below 0.8
- The models have fundamentally different architectures or training data
- You need maximum retrieval quality and can afford the compute
- The dimension mismatch is extreme (e.g., 384d to 4096d)

## References

- Schonemann, P. H. (1966). A generalized solution of the orthogonal Procrustes problem. *Psychometrika*, 31(1), 1-10.
- Wang et al. (2025). UniCon: Universal Compatibility for Old-New Embedding Spaces. [arXiv:2604.16678](https://arxiv.org/abs/2604.16678)

## License

Apache 2.0
