# Embedding Compatibility Adapters

[[Report](report.pdf)]

Your embedding provider deprecated their model, you have millions of documents embedded with the old one, and re-embedding costs days of compute. A map fitted on a few thousand paired texts repairs the mismatch: one SVD, one matrix multiply, no training loop, no GPU.

Two numbers decide whether that map will work, and both come from the paired texts before you fit anything.

| diagnostic | what it measures | what it predicts | correlation over 42 pairs |
|---|---|---|---|
| frame correlation `rho` | mean cosine between the two models' embeddings of the same text | whether you need a map at all | 0.95 with the identity baseline, -0.91 with what a map adds |
| anisotropy `kappa` | `||mean|| / RMS(x - mean)` of one model | whether a rotation is enough | 0.87 with the gain of the affine form over the rotation |
| direction | expanding (`d_s < d_t`) or compressing | which failure to expect | a rotation loses 0.099 nDCG when it expands |

`rho` is the one nobody computes. CKA, the usual geometric gate, is invariant to rotation, so it cannot see the thing a map repairs: v5-nano and v5-small have CKA 0.958 and `rho = -0.001`, and doing nothing between them retrieves 0.002 nDCG@10. jina-v3 and bge-m3 have CKA 0.729 and `rho = 0.325`, and doing nothing already reaches 0.542.

## What a bridge actually buys

42 directed pairs, seven models from five families, 13 NanoBEIR tasks, 5,000 calibration texts, 429 held-out queries. Three numbers per pair, the old model on its own index, the adapted query on the new index, and the new model natively.

| method | nDCG@10 | of source | of target | significant wins | significant losses |
|---|---|---|---|---|---|
| orthogonal Procrustes | 0.5244 | 0.805 | 0.810 | - | - |
| affine (centre + scale + rotation) | 0.5718 | 0.882 | 0.882 | 14 | 0 |
| ridge CCA, rank selected on validation | 0.5708 | 0.881 | 0.881 | 12 | 7 |
| SGD linear, InfoNCE | 0.5360 | 0.827 | 0.827 | 7 | 26 |
| SGD 3-layer MLP, InfoNCE | 0.5232 | 0.807 | 0.807 | 7 | 28 |
| kernel ridge, RBF | 0.4764 | 0.735 | 0.732 | 7 | 26 |
| random semi-orthogonal map | 0.0027 | 0.004 | 0.004 | 0 | 42 |
| Procrustes on shuffled pairs | 0.0035 | 0.005 | 0.005 | 0 | 42 |

<img src="https://raw.githubusercontent.com/hanxiao/embedding-compatibility-adapters/main/heatmap.png" alt="adapted nDCG@10 as a fraction of the source model score, orthogonal Procrustes, 42 directed pairs" width="520">

The map is not a migration. A rotation recovers 0.81 of the old index's nDCG@10 and the affine form 0.88, and neither beats keeping the old index except between two consecutive checkpoints of one family. If you migrate, the documents you already serve get worse by 4 to 51 points of nDCG@10 and the new model has to earn that back on documents it handles better.

Where the map does earn its place is the documents only the old model retrieves: a rotated query recovers 0.28 of them on average, without touching any weights.

## Quick start

```python
import numpy as np
from adapter import diagnose, affine_align, procrustes_align

# X_source, Y_target: embeddings of the same few thousand texts, one row each
print(diagnose(X_source, Y_target))
# {'rho': 0.31, 'kappa_source': 0.27, 'kappa_target': 0.74, ...}

f = affine_align(X_source, Y_target)     # centre, rotate, rescale, re-centre
adapted_queries = f(query_embeddings)    # score against the target's own index

W = procrustes_align(X_source, Y_target) # orthogonal only, when kappa is low
```

```bash
uv run adapter.py     # synthetic demo: rotation-only vs rotation-plus-mean
```

Fit on paired texts, never on the evaluation set. The calibration should cover both roles a text plays: half embedded as queries, half as passages.

## Reproducing the study

```bash
pip install -r requirements.txt
python embed_local.py qwen3_06b     # local models on a GPU; the jina models come from the API
python run_all.py --models v3,v4,v5nano,v5small,qwen3_06b,bgem3,e5large --out results.json
python paper/analysis.py --results results.json --out paper/summary.md --tables paper/tabs
python paper/figures.py --results results.json --dir paper/figs
```

`align/` holds the pipeline: `data.py` (tasks, qrels, cached embeddings), `adapters.py` (every map, closed form except the two SGD controls), `evaluate.py`, `protocol.py` (fit on calibration, select on validation, report on test), `analyses.py` (predictors, chaining, the generalized-Procrustes hub, low rank).

`paper/` holds the report source, `results.json` with every number, and the scripts that turn it into the tables and figures.

## Limits

NanoBEIR is 50 queries per task, so per-pair differences under about 0.02 nDCG are noise. The calibration is AG News for all 42 pairs, the query split uses one seed, and the operating points are selected on 17 validation queries per task. The query is adapted and the target index is left native; converting the corpus into the new space instead is equivalent only for an orthogonal map.

## License

Apache 2.0
