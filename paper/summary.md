# Results summary (8 models, 56 directed pairs)

## Native nDCG@10 (13 NanoBEIR tasks, 649 queries / 429 test)

| model | family | backbone | dim | kappa | all 649 | test 429 |
|---|---|---|---:|---:|---:|---:|
| jina-embeddings-v5-small | jina | qwen3 | 1024 | 0.34 | 0.6706 | 0.6822 |
| jina-embeddings-v5-nano | jina | eurobert | 768 | 0.35 | 0.6669 | 0.6800 |
| Qwen3-Embedding-0.6B | qwen | qwen3 | 1024 | 0.50 | 0.6479 | 0.6558 |
| snowflake-arctic-embed-m-v1.5 | snowflake | bert | 768 | 0.49 | 0.6479 | 0.6529 |
| jina-embeddings-v4 | jina | qwen25vl | 2048 | 0.99 | 0.6466 | 0.6527 |
| jina-embeddings-v3 | jina | xlmr | 1024 | 0.27 | 0.6342 | 0.6401 |
| bge-m3 | bge | xlmr | 1024 | 0.74 | 0.6040 | 0.6156 |
| multilingual-e5-large-instruct | e5 | xlmr | 1024 | 1.82 | 0.6104 | 0.6147 |

## Adapter families (mean over all directed pairs, Wilcoxon vs Procrustes)

| method | nDCG@10 | /source | /target | raw W/L | sig W/L |
|---|---:|---:|---:|---:|---:|
| Orthogonal Procrustes (ours) | 0.4998 | 0.766 | 0.771 | 0/0 | 0/0 |
| Affine, centre + scale (Plato's Key, fixed) | 0.5162 | 0.796 | 0.794 | 25/31 | 17/20 |
| Source-centred Procrustes (ours, fixed) | 0.5336 | 0.822 | 0.822 | 21/35 | 13/15 |
| Affine, centre + LS scale (fixed) | 0.4752 | 0.733 | 0.730 | 21/35 | 14/26 |
| Affine, val-selected (Plato's Key) | 0.5462 | 0.841 | 0.842 | 23/3 | 18/0 |
| Ridge regression, val-selected | 0.5029 | 0.775 | 0.774 | 18/38 | 13/25 |
| Whitened Procrustes, val-selected | 0.5210 | 0.803 | 0.802 | 28/28 | 20/18 |
| Ridge CCA (two-sided), val-selected | 0.5440 | 0.838 | 0.838 | 30/26 | 16/8 |
| SGD linear, InfoNCE | 0.5345 | 0.824 | 0.824 | 26/30 | 16/18 |
| SGD 3-layer residual MLP, InfoNCE | 0.5301 | 0.817 | 0.817 | 22/34 | 16/16 |
| Kernel ridge, RBF | 0.4549 | 0.701 | 0.699 | 12/44 | 9/36 |
| Identity, pad or truncate | 0.0567 | 0.090 | 0.091 | 1/55 | 1/53 |
| Procrustes on shuffled pairs | 0.0035 | 0.005 | 0.005 | 0/56 | 0/56 |
| Random semi-orthogonal map | 0.0027 | 0.004 | 0.004 | 0/56 | 0/56 |

## Multi-seed stability (5 random val/test splits)

| method | mean nDCG@10 | std | mean /source |
|---|---:|---:|---:|
| procrustes | 0.4931 | 0.0085 | 0.756 |
| affine-cs | 0.5105 | 0.0078 | 0.787 |
| affine | 0.5407 | 0.0085 | 0.833 |
| ridge | 0.4971 | 0.0082 | 0.766 |
| ridgecca | 0.5408 | 0.0078 | 0.834 |
| identity | 0.0568 | 0.0008 | 0.090 |
| src-centred | 0.5304 | 0.0080 | 0.817 |

## True dimensional direction (d_s vs d_t)

| method | expanding (d_s < d_t) (n=17) | same-dim (d_s = d_t) (n=22) | compressing (d_s > d_t) (n=17) |
|---|---:|---:|---:|
| procrustes | 0.5240 | 0.5015 | 0.4734 |
| affine-cs | 0.4897 | 0.5298 | 0.5251 |
| affine | 0.5400 | 0.5599 | 0.5346 |
| ridge | 0.4706 | 0.5204 | 0.5126 |
| whitened-proc | 0.4970 | 0.5335 | 0.5290 |
| ridgecca | 0.5402 | 0.5603 | 0.5266 |

## Quality direction (upgrade nat[s]<nat[t] vs downgrade nat[s]>nat[t])

| method | upgrade (n=28) | downgrade (n=28) | diff |
|---|---:|---:|---:|
| procrustes | 0.4511 | 0.5486 | -0.0975 |
| affine-cs | 0.5334 | 0.4991 | +0.0344 |
| affine | 0.5395 | 0.5529 | -0.0135 |
| ridge | 0.5160 | 0.4898 | +0.0262 |
| ridgecca | 0.5420 | 0.5459 | -0.0039 |

## Deployment: query-side vs corpus-side & hybrid progressive migration

- **procrustes** mean over 56 pairs: query-side `0.4998` vs corpus-side `0.5009` (delta `+0.0011`)
- **affine-cs** mean over 56 pairs: query-side `0.5162` vs corpus-side `0.5464` (delta `+0.0302`)
- **ridge** mean over 56 pairs: query-side `0.4980` vs corpus-side `0.5396` (delta `+0.0416`)
- **hybrid affine (uncalibrated)** mean: 0%=0.5464, 25%=0.3718, 50%=0.4729, 75%=0.5645, 100%=0.6492
- **hybrid affine (zscore_top100)** mean: 0%=0.5464, 25%=0.5270, 50%=0.5893, 75%=0.5933, 100%=0.6492

### Canonical examples: jina-v3 <-> jina-v5-nano

- `idx=v3|qry=v5nano`: query-side aff=0.5708 proc=0.5724 | corpus-side aff=0.5757 proc=0.5762
  - hybrid uncalibrated : 0%=0.5757 25%=0.3284 50%=0.4587 75%=0.5703 100%=0.6800
  - hybrid zscore_top100: 0%=0.5757 25%=0.5607 50%=0.6111 75%=0.6121 100%=0.6800
- `idx=v5nano|qry=v3`: query-side aff=0.5494 proc=0.5509 | corpus-side aff=0.5530 proc=0.5509
  - hybrid uncalibrated : 0%=0.5530 25%=0.3694 50%=0.4687 75%=0.5659 100%=0.6401
  - hybrid zscore_top100: 0%=0.5530 25%=0.5226 50%=0.5952 75%=0.6025 100%=0.6401
- `idx=v3|qry=v5small`: query-side aff=0.5760 proc=0.5723 | corpus-side aff=0.5798 proc=0.5723
  - hybrid uncalibrated : 0%=0.5798 25%=0.3267 50%=0.4528 75%=0.5678 100%=0.6822
  - hybrid zscore_top100: 0%=0.5798 25%=0.5604 50%=0.6125 75%=0.6170 100%=0.6822
- `idx=v5small|qry=v3`: query-side aff=0.5583 proc=0.5623 | corpus-side aff=0.5604 proc=0.5623
  - hybrid uncalibrated : 0%=0.5604 25%=0.3585 50%=0.4594 75%=0.5623 100%=0.6401
  - hybrid zscore_top100: 0%=0.5604 25%=0.5338 50%=0.6009 75%=0.6000 100%=0.6401

## Role-matched calibration ablation

| calibration set | query-side proc | query-side aff | corpus-side proc | corpus-side aff |
|---|---:|---:|---:|---:|
| mixed_5000 | 0.4998 | 0.5162 | 0.5009 | 0.5464 |
| mixed_2500 | 0.4842 | 0.4979 | 0.4854 | 0.5326 |
| query_2500 | 0.4782 | 0.4959 | 0.4797 | 0.5337 |
| passage_2500 | 0.4753 | 0.4609 | 0.4767 | 0.5223 |

## Chaining (all A -> B -> C triples)

- **Procrustes** (336 triples): median `0.9771`, p10 `0.9176`, min `0.7520`, >=0.95: `80.7%`
- **Affine (Plato's Key)** (336 triples): median `0.9695`, p10 `0.9136`, min `0.8324`, >=0.95: `70.2%`

## Shared Platonic hub (GPA, 40 iters, full 5k)

- **Uncentred GPA vs Procrustes**: hub `0.4996` vs pairwise `0.4998` (delta `-0.0002`, wins `23/56`)
- **Affine GPA vs Affine-CS**: hub `0.5165` vs pairwise `0.5162` (delta `+0.0002`, wins `26/56`)
- **Source-centred GPA vs Source-centred**: hub `0.5345` vs pairwise `0.5336` (delta `+0.0009`, wins `34/56`)

## One-sided low-rank Affine (zero corpus re-indexing)

| k | mean nDCG@10 | /full affine-cs | pairs |
|---:|---:|---:|---:|
| 16 | 0.0176 | 0.034 | 56 |
| 32 | 0.0403 | 0.076 | 56 |
| 64 | 0.1171 | 0.219 | 56 |
| 128 | 0.2688 | 0.507 | 56 |
| 256 | 0.4265 | 0.819 | 56 |
| 384 | 0.4782 | 0.924 | 56 |
| 512 | 0.4998 | 0.967 | 56 |
| 768 | 0.5127 | 0.993 | 56 |

## Calibration size

| n | Procrustes | Affine (Plato's Key) | Source-centred | Ridge |
|---:|---:|---:|---:|---:|
| 50 | 0.0287 | 0.0202 | 0.0314 | 0.0285 |
| 100 | 0.0904 | 0.0716 | 0.1104 | 0.0938 |
| 250 | 0.2650 | 0.2418 | 0.3064 | 0.2525 |
| 500 | 0.3780 | 0.3727 | 0.4263 | 0.3587 |
| 1000 | 0.4440 | 0.4523 | 0.4877 | 0.4367 |
| 2500 | 0.4842 | 0.4979 | 0.5215 | 0.4816 |
| 5000 | 0.4998 | 0.5162 | 0.5336 | 0.4980 |

## Predictors

| feature | r(identity) | r(affine - id) | r(affine - proc) | r(proc/src) | r(antisym) |
|---|---:|---:|---:|---:|---:|
| rho | +0.951 | -0.862 | +0.092 | +0.119 | +0.000 |
| kappa_source | +0.097 | -0.147 | +0.882 | -0.744 | -0.683 |
| kappa_target | +0.212 | -0.224 | -0.209 | +0.162 | +0.683 |
| cka | -0.073 | +0.319 | -0.060 | +0.313 | +0.000 |
| nagree | -0.057 | +0.337 | -0.111 | +0.385 | +0.000 |
| eff_rank | +0.060 | +0.095 | -0.365 | +0.470 | +0.000 |
| dim_ratio | -0.071 | +0.065 | +0.080 | -0.064 | -0.123 |
| resid_gap | -0.000 | +0.003 | -0.023 | +0.021 | +0.032 |

antisymmetric share of variance in procrustes/source: `0.441`
antisymmetric share of variance in affine/source: `0.053`

## Recoverable failures (dark memories)

- **procrustes**: mean recoverable `0.235` (range `0.008` to `0.464`), mean target-hit loss `0.338`
- **src-centred**: mean recoverable `0.258` (range `0.138` to `0.488`), mean target-hit loss `0.282`
- **affine**: mean recoverable `0.255` (range `0.145` to `0.464`), mean target-hit loss `0.272`
- **affine-cs**: mean recoverable `0.217` (range `0.071` to `0.354`), mean target-hit loss `0.310`

## Wall-clock latency

- `pair`: `v5nano->v3 (768->1024, n=5000)`
- `fit_procrustes_ms`: `302.04410800070036`
- `fit_affine_ms`: `338.4176079998724`
- `fit_ridge_ms`: `144.21471800051222`
- `single_query_us`: `75.82933800040337`
- `corpus_100k_s`: `1.0517585850011528`
- `corpus_1m_est_s`: `10.517585850011528`