# Validation summary — pruning preserves model rankings

Empirical test of the pruning method on shipped reference scores. 10 seeds × 3 leave-one-model-out splits = 30 trials per (benchmark, strategy, ratio) cell.

## Coarseness disclaimer (read first)

With only 3 reference models per benchmark, the model ranking is over 3 items. Kendall τ_b on 3 distinct values can take only 4 values: `{-1.0, -0.333, +0.333, +1.0}`. A uniform-random ranking lands at τ=+1.0 with probability 1/6 ≈ 17%. We mitigate by running 30 trials per cell and reporting τ-distribution + **held-out-model rank preservation rate** (the C2 claim) alongside τ. Read held-out-preservation as the headline; τ_mean as secondary.

AA-LCR carries an additional ±2–3% LLM-judge-noise band per item. Treat AA-LCR cell differences < ~0.07 in any rate as within-noise.


## lcb

**N items:** 315 · **Models:** gpt-oss-120b, kimi-k2.5, minimax-m2.5 · **Full-benchmark accuracy and rank (rank 1 = best):**


| Model | Full acc | Full rank |
|---|---:|---:|
| gpt-oss-120b | 0.765 | 1 |
| kimi-k2.5 | 0.629 | 2 |
| minimax-m2.5 | 0.619 | 3 |

**Pairwise full-benchmark gaps and significance (two-proportion z-test):**

| pair | Δ acc | p-value | distinguishable @ α=0.05? |
|---|---:|---:|---|
| gpt-oss-120b vs kimi-k2.5 | +13.7 pp | 0.000 | yes |
| gpt-oss-120b vs minimax-m2.5 | +14.6 pp | 0.000 | yes |
| kimi-k2.5 vs minimax-m2.5 | +1.0 pp | 0.805 | **no** (within noise) |

**Interpretation guard.** A pruned-set ranking that fails to put a within-noise pair in the same order as the full benchmark hasn't really 'failed' — the full benchmark itself doesn't distinguish those two models. The held-out preservation rate below blends distinguishable and noise-pair cases; the **per-holdout breakdown** two tables down separates them.


### Held-out-model rank preservation rate (HEADLINE — C2 claim)

Fraction of 30 trials in which the held-out model's rank position survived (it was never used during selection).

| ratio | hybrid | random | disagreement_only | stratified_only |
|---|---:|---:|---:|---:|
| 0.05 | 0.70 | 0.57 | 0.67 | 0.57 |
| 0.10 | 0.73 | 0.70 | 0.67 | 0.67 |
| 0.20 | 0.67 | 0.60 | 0.67 | 0.63 |
| 0.30 | 0.67 | 0.73 | 0.67 | 0.63 |
| 0.50 | 0.67 | 0.77 | 0.67 | 0.80 |
| 0.70 | 0.70 | 0.93 | 0.70 | 0.77 |

### Held-out preservation per held-out model (diagnostic)

Same headline metric, split by which model was held out (10 trials each). Reveals whether failure modes concentrate on the within-noise pair vs the distinguishable model.

**hybrid** (rate per held-out model):

| ratio | gpt-oss-120b | kimi-k2.5 | minimax-m2.5 |
|---|---:|---:|---:|
| 0.05 | 0.90 | 0.90 | 0.30 |
| 0.10 | 1.00 | 0.90 | 0.30 |
| 0.20 | 1.00 | 1.00 | 0.00 |
| 0.30 | 1.00 | 1.00 | 0.00 |
| 0.50 | 1.00 | 1.00 | 0.00 |
| 0.70 | 1.00 | 1.00 | 0.10 |

**random** (rate per held-out model):

| ratio | gpt-oss-120b | kimi-k2.5 | minimax-m2.5 |
|---|---:|---:|---:|
| 0.05 | 0.80 | 0.50 | 0.40 |
| 0.10 | 0.90 | 0.60 | 0.60 |
| 0.20 | 1.00 | 0.40 | 0.40 |
| 0.30 | 1.00 | 0.60 | 0.60 |
| 0.50 | 1.00 | 0.80 | 0.50 |
| 0.70 | 1.00 | 0.90 | 0.90 |

**disagreement_only** (rate per held-out model):

| ratio | gpt-oss-120b | kimi-k2.5 | minimax-m2.5 |
|---|---:|---:|---:|
| 0.05 | 0.90 | 0.90 | 0.20 |
| 0.10 | 1.00 | 1.00 | 0.00 |
| 0.20 | 1.00 | 1.00 | 0.00 |
| 0.30 | 1.00 | 1.00 | 0.00 |
| 0.50 | 1.00 | 1.00 | 0.00 |
| 0.70 | 1.00 | 1.00 | 0.10 |

**stratified_only** (rate per held-out model):

| ratio | gpt-oss-120b | kimi-k2.5 | minimax-m2.5 |
|---|---:|---:|---:|
| 0.05 | 1.00 | 0.40 | 0.30 |
| 0.10 | 1.00 | 0.60 | 0.40 |
| 0.20 | 1.00 | 0.60 | 0.30 |
| 0.30 | 1.00 | 0.60 | 0.30 |
| 0.50 | 1.00 | 0.60 | 0.80 |
| 0.70 | 1.00 | 0.60 | 0.70 |

### All-3-model rank preservation rate (secondary)

Fraction of trials where the full 3-model ranking was preserved exactly. Includes the 2 'seen' models, so partly circular.

| ratio | hybrid | random | disagreement_only | stratified_only |
|---|---:|---:|---:|---:|
| 0.05 | 0.50 | 0.40 | 0.47 | 0.27 |
| 0.10 | 0.67 | 0.50 | 0.53 | 0.43 |
| 0.20 | 0.67 | 0.40 | 0.67 | 0.37 |
| 0.30 | 0.67 | 0.60 | 0.67 | 0.37 |
| 0.50 | 0.67 | 0.50 | 0.67 | 0.63 |
| 0.70 | 0.70 | 0.90 | 0.70 | 0.63 |

### Kendall τ_b — mean ± std (tertiary)

| ratio | hybrid | random | disagreement_only | stratified_only |
|---|---:|---:|---:|---:|
| 0.05 | +0.720 ± 0.334 | +0.548 ± 0.467 | +0.725 ± 0.310 | +0.541 ± 0.386 |
| 0.10 | +0.794 ± 0.308 | +0.715 ± 0.321 | +0.769 ± 0.298 | +0.718 ± 0.350 |
| 0.20 | +0.778 ± 0.320 | +0.600 ± 0.332 | +0.794 ± 0.308 | +0.658 ± 0.316 |
| 0.30 | +0.778 ± 0.320 | +0.733 ± 0.332 | +0.778 ± 0.320 | +0.723 ± 0.290 |
| 0.50 | +0.778 ± 0.320 | +0.812 ± 0.256 | +0.778 ± 0.320 | +0.772 ± 0.317 |
| 0.70 | +0.816 ± 0.298 | +0.933 ± 0.203 | +0.800 ± 0.311 | +0.804 ± 0.294 |

### Smallest sufficient ratio

Smallest ratio at which the held-out-preservation rate reaches each threshold. `—` means no tested ratio reached it.

| strategy | rate ≥ 0.80 | rate ≥ 0.90 | rate = 1.00 |
|---|---:|---:|---:|
| hybrid | — | — | — |
| random | 0.70 (rate 0.93) | 0.70 (rate 0.93) | — |
| disagreement_only | — | — | — |
| stratified_only | 0.50 (rate 0.80) | — | — |


## aa_lcr

**N items:** 100 · **Models:** gpt-oss-120b, kimi-k2.5, minimax-m2.5 · **Full-benchmark accuracy and rank (rank 1 = best):**


| Model | Full acc | Full rank |
|---|---:|---:|
| gpt-oss-120b | 0.480 | 3 |
| kimi-k2.5 | 0.660 | 1 |
| minimax-m2.5 | 0.640 | 2 |

**Pairwise full-benchmark gaps and significance (two-proportion z-test):**

| pair | Δ acc | p-value | distinguishable @ α=0.05? |
|---|---:|---:|---|
| gpt-oss-120b vs kimi-k2.5 | -18.0 pp | 0.010 | yes |
| gpt-oss-120b vs minimax-m2.5 | -16.0 pp | 0.023 | yes |
| kimi-k2.5 vs minimax-m2.5 | +2.0 pp | 0.767 | **no** (within noise) |

**Interpretation guard.** A pruned-set ranking that fails to put a within-noise pair in the same order as the full benchmark hasn't really 'failed' — the full benchmark itself doesn't distinguish those two models. The held-out preservation rate below blends distinguishable and noise-pair cases; the **per-holdout breakdown** two tables down separates them.


### Held-out-model rank preservation rate (HEADLINE — C2 claim)

Fraction of 30 trials in which the held-out model's rank position survived (it was never used during selection).

| ratio | hybrid | random | disagreement_only | stratified_only |
|---|---:|---:|---:|---:|
| 0.05 | 0.57 | 0.37 | 0.43 | 0.67 |
| 0.10 | 0.77 | 0.57 | 0.73 | 0.53 |
| 0.20 | 0.73 | 0.70 | 0.77 | 0.83 |
| 0.30 | 0.73 | 0.77 | 0.77 | 0.70 |
| 0.50 | 0.90 | 0.87 | 0.90 | 0.77 |

### Held-out preservation per held-out model (diagnostic)

Same headline metric, split by which model was held out (10 trials each). Reveals whether failure modes concentrate on the within-noise pair vs the distinguishable model.

**hybrid** (rate per held-out model):

| ratio | gpt-oss-120b | kimi-k2.5 | minimax-m2.5 |
|---|---:|---:|---:|
| 0.05 | 0.50 | 0.60 | 0.60 |
| 0.10 | 0.60 | 0.90 | 0.80 |
| 0.20 | 0.70 | 0.70 | 0.80 |
| 0.30 | 0.90 | 0.40 | 0.90 |
| 0.50 | 1.00 | 0.70 | 1.00 |

**random** (rate per held-out model):

| ratio | gpt-oss-120b | kimi-k2.5 | minimax-m2.5 |
|---|---:|---:|---:|
| 0.05 | 0.40 | 0.60 | 0.10 |
| 0.10 | 0.70 | 0.50 | 0.50 |
| 0.20 | 0.90 | 0.70 | 0.50 |
| 0.30 | 1.00 | 0.70 | 0.60 |
| 0.50 | 1.00 | 0.80 | 0.80 |

**disagreement_only** (rate per held-out model):

| ratio | gpt-oss-120b | kimi-k2.5 | minimax-m2.5 |
|---|---:|---:|---:|
| 0.05 | 0.20 | 0.60 | 0.50 |
| 0.10 | 0.50 | 0.90 | 0.80 |
| 0.20 | 0.80 | 0.60 | 0.90 |
| 0.30 | 1.00 | 0.30 | 1.00 |
| 0.50 | 1.00 | 0.90 | 0.80 |

**stratified_only** (rate per held-out model):

| ratio | gpt-oss-120b | kimi-k2.5 | minimax-m2.5 |
|---|---:|---:|---:|
| 0.05 | 0.60 | 0.90 | 0.50 |
| 0.10 | 0.80 | 0.50 | 0.30 |
| 0.20 | 1.00 | 1.00 | 0.50 |
| 0.30 | 0.80 | 0.60 | 0.70 |
| 0.50 | 1.00 | 0.90 | 0.40 |

### All-3-model rank preservation rate (secondary)

Fraction of trials where the full 3-model ranking was preserved exactly. Includes the 2 'seen' models, so partly circular.

| ratio | hybrid | random | disagreement_only | stratified_only |
|---|---:|---:|---:|---:|
| 0.05 | 0.23 | 0.00 | 0.30 | 0.27 |
| 0.10 | 0.53 | 0.40 | 0.53 | 0.17 |
| 0.20 | 0.60 | 0.40 | 0.50 | 0.47 |
| 0.30 | 0.67 | 0.60 | 0.73 | 0.53 |
| 0.50 | 0.73 | 0.80 | 0.77 | 0.53 |

### Kendall τ_b — mean ± std (tertiary)

| ratio | hybrid | random | disagreement_only | stratified_only |
|---|---:|---:|---:|---:|
| 0.05 | +0.583 ± 0.382 | +0.182 ± 0.455 | +0.541 ± 0.424 | +0.551 ± 0.463 |
| 0.10 | +0.768 ± 0.354 | +0.370 ± 0.675 | +0.796 ± 0.310 | +0.489 ± 0.508 |
| 0.20 | +0.819 ± 0.290 | +0.745 ± 0.285 | +0.844 ± 0.221 | +0.789 ± 0.268 |
| 0.30 | +0.842 ± 0.266 | +0.782 ± 0.303 | +0.838 ± 0.285 | +0.699 ± 0.420 |
| 0.50 | +0.903 ± 0.205 | +0.867 ± 0.271 | +0.941 ± 0.137 | +0.769 ± 0.298 |

### Smallest sufficient ratio

Smallest ratio at which the held-out-preservation rate reaches each threshold. `—` means no tested ratio reached it.

| strategy | rate ≥ 0.80 | rate ≥ 0.90 | rate = 1.00 |
|---|---:|---:|---:|
| hybrid | 0.50 (rate 0.90) | 0.50 (rate 0.90) | — |
| random | 0.50 (rate 0.87) | — | — |
| disagreement_only | 0.50 (rate 0.90) | 0.50 (rate 0.90) | — |
| stratified_only | 0.20 (rate 0.83) | — | — |


---

## Reading guide

- **Held-out preservation rate** is the empirical answer to *'is this defensible for a 4th model the pruner never saw?'*. Higher is better; 1.00 is full preservation across all 30 trials.
- **All-3 preservation rate** includes the 2 training models, so it's partly circular and shown for completeness.
- **Kendall τ_b** is the standard rank-correlation metric. On 3 items it's coarse — read distribution shape, not point values.
- **Hybrid vs random** on the held-out column is the direct C3 check. Random hovers near its expected baseline; hybrid should sit clearly above. If it doesn't, the report says so.
