# evalscope (Task 2 fork: benchmark pruning)

This is a private fork of [modelscope/evalscope](https://github.com/modelscope/evalscope), developed for Cerebras' AI Engineer — Model Quality challenge, **Task 2**.

**Developed against** `modelscope/evalscope` **commit `bf3bd26`** ([`bf3bd26d96bfc9669a47b4249cc19a67e29b6a9d`](https://github.com/modelscope/evalscope/commit/bf3bd26d96bfc9669a47b4249cc19a67e29b6a9d)) — the upstream README is preserved at [`README_UPSTREAM.md`](./README_UPSTREAM.md).

---

## What this fork adds

A benchmark-compression extension to evalscope, implemented as a thin in-tree addition. **No upstream files are modified** — every change is additive.

```
evalscope_ext/                      # Top-level extension package (no upstream coupling)
├── pruners/
│   ├── core.py                     # Universal pruning core (stdlib only)
│   │                               #   • PruningInputs / PruningResult / prune()
│   │                               #   • Hybrid + 3 baselines (random, stratified_only, disagreement_only)
│   │                               #   • 1-PL Rasch opt-in; custom_tiers plug-in for non-disagreement signals
│   ├── precompute_lcb.py           # Selection scripts (1 per benchmark)
│   ├── precompute_aa_lcr.py        #   ← reads shipped reviews, writes selection cache JSONs
│   ├── precompute_mmmu.py          #   ← MMMU stream-from-HF mode included
│   ├── cache/                      # 56 pre-computed selection caches:
│   │                               #   24 LCB + 16 AA-LCR + 16 MMMU
│   │                               #   filename pattern: <bench>_<strategy>_r<ratio*100>.json
│   └── tests/
│       └── test_core.py            # 34 pytest tests
├── probes/
│   └── encoder_probe.py            # Triple-query (full / text-only / perturbed) image-encoder probe
│       └── tests/
│           └── test_encoder_probe.py  # 11 pytest tests
├── validation/
│   ├── run_validation.py           # LOMO + multi-seed sweep, produces results.json + summary.md
│   ├── metrics.py                  # Kendall τ_b, Spearman ρ (pure stdlib)
│   ├── results.json                # 1320 trials from the validation harness
│   ├── summary.md                  # Handout-ready summary table
│   └── tests/
│       └── test_metrics.py         # 18 pytest tests
└── tools/
    └── compare_runs.py             # Full-vs-pruned diff CLI (the spec's run-contract third command)

evalscope/benchmarks/               # Three pruned-adapter registrations (additive to upstream)
├── live_code_bench_pruned/
├── aa_lcr_pruned/
└── mmmu_pruned/
```

### Test counts

```
63 tests pass (34 core + 11 encoder_probe + 18 metrics)
```

### Handouts

The two graded handouts:

| File | Audience | Length |
|---|---|---|
| [`handout_a.md`](./handout_a.md) | Technical engineer | 1 page |
| [`handout_b.md`](./handout_b.md) | Mixed (PM / sales engineer / dev / test) | ½ page |

Background documentation:

| File | Contents |
|---|---|
| [`DECISIONS.md`](./DECISIONS.md) | Running log of key design choices with rationale |
| [`FINDINGS.md`](./FINDINGS.md) | Running log of verified facts, data findings, validation results |
| [`evalscope_ext/validation/summary.md`](./evalscope_ext/validation/summary.md) | Per-(benchmark, strategy, ratio) preservation rates |

---

## Setup

```bash
# Fresh venv — evalscope requires pydantic ≥ 2.5.0
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The `pydantic ≥ 2.x` requirement is upstream evalscope's, not added by this fork. A pydantic 1.x environment will fail with `cannot import name 'JsonValue' from 'pydantic'` at adapter-import time.

The pruning core itself depends only on the Python stdlib; the only extra dependency the extension introduces (and only at probe-runtime) is `tiktoken` (for the LCB content-hash recompute path) and `openai` + optional `Pillow` (for `encoder_probe`).

---

## Run contract — pruned eval and compare

Standard evalscope CLI; the three pruned benchmarks register as drop-in datasets.

```bash
# 1. Full evaluation (baseline)
evalscope eval --model <candidate> --datasets live_code_bench --output ./results_full/

# 2. Pruned evaluation — pick a cache from evalscope_ext/pruners/cache/
evalscope eval --model <candidate> \
  --datasets live_code_bench_pruned \
  --dataset-args '{"live_code_bench_pruned": {"extra_params": {"index_file": "evalscope_ext/pruners/cache/lcb_hybrid_r010.json"}}}' \
  --output ./results_pruned/

# 3. Compare
python -m evalscope_ext.tools.compare_runs --full ./results_full/ --pruned ./results_pruned/
```

The same pattern for the other two benchmarks (substitute `aa_lcr_pruned` or `mmmu_pruned`, and a matching cache):

```bash
# AA-LCR — hybrid 50-item subset (r=0.50)
evalscope eval --model <candidate> --datasets aa_lcr_pruned \
  --dataset-args '{"aa_lcr_pruned": {"extra_params": {"index_file": "evalscope_ext/pruners/cache/aa_lcr_hybrid_r050.json"}}}' \
  --output ./results_pruned/

# MMMU — hybrid 30% encoder-stress subset (r=0.30)
evalscope eval --model <candidate> --datasets mmmu_pruned \
  --dataset-args '{"mmmu_pruned": {"extra_params": {"index_file": "evalscope_ext/pruners/cache/mmmu_hybrid_r030.json"}}}' \
  --output ./results_pruned/
```

### Cache filename convention

```
<bench>_<strategy>_r<ratio×100, zero-padded>.json
e.g. lcb_hybrid_r010.json   → LCB, hybrid strategy, prune_ratio=0.10
     aa_lcr_random_r030.json → AA-LCR, random baseline, prune_ratio=0.30
     mmmu_hybrid_r020.json   → MMMU, hybrid encoder-stress, prune_ratio=0.20
```

Ratios available out of the box:

| Benchmark | Available ratios | Strategies (per ratio) |
|---|---|---|
| LCB    | 0.05, 0.10, 0.20, 0.30, 0.50, 0.70 | hybrid, random, disagreement_only, stratified_only |
| AA-LCR | 0.10, 0.20, 0.30, 0.50              | (same 4) |
| MMMU   | 0.10, 0.20, 0.30, 0.50              | (same 4) |

---

## Regenerating the selection caches

Each benchmark has a precompute script that reads the shipped reference reviews (and predictions where needed) and runs `prune()` per (strategy, ratio).

```bash
# LCB
python -m evalscope_ext.pruners.precompute_lcb \
  --reviews-dir     <…>/Evals/Part 1/reviews \
  --predictions-dir <…>/Evals/Part 1/predictions \
  --key-file        /tmp/lcb_index_to_key.json \
  --output-dir      evalscope_ext/pruners/cache \
  --ratios          0.05 0.10 0.20 0.30 0.50 0.70 \
  --strategies      hybrid random disagreement_only stratified_only

# AA-LCR
python -m evalscope_ext.pruners.precompute_aa_lcr \
  --reviews-dir <…>/Evals/Part 1/reviews \
  --output-dir  evalscope_ext/pruners/cache \
  --ratios      0.10 0.20 0.30 0.50

# MMMU (shipped 660 path)
python -m evalscope_ext.pruners.precompute_mmmu \
  --predictions-dir <…>/Evals/MMMU/predictions/glm-4.5v-fp8 \
  --reviews-dir     <…>/Evals/MMMU/reviews/glm-4.5v-fp8 \
  --output-dir      evalscope_ext/pruners/cache \
  --ratios          0.10 0.20 0.30 0.50

# MMMU (full 12K path — streams from HF, no 25GB local image pull)
python -m evalscope_ext.pruners.precompute_mmmu --source hf \
  --output-dir evalscope_ext/pruners/cache \
  --ratios     0.05 0.10 0.20
```

All scripts are deterministic per `--rng-seed`; default seed is 0.

---

## Running the validation harness

The validation harness re-runs leave-one-model-out across 10 seeds × 3 LOMO splits × 4 strategies × all ratios on both LCB and AA-LCR (MMMU has only one reference model so LOMO is structurally inapplicable).

```bash
python -m evalscope_ext.validation.run_validation \
  --reviews-dir "<…>/Evals/Part 1/reviews"
```

Output:

```
evalscope_ext/validation/results.json   — all 1320 trials, raw per-trial detail
evalscope_ext/validation/summary.md     — handout-ready tables
```

Runtime ≈ 3 seconds end-to-end. No network, no live model needed.

---

## Encoder probe runtime (Part B)

The triple-query encoder probe runs against any OpenAI-compatible chat-completions endpoint:

```bash
OPENAI_API_KEY=…  OPENAI_BASE_URL=…  \
  python -m evalscope_ext.probes.encoder_probe \
    --index-file evalscope_ext/pruners/cache/mmmu_hybrid_r030.json \
    --model      <vlm_model_name> \
    --hf-repo    MMMU/MMMU \
    --hf-split   validation \
    --variants   full text_only            # Q3 perturbed optional: append "perturbed"
    --output-dir ./probe_results/
```

It writes per-item outcomes and a `encoder_lift_by_stratum.json`. The module is unit-tested end-to-end with a fake client; the live path is identical and requires only the env vars above.

---

## Running the test suites

```bash
# All extension tests
pytest evalscope_ext/

# Specifically
pytest evalscope_ext/pruners/tests/test_core.py        # 34
pytest evalscope_ext/probes/tests/test_encoder_probe.py # 11
pytest evalscope_ext/validation/tests/test_metrics.py   # 18
```

```
63 passed in <0.2s
```

---

## What was changed in `evalscope/` itself

```
evalscope/benchmarks/live_code_bench_pruned/    NEW
evalscope/benchmarks/aa_lcr_pruned/              NEW
evalscope/benchmarks/mmmu_pruned/                NEW
```

All three are subclasses of the respective upstream adapters (`LiveCodeBenchAdapter`, `AALCRAdapter`, `MMMUAdapter`) registered under new names via `@register_benchmark`. They override only:

1. `__init__` — load the selected-id set from `extra_params["index_file"]`
2. `sample_filter` — keep the sample iff its stable key is in the selected set
3. (LCB, AA-LCR only) `record_to_sample` — inject the SHA-256 content hash into `sample.metadata` for filtering. MMMU uses upstream `metadata.id` directly, no hashing.

No upstream file is modified. The pruned adapters are auto-discovered by evalscope's registry walk over `evalscope/benchmarks/`.

---

## License

Inherits from upstream evalscope ([Apache 2.0](./LICENSE)). The extension code in `evalscope_ext/` and the new adapters under `evalscope/benchmarks/*_pruned/` are released under the same license.
