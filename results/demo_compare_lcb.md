# Plumbing demonstration — LCB pruned adapter + compare_runs

**No model endpoint was available in this environment, so this artifact does
NOT contain any live inference results.** It demonstrates exactly two
plumbing claims using shipped reference data and a synthetic compare fixture:

**Strategy key** — **hybrid** (proposed): prioritizes high-disagreement (discriminating) items plus ~15% stratified anchors from agreement items for generalization. Baselines: **disagreement_only** (split-item pool, no anchors), **stratified_only** (stratified sample of all items, no disagreement preference), **random** (uniform random).

1. The `live_code_bench_pruned` adapter, when loaded with the r=0.10 hybrid
   cache, filters the shipped 315-item LCB set down to **the exact 32 items
   that cache encodes**.
2. The `compare_runs` CLI joins a full-run report against a pruned-run report
   on `(model, dataset)` and produces the correct score-delta / sample-
   reduction table.

**Live method evidence lives elsewhere**, not in this file:
- Selection-method validation across all 30-trial cells, including
  leave-one-model-out, hybrid-vs-random, Kendall τ_b and per-held-out-model
  preservation: see [`evalscope_ext/validation/summary.md`](../evalscope_ext/validation/summary.md)
  and the raw 1320-trial detail in
  [`evalscope_ext/validation/results.json`](../evalscope_ext/validation/results.json).
- The two graded handouts: [`handout_a.md`](../handout_a.md) and
  [`handout_b.md`](../handout_b.md).

---

## 1. Adapter loads + filters to exactly 32 / 315 items

The adapter was instantiated through the same `get_benchmark()` path the
evalscope CLI takes when a reviewer runs
`evalscope eval --datasets live_code_bench_pruned --dataset-args '...'`. No
shortcut — the registry resolves `live_code_bench_pruned` to the
`LiveCodeBenchPrunedAdapter` class and constructs it with a `TaskConfig`
that carries the index_file path under `extra_params`.

```python
import json, hashlib
import evalscope.benchmarks                       # triggers auto-discovery
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.api.dataset import Sample

config = TaskConfig(
    model='dummy',
    datasets=['live_code_bench_pruned'],
    dataset_args={
        'live_code_bench_pruned': {
            'extra_params': {
                'index_file': 'evalscope_ext/pruners/cache/lcb_hybrid_r010.json'
            }
        }
    },
)
adapter = get_benchmark('live_code_bench_pruned', config)
# → LiveCodeBenchPrunedAdapter, 32 selected hashes loaded
```

Then we run the SAME `sample_filter` logic over the 315 shipped
question_contents (these were content-verified in the prior verification
step against the LCB upstream parquet — see `FINDINGS.md` 2026-06-10 entry):

```python
def make_sample(question_content: str) -> Sample:
    # mirrors record_to_sample's hash injection
    h = hashlib.sha256(question_content.strip().encode('utf-8')).hexdigest()
    return Sample(
        input=[],
        target='',
        metadata={
            'evaluation_sample': '{}',
            'contest_date': '2024-09-22T00:00:00',
            'content_hash': h,
        },
    )

with open('/tmp/lcb_decoded_qcs.json') as f:
    qcs = json.load(f)

n_pass = sum(1 for qc in qcs.values()
             if adapter.sample_filter(make_sample(qc)))
```

### Observed output

```
2026-06-11 16:53:12 - evalscope - INFO: No eval_type is provided, setting eval_type to CHECKPOINT.
2026-06-11 16:53:12 - evalscope - INFO: [live_code_bench_pruned] Loaded 32 selected hashes from 'evalscope_ext/pruners/cache/lcb_hybrid_r010.json'
Adapter class:    LiveCodeBenchPrunedAdapter
Adapter module:   evalscope.benchmarks.live_code_bench_pruned.live_code_bench_pruned_adapter
Selected hashes loaded into adapter: 32
sample_filter result on shipped 315: 32/315 pass
Expected (from cache count):         32
Match: True
```

**Result:** the adapter correctly admits exactly the 32 items the
`lcb_hybrid_r010.json` cache names and rejects the other 283. This is the
filter path a live eval run would take per sample.

### What this does NOT show

This step does not run any model. It does not exercise the LCB sandbox
grader or fetch the upstream parquet — both are downstream of the adapter
load that we just validated. The selection-method side (whether the chosen
32 items actually preserve the rank of a held-out model) is the
validation-harness job, not this demo's.

---

## 2. `compare_runs` joins full ↔ pruned and prints the diff

We create two fixture Report directories matching the schema evalscope's
`Report.to_json()` emits (`dataset_name`, `model_name`, `score`, `num`,
`metrics`...). One claims a full LCB run scored 0.7460 over 315 items; the
other claims the r=0.10 pruned run scored 0.7813 over 32 items.

The numbers are illustrative; the demonstration is that `compare_runs`
correctly **joins** `live_code_bench` ↔ `live_code_bench_pruned` and
computes the deltas:

```bash
python -m evalscope_ext.tools.compare_runs \
  --full   results/demo_fixtures/full \
  --pruned results/demo_fixtures/pruned
```

### Observed output

```
# Full vs. pruned evaluation comparison

| Model       | Dataset         | Full score | Pruned score | Δ score | Full N | Pruned N | Sample reduction |
|---          |---              |---:        |---:          |---:     |---:    |---:      |---:              |
| demo-vlm-7b | live_code_bench | 0.7460     | 0.7813       | +0.0353 | 315    | 32       | 89.8%            |
```

**Result:** `compare_runs` correctly identifies the `<bench>` ↔
`<bench>_pruned` pairing (the canonicalisation strips the `_pruned` suffix
when joining), pulls each side's `score` and `num`, and computes
`Δ score = pruned − full` and `Sample reduction = 1 − pruned_N/full_N`.

### What this does NOT show

The fixture's `0.7460 → 0.7813` shift is invented for the demo. Real
score-delta numbers for a candidate model require an OpenAI-compatible
endpoint to run the actual eval against. The point this demo establishes is
that once such an eval lands its Report JSONs into the two directories,
`compare_runs` produces the right comparison row.

---

## Where the live-method evidence is

| Claim | Where it's proven |
|---|---|
| The 32 selected items preserve the rank of a held-out distinguishable model in 100% of 30 trials | `evalscope_ext/validation/summary.md` (LCB hybrid, held-out=gpt-oss-120b row at r=0.10) |
| Hybrid τ_b at AA-LCR r=0.10 is 2.1× random's with half the variance | `evalscope_ext/validation/summary.md` (AA-LCR Kendall τ_b table) |
| kimi-vs-minimax is statistically tied (p=0.805 LCB, p=0.767 AA-LCR), so neither pruner nor full benchmark can order that pair | `evalscope_ext/validation/summary.md` (pairwise z-test tables) |
| MMMU encoder-stress score formula + the triple-query JOINT signal (`lift_text`, `lift_pert`, three-state classification ABSENT/COARSE/HEALTHY) | `evalscope_ext/probes/encoder_probe.py` + 25 tests; design rationale in `DECISIONS.md` (entry dated 2026-06-13) |

This file proves only the plumbing: the adapter selects exactly what its
cache encodes, and `compare_runs` consumes the standard report format
correctly. Both are necessary preconditions for any future live-endpoint
eval to produce meaningful numbers.
