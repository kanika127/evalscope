# Task 2 — Benchmark Compression for evalscope

## Development Context

**Developed against evalscope commit**: `bf3bd26d96bfc9669a47b4249cc19a67e29b6a9d`

This fork implements benchmark pruning/compression as an extension to the public `modelscope/evalscope` codebase.

## Architecture

### Pruned Benchmarks

Two new benchmark adapters that extend the standard LCB and AA-LCR evaluations:

- **`evalscope/benchmarks/live_code_bench_pruned/`** — registers dataset name `live_code_bench_pruned`
  - Extends `LiveCodeBenchAdapter` (inherits date filtering, scoring logic)
  - Adds `index_file` extra_param to specify a JSON file of pruned sample IDs
  - Overrides `sample_filter()` to only evaluate samples in the pruned set

- **`evalscope/benchmarks/aa_lcr_pruned/`** — registers dataset name `aa_lcr_pruned`
  - Extends `AALCRAdapter` (inherits LLM judge, long-context prompt logic)
  - Same `index_file` mechanism for sample selection

### Pruning Algorithms

Module `evalscope_ext/` (top-level, separate from `evalscope/` package):

- **`evalscope_ext/pruners/`** — pruning strategy implementations
  - `base.py` — abstract `Pruner` base class
  - `lcb_pruner.py` — `LiveCodeBenchPruner` for code generation tasks
  - `aa_lcr_pruner.py` — `AALCRPruner` for long-context reasoning tasks

- **`evalscope_ext/tools/`** — utility tools
  - `compare_runs.py` — `CompareRuns` CLI tool to compare full vs. pruned evaluation results
  - Usage: `python -m evalscope_ext.tools.compare_runs --full ./results_full/ --pruned ./results_pruned/`

## Data Flow

### CLI Usage

```bash
# Full evaluation (baseline)
evalscope eval --model <model> --datasets live_code_bench_v5 --output ./results_full/

# Pruned evaluation
evalscope eval --model <model> --datasets live_code_bench_pruned \
  --dataset-args '{"live_code_bench_pruned": {"extra_params": {"index_file": "path/to/pruned_indices.json"}}}' \
  --output ./results_pruned/

# Compare results
python -m evalscope_ext.tools.compare_runs --full ./results_full/ --pruned ./results_pruned/
```

### Index File Format

Each pruner saves a JSON file of selected sample indices:

```json
[0, 2, 5, 7, 10, ...]  // list of sample indices to include
```

The pruned adapter loads this file via `index_file` param and filters at `sample_filter()` time.

## Sample Filter Hook

The key evalscope integration point is `DefaultDataAdapter.sample_filter(sample: Sample) → bool`:
- Called for each sample **after** `record_to_sample()` converts the raw record
- Returns `True` to include the sample in evaluation, `False` to exclude
- Our pruned adapters override this to check if `sample.id` (or `sample.metadata['question_id']`) is in the pruned set
- Parent class's `sample_filter()` is called first to preserve any existing filters (e.g., date range for LCB)

## Integration Points

1. **BenchmarkMeta `extra_params`** — configuration passed via `--dataset-args`
2. **DataAdapter inheritance** — pruned adapters extend existing adapter classes
3. **Registry auto-discovery** — `*_adapter.py` files under `benchmarks/` are auto-imported

## Notes

- The pruner algorithms live outside evalscope to avoid tight coupling
- `evalscope_ext` is a separate top-level module that imports `evalscope` as a dependency
- All pruning is deterministic and reproducible (seed-controlled if needed)
