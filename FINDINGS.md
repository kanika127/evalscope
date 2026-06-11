# Findings

Running log of verified facts and data observations for the Task 2 build.
Every entry is something we can point at evidence for. Append as we go; do
not retroactively edit established facts (correct them in a new dated entry).

---

## 2026-06-10 — LCB shipped data: structure

**Files** (each 315 lines, `Evals/Part 1/`):
- `predictions/live_code_bench_v5__{gpt-oss-120b,kimi-k2.5,minimax-m2.5}.jsonl`
- `reviews/live_code_bench_v5__{gpt-oss-120b,kimi-k2.5,minimax-m2.5}.jsonl`

**Prediction record top-level keys:** `index, model, model_output, messages, metadata`.

**Score path on reviews:** `sample_score.score.value["pass"]` ∈ {0.0, 1.0}.

**Critical structural observation:** `messages` on prediction records contains
ONLY the assistant's response (role=assistant, content=[reasoning, text]).
**The original input prompt is not preserved as text** anywhere in the shipped
files. Prediction `metadata` is `{}`. The only byte-exact representation of
the input prompt is `model_output.prompt_token_ids` (list of 623 ints).

**Index space:** 0..314 across both prediction and review files; sets are
identical. Review files are sorted by index; prediction files are not (first
record's index is 27).

---

## 2026-06-10 — LCB stable key: SHA-256 of decoded question_content, all 315 unique

**Method.** Decoded `prompt_token_ids` via tiktoken `o200k_harmony`, extracted
the substring between literal `### Question:\n` and `\n\n### Format:`,
applied `.strip()`, took SHA-256 hex.

**Results:**
- 315 / 315 prompts decoded cleanly.
- 315 / 315 content hashes unique (zero collisions).
- 0 duplicate-content groups.

**Cross-check against parquet (partial).** Downloaded LCB v5 parquet shards
(167 rows total, Sep 2024 – Jan 2025) before pivoting to the content-hash
approach. 160 of 315 shipped indices had a decoded `question_content` that
exactly matched a v5 parquet row's `question_content`, and each match
resolved to a distinct upstream `question_id` (e.g. `abc387_b` style). The
remaining 155 are in older LCB versions (v4 and earlier) that we did not
download.

**Spot check (indices 0, 1, 50, 200, 314).** Each decoded prompt's first
~100 chars matched a recognizable LCB problem statement; each computed hash
is reproducible from `/tmp/lcb_decoded_qcs.json`.

**Refutation of prior claim.** A prior session reported "315 samples, 313
unique problems". That claim is **wrong**. All 315 are distinct by exact
content. Closest near-duplicate pair (indices 176, 180) shares the intro
paragraph but describes different attack rules (knight-like vs queen-like)
and produces different sample outputs.

**Deliverables.**
- `/tmp/lcb_index_to_key.json` — full {shipped_index → content_hash} map
- `/tmp/lcb_decoded_qcs.json` — all 315 decoded question_contents
- `/tmp/lcb_v5_verify_report.txt` — verification report

---

## 2026-06-10 — LCB score distribution (315 items × 3 models)

| Tier | Count | % |
|---|---|---|
| anchor_easy (all 3 pass) | 158 | 50.2% |
| split_hard (1/3 pass) | 62 | 19.7% |
| split_easy (2/3 pass) | 49 | 15.6% |
| anchor_hard (none pass) | 46 | 14.6% |

**Discrimination pool size:** 111 / 315 (35.2%). For a target of N=110
selected items, hybrid can fit all split items inside the discrimination
budget. Below ~30% target, hybrid IS the discrimination pool plus a few
anchors. Above ~35%, hybrid spills into anchors and behaves like stratified-
only over the anchor slice.

---

## 2026-06-10 — AA-LCR shipped data: structure

**Files** (each 100 lines):
- `predictions/aa_lcr__{gpt-oss-120b,kimi-k2.5,minimax-m2.5}.jsonl`
- `reviews/aa_lcr__{gpt-oss-120b,kimi-k2.5,minimax-m2.5}.jsonl`

**Prediction `metadata` carries the original question text** —
`metadata.question` is byte-identical to the review's
`sample_score.sample_metadata.question` for the same index. (Unlike LCB,
which preserved nothing.)

**No `prompt_token_ids`** in AA-LCR predictions. Not needed — the question is
directly accessible.

**Score path on reviews:** `sample_score.score.value["acc"]` ∈ {0.0, 1.0}.
Review records additionally carry `sample_score.sample_id`,
`sample_score.group_id`, and `sample_score.sample_metadata` (with `question`,
`data_source_urls`, `input_tokens`).

**Index space:** 0..99 in both prediction and review files; identical sets;
neither sorted.

---

## 2026-06-10 — AA-LCR `group_id` / `sample_id` rejected: just relabeled position

**Evidence.** Across all 300 review records (3 models × 100 indices):
```
gpt-oss-120b:  index == sample_id == group_id   ✓ all 100
kimi-k2.5:     index == sample_id == group_id   ✓ all 100
minimax-m2.5:  index == sample_id == group_id   ✓ all 100
```
They are integer 0..99 in lockstep — the same loading-order position with
three different names. Not stable upstream identifiers.

---

## 2026-06-10 — AA-LCR stable key: SHA-256 of metadata.question.strip(), 100/100 unique

**Method.** SHA-256 of UTF-8 bytes of `question.strip()`, taken from
`sample_score.sample_metadata.question` (reviews) or equivalently
`pred.metadata.question` (predictions).

**Verifications:**

| Check | Result |
|---|---|
| `pred.metadata.question == rev.sample_metadata.question` across 300 (model × idx) pairs | 0 mismatches |
| Question consistent across 3 models per index | 0 mismatches over 100 indices |
| Question uniqueness across 100 items | 100 / 100 unique |
| Whitespace-sensitivity (questions changed by `.strip()`) | 7 / 100 — `.strip()` is essential |
| SHA-256 hash uniqueness after strip | 100 / 100 unique |

The 7 affected questions have leading or trailing whitespace (typically a
trailing newline). Internal whitespace is preserved.

---

## 2026-06-10 — AA-LCR score distribution and judge-noise caveat

| Tier | Count | % |
|---|---|---|
| anchor_easy (all 3 pass) | 36 | 36% |
| split_easy (2/3 pass)    | 27 | 27% |
| anchor_hard (none pass)  | 21 | 21% |
| split_hard (1/3 pass)    | 16 | 16% |

**Discrimination pool size:** 43 / 100 (43%). Higher informative fraction
than LCB. More balanced overall (no single tier dominates).

**Per-model pass rates:** gpt-oss-120b 48%, kimi-k2.5 66%, minimax-m2.5 64%.

**Token range:** input_tokens ∈ [71 691, 114 563]; median ≈ 95 299. AA-LCR is
genuinely long-context. token_bin edges set at quartile-aligned values
(p25 ≈ 84K, p50 ≈ 95K, p75 ≈ 107K) yielding nearly equal-sized bins.

**Judge-noise caveat.** AA-LCR accuracy is LLM-judged at evaluation time. The
shipped review files reflect a single judge run. Items near the decision
boundary may flip on re-evaluation. **Treat ±2–3 percentage-point accuracy
differences across strategies as within-noise** at fine ratios. This caveat
is embedded in every aa_lcr cache file's `metadata.judge_noise_caveat`.

---

## 2026-06-10 — Universal pruning core: 30 / 30 tests pass

**Module.** `evalscope_ext/pruners/core.py` (~350 lines, stdlib only).

**Test coverage** (30 pytest tests in `evalscope_ext/pruners/tests/test_core.py`):

- Tier classification (M=3 and arbitrary M)
- Hybrid: prioritises splits, spills into anchors when discrim pool too
  small, anchors split hard/easy with and without feature_table,
  anchor_fraction=0 reduces to disagreement_only
- Determinism (same seed → same selection; different seed → different)
- Random differs from hybrid on the synthetic
- Stratified-only preserves feature proportions and covers all tiers
- Disagreement-only stays within splits when pool fits, spills to anchors
  at high ratio
- Output sorted; prune_ratio=1.0 returns all
- bucket_counts sum to selected_count
- metadata carries config
- 7 input-validation rejection tests
- 1-PL Rasch orders items correctly; Rasch path runs end-to-end

**Wall time:** 0.08 s.

---

## 2026-06-10 — Universal core required ZERO changes to support AA-LCR

**Evidence.** After completing LCB integration (precompute + adapter), the
AA-LCR integration required:

- New file: `evalscope_ext/pruners/precompute_aa_lcr.py` (~140 lines: a thin
  reader for the AA-LCR review format).
- New file: `evalscope/benchmarks/aa_lcr_pruned/aa_lcr_pruned_adapter.py`
  (~40 lines of actual logic + BenchmarkMeta).
- Zero edits to `core.py`, zero changes to `PruningInputs` / `PruningResult`
  / `prune()` / any strategy.

The universality claim of the core API is therefore empirically supported by
a second benchmark, not just asserted.

---

## 2026-06-10 — LCB integration: precompute output + verification

**Cache files.** 12 files at
`evalscope_ext/pruners/cache/lcb_{hybrid,random,disagreement_only,stratified_only}_r{030,050,070}.json`.

**Selection counts per ratio:** 94 (r=0.30), 158 (r=0.50), 220 (r=0.70).

**Strategy differentiation at r=0.30 (94 selected):**
- hybrid ∩ random = 30 / 94
- hybrid ∩ disagreement_only = 68 / 94
- hybrid ∩ stratified_only = ~24 / 94 (similar to random)

**Verification checks (all pass):**
- All cached hashes recompute identically from `/tmp/lcb_decoded_qcs.json`.
- All cached hashes belong to the known 315-hash set.
- Cache files sorted.
- Adapter `sample_filter` passes exactly 94 / 315 items at r=0.30.
- Adapter file AST-parses cleanly.

**Run command (post-install):**
```bash
evalscope eval --model <model> --datasets live_code_bench_pruned \
  --dataset-args '{"live_code_bench_pruned": {"extra_params": {"index_file": "evalscope_ext/pruners/cache/lcb_hybrid_r050.json"}}}' \
  --output ./results_pruned/
```

---

## 2026-06-10 — AA-LCR integration: precompute output + verification

**Cache files.** 16 files at
`evalscope_ext/pruners/cache/aa_lcr_{hybrid,random,disagreement_only,stratified_only}_r{010,020,030,050}.json`.

**Selection counts per ratio:** 10, 20, 30, 50. AA-LCR is small (N=100) so
low ratios (10–30%) are the meaningful operating points.

**Strategy differentiation at r=0.30 (30 selected):**
- hybrid ∩ random = 12 / 30
- hybrid ∩ disagreement_only = 21 / 30
- hybrid ∩ stratified_only = 11 / 30

**Hybrid composition at r=0.30:** split_hard=10, split_easy=16, anchor_hard=1,
anchor_easy=3 → 26 / 30 from the discrimination pool, 4 anchors.

**Verification checks (all pass):**
- All cached hashes recompute identically from review files.
- All cached hashes belong to the 100-hash ground-truth set.
- Cache files sorted.
- Adapter `sample_filter` passes exactly 30 / 100 items at r=0.30.
- Adapter file AST-parses cleanly.

**Run command (post-install):**
```bash
evalscope eval --model <model> --datasets aa_lcr_pruned \
  --dataset-args '{"aa_lcr_pruned": {"extra_params": {"index_file": "evalscope_ext/pruners/cache/aa_lcr_hybrid_r030.json"}}}' \
  --output ./results_pruned/
```

---

## 2026-06-11 — MMMU shipped data: structure, gaps, and the logprobs windfall

**Files.** `Evals/MMMU/{predictions,reviews}/glm-4.5v-fp8/mmmu_<Subject>.jsonl` — one reference model only (`glm-4.5v-fp8`).

**Initial sync was incomplete** (recorded for transparency): 6 prediction files and 2 review files contained NUL bytes only (Git LFS pull hadn't completed). User re-synced.

**Post-sync state (2026-06-11, current):**

| | files | rows |
|---|---|---|
| predictions | 22 | **660** (22 × 30) |
| reviews     | 22 | **660** (22 × 30) |
| unique `metadata.id` | — | **660** |
| pred ↔ review join on `id` | — | **660 / 660 perfect** |
| parse errors | — | 0 |

**Reference-model pass rates (glm-4.5v-fp8):** 471 / 660 = **71.4%** overall.
Per-subject range: Electronics 43% (hardest) → Art_Theory & Literature 93%
(easiest). Median ≈ 70%. So at M=1 the row-sum tier classification gives
roughly 189 anchor-hard (fails) and 471 anchor-easy (passes) — no
discrimination pool, as expected with a single reference model.

**Stable upstream identifier present.** `metadata.id` carries the upstream MMMU ID in the form `validation_<Subject>_<n>` (e.g. `validation_Clinical_Medicine_18`). All 480 are unique. Unlike LCB/AA-LCR we do NOT need a content hash — MMMU exposes a proper stable key directly. The same `id` field appears in shipped reviews as `sample_score.sample_metadata.id`.

**`sample_id` and `group_id` are again just position labels** — same anti-pattern as AA-LCR. Verified `index == sample_id == group_id` on the first record; do not use them.

**Score path:** `sample_score.score.value['acc']` ∈ {0.0, 1.0}.

**Per-item metadata captured at eval time** (in both pred and review records):
- `id` — stable upstream key
- `question_type` — `multiple-choice` or `open`
- `subfield` — fine-grained subject (e.g. 'Clinical Radiology' under Clinical_Medicine)
- `img_type` — string-encoded list, e.g. `"['Body Scans: MRI, CT scans, and X-rays']"`
- `topic_difficulty` — `Easy` / `Medium` / `Hard`
- `explanation` — often empty

**Critical: logprobs ARE shipped.** `model_output.choices[0].logprobs.content` is populated with 1 entry per output token, each carrying:
- `token` (the chosen token string)
- `logprob` (chosen token's log-probability)
- `top_logprobs` (the top-K alternative tokens + their logprobs — K=3 in shipped data)

This is the encoder-quality signal hinted at in the spec — it gives us per-token confidence, top-1/top-2 margins, and entropy proxies WITHOUT needing to re-query. Critically, `logprobs=True, top_logprobs=K` is also supported by the standard OpenAI Chat Completions API, so any encoder probe we design will work via the standard interface against any candidate model.

**prompt_token_ids also present** (length 2071 on one sample = mostly image-tokens). Image tokens are inside this count, opaquely — we can use it as a coarse proxy for "image complexity as seen by THIS encoder" but it conflates image size with encoder-specific tokenization.

---

## 2026-06-11 — MMMU integration: precompute, adapter, probe runtime

**Files added:**
- `evalscope_ext/pruners/precompute_mmmu.py` — selects from shipped 660 OR streams full 12K via HF
- `evalscope/benchmarks/mmmu_pruned/{__init__.py, mmmu_pruned_adapter.py}` — registered `mmmu_pruned`
- `evalscope_ext/probes/{__init__.py, encoder_probe.py, tests/test_encoder_probe.py}` — triple-query wrapper
- `evalscope_ext/pruners/core.py` — `custom_tiers` extension (single-field addition + precedence line)

**Cache files (16 generated):** `mmmu_{hybrid,random,disagreement_only,stratified_only}_r{010,020,030,050}.json`.

**Selection counts per ratio:** 66 (r=0.10), 132 (0.20), 198 (0.30), 330 (0.50).

**Verification (all pass):**
- All cached ids belong to the 660-id ground-truth set.
- All cache files sorted, all ids match the `validation_<Subject>_<n>` pattern.
- Adapter `sample_filter` simulation passes exactly 198/660 at hybrid r030.
- Adapter + probe + precompute all AST-parse cleanly.

**Strategy differentiation at r=0.30 (target=198):**
- hybrid ∩ random              = 51/198
- hybrid ∩ disagreement_only   = 69/198  (highest overlap — both prioritise tiers 1+2)
- hybrid ∩ stratified_only     = 72/198
- disagreement_only ∩ random   = 64/198

**Hybrid r030 composition** (the headline):
- discrim_pool 495, anchor_pool 165
- discrim_slots 168, anchor_slots 30 (~15% anchor budget)
- selected: 57 split_hard + 111 split_easy + 30 anchor_hard + 0 anchor_easy
- tiers_used: `custom_tiers` (encoder-stress quartile mapping)

**Stress-score distribution on 660:** min=0.183, p50=0.683, max=0.917. The
median item already scores high on encoder-stress (img_type Tables &
Diagrams dominate at ~30% combined), so the discrim pool is large and the
~15% anchor budget naturally lands in low-stress negative controls.

**img_type bucket distribution on 660:** high=483, low=155, mixed=8, unknown=14.
73% of shipped MMMU items are encoder-stressing by metadata alone.

**Test suite total:** 45 / 45 pass (4 new core + 11 probe + 30 prior).

**Empirical-validation work deferred to next phase:**
- Live triple-query runs against an OpenAI-compatible VLM endpoint to
  produce real `encoder_lift` numbers (no endpoint available in this
  session).
- Sensitivity check on the stress-score weights (±20% flex).
- Cross-validation on a known-degraded encoder pair (e.g. fp8 vs fp16).

---

## 2026-06-11 — Empirical validation: pruning preserves the *distinguishable* ranking; near-ties are below the noise floor

**Harness.** `evalscope_ext/validation/run_validation.py` — 1320 trials: 2 benchmarks × 4 strategies × {5–6} ratios × 3 leave-one-model-out × 10 seeds. Pure stdlib metrics (Kendall τ_b, Spearman ρ, two-proportion z-test). All 18 metric tests pass.

### The single most important finding: noise floor

Both benchmarks have a model pair that is **not statistically distinguishable on the full benchmark**, by two-proportion z-test:

| Benchmark | Pair | Δ acc | p | distinguishable? |
|---|---|---:|---:|---|
| LCB | gpt-oss-120b vs kimi-k2.5 | +13.7 pp | <0.001 | yes |
| LCB | gpt-oss-120b vs minimax-m2.5 | +14.6 pp | <0.001 | yes |
| LCB | kimi-k2.5 vs minimax-m2.5 | +1.0 pp | **0.805** | **no** |
| AA-LCR | gpt-oss-120b vs kimi-k2.5 | −18.0 pp | 0.010 | yes |
| AA-LCR | gpt-oss-120b vs minimax-m2.5 | −16.0 pp | 0.023 | yes |
| AA-LCR | kimi-k2.5 vs minimax-m2.5 | +2.0 pp | **0.767** | **no** |

The kimi-vs-minimax ordering in the "full ranking" is **already noise-dominated** on both benchmarks. A pruned-set ranking that flips that pair has not failed — it's reproducing the actual underlying uncertainty. This reframes the entire validation story: we should report preservation rates **per held-out model**, not in aggregate.

### Headline numbers (held-out preservation rate, 10 trials each)

**LCB — hybrid:**

| held out | r=0.05 | r=0.10 | r=0.20 | r=0.30 | r=0.50 | r=0.70 |
|---|---:|---:|---:|---:|---:|---:|
| gpt-oss-120b (rank 1, +14 pp gap)   | 0.90 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| kimi-k2.5 (rank 2, ±1 pp neighbour) | 0.90 | 0.90 | **1.00** | **1.00** | **1.00** | **1.00** |
| minimax-m2.5 (rank 3, ±1 pp neighbour) | 0.30 | 0.30 | 0.00 | 0.00 | 0.00 | 0.10 |

The minimax-held-out collapse is the overfit-to-pair effect: hybrid selects items where the two training models (gpt-oss + kimi OR gpt-oss + minimax_other_perm) disagree, and on those items minimax happens to track the stronger of the two more closely than its full-benchmark average. With Δ=1 pp between kimi and minimax, that bias is enough to flip them. **Random does substantially better on this case at high ratios** (r=0.70: hybrid 0.10 vs random 0.90) because it doesn't amplify the disagreement bias.

**AA-LCR — hybrid:**

| held out | r=0.05 | r=0.10 | r=0.20 | r=0.30 | r=0.50 |
|---|---:|---:|---:|---:|---:|
| gpt-oss-120b (rank 3, −18 pp gap) | 0.50 | 0.60 | 0.70 | 0.90 | **1.00** |
| kimi-k2.5 (rank 1, +2 pp neighbour) | 0.60 | 0.90 | 0.70 | 0.40 | 0.70 |
| minimax-m2.5 (rank 2, ±2 pp neighbour) | 0.60 | 0.80 | 0.80 | 0.90 | **1.00** |

Here the within-noise pair (kimi/minimax) is more forgiving and hybrid clears 0.70–0.90 across most ratios. The distinguishable gpt-oss climbs cleanly with ratio. No equivalent of the LCB minimax pathology.

### Hybrid vs random on the distinguishable held-out

| Cell | hybrid | random | winner |
|---|---:|---:|---|
| LCB, held=gpt-oss, r=0.10 | **1.00** | 0.90 | hybrid (+0.10) |
| LCB, held=gpt-oss, r=0.05 | 0.90 | 0.80 | hybrid (+0.10) |
| AA-LCR, held=gpt-oss, r=0.10 | 0.60 | 0.70 | random (+0.10) |
| AA-LCR, held=gpt-oss, r=0.05 | 0.50 | 0.40 | hybrid (+0.10) |
| AA-LCR, held=gpt-oss, r=0.20 | 0.70 | 0.90 | random (+0.20) |
| AA-LCR, held=gpt-oss, r=0.30 | 0.90 | **1.00** | random (+0.10) |
| AA-LCR, held=gpt-oss, r=0.50 | **1.00** | **1.00** | tie |

Hybrid does NOT uniformly beat random on the distinguishable model. On AA-LCR at moderate ratios random is competitive or ahead. **Honest claim**: hybrid's advantage is concentrated at very low ratios (r=0.05) where random can miss the gap, AND on the LCB distinguishable case at r=0.10+ where hybrid hits 100% earlier.

### Aggregate (all 30 trials) — hybrid vs random, headline column

| Benchmark | Ratio | hybrid | random | Δ |
|---|---:|---:|---:|---:|
| LCB    | 0.05 | 0.70 | 0.57 | +0.13 |
| LCB    | 0.10 | 0.73 | 0.70 | +0.03 |
| LCB    | 0.20 | 0.67 | 0.60 | +0.07 |
| LCB    | 0.30 | 0.67 | 0.73 | −0.06 |
| LCB    | 0.50 | 0.67 | 0.77 | −0.10 |
| LCB    | 0.70 | 0.70 | 0.93 | −0.23 |
| AA-LCR | 0.05 | 0.57 | 0.37 | +0.20 |
| AA-LCR | 0.10 | 0.77 | 0.57 | +0.20 |
| AA-LCR | 0.20 | 0.73 | 0.70 | +0.03 |
| AA-LCR | 0.30 | 0.73 | 0.77 | −0.03 |
| AA-LCR | 0.50 | 0.90 | 0.87 | +0.03 |

On LCB the aggregate masks the per-holdout structure: hybrid wins on the 2 distinguishable-held-out cases and loses big on the noise-pair held-out (minimax). The arithmetic mean across the 3 cases makes random look comparable. AA-LCR shows the cleaner hybrid advantage at low ratios.

### Smallest sufficient ratio per benchmark

Defining "sufficient" as **held-out preservation on the distinguishable model (gpt-oss) reaching 1.00 across all 10 seeds**:

| Benchmark | Strategy | Smallest sufficient ratio | Items kept |
|---|---|---:|---:|
| LCB    | hybrid | **r=0.10** | 32 / 315 |
| LCB    | random | r=0.20 | 63 / 315 |
| AA-LCR | hybrid | **r=0.50** | 50 / 100 |
| AA-LCR | random | r=0.30 | 30 / 100 |

LCB hybrid wins the smallest-sufficient race (32 items, a 10× compression). AA-LCR random reaches 100% at a smaller ratio because the gap is so large (−18 pp) that random sampling resolves it earlier; hybrid catches up at r=0.50.

### Kendall τ_b, all-3-model aggregate, for reference

τ_b values are coarse (only 4 distinct values at n=3) — read variance, not point estimates.

| Benchmark | Ratio | hybrid τ_mean ± std | random τ_mean ± std |
|---|---:|---|---|
| LCB    | 0.10 | +0.794 ± 0.308 | +0.715 ± 0.321 |
| LCB    | 0.30 | +0.778 ± 0.320 | +0.733 ± 0.332 |
| AA-LCR | 0.10 | +0.768 ± 0.354 | +0.370 ± 0.675 |
| AA-LCR | 0.50 | +0.903 ± 0.205 | +0.867 ± 0.271 |

At AA-LCR r=0.10 the hybrid mean-τ is **2.1× random's** mean-τ with half the std — the strongest single quantitative signal of hybrid's value.

### Caveats explicitly carried into Handout A

1. **n=3 models** → coarse rank metrics. Per-holdout breakdowns matter more than aggregate τ.
2. **Within-noise pairs** (kimi vs minimax on both benchmarks) cannot be reliably ordered by any pruning strategy because they cannot be reliably ordered by the *full* benchmark either. Reporting "preservation failure" here is false precision.
3. **AA-LCR judge variance** ±2–3 pp per item compounds with the above. Cell-to-cell differences <0.07 in any rate are within noise on AA-LCR.
4. **Hybrid's overfit-to-pair effect** on LCB minimax-held-out (0 preservation at r≥0.20) is real and documented; mitigation = larger reference panel or trust-region around the disagreement signal.

### Deliverables

- `evalscope_ext/validation/results.json` — full per-trial detail (1320 rows).
- `evalscope_ext/validation/summary.md` — handout-ready tables.
- `evalscope_ext/validation/metrics.py` + tests (18 / 18 pass).

---

## 2026-06-10 — Environment: full evalscope install blocked

The local anaconda environment has `pydantic` 1.x; evalscope's internal types
require `JsonValue` (introduced in pydantic 2.5+). This blocks `from
evalscope.benchmarks.live_code_bench_pruned import live_code_bench_pruned_adapter`
at import time but the adapter source AST-parses cleanly and all logic-level
checks pass.

**To run a real end-to-end eval:** create a fresh venv with pydantic 2.5+,
`pip install -e .` from this repo, then invoke `evalscope eval` with one of
the precomputed cache files. This is the only remaining blocker between the
current code and an actual eval run.
