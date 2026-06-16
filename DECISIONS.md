# Design Decisions

Running log of key design choices for the Task 2 benchmark-pruning system.
Each entry: **decision** → **rationale** → **alternatives considered** (when
relevant). Append new entries as we go; do not retroactively edit.

**Strategy key** — **hybrid** (proposed): prioritizes high-disagreement (discriminating) items plus ~15% stratified anchors from agreement items for generalization. Baselines: **disagreement_only** (split-item pool, no anchors), **stratified_only** (stratified sample of all items, no disagreement preference), **random** (uniform random).

---

## 2026-06-10 — Stable per-item key: SHA-256 of normalized question text

**Decision.** For every benchmark we prune, the per-item stable key is
`SHA-256(question_text.strip().encode("utf-8")).hexdigest()`. The pruner emits
selected hashes; the pruned adapter computes the same hash over each loaded
sample at eval time and matches.

**Why.**

- **Reorder-proof.** Numeric position (`index`, `sample_id`, `group_id`) breaks
  if the dataset is re-uploaded with rows in a different order, or if a date
  filter changes which rows are in scope.
- **Version-proof.** A hash over problem content survives upstream dataset
  bumps (v5 → v6) as long as the question text is preserved verbatim, which
  it is for the benchmarks LCB and AA-LCR.
- **Audit-friendly.** We can recompute the key over any loaded sample and
  prove it matches the selected set; no opaque lookup tables.
- **Universal.** Both benchmarks share the same key-construction rule with
  zero special-casing — see FINDINGS.md for the field-name table.

**Alternatives considered.**

- *Numeric position* — rejected. The shipped review files happen to use
  `index ∈ [0, N)`, but that's just where the row landed in this run.
- *Upstream `question_id` / `group_id` / `sample_id`* — for LCB the parquet
  does carry `question_id` (e.g. `abc387_b`), but it requires downloading
  parts of a 4 GB+ parquet to verify; for AA-LCR `group_id == sample_id ==
  index` (just a relabeled position). The content-hash approach works for
  both with one rule.
- *SHA-256 vs MD5 vs xxhash* — SHA-256 chosen for collision strength and
  stdlib availability. We don't need cryptographic security, but the cost is
  microseconds and the wins are absence of any collision argument.

---

## 2026-06-10 — `.strip()` normalization, nothing else

**Decision.** Before hashing, apply only `question_text.strip()` (leading/
trailing whitespace removal). No case-folding, no Unicode normalization, no
internal-whitespace collapsing.

**Why.** Minimal normalization is the principle: every transformation we add
is a future surprise. We verified `.strip()` is necessary (7/100 AA-LCR
questions and a small fraction of LCB texts had trailing newlines) and
nothing else is. Internal whitespace inside a question is part of the
question; collapsing it would risk false collisions.

---

## 2026-06-10 — LCB question text recovered via tiktoken `o200k_harmony` decode

**Decision.** For LCB, the input prompt is reconstructed at verification time
from `model_output.prompt_token_ids` (a list of 623 ints) using tiktoken's
`o200k_harmony` encoding. The decoded prompt is sliced between the literal
markers `### Question:\n` and `\n\n### Format:` to isolate `question_content`.

**Why.** The shipped LCB prediction files (`messages` field) only contain the
assistant's response; the original input prompt is NOT preserved as text.
`prompt_token_ids` is the only byte-exact representation of the prompt that
was actually sent to gpt-oss-120b. The harmony tokenizer is the correct
decoder (validated by checking the decoded text contains the LCB prompt-
template markers).

**Alternatives considered.**

- *Use the model's reasoning paraphrase* — model rewrites the problem in its
  own words; fuzzy and unreliable.
- *Use the parquet directly* — would require downloading a ~550 MB v5 shard
  (and v4, and v3); session disk was at 98% and modelscope CDN was throttling
  to ~1 MB/min for our region. Decode path eliminates all of this.
- *Position-based mapping (shipped index = parquet row index)* — fragile;
  there's no guarantee parquet row order is stable.

---

## 2026-06-10 — AA-LCR question text taken directly from `sample.metadata['question']`

**Decision.** For AA-LCR, the pruned adapter computes the hash over
`sample.metadata["question"]` (already populated by the parent
`AALCRAdapter.record_to_sample`).

**Why.** AA-LCR's parent adapter exposes the raw question text in
`sample.metadata`. Re-reading the same value via `record_to_sample`'s record
parameter would be equivalent but redundant; using the already-populated
metadata field makes the hash chain `record['question']` →
`sample.metadata['question']` → `content_hash` impossible to break by
intermediate transformations.

---

## 2026-06-10 — Reject `group_id` / `sample_id` for AA-LCR

**Decision.** Do not use AA-LCR's `sample_score.group_id` or
`sample_score.sample_id` fields as the stable key.

**Why.** Verified across all 100 review records for all 3 models:
`index == sample_id == group_id` — they're all the same integer 0..99. These
are loading-order positions wearing different names, not upstream identifiers.
The name "group_id" was misleading; under any reorder of the upstream dataset
they would change too.

---

## 2026-06-10 — Universal pruning core: pure-stdlib, benchmark-agnostic

**Decision.** Core algorithms live in `evalscope_ext/pruners/core.py` and have
no imports outside stdlib and no knowledge of benchmark names. Inputs are
three generic arrays:

```python
PruningInputs(item_ids, response_matrix, feature_table=None)
```

The benchmark-specific glue (precompute scripts, pruned adapters) is thin.

**Why.**

- **Testability.** Pure-stdlib + small inputs = pytest in 80 ms with no
  environment setup. We have 30 tests.
- **Auditability.** Anyone can run the core algorithm on a 12-item synthetic
  matrix and trace exactly what hybrid does. No `if benchmark == "lcb"`
  branches anywhere.
- **Universality proof.** Adding AA-LCR required ZERO core changes — only a
  new reader and a new thin adapter. If a future benchmark needs anything
  else, that's a signal to upgrade the core API rather than special-case.

---

## 2026-06-10 — 4-tier difficulty classification from row-sum

**Decision.** Each item is mapped to a tier in `{0, 1, 2, 3}` from the row sum
`s = Σ response_matrix[i]`:

| tier | predicate | semantics |
|---|---|---|
| 0 | `s == 0` | anchor-hard (all reference models fail) |
| 1 | `0 < s < M/2` | split-hard (more failures than passes) |
| 2 | `M/2 ≤ s < M` | split-easy (more passes than failures) |
| 3 | `s == M` | anchor-easy (all reference models pass) |

**Why.** At M=3 (our case for both LCB and AA-LCR) this yields the natural
buckets {all-fail, 1-of-3, 2-of-3, all-pass}. The tier is always present
without external metadata, making it the guaranteed-available stratification
axis.

---

## 2026-06-10 — Hybrid strategy = discrimination-priority + stratified anchors

**Decision.** The headline strategy:

1. **Discrimination pool** = tier 1 ∪ tier 2 (items where models disagree).
2. **Anchor pool** = tier 0 ∪ tier 3 (items where all models agree).
3. Target slots = `round(N * prune_ratio)`. Allocate `target * anchor_fraction`
   to anchors (default 0.15), remainder to discrimination.
4. Stratify both pools by `(tier × feature_columns...)`; Hamilton-allocate
   slots across strata proportional to stratum size; sample within strata
   uniformly at random.

**Why.**

- **Discrimination = information.** At binary scoring with M=3 models,
  disagreement IS the discrimination estimator — it's the only axis along
  which items vary on what we measure.
- **Anchors prevent overfitting.** Selecting only disagreement items risks
  optimizing the pruned set for these 3 reference models specifically. An
  item that's all-pass for the panel may split a 4th candidate model. The
  15% anchor budget hedges this.
- **Stratified anchors balance hard and easy.** Selecting anchors uniformly
  from the union would skew to whichever side is larger (LCB has 158
  anchor-easy vs 46 anchor-hard); stratifying ensures the anchor slice
  represents both extremes.

---

## 2026-06-10 — 1-PL Rasch opt-in, 2-PL explicitly excluded

**Decision.** Optional `use_rasch=True` flag fits a 1-PL Rasch model
(joint-MLE, ridge=1e-3, max 200 iterations) and replaces the score-derived
tier with quartile bins of estimated item difficulty. **No 2-PL.**

**Why.** 1-PL has 1 parameter per item (difficulty) + 1 per model (ability) =
N + M parameters from N×M observations. At M=3 it's already marginal.
2-PL doubles the per-item parameter count (difficulty + discrimination) —
guaranteed overfit at M=3, no useful signal. Off by default for both
benchmarks; the score-tier is the documented headline.

---

## 2026-06-10 — Hamilton (largest-remainder) slot allocation across strata

**Decision.** Given a target count and a list of stratum sizes, distribute
slots via the Hamilton method: floor of proportional share, then award
remainders to strata with the largest fractional parts (ties broken by
ascending index for determinism).

**Why.**

- **Bias-free.** Hamilton has no systematic over- or under-allocation by
  stratum size, unlike d'Hondt or Sainte-Laguë.
- **Deterministic.** Given the same input, identical output every run — no
  RNG used for allocation (RNG only enters within-stratum sampling).
- **Cap-aware.** Implementation respects per-stratum size caps (never
  allocates more than the stratum holds), with a second pass to redistribute
  any unfilled slack.

---

## 2026-06-10 — Stratify always includes score-tier as the first axis

**Decision.** Stratification axes are `(tier, feature_columns...)`, with tier
always first and always present. Feature columns degrade gracefully: zero
columns → stratify by tier only; one or more present → cross-product with
tier.

**Why.** External feature metadata can be sparse, missing, or unknown for a
new benchmark. The score-tier is computed from the response matrix itself, so
it's always available. This makes the API robust: callers don't need to
guarantee feature_table is well-populated.

---

## 2026-06-10 — `prune_ratio` = fraction KEPT

**Decision.** Throughout the codebase, `prune_ratio` is the fraction of items
to KEEP after pruning. `prune_ratio=0.30` means "keep 30% of items".

**Why.** Industry usage is split between "keep ratio" and "drop ratio". We
picked "keep" because it matches what shows up in CLI usage and cache
filenames (`_r030.json` for "30% kept"). Documented in the `prune()` docstring
and validated by the `0 < prune_ratio ≤ 1` guard.

---

## 2026-06-10 — Determinism: single seeded `random.Random` + sorted output

**Decision.** All randomness in the pruning core flows through a single
`random.Random(rng_seed)` instance. Strata are iterated in sorted order.
Selected item_ids are sorted before return.

**Why.** Reproducibility is the foundation for empirical strategy comparison.
Given identical inputs and identical seed, byte-identical output. The sorted
output also makes cache files diffable.

---

## 2026-06-10 — Precompute-and-cache architecture

**Decision.** Selection happens offline (the precompute scripts), producing
JSON cache files of selected content_hashes. Evaluation happens at eval time
through the pruned adapter, which loads the cache and filters samples by
hash.

**Why.**

- **Selection ≠ evaluation.** Selecting the subset depends on reference-model
  scores; evaluating the new candidate model depends only on the selected
  hashes. Keeping them separate means we can swap candidate models without
  re-selecting, and we can swap selection strategies without re-running
  inference.
- **Inspectable artifact.** A cache file is a checked-in record of "which
  problems were chosen, by what strategy, at what ratio". It can be reviewed,
  diffed, and signed off independently of any eval run.
- **Auditable.** The bucket counts and metadata in each cache file document
  exactly what the strategy did.

---

## 2026-06-10 — Pruned adapter pattern: subclass + 2 overrides

**Decision.** Each pruned adapter is a subclass of its parent (LCB →
`LiveCodeBenchAdapter`, AA-LCR → `AALCRAdapter`) registered under a new name
(`live_code_bench_pruned`, `aa_lcr_pruned`). It overrides exactly two methods:

1. `record_to_sample`: `sample = super().record_to_sample(record); sample.metadata["content_hash"] = compute_hash(question_text); return sample`
2. `sample_filter`: `return super().sample_filter(sample) and sample.metadata["content_hash"] in self._selected_hashes`

**Why.**

- **Minimal surface area.** Two methods × ~3 lines each = trivial to review,
  trivial to verify against parent behavior.
- **Inherits everything else.** All scoring logic, judging, prompt assembly,
  date filtering (LCB), and document loading (AA-LCR) come from the parent
  unchanged.
- **Forward-compatible.** Always calls `super().sample_filter()` so any
  filter the parent adds in the future is automatically respected.

---

## 2026-06-11 — Core extension: `PruningInputs.custom_tiers`

**Decision.** Add `custom_tiers: Optional[Sequence[int]] = None` to
`PruningInputs`. When provided, OVERRIDES the row-sum-derived 4-tier
classification used by all strategies. Tier semantics unchanged
(0=anchor-hard, 1=split-hard, 2=split-easy, 3=anchor-easy); callers
responsible for mapping their own informativeness signal onto these bins.

Tier-source precedence in `prune()`:
1. `custom_tiers` (caller-provided)
2. `use_rasch=True` (1-PL Rasch quartile bins)
3. Row-sum-derived (default)

Validation: length must equal N; each value must be `int ∈ {0,1,2,3}`.

**Why.** MMMU has 1 reference model. At M=1 the row-sum classifier collapses
to 2 tiers ({pass, fail}) and the disagreement-based hybrid degenerates.
Three alternatives existed:

- Special-case MMMU outside the core. Rejected: breaks the universality
  claim that adding a benchmark needs only a thin reader + adapter.
- Add MMMU-specific code paths to the core. Rejected: violates the
  benchmark-agnostic principle.
- Generalize the core to accept any per-item informativeness signal.
  Selected. The change is small (4 fields of validation, 1 line of precedence
  in `prune()`, no strategy-code touched), and it explicitly accommodates
  future informativeness signals (encoder-stress, IRT residuals,
  confidence-based active-learning scores) without further changes.

`tiers_used` in result metadata now reflects the chosen source ("custom_tiers"
| "rasch_quartile" | "score_sum"), so cache files document provenance.

Verified: 4 new tests pass + all 30 existing tests still pass (34/34 total).

---

## 2026-06-11 — MMMU stable key: upstream `id`, no content hashing

**Decision.** For MMMU, the stable per-item key is `metadata.id` (e.g.
`validation_Clinical_Medicine_18`). The pruner emits selected ids; the
pruned adapter filters by `sample.metadata['id'] in self._selected_ids`.

**Why.** Unlike LCB and AA-LCR, MMMU's upstream dataset exposes a stable
string identifier directly — verified unique across all 660 shipped reference
samples (`pred ↔ review` join on `id` is perfect: 660/660, 0 parse errors).
Content hashing would add no robustness and lose the natural human-readable
provenance (subject + index visible in the id).

Cache files use key `selected_ids` (canonical) but the adapter loader also
accepts `selected_hashes` (cross-compat with LCB/AA-LCR format) and a plain
list (raw hand-off).

---

## 2026-06-11 — MMMU encoder-stress score: locked weighted sum

**Decision.** Per-item encoder-stress score (range [0, 1]):

```
score = 0.45 · stress_img_type            # binary: 1 if img_type ∈ HIGH-density set
      + 0.25 · grounding_intensity         # normalized count of image-type entries
      + 0.20 · topic_difficulty_weight     # Easy 0.50 / Medium 0.75 / Hard 1.00
      + 0.10 · reference_failure_signal    # 1 if ref-model failed AND CoT > p75 tokens
```

When applied to the full 12K via HF streaming (no reference model present),
rebalance to `0.50 · img + 0.30 · grounding + 0.20 · difficulty`.

HIGH-density img_type set (axis A): Tables, Diagrams, Plots and Charts,
Trees and Graphs, Chemical Structures, Technical Blueprints, Microscopic
Images, Pathological Images, Body Scans, Medical Images, Music Sheets, Maps,
Geometric Shapes, Mathematical Notations. LOW-density set: Photographs,
Paintings, Portraits, Sculpture, Comics and Cartoons.

Stress score quartile → custom_tier mapping:
```
QUARTILE_TO_TIER = {0: 0, 1: 1, 2: 2, 3: 2}
# bottom 25% → tier 0 (anchor / negative control)
# next 25%   → tier 1 (informative-low)
# top 50%    → tier 2 (informative-high; q2+q3 lumped → tier 3 stays empty)
```

This makes the hybrid strategy at M=1 treat the top 75% by stress as the
discrim pool and the bottom 25% as the anchor / negative-control pool —
the right semantics for an encoder probe.

**Why these weights.** Axis A (img_type) is the strongest a-priori signal
and dominates at 0.45. Axis B (grounding) is the second-strongest at 0.25.
Difficulty at 0.20 supplements without dominating. Reference-failure-signal
at only 0.10 is deliberately low — the spec forbids overfitting to the
shipped reference panel, and a per-model failure signal can absorb model-
specific quirks if weighted higher.

**Why locked.** Open weighting becomes a tuning problem with no held-out
target. We commit to these in Handout A as assumptions and recommend a
sensitivity check (do top-50% selections change if weights flex ±20%?
expected: stable because img_type dominates).

---

## 2026-06-11 — Encoder probe runtime: triple-query through OpenAI Chat Completions

**Decision.** The probe runtime is a separate module
`evalscope_ext/probes/encoder_probe.py` that, per selected item, runs:

- **Q1 (full)**: text + image(s)
- **Q2 (text-only)**: same text with `<image N>` replaced by `[IMAGE WITHHELD]`
- **Q3 (perturbed)**: text + image downsampled 56×56 then re-upsampled
  — destroys fine spatial detail while preserving coarse layout.

All three are default. All queries use `logprobs=True, top_logprobs=5`.
Extract: for multiple-choice, logprob and top-1/top-2 margin of the
answer-letter token after the `ANSWER:` marker; for open-ended, mean
logprob over the response.

**Headline metric: the JOINT signal of two per-stratum lifts.**

- `lift_text = acc_full − acc_text_only`  — does the encoder contribute ANYTHING text alone cannot?
- `lift_pert = acc_full − acc_perturbed`  — does the encoder read FINE SPATIAL DETAIL, or only coarse gist?

The joint rule, applied per stratum with calibration parameters `τ_lift`,
`τ_pert` (CLI flags; defaults 0.10 / 0.05; tune per VLM family):

| State    | Rule                                            | Interpretation                                                                 |
|----------|-------------------------------------------------|--------------------------------------------------------------------------------|
| ABSENT   | `lift_text < τ_lift`                            | encoder contributes ~nothing; items text-solvable or model ignoring the image |
| COARSE   | `lift_text ≥ τ_lift` AND `lift_pert < τ_pert`   | encoder reads gist but is unmoved by destructive perturbation — the quantized / distilled encoder signature |
| HEALTHY  | both lifts ≥ their thresholds                   | encoder is reading fine detail too                                            |

The two lifts answer COMPLEMENTARY questions and together distinguish
three regimes a single lift collapses. `joint_encoder_signal` produces a
per-stratum classification and a state-count summary that the CLI renders
as `joint_encoder_signal.md` — that markdown report is the decision
artifact a reviewer or PM consumes.

**Why a separate module, not adapter-internal.** The adapter selects items;
the probe protocol is orthogonal. Wiring the triple-query into the adapter
would couple selection to runtime behavior. Keeping them separate also
allows the probe to run against ANY OpenAI-compatible endpoint (OpenAI,
Azure, vLLM, Together, Anyscale, Cerebras, LM Studio…) by env vars alone,
without an evalscope eval session.

**Why injectable client.** `run_triple_query(client: ClientFn, ...)` takes a
client function rather than constructing one internally. Unit tests pass in
a `FakeClient` that returns canned responses for distinct encoder behaviours
(detail_reader / gist_only / broken / text_solvable). Production uses
`default_openai_client()` which wraps `openai.OpenAI().chat.completions.create`.

25 tests pass: prompt construction (incl. Q3), answer extraction, logprob
lookup, numeric aggregation, `classify_state` direct-rule tests for all
three states + threshold boundaries, end-to-end joint-signal classification
trials (one per state), threshold tunability, and the UNKNOWN-when-Q3-
missing path.

---

## 2026-06-13 — Q3/perturbed: wire into a joint signal rather than cut it

**Decision.** Q3 was captured-but-unused: `encoder_lift_by_stratum` had a
`acc_lift_vs_perturbed` field but no decision rule, report, or handout
consumed it. The choice was either (a) cut the perturbed path entirely
(saving Pillow + ~50% inference cost), or (b) promote it to a co-equal
second signal alongside Q2 with a joint rule that names three
encoder-quality states. **Chose (b).**

**Why not (a).** Q2 and Q3 don't answer the same question. Q2's
text-withholding asks "does the encoder contribute *anything* text alone
cannot?" Q3's destructive perturbation asks "does the encoder read *fine
spatial detail*, or only coarse gist?" The fp8-vs-fp16 same-VLM
experiment — the canonical encoder-degradation test — produces a
quantized encoder that still passes Q2 (it's reading the image) but fails
Q3 (it lost the high-spatial-frequency content during quantization). A
single-Q2 protocol classifies that VLM as "healthy" and misses the entire
regime that customers care about. So cutting Q3 leaves the most
load-bearing degradation case undetected.

**Implementation.** `joint_encoder_signal(outcomes, τ_lift, τ_pert)` maps
each stratum to one of {ABSENT, COARSE, HEALTHY} via the rule above. A
`render_joint_report` produces a markdown summary; `main()` writes it
alongside the JSON. CLI defaults run all three variants; `--tau-lift` /
`--tau-pert` are calibration parameters with sensible starting defaults
(0.10 / 0.05), explicitly NOT fitted constants — they're tuned per VLM
family. The renamed numeric fields `lift_text` and `lift_pert` replace
the prior `encoder_lift` / `acc_lift_vs_perturbed` names so the canonical
metric is consistent across code and docs.

**Alternative considered: cut Q3.** Rejected. Would save ~50% inference
cost per item and one optional Pillow dep, but the cost saving is
hypothetical (a 2-variant smoke run still runs without Q3 if a caller
opts down) while the loss is real: the COARSE state, the entire fp8/
distillation/quantization detection story, becomes invisible.

**Honesty.** No live endpoint, so the joint rule is implemented + unit-
tested but not measured. Tests cover one trial per state via deterministic
FakeClient behaviours (gist_only → COARSE, detail_reader → HEALTHY,
broken → ABSENT, text_solvable → ABSENT) and a threshold-tunability test.
Real `lift_text`/`lift_pert` numbers and τ-calibration belong with the
fp8-vs-fp16 same-VLM experiment, listed in Handout A's "what changes with
a live endpoint" section.

---

## 2026-06-10 — `index_file` as `extra_params` entry, accepting two JSON formats

**Decision.** The pruned adapter takes its selection input via
`extra_params["index_file"]` (declared in the `BenchmarkMeta`). The file is
JSON in one of two shapes:

```json
{"selected_hashes": ["…64-hex…", "…"], "metadata": {…}}   ← canonical
["…64-hex…", "…"]                                          ← plain list
```

**Why.** The canonical shape carries diagnostics (strategy, ratio, bucket
counts, seed, model names) that aid auditing. The plain-list shape gives a
clean hand-off to anyone who wants to inject a custom hash set without
constructing the wrapper. Both shapes parse to the same `set` internally.
