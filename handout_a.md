# Handout A — Why this works

*Task 2 benchmark-compression solution, fork of evalscope at commit `bf3bd26`, branch `task2-pruner`.*

## The problem I set out to solve

The sales-engineering question is binary: *given a customer's LCB + AA-LCR thresholds, is this candidate model good enough?* Compressing a benchmark for that question is not the same as compressing for accuracy estimation. Absolute scores can shift on a pruned set — they're allowed to. What must survive is the **rank** of the candidate against models the customer already trusts (or rejects). So the design target is: preserve the rank of any model whose accuracy is statistically distinguishable from its neighbours, on the smallest possible subset.

At n=3 reference models with binary per-item scores, the discrimination axis is **disagreement**. Items where all 3 reference models pass (or all fail) carry no information for the ranking question — every model behaves identically on them. Items where they disagree (1-of-3 or 2-of-3 patterns) are the only ones where two candidates can plausibly diverge. The universal core's insight: classify items into 4 score-sum tiers (anchor-hard / split-hard / split-easy / anchor-easy), make the discrimination pool the union of the split tiers, and reserve a small stratified anchor budget (~15%) as negative controls.

## How much I pruned, and the empirical defense

The two benchmarks reward two different claims, and I separate them honestly:

**LCB: 10× compression at full preservation.** At `prune_ratio=0.10` — 32 of 315 items — hybrid preserves the rank of every reference model whose full-benchmark gap is statistically distinguishable, in **100% of 30 trials** (10 seeds × 3 leave-one-model-out splits). Random hits 100% only at r=0.20 (63 items). LCB's compression win is hybrid's.

**AA-LCR: ~2× better rank-correlation than random at aggressive ratios.** At `prune_ratio=0.10` (10 items), hybrid attains **Kendall τ_b = 0.768 ± 0.354** versus random's **0.370 ± 0.675** — 2.08× the mean correlation with roughly half the variance. AA-LCR's smallest-sufficient-ratio race actually goes to random (r=0.30, 30 items, vs hybrid's r=0.50, 50 items), so I don't claim a compression win here — the AA-LCR win is the **quality of ranking signal at very small sizes**, useful when 10 items is the most a customer will spend on a fast pre-check.

The defense was leave-one-model-out (LOMO): the pruner saw only 2 of the 3 reference models, then we evaluated the held-out model's rank on the pruned set. This is the "defensible for a 4th model" test the spec demanded.

**Honest caveats — they shape what we ship.** Two of the three reference models — kimi-k2.5 and minimax-m2.5 — are **statistically tied** on both benchmarks (Δ=1.0 pp LCB, p=0.805; Δ=2.0 pp AA-LCR, p=0.767, two-proportion z-tests). The full benchmark itself cannot reliably order them; no pruner can either. Aggregate preservation rates blend distinguishable-pair cases with this noise-floor case, which is why aggregate numbers look mixed at moderate ratios.

There's a failure mode I should name. When LCB holds out minimax (within-noise of kimi), hybrid's preservation collapses to 0–30% across r≥0.20 — the disagreement signal in the 2-model training pair systematically biases the held-out third model. Random doesn't share this bias and beats hybrid at LCB r=0.70 (0.93 vs 0.70). **Hybrid does not uniformly beat random**; its advantage concentrates at very low ratios (r=0.05–0.10) and on the information-denser AA-LCR. I ship it as "10× faster on LCB when the rank you're testing for is real, with a documented degradation when it isn't, and a 2.1×-cleaner ranking signal on AA-LCR at aggressive sizes".

## Part B — the multimodal encoder probe

MMMU has one reference model, so disagreement-pruning is structurally inapplicable. The universal core gained one honest extension here: `PruningInputs.custom_tiers`, so any per-item informativeness signal can substitute for row-sum tiers. Selection runs on

`encoder_stress_score = 0.45·img_type + 0.25·grounding + 0.20·difficulty + 0.10·ref_failure`

with weights locked and documented. `img_type` captures dense diagrams, tables, plots, microscopy, body scans, music sheets — content that stresses encoder downsampling and patch tokenisation. `grounding` captures multi-image and `<image N>` cross-references. The ref-failure weight is deliberately small (0.10) to avoid overfitting to glm-4.5v-fp8's specific failure modes.

Raw MMMU accuracy collapses encoder quality with text-only reasoning — a model can score well because its language component guesses college answers from the question text alone. I separate the two with a **triple-query protocol** through standard OpenAI chat-completions (`logprobs=True, top_logprobs=5`): Q1 text+image, Q2 the same prompt with `<image N>` replaced by `[IMAGE WITHHELD]`, optional Q3 image downsampled to 56×56 and re-upsampled. Headline metric per stratum:

`encoder_lift = acc(Q1) − acc(Q2)`

A healthy encoder shows large lift on high-stress strata and small lift on the low-stress negative-control bin (the bottom-quartile-by-stress anchors). A degraded encoder shows shrunken lift on high-stress while controls stay flat — distinguishing "encoder broken" from "model generally weak", which raw accuracy cannot.

## Assumptions

- **Distribution.** LCB scores are deterministic (sandbox grader); AA-LCR and MMMU carry ±2–3 pp single-judge variance, encoded as a tolerance band in cache metadata.
- **Scale.** 3-model panel is the spec's constraint, not the method's; the core is M-agnostic.
- **Behaviour.** Selection is reference-panel-dependent — recalibrate the cache once per panel, not per candidate.

## What changes with more inputs

- **(a) Bigger reference panel.** The LCB minimax-overfit pathology is structurally driven by the 2-model training pair after LOMO. At M=5–10 the discrimination signal averages out per-model idiosyncrasies and the panel-overfit attenuates. Larger N also pushes within-noise pairs above the detection floor.
- **(b) Live endpoint.** Part B ships as runnable code with unit-tested prompt construction and metric aggregation but no live `encoder_lift` numbers. An OpenAI-compatible endpoint unblocks real numbers, cross-validation on a known degradation pair (fp8 vs fp16 of the same VLM is the cleanest experiment), and Q3 perturbation against actual encoders.
- **(c) More time.** Sensitivity analysis on the encoder-stress weights (±20%; we expect stability because img_type dominates). A learned tier classifier replacing the hand-weighted score. 2-PL IRT once M grows past ~5. A noise-aware preservation metric that excludes statistically-tied pairs from "failure" counts.
