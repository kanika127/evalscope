# Handout A — Why this works (Technical Audience)


## Problem

A customer needs a fast, reliable signal on whether a candidate model is good enough for code generation and long-context reasoning. Running the fullsuite — 315 LiveCodeBench questions plus 100 AA-LCR questions — is expensive because token cost multiplies across hundreds of samples and every candidate model. They might want to run every model multiple times for reliability. This creates both high monetary cost and long end-to-end benchmark evaluation time that hurts release velocity.

I framed the problem as **model ranking preservation under compression**: find the smallest subset that still correctly ranks any model whose performance is statistically distinguishable from the others, while remaining defensible for a fourth unseen model.


## Approach

Most benchmark items carry little value for ranking. On LiveCodeBench, 204 out of 315 questions (65%) have zero cross-model discrimination — all three reference models either pass or fail together. On AA-LCR, 57 out of 100 questions (57%) show the same pattern. These zero-variance items tell us nothing about relative model quality.
I built the pruning method around the following principles:

- **Disagreement as the core signal**: I treated questions where the three models disagree (1/3 or 2/3 pass rate) as the primary source of ranking information. These are the items that actually help distinguish stronger models from weaker ones.
- **Generalization safeguards**: I did not remove all zero-variance questions. Instead, I kept a small stratified portion (~15%) of them for generalization. Completely dropping them would risk overfitting the pruned set to these three specific models. An item that all three models found easy or hard might still separate a fourth model.
- **1-PL Rasch for difficulty ranking**: Within the discriminating questions, I ranked items by difficulty using a 1-PL Rasch model (Item Response Theory) fitted via maximum likelihood estimation. I deliberately avoided 2-PL because estimating a per-item discrimination parameter with only three models would overfit to noise and violate the requirement that the method must work for an unseen fourth model.
- **Benchmark-adaptive stratification**: I applied stratification using whatever metadata each benchmark provides. AA-LCR uses input_tokens for context-length strata, while LiveCodeBench falls back to score-derived difficulty (since its metadata is sparse). This ensures the pruned set maintains coverage across different question types and difficulty levels.

Selection is performed once on the reference data and cached, so new models are evaluated efficiently using only the pre-selected indices. I validated this design using leave-one-model-out testing across multiple prune ratios, with Kendall’s τ_b and held-out rank preservation as the primary metrics.

Refer to evalscope/architecture_flowcharts.md.


## How Much I Pruned and Why the Subset Is Sufficient

I pruned LiveCodeBench to 32 questions (10%) and AA-LCR to 30 questions (30%) for reliable go/no-go decisions.


| Benchmark        | Total | Non-Discriminating Items | Discriminating Items | Kept | Reduction |
| ---------------- | ----: | ------------------: | ------------------: | ----------: | --------: |
| LiveCodeBench v5 |   315 |           204 (65%) |           111 (35%) |   32 (10%) |       90% |
| AA-LCR           |   100 |            57 (57%)  |           43 (43%) |   30 (30%) |       70% |






I validated sufficiency by sweeping prune ratios from 5% to 70% and measuring held-out model rank preservation across 30 leave-one-model-out trials (10 seeds × 3 folds). This directly tests defensibility for an unseen fourth model.

On LiveCodeBench, rank preservation for distinguishable models reaches 100% at 10% (32 items). Below this ratio, preservation drops noticeably. 

On AA-LCR, 30% (30 items) is the point where preservation stabilizes for confident decisions.
At very small size (10% preservation), the method still delivers strong signal quality — achieving 2.1× higher rank correlation than random sampling with lower variance.

Two of the three reference models (kimi-k2.5 and minimax-m2.5) are statistically tied on both benchmarks (Δ ≈ 1–2 pp, p-value > 0.76). The full benchmark itself cannot reliably order them, so no pruned subset can either. I report this noise floor explicitly rather than claiming perfect preservation across all cases.

The real protection for a fourth model comes from retaining a small portion of zero-variance items. These act as safeguards so the subset does not overfit to patterns observed only in the three calibration models.

All validation results, including per-holdout breakdowns and full trial data, are available in evalscope_ext/validation/results.json and evalscope_ext/validation/summary.md.


## Part B — the multimodal encoder probe

MMMU has one reference model, so disagreement-based pruning is not applicable. Instead, I designed a targeted encoder stress probe rather than a generic subset of the ~12K MMMU dataset.

I first compute a per-question **encoder stress score**:

`encoder_stress_score = 0.45·img_type + 0.25·grounding + 0.20·difficulty + 0.10·ref_failure`

This score prioritizes questions whose images are likely to stress the vision encoder — such as dense diagrams, charts, tables, medical scans, microscopy images, music-sheets captured by `img_type`, and multi-image reasoning tasks captured by `grounding` with `<image N>` cross-references. `difficulty` reflects the inherent difficulty of the question. `ref-failure`'s weight is deliberately small (0.10) to avoid overfitting to glm-4.5v-fp8's specific failure modes. Questions are binned into stress quartiles using the encoder stress score; the probe primarily selects high-stress items while including a small portion of low-stress items as controls, stratified by image type and subject. Questions that can be solved from text alone receive lower priority.

Raw MMMU accuracy collapses encoder quality with text-only reasoning — a model can score well because its language component guesses answers from the question text alone. I separate the two with a **triple-query protocol** through standard OpenAI chat-completions (`logprobs=True, top_logprobs=5`): 
- Q1: Full image + text prompt
- Q2: Same prompt with `<image N>` replaced by textual description `[IMAGE WITHHELD]`
- Q3: text + image downsampled to 56×56 and re-upsampled (destroys fine spatial detail; preserves coarse layout)

This produces two complementary signals per stratum:
- `lift_text = acc(Q1) − acc(Q2)` — measures whether the encoder contributes anything beyond text.
- `lift_pert = acc(Q1) − acc(Q3)` — measures whether the encoder captures fine spatial detail or only coarse gist.

These two lifts distinguish three regimes that a single accuracy number cannot:
| Regime | Condition | Interpretation |
|---|---|---|
| Absent  | `lift_text` low | Encoder contributes little; model mostly ignores the image or question is text-solvable |
| Coarse  | `lift_text` high but `lift_pert` low | Encoder captures gist but not fine detail (typical of quantized or heavily compressed vision encoders) |
| Healthy | both lifts high | Encoder reads both coarse structure and fine spatial detail |

By combining information from Q2 and Q3, the probe specifically surfaces image-encoder degradation rather than generic multimodal capability. Random sampling from MMMU produces only one aggregate accuracy score and cannot separate these regimes.


## Assumptions

- **Distribution**: I assumed that item discrimination is primarily a property of the question itself rather than the specific set of models used to measure it. In other words, if three models disagree on a question, it is usually because the question tests a real capability axis — not because of idiosyncratic weaknesses unique to those three models (e.g., tokenizer behavior, training data artifacts, or particular failure modes unique to these three models). This allows disagreement observed among the three reference models to generalize to a fourth unseen model, provided capability gaps are reasonably smooth.
- **Scale**: I assumed that with only three models, estimating a per-item discrimination parameter (as in 2-PL) would be unreliable and prone to overfitting. This is why I used a 1-PL Rasch model and treated 2-PL as a future extension once more models become available.
- **Behaviour**: I assumed that the three reference models span a reasonably broad capability range. This allows disagreement observed among them to generalize to a fourth unseen model, provided capability gaps remain reasonably smooth. I also assumed that LCB’s sandbox grader is deterministic while AA-LCR and MMMU scores carry single-judge noise. I accounted for the latter by explicitly reporting the noise floor between kimi-k2.5 and minimax-m2.5 rather than claiming perfect ranking preservation.


## What changes with more inputs


**(a) More data / more models**: 
With more models (M ≈ 5–10), discrimination estimates would become reliable enough to use 2-PL or 3-PL IRT. Larger panels would also reduce the impact of per-model idiosyncrasies and push statistically tied pairs above the noise floor, strengthening validation.

**(b) A live model endpoint**: 
A live endpoint would allow direct validation on a real fourth model. For Part B, the cleanest experiment would be running the probe on an fp8 vs fp16 pair of the same VLM, giving ground-truth encoder degradation signals instead of relying on simulated perturbations.

**(c) More time**: 
With more development time I would introduce a noise-aware preservation metric that excludes statistically tied model pairs, run bootstrap confidence intervals on rank preservation, and explore ensemble selection across multiple pruning signals.