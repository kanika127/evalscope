# Handout B — Why this matters and how to use it

## What shipping this changes for the customer conversation

Answering *"is this model good enough for our coding workload?"* today means running the full LCB v5 suite for every candidate — 315 items, hours of inference at production rates, billed per candidate. With the shipped pruners you can answer the same question in **32 LCB items (10× compression)** when the candidate's gap from the customer's current model is statistically real. The honest framing is binary: when the gap between candidate and incumbent is real, the pruned set sees it; when the gap is within ~1–2 pp, the *full* benchmark also can't tell, and the pruner inherits that limit — we say so up front. In practice this is a same-day go/no-go on candidates that are meaningfully better or worse, and a clean "we genuinely can't tell" answer for the close calls. AA-LCR is a quality-of-signal win at small sizes rather than a compression win: at 10 items (r=0.10), the pruned set gives **2.1× the rank-correlation** of a random 10-item sample, which is what you want when 10 items is the most a customer will spend on a fast pre-check.

## How to run it tomorrow

The pruners are standard evalscope datasets: `live_code_bench_pruned`, `aa_lcr_pruned`, `mmmu_pruned`. Drop-in to the normal command:

```bash
evalscope eval --model <candidate> \
  --datasets live_code_bench_pruned \
  --dataset-args '{"live_code_bench_pruned": {"extra_params": {"index_file": "evalscope_ext/pruners/cache/lcb_hybrid_r010.json"}}}' \
  --output ./results_pruned/

python -m evalscope_ext.tools.compare_runs --full ./results_full/ --pruned ./results_pruned/
```

24 LCB + 16 AA-LCR + 16 MMMU pre-computed caches ship at ratios r=0.05–0.70; pick the one that fits the time budget. No new training, no new infrastructure.

## What the multimodal probe gives that random sampling cannot

Raw MMMU accuracy conflates **image-encoder quality** with **text-only reasoning**: a model can score well because its language component guesses college answers from the question text alone, the encoder doing almost nothing real. The probe disentangles this in two ways. It biases selection toward items where the encoder must contribute — dense diagrams, tables, plots, microscopy, body scans — and it queries each item *twice*: once with the image, once with the image replaced by `[IMAGE WITHHELD]`. The accuracy gap is the encoder's contribution, isolated. The headline number `encoder_lift` per stratum tells you whether the encoder is doing its job, separately from whether the model knows the subject. Random sampling gives you one accuracy number with the conflation intact; this gives you the answer.

## Why a PM should care

**Speed:** turns "weeks of back-and-forth on eval cost" into a same-day answer for clear-gap candidates. **Specificity:** lets us tell a customer planning a quarterly multimodal roadmap whether *the encoder specifically* will survive their quantisation or distillation — without committing them to a deep multimodal eval contract before they know it's worth it.
