# Handout B — Why this matters and how to use it

## What shipping this changes for the customer

Shipping these pruners fundamentally changes how fast and how confidently we can answer a customer’s core question: “Is this model good enough for our workload?”
Instead of running 415 questions and waiting overnight (or longer), a sales engineer or deployment lead can now get a reliable signal in a fraction of the time and cost. We go from running the full LiveCodeBench + AA-LCR suite down to roughly 60–70 questions while still preserving the ability to correctly rank models whose performance is meaningfully different. This turns a multi-day evaluation cycle into something that can often be done within a single meeting or the same day.


## How to run it

The pruners are built directly into evalscope as regular benchmarks. Running the pruned benchmarks is straightforward. After a one-time setup, a sales engineer or deployment lead can get a go/no-go answer with just two commands.

One-time setup:
```bash
  git clone https://github.com/kanika127/evalscope.git 
  cd evalscope
  pip install -e .
```
Run a pruned evaluation on a candidate model:
```bash
evalscope eval --model <candidate> \
  --datasets live_code_bench_pruned aa_lcr_pruned \
  --work-dir ./results_pruned/
```

Compare pruned results against a full baseline:
```bash
python -m evalscope_ext.tools.compare_runs \
  --full ./results_full/ --pruned ./results_pruned/
```

`compare_runs` shows the accuracy delta, percentage of items kept, and whether the model ranking is preserved. You compare the numbers against your team’s threshold — above it, the answer is yes; below it, no. No manual interpretation is needed.

To override a ratio (e.g. AA-LCR fast triage at 10 items), pass `--dataset-args '{"aa_lcr_pruned": {"prune_ratio": 0.10}}'` — any shipped ratio in r=0.05–0.70 resolves to a precomputed cache.


## What the multimodal probe gives that random sampling cannot

If the customer’s roadmap expands to vision next quarter, the MMMU probe gives us something random sampling cannot: it specifically tests whether the image encoder is working, not just whether the model is generally capable.

Raw MMMU accuracy can be misleading. Many questions can be answered reasonably well from the text alone, so a model with a weak image encoder can still score decently by mostly ignoring the image. Random sampling mixes these with truly visual questions, so a model with a broken encoder can still score reasonably well by ignoring the images.

Our probe avoids this problem in two ways. First, our probe deliberately selects visually demanding questions (dense diagrams, charts, medical images, etc.). Second, it compares performance with the image versus with only a text description of the image, and with a perturbed (low-resolution) version of the image. By comparing performance across these conditions, we can tell whether the encoder is contributing meaningfully, whether it’s only capturing gist, or whether it’s barely being used at all. This gives us a much clearer signal about encoder quality.


## Why a PM should care

Speed wins deals. When a prospect asks a technical question about model capability, being able to give a data-backed answer quickly keeps the conversation moving and maintains momentum.

It is also more defensible. When a sophisticated buyer asks why we only ran 32 coding questions, we can explain that these are the questions that actually separate strong models from weak ones — validated by testing whether the ranking still holds when we leave one reference model out during selection.

The multimodal probe prepares us for the customer’s likely next request. When they ask about vision capabilities, we already have a targeted, low-cost way to test image encoder quality instead of running the entire 12,000-question MMMU benchmark from scratch.