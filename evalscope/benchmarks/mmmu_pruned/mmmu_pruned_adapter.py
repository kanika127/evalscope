# flake8: noqa: E501
"""Pruned MMMU adapter — encoder-stress probe subset.

Selects a pre-computed subset of MMMU items (identified by upstream `id`) at
eval time. Drop-in replacement for `mmmu`; inherits the vision-language
prompt assembly and answer extraction from MMMUAdapter.

Usage:
    evalscope eval \\
        --model <model> \\
        --datasets mmmu_pruned \\
        --dataset-args '{"mmmu_pruned": {"extra_params": {"index_file": "path/to/mmmu_hybrid_r030.json"}}}' \\
        --output ./results_pruned/

The index_file is produced by evalscope_ext/pruners/precompute_mmmu.py and
contains a JSON object with key "selected_ids": [<id>, ...] where each id
has the form `validation_<Subject>_<n>`.

Stable key: `metadata.id` (the MMMU upstream id). Verified unique across the
660 shipped reference samples; we do NOT content-hash for MMMU.

This pruned adapter pairs with `evalscope_ext.probes.encoder_probe` for the
triple-query (full / text-only / perturbed) encoder-degradation protocol —
the adapter selects items; the probe runs them through the OpenAI interface.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Set

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.mmmu.mmmu_adapter import (
    MMMUAdapter,
    OPEN_PROMPT,
    SUBSET_LIST,
)
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


def _load_selected_ids(index_file: str) -> Set[str]:
    """Load the set of selected upstream ids from an index_file.

    Accepts three formats for forward-compatibility:
        1. {"selected_ids":    [...]}   — canonical for MMMU (id-based)
        2. {"selected_hashes": [...]}   — canonical for LCB/AA-LCR (hash-based, ignored for MMMU but accepted)
        3. [<id>, ...]                  — plain list
    """
    with open(index_file) as f:
        data = json.load(f)
    if isinstance(data, list):
        return set(str(x) for x in data)
    if isinstance(data, dict):
        if "selected_ids" in data:
            return set(str(x) for x in data["selected_ids"])
        if "selected_hashes" in data:
            return set(str(x) for x in data["selected_hashes"])
    raise ValueError(
        f"index_file {index_file!r} must be a JSON list or a dict with key "
        f"'selected_ids' or 'selected_hashes'"
    )


@register_benchmark(
    BenchmarkMeta(
        name="mmmu_pruned",
        pretty_name="MMMU (Pruned, encoder-stress probe)",
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

A pre-computed subset of MMMU selected to probe image-encoder quality. The
subset is biased toward items where the encoder contributes most: dense
diagrams, tables, plots, microscopy, body scans, chemical structures, etc.,
stratified across subfield + question_type + difficulty. A small fraction
(~15%) of low-encoder-stress items is kept as a negative control to detect
non-encoder failures (so the probe can distinguish encoder degradation from
generic capability loss).

## Configuration

- `index_file` (required): path to the JSON index produced by
  `evalscope_ext/pruners/precompute_mmmu.py`.

## Key

Each MMMU item is identified by its upstream `id` (form
`validation_<Subject>_<n>`), exposed on the loaded Sample as
`sample.metadata['id']`. Unlike LCB/AA-LCR, no content hashing is needed —
MMMU exposes a proper stable identifier directly.

## Pairing with encoder_probe.py

This adapter selects the items. The actual encoder-degradation signal is
produced by `evalscope_ext.probes.encoder_probe.run_triple_query()` which
queries each selected item three times through the standard OpenAI interface
(full / text-only / perturbed) and reports per-stratum `encoder_lift` —
the gap between text+image accuracy and text-only accuracy. A degraded
encoder shrinks `encoder_lift` on high-stress strata while leaving the
low-stress controls unchanged.
""",
        dataset_id="AI-ModelScope/MMMU",
        subset_list=SUBSET_LIST,
        metric_list=["acc"],
        eval_split="validation",
        prompt_template=OPEN_PROMPT,
        extra_params={
            "index_file": {
                "type": "str",
                "description": (
                    "Path to a JSON file containing the pre-computed list of "
                    "selected MMMU ids. Produced by "
                    "evalscope_ext/pruners/precompute_mmmu.py. "
                    "Accepts {'selected_ids': [...]}, {'selected_hashes': [...]}, "
                    "or a plain list."
                ),
                "value": None,
            },
        },
    )
)
class MMMUPrunedAdapter(MMMUAdapter):
    """MMMU adapter that evaluates only a pre-selected subset.

    Subclasses MMMUAdapter to inherit:
      - VisionLanguageAdapter base behavior
      - record_to_sample (image base64 embedding + prompt template)
      - extract_answer (multiple-choice / open routing)

    Overrides:
      - __init__: load the selected-id set from extra_params['index_file'].
      - sample_filter: keep iff sample.metadata['id'] is in the selected set.

    No override of record_to_sample is needed — the parent already places
    `record['id']` into `sample.metadata['id']`, which is exactly the key we
    filter on.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        index_file: Optional[str] = self.extra_params.get("index_file")
        if not index_file:
            raise ValueError(
                "mmmu_pruned requires 'index_file' in extra_params. "
                "Generate it with: python -m evalscope_ext.pruners.precompute_mmmu"
            )
        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"index_file not found: {index_file!r}. "
                "Run evalscope_ext/pruners/precompute_mmmu.py to generate it."
            )

        self._selected_ids: Set[str] = _load_selected_ids(index_file)
        logger.info(
            f"[mmmu_pruned] Loaded {len(self._selected_ids)} "
            f"selected ids from {index_file!r}"
        )

    def sample_filter(self, sample: Sample) -> bool:
        """Keep sample only if its upstream MMMU id is in the selected set AND
        the parent's filter passes (parent has no filter; super() returns True)."""
        if not super().sample_filter(sample):
            return False
        return sample.metadata.get("id") in self._selected_ids
