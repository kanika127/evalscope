# flake8: noqa: E501
"""Pruned LiveCodeBench adapter.

Selects a pre-computed subset of LCB problems (identified by content_hash) at
eval time. Drop-in replacement for 'live_code_bench'; all scoring, date
filtering, and code execution logic is inherited from LiveCodeBenchAdapter.

Usage:
    evalscope eval \\
        --model <model> \\
        --datasets live_code_bench_pruned \\
        --dataset-args '{"live_code_bench_pruned": {"extra_params": {"index_file": "path/to/lcb_hybrid_r050.json"}}}' \\
        --output ./results_pruned/

The index_file is produced by evalscope_ext/pruners/precompute_lcb.py and
contains a JSON object with key "selected_hashes": [<sha256_hex>, ...].

Content-hash key: SHA-256(question_content.strip().encode("utf-8")).hexdigest()
This is identical to what the precompute step stores. It is reorder-proof and
parquet-version-proof because it keys on the problem text, not a row number.

prune_ratio semantics: fraction of items KEPT (same convention as core.py).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional, Set

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


def _compute_content_hash(question_content: str) -> str:
    return hashlib.sha256(question_content.strip().encode("utf-8")).hexdigest()


def _load_selected_hashes(index_file: str) -> Set[str]:
    """Load the set of selected content_hashes from an index_file.

    Accepts two formats:
        1. {"selected_hashes": [...]}   — output of precompute_lcb.py
        2. [<hash>, ...]                — plain list of hashes
    """
    with open(index_file) as f:
        data = json.load(f)
    if isinstance(data, list):
        return set(data)
    if isinstance(data, dict) and "selected_hashes" in data:
        return set(data["selected_hashes"])
    raise ValueError(
        f"index_file {index_file!r} must be a JSON list of hashes or a dict "
        f"with key 'selected_hashes'"
    )


@register_benchmark(
    BenchmarkMeta(
        name="live_code_bench_pruned",
        pretty_name="Live-Code-Bench (Pruned)",
        tags=[Tags.CODING],
        description="""
## Overview

A compressed subset of LiveCodeBench, produced by the hybrid pruning algorithm.

The subset is pre-computed from 3 reference-model evaluations and stored as a
list of content_hashes in an index_file. This adapter loads that index and
selects only the matching problems at eval time.

## Configuration

- `index_file` (required): path to the JSON index produced by
  `evalscope_ext/pruners/precompute_lcb.py`.
- All other parameters (start_date, end_date, debug, sandbox) are inherited
  from `live_code_bench` and behave identically.

## Key

Each problem is identified by SHA-256(question_content.strip()), which is
stable across parquet versions and loading order.
""",
        dataset_id="evalscope/livecodebench_code_generation_lite_parquet",
        subset_list=[
            "release_latest",
            "release_v1",
            "release_v2",
            "release_v3",
            "release_v4",
            "release_v5",
            "release_v6",
            "v1",
            "v1_v2",
            "v1_v3",
            "v1_v4",
            "v1_v5",
            "v1_v6",
            "v2",
            "v2_v3",
            "v2_v4",
            "v2_v5",
            "v2_v6",
            "v3",
            "v3_v4",
            "v3_v5",
            "v3_v6",
            "v4",
            "v4_v5",
            "v4_v6",
            "v5",
            "v5_v6",
            "v6",
        ],
        metric_list=["acc"],
        aggregation="mean_and_pass_at_k",
        eval_split="test",
        prompt_template="### Question:\n{question_content}\n\n{format_prompt} ### Answer: (use the provided format with backticks)\n\n",
        review_timeout=6,
        extra_params={
            "index_file": {
                "type": "str | null",
                "description": (
                    "Path to a JSON file containing the pre-computed list of "
                    "selected content_hashes. Produced by "
                    "evalscope_ext/pruners/precompute_lcb.py. "
                    "Accepts {'selected_hashes': [...]} or a plain list. "
                    "When null, resolved from pruning_strategy + prune_ratio "
                    "against the shipped cache directory."
                ),
                "value": None,
            },
            "pruning_strategy": {
                "type": "str",
                "description": (
                    "Selection strategy used for the default cache lookup. "
                    "One of: hybrid, random, stratified_only, disagreement_only. "
                    "Ignored if `index_file` is set explicitly."
                ),
                "value": "hybrid",
            },
            "prune_ratio": {
                "type": "float",
                "description": (
                    "Fraction of items kept (0 < r <= 1) used for the default "
                    "cache lookup. LCB default 0.10 → 32 of 315 items (the "
                    "smallest-sufficient ratio at full preservation of every "
                    "statistically distinguishable model). "
                    "Ignored if `index_file` is set explicitly."
                ),
                "value": 0.10,
            },
            "start_date": {
                "type": "str | null",
                "description": "Filter problems starting from this date (YYYY-MM-DD). Null keeps all.",
                "value": None,
            },
            "end_date": {
                "type": "str | null",
                "description": "Filter problems up to this date (YYYY-MM-DD). Null keeps all.",
                "value": None,
            },
            "debug": {
                "type": "bool",
                "description": "Enable verbose debug logging and bypass certain safety checks.",
                "value": False,
            },
        },
        sandbox_config={
            "image": "python:3.11-slim",
            "tools_config": {
                "shell_executor": {},
                "python_executor": {},
            },
        },
    )
)
class LiveCodeBenchPrunedAdapter(LiveCodeBenchAdapter):
    """LiveCodeBench adapter that evaluates only a pre-selected subset.

    Subclasses LiveCodeBenchAdapter to inherit all scoring, date filtering,
    and code-execution logic. Adds:
        1. record_to_sample: injects content_hash into sample.metadata.
        2. sample_filter: drops samples whose hash is not in the selected set.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        index_file: Optional[str] = self.extra_params.get("index_file")
        if not index_file:
            from evalscope_ext.pruners.cache_paths import default_index_file
            strategy = self.extra_params.get("pruning_strategy") or "hybrid"
            ratio = self.extra_params.get("prune_ratio") or 0.10
            index_file = default_index_file("lcb", strategy, float(ratio))
            logger.info(
                f"[live_code_bench_pruned] no index_file given; resolved "
                f"default from strategy={strategy!r}, prune_ratio={ratio}: "
                f"{index_file}"
            )
        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"index_file not found: {index_file!r}. "
                "Either pass a valid `index_file` in --dataset-args, or run "
                "`python -m evalscope_ext.pruners.precompute_lcb` to generate "
                "the shipped caches."
            )

        self._selected_hashes: Set[str] = _load_selected_hashes(index_file)
        logger.info(
            f"[live_code_bench_pruned] Loaded {len(self._selected_hashes)} "
            f"selected hashes from {index_file!r}"
        )

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Extend parent to attach content_hash to sample.metadata."""
        sample = super().record_to_sample(record)
        sample.metadata["content_hash"] = _compute_content_hash(
            record["question_content"]
        )
        return sample

    def sample_filter(self, sample: Sample) -> bool:
        """Keep sample only if its content_hash is in the selected set AND
        the parent's date filter passes."""
        if not super().sample_filter(sample):
            return False
        return sample.metadata.get("content_hash") in self._selected_hashes
