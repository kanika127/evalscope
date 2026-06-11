# flake8: noqa: E501
"""Pruned AA-LCR adapter.

Selects a pre-computed subset of AA-LCR questions (identified by content_hash)
at eval time. Inherits all LLM-judge logic from AALCRAdapter.

Usage:
    evalscope eval \\
        --model <model> \\
        --datasets aa_lcr_pruned \\
        --dataset-args '{"aa_lcr_pruned": {"extra_params": {"index_file": "path/to/aa_lcr_hybrid_r030.json"}}}' \\
        --output ./results_pruned/

The index_file is produced by evalscope_ext/pruners/precompute_aa_lcr.py and
contains a JSON object with key "selected_hashes": [<sha256_hex>, ...].

Content-hash key: SHA-256(question.strip().encode("utf-8")).hexdigest()
This is identical to what the precompute step stores and what the adapter
computes at record_to_sample time.

Note on judge noise: AA-LCR accuracy is LLM-judged (non-deterministic). The
hash-selection is stable (it keys on question text, not scores), but the
accuracy of any evaluation run may vary by ±2–3% relative to the reference run
due to judge variance. This is inherent to AA-LCR and not specific to the
pruned adapter.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional, Set

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter, PROMPT_TEMPLATE
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


def _compute_content_hash(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()


def _load_selected_hashes(index_file: str) -> Set[str]:
    """Load the set of selected content_hashes from an index_file.

    Accepts two formats:
        1. {"selected_hashes": [...]}   — output of precompute_aa_lcr.py
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
        name="aa_lcr_pruned",
        pretty_name="AA-LCR (Pruned)",
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## Overview

A compressed subset of AA-LCR (Artificial Analysis Long Context Retrieval),
produced by the hybrid pruning algorithm.

The subset is pre-computed from 3 reference-model evaluations and stored as a
list of content_hashes in an index_file. This adapter loads that index and
selects only the matching questions at eval time.

## Configuration

- `index_file` (required): path to the JSON index produced by
  `evalscope_ext/pruners/precompute_aa_lcr.py`.
- `text_dir` (optional): local directory of extracted AA-LCR text files; if
  omitted, documents are auto-downloaded (inherited from `aa_lcr`).

## Key

Each item is identified by SHA-256(question.strip()), which is stable across
dataset reloads and loading order.

## Judge noise

AA-LCR accuracy is LLM-judged and non-deterministic. Expect ±2–3% variance
across evaluation runs on the same subset.
""",
        dataset_id="evalscope/AA-LCR",
        metric_list=["acc"],
        few_shot_num=0,
        train_split=None,
        eval_split="test",
        prompt_template=PROMPT_TEMPLATE,
        extra_params={
            "index_file": {
                "type": "str",
                "description": (
                    "Path to a JSON file containing the pre-computed list of "
                    "selected content_hashes. Produced by "
                    "evalscope_ext/pruners/precompute_aa_lcr.py. "
                    "Accepts {'selected_hashes': [...]} or a plain list."
                ),
                "value": None,
            },
            "text_dir": {
                "type": "str | null",
                "description": (
                    "Local directory containing extracted AA-LCR text files; "
                    "if null will auto-download & extract."
                ),
                "value": None,
            },
        },
    )
)
class AALCRPrunedAdapter(AALCRAdapter):
    """AA-LCR adapter that evaluates only a pre-selected subset.

    Subclasses AALCRAdapter to inherit the LLM judge, long-context prompt
    assembly, and auto-download logic. Adds:
        1. record_to_sample: injects content_hash into sample.metadata.
        2. sample_filter: drops samples whose hash is not in the selected set.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        index_file: Optional[str] = self.extra_params.get("index_file")
        if not index_file:
            raise ValueError(
                "aa_lcr_pruned requires 'index_file' in extra_params. "
                "Generate it with: python -m evalscope_ext.pruners.precompute_aa_lcr"
            )
        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"index_file not found: {index_file!r}. "
                "Run evalscope_ext/pruners/precompute_aa_lcr.py to generate it."
            )

        self._selected_hashes: Set[str] = _load_selected_hashes(index_file)
        logger.info(
            f"[aa_lcr_pruned] Loaded {len(self._selected_hashes)} "
            f"selected hashes from {index_file!r}"
        )

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Extend parent to attach content_hash to sample.metadata.

        The parent stores record['question'] in sample.metadata['question'],
        so we hash that rather than re-reading the record, keeping the chain
        consistent: record → question → sample.metadata['question'] → hash.
        """
        sample = super().record_to_sample(record)
        sample.metadata["content_hash"] = _compute_content_hash(
            sample.metadata["question"]
        )
        return sample

    def sample_filter(self, sample: Sample) -> bool:
        """Keep sample only if its content_hash is in the selected set AND
        the parent's filter passes (parent is always True for AA-LCR, but
        calling super() is correct for forward-compatibility)."""
        if not super().sample_filter(sample):
            return False
        return sample.metadata.get("content_hash") in self._selected_hashes
