"""Resolve a default index_file path from (pruning_strategy, prune_ratio).

Used by the three pruned adapters so a bare
    `evalscope eval --datasets <bench>_pruned`
(no `--dataset-args`) picks the shipped per-benchmark default cache.

Filename convention (lives in evalscope_ext/pruners/cache/):

    <bench>_<strategy>_r<ratio×100, zero-padded-3>.json

    lcb_hybrid_r010.json      → LCB, hybrid, prune_ratio=0.10
    aa_lcr_random_r030.json   → AA-LCR, random, prune_ratio=0.30
    mmmu_hybrid_r020.json     → MMMU, hybrid, prune_ratio=0.20
"""
from __future__ import annotations

import os
from pathlib import Path

VALID_STRATEGIES = (
    "hybrid",
    "random",
    "stratified_only",
    "disagreement_only",
)


def _cache_dir() -> Path:
    """The packaged cache directory.

    Resolved from this module's location (works with `pip install -e .` and
    a plain checkout). Returns evalscope_ext/pruners/cache/."""
    return Path(__file__).parent / "cache"


def default_index_file(bench_prefix: str, strategy: str, ratio: float) -> str:
    """Map (bench_prefix, strategy, ratio) to a shipped cache path.

    bench_prefix    — 'lcb' | 'aa_lcr' | 'mmmu'
    strategy        — must be in VALID_STRATEGIES
    ratio           — fraction of items kept, 0 < r <= 1

    Returns the absolute path string. Raises ValueError on bad inputs;
    does NOT check filesystem existence — callers should verify with
    os.path.exists and surface a clear error if a default cache is missing.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"unknown pruning_strategy {strategy!r}; valid: {VALID_STRATEGIES}"
        )
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"prune_ratio must be in (0, 1], got {ratio}")
    fname = f"{bench_prefix}_{strategy}_r{int(round(ratio * 100)):03d}.json"
    return str(_cache_dir() / fname)
