"""Benchmark-agnostic pruning core.

Public API:
    PruningInputs   — generic inputs (item_ids, response_matrix, feature_table)
    PruningResult   — selection output + diagnostics
    prune()         — strategy dispatch: 'hybrid' (headline), 'random',
                       'stratified_only', 'disagreement_only'

The core never branches on benchmark name. Benchmark-specific glue (e.g. the
LCB content_hash key) lives in a thin wrapper layer above this.
"""

from evalscope_ext.pruners.core import PruningInputs, PruningResult, prune

__all__ = ["PruningInputs", "PruningResult", "prune"]
