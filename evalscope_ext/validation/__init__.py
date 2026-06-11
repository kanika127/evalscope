"""Empirical validation harness for the pruning method.

Public modules:
    metrics          — Kendall τ_b, Spearman ρ, rank-with-ties (stdlib only)
    run_validation   — leave-one-model-out + multi-seed sweep, writes results.json + summary.md
"""
