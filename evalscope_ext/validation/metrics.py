"""Pure-stdlib rank-correlation metrics.

Used by the validation harness to compare pruned-set model rankings against
full-benchmark model rankings. For the validation use case we always have
N=3 (three reference models), so we picked simple O(N²) implementations
that are correct in the presence of ties — τ_b handles ties correctly,
Spearman uses average ranks.

We deliberately avoid scipy: keeps the validation harness with the same
zero-dependency footprint as the universal core.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def average_ranks(values: Sequence[float]) -> List[float]:
    """Return average-rank vector for `values`. Tied items receive the mean
    of the rank positions they occupy.

    Example: average_ranks([5, 3, 5, 1]) → [2.5, 4.0, 2.5, 1.0] (wait — that's
    descending convention; we return ASCENDING ranks, so lowest value gets
    rank 1.0). For [5,3,5,1] ascending: ranks are 1→3.5 (tied), 2→2, 3→3.5,
    4→1. So result = [3.5, 2.0, 3.5, 1.0]."""
    n = len(values)
    if n == 0:
        return []
    # Sort indices by value (ascending)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # Items at sorted positions i..j tie; assign average of ranks (i+1..j+1)
        avg = sum(range(i + 1, j + 2)) / (j - i + 1)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall τ_b with tie correction.

    τ_b = (n_c − n_d) / sqrt((n_pairs − n_t_x) · (n_pairs − n_t_y))

    where n_c = pairs concordant, n_d = pairs discordant,
    n_t_x = pairs tied in x (and not in y), n_t_y = pairs tied in y (and not
    in x). Pairs tied in BOTH are excluded from c, d, n_t_x, n_t_y.

    Returns 0.0 when the denominator is 0 (e.g. all x values are tied).
    """
    n = len(x)
    if n != len(y):
        raise ValueError(f"length mismatch: {n} vs {len(y)}")
    if n < 2:
        return 0.0
    n_c = n_d = n_tx_only = n_ty_only = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue  # tied in both — excluded
            if dx == 0:
                n_tx_only += 1
            elif dy == 0:
                n_ty_only += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                n_c += 1
            else:
                n_d += 1
    n_pairs = n * (n - 1) // 2
    denom_x = n_pairs - n_tx_only
    denom_y = n_pairs - n_ty_only
    if denom_x <= 0 or denom_y <= 0:
        return 0.0
    return (n_c - n_d) / math.sqrt(denom_x * denom_y)


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman ρ — Pearson correlation on average ranks. Handles ties via
    `average_ranks` above. Returns 0.0 when either rank vector has zero
    variance."""
    n = len(x)
    if n != len(y):
        raise ValueError(f"length mismatch: {n} vs {len(y)}")
    if n < 2:
        return 0.0
    rx = average_ranks(x)
    ry = average_ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    var_x = sum((r - mx) ** 2 for r in rx)
    var_y = sum((r - my) ** 2 for r in ry)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def descending_ranks(values: Sequence[float]) -> List[int]:
    """Integer rank-1-is-best vector. Tied items receive equal ranks; the
    next non-tied position skips. For [70, 65, 65, 50] → [1, 2, 2, 4].
    For [70, 65, 60] (no ties) → [1, 2, 3].

    Used to compute the held-out model's rank position; the rank-delta
    metric is delta = full_rank − pruned_rank (0 = preserved)."""
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(range(n), key=lambda i: -values[i])  # descending
    ranks = [0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[indexed[k]] = i + 1
        i = j + 1
    return ranks
