"""Tests for the validation metrics. Small known cases."""
import math

import pytest

from evalscope_ext.validation.metrics import (
    average_ranks,
    descending_ranks,
    kendall_tau_b,
    spearman_rho,
)


# ---------------------------------------------------------------------------
# average_ranks
# ---------------------------------------------------------------------------


def test_average_ranks_no_ties():
    # Ascending ranks: smallest → 1
    assert average_ranks([5, 3, 1]) == [3.0, 2.0, 1.0]


def test_average_ranks_with_ties():
    # [5, 3, 5, 1]: sorted asc = [1, 3, 5, 5] at positions [1,2,3,4]
    # The two 5s share positions 3,4 → avg rank 3.5
    assert average_ranks([5, 3, 5, 1]) == [3.5, 2.0, 3.5, 1.0]


def test_average_ranks_all_tied():
    assert average_ranks([7, 7, 7]) == [2.0, 2.0, 2.0]


# ---------------------------------------------------------------------------
# descending_ranks (rank-1 = highest, what model-rankings use)
# ---------------------------------------------------------------------------


def test_descending_ranks_no_ties():
    assert descending_ranks([70.0, 65.0, 50.0]) == [1, 2, 3]


def test_descending_ranks_with_ties():
    # 70 → rank 1; both 65 → rank 2; 50 → rank 4 (skipping rank 3 since 2 items tied at 2)
    assert descending_ranks([70.0, 65.0, 65.0, 50.0]) == [1, 2, 2, 4]


def test_descending_ranks_all_tied():
    assert descending_ranks([0.5, 0.5, 0.5]) == [1, 1, 1]


# ---------------------------------------------------------------------------
# kendall_tau_b
# ---------------------------------------------------------------------------


def test_kendall_identical():
    assert kendall_tau_b([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert kendall_tau_b([0.1, 0.5, 0.9], [10, 20, 30]) == pytest.approx(1.0)


def test_kendall_full_reverse():
    assert kendall_tau_b([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_kendall_one_adjacent_swap_n3():
    # x = [1,2,3], y = [2,1,3] — pairs:
    #   (i=0,j=1): dx=-1, dy=1 → discordant
    #   (i=0,j=2): dx=-2, dy=-1 → concordant
    #   (i=1,j=2): dx=-1, dy=-2 → concordant
    # n_c=2, n_d=1 → τ = (2-1)/3 = 1/3
    assert kendall_tau_b([1, 2, 3], [2, 1, 3]) == pytest.approx(1 / 3)


def test_kendall_all_x_tied_returns_zero():
    assert kendall_tau_b([1, 1, 1], [3, 2, 1]) == 0.0


def test_kendall_known_n5_example():
    # x = [1,2,3,4,5], y = [3,4,1,2,5]. Hand-counted pairs:
    # (0,1) conc, (0,2) disc, (0,3) disc, (0,4) conc,
    # (1,2) disc, (1,3) disc, (1,4) conc,
    # (2,3) conc, (2,4) conc, (3,4) conc → n_c=6, n_d=4, no ties → τ = 2/10 = 0.2
    x = [1, 2, 3, 4, 5]
    y = [3, 4, 1, 2, 5]
    assert kendall_tau_b(x, y) == pytest.approx(0.2)


def test_kendall_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        kendall_tau_b([1, 2], [1, 2, 3])


# ---------------------------------------------------------------------------
# spearman_rho
# ---------------------------------------------------------------------------


def test_spearman_identical():
    assert spearman_rho([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_spearman_full_reverse():
    assert spearman_rho([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_one_swap_n3():
    # [1,2,3] vs [2,1,3]: ranks are [1,2,3] vs [2,1,3].
    # d² values: 1, 1, 0 → sum = 2 → ρ = 1 - 6*2/(3*8) = 1 - 12/24 = 0.5
    assert spearman_rho([1, 2, 3], [2, 1, 3]) == pytest.approx(0.5)


def test_spearman_handles_ties():
    # x=[1,2,2,3], y=[1,3,2,2] — Spearman with average ranks
    # rx = [1, 2.5, 2.5, 4]; ry = [1, 4, 2.5, 2.5]
    # Should be computable without error and < 1.0
    val = spearman_rho([1, 2, 2, 3], [1, 3, 2, 2])
    assert -1.0 <= val <= 1.0
    assert not math.isnan(val)


# ---------------------------------------------------------------------------
# Edge cases on the n=3 model-ranking case (our actual use case)
# ---------------------------------------------------------------------------


def test_n3_kendall_takes_only_four_values():
    """For 3 distinct values per side, τ_b ∈ {-1, -1/3, 1/3, 1}."""
    values = [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
    full = (1, 2, 3)
    taus = sorted({kendall_tau_b(full, p) for p in values})
    assert taus == pytest.approx([-1.0, -1 / 3, 1 / 3, 1.0])


def test_n3_held_rank_delta_with_ties():
    """When 2 models tie in pruned accuracy, their pruned rank is the same."""
    full_acc = [0.70, 0.65, 0.50]
    pruned_acc = [0.60, 0.60, 0.40]
    full_ranks = descending_ranks(full_acc)
    pruned_ranks = descending_ranks(pruned_acc)
    assert full_ranks == [1, 2, 3]
    assert pruned_ranks == [1, 1, 3]
    # Held-out model 1 (index 1): full rank 2, pruned rank 1 → delta = +1
    assert full_ranks[1] - pruned_ranks[1] == 1
