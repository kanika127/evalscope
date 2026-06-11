"""Unit tests for the universal pruning core.

Synthetic data: 12 items × 3 reference models, covering every tier:
    2 × tier 0 (anchor-hard, all fail)
    3 × tier 1 (split-hard, 1/3 pass)
    3 × tier 2 (split-easy, 2/3 pass)
    4 × tier 3 (anchor-easy, all pass)

Feature columns chosen so stratification has interesting structure:
    feature_a: 8 'x', 4 'y' (2:1 skew, balanced per tier)
    feature_b: 6 of 10, 6 of 20 (50/50)
"""

from __future__ import annotations

import pytest

from evalscope_ext.pruners import PruningInputs, PruningResult, prune
from evalscope_ext.pruners.core import (
    TIER_ANCHOR_EASY,
    TIER_ANCHOR_HARD,
    TIER_SPLIT_EASY,
    TIER_SPLIT_HARD,
    classify_tiers,
    fit_rasch_1pl,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_inputs() -> PruningInputs:
    item_ids = [f"i{n:02d}" for n in range(12)]
    response_matrix = [
        [0, 0, 0],  # i00 — tier 0
        [0, 0, 0],  # i01 — tier 0
        [1, 0, 0],  # i02 — tier 1
        [0, 1, 0],  # i03 — tier 1
        [0, 0, 1],  # i04 — tier 1
        [1, 1, 0],  # i05 — tier 2
        [1, 0, 1],  # i06 — tier 2
        [0, 1, 1],  # i07 — tier 2
        [1, 1, 1],  # i08 — tier 3
        [1, 1, 1],  # i09 — tier 3
        [1, 1, 1],  # i10 — tier 3
        [1, 1, 1],  # i11 — tier 3
    ]
    feature_a = ["x", "y", "x", "x", "y", "x", "x", "y", "x", "x", "x", "y"]
    feature_b = [10, 20, 10, 10, 20, 20, 20, 10, 10, 10, 20, 20]
    return PruningInputs(
        item_ids=item_ids,
        response_matrix=response_matrix,
        feature_table={"feature_a": feature_a, "feature_b": feature_b},
    )


SPLIT_IDS = {"i02", "i03", "i04", "i05", "i06", "i07"}
ANCHOR_HARD_IDS = {"i00", "i01"}
ANCHOR_EASY_IDS = {"i08", "i09", "i10", "i11"}
ANCHOR_IDS = ANCHOR_HARD_IDS | ANCHOR_EASY_IDS


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------


def test_tier_classification(synthetic_inputs):
    tiers = classify_tiers(synthetic_inputs.response_matrix)
    assert tiers == [
        TIER_ANCHOR_HARD,  # 0,0,0
        TIER_ANCHOR_HARD,  # 0,0,0
        TIER_SPLIT_HARD,   # 1,0,0
        TIER_SPLIT_HARD,   # 0,1,0
        TIER_SPLIT_HARD,   # 0,0,1
        TIER_SPLIT_EASY,   # 1,1,0
        TIER_SPLIT_EASY,   # 1,0,1
        TIER_SPLIT_EASY,   # 0,1,1
        TIER_ANCHOR_EASY,  # 1,1,1
        TIER_ANCHOR_EASY,
        TIER_ANCHOR_EASY,
        TIER_ANCHOR_EASY,
    ]


def test_tier_classification_supports_arbitrary_M():
    rm = [
        [0, 0, 0, 0],     # tier 0
        [1, 0, 0, 0],     # 1/4 → tier 1 (s*2=2 < 4)
        [1, 1, 0, 0],     # 2/4 → tier 2 (s*2=4 not <4, so tier 2)
        [1, 1, 1, 0],     # tier 2
        [1, 1, 1, 1],     # tier 3
    ]
    assert classify_tiers(rm) == [0, 1, 2, 2, 3]


# ---------------------------------------------------------------------------
# Hybrid strategy
# ---------------------------------------------------------------------------


def test_hybrid_prioritises_split_items_at_50pct(synthetic_inputs):
    # target = round(12*0.5) = 6
    # anchor_slots = round(6*0.15) = round(0.9) = 1
    # discrim_slots = 5
    result = prune(synthetic_inputs, prune_ratio=0.5, strategy="hybrid", rng_seed=0)
    assert result.target_count == 6
    assert result.selected_count == 6
    selected = set(result.selected_item_ids)
    split_chosen = selected & SPLIT_IDS
    anchor_chosen = selected & ANCHOR_IDS
    assert len(split_chosen) == 5
    assert len(anchor_chosen) == 1


def test_hybrid_spills_into_anchors_when_discrim_pool_too_small(synthetic_inputs):
    # ratio=0.83 → target = round(12*0.83) = round(9.96) = 10
    # anchor_slots = round(10*0.15) = round(1.5) = 2 (banker's)
    # discrim_slots requested = 8, but discrim pool has only 6 → spill 2 → anchor_slots=4
    result = prune(synthetic_inputs, prune_ratio=0.83, strategy="hybrid", rng_seed=0)
    assert result.target_count == 10
    assert result.selected_count == 10
    selected = set(result.selected_item_ids)
    # all 6 splits must be present
    assert SPLIT_IDS <= selected
    anchor_chosen = selected & ANCHOR_IDS
    assert len(anchor_chosen) == 4


def test_hybrid_anchors_split_between_hard_and_easy(synthetic_inputs):
    # At ratio=0.83 anchor_slots=4. The anchor pool stratifies by
    # (tier × feature_a × feature_b), which gives 5 strata over 6 items:
    #   (0, x, 10): [i00]              size 1
    #   (0, y, 20): [i01]              size 1
    #   (3, x, 10): [i08, i09]         size 2
    #   (3, x, 20): [i10]              size 1
    #   (3, y, 20): [i11]              size 1
    # Hamilton allocates 4 slots → [1, 1, 1, 1, 0] → 2 hard + 2 easy. The
    # feature-stratification deliberately pulls anchor balance away from the
    # raw 2:4 tier ratio toward a more diverse hold-out, which is the desired
    # generalization-anchor behavior.
    result = prune(synthetic_inputs, prune_ratio=0.83, strategy="hybrid", rng_seed=0)
    selected = set(result.selected_item_ids)
    n_hard = len(selected & ANCHOR_HARD_IDS)
    n_easy = len(selected & ANCHOR_EASY_IDS)
    assert n_hard >= 1, "hybrid must include at least one anchor-hard item"
    assert n_easy >= 1, "hybrid must include at least one anchor-easy item"
    assert n_hard + n_easy == 4
    # On the synthetic with these features, the exact split is 2 + 2:
    assert n_hard == 2
    assert n_easy == 2


def test_hybrid_anchors_split_when_feature_table_absent():
    """Without feature_table, anchor stratification is on tier only — so the
    pure 2 (hard) : 4 (easy) tier ratio is honored, giving Hamilton split
    1 hard + 3 easy at anchor_slots=4."""
    item_ids = [f"i{n:02d}" for n in range(12)]
    response_matrix = [
        [0, 0, 0], [0, 0, 0],
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
    ]
    inputs = PruningInputs(item_ids=item_ids, response_matrix=response_matrix)
    result = prune(inputs, prune_ratio=0.83, strategy="hybrid", rng_seed=0)
    selected = set(result.selected_item_ids)
    n_hard = len(selected & ANCHOR_HARD_IDS)
    n_easy = len(selected & ANCHOR_EASY_IDS)
    assert n_hard == 1
    assert n_easy == 3


def test_hybrid_is_deterministic(synthetic_inputs):
    r1 = prune(synthetic_inputs, prune_ratio=0.5, strategy="hybrid", rng_seed=42)
    r2 = prune(synthetic_inputs, prune_ratio=0.5, strategy="hybrid", rng_seed=42)
    assert r1.selected_item_ids == r2.selected_item_ids


def test_hybrid_seed_changes_selection(synthetic_inputs):
    # Within the discrim pool the strata each carry >1 candidate, so different
    # seeds should yield different within-stratum samples.
    r1 = prune(synthetic_inputs, prune_ratio=0.5, strategy="hybrid", rng_seed=0)
    r2 = prune(synthetic_inputs, prune_ratio=0.5, strategy="hybrid", rng_seed=999)
    assert r1.selected_item_ids != r2.selected_item_ids


def test_hybrid_no_feature_table_falls_back_to_tier_strat():
    item_ids = [f"i{n:02d}" for n in range(12)]
    response_matrix = [
        [0, 0, 0], [0, 0, 0],
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
    ]
    inputs = PruningInputs(item_ids=item_ids, response_matrix=response_matrix)
    result = prune(inputs, prune_ratio=0.5, strategy="hybrid", rng_seed=0)
    assert result.selected_count == 6
    selected = set(result.selected_item_ids)
    assert len(selected & SPLIT_IDS) == 5
    assert len(selected & ANCHOR_IDS) == 1


def test_hybrid_anchor_fraction_zero_uses_only_discrim(synthetic_inputs):
    result = prune(
        synthetic_inputs,
        prune_ratio=0.5,
        strategy="hybrid",
        rng_seed=0,
        anchor_fraction=0.0,
    )
    selected = set(result.selected_item_ids)
    assert len(selected & SPLIT_IDS) == 6
    assert len(selected & ANCHOR_IDS) == 0


# ---------------------------------------------------------------------------
# Random strategy
# ---------------------------------------------------------------------------


def test_random_is_deterministic_per_seed(synthetic_inputs):
    r1 = prune(synthetic_inputs, prune_ratio=0.5, strategy="random", rng_seed=7)
    r2 = prune(synthetic_inputs, prune_ratio=0.5, strategy="random", rng_seed=7)
    assert r1.selected_item_ids == r2.selected_item_ids


def test_random_differs_across_seeds(synthetic_inputs):
    r1 = prune(synthetic_inputs, prune_ratio=0.5, strategy="random", rng_seed=0)
    r2 = prune(synthetic_inputs, prune_ratio=0.5, strategy="random", rng_seed=99)
    assert r1.selected_item_ids != r2.selected_item_ids


def test_random_differs_from_hybrid(synthetic_inputs):
    rh = prune(synthetic_inputs, prune_ratio=0.5, strategy="hybrid", rng_seed=0)
    rr = prune(synthetic_inputs, prune_ratio=0.5, strategy="random", rng_seed=0)
    # Random picks 6 of 12 uniformly; probability of matching hybrid's specific
    # 5-split + 1-anchor pick is ~1/924. Test is robust at this size.
    assert rh.selected_item_ids != rr.selected_item_ids


# ---------------------------------------------------------------------------
# Stratified-only strategy
# ---------------------------------------------------------------------------


def test_stratified_only_keeps_feature_proportions(synthetic_inputs):
    # Source: 8 x, 4 y (2:1). With 50% target=6 and Hamilton allocation, expect
    # 4 x and 2 y (preserves ratio).
    result = prune(
        synthetic_inputs, prune_ratio=0.5, strategy="stratified_only", rng_seed=0
    )
    ids = result.selected_item_ids
    n_x = sum(
        1
        for i in ids
        if synthetic_inputs.feature_table["feature_a"][int(i[1:])] == "x"
    )
    n_y = sum(
        1
        for i in ids
        if synthetic_inputs.feature_table["feature_a"][int(i[1:])] == "y"
    )
    assert n_x == 4
    assert n_y == 2


def test_stratified_only_covers_all_tiers(synthetic_inputs):
    # Hybrid skews to split items; stratified_only should hit all 4 tiers
    # given that target=6 ≥ 4.
    result = prune(
        synthetic_inputs, prune_ratio=0.5, strategy="stratified_only", rng_seed=0
    )
    tiers_hit = set()
    rm = synthetic_inputs.response_matrix
    for id_ in result.selected_item_ids:
        idx = int(id_[1:])
        s = sum(rm[idx])
        if s == 0:
            tiers_hit.add(0)
        elif s == 3:
            tiers_hit.add(3)
        elif s == 1:
            tiers_hit.add(1)
        else:
            tiers_hit.add(2)
    assert tiers_hit == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Disagreement-only strategy
# ---------------------------------------------------------------------------


def test_disagreement_only_no_anchors_when_pool_fits(synthetic_inputs):
    # 4 of 12 = 33%; target=4 < |split|=6, so no anchors at all
    result = prune(
        synthetic_inputs, prune_ratio=0.33, strategy="disagreement_only", rng_seed=0
    )
    assert result.selected_count == 4
    selected = set(result.selected_item_ids)
    assert selected <= SPLIT_IDS


def test_disagreement_only_spills_to_anchors_at_high_ratio(synthetic_inputs):
    # 0.83 → target=10; |split|=6, so all 6 splits + 4 anchors
    result = prune(
        synthetic_inputs, prune_ratio=0.83, strategy="disagreement_only", rng_seed=0
    )
    selected = set(result.selected_item_ids)
    assert SPLIT_IDS <= selected
    assert len(selected & ANCHOR_IDS) == 4


# ---------------------------------------------------------------------------
# Result invariants
# ---------------------------------------------------------------------------


def test_output_is_sorted(synthetic_inputs):
    result = prune(synthetic_inputs, prune_ratio=0.5, strategy="hybrid", rng_seed=0)
    assert result.selected_item_ids == sorted(result.selected_item_ids)


def test_prune_ratio_one_returns_all(synthetic_inputs):
    result = prune(synthetic_inputs, prune_ratio=1.0, strategy="hybrid", rng_seed=0)
    assert result.selected_count == 12
    assert set(result.selected_item_ids) == set(synthetic_inputs.item_ids)


def test_bucket_counts_sum_to_selected(synthetic_inputs):
    result = prune(synthetic_inputs, prune_ratio=0.5, strategy="hybrid", rng_seed=0)
    tier_sum = (
        result.bucket_counts["split_hard"]
        + result.bucket_counts["split_easy"]
        + result.bucket_counts["anchor_hard"]
        + result.bucket_counts["anchor_easy"]
    )
    assert tier_sum == result.selected_count


def test_metadata_carries_config(synthetic_inputs):
    result = prune(
        synthetic_inputs,
        prune_ratio=0.5,
        strategy="hybrid",
        rng_seed=11,
        anchor_fraction=0.2,
    )
    assert result.metadata["rng_seed"] == 11
    assert result.metadata["anchor_fraction"] == 0.2
    assert result.metadata["n_items"] == 12
    assert result.metadata["n_models"] == 3
    assert result.metadata["tiers_used"] == "score_sum"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_non_binary_response_value():
    with pytest.raises(ValueError, match="not binary"):
        PruningInputs(item_ids=["a", "b"], response_matrix=[[0, 1], [2, 1]])


def test_rejects_length_mismatch():
    with pytest.raises(ValueError, match="rows"):
        PruningInputs(item_ids=["a", "b", "c"], response_matrix=[[0, 1], [1, 0]])


def test_rejects_duplicate_item_ids():
    with pytest.raises(ValueError, match="unique"):
        PruningInputs(item_ids=["a", "a"], response_matrix=[[0, 1], [1, 0]])


def test_rejects_feature_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        PruningInputs(
            item_ids=["a", "b"],
            response_matrix=[[0, 1], [1, 0]],
            feature_table={"x": ["p"]},
        )


def test_rejects_invalid_prune_ratio(synthetic_inputs):
    with pytest.raises(ValueError):
        prune(synthetic_inputs, prune_ratio=0.0)
    with pytest.raises(ValueError):
        prune(synthetic_inputs, prune_ratio=1.5)
    with pytest.raises(ValueError):
        prune(synthetic_inputs, prune_ratio=-0.1)


def test_rejects_unknown_strategy(synthetic_inputs):
    with pytest.raises(ValueError, match="unknown strategy"):
        prune(synthetic_inputs, prune_ratio=0.5, strategy="nonsense")


def test_rejects_bad_anchor_fraction(synthetic_inputs):
    with pytest.raises(ValueError):
        prune(synthetic_inputs, prune_ratio=0.5, anchor_fraction=1.5)


# ---------------------------------------------------------------------------
# Rasch (optional)
# ---------------------------------------------------------------------------


def test_rasch_orders_difficulty_correctly(synthetic_inputs):
    b, theta = fit_rasch_1pl(synthetic_inputs.response_matrix)
    assert len(b) == 12
    assert len(theta) == 3
    # all-fail items should have higher b than all-pass items
    all_fail = [b[i] for i in (0, 1)]
    all_pass = [b[i] for i in (8, 9, 10, 11)]
    assert min(all_fail) > max(all_pass)


def test_use_rasch_path_runs(synthetic_inputs):
    # The Rasch tier path should select a different but still valid set.
    result = prune(
        synthetic_inputs, prune_ratio=0.5, strategy="hybrid", rng_seed=0, use_rasch=True
    )
    assert result.metadata["tiers_used"] == "rasch_quartile"
    assert result.selected_count == 6


# ---------------------------------------------------------------------------
# custom_tiers — general per-item informativeness plug-in
# ---------------------------------------------------------------------------


def test_custom_tiers_overrides_score_sum():
    """When custom_tiers is provided, the row-sum-derived classification is
    ignored. The strategies should select according to the custom tiers."""
    item_ids = [f"i{n:02d}" for n in range(8)]
    # Response matrix says: all items are anchor-easy (all pass all 3 models).
    response_matrix = [[1, 1, 1] for _ in range(8)]
    # Caller supplies a per-item informativeness signal that contradicts
    # the score-sum tier — e.g. an encoder-stress score. Place 4 items in
    # the discrim pool (tiers 1 + 2) and 4 in the anchor pool (tiers 0 + 3).
    custom = [0, 0, 1, 1, 2, 2, 3, 3]  # 2 each in tiers 0..3
    inputs = PruningInputs(
        item_ids=item_ids,
        response_matrix=response_matrix,
        custom_tiers=custom,
    )
    result = prune(inputs, prune_ratio=0.5, strategy="hybrid", rng_seed=0)
    assert result.metadata["tiers_used"] == "custom_tiers"
    selected = {int(s[1:]) for s in result.selected_item_ids}
    # target = 4; anchor_slots = round(4*0.15) = 1; discrim_slots = 3.
    # Discrim pool = {i02,i03,i04,i05} (tiers 1,2); anchor pool = {i00,i01,i06,i07}.
    # Hybrid takes 3 from discrim + 1 from anchor pool, NOT the row-sum tiers
    # (which would have classified everything as anchor-easy → empty discrim).
    assert len(selected & {2, 3, 4, 5}) == 3, f"expected 3 discrim, got {selected & {2,3,4,5}}"
    assert len(selected & {0, 1, 6, 7}) == 1, f"expected 1 anchor, got {selected & {0,1,6,7}}"


def test_custom_tiers_takes_precedence_over_use_rasch():
    """If both custom_tiers and use_rasch=True are supplied, custom_tiers wins
    (and tiers_used reflects that)."""
    item_ids = [f"i{n:02d}" for n in range(6)]
    response_matrix = [[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1]]
    custom = [3, 3, 0, 1, 2, 2]  # arbitrary custom binning
    inputs = PruningInputs(
        item_ids=item_ids,
        response_matrix=response_matrix,
        custom_tiers=custom,
    )
    result = prune(
        inputs, prune_ratio=1.0, strategy="hybrid", rng_seed=0, use_rasch=True
    )
    assert result.metadata["tiers_used"] == "custom_tiers"
    assert result.selected_count == 6


def test_custom_tiers_rejects_bad_length():
    with pytest.raises(ValueError, match="custom_tiers"):
        PruningInputs(
            item_ids=["a", "b"],
            response_matrix=[[0, 1], [1, 0]],
            custom_tiers=[0],  # length 1, expected 2
        )


def test_custom_tiers_rejects_out_of_range_value():
    with pytest.raises(ValueError, match="custom_tiers"):
        PruningInputs(
            item_ids=["a", "b"],
            response_matrix=[[0, 1], [1, 0]],
            custom_tiers=[0, 4],  # 4 is not in {0,1,2,3}
        )
