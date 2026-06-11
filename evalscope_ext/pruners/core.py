"""Universal pruning core. Stdlib only, deterministic, benchmark-agnostic.

Inputs are three generic arrays:
    item_ids        — stable per-item identifier (any unique strings)
    response_matrix — (N items × M reference models), binary 0/1 scores
    feature_table   — optional column→values metadata for stratification

Output: a sorted list of selected item_ids plus diagnostic bucket counts.

Strategies:
    'hybrid'            — headline: split-item priority + stratified anchors
    'random'            — uniform-random baseline
    'stratified_only'   — stratify-only baseline (no disagreement preference)
    'disagreement_only' — split-only baseline (spills to anchors if needed)

The headline 'hybrid' strategy combines two ideas:

1. Discrimination by split pattern. With M reference models and binary scores,
   items where the models DISAGREE (0 < score_sum < M) carry the signal: their
   pass-rate distinguishes the reference models from one another and is the
   axis along which a future candidate model is likely to be informative.

2. Generalization anchors. Items where all reference models AGREE
   (score_sum == 0 or == M) carry zero discrimination signal for THIS panel
   of reference models, but they hedge against the case where a 4th
   (candidate) model has a different boundary — e.g. an item that is all-pass
   for the 3 reference models may split a weaker candidate. We keep a small
   stratified slice of these to prevent overfitting the pruned set to the
   reference panel.

The user emphasized the score-derived difficulty tier (0/1/2/3 at M=3) as the
guaranteed-present axis when external metadata is sparse — it is ALWAYS used
as a stratification axis, so stratification degrades gracefully to "stratify
by tier only" when feature_table is empty or None.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

VALID_STRATEGIES = ("hybrid", "random", "stratified_only", "disagreement_only")

TIER_ANCHOR_HARD = 0
TIER_SPLIT_HARD = 1
TIER_SPLIT_EASY = 2
TIER_ANCHOR_EASY = 3
TIER_NAMES = {
    TIER_ANCHOR_HARD: "anchor_hard",
    TIER_SPLIT_HARD: "split_hard",
    TIER_SPLIT_EASY: "split_easy",
    TIER_ANCHOR_EASY: "anchor_easy",
}


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PruningInputs:
    """Generic, benchmark-agnostic inputs.

    item_ids        — sequence of unique strings, length N
    response_matrix — sequence of N rows; each row has M binary values (0/1)
    feature_table   — optional mapping {column_name: sequence_of_length_N}
                      values are treated as categorical (hashed to bins)
    custom_tiers    — optional per-item informativeness signal, length N,
                      values in {0, 1, 2, 3}. When provided, OVERRIDES the
                      row-sum-derived tier classification used by the
                      strategies. Lets any per-item signal substitute for
                      disagreement-derived tiers (e.g. encoder-stress score
                      quantiles) while reusing the same stratification and
                      pool-allocation machinery. Tier semantics are unchanged:
                      0 = anchor-hard, 1 = split-hard, 2 = split-easy,
                      3 = anchor-easy. Callers are responsible for mapping
                      their signal onto these bins coherently.
    """

    item_ids: Sequence[str]
    response_matrix: Sequence[Sequence[int]]
    feature_table: Optional[Mapping[str, Sequence[Any]]] = None
    custom_tiers: Optional[Sequence[int]] = None

    def __post_init__(self) -> None:
        n = len(self.item_ids)
        if n == 0:
            raise ValueError("item_ids must be non-empty")
        if len(set(self.item_ids)) != n:
            raise ValueError("item_ids must be unique")
        if len(self.response_matrix) != n:
            raise ValueError(
                f"response_matrix has {len(self.response_matrix)} rows but expected {n}"
            )
        if n > 0:
            m = len(self.response_matrix[0])
            if m < 1:
                raise ValueError("response_matrix rows must have at least 1 model")
            for i, row in enumerate(self.response_matrix):
                if len(row) != m:
                    raise ValueError(
                        f"response_matrix row {i} has length {len(row)}, expected {m}"
                    )
                for j, v in enumerate(row):
                    if v not in (0, 1):
                        raise ValueError(
                            f"response_matrix[{i}][{j}]={v!r} is not binary 0/1"
                        )
        if self.custom_tiers is not None:
            if len(self.custom_tiers) != n:
                raise ValueError(
                    f"custom_tiers has length {len(self.custom_tiers)}, expected {n}"
                )
            for i, t in enumerate(self.custom_tiers):
                if not isinstance(t, int) or t not in (0, 1, 2, 3):
                    raise ValueError(
                        f"custom_tiers[{i}]={t!r} must be int in {{0,1,2,3}}"
                    )
        if self.feature_table is not None:
            for col, vals in self.feature_table.items():
                if len(vals) != n:
                    raise ValueError(
                        f"feature_table[{col!r}] has length {len(vals)}, expected {n}"
                    )

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_models(self) -> int:
        return len(self.response_matrix[0]) if self.response_matrix else 0


@dataclass
class PruningResult:
    selected_item_ids: List[str]
    strategy: str
    prune_ratio: float
    target_count: int
    selected_count: int
    bucket_counts: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tier + Rasch
# ---------------------------------------------------------------------------


def classify_tiers(response_matrix: Sequence[Sequence[int]]) -> List[int]:
    """Map each item to a tier in {0, 1, 2, 3} from its row sum.

    tier 0: all fail        (anchor-hard)
    tier 1: 0 < s < M/2     (split-hard, more failures than passes)
    tier 2: M/2 <= s < M    (split-easy, more passes than failures)
    tier 3: all pass        (anchor-easy)

    At M=3 this yields exactly the {0,1,2,3} = {all-fail, 1-of-3, 2-of-3,
    all-pass} buckets the LCB shipped data exhibits.
    """
    tiers: List[int] = []
    for row in response_matrix:
        m = len(row)
        s = sum(row)
        if s == 0:
            tiers.append(TIER_ANCHOR_HARD)
        elif s == m:
            tiers.append(TIER_ANCHOR_EASY)
        elif s * 2 < m:
            tiers.append(TIER_SPLIT_HARD)
        else:
            tiers.append(TIER_SPLIT_EASY)
    return tiers


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fit_rasch_1pl(
    response_matrix: Sequence[Sequence[int]],
    *,
    max_iter: int = 200,
    lr: float = 0.3,
    ridge: float = 1e-3,
    tol: float = 1e-4,
) -> Tuple[List[float], List[float]]:
    """Simple joint-MLE 1-PL Rasch fit.

    Model: P(correct | model_m, item_i) = sigmoid(theta_m - b_i)

    Returns (b: difficulties per item, theta: abilities per model).
    Gauge: theta centered at 0 (location indeterminacy resolved).

    Off by default; 1-PL is marginal at M=3 — the score-derived tier remains
    the guaranteed fallback.
    """
    n = len(response_matrix)
    m = len(response_matrix[0]) if n else 0
    if n == 0 or m == 0:
        return [], []
    b = [0.0] * n
    theta = [0.0] * m
    prev_nll = float("inf")
    for _ in range(max_iter):
        # Gradient of NLL wrt b_i: sum_m (R[i][m] - p) ; sign convention:
        # since p = sigmoid(theta - b), dp/db = -p(1-p), and after the
        # derivation d NLL / d b = sum_m (R - p). To MINIMIZE NLL, step
        # in -gradient direction: b -= lr * grad.
        for i in range(n):
            grad_b = ridge * b[i]
            for j in range(m):
                p = _sigmoid(theta[j] - b[i])
                grad_b += response_matrix[i][j] - p
            b[i] -= lr * grad_b
        for j in range(m):
            grad_th = ridge * theta[j]
            for i in range(n):
                p = _sigmoid(theta[j] - b[i])
                grad_th += p - response_matrix[i][j]
            theta[j] -= lr * grad_th
        # Re-center theta (and shift b by the same constant to keep the
        # invariant theta - b unchanged).
        mean_th = sum(theta) / m
        theta = [t - mean_th for t in theta]
        b = [bi - mean_th for bi in b]
        # convergence check
        nll = 0.0
        for i in range(n):
            for j in range(m):
                p = _sigmoid(theta[j] - b[i])
                p = min(max(p, 1e-12), 1 - 1e-12)
                nll -= response_matrix[i][j] * math.log(p) + (
                    1 - response_matrix[i][j]
                ) * math.log(1 - p)
        if abs(prev_nll - nll) < tol:
            break
        prev_nll = nll
    return b, theta


def _rasch_tiers(b: Sequence[float]) -> List[int]:
    """Bin Rasch difficulties into 4 quantile tiers, preserving the tier
    semantics: 0 = hardest (highest b), 3 = easiest (lowest b)."""
    n = len(b)
    if n == 0:
        return []
    sorted_b = sorted(b)
    # quartile cut points
    q1 = sorted_b[n // 4]
    q2 = sorted_b[n // 2]
    q3 = sorted_b[(3 * n) // 4]
    out: List[int] = []
    for bi in b:
        if bi >= q3:
            out.append(TIER_ANCHOR_HARD)
        elif bi >= q2:
            out.append(TIER_SPLIT_HARD)
        elif bi >= q1:
            out.append(TIER_SPLIT_EASY)
        else:
            out.append(TIER_ANCHOR_EASY)
    return out


# ---------------------------------------------------------------------------
# Strata + Hamilton allocation
# ---------------------------------------------------------------------------


def _build_strata(
    item_indices: Sequence[int],
    tiers: Sequence[int],
    feature_table: Optional[Mapping[str, Sequence[Any]]],
    stratify_columns: Optional[Sequence[str]],
) -> Dict[Tuple[Any, ...], List[int]]:
    """Group item indices into strata by (tier, feature_values...).

    Score-derived tier is ALWAYS included as the first stratification axis
    (the guaranteed-present fallback). Additional axes come from
    `feature_table` columns named in `stratify_columns` (or all columns
    when None).
    """
    cols: List[str]
    if feature_table is None or len(feature_table) == 0:
        cols = []
    elif stratify_columns is None:
        cols = sorted(feature_table.keys())
    else:
        cols = [c for c in stratify_columns if c in feature_table]
        cols.sort()

    strata: Dict[Tuple[Any, ...], List[int]] = defaultdict(list)
    for i in item_indices:
        key: List[Any] = [tiers[i]]
        for c in cols:
            key.append(feature_table[c][i])  # type: ignore[index]
        strata[tuple(key)].append(i)
    return dict(strata)


def _hamilton_allocate(
    strata_sizes: Sequence[int], total: int
) -> List[int]:
    """Largest-remainder (Hamilton) integer allocation.

    Distributes `total` slots across len(strata_sizes) strata in proportion to
    each stratum's size. Deterministic on ties (stable by index).
    """
    n_strata = len(strata_sizes)
    if n_strata == 0 or total <= 0:
        return [0] * n_strata
    pool_total = sum(strata_sizes)
    if pool_total == 0:
        return [0] * n_strata
    raw = [total * s / pool_total for s in strata_sizes]
    floors = [int(math.floor(r)) for r in raw]
    # never exceed the stratum size
    floors = [min(f, strata_sizes[k]) for k, f in enumerate(floors)]
    remaining = total - sum(floors)
    # distribute the remainder to strata with largest fractional parts,
    # breaking ties on lower index for determinism, and refusing any
    # stratum that's already at capacity
    fracs = sorted(
        [(raw[k] - math.floor(raw[k]), k) for k in range(n_strata)],
        key=lambda x: (-x[0], x[1]),
    )
    k_idx = 0
    while remaining > 0 and k_idx < len(fracs):
        _, k = fracs[k_idx]
        if floors[k] < strata_sizes[k]:
            floors[k] += 1
            remaining -= 1
        k_idx += 1
    # second pass: if some strata are saturated and we still have slack,
    # round-robin into strata with remaining capacity
    if remaining > 0:
        ordered = sorted(range(n_strata), key=lambda k: (-strata_sizes[k], k))
        while remaining > 0:
            progressed = False
            for k in ordered:
                if floors[k] < strata_sizes[k]:
                    floors[k] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                break
    return floors


def _stratified_sample(
    strata: Dict[Tuple[Any, ...], List[int]],
    total: int,
    rng: random.Random,
) -> List[int]:
    """Hamilton-allocate slots across strata, then sample within each."""
    if total <= 0 or not strata:
        return []
    keys = sorted(strata.keys(), key=lambda k: tuple(repr(v) for v in k))
    sizes = [len(strata[k]) for k in keys]
    quotas = _hamilton_allocate(sizes, total)
    chosen: List[int] = []
    for k, q in zip(keys, quotas):
        pool = sorted(strata[k])
        if q >= len(pool):
            chosen.extend(pool)
        else:
            chosen.extend(rng.sample(pool, q))
    return chosen


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _strategy_random(
    n: int, target: int, rng: random.Random
) -> List[int]:
    if target >= n:
        return list(range(n))
    return sorted(rng.sample(range(n), target))


def _strategy_stratified_only(
    inputs: PruningInputs,
    tiers: Sequence[int],
    target: int,
    rng: random.Random,
    stratify_columns: Optional[Sequence[str]],
) -> List[int]:
    strata = _build_strata(
        list(range(inputs.n_items)), tiers, inputs.feature_table, stratify_columns
    )
    return _stratified_sample(strata, target, rng)


def _strategy_disagreement_only(
    inputs: PruningInputs,
    tiers: Sequence[int],
    target: int,
    rng: random.Random,
) -> List[int]:
    split_idx = [i for i, t in enumerate(tiers) if t in (TIER_SPLIT_HARD, TIER_SPLIT_EASY)]
    anchor_idx = [
        i for i, t in enumerate(tiers) if t in (TIER_ANCHOR_HARD, TIER_ANCHOR_EASY)
    ]
    if target <= len(split_idx):
        return sorted(rng.sample(sorted(split_idx), target))
    chosen = list(sorted(split_idx))
    need = target - len(chosen)
    if need > 0 and anchor_idx:
        need = min(need, len(anchor_idx))
        chosen.extend(rng.sample(sorted(anchor_idx), need))
    return chosen


def _strategy_hybrid(
    inputs: PruningInputs,
    tiers: Sequence[int],
    target: int,
    rng: random.Random,
    stratify_columns: Optional[Sequence[str]],
    anchor_fraction: float,
) -> Tuple[List[int], Dict[str, int]]:
    discrim_idx = [
        i for i, t in enumerate(tiers) if t in (TIER_SPLIT_HARD, TIER_SPLIT_EASY)
    ]
    anchor_idx = [
        i for i, t in enumerate(tiers) if t in (TIER_ANCHOR_HARD, TIER_ANCHOR_EASY)
    ]

    target = min(target, len(discrim_idx) + len(anchor_idx))
    anchor_slots = int(round(target * anchor_fraction))
    anchor_slots = min(anchor_slots, len(anchor_idx))
    discrim_slots = target - anchor_slots
    if discrim_slots > len(discrim_idx):
        # spill the overflow into anchors
        overflow = discrim_slots - len(discrim_idx)
        discrim_slots = len(discrim_idx)
        anchor_slots = min(anchor_slots + overflow, len(anchor_idx))

    discrim_strata = _build_strata(
        discrim_idx, tiers, inputs.feature_table, stratify_columns
    )
    anchor_strata = _build_strata(
        anchor_idx, tiers, inputs.feature_table, stratify_columns
    )

    chosen_discrim = _stratified_sample(discrim_strata, discrim_slots, rng)
    chosen_anchor = _stratified_sample(anchor_strata, anchor_slots, rng)

    diagnostics = {
        "discrim_pool": len(discrim_idx),
        "anchor_pool": len(anchor_idx),
        "discrim_slots": discrim_slots,
        "anchor_slots": anchor_slots,
        "selected_discrim": len(chosen_discrim),
        "selected_anchor": len(chosen_anchor),
    }
    return chosen_discrim + chosen_anchor, diagnostics


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def prune(
    inputs: PruningInputs,
    prune_ratio: float,
    strategy: str = "hybrid",
    *,
    rng_seed: int = 0,
    anchor_fraction: float = 0.15,
    stratify_columns: Optional[Sequence[str]] = None,
    use_rasch: bool = False,
) -> PruningResult:
    """Select a prune_ratio fraction of items via the given strategy.

    prune_ratio is the fraction of items to KEEP (0 < r <= 1).
    """
    if not (0.0 < prune_ratio <= 1.0):
        raise ValueError(f"prune_ratio must be in (0, 1], got {prune_ratio}")
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; valid: {VALID_STRATEGIES}"
        )
    if not (0.0 <= anchor_fraction <= 1.0):
        raise ValueError(
            f"anchor_fraction must be in [0, 1], got {anchor_fraction}"
        )

    n = inputs.n_items
    target = int(round(n * prune_ratio))
    target = max(1, min(target, n))
    rng = random.Random(rng_seed)

    # Tier derivation precedence:
    #   1. custom_tiers (caller-provided per-item informativeness signal)
    #   2. 1-PL Rasch quartile bins (use_rasch=True)
    #   3. row-sum-derived 4-tier classification (default)
    if inputs.custom_tiers is not None:
        tiers = list(inputs.custom_tiers)
        tiers_source = "custom_tiers"
    elif use_rasch:
        b, _theta = fit_rasch_1pl(inputs.response_matrix)
        tiers = _rasch_tiers(b)
        tiers_source = "rasch_quartile"
    else:
        tiers = classify_tiers(inputs.response_matrix)
        tiers_source = "score_sum"

    diagnostics: Dict[str, int] = {}

    if strategy == "random":
        chosen = _strategy_random(n, target, rng)
    elif strategy == "stratified_only":
        chosen = _strategy_stratified_only(
            inputs, tiers, target, rng, stratify_columns
        )
    elif strategy == "disagreement_only":
        chosen = _strategy_disagreement_only(inputs, tiers, target, rng)
    elif strategy == "hybrid":
        chosen, diagnostics = _strategy_hybrid(
            inputs, tiers, target, rng, stratify_columns, anchor_fraction
        )
    else:  # pragma: no cover — guarded above
        raise AssertionError(strategy)

    # Tier breakdown across the selected indices
    bucket_counts: Dict[str, int] = Counter(TIER_NAMES[tiers[i]] for i in chosen)
    bucket_counts = dict(bucket_counts)
    for nm in TIER_NAMES.values():
        bucket_counts.setdefault(nm, 0)
    bucket_counts.update(diagnostics)

    selected_ids = sorted(inputs.item_ids[i] for i in chosen)
    metadata = {
        "rng_seed": rng_seed,
        "anchor_fraction": anchor_fraction,
        "stratify_columns": list(stratify_columns) if stratify_columns else None,
        "use_rasch": use_rasch,
        "n_items": n,
        "n_models": inputs.n_models,
        "tiers_used": tiers_source,
    }
    return PruningResult(
        selected_item_ids=selected_ids,
        strategy=strategy,
        prune_ratio=prune_ratio,
        target_count=target,
        selected_count=len(selected_ids),
        bucket_counts=bucket_counts,
        metadata=metadata,
    )
