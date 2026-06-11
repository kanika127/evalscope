"""Precompute pruned MMMU index sets — encoder-stress probe selection.

Usage:
    # Default: select from the 660 shipped reference samples
    python -m evalscope_ext.pruners.precompute_mmmu \\
        --predictions-dir <path/to/Evals/MMMU/predictions/glm-4.5v-fp8> \\
        --reviews-dir     <path/to/Evals/MMMU/reviews/glm-4.5v-fp8> \\
        --output-dir      evalscope_ext/pruners/cache \\
        --ratios          0.10 0.20 0.30 0.50 \\
        --strategies      hybrid random disagreement_only stratified_only

    # Full 12K via HuggingFace streaming (metadata only, NO 25GB image pull)
    python -m evalscope_ext.pruners.precompute_mmmu --source hf \\
        --output-dir evalscope_ext/pruners/cache \\
        --ratios     0.05 0.10 0.20

Stable key: the MMMU upstream id (form `validation_<Subject>_<n>`), accessed via
the adapter's `sample.metadata['id']`. Already verified unique across 660; we do
NOT hash for MMMU.

⚠ This is fundamentally different from LCB/AA-LCR pruning. We have ONE
reference model (glm-4.5v-fp8). At M=1 the disagreement-tier classification
collapses to "pass" vs "fail" — no discrimination axis. Instead we compute a
per-item ENCODER-STRESS SCORE from MMMU metadata + reference-model behavior,
quantile-bin it into custom_tiers, and let the universal core's stratification
and hybrid strategy operate on those tiers.

ENCODER-STRESS SCORE (locked weights; document as assumptions in Handout A):

    score = 0.45 * stress_img_type            # binary 1.0 if img_type ∈ HIGH-density set
          + 0.25 * grounding_intensity         # normalized count of image-type entries
          + 0.20 * topic_difficulty_weight     # 0.5 Easy / 0.75 Medium / 1.0 Hard
          + 0.10 * reference_failure_signal    # 1.0 if ref-model failed AND CoT > p75 tokens

    (HF-streaming mode rebalances: 0.50 img + 0.30 grounding + 0.20 difficulty,
    since no reference model exists for the full 12K.)

QUANTILE → TIER MAPPING:
    bottom 25%   → tier 0 (anchor / negative control, encoder-light)
    second 25%   → tier 1 (informative-low)
    third 25%    → tier 2 (informative-high)
    top 25%      → tier 2 (informative-high; lumped with q2 to keep tier 3 empty)

Hybrid then treats tier 1 + tier 2 (top 75% by stress) as the discrim pool and
tier 0 (bottom 25%) as the anchor / negative-control pool.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Image-type buckets for axis A (visual-information density)
# ---------------------------------------------------------------------------
HIGH_STRESS_IMG_TYPES = {
    "Tables",
    "Diagrams",
    "Plots and Charts",
    "Trees and Graphs",
    "Chemical Structures",
    "Technical Blueprints",
    "Microscopic Images",
    "Pathological Images",
    "Body Scans: MRI, CT scans, and X-rays",
    "Medical Images",
    "Music Sheets",
    "Maps",
    "Geometric Shapes",
    "Mathematical Notations",
}
LOW_STRESS_IMG_TYPES = {
    "Photographs",
    "Paintings",
    "Portraits",
    "Sculpture",
    "Comics and Cartoons",
    "Other",
}
DIFFICULTY_WEIGHT = {"Easy": 0.5, "Medium": 0.75, "Hard": 1.0}
QUARTILE_TO_TIER = {0: 0, 1: 1, 2: 2, 3: 2}  # see module docstring


def _parse_img_type(raw: Any) -> List[str]:
    """img_type is shipped as a Python-list-literal string, e.g. \"['Tables']\".
    Sometimes it is already a list (HF source). Returns a clean list of strs."""
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            v = ast.literal_eval(raw)
            if isinstance(v, list):
                return [str(x) for x in v]
            return [str(v)]
        except (ValueError, SyntaxError):
            return [raw]
    return [str(raw)]


def stress_img_type(img_type_list: List[str]) -> float:
    """1.0 if ANY entry is in the high-stress set, else 0.0."""
    for t in img_type_list:
        if t in HIGH_STRESS_IMG_TYPES:
            return 1.0
    return 0.0


def grounding_intensity_from_metadata(img_type_list: List[str]) -> float:
    """Proxy: normalized count of image-type categorizations.
    1 type → 0.33, 2 → 0.67, 3+ → 1.0. Conflates multi-image with multi-type
    but is the cheapest signal from metadata alone."""
    return min(len(img_type_list), 3) / 3.0


def grounding_intensity_from_question(question: str, n_image_fields: int) -> float:
    """When we have the question text (HF streaming source): count <image N>
    placeholders + image-field count, normalize."""
    placeholders = len(re.findall(r"<image\s*\d+>", question or ""))
    raw = max(placeholders, n_image_fields)
    return min(raw, 4) / 4.0


# ---------------------------------------------------------------------------
# Source: shipped predictions + reviews
# ---------------------------------------------------------------------------


def load_shipped(
    predictions_dir: str, reviews_dir: str
) -> Tuple[List[str], List[List[int]], List[Dict[str, Any]]]:
    """Return (item_ids, response_matrix, per_item_metadata).

    item_ids        — list of MMMU upstream ids (e.g. validation_Accounting_1)
    response_matrix — N × 1 binary (pass/fail) — single reference model
    per_item_metadata — list of dicts with all fields needed for stress scoring
    """
    pred_files = sorted(glob.glob(os.path.join(predictions_dir, "mmmu_*.jsonl")))
    rev_files = sorted(glob.glob(os.path.join(reviews_dir, "mmmu_*.jsonl")))
    if not pred_files:
        raise FileNotFoundError(f"No mmmu_*.jsonl in {predictions_dir!r}")
    if not rev_files:
        raise FileNotFoundError(f"No mmmu_*.jsonl in {reviews_dir!r}")

    # Load all predictions, key by upstream id
    n_tokens_by_id: Dict[str, int] = {}
    meta_by_id: Dict[str, Dict[str, Any]] = {}
    for fp in pred_files:
        with open(fp) as fh:
            for line in fh:
                if not line.strip():
                    continue
                o = json.loads(line)
                md = o["metadata"]
                upstream_id = md["id"]
                content = o["model_output"]["choices"][0]["logprobs"]["content"]
                n_tokens_by_id[upstream_id] = len(content)
                meta_by_id[upstream_id] = {
                    "id": upstream_id,
                    "subfield": md.get("subfield"),
                    "img_type_raw": md.get("img_type"),
                    "img_type_list": _parse_img_type(md.get("img_type")),
                    "topic_difficulty": md.get("topic_difficulty"),
                    "question_type": md.get("question_type"),
                    "n_response_tokens": len(content),
                }

    # Load reviews, get binary acc
    acc_by_id: Dict[str, int] = {}
    for fp in rev_files:
        with open(fp) as fh:
            for line in fh:
                if not line.strip():
                    continue
                o = json.loads(line)
                sc = o["sample_score"]
                upstream_id = sc["sample_metadata"]["id"]
                acc_by_id[upstream_id] = 1 if float(sc["score"]["value"]["acc"]) >= 0.5 else 0

    # Join on id
    common = sorted(set(meta_by_id.keys()) & set(acc_by_id.keys()))
    if not common:
        raise RuntimeError("No prediction/review intersection on metadata.id")

    item_ids = common
    response_matrix = [[acc_by_id[i]] for i in item_ids]
    per_item_meta = [dict(meta_by_id[i], passed=bool(acc_by_id[i])) for i in item_ids]
    return item_ids, response_matrix, per_item_meta


# ---------------------------------------------------------------------------
# Source: HuggingFace streaming (metadata only, no image bytes persisted)
# ---------------------------------------------------------------------------


def load_hf_streaming(
    repo: str = "MMMU/MMMU",
    split: str = "validation",
) -> Tuple[List[str], List[List[int]], List[Dict[str, Any]]]:
    """Stream the full MMMU dataset via `datasets`. Pull metadata only — image
    bytes are read by datasets in chunks and discarded after each row.
    Returns the same (item_ids, response_matrix, per_item_meta) shape as
    load_shipped(). response_matrix is N × 0 (no reference model)."""
    try:
        from datasets import get_dataset_config_names, load_dataset  # type: ignore
    except ImportError as e:
        raise ImportError(
            "`datasets` is required for --source hf. Install with: "
            "pip install datasets"
        ) from e

    item_ids: List[str] = []
    per_item_meta: List[Dict[str, Any]] = []
    subjects = get_dataset_config_names(repo)
    for subj in subjects:
        ds = load_dataset(repo, subj, split=split, streaming=True)
        for row in ds:
            # Count populated image fields
            n_img_fields = sum(1 for k in row if k.startswith("image_") and row.get(k))
            upstream_id = row.get("id")
            if not upstream_id:
                continue
            item_ids.append(upstream_id)
            per_item_meta.append({
                "id": upstream_id,
                "subfield": row.get("subfield"),
                "img_type_raw": row.get("img_type"),
                "img_type_list": _parse_img_type(row.get("img_type")),
                "topic_difficulty": row.get("topic_difficulty"),
                "question_type": row.get("question_type"),
                "n_image_fields": n_img_fields,
                "question": row.get("question") or "",
                "passed": None,  # no reference model on full HF set
            })
    response_matrix: List[List[int]] = [[] for _ in item_ids]
    return item_ids, response_matrix, per_item_meta


# ---------------------------------------------------------------------------
# Encoder-stress score
# ---------------------------------------------------------------------------


def compute_stress_scores(
    per_item_meta: List[Dict[str, Any]],
    use_reference_failure: bool,
) -> List[float]:
    """Returns scores ∈ [0, 1] per item.

    Two formulae, chosen by use_reference_failure:
      With reference (shipped): 0.45 img + 0.25 grounding + 0.20 difficulty + 0.10 ref-fail
      Without reference (HF):   0.50 img + 0.30 grounding + 0.20 difficulty
    """
    # Compute p75 of response tokens (for ref_failure_signal threshold)
    if use_reference_failure:
        n_tokens = sorted(m.get("n_response_tokens", 0) for m in per_item_meta)
        p75 = n_tokens[(len(n_tokens) * 3) // 4] if n_tokens else 0
    scores: List[float] = []
    for m in per_item_meta:
        img_list = m["img_type_list"]
        s_img = stress_img_type(img_list)
        if "question" in m:
            s_grd = grounding_intensity_from_question(
                m["question"], m.get("n_image_fields", 1)
            )
        else:
            s_grd = grounding_intensity_from_metadata(img_list)
        s_diff = DIFFICULTY_WEIGHT.get(m.get("topic_difficulty", "Medium"), 0.75)
        if use_reference_failure:
            s_ref = 1.0 if (not m["passed"] and m["n_response_tokens"] > p75) else 0.0
            score = 0.45 * s_img + 0.25 * s_grd + 0.20 * s_diff + 0.10 * s_ref
        else:
            score = 0.50 * s_img + 0.30 * s_grd + 0.20 * s_diff
        scores.append(score)
    return scores


def stress_to_custom_tiers(stress_scores: List[float]) -> List[int]:
    """Quantile-bin into 4 quartiles, then map per QUARTILE_TO_TIER."""
    n = len(stress_scores)
    if n == 0:
        return []
    # Quartile boundaries via sorted positions
    sorted_with_idx = sorted(range(n), key=lambda i: stress_scores[i])
    q_size = n / 4.0
    tiers = [0] * n
    for rank, orig_i in enumerate(sorted_with_idx):
        q = min(int(rank // q_size), 3)
        tiers[orig_i] = QUARTILE_TO_TIER[q]
    return tiers


# ---------------------------------------------------------------------------
# Feature table for stratification
# ---------------------------------------------------------------------------


def build_feature_table(per_item_meta: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Stratify on subfield + img_type_bucket + question_type + difficulty.

    img_type_bucket collapses the 30+ img_type strings into {'high', 'low', 'mixed'}
    so the cross-product remains tractable.
    """
    def bucket(img_list: List[str]) -> str:
        any_high = any(t in HIGH_STRESS_IMG_TYPES for t in img_list)
        any_low = any(t in LOW_STRESS_IMG_TYPES for t in img_list)
        if any_high and any_low:
            return "mixed"
        if any_high:
            return "high"
        if any_low:
            return "low"
        return "unknown"
    return {
        "img_type_bucket": [bucket(m["img_type_list"]) for m in per_item_meta],
        "question_type": [m.get("question_type", "unknown") for m in per_item_meta],
        "difficulty": [m.get("topic_difficulty", "Medium") for m in per_item_meta],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    source: str,
    predictions_dir: Optional[str],
    reviews_dir: Optional[str],
    hf_repo: str,
    hf_split: str,
    output_dir: str,
    ratios: List[float],
    strategies: List[str],
    rng_seed: int = 0,
    anchor_fraction: float = 0.15,
) -> None:
    from evalscope_ext.pruners import PruningInputs, prune
    from evalscope_ext.pruners.core import TIER_NAMES

    if source == "shipped":
        assert predictions_dir and reviews_dir, "shipped source requires --predictions-dir and --reviews-dir"
        print(f"Loading shipped predictions from {predictions_dir}")
        print(f"Loading shipped reviews     from {reviews_dir}")
        item_ids, matrix, per_item = load_shipped(predictions_dir, reviews_dir)
        use_ref_fail = True
        source_label = "shipped"
    elif source == "hf":
        print(f"Streaming HuggingFace dataset {hf_repo} split={hf_split} (metadata only)")
        item_ids, matrix, per_item = load_hf_streaming(hf_repo, hf_split)
        use_ref_fail = False
        source_label = f"hf:{hf_repo}/{hf_split}"
    else:
        raise ValueError(f"unknown source {source!r}; valid: shipped, hf")

    n = len(item_ids)
    print(f"  Loaded {n} items")
    if use_ref_fail:
        n_pass = sum(r[0] for r in matrix)
        print(f"  Reference pass rate: {n_pass}/{n} = {n_pass/n:.1%}")

    # Encoder-stress score + custom_tiers
    scores = compute_stress_scores(per_item, use_reference_failure=use_ref_fail)
    custom_tiers = stress_to_custom_tiers(scores)
    print(f"  Stress quartiles → tiers: {dict(Counter(custom_tiers))}")
    print(f"  Stress score: min={min(scores):.3f} p50={sorted(scores)[n//2]:.3f} max={max(scores):.3f}")

    # Feature table
    feature_table = build_feature_table(per_item)
    print(f"  img_type_bucket dist: {dict(Counter(feature_table['img_type_bucket']))}")

    # PruningInputs — note response_matrix may be N×0 if HF source (no ref model).
    # In that case we pad to N×1 with zeros so PruningInputs validation passes;
    # the row-sum tiers will be all zero but we override via custom_tiers anyway.
    if not matrix or (matrix and not matrix[0]):
        matrix = [[0] for _ in range(n)]
    inputs = PruningInputs(
        item_ids=item_ids,
        response_matrix=matrix,
        feature_table=feature_table,
        custom_tiers=custom_tiers,
    )

    os.makedirs(output_dir, exist_ok=True)

    for strategy in strategies:
        for ratio in ratios:
            result = prune(
                inputs,
                prune_ratio=ratio,
                strategy=strategy,
                rng_seed=rng_seed,
                anchor_fraction=anchor_fraction,
            )
            out = {
                "selected_ids": result.selected_item_ids,
                "metadata": {
                    "strategy": strategy,
                    "prune_ratio": ratio,
                    "target_count": result.target_count,
                    "selected_count": result.selected_count,
                    "bucket_counts": result.bucket_counts,
                    "n_items_total": n,
                    "rng_seed": rng_seed,
                    "anchor_fraction": anchor_fraction,
                    "source": source_label,
                    "tiers_used": result.metadata["tiers_used"],
                    "features_used": list(feature_table.keys()),
                    "stress_formula": (
                        "0.45*img + 0.25*grounding + 0.20*difficulty + 0.10*ref_fail"
                        if use_ref_fail
                        else "0.50*img + 0.30*grounding + 0.20*difficulty"
                    ),
                    "quartile_to_tier": QUARTILE_TO_TIER,
                    "notes": (
                        "MMMU stable key is metadata.id (form validation_<Subject>_<n>) — "
                        "no content hashing. Selection is encoder-stress driven, not "
                        "disagreement-driven (we have 1 reference model)."
                    ),
                },
            }
            fname = f"mmmu_{strategy}_r{int(ratio * 100):03d}.json"
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "w") as f:
                json.dump(out, f, indent=2)
            bc = result.bucket_counts
            print(
                f"  [{strategy:20s}  ratio={ratio:.2f}]  "
                f"selected={result.selected_count}/{n}  "
                f"anchor_hard={bc['anchor_hard']:3d}  "
                f"split_hard={bc['split_hard']:3d}  "
                f"split_easy={bc['split_easy']:3d}  "
                f"anchor_easy={bc['anchor_easy']:3d}  "
                f"→ {fpath}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["shipped", "hf"], default="shipped")
    parser.add_argument("--predictions-dir")
    parser.add_argument("--reviews-dir")
    parser.add_argument("--hf-repo", default="MMMU/MMMU")
    parser.add_argument("--hf-split", default="validation")
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).parent / "cache")
    )
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=[0.10, 0.20, 0.30, 0.50]
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["hybrid", "random", "disagreement_only", "stratified_only"],
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--anchor-fraction", type=float, default=0.15)
    args = parser.parse_args()
    run(
        source=args.source,
        predictions_dir=args.predictions_dir,
        reviews_dir=args.reviews_dir,
        hf_repo=args.hf_repo,
        hf_split=args.hf_split,
        output_dir=args.output_dir,
        ratios=args.ratios,
        strategies=args.strategies,
        rng_seed=args.rng_seed,
        anchor_fraction=args.anchor_fraction,
    )


if __name__ == "__main__":
    main()
