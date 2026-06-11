"""Precompute pruned AA-LCR index sets from the 3 shipped review files.

Usage:
    python -m evalscope_ext.pruners.precompute_aa_lcr \\
        --reviews-dir  <path/to/Evals/Part 1/reviews> \\
        --output-dir   evalscope_ext/pruners/cache \\
        --ratios       0.10 0.20 0.30 0.50 \\
        --strategies   hybrid random disagreement_only stratified_only

Outputs one JSON per (strategy, ratio):
    cache/aa_lcr_<strategy>_r<ratio>.json   — {"selected_hashes": [...], "metadata": {...}}

Stable key: SHA-256(question.strip().encode("utf-8")).hexdigest()
    extracted from: sample_score.sample_metadata.question  (review files)
    same field surfaced by adapter as: sample.metadata['question']

Score field: sample_score.score.value['acc'] (1.0 = pass, 0.0 = fail).

⚠ Judge-noise caveat: AA-LCR accuracy is determined by an LLM judge, which is
non-deterministic. The binary pass/fail in these files reflects a single judge
run at collection time. The pruning selection (and any strategy comparison)
carries this judge variance — items near the decision boundary may flip on
re-evaluation. This is inherent to AA-LCR and should be accounted for in
downstream comparison: treat ±2–3% accuracy differences across strategies as
within-noise, and prefer comparisons at coarser granularity (tier composition,
n_selected) over per-item level.

Token bins for feature_table ('token_bin'): AA-LCR items are 71K–115K tokens.
Quartile-aligned edges at p25≈84K, p50≈95K, p75≈107K give four roughly
equal-sized bins (xs/sm/md/lg).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Quartile-aligned bin edges (derived from the 100-item distribution):
#   xs: < 84_000   (≈ p0–p25)
#   sm: 84_000 – 95_000   (≈ p25–p50)
#   md: 95_000 – 107_000  (≈ p50–p75)
#   lg: >= 107_000         (≈ p75–p100)
TOKEN_BIN_EDGES = [84_000, 95_000, 107_000]


def bin_tokens(n: int) -> str:
    if n < TOKEN_BIN_EDGES[0]:
        return "xs"
    if n < TOKEN_BIN_EDGES[1]:
        return "sm"
    if n < TOKEN_BIN_EDGES[2]:
        return "md"
    return "lg"


# ---------------------------------------------------------------------------
# Load reviews
# ---------------------------------------------------------------------------


def _load_review_file(path: str) -> Tuple[Dict[int, int], Dict[int, dict]]:
    """Return (scores, metadata) both keyed by shipped_index.

    scores:   {index: binary_pass}
    metadata: {index: sample_metadata dict}
    """
    scores: Dict[int, int] = {}
    meta:   Dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            obj  = json.loads(line)
            idx  = obj["index"]
            raw  = obj["sample_score"]["score"]["value"]["acc"]
            scores[idx] = 1 if float(raw) >= 0.5 else 0
            meta[idx]   = obj["sample_score"]["sample_metadata"]
    return scores, meta


def load_response_matrix(
    reviews_dir: str,
) -> Tuple[List[int], List[List[int]], List[str], Dict[int, dict]]:
    """Load all 3 AA-LCR review files.

    Returns:
        sorted_indices   — list of 100 sorted shipped indices
        response_matrix  — (100 × 3) binary scores, model order = sorted filenames
        model_names      — list of 3 model name strings
        representative_meta — {index: sample_metadata} from the first model file
                              (question and input_tokens are the same across models)
    """
    pattern = os.path.join(reviews_dir, "aa_lcr__*.jsonl")
    files   = sorted(glob.glob(pattern))
    if len(files) != 3:
        raise FileNotFoundError(
            f"Expected exactly 3 aa_lcr__*.jsonl in {reviews_dir}, found {len(files)}: {files}"
        )
    model_names: List[str] = []
    per_scores: List[Dict[int, int]] = []
    rep_meta: Dict[int, dict] = {}
    for i, fp in enumerate(files):
        name = re.search(r"aa_lcr__(.+)\.jsonl", fp)
        model_names.append(name.group(1) if name else fp)
        sc, md = _load_review_file(fp)
        per_scores.append(sc)
        if i == 0:
            rep_meta = md

    all_indices = set(per_scores[0].keys())
    for sc in per_scores[1:]:
        all_indices &= set(sc.keys())
    sorted_idx = sorted(all_indices)

    matrix: List[List[int]] = [
        [per_scores[m][i] for m in range(len(per_scores))] for i in sorted_idx
    ]
    return sorted_idx, matrix, model_names, rep_meta


# ---------------------------------------------------------------------------
# Stable key
# ---------------------------------------------------------------------------


def build_hash_mapping(
    sorted_indices: List[int],
    meta: Dict[int, dict],
) -> Tuple[List[str], Dict[str, str]]:
    """Return (item_ids, index_to_hash) for all 100 items.

    item_ids[i]  = SHA-256(question.strip()) for sorted_indices[i]
    index_to_hash = {str(idx): hash}
    """
    item_ids: List[str] = []
    index_to_hash: Dict[str, str] = {}
    for idx in sorted_indices:
        question = meta[idx]["question"]
        h = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()
        item_ids.append(h)
        index_to_hash[str(idx)] = h
    return item_ids, index_to_hash


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------


def build_feature_table(
    sorted_indices: List[int],
    meta: Dict[int, dict],
) -> Dict[str, List[str]]:
    """Return feature_table = {'token_bin': [...]}.

    token_bin is derived from meta[idx]['input_tokens'] using quartile-aligned
    bin edges that produce roughly equal-sized buckets for the 100-item AA-LCR
    distribution.
    """
    bins = [bin_tokens(meta[idx].get("input_tokens", 0)) for idx in sorted_indices]
    return {"token_bin": bins}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    reviews_dir: str,
    output_dir: str,
    ratios: List[float],
    strategies: List[str],
    rng_seed: int = 0,
    anchor_fraction: float = 0.15,
) -> None:
    from evalscope_ext.pruners import PruningInputs, prune
    from evalscope_ext.pruners.core import TIER_NAMES, classify_tiers

    print(f"Loading response matrix from {reviews_dir} ...")
    sorted_idx, matrix, model_names, rep_meta = load_response_matrix(reviews_dir)
    n = len(sorted_idx)
    print(f"  {n} items × {len(model_names)} models: {model_names}")

    item_ids, index_to_hash = build_hash_mapping(sorted_idx, rep_meta)
    print(f"  Content hashes: {len(set(item_ids))} unique / {len(item_ids)} total")

    feature_table = build_feature_table(sorted_idx, rep_meta)
    print(f"  token_bin distribution: {dict(Counter(feature_table['token_bin']))}")

    tiers = classify_tiers(matrix)
    tier_counts = dict(Counter(TIER_NAMES[t] for t in tiers))
    print(f"  Tier breakdown: {tier_counts}")
    print()
    print(
        "  ⚠ Judge-noise caveat: AA-LCR acc is LLM-judged (single run).\n"
        "    Treat ±2–3% accuracy differences across strategies as within-noise.\n"
        "    The tier classification for borderline items may not be stable\n"
        "    across judge re-runs; prefer coarser comparisons.\n"
    )

    inputs = PruningInputs(
        item_ids=item_ids,
        response_matrix=matrix,
        feature_table=feature_table,
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
                "selected_hashes": result.selected_item_ids,
                "metadata": {
                    "strategy": strategy,
                    "prune_ratio": ratio,
                    "target_count": result.target_count,
                    "selected_count": result.selected_count,
                    "bucket_counts": result.bucket_counts,
                    "n_items_total": n,
                    "model_names": model_names,
                    "rng_seed": rng_seed,
                    "anchor_fraction": anchor_fraction,
                    "features_used": list(feature_table.keys()),
                    "judge_noise_caveat": (
                        "AA-LCR acc is LLM-judged (single run). "
                        "Tier assignments for borderline items may shift on re-evaluation."
                    ),
                },
            }
            fname = f"aa_lcr_{strategy}_r{int(ratio * 100):03d}.json"
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "w") as f:
                json.dump(out, f, indent=2)
            bc = result.bucket_counts
            print(
                f"  [{strategy:20s}  ratio={ratio:.2f}]  "
                f"selected={result.selected_count}/{n}  "
                f"split_hard={bc['split_hard']}  split_easy={bc['split_easy']}  "
                f"anchor_hard={bc['anchor_hard']}  anchor_easy={bc['anchor_easy']}  "
                f"→ {fpath}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "cache"),
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
        reviews_dir=args.reviews_dir,
        output_dir=args.output_dir,
        ratios=args.ratios,
        strategies=args.strategies,
        rng_seed=args.rng_seed,
        anchor_fraction=args.anchor_fraction,
    )


if __name__ == "__main__":
    main()
