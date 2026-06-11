"""Precompute pruned LCB index sets from the 3 shipped review/prediction files.

Usage:
    python -m evalscope_ext.pruners.precompute_lcb \\
        --reviews-dir  <path/to/Evals/Part 1/reviews> \\
        --predictions-dir <path/to/Evals/Part 1/predictions> \\
        --key-file     /tmp/lcb_index_to_key.json \\
        --output-dir   evalscope_ext/pruners/cache \\
        --ratios       0.30 0.50 0.70 \\
        --strategies   hybrid random disagreement_only stratified_only

Outputs one JSON per (strategy, ratio):
    cache/lcb_<strategy>_r<ratio>.json   — {"selected_hashes": [...], "metadata": {...}}

The mapping from shipped-index to content_hash comes from --key-file (produced
by the LCB index-verification step). If --key-file is absent, the script
recomputes the hashes from the prediction prompts via tiktoken (Tier 1 path).

Review files expected (live_code_bench_v5__*.jsonl, any order):
    live_code_bench_v5__gpt-oss-120b.jsonl
    live_code_bench_v5__kimi-k2.5.jsonl
    live_code_bench_v5__minimax-m2.5.jsonl

Score field: sample_score.score.value['pass'] (1.0 = pass, 0.0 = fail).
Prediction file used only for feature extraction (input_tokens per item); one
model's prediction file is sufficient, as prompt lengths are the same for all.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Token-count bins (feature_b): coarse difficulty proxy from prompt length.
# Buckets chosen so each has ~25% of LCB problems at typical length range.
# ---------------------------------------------------------------------------
TOKEN_BIN_EDGES = [500, 700, 900]  # < 500 → 'xs', 500-699 → 'sm', 700-899 → 'md', >= 900 → 'lg'


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


def _load_review_file(path: str) -> Dict[int, int]:
    """Return {shipped_index: binary_pass} for one model's review file."""
    scores: Dict[int, int] = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            idx = obj["index"]
            raw = obj["sample_score"]["score"]["value"]["pass"]
            scores[idx] = 1 if float(raw) > 0.0 else 0
    return scores


def load_response_matrix(
    reviews_dir: str,
) -> Tuple[List[int], List[List[int]], List[str]]:
    """Load all 3 LCB review files and return (sorted_indices, response_matrix, model_names).

    response_matrix[i][j] = binary pass for item i, model j.
    Models are sorted by filename for determinism.
    """
    pattern = os.path.join(reviews_dir, "live_code_bench_v5__*.jsonl")
    files = sorted(glob.glob(pattern))
    if len(files) != 3:
        raise FileNotFoundError(
            f"Expected exactly 3 live_code_bench_v5__*.jsonl in {reviews_dir}, found {len(files)}: {files}"
        )
    model_names: List[str] = []
    per_model: List[Dict[int, int]] = []
    for fp in files:
        name = re.search(r"live_code_bench_v5__(.+)\.jsonl", fp)
        model_names.append(name.group(1) if name else fp)
        per_model.append(_load_review_file(fp))

    # intersect indices
    all_indices = set(per_model[0].keys())
    for m in per_model[1:]:
        all_indices &= set(m.keys())
    sorted_idx = sorted(all_indices)

    matrix: List[List[int]] = [
        [per_model[m][i] for m in range(len(per_model))] for i in sorted_idx
    ]
    return sorted_idx, matrix, model_names


# ---------------------------------------------------------------------------
# Load / compute content hashes
# ---------------------------------------------------------------------------


def load_key_mapping(key_file: str) -> Dict[str, str]:
    """Return {str(shipped_index): content_hash} from the precomputed key file."""
    with open(key_file) as f:
        d = json.load(f)
    return d["mapping"]  # keys are str(int)


def compute_key_mapping_from_predictions(
    predictions_dir: str,
) -> Dict[str, str]:
    """Fallback: decode prompt_token_ids via tiktoken to extract question_content
    hashes. Uses the gpt-oss-120b predictions file (any model's would give the
    same prompt content for the same shipped index)."""
    try:
        import tiktoken  # type: ignore
    except ImportError:
        raise ImportError(
            "tiktoken is required when --key-file is not provided. "
            "Install with: pip install tiktoken"
        )
    pattern = os.path.join(predictions_dir, "live_code_bench_v5__gpt-oss-120b.jsonl")
    if not os.path.exists(pattern):
        # Try any predictions file
        candidates = glob.glob(os.path.join(predictions_dir, "live_code_bench_v5__*.jsonl"))
        if not candidates:
            raise FileNotFoundError(
                f"No live_code_bench_v5__ predictions found in {predictions_dir}"
            )
        pattern = sorted(candidates)[0]

    enc = tiktoken.get_encoding("o200k_harmony")

    def _extract_qc(decoded: str) -> Optional[str]:
        m = re.search(r"### Question:\n(.*?)\n\n### Format:", decoded, re.DOTALL)
        return m.group(1) if m else None

    mapping: Dict[str, str] = {}
    with open(pattern) as f:
        for line in f:
            obj = json.loads(line)
            idx = obj["index"]
            token_ids = obj["model_output"]["prompt_token_ids"]
            text = enc.decode(token_ids)
            qc = _extract_qc(text)
            if qc is None:
                raise ValueError(
                    f"Could not extract question_content for shipped index {idx}"
                )
            h = hashlib.sha256(qc.strip().encode("utf-8")).hexdigest()
            mapping[str(idx)] = h
    return mapping


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------


def build_feature_table(
    sorted_indices: List[int],
    predictions_dir: str,
) -> Optional[Dict[str, List[str]]]:
    """Extract per-item features from the predictions.

    Returns a feature_table with:
        'token_bin': input-token count binned into xs/sm/md/lg
    Returns None if no predictions file is found (feature table is optional).
    """
    candidates = sorted(
        glob.glob(os.path.join(predictions_dir, "live_code_bench_v5__*.jsonl"))
    )
    if not candidates:
        return None

    # Build lookup idx -> input_tokens from one predictions file
    token_lookup: Dict[int, int] = {}
    with open(candidates[0]) as f:
        for line in f:
            obj = json.loads(line)
            usage = obj.get("model_output", {}).get("usage") or {}
            n_tok = usage.get("input_tokens") or len(
                obj.get("model_output", {}).get("prompt_token_ids", [])
            )
            token_lookup[obj["index"]] = n_tok

    token_bins = [bin_tokens(token_lookup.get(i, 0)) for i in sorted_indices]
    return {"token_bin": token_bins}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    reviews_dir: str,
    predictions_dir: str,
    key_file: Optional[str],
    output_dir: str,
    ratios: List[float],
    strategies: List[str],
    rng_seed: int = 0,
    anchor_fraction: float = 0.15,
) -> None:
    from evalscope_ext.pruners import PruningInputs, prune

    print(f"Loading response matrix from {reviews_dir} ...")
    sorted_idx, matrix, model_names = load_response_matrix(reviews_dir)
    n = len(sorted_idx)
    print(f"  {n} items × {len(model_names)} models: {model_names}")

    # Build content-hash mapping
    if key_file and os.path.exists(key_file):
        print(f"Loading content-hash mapping from {key_file} ...")
        mapping = load_key_mapping(key_file)
    else:
        print(f"Recomputing content-hash mapping from predictions in {predictions_dir} ...")
        mapping = compute_key_mapping_from_predictions(predictions_dir)
        print(f"  Computed {len(mapping)} hashes")

    item_ids = [mapping[str(i)] for i in sorted_idx]

    # Feature table
    feature_table = build_feature_table(sorted_idx, predictions_dir)
    if feature_table:
        bins = feature_table["token_bin"]
        from collections import Counter
        print(f"  token_bin distribution: {dict(Counter(bins))}")
    else:
        print("  No feature table (predictions dir not found or empty)")

    inputs = PruningInputs(
        item_ids=item_ids,
        response_matrix=matrix,
        feature_table=feature_table,
    )

    # Diagnostics: tier breakdown
    from evalscope_ext.pruners.core import classify_tiers, TIER_NAMES
    from collections import Counter
    tiers = classify_tiers(matrix)
    tier_counts = dict(Counter(TIER_NAMES[t] for t in tiers))
    print(f"  Tier breakdown: {tier_counts}")

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
                    "features_used": list(feature_table.keys()) if feature_table else [],
                },
            }
            fname = f"lcb_{strategy}_r{int(ratio * 100):03d}.json"
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "w") as f:
                json.dump(out, f, indent=2)
            print(
                f"  [{strategy:20s}  ratio={ratio:.2f}]  "
                f"selected={result.selected_count}/{n}  "
                f"→ {fpath}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews-dir", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--key-file", default="/tmp/lcb_index_to_key.json")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "cache"),
    )
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.30, 0.50, 0.70])
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
        predictions_dir=args.predictions_dir,
        key_file=args.key_file,
        output_dir=args.output_dir,
        ratios=args.ratios,
        strategies=args.strategies,
        rng_seed=args.rng_seed,
        anchor_fraction=args.anchor_fraction,
    )


if __name__ == "__main__":
    main()
