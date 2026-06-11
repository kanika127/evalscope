"""Empirical validation of the pruning method on shipped review data.

For each (benchmark, strategy, ratio):
    For each LOMO split (holdout ∈ {model_0, model_1, model_2}):
        For each seed ∈ {0..9}:
            1. Run prune() using the OTHER 2 models' scores as response_matrix
            2. Compute each of the 3 models' accuracy on the SELECTED items
            3. Compute Kendall τ_b and Spearman ρ between (pruned_acc, full_acc)
            4. Record held-out model's rank delta (full_rank − pruned_rank)

Aggregate across the 10 × 3 = 30 trials per cell:
    tau_mean ± std, fraction_perfect (τ = 1.0),
    HELD-OUT-ONLY rank preservation rate (headline C2 metric),
    all-3 rank preservation rate (secondary).

Outputs:
    evalscope_ext/validation/results.json — full per-trial detail
    evalscope_ext/validation/summary.md  — handout-ready table
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from evalscope_ext.pruners import PruningInputs, prune
from evalscope_ext.validation.metrics import (
    descending_ranks,
    kendall_tau_b,
    spearman_rho,
)

# ---------------------------------------------------------------------------
# Benchmark loaders
# ---------------------------------------------------------------------------


def load_lcb(reviews_dir: str) -> Tuple[List[str], List[List[int]], List[str]]:
    """Read 3 LCB review files → (item_ids, N×3 binary matrix, model_names).

    item_id used here is the shipped index as string. (LCB's content-hash key
    isn't needed for this analysis — we just need a stable per-item label and
    the binary scores; the 3 review files agree on `index`.)
    """
    files = sorted(glob.glob(os.path.join(reviews_dir, "live_code_bench_v5__*.jsonl")))
    if len(files) != 3:
        raise FileNotFoundError(
            f"Expected 3 LCB review files in {reviews_dir!r}, found {files}"
        )
    return _load_three_model(files, "live_code_bench_v5__(.+)\\.jsonl", ["pass"])


def load_aa_lcr(reviews_dir: str) -> Tuple[List[str], List[List[int]], List[str]]:
    files = sorted(glob.glob(os.path.join(reviews_dir, "aa_lcr__*.jsonl")))
    if len(files) != 3:
        raise FileNotFoundError(
            f"Expected 3 AA-LCR review files in {reviews_dir!r}, found {files}"
        )
    return _load_three_model(files, "aa_lcr__(.+)\\.jsonl", ["acc"])


def _load_three_model(
    files: Sequence[str], name_re: str, score_path: Sequence[str]
) -> Tuple[List[str], List[List[int]], List[str]]:
    model_names: List[str] = []
    per_model: List[Dict[str, int]] = []
    for fp in files:
        m = re.search(name_re, fp)
        model_names.append(m.group(1) if m else os.path.basename(fp))
        scores: Dict[str, int] = {}
        with open(fp) as fh:
            for line in fh:
                if not line.strip():
                    continue
                o = json.loads(line)
                v = o["sample_score"]["score"]["value"]
                for k in score_path:
                    v = v[k]
                scores[str(o["index"])] = 1 if float(v) >= 0.5 else 0
        per_model.append(scores)
    # Intersect indices (should always be the full N for the shipped data)
    common = sorted(set(per_model[0]).intersection(*per_model[1:]), key=int)
    matrix = [[per_model[m][i] for m in range(3)] for i in common]
    return common, matrix, model_names


# ---------------------------------------------------------------------------
# Per-trial computation
# ---------------------------------------------------------------------------


def column_accuracy(matrix: Sequence[Sequence[int]], col: int) -> float:
    n = len(matrix)
    if n == 0:
        return 0.0
    return sum(row[col] for row in matrix) / n


def column_accuracy_on_indices(
    matrix: Sequence[Sequence[int]], indices: Sequence[int], col: int
) -> float:
    if not indices:
        return 0.0
    return sum(matrix[i][col] for i in indices) / len(indices)


def lomo_response_matrix(
    matrix: Sequence[Sequence[int]], holdout: int
) -> List[List[int]]:
    keep_cols = [c for c in range(len(matrix[0])) if c != holdout]
    return [[row[c] for c in keep_cols] for row in matrix]


def run_one_trial(
    item_ids: Sequence[str],
    matrix_full: Sequence[Sequence[int]],
    holdout: int,
    strategy: str,
    ratio: float,
    seed: int,
    full_acc: Sequence[float],
    full_ranks: Sequence[int],
) -> Dict[str, Any]:
    """One trial = one (strategy, ratio, holdout, seed) tuple."""
    matrix_train = lomo_response_matrix(matrix_full, holdout)
    inputs = PruningInputs(
        item_ids=list(item_ids),
        response_matrix=matrix_train,
    )
    result = prune(inputs, prune_ratio=ratio, strategy=strategy, rng_seed=seed)
    selected_set = set(result.selected_item_ids)
    sel_indices = [i for i, id_ in enumerate(item_ids) if id_ in selected_set]
    pruned_acc = [
        column_accuracy_on_indices(matrix_full, sel_indices, c) for c in range(3)
    ]
    pruned_ranks = descending_ranks(pruned_acc)
    tau = kendall_tau_b(full_acc, pruned_acc)
    rho = spearman_rho(full_acc, pruned_acc)
    held_full_rank = full_ranks[holdout]
    held_pruned_rank = pruned_ranks[holdout]
    held_delta = held_full_rank - held_pruned_rank
    all3_preserved = pruned_ranks == list(full_ranks)
    return {
        "strategy": strategy,
        "ratio": ratio,
        "holdout": holdout,
        "seed": seed,
        "selected_count": len(sel_indices),
        "pruned_acc": pruned_acc,
        "pruned_ranks": pruned_ranks,
        "held_full_rank": held_full_rank,
        "held_pruned_rank": held_pruned_rank,
        "held_delta": held_delta,
        "held_preserved": held_delta == 0,
        "all3_preserved": all3_preserved,
        "tau_b": tau,
        "rho": rho,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(trials: List[Dict[str, Any]]) -> Dict[Tuple[str, str, float], Dict[str, Any]]:
    by_cell: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    for t in trials:
        by_cell[(t["benchmark"], t["strategy"], t["ratio"])].append(t)

    def _stat_fields(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        taus = [r["tau_b"] for r in rows]
        rhos = [r["rho"] for r in rows]
        deltas = [r["held_delta"] for r in rows]
        held = [r["held_preserved"] for r in rows]
        all3 = [r["all3_preserved"] for r in rows]
        n_trials = len(rows)
        return {
            "n_trials": n_trials,
            "tau_mean": statistics.mean(taus),
            "tau_std": statistics.stdev(taus) if n_trials > 1 else 0.0,
            "rho_mean": statistics.mean(rhos),
            "rho_std": statistics.stdev(rhos) if n_trials > 1 else 0.0,
            "fraction_tau_perfect": sum(1 for t in taus if abs(t - 1.0) < 1e-9) / n_trials,
            "held_out_preservation_rate": sum(held) / n_trials,
            "all3_preservation_rate": sum(all3) / n_trials,
            "held_delta_mean": statistics.mean(deltas),
            "held_delta_abs_mean": statistics.mean(abs(d) for d in deltas),
        }

    out: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
    for cell, rows in by_cell.items():
        out[cell] = _stat_fields(rows)
    return out


def smallest_sufficient_ratio(
    aggregated: Dict[Tuple[str, str, float], Dict[str, Any]],
    benchmark: str,
    strategy: str,
    threshold: float,
    metric_key: str,
) -> Tuple[float, float] | Tuple[None, None]:
    """Smallest ratio where aggregated[cell][metric_key] >= threshold."""
    cells = sorted(
        (cell, stats)
        for cell, stats in aggregated.items()
        if cell[0] == benchmark and cell[1] == strategy
    )
    for (b, s, r), stats in cells:
        if stats[metric_key] >= threshold:
            return r, stats[metric_key]
    return None, None


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _two_proportion_p(p1: float, p2: float, n1: int, n2: int) -> float:
    """Two-sided two-proportion z-test p-value, normal approximation. Stdlib."""
    if n1 == 0 or n2 == 0:
        return 1.0
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    cdf = 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))
    return 2 * (1 - cdf)


def pairwise_significance_table(
    bench_meta: Dict[str, Any],
) -> List[Tuple[str, str, float, float, float]]:
    """Returns list of (model_a, model_b, gap_pp, z, p) for each model pair."""
    names = bench_meta["model_names"]
    accs = bench_meta["full_acc"]
    n = bench_meta["n_items"]
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p = _two_proportion_p(accs[i], accs[j], n, n)
            out.append(
                (
                    names[i],
                    names[j],
                    (accs[i] - accs[j]) * 100,
                    abs((accs[i] - accs[j]) / max(math.sqrt(accs[i] * (1 - accs[i]) / n), 1e-9)),
                    p,
                )
            )
    return out


def per_holdout_rates(
    trials: List[Dict[str, Any]],
    benchmark: str,
    strategy: str,
    ratio: float,
) -> List[float]:
    """Returns [rate_holdout_0, rate_holdout_1, rate_holdout_2]."""
    by_ho: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    for t in trials:
        if t["benchmark"] != benchmark or t["strategy"] != strategy or t["ratio"] != ratio:
            continue
        by_ho[t["holdout"]].append(int(t["held_preserved"]))
    return [sum(v) / len(v) if v else float("nan") for v in by_ho.values()]


def format_summary_md(
    benchmarks_meta: Dict[str, Dict[str, Any]],
    aggregated: Dict[Tuple[str, str, float], Dict[str, Any]],
    strategies: Sequence[str],
    ratios_per_benchmark: Dict[str, Sequence[float]],
    seeds: Sequence[int],
    trials: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Validation summary — pruning preserves model rankings\n")
    lines.append(
        "Empirical test of the pruning method on shipped reference scores. "
        f"{len(seeds)} seeds × 3 leave-one-model-out splits = "
        f"{len(seeds)*3} trials per (benchmark, strategy, ratio) cell.\n"
    )
    lines.append("## Coarseness disclaimer (read first)\n")
    lines.append(
        "With only 3 reference models per benchmark, the model ranking is over "
        "3 items. Kendall τ_b on 3 distinct values can take only 4 values: "
        "`{-1.0, -0.333, +0.333, +1.0}`. A uniform-random ranking lands at "
        "τ=+1.0 with probability 1/6 ≈ 17%. We mitigate by running "
        f"{len(seeds)*3} trials per cell and reporting τ-distribution + "
        "**held-out-model rank preservation rate** (the C2 claim) "
        "alongside τ. Read held-out-preservation as the headline; τ_mean as "
        "secondary.\n"
    )
    lines.append(
        "AA-LCR carries an additional ±2–3% LLM-judge-noise band per item. "
        "Treat AA-LCR cell differences < ~0.07 in any rate as within-noise.\n"
    )

    for bench, meta in benchmarks_meta.items():
        lines.append(f"\n## {bench}\n")
        lines.append(
            f"**N items:** {meta['n_items']} · **Models:** "
            f"{', '.join(meta['model_names'])} · **Full-benchmark "
            f"accuracy and rank (rank 1 = best):**\n"
        )
        lines.append("")
        lines.append("| Model | Full acc | Full rank |")
        lines.append("|---|---:|---:|")
        for name, acc, r in zip(meta["model_names"], meta["full_acc"], meta["full_ranks"]):
            lines.append(f"| {name} | {acc:.3f} | {r} |")

        # Pairwise significance — vital for interpreting preservation rates
        lines.append("")
        lines.append("**Pairwise full-benchmark gaps and significance (two-proportion z-test):**\n")
        lines.append("| pair | Δ acc | p-value | distinguishable @ α=0.05? |")
        lines.append("|---|---:|---:|---|")
        for a, b, gap, _z, p in pairwise_significance_table(meta):
            verdict = "yes" if p < 0.05 else "**no** (within noise)"
            lines.append(f"| {a} vs {b} | {gap:+.1f} pp | {p:.3f} | {verdict} |")
        lines.append("")
        lines.append(
            "**Interpretation guard.** A pruned-set ranking that fails to put a "
            "within-noise pair in the same order as the full benchmark hasn't "
            "really 'failed' — the full benchmark itself doesn't distinguish "
            "those two models. The held-out preservation rate below blends "
            "distinguishable and noise-pair cases; the **per-holdout breakdown** "
            "two tables down separates them.\n"
        )

        lines.append("")
        lines.append("### Held-out-model rank preservation rate (HEADLINE — C2 claim)")
        lines.append("")
        lines.append("Fraction of 30 trials in which the held-out model's "
                     "rank position survived (it was never used during selection).")
        lines.append("")
        ratios = ratios_per_benchmark[bench]
        header = "| ratio | " + " | ".join(strategies) + " |"
        sep = "|---|" + "|".join("---:" for _ in strategies) + "|"
        lines.append(header)
        lines.append(sep)
        for ratio in ratios:
            row = [f"{ratio:.2f}"]
            for s in strategies:
                cell = aggregated.get((bench, s, ratio), {})
                val = cell.get("held_out_preservation_rate")
                if val is None:
                    row.append("—")
                else:
                    row.append(f"{val:.2f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        lines.append("### Held-out preservation per held-out model (diagnostic)")
        lines.append("")
        lines.append(
            "Same headline metric, split by which model was held out (10 trials each). "
            "Reveals whether failure modes concentrate on the within-noise pair vs "
            "the distinguishable model."
        )
        lines.append("")
        ho_names = meta["model_names"]
        for strategy in strategies:
            lines.append(f"**{strategy}** (rate per held-out model):")
            lines.append("")
            lines.append("| ratio | " + " | ".join(ho_names) + " |")
            lines.append("|---|" + "|".join("---:" for _ in ho_names) + "|")
            for ratio in ratios_per_benchmark[bench]:
                rates = per_holdout_rates(trials, bench, strategy, ratio)
                cells = [f"{r:.2f}" if not math.isnan(r) else "—" for r in rates]
                lines.append(f"| {ratio:.2f} | " + " | ".join(cells) + " |")
            lines.append("")

        lines.append("### All-3-model rank preservation rate (secondary)")
        lines.append("")
        lines.append("Fraction of trials where the full 3-model ranking was preserved exactly. "
                     "Includes the 2 'seen' models, so partly circular.")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for ratio in ratios:
            row = [f"{ratio:.2f}"]
            for s in strategies:
                cell = aggregated.get((bench, s, ratio), {})
                val = cell.get("all3_preservation_rate")
                row.append("—" if val is None else f"{val:.2f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        lines.append("### Kendall τ_b — mean ± std (tertiary)")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for ratio in ratios:
            row = [f"{ratio:.2f}"]
            for s in strategies:
                cell = aggregated.get((bench, s, ratio), {})
                if not cell:
                    row.append("—")
                else:
                    row.append(f"{cell['tau_mean']:+.3f} ± {cell['tau_std']:.3f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # Smallest sufficient ratio (per strategy)
        lines.append("### Smallest sufficient ratio")
        lines.append("")
        lines.append("Smallest ratio at which the held-out-preservation rate "
                     "reaches each threshold. `—` means no tested ratio reached it.")
        lines.append("")
        lines.append("| strategy | rate ≥ 0.80 | rate ≥ 0.90 | rate = 1.00 |")
        lines.append("|---|---:|---:|---:|")
        for s in strategies:
            row = [s]
            for thr in (0.80, 0.90, 1.0):
                r, v = smallest_sufficient_ratio(
                    aggregated, bench, s, thr, "held_out_preservation_rate"
                )
                if r is None:
                    row.append("—")
                else:
                    row.append(f"{r:.2f} (rate {v:.2f})")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("\n---\n")
    lines.append("## Reading guide\n")
    lines.append("- **Held-out preservation rate** is the empirical answer to "
                 "*'is this defensible for a 4th model the pruner never saw?'*. "
                 "Higher is better; 1.00 is full preservation across all 30 trials.")
    lines.append("- **All-3 preservation rate** includes the 2 training models, "
                 "so it's partly circular and shown for completeness.")
    lines.append("- **Kendall τ_b** is the standard rank-correlation metric. "
                 "On 3 items it's coarse — read distribution shape, not point values.")
    lines.append("- **Hybrid vs random** on the held-out column is the direct "
                 "C3 check. Random hovers near its expected baseline; hybrid "
                 "should sit clearly above. If it doesn't, the report says so.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


DEFAULT_REVIEWS_DIR = (
    "/Users/kanika/Documents/Job Search/Cerabras AI Engg Model_Qual_Perf/"
    "ai-model-quality-challenge/Evals/Part 1/reviews"
)


def run(
    reviews_dir: str = DEFAULT_REVIEWS_DIR,
    output_dir: str = str(Path(__file__).parent),
    seeds: Sequence[int] = tuple(range(10)),
    strategies: Sequence[str] = ("hybrid", "random", "disagreement_only", "stratified_only"),
    ratios_lcb: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70),
    ratios_aa_lcr: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.50),
) -> None:
    benchmarks: Dict[str, Dict[str, Any]] = {}

    # LCB
    lcb_ids, lcb_matrix, lcb_models = load_lcb(reviews_dir)
    lcb_full_acc = [column_accuracy(lcb_matrix, c) for c in range(3)]
    lcb_full_ranks = descending_ranks(lcb_full_acc)
    benchmarks["lcb"] = {
        "model_names": lcb_models,
        "n_items": len(lcb_ids),
        "full_acc": lcb_full_acc,
        "full_ranks": lcb_full_ranks,
    }

    # AA-LCR
    aa_ids, aa_matrix, aa_models = load_aa_lcr(reviews_dir)
    aa_full_acc = [column_accuracy(aa_matrix, c) for c in range(3)]
    aa_full_ranks = descending_ranks(aa_full_acc)
    benchmarks["aa_lcr"] = {
        "model_names": aa_models,
        "n_items": len(aa_ids),
        "full_acc": aa_full_acc,
        "full_ranks": aa_full_ranks,
    }

    ratios_per_benchmark = {"lcb": ratios_lcb, "aa_lcr": ratios_aa_lcr}

    # Drive trials
    trials: List[Dict[str, Any]] = []
    for bench_name, bench_data in [
        ("lcb", (lcb_ids, lcb_matrix, lcb_full_acc, lcb_full_ranks)),
        ("aa_lcr", (aa_ids, aa_matrix, aa_full_acc, aa_full_ranks)),
    ]:
        item_ids, matrix, full_acc, full_ranks = bench_data
        for holdout in range(3):
            for strategy in strategies:
                for ratio in ratios_per_benchmark[bench_name]:
                    for seed in seeds:
                        t = run_one_trial(
                            item_ids, matrix, holdout, strategy, ratio,
                            seed, full_acc, full_ranks,
                        )
                        t["benchmark"] = bench_name
                        t["holdout_model"] = benchmarks[bench_name]["model_names"][holdout]
                        trials.append(t)

    # Aggregate
    aggregated = aggregate(trials)

    # Write per-trial JSON (raw)
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "benchmarks": benchmarks,
                "strategies": list(strategies),
                "seeds": list(seeds),
                "ratios_per_benchmark": {k: list(v) for k, v in ratios_per_benchmark.items()},
                "aggregated": [
                    {"benchmark": b, "strategy": s, "ratio": r, **stats}
                    for (b, s, r), stats in aggregated.items()
                ],
                "trials": trials,
            },
            f,
            indent=2,
        )
    print(f"[validation] wrote {results_path} ({len(trials)} trials)")

    # Write markdown summary
    md = format_summary_md(
        benchmarks, aggregated, strategies, ratios_per_benchmark, seeds, trials
    )
    summary_path = os.path.join(output_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write(md)
    print(f"[validation] wrote {summary_path}")

    # Headline-print to stdout
    print()
    print("=" * 72)
    print("HEADLINE — held-out-model rank preservation rate (30 trials per cell)")
    print("=" * 72)
    for bench in ("lcb", "aa_lcr"):
        print(f"\n{bench} — full ranking:")
        for nm, acc, rk in zip(
            benchmarks[bench]["model_names"],
            benchmarks[bench]["full_acc"],
            benchmarks[bench]["full_ranks"],
        ):
            print(f"  {nm:30s} acc={acc:.3f}  rank={rk}")
        print()
        ratios = ratios_per_benchmark[bench]
        # Per-strategy table
        col_w = max(20, max(len(s) for s in strategies))
        print(f"  {'ratio':>6s} | " + " | ".join(s.rjust(col_w) for s in strategies))
        print(f"  {'-'*6} | " + " | ".join("-"*col_w for _ in strategies))
        for ratio in ratios:
            row = [f"{ratio:6.2f}"]
            for s in strategies:
                v = aggregated.get((bench, s, ratio), {}).get("held_out_preservation_rate")
                row.append((f"{v:.2f}" if v is not None else "—").rjust(col_w))
            print("  " + " | ".join(row))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews-dir", default=DEFAULT_REVIEWS_DIR)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    run(reviews_dir=args.reviews_dir, output_dir=args.output_dir, seeds=args.seeds)


if __name__ == "__main__":
    main()
