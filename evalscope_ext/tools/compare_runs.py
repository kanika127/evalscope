"""Compare a full vs. pruned evalscope evaluation run.

Usage:
    python -m evalscope_ext.tools.compare_runs \\
        --full   ./results_full/ \\
        --pruned ./results_pruned/

Reads evalscope's per-(model, dataset) Report JSONs from both output
directories, joins them on (model_name, dataset_name), and reports the per-
benchmark score delta plus an item-count comparison. This is the third leg
of the Task 2 run contract (eval full → eval pruned → compare).

The output is a markdown-format table to stdout, plus an optional JSON dump
of the joined records for downstream tools.

evalscope writes Reports as JSON files with the shape produced by
`evalscope.report.report.Report.to_json()` — at minimum:

    {
      "dataset_name": "live_code_bench",
      "model_name":   "<model>",
      "score":        0.<float>,
      "num":          <int>,
      ...
    }

This walker is deliberately lenient: it picks up any JSON file whose
top-level keys include `dataset_name`, `model_name`, and `score`, so it
works against both the canonical Report path and any custom export path a
reviewer may have used.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    model_name: str
    dataset_name: str
    score: float
    num: Optional[int]
    source_path: str


def _walk_json_files(root: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".json"):
                yield os.path.join(dirpath, fn)


def _parse_record(path: str) -> Optional[RunRecord]:
    """Parse one JSON file into a RunRecord, if it looks like an evalscope
    Report. Returns None silently for unrelated JSON (e.g. config dumps)."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict):
        return None
    needed = ("dataset_name", "model_name", "score")
    if not all(k in d for k in needed):
        return None
    try:
        score = float(d["score"])
    except (TypeError, ValueError):
        return None
    num = d.get("num")
    if num is not None:
        try:
            num = int(num)
        except (TypeError, ValueError):
            num = None
    return RunRecord(
        model_name=str(d["model_name"]),
        dataset_name=str(d["dataset_name"]),
        score=score,
        num=num,
        source_path=path,
    )


def load_run_records(root: str) -> List[RunRecord]:
    out: List[RunRecord] = []
    for path in _walk_json_files(root):
        rec = _parse_record(path)
        if rec is not None:
            out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Join + aggregation
# ---------------------------------------------------------------------------


@dataclass
class PairedRecord:
    model_name: str
    dataset_name: str
    full_score: Optional[float]
    pruned_score: Optional[float]
    full_num: Optional[int]
    pruned_num: Optional[int]

    @property
    def score_delta(self) -> Optional[float]:
        if self.full_score is None or self.pruned_score is None:
            return None
        return self.pruned_score - self.full_score

    @property
    def num_reduction(self) -> Optional[float]:
        if self.full_num and self.pruned_num is not None and self.full_num > 0:
            return 1.0 - self.pruned_num / self.full_num
        return None


def pair_runs(
    full: List[RunRecord], pruned: List[RunRecord]
) -> List[PairedRecord]:
    """Join on (model_name, dataset_name). If a benchmark name was modified
    by the pruned-adapter registration (e.g. live_code_bench → live_code_bench_pruned),
    also accept a `<name>_pruned` match against `<name>`."""

    def _idx(records: List[RunRecord]) -> Dict[Tuple[str, str], RunRecord]:
        out: Dict[Tuple[str, str], RunRecord] = {}
        for r in records:
            out[(r.model_name, r.dataset_name)] = r
        return out

    full_idx = _idx(full)
    pruned_idx = _idx(pruned)

    def _canonical(name: str) -> str:
        return name[:-7] if name.endswith("_pruned") else name

    seen: set = set()
    paired: List[PairedRecord] = []
    for (model, dataset), full_rec in full_idx.items():
        canonical = _canonical(dataset)
        # Try exact match first, then "<dataset>_pruned"
        candidates = [
            (model, dataset),
            (model, f"{canonical}_pruned"),
        ]
        pruned_rec: Optional[RunRecord] = None
        for k in candidates:
            if k in pruned_idx and k not in seen:
                pruned_rec = pruned_idx[k]
                seen.add(k)
                break
        paired.append(
            PairedRecord(
                model_name=model,
                dataset_name=canonical,
                full_score=full_rec.score,
                pruned_score=pruned_rec.score if pruned_rec else None,
                full_num=full_rec.num,
                pruned_num=pruned_rec.num if pruned_rec else None,
            )
        )
    # Pruned-only records (no matching full)
    for (model, dataset), pruned_rec in pruned_idx.items():
        if (model, dataset) in seen:
            continue
        canonical = _canonical(dataset)
        if not any(p.model_name == model and p.dataset_name == canonical for p in paired):
            paired.append(
                PairedRecord(
                    model_name=model,
                    dataset_name=canonical,
                    full_score=None,
                    pruned_score=pruned_rec.score,
                    full_num=None,
                    pruned_num=pruned_rec.num,
                )
            )
    return sorted(paired, key=lambda p: (p.model_name, p.dataset_name))


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render_markdown(paired: List[PairedRecord]) -> str:
    if not paired:
        return "_No comparable (model, dataset) pairs found._"
    lines: List[str] = []
    lines.append("# Full vs. pruned evaluation comparison")
    lines.append("")
    lines.append(
        "| Model | Dataset | Full score | Pruned score | Δ score | "
        "Full N | Pruned N | Sample reduction |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for p in paired:
        fs = "—" if p.full_score is None else f"{p.full_score:.4f}"
        ps = "—" if p.pruned_score is None else f"{p.pruned_score:.4f}"
        delta = p.score_delta
        ds = "—" if delta is None else f"{delta:+.4f}"
        fn = "—" if p.full_num is None else str(p.full_num)
        pn = "—" if p.pruned_num is None else str(p.pruned_num)
        red = p.num_reduction
        rs = "—" if red is None else f"{red:.1%}"
        lines.append(
            f"| {p.model_name} | {p.dataset_name} | {fs} | {ps} | "
            f"{ds} | {fn} | {pn} | {rs} |"
        )
    return "\n".join(lines)


def paired_to_dicts(paired: List[PairedRecord]) -> List[Dict[str, Any]]:
    return [
        {
            "model_name": p.model_name,
            "dataset_name": p.dataset_name,
            "full_score": p.full_score,
            "pruned_score": p.pruned_score,
            "score_delta": p.score_delta,
            "full_num": p.full_num,
            "pruned_num": p.pruned_num,
            "sample_reduction": p.num_reduction,
        }
        for p in paired
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, help="Output dir of full eval run")
    parser.add_argument("--pruned", required=True, help="Output dir of pruned eval run")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to dump the paired records as JSON",
    )
    args = parser.parse_args()

    full = load_run_records(args.full)
    pruned = load_run_records(args.pruned)
    if not full:
        print(f"[warn] No Report JSONs discovered under {args.full!r}")
    if not pruned:
        print(f"[warn] No Report JSONs discovered under {args.pruned!r}")

    paired = pair_runs(full, pruned)
    print(render_markdown(paired))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(paired_to_dicts(paired), f, indent=2)
        print(f"\n[wrote {args.json_out}]")


if __name__ == "__main__":
    main()
