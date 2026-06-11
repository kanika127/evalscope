"""Encoder-quality probe — triple-query wrapper over OpenAI Chat Completions.

For each probe item, run up to three queries and capture the answer-token
logprob in each:

    Q1 — full         text + image(s)
    Q2 — text-only    same text, image attachments replaced by [IMAGE WITHHELD]
    Q3 — perturbed    (optional) text + image downsampled to 56×56 then re-upsampled

Aggregate per-stratum:

    encoder_lift = acc(Q1, stratum) − acc(Q2, stratum)

A degraded encoder shrinks `encoder_lift` on high-stress strata while
preserving it on low-stress strata (the negative-control bin written by
precompute_mmmu.py at tier 0). The differential is the probe signal.

This module is END-TO-END runnable against any OpenAI-compatible URL+key
(OpenAI, Azure, vLLM, Together, Anyscale, Cerebras's own inference endpoint,
LM Studio…). We do not execute it in this session; we ship it as importable,
unit-tested code that a reviewer can point at an endpoint via:

    OPENAI_API_KEY=... OPENAI_BASE_URL=... \\
        python -m evalscope_ext.probes.encoder_probe \\
            --index-file evalscope_ext/pruners/cache/mmmu_hybrid_r030.json \\
            --model <vlm_model_name> \\
            --hf-source MMMU/MMMU --hf-split validation \\
            --output-dir ./probe_results/

Three things are tested at unit-test time:
    1. Prompt construction (image attachment present in Q1, withheld in Q2).
    2. Answer-token logprob + top-1/top-2 margin extraction from a recorded
       chat-completion fixture.
    3. Stratum-level aggregation (encoder_lift, n_full, n_text_only).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ANSWER_LETTERS = ("A", "B", "C", "D", "E", "F")
IMAGE_WITHHELD_TOKEN = "[IMAGE WITHHELD]"


# ---------------------------------------------------------------------------
# Probe data structures
# ---------------------------------------------------------------------------


@dataclass
class ProbeItem:
    """A single MMMU question prepared for the probe."""
    id: str                              # upstream MMMU id
    question: str                        # raw question text (with <image N> placeholders)
    options: Sequence[str]               # list of answer options for MC, [] for open
    correct_answer: str                  # ground-truth answer letter or text
    question_type: str                   # 'multiple-choice' or 'open'
    images_b64: Sequence[str]            # list of base64-encoded image data URIs (data:image/png;base64,...)
    stratum: Dict[str, Any]              # subfield, img_type_bucket, stress_tier, etc.


@dataclass
class QueryResult:
    """Result of one OpenAI Chat Completion call for one probe item."""
    item_id: str
    variant: str                          # 'full' | 'text_only' | 'perturbed'
    extracted_answer: Optional[str]
    correct: bool
    answer_logprob: Optional[float]       # logprob of the chosen answer token
    answer_top1_top2_margin: Optional[float]
    raw_response_text: str = ""


@dataclass
class ProbeOutcome:
    """All variants' results for one probe item."""
    item_id: str
    stratum: Dict[str, Any]
    results_by_variant: Dict[str, QueryResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_mc_prompt_text(question: str, options: Sequence[str]) -> str:
    """Construct the text portion of an MC question."""
    parts = [
        "Answer the following multiple-choice question. Respond with only the "
        "letter of the correct option (A, B, C, …) on the last line, "
        'prefixed by "ANSWER: ".\n',
        f"Question: {question.strip()}",
        "",
        "Options:",
    ]
    for letter, opt in zip(ANSWER_LETTERS, options):
        parts.append(f"  {letter}. {opt}")
    parts.append("")
    parts.append("Provide your reasoning briefly, then end with the line:")
    parts.append("ANSWER: <letter>")
    return "\n".join(parts)


def build_open_prompt_text(question: str) -> str:
    return (
        "Answer the following question. End your response with a line:\n"
        "ANSWER: <answer>\n\n"
        f"Question: {question.strip()}\n"
    )


def _replace_image_placeholders(text: str, replacement: str) -> str:
    """Replace `<image N>` placeholders so the text-only variant still flows."""
    return re.sub(r"<image\s*\d+>", replacement, text)


def build_messages_full(item: ProbeItem) -> List[Dict[str, Any]]:
    """Build the OpenAI-style messages list for the FULL variant.

    Returns one user message with mixed text + image content parts."""
    if item.question_type == "multiple-choice":
        text = build_mc_prompt_text(item.question, item.options)
    else:
        text = build_open_prompt_text(item.question)
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for img_url in item.images_b64:
        content.append({"type": "image_url", "image_url": {"url": img_url}})
    return [{"role": "user", "content": content}]


def build_messages_text_only(item: ProbeItem) -> List[Dict[str, Any]]:
    """Build the messages list for the TEXT-ONLY variant.

    Image attachments are removed; `<image N>` placeholders in the question
    are replaced by an explicit [IMAGE WITHHELD] marker so the model knows
    something is missing rather than seeing a dangling reference."""
    if item.question_type == "multiple-choice":
        text = build_mc_prompt_text(
            _replace_image_placeholders(item.question, IMAGE_WITHHELD_TOKEN),
            item.options,
        )
    else:
        text = build_open_prompt_text(
            _replace_image_placeholders(item.question, IMAGE_WITHHELD_TOKEN),
        )
    return [{"role": "user", "content": text}]


def _perturb_data_uri(data_uri: str, target_size: int = 56) -> str:
    """Downsample an image to target_size, then re-upsample to its original
    size — a controlled perturbation that erases fine detail while preserving
    aspect/overall layout.

    Falls back to returning the original URI if PIL is not installed.
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return data_uri
    if not data_uri.startswith("data:"):
        return data_uri
    header, _, b64 = data_uri.partition(",")
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    w, h = img.size
    img_small = img.resize((target_size, target_size), Image.BILINEAR)
    img_back = img_small.resize((w, h), Image.BILINEAR)
    buf = io.BytesIO()
    img_back.save(buf, format="PNG")
    new_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{new_b64}"


def build_messages_perturbed(item: ProbeItem) -> List[Dict[str, Any]]:
    """Same as full, with each image perturbed."""
    perturbed = ProbeItem(
        id=item.id,
        question=item.question,
        options=item.options,
        correct_answer=item.correct_answer,
        question_type=item.question_type,
        images_b64=[_perturb_data_uri(u) for u in item.images_b64],
        stratum=item.stratum,
    )
    return build_messages_full(perturbed)


# ---------------------------------------------------------------------------
# Answer extraction + logprob lookup
# ---------------------------------------------------------------------------


_ANSWER_LINE_RE = re.compile(r"ANSWER:\s*([^\n]+)", re.IGNORECASE)


def extract_answer(response_text: str, question_type: str) -> Optional[str]:
    """Pull the model's final answer from an `ANSWER: X` line.

    For MC we accept the first letter only; for open we keep the full string."""
    m = _ANSWER_LINE_RE.findall(response_text)
    if not m:
        return None
    raw = m[-1].strip()
    if question_type == "multiple-choice":
        # First A-F letter in the captured text
        m2 = re.search(r"[A-Fa-f]", raw)
        return m2.group(0).upper() if m2 else None
    return raw


def find_answer_token_logprob(
    response_text: str,
    logprobs_content: Sequence[Dict[str, Any]],
    question_type: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Locate the answer token in the logprobs stream and return
    (chosen_logprob, top1_top2_margin).

    For MC: walk the token stream looking for the LAST single A/B/C/D/E/F
    token that appears after an "ANSWER" marker. Use the logprob of that
    token, and the gap between its top alternative and the runner-up.

    For open: return mean over the response (a weaker proxy).
    """
    if not logprobs_content:
        return None, None
    if question_type == "multiple-choice":
        # Find the index of "ANSWER" token, then the first single-letter token after it
        answer_idx = None
        for i, tok in enumerate(logprobs_content):
            if tok.get("token", "").strip().upper().startswith("ANSWER"):
                answer_idx = i
        if answer_idx is None:
            return None, None
        for tok in logprobs_content[answer_idx + 1 :]:
            t = tok.get("token", "").strip()
            if len(t) == 1 and t.upper() in ANSWER_LETTERS:
                top = tok.get("top_logprobs") or []
                lp = tok.get("logprob")
                margin = None
                if len(top) >= 2 and top[0].get("logprob") is not None and top[1].get("logprob") is not None:
                    margin = top[0]["logprob"] - top[1]["logprob"]
                return lp, margin
        return None, None
    # open
    lps = [t.get("logprob") for t in logprobs_content if t.get("logprob") is not None]
    if not lps:
        return None, None
    return sum(lps) / len(lps), None


# ---------------------------------------------------------------------------
# OpenAI client wrapper (injectable for tests)
# ---------------------------------------------------------------------------


# Type alias for any callable that takes (model, messages, **kwargs) and returns a
# dict with the same shape as OpenAI's chat.completions.create() response.
ClientFn = Callable[..., Dict[str, Any]]


def default_openai_client() -> ClientFn:
    """Return a wrapper around openai.OpenAI().chat.completions.create."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise ImportError(
            "`openai` package required for live probe runs. "
            "Install with: pip install openai"
        ) from e

    client = OpenAI()  # respects OPENAI_API_KEY + OPENAI_BASE_URL env vars

    def _call(
        model: str,
        messages: Sequence[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_logprobs: int = 5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        resp = client.chat.completions.create(
            model=model,
            messages=list(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            **kwargs,
        )
        # Normalize to a plain dict for downstream code + fixture replay
        return resp.model_dump()

    return _call


# ---------------------------------------------------------------------------
# Triple-query driver
# ---------------------------------------------------------------------------


def _run_one(
    client: ClientFn,
    model: str,
    item: ProbeItem,
    variant: str,
) -> QueryResult:
    """Send one variant through the client; parse response."""
    if variant == "full":
        messages = build_messages_full(item)
    elif variant == "text_only":
        messages = build_messages_text_only(item)
    elif variant == "perturbed":
        messages = build_messages_perturbed(item)
    else:
        raise ValueError(f"unknown variant {variant!r}")

    resp = client(model=model, messages=messages)
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    logprobs_obj = choice.get("logprobs") or {}
    lp_content = logprobs_obj.get("content") or []

    extracted = extract_answer(text, item.question_type)
    correct = bool(extracted and extracted.strip().upper() == item.correct_answer.strip().upper())
    lp, margin = find_answer_token_logprob(text, lp_content, item.question_type)
    return QueryResult(
        item_id=item.id,
        variant=variant,
        extracted_answer=extracted,
        correct=correct,
        answer_logprob=lp,
        answer_top1_top2_margin=margin,
        raw_response_text=text,
    )


def run_triple_query(
    client: ClientFn,
    model: str,
    items: Sequence[ProbeItem],
    variants: Sequence[str] = ("full", "text_only"),
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[ProbeOutcome]:
    """Run each item through the given variants and collect ProbeOutcome objects.

    Default variants: ("full", "text_only"). Pass ("full", "text_only", "perturbed")
    to include Q3.
    """
    outcomes: List[ProbeOutcome] = []
    for i, item in enumerate(items):
        outcome = ProbeOutcome(item_id=item.id, stratum=dict(item.stratum))
        for variant in variants:
            outcome.results_by_variant[variant] = _run_one(client, model, item, variant)
        outcomes.append(outcome)
        if on_progress:
            on_progress(i + 1, len(items))
    return outcomes


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def encoder_lift_by_stratum(
    outcomes: Sequence[ProbeOutcome],
    stratum_keys: Sequence[str] = ("img_type_bucket", "stress_tier"),
) -> Dict[Tuple[Any, ...], Dict[str, float]]:
    """Aggregate triple-query outcomes per stratum.

    Returns:
        {
          (img_type_bucket_value, stress_tier_value): {
            'n_items': int,
            'acc_full': float,
            'acc_text_only': float,
            'encoder_lift': float,         # acc_full − acc_text_only
            'mean_margin_full': float | None,
            'mean_margin_text_only': float | None,
            'acc_perturbed': float | None,
            'acc_lift_vs_perturbed': float | None,  # acc_full − acc_perturbed
          },
          ...
        }
    """
    by_key: Dict[Tuple[Any, ...], List[ProbeOutcome]] = defaultdict(list)
    for o in outcomes:
        key = tuple(o.stratum.get(k) for k in stratum_keys)
        by_key[key].append(o)

    def _mean(vals: List[Optional[float]]) -> Optional[float]:
        clean = [v for v in vals if v is not None]
        return sum(clean) / len(clean) if clean else None

    def _frac_correct(outs: List[ProbeOutcome], variant: str) -> Optional[float]:
        results = [o.results_by_variant.get(variant) for o in outs]
        results = [r for r in results if r is not None]
        if not results:
            return None
        return sum(1 for r in results if r.correct) / len(results)

    out: Dict[Tuple[Any, ...], Dict[str, float]] = {}
    for key, outs in by_key.items():
        acc_full = _frac_correct(outs, "full")
        acc_text = _frac_correct(outs, "text_only")
        acc_pert = _frac_correct(outs, "perturbed")
        margins_full = _mean([
            o.results_by_variant.get("full").answer_top1_top2_margin
            for o in outs
            if "full" in o.results_by_variant
        ])
        margins_text = _mean([
            o.results_by_variant.get("text_only").answer_top1_top2_margin
            for o in outs
            if "text_only" in o.results_by_variant
        ])
        stratum_summary: Dict[str, Any] = {
            "n_items": len(outs),
            "acc_full": acc_full,
            "acc_text_only": acc_text,
            "encoder_lift": (
                None if acc_full is None or acc_text is None else acc_full - acc_text
            ),
            "mean_margin_full": margins_full,
            "mean_margin_text_only": margins_text,
            "acc_perturbed": acc_pert,
            "acc_lift_vs_perturbed": (
                None if acc_full is None or acc_pert is None else acc_full - acc_pert
            ),
        }
        out[key] = stratum_summary
    return out


# ---------------------------------------------------------------------------
# CLI driver (live runs; requires OPENAI_API_KEY + items source)
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-file", required=True,
                        help="cache file from precompute_mmmu.py")
    parser.add_argument("--model", required=True,
                        help="model name for chat.completions.create(model=...)")
    parser.add_argument("--items-source", choices=["hf"], default="hf",
                        help="how to materialize the ProbeItems (current: HF stream)")
    parser.add_argument("--hf-repo", default="MMMU/MMMU")
    parser.add_argument("--hf-split", default="validation")
    parser.add_argument("--variants", nargs="+", default=["full", "text_only"],
                        help="subset of {full, text_only, perturbed}")
    parser.add_argument("--output-dir", default="./probe_results")
    args = parser.parse_args()

    with open(args.index_file) as f:
        cache = json.load(f)
    selected_ids = set(cache.get("selected_ids") or cache.get("selected_hashes") or cache)
    print(f"[encoder_probe] selected {len(selected_ids)} items from {args.index_file}")

    items = _materialize_items_from_hf(args.hf_repo, args.hf_split, selected_ids)
    print(f"[encoder_probe] materialized {len(items)} ProbeItems")

    client = default_openai_client()
    outcomes = run_triple_query(
        client, args.model, items, variants=tuple(args.variants),
        on_progress=lambda done, total: print(f"  [{done}/{total}]") if done % 10 == 0 else None,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, "outcomes.json")
    with open(raw_path, "w") as f:
        json.dump([_outcome_to_dict(o) for o in outcomes], f, indent=2)
    summary = encoder_lift_by_stratum(outcomes)
    summary_path = os.path.join(args.output_dir, "encoder_lift_by_stratum.json")
    with open(summary_path, "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)
    print(f"[encoder_probe] wrote {raw_path} and {summary_path}")


def _outcome_to_dict(o: ProbeOutcome) -> Dict[str, Any]:
    return {
        "item_id": o.item_id,
        "stratum": o.stratum,
        "results_by_variant": {
            v: {
                "extracted_answer": r.extracted_answer,
                "correct": r.correct,
                "answer_logprob": r.answer_logprob,
                "answer_top1_top2_margin": r.answer_top1_top2_margin,
            }
            for v, r in o.results_by_variant.items()
        },
    }


def _materialize_items_from_hf(
    repo: str, split: str, selected_ids: set
) -> List[ProbeItem]:
    """Pull selected items from HF, building ProbeItem objects with base64 images.

    This DOES download image bytes per selected row (needed to actually send
    them to the model). For r=0.10 on a 12K dataset that's 1,200 items at
    ~100KB-1MB each = ~100MB-1GB transient. We do not persist to disk; we
    keep each row in memory only long enough to encode."""
    try:
        from datasets import get_dataset_config_names, load_dataset  # type: ignore
    except ImportError as e:
        raise ImportError("`datasets` required to materialize items from HF") from e
    import ast
    items: List[ProbeItem] = []
    for subj in get_dataset_config_names(repo):
        ds = load_dataset(repo, subj, split=split, streaming=True)
        for row in ds:
            if row.get("id") not in selected_ids:
                continue
            opts = ast.literal_eval(row.get("options") or "[]")
            images_b64: List[str] = []
            for k in range(1, 8):
                fld = row.get(f"image_{k}")
                if not fld:
                    continue
                if isinstance(fld, dict) and "bytes" in fld:
                    b = fld["bytes"]
                elif hasattr(fld, "tobytes"):
                    b = fld.tobytes()
                else:
                    continue
                images_b64.append(
                    f"data:image/png;base64,{base64.b64encode(b).decode('ascii')}"
                )
            item = ProbeItem(
                id=row["id"],
                question=row.get("question", ""),
                options=opts,
                correct_answer=row.get("answer", ""),
                question_type=row.get("question_type", "open"),
                images_b64=images_b64,
                stratum={
                    "subfield": row.get("subfield"),
                    "subject": subj,
                    "img_type": row.get("img_type"),
                    "difficulty": row.get("topic_difficulty"),
                    "question_type": row.get("question_type"),
                },
            )
            items.append(item)
    return items


if __name__ == "__main__":
    main()
