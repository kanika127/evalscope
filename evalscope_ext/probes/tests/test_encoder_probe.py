"""Unit tests for encoder_probe.

Covers:
    - Prompt construction: image attached in full, withheld in text-only.
    - Answer extraction + logprob lookup on a recorded fixture.
    - Stratum-level aggregation (encoder_lift, n_items).
    - End-to-end smoke: a fake client that returns deterministic responses
      drives the triple-query path through to encoder_lift_by_stratum.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from evalscope_ext.probes.encoder_probe import (
    IMAGE_WITHHELD_TOKEN,
    ProbeItem,
    QueryResult,
    ProbeOutcome,
    build_messages_full,
    build_messages_text_only,
    encoder_lift_by_stratum,
    extract_answer,
    find_answer_token_logprob,
    run_triple_query,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mc_item(id_: str = "validation_X_1", stratum: Dict[str, Any] = None) -> ProbeItem:
    return ProbeItem(
        id=id_,
        question="<image 1> What organ is highlighted?",
        options=["liver", "kidney", "lung", "heart"],
        correct_answer="B",
        question_type="multiple-choice",
        images_b64=["data:image/png;base64,iVBORw0KGgo=" ],
        stratum=stratum or {"img_type_bucket": "high", "stress_tier": 2},
    )


def _open_item(id_: str = "validation_X_2") -> ProbeItem:
    return ProbeItem(
        id=id_,
        question="<image 1> Describe the chemical structure.",
        options=[],
        correct_answer="benzene",
        question_type="open",
        images_b64=["data:image/png;base64,iVBORw0KGgo="],
        stratum={"img_type_bucket": "high", "stress_tier": 2},
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_full_messages_include_image():
    item = _mc_item()
    msgs = build_messages_full(item)
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    parts = msgs[0]["content"]
    assert isinstance(parts, list)
    types = [p["type"] for p in parts]
    assert "text" in types and "image_url" in types
    # The text part references the question and options
    text = next(p["text"] for p in parts if p["type"] == "text")
    assert "What organ is highlighted?" in text
    assert "A. liver" in text and "D. heart" in text
    # The image_url part carries the data URI verbatim
    iu = next(p for p in parts if p["type"] == "image_url")
    assert iu["image_url"]["url"].startswith("data:image/png;base64,")


def test_text_only_messages_have_no_image_and_replace_placeholders():
    item = _mc_item()
    msgs = build_messages_text_only(item)
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    # text-only flattens to a single string content (no image parts)
    assert isinstance(msgs[0]["content"], str)
    assert IMAGE_WITHHELD_TOKEN in msgs[0]["content"]
    assert "<image 1>" not in msgs[0]["content"]


def test_open_text_only_replaces_placeholders():
    item = _open_item()
    msgs = build_messages_text_only(item)
    text = msgs[0]["content"]
    assert isinstance(text, str)
    assert IMAGE_WITHHELD_TOKEN in text
    assert "ANSWER:" in text


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def test_extract_answer_mc():
    assert extract_answer("Reasoning…\nANSWER: B", "multiple-choice") == "B"
    assert extract_answer("ANSWER: b", "multiple-choice") == "B"
    # When the model says "the correct option is B", the ANSWER: line still wins
    assert extract_answer("ANSWER: D\nFinal: D", "multiple-choice") == "D"
    # Tail line wins (the last ANSWER: line)
    assert extract_answer("ANSWER: A\nWait, ANSWER: C", "multiple-choice") == "C"


def test_extract_answer_open():
    assert extract_answer("Reasoning…\nANSWER: benzene", "open") == "benzene"
    assert extract_answer("No marker here", "open") is None


# ---------------------------------------------------------------------------
# Logprob lookup against a recorded fixture
# ---------------------------------------------------------------------------


def _fixture_logprobs_mc() -> List[Dict[str, Any]]:
    """A miniature logprobs.content list for an MC answer ending in 'ANSWER: B'.
    We don't include every preceding token — just enough that the ANSWER token
    is findable and the letter token follows it."""
    return [
        {"token": "Reasoning",      "logprob": -0.5, "top_logprobs": [{"token": "Reasoning", "logprob": -0.5}, {"token": "Thinking", "logprob": -2.0}]},
        {"token": " ",              "logprob": -0.1, "top_logprobs": [{"token": " ", "logprob": -0.1}, {"token": ":", "logprob": -3.0}]},
        {"token": "ANSWER",         "logprob": -0.2, "top_logprobs": [{"token": "ANSWER", "logprob": -0.2}, {"token": "Final", "logprob": -3.5}]},
        {"token": ":",              "logprob": -0.05, "top_logprobs": [{"token": ":", "logprob": -0.05}, {"token": " ", "logprob": -3.9}]},
        {"token": " ",              "logprob": -0.03, "top_logprobs": [{"token": " ", "logprob": -0.03}, {"token": "B", "logprob": -3.0}]},
        {"token": "B",              "logprob": -0.4, "top_logprobs": [{"token": "B", "logprob": -0.4}, {"token": "C", "logprob": -1.8}]},
    ]


def test_find_answer_token_logprob_mc():
    lp, margin = find_answer_token_logprob(
        "ANSWER: B", _fixture_logprobs_mc(), "multiple-choice"
    )
    assert lp == pytest.approx(-0.4)
    assert margin == pytest.approx(1.4)  # -0.4 - (-1.8)


def test_find_answer_token_logprob_returns_none_when_no_answer_marker():
    no_marker = [
        {"token": "Hello", "logprob": -0.5, "top_logprobs": [{"token": "Hello", "logprob": -0.5}]},
        {"token": "world", "logprob": -0.6, "top_logprobs": [{"token": "world", "logprob": -0.6}]},
    ]
    lp, margin = find_answer_token_logprob("Hello world", no_marker, "multiple-choice")
    assert lp is None and margin is None


# ---------------------------------------------------------------------------
# End-to-end with a fake client (deterministic)
# ---------------------------------------------------------------------------


class FakeClient:
    """A canned chat-completion client.

    The encoder probe is encoded as: model 'good_encoder' always answers the
    correct letter on full+image; on text-only it guesses 'A'. Model
    'degraded_encoder' answers 'A' on both. This lets the test assert that
    encoder_lift discriminates.
    """

    def __init__(self, model_behavior: str):
        self.model_behavior = model_behavior

    def __call__(self, model: str, messages, **kwargs) -> Dict[str, Any]:
        has_image = any(
            isinstance(m.get("content"), list)
            and any(p.get("type") == "image_url" for p in m["content"])
            for m in messages
        )
        if self.model_behavior == "good_encoder" and has_image:
            return self._mk_response("B")
        if self.model_behavior == "good_encoder" and not has_image:
            return self._mk_response("A")
        # degraded: always A
        return self._mk_response("A")

    @staticmethod
    def _mk_response(letter: str) -> Dict[str, Any]:
        return {
            "choices": [{
                "message": {"role": "assistant", "content": f"Reasoning…\nANSWER: {letter}"},
                "logprobs": {
                    "content": [
                        {"token": "ANSWER", "logprob": -0.2, "top_logprobs": [{"token": "ANSWER", "logprob": -0.2}, {"token": "Final", "logprob": -3.5}]},
                        {"token": ":",      "logprob": -0.05, "top_logprobs": [{"token": ":", "logprob": -0.05}, {"token": " ", "logprob": -2.0}]},
                        {"token": " ",      "logprob": -0.03, "top_logprobs": [{"token": " ", "logprob": -0.03}, {"token": letter, "logprob": -2.0}]},
                        {"token": letter,   "logprob": -0.4, "top_logprobs": [{"token": letter, "logprob": -0.4}, {"token": "C", "logprob": -1.8}]},
                    ]
                },
            }]
        }


def test_run_triple_query_with_fake_client_good_encoder():
    items = [
        _mc_item(id_=f"i{n}", stratum={"img_type_bucket": "high", "stress_tier": 2})
        for n in range(4)
    ]
    client = FakeClient("good_encoder")
    outcomes = run_triple_query(client, "good_encoder", items, variants=("full", "text_only"))
    assert len(outcomes) == 4
    for o in outcomes:
        # full → B → correct; text-only → A → wrong
        assert o.results_by_variant["full"].correct
        assert not o.results_by_variant["text_only"].correct


def test_encoder_lift_aggregates_correctly():
    # 3 high-stress items + 2 low-stress controls
    items_high = [
        _mc_item(id_=f"h{n}", stratum={"img_type_bucket": "high", "stress_tier": 2})
        for n in range(3)
    ]
    items_low = [
        _mc_item(id_=f"l{n}", stratum={"img_type_bucket": "low", "stress_tier": 0})
        for n in range(2)
    ]
    outcomes_good = run_triple_query(
        FakeClient("good_encoder"), "good_encoder", items_high + items_low
    )
    summary_good = encoder_lift_by_stratum(outcomes_good)
    # In our fake, BOTH high and low strata show full=1.0 and text-only=0.0
    # → encoder_lift = 1.0 in both. That's expected: FakeClient doesn't
    # differentiate stratum, only image presence.
    high_key = ("high", 2); low_key = ("low", 0)
    assert summary_good[high_key]["n_items"] == 3
    assert summary_good[low_key]["n_items"] == 2
    assert summary_good[high_key]["encoder_lift"] == pytest.approx(1.0)
    assert summary_good[low_key]["encoder_lift"] == pytest.approx(1.0)

    # Degraded encoder: same answer regardless of image → encoder_lift = 0
    outcomes_bad = run_triple_query(
        FakeClient("degraded_encoder"), "bad_model", items_high + items_low
    )
    summary_bad = encoder_lift_by_stratum(outcomes_bad)
    # All wrong (always "A", correct is "B") → acc_full = 0, acc_text = 0, lift = 0
    assert summary_bad[high_key]["acc_full"] == pytest.approx(0.0)
    assert summary_bad[high_key]["acc_text_only"] == pytest.approx(0.0)
    assert summary_bad[high_key]["encoder_lift"] == pytest.approx(0.0)


def test_perturbed_variant_is_optional():
    """Running with only ('full', 'text_only') should NOT compute perturbed."""
    items = [_mc_item()]
    outcomes = run_triple_query(
        FakeClient("good_encoder"), "good", items, variants=("full", "text_only")
    )
    assert "perturbed" not in outcomes[0].results_by_variant


def test_stratum_with_zero_items_not_in_summary():
    """Strata with no outcomes don't appear in the summary."""
    items = [_mc_item(stratum={"img_type_bucket": "high", "stress_tier": 2})]
    outcomes = run_triple_query(FakeClient("good_encoder"), "g", items)
    summary = encoder_lift_by_stratum(outcomes)
    assert ("high", 2) in summary
    assert ("low", 0) not in summary
