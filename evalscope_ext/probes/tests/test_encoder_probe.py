"""Unit tests for encoder_probe.

Covers:
    - Prompt construction: image attached in full + perturbed, withheld in text-only.
    - Answer extraction + logprob lookup on a recorded fixture.
    - Stratum-level numeric aggregation (lift_text, lift_pert, accs).
    - End-to-end smoke: a fake client drives the triple-query path through
      to encoder_lift_by_stratum.
    - Joint encoder signal: one trial per state (ABSENT / COARSE / HEALTHY)
      plus an at-threshold boundary case for the classification rule.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from evalscope_ext.probes.encoder_probe import (
    DEFAULT_TAU_LIFT,
    DEFAULT_TAU_PERT,
    DEFAULT_VARIANTS,
    IMAGE_WITHHELD_TOKEN,
    ProbeItem,
    ProbeOutcome,
    QueryResult,
    STATE_ABSENT,
    STATE_COARSE,
    STATE_HEALTHY,
    build_messages_full,
    build_messages_perturbed,
    build_messages_text_only,
    classify_state,
    encoder_lift_by_stratum,
    extract_answer,
    find_answer_token_logprob,
    joint_encoder_signal,
    render_joint_report,
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
    """A canned chat-completion client encoding distinct encoder behaviours.

    Behaviour names describe what the MODEL DOES per variant, not the
    classifier verdict — the classifier verdict is the test's assertion.

      'gist_only'      — correct ('B') on full and on perturbed; wrong ('A')
                         on text-only. The model needs the image but is
                         UNAFFECTED by destructive perturbation, i.e. it's
                         reading coarse gist not fine detail.
                         → joint classifier: COARSE.
      'detail_reader'  — correct on full; wrong on perturbed (the 56×56
                         crush kills the signal); wrong on text-only.
                         The model is genuinely reading fine spatial
                         content → joint classifier: HEALTHY.
      'broken'         — wrong on all three variants. Encoder contributes
                         nothing. → ABSENT.
      'text_solvable'  — correct on all three. The item didn't need the
                         image. lift_text ≈ 0 → ABSENT (correctly).
    """

    _DECISIONS = {
        "gist_only":     {"full": "B", "text_only": "A", "perturbed": "B"},
        "detail_reader": {"full": "B", "text_only": "A", "perturbed": "A"},
        "broken":        {"full": "A", "text_only": "A", "perturbed": "A"},
        "text_solvable": {"full": "B", "text_only": "B", "perturbed": "B"},
    }

    def __init__(self, behaviour: str):
        if behaviour not in self._DECISIONS:
            raise ValueError(f"unknown behaviour {behaviour!r}")
        self.behaviour = behaviour

    @staticmethod
    def _detect_variant(messages) -> str:
        # full and perturbed both have image_url parts; text_only does not.
        has_image = any(
            isinstance(m.get("content"), list)
            and any(p.get("type") == "image_url" for p in m["content"])
            for m in messages
        )
        if not has_image:
            return "text_only"
        # We use a tiny marker in the perturbed image URL to disambiguate
        # full from perturbed in the test — see `_marked_data_uri` below.
        for m in messages:
            c = m.get("content")
            if not isinstance(c, list):
                continue
            for p in c:
                if p.get("type") != "image_url":
                    continue
                url = p["image_url"]["url"]
                if "PERTURBED" in url:
                    return "perturbed"
        return "full"

    def __call__(self, model: str, messages, **kwargs) -> Dict[str, Any]:
        variant = self._detect_variant(messages)
        letter = self._DECISIONS[self.behaviour][variant]
        return self._mk_response(letter)

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


def _mark_perturbed(item: ProbeItem) -> ProbeItem:
    """For the FakeClient state tests, replace `_perturb_data_uri` with a
    marker that lets the client distinguish full from perturbed without
    needing Pillow. (PIL isn't a hard dependency.)"""
    item.images_b64.clear()
    return item


@pytest.fixture(autouse=True)
def _patch_perturb(monkeypatch):
    """Substitute _perturb_data_uri so build_messages_perturbed leaves a
    marker the FakeClient can detect (we don't want to require Pillow)."""
    from evalscope_ext.probes import encoder_probe as ep
    monkeypatch.setattr(
        ep, "_perturb_data_uri",
        lambda uri, target_size=56: uri + "#PERTURBED",
    )


def test_full_messages_include_image_default_run():
    """Sanity: run_triple_query default variants now include perturbed."""
    assert DEFAULT_VARIANTS == ("full", "text_only", "perturbed")


def test_build_messages_perturbed_has_image_with_marker():
    item = _mc_item()
    msgs = build_messages_perturbed(item)
    iu = next(p for p in msgs[0]["content"] if p["type"] == "image_url")
    assert "PERTURBED" in iu["image_url"]["url"]


def test_run_triple_query_runs_all_three_variants_by_default():
    items = [
        _mc_item(id_=f"i{n}", stratum={"img_type_bucket": "high", "stress_tier": 2})
        for n in range(2)
    ]
    outcomes = run_triple_query(FakeClient("detail_reader"), "m", items)
    assert len(outcomes) == 2
    for o in outcomes:
        assert set(o.results_by_variant.keys()) == {"full", "text_only", "perturbed"}


def test_encoder_lift_aggregates_lift_text_and_lift_pert():
    """The numeric aggregator now exposes lift_text and lift_pert (the
    pre-joint `encoder_lift` field was split — the joint signal is the
    headline). With the `gist_only` fake, acc_full = acc_perturbed = 1.0,
    acc_text_only = 0.0, so lift_text=1.0 and lift_pert=0.0."""
    items = [
        _mc_item(id_=f"i{n}", stratum={"img_type_bucket": "high", "stress_tier": 2})
        for n in range(4)
    ]
    outcomes = run_triple_query(FakeClient("gist_only"), "m", items)
    summary = encoder_lift_by_stratum(outcomes)
    key = ("high", 2)
    s = summary[key]
    assert s["n_items"] == 4
    assert s["acc_full"] == pytest.approx(1.0)
    assert s["acc_text_only"] == pytest.approx(0.0)
    assert s["acc_perturbed"] == pytest.approx(1.0)
    assert s["lift_text"] == pytest.approx(1.0)
    assert s["lift_pert"] == pytest.approx(0.0)


def test_stratum_with_zero_items_not_in_summary():
    items = [_mc_item(stratum={"img_type_bucket": "high", "stress_tier": 2})]
    outcomes = run_triple_query(FakeClient("gist_only"), "m", items)
    summary = encoder_lift_by_stratum(outcomes)
    assert ("high", 2) in summary
    assert ("low", 0) not in summary


# ---------------------------------------------------------------------------
# classify_state — direct rule tests, no I/O
# ---------------------------------------------------------------------------


def test_classify_state_absent_when_lift_text_below_tau_lift():
    assert classify_state(0.05, 0.30, tau_lift=0.10, tau_pert=0.05) == STATE_ABSENT
    # Even if lift_pert is huge, low lift_text dominates
    assert classify_state(0.00, 0.99, tau_lift=0.10, tau_pert=0.05) == STATE_ABSENT


def test_classify_state_coarse_when_lift_text_passes_lift_pert_fails():
    assert classify_state(0.40, 0.02, tau_lift=0.10, tau_pert=0.05) == STATE_COARSE


def test_classify_state_healthy_when_both_pass():
    assert classify_state(0.40, 0.20, tau_lift=0.10, tau_pert=0.05) == STATE_HEALTHY


def test_classify_state_boundary_inclusive_on_thresholds():
    """Exact-on-threshold lift counts as meeting it (>= semantics).
    Lift_text=τ_lift, lift_pert=τ_pert → HEALTHY (not COARSE, not ABSENT).
    Lift_text slightly below → ABSENT. Lift_pert slightly below → COARSE."""
    assert classify_state(0.10, 0.05, tau_lift=0.10, tau_pert=0.05) == STATE_HEALTHY
    assert classify_state(0.099999, 0.05, tau_lift=0.10, tau_pert=0.05) == STATE_ABSENT
    assert classify_state(0.10, 0.049999, tau_lift=0.10, tau_pert=0.05) == STATE_COARSE


def test_classify_state_returns_none_when_a_lift_is_missing():
    # Without Q2 or Q3 we can't classify
    assert classify_state(None, 0.20) is None
    assert classify_state(0.30, None) is None


# ---------------------------------------------------------------------------
# joint_encoder_signal — the end-to-end Part B headline classifier
# ---------------------------------------------------------------------------


def _items(behaviour_strata, n_per: int = 4) -> List[ProbeItem]:
    """Build N items per (img_type_bucket, stress_tier) stratum."""
    out = []
    for stratum in behaviour_strata:
        for n in range(n_per):
            out.append(
                _mc_item(
                    id_=f"{stratum['img_type_bucket']}_{stratum['stress_tier']}_{n}",
                    stratum=stratum,
                )
            )
    return out


def test_joint_signal_classifies_gist_only_as_coarse():
    """FakeClient('gist_only') is UNAFFECTED by destructive 56×56
    perturbation (it answers correctly on full and on perturbed) but wrong
    on text-only. So lift_text=1.0 passes τ_lift, but lift_pert=0.0 falls
    below τ_pert. The joint rule names this COARSE — exactly the
    fp8/quantized signature the dual signal is designed to catch."""
    items = _items([{"img_type_bucket": "high", "stress_tier": 2}])
    outcomes = run_triple_query(FakeClient("gist_only"), "m", items)
    joint = joint_encoder_signal(outcomes)
    s = joint["strata"]["('high', 2)"]
    assert s["lift_text"] == pytest.approx(1.0)
    assert s["lift_pert"] == pytest.approx(0.0)
    assert s["state"] == STATE_COARSE


def test_joint_signal_classifies_detail_reader_as_healthy():
    """FakeClient('detail_reader') is correct on full, WRONG on perturbed
    (the 56×56 crush destroyed the signal it was reading), and wrong on
    text-only. Both lifts = 1.0 → HEALTHY."""
    items = _items([{"img_type_bucket": "high", "stress_tier": 2}])
    outcomes = run_triple_query(FakeClient("detail_reader"), "m", items)
    joint = joint_encoder_signal(outcomes)
    s = joint["strata"]["('high', 2)"]
    assert s["lift_text"] == pytest.approx(1.0)
    assert s["lift_pert"] == pytest.approx(1.0)
    assert s["state"] == STATE_HEALTHY


def test_joint_signal_classifies_broken_as_absent():
    """All wrong everywhere → both lifts 0 → ABSENT."""
    items = _items([{"img_type_bucket": "high", "stress_tier": 2}])
    outcomes = run_triple_query(FakeClient("broken"), "m", items)
    joint = joint_encoder_signal(outcomes)
    s = joint["strata"]["('high', 2)"]
    assert s["lift_text"] == pytest.approx(0.0)
    assert s["lift_pert"] == pytest.approx(0.0)
    assert s["state"] == STATE_ABSENT


def test_joint_signal_absent_state_on_text_solvable_items():
    """If text-only resolves the items just as well as full+image, the
    encoder is contributing nothing on this stratum even if the model is
    getting them all right. The joint rule must say ABSENT."""
    items = _items([{"img_type_bucket": "low", "stress_tier": 0}])
    outcomes = run_triple_query(FakeClient("text_solvable"), "m", items)
    joint = joint_encoder_signal(outcomes)
    s = joint["strata"]["('low', 0)"]
    assert s["lift_text"] == pytest.approx(0.0)
    assert s["lift_pert"] == pytest.approx(0.0)
    assert s["state"] == STATE_ABSENT


def test_joint_signal_state_counts_match_strata():
    items = (
        _items([{"img_type_bucket": "high", "stress_tier": 2}], n_per=3) +
        _items([{"img_type_bucket": "low", "stress_tier": 0}], n_per=3)
    )
    # Run two separate behaviours via a hybrid client? Simpler: just
    # verify the count matches the sum.
    outcomes = run_triple_query(FakeClient("broken"), "m", items)
    joint = joint_encoder_signal(outcomes)
    assert joint["n_strata"] == 2
    assert sum(joint["state_counts"].values()) == 2
    assert joint["state_counts"][STATE_ABSENT] == 2


def test_render_joint_report_mentions_thresholds_and_states():
    items = _items([{"img_type_bucket": "high", "stress_tier": 2}])
    outcomes = run_triple_query(FakeClient("gist_only"), "m", items)
    joint = joint_encoder_signal(outcomes, tau_lift=0.10, tau_pert=0.05)
    text = render_joint_report(joint)
    assert "τ_lift" in text and "τ_pert" in text
    assert "HEALTHY" in text and "COARSE" in text and "ABSENT" in text
    assert "lift_text" in text and "lift_pert" in text


def test_joint_signal_threshold_tunability():
    """The thresholds are calibration parameters. Bumping them must shift
    a stratum's state — proving that downstream decisions can be tuned
    without touching the data or the rule. Uses detail_reader so both
    lifts = 1.0 (so we can move the verdict by moving the τ above 1.0)."""
    items = _items([{"img_type_bucket": "high", "stress_tier": 2}])
    outcomes = run_triple_query(FakeClient("detail_reader"), "m", items)
    # Default thresholds: HEALTHY (both lifts = 1.0)
    j1 = joint_encoder_signal(outcomes, tau_lift=0.10, tau_pert=0.05)
    assert j1["strata"]["('high', 2)"]["state"] == STATE_HEALTHY
    # Bump τ_lift above the observed lift_text=1.0 → ABSENT
    j2 = joint_encoder_signal(outcomes, tau_lift=1.01, tau_pert=0.05)
    assert j2["strata"]["('high', 2)"]["state"] == STATE_ABSENT
    # Restore τ_lift but bump τ_pert above lift_pert=1.0 → COARSE
    j3 = joint_encoder_signal(outcomes, tau_lift=0.10, tau_pert=1.01)
    assert j3["strata"]["('high', 2)"]["state"] == STATE_COARSE


def test_joint_signal_returns_unknown_when_q3_missing():
    """If the caller runs without 'perturbed', the joint rule must yield
    UNKNOWN (None state), not silently pretend Q3 was clean."""
    items = _items([{"img_type_bucket": "high", "stress_tier": 2}])
    outcomes = run_triple_query(
        FakeClient("detail_reader"), "m", items, variants=("full", "text_only")
    )
    joint = joint_encoder_signal(outcomes)
    assert joint["strata"]["('high', 2)"]["state"] is None
    assert joint["state_counts"]["unknown"] == 1
