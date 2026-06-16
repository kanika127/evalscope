"""Black-box probes that run against an OpenAI-compatible Chat Completions endpoint.

Public API:
    encoder_probe.run_triple_query         — Q1 full / Q2 text-only / Q3 perturbed (all default)
    encoder_probe.encoder_lift_by_stratum  — per-stratum numerics (lift_text, lift_pert, accs)
    encoder_probe.joint_encoder_signal     — JOINT 2-signal classifier
                                              {absent, coarse, healthy} per stratum
    encoder_probe.classify_state           — the rule from (lift_text, lift_pert, τ_lift, τ_pert)
    encoder_probe.render_joint_report      — markdown report consumed by the CLI
"""

from evalscope_ext.probes import encoder_probe

__all__ = ["encoder_probe"]
