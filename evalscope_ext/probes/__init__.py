"""Black-box probes that run against an OpenAI-compatible Chat Completions endpoint.

Public API:
    encoder_probe.run_triple_query   — Q1 full / Q2 text-only / Q3 perturbed
    encoder_probe.encoder_lift_by_stratum — aggregate triple-query results
"""

from evalscope_ext.probes import encoder_probe

__all__ = ["encoder_probe"]
