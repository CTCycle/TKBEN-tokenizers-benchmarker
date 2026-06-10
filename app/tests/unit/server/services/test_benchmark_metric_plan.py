from __future__ import annotations

from server.services.benchmark_metric_plan import build_metric_plan


###############################################################################
def test_metric_plan_flags_selected_metrics() -> None:
    plan = build_metric_plan(
        [
            "lat.encode_latency_p95_ms",
            "fid.exact_round_trip_rate",
            "fid.unknown_token_rate",
        ],
        store_per_document_stats=False,
    )
    assert plan.needs_latency is True
    assert plan.needs_fidelity is True
    assert plan.needs_round_trip is True
    assert plan.needs_unknown_rate is True
    assert plan.needs_per_document_stats is False
