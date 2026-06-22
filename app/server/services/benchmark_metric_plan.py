from __future__ import annotations

from dataclasses import dataclass

###############################################################################
@dataclass(frozen=True)
class BenchmarkMetricPlan:
    needs_latency: bool
    needs_fidelity: bool
    needs_round_trip: bool
    needs_unknown_rate: bool
    needs_character_coverage: bool
    needs_fragmentation: bool
    needs_per_document_stats: bool

###############################################################################
def build_metric_plan(
    selected_metric_keys: list[str],
    *,
    store_per_document_stats: bool,
) -> BenchmarkMetricPlan:
    selected = set(selected_metric_keys)
    needs_latency = any(key.startswith("lat.") for key in selected)
    needs_fidelity = any(key.startswith("fid.") for key in selected)
    needs_round_trip = bool(
        {"fid.exact_round_trip_rate", "fid.normalized_round_trip_rate"} & selected
    )
    needs_unknown_rate = "fid.unknown_token_rate" in selected
    needs_character_coverage = "fid.lossless_encodability_rate" in selected
    needs_fragmentation = any(
        key.startswith("frag.") or key.startswith("compression.") for key in selected
    )
    return BenchmarkMetricPlan(
        needs_latency=needs_latency,
        needs_fidelity=needs_fidelity,
        needs_round_trip=needs_round_trip,
        needs_unknown_rate=needs_unknown_rate,
        needs_character_coverage=needs_character_coverage,
        needs_fragmentation=needs_fragmentation,
        needs_per_document_stats=store_per_document_stats,
    )
