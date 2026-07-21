from __future__ import annotations

from dataclasses import dataclass

###############################################################################
@dataclass(frozen=True)
class BenchmarkMetricPlan:
    needs_throughput: bool
    needs_latency: bool
    needs_latency_distribution: bool
    needs_fidelity: bool
    needs_round_trip: bool
    needs_unknown_rate: bool
    needs_character_coverage: bool
    needs_fragmentation: bool
    needs_fragmentation_buckets: bool
    needs_resources: bool
    needs_per_document_token_counts: bool
    needs_per_document_fragmentation: bool
    needs_per_document_latency: bool
    needs_per_document_memory: bool
    needs_per_document_stats: bool

###############################################################################
def build_metric_plan(
    selected_metric_keys: list[str],
    *,
    store_per_document_stats: bool,
) -> BenchmarkMetricPlan:
    selected = set(selected_metric_keys)
    needs_throughput = any(key.startswith("eff.") for key in selected)
    needs_latency = any(key.startswith("lat.") for key in selected)
    needs_latency_distribution = bool({"lat.encode_latency_distribution", "doc.encode_latency_distribution"} & selected)
    needs_fidelity = any(key.startswith("fid.") for key in selected)
    needs_round_trip = bool(
        {"fid.exact_round_trip_rate", "fid.normalized_round_trip_rate"} & selected
    )
    needs_unknown_rate = "fid.unknown_token_rate" in selected
    needs_character_coverage = "fid.lossless_encodability_rate" in selected
    needs_fragmentation = any(
        key.startswith("frag.") or key.startswith("compression.") for key in selected
    )
    needs_fragmentation_buckets = "frag.fragmentation_by_word_length_bucket" in selected
    needs_resources = any(key.startswith("res.") for key in selected)
    needs_per_document_token_counts = "doc.tokens_count_distribution" in selected
    needs_per_document_fragmentation = bool({"doc.bytes_per_token_distribution", "doc.pieces_per_word_distribution"} & selected)
    needs_per_document_latency = "doc.encode_latency_distribution" in selected
    needs_per_document_memory = "doc.peak_rss_distribution" in selected
    return BenchmarkMetricPlan(
        needs_throughput=needs_throughput,
        needs_latency=needs_latency,
        needs_latency_distribution=needs_latency_distribution,
        needs_fidelity=needs_fidelity,
        needs_round_trip=needs_round_trip,
        needs_unknown_rate=needs_unknown_rate,
        needs_character_coverage=needs_character_coverage,
        needs_fragmentation=needs_fragmentation,
        needs_fragmentation_buckets=needs_fragmentation_buckets,
        needs_resources=needs_resources,
        needs_per_document_token_counts=needs_per_document_token_counts,
        needs_per_document_fragmentation=needs_per_document_fragmentation,
        needs_per_document_latency=needs_per_document_latency,
        needs_per_document_memory=needs_per_document_memory,
        needs_per_document_stats=store_per_document_stats or any((needs_per_document_token_counts, needs_per_document_fragmentation, needs_per_document_latency, needs_per_document_memory)),
    )
