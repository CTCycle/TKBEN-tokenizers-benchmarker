from __future__ import annotations

from TKBEN.server.services.benchmark_payloads import BenchmarkPayloadBuilder


def test_per_document_mapping_uses_encode_latency_and_peak_rss_fields() -> None:
    builder = BenchmarkPayloadBuilder()

    result = {
        "global_metrics": [],
        "per_document_stats": [
            {
                "tokenizer": "tok-a",
                "tokens_count": [1, 2],
                "bytes_per_token": [0.1, 0.2],
                "encode_latency_ms": [3.0, 4.0],
                "peak_rss_mb": [5.0, 6.0],
            }
        ],
    }

    payload = builder.build_benchmark_payload(
        result=result,
        fallback_dataset_name="custom/demo",
        config_payload={},
    )

    per_doc = payload["per_document_stats"][0]
    assert per_doc["bytes_per_token"] == [0.1, 0.2]
    assert per_doc["encode_latency_ms"] == [3.0, 4.0]
    assert per_doc["peak_rss_mb"] == [5.0, 6.0]
