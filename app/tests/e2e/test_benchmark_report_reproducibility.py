from __future__ import annotations

import os

import pytest
from playwright.sync_api import APIRequestContext


RUN_BENCHMARKS = os.getenv("E2E_RUN_BENCHMARKS", "").lower() in ("1", "true", "yes")

###############################################################################
@pytest.mark.skipif(
    not RUN_BENCHMARKS, reason="Set E2E_RUN_BENCHMARKS=1 to enable benchmark execution."
)
def test_benchmark_report_contains_reproducibility_metadata(
    api_context: APIRequestContext,
    uploaded_dataset: dict,
    uploaded_tiny_tokenizer: dict,
    job_waiter,
) -> None:
    tokenizer_name = uploaded_tiny_tokenizer["tokenizer_name"]
    run_payload = {
        "tokenizers": [tokenizer_name],
        "dataset_name": uploaded_dataset["dataset_name"],
        "custom_tokenizer_name": tokenizer_name,
        "run_name": "e2e deterministic benchmark report",
        "selected_metric_keys": [
            "eff.encode_tokens_per_second_mean",
            "frag.tokens_per_character",
        ],
        "config": {
            "max_documents": 2,
            "warmup_trials": 1,
            "timed_trials": 1,
            "batch_size": 2,
            "seed": 42,
            "parallelism": 1,
            "include_lm_metrics": False,
            "store_per_document_stats": True,
            "per_document_sample_size": 2,
        },
    }

    response = api_context.post(
        "/api/benchmarks/run",
        data=run_payload,
    )
    assert response.ok, response.text()
    job = response.json()
    status = job_waiter(
        job["job_id"],
        poll_interval=job.get("poll_interval", 1.0),
        timeout_seconds=1800.0,
    )
    assert status.get("status") == "completed", status.get("error")
    result = status.get("result", {})
    assert result.get("status") == "success"
    assert result.get("schema_version") == 1
    assert result.get("methodology_version") == "v2_semantic_honesty"
    assert result.get("dataset_name") == uploaded_dataset["dataset_name"]
    assert result.get("documents_processed") == 2
    assert result.get("tokenizers_processed") == [tokenizer_name]
    assert result.get("tokenizers_count") == 1
    assert result.get("run_name") == run_payload["run_name"]
    assert result.get("selected_metric_keys") == run_payload["selected_metric_keys"]
    assert result.get("config", {}).get("timed_trials") == 1

    tokenizer_results = result.get("tokenizer_results", [])
    assert len(tokenizer_results) == 1
    tokenizer_result = tokenizer_results[0]
    assert tokenizer_result.get("tokenizer") == tokenizer_name
    assert tokenizer_result.get("status") == "success"
    assert tokenizer_result.get("efficiency", {}).get("encode_tokens_per_second_mean", 0) > 0
    assert not (
        tokenizer_result.get("status") == "success"
        and tokenizer_result.get("error_message")
    )

    runtime_metadata = result.get("runtime_metadata")
    assert isinstance(runtime_metadata, dict)
    assert runtime_metadata.get("benchmark_config", {}).get("timed_trials") == 1
    assert runtime_metadata.get("dataset_documents_benchmarked") == 2
    assert isinstance(runtime_metadata.get("metric_availability"), dict)
    assert isinstance(result.get("raw_observations"), dict)
    assert result.get("report_id")

    report_id = int(result["report_id"])
    by_id_response = api_context.get(f"/api/benchmarks/reports/{report_id}")
    assert by_id_response.ok, by_id_response.text()
    persisted = by_id_response.json()
    assert persisted.get("report_id") == report_id
    assert persisted.get("dataset_name") == result.get("dataset_name")
    assert persisted.get("documents_processed") == result.get("documents_processed")
    assert persisted.get("tokenizers_processed") == result.get("tokenizers_processed")
    assert persisted.get("config", {}).get("timed_trials") == 1
    assert (
        persisted.get("runtime_metadata", {}).get("benchmark_config", {})
        == runtime_metadata.get("benchmark_config")
    )
