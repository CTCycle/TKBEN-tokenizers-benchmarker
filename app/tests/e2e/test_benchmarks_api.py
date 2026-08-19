"""
E2E tests for benchmark API endpoints.
Covers /api/benchmarks/run validation and optional happy-path execution.
"""

import os

import pytest
from playwright.sync_api import APIRequestContext


RUN_BENCHMARKS = os.getenv("E2E_RUN_BENCHMARKS", "").lower() in ("1", "true", "yes")

###############################################################################
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"tokenizers": [], "dataset_name": "custom/e2e_sample"}, "At least one tokenizer"),
        (
            {
                "tokenizers": ["hf-internal-testing/tiny-random-bert"],
                "dataset_name": "",
            },
            "Dataset name must be specified",
        ),
    ],
)
def test_run_benchmarks_rejects_missing_required_inputs(
    api_context: APIRequestContext,
    payload: dict,
    message: str,
) -> None:
    """POST /api/benchmarks/run should reject incomplete benchmark inputs."""
    response = api_context.post("/api/benchmarks/run", data=payload)
    assert response.status == 400
    data = response.json()
    assert message in data.get("detail", "")

###############################################################################
def test_run_benchmarks_missing_dataset_returns_400(
    api_context: APIRequestContext,
) -> None:
    """POST /api/benchmarks/run should reject unknown datasets before loading tokenizers."""
    response = api_context.post(
        "/api/benchmarks/run",
        data={
            "tokenizers": ["hf-internal-testing/tiny-random-bert"],
            "dataset_name": "missing_dataset",
        },
    )
    assert response.status == 400
    data = response.json()
    assert "not found or empty" in data.get("detail", "").lower()

###############################################################################
def test_get_benchmark_metrics_catalog_returns_categories(
    api_context: APIRequestContext,
) -> None:
    """GET /api/benchmarks/metrics/catalog should return a non-empty catalog."""
    response = api_context.get("/api/benchmarks/metrics/catalog")
    assert response.ok, response.text()
    data = response.json()
    categories = data.get("categories", [])
    assert isinstance(categories, list)
    assert len(categories) > 0
    first = categories[0]
    assert isinstance(first.get("category_key"), str)
    assert isinstance(first.get("metrics"), list)

###############################################################################
@pytest.mark.skipif(
    not RUN_BENCHMARKS,
    reason="Set E2E_RUN_BENCHMARKS=1 to enable benchmark execution.",
)
def test_benchmark_report_round_trip_includes_reproducibility_metadata(
    api_context: APIRequestContext,
    uploaded_dataset: dict,
    uploaded_tiny_tokenizer: dict,
    job_waiter,
) -> None:
    """A small benchmark should persist its current-schema report and metadata."""
    tokenizer_name = uploaded_tiny_tokenizer["tokenizer_name"]
    run_payload = {
        "tokenizers": [tokenizer_name],
        "dataset_name": uploaded_dataset["dataset_name"],
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
    job_id = job.get("job_id")
    assert job_id, "Missing job_id in benchmark response"
    job_status = job_waiter(
        job_id,
        poll_interval=job.get("poll_interval", 1.0),
        timeout_seconds=1800.0,
    )
    assert job_status.get("status") == "completed", job_status.get("error")
    data = job_status.get("result", {})
    assert data.get("status") == "success"
    assert data.get("schema_version") == 3
    assert data.get("methodology_version") == "semantic_honesty"
    assert data.get("dataset_name") == uploaded_dataset["dataset_name"]
    assert data.get("tokenizers_count", 0) >= 1
    assert data.get("documents_processed") == 2
    dashboard = data.get("dashboard")
    assert isinstance(dashboard, dict)
    assert isinstance(dashboard.get("widgets"), list)
    assert data.get("report_id")
    report_id = int(data.get("report_id"))
    assert report_id > 0
    assert data.get("run_name") == run_payload["run_name"]
    assert data.get("selected_metric_keys") == run_payload["selected_metric_keys"]
    assert data.get("config", {}).get("timed_trials") == 1

    tokenizer_results = data.get("tokenizer_results", [])
    assert len(tokenizer_results) == 1
    tokenizer_result = tokenizer_results[0]
    assert tokenizer_result.get("tokenizer") == tokenizer_name
    assert tokenizer_result.get("status") == "success"
    assert tokenizer_result.get("efficiency", {}).get(
        "encode_tokens_per_second_mean", 0
    ) > 0
    assert not tokenizer_result.get("error_message")

    runtime_metadata = data.get("runtime_metadata")
    assert isinstance(runtime_metadata, dict)
    assert runtime_metadata.get("benchmark_config", {}).get("timed_trials") == 1
    assert runtime_metadata.get("dataset_documents_benchmarked") == 2
    assert isinstance(runtime_metadata.get("benchmark_timing_boundaries"), dict)
    assert isinstance(data.get("raw_observations"), dict)

    list_response = api_context.get("/api/benchmarks/reports")
    assert list_response.ok, list_response.text()
    report_list = list_response.json().get("reports", [])
    assert any(int(item.get("report_id", 0)) == report_id for item in report_list)

    by_id_response = api_context.get(f"/api/benchmarks/reports/{report_id}")
    assert by_id_response.ok, by_id_response.text()
    by_id_data = by_id_response.json()
    assert int(by_id_data.get("report_id", 0)) == report_id
    assert by_id_data.get("dataset_name") == uploaded_dataset["dataset_name"]
    assert by_id_data.get("tokenizers_count", 0) >= 1
    assert by_id_data.get("run_name") == data.get("run_name")
    assert by_id_data.get("documents_processed") == data.get("documents_processed")
    assert by_id_data.get("tokenizers_processed") == data.get("tokenizers_processed")
    assert by_id_data.get("config", {}).get("timed_trials") == 1
    assert (
        by_id_data.get("runtime_metadata", {}).get("benchmark_config", {})
        == runtime_metadata.get("benchmark_config")
    )

    delete_response = api_context.delete(f"/api/benchmarks/reports/{report_id}")
    assert delete_response.status == 204, delete_response.text()
    assert api_context.get(f"/api/benchmarks/reports/{report_id}").status == 404
    remaining = api_context.get(
        f"/api/benchmarks/reports?search={run_payload['run_name'].replace(' ', '%20')}"
    )
    assert remaining.ok, remaining.text()
    assert all(int(item.get("report_id", 0)) != report_id for item in remaining.json().get("reports", []))
