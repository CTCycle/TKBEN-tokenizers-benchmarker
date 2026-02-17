"""
E2E tests for benchmark API endpoints.
Covers /benchmarks/run validation and optional happy-path execution.
"""
import os

import pytest
from playwright.sync_api import APIRequestContext


RUN_BENCHMARKS = os.getenv("E2E_RUN_BENCHMARKS", "").lower() in ("1", "true", "yes")


def test_run_benchmarks_requires_tokenizers(api_context: APIRequestContext) -> None:
    """POST /benchmarks/run should reject empty tokenizer lists."""
    response = api_context.post(
        "/benchmarks/run",
        data={"tokenizers": [], "dataset_name": "custom/e2e_sample"},
    )
    assert response.status == 400
    data = response.json()
    assert "At least one tokenizer" in data.get("detail", "")


def test_run_benchmarks_requires_dataset(api_context: APIRequestContext) -> None:
    """POST /benchmarks/run should reject missing dataset names."""
    response = api_context.post(
        "/benchmarks/run",
        data={"tokenizers": ["hf-internal-testing/tiny-random-bert"], "dataset_name": ""},
    )
    assert response.status == 400
    data = response.json()
    assert "Dataset name must be specified" in data.get("detail", "")


def test_run_benchmarks_missing_dataset_returns_400(
    api_context: APIRequestContext,
) -> None:
    """POST /benchmarks/run should reject unknown datasets before loading tokenizers."""
    response = api_context.post(
        "/benchmarks/run",
        data={
            "tokenizers": ["hf-internal-testing/tiny-random-bert"],
            "dataset_name": "missing_dataset",
        },
    )
    assert response.status == 400
    data = response.json()
    assert "not found or empty" in data.get("detail", "").lower()


def test_get_benchmark_metrics_catalog_returns_categories(
    api_context: APIRequestContext,
) -> None:
    """GET /benchmarks/metrics/catalog should return a non-empty catalog."""
    response = api_context.get("/benchmarks/metrics/catalog")
    assert response.ok, response.text()
    data = response.json()
    categories = data.get("categories", [])
    assert isinstance(categories, list)
    assert len(categories) > 0
    first = categories[0]
    assert isinstance(first.get("category_key"), str)
    assert isinstance(first.get("metrics"), list)


def test_list_benchmark_reports_returns_payload(
    api_context: APIRequestContext,
) -> None:
    """GET /benchmarks/reports should always return the reports array."""
    response = api_context.get("/benchmarks/reports")
    assert response.ok, response.text()
    data = response.json()
    assert isinstance(data.get("reports", []), list)


@pytest.mark.skipif(
    not RUN_BENCHMARKS,
    reason="Set E2E_RUN_BENCHMARKS=1 to enable benchmark execution.",
)
def test_run_benchmarks_with_sample_dataset(
    api_context: APIRequestContext,
    uploaded_dataset: dict,
    job_waiter,
) -> None:
    """POST /benchmarks/run should return chart data for a small dataset."""
    response = api_context.post(
        "/benchmarks/run",
        data={
            "tokenizers": ["hf-internal-testing/tiny-random-bert"],
            "dataset_name": uploaded_dataset["dataset_name"],
            "max_documents": 2,
            "run_name": "e2e benchmark report",
            "selected_metric_keys": [
                "global.tokenization_speed_tps",
                "global.oov_rate",
                "speed.tokens_per_second",
                "document.bytes_per_token",
            ],
        },
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
    assert data.get("tokenizers_count", 0) >= 1
    assert data.get("documents_processed", 0) >= 1
    assert "chart_data" in data
    assert data.get("report_id")
    report_id = int(data.get("report_id"))
    assert report_id > 0
    assert data.get("run_name") == "e2e benchmark report"
    assert isinstance(data.get("selected_metric_keys"), list)

    list_response = api_context.get("/benchmarks/reports")
    assert list_response.ok, list_response.text()
    report_list = list_response.json().get("reports", [])
    assert any(int(item.get("report_id", 0)) == report_id for item in report_list)

    by_id_response = api_context.get(f"/benchmarks/reports/{report_id}")
    assert by_id_response.ok, by_id_response.text()
    by_id_data = by_id_response.json()
    assert int(by_id_data.get("report_id", 0)) == report_id
    assert by_id_data.get("dataset_name") == uploaded_dataset["dataset_name"]
    assert by_id_data.get("tokenizers_count", 0) >= 1
