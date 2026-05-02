from __future__ import annotations

import os

import pytest
from playwright.sync_api import APIRequestContext


RUN_BENCHMARKS = os.getenv("E2E_RUN_BENCHMARKS", "").lower() in ("1", "true", "yes")


@pytest.mark.skipif(
    not RUN_BENCHMARKS,
    reason="Set E2E_RUN_BENCHMARKS=1 to enable benchmark execution.",
)
def test_benchmark_reports_round_trip_current_schema(
    api_context: APIRequestContext,
    uploaded_dataset: dict,
    job_waiter,
) -> None:
    run_response = api_context.post(
        "/api/benchmarks/run",
        data={
            "tokenizers": ["hf-internal-testing/tiny-random-bert"],
            "dataset_name": uploaded_dataset["dataset_name"],
            "run_name": "e2e benchmark reports schema",
            "config": {
                "max_documents": 2,
                "warmup_trials": 2,
                "timed_trials": 8,
                "batch_size": 16,
                "seed": 42,
                "parallelism": 1,
                "include_lm_metrics": False,
            },
        },
    )
    assert run_response.ok, run_response.text()
    job = run_response.json()
    job_id = job.get("job_id")
    assert job_id

    job_status = job_waiter(
        job_id,
        poll_interval=job.get("poll_interval", 1.0),
        timeout_seconds=1800.0,
    )
    assert job_status.get("status") == "completed", job_status.get("error")

    result = job_status.get("result", {})
    assert result.get("status") == "success"
    assert isinstance(result.get("tokenizer_results"), list)
    assert isinstance(result.get("chart_data"), dict)
    report_id = int(result.get("report_id", 0))
    assert report_id > 0

    list_response = api_context.get("/api/benchmarks/reports")
    assert list_response.ok, list_response.text()
    report_list = list_response.json().get("reports", [])
    listed = next(
        (item for item in report_list if int(item.get("report_id", 0)) == report_id),
        None,
    )
    assert listed is not None
    assert listed.get("dataset_name") == uploaded_dataset["dataset_name"]

    by_id_response = api_context.get(f"/api/benchmarks/reports/{report_id}")
    assert by_id_response.ok, by_id_response.text()
    by_id = by_id_response.json()
    assert int(by_id.get("report_id", 0)) == report_id
    assert by_id.get("dataset_name") == uploaded_dataset["dataset_name"]
    assert isinstance(by_id.get("tokenizer_results"), list)
    assert isinstance(by_id.get("chart_data"), dict)
