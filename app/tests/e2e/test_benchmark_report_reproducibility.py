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
    job_waiter,
) -> None:
    response = api_context.post(
        "/api/benchmarks/run",
        data={
            "tokenizers": ["hf-internal-testing/tiny-random-bert"],
            "dataset_name": uploaded_dataset["dataset_name"],
        },
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
    assert result.get("schema_version") == 1
    assert result.get("methodology_version") == "v1_observed_trials"
    assert isinstance(result.get("runtime_metadata"), dict)
    assert isinstance(result.get("raw_observations"), dict)
