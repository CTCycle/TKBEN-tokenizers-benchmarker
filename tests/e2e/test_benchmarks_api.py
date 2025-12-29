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
        json={"tokenizers": [], "dataset_name": "custom/e2e_sample"},
    )
    assert response.status == 400
    data = response.json()
    assert "At least one tokenizer" in data.get("detail", "")


def test_run_benchmarks_requires_dataset(api_context: APIRequestContext) -> None:
    """POST /benchmarks/run should reject missing dataset names."""
    response = api_context.post(
        "/benchmarks/run",
        json={"tokenizers": ["hf-internal-testing/tiny-random-bert"], "dataset_name": ""},
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
        json={
            "tokenizers": ["hf-internal-testing/tiny-random-bert"],
            "dataset_name": "missing_dataset",
        },
    )
    assert response.status == 400
    data = response.json()
    assert "not found or empty" in data.get("detail", "").lower()


@pytest.mark.skipif(
    not RUN_BENCHMARKS,
    reason="Set E2E_RUN_BENCHMARKS=1 to enable benchmark execution.",
)
def test_run_benchmarks_with_sample_dataset(
    api_context: APIRequestContext,
    uploaded_dataset: dict,
) -> None:
    """POST /benchmarks/run should return chart data for a small dataset."""
    response = api_context.post(
        "/benchmarks/run",
        json={
            "tokenizers": ["hf-internal-testing/tiny-random-bert"],
            "dataset_name": uploaded_dataset["dataset_name"],
            "max_documents": 2,
        },
    )
    assert response.ok, response.text()
    data = response.json()
    assert data.get("status") == "success"
    assert data.get("tokenizers_count", 0) >= 1
    assert data.get("documents_processed", 0) >= 1
    assert "chart_data" in data
