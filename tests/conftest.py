"""
Pytest configuration for TKBEN E2E tests.
Provides fixtures for Playwright page objects and API client.
"""
import os
import time
from typing import Any

import pytest
from playwright.sync_api import APIRequestContext


def _read_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    normalized = value.strip().strip('"').strip("'")
    return normalized or default


def _normalize_host(host: str) -> str:
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def _build_base_url(host_env: str, port_env: str, default_host: str, default_port: str) -> str:
    host = _normalize_host(_read_env(host_env, default_host) or default_host)
    port = _read_env(port_env, default_port) or default_port
    return f"http://{host}:{port}"


# Base URLs - prefer explicit test/runtime URLs, then fall back to host/port pairs.
UI_BASE_URL = (
    _read_env("APP_TEST_FRONTEND_URL")
    or _read_env("UI_BASE_URL")
    or _read_env("UI_URL")
    or _build_base_url("UI_HOST", "UI_PORT", "127.0.0.1", "8000")
)
API_BASE_URL = (
    _read_env("APP_TEST_BACKEND_URL")
    or _read_env("API_BASE_URL")
    or _build_base_url("FASTAPI_HOST", "FASTAPI_PORT", "127.0.0.1", "5000")
)


@pytest.fixture(scope="session")
def base_url() -> str:
    """Returns the base URL of the UI."""
    return UI_BASE_URL


@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Returns the base URL of the API."""
    return API_BASE_URL


@pytest.fixture(scope="session")
def api_context(playwright) -> APIRequestContext:
    """
    Creates an API request context for making direct HTTP calls.
    Useful for testing backend endpoints independently of the UI.
    """
    context = playwright.request.new_context(base_url=API_BASE_URL)
    yield context
    context.dispose()


def wait_for_job_completion(
    api_context: APIRequestContext,
    job_id: str,
    poll_interval: float = 1.0,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while True:
        response = api_context.get(f"/jobs/{job_id}")
        assert response.ok, f"Failed to poll job {job_id}: {response.status}"
        payload = response.json()
        status = payload.get("status")
        if status in {"completed", "failed", "cancelled"}:
            return payload
        if time.time() >= deadline:
            raise AssertionError(f"Job {job_id} timed out after {timeout_seconds} seconds")
        time.sleep(max(poll_interval, 0.1))


@pytest.fixture(scope="session")
def job_waiter(api_context: APIRequestContext):
    def _wait(job_id: str, poll_interval: float = 1.0, timeout_seconds: float = 300.0) -> dict[str, Any]:
        return wait_for_job_completion(
            api_context=api_context,
            job_id=job_id,
            poll_interval=poll_interval,
            timeout_seconds=timeout_seconds,
        )

    return _wait


@pytest.fixture(scope="session")
def sample_dataset_payload() -> tuple[str, str, bytes]:
    """
    Returns a small CSV payload used across dataset-related tests.
    """
    filename = os.getenv("E2E_SAMPLE_DATASET_FILE", "e2e_sample.csv")
    dataset_name = f"custom/{os.path.splitext(filename)[0]}"
    csv_content = "text\nHello world\nThis is a sample document\nAnother sample\n"
    return filename, dataset_name, csv_content.encode("utf-8")


@pytest.fixture(scope="session")
def uploaded_dataset(
    api_context: APIRequestContext,
    sample_dataset_payload: tuple[str, str, bytes],
    job_waiter,
) -> dict:
    """
    Uploads a small dataset once per test session and returns the API response.
    """
    filename, dataset_name, csv_bytes = sample_dataset_payload
    response = api_context.post(
        "/datasets/upload",
        multipart={
            "file": {
                "name": filename,
                "mimeType": "text/csv",
                "buffer": csv_bytes,
            }
        },
    )
    assert response.ok, f"Dataset upload failed: {response.status} {response.text()}"
    job = response.json()
    job_id = job.get("job_id")
    assert job_id, "Missing job_id in upload response"
    job_status = job_waiter(
        job_id,
        poll_interval=job.get("poll_interval", 1.0),
        timeout_seconds=300.0,
    )
    assert job_status.get("status") == "completed", job_status.get("error")
    result = job_status.get("result", {})
    assert result.get("dataset_name") == dataset_name
    return result
