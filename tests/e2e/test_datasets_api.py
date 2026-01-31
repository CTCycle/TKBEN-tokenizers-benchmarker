"""
E2E tests for dataset API endpoints.
Covers /datasets/list, /datasets/upload, and /datasets/analyze.
"""
from playwright.sync_api import APIRequestContext


def test_list_datasets_returns_list(api_context: APIRequestContext) -> None:
    """GET /datasets/list should return a list container."""
    response = api_context.get("/datasets/list")
    assert response.ok, f"Expected 200, got {response.status}"

    data = response.json()
    assert "datasets" in data
    assert isinstance(data["datasets"], list)


def test_list_datasets_includes_uploaded_dataset(
    api_context: APIRequestContext,
    uploaded_dataset: dict,
) -> None:
    """Uploaded datasets should appear in the list."""
    response = api_context.get("/datasets/list")
    assert response.ok
    data = response.json()
    assert uploaded_dataset["dataset_name"] in data.get("datasets", [])


def test_upload_requires_file(api_context: APIRequestContext) -> None:
    """POST /datasets/upload without a file should return 422."""
    response = api_context.post("/datasets/upload")
    assert response.status == 422


def test_upload_rejects_invalid_extension(api_context: APIRequestContext) -> None:
    """POST /datasets/upload with a non-CSV/XLSX file should return 400."""
    response = api_context.post(
        "/datasets/upload",
        multipart={
            "file": {
                "name": "invalid.txt",
                "mimeType": "text/plain",
                "buffer": b"not a dataset",
            }
        },
    )
    assert response.status == 400
    data = response.json()
    assert "Unsupported file type" in data.get("detail", "")


def test_upload_accepts_csv_and_returns_histogram(
    uploaded_dataset: dict,
) -> None:
    """Uploading a CSV should return a histogram payload."""
    assert uploaded_dataset.get("status") == "success"
    assert uploaded_dataset.get("document_count", 0) > 0
    assert uploaded_dataset.get("saved_count", 0) > 0

    histogram = uploaded_dataset.get("histogram", {})
    assert "bins" in histogram
    assert "counts" in histogram
    assert "min_length" in histogram
    assert "max_length" in histogram


def test_analyze_missing_dataset_returns_404(
    api_context: APIRequestContext,
) -> None:
    """POST /datasets/analyze should return 404 for missing datasets."""
    response = api_context.post(
        "/datasets/analyze",
        data={"dataset_name": "missing_dataset"},
    )
    assert response.status == 404


def test_analyze_uploaded_dataset_returns_stats(
    api_context: APIRequestContext,
    uploaded_dataset: dict,
    job_waiter,
) -> None:
    """POST /datasets/analyze should return stats for a known dataset."""
    response = api_context.post(
        "/datasets/analyze",
        data={"dataset_name": uploaded_dataset["dataset_name"]},
    )
    assert response.ok
    job = response.json()
    job_id = job.get("job_id")
    assert job_id, "Missing job_id in analyze response"
    job_status = job_waiter(
        job_id,
        poll_interval=job.get("poll_interval", 1.0),
        timeout_seconds=300.0,
    )
    assert job_status.get("status") == "completed", job_status.get("error")
    data = job_status.get("result", {})
    assert data.get("dataset_name") == uploaded_dataset["dataset_name"]
    assert data.get("analyzed_count", 0) > 0

    statistics = data.get("statistics", {})
    assert "total_documents" in statistics
    assert "mean_words_count" in statistics
    assert "median_words_count" in statistics
    assert "mean_avg_word_length" in statistics
    assert "mean_std_word_length" in statistics
