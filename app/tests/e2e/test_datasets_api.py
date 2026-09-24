"""
E2E tests for dataset API endpoints.
Covers /api/datasets/list, /api/datasets/upload, and /api/datasets/analyze.
"""

from io import BytesIO
from uuid import uuid4

from openpyxl import Workbook
from playwright.sync_api import APIRequestContext

###############################################################################
def test_list_datasets_includes_uploaded_dataset(
    api_context: APIRequestContext,
    uploaded_dataset: dict,
) -> None:
    """Uploaded datasets should appear in the list."""
    response = api_context.get("/api/datasets/list")
    assert response.ok
    data = response.json()
    previews = data.get("datasets", [])
    assert any(
        item.get("dataset_name") == uploaded_dataset["dataset_name"]
        for item in previews
        if isinstance(item, dict)
    )

###############################################################################
def test_upload_rejects_invalid_extension(api_context: APIRequestContext) -> None:
    """POST /api/datasets/upload with a non-CSV/XLSX file should return 400."""
    response = api_context.post(
        "/api/datasets/upload",
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

###############################################################################
def test_upload_rejects_legacy_xls_file(api_context: APIRequestContext) -> None:
    """The upload contract is limited to supported CSV and XLSX formats."""
    response = api_context.post(
        "/api/datasets/upload",
        multipart={
            "file": {
                "name": "legacy.xls",
                "mimeType": "application/vnd.ms-excel",
                "buffer": b"not an XLSX workbook",
            }
        },
    )
    assert response.status == 400
    assert "Use .csv or .xlsx" in response.json().get("detail", "")

###############################################################################
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

###############################################################################
def test_upload_accepts_xlsx_and_returns_histogram(
    api_context: APIRequestContext,
    job_waiter,
) -> None:
    """Uploading an XLSX file should persist its text rows and histogram."""
    dataset_name = f"custom/e2e_xlsx_import_{uuid4().hex}"
    filename = f"{dataset_name.removeprefix('custom/')}.xlsx"
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(["text", "source"])
    worksheet.append(["First workbook document.", "synthetic"])
    worksheet.append(["Second document includes UTF-8 café.", "synthetic"])
    worksheet.append(["Third document exercises Excel import.", "synthetic"])
    buffer = BytesIO()
    workbook.save(buffer)

    response = api_context.post(
        "/api/datasets/upload",
        multipart={
            "file": {
                "name": filename,
                "mimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "buffer": buffer.getvalue(),
            }
        },
    )
    assert response.status == 202, response.text()
    job = response.json()
    job_id = job.get("job_id")
    assert job_id, "Missing job_id in XLSX upload response"
    job_status = job_waiter(
        job_id,
        poll_interval=job.get("poll_interval", 1.0),
        timeout_seconds=300.0,
    )
    assert job_status.get("status") == "completed", job_status.get("error")

    result = job_status.get("result", {})
    assert result.get("dataset_name") == dataset_name
    assert result.get("document_count") == 3
    assert result.get("saved_count") == 3
    histogram = result.get("histogram", {})
    assert histogram.get("counts")
    assert histogram.get("min_length") is not None
    assert histogram.get("max_length") is not None

###############################################################################
def test_analyze_missing_dataset_returns_404(
    api_context: APIRequestContext,
) -> None:
    """POST /api/datasets/analyze should return 404 for missing datasets."""
    response = api_context.post(
        "/api/datasets/analyze",
        data={"dataset_name": "missing_dataset"},
    )
    assert response.status == 404

###############################################################################
def test_analyze_uploaded_dataset_returns_stats(
    api_context: APIRequestContext,
    uploaded_dataset: dict,
    job_waiter,
) -> None:
    """POST /api/datasets/analyze should return stats for a known dataset."""
    response = api_context.post(
        "/api/datasets/analyze",
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
    assert data.get("document_count", 0) > 0
    document_histogram = data.get("document_length_histogram", {})
    word_histogram = data.get("word_length_histogram", {})
    assert "bins" in document_histogram
    assert "counts" in document_histogram
    assert "min_length" in document_histogram
    assert "max_length" in document_histogram
    assert "bins" in word_histogram
    assert "counts" in word_histogram
    assert isinstance(data.get("most_common_words", []), list)
    assert isinstance(data.get("least_common_words", []), list)
