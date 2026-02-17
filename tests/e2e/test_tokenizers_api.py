"""
E2E tests for tokenizer API endpoints.
Covers /tokenizers/settings, /tokenizers/scan, /tokenizers/upload, /tokenizers/custom.
"""
import json
import os

import pytest
from playwright.sync_api import APIRequestContext


RUN_HF_SCAN = os.getenv("E2E_RUN_HF_SCAN", "").lower() in ("1", "true", "yes")
RUN_TOKENIZER_REPORT_FLOW = os.getenv("E2E_RUN_TOKENIZER_REPORT_FLOW", "").lower() in ("1", "true", "yes")


def _build_wordlevel_tokenizer_json() -> bytes:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"[UNK]": 0, "hello": 1, "world": 2}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    if hasattr(tokenizer, "to_str"):
        payload = tokenizer.to_str()
    else:
        payload = tokenizer.to_json()

    if isinstance(payload, dict):
        payload = json.dumps(payload)
    return str(payload).encode("utf-8")


def test_get_tokenizer_settings(api_context: APIRequestContext) -> None:
    """GET /tokenizers/settings should return configured scan limits."""
    response = api_context.get("/tokenizers/settings")
    assert response.ok
    data = response.json()
    assert "default_scan_limit" in data
    assert "max_scan_limit" in data
    assert "min_scan_limit" in data
    assert data["min_scan_limit"] <= data["default_scan_limit"] <= data["max_scan_limit"]


@pytest.mark.skipif(not RUN_HF_SCAN, reason="Set E2E_RUN_HF_SCAN=1 to enable.")
def test_scan_tokenizers_returns_identifiers(
    api_context: APIRequestContext,
) -> None:
    """GET /tokenizers/scan should return at least one tokenizer identifier."""
    response = api_context.get("/tokenizers/scan?limit=1")
    assert response.ok
    data = response.json()
    assert data.get("status") == "success"
    assert isinstance(data.get("identifiers"), list)
    assert data.get("count", 0) >= 1


def test_upload_rejects_invalid_extension(api_context: APIRequestContext) -> None:
    """POST /tokenizers/upload should reject non-json files."""
    response = api_context.post(
        "/tokenizers/upload",
        multipart={
            "file": {
                "name": "tokenizer.txt",
                "mimeType": "text/plain",
                "buffer": b"not json",
            }
        },
    )
    assert response.status == 400
    data = response.json()
    assert "File must be a .json file" in data.get("detail", "")


def test_upload_rejects_invalid_json(api_context: APIRequestContext) -> None:
    """POST /tokenizers/upload should reject invalid tokenizer JSON."""
    response = api_context.post(
        "/tokenizers/upload",
        multipart={
            "file": {
                "name": "tokenizer.json",
                "mimeType": "application/json",
                "buffer": b"{not valid json}",
            }
        },
    )
    assert response.status == 400
    data = response.json()
    assert "Failed to load tokenizer" in data.get("detail", "")


def test_upload_accepts_valid_tokenizer_json(api_context: APIRequestContext) -> None:
    """POST /tokenizers/upload should accept a valid tokenizer.json file."""
    payload = _build_wordlevel_tokenizer_json()
    response = api_context.post(
        "/tokenizers/upload",
        multipart={
            "file": {
                "name": "tokenizer.json",
                "mimeType": "application/json",
                "buffer": payload,
            }
        },
    )
    assert response.ok
    data = response.json()
    assert data.get("status") == "success"
    assert data.get("tokenizer_name", "").startswith("CUSTOM_")
    assert data.get("is_compatible") is True

    cleanup = api_context.delete("/tokenizers/custom")
    assert cleanup.ok


def test_clear_custom_tokenizers(api_context: APIRequestContext) -> None:
    """DELETE /tokenizers/custom should return a success message."""
    response = api_context.delete("/tokenizers/custom")
    assert response.ok
    data = response.json()
    assert data.get("status") == "success"


@pytest.mark.skipif(
    not RUN_TOKENIZER_REPORT_FLOW,
    reason="Set E2E_RUN_TOKENIZER_REPORT_FLOW=1 to enable.",
)
def test_tokenizer_report_flow_supports_paged_vocabulary(
    api_context: APIRequestContext,
    job_waiter,
) -> None:
    list_response = api_context.get("/tokenizers/list")
    assert list_response.ok
    list_payload = list_response.json()
    tokenizers = list_payload.get("tokenizers", [])
    if not tokenizers:
        pytest.skip("No downloaded tokenizers available for report flow test.")

    tokenizer_name = str(tokenizers[0].get("tokenizer_name", "")).strip()
    if not tokenizer_name:
        pytest.skip("No valid tokenizer_name found in /tokenizers/list response.")

    latest_response = api_context.get(
        f"/tokenizers/reports/latest?tokenizer_name={tokenizer_name}"
    )
    if latest_response.status == 404:
        generate_response = api_context.post(
            "/tokenizers/reports/generate",
            data={"tokenizer_name": tokenizer_name},
        )
        assert generate_response.ok, generate_response.text()
        generate_job = generate_response.json()
        job_id = generate_job.get("job_id")
        assert job_id
        job_status = job_waiter(
            job_id,
            poll_interval=generate_job.get("poll_interval", 1.0),
            timeout_seconds=300.0,
        )
        assert job_status.get("status") == "completed", job_status.get("error")
        report_payload = job_status.get("result", {})
    else:
        assert latest_response.ok, latest_response.text()
        report_payload = latest_response.json()

    report_id = int(report_payload.get("report_id"))

    page_one_response = api_context.get(
        f"/tokenizers/reports/{report_id}/vocabulary?offset=0&limit=200"
    )
    assert page_one_response.ok, page_one_response.text()
    page_one = page_one_response.json()

    assert page_one.get("report_id") == report_id
    assert page_one.get("offset") == 0
    assert page_one.get("limit") == 200
    assert isinstance(page_one.get("total"), int)
    assert isinstance(page_one.get("items"), list)
    assert len(page_one.get("items", [])) <= 200

    second_offset = int(page_one.get("limit", 200))
    page_two_response = api_context.get(
        f"/tokenizers/reports/{report_id}/vocabulary?offset={second_offset}&limit=200"
    )
    assert page_two_response.ok, page_two_response.text()
    page_two = page_two_response.json()

    assert page_two.get("report_id") == report_id
    assert page_two.get("offset") == second_offset
    assert page_two.get("limit") == 200
    assert page_two.get("total") == page_one.get("total")
    assert len(page_two.get("items", [])) <= 200
