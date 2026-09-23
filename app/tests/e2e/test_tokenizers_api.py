"""
E2E tests for tokenizer API endpoints.
Covers /api/tokenizers/settings, /api/tokenizers/discover, /api/tokenizers/upload, and per-item deletion.
"""

import json
import os
import re
from uuid import uuid4

import pytest
from playwright.sync_api import APIRequestContext, Page, expect


RUN_HF_DISCOVERY = os.getenv("E2E_RUN_HF_DISCOVERY", "").lower() in ("1", "true", "yes")
RUN_TOKENIZER_REPORT_FLOW = os.getenv("E2E_RUN_TOKENIZER_REPORT_FLOW", "").lower() in (
    "1",
    "true",
    "yes",
)


###############################################################################
def _build_wordlevel_tokenizer_json(vocabulary_size: int = 3) -> bytes:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"[UNK]": 0}
    vocab.update(
        {f"qa_t2_05_{token_id:04d}": token_id for token_id in range(1, vocabulary_size)}
    )
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    if hasattr(tokenizer, "to_str"):
        payload = tokenizer.to_str()
    else:
        payload = tokenizer.to_json()

    if isinstance(payload, dict):
        payload = json.dumps(payload)
    return str(payload).encode("utf-8")


###############################################################################
def test_get_tokenizer_settings(api_context: APIRequestContext) -> None:
    """GET /api/tokenizers/settings should return configured discovery limits."""
    response = api_context.get("/api/tokenizers/settings")
    assert response.ok
    data = response.json()
    assert "default_discovery_limit" in data
    assert "max_discovery_limit" in data
    assert "max_discovery_candidates" in data
    assert "metadata_candidate_multiplier" in data
    assert 1 <= data["default_discovery_limit"] <= data["max_discovery_limit"]


###############################################################################
@pytest.mark.skipif(
    not RUN_HF_DISCOVERY, reason="Set E2E_RUN_HF_DISCOVERY=1 to enable."
)
def test_discover_tokenizers_returns_bounded_structured_catalog(
    api_context: APIRequestContext,
) -> None:
    """GET /api/tokenizers/discover should return bounded structured results."""
    response = api_context.get(
        "/api/tokenizers/discover?search=bert&limit=5&pipeline_tag=fill-mask&sort=downloads"
    )
    assert response.ok
    data = response.json()
    items = data.get("items")
    assert isinstance(items, list)
    assert data.get("count") == len(items)
    assert len(items) <= 5
    assert all(
        isinstance(item.get("identifier"), str) and item["identifier"] for item in items
    )
    assert all("vocabulary_size" in item for item in items)


###############################################################################
@pytest.mark.skipif(
    not RUN_HF_DISCOVERY, reason="Set E2E_RUN_HF_DISCOVERY=1 to enable."
)
def test_discover_tokenizers_supports_empty_result(
    api_context: APIRequestContext,
) -> None:
    response = api_context.get(
        "/api/tokenizers/discover?search=tkben-no-such-repository-8f72&limit=5"
    )
    assert response.ok
    assert response.json().get("items") == []


###############################################################################
def test_upload_rejects_invalid_extension(api_context: APIRequestContext) -> None:
    """POST /api/tokenizers/upload should reject non-json files."""
    response = api_context.post(
        "/api/tokenizers/upload",
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


###############################################################################
def test_upload_rejects_invalid_json(api_context: APIRequestContext) -> None:
    """POST /api/tokenizers/upload should reject invalid tokenizer JSON."""
    response = api_context.post(
        "/api/tokenizers/upload",
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


###############################################################################
def test_upload_accepts_valid_tokenizer_json(api_context: APIRequestContext) -> None:
    """POST /api/tokenizers/upload should accept a valid tokenizer.json file."""
    payload = _build_wordlevel_tokenizer_json()
    response = api_context.post(
        "/api/tokenizers/upload",
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


###############################################################################
def test_custom_tokenizer_can_be_deleted_and_repeated_delete_is_not_found(
    api_context: APIRequestContext,
    tiny_tokenizer_json: bytes,
) -> None:
    """Per-item deletion removes the durable custom tokenizer."""
    stem = f"qa_delete_custom_{uuid4().hex[:8]}"
    upload = api_context.post(
        "/api/tokenizers/upload",
        multipart={
            "file": {
                "name": f"{stem}.json",
                "mimeType": "application/json",
                "buffer": tiny_tokenizer_json,
            }
        },
    )
    assert upload.ok, upload.text()
    tokenizer_name = upload.json()["tokenizer_name"]

    listed = api_context.get("/api/tokenizers/list")
    assert listed.ok
    assert tokenizer_name in {
        str(item.get("tokenizer_name")) for item in listed.json().get("tokenizers", [])
    }

    deleted = api_context.delete(
        f"/api/tokenizers/delete?tokenizer_name={tokenizer_name}"
    )
    assert deleted.status == 200, deleted.text()

    refreshed = api_context.get("/api/tokenizers/list")
    assert refreshed.ok
    assert tokenizer_name not in {
        str(item.get("tokenizer_name"))
        for item in refreshed.json().get("tokenizers", [])
    }

    repeated = api_context.delete(
        f"/api/tokenizers/delete?tokenizer_name={tokenizer_name}"
    )
    assert repeated.status == 404


###############################################################################
@pytest.mark.skipif(
    not RUN_TOKENIZER_REPORT_FLOW,
    reason="Set E2E_RUN_TOKENIZER_REPORT_FLOW=1 to enable.",
)
def test_tokenizer_report_flow_supports_paged_vocabulary(
    api_context: APIRequestContext,
    job_waiter,
    page: Page,
    base_url: str,
) -> None:
    stem = f"qa_t2_05_{uuid4().hex[:8]}"
    upload_response = api_context.post(
        "/api/tokenizers/upload",
        multipart={
            "file": {
                "name": f"{stem}.json",
                "mimeType": "application/json",
                "buffer": _build_wordlevel_tokenizer_json(vocabulary_size=1207),
            }
        },
    )
    assert upload_response.ok, upload_response.text()
    tokenizer_name = str(upload_response.json()["tokenizer_name"])
    assert tokenizer_name == f"CUSTOM_{stem}"

    try:
        latest_url = f"/api/tokenizers/reports/latest?tokenizer_name={tokenizer_name}"
        assert api_context.get(latest_url).status == 404

        browser_errors: list[str] = []
        browser_http_errors: list[tuple[int, str]] = []
        page.on("pageerror", lambda error: browser_errors.append(str(error)))
        page.on(
            "console",
            lambda message: (
                browser_errors.append(message.text) if message.type == "error" else None
            ),
        )
        page.on(
            "response",
            lambda response: (
                browser_http_errors.append((response.status, response.url))
                if response.status >= 400
                else None
            ),
        )
        page.goto(f"{base_url}/tokenizers")

        report_button = page.get_by_role(
            "button",
            name=f"Generate or open tokenizer report for {tokenizer_name}",
            exact=True,
        )
        expect(report_button).to_be_visible(timeout=30_000)
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.endswith("/api/tokenizers/reports/generate")
            ),
            timeout=30_000,
        ) as generate_response_info:
            report_button.click()

        generate_response = generate_response_info.value
        assert generate_response.status == 202, generate_response.text()
        generate_job = generate_response.json()
        job_id = str(generate_job.get("job_id", ""))
        assert job_id
        job_status = job_waiter(
            job_id,
            poll_interval=generate_job.get("poll_interval", 1.0),
            timeout_seconds=300.0,
        )
        assert job_status.get("status") == "completed", job_status.get("error")

        report_payload = job_status.get("result", {})
        report_id = int(report_payload["report_id"])
        assert report_payload.get("vocabulary_size") == 1207

        latest_response = api_context.get(latest_url)
        assert latest_response.ok, latest_response.text()
        latest_payload = latest_response.json()
        assert latest_payload.get("report_id") == report_id
        assert latest_payload.get("vocabulary_size") == 1207

        expected_pages = ((0, 500), (500, 500), (1000, 207))
        for offset, expected_count in expected_pages:
            vocabulary_response = api_context.get(
                f"/api/tokenizers/reports/{report_id}/vocabulary"
                f"?offset={offset}&limit=500"
            )
            assert vocabulary_response.ok, vocabulary_response.text()
            vocabulary_page = vocabulary_response.json()
            assert vocabulary_page.get("report_id") == report_id
            assert vocabulary_page.get("offset") == offset
            assert vocabulary_page.get("limit") == 500
            assert vocabulary_page.get("total") == 1207
            items = vocabulary_page.get("items", [])
            assert len(items) == expected_count
            assert [item["token_id"] for item in items] == list(
                range(offset, offset + expected_count)
            )

        dashboard = page.get_by_role("region", name="Tokenizers Dashboard")
        report_description = dashboard.locator(".panel-description").first
        report_label = f"Report {report_id} for {tokenizer_name}"
        expect(report_description).to_have_text(report_label, timeout=30_000)
        report_table = dashboard.get_by_role("table").first
        expect(
            report_table.get_by_role(
                "row",
                name=re.compile(r"^Vocabulary size 1[,\.\u00a0\u202f ]?207$"),
            )
        ).to_be_visible()

        vocabulary_panel = page.get_by_role("complementary", name="Vocabulary Preview")
        vocabulary_table = vocabulary_panel.get_by_role(
            "table", name="Tokenizer vocabulary preview"
        )
        page_summary = vocabulary_panel.locator(
            ".tokenizer-vocabulary-footer .panel-description"
        )
        previous_button = vocabulary_panel.get_by_role(
            "button", name="Previous", exact=True
        )
        next_button = vocabulary_panel.get_by_role("button", name="Next", exact=True)

        summary_separator = r"[,\.\u00a0\u202f ]?"
        expect(page_summary).to_have_text(
            re.compile(rf"^Showing 1-500 of 1{summary_separator}207$"),
            timeout=30_000,
        )
        expect(previous_button).to_be_disabled()
        expect(next_button).to_be_enabled()
        expect(vocabulary_table).to_contain_text("qa_t2_05_0001")
        next_button.click()
        expect(page_summary).to_have_text(
            re.compile(
                rf"^Showing 501-1{summary_separator}000 of 1{summary_separator}207$"
            )
        )
        expect(vocabulary_table).to_contain_text("qa_t2_05_0500")
        next_button.click()
        expect(page_summary).to_have_text(
            re.compile(
                rf"^Showing 1{summary_separator}001-1{summary_separator}207 "
                rf"of 1{summary_separator}207$"
            )
        )
        expect(vocabulary_table).to_contain_text("qa_t2_05_1206")
        expect(next_button).to_be_disabled()
        expect(previous_button).to_be_enabled()
        previous_button.click()
        expect(page_summary).to_have_text(
            re.compile(
                rf"^Showing 501-1{summary_separator}000 of 1{summary_separator}207$"
            )
        )

        screenshot_path = os.getenv("TKBEN_T2_05_SCREENSHOT")
        if screenshot_path:
            page.screenshot(path=screenshot_path, full_page=True)

        page.reload()
        report_button = page.get_by_role(
            "button",
            name=f"Generate or open tokenizer report for {tokenizer_name}",
            exact=True,
        )
        expect(report_button).to_be_visible(timeout=30_000)
        reloaded_latest = api_context.get(latest_url)
        assert reloaded_latest.ok, reloaded_latest.text()
        assert reloaded_latest.json().get("report_id") == report_id
        report_button.click()
        expect(report_description).to_have_text(report_label, timeout=30_000)
        expect(page_summary).to_have_text(
            re.compile(rf"^Showing 1-500 of 1{summary_separator}207$")
        )
        expect(vocabulary_table).to_contain_text("qa_t2_05_0001")
        expected_missing_report = [
            (status, url)
            for status, url in browser_http_errors
            if status == 404 and "/api/tokenizers/reports/latest?" in url
        ]
        unexpected_http_errors = [
            (status, url)
            for status, url in browser_http_errors
            if (status, url) not in expected_missing_report
        ]
        assert len(expected_missing_report) == 1, browser_http_errors
        assert unexpected_http_errors == [], unexpected_http_errors
        assert len(browser_errors) == len(expected_missing_report), browser_errors
        assert all("404 (Not Found)" in error for error in browser_errors)
    finally:
        delete_response = api_context.delete(
            f"/api/tokenizers/delete?tokenizer_name={tokenizer_name}"
        )
        assert delete_response.status == 200, delete_response.text()
        assert (
            api_context.get(
                f"/api/tokenizers/reports/latest?tokenizer_name={tokenizer_name}"
            ).status
            == 404
        )
