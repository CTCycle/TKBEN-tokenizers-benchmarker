"""
E2E tests for tokenizer API endpoints.
Covers /api/tokenizers/settings, /api/tokenizers/discover, /api/tokenizers/upload, and per-item deletion.
"""

import json
import os
import re
from pathlib import Path
from uuid import uuid4

import pytest
from playwright.sync_api import APIRequestContext, Page, expect
from server.common.path import TOKENIZERS_PATH


RUN_HF_DISCOVERY = os.getenv("E2E_RUN_HF_DISCOVERY", "").lower() in ("1", "true", "yes")
RUN_TOKENIZER_REPORT_FLOW = os.getenv("E2E_RUN_TOKENIZER_REPORT_FLOW", "").lower() in (
    "1",
    "true",
    "yes",
)
RUN_TOKENIZER_UI_LIFECYCLE = os.getenv(
    "E2E_RUN_TOKENIZER_UI_LIFECYCLE", ""
).lower() in ("1", "true", "yes")
TOKENIZER_RESTART_EXPECTED_NAME = os.getenv("E2E_TOKENIZER_EXPECTED_NAME", "").strip()
TOKENIZER_RESTART_STEM = "qa_t2_04_restart_long_identifier_0123456789abcdef"


###############################################################################
def _build_wordlevel_tokenizer_json(
    vocabulary_size: int = 3,
    token_prefix: str = "qa_t2_05_",
) -> bytes:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"[UNK]": 0}
    vocab.update(
        {f"{token_prefix}{token_id:04d}": token_id for token_id in range(1, vocabulary_size)}
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


def _assert_tokenizer_artifact_matches_vocabulary(
    artifact_path: Path,
    expected_payload: bytes,
) -> None:
    stored = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected = json.loads(expected_payload)
    assert stored["model"]["vocab"] == expected["model"]["vocab"]


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
    tokenizer_name = str(data.get("tokenizer_name", ""))
    try:
        assert data.get("status") == "success"
        assert tokenizer_name.startswith("CUSTOM_")
        assert data.get("is_compatible") is True
    finally:
        if tokenizer_name:
            deleted = api_context.delete(
                f"/api/tokenizers/delete?tokenizer_name={tokenizer_name}"
            )
            assert deleted.status == 200, deleted.text()


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

        screenshot_dir_value = os.getenv("TKBEN_TOKENIZER_QA_SCREENSHOT_DIR")
        if screenshot_dir_value:
            screenshot_dir = Path(screenshot_dir_value)
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            original_viewport = {"width": 1280, "height": 720}
            for width, height in (
                (1920, 1080),
                (1440, 900),
                (1024, 768),
                (390, 844),
            ):
                page.set_viewport_size({"width": width, "height": height})
                page.evaluate("window.scrollTo(0, 0)")
                dimensions = page.evaluate(
                    "() => ({viewport: window.innerWidth, document: document.documentElement.scrollWidth})"
                )
                assert dimensions["document"] <= dimensions["viewport"] + 1, (
                    f"Tokenizer report overflows at {width}x{height}: {dimensions}"
                )
                page.screenshot(
                    path=str(screenshot_dir / f"tokenizers-report-{width}x{height}.png"),
                    full_page=False,
                )
            page.set_viewport_size(original_viewport)
            page.evaluate("window.scrollTo(0, 0)")

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


###############################################################################
@pytest.mark.skipif(
    not RUN_TOKENIZER_UI_LIFECYCLE,
    reason="Set E2E_RUN_TOKENIZER_UI_LIFECYCLE=1 to enable.",
)
def test_custom_tokenizer_ui_lifecycle_and_responsive_states(
    api_context: APIRequestContext,
    page: Page,
    base_url: str,
) -> None:
    """Exercise the local upload UI, overwrite collision, and responsive states."""
    screenshot_dir_value = os.getenv("TKBEN_TOKENIZER_QA_SCREENSHOT_DIR")
    screenshot_dir = Path(screenshot_dir_value) if screenshot_dir_value else None
    if screenshot_dir:
        screenshot_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_name = f"CUSTOM_{TOKENIZER_RESTART_STEM}"
    stale_fixture = api_context.delete(
        f"/api/tokenizers/delete?tokenizer_name={tokenizer_name}"
    )
    assert stale_fixture.status in (200, 404), stale_fixture.text()

    def fulfill_empty_discovery(route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"count": 0, "items": []}),
        )

    page.route("**/api/tokenizers/discover*", fulfill_empty_discovery)
    page.goto(f"{base_url}/tokenizers", wait_until="domcontentloaded")
    # The empty catalogue and a controlled slow response make the transient
    # states observable without contacting the external provider.
    page.get_by_label("Search tokenizers", exact=True).fill(
        "tkben-t2-04-no-catalog-match"
    )
    page.get_by_label("Source", exact=True).select_option("custom")
    expect(page.get_by_text("No tokenizers match the current filters.")).to_be_visible()
    if screenshot_dir:
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(
            path=str(screenshot_dir / "tokenizers-empty-390x844.png"),
            full_page=False,
        )

    pending_catalog_routes = []

    def hold_catalog(route) -> None:
        pending_catalog_routes.append(route)

    page.set_viewport_size({"width": 1024, "height": 768})
    page.route("**/api/tokenizers/list*", hold_catalog)
    try:
        page.reload(wait_until="domcontentloaded")
        expect(page.get_by_text("Loading tokenizers...", exact=True)).to_be_visible(
            timeout=10_000
        )
        assert pending_catalog_routes, "Tokenizer list request was not intercepted."
        if screenshot_dir:
            page.screenshot(
                path=str(screenshot_dir / "tokenizers-loading-1024x768.png"),
                full_page=False,
            )
        for width, height in (
            (1920, 1080),
            (1440, 900),
            (1024, 768),
            (390, 844),
        ):
            page.set_viewport_size({"width": width, "height": height})
            page.evaluate("window.scrollTo(0, 0)")
            dimensions = page.evaluate(
                "() => ({viewport: window.innerWidth, document: document.documentElement.scrollWidth})"
            )
            assert dimensions["document"] <= dimensions["viewport"] + 1, (
                f"Loading tokenizer route overflows at {width}x{height}: {dimensions}"
            )
    finally:
        for route in pending_catalog_routes:
            route.continue_()
        page.unroute("**/api/tokenizers/list*", hold_catalog)
    expect(page.get_by_text("Loading tokenizers...", exact=True)).to_be_hidden(
        timeout=10_000
    )

    def fail_catalog(route) -> None:
        route.fulfill(
            status=503,
            content_type="application/json",
            body=json.dumps({"detail": "Controlled Tokenizers catalogue error"}),
        )

    page.route("**/api/tokenizers/list*", fail_catalog)
    page.reload(wait_until="domcontentloaded")
    expect(page.get_by_role("alert")).to_contain_text(
        "Controlled Tokenizers catalogue error", timeout=10_000
    )
    for width, height in (
        (1920, 1080),
        (1440, 900),
        (1024, 768),
        (390, 844),
    ):
        page.set_viewport_size({"width": width, "height": height})
        page.evaluate("window.scrollTo(0, 0)")
        dimensions = page.evaluate(
            "() => ({viewport: window.innerWidth, document: document.documentElement.scrollWidth})"
        )
        assert dimensions["document"] <= dimensions["viewport"] + 1, (
            f"Error tokenizer route overflows at {width}x{height}: {dimensions}"
        )
        if screenshot_dir and width == 1024:
            page.screenshot(
                path=str(screenshot_dir / "tokenizers-error-1024x768.png"),
                full_page=False,
            )
    page.unroute("**/api/tokenizers/list*", fail_catalog)
    page.reload(wait_until="domcontentloaded")
    expect(page.get_by_role("alert")).to_have_count(0, timeout=10_000)
    page.get_by_label("Search tokenizers", exact=True).fill(
        "tkben-t2-04-no-catalog-match"
    )
    page.get_by_label("Source", exact=True).select_option("custom")
    expect(page.get_by_text("No tokenizers match the current filters.")).to_be_visible(
        timeout=10_000
    )

    add_button = page.get_by_role("button", name="Add tokenizer", exact=True)
    for width, height in (
        (1920, 1080),
        (1440, 900),
        (1024, 768),
        (390, 844),
    ):
        page.set_viewport_size({"width": width, "height": height})
        page.evaluate("window.scrollTo(0, 0)")
        dimensions = page.evaluate(
            "() => ({viewport: window.innerWidth, document: document.documentElement.scrollWidth})"
        )
        assert dimensions["document"] <= dimensions["viewport"] + 1, (
            f"Empty tokenizer route overflows at {width}x{height}: {dimensions}"
        )

        add_button.click()
        dialog = page.get_by_role("dialog", name="Tokenizer Manager")
        expect(dialog).to_be_visible()
        bounds = dialog.bounding_box()
        assert bounds is not None
        assert bounds["x"] >= -1 and bounds["x"] + bounds["width"] <= width + 1, (
            f"Tokenizer manager is horizontally clipped at {width}x{height}: {bounds}"
        )
        assert bounds["y"] >= -1 and bounds["y"] + bounds["height"] <= height + 1, (
            f"Tokenizer manager is vertically clipped at {width}x{height}: {bounds}"
        )
        if screenshot_dir and width == 390:
            page.screenshot(
                path=str(screenshot_dir / "tokenizers-manager-390x844.png"),
                full_page=False,
            )

        discover_tab = page.get_by_role("tab", name="Discover", exact=True)
        discover_tab.press("ArrowRight")
        expect(page.get_by_role("tab", name="Add by name", exact=True)).to_have_attribute(
            "aria-selected", "true"
        )
        page.get_by_role("tab", name="Add by name", exact=True).press("End")
        upload_tab = page.get_by_role("tab", name="Upload JSON", exact=True)
        expect(upload_tab).to_have_attribute("aria-selected", "true")
        page.keyboard.press("Escape")
        expect(dialog).to_be_hidden()
        expect(add_button).to_be_focused()

    page.set_viewport_size({"width": 1440, "height": 900})
    page.get_by_label("Search tokenizers", exact=True).fill(TOKENIZER_RESTART_STEM)
    page.get_by_label("Source", exact=True).select_option("custom")
    expect(page.get_by_text("No tokenizers match the current filters.")).to_be_visible(
        timeout=10_000
    )
    filename = f"{TOKENIZER_RESTART_STEM}.json"
    artifact_path = Path(TOKENIZERS_PATH) / tokenizer_name.replace("/", "__") / "tokenizer.json"
    keep_for_restart = os.getenv("E2E_TOKENIZER_KEEP_FOR_RESTART", "").lower() in (
        "1",
        "true",
        "yes",
    )
    completed = False
    try:
        first_payload = _build_wordlevel_tokenizer_json(
            vocabulary_size=5,
            token_prefix="qa_t2_04_original_",
        )
        second_payload = _build_wordlevel_tokenizer_json(
            vocabulary_size=7,
            token_prefix="qa_t2_04_replacement_",
        )
        for payload in (first_payload, second_payload):
            add_button.click()
            page.get_by_role("tab", name="Upload JSON", exact=True).click()
            with page.expect_response(
                lambda response: (
                    response.request.method == "POST"
                    and response.url.endswith("/api/tokenizers/upload")
                ),
                timeout=30_000,
            ) as upload_response_info:
                page.locator("#tokenizer-json-upload").set_input_files(
                    {
                        "name": filename,
                        "mimeType": "application/json",
                        "buffer": payload,
                    }
                )
            upload_response = upload_response_info.value
            assert upload_response.status == 200, upload_response.text()
            assert upload_response.json().get("tokenizer_name") == tokenizer_name
            expect(page.get_by_text(tokenizer_name, exact=True)).to_be_visible(
                timeout=30_000
            )
            _assert_tokenizer_artifact_matches_vocabulary(artifact_path, payload)

        listed = api_context.get("/api/tokenizers/list")
        assert listed.ok, listed.text()
        matching_entries = [
            item
            for item in listed.json().get("tokenizers", [])
            if item.get("tokenizer_name") == tokenizer_name
        ]
        assert len(matching_entries) == 1, matching_entries
        _assert_tokenizer_artifact_matches_vocabulary(artifact_path, second_payload)

        for width, height in (
            (1920, 1080),
            (1440, 900),
            (1024, 768),
            (390, 844),
        ):
            page.set_viewport_size({"width": width, "height": height})
            page.evaluate("window.scrollTo(0, 0)")
            dimensions = page.evaluate(
                "() => ({viewport: window.innerWidth, document: document.documentElement.scrollWidth})"
            )
            assert dimensions["document"] <= dimensions["viewport"] + 1, (
                f"Long tokenizer identifier overflows at {width}x{height}: {dimensions}"
            )
            expect(page.get_by_text(tokenizer_name, exact=True)).to_be_visible()
            if screenshot_dir:
                page.screenshot(
                    path=str(
                        screenshot_dir / f"tokenizers-populated-{width}x{height}.png"
                    ),
                    full_page=False,
                )
        completed = True
    finally:
        if not (completed and keep_for_restart):
            deleted = api_context.delete(
                f"/api/tokenizers/delete?tokenizer_name={tokenizer_name}"
            )
            assert deleted.status in (200, 404), deleted.text()

    page.set_viewport_size({"width": 1280, "height": 720})


###############################################################################
@pytest.mark.skipif(
    not TOKENIZER_RESTART_EXPECTED_NAME,
    reason="Set E2E_TOKENIZER_EXPECTED_NAME after the launcher restart.",
)
def test_custom_tokenizer_survives_restart_and_ui_deletion(
    api_context: APIRequestContext,
    job_waiter,
    page: Page,
    base_url: str,
) -> None:
    """Verify a custom upload survives restart, runs, and deletes from the UI."""
    tokenizer_name = TOKENIZER_RESTART_EXPECTED_NAME
    assert tokenizer_name == f"CUSTOM_{TOKENIZER_RESTART_STEM}"
    artifact_path = Path(TOKENIZERS_PATH) / tokenizer_name.replace("/", "__") / "tokenizer.json"

    listed = api_context.get("/api/tokenizers/list")
    assert listed.ok, listed.text()
    matching_entries = [
        item
        for item in listed.json().get("tokenizers", [])
        if item.get("tokenizer_name") == tokenizer_name
    ]
    assert len(matching_entries) == 1, matching_entries
    assert artifact_path.is_file()

    page.route(
        "**/api/tokenizers/discover*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"count": 0, "items": []}),
        ),
    )
    page.goto(f"{base_url}/tokenizers", wait_until="domcontentloaded")
    expect(page.get_by_text(tokenizer_name, exact=True)).to_be_visible(timeout=30_000)

    report_button = page.get_by_role(
        "button",
        name=f"Generate or open tokenizer report for {tokenizer_name}",
        exact=True,
    )
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
    job_status = job_waiter(
        str(generate_job.get("job_id", "")),
        poll_interval=generate_job.get("poll_interval", 1.0),
        timeout_seconds=300.0,
    )
    assert job_status.get("status") == "completed", job_status.get("error")
    report_payload = job_status.get("result", {})
    report_id = int(report_payload["report_id"])
    assert report_payload.get("vocabulary_size") == 7
    latest = api_context.get(
        f"/api/tokenizers/reports/latest?tokenizer_name={tokenizer_name}"
    )
    assert latest.ok, latest.text()
    assert latest.json().get("report_id") == report_id
    assert latest.json().get("vocabulary_size") == 7
    vocabulary = page.get_by_role("table", name="Tokenizer vocabulary preview")
    expect(vocabulary).to_contain_text("qa_t2_04_replacement_0001", timeout=30_000)

    page.on("dialog", lambda dialog: dialog.accept())
    page.get_by_role(
        "button", name=f"Remove {tokenizer_name}", exact=True
    ).click()
    expect(page.get_by_text(tokenizer_name, exact=True)).to_have_count(0, timeout=30_000)
    refreshed = api_context.get("/api/tokenizers/list")
    assert refreshed.ok, refreshed.text()
    assert tokenizer_name not in {
        str(item.get("tokenizer_name"))
        for item in refreshed.json().get("tokenizers", [])
    }
    assert api_context.get(
        f"/api/tokenizers/reports/latest?tokenizer_name={tokenizer_name}"
    ).status == 404
    assert not artifact_path.exists()
