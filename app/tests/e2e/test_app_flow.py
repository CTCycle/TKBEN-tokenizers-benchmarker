"""
E2E tests for UI navigation and page rendering.
Targets datasets, tokenizers, and cross benchmark workflows.
"""

import re
from urllib.parse import quote
from uuid import uuid4

from playwright.sync_api import Page, expect
from playwright.sync_api import APIRequestContext


def _upload_dataset_for_ui_test(
    api_context: APIRequestContext,
    job_waiter,
    stem: str,
) -> str:
    filename = f"{stem}.csv"
    dataset_name = f"custom/{stem}"
    csv_content = "text\nhello world\nqa row\n"
    response = api_context.post(
        "/api/datasets/upload",
        multipart={
            "file": {
                "name": filename,
                "mimeType": "text/csv",
                "buffer": csv_content.encode("utf-8"),
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
    return dataset_name


class TestAppShell:
    """Tests for core layout and routing."""

    def test_root_redirects_to_dataset(self, page: Page, base_url: str) -> None:
        """The root route should redirect to the dataset page."""
        page.goto(base_url)
        expect(page).to_have_url(re.compile(r".*/dataset/?$"))
        expect(page.get_by_text("Dataset Usage")).to_be_visible()

    def test_sidebar_links_are_visible(self, page: Page, base_url: str) -> None:
        """Sidebar navigation should expose primary sections."""
        page.goto(f"{base_url}/dataset")
        expect(page.get_by_role("button", name="Datasets")).to_be_visible()
        expect(page.get_by_role("button", name="Tokenizers")).to_be_visible()
        expect(page.get_by_role("button", name="Cross Benchmark")).to_be_visible()

    def test_unknown_route_redirects_to_dataset(
        self, page: Page, base_url: str
    ) -> None:
        """Unknown routes should redirect back to the dataset page."""
        page.goto(f"{base_url}/does-not-exist")
        expect(page).to_have_url(re.compile(r".*/dataset/?$"))


class TestDatasetPage:
    """Tests for dataset page UI elements."""

    def test_dataset_page_panels_render(self, page: Page, base_url: str) -> None:
        """Dataset page should show the main layout panels."""
        page.goto(f"{base_url}/dataset")
        expect(page.get_by_text("Dataset Usage")).to_be_visible()
        expect(page.get_by_text("Dataset Preview")).to_be_visible()
        expect(page.get_by_role("button", name="Add dataset")).to_be_visible()

    def test_row_click_loads_latest_report_for_selected_dataset(
        self,
        page: Page,
        base_url: str,
        api_context: APIRequestContext,
        job_waiter,
    ) -> None:
        """Clicking a dataset row should fetch latest report and update dashboard state."""
        with_report = f"qa_row_report_{uuid4().hex[:8]}"
        without_report = f"qa_row_noreport_{uuid4().hex[:8]}"
        with_report_dataset = _upload_dataset_for_ui_test(
            api_context=api_context,
            job_waiter=job_waiter,
            stem=with_report,
        )
        without_report_dataset = _upload_dataset_for_ui_test(
            api_context=api_context,
            job_waiter=job_waiter,
            stem=without_report,
        )

        analyze_response = api_context.post(
            "/api/datasets/analyze",
            data={"dataset_name": with_report_dataset},
        )
        assert analyze_response.ok, (
            f"Analyze request failed: {analyze_response.status} {analyze_response.text()}"
        )
        analyze_job = analyze_response.json()
        analyze_job_id = analyze_job.get("job_id")
        assert analyze_job_id, "Missing job_id in analyze response"
        analyze_status = job_waiter(
            analyze_job_id,
            poll_interval=analyze_job.get("poll_interval", 1.0),
            timeout_seconds=300.0,
        )
        assert analyze_status.get("status") == "completed", analyze_status.get("error")

        page.goto(f"{base_url}/dataset")
        no_report_row = (
            page.locator(".dataset-preview-row")
            .filter(has_text=without_report_dataset)
            .first
        )
        with_report_row = (
            page.locator(".dataset-preview-row")
            .filter(has_text=with_report_dataset)
            .first
        )

        no_report_encoded = quote(without_report_dataset, safe="")
        with page.expect_response(
            lambda response: (
                "/api/datasets/reports/latest" in response.url
                and f"dataset_name={no_report_encoded}" in response.url
            )
        ):
            no_report_row.click(position={"x": 20, "y": 20})
        expect(
            page.locator(".dashboard-panel .panel-description").first
        ).to_contain_text("Load a saved report or run validation")

        with_report_encoded = quote(with_report_dataset, safe="")
        with page.expect_response(
            lambda response: (
                "/api/datasets/reports/latest" in response.url
                and f"dataset_name={with_report_encoded}" in response.url
            )
        ):
            with_report_row.click(position={"x": 20, "y": 20})
        expect(
            page.locator(".dashboard-panel .panel-description").first
        ).to_contain_text(f"Latest persisted session for {with_report_dataset}")

    def test_row_click_suppresses_not_found_banner_but_explicit_load_keeps_it(
        self,
        page: Page,
        base_url: str,
        api_context: APIRequestContext,
        job_waiter,
    ) -> None:
        """
        Selecting a dataset row with no report should not show a not-found banner,
        while explicit load action should still show it.
        """
        without_report = f"qa_row_noreport_only_{uuid4().hex[:8]}"
        without_report_dataset = _upload_dataset_for_ui_test(
            api_context=api_context,
            job_waiter=job_waiter,
            stem=without_report,
        )

        page.goto(f"{base_url}/dataset")
        row = (
            page.locator(".dataset-preview-row")
            .filter(has_text=without_report_dataset)
            .first
        )

        row.click(position={"x": 20, "y": 20})
        expect(page.locator(".dismissible-banner,[role='alert']")).to_have_count(0)

        row.locator("button[aria-label='Load latest saved report']").click()
        expect(page.locator(".dismissible-banner,[role='alert']")).to_contain_text(
            "No validation report found"
        )


class TestTokenizersPage:
    """Tests for tokenizers page UI elements."""

    def test_tokenizers_page_loads(self, page: Page, base_url: str) -> None:
        """Tokenizers page should render selection and report panels."""
        page.goto(f"{base_url}/tokenizers")
        expect(page.get_by_text("Tokenizer Selection")).to_be_visible()
        expect(page.get_by_text("Tokenizers Dashboard")).to_be_visible()


class TestCrossBenchmarkPage:
    """Tests for cross benchmark page UI elements."""

    def test_cross_benchmark_page_loads_controls(
        self, page: Page, base_url: str
    ) -> None:
        """Cross benchmark page should render control panel and report selector."""
        page.goto(f"{base_url}/cross-benchmark")
        expect(page.get_by_text("Tokenizer Benchmark")).to_be_visible()
        expect(page.get_by_role("button", name="Start benchmark")).to_be_visible()
        expect(page.locator("#benchmark-report-selector")).to_be_visible()

    def test_cross_benchmark_wizard_navigation_and_validation(
        self,
        page: Page,
        base_url: str,
    ) -> None:
        """Wizard should open, navigate to step 2, and enforce required step-2 selections."""
        page.goto(f"{base_url}/cross-benchmark")
        page.get_by_role("button", name="Start benchmark").click()
        expect(page.get_by_text("1. Metrics")).to_be_visible()
        expect(page.get_by_text("2. Inputs")).to_be_visible()
        expect(page.get_by_text("3. Summary")).to_be_visible()
        page.get_by_role("button", name="Next").click()
        expect(page.get_by_role("button", name="Back")).to_be_enabled()
        expect(page.get_by_role("button", name="Next")).to_be_disabled()

    def test_cross_benchmark_shows_diagnostics_for_failed_tokenizer_report(
        self,
        page: Page,
        base_url: str,
    ) -> None:
        """Cross benchmark dashboard should surface failure rows and metric availability diagnostics."""

        page.route(
            "**/api/tokenizers/list",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body='{"tokenizers":[{"tokenizer_name":"ok/tokenizer"}],"count":1}',
            ),
        )
        page.route(
            "**/api/datasets/list",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body='{"datasets":[{"dataset_name":"custom/sample","document_count":2}]}',
            ),
        )
        page.route(
            "**/api/benchmarks/metrics/catalog",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body='{"categories":[{"category_key":"efficiency","category_label":"Efficiency","metrics":[]}]}',
            ),
        )
        page.route(
            "**/api/benchmarks/reports*",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=(
                    '{"reports":[{"report_id":1,"report_version":2,"created_at":"2026-01-01T00:00:00Z",'
                    '"run_name":"mock run","dataset_name":"custom/sample","documents_processed":2,'
                    '"tokenizers_count":2,"tokenizers_processed":["ok/tokenizer","broken/tokenizer"],'
                    '"selected_metric_keys":["eff.encode_tokens_per_second_mean"]}]}'
                ),
            ),
        )
        page.route(
            "**/api/benchmarks/reports/1",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=(
                    '{"status":"success","schema_version":1,"methodology_version":"v2_semantic_honesty",'
                    '"report_id":1,"report_version":2,"created_at":"2026-01-01T00:00:00Z","run_name":"mock run",'
                    '"selected_metric_keys":["eff.encode_tokens_per_second_mean"],"dataset_name":"custom/sample",'
                    '"documents_processed":2,"tokenizers_processed":["ok/tokenizer","broken/tokenizer"],'
                    '"tokenizers_count":2,"config":{"max_documents":0,"warmup_trials":2,"timed_trials":8,'
                    '"batch_size":16,"seed":42,"parallelism":1,"include_lm_metrics":false,'
                    '"add_special_tokens":false,"padding":false,"truncation":false,"max_length":null,'
                    '"store_per_document_stats":false,"per_document_sample_size":500},'
                    '"hardware_profile":{"runtime":"","os":"","cpu_model":null,"cpu_logical_cores":null,"memory_total_mb":null},'
                    '"trial_summary":{"warmup_trials":2,"timed_trials":8},'
                    '"tokenizer_results":['
                    '{"tokenizer":"ok/tokenizer","status":"success","error_type":null,"error_message":null,'
                    '"tokenizer_family":"unknown","runtime_backend":"transformers_auto","vocabulary_size":10,'
                    '"added_tokens":0,"special_token_share":0.0,'
                    '"efficiency":{"encode_tokens_per_second_mean":10.0,"encode_tokens_per_second_ci95_low":9.0,'
                    '"encode_tokens_per_second_ci95_high":11.0,"encode_chars_per_second_mean":100.0,'
                    '"encode_bytes_per_second_mean":100.0,"encode_only_wall_time_seconds":1.0,'
                    '"dataset_stream_wall_time_seconds":0.2,"postprocess_wall_time_seconds":0.3,'
                    '"end_to_end_wall_time_seconds":1.6,"load_time_seconds":0.0},'
                    '"latency":{"encode_latency_p50_ms":1.0,"encode_latency_p95_ms":2.0,"encode_latency_p99_ms":3.0,"sample_count":8},'
                    '"fidelity":{"exact_round_trip_rate":1.0,"normalized_round_trip_rate":1.0,'
                    '"unknown_token_rate":0.0,"byte_fallback_rate":null,"lossless_encodability_rate":100.0},'
                    '"fragmentation":{"tokens_per_character":0.5,"characters_per_token":2.0,'
                    '"tokens_per_byte":0.5,"bytes_per_token":2.0,"pieces_per_word_mean":1.0,'
                    '"fragmentation_by_word_length_bucket":[{"bucket":"short_1_4","pieces_per_word_mean":1.0}]},'
                    '"resources":{"peak_rss_mb":10.0,"memory_delta_mb":1.0}},'
                    '{"tokenizer":"broken/tokenizer","status":"failed","error_type":"RuntimeError",'
                    '"error_message":"broken tokenizer","tokenizer_family":"unknown","runtime_backend":"transformers_auto",'
                    '"vocabulary_size":0,"added_tokens":0,"special_token_share":0.0,'
                    '"efficiency":{"encode_tokens_per_second_mean":0.0,"encode_tokens_per_second_ci95_low":0.0,'
                    '"encode_tokens_per_second_ci95_high":0.0,"encode_chars_per_second_mean":0.0,'
                    '"encode_bytes_per_second_mean":0.0,"encode_only_wall_time_seconds":0.0,'
                    '"dataset_stream_wall_time_seconds":0.2,"postprocess_wall_time_seconds":0.0,'
                    '"end_to_end_wall_time_seconds":0.0,"load_time_seconds":0.0},'
                    '"latency":{"encode_latency_p50_ms":0.0,"encode_latency_p95_ms":0.0,"encode_latency_p99_ms":0.0,"sample_count":0},'
                    '"fidelity":{"exact_round_trip_rate":0.0,"normalized_round_trip_rate":0.0,'
                    '"unknown_token_rate":null,"byte_fallback_rate":null,"lossless_encodability_rate":null},'
                    '"fragmentation":{"tokens_per_character":0.0,"characters_per_token":0.0,'
                    '"tokens_per_byte":0.0,"bytes_per_token":0.0,"pieces_per_word_mean":0.0,'
                    '"fragmentation_by_word_length_bucket":[]},'
                    '"resources":{"peak_rss_mb":0.0,"memory_delta_mb":0.0}}],'
                    '"chart_data":{"efficiency":[{"tokenizer":"ok/tokenizer","value":10.0,"ci95_low":9.0,"ci95_high":11.0}],'
                    '"fidelity":[{"tokenizer":"ok/tokenizer","value":1.0}],"vocabulary":[{"tokenizer":"ok/tokenizer","value":10.0}],'
                    '"fragmentation":[{"tokenizer":"ok/tokenizer","value":1.0}],"latency_or_memory_distribution":[]},'
                    '"per_document_stats":[],'
                    '"runtime_metadata":{"metric_availability":{"resource_metrics":true,"latency_distribution":false,'
                    '"byte_fallback_rate":false,"unknown_token_rate":true,"vocab_character_overlap":true,'
                    '"fragmentation_word_length_bucket":true,"per_document_stats":false},"benchmark_timing_boundaries":{"encode_only_definition":"encode only",'
                    '"dataset_stream_definition":"stream","postprocess_definition":"postprocess"}},'
                    '"raw_observations":{"broken/tokenizer":[{"error":"RuntimeError","message":"broken tokenizer"}]}}'
                ),
            ),
        )

        page.goto(f"{base_url}/cross-benchmark")
        report_selector = page.locator("#benchmark-report-selector")
        expect(report_selector).to_contain_text("mock run")
        report_selector.select_option("1")
        expect(page.get_by_text("Run Diagnostics")).to_be_visible()
        expect(page.get_by_role("cell", name="broken/tokenizer").first).to_be_visible()
        expect(page.get_by_text("RuntimeError")).to_be_visible()
        latency_diag = page.get_by_text("Latency distribution:", exact=False)
        expect(latency_diag).to_be_visible()
        expect(latency_diag).to_contain_text("unavailable")
