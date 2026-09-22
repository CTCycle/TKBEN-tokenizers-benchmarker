"""
E2E tests for UI navigation and page rendering.
Targets datasets, tokenizers, and cross benchmark workflows.
"""

import csv
import io
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse
from uuid import uuid4

import pytest
from playwright.sync_api import Page, expect
from playwright.sync_api import APIRequestContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.repositories.datasets import DatasetRepository
from server.repositories.schemas.models import Tokenizer, TokenizerReport
from server.services.tokenizers import TokenizersService

###############################################################################
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


###############################################################################
def _request_query(url: str) -> dict[str, list[str]]:
    return parse_qs(urlparse(url).query, keep_blank_values=True)


###############################################################################
def _matches_catalog_request(request, endpoint: str, expected: dict[str, list[str]]) -> bool:
    return (
        urlparse(request.url).path.endswith(endpoint)
        and _request_query(request.url) == expected
    )


###############################################################################
def _expect_catalog_request(
    page: Page,
    endpoint: str,
    expected: dict[str, list[str]],
    action,
) -> None:
    with page.expect_request(
        lambda request: _matches_catalog_request(request, endpoint, expected)
    ):
        action()


###############################################################################
def _seed_dataset_catalog_fixture(rows: list[tuple[str, int]]) -> None:
    repository = DatasetRepository()
    for dataset_name, document_count in rows:
        dataset_id = repository.begin_dataset_import(dataset_name)
        repository.finalize_dataset_import(dataset_id, document_count)


###############################################################################
def _cleanup_dataset_catalog_fixture(dataset_names: list[str]) -> None:
    repository = DatasetRepository()
    for dataset_name in dataset_names:
        repository.delete_dataset(dataset_name)


###############################################################################
def _seed_tokenizer_catalog_fixture(
    rows: list[tuple[str, str, int]],
    artifact: bytes,
) -> None:
    service = TokenizersService()
    for tokenizer_name, source, vocabulary_size in rows:
        service.repository.upsert_tokenizer_source(tokenizer_name, source=source)
        service.persist_custom_tokenizer_artifact(tokenizer_name, artifact)
        with Session(bind=service.repository.database.backend.engine) as session:
            tokenizer_id = session.execute(
                select(Tokenizer.id).where(Tokenizer.name == tokenizer_name)
            ).scalar_one()
            session.add(
                TokenizerReport(
                    tokenizer_id=int(tokenizer_id),
                    report_version=5,
                    created_at=datetime.now(timezone.utc),
                    metadata_json={"vocabulary_size": vocabulary_size},
                    token_length_histogram={},
                )
            )
            session.commit()


###############################################################################
def _cleanup_tokenizer_catalog_fixture(tokenizer_names: list[str]) -> None:
    service = TokenizersService()
    for tokenizer_name in tokenizer_names:
        service.remove_tokenizer(tokenizer_name)


###############################################################################
def _expected_dataset_names(
    catalog: list[dict],
    *,
    search: str = "",
    source: str = "all",
    operator: str = "at_least",
    document_count: int | None = None,
) -> list[str]:
    search_term = search.strip().casefold()
    expected = []
    for item in catalog:
        name = str(item["dataset_name"])
        count = int(item.get("document_count") or 0)
        if search_term and search_term not in name.casefold():
            continue
        is_custom = name.startswith("custom/")
        if source == "custom" and not is_custom:
            continue
        if source == "public" and is_custom:
            continue
        if document_count is not None:
            if operator == "at_most" and count > document_count:
                continue
            if operator == "at_least" and count < document_count:
                continue
        expected.append(name)
    return sorted(expected)


###############################################################################
def _expected_tokenizer_names(
    catalog: list[dict],
    *,
    search: str = "",
    source: str = "all",
    operator: str = "at_least",
    vocabulary_size: int | None = None,
) -> list[str]:
    search_term = search.strip().casefold()
    expected = []
    for item in catalog:
        name = str(item["tokenizer_name"])
        item_source = str(item.get("source") or "")
        item_size = item.get("vocabulary_size")
        if search_term and search_term not in name.casefold():
            continue
        if source != "all" and item_source != source:
            continue
        if vocabulary_size is not None:
            if item_size is None:
                continue
            if operator == "at_most" and int(item_size) > vocabulary_size:
                continue
            if operator == "at_least" and int(item_size) < vocabulary_size:
                continue
        expected.append(name)
    return sorted(expected)


###############################################################################
def _assert_rendered_catalog_names(locator, expected: list[str]) -> None:
    expect(locator).to_have_count(len(expected))
    for name in expected:
        expect(locator.filter(has_text=name)).to_have_count(1)
    assert sorted(text.strip() for text in locator.all_text_contents()) == expected

###############################################################################
class TestAppShell:
    """Tests for core layout and routing."""

    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("path", ["", "/does-not-exist"])
    def test_routes_fall_back_to_dataset(
        self, page: Page, base_url: str, path: str
    ) -> None:
        """Root and wildcard routes should land on the dataset workflow."""
        page.goto(f"{base_url}{path}")
        expect(page).to_have_url(re.compile(r".*/dataset/?$"))
        expect(page.get_by_text("Dataset Usage")).to_be_visible()

###############################################################################
class TestDatasetPage:
    """Tests for dataset page UI elements."""

    # -------------------------------------------------------------------------
    def test_validation_pipeline_populates_all_metric_families_and_persists_dashboard(
        self,
        page: Page,
        base_url: str,
        api_context: APIRequestContext,
        job_waiter,
    ) -> None:
        """A populated local dataset should exercise every metric family in the UI."""
        dataset_stem = f"qa_t2_02_{uuid4().hex[:10]}"
        dataset_name = f"custom/{dataset_stem}"
        filename = f"{dataset_stem}.csv"
        documents = [
            "The cat sat. The cat sat!",
            "The cat sat. The cat sat!",
            "Visit https://example.com and email me@test.com.\n<tag>AA 123</tag>",
            "Café, unusual vocabulary crosses several paragraphs.\n"
            "A second line adds longer words and varied punctuation!",
        ]
        dataset_created = False
        browser_errors: list[str] = []

        catalog_response = api_context.get("/api/datasets/metrics/catalog")
        assert catalog_response.ok, catalog_response.text()
        categories = catalog_response.json().get("categories", [])
        expected_category_keys = {
            "corpus_scale",
            "lexical_diversity",
            "word_character_signals",
            "document_quality",
            "structural_regularity",
            "compression_redundancy",
        }
        assert {
            category.get("category_key") for category in categories
        } == expected_category_keys
        metric_keys = {
            metric["key"]
            for category in categories
            for metric in category.get("metrics", [])
            if isinstance(metric.get("key"), str)
        }
        assert metric_keys

        csv_buffer = io.StringIO(newline="")
        writer = csv.writer(csv_buffer, lineterminator="\n")
        writer.writerow(["text"])
        writer.writerows((document,) for document in documents)

        try:
            upload_response = api_context.post(
                "/api/datasets/upload",
                multipart={
                    "file": {
                        "name": filename,
                        "mimeType": "text/csv",
                        "buffer": csv_buffer.getvalue().encode("utf-8"),
                    }
                },
            )
            assert upload_response.ok, (
                f"Dataset upload failed: {upload_response.status} {upload_response.text()}"
            )
            dataset_created = True
            upload_job = upload_response.json()
            upload_job_id = upload_job.get("job_id")
            assert upload_job_id, "Missing job_id in upload response"
            upload_status = job_waiter(
                upload_job_id,
                poll_interval=upload_job.get("poll_interval", 1.0),
                timeout_seconds=300.0,
            )
            assert upload_status.get("status") == "completed", upload_status.get("error")
            assert upload_status.get("result", {}).get("document_count") == len(documents)

            page.on(
                "console",
                lambda message: browser_errors.append(message.text)
                if message.type == "error"
                else None,
            )
            page.goto(f"{base_url}/dataset")
            dataset_row = page.locator(".dataset-preview-row").filter(
                has_text=dataset_name
            ).first
            expect(dataset_row).to_be_visible()
            dataset_row.get_by_role(
                "button", name=f"Run validation pipeline for {dataset_name}"
            ).click()

            for category in categories:
                expect(
                    page.get_by_role(
                        "checkbox", name=category["category_label"], exact=True
                    )
                ).to_be_checked()

            page.get_by_role("button", name="Next", exact=True).click()
            expect(page.get_by_role("radio", name="Fraction", exact=True)).to_be_checked()
            expect(
                page.get_by_role(
                    "checkbox", name="Exclude empty documents", exact=True
                )
            ).to_be_checked()
            page.get_by_role("button", name="Next", exact=True).click()
            expect(
                page.locator(".validation-summary")
            ).to_contain_text(f"Selected metrics: {len(metric_keys)}")

            with page.expect_request(
                lambda request: request.method == "POST"
                and urlparse(request.url).path == "/api/datasets/analyze"
            ) as analyze_request:
                page.get_by_role("button", name="Run Validation", exact=True).click()
            request_payload = analyze_request.value.post_data_json
            assert request_payload.get("dataset_name") == dataset_name
            assert set(request_payload.get("selected_metric_keys", [])) == metric_keys
            assert request_payload.get("sampling") == {"fraction": 1}
            assert request_payload.get("filters") == {
                "min_length": None,
                "max_length": None,
                "exclude_empty": True,
            }

            dashboard = page.locator(".dataset-dashboard")
            expect(
                dashboard.locator(".panel-description").first
            ).to_contain_text(
                f"Latest persisted session for {dataset_name}", timeout=300_000
            )

            report_response = api_context.get(
                "/api/datasets/reports/latest",
                params={"dataset_name": dataset_name},
            )
            assert report_response.ok, report_response.text()
            report = report_response.json()
            assert report.get("report_id")
            assert set(report.get("selected_metric_keys", [])) == metric_keys
            aggregate = report.get("aggregate_statistics", {})
            family_signals = {
                "corpus_scale": ["corpus.document_count", "doc.length_mean"],
                "lexical_diversity": ["words.shannon_entropy", "words.zipf_slope"],
                "word_character_signals": [
                    "chars.entropy",
                    "chars.punctuation_ratio",
                ],
                "document_quality": [
                    "quality.exact_duplicate_rate",
                    "quality.near_duplicate_rate",
                ],
                "structural_regularity": [
                    "structure.url_density",
                    "structure.email_density",
                    "structure.html_tag_ratio",
                ],
                "compression_redundancy": [
                    "compression.ratio",
                    "compression.avg_repetition_factor",
                ],
            }
            for family, keys in family_signals.items():
                for key in keys:
                    value = aggregate.get(key)
                    assert isinstance(value, (int, float)) and math.isfinite(value), (
                        f"{family} metric {key} was not a finite aggregate: {value!r}"
                    )
            nonzero_signals = (
                "corpus.document_count",
                "doc.length_mean",
                "words.shannon_entropy",
                "chars.entropy",
                "chars.punctuation_ratio",
                "quality.exact_duplicate_rate",
                "structure.url_density",
                "structure.email_density",
                "compression.ratio",
            )
            for key in nonzero_signals:
                assert abs(aggregate[key]) > 0, f"Metric {key} was zero"

            histogram_payloads = {
                "hist.document_length": report.get("document_length_histogram"),
                "hist.word_length": report.get("word_length_histogram"),
            }
            for histogram_key, histogram in histogram_payloads.items():
                assert isinstance(histogram, dict), histogram_key
                assert histogram.get("bins") and histogram.get("counts"), histogram_key
                assert sum(histogram["counts"]) > 0, histogram_key
            assert report.get("most_common_words")
            assert report.get("word_cloud_terms")

            expect(dashboard.get_by_text("Aggregate Stats", exact=True)).to_be_visible()
            expect(dashboard.get_by_text("Word Metrics", exact=True)).to_be_visible()
            expect(
                dashboard.get_by_role(
                    "img", name="Character composition donut chart", exact=True
                )
            ).to_be_visible()
            histogram_charts = dashboard.locator(".dataset-histogram-chart")
            for index, histogram_name in enumerate((
                "Document length histogram",
                "Word length histogram",
            )):
                histogram_chart = histogram_charts.nth(index)
                expect(
                    histogram_chart.get_by_role(
                        "img", name=re.compile(histogram_name)
                    )
                ).to_be_visible()
                assert histogram_chart.locator(".dataset-histogram-bar").count() > 0
            expect(dashboard.locator(".dataset-zipf-chart")).to_be_visible()
            expect(dashboard.get_by_text("Entropy Gauge", exact=True)).to_be_visible()
            expect(
                dashboard.get_by_text("Duplicate Indicators", exact=True)
            ).to_be_visible()
            expect(dashboard.get_by_text("Concentration", exact=True)).to_be_visible()
            expect(dashboard.locator(".dataset-word-cloud-term").first).to_be_visible()

            for metric_label in ("Mean length", "Length CV", "Vocabulary size", "Entropy"):
                metric_row = dashboard.locator(".dataset-table tr").filter(
                    has_text=metric_label
                )
                expect(metric_row).to_be_visible()
                expect(metric_row.locator("td")).not_to_have_text("—")
            expect(
                dashboard.locator(".dataset-table tr").filter(has_text="Empty count")
            ).to_contain_text("0")

            page.reload()
            dataset_row = page.locator(".dataset-preview-row").filter(
                has_text=dataset_name
            ).first
            with page.expect_response(
                lambda response: response.request.method == "GET"
                and "/api/datasets/reports/latest" in response.url
                and dataset_stem in response.url
            ):
                dataset_row.click(position={"x": 20, "y": 20})
            dashboard = page.locator(".dataset-dashboard")
            expect(
                dashboard.locator(".panel-description").first
            ).to_contain_text(f"Latest persisted session for {dataset_name}")
            expect(dashboard.get_by_text("Aggregate Stats", exact=True)).to_be_visible()
            expect(dashboard.locator(".dataset-word-cloud-term").first).to_be_visible()
            assert not browser_errors, f"Browser console errors: {browser_errors}"

            screenshot_dir = (
                Path(__file__).resolve().parents[3]
                / "runtimes"
                / "cache"
                / "t2-02-validation-logs"
            )
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            page.screenshot(
                path=str(
                    screenshot_dir
                    / "tkben-t2-02-dataset-metric-families-20260922.png"
                ),
                full_page=True,
            )
        finally:
            if dataset_created:
                delete_response = api_context.delete(
                    "/api/datasets/delete", params={"dataset_name": dataset_name}
                )
                assert delete_response.status in {200, 404}, delete_response.text()
                latest_after_cleanup = api_context.get(
                    "/api/datasets/reports/latest",
                    params={"dataset_name": dataset_name},
                )
                assert latest_after_cleanup.status == 404, latest_after_cleanup.text()

    # -------------------------------------------------------------------------
    def test_populated_catalog_filter_matrix(
        self,
        page: Page,
        base_url: str,
        api_context: APIRequestContext,
    ) -> None:
        """Dataset controls serialize every supported filter and render the API result."""
        baseline_response = api_context.get("/api/datasets/list")
        assert baseline_response.ok, baseline_response.text()
        baseline_catalog = baseline_response.json().get("datasets", [])

        suffix = uuid4().hex[:8]
        fixture_rows = [
            (f"hf/t1-05-alpha-{suffix}", 2),
            (f"hf/t1-05-beta-{suffix}", 5),
            (f"custom/t1-05-gamma-{suffix}", 5),
            (f"custom/t1-05-delta-{suffix}", 8),
        ]
        fixture_names = [name for name, _ in fixture_rows]
        _seed_dataset_catalog_fixture(fixture_rows)
        catalog = [*baseline_catalog, *(
            {"dataset_name": name, "document_count": count}
            for name, count in fixture_rows
        )]

        try:
            request_urls: list[str] = []
            page.on(
                "request",
                lambda request: request_urls.append(request.url)
                if urlparse(request.url).path.endswith("/api/datasets/list")
                else None,
            )
            page.goto(f"{base_url}/dataset")

            dataset_names = page.locator(
                ".dataset-preview-row:not(.dataset-preview-row--header) .dataset-preview-name"
            )
            expected_all = _expected_dataset_names(catalog)
            _assert_rendered_catalog_names(dataset_names, expected_all)
            assert any(
                _request_query(url) == {}
                for url in request_urls
            )

            search = page.get_by_label("Search datasets")
            source = page.get_by_label("Source")
            operator = page.get_by_label("Documents comparison")
            document_count = page.get_by_label("Document count")
            search_term = f"beta-{suffix}"

            _expect_catalog_request(
                page,
                "/api/datasets/list",
                {
                    "search": [search_term],
                    "document_count_operator": ["at_least"],
                },
                lambda: search.fill(f"  {search_term}  "),
            )
            _assert_rendered_catalog_names(dataset_names, [fixture_rows[1][0]])

            def clear_search_and_select_public() -> None:
                search.fill("")
                source.select_option("public")

            _expect_catalog_request(
                page,
                "/api/datasets/list",
                {
                    "source": ["public"],
                    "document_count_operator": ["at_least"],
                },
                clear_search_and_select_public,
            )
            public_expected = _expected_dataset_names(catalog, source="public")
            _assert_rendered_catalog_names(dataset_names, public_expected)

            _expect_catalog_request(
                page,
                "/api/datasets/list",
                {
                    "source": ["custom"],
                    "document_count_operator": ["at_least"],
                },
                lambda: source.select_option("custom"),
            )
            custom_expected = _expected_dataset_names(catalog, source="custom")
            _assert_rendered_catalog_names(dataset_names, custom_expected)

            def select_all_and_set_at_least_boundary() -> None:
                source.select_option("")
                document_count.fill("5")

            _expect_catalog_request(
                page,
                "/api/datasets/list",
                {
                    "document_count_operator": ["at_least"],
                    "document_count": ["5"],
                },
                select_all_and_set_at_least_boundary,
            )
            at_least_expected = _expected_dataset_names(
                catalog, document_count=5
            )
            _assert_rendered_catalog_names(dataset_names, at_least_expected)

            _expect_catalog_request(
                page,
                "/api/datasets/list",
                {
                    "document_count_operator": ["at_most"],
                    "document_count": ["5"],
                },
                lambda: operator.select_option("at_most"),
            )
            at_most_expected = _expected_dataset_names(
                catalog, operator="at_most", document_count=5
            )
            _assert_rendered_catalog_names(dataset_names, at_most_expected)

            def set_combined_dataset_filters() -> None:
                search.fill(f"  {search_term}  ")
                source.select_option("public")
                operator.select_option("at_least")

            combined_query = {
                "search": [search_term],
                "source": ["public"],
                "document_count_operator": ["at_least"],
                "document_count": ["5"],
            }
            _expect_catalog_request(
                page,
                "/api/datasets/list",
                combined_query,
                set_combined_dataset_filters,
            )
            combined_expected = _expected_dataset_names(
                catalog,
                search=search_term,
                source="public",
                document_count=5,
            )
            _assert_rendered_catalog_names(dataset_names, combined_expected)

            no_match_term = f"no-match-{suffix}"
            no_match_query = {**combined_query, "search": [no_match_term]}
            _expect_catalog_request(
                page,
                "/api/datasets/list",
                no_match_query,
                lambda: search.fill(no_match_term),
            )
            expect(dataset_names).to_have_count(0)
            expect(
                page.get_by_text("No datasets match the current filters.", exact=True)
            ).to_be_visible()

            def clear_dataset_filters() -> None:
                search.fill("")
                source.select_option("")
                document_count.fill("")

            _expect_catalog_request(
                page,
                "/api/datasets/list",
                {"document_count_operator": ["at_least"]},
                clear_dataset_filters,
            )
            _assert_rendered_catalog_names(dataset_names, expected_all)
        finally:
            _cleanup_dataset_catalog_fixture(fixture_names)

    # -------------------------------------------------------------------------
    def test_catalog_race_keeps_loading_owned_by_newest_request(
        self, page: Page, base_url: str
    ) -> None:
        """An older catalog response must not clear loading for a newer request."""
        page.add_init_script(
            """
            (() => {
              const nativeFetch = globalThis.fetch.bind(globalThis);
              globalThis.fetch = (input, init) => {
                const requestUrl = new URL(
                  typeof input === 'string' ? input : input.url,
                  window.location.href,
                );
                if (!requestUrl.pathname.endsWith('/api/datasets/list')) {
                  return nativeFetch(input, init);
                }
                const search = requestUrl.searchParams.get('search');
                const delay = search === 'first' ? 500 : search === 'second' ? 1000 : 0;
                const datasetName = search ? `custom/${search}` : 'custom/base';
                return new Promise((resolve) => {
                  window.setTimeout(() => resolve(new Response(JSON.stringify({
                    datasets: [{ dataset_name: datasetName, document_count: 1 }],
                    count: 1,
                  }), {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                  })), delay);
                });
              };
            })();
            """
        )
        page.goto(f"{base_url}/dataset")
        search_input = page.get_by_label("Search datasets")
        expect(search_input).to_be_visible()

        search_input.fill("first")
        page.wait_for_timeout(300)
        search_input.fill("second")
        page.wait_for_timeout(500)

        expect(page.get_by_text("Loading datasets...", exact=True)).to_be_visible()
        expect(
            page.locator(".dataset-preview-row").filter(has_text="custom/second")
        ).to_have_count(0)

        page.wait_for_timeout(850)
        expect(page.get_by_text("Loading datasets...", exact=True)).to_have_count(0)
        expect(
            page.locator(".dataset-preview-row").filter(has_text="custom/second")
        ).to_have_count(1)

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def test_dataset_without_report_can_be_deleted_and_disappears_from_catalog(
        self,
        page: Page,
        base_url: str,
        api_context: APIRequestContext,
        job_waiter,
    ) -> None:
        """A no-report dataset remains selectable and is removed from UI and API state."""
        dataset_name = _upload_dataset_for_ui_test(
            api_context=api_context,
            job_waiter=job_waiter,
            stem=f"qa_delete_noreport_{uuid4().hex[:8]}",
        )

        page.goto(f"{base_url}/dataset")
        row = page.locator(".dataset-preview-row").filter(has_text=dataset_name).first
        expect(row).to_be_visible()

        delete_count = 0

        def count_delete(request) -> None:
            nonlocal delete_count
            if request.method == "DELETE" and "/api/datasets/delete" in request.url:
                delete_count += 1

        page.on("request", count_delete)
        page.once("dialog", lambda dialog: dialog.accept())
        with page.expect_response(
            lambda response: (
                response.request.method == "DELETE"
                and "/api/datasets/delete" in response.url
                and response.status == 200
            )
        ):
            row.get_by_role("button", name="Remove dataset").click()

        expect(
            page.locator(".dataset-preview-row").filter(has_text=dataset_name)
        ).to_have_count(0)
        assert delete_count == 1

        refreshed = api_context.get("/api/datasets/list")
        assert refreshed.ok
        assert dataset_name not in {
            str(item.get("dataset_name"))
            for item in refreshed.json().get("datasets", [])
        }

###############################################################################
class TestTokenizersPage:
    """Tests for tokenizers page UI elements."""

    # -------------------------------------------------------------------------
    def test_populated_catalog_filter_matrix(
        self,
        page: Page,
        base_url: str,
        api_context: APIRequestContext,
        tiny_tokenizer_json: bytes,
    ) -> None:
        """Tokenizer controls serialize source and vocabulary filters end to end."""
        baseline_response = api_context.get("/api/tokenizers/list")
        assert baseline_response.ok, baseline_response.text()
        baseline_catalog = baseline_response.json().get("tokenizers", [])

        suffix = uuid4().hex[:8]
        fixture_rows = [
            (f"hf/t1-05-alpha-{suffix}", "huggingface", 2),
            (f"hf/t1-05-beta-{suffix}", "huggingface", 5),
            (f"CUSTOM_t1-05-gamma-{suffix}", "custom", 5),
            (f"CUSTOM_t1-05-delta-{suffix}", "custom", 8),
        ]
        fixture_names = [name for name, _, _ in fixture_rows]
        _seed_tokenizer_catalog_fixture(fixture_rows, tiny_tokenizer_json)
        catalog = [
            *baseline_catalog,
            *(
                {
                    "tokenizer_name": name,
                    "source": source,
                    "vocabulary_size": vocabulary_size,
                }
                for name, source, vocabulary_size in fixture_rows
            ),
        ]

        try:
            request_urls: list[str] = []
            page.on(
                "request",
                lambda request: request_urls.append(request.url)
                if urlparse(request.url).path.endswith("/api/tokenizers/list")
                else None,
            )
            page.goto(f"{base_url}/tokenizers")

            tokenizer_names = page.locator(".tokenizer-preview-name")
            expected_all = _expected_tokenizer_names(catalog)
            _assert_rendered_catalog_names(tokenizer_names, expected_all)
            assert any(
                _request_query(url) == {}
                for url in request_urls
            )

            search = page.get_by_label("Search tokenizers")
            source = page.get_by_label("Source")
            operator = page.get_by_label("Vocabulary comparison")
            vocabulary_size = page.get_by_label("Vocabulary size")
            search_term = f"beta-{suffix}"

            _expect_catalog_request(
                page,
                "/api/tokenizers/list",
                {
                    "search": [search_term],
                    "vocabulary_size_operator": ["at_least"],
                },
                lambda: search.fill(f"  {search_term}  "),
            )
            _assert_rendered_catalog_names(tokenizer_names, [fixture_rows[1][0]])

            def clear_search_and_select_huggingface() -> None:
                search.fill("")
                source.select_option("hugging_face")

            _expect_catalog_request(
                page,
                "/api/tokenizers/list",
                {
                    "source": ["huggingface"],
                    "vocabulary_size_operator": ["at_least"],
                },
                clear_search_and_select_huggingface,
            )
            huggingface_expected = _expected_tokenizer_names(
                catalog, source="huggingface"
            )
            _assert_rendered_catalog_names(tokenizer_names, huggingface_expected)

            _expect_catalog_request(
                page,
                "/api/tokenizers/list",
                {
                    "source": ["custom"],
                    "vocabulary_size_operator": ["at_least"],
                },
                lambda: source.select_option("custom"),
            )
            custom_expected = _expected_tokenizer_names(catalog, source="custom")
            _assert_rendered_catalog_names(tokenizer_names, custom_expected)

            def select_all_and_set_at_least_boundary() -> None:
                source.select_option("")
                vocabulary_size.fill("5")

            _expect_catalog_request(
                page,
                "/api/tokenizers/list",
                {
                    "vocabulary_size_operator": ["at_least"],
                    "vocabulary_size": ["5"],
                },
                select_all_and_set_at_least_boundary,
            )
            at_least_expected = _expected_tokenizer_names(
                catalog, vocabulary_size=5
            )
            _assert_rendered_catalog_names(tokenizer_names, at_least_expected)

            _expect_catalog_request(
                page,
                "/api/tokenizers/list",
                {
                    "vocabulary_size_operator": ["at_most"],
                    "vocabulary_size": ["5"],
                },
                lambda: operator.select_option("at_most"),
            )
            at_most_expected = _expected_tokenizer_names(
                catalog, operator="at_most", vocabulary_size=5
            )
            _assert_rendered_catalog_names(tokenizer_names, at_most_expected)

            def set_combined_tokenizer_filters() -> None:
                search.fill(f"  {search_term}  ")
                source.select_option("hugging_face")
                operator.select_option("at_least")

            combined_query = {
                "search": [search_term],
                "source": ["huggingface"],
                "vocabulary_size_operator": ["at_least"],
                "vocabulary_size": ["5"],
            }
            _expect_catalog_request(
                page,
                "/api/tokenizers/list",
                combined_query,
                set_combined_tokenizer_filters,
            )
            combined_expected = _expected_tokenizer_names(
                catalog,
                search=search_term,
                source="huggingface",
                vocabulary_size=5,
            )
            _assert_rendered_catalog_names(tokenizer_names, combined_expected)

            no_match_term = f"no-match-{suffix}"
            no_match_query = {**combined_query, "search": [no_match_term]}
            _expect_catalog_request(
                page,
                "/api/tokenizers/list",
                no_match_query,
                lambda: search.fill(no_match_term),
            )
            expect(tokenizer_names).to_have_count(0)
            expect(
                page.get_by_text("No tokenizers match the current filters.", exact=True)
            ).to_be_visible()

            def clear_tokenizer_filters() -> None:
                search.fill("")
                source.select_option("")
                vocabulary_size.fill("")

            _expect_catalog_request(
                page,
                "/api/tokenizers/list",
                {"vocabulary_size_operator": ["at_least"]},
                clear_tokenizer_filters,
            )
            _assert_rendered_catalog_names(tokenizer_names, expected_all)
        finally:
            _cleanup_tokenizer_catalog_fixture(fixture_names)

    # -------------------------------------------------------------------------
    def test_tokenizer_catalog_race_keeps_newest_rows_and_loading_state(
        self, page: Page, base_url: str
    ) -> None:
        """Stale tokenizer catalog responses cannot replace the newest request."""
        page.add_init_script(
            """
            (() => {
              const nativeFetch = globalThis.fetch.bind(globalThis);
              globalThis.fetch = (input, init) => {
                const requestUrl = new URL(
                  typeof input === 'string' ? input : input.url,
                  window.location.href,
                );
                if (!requestUrl.pathname.endsWith('/api/tokenizers/list')) {
                  return nativeFetch(input, init);
                }
                const search = requestUrl.searchParams.get('search') || '';
                const delays = {
                  first: 2000,
                  second: 200,
                  reset: 0,
                  fast: 600,
                  slow: 2000,
                };
                const datasetName = search ? `custom/race-${search}` : 'custom/base';
                const body = search === 'reset'
                  ? { tokenizers: [], count: 0 }
                  : search
                    ? { tokenizers: [{ tokenizer_name: datasetName, source: 'custom', vocabulary_size: 3 }], count: 1 }
                    : { tokenizers: [], count: 0 };
                return new Promise((resolve) => {
                  window.setTimeout(() => resolve(new Response(JSON.stringify(body), {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                  })), delays[search] ?? 0);
                });
              };
            })();
            """
        )
        page.goto(f"{base_url}/tokenizers")
        expect(
            page.get_by_text("No tokenizers match the current filters.", exact=True)
        ).to_be_visible()
        search = page.get_by_label("Search tokenizers")

        search.fill("first")
        page.wait_for_timeout(320)
        search.fill("second")
        page.wait_for_timeout(500)
        expect(page.get_by_text("Loading tokenizers...", exact=True)).to_have_count(0)
        expect(
            page.locator(".tokenizer-preview-row").filter(has_text="custom/race-second")
        ).to_have_count(1)
        expect(
            page.locator(".tokenizer-preview-row").filter(has_text="custom/race-first")
        ).to_have_count(0)

        page.wait_for_timeout(500)
        expect(
            page.locator(".tokenizer-preview-row").filter(has_text="custom/race-second")
        ).to_have_count(1)
        expect(
            page.locator(".tokenizer-preview-row").filter(has_text="custom/race-first")
        ).to_have_count(0)

        search.fill("reset")
        page.wait_for_timeout(400)
        expect(
            page.get_by_text("No tokenizers match the current filters.", exact=True)
        ).to_be_visible()

        search.fill("fast")
        page.wait_for_timeout(320)
        search.fill("slow")
        page.wait_for_timeout(400)
        expect(page.get_by_text("Loading tokenizers...", exact=True)).to_be_visible()
        expect(
            page.locator(".tokenizer-preview-row").filter(has_text="custom/race-fast")
        ).to_have_count(0)
        page.wait_for_timeout(2200)
        expect(page.get_by_text("Loading tokenizers...", exact=True)).to_have_count(0)
        expect(
            page.locator(".tokenizer-preview-row").filter(has_text="custom/race-slow")
        ).to_have_count(1)
        expect(
            page.locator(".tokenizer-preview-row").filter(has_text="custom/race-fast")
        ).to_have_count(0)

    # -------------------------------------------------------------------------
    def test_tokenizer_discovery_race_ignores_stale_results_and_errors(
        self, page: Page, base_url: str
    ) -> None:
        """Discovery sequencing ignores an obsolete response or error in either order."""
        page.add_init_script(
            """
            (() => {
              const nativeFetch = globalThis.fetch.bind(globalThis);
              globalThis.fetch = (input, init) => {
                const requestUrl = new URL(
                  typeof input === 'string' ? input : input.url,
                  window.location.href,
                );
                if (requestUrl.pathname.endsWith('/api/tokenizers/list')) {
                  return Promise.resolve(new Response('{"tokenizers":[],"count":0}', {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                  }));
                }
                if (!requestUrl.pathname.endsWith('/api/tokenizers/discover')) {
                  return nativeFetch(input, init);
                }
                const search = requestUrl.searchParams.get('search') || '';
                const isError = search === 'old-error' || search === 'new-error';
                const delay = search === 'old-error' || search === 'old-success' ? 700 : 150;
                const body = isError
                  ? { detail: `discovery ${search} failed` }
                  : { items: [{ identifier: `local/${search}`, pipeline_tag: 'fill-mask', downloads: 1, likes: 1, gated: false, tags: [], vocabulary_size: 3 }], count: 1, fetched_count: 1 };
                return new Promise((resolve) => {
                  window.setTimeout(() => resolve(new Response(JSON.stringify(body), {
                    status: isError ? 500 : 200,
                    headers: { 'Content-Type': 'application/json' },
                  })), delay);
                });
              };
            })();
            """
        )
        page.goto(f"{base_url}/tokenizers")
        page.get_by_role("button", name="Add tokenizer").click()
        dialog = page.get_by_role("dialog", name="Tokenizer Manager")
        search = dialog.get_by_label("Search")
        form = dialog.locator("form.tokenizer-discovery-form")

        search.fill("old-error")
        form.evaluate("(form) => form.requestSubmit()")
        page.wait_for_timeout(100)
        search.fill("new-success")
        form.evaluate("(form) => form.requestSubmit()")
        page.wait_for_timeout(400)
        expect(dialog.get_by_text("local/new-success", exact=True)).to_be_visible()
        expect(dialog.get_by_role("alert")).to_have_count(0)
        page.wait_for_timeout(500)
        expect(dialog.get_by_text("local/new-success", exact=True)).to_be_visible()
        expect(dialog.get_by_role("alert")).to_have_count(0)

        search.fill("old-success")
        form.evaluate("(form) => form.requestSubmit()")
        page.wait_for_timeout(100)
        search.fill("new-error")
        form.evaluate("(form) => form.requestSubmit()")
        page.wait_for_timeout(400)
        expect(dialog.get_by_role("alert")).to_be_visible()
        expect(dialog.get_by_text("local/old-success", exact=True)).to_have_count(0)
        page.wait_for_timeout(500)
        expect(dialog.get_by_role("alert")).to_be_visible()
        expect(dialog.get_by_text("local/old-success", exact=True)).to_have_count(0)

    # -------------------------------------------------------------------------
    def test_tokenizer_manager_discovery_controls_and_empty_state(
        self, page: Page, base_url: str
    ) -> None:
        """Tokenizer discovery uses the structured backend contract and advanced filters."""
        page.route(
            "**/api/tokenizers/list*",
            lambda route: route.fulfill(json={"tokenizers": [], "count": 0}),
        )

        def fulfill_discovery(route) -> None:
            url = route.request.url
            if "unlikely-query" in url:
                route.fulfill(json={"items": [], "count": 0, "fetched_count": 0})
                return
            route.fulfill(
                json={
                    "items": [
                        {
                            "identifier": "google/bert-base-uncased",
                            "pipeline_tag": "fill-mask",
                            "library_name": "transformers",
                            "downloads": 1234,
                            "likes": 12,
                            "last_modified": None,
                            "gated": False,
                            "tags": ["core"],
                            "vocabulary_size": None,
                        }
                    ],
                    "count": 1,
                    "fetched_count": 1,
                }
            )

        page.route("**/api/tokenizers/discover*", fulfill_discovery)
        page.goto(f"{base_url}/tokenizers")
        page.get_by_role("button", name="Add tokenizer").click()

        dialog = page.get_by_role("dialog", name="Tokenizer Manager")
        expect(dialog).to_be_visible()
        expect(dialog.get_by_label("Search")).to_be_visible()
        expect(dialog.get_by_role("spinbutton", name="Results")).to_be_visible()
        expect(dialog.get_by_label("Category")).to_be_visible()
        expect(dialog.get_by_label("Sort")).to_be_visible()
        expect(dialog.get_by_role("button", name="Advanced filters")).to_have_attribute(
            "aria-expanded", "false"
        )
        dialog.get_by_role("button", name="Advanced filters").click()
        expect(dialog.get_by_label("Author")).to_be_visible()
        expect(dialog.get_by_label("Required tags")).to_be_visible()

        expect(
            dialog.get_by_text("google/bert-base-uncased", exact=True)
        ).to_be_visible()
        expect(dialog.get_by_text("Unknown", exact=True)).to_be_visible()
        expect(dialog.get_by_text("1234 downloads", exact=True)).to_be_visible()

        dialog.get_by_label("Search").fill("unlikely-query")
        with page.expect_request(
            lambda request: (
                "/api/tokenizers/discover" in request.url
                and "unlikely-query" in request.url
            )
        ):
            dialog.get_by_role("button", name="Search Hugging Face").click()
        expect(
            dialog.get_by_text(
                "No tokenizer repositories match this query.", exact=True
            )
        ).to_be_visible()

    # -------------------------------------------------------------------------
    def test_custom_tokenizer_delete_removes_preview_row_and_backend_item(
        self,
        page: Page,
        base_url: str,
        api_context: APIRequestContext,
        tiny_tokenizer_json: bytes,
    ) -> None:
        """Confirmed custom-tokenizer deletion updates the preview without a reload."""
        stem = f"qa_ui_delete_custom_{uuid4().hex[:8]}"
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

        page.goto(f"{base_url}/tokenizers")
        row = (
            page.locator(".tokenizer-preview-row").filter(has_text=tokenizer_name).first
        )
        expect(row).to_be_visible()

        page.once("dialog", lambda dialog: dialog.accept())
        with page.expect_response(
            lambda response: (
                response.request.method == "DELETE"
                and "/api/tokenizers/delete" in response.url
                and response.status == 200
            )
        ):
            row.get_by_role("button", name=f"Remove {tokenizer_name}").click()

        expect(
            page.locator(".tokenizer-preview-row").filter(has_text=tokenizer_name)
        ).to_have_count(0)
        refreshed = api_context.get("/api/tokenizers/list")
        assert refreshed.ok
        assert tokenizer_name not in {
            str(item.get("tokenizer_name"))
            for item in refreshed.json().get("tokenizers", [])
        }

###############################################################################
class TestCrossBenchmarkPage:
    """Tests for cross benchmark page UI elements."""

    # -------------------------------------------------------------------------
    def test_cross_benchmark_wizard_navigation_and_validation(
        self,
        page: Page,
        base_url: str,
    ) -> None:
        """Wizard should open, navigate to step 2, and enforce required step-2 selections."""
        page.goto(f"{base_url}/cross-benchmark")
        page.get_by_role("button", name="Run benchmark").click()
        expect(page.get_by_text("1. Metrics")).to_be_visible()
        expect(page.get_by_text("2. Inputs")).to_be_visible()
        expect(page.get_by_text("3. Summary")).to_be_visible()
        page.get_by_role("button", name="Next").click()
        expect(page.get_by_role("button", name="Back")).to_be_enabled()
        expect(page.get_by_role("button", name="Next")).to_be_disabled()

    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("viewport", [(1440, 900), (1024, 900), (390, 844)])
    def test_benchmark_actions_title_has_own_row_without_overflow(
        self,
        page: Page,
        base_url: str,
        viewport: tuple[int, int],
    ) -> None:
        """The shared command navbar keeps its title above the responsive action grid."""
        page.set_viewport_size({"width": viewport[0], "height": viewport[1]})
        page.goto(f"{base_url}/cross-benchmark")
        navbar = page.locator(".cross-benchmark-command-navbar")
        title = page.locator(".cross-benchmark-command-navbar__title")
        content = page.locator(".cross-benchmark-command-navbar__content")
        expect(title).to_be_visible()
        expect(content).to_be_visible()

        navbar_box = navbar.bounding_box()
        title_box = title.bounding_box()
        content_box = content.bounding_box()
        assert navbar_box and title_box and content_box
        assert content_box["y"] >= title_box["y"] + title_box["height"]
        assert content_box["x"] >= navbar_box["x"]
        assert (
            content_box["x"] + content_box["width"]
            <= navbar_box["x"] + navbar_box["width"] + 1
        )

    # -------------------------------------------------------------------------
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
                    '{"reports":[{"report_id":1,"report_version":5,"created_at":"2026-01-01T00:00:00Z",'
                    '"run_name":"mock run","dataset_name":"custom/sample","documents_processed":2,'
                    '"tokenizers_count":2,"tokenizers_processed":["ok/tokenizer","broken/tokenizer"],'
                    '"selected_metric_keys":["eff.encode_tokens_per_second_mean"]}],"total":1,"offset":0,"limit":25}'
                ),
            ),
        )
        page.route(
            "**/api/benchmarks/reports/1",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=(
                    '{"status":"success","schema_version":3,"methodology_version":"semantic_honesty",'
                    '"report_id":1,"report_version":5,"created_at":"2026-01-01T00:00:00Z","run_name":"mock run",'
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
                    '"dashboard":{"widgets":[],"available_widget_ids":[],"available_metric_keys":[],"unavailable_selected_metric_keys":[]},'
                    '"per_document_stats":[],'
                    '"runtime_metadata":{"benchmark_timing_boundaries":{"encode_only_definition":"encode only",'
                    '"dataset_stream_definition":"stream","postprocess_definition":"postprocess"}},'
                    '"raw_observations":{"broken/tokenizer":[{"error":"RuntimeError","message":"broken tokenizer"}]}}'
                ),
            ),
        )

        page.goto(f"{base_url}/cross-benchmark")
        expect(page.get_by_text("mock run", exact=True)).to_be_visible()
        page.get_by_role("button", name=re.compile(r"Reports \(1\)")).first.click()
        expect(page.get_by_role("dialog", name="Benchmark Reports")).to_be_visible()
        page.get_by_role("button", name=re.compile("mock run")).click()
        expect(page.get_by_text("Run Diagnostics")).to_be_visible()
        expect(
            page.get_by_text("broken/tokenizer: RuntimeError", exact=False)
        ).to_be_visible()
