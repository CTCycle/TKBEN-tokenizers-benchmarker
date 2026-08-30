"""Current-schema browser coverage for the cross-benchmark dashboard."""

from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Page, expect


###############################################################################
def _route_dashboard_api(page: Page) -> None:
    page.route(
        "**/api/tokenizers/list",
        lambda route: route.fulfill(json={"tokenizers": [], "count": 0}),
    )
    page.route(
        "**/api/datasets/list",
        lambda route: route.fulfill(json={"datasets": [], "count": 0}),
    )
    page.route(
        "**/api/benchmarks/metrics/catalog",
        lambda route: route.fulfill(json={"categories": []}),
    )
    page.route(
        "**/api/benchmarks/reports?*",
        lambda route: route.fulfill(
            json={
                "reports": [
                    {
                        "report_id": 101,
                        "report_version": 5,
                        "created_at": "2026-07-22T12:00:00Z",
                        "run_name": "Dashboard QA",
                        "dataset_name": "custom/qa",
                        "documents_processed": 2,
                        "tokenizers_count": 2,
                        "tokenizers_processed": ["alpha", "beta"],
                        "selected_metric_keys": [
                            "efficiency.speed",
                            "vocabulary.size",
                            "latency.distribution",
                            "fragmentation.bucket",
                        ],
                    }
                ],
                "total": 1,
                "offset": 0,
                "limit": 25,
            }
        ),
    )
    page.route(
        "**/api/benchmarks/reports/101",
        lambda route: route.fulfill(
            json={
                "status": "success",
                "schema_version": 3,
                "report_id": 101,
                "report_version": 5,
                "created_at": "2026-07-22T12:00:00Z",
                "run_name": "Dashboard QA",
                "dataset_name": "custom/qa",
                "documents_processed": 2,
                "tokenizers_count": 2,
                "tokenizers_processed": ["alpha", "beta"],
                "selected_metric_keys": [
                    "efficiency.speed",
                    "vocabulary.size",
                    "latency.distribution",
                    "fragmentation.bucket",
                ],
                "config": {
                    "max_documents": 2,
                    "warmup_trials": 1,
                    "timed_trials": 2,
                    "batch_size": 1,
                    "seed": 42,
                    "parallelism": 1,
                    "include_lm_metrics": False,
                    "add_special_tokens": False,
                    "padding": False,
                    "truncation": False,
                    "max_length": None,
                    "store_per_document_stats": False,
                    "per_document_sample_size": 2,
                },
                "hardware_profile": {
                    "runtime": "Python",
                    "os": "Windows",
                    "cpu_model": None,
                    "cpu_logical_cores": None,
                    "memory_total_mb": None,
                },
                "trial_summary": {"warmup_trials": 1, "timed_trials": 2},
                "tokenizer_results": [],
                "per_document_stats": [],
                "runtime_metadata": {},
                "raw_observations": {},
                "dashboard": {
                    "available_widget_ids": ["efficiency.speed", "vocabulary.size"],
                    "available_metric_keys": ["efficiency.speed", "vocabulary.size"],
                    "unavailable_selected_metric_keys": [],
                    "widgets": [
                        {
                            "widget_id": "efficiency.speed",
                            "metric_keys": ["efficiency.speed"],
                            "category_key": "efficiency",
                            "category_label": "Efficiency",
                            "label": "Tokenization speed",
                            "description": "Mean throughput",
                            "unit": "tokens/s",
                            "display_format": "number",
                            "default_visualization": "interval_bar",
                            "compatible_visualizations": [
                                "interval_bar",
                                "dot_whisker",
                            ],
                            "default_visible": True,
                            "width": "standard",
                            "points": [
                                {
                                    "tokenizer": "alpha",
                                    "value": 1500,
                                    "interval_low": 1400,
                                    "interval_high": 1600,
                                },
                                {
                                    "tokenizer": "beta",
                                    "value": 1200,
                                    "interval_low": 1100,
                                    "interval_high": 1300,
                                },
                            ],
                            "distributions": [],
                            "buckets": [],
                            "histogram_bins": [],
                        },
                        {
                            "widget_id": "vocabulary.size",
                            "metric_keys": ["vocabulary.size"],
                            "category_key": "vocabulary",
                            "category_label": "Vocabulary",
                            "label": "Vocabulary size",
                            "description": "Vocabulary entries",
                            "unit": "tokens",
                            "display_format": "number",
                            "default_visualization": "bar",
                            "compatible_visualizations": ["bar", "horizontal_bar"],
                            "default_visible": True,
                            "width": "standard",
                            "points": [
                                {
                                    "tokenizer": "alpha",
                                    "value": 5000,
                                    "interval_low": None,
                                    "interval_high": None,
                                },
                                {
                                    "tokenizer": "beta",
                                    "value": 6000,
                                    "interval_low": None,
                                    "interval_high": None,
                                },
                            ],
                            "distributions": [],
                            "buckets": [],
                            "histogram_bins": [],
                        },
                        {
                            "widget_id": "latency.distribution",
                            "metric_keys": ["latency.distribution"],
                            "category_key": "latency",
                            "category_label": "Latency",
                            "label": "Encode latency distribution",
                            "description": "Per-document latency",
                            "unit": "ms",
                            "display_format": "milliseconds",
                            "default_visualization": "box_plot",
                            "compatible_visualizations": ["box_plot", "histogram"],
                            "default_visible": True,
                            "width": "wide",
                            "points": [],
                            "distributions": [
                                {
                                    "tokenizer": "alpha",
                                    "min": 1,
                                    "q1": 2,
                                    "median": 3,
                                    "q3": 4,
                                    "max": 6,
                                    "sample_count": 10,
                                },
                                {
                                    "tokenizer": "beta",
                                    "min": 2,
                                    "q1": 3,
                                    "median": 4,
                                    "q3": 5,
                                    "max": 7,
                                    "sample_count": 10,
                                },
                            ],
                            "buckets": [],
                            "histogram_bins": [
                                {
                                    "tokenizer": "alpha",
                                    "bin_low": 1,
                                    "bin_high": 4,
                                    "count": 8,
                                    "proportion": 0.8,
                                },
                                {
                                    "tokenizer": "alpha",
                                    "bin_low": 4,
                                    "bin_high": 7,
                                    "count": 2,
                                    "proportion": 0.2,
                                },
                                {
                                    "tokenizer": "beta",
                                    "bin_low": 1,
                                    "bin_high": 4,
                                    "count": 7,
                                    "proportion": 0.7,
                                },
                                {
                                    "tokenizer": "beta",
                                    "bin_low": 4,
                                    "bin_high": 7,
                                    "count": 3,
                                    "proportion": 0.3,
                                },
                            ],
                        },
                        {
                            "widget_id": "fragmentation.bucket",
                            "metric_keys": ["fragmentation.bucket"],
                            "category_key": "fragmentation",
                            "category_label": "Fragmentation",
                            "label": "Word-length bucket comparison",
                            "description": "Pieces per word by bucket",
                            "unit": "pieces/word",
                            "display_format": "ratio",
                            "default_visualization": "grouped_bar",
                            "compatible_visualizations": ["grouped_bar"],
                            "default_visible": True,
                            "width": "wide",
                            "points": [],
                            "distributions": [],
                            "buckets": [
                                {
                                    "tokenizer": "alpha",
                                    "bucket": "short_1_4",
                                    "value": 1.2,
                                },
                                {
                                    "tokenizer": "alpha",
                                    "bucket": "long_9_plus",
                                    "value": 1.8,
                                },
                                {
                                    "tokenizer": "beta",
                                    "bucket": "short_1_4",
                                    "value": 1.1,
                                },
                                {
                                    "tokenizer": "beta",
                                    "bucket": "long_9_plus",
                                    "value": 2.0,
                                },
                            ],
                            "histogram_bins": [],
                        },
                    ],
                },
            }
        ),
    )


###############################################################################
def test_cross_benchmark_dashboard_customization_and_accessible_data(
    page: Page, base_url: str
) -> None:
    _route_dashboard_api(page)
    page.route(
        "**/api/exports/dashboard/pdf",
        lambda route: route.fulfill(status=500, json={"detail": "Export unavailable"}),
    )
    page.add_init_script(
        "if (!window.sessionStorage.getItem('tkben-visualization-fixture-seeded')) {"
        "window.localStorage.setItem('tkben:cross-benchmark-dashboard-layout:v3', "
        "JSON.stringify({version: 3, ordered_widget_ids: [], hidden_widget_ids: [], "
        "visualization_by_widget_id: {'vocabulary.size': 'heatmap'}}));"
        "window.sessionStorage.setItem('tkben-visualization-fixture-seeded', '1');}"
    )

    page.goto(f"{base_url}/cross-benchmark")
    expect(page.get_by_role("heading", name="Tokenization speed")).to_be_visible()
    expect(page.locator(".benchmark-visualization-button")).to_have_count(7)
    expect(
        page.get_by_role("button", name="Use Vertical bar chart for Vocabulary size")
    ).to_have_attribute("aria-pressed", "true")
    switcher = page.locator(".benchmark-visualization-switcher").first
    expect(switcher).to_have_css("border-style", "none")
    expect(switcher).to_have_css("background-color", "rgba(0, 0, 0, 0)")
    expect(switcher).to_have_css("padding", "0px")
    expect(page.locator(".benchmark-visualization-button").first).to_have_css(
        "width", "32px"
    )
    expect(page.locator(".benchmark-visualization-button").first).to_have_css(
        "height", "32px"
    )

    def assert_centered_content(widget_label: str, maximum_width: float) -> None:
        widget = page.locator(f'article[aria-label="{widget_label} widget"]')
        stage = widget.locator(".benchmark-chart-stage")
        content = widget.locator(".benchmark-chart-stage__content")
        stage_box = stage.bounding_box()
        content_box = content.bounding_box()
        assert stage_box is not None and content_box is not None
        assert (
            abs(
                (stage_box["x"] + stage_box["width"] / 2)
                - (content_box["x"] + content_box["width"] / 2)
            )
            <= 1
        )
        assert content_box["width"] <= maximum_width + 1
        assert content_box["width"] >= stage_box["width"] * 0.65

    assert_centered_content("Vocabulary size", 760)
    assert_centered_content("Encode latency distribution", 1280)
    bar_rectangles = page.locator(
        'article[aria-label="Vocabulary size widget"] .recharts-bar-rectangle'
    )
    expect(bar_rectangles).to_have_count(2)
    bar_widths = bar_rectangles.evaluate_all(
        "elements => elements.map(element => element.getBoundingClientRect().width)"
    )
    assert max(bar_widths) <= 45

    page.get_by_role(
        "button", name="Use Horizontal bar chart for Vocabulary size"
    ).click()
    expect(
        page.get_by_role("button", name="Use Horizontal bar chart for Vocabulary size")
    ).to_have_attribute("aria-pressed", "true")
    page.reload()
    expect(
        page.get_by_role("button", name="Use Horizontal bar chart for Vocabulary size")
    ).to_have_attribute("aria-pressed", "true")

    tables = page.get_by_text("View data table")
    expect(tables).to_have_count(4)
    tables.first.click()
    expect(page.get_by_text("Tokenization speed values by tokenizer")).to_be_visible()


###############################################################################
def test_cross_benchmark_report_manager_search_pagination_and_inline_delete(
    page: Page, base_url: str
) -> None:
    """Report management stays server-backed and uses inline deletion confirmation."""
    _route_dashboard_api(page)
    page.route(
        "**/api/exports/dashboard/pdf",
        lambda route: route.fulfill(status=500, json={"detail": "Export unavailable"}),
    )
    reports = [
        {
            "report_id": report_id,
            "report_version": 5,
            "created_at": f"2026-07-{(report_id % 28) + 1:02d}T12:00:00Z",
            "run_name": f"Report {report_id}",
            "dataset_name": "custom/qa",
            "documents_processed": 2,
            "tokenizers_count": 2,
            "tokenizers_processed": ["alpha", "beta"],
            "selected_metric_keys": [],
        }
        for report_id in range(101, 152)
    ]
    observed_urls: list[str] = []

    def route_reports(route) -> None:
        query = parse_qs(urlparse(route.request.url).query)
        observed_urls.append(route.request.url)
        search = query.get("search", [""])[0].casefold()
        offset = int(query.get("offset", ["0"])[0])
        limit = int(query.get("limit", ["25"])[0])
        filtered = [
            item
            for item in reports
            if not search or search in item["run_name"].casefold()
        ]
        route.fulfill(
            json={
                "reports": filtered[offset : offset + limit],
                "total": len(filtered),
                "offset": offset,
                "limit": limit,
            }
        )

    page.route("**/api/benchmarks/reports?*", route_reports)
    page.route(
        "**/api/benchmarks/reports/102",
        lambda route: (
            route.fulfill(status=204)
            if route.request.method == "DELETE"
            else route.fallback()
        ),
    )
    page.goto(f"{base_url}/cross-benchmark")
    page.get_by_role("button", name="Reports (51)").click()
    expect(page.get_by_role("dialog", name="Benchmark Reports")).to_be_visible()
    expect(page.locator(".benchmark-report-row")).to_have_count(25)
    expect(page.get_by_text("51 reports", exact=True)).to_be_visible()

    page.get_by_role("button", name="Next").click()
    page.wait_for_timeout(300)
    assert any("offset=25" in url for url in observed_urls)
    expect(page.get_by_text("Report 126", exact=True)).to_be_visible()
    page.get_by_role("button", name="Previous").click()
    page.wait_for_timeout(300)
    assert any("offset=0" in url for url in observed_urls)

    page.get_by_label("Search reports").fill("Report 102")
    page.wait_for_timeout(350)
    expect(page.get_by_text("Report 102", exact=True)).to_be_visible()
    delete_button = page.get_by_role("button", name="Delete report")
    delete_button.click()
    expect(
        page.get_by_text("Delete this report permanently?", exact=True)
    ).to_be_visible()
    page.get_by_role("button", name="Cancel").click()
    expect(
        page.get_by_text("Delete this report permanently?", exact=True)
    ).to_have_count(0)
    delete_button.click()
    with page.expect_request(
        lambda request: (
            request.method == "DELETE" and request.url.endswith("/reports/102")
        )
    ):
        page.get_by_role("button", name="Delete", exact=True).click()
    page.wait_for_timeout(300)
    expect(page.get_by_role("heading", name="Tokenization speed")).to_be_visible()
    page.get_by_role("dialog", name="Benchmark Reports").get_by_role(
        "button", name="Close benchmark report manager"
    ).click()

    customize = page.get_by_role("button", name="Customize benchmark dashboard")
    customize.click()
    expect(
        page.get_by_role("dialog", name="Customize benchmark dashboard")
    ).to_be_visible()
    page.get_by_role(
        "checkbox", name="Vocabulary size — Vocabulary entries (tokens, bar)"
    ).uncheck()
    page.get_by_role("button", name="Apply").click()

    restore = page.get_by_role("button", name="Restore default layout")
    expect(restore).to_be_enabled()
    restore.click()
    expect(restore).to_be_disabled()
    expect(
        page.get_by_role("button", name="Use Vertical bar chart for Vocabulary size")
    ).to_have_attribute("aria-pressed", "true")

    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("window.innerWidth >= document.documentElement.scrollWidth")

    page.get_by_role("button", name="Export dashboard report as PDF").click()
    expect(page.get_by_role("alert")).to_contain_text("Export unavailable")
