"""Current-schema browser coverage for the cross-benchmark dashboard."""

from playwright.sync_api import Page, expect

###############################################################################
def _route_dashboard_api(page: Page) -> None:
    page.route("**/api/tokenizers/list", lambda route: route.fulfill(json={"tokenizers": [], "count": 0}))
    page.route("**/api/datasets/list", lambda route: route.fulfill(json={"datasets": [], "count": 0}))
    page.route("**/api/benchmarks/metrics/catalog", lambda route: route.fulfill(json={"categories": []}))
    page.route(
        "**/api/benchmarks/reports?*",
        lambda route: route.fulfill(json={"reports": [{"report_id": 101, "report_version": 4, "created_at": "2026-07-22T12:00:00Z", "run_name": "Dashboard QA", "dataset_name": "custom/qa", "documents_processed": 2, "tokenizers_count": 2, "tokenizers_processed": ["alpha", "beta"], "selected_metric_keys": ["efficiency.speed", "vocabulary.size", "latency.distribution", "fragmentation.bucket"]}]}),
    )
    page.route(
        "**/api/benchmarks/reports/101",
        lambda route: route.fulfill(json={
            "status": "success", "schema_version": 2, "report_id": 101, "report_version": 4, "created_at": "2026-07-22T12:00:00Z", "run_name": "Dashboard QA", "dataset_name": "custom/qa", "documents_processed": 2, "tokenizers_count": 2, "tokenizers_processed": ["alpha", "beta"], "selected_metric_keys": ["efficiency.speed", "vocabulary.size", "latency.distribution", "fragmentation.bucket"],
            "config": {"max_documents": 2, "warmup_trials": 1, "timed_trials": 2, "batch_size": 1, "seed": 42, "parallelism": 1, "include_lm_metrics": False, "add_special_tokens": False, "padding": False, "truncation": False, "max_length": None, "store_per_document_stats": False, "per_document_sample_size": 2},
            "hardware_profile": {"runtime": "Python", "os": "Windows", "cpu_model": None, "cpu_logical_cores": None, "memory_total_mb": None}, "trial_summary": {"warmup_trials": 1, "timed_trials": 2}, "tokenizer_results": [], "per_document_stats": [], "runtime_metadata": {}, "raw_observations": {},
            "dashboard": {"available_widget_ids": ["efficiency.speed", "vocabulary.size"], "available_metric_keys": ["efficiency.speed", "vocabulary.size"], "unavailable_selected_metric_keys": [], "widgets": [
                {"widget_id": "efficiency.speed", "metric_keys": ["efficiency.speed"], "category_key": "efficiency", "category_label": "Efficiency", "label": "Tokenization speed", "description": "Mean throughput", "unit": "tokens/s", "display_format": "number", "default_visualization": "interval_bar", "compatible_visualizations": ["interval_bar", "dot_whisker"], "default_visible": True, "width": "wide", "points": [{"tokenizer": "alpha", "value": 1500, "interval_low": 1400, "interval_high": 1600}, {"tokenizer": "beta", "value": 1200, "interval_low": 1100, "interval_high": 1300}], "distributions": [], "buckets": []},
                {"widget_id": "vocabulary.size", "metric_keys": ["vocabulary.size"], "category_key": "vocabulary", "category_label": "Vocabulary", "label": "Vocabulary size", "description": "Vocabulary entries", "unit": "tokens", "display_format": "number", "default_visualization": "bar", "compatible_visualizations": ["bar", "lollipop"], "default_visible": True, "width": "standard", "points": [{"tokenizer": "alpha", "value": 5000, "interval_low": None, "interval_high": None}, {"tokenizer": "beta", "value": 6000, "interval_low": None, "interval_high": None}], "distributions": [], "buckets": []},
                {"widget_id": "latency.distribution", "metric_keys": ["latency.distribution"], "category_key": "latency", "category_label": "Latency", "label": "Encode latency distribution", "description": "Per-document latency", "unit": "ms", "display_format": "milliseconds", "default_visualization": "box_plot", "compatible_visualizations": ["box_plot", "range_plot"], "default_visible": True, "width": "wide", "points": [], "distributions": [{"tokenizer": "alpha", "min": 1, "q1": 2, "median": 3, "q3": 4, "max": 6, "sample_count": 10}, {"tokenizer": "beta", "min": 2, "q1": 3, "median": 4, "q3": 5, "max": 7, "sample_count": 10}], "buckets": []},
                {"widget_id": "fragmentation.bucket", "metric_keys": ["fragmentation.bucket"], "category_key": "fragmentation", "category_label": "Fragmentation", "label": "Word-length bucket comparison", "description": "Pieces per word by bucket", "unit": "pieces/word", "display_format": "ratio", "default_visualization": "grouped_bar", "compatible_visualizations": ["grouped_bar", "heatmap"], "default_visible": True, "width": "wide", "points": [], "distributions": [], "buckets": [{"tokenizer": "alpha", "bucket": "short_1_4", "value": 1.2}, {"tokenizer": "alpha", "bucket": "long_9_plus", "value": 1.8}, {"tokenizer": "beta", "bucket": "short_1_4", "value": 1.1}, {"tokenizer": "beta", "bucket": "long_9_plus", "value": 2.0}]}
            ]},
        }),
    )

###############################################################################
def test_cross_benchmark_dashboard_customization_and_accessible_data(
    page: Page, base_url: str
) -> None:
    _route_dashboard_api(page)
    page.route("**/api/exports/dashboard/pdf", lambda route: route.fulfill(status=500, json={"detail": "Export unavailable"}))
    page.add_init_script(
        "if (!window.sessionStorage.getItem('tkben-visualization-fixture-seeded')) {"
        "window.localStorage.setItem('tkben:cross-benchmark-dashboard-layout:v2', "
        "JSON.stringify({version: 2, order: [], hidden_widget_ids: [], known_widget_ids: [], "
        "visualization_by_widget_id: {'vocabulary.size': 'heatmap'}}));"
        "window.sessionStorage.setItem('tkben-visualization-fixture-seeded', '1');}"
    )

    page.goto(f"{base_url}/cross-benchmark")
    expect(page.get_by_role("heading", name="Tokenization speed")).to_be_visible()
    expect(page.locator(".benchmark-visualization-button")).to_have_count(8)
    expect(page.get_by_role("button", name="Use Bar chart for Vocabulary size")).to_have_attribute("aria-pressed", "true")
    expect(page.get_by_role("button", name="Use Heatmap for Word-length bucket comparison")).to_be_visible()

    page.get_by_role("button", name="Use Lollipop chart for Vocabulary size").click()
    expect(page.get_by_role("button", name="Use Lollipop chart for Vocabulary size")).to_have_attribute("aria-pressed", "true")
    page.reload()
    expect(page.get_by_role("button", name="Use Lollipop chart for Vocabulary size")).to_have_attribute("aria-pressed", "true")

    tables = page.get_by_text("View data table")
    expect(tables).to_have_count(4)
    tables.first.click()
    expect(page.get_by_text("Tokenization speed values by tokenizer")).to_be_visible()
    expect(page.get_by_role("cell", name="1500")).to_be_visible()

    customize = page.get_by_role("button", name="Customize benchmark dashboard")
    customize.click()
    expect(page.get_by_role("dialog", name="Customize benchmark dashboard")).to_be_visible()
    page.get_by_role("checkbox", name="Vocabulary size — Vocabulary entries (tokens, bar)").uncheck()
    page.get_by_role("button", name="Apply").click()

    restore = page.get_by_role("button", name="Restore default layout")
    expect(restore).to_be_enabled()
    restore.click()
    expect(restore).to_be_disabled()
    expect(page.get_by_role("button", name="Use Bar chart for Vocabulary size")).to_have_attribute("aria-pressed", "true")

    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("window.innerWidth >= document.documentElement.scrollWidth")

    page.get_by_role("button", name="Export dashboard report as PDF").click()
    expect(page.get_by_role("alert")).to_contain_text("Export unavailable")
