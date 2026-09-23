"""Live local coverage for the populated Cross Benchmark workflow."""

import os
from pathlib import Path
from uuid import uuid4

import pytest
from playwright.sync_api import APIRequestContext, Page, expect


RUN_BENCHMARKS = os.getenv("E2E_RUN_BENCHMARKS", "").lower() in {
    "1",
    "true",
    "yes",
}
REPO_ROOT = Path(__file__).resolve().parents[3]
SCREENSHOT_PATH = (
    REPO_ROOT / "assets" / "QA" / "tkben-t2-06-cross-benchmark-wizard-20260923.png"
)


@pytest.mark.skipif(
    not RUN_BENCHMARKS,
    reason="Set E2E_RUN_BENCHMARKS=1 to enable local benchmark execution.",
)
def test_cross_benchmark_wizard_runs_and_reloads_local_report(
    api_context: APIRequestContext,
    base_url: str,
    job_waiter,
    page: Page,
    tiny_tokenizer_json: bytes,
) -> None:
    """A local report created in the wizard renders and reloads from storage."""
    stem = f"t2_06_{uuid4().hex[:8]}"
    tokenizer_stem = uuid4().hex[:5]
    dataset_name = f"custom/{stem}"
    tokenizer_name: str | None = None
    report_id: int | None = None
    dataset_created = False
    browser_errors: list[str] = []
    browser_http_errors: list[tuple[int, str]] = []

    try:
        dataset_response = api_context.post(
            "/api/datasets/upload",
            multipart={
                "file": {
                    "name": f"{stem}.csv",
                    "mimeType": "text/csv",
                    "buffer": b"text\nHello world\nThis is a sample document\n",
                }
            },
        )
        assert dataset_response.ok, dataset_response.text()
        dataset_created = True
        dataset_job = dataset_response.json()
        dataset_job_id = str(dataset_job.get("job_id", ""))
        assert dataset_job_id, "Missing dataset upload job ID"
        dataset_status = job_waiter(
            dataset_job_id,
            poll_interval=dataset_job.get("poll_interval", 1.0),
            timeout_seconds=300.0,
        )
        assert dataset_status.get("status") == "completed", dataset_status.get("error")
        assert dataset_status.get("result", {}).get("dataset_name") == dataset_name

        tokenizer_response = api_context.post(
            "/api/tokenizers/upload",
            multipart={
                "file": {
                    "name": f"{tokenizer_stem}.json",
                    "mimeType": "application/json",
                    "buffer": tiny_tokenizer_json,
                }
            },
        )
        assert tokenizer_response.ok, tokenizer_response.text()
        tokenizer = tokenizer_response.json()
        tokenizer_name = str(tokenizer.get("tokenizer_name", ""))
        assert tokenizer_name == f"CUSTOM_{tokenizer_stem}"
        assert tokenizer.get("is_compatible") is True

        catalog_response = api_context.get("/api/benchmarks/metrics/catalog")
        assert catalog_response.ok, catalog_response.text()
        catalog = catalog_response.json()
        metric_options = [
            metric
            for category in catalog.get("categories", [])
            for metric in category.get("metrics", [])
        ]
        selected_metric_key = "eff.encode_tokens_per_second_mean"
        selected_metric_index = next(
            (
                index
                for index, metric in enumerate(metric_options)
                if metric.get("key") == selected_metric_key
            ),
            None,
        )
        assert selected_metric_index is not None, (
            f"Metric {selected_metric_key} is missing from the catalog"
        )

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
        page.set_viewport_size({"width": 1440, "height": 900})
        page.goto(f"{base_url}/cross-benchmark")

        page.get_by_role("button", name="Run benchmark").click()
        dialog = page.get_by_role("dialog", name="Run benchmark")
        expect(dialog).to_be_visible()

        metric_checkboxes = dialog.locator(
            ".benchmark-wizard-tree-children input[type='checkbox']"
        )
        expect(metric_checkboxes).to_have_count(len(metric_options))
        for index in range(len(metric_options)):
            checkbox = metric_checkboxes.nth(index)
            if checkbox.is_checked() and index != selected_metric_index:
                checkbox.uncheck()
            elif index == selected_metric_index and not checkbox.is_checked():
                checkbox.check()
        dialog.get_by_role("button", name="Next").click()

        tokenizer_option = dialog.get_by_text(tokenizer_name, exact=True).locator(
            "xpath=ancestor::label[1]"
        )
        tokenizer_option.locator("input[type='checkbox']").check()
        dialog.locator("#benchmark-dataset").select_option(value=dataset_name)
        document_range = dialog.locator("#benchmark-documents")
        document_range.focus()
        document_range.press("Home")
        document_range.press("ArrowRight")
        document_range.press("ArrowRight")
        expect(document_range).to_have_value("2")
        dialog.get_by_role("button", name="Next").click()

        run_name = f"T2-06 {stem}"
        dialog.get_by_label("Run Name").fill(run_name)
        dialog.get_by_label("Warmup trials").fill("0")
        dialog.get_by_label("Timed trials").fill("1")
        dialog.get_by_label("Batch size").fill("1")
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.endswith("/api/benchmarks/run")
            ),
            timeout=30_000,
        ) as run_response_info:
            dialog.get_by_role("button", name="Start benchmark").click()

        run_response = run_response_info.value
        assert run_response.ok, run_response.text()
        run_job = run_response.json()
        run_job_id = str(run_job.get("job_id", ""))
        assert run_job_id, "Missing benchmark job ID"
        run_status = job_waiter(
            run_job_id,
            poll_interval=run_job.get("poll_interval", 1.0),
            timeout_seconds=300.0,
        )
        assert run_status.get("status") == "completed", run_status.get("error")
        generated_report = run_status.get("result", {})
        raw_report_id = generated_report.get("report_id")
        if raw_report_id is not None:
            report_id = int(raw_report_id)
        assert generated_report.get("status") == "success"
        assert generated_report.get("dataset_name") == dataset_name
        assert generated_report.get("documents_processed") == 2
        assert generated_report.get("selected_metric_keys") == [selected_metric_key]
        assert generated_report.get("config", {}).get("max_documents") == 2
        assert generated_report.get("config", {}).get("warmup_trials") == 0
        assert generated_report.get("config", {}).get("timed_trials") == 1

        assert raw_report_id is not None
        assert report_id is not None
        stored_response = api_context.get(f"/api/benchmarks/reports/{report_id}")
        assert stored_response.ok, stored_response.text()
        stored_report = stored_response.json()
        assert int(stored_report.get("report_id", 0)) == report_id
        assert stored_report.get("run_name") == run_name
        assert stored_report.get("dataset_name") == dataset_name

        widgets = stored_report.get("dashboard", {}).get("widgets", [])
        assert widgets, "The persisted report has no dashboard widgets"
        widget_label = str(widgets[0].get("label", ""))
        assert widget_label
        current_report = page.locator('section[aria-label="Current report"]')
        expect(current_report.get_by_text(run_name, exact=True)).to_be_visible(
            timeout=300_000
        )
        expect(page.get_by_role("heading", name=widget_label)).to_be_visible()
        report_widget = page.locator('article[aria-label$=" widget"]').first
        expect(report_widget).to_be_visible()
        expect(
            report_widget.locator(".benchmark-chart-stage svg").first
        ).to_be_visible()

        report_url = f"/api/benchmarks/reports/{report_id}"
        with page.expect_response(
            lambda response: (
                response.request.method == "GET" and response.url.endswith(report_url)
            ),
            timeout=30_000,
        ) as reload_report_info:
            page.reload()
        reloaded_report_response = reload_report_info.value
        assert reloaded_report_response.ok, reloaded_report_response.text()
        assert int(reloaded_report_response.json().get("report_id", 0)) == report_id
        current_report = page.locator('section[aria-label="Current report"]')
        expect(current_report.get_by_text(run_name, exact=True)).to_be_visible(
            timeout=30_000
        )
        expect(page.get_by_role("heading", name=widget_label)).to_be_visible()
        expect(
            page.locator('article[aria-label$=" widget"]')
            .first.locator(".benchmark-chart-stage svg")
            .first
        ).to_be_visible()
        assert browser_errors == [], browser_errors
        assert browser_http_errors == [], browser_http_errors

        SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
    finally:
        if report_id is not None:
            delete_report = api_context.delete(f"/api/benchmarks/reports/{report_id}")
            assert delete_report.status in {204, 404}, delete_report.text()
            assert api_context.get(f"/api/benchmarks/reports/{report_id}").status == 404
        if tokenizer_name:
            delete_tokenizer = api_context.delete(
                "/api/tokenizers/delete", params={"tokenizer_name": tokenizer_name}
            )
            assert delete_tokenizer.status in {200, 404}, delete_tokenizer.text()
            refreshed_tokenizers = api_context.get("/api/tokenizers/list")
            assert refreshed_tokenizers.ok, refreshed_tokenizers.text()
            assert tokenizer_name not in {
                str(item.get("tokenizer_name"))
                for item in refreshed_tokenizers.json().get("tokenizers", [])
            }
        if dataset_created:
            delete_dataset = api_context.delete(
                "/api/datasets/delete", params={"dataset_name": dataset_name}
            )
            assert delete_dataset.status in {200, 404}, delete_dataset.text()
            refreshed_datasets = api_context.get("/api/datasets/list")
            assert refreshed_datasets.ok, refreshed_datasets.text()
            assert dataset_name not in {
                str(item.get("dataset_name"))
                for item in refreshed_datasets.json().get("datasets", [])
            }
