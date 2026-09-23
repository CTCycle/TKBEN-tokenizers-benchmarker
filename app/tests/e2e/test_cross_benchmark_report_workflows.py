"""Live local coverage for populated Cross Benchmark report workflows."""

import os
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from playwright.sync_api import APIRequestContext, Page, expect


RUN_BENCHMARKS = os.getenv("E2E_RUN_BENCHMARKS", "").lower() in {
    "1",
    "true",
    "yes",
}
REPO_ROOT = Path(__file__).resolve().parents[3]
QA_DIR = REPO_ROOT / "assets" / "QA"


def _create_report_in_wizard(
    *,
    api_context: APIRequestContext,
    base_url: str,
    job_waiter,
    page: Page,
    dataset_name: str,
    tokenizer_names: list[str],
    run_name: str,
    registered_report_ids: set[int],
) -> dict[str, Any]:
    catalog_response = api_context.get("/api/benchmarks/metrics/catalog")
    assert catalog_response.ok, catalog_response.text()
    metric_options = [
        metric
        for category in catalog_response.json().get("categories", [])
        for metric in category.get("metrics", [])
    ]
    metric_key = "eff.encode_tokens_per_second_mean"
    selected_metric_index = next(
        (index for index, metric in enumerate(metric_options) if metric.get("key") == metric_key),
        None,
    )
    assert selected_metric_index is not None, f"Metric {metric_key} is missing"

    page.goto(f"{base_url}/cross-benchmark")
    page.get_by_role("button", name="Run benchmark").click()
    dialog = page.get_by_role("dialog", name="Run benchmark")
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

    for tokenizer_name in tokenizer_names:
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

    dialog.get_by_label("Run Name").fill(run_name)
    dialog.get_by_label("Warmup trials").fill("0")
    dialog.get_by_label("Timed trials").fill("1")
    dialog.get_by_label("Batch size").fill("1")
    with page.expect_response(
        lambda response: response.request.method == "POST"
        and response.url.endswith("/api/benchmarks/run"),
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
    report_id = run_status.get("result", {}).get("report_id")
    assert report_id is not None
    registered_report_ids.add(int(report_id))

    stored_response = api_context.get(f"/api/benchmarks/reports/{int(report_id)}")
    assert stored_response.ok, stored_response.text()
    report = stored_response.json()
    assert report.get("run_name") == run_name
    assert report.get("dataset_name") == dataset_name
    assert report.get("tokenizers_processed") == tokenizer_names
    assert report.get("config", {}).get("max_documents") == 2
    assert report.get("config", {}).get("warmup_trials") == 0
    assert report.get("config", {}).get("timed_trials") == 1
    expect(
        page.locator('section[aria-label="Current report"]').get_by_text(
            run_name, exact=True
        )
    ).to_be_visible(timeout=30_000)
    return report


def _seed_manager_reports(
    source_report: dict[str, Any], stem: str, count: int
) -> list[int]:
    """Persist current-schema manager rows in the isolated application database."""
    from server.services.benchmark_reports import BenchmarkReportService

    service = BenchmarkReportService()
    payload = deepcopy(source_report)
    for key in ("report_id", "tags"):
        payload.pop(key, None)

    created: list[int] = []
    now = datetime.now(timezone.utc)
    try:
        for index in range(count):
            fixture = deepcopy(payload)
            fixture["run_name"] = f"{stem} manager fixture {index:02d}"
            fixture["created_at"] = (
                (now - timedelta(days=1, seconds=index))
                .isoformat()
                .replace("+00:00", "Z")
            )
            created.append(service.save_benchmark_report(fixture))
    except Exception:
        for report_id in reversed(created):
            service.delete_benchmark_report(report_id)
        raise
    return created


def _assert_live_announcement_is_visually_hidden(page: Page) -> None:
    announcer = page.locator(".cdk-live-announcer-element")
    expect(announcer).to_have_text("Dialog closed")
    box = announcer.bounding_box()
    assert box is not None
    assert box["width"] <= 1 and box["height"] <= 1, (
        f"The accessibility announcement is rendered visibly: {box}"
    )


@pytest.mark.skipif(
    not RUN_BENCHMARKS,
    reason="Set E2E_RUN_BENCHMARKS=1 to enable local benchmark execution.",
)
def test_live_cross_benchmark_report_manager_and_dashboard_workflows(
    api_context: APIRequestContext,
    base_url: str,
    job_waiter,
    page: Page,
    tiny_tokenizer_json: bytes,
) -> None:
    """Real reports retain manager edits and dashboard choices across reloads."""
    stem = f"t3_0304_{uuid4().hex[:8]}"
    dataset_name = f"custom/{stem}"
    tokenizer_names: list[str] = []
    report_ids: set[int] = set()
    fixture_ids: list[int] = []
    dataset_created = False
    browser_errors: list[str] = []
    browser_http_errors: list[tuple[int, str]] = []

    try:
        QA_DIR.mkdir(parents=True, exist_ok=True)
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

        for suffix in ("alpha", "beta"):
            tokenizer_response = api_context.post(
                "/api/tokenizers/upload",
                multipart={
                    "file": {
                        "name": f"{stem}_{suffix}.json",
                        "mimeType": "application/json",
                        "buffer": tiny_tokenizer_json,
                    }
                },
            )
            assert tokenizer_response.ok, tokenizer_response.text()
            tokenizer_name = str(tokenizer_response.json().get("tokenizer_name", ""))
            assert tokenizer_name == f"CUSTOM_{stem}_{suffix}"
            tokenizer_names.append(tokenizer_name)

        page.on("pageerror", lambda error: browser_errors.append(str(error)))
        page.on(
            "console",
            lambda message: browser_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on(
            "response",
            lambda response: browser_http_errors.append((response.status, response.url))
            if response.status >= 400
            else None,
        )
        page.set_viewport_size({"width": 1440, "height": 900})

        one_tokenizer_name = f"{stem} one tokenizer"
        one_tokenizer_report = _create_report_in_wizard(
            api_context=api_context,
            base_url=base_url,
            job_waiter=job_waiter,
            page=page,
            dataset_name=dataset_name,
            tokenizer_names=[tokenizer_names[0]],
            run_name=one_tokenizer_name,
            registered_report_ids=report_ids,
        )
        point_widget = next(
            (
                widget
                for widget in one_tokenizer_report.get("dashboard", {}).get(
                    "widgets", []
                )
                if len(widget.get("points", [])) == 1
            ),
            None,
        )
        assert point_widget is not None, "One-tokenizer report has no point widget"
        point_card = page.locator(
            f'article[aria-label="{point_widget["label"]} widget"]'
        )
        tick_label = point_card.locator(".benchmark-axis-tick")
        expect(tick_label).to_have_attribute("aria-label", tokenizer_names[0])
        expect(tick_label).to_have_attribute("text-anchor", "middle")
        assert tick_label.get_attribute("transform") is None
        tick_box = tick_label.bounding_box()
        axis_title = point_card.locator(".benchmark-axis-label").filter(
            has_text="Tokenizer"
        )
        axis_box = axis_title.bounding_box()
        assert tick_box is not None and axis_box is not None
        assert tick_box["y"] + tick_box["height"] <= axis_box["y"], (
            "Single-tokenizer tick label overlaps the Tokenizer axis title"
        )
        point_card.screenshot(
            path=str(QA_DIR / "tkben-t3-03-single-tokenizer-chart-20260923.png")
        )

        two_tokenizer_name = f"{stem} two tokenizer"
        two_tokenizer_report = _create_report_in_wizard(
            api_context=api_context,
            base_url=base_url,
            job_waiter=job_waiter,
            page=page,
            dataset_name=dataset_name,
            tokenizer_names=tokenizer_names,
            run_name=two_tokenizer_name,
            registered_report_ids=report_ids,
        )
        two_tokenizer_id = int(two_tokenizer_report["report_id"])
        fixture_ids = _seed_manager_reports(two_tokenizer_report, stem, count=24)
        report_ids.update(fixture_ids)

        report_list_response = api_context.get(
            "/api/benchmarks/reports",
            params={"search": stem, "offset": 0, "limit": 100},
        )
        assert report_list_response.ok, report_list_response.text()
        assert report_list_response.json().get("total") == 26

        page.reload()
        report_selector = page.locator(".cross-benchmark-report-selector")
        expect(report_selector).to_have_text("Reports (26)")
        expect(
            page.locator('section[aria-label="Current report"]').get_by_text(
                two_tokenizer_name, exact=True
            )
        ).to_be_visible()
        review_pause_seconds = min(
            120.0, max(0.0, float(os.getenv("E2E_BROWSER_REVIEW_PAUSE_SECONDS", "0")))
        )
        if review_pause_seconds:
            print(
                f"Browser review pause: {review_pause_seconds:g} seconds with "
                f"{report_list_response.json().get('total')} isolated reports loaded.",
                flush=True,
            )
            time.sleep(review_pause_seconds)

        baseline = page.get_by_label("Baseline")
        baseline.select_option(tokenizer_names[0])
        comparison_labels = page.locator(".benchmark-comparison-strip__label")
        expect(comparison_labels).to_have_count(3)
        expect(comparison_labels.first).to_have_text(
            f"Baseline: {tokenizer_names[0]}"
        )
        page.get_by_text("View data table").first.click()
        comparison_table = page.locator("table.benchmark-data-table").first
        expect(comparison_table.get_by_role("columnheader", name="Δ vs baseline")).to_be_visible()
        baseline_row = comparison_table.get_by_role(
            "rowheader", name=tokenizer_names[0]
        ).locator("xpath=..")
        candidate_row = comparison_table.get_by_role(
            "rowheader", name=tokenizer_names[1]
        ).locator("xpath=..")
        expect(baseline_row.locator("td").last).to_have_text("Baseline")
        candidate_delta = candidate_row.locator("td").last.inner_text()
        assert candidate_delta.endswith("%"), candidate_delta
        float(candidate_delta.removesuffix("%").replace("+", ""))
        assert page.locator(".benchmark-comparison-strip").count() > 0
        stored_baselines = page.evaluate(
            "() => JSON.parse(localStorage.getItem('tkben:cross-benchmark-baselines:v1') || '{}')"
        )
        assert stored_baselines.get(str(two_tokenizer_id)) == tokenizer_names[0], (
            f"Baseline preference was not persisted before reload: {stored_baselines}"
        )

        with page.expect_response(
            lambda response: response.request.method == "GET"
            and response.url.endswith(
                f"/api/benchmarks/reports/{two_tokenizer_id}"
            )
        ) as reloaded_report_response_info:
            page.reload()
        reloaded_report_response = reloaded_report_response_info.value
        assert reloaded_report_response.ok, reloaded_report_response.text()
        assert reloaded_report_response.json().get("report_id") == two_tokenizer_id
        expect(
            page.locator('section[aria-label="Current report"]').get_by_text(
                two_tokenizer_name, exact=True
            )
        ).to_be_visible()
        restored_baselines = page.evaluate(
            "() => JSON.parse(localStorage.getItem('tkben:cross-benchmark-baselines:v1') || '{}')"
        )
        assert restored_baselines.get(str(two_tokenizer_id)) == tokenizer_names[0], (
            f"Baseline preference storage changed during reload: {restored_baselines}"
        )
        reloaded_baseline_value = page.get_by_label("Baseline").input_value()
        reloaded_comparison_count = page.locator(
            ".benchmark-comparison-strip__label"
        ).count()
        assert reloaded_baseline_value == tokenizer_names[0], (
            "Baseline preference did not hydrate into the rendered dashboard: "
            f"report={two_tokenizer_id}, value={reloaded_baseline_value!r}, "
            f"comparison_strips={reloaded_comparison_count}, "
            f"preferences={restored_baselines}"
        )
        expect(page.locator(".benchmark-comparison-strip__label").first).to_have_text(
            f"Baseline: {tokenizer_names[0]}"
        )

        reports_button = page.locator(".cross-benchmark-report-selector")
        reports_button.click()
        manager = page.get_by_role("dialog", name="Benchmark Reports")
        expect(manager.locator(".benchmark-report-row")).to_have_count(25)
        expect(manager.get_by_text("Showing 1–25 of 26", exact=True)).to_be_visible()
        with page.expect_response(
            lambda response: response.request.method == "GET"
            and "/api/benchmarks/reports?" in response.url
            and "offset=25" in response.url
        ):
            manager.get_by_role("button", name="Next").click()
        expect(manager.locator(".benchmark-report-row")).to_have_count(1)
        expect(manager.get_by_text("Showing 26–26 of 26", exact=True)).to_be_visible()
        with page.expect_response(
            lambda response: response.request.method == "GET"
            and "/api/benchmarks/reports?" in response.url
            and "offset=0" in response.url
        ):
            manager.get_by_role("button", name="Previous").click()
        expect(manager.locator(".benchmark-report-row")).to_have_count(25)

        tag_row = manager.locator(".benchmark-report-row").filter(
            has_text=two_tokenizer_name
        )
        expect(tag_row).to_have_count(1)
        tag_row.get_by_role("button", name="Edit tags").click()
        tag_input = tag_row.locator(f"#report-tags-{two_tokenizer_id}")
        tag_input.fill(f"live, {stem}")
        with page.expect_response(
            lambda response: response.request.method == "PATCH"
            and response.url.endswith(f"/reports/{two_tokenizer_id}/tags")
        ) as tag_response_info:
            tag_row.get_by_role("button", name="Save tags").click()
        assert tag_response_info.value.ok
        persisted_tags = api_context.get(
            f"/api/benchmarks/reports/{two_tokenizer_id}"
        ).json().get("tags")
        assert persisted_tags == ["live", stem]
        expect(tag_row.get_by_text("live", exact=True)).to_be_visible()
        manager.get_by_role(
            "button", name="Close benchmark report manager"
        ).click()
        page.reload()
        page.locator(".cross-benchmark-report-selector").click()
        manager = page.get_by_role("dialog", name="Benchmark Reports")
        persisted_tag_row = manager.locator(".benchmark-report-row").filter(
            has_text=two_tokenizer_name
        )
        expect(persisted_tag_row.get_by_text("live", exact=True)).to_be_visible()
        expect(persisted_tag_row.get_by_text(stem, exact=True)).to_be_visible()
        persisted_tag_row.locator(".benchmark-report-row-select").click()
        expect(
            page.locator('section[aria-label="Current report"]').get_by_text(
                two_tokenizer_name, exact=True
            )
        ).to_be_visible()

        page.get_by_role("button", name="Clone benchmark").click()
        clone_dialog = page.get_by_role("dialog", name="Clone benchmark")
        expect(clone_dialog).to_be_visible()
        expect(clone_dialog.get_by_label("Run Name")).to_have_value(
            f"Clone of {two_tokenizer_name}"
        )
        expect(clone_dialog.get_by_label("Timed trials")).to_have_value("1")
        expect(clone_dialog.get_by_label("Warmup trials")).to_have_value("0")
        clone_dialog.get_by_role("button", name="Back").click()
        expect(clone_dialog.locator("#benchmark-dataset")).to_have_value(dataset_name)
        selected_tokenizers = clone_dialog.locator(
            ".benchmark-wizard-tokenizer-option input[type='checkbox']"
        )
        expect(selected_tokenizers).to_have_count(2)
        expect(selected_tokenizers.nth(0)).to_be_checked()
        expect(selected_tokenizers.nth(1)).to_be_checked()
        page.keyboard.press("Escape")
        expect(clone_dialog).to_have_count(0)

        for width, height in ((1920, 1080), (1440, 900), (1024, 768), (390, 844)):
            page.set_viewport_size({"width": width, "height": height})
            expect(
                page.locator('section[aria-label="Current report"]').get_by_text(
                    two_tokenizer_name, exact=True
                )
            ).to_be_visible()
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            ), f"Page overflows horizontally at {width}x{height}"
            page.screenshot(
                path=str(QA_DIR / f"tkben-t3-03-04-cross-benchmark-{width}x{height}-20260923.png"),
                full_page=True,
            )

        for width, height in ((1920, 1080), (1440, 900), (1024, 768), (390, 844)):
            page.set_viewport_size({"width": width, "height": height})
            page.locator(".cross-benchmark-report-selector").click()
            manager = page.get_by_role("dialog", name="Benchmark Reports")
            manager_box = manager.evaluate(
                "element => { const { x, y, width, height } = element.getBoundingClientRect(); return { x, y, width, height }; }"
            )
            assert manager_box["x"] >= 0 and manager_box["x"] + manager_box["width"] <= width
            assert manager_box["y"] >= 0 and manager_box["y"] + manager_box["height"] <= height
            if width == 390:
                page.screenshot(
                    path=str(QA_DIR / "tkben-t3-03-report-manager-390x844-20260923.png")
                )
            page.keyboard.press("Escape")
            expect(page.locator(".cross-benchmark-report-selector")).to_be_focused()
            _assert_live_announcement_is_visually_hidden(page)

            clone_button = page.get_by_role("button", name="Clone benchmark")
            clone_button.click()
            clone_dialog = page.get_by_role("dialog", name="Clone benchmark")
            clone_box = clone_dialog.evaluate(
                "element => { const { x, y, width, height } = element.getBoundingClientRect(); return { x, y, width, height }; }"
            )
            assert clone_box["x"] >= 0 and clone_box["x"] + clone_box["width"] <= width
            assert clone_box["y"] >= 0 and clone_box["y"] + clone_box["height"] <= height
            page.keyboard.press("Escape")
            expect(clone_button).to_be_focused()
            _assert_live_announcement_is_visually_hidden(page)

            customize_button = page.get_by_role(
                "button", name="Customize benchmark dashboard"
            )
            customize_button.click()
            customize_dialog = page.get_by_role(
                "dialog", name="Customize benchmark dashboard"
            )
            customize_box = customize_dialog.evaluate(
                "element => { const { x, y, width, height } = element.getBoundingClientRect(); return { x, y, width, height }; }"
            )
            assert customize_box["x"] >= 0 and customize_box["x"] + customize_box["width"] <= width
            assert customize_box["y"] >= 0 and customize_box["y"] + customize_box["height"] <= height
            page.keyboard.press("Escape")
            expect(customize_button).to_be_focused()
            _assert_live_announcement_is_visually_hidden(page)
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            ), f"Page or dialog overflows horizontally at {width}x{height}"

        page.set_viewport_size({"width": 1440, "height": 900})
        widget_list = page.locator('section[aria-label="Benchmark metric widgets"]')
        widgets = widget_list.locator("article[role='listitem']")
        original_order = widgets.evaluate_all(
            "elements => elements.map(element => element.getAttribute('aria-label'))"
        )
        assert len(original_order) >= 2
        widgets.nth(0).focus()
        widgets.nth(0).press("Space")
        widgets.nth(0).press("ArrowDown")
        widgets.nth(0).press("Space")
        reordered = widgets.evaluate_all(
            "elements => elements.map(element => element.getAttribute('aria-label'))"
        )
        assert reordered != original_order
        page.reload()
        widgets = page.locator(
            'section[aria-label="Benchmark metric widgets"] article[role="listitem"]'
        )
        expect(widgets).to_have_count(len(reordered))
        assert widgets.evaluate_all(
            "elements => elements.map(element => element.getAttribute('aria-label'))"
        ) == reordered

        horizontal_bar = page.get_by_role(
            "button", name="Use Horizontal bar chart for Vocabulary size"
        )
        expect(horizontal_bar).to_be_visible()
        horizontal_bar.click()
        expect(horizontal_bar).to_have_attribute("aria-pressed", "true")
        page.reload()
        expect(
            page.get_by_role(
                "button", name="Use Horizontal bar chart for Vocabulary size"
            )
        ).to_have_attribute("aria-pressed", "true")

        customize = page.get_by_role(
            "button", name="Customize benchmark dashboard"
        )
        customize.click()
        customize_dialog = page.get_by_role(
            "dialog", name="Customize benchmark dashboard"
        )
        expect(customize_dialog).to_be_visible()
        customize_dialog.press("Escape")
        expect(customize_dialog).to_have_count(0)
        expect(customize).to_be_focused()
        customize.click()
        customize_dialog = page.get_by_role(
            "dialog", name="Customize benchmark dashboard"
        )
        vocabulary_widget_row = customize_dialog.locator(
            "label.metric-selection-row"
        ).filter(has_text="Vocabulary size")
        expect(vocabulary_widget_row).to_have_count(1)
        vocabulary_widget = vocabulary_widget_row.get_by_role("checkbox")
        vocabulary_widget.uncheck()
        customize_dialog.get_by_role("button", name="Apply").click()
        expect(
            page.locator('article[aria-label="Vocabulary size widget"]')
        ).to_have_count(0)
        page.reload()
        expect(
            page.locator('article[aria-label="Vocabulary size widget"]')
        ).to_have_count(0)
        page.get_by_role("button", name="Restore default layout").click()
        expect(
            page.get_by_role("button", name="Use Vertical bar chart for Vocabulary size")
        ).to_have_attribute("aria-pressed", "true")

        page.locator(".cross-benchmark-report-selector").click()
        manager = page.get_by_role("dialog", name="Benchmark Reports")
        search = manager.get_by_label("Search reports")
        search.fill(f"{stem} manager fixture 00")
        delete_row = manager.locator(".benchmark-report-row").filter(
            has_text=f"{stem} manager fixture 00"
        )
        expect(delete_row).to_have_count(1)
        delete_id = fixture_ids[0]
        delete_row.get_by_role("button", name="Delete report").click()
        expect(
            delete_row.get_by_text("Delete this report permanently?", exact=True)
        ).to_be_visible()
        delete_row.get_by_role("button", name="Cancel").click()
        expect(
            delete_row.get_by_text("Delete this report permanently?", exact=True)
        ).to_have_count(0)
        delete_row.get_by_role("button", name="Delete report").click()
        with page.expect_response(
            lambda response: response.request.method == "DELETE"
            and response.url.endswith(f"/reports/{delete_id}")
        ) as delete_response_info:
            delete_row.get_by_role("button", name="Delete", exact=True).click()
        assert delete_response_info.value.status == 204
        expect(delete_row).to_have_count(0)
        assert api_context.get(f"/api/benchmarks/reports/{delete_id}").status == 404
        manager.get_by_role("button", name="Close benchmark report manager").click()

        assert browser_errors == [], browser_errors
        assert browser_http_errors == [], browser_http_errors
    finally:
        for report_id in sorted(report_ids, reverse=True):
            delete_report = api_context.delete(
                f"/api/benchmarks/reports/{report_id}"
            )
            assert delete_report.status in {204, 404}, delete_report.text()
            assert api_context.get(f"/api/benchmarks/reports/{report_id}").status == 404

        for tokenizer_name in tokenizer_names:
            delete_tokenizer = api_context.delete(
                "/api/tokenizers/delete", params={"tokenizer_name": tokenizer_name}
            )
            assert delete_tokenizer.status in {200, 404}, delete_tokenizer.text()
        refreshed_tokenizers = api_context.get("/api/tokenizers/list")
        assert refreshed_tokenizers.ok, refreshed_tokenizers.text()
        remaining_tokenizers = {
            str(item.get("tokenizer_name"))
            for item in refreshed_tokenizers.json().get("tokenizers", [])
        }
        assert remaining_tokenizers.isdisjoint(tokenizer_names)

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

        leftovers = api_context.get(
            "/api/benchmarks/reports",
            params={"search": stem, "offset": 0, "limit": 100},
        )
        assert leftovers.ok, leftovers.text()
        assert leftovers.json().get("total") == 0
