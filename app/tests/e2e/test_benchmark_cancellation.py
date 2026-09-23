"""Opt-in live coverage for cancelling and immediately rerunning a benchmark."""

import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from playwright.sync_api import APIRequestContext, Locator, Page, expect


RUN_BENCHMARKS = os.getenv("E2E_RUN_BENCHMARKS", "").lower() in {
    "1",
    "true",
    "yes",
}
REPO_ROOT = Path(__file__).resolve().parents[3]
SCREENSHOT_PATH = (
    REPO_ROOT / "assets" / "QA" / "tkben-t2-07-benchmark-cancellation-20260923.png"
)
CANCELLATION_SCREENSHOT_PATH = (
    REPO_ROOT
    / "assets"
    / "QA"
    / "tkben-t2-07-benchmark-cancellation-running-20260923.png"
)
METRIC_KEY = "eff.encode_tokens_per_second_mean"


def _select_metric(dialog: Locator, api_context: APIRequestContext) -> None:
    catalog_response = api_context.get("/api/benchmarks/metrics/catalog")
    assert catalog_response.ok, catalog_response.text()
    catalog = catalog_response.json()
    metric_options = [
        metric
        for category in catalog.get("categories", [])
        for metric in category.get("metrics", [])
    ]
    metric_index = next(
        (
            index
            for index, metric in enumerate(metric_options)
            if metric.get("key") == METRIC_KEY
        ),
        None,
    )
    assert metric_index is not None, f"Metric {METRIC_KEY} is missing from the catalog"

    checkboxes = dialog.locator(
        ".benchmark-wizard-tree-children input[type='checkbox']"
    )
    expect(checkboxes).to_have_count(len(metric_options))
    for index in range(len(metric_options)):
        checkbox = checkboxes.nth(index)
        if checkbox.is_checked() and index != metric_index:
            checkbox.uncheck()
        elif index == metric_index and not checkbox.is_checked():
            checkbox.check()
    dialog.get_by_role("button", name="Next").click()


def _set_document_limit(dialog: Locator, value: int) -> None:
    slider = dialog.locator("#benchmark-documents")
    slider.evaluate(
        """(input, nextValue) => {
          const setter = Object.getOwnPropertyDescriptor(
            HTMLInputElement.prototype,
            'value',
          )?.set;
          if (!setter) throw new Error('HTMLInputElement value setter is unavailable');
          setter.call(input, String(nextValue));
          input.dispatchEvent(new Event('input', { bubbles: true }));
          input.dispatchEvent(new Event('change', { bubbles: true }));
        }""",
        value,
    )
    expect(slider).to_have_value(str(value))


def _start_benchmark(
    *,
    page: Page,
    api_context: APIRequestContext,
    tokenizer_name: str,
    dataset_name: str,
    run_name: str,
    max_documents: int,
    timed_trials: int,
) -> tuple[str, dict[str, Any]]:
    page.get_by_role("button", name="Run benchmark").click()
    dialog = page.get_by_role("dialog", name="Run benchmark")
    expect(dialog).to_be_visible()

    _select_metric(dialog, api_context)
    tokenizer_option = dialog.get_by_text(tokenizer_name, exact=True).locator(
        "xpath=ancestor::label[1]"
    )
    tokenizer_option.locator("input[type='checkbox']").check()
    dialog.locator("#benchmark-dataset").select_option(value=dataset_name)
    _set_document_limit(dialog, max_documents)
    dialog.get_by_role("button", name="Next").click()

    dialog.get_by_label("Run Name").fill(run_name)
    dialog.get_by_label("Warmup trials").fill("0")
    dialog.get_by_label("Timed trials").fill(str(timed_trials))
    dialog.get_by_label("Batch size").fill("16")
    with page.expect_response(
        lambda response: (
            response.request.method == "POST"
            and response.url.endswith("/api/benchmarks/run")
        ),
        timeout=30_000,
    ) as response_info:
        dialog.get_by_role("button", name="Start benchmark").click()
    response = response_info.value
    assert response.ok, response.text()
    payload = response.json()
    job_id = str(payload.get("job_id", ""))
    assert job_id, "Missing benchmark job ID"
    return job_id, payload


def _wait_for_running_progress(
    api_context: APIRequestContext,
    job_id: str,
    minimum_progress: float,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = api_context.get(f"/api/jobs/{job_id}")
        assert response.ok, f"Failed to poll job {job_id}: {response.status}"
        status = response.json()
        assert status.get("status") not in {"completed", "failed", "cancelled"}, status
        if (
            status.get("status") == "running"
            and float(status.get("progress", 0.0)) >= minimum_progress
        ):
            return status
        time.sleep(0.1)
    raise AssertionError(
        f"Job {job_id} did not remain active at {minimum_progress}% progress"
    )


@pytest.mark.skipif(
    not RUN_BENCHMARKS,
    reason="Set E2E_RUN_BENCHMARKS=1 to enable local benchmark execution.",
)
def test_benchmark_can_be_cancelled_and_immediately_rerun(
    api_context: APIRequestContext,
    base_url: str,
    job_waiter,
    page: Page,
    tiny_tokenizer_json: bytes,
) -> None:
    """A cancelled local run saves no report and does not block a later run."""
    stem = f"t2_07_{uuid4().hex[:8]}"
    tokenizer_stem = uuid4().hex[:5]
    dataset_name = f"custom/{stem}"
    cancelled_run_name = f"T2-07 cancelled {stem}"
    rerun_name = f"T2-07 rerun {stem}"
    tokenizer_name: str | None = None
    report_id: int | None = None
    first_job_id: str | None = None
    second_job_id: str | None = None
    dataset_created = False
    browser_errors: list[str] = []
    browser_http_errors: list[tuple[int, str]] = []

    try:
        rows = "\n".join(
            f"benchmark cancellation sample row {index}" for index in range(10_000)
        )
        dataset_response = api_context.post(
            "/api/datasets/upload",
            multipart={
                "file": {
                    "name": f"{stem}.csv",
                    "mimeType": "text/csv",
                    "buffer": f"text\n{rows}\n".encode("utf-8"),
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
            poll_interval=0.1,
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
        run_button = page.get_by_role("button", name="Run benchmark")
        expect(run_button).to_be_enabled(timeout=30_000)

        first_job_id, _ = _start_benchmark(
            page=page,
            api_context=api_context,
            tokenizer_name=tokenizer_name,
            dataset_name=dataset_name,
            run_name=cancelled_run_name,
            max_documents=10_000,
            timed_trials=200,
        )
        _wait_for_running_progress(api_context, first_job_id, minimum_progress=20.0)
        cancel_button = page.get_by_role("button", name="Cancel benchmark")
        expect(cancel_button).to_be_enabled()
        CANCELLATION_SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(CANCELLATION_SCREENSHOT_PATH), full_page=True)
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.endswith(f"/api/jobs/{first_job_id}/cancel")
            ),
            timeout=30_000,
        ) as cancel_response_info:
            cancel_button.click()
        cancel_response = cancel_response_info.value
        assert cancel_response.ok, cancel_response.text()
        assert cancel_response.json().get("status") == "running"
        cancelled_status = job_waiter(
            first_job_id,
            poll_interval=0.1,
            timeout_seconds=300.0,
        )
        assert cancelled_status.get("status") == "cancelled", cancelled_status
        expect(page.locator(".benchmark-job-progress")).to_have_count(0, timeout=30_000)
        expect(run_button).to_be_enabled()

        cancelled_reports_response = api_context.get(
            "/api/benchmarks/reports",
            params={"search": cancelled_run_name},
        )
        assert cancelled_reports_response.ok, cancelled_reports_response.text()
        cancelled_reports = cancelled_reports_response.json().get("reports", [])
        assert not any(
            report.get("run_name") == cancelled_run_name for report in cancelled_reports
        ), "A cancelled benchmark unexpectedly persisted a report"

        second_job_id, second_job = _start_benchmark(
            page=page,
            api_context=api_context,
            tokenizer_name=tokenizer_name,
            dataset_name=dataset_name,
            run_name=rerun_name,
            max_documents=2,
            timed_trials=1,
        )
        rerun_status = job_waiter(
            second_job_id,
            poll_interval=second_job.get("poll_interval", 0.1),
            timeout_seconds=300.0,
        )
        assert rerun_status.get("status") == "completed", rerun_status.get("error")
        rerun = rerun_status.get("result", {})
        assert rerun.get("status") == "success"
        assert rerun.get("run_name") == rerun_name
        assert rerun.get("dataset_name") == dataset_name
        assert rerun.get("documents_processed") == 2
        report_id = int(rerun["report_id"])

        report_response = api_context.get(f"/api/benchmarks/reports/{report_id}")
        assert report_response.ok, report_response.text()
        assert int(report_response.json().get("report_id", 0)) == report_id
        current_report = page.locator('section[aria-label="Current report"]')
        expect(current_report.get_by_text(rerun_name, exact=True)).to_be_visible(
            timeout=300_000
        )
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
        for job_id in (first_job_id, second_job_id):
            if not job_id:
                continue
            status_response = api_context.get(f"/api/jobs/{job_id}")
            if status_response.ok and status_response.json().get("status") in {
                "pending",
                "running",
            }:
                api_context.post(f"/api/jobs/{job_id}/cancel", data={})
                job_waiter(job_id, poll_interval=0.1, timeout_seconds=300.0)
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
