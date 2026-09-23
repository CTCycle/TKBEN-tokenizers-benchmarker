"""Opt-in live coverage for cancelling and immediately rerunning a benchmark."""

import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from playwright.sync_api import APIRequestContext, Locator, Page, expect
from tokenizers import Tokenizer, models, pre_tokenizers, processors


RUN_BENCHMARKS = os.getenv("E2E_RUN_BENCHMARKS", "").lower() in {
    "1",
    "true",
    "yes",
}
REPO_ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN_OUTPUT_DIR = os.getenv("E2E_QA_OUTPUT_DIR")
QA_OUTPUT_DIR = (
    Path(CAMPAIGN_OUTPUT_DIR)
    if CAMPAIGN_OUTPUT_DIR
    else REPO_ROOT / "assets" / "QA"
)
SCREENSHOT_PATH = QA_OUTPUT_DIR / (
    "t5-04-completed-rerun.png"
    if CAMPAIGN_OUTPUT_DIR
    else "tkben-t2-07-benchmark-cancellation-20260923.png"
)
CANCELLATION_SCREENSHOT_PATH = QA_OUTPUT_DIR / (
    "t5-04-running-progress.png"
    if CAMPAIGN_OUTPUT_DIR
    else "tkben-t2-07-benchmark-cancellation-running-20260923.png"
)
RUN_OPTIONS_SCREENSHOT_PATH = QA_OUTPUT_DIR / "t3-02-run-options.png"
METRIC_KEYS = {
    "eff.encode_tokens_per_second_mean",
    "lat.encode_latency_distribution",
    "res.peak_rss_mb",
    "res.memory_delta_mb",
}
DOCUMENT_DISTRIBUTION_METRIC_KEY = "doc.tokens_count_distribution"


def _campaign_tokenizer_json() -> bytes:
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "[PAD]": 0,
                "[UNK]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "benchmark": 4,
                "cancellation": 5,
                "sample": 6,
                "row": 7,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return tokenizer.to_str().encode("utf-8")


def _select_metric(
    dialog: Locator,
    api_context: APIRequestContext,
    *,
    include_document_distribution: bool = False,
) -> None:
    catalog_response = api_context.get("/api/benchmarks/metrics/catalog")
    assert catalog_response.ok, catalog_response.text()
    catalog = catalog_response.json()
    metric_options = [
        metric
        for category in catalog.get("categories", [])
        for metric in category.get("metrics", [])
    ]
    selected_metric_keys = set(METRIC_KEYS)
    if include_document_distribution:
        selected_metric_keys.add(DOCUMENT_DISTRIBUTION_METRIC_KEY)
    metric_indexes = {
        index
        for index, metric in enumerate(metric_options)
        if metric.get("key") in selected_metric_keys
    }
    assert len(metric_indexes) == len(selected_metric_keys), (
        "A required metric is missing from the catalog"
    )

    checkboxes = dialog.locator(
        ".benchmark-wizard-tree-children input[type='checkbox']"
    )
    expect(checkboxes).to_have_count(len(metric_options))
    for index in range(len(metric_options)):
        checkbox = checkboxes.nth(index)
        if checkbox.is_checked() and index not in metric_indexes:
            checkbox.uncheck()
        elif index in metric_indexes and not checkbox.is_checked():
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
    non_default_options: bool = False,
    include_document_distribution: bool = False,
) -> tuple[str, dict[str, Any]]:
    page.get_by_role("button", name="Run benchmark").click()
    dialog = page.get_by_role("dialog", name="Run benchmark")
    expect(dialog).to_be_visible()

    _select_metric(
        dialog,
        api_context,
        include_document_distribution=include_document_distribution,
    )
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
    if non_default_options:
        dialog.get_by_label("Seed").fill("99")
        dialog.get_by_label("Parallelism").fill("2")
        dialog.get_by_label("Add special tokens").check()
        dialog.get_by_label("Enable padding").check()
        dialog.get_by_label("Enable truncation").check()
        dialog.get_by_label("Max length").fill("4")
        dialog.get_by_label("Store per-document stats").check()
        dialog.get_by_label("Per-document sample size").fill("2")
    if CAMPAIGN_OUTPUT_DIR:
        for label in (
            "Add special tokens",
            "Enable padding",
            "Enable truncation",
            "Max length",
            "Store per-document stats",
            "Per-document sample size",
        ):
            expect(dialog.get_by_text(label, exact=True)).to_be_visible()
        assert dialog.locator(
            ".benchmark-wizard-advanced-settings input[type='checkbox']"
        ).count() == 4
        QA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(RUN_OPTIONS_SCREENSHOT_PATH), full_page=True)
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
    progress_status: dict[str, Any] | None = None
    progress_text: str | None = None
    cancellation_status: dict[str, Any] | None = None
    cancelled_report_found = False
    cancellation_response_ms: float | None = None
    backend_rss_samples_mb: list[float] = []
    backend_rss_error: str | None = None
    rerun_report: dict[str, Any] | None = None

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
                    "buffer": _campaign_tokenizer_json(),
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
        progress_status = _wait_for_running_progress(
            api_context, first_job_id, minimum_progress=20.0
        )
        progress_indicator = page.locator(".benchmark-job-progress")
        expect(progress_indicator).to_be_visible()
        expect(progress_indicator).to_contain_text(
            f"{float(progress_status['progress']):g}%", timeout=30_000
        )
        progress_text = progress_indicator.inner_text()
        cancel_button = page.get_by_role("button", name="Cancel benchmark")
        expect(cancel_button).to_be_enabled()
        CANCELLATION_SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(CANCELLATION_SCREENSHOT_PATH), full_page=True)
        backend_pid = os.getenv("TKBEN_BACKEND_PID")
        if backend_pid:
            try:
                import psutil

                backend_process = psutil.Process(int(backend_pid))
                backend_command = " ".join(backend_process.cmdline())
                assert "uvicorn server.app:app" in backend_command, (
                    "Configured backend PID does not belong to the benchmark server"
                )
                for _ in range(8):
                    backend_rss_samples_mb.append(
                        float(backend_process.memory_info().rss / (1024 * 1024))
                    )
                    page.wait_for_timeout(100)
            except Exception as exc:
                backend_rss_error = f"{type(exc).__name__}: {exc}"
        else:
            backend_rss_error = "TKBEN_BACKEND_PID was not provided"
        cancel_started = time.perf_counter()
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.endswith(f"/api/jobs/{first_job_id}/cancel")
            ),
            timeout=30_000,
        ) as cancel_response_info:
            cancel_button.click()
        cancel_response = cancel_response_info.value
        cancellation_response_ms = (time.perf_counter() - cancel_started) * 1000
        assert cancel_response.ok, cancel_response.text()
        assert cancel_response.json().get("status") == "running"
        cancellation_status = job_waiter(
            first_job_id,
            poll_interval=0.1,
            timeout_seconds=300.0,
        )
        assert cancellation_status.get("status") == "cancelled", cancellation_status
        expect(page.locator(".benchmark-job-progress")).to_have_count(0, timeout=30_000)
        expect(run_button).to_be_enabled()

        cancelled_reports_response = api_context.get(
            "/api/benchmarks/reports",
            params={"search": cancelled_run_name},
        )
        assert cancelled_reports_response.ok, cancelled_reports_response.text()
        cancelled_reports = cancelled_reports_response.json().get("reports", [])
        cancelled_report_found = any(
            report.get("run_name") == cancelled_run_name for report in cancelled_reports
        )
        assert not cancelled_report_found, "A cancelled benchmark unexpectedly persisted a report"

        second_job_id, second_job = _start_benchmark(
            page=page,
            api_context=api_context,
            tokenizer_name=tokenizer_name,
            dataset_name=dataset_name,
            run_name=rerun_name,
            max_documents=2,
            timed_trials=1,
            non_default_options=True,
            include_document_distribution=True,
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
        rerun_report = report_response.json()
        assert int(rerun_report.get("report_id", 0)) == report_id
        assert rerun_report.get("config") == {
            "max_documents": 2,
            "warmup_trials": 0,
            "timed_trials": 1,
            "batch_size": 16,
            "seed": 99,
            "parallelism": 2,
            "add_special_tokens": True,
            "padding": True,
            "truncation": True,
            "max_length": 4,
            "store_per_document_stats": True,
            "per_document_sample_size": 2,
        }
        rerun_result = rerun_report.get("tokenizer_results", [{}])[0]
        assert isinstance(rerun_result.get("resources", {}).get("peak_rss_mb"), (int, float))
        assert isinstance(rerun_result.get("resources", {}).get("memory_delta_mb"), (int, float))
        assert DOCUMENT_DISTRIBUTION_METRIC_KEY in rerun_report.get(
            "selected_metric_keys", []
        )
        assert len(rerun_report.get("per_document_stats", [])) == 1
        assert len(rerun_report["per_document_stats"][0].get("tokens_count", [])) == 2
        assert all(
            int(row["token_count"]) <= int(row["documents"]) * 4
            for row in rerun_report.get("raw_observations", {}).get(tokenizer_name, [])
        )
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
        if CAMPAIGN_OUTPUT_DIR:
            import json

            QA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (QA_OUTPUT_DIR / "t5-04-cancellation.json").write_text(
                json.dumps(
                    {
                        "gate": "T5-04",
                        "workload_documents": 10_000,
                        "visible_progress": progress_status,
                        "progress_indicator_text": progress_text,
                        "cancelled_job_status": (
                            cancellation_status.get("status")
                            if cancellation_status
                            else None
                        ),
                        "cancelled_run_report_found": cancelled_report_found,
                        "cancel_click_response_ms": cancellation_response_ms,
                        "backend_rss_samples_mb": backend_rss_samples_mb,
                        "backend_rss_error": backend_rss_error,
                        "rerun_rendered": rerun_report is not None,
                        "rerun_report": rerun_report,
                        "browser_errors": browser_errors,
                        "browser_http_errors": browser_http_errors,
                        "screenshots": [
                            str(CANCELLATION_SCREENSHOT_PATH),
                            str(SCREENSHOT_PATH),
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
