# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import os
import pytest
from playwright.sync_api import APIRequestContext
import json
from pathlib import Path
from typing import Any
from uuid import uuid4
from tokenizers import Tokenizer
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import processors
import time
from playwright.sync_api import Locator
from playwright.sync_api import Page
from playwright.sync_api import expect

###############################################################################
# E2E tests for benchmark API endpoints.


RUN_BENCHMARKS = os.getenv("E2E_RUN_BENCHMARKS", "").lower() in ("1", "true", "yes")


###############################################################################
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"tokenizers": [], "dataset_name": "custom/e2e_sample"},
            "At least one tokenizer",
        ),
        (
            {
                "tokenizers": ["hf-internal-testing/tiny-random-bert"],
                "dataset_name": "",
            },
            "Dataset name must be specified",
        ),
    ],
)
def test_run_benchmarks_rejects_missing_required_inputs(
    api_context: APIRequestContext,
    payload: dict,
    message: str,
) -> None:
    """POST /api/benchmarks/run should reject incomplete benchmark inputs."""
    response = api_context.post("/api/benchmarks/run", data=payload)
    assert response.status == 400
    data = response.json()
    assert message in data.get("detail", "")


###############################################################################
def test_run_benchmarks_missing_dataset_returns_400(
    api_context: APIRequestContext,
) -> None:
    """POST /api/benchmarks/run should reject unknown datasets before loading tokenizers."""
    response = api_context.post(
        "/api/benchmarks/run",
        data={
            "tokenizers": ["hf-internal-testing/tiny-random-bert"],
            "dataset_name": "missing_dataset",
        },
    )
    assert response.status == 400
    data = response.json()
    assert "not found or empty" in data.get("detail", "").lower()


###############################################################################
def test_get_benchmark_metrics_catalog_returns_categories(
    api_context: APIRequestContext,
) -> None:
    """GET /api/benchmarks/metrics/catalog should return a non-empty catalog."""
    response = api_context.get("/api/benchmarks/metrics/catalog")
    assert response.ok, response.text()
    data = response.json()
    categories = data.get("categories", [])
    assert isinstance(categories, list)
    assert len(categories) > 0
    first = categories[0]
    assert isinstance(first.get("category_key"), str)
    assert isinstance(first.get("metrics"), list)


###############################################################################
@pytest.mark.skipif(
    not RUN_BENCHMARKS,
    reason="Set E2E_RUN_BENCHMARKS=1 to enable benchmark execution.",
)
def test_benchmark_report_round_trip_includes_reproducibility_metadata(
    api_context: APIRequestContext,
    uploaded_dataset: dict,
    uploaded_tiny_tokenizer: dict,
    job_waiter,
) -> None:
    """A small benchmark should persist its current-schema report and metadata."""
    tokenizer_name = uploaded_tiny_tokenizer["tokenizer_name"]
    run_payload = {
        "tokenizers": [tokenizer_name],
        "dataset_name": uploaded_dataset["dataset_name"],
        "run_name": "e2e deterministic benchmark report",
        "selected_metric_keys": [
            "eff.encode_tokens_per_second_mean",
            "frag.tokens_per_character",
        ],
        "config": {
            "max_documents": 2,
            "warmup_trials": 1,
            "timed_trials": 1,
            "batch_size": 2,
            "seed": 42,
            "parallelism": 1,
            "store_per_document_stats": True,
            "per_document_sample_size": 2,
        },
    }
    response = api_context.post(
        "/api/benchmarks/run",
        data=run_payload,
    )
    assert response.ok, response.text()
    job = response.json()
    job_id = job.get("job_id")
    assert job_id, "Missing job_id in benchmark response"
    job_status = job_waiter(
        job_id,
        poll_interval=job.get("poll_interval", 1.0),
        timeout_seconds=1800.0,
    )
    assert job_status.get("status") == "completed", job_status.get("error")
    data = job_status.get("result", {})
    assert data.get("status") == "success"
    assert data.get("schema_version") == 3
    assert data.get("methodology_version") == "semantic_honesty"
    assert data.get("dataset_name") == uploaded_dataset["dataset_name"]
    assert data.get("tokenizers_count", 0) >= 1
    assert data.get("documents_processed") == 2
    dashboard = data.get("dashboard")
    assert isinstance(dashboard, dict)
    assert isinstance(dashboard.get("widgets"), list)
    assert data.get("report_id")
    report_id = int(data.get("report_id"))
    assert report_id > 0
    assert data.get("run_name") == run_payload["run_name"]
    assert data.get("selected_metric_keys") == run_payload["selected_metric_keys"]
    assert data.get("config", {}).get("timed_trials") == 1

    tokenizer_results = data.get("tokenizer_results", [])
    assert len(tokenizer_results) == 1
    tokenizer_result = tokenizer_results[0]
    assert tokenizer_result.get("tokenizer") == tokenizer_name
    assert tokenizer_result.get("status") == "success"
    assert (
        tokenizer_result.get("efficiency", {}).get("encode_tokens_per_second_mean", 0)
        > 0
    )
    assert not tokenizer_result.get("error_message")

    runtime_metadata = data.get("runtime_metadata")
    assert isinstance(runtime_metadata, dict)
    assert runtime_metadata.get("benchmark_config", {}).get("timed_trials") == 1
    assert runtime_metadata.get("dataset_documents_benchmarked") == 2
    assert isinstance(runtime_metadata.get("benchmark_timing_boundaries"), dict)
    assert isinstance(data.get("raw_observations"), dict)

    list_response = api_context.get("/api/benchmarks/reports")
    assert list_response.ok, list_response.text()
    report_list = list_response.json().get("reports", [])
    assert any(int(item.get("report_id", 0)) == report_id for item in report_list)

    by_id_response = api_context.get(f"/api/benchmarks/reports/{report_id}")
    assert by_id_response.ok, by_id_response.text()
    by_id_data = by_id_response.json()
    assert int(by_id_data.get("report_id", 0)) == report_id
    assert by_id_data.get("dataset_name") == uploaded_dataset["dataset_name"]
    assert by_id_data.get("tokenizers_count", 0) >= 1
    assert by_id_data.get("run_name") == data.get("run_name")
    assert by_id_data.get("documents_processed") == data.get("documents_processed")
    assert by_id_data.get("tokenizers_processed") == data.get("tokenizers_processed")
    assert by_id_data.get("config", {}).get("timed_trials") == 1
    assert by_id_data.get("runtime_metadata", {}).get(
        "benchmark_config", {}
    ) == runtime_metadata.get("benchmark_config")

    delete_response = api_context.delete(f"/api/benchmarks/reports/{report_id}")
    assert delete_response.status == 204, delete_response.text()
    assert api_context.get(f"/api/benchmarks/reports/{report_id}").status == 404
    remaining = api_context.get(
        f"/api/benchmarks/reports?search={run_payload['run_name'].replace(' ', '%20')}"
    )
    assert remaining.ok, remaining.text()
    assert all(
        int(item.get("report_id", 0)) != report_id
        for item in remaining.json().get("reports", [])
    )


###############################################################################
# Opt-in live evidence for benchmark measurement and run configuration gates.


RUN_BENCHMARKS__benchmark_validation_campaign = os.getenv(
    "E2E_RUN_BENCHMARKS", ""
).lower() in {"1", "true", "yes"}
REPO_ROOT = Path(__file__).resolve().parents[3]
QA_OUTPUT_DIR = Path(
    os.getenv(
        "E2E_QA_OUTPUT_DIR",
        str(REPO_ROOT / "assets" / "QA" / "tkben-benchmark-validation-campaign"),
    )
)
PERFORMANCE_METRICS = [
    "eff.encode_tokens_per_second_mean",
    "lat.encode_latency_distribution",
    "res.peak_rss_mb",
    "res.memory_delta_mb",
]
CONFIG_FIELDS = {
    "max_documents",
    "warmup_trials",
    "timed_trials",
    "batch_size",
    "seed",
    "parallelism",
    "add_special_tokens",
    "padding",
    "truncation",
    "max_length",
    "store_per_document_stats",
    "per_document_sample_size",
}


###############################################################################
def _tokenizer_json() -> bytes:
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "[PAD]": 0,
                "[UNK]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "validation": 4,
                "sample": 5,
                "repeat": 6,
                "dataset": 7,
                "document": 8,
                "benchmark": 9,
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


###############################################################################
def _config(
    *,
    documents: int = 60,
    warmup_trials: int = 0,
    timed_trials: int = 1,
    batch_size: int = 10,
    seed: int = 17,
    parallelism: int = 1,
    add_special_tokens: bool = False,
    padding: bool = False,
    truncation: bool = False,
    max_length: int | None = None,
    store_per_document_stats: bool = False,
    per_document_sample_size: int = 11,
) -> dict[str, Any]:
    return {
        "max_documents": documents,
        "warmup_trials": warmup_trials,
        "timed_trials": timed_trials,
        "batch_size": batch_size,
        "seed": seed,
        "parallelism": parallelism,
        "add_special_tokens": add_special_tokens,
        "padding": padding,
        "truncation": truncation,
        "max_length": max_length,
        "store_per_document_stats": store_per_document_stats,
        "per_document_sample_size": per_document_sample_size,
    }


###############################################################################
def _total_observed_tokens(report: dict[str, Any], tokenizer_name: str) -> int:
    observations = report.get("raw_observations", {}).get(tokenizer_name, [])
    assert observations and all("token_count" in row for row in observations)
    return sum(int(row["token_count"]) for row in observations)


###############################################################################
def _assert_config_shape(report: dict[str, Any], expected: dict[str, Any]) -> None:
    actual = report.get("config", {})
    assert set(actual) == CONFIG_FIELDS
    assert actual == expected


###############################################################################
@pytest.mark.skipif(
    not RUN_BENCHMARKS__benchmark_validation_campaign,
    reason="Set E2E_RUN_BENCHMARKS=1 to enable benchmark execution.",
)
def test_live_benchmark_measurements_and_run_options(
    api_context: APIRequestContext,
    job_waiter,
) -> None:
    """Collect repeated 1,000-document reports and verify supported run options."""
    stem = f"t3_campaign_{uuid4().hex[:8]}"
    dataset_name = f"custom/{stem}"
    tokenizer_stem = f"t3_campaign_{uuid4().hex[:8]}"
    tokenizer_name: str | None = None
    parallel_tokenizer_name: str | None = None
    dataset_created = False
    upload_job_id: str | None = None
    job_ids: list[str] = []
    job_terminal_statuses: dict[str, str] = {}
    tokenizer_names_to_delete: list[str] = []
    report_ids: list[int] = []
    evidence: dict[str, Any] = {
        "gate_scope": ["T3-01", "T3-02"],
        "dataset_name": dataset_name,
        "workload_documents": 1_000,
        "reports": [],
        "configuration_reports": [],
        "configuration_checks": {},
    }
    completed = False

    def run_report(
        run_name: str,
        config: dict[str, Any],
        *,
        tokenizers: list[str] | None = None,
        include_document_distribution: bool = False,
    ) -> dict[str, Any]:
        response = api_context.post(
            "/api/benchmarks/run",
            data={
                "tokenizers": tokenizers or [tokenizer_name],
                "dataset_name": dataset_name,
                "run_name": run_name,
                "selected_metric_keys": PERFORMANCE_METRICS
                + (
                    ["doc.tokens_count_distribution"]
                    if include_document_distribution
                    else []
                ),
                "config": config,
            },
        )
        assert response.ok, response.text()
        job = response.json()
        job_id = str(job.get("job_id", ""))
        assert job_id, "Benchmark response did not include a job ID"
        job_ids.append(job_id)
        status = job_waiter(
            job_id,
            poll_interval=job.get("poll_interval", 0.1),
            timeout_seconds=1800.0,
        )
        assert status.get("status") == "completed", status
        result = status.get("result", {})
        assert result.get("status") == "success", result
        report_id = int(result.get("report_id", 0))
        assert report_id > 0
        report_ids.append(report_id)
        persisted = api_context.get(f"/api/benchmarks/reports/{report_id}")
        assert persisted.ok, persisted.text()
        report = persisted.json()
        assert int(report.get("report_id", 0)) == report_id
        _assert_config_shape(report, config)
        evidence["configuration_reports"].append(report)
        return report

    try:
        words = ["validation", "sample", "repeat", "dataset", "document", "benchmark"]
        rows = [
            " ".join(
                words[(index + offset) % len(words)] for offset in range(1 + index % 6)
            )
            for index in range(1_000)
        ]
        dataset_response = api_context.post(
            "/api/datasets/upload",
            multipart={
                "file": {
                    "name": f"{stem}.csv",
                    "mimeType": "text/csv",
                    "buffer": f"text\n{'\n'.join(rows)}\n".encode("utf-8"),
                }
            },
        )
        assert dataset_response.ok, dataset_response.text()
        dataset_created = True
        upload_job_id = str(dataset_response.json().get("job_id", ""))
        assert upload_job_id
        job_ids.append(upload_job_id)
        upload_status = job_waiter(
            upload_job_id, poll_interval=0.1, timeout_seconds=300.0
        )
        assert upload_status.get("status") == "completed", upload_status
        assert upload_status.get("result", {}).get("dataset_name") == dataset_name
        assert upload_status.get("result", {}).get("document_count") == 1_000

        tokenizer_response = api_context.post(
            "/api/tokenizers/upload",
            multipart={
                "file": {
                    "name": f"{tokenizer_stem}.json",
                    "mimeType": "application/json",
                    "buffer": _tokenizer_json(),
                }
            },
        )
        assert tokenizer_response.ok, tokenizer_response.text()
        tokenizer = tokenizer_response.json()
        tokenizer_name = str(tokenizer.get("tokenizer_name", ""))
        assert tokenizer_name == f"CUSTOM_{tokenizer_stem}"
        assert tokenizer.get("is_compatible") is True
        tokenizer_names_to_delete.append(tokenizer_name)

        baseline_config = _config(
            documents=1_000,
            warmup_trials=2,
            timed_trials=8,
            batch_size=100,
            seed=42,
            parallelism=1,
            store_per_document_stats=False,
            per_document_sample_size=500,
        )
        baseline_reports = [
            run_report(f"T3-01 baseline {index + 1} {stem}", baseline_config)
            for index in range(3)
        ]
        for report in baseline_reports:
            assert report.get("documents_processed") == 1_000
            assert report.get("tokenizers_processed") == [tokenizer_name]
            result = report["tokenizer_results"][0]
            assert result.get("status") == "success"
            efficiency = result["efficiency"]
            mean_tps = efficiency.get("encode_tokens_per_second_mean")
            ci_low = efficiency.get("encode_tokens_per_second_ci95_low")
            ci_high = efficiency.get("encode_tokens_per_second_ci95_high")
            assert isinstance(mean_tps, (int, float)) and mean_tps > 0
            assert isinstance(ci_low, (int, float)) and ci_low <= mean_tps
            assert isinstance(ci_high, (int, float)) and ci_high >= mean_tps
            latency = result["latency"]
            p50, p95, p99 = (
                latency.get("encode_latency_p50_ms"),
                latency.get("encode_latency_p95_ms"),
                latency.get("encode_latency_p99_ms"),
            )
            assert isinstance(p50, (int, float)) and p50 >= 0
            assert isinstance(p95, (int, float)) and p50 <= p95
            assert isinstance(p99, (int, float)) and p95 <= p99
            observations = report["raw_observations"][tokenizer_name]
            assert latency.get("sample_count") == len(observations)
            assert len(observations) == 80
            assert all(
                isinstance(row.get("rss_before_mb"), (int, float))
                and isinstance(row.get("rss_after_mb"), (int, float))
                and isinstance(row.get("peak_rss_mb"), (int, float))
                for row in observations
            )
            resources = result["resources"]
            assert isinstance(resources.get("peak_rss_mb"), (int, float))
            assert isinstance(resources.get("memory_delta_mb"), (int, float))
            assert resources["peak_rss_mb"] > 0
            assert resources["memory_delta_mb"] >= 0
            for phase in (
                "encode_only_wall_time_seconds",
                "dataset_stream_wall_time_seconds",
                "postprocess_wall_time_seconds",
                "end_to_end_wall_time_seconds",
            ):
                assert isinstance(efficiency.get(phase), (int, float))
                assert efficiency[phase] >= 0
            hardware = report["hardware_profile"]
            assert hardware.get("runtime") and hardware.get("os")
            assert isinstance(hardware.get("cpu_logical_cores"), int)
            assert isinstance(hardware.get("memory_total_mb"), (int, float))
            evidence["reports"].append(report)

        evidence["runtime_metadata"] = baseline_reports[0].get("runtime_metadata", {})
        evidence["hardware_profile"] = baseline_reports[0].get("hardware_profile", {})
        evidence["baseline_comparison_config"] = baseline_config

        short_config = _config()
        plain_report = run_report(f"T3-02 plain options {stem}", short_config)
        plain_tokens = _total_observed_tokens(plain_report, tokenizer_name)

        special_config = _config(add_special_tokens=True)
        special_report = run_report(f"T3-02 special tokens {stem}", special_config)
        special_tokens = _total_observed_tokens(special_report, tokenizer_name)
        assert special_tokens == plain_tokens + (2 * short_config["max_documents"])

        padding_config = _config(padding=True)
        padding_report = run_report(f"T3-02 padding {stem}", padding_config)
        padding_tokens = _total_observed_tokens(padding_report, tokenizer_name)
        assert padding_tokens > plain_tokens

        truncation_config = _config(truncation=True, max_length=3)
        truncation_report = run_report(f"T3-02 truncation {stem}", truncation_config)
        truncation_tokens = _total_observed_tokens(truncation_report, tokenizer_name)
        assert truncation_tokens < plain_tokens
        assert all(
            int(row["token_count"]) <= int(row["documents"]) * 3
            for row in truncation_report["raw_observations"][tokenizer_name]
        )

        non_default_config = _config(
            warmup_trials=1,
            timed_trials=2,
            batch_size=7,
            seed=99,
            parallelism=2,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=4,
            store_per_document_stats=True,
            per_document_sample_size=23,
        )
        configured_report = run_report(
            f"T3-02 all options {stem}",
            non_default_config,
            include_document_distribution=True,
        )
        assert _total_observed_tokens(configured_report, tokenizer_name) == 60 * 2 * 4
        per_document = configured_report.get("per_document_stats", [])
        assert len(per_document) == 1
        assert len(per_document[0].get("tokens_count", [])) == 23
        assert (
            configured_report.get("runtime_metadata", {})
            .get("benchmark_config", {})
            .get("parallelism")
            == 2
        )

        parallel_tokenizer_stem = f"t3_parallel_{uuid4().hex[:8]}"
        parallel_tokenizer_response = api_context.post(
            "/api/tokenizers/upload",
            multipart={
                "file": {
                    "name": f"{parallel_tokenizer_stem}.json",
                    "mimeType": "application/json",
                    "buffer": _tokenizer_json(),
                }
            },
        )
        assert parallel_tokenizer_response.ok, parallel_tokenizer_response.text()
        parallel_tokenizer = parallel_tokenizer_response.json()
        parallel_tokenizer_name = str(parallel_tokenizer.get("tokenizer_name", ""))
        assert parallel_tokenizer_name == f"CUSTOM_{parallel_tokenizer_stem}"
        assert parallel_tokenizer.get("is_compatible") is True
        tokenizer_names_to_delete.append(parallel_tokenizer_name)

        parallel_tokenizer_names = [tokenizer_name, parallel_tokenizer_name]
        parallel_workload = _config(
            documents=1_000,
            warmup_trials=1,
            timed_trials=2,
            batch_size=100,
            seed=99,
            parallelism=1,
            store_per_document_stats=True,
            per_document_sample_size=23,
        )
        serial_parallelism_report = run_report(
            f"T3-02 parallelism one {stem}",
            parallel_workload,
            tokenizers=parallel_tokenizer_names,
            include_document_distribution=True,
        )
        serial_execution = serial_parallelism_report.get("runtime_metadata", {}).get(
            "benchmark_execution", {}
        )
        assert serial_execution.get("requested_parallelism") == 1
        assert serial_execution.get("effective_parallelism") == 1
        assert serial_execution.get("tokenizer_count") == 2
        assert serial_execution.get("max_concurrent_workers_observed") == 1

        parallel_workload = {**parallel_workload, "parallelism": 2}
        parallelism_report = run_report(
            f"T3-02 parallelism two {stem}",
            parallel_workload,
            tokenizers=parallel_tokenizer_names,
            include_document_distribution=True,
        )
        parallel_execution = parallelism_report.get("runtime_metadata", {}).get(
            "benchmark_execution", {}
        )
        assert parallel_execution.get("requested_parallelism") == 2
        assert parallel_execution.get("effective_parallelism") == 2
        assert parallel_execution.get("tokenizer_count") == 2
        assert parallel_execution.get("max_concurrent_workers_observed") == 2

        for report in (serial_parallelism_report, parallelism_report):
            assert report.get("tokenizers_processed") == parallel_tokenizer_names
            assert [
                item.get("tokenizer") for item in report.get("tokenizer_results", [])
            ] == parallel_tokenizer_names
            assert all(
                item.get("status") == "success"
                for item in report.get("tokenizer_results", [])
            )
            observations = report.get("raw_observations", {})
            assert list(observations) == parallel_tokenizer_names
            assert all(
                len(observations[name]) == 20
                and all("token_count" in row for row in observations[name])
                for name in parallel_tokenizer_names
            )
            per_document = report.get("per_document_stats", [])
            assert [item.get("tokenizer") for item in per_document] == (
                parallel_tokenizer_names
            )
            assert all(len(item.get("tokens_count", [])) == 23 for item in per_document)
            assert report.get("dashboard", {}).get("widgets")

        parallelism_evidence = {
            "tokenizers": parallel_tokenizer_names,
            "parallelism_1": {
                "persisted_config": serial_parallelism_report.get("config", {}),
                "execution_metadata": serial_execution,
                "tokenizer_result_order": [
                    item.get("tokenizer")
                    for item in serial_parallelism_report["tokenizer_results"]
                ],
                "raw_observation_counts": {
                    name: len(serial_parallelism_report["raw_observations"][name])
                    for name in parallel_tokenizer_names
                },
                "per_document_order": [
                    item.get("tokenizer")
                    for item in serial_parallelism_report["per_document_stats"]
                ],
                "report_loaded": True,
            },
            "parallelism_2": {
                "persisted_config": parallelism_report.get("config", {}),
                "execution_metadata": parallel_execution,
                "tokenizer_result_order": [
                    item.get("tokenizer")
                    for item in parallelism_report["tokenizer_results"]
                ],
                "raw_observation_counts": {
                    name: len(parallelism_report["raw_observations"][name])
                    for name in parallel_tokenizer_names
                },
                "per_document_order": [
                    item.get("tokenizer")
                    for item in parallelism_report["per_document_stats"]
                ],
                "report_loaded": True,
            },
            "concurrency_evidence": (
                "runtime worker high-water counter observed both tokenizer "
                "workloads active concurrently; no speedup assumption used"
            ),
        }

        evidence["configuration_checks"] = {
            "plain_total_tokens": plain_tokens,
            "special_tokens_total": special_tokens,
            "padding_total_tokens": padding_tokens,
            "truncated_total_tokens": truncation_tokens,
            "all_options_config": non_default_config,
            "all_options_total_tokens": _total_observed_tokens(
                configured_report, tokenizer_name
            ),
            "per_document_statistics_count": len(
                per_document[0].get("tokens_count", [])
            ),
            "parallelism_execution": parallelism_evidence,
        }
        completed = True
    finally:
        for job_id in job_ids:
            job_status_response = api_context.get(f"/api/jobs/{job_id}")
            assert job_status_response.ok, job_status_response.text()
            job_status = job_status_response.json().get("status")
            if job_status in {"pending", "running"}:
                api_context.post(f"/api/jobs/{job_id}/cancel", data={})
                terminal_status = job_waiter(
                    job_id, poll_interval=0.1, timeout_seconds=300.0
                )
                job_status = terminal_status.get("status")
            assert job_status in {"completed", "failed", "cancelled"}
            job_terminal_statuses[job_id] = str(job_status)
        for report_id in report_ids:
            delete_report = api_context.delete(f"/api/benchmarks/reports/{report_id}")
            assert delete_report.status in {204, 404}, delete_report.text()
            assert api_context.get(f"/api/benchmarks/reports/{report_id}").status == 404
        for name in tokenizer_names_to_delete:
            delete_tokenizer = api_context.delete(
                "/api/tokenizers/delete", params={"tokenizer_name": name}
            )
            assert delete_tokenizer.status in {200, 404}, delete_tokenizer.text()
        if dataset_created:
            delete_dataset = api_context.delete(
                "/api/datasets/delete", params={"dataset_name": dataset_name}
            )
            assert delete_dataset.status in {200, 404}, delete_dataset.text()
        evidence["cleanup"] = {
            "terminal_job_statuses": job_terminal_statuses,
            "report_count_deleted": len(report_ids),
            "tokenizer_names_deleted": tokenizer_names_to_delete,
            "dataset_delete_requested": dataset_created,
        }
        evidence["outcome"] = "passed" if completed else "failed"
        QA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (QA_OUTPUT_DIR / "t3-benchmark-measurements-and-options.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8"
        )


###############################################################################
# Opt-in live coverage for cancelling and immediately rerunning a benchmark.


RUN_BENCHMARKS__benchmark_cancellation = os.getenv(
    "E2E_RUN_BENCHMARKS", ""
).lower() in {
    "1",
    "true",
    "yes",
}
METRIC_KEYS = {
    "eff.encode_tokens_per_second_mean",
    "lat.encode_latency_distribution",
    "res.peak_rss_mb",
    "res.memory_delta_mb",
}
DOCUMENT_DISTRIBUTION_METRIC_KEY = "doc.tokens_count_distribution"


###############################################################################
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


###############################################################################
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


###############################################################################
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


###############################################################################
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
    if os.getenv("E2E_QA_OUTPUT_DIR"):
        for label in (
            "Add special tokens",
            "Enable padding",
            "Enable truncation",
            "Max length",
            "Store per-document stats",
            "Per-document sample size",
        ):
            expect(dialog.get_by_text(label, exact=True)).to_be_visible()
        assert (
            dialog.locator(
                ".benchmark-wizard-advanced-settings input[type='checkbox']"
            ).count()
            == 4
        )
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


###############################################################################
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


###############################################################################
@pytest.mark.skipif(
    not RUN_BENCHMARKS__benchmark_cancellation,
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
        assert not cancelled_report_found, (
            "A cancelled benchmark unexpectedly persisted a report"
        )

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
        assert isinstance(
            rerun_result.get("resources", {}).get("peak_rss_mb"), (int, float)
        )
        assert isinstance(
            rerun_result.get("resources", {}).get("memory_delta_mb"), (int, float)
        )
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
        if os.getenv("E2E_QA_OUTPUT_DIR"):
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
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
