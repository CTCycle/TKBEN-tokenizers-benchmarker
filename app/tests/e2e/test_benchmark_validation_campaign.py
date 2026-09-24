"""Opt-in live evidence for benchmark measurement and run configuration gates."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from playwright.sync_api import APIRequestContext
from tokenizers import Tokenizer, models, pre_tokenizers, processors


RUN_BENCHMARKS = os.getenv("E2E_RUN_BENCHMARKS", "").lower() in {"1", "true", "yes"}
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


def _total_observed_tokens(report: dict[str, Any], tokenizer_name: str) -> int:
    observations = report.get("raw_observations", {}).get(tokenizer_name, [])
    assert observations and all("token_count" in row for row in observations)
    return sum(int(row["token_count"]) for row in observations)


def _assert_config_shape(report: dict[str, Any], expected: dict[str, Any]) -> None:
    actual = report.get("config", {})
    assert set(actual) == CONFIG_FIELDS
    assert actual == expected


@pytest.mark.skipif(
    not RUN_BENCHMARKS,
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
                + (["doc.tokens_count_distribution"] if include_document_distribution else []),
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
            " ".join(words[(index + offset) % len(words)] for offset in range(1 + index % 6))
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
        assert configured_report.get("runtime_metadata", {}).get(
            "benchmark_config", {}
        ).get("parallelism") == 2

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
        parallel_tokenizer_name = str(
            parallel_tokenizer.get("tokenizer_name", "")
        )
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
            assert all(
                len(item.get("tokens_count", [])) == 23 for item in per_document
            )
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
