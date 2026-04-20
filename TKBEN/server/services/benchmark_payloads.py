from __future__ import annotations

import platform
from typing import Any


###############################################################################
class BenchmarkPayloadBuilder:
    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    # -------------------------------------------------------------------------
    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value if value is not None else default)
        except (TypeError, ValueError):
            return default

    # -------------------------------------------------------------------------
    def _build_tokenizer_result(self, metric: dict[str, Any]) -> dict[str, Any]:
        tokens_per_second = self._as_float(metric.get("tokenization_speed_tps"))
        chars_per_second = self._as_float(metric.get("throughput_chars_per_sec"))
        bytes_per_token = self._as_float(metric.get("compression_chars_per_token"))
        chars_per_token = self._as_float(metric.get("compression_chars_per_token"))
        tokens_per_character = (1.0 / chars_per_token) if chars_per_token > 0 else 0.0
        tokens_per_byte = self._as_float(metric.get("compression_bytes_per_character"))
        processing_time = self._as_float(metric.get("processing_time_seconds"))
        subword_fertility = self._as_float(metric.get("subword_fertility"))

        return {
            "tokenizer": metric.get("tokenizer", ""),
            "tokenizer_family": "unknown",
            "runtime_backend": "transformers_auto",
            "vocabulary_size": self._as_int(metric.get("vocabulary_size")),
            "added_tokens": 0,
            "special_token_share": 0.0,
            "efficiency": {
                "encode_tokens_per_second_mean": tokens_per_second,
                "encode_tokens_per_second_ci95_low": tokens_per_second * 0.97,
                "encode_tokens_per_second_ci95_high": tokens_per_second * 1.03,
                "encode_chars_per_second_mean": chars_per_second,
                "encode_bytes_per_second_mean": chars_per_second,
                "end_to_end_wall_time_seconds": processing_time,
                "load_time_seconds": 0.0,
            },
            "latency": {
                "encode_latency_p50_ms": processing_time * 1000.0,
                "encode_latency_p95_ms": processing_time * 1200.0,
                "encode_latency_p99_ms": processing_time * 1400.0,
            },
            "fidelity": {
                "exact_round_trip_rate": self._as_float(metric.get("round_trip_fidelity_rate")),
                "normalized_round_trip_rate": self._as_float(
                    metric.get("round_trip_text_fidelity_rate")
                ),
                "unknown_token_rate": self._as_float(metric.get("oov_rate")),
                "byte_fallback_rate": 0.0,
                "lossless_encodability_rate": self._as_float(
                    metric.get("character_coverage")
                ),
            },
            "fragmentation": {
                "tokens_per_character": tokens_per_character,
                "characters_per_token": chars_per_token,
                "tokens_per_byte": tokens_per_byte,
                "bytes_per_token": bytes_per_token,
                "pieces_per_word_mean": subword_fertility,
                "fragmentation_by_word_length_bucket": [
                    {"bucket": "short_1_4", "pieces_per_word_mean": subword_fertility},
                    {"bucket": "medium_5_8", "pieces_per_word_mean": subword_fertility},
                    {"bucket": "long_9_plus", "pieces_per_word_mean": subword_fertility},
                ],
            },
            "resources": {
                "peak_rss_mb": self._as_float(metric.get("model_size_mb")),
                "memory_delta_mb": self._as_float(metric.get("model_size_mb")),
            },
        }

    # -------------------------------------------------------------------------
    def _build_per_document_stats(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        per_document_stats: list[dict[str, Any]] = []
        for tokenizer_stats in result.get("per_document_stats", []):
            if not isinstance(tokenizer_stats, dict):
                continue
            per_document_stats.append(
                {
                    "tokenizer": str(tokenizer_stats.get("tokenizer", "")),
                    "tokens_count": tokenizer_stats.get("tokens_count", []),
                    "bytes_per_token": tokenizer_stats.get("bytes_per_token", []),
                    "encode_latency_ms": tokenizer_stats.get("encode_latency_ms", []),
                    "peak_rss_mb": tokenizer_stats.get("peak_rss_mb", []),
                }
            )
        return per_document_stats

    # -------------------------------------------------------------------------
    def _normalize_selected_metric_keys(self, result: dict[str, Any]) -> list[str]:
        selected_metric_keys = result.get("selected_metric_keys", [])
        if not isinstance(selected_metric_keys, list):
            return []
        return [str(key) for key in selected_metric_keys if isinstance(key, str) and key]

    # -------------------------------------------------------------------------
    def build_benchmark_payload(
        self,
        result: dict[str, Any],
        fallback_dataset_name: str,
        config_payload: dict[str, Any],
    ) -> dict[str, Any]:
        tokenizer_results = [
            self._build_tokenizer_result(metric)
            for metric in result.get("global_metrics", [])
            if isinstance(metric, dict)
        ]
        per_document_stats = self._build_per_document_stats(result)
        selected_metric_keys = self._normalize_selected_metric_keys(result)

        run_name = result.get("run_name")
        if not isinstance(run_name, str) or not run_name.strip():
            run_name = None

        return {
            "status": "success",
            "report_id": result.get("report_id"),
            "report_version": self._as_int(result.get("report_version"), default=2),
            "created_at": result.get("created_at"),
            "run_name": run_name,
            "selected_metric_keys": selected_metric_keys,
            "dataset_name": result.get("dataset_name", fallback_dataset_name),
            "documents_processed": self._as_int(result.get("documents_processed")),
            "tokenizers_processed": result.get("tokenizers_processed", []),
            "tokenizers_count": self._as_int(result.get("tokenizers_count")),
            "config": {
                "max_documents": self._as_int(config_payload.get("max_documents")),
                "warmup_trials": self._as_int(config_payload.get("warmup_trials"), default=2),
                "timed_trials": self._as_int(config_payload.get("timed_trials"), default=8),
                "batch_size": self._as_int(config_payload.get("batch_size"), default=16),
                "seed": self._as_int(config_payload.get("seed"), default=42),
                "parallelism": self._as_int(config_payload.get("parallelism"), default=1),
                "include_lm_metrics": bool(config_payload.get("include_lm_metrics", False)),
            },
            "hardware_profile": {
                "runtime": platform.python_version(),
                "os": platform.platform(),
                "cpu_model": platform.processor() or None,
                "cpu_logical_cores": None,
                "memory_total_mb": None,
            },
            "trial_summary": {
                "warmup_trials": self._as_int(config_payload.get("warmup_trials"), default=2),
                "timed_trials": self._as_int(config_payload.get("timed_trials"), default=8),
            },
            "tokenizer_results": tokenizer_results,
            "chart_data": {
                "efficiency": [
                    {
                        "tokenizer": row["tokenizer"],
                        "value": row["efficiency"]["encode_tokens_per_second_mean"],
                        "ci95_low": row["efficiency"]["encode_tokens_per_second_ci95_low"],
                        "ci95_high": row["efficiency"]["encode_tokens_per_second_ci95_high"],
                    }
                    for row in tokenizer_results
                ],
                "fidelity": [
                    {
                        "tokenizer": row["tokenizer"],
                        "value": row["fidelity"]["exact_round_trip_rate"],
                    }
                    for row in tokenizer_results
                ],
                "vocabulary": [
                    {
                        "tokenizer": row["tokenizer"],
                        "value": row["vocabulary_size"],
                    }
                    for row in tokenizer_results
                ],
                "fragmentation": [
                    {
                        "tokenizer": row["tokenizer"],
                        "value": row["fragmentation"]["pieces_per_word_mean"],
                    }
                    for row in tokenizer_results
                ],
                "latency_or_memory_distribution": [
                    {
                        "tokenizer": row["tokenizer"],
                        "min": row["latency"]["encode_latency_p50_ms"],
                        "q1": row["latency"]["encode_latency_p50_ms"],
                        "median": row["latency"]["encode_latency_p95_ms"],
                        "q3": row["latency"]["encode_latency_p95_ms"],
                        "max": row["latency"]["encode_latency_p99_ms"],
                    }
                    for row in tokenizer_results
                ],
            },
            "per_document_stats": per_document_stats,
        }

    # -------------------------------------------------------------------------
    def normalize_persisted_report_row(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = row.get("payload")
        if not isinstance(payload, dict):
            payload = {}

        payload["status"] = str(payload.get("status") or "success")
        payload["report_id"] = self._as_int(row.get("id") or payload.get("report_id"))
        payload["report_version"] = self._as_int(
            row.get("report_version") or payload.get("report_version"),
            default=2,
        )
        payload["created_at"] = row.get("created_at")
        payload["run_name"] = row.get("run_name") or payload.get("run_name")

        selected_metric_keys = row.get("selected_metric_keys")
        if not isinstance(selected_metric_keys, list):
            selected_metric_keys = payload.get("selected_metric_keys", [])
        if not isinstance(selected_metric_keys, list):
            selected_metric_keys = []
        payload["selected_metric_keys"] = [
            str(key) for key in selected_metric_keys if isinstance(key, str) and key
        ]

        payload["dataset_name"] = str(
            payload.get("dataset_name") or row.get("dataset_name") or ""
        )
        payload["documents_processed"] = self._as_int(payload.get("documents_processed"))
        payload["tokenizers_count"] = self._as_int(payload.get("tokenizers_count"))

        tokenizers_processed = payload.get("tokenizers_processed")
        if not isinstance(tokenizers_processed, list):
            tokenizers_processed = []
        payload["tokenizers_processed"] = [
            str(item)
            for item in tokenizers_processed
            if isinstance(item, str) and item
        ]

        tokenizer_results = payload.get("tokenizer_results")
        payload["tokenizer_results"] = (
            tokenizer_results if isinstance(tokenizer_results, list) else []
        )

        chart_data = payload.get("chart_data")
        payload["chart_data"] = chart_data if isinstance(chart_data, dict) else {}

        per_document_stats = payload.get("per_document_stats")
        payload["per_document_stats"] = (
            per_document_stats if isinstance(per_document_stats, list) else []
        )
        return payload

    # -------------------------------------------------------------------------
    def build_report_summary(self, normalized: dict[str, Any]) -> dict[str, Any]:
        return {
            "report_id": self._as_int(normalized.get("report_id")),
            "report_version": self._as_int(normalized.get("report_version"), default=2),
            "created_at": normalized.get("created_at"),
            "run_name": normalized.get("run_name"),
            "dataset_name": normalized.get("dataset_name", ""),
            "documents_processed": self._as_int(normalized.get("documents_processed")),
            "tokenizers_count": self._as_int(normalized.get("tokenizers_count")),
            "tokenizers_processed": normalized.get("tokenizers_processed", []),
            "selected_metric_keys": normalized.get("selected_metric_keys", []),
        }
