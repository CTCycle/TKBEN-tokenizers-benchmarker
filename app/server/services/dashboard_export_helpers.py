from __future__ import annotations

import json
from typing import Any


###############################################################################
class DashboardExportFormatting:
    # -------------------------------------------------------------------------
    def _parse_zipf_curve(self, value: Any) -> list[dict[str, float]]:
        parsed = self._parse_json_like(value)
        if not isinstance(parsed, list):
            return []
        points: list[dict[str, float]] = []
        for index, item in enumerate(parsed):
            if isinstance(item, list) and len(item) >= 2:
                rank = self._to_number(item[0], index + 1)
                freq = self._to_number(item[1], 0.0)
            elif isinstance(item, dict):
                rank = self._to_number(item.get("rank"), index + 1)
                freq = self._to_number(item.get("frequency") or item.get("count"), 0.0)
            else:
                continue
            if rank > 0 and freq > 0:
                points.append({"rank": rank, "frequency": freq})
        points.sort(key=lambda item: item["rank"])
        return points[:300]

    # -------------------------------------------------------------------------
    def _parse_word_frequency(self, value: Any) -> list[dict[str, int]]:
        if not isinstance(value, list):
            return []
        rows: list[dict[str, int]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            word = str(item.get("word") or item.get("token") or "").strip()
            if not word:
                continue
            count = int(round(max(0.0, self._to_number(item.get("count"), 0.0))))
            if count <= 0:
                continue
            rows.append({"word": word, "count": count})
        rows.sort(key=lambda item: (-item["count"], item["word"]))
        return rows

    # -------------------------------------------------------------------------
    def _parse_vocabulary_items(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        rows: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "token_id": int(round(self._to_number(item.get("token_id"), 0))),
                    "token": str(item.get("token") or ""),
                    "length": int(round(self._to_number(item.get("length"), 0))),
                }
            )
        return rows

    # -------------------------------------------------------------------------
    def _select_distribution(
        self, distributions: Any, selected_tokenizer: str
    ) -> dict[str, Any] | None:
        if not isinstance(distributions, list) or not distributions:
            return None
        if selected_tokenizer:
            for item in distributions:
                if (
                    isinstance(item, dict)
                    and str(item.get("tokenizer") or "") == selected_tokenizer
                ):
                    return item
        for item in distributions:
            if isinstance(item, dict):
                return item
        return None

    # -------------------------------------------------------------------------
    def _build_benchmark_metrics_table(
        self, global_metrics: list[Any]
    ) -> list[list[str]]:
        rows: list[list[str]] = []
        for item in global_metrics[:18]:
            if not isinstance(item, dict):
                continue
            rows.append(
                [
                    self._short_name(str(item.get("tokenizer") or "N/A")),
                    self._format_percent(item.get("word_recovery_rate")),
                    self._format_percent(item.get("character_coverage")),
                    self._format_number(item.get("subword_fertility"), 4),
                    self._format_number(item.get("token_distribution_entropy"), 4),
                ]
            )
        if not rows:
            return [["N/A", "N/A", "N/A", "N/A", "N/A"]]
        return rows

    # -------------------------------------------------------------------------
    def _benchmark_global_metrics(
        self,
        source: dict[str, Any],
        tokenizer_results: list[Any],
    ) -> list[dict[str, Any]]:
        global_metrics = source.get("global_metrics")
        if isinstance(global_metrics, list):
            return [item for item in global_metrics if isinstance(item, dict)]

        converted: list[dict[str, Any]] = []
        for result in tokenizer_results:
            if not isinstance(result, dict):
                continue
            fidelity = result.get("fidelity")
            if not isinstance(fidelity, dict):
                fidelity = {}
            fragmentation = result.get("fragmentation")
            if not isinstance(fragmentation, dict):
                fragmentation = {}
            converted.append(
                {
                    "tokenizer": str(result.get("tokenizer") or ""),
                    "oov_rate": self._to_number(fidelity.get("unknown_token_rate")),
                    "round_trip_fidelity_rate": self._to_number(
                        fidelity.get("exact_round_trip_rate")
                    ),
                    "word_recovery_rate": 0.0,
                    "character_coverage": self._to_number(
                        fidelity.get("lossless_encodability_rate")
                    ),
                    "subword_fertility": self._to_number(
                        fragmentation.get("pieces_per_word_mean")
                    ),
                    "token_distribution_entropy": 0.0,
                }
            )
        return converted

    # -------------------------------------------------------------------------
    def _benchmark_chart_data(
        self,
        source: dict[str, Any],
        tokenizer_results: list[Any],
    ) -> dict[str, Any]:
        chart_data = source.get("chart_data")
        if not isinstance(chart_data, dict):
            chart_data = {}
        if {
            "speed_metrics",
            "vocabulary_stats",
            "token_length_distributions",
        }.issubset(chart_data.keys()):
            return chart_data

        by_tokenizer = {
            str(item.get("tokenizer") or ""): item
            for item in tokenizer_results
            if isinstance(item, dict)
        }
        efficiency = chart_data.get("efficiency")
        vocabulary = chart_data.get("vocabulary")
        fragmentation = chart_data.get("fragmentation")
        if not isinstance(efficiency, list):
            efficiency = []
        if not isinstance(vocabulary, list):
            vocabulary = []
        if not isinstance(fragmentation, list):
            fragmentation = []

        speed_metrics: list[dict[str, Any]] = []
        for item in efficiency:
            if not isinstance(item, dict):
                continue
            tokenizer_name = str(item.get("tokenizer") or "")
            result = by_tokenizer.get(tokenizer_name, {})
            efficiency_metrics = result.get("efficiency") if isinstance(result, dict) else {}
            if not isinstance(efficiency_metrics, dict):
                efficiency_metrics = {}
            speed_metrics.append(
                {
                    "tokenizer": tokenizer_name,
                    "tokens_per_second": self._to_number(item.get("value")),
                    "chars_per_second": self._to_number(
                        efficiency_metrics.get("encode_chars_per_second_mean")
                    ),
                    "processing_time_seconds": self._to_number(
                        efficiency_metrics.get("end_to_end_wall_time_seconds")
                    ),
                }
            )

        vocabulary_stats: list[dict[str, Any]] = []
        for item in vocabulary:
            if not isinstance(item, dict):
                continue
            vocabulary_stats.append(
                {
                    "tokenizer": str(item.get("tokenizer") or ""),
                    "vocabulary_size": int(round(self._to_number(item.get("value")))),
                }
            )

        token_length_distributions: list[dict[str, Any]] = []
        for item in fragmentation:
            if not isinstance(item, dict):
                continue
            tokenizer_name = str(item.get("tokenizer") or "")
            result = by_tokenizer.get(tokenizer_name, {})
            fragmentation_metrics = (
                result.get("fragmentation") if isinstance(result, dict) else {}
            )
            if not isinstance(fragmentation_metrics, dict):
                fragmentation_metrics = {}
            buckets = fragmentation_metrics.get("fragmentation_by_word_length_bucket")
            if not isinstance(buckets, list):
                buckets = []
            bins = []
            for index, bucket in enumerate(buckets):
                if not isinstance(bucket, dict):
                    continue
                bins.append(
                    {
                        "bin_start": int(index * 4 + 1),
                        "bin_end": int(index * 4 + 4),
                        "count": int(
                            round(
                                self._to_number(bucket.get("pieces_per_word_mean"), 0.0)
                                * 100.0
                            )
                        ),
                    }
                )
            token_length_distributions.append(
                {
                    "tokenizer": tokenizer_name,
                    "bins": bins,
                }
            )

        return {
            "speed_metrics": speed_metrics,
            "vocabulary_stats": vocabulary_stats,
            "token_length_distributions": token_length_distributions,
        }

    # -------------------------------------------------------------------------
    def _extract_nested(self, payload: dict[str, Any], key: str) -> dict[str, Any]:
        candidate = payload.get(key)
        return candidate if isinstance(candidate, dict) else {}

    # -------------------------------------------------------------------------
    def _short_name(self, tokenizer_name: str, max_length: int = 24) -> str:
        trimmed = tokenizer_name.strip()
        if not trimmed:
            return "N/A"
        short = trimmed.split("/")[-1] or trimmed
        if len(short) <= max_length:
            return short
        return f"{short[: max(1, max_length - 3)]}..."

    # -------------------------------------------------------------------------
    def _to_number(self, value: Any, fallback: float = 0.0) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return fallback
        return fallback

    # -------------------------------------------------------------------------
    def _format_count(self, value: Any) -> str:
        return f"{int(round(max(0.0, self._to_number(value, 0.0)))):,}"

    # -------------------------------------------------------------------------
    def _format_number(self, value: Any, decimals: int) -> str:
        return f"{self._to_number(value, 0.0):.{decimals}f}"

    # -------------------------------------------------------------------------
    def _format_percent(self, value: Any) -> str:
        numeric = self._to_number(value, 0.0)
        if numeric <= 1.0:
            numeric *= 100.0
        return f"{numeric:.2f}%"

    # -------------------------------------------------------------------------
    def _as_percent(self, value: Any) -> float:
        numeric = self._to_number(value, 0.0)
        if numeric <= 1.0:
            numeric *= 100.0
        return numeric

    # -------------------------------------------------------------------------
    def _parse_json_like(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return value
