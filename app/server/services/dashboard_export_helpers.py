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
    def _parse_word_frequency(self, value: Any) -> list[dict[str, str | int]]:
        if not isinstance(value, list):
            return []
        rows: list[dict[str, str | int]] = []
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
        rows.sort(key=lambda item: (-int(item["count"]), str(item["word"])))
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
        if value is None:
            return "N/A"
        return f"{self._to_number(value, 0.0):.{decimals}f}"

    # -------------------------------------------------------------------------
    def _format_percent(self, value: Any) -> str:
        if value is None:
            return "N/A"
        numeric = self._to_number(value, 0.0)
        if numeric <= 1.0:
            numeric *= 100.0
        return f"{numeric:.2f}%"

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
