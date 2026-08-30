from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


###############################################################################
@dataclass
class LengthStatistics:
    document_count: int = 0
    total_length: int = 0
    min_length: int | None = None
    max_length: int | None = None

    # -------------------------------------------------------------------------
    def update(self, length: int) -> None:
        self.document_count += 1
        self.total_length += length
        if self.min_length is None or length < self.min_length:
            self.min_length = length
        if self.max_length is None or length > self.max_length:
            self.max_length = length

    # -------------------------------------------------------------------------
    def resolved_min(self) -> int:
        return self.min_length if self.min_length is not None else 0

    # -------------------------------------------------------------------------
    def resolved_max(self) -> int:
        return self.max_length if self.max_length is not None else 0

    # -------------------------------------------------------------------------
    def mean(self) -> float:
        if self.document_count == 0:
            return 0.0
        return self.total_length / self.document_count


###############################################################################
class HistogramBuilder:
    # -------------------------------------------------------------------------
    def __init__(self, stats: LengthStatistics, bins: int) -> None:
        self.stats = stats
        self.bin_count = bins if stats.document_count > 0 else 0
        self.bin_width = self._compute_bin_width()
        self.bin_edges = self._build_bin_edges()
        self.counts = [0] * (len(self.bin_edges) - 1) if self.bin_edges else []

    # -------------------------------------------------------------------------
    def _compute_bin_width(self) -> int:
        if self.stats.document_count == 0:
            return 1
        span = (self.stats.resolved_max() - self.stats.resolved_min()) + 1
        return max(1, math.ceil(span / max(1, self.bin_count)))

    # -------------------------------------------------------------------------
    def _build_bin_edges(self) -> list[int]:
        if self.stats.document_count == 0:
            return []
        edges = [self.stats.resolved_min()]
        for _ in range(self.bin_count):
            edges.append(edges[-1] + self.bin_width)
        return edges

    # -------------------------------------------------------------------------
    def add(self, length: int) -> None:
        if not self.counts:
            return
        base = self.stats.resolved_min()
        index = min(
            (length - base) // self.bin_width,
            len(self.counts) - 1,
        )
        self.counts[index] += 1

    # -------------------------------------------------------------------------
    def add_batch(self, length: int, count: int) -> None:
        if not self.counts or count <= 0:
            return
        base = self.stats.resolved_min()
        index = min(
            (length - base) // self.bin_width,
            len(self.counts) - 1,
        )
        self.counts[index] += count

    # -------------------------------------------------------------------------
    def _midpoint(self, index: int) -> float:
        start = self.bin_edges[index]
        end = self.bin_edges[index + 1]
        return (start + end - 1) / 2.0

    # -------------------------------------------------------------------------
    def _build_labels(self) -> list[str]:
        labels: list[str] = []
        if not self.counts:
            return labels
        for idx in range(len(self.counts)):
            left = int(self.bin_edges[idx])
            right = int(self.bin_edges[idx + 1] - 1)
            if right < left:
                right = left
            labels.append(f"{left}-{right}" if left != right else f"{left}")
        return labels

    # -------------------------------------------------------------------------
    def _median(self) -> float:
        if not self.counts or self.stats.document_count == 0:
            return 0.0
        low_value, high_value = self._resolve_median_bounds()
        return (low_value + high_value) / 2.0

    # -------------------------------------------------------------------------
    def _resolve_median_bounds(self) -> tuple[float, float]:
        low_rank = (self.stats.document_count - 1) // 2
        high_rank = self.stats.document_count // 2
        cumulative = 0
        low_value: float | None = None
        high_value: float | None = None
        for idx, count in enumerate(self.counts):
            if count == 0:
                cumulative += count
                continue
            cumulative += count
            if low_value is None and cumulative > low_rank:
                low_value = self._midpoint(idx)
            if high_value is None and cumulative > high_rank:
                high_value = self._midpoint(idx)
            if low_value is not None and high_value is not None:
                break
        if low_value is None:
            low_value = self.stats.mean()
        if high_value is None:
            high_value = low_value
        return low_value, high_value

    # -------------------------------------------------------------------------
    def build(self) -> dict[str, Any]:
        if self.stats.document_count == 0:
            return {
                "bins": [],
                "counts": [],
                "bin_edges": [],
                "min_length": 0,
                "max_length": 0,
                "mean_length": 0.0,
                "median_length": 0.0,
            }
        return {
            "bins": self._build_labels(),
            "counts": self.counts,
            "bin_edges": [float(edge) for edge in self.bin_edges],
            "min_length": self.stats.resolved_min(),
            "max_length": self.stats.resolved_max(),
            "mean_length": self.stats.mean(),
            "median_length": self._median(),
        }
