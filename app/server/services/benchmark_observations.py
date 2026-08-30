from __future__ import annotations

from dataclasses import dataclass
from typing import Any


###############################################################################
@dataclass(frozen=True)
class TokenizerRunConfig:
    add_special_tokens: bool = False
    padding: bool = False
    truncation: bool = False
    max_length: int | None = None
    batch_size: int = 64


###############################################################################
@dataclass(frozen=True)
class BatchObservation:
    tokenizer_id: str
    trial_index: int
    batch_index: int
    documents: int
    input_utf8_bytes: int
    token_count: int
    unknown_token_count: int | None
    elapsed_ns: int
    peak_rss_mb: float | None
    error_count: int = 0


###############################################################################
@dataclass(frozen=True)
class BenchmarkEnvironment:
    python_version: str
    platform: str
    processor: str
    cpu_count_logical: int | None
    package_versions: dict[str, str]
    environment: dict[str, str]
    extra: dict[str, Any]
