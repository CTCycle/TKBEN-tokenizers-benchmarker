# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, field_validator, model_validator

###############################################################################
class _SettingsContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

###############################################################################
class _SettingsPatchContract(_SettingsContract):

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def reject_explicit_nulls(self) -> "_SettingsPatchContract":
        for field_name in self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} must be provided as a value.")
        return self

###############################################################################
class RuntimeTokenizerSettingsResponse(_SettingsContract):
    default_discovery_limit: int = Field(
        description="Default tokenizer discovery result count"
    )
    max_discovery_limit: int = Field(description="Maximum tokenizer discovery result count")
    max_discovery_candidates: int = Field(
        description="Maximum tokenizer repositories considered during discovery"
    )
    metadata_candidate_multiplier: int = Field(
        description="Tokenizer discovery metadata over-fetch multiplier"
    )
    max_upload_bytes: int = Field(description="Maximum tokenizer upload size in bytes")

###############################################################################
class RuntimeDatasetSettingsResponse(_SettingsContract):
    histogram_bins: int = Field(
        description="Number of bins used in new dataset and tokenizer histograms"
    )
    streaming_batch_size: int = Field(
        description="Dataset processing batch size for new operations"
    )
    max_upload_bytes: int = Field(description="Maximum dataset upload size in bytes")
    download_timeout_seconds: float = Field(description="Dataset download timeout in seconds")
    download_retry_attempts: int = Field(description="Dataset download retry attempts")
    download_retry_backoff_seconds: float = Field(
        description="Delay between dataset download retries in seconds"
    )

###############################################################################
class RuntimeBenchmarkSettingsResponse(_SettingsContract):
    default_max_documents: int = Field(
        description="Default document cap for newly opened benchmark runs"
    )
    default_batch_size: int = Field(
        description="Default tokenizer batch size for newly opened benchmark runs"
    )
    default_parallelism: int = Field(
        description="Default tokenizer parallelism for newly opened benchmark runs"
    )
    streaming_batch_size: int = Field(
        description="Benchmark processing batch size for new runs"
    )

###############################################################################
class RuntimeJobSettingsResponse(_SettingsContract):
    polling_interval: float = Field(
        description="Polling interval for newly created asynchronous jobs in seconds"
    )
    max_stored_jobs: int = Field(
        description="Maximum number of jobs kept in the database before the oldest finished jobs are removed"
    )

###############################################################################
class RuntimeSettingsValues(_SettingsContract):
    tokenizers: RuntimeTokenizerSettingsResponse
    datasets: RuntimeDatasetSettingsResponse
    benchmarks: RuntimeBenchmarkSettingsResponse
    jobs: RuntimeJobSettingsResponse


###############################################################################
RuntimeSettingKey = Literal[
    "tokenizers.default_discovery_limit",
    "tokenizers.max_discovery_limit",
    "tokenizers.max_discovery_candidates",
    "tokenizers.metadata_candidate_multiplier",
    "tokenizers.max_upload_bytes",
    "datasets.histogram_bins",
    "datasets.streaming_batch_size",
    "datasets.max_upload_bytes",
    "datasets.download_timeout_seconds",
    "datasets.download_retry_attempts",
    "datasets.download_retry_backoff_seconds",
    "benchmarks.default_max_documents",
    "benchmarks.default_batch_size",
    "benchmarks.default_parallelism",
    "benchmarks.streaming_batch_size",
    "jobs.polling_interval",
    "jobs.max_stored_jobs",
]

###############################################################################
class RuntimeSettingsResponse(_SettingsContract):
    revision: int = Field(ge=0)
    settings: RuntimeSettingsValues
    defaults: RuntimeSettingsValues
    overridden_keys: list[RuntimeSettingKey] = Field(default_factory=list)
    warning: str | None = None

###############################################################################
DashboardVisualization = Literal[
    "bar",
    "horizontal_bar",
    "interval_bar",
    "dot_whisker",
    "box_plot",
    "histogram",
    "grouped_bar",
    "heatmap",
]


###############################################################################
class DashboardUiPreferences(_SettingsContract):
    ordered_widget_ids: list[str] = Field(default_factory=list)
    hidden_widget_ids: list[str] = Field(default_factory=list)
    visualization_by_widget_id: dict[str, DashboardVisualization] = Field(
        default_factory=dict
    )
    baseline_by_report_id: dict[str, str] = Field(default_factory=dict)
    tokenizer_colors: dict[str, str] = Field(default_factory=dict)

    # -------------------------------------------------------------------------
    @field_validator("ordered_widget_ids", "hidden_widget_ids")
    @classmethod
    def unique_non_empty_ids(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value) or len(value) != len(set(value)):
            raise ValueError("Widget IDs must be non-empty and unique.")
        return value

    # -------------------------------------------------------------------------
    @field_validator("baseline_by_report_id")
    @classmethod
    def valid_baseline_preferences(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            not report_id.isdecimal()
            or int(report_id) <= 0
            or not tokenizer.strip()
            for report_id, tokenizer in value.items()
        ):
            raise ValueError("Baseline preferences require a report ID and tokenizer.")
        return value

    # -------------------------------------------------------------------------
    @field_validator("tokenizer_colors")
    @classmethod
    def valid_tokenizer_colors(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            not tokenizer.strip() or not re.fullmatch(r"#[0-9a-fA-F]{6}", color)
            for tokenizer, color in value.items()
        ):
            raise ValueError("Tokenizer colors require a tokenizer and #rrggbb color.")
        return value


###############################################################################
class DashboardUiPreferencesRequest(_SettingsContract):
    expected_revision: StrictInt = Field(ge=0)
    preferences: DashboardUiPreferences


###############################################################################
class DashboardUiPreferencesResponse(_SettingsContract):
    revision: int = Field(ge=0)
    preferences: DashboardUiPreferences

###############################################################################
class RuntimeTokenizerSettingsPatch(_SettingsPatchContract):
    default_discovery_limit: StrictInt | None = Field(default=None, ge=1, le=250)
    max_discovery_limit: StrictInt | None = Field(default=None, ge=1, le=250)
    max_discovery_candidates: StrictInt | None = Field(default=None, ge=1)
    metadata_candidate_multiplier: StrictInt | None = Field(
        default=None,
        ge=1,
        le=10,
    )
    max_upload_bytes: StrictInt | None = Field(default=None, ge=1)

###############################################################################
class RuntimeDatasetSettingsPatch(_SettingsPatchContract):
    histogram_bins: StrictInt | None = Field(default=None, ge=5, le=100)
    streaming_batch_size: StrictInt | None = Field(default=None, ge=100)
    max_upload_bytes: StrictInt | None = Field(default=None, ge=1)
    download_timeout_seconds: FiniteFloat | None = Field(default=None, ge=1.0)
    download_retry_attempts: StrictInt | None = Field(default=None, ge=1, le=10)
    download_retry_backoff_seconds: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        le=60.0,
    )

###############################################################################
class RuntimeBenchmarkSettingsPatch(_SettingsPatchContract):
    default_max_documents: StrictInt | None = Field(default=None, ge=1, le=100_000)
    default_batch_size: StrictInt | None = Field(default=None, ge=1, le=4096)
    default_parallelism: StrictInt | None = Field(default=None, ge=1, le=128)
    streaming_batch_size: StrictInt | None = Field(default=None, ge=100)

###############################################################################
class RuntimeJobSettingsPatch(_SettingsPatchContract):
    polling_interval: FiniteFloat | None = Field(default=None, ge=0.25)
    max_stored_jobs: StrictInt | None = Field(default=None, ge=1)

###############################################################################
class RuntimeSettingsPatchRequest(_SettingsPatchContract):
    expected_revision: StrictInt = Field(ge=0)
    tokenizers: RuntimeTokenizerSettingsPatch | None = None
    datasets: RuntimeDatasetSettingsPatch | None = None
    benchmarks: RuntimeBenchmarkSettingsPatch | None = None
    jobs: RuntimeJobSettingsPatch | None = None

###############################################################################
class RuntimeSettingsResetRequest(_SettingsContract):
    expected_revision: StrictInt = Field(ge=0)
    keys: list[RuntimeSettingKey] | None = None
    all: bool = False

    # -------------------------------------------------------------------------
    @field_validator("keys")
    @classmethod
    def unique_keys(
        cls,
        value: list[RuntimeSettingKey] | None,
    ) -> list[RuntimeSettingKey] | None:
        if value is None:
            return None
        return list(dict.fromkeys(value))

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_reset_scope(self) -> "RuntimeSettingsResetRequest":
        if self.all and self.keys:
            raise ValueError("Specify either 'keys' or 'all', not both.")
        return self


__all__ = [
    "DashboardUiPreferences",
    "DashboardUiPreferencesRequest",
    "DashboardUiPreferencesResponse",
    "RuntimeBenchmarkSettingsPatch",
    "RuntimeBenchmarkSettingsResponse",
    "RuntimeDatasetSettingsPatch",
    "RuntimeDatasetSettingsResponse",
    "RuntimeJobSettingsPatch",
    "RuntimeJobSettingsResponse",
    "RuntimeSettingKey",
    "RuntimeSettingsPatchRequest",
    "RuntimeSettingsResetRequest",
    "RuntimeSettingsResponse",
    "RuntimeSettingsValues",
    "RuntimeTokenizerSettingsPatch",
    "RuntimeTokenizerSettingsResponse",
]


###############################################################################
class PathSettingsRequest(_SettingsContract):
    data_dir: str = Field(min_length=1, max_length=1024)
    log_dir: str = Field(min_length=1, max_length=1024)

###############################################################################
class PathSettingsResponse(_SettingsContract):
    data_dir: str = Field(description="Configured data directory")
    log_dir: str = Field(description="Configured log directory")
    active_data_dir: str = Field(description="Data directory used by the running backend")
    active_log_dir: str = Field(description="Log directory used by the running backend")
    restart_required: bool = Field(
        description="Whether a backend restart is needed to apply the configured paths"
    )
