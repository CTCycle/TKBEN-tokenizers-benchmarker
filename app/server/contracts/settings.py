from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, field_validator, model_validator


class _SettingsContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _SettingsPatchContract(_SettingsContract):
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
    streaming_batch_size: int = Field(
        description="Benchmark processing batch size for new runs"
    )
    default_max_documents: int = Field(
        description="Default maximum document count for new benchmark runs"
    )
    default_warmup_trials: int = Field(
        description="Default warmup trial count for new benchmark runs"
    )
    default_timed_trials: int = Field(
        description="Default timed trial count for new benchmark runs"
    )
    default_batch_size: int = Field(
        description="Default tokenizer encode batch size for new benchmark runs"
    )
    default_parallelism: int = Field(
        description="Default tokenizer parallelism for new benchmark runs"
    )


###############################################################################
class RuntimeJobSettingsResponse(_SettingsContract):
    polling_interval: float = Field(
        description="Polling interval for newly created asynchronous jobs in seconds"
    )
    terminal_retention_seconds: float = Field(
        description="Retention time for completed, failed, and cancelled in-memory jobs"
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
    "benchmarks.streaming_batch_size",
    "benchmarks.default_max_documents",
    "benchmarks.default_warmup_trials",
    "benchmarks.default_timed_trials",
    "benchmarks.default_batch_size",
    "benchmarks.default_parallelism",
    "jobs.polling_interval",
    "jobs.terminal_retention_seconds",
]


###############################################################################
class RuntimeSettingsResponse(_SettingsContract):
    revision: int = Field(ge=0)
    settings: RuntimeSettingsValues
    defaults: RuntimeSettingsValues
    overridden_keys: list[RuntimeSettingKey] = Field(default_factory=list)
    warning: str | None = None


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
    streaming_batch_size: StrictInt | None = Field(default=None, ge=100)
    default_max_documents: StrictInt | None = Field(default=None, ge=1, le=100_000)
    default_warmup_trials: StrictInt | None = Field(default=None, ge=0, le=100)
    default_timed_trials: StrictInt | None = Field(default=None, ge=1, le=200)
    default_batch_size: StrictInt | None = Field(default=None, ge=1, le=4096)
    default_parallelism: StrictInt | None = Field(default=None, ge=1, le=128)


###############################################################################
class RuntimeJobSettingsPatch(_SettingsPatchContract):
    polling_interval: FiniteFloat | None = Field(default=None, ge=0.25)
    terminal_retention_seconds: FiniteFloat | None = Field(default=None, ge=0.0)


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

    @field_validator("keys")
    @classmethod
    def unique_keys(
        cls,
        value: list[RuntimeSettingKey] | None,
    ) -> list[RuntimeSettingKey] | None:
        if value is None:
            return None
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def validate_reset_scope(self) -> "RuntimeSettingsResetRequest":
        if self.all and self.keys:
            raise ValueError("Specify either 'keys' or 'all', not both.")
        return self


__all__ = [
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
