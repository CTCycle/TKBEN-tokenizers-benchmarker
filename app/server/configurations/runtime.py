from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, ValidationError, model_validator

from server.common.utils.logger import logger
from server.configurations.settings import (
    BenchmarkSettings,
    DatasetSettings,
    JobsSettings,
    ServerSettings,
    TokenizerSettings,
)


RUNTIME_SETTINGS_SCHEMA_VERSION = 1
RUNTIME_SETTINGS_FILENAME = "runtime-settings.json"
RUNTIME_SETTINGS_WARNING = (
    "Saved runtime settings could not be loaded; typed defaults are in use. "
    "Save or reset settings to replace the file."
)


###############################################################################
class RuntimeSettingsError(RuntimeError):
    """Base error for application-managed runtime settings operations."""


###############################################################################
class RuntimeSettingsValidationError(RuntimeSettingsError):
    """Raised when a runtime override cannot produce a valid settings snapshot."""


###############################################################################
class RuntimeSettingsPersistenceError(RuntimeSettingsError):
    """Raised when a validated runtime settings document cannot be persisted."""


###############################################################################
class RuntimeSettingsConflictError(RuntimeSettingsError):
    """Raised when a caller tries to update an obsolete settings revision."""

    def __init__(self, expected_revision: int, actual_revision: int) -> None:
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        super().__init__(
            "Runtime settings changed since this form was loaded. "
            "Reload the settings and try again."
        )


###############################################################################
class _StrictOverrideModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @model_validator(mode="after")
    def reject_explicit_nulls(self) -> "_StrictOverrideModel":
        for field_name in self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} must be provided as a value.")
        return self


###############################################################################
class _TokenizerOverrides(_StrictOverrideModel):
    default_discovery_limit: StrictInt | None = Field(
        default=None,
        ge=1,
        le=250,
    )
    max_discovery_limit: StrictInt | None = Field(default=None, ge=1, le=250)
    max_discovery_candidates: StrictInt | None = Field(default=None, ge=1)
    metadata_candidate_multiplier: StrictInt | None = Field(
        default=None,
        ge=1,
        le=10,
    )
    max_upload_bytes: StrictInt | None = Field(default=None, ge=1)


###############################################################################
class _DatasetOverrides(_StrictOverrideModel):
    histogram_bins: StrictInt | None = Field(default=None, ge=5, le=100)
    streaming_batch_size: StrictInt | None = Field(default=None, ge=100)
    max_upload_bytes: StrictInt | None = Field(default=None, ge=1)
    download_timeout_seconds: FiniteFloat | None = Field(default=None, ge=1.0)
    download_retry_attempts: StrictInt | None = Field(
        default=None,
        ge=1,
        le=10,
    )
    download_retry_backoff_seconds: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        le=60.0,
    )


###############################################################################
class _BenchmarkOverrides(_StrictOverrideModel):
    default_max_documents: StrictInt | None = Field(default=None, ge=1, le=100_000)
    default_batch_size: StrictInt | None = Field(default=None, ge=1, le=4096)
    default_parallelism: StrictInt | None = Field(default=None, ge=1, le=128)
    streaming_batch_size: StrictInt | None = Field(default=None, ge=100)


###############################################################################
class _JobsOverrides(_StrictOverrideModel):
    polling_interval: FiniteFloat | None = Field(default=None, ge=0.25)


###############################################################################
class _RuntimeOverrides(_StrictOverrideModel):
    tokenizers: _TokenizerOverrides | None = None
    datasets: _DatasetOverrides | None = None
    benchmarks: _BenchmarkOverrides | None = None
    jobs: _JobsOverrides | None = None


###############################################################################
class _RuntimeOverrideDocument(_StrictOverrideModel):
    schema_version: StrictInt
    revision: StrictInt = Field(ge=0)
    overrides: _RuntimeOverrides = Field(default_factory=_RuntimeOverrides)


###############################################################################
@dataclass(frozen=True, slots=True)
class RuntimeSettingsState:
    settings: ServerSettings
    defaults: ServerSettings
    revision: int
    overridden_keys: tuple[str, ...]
    warning: str | None


###############################################################################
class RuntimeSettingsStore:
    """Own sparse runtime overrides and the effective immutable settings snapshot."""

    _GROUP_MODELS: ClassVar[dict[str, type[_StrictOverrideModel]]] = {
        "tokenizers": _TokenizerOverrides,
        "datasets": _DatasetOverrides,
        "benchmarks": _BenchmarkOverrides,
        "jobs": _JobsOverrides,
    }
    _EDITABLE_KEYS: ClassVar[tuple[str, ...]] = (
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
    )

    def __init__(self, defaults: ServerSettings, path: str | Path | None = None) -> None:
        self._lock = RLock()
        self._defaults = defaults
        self._path = (
            Path(path)
            if path is not None
            else defaults.paths.resources / RUNTIME_SETTINGS_FILENAME
        ).expanduser().resolve()
        self._effective = defaults
        self._overrides: dict[str, dict[str, object]] = {}
        self._revision = 0
        self._warning: str | None = None
        self._load_from_disk()

    @property
    def path(self) -> Path:
        return self._path

    @classmethod
    def editable_keys(cls) -> tuple[str, ...]:
        return cls._EDITABLE_KEYS

    def get_state(self) -> RuntimeSettingsState:
        with self._lock:
            return self._state()

    def reload(self) -> RuntimeSettingsState:
        with self._lock:
            self._effective = self._defaults
            self._overrides = {}
            self._revision = 0
            self._warning = None
            self._load_from_disk()
            return self._state()

    def apply_patch(
        self,
        patch: Mapping[str, Mapping[str, object]] | BaseModel,
        *,
        expected_revision: int,
    ) -> RuntimeSettingsState:
        with self._lock:
            self._validate_revision(expected_revision)
            raw_patch = self._patch_mapping(patch)
            candidate = self._copy_overrides(self._overrides)

            for group_name, values in raw_patch.items():
                if group_name not in self._GROUP_MODELS:
                    raise RuntimeSettingsValidationError(
                        f"Unknown runtime settings group: {group_name}."
                    )
                if not isinstance(values, Mapping):
                    raise RuntimeSettingsValidationError(
                        f"Runtime settings group '{group_name}' must be an object."
                    )
                merged = {**candidate.get(group_name, {}), **dict(values)}
                candidate[group_name] = merged

            candidate = self._canonicalize_overrides(candidate)
            return self._commit(candidate)

    def reset(
        self,
        *,
        expected_revision: int,
        keys: Iterable[str] | None = None,
        reset_all: bool = False,
    ) -> RuntimeSettingsState:
        with self._lock:
            self._validate_revision(expected_revision)
            requested_keys = list(keys) if keys is not None else []
            if reset_all or not requested_keys:
                candidate: dict[str, dict[str, object]] = {}
            else:
                candidate = self._copy_overrides(self._overrides)
                for key in requested_keys:
                    if key not in self._EDITABLE_KEYS:
                        raise RuntimeSettingsValidationError(
                            f"Unknown runtime setting: {key}."
                        )
                    group_name, field_name = key.split(".", maxsplit=1)
                    group = candidate.get(group_name)
                    if group is None:
                        continue
                    group.pop(field_name, None)
                    if not group:
                        candidate.pop(group_name, None)

            candidate = self._canonicalize_overrides(candidate)
            return self._commit(candidate)

    def _state(self) -> RuntimeSettingsState:
        return RuntimeSettingsState(
            settings=self._effective,
            defaults=self._defaults,
            revision=self._revision,
            overridden_keys=self._overridden_keys(self._overrides),
            warning=self._warning,
        )

    def _load_from_disk(self) -> None:
        if not self._path.is_file():
            return

        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            document = _RuntimeOverrideDocument.model_validate(payload)
            if document.schema_version != RUNTIME_SETTINGS_SCHEMA_VERSION:
                raise RuntimeSettingsValidationError(
                    "Unsupported runtime settings schema version."
                )
            raw_overrides = cast(
                Mapping[str, object],
                document.overrides.model_dump(exclude_none=True),
            )
            overrides = self._canonicalize_overrides(raw_overrides)
            self._effective = self._build_effective(overrides)
            self._overrides = overrides
            self._revision = document.revision
        except (OSError, TypeError, ValueError, ValidationError, RuntimeSettingsError):
            self._effective = self._defaults
            self._overrides = {}
            self._revision = 0
            self._warning = RUNTIME_SETTINGS_WARNING
            logger.warning("Invalid runtime settings overrides; using typed defaults.")

    def _validate_revision(self, expected_revision: int) -> None:
        if expected_revision != self._revision:
            raise RuntimeSettingsConflictError(expected_revision, self._revision)

    def _commit(self, overrides: dict[str, dict[str, object]]) -> RuntimeSettingsState:
        effective = self._build_effective(overrides)
        if (
            overrides == self._overrides
            and self._warning is None
            and (overrides or not self._path.exists())
        ):
            return self._state()

        next_revision = self._revision + 1
        self._persist(overrides, next_revision)
        self._overrides = overrides
        self._effective = effective
        self._revision = next_revision
        self._warning = None
        return self._state()

    def _build_effective(
        self,
        overrides: Mapping[str, Mapping[str, object]],
    ) -> ServerSettings:
        try:
            datasets = DatasetSettings.model_validate(
                {**self._defaults.datasets.model_dump(), **overrides.get("datasets", {})}
            )
            tokenizers = TokenizerSettings.model_validate(
                {
                    **self._defaults.tokenizers.model_dump(),
                    **overrides.get("tokenizers", {}),
                }
            )
            benchmarks = BenchmarkSettings.model_validate(
                {
                    **self._defaults.benchmarks.model_dump(),
                    **overrides.get("benchmarks", {}),
                }
            )
            jobs = JobsSettings.model_validate(
                {**self._defaults.jobs.model_dump(), **overrides.get("jobs", {})}
            )
            payload = self._defaults.model_dump()
            payload.update(
                {
                    "datasets": datasets,
                    "tokenizers": tokenizers,
                    "benchmarks": benchmarks,
                    "jobs": jobs,
                }
            )
            return ServerSettings.model_validate(payload)
        except ValidationError as exc:
            raise RuntimeSettingsValidationError(
                self._format_validation_error(exc)
            ) from exc

    def _canonicalize_overrides(
        self,
        raw_overrides: Mapping[str, object],
    ) -> dict[str, dict[str, object]]:
        canonical: dict[str, dict[str, object]] = {}
        for group_name, values in raw_overrides.items():
            group_model = self._GROUP_MODELS.get(group_name)
            if group_model is None:
                raise RuntimeSettingsValidationError(
                    f"Unknown runtime settings group: {group_name}."
                )
            if not isinstance(values, Mapping):
                raise RuntimeSettingsValidationError(
                    f"Runtime settings group '{group_name}' must be an object."
                )
            try:
                validated = group_model.model_validate(values)
            except ValidationError as exc:
                raise RuntimeSettingsValidationError(
                    self._format_validation_error(exc)
                ) from exc
            values_without_none = validated.model_dump(exclude_none=True)
            defaults = getattr(self._defaults, group_name)
            non_default = {
                field_name: value
                for field_name, value in values_without_none.items()
                if value != getattr(defaults, field_name)
            }
            if non_default:
                canonical[group_name] = non_default
        return canonical

    def _persist(
        self,
        overrides: Mapping[str, Mapping[str, object]],
        revision: int,
    ) -> None:
        if not overrides:
            try:
                if self._path.exists():
                    self._path.unlink()
            except OSError as exc:
                raise RuntimeSettingsPersistenceError(
                    "Unable to remove persisted runtime settings."
                ) from exc
            return

        temporary_path: str | None = None
        descriptor: int | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                dir=self._path.parent,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
                descriptor = None
                json.dump(
                    {
                        "schema_version": RUNTIME_SETTINGS_SCHEMA_VERSION,
                        "revision": revision,
                        "overrides": overrides,
                    },
                    destination,
                    indent=2,
                    sort_keys=True,
                )
                destination.write("\n")
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary_path, self._path)
            temporary_path = None
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeSettingsPersistenceError(
                "Unable to persist runtime settings."
            ) from exc
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass

    def _copy_overrides(
        self,
        overrides: Mapping[str, Mapping[str, object]],
    ) -> dict[str, dict[str, object]]:
        return {group_name: dict(values) for group_name, values in overrides.items()}

    def _overridden_keys(
        self,
        overrides: Mapping[str, Mapping[str, object]],
    ) -> tuple[str, ...]:
        return tuple(
            key
            for key in self._EDITABLE_KEYS
            if key.split(".", maxsplit=1)[1]
            in overrides.get(key.split(".", maxsplit=1)[0], {})
        )

    def _patch_mapping(
        self,
        patch: Mapping[str, Mapping[str, object]] | BaseModel,
    ) -> Mapping[str, object]:
        if isinstance(patch, BaseModel):
            raw_patch = patch.model_dump(exclude_unset=True, exclude_none=True)
        elif isinstance(patch, Mapping):
            raw_patch = dict(patch)
        else:
            raise RuntimeSettingsValidationError("Runtime settings patch must be an object.")
        return cast(Mapping[str, object], raw_patch)

    @staticmethod
    def _format_validation_error(exc: ValidationError) -> str:
        messages: list[str] = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ()))
            message = str(error.get("msg", "Invalid value."))
            messages.append(f"{location}: {message}" if location else message)
        return "Invalid runtime settings. " + " ".join(messages)


__all__ = [
    "RUNTIME_SETTINGS_FILENAME",
    "RUNTIME_SETTINGS_SCHEMA_VERSION",
    "RuntimeSettingsConflictError",
    "RuntimeSettingsError",
    "RuntimeSettingsPersistenceError",
    "RuntimeSettingsState",
    "RuntimeSettingsStore",
    "RuntimeSettingsValidationError",
]
