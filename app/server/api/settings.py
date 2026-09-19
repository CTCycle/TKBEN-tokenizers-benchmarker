from __future__ import annotations

from typing import cast

from fastapi import APIRouter, HTTPException, Request, status

from server.common.constants import API_ROUTE_SETTINGS_RESET, API_ROUTER_PREFIX_SETTINGS
from server.configurations import (
    apply_runtime_settings_patch,
    get_runtime_settings_state,
    reset_runtime_settings,
)
from server.configurations.runtime import (
    RuntimeSettingsConflictError,
    RuntimeSettingsPersistenceError,
    RuntimeSettingsValidationError,
)
from server.configurations.settings import ServerSettings
from server.contracts.settings import (
    RuntimeBenchmarkSettingsResponse,
    RuntimeDatasetSettingsResponse,
    RuntimeJobSettingsResponse,
    RuntimeSettingsPatchRequest,
    RuntimeSettingsResetRequest,
    RuntimeSettingsResponse,
    RuntimeSettingKey,
    RuntimeSettingsValues,
    RuntimeTokenizerSettingsResponse,
)


router = APIRouter(prefix=API_ROUTER_PREFIX_SETTINGS, tags=["settings"])

###############################################################################
def _build_response() -> RuntimeSettingsResponse:
    state = get_runtime_settings_state()
    return RuntimeSettingsResponse(
        revision=state.revision,
        settings=_runtime_values(state.settings),
        defaults=_runtime_values(state.defaults),
        overridden_keys=cast(list[RuntimeSettingKey], list(state.overridden_keys)),
        warning=state.warning,
    )

###############################################################################
def _runtime_values(settings: ServerSettings) -> RuntimeSettingsValues:
    return RuntimeSettingsValues(
        tokenizers=RuntimeTokenizerSettingsResponse(
            default_discovery_limit=settings.tokenizers.default_discovery_limit,
            max_discovery_limit=settings.tokenizers.max_discovery_limit,
            max_discovery_candidates=settings.tokenizers.max_discovery_candidates,
            metadata_candidate_multiplier=settings.tokenizers.metadata_candidate_multiplier,
            max_upload_bytes=settings.tokenizers.max_upload_bytes,
        ),
        datasets=RuntimeDatasetSettingsResponse(
            histogram_bins=settings.datasets.histogram_bins,
            streaming_batch_size=settings.datasets.streaming_batch_size,
            max_upload_bytes=settings.datasets.max_upload_bytes,
            download_timeout_seconds=settings.datasets.download_timeout_seconds,
            download_retry_attempts=settings.datasets.download_retry_attempts,
            download_retry_backoff_seconds=settings.datasets.download_retry_backoff_seconds,
        ),
        benchmarks=RuntimeBenchmarkSettingsResponse(
            default_max_documents=settings.benchmarks.default_max_documents,
            default_batch_size=settings.benchmarks.default_batch_size,
            default_parallelism=settings.benchmarks.default_parallelism,
            streaming_batch_size=settings.benchmarks.streaming_batch_size,
        ),
        jobs=RuntimeJobSettingsResponse(
            polling_interval=settings.jobs.polling_interval,
        ),
    )

###############################################################################
def _raise_settings_error(exc: RuntimeSettingsValidationError) -> None:
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=str(exc),
    ) from exc

###############################################################################
def _replace_application_snapshot(request: Request) -> None:
    request.app.state.settings = get_runtime_settings_state().settings

###############################################################################
@router.get(
    "",
    response_model=RuntimeSettingsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_settings() -> RuntimeSettingsResponse:
    return _build_response()

###############################################################################
@router.patch(
    "",
    response_model=RuntimeSettingsResponse,
    status_code=status.HTTP_200_OK,
)
async def patch_settings(
    request: Request,
    payload: RuntimeSettingsPatchRequest,
) -> RuntimeSettingsResponse:
    patch = payload.model_dump(
        exclude={"expected_revision"},
        exclude_unset=True,
        exclude_none=True,
    )
    try:
        apply_runtime_settings_patch(
            patch,
            expected_revision=payload.expected_revision,
        )
    except RuntimeSettingsConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except RuntimeSettingsPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to persist runtime settings.",
        ) from exc
    except RuntimeSettingsValidationError as exc:
        _raise_settings_error(exc)

    response = _build_response()
    _replace_application_snapshot(request)
    return response

###############################################################################
@router.post(
    API_ROUTE_SETTINGS_RESET,
    response_model=RuntimeSettingsResponse,
    status_code=status.HTTP_200_OK,
)
async def reset_settings(
    request: Request,
    payload: RuntimeSettingsResetRequest,
) -> RuntimeSettingsResponse:
    try:
        reset_runtime_settings(
            expected_revision=payload.expected_revision,
            keys=payload.keys,
            reset_all=payload.all,
        )
    except RuntimeSettingsConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except RuntimeSettingsPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to persist runtime settings.",
        ) from exc
    except RuntimeSettingsValidationError as exc:
        _raise_settings_error(exc)

    response = _build_response()
    _replace_application_snapshot(request)
    return response


__all__ = ["router"]
