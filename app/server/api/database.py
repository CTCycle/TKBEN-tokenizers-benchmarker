# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import text

from server.configurations.database_configuration import StoredDatabaseConfiguration
from server.configurations.settings import build_server_settings
from server.configurations.startup import (
    activate_runtime_settings_store,
    build_database_runtime_settings_store,
    get_database_configuration_store,
    get_runtime_settings_store,
)
from server.contracts.database import (
    DatabaseConfigurationRequest,
    DatabaseConfigurationResponse,
    DatabaseConnectionTestResponse,
)
from server.repositories.database.backend import TKBENDatabase, get_database
from server.repositories.database.initializer import initialize_database
from server.repositories.jobs import JobRepository
from server.repositories.runtime_settings import DatabaseRuntimeSettingsPersistence
from server.services.jobs import JobManager

router = APIRouter(prefix="/database", tags=["database"])


def _configuration_response(
    configuration: StoredDatabaseConfiguration,
    *,
    connection_status: Literal["connected", "unavailable"],
    message: str | None = None,
) -> DatabaseConfigurationResponse:
    return DatabaseConfigurationResponse(
        backend=configuration.backend,
        active=True,
        connection_status=connection_status,  # type: ignore[arg-type]
        sqlite_path=configuration.sqlite_path,
        host=configuration.host,
        port=configuration.port,
        database_name=configuration.database_name,
        username=configuration.username,
        password_configured=bool(configuration.encrypted_password),
        ssl=configuration.ssl,
        ssl_ca=configuration.ssl_ca,
        connect_timeout=configuration.connect_timeout,
        insert_batch_size=configuration.insert_batch_size,
        message=message,
    )


def _test_engine(database: TKBENDatabase) -> None:
    with database.backend.engine.connect() as connection:
        if connection.execute(text("SELECT 1")).scalar_one() != 1:
            raise RuntimeError("Database health query returned an unexpected value.")


@router.get("", response_model=DatabaseConfigurationResponse)
async def get_database_configuration() -> DatabaseConfigurationResponse:
    configuration_store = get_database_configuration_store()
    configuration = configuration_store.load()
    try:
        await asyncio.to_thread(_test_engine, get_database())
    except Exception:
        return _configuration_response(
            configuration,
            connection_status="unavailable",
            message="The active database is unavailable. Check the connection settings.",
        )
    return _configuration_response(configuration, connection_status="connected")


@router.post("/test", response_model=DatabaseConnectionTestResponse)
async def test_database_connection(
    payload: DatabaseConfigurationRequest,
) -> DatabaseConnectionTestResponse:
    configuration_store = get_database_configuration_store()
    candidate = configuration_store.build_candidate(**payload.model_dump())
    candidate_settings = configuration_store.to_database_settings(candidate)
    candidate_database = TKBENDatabase(candidate_settings)
    try:
        await asyncio.to_thread(_test_engine, candidate_database)
    except Exception:
        candidate_database.backend.engine.dispose()
        return DatabaseConnectionTestResponse(
            backend=candidate.backend,
            success=False,
            message=(
                "Unable to connect with these settings. Verify the host, database, "
                "credentials, and SSL options."
            ),
        )
    candidate_database.backend.engine.dispose()
    return DatabaseConnectionTestResponse(
        backend=candidate.backend,
        success=True,
        message="Connection successful.",
    )


@router.post("/switch", response_model=DatabaseConfigurationResponse)
async def switch_database(
    request: Request,
    payload: DatabaseConfigurationRequest,
) -> DatabaseConfigurationResponse:
    gate = request.app.state.database_switch_gate
    configuration_store = get_database_configuration_store()

    async with gate.switch():
        job_manager: JobManager = request.app.state.job_manager
        if job_manager.is_job_running():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Wait for pending and running jobs to finish before switching databases.",
            )

        previous_configuration = configuration_store.load()
        candidate_configuration = configuration_store.build_candidate(
            **payload.model_dump()
        )
        candidate_database: TKBENDatabase | None = None
        config_saved = False
        database_activated = False
        previous_database = None
        previous_runtime_store = get_runtime_settings_store()
        previous_app_settings = request.app.state.settings
        previous_job_repository = request.app.state.job_repository
        previous_job_manager = request.app.state.job_manager

        try:
            candidate_settings = build_server_settings(
                database_settings=configuration_store.to_database_settings(
                    candidate_configuration
                )
            )
            await asyncio.to_thread(
                initialize_database,
                settings=candidate_settings,
                startup=False,
            )
            candidate_database = TKBENDatabase(candidate_settings)
            await asyncio.to_thread(_test_engine, candidate_database)

            candidate_runtime_store = build_database_runtime_settings_store(
                candidate_settings,
                DatabaseRuntimeSettingsPersistence(candidate_database),
            )
            candidate_job_repository = JobRepository(candidate_database)
            candidate_job_manager = JobManager(
                terminal_retention_seconds=(
                    candidate_runtime_store.get_state().settings.jobs.terminal_retention_seconds
                ),
                repository=candidate_job_repository,
            )
            await asyncio.to_thread(candidate_job_manager.restore_persisted_jobs)

            configuration_store.save(candidate_configuration)
            config_saved = True
            shared_database = get_database()
            previous_database = shared_database.activate(candidate_database)
            database_activated = True
            active_state = activate_runtime_settings_store(candidate_runtime_store)
            request.app.state.settings = active_state.settings
            request.app.state.job_repository = candidate_job_repository
            request.app.state.job_manager = candidate_job_manager
            request.app.state.database_available = True
            request.app.state.database_startup_error = None
        except HTTPException:
            raise
        except Exception as exc:
            if database_activated and previous_database is not None:
                get_database().restore(previous_database)
            if config_saved:
                configuration_store.save(previous_configuration)
            activate_runtime_settings_store(previous_runtime_store)
            request.app.state.settings = previous_app_settings
            request.app.state.job_repository = previous_job_repository
            request.app.state.job_manager = previous_job_manager
            if candidate_database is not None:
                candidate_database.backend.engine.dispose()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "The database could not be activated. The previous database "
                    "remains active. Check connectivity and schema permissions."
                ),
            ) from exc

        if previous_database is not None:
            try:
                previous_database[1].engine.dispose()
            except Exception:
                pass
        return _configuration_response(
            candidate_configuration,
            connection_status="connected",
            message="Database switched. Settings and records now come from the selected database.",
        )
