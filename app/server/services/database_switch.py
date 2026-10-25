# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

from sqlalchemy import text

from server.configurations.database_configuration import StoredDatabaseConfiguration
from server.configurations.settings import ServerSettings, build_server_settings
from server.configurations.startup import (
    activate_runtime_settings_store,
    build_database_runtime_settings_store,
    get_database_configuration_store,
    get_runtime_settings_store,
)
from server.contracts.database import DatabaseConfigurationRequest
from server.repositories.database.backend import TKBENDatabase, get_database
from server.repositories.database.initializer import initialize_database
from server.repositories.jobs import JobRepository
from server.repositories.runtime_settings import DatabaseRuntimeSettingsPersistence
from server.services.jobs import JobManager


###############################################################################
class DatabaseSwitchGate:
    """Drain ordinary API calls while a validated database is activated."""

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._switch_lock = asyncio.Lock()
        self._active_requests = 0
        self._switching = False

    # -------------------------------------------------------------------------
    @asynccontextmanager
    async def request(self) -> AsyncIterator[None]:
        async with self._condition:
            await self._condition.wait_for(lambda: not self._switching)
            self._active_requests += 1
        try:
            yield
        finally:
            async with self._condition:
                self._active_requests -= 1
                if self._active_requests == 0:
                    self._condition.notify_all()

    # -------------------------------------------------------------------------
    @asynccontextmanager
    async def switch(self) -> AsyncIterator[None]:
        async with self._switch_lock:
            async with self._condition:
                self._switching = True
                await self._condition.wait_for(lambda: self._active_requests == 0)
            try:
                yield
            finally:
                async with self._condition:
                    self._switching = False
                    self._condition.notify_all()


###############################################################################
class DatabaseSwitchApplicationState(Protocol):
    database_switch_gate: DatabaseSwitchGate
    database_available: bool
    database_startup_error: str | None
    job_manager: JobManager
    job_repository: JobRepository | None
    settings: ServerSettings


###############################################################################
class DatabaseSwitchBusyError(RuntimeError):
    """Raised when a database switch would interrupt a managed job."""


###############################################################################
class DatabaseSwitchActivationError(RuntimeError):
    """Raised when a validated database cannot be activated safely."""


###############################################################################
def _test_engine(database: TKBENDatabase) -> None:
    with database.backend.engine.connect() as connection:
        if connection.execute(text("SELECT 1")).scalar_one() != 1:
            raise RuntimeError("Database health query returned an unexpected value.")


###############################################################################
async def read_active_database_configuration() -> tuple[
    StoredDatabaseConfiguration, bool
]:
    configuration = get_database_configuration_store().load()
    try:
        await asyncio.to_thread(_test_engine, get_database())
    except Exception:
        return configuration, False
    return configuration, True


###############################################################################
async def test_database_configuration(
    payload: DatabaseConfigurationRequest,
) -> tuple[StoredDatabaseConfiguration, bool]:
    configuration_store = get_database_configuration_store()
    candidate = configuration_store.build_candidate(**payload.model_dump())
    candidate_settings = configuration_store.to_database_settings(candidate)
    candidate_database = TKBENDatabase(candidate_settings)
    try:
        await asyncio.to_thread(_test_engine, candidate_database)
    except Exception:
        candidate_database.backend.engine.dispose()
        return candidate, False
    candidate_database.backend.engine.dispose()
    return candidate, True


###############################################################################
async def activate_database_configuration(
    state: DatabaseSwitchApplicationState,
    payload: DatabaseConfigurationRequest,
) -> StoredDatabaseConfiguration:
    configuration_store = get_database_configuration_store()

    async with state.database_switch_gate.switch():
        if state.job_manager.is_job_running():
            raise DatabaseSwitchBusyError(
                "Wait for pending and running jobs to finish before switching databases."
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
        previous_app_settings = state.settings
        previous_job_repository = state.job_repository
        previous_job_manager = state.job_manager

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
                max_stored_jobs=(
                    candidate_runtime_store.get_state().settings.jobs.max_stored_jobs
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
            state.settings = active_state.settings
            state.job_repository = candidate_job_repository
            state.job_manager = candidate_job_manager
            state.database_available = True
            state.database_startup_error = None
        except Exception as exc:
            if database_activated and previous_database is not None:
                get_database().restore(previous_database)
            if config_saved:
                configuration_store.save(previous_configuration)
            activate_runtime_settings_store(previous_runtime_store)
            state.settings = previous_app_settings
            state.job_repository = previous_job_repository
            state.job_manager = previous_job_manager
            if candidate_database is not None:
                candidate_database.backend.engine.dispose()
            raise DatabaseSwitchActivationError(
                "The database could not be activated. The previous database "
                "remains active. Check connectivity and schema permissions."
            ) from exc

        if previous_database is not None:
            try:
                previous_database[1].engine.dispose()
            except Exception:
                pass
        return candidate_configuration
