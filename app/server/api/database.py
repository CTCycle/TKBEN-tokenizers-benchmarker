# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import Literal, cast

from fastapi import APIRouter, HTTPException, Request, status

from server.configurations.database_configuration import StoredDatabaseConfiguration
from server.contracts.database import (
    DatabaseConfigurationRequest,
    DatabaseConfigurationResponse,
    DatabaseConnectionTestResponse,
)
from server.services.database_switch import (
    DatabaseSwitchActivationError,
    DatabaseSwitchApplicationState,
    DatabaseSwitchBusyError,
    activate_database_configuration,
    read_active_database_configuration,
    test_database_configuration,
)

router = APIRouter(prefix="/database", tags=["database"])


###############################################################################
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


###############################################################################
@router.get("", response_model=DatabaseConfigurationResponse)
async def get_database_configuration() -> DatabaseConfigurationResponse:
    configuration, connected = await read_active_database_configuration()
    if not connected:
        return _configuration_response(
            configuration,
            connection_status="unavailable",
            message="The active database is unavailable. Check the connection settings.",
        )
    return _configuration_response(configuration, connection_status="connected")


###############################################################################
@router.post("/test", response_model=DatabaseConnectionTestResponse)
async def test_database_connection(
    payload: DatabaseConfigurationRequest,
) -> DatabaseConnectionTestResponse:
    candidate, connected = await test_database_configuration(payload)
    return DatabaseConnectionTestResponse(
        backend=candidate.backend,
        success=connected,
        message=(
            "Connection successful."
            if connected
            else "Unable to connect with these settings. Verify the host, database, "
            "credentials, and SSL options."
        ),
    )


###############################################################################
@router.post("/switch", response_model=DatabaseConfigurationResponse)
async def switch_database(
    request: Request,
    payload: DatabaseConfigurationRequest,
) -> DatabaseConfigurationResponse:
    try:
        candidate_configuration = await activate_database_configuration(
            cast(DatabaseSwitchApplicationState, request.app.state),
            payload,
        )
    except DatabaseSwitchBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except DatabaseSwitchActivationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return _configuration_response(
        candidate_configuration,
        connection_status="connected",
        message="Database switched. Settings and records now come from the selected database.",
    )
