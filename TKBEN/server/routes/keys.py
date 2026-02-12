from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from TKBEN.server.common.constants import (
    API_ROUTE_KEYS_ACTIVATE,
    API_ROUTE_KEYS_CREATE,
    API_ROUTE_KEYS_DEACTIVATE,
    API_ROUTE_KEYS_DELETE,
    API_ROUTE_KEYS_LIST,
    API_ROUTE_KEYS_REVEAL,
    API_ROUTER_PREFIX_KEYS,
)
from TKBEN.server.entities.keys import (
    HFAccessKeyActivateResponse,
    HFAccessKeyCreateRequest,
    HFAccessKeyDeleteResponse,
    HFAccessKeyListItem,
    HFAccessKeyListResponse,
    HFAccessKeyRevealResponse,
)
from TKBEN.server.services.keys import (
    HFAccessKeyConflictError,
    HFAccessKeyNotFoundError,
    HFAccessKeyService,
    HFAccessKeyValidationError,
)


router = APIRouter(prefix=API_ROUTER_PREFIX_KEYS, tags=["keys"])


###############################################################################
@router.post(
    API_ROUTE_KEYS_CREATE,
    response_model=HFAccessKeyListItem,
    status_code=status.HTTP_201_CREATED,
)
async def create_key(request: HFAccessKeyCreateRequest) -> HFAccessKeyListItem:
    service = HFAccessKeyService()
    try:
        created_key = service.add_key(request.key_value)
    except HFAccessKeyValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except HFAccessKeyConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return HFAccessKeyListItem(**created_key)


###############################################################################
@router.get(
    API_ROUTE_KEYS_LIST,
    response_model=HFAccessKeyListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_keys() -> HFAccessKeyListResponse:
    service = HFAccessKeyService()
    return HFAccessKeyListResponse(keys=[HFAccessKeyListItem(**key) for key in service.list_keys()])


###############################################################################
@router.delete(
    API_ROUTE_KEYS_DELETE,
    response_model=HFAccessKeyDeleteResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_key(
    key_id: int,
    confirm: bool = Query(default=False),
) -> HFAccessKeyDeleteResponse:
    service = HFAccessKeyService()
    try:
        service.delete_key(key_id=key_id, confirm=confirm)
    except HFAccessKeyValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except HFAccessKeyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return HFAccessKeyDeleteResponse()


###############################################################################
@router.post(
    API_ROUTE_KEYS_ACTIVATE,
    response_model=HFAccessKeyActivateResponse,
    status_code=status.HTTP_200_OK,
)
async def activate_key(key_id: int) -> HFAccessKeyActivateResponse:
    service = HFAccessKeyService()
    try:
        service.set_active_key(key_id)
    except HFAccessKeyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return HFAccessKeyActivateResponse()


###############################################################################
@router.post(
    API_ROUTE_KEYS_DEACTIVATE,
    response_model=HFAccessKeyActivateResponse,
    status_code=status.HTTP_200_OK,
)
async def deactivate_key(key_id: int) -> HFAccessKeyActivateResponse:
    service = HFAccessKeyService()
    try:
        service.clear_active_key(key_id)
    except HFAccessKeyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return HFAccessKeyActivateResponse(message="Active key cleared.")


###############################################################################
@router.post(
    API_ROUTE_KEYS_REVEAL,
    response_model=HFAccessKeyRevealResponse,
    status_code=status.HTTP_200_OK,
)
async def reveal_key(key_id: int) -> HFAccessKeyRevealResponse:
    service = HFAccessKeyService()
    try:
        revealed_key = service.get_revealed_key(key_id)
    except HFAccessKeyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return HFAccessKeyRevealResponse(id=key_id, key_value=revealed_key)
