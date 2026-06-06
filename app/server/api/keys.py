from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, HTTPException, Query, status

from server.common.constants import (
    ALLOW_KEY_REVEAL_DEFAULT,
    API_ROUTE_KEYS_ACTIVATE,
    API_ROUTE_KEYS_CREATE,
    API_ROUTE_KEYS_DEACTIVATE,
    API_ROUTE_KEYS_DELETE,
    API_ROUTE_KEYS_LIST,
    API_ROUTE_KEYS_REVEAL,
    API_ROUTER_PREFIX_KEYS,
)
from server.common.utils.types import coerce_bool
from server.domain.keys import (
    HFAccessKeyActivateResponse,
    HFAccessKeyCreateRequest,
    HFAccessKeyDeleteResponse,
    HFAccessKeyListItem,
    HFAccessKeyListResponse,
    HFAccessKeyRevealResponse,
)
from server.services.keys import (
    HFAccessKeyConflictError,
    HFAccessKeyNotFoundError,
    HFAccessKeyService,
    HFAccessKeyValidationError,
)


router = APIRouter(prefix=API_ROUTER_PREFIX_KEYS, tags=["keys"])


###############################################################################
def is_key_reveal_enabled() -> bool:
    return coerce_bool(
        os.getenv("ALLOW_KEY_REVEAL"),
        ALLOW_KEY_REVEAL_DEFAULT,
    )


###############################################################################
@router.post(
    API_ROUTE_KEYS_CREATE,
    response_model=HFAccessKeyListItem,
    status_code=status.HTTP_201_CREATED,
)
async def create_key(request: HFAccessKeyCreateRequest) -> HFAccessKeyListItem:
    service = HFAccessKeyService()
    try:
        created_key = await asyncio.to_thread(service.add_key, request.key_value)
    except HFAccessKeyValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except HFAccessKeyConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    return HFAccessKeyListItem(**created_key)


###############################################################################
@router.get(
    API_ROUTE_KEYS_LIST,
    response_model=HFAccessKeyListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_keys() -> HFAccessKeyListResponse:
    service = HFAccessKeyService()
    keys = await asyncio.to_thread(service.list_keys)
    return HFAccessKeyListResponse(keys=[HFAccessKeyListItem(**key) for key in keys])


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
        await asyncio.to_thread(service.delete_key, key_id=key_id, confirm=confirm)

    except HFAccessKeyValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except HFAccessKeyNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
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
        await asyncio.to_thread(service.set_active_key, key_id)
    except HFAccessKeyNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
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
        await asyncio.to_thread(service.clear_active_key, key_id)
    except HFAccessKeyNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return HFAccessKeyActivateResponse(message="Active key cleared.")


###############################################################################
@router.post(
    API_ROUTE_KEYS_REVEAL,
    response_model=HFAccessKeyRevealResponse,
    status_code=status.HTTP_200_OK,
)
async def reveal_key(key_id: int) -> HFAccessKeyRevealResponse:
    if not is_key_reveal_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Key reveal is disabled by server policy.",
        )
    service = HFAccessKeyService()
    try:
        revealed_key = await asyncio.to_thread(service.get_revealed_key, key_id)

    except HFAccessKeyValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except HFAccessKeyNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return HFAccessKeyRevealResponse(id=key_id, key_value=revealed_key)
