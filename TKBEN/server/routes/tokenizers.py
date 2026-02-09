from __future__ import annotations

import asyncio
import os
import tempfile

import anyio
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from tokenizers import Tokenizer

from TKBEN.server.entities.tokenizers import (
    TokenizerScanResponse,
    TokenizerSettingsResponse,
    TokenizerUploadResponse,
)
from TKBEN.server.configurations.server import server_settings
from TKBEN.server.utils.constants import (
    API_ROUTE_TOKENIZERS_CUSTOM,
    API_ROUTE_TOKENIZERS_SCAN,
    API_ROUTE_TOKENIZERS_SETTINGS,
    API_ROUTE_TOKENIZERS_UPLOAD,
    API_ROUTER_PREFIX_TOKENIZERS,
)
from TKBEN.server.utils.logger import logger
from TKBEN.server.services.benchmarks import BenchmarkTools
from TKBEN.server.services.tokenizers import TokenizersService

router = APIRouter(prefix=API_ROUTER_PREFIX_TOKENIZERS, tags=["tokenizers"])


###############################################################################
@router.get(
    API_ROUTE_TOKENIZERS_SETTINGS,
    response_model=TokenizerSettingsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_tokenizer_settings() -> TokenizerSettingsResponse:
    """
    Get tokenizer configuration settings.

    Returns:
        TokenizerSettingsResponse with default, min, and max scan limits.
    """
    return TokenizerSettingsResponse(
        default_scan_limit=server_settings.tokenizers.default_scan_limit,
        max_scan_limit=server_settings.tokenizers.max_scan_limit,
        min_scan_limit=server_settings.tokenizers.min_scan_limit,
    )


###############################################################################
@router.get(
    API_ROUTE_TOKENIZERS_SCAN,
    response_model=TokenizerScanResponse,
    status_code=status.HTTP_200_OK,
)
async def scan_tokenizers(
    limit: int = Query(default=None),
    hf_access_token: str | None = Query(default=None),
) -> TokenizerScanResponse:
    """
    Scan HuggingFace for the most popular tokenizer identifiers.

    Args:
        limit: Maximum number of tokenizers to fetch. Defaults to configured value.
        hf_access_token: Optional HuggingFace access token for authenticated requests.

    Returns:
        TokenizerScanResponse containing the list of tokenizer identifiers.
    """
    # Use configured defaults if limit not provided, and clamp to configured bounds
    min_limit = server_settings.tokenizers.min_scan_limit
    max_limit = server_settings.tokenizers.max_scan_limit
    default_limit = server_settings.tokenizers.default_scan_limit

    if limit is None:
        limit = default_limit
    else:
        limit = max(min_limit, min(limit, max_limit))

    logger.info("Scanning HuggingFace for tokenizers (limit=%s)", limit)

    service = TokenizersService(hf_access_token)

    try:
        identifiers = await asyncio.to_thread(service.get_tokenizer_identifiers, limit)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to scan tokenizers from HuggingFace")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tokenizers from HuggingFace.",
        ) from exc

    logger.info("Successfully fetched %s tokenizer identifiers", len(identifiers))
    return TokenizerScanResponse(
        status="success",
        identifiers=identifiers,
        count=len(identifiers),
    )


###############################################################################
# Store uploaded custom tokenizers in memory for the session
custom_tokenizers: dict[str, Tokenizer] = {}


def get_custom_tokenizers() -> dict[str, Tokenizer]:
    """Get the currently loaded custom tokenizers."""
    return custom_tokenizers


def clear_custom_tokenizers() -> None:
    """Clear all custom tokenizers."""
    custom_tokenizers.clear()


def _create_temp_tokenizer_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    return path


def _unlink_temp_tokenizer_path(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


###############################################################################
@router.post(
    API_ROUTE_TOKENIZERS_UPLOAD,
    response_model=TokenizerUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_custom_tokenizer(
    file: UploadFile = File(...),
) -> TokenizerUploadResponse:
    """
    Upload a custom tokenizer.json file.

    The tokenizer will be loaded and validated for compatibility.
    If valid, it will be stored and can be used in benchmark runs.

    Args:
        file: The tokenizer.json file to upload.

    Returns:
        TokenizerUploadResponse with the assigned tokenizer name.
    """
    # Validate file extension
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a .json file (tokenizer.json)",
        )

    # Read file content
    content = await file.read()

    # Save to temp file and load
    tmp_path = None
    try:
        tmp_path = await anyio.to_thread.run_sync(_create_temp_tokenizer_path)
        async with await anyio.open_file(tmp_path, "wb") as tmp:
            await tmp.write(content)

        tokenizer = await anyio.to_thread.run_sync(Tokenizer.from_file, tmp_path)
    except Exception as exc:
        logger.warning("Failed to load tokenizer from uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load tokenizer: {exc}",
        ) from exc
    finally:
        if tmp_path:
            await anyio.to_thread.run_sync(_unlink_temp_tokenizer_path, tmp_path)

    # Check compatibility
    tools = BenchmarkTools()
    is_compatible = tools.is_tokenizer_compatible(tokenizer)

    # Generate name from filename
    base_name = os.path.splitext(file.filename)[0]
    tokenizer_name = f"CUSTOM_{base_name}"

    if is_compatible:
        custom_tokenizers[tokenizer_name] = tokenizer
        logger.info("Loaded custom tokenizer: %s", tokenizer_name)
    else:
        logger.warning("Custom tokenizer %s is not compatible", tokenizer_name)

    return TokenizerUploadResponse(
        status="success",
        tokenizer_name=tokenizer_name,
        is_compatible=is_compatible,
    )


###############################################################################
@router.delete(
    API_ROUTE_TOKENIZERS_CUSTOM,
    status_code=status.HTTP_200_OK,
)
async def delete_custom_tokenizers() -> dict:
    """
    Clear all uploaded custom tokenizers.

    Returns:
        Status message.
    """
    clear_custom_tokenizers()
    return {"status": "success", "message": "Custom tokenizers cleared"}
