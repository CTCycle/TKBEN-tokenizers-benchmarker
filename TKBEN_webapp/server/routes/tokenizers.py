from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query, status

from TKBEN_webapp.server.schemas.tokenizers import (
    TokenizerScanResponse,
    TokenizerSettingsResponse,
)
from TKBEN_webapp.server.utils.configurations.server import server_settings
from TKBEN_webapp.server.utils.logger import logger
from TKBEN_webapp.server.utils.services.tokenizers import TokenizersService

router = APIRouter(prefix="/tokenizers", tags=["tokenizers"])


###############################################################################
@router.get(
    "/settings",
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
    "/scan", response_model=TokenizerScanResponse, status_code=status.HTTP_200_OK
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

