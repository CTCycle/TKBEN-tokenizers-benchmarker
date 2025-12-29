"""
E2E tests for tokenizer API endpoints.
Covers /tokenizers/settings, /tokenizers/scan, /tokenizers/upload, /tokenizers/custom.
"""
import json
import os

import pytest
from playwright.sync_api import APIRequestContext


RUN_HF_SCAN = os.getenv("E2E_RUN_HF_SCAN", "").lower() in ("1", "true", "yes")


def _build_wordlevel_tokenizer_json() -> bytes:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"[UNK]": 0, "hello": 1, "world": 2}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    if hasattr(tokenizer, "to_str"):
        payload = tokenizer.to_str()
    else:
        payload = tokenizer.to_json()

    if isinstance(payload, dict):
        payload = json.dumps(payload)
    return str(payload).encode("utf-8")


def test_get_tokenizer_settings(api_context: APIRequestContext) -> None:
    """GET /tokenizers/settings should return configured scan limits."""
    response = api_context.get("/tokenizers/settings")
    assert response.ok
    data = response.json()
    assert "default_scan_limit" in data
    assert "max_scan_limit" in data
    assert "min_scan_limit" in data
    assert data["min_scan_limit"] <= data["default_scan_limit"] <= data["max_scan_limit"]


@pytest.mark.skipif(not RUN_HF_SCAN, reason="Set E2E_RUN_HF_SCAN=1 to enable.")
def test_scan_tokenizers_returns_identifiers(
    api_context: APIRequestContext,
) -> None:
    """GET /tokenizers/scan should return at least one tokenizer identifier."""
    response = api_context.get("/tokenizers/scan?limit=1")
    assert response.ok
    data = response.json()
    assert data.get("status") == "success"
    assert isinstance(data.get("identifiers"), list)
    assert data.get("count", 0) >= 1


def test_upload_rejects_invalid_extension(api_context: APIRequestContext) -> None:
    """POST /tokenizers/upload should reject non-json files."""
    response = api_context.post(
        "/tokenizers/upload",
        multipart={
            "file": {
                "name": "tokenizer.txt",
                "mimeType": "text/plain",
                "buffer": b"not json",
            }
        },
    )
    assert response.status == 400
    data = response.json()
    assert "File must be a .json file" in data.get("detail", "")


def test_upload_rejects_invalid_json(api_context: APIRequestContext) -> None:
    """POST /tokenizers/upload should reject invalid tokenizer JSON."""
    response = api_context.post(
        "/tokenizers/upload",
        multipart={
            "file": {
                "name": "tokenizer.json",
                "mimeType": "application/json",
                "buffer": b"{not valid json}",
            }
        },
    )
    assert response.status == 400
    data = response.json()
    assert "Failed to load tokenizer" in data.get("detail", "")


def test_upload_accepts_valid_tokenizer_json(api_context: APIRequestContext) -> None:
    """POST /tokenizers/upload should accept a valid tokenizer.json file."""
    payload = _build_wordlevel_tokenizer_json()
    response = api_context.post(
        "/tokenizers/upload",
        multipart={
            "file": {
                "name": "tokenizer.json",
                "mimeType": "application/json",
                "buffer": payload,
            }
        },
    )
    assert response.ok
    data = response.json()
    assert data.get("status") == "success"
    assert data.get("tokenizer_name", "").startswith("CUSTOM_")
    assert data.get("is_compatible") is True

    cleanup = api_context.delete("/tokenizers/custom")
    assert cleanup.ok


def test_clear_custom_tokenizers(api_context: APIRequestContext) -> None:
    """DELETE /tokenizers/custom should return a success message."""
    response = api_context.delete("/tokenizers/custom")
    assert response.ok
    data = response.json()
    assert data.get("status") == "success"
