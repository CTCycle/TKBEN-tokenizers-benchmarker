# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

import pytest
from playwright.sync_api import APIRequestContext

###############################################################################
def _custom_dataset_names(api_context: APIRequestContext) -> set[str]:
    response = api_context.get("/api/datasets/list")
    if not response.ok:
        return set()
    return {
        item["dataset_name"]
        for item in response.json().get("datasets", [])
        if isinstance(item, dict)
        and str(item.get("dataset_name", "")).startswith("custom/")
    }

###############################################################################
def _custom_tokenizer_names(api_context: APIRequestContext) -> set[str]:
    response = api_context.get("/api/tokenizers/list")
    if not response.ok:
        return set()
    return {
        item["tokenizer_name"]
        for item in response.json().get("tokenizers", [])
        if isinstance(item, dict) and item.get("source") == "custom"
    }

###############################################################################
@pytest.fixture(scope="session", autouse=True)
def remove_artifacts_created_by_session(api_context: APIRequestContext):
    """Delete custom datasets and tokenizers that E2E tests leave behind."""
    datasets = _custom_dataset_names(api_context)
    tokenizers = _custom_tokenizer_names(api_context)
    yield
    for dataset_name in _custom_dataset_names(api_context) - datasets:
        api_context.delete(
            "/api/datasets/delete", params={"dataset_name": dataset_name}
        )
    for tokenizer_name in _custom_tokenizer_names(api_context) - tokenizers:
        api_context.delete(
            "/api/tokenizers/delete", params={"tokenizer_name": tokenizer_name}
        )
