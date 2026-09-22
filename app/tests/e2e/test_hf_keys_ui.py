"""Rendered coverage for Settings key management and SQLite protection."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from playwright.sync_api import APIRequestContext, Locator, Page, expect

from server.configurations.settings import build_server_settings

TEST_KEY_ENV = "TKBEN_TEST_HF_KEY"


def _read_keys(api_context: APIRequestContext) -> list[dict[str, Any]]:
    response = api_context.get("/api/keys")
    if response.status != 200:
        pytest.fail(f"GET /api/keys returned HTTP {response.status}.")
    payload = response.json()
    keys = payload.get("keys")
    if not isinstance(keys, list):
        pytest.fail("GET /api/keys did not return a key list.")
    return keys


def _assert_public_list(
    api_context: APIRequestContext, plaintext_values: tuple[str, ...]
) -> list[dict[str, Any]]:
    keys = _read_keys(api_context)
    serialized = json.dumps(keys, sort_keys=True)
    if any(value and value in serialized for value in plaintext_values):
        pytest.fail("The key list response exposed a plaintext key.")
    if any("key_value" in item for item in keys):
        pytest.fail("The key list response included a key_value field.")
    return keys


def _key_row(panel: Locator, keys: list[dict[str, Any]], key_id: int) -> Locator:
    try:
        index = next(
            index for index, item in enumerate(keys) if item.get("id") == key_id
        )
    except StopIteration:
        pytest.fail("The expected key is missing from the API list.")
    rows = panel.locator(".key-manager-row")
    expect(rows).to_have_count(len(keys))
    return rows.nth(index)


def _expect_key_response(response: Any, method: str, path_suffix: str) -> bool:
    return response.request.method == method and urlparse(response.url).path.endswith(
        path_suffix
    )


def _add_key(page: Page, key_value: str) -> dict[str, Any]:
    key_input = page.get_by_role("textbox", name="Hugging Face key")
    key_input.fill(key_value)
    with page.expect_response(
        lambda response: _expect_key_response(response, "POST", "/api/keys")
    ) as response_info:
        page.get_by_role("button", name="Add", exact=True).click()
    response = response_info.value
    if response.status != 201:
        pytest.fail(f"Adding a key returned HTTP {response.status}.")
    expect(key_input).to_have_value("")
    item = response.json()
    if "key_value" in item or key_value in json.dumps(item, sort_keys=True):
        pytest.fail("The create response exposed a plaintext key.")
    if not isinstance(item.get("id"), int):
        pytest.fail("The create response did not include a key ID.")
    return item


def _sqlite_paths() -> tuple[Path, Path, Path]:
    settings = build_server_settings()
    if not settings.database.embedded_database:
        pytest.fail(
            "Direct key ciphertext inspection requires the embedded SQLite database."
        )
    return (
        settings.database.sqlite_path,
        settings.security.hf_keys_encryption_material_file,
        settings.paths.logs,
    )


def _assert_ciphertext(key_id: int, plaintext: str) -> None:
    database_path, material_path, _ = _sqlite_paths()
    if not database_path.is_file():
        pytest.fail("The configured SQLite database file is missing.")
    if not material_path.is_file():
        pytest.fail("The separate key-encryption material file is missing.")
    if material_path.resolve() == database_path.resolve():
        pytest.fail("Key-encryption material is not stored separately from SQLite.")
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT key_value FROM hf_access_keys WHERE id = ?", (key_id,)
        ).fetchone()
    if row is None or not isinstance(row[0], str) or not row[0]:
        pytest.fail("The key row has no stored value.")
    stored_value = row[0]
    if stored_value == plaintext or plaintext in stored_value:
        pytest.fail("The database contains the plaintext key instead of ciphertext.")


def _assert_active_count(expected: int) -> None:
    database_path, _, _ = _sqlite_paths()
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM hf_access_keys WHERE is_active = 1"
        ).fetchone()
    if row is None or row[0] != expected:
        pytest.fail("The SQLite active-key count did not match the expected state.")


def _assert_database_key_state(
    expected_ids: set[int], expected_active_ids: set[int]
) -> None:
    database_path, _, _ = _sqlite_paths()
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("SELECT id, is_active FROM hf_access_keys").fetchall()
    if {row[0] for row in rows} != expected_ids:
        pytest.fail("The SQLite key IDs did not return to their pre-test state.")
    if {row[0] for row in rows if row[1]} != expected_active_ids:
        pytest.fail("The SQLite active-key state did not return to its pre-test state.")


def _assert_plaintext_absent_from_logs(plaintext_values: tuple[str, ...]) -> None:
    _, _, log_directory = _sqlite_paths()
    if not log_directory.is_dir():
        return
    for log_path in log_directory.rglob("*.log"):
        try:
            with log_path.open("r", encoding="utf-8", errors="ignore") as stream:
                if any(value in line for line in stream for value in plaintext_values):
                    pytest.fail("A plaintext test key appeared in an application log.")
        except OSError:
            pytest.fail("An application log could not be checked for plaintext keys.")


def _cleanup_created_keys(
    api_context: APIRequestContext,
    created_ids: set[int],
    original_active_ids: set[int],
) -> None:
    current = {item["id"]: item for item in _read_keys(api_context)}
    for key_id in created_ids:
        item = current.get(key_id)
        if item is None:
            continue
        if item.get("is_active"):
            response = api_context.post(f"/api/keys/{key_id}/deactivate")
            if response.status != 200:
                pytest.fail("Cleanup could not deactivate a validation-created key.")
        response = api_context.delete(f"/api/keys/{key_id}?confirm=true")
        if response.status != 200:
            pytest.fail("Cleanup could not delete a validation-created key.")
    if original_active_ids:
        original_id = next(iter(original_active_ids))
        response = api_context.post(f"/api/keys/{original_id}/activate")
        if response.status != 200:
            pytest.fail("Cleanup could not restore the original active key.")


def _settings_keys_panel(page: Page, base_url: str) -> Locator:
    page.goto(f"{base_url}/settings")
    tab = page.get_by_role("tab", name="Keys")
    expect(tab).to_be_visible()
    tab.click()
    panel = page.get_by_role("tabpanel", name="Keys")
    expect(panel).to_be_visible()
    return panel


def test_settings_keys_rendered_lifecycle_and_sqlite_ciphertext(
    page: Page, base_url: str, api_context: APIRequestContext
) -> None:
    baseline = _read_keys(api_context)
    baseline_ids = {item["id"] for item in baseline}
    original_active_ids = {item["id"] for item in baseline if item.get("is_active")}
    if len(original_active_ids) > 1:
        pytest.skip(
            "The existing key list has multiple active rows; no mutation was made."
        )

    created_ids: set[int] = set()
    suffix = uuid4().hex
    key_a = f"hf_tkben_e2e_{suffix}"
    key_b = f"{key_a}_switch"
    test_values = (key_a, key_b)
    try:
        panel = _settings_keys_panel(page, base_url)
        rows = panel.locator(".key-manager-row")
        if baseline:
            expect(rows).to_have_count(len(baseline))
            for item in baseline:
                expect(
                    panel.get_by_text(item["masked_preview"], exact=True)
                ).to_be_visible()
        else:
            expect(panel.get_by_text("No keys stored.", exact=True)).to_be_visible()

        first = _add_key(page, key_a)
        first_id = first["id"]
        created_ids.add(first_id)
        keys = _assert_public_list(api_context, test_values)
        first_row = _key_row(panel, keys, first_id)
        expect(first_row.locator(".key-manager-preview")).to_have_text(
            first["masked_preview"]
        )
        if key_a in panel.inner_text():
            pytest.fail("The rendered key list exposed a plaintext key.")
        _assert_ciphertext(first_id, key_a)

        key_input = page.get_by_role("textbox", name="Hugging Face key")
        key_input.fill(key_a)
        with page.expect_response(
            lambda response: _expect_key_response(response, "POST", "/api/keys")
        ) as duplicate_info:
            page.get_by_role("button", name="Add", exact=True).click()
        if duplicate_info.value.status != 409:
            pytest.fail(
                f"Duplicate insertion returned HTTP {duplicate_info.value.status}, not 409."
            )
        expect(page.get_by_role("alert")).to_contain_text("already stored")
        keys = _assert_public_list(api_context, test_values)
        if len(keys) != len(baseline) + 1:
            pytest.fail("Duplicate insertion changed the key count.")

        first_row = _key_row(panel, keys, first_id)
        with page.expect_response(
            lambda response: _expect_key_response(
                response, "POST", f"/api/keys/{first_id}/activate"
            )
        ):
            first_row.get_by_role("button", name="Activate key").click()
        expect(first_row.get_by_role("button", name="Deactivate key")).to_be_visible()

        keys = _assert_public_list(api_context, test_values)
        first_row = _key_row(panel, keys, first_id)
        with page.expect_response(
            lambda response: _expect_key_response(
                response, "POST", f"/api/keys/{first_id}/deactivate"
            )
        ):
            first_row.get_by_role("button", name="Deactivate key").click()
        expect(first_row.get_by_role("button", name="Activate key")).to_be_visible()

        keys = _assert_public_list(api_context, test_values)
        first_row = _key_row(panel, keys, first_id)
        with page.expect_response(
            lambda response: _expect_key_response(
                response, "POST", f"/api/keys/{first_id}/activate"
            )
        ):
            first_row.get_by_role("button", name="Activate key").click()
        expect(first_row.get_by_role("button", name="Deactivate key")).to_be_visible()

        second = _add_key(page, key_b)
        second_id = second["id"]
        created_ids.add(second_id)
        keys = _assert_public_list(api_context, test_values)
        second_row = _key_row(panel, keys, second_id)
        _assert_ciphertext(second_id, key_b)

        with page.expect_response(
            lambda response: _expect_key_response(
                response, "POST", f"/api/keys/{second_id}/activate"
            )
        ):
            second_row.get_by_role("button", name="Activate key").click()
        keys = _assert_public_list(api_context, test_values)
        active_ids = {item["id"] for item in keys if item.get("is_active")}
        if active_ids != {second_id}:
            pytest.fail(
                "Activating the second key did not leave exactly that key active."
            )
        _assert_active_count(1)
        first_row = _key_row(panel, keys, first_id)
        second_row = _key_row(panel, keys, second_id)
        expect(first_row.get_by_role("button", name="Activate key")).to_be_visible()
        expect(second_row.get_by_role("button", name="Deactivate key")).to_be_visible()

        page.once("dialog", lambda dialog: dialog.accept())
        with page.expect_response(
            lambda response: _expect_key_response(
                response, "DELETE", f"/api/keys/{second_id}"
            )
        ) as active_delete_info:
            second_row.get_by_role("button", name="Delete key").click()
        if active_delete_info.value.status != 400:
            pytest.fail(
                f"Active-key deletion returned HTTP {active_delete_info.value.status}, not 400."
            )
        expect(page.get_by_role("alert")).to_contain_text("active Hugging Face key")
        keys = _assert_public_list(api_context, test_values)
        second_row = _key_row(panel, keys, second_id)

        with page.expect_response(
            lambda response: _expect_key_response(
                response, "POST", f"/api/keys/{second_id}/deactivate"
            )
        ):
            second_row.get_by_role("button", name="Deactivate key").click()

        keys = _assert_public_list(api_context, test_values)
        first_row = _key_row(panel, keys, first_id)
        with page.expect_response(
            lambda response: _expect_key_response(
                response, "POST", f"/api/keys/{first_id}/activate"
            )
        ):
            first_row.get_by_role("button", name="Activate key").click()
        keys = _assert_public_list(api_context, test_values)
        active_ids = {item["id"] for item in keys if item.get("is_active")}
        if active_ids != {first_id}:
            pytest.fail("Reactivating the first key did not switch the active state.")

        first_row = _key_row(panel, keys, first_id)
        with page.expect_response(
            lambda response: _expect_key_response(
                response, "POST", f"/api/keys/{first_id}/reveal"
            )
        ) as reveal_info:
            first_row.get_by_role("button", name="Reveal key").click()
        if reveal_info.value.status != 403:
            pytest.fail(
                f"The default reveal policy returned HTTP {reveal_info.value.status}, not 403."
            )
        expect(page.get_by_role("alert")).to_contain_text(
            "Key reveal is disabled by server policy"
        )
        expect(first_row.locator(".key-manager-preview")).to_have_text(
            first["masked_preview"]
        )
        if key_a in panel.inner_text():
            pytest.fail(
                "A forbidden reveal replaced the masked preview with plaintext."
            )

        with page.expect_response(
            lambda response: _expect_key_response(
                response, "POST", f"/api/keys/{first_id}/deactivate"
            )
        ):
            first_row.get_by_role("button", name="Deactivate key").click()
        keys = _assert_public_list(api_context, test_values)
        first_row = _key_row(panel, keys, first_id)
        page.once("dialog", lambda dialog: dialog.accept())
        with page.expect_response(
            lambda response: _expect_key_response(
                response, "DELETE", f"/api/keys/{first_id}"
            )
        ) as first_delete_info:
            first_row.get_by_role("button", name="Delete key").click()
        if first_delete_info.value.status != 200:
            pytest.fail(
                f"Deleting the inactive first key returned HTTP {first_delete_info.value.status}."
            )
        keys = _assert_public_list(api_context, test_values)
        if any(item["id"] == first_id for item in keys):
            pytest.fail("The deleted first key remained in the API list.")
        expect(rows).to_have_count(len(keys))

        second_row = _key_row(panel, keys, second_id)
        page.once("dialog", lambda dialog: dialog.accept())
        with page.expect_response(
            lambda response: _expect_key_response(
                response, "DELETE", f"/api/keys/{second_id}"
            )
        ) as second_delete_info:
            second_row.get_by_role("button", name="Delete key").click()
        if second_delete_info.value.status != 200:
            pytest.fail(
                f"Deleting the inactive second key returned HTTP {second_delete_info.value.status}."
            )
        final_keys = _read_keys(api_context)
        if {item["id"] for item in final_keys} != baseline_ids:
            pytest.fail("The key list did not return to its pre-test IDs.")
        expect(rows).to_have_count(len(final_keys))
        _assert_plaintext_absent_from_logs(test_values)
    finally:
        _cleanup_created_keys(api_context, created_ids, original_active_ids)
        final_keys = _read_keys(api_context)
        if {item["id"] for item in final_keys} != baseline_ids:
            pytest.fail("Cleanup did not restore the pre-test key IDs.")
        if {
            item["id"] for item in final_keys if item.get("is_active")
        } != original_active_ids:
            pytest.fail("Cleanup did not restore the original active-key state.")
        _assert_database_key_state(baseline_ids, original_active_ids)


@pytest.mark.skipif(
    not os.environ.get(TEST_KEY_ENV),
    reason="TKBEN_TEST_HF_KEY is unavailable; supplied-credential coverage was skipped.",
)
def test_supplied_hf_key_is_masked_encrypted_and_not_revealed(
    page: Page, base_url: str, api_context: APIRequestContext
) -> None:
    key_value = os.environ[TEST_KEY_ENV]
    if not key_value.startswith("hf_") or any(
        character.isspace() for character in key_value
    ):
        pytest.fail(
            "TKBEN_TEST_HF_KEY is not a valid Hugging Face key; value withheld."
        )

    baseline = _read_keys(api_context)
    baseline_ids = {item["id"] for item in baseline}
    original_active_ids = {item["id"] for item in baseline if item.get("is_active")}
    if len(original_active_ids) > 1:
        pytest.skip(
            "The existing key list has multiple active rows; no mutation was made."
        )
    created_ids: set[int] = set()
    try:
        panel = _settings_keys_panel(page, base_url)
        key_input = page.get_by_role("textbox", name="Hugging Face key")
        key_input.fill(key_value)
        with page.expect_response(
            lambda response: _expect_key_response(response, "POST", "/api/keys")
        ) as create_info:
            page.get_by_role("button", name="Add", exact=True).click()
        if create_info.value.status == 409:
            pytest.skip(
                "The supplied credential is already stored; no key was created."
            )
        if create_info.value.status != 201:
            pytest.fail(
                f"Adding the supplied key returned HTTP {create_info.value.status}."
            )
        expect(key_input).to_have_value("")
        item = create_info.value.json()
        if "key_value" in item or key_value in json.dumps(item, sort_keys=True):
            pytest.fail("The create response exposed the supplied credential.")
        if not isinstance(item.get("id"), int):
            pytest.fail("The create response did not include a key ID.")
        key_id = item["id"]
        created_ids.add(key_id)
        keys = _assert_public_list(api_context, (key_value,))
        row = _key_row(panel, keys, key_id)
        expect(row.locator(".key-manager-preview")).to_have_text(item["masked_preview"])
        if key_value in panel.inner_text():
            pytest.fail("The rendered key list exposed the supplied credential.")
        _assert_ciphertext(key_id, key_value)

        with page.expect_response(
            lambda response: _expect_key_response(
                response, "POST", f"/api/keys/{key_id}/reveal"
            )
        ) as reveal_info:
            row.get_by_role("button", name="Reveal key").click()
        if reveal_info.value.status != 403:
            pytest.fail(
                f"The default reveal policy returned HTTP {reveal_info.value.status}, not 403."
            )
        expect(row.locator(".key-manager-preview")).to_have_text(item["masked_preview"])
        _assert_plaintext_absent_from_logs((key_value,))
    finally:
        _cleanup_created_keys(api_context, created_ids, original_active_ids)
    final_ids = {item["id"] for item in _read_keys(api_context)}
    if final_ids != baseline_ids:
        pytest.fail("The key list did not return to its pre-test IDs.")
    _assert_database_key_state(baseline_ids, original_active_ids)
