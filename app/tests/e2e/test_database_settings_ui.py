# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any
from pathlib import Path

import pytest
from playwright.sync_api import APIRequestContext, Page, expect


###############################################################################
def _restore_runtime_settings(
    api_context: APIRequestContext, original: dict[str, Any]
) -> None:
    current_response = api_context.get("/api/settings")
    assert current_response.ok, current_response.text()
    reset_response = api_context.post(
        "/api/settings/reset",
        data={"expected_revision": current_response.json()["revision"], "all": True},
    )
    assert reset_response.ok, reset_response.text()
    keys = original.get("overridden_keys", [])
    if not keys:
        return
    patch: dict[str, Any] = {"expected_revision": reset_response.json()["revision"]}
    for key in keys:
        group, field = key.split(".", maxsplit=1)
        patch.setdefault(group, {})[field] = original["settings"][group][field]
    response = api_context.patch("/api/settings", data=patch)
    assert response.ok, response.text()


###############################################################################
def _switch_request(configuration: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": configuration["backend"],
        "sqlite_path": configuration.get("sqlite_path"),
        "host": configuration.get("host"),
        "port": configuration.get("port"),
        "database_name": configuration.get("database_name"),
        "username": configuration.get("username"),
        "clear_password": False,
        "ssl": configuration.get("ssl", False),
        "ssl_ca": configuration.get("ssl_ca"),
        "connect_timeout": configuration["connect_timeout"],
        "insert_batch_size": configuration["insert_batch_size"],
    }


###############################################################################
def _switch_back_if_needed(
    api_context: APIRequestContext,
    original_configuration: dict[str, Any],
) -> None:
    current_response = api_context.get("/api/database")
    if not current_response.ok:
        return
    current = current_response.json()
    if (
        current["backend"] == original_configuration["backend"]
        and current.get("sqlite_path") == original_configuration.get("sqlite_path")
        and current.get("host") == original_configuration.get("host")
        and current.get("database_name") == original_configuration.get("database_name")
    ):
        return
    response = api_context.post(
        "/api/database/switch",
        data=_switch_request(original_configuration),
    )
    assert response.ok, response.text()


###############################################################################
def test_database_tab_tests_switches_and_reloads_database_local_settings(
    page: Page,
    base_url: str,
    api_context: APIRequestContext,
    tmp_path: Path,
) -> None:
    configuration_response = api_context.get("/api/database")
    assert configuration_response.ok, configuration_response.text()
    original_configuration = configuration_response.json()
    if original_configuration["backend"] != "sqlite":
        pytest.skip("This UI workflow requires the isolated SQLite starting backend.")
    assert original_configuration["connection_status"] == "connected"

    original_settings_response = api_context.get("/api/settings")
    assert original_settings_response.ok, original_settings_response.text()
    original_settings = original_settings_response.json()

    target_path = tmp_path / "settings-switch-target.db"
    configured_settings_value = original_settings["settings"]["datasets"]["histogram_bins"]
    distinct_value = 30 if configured_settings_value != 30 else 31

    try:
        page.goto(f"{base_url}/settings")
        expect(page.locator(".settings-form")).to_be_visible()
        histogram = page.get_by_label("Histogram bins")
        with page.expect_response(
            lambda response: response.url.endswith("/api/settings")
            and response.request.method == "PATCH"
        ):
            histogram.fill(str(distinct_value))
        expect(histogram).to_have_value(str(distinct_value))

        page.get_by_role("tab", name="Database").click()
        expect(page.locator("#settings-panel-database")).to_be_visible()
        expect(page.locator(".database-active-status")).to_contain_text("SQLite · Connected")
        switch_button = page.get_by_role("button", name="Switch database")
        expect(switch_button).to_be_disabled()

        page.get_by_label("Database backend").select_option("postgresql")
        page.get_by_label("Host").fill("127.0.0.1")
        page.get_by_label("Port").fill("1")
        page.get_by_label("Database name").fill("unavailable")
        page.get_by_label("Username").fill("unavailable")
        page.get_by_label("Connection timeout (seconds)").fill("1")
        page.get_by_role("button", name="Test connection").click()
        expect(page.locator(".database-result--failure")).to_contain_text(
            "Unable to connect"
        )
        expect(switch_button).to_be_disabled()
        unchanged_configuration = api_context.get("/api/database")
        assert unchanged_configuration.ok, unchanged_configuration.text()
        assert unchanged_configuration.json()["sqlite_path"] == original_configuration["sqlite_path"]

        page.get_by_label("Database backend").select_option("sqlite")
        page.get_by_label("Database file path").fill(str(target_path))
        page.get_by_label("Connection timeout (seconds)").fill(
            str(original_configuration["connect_timeout"])
        )
        page.get_by_role("button", name="Test connection").click()
        expect(page.locator(".database-result--success")).to_contain_text(
            "Connection successful."
        )
        expect(switch_button).to_be_enabled()
        unchanged_configuration = api_context.get("/api/database")
        assert unchanged_configuration.ok, unchanged_configuration.text()
        assert unchanged_configuration.json()["sqlite_path"] == original_configuration["sqlite_path"]
        with page.expect_navigation(wait_until="load"):
            switch_button.click()

        switched_settings = api_context.get("/api/settings")
        assert switched_settings.ok, switched_settings.text()
        assert switched_settings.json()["settings"]["datasets"]["histogram_bins"] == 20

        page.get_by_role("tab", name="Database").click()
        expect(page.locator(".database-active-status")).to_contain_text("SQLite · Connected")
        assert page.get_by_label("Database file path").input_value() == str(target_path)
        page.get_by_label("Database file path").fill(original_configuration["sqlite_path"])
        page.get_by_role("button", name="Test connection").click()
        expect(page.locator(".database-result--success")).to_contain_text(
            "Connection successful."
        )
        with page.expect_navigation(wait_until="load"):
            page.get_by_role("button", name="Switch database").click()

        restored_settings = api_context.get("/api/settings")
        assert restored_settings.ok, restored_settings.text()
        assert (
            restored_settings.json()["settings"]["datasets"]["histogram_bins"]
            == distinct_value
        )
        active_configuration = api_context.get("/api/database").json()
        assert active_configuration["sqlite_path"] == original_configuration["sqlite_path"]
        page.get_by_role("tab", name="Database").click()
        expect(page.locator(".database-active-status")).to_contain_text(
            "SQLite · Connected"
        )
        qa_directory = Path(__file__).resolve().parents[3] / "assets" / "QA"
        qa_directory.mkdir(parents=True, exist_ok=True)
        page.screenshot(
            path=str(qa_directory / "database-settings-latest.png"),
            full_page=True,
        )
    finally:
        _switch_back_if_needed(api_context, original_configuration)
        _restore_runtime_settings(api_context, original_settings)
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{target_path}{suffix}").unlink(missing_ok=True)
