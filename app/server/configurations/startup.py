# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from collections.abc import Iterable, Mapping
from threading import RLock

from server.configurations.database_configuration import DatabaseConfigurationStore
from server.configurations.environment import ensure_environment_loaded
from server.configurations.runtime import (
    RuntimeSettingsPersistence,
    RuntimeSettingsState,
    RuntimeSettingsStore,
)
from server.configurations.settings import ServerSettings, build_server_settings


_DEFAULT_SETTINGS_LOCK = RLock()
_default_settings: ServerSettings | None = None
_runtime_settings_store: RuntimeSettingsStore | None = None
_database_configuration_store: DatabaseConfigurationStore | None = None


###############################################################################
def _initialize_settings_locked() -> None:
    global _default_settings, _runtime_settings_store, _database_configuration_store
    ensure_environment_loaded()

    bootstrap = build_server_settings()
    database_store = DatabaseConfigurationStore(
        bootstrap.paths.data,
        bootstrap.security.hf_keys_encryption_material_file,
    )
    database_configuration = database_store.load()
    defaults = build_server_settings(
        database_settings=database_store.to_database_settings(database_configuration)
    )
    store = RuntimeSettingsStore(defaults)
    _database_configuration_store = database_store
    _runtime_settings_store = store
    _default_settings = store.get_state().settings


###############################################################################
def get_database_configuration_store() -> DatabaseConfigurationStore:
    with _DEFAULT_SETTINGS_LOCK:
        if _database_configuration_store is None:
            _initialize_settings_locked()
        assert _database_configuration_store is not None
        return _database_configuration_store


###############################################################################
def build_database_runtime_settings_store(
    settings: ServerSettings, persistence: RuntimeSettingsPersistence
) -> RuntimeSettingsStore:
    return RuntimeSettingsStore(settings, persistence)


###############################################################################
def activate_runtime_settings_store(
    store: RuntimeSettingsStore,
) -> RuntimeSettingsState:
    global _default_settings, _runtime_settings_store
    with _DEFAULT_SETTINGS_LOCK:
        _runtime_settings_store = store
        state = store.get_state()
        _default_settings = state.settings
        return state


###############################################################################
def get_server_settings() -> ServerSettings:
    """Return the canonical effective immutable settings snapshot."""

    global _default_settings
    with _DEFAULT_SETTINGS_LOCK:
        if _default_settings is None:
            _initialize_settings_locked()
        assert _default_settings is not None
        return _default_settings


###############################################################################
def get_runtime_settings_store() -> RuntimeSettingsStore:
    with _DEFAULT_SETTINGS_LOCK:
        if _runtime_settings_store is None:
            _initialize_settings_locked()
        assert _runtime_settings_store is not None
        return _runtime_settings_store


###############################################################################
def get_runtime_settings_state() -> RuntimeSettingsState:
    return get_runtime_settings_store().get_state()


###############################################################################
def apply_runtime_settings_patch(
    patch: Mapping[str, Mapping[str, object]],
    *,
    expected_revision: int,
) -> RuntimeSettingsState:
    global _default_settings
    with _DEFAULT_SETTINGS_LOCK:
        store = get_runtime_settings_store()
        state = store.apply_patch(patch, expected_revision=expected_revision)
        _default_settings = state.settings
    return state


###############################################################################
def reset_runtime_settings(
    *,
    expected_revision: int,
    keys: Iterable[str] | None = None,
    reset_all: bool = False,
) -> RuntimeSettingsState:
    global _default_settings
    with _DEFAULT_SETTINGS_LOCK:
        store = get_runtime_settings_store()
        state = store.reset(
            expected_revision=expected_revision,
            keys=keys,
            reset_all=reset_all,
        )
        _default_settings = state.settings
    return state


###############################################################################
def is_key_reveal_enabled() -> bool:
    return get_server_settings().security.allow_key_reveal


###############################################################################
def reset_settings_cache_for_tests() -> None:
    global _default_settings, _runtime_settings_store, _database_configuration_store
    with _DEFAULT_SETTINGS_LOCK:
        _default_settings = None
        _runtime_settings_store = None
        _database_configuration_store = None
