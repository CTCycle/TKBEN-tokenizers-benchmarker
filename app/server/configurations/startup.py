from __future__ import annotations

from collections.abc import Iterable, Mapping
from threading import RLock

from server.configurations.environment import ensure_environment_loaded
from server.configurations.runtime import RuntimeSettingsState, RuntimeSettingsStore
from server.configurations.settings import ServerSettings, build_server_settings


_DEFAULT_SETTINGS_LOCK = RLock()
_default_settings: ServerSettings | None = None
_runtime_settings_store: RuntimeSettingsStore | None = None


###############################################################################
def _initialize_settings_locked() -> None:
    global _default_settings, _runtime_settings_store
    ensure_environment_loaded(force=True)
    defaults = build_server_settings()
    store = RuntimeSettingsStore(defaults)
    _runtime_settings_store = store
    _default_settings = store.get_state().settings


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
    global _default_settings, _runtime_settings_store
    with _DEFAULT_SETTINGS_LOCK:
        _default_settings = None
        _runtime_settings_store = None
