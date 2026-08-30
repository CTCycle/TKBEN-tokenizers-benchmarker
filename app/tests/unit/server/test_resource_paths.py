from __future__ import annotations

from pathlib import Path

from server.common.path import ROOT_DIR, _resolve_resource_path


###############################################################################
def test_default_resource_path_is_app_resources() -> None:
    assert _resolve_resource_path(None) == (ROOT_DIR / "app/resources").resolve()


###############################################################################
def test_relative_resource_path_is_root_relative() -> None:
    configured = _resolve_resource_path("custom/resources")

    assert configured == (ROOT_DIR / "custom/resources").resolve()


###############################################################################
def test_absolute_resource_path_is_preserved(tmp_path: Path) -> None:
    assert _resolve_resource_path(str(tmp_path)) == tmp_path.resolve()
