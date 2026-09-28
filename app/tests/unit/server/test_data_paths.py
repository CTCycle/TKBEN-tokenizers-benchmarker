from __future__ import annotations

from pathlib import Path

from server.common.path import ROOT_DIR, _resolve_data_path

###############################################################################
def test_default_data_path_is_repository_data() -> None:
    assert _resolve_data_path(None) == (ROOT_DIR / "data").resolve()

###############################################################################
def test_relative_data_path_is_root_relative() -> None:
    configured = _resolve_data_path("custom/data")

    assert configured == (ROOT_DIR / "custom/data").resolve()

###############################################################################
def test_absolute_data_path_is_preserved(tmp_path: Path) -> None:
    assert _resolve_data_path(str(tmp_path)) == tmp_path.resolve()
