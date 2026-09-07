from __future__ import annotations

from pathlib import Path

import pytest

from server.common.utils.security import ensure_path_is_within, normalize_upload_stem

###############################################################################
def test_normalize_upload_stem_handles_windows_style_names() -> None:
    assert normalize_upload_stem(r"nested\folder\Sample File.csv") == "Sample_File"

###############################################################################
def test_normalize_upload_stem_handles_posix_style_names() -> None:
    assert normalize_upload_stem("nested/folder/report.final.xlsx") == "report.final"

###############################################################################
def test_ensure_path_is_within_accepts_str_and_path_inputs(tmp_path: Path) -> None:
    base_path = tmp_path / "datasets"
    candidate_path = base_path / "safe" / "artifact.json"

    resolved = ensure_path_is_within(str(base_path), candidate_path)

    assert resolved == str(candidate_path.resolve())

###############################################################################
def test_ensure_path_is_within_rejects_escape_attempts(tmp_path: Path) -> None:
    base_path = tmp_path / "datasets"
    candidate_path = base_path / ".." / "outside.txt"

    with pytest.raises(ValueError, match="escapes the allowed base directory"):
        ensure_path_is_within(base_path, candidate_path)
