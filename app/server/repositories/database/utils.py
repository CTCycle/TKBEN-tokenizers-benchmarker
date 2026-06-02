from __future__ import annotations

from pathlib import Path

WINDOWS_EXTENDED_PATH_PREFIX = "\\\\?\\"


# -----------------------------------------------------------------------------
def normalize_sqlite_path(path: str | Path) -> str:
    normalized = str(path)
    if normalized.startswith(WINDOWS_EXTENDED_PATH_PREFIX):
        return normalized[len(WINDOWS_EXTENDED_PATH_PREFIX) :]
    return normalized

