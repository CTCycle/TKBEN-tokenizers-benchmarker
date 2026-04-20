from __future__ import annotations


WINDOWS_EXTENDED_PATH_PREFIX = "\\\\?\\"


# -----------------------------------------------------------------------------
def normalize_sqlite_path(path: str) -> str:
    if path.startswith(WINDOWS_EXTENDED_PATH_PREFIX):
        return path[len(WINDOWS_EXTENDED_PATH_PREFIX) :]
    return path

