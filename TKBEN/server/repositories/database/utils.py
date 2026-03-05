from __future__ import annotations


WINDOWS_EXTENDED_PATH_PREFIX = "\\\\?\\"


# -----------------------------------------------------------------------------
def normalize_sqlite_path(path: str) -> str:
    if path.startswith(WINDOWS_EXTENDED_PATH_PREFIX):
        return path[len(WINDOWS_EXTENDED_PATH_PREFIX):]
    return path


# -----------------------------------------------------------------------------
def normalize_postgres_engine(engine: str | None) -> str:
    if not engine:
        return "postgresql+psycopg"
    lowered = engine.lower()
    if lowered in {"postgres", "postgresql"}:
        return "postgresql+psycopg"
    return engine
