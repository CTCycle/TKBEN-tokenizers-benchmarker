from TKBEN.server.repositories.database.initializer import initialize_database
from TKBEN.server.repositories.database.postgres import PostgresRepository
from TKBEN.server.repositories.database.sqlite import SQLiteRepository

__all__ = [
    "BACKEND_FACTORIES",
    "DatabaseBackend",
    "TKBENWebappDatabase",
    "database",
    "initialize_database",
    "PostgresRepository",
    "SQLiteRepository",
]


# -----------------------------------------------------------------------------
def __getattr__(name: str):
    if name in {"BACKEND_FACTORIES", "DatabaseBackend", "TKBENWebappDatabase", "database"}:
        from TKBEN.server.repositories.database.backend import (
            BACKEND_FACTORIES,
            DatabaseBackend,
            TKBENWebappDatabase,
            database,
        )

        exports = {
            "BACKEND_FACTORIES": BACKEND_FACTORIES,
            "DatabaseBackend": DatabaseBackend,
            "TKBENWebappDatabase": TKBENWebappDatabase,
            "database": database,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
