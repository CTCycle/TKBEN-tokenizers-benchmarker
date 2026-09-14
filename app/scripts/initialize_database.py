from __future__ import annotations

import time

from server.common.utils.logger import configure_logging, logger
from server.configurations import get_server_settings
from server.repositories.database.initializer import (
    DatabaseMigrationError,
    initialize_database,
)


###############################################################################
if __name__ == "__main__":
    settings = get_server_settings()
    configure_logging(settings.paths.logs)

    start = time.perf_counter()
    logger.info("Starting database initialization")
    try:
        initialize_database(settings=settings)
    except DatabaseMigrationError as exc:
        logger.error("Database initialization failed: %s", exc)
        raise SystemExit(1) from exc
    elapsed = time.perf_counter() - start
    logger.info("Database initialization completed in %.2f seconds", elapsed)
