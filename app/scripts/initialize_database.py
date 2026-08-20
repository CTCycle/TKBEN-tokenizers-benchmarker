from __future__ import annotations

import time

from server.common.utils.logger import logger
from server.repositories.database.initializer import (
    DatabaseMigrationError,
    initialize_database,
)


###############################################################################
if __name__ == "__main__":
    start = time.perf_counter()
    logger.info("Starting database initialization")
    try:
        initialize_database()
    except DatabaseMigrationError as exc:
        logger.error("Database initialization failed: %s", exc)
        raise SystemExit(1) from exc
    elapsed = time.perf_counter() - start
    logger.info("Database initialization completed in %.2f seconds", elapsed)
