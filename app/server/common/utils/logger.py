# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import logging
import logging.config
from datetime import datetime
from pathlib import Path
from threading import RLock


logger = logging.getLogger("app")
_LOGGING_LOCK = RLock()
_configured_log_directory: Path | None = None
_NOISY_LIBRARY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "filelock",
    "fsspec",
    "huggingface_hub",
    "datasets",
    "asyncio",
    "multipart",
    "alembic",
    "sqlalchemy.engine",
)


###############################################################################
class FailedRequestAccessFilter(logging.Filter):
    # uvicorn access records carry (client, method, path, http_version, status)

    # -------------------------------------------------------------------------
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and args and isinstance(args[-1], int):
            return args[-1] >= 400
        return True


###############################################################################
def configure_logging(log_directory: str | Path) -> Path:
    global _configured_log_directory
    resolved_directory = Path(log_directory).expanduser().resolve()

    with _LOGGING_LOCK:
        if _configured_log_directory == resolved_directory:
            return resolved_directory

        resolved_directory.mkdir(parents=True, exist_ok=True)
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = resolved_directory / f"TKBEN_{current_timestamp}.log"

        logging.config.dictConfig(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "default": {
                        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        "datefmt": "%d-%m-%Y %H:%M:%S",
                    },
                    "minimal": {
                        "format": "%(levelname)s - %(message)s",
                    },
                },
                "filters": {
                    "failed_requests_only": {
                        "()": f"{__name__}.FailedRequestAccessFilter"
                    },
                },
                "loggers": {
                    # Successful requests (polling, uploads) are noise; 4xx/5xx remain.
                    "uvicorn.access": {"filters": ["failed_requests_only"]},
                    **{name: {"level": "WARNING"} for name in _NOISY_LIBRARY_LOGGERS},
                },
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "level": "INFO",
                        "formatter": "minimal",
                    },
                    "file": {
                        "class": "logging.FileHandler",
                        "level": "DEBUG",
                        "formatter": "default",
                        "filename": str(log_filename),
                        "mode": "a",
                        "encoding": "utf-8",
                    },
                },
                "root": {
                    "level": "DEBUG",
                    "handlers": ["console", "file"],
                },
            }
        )
        _configured_log_directory = resolved_directory
        return resolved_directory
