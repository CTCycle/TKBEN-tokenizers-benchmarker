# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import signal
from collections.abc import Generator
from contextlib import contextmanager

import uvicorn

from server.app import app


###############################################################################
class LauncherOwnedServer(uvicorn.Server):
    """Leave Ctrl+C to the launcher so shutdown uses the authenticated API."""

    # -------------------------------------------------------------------------
    @contextmanager
    def capture_signals(self) -> Generator[None, None, None]:
        yield


###############################################################################
def main() -> None:
    parser = argparse.ArgumentParser(description="TKBEN source-mode server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    arguments = parser.parse_args()

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    server = LauncherOwnedServer(
        uvicorn.Config(
            app,
            host=arguments.host,
            port=arguments.port,
            log_level="info",
            # Route uvicorn records through the central TKBEN logging config.
            log_config=None,
        )
    )
    app.state.launcher_server = server
    server.run()


if __name__ == "__main__":
    main()
