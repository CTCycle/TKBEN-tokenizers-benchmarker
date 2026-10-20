# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator


class DatabaseSwitchGate:
    """Drain ordinary API calls while a validated database is activated."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._switch_lock = asyncio.Lock()
        self._active_requests = 0
        self._switching = False

    @asynccontextmanager
    async def request(self) -> AsyncIterator[None]:
        async with self._condition:
            await self._condition.wait_for(lambda: not self._switching)
            self._active_requests += 1
        try:
            yield
        finally:
            async with self._condition:
                self._active_requests -= 1
                if self._active_requests == 0:
                    self._condition.notify_all()

    @asynccontextmanager
    async def switch(self) -> AsyncIterator[None]:
        async with self._switch_lock:
            async with self._condition:
                self._switching = True
                await self._condition.wait_for(lambda: self._active_requests == 0)
            try:
                yield
            finally:
                async with self._condition:
                    self._switching = False
                    self._condition.notify_all()
