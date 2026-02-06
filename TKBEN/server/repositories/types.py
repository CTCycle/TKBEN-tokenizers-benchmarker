from __future__ import annotations

from typing import Any

from sqlalchemy import JSON
from sqlalchemy.types import TypeDecorator


###############################################################################
class JSONSequence(TypeDecorator):
    impl = JSON
    cache_ok = True

    def process_bind_param(self, value, dialect) -> Any | None:  # type: ignore[override]
        return value

    def process_result_value(self, value, dialect) -> Any | None:  # type: ignore[override]
        return value


###############################################################################
class IntSequence(JSONSequence):
    pass
