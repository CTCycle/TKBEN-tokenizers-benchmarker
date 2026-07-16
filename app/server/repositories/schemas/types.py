from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, JSON
from sqlalchemy.types import TypeDecorator

###############################################################################
class UTCDateTime(TypeDecorator[datetime]):
    impl = DateTime
    cache_ok = True

    # -------------------------------------------------------------------------
    def load_dialect_impl(self, dialect):  # type: ignore[no-untyped-def]
        return dialect.type_descriptor(DateTime(timezone=True))

    # -------------------------------------------------------------------------
    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:  # type: ignore[no-untyped-def]
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    # -------------------------------------------------------------------------
    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:  # type: ignore[no-untyped-def]
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

###############################################################################
class JSONObject(TypeDecorator[dict[str, Any]]):
    impl = JSON
    cache_ok = True

    # -------------------------------------------------------------------------
    def process_bind_param(self, value, dialect):  # type: ignore[no-untyped-def]
        if value is not None and not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value

###############################################################################
class JSONArray(TypeDecorator[list[Any]]):
    impl = JSON
    cache_ok = True

    # -------------------------------------------------------------------------
    def process_bind_param(self, value, dialect):  # type: ignore[no-untyped-def]
        if value is not None and not isinstance(value, list):
            raise ValueError("Expected a JSON array")
        return value


# Removed compatibility aliases: the canonical schema uses explicit shapes.
