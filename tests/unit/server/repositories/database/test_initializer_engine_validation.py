from __future__ import annotations

import pytest

from TKBEN.server.repositories.database.initializer import _resolve_postgres_engine


def test_resolve_postgres_engine_accepts_psycopg() -> None:
    assert _resolve_postgres_engine("postgresql+psycopg") == "postgresql+psycopg"


@pytest.mark.parametrize("engine", ["postgres", "postgresql", "postgresql+psycopg2"])
def test_resolve_postgres_engine_rejects_legacy_aliases(engine: str) -> None:
    with pytest.raises(ValueError, match="Unsupported database engine"):
        _resolve_postgres_engine(engine)
