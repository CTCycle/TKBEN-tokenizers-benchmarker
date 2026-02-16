from __future__ import annotations

from TKBEN.server.repositories.database.initializer import (
    build_postgres_create_database_sql,
)


###############################################################################
def test_build_postgres_create_database_sql_uses_template0_for_utf8() -> None:
    statement = build_postgres_create_database_sql("TKBEN")
    sql = str(statement)

    assert 'CREATE DATABASE "TKBEN"' in sql
    assert "ENCODING 'UTF8'" in sql
    assert "TEMPLATE template0" in sql


###############################################################################
def test_build_postgres_create_database_sql_escapes_identifier_quotes() -> None:
    statement = build_postgres_create_database_sql('TK"BEN')
    sql = str(statement)

    assert 'CREATE DATABASE "TK""BEN"' in sql
