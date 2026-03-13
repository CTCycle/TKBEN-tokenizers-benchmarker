from __future__ import annotations

import sqlalchemy
from sqlalchemy.sql.elements import TextClause


SELECT_TOKENIZER_EXISTS_BY_NAME: TextClause = sqlalchemy.text(
    'SELECT 1 FROM "tokenizer" WHERE "name" = :name LIMIT 1'
)

INSERT_TOKENIZER_IF_MISSING: TextClause = sqlalchemy.text(
    'INSERT INTO "tokenizer" ("name") '
    "VALUES (:name) "
    'ON CONFLICT ("name") DO NOTHING'
)

SELECT_TOKENIZER_NAMES_ASC: TextClause = sqlalchemy.text(
    'SELECT "name" FROM "tokenizer" ORDER BY "name" ASC'
)

SELECT_TOKENIZER_ID_AND_NAME_BY_NAME: TextClause = sqlalchemy.text(
    'SELECT "id", "name" FROM "tokenizer" WHERE "name" = :name'
)
