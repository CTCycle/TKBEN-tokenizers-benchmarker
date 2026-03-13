from __future__ import annotations

import sqlalchemy
from sqlalchemy.sql.elements import TextClause


SELECT_TOKENIZER_EXISTS_BY_NAME: TextClause = sqlalchemy.text(
    'SELECT 1 FROM "tokenizer" WHERE "name" = :name LIMIT 1'
)

SELECT_DATASET_DOCUMENT_ROWS_BY_DATASET_NAME: TextClause = sqlalchemy.text(
    'SELECT dd."id", dd."text" '
    'FROM "dataset_document" dd '
    'JOIN "dataset" d ON d."id" = dd."dataset_id" '
    'WHERE d."name" = :dataset ORDER BY dd."id"'
)

SELECT_DATASET_DOCUMENT_COUNT_BY_DATASET_NAME: TextClause = sqlalchemy.text(
    "SELECT COUNT(*) "
    'FROM "dataset_document" dd '
    'JOIN "dataset" d ON d."id" = dd."dataset_id" '
    'WHERE d."name" = :dataset'
)

SELECT_TOKENIZATION_DATASET_STATS_IDS_BY_DATASET_ID: TextClause = sqlalchemy.text(
    'SELECT "id", "tokenizer_id" '
    'FROM "tokenization_dataset_stats" '
    'WHERE "dataset_id" = :dataset_id'
)

SELECT_DATASET_ID_BY_NAME: TextClause = sqlalchemy.text(
    'SELECT "id" FROM "dataset" WHERE "name" = :dataset LIMIT 1'
)

INSERT_TOKENIZER_IF_MISSING: TextClause = sqlalchemy.text(
    'INSERT INTO "tokenizer" ("name") '
    "VALUES (:name) "
    'ON CONFLICT ("name") DO NOTHING'
)

SELECT_TOKENIZER_ID_AND_NAME_BY_NAME: TextClause = sqlalchemy.text(
    'SELECT "id", "name" FROM "tokenizer" WHERE "name" = :name'
)
