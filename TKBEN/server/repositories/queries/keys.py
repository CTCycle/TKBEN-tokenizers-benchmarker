from __future__ import annotations

import sqlalchemy
from sqlalchemy.sql.elements import TextClause


UPDATE_HF_ACCESS_KEY_VALUE_BY_ID: TextClause = sqlalchemy.text(
    'UPDATE "hf_access_keys" SET "key_value" = :key_value WHERE "id" = :key_id'
)

SELECT_HF_ACCESS_KEYS_FOR_LIST: TextClause = sqlalchemy.text(
    'SELECT "id", "key_value", "created_at", "is_active" '
    'FROM "hf_access_keys" ORDER BY "created_at" DESC, "id" DESC'
)

SELECT_HF_KEY_VALUES: TextClause = sqlalchemy.text('SELECT "key_value" FROM "hf_access_keys"')

INSERT_HF_ACCESS_KEY: TextClause = sqlalchemy.text(
    'INSERT INTO "hf_access_keys" ("key_value", "created_at", "is_active") '
    "VALUES (:key_value, :created_at, :is_active)"
)

SELECT_HF_ACCESS_KEY_BY_VALUE: TextClause = sqlalchemy.text(
    'SELECT "id", "created_at", "is_active" '
    'FROM "hf_access_keys" WHERE "key_value" = :key_value LIMIT 1'
)

SELECT_HF_KEY_VALUE_BY_ID: TextClause = sqlalchemy.text(
    'SELECT "key_value" FROM "hf_access_keys" WHERE "id" = :key_id LIMIT 1'
)

SELECT_HF_ACCESS_KEY_IS_ACTIVE_BY_ID: TextClause = sqlalchemy.text(
    'SELECT "is_active" FROM "hf_access_keys" WHERE "id" = :key_id LIMIT 1'
)

DELETE_HF_ACCESS_KEY_BY_ID: TextClause = sqlalchemy.text(
    'DELETE FROM "hf_access_keys" WHERE "id" = :key_id'
)

SELECT_HF_ACCESS_KEY_ID_AND_IS_ACTIVE_BY_ID: TextClause = sqlalchemy.text(
    'SELECT "id", "is_active" FROM "hf_access_keys" WHERE "id" = :key_id LIMIT 1'
)

UPDATE_HF_ACCESS_KEYS_SET_ACTIVE_EXCEPT_ID: TextClause = sqlalchemy.text(
    'UPDATE "hf_access_keys" SET "is_active" = :is_active WHERE "id" != :key_id'
)

UPDATE_HF_ACCESS_KEY_SET_ACTIVE_BY_ID: TextClause = sqlalchemy.text(
    'UPDATE "hf_access_keys" SET "is_active" = :is_active WHERE "id" = :key_id'
)

SELECT_HF_ACCESS_KEY_ID_BY_ID: TextClause = sqlalchemy.text(
    'SELECT "id" FROM "hf_access_keys" WHERE "id" = :key_id LIMIT 1'
)

SELECT_ACTIVE_HF_ACCESS_KEY: TextClause = sqlalchemy.text(
    'SELECT "id", "key_value" FROM "hf_access_keys" '
    'WHERE "is_active" = :is_active ORDER BY "id" DESC LIMIT 1'
)
