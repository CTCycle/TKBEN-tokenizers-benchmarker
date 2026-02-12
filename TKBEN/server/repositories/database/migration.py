from __future__ import annotations

import re

import sqlalchemy
from sqlalchemy import inspect
from sqlalchemy.engine import Connection, Engine

from TKBEN.server.common.utils.logger import logger

LEGACY_TABLE_NAMES = (
    "text_dataset",
    "text_dataset_statistics",
    "text_dataset_reports",
    "tokenization_local_stats",
    "tokenization_global_stats",
    "vocabulary",
    "vocabulary_statistics",
)

LEGACY_VIEW_SQL: dict[str, str] = {
    "text_dataset": (
        'SELECT dd.id AS "id", d.name AS "name", dd.text AS "text" '
        'FROM dataset_document dd '
        "JOIN dataset d ON d.id = dd.dataset_id"
    ),
    "text_dataset_statistics": (
        'SELECT dds.document_id AS "id", '
        'd.name AS "name", '
        'dds.document_id AS "text_id", '
        'dds.words_count AS "words_count", '
        'dds.avg_words_length AS "avg_words_length", '
        'dds.std_words_length AS "std_words_length" '
        "FROM dataset_document_statistics dds "
        "JOIN dataset_document dd ON dd.id = dds.document_id "
        "JOIN dataset d ON d.id = dd.dataset_id"
    ),
    "text_dataset_reports": (
        'SELECT dr.dataset_id AS "id", '
        'd.name AS "name", '
        'dr.document_statistics AS "document_statistics", '
        'dr.word_statistics AS "word_statistics", '
        'dr.most_common_words AS "most_common_words", '
        'dr.least_common_words AS "least_common_words" '
        "FROM dataset_report dr "
        "JOIN dataset d ON d.id = dr.dataset_id"
    ),
    "tokenization_local_stats": (
        'SELECT tds.id AS "id", '
        't.name AS "tokenizer", '
        'd.name AS "name", '
        'tds.document_id AS "text_id", '
        'tds.tokens_count AS "tokens_count", '
        'tds.tokens_to_words_ratio AS "tokens_to_words_ratio", '
        'tds.bytes_per_token AS "bytes_per_token", '
        'tds.boundary_preservation_rate AS "boundary_preservation_rate", '
        'tds.round_trip_token_fidelity AS "round_trip_token_fidelity", '
        'tds.round_trip_text_fidelity AS "round_trip_text_fidelity", '
        'tds.determinism_stability AS "determinism_stability", '
        'tds.bytes_per_character AS "bytes_per_character" '
        "FROM tokenization_document_stats tds "
        "JOIN tokenizer t ON t.id = tds.tokenizer_id "
        "JOIN dataset_document dd ON dd.id = tds.document_id "
        "JOIN dataset d ON d.id = dd.dataset_id"
    ),
    "tokenization_global_stats": (
        'SELECT g.id AS "id", '
        't.name AS "tokenizer", '
        'd.name AS "name", '
        'g.tokenization_speed_tps AS "tokenization_speed_tps", '
        'g.throughput_chars_per_sec AS "throughput_chars_per_sec", '
        'g.model_size_mb AS "model_size_mb", '
        'g.vocabulary_size AS "vocabulary_size", '
        'g.subword_fertility AS "subword_fertility", '
        'g.oov_rate AS "oov_rate", '
        'g.word_recovery_rate AS "word_recovery_rate", '
        'gd.character_coverage AS "character_coverage", '
        'gd.segmentation_consistency AS "segmentation_consistency", '
        'gd.determinism_rate AS "determinism_rate", '
        'gd.token_distribution_entropy AS "token_distribution_entropy", '
        'gd.rare_token_tail_1 AS "rare_token_tail_1", '
        'gd.rare_token_tail_2 AS "rare_token_tail_2", '
        'gd.boundary_preservation_rate AS "boundary_preservation_rate", '
        'gd.compression_chars_per_token AS "compression_chars_per_token", '
        'gd.compression_bytes_per_character AS "compression_bytes_per_character", '
        'gd.round_trip_fidelity_rate AS "round_trip_fidelity_rate", '
        'gd.round_trip_text_fidelity_rate AS "round_trip_text_fidelity_rate", '
        'gd.token_id_ordering_monotonicity AS "token_id_ordering_monotonicity", '
        'gd.token_unigram_coverage AS "token_unigram_coverage" '
        "FROM tokenization_dataset_stats g "
        "JOIN tokenizer t ON t.id = g.tokenizer_id "
        "JOIN dataset d ON d.id = g.dataset_id "
        "LEFT JOIN tokenization_dataset_stats_detail gd "
        "ON gd.global_stats_id = g.id"
    ),
    "vocabulary": (
        'SELECT tv.id AS "id", '
        't.name AS "tokenizer", '
        'tv.token_id AS "token_id", '
        'tv.vocabulary_tokens AS "vocabulary_tokens", '
        'tv.decoded_tokens AS "decoded_tokens" '
        "FROM tokenizer_vocabulary tv "
        "JOIN tokenizer t ON t.id = tv.tokenizer_id"
    ),
    "vocabulary_statistics": (
        'SELECT tvs.tokenizer_id AS "id", '
        't.name AS "tokenizer", '
        'tvs.vocabulary_size AS "vocabulary_size", '
        'tvs.decoded_tokens AS "decoded_tokens", '
        'tvs.number_shared_tokens AS "number_shared_tokens", '
        'tvs.number_unshared_tokens AS "number_unshared_tokens", '
        'tvs.percentage_subwords AS "percentage_subwords", '
        'tvs.percentage_true_words AS "percentage_true_words" '
        "FROM tokenizer_vocabulary_statistics tvs "
        "JOIN tokenizer t ON t.id = tvs.tokenizer_id"
    ),
}


# -----------------------------------------------------------------------------
def _is_valid_identifier(name: str) -> bool:
    return re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name) is not None


# -----------------------------------------------------------------------------
def _relation_exists(conn: Connection, name: str) -> bool:
    inspector = inspect(conn)
    return name in inspector.get_table_names() or name in inspector.get_view_names()


# -----------------------------------------------------------------------------
def _table_exists(conn: Connection, name: str) -> bool:
    inspector = inspect(conn)
    return name in inspector.get_table_names()


# -----------------------------------------------------------------------------
def _table_source(conn: Connection, table_name: str) -> str | None:
    legacy_name = f"{table_name}_legacy"
    if _table_exists(conn, legacy_name):
        return legacy_name
    if _table_exists(conn, table_name):
        return table_name
    return None


# -----------------------------------------------------------------------------
def _rename_legacy_tables(conn: Connection) -> None:
    for table_name in LEGACY_TABLE_NAMES:
        if not _table_exists(conn, table_name):
            continue
        legacy_name = f"{table_name}_legacy"
        if _table_exists(conn, legacy_name):
            logger.warning(
                "Legacy source table %s already exists; skipping rename for %s.",
                legacy_name,
                table_name,
            )
            continue
        conn.execute(
            sqlalchemy.text(f'ALTER TABLE "{table_name}" RENAME TO "{legacy_name}"')
        )
        logger.info("Renamed legacy table %s -> %s", table_name, legacy_name)


# -----------------------------------------------------------------------------
def _insert_distinct_names(
    conn: Connection,
    target_table: str,
    target_column: str,
    source_table: str,
    source_column: str,
) -> None:
    if not (_is_valid_identifier(target_table) and _is_valid_identifier(source_table)):
        raise ValueError("Invalid identifier detected in migration.")
    if not (_is_valid_identifier(target_column) and _is_valid_identifier(source_column)):
        raise ValueError("Invalid column identifier detected in migration.")

    conn.execute(
        sqlalchemy.text(
            f'INSERT INTO "{target_table}" ("{target_column}") '
            f'SELECT DISTINCT src."{source_column}" '
            f'FROM "{source_table}" src '
            f'WHERE src."{source_column}" IS NOT NULL '
            f'AND NOT EXISTS ('
            f'SELECT 1 FROM "{target_table}" t '
            f'WHERE t."{target_column}" = src."{source_column}")'
        )
    )


# -----------------------------------------------------------------------------
def _backfill_dimensions(conn: Connection) -> None:
    dataset_sources = (
        ("text_dataset", "name"),
        ("text_dataset_reports", "name"),
        ("text_dataset_statistics", "name"),
        ("tokenization_local_stats", "name"),
        ("tokenization_global_stats", "name"),
    )
    tokenizer_sources = (
        ("tokenization_local_stats", "tokenizer"),
        ("tokenization_global_stats", "tokenizer"),
        ("vocabulary", "tokenizer"),
        ("vocabulary_statistics", "tokenizer"),
    )

    for source_table, source_column in dataset_sources:
        resolved = _table_source(conn, source_table)
        if resolved is None:
            continue
        _insert_distinct_names(conn, "dataset", "name", resolved, source_column)

    for source_table, source_column in tokenizer_sources:
        resolved = _table_source(conn, source_table)
        if resolved is None:
            continue
        _insert_distinct_names(conn, "tokenizer", "name", resolved, source_column)


# -----------------------------------------------------------------------------
def _backfill_dataset_documents(conn: Connection) -> None:
    source = _table_source(conn, "text_dataset")
    if source is None:
        return

    conn.execute(
        sqlalchemy.text(
            f'INSERT INTO "dataset_document" ("id", "dataset_id", "text") '
            f'SELECT src."id", d."id", src."text" '
            f'FROM "{source}" src '
            f'JOIN "dataset" d ON d."name" = src."name" '
            f'WHERE NOT EXISTS ('
            f'SELECT 1 FROM "dataset_document" dd WHERE dd."id" = src."id")'
        )
    )


# -----------------------------------------------------------------------------
def _backfill_document_statistics(conn: Connection) -> None:
    source = _table_source(conn, "text_dataset_statistics")
    if source is None:
        return

    conn.execute(
        sqlalchemy.text(
            f'INSERT INTO "dataset_document_statistics" ('
            f'"document_id", "words_count", "avg_words_length", "std_words_length") '
            f'SELECT src."text_id", src."words_count", src."avg_words_length", '
            f'src."std_words_length" '
            f'FROM "{source}" src '
            f'WHERE src."text_id" IS NOT NULL '
            f'AND EXISTS ('
            f'SELECT 1 FROM "dataset_document" dd WHERE dd."id" = src."text_id") '
            f'AND NOT EXISTS ('
            f'SELECT 1 FROM "dataset_document_statistics" dds '
            f'WHERE dds."document_id" = src."text_id")'
        )
    )


# -----------------------------------------------------------------------------
def _backfill_dataset_reports(conn: Connection) -> None:
    source = _table_source(conn, "text_dataset_reports")
    if source is None:
        return

    conn.execute(
        sqlalchemy.text(
            f'INSERT INTO "dataset_report" ('
            f'"dataset_id", "document_statistics", "word_statistics", '
            f'"most_common_words", "least_common_words") '
            f'SELECT d."id", src."document_statistics", src."word_statistics", '
            f'src."most_common_words", src."least_common_words" '
            f'FROM "{source}" src '
            f'JOIN "dataset" d ON d."name" = src."name" '
            f'WHERE NOT EXISTS ('
            f'SELECT 1 FROM "dataset_report" dr WHERE dr."dataset_id" = d."id")'
        )
    )


# -----------------------------------------------------------------------------
def _backfill_tokenization_document_stats(conn: Connection) -> None:
    source = _table_source(conn, "tokenization_local_stats")
    if source is None:
        return

    conn.execute(
        sqlalchemy.text(
            f'INSERT INTO "tokenization_document_stats" ('
            f'"tokenizer_id", "document_id", "tokens_count", "tokens_to_words_ratio", '
            f'"bytes_per_token", "boundary_preservation_rate", '
            f'"round_trip_token_fidelity", "round_trip_text_fidelity", '
            f'"determinism_stability", "bytes_per_character") '
            f'SELECT t."id", src."text_id", src."tokens_count", '
            f'src."tokens_to_words_ratio", src."bytes_per_token", '
            f'src."boundary_preservation_rate", src."round_trip_token_fidelity", '
            f'src."round_trip_text_fidelity", src."determinism_stability", '
            f'src."bytes_per_character" '
            f'FROM "{source}" src '
            f'JOIN "tokenizer" t ON t."name" = src."tokenizer" '
            f'JOIN "dataset_document" dd ON dd."id" = src."text_id" '
            f'WHERE NOT EXISTS ('
            f'SELECT 1 FROM "tokenization_document_stats" tds '
            f'WHERE tds."tokenizer_id" = t."id" '
            f'AND tds."document_id" = src."text_id")'
        )
    )


# -----------------------------------------------------------------------------
def _backfill_tokenization_dataset_stats(conn: Connection) -> None:
    source = _table_source(conn, "tokenization_global_stats")
    if source is None:
        return

    conn.execute(
        sqlalchemy.text(
            f'INSERT INTO "tokenization_dataset_stats" ('
            f'"tokenizer_id", "dataset_id", "tokenization_speed_tps", '
            f'"throughput_chars_per_sec", "model_size_mb", "vocabulary_size", '
            f'"subword_fertility", "oov_rate", "word_recovery_rate") '
            f'SELECT t."id", d."id", src."tokenization_speed_tps", '
            f'src."throughput_chars_per_sec", src."model_size_mb", '
            f'src."vocabulary_size", src."subword_fertility", src."oov_rate", '
            f'src."word_recovery_rate" '
            f'FROM "{source}" src '
            f'JOIN "tokenizer" t ON t."name" = src."tokenizer" '
            f'JOIN "dataset" d ON d."name" = src."name" '
            f'WHERE NOT EXISTS ('
            f'SELECT 1 FROM "tokenization_dataset_stats" g '
            f'WHERE g."tokenizer_id" = t."id" '
            f'AND g."dataset_id" = d."id")'
        )
    )

    conn.execute(
        sqlalchemy.text(
            f'INSERT INTO "tokenization_dataset_stats_detail" ('
            f'"global_stats_id", "character_coverage", "segmentation_consistency", '
            f'"determinism_rate", "token_distribution_entropy", "rare_token_tail_1", '
            f'"rare_token_tail_2", "boundary_preservation_rate", '
            f'"compression_chars_per_token", "compression_bytes_per_character", '
            f'"round_trip_fidelity_rate", "round_trip_text_fidelity_rate", '
            f'"token_id_ordering_monotonicity", "token_unigram_coverage") '
            f'SELECT g."id", src."character_coverage", src."segmentation_consistency", '
            f'src."determinism_rate", src."token_distribution_entropy", '
            f'src."rare_token_tail_1", src."rare_token_tail_2", '
            f'src."boundary_preservation_rate", src."compression_chars_per_token", '
            f'src."compression_bytes_per_character", src."round_trip_fidelity_rate", '
            f'src."round_trip_text_fidelity_rate", '
            f'src."token_id_ordering_monotonicity", src."token_unigram_coverage" '
            f'FROM "{source}" src '
            f'JOIN "tokenizer" t ON t."name" = src."tokenizer" '
            f'JOIN "dataset" d ON d."name" = src."name" '
            f'JOIN "tokenization_dataset_stats" g '
            f'ON g."tokenizer_id" = t."id" AND g."dataset_id" = d."id" '
            f'WHERE NOT EXISTS ('
            f'SELECT 1 FROM "tokenization_dataset_stats_detail" gd '
            f'WHERE gd."global_stats_id" = g."id")'
        )
    )


# -----------------------------------------------------------------------------
def _backfill_tokenizer_vocabulary(conn: Connection) -> None:
    stats_source = _table_source(conn, "vocabulary_statistics")
    if stats_source is not None:
        conn.execute(
            sqlalchemy.text(
                f'INSERT INTO "tokenizer_vocabulary_statistics" ('
                f'"tokenizer_id", "vocabulary_size", "decoded_tokens", '
                f'"number_shared_tokens", "number_unshared_tokens", '
                f'"percentage_subwords", "percentage_true_words") '
                f'SELECT t."id", src."vocabulary_size", src."decoded_tokens", '
                f'src."number_shared_tokens", src."number_unshared_tokens", '
                f'src."percentage_subwords", src."percentage_true_words" '
                f'FROM "{stats_source}" src '
                f'JOIN "tokenizer" t ON t."name" = src."tokenizer" '
                f'WHERE NOT EXISTS ('
                f'SELECT 1 FROM "tokenizer_vocabulary_statistics" tvs '
                f'WHERE tvs."tokenizer_id" = t."id")'
            )
        )

    vocabulary_source = _table_source(conn, "vocabulary")
    if vocabulary_source is None:
        return

    conn.execute(
        sqlalchemy.text(
            f'INSERT INTO "tokenizer_vocabulary" ('
            f'"tokenizer_id", "token_id", "vocabulary_tokens", "decoded_tokens") '
            f'SELECT t."id", src."token_id", src."vocabulary_tokens", src."decoded_tokens" '
            f'FROM "{vocabulary_source}" src '
            f'JOIN "tokenizer" t ON t."name" = src."tokenizer" '
            f'WHERE NOT EXISTS ('
            f'SELECT 1 FROM "tokenizer_vocabulary" tv '
            f'WHERE tv."tokenizer_id" = t."id" '
            f'AND tv."token_id" = src."token_id")'
        )
    )


# -----------------------------------------------------------------------------
def _reset_postgres_sequences(conn: Connection) -> None:
    if conn.dialect.name != "postgresql":
        return

    for table_name in (
        "dataset",
        "tokenizer",
        "dataset_document",
        "tokenization_document_stats",
        "tokenization_dataset_stats",
        "tokenizer_vocabulary",
    ):
        if not _table_exists(conn, table_name):
            continue
        max_id = conn.execute(
            sqlalchemy.text(f'SELECT COALESCE(MAX("id"), 0) FROM "{table_name}"')
        ).scalar()
        next_value = int(max(max_id or 0, 1))
        conn.execute(
            sqlalchemy.text(
                "SELECT setval(pg_get_serial_sequence(:table_name, 'id'), :next_value, true)"
            ),
            {"table_name": table_name, "next_value": next_value},
        )


# -----------------------------------------------------------------------------
def _create_legacy_compatibility_views(conn: Connection) -> None:
    for view_name, select_sql in LEGACY_VIEW_SQL.items():
        if not _is_valid_identifier(view_name):
            raise ValueError("Invalid view name in compatibility definitions.")
        if _relation_exists(conn, view_name):
            if _table_exists(conn, view_name):
                logger.warning(
                    "Skipping compatibility view %s because a table with that name exists.",
                    view_name,
                )
                continue
            conn.execute(sqlalchemy.text(f'DROP VIEW IF EXISTS "{view_name}"'))
        conn.execute(sqlalchemy.text(f'CREATE VIEW "{view_name}" AS {select_sql}'))


# -----------------------------------------------------------------------------
def run_schema_migration(engine: Engine) -> None:
    with engine.begin() as conn:
        _rename_legacy_tables(conn)
        _backfill_dimensions(conn)
        _backfill_dataset_documents(conn)
        _backfill_document_statistics(conn)
        _backfill_dataset_reports(conn)
        _backfill_tokenization_document_stats(conn)
        _backfill_tokenization_dataset_stats(conn)
        _backfill_tokenizer_vocabulary(conn)
        _reset_postgres_sequences(conn)
        _create_legacy_compatibility_views(conn)

