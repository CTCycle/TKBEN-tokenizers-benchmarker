from __future__ import annotations


CREATE_FREQUENCIES_TABLE = (
    "CREATE TABLE IF NOT EXISTS frequencies "
    "(token TEXT PRIMARY KEY, count INTEGER NOT NULL)"
)
CREATE_FREQUENCIES_COUNT_INDEX = (
    "CREATE INDEX IF NOT EXISTS ix_frequencies_count ON frequencies(count)"
)
UPSERT_FREQUENCY_COUNT = (
    "INSERT INTO frequencies (token, count) VALUES (?, ?) "
    "ON CONFLICT(token) DO UPDATE SET count = count + excluded.count"
)

SELECT_TOTAL_FREQUENCY_COUNT = "SELECT COALESCE(SUM(count), 0) FROM frequencies"
SELECT_UNIQUE_FREQUENCY_COUNT = "SELECT COUNT(*) FROM frequencies"
SELECT_ALL_TOKEN_COUNTS = "SELECT token, count FROM frequencies"
SELECT_TOP_K_TOKEN_COUNTS = (
    "SELECT token, count FROM frequencies ORDER BY count DESC, token ASC LIMIT ?"
)
SELECT_BOTTOM_K_TOKEN_COUNTS = (
    "SELECT token, count FROM frequencies ORDER BY count ASC, token ASC LIMIT ?"
)
SELECT_SUM_TOP_K_COUNTS = (
    "SELECT COALESCE(SUM(count), 0) "
    "FROM (SELECT count FROM frequencies ORDER BY count DESC, token ASC LIMIT ?)"
)
SELECT_SUM_BOTTOM_K_COUNTS = (
    "SELECT COALESCE(SUM(count), 0) "
    "FROM (SELECT count FROM frequencies ORDER BY count ASC, token ASC LIMIT ?)"
)
SELECT_LONGEST_K_TOKEN_COUNTS = (
    "SELECT token, count FROM frequencies "
    "ORDER BY LENGTH(token) DESC, token ASC LIMIT ?"
)
SELECT_SHORTEST_K_TOKEN_COUNTS = (
    "SELECT token, count FROM frequencies "
    "ORDER BY LENGTH(token) ASC, token ASC LIMIT ?"
)
SELECT_FREQUENCY_OF_FREQUENCY_COUNT = "SELECT COUNT(*) FROM frequencies WHERE count = ?"


def select_sorted_token_counts_query(descending: bool) -> str:
    order = "DESC" if descending else "ASC"
    return f"SELECT token, count FROM frequencies ORDER BY count {order}, token ASC"
