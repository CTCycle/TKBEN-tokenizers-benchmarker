"""
E2E tests for database browser API endpoints.
Covers /browser/tables and /browser/data.
"""
from playwright.sync_api import APIRequestContext


def test_list_tables_returns_expected_names(api_context: APIRequestContext) -> None:
    """GET /browser/tables should return known table names."""
    response = api_context.get("/browser/tables")
    assert response.ok
    data = response.json()
    assert "tables" in data
    assert isinstance(data["tables"], list)

    names = {table["name"] for table in data["tables"]}
    assert "TEXT_DATASET" in names
    assert "TOKENIZATION_GLOBAL_METRICS" in names


def test_fetch_table_data_requires_table_param(
    api_context: APIRequestContext,
) -> None:
    """GET /browser/data without a table param should return 422."""
    response = api_context.get("/browser/data")
    assert response.status == 422


def test_fetch_table_data_returns_schema(
    api_context: APIRequestContext,
) -> None:
    """GET /browser/data should return columns, data, and statistics."""
    response = api_context.get("/browser/data?table=TEXT_DATASET&offset=0&limit=5")
    assert response.ok
    data = response.json()
    assert "columns" in data
    assert "data" in data
    assert "statistics" in data
    assert "has_more" in data
    assert isinstance(data["columns"], list)
    assert isinstance(data["data"], list)
    assert "total_rows" in data["statistics"]
    assert "column_count" in data["statistics"]
    assert "rows_returned" in data["statistics"]
    assert "offset" in data["statistics"]


def test_fetch_table_data_contains_uploaded_rows(
    api_context: APIRequestContext,
    uploaded_dataset: dict,
) -> None:
    """Uploaded datasets should be visible in the TEXT_DATASET table."""
    dataset_name = uploaded_dataset["dataset_name"]
    offset = 0
    limit = 1000

    while True:
        response = api_context.get(
            f"/browser/data?table=TEXT_DATASET&offset={offset}&limit={limit}"
        )
        assert response.ok
        data = response.json()
        rows = data.get("data", [])
        if any(row.get("dataset_name") == dataset_name for row in rows):
            return

        rows_returned = data.get("statistics", {}).get("rows_returned", len(rows))
        if rows_returned == 0 or not data.get("has_more"):
            break

        offset += rows_returned

    assert False, f"Dataset {dataset_name} not found in TEXT_DATASET"
