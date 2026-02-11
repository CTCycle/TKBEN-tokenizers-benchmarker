from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from TKBEN.server.repositories.database.backend import database
from TKBEN.server.configurations.server import server_settings
from TKBEN.server.utils.constants import (
    API_ROUTE_BROWSER_DATA,
    API_ROUTE_BROWSER_TABLES,
    API_ROUTER_PREFIX_BROWSER,
)
from TKBEN.server.utils.logger import logger

router = APIRouter(prefix=API_ROUTER_PREFIX_BROWSER, tags=["browser"])


# Mapping from table names to friendly display names
TABLE_DISPLAY_NAMES: dict[str, str] = {
    "dataset": "Datasets",
    "dataset_document": "Dataset Documents",
    "dataset_document_statistics": "Dataset Document Statistics",
    "dataset_report": "Dataset Reports",
    "tokenizer": "Tokenizers",
    "tokenization_document_stats": "Tokenization Document Statistics",
    "tokenization_dataset_stats": "Tokenization Dataset Stats (Core)",
    "tokenization_dataset_stats_detail": "Tokenization Dataset Stats (Detail)",
    "tokenizer_vocabulary": "Tokenizer Vocabulary",
    "tokenizer_vocabulary_statistics": "Tokenizer Vocabulary Statistics",
    "tokenization_local_stats": "Tokenization Local Statistics",
    "tokenization_global_stats": "Tokenization Global Stats",
    "vocabulary_statistics": "Vocabulary Statistics",
    "vocabulary": "Vocabulary",
    "text_dataset": "Text Datasets",
    "text_dataset_statistics": "Text Dataset Statistics",
    "text_dataset_reports": "Text Dataset Reports",
}


###############################################################################
@router.get(API_ROUTE_BROWSER_TABLES)
async def list_tables() -> dict:
    """
    List all available tables in the database with friendly names.
    
    Returns:
        Dictionary with table mappings (technical name -> display name)
    """
    try:
        table_names = database.get_table_names()
        table_names = [name for name in table_names if not name.endswith("_legacy")]
        tables = []
        for name in table_names:
            display_name = TABLE_DISPLAY_NAMES.get(name, name)
            tables.append({
                "name": name,
                "display_name": display_name,
            })
        return {"tables": tables}
    except Exception as e:
        logger.error("Failed to list tables: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tables: {str(e)}",
        )


###############################################################################
@router.get(API_ROUTE_BROWSER_DATA)
async def get_table_data(
    table: str = Query(..., description="Table name to fetch data from"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int | None = Query(None, ge=1, description="Limit for pagination (defaults to config)"),
) -> dict:
    """
    Fetch paginated data from a table.
    
    Args:
        table: The table name to query
        offset: Starting row offset (0-based)
        limit: Maximum number of rows to return (defaults to browse_batch_size config)
    
    Returns:
        Dictionary with columns, data rows, statistics, and has_more flag
    """
    try:
        # Use configured batch size if limit not specified
        effective_limit = limit or server_settings.database.browse_batch_size
        
        # Fetch paginated data
        df = database.load_paginated(table, offset, effective_limit)
        
        # Get statistics
        total_rows = database.count_rows(table)
        column_count = database.get_column_count(table)
        
        # Check if there are more rows
        has_more = (offset + len(df)) < total_rows
        
        # Convert DataFrame to list of dicts for JSON serialization
        columns = list(df.columns)
        data = df.to_dict(orient="records")
        
        return {
            "table": table,
            "columns": columns,
            "data": data,
            "statistics": {
                "total_rows": total_rows,
                "column_count": column_count,
                "rows_returned": len(data),
                "offset": offset,
            },
            "has_more": has_more,
        }
    except Exception as e:
        logger.error("Failed to fetch table data for %s: %s", table, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch table data: {str(e)}",
        )
