const API_BASE_URL = '/api';

export interface TableInfo {
    name: string;
    display_name: string;
}

export interface TableListResponse {
    tables: TableInfo[];
}

export interface TableStatistics {
    total_rows: number;
    column_count: number;
    rows_returned: number;
    offset: number;
}

export interface TableDataResponse {
    table: string;
    columns: string[];
    data: Record<string, unknown>[];
    statistics: TableStatistics;
    has_more: boolean;
}

/**
 * Fetch list of available tables in the database.
 * @returns Promise with the list of tables
 */
export async function fetchTables(): Promise<TableListResponse> {
    const response = await fetch(`${API_BASE_URL}/browser/tables`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to fetch tables: ${response.status}`);
    }

    return response.json();
}

/**
 * Fetch paginated data from a table.
 * @param table - Table name to fetch from
 * @param offset - Starting row offset
 * @param limit - Maximum number of rows to return (optional, uses server default)
 * @returns Promise with table data and statistics
 */
export async function fetchTableData(
    table: string,
    offset: number = 0,
    limit?: number
): Promise<TableDataResponse> {
    const params = new URLSearchParams({
        table,
        offset: offset.toString(),
    });
    if (limit !== undefined) {
        params.append('limit', limit.toString());
    }

    const response = await fetch(`${API_BASE_URL}/browser/data?${params}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to fetch table data: ${response.status}`);
    }

    return response.json();
}
