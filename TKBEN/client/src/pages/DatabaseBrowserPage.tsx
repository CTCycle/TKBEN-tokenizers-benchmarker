import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchTableData, fetchTables } from '../services/databaseApi';
import type { TableDataResponse, TableInfo } from '../services/databaseApi';

const DatabaseBrowserPage = () => {
    const [tables, setTables] = useState<TableInfo[]>([]);
    const [selectedTable, setSelectedTable] = useState<string>('');
    const [tableData, setTableData] = useState<TableDataResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [loadingMore, setLoadingMore] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const tableContainerRef = useRef<HTMLDivElement>(null);

    // Load available tables on mount
    useEffect(() => {
        const loadTables = async () => {
            try {
                const response = await fetchTables();
                setTables(response.tables);
                if (response.tables.length > 0) {
                    setSelectedTable(response.tables[0].name);
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load tables');
            }
        };
        loadTables();
    }, []);

    // Fetch data when table selection changes
    const handleTableChange = async (tableName: string) => {
        setSelectedTable(tableName);
        setTableData(null);
        setError(null);

        // Auto-fetch data for the new table
        setLoading(true);
        try {
            const response = await fetchTableData(tableName, 0);
            setTableData(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load table data');
        } finally {
            setLoading(false);
        }
    };

    // Fetch data for the selected table
    const handleRefresh = useCallback(async () => {
        if (!selectedTable) return;

        setLoading(true);
        setError(null);
        try {
            const response = await fetchTableData(selectedTable, 0);
            setTableData(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load table data');
        } finally {
            setLoading(false);
        }
    }, [selectedTable]);

    // Load more data when scrolling to bottom
    const handleLoadMore = useCallback(async () => {
        if (!selectedTable || !tableData || !tableData.has_more || loadingMore) return;

        setLoadingMore(true);
        try {
            const newOffset = tableData.statistics.offset + tableData.statistics.rows_returned;
            const response = await fetchTableData(selectedTable, newOffset);
            setTableData(prev => {
                if (!prev) return response;
                return {
                    ...response,
                    data: [...prev.data, ...response.data],
                    statistics: {
                        ...response.statistics,
                        rows_returned: prev.statistics.rows_returned + response.statistics.rows_returned,
                    },
                };
            });
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load more data');
        } finally {
            setLoadingMore(false);
        }
    }, [selectedTable, tableData, loadingMore]);

    // Handle scroll to detect when user reaches bottom
    const handleScroll = useCallback(() => {
        const container = tableContainerRef.current;
        if (!container) return;

        const { scrollTop, scrollHeight, clientHeight } = container;
        // Load more when within 100px of bottom
        if (scrollHeight - scrollTop - clientHeight < 100) {
            handleLoadMore();
        }
    }, [handleLoadMore]);

    const formatNumber = (num: number) => num.toLocaleString();

    const getDisplayName = (tableName: string) => {
        const table = tables.find(t => t.name === tableName);
        return table?.display_name || tableName;
    };

    const buildRowKey = (row: Record<string, unknown>) => {
        const keyFromColumns = tableData?.columns
            .map((col) => String(row[col] ?? ''))
            .join('|');
        return keyFromColumns && keyFromColumns.trim().length > 0
            ? keyFromColumns
            : JSON.stringify(row);
    };

    const renderTableContent = () => {
        if (loading) {
            return (
                <div className="loading-container">
                    <div className="spinner" />
                    <p>Loading table data...</p>
                </div>
            );
        }

        if (tableData) {
            return (
                <>
                    <table className="db-table">
                        <thead>
                            <tr>
                                {tableData.columns.map((col) => (
                                    <th key={col}>{col}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {tableData.data.map((row) => {
                                const rowKey = buildRowKey(row);
                                return (
                                    <tr key={rowKey}>
                                        {tableData.columns.map((col) => (
                                            <td key={col}>
                                                {String(row[col] ?? '')}
                                            </td>
                                        ))}
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                    {loadingMore && (
                        <div className="db-loading-more">
                            <div className="spinner" />
                            <span>Loading more...</span>
                        </div>
                    )}
                    {!tableData.has_more && tableData.data.length > 0 && (
                        <div className="db-end-message">
                            All {formatNumber(tableData.statistics.total_rows)} rows loaded
                        </div>
                    )}
                </>
            );
        }

        return (
            <div className="db-empty-state">
                <p>Select a table to view data</p>
            </div>
        );
    };

    return (
        <div className="db-browser-page">
            <section className="panel db-browser-panel">
                <header className="panel-header">
                    <div>
                        <p className="panel-label">Database Browser</p>
                        <p className="panel-description">
                            Browse data from database tables.
                        </p>
                    </div>
                </header>

                <div className="db-browser-controls">
                    <div className="db-browser-top-row">
                        <div className="db-browser-select-row">
                            <label className="field-label" htmlFor="table-select">Select Table</label>
                            <select
                                id="table-select"
                                className="text-input"
                                value={selectedTable}
                                onChange={(e) => handleTableChange(e.target.value)}
                                disabled={loading}
                            >
                                {tables.map((table) => (
                                    <option key={table.name} value={table.name}>
                                        {table.display_name}
                                    </option>
                                ))}
                            </select>
                            <button
                                type="button"
                                className="primary-button"
                                onClick={handleRefresh}
                                disabled={loading || !selectedTable}
                            >
                                {loading ? 'Loading...' : 'Refresh'}
                            </button>
                        </div>

                        {tableData && (
                            <div className="db-browser-stats">
                                <span className="db-stat-item">
                                    <strong>Rows:</strong> {formatNumber(tableData.statistics.total_rows)}
                                </span>
                                <span className="db-stat-item">
                                    <strong>Columns:</strong> {tableData.statistics.column_count}
                                </span>
                                <span
                                    className="db-stat-item db-stat-item--table"
                                    title={getDisplayName(selectedTable)}
                                >
                                    <strong>Table:</strong> {getDisplayName(selectedTable)}
                                </span>
                            </div>
                        )}
                    </div>
                </div>

                {error && (
                    <div className="error-banner">
                        <span>{error}</span>
                        <button onClick={() => setError(null)}>×</button>
                    </div>
                )}

                <div
                    className="db-table-container"
                    ref={tableContainerRef}
                    onScroll={handleScroll}
                >
                    {renderTableContent()}
                </div>
            </section>
        </div>
    );
};

export default DatabaseBrowserPage;
