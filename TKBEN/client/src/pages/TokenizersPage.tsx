import { useMemo } from 'react';
import { useTokenizers } from '../contexts/TokenizersContext';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

type TokenizersPageProps = {
  showDashboard?: boolean;
  embedded?: boolean;
};

const TokenizersPage = ({ showDashboard = true, embedded = false }: TokenizersPageProps) => {
  const {
    scanInProgress,
    scanError,
    fetchedTokenizers,
    selectedTokenizer,
    tokenizers,
    customTokenizerName,
    customTokenizerUploading,
    maxDocuments,
    availableDatasets,
    selectedDataset,
    datasetsLoading,
    benchmarkInProgress,
    benchmarkError,
    benchmarkResult,
    benchmarkProgress,
    customTokenizerInputRef,
    setSelectedTokenizer,
    setTokenizers,
    setMaxDocuments,
    setSelectedDataset,
    setScanError,
    setBenchmarkError,
    addTokenizer,
    handleScan,
    handleRunBenchmarks,
    refreshDatasets,
    handleUploadCustomTokenizer,
    handleClearCustomTokenizer,
    triggerCustomTokenizerUpload,
  } = useTokenizers();

  const chartStats = useMemo(
    () => [
      { label: 'Queued runs', value: tokenizers.length + (customTokenizerName ? 1 : 0) },
      {
        label: 'Avg. throughput',
        value: benchmarkResult?.global_metrics?.[0]?.tokenization_speed_tps
          ? `${Math.round(benchmarkResult.global_metrics[0].tokenization_speed_tps).toLocaleString()} tok/s`
          : '0 tok/s'
      },
      { label: 'Custom tokenizer', value: customTokenizerName ? 'loaded' : 'none' },
    ],
    [tokenizers.length, customTokenizerName, benchmarkResult],
  );

  // Prepare vocabulary data for Recharts
  const vocabularyChartData = useMemo(() => {
    if (!benchmarkResult?.chart_data?.vocabulary_stats) return [];
    return benchmarkResult.chart_data.vocabulary_stats.map((stat) => ({
      name: stat.tokenizer.split('/').pop() || stat.tokenizer,
      'Vocabulary Size': stat.vocabulary_size,
      'Subwords': stat.subwords_count,
      'True Words': stat.true_words_count,
    }));
  }, [benchmarkResult]);

  // Prepare speed data for Recharts
  const speedChartData = useMemo(() => {
    if (!benchmarkResult?.chart_data?.speed_metrics) return [];
    return benchmarkResult.chart_data.speed_metrics.map((stat) => ({
      name: stat.tokenizer.split('/').pop() || stat.tokenizer,
      'Tokens/sec': Math.round(stat.tokens_per_second),
      'Chars/sec': Math.round(stat.chars_per_second),
    }));
  }, [benchmarkResult]);

  const pageContent = (
    <div className={`page-grid tokenizers-page${showDashboard ? '' : ' tokenizers-page--single'}`}>
        <section className="panel large-panel">
          <header className="panel-header">
            <div>
              <p className="panel-label">Select tokenizers</p>
              <p className="panel-description">
                Manage Hugging Face tokenizer IDs in a single text area and fetch additional IDs via scan.
              </p>
            </div>
            <div className="tokenizer-select">
              <select
                value={selectedTokenizer}
                onChange={(event) => setSelectedTokenizer(event.target.value)}
                className="text-input"
              >
                {fetchedTokenizers.length === 0 ? (
                  <option value="">Click Scan to fetch tokenizers</option>
                ) : (
                  fetchedTokenizers.map((tokenizer) => (
                    <option key={tokenizer} value={tokenizer}>
                      {tokenizer}
                    </option>
                  ))
                )}
              </select>
              <button
                type="button"
                className="primary-button ghost"
                onClick={() => addTokenizer(selectedTokenizer)}
                disabled={!selectedTokenizer}
              >
                Add
              </button>
              <button
                type="button"
                className="primary-button"
                onClick={handleScan}
                disabled={scanInProgress}
              >
                {scanInProgress ? 'Scanning...' : 'Scan'}
              </button>
            </div>
          </header>
          <div className="panel-body">
            {scanError && (
              <div className="error-banner" role="alert">
                <span>{scanError}</span>
                <button type="button" aria-label="Dismiss" onClick={() => setScanError(null)}>
                  ×
                </button>
              </div>
            )}
            {benchmarkError && (
              <div className="error-banner" role="alert">
                <span>{benchmarkError}</span>
                <button type="button" aria-label="Dismiss" onClick={() => setBenchmarkError(null)}>
                  ×
                </button>
              </div>
            )}
            <textarea
              className="text-area"
              placeholder="Paste tokenizer IDs here, one per line..."
              value={tokenizers.join('\n')}
              onChange={(event) =>
                setTokenizers(event.target.value.split('\n').filter((t) => t.trim()))
              }
            />
            <div className="field-row">
              <label className="field-label" htmlFor="datasetSelect">
                Dataset:
              </label>
              <select
                id="datasetSelect"
                value={selectedDataset}
                onChange={(event) => setSelectedDataset(event.target.value)}
                className="text-input"
                disabled={datasetsLoading}
              >
                {availableDatasets.length === 0 ? (
                  <option value="">Click Refresh to load</option>
                ) : (
                  availableDatasets.map((ds) => (
                    <option key={ds} value={ds}>
                      {ds}
                    </option>
                  ))
                )}
              </select>
              <button
                type="button"
                className="primary-button ghost"
                onClick={refreshDatasets}
                disabled={datasetsLoading}
              >
                {datasetsLoading ? 'Loading...' : 'Refresh'}
              </button>
            </div>
            <div className="field-row">
              <label className="field-label" htmlFor="maxDocs">
                Max documents:
              </label>
              <input
                id="maxDocs"
                type="number"
                className="text-input"
                value={maxDocuments}
                onChange={(event) => setMaxDocuments(Number(event.target.value))}
                min={0}
              />
            </div>
            <div className="checkbox-group">
              <div className="custom-tokenizer-row">
                <input
                  type="file"
                  ref={customTokenizerInputRef}
                  onChange={handleUploadCustomTokenizer}
                  accept=".json"
                  style={{ display: 'none' }}
                />
                <button
                  type="button"
                  className="primary-button ghost"
                  onClick={triggerCustomTokenizerUpload}
                  disabled={customTokenizerUploading}
                >
                  {customTokenizerUploading ? 'Uploading...' : 'Upload tokenizer.json'}
                </button>
                {customTokenizerName && (
                  <span className="custom-tokenizer-badge">
                    {customTokenizerName}
                    <button
                      type="button"
                      className="clear-button"
                      onClick={handleClearCustomTokenizer}
                      aria-label="Clear custom tokenizer"
                    >
                      ×
                    </button>
                  </span>
                )}
              </div>
            </div>
          </div>
          <footer className="panel-footer">
            <button
              type="button"
              className="primary-button"
              onClick={handleRunBenchmarks}
              disabled={benchmarkInProgress || (tokenizers.length === 0 && !customTokenizerName) || !selectedDataset}
            >
              {benchmarkInProgress ? 'Running benchmarks...' : 'Run Benchmarks'}
            </button>
          </footer>
        </section>
        {showDashboard && (
          <aside className="panel dashboard-panel">
            <header className="panel-header">
              <div>
                <p className="panel-label">Benchmark Dashboard</p>
                <p className="panel-description">
                  {benchmarkResult
                    ? `Processed ${benchmarkResult.documents_processed} documents with ${benchmarkResult.tokenizers_count} tokenizers`
                    : 'Charts will appear after running benchmarks.'}
                </p>
              </div>
            </header>

            {/* Vocabulary Size Chart */}
            {vocabularyChartData.length > 0 && (
              <div className="chart-container">
                <h3 className="chart-title">Vocabulary Size Comparison</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={vocabularyChartData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis type="number" stroke="#888" />
                    <YAxis dataKey="name" type="category" stroke="#888" width={100} tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #333' }}
                      labelStyle={{ color: '#fff' }}
                    />
                    <Legend />
                    <Bar dataKey="Vocabulary Size" fill="#4fc3f7" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Speed Comparison Chart */}
            {speedChartData.length > 0 && (
              <div className="chart-container">
                <h3 className="chart-title">Tokenization Speed</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={speedChartData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis type="number" stroke="#888" />
                    <YAxis dataKey="name" type="category" stroke="#888" width={100} tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #333' }}
                      labelStyle={{ color: '#fff' }}
                    />
                    <Legend />
                    <Bar dataKey="Tokens/sec" fill="#81c784" radius={[0, 4, 4, 0]} />
                    <Bar dataKey="Chars/sec" fill="#ffb74d" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* KPI Cards */}
            <div className="dashboard-grid">
              {chartStats.map((stat) => (
                <div key={stat.label} className="stat-card">
                  <p className="stat-label">{stat.label}</p>
                  <p className="stat-value">{stat.value}</p>
                </div>
              ))}
            </div>

            {/* Empty state with spinner */}
            {!benchmarkResult && (
              <div className="chart-empty-state">
                {benchmarkInProgress ? (
                  <>
                    <div className="spinner" />
                    <p>
                      Processing benchmarks
                      {benchmarkProgress !== null ? ` (${Math.round(benchmarkProgress)}%)` : ''}...
                    </p>
                    <span>Analyzing tokenizers and computing metrics. This may take a few minutes.</span>
                  </>
                ) : (
                  <p>Run benchmarks to see charts.</p>
                )}
              </div>
            )}

            <ul className="insights-list">
              <li>Tokenizer list stays synchronized with the text area for quick edits and bulk paste.</li>
              {benchmarkResult && (
                <li>Benchmarks complete! Charts are now interactive - hover for details.</li>
              )}
            </ul>
          </aside>
        )}
    </div>
  );

  if (embedded) {
    return pageContent;
  }

  return <div className="page-scroll">{pageContent}</div>;
};

export default TokenizersPage;
