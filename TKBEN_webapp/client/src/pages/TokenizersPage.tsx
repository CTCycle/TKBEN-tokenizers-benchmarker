import { useMemo } from 'react';
import { useTokenizers } from '../contexts/TokenizersContext';

const TokenizersPage = () => {
  const {
    scanInProgress,
    scanError,
    fetchedTokenizers,
    selectedTokenizer,
    tokenizers,
    includeCustom,
    includeNSL,
    maxDocuments,
    datasetName,
    benchmarkInProgress,
    benchmarkError,
    benchmarkResult,
    selectedPlot,
    setSelectedTokenizer,
    setTokenizers,
    setIncludeCustom,
    setIncludeNSL,
    setMaxDocuments,
    setDatasetName,
    setSelectedPlot,
    setScanError,
    setBenchmarkError,
    addTokenizer,
    handleScan,
    handleRunBenchmarks,
    handleDownloadPlot,
  } = useTokenizers();

  const chartStats = useMemo(
    () => [
      { label: 'Queued runs', value: tokenizers.length },
      {
        label: 'Avg. throughput',
        value: benchmarkResult?.global_metrics?.[0]?.tokenization_speed_tps
          ? `${Math.round(benchmarkResult.global_metrics[0].tokenization_speed_tps).toLocaleString()} tok/s`
          : '0 tok/s'
      },
      { label: 'Custom tokenizer', value: includeCustom ? 'yes' : 'no' },
    ],
    [tokenizers.length, includeCustom, benchmarkResult],
  );

  return (
    <div className="page-scroll">
      <div className="page-grid tokenizers-page">
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
                disabled={fetchedTokenizers.length === 0}
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
                disabled={!selectedTokenizer || fetchedTokenizers.length === 0}
              >
                Add
              </button>
              <button
                type="button"
                className="primary-button"
                onClick={handleScan}
                disabled={scanInProgress}
              >
                {scanInProgress ? 'Scanning.' : 'Scan'}
              </button>
            </div>
          </header>
          {(scanError || benchmarkError) && (
            <div className="error-banner">
              <span>{scanError || benchmarkError}</span>
              <button
                type="button"
                onClick={() => { setScanError(null); setBenchmarkError(null); }}
                aria-label="Dismiss error"
              >
                ×
              </button>
            </div>
          )}
          <div className="panel-body">
            <textarea
              className="tokenizer-textarea"
              value={tokenizers.join('\n')}
              onChange={(event) => setTokenizers(event.target.value.split('\n').filter(Boolean))}
              placeholder="Add tokenizer IDs here, one per line..."
            />
            <div className="benchmark-options">
              <div className="input-group">
                <label htmlFor="dataset-name">Dataset name:</label>
                <input
                  id="dataset-name"
                  type="text"
                  className="text-input"
                  value={datasetName}
                  onChange={(e) => setDatasetName(e.target.value)}
                  placeholder="e.g., wikitext/wikitext-2-raw-v1"
                />
              </div>
              <div className="input-group">
                <label htmlFor="max-documents">Max documents:</label>
                <input
                  id="max-documents"
                  type="number"
                  className="text-input"
                  value={maxDocuments}
                  onChange={(e) => setMaxDocuments(parseInt(e.target.value, 10) || 0)}
                  min={0}
                  step={100}
                />
              </div>
            </div>
            <div className="checkbox-group">
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={includeCustom}
                  onChange={(event) => setIncludeCustom(event.target.checked)}
                />
                <span>Include custom tokenizer</span>
              </label>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={includeNSL}
                  onChange={(event) => setIncludeNSL(event.target.checked)}
                />
                <span>Calculate Normalized Sequence Length (NSL)</span>
              </label>
            </div>
          </div>
          <footer className="panel-footer">
            <button
              type="button"
              className="primary-button"
              onClick={handleRunBenchmarks}
              disabled={benchmarkInProgress || tokenizers.length === 0}
            >
              {benchmarkInProgress ? 'Running benchmarks...' : 'Run Benchmarks'}
            </button>
          </footer>
        </section>
        <aside className="panel dashboard-panel">
          <header className="panel-header">
            <div>
              <p className="panel-label">Benchmark status</p>
              <p className="panel-description">
                {benchmarkResult
                  ? `Processed ${benchmarkResult.documents_processed} documents with ${benchmarkResult.tokenizers_count} tokenizers`
                  : 'Visual placeholders for results and runtime statistics.'}
              </p>
            </div>
          </header>
          <div className="chart-placeholder">
            {selectedPlot ? (
              <div className="plot-container">
                <img
                  src={`data:image/png;base64,${selectedPlot.data}`}
                  alt={selectedPlot.name}
                  style={{ maxWidth: '100%', height: 'auto' }}
                />
                <div className="plot-actions">
                  {benchmarkResult?.plots.map((plot) => (
                    <button
                      key={plot.name}
                      type="button"
                      className={`plot-tab ${selectedPlot?.name === plot.name ? 'active' : ''}`}
                      onClick={() => setSelectedPlot(plot)}
                    >
                      {plot.name.replace(/_/g, ' ')}
                    </button>
                  ))}
                  <button
                    type="button"
                    className="primary-button ghost"
                    onClick={() => selectedPlot && handleDownloadPlot(selectedPlot)}
                  >
                    Download
                  </button>
                </div>
              </div>
            ) : (
              <>
                <canvas width="320" height="180" />
                <p>{benchmarkInProgress ? 'Processing benchmarks...' : 'Plot canvas ready for rendering charts.'}</p>
              </>
            )}
          </div>
          <div className="dashboard-grid">
            {chartStats.map((stat) => (
              <div key={stat.label} className="stat-card">
                <p className="stat-label">{stat.label}</p>
                <p className="stat-value">{stat.value}</p>
              </div>
            ))}
          </div>
          <ul className="insights-list">
            <li>Tokenizer list stays synchronized with the text area for quick edits and bulk paste.</li>
            <li>Scanning keeps identifiers synchronized with Hugging Face.</li>
            {benchmarkResult && (
              <li>Benchmarks complete! {benchmarkResult.plots.length} plots generated.</li>
            )}
          </ul>
        </aside>
      </div>
    </div>
  );
};

export default TokenizersPage;
