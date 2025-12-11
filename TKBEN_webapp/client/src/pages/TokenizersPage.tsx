import { useMemo } from 'react';
import { useTokenizers } from '../contexts/TokenizersContext';

const TokenizersPage = () => {
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
    selectedPlot,
    customTokenizerInputRef,
    setSelectedTokenizer,
    setTokenizers,
    setMaxDocuments,
    setSelectedDataset,
    setSelectedPlot,
    setScanError,
    setBenchmarkError,
    addTokenizer,
    handleScan,
    handleRunBenchmarks,
    handleDownloadPlot,
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
                <label htmlFor="dataset-select">Dataset:</label>
                <div className="dataset-select-row">
                  <select
                    id="dataset-select"
                    className="text-input"
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    disabled={datasetsLoading}
                  >
                    {availableDatasets.length === 0 ? (
                      <option value="">
                        {datasetsLoading ? 'Loading...' : 'Click Refresh to load'}
                      </option>
                    ) : (
                      availableDatasets.map((dataset) => (
                        <option key={dataset} value={dataset}>
                          {dataset}
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
                    {datasetsLoading ? '...' : 'Refresh'}
                  </button>
                </div>
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
