import { useDataset } from '../contexts/DatasetContext';

const DatasetPage = () => {
  const {
    datasetName,
    selectedCorpus,
    selectedConfig,
    loading,
    error,
    datasetLoaded,
    stats,
    histogram,
    loadProgress,
    analyzing,
    analysisStats,
    analysisProgress,
    fileInputRef,
    availableDatasets,
    selectedAnalysisDataset,
    analysisDatasetLoading,
    setError,
    handleCorpusChange,
    handleConfigChange,
    handleLoadDataset,
    handleUploadClick,
    handleFileChange,
    handleAnalyzeDataset,
    setSelectedAnalysisDataset,
    refreshAvailableDatasets,
  } = useDataset();

  const corpusInputId = 'corpus-input';
  const configInputId = 'config-input';
  const analysisDatasetSelectId = 'analysis-dataset-select';

  const formatNumber = (num: number) => {
    return num.toLocaleString();
  };

  // Compute histogram bar heights relative to max count
  const renderHistogram = () => {
    if (loading) {
      const progressLabel = loadProgress !== null ? ` (${Math.round(loadProgress)}%)` : '';
      return (
        <div className="loading-container">
          <div className="spinner" />
          <p>Processing dataset{progressLabel}...</p>
          <span>Downloading from HuggingFace, then saving to database. This may take several minutes for large datasets.</span>
        </div>
      );
    }

    if (!histogram || histogram.counts.length === 0) {
      return (
        <>
          <p>Dataset distribution preview</p>
          <span>Load a dataset to see document length distribution.</span>
        </>
      );
    }

    const maxCount = Math.max(...histogram.counts);

    return (
      <div className="histogram-container">
        <p className="histogram-title">Document Length Distribution</p>
        <div className="histogram-chart">
          {histogram.counts.map((count, index) => {
            const heightPercent = maxCount > 0 ? (count / maxCount) * 100 : 0;
            return (
              <div
                key={`${histogram.bins[index]}-${count}`}
                className="histogram-bar-wrapper"
                title={`${histogram.bins[index]}: ${formatNumber(count)} docs`}
              >
                <div
                  className="histogram-bar"
                  style={{ height: `${heightPercent}%` }}
                />
              </div>
            );
          })}
        </div>
        <div className="histogram-labels">
          <span>{formatNumber(histogram.min_length)} chars</span>
          <span>{formatNumber(histogram.max_length)} chars</span>
        </div>
      </div>
    );
  };

  const renderAnalysisContent = () => {
    if (analyzing) {
      const progressLabel = analysisProgress !== null ? ` (${Math.round(analysisProgress)}%)` : '';
      return (
        <div className="loading-container">
          <div className="spinner" />
          <p>Analyzing dataset{progressLabel}...</p>
          <span>Computing word counts and word length statistics for each document.</span>
        </div>
      );
    }

    if (!analysisStats) {
      return null;
    }

    return (
      <>
        <div className="dashboard-grid">
          <div className="stat-card">
            <p className="stat-label">Analyzed Docs</p>
            <p className="stat-value">
              {formatNumber(analysisStats.total_documents)}
            </p>
          </div>
          <div className="stat-card">
            <p className="stat-label">Mean Words/Doc</p>
            <p className="stat-value">
              {formatNumber(Math.round(analysisStats.mean_words_count))}
            </p>
          </div>
          <div className="stat-card">
            <p className="stat-label">Median Words/Doc</p>
            <p className="stat-value">
              {formatNumber(Math.round(analysisStats.median_words_count))}
            </p>
          </div>
        </div>
        <ul className="insights-list">
          <li>
            Average word length: <strong>{analysisStats.mean_avg_word_length.toFixed(2)}</strong> characters
          </li>
          <li>
            Word length variability: <strong>{analysisStats.mean_std_word_length.toFixed(2)}</strong> (std dev)
          </li>
        </ul>
      </>
    );
  };

  return (
    <div className="page-scroll">
      <div className="page-grid dataset-page-layout">
        {/* Left column: Load + Analysis */}
        <div className="dataset-left-column">
          <section className="panel">
            <header className="panel-header">
              <div>
                <p className="panel-label">Download dataset</p>
                <p className="panel-description">
                  Download from HuggingFace or upload a custom CSV/Excel file.
                </p>
              </div>
            </header>
            <div className="panel-body">
              {error && (
                <div className="error-banner">
                  <span>{error}</span>
                  <button onClick={() => setError(null)}>×</button>
                </div>
              )}
              <div className="input-stack">
                <label className="field-label" htmlFor={corpusInputId}>
                  Corpus
                </label>
                <input
                  id={corpusInputId}
                  className="text-input"
                  value={selectedCorpus}
                  onChange={(event) => handleCorpusChange(event.target.value)}
                  disabled={loading}
                />
              </div>
              <div className="input-stack">
                <label className="field-label" htmlFor={configInputId}>
                  Configuration
                </label>
                <input
                  id={configInputId}
                  className="text-input"
                  value={selectedConfig}
                  onChange={(event) => handleConfigChange(event.target.value)}
                  disabled={loading}
                />
              </div>
            </div>
            <footer className="panel-footer">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".csv,.xlsx,.xls"
                style={{ display: 'none' }}
              />
              <button
                type="button"
                className="primary-button"
                onClick={handleLoadDataset}
                disabled={loading}
              >
                {loading ? 'Downloading...' : 'Download dataset'}
              </button>
              <button
                type="button"
                className="primary-button ghost"
                onClick={handleUploadClick}
                disabled={loading}
              >
                Upload custom dataset
              </button>
            </footer>
          </section>

          {/* Analysis Tools Section */}
          <section className="panel analysis-tools">
            <header className="panel-header">
              <div>
                <p className="panel-label">Analysis Tools</p>
                <p className="panel-description">
                  Analyze word-level statistics for any loaded dataset.
                </p>
              </div>
            </header>
            <div className="panel-body">
              <div className="input-stack">
                <label className="field-label" htmlFor={analysisDatasetSelectId}>
                  Select Dataset to Analyze
                </label>
                <div className="dataset-select-row">
                  <select
                    id={analysisDatasetSelectId}
                    className="text-input"
                    value={selectedAnalysisDataset}
                    onChange={(e) => setSelectedAnalysisDataset(e.target.value)}
                    disabled={analysisDatasetLoading || analyzing}
                  >
                    {availableDatasets.length === 0 ? (
                      <option value="">
                        {analysisDatasetLoading ? 'Loading...' : 'Click Refresh to load datasets'}
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
                    onClick={refreshAvailableDatasets}
                    disabled={analysisDatasetLoading}
                  >
                    {analysisDatasetLoading ? 'Loading...' : 'Refresh'}
                  </button>
                </div>
              </div>
            </div>
            <footer className="panel-footer">
              <button
                type="button"
                className="primary-button"
                disabled={!selectedAnalysisDataset || analyzing}
                onClick={handleAnalyzeDataset}
              >
                {analyzing ? 'Analyzing...' : 'Analyze Dataset'}
              </button>
            </footer>
          </section>

          {/* Analysis Results */}
          {(analyzing || analysisStats) && (
            <aside className="panel dashboard-panel">
              <header className="panel-header">
                <div>
                  <p className="panel-label">Word Analysis</p>
                  <p className="panel-description">
                    {analysisStats
                      ? `Word-level statistics for ${selectedAnalysisDataset}`
                      : 'Computing word-level statistics...'}
                  </p>
                </div>
              </header>
              {renderAnalysisContent()}
            </aside>
          )}
        </div>

        {/* Right column: Overview */}
        <aside className="panel dashboard-panel dataset-right-column">
          <header className="panel-header">
            <div>
              <p className="panel-label">Dataset overview</p>
              <p className="panel-description">
                {datasetLoaded && datasetName
                  ? `Stats for ${datasetName}`
                  : `Select and load a dataset to view stats`}
              </p>
            </div>
          </header>
          <div className="dashboard-grid">
            <div className="stat-card">
              <p className="stat-label">Documents</p>
              <p className="stat-value">
                {stats ? formatNumber(stats.documentCount) : '—'}
              </p>
            </div>
            <div className="stat-card">
              <p className="stat-label">Mean Length</p>
              <p className="stat-value">
                {stats ? formatNumber(Math.round(stats.meanLength)) : '—'}
              </p>
            </div>
            <div className="stat-card">
              <p className="stat-label">Median Length</p>
              <p className="stat-value">
                {stats ? formatNumber(Math.round(stats.medianLength)) : '—'}
              </p>
            </div>
          </div>
          <div className="chart-placeholder">
            {renderHistogram()}
          </div>
          <ul className="insights-list">
            <li>
              Pre-processing removes empty entries automatically before statistics are computed.
            </li>
            {stats && (
              <li>
                Document lengths range from{' '}
                <strong>{formatNumber(stats.minLength)}</strong> to{' '}
                <strong>{formatNumber(stats.maxLength)}</strong> characters.
              </li>
            )}
          </ul>
        </aside>
      </div>
    </div>
  );
};

export default DatasetPage;
