import { useCallback, useState } from 'react';
import { downloadDataset } from '../services/datasetsApi';
import type { DatasetDownloadResponse, HistogramData } from '../types/api';

const datasetOptions = [
  { corpus: 'wikitext', configs: ['wikitext-103-v1', 'wikitext-2-v1'] },
  { corpus: 'openwebtext', configs: ['default'] },
  { corpus: 'c4', configs: ['en', 'realnewslike'] },
];

interface DatasetStats {
  documentCount: number;
  meanLength: number;
  medianLength: number;
  minLength: number;
  maxLength: number;
}

const DatasetPage = () => {
  const [useCustom, setUseCustom] = useState(false);
  const [selectedCorpus, setSelectedCorpus] = useState(datasetOptions[0].corpus);
  const [selectedConfig, setSelectedConfig] = useState(datasetOptions[0].configs[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datasetLoaded, setDatasetLoaded] = useState(false);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [histogram, setHistogram] = useState<HistogramData | null>(null);

  const handleCorpusChange = (value: string) => {
    setSelectedCorpus(value);
    const option = datasetOptions.find((item) => item.corpus === value);
    if (option) {
      setSelectedConfig(option.configs[0]);
    }
    // Reset loaded state when corpus changes
    setDatasetLoaded(false);
    setStats(null);
    setHistogram(null);
  };

  const handleConfigChange = (value: string) => {
    setSelectedConfig(value);
    setDatasetLoaded(false);
    setStats(null);
    setHistogram(null);
  };

  const handleLoadDataset = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response: DatasetDownloadResponse = await downloadDataset({
        corpus: selectedCorpus,
        config: selectedConfig,
      });

      setStats({
        documentCount: response.document_count,
        meanLength: response.histogram.mean_length,
        medianLength: response.histogram.median_length,
        minLength: response.histogram.min_length,
        maxLength: response.histogram.max_length,
      });
      setHistogram(response.histogram);
      setDatasetLoaded(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dataset');
    } finally {
      setLoading(false);
    }
  }, [selectedCorpus, selectedConfig]);

  const formatNumber = (num: number) => {
    return num.toLocaleString();
  };

  // Compute histogram bar heights relative to max count
  const renderHistogram = () => {
    if (loading) {
      return (
        <div className="loading-container">
          <div className="spinner" />
          <p>Processing dataset...</p>
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
                key={index}
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

  return (
    <div className="page-scroll">
      <div className="page-grid dataset-page">
        <section className="panel">
          <header className="panel-header">
            <div>
              <p className="panel-label">Select dataset</p>
              <p className="panel-description">
                Open access datasets are identified by corpus and configuration.
              </p>
            </div>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={useCustom}
                onChange={(event) => setUseCustom(event.target.checked)}
              />
              <span>Use custom dataset</span>
            </label>
          </header>
          <div className="panel-body">
            {error && (
              <div className="error-banner">
                <span>{error}</span>
                <button onClick={() => setError(null)}>×</button>
              </div>
            )}
            <div className="input-stack">
              <label className="field-label">Corpus</label>
              <input
                className="text-input"
                value={selectedCorpus}
                onChange={(event) => handleCorpusChange(event.target.value)}
                disabled={useCustom || loading}
              />
            </div>
            <div className="input-stack">
              <label className="field-label">Configuration</label>
              <input
                className="text-input"
                value={selectedConfig}
                onChange={(event) => handleConfigChange(event.target.value)}
                disabled={useCustom || loading}
              />
            </div>
          </div>
          <footer className="panel-footer">
            <button
              type="button"
              className="primary-button"
              onClick={handleLoadDataset}
              disabled={loading || useCustom}
            >
              {loading ? 'Loading...' : 'Load dataset'}
            </button>
            <button
              type="button"
              className="primary-button ghost"
              disabled={!datasetLoaded || loading}
            >
              Analyze dataset
            </button>
          </footer>
        </section>
        <aside className="panel dashboard-panel">
          <header className="panel-header">
            <div>
              <p className="panel-label">Dataset overview</p>
              <p className="panel-description">
                {datasetLoaded
                  ? `Stats for ${selectedCorpus}/${selectedConfig}`
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
