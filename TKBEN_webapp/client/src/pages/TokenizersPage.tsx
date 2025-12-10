import { useMemo, useState } from 'react';
import { scanTokenizers } from '../services/tokenizersApi';

const TokenizersPage = () => {
  const [scanInProgress, setScanInProgress] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [fetchedTokenizers, setFetchedTokenizers] = useState<string[]>([]);
  const [selectedTokenizer, setSelectedTokenizer] = useState('');
  const [tokenizers, setTokenizers] = useState<string[]>([]);
  const [includeCustom, setIncludeCustom] = useState(false);
  const [includeNSL, setIncludeNSL] = useState(false);

  const chartStats = useMemo(
    () => [
      { label: 'Queued runs', value: tokenizers.length },
      { label: 'Avg. throughput', value: `${(tokenizers.length * 320).toLocaleString()} tok/s` },
      { label: 'Custom tokenizer', value: includeCustom ? 'yes' : 'no' },
    ],
    [tokenizers.length, includeCustom],
  );

  const addTokenizer = (value: string) => {
    if (!value || tokenizers.includes(value)) {
      return;
    }
    setTokenizers((list) => [...list, value]);
  };

  const removeTokenizer = (value: string) => {
    setTokenizers((list) => list.filter((item) => item !== value));
  };

  const handleScan = async () => {
    setScanInProgress(true);
    setScanError(null);

    try {
      // Call without limit to use server-configured default
      const response = await scanTokenizers();
      setFetchedTokenizers(response.identifiers);
      // Auto-select first tokenizer if available
      if (response.identifiers.length > 0 && !selectedTokenizer) {
        setSelectedTokenizer(response.identifiers[0]);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to scan tokenizers';
      setScanError(errorMessage);
      console.error('Scan error:', error);
    } finally {
      setScanInProgress(false);
    }
  };

  return (
    <div className="page-scroll">
      <div className="page-grid tokenizers-page">
        <section className="panel large-panel">
          <header className="panel-header">
            <div>
              <p className="panel-label">Select tokenizers</p>
              <p className="panel-description">
                Organize Hugging Face tokenizers using chips and fetch additional IDs via scan.
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
                {scanInProgress ? 'Scanning…' : 'Scan'}
              </button>
            </div>
          </header>
          {scanError && (
            <div className="error-banner">
              <span>{scanError}</span>
              <button type="button" onClick={() => setScanError(null)} aria-label="Dismiss error">
                ×
              </button>
            </div>
          )}
          <div className="panel-body">
            <div className="chip-container">
              {tokenizers.map((tokenizer) => (
                <span key={tokenizer} className="chip">
                  {tokenizer}
                  <button type="button" onClick={() => removeTokenizer(tokenizer)} aria-label={`Remove ${tokenizer}`}>
                    ×
                  </button>
                </span>
              ))}
            </div>
            <textarea
              className="tokenizer-textarea"
              value={tokenizers.join('\n')}
              onChange={(event) => setTokenizers(event.target.value.split('\n').filter(Boolean))}
            />
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
            <button type="button" className="primary-button">
              Run benchmarks
            </button>
            <button type="button" className="primary-button ghost">
              Generate visual benchmarks
            </button>
          </footer>
        </section>
        <aside className="panel dashboard-panel">
          <header className="panel-header">
            <div>
              <p className="panel-label">Benchmark status</p>
              <p className="panel-description">
                Visual placeholders for results and runtime statistics.
              </p>
            </div>
          </header>
          <div className="chart-placeholder">
            <canvas width="320" height="180" />
            <p>Plot canvas ready for rendering charts.</p>
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
            <li>Tokenizer chips ensure quick toggling without editing long lists.</li>
            <li>Scanning keeps identifiers synchronized with Hugging Face.</li>
            <li>Canvas can render throughput plots once backend APIs deliver data.</li>
          </ul>
        </aside>
      </div>
    </div>
  );
};

export default TokenizersPage;
