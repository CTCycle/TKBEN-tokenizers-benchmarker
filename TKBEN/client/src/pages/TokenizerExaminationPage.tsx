import TokenizersPage from './TokenizersPage';
import { useTokenizers } from '../contexts/TokenizersContext';

const formatLabel = (value: string) =>
  value.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());

const formatValue = (value: unknown) => {
  if (value === null || value === undefined) {
    return 'null';
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return '[]';
    }
    return value.join(', ');
  }
  if (typeof value === 'boolean') {
    return value ? 'yes' : 'no';
  }
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4);
  }
  return String(value);
};

const TokenizerExaminationPage = () => {
  const {
    tokenizerReport,
    tokenizerVocabulary,
    tokenizerVocabularyTotal,
    tokenizerVocabularyLoading,
    handleLoadMoreTokenizerVocabulary,
  } = useTokenizers();

  const histogram = tokenizerReport?.token_length_histogram ?? null;
  const maxCount = histogram?.counts.length ? Math.max(...histogram.counts) : 0;
  const globalStatsEntries = tokenizerReport
    ? Object.entries(tokenizerReport.global_stats ?? {})
    : [];
  const canLoadMore = tokenizerVocabulary.length < tokenizerVocabularyTotal;

  return (
    <div className="page-scroll">
      <div className="tokenizer-exam-layout">
        <div className="merged-page-row">
          <TokenizersPage showDashboard={false} embedded />
        </div>
        <aside className="panel dashboard-panel dashboard-plain tokenizer-report-dashboard">
          <header className="panel-header">
            <div>
              <p className="panel-label">Tokenizer Metadata Dashboard</p>
              <p className="panel-description">
                {tokenizerReport
                  ? `Report ${tokenizerReport.report_id} for ${tokenizerReport.tokenizer_name}`
                  : 'Run validation from tokenizer preview to populate this dashboard.'}
              </p>
            </div>
          </header>

          {!tokenizerReport && (
            <div className="chart-placeholder">
              <p>No tokenizer report loaded.</p>
              <span>Use the validation or load-latest buttons in tokenizer preview.</span>
            </div>
          )}

          {tokenizerReport && (
            <>
              <div className="tokenizer-meta-table-wrap">
                <table className="tokenizer-meta-table">
                  <tbody>
                    <tr>
                      <th>Description</th>
                      <td>{tokenizerReport.description ?? 'null'}</td>
                    </tr>
                    <tr>
                      <th>Vocabulary Size</th>
                      <td>{tokenizerReport.vocabulary_size.toLocaleString()}</td>
                    </tr>
                    {globalStatsEntries.map(([key, value]) => (
                      <tr key={key}>
                        <th>{formatLabel(key)}</th>
                        <td>{formatValue(value)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="chart-block">
                {!histogram || histogram.counts.length === 0 ? (
                  <div className="chart-placeholder">
                    <p>Token Length Histogram</p>
                    <span>No histogram data in this report.</span>
                  </div>
                ) : (
                  <div className="histogram-container">
                    <p className="histogram-title">Token Length Histogram (characters)</p>
                    <div className="histogram-chart">
                      {histogram.counts.map((count, index) => {
                        const heightPercent = maxCount > 0 ? (count / maxCount) * 100 : 0;
                        return (
                          <div
                            key={`${histogram.bins[index]}-${count}`}
                            className="histogram-bar-wrapper"
                            title={`${histogram.bins[index]}: ${count.toLocaleString()} tokens`}
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
                      <span>{histogram.min_length.toLocaleString()} chars</span>
                      <span>{histogram.max_length.toLocaleString()} chars</span>
                    </div>
                  </div>
                )}
              </div>

              <div className="tokenizer-vocabulary-panel">
                <div className="tokenizer-vocabulary-header">
                  <p className="panel-label">Vocabulary</p>
                  <span className="panel-description">
                    {tokenizerVocabulary.length.toLocaleString()} / {tokenizerVocabularyTotal.toLocaleString()} loaded
                  </span>
                </div>
                <div className="tokenizer-vocabulary-list">
                  {tokenizerVocabulary.map((item) => (
                    <div key={`${item.token_id}-${item.token}`} className="tokenizer-vocabulary-row">
                      <span className="tokenizer-vocabulary-id">{item.token_id}</span>
                      <span className="tokenizer-vocabulary-token">{item.token}</span>
                      <span className="tokenizer-vocabulary-length">{item.length}</span>
                    </div>
                  ))}
                  {tokenizerVocabulary.length === 0 && (
                    <div className="word-frequency-empty">No vocabulary rows loaded.</div>
                  )}
                </div>
                <div className="tokenizer-vocabulary-actions">
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => void handleLoadMoreTokenizerVocabulary()}
                    disabled={!canLoadMore || tokenizerVocabularyLoading}
                  >
                    {tokenizerVocabularyLoading ? 'Loading...' : 'Load More'}
                  </button>
                </div>
              </div>
            </>
          )}
        </aside>
      </div>
    </div>
  );
};

export default TokenizerExaminationPage;
