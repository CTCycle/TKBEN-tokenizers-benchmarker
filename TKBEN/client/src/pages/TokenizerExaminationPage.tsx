import TokenizersPage from './TokenizersPage';
import { useTokenizers } from '../contexts/TokenizersContext';
import type { TokenizerSubwordWordStats } from '../types/api';

const formatNumber = (value: number | null | undefined, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '0';
  }
  if (Number.isInteger(value)) {
    return value.toLocaleString();
  }
  return value.toFixed(digits);
};

const formatPercent = (value: number | null | undefined) => `${formatNumber(value, 2)}%`;

const toSubwordStats = (value: unknown): TokenizerSubwordWordStats | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const payload = value as Partial<TokenizerSubwordWordStats>;
  if (typeof payload.subword_count !== 'number' || typeof payload.word_count !== 'number') {
    return null;
  }
  return {
    heuristic: String(payload.heuristic ?? ''),
    subword_count: payload.subword_count,
    word_count: payload.word_count,
    considered_count: Number(payload.considered_count ?? 0),
    subword_percentage: Number(payload.subword_percentage ?? 0),
    word_percentage: Number(payload.word_percentage ?? 0),
    subword_to_word_ratio:
      payload.subword_to_word_ratio === null || payload.subword_to_word_ratio === undefined
        ? null
        : Number(payload.subword_to_word_ratio),
  };
};

const TokenizerExaminationPage = () => {
  const {
    tokenizerReport,
    tokenizerVocabulary,
    tokenizerVocabularyOffset,
    tokenizerVocabularyLimit,
    tokenizerVocabularyTotal,
    tokenizerVocabularyLoading,
    handleNextTokenizerVocabularyPage,
    handlePreviousTokenizerVocabularyPage,
    handleTokenizerVocabularyPageSizeChange,
  } = useTokenizers();

  const histogram = tokenizerReport?.token_length_histogram ?? null;
  const maxCount = histogram?.counts.length ? Math.max(...histogram.counts) : 0;

  const globalStats = tokenizerReport?.global_stats ?? {};
  const specialTokens = Array.isArray(globalStats.special_tokens)
    ? globalStats.special_tokens.map((item) => String(item))
    : [];
  const subwordStats = toSubwordStats(globalStats.subword_word_stats);

  const createdAtLabel = tokenizerReport?.created_at
    ? new Date(tokenizerReport.created_at).toLocaleString()
    : 'Not available';

  const vocabularyStart = tokenizerVocabularyTotal > 0 ? tokenizerVocabularyOffset + 1 : 0;
  const vocabularyEnd = tokenizerVocabularyTotal > 0
    ? Math.min(tokenizerVocabularyOffset + tokenizerVocabulary.length, tokenizerVocabularyTotal)
    : 0;
  const canGoPrevious = tokenizerVocabularyOffset > 0;
  const canGoNext = tokenizerVocabularyOffset + tokenizerVocabularyLimit < tokenizerVocabularyTotal;

  return (
    <div className="page-scroll">
      <div className="tokenizer-exam-layout">
        <div className="merged-page-row">
          <TokenizersPage showDashboard={false} embedded />
        </div>

        <div className="tokenizer-report-split">
          <section className="panel dashboard-panel dashboard-plain tokenizer-report-left">
            <header className="panel-header">
              <div>
                <p className="panel-label">Tokenizers Dashboard</p>
                <p className="panel-description">
                  {tokenizerReport
                    ? `Report ${tokenizerReport.report_id} for ${tokenizerReport.tokenizer_name}`
                    : 'Open a tokenizer report from the preview list to populate this dashboard.'}
                </p>
              </div>
            </header>

            {!tokenizerReport && (
              <div className="chart-placeholder">
                <p>No tokenizer report loaded.</p>
                <span>Use the report icon in tokenizer preview.</span>
              </div>
            )}

            {tokenizerReport && (
              <div className="tokenizer-report-grid">
                <article className="tokenizer-report-card">
                  <p className="panel-label">Report Overview</p>
                  <table className="tokenizer-meta-table">
                    <tbody>
                      <tr><th>Tokenizer</th><td>{tokenizerReport.tokenizer_name}</td></tr>
                      <tr><th>Created</th><td>{createdAtLabel}</td></tr>
                      <tr><th>Report ID</th><td>{tokenizerReport.report_id}</td></tr>
                      <tr><th>Report Version</th><td>{tokenizerReport.report_version}</td></tr>
                      <tr>
                        <th>Hugging Face</th>
                        <td>
                          {tokenizerReport.huggingface_url ? (
                            <a href={tokenizerReport.huggingface_url} target="_blank" rel="noreferrer">
                              {tokenizerReport.huggingface_url}
                            </a>
                          ) : 'Not available'}
                        </td>
                      </tr>
                      <tr>
                        <th>Description</th>
                        <td>{tokenizerReport.description ?? 'Not available'}</td>
                      </tr>
                    </tbody>
                  </table>
                </article>

                <article className="tokenizer-report-card">
                  <p className="panel-label">Tokenizer Metadata</p>
                  <table className="tokenizer-meta-table">
                    <tbody>
                      <tr><th>Vocabulary size</th><td>{formatNumber(tokenizerReport.vocabulary_size, 0)}</td></tr>
                      <tr><th>Algorithm</th><td>{String(globalStats.tokenizer_algorithm ?? 'Not available')}</td></tr>
                      <tr><th>Tokenizer class</th><td>{String(globalStats.tokenizer_class ?? 'Not available')}</td></tr>
                      <tr><th>do_lower_case</th><td>{String(globalStats.do_lower_case ?? 'Not available')}</td></tr>
                      <tr><th>Normalization</th><td>{String(globalStats.normalization_hint ?? 'Not available')}</td></tr>
                      <tr><th>Persistence mode</th><td>{String(globalStats.persistence_mode ?? 'Not available')}</td></tr>
                    </tbody>
                  </table>
                </article>

                <article className="tokenizer-report-card">
                  <p className="panel-label">Token Length Histogram</p>
                  {!histogram || histogram.counts.length === 0 ? (
                    <div className="chart-placeholder">
                      <p>No histogram data in this report.</p>
                    </div>
                  ) : (
                    <>
                      <div className="histogram-container">
                        <div className="histogram-chart">
                          {histogram.counts.map((count, index) => {
                            const heightPercent = maxCount > 0 ? (count / maxCount) * 100 : 0;
                            return (
                              <div
                                key={`${histogram.bins[index]}-${count}`}
                                className="histogram-bar-wrapper"
                                title={`${histogram.bins[index]}: ${count.toLocaleString()} tokens`}
                              >
                                <div className="histogram-bar" style={{ height: `${heightPercent}%` }} />
                              </div>
                            );
                          })}
                        </div>
                      </div>
                      <table className="tokenizer-meta-table tokenizer-meta-table-compact">
                        <tbody>
                          <tr><th>Min length</th><td>{formatNumber(histogram.min_length, 0)}</td></tr>
                          <tr><th>Max length</th><td>{formatNumber(histogram.max_length, 0)}</td></tr>
                          <tr><th>Mean length</th><td>{formatNumber(histogram.mean_length)}</td></tr>
                          <tr><th>Median length</th><td>{formatNumber(histogram.median_length)}</td></tr>
                        </tbody>
                      </table>
                    </>
                  )}
                </article>

                <article className="tokenizer-report-card">
                  <p className="panel-label">Vocabulary Composition</p>
                  <table className="tokenizer-meta-table tokenizer-meta-table-compact">
                    <tbody>
                      <tr><th>Special tokens count</th><td>{formatNumber(Number(globalStats.special_tokens_count ?? 0), 0)}</td></tr>
                      <tr><th>Has special tokens</th><td>{String(globalStats.has_special_tokens ?? false)}</td></tr>
                      <tr>
                        <th>Special tokens</th>
                        <td>{specialTokens.length > 0 ? specialTokens.join(', ') : 'None'}</td>
                      </tr>
                      <tr><th>Subword count</th><td>{formatNumber(subwordStats?.subword_count ?? 0, 0)}</td></tr>
                      <tr><th>Word count</th><td>{formatNumber(subwordStats?.word_count ?? 0, 0)}</td></tr>
                      <tr><th>Subword %</th><td>{formatPercent(subwordStats?.subword_percentage ?? 0)}</td></tr>
                      <tr><th>Word %</th><td>{formatPercent(subwordStats?.word_percentage ?? 0)}</td></tr>
                      <tr>
                        <th>Subword/word ratio</th>
                        <td>
                          {subwordStats?.subword_to_word_ratio === null
                            ? 'Not available'
                            : formatNumber(subwordStats?.subword_to_word_ratio ?? 0)}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </article>
              </div>
            )}
          </section>

          <aside className="panel dashboard-panel dashboard-plain tokenizer-report-right">
            <header className="panel-header">
              <div>
                <p className="panel-label">Vocabulary Preview</p>
                <p className="panel-description">
                  Offset {tokenizerVocabularyOffset.toLocaleString()} | Limit {tokenizerVocabularyLimit.toLocaleString()} | Total {tokenizerVocabularyTotal.toLocaleString()}
                </p>
              </div>
              <div className="tokenizer-vocabulary-controls">
                <label htmlFor="tokenizer-page-size" className="panel-description">Page size</label>
                <select
                  id="tokenizer-page-size"
                  className="text-input"
                  value={tokenizerVocabularyLimit}
                  onChange={(event) => void handleTokenizerVocabularyPageSizeChange(Number(event.target.value))}
                  disabled={tokenizerVocabularyLoading}
                >
                  {[100, 250, 500, 1000].map((size) => (
                    <option key={size} value={size}>{size}</option>
                  ))}
                </select>
              </div>
            </header>

            {!tokenizerReport && (
              <div className="chart-placeholder">
                <p>No vocabulary to display.</p>
              </div>
            )}

            {tokenizerReport && (
              <>
                <div className="tokenizer-vocabulary-table-shell">
                  <div className="tokenizer-vocabulary-table-header">
                    <span>token_id</span>
                    <span>token</span>
                    <span>length</span>
                  </div>
                  <div className="tokenizer-vocabulary-list tokenizer-vocabulary-list--paged">
                    {tokenizerVocabulary.map((item) => (
                      <div key={`${item.token_id}-${item.token}`} className="tokenizer-vocabulary-row tokenizer-vocabulary-row--paged">
                        <span className="tokenizer-vocabulary-id">{item.token_id}</span>
                        <span className="tokenizer-vocabulary-token">{item.token}</span>
                        <span className="tokenizer-vocabulary-length">{item.length}</span>
                      </div>
                    ))}
                    {tokenizerVocabulary.length === 0 && (
                      <div className="word-frequency-empty">No vocabulary rows loaded.</div>
                    )}
                  </div>
                </div>

                <div className="tokenizer-vocabulary-footer">
                  <span className="panel-description">
                    Showing {vocabularyStart.toLocaleString()}-{vocabularyEnd.toLocaleString()} of {tokenizerVocabularyTotal.toLocaleString()}
                  </span>
                  <div className="tokenizer-vocabulary-actions">
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => void handlePreviousTokenizerVocabularyPage()}
                      disabled={!canGoPrevious || tokenizerVocabularyLoading}
                    >
                      Previous
                    </button>
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => void handleNextTokenizerVocabularyPage()}
                      disabled={!canGoNext || tokenizerVocabularyLoading}
                    >
                      Next
                    </button>
                  </div>
                </div>
              </>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
};

export default TokenizerExaminationPage;
