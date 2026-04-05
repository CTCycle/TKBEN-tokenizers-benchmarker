import { useEffect, useRef, useState } from 'react';
import ChartPlaceholder from '../components/ChartPlaceholder';
import DashboardExportButton from '../components/DashboardExportButton';
import TokenizersPage from './TokenizersPage';
import { useTokenizers } from '../contexts/TokenizersContext';
import type { TokenizerVocabularyStats } from '../types/api';

const NOT_AVAILABLE = 'N/A';

const formatNumber = (value: number | null | undefined, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return NOT_AVAILABLE;
  }
  if (Number.isInteger(value)) {
    return value.toLocaleString();
  }
  return value.toFixed(digits);
};

const formatOptionalPercent = (value: number | null | undefined) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return NOT_AVAILABLE;
  }
  return `${formatNumber(value, 2)}%`;
};

const toOptionalNumber = (value: unknown): number | null => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return null;
  }
  return value;
};

const toVocabularyStats = (value: unknown): TokenizerVocabularyStats | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const payload = value as Partial<TokenizerVocabularyStats>;

  const parseNumber = (candidate: unknown): number | null => {
    if (typeof candidate !== 'number' || Number.isNaN(candidate)) {
      return null;
    }
    return candidate;
  };

  const parseString = (candidate: unknown): string | null => {
    if (typeof candidate !== 'string' || !candidate.trim()) {
      return null;
    }
    return candidate.trim();
  };

  return {
    min_token_length: parseNumber(payload.min_token_length),
    mean_token_length: parseNumber(payload.mean_token_length),
    median_token_length: parseNumber(payload.median_token_length),
    max_token_length: parseNumber(payload.max_token_length),
    mean_token_bytes: parseNumber(payload.mean_token_bytes),
    token_string_entropy: parseNumber(payload.token_string_entropy),
    special_tokens_in_vocab_count: parseNumber(payload.special_tokens_in_vocab_count),
    special_tokens_in_vocab_percentage: parseNumber(payload.special_tokens_in_vocab_percentage),
    byte_fallback_support: typeof payload.byte_fallback_support === 'boolean' ? payload.byte_fallback_support : null,
    unknown_token_representation: parseString(payload.unknown_token_representation),
    normalization_behavior: parseString(payload.normalization_behavior),
    vocabulary_density: parseNumber(payload.vocabulary_density),
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
  const leftPanelRef = useRef<HTMLElement | null>(null);
  const [vocabularyPanelHeight, setVocabularyPanelHeight] = useState<number | null>(null);

  useEffect(() => {
    const syncVocabularyPanelHeight = () => {
      const leftPanel = leftPanelRef.current;
      if (!leftPanel) {
        setVocabularyPanelHeight(null);
        return;
      }
      if (window.innerWidth <= 1100) {
        setVocabularyPanelHeight(null);
        return;
      }
      setVocabularyPanelHeight(Math.max(0, Math.floor(leftPanel.getBoundingClientRect().height)));
    };

    syncVocabularyPanelHeight();

    const observer = new ResizeObserver(() => {
      syncVocabularyPanelHeight();
    });
    if (leftPanelRef.current) {
      observer.observe(leftPanelRef.current);
    }
    window.addEventListener('resize', syncVocabularyPanelHeight);

    return () => {
      observer.disconnect();
      window.removeEventListener('resize', syncVocabularyPanelHeight);
    };
  }, [tokenizerReport]);

  const histogram = tokenizerReport?.token_length_histogram ?? null;
  const maxCount = histogram?.counts.length ? Math.max(...histogram.counts) : 0;

  const globalStats = tokenizerReport?.global_stats ?? {};
  const vocabularyStats = toVocabularyStats(globalStats.vocabulary_stats);

  const vocabularyStart = tokenizerVocabularyTotal > 0 ? tokenizerVocabularyOffset + 1 : 0;
  const vocabularyEnd = tokenizerVocabularyTotal > 0
    ? Math.min(tokenizerVocabularyOffset + tokenizerVocabulary.length, tokenizerVocabularyTotal)
    : 0;
  const canGoPrevious = tokenizerVocabularyOffset > 0;
  const canGoNext = tokenizerVocabularyOffset + tokenizerVocabularyLimit < tokenizerVocabularyTotal;
  const tokenizerExportReportName = tokenizerReport
    ? `tokenizer-${tokenizerReport.tokenizer_name}-report-${tokenizerReport.report_id}`
    : 'tokenizer-dashboard-report';

  return (
    <div className="page-scroll tokenizer-exam-scroll">
      <div className="tokenizer-exam-layout">
        <div className="merged-page-row">
          <TokenizersPage showDashboard={false} embedded />
        </div>

        <div className="tokenizer-report-split">
          <section
            ref={leftPanelRef}
            className="panel dashboard-panel dashboard-plain tokenizer-report-left"
          >
            <header className="panel-header">
              <div>
                <p className="panel-label">Tokenizers Dashboard</p>
                <p className="panel-description">
                  {tokenizerReport
                    ? `Report ${tokenizerReport.report_id} for ${tokenizerReport.tokenizer_name}`
                    : 'Open a tokenizer report from the preview list to populate this dashboard.'}
                </p>
              </div>
              <div className="dashboard-export-header-actions">
                <DashboardExportButton
                  dashboardType="tokenizer"
                  reportName={tokenizerExportReportName}
                  dashboardPayload={tokenizerReport
                    ? {
                      report: tokenizerReport,
                      vocabulary_items: tokenizerVocabulary,
                    }
                    : null}
                />
              </div>
            </header>

            {!tokenizerReport && (
              <ChartPlaceholder
                className="tokenizer-dashboard-empty"
                message="No tokenizer report loaded."
                detail="Use the report icon in tokenizer preview."
              />
            )}

            {tokenizerReport && (
              <div className="tokenizer-report-grid">
                <article className="tokenizer-report-card tokenizer-report-card--basics">
                  <p className="panel-label">Basics</p>
                  <table className="tokenizer-meta-table">
                    <tbody>
                      <tr><th>Tokenizer</th><td>{tokenizerReport.tokenizer_name}</td></tr>
                      <tr><th>Family</th><td>{String(globalStats.tokenizer_family ?? NOT_AVAILABLE)}</td></tr>
                      <tr><th>Backend</th><td>{String(globalStats.runtime_backend ?? NOT_AVAILABLE)}</td></tr>
                      <tr><th>Vocabulary size</th><td>{formatNumber(tokenizerReport.vocabulary_size, 0)}</td></tr>
                      <tr><th>Base vocabulary size</th><td>{formatNumber(toOptionalNumber(globalStats.base_vocabulary_size), 0)}</td></tr>
                      <tr><th>Model max length</th><td>{formatNumber(toOptionalNumber(globalStats.model_max_length), 0)}</td></tr>
                      <tr><th>Normalization</th><td>{String(globalStats.normalization_policy ?? NOT_AVAILABLE)}</td></tr>
                      <tr><th>Pretokenization</th><td>{String(globalStats.pretokenization_policy ?? NOT_AVAILABLE)}</td></tr>
                      <tr><th>Fallback policy</th><td>{String(globalStats.fallback_policy ?? NOT_AVAILABLE)}</td></tr>
                      <tr><th>Special tokens count</th><td>{formatNumber(toOptionalNumber(globalStats.special_tokens_count), 0)}</td></tr>
                      <tr><th>Added tokens count</th><td>{formatNumber(toOptionalNumber(globalStats.added_tokens_count), 0)}</td></tr>
                      <tr>
                        <th>Hugging Face</th>
                        <td>
                          {tokenizerReport.huggingface_url ? (
                            <a href={tokenizerReport.huggingface_url} target="_blank" rel="noreferrer">
                              {tokenizerReport.huggingface_url}
                            </a>
                          ) : NOT_AVAILABLE}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </article>

                <article className="tokenizer-report-card tokenizer-report-card--histogram">
                  <p className="panel-label">Token Length Histogram</p>
                  {!histogram || histogram.counts.length === 0 ? (
                    <ChartPlaceholder message="No histogram data in this report." />
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

                <article className="tokenizer-report-card tokenizer-report-card--vocabulary">
                  <p className="panel-label">Vocabulary Stats</p>
                  <table className="tokenizer-meta-table tokenizer-meta-table-compact">
                    <tbody>
                      <tr><th>Min token length</th><td>{formatNumber(vocabularyStats?.min_token_length, 0)}</td></tr>
                      <tr><th>Mean token length</th><td>{formatNumber(vocabularyStats?.mean_token_length, 2)}</td></tr>
                      <tr><th>Median token length</th><td>{formatNumber(vocabularyStats?.median_token_length, 2)}</td></tr>
                      <tr><th>Max token length</th><td>{formatNumber(vocabularyStats?.max_token_length, 0)}</td></tr>
                      <tr><th>Mean token bytes</th><td>{formatNumber(vocabularyStats?.mean_token_bytes, 2)}</td></tr>
                      <tr><th>Token string entropy</th><td>{formatNumber(vocabularyStats?.token_string_entropy, 4)}</td></tr>
                      <tr><th>Special tokens in vocab</th><td>{formatNumber(vocabularyStats?.special_tokens_in_vocab_count, 0)}</td></tr>
                      <tr><th>Special in vocab %</th><td>{formatOptionalPercent(vocabularyStats?.special_tokens_in_vocab_percentage)}</td></tr>
                      <tr><th>Byte fallback support</th><td>{vocabularyStats?.byte_fallback_support == null ? NOT_AVAILABLE : (vocabularyStats.byte_fallback_support ? 'Yes' : 'No')}</td></tr>
                      <tr><th>Unknown token representation</th><td>{vocabularyStats?.unknown_token_representation ?? NOT_AVAILABLE}</td></tr>
                      <tr><th>Normalization behavior</th><td>{vocabularyStats?.normalization_behavior ?? NOT_AVAILABLE}</td></tr>
                      <tr><th>Vocabulary density</th><td>{formatNumber(vocabularyStats?.vocabulary_density, 4)}</td></tr>
                    </tbody>
                  </table>
                </article>
              </div>
            )}
          </section>

          <aside
            className="panel dashboard-panel dashboard-plain tokenizer-report-right"
            style={vocabularyPanelHeight
              ? { height: `${vocabularyPanelHeight}px` }
              : undefined}
          >
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
              <ChartPlaceholder
                className="tokenizer-vocabulary-empty"
                message="No vocabulary to display."
              />
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
