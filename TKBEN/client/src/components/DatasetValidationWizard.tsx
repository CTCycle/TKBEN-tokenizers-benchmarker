import { useEffect, useMemo, useState } from 'react';
import type { DatasetAnalysisRequest, DatasetMetricCatalogCategory } from '../types/api';

type DatasetValidationWizardProps = {
  isOpen: boolean;
  datasetName: string | null;
  categories: DatasetMetricCatalogCategory[];
  loadingCategories: boolean;
  validating: boolean;
  onRetryCatalogLoad?: () => void;
  onClose: () => void;
  onRun: (request: Partial<DatasetAnalysisRequest>) => Promise<void>;
};

type SamplingMode = 'fraction' | 'count';

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const DatasetValidationWizard = ({
  isOpen,
  datasetName,
  categories,
  loadingCategories,
  validating,
  onRetryCatalogLoad,
  onClose,
  onRun,
}: DatasetValidationWizardProps) => {
  const [step, setStep] = useState(0);
  const allMetricKeys = useMemo(
    () => categories.flatMap((category) => category.metrics.map((metric) => metric.key)),
    [categories],
  );
  const [selectedMetricKeys, setSelectedMetricKeys] = useState<string[]>([]);
  const [samplingMode, setSamplingMode] = useState<SamplingMode>('fraction');
  const [samplingFraction, setSamplingFraction] = useState(1);
  const [samplingCount, setSamplingCount] = useState(1000);
  const [minLength, setMinLength] = useState('');
  const [maxLength, setMaxLength] = useState('');
  const [excludeEmpty, setExcludeEmpty] = useState(true);
  const [sessionName, setSessionName] = useState('');
  const catalogUnavailable = !loadingCategories && categories.length === 0;

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    setStep(0);
    if (allMetricKeys.length > 0) {
      setSelectedMetricKeys((current) => (current.length > 0 ? current : allMetricKeys));
    }
  }, [allMetricKeys, isOpen]);

  const toggleMetric = (metricKey: string, enabled: boolean) => {
    setSelectedMetricKeys((current) => {
      if (enabled) {
        if (current.includes(metricKey)) {
          return current;
        }
        return [...current, metricKey];
      }
      return current.filter((item) => item !== metricKey);
    });
  };

  const toggleCategory = (category: DatasetMetricCatalogCategory, enabled: boolean) => {
    const keys = category.metrics.map((metric) => metric.key);
    setSelectedMetricKeys((current) => {
      if (enabled) {
        return Array.from(new Set([...current, ...keys]));
      }
      return current.filter((item) => !keys.includes(item));
    });
  };

  const runPipeline = async () => {
    if (!datasetName) return;
    const request: Partial<DatasetAnalysisRequest> = {
      session_name: sessionName.trim() ? sessionName.trim() : null,
      sampling: samplingMode === 'fraction'
        ? { fraction: clamp(samplingFraction, 0.01, 1) }
        : { count: Math.max(1, Math.floor(samplingCount)) },
      filters: {
        min_length: minLength.trim() ? Math.max(0, Number(minLength)) : null,
        max_length: maxLength.trim() ? Math.max(0, Number(maxLength)) : null,
        exclude_empty: excludeEmpty,
      },
      metric_parameters: {},
    };
    if (selectedMetricKeys.length > 0) {
      request.selected_metric_keys = selectedMetricKeys;
    }
    void onRun(request);
    onClose();
    setStep(0);
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true">
      <div className="modal-card validation-wizard-modal">
        <header className="validation-wizard-header">
          <div>
            <p className="panel-label">Run Validation Pipeline</p>
            <p className="panel-description">
              Configure metrics and sampling for dataset analysis.
            </p>
          </div>
          <button
            type="button"
            className="icon-button subtle"
            onClick={onClose}
            aria-label="Close validation wizard"
          >
            ×
          </button>
        </header>

        <div className="validation-wizard-steps">
          <span className={step === 0 ? 'active' : ''}>1. Metric Selection</span>
          <span className={step === 1 ? 'active' : ''}>2. Sampling & Filters</span>
          <span className={step === 2 ? 'active' : ''}>3. Confirmation</span>
        </div>

        <div className="validation-wizard-body">
          {step === 0 && (
            <div className="validation-tree-container">
              {loadingCategories ? (
                <div className="chart-placeholder">
                  <p>Loading metrics catalog...</p>
                </div>
              ) : categories.length === 0 ? (
                <div className="chart-placeholder">
                  <p>Metrics catalog is unavailable.</p>
                  <span>Continue to run with backend default metrics or retry loading the catalog.</span>
                  {onRetryCatalogLoad && (
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={onRetryCatalogLoad}
                      disabled={loadingCategories || validating}
                    >
                      Retry catalog load
                    </button>
                  )}
                </div>
              ) : (
                categories.map((category) => {
                  const categoryKeys = category.metrics.map((metric) => metric.key);
                  const selectedCount = categoryKeys.filter((key) => selectedMetricKeys.includes(key)).length;
                  const categoryChecked = selectedCount === categoryKeys.length && categoryKeys.length > 0;
                  return (
                    <div key={category.category_key} className="validation-tree-category">
                      <label className="checkbox">
                        <input
                          type="checkbox"
                          checked={categoryChecked}
                          onChange={(event) => toggleCategory(category, event.target.checked)}
                        />
                        <span>
                          {category.category_label} ({selectedCount}/{categoryKeys.length})
                        </span>
                      </label>
                      <div className="validation-tree-children">
                        {category.metrics.map((metric) => (
                          <label key={metric.key} className="checkbox">
                            <input
                              type="checkbox"
                              checked={selectedMetricKeys.includes(metric.key)}
                              onChange={(event) => toggleMetric(metric.key, event.target.checked)}
                            />
                            <span>{metric.label}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          )}

          {step === 1 && (
            <div className="validation-form-grid">
              <div className="input-stack">
                <label className="field-label">Sampling Mode</label>
                <div className="field-row">
                  <label className="checkbox">
                    <input
                      type="radio"
                      checked={samplingMode === 'fraction'}
                      onChange={() => setSamplingMode('fraction')}
                    />
                    <span>Fraction</span>
                  </label>
                  <label className="checkbox">
                    <input
                      type="radio"
                      checked={samplingMode === 'count'}
                      onChange={() => setSamplingMode('count')}
                    />
                    <span>Document Count</span>
                  </label>
                </div>
              </div>

              {samplingMode === 'fraction' ? (
                <div className="input-stack">
                  <label className="field-label">Fraction ({samplingFraction.toFixed(2)})</label>
                  <input
                    type="range"
                    min={0.01}
                    max={1}
                    step={0.01}
                    value={samplingFraction}
                    onChange={(event) => setSamplingFraction(Number(event.target.value))}
                  />
                </div>
              ) : (
                <div className="input-stack">
                  <label className="field-label">Document Count ({Math.max(1, Math.floor(samplingCount))})</label>
                  <input
                    type="range"
                    min={1}
                    max={100000}
                    step={1}
                    value={Math.max(1, Math.floor(samplingCount))}
                    onChange={(event) => setSamplingCount(Math.max(1, Number(event.target.value) || 1))}
                  />
                  <input
                    className="text-input"
                    value={samplingCount}
                    onChange={(event) => setSamplingCount(Math.max(1, Number(event.target.value) || 1))}
                  />
                </div>
              )}

              <div className="input-stack">
                <label className="field-label">Min Length</label>
                <input
                  className="text-input"
                  value={minLength}
                  onChange={(event) => setMinLength(event.target.value)}
                  placeholder="Optional"
                />
              </div>
              <div className="input-stack">
                <label className="field-label">Max Length</label>
                <input
                  className="text-input"
                  value={maxLength}
                  onChange={(event) => setMaxLength(event.target.value)}
                  placeholder="Optional"
                />
              </div>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={excludeEmpty}
                  onChange={(event) => setExcludeEmpty(event.target.checked)}
                />
                <span>Exclude empty documents</span>
              </label>
            </div>
          )}

          {step === 2 && (
            <div className="validation-summary">
              <div className="input-stack">
                <label className="field-label">Session Name (optional)</label>
                <input
                  className="text-input"
                  value={sessionName}
                  onChange={(event) => setSessionName(event.target.value)}
                  placeholder="e.g. English subset quality pass"
                />
              </div>
              <p className="panel-description">
                Dataset: <strong>{datasetName ?? 'N/A'}</strong>
              </p>
              <p className="panel-description">
                Selected metrics: <strong>{selectedMetricKeys.length}</strong>
              </p>
              <p className="panel-description">
                Sampling: <strong>{samplingMode === 'fraction'
                  ? `fraction ${samplingFraction.toFixed(2)}`
                  : `count ${Math.max(1, Math.floor(samplingCount))}`}</strong>
              </p>
              <p className="panel-description">
                Filters: <strong>
                  min={minLength || 'none'}, max={maxLength || 'none'}, exclude_empty={String(excludeEmpty)}
                </strong>
              </p>
            </div>
          )}
        </div>

        <footer className="validation-wizard-footer">
          <button
            type="button"
            className="secondary-button"
            onClick={() => setStep((current) => Math.max(0, current - 1))}
            disabled={step === 0 || validating}
          >
            Back
          </button>
          {step < 2 ? (
            <button
              type="button"
              className="primary-button"
              onClick={() => setStep((current) => Math.min(2, current + 1))}
              disabled={(step === 0 && !catalogUnavailable && selectedMetricKeys.length === 0) || validating}
            >
              Next
            </button>
          ) : (
            <button
              type="button"
              className="primary-button"
              onClick={() => void runPipeline()}
              disabled={validating || !datasetName}
            >
              {validating ? 'Running...' : 'Run Validation'}
            </button>
          )}
        </footer>
      </div>
    </div>
  );
};

export default DatasetValidationWizard;
