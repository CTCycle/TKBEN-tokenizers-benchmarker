import { useEffect, useState } from 'react';
import type { DatasetAnalysisRequest, DatasetMetricCatalogCategory } from '../types/api';
import MetricSelectionTree from './MetricSelectionTree';
import { useMetricSelection } from '../hooks/useMetricSelection';

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
  const {
    selectedMetricKeys,
    ensureSelectionInitialized,
    toggleMetric,
    toggleCategoryByKeys,
  } = useMetricSelection(categories);
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
    /* eslint-disable react-hooks/set-state-in-effect */
    setStep(0);
    ensureSelectionInitialized();
    /* eslint-enable react-hooks/set-state-in-effect */
  }, [ensureSelectionInitialized, isOpen]);

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
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="validation-wizard-title">
      <div className="modal-card validation-wizard-modal">
        <header className="validation-wizard-header">
          <div>
            <p id="validation-wizard-title" className="panel-label">Run Validation Pipeline</p>
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
                <MetricSelectionTree
                  categories={categories}
                  selectedMetricKeys={selectedMetricKeys}
                  onToggleMetric={toggleMetric}
                  onToggleCategory={toggleCategoryByKeys}
                  categoryClassName="validation-tree-category"
                  childrenClassName="validation-tree-children"
                />
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

