import { useEffect, useMemo, useState } from 'react';
import type { BenchmarkMetricCatalogCategory } from '../types/api';

type BenchmarkRunWizardProps = {
  isOpen: boolean;
  categories: BenchmarkMetricCatalogCategory[];
  availableTokenizers: string[];
  availableDatasets: string[];
  defaultDatasetName: string | null;
  defaultMaxDocuments: number;
  running: boolean;
  onClose: () => void;
  onRun: (payload: {
    tokenizers: string[];
    dataset_name: string;
    max_documents: number;
    run_name: string;
    selected_metric_keys: string[];
  }) => Promise<void>;
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const BenchmarkRunWizard = ({
  isOpen,
  categories,
  availableTokenizers,
  availableDatasets,
  defaultDatasetName,
  defaultMaxDocuments,
  running,
  onClose,
  onRun,
}: BenchmarkRunWizardProps) => {
  const [step, setStep] = useState(0);
  const [selectedMetricKeys, setSelectedMetricKeys] = useState<string[]>([]);
  const [selectedTokenizers, setSelectedTokenizers] = useState<string[]>([]);
  const [datasetName, setDatasetName] = useState('');
  const [maxDocuments, setMaxDocuments] = useState(1000);
  const [runName, setRunName] = useState('');

  const allMetricKeys = useMemo(
    () => categories.flatMap((category) => category.metrics.map((metric) => metric.key)),
    [categories],
  );

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    setStep(0);
    setSelectedMetricKeys(allMetricKeys);
    setSelectedTokenizers([]);
    const preferredDataset = defaultDatasetName && availableDatasets.includes(defaultDatasetName)
      ? defaultDatasetName
      : availableDatasets[0] ?? '';
    setDatasetName(preferredDataset);
    setMaxDocuments(clamp(Math.floor(defaultMaxDocuments || 1000), 1, 100000));
    setRunName('');
  }, [allMetricKeys, availableDatasets, defaultDatasetName, defaultMaxDocuments, isOpen]);

  const toggleMetric = (metricKey: string, enabled: boolean) => {
    setSelectedMetricKeys((current) => {
      if (enabled) {
        if (current.includes(metricKey)) {
          return current;
        }
        return [...current, metricKey];
      }
      return current.filter((key) => key !== metricKey);
    });
  };

  const toggleCategory = (category: BenchmarkMetricCatalogCategory, enabled: boolean) => {
    const keys = category.metrics.map((metric) => metric.key);
    setSelectedMetricKeys((current) => {
      if (enabled) {
        return Array.from(new Set([...current, ...keys]));
      }
      return current.filter((key) => !keys.includes(key));
    });
  };

  const toggleTokenizer = (tokenizerName: string) => {
    setSelectedTokenizers((current) => {
      if (current.includes(tokenizerName)) {
        return current.filter((item) => item !== tokenizerName);
      }
      return [...current, tokenizerName];
    });
  };

  const runBenchmark = async () => {
    if (!datasetName || selectedTokenizers.length === 0 || selectedMetricKeys.length === 0 || !runName.trim()) {
      return;
    }
    await onRun({
      tokenizers: selectedTokenizers,
      dataset_name: datasetName,
      max_documents: clamp(Math.floor(maxDocuments), 1, 100000),
      run_name: runName.trim(),
      selected_metric_keys: selectedMetricKeys,
    });
  };

  if (!isOpen) {
    return null;
  }

  const canProceedFromStepOne = selectedMetricKeys.length > 0;
  const canProceedFromStepTwo = selectedTokenizers.length > 0 && Boolean(datasetName);
  const canRun = canProceedFromStepOne && canProceedFromStepTwo && Boolean(runName.trim());

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true">
      <div className="modal-card benchmark-wizard-modal">
        <header className="benchmark-wizard-header">
          <div>
            <p className="panel-label">Run Tokenizer Benchmark</p>
            <p className="panel-description">
              Configure metrics, inputs, and run metadata for a persisted benchmark report.
            </p>
          </div>
          <button
            type="button"
            className="icon-button subtle"
            onClick={onClose}
            aria-label="Close benchmark wizard"
            disabled={running}
          >
            ×
          </button>
        </header>

        <div className="benchmark-wizard-steps">
          <span className={step === 0 ? 'active' : ''}>1. Metrics</span>
          <span className={step === 1 ? 'active' : ''}>2. Inputs</span>
          <span className={step === 2 ? 'active' : ''}>3. Summary</span>
        </div>

        <div className="benchmark-wizard-body">
          {step === 0 && (
            <div className="benchmark-wizard-tree">
              {categories.length === 0 ? (
                <div className="chart-placeholder">
                  <p>No metrics catalog available.</p>
                </div>
              ) : (
                categories.map((category) => {
                  const categoryKeys = category.metrics.map((metric) => metric.key);
                  const selectedCount = categoryKeys.filter((key) => selectedMetricKeys.includes(key)).length;
                  const allSelected = categoryKeys.length > 0 && selectedCount === categoryKeys.length;
                  return (
                    <div key={category.category_key} className="benchmark-wizard-tree-category">
                      <label className="checkbox">
                        <input
                          type="checkbox"
                          checked={allSelected}
                          onChange={(event) => toggleCategory(category, event.target.checked)}
                        />
                        <span>{category.category_label} ({selectedCount}/{categoryKeys.length})</span>
                      </label>
                      <div className="benchmark-wizard-tree-children">
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
            <div className="benchmark-wizard-inputs">
              <div className="input-stack">
                <label className="field-label">Tokenizers</label>
                <div className="benchmark-wizard-tokenizer-list" role="listbox" aria-multiselectable="true">
                  {availableTokenizers.length === 0 ? (
                    <div className="chart-placeholder">
                      <p>No tokenizers available. Download tokenizers first.</p>
                    </div>
                  ) : (
                    availableTokenizers.map((tokenizerName) => {
                      const selected = selectedTokenizers.includes(tokenizerName);
                      return (
                        <button
                          key={tokenizerName}
                          type="button"
                          className={`benchmark-wizard-tokenizer-pill${selected ? ' selected' : ''}`}
                          onClick={() => toggleTokenizer(tokenizerName)}
                        >
                          {tokenizerName}
                        </button>
                      );
                    })
                  )}
                </div>
              </div>

              <div className="benchmark-wizard-input-grid">
                <div className="input-stack">
                  <label className="field-label" htmlFor="benchmark-wizard-dataset">
                    Dataset
                  </label>
                  <select
                    id="benchmark-wizard-dataset"
                    className="text-input"
                    value={datasetName}
                    onChange={(event) => setDatasetName(event.target.value)}
                  >
                    {availableDatasets.length === 0 ? (
                      <option value="">No datasets available</option>
                    ) : (
                      availableDatasets.map((dataset) => (
                        <option key={dataset} value={dataset}>{dataset}</option>
                      ))
                    )}
                  </select>
                </div>

                <div className="input-stack">
                  <label className="field-label" htmlFor="benchmark-wizard-documents">
                    Documents processed ({clamp(Math.floor(maxDocuments), 1, 100000).toLocaleString()})
                  </label>
                  <input
                    id="benchmark-wizard-documents"
                    type="range"
                    min={1}
                    max={100000}
                    step={1}
                    value={clamp(Math.floor(maxDocuments), 1, 100000)}
                    onChange={(event) => setMaxDocuments(Math.max(1, Number(event.target.value) || 1))}
                  />
                </div>
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="benchmark-wizard-summary">
              <div className="input-stack">
                <label className="field-label" htmlFor="benchmark-wizard-run-name">
                  Run Name
                </label>
                <input
                  id="benchmark-wizard-run-name"
                  className="text-input"
                  value={runName}
                  onChange={(event) => setRunName(event.target.value)}
                  placeholder="e.g. production-tokenizer-comparison"
                  required
                />
              </div>
              <p className="panel-description">dataset_name: <strong>{datasetName || 'N/A'}</strong></p>
              <p className="panel-description">documents_processed: <strong>{clamp(Math.floor(maxDocuments), 1, 100000).toLocaleString()}</strong></p>
              <p className="panel-description">tokenizers_count: <strong>{selectedTokenizers.length}</strong></p>
              <p className="panel-description">
                tokenizers_processed: <strong>{selectedTokenizers.length > 0 ? selectedTokenizers.join(', ') : 'N/A'}</strong>
              </p>
              <p className="panel-description">
                selected metrics: <strong>{selectedMetricKeys.length.toLocaleString()}</strong>
              </p>
            </div>
          )}
        </div>

        <footer className="benchmark-wizard-footer">
          <button
            type="button"
            className="secondary-button"
            onClick={() => setStep((current) => Math.max(0, current - 1))}
            disabled={step === 0 || running}
          >
            Back
          </button>
          {step < 2 ? (
            <button
              type="button"
              className="primary-button"
              onClick={() => setStep((current) => Math.min(2, current + 1))}
              disabled={running || (step === 0 ? !canProceedFromStepOne : !canProceedFromStepTwo)}
            >
              Next
            </button>
          ) : (
            <button
              type="button"
              className="primary-button"
              onClick={() => void runBenchmark()}
              disabled={running || !canRun}
            >
              {running ? 'Running...' : 'Confirm and Run'}
            </button>
          )}
        </footer>
      </div>
    </div>
  );
};

export default BenchmarkRunWizard;
