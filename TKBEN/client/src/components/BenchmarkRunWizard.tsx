import { useEffect, useMemo, useState } from 'react';
import type { BenchmarkMetricCatalogCategory, BenchmarkRunRequest } from '../types/api';
import MetricSelectionTree from './MetricSelectionTree';
import { useMetricSelection } from '../hooks/useMetricSelection';

type BenchmarkRunWizardPayload = Omit<BenchmarkRunRequest, 'custom_tokenizer_name'> & {
  run_name: string;
  selected_metric_keys: string[];
};

type BenchmarkRunWizardProps = {
  isOpen: boolean;
  categories: BenchmarkMetricCatalogCategory[];
  availableTokenizers: string[];
  availableDatasets: string[];
  defaultDatasetName: string | null;
  defaultMaxDocuments: number;
  running: boolean;
  onClose: () => void;
  onRun: (payload: BenchmarkRunWizardPayload) => Promise<void>;
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const MAX_SELECTED_TOKENIZERS = 5;

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
  const [selectedTokenizers, setSelectedTokenizers] = useState<string[]>([]);
  const [datasetName, setDatasetName] = useState('');
  const [maxDocuments, setMaxDocuments] = useState(1000);
  const [runName, setRunName] = useState('');
  const [warmupTrials, setWarmupTrials] = useState(2);
  const [timedTrials, setTimedTrials] = useState(8);
  const [batchSize, setBatchSize] = useState(16);
  const [seed, setSeed] = useState(42);
  const [parallelism, setParallelism] = useState(1);
  const [includeLmMetrics, setIncludeLmMetrics] = useState(false);
  const [tokenizerQuery, setTokenizerQuery] = useState('');

  const {
    selectedMetricKeys,
    toggleMetric,
    toggleCategoryByKeys,
    resetSelectionToAll,
  } = useMetricSelection(categories);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    /* eslint-disable react-hooks/set-state-in-effect */
    setStep(0);
    resetSelectionToAll();
    setSelectedTokenizers([]);
    const preferredDataset = defaultDatasetName && availableDatasets.includes(defaultDatasetName)
      ? defaultDatasetName
      : availableDatasets[0] ?? '';
    setDatasetName(preferredDataset);
    setMaxDocuments(clamp(Math.floor(defaultMaxDocuments || 1000), 1, 100000));
    setRunName('');
    setTokenizerQuery('');
    setWarmupTrials(2);
    setTimedTrials(8);
    setBatchSize(16);
    setSeed(42);
    setParallelism(1);
    setIncludeLmMetrics(false);
    /* eslint-enable react-hooks/set-state-in-effect */
  }, [availableDatasets, defaultDatasetName, defaultMaxDocuments, isOpen, resetSelectionToAll]);

  const toggleTokenizer = (tokenizerName: string) => {
    setSelectedTokenizers((current) => {
      if (current.includes(tokenizerName)) {
        return current.filter((item) => item !== tokenizerName);
      }
      if (current.length >= MAX_SELECTED_TOKENIZERS) {
        return current;
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
      config: {
        max_documents: clamp(Math.floor(maxDocuments), 1, 100000),
        warmup_trials: clamp(Math.floor(warmupTrials), 0, 100),
        timed_trials: clamp(Math.floor(timedTrials), 1, 200),
        batch_size: clamp(Math.floor(batchSize), 1, 4096),
        seed: Math.floor(seed),
        parallelism: clamp(Math.floor(parallelism), 1, 128),
        include_lm_metrics: includeLmMetrics,
      },
      run_name: runName.trim(),
      selected_metric_keys: selectedMetricKeys,
    });
  };

  const filteredTokenizers = useMemo(() => {
    const query = tokenizerQuery.trim().toLowerCase();
    if (!query) {
      return availableTokenizers;
    }
    return availableTokenizers.filter((tokenizerName) => tokenizerName.toLowerCase().includes(query));
  }, [availableTokenizers, tokenizerQuery]);

  if (!isOpen) {
    return null;
  }

  const canProceedFromStepOne = selectedMetricKeys.length > 0;
  const canProceedFromStepTwo = selectedTokenizers.length > 0 && Boolean(datasetName);
  const canRun = canProceedFromStepOne && canProceedFromStepTwo && Boolean(runName.trim());

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="benchmark-wizard-title">
      <div className="modal-card benchmark-wizard-modal">
        <header className="benchmark-wizard-header">
          <div>
            <p id="benchmark-wizard-title" className="panel-label">Run Tokenizer Benchmark</p>
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
            categories.length === 0 ? (
              <div className="chart-placeholder">
                <p>No metrics catalog available.</p>
              </div>
            ) : (
              <MetricSelectionTree
                categories={categories}
                selectedMetricKeys={selectedMetricKeys}
                onToggleMetric={toggleMetric}
                onToggleCategory={toggleCategoryByKeys}
                rootClassName="benchmark-wizard-tree"
                categoryClassName="benchmark-wizard-tree-category"
                childrenClassName="benchmark-wizard-tree-children"
              />
            )
          )}

          {step === 1 && (
            <div className="benchmark-wizard-inputs">
              <div className="input-stack benchmark-wizard-input-stack">
                <label className="field-label">Tokenizers</label>
                <input
                  className="text-input"
                  value={tokenizerQuery}
                  onChange={(event) => setTokenizerQuery(event.target.value)}
                  placeholder="Search tokenizers"
                  aria-label="Search tokenizers"
                />
                <p className="panel-description">
                  Selected {selectedTokenizers.length} of {MAX_SELECTED_TOKENIZERS}
                </p>
                {selectedTokenizers.length === 0 && (
                  <p className="benchmark-wizard-validation-message" role="status">
                    Select at least one tokenizer to continue.
                  </p>
                )}
                <div className="benchmark-wizard-tokenizer-list">
                  {availableTokenizers.length === 0 ? (
                    <div className="chart-placeholder">
                      <p>No tokenizers available. Download tokenizers first.</p>
                    </div>
                  ) : filteredTokenizers.length === 0 ? (
                    <div className="chart-placeholder">
                      <p>No tokenizers match your search.</p>
                    </div>
                  ) : (
                    filteredTokenizers.map((tokenizerName) => {
                      const selected = selectedTokenizers.includes(tokenizerName);
                      const limitReached = !selected && selectedTokenizers.length >= MAX_SELECTED_TOKENIZERS;
                      return (
                        <label
                          key={tokenizerName}
                          className={`benchmark-wizard-tokenizer-option${selected ? ' selected' : ''}${limitReached ? ' benchmark-wizard-tokenizer-option--disabled' : ''}`}
                        >
                          <input
                            type="checkbox"
                            checked={selected}
                            disabled={limitReached}
                            onChange={() => toggleTokenizer(tokenizerName)}
                          />
                          <span>{tokenizerName}</span>
                        </label>
                      );
                    })
                  )}
                </div>
              </div>

              <div className="benchmark-wizard-input-grid">
                <div className="input-stack benchmark-wizard-input-stack benchmark-wizard-input-stack--dataset">
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

                <div className="input-stack benchmark-wizard-input-stack benchmark-wizard-input-stack--range">
                  <label className="field-label" htmlFor="benchmark-wizard-documents">
                    Documents processed ({clamp(Math.floor(maxDocuments), 1, 100000).toLocaleString()})
                  </label>
                  <input
                    id="benchmark-wizard-documents"
                    className="benchmark-wizard-range"
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
              <p className="panel-description">warmup_trials: <strong>{clamp(Math.floor(warmupTrials), 0, 100)}</strong></p>
              <p className="panel-description">timed_trials: <strong>{clamp(Math.floor(timedTrials), 1, 200)}</strong></p>
              <p className="panel-description">batch_size: <strong>{clamp(Math.floor(batchSize), 1, 4096)}</strong></p>
              <p className="panel-description">seed: <strong>{Math.floor(seed)}</strong></p>
              <p className="panel-description">parallelism: <strong>{clamp(Math.floor(parallelism), 1, 128)}</strong></p>
              <p className="panel-description">lm_metrics: <strong>{includeLmMetrics ? 'enabled' : 'disabled'}</strong></p>
              <div className="benchmark-wizard-tokenizer-summary-wrap">
                <p className="panel-description">tokenizers_processed:</p>
                <ul className="benchmark-wizard-tokenizer-summary">
                  {selectedTokenizers.length > 0 ? (
                    selectedTokenizers.map((tokenizerName) => (
                      <li key={tokenizerName}>{tokenizerName}</li>
                    ))
                  ) : (
                    <li>N/A</li>
                  )}
                </ul>
              </div>
              <p className="panel-description">
                selected metrics: <strong>{selectedMetricKeys.length.toLocaleString()}</strong>
              </p>
              <div className="benchmark-wizard-input-grid">
                <div className="input-stack">
                  <label className="field-label" htmlFor="benchmark-wizard-warmup">Warmup trials</label>
                  <input id="benchmark-wizard-warmup" className="text-input" type="number" min={0} max={100} value={warmupTrials} onChange={(event) => setWarmupTrials(Number(event.target.value) || 0)} />
                </div>
                <div className="input-stack">
                  <label className="field-label" htmlFor="benchmark-wizard-timed">Timed trials</label>
                  <input id="benchmark-wizard-timed" className="text-input" type="number" min={1} max={200} value={timedTrials} onChange={(event) => setTimedTrials(Number(event.target.value) || 1)} />
                </div>
                <div className="input-stack">
                  <label className="field-label" htmlFor="benchmark-wizard-batch">Batch size</label>
                  <input id="benchmark-wizard-batch" className="text-input" type="number" min={1} max={4096} value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value) || 1)} />
                </div>
                <div className="input-stack">
                  <label className="field-label" htmlFor="benchmark-wizard-seed">Seed</label>
                  <input id="benchmark-wizard-seed" className="text-input" type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value) || 0)} />
                </div>
                <div className="input-stack">
                  <label className="field-label" htmlFor="benchmark-wizard-parallelism">Parallelism</label>
                  <input id="benchmark-wizard-parallelism" className="text-input" type="number" min={1} max={128} value={parallelism} onChange={(event) => setParallelism(Number(event.target.value) || 1)} />
                </div>
                <label className="checkbox">
                  <input type="checkbox" checked={includeLmMetrics} onChange={(event) => setIncludeLmMetrics(event.target.checked)} />
                  <span>Enable LM-backed metrics</span>
                </label>
              </div>
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

