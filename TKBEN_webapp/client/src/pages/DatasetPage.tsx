import { useMemo, useState } from 'react';

const datasetOptions = [
  { corpus: 'wikitext', configs: ['wikitext-103-v1', 'wikitext-2-v1'] },
  { corpus: 'openwebtext', configs: ['default'] },
  { corpus: 'c4', configs: ['en', 'realnewslike'] },
];

const DatasetPage = () => {
  const [useCustom, setUseCustom] = useState(false);
  const [selectedCorpus, setSelectedCorpus] = useState(datasetOptions[0].corpus);
  const [selectedConfig, setSelectedConfig] = useState(datasetOptions[0].configs[0]);
  const [numDocs, setNumDocs] = useState(50000);
  const [removeInvalid, setRemoveInvalid] = useState(true);

  const stats = useMemo(() => {
    const baseDocs = numDocs;
    const tokens = baseDocs * 200;
    return [
      { label: 'Documents', value: baseDocs.toLocaleString() },
      { label: 'Tokens (est.)', value: tokens.toLocaleString() },
      { label: 'Languages', value: selectedCorpus === 'c4' ? '4' : '1' },
      { label: 'Last scan', value: '3 hours ago' },
    ];
  }, [numDocs, selectedCorpus]);

  const handleCorpusChange = (value: string) => {
    setSelectedCorpus(value);
    const option = datasetOptions.find((item) => item.corpus === value);
    if (option) {
      setSelectedConfig(option.configs[0]);
    }
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
            <div className="input-stack">
              <label className="field-label">Corpus</label>
              <input
                className="text-input"
                value={selectedCorpus}
                onChange={(event) => handleCorpusChange(event.target.value)}
                disabled={useCustom}
              />
            </div>
            <div className="input-stack">
              <label className="field-label">Configuration</label>
              <input
                className="text-input"
                value={selectedConfig}
                onChange={(event) => setSelectedConfig(event.target.value)}
                disabled={useCustom}
              />
            </div>
            <div className="input-row">
              <div className="input-stack">
                <label className="field-label">Number of documents</label>
                <input
                  type="number"
                  className="text-input"
                  min={1}
                  max={1000000000}
                  value={numDocs}
                  onChange={(event) => setNumDocs(Number(event.target.value))}
                />
              </div>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={removeInvalid}
                  onChange={(event) => setRemoveInvalid(event.target.checked)}
                />
                <span>Remove invalid documents</span>
              </label>
            </div>
          </div>
          <footer className="panel-footer">
            <button type="button" className="primary-button">
              Load dataset
            </button>
            <button type="button" className="primary-button ghost">
              Analyze dataset
            </button>
          </footer>
        </section>
        <aside className="panel dashboard-panel">
          <header className="panel-header">
            <div>
              <p className="panel-label">Dataset overview</p>
              <p className="panel-description">
                Quick stats for {selectedCorpus}/{selectedConfig}
              </p>
            </div>
          </header>
          <div className="dashboard-grid">
            {stats.map((item) => (
              <div key={item.label} className="stat-card">
                <p className="stat-label">{item.label}</p>
                <p className="stat-value">{item.value}</p>
              </div>
            ))}
          </div>
          <div className="chart-placeholder">
            <p>Dataset distribution preview</p>
            <span>This area can host small histograms or pie charts.</span>
          </div>
          <ul className="insights-list">
            <li>
              Default pre-processing removes empty entries:
              <strong>{removeInvalid ? ' enabled' : ' disabled'}</strong>.
            </li>
            <li>Custom datasets leverage the same benchmarking pipeline.</li>
          </ul>
        </aside>
      </div>
    </div>
  );
};

export default DatasetPage;
