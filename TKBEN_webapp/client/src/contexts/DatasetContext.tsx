import { createContext, useContext, useCallback, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { analyzeDataset, downloadDataset, uploadCustomDataset } from '../services/datasetsApi';
import type { DatasetStatisticsSummary, HistogramData } from '../types/api';

interface DatasetStats {
    documentCount: number;
    meanLength: number;
    medianLength: number;
    minLength: number;
    maxLength: number;
}

interface DatasetContextType {
    // State
    datasetName: string | null;
    selectedCorpus: string;
    selectedConfig: string;
    loading: boolean;
    error: string | null;
    datasetLoaded: boolean;
    stats: DatasetStats | null;
    histogram: HistogramData | null;
    analyzing: boolean;
    analysisStats: DatasetStatisticsSummary | null;
    fileInputRef: React.RefObject<HTMLInputElement | null>;

    // Actions
    setSelectedCorpus: (corpus: string) => void;
    setSelectedConfig: (config: string) => void;
    setError: (error: string | null) => void;
    handleCorpusChange: (value: string) => void;
    handleConfigChange: (value: string) => void;
    handleLoadDataset: () => Promise<void>;
    handleUploadClick: () => void;
    handleFileChange: (event: React.ChangeEvent<HTMLInputElement>) => Promise<void>;
    handleAnalyzeDataset: () => Promise<void>;
}

const datasetOptions = [
    { corpus: 'wikitext', configs: ['wikitext-103-v1', 'wikitext-2-v1'] },
    { corpus: 'openwebtext', configs: ['default'] },
    { corpus: 'c4', configs: ['en', 'realnewslike'] },
];

const DatasetContext = createContext<DatasetContextType | null>(null);

export const DatasetProvider = ({ children }: { children: ReactNode }) => {
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const [datasetName, setDatasetName] = useState<string | null>(null);
    const [selectedCorpus, setSelectedCorpus] = useState(datasetOptions[0].corpus);
    const [selectedConfig, setSelectedConfig] = useState(datasetOptions[0].configs[0]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [datasetLoaded, setDatasetLoaded] = useState(false);
    const [stats, setStats] = useState<DatasetStats | null>(null);
    const [histogram, setHistogram] = useState<HistogramData | null>(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [analysisStats, setAnalysisStats] = useState<DatasetStatisticsSummary | null>(null);

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
        setAnalysisStats(null);
    };

    const handleConfigChange = (value: string) => {
        setSelectedConfig(value);
        setDatasetLoaded(false);
        setStats(null);
        setHistogram(null);
        setAnalysisStats(null);
    };

    const handleLoadDataset = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await downloadDataset({
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
            setDatasetName(response.dataset_name);
            setDatasetLoaded(true);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load dataset');
        } finally {
            setLoading(false);
        }
    }, [selectedCorpus, selectedConfig]);

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setLoading(true);
        setError(null);

        try {
            const response = await uploadCustomDataset(file);

            setStats({
                documentCount: response.document_count,
                meanLength: response.histogram.mean_length,
                medianLength: response.histogram.median_length,
                minLength: response.histogram.min_length,
                maxLength: response.histogram.max_length,
            });
            setHistogram(response.histogram);
            setDatasetName(response.dataset_name);
            setDatasetLoaded(true);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to upload dataset');
        } finally {
            setLoading(false);
            // Reset file input so the same file can be uploaded again if needed
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    }, []);

    const handleAnalyzeDataset = useCallback(async () => {
        if (!datasetName) return;

        setAnalyzing(true);
        setError(null);

        try {
            const response = await analyzeDataset({ dataset_name: datasetName });
            setAnalysisStats(response.statistics);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to analyze dataset');
        } finally {
            setAnalyzing(false);
        }
    }, [datasetName]);

    const value: DatasetContextType = {
        // State
        datasetName,
        selectedCorpus,
        selectedConfig,
        loading,
        error,
        datasetLoaded,
        stats,
        histogram,
        analyzing,
        analysisStats,
        fileInputRef,

        // Actions
        setSelectedCorpus,
        setSelectedConfig,
        setError,
        handleCorpusChange,
        handleConfigChange,
        handleLoadDataset,
        handleUploadClick,
        handleFileChange,
        handleAnalyzeDataset,
    };

    return (
        <DatasetContext.Provider value={value}>
            {children}
        </DatasetContext.Provider>
    );
};

export const useDataset = (): DatasetContextType => {
    const context = useContext(DatasetContext);
    if (!context) {
        throw new Error('useDataset must be used within a DatasetProvider');
    }
    return context;
};
