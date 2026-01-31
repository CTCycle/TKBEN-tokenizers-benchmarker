import { createContext, useContext, useCallback, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { analyzeDataset, downloadDataset, uploadCustomDataset, fetchAvailableDatasets } from '../services/datasetsApi';
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
    loadProgress: number | null;
    analyzing: boolean;
    analysisStats: DatasetStatisticsSummary | null;
    analysisProgress: number | null;
    fileInputRef: React.RefObject<HTMLInputElement | null>;
    availableDatasets: string[];
    selectedAnalysisDataset: string;
    analysisDatasetLoading: boolean;

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
    setSelectedAnalysisDataset: (name: string) => void;
    refreshAvailableDatasets: () => Promise<void>;
}

const DatasetContext = createContext<DatasetContextType | null>(null);

export const DatasetProvider = ({ children }: { children: ReactNode }) => {
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const [datasetName, setDatasetName] = useState<string | null>(null);
    const [selectedCorpus, setSelectedCorpus] = useState('wikitext');
    const [selectedConfig, setSelectedConfig] = useState('wikitext-2-v1');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [datasetLoaded, setDatasetLoaded] = useState(false);
    const [stats, setStats] = useState<DatasetStats | null>(null);
    const [histogram, setHistogram] = useState<HistogramData | null>(null);
    const [loadProgress, setLoadProgress] = useState<number | null>(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [analysisStats, setAnalysisStats] = useState<DatasetStatisticsSummary | null>(null);
    const [analysisProgress, setAnalysisProgress] = useState<number | null>(null);
    const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
    const [selectedAnalysisDataset, setSelectedAnalysisDataset] = useState('');
    const [analysisDatasetLoading, setAnalysisDatasetLoading] = useState(false);

    const refreshAvailableDatasets = useCallback(async () => {
        setAnalysisDatasetLoading(true);
        try {
            const response = await fetchAvailableDatasets();
            setAvailableDatasets(response.datasets);
            // Auto-select first dataset if none selected
            if (response.datasets.length > 0 && !selectedAnalysisDataset) {
                setSelectedAnalysisDataset(response.datasets[0]);
            }
        } catch (err) {
            console.error('Failed to fetch datasets:', err);
        } finally {
            setAnalysisDatasetLoading(false);
        }
    }, [selectedAnalysisDataset]);

    const handleCorpusChange = (value: string) => {
        setSelectedCorpus(value);
        setSelectedConfig('');
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
        setLoadProgress(0);

        try {
            const response = await downloadDataset({
                corpus: selectedCorpus,
                config: selectedConfig,
            }, (status) => setLoadProgress(status.progress));

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
            // Refresh available datasets after loading new one
            await refreshAvailableDatasets();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load dataset');
        } finally {
            setLoading(false);
            setLoadProgress(null);
        }
    }, [selectedCorpus, selectedConfig, refreshAvailableDatasets]);

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setLoading(true);
        setError(null);
        setLoadProgress(0);

        try {
            const response = await uploadCustomDataset(file, (status) => setLoadProgress(status.progress));

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
            // Refresh available datasets after upload
            await refreshAvailableDatasets();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to upload dataset');
        } finally {
            setLoading(false);
            setLoadProgress(null);
            // Reset file input so the same file can be uploaded again if needed
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    }, [refreshAvailableDatasets]);

    const handleAnalyzeDataset = useCallback(async () => {
        if (!selectedAnalysisDataset) return;

        setAnalyzing(true);
        setError(null);
        setAnalysisProgress(0);

        try {
            const response = await analyzeDataset(
                { dataset_name: selectedAnalysisDataset },
                (status) => setAnalysisProgress(status.progress),
            );
            setAnalysisStats(response.statistics);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to analyze dataset');
        } finally {
            setAnalyzing(false);
            setAnalysisProgress(null);
        }
    }, [selectedAnalysisDataset]);

    const value = useMemo<DatasetContextType>(() => ({
        // State
        datasetName,
        selectedCorpus,
        selectedConfig,
        loading,
        error,
        datasetLoaded,
        stats,
        histogram,
        loadProgress,
        analyzing,
        analysisStats,
        analysisProgress,
        fileInputRef,
        availableDatasets,
        selectedAnalysisDataset,
        analysisDatasetLoading,

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
        setSelectedAnalysisDataset,
        refreshAvailableDatasets,
    }), [
        datasetName,
        selectedCorpus,
        selectedConfig,
        loading,
        error,
        datasetLoaded,
        stats,
        histogram,
        loadProgress,
        analyzing,
        analysisStats,
        analysisProgress,
        fileInputRef,
        availableDatasets,
        selectedAnalysisDataset,
        analysisDatasetLoading,
        setSelectedCorpus,
        setSelectedConfig,
        setError,
        handleCorpusChange,
        handleConfigChange,
        handleLoadDataset,
        handleUploadClick,
        handleFileChange,
        handleAnalyzeDataset,
        setSelectedAnalysisDataset,
        refreshAvailableDatasets,
    ]);

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
