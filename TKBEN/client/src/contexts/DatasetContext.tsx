import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useRef,
    useState,
} from 'react';
import type { ReactNode } from 'react';
import {
    deleteDataset,
    downloadDataset,
    fetchLatestDatasetReport,
    fetchAvailableDatasets,
    uploadCustomDataset,
    validateDataset,
} from '../services/datasetsApi';
import type { DatasetAnalysisResponse, DatasetPreviewItem, HistogramData } from '../types/api';

interface DatasetStats {
    documentCount: number;
    meanLength: number;
    medianLength: number;
    minLength: number;
    maxLength: number;
}

interface DatasetContextType {
    datasetName: string | null;
    selectedCorpus: string;
    selectedConfig: string;
    loading: boolean;
    error: string | null;
    datasetLoaded: boolean;
    stats: DatasetStats | null;
    histogram: HistogramData | null;
    loadProgress: number | null;
    validating: boolean;
    validationReport: DatasetAnalysisResponse | null;
    validationProgress: number | null;
    fileInputRef: React.RefObject<HTMLInputElement | null>;
    availableDatasets: DatasetPreviewItem[];
    datasetsLoading: boolean;
    activeValidationDataset: string | null;
    activeReportLoadDataset: string | null;
    removingDataset: string | null;

    setSelectedCorpus: (corpus: string) => void;
    setSelectedConfig: (config: string) => void;
    setError: (error: string | null) => void;
    handleCorpusChange: (value: string) => void;
    handleConfigChange: (value: string) => void;
    handleLoadDataset: () => Promise<void>;
    handleUploadClick: () => void;
    handleFileChange: (event: React.ChangeEvent<HTMLInputElement>) => Promise<void>;
    handleSelectDataset: (datasetName: string) => void;
    handleValidateDataset: (datasetName: string) => Promise<void>;
    handleLoadLatestDatasetReport: (datasetName: string) => Promise<void>;
    handleDeleteDataset: (datasetName: string) => Promise<void>;
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
    const [validating, setValidating] = useState(false);
    const [validationReport, setValidationReport] = useState<DatasetAnalysisResponse | null>(null);
    const [validationProgress, setValidationProgress] = useState<number | null>(null);
    const [availableDatasets, setAvailableDatasets] = useState<DatasetPreviewItem[]>([]);
    const [datasetsLoading, setDatasetsLoading] = useState(false);
    const [activeValidationDataset, setActiveValidationDataset] = useState<string | null>(null);
    const [activeReportLoadDataset, setActiveReportLoadDataset] = useState<string | null>(null);
    const [removingDataset, setRemovingDataset] = useState<string | null>(null);

    const refreshAvailableDatasets = useCallback(async () => {
        setDatasetsLoading(true);
        try {
            const response = await fetchAvailableDatasets();
            setAvailableDatasets(response.datasets);
        } catch (err) {
            console.error('Failed to fetch datasets:', err);
        } finally {
            setDatasetsLoading(false);
        }
    }, []);

    useEffect(() => {
        void refreshAvailableDatasets();
    }, [refreshAvailableDatasets]);

    const handleCorpusChange = (value: string) => {
        setSelectedCorpus(value);
        setSelectedConfig('');
        setDatasetLoaded(false);
        setStats(null);
        setHistogram(null);
    };

    const handleConfigChange = (value: string) => {
        setSelectedConfig(value);
        setDatasetLoaded(false);
        setStats(null);
        setHistogram(null);
    };

    const handleLoadDataset = useCallback(async () => {
        setLoading(true);
        setError(null);
        setLoadProgress(0);

        try {
            const normalizedConfig = selectedConfig.trim();
            const response = await downloadDataset(
                {
                    corpus: selectedCorpus,
                    configs: normalizedConfig
                        ? { configuration: normalizedConfig }
                        : {},
                },
                (status) => setLoadProgress(status.progress),
            );

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
            await refreshAvailableDatasets();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load dataset');
        } finally {
            setLoading(false);
            setLoadProgress(null);
        }
    }, [refreshAvailableDatasets, selectedCorpus, selectedConfig]);

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = useCallback(
        async (event: React.ChangeEvent<HTMLInputElement>) => {
            const file = event.target.files?.[0];
            if (!file) return;

            setLoading(true);
            setError(null);
            setLoadProgress(0);

            try {
                const response = await uploadCustomDataset(file, (status) =>
                    setLoadProgress(status.progress),
                );

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
                await refreshAvailableDatasets();
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to upload dataset');
            } finally {
                setLoading(false);
                setLoadProgress(null);
                if (fileInputRef.current) {
                    fileInputRef.current.value = '';
                }
            }
        },
        [refreshAvailableDatasets],
    );

    const handleSelectDataset = useCallback(
        (targetDataset: string) => {
            if (!targetDataset) return;
            if (datasetName !== targetDataset) {
                setStats(null);
                setHistogram(null);
            }
            if (
                validationReport?.dataset_name &&
                validationReport.dataset_name !== targetDataset
            ) {
                setValidationReport(null);
            }
            setDatasetName(targetDataset);
            setDatasetLoaded(true);
        },
        [datasetName, validationReport],
    );

    const handleValidateDataset = useCallback(async (targetDataset: string) => {
        if (!targetDataset) return;

        setValidating(true);
        setError(null);
        setValidationProgress(0);
        setActiveValidationDataset(targetDataset);

        try {
            const response = await validateDataset(
                { dataset_name: targetDataset },
                (status) => setValidationProgress(status.progress),
            );
            setValidationReport(response);
            setDatasetName(response.dataset_name);
            setDatasetLoaded(true);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to validate dataset');
        } finally {
            setValidating(false);
            setValidationProgress(null);
            setActiveValidationDataset(null);
        }
    }, []);

    const handleLoadLatestDatasetReport = useCallback(async (targetDataset: string) => {
        if (!targetDataset) return;

        setError(null);
        setActiveReportLoadDataset(targetDataset);

        try {
            const response = await fetchLatestDatasetReport(targetDataset);
            setValidationReport(response);
            setDatasetName(response.dataset_name);
            setDatasetLoaded(true);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load latest dataset report');
        } finally {
            setActiveReportLoadDataset(null);
        }
    }, []);

    const handleDeleteDataset = useCallback(
        async (targetDataset: string) => {
            if (!targetDataset) return;

            setRemovingDataset(targetDataset);
            setError(null);

            try {
                await deleteDataset(targetDataset);
                if (validationReport?.dataset_name === targetDataset) {
                    setValidationReport(null);
                }
                if (datasetName === targetDataset) {
                    setDatasetName(null);
                    setDatasetLoaded(false);
                    setStats(null);
                    setHistogram(null);
                }
                await refreshAvailableDatasets();
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to delete dataset');
            } finally {
                setRemovingDataset(null);
            }
        },
        [datasetName, refreshAvailableDatasets, validationReport],
    );

    const value = useMemo<DatasetContextType>(
        () => ({
            datasetName,
            selectedCorpus,
            selectedConfig,
            loading,
            error,
            datasetLoaded,
            stats,
            histogram,
            loadProgress,
            validating,
            validationReport,
            validationProgress,
            fileInputRef,
            availableDatasets,
            datasetsLoading,
            activeValidationDataset,
            activeReportLoadDataset,
            removingDataset,
            setSelectedCorpus,
            setSelectedConfig,
            setError,
            handleCorpusChange,
            handleConfigChange,
            handleLoadDataset,
            handleUploadClick,
            handleFileChange,
            handleSelectDataset,
            handleValidateDataset,
            handleLoadLatestDatasetReport,
            handleDeleteDataset,
            refreshAvailableDatasets,
        }),
        [
            datasetName,
            selectedCorpus,
            selectedConfig,
            loading,
            error,
            datasetLoaded,
            stats,
            histogram,
            loadProgress,
            validating,
            validationReport,
            validationProgress,
            fileInputRef,
            availableDatasets,
            datasetsLoading,
            activeValidationDataset,
            activeReportLoadDataset,
            removingDataset,
            handleCorpusChange,
            handleConfigChange,
            handleLoadDataset,
            handleUploadClick,
            handleFileChange,
            handleSelectDataset,
            handleValidateDataset,
            handleLoadLatestDatasetReport,
            handleDeleteDataset,
            refreshAvailableDatasets,
        ],
    );

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
