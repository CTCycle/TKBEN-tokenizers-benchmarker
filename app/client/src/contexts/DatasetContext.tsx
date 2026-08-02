import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useState,
} from 'react';
import type { ReactNode } from 'react';
import {
    deleteDataset,
    downloadDataset,
    fetchDatasetMetricsCatalog,
    fetchLatestDatasetReport,
    uploadCustomDataset,
    validateDataset,
} from '../services/datasetsApi';
import type {
    DatasetAnalysisRequest,
    DatasetAnalysisResponse,
    DatasetMetricCatalogCategory,
    DatasetPreviewItem,
    DatasetCatalogFilters,
} from '../types/api';
import { useAvailableDatasets } from '../hooks/useAvailableDatasets';
import { useFileInputControl } from '../hooks/useFileInputControl';

interface DatasetContextType {
    datasetName: string | null;
    selectedCorpus: string;
    selectedConfig: string;
    loading: boolean;
    error: string | null;
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
    metricsCatalog: DatasetMetricCatalogCategory[];
    metricsCatalogLoading: boolean;

    setError: (error: string | null) => void;
    handleCorpusChange: (value: string) => void;
    handleConfigChange: (value: string) => void;
    handleLoadDataset: () => Promise<void>;
    handleUploadClick: () => void;
    handleFileChange: (event: React.ChangeEvent<HTMLInputElement>) => Promise<void>;
    handleSelectDataset: (datasetName: string) => void;
    handleValidateDataset: (
        datasetName: string,
        requestOverrides?: Partial<DatasetAnalysisRequest>,
    ) => Promise<void>;
    handleLoadLatestDatasetReport: (
        datasetName: string,
        options?: { suppressNotFoundError?: boolean },
    ) => Promise<void>;
    handleDeleteDataset: (datasetName: string) => Promise<void>;
    refreshAvailableDatasets: (filters?: DatasetCatalogFilters) => Promise<void>;
    loadMetricsCatalog: () => Promise<void>;
}

const DatasetContext = createContext<DatasetContextType | null>(null);
const LAST_DATASET_REPORT_STORAGE_KEY = 'tkben:last-dataset-report';

export const DatasetProvider = ({ children }: { children: ReactNode }) => {
    const {
        inputRef: fileInputRef,
        openFileDialog: handleUploadClick,
        resetFileInput,
    } = useFileInputControl();
    const [datasetName, setDatasetName] = useState<string | null>(null);
    const [selectedCorpus, setSelectedCorpus] = useState('wikitext');
    const [selectedConfig, setSelectedConfig] = useState('wikitext-2-v1');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [loadProgress, setLoadProgress] = useState<number | null>(null);
    const [validating, setValidating] = useState(false);
    const [validationReport, setValidationReport] = useState<DatasetAnalysisResponse | null>(null);
    const [validationProgress, setValidationProgress] = useState<number | null>(null);
    const {
        availableDatasets,
        datasetsLoading,
        datasetsInitialized,
        refreshAvailableDatasets: refreshAvailableDatasetsInternal,
    } = useAvailableDatasets();
    const [activeValidationDataset, setActiveValidationDataset] = useState<string | null>(null);
    const [activeReportLoadDataset, setActiveReportLoadDataset] = useState<string | null>(null);
    const [removingDataset, setRemovingDataset] = useState<string | null>(null);
    const [metricsCatalog, setMetricsCatalog] = useState<DatasetMetricCatalogCategory[]>([]);
    const [metricsCatalogLoading, setMetricsCatalogLoading] = useState(false);

    const refreshAvailableDatasets = useCallback(async (filters: DatasetCatalogFilters = {}) => {
        try {
            await refreshAvailableDatasetsInternal(filters);
        } catch (err) {
            console.error('Failed to fetch datasets:', err);
        }
    }, [refreshAvailableDatasetsInternal]);

    const loadMetricsCatalog = useCallback(async () => {
        setMetricsCatalogLoading(true);
        try {
            const response = await fetchDatasetMetricsCatalog();
            setMetricsCatalog(response.categories ?? []);
        } catch (err) {
            console.error('Failed to fetch dataset metrics catalog:', err);
        } finally {
            setMetricsCatalogLoading(false);
        }
    }, []);

    useEffect(() => {
        const timeoutId = window.setTimeout(() => {
            void refreshAvailableDatasets();
            void loadMetricsCatalog();
        }, 0);

        return () => window.clearTimeout(timeoutId);
    }, [refreshAvailableDatasets, loadMetricsCatalog]);

    const handleCorpusChange = useCallback((value: string) => {
        setSelectedCorpus(value);
        setSelectedConfig('');
    }, []);

    const handleConfigChange = useCallback((value: string) => {
        setSelectedConfig(value);
    }, []);

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

            setDatasetName(response.dataset_name);
            await refreshAvailableDatasets();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load dataset');
        } finally {
            setLoading(false);
            setLoadProgress(null);
        }
    }, [refreshAvailableDatasets, selectedCorpus, selectedConfig]);

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

                setDatasetName(response.dataset_name);
                await refreshAvailableDatasets();
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to upload dataset');
            } finally {
                setLoading(false);
                setLoadProgress(null);
                resetFileInput();
            }
        },
        [refreshAvailableDatasets, resetFileInput],
    );

    const handleSelectDataset = useCallback(
        (targetDataset: string) => {
            if (!targetDataset) return;
            if (
                validationReport?.dataset_name &&
                validationReport.dataset_name !== targetDataset
            ) {
                setValidationReport(null);
            }
            setDatasetName(targetDataset);
        },
        [validationReport],
    );

    const handleValidateDataset = useCallback(async (
        targetDataset: string,
        requestOverrides?: Partial<DatasetAnalysisRequest>,
    ) => {
        if (!targetDataset) return;

        setValidating(true);
        setError(null);
        setValidationProgress(0);
        setActiveValidationDataset(targetDataset);

        try {
            const requestPayload: DatasetAnalysisRequest = {
                dataset_name: targetDataset,
                ...requestOverrides,
            };
            const response = await validateDataset(
                requestPayload,
                (status) => setValidationProgress(status.progress),
            );
            setValidationReport(response);
            setDatasetName(response.dataset_name);
            window.localStorage.setItem(
                LAST_DATASET_REPORT_STORAGE_KEY,
                response.dataset_name,
            );
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to validate dataset');
        } finally {
            setValidating(false);
            setValidationProgress(null);
            setActiveValidationDataset(null);
        }
    }, []);

    const handleLoadLatestDatasetReport = useCallback(async (
        targetDataset: string,
        options?: { suppressNotFoundError?: boolean },
    ) => {
        if (!targetDataset) return;

        setError(null);
        setActiveReportLoadDataset(targetDataset);

        try {
            const response = await fetchLatestDatasetReport(targetDataset);
            setValidationReport(response);
            setDatasetName(response.dataset_name);
            window.localStorage.setItem(
                LAST_DATASET_REPORT_STORAGE_KEY,
                response.dataset_name,
            );
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to load latest dataset report';
            const isNoReportFound = message.toLowerCase().includes('no validation report found');
            if (!(options?.suppressNotFoundError && isNoReportFound)) {
                setError(message);
            }
            if (options?.suppressNotFoundError && isNoReportFound) {
                setValidationReport(null);
                const savedDataset = window.localStorage.getItem(LAST_DATASET_REPORT_STORAGE_KEY);
                if (savedDataset === targetDataset) {
                    window.localStorage.removeItem(LAST_DATASET_REPORT_STORAGE_KEY);
                }
            }
        } finally {
            setActiveReportLoadDataset(null);
        }
    }, []);

    useEffect(() => {
        if (!datasetsInitialized || datasetsLoading || validationReport || activeReportLoadDataset) {
            return;
        }

        const savedDataset = window.localStorage.getItem(LAST_DATASET_REPORT_STORAGE_KEY)?.trim();
        if (!savedDataset) {
            return;
        }

        const datasetExists = availableDatasets.some(
            (dataset) => dataset.dataset_name === savedDataset,
        );
        if (!datasetExists) {
            window.localStorage.removeItem(LAST_DATASET_REPORT_STORAGE_KEY);
            return;
        }

        const timeoutId = window.setTimeout(() => {
            void handleLoadLatestDatasetReport(savedDataset, { suppressNotFoundError: true });
        }, 0);

        return () => window.clearTimeout(timeoutId);
    }, [
        activeReportLoadDataset,
        availableDatasets,
        datasetsInitialized,
        datasetsLoading,
        handleLoadLatestDatasetReport,
        validationReport,
    ]);

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
                }
                if (window.localStorage.getItem(LAST_DATASET_REPORT_STORAGE_KEY) === targetDataset) {
                    window.localStorage.removeItem(LAST_DATASET_REPORT_STORAGE_KEY);
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
            metricsCatalog,
            metricsCatalogLoading,
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
            loadMetricsCatalog,
        }),
        [
            datasetName,
            selectedCorpus,
            selectedConfig,
            loading,
            error,
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
            metricsCatalog,
            metricsCatalogLoading,
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
            loadMetricsCatalog,
        ],
    );

    return (
        <DatasetContext.Provider value={value}>
            {children}
        </DatasetContext.Provider>
    );
};

// eslint-disable-next-line react-refresh/only-export-components
export const useDataset = (): DatasetContextType => {
    const context = useContext(DatasetContext);
    if (!context) {
        throw new Error('useDataset must be used within a DatasetProvider');
    }
    return context;
};
