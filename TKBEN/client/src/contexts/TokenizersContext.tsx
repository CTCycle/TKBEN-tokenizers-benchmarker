import { createContext, useContext, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import {
    clearCustomTokenizers,
    downloadTokenizers as downloadTokenizersApi,
    fetchDownloadedTokenizers,
    fetchLatestTokenizerReportOrNull,
    fetchTokenizerReportVocabularyPage,
    generateTokenizerReport,
    scanTokenizers,
    uploadCustomTokenizer,
} from '../services/tokenizersApi';
import { runBenchmarks } from '../services/benchmarksApi';
import { fetchAvailableDatasets } from '../services/datasetsApi';
import type { BenchmarkRunResponse, TokenizerReportResponse, TokenizerVocabularyItem } from '../types/api';

const DEFAULT_TOKENIZER_VOCABULARY_LIMIT = 500;

interface TokenizersContextType {
    // State
    scanInProgress: boolean;
    scanError: string | null;
    downloadInProgress: boolean;
    downloadProgress: number | null;
    downloadWarning: string | null;
    fetchedTokenizers: string[];
    selectedTokenizer: string;
    tokenizers: string[];
    customTokenizerName: string | null;
    customTokenizerUploading: boolean;
    maxDocuments: number;
    availableDatasets: string[];
    selectedDataset: string;
    datasetsLoading: boolean;
    benchmarkInProgress: boolean;
    benchmarkError: string | null;
    benchmarkResult: BenchmarkRunResponse | null;
    benchmarkProgress: number | null;
    activeOpeningTokenizer: string | null;
    tokenizerReport: TokenizerReportResponse | null;
    tokenizerVocabulary: TokenizerVocabularyItem[];
    tokenizerVocabularyOffset: number;
    tokenizerVocabularyLimit: number;
    tokenizerVocabularyTotal: number;
    tokenizerVocabularyLoading: boolean;
    customTokenizerInputRef: React.RefObject<HTMLInputElement | null>;

    // Actions
    setSelectedTokenizer: (tokenizer: string) => void;
    setTokenizers: (tokenizers: string[]) => void;
    setMaxDocuments: (value: number) => void;
    setSelectedDataset: (name: string) => void;
    setScanError: (error: string | null) => void;
    setDownloadWarning: (warning: string | null) => void;
    setBenchmarkError: (error: string | null) => void;
    addTokenizer: (tokenizer: string) => void;
    downloadTokenizers: (tokenizerIds: string[]) => Promise<void>;
    handleScan: () => Promise<void>;
    handleRunBenchmarks: () => Promise<void>;
    handleOpenTokenizerReport: (tokenizerName: string) => Promise<void>;
    handleNextTokenizerVocabularyPage: () => Promise<void>;
    handlePreviousTokenizerVocabularyPage: () => Promise<void>;
    handleTokenizerVocabularyPageSizeChange: (limit: number) => Promise<void>;
    refreshDatasets: () => Promise<void>;
    handleUploadCustomTokenizer: (event: React.ChangeEvent<HTMLInputElement>) => Promise<void>;
    handleClearCustomTokenizer: () => Promise<void>;
    triggerCustomTokenizerUpload: () => void;
}

const TokenizersContext = createContext<TokenizersContextType | null>(null);

export const TokenizersProvider = ({ children }: { children: ReactNode }) => {
    const customTokenizerInputRef = useRef<HTMLInputElement | null>(null);
    const [scanInProgress, setScanInProgress] = useState(false);
    const [scanError, setScanError] = useState<string | null>(null);
    const [downloadInProgress, setDownloadInProgress] = useState(false);
    const [downloadProgress, setDownloadProgress] = useState<number | null>(null);
    const [downloadWarning, setDownloadWarning] = useState<string | null>(null);
    const [fetchedTokenizers, setFetchedTokenizers] = useState<string[]>([]);
    const [selectedTokenizer, setSelectedTokenizer] = useState('');
    const [tokenizers, setTokenizers] = useState<string[]>([]);
    const [customTokenizerName, setCustomTokenizerName] = useState<string | null>(null);
    const [customTokenizerUploading, setCustomTokenizerUploading] = useState(false);
    const [maxDocuments, setMaxDocuments] = useState(1000);
    const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
    const [selectedDataset, setSelectedDataset] = useState('');
    const [datasetsLoading, setDatasetsLoading] = useState(false);

    // Benchmark state
    const [benchmarkInProgress, setBenchmarkInProgress] = useState(false);
    const [benchmarkError, setBenchmarkError] = useState<string | null>(null);
    const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkRunResponse | null>(null);
    const [benchmarkProgress, setBenchmarkProgress] = useState<number | null>(null);

    // Tokenizer report state
    const [activeOpeningTokenizer, setActiveOpeningTokenizer] = useState<string | null>(null);
    const [tokenizerReport, setTokenizerReport] = useState<TokenizerReportResponse | null>(null);
    const [tokenizerVocabulary, setTokenizerVocabulary] = useState<TokenizerVocabularyItem[]>([]);
    const [tokenizerVocabularyOffset, setTokenizerVocabularyOffset] = useState(0);
    const [tokenizerVocabularyLimit, setTokenizerVocabularyLimit] = useState(DEFAULT_TOKENIZER_VOCABULARY_LIMIT);
    const [tokenizerVocabularyTotal, setTokenizerVocabularyTotal] = useState(0);
    const [tokenizerVocabularyLoading, setTokenizerVocabularyLoading] = useState(false);

    const refreshDatasets = useCallback(async () => {
        setDatasetsLoading(true);
        try {
            const response = await fetchAvailableDatasets();
            const datasetNames = response.datasets.map((dataset) => dataset.dataset_name);
            setAvailableDatasets(datasetNames);
            if (datasetNames.length > 0 && !selectedDataset) {
                setSelectedDataset(datasetNames[0]);
            }
        } catch (error) {
            console.error('Failed to fetch datasets:', error);
        } finally {
            setDatasetsLoading(false);
        }
    }, [selectedDataset]);

    const addTokenizer = useCallback((value: string) => {
        if (!value) return;
        setTokenizers((list) => {
            if (list.includes(value)) return list;
            return [...list, value];
        });
    }, []);

    const refreshTokenizers = useCallback(async () => {
        try {
            const response = await fetchDownloadedTokenizers();
            setTokenizers(response.tokenizers.map((item) => item.tokenizer_name));
        } catch (error) {
            console.error('Failed to fetch tokenizers:', error);
        }
    }, []);

    const downloadTokenizers = useCallback(async (tokenizerIds: string[]) => {
        if (downloadInProgress) {
            return;
        }

        const requested = Array.from(
            new Set(tokenizerIds.map((tokenizerId) => tokenizerId.trim()).filter(Boolean)),
        );

        if (requested.length === 0) {
            return;
        }

        setDownloadInProgress(true);
        setDownloadProgress(0);
        setDownloadWarning(null);
        setScanError(null);

        try {
            const response = await downloadTokenizersApi(
                { tokenizers: requested },
                (status) => setDownloadProgress(status.progress),
            );
            if (response.already_downloaded_count > 0) {
                setDownloadWarning('Tokenizer already downloaded.');
            }
            if (response.failed_count > 0) {
                setScanError(
                    `Failed to download ${response.failed_count} tokenizer(s).`,
                );
            }
            await refreshTokenizers();
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to download tokenizers';
            setScanError(errorMessage);
            console.error('Tokenizer download error:', error);
        } finally {
            setDownloadInProgress(false);
        }
    }, [downloadInProgress, refreshTokenizers]);

    const handleScan = useCallback(async () => {
        setScanInProgress(true);
        setScanError(null);

        try {
            const response = await scanTokenizers();
            setFetchedTokenizers(response.identifiers);
            if (response.identifiers.length > 0) {
                setSelectedTokenizer((current) => current || response.identifiers[0]);
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to scan tokenizers';
            setScanError(errorMessage);
            console.error('Scan error:', error);
        } finally {
            setScanInProgress(false);
        }
    }, []);

    const triggerCustomTokenizerUpload = useCallback(() => {
        customTokenizerInputRef.current?.click();
    }, []);

    const handleUploadCustomTokenizer = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setCustomTokenizerUploading(true);
        setBenchmarkError(null);

        try {
            const response = await uploadCustomTokenizer(file);
            if (response.is_compatible) {
                setCustomTokenizerName(response.tokenizer_name);
            } else {
                setBenchmarkError(`Tokenizer "${response.tokenizer_name}" is not compatible.`);
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to upload tokenizer';
            setBenchmarkError(errorMessage);
            console.error('Upload error:', error);
        } finally {
            setCustomTokenizerUploading(false);
            if (customTokenizerInputRef.current) {
                customTokenizerInputRef.current.value = '';
            }
        }
    }, []);

    const handleClearCustomTokenizer = useCallback(async () => {
        try {
            await clearCustomTokenizers();
            setCustomTokenizerName(null);
        } catch (error) {
            console.error('Failed to clear custom tokenizer:', error);
        }
    }, []);

    const handleRunBenchmarks = useCallback(async () => {
        if (tokenizers.length === 0 && !customTokenizerName) {
            setBenchmarkError('Please add at least one tokenizer to benchmark.');
            return;
        }

        if (!selectedDataset) {
            setBenchmarkError('Please select a dataset for benchmarking.');
            return;
        }

        setBenchmarkInProgress(true);
        setBenchmarkError(null);
        setBenchmarkResult(null);
        setBenchmarkProgress(0);

        try {
            const response = await runBenchmarks({
                tokenizers,
                dataset_name: selectedDataset,
                max_documents: maxDocuments,
                custom_tokenizer_name: customTokenizerName || undefined,
            }, (status) => setBenchmarkProgress(status.progress));
            setBenchmarkResult(response);
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to run benchmarks';
            setBenchmarkError(errorMessage);
            console.error('Benchmark error:', error);
        } finally {
            setBenchmarkInProgress(false);
            setBenchmarkProgress(null);
        }
    }, [tokenizers, selectedDataset, maxDocuments, customTokenizerName]);

    const loadTokenizerVocabularyPage = useCallback(async (
        reportId: number,
        offset: number,
        limit: number,
    ) => {
        setTokenizerVocabularyLoading(true);
        try {
            const page = await fetchTokenizerReportVocabularyPage(reportId, offset, limit);
            setTokenizerVocabulary(page.items);
            setTokenizerVocabularyOffset(page.offset);
            setTokenizerVocabularyLimit(page.limit);
            setTokenizerVocabularyTotal(page.total);
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to load tokenizer vocabulary';
            setBenchmarkError(errorMessage);
        } finally {
            setTokenizerVocabularyLoading(false);
        }
    }, []);

    const handleOpenTokenizerReport = useCallback(async (tokenizerName: string) => {
        const normalized = tokenizerName.trim();
        if (!normalized) return;

        setActiveOpeningTokenizer(normalized);
        setBenchmarkError(null);

        try {
            let report = await fetchLatestTokenizerReportOrNull(normalized);
            if (report === null) {
                report = await generateTokenizerReport({ tokenizer_name: normalized });
            }
            setTokenizerReport(report);
            await loadTokenizerVocabularyPage(report.report_id, 0, tokenizerVocabularyLimit);
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to open tokenizer report';
            setBenchmarkError(errorMessage);
        } finally {
            setActiveOpeningTokenizer(null);
        }
    }, [loadTokenizerVocabularyPage, tokenizerVocabularyLimit]);

    const handleNextTokenizerVocabularyPage = useCallback(async () => {
        if (!tokenizerReport || tokenizerVocabularyLoading) {
            return;
        }
        const nextOffset = tokenizerVocabularyOffset + tokenizerVocabularyLimit;
        if (nextOffset >= tokenizerVocabularyTotal) {
            return;
        }
        await loadTokenizerVocabularyPage(
            tokenizerReport.report_id,
            nextOffset,
            tokenizerVocabularyLimit,
        );
    }, [
        loadTokenizerVocabularyPage,
        tokenizerReport,
        tokenizerVocabularyLoading,
        tokenizerVocabularyOffset,
        tokenizerVocabularyLimit,
        tokenizerVocabularyTotal,
    ]);

    const handlePreviousTokenizerVocabularyPage = useCallback(async () => {
        if (!tokenizerReport || tokenizerVocabularyLoading) {
            return;
        }
        const previousOffset = Math.max(0, tokenizerVocabularyOffset - tokenizerVocabularyLimit);
        if (previousOffset === tokenizerVocabularyOffset && tokenizerVocabularyOffset === 0) {
            return;
        }
        await loadTokenizerVocabularyPage(
            tokenizerReport.report_id,
            previousOffset,
            tokenizerVocabularyLimit,
        );
    }, [
        loadTokenizerVocabularyPage,
        tokenizerReport,
        tokenizerVocabularyLoading,
        tokenizerVocabularyOffset,
        tokenizerVocabularyLimit,
    ]);

    const handleTokenizerVocabularyPageSizeChange = useCallback(async (limit: number) => {
        const normalizedLimit = Math.max(1, Math.min(5000, Math.floor(limit || DEFAULT_TOKENIZER_VOCABULARY_LIMIT)));
        setTokenizerVocabularyLimit(normalizedLimit);

        if (!tokenizerReport) {
            return;
        }

        await loadTokenizerVocabularyPage(tokenizerReport.report_id, 0, normalizedLimit);
    }, [loadTokenizerVocabularyPage, tokenizerReport]);

    useEffect(() => {
        void refreshTokenizers();
    }, [refreshTokenizers]);

    const value = useMemo<TokenizersContextType>(() => ({
        // State
        scanInProgress,
        scanError,
        downloadInProgress,
        downloadProgress,
        downloadWarning,
        fetchedTokenizers,
        selectedTokenizer,
        tokenizers,
        customTokenizerName,
        customTokenizerUploading,
        maxDocuments,
        availableDatasets,
        selectedDataset,
        datasetsLoading,
        benchmarkInProgress,
        benchmarkError,
        benchmarkResult,
        benchmarkProgress,
        activeOpeningTokenizer,
        tokenizerReport,
        tokenizerVocabulary,
        tokenizerVocabularyOffset,
        tokenizerVocabularyLimit,
        tokenizerVocabularyTotal,
        tokenizerVocabularyLoading,
        customTokenizerInputRef,

        // Actions
        setSelectedTokenizer,
        setTokenizers,
        setMaxDocuments,
        setSelectedDataset,
        setScanError,
        setDownloadWarning,
        setBenchmarkError,
        addTokenizer,
        downloadTokenizers,
        handleScan,
        handleRunBenchmarks,
        handleOpenTokenizerReport,
        handleNextTokenizerVocabularyPage,
        handlePreviousTokenizerVocabularyPage,
        handleTokenizerVocabularyPageSizeChange,
        refreshDatasets,
        handleUploadCustomTokenizer,
        handleClearCustomTokenizer,
        triggerCustomTokenizerUpload,
    }), [
        scanInProgress,
        scanError,
        downloadInProgress,
        downloadProgress,
        downloadWarning,
        fetchedTokenizers,
        selectedTokenizer,
        tokenizers,
        customTokenizerName,
        customTokenizerUploading,
        maxDocuments,
        availableDatasets,
        selectedDataset,
        datasetsLoading,
        benchmarkInProgress,
        benchmarkError,
        benchmarkResult,
        benchmarkProgress,
        activeOpeningTokenizer,
        tokenizerReport,
        tokenizerVocabulary,
        tokenizerVocabularyOffset,
        tokenizerVocabularyLimit,
        tokenizerVocabularyTotal,
        tokenizerVocabularyLoading,
        customTokenizerInputRef,
        setSelectedTokenizer,
        setTokenizers,
        setMaxDocuments,
        setSelectedDataset,
        setScanError,
        setDownloadWarning,
        setBenchmarkError,
        addTokenizer,
        downloadTokenizers,
        handleScan,
        handleRunBenchmarks,
        handleOpenTokenizerReport,
        handleNextTokenizerVocabularyPage,
        handlePreviousTokenizerVocabularyPage,
        handleTokenizerVocabularyPageSizeChange,
        refreshDatasets,
        handleUploadCustomTokenizer,
        handleClearCustomTokenizer,
        triggerCustomTokenizerUpload,
    ]);

    return (
        <TokenizersContext.Provider value={value}>
            {children}
        </TokenizersContext.Provider>
    );
};

export const useTokenizers = (): TokenizersContextType => {
    const context = useContext(TokenizersContext);
    if (!context) {
        throw new Error('useTokenizers must be used within a TokenizersProvider');
    }
    return context;
};
