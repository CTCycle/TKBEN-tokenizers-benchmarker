import { createContext, useContext, useCallback, useState, useRef } from 'react';
import type { ReactNode } from 'react';
import { scanTokenizers, uploadCustomTokenizer, clearCustomTokenizers } from '../services/tokenizersApi';
import { runBenchmarks } from '../services/benchmarksApi';
import { fetchAvailableDatasets } from '../services/datasetsApi';
import type { BenchmarkRunResponse, PlotData } from '../types/api';

interface TokenizersContextType {
    // State
    scanInProgress: boolean;
    scanError: string | null;
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
    selectedPlot: PlotData | null;
    customTokenizerInputRef: React.RefObject<HTMLInputElement | null>;

    // Actions
    setSelectedTokenizer: (tokenizer: string) => void;
    setTokenizers: (tokenizers: string[]) => void;
    setMaxDocuments: (value: number) => void;
    setSelectedDataset: (name: string) => void;
    setSelectedPlot: (plot: PlotData | null) => void;
    setScanError: (error: string | null) => void;
    setBenchmarkError: (error: string | null) => void;
    addTokenizer: (tokenizer: string) => void;
    handleScan: () => Promise<void>;
    handleRunBenchmarks: () => Promise<void>;
    handleDownloadPlot: (plot: PlotData) => void;
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
    const [selectedPlot, setSelectedPlot] = useState<PlotData | null>(null);

    const refreshDatasets = useCallback(async () => {
        setDatasetsLoading(true);
        try {
            const response = await fetchAvailableDatasets();
            setAvailableDatasets(response.datasets);
            if (response.datasets.length > 0 && !selectedDataset) {
                setSelectedDataset(response.datasets[0]);
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
        setSelectedPlot(null);

        try {
            const response = await runBenchmarks({
                tokenizers,
                dataset_name: selectedDataset,
                max_documents: maxDocuments,
                custom_tokenizer_name: customTokenizerName || undefined,
            });
            setBenchmarkResult(response);
            if (response.plots && response.plots.length > 0) {
                setSelectedPlot(response.plots[0]);
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to run benchmarks';
            setBenchmarkError(errorMessage);
            console.error('Benchmark error:', error);
        } finally {
            setBenchmarkInProgress(false);
        }
    }, [tokenizers, selectedDataset, maxDocuments, customTokenizerName]);

    const handleDownloadPlot = useCallback((plot: PlotData) => {
        const link = document.createElement('a');
        link.href = `data:image/png;base64,${plot.data}`;
        link.download = `${plot.name}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }, []);

    const value: TokenizersContextType = {
        // State
        scanInProgress,
        scanError,
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
        selectedPlot,
        customTokenizerInputRef,

        // Actions
        setSelectedTokenizer,
        setTokenizers,
        setMaxDocuments,
        setSelectedDataset,
        setSelectedPlot,
        setScanError,
        setBenchmarkError,
        addTokenizer,
        handleScan,
        handleRunBenchmarks,
        handleDownloadPlot,
        refreshDatasets,
        handleUploadCustomTokenizer,
        handleClearCustomTokenizer,
        triggerCustomTokenizerUpload,
    };

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
