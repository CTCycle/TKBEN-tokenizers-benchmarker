import { createContext, useContext, useCallback, useState } from 'react';
import type { ReactNode } from 'react';
import { scanTokenizers } from '../services/tokenizersApi';
import { runBenchmarks } from '../services/benchmarksApi';
import type { BenchmarkRunResponse, PlotData } from '../types/api';

interface TokenizersContextType {
    // State
    scanInProgress: boolean;
    scanError: string | null;
    fetchedTokenizers: string[];
    selectedTokenizer: string;
    tokenizers: string[];
    includeCustom: boolean;
    includeNSL: boolean;
    maxDocuments: number;
    datasetName: string;
    benchmarkInProgress: boolean;
    benchmarkError: string | null;
    benchmarkResult: BenchmarkRunResponse | null;
    selectedPlot: PlotData | null;

    // Actions
    setSelectedTokenizer: (tokenizer: string) => void;
    setTokenizers: (tokenizers: string[]) => void;
    setIncludeCustom: (value: boolean) => void;
    setIncludeNSL: (value: boolean) => void;
    setMaxDocuments: (value: number) => void;
    setDatasetName: (name: string) => void;
    setSelectedPlot: (plot: PlotData | null) => void;
    setScanError: (error: string | null) => void;
    setBenchmarkError: (error: string | null) => void;
    addTokenizer: (tokenizer: string) => void;
    handleScan: () => Promise<void>;
    handleRunBenchmarks: () => Promise<void>;
    handleDownloadPlot: (plot: PlotData) => void;
}

const TokenizersContext = createContext<TokenizersContextType | null>(null);

export const TokenizersProvider = ({ children }: { children: ReactNode }) => {
    const [scanInProgress, setScanInProgress] = useState(false);
    const [scanError, setScanError] = useState<string | null>(null);
    const [fetchedTokenizers, setFetchedTokenizers] = useState<string[]>([]);
    const [selectedTokenizer, setSelectedTokenizer] = useState('');
    const [tokenizers, setTokenizers] = useState<string[]>([]);
    const [includeCustom, setIncludeCustom] = useState(false);
    const [includeNSL, setIncludeNSL] = useState(false);
    const [maxDocuments, setMaxDocuments] = useState(1000);
    const [datasetName, setDatasetName] = useState('');

    // Benchmark state
    const [benchmarkInProgress, setBenchmarkInProgress] = useState(false);
    const [benchmarkError, setBenchmarkError] = useState<string | null>(null);
    const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkRunResponse | null>(null);
    const [selectedPlot, setSelectedPlot] = useState<PlotData | null>(null);

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

    const handleRunBenchmarks = useCallback(async () => {
        if (tokenizers.length === 0) {
            setBenchmarkError('Please add at least one tokenizer to benchmark.');
            return;
        }

        if (!datasetName.trim()) {
            setBenchmarkError('Please enter a dataset name (e.g., "wikitext/wikitext-2-raw-v1").');
            return;
        }

        setBenchmarkInProgress(true);
        setBenchmarkError(null);
        setBenchmarkResult(null);
        setSelectedPlot(null);

        try {
            const response = await runBenchmarks({
                tokenizers,
                dataset_name: datasetName.trim(),
                max_documents: maxDocuments,
                include_custom_tokenizer: includeCustom,
                include_nsl: includeNSL,
            });
            setBenchmarkResult(response);
            // Auto-select first plot if available
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
    }, [tokenizers, datasetName, maxDocuments, includeCustom, includeNSL]);

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
        includeCustom,
        includeNSL,
        maxDocuments,
        datasetName,
        benchmarkInProgress,
        benchmarkError,
        benchmarkResult,
        selectedPlot,

        // Actions
        setSelectedTokenizer,
        setTokenizers,
        setIncludeCustom,
        setIncludeNSL,
        setMaxDocuments,
        setDatasetName,
        setSelectedPlot,
        setScanError,
        setBenchmarkError,
        addTokenizer,
        handleScan,
        handleRunBenchmarks,
        handleDownloadPlot,
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
