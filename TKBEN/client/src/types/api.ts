/**
 * Response from the tokenizer scan endpoint
 */
export interface TokenizerScanResponse {
    status: string;
    identifiers: string[];
    count: number;
}

/**
 * Response from the tokenizer settings endpoint
 */
export interface TokenizerSettingsResponse {
    default_scan_limit: number;
    max_scan_limit: number;
    min_scan_limit: number;
}

/**
 * Generic API error response
 */
export interface ApiError {
    detail: string;
}

/**
 * Histogram data for document length distribution
 */
export interface HistogramData {
    bins: string[];
    counts: number[];
    bin_edges: number[];
    min_length: number;
    max_length: number;
    mean_length: number;
    median_length: number;
}

/**
 * Request for downloading a dataset from HuggingFace
 */
export interface DatasetDownloadRequest {
    corpus: string;
    config?: string | null;
    hf_access_token?: string | null;
}

/**
 * Response from the dataset download endpoint
 */
export interface DatasetDownloadResponse {
    status: string;
    dataset_name: string;
    text_column: string;
    document_count: number;
    saved_count: number;
    histogram: HistogramData;
}

/**
 * Response from the custom dataset upload endpoint
 */
export interface CustomDatasetUploadResponse {
    status: string;
    dataset_name: string;
    text_column: string;
    document_count: number;
    saved_count: number;
    histogram: HistogramData;
}

/**
 * Request for analyzing a dataset
 */
export interface DatasetAnalysisRequest {
    dataset_name: string;
}

/**
 * Summary of word-level statistics from dataset analysis
 */
export interface DatasetStatisticsSummary {
    total_documents: number;
    mean_words_count: number;
    median_words_count: number;
    mean_avg_word_length: number;
    mean_std_word_length: number;
}

/**
 * Response from the dataset analysis endpoint
 */
export interface DatasetAnalysisResponse {
    status: string;
    dataset_name: string;
    analyzed_count: number;
    statistics: DatasetStatisticsSummary;
}

/**
 * Response from the list datasets endpoint
 */
export interface DatasetListResponse {
    datasets: string[];
}

/**
 * Global metrics for a single tokenizer benchmark
 */
export interface GlobalMetrics {
    tokenizer: string;
    dataset_name: string;
    tokenization_speed_tps: number;
    throughput_chars_per_sec: number;
    vocabulary_size: number;
    avg_sequence_length: number;
    median_sequence_length: number;
    subword_fertility: number;
    oov_rate: number;
    word_recovery_rate: number;
    character_coverage: number;
    determinism_rate: number;
    boundary_preservation_rate: number;
    round_trip_fidelity_rate: number;
}

/**
 * Vocabulary statistics for chart
 */
export interface VocabularyStats {
    tokenizer: string;
    vocabulary_size: number;
    subwords_count: number;
    true_words_count: number;
    subwords_percentage: number;
}

/**
 * Token length histogram bin
 */
export interface TokenLengthBin {
    bin_start: number;
    bin_end: number;
    count: number;
}

/**
 * Token length distribution for a tokenizer
 */
export interface TokenLengthDistribution {
    tokenizer: string;
    bins: TokenLengthBin[];
    mean: number;
    std: number;
}

/**
 * Speed metrics for comparison
 */
export interface SpeedMetric {
    tokenizer: string;
    tokens_per_second: number;
    chars_per_second: number;
    processing_time_seconds: number;
}

/**
 * Chart data for frontend visualization
 */
export interface ChartData {
    vocabulary_stats: VocabularyStats[];
    token_length_distributions: TokenLengthDistribution[];
    speed_metrics: SpeedMetric[];
}

/**
 * Request for running tokenizer benchmarks
 */
export interface BenchmarkRunRequest {
    tokenizers: string[];
    dataset_name: string;
    max_documents?: number;
    custom_tokenizer_name?: string;
}

/**
 * Response from custom tokenizer upload
 */
export interface TokenizerUploadResponse {
    status: string;
    tokenizer_name: string;
    is_compatible: boolean;
}

/**
 * Response from the benchmark run endpoint
 */
export interface BenchmarkRunResponse {
    status: string;
    dataset_name: string;
    documents_processed: number;
    tokenizers_processed: string[];
    tokenizers_count: number;
    global_metrics: GlobalMetrics[];
    chart_data: ChartData;
}

