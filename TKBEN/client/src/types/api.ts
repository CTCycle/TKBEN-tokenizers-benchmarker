/**
 * Response from the tokenizer scan endpoint
 */
export interface TokenizerScanResponse {
    status: string;
    identifiers: string[];
    count: number;
}

/**
 * Download request for tokenizer persistence
 */
export interface TokenizerDownloadRequest {
    tokenizers: string[];
}

/**
 * Download response for tokenizer persistence
 */
export interface TokenizerDownloadResponse {
    status: string;
    downloaded: string[];
    already_downloaded: string[];
    failed: string[];
    requested_count: number;
    downloaded_count: number;
    already_downloaded_count: number;
    failed_count: number;
}

/**
 * Persisted tokenizer item
 */
export interface TokenizerListItem {
    tokenizer_name: string;
}

/**
 * Persisted tokenizer list response
 */
export interface TokenizerListResponse {
    tokenizers: TokenizerListItem[];
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
 * Hugging Face access key entry.
 */
export interface HFAccessKeyListItem {
    id: number;
    created_at: string;
    is_active: boolean;
    masked_preview: string;
}

/**
 * Hugging Face access key list response.
 */
export interface HFAccessKeyListResponse {
    keys: HFAccessKeyListItem[];
}

/**
 * Hugging Face access key reveal response.
 */
export interface HFAccessKeyRevealResponse {
    id: number;
    key_value: string;
}

/**
 * Generic API error response
 */
export interface ApiError {
    detail: string;
}

/**
 * Response from a job start endpoint
 */
export interface JobStartResponse {
    job_id: string;
    job_type: string;
    status: string;
    message: string;
    poll_interval: number;
}

/**
 * Response for job status polling
 */
export interface JobStatusResponse {
    job_id: string;
    job_type: string;
    status: string;
    progress: number;
    result?: Record<string, unknown> | null;
    error?: string | null;
}

/**
 * Response from a job cancel endpoint
 */
export interface JobCancelResponse {
    job_id: string;
    success: boolean;
    message: string;
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
    configs: {
        configuration?: string | null;
    };
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
    session_name?: string | null;
    selected_metric_keys?: string[] | null;
    sampling?: Record<string, unknown> | null;
    filters?: Record<string, unknown> | null;
    metric_parameters?: Record<string, unknown> | null;
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
 * Word frequency item
 */
export interface WordFrequency {
    word: string;
    count: number;
}

export interface WordLengthItem {
    word: string;
    length: number;
    count: number;
}

export interface WordCloudTerm {
    word: string;
    count: number;
    weight: number;
}

export interface PerDocumentStats {
    document_ids: number[];
    document_lengths: number[];
    word_counts: number[];
    avg_word_lengths: number[];
    std_word_lengths: number[];
}

/**
 * Response from the dataset analysis endpoint
 */
export interface DatasetAnalysisResponse {
    status: string;
    report_id: number | null;
    report_version: number;
    created_at: string | null;
    dataset_name: string;
    session_name?: string | null;
    selected_metric_keys?: string[];
    session_parameters?: Record<string, unknown>;
    document_count: number;
    document_length_histogram: HistogramData;
    word_length_histogram: HistogramData;
    min_document_length: number;
    max_document_length: number;
    most_common_words: WordFrequency[];
    least_common_words: WordFrequency[];
    longest_words: WordLengthItem[];
    shortest_words: WordLengthItem[];
    word_cloud_terms: WordCloudTerm[];
    aggregate_statistics: Record<string, unknown>;
    per_document_stats: PerDocumentStats | null;
}

export interface DatasetMetricCatalogMetric {
    key: string;
    label: string;
    description: string;
    scope: string;
    value_kind: string;
    core: boolean;
}

export interface DatasetMetricCatalogCategory {
    category_key: string;
    category_label: string;
    metrics: DatasetMetricCatalogMetric[];
}

export interface DatasetMetricCatalogResponse {
    categories: DatasetMetricCatalogCategory[];
}

/**
 * Dataset preview item
 */
export interface DatasetPreviewItem {
    dataset_name: string;
    document_count: number;
}

/**
 * Response from the list datasets endpoint
 */
export interface DatasetListResponse {
    datasets: DatasetPreviewItem[];
}

/**
 * Global metrics for a single tokenizer benchmark
 */
export interface GlobalMetrics {
    tokenizer: string;
    dataset_name: string;
    tokenization_speed_tps: number;
    throughput_chars_per_sec: number;
    processing_time_seconds: number;
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
    model_size_mb?: number;
    segmentation_consistency?: number;
    token_distribution_entropy?: number;
    rare_token_tail_1?: number;
    rare_token_tail_2?: number;
    compression_chars_per_token?: number;
    compression_bytes_per_character?: number;
    round_trip_text_fidelity_rate?: number;
    token_id_ordering_monotonicity?: number;
    token_unigram_coverage?: number;
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
    run_name?: string | null;
    selected_metric_keys?: string[] | null;
}

export interface BenchmarkPerDocumentTokenizerStats {
    tokenizer: string;
    tokens_count: number[];
    tokens_to_words_ratio: number[];
    bytes_per_token: number[];
    boundary_preservation_rate: number[];
    round_trip_token_fidelity: number[];
    round_trip_text_fidelity: number[];
    determinism_stability: number[];
    bytes_per_character: number[];
}

export interface BenchmarkMetricCatalogMetric {
    key: string;
    label: string;
    description: string;
    scope: string;
    value_kind: string;
    core: boolean;
}

export interface BenchmarkMetricCatalogCategory {
    category_key: string;
    category_label: string;
    metrics: BenchmarkMetricCatalogMetric[];
}

export interface BenchmarkMetricCatalogResponse {
    categories: BenchmarkMetricCatalogCategory[];
}

export interface BenchmarkReportSummary {
    report_id: number;
    report_version: number;
    created_at: string | null;
    run_name: string | null;
    dataset_name: string;
    documents_processed: number;
    tokenizers_count: number;
    tokenizers_processed: string[];
    selected_metric_keys: string[];
}

export interface BenchmarkReportListResponse {
    reports: BenchmarkReportSummary[];
}

/**
 * Response from custom tokenizer upload
 */
export interface TokenizerUploadResponse {
    status: string;
    tokenizer_name: string;
    is_compatible: boolean;
}

export interface TokenizerValidationGenerateRequest {
    tokenizer_name: string;
}

export interface TokenizerSubwordWordStats {
    heuristic: string;
    subword_count: number;
    word_count: number;
    considered_count: number;
    subword_percentage: number;
    word_percentage: number;
    subword_to_word_ratio: number | null;
}

export interface TokenizerVocabularyStats {
    heuristic?: string | null;
    min_token_length?: number | null;
    mean_token_length?: number | null;
    median_token_length?: number | null;
    max_token_length?: number | null;
    subword_like_count?: number | null;
    subword_like_percentage?: number | null;
    special_tokens_in_vocab_count?: number | null;
    special_tokens_in_vocab_percentage?: number | null;
    unique_token_lengths?: number | null;
    empty_token_count?: number | null;
    considered_non_special_count?: number | null;
}

export interface TokenizerGlobalStats extends Record<string, unknown> {
    vocabulary_size?: number;
    base_vocabulary_size?: number | null;
    tokenizer_algorithm?: string | null;
    tokenizer_class?: string | null;
    backend_tokenizer_class?: string | null;
    has_special_tokens?: boolean;
    special_tokens?: string[];
    special_tokens_count?: number;
    special_tokens_ids_count?: number;
    model_max_length?: number | null;
    padding_side?: string | null;
    added_tokens_count?: number;
    do_lower_case?: boolean | null;
    normalization_hint?: string | null;
    token_length_measure?: string | null;
    persistence_mode?: string;
    persistence_reason?: string;
    subword_word_stats?: TokenizerSubwordWordStats;
    vocabulary_stats?: TokenizerVocabularyStats;
}

export interface TokenizerReportResponse {
    status: string;
    report_id: number;
    report_version: number;
    created_at: string;
    tokenizer_name: string;
    description: string | null;
    huggingface_url?: string | null;
    global_stats: TokenizerGlobalStats;
    token_length_histogram: HistogramData;
    vocabulary_size: number;
}

export interface TokenizerVocabularyItem {
    token_id: number;
    token: string;
    length: number;
}

export interface TokenizerVocabularyPageResponse {
    status: string;
    report_id: number;
    tokenizer_name: string;
    offset: number;
    limit: number;
    total: number;
    items: TokenizerVocabularyItem[];
}

/**
 * Response from the benchmark run endpoint
 */
export interface BenchmarkRunResponse {
    status: string;
    report_id: number | null;
    report_version: number;
    created_at: string | null;
    run_name: string | null;
    selected_metric_keys: string[];
    dataset_name: string;
    documents_processed: number;
    tokenizers_processed: string[];
    tokenizers_count: number;
    global_metrics: GlobalMetrics[];
    chart_data: ChartData;
    per_document_stats: BenchmarkPerDocumentTokenizerStats[];
}

