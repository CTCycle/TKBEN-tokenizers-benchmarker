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
    result?: unknown | null;
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
export interface BenchmarkRunConfig {
    max_documents?: number;
    warmup_trials: number;
    timed_trials: number;
    batch_size: number;
    seed: number;
    parallelism: number;
    include_lm_metrics: boolean;
}

export interface BenchmarkHardwareProfile {
    runtime: string;
    os: string;
    cpu_model?: string | null;
    cpu_logical_cores?: number | null;
    memory_total_mb?: number | null;
}

export interface BenchmarkTrialSummary {
    warmup_trials: number;
    timed_trials: number;
}

export interface BenchmarkEfficiencyMetrics {
    encode_tokens_per_second_mean: number;
    encode_tokens_per_second_ci95_low: number;
    encode_tokens_per_second_ci95_high: number;
    encode_chars_per_second_mean: number;
    encode_bytes_per_second_mean: number;
    end_to_end_wall_time_seconds: number;
    load_time_seconds: number;
}

export interface BenchmarkLatencyMetrics {
    encode_latency_p50_ms: number;
    encode_latency_p95_ms: number;
    encode_latency_p99_ms: number;
}

export interface BenchmarkFidelityMetrics {
    exact_round_trip_rate: number;
    normalized_round_trip_rate: number;
    unknown_token_rate: number;
    byte_fallback_rate: number;
    lossless_encodability_rate: number;
}

export interface BenchmarkFragmentationBucket {
    bucket: string;
    pieces_per_word_mean: number;
}

export interface BenchmarkFragmentationMetrics {
    tokens_per_character: number;
    characters_per_token: number;
    tokens_per_byte: number;
    bytes_per_token: number;
    pieces_per_word_mean: number;
    fragmentation_by_word_length_bucket: BenchmarkFragmentationBucket[];
}

export interface BenchmarkResourceMetrics {
    peak_rss_mb: number;
    memory_delta_mb: number;
}

export interface BenchmarkTokenizerResult {
    tokenizer: string;
    tokenizer_family: string;
    runtime_backend: string;
    vocabulary_size: number;
    added_tokens: number;
    special_token_share: number;
    efficiency: BenchmarkEfficiencyMetrics;
    latency: BenchmarkLatencyMetrics;
    fidelity: BenchmarkFidelityMetrics;
    fragmentation: BenchmarkFragmentationMetrics;
    resources: BenchmarkResourceMetrics;
}

export interface BenchmarkSeriesPoint {
    tokenizer: string;
    value: number;
    ci95_low?: number;
    ci95_high?: number;
}

export interface BenchmarkDistributionPoint {
    tokenizer: string;
    min: number;
    q1: number;
    median: number;
    q3: number;
    max: number;
}

export interface BenchmarkChartDataV2 {
    efficiency: BenchmarkSeriesPoint[];
    fidelity: BenchmarkSeriesPoint[];
    vocabulary: BenchmarkSeriesPoint[];
    fragmentation: BenchmarkSeriesPoint[];
    latency_or_memory_distribution: BenchmarkDistributionPoint[];
}

/**
 * Request for running tokenizer benchmarks
 */
export interface BenchmarkRunRequest {
    tokenizers: string[];
    dataset_name: string;
    config: BenchmarkRunConfig;
    custom_tokenizer_name?: string;
    run_name?: string | null;
    selected_metric_keys?: string[] | null;
}

export interface BenchmarkPerDocumentTokenizerStats {
    tokenizer: string;
    tokens_count?: number[];
    pieces_per_word?: number[];
    bytes_per_token: number[];
    encode_latency_ms?: number[];
    peak_rss_mb?: number[];
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

export interface TokenizerVocabularyStats {
    min_token_length?: number | null;
    mean_token_length?: number | null;
    median_token_length?: number | null;
    max_token_length?: number | null;
    mean_token_bytes?: number | null;
    token_string_entropy?: number | null;
    special_tokens_in_vocab_count?: number | null;
    special_tokens_in_vocab_percentage?: number | null;
    byte_fallback_support?: boolean | null;
    unknown_token_representation?: string | null;
    normalization_behavior?: string | null;
    vocabulary_density?: number | null;
}

export interface TokenizerGlobalStats extends Record<string, unknown> {
    vocabulary_size?: number;
    base_vocabulary_size?: number | null;
    tokenizer_family?: string | null;
    runtime_backend?: string | null;
    has_special_tokens?: boolean;
    special_tokens?: string[];
    special_tokens_count?: number;
    special_tokens_ids_count?: number;
    model_max_length?: number | null;
    padding_side?: string | null;
    added_tokens_count?: number;
    normalization_policy?: string | null;
    pretokenization_policy?: string | null;
    fallback_policy?: string | null;
    unknown_token_policy?: string | null;
    byte_fallback_enabled?: boolean | null;
    token_length_measure?: string | null;
    persistence_mode?: string;
    persistence_reason?: string;
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
    config: BenchmarkRunConfig;
    hardware_profile: BenchmarkHardwareProfile;
    trial_summary: BenchmarkTrialSummary;
    tokenizer_results: BenchmarkTokenizerResult[];
    chart_data: BenchmarkChartDataV2;
    per_document_stats: BenchmarkPerDocumentTokenizerStats[];
}

export type DashboardType = 'dataset' | 'tokenizer' | 'benchmark';
