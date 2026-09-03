export type SupportedTokenizerPipeline =
    | 'text-generation'
    | 'fill-mask'
    | 'text-classification'
    | 'token-classification'
    | 'text2text-generation'
    | 'question-answering'
    | 'sentence-similarity'
    | 'translation'
    | 'summarization'
    | 'zero-shot-classification';

export type TokenizerDiscoverySort = 'downloads' | 'likes' | 'last_modified' | 'created_at';
export type TokenizerDiscoveryAccess = 'all' | 'public' | 'gated';
export type VocabularySort = 'none' | 'ascending' | 'descending';

export interface TokenizerDiscoveryQuery {
    search?: string;
    limit?: number;
    pipeline_tag?: SupportedTokenizerPipeline;
    author?: string;
    include_tags?: string[];
    exclude_tags?: string[];
    access?: TokenizerDiscoveryAccess;
    sort?: TokenizerDiscoverySort;
    vocabulary_operator?: ComparisonOperator;
    vocabulary_size?: number;
    vocabulary_sort?: VocabularySort;
}

export interface TokenizerDiscoveryItem {
    identifier: string;
    pipeline_tag: string | null;
    library_name: string | null;
    downloads: number | null;
    likes: number | null;
    last_modified: string | null;
    gated: boolean | string | null;
    tags: string[];
    vocabulary_size: number | null;
}

export interface TokenizerDiscoveryResponse {
    items: TokenizerDiscoveryItem[];
    count: number;
    fetched_count: number;
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
    failed_details: string[];
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
    source: 'huggingface' | 'custom';
    has_report: boolean;
    vocabulary_size: number | null;
}

/**
 * Persisted tokenizer list response
 */
export interface TokenizerListResponse {
    tokenizers: TokenizerListItem[];
    count: number;
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
 * Histogram data for a length distribution.
 */
export interface HistogramData {
    bins: string[];
    counts: number[];
    bin_edges: number[];
    min_length: number;
    max_length: number;
    mean_length: number;
    median_length: number;
    token_length_std?: number | null;
    token_length_p90?: number | null;
    token_length_cv?: number | null;
    single_character_token_percentage?: number | null;
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
    count: number;
}

export type ComparisonOperator = 'at_least' | 'at_most';
export interface DatasetCatalogFilters {
    search?: string;
    source?: 'all' | 'public' | 'custom';
    document_count_operator?: ComparisonOperator;
    document_count?: number;
}
export interface TokenizerCatalogFilters {
    search?: string;
    source?: 'all' | 'huggingface' | 'custom';
    vocabulary_size_operator?: ComparisonOperator;
    vocabulary_size?: number;
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
    add_special_tokens?: boolean;
    padding?: boolean;
    truncation?: boolean;
    max_length?: number | null;
    store_per_document_stats?: boolean;
    per_document_sample_size?: number;
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
    encode_tokens_per_second_mean: number | null;
    encode_tokens_per_second_ci95_low: number | null;
    encode_tokens_per_second_ci95_high: number | null;
    encode_chars_per_second_mean: number | null;
    encode_bytes_per_second_mean: number | null;
    encode_only_wall_time_seconds: number | null;
    dataset_stream_wall_time_seconds: number | null;
    postprocess_wall_time_seconds: number | null;
    end_to_end_wall_time_seconds: number | null;
    load_time_seconds: number | null;
}

export interface BenchmarkLatencyMetrics {
    encode_latency_p50_ms: number | null;
    encode_latency_p95_ms: number | null;
    encode_latency_p99_ms: number | null;
    sample_count: number | null;
}

export interface BenchmarkFidelityMetrics {
    exact_round_trip_rate: number | null;
    normalized_round_trip_rate: number | null;
    unknown_token_rate: number | null;
    byte_fallback_rate: number | null;
    lossless_encodability_rate: number | null;
}

export interface BenchmarkFragmentationBucket {
    bucket: string;
    pieces_per_word_mean: number;
}

export interface BenchmarkFragmentationMetrics {
    tokens_per_character: number | null;
    characters_per_token: number | null;
    tokens_per_byte: number | null;
    bytes_per_token: number | null;
    pieces_per_word_mean: number | null;
    fragmentation_by_word_length_bucket: BenchmarkFragmentationBucket[];
}

export interface BenchmarkResourceMetrics {
    peak_rss_mb: number | null;
    memory_delta_mb: number | null;
}

export interface BenchmarkTokenizerResult {
    tokenizer: string;
    status: string;
    error_type?: string | null;
    error_message?: string | null;
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

export interface BenchmarkDashboardPoint { tokenizer: string; value: number; interval_low: number | null; interval_high: number | null; }
export interface BenchmarkDashboardDistribution { tokenizer: string; min: number; q1: number; median: number; q3: number; max: number; sample_count: number; }
export interface BenchmarkDashboardBucketPoint { tokenizer: string; bucket: string; value: number; }
export interface BenchmarkDashboardHistogramBin { tokenizer: string; bin_low: number; bin_high: number; count: number; proportion: number; }
export type BenchmarkVisualizationKind = 'bar' | 'horizontal_bar' | 'interval_bar' | 'dot_whisker' | 'box_plot' | 'histogram' | 'grouped_bar' | 'heatmap';
export interface BenchmarkDashboardWidgetData {
  widget_id: string; metric_keys: string[]; category_key: string; category_label: string;
  label: string; description: string; unit: string; display_format: string;
  default_visualization: BenchmarkVisualizationKind; compatible_visualizations: BenchmarkVisualizationKind[]; default_visible: boolean;
  width: 'standard' | 'wide'; points: BenchmarkDashboardPoint[];
  distributions: BenchmarkDashboardDistribution[]; buckets: BenchmarkDashboardBucketPoint[]; histogram_bins: BenchmarkDashboardHistogramBin[];
}
export interface BenchmarkDashboardData { widgets: BenchmarkDashboardWidgetData[]; available_widget_ids: string[]; available_metric_keys: string[]; unavailable_selected_metric_keys: string[]; }

/**
 * Request for running tokenizer benchmarks
 */
export interface BenchmarkRunRequest {
    tokenizers: string[];
    dataset_name: string;
    config: BenchmarkRunConfig;
    run_name?: string | null;
    selected_metric_keys?: string[] | null;
}

export type BenchmarkRunWizardPayload = Omit<
    BenchmarkRunRequest,
    'run_name' | 'selected_metric_keys'
> & {
    run_name: string;
    selected_metric_keys: string[];
};

export interface BenchmarkPerDocumentTokenizerStats {
    tokenizer: string;
    tokens_count: number[];
    pieces_per_word: (number | null)[];
    bytes_per_token: number[];
    encode_latency_ms: (number | null)[];
    peak_rss_mb: (number | null)[];
}

export interface BenchmarkMetricCatalogMetric {
    key: string;
    label: string;
    description: string;
    scope: string;
    value_kind: string;
    core: boolean;
    unit: string;
    display_format: string;
    default_visualization: BenchmarkVisualizationKind;
    compatible_visualizations: BenchmarkVisualizationKind[];
    default_visible: boolean;
    width: string;
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

export type BenchmarkReportSort = 'newest' | 'oldest';
export interface BenchmarkReportQuery {
    search?: string;
    sort?: BenchmarkReportSort;
    offset?: number;
    limit?: number;
}

export interface BenchmarkReportListResponse {
    reports: BenchmarkReportSummary[];
    total: number;
    offset: number;
    limit: number;
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
    token_length_std?: number | null;
    token_length_p90?: number | null;
    token_length_cv?: number | null;
    single_character_token_percentage?: number | null;
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
    schema_version: number;
    methodology_version: string;
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
    dashboard: BenchmarkDashboardData;
    per_document_stats: BenchmarkPerDocumentTokenizerStats[];
    runtime_metadata: Record<string, unknown>;
    raw_observations: Record<string, Record<string, unknown>[]>;
}

export type DashboardType = 'dataset' | 'tokenizer' | 'benchmark';
