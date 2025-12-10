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
