import type {
    JobStartResponse,
    JobStatusResponse,
    TokenizerDownloadRequest,
    TokenizerDownloadResponse,
    TokenizerListResponse,
    TokenizerReportResponse,
    TokenizerScanResponse,
    TokenizerSettingsResponse,
    TokenizerValidationGenerateRequest,
    TokenizerVocabularyPageResponse,
    TokenizerUploadResponse,
} from '../types/api';

import { API_ENDPOINTS } from '../constants';
import { waitForJobResult } from './jobsApi';

/**
 * Get tokenizer configuration settings from the server.
 * @returns Promise with tokenizer settings (scan limits)
 */
export async function getTokenizerSettings(): Promise<TokenizerSettingsResponse> {
    const response = await fetch(API_ENDPOINTS.TOKENIZERS_SETTINGS);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to fetch settings: ${response.status}`);
    }

    return response.json();
}

/**
 * Scan HuggingFace for the most popular tokenizer identifiers.
 * @param limit - Maximum number of tokenizers to fetch. If not provided, uses server default.
 * @returns Promise with the scan response containing tokenizer identifiers
 */
export async function scanTokenizers(limit?: number): Promise<TokenizerScanResponse> {
    const params = new URLSearchParams();

    if (limit !== undefined) {
        params.append('limit', String(limit));
    }

    const queryString = params.toString();
    const url = queryString
        ? `${API_ENDPOINTS.TOKENIZERS_SCAN}?${queryString}`
        : API_ENDPOINTS.TOKENIZERS_SCAN;

    const response = await fetch(url);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to scan tokenizers: ${response.status}`);
    }

    return response.json();
}

/**
 * List persisted tokenizers available for benchmarking.
 */
export async function fetchDownloadedTokenizers(): Promise<TokenizerListResponse> {
    const response = await fetch(API_ENDPOINTS.TOKENIZERS_LIST);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to list tokenizers: ${response.status}`);
    }

    return response.json();
}

/**
 * Download tokenizer IDs and persist them for benchmark usage.
 */
export async function downloadTokenizers(
    request: TokenizerDownloadRequest,
    onUpdate?: (status: JobStatusResponse) => void,
): Promise<TokenizerDownloadResponse> {
    const response = await fetch(API_ENDPOINTS.TOKENIZERS_DOWNLOAD, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to download tokenizers: ${response.status}`);
    }

    const job = await response.json() as JobStartResponse;
    return waitForJobResult<TokenizerDownloadResponse>(job, { onUpdate });
}

/**
 * Upload a custom tokenizer.json file.
 * @param file - The tokenizer.json file to upload
 * @returns Promise with the upload response
 */
export async function uploadCustomTokenizer(file: File): Promise<TokenizerUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(API_ENDPOINTS.TOKENIZERS_UPLOAD, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to upload tokenizer: ${response.status}`);
    }

    return response.json();
}

/**
 * Clear all uploaded custom tokenizers from the server.
 */
export async function clearCustomTokenizers(): Promise<void> {
    const response = await fetch(API_ENDPOINTS.TOKENIZERS_CUSTOM, {
        method: 'DELETE',
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to clear tokenizers: ${response.status}`);
    }
}

/**
 * Generate and persist a tokenizer metadata report.
 */
export async function generateTokenizerReport(
    request: TokenizerValidationGenerateRequest,
    onUpdate?: (status: JobStatusResponse) => void,
): Promise<TokenizerReportResponse> {
    const response = await fetch(API_ENDPOINTS.TOKENIZERS_REPORT_GENERATE, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to generate tokenizer report: ${response.status}`);
    }

    const job = await response.json() as JobStartResponse;
    return waitForJobResult<TokenizerReportResponse>(job, { onUpdate });
}

/**
 * Load latest persisted tokenizer report.
 */
export async function fetchLatestTokenizerReport(
    tokenizerName: string,
): Promise<TokenizerReportResponse> {
    const response = await fetch(
        `${API_ENDPOINTS.TOKENIZERS_REPORT_LATEST}?tokenizer_name=${encodeURIComponent(tokenizerName)}`,
    );

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to load latest tokenizer report: ${response.status}`);
    }

    return response.json();
}

/**
 * Load latest persisted tokenizer report, returning null if not found.
 */
export async function fetchLatestTokenizerReportOrNull(
    tokenizerName: string,
): Promise<TokenizerReportResponse | null> {
    const response = await fetch(
        `${API_ENDPOINTS.TOKENIZERS_REPORT_LATEST}?tokenizer_name=${encodeURIComponent(tokenizerName)}`,
    );

    if (response.status === 404) {
        return null;
    }
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to load latest tokenizer report: ${response.status}`);
    }

    return response.json();
}

/**
 * Load tokenizer report by id.
 */
export async function fetchTokenizerReportById(
    reportId: number,
): Promise<TokenizerReportResponse> {
    const response = await fetch(`${API_ENDPOINTS.TOKENIZERS_REPORT_BY_ID}/${reportId}`);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to load tokenizer report: ${response.status}`);
    }

    return response.json();
}

/**
 * Load a page of tokenizer vocabulary rows for a report.
 */
export async function fetchTokenizerReportVocabularyPage(
    reportId: number,
    offset = 0,
    limit = 500,
): Promise<TokenizerVocabularyPageResponse> {
    const params = new URLSearchParams({
        offset: String(offset),
        limit: String(limit),
    });
    const response = await fetch(
        `${API_ENDPOINTS.TOKENIZERS_REPORT_BY_ID}/${reportId}/vocabulary?${params.toString()}`,
    );

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to load tokenizer vocabulary: ${response.status}`);
    }

    return response.json();
}
