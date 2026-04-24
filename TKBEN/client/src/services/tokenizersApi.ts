import type {
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
import { ensureOkResponse, parseRecordPayload, readJobStartResponse, readJsonResponse } from './responseGuards';

/**
 * Get tokenizer configuration settings from the server.
 * @returns Promise with tokenizer settings (scan limits)
 */
export async function getTokenizerSettings(): Promise<TokenizerSettingsResponse> {
    const response = await fetch(API_ENDPOINTS.TOKENIZERS_SETTINGS);
    return readJsonResponse(response, 'Failed to fetch settings');
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

    return readJsonResponse(response, 'Failed to scan tokenizers');
}

/**
 * List persisted tokenizers available for benchmarking.
 */
export async function fetchDownloadedTokenizers(): Promise<TokenizerListResponse> {
    const response = await fetch(API_ENDPOINTS.TOKENIZERS_LIST);
    return readJsonResponse(response, 'Failed to list tokenizers');
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

    const job = await readJobStartResponse(response, 'Failed to download tokenizers');
    return waitForJobResult<TokenizerDownloadResponse>(job, {
        onUpdate,
        parseResult: (result) => parseRecordPayload<TokenizerDownloadResponse>(result, 'tokenizer download job result'),
    });
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

    return readJsonResponse(response, 'Failed to upload tokenizer');
}

/**
 * Clear all uploaded custom tokenizers from the server.
 */
export async function clearCustomTokenizers(): Promise<void> {
    const response = await fetch(API_ENDPOINTS.TOKENIZERS_CUSTOM, {
        method: 'DELETE',
    });

    await ensureOkResponse(response, 'Failed to clear tokenizers');
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

    const job = await readJobStartResponse(response, 'Failed to generate tokenizer report');
    return waitForJobResult<TokenizerReportResponse>(job, {
        onUpdate,
        parseResult: (result) => parseRecordPayload<TokenizerReportResponse>(result, 'tokenizer report job result'),
    });
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

    return readJsonResponse(response, 'Failed to load latest tokenizer report');
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
    return readJsonResponse(response, 'Failed to load latest tokenizer report');
}

/**
 * Load tokenizer report by id.
 */
export async function fetchTokenizerReportById(
    reportId: number,
): Promise<TokenizerReportResponse> {
    const response = await fetch(`${API_ENDPOINTS.TOKENIZERS_REPORT_BY_ID}/${reportId}`);

    return readJsonResponse(response, 'Failed to load tokenizer report');
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

    return readJsonResponse(response, 'Failed to load tokenizer vocabulary');
}
