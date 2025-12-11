import type { TokenizerScanResponse, TokenizerSettingsResponse, TokenizerUploadResponse } from '../types/api';

const API_BASE_URL = '/api';

/**
 * Get tokenizer configuration settings from the server.
 * @returns Promise with tokenizer settings (scan limits)
 */
export async function getTokenizerSettings(): Promise<TokenizerSettingsResponse> {
    const response = await fetch(`${API_BASE_URL}/tokenizers/settings`);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to fetch settings: ${response.status}`);
    }

    return response.json();
}

/**
 * Scan HuggingFace for the most popular tokenizer identifiers.
 * @param limit - Maximum number of tokenizers to fetch. If not provided, uses server default.
 * @param hfAccessToken - Optional HuggingFace access token
 * @returns Promise with the scan response containing tokenizer identifiers
 */
export async function scanTokenizers(
    limit?: number,
    hfAccessToken?: string
): Promise<TokenizerScanResponse> {
    const params = new URLSearchParams();

    if (limit !== undefined) {
        params.append('limit', String(limit));
    }

    if (hfAccessToken) {
        params.append('hf_access_token', hfAccessToken);
    }

    const queryString = params.toString();
    const url = queryString
        ? `${API_BASE_URL}/tokenizers/scan?${queryString}`
        : `${API_BASE_URL}/tokenizers/scan`;

    const response = await fetch(url);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to scan tokenizers: ${response.status}`);
    }

    return response.json();
}

/**
 * Upload a custom tokenizer.json file.
 * @param file - The tokenizer.json file to upload
 * @returns Promise with the upload response
 */
export async function uploadCustomTokenizer(file: File): Promise<TokenizerUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/tokenizers/upload`, {
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
    const response = await fetch(`${API_BASE_URL}/tokenizers/custom`, {
        method: 'DELETE',
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to clear tokenizers: ${response.status}`);
    }
}
