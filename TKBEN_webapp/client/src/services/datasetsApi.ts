import type { DatasetDownloadRequest, DatasetDownloadResponse } from '../types/api';

const API_BASE_URL = '/api';

// 10 minute timeout for large dataset downloads
import { DOWNLOAD_TIMEOUT_MS } from '../constants';

/**
 * Download a dataset from HuggingFace and save it to the database.
 * @param request - Dataset download parameters
 * @returns Promise with the download response including histogram data
 */
export async function downloadDataset(
    request: DatasetDownloadRequest
): Promise<DatasetDownloadResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), DOWNLOAD_TIMEOUT_MS);

    try {
        const response = await fetch(`${API_BASE_URL}/datasets/download`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `Failed to download dataset: ${response.status}`);
        }

        return response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof Error && error.name === 'AbortError') {
            throw new Error('Dataset download timed out. The dataset may be too large.');
        }
        throw error;
    }
}
