import type { CustomDatasetUploadResponse, DatasetAnalysisRequest, DatasetAnalysisResponse, DatasetDownloadRequest, DatasetDownloadResponse, DatasetListResponse } from '../types/api';
import { API_ENDPOINTS, DOWNLOAD_TIMEOUT_MS } from '../constants';

// 10 minute timeout for large dataset downloads

/**
 * Fetch list of available datasets from the database.
 * @returns Promise with the list of dataset names
 */
export async function fetchAvailableDatasets(): Promise<DatasetListResponse> {
    const response = await fetch(API_ENDPOINTS.DATASETS_LIST, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to fetch datasets: ${response.status}`);
    }

    return response.json();
}

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
        const response = await fetch(API_ENDPOINTS.DATASETS_DOWNLOAD, {
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

/**
 * Upload a custom CSV/Excel dataset file and save it to the database.
 * @param file - The file to upload (.csv, .xlsx, or .xls)
 * @returns Promise with the upload response including histogram data
 */
export async function uploadCustomDataset(
    file: File
): Promise<CustomDatasetUploadResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), DOWNLOAD_TIMEOUT_MS);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(API_ENDPOINTS.DATASETS_UPLOAD, {
            method: 'POST',
            body: formData,
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `Failed to upload dataset: ${response.status}`);
        }

        return response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof Error && error.name === 'AbortError') {
            throw new Error('Dataset upload timed out. The file may be too large.');
        }
        throw error;
    }
}

/**
 * Analyze a dataset to compute word-level statistics.
 * @param request - Dataset analysis parameters (dataset_name)
 * @returns Promise with the analysis response including word-level statistics
 */
export async function analyzeDataset(
    request: DatasetAnalysisRequest
): Promise<DatasetAnalysisResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), DOWNLOAD_TIMEOUT_MS);

    try {
        const response = await fetch(API_ENDPOINTS.DATASETS_ANALYZE, {
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
            throw new Error(errorData.detail || `Failed to analyze dataset: ${response.status}`);
        }

        return response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof Error && error.name === 'AbortError') {
            throw new Error('Dataset analysis timed out. The dataset may be too large.');
        }
        throw error;
    }
}
