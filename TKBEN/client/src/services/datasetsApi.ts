import type {
    CustomDatasetUploadResponse,
    DatasetAnalysisRequest,
    DatasetAnalysisResponse,
    DatasetDownloadRequest,
    DatasetDownloadResponse,
    DatasetListResponse,
    JobStartResponse,
    JobStatusResponse,
} from '../types/api';
import { API_ENDPOINTS, DOWNLOAD_TIMEOUT_MS } from '../constants';
import { waitForJobResult } from './jobsApi';

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
    request: DatasetDownloadRequest,
    onUpdate?: (status: JobStatusResponse) => void,
): Promise<DatasetDownloadResponse> {

    try {
        const response = await fetch(API_ENDPOINTS.DATASETS_DOWNLOAD, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `Failed to download dataset: ${response.status}`);
        }

        const job = await response.json() as JobStartResponse;
        return waitForJobResult<DatasetDownloadResponse>(job, {
            onUpdate,
            timeoutMs: DOWNLOAD_TIMEOUT_MS,
        });
    } catch (error) {
        throw error;
    }
}

/**
 * Upload a custom CSV/Excel dataset file and save it to the database.
 * @param file - The file to upload (.csv, .xlsx, or .xls)
 * @returns Promise with the upload response including histogram data
 */
export async function uploadCustomDataset(
    file: File,
    onUpdate?: (status: JobStatusResponse) => void,
): Promise<CustomDatasetUploadResponse> {

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(API_ENDPOINTS.DATASETS_UPLOAD, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `Failed to upload dataset: ${response.status}`);
        }

        const job = await response.json() as JobStartResponse;
        return waitForJobResult<CustomDatasetUploadResponse>(job, {
            onUpdate,
            timeoutMs: DOWNLOAD_TIMEOUT_MS,
        });
    } catch (error) {
        throw error;
    }
}

/**
 * Validate a dataset and compute document/word-level statistics.
 * @param request - Dataset analysis parameters (dataset_name)
 * @returns Promise with the validation response including histograms and word frequencies
 */
export async function validateDataset(
    request: DatasetAnalysisRequest,
    onUpdate?: (status: JobStatusResponse) => void,
): Promise<DatasetAnalysisResponse> {

    try {
        const response = await fetch(API_ENDPOINTS.DATASETS_ANALYZE, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `Failed to analyze dataset: ${response.status}`);
        }

        const job = await response.json() as JobStartResponse;
        return waitForJobResult<DatasetAnalysisResponse>(job, {
            onUpdate,
            timeoutMs: DOWNLOAD_TIMEOUT_MS,
        });
    } catch (error) {
        throw error;
    }
}

/**
 * Fetch the latest persisted validation report for a dataset.
 */
export async function fetchLatestDatasetReport(
    datasetName: string,
): Promise<DatasetAnalysisResponse> {
    const response = await fetch(
        `${API_ENDPOINTS.DATASETS_REPORT_LATEST}?dataset_name=${encodeURIComponent(datasetName)}`,
        {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        },
    );

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to load latest dataset report: ${response.status}`);
    }

    return response.json();
}

/**
 * Fetch a dataset validation report by id.
 */
export async function fetchDatasetReportById(
    reportId: number,
): Promise<DatasetAnalysisResponse> {
    const response = await fetch(`${API_ENDPOINTS.DATASETS_REPORT_BY_ID}/${reportId}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to fetch dataset report: ${response.status}`);
    }

    return response.json();
}

/**
 * Remove a dataset and related reports from the database.
 * @param datasetName - Dataset identifier
 */
export async function deleteDataset(datasetName: string): Promise<{ status: string; dataset_name: string; message: string }> {
    const response = await fetch(`${API_ENDPOINTS.DATASETS_DELETE}?dataset_name=${encodeURIComponent(datasetName)}`, {
        method: 'DELETE',
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to delete dataset: ${response.status}`);
    }

    return response.json();
}
