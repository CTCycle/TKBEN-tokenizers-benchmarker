import type {
    CustomDatasetUploadResponse,
    DatasetAnalysisRequest,
    DatasetAnalysisResponse,
    DatasetDownloadRequest,
    DatasetDownloadResponse,
    DatasetListResponse,
    DatasetMetricCatalogResponse,
    JobStatusResponse,
} from '../types/api';
import { API_ENDPOINTS, DOWNLOAD_TIMEOUT_MS } from '../constants';
import { waitForJobResult } from './jobsApi';
import { parseRecordPayload, readJobStartResponse, readJsonResponse } from './responseGuards';

const sanitizeDatasetJobErrorMessage = (error: unknown, fallback: string): string => {
    const message = error instanceof Error ? error.message : fallback;
    const normalized = message.toLowerCase();

    if (normalized.includes('too many sql variables')) {
        return 'Dataset validation could not be completed because the backend hit a database batching limit. Please retry after updating the backend fix.';
    }

    if (normalized.includes('sqlite') || normalized.includes('sqlalchemy')) {
        return 'Dataset validation failed while persisting analysis results. Please retry or inspect the backend logs for details.';
    }

    return message;
};

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
    return readJsonResponse(response, 'Failed to fetch datasets');
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
    const response = await fetch(API_ENDPOINTS.DATASETS_DOWNLOAD, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
    });

    const job = await readJobStartResponse(response, 'Failed to download dataset');
    return waitForJobResult<DatasetDownloadResponse>(job, {
        onUpdate,
        timeoutMs: DOWNLOAD_TIMEOUT_MS,
        parseResult: (result) => parseRecordPayload<DatasetDownloadResponse>(result, 'dataset download job result'),
    });
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

    const response = await fetch(API_ENDPOINTS.DATASETS_UPLOAD, {
        method: 'POST',
        body: formData,
    });

    const job = await readJobStartResponse(response, 'Failed to upload dataset');
    return waitForJobResult<CustomDatasetUploadResponse>(job, {
        onUpdate,
        timeoutMs: DOWNLOAD_TIMEOUT_MS,
        parseResult: (result) => parseRecordPayload<CustomDatasetUploadResponse>(result, 'dataset upload job result'),
    });
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
    const response = await fetch(API_ENDPOINTS.DATASETS_ANALYZE, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
    });

    const job = await readJobStartResponse(response, 'Failed to analyze dataset');
    try {
        return await waitForJobResult<DatasetAnalysisResponse>(job, {
            onUpdate,
            timeoutMs: DOWNLOAD_TIMEOUT_MS,
            parseResult: (result) => parseRecordPayload<DatasetAnalysisResponse>(result, 'dataset analysis job result'),
        });
    } catch (error) {
        throw new Error(sanitizeDatasetJobErrorMessage(error, 'Failed to analyze dataset'));
    }
}

/**
 * Fetch dataset metrics catalog for validation wizard.
 */
export async function fetchDatasetMetricsCatalog(): Promise<DatasetMetricCatalogResponse> {
    const response = await fetch(API_ENDPOINTS.DATASETS_METRICS_CATALOG, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    });

    return readJsonResponse(response, 'Failed to fetch metrics catalog');
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

    return readJsonResponse(response, 'Failed to load latest dataset report');
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

    return readJsonResponse(response, 'Failed to fetch dataset report');
}

/**
 * Remove a dataset and related reports from the database.
 * @param datasetName - Dataset identifier
 */
export async function deleteDataset(datasetName: string): Promise<{ status: string; dataset_name: string; message: string }> {
    const response = await fetch(`${API_ENDPOINTS.DATASETS_DELETE}?dataset_name=${encodeURIComponent(datasetName)}`, {
        method: 'DELETE',
    });

    return readJsonResponse(response, 'Failed to delete dataset');
}
