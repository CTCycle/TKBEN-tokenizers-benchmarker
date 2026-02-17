import type {
    BenchmarkMetricCatalogResponse,
    BenchmarkReportListResponse,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    JobStartResponse,
    JobStatusResponse,
} from '../types/api';

import { API_ENDPOINTS } from '../constants';
import { waitForJobResult } from './jobsApi';

// 30 minute timeout for benchmark runs (can be very long-running)
const BENCHMARK_TIMEOUT_MS = 30 * 60 * 1000;

/**
 * Run tokenizer benchmarks on specified tokenizers using a loaded dataset.
 * @param request - Benchmark run parameters including tokenizers, dataset, and options
 * @returns Promise with the benchmark response including metrics and plots
 */
export async function runBenchmarks(
    request: BenchmarkRunRequest,
    onUpdate?: (status: JobStatusResponse) => void,
): Promise<BenchmarkRunResponse> {

    try {
        const response = await fetch(API_ENDPOINTS.BENCHMARKS_RUN, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `Failed to run benchmarks: ${response.status}`);
        }

        const job = await response.json() as JobStartResponse;
        return waitForJobResult<BenchmarkRunResponse>(job, {
            onUpdate,
            timeoutMs: BENCHMARK_TIMEOUT_MS,
        });
    } catch (error) {
        throw error;
    }
}

/**
 * Fetch benchmark metrics catalog for the benchmark wizard.
 */
export async function fetchBenchmarkMetricsCatalog(): Promise<BenchmarkMetricCatalogResponse> {
    const response = await fetch(API_ENDPOINTS.BENCHMARKS_METRICS_CATALOG, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to fetch benchmark metrics catalog: ${response.status}`);
    }

    return response.json();
}

/**
 * Fetch available persisted benchmark reports.
 */
export async function fetchBenchmarkReports(limit = 200): Promise<BenchmarkReportListResponse> {
    const params = new URLSearchParams({
        limit: String(limit),
    });
    const response = await fetch(`${API_ENDPOINTS.BENCHMARKS_REPORTS}?${params.toString()}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to fetch benchmark reports: ${response.status}`);
    }

    return response.json();
}

/**
 * Fetch a persisted benchmark report by id.
 */
export async function fetchBenchmarkReportById(reportId: number): Promise<BenchmarkRunResponse> {
    const response = await fetch(`${API_ENDPOINTS.BENCHMARKS_REPORT_BY_ID}/${reportId}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to fetch benchmark report: ${response.status}`);
    }

    return response.json();
}
