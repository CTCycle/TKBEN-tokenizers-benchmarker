import type { BenchmarkRunRequest, BenchmarkRunResponse, JobStartResponse, JobStatusResponse } from '../types/api';

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
