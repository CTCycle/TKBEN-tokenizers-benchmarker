import type { BenchmarkRunRequest, BenchmarkRunResponse } from '../types/api';

import { API_ENDPOINTS } from '../constants';

// 30 minute timeout for benchmark runs (can be very long-running)
const BENCHMARK_TIMEOUT_MS = 30 * 60 * 1000;

/**
 * Run tokenizer benchmarks on specified tokenizers using a loaded dataset.
 * @param request - Benchmark run parameters including tokenizers, dataset, and options
 * @returns Promise with the benchmark response including metrics and plots
 */
export async function runBenchmarks(
    request: BenchmarkRunRequest
): Promise<BenchmarkRunResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), BENCHMARK_TIMEOUT_MS);

    try {
        const response = await fetch(API_ENDPOINTS.BENCHMARKS_RUN, {
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
            throw new Error(errorData.detail || `Failed to run benchmarks: ${response.status}`);
        }

        return response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof Error && error.name === 'AbortError') {
            throw new Error('Benchmark run timed out. Try reducing the number of documents or tokenizers.');
        }
        throw error;
    }
}
