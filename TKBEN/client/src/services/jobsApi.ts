import type { JobCancelResponse, JobStartResponse, JobStatusResponse } from '../types/api';
import { API_ENDPOINTS } from '../constants';
import { parseJobStatusResponse, readJsonResponse } from './responseGuards';

const MIN_POLL_INTERVAL_MS = 250;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function fetchJobStatus(jobId: string): Promise<JobStatusResponse> {
    const response = await fetch(`${API_ENDPOINTS.JOBS}/${jobId}`);
    return parseJobStatusResponse(await readJsonResponse(response, 'Failed to fetch job status'));
}

export async function cancelJob(jobId: string): Promise<JobCancelResponse> {
    const response = await fetch(`${API_ENDPOINTS.JOBS}/${jobId}`, { method: 'DELETE' });
    return readJsonResponse(response, 'Failed to cancel job');
}

interface JobPollOptions<T> {
    onUpdate?: (status: JobStatusResponse) => void;
    timeoutMs?: number;
    parseResult?: (result: unknown) => T;
}

export async function waitForJobResult<T>(
    job: JobStartResponse,
    options: JobPollOptions<T> = {},
): Promise<T> {
    const pollIntervalMs = Math.max(MIN_POLL_INTERVAL_MS, Math.round(job.poll_interval * 1000));
    const startedAt = Date.now();

    while (true) {
        const status = await fetchJobStatus(job.job_id);
        options.onUpdate?.(status);

        if (status.status === 'completed') {
            if (status.result !== undefined && status.result !== null) {
                return options.parseResult
                    ? options.parseResult(status.result)
                    : (status.result as T);
            }
            throw new Error('Job completed without a result payload.');
        }
        if (status.status === 'failed') {
            throw new Error(status.error || 'Job failed.');
        }
        if (status.status === 'cancelled') {
            throw new Error('Job was cancelled.');
        }

        if (options.timeoutMs && Date.now() - startedAt > options.timeoutMs) {
            throw new Error('Job timed out.');
        }

        await sleep(pollIntervalMs);
    }
}
