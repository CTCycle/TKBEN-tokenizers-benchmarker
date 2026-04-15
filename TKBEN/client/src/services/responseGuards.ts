import type { JobStartResponse, JobStatusResponse } from '../types/api';

type JsonRecord = Record<string, unknown>;

const isRecord = (value: unknown): value is JsonRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const readString = (record: JsonRecord, key: string): string | null => {
  const value = record[key];
  return typeof value === 'string' ? value : null;
};

const readNumber = (record: JsonRecord, key: string): number | null => {
  const value = record[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
};

export const parseJobStartResponse = (value: unknown): JobStartResponse => {
  if (!isRecord(value)) {
    throw new Error('Invalid job start response payload.');
  }

  const jobId = readString(value, 'job_id');
  const jobType = readString(value, 'job_type');
  const status = readString(value, 'status');
  const message = readString(value, 'message');
  const pollInterval = readNumber(value, 'poll_interval');

  if (!jobId || !jobType || !status || !message || pollInterval === null) {
    throw new Error('Job start response is missing required fields.');
  }

  return {
    job_id: jobId,
    job_type: jobType,
    status,
    message,
    poll_interval: pollInterval,
  };
};

export const parseJobStatusResponse = (value: unknown): JobStatusResponse => {
  if (!isRecord(value)) {
    throw new Error('Invalid job status response payload.');
  }

  const jobId = readString(value, 'job_id');
  const jobType = readString(value, 'job_type');
  const status = readString(value, 'status');
  const progress = readNumber(value, 'progress');

  if (!jobId || !jobType || !status || progress === null) {
    throw new Error('Job status response is missing required fields.');
  }

  const errorValue = value.error;
  const result = Object.prototype.hasOwnProperty.call(value, 'result')
    ? value.result ?? null
    : undefined;

  return {
    job_id: jobId,
    job_type: jobType,
    status,
    progress,
    result,
    error: typeof errorValue === 'string' ? errorValue : null,
  };
};

export const parseRecordPayload = <T>(
  value: unknown,
  label: string,
): T => {
  if (!isRecord(value)) {
    throw new Error(`Invalid ${label} payload.`);
  }
  return value as T;
};
