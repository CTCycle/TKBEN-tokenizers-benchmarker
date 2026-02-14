// API and network configuration constants

// Base URL for backend API calls (proxied by Vite in dev via `/api`)
export const API_BASE_URL = '/api';

// Timeout for dataset download operations (10 minutes)
export const DOWNLOAD_TIMEOUT_MS = 10 * 60 * 1000;

export const API_ENDPOINTS = {
    DATASETS_LIST: `${API_BASE_URL}/datasets/list`,
    DATASETS_DOWNLOAD: `${API_BASE_URL}/datasets/download`,
    DATASETS_UPLOAD: `${API_BASE_URL}/datasets/upload`,
    DATASETS_ANALYZE: `${API_BASE_URL}/datasets/analyze`,
    DATASETS_DELETE: `${API_BASE_URL}/datasets/delete`,
    DATASETS_REPORT_LATEST: `${API_BASE_URL}/datasets/reports/latest`,
    DATASETS_REPORT_BY_ID: `${API_BASE_URL}/datasets/reports`,

    TOKENIZERS_SETTINGS: `${API_BASE_URL}/tokenizers/settings`,
    TOKENIZERS_SCAN: `${API_BASE_URL}/tokenizers/scan`,
    TOKENIZERS_LIST: `${API_BASE_URL}/tokenizers/list`,
    TOKENIZERS_DOWNLOAD: `${API_BASE_URL}/tokenizers/download`,
    TOKENIZERS_UPLOAD: `${API_BASE_URL}/tokenizers/upload`,
    TOKENIZERS_CUSTOM: `${API_BASE_URL}/tokenizers/custom`,
    TOKENIZERS_REPORT_GENERATE: `${API_BASE_URL}/tokenizers/reports/generate`,
    TOKENIZERS_REPORT_LATEST: `${API_BASE_URL}/tokenizers/reports/latest`,
    TOKENIZERS_REPORT_BY_ID: `${API_BASE_URL}/tokenizers/reports`,

    KEYS: `${API_BASE_URL}/keys`,

    BENCHMARKS_RUN: `${API_BASE_URL}/benchmarks/run`,

    JOBS: `${API_BASE_URL}/jobs`,
} as const;
