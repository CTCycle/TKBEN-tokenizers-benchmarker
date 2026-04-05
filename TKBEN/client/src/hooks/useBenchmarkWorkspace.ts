import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fetchBenchmarkMetricsCatalog,
  fetchBenchmarkReportById,
  fetchBenchmarkReports,
  runBenchmarks,
} from '../services/benchmarksApi';
import { fetchAvailableDatasets } from '../services/datasetsApi';
import { fetchDownloadedTokenizers } from '../services/tokenizersApi';
import type {
  BenchmarkMetricCatalogCategory,
  BenchmarkReportSummary,
  BenchmarkRunResponse,
} from '../types/api';

export type BenchmarkRunPayload = {
  tokenizers: string[];
  dataset_name: string;
  config: {
    max_documents?: number;
    warmup_trials: number;
    timed_trials: number;
    batch_size: number;
    seed: number;
    parallelism: number;
    include_lm_metrics: boolean;
  };
  run_name: string;
  selected_metric_keys: string[];
};

type BenchmarkWorkspaceResult = {
  tokenizers: string[];
  datasets: string[];
  metricCategories: BenchmarkMetricCatalogCategory[];
  reports: BenchmarkReportSummary[];
  selectedReportId: number | null;
  activeReport: BenchmarkRunResponse | null;
  loadingPage: boolean;
  loadingReport: boolean;
  runningBenchmark: boolean;
  error: string | null;
  clearError: () => void;
  loadReportById: (reportId: number) => Promise<void>;
  runFromWizard: (payload: BenchmarkRunPayload) => Promise<boolean>;
};

const getErrorMessage = (error: unknown, fallback: string): string =>
  error instanceof Error ? error.message : fallback;

export const useBenchmarkWorkspace = (): BenchmarkWorkspaceResult => {
  const [tokenizers, setTokenizers] = useState<string[]>([]);
  const [datasets, setDatasets] = useState<string[]>([]);
  const [metricCategories, setMetricCategories] = useState<BenchmarkMetricCatalogCategory[]>([]);
  const [reports, setReports] = useState<BenchmarkReportSummary[]>([]);
  const [selectedReportId, setSelectedReportId] = useState<number | null>(null);
  const [activeReport, setActiveReport] = useState<BenchmarkRunResponse | null>(null);
  const [loadingPage, setLoadingPage] = useState(true);
  const [loadingReport, setLoadingReport] = useState(false);
  const [runningBenchmark, setRunningBenchmark] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const selectedReportIdRef = useRef<number | null>(null);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const setSelectedReport = useCallback((reportId: number | null) => {
    selectedReportIdRef.current = reportId;
    setSelectedReportId(reportId);
  }, []);

  const loadReportById = useCallback(async (reportId: number) => {
    setLoadingReport(true);
    try {
      const report = await fetchBenchmarkReportById(reportId);
      setError(null);
      setActiveReport(report);
      setSelectedReport(reportId);
    } catch (loadError) {
      setError(getErrorMessage(loadError, 'Failed to load report'));
    } finally {
      setLoadingReport(false);
    }
  }, [setSelectedReport]);

  const refreshReportSummaries = useCallback(async (preferredReportId?: number | null) => {
    const listResponse = await fetchBenchmarkReports(200);
    const list = listResponse.reports ?? [];
    setReports(list);

    const targetReportId = preferredReportId ?? selectedReportIdRef.current ?? list[0]?.report_id ?? null;
    if (targetReportId) {
      await loadReportById(targetReportId);
      return;
    }

    setActiveReport(null);
    setSelectedReport(null);
  }, [loadReportById, setSelectedReport]);

  useEffect(() => {
    const loadInitial = async () => {
      setLoadingPage(true);
      setError(null);
      try {
        const [tokenizerResponse, datasetResponse, categoryResponse] = await Promise.all([
          fetchDownloadedTokenizers(),
          fetchAvailableDatasets(),
          fetchBenchmarkMetricsCatalog(),
        ]);
        setTokenizers(tokenizerResponse.tokenizers.map((item) => item.tokenizer_name));
        setDatasets(datasetResponse.datasets.map((item) => item.dataset_name));
        setMetricCategories(categoryResponse.categories ?? []);
        await refreshReportSummaries();
      } catch (loadError) {
        setError(getErrorMessage(loadError, 'Failed to load benchmark workspace'));
      } finally {
        setLoadingPage(false);
      }
    };

    void loadInitial();
  }, [refreshReportSummaries]);

  const runFromWizard = useCallback(async (payload: BenchmarkRunPayload) => {
    setRunningBenchmark(true);
    setError(null);
    try {
      const report = await runBenchmarks(payload);
      setActiveReport(report);
      await refreshReportSummaries(report.report_id);
      return true;
    } catch (runError) {
      setError(getErrorMessage(runError, 'Failed to run benchmark'));
      return false;
    } finally {
      setRunningBenchmark(false);
    }
  }, [refreshReportSummaries]);

  return {
    tokenizers,
    datasets,
    metricCategories,
    reports,
    selectedReportId,
    activeReport,
    loadingPage,
    loadingReport,
    runningBenchmark,
    error,
    clearError,
    loadReportById,
    runFromWizard,
  };
};
