import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fetchBenchmarkMetricsCatalog,
  fetchBenchmarkReportById,
  fetchBenchmarkReports,
  runBenchmarks,
} from '../services/benchmarksApi';
import { cancelJob } from '../services/jobsApi';
import { fetchAvailableDatasets } from '../services/datasetsApi';
import { fetchDownloadedTokenizers } from '../services/tokenizersApi';
import type {
  BenchmarkMetricCatalogCategory,
  BenchmarkReportSummary,
  BenchmarkRunResponse,
  BenchmarkRunWizardPayload,
} from '../types/api';

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
  cancelBenchmark: () => Promise<boolean>;
  error: string | null;
  clearError: () => void;
  loadReportById: (reportId: number) => Promise<void>;
  runFromWizard: (payload: BenchmarkRunWizardPayload) => Promise<boolean>;
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
  const [activeBenchmarkJobId, setActiveBenchmarkJobId] = useState<string | null>(null);
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

  const runFromWizard = useCallback(async (payload: BenchmarkRunWizardPayload) => {
    setRunningBenchmark(true);
    setError(null);
    try {
      const report = await runBenchmarks(payload, undefined, (job) => setActiveBenchmarkJobId(job.job_id));
      setActiveReport(report);
      await refreshReportSummaries(report.report_id);
      return true;
    } catch (runError) {
      setError(getErrorMessage(runError, 'Failed to run benchmark'));
      return false;
    } finally {
      setActiveBenchmarkJobId(null);
      setRunningBenchmark(false);
    }
  }, [refreshReportSummaries]);

  const cancelBenchmark = useCallback(async () => {
    if (!activeBenchmarkJobId) {
      return false;
    }
    try {
      await cancelJob(activeBenchmarkJobId);
      return true;
    } catch (cancelError) {
      setError(getErrorMessage(cancelError, 'Failed to stop benchmark'));
      return false;
    }
  }, [activeBenchmarkJobId]);

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
    cancelBenchmark,
    error,
    clearError,
    loadReportById,
    runFromWizard,
  };
};
