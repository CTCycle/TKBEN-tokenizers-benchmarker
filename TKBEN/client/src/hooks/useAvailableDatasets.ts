import { useCallback, useState } from 'react';
import { fetchAvailableDatasets } from '../services/datasetsApi';
import type { DatasetPreviewItem } from '../types/api';

type UseAvailableDatasetsResult = {
  availableDatasets: DatasetPreviewItem[];
  datasetsLoading: boolean;
  refreshAvailableDatasets: () => Promise<DatasetPreviewItem[]>;
};

export const useAvailableDatasets = (): UseAvailableDatasetsResult => {
  const [availableDatasets, setAvailableDatasets] = useState<DatasetPreviewItem[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);

  const refreshAvailableDatasets = useCallback(async () => {
    setDatasetsLoading(true);
    try {
      const response = await fetchAvailableDatasets();
      setAvailableDatasets(response.datasets);
      return response.datasets;
    } finally {
      setDatasetsLoading(false);
    }
  }, []);

  return {
    availableDatasets,
    datasetsLoading,
    refreshAvailableDatasets,
  };
};
