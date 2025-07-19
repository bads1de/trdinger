import { useState, useCallback, useEffect } from "react";
import { useApiCall } from "./useApiCall";
import { BACKEND_API_URL } from "@/constants";

export interface FearGreedIndexData {
  id: number;
  value: number;
  value_classification: string;
  data_timestamp: string;
  timestamp: string;
  created_at: string;
  updated_at: string;
}

export interface FearGreedDataStatus {
  success: boolean;
  data_range: {
    oldest_data: string | null;
    newest_data: string | null;
    total_count: number;
  };
  latest_timestamp: string | null;
  current_time: string;
  error?: string;
}

export interface FearGreedCollectionResult {
  success: boolean;
  message: string;
  fetched_count: number;
  inserted_count: number;
  collection_type?: string;
  error?: string;
}

export const useFearGreedData = () => {
  const [data, setData] = useState<FearGreedIndexData[]>([]);
  const [status, setStatus] = useState<FearGreedDataStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [params, setParams] = useState<{
    limit: number;
    start_date?: string;
    end_date?: string;
  }>({ limit: 30 });

  const { execute: fetchStatusApi } = useApiCall<{
    success: boolean;
    data: FearGreedDataStatus;
  }>();
  const { execute: collectDataApi } = useApiCall<{
    success: boolean;
    data: FearGreedCollectionResult;
  }>();

  const fetchDataInternal = useCallback(
    async (currentParams: typeof params) => {
      setLoading(true);
      setError(null);
      try {
        const apiUrl = new URL("/api/fear-greed/data", BACKEND_API_URL);
        apiUrl.searchParams.set("limit", currentParams.limit.toString());
        if (currentParams.start_date) {
          apiUrl.searchParams.set("start_date", currentParams.start_date);
        }
        if (currentParams.end_date) {
          apiUrl.searchParams.set("end_date", currentParams.end_date);
        }

        const response = await fetch(apiUrl.toString());
        const result = await response.json();

        if (!result.success) {
          throw new Error(
            result.message || "Fear & Greed Index データ取得に失敗しました"
          );
        }
        setData(result.data?.data || result.data || []);
      } catch (e) {
        const errorMessage = e instanceof Error ? e.message : "不明なエラー";
        setError(errorMessage);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const refetch = useCallback(() => {
    fetchDataInternal(params);
  }, [params, fetchDataInternal]);

  useEffect(() => {
    fetchDataInternal(params);
  }, [params, fetchDataInternal]);

  const fetchData = useCallback(
    async (limit: number = 30, startDate?: string, endDate?: string) => {
      setParams({ limit, start_date: startDate, end_date: endDate });
    },
    []
  );

  const fetchLatestData = useCallback(async (limit: number = 30) => {
    setParams({ limit });
  }, []);

  const fetchStatus = useCallback(async () => {
    // This still uses a frontend route, assuming it might be more complex
    // or has other reasons to be a route. If not, this should also be changed.
    const result = await fetchStatusApi("/api/data/fear-greed/status");
    if (result && result.success) {
      setStatus(result.data);
    }
  }, [fetchStatusApi]);

  const collectData = useCallback(
    async (limit: number = 30): Promise<FearGreedCollectionResult> => {
      const result = await collectDataApi(
        `/api/data/fear-greed/collect?limit=${limit}`,
        { method: "POST" }
      );
      if (result && result.success) {
        refetch();
        fetchStatus();
        return result.data;
      }
      return {
        success: false,
        message: "データ収集に失敗しました",
        fetched_count: 0,
        inserted_count: 0,
        error: error || "Unknown error",
      };
    },
    [collectDataApi, refetch, fetchStatus, error]
  );

  const collectHistoricalData = useCallback(
    async (limit: number = 1000): Promise<FearGreedCollectionResult> => {
      const result = await collectDataApi(
        `/api/data/fear-greed/collect-historical?limit=${limit}`,
        { method: "POST" }
      );
      if (result && result.success) {
        refetch();
        fetchStatus();
        return result.data;
      }
      return {
        success: false,
        message: "履歴データ収集に失敗しました",
        fetched_count: 0,
        inserted_count: 0,
        error: error || "Unknown error",
      };
    },
    [collectDataApi, refetch, fetchStatus, error]
  );

  const collectIncrementalData =
    useCallback(async (): Promise<FearGreedCollectionResult> => {
      const result = await collectDataApi(
        "/api/data/fear-greed/collect-incremental",
        { method: "POST" }
      );
      if (result && result.success) {
        refetch();
        fetchStatus();
        return result.data;
      }
      return {
        success: false,
        message: "差分データ収集に失敗しました",
        fetched_count: 0,
        inserted_count: 0,
        error: error || "Unknown error",
      };
    }, [collectDataApi, refetch, fetchStatus, error]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  return {
    data,
    loading,
    error,
    status,
    params,
    fetchData,
    fetchLatestData,
    fetchStatus,
    collectData,
    collectHistoricalData,
    collectIncrementalData,
    refetch,
  };
};
