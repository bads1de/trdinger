import { useState, useCallback, useEffect } from "react";
import { useDataFetching } from "./useDataFetching";
import { usePostRequest } from "./usePostRequest";
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
  const { data, loading, error, params, setParams, refetch } = useDataFetching<
    FearGreedIndexData,
    { limit: number; start_date?: string; end_date?: string }
  >({
    endpoint: "/api/fear-greed/latest",
    initialParams: { limit: 30 },
    dataPath: "data.data",
    disableAutoFetch: false,
    errorMessage: "Fear & Greed Index データの取得に失敗しました",
  });

  // ステータス機能を有効化
  const [status, setStatus] = useState<FearGreedDataStatus | null>(null);
  const { execute: fetchStatusData, loading: statusLoading } =
    useApiCall<FearGreedDataStatus>();

  const fetchStatus = useCallback(async () => {
    await fetchStatusData("/api/fear-greed/status", {
      method: "GET",
      onSuccess: (response) => {
        setStatus(response);
      },
      onError: (error) => {
        console.error("Fear & Greed Index ステータス取得エラー:", error);
      },
    });
  }, [fetchStatusData]);

  const { sendPostRequest, isLoading: isCollecting } =
    usePostRequest<FearGreedCollectionResult>();

  const handleCollection = useCallback(
    async (endpoint: string) => {
      const { success, data, error } = await sendPostRequest(endpoint);
      if (success && data) {
        refetch();
        fetchStatus();
        return data;
      }
      throw new Error(error || "データ収集に失敗しました");
    },
    [sendPostRequest, refetch, fetchStatus]
  );

  const collectData = useCallback(
    (limit: number = 30) =>
      handleCollection(`/api/fear-greed/collect?limit=${limit}`),
    [handleCollection]
  );

  const collectHistoricalData = useCallback(
    (limit: number = 1000) =>
      handleCollection(`/api/fear-greed/collect-historical?limit=${limit}`),
    [handleCollection]
  );

  const collectIncrementalData = useCallback(
    () => handleCollection("/api/fear-greed/collect"),
    [handleCollection]
  );

  const fetchData = useCallback(
    (limit: number = 30, startDate?: string, endDate?: string) => {
      setParams({ limit, start_date: startDate, end_date: endDate });
    },
    [setParams]
  );

  const fetchLatestData = useCallback(
    async (limit: number = 30) => {
      // データが空の場合は、まずデータ収集を試行
      if (data.length === 0) {
        try {
          await collectData(30);
        } catch (error) {
          console.warn("自動データ収集に失敗しました:", error);
        }
      }
      setParams({ limit, start_date: undefined, end_date: undefined });
    },
    [setParams, data.length, collectData]
  );

  return {
    data,
    loading: loading || isCollecting || statusLoading,
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
