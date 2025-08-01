import { useState, useCallback, useEffect } from "react";
import { useDataFetching } from "./useDataFetching";
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

/**
 * Fear & Greed Indexデータ管理フック
 *
 * Fear & Greed Indexデータの取得、収集、ステータス管理機能を提供します。
 * データの取得、収集、ステータス確認、パラメータ設定などの機能を統合的に管理します。
 *
 * @example
 * ```tsx
 * const {
 *   data,
 *   loading,
 *   error,
 *   status,
 *   fetchData,
 *   fetchLatestData,
 *   collectData,
 *   refetch
 * } = useFearGreedData();
 *
 * // 最新データを取得
 * fetchLatestData(30);
 *
 * // データを収集
 * collectData(50);
 *
 * // 手動で再取得
 * refetch();
 * ```
 *
 * @returns {{
 *   data: FearGreedIndexData[],
 *   loading: boolean,
 *   error: string | null,
 *   status: FearGreedDataStatus | null,
 *   params: { limit: number; start_date?: string; end_date?: string },
 *   fetchData: (limit: number, startDate?: string, endDate?: string) => void,
 *   fetchLatestData: (limit: number) => Promise<void>,
 *   fetchStatus: () => Promise<void>,
 *   collectData: (limit: number) => Promise<void>,
 *   collectHistoricalData: (limit: number) => Promise<void>,
 *   collectIncrementalData: () => Promise<void>,
 *   refetch: () => Promise<void>
 * }} Fear & Greed Indexデータ管理関連の状態と操作関数
 */
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

  const { execute: executeCollection, loading: isCollecting } =
    useApiCall<FearGreedCollectionResult>();

  const handleCollection = useCallback(
    async (endpoint: string) => {
      const result = await executeCollection(endpoint, {
        method: "POST",
        onSuccess: (data) => {
          refetch();
          fetchStatus();
        },
      });

      if (result) {
        return result;
      }

      throw new Error("データ収集に失敗しました");
    },
    [executeCollection, refetch, fetchStatus]
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
    /** Fear & Greed Indexデータの配列 */
    data,
    /** データ取得中・収集中・ステータス取得中のローディング状態 */
    loading: loading || isCollecting || statusLoading,
    /** エラーメッセージ */
    error,
    /** データステータス情報 */
    status,
    /** 現在のクエリパラメータ */
    params,
    /** データを取得する関数 */
    fetchData,
    /** 最新データを取得する関数 */
    fetchLatestData,
    /** ステータスを取得する関数 */
    fetchStatus,
    /** データを収集する関数 */
    collectData,
    /** 履歴データを収集する関数 */
    collectHistoricalData,
    /** 差分データを収集する関数 */
    collectIncrementalData,
    /** データを再取得する関数 */
    refetch,
  };
};
