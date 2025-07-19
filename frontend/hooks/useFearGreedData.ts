/**
 * Fear & Greed Index データ管理フック
 *
 * Fear & Greed Index データの取得、収集、状態管理を行います。
 */

import { useState, useCallback, useEffect } from "react";
import {
  FearGreedIndexData,
  FearGreedIndexResponse,
} from "@/app/api/data/fear-greed/route";
import { useApiCall } from "./useApiCall";
import { useDataFetching } from "./useDataFetching";

/**
 * Fear & Greed Index データ状態の型定義
 */
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

/**
 * Fear & Greed Index 収集結果の型定義
 */
export interface FearGreedCollectionResult {
  success: boolean;
  message: string;
  fetched_count: number;
  inserted_count: number;
  collection_type?: string;
  error?: string;
}

interface FearGreedParams {
  limit: number;
  start_date?: string;
  end_date?: string;
}

/**
 * Fear & Greed Index データ管理フック
 */
export const useFearGreedData = () => {
  const [status, setStatus] = useState<FearGreedDataStatus | null>(null);

  // 基本的なデータ取得は共通フックを使用
  const { data, loading, error, params, setParams, refetch } = useDataFetching<
    FearGreedIndexData,
    FearGreedParams
  >({
    endpoint: "/api/data/fear-greed/latest",
    initialParams: { limit: 30 },
    dataPath: "data",
    errorMessage: "Fear & Greed Index データの取得中にエラーが発生しました",
  });

  // 収集機能用のAPI呼び出し
  const { execute: fetchStatusApi } = useApiCall<{
    success: boolean;
    data: FearGreedDataStatus;
  }>();
  const { execute: collectDataApi } = useApiCall<{
    success: boolean;
    data: FearGreedCollectionResult;
  }>();

  /**
   * Fear & Greed Index データを取得（パラメータ指定）
   */
  const fetchData = useCallback(
    async (limit: number = 30, startDate?: string, endDate?: string) => {
      setParams({
        limit,
        start_date: startDate,
        end_date: endDate,
      });
    },
    [setParams]
  );

  /**
   * 最新のFear & Greed Index データを取得
   */
  const fetchLatestData = useCallback(
    async (limit: number = 30) => {
      setParams({ limit });
    },
    [setParams]
  );

  /**
   * Fear & Greed Index データの状態を取得
   */
  const fetchStatus = useCallback(async () => {
    const result = await fetchStatusApi("/api/data/fear-greed/status");
    if (result && result.success) {
      setStatus(result.data);
    }
  }, [fetchStatusApi]);

  /**
   * Fear & Greed Index データを収集
   */
  const collectData = useCallback(
    async (limit: number = 30): Promise<FearGreedCollectionResult> => {
      const result = await collectDataApi(
        `/api/data/fear-greed/collect?limit=${limit}`,
        {
          method: "POST",
        }
      );

      if (result && result.success) {
        await refetch();
        await fetchStatus();
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

  /**
   * 履歴データを収集（全期間）
   */
  const collectHistoricalData = useCallback(
    async (limit: number = 1000): Promise<FearGreedCollectionResult> => {
      const result = await collectDataApi(
        `/api/data/fear-greed/collect-historical?limit=${limit}`,
        {
          method: "POST",
        }
      );

      if (result && result.success) {
        await refetch();
        await fetchStatus();
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

  /**
   * 差分データを収集
   */
  const collectIncrementalData =
    useCallback(async (): Promise<FearGreedCollectionResult> => {
      const result = await collectDataApi(
        "/api/data/fear-greed/collect-incremental",
        {
          method: "POST",
        }
      );

      if (result && result.success) {
        await refetch();
        await fetchStatus();
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

  // 初回データ取得
  useEffect(() => {
    fetchLatestData();
    fetchStatus();
  }, [fetchLatestData, fetchStatus]);

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
