/**
 * Fear & Greed Index データ管理フック
 *
 * Fear & Greed Index データの取得、収集、状態管理を行います。
 */

import { useState, useEffect, useCallback } from "react";
import { FearGreedIndexData, FearGreedIndexResponse } from "@/app/api/data/fear-greed/route";
import { useApiCall } from "./useApiCall";

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

/**
 * Fear & Greed Index データ管理フック
 */
export const useFearGreedData = () => {
  const [data, setData] = useState<FearGreedIndexData[]>([]);
  const [status, setStatus] = useState<FearGreedDataStatus | null>(null);

  const { execute: fetchDataApi, loading, error } = useApiCall<FearGreedIndexResponse>();
  const { execute: fetchStatusApi } = useApiCall<{ success: boolean; data: FearGreedDataStatus }>();
  const { execute: collectDataApi } = useApiCall<{ success: boolean; data: FearGreedCollectionResult }>();

  /**
   * Fear & Greed Index データを取得
   */
  const fetchData = useCallback(async (
    limit: number = 30,
    startDate?: string,
    endDate?: string
  ) => {
    const params = new URLSearchParams();
    params.set("limit", limit.toString());
    if (startDate) params.set("start_date", startDate);
    if (endDate) params.set("end_date", endDate);

    const result = await fetchDataApi(`/api/data/fear-greed?${params.toString()}`);
    if (result && result.success) {
      setData(result.data);
    }
  }, [fetchDataApi]);

  /**
   * 最新のFear & Greed Index データを取得
   */
  const fetchLatestData = useCallback(async (limit: number = 30) => {
    const result = await fetchDataApi(`/api/data/fear-greed/latest?limit=${limit}`);
    if (result && result.success) {
      setData(result.data);
    }
  }, [fetchDataApi]);

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
  const collectData = useCallback(async (limit: number = 30): Promise<FearGreedCollectionResult> => {
    const result = await collectDataApi(`/api/data/fear-greed/collect?limit=${limit}`, {
      method: "POST",
    });

    if (result && result.success) {
      await fetchLatestData();
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
  }, [collectDataApi, fetchLatestData, fetchStatus, error]);

  /**
   * 履歴データを収集（全期間）
   */
  const collectHistoricalData = useCallback(async (limit: number = 1000): Promise<FearGreedCollectionResult> => {
    const result = await collectDataApi(`/api/data/fear-greed/collect-historical?limit=${limit}`, {
      method: "POST",
    });

    if (result && result.success) {
      await fetchLatestData();
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
  }, [collectDataApi, fetchLatestData, fetchStatus, error]);

  /**
   * 差分データを収集
   */
  const collectIncrementalData = useCallback(async (): Promise<FearGreedCollectionResult> => {
    const result = await collectDataApi("/api/data/fear-greed/collect-incremental", {
      method: "POST",
    });

    if (result && result.success) {
      await fetchLatestData();
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
  }, [collectDataApi, fetchLatestData, fetchStatus, error]);

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
    fetchData,
    fetchLatestData,
    fetchStatus,
    collectData,
    collectHistoricalData,
    collectIncrementalData,
  };
};
