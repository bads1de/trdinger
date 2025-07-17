/**
 * Fear & Greed Index データ管理フック
 *
 * Fear & Greed Index データの取得、収集、状態管理を行います。
 */

import { useState, useEffect, useCallback } from "react";
import { FearGreedIndexData, FearGreedIndexResponse } from "@/app/api/data/fear-greed/route";

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
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<FearGreedDataStatus | null>(null);

  /**
   * Fear & Greed Index データを取得
   */
  const fetchData = useCallback(async (
    limit: number = 30,
    startDate?: string,
    endDate?: string
  ) => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      params.set("limit", limit.toString());
      if (startDate) params.set("start_date", startDate);
      if (endDate) params.set("end_date", endDate);

      const response = await fetch(`/api/data/fear-greed?${params.toString()}`);
      
      if (!response.ok) {
        throw new Error(`データ取得に失敗しました: ${response.status}`);
      }

      const result: FearGreedIndexResponse = await response.json();
      
      if (result.success) {
        setData(result.data);
      } else {
        throw new Error(result.message);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "データ取得中にエラーが発生しました";
      setError(errorMessage);
      console.error("Fear & Greed Index データ取得エラー:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * 最新のFear & Greed Index データを取得
   */
  const fetchLatestData = useCallback(async (limit: number = 30) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/data/fear-greed/latest?limit=${limit}`);
      
      if (!response.ok) {
        throw new Error(`最新データ取得に失敗しました: ${response.status}`);
      }

      const result: FearGreedIndexResponse = await response.json();
      
      if (result.success) {
        setData(result.data);
      } else {
        throw new Error(result.message);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "最新データ取得中にエラーが発生しました";
      setError(errorMessage);
      console.error("Fear & Greed Index 最新データ取得エラー:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Fear & Greed Index データの状態を取得
   */
  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch("/api/data/fear-greed/status");
      
      if (!response.ok) {
        throw new Error(`状態取得に失敗しました: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setStatus(result.data);
      } else {
        throw new Error(result.message);
      }
    } catch (err) {
      console.error("Fear & Greed Index 状態取得エラー:", err);
    }
  }, []);

  /**
   * Fear & Greed Index データを収集
   */
  const collectData = useCallback(async (limit: number = 30): Promise<FearGreedCollectionResult> => {
    try {
      const response = await fetch(`/api/data/fear-greed/collect?limit=${limit}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      
      if (!response.ok) {
        throw new Error(`データ収集に失敗しました: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        // データ収集後に最新データを再取得
        await fetchLatestData();
        await fetchStatus();
        return result.data;
      } else {
        throw new Error(result.message);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "データ収集中にエラーが発生しました";
      console.error("Fear & Greed Index データ収集エラー:", err);
      return {
        success: false,
        message: errorMessage,
        fetched_count: 0,
        inserted_count: 0,
        error: errorMessage,
      };
    }
  }, [fetchLatestData, fetchStatus]);

  /**
   * 履歴データを収集（全期間）
   */
  const collectHistoricalData = useCallback(async (limit: number = 1000): Promise<FearGreedCollectionResult> => {
    try {
      const response = await fetch(`/api/data/fear-greed/collect-historical?limit=${limit}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      
      if (!response.ok) {
        throw new Error(`履歴データ収集に失敗しました: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        // データ収集後に最新データを再取得
        await fetchLatestData();
        await fetchStatus();
        return result.data;
      } else {
        throw new Error(result.message);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "履歴データ収集中にエラーが発生しました";
      console.error("Fear & Greed Index 履歴データ収集エラー:", err);
      return {
        success: false,
        message: errorMessage,
        fetched_count: 0,
        inserted_count: 0,
        error: errorMessage,
      };
    }
  }, [fetchLatestData, fetchStatus]);

  /**
   * 差分データを収集
   */
  const collectIncrementalData = useCallback(async (): Promise<FearGreedCollectionResult> => {
    try {
      const response = await fetch("/api/data/fear-greed/collect-incremental", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      
      if (!response.ok) {
        throw new Error(`差分データ収集に失敗しました: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        // データ収集後に最新データを再取得
        await fetchLatestData();
        await fetchStatus();
        return result.data;
      } else {
        throw new Error(result.message);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "差分データ収集中にエラーが発生しました";
      console.error("Fear & Greed Index 差分データ収集エラー:", err);
      return {
        success: false,
        message: errorMessage,
        fetched_count: 0,
        inserted_count: 0,
        error: errorMessage,
      };
    }
  }, [fetchLatestData, fetchStatus]);

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
