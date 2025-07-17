/**
 * 外部市場データ管理フック
 *
 * 外部市場データ（SP500、NASDAQ、DXY、VIX）の取得・収集機能を提供します。
 */

import { useState, useEffect, useCallback } from "react";
import { useApiCall } from "./useApiCall";

/**
 * 外部市場データの型定義
 */
export interface ExternalMarketData {
  id: number;
  symbol: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number | null;
  data_timestamp: string;
  timestamp: string;
  created_at: string;
  updated_at: string;
}

/**
 * 外部市場データ収集結果の型定義
 */
export interface ExternalMarketCollectionResult {
  success: boolean;
  fetched_count: number;
  inserted_count: number;
  message: string;
  collection_type?: string;
  latest_timestamp_before?: string;
  error?: string;
}

/**
 * 外部市場データ状態の型定義
 */
export interface ExternalMarketDataStatus {
  success: boolean;
  statistics: {
    count: number;
    symbols: string[];
    symbol_count: number;
    date_range?: {
      oldest: string;
      newest: string;
    };
  };
  latest_timestamp?: string;
  current_time: string;
  error?: string;
}

/**
 * 利用可能なシンボル
 */
export const EXTERNAL_MARKET_SYMBOLS = {
  "^GSPC": "S&P 500",
  "^IXIC": "NASDAQ Composite",
  "DX-Y.NYB": "US Dollar Index",
  "^VIX": "CBOE Volatility Index",
} as const;

export type ExternalMarketSymbol = keyof typeof EXTERNAL_MARKET_SYMBOLS;

/**
 * 外部市場データ管理フック
 */
export const useExternalMarketData = () => {
  const [data, setData] = useState<ExternalMarketData[]>([]);
  const [status, setStatus] = useState<ExternalMarketDataStatus | null>(null);

  const { execute: fetchDataApi, loading, error } = useApiCall<{ success: boolean; data: ExternalMarketData[] }>();
  const { execute: collectDataApi } = useApiCall<{ success: boolean; data: ExternalMarketCollectionResult }>();
  const { execute: fetchStatusApi } = useApiCall<{ success: boolean; data: ExternalMarketDataStatus }>();

  /**
   * 外部市場データを取得
   */
  const fetchData = useCallback(
    async (symbol?: string, limit: number = 100) => {
      const url = new URL(
        "/api/data/external-market",
        window.location.origin
      );
      if (symbol) {
        url.searchParams.set("symbol", symbol);
      }
      url.searchParams.set("limit", limit.toString());

      const result = await fetchDataApi(url.toString());
      if (result && result.success) {
        setData(result.data);
      }
    },
    [fetchDataApi]
  );

  /**
   * 最新の外部市場データを取得
   */
  const fetchLatestData = useCallback(
    async (symbol?: string, limit: number = 30) => {
      const url = new URL(
        "/api/data/external-market/latest",
        window.location.origin
      );
      if (symbol) {
        url.searchParams.set("symbol", symbol);
      }
      url.searchParams.set("limit", limit.toString());

      const result = await fetchDataApi(url.toString());
      if (result && result.success) {
        setData(result.data);
      }
    },
    [fetchDataApi]
  );

  /**
   * 外部市場データを収集
   */
  const collectData = useCallback(
    async (
      symbols?: string[],
      period: string = "1mo"
    ): Promise<ExternalMarketCollectionResult> => {
      const url = new URL(
        "/api/data/external-market/collect",
        window.location.origin
      );
      url.searchParams.set("period", period);

      if (symbols && symbols.length > 0) {
        symbols.forEach((symbol) => {
          url.searchParams.append("symbols", symbol);
        });
      }

      const result = await collectDataApi(url.toString(), { method: "POST" });
      if (result && result.success) {
        return result.data;
      }
      throw new Error(error || "外部市場データの収集に失敗しました");
    },
    [collectDataApi, error]
  );

  /**
   * 外部市場データの差分収集
   */
  const collectIncrementalData = useCallback(
    async (symbols?: string[]): Promise<ExternalMarketCollectionResult> => {
      const url = new URL(
        "/api/data/external-market/collect-incremental",
        window.location.origin
      );

      if (symbols && symbols.length > 0) {
        symbols.forEach((symbol) => {
          url.searchParams.append("symbols", symbol);
        });
      }

      const result = await collectDataApi(url.toString(), { method: "POST" });
      if (result && result.success) {
        return result.data;
      }
      throw new Error(error || "外部市場データの差分収集に失敗しました");
    },
    [collectDataApi, error]
  );

  /**
   * 外部市場データの履歴データを収集
   */
  const collectHistoricalData = useCallback(
    async (
      symbols?: string[],
      period: string = "5y",
      startDate?: string,
      endDate?: string
    ): Promise<ExternalMarketCollectionResult> => {
      const url = new URL(
        "/api/data/external-market/collect-historical",
        window.location.origin
      );
      url.searchParams.set("period", period);

      if (symbols && symbols.length > 0) {
        symbols.forEach((symbol) => {
          url.searchParams.append("symbols", symbol);
        });
      }

      if (startDate) {
        url.searchParams.set("start_date", startDate);
      }

      if (endDate) {
        url.searchParams.set("end_date", endDate);
      }

      const result = await collectDataApi(url.toString(), { method: "POST" });
      if (result && result.success) {
        return result.data;
      }
      throw new Error(error || "外部市場データの履歴収集に失敗しました");
    },
    [collectDataApi, error]
  );

  /**
   * 外部市場データの状態を取得
   */
  const fetchStatus = useCallback(async () => {
    const url = new URL(
      "/api/data/external-market/status",
      window.location.origin
    );
    const result = await fetchStatusApi(url.toString());
    if (result && result.success) {
      setStatus(result.data);
    }
  }, [fetchStatusApi]);

  /**
   * データを再取得
   */
  const refetch = useCallback(() => {
    fetchLatestData();
  }, [fetchLatestData]);

  // 初期化時にデータ状態のみを取得（データは必要に応じて別途取得）
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  return {
    data,
    loading,
    error,
    status,
    fetchData,
    fetchLatestData,
    collectData,
    collectIncrementalData,
    collectHistoricalData,
    fetchStatus,
    refetch,
  };
};
