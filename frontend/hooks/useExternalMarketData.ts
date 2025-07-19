import { useState, useCallback } from "react";
import { useApiCall } from "./useApiCall";
import { BACKEND_API_URL } from "@/constants";

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

export interface ExternalMarketCollectionResult {
  success: boolean;
  fetched_count: number;
  inserted_count: number;
  message: string;
  collection_type?: string;
  latest_timestamp_before?: string;
  error?: string;
}

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

export const EXTERNAL_MARKET_SYMBOLS = {
  "^GSPC": "S&P 500",
  "^IXIC": "NASDAQ Composite",
  "DX-Y.NYB": "US Dollar Index",
  "^VIX": "CBOE Volatility Index",
} as const;

export type ExternalMarketSymbol = keyof typeof EXTERNAL_MARKET_SYMBOLS;

export const useExternalMarketData = () => {
  const [data, setData] = useState<ExternalMarketData[]>([]);
  const status: ExternalMarketDataStatus | null = null;

  const {
    execute: fetchDataApi,
    loading,
    error,
  } = useApiCall<{ success: boolean; data: { data: ExternalMarketData[] } }>();
  const { execute: collectDataApi } = useApiCall<{
    success: boolean;
    data: ExternalMarketCollectionResult;
  }>();

  const fetchData = useCallback(
    async (symbol?: string, limit: number = 100) => {
      const url = new URL("/api/external-market/", BACKEND_API_URL);
      if (symbol) {
        url.searchParams.set("symbol", symbol);
      }
      url.searchParams.set("limit", limit.toString());

      const result = await fetchDataApi(url.toString());
      if (result && result.success) {
        setData(result.data.data);
      }
    },
    [fetchDataApi]
  );

  const fetchLatestData = useCallback(
    async (symbol?: string, limit: number = 30) => {
      const url = new URL("/api/external-market/latest", BACKEND_API_URL);
      if (symbol) {
        url.searchParams.set("symbol", symbol);
      }
      url.searchParams.set("limit", limit.toString());

      const result = await fetchDataApi(url.toString());
      if (result && result.success) {
        setData(result.data.data);
      }
    },
    [fetchDataApi]
  );

  const collectData = useCallback(
    async (
      symbols?: string[],
      _period: string = "1mo" // period is not used in backend, but kept for compatibility
    ): Promise<ExternalMarketCollectionResult> => {
      const url = new URL("/api/external-market/collect", BACKEND_API_URL);

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

  const collectIncrementalData = useCallback(
    async (symbols?: string[]): Promise<ExternalMarketCollectionResult> => {
      // This now calls the same endpoint as collectData, as there is no specific incremental endpoint
      return collectData(symbols);
    },
    [collectData]
  );

  const collectHistoricalData = useCallback(
    async (
      symbols?: string[],
      _period: string = "5y",
      _startDate?: string,
      _endDate?: string
    ): Promise<ExternalMarketCollectionResult> => {
      // Backend does not support period, startDate, endDate for historical collection via this endpoint.
      // It collects based on its own logic. We will call the collect endpoint.
      return collectData(symbols);
    },
    [collectData]
  );

  // ステータス機能は削除されたエンドポイントのため、一時的に無効化
  const fetchStatus = useCallback(async () => {
    // 何もしない
  }, []);

  const refetch = useCallback(() => {
    fetchLatestData();
  }, [fetchLatestData]);

  // useEffect(() => {
  //   fetchStatus();
  // }, [fetchStatus]);

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
