/**
 * 外部市場データ管理フック
 *
 * 外部市場データ（SP500、NASDAQ、DXY、VIX）の取得・収集機能を提供します。
 */

import { useState, useEffect, useCallback } from "react";
import { BACKEND_API_URL } from "@/constants";

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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [status, setStatus] = useState<ExternalMarketDataStatus | null>(null);

  /**
   * 外部市場データを取得
   */
  const fetchData = useCallback(
    async (symbol?: string, limit: number = 100) => {
      setLoading(true);
      setError("");

      try {
        const url = new URL(
          "/api/data/external-market",
          window.location.origin
        );
        if (symbol) {
          url.searchParams.set("symbol", symbol);
        }
        url.searchParams.set("limit", limit.toString());

        const response = await fetch(url.toString());
        const result = await response.json();

        if (!response.ok) {
          throw new Error(
            result.message || "外部市場データの取得に失敗しました"
          );
        }

        if (result.success) {
          setData(result.data);
        } else {
          throw new Error(
            result.message || "外部市場データの取得に失敗しました"
          );
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error
            ? err.message
            : "外部市場データの取得に失敗しました";
        setError(errorMessage);
        console.error("外部市場データ取得エラー:", err);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  /**
   * 最新の外部市場データを取得
   */
  const fetchLatestData = useCallback(
    async (symbol?: string, limit: number = 30) => {
      setLoading(true);
      setError("");

      try {
        const url = new URL(
          "/api/data/external-market/latest",
          window.location.origin
        );
        if (symbol) {
          url.searchParams.set("symbol", symbol);
        }
        url.searchParams.set("limit", limit.toString());

        const response = await fetch(url.toString());
        const result = await response.json();

        if (!response.ok) {
          throw new Error(
            result.message || "最新外部市場データの取得に失敗しました"
          );
        }

        if (result.success) {
          setData(result.data);
        } else {
          throw new Error(
            result.message || "最新外部市場データの取得に失敗しました"
          );
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error
            ? err.message
            : "最新外部市場データの取得に失敗しました";
        setError(errorMessage);
        console.error("最新外部市場データ取得エラー:", err);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  /**
   * 外部市場データを収集
   */
  const collectData = useCallback(
    async (
      symbols?: string[],
      period: string = "1mo"
    ): Promise<ExternalMarketCollectionResult> => {
      try {
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

        const response = await fetch(url.toString(), {
          method: "POST",
        });

        const result = await response.json();

        if (!response.ok) {
          throw new Error(
            result.message || "外部市場データの収集に失敗しました"
          );
        }

        if (result.success) {
          return result.data;
        } else {
          throw new Error(
            result.message || "外部市場データの収集に失敗しました"
          );
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error
            ? err.message
            : "外部市場データの収集に失敗しました";
        console.error("外部市場データ収集エラー:", err);
        throw new Error(errorMessage);
      }
    },
    []
  );

  /**
   * 外部市場データの差分収集
   */
  const collectIncrementalData = useCallback(
    async (symbols?: string[]): Promise<ExternalMarketCollectionResult> => {
      try {
        const url = new URL(
          "/api/data/external-market/collect-incremental",
          window.location.origin
        );

        if (symbols && symbols.length > 0) {
          symbols.forEach((symbol) => {
            url.searchParams.append("symbols", symbol);
          });
        }

        const response = await fetch(url.toString(), {
          method: "POST",
        });

        const result = await response.json();

        if (!response.ok) {
          throw new Error(
            result.message || "外部市場データの差分収集に失敗しました"
          );
        }

        if (result.success) {
          return result.data;
        } else {
          throw new Error(
            result.message || "外部市場データの差分収集に失敗しました"
          );
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error
            ? err.message
            : "外部市場データの差分収集に失敗しました";
        console.error("外部市場データ差分収集エラー:", err);
        throw new Error(errorMessage);
      }
    },
    []
  );

  /**
   * 外部市場データの状態を取得
   */
  const fetchStatus = useCallback(async () => {
    try {
      const url = new URL(
        "/api/data/external-market/status",
        window.location.origin
      );
      const response = await fetch(url.toString());
      const result = await response.json();

      if (!response.ok) {
        throw new Error(
          result.message || "外部市場データ状態の取得に失敗しました"
        );
      }

      if (result.success) {
        setStatus(result.data);
      } else {
        throw new Error(
          result.message || "外部市場データ状態の取得に失敗しました"
        );
      }
    } catch (err) {
      console.error("外部市場データ状態取得エラー:", err);
    }
  }, []);

  /**
   * データを再取得
   */
  const refetch = useCallback(() => {
    fetchLatestData();
  }, [fetchLatestData]);

  // 初期化時にデータ状態を取得
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
    fetchStatus,
    refetch,
  };
};
