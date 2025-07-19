import { useState, useEffect, useCallback } from "react";
import { PriceData, TimeFrame } from "@/types/market-data";
import { BACKEND_API_URL } from "@/constants";

interface OhlcvParams {
  symbol: string;
  timeframe: TimeFrame;
  limit: number;
  startDate?: string;
  endDate?: string;
}

export const useOhlcvData = (
  symbol: string,
  timeframe: TimeFrame,
  initialLimit = 100
) => {
  const [data, setData] = useState<PriceData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState(initialLimit);

  const fetchData = useCallback(async (params: OhlcvParams) => {
    setLoading(true);
    setError(null);

    try {
      // バックエンドAPIのURLを構築
      const apiUrl = new URL("/api/market-data/ohlcv", BACKEND_API_URL);
      apiUrl.searchParams.set("symbol", params.symbol);
      apiUrl.searchParams.set("timeframe", params.timeframe);
      apiUrl.searchParams.set("limit", params.limit.toString());

      if (params.startDate) {
        apiUrl.searchParams.set("start_date", params.startDate);
      }
      if (params.endDate) {
        apiUrl.searchParams.set("end_date", params.endDate);
      }

      // バックエンドAPIを直接呼び出し
      const response = await fetch(apiUrl.toString(), {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        // タイムアウトを設定（30秒）
        signal: AbortSignal.timeout(30000),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          `データベースAPIエラー: ${response.status} ${response.statusText} - ${
            errorData.detail?.message || errorData.message || "Unknown error"
          }`
        );
      }

      const backendData = await response.json();

      if (!backendData.success) {
        throw new Error(
          backendData.message || "データベースAPIからエラーレスポンス"
        );
      }

      // バックエンドのOHLCVデータをフロントエンド形式に変換
      const ohlcvData = backendData.data.ohlcv_data;

      // データが配列であることを確認
      if (!Array.isArray(ohlcvData)) {
        console.error("OHLCVデータの型:", typeof ohlcvData);
        console.error("OHLCVデータの内容:", ohlcvData);
        throw new Error(
          "バックエンドから返されたOHLCVデータが配列ではありません"
        );
      }

      const priceData: PriceData[] = ohlcvData.map((candle: number[]) => {
        const [timestamp, open, high, low, close, volume] = candle;

        return {
          timestamp: new Date(timestamp).toISOString(),
          open: Number(open.toFixed(2)),
          high: Number(high.toFixed(2)),
          low: Number(low.toFixed(2)),
          close: Number(close.toFixed(2)),
          volume: Number(volume.toFixed(2)),
        };
      });

      setData(priceData);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "不明なエラーが発生しました";
      setError(errorMessage);
      console.error("OHLCVデータ取得エラー:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  const refetch = useCallback(() => {
    fetchData({
      symbol,
      timeframe,
      limit,
    });
  }, [symbol, timeframe, limit, fetchData]);

  const setNewLimit = useCallback((newLimit: number) => {
    setLimit(newLimit);
  }, []);

  useEffect(() => {
    if (symbol && timeframe) {
      fetchData({
        symbol,
        timeframe,
        limit,
      });
    }
  }, [symbol, timeframe, limit, fetchData]);

  return {
    data,
    loading,
    error,
    refetch,
    setLimit: setNewLimit,
    limit,
  };
};
