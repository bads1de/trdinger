import { useState, useEffect, useCallback } from "react";
import { PriceData, TimeFrame } from "@/types/market-data";
import { useApiCall } from "./useApiCall";

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
  const [limit, setLimit] = useState(initialLimit);
  const { execute, loading, error } = useApiCall();

  const fetchData = useCallback(
    async (params: OhlcvParams) => {
      // URLパラメータを構築
      const searchParams = new URLSearchParams({
        symbol: params.symbol,
        timeframe: params.timeframe,
        limit: params.limit.toString(),
      });

      if (params.startDate) {
        searchParams.set("start_date", params.startDate);
      }
      if (params.endDate) {
        searchParams.set("end_date", params.endDate);
      }

      const endpoint = `/api/market-data/ohlcv?${searchParams.toString()}`;

      await execute(endpoint, {
        method: "GET",
        onSuccess: (backendData) => {
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
        },
        onError: (errorMessage) => {
          console.error("OHLCVデータ取得エラー:", errorMessage);
        },
      });
    },
    [execute]
  );

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
