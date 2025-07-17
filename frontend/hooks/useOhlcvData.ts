import { useState, useEffect, useCallback } from "react";
import { PriceData, TimeFrame, OHLCVResponse } from "@/types/market-data";
import { useApiCall } from "./useApiCall";

export const useOhlcvData = (
  symbol: string,
  timeframe: TimeFrame,
  initialLimit = 100
) => {
  const [data, setData] = useState<PriceData[]>([]);
  const [limit, setLimit] = useState<number>(initialLimit);
  const { execute, loading, error } = useApiCall<OHLCVResponse>();

  const fetchOhlcvData = useCallback(async () => {
    if (!symbol || !timeframe) return;

    const params = new URLSearchParams({
      symbol,
      timeframe,
      limit: limit.toString(),
    });

    await execute(`/api/data/candlesticks?${params}`, {
      method: "GET",
      onSuccess: (response) => {
        if (response.success) {
          setData(response.data.ohlcv);
        }
      },
      onError: (errorMessage) => {
        console.error("OHLCVデータの取得中にエラーが発生しました:", errorMessage);
      }
    });
  }, [symbol, timeframe, limit, execute]);

  useEffect(() => {
    fetchOhlcvData();
  }, [fetchOhlcvData]);

  return { data, loading, error, refetch: fetchOhlcvData, setLimit };
};
