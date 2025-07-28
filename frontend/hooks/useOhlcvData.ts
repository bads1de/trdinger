import { useCallback } from "react";
import { PriceData, TimeFrame } from "@/types/market-data";
import { useParameterizedDataFetching } from "./useDataFetching";

interface OhlcvParams {
  symbol: string;
  timeframe: TimeFrame;
  limit: number;
  start_date?: string;
  end_date?: string;
}

export const useOhlcvData = (
  symbol: string,
  timeframe: TimeFrame,
  initialLimit = 100
) => {
  const {
    data,
    loading,
    error,
    params,
    setParams,
    refetch,
  } = useParameterizedDataFetching<PriceData, OhlcvParams>(
    "/api/market-data/ohlcv",
    { symbol, timeframe, limit: initialLimit },
    {
      transform: (response: any) => {
        const ohlcvData = response.data?.ohlcv_data || [];
        if (!Array.isArray(ohlcvData)) {
          console.error("OHLCV data is not an array:", ohlcvData);
          return [];
        }
        return ohlcvData.map((candle: number[]) => {
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
      },
      dependencies: [symbol, timeframe],
      errorMessage: "OHLCVデータの取得に失敗しました",
    }
  );

  const setLimit = useCallback(
    (newLimit: number) => {
      setParams({ limit: newLimit });
    },
    [setParams]
  );

  return {
    data,
    loading,
    error,
    refetch,
    setLimit,
    limit: params.limit,
  };
};
