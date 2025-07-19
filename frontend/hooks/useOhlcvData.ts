import { PriceData, TimeFrame } from "@/types/market-data";
import { useDataFetching } from "./useDataFetching";

interface OhlcvParams {
  symbol: string;
  timeframe: TimeFrame;
  limit: number;
}

export const useOhlcvData = (
  symbol: string,
  timeframe: TimeFrame,
  initialLimit = 100
) => {
  const { data, loading, error, params, setParams, refetch } = useDataFetching<
    PriceData,
    OhlcvParams
  >({
    endpoint: "/api/data/candlesticks",
    initialParams: {
      symbol,
      timeframe,
      limit: initialLimit,
    },
    dataPath: "data.ohlcv",
    dependencies: [symbol, timeframe],
    errorMessage: "OHLCVデータの取得中にエラーが発生しました",
  });

  const setLimit = (newLimit: number) => {
    setParams({ limit: newLimit });
  };

  return {
    data,
    loading,
    error,
    refetch,
    setLimit,
    limit: params.limit,
  };
};
