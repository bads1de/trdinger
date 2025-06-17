import { useState, useEffect, useCallback } from "react";
import { PriceData, TimeFrame, OHLCVResponse } from "@/types/strategy";

export const useOhlcvData = (
  symbol: string,
  timeframe: TimeFrame,
  initialLimit = 100
) => {
  const [data, setData] = useState<PriceData[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [limit, setLimit] = useState<number>(initialLimit);

  const fetchOhlcvData = useCallback(async () => {
    if (!symbol || !timeframe) return;

    setLoading(true);
    setError("");

    try {
      const params = new URLSearchParams({
        symbol,
        timeframe,
        limit: limit.toString(),
      });
      const response = await fetch(`/api/data/candlesticks?${params}`);
      const result: OHLCVResponse = await response.json();

      if (result.success) {
        setData(result.data.ohlcv);
      } else {
        setError(result.message || "Failed to fetch OHLCV data");
      }
    } catch (err) {
      setError("An error occurred while fetching OHLCV data");
      console.error("Error fetching OHLCV data:", err);
    } finally {
      setLoading(false);
    }
  }, [symbol, timeframe, limit]);

  useEffect(() => {
    fetchOhlcvData();
  }, [fetchOhlcvData]);

  return { data, loading, error, refetch: fetchOhlcvData, setLimit };
};
