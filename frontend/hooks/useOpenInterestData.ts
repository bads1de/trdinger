import { useState, useEffect, useCallback } from "react";
import { OpenInterestData, OpenInterestResponse } from "@/types/strategy";

export const useOpenInterestData = (symbol: string, initialLimit = 100) => {
  const [data, setData] = useState<OpenInterestData[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [limit, setLimit] = useState<number>(initialLimit);

  const fetchOpenInterestData = useCallback(async () => {
    if (!symbol) return;

    setLoading(true);
    setError("");

    try {
      const params = new URLSearchParams({
        symbol,
        limit: limit.toString(),
      });
      const response = await fetch(`/api/data/open-interest?${params}`);
      const result: OpenInterestResponse = await response.json();

      if (result.success) {
        setData(result.data.open_interest);
      } else {
        setError(result.message || "Failed to fetch open interest data");
      }
    } catch (err) {
      setError("An error occurred while fetching open interest data");
      console.error("Error fetching open interest data:", err);
    } finally {
      setLoading(false);
    }
  }, [symbol, limit]);

  useEffect(() => {
    fetchOpenInterestData();
  }, [fetchOpenInterestData]);

  return { data, loading, error, refetch: fetchOpenInterestData, setLimit };
};
