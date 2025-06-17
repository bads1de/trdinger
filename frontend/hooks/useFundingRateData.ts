import { useState, useEffect, useCallback } from "react";
import { FundingRateData, FundingRateResponse } from "@/types/strategy";

export const useFundingRateData = (symbol: string, initialLimit = 100) => {
  const [data, setData] = useState<FundingRateData[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [limit, setLimit] = useState<number>(initialLimit);

  const fetchFundingRateData = useCallback(async () => {
    if (!symbol) return;

    setLoading(true);
    setError("");

    try {
      const params = new URLSearchParams({
        symbol,
        limit: limit.toString(),
      });
      const response = await fetch(`/api/data/funding-rates?${params}`);
      const result: FundingRateResponse = await response.json();

      if (result.success) {
        setData(result.data.funding_rates);
      } else {
        setError(result.message || "Failed to fetch funding rate data");
      }
    } catch (err) {
      setError("An error occurred while fetching funding rate data");
      console.error("Error fetching funding rate data:", err);
    } finally {
      setLoading(false);
    }
  }, [symbol, limit]);

  useEffect(() => {
    fetchFundingRateData();
  }, [fetchFundingRateData]);

  return { data, loading, error, refetch: fetchFundingRateData, setLimit };
};
