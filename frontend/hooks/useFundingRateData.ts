import { useState, useEffect, useCallback } from "react";
import { FundingRateData, FundingRateResponse } from "@/types/funding-rate";
import { useApiCall } from "./useApiCall";

export const useFundingRateData = (symbol: string, initialLimit = 100) => {
  const [data, setData] = useState<FundingRateData[]>([]);
  const [limit, setLimit] = useState<number>(initialLimit);
  const { execute, loading, error } = useApiCall<FundingRateResponse>();

  const fetchFundingRateData = useCallback(async () => {
    if (!symbol) return;

    const params = new URLSearchParams({
      symbol,
      limit: limit.toString(),
    });

    await execute(`/api/data/funding-rates?${params}`, {
      method: "GET",
      onSuccess: (response) => {
        if (response.success) {
          setData(response.data.funding_rates);
        }
      },
      onError: (errorMessage) => {
        console.error("資金調達データの取得中にエラーが発生しました:", errorMessage);
      }
    });
  }, [symbol, limit, execute]);

  useEffect(() => {
    fetchFundingRateData();
  }, [fetchFundingRateData]);

  return { data, loading, error, refetch: fetchFundingRateData, setLimit };
};
