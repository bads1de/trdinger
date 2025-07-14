import { useState, useEffect, useCallback } from "react";
import { OpenInterestData, OpenInterestResponse } from "@/types/strategy";
import { useApiCall } from "./useApiCall";

export const useOpenInterestData = (symbol: string, initialLimit = 100) => {
  const [data, setData] = useState<OpenInterestData[]>([]);
  const [limit, setLimit] = useState<number>(initialLimit);
  const { execute, loading, error } = useApiCall<OpenInterestResponse>();

  const fetchOpenInterestData = useCallback(async () => {
    if (!symbol) return;

    const params = new URLSearchParams({
      symbol,
      limit: limit.toString(),
    });

    await execute(`/api/data/open-interest?${params}`, {
      method: "GET",
      onSuccess: (response) => {
        if (response.success) {
          setData(response.data.open_interest);
        }
      },
      onError: (errorMessage) => {
        console.error("オープンインタレストのデータ取得中にエラーが発生しました:", errorMessage);
      }
    });
  }, [symbol, limit, execute]);

  useEffect(() => {
    fetchOpenInterestData();
  }, [fetchOpenInterestData]);

  return { data, loading, error, refetch: fetchOpenInterestData, setLimit };
};
