import { useState, useEffect, useCallback } from "react";
import { OpenInterestData } from "@/types/open-interest";
import { BACKEND_API_URL } from "@/constants";

interface OpenInterestParams {
  symbol: string;
  limit: number;
  startDate?: string;
  endDate?: string;
}

export const useOpenInterestData = (symbol: string, initialLimit = 100) => {
  const [data, setData] = useState<OpenInterestData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState(initialLimit);

  const fetchData = useCallback(async (params: OpenInterestParams) => {
    setLoading(true);
    setError(null);

    try {
      const apiUrl = new URL("/api/open-interest/", BACKEND_API_URL);
      apiUrl.searchParams.set("symbol", params.symbol);
      apiUrl.searchParams.set("limit", params.limit.toString());

      if (params.startDate) {
        apiUrl.searchParams.set("start_date", params.startDate);
      }
      if (params.endDate) {
        apiUrl.searchParams.set("end_date", params.endDate);
      }

      const response = await fetch(apiUrl.toString(), {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        console.error(
          `バックエンドAPIエラー: ${response.status} ${response.statusText}`
        );
        throw new Error(`バックエンドAPIエラー: ${response.status}`);
      }

      const backendData = await response.json();

      if (!backendData.success) {
        throw new Error(backendData.message || "データの取得に失敗しました");
      }

      setData(backendData.data.open_interest);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "不明なエラーが発生しました";
      setError(errorMessage);
      console.error("オープンインタレストデータの取得エラー:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  const refetch = useCallback(() => {
    fetchData({
      symbol,
      limit,
    });
  }, [symbol, limit, fetchData]);

  const setNewLimit = useCallback((newLimit: number) => {
    setLimit(newLimit);
  }, []);

  useEffect(() => {
    if (symbol) {
      fetchData({
        symbol,
        limit,
      });
    }
  }, [symbol, limit, fetchData]);

  return {
    data,
    loading,
    error,
    refetch,
    setLimit: setNewLimit,
    limit,
  };
};
