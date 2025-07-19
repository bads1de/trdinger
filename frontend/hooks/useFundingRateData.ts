import { useState, useEffect, useCallback } from "react";
import { FundingRateData } from "@/types/funding-rate";
import { BACKEND_API_URL } from "@/constants";

interface FundingRateParams {
  symbol: string;
  limit: number;
  startDate?: string;
  endDate?: string;
}

export const useFundingRateData = (symbol: string, initialLimit = 100) => {
  const [data, setData] = useState<FundingRateData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState(initialLimit);

  const fetchData = useCallback(async (params: FundingRateParams) => {
    setLoading(true);
    setError(null);

    try {
      const apiUrl = new URL("/api/funding-rates/", BACKEND_API_URL);
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
        const errorText = await response.text();
        console.error(
          `バックエンドAPIエラー: ${response.status} - ${errorText}`
        );
        throw new Error(
          `バックエンドAPIエラー: ${response.status} - ${response.statusText}`
        );
      }

      const backendData = await response.json();

      if (!backendData.success) {
        throw new Error(backendData.message || "データの取得に失敗しました");
      }

      const fundingRateData = backendData.data.funding_rates;
      const fundingRates: FundingRateData[] = fundingRateData.map(
        (rate: any) => ({
          symbol: rate.symbol,
          funding_rate: Number(rate.funding_rate),
          funding_timestamp: rate.funding_timestamp,
          timestamp: rate.timestamp,
          next_funding_timestamp: rate.next_funding_timestamp || null,
          mark_price: rate.mark_price ? Number(rate.mark_price) : null,
          index_price: rate.index_price ? Number(rate.index_price) : null,
        })
      );

      setData(fundingRates);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "不明なエラーが発生しました";
      setError(errorMessage);
      console.error("資金調達率データの取得エラー:", err);
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
