import { useCallback } from "react";
import { FundingRateData } from "@/types/funding-rate";
import { useParameterizedDataFetching } from "./useDataFetching";

interface FundingRateParams {
  symbol: string;
  limit: number;
  start_date?: string;
  end_date?: string;
}

export const useFundingRateData = (symbol: string, initialLimit = 100) => {
  const {
    data,
    loading,
    error,
    params,
    setParams,
    refetch,
  } = useParameterizedDataFetching<FundingRateData, FundingRateParams>(
    "/api/funding-rates/",
    { symbol, limit: initialLimit },
    {
      transform: (response: any) => {
        const fundingRateData = response.data?.funding_rates || [];
        return fundingRateData.map((rate: any) => ({
          symbol: rate.symbol,
          funding_rate: Number(rate.funding_rate),
          funding_timestamp: rate.funding_timestamp,
          timestamp: rate.timestamp,
          next_funding_timestamp: rate.next_funding_timestamp || null,
          mark_price: rate.mark_price ? Number(rate.mark_price) : null,
          index_price: rate.index_price ? Number(rate.index_price) : null,
        }));
      },
      dependencies: [symbol],
      errorMessage: "資金調達率データの取得に失敗しました",
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
