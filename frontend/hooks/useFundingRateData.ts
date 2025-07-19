import { FundingRateData } from "@/types/funding-rate";
import { useDataFetching } from "./useDataFetching";

interface FundingRateParams {
  symbol: string;
  limit: number;
}

export const useFundingRateData = (symbol: string, initialLimit = 100) => {
  const { data, loading, error, params, setParams, refetch } = useDataFetching<
    FundingRateData,
    FundingRateParams
  >({
    endpoint: "/api/data/funding-rates",
    initialParams: {
      symbol,
      limit: initialLimit,
    },
    dataPath: "data.funding_rates",
    dependencies: [symbol],
    errorMessage: "資金調達データの取得中にエラーが発生しました",
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
