import { OpenInterestData } from "@/types/open-interest";
import { useDataFetching } from "./useDataFetching";

interface OpenInterestParams {
  symbol: string;
  limit: number;
}

export const useOpenInterestData = (symbol: string, initialLimit = 100) => {
  const { data, loading, error, params, setParams, refetch } = useDataFetching<
    OpenInterestData,
    OpenInterestParams
  >({
    endpoint: "/api/data/open-interest",
    initialParams: {
      symbol,
      limit: initialLimit,
    },
    dataPath: "data.open_interest",
    dependencies: [symbol],
    errorMessage: "オープンインタレストのデータ取得中にエラーが発生しました",
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
