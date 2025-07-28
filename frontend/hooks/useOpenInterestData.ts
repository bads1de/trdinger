import { useCallback } from "react";
import { OpenInterestData } from "@/types/open-interest";
import { useParameterizedDataFetching } from "./useDataFetching";

interface OpenInterestParams {
  symbol: string;
  limit: number;
  start_date?: string;
  end_date?: string;
}

export const useOpenInterestData = (symbol: string, initialLimit = 100) => {
  const {
    data,
    loading,
    error,
    params,
    setParams,
    refetch,
  } = useParameterizedDataFetching<OpenInterestData, OpenInterestParams>(
    "/api/open-interest/",
    { symbol, limit: initialLimit },
    {
      dataPath: "data.open_interest",
      dependencies: [symbol],
      errorMessage: "オープンインタレストデータの取得に失敗しました",
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
