import { useState, useCallback } from "react";
import { useApiCall } from "@/hooks/useApiCall";
import { useParameterizedDataFetching } from "./useDataFetching";
import { BacktestResult } from "@/types/backtest";

interface BacktestResultsParams {
  limit: number;
  offset?: number;
  symbol?: string;
  strategy_name?: string;
}

export const useBacktestResults = () => {
  const [selectedResult, setSelectedResult] = useState<BacktestResult | null>(
    null
  );
  const [total, setTotal] = useState(0);

  const {
    data: results,
    loading,
    error,
    refetch: loadResults,
  } = useParameterizedDataFetching<BacktestResult, BacktestResultsParams>(
    "/api/backtest/results/",
    { limit: 20, offset: 0 },
    {
      transform: (response: any) => {
        setTotal(response.total || 0);
        return response.results || [];
      },
      errorMessage: "バックテスト結果の取得に失敗しました",
    }
  );

  const { execute: deleteResultApi, loading: deleteLoading } = useApiCall();
  const { execute: deleteAllResultsApi, loading: deleteAllLoading } =
    useApiCall();

  const handleResultSelect = (result: BacktestResult) => {
    setSelectedResult(result);
  };

  const handleDeleteResult = async (result: BacktestResult) => {
    await deleteResultApi(`/api/backtest/results/${result.id}`, {
      method: "DELETE",
      confirmMessage: `バックテスト結果「${result.strategy_name}」を削除しますか？\nこの操作は取り消せません。`,
      onSuccess: () => {
        loadResults();
        if (selectedResult?.id === result.id) {
          setSelectedResult(null);
        }
      },
      onError: (err) => {
        console.error("Delete failed:", err);
      },
    });
  };

  const handleDeleteAllResults = async () => {
    await deleteAllResultsApi("/api/backtest/results-all", {
      method: "DELETE",
      confirmMessage: `すべてのバックテスト結果を削除しますか？\n現在${results.length}件の結果があります。\nこの操作は取り消せません。`,
      onSuccess: () => {
        loadResults();
        setSelectedResult(null);
      },
      onError: (err) => {
        console.error("Delete all failed:", err);
      },
    });
  };

  return {
    results,
    total,
    selectedResult,
    resultsLoading: loading,
    deleteLoading,
    deleteAllLoading,
    error,
    loadResults,
    handleResultSelect,
    handleDeleteResult,
    handleDeleteAllResults,
    setSelectedResult,
  };
};
