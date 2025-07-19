import { useState, useCallback, useEffect } from "react";
import { useApiCall } from "@/hooks/useApiCall";
import { BacktestResult } from "@/types/backtest";
import { BACKEND_API_URL } from "@/constants";

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
  const [results, setResults] = useState<BacktestResult[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [params, setParams] = useState<BacktestResultsParams>({ limit: 20 });

  const { execute: deleteResultApi, loading: deleteLoading } = useApiCall();
  const { execute: deleteAllResultsApi, loading: deleteAllLoading } =
    useApiCall();

  const loadResults = useCallback(
    async (newParams?: Partial<BacktestResultsParams>) => {
      const currentParams = { ...params, ...newParams };
      setParams(currentParams);
      setLoading(true);
      setError(null);
      try {
        const backendUrl = new URL(`${BACKEND_API_URL}/api/backtest/results/`);
        backendUrl.searchParams.set("limit", String(currentParams.limit));
        backendUrl.searchParams.set(
          "offset",
          String(currentParams.offset || 0)
        );
        if (currentParams.symbol) {
          backendUrl.searchParams.set("symbol", currentParams.symbol);
        }
        if (currentParams.strategy_name) {
          backendUrl.searchParams.set(
            "strategy_name",
            currentParams.strategy_name
          );
        }

        const response = await fetch(backendUrl.toString());
        if (!response.ok) {
          throw new Error(`Backend API error: ${response.status}`);
        }
        const data = await response.json();
        setResults(data.results || []);
        setTotal(data.total || 0);
      } catch (e) {
        const errorMessage = e instanceof Error ? e.message : "不明なエラー";
        setError(errorMessage);
      } finally {
        setLoading(false);
      }
    },
    [params]
  );

  useEffect(() => {
    loadResults();
  }, []); // Initial load

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
