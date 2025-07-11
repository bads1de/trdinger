import { useState, useEffect } from "react";
import { useApiCall } from "@/hooks/useApiCall";
import { BacktestResult } from "@/types/backtest";

export const useBacktestResults = () => {
  const [results, setResults] = useState<BacktestResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<BacktestResult | null>(
    null
  );

  const { execute: fetchResults, loading: resultsLoading } = useApiCall<{
    results: BacktestResult[];
    total: number;
  }>();
  const { execute: deleteResult, loading: deleteLoading } = useApiCall();
  const { execute: deleteAllResults, loading: deleteAllLoading } = useApiCall();

  // 結果一覧を取得
  const loadResults = async () => {
    const response = await fetchResults("/api/backtest/results?limit=20");
    if (response) {
      setResults(response.results);
    }
  };

  // ページ読み込み時に結果一覧を取得
  useEffect(() => {
    loadResults();
  }, []);

  // 結果選択
  const handleResultSelect = (result: BacktestResult) => {
    setSelectedResult(result);
  };

  // 結果削除
  const handleDeleteResult = async (result: BacktestResult) => {
    const response = await deleteResult(`/api/backtest/results/${result.id}`, {
      method: "DELETE",
      confirmMessage: `バックテスト結果「${result.strategy_name}」を削除しますか？\nこの操作は取り消せません。`,
      onSuccess: () => {
        // 削除成功時は一覧を更新
        loadResults();
        // 選択中の結果が削除された場合はクリア
        if (selectedResult?.id === result.id) {
          setSelectedResult(null);
        }
      },
      onError: (error) => {
        console.error("Delete failed:", error);
      },
    });
  };

  // すべての結果削除
  const handleDeleteAllResults = async () => {
    const response = await deleteAllResults("/api/backtest/results/all", {
      method: "DELETE",
      confirmMessage: `すべてのバックテスト結果を削除しますか？\n現在${results.length}件の結果があります。\nこの操作は取り消せません。`,
      onSuccess: () => {
        // 削除成功時は一覧を更新
        loadResults();
        // 選択中の結果をクリア
        setSelectedResult(null);
      },
      onError: (error) => {
        console.error("Delete all failed:", error);
      },
    });
  };

  return {
    results,
    selectedResult,
    resultsLoading,
    deleteLoading,
    deleteAllLoading,
    loadResults,
    handleResultSelect,
    handleDeleteResult,
    handleDeleteAllResults,
    setSelectedResult,
  };
};
