/**
 * バックテスト結果管理用カスタムフック
 *
 * バックテスト結果の取得、選択、削除などの機能を提供します。
 * ページネーション、フィルタリング、一括削除などの機能をサポートします。
 */

import { useState, useCallback } from "react";
import { useApiCall } from "@/hooks/useApiCall";
import { useParameterizedDataFetching } from "./useDataFetching";
import { BacktestResult } from "@/types/backtest";

/**
 * バックテスト結果取得パラメータの型
 */
interface BacktestResultsParams {
  /** 取得件数 */
  limit: number;
  /** オフセット（ページネーション用） */
  offset?: number;
  /** フィルタリングするシンボル */
  symbol?: string;
  /** フィルタリングする戦略名 */
  strategy_name?: string;
}

/**
 * バックテスト結果管理フック
 *
 * バックテスト結果の取得、選択、削除などの機能を提供します。
 * ページネーション、フィルタリング、一括削除などの機能をサポートします。
 *
 * @example
 * ```tsx
 * const {
 *   results,
 *   total,
 *   selectedResult,
 *   resultsLoading,
 *   loadResults,
 *   handleResultSelect,
 *   handleDeleteResult
 * } = useBacktestResults();
 *
 * // 結果を選択
 * handleResultSelect(result);
 *
 * // 結果を削除
 * handleDeleteResult(result);
 * ```
 *
 * @returns {{
 *   results: BacktestResult[],
 *   total: number,
 *   selectedResult: BacktestResult | null,
 *   resultsLoading: boolean,
 *   deleteLoading: boolean,
 *   deleteAllLoading: boolean,
 *   error: string | null,
 *   loadResults: () => Promise<void>,
 *   handleResultSelect: (result: BacktestResult) => void,
 *   handleDeleteResult: (result: BacktestResult) => Promise<void>,
 *   handleDeleteAllResults: () => Promise<void>,
 *   setSelectedResult: (result: BacktestResult | null) => void
 * }} バックテスト結果管理関連の状態と操作関数
 */
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
    /** バックテスト結果のリスト */
    results,
    /** 総結果件数 */
    total,
    /** 現在選択されているバックテスト結果 */
    selectedResult,
    /** 結果取得中のローディング状態 */
    resultsLoading: loading,
    /** 削除実行中のローディング状態 */
    deleteLoading,
    /** 全削除実行中のローディング状態 */
    deleteAllLoading,
    /** エラーメッセージ */
    error,
    /** 結果を再取得する関数 */
    loadResults,
    /** 結果を選択する関数 */
    handleResultSelect,
    /** 結果を削除する関数 */
    handleDeleteResult,
    /** 全結果を削除する関数 */
    handleDeleteAllResults,
    /** 選択中の結果を設定する関数 */
    setSelectedResult,
  };
};
