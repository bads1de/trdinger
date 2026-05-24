import { useCallback } from "react";
import { useDataFetching } from "./useDataFetching";
import { wrapInArray } from "@/utils/hookUtils";
import { getScoreBadgeVariant, getStatusBadgeVariant } from "@/utils/mlModelUtils";
import type { ModelStatusResponse } from "@/types/ml-model";

/**
 * モデルパフォーマンス管理フック
 *
 * 機械学習モデルのパフォーマンス情報を取得・管理します。
 * モデルの状態、性能指標、UI表示用のユーティリティ関数を提供します。
 *
 * @example
 * ```tsx
 * const {
 *   modelStatus,
 *   loading,
 *   error,
 *   loadModelStatus,
 *   getScoreBadgeVariant,
 *   getStatusBadgeVariant
 * } = useModelPerformance();
 *
 * // モデルパフォーマンス情報を取得
 * loadModelStatus();
 *
 * // スコアに応じたバリアントを取得
 * const variant = getScoreBadgeVariant(0.85);
 *
 * // 状態に応じたバリアントを取得
 * const statusVariant = getStatusBadgeVariant();
 * ```
 *
 * @returns {{
 *   modelStatus: ModelStatusResponse | null,
 *   loading: boolean,
 *   error: string | null,
 *   loadModelStatus: () => Promise<void>,
 *   getScoreBadgeVariant: (score?: number) => string,
 *   getStatusBadgeVariant: () => string
 * }} モデルパフォーマンス管理関連の状態と操作関数
 */
export const useModelPerformance = () => {
  const {
    data: modelStatusArray,
    loading,
    error,
    refetch: loadModelStatus,
  } = useDataFetching<ModelStatusResponse>({
    endpoint: "/api/ml/status",
    transform: wrapInArray,
    errorMessage: "モデルパフォーマンスの取得に失敗しました",
  });

  const modelStatus = modelStatusArray.length > 0 ? modelStatusArray[0] : null;

  /**
   * 状態に応じたバッジバリアントを取得
   *
   * @returns {string} バッジバリアント（"default"、"success"、"outline"）
   */
  const getStatusBadgeVariantLocal = useCallback(() => {
    return getStatusBadgeVariant(modelStatus?.is_training, modelStatus?.is_model_loaded);
  }, [modelStatus?.is_training, modelStatus?.is_model_loaded]);

  return {
    /** モデル状態情報 */
    modelStatus,
    /** ローディング状態 */
    loading,
    /** エラーメッセージ */
    error,
    /** モデル状態を再取得する関数 */
    loadModelStatus,
    /** スコアに応じたバッジバリアントを取得する関数 */
    getScoreBadgeVariant,
    /** 状態に応じたバッジバリアントを取得する関数 */
    getStatusBadgeVariant: getStatusBadgeVariantLocal,
  };
};
