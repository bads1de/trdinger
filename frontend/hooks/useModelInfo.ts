import { useEffect } from "react";
import { useDataFetching } from "./useDataFetching";
import { wrapInArray } from "@/utils/hookUtils";
import { getScoreBadgeVariant, getModelTypeBadgeVariant } from "@/utils/mlModelUtils";
import type { ModelStatusResponse } from "@/types/ml-model";

/**
 * モデル情報管理フック
 *
 * 機械学習モデルの状態情報と詳細情報を取得・管理します。
 * 自動更新機能やUI表示用のユーティリティ関数も提供します。
 *
 * @example
 * ```tsx
 * const {
 *   modelStatus,
 *   loading,
 *   error,
 *   loadModelStatus,
 *   getModelTypeBadgeVariant,
 *   getAccuracyBadgeVariant
 * } = useModelInfo(30); // 30秒ごとに自動更新
 *
 * // モデル状態を手動で更新
 * loadModelStatus();
 *
 * // モデルタイプに応じたバリアントを取得
 * const variant = getModelTypeBadgeVariant('lightgbm');
 *
 * // 精度に応じたバリアントを取得
 * const accuracyVariant = getAccuracyBadgeVariant(0.85);
 * ```
 *
 * @param {number} autoRefreshInterval - 自動更新間隔（秒）。指定しない場合は自動更新しない
 * @returns {{
 *   modelStatus: ModelStatusResponse | null,
 *   loading: boolean,
 *   error: string | null,
 *   loadModelStatus: () => Promise<void>,
 *   getModelTypeBadgeVariant: (modelType?: string) => string,
 *   getAccuracyBadgeVariant: (accuracy?: number) => string
 * }} モデル情報管理関連の状態と操作関数
 */
export const useModelInfo = (autoRefreshInterval?: number) => {
  const {
    data: modelStatusArray,
    loading,
    error,
    refetch: loadModelStatus,
  } = useDataFetching<ModelStatusResponse>({
    endpoint: "/api/ml/status",
    transform: wrapInArray,
    errorMessage: "モデル状態の取得に失敗しました",
  });

  const modelStatus = modelStatusArray.length > 0 ? modelStatusArray[0] : null;

  useEffect(() => {
    if (autoRefreshInterval && autoRefreshInterval > 0) {
      const interval = setInterval(loadModelStatus, autoRefreshInterval * 1000);
      return () => clearInterval(interval);
    }
  }, [autoRefreshInterval, loadModelStatus]);

  // 共通ユーティリティからバッジバリアント関数を提供

  return {
    /** モデル状態情報 */
    modelStatus,
    /** ローディング状態 */
    loading,
    /** エラーメッセージ */
    error,
    /** モデル状態を再取得する関数 */
    loadModelStatus,
    /** モデルタイプに応じたバッジバリアントを取得する関数 */
    getModelTypeBadgeVariant,
    /** 精度に応じたバッジバリアントを取得する関数 */
    getAccuracyBadgeVariant: getScoreBadgeVariant,
  };
};
