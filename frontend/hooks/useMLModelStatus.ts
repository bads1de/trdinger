import { useCallback } from "react";
import { useDataFetching } from "./useDataFetching";
import { wrapInArray } from "@/utils/hookUtils";
import type { ModelStatusResponse, FeatureImportance } from "@/types/ml-model";

/**
 * MLモデル状態管理フック
 *
 * 機械学習モデルの状態情報と特徴量重要度を取得・管理します。
 * モデルの読み込み状態、学習状態、性能指標、特徴量重要度などの情報を提供します。
 *
 * @example
 * ```tsx
 * const {
 *   modelStatus,
 *   featureImportance,
 *   isLoading,
 *   error,
 *   fetchModelStatus,
 *   fetchFeatureImportance
 * } = useMLModelStatus();
 *
 * // モデル状態を再取得
 * fetchModelStatus();
 *
 * // 特徴量重要度を再取得
 * fetchFeatureImportance();
 * ```
 *
 * @returns {{
 *   modelStatus: ModelStatus | null,
 *   featureImportance: FeatureImportance,
 *   isLoading: boolean,
 *   error: string | null,
 *   fetchModelStatus: () => Promise<void>,
 *   fetchFeatureImportance: () => Promise<void>
 * }} MLモデル状態管理関連の状態と操作関数
 */
export const useMLModelStatus = () => {
  // /api/ml/status を useDataFetching に置換（単一オブジェクトを配列化）
  const {
    data: statusArray,
    loading: statusLoading,
    error: statusError,
    refetch: refetchStatus,
  } = useDataFetching<ModelStatusResponse>({
    endpoint: "/api/ml/status",
    transform: wrapInArray,
    errorMessage: "モデル状態の取得に失敗しました",
  });

  const modelStatus = statusArray.length > 0 ? statusArray[0] : null;

  // /api/ml/feature-importance も useDataFetching で取得
  const {
    data: importanceArray,
    loading: importanceLoading,
    error: importanceError,
    refetch: refetchImportance,
  } = useDataFetching<{ feature_importance: FeatureImportance }>({
    endpoint: "/api/ml/feature-importance",
    transform: wrapInArray,
    errorMessage: "特徴量重要度の取得に失敗しました",
  });

  const featureImportance =
    importanceArray.length > 0
      ? importanceArray[0]?.feature_importance || {}
      : {};

  // 既存APIと同じ外部インターフェースを保つため、refetch関数を公開
  const fetchModelStatus = useCallback(() => {
    return refetchStatus();
  }, [refetchStatus]);

  const fetchFeatureImportance = useCallback(() => {
    return refetchImportance();
  }, [refetchImportance]);

  return {
    /** モデル状態情報 */
    modelStatus,
    /** 特徴量重要度情報 */
    featureImportance,
    /** データ取得中のローディング状態 */
    isLoading: statusLoading || importanceLoading,
    /** エラーメッセージ */
    error: statusError || importanceError || null,
    /** モデル状態を再取得する関数 */
    fetchModelStatus,
    /** 特徴量重要度を再取得する関数 */
    fetchFeatureImportance,
  };
};
