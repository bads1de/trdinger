import { useCallback } from "react";
import { useDataFetching } from "./useDataFetching";
import { useApiCall } from "./useApiCall";

export interface MLModel {
  id: string;
  name: string;
  path: string;
  size_mb: number;
  modified_at: string;
  directory: string;

  // 基本性能指標
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;

  // AUC指標
  auc_score?: number;
  auc_roc?: number;
  auc_pr?: number;

  // 高度な指標
  balanced_accuracy?: number;
  matthews_corrcoef?: number;
  cohen_kappa?: number;

  // 専門指標
  specificity?: number;
  sensitivity?: number;
  npv?: number;
  ppv?: number;

  // 確率指標
  log_loss?: number;
  brier_score?: number;

  // モデル情報
  feature_count?: number;
  model_type?: string;
  training_samples?: number;
  test_samples?: number;
  num_classes?: number;
  best_iteration?: number;

  // 学習パラメータ
  train_test_split?: number;
  random_state?: number;

  // 詳細データ
  feature_importance?: Record<string, number>;
  classification_report?: Record<string, any>;

  // 状態
  is_active?: boolean;
}

/**
 * MLモデル管理フック
 *
 * 機械学習モデルの取得、削除、管理機能を提供します。
 * モデル一覧の取得、個別モデルの削除、ローディング状態の管理などをサポートします。
 *
 * @example
 * ```tsx
 * const {
 *   models,
 *   isLoading,
 *   error,
 *   isDeleting,
 *   fetchModels,
 *   deleteModel
 * } = useMLModels(50);
 *
 * // モデル一覧を再取得
 * fetchModels();
 *
 * // モデルを削除
 * deleteModel('model-id');
 *
 * // すべてのモデルを削除
 * deleteAllModels();
 * ```
 *
 * @param {number} [limit] - 取得するモデルの最大件数、指定しない場合は全件取得
 * @returns {{
 *   models: MLModel[],
 *   isLoading: boolean,
 *   error: string | null,
 *   isDeleting: boolean,
 *   fetchModels: () => Promise<void>,
 *   deleteModel: (modelId: string) => Promise<void>,
 *   deleteAllModels: () => Promise<void>
 * }} MLモデル管理関連の状態と操作関数
 */
export const useMLModels = (limit?: number) => {
  // データを変換する関数をメモ化
  const transformModels = useCallback(
    (response: any) => {
      let modelList = response.models || [];

      if (limit) {
        modelList = modelList.slice(0, limit);
      }

      return modelList;
    },
    [limit] // limit が変更されたときのみ関数を再生成
  );

  // 基本的なデータ取得は共通フックを使用
  const {
    data: models,
    loading: isLoading,
    error,
    refetch: fetchModels,
  } = useDataFetching<MLModel>({
    endpoint: "/api/ml/models",
    dataPath: "models",
    transform: transformModels,
    errorMessage: "MLモデルの取得中にエラーが発生しました",
  });

  // 削除・バックアップ操作用のAPI呼び出し
  const { execute, loading: isDeleting } = useApiCall();

  const deleteModel = useCallback(
    async (modelId: string) => {
      await execute(`/api/ml/models/${modelId}`, {
        method: "DELETE",
        confirmMessage: "このモデルを削除しますか？この操作は取り消せません。",
        onSuccess: () => {
          fetchModels();
        },
      });
    },
    [execute, fetchModels]
  );

  const deleteAllModels = useCallback(
    async () => {
      await execute(`/api/ml/models/all`, {
        method: "DELETE",
        confirmMessage: "すべてのモデルを削除しますか？この操作は取り消せません。",
        onSuccess: () => {
          fetchModels();
        },
      });
    },
    [execute, fetchModels]
  );

  return {
    /** MLモデルの配列 */
    models,
    /** モデル取得中のローディング状態 */
    isLoading,
    /** エラーメッセージ */
    error,
    /** モデル削除中のローディング状態 */
    isDeleting,
    /** モデル一覧を再取得する関数 */
    fetchModels,
    /** モデルを削除する関数 */
    deleteModel,
    /** すべてのモデルを削除する関数 */
    deleteAllModels,
  };
};
