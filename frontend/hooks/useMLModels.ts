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

  return {
    models,
    isLoading,
    error,
    isDeleting,
    fetchModels,
    deleteModel,
  };
};
