import { useCallback } from "react";
import { useApiCall } from "./useApiCall";
import { useDataFetching } from "./useDataFetching";

export interface MLModel {
  id: string;
  name: string;
  path: string;
  size_mb: number;
  modified_at: string;
  directory: string;
  accuracy?: number;
  feature_count?: number;
  model_type?: string;
  is_active?: boolean;
}

export const useMLModels = (limit?: number) => {
  // 基本的なデータ取得は共通フックを使用
  const {
    data: models,
    loading: isLoading,
    error,
    refetch: fetchModels,
  } = useDataFetching<MLModel>({
    endpoint: "/api/ml/models",
    dataPath: "models",
    transform: (response) => {
      let modelList = response.models || [];
      if (limit) {
        modelList = modelList.slice(0, limit);
      }
      return modelList;
    },
    errorMessage: "MLモデルの取得中にエラーが発生しました",
  });

  // 削除・バックアップ操作用のAPI呼び出し
  const { execute: deleteModelApi, loading: isDeleting } = useApiCall();
  const { execute: backupModelApi, loading: isBackingUp } = useApiCall();

  const deleteModel = useCallback(
    async (modelId: string) => {
      await deleteModelApi(`/api/ml/models/${modelId}`, {
        method: "DELETE",
        confirmMessage: "このモデルを削除しますか？この操作は取り消せません。",
        onSuccess: () => {
          fetchModels(); // リストを更新
        },
      });
    },
    [deleteModelApi, fetchModels]
  );

  const backupModel = useCallback(
    async (modelId: string) => {
      await backupModelApi(`/api/ml/models/${modelId}/backup`, {
        method: "POST",
        onSuccess: () => {
          alert("モデルのバックアップが完了しました");
        },
      });
    },
    [backupModelApi]
  );

  return {
    models,
    isLoading,
    error,
    isDeleting,
    isBackingUp,
    fetchModels,
    deleteModel,
    backupModel,
  };
};
