import { useCallback } from "react";
import { useDataFetching } from "./useDataFetching";
import { useDeleteRequest } from "./useDeleteRequest";
import { usePostRequest } from "./usePostRequest";

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
    transform: transformModels, // メモ化した関数を渡す
    errorMessage: "MLモデルの取得中にエラーが発生しました",
  });

  // 削除・バックアップ操作用のAPI呼び出し
  const { sendDeleteRequest, isLoading: isDeleting } = useDeleteRequest();

  const deleteModel = useCallback(
    async (modelId: string) => {
      if (
        window.confirm("このモデルを削除しますか？この操作は取り消せません。")
      ) {
        const { success } = await sendDeleteRequest(
          `/api/ml/models/${modelId}`
        );
        if (success) {
          fetchModels(); // リストを更新
        }
      }
    },
    [sendDeleteRequest, fetchModels]
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
