import { useState, useCallback } from "react";
import { useApiCall } from "./useApiCall";

export interface ModelInfo {
  name: string;
  path: string;
  size_mb: number;
  modified_at: string;
  model_type: string;
  trainer_type: string;
  feature_count: number;
  has_feature_importance: boolean;
  feature_importance_count: number;
}

export interface CurrentModelInfo {
  loaded: boolean;
  trainer_type?: string;
  is_trained?: boolean;
  model_type?: string;
  has_feature_importance?: boolean;
  feature_importance_count?: number;
  message?: string;
  error?: string;
}

export interface ModelsListResponse {
  models: ModelInfo[];
  total_count: number;
  error?: string;
}

export interface LoadModelResponse {
  success: boolean;
  message?: string;
  error?: string;
  current_model?: CurrentModelInfo;
}

export const useModelManagement = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [currentModel, setCurrentModel] = useState<CurrentModelInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { execute: fetchData } = useApiCall();

  // モデル一覧を取得
  const fetchModels = useCallback(async () => {
    setLoading(true);
    setError(null);

    await fetchData("/api/ml/models", {
      method: "GET",
      onSuccess: (data) => {
        setModels(data.models || []);
        if (data.error) {
          setError(data.error);
        }
      },
      onError: (errorMessage: string) => {
        setError(errorMessage);
        setModels([]);
      },
      onFinally: () => {
        setLoading(false);
      },
    });
  }, [fetchData]);

  // 現在のモデル情報を取得
  const fetchCurrentModel = useCallback(async () => {
    await fetchData("/api/ml/models/current", {
      method: "GET",
      onSuccess: (data) => {
        setCurrentModel(data);
      },
      onError: (errorMessage: string) => {
        setCurrentModel({ loaded: false, error: errorMessage });
      },
    });
  }, [fetchData]);

  // 指定されたモデルを読み込み
  const loadModel = useCallback(
    async (modelName: string) => {
      setLoading(true);
      setError(null);

      await fetchData(`/api/ml/models/${encodeURIComponent(modelName)}/load`, {
        method: "POST",
        onSuccess: (data) => {
          if (data.success) {
            setCurrentModel(data.current_model || null);
            // モデル一覧を再取得（状態が変わった可能性があるため）
            fetchModels();
          } else {
            setError(data.error || "モデル読み込みに失敗しました");
          }
        },
        onError: (errorMessage: string) => {
          setError(errorMessage);
        },
        onFinally: () => {
          setLoading(false);
        },
      });
    },
    [fetchData, fetchModels]
  );

  // モデル情報をリフレッシュ
  const refreshModels = useCallback(async () => {
    await Promise.all([fetchModels(), fetchCurrentModel()]);
  }, [fetchModels, fetchCurrentModel]);

  return {
    models,
    currentModel,
    loading,
    error,
    fetchModels,
    fetchCurrentModel,
    loadModel,
    refreshModels,
  };
};
