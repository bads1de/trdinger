import { useEffect, useCallback } from "react";
import { useDataFetching } from "./useDataFetching";

interface ModelStatus {
  is_model_loaded: boolean;
  is_trained: boolean;
  last_predictions: {
    up: number;
    down: number;
    range: number;
  };
  feature_count: number;
  model_info?: {
    accuracy: number;
    model_type: string;
    last_updated: string;
    training_samples: number;
  };
}

interface FeatureImportance {
  [key: string]: number;
}

export const useMLModelStatus = () => {
  // /api/ml/status を useDataFetching に置換（単一オブジェクトを配列化）
  const {
    data: statusArray,
    loading: statusLoading,
    error: statusError,
    refetch: refetchStatus,
  } = useDataFetching<ModelStatus>({
    endpoint: "/api/ml/status",
    transform: (response) => [response],
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
    transform: (response) => [response],
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
    modelStatus,
    featureImportance,
    isLoading: statusLoading || importanceLoading,
    error: statusError || importanceError || null,
    fetchModelStatus,
    fetchFeatureImportance,
  };
};
