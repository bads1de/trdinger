import { useState, useEffect, useCallback } from "react";
import { useApiCall } from "./useApiCall";

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
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance>(
    {}
  );
  const {
    execute: fetchStatus,
    loading: isLoading,
    error,
  } = useApiCall<ModelStatus>();
  const { execute: fetchImportance, loading: isLoadingImportance } =
    useApiCall<{ feature_importance: FeatureImportance }>();

  const fetchModelStatus = useCallback(async () => {
    await fetchStatus("/api/ml/status", {
      method: "GET",
      onSuccess: (data) => {
        setModelStatus(data);
      },
      onError: (errorMessage) => {
        console.error("モデル状態の取得に失敗しました:", errorMessage);
      },
    });
  }, [fetchStatus]);

  const fetchFeatureImportance = useCallback(async () => {
    await fetchImportance("/api/ml/feature-importance", {
      method: "GET",
      onSuccess: (data) => {
        setFeatureImportance(data.feature_importance || {});
      },
      onError: (errorMessage) => {
        console.error("特徴量重要度の取得に失敗:", errorMessage);
      },
    });
  }, [fetchImportance]);

  useEffect(() => {
    fetchModelStatus();
    fetchFeatureImportance();
  }, [fetchModelStatus, fetchFeatureImportance]);

  return {
    modelStatus,
    featureImportance,
    isLoading: isLoading || isLoadingImportance,
    error,
    fetchModelStatus,
    fetchFeatureImportance,
  };
};
