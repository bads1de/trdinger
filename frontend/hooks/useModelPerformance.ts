import { useState, useEffect, useCallback } from "react";
import { useApiCall } from "@/hooks/useApiCall";
import { formatTrainingTime } from "@/utils/formatters";

interface PerformanceMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  auc_score?: number;
  loss?: number;
  val_accuracy?: number;
  val_loss?: number;
  training_time?: number;
  last_evaluation?: string;
}

interface ModelStatusResponse {
  is_model_loaded: boolean;
  is_trained: boolean;
  last_predictions?: {
    up: number;
    down: number;
    range: number;
  };
  feature_count: number;
  model_info?: {
    accuracy?: number;
    model_type?: string;
    training_samples?: number;
    last_updated?: string;
  };
  performance_metrics?: PerformanceMetrics;
  is_training?: boolean;
  training_progress?: number;
  status?: string;
}

export const useModelPerformance = () => {
  const [modelStatus, setModelStatus] = useState<ModelStatusResponse | null>(
    null
  );

  const {
    execute: fetchModelStatus,
    loading,
    error,
    reset,
  } = useApiCall<ModelStatusResponse>();

  const loadModelStatus = useCallback(async () => {
    reset();
    await fetchModelStatus("/api/ml/status", {
      method: "GET",
      onSuccess: (response) => {
        setModelStatus(response);
      },
      onError: (errorMessage) => {
        console.error("モデル状態取得エラー:", errorMessage);
      },
    });
  }, [fetchModelStatus, reset]);

  useEffect(() => {
    loadModelStatus();
  }, [loadModelStatus]);

  const getScoreBadgeVariant = (score?: number) => {
    if (!score) return "outline";

    if (score >= 0.8) return "success";

    if (score >= 0.7) return "warning";

    return "destructive";
  };

  const getStatusBadgeVariant = () => {
    if (modelStatus?.is_training) return "default";

    if (modelStatus?.is_model_loaded) return "success";

    return "outline";
  };

  return {
    modelStatus,
    loading,
    error,
    loadModelStatus,
    getScoreBadgeVariant,
    getStatusBadgeVariant,
  };
};
