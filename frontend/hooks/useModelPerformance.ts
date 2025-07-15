import { useState, useEffect, useCallback } from "react";
import { useApiCall } from "@/hooks/useApiCall";

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

  const getScoreColor = (score?: number) => {
    if (!score) return "text-gray-400";

    if (score >= 0.8) return "text-green-400";

    if (score >= 0.7) return "text-yellow-400";

    if (score >= 0.6) return "text-orange-400";

    return "text-red-400";
  };

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

  const formatTrainingTime = (seconds?: number) => {
    if (!seconds) return "不明";

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}時間${minutes}分${secs}秒`;
    }

    if (minutes > 0) {
      return `${minutes}分${secs}秒`;
    }

    return `${secs}秒`;
  };

  return {
    modelStatus,
    loading,
    error,
    loadModelStatus,
    getScoreColor,
    getScoreBadgeVariant,
    getStatusBadgeVariant,
    formatTrainingTime,
  };
};
