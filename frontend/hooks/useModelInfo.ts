import { useState, useEffect, useCallback } from "react";
import { useApiCall } from "@/hooks/useApiCall";

interface ModelInfo {
  model_name?: string;
  model_type?: string;
  accuracy?: number;
  last_updated?: string;
  training_samples?: number;
  file_size_mb?: number;
  parameters?: Record<string, any>;
  training_config?: Record<string, any>;
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
  model_info?: ModelInfo;
  model_path?: string;
  last_prediction_time?: string;
  prediction_count?: number;
}

export const useModelInfo = (autoRefreshInterval?: number) => {
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

  useEffect(() => {
    if (autoRefreshInterval && autoRefreshInterval > 0) {
      const interval = setInterval(loadModelStatus, autoRefreshInterval * 1000);
      return () => clearInterval(interval);
    }
  }, [autoRefreshInterval, loadModelStatus]);

  const formatDateTime = (dateString?: string) => {
    if (!dateString) return "不明";
    try {
      return new Date(dateString).toLocaleString("ja-JP", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return "不明";
    }
  };

  const formatFileSize = (sizeInMB?: number) => {
    if (!sizeInMB) return "不明";
    if (sizeInMB < 1) {
      return `${(sizeInMB * 1024).toFixed(1)} KB`;
    }
    return `${sizeInMB.toFixed(1)} MB`;
  };

  const getModelTypeBadgeVariant = (modelType?: string) => {
    switch (modelType?.toLowerCase()) {
      case "lightgbm":
        return "default";
      case "randomforest":
        return "secondary";
      case "xgboost":
        return "outline";
      default:
        return "outline";
    }
  };

  const getAccuracyBadgeVariant = (accuracy?: number) => {
    if (!accuracy) return "outline";
    if (accuracy >= 0.8) return "success";
    if (accuracy >= 0.7) return "warning";
    return "destructive";
  };

  return {
    modelStatus,
    loading,
    error,
    loadModelStatus,
    formatDateTime,
    formatFileSize,
    getModelTypeBadgeVariant,
    getAccuracyBadgeVariant,
  };
};
