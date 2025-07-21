import { useState, useEffect, useCallback } from "react";
import { formatDateTime, formatFileSize } from "@/utils/formatters";
import { useApiCall } from "@/hooks/useApiCall";

interface ModelInfo {
  model_name?: string;
  model_type?: string;

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
  last_updated?: string;
  training_samples?: number;
  test_samples?: number;
  file_size_mb?: number;
  feature_count?: number;
  num_classes?: number;
  best_iteration?: number;

  // 学習パラメータ
  train_test_split?: number;
  random_state?: number;

  // 詳細データ
  feature_importance?: Record<string, number>;
  classification_report?: Record<string, any>;
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

  const getModelTypeBadgeVariant = (modelType?: string) => {
    switch (modelType?.toLowerCase()) {
      case "lightgbm":
        return "default";
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
    getModelTypeBadgeVariant,
    getAccuracyBadgeVariant,
  };
};
