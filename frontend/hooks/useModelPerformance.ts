import { useState, useEffect, useCallback } from "react";
import { useApiCall } from "@/hooks/useApiCall";
import { formatTrainingTime } from "@/utils/formatters";

interface PerformanceMetrics {
  // 基本指標
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;

  // AUC指標
  auc_score?: number; // 後方互換性のため保持
  auc_roc?: number;
  auc_pr?: number;

  // 高度な指標
  balanced_accuracy?: number;
  matthews_corrcoef?: number;
  cohen_kappa?: number;

  // 専門指標
  specificity?: number;
  sensitivity?: number;
  npv?: number; // Negative Predictive Value
  ppv?: number; // Positive Predictive Value

  // 確率指標
  log_loss?: number;
  brier_score?: number;

  // その他
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
    model_type?: string;
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
