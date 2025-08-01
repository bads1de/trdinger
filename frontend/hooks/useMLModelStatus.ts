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
  performance_metrics?: {
    // 基本指標
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

    // その他
    loss?: number;
    val_accuracy?: number;
    val_loss?: number;
    training_time?: number;
    last_evaluation?: string;
  };
  is_training?: boolean;
  training_progress?: number;
  status?: string;
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
