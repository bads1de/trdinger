import { useEffect, useCallback } from "react";
import { useDataFetching } from "./useDataFetching";

/**
 * パフォーマンス指標インターフェース
 *
 * 機械学習モデルの性能評価指標を保持します。
 */
interface PerformanceMetrics {
  // 基本指標
  /** 精度 */
  accuracy?: number;
  /** 適合率 */
  precision?: number;
  /** 再現率 */
  recall?: number;
  /** F1スコア */
  f1_score?: number;

  // AUC指標
  /** AUCスコア */
  auc_score?: number;
  /** ROC-AUCスコア */
  auc_roc?: number;
  /** PR-AUCスコア */
  auc_pr?: number;

  // 高度な指標
  /** バランス精度 */
  balanced_accuracy?: number;
  /** マシューズ相関係数 */
  matthews_corrcoef?: number;
  /** コーエンのカッパ係数 */
  cohen_kappa?: number;

  // 専門指標
  /** 特異度 */
  specificity?: number;
  /** 感度 */
  sensitivity?: number;
  /** 陰性的中率 (Negative Predictive Value) */
  npv?: number;
  /** 陽性的中率 (Positive Predictive Value) */
  ppv?: number;

  // 確率指標
  /** 対数損失 */
  log_loss?: number;
  /** ブライアースコア */
  brier_score?: number;

  // その他
  /** 損失 */
  loss?: number;
  /** 検証精度 */
  val_accuracy?: number;
  /** 検証損失 */
  val_loss?: number;
  /** トレーニング時間 */
  training_time?: number;
  /** 最終評価日時 */
  last_evaluation?: string;
}

/**
 * モデル状態レスポンスインターフェース
 *
 * モデルの状態とパフォーマンス情報を保持します。
 */
interface ModelStatusResponse {
  /** モデルがロードされているかどうか */
  is_model_loaded: boolean;
  /** モデルがトレーニング済みかどうか */
  is_trained: boolean;
  /** 最後の予測結果 */
  last_predictions?: {
    /** 上昇予測確率 */
    up: number;
    /** 下降予測確率 */
    down: number;
    /** レンジ予測確率 */
    range: number;
  };
  /** 特徴量数 */
  feature_count: number;
  /** モデル詳細情報 */
  model_info?: {
    // 基本性能指標
    /** 精度 */
    accuracy?: number;
    /** 適合率 */
    precision?: number;
    /** 再現率 */
    recall?: number;
    /** F1スコア */
    f1_score?: number;

    // AUC指標
    /** AUCスコア */
    auc_score?: number;
    /** ROC-AUCスコア */
    auc_roc?: number;
    /** PR-AUCスコア */
    auc_pr?: number;

    // 高度な指標
    /** バランス精度 */
    balanced_accuracy?: number;
    /** マシューズ相関係数 */
    matthews_corrcoef?: number;
    /** コーエンのカッパ係数 */
    cohen_kappa?: number;

    // 専門指標
    /** 特異度 */
    specificity?: number;
    /** 感度 */
    sensitivity?: number;
    /** 陰性的中率 */
    npv?: number;
    /** 陽性的中率 */
    ppv?: number;

    // 確率指標
    /** 対数損失 */
    log_loss?: number;
    /** ブライアースコア */
    brier_score?: number;

    // モデル情報
    /** モデルタイプ */
    model_type?: string;
    /** 最終更新日時 */
    last_updated?: string;
    /** トレーニングサンプル数 */
    training_samples?: number;
    /** テストサンプル数 */
    test_samples?: number;
    /** ファイルサイズ（MB） */
    file_size_mb?: number;
    /** 特徴量数 */
    feature_count?: number;
    /** クラス数 */
    num_classes?: number;
    /** 最良イテレーション */
    best_iteration?: number;

    // 学習パラメータ
    /** トレーニング・テスト分割比率 */
    train_test_split?: number;
    /** 乱数シード */
    random_state?: number;

    // 詳細データ
    /** 特徴量重要度 */
    feature_importance?: Record<string, number>;
    /** 分類レポート */
    classification_report?: Record<string, any>;
  };
  /** パフォーマンス指標 */
  performance_metrics?: PerformanceMetrics;
  /** トレーニング中かどうか */
  is_training?: boolean;
  /** トレーニング進捗（0-100） */
  training_progress?: number;
  /** 状態 */
  status?: string;
}

/**
 * モデルパフォーマンス管理フック
 *
 * 機械学習モデルのパフォーマンス情報を取得・管理します。
 * モデルの状態、性能指標、UI表示用のユーティリティ関数を提供します。
 *
 * @example
 * ```tsx
 * const {
 *   modelStatus,
 *   loading,
 *   error,
 *   loadModelStatus,
 *   getScoreBadgeVariant,
 *   getStatusBadgeVariant
 * } = useModelPerformance();
 *
 * // モデルパフォーマンス情報を取得
 * loadModelStatus();
 *
 * // スコアに応じたバリアントを取得
 * const variant = getScoreBadgeVariant(0.85);
 *
 * // 状態に応じたバリアントを取得
 * const statusVariant = getStatusBadgeVariant();
 * ```
 *
 * @returns {{
 *   modelStatus: ModelStatusResponse | null,
 *   loading: boolean,
 *   error: string | null,
 *   loadModelStatus: () => Promise<void>,
 *   getScoreBadgeVariant: (score?: number) => string,
 *   getStatusBadgeVariant: () => string
 * }} モデルパフォーマンス管理関連の状態と操作関数
 */
export const useModelPerformance = () => {
  const {
    data: modelStatusArray,
    loading,
    error,
    refetch: loadModelStatus,
  } = useDataFetching<ModelStatusResponse>({
    endpoint: "/api/ml/status",
    transform: (response) => [response],
    errorMessage: "モデルパフォーマンスの取得に失敗しました",
  });

  const modelStatus = modelStatusArray.length > 0 ? modelStatusArray[0] : null;

  /**
   * スコアに応じたバッジバリアントを取得
   *
   * @param {number} score - スコア（0-1）
   * @returns {string} バッジバリアント（"success"、"warning"、"destructive"、"outline"）
   */
  const getScoreBadgeVariant = (score?: number) => {
    if (!score) return "outline";

    if (score >= 0.8) return "success";

    if (score >= 0.7) return "warning";

    return "destructive";
  };

  /**
   * 状態に応じたバッジバリアントを取得
   *
   * @returns {string} バッジバリアント（"default"、"success"、"outline"）
   */
  const getStatusBadgeVariant = () => {
    if (modelStatus?.is_training) return "default";

    if (modelStatus?.is_model_loaded) return "success";

    return "outline";
  };

  return {
    /** モデル状態情報 */
    modelStatus,
    /** ローディング状態 */
    loading,
    /** エラーメッセージ */
    error,
    /** モデル状態を再取得する関数 */
    loadModelStatus,
    /** スコアに応じたバッジバリアントを取得する関数 */
    getScoreBadgeVariant,
    /** 状態に応じたバッジバリアントを取得する関数 */
    getStatusBadgeVariant,
  };
};
