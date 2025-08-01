import { useEffect } from "react";
import { useDataFetching } from "./useDataFetching";

/**
 * モデル情報インターフェース
 *
 * 機械学習モデルの詳細情報と性能指標を保持します。
 */
interface ModelInfo {
  /** モデル名 */
  model_name?: string;
  /** モデルタイプ */
  model_type?: string;

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
  /** モデルパラメータ */
  parameters?: Record<string, any>;
  /** トレーニング設定 */
  training_config?: Record<string, any>;
}

/**
 * モデル状態レスポンスインターフェース
 *
 * モデルの現在の状態と基本情報を保持します。
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
  model_info?: ModelInfo;
  /** モデルファイルパス */
  model_path?: string;
  /** 最終予測時刻 */
  last_prediction_time?: string;
  /** 予測実行回数 */
  prediction_count?: number;
}

/**
 * モデル情報管理フック
 *
 * 機械学習モデルの状態情報と詳細情報を取得・管理します。
 * 自動更新機能やUI表示用のユーティリティ関数も提供します。
 *
 * @example
 * ```tsx
 * const {
 *   modelStatus,
 *   loading,
 *   error,
 *   loadModelStatus,
 *   getModelTypeBadgeVariant,
 *   getAccuracyBadgeVariant
 * } = useModelInfo(30); // 30秒ごとに自動更新
 *
 * // モデル状態を手動で更新
 * loadModelStatus();
 *
 * // モデルタイプに応じたバリアントを取得
 * const variant = getModelTypeBadgeVariant('lightgbm');
 *
 * // 精度に応じたバリアントを取得
 * const accuracyVariant = getAccuracyBadgeVariant(0.85);
 * ```
 *
 * @param {number} autoRefreshInterval - 自動更新間隔（秒）。指定しない場合は自動更新しない
 * @returns {{
 *   modelStatus: ModelStatusResponse | null,
 *   loading: boolean,
 *   error: string | null,
 *   loadModelStatus: () => Promise<void>,
 *   getModelTypeBadgeVariant: (modelType?: string) => string,
 *   getAccuracyBadgeVariant: (accuracy?: number) => string
 * }} モデル情報管理関連の状態と操作関数
 */
export const useModelInfo = (autoRefreshInterval?: number) => {
  const {
    data: modelStatusArray,
    loading,
    error,
    refetch: loadModelStatus,
  } = useDataFetching<ModelStatusResponse>({
    endpoint: "/api/ml/status",
    transform: (response) => [response],
    errorMessage: "モデル状態の取得に失敗しました",
  });

  const modelStatus = modelStatusArray.length > 0 ? modelStatusArray[0] : null;

  useEffect(() => {
    if (autoRefreshInterval && autoRefreshInterval > 0) {
      const interval = setInterval(loadModelStatus, autoRefreshInterval * 1000);
      return () => clearInterval(interval);
    }
  }, [autoRefreshInterval, loadModelStatus]);

  /**
   * モデルタイプに応じたバッジバリアントを取得
   *
   * @param {string} modelType - モデルタイプ
   * @returns {string} バッジバリアント（"default"または"outline"）
   */
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

  /**
   * 精度に応じたバッジバリアントを取得
   *
   * @param {number} accuracy - 精度（0-1）
   * @returns {string} バッジバリアント（"success"、"warning"、"destructive"、"outline"）
   */
  const getAccuracyBadgeVariant = (accuracy?: number) => {
    if (!accuracy) return "outline";

    if (accuracy >= 0.8) return "success";

    if (accuracy >= 0.7) return "warning";

    return "destructive";
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
    /** モデルタイプに応じたバッジバリアントを取得する関数 */
    getModelTypeBadgeVariant,
    /** 精度に応じたバッジバリアントを取得する関数 */
    getAccuracyBadgeVariant,
  };
};
