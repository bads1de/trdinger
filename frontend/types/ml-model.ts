/**
 * MLモデル関連の共通型定義
 *
 * 複数フックで重複していた型定義を集約し、DRY原則に従います。
 */

/**
 * モデル情報
 *
 * 機械学習モデルの詳細情報と性能指標を保持します。
 */
export interface ModelInfo {
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
 * パフォーマンス指標
 *
 * 機械学習モデルの性能評価指標を保持します。
 */
export interface PerformanceMetrics {
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
  /** 陰性的中率 */
  npv?: number;
  /** 陽性的中率 */
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
 * 予測結果
 *
 * モデルの最新の予測確率を保持します。
 */
export interface LastPredictions {
  /** 上昇予測確率 */
  up: number;
  /** 下降予測確率 */
  down: number;
  /** レンジ予測確率 */
  range: number;
}

/**
 * モデル状態レスポンス
 *
 * モデルの現在の状態と基本情報を保持します。
 */
export interface ModelStatusResponse {
  /** モデルがロードされているかどうか */
  is_model_loaded: boolean;
  /** モデルがトレーニング済みかどうか */
  is_trained: boolean;
  /** 最後の予測結果 */
  last_predictions?: LastPredictions;
  /** 特徴量数 */
  feature_count: number;
  /** モデル詳細情報 */
  model_info?: ModelInfo;
  /** パフォーマンス指標 */
  performance_metrics?: PerformanceMetrics;
  /** トレーニング中かどうか */
  is_training?: boolean;
  /** トレーニング進捗（0-100） */
  training_progress?: number;
  /** 状態 */
  status?: string;
  /** モデルファイルパス */
  model_path?: string;
  /** 最終予測時刻 */
  last_prediction_time?: string;
  /** 予測実行回数 */
  prediction_count?: number;
}

/**
 * 特徴量重要度
 */
export interface FeatureImportance {
  [key: string]: number;
}

/**
 * 特徴量重要度レスポンス
 */
export interface FeatureImportanceResponse {
  feature_importance: FeatureImportance;
}
