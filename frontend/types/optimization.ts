/**
 * バックテスト関連の型定義
 */

import { BacktestConfig } from "./backtest";

/**
 * GA戦略生成の設定
 */

export interface WaveletConfig {
  enabled: boolean;
  base_wavelet?: string;
  scales?: number[];
  target_columns?: string[];
  [key: string]: unknown;
}

export interface EarlyTerminationSettingsConfig {
  enabled?: boolean;
  max_drawdown?: number | null;
  min_trades?: number | null;
  min_trade_check_progress?: number;
  trade_pace_tolerance?: number;
  min_expectancy?: number | null;
  expectancy_min_trades?: number;
  expectancy_progress?: number;
}

export interface GAEvaluationConfig {
  enable_parallel?: boolean;
  max_workers?: number | null;
  timeout?: number;
  enable_multi_fidelity_evaluation?: boolean;
  multi_fidelity_window_ratio?: number;
  multi_fidelity_oos_ratio?: number;
  multi_fidelity_candidate_ratio?: number;
  multi_fidelity_min_candidates?: number;
  early_termination_settings?: EarlyTerminationSettingsConfig;
  oos_split_ratio?: number;
  oos_fitness_weight?: number;
  enable_walk_forward?: boolean;
  wfa_n_folds?: number;
  wfa_train_ratio?: number;
  wfa_anchored?: boolean;
}

export interface FitnessSharingConfig {
  enable_fitness_sharing?: boolean;
  sharing_radius?: number;
  sharing_alpha?: number;
  sampling_threshold?: number;
  sampling_ratio?: number;
}

export interface GAHybridConfig {
  mode?: boolean;
  model_type?: string;
  model_types?: string[];
  volatility_gate_enabled?: boolean;
  volatility_model_path?: string;
  ml_filter_enabled?: boolean;
  ml_model_path?: string;
  preprocess_features?: boolean;
}

export interface GAConfig {
  /** 実験名（UI表示/識別用） */
  experiment_name: string;
  /** バックテスト基礎設定（全個体に適用） */
  base_config: BacktestConfig;
  /** GAの動作パラメータ */
  ga_config: {
    /** 個体数（母集団サイズ） */
    population_size: number;
    /** 世代数（進化反復回数） */
    generations: number;
    /** 突然変異率（0-1） */
    mutation_rate: number;
    /** 交叉率（0-1） */
    crossover_rate: number;
    /** エリート数（世代間で生存させる上位個体数） */
    elite_size: number;
    /** 使用する指標の最大数（探索空間制御） */
    max_indicators: number;
    /** フィットネスの重み（加重合成の係数） */
    fitness_weights: {
      total_return: number;
      sharpe_ratio: number;
      max_drawdown: number;
      win_rate: number;
    };
    /** 実行時の制約（スクリーニング条件） */
    fitness_constraints: {
      /** 最低必要取引数（サンプル不足の対策） */
      min_trades: number;
      /** 最大DDの許容上限（0-1） */
      max_drawdown_limit: number;
      /** 最低シャープレシオ */
      min_sharpe_ratio: number;
    };

    /** 最低使用指標数 */
    min_indicators?: number;
    /** 最低条件数 */
    min_conditions?: number;
    /** 最大条件数 */
    max_conditions?: number;

    // ペナルティ設定
    /** ゼロトレードペナルティ */
    zero_trades_penalty?: number;
    /** 制約違反ペナルティ */
    constraint_violation_penalty?: number;

    /** フィットネスシェアリング設定 */
    fitness_sharing?: FitnessSharingConfig;

    // 目的関数設定
    /** 目的関数名の配列（fitness_values の並びと一致） */
    objectives?: string[];
    /** 目的関数ごとの重み（objectives と同じ順序） */
    objective_weights?: number[];
    /** 動的重み付け（レジーム適応） */
    dynamic_objective_reweighting?: boolean;

    // TPSL設定
    /** TPSL手法の制約リスト */
    tpsl_method_constraints?: string[];
    /** SL範囲 [min, max] */
    tpsl_sl_range?: number[];
    /** TP範囲 [min, max] */
    tpsl_tp_range?: number[];
    /** リスクリワード比範囲 [min, max] */
    tpsl_rr_range?: number[];
    /** ATR倍率範囲 [min, max] */
    tpsl_atr_multiplier_range?: number[];

    /** 評価設定 */
    evaluation_config?: GAEvaluationConfig;

    // マルチタイムフレーム
    /** MTFを有効化するか */
    enable_multi_timeframe?: boolean;
    /** 利用可能なタイムフレーム */
    available_timeframes?: string[];
    /** MTF指標の生成確率 */
    mtf_indicator_probability?: number;

    // 遺伝子生成重み
    /** 価格データの重み */
    price_data_weight?: number;
    /** 出来高データの重み */
    volume_data_weight?: number;
    /** OI/FRデータの重み */
    oi_fr_data_weight?: number;

    // 高度な遺伝的演算子設定
    crossover_field_selection_probability?: number;
    indicator_param_mutation_range?: number[];
    risk_param_mutation_range?: number[];
    indicator_add_delete_probability?: number;
    indicator_add_vs_delete_probability?: number;
    condition_change_probability_multiplier?: number;
    condition_selection_probability?: number;
    condition_operator_switch_probability?: number;
    tpsl_gene_creation_probability_multiplier?: number;
    position_sizing_gene_creation_probability_multiplier?: number;
    numeric_threshold_probability?: number;

    // パラメータ範囲プリセット
    /** パラメータ範囲プリセット名 */
    parameter_range_preset?: string;

    // ハイブリッドGA+ML設定
    /** ハイブリッド設定 */
    hybrid_config?: GAHybridConfig;
  };
}

// TP/SL戦略の種類
export type TPSLStrategy =
  | "fixed_percentage"
  | "risk_reward_ratio"
  | "volatility_based"
  | "statistical"
  | "adaptive";

// ボラティリティ感度
export type VolatilitySensitivity = "low" | "medium" | "high";

// TP/SL自動決定結果
export interface TPSLResult {
  /** 損切り幅（割合。例: 0.02 = 2%） */
  stop_loss_pct: number;
  /** 利確幅（割合。例: 0.04 = 4%） */
  take_profit_pct: number;
  /** リスクリワード比（TP/SL 比） */
  risk_reward_ratio: number;
  /** 使用された推定戦略名 */
  strategy_used: string;
  /** 推定の信頼度（0-1） */
  confidence_score: number;
  /** 参考情報（根拠となる統計など） */
  metadata?: Record<string, any>;
}

// TP/SL設定プリセット
export interface TPSLPreset {
  /** プリセット名 */
  name: string;
  /** 説明文 */
  description: string;
  /** 戦略種別 */
  strategy: TPSLStrategy;
  /** 1トレード当たりの最大リスク（資金比率） */
  max_risk_per_trade: number;
  /** 好ましいリスクリワード比 */
  preferred_risk_reward_ratio: number;
  /** ボラティリティ感度 */
  volatility_sensitivity: VolatilitySensitivity;
}

// 多目的最適化結果
export interface MultiObjectiveGAResult {
  /** API 成否 */
  success: boolean;
  /** 結果本体 */
  result: {
    /** 最良戦略（総合評価の最上位） */
    best_strategy: {
      /** 遺伝子表現（指標/ルール/リスク管理の内部表現） */
      gene_data: Record<string, any>;
      /** 総合フィットネススコア（単目的 or 重み付け合成） */
      fitness_score: number;
      /** 各目的のスコア（objectives と同順） */
      fitness_values: number[];
    };
    /** パレート前線（非劣解の集合） */
    pareto_front: Array<{
      strategy: Record<string, any>;
      fitness_values: number[];
    }>;
    /** 目的関数名の配列（fitness_values と対応） */
    objectives: string[];
    /** 評価した総戦略数 */
    total_strategies: number;
    /** 実行時間（秒） */
    execution_time: number;
  };
  /** メッセージ（補足/警告） */
  message: string;
}
/**
 * マルチ目的最適化の設定
 */
export interface MultiObjectiveConfig {
  /** バックテスト基礎設定 */
  base_config: BacktestConfig;
  /** 最適化パラメータ */
  optimization_params: {
    /** 目的関数名の配列（例: ["total_return","max_drawdown"]） */
    objectives: string[];
    /** 目的関数の重み（objectives と同順） */
    weights: number[];
    /** 最適化方式（grid: 全探索 / sambo: サンプリングベース） */
    method: "grid" | "sambo";
    /** 最大試行回数（任意） */
    max_tries?: number;
    /** チューニング対象パラメータの探索範囲（[min, max, step]） */
    parameters: {
      [key: string]: [number, number, number];
    };
  };
}

// パレート最適解
export interface ParetoSolution {
  /** 戦略の内部表現（エンコード済み設定） */
  strategy: Record<string, any>;
  /** 各目的のスコア（objectives と同順） */
  fitness_values: number[];
  /** 目的関数名配列（スコア配列と対応） */
  objectives: string[];
}

// 多目的最適化の目的関数定義
export interface ObjectiveDefinition {
  /** 論理名（内部キー） */
  name: string;
  /** 表示名（UI用） */
  display_name: string;
  /** 説明文 */
  description: string;
  /** 方向性（最大化/最小化） */
  type: "maximize" | "minimize";
  /** 重み（加重合成に用いる係数） */
  weight: number;
}

// 簡素化されたGA設定（新しいUI用）
export interface SimplifiedGAConfig {
  /** 実験名 */
  experiment_name: string;
  /** バックテスト基礎設定（最小限構成） */
  base_config: {
    /** 戦略名 */
    strategy_name: string;
    /** シンボル */
    symbol: string;
    /** 時間軸 */
    timeframe: string;
    /** 期間開始 */
    start_date: string;
    /** 期間終了 */
    end_date: string;
    /** 初期資金 */
    initial_capital: number;
    /** 片道手数料率（0-1） */
    commission_rate: number;
    /** 戦略設定（型+パラメータ） */
    strategy_config: {
      /** 戦略タイプ */
      strategy_type: string;
      /** パラメータ辞書 */
      parameters: Record<string, any>;
    };
  };
}
