/**
 * バックテスト関連の型定義
 */

import { BacktestConfig } from "./backtest";

/**
 * GA戦略生成の設定
 */

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
  /** 評価モード: "single" | "oos" | "walk_forward" | "purged_kfold" | "auto"。未指定("auto")はフラグから自動判定 */
  evaluation_mode?: string;
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

export interface GAValidationConfig {
  /** 自動検証パイプラインを有効化するか */
  enabled?: boolean;
  /** WFA のフォールド合格率の下限（0.0-1.0） */
  min_pass_rate?: number;
  /** 集約プライマリフィットネスの下限（null の場合はチェックしない） */
  min_primary_fitness?: number | null;
  /** 全フォールドの最小取引回数（null の場合はチェックしない） */
  min_trades?: number | null;
  /** 全フォールドの最大ドローダウン上限（null の場合はチェックしない） */
  max_drawdown?: number | null;
  /** PBO ゲートを有効化するか（負けフォールド比率の判定） */
  enable_pbo_gate?: boolean;
  /** 合格できる負けフォールドの最大比率（0.0-1.0） */
  pbo_threshold?: number;
  /** Deflated Sharpe Ratio ゲートを有効化するか（多重検定補正） */
  enable_dsr_gate?: boolean;
  /** DSR の合格下限（0.0-1.0） */
  min_dsr?: number;
  /** DSR の有効試行数（null の場合は population_size * generations） */
  dsr_effective_trials?: number | null;
  /** DSR の帰無分布シャープレシオ標準偏差 */
  dsr_sigma_sharpe?: number;
  /** 検証用 WFA フォールド数 */
  wfa_n_folds?: number;
  /** 検証用 WFA トレーニング比率 */
  wfa_train_ratio?: number;
  /** 検証用 WFA anchored モード */
  wfa_anchored?: boolean;
  /** 候補戦略も検証するか */
  validate_candidates?: boolean;
  /** 候補検証の対象数 */
  max_candidates?: number;
}

export interface GAIterativeImprovementConfig {
  /** 反復改善ループを有効化するか */
  enabled?: boolean;
  /** 注入する過去戦略の最大数 */
  max_seed_strategies?: number;
  /** シードとして再利用する戦略の最低フィットネス（null の場合はチェックしない） */
  min_fitness?: number | null;
  /** 自動検証に合格した戦略のみをシードにするか */
  validation_passed_only?: boolean;
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



    /** 評価設定 */
    evaluation_config?: GAEvaluationConfig;

    /** 自動検証パイプライン設定 */
    validation_config?: GAValidationConfig;

    /** 反復改善ループ設定（合格した過去戦略をシードとして再利用） */
    iterative_improvement_config?: GAIterativeImprovementConfig;

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
  };
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
